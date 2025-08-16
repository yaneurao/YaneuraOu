#include "dlshogi_searcher.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include <sstream> // stringstream
#include <cstring> // memcpy
#include <numeric> // accumulate

#include "dlshogi_types.h"
#include "UctSearch.h"
#include "PrintInfo.h"
#include "FukauraOuEngine.h"

#include "../../search.h"
#include "../../thread.h"
#include "../../mate/mate.h"
#include "../../engine.h"

namespace dlshogi {

// --------------------------------------------------------------------
//  DlshogiSearcher : dlshogiの探索部で、globalになっていたものをクラス化したもの。
// --------------------------------------------------------------------

DlshogiSearcher::DlshogiSearcher(FukauraOuEngine& engine) :
	engine(engine)
{
	gc                   = std::make_unique<NodeGarbageCollector>();
    interruption_checker = std::make_unique<SearchInterruptionChecker>(this);
}

void DlshogiSearcher::add_options(OptionsMap& options) {
	search_options.add_options(options);
    search_limits.time_manager.add_options(options);
}


// GPUの初期化、各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
// "isready"に対して呼び出される。
// スレッドの生成ついでに、詰将棋探索系の初期化もここで行う。
void DlshogiSearcher::InitGPU(const std::string& model_path , std::vector<int> thread_settings, int policy_value_batch_maxsize)
{
	// ----------------------
	// 必要なスレッド数の算出
	// ----------------------

	// トータルのスレッド数go
    size_t total_thread_num = std::accumulate(thread_settings.begin(), thread_settings.end(), 0);

	// スレッド数の合計が0はさすがにおかしい。
	if (total_thread_num == 0)
	{
		sync_cout << "info string Error! : total threads = 0 " << sync_endl;
		return;
	}

	// ----------------------
	//    スレッドの確保
	// ----------------------

	/*
		📓 確保するスレッド数について

			以前のふかうら王では、探索の終了条件を満たしたかを監視するためのスレッドもThreadPoolから割り当てていたのだが、
			どのみち監視スレッドはCPU負荷が高くないのでこれをわざわざ探索スレッドかのように扱うのは面倒なのでやめることにした。

			また、dfpnスレッドも、あまり考えても仕方がないかと思った。(どうせ、これがあるとメモリ帯域をかなり食い潰すわけで…)

			あと、GC用のスレッドも探索スレッドから割り当てたのだが、それは良くないアイデアだった。
			(GC処理が終わらなくて、全探索スレッドの終了を待つコードになっているから、bestmoveが返せないことがあるため。)


		💡 やねうら王のThreadPoolクラスは、前回と異なるスレッド数であれば自動的に再確保される。
	*/ 

    auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken) {
            return std::make_unique<FukauraOuWorker>(

              // Worker基底classが渡して欲しいもの。
              engine.options, engine.threads, threadIdx, numaAccessToken,

              // 追加でFukauraOuEngineからもらいたいもの
              *this, engine);
        };

	// 探索の終了条件を満たしたかを監視するためのスレッド数
	const int search_interruption_check_thread_num = 1;

    engine.threads.set(engine.numaContext.get_numa_config(), engine.options,
                           total_thread_num + search_interruption_check_thread_num, worker_factory);

	// このタイミングで確保しなおす。

	if (thread_settings != last_thread_settings)
    {
        search_groups        = std::make_unique<UctSearcherGroup[]>(thread_settings.size());
        search_groups_size   = thread_settings.size();
        last_thread_settings = thread_settings;
    }

	// モデルの読み込み
    ElapsedTimer time;
        
	for (size_t i = 0; i < search_groups_size ; i++)
		if (thread_settings[i] > 0)
			search_groups[i].Initialize(model_path , thread_settings[i],/* gpu_id = */int(i), policy_value_batch_maxsize);

	sync_cout << "info string All model files have been loaded. " << time.elapsed() << "ms." << sync_endl;

	// ----------------------
	// 探索スレッドとUctSearcherの紐付け
	// ----------------------

	// 何番目のスレッドがどのUctSearcherGroupのインスタンスに割り当たるのかのmapperだけ設定しておいてやる。
	// Thread::search()からは、このmapperを使って、UctSearchGroup::search()を呼び出す。

	thread_id_to_uct_searcher.clear();

	for (size_t i = 0; i < search_groups_size ; ++i)
		for(int j = 0;j < thread_settings[i];++j)
			thread_id_to_uct_searcher.push_back(search_groups[i].get_uct_searcher(j));

	// GC用のスレッドにもスレッド番号を連番で与えておく。
	// (WinProcGroup::bindThisThread()用)
    //gc->set_thread_id(total_thread_num  /* + dfpn_thread_num*/);

	// ----------------------
	// 詰将棋探索系の初期化
	// ----------------------

	// leaf nodeでの詰み探索用のMateSolverの初期化
	for (auto& uct_searcher : thread_id_to_uct_searcher)
		uct_searcher->InitMateSearcher(search_options);

#if defined(USE_POLICY_BOOK)
	policy_book.read_book();
#endif

}

// 探索スレッドの終了(main thread以外)
void DlshogiSearcher::TeminateThreads()
{
	// 全スレッドに停止命令を送る。
	search_limits.interruption = true;
	// →　これでdlshogi関係の探索スレッドは停止するはず。
	// Threads.stopはここでは変更しないようにする。(呼び出し元でponderの時に待つのに必要だから)

	// 各スレッドが終了するのを待機する(main以外の探索スレッドが終了するのを待機する)
    engine.threads.wait_for_search_finished();
}

// 対局開始時に呼び出されるハンドラ
void DlshogiSearcher::NewGame()
{
}

// 対局終了時に呼び出されるハンドラ
void DlshogiSearcher::GameOver()
{
#if defined(ENABLE_POLICY_BOOK_LEARN)
	// 今回の棋譜をPolicyBookを書き出す必要がある。
	auto last_position_cmd = Threads.main()->last_position_cmd_string;
	auto sfen = last_position_cmd.substr(strlen("position "));
	policy_book.append_sfen_to_db_bin(sfen);
#endif

}

// 投了の閾値設定
//   resign_threshold : 1000分率で勝率を指定する。この値になると投了する。
void DlshogiSearcher::SetResignThreshold(const int resign_threshold)
{
	search_options.RESIGN_THRESHOLD = (float)resign_threshold / 1000.0f;
}

// 千日手の価値設定
//   value_black           : この値をroot colorが先手番の時の千日手の価値とみなす。(千分率)
//   value_white           : この値をroot colorが後手番の時の千日手の価値とみなす。(千分率)
//   draw_value_from_black : エンジンオプションの"Draw_Value_From_Black"の値。
void DlshogiSearcher::SetDrawValue(const int value_black, const int value_white)
{
	search_options.draw_value_black = (float)value_black / 1000.0f;
	search_options.draw_value_white = (float)value_white / 1000.0f;
}

// →　これは、エンジンオプションの"MaxMovesToDraw"を自動的にDlshogiSearcher::SetLimits()で設定するので不要。
//// 引き分けとする手数の設定
//void SetDrawPly(const int ply)
//{
//	draw_ply = ply;
//}

// PV lineの詰み探索の設定
// threads : スレッド数
// nodes   : 1局面で詰探索する最大ノード数。
void DlshogiSearcher::SetPvMateSearch(const int threads, /*const int depth,*/ const int nodes)
{
	// 現在生成されているthread数とnodesがぴったり一致するなら、生成しなおす必要はない。
	if (threads == int(pv_mate_searchers.size()) &&
		(threads == 0 || pv_mate_searchers[0].get_nodes_limit() == nodes))
		return; 

	// 個数が異なるので生成しなおす。

	pv_mate_searchers.clear(); // いったんすべて開放
	pv_mate_searchers.reserve(threads);

	for (int i = 0; i < threads; i++)
		pv_mate_searchers.emplace_back(nodes, this);
}

//  UCT探索の初期設定
void DlshogiSearcher::InitializeUctSearch() {

    if (!tree)
        tree = std::make_unique<NodeTree>(gc.get());
    //search_groups = std::make_unique<UctSearcherGroup[]>(max_gpu);
    // →　これもっと早い段階で行わないと間に合わない。コンストラクタに移動させる。

    // dlshogiにはないが、dlshogiでglobalだった変数にアクセスするために、
    // UctSearcherGroupは、DlshogiSearcher*を持たなければならない。
    for (size_t i = 0; i < search_groups_size; ++i)
        search_groups[i].set_dlsearcher(this);
}

//  UCT探索の終了処理
void DlshogiSearcher::TerminateUctSearch() {
    for (auto& searcher : pv_mate_searchers)
        searcher.Term();

    // やねうら王では、詰み以外の探索スレッドはThreadPoolクラスで生成しているのでここは関係ない。

    //#ifdef THREAD_POOL
    //	if (search_groups) {
    //		for (int i = 0; i < max_gpu; i++)
    //			search_groups[i].Term();
    //	}
    //#endif
}

// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
//    limits.nodes        : 探索を打ち切るnode数   　→  search_limits.nodes_limitに反映する。
//    limits.movetime     : 思考時間固定時の指定     →　search_limits.movetimeに反映する。
// などなど。
// その他、"go"コマンドで渡された残り時間等から、今回の思考時間を算出し、search_limits.time_managerに反映する。
void DlshogiSearcher::SetLimits(const Position& pos, const Search::LimitsType& limits) {
    auto& s = search_limits;
    auto& o = search_options;

    // go infiniteされているのか
    s.infinite = limits.infinite;

    // 探索を打ち切るnode数
    s.nodes_limit = (NodeCountType) limits.nodes;

    // 思考時間固定の時の指定
    s.movetime = limits.movetime;

    // dlshogiのコード
#if 0
	// ノード数固定ならばそれを設定。
	// このときタイムマネージメント不要
	if (limits.nodes) {
		search_limits.node_limit = (NodeCountType)(u64)limits.nodes;
		return;
	}

	const Color color = pos->side_to_move();
	const int divisor = 14 + std::max(0, 30 - pos->game_ply());

	// 自分側の残り時間を divisorで割り算して、それを今回の目安時間とする。
	search_limits.remaining_time[BLACK] = limits.time[BLACK];
	search_limits.remaining_time[WHITE] = limits.time[WHITE];

	search_limits.time_limit = search_limits.remaining_time[color] / divisor + limits.inc[color];

	// 最小思考時間の設定。(これ以上考える)
	search_limits.minimum_time = limits.movetime;

	// 無制限に思考するなら、NodeCountTypeの最大値を設定しておく。
	search_limits.node_limit = limits.infinite ? std::numeric_limits<NodeCountType>::max() : (NodeCountType)(u64)limits.nodes;

	// 思考時間が固定ではなく、ノード数固定でもないなら、探索の状況によっては時間延長してもよい。
	search_limits.extend_time = limits.movetime == 0 && limits.nodes == 0;
#endif

    // 持ち時間制御は、やねうら王の time managerをそのまま用いる。
    search_limits.time_manager.init(limits, pos.side_to_move(), pos.game_ply(), engine.options,
                                    o.max_moves_to_draw);
}

// ノード数固定にしたい時は、USIの"go nodes XXX"ででき、これは、SetLimits()で反映するので↓は不要。

//// 1手のプレイアウト回数を固定したモード
//// 0の場合は無効
//void SetConstPlayout(const int playout)
//{
//	const_playout = playout;
//}

// 終了させるために、search_groupsを開放する。
void DlshogiSearcher::FinalizeUctSearch() {
    TerminateUctSearch();

    search_groups.reset();
    tree.reset();  // treeの開放を行う時にGCが必要なのでCGをあとから開放
    gc.reset();
}


// UCTアルゴリズムによる着手生成
// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
// ※　事前にSetLimits()で探索条件を設定しておくこと。
//   pos            : 探索開始局面
//   game_root_sfen : ゲーム開始局面のsfen文字列
//   moves          : ゲーム開始局面からの手順
//   ponderMove     : [Out] ponderの指し手(ないときはMove::none()が代入される)
//   返し値 : この局面でのbestな指し手
// ponderの場合は、呼び出し元で待機すること。
Move DlshogiSearcher::UctSearchGenmove(Position&                pos,
                                       const std::string&       game_root_sfen,
                                       const std::vector<Move>& moves,
                                       Move&                    ponderMove) {

	// 詰み探索スレッドの停止フラグの初期化
    for (auto& searcher : pv_mate_searchers)
        searcher.Stop(false);

    // これ[Out]なのでとりあえず初期化しておかないと忘れてしまう。
    ponderMove = Move::none();

    // 探索停止フラグをreset。
    // →　やねうら王では使わない。Threads.stopかsearch_limits.interruptionを使う。
    //search_limits.uct_search_stop = false;

    // begin_timeを初期化したかのフラグ
    // →　やねうら王ではTimerクラスを使ってgoコマンドからの経過時間を計測しているので不要
    //init_search_begin_time = false;

    // 中断フラグのリセット
    search_limits.interruption = false;

    // ゲーム木を現在の局面にリセット
    tree->ResetToPosition(game_root_sfen, moves);

    // ルート局面をコピーしておく。(詰み探索の開始局面などで使いたいため)
    std::memcpy(&pos_root, &pos, sizeof(Position));

    // 探索開始局面
    Node* current_root         = tree->GetCurrentHead();
    search_limits.current_root = tree->GetCurrentHead();

    // "go ponder"で呼び出されているかのフラグの設定
    //search_limits.pondering = ponder;

    // 探索ノード数のクリア
    search_limits.nodes_searched = 0;

    // 対局開始からの手数を設定しておく。(持ち時間制御などで使いたいため)
    search_limits.game_ply = pos.game_ply();

    // UCTの初期化。
    // 探索開始局面の初期化
    ExpandRoot(&pos, search_options.generate_all_legal_moves);

    // 前回、この現在の探索局面を何回訪問したのか
    const NodeCountType pre_simulated =
      current_root->move_count != NOT_EXPANDED ? current_root->move_count.load() : 0;

	// 探索をスキップしたかのフラグ
	bool search_skipped = true;
    Book::ProbeResult probeResult;

    // ---------------------
    //     詰まされチェック
    // ---------------------

    // ExpandRoot()が呼び出されている以上、子ノードの初期化は完了していて、この局面の合法手の数がchild_numに
    // 代入されているはず。これが0であれば、合法手がないということだから、詰みの局面であり、探索ができることは何もない。
    const ChildNumType child_num = current_root->child_num;
    if (child_num == 0)
    {
        probeResult.bestmove = Move::resign();
        probeResult.bestscore = mated_in(1);
		goto SEARCH_SKIP;
    }

    // ---------------------
    //     宣言勝ちのチェック
    // ---------------------

    {
        // 宣言勝ちか？
        Move move = pos.DeclarationWin();

        if (move)
        {
            // 宣言勝ち
            // 💡 sync_cout << "info score mate 1 pv MOVE_WIN" << sync_endl;

			probeResult.bestmove = move;
            probeResult.bestscore = mate_in(1);
            goto SEARCH_SKIP;
        }

        // 詰みはdf-pnで詰め探索をしているので何もせずとも普通に見つけるはず…。
        // ここでN手詰めを呼び出すと、解けた時に読み筋が出ないのであまり嬉しくない。
    }

    // ---------------------
    //     定跡の選択部
    // ---------------------

    // 定跡DBにhitするか調べる。

    probeResult = book.probe(pos, engine.updateContext);
    if (probeResult.bestmove)
        goto SEARCH_SKIP;

    // ---------------------
    //     並列探索の開始
    // ---------------------

	search_skipped        = false;

    // 探索時間とプレイアウト回数の予測値を出力
    if (search_options.debug_message)
        UctPrint::PrintPlayoutLimits(search_limits.time_manager, search_limits.nodes_limit);

    // --- 詰みルーチン用の初期化

    rootMateMove = Move::none();

    // 詰みフラグのリセット。これ、リセットしておかないと同じ局面で二度goされたときに、二度目の時に
    // すでに詰み探索が終わっている扱いになり、rootMateMoveがセットされない。
    current_root->dfpn_checked = false;

    // PVの詰み探索スレッド開始
    for (auto& searcher : pv_mate_searchers)
        searcher.Run();

    // 探索スレッドの開始
    //StartThreads();

	// main以外のthreadを開始する
	engine.threads.start_searching();
	// 💡 FukauraOuWorker::start_searching()が呼び出され、FukauraOuWorker::parallel_search()から、
	//     このclassのparallel_search()がよびだされる 。

	// main thread(このスレッド)も探索に参加する。
	// 💡 main threadは thread id == 0と決まっている。
    parallel_search(pos, 0);

	/*
		📓 これで探索が始まって、このあとmainスレッドが帰還する。
			そのあと全探索スレッドの終了を待ってからPV,bestmoveを返す。
			(そうしないとvirtual lossがある状態でbest nodeを拾おうとしてしまう)
	*/ 

    // PVの詰み探索スレッド停止(まず停止命令だけ送っておく)
    for (auto& searcher : pv_mate_searchers)
        searcher.Stop();

    // 探索スレッドの終了(とすべてのスレッドの終了の待機)
    TeminateThreads();

    // PVの詰み探索スレッド終了待機
    for (auto& searcher : pv_mate_searchers)
        searcher.Join();

SEARCH_SKIP:
    // ---------------------
    //     PVの出力
    // ---------------------

    // デバッグ用のメッセージ出力
    if (search_options.debug_message)
    {
        // 探索にかかった時間を求める
        const TimePoint finish_time = search_limits.time_manager.elapsed_time();
        //search_limits.remaining_time[pos->side_to_move()] -= finish_time;

        // 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
        UctPrint::PrintPlayoutInformation(current_root, &search_limits, finish_time, pre_simulated);
    }

	if (search_skipped)
	{
        // search_skippedのときは、自前でPVを構築する。
        // 💡 このとき、rootMovesの情報を使わないようにしたい。

        Search::InfoFull info;
        info.depth     = 0;
        info.selDepth  = 0;
        info.multiPV   = 1;
        info.score     = probeResult.bestscore;
        TimePoint time = std::max(TimePoint(1), search_limits.time_manager.elapsed_time());
        info.timeMs    = time;
        info.nodes     = 0;
        info.nps       = 0;
        std::string pv = probeResult.bestmove.to_usi_string();
        if (probeResult.pondermove)
            pv += " " + probeResult.pondermove.to_usi_string();
        info.pv       = pv;
        info.hashfull = 0; // 不明
        engine.updateContext.onUpdateFull(info);

		ponderMove = probeResult.pondermove;
        return probeResult.bestmove;
	}

    // PVの取得と表示
    auto best = UctPrint::get_best_move_multipv(current_root, search_limits, search_options, engine.updateContext);

    // ---------------------
    //   思考した指し手を返す
    // ---------------------

    // ponderモードでは指し手自体は返さない。
    // →　やねうら王では、stopが来るまで待機して返す。
    //  dlshogiの実装はいったんUCT探索を終了させるようになっているのでコメントアウト。
    //if (pondering)
    //	return Move::none();

    // あとで
    // 探索の延長判定
    // →　これは探索の停止判定で行うから削除

    // この時点で探索スレッドをすべて停止させないと
    // Virtual Lossを元に戻す前にbestmoveを選出してしまう。

    // df-pnルーチンが詰みを見つけている。
    // (この時のponderはセットなしでいいと思う。どうせ残りもdf-pnが見つけるので…)
    if (rootMateMove != Move::none())
        return rootMateMove;

    // 評価値が投了値を下回っていたら投了
    if (best.wp < search_options.RESIGN_THRESHOLD)
    {
        ponderMove = Move::none();
        return Move::resign();
    }

    // それに対するponderの指し手もあるはずなのでそれをセットしておく。
    ponderMove = best.ponder;

    return best.move;
}

// Root Node(探索開始局面)を展開する。
// generate_all : 歩の不成なども生成する。
void DlshogiSearcher::ExpandRoot(const Position* pos, bool generate_all) {
    Node* current_head = tree->GetCurrentHead();
    if (current_head->child_num == 0)
    {
        current_head->ExpandNode(pos, generate_all);
    }
}

// PV表示の確認
// SearchInterruptionCheckerから呼び出される。
void DlshogiSearcher::OutputPvCheck() {
    auto& s = search_limits;
    auto& o = search_options;

    // "PvInterval"が0のときはPVの出力は行わない。
    if (!o.pv_interval)
        return;

    // 前回からの経過時刻。
    const auto elapsed_time = s.time_manager.elapsed_time();
    if (elapsed_time > s.last_pv_print + o.pv_interval)
    {

        // PV表示
        //get_and_print_pv();
        UctPrint::get_best_move_multipv(tree->GetCurrentHead(), search_limits, search_options, engine.updateContext);

        // 出力が終わった時点から数えて pv_interval後以降に再度表示。
        // (前回の出力時刻から数えてしまうと、PVの出力がたくさんあるとき出力が間に合わなくなる)
        s.last_pv_print = elapsed_time;
    }
}

// posからply先のpvのhash keyを返す。
// ply = 残り手数(3を設定すると現局面と合わせて4手分をkeysに出力)
// keys には keys[0]がply手後、keys[1]がply-1手後 .. keys[ply-1]に0手後のhash keyが返る。
void pv_key(Position& pos, Node* node, int ply, Key64 keys[]) {
    if (ply == 0)
        keys[ply] = pos.key();
    else if (node == nullptr || node->child_num == 0)
        keys[ply] = 0;
    else
    {
        if (node->child_nodes.get() == nullptr)
            return;
        // child nodesが展開されていない。

        ChildNumType max_i = 0;
        for (ChildNumType i = 1; i < node->child_num; ++i)
            if (node->child[i].move_count > node->child[max_i].move_count)
                max_i = i;

        StateInfo si;
        Move      m = node->child[max_i].move;
        //sync_cout << to_usi_string(m) << sync_endl;
        pos.do_move(m, si);
        pv_key(pos, node->child_nodes[max_i].get(), ply - 1, keys);
        pos.undo_move(m);
    }
}

//  探索停止の確認
// SearchInterruptionCheckerから呼び出される。
void DlshogiSearcher::InterruptionCheck(const Position& rootPos) {
    auto& s = search_limits;
    auto& o = search_options;

    // すでに中断することが確定しているのでここでは何もしない。
    if (s.interruption || engine.threads.stop)
        return;

    // 思考時間は、ponderhitからの時間で考える。
    const auto elapsed_from_ponderhit = s.time_manager.elapsed_time_from_ponderhit();

    //if (s.time_manager.search_end)
    //	sync_cout << "wait = " << elapsed_from_ponderhit << sync_endl;

    // 探索を停止させる関数
    // (デバッグ表示ありなら、この止めたタイミングを出力する。
    // これにより、探索を停止させてから何ms 停止するまでに要するかがわかる。
    auto interrupt = [&]() {
        if (o.debug_message)
            sync_cout << "info string search search_end = " << s.time_manager.search_end
                      << "[ms], interruption time = " << elapsed_from_ponderhit << "[ms]"
                      << sync_endl;

        s.interruption = true;
    };

    // 探索depth固定
    // →　PV掘らないとわからないので実装しない。

    // 探索ノード固定(NodesLimitで設定する)
    //   ※　この時、時間制御は行わない
    if (s.nodes_limit)
    {
        if (s.nodes_searched >= s.nodes_limit)
            interrupt();

        // 時間制御不要なのでノード数を超えていなくともここでリターンする。
        return;
    }

    // hashfull
    // s.current_root->move_count == NOT_EXPANDED  開始まもなくはこれでありうるので、
    // +1してから比較する。(NOT_EXPANDEDはu32::max()なので+1すると0になる)
    //
    // current_root->move_countは現在のルートからの展開局面数だと近似できる(少なくとも、これ以上の
    //   局面の情報はメモリ上に存在しないことは保証されている)ので、uct_node_limitがこれを上回るまで
    //   という条件にしておく。
    // 本当は、expand_node()した回数で制限をかけたいのだが、current_rootにぶらさがっているノード数をカウントする
    // 手段がないため、このようにしておく。
    //
    // stochastic ponderでponderhitした場合もいったん思考を停止して、次の局面で(つまりcurrent_rootが次の局面に進んで)、
    // 再度思考するので、この処理で問題ない。
    if ((NodeCountType) (s.current_root->move_count + 1) > o.uct_node_limit)
    {
        // これは、時間制御の対象外。
        // ただちに中断すべき。(メモリ足りなくなって死ぬので)

        interrupt();
        return;
    }

    // リミットなしなので"stop"が来るまで停止しない。
    // ただしhashfullの判定は先にやっておかないと、メモリ使い切ってしまう。
    if (s.infinite)
        return;

    // "go ponder"で呼び出されて、"ponderhit"が来ていないなら持ち時間制御の対象外。
    if (search_limits.ponder)
        return;

    // -- 時間制御

    // df-pnが詰みの指し手を見つけている。
    if (rootMateMove != Move::none())
    {
        interrupt();
        return;
    }

    // 探索時間固定
    // "go movetime XXX"のように指定するのでGUI側が対応していないかもしれないが。
    // 探索時間固定なのにponderしていることは普通ないと思うので、ponder後の時間経過を見て停止させて良いと思う。
    if (s.movetime && elapsed_from_ponderhit > s.movetime)
    {
        // 時間固定なので、それを超過していたら、ただちに中断すべき。
        interrupt();
        return;
    }

    // 終了時刻が確定しているなら、そこ以降の時刻であれば停止させないといけない。
    if (s.time_manager.search_end)
    {
        if (elapsed_from_ponderhit >= s.time_manager.search_end)
        {
            // 終了予定時刻より時間が超過している。
            interrupt();
            return;
        }
        else
        {
            // 探索終了時刻は設定されているのでこれ以上、探索打ち切りの判定は不要。
            return;
        }
    }

    // 残り最適時間。残り最大時間。
    // このInterruptionCheck関数を呼び出す間隔(kCheckIntervalMs)を差し引いて考える必要がある。
    auto mimimum =
      std::max(s.time_manager.minimum() - SearchInterruptionChecker::kCheckIntervalMs, s64(0));
    auto optimum =
      std::max(s.time_manager.optimum() - SearchInterruptionChecker::kCheckIntervalMs, s64(0));
    auto maximum =
      std::max(s.time_manager.maximum() - SearchInterruptionChecker::kCheckIntervalMs, s64(0));

    const Node* current_root = tree->GetCurrentHead();
    const int   child_num    = current_root->child_num;
    if (child_num == 1)
    {
        // one replyなので最小時間だけ使い切れば良い。
        s.time_manager.search_end = mimimum;
        return;
    }

    // 最小思考時間を使っていない。
    if (elapsed_from_ponderhit < mimimum)
        return;

    // 最大思考時間を超過している。
    if (elapsed_from_ponderhit > maximum)
    {
        // この場合も余計な時間制御は不要。ただちに中断すべき。
        interrupt();
        return;
    }

    const ChildNode* uct_child = current_root->child.get();

    // 詰みを発見しているのか？
    for (size_t i = 0; i < current_root->child_num; ++i)
        if (uct_child[i].IsLose())
        {
            // 詰みを(少なくとも1つは)発見していた。
            interrupt();
            return;
            // ⇨　これreturnするまでにこのフラグ書き換えられたら堪らんのだが…。
            // 　即詰み手順、実は千日手だったから書き換えられてしまうパターンがあるか…。
            //   通常の対局で現実的に起こり得ないと思うので、まあ、いいや。
        }

    // 序盤32手目まで少なめのおむすび型にする。
    // TODO: パラメータの調整、optimizerでやるべき。
    // Time management (LC0 blog)     : https://lczero.org/blog/2018/09/time-management/
    // PR1195: Time management update : https://lczero.org/dev/docs/timemgr/
    double game_ply_factor =
      s.game_ply < 20
        ? 1.5  // 序盤では時間あまり使わないように。(時間を使ったところでそんなに良い指し手になるわけではないから)
      : s.game_ply < 30 ? 3.5
      : s.game_ply < 40 ? 4.0
                        : 3.0;
    // ⇑ここ、なめらかなほうがいいのかも知れないが、
    // もともと目分量で決めてるものなので細かいことは気にしないことにする。

    // maximum時間を基準に考えるので、これをoptimumをベースとして再計算する。
    maximum = std::min((s64) (optimum * game_ply_factor), maximum);

    // elapsed         : "go" , もしくは"go ponder"～"ponderhit"(のponderhit)からの経過時間
    // s.node_searched : 今回探索したノード数
    // nps(nodes per second) = 今回探索したノード数 / elapsed
    //                       = s.node_searched / (elapsed + 1)
    // ⇨ 0除算になるといけないので 分母を(elapsed + 1)にしている。
    //
    // rest_max_po(最大残りplayout) = nps × 最大までの残り時間
    // 最大までの残り時間 = maximum - elapsed_from_ponderhit
    // よって、
    // rest_max_po = s.node_searched * (maximum - elapsed_from_ponderhit) / (elapsed + 1)
    // 同様に、
    // rest_optimum_po(optimum timeまでの残りplayout) = nps × optimum timeまでの残り時間
    // optimum timeまでの残り時間 = optimum - elapsed_from_ponderhit
    // rest_optimum_po = s.node_searched * (optimum - elapsed_from_ponderhit) / (elapsed + 1)

    auto elapsed = s.time_manager.elapsed_time();

    // 残りoptimum po(予測値)
    //s64 rest_optimum_po = std::max((s64)(s.nodes_searched * (optimum - elapsed_from_ponderhit) / (elapsed + 1)), (s64)0);

    // 最大残りpo(予測値)
    s64 rest_maximum_po = std::max(
      (s64) (s.nodes_searched * (maximum - elapsed_from_ponderhit) / (elapsed + 1)), (s64) 0);

    // 残りの探索を全て次善手に費やしても optimum_timeまでに
    // 最善手を超えられない場合は探索を打ち切る。

    // best_searched   : move_countが最大の指し手のmove_count
    // second_searched : move_countが2番目の指し手のmove_count
    // すなわち、best_searched >= second_searched が成り立つ。

    // その時のindex
    int best_i = 0, second_i = -1, third_i = -1;

    // 探索回数が最も多い手と次に多い手の評価値を求める。
    for (int i = 1; i < child_num; i++)
    {
        if (uct_child[i].move_count > uct_child[best_i].move_count)
        {
            third_i  = second_i;
            second_i = best_i;
            best_i   = i;
        }
        else if (second_i == -1 || uct_child[i].move_count > uct_child[second_i].move_count)
        {
            third_i  = second_i;
            second_i = i;
        }
        else if (third_i == -1 || uct_child[i].move_count > uct_child[third_i].move_count)
            third_i = i;
    }

    NodeCountType best_searched   = uct_child[best_i].move_count;
    NodeCountType second_searched = uct_child[second_i].move_count;

    // best_winrate   : move_countが最大の指し手の勝率
    // second_winrate : move_countが2番目の指し手の勝率
    // ※　best_winrate >= second_winrate とは限らないので注意。

    const WinType delta          = (WinType) 0.00001f;  // 0割回避のための微小な値
    WinType       best_winrate   = uct_child[best_i].win / (best_searched + delta);
    WinType       second_winrate = uct_child[second_i].win / (second_searched + delta);

    // 条件に該当したらbreak(思考を終了)、さもなくばreturnするためのfor loop。
    for (;;)
    {
        if (rest_maximum_po > 0 /* maximum時間が残っている */)
        {
            WinType eval_diff = best_winrate - second_winrate;
            bool    converged = false;

            // special case : 指し手が合流してると推測されるケース。
            if (
              second_i != third_i && third_i != -1 && elapsed >= optimum / 8
              // && std::abs(eval_diff) < 0.02 && best_searched < second_searched * 1.1
              // ⇑この条件なくてもいいや。(桂成と桂不成みたいなケースにおいてはpolicyに差があるからevalが近い値にならない。
            )
            {
                // 指し手が本当に合流しているかPVの4手先を辿って確認する。
                // →　合流した結果千日手になるパターンは、nodeが作られてないから、このチェックにひっかからない。

                Node* node1 = current_root->child_nodes[best_i].get();
                Node* node2 = current_root->child_nodes[second_i].get();

                if (node1 != nullptr && node2 != nullptr)
                {
                    // rootPosはスレッドごとに用意されているのでmemcpyして問題ない。
                    Position pos;
                    std::memcpy(&pos, &rootPos, sizeof(Position));
                    StateInfo si;

                    Move m1 = uct_child[best_i].move;
                    Move m2 = uct_child[second_i].move;

                    //sync_cout << to_usi_string(m1) << sync_endl;
                    //sync_cout << to_usi_string(m2) << sync_endl;

                    pos.do_move(m1, si);
                    Key64 k1[4] = {}, k2[4] = {};
                    pv_key(pos, node1, 3, k1);  // 4手先までのhash key
                    pos.undo_move(m1);
                    pos.do_move(m2, si);
                    pv_key(pos, node2, 3, k2);  // 4手先までのhash key
                    pos.undo_move(m2);

                    // 現局面から数えてPVの2手先が一致するか4手先が一致するか。
                    // k1[0] = 4手先のhash key。k1[1] = 3手先のhash key。
                    // k1[2] = 2手先のhash key。k1[3] = 1手先のhash key。
                    if ((k1[0] == k2[0] && k1[0] != 0) || (k1[2] == k2[2] && k1[2] != 0))
                    {
                        // 合流しているので3番目の指し手と比較する。

                        // 他の指し手が台頭してきていれば良いのだが..
                        NodeCountType third_searched = uct_child[third_i].move_count;
                        WinType third_winrate = uct_child[third_i].win / (third_searched + delta);

                        // 3番目の指し手を2番目の指し手とみなす。
                        // これでこのあとの早期終了条件を満たすならそれで停止させれば良い。
                        second_searched = third_searched;
                        second_winrate  = third_winrate;
                        eval_diff       = best_winrate - third_winrate;

                        converged = true;
                    }
                }
            }

            // 安定した探索であると言える条件は、bestの訪問回数がsecondの1.5倍以上(この条件、重要)かつ、
            // bestの期待勝率がsecondの期待勝率を上回ること。
            if (best_winrate >= second_winrate && best_searched >= second_searched * 1.5)
            {
                // bestとsecondの勝率に応じて早期に思考を終了しても良いという考え。
                // 勝率差0.2なら、探索が早期に終了して良いと思う。
                WinType ratio = std::max(1.0 - eval_diff * 5, 0.0);

                // 経過時間がoptimum /8 を超えてるのに残りmaximum時間をすべて用いても訪問数が逆転しない。
                // ただしこの時、eval_diffが0.1なら50%というように、eval_diffの値に応じてrest_optimum_poを減らして考える。
                if (elapsed >= optimum / 8
                    && best_searched > second_searched + rest_maximum_po * ratio)
                {
                    if (o.debug_message)
                        sync_cout << "info string interrupted by early exit"
                                  << " , best_searched > second_searched + rest_maximum_po * "
                                  << ratio << " "
                                  << " , best_searched = " << best_searched
                                  << " , second_searched = " << second_searched
                                  << " , rest_maximum_po = " << rest_maximum_po
                                  << " , elapsed = " << elapsed << " , eval_diff = " << eval_diff
                                  << " , ratio = " << ratio << " , converged = " << converged
                                  << sync_endl;

                    break;
                }
            }

            // いま、おそらく best_winrate < second_winrate なので
            // これの行く末を見守る必要がある。(best_winrate >= second_winrateであって欲しい)
            // しかし、どう頑張っても訪問回数で叶わないなら、諦める。

            // maximum時間を超えていて、残り時間をすべて使っても訪問回数が逆転しない。
            // ⇨　残念だけど、あきらめる。

            if (best_searched > second_searched + rest_maximum_po)
            {
                if (o.debug_message)
                    sync_cout << "info string interrupted by retirement"
                              << " , best_searched > second_searched + rest_maximum_po"
                              << " , best_searched = " << best_searched
                              << " , second_searched = " << second_searched
                              << " , rest_maximum_po = " << rest_maximum_po << sync_endl;

                break;
            }
        }
        else
        {  // rest_optimum_po == 0

            if (o.debug_message)
                sync_cout << "info string maximum time is over"
                          << " , rest_maximum_po == 0"
                          << " , best_winrate = " << best_winrate
                          << " , second_winrate = " << second_winrate
                          << " , best_searched = " << best_searched
                          << " , second_searched = " << second_searched << sync_endl;

            break;
        }

        // どの条件にも該当しなかった。
        return;
    }
    // 残り時間くりあげて使って、終了すべき。
    s.time_manager.search_end = s.time_manager.round_up(elapsed_from_ponderhit);
}

// 並列探索を行う。
//   rootPos   : 探索開始局面
//   thread_id : スレッドID
// ※ やねうら王独自拡張
void DlshogiSearcher::parallel_search(const Position& rootPos, size_t thread_id)
{
	// このrootPosはスレッドごとに用意されていることが保証されているから
	// 単純なメモリコピー可能。

	// thread_id、割り当てられている末尾の1つは、GC用とSearchInterruptionChecker用なので
	// このidに応じて、処理を割り当てる。

	// やねうら王側でスレッド生成を一元管理するための、わりとシンプルで面白い実装だと思う。

	// GCもここから割り当てたかったが、そうすると、ガーベジの途中で探索が終了できなくて
	// bestmoveが返せないことがあるようなので、良くなかった。

	auto s = thread_id_to_uct_searcher.size();

	// 通常の探索スレッドは、並列UCT探索へ。
	if (thread_id < s)
		thread_id_to_uct_searcher[thread_id]->ParallelUctSearchStart(rootPos);

	// 探索終了判定用。
	else if (thread_id == s)
		interruption_checker->Worker(rootPos);

	else
		ASSERT_LV3(false);
}

// --------------------------------------------------------------------
//  SearchInterruptionChecker : 探索停止チェックを行うスレッド
// --------------------------------------------------------------------

// ガーベジ用のスレッドが実行するworker
// 探索開始時にこの関数を呼び出す。
void SearchInterruptionChecker::Worker(const Position& rootPos)
{
	// スレッド停止命令が来るまで、kCheckIntervalMs[ms]ごとにInterruptionCheck()を実行する。

	// 終了判定用の関数
	auto stop = [&]() {
		return ds->engine.threads.stop.load() || ds->search_limits.interruption;
	};

	// 最後に出力した時刻をリセットしておく。
	ds->ResetLastPvPrint();

	while (!stop() ) {
		std::this_thread::sleep_for(std::chrono::milliseconds(kCheckIntervalMs));

		// 探索の終了チェック
		ds->InterruptionCheck(rootPos);

		// ここにも終了判定を入れておいたほうが、探索停止確定にPV出力しなくてよろしい。
		if (stop())
			break;

		// PVの表示チェック
		ds->OutputPvCheck();
	};
}

} // namespace dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP)
