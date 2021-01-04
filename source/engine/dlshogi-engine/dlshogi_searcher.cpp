#include "dlshogi_searcher.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "dlshogi_types.h"
#include "UctSearch.h"
#include "PrintInfo.h"

#include "../../search.h"
#include "../../thread.h"
#include "../../mate/mate.h"

namespace dlshogi
{
	// --------------------------------------------------------------------
	//  DlshogiSearcher : dlshogiの探索部で、globalになっていたものをクラス化したもの。
	// --------------------------------------------------------------------

	DlshogiSearcher::DlshogiSearcher()
	{
		search_groups = std::make_unique<UctSearcherGroup[]>(max_gpu);
		gc = std::make_unique<NodeGarbageCollector>();
		interruption_checker = std::make_unique<SearchInterruptionChecker>(this);
	}

	// エンジンオプションの"USI_Ponder"の値をセットする。
	// "bestmove XXX ponder YYY"
	// のようにponderの指し手を返すようになる。
	void DlshogiSearcher::SetPonderingMode(bool flag)
	{
		search_options.usi_ponder = flag;
	}

	// GPUの初期化、各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
	// "isready"に対して呼び出される。
	void DlshogiSearcher::InitGPU(const std::vector<std::string>& model_paths , std::vector<int> new_thread, std::vector<int> policy_value_batch_maxsizes)
	{
		ASSERT_LV3(model_paths.size() == max_gpu);
		ASSERT_LV3(new_thread.size() == max_gpu);
		ASSERT_LV3(policy_value_batch_maxsizes.size() == max_gpu);

		// この時、前回設定と異なるなら、スレッドの再確保を行う必要がある。

		// トータルのスレッド数go
		size_t total_thread_num = 0;
		for (int i = 0; i < max_gpu; ++i)
			total_thread_num += new_thread[i];

		// スレッド数の合計が0はさすがにおかしい。
		if (total_thread_num == 0)
		{
			sync_cout << "Error! : total threads = 0 " << sync_endl;
			return;
		}

		// まず、数だけ確保
		// やねうら王のThreadPoolクラスは、前回と異なるスレッド数であれば自動的に再確保される。
		// interruption_check_threadの分、+1して確保する。
		// GC用のスレッドも探索スレッドから割り当てたのだが、それは良くないアイデアだった。
		// ※　GC処理が終わらなくて、全探索スレッドの終了を待つコードになっているから、bestmoveが返せないことがある。
		Threads.set(total_thread_num + 1);

		for (int i = 0; i < max_gpu; i++) {
			if (new_thread[i] > 0) {
				int policy_value_batch_maxsize = policy_value_batch_maxsizes[i];

				//	DNN_Batch_Size2～8は、0が設定されている場合、DNN_Batch_Size1の値が採用される。(そのスレッド数が1以上であるとき)
				if (policy_value_batch_maxsize == 0)
					policy_value_batch_maxsize = policy_value_batch_maxsizes[0];

				std::string path = model_paths[i];

				// paths[1]..paths[7]は、空であればpaths[0]と同様であるものとする。
				if (i > 0 && path == "")
					path = model_paths[0];

				search_groups[i].Initialize(path , new_thread[i],/* gpu_id = */i, policy_value_batch_maxsize);
			}
		}

		// -- やねうら王独自拡張

		// 何番目のスレッドがどのUctSearcherGroupのインスタンスに割り当たるのかのmapperだけ設定しておいてやる。
		// Thread::search()からは、このmapperを使って、UctSearchGroup::search()を呼び出す。

		thread_id_to_uct_searcher.clear();

		for (int i = 0; i < max_gpu; ++i)
			for(int j = 0;j < new_thread[i];++j)
				thread_id_to_uct_searcher.push_back(search_groups[i].get_uct_searcher(j));

		// GC用のスレッドにもスレッド番号を連番で与えておく。
		// (WinProcGroup::bindThisThread()用)
		gc->set_thread_id(thread_id_to_uct_searcher.size());
	}

	// 対局開始時に呼び出されるハンドラ
	void DlshogiSearcher::NewGame()
	{
	}

	// 対局終了時に呼び出されるハンドラ
	void DlshogiSearcher::GameOver()
	{
	}

	// 投了の閾値設定
	//   resign_threshold : 1000分率で勝率を指定する。この値になると投了する。
	void DlshogiSearcher::SetResignThreshold(const int resign_threshold)
	{
		search_options.RESIGN_THRESHOLD = (float)resign_threshold / 1000.0f;
	}

	// 千日手の価値設定
	//   value_black           : この値を先手の千日手の価値とみなす。(千分率)
	//   value_white           : この値を後手の千日手の価値とみなす。(千分率)
	//   draw_value_from_black : エンジンオプションの"Draw_Value_From_Black"の値。
	void DlshogiSearcher::SetDrawValue(const int value_black, const int value_white,bool draw_value_from_black)
	{
		search_options.draw_value_black = (float)value_black / 1000.0f;
		search_options.draw_value_white = (float)value_white / 1000.0f;
		search_options.draw_value_from_black = draw_value_from_black;
	}

	// →　これは、エンジンオプションの"MaxMovesToDraw"を自動的にDlshogiSearcher::SetLimits()で設定するので不要。
	//// 引き分けとする手数の設定
	//void SetDrawPly(const int ply)
	//{
	//	draw_ply = ply;
	//}

	//  ノード再利用の設定
	//    flag : 探索したノードの再利用をするのか
	void DlshogiSearcher::SetReuseSubtree(bool flag)
	{
		search_options.reuse_subtree = flag;
	}

	// PV表示間隔設定
	void DlshogiSearcher::SetPvInterval(const TimePoint interval)
	{
		search_options.pv_interval = interval;
	}

	// DebugMessageの出力。
	// エンジンオプションの"DebugMessage"の値をセットする。
	// search_options.debug_messageに反映される。
	void DlshogiSearcher::SetDebugMessage(bool flag)
	{
		search_options.debug_message = flag;
	}

	//  UCT探索の初期設定
	void DlshogiSearcher::InitializeUctSearch(NodeCountType uct_node_limit)
	{
		search_options.uct_node_limit = uct_node_limit;

		if (!tree) tree = std::make_unique<NodeTree>(gc.get());
		//search_groups = std::make_unique<UctSearcherGroup[]>(max_gpu);
		// →　これもっと早い段階で行わないと間に合わない。コンストラクタに移動させる。

		// dlshogiにはないが、dlshogiでglobalだった変数にアクセスするために、
		// UctSearcherGroupは、DlshogiSearcher*を持たなければならない。
		for (int i = 0; i < max_gpu; ++i)
			search_groups[i].set_dlsearcher(this);
	}

	//  UCT探索の終了処理
	void DlshogiSearcher::TerminateUctSearch()
	{
		// やねうら王では、スレッドはThreadPoolクラスで生成しているのでここは関係ない。

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
	//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limits.max_moves_to_drawに反映する。
	// などなど。
	// その他、"go"コマンドで渡された残り時間等から、今回の思考時間を算出し、search_limits.time_managerに反映する。
	void DlshogiSearcher::SetLimits(const Position* pos, const Search::LimitsType& limits)
	{
		auto& s = search_limits;
		auto& o = search_options;

		// go infiniteされているのか
		s.infinite = limits.infinite;

		// 探索を打ち切るnode数
		s.nodes_limit = (NodeCountType)limits.nodes;

		// 思考時間固定の時の指定
		s.movetime = limits.movetime;

		// 引き分けになる手数の設定。
		o.max_moves_to_draw = limits.max_game_ply;

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
		search_limits.time_manager.init(limits,pos->side_to_move(),pos->game_ply());

	}

	// ノード数固定にしたい時は、USIの"go nodes XXX"ででき、これは、SetLimits()で反映するので↓は不要。

	//// 1手のプレイアウト回数を固定したモード
	//// 0の場合は無効
	//void SetConstPlayout(const int playout)
	//{
	//	const_playout = playout;
	//}

	// 終了させるために、search_groupsを開放する。
	void DlshogiSearcher::FinalizeUctSearch()
	{
		search_groups.release();
		tree.release(); // treeの開放を行う時にGCが必要なのでCGをあとから開放
		gc.release();
	}


	// UCTアルゴリズムによる着手生成
	// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
	// ※　事前にSetLimits()で探索条件を設定しておくこと。
	//   pos           : 探索開始局面
	//   gameRootSfen  : 対局開始局面のsfen文字列(探索開始局面ではない)
	//   moves         : 探索開始局面からの手順
	//   ponderMove    : [Out] ponderの指し手(ないときはMOVE_NONEが代入される)
	//   start_threads    : この関数を呼び出すと全スレッドがParallelUctSearch()を呼び出して探索を開始する。
	//   teminate_threads :	ponderが解除されるのを待機して、そのあとstart_threadsで開始した全スレッドが終了するのを待機する。
	//                    // start_threadsを呼び出さずにteminate_threads()だけ呼び出すことがある。
	// 返し値 : この局面でのbestな指し手
	// この関数は、定跡にhitした時や宣言勝ちなどで、実際の探索を行わない場合でもteminate_threads()を呼び出してから
	// リターンすることは保証する。
	Move DlshogiSearcher::UctSearchGenmove(Position* pos, const std::string& gameRootSfen, const std::vector<Move>& moves, Move& ponderMove,
		const std::function<void()>& start_threads,
		const std::function<void()>& terminate_threads
		)
	{
		// 定跡にhitした場合や詰みを見つけた場合、宣言勝ちをした場合など、ponder中でも
		// この関数は速攻でbestmoveを返すことはある。
		// しかしponder中ならそれが解除されるまでbestmoveを返すのを待機しなければならない。
		SCOPE_EXIT(terminate_threads(););

		// これ[Out]なのでとりあえず初期化しておかないと忘れてしまう。
		ponderMove = MOVE_NONE;

		// 探索停止フラグをreset。
		// →　やねうら王では使わない。Threads.stopかsearch_limits.interruptionを使う。
		//search_limits.uct_search_stop = false;

		// begin_timeを初期化したかのフラグ
		// →　やねうら王ではTimerクラスを使ってgoコマンドからの経過時間を計測しているので不要
		//init_search_begin_time = false;

		// 探索開始時にタイマーをリセットして経過時間を計測する。
		search_limits.time_manager.reset();

		// 中断フラグのリセット
		search_limits.interruption = false;

		// ゲーム木を現在の局面にリセット
		tree->ResetToPosition(gameRootSfen, moves);

		// ルート局面をグローバル変数に保存
		//pos_root =  pos;

		// 探索開始局面
		const Node* current_root = tree->GetCurrentHead();
		search_limits.current_root = tree->GetCurrentHead();

		// "go ponder"で呼び出されているかのフラグの設定
		//search_limits.pondering = ponder;

		// 探索ノード数のクリア
		search_limits.nodes_searched = 0;

		// UCTの初期化。
		// 探索開始局面の初期化
		ExpandRoot(pos , search_options.generate_all_legal_moves );

		// ---------------------
		//     詰まされチェック
		// ---------------------

		// ExpandRoot()が呼び出されている以上、子ノードの初期化は完了していて、この局面の合法手の数がchild_numに
		// 代入されているはず。これが0であれば、合法手がないということだから、詰みの局面であり、探索ができることは何もない。
		const int child_num = current_root->child_num;
		if (child_num == 0) {
			// 投了しておく。
			return MOVE_RESIGN;
		}

		// ---------------------
		//     詰みのチェック
		// ---------------------

		// この局面で子ノードから伝播されて、勝ちが確定している。
		if (current_root->value_win == VALUE_WIN)
		{
			if (!pos->in_check())
			{
				// 詰みを発見しているはず
				// Solverで解かせてみる。

				// 宣言勝ちか？
				Move move = pos->DeclarationWin();

				if (move)
					return move;

				// いくらなんでも5手詰めで発見できるはずだが？
				move = mate_solver.mate_odd_ply(*pos, 5 ,true);

				if (move)
					return move;

				// これ、子ノードのいずれかは勝ちのはずだからその子ノードを選択したほうが良いのでは…。
				// 伝播された勝ちが本当ならば、だが。
				// とりま、↓以下、普通に探索する。
			}
		}

		// ---------------------
		//     定跡の選択部
		// ---------------------

		// 定跡DBにhitするか調べる。

		// main threadの取得
		auto th = pos->this_thread();

		if (book.probe(*th, Search::Limits))
		{
			// 定跡にhitしている以上、合法手がここに格納されているはず。
			Move bestMove   = th->rootMoves[0].pv[0];
			     ponderMove = th->rootMoves[0].pv[1];

			return bestMove;
		}

		// ---------------------
		//     即詰みの用のdf-pn
		// ---------------------

		// 長手数の詰みはdf-pn呼び出す
		// 呼び出したほうが強いのかがわからないので検証環境を用意してから実装する。


		// ---------------------
		//     並列探索の開始
		// ---------------------

		// 前回、この現在の探索局面を何回訪問したのか
		NodeCountType pre_simulated = current_root->move_count;

		// 探索時間とプレイアウト回数の予測値を出力
		if (search_options.debug_message)
			UctPrint::PrintPlayoutLimits(search_limits.time_manager , search_limits.nodes_limit);

		// 探索スレッドの開始
		start_threads();

		// 探索スレッドの終了
		terminate_threads();

		// ---------------------
		//     PV出力して終了
		// ---------------------

		// ponderモードでは指し手自体は返さない。
		// →　やねうら王では、stopが来るまで待機して返す。
		//  dlshogiの実装はいったんUCT探索を終了させるようになっているのでコメントアウト。
		//if (pondering)
		//	return MOVE_NONE;

		// あとで
		// 探索の延長判定
		// →　これは探索の停止判定で行うから削除

		// この時点で探索スレッドをすべて停止させないと
		// Virtual Lossを元に戻す前にbestmoveを選出してしまう。

		// PVの取得と表示
		auto best = UctPrint::get_best_move_multipv(current_root , search_limits , search_options);
		// それに対するponderの指し手もあるはずなのでそれをセットしておく。
		ponderMove = best.ponder;

		// 探索にかかった時間を求める
		const TimePoint finish_time = search_limits.time_manager.elapsed();
		//search_limits.remaining_time[pos->side_to_move()] -= finish_time;

		// デバッグ用のメッセージ出力
		if (search_options.debug_message)
		{
			// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
			UctPrint::PrintPlayoutInformation(current_root, &search_limits, finish_time, pre_simulated);
		}

		return best.move;
	}

	// Root Node(探索開始局面)を展開する。
	// generate_all : 歩の不成なども生成する。
	void DlshogiSearcher::ExpandRoot(const Position* pos , bool generate_all)
	{
		Node* current_head = tree->GetCurrentHead();
		if (current_head->child_num == 0) {
			current_head->ExpandNode(pos , generate_all);
		}
	}

	// PV表示の確認
	// SearchInterruptionCheckerから呼び出される。
	void DlshogiSearcher::OutputPvCheck()
	{
		auto& s = search_limits;
		auto& o = search_options;

		// "PvInterval"が0のときはPVの出力は行わない。
		if (!o.pv_interval)
			return;

		// 前回からの経過時刻。
		const auto elapsed_time = s.time_manager.elapsed();
		if (elapsed_time > s.last_pv_print + o.pv_interval) {

			// PV表示
			//get_and_print_pv();
			UctPrint::get_best_move_multipv(tree->GetCurrentHead() , search_limits , search_options );

			// 出力が終わった時点から数えて pv_interval後以降に再度表示。
			// (前回の出力時刻から数えてしまうと、PVの出力がたくさんあるとき出力が間に合わなくなる)
			s.last_pv_print = elapsed_time;
		}
	}

	//  探索停止の確認
	// SearchInterruptionCheckerから呼び出される。
	void DlshogiSearcher::InterruptionCheck()
	{
		auto& s = search_limits;
		auto& o = search_options;

		// すでに中断することが確定しているのでここでは何もしない。
		if (s.interruption || Threads.stop)
			return;

		// 思考時間は、ponderhitからの時間で考える。
		const auto elapsed_from_ponderhit = s.time_manager.elapsed_from_ponderhit();

		//if (s.time_manager.search_end)
		//	sync_cout << "wait = " << elapsed_from_ponderhit << sync_endl;

		// 探索を停止させる関数
		// (デバッグ表示ありなら、この止めたタイミングを出力する。
		// これにより、探索を停止させてから何ms 停止するまでに要するかがわかる。
		auto interrupt = [&]()
		{
			if (o.debug_message)
				sync_cout << "info string search search_end = " << s.time_manager.search_end
						  << "[ms], interruption time = " << elapsed_from_ponderhit << "[ms]" << sync_endl;

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
		if ((NodeCountType)s.current_root->move_count >= o.uct_node_limit)
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
		if (Threads.main()->ponder)
			return;
			
		// -- 時間制御

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
			if (elapsed_from_ponderhit >= s.time_manager.search_end)
			{
				// 終了予定時刻より時間が超過している。
				interrupt();
				return;
			}
			else {
				// 探索終了時刻は設定されているのでこれ以上、探索打ち切りの判定は不要。
			return;
		}

		const Node* current_root = tree->GetCurrentHead();
		const int child_num = current_root->child_num;
		if (child_num == 1)
		{
			// one replyなので最小時間だけ使い切れば良い。
			s.time_manager.search_end = s.time_manager.minimum();
			return;
		}

		// 最小思考時間を使っていない。
		if (elapsed_from_ponderhit < s.time_manager.minimum())
			return ;

		// 最大思考時間を超過している。
		if (elapsed_from_ponderhit > s.time_manager.maximum())
		{
			// この場合も余計な時間制御は不要。ただちに中断すべき。
			interrupt();
			return;
		}

		// 最適時間の1/3は必ず使うべき。
		if (elapsed_from_ponderhit < s.time_manager.optimum() * 1/3)
			return ;

		// 最適時間のrate倍を基準時間と考える。
		// なぜなら、残り時間を使っても1番目の指し手が2番目の指し手を超えないことが判明して探索を打ち切ることがあるから(頻度はそれほど高くない)
		// 平均的にoptimum()の時間が使えていない。そこでoptimumに係数をかけて、それを基準に考える。
		const float rate = 1.2f;
		auto optimum = std::min((TimePoint)(s.time_manager.optimum() * rate), s.time_manager.maximum());

		if (elapsed_from_ponderhit >= optimum)
		{
			// 残り時間くりあげて使って、終了すべき。
			s.time_manager.search_end = s.time_manager.round_up(elapsed_from_ponderhit);
			return;
		}

		// 残りの探索を全て次善手に費やしても optimum_timeまでに
		// 最善手を超えられない場合は探索を打ち切る。
		// TODO : 勝率の差とかも考慮したほうが良いのではないだろうか…。

		NodeCountType max_searched = 0, second = 0;
		const ChildNode* uct_child = current_root->child.get();

		// 探索回数が最も多い手と次に多い手を求める
		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].move_count > max_searched) {
				second = max_searched;
				max_searched = uct_child[i].move_count;
			}
			else if (uct_child[i].move_count > second) {
				second = uct_child[i].move_count;
			}
		}

		// elapsed : "go" , "go ponder"からの経過時間
		// rest_po : 残り何回ぐらいplayoutできそうか。

		// nps = 今回探索したノード数 / "go ponder"からの経過時間
		// 今回探索したノード数 = node_searched
		// なので、nps = node_searched / (e + 1)
		// ※　0のとき0除算になるので分母を +1 する。
		// rest_po = nps × 残り時間
		// 残り時間 = optimum - elapsed
		// なので、
		// rest_po = (node_searched - pre_simulated)*(optimum - elapsed) / (e + 1)
		
		auto elapsed = s.time_manager.elapsed();
		s64 rest_po = (s64)(s.nodes_searched * (optimum - elapsed_from_ponderhit) / (elapsed + 1) );

		// 何か条件をいじっているときに、rest_poがマイナスになりうるようになってしまうことがあるので
		// マイナスであれば0とみなす。
		rest_po = std::max(rest_po, (s64)0);

		if (max_searched - second > rest_po) {
			if (o.debug_message)
				sync_cout << "info string interrupt_no_movechange , max_searched = " << max_searched << " , second = " << second
						  << " , rest po = " << rest_po << sync_endl;

			// だめぽ。時間を秒のギリギリまで使ったら、もう探索を切り上げる。

			// "ponderhit"しているときは、そこからの経過時間を丸める。
			// "ponderhit"していないときは開始からの経過時間を丸める。
			// そのいずれもtime_manager.elapsed_from_ponderhit()から計算して良い。
			s.time_manager.search_end = s.time_manager.round_up(elapsed_from_ponderhit);
		}
	}

#if 0
	//  思考時間延長の確認
	bool DlshogiSearcher::ExtendTime()
	{
		// あとでよく見る。

		// 1番探索した指し手のノード数、2番目に探索した指し手のノード数
		NodeCountType max = 0, second = 0;
		float max_eval = 0, second_eval = 0;
		const Node* current_root = tree->GetCurrentHead();
		const int child_num = current_root->child_num;
		const ChildNode *uct_child = current_root->child.get();

		// 探索回数が最も多い手と次に多い手を求める
		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].move_count > max) {
				second = max;
				second_eval = max_eval;
				max = uct_child[i].move_count;
				max_eval = uct_child[i].win / uct_child[i].move_count;
			}
			else if (uct_child[i].move_count > second) {
				second = uct_child[i].move_count;
				second_eval = uct_child[i].win / uct_child[i].move_count;
			}
		}

		// 最善手の探索回数がが次善手の探索回数の1.5倍未満
		// もしくは、勝率が逆なら探索延長
		if (max < second * 1.5 || max_eval < second_eval) {
			return true;
		}
		else {
			return false;
		}
	}
#endif

	// 並列探索を行う。
	//   rootPos   : 探索開始局面
	//   thread_id : スレッドID
	// ※ やねうら王独自拡張
	void DlshogiSearcher::parallel_search(const Position& rootPos, size_t thread_id)
	{
		// このrootPosはスレッドごとに用意されているから単純なメモリコピー可能。

		// thread_id、割り当てられている末尾の1つは、GC用とSearchInterruptionChecker用なので
		// このidに応じて、処理を割り当てる。

		// やねうら王側でスレッド生成を一元管理するための、わりとシンプルで面白い実装だと思う。

		// GCもここから割り当てたかったが、そうすると、ガーベジの途中で探索が終了できなくて
		// bestmoveが返せないことがあるようなので、良くなかった。

		auto s = thread_id_to_uct_searcher.size();

		// 通常の探索スレッドは、並列UCT探索へ。
		if (thread_id < s)
		thread_id_to_uct_searcher[thread_id]->ParallelUctSearch(rootPos);

		// 探索終了判定用。
		else if (thread_id == s)
			interruption_checker->Worker();

		else
			ASSERT_LV3(false);
	}

	// --------------------------------------------------------------------
	//  SearchInterruptionChecker : 探索停止チェックを行うスレッド
	// --------------------------------------------------------------------

	// ガーベジ用のスレッドが実行するworker
	// 探索開始時にこの関数を呼び出す。
	void SearchInterruptionChecker::Worker()
	{
		// スレッド停止命令が来るまで、kCheckIntervalMs[ms]ごとにInterruptionCheck()を実行する。

		// 終了判定用の関数
		auto stop = [&]() {
			return Threads.stop.load() || ds->search_limits.interruption;
		};

		// 最後に出力した時刻をリセットしておく。
		ds->ResetLastPvPrint();

		while (!stop() ) {
			std::this_thread::sleep_for(std::chrono::milliseconds(kCheckIntervalMs));

			// 探索の終了チェック
			ds->InterruptionCheck();

			// ここにも終了判定を入れておいたほうが、探索停止確定にPV出力しなくてよろしい。
			if (stop())
				break;

			// PVの表示チェック
			ds->OutputPvCheck();
		};
	}
}

#endif // defined(YANEURAOU_ENGINE_DEEP)
