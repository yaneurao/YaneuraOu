#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "FukauraOuEngine.h"

#include "../../position.h"
#include "../../usi.h"
#include "../../thread.h"

//#include "dlshogi_types.h"
//#include "Node.h"
//#include "UctSearch.h"
//#include "dlshogi_searcher.h"
//#include "dlshogi_min.h"

#include "../../eval/deep/nn.h"
#include "../../eval/deep/nn_types.h"

using namespace YaneuraOu;

namespace dlshogi {

void SearchOptions::add_options(OptionsMap& options) {

	// USI_Ponder
    options.add("USI_Ponder", Option(false, [&](const Option& o) {
                    usi_ponder = o;
                    return std::nullopt;
                }));

	// PV出力間隔
    options.add("PV_Interval", Option(500, 0, int_max, [&](const Option& o) {
                    pv_interval = TimePoint(o);
                    return std::nullopt;
                }));

    // ノードを再利用するか。
    options.add("ReuseSubtree", Option(true, [&](const Option& o) {
                    reuse_subtree = o;
                    return std::nullopt;
                }));


    // 勝率を評価値に変換する時の定数。
    options.add("Eval_Coef", Option(285, 1, 10000, [&](const Option& o) {
                    eval_coef = float(o);
                    return std::nullopt;
                }));

    // 投了値 : 1000分率で
    options.add("Resign_Threshold", Option(0, 0, 1000, [&](const Option& o) {
                    RESIGN_THRESHOLD = int(options["Resign_Threshold"]) / 1000.0f;
                    return std::nullopt;
                }));

    // デバッグ用のメッセージ出力の有無。
    options.add("DebugMessage", Option(false, [&](const Option& o) {
                    debug_message = o;
                    return std::nullopt;
                }));

    // 💡 UCTノードの上限(この値を10億以上にするならWIN_TYPE_DOUBLEをdefineしてコンパイルしないと
    //     MCTSする時の勝率の計算精度足りないし、あとメモリも2TBは載ってないと足りないと思う…)

    //     これはノード制限ではなく、ノード上限を示す。この値を超えたら思考を中断するが、
    // 　  この値を超えていなくとも、持ち時間制御によって思考は中断する。
    // ※　探索ノード数を固定したい場合は、NodesLimitオプションを使うべし。
	options.add("UCT_NodeLimit", Option(10000000, 10, 1000000000, [&](const Option& o) {
                    uct_node_limit = NodeCountType(o);
                    return std::nullopt;
                }));

    // 引き分けの時の値 : 1000分率で
    // 引き分けの局面では、この値とみなす。
    // root color(探索開始局面の手番)に応じて、2通り。

    options.add("DrawValueBlack", Option(500, 0, 1000, [&](const Option& o) {
                    draw_value_black = int(o) / 1000.0f;
                    return std::nullopt;
                }));
    options.add("DrawValueWhite", Option(500, 0, 1000, [&](const Option& o) {
                    draw_value_white = int(o) / 1000.0f;
                    return std::nullopt;
                }));

    // --- PUCTの時の定数

	// これ、探索パラメーターの一種と考えられるから、最適な値を事前にチューニングして設定するように
    // しておき、ユーザーからは触れない(触らなくても良い)ようにしておく。
    // →　dlshogiはoptimizerで最適化するために外だししているようだ。

    // fpu_reductionの値を100分率で設定。
    // c_fpu_reduction_rootは、rootでのfpu_reductionの値。
    options.add("C_fpu_reduction", Option(27, 0, 100, [&](const Option& o) {
                    c_fpu_reduction = o / 100.0f;
                    return std::nullopt;
                }));
    options.add("C_fpu_reduction_root", Option(0, 0, 100, [&](const Option& o) {
                    c_fpu_reduction_root = o / 100.0f;
                    return std::nullopt;
                }));

    options.add("C_init", Option(144, 0, 500, [&](const Option& o) {
                    c_init = o / 100.0f;
                    return std::nullopt;
                }));
    options.add("C_base", Option(28288, 10000, 100000, [&](const Option& o) {
                    c_base = NodeCountType(o);
                    return std::nullopt;
                }));
    options.add("C_init_root", Option(116, 0, 500, [&](const Option& o) {
                    c_init_root = o / 100.0f;
                    return std::nullopt;
                }));
    options.add("C_base_root", Option(25617, 10000, 100000, [&](const Option& o) {
                    c_base_root = NodeCountType(o);
                    return std::nullopt;
                }));

    // softmaxの時のボルツマン温度設定
    // これは、dlshogiの"Softmax_Temperature"の値。(174) = 1.74
    // ※ 100分率で指定する。
    // hcpe3から学習させたmodelの場合、1.40～1.50ぐらいにしないといけない。
    // cf. https://tadaoyamaoka.hatenablog.com/entry/2021/04/05/215431

    options.add("Softmax_Temperature", Option(174, 1, 10000, [&](const Option& o) {
                    Eval::dlshogi::set_softmax_temperature( o / 100.0f);
                    return std::nullopt;
                }));

#if DLSHOGI
    //(*this)["Const_Playout"]               = USIOption(0, 0, int_max);
    // 🤔 Playout数固定。これはNodesLimitでできるので不要。

	dfpn_min_search_millisecs = options["DfPn_Min_Search_Millisecs"];
    // →　ふかうら王では、rootのdf-pnは、node数を指定することにした。
#endif

    // → leaf nodeではdf-pnに変更。
    // 探索ノード数の上限値を設定する。0 : 呼び出さない。
	options.add("LeafDfpnNodesLimit", Option(40, 0, 10000, [&](const Option& o) {
                    leaf_dfpn_nodes_limit = NodeCountType(o);
                    return std::nullopt;
                }));

    // PV lineの即詰みを調べるスレッドの数と1局面当たりの最大探索ノード数。
    options.add("PV_Mate_Search_Threads", Option(1, 0, 256));
    options.add("PV_Mate_Search_Nodes", Option(500000, 0, UINT32_MAX));


}

void FukauraOuEngine::add_nn_options()
{
    OptionsMap& options = get_options();

    // 各GPU用のDNNモデル名と、そのGPU用のUCT探索のスレッド数と、そのGPUに一度に何個の局面をまとめて評価(推論)を行わせるのか。

	// 使用するGPUの最大
	options.add("Max_GPU", Option(1, 1, 1024));

	// 無効化するGPU(カンマ区切りで)
	options.add("Disabled_GPU", Option(""));

    // RTX 3090で10bなら4、15bなら2で最適。
    options.add("UCT_Threads", Option(2, 0, 256));

#if defined(COREML)
    // Core MLでは、ONNXではなく独自形式のモデルが必要。
    options.add("DNN_Model", Option(R"(model.mlmodel)"));
#else
    options.add("DNN_Model", Option(R"(model.onnx)"));
#endif

#if defined(TENSOR_RT) || defined(ORT_TRT)
    // 通常時の推奨128 , 検討の時は推奨256。
    options.add("DNN_Batch_Size", Option(128, 1, 1024));
#elif defined(ONNXRUNTIME)
    // CPUを使っていることがあるので、default値、ちょっと少なめにしておく。
    options.add("DNN_Batch_Size", Option(32, 1, 1024));
#elif defined(COREML)
    // M1チップで8程度でスループットが飽和する。
    options.add("DNN_Batch_Size", Option(8, 1, 1024));
#endif
}

// ふかうら王のエンジンオプションを生やす
void FukauraOuEngine::add_options() {

	// NN関係の設定を生やす。
	add_nn_options();

    // 定跡のオプションを生やす
    book.add_options(options);

	// SearchOptionのオプションを生やす。
	manager.search_options.add_options(options);
}

// "Max_GPU","Disabled_GPU"と"UCT_Threads"の設定値から、各GPUのスレッド数の設定を返す。
std::vector<int> FukauraOuEngine::get_thread_settings() {

    // GPUのデバイス数を取得する
    int device_count = Eval::dlshogi::NN::get_device_count();

	// GPUの最大数
    int max_gpu    = std::min(int(options["Max_GPU"]), device_count);

	// 各GPUのスレッド数
	int thread_num = int(options["UCT_Threads"]);

	// スレッド設定
	std::vector<int> thread_settings;

	// GPUの数だけスレッド数を設定
    thread_settings.assign(thread_num, max_gpu);

	for (auto&& disabled : split(std::string(options["Disabled_GPU"]), ","))
	{
        int d = StringExtension::to_int(std::string(disabled), 0);
        if (d == 0)
			// 🤔 これ、parse失敗した警告を出しておいたほうがいいか？
            continue;

		// 番号は1 originである。
		if (1 <= d && d <= max_gpu)
			// 無効化するGPU番号に対応するスレッド数を0に設定する。
            thread_settings[d - 1] = 0;
	}

	return thread_settings;
}

void FukauraOuEngine::init_gpu()
{
    auto& options = get_options();

	// 各GPUのスレッド設定。無効化されているdeviceは0。
    auto thread_settings = get_thread_settings();

	// DNNのbatch sizeの設定。
    int dnn_batch_size = int(options["DNN_Batch_Size"]);

#if 0
    // ※　InitGPU()に先だってSetMateLimits()でのmate solverの初期化が必要。この呼出をInitGPU()のあとにしないこと！
    searcher.SetMateLimits((int) Options["MaxMovesToDraw"],
                           (u32) Options["RootMateSearchNodesLimit"],
                           (u32) Options["LeafDfpnNodesLimit"] /*Options["MateSearchPly"]*/);
#endif

	//InitGPU(Eval::dlshogi::ModelPaths, thread_settings, dnn_batch_size);

#if 0

    // PV lineの詰み探索の設定
    searcher.SetPvMateSearch(int(Options["PV_Mate_Search_Threads"]),
                             int(Options["PV_Mate_Search_Nodes"]));
#endif
}


// "isready"コマンド応答。
void FukauraOuEngine::isready() {

	// 評価関数テーブルの初期化(起動時でも良い)
	Eval::dlshogi::init();

    // -----------------------
    //   定跡の読み込み
    // -----------------------

    book.read_book();

    // -----------------------
    //   GPUの初期化
    // -----------------------

	init_gpu();





#if 0
	// dlshogiでは、
	// "isready"に対してnode limit = 1 , batch_size = 128 で探索しておく。
	// 初期局面に対してはわりと得か？

	// 初回探索をキャッシュ
	Position pos_tmp;
	StateInfo si;
	pos_tmp.set_hirate(&si);
	LimitsType limits;
	limits.nodes = 1;
	searcher.SetLimits(limits);
	Move ponder;
	auto start_threads = [&]() {
		searcher.parallel_search(pos_tmp,0);
	};
	searcher.UctSearchGenmove(&pos_tmp, pos_tmp.sfen(), {}, ponder , false, start_threads);
#endif
}

// エンジン作者名の変更
std::string FukauraOuEngine::get_engine_author() const { return "Tadao Yamaoka , yaneurao"; }


// 🚧 工事中 🚧


// やねうら王フレームワークと、dlshogiの橋渡しを行うコード

#if 0

// 探索部本体。とりま、globalに配置しておく。
DlshogiSearcher searcher;


// "go"コマンドに対して呼び出される。
void MainThread::search()
{
	// 開始局面の手番をglobalに格納しておいたほうが便利。
	searcher.search_limits.root_color = rootPos.side_to_move();

	// "NodesLimit"の値など、今回の"go"コマンドによって決定した値が反映される。
	searcher.SetLimits(&rootPos,Search::Limits);

	// MultiPV
	// ※　dlshogiでは現状未サポートだが、欲しいので追加しておく。
	// これは、isreadyのあと、goの直前まで変更可能
	searcher.search_options.multi_pv = (ChildNumType)Options["MultiPV"];

	// "position"コマンドが送られずに"go"がきた。
	if (game_root_sfen.empty())
		game_root_sfen = SFEN_HIRATE;

	Move ponderMove;
	Move move = searcher.UctSearchGenmove(&rootPos, game_root_sfen , moves_from_game_root , ponderMove);

	// ponder中であれば、呼び出し元で待機しなければならない。

	// 最大depth深さに到達したときに、ここまで実行が到達するが、
	// まだThreads.stopが生じていない。しかし、ponder中や、go infiniteによる探索の場合、
	// USI(UCI)プロトコルでは、"stop"や"ponderhit"コマンドをGUIから送られてくるまでbest moveを出力してはならない。
	// それゆえ、単にここでGUIからそれらのいずれかのコマンドが送られてくるまで待つ。
	// "stop"が送られてきたらThreads.stop == trueになる。
	// "ponderhit"が送られてきたらThreads.ponder == falseになるので、それを待つ。(stopOnPonderhitは用いない)
	// "go infinite"に対してはstopが送られてくるまで待つ。
	// ちなみにStockfishのほう、ここのコードに長らく同期上のバグがあった。
	// やねうら王のほうは、かなり早くからこの構造で書いていた。最近のStockfishではこの書き方に追随した。
	while (!Threads.stop && (this->ponder || Search::Limits.infinite))
	{
		//	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
		// (思考のためには計算資源を使っていないので。)
		Tools::sleep(1);

		// Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。
	}

	// silent modeでないなら、bestmoveとponderの指し手を出力する。
	if (!Search::Limits.silent)
	{
		sync_cout << "bestmove " << to_usi_string(move);

		// USI_Ponderがtrueならば、bestmoveに続けて、ponderの指し手も出力する。
		if (searcher.search_options.usi_ponder && ponderMove)
			std::cout << " ponder " << to_usi_string(ponderMove);

		std::cout << sync_endl;
	}
}

void Thread::search()
{
	// searcherが、このスレッドがどのインスタンスの
	// UCTSearcher::ParallelUctSearch()を呼び出すかを知っている。

	// このrootPosはスレッドごとに用意されているからコピー可能。

	searcher.parallel_search(rootPos,thread_id());
}

//MainThread::~MainThread()
//{
//	searcher.FinalizeUctSearch();
//}
// ⇨　まあ、プロセス終了するんだから開放されるやろ…。

namespace dlshogi
{
	// 探索結果を返す。
	//   Threads.start_thinking(pos, states , limits);
	//   Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。
	// のようにUSIのgoコマンド相当で探索したあと、rootの各候補手とそれに対応する評価値を返す。
	void GetSearchResult(std::vector < std::pair<Move, float>>& result)
	{
		// root node
		Node* root_node = searcher.search_limits.current_root;

		// 子ノードの数
		ChildNumType num = root_node->child_num;

		// 返し値として返す用のコンテナ
		result.clear();
		result.reserve(num);

		for (ChildNumType i = 0; i < num; ++i)
		{
			auto& child = root_node->child[i];
			Move m = child.move;
			// move_count == 0であって欲しくはないのだが…。
			float win = child.move_count == 0 ? child.nnrate : (float)child.win / child.move_count;
			result.emplace_back(std::pair<Move, float>(m, win));
		}
	}

	// 訪問回数上位の訪問回数自体を格納しておく構造体
	struct TopVisited
	{
		// n : 訪問回数の上位n個のPVを格納する
		TopVisited(size_t n_) : n(n_)
		{
			// n個は格納するはずなので事前に確保しておく。
			tops.reserve(n);
		}

		// 現在の上位N番目(ビリ)の訪問回数
		NodeCountType nth_nodes() const {
			if (tops.size() == 0)
				return 0;
			return tops.back();
		}

		// 訪問回数としてmを追加する。
		// topsには上位N個が残る。
		void append(NodeCountType m)
		{
			// 要素数が足りていないなら末尾にダミーの要素を追加しておく。
			if (tops.size() < n)
				tops.push_back(0);
			
			// 下位から順番にmがinsertできるところを探す。その間、要素を後ろにスライドさせていく。
			for(size_t i = tops.size() - 1 ; ; --i)
			{
				auto c = (i == 0) ?  std::numeric_limits<NodeCountType>::max() : tops[i - 1];

				// topsを後ろから見ていって、mを超える要素があればその一つ手前に挿入すれば良い。
				// これは必ず見つかる。
				// なぜなら、i == 0のとき c = max となるから、これよりmが大きいことはなく、
				// そこで必ずこの↓ifが成立する。
				if (c >= m)
				{
					// 挿入するところが見つかった。
					ASSERT_LV3(i < tops.size() );
					tops[i] = m;
					break;
				}

				// 要素をひとつ後ろにスライドさせる。
				tops[i] = c;
			}
		}

		// 保持している要素の個数を返す。
		size_t size() const { return tops.size(); }

	private:
		// 現在のtop N
		std::vector<NodeCountType> tops;

		// 上位n個を格納する。
		size_t n;
	};

	// 訪問回数上位n位の、訪問回数を取得するためのDFS(Depth First Search)
	//   node : 探索開始node
	//   tv   : 訪問回数上位 n
	//   ply  : rootからの手数
	void dfs_for_node_visited(Node* node, TopVisited& tv , int ply, bool same_color)
	{
		// rootではない偶数局面で、その訪問回数が現在のn thより多いならそれを記録する。
		if (   ply != 0
			&& (ply % 2) == (same_color ? 0 : 1)
			&& node->move_count > tv.nth_nodes()
			)
			tv.append(node->move_count);

		// 子ノードの数
		ChildNumType num = node->child_num;
		for (int i = 0; i < num; ++i)
		{
			auto& child = node->child[i];
			NodeCountType move_count = child.move_count;

			// n番目を超えているのでこの訪問回数を追加する。
			if (move_count > tv.nth_nodes())
				// このnodeを再帰的に辿る必要がある。
				// 超えていないものは辿らない、すなわち枝刈りする。
				dfs_for_node_visited(node->child_nodes[i].get(), tv , ply + 1, same_color);
		}
	}

	// 訪問回数上位n位の、訪問回数を取得するためのDFS(Depth First Search)
	//   node  : 探索開始node
	//   tv    : 訪問回数上位 n
	//   ply   : rootからの手数
	//   sfens : 訪問回数上位 n のnodeまでのroot nodeからの手順文字列(先頭にスペースが入る)
	//		→　これは返し値
	//   pv    : rootから現在の局面までの手順
	void dfs_for_sfen(Node* node, TopVisited& tv , int ply , bool same_color, SfenNodeList& snlist , std::vector<Move>& pv)
	{
		// rootではない偶数局面で、その訪問回数が現在のn thより多いならそれを記録する。
		if (   ply != 0
			&& (ply % 2) == (same_color ? 0 : 1)
			&& node->move_count >= tv.nth_nodes()
			&& snlist.size() < tv.size()
			)
		{
			std::string pv_string;

			// PV手順をUSIの文字列化して連結する。(先頭にスペースが入る)
			for(auto m : pv)
				pv_string += " " + to_usi_string(m);
			snlist.emplace_back(SfenNode(pv_string, node->move_count));
		}

		// 子ノードの数
		ChildNumType num = node->child_num;
		for (int i = 0; i < num; ++i)
		{
			auto& child = node->child[i];

			NodeCountType move_count = child.move_count;
			Move          m          = child.move;

			// n番目以上なのでこの訪問回数を追加する。
			if (   move_count >= tv.nth_nodes()
				&& node->child_nodes          != nullptr
				&& node->child_nodes[i].get() != nullptr)
			{
				// このnodeを再帰的に辿る必要がある。
				// move_count以下のものは辿らない、すなわち枝刈りする。
				pv.push_back(m);
				dfs_for_sfen(node->child_nodes[i].get(), tv , ply + 1, same_color, snlist, pv);
				pv.pop_back();
			}
		}
	}

	// 訪問回数上位 n 個の局面のsfen文字列を返す。文字列の先頭にスペースが入る。
	void GetTopVisitedNodes(size_t n, SfenNodeList& snlist, bool same_color)
	{
		// root node
		Node* root_node = searcher.search_limits.current_root;

		// n番目の訪問回数xをまずDFSで確定させて、
		// そのあとx以上の訪問回数を持つ偶数node(rootと同じ手番を持つnode)を列挙すれば良い。

		TopVisited tv(n);
		dfs_for_node_visited(root_node, tv , 0, same_color);

		// n番目まで訪問回数が確定したので、再度dfsして局面を取り出す。

		std::vector<Move> pv;
		dfs_for_sfen        (root_node, tv , 0, same_color, snlist, pv);

		// 上位n個(同じ訪問回数のものがあると溢れてる可能性があるのでsortして上位n個を取り出す。)
		std::sort(snlist.begin(), snlist.end());

		if (snlist.size() > n)
			snlist.resize(n);
	}

	// 探索したノード数を返す。
	// これは、ThreadPool classがnodes_searched()で返す値とは異なる。
	//  →　そちらは、Position::do_move()した回数。
	// こちらは、GPUでevaluate()を呼び出した回数
	u64 nodes_visited()
	{
		return (u64)searcher.search_limits.nodes_searched;
	}
}

// USIの"gameover"に対して呼び出されるハンドラ。
void gameover_handler(const std::string& cmd)
{
	// dlshogiのゲームオーバーのハンドラを呼び出す。
	searcher.GameOver();
}

#endif

} // namespace dlshogi

namespace {

// 自作のエンジンのentry point
void engine_main() {

    // ここで作ったエンジン
    dlshogi::FukauraOuEngine engine;

    // USIコマンドの応答部
    YaneuraOu::USIEngine usi;
    usi.set_engine(engine);  // エンジン実装を差し替える。

    // USIコマンドの応答のためのループ
    usi.loop();
}

// このentry pointを登録しておく。
static YaneuraOu::EngineFuncRegister r(engine_main, "FukauraOuEngine", 0);
}


#endif // defined(YANEURAOU_ENGINE_DEEP)
