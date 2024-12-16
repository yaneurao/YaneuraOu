#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"
#include "../../usi.h"
#include "../../thread.h"

#include "dlshogi_types.h"
#include "Node.h"
#include "UctSearch.h"
#include "dlshogi_searcher.h"
#include "dlshogi_min.h"

#include "../../eval/deep/nn_types.h"

// やねうら王フレームワークと、dlshogiの橋渡しを行うコード

// --- やねうら王のsearchのoverride

using namespace dlshogi;
using namespace Eval::dlshogi;

// 探索部本体。とりま、globalに配置しておく。
DlshogiSearcher searcher;

void USI::extra_option(USI::OptionsMap& o)
{
	// エンジンオプションを生やす

	//   定跡のオプションを生やす
	searcher.book.init(o);

#if 0
    (*this)["Book_File"]                   = USIOption("book.bin");
    (*this)["Best_Book_Move"]              = USIOption(true);
    (*this)["OwnBook"]                     = USIOption(false);
    (*this)["Min_Book_Score"]              = USIOption(-3000, -ScoreInfinite, ScoreInfinite);
    (*this)["USI_Ponder"]                  = USIOption(false);
    (*this)["Stochastic_Ponder"]           = USIOption(true);
    (*this)["Time_Margin"]                 = USIOption(1000, 0, int_max);
    (*this)["Mate_Root_Search"]            = USIOption(29, 0, 35);
    (*this)["DfPn_Hash"]                   = USIOption(2048, 64, 4096); // DfPnハッシュサイズ
    (*this)["DfPn_Min_Search_Millisecs"]   = USIOption(300, 0, int_max);
#endif

#if defined(MAKE_BOOK)
	// 定跡を生成するときはPV出力は抑制したほうが良さげ。
    o["PV_Interval"]                 << USI::Option(0, 0, int_max);
    o["Save_Book_Interval"]          << USI::Option(100, 0, int_max);
#else
    o["PV_Interval"]                 << USI::Option(500, 0, int_max);
#endif // defined(MAKE_BOOK)
	
	// UCTノードの上限(この値を10億以上にするならWIN_TYPE_DOUBLEをdefineしてコンパイルしないと
	// MCTSする時の勝率の計算精度足りないし、あとメモリも2TBは載ってないと足りないと思う…)
	o["UCT_NodeLimit"]				 << USI::Option(10000000, 10, 1000000000);

	// デバッグ用のメッセージ出力の有無
	o["DebugMessage"]                << USI::Option(false);

	// ノードを再利用するか。
    o["ReuseSubtree"]                << USI::Option(true);

	// 勝率を評価値に変換する時の定数。
	o["Eval_Coef"]                   << USI::Option(285, 1, 10000);

	// 投了値 : 1000分率で
	o["Resign_Threshold"]            << USI::Option(0, 0, 1000);

	// 引き分けの時の値 : 1000分率で
	// 引き分けの局面では、この値とみなす。
	// root color(探索開始局面の手番)に応じて、2通り。

	o["DrawValueBlack"]              << USI::Option(500, 0, 1000);
	o["DrawValueWhite"]              << USI::Option(500, 0, 1000);

	// --- PUCTの時の定数
	// これ、探索パラメーターの一種と考えられるから、最適な値を事前にチューニングして設定するように
	// しておき、ユーザーからは触れない(触らなくても良い)ようにしておく。
	// →　dlshogiはoptimizerで最適化するために外だししているようだ。

	// fpu_reductionの値を100分率で設定。
	// c_fpu_reduction_rootは、rootでのfpu_reductionの値。
    o["C_fpu_reduction"]             << USI::Option(27, 0, 100);
    o["C_fpu_reduction_root"]        << USI::Option(0, 0, 100);

    o["C_init"]                      << USI::Option(144, 0, 500);
    o["C_base"]                      << USI::Option(28288, 10000, 100000);
    o["C_init_root"]                 << USI::Option(116, 0, 500);
    o["C_base_root"]                 << USI::Option(25617, 10000, 100000);

	// 探索のSoftmaxの温度
	o["Softmax_Temperature"]		 << USI::Option( 174 , 1, 10000);

	// 各GPU用のDNNモデル名と、そのGPU用のUCT探索のスレッド数と、そのGPUに一度に何個の局面をまとめて評価(推論)を行わせるのか。
	// GPUは最大で8個まで扱える。

	// RTX 3090で10bなら4、15bなら2で最適。
    o["UCT_Threads1"]                << USI::Option(2, 0, 256);
	for (int i = 2; i <= max_gpu ; ++i)
		o["UCT_Threads" + std::to_string(i)] << USI::Option(0, 0, 256);

#if defined(COREML)
	// Core MLでは、ONNXではなく独自形式のモデルが必要。
    o["DNN_Model1"]                  << USI::Option(R"(model.mlmodel)");
#else
    o["DNN_Model1"]                  << USI::Option(R"(model.onnx)");
#endif
	for (int i = 2; i <= max_gpu ; ++i)
		o["DNN_Model" + std::to_string(i)] << USI::Option("");

#if defined(TENSOR_RT) || defined(ORT_TRT)
	// 通常時の推奨128 , 検討の時は推奨256。
	o["DNN_Batch_Size1"]             << USI::Option(128, 1, 1024);
#elif defined(ONNXRUNTIME)
	// CPUを使っていることがあるので、default値、ちょっと少なめにしておく。
	o["DNN_Batch_Size1"]             << USI::Option(32, 1, 1024);
#elif defined(COREML)
	// M1チップで8程度でスループットが飽和する。
	o["DNN_Batch_Size1"]             << USI::Option(8, 1, 1024);
#endif
	for (int i = 2; i <= max_gpu ; ++i)
		o["DNN_Batch_Size" + std::to_string(i)] << USI::Option(0, 0, 1024);

    //(*this)["Const_Playout"]               = USIOption(0, 0, int_max);
	// →　Playout数固定。これはNodesLimitでできるので不要。

	// PV lineの即詰みを調べるスレッドの数と1局面当たりの最大探索ノード数。
	o["PV_Mate_Search_Threads"]     << USI::Option(1, 0, 256);
	o["PV_Mate_Search_Nodes"]       << USI::Option(500000, 0, UINT32_MAX);

	// → leaf nodeではdf-pnに変更。
	// 探索ノード数の上限値を設定する。0 : 呼び出さない。
	o["LeafDfpnNodesLimit"]			<< USI::Option(40, 0, 10000);
}

// "isready"コマンドに対する初回応答
void Search::init(){}

// "isready"コマンド時に毎回呼び出される。
void Search::clear()
{
	// エンジンオプションの反映

	// -----------------------
	//   定跡の読み込み
	// -----------------------

	searcher.book.read_book();

#if 0
	// オプション設定
	dfpn_min_search_millisecs = options["DfPn_Min_Search_Millisecs"];

	// →　ふかうら王では、rootのdf-pnは、node数を指定することにした。
#endif

	searcher.SetPvInterval((TimePoint)Options["PV_Interval"]);

	// ノードを再利用するかの設定。
	searcher.SetReuseSubtree(Options["ReuseSubtree"]);

	// 勝率を評価値に変換する時の定数を設定。
	searcher.SetEvalCoef((int)Options["Eval_Coef"]);

	// 投了値
	searcher.SetResignThreshold((int)Options["Resign_Threshold"]);

	// デバッグ用のメッセージ出力の有無。
	searcher.SetDebugMessage(Options["DebugMessage"]);

	// スレッド数と各GPUのbatchsizeをsearcherに設定する。

	std::vector<int> new_thread;
	std::vector<int> new_policy_value_batch_maxsize;

	for (int i = 1; i <= max_gpu; ++i)
	{
		// GPU_unlimited() なら、すべてUCT_Threads1, DNN_Batch_Size1を参照する。
		new_thread.emplace_back((int)Options["UCT_Threads" + std::to_string(i)]);
		new_policy_value_batch_maxsize.emplace_back((int)Options["DNN_Batch_Size" + std::to_string(i)]);
	}
	
	// 対応デバイス数を取得する
	int device_count = NN::get_device_count();

	std::vector<int> thread_nums;
	std::vector<int> policy_value_batch_maxsizes;
	for (int i = 0; i < max_gpu ; ++i)
	{
		// 対応デバイス数以上のデバイスIDのスレッド数は 0 として扱う(デバイスの無効化)
		thread_nums.push_back(i < device_count ? new_thread[i] : 0);
		policy_value_batch_maxsizes.push_back(new_policy_value_batch_maxsize[i]);
	}

	// ※　InitGPU()に先だってSetMateLimits()でのmate solverの初期化が必要。この呼出をInitGPU()のあとにしないこと！
	searcher.SetMateLimits((int)Options["MaxMovesToDraw"] , (u32)Options["RootMateSearchNodesLimit"] , (u32)Options["LeafDfpnNodesLimit"] /*Options["MateSearchPly"]*/);
	searcher.InitGPU(Eval::dlshogi::ModelPaths , thread_nums, policy_value_batch_maxsizes);

	// その他、dlshogiにはあるけど、サポートしないもの。

	// EvalDir　　　 →　dlshogiではサポートされていないが、やねうら王は、EvalDirにあるモデルファイルを読み込むようにする。

	auto& search_options = searcher.search_options;
	search_options.c_fpu_reduction      =                Options["C_fpu_reduction"     ] / 100.0f;
	search_options.c_fpu_reduction_root =                Options["C_fpu_reduction_root"] / 100.0f;

	search_options.c_init               =                Options["C_init"              ] / 100.0f;
	search_options.c_base               = (NodeCountType)Options["C_base"              ];
	search_options.c_init_root          =                Options["C_init_root"         ] / 100.0f;
	search_options.c_base_root          = (NodeCountType)Options["C_base_root"         ];

	// softmaxの時のボルツマン温度設定
	// これは、dlshogiの"Softmax_Temperature"の値。(174) = 1.74
	// ※ 100分率で指定する。
	// hcpe3から学習させたmodelの場合、1.40～1.50ぐらいにしないといけない。
	// cf. https://tadaoyamaoka.hatenablog.com/entry/2021/04/05/215431

	Eval::dlshogi::set_softmax_temperature(Options["Softmax_Temperature"] / 100.0f);

	searcher.SetDrawValue(
		(int)Options["DrawValueBlack"],
		(int)Options["DrawValueWhite"]);

	searcher.SetPonderingMode(Options["USI_Ponder"]);

	// UCT_NodeLimit : これはノード制限ではなく、ノード上限を示す。この値を超えたら思考を中断するが、
	// 　この値を超えていなくとも、持ち時間制御によって思考は中断する。
	// ※　探索ノード数を固定したい場合は、NodesLimitオプションを使うべし。
	searcher.InitializeUctSearch((NodeCountType)Options["UCT_NodeLimit"]);

	// PV lineの詰み探索の設定
	searcher.SetPvMateSearch(int(Options["PV_Mate_Search_Threads"]), int(Options["PV_Mate_Search_Nodes"]));


#if 0
	// dlshogiでは、
	// "isready"に対してnode limit = 1 , batch_size = 128 で探索しておく。
	// 初期局面に対してはわりと得か？

	// 初回探索をキャッシュ
	Position pos_tmp;
	StateInfo si;
	pos_tmp.set_hirate(&si,Threads.main());
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

#endif // defined(YANEURAOU_ENGINE_DEEP)
