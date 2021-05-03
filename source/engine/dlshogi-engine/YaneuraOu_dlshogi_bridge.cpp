#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"
#include "../../usi.h"
#include "../../thread.h"

#include "dlshogi_types.h"
#include "Node.h"
#include "UctSearch.h"
#include "dlshogi_searcher.h"

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
    (*this)["Time_Margin"]                 = USIOption(1000, 0, INT_MAX);
    (*this)["Mate_Root_Search"]            = USIOption(29, 0, 35);
    (*this)["DfPn_Hash"]                   = USIOption(2048, 64, 4096); // DfPnハッシュサイズ
    (*this)["DfPn_Min_Search_Millisecs"]   = USIOption(300, 0, INT_MAX);
#endif

#ifdef MAKE_BOOK
	// 定跡を生成するときはPV出力は抑制したほうが良さげ。
    o["PV_Interval"]                 << USI::Option(0, 0, INT_MAX);
    o["Save_Book_Interval"]          << USI::Option(100, 0, INT_MAX);
#else
    o["PV_Interval"]                 << USI::Option(500, 0, INT_MAX);
#endif // !MAKE_BOOK

	o["UCT_NodeLimit"]				 << USI::Option(10000000, 100000, 1000000000); // UCTノードの上限
																				   // デバッグ用のメッセージ出力の有無
	o["DebugMessage"]                << USI::Option(false);

	// ノードを再利用するか。
    o["ReuseSubtree"]                << USI::Option(true);

	// 投了値 : 1000分率で
	o["Resign_Threshold"]            << USI::Option(0, 0, 1000);

	// 引き分けの時の値 : 1000分率で
	// 引き分けの局面では、この値とみなす。
	// root color(探索開始局面の手番)に応じて、2通り。
	
	o["DrawValueBlack"]            << USI::Option(500, 0, 1000);
	o["DrawValueWhite"]            << USI::Option(500, 0, 1000);

	// --- PUCTの時の定数
	// これ、探索パラメーターの一種と考えられるから、最適な値を事前にチューニングして設定するように
	// しておき、ユーザーからは触れない(触らなくても良い)ようにしておく。
	// →　dlshogiはoptimizerで最適化するために外だししているようだ。

#if 0
	// fpu_reductionの値を100分率で設定。
	// c_fpu_reduction_rootは、rootでのfpu_reductionの値。
    o["C_fpu_reduction"]             << USI::Option(27, 0, 100);
    o["C_fpu_reduction_root"]        << USI::Option(0, 0, 100);

    o["C_init"]                      << USI::Option(144, 0, 500);
    o["C_base"]                      << USI::Option(28288, 10000, 100000);
    o["C_init_root"]                 << USI::Option(116, 0, 500);
    o["C_base_root"]                 << USI::Option(25617, 10000, 100000);

#endif
	// 探索のSoftmaxの温度
	o["Softmax_Temperature"]		 << USI::Option( 1740 /* 方策分布を学習させた場合、1400から1500ぐらいが最適値らしいが… */ , 1, 5000);

	// 各GPU用のDNNモデル名と、そのGPU用のUCT探索のスレッド数と、そのGPUに一度に何個の局面をまとめて評価(推論)を行わせるのか。
	// GPUは最大で8個まで扱える。

    o["UCT_Threads1"]                << USI::Option(4, 0, 256);
    o["UCT_Threads2"]                << USI::Option(0, 0, 256);
    o["UCT_Threads3"]                << USI::Option(0, 0, 256);
    o["UCT_Threads4"]                << USI::Option(0, 0, 256);
    o["UCT_Threads5"]                << USI::Option(0, 0, 256);
    o["UCT_Threads6"]                << USI::Option(0, 0, 256);
    o["UCT_Threads7"]                << USI::Option(0, 0, 256);
    o["UCT_Threads8"]                << USI::Option(0, 0, 256);
    o["DNN_Model1"]                  << USI::Option(R"(model.onnx)");
    o["DNN_Model2"]                  << USI::Option("");
    o["DNN_Model3"]                  << USI::Option("");
    o["DNN_Model4"]                  << USI::Option("");
    o["DNN_Model5"]                  << USI::Option("");
    o["DNN_Model6"]                  << USI::Option("");
    o["DNN_Model7"]                  << USI::Option("");
    o["DNN_Model8"]                  << USI::Option("");

#if defined(ONNXRUNTIME)
	// CPUを使っていることがあるので、default値、ちょっと少なめにしておく。
	o["DNN_Batch_Size1"]             << USI::Option(32, 1, 1024);
#elif defined(TENSOR_RT)
	// 通常時の推奨128 , 検討の時は推奨256。
	o["DNN_Batch_Size1"]             << USI::Option(128, 1, 1024);
#endif
	o["DNN_Batch_Size2"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size3"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size4"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size5"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size6"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size7"]             << USI::Option(0, 0, 65536);
    o["DNN_Batch_Size8"]             << USI::Option(0, 0, 65536);

#if defined(ORT_MKL)
	// nn_onnx_runtime.cpp の NNOnnxRuntime::load() で使用するオプション。 
	// グラフ全体のスレッド数?（default値1）ORT_MKLでは効果が無いかもしれない。
	o["InterOpNumThreads"]           << USI::Option(1, 1, 65536);
	// ノード内の実行並列化の際のスレッド数設定（default値4、NNUE等でのThreads相当）
	o["IntraOpNumThreads"]           << USI::Option(4, 1, 65536);
#endif

    //(*this)["Const_Playout"]               = USIOption(0, 0, INT_MAX);
	// →　Playout数固定。これはNodeLimitでできるので不要。

	// → leaf nodeではdf-pnに変更。
	// 探索ノード数の上限値を設定する。0 : 呼び出さない。
	o["LeafDfpnNodesLimit"]			<< USI::Option(40, 0, 10000);

	// root nodeでのdf-pn詰将棋探索の最大ノード数
	o["RootMateSearchNodesLimit"]	<< USI::Option(1000000, 0, UINT32_MAX);
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

	searcher.SetGetnerateAllLegalMoves(Options["GenerateAllLegalMoves"]);

	// ノードを再利用するかの設定。
	searcher.SetReuseSubtree(Options["ReuseSubtree"]);

	// 投了値
	searcher.SetResignThreshold((int)Options["Resign_Threshold"]);

	// デバッグ用のメッセージ出力の有無。
	searcher.SetDebugMessage(Options["DebugMessage"]);

	// スレッド数と各GPUのbatchsizeをsearcherに設定する。

	const int new_thread[max_gpu] = {
		(int)Options["UCT_Threads1"], (int)Options["UCT_Threads2"], (int)Options["UCT_Threads3"], (int)Options["UCT_Threads4"],
		(int)Options["UCT_Threads5"], (int)Options["UCT_Threads6"], (int)Options["UCT_Threads7"], (int)Options["UCT_Threads8"]
	};
	const int new_policy_value_batch_maxsize[max_gpu] = {
		(int)Options["DNN_Batch_Size1"], (int)Options["DNN_Batch_Size2"], (int)Options["DNN_Batch_Size3"], (int)Options["DNN_Batch_Size4"],
		(int)Options["DNN_Batch_Size5"], (int)Options["DNN_Batch_Size6"], (int)Options["DNN_Batch_Size7"], (int)Options["DNN_Batch_Size8"]
	};

	// 対応デバイス数を取得する
	int device_count = NN::get_device_count();

	std::vector<int> thread_nums;
	std::vector<int> policy_value_batch_maxsizes;
	for (int i = 0; i < max_gpu; ++i)
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

	// 以下も、探索パラメーターだから、いらない。開発側が最適値にチューニングすべきという考え。
	// C_fpu_reduction , C_fpu_reduction_root , C_init , C_base , C_init_root , C_base_root
#if 0
	search_options.c_fpu_reduction      =                Options["C_fpu_reduction"     ] / 100.0f;
	search_options.c_fpu_reduction_root =                Options["C_fpu_reduction_root"] / 100.0f;

	search_options.c_init               =                Options["C_init"              ] / 100.0f;
	search_options.c_base               = (NodeCountType)Options["C_base"              ];
	search_options.c_init_root          =                Options["C_init_root"         ] / 100.0f;
	search_options.c_base_root          = (NodeCountType)Options["C_base_root"         ];

#endif

	// softmaxの時のボルツマン温度設定
	// これは、dlshogiの"Softmax_Temperature"の値。(1740) = 1.740
	// ※ dlshogiは100分率で指定する。ふかうら王では1000分率で指定する。
	// hcpe3から学習させたmodelの場合、1.40～1.50ぐらいにしないといけない。
	// cf. https://tadaoyamaoka.hatenablog.com/entry/2021/04/05/215431

	Eval::dlshogi::set_softmax_temperature(Options["Softmax_Temperature"] / 1000.0f);

	searcher.SetDrawValue(
		(int)Options["DrawValueBlack"],
		(int)Options["DrawValueWhite"]);

	searcher.SetPonderingMode(Options["USI_Ponder"]);

	// UCT_NodeLimit : これはノード制限ではなく、ノード上限を示す。この値を超えたら思考を中断するが、
	// 　この値を超えていなくとも、持ち時間制御によって思考は中断する。
	searcher.InitializeUctSearch((NodeCountType)Options["UCT_NodeLimit"]);

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


#endif // defined(YANEURAOU_ENGINE_DEEP)
