#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "FukauraOuEngine.h"

#include "../../position.h"
#include "../../usi.h"
#include "../../thread.h"

#include "dlshogi_searcher.h"
#include "UctSearch.h"

#include "../../eval/deep/nn.h"
#include "../../eval/deep/nn_types.h"

using namespace YaneuraOu;

namespace dlshogi {


FukauraOuEngine::FukauraOuEngine() :
    searcher(*this){}

// NN関係のoptionを生やす。
void FukauraOuEngine::add_nn_options()
{
    OptionsMap& options = get_options();

    // 各GPU用のDNNモデル名と、そのGPU用のUCT探索のスレッド数と、そのGPUに一度に何個の局面をまとめて評価(推論)を行わせるのか。

    options.add("EvalDir", Option("eval", [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    return std::nullopt;
                }));

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

	// 全エンジン共通optionを生やす(ただしThreadsは不要)
    Engine::add_base_options();

	// 探索部で用いるoptionを生やす。
    searcher.add_options(options);

	// NN関係のoptionを生やす。
    add_nn_options();

    // 定跡関係のoptionを生やす
    searcher.book.add_options(options);
}

// "Max_GPU","Disabled_GPU"と"UCT_Threads"の設定値から、各GPUのスレッド数の設定を返す。
std::vector<int> FukauraOuEngine::get_thread_settings() {

    int option_max_gpu = int(options["Max_GPU"]);

    // GPUのデバイス数を取得する
    int device_count = Eval::dlshogi::NN::get_device_count();

	// 取得できなかった時は-1が返るので、その時はオプション設定に従う。
    if (device_count == -1)
        device_count = option_max_gpu;

    // GPUの最大数
    int max_gpu = std::min(option_max_gpu, device_count);

    // 各GPUのスレッド数
    int thread_num = int(options["UCT_Threads"]);

    // スレッド設定
    std::vector<int> thread_settings;

    // GPUの数だけスレッド数を設定
    thread_settings.assign(max_gpu, thread_num);

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
	// 📝 GPUの数に応じてthreadの確保を行うのでthreadの確保はこのタイミングで行われる。

	auto& options = get_options();

	// 各GPUのスレッド設定。無効化されているdeviceは0。
    auto thread_settings = get_thread_settings();

	// DNNのbatch sizeの設定。
    int dnn_batch_size = int(options["DNN_Batch_Size"]);

	// 評価関数モデルのPATH。
	auto eval_dir = options["EvalDir"];
    auto model_name = options["DNN_Model"];
    auto model_path = Path::Combine(eval_dir, model_name);

	searcher.InitGPU(model_path, thread_settings, dnn_batch_size);
}


// "isready"コマンド応答。
void FukauraOuEngine::isready() {

    // -----------------------
    // 評価関数テーブルの初期化(起動時でも良い)
    // -----------------------
    Eval::dlshogi::init();

    // -----------------------
    //   定跡の読み込み
    // -----------------------

    searcher.book.read_book();

    // -----------------------
    //   GPUの初期化
    // -----------------------

	init_gpu();

    // -----------------------
    //   探索部の初期化
    // -----------------------

	// 探索部の初期化
	searcher.InitializeUctSearch();

	// PV lineの詰み探索の設定
	searcher.SetPvMateSearch(int(options["PV_Mate_Search_Threads"]), int(options["PV_Mate_Search_Nodes"]));

	// 🤔 "isready"に対してnode limit = 1 , batch_size = 128 で探索したほうがいいかも。(dlshogiはそうなっている)

	// 基底classのisready()の呼び出し。
	Engine::isready();
}

// 🌈 "ponderhit"に対する処理。
void FukauraOuEngine::set_ponderhit(bool b) {

	// 📝 ponderhitしたので、やねうら王では、
    //     現在時刻をtimer classに保存しておく必要がある。
	// 💡 ponderフラグを変更する前にこちらを先に実行しないと
	//     ponderフラグを見てponderhitTimeを参照して間違った計算をしてしまう。
    searcher.search_limits.time_manager.ponderhitTime = now();

	searcher.search_limits.ponder           = b;
}

// エンジン作者名の変更
std::string FukauraOuEngine::get_engine_author() const { return "Tadao Yamaoka , yaneurao"; }


// 🌈 やねうら王フレームワークと、dlshogiの橋渡しを行うコード 🌈

FukauraOuWorker::FukauraOuWorker(OptionsMap&               options,
                                 ThreadPool&               threads,
                                 size_t                    threadIdx,
                                 NumaReplicatedAccessToken numaAccessToken,
                                 DlshogiSearcher&          searcher,
                                 FukauraOuEngine&          engine) :
    Worker(options, threads, threadIdx, numaAccessToken),
    searcher(searcher),
    engine(engine) {}

void FukauraOuWorker::pre_start_searching() {

	// 入玉ルールを反映させる必要がある。
    rootPos.set_ekr(searcher.search_options.enteringKingRule);

	if (is_mainthread())
        // 🌈 Stockfishでthread.cppにあった初期化の一部はSearchManager::pre_start_searching()に移動させた。
        searcher.search_limits.ponder = limits.ponderMode;
}


// "go"コマンドに対して呼び出される。
void FukauraOuWorker::start_searching()
{
    if (!is_mainthread())
    {
        parallel_search();
        return;
    }

    // 開始局面の手番をglobalに格納しておいたほうが便利。
    searcher.search_limits.root_color = rootPos.side_to_move();

    // "NodesLimit"の値など、今回の"go"コマンドによって決定した値が反映される。
    searcher.SetLimits(rootPos, limits);

    // "position"コマンドが送られずに"go"がきた。
    if (engine.game_root_sfen.empty())
        engine.game_root_sfen = StartSFEN;

    Move ponderMove;
    Move move = searcher.UctSearchGenmove(rootPos, engine.game_root_sfen,
                                          engine.moves_from_game_root, ponderMove);

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
    while (!threads.stop && (searcher.search_limits.ponder || limits.infinite))
    {
        //	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
        // (思考のためには計算資源を使っていないので。)
        Tools::sleep(1);

        // Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。
    }


    std::string bestmove = to_usi_string(move);
    std::string ponder;
    if (ponderMove)
        ponder = to_usi_string(ponderMove);

    engine.updateContext.onBestmove(bestmove, ponder);
}

void FukauraOuWorker::parallel_search()
{
	// searcherが、このスレッドがどのインスタンスの
	// UCTSearcher::ParallelUctSearch()を呼び出すかを知っている。

	// このrootPosはスレッドごとに用意されているからコピー可能。
	
	searcher.parallel_search(rootPos, threadIdx);
}

FukauraOuWorker::~FukauraOuWorker()
{
	searcher.FinalizeUctSearch();
}

#if 0

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
