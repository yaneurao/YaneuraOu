#ifndef __UCTSEARCH_H_INCLUDED__
#define __UCTSEARCH_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)
#include "../../position.h"
#include "../../mate/mate.h"

#include "Node.h"
#include "PvMateSearch.h"

// この探索部は、NN専用なので直接読み込む。

#include "../../eval/deep/nn_types.h"
#include "../../eval/deep/nn.h"

using namespace Eval::dlshogi;

namespace dlshogi
{
	class UctSearcher;
	class DlshogiSearcher;
	struct SearchOptions;

	// UctSearcher(探索用スレッド)をGPU一つ利用する分ずつひとまとめにしたもの。
	// 一つのGPUにつき、UctSearchThreadGroupひとつが対応する。
	class UctSearcherGroup
	{
	public:
		UctSearcherGroup() :  threads(0) , gpu_id(-1) , policy_value_batch_maxsize(0){}

		// 初期化
		// "isready"に対して呼び出される。
		// スレッド生成は、やねうら王フレームワーク側で行う。
		//   model_path                 : 読み込むmodel path
		//   new_thread                 : このインスタンスが確保するUctSearcherの数
		//   gpu_id                     : このインスタンスに紐付けられているGPU ID
		//   policy_value_batch_maxsize : このインスタンスが生成したスレッドがNNのforward()を呼び出す時のbatchsize
		void Initialize(const std::string& model_path , const int new_thread, const int gpu_id, const int policy_value_batch_maxsize);

		// ニューラルネットのforward() (順方向の伝播 = 推論)を呼び出す。
		void nn_forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
		{
#if !defined(UNPACK_CUDA)
			// 入力特徴量を展開する。GPU側で展開する場合は不要。
			extract_input_features(batch_size, p1, p2, x1, x2);
#endif
			mutex_gpu.lock();
			nn->forward(batch_size, p1, p2, x1, x2, y1, y2);
			mutex_gpu.unlock();
		}

		// 各探索スレッドは探索開始時に(nn_forward()の呼び出しまでに)、この関数を呼び出してスレッドとGPUとを紐付けないといけない。
		void set_device() { nn->set_device(gpu_id); }

		// やねうら王では、スレッドの生成～解体はThreadクラスが行うので、これらはコメントアウト。

		//	void Run();
		//	void Join();
		//#ifdef THREAD_POOL
		//	void Term();
		//#endif

		// -- やねうら王独自拡張

		// GPU側のメモリ確保を抽象化。

		// forward()に渡す用のメモリを確保して返す。
		//   T型のnum要素からなる配列を確保して返す。
		template <class T>
		T* gpu_memalloc(size_t num) { return (T*)nn->alloc(sizeof(T) * num ); }

		template <class T>
		void gpu_memfree(T* t) { nn->free((void*)t); }

		// dlshogiではglobalだった変数へのアクセスはこれを通じて行う。
		// set_dlsearcher()をコンストラクタ呼び出し直後に行うものとする。
		DlshogiSearcher* get_dlsearcher() const { return dlshogi_searcher; }
		void set_dlsearcher(DlshogiSearcher* ds) { dlshogi_searcher = ds; }

		// 保持しているn番目のUctSearcherを返す。
		UctSearcher* get_uct_searcher(int n) { return &searchers[n]; }

	private:

		// dlshogiではglobalだった変数
		DlshogiSearcher* dlshogi_searcher;

		// このインスタンスが関連付いているGPUのID
		// Initialize()で引数として渡される。
		int gpu_id;

		// このインスタンスが確保するスレッド数
		// Initialize()で引数として渡される。
		size_t threads;

		// UCTSearcher
		// このインスタンスはthreadsの数だけ作成されている。
		std::vector<UctSearcher> searchers;

		// ニューラルネット
		std::shared_ptr<Eval::dlshogi::NN> nn;

		// このインスタンスが生成したスレッドがNNのforward()を呼び出す時のbatchsize
		// Initialize()で引数として渡される。
		int policy_value_batch_maxsize;

		// ↑のnnにアクセスする時のmutex
		std::mutex mutex_gpu;

		// --- やねうら王独自拡張

		// nnが保持しているモデルのpath。
		// 異なるモデルになった時に前のものを開放して確保しなおす。
		std::string model_path;
	};

	// leaf nodeまでに辿ったNodeを記録しておく構造体。
	// ※　dlshogiではtrajectory_t
	struct NodeTrajectory {
		Node*			node;   // Node
		ChildNumType	index;  // そのnodeで何番目のchild[index]を選択したのか

		NodeTrajectory(Node* node, ChildNumType index) : node(node), index(index) {}
	};

	// NodeTrajectoryのvector
	// ※　dlshogiではtrajectories_t
	typedef std::vector<NodeTrajectory> NodeTrajectories;

	// 訪問したNodeに対してEvalNodeが完了した時に辿るための構造体。
	// ※　dlshogiではvisitor_t
	struct NodeVisitor {
		// 訪問してきたNode
		NodeTrajectories trajectories;

		// leaf nodeでのvalue_win。(これを辿ってきたNodeに対して符号を反転させながら伝播させていく)
		// Eval()したときに、NNから返ってきたこの局面のvalue(期待勝率)の値。
		// ただし詰み探索などで子ノードから伝播した場合、以下の定数をとることがある。
		// ・このノードで勝ちなら      VALUE_WIN   // 子ノードで一つでもVALUE_LOSEがあればその指し手を選択するので       VALUE_WIN
		// ・このノードで負けなら      VALUE_LOSE  // 子ノードがすべてVALUE_WINであればどうやってもこの局面では負けなのでVALUE_LOSE
		// ・このノードで引き分けなら、VALUE_DRAW
		// 備考) RepetitionWin (連続王手の千日手による反則勝ち) , RepetitionSuperior(優等局面)の場合も、VALUE_WINに含まれる。
		//       RepetitionLose(連続王手の千日手による反則負け) , RepetitionSuperior(劣等局面)の場合も、VALUE_LOSEに含まれる。
		// この変数は、UctSearcher::SelectMaxUcbChild()を呼び出した時に、子ノードを調べて、その結果が代入される。
		float value_win;
	};

	// バッチの要素
	// EvalNode()ごとにどのNodeとColorから呼び出されたのかを記録しておく構造体。
	// NNから返し値がもらえた時に、ここに記録されているNodeについて、その情報を更新する。
	struct BatchElement {
		Node*	node;     // どのNodeに対するEvalNode()なのか。
		Color	color;    // その時の手番

		// 通常の探索では、このポインターはNodeVisitor::value_win を指している。
		float* value_win; // leaf nodeでのvalue_winの値(これを辿ってきたNodeに対して符号を反転させながら伝播させていく)
	};

	// UCT探索を行う、それぞれのスレッドを表現する。
	// UctSearcherGroupは、このインスタンスを集めたもの。1つのGPUに対してUctSearcherGroupのインスタンスが1つ割り当たる。
	class UctSearcher
	{
	public:
		UctSearcher(UctSearcherGroup* grp, const int thread_id, const int policy_value_batch_maxsize) :
			grp(grp),
			thread_id(thread_id),
			// やねうら王では、スレッドはこのクラスが保有しないので、スレッドhandle不要。
			//		handle(nullptr),
			//#ifdef THREAD_POOL
			//		ready_th(true),
			//		term_th(false),
			//#endif
			policy_value_batch_maxsize(policy_value_batch_maxsize),
			// df-pn mate solverをleaf nodeで使う。
			mate_solver(Mate::Dfpn::DfpnSolverType::Node16bitOrdering)
		{
			// 推論(NN::forward())のためのメモリを動的に確保する。
			// GPUを利用する場合は、GPU側のメモリを確保しなければならないので、alloc()は抽象化されている。

			packed_features1 = grp->gpu_memalloc<PType>((policy_value_batch_maxsize * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB) + 7) >> 3);
			packed_features2 = grp->gpu_memalloc<PType>((policy_value_batch_maxsize * ((int)MAX_FEATURES2_NUM) + 7) >> 3);
			features1 = grp->gpu_memalloc<NN_Input1       >(policy_value_batch_maxsize);
			features2 = grp->gpu_memalloc<NN_Input2       >(policy_value_batch_maxsize);
			y1        = grp->gpu_memalloc<NN_Output_Policy>(policy_value_batch_maxsize);
			y2        = grp->gpu_memalloc<NN_Output_Value >(policy_value_batch_maxsize);

			policy_value_batch = new BatchElement[policy_value_batch_maxsize];

	#ifdef MAKE_BOOK
			policy_value_book_key = new Key[policy_value_batch_maxsize];
	#endif
		}

		// move counstructor
		UctSearcher(UctSearcher&& o) :
			grp(o.grp),
			thread_id(o.thread_id),
			mt(std::move(o.mt)),
			packed_features1(o.packed_features1),packed_features2(o.packed_features2),
			features1(o.features1),features2(o.features2),y1(o.y1),y2(o.y2),
			mate_solver(std::move(o.mate_solver))
		{
			o.packed_features1 = nullptr;
			o.packed_features2 = nullptr;
			o.features1 = nullptr;
			o.features2 = nullptr;
			o.y1 = nullptr;
			o.y2 = nullptr;
			policy_value_batch = nullptr;
		}

		~UctSearcher() {
			if (features1) // move counstructorによって解体後でないことをチェック
			{
				grp->gpu_memfree<PType           >(packed_features1);
				grp->gpu_memfree<PType           >(packed_features2);
				grp->gpu_memfree<NN_Input1       >(features1);
				grp->gpu_memfree<NN_Input2       >(features2);
				grp->gpu_memfree<NN_Output_Policy>(y1);
				grp->gpu_memfree<NN_Output_Value >(y2);

				delete[] policy_value_batch;
			}
		 }

		// -- やねうら王ではこのクラスはスレッド生成～解体に関与しない。

		//void Run();
		//void Join();
		//void Term();

		// UCTアルゴリズムによる並列探索の各スレッドのEntry Point
		// ※　Thread::search()から呼び出す。
		void ParallelUctSearchStart(const Position& rootPos);

		// leaf node用の詰め将棋ルーチンの初期化(alloc)を行う。
		// ※　SetLimits()が"go"に対してしか呼び出されていないからmax_moves_to_drawは未確定なのでここで設定するわけにはいかない。
		void InitMateSearcher(const SearchOptions& options);

		// "go"に対して探索を開始する時に呼び出す。
		// "go"に対してしかmax_moves_to_drawは未確定なので、それが確定してから呼び出す。
		void SetMateSearcher(const SearchOptions& options);

		// 初期化時に呼び出す。
		// policy_value_batch_maxsize と同数のダミーデータを作成し、推論を行う。
		void DummyForward();

	private:
		//  並列処理で呼び出す関数
		//  UCTアルゴリズムを反復する
		void ParallelUctSearch(const Position& rootPos);

		// UCT探索を行う関数
		// 1回の呼び出しにつき, 1プレイアウトする。
		// (leaf nodeで呼び出すものとする)
		//   pos          : UCT探索を行う開始局面
		//   current      : UCT探索を行う開始局面
		//   visitor      : 探索開始局面(tree.GetCurrentHead())から、currentに至る手順。あるNodeで何番目のchildを選択したかという情報。
		//
		// 返し値 : currentの局面の期待勝率を返すが、以下の特殊な定数を取ることがある。
		//   QUEUING      : 評価関数を呼び出した。(呼び出しはqueuingされていて、完了はしていない)
		//   DISCARDED    : 他のスレッドがすでにこのnodeの評価関数の呼び出しをしたあとであったので、何もせずにリターンしたことを示す。
		//
		float UctSearch(Position* pos, ChildNode* parent, Node* current, NodeVisitor& visitor);

		//  UCBが最大となる子ノードのインデックスを返す関数
		//    pos     : 調べたい局面
		//    current : 調べたい局面
		//  current->value_winに、子ノードを調べた結果が代入される。
		ChildNumType SelectMaxUcbChild(ChildNode* parent, Node* current);

		// Evaluateを呼び出すリスト(queue)に追加する。
		void QueuingNode(const Position* pos, Node* node, float* value_win);

		// ノードを評価
		void EvalNode();

		// 自分の所属するグループ
		UctSearcherGroup* grp;

		// スレッド識別番号(これは、プロセス全体でunique)
		// log出力の時や乱数のseedのために使う。
		int thread_id;

		// 乱数生成器
		PRNG mt;

		//// スレッドのハンドル
		//thread *handle;
		//#ifdef THREAD_POOL
		//	// スレッドプール用
		//	std::mutex mtx_th;
		//	std::condition_variable cond_th;
		//	bool ready_th;
		//	bool term_th;
		//#endif

		// コンストラクタで渡された、このスレッドが扱う、NNへのbatchの個数。
		int policy_value_batch_maxsize;

		// これは、policy_value_batch_maxsize分、事前に確保されている。
		Eval::dlshogi::PType* packed_features1;
		Eval::dlshogi::PType* packed_features2;
		Eval::dlshogi::NN_Input1* features1;
		Eval::dlshogi::NN_Input2* features2;

		Eval::dlshogi::NN_Output_Policy* y1;
		Eval::dlshogi::NN_Output_Value * y2;

		// EvalNode()ごとにどのNodeとColorから呼び出されたのかを記録しておく配列
		// NNから返し値がもらえた時に、ここに記録されているNodeについて、その情報を更新する。
		BatchElement* policy_value_batch;

	#ifdef MAKE_BOOK
		Key* policy_value_book_key;
	#endif

		// features1[],features2[],policy_value_batch[],policy_value_book_key[],の次に使用するindexを示している。
		// batch分溜まったら、まとめてGPUに投げてEvalする。
		int current_policy_value_batch_index;

		// NodeTreeを取得
		NodeTree* get_node_tree() const;

		// 奇数手詰め用のsolver
		// Mate::MateSolver mate_solver;

		// leaf node用のdf-pn solver
		Mate::Dfpn::MateDfpnSolver mate_solver;
	};

	// 訪問回数が最大の子ノードを選択
	extern unsigned int select_max_child_node(const Node* uct_node);

}

#endif

#endif // ndef __UCTSEARCH_H_INCLUDED__
