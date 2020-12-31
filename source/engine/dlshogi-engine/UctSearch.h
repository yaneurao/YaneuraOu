#ifndef __UCTSEARCH_H_INCLUDED__
#define __UCTSEARCH_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)
#include "../../position.h"
#include "../../mate/mate.h"

#include "Node.h"

// この探索部は、NN専用なので直接読み込む。

#include "../../eval/deep/nn_types.h"
#include "../../eval/deep/nn.h"

using namespace Eval::dlshogi;

namespace dlshogi
{
	class UctSearcher;
	class DlshogiSearcher;

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
		void nn_forward(const int batch_size, Eval::dlshogi::NN_Input1* x1, Eval::dlshogi::NN_Input2* x2, Eval::dlshogi::NN_Output_Policy* y1, Eval::dlshogi::NN_Output_Value* y2)
		{
			mutex_gpu.lock();
			nn->forward(batch_size, x1, x2, y1, y2);
			mutex_gpu.unlock();
		}

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


	// バッチの要素
	// EvalNode()ごとにどのNodeとColorから呼び出されたのかを記録しておく構造体。
	// NNから返し値がもらえた時に、ここに記録されているNodeについて、その情報を更新する。
	struct BatchElement {
		Node*	node;  // どのNodeに対するEvalNode()なのか。
		Color	color; // その時の手番
		//u64     posKey; // その時のPosition::key()
	};

	// leaf nodeまでに辿ったNodeを記録しておく構造体。
	struct NodeTrajectory {
		Node*	node;   // Node
		u16		index;  // そのnodeで何番目のchild[index]を選択したのか

		NodeTrajectory(Node* node, u16 index) : node(node), index(index) {}
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
			policy_value_batch_maxsize(policy_value_batch_maxsize)
		{
			// 推論(NN::forward())のためのメモリを動的に確保する。
			// GPUを利用する場合は、GPU側のメモリを確保しなければならないので、alloc()は抽象化されている。

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
			features1(o.features1),features2(o.features2),y1(o.y1),y2(o.y2)
		{
			o.features1 = nullptr;
			o.features2 = nullptr;
			o.y1 = nullptr;
			o.y2 = nullptr;
			policy_value_batch = nullptr;
		}

		~UctSearcher() { 
			if (features1) // move counstructorによって解体後でないことをチェック
			{
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

		//  並列処理で呼び出す関数
		//  UCTアルゴリズムを反復する
		// ※　Thread::search()から呼び出す。
		void ParallelUctSearch(const Position& rootPos);

	private:

		//  UCT探索(1回の呼び出しにつき, 1回の探索)
		float UctSearch(Position* pos, Node* current, const int depth, std::vector<NodeTrajectory>& trajectories);

		// UCB値が最大の子ノードを返す
		int SelectMaxUcbChild(const Position* pos, Node* current, const int depth);

		// ノードをキューに追加
		void QueuingNode(const Position* pos, Node* node);

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
		Mate::MateSolver mate_solver;
	};


}

#endif

#endif // ndef __UCTSEARCH_H_INCLUDED__
