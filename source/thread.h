#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
//#include <cstddef>
//#include <cstdint>
#include <mutex>
#include <vector>

#include "movepick.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "thread_win32_osx.h"
//#include "types.h"
#include "history.h"

#if defined(EVAL_LEARN)
// 学習用の実行ファイルでは、スレッドごとに置換表を持ちたい。
#include "tt.h"
#endif

namespace YaneuraOu {

// --------------------
// スレッドの属するNumaを管理する
// --------------------

// Sometimes we don't want to actually bind the threads, but the recipient still
// needs to think it runs on *some* NUMA node, such that it can access structures
// that rely on NUMA node knowledge. This class encapsulates this optional process
// such that the recipient does not need to know whether the binding happened or not.

// 時にはスレッドを実際にバインドしたくない場合もありますが、
// 受け手側は、それが 何らかの NUMAノード上で実行されていると認識する必要があります。
// これは、NUMAノードに関する情報を必要とする構造体にアクセスするためです。
// このクラスは、このバインドが行われたかどうかを受け手が知る必要がないように、
// このオプションのプロセスをカプセル化します。

class OptionalThreadToNumaNodeBinder {
public:
	OptionalThreadToNumaNodeBinder(NumaIndex n) :
		numaConfig(nullptr),
		numaId(n) {}

	OptionalThreadToNumaNodeBinder(const NumaConfig& cfg, NumaIndex n) :
		numaConfig(&cfg),
		numaId(n) {}

	/*
	 📝 OptionalThreadToNumaNodeBinder binder;
         に対してbinder() とやるとこのoperatorが呼び出されて、
	     NumaReplicatedAccessTokenがもらえる。
	 
	     そのあと (void) (networks[numaAccessToken]); とやると
	     (これは、Search::Worker::ensure_network_replicated()で行っている)
	     適切なNUMAに割り当てられるようになっている。
	 
	     これは、LazyNumaReplicatedのoperator[]でensure_present()が呼び出されて
	     NumaConfig.execute_on_numa_node()が呼び出されるからである。
	*/
	NumaReplicatedAccessToken operator()() const {
		if (numaConfig != nullptr)
			// このスレッドを適切なNUMAで実行できるようにOSレベルでbindする。
			return numaConfig->bind_current_thread_to_numa_node(numaId);
		else
			return NumaReplicatedAccessToken(numaId);
	}

private:
	const NumaConfig* numaConfig;
	NumaIndex         numaId;
};


// --------------------
// 探索時に用いるスレッド
// --------------------

// Abstraction of a thread. It contains a pointer to the worker and a native thread.
// After construction, the native thread is started with idle_loop()
// waiting for a signal to start searching.
// When the signal is received, the thread starts searching and when
// the search is finished, it goes back to idle_loop() waiting for a new signal.

// (探索用の)スレッドの抽象化です。これはワーカーへのポインタとネイティブスレッドを含みます。
// 構築後、ネイティブスレッドは idle_loop() で開始され、開始信号を待ちます。
// 信号を受け取るとスレッドは検索を開始し、検索が終了すると再び idle_loop() に戻り、新しい信号を待ちます。
// ⇨  探索時に用いる、それぞれのスレッド。これを探索用スレッド数だけ確保する。
//    ただしメインスレッドはこのclassを継承してMainThreadにして使う。

namespace Search {
	class Worker;
	typedef std::function<std::unique_ptr<Worker>(size_t /*thread_idx*/, NumaReplicatedAccessToken /*token*/)> WorkerFactory;
}

class Thread {
public:

	// thread_id : ThreadPoolで何番目のthreadであるか。この値は、idx(スレッドID)となる。
	Thread(
#if STOCKFISH
		Search::SharedState&,
		std::unique_ptr<Search::ISearchManager>,
#else
		// 📌 やねうら王では、SharedStateとISearchManagerを使わずに、Workerのfactoryを使ってWorkerを直接生成する。
		Search::WorkerFactory          factory,
#endif
		size_t                         thread_id,
		OptionalThreadToNumaNodeBinder binder
	);
	virtual ~Thread();

	// スレッド起動後、この関数が呼び出される。
	void idle_loop();

	// ------------------------------
	//      同期待ちのwait等
	// ------------------------------

	// workerに探索を開始させる。
	// 📝 "go"コマンドに対して呼び出される。
	void start_searching();

	// このクラスが保持している探索で必要なテーブル(historyなど)をクリアする。
	// 📝 "usinewgame"に対して呼び出される。
	void clear_worker();

	// jobを実行する。jobは引数fで渡す。
	void run_custom_job(std::function<void()> f);

	// 評価関数パラメーターが、このthreadが属するNUMAにも配置されているかを確かめて、
	// 配置されていなければ、評価関数パラメーターをコピーする。
	void ensure_network_replicated();

	// Thread has been slightly altered to allow running custom jobs, so
	// this name is no longer correct. However, this class (and ThreadPool)
	// require further work to make them properly generic while maintaining
	// appropriate specificity regarding search, from the point of view of an
	// outside user, so renaming of this function is left for whenever that happens.

	// スレッドはカスタムジョブを実行できるように少し変更されたため、
	// この名前（関数名）はもはや正確ではありません。
	// ただし、このクラス（および ThreadPool）は、外部の利用者から見て
	// 探索に関する適切な特異性を維持しつつ、真に汎用的にするための
	// さらなる作業が必要なため、この関数のリネームはその作業が
	// 実施される時まで保留されています。

	// start_searching()で開始した探索の終了を待機する。
	// 💡 searchingフラグがfalseになるのを待つ。

	void   wait_for_search_finished();

	// Threadの自身のスレッド番号を返す。0 origin。
	// コンストラクタで渡したthread_idが返ってくる。
	size_t id() const { return idx; }

	// 実行しているworker
	std::unique_ptr<Search::Worker> worker;

	// 実行しているjob
	std::function<void()>           jobFunc;

private:
	// exitフラグやsearchingフラグの状態を変更するときのmutex
	std::mutex                mutex;

	// idle_loop()で待機しているときに待つ対象
	std::condition_variable   cv;

	// thread id。main threadなら0。slaveなら1から順番に値が割当てられる。
	// nthreadsは、スレッド数。(options["Threads"]の値)
	size_t                    idx
#if STOCKFISH
          // 📌 nthreads使わないと思う。やねうら王ではコメントアウト
          , nthreads
#endif
		;

	// exit      : このフラグが立ったら終了する。
	// searching : 探索中であるかを表すフラグ。プログラムを簡素化するため、事前にtrueにしてある。
	bool                      exit = false, searching = true;  // Set before starting std::thread
															   // std::threadが始まる前にセットされる

	// stack領域を増やしたstd::thread
	// Workerは、このthreadに割り当てて実行する。
	NativeThread              stdThread;

	// このスレッドおよび評価関数パラメーターが、どのNUMAに属するか。
	NumaReplicatedAccessToken numaAccessToken;
};

// 思考で用いるスレッドの集合体

// ThreadPool struct handles all the threads-related stuff like init, starting,
// parking and, most importantly, launching a thread. All the access to threads
// is done through this class.

// ThreadPool構造体は、スレッドの初期化、起動、待機、
// そして最も重要なスレッドの実行など、スレッドに関するすべての処理を扱います。
// スレッドへのすべてのアクセスは、このクラスを通して行われます。

class ThreadPool {
public:
	ThreadPool() {}

	~ThreadPool() {
		// destroy any existing thread(s)
		// 存在するthreadを解体する

		if (threads.size() > 0)
		{
			main_thread()->wait_for_search_finished();

			threads.clear();
		}
	}

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&) = delete;

	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&&) = delete;

	// mainスレッドに思考を開始させる。
	void   start_thinking(const OptionsMap&, Position&, StateListPtr&, Search::LimitsType);

	// 指定したthreadIdのthreadで、jobとしてfを実行する。
	void   run_on_thread(size_t threadId, std::function<void()> f);

	// 指定したthreadIdのthreadが、jobを実行終わるのを待つ。
	void   wait_on_thread(size_t threadId);

	// このThreadPoolが保持しているthread数
	size_t num_threads() const;

	// set()で生成したスレッドの初期化
    // 💡 各ThreadのWorkerに対してclear()が呼び出される。
    void clear();

	// requested_threadsの数になるように、スレッド数を変更する。
    // 💡 各ThreadのWorkerに対してclear()が1度以上呼び出されることは保証されている。
    void set(const NumaConfig&            numaConfig,
#if STOCKFISH
             Search::SharedState,
             const Search::SearchManager::UpdateContext&);
#else
             const OptionsMap&            options,
             size_t                       requested_threads,
             const Search::WorkerFactory& worker_factory);
#endif
    /*
	   💡 Stockfishでは、
            Search::SharedState,
            const Search::SearchManager::UpdateContext&
         を渡しているが、やねうら王ではこれらを分離する。

		 また、Stockfishでは、options["Threads"]から生成するスレッド数を決めているが、
         やねうら王では、DL系でこのエンジンオプションからスレッド数をを決めたくないので
         ここに柔軟性を持たせる。
    
       📝 スレッド数を変更するということは、スレッド数が足りなければ、スレッドを生成しなければならない。
           スレッド(Thread class)は、その実行jobとしてWorker classの派生classを持っているので、
           スレッド生成のためにはWorkerの生成を行う能力が必要である。そのため、ここでは、WorkerFactoryを渡している。
    
       ⚠ このmethodはEngine::resize_threads()からのみ呼び出される。
    */

#if STOCKFISH
	Search::SearchManager* main_manager();
#else
	// 📝 やねうら王では、SearchManagerはEngine派生classが持つように変更した。
#endif

	// mainスレッドを取得する。これはthis[0]がそう。
    Thread* main_thread() const { return threads.front().get(); }

	// 今回、goコマンド以降に探索したノード数
    // →　これはPosition::do_move()を呼び出した回数。
    // ※　dlshogiエンジンで、探索ノード数が知りたい場合は、
    // 　dlshogi::nodes_visited()を呼び出すこと。
    uint64_t nodes_searched() const;

#if STOCKFISH
	// 💡 tablebaseにhitした回数。将棋では使わない。
	uint64_t               tb_hits() const;

	// ⚠ これは、やねうら王の標準探索エンジンでしか使わないので、
    //     YaneuraOuEngine側に移動させる。
	Thread*  get_best_thread() const;
#endif

	// メイン以外のすべてのスレッドのstart_searching()を呼び出す。(並列探索の開始)
    // 💡 このmethodはmainスレッドで呼び出す。
    void start_searching();

	// 探索の終了(すべてのスレッドの終了)を待つ
    // start_searching()で開始したすべてのスレッドの終了を待つ。
    void wait_for_search_finished() const;


	std::vector<size_t> get_bound_thread_count_by_numa_node() const;

	// 評価関数パラメーターがいま実行しているNUMAに配置されているようにする。
	void ensure_network_replicated();

	// stop          : 探索の停止フラグ
	// abortedSearch : 探索自体を破棄するためのフラグ
	//		           🤔 このフラグ、必要なのか？
	// increaseDepth : aspiration searchでdepthが増えていっているかのフラグ
	//                 🤔 このフラグはSearchManagerに移動

	std::atomic_bool stop, abortedSearch
#if STOCKFISH
		, increaseDepth
#endif
		;

	auto cbegin() const noexcept { return threads.cbegin(); }
	auto begin() noexcept { return threads.begin(); }
	auto end() noexcept { return threads.end(); }
	auto cend() const noexcept { return threads.cend(); }
	auto size() const noexcept { return threads.size(); }
	auto empty() const noexcept { return threads.empty(); }

	// 抱えているすべてのThread
	// 💡 Stockfishではprivate memberなのだが、
	//     やねうら王ではEngine派生class側からアクセスしたいことがあるのでpublicに変更。
	std::vector<std::unique_ptr<Thread>> threads;

private:
	// 現局面までのStateInfoのlist
	StateListPtr                         setupStates;

	// 各threadに対応するNUMAのindex
	std::vector<NumaIndex>               boundThreadToNumaNode;

	// Threadクラスの特定のメンバー変数を足し合わせたものを返す。
	// 💡 nodesの集計に用いる。
	uint64_t accumulate(std::atomic<uint64_t> Search::Worker::* member) const {

		uint64_t sum = 0;
		for (auto&& th : threads)
			sum += (th->worker.get()->*member).load(std::memory_order_relaxed);
		return sum;
	}
};

} // namespace YaneuraOu

#endif // #ifndef THREAD_H_INCLUDED
