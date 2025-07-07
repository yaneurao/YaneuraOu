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

	NumaReplicatedAccessToken operator()() const {
		if (numaConfig != nullptr)
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

class Thread {
public:

	// thread_id : ThreadPoolで何番目のthreadであるか。この値は、idx(スレッドID)となる。
	Thread(Search::SharedState&,
		std::unique_ptr<Search::ISearchManager>,
		size_t thread_id,
		OptionalThreadToNumaNodeBinder);
	virtual ~Thread();

	// スレッド起動後、この関数が呼び出される。
	void idle_loop();

	// ------------------------------
	//      同期待ちのwait等
	// ------------------------------

	// workerを開始させるときに呼び出す。
	void start_searching();

	// このクラスが保持している探索で必要なテーブル(historyなど)をクリアする。
	void clear_worker();

	void run_custom_job(std::function<void()> f);

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

	// 💡 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)

	void   wait_for_search_finished();

	// Threadの自身のスレッド番号を返す。0 origin。
	// コンストラクタで渡したthread_idが返ってくる。
	size_t id() const { return idx; }

	std::unique_ptr<Search::Worker> worker;
	std::function<void()>           jobFunc;

private:
	// exitフラグやsearchingフラグの状態を変更するときのmutex
	std::mutex                mutex;

	// idle_loop()で待機しているときに待つ対象
	std::condition_variable   cv;

	// thread id。main threadなら0。slaveなら1から順番に値が割当てられる。
	// nthreadsは、スレッド数。(options["Threads"]の値)
	size_t                    idx, nthreads;

	// exit      : このフラグが立ったら終了する。
	// searching : 探索中であるかを表すフラグ。プログラムを簡素化するため、事前にtrueにしてある。
	bool                      exit = false, searching = true;  // Set before starting std::thread

	// stack領域を増やしたstd::thread
	NativeThread              stdThread;

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

	void   run_on_thread(size_t threadId, std::function<void()> f);
	void   wait_on_thread(size_t threadId);
	size_t num_threads() const;

	// set()で生成したスレッドの初期化
	void   clear();

	// スレッド数を変更する。
	void   set(const NumaConfig& numaConfig,
		Search::SharedState,
		const Search::SearchManager::UpdateContext&);

	Search::SearchManager* main_manager();

	// mainスレッドを取得する。これはthis[0]がそう。
	Thread* main_thread() const { return threads.front().get(); }

	// 今回、goコマンド以降に探索したノード数
	// →　これはPosition::do_move()を呼び出した回数。
	// ※　dlshogiエンジンで、探索ノード数が知りたい場合は、
	// 　dlshogi::nodes_visited()を呼び出すこと。
	uint64_t               nodes_searched() const;

	// tablebaseにhitした回数。将棋では使わない。
	//uint64_t               tb_hits() const;

	Thread* get_best_thread() const;

	// メイン以外のすべてのスレッドのstart_searching()を呼び出す。(並列探索の開始)
	void                   start_searching();
	void                   wait_for_search_finished() const;

	std::vector<size_t> get_bound_thread_count_by_numa_node() const;

	void ensure_network_replicated();

	std::atomic_bool stop, abortedSearch, increaseDepth;

	auto cbegin() const noexcept { return threads.cbegin(); }
	auto begin() noexcept { return threads.begin(); }
	auto end() noexcept { return threads.end(); }
	auto cend() const noexcept { return threads.cend(); }
	auto size() const noexcept { return threads.size(); }
	auto empty() const noexcept { return threads.empty(); }

private:

	// 現局面までのStateInfoのlist
	StateListPtr                         setupStates;

	// vector<Thread*>からこのclassを継承させるのはやめて、このメンバーとして持たせるようにした。
	std::vector<std::unique_ptr<Thread>> threads;
	std::vector<NumaIndex>               boundThreadToNumaNode;

	// Threadクラスの特定のメンバー変数を足し合わせたものを返す。
	uint64_t accumulate(std::atomic<uint64_t> Search::Worker::* member) const {

		uint64_t sum = 0;
		for (auto&& th : threads)
			sum += (th->worker.get()->*member).load(std::memory_order_relaxed);
		return sum;
	}
};

} // namespace YaneuraOu

#endif // #ifndef THREAD_H_INCLUDED
