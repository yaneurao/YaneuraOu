#include "thread.h"

#include <algorithm> // std::count
#include <cassert>
#include <cmath>     // std::abs
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "bitboard.h"
#include "history.h"
#include "memory.h"
#include "movegen.h"
#include "search.h"
#include "timeman.h"
#include "types.h"
#include "usi.h"
#include "usioption.h"

namespace YaneuraOu {

// Constructor launches the thread and waits until it goes to sleep
// in idle_loop(). Note that 'searching' and 'exit' should be already set.

// コンストラクタはスレッドを起動し、スレッドが idle_loop() 内で
// スリープ状態に入るまで待機します。
// 'searching' および 'exit' は、すでに設定されている必要がある点に注意してください。

Thread::Thread(Search::SharedState& sharedState,
               //std::unique_ptr<Search::SearchManager> sm,
               Search::WorkerFactory          worker_factory,
               usize                         n,               // threadIdx
               usize                         numaN,           // numaThreadIdx
               usize                         totalNumaCount,  // numaTotal
               OptionalThreadToNumaNodeBinder binder) :
    idx(n),
    idxInNuma(numaN),
    totalNuma(totalNumaCount),
    stdThread(
      create_native_thread(NativeThreadOptions{}.setLargeStack(true), &Thread::idle_loop, this)) {

    if (!stdThread.joinable())
    {
        std::cerr << "Failed to create search thread\n";
        std::exit(EXIT_FAILURE);
    }

#if !defined(__EMSCRIPTEN__)

    // スレッドはsearching == trueで開始するので、idle_loop()で待機状態になるまで待つ。
    wait_for_search_finished();

#if STOCKFISH
	run_custom_job([this, &binder, &sharedState /*, &sm*/, n]() {
#else
	run_custom_job([this, &binder, &sharedState , n , worker_factory]() {
#endif
		// Use the binder to [maybe] bind the threads to a NUMA node before doing
        // the Worker allocation. Ideally we would also allocate the SearchManager
        // here, but that's minor.

        // スレッドを Worker 割り当ての前に NUMA ノードに（必要なら）バインドするために binder を使う。
        // 理想的にはここで SearchManager も割り当てたいが、それは些細なことだ。

        this->numaAccessToken = binder();
#if STOCKFISH
        this->worker = make_unique_large_page<Search::Worker>(
          sharedState, std::move(sm), n, idxInNuma, totalNuma, this->numaAccessToken);
#else
		// 🌈 やねうら王では、ここでworker_factoryを使ってWorker派生classを生成する。
		this->worker = std::move(worker_factory(sharedState,
												{n, idxInNuma, totalNuma, this->numaAccessToken}));
#endif
    });

    // Workerを生成するジョブが完了するまで待つ。
    wait_for_search_finished();

#else
    // yaneuraou.wasm
    // wait_for_search_finished すると、ブラウザのメインスレッドをブロックしデッドロックが発生するため、待機しない。
    //
    // 新しいスレッドが cv を設定するのを待ってから、ブラウザに処理をパスしたいが、
    // 新しいスレッド用のworkerを作成するためには、いったんブラウザに処理をパスする必要がある。
    //
    // https://bugzilla.mozilla.org/show_bug.cgi?id=1049079
    //
    // 📝 同じ理由でrun_custom_job()の完了も待てないため、workerは生成スレッド上で直接生成する。
    //     run_custom_job()に任せると、ジョブが実行される前にThread::clear_worker()が
    //     worker == nullptrのまま呼ばれうる。
    //     run_custom_job()を経由する目的はNUMAノードにbindしたスレッド上で
    //     workerのメモリを確保することだが、wasmにNUMAは無い。
    this->numaAccessToken = binder();
    this->worker          = std::move(
      worker_factory(sharedState, {n, idxInNuma, totalNuma, this->numaAccessToken}));
#endif
}

// Destructor wakes up the thread in idle_loop() and waits
// for its termination. Thread should be already waiting.

// デストラクタは idle_loop() 内でスレッドを起こし、
// 終了するのを待ちます。
// スレッドはすでに待機状態にある必要があります。

Thread::~Thread() {

    // 探索中にスレッドオブジェクトが解体されることはない。
    assert(!searching);

    // 探索は終わっているのでexitフラグをセットしてstart_searching()を呼べば終了するはず。
    exit = true;
    start_searching();
    stdThread.join();
}

// Wakes up the thread that will start the search
// 探索を開始するスレッドを起こします

void Thread::start_searching() {
    assert(worker != nullptr);
    run_custom_job([this]() { worker->start_searching(); });
}

// Clears the histories for the thread worker (usually before a new game)
// スレッドワーカーの履歴をクリアします（通常は新しい対局の前に実行されます）

void Thread::clear_worker() {
    assert(worker != nullptr);
    run_custom_job([this]() { worker->clear(); });
}

// Blocks on the condition variable until the thread has finished searching
// 条件変数でブロックし、スレッドが探索を完了するのを待ちます

void Thread::wait_for_search_finished() {

    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
}

// Launching a function in the thread
// スレッド内で関数を実行します

void Thread::run_custom_job(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return !searching; });
        jobFunc   = std::move(f);
        searching = true;
    }
    cv.notify_one();
}

void Thread::ensure_network_replicated() { worker->ensure_network_replicated(); }

// Thread gets parked here, blocked on the condition variable
// when the thread has no work to do.

// スレッドに処理すべき仕事がないとき、ここで待機状態（パーク）になり、
// 条件変数でブロックされます。

void Thread::idle_loop() {
    while (true)
    {
        std::unique_lock<std::mutex> lk(mutex);
        searching = false;
        cv.notify_one();  // Wake up anyone waiting for search finished
                          // 探索の完了を待っているスレッドを1つ起こします

        cv.wait(lk, [&] { return searching; });

        if (exit)
            return;

        std::function<void()> job = std::move(jobFunc);
        jobFunc                   = nullptr;

        lk.unlock();

        if (job)
            job();
    }
}


//Search::SearchManager* ThreadPool::main_manager() { return main_thread()->worker->main_manager(); }

u64 ThreadPool::nodes_searched() const { return accumulate(&Search::Worker::nodes); }
//u64 ThreadPool::tb_hits() const { return accumulate(&Search::Worker::tbHits); }

static usize next_power_of_two(u64 count) { return count > 1 ? (2ULL << msb(count - 1)) : 1; }

// Creates/destroys threads to match the requested number.
// Created and launched threads will immediately go to sleep in idle_loop.
// Upon resizing, threads are recreated to allow for binding if necessary.

// 要求されたスレッド数に合わせてスレッドを作成・破棄します。
// 作成され起動されたスレッドは、すぐに idle_loop 内でスリープ状態に入ります。
// リサイズ時には、必要に応じてスレッドのバインディングを可能にするために再作成されます。

void ThreadPool::set(const NumaConfig&   numaConfig,
                     Search::SharedState sharedState,
                     const Search::UpdateContext& updateContext,
                     // 🤔 やねうら王ではさらに抽象化する。
                     usize                       requested_threads,
                     const Search::WorkerFactory& worker_factory
) {

    /*  📓
		このあと、スレッドをいったん全部解体しているのは、確保するスレッド数がいま確保しているスレッド数と
		変わらないとしても、NumaPolicyに変更があると、割り当て方法が変わるからである。

		NumaPolicyとoptions["Threads"]に変更がなければ、再確保せずに済むのだが、
		worker_factoryが一致しない場合作り直す必要があり、その判定が難しいので毎回再確保することにする。
	*/

    // いま生成済みのスレッドは全部解体してしまう。
    if (threads.size() > 0)  // destroy any existing thread(s)
    {
        main_thread()->wait_for_search_finished();

        // 📝 これは、vector::clear()を呼び出してスレッドを解体している。
        threads.clear();

        boundThreadToNumaNode.clear();
    }

#if STOCKFISH
    const usize requested = sharedState.options["Threads"];
#else
    const usize requested = requested_threads;
    // 🤔 やねうら王では、ここ、"Threads"の値を反映させたくない。(DL系などで、ここに柔軟性が必要)
#endif

    if (requested > 0)  // create new thread(s)
    {
        // Binding threads may be problematic when there's multiple NUMA nodes and
        // multiple Stockfish instances running. In particular, if each instance
        // runs a single thread then they would all be mapped to the first NUMA node.
        // This is undesirable, and so the default behaviour (i.e. when the user does not
        // change the NumaConfig UCI setting) is to not bind the threads to processors
        // unless we know for sure that we span NUMA nodes and replication is required.

        // スレッドのバインディングは、複数のNUMAノードや複数のStockfishインスタンスが
        // 実行されている場合に問題を引き起こす可能性があります。
        // 特に、各インスタンスが1スレッドだけで動作する場合、それらはすべて
        // 最初のNUMAノードに割り当てられてしまうことになります。
        // これは望ましくないため、デフォルトの動作（つまり、ユーザーが
        // NumaConfig UCI設定を変更していない場合）は、
        // NUMAノードをまたいでおりレプリケーションが必要であることが
        // 確実に分かっている場合を除き、スレッドをプロセッサにバインドしないようになっています。

        // NumaPolicy
        //   none     ... バインドしない(1PCで複数エンジンを動かすときはこちらにすべき。)
        //   system   ... システムから利用可能なNUMA情報を取得。
        //   auto     ... systemとnoneを自動選択。
        //   hardware ... Windows10など古いシステムでスレッドを使い切らない時用。
        // 💡 詳しくは、やねうら王Wikiの「思考エンジンオプション」の説明を参考にすること。

        // options["NumaPolicy"]と要求されたスレッド数から考慮して、スレッドのbindが必要であるかを判定する。

        const std::string numaPolicy(sharedState.options["NumaPolicy"]);
        const bool        doBindThreads = [&]() {
            if (numaPolicy == "none")
                return false;

            if (numaPolicy == "auto")
                return numaConfig.suggests_binding_threads(requested);

            // numaPolicy == "system", or explicitly set by the user
            // numaPolicy が "system" であるか、またはユーザーによって明示的に設定された場合

            return true;
        }();

        std::map<NumaIndex, usize> counts;

        boundThreadToNumaNode = doBindThreads
                                ? numaConfig.distribute_threads_among_numa_nodes(requested)
                                : std::vector<NumaIndex>{};

        if (boundThreadToNumaNode.empty())
            counts[0] = requested;  // Pretend all threads are part of numa node 0
        else
            for (usize i = 0; i < boundThreadToNumaNode.size(); ++i)
                counts[boundThreadToNumaNode[i]]++;

        sharedState.sharedHistories.clear();
        for (auto pair : counts)
        {
            NumaIndex numaIndex = pair.first;
            u64       count     = pair.second;
            auto      f         = [&]() {
                sharedState.sharedHistories.try_emplace(numaIndex, next_power_of_two(count));
            };
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaIndex, f);
            else
                f();
        }

        auto threadsPerNode = counts;
        counts.clear();

        while (threads.size() < requested)
        {
            const usize    threadId      = threads.size();
            const NumaIndex numaId        = doBindThreads ? boundThreadToNumaNode[threadId] : 0;
            auto create_thread = [&]() {
#if STOCKFISH
                auto manager =
                  threadId == 0 ? std::make_unique<Search::SearchManager>(updateContext) : nullptr;
                // 💡 以前のStockfishはmain thread以外にNullSearchManagerを渡していたが、現行ではnullptr。
                //     やねうら王ではSearchManagerをEngine派生classが持ち、WorkerFactory経由でWorkerを生成する。
#endif

                // When not binding threads we want to force all access to happen
                // from the same NUMA node, because in case of NUMA replicated memory
                // accesses we don't want to trash cache in case the threads get scheduled
                // on the same NUMA node.

                // スレッドをバインドしない場合、すべてのアクセスが同じNUMAノードから
                // 行われるように強制したいと考えています。
                // なぜなら、NUMAのレプリケートメモリにアクセスする場合、
                // スレッドが同じNUMAノードにスケジューリングされるときに
                // キャッシュが破棄されるのを防ぎたいからです。

                auto binder = doBindThreads ? OptionalThreadToNumaNodeBinder(numaConfig, numaId)
                                            : OptionalThreadToNumaNodeBinder(numaId);
#if STOCKFISH
                threads.emplace_back(std::make_unique<Thread>(sharedState, std::move(manager),
                                                              threadId, counts[numaId]++,
                                                              threadsPerNode[numaId], binder));
#else
                threads.emplace_back(std::make_unique<Thread>(sharedState, worker_factory, threadId,
                                                              counts[numaId]++,
                                                              threadsPerNode[numaId], binder));
#endif
            };

            // Ensure the worker thread inherits the intended NUMA affinity at creation.
            // 生成時にスレッドが意図したNUMAアフィニティを継承するようにする。
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaId, create_thread);
            else
                create_thread();
        }

        // 生成したスレッドに対してThread::clear_worker()を呼び出す。
        // 🤔 std::make_unique<Thread>()でもWorker::clear()が呼び出されるので
        //     起動時には二重にclearしてしまうが、仕方がないか…。
        clear();

        // 🤔 これ、ThreadPool::clear()のなかでやっているので不要なのでは…。
        main_thread()->wait_for_search_finished();
    }
}


// Sets threadPool data to initial values
// threadPool のデータを初期値に設定する

// 📝 このmethodは、resize_threads()に対して呼び出される。
//     resize_threads()は、"isready"コマンドに対して呼び出されるので、
//     つまりは、このmethodは対局ごとに対局開始時に必ず呼び出される。

void ThreadPool::clear() {
	if (threads.empty())
		return;

	for (auto&& th : threads)
		th->clear_worker();

	for (auto&& th : threads)
		th->wait_for_search_finished();

	// 🤔 これはEngine派生class側で行うべき。
	//     ここにあったコードは、YaneuraOuEngine::clear()に移動させた。
#if STOCKFISH
	// These two affect the time taken on the first move of a game:
	main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
	main_manager()->previousTimeReduction = 0.85;

	main_manager()->callsCnt = 0;
	main_manager()->bestPreviousScore = VALUE_INFINITE;
	main_manager()->originalTimeAdjust = -1;
	main_manager()->tm.clear();
#endif
}

void ThreadPool::run_on_thread(usize threadId, std::function<void()> f) {
	assert(threads.size() > threadId);
	threads[threadId]->run_custom_job(std::move(f));
}

void ThreadPool::wait_on_thread(usize threadId) {
	assert(threads.size() > threadId);
	threads[threadId]->wait_for_search_finished();
}

usize ThreadPool::num_threads() const { return threads.size(); }


// Wakes up main thread waiting in idle_loop() and returns immediately.
// Main thread will wake up other threads and start the search.

// idle_loop() で待機しているメインスレッドを起こし、すぐにリターンする。
// メインスレッドは他のスレッドを起こして探索を開始する。

void ThreadPool::start_thinking(const OptionsMap&  options,
                                Position&          pos,
                                StateListPtr&      states,
                                Search::LimitsType limits) {

    main_thread()->wait_for_search_finished();

    // 📝 increaseDepthはmain_managerに移動させた。
    //     ここのある初期化のうち、stop以外は、Worker派生classで処理すべき。
    // 🌈 SearchManager::pre_start_searching()に移動させた。
	//     これは、Workerの派生classのpre_start_searching()から呼び出される。
#if STOCKFISH
    main_manager()->stopOnPonderhit = stop = false;
    main_manager()->ponder                                 = limits.ponderMode;
    increaseDepth                                          = true;
#else
    stop = false;
#endif

    Search::RootMoves rootMoves;
#if STOCKFISH
    for (const auto& usiMove : limits.searchmoves)
    {
        auto move = USIEngine::to_move(pos, usiMove);
        if (move != Move::none())
            rootMoves.emplace_back(move);
    }

    if (rootMoves.empty())
        for (const auto& m : MoveList<LEGAL_ALL>(pos))
            rootMoves.emplace_back(m);

#else

    // USIEngine::to_move()は不成も含めて合法性を確認するため、再度リストを検索しない。
    // GenerateAllLegalMoves == falseでも、searchmovesで明示された角不成などは許可する。
    for (const auto& usiMove : limits.searchmoves)
    {
        auto move = USIEngine::to_move(pos, usiMove);
        // resign、win、passなどは探索対象の通常の指し手ではない。
        if (move.is_ok())
            rootMoves.emplace_back(move);
    }

    // searchmovesが指定されない場合はGenerateAllLegalMovesを反映させる。
    // 指定された手がすべて無効だった場合も、従来どおり通常の探索に戻す。
    if (rootMoves.empty())
    {
        bool generate_all_legal_moves =
          options.count("GenerateAllLegalMoves") && options["GenerateAllLegalMoves"];

        if (generate_all_legal_moves)
            for (const auto& m : MoveList<LEGAL_ALL>(pos))
                rootMoves.emplace_back(m);
        else
            for (const auto& m : MoveList<LEGAL>(pos))
                rootMoves.emplace_back(m);
    }

#endif

    //Tablebases::Config tbConfig = Tablebases::rank_root_moves(options, pos, rootMoves);
    // ⇨  Tablebasesは将棋では用いないのでコメントアウト

    // After ownership transfer 'states' becomes empty, so if we stop the search
    // and call 'go' again without setting a new position states.get() == nullptr.

    // 所有権の移動後、'states' は空になるため、検索を中断して
    // 新しい局面を設定せずに再度 'go' を呼び出すと、states.get() == nullptr となる。

    assert(states.get() || setupStates.get());

    if (states.get())
        setupStates = std::move(states);  // Ownership transfer, states is now empty

    // We use Position::set() to set root position across threads. But there are
    // some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
    // be deduced from a fen string, so set() clears them and they are set from
    // setupStates->back() later. The rootState is per thread, earlier states are
    // shared since they are read-only.

    // 複数のスレッドでルート局面を設定するために Position::set() を使用します。
    // しかし、StateInfo の一部のフィールド（previous、pliesFromNull、capturedPiece）は
    // FEN 文字列からは推測できないため、set() はそれらをクリアし、後で setupStates->back() から設定されます。
    // rootState はスレッドごとに個別ですが、それ以前の状態は読み取り専用のため共有されます。

    for (auto&& th : threads)
    {
        th->run_custom_job([&]() {
#if STOCKFISH
            th->worker->limits = limits;
            th->worker->nodes = th->worker->tbHits = th->worker->bestMoveChanges = 0;
            th->worker->nmpMinPly                                                = 0;
            th->worker->rootDepth = th->worker->completedDepth = 0;
            th->worker->rootMoves                              = rootMoves;
            th->worker->rootPos.set(pos.fen(), pos.is_chess960(), &th->worker->rootState);
            th->worker->rootState = setupStates->back();
            th->worker->tbConfig  = tbConfig;
#else

            th->worker->limits = limits;
            th->worker->nodes  = 0;
#endif

			// 📝 tbHits、tbConfigは将棋では使わない。

#if STOCKFISH
            th->worker->nmpMinPly = 0;
			th->worker->bestMoveChanges = 0;
            th->worker->rootDepth = th->worker->completedDepth = 0;

            // 🤔 やねうら王では、Worker派生classのpre_start_searching()で行うようにする。
            //     やねうら王では、void Search::YaneuraOuWorker::pre_start_searching()で行っている。
#endif

            th->worker->rootMoves = rootMoves;
            th->worker->rootPos.set(pos.sfen(), &th->worker->rootState);
            th->worker->rootState = setupStates->back();

#if !STOCKFISH
			// ⚠ どうせなら、↑でworker->rootPos.set()が終わってから呼び出したい。
			//     (rootPosを使って入玉判定などを行いたいため)
            th->worker->pre_start_searching();
#endif
		});
    }

    for (auto&& th : threads)
        th->wait_for_search_finished();

    main_thread()->start_searching();
}

// ⚠ このメソッドは、やねうら王の標準探索エンジンでしか使わないので、
//     YaneuraOuEngine側に移動させた。
//Thread* ThreadPool::get_best_thread() const


// Start non-main threads.
// Will be invoked by main thread after it has started searching.

// メインスレッド以外のスレッドを開始する。
// メインスレッドが探索を開始した後に呼び出される。

void ThreadPool::start_searching() {

	for (auto&& th : threads)
		if (th != threads.front())
			th->start_searching();
}


// Wait for non-main threads

// メインスレッド以外のスレッドを待機する

void ThreadPool::wait_for_search_finished() const {

	for (auto&& th : threads)
		if (th != threads.front())
			th->wait_for_search_finished();
}

std::vector<usize> ThreadPool::get_bound_thread_to_numa_node() const {
    return boundThreadToNumaNode;
}

std::vector<usize> ThreadPool::get_bound_thread_count_by_numa_node() const {
	std::vector<usize> counts;

	if (!boundThreadToNumaNode.empty())
	{
		NumaIndex highestNumaNode = 0;
		for (NumaIndex n : boundThreadToNumaNode)
			if (n > highestNumaNode)
				highestNumaNode = n;

		counts.resize(highestNumaNode + 1, 0);

		for (NumaIndex n : boundThreadToNumaNode)
			counts[n] += 1;
	}

	return counts;
}

usize ThreadPool::numa_nodes() const {
    std::unordered_set<usize> seen;
    for (NumaIndex n : boundThreadToNumaNode)
        seen.insert(n);
    return std::max(seen.size(), usize(1));
}

void ThreadPool::ensure_network_replicated() {
	for (auto&& th : threads)
		th->ensure_network_replicated();
}

} // namespace YaneuraOu
