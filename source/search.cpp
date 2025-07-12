#include "search.h"
#include "evaluate.h"

namespace YaneuraOu {

Search::Worker::Worker(
	OptionsMap& options, ThreadPool& threads, size_t threadIdx, NumaReplicatedAccessToken numaAccessToken
	/*
						SharedState&                    sharedState,
                        std::unique_ptr<ISearchManager> sm,
                        size_t                          threadId,
                        NumaReplicatedAccessToken       token
	*/
	) :
    // Unpack the SharedState struct into member variables
	// 💡 StockfishではSharedState構造体から、このclassのmember変数を初期化しているが、
	//     SharedStateは、やねうら王では採用しないことにした。

	options(options),
	threads(threads),
	threadIdx(threadIdx),
    numaAccessToken(numaAccessToken)

	#if 0
    manager(std::move(sm)),
    networks(sharedState.networks),
    tt(sharedState.tt)
	refreshTable(networks[token])
	#endif
{
    //clear();

	// 🤔　このclear()は不要だと思う。
	//      Engineのコンストラクタや、Threads変更時にはresize_threads()が呼び出されるし、
	//      resize_threads()のなかでThreadPool::clear()が呼び出され、そのなかからWorker::clear()が呼び出される。
	//      また、"usinewgame"に対して Engine.search_clear()が呼び出されるので、そこからもWorker::clear()が呼び出される。
}

// TODO : あとで
#if 0
void Search::Worker::ensure_network_replicated() {
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
}
#endif

// void Search::Worker::start_searching()
// 💡　エンジン実装部で定義する。

void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st) {
    do_move(pos, move, st, pos.gives_check(move));
}

void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck) {
#if 0
	// accumulatorStackを用いる実装。
	// TODO : あとで

	DirtyPiece dp = pos.do_move(move, st, givesCheck /*, &tt */);
	// 📝　やねうら王では、TTのprefetchをしないので、ttを渡す必要がない。
	nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);
#else
	pos.do_move(move, st, givesCheck);
	nodes.fetch_add(1, std::memory_order_relaxed);
#endif
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st)
{
	pos.do_null_move(st /*, tt*/);
	// 📝　やねうら王では、TTのprefetchをしないので、ttを渡す必要がない。
}

void Search::Worker::undo_move(Position& pos, const Move move) {
	pos.undo_move(move);

	// accumulatorStackを用いる実装はEngine派生class側で行うべき。
	// TODO : あとで。
	//accumulatorStack.pop();
}

void Search::Worker::undo_null_move(Position& pos) { pos.undo_null_move(); }

// Reset histories, usually before a new game
//void Search::Worker::clear()
// 💡　エンジン実装部で定義する。

} // namespace YaneuraOu

