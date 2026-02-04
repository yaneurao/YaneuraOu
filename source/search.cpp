#include "search.h"
#include "evaluate.h"

namespace YaneuraOu {

Search::Worker::Worker(
						SharedState& sharedState,
                        //std::unique_ptr<ISearchManager> sm,
						size_t threadIdx,
						size_t                          numaThreadId,
						size_t                          numaTotalThreads,
						NumaReplicatedAccessToken		token
	) :
    // Unpack the SharedState struct into member variables
	// 🌈 これは、やねうら王ではYaneuraOuWorkerのほうが持っている。
    //sharedHistory(sharedState.sharedHistories.at(token.get_numa_index())),
	options(sharedState.options),
	threads(sharedState.threads),
	threadIdx(threadIdx),
    numaThreadIdx(numaThreadId),
    numaTotal(numaTotalThreads),
    numaAccessToken(token),
    //manager(std::move(sm)),
	tt(sharedState.tt)
    //networks(sharedState.networks),
	//refreshTable(networks[token])
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

// Reset histories, usually before a new game
//void Search::Worker::clear()
// 💡　エンジン実装部で定義する。


} // namespace YaneuraOu

