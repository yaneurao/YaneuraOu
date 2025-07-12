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

// Reset histories, usually before a new game
//void Search::Worker::clear()
// 💡　エンジン実装部で定義する。


} // namespace YaneuraOu

