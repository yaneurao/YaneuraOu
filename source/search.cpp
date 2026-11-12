#include "search.h"

#include <algorithm>
#include <cstdint>
#include "evaluate.h"
#include "misc.h"
#include "types.h"

namespace YaneuraOu {

Search::Worker::Worker(SharedState& sharedState,
#if STOCKFISH
                       std::unique_ptr<SearchManager> sm,
                       usize                           threadId,
                       usize                           numaThreadId,
                       usize                           numaTotalThreads,
                       NumaReplicatedAccessToken       token
#else
                       const ThreadIds& ids
#endif
) : 
    // Unpack the SharedState struct into member variables
    // SharedState構造体の内容を各メンバー変数へ展開する。
	// 🌈 これは、やねうら王ではYaneuraOuWorkerのほうが持っている。
    //sharedHistory(sharedState.sharedHistories.at(token.get_numa_index())),
	options(sharedState.options),
	threads(sharedState.threads),
#if STOCKFISH
	threadIdx(threadId),
    numaThreadIdx(numaThreadId),
    numaTotal(numaTotalThreads),
    numaAccessToken(token),
    manager(std::move(sm)),
#else
	threadIdx(ids.threadIdx),
    numaThreadIdx(ids.numaThreadIdx),
    numaTotal(ids.numaTotal),
    numaAccessToken(ids.numaAccessToken),
#endif
	tt(sharedState.tt)
    //network(sharedState.network),
	//refreshTable(network[token])
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
    // 一度アクセスして、遅延初期化を実行しておく。
    // 探索中に初期化が発生するのを避けるため。
    (void) (network[numaAccessToken]);
}
#endif

// void Search::Worker::start_searching()
// 💡　エンジン実装部で定義する。

// Reset histories, usually before a new game
// 通常、新しい対局の前に履歴をリセットする。
//void Search::Worker::clear()
// 💡　エンジン実装部で定義する。


} // namespace YaneuraOu

