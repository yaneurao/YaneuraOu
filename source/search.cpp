#include "search.h"

#include <algorithm>
#include <cstdint>
#include "evaluate.h"
#include "misc.h"
#include "types.h"

namespace YaneuraOu {

#if STOCKFISH
Search::Worker::Worker(SharedState&                   sharedState,
                       std::unique_ptr<SearchManager> sm,
                       usize                          threadId,
                       usize                          numaThreadId,
                       usize                          numaTotalThreads,
                       NumaReplicatedAccessToken      token) :
#else
Search::Worker::Worker(SharedState& sharedState, const ThreadIds& ids) :
#endif
    // Unpack the SharedState struct into member variables
    // SharedState構造体の内容を各メンバー変数へ展開する。
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
    options(sharedState.options),
    threads(sharedState.threads),
    tt(sharedState.tt) {
    // sharedHistory・network・refreshTableは、派生クラスYaneuraOuWorkerが保持する。
    // Worker生成後、ThreadPool::set()からThreadPool::clear()を経てclear()を呼ぶ。
    // 派生Workerの初期化も必要なため、この基底コンストラクターではclear()を呼ばない。
}

} // namespace YaneuraOu

