// Stockfishとのinclude構造・型名の互換性を維持するための共通入口。
// 評価関数の種類に応じて、やねうら王の実装・型の別名・ダミー実装を提供する。

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "../config.h"
#include "../evaluate.h"

// 探索・Engine側の共通入口。評価実装の内部からは実装ヘッダーを直接読む。
// このヘッダーから実装を選び、実装からこの入口を読み返す循環を作らない。
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
#include "../eval/nnue/evaluate_nnue.h"
#else
namespace YaneuraOu::Eval {

// NNUEの重みを持たないエンジンでも、同じ形でネットワークを参照できるようにする。
class NetworkStorage {
   public:
    explicit NetworkStorage(NumaReplicationContext&) {}
    const NetworkStorage& operator[](const NumaReplicatedAccessToken&) const { return *this; }
};

} // namespace YaneuraOu::Eval
#endif

#endif
