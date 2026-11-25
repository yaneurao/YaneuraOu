#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include "../config.h"
#include "../evaluate.h"

// 探索側の共通入口。評価実装内部は、この入口ではなく実装ヘッダーを直接読む。
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
#include "../eval/nnue/nnue_accumulator.h"
#else
#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
#include "../eval/classic_accumulator.h"
#endif

namespace YaneuraOu::Eval::NNUE {

// Stockfishと同じ型名で探索から扱うための選択層。
// NNUE以外でもこの名前空間を使うが、NNUEの累積値やネットワークを確保するわけではない。
#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
using Dirties = Eval::DirtyPiece;
using AccumulatorStack = Eval::Classic::AccumulatorStack;
#else
using Dirties = Eval::EmptyDirties;

// 累積値を使わない評価関数も、同じ形で初期化・着手・undoを呼べるようにする。
// 差分の配列やスタック位置は持たず、push()/reset()/pop()による更新も行わない。
class AccumulatorStack {
   public:
    Dirties& push([[maybe_unused]] const Position& pos) noexcept { return dirties; }
    void reset() noexcept {}
    void pop() noexcept {}

   private:
    Dirties dirties;
};
#endif

// Finny Tablesを使わない評価関数も同じ形でclear()を呼べるようにする。
class AccumulatorCaches {
   public:
    template<typename Network>
    void clear(const Network&) noexcept {}
};

} // namespace YaneuraOu::Eval::NNUE
#endif

#endif
