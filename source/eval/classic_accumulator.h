#ifndef CLASSIC_ACCUMULATOR_H_INCLUDED
#define CLASSIC_ACCUMULATOR_H_INCLUDED

#include "../config.h"

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
#include <array>
#include <cassert>

#include "../evaluate.h"
#include "evalsum.h"

namespace YaneuraOu::Eval::Classic {

// KPPT・KPP_KKPTの評価状態。盤面の履歴とは分離し、探索Workerが保持する。
struct Accumulator {
    EvalSum sum;
    DirtyPiece dirtyPiece;
    Value materialValue;
    Accumulator* previous;
};

class AccumulatorStack {
   public:
    AccumulatorStack() noexcept { reset(); }
    Accumulator& latest() noexcept { return accumulators[size - 1]; }

    void reset() noexcept {
        size = 1;
        invalidate(latest(), nullptr);
    }

    DirtyPiece& push() noexcept {
        assert(size < accumulators.size());
        auto* previous = &latest();
        auto& next = accumulators[size++];
        invalidate(next, previous);
        return next.dirtyPiece;
    }

    // null moveでは盤上・手駒とも変わらないので積まず、同じ累積値を使う。
    void pop() noexcept {
        assert(size > 1);
        --size;
    }

   private:
    static void invalidate(Accumulator& state, Accumulator* previous) noexcept {
        state.sum.p[0][0] = VALUE_NOT_EVALUATED;
        state.dirtyPiece = {};
        state.materialValue = 0;
        state.previous = previous;
    }

    std::array<Accumulator, MAX_PLY + 1> accumulators;
    usize size = 1;
};

} // namespace YaneuraOu::Eval::Classic

#endif
#endif
