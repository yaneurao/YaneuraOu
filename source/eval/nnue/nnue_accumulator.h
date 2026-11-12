// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../types.h"
#include "../../misc.h"
#include "nnue_architecture.h"
#include "../../evaluate.h"

#include <array>
#include <cassert>

namespace YaneuraOu {
namespace Eval::NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 従来NNUEでは最終評価値も保持する。SFNNでは手番に依存しないFT累積値だけを保持する。
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  i16
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  bool computed_accumulation = false;
#if defined(EVAL_NNUE)
  Value score = VALUE_ZERO;
  bool computed_score = false;
#endif
};

#if defined(EVAL_SFNN)
// Changes are written directly into the next accumulator frame.
// 着手の差分は、次の累積値フレームへ直接書き込む。
struct Dirties {
    Eval::DirtyPiece dirtyPiece;
};

struct AccumulatorState: public Accumulator, Dirties {
    AccumulatorState* previous = nullptr;
};

class AccumulatorStack {
   public:
    static constexpr usize MaxSize = MAX_PLY + 1;

    AccumulatorStack() noexcept { reset(); }
    const AccumulatorState& latest() const noexcept { return accumulators[size - 1]; }
    AccumulatorState& latest() noexcept { return accumulators[size - 1]; }

    void reset() noexcept {
        size = 1;
        auto& st = latest();
        st.computed_accumulation = false;
        st.previous = nullptr;
        st.dirtyPiece.dirty_num = 0;
    }

    Dirties& push([[maybe_unused]] const Position& pos) noexcept {
        assert(size < MaxSize);
        auto* previous = &latest();
        auto& st = accumulators[size++];
        st.computed_accumulation = false;
        st.previous = previous;
        st.dirtyPiece.dirty_num = 0;
        return st;
    }

    void pop() noexcept {
        assert(size > 1);
        --size;
    }

   private:
    std::array<AccumulatorState, MaxSize> accumulators;
    usize size = 1;
};
#else
using AccumulatorState = StateInfo;
#endif

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
