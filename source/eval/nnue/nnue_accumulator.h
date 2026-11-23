// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_INCLUDED
#define CLASSIC_NNUE_ACCUMULATOR_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../types.h"
#include "../../misc.h"
#include "nnue_architecture.h"
#include "../../evaluate.h"

#include <array>
#include <cassert>
#include <cstring>

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
#if defined(EVAL_NNUE)
        st.computed_score = false;
#endif
        st.previous = nullptr;
        st.dirtyPiece.dirty_num = 0;
    }

    Dirties& push([[maybe_unused]] const Position& pos) noexcept {
        assert(size < MaxSize);
        auto* previous = &latest();
        auto& st = accumulators[size++];
        st.computed_accumulation = false;
#if defined(EVAL_NNUE)
        st.computed_score = false;
#endif
        st.previous = previous;
        st.dirtyPiece.dirty_num = 0;
        return st;
    }

#if defined(EVAL_NNUE)
    void push_null_move(const Position& pos) noexcept {
        auto* previous = &latest();
        push(pos);
        // Copy bytes even when the parent FT has not been computed yet.
        // 親のFTが未計算でも、未初期化の要素を値として読み出さずバイト列でコピーする。
        std::memcpy(latest().accumulation, previous->accumulation,
                    sizeof(latest().accumulation));
        latest().computed_accumulation = previous->computed_accumulation;
        latest().computed_score = false;
    }
#endif

    void pop() noexcept {
        assert(size > 1);
        --size;
    }

   private:
    std::array<AccumulatorState, MaxSize> accumulators;
    usize size = 1;
};


} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
