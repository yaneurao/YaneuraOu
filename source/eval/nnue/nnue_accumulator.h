// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_INCLUDED
#define CLASSIC_NNUE_ACCUMULATOR_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../types.h"
#include "../../misc.h"
#include "nnue_architecture.h"
#include "../../evaluate.h"
#include "../../memory.h"
#if defined(USE_FINNY_TABLES)
#include "features/half_ka_hm2.h"
#include "features/index_list.h"
#endif

#include <array>
#include <cassert>
#include <type_traits>

namespace YaneuraOu {
namespace Eval::NNUE {

// Per-worker Finny tables; the shared network contains only immutable parameters.
// WorkerごとのFinny Tables。共有ネットワークには読み取り専用のパラメーターだけを置く。
class AccumulatorCaches {
   public:
#if defined(USE_FINNY_TABLES)
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE2) || defined(USE_MMX) || defined(USE_NEON)
    static constexpr bool kUseDirectFinny = std::is_same_v<
      RawFeatures, Features::FeatureSet<Features::HalfKA_hm2<Features::Side::kFriend>>>;
#else
    static constexpr bool kUseDirectFinny = false;
#endif
    struct FinnyPosition {
        std::array<Piece, SQ_NB_PLUS1> pieces{};
        Hand hands[COLOR_NB]{};
        int pieceCount = 0;
    };

    struct alignas(kCacheLineSize) Entry {
        i16 accumulation[kTransformedFeatureDimensions];
        // Raw snapshots and generic feature lists are alternative cache formats.
        // 生の盤面と汎用特徴量リストは別々の保存方式なので、同じ領域を共用する。
        union {
            FinnyPosition position;
            Features::IndexList active_indices;
        };
        Entry(): Entry(std::bool_constant<kUseDirectFinny>{}) {}
        bool is_initialized() const {
            // HalfKA_hm2 requires both kings, so zero pieces denotes an unused snapshot.
            // HalfKA_hm2は両玉が必要なので、駒数0を未使用の盤面として扱える。
            if constexpr (kUseDirectFinny) return position.pieceCount != 0;
            else return active_indices.size() != 0;
        }
        void reset_position() {
            if constexpr (kUseDirectFinny) position = {};
            else active_indices.resize(0);
        }
       private:
        Entry(std::true_type): position{} {}
        Entry(std::false_type): active_indices{} {}
    };

    struct Data {
        using TriggerEntries = std::array<std::array<Entry, SQ_NB>, COLOR_NB>;
        std::array<TriggerEntries, kRefreshTriggers.size()> entries;
    };

    Data& operator*() { assert(storage); return *storage; }
#endif

    // Called on the bound worker thread, also after replacing the network.
    // NUMAにbindしたWorkerスレッド上で呼ぶ。ネットワークを交換した後も呼び直す。
    template<typename Network>
    void clear([[maybe_unused]] const Network& network) {
#if defined(USE_FINNY_TABLES)
        if constexpr (kTransformedFeatureDimensions <= 4096) {
            if (!storage) storage = make_unique_large_page<Data>();
            for (auto& triggers : storage->entries)
                for (auto& perspectives : triggers)
                    for (auto& entry : perspectives)
                        entry.reset_position();
        }
#endif
        // Unlike Stockfish, invalidate only snapshots here; copy biases on first access.
        // Stockfishと異なり、ここでは盤面等だけを無効化し、biasは初回アクセスでコピーする。
        // 従来のFinny再構築経路を維持し、未使用エントリーの累積値は読み込まない。
    }

    bool empty() const {
#if defined(USE_FINNY_TABLES)
        return !storage;
#else
        return true;
#endif
    }

   private:
#if defined(USE_FINNY_TABLES)
    // Keep the large cache allocation and first touch on the worker's NUMA node.
    // 大きなキャッシュの確保と最初の書き込みをWorkerのNUMAノードで行う。
    LargePagePtr<Data> storage;
#endif
};

// 入力特徴量をアフィン変換した結果を保持するクラス
// SFNN・従来NNUEとも、手番に依存しないFT累積値だけを保持する。
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  i16
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  bool computed_accumulation = false;
};

// Changes are written directly into the next accumulator frame.
// 着手の差分は、次の累積値フレームへ直接書き込む。
struct Dirties {
    Eval::DirtyPiece dirtyPiece;
};

#if defined(EVAL_SFNN) && defined(NNUE_SFNN_PROGRESS_BUCKETS) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress { struct Parameters; }
#endif

struct AccumulatorState: public Accumulator, Dirties {
    AccumulatorState* previous = nullptr;
#if defined(EVAL_SFNN) && defined(NNUE_SFNN_PROGRESS_BUCKETS) && NNUE_SFNN_PROGRESS_BUCKETS != 1
    // 同じ盤面を表すフレームではnull moveの前後でも進行度の和を共有できる。
    i64 progressSum = 0;
    const Progress::Parameters* progressParameters = nullptr;
#endif
};

class AccumulatorStack {
   public:
    static constexpr usize MaxSize = MAX_PLY + 1;

    AccumulatorStack() noexcept { reset(); }
    const AccumulatorState& latest() const noexcept { return accumulators[size - 1]; }
    AccumulatorState& latest() noexcept { return accumulators[size - 1]; }

    void reset() noexcept {
        // 探索の初回evaluate()では局面の差分更新ができないので、
        // SFNN・従来NNUEとも累積値を未計算の状態へ戻す。
        size = 1;
        auto& st = latest();
        st.computed_accumulation = false;
        st.previous = nullptr;
        st.dirtyPiece = {};
#if defined(EVAL_SFNN) && defined(NNUE_SFNN_PROGRESS_BUCKETS) && NNUE_SFNN_PROGRESS_BUCKETS != 1
        st.progressParameters = nullptr;
#endif
    }

    Dirties& push([[maybe_unused]] const Position& pos) noexcept {
        assert(size < MaxSize);
        auto* previous = &latest();
        auto& st = accumulators[size++];
        st.computed_accumulation = false;
        st.previous = previous;
        st.dirtyPiece = {};
#if defined(EVAL_SFNN) && defined(NNUE_SFNN_PROGRESS_BUCKETS) && NNUE_SFNN_PROGRESS_BUCKETS != 1
        st.progressParameters = nullptr;
#endif
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


} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
