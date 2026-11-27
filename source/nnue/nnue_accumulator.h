// Stockfishとのinclude構造・型名の互換性を維持するための共通入口。
// 評価関数の種類に応じて、やねうら王の実装・型の別名・ダミー実装を提供する。

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include "../config.h"
#include "../evaluate.h"

// 探索・評価実装の共通入口。NNUE/SFNNの実装と、他の評価関数用の型をここで選ぶ。
// 旧評価関数のEvalSum・Accumulatorはこのヘッダーの旧評価分岐にまとめる。
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
#include "../types.h"
#include "../misc.h"
#include "nnue_architecture.h"
#include "../memory.h"
#if defined(USE_FINNY_TABLES)
#include "features/index_list.h"
#endif

#include <array>
#include <cassert>
#include <type_traits>

namespace YaneuraOu::Eval::NNUE {

// Per-worker Finny tables; the shared network contains only immutable parameters.
// WorkerごとのFinny Tables。共有ネットワークには読み取り専用のパラメーターだけを置く。
class AccumulatorCaches {
   public:
#if defined(USE_FINNY_TABLES)
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE2) || defined(USE_NEON)
    static constexpr bool kUseDirectFinny = kUseHalfKAHm2;
#else
    static constexpr bool kUseDirectFinny = false;
#endif
    struct FinnyPosition {
        std::array<Piece, SQ_NB_PLUS1> pieces{};
        Hand hands[COLOR_NB]{};
        int pieceCount = 0;
    };

    struct alignas(CacheLineSize) Entry {
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

// AccumulatorStateのうち、FT累積値の部分だけを参照する。
// 差分や前フレームへのリンクを含めず、累積値だけをコピーする場合にも使う。
inline Accumulator& accumulator_of(AccumulatorState* state) { return *state; }
inline const Accumulator& accumulator_of(const AccumulatorState* state) { return *state; }

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
        invalidate(latest(), nullptr);
    }

    Dirties& push([[maybe_unused]] const Position& pos) noexcept {
        assert(size < MaxSize);
        auto* previous = &latest();
        auto& st = accumulators[size++];
        invalidate(st, previous);
        return st;
    }

    void pop() noexcept {
        assert(size > 1);
        --size;
    }

   private:
    // Reset frame metadata without clearing the FT accumulation arrays.
    // FT累積値の配列は消去せず、計算済みフラグ・履歴リンク・着手差分・進行度キャッシュの参照を初期化する。
    static void invalidate(AccumulatorState& st, AccumulatorState* previous) noexcept {
        st.computed_accumulation = false;
        st.previous = previous;
        st.dirtyPiece = {};
#if defined(EVAL_SFNN) && defined(NNUE_SFNN_PROGRESS_BUCKETS) && NNUE_SFNN_PROGRESS_BUCKETS != 1
        st.progressParameters = nullptr;
#endif
    }

    std::array<AccumulatorState, MaxSize> accumulators;
    usize size = 1;
};


} // namespace YaneuraOu::Eval::NNUE
#else
#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
#include <array>
#include <cassert>
#include <ostream>

// KPPT・KPP_KKPT専用。NNUE/SFNNやダミー構成ではこれらの定義は読み込まない。
namespace YaneuraOu::Eval {

	// KPPT,KPP_PPTで使うためのヘルパクラス
	// 手番つきの評価値の合計を計算するために用いる。

	// std::array<T,2>に対して 基本的な演算を提供する。
	template <typename Tl, typename Tr>
	FORCE_INLINE std::array<Tl, 2> operator += (std::array<Tl, 2> & lhs, const std::array<Tr, 2> & rhs) {
		lhs[0] += (Tl)rhs[0];
		lhs[1] += (Tl)rhs[1];
		return lhs;
	}
	template <typename Tl, typename Tr>
	FORCE_INLINE std::array<Tl, 2> operator -= (std::array<Tl, 2> & lhs, const std::array<Tr, 2> & rhs) {
		lhs[0] -= (Tl)rhs[0];
		lhs[1] -= (Tl)rhs[1];
		return lhs;
	}
	template <typename Tl, typename Tr>
	FORCE_INLINE bool operator == (std::array<Tl, 2> & lhs, const std::array<Tr, 2> & rhs) {
		return lhs[0] == rhs[0] && lhs[1] == rhs[1];
	}
	template <typename Tl, typename Tr>
	FORCE_INLINE bool operator != (std::array<Tl, 2> & lhs, const std::array<Tr, 2> & rhs) {
		return !(lhs == rhs);
	}
	template <typename Tl>
	FORCE_INLINE std::array<Tl, 2> operator - (const std::array<Tl, 2> & rhs) {
		std::array<Tl, 2> a;
		a[0] = -rhs[0];
		a[1] = -rhs[1];
		return a;
	}
	template <typename Tl>
	FORCE_INLINE std::array<Tl, 2> operator + (const std::array<Tl, 2> & lhs, const std::array<Tl, 2> & rhs) {
		std::array<Tl, 2> tmp = lhs;
		tmp += rhs;
		return tmp;
	}
	template <typename Tl>
	FORCE_INLINE std::array<Tl, 2> operator - (const std::array<Tl, 2> & lhs, const std::array<Tl, 2> & rhs) {
		std::array<Tl, 2> tmp = lhs;
		tmp -= rhs;
		return tmp;
	}
	template <typename Tl>
	FORCE_INLINE std::array<Tl, 2> operator * (const std::array<Tl, 2> & rhs, int n) {
		std::array<Tl, 2> a;
		a[0] = rhs[0] * n;
		a[1] = rhs[1] * n;
		return a;
	}
	template <typename Tl>
	FORCE_INLINE std::array<Tl, 2> operator / (const std::array<Tl, 2> & rhs, int n) {
		std::array<Tl, 2> a;
		a[0] = rhs[0] / n;
		a[1] = rhs[1] / n;
		return a;
	}

	// 与えられたarrayが0ベクトルであるかどうかを判定する。
	template <typename T>
	bool is_zero(const std::array<T, 2> v)
	{
		return v[0] == T(0) && v[1] == T(0);
	}

	// デバッグ用出力
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::array<T, 2> v)
	{
		os << "{ " << v[0] << "," << v[1] << " } ";
		return os;
	}

	//
	// 手番つきの評価値を足していくときに使うclass
	//

	// EvalSum sum;
	// に対して
	// sum.p[0] = ΣBKKP
	// sum.p[1] = ΣWKPP
	// sum.p[2] = ΣKK (or ΣPPなど)
	// (それぞれに手番は加味されているものとする)
	// sum.sum() == ΣBKPP - ΣWKPP + ΣKK (or ΣPPなど)

	// EvalSumクラスは、コンストラクタでの初期化が保証できないので(オーバーヘッドがあるのでやりたくないので)
	// GCC 7.1.0以降で警告が出るのを回避できない。ゆえに、このクラスではこの警告を抑制する。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

	struct alignas(32) EvalSum {

#if defined(USE_AVX2)
		EvalSum(const EvalSum & es) {
			_mm256_store_si256(&mm, es.mm);
		}
		EvalSum& operator = (const EvalSum & rhs) {
			_mm256_store_si256(&mm, rhs.mm);
			return *this;
		}
#elif defined(USE_SSE2)
		EvalSum(const EvalSum & es) {
			_mm_store_si128(&m[0], es.m[0]);
			_mm_store_si128(&m[1], es.m[1]);
		}
		EvalSum& operator = (const EvalSum & rhs) {
			_mm_store_si128(&m[0], rhs.m[0]);
			_mm_store_si128(&m[1], rhs.m[1]);
			return *this;
		}
#endif

		EvalSum() {}

		// この局面の手番は c側にあるものとする。c側から見た評価値を返す。
		i32 sum(const Color c) const
		{
			// NDF(2014)の手番評価の手法。
			// cf. http://www.computer-shogi.org/wcsc24/appeal/NineDayFever/NDF.txt

#if defined(EVAL_KPP_KKPT)

			// p[0][1]とp[1][1]は使っていないタイプのEvalSum
			const i32 scoreBoard = p[0][0] - p[1][0] + p[2][0];
			// 手番に依存する評価値合計
			const i32 scoreTurn = p[2][1];

#else // EVAL_KPPT , EVAL_KKPPT などはこちら。

			// 手番に依存しない評価値合計
			// p[1][0]はΣWKPPなので符号はマイナス。
			const i32 scoreBoard = p[0][0] - p[1][0] + p[2][0];
			// 手番に依存する評価値合計
			const i32 scoreTurn = p[0][1] + p[1][1] + p[2][1];
#endif

			// この関数は手番側から見た評価値を返すのでscoreTurnは必ずプラス

			return (c == BLACK ? scoreBoard : -scoreBoard) + scoreTurn;
		}

		EvalSum& operator += (const EvalSum & rhs)
		{
#if defined(USE_AVX2)
			mm = _mm256_add_epi32(mm, rhs.mm);
#elif defined(USE_SSE2)
			m[0] = _mm_add_epi32(m[0], rhs.m[0]);
			m[1] = _mm_add_epi32(m[1], rhs.m[1]);
#else
			p[0][0] += rhs.p[0][0];
			p[0][1] += rhs.p[0][1];
			p[1][0] += rhs.p[1][0];
			p[1][1] += rhs.p[1][1];
			p[2][0] += rhs.p[2][0];
			p[2][1] += rhs.p[2][1];
#endif
			return *this;
		}
		EvalSum& operator -= (const EvalSum & rhs)
		{
#if defined(USE_AVX2)
			mm = _mm256_sub_epi32(mm, rhs.mm);
#elif defined(USE_SSE2)
			m[0] = _mm_sub_epi32(m[0], rhs.m[0]);
			m[1] = _mm_sub_epi32(m[1], rhs.m[1]);
#else
			p[0][0] -= rhs.p[0][0];
			p[0][1] -= rhs.p[0][1];
			p[1][0] -= rhs.p[1][0];
			p[1][1] -= rhs.p[1][1];
			p[2][0] -= rhs.p[2][0];
			p[2][1] -= rhs.p[2][1];
#endif
			return *this;
		}
		EvalSum operator + (const EvalSum & rhs) const { return EvalSum(*this) += rhs; }
		EvalSum operator - (const EvalSum & rhs) const { return EvalSum(*this) -= rhs; }

		// 評価値が計算済みであるかを判定する。
		// このEvalSumに値が入っていないときは、AccumulatorStackのreset()/push()で
		// p[0][0]にVALUE_NOT_EVALUATEDを設定することになっている。
		bool evaluated() const { return p[0][0] != VALUE_NOT_EVALUATED; }

		union {
			// KPP・KK/KKPの3成分を保持し、4番目は32バイトのSIMD演算用パディングとする。
			std::array<std::array<i32, 2>, 4> p;

#if defined(USE_AVX2)
			__m256i mm;
			__m128i m[2];
#elif defined(USE_SSE2)
			__m128i m[2];
#endif
		};
	};

	// 抑制していた警告を元に戻す。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

	// 出力用　デバッグ用。
	static std::ostream& operator<<(std::ostream& os, const EvalSum& sum)
	{
		os << "sum BKPP   = " << sum.p[0][0] << " + " << sum.p[0][1] << std::endl;
		os << "sum WKPP   = " << sum.p[1][0] << " + " << sum.p[1][1] << std::endl;
		os << "sum KK,KKP = " << sum.p[2][0] << " + " << sum.p[2][1] << std::endl;
		return os;
	}

	// 比較演算子

	static bool operator == (const EvalSum& lhs, const EvalSum rhs) { return lhs.p[0] == rhs.p[0] && lhs.p[1] == rhs.p[1] && lhs.p[2] == rhs.p[2]; }
	static bool operator != (const EvalSum& lhs, const EvalSum rhs) { return !(lhs == rhs); }

} // namespace YaneuraOu::Eval

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

    DirtyPiece& push([[maybe_unused]] const Position& pos) noexcept {
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
