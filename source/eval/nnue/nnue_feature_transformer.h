// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"
#include "../../extra/tools.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#if defined(EVAL_SFNN)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "../../types.h"
#include "nnue_common.h"
#include "nnue_architecture.h"
#include "nnue_accumulator.h"
#include "features/index_list.h"

#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
#include "features/half_ka_hm2.h"
#endif

#include <algorithm>  // std::clamp
#include <array>
#include <cstring>  // std::memset()

// Lossless packed FT weights; keep unsupported builds on the original path.
#if defined(NNUE_FT_SPARSE8) && defined(USE_AVX512) && defined(EVAL_SFNN)
#define NNUE_FT_PACKED8
#include <vector>
#endif

#if defined(USE_FINNY_TABLES)
#include <cstdint>
#include <memory>
#endif

namespace YaneuraOu {
namespace Eval::NNUE {

inline AccumulatorState* resolve_state([[maybe_unused]] const Position& pos,
                                       AccumulatorState* state) {
#if defined(EVAL_NNUE)
    return state ? state : pos.state();
#else
    assert(state);
    return state;
#endif
}

inline Accumulator& accumulator_of(AccumulatorState* state) {
#if defined(EVAL_SFNN)
    return *state;
#else
    return state->accumulator;
#endif
}
inline const Accumulator& accumulator_of(const AccumulatorState* state) {
#if defined(EVAL_SFNN)
    return *state;
#else
    return state->accumulator;
#endif
}
inline void invalidate_score([[maybe_unused]] Accumulator& accumulator) {
#if defined(EVAL_NNUE)
    accumulator.computed_score = false;
#endif
}

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// ベクトル命令が有効な場合、変数のタイルを、
// 各タイルがCPUのベクトルレジスタに収まるように、更新してリフレッシュする。
#define VECTOR

#if defined(USE_AVX512)
using vec_t = __m512i;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
#define vec_set_16(a) _mm512_set1_epi16(a)
#define vec_max_16(a, b) _mm512_max_epi16(a, b)
#define vec_min_16(a, b) _mm512_min_epi16(a, b)
#define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
#define vec_zero() _mm512_setzero_si512()
#if defined(NNUE_FT_REGS)
static_assert(NNUE_FT_REGS >= 1 && NNUE_FT_REGS <= 24, "NNUE_FT_REGS must be in [1, 24]");
// FT 2304 is the measured configuration; keep other widths unchanged.
static constexpr IndexType kNumRegs = kTransformedFeatureDimensions == 2304 ? NNUE_FT_REGS : 8;
#else
static constexpr IndexType kNumRegs = kTransformedFeatureDimensions == 2304 ? 24 : 8;
#endif

#elif defined(USE_AVX2)
using vec_t = __m256i;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
#define vec_set_16(a) _mm256_set1_epi16(a)
#define vec_max_16(a, b) _mm256_max_epi16(a, b)
#define vec_min_16(a, b) _mm256_min_epi16(a, b)
#define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
#define vec_zero() _mm256_setzero_si256()
static constexpr IndexType kNumRegs = 16;

#elif defined(USE_SSE2)
using vec_t = __m128i;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_epi16(a, b)
#define vec_sub_16(a, b) _mm_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
#define vec_set_16(a) _mm_set1_epi16(a)
#define vec_max_16(a, b) _mm_max_epi16(a, b)
#define vec_min_16(a, b) _mm_min_epi16(a, b)
#define vec_slli_16(a, b) _mm_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm_packus_epi16(a, b)
#define vec_zero() _mm_setzero_si128()
static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

#elif defined(USE_MMX)
using vec_t = __m64;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero() _mm_setzero_si64()
static constexpr IndexType kNumRegs = 8;

#elif defined(USE_NEON)
using vec_t = int16x8_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) vaddq_s16(a, b)
#define vec_sub_16(a, b) vsubq_s16(a, b)
#define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
#define vec_set_16(a) vdupq_n_s16(a)
#define vec_max_16(a, b) vmaxq_s16(a, b)
#define vec_min_16(a, b) vminq_s16(a, b)
#define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
#define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
#define vec_zero() \
	vec_t { 0 }
static constexpr IndexType kNumRegs = 16;

#else
#undef VECTOR

#endif

/*
 例) SFNN1536のときのkNumChunksの計算

┌─────────┬───────────────┬─────────────────┬────────────┐
│  SIMD            │ sizeof(vec_t)                │ / sizeof(int16)                  │ kNumChunks             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX-512          │ 64                           │ 32                               │ 1536/32=48             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX2             │ 32                           │ 16                               │ 1536/16=96             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ SSE2             │ 16                           │ 8                                │ 1536/8=192             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ NEON             │ 16                           │ 8                                │ 1536/8=192             │
└─────────┴───────────────┴─────────────────┴────────────┘
*/

constexpr IndexType MaxChunkSize = 16;

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

#if defined(VECTOR)
	//static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
	//static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
	// ⇨  AVX-512でこの制約守れないっぽ。
#endif

   public:
	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;
	using BiasType   = i16;
	using WeightType = i16;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions  = RawFeatures::kDimensions;
#if defined(USE_ELEMENT_WISE_MULTIPLY)
	static constexpr IndexType kOutputDimensions = kHalfDimensions;
#else
	static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;
#endif

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr usize kBufferSize = kOutputDimensions * sizeof(OutputType);

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr u32 GetHashValue() {
#if defined(EVAL_SFNN)
		// 学習部と整合性とるの面倒なのでSFNNのときはこれに固定しておく。
		return 0x5f134ab8u;
#else
		return RawFeatures::kHashValue ^ kOutputDimensions;
#endif
	}

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(kInputDimensions) + "->"
		       + std::to_string(kHalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
#if defined(USE_ELEMENT_WISE_MULTIPLY)
#if defined(NNUE_FT_PACKED8)
		sparse8_enabled_ = false;
#endif
		read_leb_128<BiasType>(stream, biases_, kHalfDimensions);
		read_leb_128<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#if defined(NNUE_FT_PACKED8)
		if (!stream.fail())
			prepare_sparse8_weights();
#endif

#if defined(USE_FINNY_TABLES)
		if (!stream.fail())
			++finny_generation_;
#endif
#else
		for (usize i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (usize i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
#if defined(USE_FINNY_TABLES)
		if (!stream.fail())
			++finny_generation_;
#endif
#endif
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
#if defined(NNUE_FT_PACKED8)
		if (sparse8_enabled_) {
			// Preserve the existing writer's int16 layout, including permutation/scaling.
			alignas(kCacheLineSize) WeightType column[kHalfDimensions];
			const auto* packed = reinterpret_cast<const i8*>(weights_);
			for (IndexType i = 0; i < kInputDimensions; ++i) {
				for (IndexType j = 0; j < kHalfDimensions; ++j)
					column[j] = int(packed[i * kHalfDimensions + j]) * 2;
				apply_sparse8_corrections(column, i, 1);
				stream.write(reinterpret_cast<const char*>(column), sizeof(column));
			}
		} else
#endif
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		const auto now = state;
		if (accumulator_of(now).computed_accumulation) {
			return true;
		}
		const auto prev = now->previous;
		if (prev && accumulator_of(prev).computed_accumulation) {
#if defined(NNUE_FT_PACKED8)
			if (sparse8_enabled_) update_accumulator<true>(pos, state);
			else
#endif
				update_accumulator<false>(pos, state);
			return true;
		}
#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
        if constexpr (kUseMultiPly) {
#if defined(NNUE_FT_PACKED8)
            if (sparse8_enabled_) return update_from_computed_ancestor<true>(pos, state);
#endif
            return update_from_computed_ancestor<false>(pos, state);
        }
#endif
        return false;
    }

#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
    // Merge changes directly from a computed ancestor, skipping intermediate FT states.
    // 計算済み祖先からの変化をまとめ、中間局面のFT計算を省略する。
    template<bool Packed>
    bool update_from_computed_ancestor(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
        const auto* ancestor = state;
        std::array<BonaPiece, PIECE_NUMBER_NB> old_pieces[2];
        std::array<bool, PIECE_NUMBER_NB> changed{};
        const BonaPiece* current_pieces[2] = {
          pos.eval_list()->piece_list_fb(), pos.eval_list()->piece_list_fw()};
        for (int c = 0; c < COLOR_NB; ++c)
            std::copy_n(current_pieces[c], PIECE_NUMBER_NB, old_pieces[c].begin());

        // Count state links; SFNN stores no additional state for null moves.
        // 前状態へのリンク数で数える。SFNNではnull move用の状態を積まない。
        int distance = 0;
        while (ancestor && !accumulator_of(ancestor).computed_accumulation
               && distance < USE_LAZY_EVALUATE) {
            if (!ancestor->previous)
                return false;
            const auto& dirty = ancestor->dirtyPiece;
            for (int n = 0; n < dirty.dirty_num; ++n) {
                const auto piece = dirty.pieceNo[n];
                // King moves change feature coordinates; fall back to the ordinary refresh.
                // 玉の移動で特徴量の座標系が変わる場合は、従来のrefresh経路へ戻す。
                if (piece >= PIECE_NUMBER_KING)
                    return false;
                changed[piece] = true;
                for (int c = 0; c < COLOR_NB; ++c)
                    old_pieces[c][piece] = dirty.changed_piece[n].old_piece.from[c];
            }
            ++distance;
            ancestor = ancestor->previous;
        }
        if (!ancestor || !accumulator_of(ancestor).computed_accumulation
            || distance > USE_LAZY_EVALUATE)
            return false;

        auto& accumulator = accumulator_of(state);
        for (int c = 0; c < COLOR_NB; ++c) {
            const auto king_piece = current_pieces[c][PIECE_NUMBER_KING + c];
            const Square king = static_cast<Square>((king_piece - f_king) % SQ_NB);
            Features::IndexList removed, added;
            for (int piece = 0; piece < PIECE_NUMBER_NB; ++piece) {
                if (!changed[piece] || old_pieces[c][piece] == current_pieces[c][piece])
                    continue;
                removed.push_back(Features::HalfKA_hm2<Features::Side::kFriend>::MakeIndex(
                  king, old_pieces[c][piece]));
                added.push_back(Features::HalfKA_hm2<Features::Side::kFriend>::MakeIndex(
                  king, current_pieces[c][piece]));
            }
#if defined(VECTOR)
            update_accumulator_tiled<Packed>(accumulator_of(ancestor).accumulation[c][0],
              accumulator.accumulation[c][0], [&](vec_t* acc, IndexType offset) {
                  for (auto index : removed) sub_weight_from_tile<Packed>(acc, index, offset);
                  for (auto index : added) add_weight_to_tile<Packed>(acc, index, offset);
              });
#else
            auto* destination = accumulator.accumulation[c][0];
            std::copy_n(accumulator_of(ancestor).accumulation[c][0], kHalfDimensions, destination);
            for (auto index : removed)
                for (IndexType j = 0; j < kHalfDimensions; ++j)
                    destination[j] -= weights_[kHalfDimensions * index + j];
            for (auto index : added)
                for (IndexType j = 0; j < kHalfDimensions; ++j)
                    destination[j] += weights_[kHalfDimensions * index + j];
#endif
        }
        accumulator.computed_accumulation = true;
        invalidate_score(accumulator);
        return true;
    }
#endif

#if defined(EVAL_SFNN) && defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
    // A refreshed child can also supply uncomputed parent accumulators.
    // 全計算した子局面から逆向きに差分更新し、未計算の親局面も準備する。
    template<bool Packed>
    void backward_update_incremental(const Position& pos, AccumulatorState* state) const {
        int distance = 0;
        while (state->previous && !state->previous->computed_accumulation) {
            auto* parent = state->previous;
            ++distance;
            if (distance > USE_LAZY_EVALUATE)
                break;
            const auto& dirty = state->dirtyPiece;
            bool kingMoved = false;
            for (int n = 0; n < dirty.dirty_num; ++n)
                kingMoved |= dirty.pieceNo[n] >= PIECE_NUMBER_KING;
            if (kingMoved)
                break;

            // Until a king moves, all these states use the current feature coordinates.
            // 玉が動く手の直前まで、各フレームは現在と同じ特徴量の座標系を使える。
            Features::IndexList removed[2], added[2];
            bool reset[2]{};
            RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[0], removed, added,
                                              reset, dirty);
            for (const Color perspective : COLOR) {
                assert(!reset[perspective]);
                const auto* source = state->accumulation[perspective][0];
                auto* target = parent->accumulation[perspective][0];
#if defined(VECTOR)
                update_accumulator_tiled<Packed>(source, target, [&](vec_t* acc, IndexType offset) {
                    for (auto index : added[perspective]) sub_weight_from_tile<Packed>(acc, index, offset);
                    for (auto index : removed[perspective]) add_weight_to_tile<Packed>(acc, index, offset);
                });
#else
                std::copy_n(source, kHalfDimensions, target);
                for (auto index : added[perspective])
                    for (IndexType j = 0; j < kHalfDimensions; ++j)
                        target[j] -= weights_[kHalfDimensions * index + j];
                for (auto index : removed[perspective])
                    for (IndexType j = 0; j < kHalfDimensions; ++j)
                        target[j] += weights_[kHalfDimensions * index + j];
#endif
            }
            parent->computed_accumulation = true;
            state = parent;
        }
    }
#endif

    void EnsureAccumulator(const Position& pos, bool refresh, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		if (refresh || !UpdateAccumulatorIfPossible(pos, state)) {
#if defined(NNUE_FT_PACKED8)
			if (sparse8_enabled_) refresh_accumulator<true>(pos, state);
			else
#endif
				refresh_accumulator<false>(pos, state);
#if defined(EVAL_SFNN) && defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
            if constexpr (kUseMultiPly) {
#if defined(NNUE_FT_PACKED8)
                if (sparse8_enabled_) backward_update_incremental<true>(pos, state);
                else
#endif
                    backward_update_incremental<false>(pos, state);
            }
#endif
		}
	}

#if defined(ENABLE_TEST_CMD)
	// Independent reference for checking both incremental updates and Finny.
	// Deliberately bypass all cached accumulators.
	void RefreshAccumulatorForTest(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		auto& accumulator = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (const auto perspective : COLOR) {
#if defined(NNUE_FT_PACKED8)
				if (sparse8_enabled_) refresh_accumulator_from_scratch<true>(
					accumulator.accumulation[perspective][i], active_indices[perspective], i);
				else
#endif
					refresh_accumulator_from_scratch<false>(
						accumulator.accumulation[perspective][i], active_indices[perspective], i);
			}
		}
		accumulator.computed_accumulation = true;
		invalidate_score(accumulator);
	}
#endif

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, bool refresh, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		EnsureAccumulator(pos, refresh, state);
		const auto& accumulation = accumulator_of(state).accumulation;

#if defined(USE_ELEMENT_WISE_MULTIPLY)

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
			// Packed output is sizeof(vec_t) bytes for each SIMD register
#if defined(USE_AVX512)
			constexpr IndexType OutputChunkSize = 64;
#else
			constexpr IndexType OutputChunkSize = kSimdWidth;
#endif
		static_assert((kHalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / OutputChunkSize;

		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127 * 2);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);

			constexpr int shift =
#if defined(USE_SSE2)
				7;
#else
				6;
#endif

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				const vec_t sum0a =
					vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 0], One), Zero), shift);
				const vec_t sum0b =
					vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 1], One), Zero), shift);
				const vec_t sum1a = vec_min_16(in1[j * 2 + 0], One);
				const vec_t sum1b = vec_min_16(in1[j * 2 + 1], One);

				const vec_t pa = vec_mulhi_16(sum0a, sum1a);
				const vec_t pb = vec_mulhi_16(sum0b, sum1b);

				out[j] = vec_packus_16(pa, pb);
			}

		}

#else
		constexpr int shift =
#if defined(VECTOR) && !defined(USE_SSE2)
			6;
#else
			7;
#endif

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			for (IndexType j = 0; j < kHalfDimensions / 2; ++j)
			{
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
				sum0 = std::clamp<BiasType>(sum0, 0, 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, 127 * 2);
				const int product = (int(sum0) << shift) * int(sum1);
				const int value = product >> 16;
				output[offset + j] = static_cast<OutputType>(std::clamp(value, 0, 255));
			}

		}
#endif

#else

		// 以下は旧NNUEのコード。
		// ループ本体がx86とNEONで異なる（2入力→1出力 vs 1入力→1出力）ため、
		// kNumChunksの意味自体がアーキテクチャごとに違うため、共通化しにくい。触らないことにする。

#if defined(USE_AVX512)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth * 2);
		static_assert(kHalfDimensions % (kSimdWidth * 2) == 0);
		const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
		const __m512i kZero    = _mm512_setzero_si512();

#elif defined(USE_AVX2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		constexpr int       kControl   = 0b11011000;
		const __m256i       kZero      = _mm256_setzero_si256();

#elif defined(USE_SSE2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#if defined(USE_SSE41)
		const __m128i kZero = _mm_setzero_si128();
#else  // SSE41非対応だがSSE2は使える環境
		const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		const __m64         k0x80s     = _mm_set1_pi8(-128);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		const int8x8_t      kZero      = {0};
#endif
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = kHalfDimensions * p;
#if defined(USE_AVX512)
			auto out = reinterpret_cast<__m512i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m512i sum0 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m512i sum1 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm512_add_epi16(
					    sum0,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm512_add_epi16(
					    sum1,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}
				_mm512_store_si512(&out[j], _mm512_permutexvar_epi64(
								 kControl, _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
			}

#elif defined(USE_AVX2)
			auto out = reinterpret_cast<__m256i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
					__m256i sum0 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
					__m256i sum1 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
					for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
						sum0 = _mm256_add_epi16(
							sum0,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 0]));
						sum1 = _mm256_add_epi16(
							sum1,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 1]));
					}
					_mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
									 _mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), kZero), kControl));
			}

#elif defined(USE_SSE2)
			auto out = reinterpret_cast<__m128i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m128i sum0 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m128i sum1 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm_add_epi16(sum0,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm_add_epi16(sum1,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}

				const __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
				_mm_store_si128(&out[j],
#if defined(USE_SSE41)
				                _mm_max_epi8(packedbytes, kZero)
#else  // SSE41非対応だがSSE2は使える環境
				                _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif
				);
			}

#elif defined(USE_MMX)
			// USE_MMX を config.h では現状、有効化することがないので dead code
			auto out = reinterpret_cast<__m64*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m64       sum0 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m64       sum1 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				const __m64 packedbytes = _mm_packs_pi16(sum0, sum1);
				out[j]                  = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
			}

#elif defined(USE_NEON)
			const auto out = reinterpret_cast<int8x8_t*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				int16x8_t sum = reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][0])[j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum = vaddq_s16(sum, reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][i])[j]);
				}
				out[j] = vmax_s8(vqmovn_s16(sum), kZero);
			}
#else
			for (IndexType j = 0; j < kHalfDimensions; ++j) {
				BiasType sum = accumulation[perspectives[p]][0][j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum += accumulation[perspectives[p]][i][j];
				}
				output[offset + j] = static_cast<OutputType>(std::clamp<int>(sum, 0, 127));
			}
#endif
		}
#if defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		_mm_empty();
#endif
#endif
	}

   private:
	static void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_set_epi32 packs in the order [15 11 7 3 14 10 6 2 13 9 5 1 12 8 4 0]
		uint64_t tmp0 = v[4], tmp1 = v[5];
		v[4] = v[6], v[5] = v[7];
		v[6] = tmp0, v[7] = tmp1;
		tmp0 = v[8], tmp1 = v[9];
		v[8] = v[12], v[9] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // _mm256_set_epi32 packs in the order [7 3 6 2 5 1 4 0]
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	static void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = v[8], v[5] = v[9];
		v[8] = tmp0, v[9] = tmp1;
		tmp0 = v[6], tmp1 = v[7];
		v[6] = v[12], v[7] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases_[0]));
		for (IndexType i = 0; i < kHalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
			order_fn(&b[i]);

		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			uint64_t* w =
				reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights_[j * kHalfDimensions]));
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights_[j * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

#if defined(NNUE_FT_PACKED8)
	// The signed low byte stores half the already-scaled weight. Only values
	// outside [-128,127] need a (position, residual) correction. The first half
	// of weights_ holds bytes; its second half holds correction records.
	// Allocation size stays unchanged; the frequently read weight data is halved.
	struct Sparse8Correction {
		u16 position;
		i16 residual;
	};
	static_assert(sizeof(Sparse8Correction) == 4);
	static_assert(kHalfDimensions == 2304, "NNUE_FT_SPARSE8 currently supports FT2304 only");
	static constexpr usize kSparse8Bytes = usize(kHalfDimensions) * kInputDimensions;

	void prepare_sparse8_weights() {
		usize count = 0;
		for (usize i = 0; i < kSparse8Bytes; ++i) {
			const int value = weights_[i] / 2;
			count += value < -128 || value > 127;
		}
		// Check before overwriting any int16 weights. Keep outliers below
		// 1/1024 to bound scalar fixups (also well within the available space).
		// Other models retain the original implementation and file format.
		if (count > kSparse8Bytes / 1024)
			return;
		std::vector<Sparse8Correction> corrections;
		corrections.reserve(count);
		auto* packed = reinterpret_cast<i8*>(weights_);
		for (IndexType i = 0; i < kInputDimensions; ++i) {
			sparse8_offsets_[i] = static_cast<u32>(corrections.size());
			for (IndexType j = 0; j < kHalfDimensions; ++j) {
				const int value = weights_[usize(i) * kHalfDimensions + j];
				const auto base = static_cast<i8>(value / 2);
				// Forward packing cannot overwrite the next unread int16.
				packed[usize(i) * kHalfDimensions + j] = base;
				const int residual = value - int(base) * 2;
				if (residual)
					corrections.push_back({u16(j), i16(residual)});
			}
		}
		sparse8_offsets_[kInputDimensions] = static_cast<u32>(corrections.size());
		if (!corrections.empty())
			std::memcpy(packed + kSparse8Bytes, corrections.data(), corrections.size() * sizeof(Sparse8Correction));
		sparse8_enabled_ = true;
	}

#if !defined(NO_PREFETCH)
	// 実際に加減算する列の先頭だけを先読みする。列全体には命令を発行しない。
	void prefetch_sparse8_columns(const Features::IndexList& indices) const {
		const auto* base = reinterpret_cast<const i8*>(weights_);
		for (const auto index : indices)
			_mm_prefetch(reinterpret_cast<const char*>(base + usize(kHalfDimensions) * index), _MM_HINT_T2);
	}
#endif

	void apply_sparse8_corrections(BiasType* destination, IndexType index, int sign) const {
		const auto* bytes = reinterpret_cast<const unsigned char*>(weights_) + kSparse8Bytes;
		for (auto i = sparse8_offsets_[index]; i < sparse8_offsets_[index + 1]; ++i) {
			Sparse8Correction item;
			// memcpy avoids aliasing/lifetime assumptions about the int16 storage.
			std::memcpy(&item, bytes + usize(i) * sizeof(item), sizeof(item));
			destination[item.position] += sign * item.residual;
		}
	}

	template <bool Add>
	void change_sparse8_tile(vec_t* acc, IndexType index, IndexType offset) const {
		// A final callback after the SIMD pass applies the sparse corrections
		// directly to the destination, once per column rather than once per tile.
		if (offset == kHalfDimensions) {
			apply_sparse8_corrections(reinterpret_cast<BiasType*>(acc), index, Add ? 1 : -1);
			return;
		}
		const auto* packed = reinterpret_cast<const i8*>(weights_) + usize(kHalfDimensions) * index + offset;
		for (IndexType k = 0; k < kTileRegs; ++k) {
			const auto w = _mm512_slli_epi16(_mm512_cvtepi8_epi16(
				_mm256_load_si256(reinterpret_cast<const __m256i*>(packed + k * 32))), 1);
			if constexpr (Add) acc[k] = vec_add_16(acc[k], w);
			else acc[k] = vec_sub_16(acc[k], w);
		}
	}
#endif

#if defined(VECTOR)
	// 変更された各特徴量ごとにaccumulator全体を読み書きするのを避けるため、
	// SIMDレジスタに収まるタイル単位で差分をまとめて適用する。
	static constexpr IndexType kVectorHeight = sizeof(vec_t) / sizeof(BiasType);
	static_assert(kHalfDimensions % kVectorHeight == 0, "kVectorHeight must divide kHalfDimensions");
	static constexpr IndexType kNumVectorChunks = kHalfDimensions / kVectorHeight;
	static constexpr IndexType kTileRegs = [] {
		IndexType regs = std::min(kNumRegs, kNumVectorChunks);
		while (kNumVectorChunks % regs != 0)
			--regs;
		return regs;
	}();
	static constexpr IndexType kTileHeight = kTileRegs * kVectorHeight;

#if defined(NNUE_FT_SPECIALIZED) || defined(NNUE_FT_PACKED8)
	template <IndexType Removed, IndexType Added, bool Packed>
	void update_accumulator_fixed(const BiasType* source, BiasType* destination,
	                              const Features::IndexList& removed,
	                              const Features::IndexList& added) const {
#if defined(NNUE_FT_PACKED8)
		if constexpr (Packed) {
			for (IndexType offset = 0; offset < kHalfDimensions; offset += kTileHeight) {
				vec_t acc[kTileRegs];
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_load(reinterpret_cast<const vec_t*>(source + offset) + k);
				for (IndexType n = 0; n < Removed; ++n)
					change_sparse8_tile<false>(acc, removed[n], offset);
				for (IndexType n = 0; n < Added; ++n)
					change_sparse8_tile<true>(acc, added[n], offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					vec_store(reinterpret_cast<vec_t*>(destination + offset) + k, acc[k]);
			}
			for (IndexType n = 0; n < Removed; ++n)
				apply_sparse8_corrections(destination, removed[n], -1);
			for (IndexType n = 0; n < Added; ++n)
				apply_sparse8_corrections(destination, added[n], 1);
			return;
		}
#endif
		const WeightType* remove_columns[Removed];
		const WeightType* add_columns[Added];
		for (IndexType n = 0; n < Removed; ++n)
			remove_columns[n] = weights_ + kHalfDimensions * removed[n];
		for (IndexType n = 0; n < Added; ++n)
			add_columns[n] = weights_ + kHalfDimensions * added[n];
		for (IndexType offset = 0; offset < kHalfDimensions; offset += kTileHeight) {
			vec_t acc[kTileRegs];
			const auto* src = reinterpret_cast<const vec_t*>(source + offset);
			for (IndexType k = 0; k < kTileRegs; ++k)
				acc[k] = vec_load(src + k);
			for (IndexType n = 0; n < Removed; ++n) {
				const auto* column = reinterpret_cast<const vec_t*>(remove_columns[n] + offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_sub_16(acc[k], vec_load(column + k));
			}
			for (IndexType n = 0; n < Added; ++n) {
				const auto* column = reinterpret_cast<const vec_t*>(add_columns[n] + offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_add_16(acc[k], vec_load(column + k));
			}
			auto* dst = reinterpret_cast<vec_t*>(destination + offset);
			for (IndexType k = 0; k < kTileRegs; ++k)
				vec_store(dst + k, acc[k]);
		}
	}
#endif

	template <bool Packed, typename ApplyChanges>
	void update_accumulator_tiled(
		const BiasType* source, BiasType* destination,
		ApplyChanges apply_changes) const {
		for (IndexType tile_offset = 0; tile_offset < kHalfDimensions; tile_offset += kTileHeight) {
			vec_t acc[kTileRegs];

			if (source) {
				const auto* source_tile = reinterpret_cast<const vec_t*>(source + tile_offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_load(source_tile + k);
			} else {
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_zero();
			}

			apply_changes(acc, tile_offset);

			auto* destination_tile = reinterpret_cast<vec_t*>(destination + tile_offset);
			for (IndexType k = 0; k < kTileRegs; ++k)
				vec_store(destination_tile + k, acc[k]);
		}

#if defined(NNUE_FT_PACKED8)
		if constexpr (Packed)
			apply_changes(reinterpret_cast<vec_t*>(destination), kHalfDimensions);
#endif
	}

	template <bool Packed = false>
	void add_weight_to_tile(vec_t* acc, IndexType index, IndexType tile_offset) const {
#if defined(NNUE_FT_PACKED8)
		if constexpr (Packed) {
			change_sparse8_tile<true>(acc, index, tile_offset);
			return;
		}
#endif

		const auto* column = reinterpret_cast<const vec_t*>(
			&weights_[kHalfDimensions * index + tile_offset]);
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_add_16(acc[k], vec_load(column + k));
	}

	template <bool Packed = false>
	void sub_weight_from_tile(vec_t* acc, IndexType index, IndexType tile_offset) const {
#if defined(NNUE_FT_PACKED8)
		if constexpr (Packed) {
			change_sparse8_tile<false>(acc, index, tile_offset);
			return;
		}
#endif

		const auto* column = reinterpret_cast<const vec_t*>(
			&weights_[kHalfDimensions * index + tile_offset]);
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_sub_16(acc[k], vec_load(column + k));
	}
#endif

	// StockfishのAccumulatorCaches(Finny Tables)と同じ発想。
	// 王位置ごとにrefresh済みaccumulatorを持ち、次回はactive featureの差分だけを適用する。
#if defined(USE_FINNY_TABLES)
	static constexpr bool kUseFinnyTables = kHalfDimensions <= 4096;

	struct alignas(kCacheLineSize) FinnyEntry {
		BiasType accumulation[kHalfDimensions];
		Features::IndexList active_indices;
		bool initialized = false;
	};

	struct FinnyCache {
		using TriggerEntries = std::array<std::array<FinnyEntry, SQ_NB>, COLOR_NB>;

		const FeatureTransformer* owner = nullptr;
		std::uint64_t generation = 0;
		std::array<TriggerEntries, kRefreshTriggers.size()> entries;

		void reset(const FeatureTransformer* new_owner, std::uint64_t new_generation) {
			owner = new_owner;
			generation = new_generation;
			for (auto& trigger_entries : entries)
				for (auto& perspective_entries : trigger_entries)
					for (auto& entry : perspective_entries)
						entry.initialized = false;
		}
	};

	FinnyCache& finny_cache() const {
		static thread_local std::unique_ptr<FinnyCache> cache;
		if (!cache)
			cache = std::make_unique<FinnyCache>();
		if (cache->owner != this || cache->generation != finny_generation_)
			cache->reset(this, finny_generation_);
		return *cache;
	}

	static Square finny_bucket_square(
		const Position& pos, Features::TriggerEvent trigger, Color perspective) {
		switch (trigger) {
		case Features::TriggerEvent::kFriendKingMoved:
			return pos.square<KING>(perspective);
		case Features::TriggerEvent::kEnemyKingMoved:
			return pos.square<KING>(~perspective);
		case Features::TriggerEvent::kAnyKingMoved:
			return pos.square<KING>(perspective);
		default:
			return SQ_ZERO;
		}
	}

#if defined(NNUE_FINNY_CANCEL)
	static void cancel_common_indices(Features::IndexList& removed, Features::IndexList& added) {
		if (removed.size() < 2 || added.size() < 2)
			return;
		// Multiset difference: preserve repeated features (e.g. missing pieces).
		std::sort(removed.begin(), removed.end());
		std::sort(added.begin(), added.end());
		usize r = 0, a = 0, r_out = 0, a_out = 0;
		while (r < removed.size() && a < added.size()) {
			if (removed[r] == added[a]) { ++r; ++a; }
			else if (removed[r] < added[a]) removed[r_out++] = removed[r++];
			else added[a_out++] = added[a++];
		}
		while (r < removed.size()) removed[r_out++] = removed[r++];
		while (a < added.size()) added[a_out++] = added[a++];
		removed.resize(r_out);
		added.resize(a_out);
	}
#endif

	static void make_index_diff(
		const Features::IndexList& old_active,
		const Features::IndexList& new_active,
		Features::IndexList& removed,
		Features::IndexList& added) {
		if (old_active.size() == new_active.size()) {
			for (usize i = 0; i < old_active.size(); ++i) {
				if (old_active[i] == new_active[i])
					continue;
				removed.push_back(old_active[i]);
				added.push_back(new_active[i]);
			}
#if defined(NNUE_FINNY_CANCEL)
			cancel_common_indices(removed, added);
#endif
			return;
		}

		bool old_matched[RawFeatures::kMaxActiveDimensions] = {};
		bool new_matched[RawFeatures::kMaxActiveDimensions] = {};

		for (usize oi = 0; oi < old_active.size(); ++oi) {
			for (usize ni = 0; ni < new_active.size(); ++ni) {
				if (!new_matched[ni] && old_active[oi] == new_active[ni]) {
					old_matched[oi] = true;
					new_matched[ni] = true;
					break;
				}
			}
		}

		for (usize oi = 0; oi < old_active.size(); ++oi)
			if (!old_matched[oi])
				removed.push_back(old_active[oi]);
		for (usize ni = 0; ni < new_active.size(); ++ni)
			if (!new_matched[ni])
				added.push_back(new_active[ni]);
	}

	static void copy_index_list(
		Features::IndexList& destination,
		const Features::IndexList& source) {
		destination.resize(source.size());
		for (usize i = 0; i < source.size(); ++i)
			destination[i] = source[i];
	}
#endif

	template <bool Packed = false>
	void refresh_accumulator_from_scratch(
		BiasType* current, const Features::IndexList& active_indices, IndexType trigger_index) const {
#if defined(VECTOR)
		const auto* source = trigger_index == 0 ? biases_ : nullptr;
		update_accumulator_tiled<Packed>(
			source, current,
			[&](vec_t* acc, IndexType tile_offset) {
				for (const auto index : active_indices)
					add_weight_to_tile<Packed>(acc, index, tile_offset);
			});
#else
		if (trigger_index == 0)
			std::memcpy(current, biases_, kHalfDimensions * sizeof(BiasType));
		else
			std::memset(current, 0, kHalfDimensions * sizeof(BiasType));

		for (const auto index : active_indices) {
			const IndexType offset = kHalfDimensions * index;
			for (IndexType j = 0; j < kHalfDimensions; ++j)
				current[j] += weights_[offset + j];
		}
#endif
	}

#if defined(USE_FINNY_TABLES)
#if defined(VECTOR)
	template <bool Packed, typename ApplyChanges>
	void update_accumulator_tiled_to_two(
		const BiasType* source, BiasType* destination0, BiasType* destination1,
		ApplyChanges apply_changes) const {
		for (IndexType tile_offset = 0; tile_offset < kHalfDimensions; tile_offset += kTileHeight) {
			vec_t acc[kTileRegs];

			if (source) {
				const auto* source_tile = reinterpret_cast<const vec_t*>(source + tile_offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_load(source_tile + k);
			} else {
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_zero();
			}

			apply_changes(acc, tile_offset);

			auto* destination0_tile = reinterpret_cast<vec_t*>(destination0 + tile_offset);
			auto* destination1_tile = reinterpret_cast<vec_t*>(destination1 + tile_offset);
			for (IndexType k = 0; k < kTileRegs; ++k) {
				vec_store(destination0_tile + k, acc[k]);
				vec_store(destination1_tile + k, acc[k]);
			}
		}

#if defined(NNUE_FT_PACKED8)
		if constexpr (Packed) {
			apply_changes(reinterpret_cast<vec_t*>(destination0), kHalfDimensions);
			if (destination1 != destination0)
				apply_changes(reinterpret_cast<vec_t*>(destination1), kHalfDimensions);
		}
#endif
	}
#endif

	template <bool Packed = false>
	void refresh_accumulator_using_finny_entry(
		BiasType* current,
		FinnyEntry& entry,
		const Features::IndexList& active_indices,
		IndexType trigger_index) const {
		Features::IndexList removed_indices, added_indices;
		if (entry.initialized)
			make_index_diff(entry.active_indices, active_indices, removed_indices, added_indices);
		bool rebuild = !entry.initialized;
#if defined(USE_FINNY_RESET)
		// Avoid reading more weight columns than rebuilding from the bias.
		rebuild = rebuild || removed_indices.size() + added_indices.size() >= active_indices.size();
#endif

#if defined(NNUE_FT_PACKED8) && !defined(NO_PREFETCH)
		if constexpr (Packed) {
			if (rebuild)
				prefetch_sparse8_columns(active_indices);
			else {
				prefetch_sparse8_columns(removed_indices);
				prefetch_sparse8_columns(added_indices);
			}
		}
#endif
		if (rebuild) {
#if defined(VECTOR)
			const auto* source = trigger_index == 0 ? biases_ : nullptr;
			update_accumulator_tiled_to_two<Packed>(
				source, entry.accumulation, current,
				[&](vec_t* acc, IndexType tile_offset) {
					for (const auto index : active_indices)
						add_weight_to_tile<Packed>(acc, index, tile_offset);
				});
#else
			refresh_accumulator_from_scratch<Packed>(entry.accumulation, active_indices, trigger_index);
			std::memcpy(current, entry.accumulation, kHalfDimensions * sizeof(BiasType));
#endif
			copy_index_list(entry.active_indices, active_indices);
			entry.initialized = true;
		} else {
#if defined(VECTOR)
			update_accumulator_tiled_to_two<Packed>(
				entry.accumulation, entry.accumulation, current,
				[&](vec_t* acc, IndexType tile_offset) {
					for (const auto index : removed_indices)
						sub_weight_from_tile<Packed>(acc, index, tile_offset);
					for (const auto index : added_indices)
						add_weight_to_tile<Packed>(acc, index, tile_offset);
				});
#else
			for (const auto index : removed_indices) {
				const IndexType offset = kHalfDimensions * index;
				for (IndexType j = 0; j < kHalfDimensions; ++j)
					entry.accumulation[j] -= weights_[offset + j];
			}
			for (const auto index : added_indices) {
				const IndexType offset = kHalfDimensions * index;
				for (IndexType j = 0; j < kHalfDimensions; ++j)
					entry.accumulation[j] += weights_[offset + j];
			}
#endif
			copy_index_list(entry.active_indices, active_indices);
#if !defined(VECTOR)
			std::memcpy(current, entry.accumulation, kHalfDimensions * sizeof(BiasType));
#endif
		}
	}

	template <bool Packed = false>
	void refresh_accumulator_with_finny_cache(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		auto& cache = finny_cache();

		auto& accumulator = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			const auto trigger = kRefreshTriggers[i];
			RawFeatures::AppendActiveIndices(pos, trigger, active_indices);
			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
				const Square bucket = finny_bucket_square(pos, trigger, perspective);
				auto& entry = cache.entries[i][perspective][bucket];
				refresh_accumulator_using_finny_entry<Packed>(
					accumulator.accumulation[perspective][i], entry,
					active_indices[perspective], i);
			}
		}

		accumulator.computed_accumulation = true;
		invalidate_score(accumulator);
	}
#endif

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	template <bool Packed = false>
	void refresh_accumulator(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
#if defined(USE_FINNY_TABLES)
		if constexpr (kUseFinnyTables) {
			refresh_accumulator_with_finny_cache<Packed>(pos, state);
			return;
		}
#endif

		auto& accumulator = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
#if defined(VECTOR)
				auto* current = accumulator.accumulation[perspective][i];
				refresh_accumulator_from_scratch<Packed>(current, active_indices[perspective], i);
#else
				refresh_accumulator_from_scratch<Packed>(
					accumulator.accumulation[perspective][i], active_indices[perspective], i);
#endif
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		invalidate_score(accumulator);
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	template <bool Packed = false>
	void update_accumulator(const Position& pos, AccumulatorState* state) const {
        state = resolve_state(pos, state);
		const auto& prev_accumulator = accumulator_of(state->previous);
		auto&      accumulator      = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2]{};
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset, state->dirtyPiece);
#if defined(NNUE_FT_PACKED8) && !defined(NO_PREFETCH)
			if constexpr (Packed) {
				for (int c = 0; c < COLOR_NB; ++c) {
					if (!reset[c]) {
						prefetch_sparse8_columns(removed_indices[c]);
						prefetch_sparse8_columns(added_indices[c]);
					}
				}
			}
#endif

			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
#if defined(USE_FINNY_TABLES) && defined(USE_FINNY_RESET)
				if constexpr (kUseFinnyTables) {
					if (reset[perspective]) {
						auto& cache = finny_cache();
						const auto bucket = finny_bucket_square(pos, kRefreshTriggers[i], perspective);
						auto& entry = cache.entries[i][perspective][bucket];
						// On reset, added_indices contains every active feature.
						// The other perspective keeps its ordinary incremental update.
						refresh_accumulator_using_finny_entry<Packed>(
							accumulator.accumulation[perspective][i], entry,
							added_indices[perspective], i);
						continue;
					}
				}
#endif
#if defined(VECTOR)
				auto* current = accumulator.accumulation[perspective][i];
				if (reset[perspective]) {
					const auto* source = i == 0 ? biases_ : nullptr;
					update_accumulator_tiled<Packed>(
						source, current,
						[&](vec_t* acc, IndexType tile_offset) {
							for (const auto index : added_indices[perspective])
								add_weight_to_tile<Packed>(acc, index, tile_offset);
						});
				} else {
#if defined(NNUE_FT_SPECIALIZED) || defined(NNUE_FT_PACKED8)
#if defined(NNUE_FT_PACKED8) && !defined(NNUE_FT_SPECIALIZED)
					if constexpr (Packed) {
#endif
					const auto& removed = removed_indices[perspective];
					const auto& added = added_indices[perspective];
					const auto* source = prev_accumulator.accumulation[perspective][i];
					if (removed.size() == 1 && added.size() == 1) {
						update_accumulator_fixed<1, 1, Packed>(source, current, removed, added);
						continue;
					}
					if (removed.size() == 2 && added.size() == 2) {
						update_accumulator_fixed<2, 2, Packed>(source, current, removed, added);
						continue;
					}
#if defined(NNUE_FT_PACKED8) && !defined(NNUE_FT_SPECIALIZED)
					}
#endif
#endif
					update_accumulator_tiled<Packed>(
						prev_accumulator.accumulation[perspective][i],
						current,
						[&](vec_t* acc, IndexType tile_offset) {
							for (const auto index : removed_indices[perspective])
								sub_weight_from_tile<Packed>(acc, index, tile_offset);
							for (const auto index : added_indices[perspective])
								add_weight_to_tile<Packed>(acc, index, tile_offset);
						});
				}
#else
				if (reset[perspective]) {
					if (i == 0) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
					}
				}
				// Difference calculation for features that changed from 0 to 1
				// 0から1に変化した特徴量に関する差分計算
				for (const auto index : added_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;
					for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
					}
				}
#endif
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		invalidate_score(accumulator);
	}

	// parameter type
	// パラメータの型

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kHalfDimensions * kInputDimensions];
#if defined(NNUE_FT_PACKED8)
	u32 sparse8_offsets_[kInputDimensions + 1];
	bool sparse8_enabled_ = false;
#endif

#if defined(USE_FINNY_TABLES)
	std::uint64_t finny_generation_ = 0;
#endif
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#if defined(NNUE_FT_PACKED8)
#undef NNUE_FT_PACKED8
#endif

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
