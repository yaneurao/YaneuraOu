// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iosfwd>

#if defined(EVAL_SFNN)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "simd.h"
#include "features/index_list.h"

#if defined(USE_FINNY_TABLES) || (defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1)
#include "features/half_ka_hm2.h"
#endif


// Lossless packed FT weights; keep unsupported builds on the original path.
#if defined(NNUE_FT_SPARSE8) && (defined(USE_AVX512) || defined(USE_AVX2)) && defined(EVAL_SFNN)
#define NNUE_FT_PACKED8
#include <limits>
#include <vector>
#endif


namespace YaneuraOu {
namespace Eval::NNUE {

/*
 例) FT1536のときのkNumChunksの計算

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

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
#if defined(VECTOR)
    using vec_t = SIMD::vec_t;
#endif
  // SIMD width (in bytes)
  // SIMD幅（バイト単位）
  // Stockfishでは削除済みだが、従来NNUEの変換ではこの幅を使う。
  // AVX512でも値は32。sizeof(vec_t)へ置換すると既存の変換処理が変わるため維持する。
  #if defined(USE_AVX2)
  static constexpr usize kSimdWidth = 32;
  #elif defined(USE_SSE2)
  static constexpr usize kSimdWidth = 16;
  #elif defined(USE_MMX)
  static constexpr usize kSimdWidth = 8;

  #elif defined(USE_NEON)
  static constexpr usize kSimdWidth = 16;
  #elif defined(USE_WASM_SIMD)
  static constexpr usize kSimdWidth = 16;
  #endif

   public:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType HalfDimensions = kTransformedFeatureDimensions;

	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;
	using BiasType   = Eval::NNUE::BiasType;
	using WeightType = Eval::NNUE::WeightType;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType InputDimensions  = RawFeatures::Dimensions;
#if defined(USE_ELEMENT_WISE_MULTIPLY)
	static constexpr IndexType OutputDimensions = HalfDimensions;
#else
	static constexpr IndexType OutputDimensions = HalfDimensions * 2;
#endif

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr usize BufferSize = OutputDimensions * sizeof(OutputType);

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr u32 get_hash_value() {
#if defined(EVAL_SFNN)
		// 学習部と整合性とるの面倒なのでSFNNのときはこれに固定しておく。
		return 0x5f134ab8u;
#else
		return RawFeatures::HashValue ^ OutputDimensions;
#endif
	}

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(InputDimensions) + "->"
		       + std::to_string(HalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	bool read_parameters(std::istream& stream) {
#if defined(USE_ELEMENT_WISE_MULTIPLY)
#if defined(NNUE_FT_PACKED8)
		sparse8_enabled_ = false;
#endif
		read_leb_128<BiasType>(stream, biases, HalfDimensions);
		read_leb_128<WeightType>(stream, weights, HalfDimensions * InputDimensions);

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#if defined(NNUE_FT_PACKED8)
		if (!stream.fail())
			prepare_sparse8_weights();
#endif

#else
		read_little_endian(stream, biases, HalfDimensions);
		read_little_endian(stream, weights, HalfDimensions * InputDimensions);
#endif
		return !stream.fail();
	}

	// Write network parameters
	// パラメータを書き込む
	bool write_parameters(std::ostream& stream) const {
		write_little_endian(stream, biases, HalfDimensions);
#if defined(NNUE_FT_PACKED8)
		if (sparse8_enabled_) {
			// Preserve the existing writer's int16 layout, including permutation/scaling.
			alignas(CacheLineSize) WeightType column[HalfDimensions];
			const auto* packed = reinterpret_cast<const i8*>(weights);
			for (IndexType i = 0; i < InputDimensions; ++i) {
				for (IndexType j = 0; j < HalfDimensions; ++j)
					column[j] = int(packed[i * HalfDimensions + j]) * 2;
				apply_sparse8_corrections(column, i, 1);
				write_little_endian(stream, column, HalfDimensions);
			}
		} else
#endif
		write_little_endian(stream, weights, HalfDimensions * InputDimensions);
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool update_accumulator_if_possible(const Position& pos, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
		const auto now = state;
		if (accumulator_of(now).computed_accumulation) {
			return true;
		}
		const auto prev = now->previous;
		if (prev && accumulator_of(prev).computed_accumulation) {
#if defined(NNUE_FT_PACKED8)
			if (sparse8_enabled_) update_accumulator<true>(pos, state, refreshTable);
			else
#endif
				update_accumulator<false>(pos, state, refreshTable);
			return true;
		}
#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
        if constexpr (kUseMultiPly) {
#if defined(NNUE_FT_PACKED8)
            if (sparse8_enabled_) return update_from_computed_ancestor<true>(pos, state, refreshTable);
#endif
            return update_from_computed_ancestor<false>(pos, state, refreshTable);
        }
#endif
        return false;
    }

#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
    // Merge changes directly from a computed ancestor, skipping intermediate FT states.
    // 計算済み祖先からの変化をまとめ、中間局面のFT計算を省略する。
    template<bool Packed>
    bool update_from_computed_ancestor(const Position& pos, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
        const auto* ancestor = state;
        // Find a usable ancestor before converting any feature indices.
        // 利用できる祖先を先に探し、差分更新できない場合の特徴量変換を省く。
        int distance = 0;
        bool reset[COLOR_NB]{};
        while (ancestor && !accumulator_of(ancestor).computed_accumulation
               && distance < USE_LAZY_EVALUATE) {
            if (!ancestor->previous)
                return false;
            const auto& dirty = ancestor->dirtyPiece;
            if (dirty.king_moved()) {
                reset[color_of(dirty.pc)] = true;
                if (reset[BLACK] && reset[WHITE])
                    return false;
            }
            ++distance;
            ancestor = ancestor->previous;
        }
        if (!ancestor || !accumulator_of(ancestor).computed_accumulation)
            return false;

        // Cancel matching columns in FT coordinates, without a BonaPiece round trip.
        // BonaPieceを経由せずFTの座標系で相殺する。中間で移動した駒の列は読み込まない。
        std::array<IndexType, 2 * USE_LAZY_EVALUATE> removedPieces[2], addedPieces[2];
        int removedCount[2]{}, addedCount[2]{};
        for (const Color color : COLOR) {
            if (reset[color]) continue;
            const Square king = color == BLACK ? pos.square<KING>(color)
                                               : Inv(pos.square<KING>(color));
            const typename Features::HalfKA_hm2<Features::Side::kFriend>::IndexContext indices(king, color);
            auto record = [](auto& opposite, int& oppositeCount, IndexType value,
                             auto& own, int& ownCount) {
                for (int i = 0; i < oppositeCount; ++i)
                    if (opposite[i] == value) {
                        opposite[i] = opposite[--oppositeCount];
                        return;
                    }
                own[ownCount++] = value;
            };
            auto remove = [&](IndexType index) {
                record(addedPieces[color], addedCount[color], index,
                       removedPieces[color], removedCount[color]);
            };
            auto add = [&](IndexType index) {
                record(removedPieces[color], removedCount[color], index,
                       addedPieces[color], addedCount[color]);
            };
            for (auto* current = state; current != ancestor; current = current->previous) {
                const auto& dirty = current->dirtyPiece;
                if (!dirty.count()) continue;
                remove(dirty.from == SQ_NONE
                  ? indices.hand(color_of(dirty.pc), raw_type_of(dirty.pc), dirty.handSlot)
                  : indices.board(dirty.pc, dirty.from));
                add(indices.board(dirty.after, dirty.to));
                if (dirty.captured != NO_PIECE) {
                    remove(indices.board(dirty.captured, dirty.to));
                    add(indices.hand(color_of(dirty.pc), raw_type_of(dirty.captured), dirty.handSlot));
                }
            }
        }

        auto& accumulator = accumulator_of(state);
        for (int c = 0; c < COLOR_NB; ++c) {
            const auto color = Color(c);
            // A king move invalidates only that king's perspective, as in Stockfish.
            // Stockfishと同様に、玉を動かした側だけを更新し直し、他方は祖先の累積値を使う。
            if (reset[c]) {
#if defined(USE_FINNY_TABLES) && defined(USE_FINNY_RESET) && defined(VECTOR)
                if (kUseFinnyTables && !refreshTable.empty()) {
                    auto& cache = *refreshTable;
                    auto& entry = cache.entries[0][c][pos.square<KING>(color)];
                    refresh_from_position<Packed>(pos, color, accumulator.accumulation[c][0], entry, 0);
                    continue;
                }
#endif
                Features::IndexList active[2];
                RawFeatures::append_active_indices(pos, kRefreshTriggers[0], active);
                refresh_accumulator_from_scratch<Packed>(accumulator.accumulation[c][0], active[c], 0);
                continue;
            }
            Features::IndexList removed, added;
            for (int i = 0; i < removedCount[c]; ++i)
                removed.push_back(removedPieces[c][i]);
            for (int i = 0; i < addedCount[c]; ++i)
                added.push_back(addedPieces[c][i]);
#if defined(VECTOR)
            update_accumulator_tiled<Packed>(accumulator_of(ancestor).accumulation[c][0],
              accumulator.accumulation[c][0], [&](vec_t* acc, IndexType offset) {
                  for (auto index : removed) sub_weight_from_tile<Packed>(acc, index, offset);
                  for (auto index : added) add_weight_to_tile<Packed>(acc, index, offset);
              });
#else
            auto* destination = accumulator.accumulation[c][0];
            std::copy_n(accumulator_of(ancestor).accumulation[c][0], HalfDimensions, destination);
            for (auto index : removed)
                for (IndexType j = 0; j < HalfDimensions; ++j)
                    destination[j] -= weights[HalfDimensions * index + j];
            for (auto index : added)
                for (IndexType j = 0; j < HalfDimensions; ++j)
                    destination[j] += weights[HalfDimensions * index + j];
#endif
        }
        accumulator.computed_accumulation = true;
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
            const bool kingMoved = dirty.king_moved();
            if (kingMoved)
                break;

            // Until a king moves, all these states use the current feature coordinates.
            // 玉が動く手の直前まで、各フレームは現在と同じ特徴量の座標系を使える。
            Features::IndexList removed[2], added[2];
            bool reset[2]{};
            RawFeatures::append_changed_indices(pos, kRefreshTriggers[0], removed, added,
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
                std::copy_n(source, HalfDimensions, target);
                for (auto index : added[perspective])
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        target[j] -= weights[HalfDimensions * index + j];
                for (auto index : removed[perspective])
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        target[j] += weights[HalfDimensions * index + j];
#endif
            }
            parent->computed_accumulation = true;
            state = parent;
        }
    }
#endif

    void ensure_accumulator(const Position& pos, bool refresh, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
		if (refresh || !update_accumulator_if_possible(pos, state, refreshTable)) {
#if defined(NNUE_FT_PACKED8)
			if (sparse8_enabled_) refresh_accumulator<true>(pos, state, refreshTable);
			else
#endif
				refresh_accumulator<false>(pos, state, refreshTable);
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
	void refresh_accumulator_for_test(const Position& pos, AccumulatorState* state) const {
        assert(state);
		auto& accumulator = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::append_active_indices(pos, kRefreshTriggers[i], active_indices);
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
	}
#endif

	// Convert input features
	// 入力特徴量を変換する
	void transform(const Position& pos, OutputType* output, bool refresh, AccumulatorState* state,
                   AccumulatorCaches& refreshTable) const {
        assert(state);
		ensure_accumulator(pos, refresh, state, refreshTable);
		const auto& accumulation = accumulator_of(state).accumulation;

#if defined(USE_ELEMENT_WISE_MULTIPLY)

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
			// Packed output is sizeof(vec_t) bytes for each SIMD register
			constexpr IndexType OutputChunkSize = MaxChunkSize;
		static_assert((HalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127 * 2);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (HalfDimensions / 2) * p;

			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][HalfDimensions / 2]));
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
			const IndexType offset = (HalfDimensions / 2) * p;

			for (IndexType j = 0; j < HalfDimensions / 2; ++j)
			{
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + HalfDimensions / 2];
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
		constexpr IndexType kNumChunks = HalfDimensions / (kSimdWidth * 2);
		static_assert(HalfDimensions % (kSimdWidth * 2) == 0);
		const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
		const __m512i kZero    = _mm512_setzero_si512();

#elif defined(USE_AVX2)
		constexpr IndexType kNumChunks = HalfDimensions / kSimdWidth;
		constexpr int       kControl   = 0b11011000;
		const __m256i       kZero      = _mm256_setzero_si256();

#elif defined(USE_SSE2)
		constexpr IndexType kNumChunks = HalfDimensions / kSimdWidth;
#if defined(USE_SSE41)
		const __m128i kZero = _mm_setzero_si128();
#else  // SSE41非対応だがSSE2は使える環境
		const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		constexpr IndexType kNumChunks = HalfDimensions / kSimdWidth;
		const __m64         k0x80s     = _mm_set1_pi8(-128);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks = HalfDimensions / (kSimdWidth / 2);
		const int8x8_t      kZero      = {0};
#endif
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = HalfDimensions * p;
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
			for (IndexType j = 0; j < HalfDimensions; ++j) {
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
	static void order_packs([[maybe_unused]] u64* v) {
#if defined(USE_AVX512)  // _mm512_set_epi32 packs in the order [15 11 7 3 14 10 6 2 13 9 5 1 12 8 4 0]
		u64 tmp0 = v[4], tmp1 = v[5];
		v[4] = v[6], v[5] = v[7];
		v[6] = tmp0, v[7] = tmp1;
		tmp0 = v[8], tmp1 = v[9];
		v[8] = v[12], v[9] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // _mm256_set_epi32 packs in the order [7 3 6 2 5 1 4 0]
		u64 tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	static void inverse_order_packs([[maybe_unused]] u64* v) {
#if defined(USE_AVX512)
		u64 tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = v[8], v[5] = v[9];
		v[8] = tmp0, v[9] = tmp1;
		tmp0 = v[6], tmp1 = v[7];
		v[6] = v[12], v[7] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
		u64 tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	void permute_weights([[maybe_unused]] void (*order_fn)(u64*)) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		u64* b = reinterpret_cast<u64*>(const_cast<BiasType*>(&biases[0]));
		for (IndexType i = 0; i < HalfDimensions * sizeof(BiasType) / sizeof(u64); i += di)
			order_fn(&b[i]);

		for (IndexType j = 0; j < InputDimensions; ++j)
		{
			u64* w =
				reinterpret_cast<u64*>(const_cast<WeightType*>(&weights[j * HalfDimensions]));
			for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(u64);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < InputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights[j * HalfDimensions]);
			for (IndexType i = 0; i < HalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases);
		for (IndexType i = 0; i < HalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

#if defined(NNUE_FT_PACKED8)
	// The signed low byte stores half the already-scaled weight. Only values
	// outside [-128,127] need a (position, residual) correction. The first half
	// of weights holds bytes; its second half holds correction records.
	// Allocation size stays unchanged; the frequently read weight data is halved.
	struct Sparse8Correction {
		u16 position;
		i16 residual;
	};
	static_assert(sizeof(Sparse8Correction) == 4);
	// Packed rows and SIMD tiles use the actual FT width; only correction positions are u16.
	// 重みの列とSIMDタイルはFT幅に追従する。補正位置がu16に収まることだけを確認する。
	static_assert(HalfDimensions - 1 <= std::numeric_limits<u16>::max(),
	              "NNUE_FT_SPARSE8 correction positions must fit in u16");
	static constexpr usize kSparse8Bytes = usize(HalfDimensions) * InputDimensions;

	void prepare_sparse8_weights() {
		usize count = 0;
		for (usize i = 0; i < kSparse8Bytes; ++i) {
			const int value = weights[i] / 2;
			count += value < -128 || value > 127;
		}
		// Check before overwriting any int16 weights. Keep outliers below
		// 1/128 to bound scalar fixups (also well within the available space).
		// Other models retain the original implementation and file format.
		// 16bitの重みを上書きする前に、範囲外の重みが全体の1/128以下かを確認する。
		// 補正で元の整数値を保ち、これを超えるモデルは従来の16bit経路を使う。
		// 評価関数のファイル形式と確保サイズは変更しない。
		if (count > kSparse8Bytes / 128)
			return;
		std::vector<Sparse8Correction> corrections;
		corrections.reserve(count);
		auto* packed = reinterpret_cast<i8*>(weights);
		for (IndexType i = 0; i < InputDimensions; ++i) {
			sparse8_offsets_[i] = static_cast<u32>(corrections.size());
			for (IndexType j = 0; j < HalfDimensions; ++j) {
				const int value = weights[usize(i) * HalfDimensions + j];
				const auto base = static_cast<i8>(value / 2);
				// Forward packing cannot overwrite the next unread int16.
				packed[usize(i) * HalfDimensions + j] = base;
				const int residual = value - int(base) * 2;
				if (residual)
					corrections.push_back({u16(j), i16(residual)});
			}
		}
		sparse8_offsets_[InputDimensions] = static_cast<u32>(corrections.size());
		if (!corrections.empty())
			std::memcpy(packed + kSparse8Bytes, corrections.data(), corrections.size() * sizeof(Sparse8Correction));
		sparse8_enabled_ = true;
	}

#if !defined(NO_PREFETCH)
	// 実際に加減算する列の先頭だけを先読みする。列全体には命令を発行しない。
	void prefetch_sparse8_columns(const Features::IndexList& indices) const {
		const auto* base = reinterpret_cast<const i8*>(weights);
		for (const auto index : indices)
			_mm_prefetch(reinterpret_cast<const char*>(base + usize(HalfDimensions) * index), _MM_HINT_T2);
	}
#endif

	void apply_sparse8_corrections(BiasType* destination, IndexType index, int sign) const {
		const auto* bytes = reinterpret_cast<const unsigned char*>(weights) + kSparse8Bytes;
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
		if (offset == HalfDimensions) {
			apply_sparse8_corrections(reinterpret_cast<BiasType*>(acc), index, Add ? 1 : -1);
			return;
		}
		const auto* packed = reinterpret_cast<const i8*>(weights) + usize(HalfDimensions) * index + offset;
		for (IndexType k = 0; k < kTileRegs; ++k) {
#if defined(USE_AVX512)
			const auto w = _mm512_slli_epi16(_mm512_cvtepi8_epi16(
				_mm256_load_si256(reinterpret_cast<const __m256i*>(packed + k * 32))), 1);
#else
			// Expand only the bytes used by this AVX2 vector; corrections stay unchanged.
			// AVX2の1ベクトル分だけ8bitから16bitへ展開する。範囲外の重みの補正は共通。
			const auto w = _mm256_slli_epi16(_mm256_cvtepi8_epi16(
				_mm_load_si128(reinterpret_cast<const __m128i*>(packed + k * 16))), 1);
#endif
			if constexpr (Add) acc[k] = vec_add_16(acc[k], w);
			else acc[k] = vec_sub_16(acc[k], w);
		}
	}
#endif

#if defined(VECTOR)
	// 変更された各特徴量ごとにaccumulator全体を読み書きするのを避けるため、
	// SIMDレジスタに収まるタイル単位で差分をまとめて適用する。
	static constexpr IndexType kVectorHeight = sizeof(vec_t) / sizeof(BiasType);
	static_assert(HalfDimensions % kVectorHeight == 0, "kVectorHeight must divide HalfDimensions");
	static constexpr IndexType kNumVectorChunks = HalfDimensions / kVectorHeight;
	static constexpr IndexType kTileRegs = [] {
		IndexType regs = std::min<IndexType>(NumRegistersSIMD, kNumVectorChunks);
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
			for (IndexType offset = 0; offset < HalfDimensions; offset += kTileHeight) {
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
			remove_columns[n] = weights + HalfDimensions * removed[n];
		for (IndexType n = 0; n < Added; ++n)
			add_columns[n] = weights + HalfDimensions * added[n];
		for (IndexType offset = 0; offset < HalfDimensions; offset += kTileHeight) {
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
		for (IndexType tile_offset = 0; tile_offset < HalfDimensions; tile_offset += kTileHeight) {
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
			apply_changes(reinterpret_cast<vec_t*>(destination), HalfDimensions);
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
			&weights[HalfDimensions * index + tile_offset]);
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
			&weights[HalfDimensions * index + tile_offset]);
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_sub_16(acc[k], vec_load(column + k));
	}
#endif

	// StockfishのAccumulatorCaches(Finny Tables)と同じ発想。
	// 王位置ごとにrefresh済みaccumulatorを持ち、次回はactive featureの差分だけを適用する。
#if defined(USE_FINNY_TABLES)
	static constexpr bool kUseFinnyTables = HalfDimensions <= 4096;
    static constexpr bool kUseDirectFinny = AccumulatorCaches::kUseDirectFinny;
    using FinnyEntry = AccumulatorCaches::Entry;

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

    static void make_index_diff(
        const Features::IndexList& old_active,
        const Features::IndexList& new_active,
        Features::IndexList& removed,
        Features::IndexList& added) {
        // 入力の並びは駒番号ではなく盤面の列挙順。位置ではなく多重集合として比較する。
        // 駒落ち等の0特徴量が複数あっても、それぞれ一度ずつ対応させる。
#if defined(USE_AVX512)
        if constexpr (RawFeatures::MaxActiveDimensions <= 64) {
            u64 oldMatched = 0, newMatched = 0;
            constexpr usize maxBlocks = (RawFeatures::MaxActiveDimensions + 15) / 16;
            __m512i blocks[maxBlocks];
            __mmask16 masks[maxBlocks];
            const usize blockCount = (new_active.size() + 15) / 16;
            const usize common = std::min(old_active.size(), new_active.size());
            for (usize b = 0; b < blockCount; ++b) {
                const usize count = std::min(usize(16), new_active.size() - 16 * b);
                masks[b] = __mmask16((u32(1) << count) - 1);
                blocks[b] = _mm512_maskz_loadu_epi32(masks[b], new_active.begin() + 16 * b);
                if (16 * b < common) {
                    const usize countCommon = std::min(usize(16), common - 16 * b);
                    const __mmask16 commonMask = __mmask16((u32(1) << countCommon) - 1);
                    const auto old = _mm512_maskz_loadu_epi32(commonMask, old_active.begin() + 16 * b);
                    oldMatched |= u64(_mm512_mask_cmpeq_epi32_mask(commonMask, old, blocks[b])) << (16 * b);
                }
            }
            newMatched = oldMatched;
            const u64 fullMask = new_active.size() == 64 ? ~u64(0)
                                 : (u64(1) << new_active.size()) - 1;
            if (old_active.size() == new_active.size() && oldMatched == fullMask)
                return;
            for (usize i = 0; i < old_active.size(); ++i) {
                if (oldMatched & (u64(1) << i))
                    continue;
                const auto value = _mm512_set1_epi32(old_active[i]);
                u64 matches = 0;
                for (usize b = 0; b < blockCount; ++b)
                    matches |= u64(_mm512_mask_cmpeq_epi32_mask(masks[b], value, blocks[b])) << (16 * b);
                matches &= ~newMatched;
                if (matches)
                    newMatched |= matches & (~matches + 1);
                else
                    removed.push_back(old_active[i]);
            }
            for (usize i = 0; i < new_active.size(); ++i)
                if (!(newMatched & (u64(1) << i)))
                    added.push_back(new_active[i]);
            return;
        }
#endif
        bool oldMatched[RawFeatures::MaxActiveDimensions]{};
        bool newMatched[RawFeatures::MaxActiveDimensions]{};
        for (usize i = 0; i < std::min(old_active.size(), new_active.size()); ++i)
            if (old_active[i] == new_active[i])
                oldMatched[i] = newMatched[i] = true;
        for (usize oi = 0; oi < old_active.size(); ++oi) {
            if (oldMatched[oi])
                continue;
            for (usize ni = 0; ni < new_active.size(); ++ni)
                if (!newMatched[ni] && old_active[oi] == new_active[ni]) {
                    oldMatched[oi] = newMatched[ni] = true;
                    break;
                }
        }
        for (usize i = 0; i < old_active.size(); ++i)
            if (!oldMatched[i])
                removed.push_back(old_active[i]);
        for (usize i = 0; i < new_active.size(); ++i)
            if (!newMatched[i])
                added.push_back(new_active[i]);
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
		const auto* source = trigger_index == 0 ? biases : nullptr;
		update_accumulator_tiled<Packed>(
			source, current,
			[&](vec_t* acc, IndexType tile_offset) {
				for (const auto index : active_indices)
					add_weight_to_tile<Packed>(acc, index, tile_offset);
			});
#else
		if (trigger_index == 0)
			std::memcpy(current, biases, HalfDimensions * sizeof(BiasType));
		else
			std::memset(current, 0, HalfDimensions * sizeof(BiasType));

		for (const auto index : active_indices) {
			const IndexType offset = HalfDimensions * index;
			for (IndexType j = 0; j < HalfDimensions; ++j)
				current[j] += weights[offset + j];
		}
#endif
	}

#if defined(USE_FINNY_TABLES)
#if defined(VECTOR)
	template <bool Packed, typename ApplyChanges>
	void update_accumulator_tiled_to_two(
		const BiasType* source, BiasType* destination0, BiasType* destination1,
		ApplyChanges apply_changes) const {
		for (IndexType tile_offset = 0; tile_offset < HalfDimensions; tile_offset += kTileHeight) {
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
			apply_changes(reinterpret_cast<vec_t*>(destination0), HalfDimensions);
			if (destination1 != destination0)
				apply_changes(reinterpret_cast<vec_t*>(destination1), HalfDimensions);
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
		if (entry.is_initialized())
			make_index_diff(entry.active_indices, active_indices, removed_indices, added_indices);
		bool rebuild = !entry.is_initialized();
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
			const auto* source = trigger_index == 0 ? biases : nullptr;
			update_accumulator_tiled_to_two<Packed>(
				source, entry.accumulation, current,
				[&](vec_t* acc, IndexType tile_offset) {
					for (const auto index : active_indices)
						add_weight_to_tile<Packed>(acc, index, tile_offset);
				});
#else
			refresh_accumulator_from_scratch<Packed>(entry.accumulation, active_indices, trigger_index);
			std::memcpy(current, entry.accumulation, HalfDimensions * sizeof(BiasType));
#endif
			copy_index_list(entry.active_indices, active_indices);
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
				const IndexType offset = HalfDimensions * index;
				for (IndexType j = 0; j < HalfDimensions; ++j)
					entry.accumulation[j] -= weights[offset + j];
			}
			for (const auto index : added_indices) {
				const IndexType offset = HalfDimensions * index;
				for (IndexType j = 0; j < HalfDimensions; ++j)
					entry.accumulation[j] += weights[offset + j];
			}
#endif
			copy_index_list(entry.active_indices, active_indices);
#if !defined(VECTOR)
			std::memcpy(current, entry.accumulation, HalfDimensions * sizeof(BiasType));
#endif
		}
	}

#if defined(VECTOR)
    // Compare cached board data as Stockfish does, converting only changed squares to features.
    // Stockfish同様に保存盤面を比較し、変化したマスだけを特徴量番号へ変換する。
    template<bool Packed>
    void refresh_from_position(const Position& pos, Color perspective, BiasType* current,
                               FinnyEntry& entry, IndexType trigger_index) const {
        const Square king = perspective == BLACK ? pos.square<KING>(perspective)
                                                 : Inv(pos.square<KING>(perspective));
        Features::IndexList removed, added;
        const Features::HalfKA_hm2<Features::Side::kFriend>::IndexContext indices(king, perspective);
        int pieceCount = entry.position.pieceCount;
        const auto& currentPieces = pos.piece_array();
#if defined(USE_AVX512)
        constexpr int compareWidth = 64;
#else
        constexpr int compareWidth = 32;
#endif
        for (int base = 0; base < SQ_NB; base += compareWidth) {
            const int count = std::min(compareWidth, int(SQ_NB) - base);
            u64 changed = 0;
#if defined(USE_AVX512)
            const __mmask64 mask = count == 64 ? ~u64(0) : (u64(1) << count) - 1;
            const auto previous = _mm512_maskz_loadu_epi8(mask, entry.position.pieces.data() + base);
            const auto currentPc = _mm512_maskz_loadu_epi8(mask, currentPieces.data() + base);
            changed = _mm512_mask_cmpneq_epi8_mask(mask, previous, currentPc);
#elif defined(USE_AVX2)
            if (count == 32) {
                const auto previous = _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(entry.position.pieces.data() + base));
                const auto currentPc = _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(currentPieces.data() + base));
                changed = ~u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(previous, currentPc)));
            } else {
                int offset = 0;
                if (count >= 16) {
                    const auto previous = _mm_loadu_si128(
                      reinterpret_cast<const __m128i*>(entry.position.pieces.data() + base));
                    const auto currentPc = _mm_loadu_si128(
                      reinterpret_cast<const __m128i*>(currentPieces.data() + base));
                    changed = u16(~_mm_movemask_epi8(_mm_cmpeq_epi8(previous, currentPc)));
                    offset = 16;
                }
                // The final 17 squares use a 16-byte load plus one scalar comparison.
                // 末尾17マスは16バイトの読み込みと1マスの比較に分け、盤面配列の外を読まない。
                for (; offset < count; ++offset)
                    changed |= u64(entry.position.pieces[base + offset] != currentPieces[base + offset]) << offset;
            }
#else
            for (int offset = 0; offset < count; ++offset)
                changed |= u64(entry.position.pieces[base + offset] != currentPieces[base + offset]) << offset;
#endif
            while (changed) {
                const Square sq = Square(base + int(lsb(changed)));
                changed &= changed - 1;
                const Piece before = entry.position.pieces[sq], after = currentPieces[sq];
                // Promoted minor pieces share the gold feature in these nets.
                if (before != NO_PIECE && after != NO_PIECE
                    && kpp_board_index[before].from[perspective] == kpp_board_index[after].from[perspective])
                    continue;
                if (before != NO_PIECE) {
                    removed.push_back(indices.board(before, sq));
                    --pieceCount;
                }
                if (after != NO_PIECE) {
                    added.push_back(indices.board(after, sq));
                    ++pieceCount;
                }
            }
        }
        for (Color c : {BLACK, WHITE}) {
            const Hand before = entry.position.hands[c], after = pos.hand_of(c);
            if (before == after) continue;
            for (PieceType pt = PAWN; pt < KING; ++pt) {
                const int oldCount = hand_count(before, pt), newCount = hand_count(after, pt);
                pieceCount += newCount - oldCount;
                for (int n = newCount; n < oldCount; ++n)
                    removed.push_back(indices.hand(c, pt, n));
                for (int n = oldCount; n < newCount; ++n)
                    added.push_back(indices.hand(c, pt, n));
            }
        }
        if (entry.is_initialized()) {
            for (int n = entry.position.pieceCount; n < pieceCount; ++n)
                removed.push_back(indices.zero());
            for (int n = pieceCount; n < entry.position.pieceCount; ++n)
                added.push_back(indices.zero());
        }
        bool rebuild = !entry.is_initialized();
#if defined(USE_FINNY_RESET)
        rebuild = rebuild || removed.size() + added.size() >= 40;
#endif
        Features::IndexList active;
        if (rebuild)
            Features::HalfKA_hm2<Features::Side::kFriend>::append_active_indices(pos, perspective, &active);
#if defined(NNUE_FT_PACKED8) && !defined(NO_PREFETCH)
        if constexpr (Packed) {
            if (rebuild) prefetch_sparse8_columns(active);
            else {
                prefetch_sparse8_columns(removed);
                prefetch_sparse8_columns(added);
            }
        }
#endif
        if (rebuild) {
            const auto* source = trigger_index == 0 ? biases : nullptr;
            update_accumulator_tiled_to_two<Packed>(source, entry.accumulation, current,
              [&](vec_t* acc, IndexType offset) {
                  for (auto idx : active) add_weight_to_tile<Packed>(acc, idx, offset);
              });
        } else {
            update_accumulator_tiled_to_two<Packed>(entry.accumulation, entry.accumulation, current,
              [&](vec_t* acc, IndexType offset) {
                  for (auto idx : removed) sub_weight_from_tile<Packed>(acc, idx, offset);
                  for (auto idx : added) add_weight_to_tile<Packed>(acc, idx, offset);
              });
        }
        entry.position.pieces = currentPieces;
        entry.position.hands[BLACK] = pos.hand_of(BLACK);
        entry.position.hands[WHITE] = pos.hand_of(WHITE);
        entry.position.pieceCount = pieceCount;
    }
#endif

	template <bool Packed = false>
	void refresh_accumulator_with_finny_cache(const Position& pos, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
		auto& cache = *refreshTable;

		auto& accumulator = accumulator_of(state);
#if defined(VECTOR)
        if constexpr (kUseDirectFinny) {
            for (Color perspective : {BLACK, WHITE}) {
                auto& entry = cache.entries[0][perspective][pos.square<KING>(perspective)];
                refresh_from_position<Packed>(pos, perspective,
                  accumulator.accumulation[perspective][0], entry, 0);
            }
            accumulator.computed_accumulation = true;
            return;
        }
#endif
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			const auto trigger = kRefreshTriggers[i];
			RawFeatures::append_active_indices(pos, trigger, active_indices);
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
	}
#endif

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	template <bool Packed = false>
	void refresh_accumulator(const Position& pos, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
#if defined(USE_FINNY_TABLES)
		if (kUseFinnyTables && !refreshTable.empty()) {
			refresh_accumulator_with_finny_cache<Packed>(pos, state, refreshTable);
			return;
		}
#endif

		auto& accumulator = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::append_active_indices(pos, kRefreshTriggers[i], active_indices);
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
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	template <bool Packed = false>
	void update_accumulator(const Position& pos, AccumulatorState* state, [[maybe_unused]] AccumulatorCaches& refreshTable) const {
        assert(state);
		const auto& prev_accumulator = accumulator_of(state->previous);
		auto&      accumulator      = accumulator_of(state);
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2]{};
#if defined(USE_FINNY_TABLES) && defined(USE_FINNY_RESET) && defined(VECTOR)
            if constexpr (kUseDirectFinny) {
                for (Color c : {BLACK, WHITE}) {
                    reset[c] = state->dirtyPiece.pc == make_piece(c, KING);
                    if (!reset[c])
                        Features::HalfKA_hm2<Features::Side::kFriend>::append_changed_indices(
                          pos, c, &removed_indices[c], &added_indices[c], state->dirtyPiece);
                }
            } else
#endif
			RawFeatures::append_changed_indices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset, state->dirtyPiece);
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
				if (kUseFinnyTables && !refreshTable.empty()) {
					if (reset[perspective]) {
						auto& cache = *refreshTable;
						const auto bucket = finny_bucket_square(pos, kRefreshTriggers[i], perspective);
						auto& entry = cache.entries[i][perspective][bucket];
						// On reset, added_indices contains every active feature.
						// The other perspective keeps its ordinary incremental update.
#if defined(VECTOR)
                        if constexpr (kUseDirectFinny)
                            refresh_from_position<Packed>(pos, perspective,
                              accumulator.accumulation[perspective][i], entry, i);
                        else
#endif
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
					const auto* source = i == 0 ? biases : nullptr;
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
						std::memcpy(accumulator.accumulation[perspective][i], biases,
						            HalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, HalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            HalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = HalfDimensions * index;
						for (IndexType j = 0; j < HalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights[offset + j];
						}
					}
				}
				// Difference calculation for features that changed from 0 to 1
				// 0から1に変化した特徴量に関する差分計算
				for (const auto index : added_indices[perspective]) {
					const IndexType offset = HalfDimensions * index;
					for (IndexType j = 0; j < HalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights[offset + j];
					}
				}
#endif
			}
		}

		accumulator.computed_accumulation = true;
	}

    // LEB128圧縮データを識別するためのmagic string。
    static constexpr char Leb128MagicString[] = "COMPRESSED_LEB128";
    static constexpr usize Leb128MagicStringSize = sizeof(Leb128MagicString) - 1;

    // 将棋用FTファイルのsigned LEB128形式を読み込む。従来の復号処理を維持する。
    template<typename IntType>
    static void read_leb_128(std::istream& stream, IntType* out, usize count) {
        // Check the presence of our LEB128 magic string
        // LEB128圧縮データの識別文字列を確認する。
        char leb128MagicString[Leb128MagicStringSize];
        stream.read(leb128MagicString, Leb128MagicStringSize);
        assert(strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) == 0);

        static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");

        const u32 BUF_SIZE = 4096;
        u8 buf[BUF_SIZE];
        auto bytes_left = read_little_endian<u32>(stream);
        u32 buf_pos = BUF_SIZE;
        for (usize i = 0; i < count; ++i)
        {
            IntType result = 0;
            usize shift = 0;
            do
            {
                if (buf_pos == BUF_SIZE)
                {
                    stream.read(reinterpret_cast<char*>(buf), std::min(bytes_left, BUF_SIZE));
                    buf_pos = 0;
                }

                u8 byte = buf[buf_pos++];
                --bytes_left;
                result |= (byte & 0x7f) << shift;
                shift += 7;

                if ((byte & 0x80) == 0)
                {
                    out[i] = (sizeof(IntType) * 8 <= shift || (byte & 0x40) == 0)
                      ? result : result | ~((1 << shift) - 1);
                    break;
                }
            } while (shift < sizeof(IntType) * 8);
        }
        assert(bytes_left == 0);
    }

	// parameter
	// パラメータ
	// Stockfishはstd::arrayだが、ここは既存のコード生成を保つため生配列を維持する。
	alignas(CacheLineSize) BiasType biases[HalfDimensions];
	alignas(CacheLineSize) WeightType weights[HalfDimensions * InputDimensions];
#if defined(NNUE_FT_PACKED8)
	u32 sparse8_offsets_[InputDimensions + 1];
	bool sparse8_enabled_ = false;
#endif

};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#if defined(NNUE_FT_PACKED8)
#undef NNUE_FT_PACKED8
#endif

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
