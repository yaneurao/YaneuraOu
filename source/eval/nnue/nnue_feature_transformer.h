// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef _NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define _NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring>  // std::memset()

namespace Eval::NNUE {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// ベクトル命令が有効な場合、変数のタイルを、
// 各タイルがCPUのベクトルレジスタに収まるように、更新してリフレッシュする。
#define VECTOR

#if defined(USE_AVX512)
typedef __m512i vec_t;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_zero _mm512_setzero_si512()
static constexpr IndexType kNumRegs = 8;  // only 8 are needed

#elif defined(USE_AVX2)
typedef __m256i vec_t;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_zero _mm256_setzero_si256()
static constexpr IndexType kNumRegs = 16;

#elif defined(USE_SSE2)
typedef __m128i vec_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_epi16(a, b)
#define vec_sub_16(a, b) _mm_sub_epi16(a, b)
#define vec_zero _mm_setzero_si128()
static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

#elif defined(USE_MMX)
typedef __m64 vec_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero _mm_setzero_si64()
static constexpr IndexType kNumRegs = 8;

#elif defined(USE_NEON)
typedef int16x8_t vec_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) vaddq_s16(a, b)
#define vec_sub_16(a, b) vsubq_s16(a, b)
#define vec_zero \
	{ 0 }
static constexpr IndexType kNumRegs = 16;

#else
#undef VECTOR

#endif

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

#if defined(VECTOR)
	static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
	static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
#endif

   public:
	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions  = RawFeatures::kDimensions;
	static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr std::uint32_t GetHashValue() { return RawFeatures::kHashValue ^ kOutputDimensions; }

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(kInputDimensions) + "->" +
		       std::to_string(kHalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	bool ReadParameters(std::istream& stream) {
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
		return !stream.fail();
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos) const {
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
			return true;
		}
		const auto prev = now->previous;
		if (prev && prev->accumulator.computed_accumulation) {
			update_accumulator(pos);
			return true;
		}
		return false;
	}

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, bool refresh) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;

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
				_mm512_store_si512(&out[j], _mm512_permutexvar_epi64(
				                                kControl, _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
			}

#elif defined(USE_AVX2)
			auto out = reinterpret_cast<__m256i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m256i sum0 =
				    _mm256_load_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m256i sum1 =
				    _mm256_load_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm256_add_epi16(
					    sum0, reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm256_add_epi16(
					    sum1, reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
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
				output[offset + j] = static_cast<OutputType>(std::max<int>(0, std::min<int>(127, sum)));
			}
#endif
		}
#if defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		_mm_empty();
#endif
	}

   private:
	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;
					auto accumulation      = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
					auto column            = reinterpret_cast<const vec_t*>(&weights_[offset]);
					constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
					for (IndexType j = 0; j < kNumChunks; ++j) {
						accumulation[j] = vec_add_16(accumulation[j], column[j]);
					}
				}
#else
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
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
		accumulator.computed_score = false;
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
				auto accumulation              = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
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
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
						}
#endif
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// parameter type
	// パラメータの型
	using BiasType   = std::int16_t;
	using WeightType = std::int16_t;

	// Make the learning class a friend
	// 学習用クラスをfriendにする
	friend class Trainer<FeatureTransformer>;

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kHalfDimensions * kInputDimensions];
};  // class FeatureTransformer

}  // namespace Eval::NNUE

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
