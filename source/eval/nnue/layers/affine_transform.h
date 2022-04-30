﻿// Definition of layer AffineTransform of NNUE evaluation function
// NNUE評価関数の層AffineTransformの定義

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval::NNUE::Layers {

// Affine transformation layer
// アフィン変換層
template <typename PreviousLayer, IndexType OutputDimensions>
class AffineTransform {
   public:
	// Input/output type
	// 入出力の型
	using InputType  = typename PreviousLayer::OutputType;
	using OutputType = std::int32_t;
	static_assert(std::is_same<InputType, std::uint8_t>::value, "");

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions       = PreviousLayer::kOutputDimensions;
	static constexpr IndexType kOutputDimensions      = OutputDimensions;
	static constexpr IndexType kPaddedInputDimensions = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

	// Size of forward propagation buffer used in this layer
	// この層で使用する順伝播用バッファのサイズ
	static constexpr std::size_t kSelfBufferSize =
	    CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

	// Size of the forward propagation buffer used from the input layer to this layer
	// 入力層からこの層までで使用する順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = PreviousLayer::kBufferSize + kSelfBufferSize;

	// 評価関数ファイルに埋め込むハッシュ値
	// Hash value embedded in the evaluation file
	static constexpr std::uint32_t GetHashValue() {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= PreviousLayer::GetHashValue() >> 1;
		hash_value ^= PreviousLayer::GetHashValue() << 31;
		return hash_value;
	}

	// 入力層からこの層までの構造を表す文字列
	static std::string GetStructureString() {
		return "AffineTransform[" + std::to_string(kOutputDimensions) + "<-" + std::to_string(kInputDimensions) + "](" +
		       PreviousLayer::GetStructureString() + ")";
	}

	// Read network parameters
	// パラメータを読み込む
	bool ReadParameters(std::istream& stream) {
		if (!previous_layer_.ReadParameters(stream))
			return false;
		for (std::size_t i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
		return !stream.fail();
	}

	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		if (!previous_layer_.WriteParameters(stream))
			return false;
		// TODO : endiannessの調整するコード必要なのでは。(やね)
		stream.write(reinterpret_cast<const char*>(biases_), kOutputDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_),
		             kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType));
		return !stream.fail();
	}

	// Forward propagation
	// 順伝播
	const OutputType* Propagate(const TransformedFeatureType* transformed_features, char* buffer) const {
		const auto input = previous_layer_.Propagate(transformed_features, buffer + kSelfBufferSize);

#if defined(USE_WASM_SIMD)
		{
			// Simplify variable names (y = Ax + b)
			constexpr int n = kInputDimensions;
			constexpr int m = kOutputDimensions;
			constexpr int n_stride = kPaddedInputDimensions;
			auto A = *reinterpret_cast<const int8_t(*)[m][n_stride]>(weights_);
			auto x = *reinterpret_cast<const uint8_t(*)[n]>(input);
			auto b = *reinterpret_cast<const int32_t(*)[m]>(biases_);
			auto y = *reinterpret_cast<int32_t(*)[m]>(buffer);
			emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
			return y;
		}
#endif

#if defined(USE_AVX512)

		[[maybe_unused]] const __m512i kOnes512 = _mm512_set1_epi16(1);

		[[maybe_unused]] auto m512_hadd = [](__m512i sum, int bias) -> int {
			return _mm512_reduce_add_epi32(sum) + bias;
		};

		// This function takes
		//   sum0 = [xmm0a, xmm0b, xmm0c, xmm0d]
		//   sum1 = [xmm1a, xmm1b, xmm1c, xmm1d]
		//   sum2 = [xmm2a, xmm2b, xmm2c, xmm2d]
		//   sum3 = [xmm3a, xmm3b, xmm3c, xmm3d]
		// and returns
		//   ret = [
		//     reduce_add_epi32(xmm0a), reduce_add_epi32(xmm1a), reduce_add_epi32(xmm2a), reduce_add_epi32(xmm3a),
		//     reduce_add_epi32(xmm0b), reduce_add_epi32(xmm1b), reduce_add_epi32(xmm2b), reduce_add_epi32(xmm3b),
		//     reduce_add_epi32(xmm0c), reduce_add_epi32(xmm1c), reduce_add_epi32(xmm2c), reduce_add_epi32(xmm3c),
		//     reduce_add_epi32(xmm0d), reduce_add_epi32(xmm1d), reduce_add_epi32(xmm2d), reduce_add_epi32(xmm3d)
		//   ]
		[[maybe_unused]] auto m512_hadd128x16_interleave = [](__m512i sum0, __m512i sum1, __m512i sum2,
		                                                      __m512i sum3) -> __m512i {
			__m512i sum01a = _mm512_unpacklo_epi32(sum0, sum1);
			__m512i sum01b = _mm512_unpackhi_epi32(sum0, sum1);

			__m512i sum23a = _mm512_unpacklo_epi32(sum2, sum3);
			__m512i sum23b = _mm512_unpackhi_epi32(sum2, sum3);

			__m512i sum01 = _mm512_add_epi32(sum01a, sum01b);
			__m512i sum23 = _mm512_add_epi32(sum23a, sum23b);

			__m512i sum0123a = _mm512_unpacklo_epi64(sum01, sum23);
			__m512i sum0123b = _mm512_unpackhi_epi64(sum01, sum23);

			return _mm512_add_epi32(sum0123a, sum0123b);
		};

		[[maybe_unused]] auto m512_haddx4 = [m512_hadd128x16_interleave](__m512i sum0, __m512i sum1, __m512i sum2,
		                                                                 __m512i sum3, __m128i bias) -> __m128i {
			__m512i sum = m512_hadd128x16_interleave(sum0, sum1, sum2, sum3);

			__m256i sum256lo = _mm512_castsi512_si256(sum);
			__m256i sum256hi = _mm512_extracti64x4_epi64(sum, 1);

			sum256lo = _mm256_add_epi32(sum256lo, sum256hi);

			__m128i sum128lo = _mm256_castsi256_si128(sum256lo);
			__m128i sum128hi = _mm256_extracti128_si256(sum256lo, 1);

			return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
		};

		[[maybe_unused]] auto m512_haddx8 = [m512_hadd128x16_interleave](
		                                        __m512i sum0, __m512i sum1, __m512i sum2, __m512i sum3, __m512i sum4,
		                                        __m512i sum5, __m512i sum6, __m512i sum7, __m256i bias) -> __m256i {
			__m512i suma = m512_hadd128x16_interleave(sum0, sum1, sum2, sum3);
			__m512i sumb = m512_hadd128x16_interleave(sum4, sum5, sum6, sum7);

			__m512i indices0 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
			__m512i indices1 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
			__m512i x        = _mm512_add_epi32(_mm512_permutex2var_epi64(suma, indices0, sumb),
                                         _mm512_permutex2var_epi64(suma, indices1, sumb));

			__m256i sum256lo = _mm512_castsi512_si256(x);
			__m256i sum256hi = _mm512_extracti64x4_epi64(x, 1);

			return _mm256_add_epi32(_mm256_add_epi32(sum256lo, sum256hi), bias);
		};

		[[maybe_unused]] auto m512_hadd256x8 = [m512_hadd128x16_interleave](__m512i sum0, __m512i sum1, __m512i sum2,
		                                                                    __m512i sum3, __m256i bias) -> __m256i {
			__m512i sum = m512_hadd128x16_interleave(sum0, sum1, sum2, sum3);

			__m512i indices = _mm512_setr_epi32(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);
			sum             = _mm512_permutexvar_epi32(indices, sum);

			__m256i sum256lo = _mm512_castsi512_si256(sum);
			__m256i sum256hi = _mm512_extracti64x4_epi64(sum, 1);

			return _mm256_add_epi32(_mm256_hadd_epi32(sum256lo, sum256hi), bias);
		};

		[[maybe_unused]] auto m512_hadd256x16 =
		    [m512_hadd128x16_interleave](__m512i sum0, __m512i sum1, __m512i sum2, __m512i sum3, __m512i sum4,
		                                 __m512i sum5, __m512i sum6, __m512i sum7, __m512i bias) -> __m512i {
			__m512i suma = m512_hadd128x16_interleave(sum0, sum1, sum2, sum3);
			__m512i sumb = m512_hadd128x16_interleave(sum4, sum5, sum6, sum7);

			__m512i indices0 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
			__m512i indices1 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
			__m512i x        = _mm512_add_epi32(_mm512_permutex2var_epi64(suma, indices0, sumb),
                                         _mm512_permutex2var_epi64(suma, indices1, sumb));

			__m512i indices = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
			return _mm512_add_epi32(_mm512_permutexvar_epi32(indices, x), bias);
		};

		[[maybe_unused]] auto m512_add_dpbusd_epi32 = [=](__m512i& acc, __m512i a, __m512i b) {
#if defined(USE_VNNI)
			acc = _mm512_dpbusd_epi32(acc, a, b);
#else
			__m512i product0 = _mm512_maddubs_epi16(a, b);
			product0         = _mm512_madd_epi16(product0, kOnes512);
			acc              = _mm512_add_epi32(acc, product0);
#endif
		};

		[[maybe_unused]] auto m512_add_dpbusd_epi32x2 = [=](__m512i& acc, __m512i a0, __m512i b0, __m512i a1,
		                                                    __m512i b1) {
#if defined(USE_VNNI)
			acc = _mm512_dpbusd_epi32(acc, a0, b0);
			acc = _mm512_dpbusd_epi32(acc, a1, b1);
#else
			__m512i product0 = _mm512_maddubs_epi16(a0, b0);
			__m512i product1 = _mm512_maddubs_epi16(a1, b1);
			product0         = _mm512_adds_epi16(product0, product1);
			product0         = _mm512_madd_epi16(product0, kOnes512);
			acc              = _mm512_add_epi32(acc, product0);
#endif
		};

#endif
#if defined(USE_AVX2)

		[[maybe_unused]] const __m256i kOnes256 = _mm256_set1_epi16(1);

		[[maybe_unused]] auto m256_hadd = [](__m256i sum, int bias) -> int {
			__m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
			sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
			sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
			return _mm_cvtsi128_si32(sum128) + bias;
		};

		[[maybe_unused]] auto m256_haddx4 = [](__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3,
		                                       __m128i bias) -> __m128i {
			sum0 = _mm256_hadd_epi32(sum0, sum1);
			sum2 = _mm256_hadd_epi32(sum2, sum3);

			sum0 = _mm256_hadd_epi32(sum0, sum2);

			__m128i sum128lo = _mm256_castsi256_si128(sum0);
			__m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

			return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
		};

		[[maybe_unused]] auto m256_add_dpbusd_epi32 = [=](__m256i& acc, __m256i a, __m256i b) {
#if defined(USE_VNNI)
			acc = _mm256_dpbusd_epi32(acc, a, b);
#else
			__m256i product0 = _mm256_maddubs_epi16(a, b);
			product0         = _mm256_madd_epi16(product0, kOnes256);
			acc              = _mm256_add_epi32(acc, product0);
#endif
		};

		[[maybe_unused]] auto m256_add_dpbusd_epi32x2 = [=](__m256i& acc, __m256i a0, __m256i b0, __m256i a1,
		                                                    __m256i b1) {
#if defined(USE_VNNI)
			acc = _mm256_dpbusd_epi32(acc, a0, b0);
			acc = _mm256_dpbusd_epi32(acc, a1, b1);
#else
			__m256i product0 = _mm256_maddubs_epi16(a0, b0);
			__m256i product1 = _mm256_maddubs_epi16(a1, b1);
			product0         = _mm256_adds_epi16(product0, product1);
			product0         = _mm256_madd_epi16(product0, kOnes256);
			acc              = _mm256_add_epi32(acc, product0);
#endif
		};

#endif

#if defined(USE_SSSE3)

		[[maybe_unused]] const __m128i kOnes128 = _mm_set1_epi16(1);

		[[maybe_unused]] auto m128_hadd = [](__m128i sum, int bias) -> int {
			sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));  //_MM_PERM_BADC
			sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));  //_MM_PERM_CDAB
			return _mm_cvtsi128_si32(sum) + bias;
		};

		[[maybe_unused]] auto m128_haddx4 = [](__m128i sum0, __m128i sum1, __m128i sum2, __m128i sum3,
		                                       __m128i bias) -> __m128i {
			sum0 = _mm_hadd_epi32(sum0, sum1);
			sum2 = _mm_hadd_epi32(sum2, sum3);

			sum0 = _mm_hadd_epi32(sum0, sum2);

			return _mm_add_epi32(sum0, bias);
		};

		[[maybe_unused]] auto m128_add_dpbusd_epi32 = [=](__m128i& acc, __m128i a, __m128i b) {
			__m128i product0 = _mm_maddubs_epi16(a, b);
			product0         = _mm_madd_epi16(product0, kOnes128);
			acc              = _mm_add_epi32(acc, product0);
		};

		[[maybe_unused]] auto m128_add_dpbusd_epi32x2 = [=](__m128i& acc, __m128i a0, __m128i b0, __m128i a1,
		                                                    __m128i b1) {
			__m128i product0 = _mm_maddubs_epi16(a0, b0);
			__m128i product1 = _mm_maddubs_epi16(a1, b1);
			product0         = _mm_adds_epi16(product0, product1);
			product0         = _mm_madd_epi16(product0, kOnes128);
			acc              = _mm_add_epi32(acc, product0);
		};

#endif

#if defined(USE_AVX512)

		constexpr IndexType kNumChunks512 = kPaddedInputDimensions / (kSimdWidth * 2);
		constexpr IndexType kNumChunks256 = kPaddedInputDimensions / kSimdWidth;

		const auto output = reinterpret_cast<OutputType*>(buffer);

		// Since to saturate a zmm register it takes 64 bytes we
		// cannot use AVX512 for the smaller affine transforms.
		// Instead we fallback to a AVX2 implementation if the
		// kInputDimensions isn't a multiple of 64.
		// Note that this means that for example for
		// kInputDimensions of 96 we fallback to AVX2 even though
		// the first 64 elements could be processed with AVX512.
		// This is caused by mixing the __m256 and __m512 variables
		// required to better handle that case and it would
		// require handling more cases statically not to lose performance.
		// This should be revisited if such input dimensions are to be considered.
		[[maybe_unused]] const auto input_vector512 = reinterpret_cast<const __m512i*>(input);
		[[maybe_unused]] const auto input_vector256 = reinterpret_cast<const __m256i*>(input);

		// kOutputDimensions is either 1 or a multiple of kSimdWidth
		// because then it is also an input dimension.
		if constexpr (kOutputDimensions % 16 == 0 && kNumChunks256 == 1) {
			for (IndexType i = 0; i < kOutputDimensions; i += 16) {
				const IndexType offset01a = (i + 0) * kPaddedInputDimensions;
				const IndexType offset23a = (i + 2) * kPaddedInputDimensions;
				const IndexType offset45a = (i + 4) * kPaddedInputDimensions;
				const IndexType offset67a = (i + 6) * kPaddedInputDimensions;
				const IndexType offset01b = (i + 8) * kPaddedInputDimensions;
				const IndexType offset23b = (i + 10) * kPaddedInputDimensions;
				const IndexType offset45b = (i + 12) * kPaddedInputDimensions;
				const IndexType offset67b = (i + 14) * kPaddedInputDimensions;

				const __m512i bias   = *reinterpret_cast<const __m512i*>(&biases_[i]);
				__m512i*      outptr = reinterpret_cast<__m512i*>(&output[i]);

				__m512i sum01a = _mm512_setzero_si512();
				__m512i sum23a = _mm512_setzero_si512();
				__m512i sum45a = _mm512_setzero_si512();
				__m512i sum67a = _mm512_setzero_si512();
				__m512i sum01b = _mm512_setzero_si512();
				__m512i sum23b = _mm512_setzero_si512();
				__m512i sum45b = _mm512_setzero_si512();
				__m512i sum67b = _mm512_setzero_si512();

				const auto row01a = *reinterpret_cast<const __m512i*>(&weights_[offset01a]);
				const auto row23a = *reinterpret_cast<const __m512i*>(&weights_[offset23a]);
				const auto row45a = *reinterpret_cast<const __m512i*>(&weights_[offset45a]);
				const auto row67a = *reinterpret_cast<const __m512i*>(&weights_[offset67a]);
				const auto row01b = *reinterpret_cast<const __m512i*>(&weights_[offset01b]);
				const auto row23b = *reinterpret_cast<const __m512i*>(&weights_[offset23b]);
				const auto row45b = *reinterpret_cast<const __m512i*>(&weights_[offset45b]);
				const auto row67b = *reinterpret_cast<const __m512i*>(&weights_[offset67b]);

				const __m256i in256 = input_vector256[0];
				const __m512i in    = _mm512_inserti64x4(_mm512_castsi256_si512(in256), in256, 1);

				m512_add_dpbusd_epi32(sum01a, in, row01a);
				m512_add_dpbusd_epi32(sum23a, in, row23a);
				m512_add_dpbusd_epi32(sum45a, in, row45a);
				m512_add_dpbusd_epi32(sum67a, in, row67a);
				m512_add_dpbusd_epi32(sum01b, in, row01b);
				m512_add_dpbusd_epi32(sum23b, in, row23b);
				m512_add_dpbusd_epi32(sum45b, in, row45b);
				m512_add_dpbusd_epi32(sum67b, in, row67b);

				*outptr = m512_hadd256x16(sum01a, sum23a, sum45a, sum67a, sum01b, sum23b, sum45b, sum67b, bias);
			}
		} else if constexpr (kOutputDimensions % 4 == 0) {
			for (IndexType i = 0; i < kOutputDimensions; i += 4) {
				const IndexType offset0 = (i + 0) * kPaddedInputDimensions;
				const IndexType offset1 = (i + 1) * kPaddedInputDimensions;
				const IndexType offset2 = (i + 2) * kPaddedInputDimensions;
				const IndexType offset3 = (i + 3) * kPaddedInputDimensions;

				const __m128i bias   = *reinterpret_cast<const __m128i*>(&biases_[i]);
				__m128i*      outptr = reinterpret_cast<__m128i*>(&output[i]);

				if constexpr (kPaddedInputDimensions % (kSimdWidth * 2) == 0) {
					__m512i sum0 = _mm512_setzero_si512();
					__m512i sum1 = _mm512_setzero_si512();
					__m512i sum2 = _mm512_setzero_si512();
					__m512i sum3 = _mm512_setzero_si512();

					const auto row0 = reinterpret_cast<const __m512i*>(&weights_[offset0]);
					const auto row1 = reinterpret_cast<const __m512i*>(&weights_[offset1]);
					const auto row2 = reinterpret_cast<const __m512i*>(&weights_[offset2]);
					const auto row3 = reinterpret_cast<const __m512i*>(&weights_[offset3]);

					int j = 0;
					if (!canSaturate16x4[i / 4]) {
						for (; j < (int)kNumChunks512 - 1; j += 2) {
							const __m512i in0 = input_vector512[j];
							const __m512i in1 = input_vector512[j + 1];

							m512_add_dpbusd_epi32x2(sum0, in0, row0[j], in1, row0[j + 1]);
							m512_add_dpbusd_epi32x2(sum1, in0, row1[j], in1, row1[j + 1]);
							m512_add_dpbusd_epi32x2(sum2, in0, row2[j], in1, row2[j + 1]);
							m512_add_dpbusd_epi32x2(sum3, in0, row3[j], in1, row3[j + 1]);
						}
					}
					for (; j < (int)kNumChunks512; ++j) {
						const __m512i in = input_vector512[j];

						m512_add_dpbusd_epi32(sum0, in, row0[j]);
						m512_add_dpbusd_epi32(sum1, in, row1[j]);
						m512_add_dpbusd_epi32(sum2, in, row2[j]);
						m512_add_dpbusd_epi32(sum3, in, row3[j]);
					}

					*outptr = m512_haddx4(sum0, sum1, sum2, sum3, bias);
				} else {
					__m256i sum0 = _mm256_setzero_si256();
					__m256i sum1 = _mm256_setzero_si256();
					__m256i sum2 = _mm256_setzero_si256();
					__m256i sum3 = _mm256_setzero_si256();

					const auto row0 = reinterpret_cast<const __m256i*>(&weights_[offset0]);
					const auto row1 = reinterpret_cast<const __m256i*>(&weights_[offset1]);
					const auto row2 = reinterpret_cast<const __m256i*>(&weights_[offset2]);
					const auto row3 = reinterpret_cast<const __m256i*>(&weights_[offset3]);

					for (IndexType j = 0; j < kNumChunks256; ++j) {
						const __m256i in = input_vector256[j];

						m256_add_dpbusd_epi32(sum0, in, row0[j]);
						m256_add_dpbusd_epi32(sum1, in, row1[j]);
						m256_add_dpbusd_epi32(sum2, in, row2[j]);
						m256_add_dpbusd_epi32(sum3, in, row3[j]);
					}

					*outptr = m256_haddx4(sum0, sum1, sum2, sum3, bias);
				}
			}
		} else if constexpr (kOutputDimensions == 1) {
			if constexpr (kPaddedInputDimensions % (kSimdWidth * 2) == 0) {
				__m512i sum0 = _mm512_setzero_si512();

				const auto row0 = reinterpret_cast<const __m512i*>(&weights_[0]);

				for (IndexType j = 0; j < kNumChunks512; ++j) {
					const __m512i in = input_vector512[j];

					m512_add_dpbusd_epi32(sum0, in, row0[j]);
				}

				output[0] = m512_hadd(sum0, biases_[0]);
			} else {
				__m256i sum0 = _mm256_setzero_si256();

				const auto row0 = reinterpret_cast<const __m256i*>(&weights_[0]);

				for (IndexType j = 0; j < kNumChunks256; ++j) {
					const __m256i in = input_vector256[j];

					m256_add_dpbusd_epi32(sum0, in, row0[j]);
				}

				output[0] = m256_hadd(sum0, biases_[0]);
			}
		} else {
			// This case can never happen because kOutputDimensions
			// is always 1 or a multiple of kSimdWidth.
			ASSERT_LV5(false);
		}

#elif defined(USE_AVX2)

		constexpr IndexType kNumChunks = kPaddedInputDimensions / kSimdWidth;

		const auto output       = reinterpret_cast<OutputType*>(buffer);
		const auto input_vector = reinterpret_cast<const __m256i*>(input);

		// kOutputDimensions is either 1 or a multiple of kSimdWidth
		// because then it is also an input dimension.
		if constexpr (kOutputDimensions % 4 == 0) {
			for (IndexType i = 0; i < kOutputDimensions; i += 4) {
				const IndexType offset0 = (i + 0) * kPaddedInputDimensions;
				const IndexType offset1 = (i + 1) * kPaddedInputDimensions;
				const IndexType offset2 = (i + 2) * kPaddedInputDimensions;
				const IndexType offset3 = (i + 3) * kPaddedInputDimensions;

				const __m128i bias   = *reinterpret_cast<const __m128i*>(&biases_[i]);
				__m128i*      outptr = reinterpret_cast<__m128i*>(&output[i]);

				__m256i sum0 = _mm256_setzero_si256();
				__m256i sum1 = _mm256_setzero_si256();
				__m256i sum2 = _mm256_setzero_si256();
				__m256i sum3 = _mm256_setzero_si256();

				const auto row0 = reinterpret_cast<const __m256i*>(&weights_[offset0]);
				const auto row1 = reinterpret_cast<const __m256i*>(&weights_[offset1]);
				const auto row2 = reinterpret_cast<const __m256i*>(&weights_[offset2]);
				const auto row3 = reinterpret_cast<const __m256i*>(&weights_[offset3]);

				int j = 0;
				if (!canSaturate16x4[i / 4]) {
					for (; j < (int)kNumChunks - 1; j += 2) {
						const __m256i in0 = input_vector[j];
						const __m256i in1 = input_vector[j + 1];

						m256_add_dpbusd_epi32x2(sum0, in0, row0[j], in1, row0[j + 1]);
						m256_add_dpbusd_epi32x2(sum1, in0, row1[j], in1, row1[j + 1]);
						m256_add_dpbusd_epi32x2(sum2, in0, row2[j], in1, row2[j + 1]);
						m256_add_dpbusd_epi32x2(sum3, in0, row3[j], in1, row3[j + 1]);
					}
				}
				for (; j < (int)kNumChunks; ++j) {
					const __m256i in = input_vector[j];

					m256_add_dpbusd_epi32(sum0, in, row0[j]);
					m256_add_dpbusd_epi32(sum1, in, row1[j]);
					m256_add_dpbusd_epi32(sum2, in, row2[j]);
					m256_add_dpbusd_epi32(sum3, in, row3[j]);
				}

				*outptr = m256_haddx4(sum0, sum1, sum2, sum3, bias);
			}
		} else if constexpr (kOutputDimensions == 1) {
			__m256i sum0 = _mm256_setzero_si256();

			const auto row0 = reinterpret_cast<const __m256i*>(&weights_[0]);

			for (IndexType j = 0; j < kNumChunks; ++j) {
				const __m256i in = input_vector[j];

				m256_add_dpbusd_epi32(sum0, in, row0[j]);
			}

			output[0] = m256_hadd(sum0, biases_[0]);
		} else {
			// This case can never happen because kOutputDimensions
			// is always 1 or a multiple of kSimdWidth.
			ASSERT_LV5(false);
		}

#elif defined(USE_SSSE3)

		constexpr IndexType kNumChunks = kPaddedInputDimensions / kSimdWidth;

		auto       output       = reinterpret_cast<OutputType*>(buffer);
		const auto input_vector = reinterpret_cast<const __m128i*>(input);

		// kOutputDimensions is either 1 or a multiple of kSimdWidth
		// because then it is also an input dimension.
		if constexpr (kOutputDimensions % 4 == 0) {
			for (IndexType i = 0; i < kOutputDimensions; i += 4) {
				const IndexType offset0 = (i + 0) * kPaddedInputDimensions;
				const IndexType offset1 = (i + 1) * kPaddedInputDimensions;
				const IndexType offset2 = (i + 2) * kPaddedInputDimensions;
				const IndexType offset3 = (i + 3) * kPaddedInputDimensions;

				const __m128i bias   = *reinterpret_cast<const __m128i*>(&biases_[i]);
				__m128i*      outptr = reinterpret_cast<__m128i*>(&output[i]);

				__m128i sum0 = _mm_setzero_si128();
				__m128i sum1 = _mm_setzero_si128();
				__m128i sum2 = _mm_setzero_si128();
				__m128i sum3 = _mm_setzero_si128();

				const auto row0 = reinterpret_cast<const __m128i*>(&weights_[offset0]);
				const auto row1 = reinterpret_cast<const __m128i*>(&weights_[offset1]);
				const auto row2 = reinterpret_cast<const __m128i*>(&weights_[offset2]);
				const auto row3 = reinterpret_cast<const __m128i*>(&weights_[offset3]);

				int j = 0;
				if (!canSaturate16x4[i / 4]) {
					for (; j < (int)kNumChunks - 1; j += 2) {
						const __m128i in0 = input_vector[j];
						const __m128i in1 = input_vector[j + 1];

						m128_add_dpbusd_epi32x2(sum0, in0, row0[j], in1, row0[j + 1]);
						m128_add_dpbusd_epi32x2(sum1, in0, row1[j], in1, row1[j + 1]);
						m128_add_dpbusd_epi32x2(sum2, in0, row2[j], in1, row2[j + 1]);
						m128_add_dpbusd_epi32x2(sum3, in0, row3[j], in1, row3[j + 1]);
					}
				}
				for (; j < (int)kNumChunks; ++j) {
					const __m128i in = input_vector[j];

					m128_add_dpbusd_epi32(sum0, in, row0[j]);
					m128_add_dpbusd_epi32(sum1, in, row1[j]);
					m128_add_dpbusd_epi32(sum2, in, row2[j]);
					m128_add_dpbusd_epi32(sum3, in, row3[j]);
				}

				*outptr = m128_haddx4(sum0, sum1, sum2, sum3, bias);
			}
		} else if constexpr (kOutputDimensions == 1) {
			__m128i sum0 = _mm_setzero_si128();

			const auto row0 = reinterpret_cast<const __m128i*>(&weights_[0]);

			for (int j = 0; j < (int)kNumChunks; ++j) {
				const __m128i in = input_vector[j];

				m128_add_dpbusd_epi32(sum0, in, row0[j]);
			}

			output[0] = m128_hadd(sum0, biases_[0]);
		} else {
			// This case can never happen because kOutputDimensions
			// is always 1 or a multiple of kSimdWidth.
			ASSERT_LV5(false);
		}

#else

		// Use old implementation for the other architectures.

		auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_SSE2)
		constexpr IndexType kNumChunks = kPaddedInputDimensions / kSimdWidth;
#ifndef USE_SSSE3
		const __m128i kZeros = _mm_setzero_si128();
#else
		const __m128i kOnes = _mm_set1_epi16(1);
#endif
		const auto input_vector = reinterpret_cast<const __m128i*>(input);

#elif defined(USE_MMX)
		constexpr IndexType kNumChunks   = kPaddedInputDimensions / kSimdWidth;
		const __m64         kZeros       = _mm_setzero_si64();
		const auto          input_vector = reinterpret_cast<const __m64*>(input);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks   = kPaddedInputDimensions / kSimdWidth;
		const auto          input_vector = reinterpret_cast<const int8x8_t*>(input);
#endif

		for (IndexType i = 0; i < kOutputDimensions; ++i) {
			const IndexType offset = i * kPaddedInputDimensions;

#if defined(USE_SSE2)
			__m128i    sum_lo = _mm_cvtsi32_si128(biases_[i]);
			__m128i    sum_hi = kZeros;
			const auto row    = reinterpret_cast<const __m128i*>(&weights_[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m128i row_j             = _mm_load_si128(&row[j]);
				__m128i input_j           = _mm_load_si128(&input_vector[j]);
				__m128i extended_row_lo   = _mm_srai_epi16(_mm_unpacklo_epi8(row_j, row_j), 8);
				__m128i extended_row_hi   = _mm_srai_epi16(_mm_unpackhi_epi8(row_j, row_j), 8);
				__m128i extended_input_lo = _mm_unpacklo_epi8(input_j, kZeros);
				__m128i extended_input_hi = _mm_unpackhi_epi8(input_j, kZeros);
				__m128i product_lo        = _mm_madd_epi16(extended_row_lo, extended_input_lo);
				__m128i product_hi        = _mm_madd_epi16(extended_row_hi, extended_input_hi);
				sum_lo                    = _mm_add_epi32(sum_lo, product_lo);
				sum_hi                    = _mm_add_epi32(sum_hi, product_hi);
			}
			__m128i sum           = _mm_add_epi32(sum_lo, sum_hi);
			__m128i sum_high_64   = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
			sum                   = _mm_add_epi32(sum, sum_high_64);
			__m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
			sum                   = _mm_add_epi32(sum, sum_second_32);
			output[i]             = _mm_cvtsi128_si32(sum);

#elif defined(USE_MMX)
			__m64      sum_lo = _mm_cvtsi32_si64(biases_[i]);
			__m64      sum_hi = kZeros;
			const auto row    = reinterpret_cast<const __m64*>(&weights_[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m64 row_j             = row[j];
				__m64 input_j           = input_vector[j];
				__m64 extended_row_lo   = _mm_srai_pi16(_mm_unpacklo_pi8(row_j, row_j), 8);
				__m64 extended_row_hi   = _mm_srai_pi16(_mm_unpackhi_pi8(row_j, row_j), 8);
				__m64 extended_input_lo = _mm_unpacklo_pi8(input_j, kZeros);
				__m64 extended_input_hi = _mm_unpackhi_pi8(input_j, kZeros);
				__m64 product_lo        = _mm_madd_pi16(extended_row_lo, extended_input_lo);
				__m64 product_hi        = _mm_madd_pi16(extended_row_hi, extended_input_hi);
				sum_lo                  = _mm_add_pi32(sum_lo, product_lo);
				sum_hi                  = _mm_add_pi32(sum_hi, product_hi);
			}
			__m64 sum = _mm_add_pi32(sum_lo, sum_hi);
			sum       = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
			output[i] = _mm_cvtsi64_si32(sum);

#elif defined(USE_NEON)
			int32x4_t  sum = {biases_[i]};
			const auto row = reinterpret_cast<const int8x8_t*>(&weights_[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				int16x8_t product = vmull_s8(input_vector[j * 2], row[j * 2]);
				product           = vmlal_s8(product, input_vector[j * 2 + 1], row[j * 2 + 1]);
				sum               = vpadalq_s16(sum, product);
			}
			output[i] = sum[0] + sum[1] + sum[2] + sum[3];

#else
			// CPUに依存しないコード
			OutputType sum = biases_[i];
			for (IndexType j = 0; j < kInputDimensions; ++j) {
				sum += weights_[offset + j] * input[j];
			}
			output[i] = sum;
#endif
		}
#if defined(USE_MMX)
		_mm_empty();
#endif

#endif

		return output;
	}

   private:
	// パラメータの型
	using BiasType   = OutputType;
	using WeightType = std::int8_t;

	// 学習用クラスをfriendにする
	friend class Trainer<AffineTransform>;

	// この層の直前の層
	PreviousLayer previous_layer_;

	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
	alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
	union {
		uint32_t canSaturate16x4[(kOutputDimensions + 3) / 4];
		bool     canSaturate16[kOutputDimensions];
	};
};

}  // namespace Eval::NNUE::Layers

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
