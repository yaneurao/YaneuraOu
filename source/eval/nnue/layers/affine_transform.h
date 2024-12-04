// Definition of layer AffineTransform of NNUE evaluation function
// NNUE評価関数の層AffineTransformの定義

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "simd.h"

namespace Eval::NNUE::Layers {

template<IndexType kInputDimensions, IndexType kPaddedInputDimensions, IndexType kOutputDimensions>
static void affine_transform_non_ssse3(std::int32_t*       output,
                                       const std::int8_t*  weights,
                                       const std::int32_t* biases,
                                       const std::uint8_t* input) {
#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
#if defined(USE_SSE2)
    // At least a multiple of 16, with SSE2.
    constexpr IndexType kNumChunks   = CeilToMultiple<IndexType>(kInputDimensions, 16) / 16;
    const __m128i       kZeros       = _mm_setzero_si128();
    const auto          inputVector = reinterpret_cast<const __m128i*>(input);

#elif defined(USE_NEON)
    constexpr IndexType kNumChunks   = CeilToMultiple<IndexType>(kInputDimensions, 16) / 16;
    const auto          inputVector = reinterpret_cast<const int8x8_t*>(input);
#endif

    for (IndexType i = 0; i < kOutputDimensions; ++i)
    {
        const IndexType offset = i * kPaddedInputDimensions;

#if defined(USE_SSE2)
        __m128i    sumLo = _mm_cvtsi32_si128(biases[i]);
        __m128i    sumHi = kZeros;
        const auto row   = reinterpret_cast<const __m128i*>(&weights[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j)
        {
            __m128i row_j           = _mm_load_si128(&row[j]);
            __m128i input_j         = _mm_load_si128(&inputVector[j]);
            __m128i extendedRowLo   = _mm_srai_epi16(_mm_unpacklo_epi8(row_j, row_j), 8);
            __m128i extendedRowHi   = _mm_srai_epi16(_mm_unpackhi_epi8(row_j, row_j), 8);
            __m128i extendedInputLo = _mm_unpacklo_epi8(input_j, kZeros);
            __m128i extendedInputHi = _mm_unpackhi_epi8(input_j, kZeros);
            __m128i productLo       = _mm_madd_epi16(extendedRowLo, extendedInputLo);
            __m128i productHi       = _mm_madd_epi16(extendedRowHi, extendedInputHi);
            sumLo                   = _mm_add_epi32(sumLo, productLo);
            sumHi                   = _mm_add_epi32(sumHi, productHi);
        }
        __m128i sum           = _mm_add_epi32(sumLo, sumHi);
        __m128i sumHigh_64    = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
        sum                   = _mm_add_epi32(sum, sumHigh_64);
        __m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
        sum                   = _mm_add_epi32(sum, sum_second_32);
        output[i]             = _mm_cvtsi128_si32(sum);

#elif defined(USE_NEON)

        int32x4_t  sum = {biases[i]};
        const auto row = reinterpret_cast<const int8x8_t*>(&weights[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j)
        {
            int16x8_t product = vmull_s8(inputVector[j * 2], row[j * 2]);
            product           = vmlal_s8(product, inputVector[j * 2 + 1], row[j * 2 + 1]);
            sum               = vpadalq_s16(sum, product);
        }
        output[i] = sum[0] + sum[1] + sum[2] + sum[3];

#endif
    }
#else
    std::memcpy(output, biases, sizeof(std::int32_t) * kOutputDimensions);

    // Traverse weights in transpose order to take advantage of input sparsity
    for (IndexType i = 0; i < kInputDimensions; ++i)
        if (input[i])
        {
            const std::int8_t* w  = &weights[i];
            const int          in = input[i];
            for (IndexType j = 0; j < kOutputDimensions; ++j)
                output[j] += w[j * kPaddedInputDimensions] * in;
        }
#endif
}

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

	static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (kPaddedInputDimensions / 4) * kOutputDimensions * 4
             + i / kPaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
        return kOutputDimensions % 4 == 0 ? get_weight_index_scrambled(i) : i;
#else
        return i;
#endif
    }

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
		Tools::Result result = previous_layer_.ReadParameters(stream);
		if (result.is_not_ok()) return result;
		for (std::size_t i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
			weights_[get_weight_index(IndexType(i))] = read_little_endian<WeightType>(stream);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
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
		const auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)

		if constexpr (kOutputDimensions > 1)
		{
#if defined(USE_AVX512)
			if constexpr (kOutputDimensions % 16 == 0)
			{
				constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = kOutputDimensions / 16;

				const auto   input32 = reinterpret_cast<const std::int32_t*>(input);
				const __m512i* biasvec = reinterpret_cast<const __m512i*>(biases_);
				__m512i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m512i in = _mm512_set1_epi32(input32[i]);
					const auto  col  = reinterpret_cast<const __m512i*>(&weights_[i * kOutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						Simd::m512_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m512i* outptr = reinterpret_cast<__m512i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_AVX2)
			if constexpr (kOutputDimensions % 8 == 0)
			{
				constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = kOutputDimensions / 8;

				const auto   input32 = reinterpret_cast<const std::int32_t*>(input);
				const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases_);
				__m256i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m256i in = _mm256_set1_epi32(input32[i]);
					const auto  col  = reinterpret_cast<const __m256i*>(&weights_[i * kOutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						Simd::m256_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m256i* outptr = reinterpret_cast<__m256i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_SSSE3)
			if constexpr (kOutputDimensions % 4 == 0)
			{
				constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = kOutputDimensions / 4;

				const auto   input32 = reinterpret_cast<const std::int32_t*>(input);
				const __m128i* biasvec = reinterpret_cast<const __m128i*>(biases_);
				__m128i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m128i in = _mm_set1_epi32(input32[i]);
					const auto  col = reinterpret_cast<const __m128i*>(&weights_[i * kOutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						Simd::m128_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m128i* outptr = reinterpret_cast<__m128i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_NEON_DOTPROD)
			if constexpr (kOutputDimensions % 4 == 0)
			{
				constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = kOutputDimensions / 4;

				const auto       input32 = reinterpret_cast<const std::int32_t*>(input);
				const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases_);
				int32x4_t        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const int32x4_t in = vdupq_n_s32(input32[i]);
					const auto  col = reinterpret_cast<const int32x4_t*>(&weights_[i * kOutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						Simd::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

				affine_transform_non_ssse3<kInputDimensions, kPaddedInputDimensions, kOutputDimensions>(
				output, weights_, biases_, input);
		}
		else if constexpr (kOutputDimensions == 1)
		{
    // We cannot use AVX512 for the last layer because there are only 32 inputs
    // and the buffer is not padded to 64 elements.
#if defined(USE_AVX2)
			using vec_t = __m256i;
#define vec_setzero() _mm256_setzero_si256()
#define vec_set_32 _mm256_set1_epi32
#define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
#define vec_hadd Simd::m256_hadd
#elif defined(USE_SSSE3)
			using vec_t = __m128i;
#define vec_setzero() _mm_setzero_si128()
#define vec_set_32 _mm_set1_epi32
#define vec_add_dpbusd_32 Simd::m128_add_dpbusd_epi32
#define vec_hadd Simd::m128_hadd
#elif defined(USE_NEON_DOTPROD)
			using vec_t = int32x4_t;
#define vec_setzero() vdupq_n_s32(0)
#define vec_set_32 vdupq_n_s32
#define vec_add_dpbusd_32(acc, a, b) \
			Simd::dotprod_m128_add_dpbusd_epi32(acc, vreinterpretq_s8_s32(a), \
												vreinterpretq_s8_s32(b))
#define vec_hadd Simd::neon_m128_hadd
#endif

			const auto inputVector = reinterpret_cast<const vec_t*>(input);
            static constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(InputType);

            static_assert(kPaddedInputDimensions % kInputSimdWidth == 0);

            constexpr IndexType kNumChunks = kPaddedInputDimensions / kInputSimdWidth;
            vec_t               sum0      = vec_setzero();
            const auto          row0      = reinterpret_cast<const vec_t*>(&weights_[0]);

            for (int j = 0; j < int(kNumChunks); ++j)
            {
                const vec_t in = inputVector[j];
                vec_add_dpbusd_32(sum0, in, row0[j]);
            }

            output[0] = vec_hadd(sum0, biases_[0]);

#undef vec_setzero
#undef vec_set_32
#undef vec_add_dpbusd_32
#undef vec_hadd
		}

#else
        // Use dense implementation for the other architectures.
        affine_transform_non_ssse3<kInputDimensions, kPaddedInputDimensions, kOutputDimensions>(
          output, weights_, biases_, input);
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
};

}  // namespace Eval::NNUE::Layers

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
