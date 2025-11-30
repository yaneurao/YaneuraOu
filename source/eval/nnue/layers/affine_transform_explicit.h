// Definition of layer AffineTransformExplicit of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_EXPLICIT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_EXPLICIT_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "simd.h"
#include "affine_transform.h" // For affine_transform_unaligned

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// Affine transformation layer (Explicit Dimensions)
template<IndexType InputDimensions, IndexType OutputDimensions>
class AffineTransformExplicit {
   public:
        // Input/output type
        using InputType = std::uint8_t;
        using OutputType = std::int32_t;

        // Number of input/output dimensions
        static constexpr IndexType kInputDimensions       = InputDimensions;
        static constexpr IndexType kOutputDimensions      = OutputDimensions;
        static constexpr IndexType kPaddedInputDimensions = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);
        static constexpr IndexType kPaddedOutputDimensions = CeilToMultiple<IndexType>(kOutputDimensions, kMaxSimdWidth);

        using OutputBuffer = OutputType[kPaddedOutputDimensions];

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
                std::uint32_t hash_value = 0xCC03DAE4u;
                hash_value += kOutputDimensions;
                hash_value ^= prevHash >> 1;
                hash_value ^= prevHash << 31;
                return hash_value;
        }

        // Structure string
        static std::string GetStructureString() {
                return "AffineTransform[" + std::to_string(kOutputDimensions) + "<-" + std::to_string(kInputDimensions) + "]";
        }

        static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (kPaddedInputDimensions / 4) * kOutputDimensions * 4
             + i / kPaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
        return get_weight_index_scrambled(i);
#else
        return i;
#endif
    }

        // Read network parameters
        Tools::Result ReadParameters(std::istream& stream) {
                for (std::size_t i = 0; i < kOutputDimensions; ++i)
                        biases_[i] = read_little_endian<BiasType>(stream);
                for (std::size_t i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
                        weights_[get_weight_index(IndexType(i))] = read_little_endian<WeightType>(stream);
                return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
        }

        // Write network parameters
        bool WriteParameters(std::ostream& stream) const {
                stream.write(reinterpret_cast<const char*>(biases_), kOutputDimensions * sizeof(BiasType));
                stream.write(reinterpret_cast<const char*>(weights_),
                             kOutputDimensions * kPaddedInputDimensions * sizeof(WeightType));
                return !stream.fail();
        }

        // Forward propagation
        void Propagate(const InputType* input, OutputType* output) const {
#if defined(USE_WASM_SIMD)
                {
                        constexpr int n = kInputDimensions;
                        constexpr int m = kOutputDimensions;
                        constexpr int n_stride = kPaddedInputDimensions;
                        auto A = *reinterpret_cast<const int8_t(*)[m][n_stride]>(weights_);
                        auto x = *reinterpret_cast<const uint8_t(*)[n]>(input);
                        auto b = *reinterpret_cast<const int32_t(*)[m]>(biases_);
                        auto y = *reinterpret_cast<int32_t(*)[m]>(output);
                        emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
                        return; // void return
                }
#endif

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

                        {}
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
        affine_transform_unaligned<kInputDimensions, kPaddedInputDimensions, kOutputDimensions>(
          output, weights_, biases_, input);
#endif
        }

   private:
        using BiasType   = OutputType;
        using WeightType = std::int8_t;

        friend class Trainer<AffineTransformExplicit>;

        alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
        alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // NNUE_LAYERS_AFFINE_TRANSFORM_EXPLICIT_H_INCLUDED
