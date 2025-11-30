// Definition of layer AffineTransformSparseInputExplicit of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_EXPLICIT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_EXPLICIT_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "affine_transform.h" // For affine_transform_unaligned
#include "simd.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

#if defined(USE_SSSE3) || USE_NEON >= 8

alignas(kCacheLineSize) static inline const
  std::array<std::array<std::uint16_t, 8>, 256> lookup_indices_explicit = []() {
      std::array<std::array<std::uint16_t, 8>, 256> v{};
      for (unsigned i = 0; i < 256; ++i)
      {
          std::uint64_t j = i, k = 0;
          while (j)
              v[i][k++] = pop_lsb(j);
      }
      return v;
  }();

// Find indices of nonzero numbers in an int32_t array
template<const IndexType kInputDimensions>
static void find_nnz_explicit(const std::int32_t* input, std::uint16_t* out, IndexType& count_out) {
#if defined(USE_SSSE3)
#if defined(USE_AVX512)
    using vec_t = __m512i;
#define vec_nnz(a) _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
#elif defined(USE_AVX2)
    using vec_t = __m256i;
#if defined(USE_VNNI) && !defined(USE_AVXVNNI)
#define vec_nnz(a) _mm256_cmpgt_epi32_mask(a, _mm256_setzero_si256())
#else
#define vec_nnz(a) \
                        _mm256_movemask_ps( \
                    _mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())))
        #endif
#elif defined(USE_SSSE3)
    using vec_t = __m128i;
#define vec_nnz(a) \
                _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128())))
#endif
    using vec128_t = __m128i;
#define vec128_zero _mm_setzero_si128()
#define vec128_set_16(a) _mm_set1_epi16(a)
#define vec128_load(a) _mm_load_si128(a)
#define vec128_storeu(a, b) _mm_storeu_si128(a, b)
#define vec128_add(a, b) _mm_add_epi16(a, b)
#elif defined(USE_NEON)
    using vec_t                        = uint32x4_t;
        static constexpr std::uint32_t Mask[4] = {1, 2, 4, 8};
#define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    using vec128_t                     = uint16x8_t;
#define vec128_zero vdupq_n_u16(0)
#define vec128_set_16(a) vdupq_n_u16(a)
#define vec128_load(a) vld1q_u16(reinterpret_cast<const std::uint16_t*>(a))
#define vec128_storeu(a, b) vst1q_u16(reinterpret_cast<std::uint16_t*>(a), b)
#define vec128_add(a, b) vaddq_u16(a, b)
#endif

    constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(std::int32_t);
    // Inputs are processed kInputSimdWidth at a time and outputs are processed 8 at a time so we process in chunks of max(kInputSimdWidth, 8)
    constexpr IndexType kChunkSize       = std::max<IndexType>(kInputSimdWidth, 8);
    constexpr IndexType kNumChunks       = kInputDimensions / kChunkSize;
    constexpr IndexType kInputsPerChunk  = kChunkSize / kInputSimdWidth;
    constexpr IndexType kOutputsPerChunk = kChunkSize / 8;

    const auto     inputVector = reinterpret_cast<const vec_t*>(input);
    IndexType      count       = 0;
    vec128_t       base        = vec128_zero;
    const vec128_t increment   = vec128_set_16(8);
    for (IndexType i = 0; i < kNumChunks; ++i)
    {
        // bitmask of nonzero values in this chunk
        unsigned nnz = 0;
        for (IndexType j = 0; j < kInputsPerChunk; ++j)
        {
            const vec_t inputChunk = inputVector[i * kInputsPerChunk + j];
            nnz |= unsigned(vec_nnz(inputChunk)) << (j * kInputSimdWidth);
        }
        for (IndexType j = 0; j < kOutputsPerChunk; ++j)
        {
            const auto lookup = (nnz >> (j * 8)) & 0xFF;
            const auto offsets =
              vec128_load(reinterpret_cast<const vec128_t*>(&lookup_indices_explicit[lookup]));
            vec128_storeu(reinterpret_cast<vec128_t*>(out + count), vec128_add(base, offsets));
            count += POPCNT32(lookup);
            base = vec128_add(base, increment);
        }
    }
    count_out = count;
}
#undef vec_nnz
#undef vec128_zero
#undef vec128_set_16
#undef vec128_load
#undef vec128_storeu
#undef vec128_add

#endif

// AffineTransform layer that takes block-sparse input (Explicit Dimensions)
template <IndexType InputDimensions, IndexType OutputDimensions>
class AffineTransformSparseInputExplicit {
public:
        using InputType = std::uint8_t;
        using OutputType = std::int32_t;

        // Number of input/output dimensions
        static constexpr IndexType kInputDimensions       = InputDimensions;
        static constexpr IndexType kOutputDimensions      = OutputDimensions;
        static constexpr IndexType kPaddedInputDimensions = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);
        static constexpr IndexType kPaddedOutputDimensions = CeilToMultiple<IndexType>(kOutputDimensions, kMaxSimdWidth);

#if defined(USE_SSSE3) || USE_NEON >= 8
    static constexpr IndexType kChunkSize = 4;
#else
    static constexpr IndexType kChunkSize = 1;
#endif

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
                return "AffineTransformSparseInput[" + std::to_string(kOutputDimensions) + "<-" + std::to_string(kInputDimensions) + "]";
        }

        static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / kChunkSize) % (kPaddedInputDimensions / kChunkSize) * kOutputDimensions * kChunkSize
             + i / kPaddedInputDimensions * kChunkSize + i % kChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || USE_NEON >= 8
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

#if defined(USE_SSSE3) || USE_NEON >= 8

#if defined(USE_AVX512)
        if constexpr (kOutputDimensions % 16 == 0)
        {
            constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 16;
            std::uint16_t       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const std::int32_t*>(input);

            find_nnz_explicit<kNumChunks>(input32, nnz, count);

            const __m512i* biasvec = reinterpret_cast<const __m512i*>(biases_);
            __m512i        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto    i  = nnz[j];
                const __m512i in = _mm512_set1_epi32(input32[i]);
                const auto    col =
                reinterpret_cast<const __m512i*>(&weights_[i * kOutputDimensions * kChunkSize]);
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
            constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 8;
            std::uint16_t       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const std::int32_t*>(input);

            find_nnz_explicit<kNumChunks>(input32, nnz, count);

            const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases_);
            __m256i        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto    i  = nnz[j];
                const __m256i in = _mm256_set1_epi32(input32[i]);
                const auto    col =
                reinterpret_cast<const __m256i*>(&weights_[i * kOutputDimensions * kChunkSize]);
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
            constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 4;
            std::uint16_t       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const std::int32_t*>(input);

            find_nnz_explicit<kNumChunks>(input32, nnz, count);

            const __m128i* biasvec = reinterpret_cast<const __m128i*>(biases_);
            __m128i        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto    i  = nnz[j];
                const __m128i in = _mm_set1_epi32(input32[i]);
                const auto    col =
                reinterpret_cast<const __m128i*>(&weights_[i * kOutputDimensions * kChunkSize]);
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
        if constexpr (kOutputDimensions % 8 == 0)
        {
            constexpr IndexType kNumChunks = CeilToMultiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 8;
            std::uint16_t       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const std::int32_t*>(input);

            find_nnz_explicit<kNumChunks>(input32, nnz, count);

            const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases_);
            int32x4_t        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto      i  = nnz[j];
                const int8x16_t in = vreinterpretq_s8_u32(vdupq_n_u32(input32[i]));
                const auto     col =
                reinterpret_cast<const int8x16_t*>(&weights_[i * kOutputDimensions * kChunkSize]);
                for (IndexType k = 0; k < kNumRegs; ++k)
                    Simd::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
            }

            int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);

            for (IndexType k = 0; k < kNumRegs; ++k)
                outptr[k] = acc[k];
        }
        else
#endif
            affine_transform_unaligned<kInputDimensions, kPaddedInputDimensions, kOutputDimensions>(
              output, weights_, biases_, input);

#undef vec_set_32
#undef vec_add_dpbusd_32

#else
        // Use dense implementation for the other architectures.
        affine_transform_unaligned<kInputDimensions, kPaddedInputDimensions, kOutputDimensions>(
          output, weights_, biases_, input);
#endif
        }

   private:
        using BiasType   = OutputType;
        using WeightType = std::int8_t;

        friend class Trainer<AffineTransformSparseInputExplicit>;

        alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
        alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_EXPLICIT_H_INCLUDED
