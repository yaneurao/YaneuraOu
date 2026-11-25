// Definition of layer AffineTransform of NNUE evaluation function
// NNUE評価関数の層AffineTransformの定義

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <cstdint>
#include <iostream>

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"
#include "../simd.h"

namespace YaneuraOu::Eval::NNUE::Layers {

// 旧NNUE・SFNNで共用する非整列入力向けの実装。既存のSIMD経路を維持する。

template<IndexType InputDimensions, IndexType PaddedInputDimensions, IndexType OutputDimensions>
static void affine_transform_unaligned(i32*       output,
                                       const i8*  weights,
                                       const i32* biases,
                                       const u8* input) {
#if defined(USE_SSE2) || defined(USE_NEON)

#if defined(USE_AVX512)
	constexpr IndexType kNumChunks  = ceil_to_multiple<IndexType>(InputDimensions, 64) / 64;
    const __m512i       kZeros      = _mm512_setzero_si512();
    const auto          inputVector = reinterpret_cast<const __m512i*>(input);
#elif defined(USE_AVX2)
	constexpr IndexType kNumChunks  = ceil_to_multiple<IndexType>(InputDimensions, 32) / 32;
    const __m256i       kZeros      = _mm256_setzero_si256();
    const auto          inputVector = reinterpret_cast<const __m256i*>(input);
#elif defined(USE_SSE2)
    // At least a multiple of 16, with SSE2.
    constexpr IndexType kNumChunks  = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const __m128i       kZeros      = _mm_setzero_si128();
    const auto          inputVector = reinterpret_cast<const __m128i*>(input);

#elif defined(USE_NEON)
    constexpr IndexType kNumChunks  = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const auto          inputVector = reinterpret_cast<const int8x8_t*>(input);
#endif

    for (IndexType i = 0; i < OutputDimensions; ++i)
    {
        const IndexType offset = i * PaddedInputDimensions;

#if defined(USE_AVX512)

        __m512i    sumLo = _mm512_castsi128_si512(_mm_cvtsi32_si128(biases[i]));
        __m512i    sumHi = _mm512_setzero_si512();
        const auto row   = reinterpret_cast<const __m512i*>(&weights[offset]);

        for (IndexType j = 0; j < kNumChunks; ++j)
        {
            __m512i row_j           = _mm512_load_si512(&row[j]);
            __m512i input_j         = _mm512_load_si512(reinterpret_cast<const __m512i*>(&inputVector[j]));
            __m512i extendedRowLo   = _mm512_srai_epi16(_mm512_unpacklo_epi8(row_j, row_j), 8);
            __m512i extendedRowHi   = _mm512_srai_epi16(_mm512_unpackhi_epi8(row_j, row_j), 8);
            __m512i extendedInputLo = _mm512_unpacklo_epi8(input_j, _mm512_setzero_si512());
            __m512i extendedInputHi = _mm512_unpackhi_epi8(input_j, _mm512_setzero_si512());
            __m512i productLo       = _mm512_madd_epi16(extendedRowLo, extendedInputLo);
            __m512i productHi       = _mm512_madd_epi16(extendedRowHi, extendedInputHi);
            sumLo                   = _mm512_add_epi32(sumLo, productLo);
            sumHi                   = _mm512_add_epi32(sumHi, productHi);
        }

        __m512i sum       = _mm512_add_epi32(sumLo, sumHi);
        __m256i sumLow256 = _mm512_castsi512_si256(sum);
        __m256i sumHigh256 = _mm512_extracti64x4_epi64(sum, 1);
        __m256i finalSum256 = _mm256_add_epi32(sumLow256, sumHigh256);

        __m128i sumLow128  = _mm256_castsi256_si128(finalSum256);
        __m128i sumHigh128 = _mm256_extracti128_si256(finalSum256, 1);
        __m128i finalSum   = _mm_add_epi32(sumLow128, sumHigh128);

        __m128i sumHigh_64    = _mm_shuffle_epi32(finalSum, _MM_SHUFFLE(1, 0, 3, 2));
        finalSum              = _mm_add_epi32(finalSum, sumHigh_64);
        __m128i sum_second_32 = _mm_shufflelo_epi16(finalSum, _MM_SHUFFLE(1, 0, 3, 2));
        finalSum              = _mm_add_epi32(finalSum, sum_second_32);
        output[i]             = _mm_cvtsi128_si32(finalSum);

#elif defined(USE_AVX2)

        __m256i    sumLo = _mm256_castsi128_si256(_mm_cvtsi32_si128(biases[i]));
        __m256i    sumHi = _mm256_setzero_si256();
        const auto row   = reinterpret_cast<const __m256i*>(&weights[offset]);

        for (IndexType j = 0; j < kNumChunks; ++j)
        {
            __m256i row_j           = _mm256_load_si256(&row[j]);
            __m256i input_j         = _mm256_load_si256(reinterpret_cast<const __m256i*>(&inputVector[j]));
            __m256i extendedRowLo   = _mm256_srai_epi16(_mm256_unpacklo_epi8(row_j, row_j), 8);
            __m256i extendedRowHi   = _mm256_srai_epi16(_mm256_unpackhi_epi8(row_j, row_j), 8);
            __m256i extendedInputLo = _mm256_unpacklo_epi8(input_j, _mm256_setzero_si256());
            __m256i extendedInputHi = _mm256_unpackhi_epi8(input_j, _mm256_setzero_si256());
            __m256i productLo       = _mm256_madd_epi16(extendedRowLo, extendedInputLo);
            __m256i productHi       = _mm256_madd_epi16(extendedRowHi, extendedInputHi);
            sumLo                   = _mm256_add_epi32(sumLo, productLo);
            sumHi                   = _mm256_add_epi32(sumHi, productHi);
        }

        __m256i sum           = _mm256_add_epi32(sumLo, sumHi);
        __m128i sumLow128     = _mm256_castsi256_si128(sum);
        __m128i sumHigh128    = _mm256_extracti128_si256(sum, 1);
        __m128i finalSum      = _mm_add_epi32(sumLow128, sumHigh128);
        __m128i sumHigh_64    = _mm_shuffle_epi32(finalSum, _MM_SHUFFLE(1, 0, 3, 2));
        finalSum              = _mm_add_epi32(finalSum, sumHigh_64);
        __m128i sum_second_32 = _mm_shufflelo_epi16(finalSum, _MM_SHUFFLE(1, 0, 3, 2));
        finalSum              = _mm_add_epi32(finalSum, sum_second_32);
        output[i]             = _mm_cvtsi128_si32(finalSum);

#elif defined(USE_SSE2)

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
    std::memcpy(output, biases, sizeof(i32) * OutputDimensions);

    // Traverse weights in transpose order to take advantage of input sparsity
    for (IndexType i = 0; i < InputDimensions; ++i)
        if (input[i])
        {
            const i8* w  = &weights[i];
            const int          in = input[i];
            for (IndexType j = 0; j < OutputDimensions; ++j)
                output[j] += w[j * PaddedInputDimensions] * in;
        }
#endif
}

#if defined(EVAL_SFNN)
// Affine transformation layer
// 入力・出力の次元を直接指定するSFNN用レイヤー。
// Stockfishと同じAPIを使い、既存の重み配置・パディング・融合計算は維持する。
template<IndexType InDims, IndexType OutDims>
class AffineTransform {
        template<IndexType, IndexType>
        friend class AffineTransform;

   public:
        // Input/output type
        // 入出力の型
        using InputType = u8;
        using OutputType = i32;

        // Number of input/output dimensions
        // 入出力の次元数
        static constexpr IndexType InputDimensions       = InDims;
        static constexpr IndexType OutputDimensions      = OutDims;
        static constexpr IndexType PaddedInputDimensions = ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
        static constexpr IndexType PaddedOutputDimensions = ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);
        static constexpr bool kHasClippedReLUPackedWeights =
                InputDimensions == 64 && OutputDimensions == 1;
        static constexpr IndexType kClippedReLUPackedWeightDimensions =
                kHasClippedReLUPackedWeights ? PaddedInputDimensions : 1;

        using OutputBuffer = OutputType[PaddedOutputDimensions];

        // Hash value embedded in the evaluation file
        // 評価関数ファイルに埋め込むハッシュ値
        static constexpr u32 get_hash_value(u32 prevHash) {
                u32 hashValue = 0xCC03DAE4u;
                hashValue += OutputDimensions;
                hashValue ^= prevHash >> 1;
                hashValue ^= prevHash << 31;
                return hashValue;
        }

        // Structure string
        // この層の構造を表す文字列
        static std::string get_structure_string() {
                return "AffineTransform[" + std::to_string(OutputDimensions) + "<-" + std::to_string(InputDimensions) + "]";
        }

        static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
             + i / PaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
        return OutputDimensions % 4 == 0 ? get_weight_index_scrambled(i) : i;
#else
        return i;
#endif
    }

        // Read network parameters
        // パラメーターを読み込む。重みの並び替えと融合計算用の複製は従来どおり行う。
        bool read_parameters(std::istream& stream) {
                for (usize i = 0; i < OutputDimensions; ++i)
                        biases[i] = read_little_endian<BiasType>(stream);
                for (usize i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
                        weights[get_weight_index(IndexType(i))] = read_little_endian<WeightType>(stream);
                if constexpr (kHasClippedReLUPackedWeights) {
                        constexpr IndexType kDwordCount = PaddedInputDimensions / 4;
                        constexpr IndexType kPerm[kDwordCount] = {
                                0, 4, 8, 12, 1, 5, 9, 13,
                                2, 6, 10, 14, 3, 7, 11, 15};
                        for (IndexType lane = 0; lane < kDwordCount; ++lane) {
                                const IndexType src_lane = kPerm[lane];
                                for (IndexType b = 0; b < 4; ++b) {
                                        weights_clipped_relu_packed_[src_lane * 4 + b] =
                                                weights[lane * 4 + b];
                                }
                        }
                }
                return !stream.fail();
        }

        // Write network parameters
        // パラメーターを書き込む。既存の内部配置での書き出しを維持する。
        bool write_parameters(std::ostream& stream) const {
                write_little_endian(stream, biases, OutputDimensions);
                write_little_endian(stream, weights, OutputDimensions * PaddedInputDimensions);
                return !stream.fail();
        }

        // Forward propagation
        // 順伝播
        void propagate(const InputType* input, OutputType* output) const {
#if defined(USE_WASM_SIMD)
                {
                        constexpr int n = InputDimensions;
                        constexpr int m = OutputDimensions;
                        constexpr int n_stride = PaddedInputDimensions;
                        auto A = *reinterpret_cast<const i8(*)[m][n_stride]>(weights);
                        auto x = *reinterpret_cast<const u8(*)[n]>(input);
                        auto b = *reinterpret_cast<const i32(*)[m]>(biases);
                        auto y = *reinterpret_cast<i32(*)[m]>(output);
                        emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
                        return; // void return
                }
#endif

#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)

                if constexpr (OutputDimensions > 1)
                {
#if defined(USE_AVX512)
                        if constexpr (OutputDimensions % 16 == 0)
                        {
                                constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
                                constexpr IndexType kNumRegs = OutputDimensions / 16;

                                constexpr IndexType kNumAccums = kNumRegs;
#if defined(USE_NNUE_VNNI)
                                constexpr IndexType kActualNumRegs = 2 * kNumAccums;
#else
                                constexpr IndexType kActualNumRegs = kNumAccums;
#endif

                                const auto   input32 = reinterpret_cast<const i32*>(input);
                                const __m512i* biasvec = reinterpret_cast<const __m512i*>(biases);
                                __m512i        acc[kActualNumRegs];

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        acc[k] = biasvec[k];
#if defined(USE_NNUE_VNNI)
                                for (IndexType k = kNumAccums; k < kActualNumRegs; ++k)
                                        acc[k] = _mm512_setzero_si512();
#endif

#if defined(USE_NNUE_VNNI)
                                IndexType i = 0;
                                for (; i + 1 < kNumChunks; i += 2)
                                {
                                        const __m512i in0 = _mm512_set1_epi32(input32[i]);
                                        const __m512i in1 = _mm512_set1_epi32(input32[i + 1]);
                                        const auto col0 = reinterpret_cast<const __m512i*>(&weights[i * OutputDimensions * 4]);
                                        const auto col1 = reinterpret_cast<const __m512i*>(&weights[(i + 1) * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumAccums; ++k) {
                                                SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
                                                SIMD::m512_add_dpbusd_epi32(acc[k + kNumAccums], in1, col1[k]);
                                        }
                                }

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        acc[k] = _mm512_add_epi32(acc[k], acc[k + kNumAccums]);

                                for (; i < kNumChunks; ++i)
#else
                                for (IndexType i = 0; i < kNumChunks; ++i)
#endif
                                {
                                        const __m512i in = _mm512_set1_epi32(input32[i]);
                                        const auto  col  = reinterpret_cast<const __m512i*>(&weights[i * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumAccums; ++k)
                                                SIMD::m512_add_dpbusd_epi32(acc[k], in, col[k]);
                                }

                                __m512i* outptr = reinterpret_cast<__m512i*>(output);

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        outptr[k] = acc[k];
                        }
                        else
#endif

#if defined(USE_AVX2)
                        if constexpr (OutputDimensions % 8 == 0)
                        {
                                constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
                                constexpr IndexType kNumRegs = OutputDimensions / 8;

                                constexpr IndexType kNumAccums = kNumRegs;
#if defined(USE_AVXVNNI)
                                constexpr IndexType kActualNumRegs = 2 * kNumAccums;
#else
                                constexpr IndexType kActualNumRegs = kNumAccums;
#endif

                                const auto   input32 = reinterpret_cast<const i32*>(input);
                                const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases);
                                __m256i        acc[kActualNumRegs];

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        acc[k] = biasvec[k];
#if defined(USE_AVXVNNI)
                                for (IndexType k = kNumAccums; k < kActualNumRegs; ++k)
                                        acc[k] = _mm256_setzero_si256();
#endif

#if defined(USE_AVXVNNI)
                                IndexType i = 0;
                                for (; i + 1 < kNumChunks; i += 2)
                                {
                                        const __m256i in0 = _mm256_set1_epi32(input32[i]);
                                        const __m256i in1 = _mm256_set1_epi32(input32[i + 1]);
                                        const auto col0 = reinterpret_cast<const __m256i*>(&weights[i * OutputDimensions * 4]);
                                        const auto col1 = reinterpret_cast<const __m256i*>(&weights[(i + 1) * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumAccums; ++k) {
                                                SIMD::m256_add_dpbusd_epi32(acc[k], in0, col0[k]);
                                                SIMD::m256_add_dpbusd_epi32(acc[k + kNumAccums], in1, col1[k]);
                                        }
                                }

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        acc[k] = _mm256_add_epi32(acc[k], acc[k + kNumAccums]);

                                for (; i < kNumChunks; ++i)
#else
                                for (IndexType i = 0; i < kNumChunks; ++i)
#endif
                                {
                                        const __m256i in = _mm256_set1_epi32(input32[i]);
                                        const auto  col  = reinterpret_cast<const __m256i*>(&weights[i * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumAccums; ++k)
                                                SIMD::m256_add_dpbusd_epi32(acc[k], in, col[k]);
                                }

                                __m256i* outptr = reinterpret_cast<__m256i*>(output);

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        outptr[k] = acc[k];
                        }
                        else
#endif

#if defined(USE_SSSE3)
                        if constexpr (OutputDimensions % 4 == 0)
                        {
                                constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
                                constexpr IndexType kNumRegs = OutputDimensions / 4;

                                const auto   input32 = reinterpret_cast<const i32*>(input);
                                const __m128i* biasvec = reinterpret_cast<const __m128i*>(biases);
                                __m128i        acc[kNumRegs];

                                for (IndexType k = 0; k < kNumRegs; ++k)
                                        acc[k] = biasvec[k];

                                for (IndexType i = 0; i < kNumChunks; ++i)
                                {
                                        const __m128i in = _mm_set1_epi32(input32[i]);
                                        const auto  col = reinterpret_cast<const __m128i*>(&weights[i * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumRegs; ++k)
                                                SIMD::m128_add_dpbusd_epi32(acc[k], in, col[k]);
                                }

                                __m128i* outptr = reinterpret_cast<__m128i*>(output);

                                for (IndexType k = 0; k < kNumRegs; ++k)
                                        outptr[k] = acc[k];
                        }
                        else
#endif

#if defined(USE_NEON_DOTPROD)
                        if constexpr (OutputDimensions % (sizeof(int32x4_t) / sizeof(OutputType)) == 0)
                        {
                                constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
                                constexpr IndexType kOutputSimdWidth = sizeof(int32x4_t) / sizeof(OutputType);
                                constexpr IndexType kNumRegs = OutputDimensions / kOutputSimdWidth;

                                const auto       input32 = reinterpret_cast<const i32*>(input);
                                const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases);
                                int32x4_t        acc[kNumRegs];

                                for (IndexType k = 0; k < kNumRegs; ++k)
                                        acc[k] = biasvec[k];

                                for (IndexType i = 0; i < kNumChunks; ++i)
                                {
                                        const int32x4_t in = vdupq_n_s32(input32[i]);
                                        const auto  col = reinterpret_cast<const int32x4_t*>(&weights[i * OutputDimensions * 4]);

                                        for (IndexType k = 0; k < kNumRegs; ++k)
                                                SIMD::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
                                }

                                int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);

                                for (IndexType k = 0; k < kNumRegs; ++k)
                                        outptr[k] = acc[k];
                        }
                        else
#endif

                        affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
                          output, weights, biases, input);
                }
                else if constexpr (OutputDimensions == 1)
                {
    // We cannot use AVX512 for the last layer because there are only 32 inputs
    // and the buffer is not padded to 64 elements.
#if defined(USE_AVX2)
                        using vec_t = __m256i;
#define vec_setzero() _mm256_setzero_si256()
#define vec_set_32 _mm256_set1_epi32
#define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
#define vec_hadd SIMD::m256_hadd
#elif defined(USE_SSSE3)
                        using vec_t = __m128i;
#define vec_setzero() _mm_setzero_si128()
#define vec_set_32 _mm_set1_epi32
#define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
#define vec_hadd SIMD::m128_hadd
#elif defined(USE_NEON_DOTPROD)
                        using vec_t = int32x4_t;
#define vec_setzero() vdupq_n_s32(0)
#define vec_set_32 vdupq_n_s32
#define vec_add_dpbusd_32(acc, a, b) \
                        SIMD::dotprod_m128_add_dpbusd_epi32(acc, vreinterpretq_s8_s32(a), \
                                                                                       vreinterpretq_s8_s32(b))
#define vec_hadd SIMD::neon_m128_hadd
#endif

                        const auto inputVector = reinterpret_cast<const vec_t*>(input);
            static constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(InputType);

            static_assert(PaddedInputDimensions % kInputSimdWidth == 0);

            constexpr IndexType kNumChunks = PaddedInputDimensions / kInputSimdWidth;
            vec_t               sum0      = vec_setzero();
            const auto          row0      = reinterpret_cast<const vec_t*>(&weights[0]);

            for (int j = 0; j < int(kNumChunks); ++j)
            {
                const vec_t in = inputVector[j];
                vec_add_dpbusd_32(sum0, in, row0[j]);
            }

            output[0] = vec_hadd(sum0, biases[0]);

#undef vec_setzero
#undef vec_set_32
#undef vec_add_dpbusd_32
#undef vec_hadd
                }

#else
        // Use dense implementation for the other architectures.
        affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights, biases, input);
#endif
        }

#if defined(USE_AVX512)
        void propagate_clipped_relu_packed_to_output(
            const u32* input32,
            const AffineTransform<OutputDimensions, 1>& next_layer,
            OutputType* output) const {
                if constexpr (OutputDimensions == 64) {
                        constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
                        constexpr IndexType kNumRegs = OutputDimensions / 16;

                        constexpr IndexType kNumAccums = kNumRegs;
#if defined(USE_NNUE_VNNI)
                        constexpr IndexType kActualNumRegs = 2 * kNumAccums;
#else
                        constexpr IndexType kActualNumRegs = kNumAccums;
#endif

                        const auto biasvec = reinterpret_cast<const __m512i*>(biases);
                        __m512i acc[kActualNumRegs];

                        for (IndexType k = 0; k < kNumAccums; ++k)
                                acc[k] = biasvec[k];
#if defined(USE_NNUE_VNNI)
                        for (IndexType k = kNumAccums; k < kActualNumRegs; ++k)
                                acc[k] = _mm512_setzero_si512();
#endif

#if defined(USE_NNUE_VNNI)
                        IndexType i = 0;
                        for (; i + 1 < kNumChunks; i += 2) {
                                const __m512i in0 = _mm512_set1_epi32(static_cast<int>(input32[i]));
                                const __m512i in1 = _mm512_set1_epi32(static_cast<int>(input32[i + 1]));
                                const auto col0 = reinterpret_cast<const __m512i*>(
                                        &weights[i * OutputDimensions * 4]);
                                const auto col1 = reinterpret_cast<const __m512i*>(
                                        &weights[(i + 1) * OutputDimensions * 4]);

                                for (IndexType k = 0; k < kNumAccums; ++k) {
                                        SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
                                        SIMD::m512_add_dpbusd_epi32(acc[k + kNumAccums], in1, col1[k]);
                                }
                        }

                        for (IndexType k = 0; k < kNumAccums; ++k)
                                acc[k] = _mm512_add_epi32(acc[k], acc[k + kNumAccums]);

                        for (; i < kNumChunks; ++i) {
#else
                        for (IndexType i = 0; i < kNumChunks; ++i) {
#endif
                                const __m512i in = _mm512_set1_epi32(static_cast<int>(input32[i]));
                                const auto col = reinterpret_cast<const __m512i*>(
                                        &weights[i * OutputDimensions * 4]);

                                for (IndexType k = 0; k < kNumAccums; ++k)
                                        SIMD::m512_add_dpbusd_epi32(acc[k], in, col[k]);
                        }

                        const __m512i kZero = _mm512_setzero_si512();
                        const __m512i words0 = _mm512_srai_epi16(
                                _mm512_packs_epi32(acc[0], acc[1]), WeightScaleBits);
                        const __m512i words1 = _mm512_srai_epi16(
                                _mm512_packs_epi32(acc[2], acc[3]), WeightScaleBits);
                        const __m512i clipped =
                                _mm512_max_epi8(_mm512_packs_epi16(words0, words1), kZero);

                        __m512i sum = _mm512_setzero_si512();
                        const auto row0 =
                                reinterpret_cast<const __m512i*>(&next_layer.weights_clipped_relu_packed_[0]);
                        SIMD::m512_add_dpbusd_epi32(sum, clipped, row0[0]);
                        const __m256i sum_lo = _mm512_castsi512_si256(sum);
                        const __m256i sum_hi = _mm512_extracti64x4_epi64(sum, 1);
                        output[0] = SIMD::m256_hadd(_mm256_add_epi32(sum_lo, sum_hi),
                                                     next_layer.biases[0]);
                } else {
                        static_assert(OutputDimensions == 64,
                                "propagate_clipped_relu_packed_to_output currently supports 64 hidden units.");
                }
        }

        // Fused path for SFNN's tiny tail: Affine(14->64) -> ClippedReLU(64)
        // -> Affine(64->1).  This avoids writing and re-reading the 64-byte
        // activation buffer while preserving ClippedReLU's byte order.
        void propagate_clipped_relu_to_output(
            const InputType* input,
            const AffineTransform<OutputDimensions, 1>& next_layer,
            OutputType* output) const {
                propagate_clipped_relu_packed_to_output(
                    reinterpret_cast<const u32*>(input), next_layer, output);
        }
#endif

   private:
        using BiasType   = OutputType;
        using WeightType = i8;

        alignas(CacheLineSize) BiasType biases[OutputDimensions];
        alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
        alignas(CacheLineSize) WeightType
                weights_clipped_relu_packed_[kClippedReLUPackedWeightDimensions];
};
#else
// 従来NNUEは前段レイヤーを所有するため、再帰型APIとバッファ配置を維持する。
// Affine transformation layer
// アフィン変換層
template <typename PreviousLayer, IndexType OutDims>
class AffineTransform {
   public:
	// Input/output type
	// 入出力の型
	using InputType  = typename PreviousLayer::OutputType;
	using OutputType = i32;
	static_assert(std::is_same<InputType, u8>::value, "");

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType InputDimensions       = PreviousLayer::OutputDimensions;
	static constexpr IndexType OutputDimensions      = OutDims;
	static constexpr IndexType PaddedInputDimensions = ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);

	// Size of forward propagation buffer used in this layer
	// この層で使用する順伝播用バッファのサイズ
	static constexpr usize SelfBufferSize =
	    ceil_to_multiple(OutputDimensions * sizeof(OutputType), CacheLineSize);

	// Size of the forward propagation buffer used from the input layer to this layer
	// 入力層からこの層までで使用する順伝播用バッファのサイズ
	static constexpr usize BufferSize = PreviousLayer::BufferSize + SelfBufferSize;

	// 評価関数ファイルに埋め込むハッシュ値
	// Hash value embedded in the evaluation file
	static constexpr u32 get_hash_value() {
		u32 hashValue = 0xCC03DAE4u;
		hashValue += OutputDimensions;
		hashValue ^= PreviousLayer::get_hash_value() >> 1;
		hashValue ^= PreviousLayer::get_hash_value() << 31;
		return hashValue;
	}

	// ハッシュ値を前段の値から更新するときのヘルパー
	static constexpr u32 get_hash_value(u32 prevHash) {
		u32 hashValue = 0xCC03DAE4u;
		hashValue += OutputDimensions;
		hashValue ^= prevHash >> 1;
		hashValue ^= prevHash << 31;
		return hashValue;
	}

	// 入力層からこの層までの構造を表す文字列
	static std::string GetStructureString() {
		return "AffineTransform[" + std::to_string(OutputDimensions) + "<-" + std::to_string(InputDimensions) + "](" +
		       PreviousLayer::GetStructureString() + ")";
	}

	static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
             + i / PaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
        return OutputDimensions % 4 == 0 ? get_weight_index_scrambled(i) : i;
#else
        return i;
#endif
    }

	// Read network parameters
	// パラメータを読み込む
	bool read_parameters(std::istream& stream) {
		if (!previous_layer_.read_parameters(stream)) return false;
		for (usize i = 0; i < OutputDimensions; ++i)
			biases[i] = read_little_endian<BiasType>(stream);
		for (usize i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
			weights[get_weight_index(IndexType(i))] = read_little_endian<WeightType>(stream);
		return !stream.fail();
	}

	// パラメータを書き込む
	bool write_parameters(std::ostream& stream) const {
		if (!previous_layer_.write_parameters(stream))
			return false;
		// 実行環境によらずlittle-endianで書き込む。
		write_little_endian(stream, biases, OutputDimensions);
		write_little_endian(stream, weights, OutputDimensions * PaddedInputDimensions);
		return !stream.fail();
	}

	// Forward propagation
	// 順伝播
	const OutputType* propagate(const TransformedFeatureType* transformed_features, char* buffer) const {
		const auto input = previous_layer_.propagate(transformed_features, buffer + SelfBufferSize);

#if defined(USE_WASM_SIMD)
		{
			// Simplify variable names (y = Ax + b)
			constexpr int n = InputDimensions;
			constexpr int m = OutputDimensions;
			constexpr int n_stride = PaddedInputDimensions;
			auto A = *reinterpret_cast<const i8(*)[m][n_stride]>(weights);
			auto x = *reinterpret_cast<const u8(*)[n]>(input);
			auto b = *reinterpret_cast<const i32(*)[m]>(biases);
			auto y = *reinterpret_cast<i32(*)[m]>(buffer);
			emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
			return y;
		}
#endif
		const auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)

		if constexpr (OutputDimensions > 1)
		{
#if defined(USE_AVX512)
			if constexpr (OutputDimensions % 16 == 0)
			{
				constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = OutputDimensions / 16;

				const auto   input32 = reinterpret_cast<const i32*>(input);
				const __m512i* biasvec = reinterpret_cast<const __m512i*>(biases);
				__m512i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m512i in = _mm512_set1_epi32(input32[i]);
					const auto  col  = reinterpret_cast<const __m512i*>(&weights[i * OutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						SIMD::m512_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m512i* outptr = reinterpret_cast<__m512i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_AVX2)
			if constexpr (OutputDimensions % 8 == 0)
			{
				constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = OutputDimensions / 8;

				const auto   input32 = reinterpret_cast<const i32*>(input);
				const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases);
				__m256i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m256i in = _mm256_set1_epi32(input32[i]);
					const auto  col  = reinterpret_cast<const __m256i*>(&weights[i * OutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						SIMD::m256_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m256i* outptr = reinterpret_cast<__m256i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_SSSE3)
			if constexpr (OutputDimensions % 4 == 0)
			{
				constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
				constexpr IndexType kNumRegs = OutputDimensions / 4;

				const auto   input32 = reinterpret_cast<const i32*>(input);
				const __m128i* biasvec = reinterpret_cast<const __m128i*>(biases);
				__m128i        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const __m128i in = _mm_set1_epi32(input32[i]);
					const auto  col = reinterpret_cast<const __m128i*>(&weights[i * OutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						SIMD::m128_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				__m128i* outptr = reinterpret_cast<__m128i*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

#if defined(USE_NEON_DOTPROD)
			if constexpr (OutputDimensions % (sizeof(int32x4_t) / sizeof(OutputType)) == 0)
			{
				constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
				constexpr IndexType kOutputSimdWidth = sizeof(int32x4_t) / sizeof(OutputType);
				constexpr IndexType kNumRegs = OutputDimensions / kOutputSimdWidth;

				const auto       input32 = reinterpret_cast<const i32*>(input);
				const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases);
				int32x4_t        acc[kNumRegs];

				for (IndexType k = 0; k < kNumRegs; ++k)
					acc[k] = biasvec[k];
				
				for (IndexType i = 0; i < kNumChunks; ++i)
				{
					const int32x4_t in = vdupq_n_s32(input32[i]);
					const auto  col = reinterpret_cast<const int32x4_t*>(&weights[i * OutputDimensions * 4]);

					for (IndexType k = 0; k < kNumRegs; ++k)
						SIMD::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
				}

				int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);
				
				for (IndexType k = 0; k < kNumRegs; ++k)
					outptr[k] = acc[k];
			}
			else
#endif

				affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
				output, weights, biases, input);
		}
		else if constexpr (OutputDimensions == 1)
		{
    // We cannot use AVX512 for the last layer because there are only 32 inputs
    // and the buffer is not padded to 64 elements.
#if defined(USE_AVX2)
			using vec_t = __m256i;
#define vec_setzero() _mm256_setzero_si256()
#define vec_set_32 _mm256_set1_epi32
#define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
#define vec_hadd SIMD::m256_hadd
#elif defined(USE_SSSE3)
			using vec_t = __m128i;
#define vec_setzero() _mm_setzero_si128()
#define vec_set_32 _mm_set1_epi32
#define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
#define vec_hadd SIMD::m128_hadd
#elif defined(USE_NEON_DOTPROD)
			using vec_t = int32x4_t;
#define vec_setzero() vdupq_n_s32(0)
#define vec_set_32 vdupq_n_s32
#define vec_add_dpbusd_32(acc, a, b) \
			SIMD::dotprod_m128_add_dpbusd_epi32(acc, vreinterpretq_s8_s32(a), \
												vreinterpretq_s8_s32(b))
#define vec_hadd SIMD::neon_m128_hadd
#endif

			const auto inputVector = reinterpret_cast<const vec_t*>(input);
            static constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(InputType);

            static_assert(PaddedInputDimensions % kInputSimdWidth == 0);

            constexpr IndexType kNumChunks = PaddedInputDimensions / kInputSimdWidth;
            vec_t               sum0      = vec_setzero();
            const auto          row0      = reinterpret_cast<const vec_t*>(&weights[0]);

            for (int j = 0; j < int(kNumChunks); ++j)
            {
                const vec_t in = inputVector[j];
                vec_add_dpbusd_32(sum0, in, row0[j]);
            }

            output[0] = vec_hadd(sum0, biases[0]);

#undef vec_setzero
#undef vec_set_32
#undef vec_add_dpbusd_32
#undef vec_hadd
		}

#else
        // Use dense implementation for the other architectures.
        affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights, biases, input);
#endif

		return output;
	}

   private:
	// パラメータの型
	using BiasType   = OutputType;
	using WeightType = i8;

	// この層の直前の層
	PreviousLayer previous_layer_;

	// パラメータ
	alignas(CacheLineSize) BiasType biases[OutputDimensions];
	alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};
#endif

} // namespace YaneuraOu::Eval::NNUE::Layers

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
