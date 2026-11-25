// Definition of the AffineTransform layer with block-sparse input in the NNUE evaluation function
// NNUE評価関数におけるブロック疎な入力を持つAffineTransform層の定義

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <iosfwd>

#include "../../config.h"
#include "../../extra/tools.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"
#include "affine_transform.h"
#include "../simd.h"

namespace YaneuraOu::Eval::NNUE::Layers {

#if defined(EVAL_SFNN)
// Stockfishと同じ明示次元型のAPIを使う。非ゼロ抽出とFTからの直接計算は既存の最適化を維持する。
#if defined(USE_SSSE3) || USE_NEON >= 8

alignas(CacheLineSize) static inline const
  std::array<std::array<u16, 8>, 256> lookup_indices = []() {
      std::array<std::array<u16, 8>, 256> v{};
      for (unsigned i = 0; i < 256; ++i)
      {
          std::uint64_t j = i, k = 0;
          while (j)
              v[i][k++] = pop_lsb(j);
      }
      return v;
  }();

// Find indices of nonzero numbers in an i32 array
// i32配列の非ゼロ要素のインデックスを列挙する。
template<const IndexType InputDimensions>
static void find_nnz(const i32* input, u16* out, IndexType& count_out) {
#if defined(USE_SSSE3)
#if defined(USE_AVX512)
	using vec_t = __m512i;
	#define vec_nnz(a) _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
#elif defined(USE_AVX2)
    using vec_t = __m256i;
#if defined(USE_NNUE_VNNI) && !defined(USE_AVXVNNI)
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
        static constexpr u32 Mask[4] = {1, 2, 4, 8};
#define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    using vec128_t                     = uint16x8_t;
#define vec128_zero vdupq_n_u16(0)
#define vec128_set_16(a) vdupq_n_u16(a)
#define vec128_load(a) vld1q_u16(reinterpret_cast<const u16*>(a))
#define vec128_storeu(a, b) vst1q_u16(reinterpret_cast<u16*>(a), b)
#define vec128_add(a, b) vaddq_u16(a, b)
#endif

    constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(i32);
    // Inputs are processed kInputSimdWidth at a time and outputs are processed 8 at a time so we process in chunks of max(kInputSimdWidth, 8)
    constexpr IndexType ChunkSize       = std::max<IndexType>(kInputSimdWidth, 8);
    constexpr IndexType kNumChunks       = InputDimensions / ChunkSize;
    constexpr IndexType kInputsPerChunk  = ChunkSize / kInputSimdWidth;
    constexpr IndexType kOutputsPerChunk = ChunkSize / 8;

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
              vec128_load(reinterpret_cast<const vec128_t*>(&lookup_indices[lookup]));
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
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
public:
        using InputType = u8;
        using OutputType = i32;

        // Number of input/output dimensions
        // 入出力の次元数
        static constexpr IndexType InputDimensions       = InDims;
        static constexpr IndexType OutputDimensions      = OutDims;
        // Stockfishの対応する層には OutputDimensions % 16 == 0 の制約がある。
        // やねうら王はH1=7のSFNNを現役で使っており、ショートカット込みのfc0は8出力になる。
        // 同じ制約を導入すると既存のH1=7構成をビルドできなくなるため、8出力の対応を維持する。
        static constexpr IndexType PaddedInputDimensions = ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
        static constexpr IndexType PaddedOutputDimensions = ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

#if defined(USE_SSSE3) || USE_NEON >= 8
    static constexpr IndexType ChunkSize = 4;
#else
    static constexpr IndexType ChunkSize = 1;
#endif

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
                return "AffineTransformSparseInput[" + std::to_string(OutputDimensions) + "<-" + std::to_string(InputDimensions) + "]";
        }

        static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if defined(USE_SSSE3) || USE_NEON >= 8
        return OutputDimensions % 4 == 0 ? get_weight_index_scrambled(i) : i;
#else
        return i;
#endif
    }

        // Read network parameters
        // パラメーターを読み込む。既存の重みの並び替えを維持する。
        bool read_parameters(std::istream& stream) {
                for (usize i = 0; i < OutputDimensions; ++i)
                        biases_[i] = read_little_endian<BiasType>(stream);
                for (usize i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
                        weights_[get_weight_index(IndexType(i))] = read_little_endian<WeightType>(stream);
                return !stream.fail();
        }

        // Write network parameters
        // パラメーターを書き込む。既存の内部配置での書き出しを維持する。
        bool write_parameters(std::ostream& stream) const {
                stream.write(reinterpret_cast<const char*>(biases_), OutputDimensions * sizeof(BiasType));
                stream.write(reinterpret_cast<const char*>(weights_),
                             OutputDimensions * PaddedInputDimensions * sizeof(WeightType));
                return !stream.fail();
        }

#if defined(USE_AVX512) && defined(EVAL_SFNN)
        // Fused EWM + sparse affine for 16 outputs (H1=15 with shortcut or H1=16).
        // Keep the existing on-disk and in-memory weight ordering.
        // ファイル上・メモリ上の重み配置を保ち、EWMと疎なアフィン変換を融合して計算する。
        template <IndexType HalfDimensions, typename AccumulationType>
        void PropagateSfnn16FromAccumulator(const AccumulationType& accumulation,
                                            Color sideToMove, OutputType* output) const {
                static_assert(InputDimensions == HalfDimensions);
                static_assert(OutputDimensions == 16);
                static_assert((HalfDimensions / 2) % 64 == 0);
                constexpr IndexType chunks = (HalfDimensions / 2) / 64;
                constexpr int shift =
#if defined(USE_SSE2)
                    7;
#else
                    6;
#endif
                const __m512i zero = _mm512_setzero_si512();
                const __m512i one = _mm512_set1_epi16(254);
                __m512i sum0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(biases_));
                __m512i sum1 = zero, sum2 = zero;
                const Color perspectives[2] = {sideToMove, ~sideToMove};
                alignas(CacheLineSize) u32 input32[16];
                for (IndexType p = 0; p < 2; ++p) {
                        const auto* in0 = reinterpret_cast<const __m512i*>(&accumulation[perspectives[p]][0][0]);
                        const auto* in1 = reinterpret_cast<const __m512i*>(&accumulation[perspectives[p]][0][HalfDimensions / 2]);
                        for (IndexType chunk = 0; chunk < chunks; ++chunk) {
                                const __m512i a = _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(in0[chunk * 2], one), zero), shift);
                                const __m512i b = _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(in0[chunk * 2 + 1], one), zero), shift);
                                const __m512i c = _mm512_min_epi16(in1[chunk * 2], one);
                                const __m512i d = _mm512_min_epi16(in1[chunk * 2 + 1], one);
                                const __m512i transformed = _mm512_packus_epi16(_mm512_mulhi_epi16(a, c), _mm512_mulhi_epi16(b, d));
                                unsigned bits = _mm512_cmpneq_epi32_mask(transformed, zero);
                                if (!bits) continue;
                                _mm512_store_si512(reinterpret_cast<__m512i*>(input32), transformed);
                                const IndexType base = (p * chunks + chunk) * 16;
                                auto add = [&](__m512i& sum) {
                                        const IndexType bit = pop_lsb(bits);
                                        const __m512i input = _mm512_set1_epi32(static_cast<int>(input32[bit]));
                                        const auto* column = reinterpret_cast<const __m512i*>(
                                            &weights_[(base + bit) * OutputDimensions * ChunkSize]);
                                        SIMD::m512_add_dpbusd_epi32(sum, input, column[0]);
                                };
                                while (bits) {
                                        // Three independent VNNI chains hide dot-product latency.
                                        add(sum0);
                                        if (!bits) break;
                                        add(sum1);
                                        if (!bits) break;
                                        add(sum2);
                                }
                        }
                }
                const __m512i result = _mm512_add_epi32(_mm512_add_epi32(sum0, sum1), sum2);
                _mm512_store_si512(reinterpret_cast<__m512i*>(output), result);
                for (IndexType out = OutputDimensions; out < PaddedOutputDimensions; ++out)
                        output[out] = OutputType{};
        }

        // SFNNのEWM Transform()で一度 u8 配列を作らず、accumulatorからfc_0へ直接流す。
        // 現状は kHidden1Dims + 1 == 8 のSFNNで使う。
        template <IndexType HalfDimensions, typename AccumulationType>
        void PropagateSfnnFromAccumulator(const AccumulationType& accumulation,
                                          Color sideToMove,
                                          OutputType* output) const {
                static_assert(InputDimensions == HalfDimensions);
                static_assert(OutputDimensions == 8);
                static_assert((HalfDimensions / 2) % 64 == 0);

                constexpr IndexType kChunksPerPerspective = (HalfDimensions / 2) / 64;
                constexpr IndexType kInput32PerVector = 16;
                constexpr int shift =
#if defined(USE_SSE2)
                    7;
#else
                    6;
#endif

                const __m512i zero = _mm512_setzero_si512();
                const __m512i one = _mm512_set1_epi16(127 * 2);
                const Color perspectives[2] = { sideToMove, ~sideToMove };

                const auto biasvec = reinterpret_cast<const __m256i*>(biases_);
#if defined(USE_NNUE_VNNI)
                // Stockfishと同じ方針で、VNNIの高レイテンシなdot productを
                // 複数の依存チェーンに分け、最後にmergeする。
                __m256i out_acc0 = _mm256_load_si256(biasvec);
                __m256i out_acc1 = _mm256_setzero_si256();
                __m256i out_acc2 = _mm256_setzero_si256();
#else
                __m256i acc = _mm256_load_si256(biasvec);
#endif
                alignas(CacheLineSize) u32 input32[kInput32PerVector];

                for (IndexType p = 0; p < 2; ++p) {
                        const auto perspective = perspectives[p];
                        const auto acc0 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][0]);
                        const auto acc1 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][HalfDimensions / 2]);

                        for (IndexType chunk = 0; chunk < kChunksPerPerspective; ++chunk) {
                                const __m512i sum0a =
                                    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 0], one), zero), shift);
                                const __m512i sum0b =
                                    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 1], one), zero), shift);
                                const __m512i sum1a = _mm512_min_epi16(acc1[chunk * 2 + 0], one);
                                const __m512i sum1b = _mm512_min_epi16(acc1[chunk * 2 + 1], one);
                                const __m512i pa = _mm512_mulhi_epi16(sum0a, sum1a);
                                const __m512i pb = _mm512_mulhi_epi16(sum0b, sum1b);
                                const __m512i transformed = _mm512_packus_epi16(pa, pb);

                                unsigned nnz = _mm512_cmpneq_epi32_mask(transformed, zero);
                                if (!nnz)
                                        continue;

                                _mm512_store_si512(reinterpret_cast<__m512i*>(input32), transformed);
                                const IndexType base = (p * kChunksPerPerspective + chunk) * kInput32PerVector;

#if defined(USE_NNUE_VNNI)
                                unsigned bits = nnz;
                                while (bits) {
                                        const IndexType bit0 = pop_lsb(bits);
                                        const IndexType i0 = base + bit0;
                                        const __m256i in0 = _mm256_set1_epi32(static_cast<int>(input32[bit0]));
                                        const auto col0 =
                                            reinterpret_cast<const __m256i*>(&weights_[i0 * OutputDimensions * ChunkSize]);
                                        SIMD::m256_add_dpbusd_epi32(out_acc0, in0, col0[0]);

                                        if (!bits)
                                                break;
                                        const IndexType bit1 = pop_lsb(bits);
                                        const IndexType i1 = base + bit1;
                                        const __m256i in1 = _mm256_set1_epi32(static_cast<int>(input32[bit1]));
                                        const auto col1 =
                                            reinterpret_cast<const __m256i*>(&weights_[i1 * OutputDimensions * ChunkSize]);
                                        SIMD::m256_add_dpbusd_epi32(out_acc1, in1, col1[0]);

                                        if (!bits)
                                                break;
                                        const IndexType bit2 = pop_lsb(bits);
                                        const IndexType i2 = base + bit2;
                                        const __m256i in2 = _mm256_set1_epi32(static_cast<int>(input32[bit2]));
                                        const auto col2 =
                                            reinterpret_cast<const __m256i*>(&weights_[i2 * OutputDimensions * ChunkSize]);
                                        SIMD::m256_add_dpbusd_epi32(out_acc2, in2, col2[0]);
                                }
#else
                                for (IndexType half = 0; half < 2; ++half) {
                                        const unsigned lookup = (nnz >> (half * 8)) & 0xff;
                                        const auto& offsets = lookup_indices[lookup];
                                        const IndexType offset_base = half * 8;
                                        const IndexType count = POPCNT32(lookup);

                                        for (IndexType j = 0; j < count; ++j) {
                                                const IndexType bit = offset_base + offsets[j];
                                                const IndexType i = base + bit;
                                                const __m256i in = _mm256_set1_epi32(static_cast<int>(input32[bit]));
                                                const auto col =
                                                    reinterpret_cast<const __m256i*>(&weights_[i * OutputDimensions * ChunkSize]);
                                                SIMD::m256_add_dpbusd_epi32(acc, in, col[0]);
                                        }
                                }
#endif
                        }
                }

#if defined(USE_NNUE_VNNI)
                const __m256i acc = _mm256_add_epi32(_mm256_add_epi32(out_acc0, out_acc1), out_acc2);
#endif
                _mm256_store_si256(reinterpret_cast<__m256i*>(output), acc);
                for (IndexType out = OutputDimensions; out < PaddedOutputDimensions; ++out)
                        output[out] = OutputType{};
        }
#endif

        // Forward propagation
        // 順伝播
        void propagate(const InputType* input, OutputType* output) const {
#if defined(USE_WASM_SIMD)
                {
                        constexpr int n = InputDimensions;
                        constexpr int m = OutputDimensions;
                        constexpr int n_stride = PaddedInputDimensions;
                        auto A = *reinterpret_cast<const i8(*)[m][n_stride]>(weights_);
                        auto x = *reinterpret_cast<const u8(*)[n]>(input);
                        auto b = *reinterpret_cast<const i32(*)[m]>(biases_);
                        auto y = *reinterpret_cast<i32(*)[m]>(output);
                        emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
                        return; // void return
                }
#endif

#if defined(USE_SSSE3) || USE_NEON >= 8

#if defined(USE_AVX512)
        if constexpr (OutputDimensions % 16 == 0)
        {
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
            constexpr IndexType kNumRegs   = OutputDimensions / 16;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            find_nnz<kNumChunks>(input32, nnz, count);

            constexpr IndexType kNumAccums = kNumRegs;
#if defined(USE_NNUE_VNNI)
            constexpr IndexType kActualNumRegs = 3 * kNumAccums;
#else
            constexpr IndexType kActualNumRegs = kNumAccums;
#endif

            const __m512i* biasvec = reinterpret_cast<const __m512i*>(biases_);
            __m512i        acc[kActualNumRegs];

            for (IndexType k = 0; k < kNumAccums; ++k)
                acc[k] = biasvec[k];
#if defined(USE_NNUE_VNNI)
            for (IndexType k = kNumAccums; k < kActualNumRegs; ++k)
                acc[k] = _mm512_setzero_si512();
#endif

#if defined(USE_NNUE_VNNI)
            IndexType j = 0;
            for (; j + 2 < count; j += 3)
            {
                const auto i0 = nnz[j + 0];
                const auto i1 = nnz[j + 1];
                const auto i2 = nnz[j + 2];
                const __m512i in0 = _mm512_set1_epi32(input32[i0]);
                const __m512i in1 = _mm512_set1_epi32(input32[i1]);
                const __m512i in2 = _mm512_set1_epi32(input32[i2]);
                const auto col0 =
                    reinterpret_cast<const __m512i*>(&weights_[i0 * OutputDimensions * ChunkSize]);
                const auto col1 =
                    reinterpret_cast<const __m512i*>(&weights_[i1 * OutputDimensions * ChunkSize]);
                const auto col2 =
                    reinterpret_cast<const __m512i*>(&weights_[i2 * OutputDimensions * ChunkSize]);

                for (IndexType k = 0; k < kNumAccums; ++k) {
                    SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
                    SIMD::m512_add_dpbusd_epi32(acc[k + kNumAccums], in1, col1[k]);
                    SIMD::m512_add_dpbusd_epi32(acc[k + 2 * kNumAccums], in2, col2[k]);
                }
            }

            for (IndexType k = 0; k < kNumAccums; ++k)
                acc[k] = _mm512_add_epi32(_mm512_add_epi32(acc[k], acc[k + kNumAccums]), acc[k + 2 * kNumAccums]);

            for (; j < count; ++j)
#else
            for (IndexType j = 0; j < count; ++j)
#endif
            {
                const auto    i  = nnz[j];
                const __m512i in = _mm512_set1_epi32(input32[i]);
                const auto    col =
                    reinterpret_cast<const __m512i*>(&weights_[i * OutputDimensions * ChunkSize]);
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
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
            constexpr IndexType kNumRegs   = OutputDimensions / 8;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            find_nnz<kNumChunks>(input32, nnz, count);

            constexpr IndexType kNumAccums = kNumRegs;
#if defined(USE_AVXVNNI)
            constexpr IndexType kActualNumRegs = 2 * kNumAccums;
#else
            constexpr IndexType kActualNumRegs = kNumAccums;
#endif

            const __m256i* biasvec = reinterpret_cast<const __m256i*>(biases_);
            __m256i        acc[kActualNumRegs];

            for (IndexType k = 0; k < kNumAccums; ++k)
                acc[k] = biasvec[k];
#if defined(USE_AVXVNNI)
            for (IndexType k = kNumAccums; k < kActualNumRegs; ++k)
                acc[k] = _mm256_setzero_si256();
#endif

#if defined(USE_AVXVNNI)
            IndexType j = 0;
            for (; j + 1 < count; j += 2)
            {
                const auto i0 = nnz[j + 0];
                const auto i1 = nnz[j + 1];
                const __m256i in0 = _mm256_set1_epi32(input32[i0]);
                const __m256i in1 = _mm256_set1_epi32(input32[i1]);
                const auto col0 =
                    reinterpret_cast<const __m256i*>(&weights_[i0 * OutputDimensions * ChunkSize]);
                const auto col1 =
                    reinterpret_cast<const __m256i*>(&weights_[i1 * OutputDimensions * ChunkSize]);

                for (IndexType k = 0; k < kNumAccums; ++k) {
                    SIMD::m256_add_dpbusd_epi32(acc[k], in0, col0[k]);
                    SIMD::m256_add_dpbusd_epi32(acc[k + kNumAccums], in1, col1[k]);
                }
            }

            for (IndexType k = 0; k < kNumAccums; ++k)
                acc[k] = _mm256_add_epi32(acc[k], acc[k + kNumAccums]);

            for (; j < count; ++j)
#else
            for (IndexType j = 0; j < count; ++j)
#endif
            {
                const auto    i  = nnz[j];
                const __m256i in = _mm256_set1_epi32(input32[i]);
                const auto    col =
                    reinterpret_cast<const __m256i*>(&weights_[i * OutputDimensions * ChunkSize]);
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
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
            constexpr IndexType kNumRegs   = OutputDimensions / 4;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            find_nnz<kNumChunks>(input32, nnz, count);

            const __m128i* biasvec = reinterpret_cast<const __m128i*>(biases_);
            __m128i        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto    i  = nnz[j];
                const __m128i in = _mm_set1_epi32(input32[i]);
                const auto    col =
                reinterpret_cast<const __m128i*>(&weights_[i * OutputDimensions * ChunkSize]);
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
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
            constexpr IndexType kOutputSimdWidth = sizeof(int32x4_t) / sizeof(OutputType);
            constexpr IndexType kNumRegs   = OutputDimensions / kOutputSimdWidth;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            find_nnz<kNumChunks>(input32, nnz, count);

            const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases_);
            int32x4_t        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto      i  = nnz[j];
                const int8x16_t in = vreinterpretq_s8_u32(vdupq_n_u32(input32[i]));
                const auto     col =
                reinterpret_cast<const int8x16_t*>(&weights_[i * OutputDimensions * ChunkSize]);
                for (IndexType k = 0; k < kNumRegs; ++k)
                    SIMD::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
            }

            int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);

            for (IndexType k = 0; k < kNumRegs; ++k)
                outptr[k] = acc[k];
        }
        else
#endif

#if defined(USE_NEON) && !defined(USE_NEON_DOTPROD)
        if constexpr (OutputDimensions % (sizeof(int32x4_t) / sizeof(OutputType)) == 0)
        {
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
            constexpr IndexType kOutputSimdWidth = sizeof(int32x4_t) / sizeof(OutputType);
            constexpr IndexType kNumRegs   = OutputDimensions / kOutputSimdWidth;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            find_nnz<kNumChunks>(input32, nnz, count);

            const int32x4_t* biasvec = reinterpret_cast<const int32x4_t*>(biases_);
            int32x4_t        acc[kNumRegs];

            for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasvec[k];

            for (IndexType j = 0; j < count; ++j)
            {
                const auto      i  = nnz[j];
                const int8x16_t in = vreinterpretq_s8_u32(vdupq_n_u32(input32[i]));
                const auto     col =
                reinterpret_cast<const int8x16_t*>(&weights_[i * OutputDimensions * ChunkSize]);
                for (IndexType k = 0; k < kNumRegs; ++k)
                    SIMD::neon_m128_add_dpbusd_epi32(acc[k], in, col[k]);
            }

            int32x4_t* outptr = reinterpret_cast<int32x4_t*>(output);

            for (IndexType k = 0; k < kNumRegs; ++k)
                outptr[k] = acc[k];
        }
        else
#endif
            affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
              output, weights_, biases_, input);

#undef vec_set_32
#undef vec_add_dpbusd_32

#else
        // Use dense implementation for the other architectures.
        affine_transform_unaligned<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights_, biases_, input);
#endif
        }

   private:
        using BiasType   = OutputType;
        using WeightType = i8;

        alignas(CacheLineSize) BiasType biases_[OutputDimensions];
        alignas(CacheLineSize) WeightType weights_[OutputDimensions * PaddedInputDimensions];
};

// 生成済みアーキテクチャヘッダーで使われている旧型名だけを受け付ける。
// 生成器は新名称を出力する。実体は同じで、別の層や重み配列を持たない。
template<IndexType InDims, IndexType OutDims>
using AffineTransformSparseInputExplicit = AffineTransformSparseInput<InDims, OutDims>;
#else
// 従来NNUEでは前段レイヤーを所有するため、現行Stockfishの明示次元型とはAPIが異なる。
// SFNNの明示次元型とは分岐を分け、従来の再帰型APIとバッファ配置を維持する。

#if defined(USE_SSSE3) || USE_NEON >= 8

alignas(CacheLineSize) static inline const
  std::array<std::array<u16, 8>, 256> lookup_indices = []() {
      std::array<std::array<u16, 8>, 256> v{};
      for (unsigned i = 0; i < 256; ++i)
      {
          std::uint64_t j = i, k = 0;
          while (j)
              v[i][k++] = pop_lsb(j);
      }
      return v;
  }();

// Find indices of nonzero numbers in an i32 array
template<const IndexType kInputDimensions>
void find_nnz(const i32* input, u16* out, IndexType& count_out) {
#if defined(USE_SSSE3)
#if defined(USE_AVX512)
    using vec_t = __m512i;
#define vec_nnz(a) _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
#elif defined(USE_AVX2)
    using vec_t = __m256i;
#if defined(USE_NNUE_VNNI) && !defined(USE_AVXVNNI)
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
	static constexpr u32 Mask[4] = {1, 2, 4, 8};
#define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    using vec128_t                     = uint16x8_t;
#define vec128_zero vdupq_n_u16(0)
#define vec128_set_16(a) vdupq_n_u16(a)
#define vec128_load(a) vld1q_u16(reinterpret_cast<const u16*>(a))
#define vec128_storeu(a, b) vst1q_u16(reinterpret_cast<u16*>(a), b)
#define vec128_add(a, b) vaddq_u16(a, b)
#endif

    constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(i32);
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
              vec128_load(reinterpret_cast<const vec128_t*>(&lookup_indices[lookup]));
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

// AffineTransform layer that takes block-sparse input
// ブロック疎な入力を受け取るアフィン変換層
template <typename PreviousLayer, IndexType OutputDimensions>
class AffineTransformSparseInput {
   public:
	// Input/output type
	// 入出力の型
	using InputType  = typename PreviousLayer::OutputType;
	using OutputType = i32;
	static_assert(std::is_same<InputType, u8>::value, "");

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions       = PreviousLayer::kOutputDimensions;
	static constexpr IndexType kOutputDimensions      = OutputDimensions;
	static constexpr IndexType kPaddedInputDimensions = ceil_to_multiple<IndexType>(kInputDimensions, MaxSimdWidth);

	// Size of forward propagation buffer used in this layer
	// この層で使用する順伝播用バッファのサイズ
	static constexpr usize kSelfBufferSize =
	    ceil_to_multiple(kOutputDimensions * sizeof(OutputType), CacheLineSize);

	// Size of the forward propagation buffer used from the input layer to this layer
	// 入力層からこの層までで使用する順伝播用バッファのサイズ
	static constexpr usize kBufferSize = PreviousLayer::kBufferSize + kSelfBufferSize;

#if defined(USE_SSSE3) || USE_NEON >= 8
    static constexpr IndexType kChunkSize = 4;
#else
    static constexpr IndexType kChunkSize = 1;
#endif

	// 評価関数ファイルに埋め込むハッシュ値
	// Hash value embedded in the evaluation file
	static constexpr u32 GetHashValue() {
		u32 hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= PreviousLayer::GetHashValue() >> 1;
		hash_value ^= PreviousLayer::GetHashValue() << 31;
		return hash_value;
	}

	// ハッシュ値を前段の値から更新するときのヘルパー
	static constexpr u32 GetHashValue(u32 prevHash) {
		u32 hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= prevHash >> 1;
		hash_value ^= prevHash << 31;
		return hash_value;
	}

	// 入力層からこの層までの構造を表す文字列
	static std::string GetStructureString() {
		return "AffineTransformSparseInput[" + std::to_string(kOutputDimensions) + "<-" + std::to_string(kInputDimensions) + "](" +
		       PreviousLayer::GetStructureString() + ")";
	}

	static constexpr IndexType GetWeightIndexScrambled(IndexType i) {
        return (i / kChunkSize) % (kPaddedInputDimensions / kChunkSize) * kOutputDimensions * kChunkSize
             + i / kPaddedInputDimensions * kChunkSize + i % kChunkSize;
    }

    static constexpr IndexType GetWeightIndex(IndexType i) {
#if defined(USE_SSSE3) || USE_NEON >= 8
        return kOutputDimensions % 4 == 0 ? GetWeightIndexScrambled(i) : i;
#else
        return i;
#endif
    }

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
		Tools::Result result = previous_layer_.ReadParameters(stream);
		if (result.is_not_ok()) return result;
		for (usize i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);
		for (usize i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
			weights_[GetWeightIndex(IndexType(i))] = read_little_endian<WeightType>(stream);
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
		const auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_WASM_SIMD)
		{
			// Simplify variable names (y = Ax + b)
			constexpr int n = kInputDimensions;
			constexpr int m = kOutputDimensions;
			constexpr int n_stride = kPaddedInputDimensions;
			auto A = *reinterpret_cast<const i8(*)[m][n_stride]>(weights_);
			auto x = *reinterpret_cast<const u8(*)[n]>(input);
			auto b = *reinterpret_cast<const i32(*)[m]>(biases_);
			auto y = *reinterpret_cast<i32(*)[m]>(buffer);
			emscripten_wasm_simd::affine<n, m, n_stride>(A, x, b, y);
			return y;
		}
#endif

#if defined(USE_SSSE3) || USE_NEON >= 8

#if defined(USE_AVX512)
        if constexpr (kOutputDimensions % 16 == 0)
        {
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 16;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            // Find indices of nonzero 32-bit blocks
            find_nnz<kNumChunks>(input32, nnz, count);

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
                    SIMD::m512_add_dpbusd_epi32(acc[k], in, col[k]);
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
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 8;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            // Find indices of nonzero 32-bit blocks
            find_nnz<kNumChunks>(input32, nnz, count);

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
                    SIMD::m256_add_dpbusd_epi32(acc[k], in, col[k]);
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
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kNumRegs   = kOutputDimensions / 4;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            // Find indices of nonzero 32-bit blocks
            find_nnz<kNumChunks>(input32, nnz, count);

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
                    SIMD::m128_add_dpbusd_epi32(acc[k], in, col[k]);
            }

            __m128i* outptr = reinterpret_cast<__m128i*>(output);

            for (IndexType k = 0; k < kNumRegs; ++k)
                outptr[k] = acc[k];
        }
        else
#endif

#if defined(USE_NEON_DOTPROD)
        if constexpr (kOutputDimensions % (sizeof(int32x4_t) / sizeof(OutputType)) == 0)
        {
            constexpr IndexType kNumChunks = ceil_to_multiple<IndexType>(kInputDimensions, 8) / kChunkSize;
            constexpr IndexType kOutputSimdWidth = sizeof(int32x4_t) / sizeof(OutputType);
            constexpr IndexType kNumRegs   = kOutputDimensions / kOutputSimdWidth;
            u16       nnz[kNumChunks];
            IndexType           count;

            const auto input32 = reinterpret_cast<const i32*>(input);

            // Find indices of nonzero 32-bit blocks
            find_nnz<kNumChunks>(input32, nnz, count);

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
                    SIMD::dotprod_m128_add_dpbusd_epi32(acc[k], in, col[k]);
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

		return output;
	}

   private:
	// パラメータの型
	using BiasType   = OutputType;
	using WeightType = i8;

	// この層の直前の層
	PreviousLayer previous_layer_;

	// パラメータ
	alignas(CacheLineSize) BiasType biases_[kOutputDimensions];
	alignas(CacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
};
#endif

} // namespace YaneuraOu::Eval::NNUE::Layers

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif  // ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
