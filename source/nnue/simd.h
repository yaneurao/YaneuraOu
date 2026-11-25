#ifndef NNUE_SIMD_H_INCLUDED
#define NNUE_SIMD_H_INCLUDED

#include "../config.h"

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>

#elif defined(USE_LASX)
    #include <lasxintrin.h>
    #include <lsxintrin.h>

#elif defined(USE_LSX)
    #include <lsxintrin.h>

#elif defined(USE_RVV)
    #include <riscv_vector.h>

#endif

#include "../types.h"
#include "nnue_common.h"

// CPU capability switch for NNUE dot-product code. For SFNN H1=7 networks
// (fc0: ->8, packed tail: 14->64->1), measured AVX512VNNI was slower than
// AVX512 maddubs/madd, so only that NNUE shape uses the fallback.
// やねうら王はH1=7のSFNNを現役で使っており、ショートカット込みのfc0は8出力になる。
// Stockfishの対応する疎なAffine層は16出力の倍数を要求し、この8出力構成を対象にしていない。
// H1=7では当時の測定でVNNI版よりmaddubs/madd版が速かったため、その形状だけ既存の経路を使う。
// Stockfishと層幅が異なるので、VNNIの選択条件を無条件に同じにはしない。
#if defined(USE_VNNI) && !defined(NNUE_SFNN_HIDDEN1_7)
    #define USE_NNUE_VNNI
#endif

// Stockfishと同じ配置・名前空間でNNUEのSIMDヘルパーを提供する。
// FT用のベクトル型・マクロもここに集約し、評価実装から共有する。
// LSX/LASX/RVVのincludeは命令の宣言を読むためのもので、計算経路の移植とは別。
namespace YaneuraOu::Eval::NNUE::SIMD {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// CPUのベクトルレジスタに収まるタイル単位で累積値を更新・再構築する。
#define VECTOR

#if defined(USE_AVX512)
using vec_t = __m512i;
#define vec_nnz(a) _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512())
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
#define NumRegistersSIMD NNUE_FT_REGS
#define MaxChunkSize 64
#else
// Stockfishの16とは異なり、既存のAVX512向け24レジスタ最適化を維持する。
#define NumRegistersSIMD 24
#define MaxChunkSize 64
#endif

#elif defined(USE_AVX2)
// Stockfishの12とは異なり、既存の16レジスタ設定を維持する。
using vec_t = __m256i;
#if defined(USE_NNUE_VNNI) && !defined(USE_AVXVNNI)
#define vec_nnz(a) _mm256_cmpgt_epi32_mask(a, _mm256_setzero_si256())
#else
#define vec_nnz(a) _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())))
#endif
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
#define NumRegistersSIMD 16
#define MaxChunkSize 32

#elif defined(USE_SSE2)
// Stockfishの12/6とは異なり、既存の64bit/32bit向け16/8レジスタ設定を維持する。
using vec_t = __m128i;
#if defined(USE_SSSE3)
#define vec_nnz(a) _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128())))
#endif
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
#define NumRegistersSIMD (Is64Bit ? 16 : 8)
#define MaxChunkSize 16

#elif defined(USE_MMX)
using vec_t = __m64;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero() _mm_setzero_si64()
#define NumRegistersSIMD 8
#define MaxChunkSize 8

#elif defined(USE_NEON)
using vec_t = int16x8_t;
#if USE_NEON >= 8
// Nonzero masks use the same 32-bit chunks as the x86 sparse input path.
// x86の疎な入力経路と同じ32bitチャンク単位で非ゼロ判定する。
inline unsigned neon_vec_nnz(vec_t a) {
    static constexpr u32 Mask[4] = {1, 2, 4, 8};
    const auto chunks = vreinterpretq_u32_s16(a);
    return vaddvq_u32(vandq_u32(vtstq_u32(chunks, chunks), vld1q_u32(Mask)));
}
#define vec_nnz(a) SIMD::neon_vec_nnz(a)
#endif
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
#define NumRegistersSIMD 16
#define MaxChunkSize 16

#else
#undef VECTOR

#endif

#ifdef VECTOR
static_assert(MaxChunkSize == sizeof(vec_t), "MaxChunkSize must equal one vector.");
#endif



#if defined(USE_AVX512)

[[maybe_unused]] static int m512_hadd(__m512i sum, int bias) {
    return _mm512_reduce_add_epi32(sum) + bias;
}

[[maybe_unused]] static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {

    #if defined(USE_NNUE_VNNI)
    acc = _mm512_dpbusd_epi32(acc, a, b);
    #else
    __m512i product0 = _mm512_maddubs_epi16(a, b);
    product0         = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
    acc              = _mm512_add_epi32(acc, product0);
    #endif
}

#endif

#if defined(USE_AVX2)

[[maybe_unused]] static int m256_hadd(__m256i sum, int bias) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {

    #if defined(USE_NNUE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
    #else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0         = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc              = _mm256_add_epi32(acc, product0);
    #endif
}

#endif

#if defined(USE_SSSE3)

[[maybe_unused]] static int m128_hadd(__m128i sum, int bias) {
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));  //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));  //_MM_PERM_CDAB
    return _mm_cvtsi128_si32(sum) + bias;
}

[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {
    __m128i product0 = _mm_maddubs_epi16(a, b);
    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));
    acc              = _mm_add_epi32(acc, product0);
}

#endif

#if defined(USE_NEON_DOTPROD)

[[maybe_unused]] static void dotprod_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {
    acc = vdotq_s32(acc, a, b);
}
#endif

#if defined(USE_NEON)

[[maybe_unused]] static int neon_m128_reduce_add_epi32(int32x4_t s) {
    #if USE_NEON >= 8
    return vaddvq_s32(s);
    #else
    return s[0] + s[1] + s[2] + s[3];
    #endif
}

[[maybe_unused]] static int neon_m128_hadd(int32x4_t sum, int bias) {
    return neon_m128_reduce_add_epi32(sum) + bias;
}

#endif

#if USE_NEON >= 8
[[maybe_unused]] static void neon_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t product1 = vmull_high_s8(a, b);
    int16x8_t sum      = vpaddq_s16(product0, product1);
    acc                = vpadalq_s16(acc, sum);
}

#endif


} // namespace YaneuraOu::Eval::NNUE::SIMD

#endif // NNUE_SIMD_H_INCLUDED
