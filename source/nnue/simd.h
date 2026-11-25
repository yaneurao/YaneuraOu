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

// HACK: Use _mm256_loadu_si256() instead of _mm256_load_si256. Otherwise a binary
//       compiled with older g++ crashes because the output memory is not aligned
//       even though alignas is specified.

// HACK : _mm256_loadu_si256()を_mm256_load_si256の代わりに使え。さもなくば、
//       古いg++でコンパイルされた実行ファイルはクラッシュする。なぜなら、
//       alignasが指定されているのにalignされていないコードを生成しやがるからだ。

#if defined(USE_AVX2)
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) && !defined(__clang__)
#define _mm256_loadA_si256  _mm256_loadu_si256
#define _mm256_storeA_si256 _mm256_storeu_si256
#else
#define _mm256_loadA_si256  _mm256_load_si256
#define _mm256_storeA_si256 _mm256_store_si256
#endif
#endif

#if defined(USE_AVX512)
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) && !defined(__clang__)
#define _mm512_loadA_si512   _mm512_loadu_si512
#define _mm512_storeA_si512  _mm512_storeu_si512
#else
#define _mm512_loadA_si512   _mm512_load_si512
#define _mm512_storeA_si512  _mm512_store_si512
#endif
#endif

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
// FT用のベクトル型・マクロは、現在はnnue_feature_transformer.h側で定義する。
// LSX/LASX/RVVのincludeは命令の宣言を読むためのもので、計算経路の移植とは別。
namespace YaneuraOu::Eval::NNUE::SIMD {


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
