// Helpers for nonzero NNUE input chunks.
// NNUE入力の非ゼロチャンクを扱う補助処理。

#ifndef NNZ_HELPER_H_INCLUDED
#define NNZ_HELPER_H_INCLUDED

#include <array>

#include "../config.h"

#if defined(EVAL_NNUE) || defined(EVAL_SFNN)
#include "nnue_common.h"
#include "simd.h"

namespace YaneuraOu::Eval::NNUE {

// Stockfishのnnz_helper.hと同じく、非ゼロ入力の列挙をAffine層から分離する。
// 現行StockfishはFT変換中にNNZInfoへ記録するが、やねうら王は既存の変換後の走査と、
// FT・Affine融合経路を維持する。このヘルパーは旧NNUEとSFNNで共用する。
// 入力は各バイトが0～127の活性化値で、i32単位の比較でも非ゼロなら正になる。

#if defined(USE_SSSE3) || USE_NEON >= 8

// For each 8-bit mask, list the set bits in increasing order.
// 8bitマスクの立っているビットを昇順に並べる。
alignas(CacheLineSize) inline constexpr
  std::array<std::array<u16, 8>, 256> lookup_indices = []() {
      std::array<std::array<u16, 8>, 256> v{};
      for (unsigned i = 0; i < 256; ++i)
      {
          unsigned count = 0;
          for (u16 bit = 0; bit < 8; ++bit)
              if (i & (1u << bit))
                  v[i][count++] = bit;
      }
      return v;
  }();

// Find nonzero i32 chunks containing 0..127 activation bytes.
// 各バイトが0～127の活性化値であるi32チャンクの非ゼロ要素を列挙する。
template<IndexType InputDimensions>
inline void find_nnz(const i32* input, u16* out, IndexType& count_out) {
    using SIMD::vec_t;
    constexpr IndexType kInputSimdWidth = sizeof(vec_t) / sizeof(i32);
    // Inputs are processed kInputSimdWidth at a time and outputs are processed 8 at a time.
    // 入力SIMD幅と8要素の出力単位の両方を満たす大きさで処理する。
    constexpr IndexType ChunkSize       = std::max<IndexType>(kInputSimdWidth, 8);
    constexpr IndexType kNumChunks       = InputDimensions / ChunkSize;
    constexpr IndexType kInputsPerChunk  = ChunkSize / kInputSimdWidth;
    constexpr IndexType kOutputsPerChunk = ChunkSize / 8;

    const auto     inputVector = reinterpret_cast<const vec_t*>(input);
    IndexType      count       = 0;
#if defined(USE_SSSE3)
    __m128i base = _mm_setzero_si128();
    const __m128i increment = _mm_set1_epi16(8);
#elif defined(USE_NEON)
    uint16x8_t base = vdupq_n_u16(0);
    const uint16x8_t increment = vdupq_n_u16(8);
#endif
    for (IndexType i = 0; i < kNumChunks; ++i)
    {
        // Bitmask of nonzero values in this chunk
        // このチャンクにある非ゼロ要素のビットマスク。
        unsigned nnz = 0;
        for (IndexType j = 0; j < kInputsPerChunk; ++j)
        {
            const vec_t inputChunk = inputVector[i * kInputsPerChunk + j];
            nnz |= unsigned(vec_nnz(inputChunk)) << (j * kInputSimdWidth);
        }
        for (IndexType j = 0; j < kOutputsPerChunk; ++j)
        {
            const auto lookup = (nnz >> (j * 8)) & 0xFF;
#if defined(USE_SSSE3)
            const auto offsets = _mm_load_si128(
              reinterpret_cast<const __m128i*>(&lookup_indices[lookup]));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_add_epi16(base, offsets));
#else
            const auto offsets = vld1q_u16(reinterpret_cast<const u16*>(&lookup_indices[lookup]));
            vst1q_u16(out + count, vaddq_u16(base, offsets));
#endif
            count += POPCNT32(lookup);
#if defined(USE_SSSE3)
            base = _mm_add_epi16(base, increment);
#else
            base = vaddq_u16(base, increment);
#endif
        }
    }
    count_out = count;
}

#endif

}  // namespace YaneuraOu::Eval::NNUE
#endif  // defined(EVAL_NNUE) || defined(EVAL_SFNN)

#endif  // NNZ_HELPER_H_INCLUDED
