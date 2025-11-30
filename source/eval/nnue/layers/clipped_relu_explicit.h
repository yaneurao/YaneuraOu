// Definition of layer ClippedReLUExplicit of NNUE evaluation function

#ifndef NNUE_LAYERS_CLIPPED_RELU_EXPLICIT_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_EXPLICIT_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// Clipped ReLU (Explicit Dimensions)
template <IndexType InputDimensions>
class ClippedReLUExplicit {
 public:
  using InputType = std::int32_t;
  using OutputType = std::uint8_t;

  static constexpr IndexType kInputDimensions = InputDimensions;
  static constexpr IndexType kOutputDimensions = kInputDimensions;
  static constexpr IndexType PaddedOutputDimensions =
          CeilToMultiple<IndexType>(kOutputDimensions, 32);

  using OutputBuffer = OutputType[PaddedOutputDimensions];

  static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
    std::uint32_t hash_value = 0x538D24C7u;
        hash_value += prevHash;
    return hash_value;
  }

  static std::string GetStructureString() {
    return "ClippedReLU[" +
        std::to_string(kOutputDimensions) + "]";
  }

  Tools::Result ReadParameters(std::istream& stream) {
    return Tools::ResultCode::Ok;
  }

  bool WriteParameters(std::ostream& stream) const {
    return true;
  }

  void Propagate(const InputType* input, OutputType* output) const {

  #if defined(USE_AVX512)
    if constexpr (kInputDimensions % 64 == 0)
    {
      constexpr IndexType kNumChunks = kInputDimensions / 64;
      const __m512i kZero = _mm512_setzero_si512();
      const __m512i kOffsets = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
      const auto in = reinterpret_cast<const __m512i*>(input);
      const auto out = reinterpret_cast<__m512i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m512i words0 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 0]),
              _mm512_loadA_si512(&in[i * 4 + 1])), kWeightScaleBits);
        const __m512i words1 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 2]),
              _mm512_loadA_si512(&in[i * 4 + 3])), kWeightScaleBits);
          _mm512_storeA_si512(&out[i], _mm512_permutexvar_epi32(kOffsets, _mm512_max_epi8(
            _mm512_packs_epi16(words0, words1), kZero)));
      }
    }
    else

  #endif

  #if defined(USE_AVX2)
    if constexpr (kInputDimensions % 32 == 0)
    {
      constexpr IndexType kNumChunks = kInputDimensions / 32;
      const __m256i kZero = _mm256_setzero_si256();
      const __m256i kOffsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      const auto in = reinterpret_cast<const __m256i*>(input);
      const auto out = reinterpret_cast<__m256i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 0]),
              _mm256_loadA_si256(&in[i * 4 + 1])), kWeightScaleBits);
        const __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 2]),
              _mm256_loadA_si256(&in[i * 4 + 3])), kWeightScaleBits);
          _mm256_storeA_si256(&out[i], _mm256_permutevar8x32_epi32(_mm256_max_epi8(
            _mm256_packs_epi16(words0, words1), kZero), kOffsets));
      }
    }
    else

  #endif

  #if defined(USE_SSE2)
    if constexpr (kInputDimensions % 16 == 0)
    {
      constexpr IndexType kNumChunks = kInputDimensions / 16;

  #ifdef USE_SSE41
      const __m128i kZero = _mm_setzero_si128();
  #else
      const __m128i k0x80s = _mm_set1_epi8(-128);
  #endif
      const auto in = reinterpret_cast<const __m128i*>(input);
      const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < kNumChunks; ++i) {
          const __m128i words0 = _mm_srai_epi16(_mm_packs_epi32(
              _mm_load_si128(&in[i * 4 + 0]),
              _mm_load_si128(&in[i * 4 + 1])), kWeightScaleBits);
          const __m128i words1 = _mm_srai_epi16(_mm_packs_epi32(
              _mm_load_si128(&in[i * 4 + 2]),
              _mm_load_si128(&in[i * 4 + 3])), kWeightScaleBits);
          const __m128i packedbytes = _mm_packs_epi16(words0, words1);
          _mm_store_si128(&out[i],
  #if defined(USE_SSE41)
              _mm_max_epi8(packedbytes, kZero)
  #else // SSE4非対応だがSSE3は使える環境
              _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
  #endif
          );
      }
    }
    else

  #endif

  #if defined(USE_MMX)
    if constexpr (kInputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = kInputDimensions / 8;
      const __m64 k0x80s = _mm_set1_pi8(-128);
      const auto in = reinterpret_cast<const __m64*>(input);
      const auto out = reinterpret_cast<__m64*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m64 words0 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 0], in[i * 4 + 1]),
            kWeightScaleBits);
        const __m64 words1 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 2], in[i * 4 + 3]),
            kWeightScaleBits);
        const __m64 packedbytes = _mm_packs_pi16(words0, words1);
        out[i] = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
      }
      _mm_empty();
    }
    else

  #endif

  #if defined(USE_NEON)
    if constexpr (kInputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = kInputDimensions / 8;
      const int8x8_t kZero = {0};
      const auto in = reinterpret_cast<const int32x4_t*>(input);
      const auto out = reinterpret_cast<int8x8_t*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        int16x8_t shifted;
        const auto pack = reinterpret_cast<int16x4_t*>(&shifted);
        pack[0] = vqshrn_n_s32(in[i * 2 + 0], kWeightScaleBits);
        pack[1] = vqshrn_n_s32(in[i * 2 + 1], kWeightScaleBits);
        out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
      }
    }
    else

  #endif

    for (IndexType i = 0; i < kInputDimensions; ++i) {
      output[i] = static_cast<OutputType>(
          std::max(0, std::min(127, input[i] >> kWeightScaleBits)));
    }
  }

 private:
   friend class Trainer<ClippedReLUExplicit>;
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // NNUE_LAYERS_CLIPPED_RELU_EXPLICIT_H_INCLUDED
