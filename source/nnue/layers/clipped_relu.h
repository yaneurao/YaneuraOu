// Definition of layer ClippedReLU of NNUE evaluation function
// NNUE評価関数の層ClippedReLUの定義

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../../config.h"

#if defined(EVAL_NNUE) || defined(EVAL_SFNN)
#include "../nnue_common.h"
#include "../simd.h"

namespace YaneuraOu::Eval::NNUE::Layers {

#if defined(EVAL_SFNN)
// Clipped ReLU
// 入力次元を直接指定するSFNN用のレイヤー。既存のSIMD経路とパディングを維持する。
template<IndexType InDims>
class ClippedReLU {
 public:
  // Input/output type
  // 入出力の型
  using InputType = i32;
  using OutputType = u8;

  // Number of input/output dimensions
  // 入出力の次元数
  static constexpr IndexType InputDimensions = InDims;
  static constexpr IndexType OutputDimensions = InputDimensions;
  static constexpr IndexType PaddedOutputDimensions =
          ceil_to_multiple<IndexType>(OutputDimensions, 32);

  using OutputBuffer = OutputType[PaddedOutputDimensions];

  // Hash value embedded in the evaluation file
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 get_hash_value(u32 prevHash) {
    u32 hashValue = 0x538D24C7u;
        hashValue += prevHash;
    return hashValue;
  }

  static std::string get_structure_string() {
    return "ClippedReLU[" +
        std::to_string(OutputDimensions) + "]";
  }

  // Read network parameters
  // パラメーターを読み込む。この層自身は重みを持たない。
  bool read_parameters(std::istream&) { return true; }

  // Write network parameters
  // パラメーターを書き込む。この層自身は重みを持たない。
  bool write_parameters(std::ostream&) const { return true; }

  // Forward propagation
  // 順伝播
  void propagate(const InputType* input, OutputType* output) const {

  #if defined(USE_AVX512)
    if constexpr (InputDimensions % 64 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 64;
      const __m512i kZero = _mm512_setzero_si512();
      const __m512i kOffsets = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
      const auto in = reinterpret_cast<const __m512i*>(input);
      const auto out = reinterpret_cast<__m512i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m512i words0 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 0]),
              _mm512_loadA_si512(&in[i * 4 + 1])), WeightScaleBits);
        const __m512i words1 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 2]),
              _mm512_loadA_si512(&in[i * 4 + 3])), WeightScaleBits);
          _mm512_storeA_si512(&out[i], _mm512_permutexvar_epi32(kOffsets, _mm512_max_epi8(
            _mm512_packs_epi16(words0, words1), kZero)));
      }
    }
    else

  #endif

  #if defined(USE_AVX2)
    if constexpr (InputDimensions % 32 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 32;
      const __m256i kZero = _mm256_setzero_si256();
      const __m256i kOffsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      const auto in = reinterpret_cast<const __m256i*>(input);
      const auto out = reinterpret_cast<__m256i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 0]),
              _mm256_loadA_si256(&in[i * 4 + 1])), WeightScaleBits);
        const __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 2]),
              _mm256_loadA_si256(&in[i * 4 + 3])), WeightScaleBits);
          _mm256_storeA_si256(&out[i], _mm256_permutevar8x32_epi32(_mm256_max_epi8(
            _mm256_packs_epi16(words0, words1), kZero), kOffsets));
      }
    }
    else

  #endif

  #if defined(USE_SSE2)
    if constexpr (InputDimensions % 16 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 16;

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
              _mm_load_si128(&in[i * 4 + 1])), WeightScaleBits);
          const __m128i words1 = _mm_srai_epi16(_mm_packs_epi32(
              _mm_load_si128(&in[i * 4 + 2]),
              _mm_load_si128(&in[i * 4 + 3])), WeightScaleBits);
          const __m128i packedbytes = _mm_packs_epi16(words0, words1);
          _mm_store_si128(&out[i],
  #if defined(USE_SSE41)
              _mm_max_epi8(packedbytes, kZero)
  #else // SSE4.1を使わないSSE2経路
              _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
  #endif
          );
      }
    }
    else

  #endif

  #if defined(USE_MMX)
    if constexpr (InputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 8;
      const __m64 k0x80s = _mm_set1_pi8(-128);
      const auto in = reinterpret_cast<const __m64*>(input);
      const auto out = reinterpret_cast<__m64*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m64 words0 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 0], in[i * 4 + 1]),
            WeightScaleBits);
        const __m64 words1 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 2], in[i * 4 + 3]),
            WeightScaleBits);
        const __m64 packedbytes = _mm_packs_pi16(words0, words1);
        out[i] = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
      }
      _mm_empty();
    }
    else

  #endif

  #if defined(USE_NEON)
    if constexpr (InputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 8;
      const int8x8_t kZero = {0};
      const auto in = reinterpret_cast<const int32x4_t*>(input);
      const auto out = reinterpret_cast<int8x8_t*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        int16x8_t shifted;
        const auto pack = reinterpret_cast<int16x4_t*>(&shifted);
        pack[0] = vqshrn_n_s32(in[i * 2 + 0], WeightScaleBits);
        pack[1] = vqshrn_n_s32(in[i * 2 + 1], WeightScaleBits);
        out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
      }
    }
    else

  #endif

    for (IndexType i = 0; i < InputDimensions; ++i) {
      output[i] = static_cast<OutputType>(
          std::clamp(input[i] >> WeightScaleBits, 0, 127));
    }
    if constexpr (PaddedOutputDimensions > OutputDimensions) {
      std::fill(output + OutputDimensions, output + PaddedOutputDimensions, OutputType{0});
    }
  }

};
#else
// 従来NNUEは前段レイヤーを所有し、順伝播・読み込みを前段へ再帰させる。
// 明示次元型のStockfish/SFNNとはAPIが異なるため、従来の型とバッファ配置を残す。
// Clipped ReLU
template <typename PreviousLayer>
class ClippedReLU {
 public:
  // Input/output type
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = u8;
  static_assert(std::is_same<InputType, i32>::value, "");

  // Number of input/output dimensions
  // 入出力の次元数
  static constexpr IndexType InputDimensions =
      PreviousLayer::OutputDimensions;
  static constexpr IndexType OutputDimensions = InputDimensions;
  static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

  // Size of forward propagation buffer used in this layer
  // この層で使用する順伝播用バッファのサイズ
  static constexpr usize SelfBufferSize =
      ceil_to_multiple(OutputDimensions * sizeof(OutputType), CacheLineSize);

  // Size of the forward propagation buffer used from the input layer to this layer
  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr usize BufferSize =
      PreviousLayer::BufferSize + SelfBufferSize;

  // Hash value embedded in the evaluation file
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 get_hash_value() {
    u32 hashValue = 0x538D24C7u;
    hashValue += PreviousLayer::get_hash_value();
    return hashValue;
  }

  // ハッシュ値を前段の値から更新するときのヘルパー
  static constexpr u32 get_hash_value(u32 prevHash) {
    return 0x538D24C7u + prevHash;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "ClippedReLU[" +
        std::to_string(OutputDimensions) + "](" +
        PreviousLayer::GetStructureString() + ")";
  }

  // Read network parameters
  // パラメータを読み込む
  bool read_parameters(std::istream& stream) {
    return previous_layer_.read_parameters(stream);
  }

  // パラメータを書き込む
  bool write_parameters(std::ostream& stream) const {
    return previous_layer_.write_parameters(stream);
  }

  // Forward propagation
  // 順伝播
  const OutputType* propagate(
      const TransformedFeatureType* transformed_features, char* buffer) const {
    const auto input = previous_layer_.propagate(
        transformed_features, buffer + SelfBufferSize);
    const auto output = reinterpret_cast<OutputType*>(buffer);

  // 端数が生じないように、適切なSIMD幅を選択する。
  #if defined(USE_AVX512)
    if constexpr (InputDimensions % 64 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 64;
      const __m512i kZero = _mm512_setzero_si512();
      const __m512i kOffsets = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
      const auto in = reinterpret_cast<const __m512i*>(input);
      const auto out = reinterpret_cast<__m512i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m512i words0 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 0]),
              _mm512_loadA_si512(&in[i * 4 + 1])), WeightScaleBits);
        const __m512i words1 = _mm512_srai_epi16(_mm512_packs_epi32(
              _mm512_loadA_si512(&in[i * 4 + 2]),
              _mm512_loadA_si512(&in[i * 4 + 3])), WeightScaleBits);
          _mm512_storeA_si512(&out[i], _mm512_permutexvar_epi32(kOffsets, _mm512_max_epi8(
            _mm512_packs_epi16(words0, words1), kZero)));
      }
    }
    else
  
  #endif

  #if defined(USE_AVX2)
    if constexpr (InputDimensions % 32 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 32;
      const __m256i kZero = _mm256_setzero_si256();
      const __m256i kOffsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      const auto in = reinterpret_cast<const __m256i*>(input);
      const auto out = reinterpret_cast<__m256i*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 0]),
              _mm256_loadA_si256(&in[i * 4 + 1])), WeightScaleBits);
        const __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(
              _mm256_loadA_si256(&in[i * 4 + 2]),
              _mm256_loadA_si256(&in[i * 4 + 3])), WeightScaleBits);
          _mm256_storeA_si256(&out[i], _mm256_permutevar8x32_epi32(_mm256_max_epi8(
            _mm256_packs_epi16(words0, words1), kZero), kOffsets));
      }
    }
    else

  #endif

  #if defined(USE_SSE2)
    if constexpr (InputDimensions % 16 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 16;

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
              _mm_load_si128(&in[i * 4 + 1])), WeightScaleBits);
          const __m128i words1 = _mm_srai_epi16(_mm_packs_epi32(
              _mm_load_si128(&in[i * 4 + 2]),
              _mm_load_si128(&in[i * 4 + 3])), WeightScaleBits);
          const __m128i packedbytes = _mm_packs_epi16(words0, words1);
          _mm_store_si128(&out[i],
  #if defined(USE_SSE41)
              _mm_max_epi8(packedbytes, kZero)
  #else // SSE4.1を使わないSSE2経路
              _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
  #endif
          );
      }
    }
    else
  
  #endif

  #if defined(USE_MMX)
    if constexpr (InputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 8;
      const __m64 k0x80s = _mm_set1_pi8(-128);
      const auto in = reinterpret_cast<const __m64*>(input);
      const auto out = reinterpret_cast<__m64*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m64 words0 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 0], in[i * 4 + 1]),
            WeightScaleBits);
        const __m64 words1 = _mm_srai_pi16(
            _mm_packs_pi32(in[i * 4 + 2], in[i * 4 + 3]),
            WeightScaleBits);
        const __m64 packedbytes = _mm_packs_pi16(words0, words1);
        out[i] = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
      }
      _mm_empty();
    }
    else
  
  #endif

  #if defined(USE_NEON)
    if constexpr (InputDimensions % 8 == 0)
    {
      constexpr IndexType kNumChunks = InputDimensions / 8;
      const int8x8_t kZero = {0};
      const auto in = reinterpret_cast<const int32x4_t*>(input);
      const auto out = reinterpret_cast<int8x8_t*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        int16x8_t shifted;
        const auto pack = reinterpret_cast<int16x4_t*>(&shifted);
        pack[0] = vqshrn_n_s32(in[i * 2 + 0], WeightScaleBits);
        pack[1] = vqshrn_n_s32(in[i * 2 + 1], WeightScaleBits);
        out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
      }
    }
    else

  #endif

    for (IndexType i = 0; i < InputDimensions; ++i) {
      output[i] = static_cast<OutputType>(
          std::clamp(input[i] >> WeightScaleBits, 0, 127));
    }
    if constexpr (PaddedOutputDimensions > OutputDimensions) {
      std::fill(output + OutputDimensions, output + PaddedOutputDimensions, OutputType{0});
    }
    return output;
  }

 private:
   // この層の直前の層
   PreviousLayer previous_layer_;
};
#endif

} // namespace YaneuraOu::Eval::NNUE::Layers
#endif // defined(EVAL_NNUE) || defined(EVAL_SFNN)

#endif // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
