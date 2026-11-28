// NNUE evaluation function layer InputSlice definition
// NNUE評価関数の層InputSliceの定義

#ifndef NNUE_LAYERS_INPUT_SLICE_H_INCLUDED
#define NNUE_LAYERS_INPUT_SLICE_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"

namespace YaneuraOu::Eval::NNUE::Layers {

// Input layer
// 入力層
template <IndexType OutDims, IndexType Offset = 0>
class InputSlice {
 public:
  // Need to maintain alignment
  // アライメントを維持する必要がある
  static_assert(Offset % MaxSimdWidth == 0, "");

  // Output type
  // 出力の型
  using OutputType = TransformedFeatureType;

  // Output dimensionality
  // 出力の次元数
  static constexpr IndexType OutputDimensions = OutDims;

  // Size of forward propagation buffer used from the input layer to this layer
  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr usize BufferSize = 0;

  // Hash value embedded in the evaluation file
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 get_hash_value() {
    u32 hashValue = 0xEC42E90Du;
    hashValue ^= OutputDimensions ^ (Offset << 10);
    return hashValue;
  }

  // A string that represents the structure from the input layer to this layer
  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "InputSlice[" + std::to_string(OutputDimensions) + "(" +
        std::to_string(Offset) + ":" +
        std::to_string(Offset + OutputDimensions) + ")]";
  }

  // この層には読み込むパラメータがない。
  bool read_parameters(std::istream& /*stream*/) {
    return true;
  }

  // この層には書き出すパラメータがない。
  bool write_parameters(std::ostream& /*stream*/) const {
    return true;
  }

  // Forward propagation
  // 順伝播
  const OutputType* propagate(
      const TransformedFeatureType* transformed_features,
      char* /*buffer*/) const {
    return transformed_features + Offset;
  }
};

} // namespace YaneuraOu::Eval::NNUE::Layers

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // #ifndef NNUE_LAYERS_INPUT_SLICE_H_INCLUDED
