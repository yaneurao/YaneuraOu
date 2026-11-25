// NNUE評価関数の層Sumの定義

#ifndef NNUE_LAYERS_SUM_H_INCLUDED
#define NNUE_LAYERS_SUM_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// 複数の層の出力の和を取る層
template <typename FirstPreviousLayer, typename... RemainingPreviousLayers>
class Sum : public Sum<RemainingPreviousLayers...> {
 private:
  using Head = FirstPreviousLayer;
  using Tail = Sum<RemainingPreviousLayers...>;

 public:
  // 入出力の型
  using InputType = typename Head::OutputType;
  using OutputType = InputType;
  static_assert(std::is_same<InputType, typename Tail::InputType>::value, "");

  // 入出力の次元数
  static constexpr IndexType InputDimensions = Head::OutputDimensions;
  static constexpr IndexType OutputDimensions = InputDimensions;
  static_assert(InputDimensions == Tail::InputDimensions , "");

  // この層で使用する順伝播用バッファのサイズ
  static constexpr usize SelfBufferSize =
      ceil_to_multiple(OutputDimensions * sizeof(OutputType), CacheLineSize);

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr usize BufferSize =
      std::max(Head::BufferSize + SelfBufferSize, Tail::BufferSize);

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 get_hash_value() {
    u32 hashValue = 0xBCE400B4u;
    hashValue ^= Head::get_hash_value() >> 1;
    hashValue ^= Head::get_hash_value() << 31;
    hashValue ^= Tail::get_hash_value() >> 2;
    hashValue ^= Tail::get_hash_value() << 30;
    return hashValue;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "Sum[" +
        std::to_string(OutputDimensions) + "](" + GetSummandsString() + ")";
  }

  // パラメータを読み込む
  bool read_parameters(std::istream& stream) {
    if (!Tail::read_parameters(stream)) return false;
    return previous_layer_.read_parameters(stream);
  }

  // パラメータを書き込む
  bool write_parameters(std::ostream& stream) const {
    if (!Tail::write_parameters(stream)) return false;
    return previous_layer_.write_parameters(stream);
  }

  // 順伝播
  const OutputType* propagate(
      const TransformedFeatureType* transformed_features, char* buffer) const {
    Tail::propagate(transformed_features, buffer);
    const auto head_output = previous_layer_.propagate(
        transformed_features, buffer + SelfBufferSize);
    const auto output = reinterpret_cast<OutputType*>(buffer);
    for (IndexType i = 0; i < OutputDimensions; ++i) {
      output[i] += head_output[i];
    }
    return output;
  }

 protected:
  // 和を取る対象となる層のリストを表す文字列
  static std::string GetSummandsString() {
    return Head::GetStructureString() + "," + Tail::GetSummandsString();
  }

  // この層の直前の層
  FirstPreviousLayer previous_layer_;
};

// 複数の層の出力の和を取る層（テンプレート引数が1つの場合）
template <typename PreviousLayer>
class Sum<PreviousLayer> {
 public:
  // 入出力の型
  using InputType = typename PreviousLayer::OutputType;
  using OutputType = InputType;

  // 入出力の次元数
  static constexpr IndexType InputDimensions =
      PreviousLayer::OutputDimensions;
  static constexpr IndexType OutputDimensions = InputDimensions;

  // 入力層からこの層までで使用する順伝播用バッファのサイズ
  static constexpr usize BufferSize = PreviousLayer::BufferSize;

  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 get_hash_value() {
    u32 hashValue = 0xBCE400B4u;
    hashValue ^= PreviousLayer::get_hash_value() >> 1;
    hashValue ^= PreviousLayer::get_hash_value() << 31;
    return hashValue;
  }

  // 入力層からこの層までの構造を表す文字列
  static std::string GetStructureString() {
    return "Sum[" +
        std::to_string(OutputDimensions) + "](" + GetSummandsString() + ")";
  }

  // パラメータを読み込む
  bool read_parameters(std::istream& stream) {
    return previous_layer_.read_parameters(stream);
  }

  // パラメータを書き込む
  bool write_parameters(std::ostream& stream) const {
    return previous_layer_.write_parameters(stream);
  }

  // 順伝播
  const OutputType* propagate(
      const TransformedFeatureType* transformed_features, char* buffer) const {
    return previous_layer_.propagate(transformed_features, buffer);
  }

 protected:
  // 和を取る対象となる層のリストを表す文字列
  static std::string GetSummandsString() {
    return PreviousLayer::GetStructureString();
  }

  // この層の直前の層
  PreviousLayer previous_layer_;
};

} // namespace Eval::NNUE::Layers
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
