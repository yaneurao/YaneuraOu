// NNUE評価関数の特徴量変換クラステンプレートのHalfKPE9用特殊化

#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9_H_

#include "../../../../config.h"

#if defined(EVAL_NNUE)

#include "../../features/half_kpe9.h"
#include "../../features/half_kp.h"
#include "../../features/pe9.h"
#include "../../features/p.h"
#include "../../features/half_relative_kp.h"
#include "factorizer.h"

namespace Eval {

namespace NNUE {

namespace Features {

// 入力特徴量を学習用特徴量に変換するクラステンプレート
// HalfKPE9用特殊化
template <Side AssociatedKing>
class Factorizer<HalfKPE9<AssociatedKing>> {
 private:
  using FeatureType = HalfKPE9<AssociatedKing>;

  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType kMaxActiveDimensions =
      FeatureType::kMaxActiveDimensions;

  // 学習用特徴量の種類
  enum TrainingFeatureType {
    kFeaturesHalfKPE9,
    kFeaturesHalfKP,
    kFeaturesHalfK,
    kFeaturesPE9,
    kFeaturesP,
    kFeaturesHalfRelativeKP,
    //
    kNumTrainingFeatureTypes,
  };

  // 学習用特徴量の情報
  static constexpr FeatureProperties kProperties[] = {
    // kFeaturesHalfKPE9
    {true, FeatureType::kDimensions},
    // kFeaturesHalfKP
    {true, Factorizer<HalfKP<AssociatedKing>>::GetDimensions()},
    // kFeaturesHalfK
    {true, SQ_NB},
    // kFeaturesPE9
    {true, Factorizer<PE9>::GetDimensions()},
    // kFeaturesP
    {true, Factorizer<P>::GetDimensions()},
    // kFeaturesHalfRelativeKP
    {true, Factorizer<HalfRelativeKP<AssociatedKing>>::GetDimensions()},
  };
  static_assert(GetArrayLength(kProperties) == kNumTrainingFeatureTypes, "");

 public:
  // 学習用特徴量の次元数を取得する
  static constexpr IndexType GetDimensions() {
    return GetActiveDimensions(kProperties);
  }

  // 学習用特徴量のインデックスと学習率のスケールを取得する
  static void AppendTrainingFeatures(
      IndexType base_index, std::vector<TrainingFeature>* training_features) {
    // kFeaturesHalfKPE9
    IndexType index_offset = AppendBaseFeature<FeatureType>(
        kProperties[kFeaturesHalfKPE9], base_index, training_features);

    IndexType index_HalfKP = base_index % (static_cast<IndexType>(fe_end) * static_cast<IndexType>(SQ_NB));
    const auto sq_k = static_cast<Square>(index_HalfKP / fe_end);
    const auto p = static_cast<BonaPiece>(index_HalfKP % fe_end);

    int effect_index = base_index / (static_cast<IndexType>(fe_end) * static_cast<IndexType>(SQ_NB));
    int effect1 = effect_index / 3;
    int effect2 = effect_index % 3;

    // kFeaturesHalfKP
    index_offset += InheritFeaturesIfRequired<HalfKP<AssociatedKing>>(index_offset, kProperties[kFeaturesHalfKP], index_HalfKP, training_features);

    // kFeaturesHalfK
    {
      const auto& properties = kProperties[kFeaturesHalfK];
      if (properties.active) {
        training_features->emplace_back(index_offset + sq_k);
        index_offset += properties.dimensions;
      }
    }

    // kFeaturesPE9
    {
      index_offset += InheritFeaturesIfRequired<PE9>(
          index_offset, kProperties[kFeaturesPE9], PE9::MakeIndex(p, effect1, effect2), training_features);
    }

    // kFeaturesP
    index_offset += InheritFeaturesIfRequired<P>(
        index_offset, kProperties[kFeaturesP], p, training_features);

    // kFeaturesHalfRelativeKP
    if (p >= fe_hand_end) {
      index_offset += InheritFeaturesIfRequired<HalfRelativeKP<AssociatedKing>>(
          index_offset, kProperties[kFeaturesHalfRelativeKP],
          HalfRelativeKP<AssociatedKing>::MakeIndex(sq_k, p),
          training_features);
    } else {
      index_offset += SkipFeatures(kProperties[kFeaturesHalfRelativeKP]);
    }

    ASSERT_LV5(index_offset == GetDimensions());
  }
};

template <Side AssociatedKing>
constexpr FeatureProperties Factorizer<HalfKPE9<AssociatedKing>>::kProperties[];

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
