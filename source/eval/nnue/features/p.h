// NNUE評価関数の入力特徴量Pの定義

#ifndef CLASSIC_NNUE_FEATURES_P_H
#define CLASSIC_NNUE_FEATURES_P_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量P：玉以外の駒のBonaPiece
class P {
 public:
  // 特徴量名
  static constexpr const char* kName = "P";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t kHashValue = 0x764CFB4Bu;
  // 特徴量の次元数
  static constexpr IndexType kDimensions = fe_end;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType kMaxActiveDimensions = PIECE_NUMBER_KING;
  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
