// NNUE評価関数の入力特徴量Pの定義

#ifndef NNUE_FEATURES_P_H_INCLUDED
#define NNUE_FEATURES_P_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

// 特徴量P：玉以外の駒のBonaPiece
class P {
 public:
  // 特徴量名
  static constexpr const char* kName = "P";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 HashValue = 0x764CFB4Bu;
  // 特徴量の次元数
  static constexpr IndexType Dimensions = fe_end;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType MaxActiveDimensions = PIECE_NUMBER_KING;
  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void append_active_indices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void append_changed_indices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp);
};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
