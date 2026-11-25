//Common header of input features of NNUE evaluation function
// NNUE評価関数の入力特徴量の共通ヘッダ

#ifndef NNUE_FEATURES_COMMON_H_INCLUDED
#define NNUE_FEATURES_COMMON_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "../nnue_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// インデックスリストの型
class IndexList;

// 特徴量セットを表すクラステンプレート
template <typename... FeatureTypes>
class FeatureSet;

// Trigger to perform full calculations instead of difference only
// 差分計算の代わりに全計算を行うタイミングの種類
enum class TriggerEvent {
  kNone,             // 可能な場合は常に差分計算する
  kFriendKingMoved,  // 自玉が移動した場合に全計算する  // calculate full evaluation when own king moves
  kEnemyKingMoved,   // 敵玉が移動した場合に全計算する
  kAnyKingMoved,     // どちらかの玉が移動した場合に全計算する
  kAnyPieceMoved,    // 常に全計算する
};

// 特徴量の視点側(perspective)か、その相手側か。現在の手番とは独立。
enum class Side {
  kFriend,  // 視点側
  kEnemy,   // 視点の相手側
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // #ifndef NNUE_FEATURES_COMMON_H_INCLUDED

