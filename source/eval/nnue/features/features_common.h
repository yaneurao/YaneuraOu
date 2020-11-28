//Common header of input features of NNUE evaluation function
// NNUE評価関数の入力特徴量の共通ヘッダ

#ifndef NNUE_FEATURES_COMMON_H_INCLUDED
#define NNUE_FEATURES_COMMON_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "../nnue_common.h"

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

// 手番側or相手側
enum class Side {
  kFriend,  // 手番側 // side to move
  kEnemy,   // 相手側
};

}  // namespace Eval::NNUE::Features


#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_FEATURES_COMMON_H_INCLUDED

