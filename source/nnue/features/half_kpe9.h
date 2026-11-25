// NNUE評価関数の入力特徴量HalfKPE9の定義

#ifndef NNUE_FEATURES_HALF_KPE9_H_INCLUDED
#define NNUE_FEATURES_HALF_KPE9_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {


// 特徴量HalfKPE9：自玉または敵玉の位置と、玉以外の駒の位置と、利き数の組み合わせ
// ・利き数は先後各々最大2までに制限。0～2で3通り。先後で3*3=9通り。なお、持ち駒の場合は利き数0とする。
template <Side AssociatedKing>
class HalfKPE9 {
 public:
  // 特徴量名
  static constexpr const char* kName =
      (AssociatedKing == Side::kFriend) ? "HalfKPE9(Friend)" : "HalfKPE9(Enemy)";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 HashValue =
      0x5D69D5B9u ^ (AssociatedKing == Side::kFriend);
  // 特徴量の次元数
  static constexpr IndexType Dimensions =
      static_cast<IndexType>(SQ_NB) * static_cast<IndexType>(fe_end) * 3 * 3;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType MaxActiveDimensions = PIECE_NUMBER_KING;

  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger =
      (AssociatedKing == Side::kFriend) ?
      TriggerEvent::kFriendKingMoved : TriggerEvent::kEnemyKingMoved;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void append_active_indices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void append_changed_indices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp);

  // 玉の位置とBonaPieceと利き数から特徴量のインデックスを求める
  static IndexType make_index(Square sq_k, BonaPiece p, int effect1, int effect2);

};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
