// NNUE評価関数の入力特徴量HalfKPE9の定義

#ifndef _NNUE_FEATURES_HALF_KPE9_H_
#define _NNUE_FEATURES_HALF_KPE9_H_

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace Eval {

namespace NNUE {

namespace Features {

// 特徴量HalfKPE9：自玉または敵玉の位置と、玉以外の駒の位置と、利き数の組み合わせ
// ・利き数は先後各々最大2までに制限。0～2で3通り。先後で3*3=9通り。なお、持ち駒の場合は利き数0とする。
template <Side AssociatedKing>
class HalfKPE9 {
 public:
  // 特徴量名
  static constexpr const char* kName =
      (AssociatedKing == Side::kFriend) ? "HalfKPE9(Friend)" : "HalfKPE9(Enemy)";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t kHashValue =
      0x5D69D5B9u ^ (AssociatedKing == Side::kFriend);
  // 特徴量の次元数
  static constexpr IndexType kDimensions =
      static_cast<IndexType>(SQ_NB) * static_cast<IndexType>(fe_end) * 3 * 3;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType kMaxActiveDimensions = PIECE_NUMBER_KING;

  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger =
      (AssociatedKing == Side::kFriend) ?
      TriggerEvent::kFriendKingMoved : TriggerEvent::kEnemyKingMoved;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);

  // 玉の位置とBonaPieceと利き数から特徴量のインデックスを求める
  static IndexType MakeIndex(Square sq_k, BonaPiece p, int effect1, int effect2);

 private:
  // 駒の情報を取得する
  static void GetPieces(const Position& pos, Color perspective,
                        BonaPiece** pieces, Square* sq_target_k);
};

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
