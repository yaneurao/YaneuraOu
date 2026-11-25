// NNUE評価関数の入力特徴量HalfRelativeKPの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "half_relative_kp.h"
#include "../../position.h"
#include "index_list.h"
#include "king_piece_helpers.h"

namespace YaneuraOu::Eval::NNUE::Features {

// 玉の位置とBonaPieceから特徴量のインデックスを求める
template <Side AssociatedKing>
inline IndexType HalfRelativeKP<AssociatedKing>::make_index(
    Square sq_k, BonaPiece p) {
  constexpr IndexType W = kBoardWidth;
  constexpr IndexType H = kBoardHeight;
  const IndexType piece_index = (p - fe_hand_end) / SQ_NB;
  const Square sq_p = static_cast<Square>((p - fe_hand_end) % SQ_NB);
  const IndexType relative_file = file_of(sq_p) - file_of(sq_k) + (W / 2);
  const IndexType relative_rank = rank_of(sq_p) - rank_of(sq_k) + (H / 2);
  return H * W * piece_index + H * relative_file + relative_rank;
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
template <Side AssociatedKing>
void HalfRelativeKP<AssociatedKing>::append_active_indices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::MaxActiveDimensions < MaxActiveDimensions) return;

  BonaPiece pieces[40];
  Square sq_target_k;
  collect_pieces_with_king<AssociatedKing>(pos, perspective, pieces, &sq_target_k);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    if (pieces[i] >= fe_hand_end) {
      active->push_back(make_index(sq_target_k, pieces[i]));
    }
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
template <Side AssociatedKing>
void HalfRelativeKP<AssociatedKing>::append_changed_indices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
  const Square sq_target_k = oriented_king_square<AssociatedKing>(pos, perspective);
  for (int i = 0; i < dp.count(); ++i) {
    if (dp.is_king(i)) continue;
    const auto old_p = static_cast<BonaPiece>(
        dp.change(i).old_piece.from[perspective]);
    if (old_p >= fe_hand_end) {
      removed->push_back(make_index(sq_target_k, old_p));
    }
    const auto new_p = static_cast<BonaPiece>(
        dp.change(i).new_piece.from[perspective]);
    if (new_p >= fe_hand_end) {
      added->push_back(make_index(sq_target_k, new_p));
    }
  }
}

template class HalfRelativeKP<Side::kFriend>;
template class HalfRelativeKP<Side::kEnemy>;

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
