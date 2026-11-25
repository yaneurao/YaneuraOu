// NNUE評価関数の入力特徴量Pの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "p.h"
#include "index_list.h"

namespace YaneuraOu::Eval::NNUE::Features {

// 特徴量のうち、値が1であるインデックスのリストを取得する
void P::append_active_indices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::MaxActiveDimensions < MaxActiveDimensions) return;

  BonaPiece pieces[40];
  collect_bona_pieces(pos, perspective, pieces);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    active->push_back(pieces[i]);
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
void P::append_changed_indices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
  for (int i = 0; i < dp.count(); ++i) {
    if (dp.is_king(i)) continue;
    removed->push_back(dp.change(i).old_piece.from[perspective]);
    added->push_back(dp.change(i).new_piece.from[perspective]);
  }
}

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
