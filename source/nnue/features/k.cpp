// NNUE評価関数の入力特徴量Kの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "k.h"
#include "../../position.h"
#include "index_list.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量のうち、値が1であるインデックスのリストを取得する
void K::append_active_indices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::MaxActiveDimensions < MaxActiveDimensions) return;

  for (Color c : {BLACK, WHITE}) {
    const Square sq = pos.square<KING>(c);
    ASSERT_LV5(is_ok(sq));
    active->push_back(board_bona_piece(make_piece(c, KING), sq).from[perspective] - fe_end);
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
void K::append_changed_indices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
  if (dp.king_moved()) {
    removed->push_back(
        dp.change(0).old_piece.from[perspective] - fe_end);
    added->push_back(
        dp.change(0).new_piece.from[perspective] - fe_end);
  }
}


} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
