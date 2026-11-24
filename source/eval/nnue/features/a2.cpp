// NNUE評価関数の入力特徴量A2の定義

#include "../../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "a2.h"
#include "index_list.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

namespace {
  // 後手玉(e_king..fe_end2)を自玉plane(f_king..e_king)に折りたたむv2マッピング
  inline IndexType MapToA2Index(BonaPiece bp) {
    return static_cast<IndexType>(bp >= e_king ? bp - SQ_NB : bp);
  }
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
void A2::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  BonaPiece pieces[40];
  collect_bona_pieces(pos, perspective, pieces);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
    active->push_back(MapToA2Index(pieces[i]));
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
void A2::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
  for (int i = 0; i < dp.count(); ++i) {
    removed->push_back(MapToA2Index(static_cast<BonaPiece>(
        dp.change(i).old_piece.from[perspective])));
    added->push_back(MapToA2Index(static_cast<BonaPiece>(
        dp.change(i).new_piece.from[perspective])));
  }
}

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
