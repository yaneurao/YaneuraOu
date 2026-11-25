// NNUE評価関数の入力特徴量PE9の定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN)) && defined(LONG_EFFECT_LIBRARY) && defined(USE_BOARD_EFFECT_PREV)

#include "pe9.h"
#include "../../position.h"
#include "index_list.h"
#include "effect_piece_helpers.h"

namespace YaneuraOu::Eval::NNUE::Features {

// BonaPieceと利き数から特徴量のインデックスを求める
IndexType PE9::make_index(BonaPiece p, int effect1, int effect2) {
  return p + (static_cast<IndexType>(fe_end) * (effect1 * 3 + effect2));
}

// 駒の情報を取得する
inline void PE9::GetPieces(
    const Position& pos, Color perspective,
    BonaPiece* pieces) {
  collect_bona_pieces(pos, perspective, pieces);
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
void PE9::append_active_indices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::MaxActiveDimensions < MaxActiveDimensions) return;

  BonaPiece pieces[40];
  GetPieces(pos, perspective, pieces);

  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    BonaPiece p = pieces[i];
    Square sq_p = GetSquareFromBonaPiece(p);

    active->push_back(make_index(p
        , GetEffectCount(pos, sq_p, perspective, perspective, false)
        , GetEffectCount(pos, sq_p, perspective, ~perspective, false)
      ));
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
void PE9::append_changed_indices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
  BonaPiece pieces[40];
  GetPieces(pos, perspective, pieces);

  for (int i = 0; i < dp.count(); ++i) {
    if (dp.is_king(i)) continue;
    
    const auto old_p = static_cast<BonaPiece>(dp.change(i).old_piece.from[perspective]);
    Square old_sq_p = GetSquareFromBonaPiece(old_p);
    removed->push_back(make_index(old_p
        , GetEffectCount(pos, old_sq_p, perspective, perspective, true)
        , GetEffectCount(pos, old_sq_p, perspective, ~perspective, true)
      ));

    const auto new_p = static_cast<BonaPiece>(dp.change(i).new_piece.from[perspective]);
    Square new_sq_p = GetSquareFromBonaPiece(new_p);
    added->push_back(make_index(new_p
        , GetEffectCount(pos, new_sq_p, perspective, perspective, false)
        , GetEffectCount(pos, new_sq_p, perspective, ~perspective, false)
      ));
  }

  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    if (IsDirty(dp, pieces[i], perspective)) {
      continue;
    }

    BonaPiece p = pieces[i];
    Square sq_p = GetSquareFromBonaPiece(p);

    int effectCount_prev_1 = GetEffectCount(pos, sq_p, perspective, perspective, true);
    int effectCount_prev_2 = GetEffectCount(pos, sq_p, perspective, ~perspective, true);
    int effectCount_now_1 = GetEffectCount(pos, sq_p, perspective, perspective, false);
    int effectCount_now_2 = GetEffectCount(pos, sq_p, perspective, ~perspective, false);

    if (   effectCount_prev_1 != effectCount_now_1
        || effectCount_prev_2 != effectCount_now_2) {
      removed->push_back(make_index(p, effectCount_prev_1, effectCount_prev_2));
      added->push_back(make_index(p, effectCount_now_1, effectCount_now_2));
    }
  }
}

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
