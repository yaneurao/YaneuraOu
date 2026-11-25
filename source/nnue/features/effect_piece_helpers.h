// Helpers shared by the attack-count features HalfKPE9 and PE9.
// 利き数を用いるHalfKPE9・PE9で共用する補助処理。

#ifndef NNUE_FEATURES_EFFECT_PIECE_HELPERS_H_INCLUDED
#define NNUE_FEATURES_EFFECT_PIECE_HELPERS_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN)) && defined(LONG_EFFECT_LIBRARY) && defined(USE_BOARD_EFFECT_PREV)

#include "../../evaluate.h"
#include "../../position.h"

namespace YaneuraOu::Eval::NNUE::Features {

// Return the board square in the BonaPiece perspective, or SQ_NB for hand pieces.
// BonaPieceの視点での盤上のマスを返す。持駒にはSQ_NBを返す。
inline Square GetSquareFromBonaPiece(BonaPiece p) {
  if (p < fe_hand_end) {
    return SQ_NB;
  }
  else {
    return static_cast<Square>((p - fe_hand_end) % SQ_NB);
  }
}

// Count attacks in board coordinates, capped at two; hand pieces have no attacks.
// BonaPieceの視点から盤面の座標へ戻して利き数を取得する。上限2、持駒は0。
inline int GetEffectCount(const Position& pos, Square sq_p, Color perspective_org, Color perspective, bool prev_effect) {
  if (sq_p == SQ_NB) {
    return 0;
  }
  else {
    if (perspective_org == WHITE) {
      sq_p = Inv(sq_p);
    }

    if (prev_effect) {
      return std::min(int(pos.board_effect_prev[perspective].effect(sq_p)), 2);
    }
    else {
      return std::min(int(pos.board_effect[perspective].effect(sq_p)), 2);
    }
  }
}

// Skip pieces already covered by the move changes when comparing attack counts.
// 利き数の変化を調べる際、着手差分で処理済みの駒を除外するための判定。
inline bool IsDirty(const Eval::DirtyPiece& dp, BonaPiece p, Color perspective) {
  for (int i = 0; i < dp.count(); ++i)
    if (p == dp.change(i).new_piece.from[perspective])
      return true;
  return false;
}

} // namespace YaneuraOu::Eval::NNUE::Features

#endif

#endif // NNUE_FEATURES_EFFECT_PIECE_HELPERS_H_INCLUDED
