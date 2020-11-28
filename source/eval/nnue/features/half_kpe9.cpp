// NNUE評価関数の入力特徴量HalfKPE9の定義

#include "../../../config.h"

#if defined(EVAL_NNUE) && defined(LONG_EFFECT_LIBRARY) && defined(USE_BOARD_EFFECT_PREV)

#include "half_kpe9.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

inline Square GetSquareFromBonaPiece(BonaPiece p) {
  if (p < fe_hand_end) {
    return SQ_NB;
  }
  else {
    return static_cast<Square>((p - fe_hand_end) % SQ_NB);
  }
}

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

inline bool IsDirty(const Eval::DirtyPiece& dp, PieceNumber pn) {
  for (int i = 0; i < dp.dirty_num; ++i) {
    if (pn == dp.pieceNo[i]) {
      return true;
    }
  }
  return false;
}

// 玉の位置とBonaPieceと利き数から特徴量のインデックスを求める
template <Side AssociatedKing>
inline IndexType HalfKPE9<AssociatedKing>::MakeIndex(Square sq_k, BonaPiece p, int effect1, int effect2) {
  return (static_cast<IndexType>(fe_end) * static_cast<IndexType>(sq_k) + p)
       + (static_cast<IndexType>(fe_end) * static_cast<IndexType>(SQ_NB) * (effect1 * 3 + effect2));
}

// 駒の情報を取得する
template <Side AssociatedKing>
inline void HalfKPE9<AssociatedKing>::GetPieces(
    const Position& pos, Color perspective,
    BonaPiece** pieces, Square* sq_target_k) {
  *pieces = (perspective == BLACK) ?
      pos.eval_list()->piece_list_fb() :
      pos.eval_list()->piece_list_fw();
  const PieceNumber target = (AssociatedKing == Side::kFriend) ?
      static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective) :
      static_cast<PieceNumber>(PIECE_NUMBER_KING + ~perspective);
  *sq_target_k = static_cast<Square>(((*pieces)[target] - f_king) % SQ_NB);
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKPE9<AssociatedKing>::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  auto& board_effect = pos.board_effect;

  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    BonaPiece p = pieces[i];
    Square sq_p = GetSquareFromBonaPiece(p);

    active->push_back(MakeIndex(sq_target_k, p
        , GetEffectCount(pos, sq_p, perspective, perspective, false)
        , GetEffectCount(pos, sq_p, perspective, ~perspective, false)
      ));
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKPE9<AssociatedKing>::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {
  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  const auto& dp = pos.state()->dirtyPiece;

  for (int i = 0; i < dp.dirty_num; ++i) {
    if (dp.pieceNo[i] >= PIECE_NUMBER_KING) continue;
    
    const auto old_p = static_cast<BonaPiece>(dp.changed_piece[i].old_piece.from[perspective]);
    Square old_sq_p = GetSquareFromBonaPiece(old_p);
    removed->push_back(MakeIndex(sq_target_k, old_p
        , GetEffectCount(pos, old_sq_p, perspective, perspective, true)
        , GetEffectCount(pos, old_sq_p, perspective, ~perspective, true)
      ));

    const auto new_p = static_cast<BonaPiece>(dp.changed_piece[i].new_piece.from[perspective]);
    Square new_sq_p = GetSquareFromBonaPiece(new_p);
    added->push_back(MakeIndex(sq_target_k, new_p
        , GetEffectCount(pos, new_sq_p, perspective, perspective, false)
        , GetEffectCount(pos, new_sq_p, perspective, ~perspective, false)
      ));
  }

  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    if (IsDirty(dp, i)) {
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
      removed->push_back(MakeIndex(sq_target_k, p, effectCount_prev_1, effectCount_prev_2));
      added->push_back(MakeIndex(sq_target_k, p, effectCount_now_1, effectCount_now_2));
    }
  }
}

template class HalfKPE9<Side::kFriend>;
template class HalfKPE9<Side::kEnemy>;

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
