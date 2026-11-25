// Shared piece collection for shogi HalfKP/HalfKA feature families.
// 将棋のHalfKP・HalfKA系の特徴量で共用する駒情報の収集処理。

#ifndef NNUE_FEATURES_KING_PIECE_HELPERS_H_INCLUDED
#define NNUE_FEATURES_KING_PIECE_HELPERS_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE) || defined(EVAL_SFNN)
#include "../../evaluate.h"
#include "../../position.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

// Return the associated king square in the feature's perspective, not the side to move.
// 特徴量の視点に合わせた玉位置を返す。perspectiveは現在の手番とは独立。
// 全計算で駒を収集済みの場合は、下のcollect_pieces_with_kingでその結果から玉位置を得る。
// Passing perspective by reference preserves GCC code for the friend-king feature path.
// 自玉を使う現行構成でGCCの生成コードを維持するため、perspectiveは参照で渡す。
template<Side AssociatedKing>
inline Square oriented_king_square(const Position& pos, const Color& perspective) {
    const Color target = AssociatedKing == Side::kFriend ? perspective : ~perspective;
    return perspective == BLACK ? pos.square<KING>(target) : Inv(pos.square<KING>(target));
}

// Preserve the fixed BonaPiece order and derive the perspective-oriented king square
// from that same collection. No extra board scan or king lookup is needed.
// BonaPieceの固定順序を維持し、同じ収集結果から視点に合わせた玉のマスを求める。
// 盤面の再走査や玉の位置の再取得は行わない。
template<Side AssociatedKing>
inline void collect_pieces_with_king(const Position& pos, Color perspective,
                                    BonaPiece* pieces, Square* sq_target_k) {
    collect_bona_pieces(pos, perspective, pieces);
    const PieceNumber target = (AssociatedKing == Side::kFriend) ?
      static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective) :
      static_cast<PieceNumber>(PIECE_NUMBER_KING + ~perspective);
    *sq_target_k = static_cast<Square>((pieces[target] - f_king) % SQ_NB);
}

} // namespace YaneuraOu::Eval::NNUE::Features
#endif // defined(EVAL_NNUE) || defined(EVAL_SFNN)

#endif // NNUE_FEATURES_KING_PIECE_HELPERS_H_INCLUDED
