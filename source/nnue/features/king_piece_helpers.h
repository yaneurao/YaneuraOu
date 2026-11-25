// Shared piece collection and feature-coordinate helpers for shogi HalfKP/HalfKA.
// 将棋のHalfKP・HalfKA系で共用する駒情報収集・特徴量座標変換の補助処理。

#ifndef NNUE_FEATURES_KING_PIECE_HELPERS_H_INCLUDED
#define NNUE_FEATURES_KING_PIECE_HELPERS_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE) || defined(EVAL_SFNN)
#include "../../evaluate.h"
#include "../../position.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

// Mirror a board piece for a horizontally mirrored king; hand pieces are unchanged.
// 左右反転した玉に合わせて盤上の駒を反転する。持駒は反転しない。
inline BonaPiece mirror_piece(BonaPiece p) {
    if (p >= fe_hand_end) {
        IndexType piece_index = (p - fe_hand_end) / SQ_NB;
        Square sq_p = static_cast<Square>((p - fe_hand_end) % SQ_NB);
        sq_p = Mir(sq_p);
        p = static_cast<BonaPiece>(fe_hand_end + piece_index * static_cast<IndexType>(SQ_NB) + sq_p);
    }
    return p;
}

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
