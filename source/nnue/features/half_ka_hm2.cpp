// NNUE評価関数の入力特徴量HalfKA_hmの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "half_ka_hm2.h"
#include "index_list.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

	// 玉の位置とBonaPieceから特徴量のインデックスを求める
	template <Side AssociatedKing>
	inline IndexType HalfKA_hm2<AssociatedKing>::MakeIndex(Square sq_k, BonaPiece p) {
		if (sq_k >= SQ_61) {
			// 玉が6筋～9筋にいる場合、4筋～1筋に反転する。
			sq_k = Mir(sq_k);

			if (p >= fe_hand_end) {
				// 持駒は反転しない。
				IndexType piece_index = (p - fe_hand_end) / SQ_NB;
				Square sq_p = static_cast<Square>((p - fe_hand_end) % SQ_NB);
				sq_p = Mir(sq_p);
				p = static_cast<BonaPiece>(fe_hand_end + piece_index * static_cast<IndexType>(SQ_NB) + sq_p);
			}
		}
		// 後手玉は自玉と同じPLANEに持っていく
		return static_cast<IndexType>(e_king) * static_cast<IndexType>(sq_k) + static_cast<IndexType>(p >= e_king ? p - SQ_NB : p);
	}

	// 特徴量のうち、値が1であるインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA_hm2<AssociatedKing>::AppendActiveIndices(
		const Position& pos, Color perspective, IndexList* active) {
		// コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
		if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

        const Color target = AssociatedKing == Side::kFriend ? perspective : ~perspective;
        const Square king = perspective == BLACK ? pos.square<KING>(target)
                                                  : Inv(pos.square<KING>(target));
        const IndexContext indices(king, perspective);
        int count = 0;
        Bitboard occupied = pos.pieces();
        while (occupied) {
            const Square sq = occupied.pop_lsb();
            active->push_back(indices.board(pos.piece_on(sq), sq));
            ++count;
        }
        for (Color c : {BLACK, WHITE}) {
            const Hand hand = pos.hand_of(c);
            if (hand == Hand(0)) continue;
            for (PieceType pt = PAWN; pt < KING; ++pt) {
                const int n = hand_count(hand, pt);
                for (int i = 0; i < n; ++i)
                    active->push_back(indices.hand(c, pt, i));
                count += n;
            }
        }
        for (; count < 40; ++count)
            active->push_back(indices.zero());
	}

	// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA_hm2<AssociatedKing>::AppendChangedIndices(
		const Position& pos, Color perspective,
		IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
        if (dp.count() == 0) return;
        const Color target = AssociatedKing == Side::kFriend ? perspective : ~perspective;
        const Square king = perspective == BLACK ? pos.square<KING>(target)
                                                  : Inv(pos.square<KING>(target));
        const IndexContext indices(king, perspective);
        if (dp.from == SQ_NONE)
            removed->push_back(indices.hand(color_of(dp.pc), raw_type_of(dp.pc), dp.handSlot));
        else
            removed->push_back(indices.board(dp.pc, dp.from));
        added->push_back(indices.board(dp.after, dp.to));
        if (dp.captured != NO_PIECE) {
            removed->push_back(indices.board(dp.captured, dp.to));
            added->push_back(indices.hand(color_of(dp.pc), raw_type_of(dp.captured), dp.handSlot));
        }
	}

	template class HalfKA_hm2<Side::kFriend>;
	template class HalfKA_hm2<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
