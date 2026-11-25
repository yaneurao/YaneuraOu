// NNUE評価関数の入力特徴量HalfKA_hmの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "half_ka_hm1.h"
#include "../../position.h"
#include "index_list.h"
#include "king_piece_helpers.h"

namespace YaneuraOu::Eval::NNUE::Features {

	// 玉の位置とBonaPieceから特徴量のインデックスを求める
	template <Side AssociatedKing>
	inline IndexType HalfKA_hm1<AssociatedKing>::make_index(Square sq_k, BonaPiece p) {
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
		// 両玉を区別して含める
		return static_cast<IndexType>(fe_end2) * static_cast<IndexType>(sq_k) + static_cast<IndexType>(p);
	}

	// 特徴量のうち、値が1であるインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA_hm1<AssociatedKing>::append_active_indices(
		const Position& pos, Color perspective, IndexList* active) {
		// コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
		if (RawFeatures::MaxActiveDimensions < MaxActiveDimensions) return;

		BonaPiece pieces[40];
		Square sq_target_k;
		collect_pieces_with_king<AssociatedKing>(pos, perspective, pieces, &sq_target_k);
		for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
			active->push_back(make_index(sq_target_k, pieces[i]));
		}
	}

	// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA_hm1<AssociatedKing>::append_changed_indices(
		const Position& pos, Color perspective,
		IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
		const Square sq_target_k = oriented_king_square<AssociatedKing>(pos, perspective);
		for (int i = 0; i < dp.count(); ++i) {
			const auto old_p = static_cast<BonaPiece>(
				dp.change(i).old_piece.from[perspective]);
			removed->push_back(make_index(sq_target_k, old_p));
			const auto new_p = static_cast<BonaPiece>(
				dp.change(i).new_piece.from[perspective]);
			added->push_back(make_index(sq_target_k, new_p));
		}
	}

	template class HalfKA_hm1<Side::kFriend>;
	template class HalfKA_hm1<Side::kEnemy>;

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
