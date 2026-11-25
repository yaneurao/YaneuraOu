// NNUE評価関数の入力特徴量HalfKAの定義

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "half_ka2.h"
#include "../../position.h"
#include "index_list.h"
#include "king_piece_helpers.h"

namespace YaneuraOu::Eval::NNUE::Features {

	// 玉の位置とBonaPieceから特徴量のインデックスを求める
	template <Side AssociatedKing>
	inline IndexType HalfKA2<AssociatedKing>::make_index(Square sq_k, BonaPiece p) {
		// 後手玉は自玉と同じPLANEに持っていく
		return static_cast<IndexType>(e_king) * static_cast<IndexType>(sq_k) + static_cast<IndexType>(p >= e_king ? p - SQ_NB : p);
	}

	// 特徴量のうち、値が1であるインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA2<AssociatedKing>::append_active_indices(
		const Position& pos, Color perspective, IndexList* active) {
		append_halfka_active_indices<HalfKA2, AssociatedKing>(pos, perspective, active);
	}

	// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
	template <Side AssociatedKing>
	void HalfKA2<AssociatedKing>::append_changed_indices(
		const Position& pos, Color perspective,
		IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp) {
		const Square sq_target_k = oriented_king_square<AssociatedKing>(pos, perspective);
		append_halfka_changed_indices<HalfKA2>(sq_target_k, perspective, removed, added, dp);
	}

	template class HalfKA2<Side::kFriend>;
	template class HalfKA2<Side::kEnemy>;

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
