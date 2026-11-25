// NNUE評価関数の入力特徴量HalfKA_hmの定義

#ifndef NNUE_FEATURES_HALF_KA_HM1_H_INCLUDED
#define NNUE_FEATURES_HALF_KA_HM1_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

	// 特徴量HalfKA_hm1：視点側または相手側の玉と、手駒・両玉を含む全駒のBonaPieceの組み合わせ
	// 6筋～9筋に玉がいる場合、4筋～1筋に反転させる
	template <Side AssociatedKing>
	class HalfKA_hm1 {
	public:
		// 特徴量名
		static constexpr const char* kName =
			(AssociatedKing == Side::kFriend) ? "HalfKA_hm1(Friend)" : "HalfKA_hm(Enemy)";
		// 評価関数ファイルに埋め込むハッシュ値
		static constexpr u32 HashValue =
			0x7f134cb9u ^ (AssociatedKing == Side::kFriend);
		// 特徴量の次元数
		// 両玉を区別して含める
		static constexpr IndexType Dimensions =
			5 * static_cast<IndexType>(FILE_NB) * static_cast<IndexType>(fe_end2);
		// 特徴量のうち、同時に値が1となるインデックスの数の最大値
		static constexpr IndexType MaxActiveDimensions = PIECE_NUMBER_NB;
		// 差分計算の代わりに全計算を行うタイミング
		static constexpr TriggerEvent kRefreshTrigger =
			(AssociatedKing == Side::kFriend) ?
			TriggerEvent::kFriendKingMoved : TriggerEvent::kEnemyKingMoved;

		// 特徴量のうち、値が1であるインデックスのリストを取得する
		static void append_active_indices(const Position& pos, Color perspective,
			IndexList* active);

		// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
		static void append_changed_indices(const Position& pos, Color perspective,
			IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp);

		// 玉の位置とBonaPieceから特徴量のインデックスを求める
		static IndexType make_index(Square sq_k, BonaPiece p);

	};

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
