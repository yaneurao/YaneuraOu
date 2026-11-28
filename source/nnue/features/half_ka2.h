// NNUE評価関数の入力特徴量HalfKAの定義

#ifndef NNUE_FEATURES_HALF_KA2_H_INCLUDED
#define NNUE_FEATURES_HALF_KA2_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

	// 特徴量HalfKA2：視点側または相手側の玉と、手駒・両玉を含む全駒のBonaPieceの組み合わせ。両玉は同じ駒種の面にまとめる
	template <Side AssociatedKing>
	class HalfKA2 {
	public:
		// 特徴量名
		static constexpr const char* kName =
			(AssociatedKing == Side::kFriend) ? "HalfKA2(Friend)" : "HalfKA(Enemy)";
		// 評価関数ファイルに埋め込むハッシュ値
		static constexpr u32 HashValue =
			0x5f234cb9u ^ (AssociatedKing == Side::kFriend);
		// 特徴量の次元数
		static constexpr IndexType Dimensions =
			static_cast<IndexType>(SQ_NB) * static_cast<IndexType>(e_king);
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

        // 盤面と手駒からFinny更新用の特徴量番号を直接求める。
        struct IndexContext {
            IndexContext(Square king, Color c) : perspective(c), kingOffset(IndexType(e_king) * IndexType(king)) {}
            IndexType board(Piece pc, Square sq) const {
                const auto base = kpp_board_index[pc].from[perspective];
                const Square oriented = perspective == BLACK ? sq : Inv(sq);
                return kingOffset + (base >= e_king ? base - SQ_NB : base) + IndexType(oriented);
            }
            IndexType hand(Color c, PieceType pt, int slot) const {
                return kingOffset + kpp_hand_index[c][pt].from[perspective] + slot;
            }
            IndexType zero() const { return kingOffset; }
            Color perspective;
            IndexType kingOffset;
        };

	};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
