// NNUE評価関数の入力特徴量HalfKA_hmの定義

#ifndef NNUE_FEATURES_HALF_KA_HM2_H_INCLUDED
#define NNUE_FEATURES_HALF_KA_HM2_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include <array>

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

	// 特徴量HalfKP_vm：自玉または敵玉の位置と、すべての駒の位置の組み合わせ
	// 6筋～9筋に玉がいる場合、4筋～1筋に反転させる
	template <Side AssociatedKing>
	class HalfKA_hm2 {
	public:
		// 特徴量名
		static constexpr const char* kName =
			(AssociatedKing == Side::kFriend) ? "HalfKA_hm2(Friend)" : "HalfKA_hm(Enemy)";
		// 評価関数ファイルに埋め込むハッシュ値
		static constexpr u32 HashValue =
			0x7f234cb9u ^ (AssociatedKing == Side::kFriend);
		// 特徴量の次元数
		static constexpr IndexType Dimensions =
			5 * static_cast<IndexType>(FILE_NB) * static_cast<IndexType>(e_king);
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

        // Map raw board/hand data directly, as Stockfish does in make_index().
        // Stockfishのmake_index()と同様に、盤面・手駒から入力番号を直接求める。
        struct IndexContext {
            static inline constexpr auto Squares = [] {
                std::array<std::array<Square, SQ_NB>, 4> table{};
                for (int c = 0; c < 2; ++c)
                    for (int mirror = 0; mirror < 2; ++mirror)
                        for (int sq = 0; sq < SQ_NB; ++sq) {
                            const Square s = c == BLACK ? Square(sq) : Inv(Square(sq));
                            table[2 * c + mirror][sq] = mirror ? Mir(s) : s;
                        }
                return table;
            }();

            IndexContext(Square king, Color c):
              perspective(c), kingOffset(IndexType(e_king) * IndexType(king >= SQ_61 ? Mir(king) : king)),
              squares(Squares[2 * c + (king >= SQ_61)].data()) {}

            IndexType board(Piece pc, Square sq) const {
                const auto base = kpp_board_index[pc].from[perspective];
                return kingOffset + (base >= e_king ? base - SQ_NB : base) + IndexType(squares[sq]);
            }
            IndexType hand(Color c, PieceType pt, int slot) const {
                return kingOffset + kpp_hand_index[c][pt].from[perspective] + slot;
            }
            IndexType zero() const { return kingOffset; }

            Color perspective;
            IndexType kingOffset;
            const Square* squares;
        };
	};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
