// NNUE評価関数の入力特徴量HalfKP_vmの定義

#ifndef CLASSIC_NNUE_FEATURES_HALF_KP_VM_H
#define CLASSIC_NNUE_FEATURES_HALF_KP_VM_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量HalfKP_vm：自玉または敵玉の位置と、玉以外の駒の位置の組み合わせ
// 6筋～9筋に玉がいる場合、4筋～1筋に反転させる
template <Side AssociatedKing>
class HalfKP_vm {
public:
	// 特徴量名
	static constexpr const char* kName =
		(AssociatedKing == Side::kFriend) ? "HalfKP_vm(Friend)" : "HalfKP_vm(Enemy)";
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr std::uint32_t kHashValue =
		0x0B6B1D9Bu ^ (AssociatedKing == Side::kFriend);
	// 特徴量の次元数
	static constexpr IndexType kDimensions =
		5 * static_cast<IndexType>(FILE_NB) * static_cast<IndexType>(fe_end);
	// 特徴量のうち、同時に値が1となるインデックスの数の最大値
	static constexpr IndexType kMaxActiveDimensions = PIECE_NUMBER_KING;
	// 差分計算の代わりに全計算を行うタイミング
	static constexpr TriggerEvent kRefreshTrigger =
		(AssociatedKing == Side::kFriend) ?
		TriggerEvent::kFriendKingMoved : TriggerEvent::kEnemyKingMoved;

	// 特徴量のうち、値が1であるインデックスのリストを取得する
	static void AppendActiveIndices(const Position& pos, Color perspective,
		IndexList* active);

	// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
	static void AppendChangedIndices(const Position& pos, Color perspective,
		IndexList* removed, IndexList* added);

	// 玉の位置とBonaPieceから特徴量のインデックスを求める
	static IndexType MakeIndex(Square sq_k, BonaPiece p);

private:
	// 駒の情報を取得する
	static void GetPieces(const Position& pos, Color perspective,
		BonaPiece** pieces, Square* sq_target_k);
};

} // namespace Eval::NNUE::Features
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
