#ifndef __MATE_MOVE_PICKER_H_INCLUDED__
#define __MATE_MOVE_PICKER_H_INCLUDED__

#include "../config.h"

#if defined(USE_MATE_SOLVER) || defined(USE_MATE_DFPN)

#include "../position.h"
#include "../evaluate.h" // CapturePieceValue

namespace Mate
{
	// ===================================
	//   MovePickerあとで改良する。
	// ===================================

	// 詰将棋Solverで使うMovePicker

	// 王手の可能な組み合わせ。91通り？
	// https://tadaoyamaoka.hatenablog.com/entry/2018/06/03/225012
	// 自信がないので少し多めに確保する。
	const constexpr size_t MaxCheckMoves = 100;

	// n手詰み探索用のMovePicker
	// MateSolver , DfpnSolverで用いる。
	//
	// or_node       : どれか一つでも詰みがあれば良いノードであるか。(詰みまで奇数手であればそう)
	// INCHECK       : 王手がかかっているか
	// GEN_ALL       : 歩の不成なども生成するのか
	// MOVE_ORDERING : 指し手の並び替え or 点数付け等を行うのか。
	template <bool or_node, bool INCHECK , bool GEN_ALL , bool MoveOrdering>
	class MovePicker {
	public:
		explicit MovePicker(const Position& pos)
		{

			// givesCheckを呼び出すのかのフラグ
			bool doGivesCheck = false;
			if (or_node) {
				// ORノードなのですべての王手の指し手を生成。

				// dlshogi、ここ、王手になる指し手を生成して、自玉に王手がかかっている場合、
				// 回避手になっているかをpseudo_legal()でチェックしているが、
				// pseudo_legal()のほうはわりと重い処理なので、自玉に王手がかかっているなら
				// evasionの指し手を生成して、それがgives_check()で王手の指し手になっているか
				// 見たほうが自然では？

				if (INCHECK)
				{
					last = GEN_ALL ? generateMoves<EVASIONS_ALL>(pos, moveList) : generateMoves<EVASIONS>(pos, moveList);
					// これが王手になるかはのちほどチェックする。
					doGivesCheck = true;
				}
				else {
					last = GEN_ALL ? generateMoves<CHECKS_ALL>(pos, moveList) : generateMoves<CHECKS>(pos, moveList);
				}
			}
			else {
				// ANDノードなので回避の指し手のみを生成
				// (王手になる指し手も含まれる)
				last = GEN_ALL ? generateMoves<EVASIONS_ALL>(pos, moveList): generateMoves<EVASIONS>(pos, moveList);
			}

			// 以下の２つの指し手は除外する。
			// 1. doGivesCheck==trueなのに、王手になる指し手ではない。
			// 2. legalではない。
			last = std::remove_if(moveList, last, [&](const auto& ml) {
				return (doGivesCheck && !pos.gives_check(ml.move)) || !pos.legal(ml.move);
			});

			// それぞれの指し手に対して点数をつける。
			if (MoveOrdering)
			{
				// TODO : あとで
			}

			ASSERT_LV3(size() <= MaxCheckMoves);
		}

		size_t size() const { return static_cast<size_t>(last - moveList); }
		ExtMove* begin() { return &moveList[0]; }
		ExtMove* end() { return last; }
		bool empty() const { return size() == 0; }

	private:
		ExtMove moveList[MaxCheckMoves];
		ExtMove* last;
	};
}
#endif


#endif // ndef __MATE_MOVE_PICKER_H_INCLUDED__
