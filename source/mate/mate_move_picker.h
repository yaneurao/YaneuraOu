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
#if !defined (USE_PIECE_VALUE)
				std::cout << "Error! : define USE_PIECE_VALUE" << std::endl;
#endif

				// 現在の駒割りに、captureする指し手に対してcaptureする駒を加点して、
				// あと、駒の移動先の升が受け方の玉から近いかに対して加点して、その指し手の価値として返す。
				// 大雑把なオーダリングだが、異なる深さのleaf nodeの有望さの比較がそれによってきちんとできるなら
				// わりとアリなのでは…。

				// 先手から見た駒割
				int value = pos.state()->materialValue;

				// いまの手番側
				auto us = pos.side_to_move();

				// and node側の玉の手番
				Color king_color = ((us == BLACK) ^ or_node) ? BLACK : WHITE;
				auto ksq = pos.king_square(king_color);

				// 玉は下段のほうが詰ませやすいはずなので下段の玉に対して加点する。
				// ※　詰将棋だと ksq = SQ_NBもありうる
				constexpr int king_rank_bonus[COLOR_NB][10] = {
					{ -600,-300,-200,  0,  0,  0,   0,  50, 100,0 },
					{  100,  50,   0,  0,  0,  0,-200,-300,-600,0 }
				};
				// and node側の玉の段に対するbonus。(or nodeから見て)
				int king_bonus = king_rank_bonus[king_color][rank_of(ksq)];

#if 0
				// 手駒の枚数に対して加点。(詰将棋において手駒は多いほうが圧倒的に詰ませやすい)
				// 歩 = 1点。香2点、桂2点、銀3点、金4点、角3点、飛5点みたいな。
				// TODO : 攻め方と受け方で点数変えるべきか？

				constexpr int hand_value[8] = { 0,1,2,2,5,4,4,6 };

				auto our_hand = pos.hand_of(us);
				int our_hand_value =
					hand_count(our_hand, PAWN) * hand_value[1]
					+ hand_count(our_hand, LANCE) * hand_value[2]
					+ hand_count(our_hand, KNIGHT) * hand_value[3]
					+ hand_count(our_hand, SILVER) * hand_value[4]
					+ hand_count(our_hand, BISHOP) * hand_value[5]
					+ hand_count(our_hand, ROOK) * hand_value[6]
					+ hand_count(our_hand, GOLD) * hand_value[7];

				auto their_hand = pos.hand_of(~us);
				int there_hand_value =
					hand_count(their_hand, PAWN) * hand_value[1]
					+ hand_count(their_hand, LANCE) * hand_value[2]
					+ hand_count(their_hand, KNIGHT) * hand_value[3]
					+ hand_count(their_hand, SILVER) * hand_value[4]
					+ hand_count(their_hand, BISHOP) * hand_value[5]
					+ hand_count(their_hand, ROOK) * hand_value[6]
					+ hand_count(their_hand, GOLD) * hand_value[7];
#endif
				// →　良くないようであった。

				// 最後に調べたtoとそこの利きの数
				Square lastTo = SQ_NB;
				// 味方の利きの数、敵の利きの数
				// ※ 未初期化かもしれないという警告がでて気持ち悪いのでゼロ初期化している。
				int our_effect = 0 , their_effect = 0;

				for (auto& m : *this)
				{
					Move move = m.move;

					// この指し手によって捕獲する駒
					auto to = to_sq(move);
					auto cap = pos.piece_on(to);
					// v : この指し手を指したあとの先手から見た駒割
					int v = value + (cap == NO_PIECE ? 0 : - Eval::PieceValue[cap]*2);

					auto pc = pos.moved_piece_after(move);

					// 成れるなら、成る価値を加算
					if (is_promote(move))
						v += Eval::PieceValue[pc] - Eval::PieceValue[raw_of(pc)];

#if 1
					// 利きの考慮)
					// 　7手以上の時はこれを入れたほうが少ないノード数で解けるのだが、
					// 　利きの数を調べることでnpsが14%ぐらいダウンするので、node数を10%増やすほうがaccuracy上がるから、
					// 　結局、効果がないに等しい。
					// 　利きの数、もう少し速く求められると良いのだが…。
					// 　※　toが同じなら1回で済むはずだし…。何かと無駄が多い。

					// 駒の移動先の利き、相手のほうが数が多いなら…。
					// 直後に取り返されそう..

					// 駒打ちか？
					bool drop = is_drop(move);

					if (lastTo != to)
					{
						// 利きの数を調べるの、わりと重いのでcacheしておく。
						lastTo = to;

						// 味方の利きの数、敵の利きの数
						our_effect = pos.attackers_to(us, to).pop_count();
						their_effect = pos.attackers_to(~us, to).pop_count();
					}

					if (  ( drop && our_effect     < their_effect)
						||(!drop && our_effect - 1 < their_effect))
						// 駒の移動であるから、移動させる駒が利いていたということであり、利きを -1 して考える必要がある。
					{
						// 移動後の駒(先後の区別あり)
						// 駒打ちの場合も打ったあとの駒
						v -= Eval::PieceValue[pc] + Eval::PieceValue[raw_of(pc)]; // これは直後に取られるであろうから減点。
					}
#endif

					// これをor node側から見た駒割りに変換する。
					// or node かつ 先手 →　このまま
					// or node かつ 後手 →  符号反転
					// and node かつ 先手 →　符号反転
					// and node かつ 後手 →　このまま
					// つまりは、 NOT (or_node XOR 手番==先手)なら符号反転
					if (or_node ^ (us == BLACK))
						v = -v;

					// or_nodeでは、駒の移動先が相手の王様から近いことに対して加点する。
					// (遠くから打つ手を優先されるとかなわんため)
					// and_nodeでは、駒打ち(合駒)などが王様から近いほうが嬉しいが、
					// これをor_nodeから見たスコアにするためには、遠いことに対して加点しないといけない

					if (or_node)
						v += -dist(ksq, to);
					else
						v += +dist(ksq, to);

					//v += e;

					v += king_bonus;

					m.value = v;
				}

#if 0
				std::cout << pos << std::endl;
				for (auto& m : *this)
				{
					std::cout << m.move << " , " << m.value << std::endl;
				}
#endif
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
