#include "mate.h"
#if defined(USE_MATE_SOLVER)
#include "../position.h"
#include "mate_move_picker.h"

// ---------------------
// mate_odd_ply()
// ---------------------

// dlshogiのN手詰めルーチン

namespace Mate {

	// 3手詰めチェック
	// mated_even_ply()から内部的に呼び出される。
	template <bool INCHECK , bool GEN_ALL>
	FORCE_INLINE Move MateSolver::mate_3ply(Position& pos)
	{
		// OR節点

		StateInfo si;
		StateInfo si2;

		for (const auto& ml : MovePicker<true, INCHECK , GEN_ALL , false /* no ordering */>(pos))
		{
			auto m = ml.move;

			pos.do_move(m, si, true);

			// and node

			// この局面で詰まされる指し手を発見したか？(この関数の開始局面から見ると、指してmで詰む)
			bool found_mate = false;

			// 詰み探索開始局面までの手数で千日手を調べる。
			auto rep = mate_repetition(pos, pos.game_ply() - root_game_ply , MAX_REPETITION_PLY , false/* and_node*/);

			switch(rep)
			{
			case MateRepetitionState::Mated:
				found_mate = true;
				break;

			case MateRepetitionState::Mate:
				// 詰めたので、or nodeの指し手としては不詰扱い
				break;

			case MateRepetitionState::Unknown:
				// いずれでもないので、きちんと調べる必要がある。

				// この局面ですべてのevasion(王手回避手)を試す
				MovePicker<false, false, GEN_ALL, false /* no ordering */> move_picker2(pos);

				if (move_picker2.size() == 0) {
					// 回避手がない = 指し手mで詰んだ。
					found_mate = true;
				}
				else {
					for (const auto& ml2 : move_picker2)
					{
						auto m2 = ml2.move;

						// この指し手で逆王手になるなら、不詰めとして扱う
						// (我々は王手がかかっている局面で呼び出せる1手詰めルーチンを持っていないので)
						if (pos.gives_check(m2))
							goto NEXT_CHECK;

						pos.do_move(m2, si2, /* givesCheck */false);

						// mate_1ply()は王手がかかっている時に呼べないが、
						// この局面は王手はかかっていないから問題ない。

						if (!Mate::mate_1ply(pos)) {
							// 詰んでないので、m2で詰みを逃れている。
							pos.undo_move(m2);
							goto NEXT_CHECK;
						}

						pos.undo_move(m2);
					}
					// 逃れる指し手、すべてで詰んだので、指し手mで詰む。
					found_mate = true;
				}
				break;

			}
			NEXT_CHECK:;
			pos.undo_move(m);

			if (found_mate)
				return m;
		}
		return MOVE_NONE;
	}

	// mate_odd_ply()の王手がかかっているかをtemplateにしたやつ。
	// ※　dlshogiのmateMoveInOddPlyReturnMove()を参考にさせていただいています。
	// InCheck : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか
	template <bool INCHECK , bool GEN_ALL>
	Move MateSolver::mate_odd_ply(Position& pos, const int ply)
	{
		if (ply == 3)
			return mate_3ply<INCHECK,GEN_ALL>(pos);
		else if (ply == 1)
			// 王手がかかっていないなら1手詰めを呼び出せるが、王手がかかっているなら1手詰めを呼べないので
			// evasionのなかから詰む指し手を探す必要がある。レアケースなので、ここでは不詰み扱いをしておく。
			return !pos.in_check() ? mate_1ply(pos) : MOVE_NONE;

		// OR接点なので一つでも詰みを見つけたらそれで良し。

		// すべての合法手について
		for (const auto& ml : MovePicker<true, INCHECK , GEN_ALL , false /* no ordering */>(pos)) {

			// MovePickerの指し手で1手進める。
			// これが合法手であることは、MovePickerのほうで保証されている。
			auto m = ml.move;

			StateInfo state;
			// これが王手であることはわかっているので第3引数はtrueで固定しておく。
			pos.do_move(m, state, true);

			// and node

			// or接点から見た詰みを見つけたかのフラグ
			bool found_mate = false;

			// 詰み探索開始局面までの手数で千日手を調べる。
			auto rep = mate_repetition(pos, pos.game_ply() - root_game_ply , MAX_REPETITION_PLY , false/* and_node*/);

			switch (rep)
			{
			case MateRepetitionState::Mated:
				found_mate = true;
				break;

			case MateRepetitionState::Mate:
				// 詰めたので、or nodeの指し手としては不詰扱い
				break;

			case MateRepetitionState::Unknown:
				// いずれでもないので、きちんと調べる必要がある。
				// さらにこの局面から偶数手で相手が詰まされるかのチェック
				found_mate = mated_even_ply<GEN_ALL>(pos, ply - 1) == MOVE_NONE /* 回避手がない == 詰み */;
				break;

			default: UNREACHABLE;
			}

			pos.undo_move(m);

			// 詰みを見つけたならその最初の手を返す。
			if (found_mate)
				return m;
		}

		// 詰みを発見できなかった。
		return MOVE_NONE;
	}
		
	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// GEN_ALL : 歩の不成も生成するのか。
	template <bool GEN_ALL>
	Move MateSolver::mated_even_ply(Position& pos, const int ply)
	{
		// AND節点なのでこの局面のすべての指し手に対して(手番側が)詰まなければならない。
		// 一つでも詰まない手があるならfalseを返す。

		// すべてのEvasion(王手回避の指し手)について
		for (const auto& ml : MovePicker<false, false , GEN_ALL , false /* no ordering */>(pos))
		{
			//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
			auto m = ml.move;

			// この指し手で王手になるのか
			const bool givesCheck = pos.gives_check(m);

			// 1手動かす
			StateInfo state;
			pos.do_move(m, state, givesCheck);

			// or node

			// この関数の呼び出し時の手番側が逃れる指し手を見つけたか。
			// これがtrueになった時は、その指し手で逃れているので、この関数はfalseを返す。
			bool found_escape = false;

			// 詰み探索開始局面までの手数で千日手を調べる。
			auto rep = mate_repetition(pos, pos.game_ply() - root_game_ply , MAX_REPETITION_PLY , true/* or_node*/);

			switch (rep)
			{
			case MateRepetitionState::Mated:
				// 詰まされる手が見つからなかった(逃れている)時点で終了
				found_escape = true;
				break;

			case MateRepetitionState::Mate:
				// この関数に与えられた局面から見て相手が勝ち = 詰みを逃れていない
				break;

			case MateRepetitionState::Unknown:
				// いずれでもないので、きちんと調べる必要がある。
				if (ply == 4)
					// 3手詰めかどうか
					found_escape = !(givesCheck ? mate_3ply<true, GEN_ALL>(pos) : mate_3ply<false, GEN_ALL>(pos));
				//	// 詰みが見つからなかったら、逃れている。
				else
					// 奇数手詰めかどうか
					found_escape = !(givesCheck ? mate_odd_ply<true , GEN_ALL>(pos, ply - 1) : mate_odd_ply<false , GEN_ALL>(pos, ply - 1));
					// 詰みが見つからなかったら、逃れている。
				break;

			default: UNREACHABLE;
			}

			pos.undo_move(m);

			// 1手でも逃れる指し手を見つけた場合は、与えられた局面で詰まされない。
			if (found_escape)
				return m; // この指し手で逃れている。
		}

		// 詰みを逃れる指し手を見つけられなかった = 詰み
		return MOVE_NONE;
	}

	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか。
	Move MateSolver::mated_even_ply(Position& pos, const int ply, bool gen_all)
	{
		return gen_all ? mated_even_ply<true>(pos, ply) : mated_even_ply<false>(pos, ply);
	}


	// 奇数手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// GEN_ALL : 歩の不成も生成するのか
	Move MateSolver::mate_odd_ply(Position& pos, const int depth , bool GEN_ALL)
	{
		// 開始局面でのgame_ply()を保存しておかないと、千日手の判定の時に困る。
		root_game_ply = pos.game_ply();

		// 2×2の四通りのtemplateを呼び分ける。
		return GEN_ALL ? (pos.in_check() ? mate_odd_ply<true,true >(pos, depth) : mate_odd_ply<false,true >(pos, depth))
		               : (pos.in_check() ? mate_odd_ply<true,false>(pos, depth) : mate_odd_ply<false,false>(pos, depth));
	}


} // namespace Mate

#endif // defined(USE_MATE_SOLVER)
