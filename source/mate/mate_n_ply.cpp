#include "mate.h"
#if defined(USE_MATE_1PLY)
#include "../position.h"

namespace Mate
{
	// 利きのある場所への取れない近接王手からの3手詰め
	Move weak_mate_n_ply(const Position& pos, int ply)
	{
		// 1手詰めであるならこれを返す
		Move m = Mate::mate1ply(pos);
		if (m)
			return m;

		// 詰まない
		if (ply <= 1)
			return MOVE_NONE;

		Color us = pos.side_to_move();
		Color them = ~us;
		Bitboard around8 = kingEffect(pos.king_square(them));

		// const剥がし
		Position* This = ((Position*)&pos);

		StateInfo si;
		StateInfo si2;

		// 近接王手で味方の利きがあり、敵の利きのない場所を探す。
		for (auto m : MoveList<CHECKS>(pos))
		{
			// 近接王手で、この指し手による駒の移動先に敵の駒がない。
			Square to = to_sq(m);
			if ((around8 & to)

#if ! defined(LONG_EFFECT_LIBRARY)
				// toに利きがあるかどうか。mが移動の指し手の場合、mの元の利きを取り除く必要がある。
				&& (is_drop(m) ? pos.effected_to(us, to) : (pos.attackers_to(us, to, pos.pieces() ^ from_sq(m)) ^ from_sq(m)))

				// 敵玉の利きは必ずtoにあるのでそれを除いた利きがあるかどうか。
				&& (pos.attackers_to(them, to, pos.pieces()) ^ pos.king_square(them))
#else
				&& (is_drop(m) ? pos.effected_to(us, to) :
					pos.board_effect[us].effect(to) >= 2 ||
					(pos.long_effect.directions_of(us, from_sq(m)) & Effect8::directions_of(from_sq(m), to)) != 0)

				// 敵玉の利きがあるので2つ以上なければそれで良い。
				&& (pos.board_effect[them].effect(to) <= 1)
#endif
				)
			{
				if (!pos.legal(m))
					continue;

				ASSERT_LV3(pos.gives_check(m));

				This->do_move(m, si, true);

				ASSERT_LV3(pos.in_check());

				// この局面ですべてのevasionを試す
				for (auto m2 : MoveList<EVASIONS>(pos))
				{
					if (!pos.legal(m2))
						continue;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (pos.gives_check(m2))
						goto NEXT_CHECK;

					This->do_move(m2, si2, false);

					ASSERT_LV3(!pos.in_check());

					if (!weak_mate_n_ply(pos , ply - 2))
					{
						// 詰んでないので、m2で詰みを逃れている。
						This->undo_move(m2);
						goto NEXT_CHECK;
					}

					This->undo_move(m2);
				}

				// すべて詰んだ
				This->undo_move(m);

				// mによって3手で詰む。
				return m;

			NEXT_CHECK:;
				This->undo_move(m);
			}
		}
		return MOVE_NONE;
	}
} // namespace Mate

#endif // if defined(MATE_1PLY)...
