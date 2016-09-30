#include "mate1ply.h"
#if defined(USE_MATE_1PLY)
#include "../position.h"

// 利きのある場所への取れない近接王手からの3手詰め
Move Position::weak_mate_n_ply(int ply) const
{
	// 1手詰めであるならこれを返す
	Move m = mate1ply();
	if (m)
		return m;

	// 詰まない
	if (ply <= 1)
		return MOVE_NONE;

	Color us = side_to_move();
	Color them = ~us;
	Bitboard around8 = kingEffect(king_square(them));

	// const剥がし
	Position* This = ((Position*)this);

	StateInfo si;
	StateInfo si2;

	// 近接王手で味方の利きがあり、敵の利きのない場所を探す。
	for (auto m : MoveList<CHECKS>(*this))
	{
		// 近接王手で、この指し手による駒の移動先に敵の駒がない。
		Square to = to_sq(m);
		if ((around8 & to)

#ifndef LONG_EFFECT_LIBRARY
			// toに利きがあるかどうか。mが移動の指し手の場合、mの元の利きを取り除く必要がある。
			&& (is_drop(m) ? effected_to(us, to) : (attackers_to(us, to, pieces() ^ from_sq(m)) ^ from_sq(m)))

			// 敵玉の利きは必ずtoにあるのでそれを除いた利きがあるかどうか。
			&& (attackers_to(them,to,pieces()) ^ king_square(them))
#else
			&& (is_drop(m) ? effected_to(us, to) :
					board_effect[us].effect(to) >= 2 ||
					(long_effect.directions_of(us, from_sq(m)) & Effect8::directions_of(from_sq(m), to)) != 0)

			// 敵玉の利きがあるので2つ以上なければそれで良い。
			&& (board_effect[them].effect(to) <= 1)
#endif
			)
		{
			if (!legal(m))
				continue;

			ASSERT_LV3(gives_check(m));

			This->do_move(m,si,true);

			ASSERT_LV3(in_check());

			// この局面ですべてのevasionを試す
			for (auto m2 : MoveList<EVASIONS>(*this))
			{
				if (!legal(m2))
					continue;

				// この指し手で逆王手になるなら、不詰めとして扱う
				if (gives_check(m2))
					goto NEXT_CHECK;

				This->do_move(m2, si2, false);

				ASSERT_LV3(!in_check());

				if (!weak_mate_n_ply(ply-2))
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


#endif // if defined(MATE_1PLY)...
