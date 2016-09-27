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
		if ((around8 & to) && effected_to(us, to) && !effected_to(them, to) )
		{
			This->do_move(m,si,true);

			// この局面ですべてのevasionを試す
			for (auto m2 : MoveList<EVASIONS>(*this))
			{
				This->do_move(m2, si2);

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
