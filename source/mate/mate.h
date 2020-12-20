#ifndef __MATE_H_INCLUDED__
#define __MATE_H_INCLUDED__

#include "../types.h"
#if defined (USE_MATE_1PLY)

namespace Mate
{
	// Mate関連で使うテーブルの初期化
	// ※　Bitboard::init()から呼び出される。
	void init();

	// 現局面で1手詰めであるかを判定する。1手詰めであればその指し手を返す。
	// ただし1手詰めであれば確実に詰ませられるわけではなく、簡単に判定できそうな近接王手による
	// 1手詰めのみを判定する。(要するに判定に漏れがある。)
	Move mate1ply(const Position& pos);

	// 1手で詰むならばその指し手を返す。なければMOVE_NONEを返す
	template <Color Us>
	Move is_mate_in_1ply_imp(const Position& pos);

	// これ↓あとで消すかも。

	// 利きのある場所への取れない近接王手からの3手詰め
	Move weak_mate_n_ply(const Position& pos,int ply);
}

#endif // namespace Mate

#endif // __MATE_H_INCLUDED__

