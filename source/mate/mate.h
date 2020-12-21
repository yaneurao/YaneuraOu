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
	// 詰みがある場合は、その指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// 王手がかかっている局面で呼び出してはならない。(この関数は、王手を回避しながらの詰みを探すわけではない)
	Move mate_1ply(const Position& pos);

	// mate_1ply()の手番をtemplateにしたやつ。
	template <Color Us>
	Move mate_1ply_imp(const Position& pos);

	// n手詰め
#if defined(USE_MATE_N_PLY)

	// 利きのある場所への取れない近接王手からの3手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	Move weak_mate_3ply(const Position& pos,int ply);

	// 奇数手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか
	Move mate_odd_ply(Position& pos, const int ply , bool gen_all);

	// mate_odd_ply()の王手がかかっているかをtemplateにしたやつ。
	// ※　dlshogiのmateMoveInOddPlyReturnMove()を参考にさせていただいています。
	// INCHECK : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか
	template <bool INCHECK,bool GEN_ALL>
	Move mate_odd_ply(Position& pos, const int ply);

	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか。
	Move mated_even_ply(Position& pos, const int ply , bool gen_all);

	// mated_even_ply()のtemplate版。
	// GEN_ALL : 歩の不成も生成するのか。
	template <bool GEN_ALL>
	Move mated_even_ply(Position& pos, const int ply);

	// 3手詰めチェック
	// 手番側が王手でないこと
	// mate_even_ply()から内部的に呼び出される。
	// INCHECK : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか。
	template <bool INCHECK , bool GEN_ALL>
	Move mate_3ply(Position& pos);

#endif

}

#endif // namespace Mate

#endif // __MATE_H_INCLUDED__

