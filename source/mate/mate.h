#ifndef __MATE_H_INCLUDED__
#define __MATE_H_INCLUDED__

#include "../types.h"
#if defined (USE_MATE_1PLY)

#include <vector>

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

	// 利きのある場所への取れない近接王手からの3手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	Move weak_mate_3ply(const Position& pos, int ply);

	// Mate::MateRepetition() で千日手判定を行った時の返し値
	enum class MateRepetitionState
	{
		Mate   , // 詰み扱い
		Mated  , // 詰まされた扱い
		Unknown, // そのいずれでもない
	};

	// 連続王手などの千日手判定を行う。
	// plies_from_root : 詰み探索開始局面からの手数
	// repetition_ply  : 千日手判定を行う最大手数 (16ぐらいでいいと思う)
	// or_node         : 攻め方のnodeであればtrue。受け方のnodeであればfalse。
	MateRepetitionState mate_repetition(const Position& pos, const int plies_from_root , const int repetition_ply, bool or_node);

	// MateSolverやMateDfpnSolverで千日手判定のために遡る最大手数
	constexpr int MAX_REPETITION_PLY = 16;

	// n手詰め
#if defined(USE_MATE_SOLVER)

	// 詰み探索。内部的にメモリを確保しているわけではないので、
	// 詰み探索が必要な時に気軽にこのインスタンスを生成して問題ない。
	class MateSolver
	{
	public:
	// 奇数手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか
		Move mate_odd_ply(Position& pos, const int ply, bool gen_all);

	private:

	// mate_odd_ply()の王手がかかっているかをtemplateにしたやつ。
	// ※　dlshogiのmateMoveInOddPlyReturnMove()を参考にさせていただいています。
	// INCHECK : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか
		template <bool INCHECK, bool GEN_ALL>
	Move mate_odd_ply(Position& pos, const int ply);

	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか。
		Move mated_even_ply(Position& pos, const int ply, bool gen_all);

	// mated_even_ply()のtemplate版。
	// GEN_ALL : 歩の不成も生成するのか。
	template <bool GEN_ALL>
	Move mated_even_ply(Position& pos, const int ply);

	// 3手詰めチェック
	// 手番側が王手でないこと
	// mate_even_ply()から内部的に呼び出される。
	// INCHECK : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか。
		template <bool INCHECK, bool GEN_ALL>
	Move mate_3ply(Position& pos);

	private:
		// 探索開始時のgame_plyを保存しておく。
		// 千日手判定のためにこの局面以前に遡る時と、そうでない時とで処理が異なるので。
		int root_game_ply;
	};

#endif
}

#if defined(USE_MATE_DFPN)

namespace Mate::Dfpn
{
	template <typename NodeCountType>
	class DfpnImp;

	// df-pn詰将棋ルーチン
	// 事前にメモリ確保やら何やらしないといけないのでクラス化してある。
	// 古来からあるdf-pnとは異なる、新しいコンセプトによる実装。
	class MateDfpnSolver
	{
	public:
		// このクラスを用いるには、この関数を呼び出して事前にdf-pn用のメモリを確保する必要がある。
		// size_mb [MB] だけ探索用のメモリを確保する。
		void alloc(size_t size_mb);

		// 探索して初手を返す。
		// nodes_limit内に解ければその初手が返る。
		// 不詰が証明できれば、MOVE_NULL、解がわからなかった場合は、MOVE_NONEが返る。
		// nodes_limit : ノード制限。0を指定するとノード制限なし。(ただしメモリの制限から解けないことはある)
		Move mate_dfpn(Position& pos , u32 nodes_limit);

		// mate_dfpn()がMOVE_NULL,MOVE_NONE以外を返した場合にその手順を取得する。
		// ※　最短手順である保証はない。
		std::vector<Move> get_pv() const;

		// 解けた時に今回の探索ノード数を取得する。
		u32 get_node_searched() const;

		// 解けた時に今回の詰み手数を取得する。
		int get_mate_ply() const;

		// mate_dfpn()でMOVE_NONE以外が返ってきた時にメモリが不足しているかを返す。
		bool is_out_of_memory() const;

		MateDfpnSolver();

	private:
		// 実装本体。
		std::shared_ptr<DfpnImp<u32>> imp;
	};
} // namespace Mate::Dfpn
#endif


#endif // namespace Mate

#endif // __MATE_H_INCLUDED__

