#include "mate.h"
#if defined(USE_MATE_DFPN)

	// 開発中

	// ===================================
	//       class MateDfpnSolver
	// ===================================

	// このクラスを用いるには、この関数を呼び出して事前にDfpn用のメモリを確保する必要がある。
	// size_mb [MB] だけ探索用のメモリを確保する。
	void MateDfpnSolver::alloc(size_t size_mb) {  }

	// 探索して初手を返す。
	// nodes_limit内に解ければその初手が返る。
	// 不詰が証明できれば、MOVE_NULL、解がわからなかった場合は、MOVE_NONEが返る。
	// nodes_limit : ノード制限。0を指定するとノード制限なし。(ただしメモリの制限から解けないことはある)
	Move MateDfpnSolver::mate_dfpn(Position& pos, u32 nodes_limit) { return MOVE_NONE; }

	// mate_dfpn()がMOVE_NULL,MOVE_NONE以外を返した場合にその手順を取得する。
	// ※　最短手順である保証はない。
	std::vector<Move> MateDfpnSolver::get_pv() const { return std::vector<Move>(); }

	// 解けた時に今回の探索ノード数を取得する。
	u32 MateDfpnSolver::get_node_searched() const { return (u32)0; }

	// 解けた時に今回の詰み手数を取得する。
	int MateDfpnSolver::get_mate_ply() const { return 0; }

	// mate_dfpn()でMOVE_NONE以外が返ってきた時にメモリが不足しているかを返す。
	bool MateDfpnSolver::is_out_of_memory() const { return false; }


	MateDfpnSolver::MateDfpnSolver() {}
			

} // namespace Mate::DFPN


#endif // defined(USE_DFPN)
