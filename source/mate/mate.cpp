#include "mate.h"
#if defined (USE_MATE_1PLY)
#include "../position.h"

namespace Mate
{
	// 1手詰めテーブルの初期化関数
	// ※　これは、mate1ply_without_effect.cppか、mate1ply_with_effect.cppのいずれかで定義されている。
	extern void init_mate1ply();

	// Mate関連で使うテーブルの初期化
	void init()
	{
		init_mate1ply();
	}

	Move mate1ply(const Position& pos)
	{
		return pos.side_to_move() == BLACK ? Mate::is_mate_in_1ply_imp<BLACK>(pos) : Mate::is_mate_in_1ply_imp<WHITE>(pos);
	}

}

#endif // defined (USE_MATE_1PLY)
