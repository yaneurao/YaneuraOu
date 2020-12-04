#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD)

#include "book.h"
#include "../position.h"
#include "../thread.h"

using namespace std;
using namespace Book;

namespace Book
{
	// 2021年に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"mcts_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2021(Position& pos, istringstream& is, const string& token)
	{
		if (token == "mcts_tree")
		{
			// 作業中
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)

