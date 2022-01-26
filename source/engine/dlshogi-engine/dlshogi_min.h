#ifndef __DLSHOGI_MIN_H_INCLUDED__
#define __DLSHOGI_MIN_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include <vector>
#include "../../types.h"

// 外部から探索部の一部の機能だけ呼び出したい時用のheader

namespace dlshogi {

	// 探索結果を返す。
	//   Threads.start_thinking(pos, states , limits);
	//   Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。
	// のようにUSIのgoコマンド相当で探索したあと、rootの各候補手とそれに対応する評価値を返す。
	//
	// ※　実際の使用例は、make_book2021.cppのthink_sub()にあるのでそれも参考にすること。
	extern std::vector<std::pair<Move, float>> GetSearchResult();

	// 探索したノード数を返す。
	// これは、ThreadPool classがnodes_searched()で返す値とは異なる。
	//  →　そちらは、Position::do_move()した回数。
	// こちらは、GPUでevaluate()を呼び出した回数。俗に言うnodes visited。
	extern u64 nodes_visited();
}

namespace Eval::dlshogi {

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	// 	 eval_coef : 勝率を評価値に変換する時の定数。default = 756
	// 
	// 返し値 :
	//   +29900は、評価値の最大値
	//   -29900は、評価値の最小値
	//   +30000,-30000は、(おそらく)詰みのスコア
	Value value_to_cp(const float score, float eval_coef);
}

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ifndef __DLSHOGI_MIN_H_INCLUDED__
