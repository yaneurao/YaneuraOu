#ifndef DLSHOGI_MIN_H_INCLUDED
#define DLSHOGI_MIN_H_INCLUDED
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include <vector>
#include "../../types.h"

// 外部から探索部の一部の機能だけ呼び出したい時用のheader

namespace YaneuraOu {
namespace dlshogi {

	// USIのgoコマンド相当の探索が終了したあと、rootの各候補手と対応する評価値を返す。
	// Engine経由ならengine.go(limits)で探索を開始し、
	// engine.wait_for_search_finished()で終了を待ってから呼び出す。
	// 実装はengine/dlshogi-engine/FukauraOuEngine.cppを参照。
	void GetSearchResult(std::vector<std::pair<Move, float>>& result);

	// sfenとnode数を保持する構造体
	struct SfenNode
	{
		SfenNode(){}
		SfenNode(const std::string& sfen_, u64 nodes_):
			sfen(sfen_), nodes(nodes_) {}

		std::string sfen;
		u64 nodes;

		// sortのための比較演算子
		bool operator <(const SfenNode& rhs) const
		{
			// sort()した時に降順でソートされて欲しい。
			return nodes > rhs.nodes;
		}
	};
	typedef std::vector<SfenNode> SfenNodeList;

	// 訪問回数上位 n 個の局面のsfen文字列を返す。(ただし探索開始局面と同じ手番になるように偶数手になるように)
	// ここで得られた文字列は、探索開始局面のsfenに指し手として文字列の結合をして使う。文字列の先頭にスペースが入る。
	// same_colorはtrueならrootと同じ手番の局面(偶数局面)、falseならrootと異なる手番の局面(奇数局面)を返す。
	// 例)
	// 　探索開始局面 = "startpos"
	//   same_color   = true
	// 　返ってきたsfens = [" 7g7f 8c8d", " 2g2f 3c3d"]
	//
	//   この時、実際にposition文字列として有効なsfen文字列は、
	//    "startpos moves 7g7f 8c8d"
	//    "startpos moves 2g2f 3c3d"
	//   なので、そうなるように文字列を結合すること。
	void GetTopVisitedNodes(size_t n, SfenNodeList& sfens, bool same_color);

	// 探索したノード数を返す。
	// これは、ThreadPool classがnodes_searched()で返す値とは異なる。
	//  →　そちらは、Position::do_move()した回数。
	// こちらは、GPUでevaluate()を呼び出した回数。俗に言うnodes visited。
	u64 nodes_visited();

} // namespace dlshogi
} // namespace YaneuraOu


#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ifndef DLSHOGI_MIN_H_INCLUDED
