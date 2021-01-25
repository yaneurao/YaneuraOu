#include "Node.h"
#if defined(YANEURAOU_ENGINE_DEEP)
#include "../../misc.h"

namespace dlshogi
{
	// --- struct Node

	// 引数のmoveで指定した子ノード以外の子ノードをすべて開放する。
	// 前回探索した局面からmoveの指し手を選んだ局面の以外の情報を開放するのに用いる。
	Node* Node::ReleaseChildrenExceptOne(NodeGarbageCollector* gc, const Move move)
	{
		if (child_num > 0 && child_nodes) {
			bool found = false;
			for (int i = 0; i < child_num; ++i)
			{
				auto& uct_child = child[i];
				auto& child_node = child_nodes[i];
				if (uct_child.move == move) {
					found = true;
					// 子ノードへのedgeは見つかっているけど実体がまだ。
					if (!child_node)
	                    // 新しいノードを作成する
	                    child_node = std::make_unique<Node>();

					// 0番目の要素に移動させる。
					if (i != 0) {
						child[0] = std::move(uct_child);
						child_nodes[0] = std::move(child_node);
					}
				}
				else {
					// 子ノードを削除（ガベージコレクタに追加）
					if (child_node)
						gc->AddToGcQueue(std::move(child_node));
				}
			}

			if (found) {
				// 子ノードを1つにする。
				child_num = 1;
				return child_nodes[0].get();
			}
			else {
				// 子ノードが見つからなかった場合、新しいノードを作成する
				CreateSingleChildNode(move);
				InitChildNodes();
				return (child_nodes[0] = std::make_unique<Node>()).get();
			}
		}
		else {
			// 子ノード未展開、または子ノードへのポインタ配列が未初期化の場合
			CreateSingleChildNode(move);
			// 子ノードへのポインタ配列を初期化する
			InitChildNodes();
			return (child_nodes[0] = std::make_unique<Node>()).get();
		}
	}

	// --- class NodeTree

	// ゲーム開始局面からの手順を渡して、node tree内からこの局面を探す。
	// もし見つかれば、node treeの再利用を試みる。
	// 新しい位置が古い位置と同じゲームであるかどうかを返す。
	//   game_root_sfen : ゲーム開始局面のsfen文字列
	//   moves          : game_root_sfenの局面からposに至る指し手
	// ※　位置が完全に異なる場合、または以前よりも短い指し手がmovesとして与えられている場合は、falseを返す
	bool NodeTree::ResetToPosition(const std::string& game_root_sfen , const std::vector<Move>& moves)
	{
		// 前回思考した時とは異なるゲーム開始局面であるなら異なるゲームである。
		if (this->game_root_sfen != game_root_sfen)
		{
			DeallocateTree();
			this->game_root_sfen = game_root_sfen;
		}

		// root nodeがまだ生成されていない
		if (!game_root_node)
		{
			// 新しい対局であり、一度目のこの関数の呼び出しであるから、現在の局面のために新規のNodeを作成し、
			// このNodeが対局開始のnodeであり、かつ、探索のroot nodeであると設定しておく。
			game_root_node = std::make_unique<Node>();
			current_head = game_root_node.get();
		}

		// 前回の探索開始局面
		Node* old_head = current_head;

		// nodeを辿って行った時の一つ前のnode
		Node* prev_head = nullptr;

		// 現在のnode。ゲーム開始局面から辿っていく。
		current_head = game_root_node.get();

		// 前回の探索rootの局面が、与えられた手順中に見つかったのかのフラグ
		bool seen_old_head = (game_root_node.get() == old_head);

		// 対局開始局面から、指し手集合movesで1手ずつ進めていく。
		for (const auto& move : moves) {
			// 一つ前のnode
			prev_head = current_head;

			// 現在の局面に到達する経路だけを残して他のノードを開放する。(なければNodeを作るのでnullptrになることはない)
			current_head = current_head->ReleaseChildrenExceptOne(gc,move);
			
			// 途中でold_headが見つかったならseen_old_headをtrueに。
			// ここを超えて進んだなら、前回の探索結果が使える。
			seen_old_head |= old_head == current_head;
		}

		// 以前の局面に戻っているのか何かは知らないが、存在しないので新規扱いでいいのでは…。
		if (!seen_old_head)
			DeallocateTree();

		return seen_old_head;
	}

	void NodeTree::DeallocateTree()
	{
		// ゲームツリーを保持しているならそれを開放する。
		// (保持していない時は何もしない)
		gc->AddToGcQueue(std::move(game_root_node));
		game_root_node = std::make_unique<Node>();
		current_head = game_root_node.get();
	}

}

#endif // defined(YANEURAOU_ENGINE_DEEP)
