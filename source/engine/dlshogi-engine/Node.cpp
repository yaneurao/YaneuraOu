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
		bool found = false;
		for (int i = 0; i < child_num; ++i)
		{
			auto& uct_child = child[i];
			if (uct_child.move == move) {
				found = true;
				// 子ノードへのedgeは見つかっているけど実体がまだ。
				if (!uct_child.node)
	                uct_child.node = std::make_unique<Node>();

				// 0番目の要素に移動させる。
				if (i != 0)
					child[0] = std::move(uct_child);
			}
			else {
				// 子ノードを削除（ガベージコレクタに追加）
				if (uct_child.node)
	                gc->AddToGcQueue(std::move(uct_child.node));
			}
		}

		if (found) {
			// 子ノードを1つにする。
			child_num = 1;
			return child[0].node.get();
		}
		else {
	        // 子ノードが見つからなかった場合、新しいノードを作成する
			CreateSingleChildNode(move);
			child[0].node = std::make_unique<Node>();
			return child[0].node.get();
		}
	}

	// --- class NodeTree

	// 局面(Position)を渡して、node tree内からこの局面を探す。
	// もし見つかれば、node treeの再利用を試みる。
	// 新しい位置が古い位置と同じゲームであるかどうかを返す。
	//   sfen  : 今回探索するrootとなる局面のsfen文字列
	//   moves : 初期局面からposに至る指し手
	// ※　位置が完全に異なる場合、または以前よりも短い指し手がmovesとして与えられている場合は、falseを返す
	bool NodeTree::ResetToPosition(const std::string& rootSfen, const std::vector<Move>& moves)
	{
		if (game_root_sfen != rootSfen)
		{
			// 対局の開始局面のsfen文字列が異なる以上、完全に別のゲームであるから、
			// 前のゲームツリーを完全に開放する。
			DeallocateTree();

			// 対局開始局面のsfenの設定
			game_root_sfen = rootSfen;
		}

		if (!game_root_node)
		{
			// 新しい対局であり、一度目のこの関数の呼び出しであるから、現在の局面のために新規のNodeを作成し、
			// このNodeが対局開始のnodeであり、かつ、探索のroot nodeであると設定しておく。
			game_root_node = std::make_unique<Node>();
			current_head = game_root_node.get();
		}

		// 前回の探索開始局面
		Node* old_head = current_head;

		// 前回の探索開始局面の一つ前の局面(これは保持していないので最初はnullptr)
		Node* prev_head = nullptr;

		// 対局開始局面から探していき、現在の局面に到達する経路だけを残して他のノードを開放する。
		current_head = game_root_node.get();

		// 見つかったのかのフラグ
		bool seen_old_head = (game_root_node.get() == old_head);

		for (const auto& move : moves) {
			// 対局開始局面から、指し手集合movesで1手ずつ進めていく。
			prev_head = current_head;
			// 指し手以外の子ノードを開放する
			current_head = current_head->ReleaseChildrenExceptOne(gc,move);
			
			// 途中でold_headが見つかったならseen_old_headをtrueに。
			seen_old_head |= old_head == current_head;
		}

		// TODO ここの処理、あとでよくかんがえる

		// MakeMoveは兄弟が存在しないことを保証する 
		// ただし、古いヘッドが現れない場合は、以前に検索された位置の祖先である位置がある可能性があることを意味する
		// つまり、古い子が以前にトリミングされていても、current_head_は古いデータを保持する可能性がある
		// その場合、current_head_をリセットする必要がある
		if (!seen_old_head && current_head != old_head) {
			if (prev_head) {
				ASSERT_LV3(prev_head->child_num == 1);
				auto& prev_uct_child = prev_head->child[0];
				gc->AddToGcQueue(std::move(prev_uct_child.node));
				prev_uct_child.node = std::make_unique<Node>();
				current_head = prev_uct_child.node.get();
			}
			else {
				// 開始局面に戻った場合
				DeallocateTree();
			}
		}


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
