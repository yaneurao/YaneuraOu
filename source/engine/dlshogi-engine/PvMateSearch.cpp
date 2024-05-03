#ifndef __PV_MATE_SEARCH_CPP__
#define __PV_MATE_SEARCH_CPP__

#include "PvMateSearch.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include <cstring> // memcpy

#include "dlshogi_searcher.h"
#include "UctSearch.h"

namespace dlshogi
{
	// ゲーム木
	//extern std::unique_ptr<NodeTree> tree;
	//extern const Position* pos_root;

	PvMateSearcher::PvMateSearcher(const int nodes, DlshogiSearcher* dl_searcher) :
		ready_th(true),
		term_th(false),
		th(nullptr),
		dfpn(Mate::Dfpn::DfpnSolverType::Node48bitOrdering),
		dl_searcher(dl_searcher)
	{
		// 子ノードを展開するから、探索ノード数の8倍ぐらいのメモリを要する
		dfpn.alloc_by_nodes_limit((size_t)(nodes * 8));
		nodes_limit = nodes;
		// 最大探索深さ。これを超えると引き分けだから不詰扱い。
		dfpn.set_max_game_ply(dl_searcher->search_options.max_moves_to_draw);
	}

	// uct_nodeからPVを辿り、詰み探索をしていない局面を見つけたらdf-pnで詰探索を1回行う。
	// 詰み探索を行う局面がないか、詰探索を1回したならば、returnする。
	// 引数について)
	// root == trueの時。
	//	uct_node   : root node
	//  child_node : nullptr
	//  ⇨　uct_nodeが詰むなら、uct_node->child[i].move == moveの指し手に対して、詰み情報を書く。(SetLose)
	// 
	// root == falseの時
	//  uct_node   : child_nodeで辿った先のnode
	//  child_node : uct_nodeに到達する指し手
	//  ⇨　uct_nodeが詰むなら、child_node.SetWin()してやる。
	//
	// このnodeがroot nodeでかつ詰み探索がまだであるなら、探索する。
	// (dlshogiにはない処理だが、ふかうら王ではPV lineのmate searchを１本化したいのでこうする)
	void PvMateSearcher::SearchInner(Position& pos, Node* uct_node, ChildNode* child_node, bool root)
	{
		// 停止
		if (stop) return;

		// いまから詰み探索済みフラグをチェックするのでlockが必要
		Node::mtx_dfpn.lock();

		// このnodeは詰み探索済みであるか？
		if (!(uct_node->dfpn_proven_unsolvable || uct_node->dfpn_checked))
		{
			// 詰み探索まだ。

			// いったん詰み探索をしたことにする。(他のスレッドがこの局面を重複して探索しないように。)
			uct_node->dfpn_checked = true;
			Node::mtx_dfpn.unlock();

			// 詰みの場合、ノードを更新
			Move mate_move = dfpn.mate_dfpn(pos, nodes_limit);
			if (is_ok(mate_move)) {
				// 詰みを発見した。

				// rootで詰みを発見したのでメッセージを出力しておく。
				if (root)
				{
					// 何手詰めか
					int mate_ply = dfpn.get_mate_ply();
					sync_cout << "info string found the root mate by df-pn , move = " << to_usi_string(mate_move) << " ,ply = " << mate_ply << sync_endl;

					// 手数保存しておく。
					//uct_node->dfpn_mate_ply = mate_ply;

					// 読み筋を出力する。
					std::stringstream pv;
					for (auto move : dfpn.get_pv())
						pv << ' ' << to_usi_string(move);
					sync_cout << "info score mate " << mate_ply << " pv" << pv.str() << sync_endl;

					// moveの指し手をSetLose()する。
					// rootのchildは存在することが保証されている。(Expandしてから探索を開始するので)
					auto* child = uct_node->child.get();
					for (size_t i = 0; i < uct_node->child_num; ++i)
						if (child[i].getMove() == mate_move)
							// Node::Moveは上位8bitを使っているので.moveではなく.getMove()を用いる。
						{
							child[i].SetLose();
							return;
						}

				} else {

					// SetWinしておけばPV line上なので次のUctSearchで更新されるはず。
					// ここがrootなら、これでrootは詰みを発見するので自動的に探索が停止する。
					// ゆえに、rootであるかの判定等は不要である。
					child_node->SetWin();
					// ⇨　moveの指し手を指したら、子ノードの局面に即詰みがあるということなので
					//   現局面は負けの局面であることに注意。

#if 0
					// 何手詰めか
					int mate_ply = dfpn.get_mate_ply();
					int gamePly = pos.game_ply();
					sync_cout << "info string found the pv mate by df-pn , move = " << to_usi_string(move)
						<< " ,ply = " << mate_ply << " , gamePly = " << gamePly << sync_endl;

					// 手数保存しておく。
					uct_node->mate_ply = mate_ply;

					// 読み筋を出力する。
					std::stringstream pv;
					for (auto move : dfpn.get_pv())
						pv << ' ' << to_usi_string(move);
					sync_cout << "info string score mate " << mate_ply << " pv" << pv.str() << sync_endl;
#endif
				}
			}
			else if (stop) {
				// 途中で停止された場合、未探索に戻す。
				std::lock_guard<std::mutex> lock(Node::mtx_dfpn);
				uct_node->dfpn_checked = false;
			}
			// 探索中にPVが変わっている可能性があるため、ルートに戻る。
		}
		else {
			Node::mtx_dfpn.unlock();

			// 詰み探索済みであることがわかったので、
			// 子が展開済みの場合、PV上の次の手へ

			// 未展開の場合、終了する
			if (!uct_node->IsEvaled() || !uct_node->child) {
				std::this_thread::yield();
				return;
			}

			// 訪問回数が最大の子ノードを選択
			// ⇨　rootの時だけ時々second以降に行ってもいいかも..？
			const auto next_index = select_max_child_node(uct_node);

			auto uct_child = uct_node->child.get();

			// このPVが詰みが判明している場合、終了する
			if (uct_child[next_index].IsWin() || uct_child[next_index].IsLose()) {
				std::this_thread::yield();
				return;
			}

			// まだ子Nodeが生成されていないか？
			if (!uct_node->child_nodes)
				return;

			auto uct_next = uct_node->child_nodes[next_index].get();
			if (!uct_next)
				return;

			if (uct_node->child_nodes && uct_node->child_nodes[next_index])
			{
				// 1手進める。
				StateInfo st;
				Move m = uct_node->child[next_index].getMove();
				pos.do_move(m, st);

				// 再帰的に子を辿っていく。
				SearchInner(pos, uct_next, &uct_node->child[next_index], false);
			}
			else
				std::this_thread::yield();
		}
	}

	void PvMateSearcher::Run()
	{
		// th == nullptrなら、poolしているthread自体がないのでとりあえず生成をする。
		if (th == nullptr) {
			th = new std::thread([&]() {
				while (!term_th) {
					// 停止になるまで繰り返す
					while (!stop) {
						// 盤面のコピー
						// Position pos_copy(*pos_root);
						Position pos_copy;
						auto& pos_root = dl_searcher->pos_root;
						std::memcpy(&pos_copy, &pos_root, sizeof(Position));
						// ⇨ pos_rootはglobalに存在していてかつimmutableと考えられるので、
						// memcpyしても StateInfo* stが参照する先などは変わらず存在するから大丈夫。

						// PV上の詰み探索
						auto* tree = dl_searcher->get_node_tree();
						SearchInner(pos_copy, tree->GetCurrentHead(), nullptr, true);
					}

					std::unique_lock<std::mutex> lk(mtx_th);
					ready_th = false;
					cond_th.notify_all();

					// スレッドを停止しないで待機する
					cond_th.wait(lk, [this] { return ready_th || term_th; });
				}
				});
		}
		else {
			// poolしているスレッドがあるはずなので、そのスレッドを再開する。
			// スレッドを再開する。
			std::unique_lock<std::mutex> lk(mtx_th);
			ready_th = true;
			cond_th.notify_all();
		}
	}

	void PvMateSearcher::Stop(const bool stop)
	{
		dfpn.dfpn_stop(stop);
		this->stop = stop;
	}

	void PvMateSearcher::Join()
	{
		std::unique_lock<std::mutex> lk(mtx_th);
		if (ready_th && !term_th)
			cond_th.wait(lk, [this] { return !ready_th || term_th; });
	}

	// スレッドを終了
	void PvMateSearcher::Term()
	{
		{
			std::unique_lock<std::mutex> lk(mtx_th);
			term_th = true;
			ready_th = false;
			cond_th.notify_all();
		}
		if (th)
		{
			th->join();
			delete th;
			th = nullptr; // 二重Term()に対して安全に
		}
	}
}


#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __PV_MATE_SEARCH_CPP__
