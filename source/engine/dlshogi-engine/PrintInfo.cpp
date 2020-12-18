#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include "PrintInfo.h"
#include "Node.h"
#include "UctSearch.h"
#include "dlshogi_searcher.h"

#include "../../usi.h"

#include <iomanip>		// std::setw()
#include <sstream>		// std::stringstream

namespace dlshogi::UctPrint
{
	// ---   print PV   ---

	// ChildNode.value_winとmove_countを組にした構造体
	struct ValueWinMoveCount
	{
		//ValueWinMoveCount(WinCountType value_win_, NodeCountType move_count_) : value_win(value_win_), move_count(move_count_) {}
		ValueWinMoveCount(const ChildNode& child) : child(const_cast<ChildNode*>(&child)) , value_win(child.node ? (WinCountType)child.node->value_win : 0), move_count(child.move_count) {}

		WinCountType value_win;
		NodeCountType move_count;
		ChildNode* child;
	};

	// ChildNodeのvalue_winとmove_count の組を与えて、どちらのほうが良い指し手であるかを判定する。
	// 前者のほうが優れていれば、trueが返る。
	bool is_superior_to(const ValueWinMoveCount& lhs,const ValueWinMoveCount& rhs)
	{
		// 子がVALUE_LOSEなら勝ちなので、この指し手を優先して選択。しかし、両方がVALUE_LOSEなら、move_countが多いほうを選択。
		// 子がVALUE_WINなら負けなので、そうでないほうを選びたいが、どちらもVALUE_WINなら、move_countが多いほうが良い指し手のはずなので、move_countが多いほうを選択。

		// win_typeとして、 勝ち = 2 , わからない = 1 , 負け = 0 を返す。
		auto win_type_of = [](WinCountType v) { return v == VALUE_LOSE ? 2 : v == VALUE_WIN ? 0 : 1; };

		int t1 = win_type_of(lhs.value_win);
		int t2 = win_type_of(rhs.value_win);

		// 同じwin_typeなら、move_countが大きいほう。さもなくば、win_typeの値が大きいほうの順番で並んでほしい。
		return t1 == t2 ? lhs.move_count > rhs.move_count : t1 > t2;
	}


	// 子ノード同士の優劣を比較して、親局面から見てどちらを選択すべきであるかを判定する。
	// lhsのほうがrhsより優れていたらtrueを返す。
	bool is_superior_to(const ChildNode& lhs, const ChildNode& rhs)
	{
		return is_superior_to(ValueWinMoveCount(lhs), ValueWinMoveCount(rhs));
	}

	// あるNodeで選択すべき指し手を表現する。
	struct BestMove
	{
		// 指し手
		Move move;

		// moveを選んだ時の勝率
		WinCountType wp;

		// moveを指した時に遷移するNode
		Node* node;

		BestMove() : move(MOVE_NONE), wp(0), node(nullptr){}
		BestMove(Move move_,WinCountType wp_,Node* node_) :move(move_), wp(wp_) , node(node_) {}
	};

	BestMovePonder::BestMovePonder() : move(MOVE_NONE), wp(0), ponder(MOVE_NONE) {}


	// あるnodeの子ノードのbestなやつを選択する。
	//   返し値 : 子ノードのindexが返る。 node.child[index] のようにして使う。
	//            子nodeがない場合(詰みの局面)は、-1が返る。
	int select_best_child(Node* node)
	{
		int child_num = node->child_num;
		if (child_num == 0)
			return -1;

		// 子ノードすべてから、一番優れたChildNodeを選択してそれを返す。
		ChildNode* child = node->child.get();
		int best_child = 0;
		
		for (int i = 1; i < child_num; ++i)
			if (is_superior_to(child[i],child[best_child]))
				best_child = i;

		return best_child;
	}

	// あるnodeの子ノードのbestのやつの指し手を返す。
	// 詰みの局面ならMOVE_NONEが返る。
	BestMove select_best_move(const Node* node)
	{
		// 子ノードが未展開であるなら比較自体ができない。
		if (node->move_count == 0 || !node->child)
			return BestMove();

		// 子ノードすべてから、一番優れたChildNodeを選択してそれを返す。
		const ChildNode* child = node->child.get();

		// 一番良かった子ノードのindex
		int best_child = 0;

		// 一つはVALUE_LOSEであった == その指し手を選んで勝ち = 期待勝率 100%
		bool found_lose = false;

		// すべてVALUE_WINであった == どうやっても負け = 期待勝率 0%
		bool all_win  = true;

		for (int i = 1; i < node->child_num; ++i)
		{
			const auto& c = child[i];
			const Node* child_node = c.node.get();
			const WinCountType child_value_win = child_node ? (WinCountType)child_node->value_win : 0;

			found_lose  |= child_value_win == VALUE_LOSE;
			all_win     &= child_value_win == VALUE_WIN;

			if (is_superior_to(c,child[best_child]))
				best_child = i;
		}

		// 期待勝率
		WinCountType wp = 
			  found_lose ? (WinCountType)1.0
			: all_win    ? (WinCountType)0.0
			: child[best_child].move_count == 0 ? (WinCountType)0.5 // わからんからとりあえず0.5にしとく。
			: child[best_child].win / child[best_child].move_count;

		return BestMove(node->child[best_child].move, wp ,child[best_child].node.get());
	}

	// あるnodeの子ノードのbestのやつの指し手を返す。
	// 詰みの局面ならMOVE_NONEが返る。
	std::vector<BestMove> select_best_moves(const Node* node , size_t multiPv)
	{
		std::vector<BestMove> bests;

		// node->child_num != 0であるなら、ノードは展開されてはいるが、move_count == 0 であるなら、
		// この時、nnrateまで拾う意味はないと思うので、無視する。
		// そもそもrootはExpandRoot()で展開されているし、そのあと、1ノードでも読めば、どの子かは展開されているはずだから、
		// rootの1番目の指し手自体は存在するはず。(2番目以降は存在性を保証しなくていいと思う)
		if (node->child_num == 0 || node->move_count == 0 )
			return bests;

		// 子ノードすべてから、上位multiPv個の優れたChildNodeを選択してそれを返す。

		const ChildNode* child = node->child.get();

		// 子ノードのValueWinMoveCountの一覧を作って、この上位 multiPV個を選出する。
		std::vector<std::pair<int,ValueWinMoveCount>> list;
		list.reserve(node->child_num);
		for (int i = 0; i < node->child_num; ++i)
			list.emplace_back(i, ValueWinMoveCount(child[i]));

		// 上位 multiPv個をソートして、あとは捨てる
		multiPv = std::min(multiPv, list.size());
		std::partial_sort(list.begin(), list.begin() + multiPv , list.end(), [](auto& rhs, auto& lhs){ return is_superior_to(rhs.second,lhs.second); });

		// listには良い順に並んでいる。例えば、1番良いChildNodeは、child[list[0].first]
		for (int i = 0; i < multiPv ; ++i)
		{
			const auto& c = child[list[i].first];
			if (c.node == nullptr)
				continue;

			const Node* child_node = c.node.get();
			const WinCountType child_value_win = child_node ? (WinCountType)child_node->value_win : 0;

			WinCountType wp = 
				   child_value_win == VALUE_LOSE ? (WinCountType)1.0
				:  child_value_win == VALUE_WIN  ? (WinCountType)0.0
				:  child_node->move_count == 0   ? (WinCountType)0.5 // 未訪問なら0.5にしとく。
				:  (1 - child_node->win / child_node->move_count);

			bests.push_back(BestMove(c.move, wp, c.node.get()));
		}
		return bests;
	}


	// rootNodeのベストな指し手とponderの指し手を取得する。期待勝率も計算して代入して返す。
	BestMovePonder get_best_move_ponder(Node* rootNode)
	{
		// 普通にbestな子ノードを選択。
		auto best = select_best_move(rootNode);

		// 子ノードに移動してponderの指し手をもらう
		Move ponderMove = best.node ? select_best_move(best.node).move : MOVE_NONE;

		return BestMovePonder(best.move, best.wp, ponderMove);
	}

	// あるノード以降のPV(最善応手列)を取得する。
	void  get_pv(Node* node , std::vector<Move>& moves)
	{
		while (node)
		{
			auto best_child = select_best_child(node);
			if (best_child == -1)
				break;

			moves.push_back(node->child[best_child].move);
			if (!node->child)
				break;

			if (node->move_count == 0)
				break;

			node = node->child[best_child].node.get();
		}
	}

	// 読み筋を出力する。
	// multipv      : Options["MultiPV"]の値
	// multipv_num  : MultiPVの何番目の指し手であるか。(0～multipv-1の間の数値)
	std::string pv_to_string(BestMove best, std::vector<Move>& moves , size_t multipv , int multipv_num , const std::string& nps)
	{
		std::stringstream ss;

		// 勝率を[centi-pawn]に変換
		int cp = Eval::dlshogi::value_to_cp(best.wp);

		ss << "info" << nps;

		// MultiPVが2以上でないなら、"multipv .."は出力しないようにする。(MultiPV非対応なGUIかも知れないので)
		if (multipv > 1)
			ss << " multipv " << (multipv_num + 1);

		ss << " depth " << moves.size() << " score cp " << cp;
		
		// 読み筋
		if (moves.size())
		{
			ss << " pv";
			for (auto m : moves)
				ss << ' ' << to_usi_string(m);
		}

		return ss.str();
	}

	// ベストの指し手とponderの指し手の取得
	BestMovePonder get_best_move_multipv(const Node* rootNode , const SearchLimit& po_info , const SearchOptions& options )
	{
		size_t multiPv = options.multi_pv;
		
		// 探索にかかった時間を求める
		auto finish_time = std::max((TimePoint)1, po_info.begin_time.elapsed());
		std::stringstream nps;
		nps << " nps "      << (po_info.node_searched * 1000LL / (u64)finish_time)
			<< " time "     <<  finish_time
			<< " nodes "    <<  po_info.node_searched
			<< " hashfull " << (po_info.current_root->move_count * 1000LL / options.uct_node_limit);
		
		if (multiPv <= 1)
		{
			auto best = select_best_move(rootNode);
			if (best.move == MOVE_NONE)
				return BestMovePonder();

			// 2手目以降のPVを取得する。
			std::vector<Move> moves = { best.move };
			get_pv(best.node,moves);

			sync_cout << pv_to_string(best, moves ,multiPv ,0 , nps.str() ) << sync_endl;

			Move ponder = moves.size() >= 2 ? moves[1] : MOVE_NONE;
			return BestMovePonder(best.move, best.wp, ponder );

		} else {

			// MultiPVなので、現在のnodeで複数の候補手を表示する。

			auto bests = select_best_moves(rootNode , multiPv);
			if (bests.size() == 0)
				return BestMovePonder();

			Move ponder = MOVE_NONE;
			for(int i = 0; i < bests.size() ; ++i)
			{
				auto best = bests[i];

				std::vector<Move> moves = { best.move };
				get_pv(best.node, moves);

				sync_cout << pv_to_string(best, moves, multiPv, i , nps.str() ) << sync_endl;

				if (i == 0 && moves.size() >= 2)
					ponder = moves[1];
			}

			return BestMovePonder(bests[0].move, bests[0].wp, ponder);
		}

	}

	// --- Debug Message ---

	// 探索の情報の表示
	void PrintPlayoutInformation(const Node* root, const SearchLimit* po_info, const TimePoint finish_time, const NodeCountType pre_simulated)
	{
		double finish_time_sec = finish_time / 1000.0;

		// rootの訪問回数がPlayoutの回数
		sync_cout << "All Playouts       :  " << std::setw(7) << root->move_count << sync_endl;

		// 前回の探索での現在のrootの局面の訪問回数(この分は今回探索していない)
		sync_cout << "Pre Simulated      :  " << std::setw(7) << pre_simulated << sync_endl;

		// 今回の思考時間
		sync_cout << "Thinking Time      :  " << std::setw(7) << finish_time_sec << " sec" << sync_endl;

		// 期待勝率
		// 今回の探索の期待勝率は、rootの局面の勝ち数 / rootの訪問回数
		// double型で計算するのでゼロ除算の例外はでないが、-Nan/INFとか出ても嫌なので分母が0なら表示しないようにしておく。
		if (root->move_count)
		{
			double winning_percentage = (double)root->win / root->move_count;
			sync_cout << "Winning Percentage :  " << std::setw(7) << (winning_percentage * 100) << "%" << sync_endl;
		}

		// 思考時間が0でないなら、Playout/secを出す。
		if (finish_time_sec != 0.0)
			sync_cout << "Playout Speed      :  " << std::setw(7) << (int)(po_info->node_searched / finish_time_sec) << " PO/sec " << sync_endl;

	}

	// 探索時間の出力
	void PrintPlayoutLimits(const TimePoint time_limit, const int playout_limit)
	{
		sync_cout << "Time Limit    : " << time_limit    << " Sec" << sync_endl;
		sync_cout << "Playout Limit : " << playout_limit << " PO"  << sync_endl;
	}

	// 再利用した探索回数の出力
	void PrintReuseCount(const int count)
	{
		sync_cout << "Reuse : " << count << " Playouts" << sync_endl;
	}

}


#endif // defined(YANEURAOU_ENGINE_DEEP)
