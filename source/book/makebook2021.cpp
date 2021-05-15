#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
// いまのところ、ふかうら王のみ対応。気が向いたら、NNUEにも対応させるが、NNUEの評価関数だとMCTS、あまり相性良くない気も…。

// -----------------------
// MCTSで定跡を生成する
// -----------------------

/*
	dlshogiの定跡生成部を参考にしつつも独自の改良を加えた。

	大きな改良点) 親ノードに訪問回数を伝播する。

		dlshogiはこれをやっていない。
		これを行うと、局面に循環があると訪問回数が発散してしまうからだと思う。
		// これだと訪問回数を親ノードに伝播しないと、それぞれの局面で少ないPlayoutで事前に思考しているに過ぎない。

		そこで、親ノードに訪問回数は伝播するが、局面の合流は処理しないようにする。

		この場合、合流が多い変化では同じ局面を何度も探索することになりパフォーマンスが落ちるが、
		それを改善するためにleaf nodeでの思考結果はcacheしておき、同じ局面に対しては探索部を２回
		呼び出さないようにして解決する。

		また合流を処理しないため、同一の局面であっても経路によって異なるNodeとなるが、書き出す定跡ファイルとは別に
		この経路情報を別のファイルに保存しておくので、前回の定跡の生成途中から再開できる。

		その他、いくつかのテクニックを導入することでMCTSで定跡を生成する上での問題点を解決している。

*/

/*
	それぞれのfileフォーマット

	入力)
		root_sfen.txt
		// 探索開始局面
		例)
		sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
		↑のようなsfen形式か、
		startpos moves 7g7f ...
		のようなUSIプロトコルの"position"コマンドとして送られる形式でも可。
		// この場合、それぞれの局面が探索開始対象となる。

	出力)
		leaf_sfen.txt
			探索済leaf nodeのsfenとそのvalue,指し手
			例)
				sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 // sfen
				0.5,10000,7g7f,100
				// この局面の探索した時のvalue(float値,手番側から見た値),playoutの回数,1つ目の指し手(ないときはresign),1つ目の指し手の探索回数

		mctsbook.serialized
			探索tree(内部状態)をそのまま書き出したの。

		user_book_mcts.db
			やねうら王で定跡ファイルとして使えるファイル

*/

#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_set>

#include "../usi.h"
#include "../misc.h"
#include "../thread.h"
#include "../book/book.h"
#include "../learn/learn.h"

using namespace std;
using namespace Book;

// positionコマンドのparserを呼び出したい。
extern void position_cmd(Position& pos, istringstream& is, StateListPtr& states);

namespace {

	// =================================================================
	//                        探索済みのNode
	// =================================================================

	// 生の探索結果
	struct SearchResult
	{
		float		value;	// 探索した時のこの局面の評価値(手番側から見た値)
		Move16		move;	// 探索した時のこの局面の最善手(MOVE_RESIGN,MOVE_WINもありうる)
	};

	// ↓これを引換券にして、↑のinstanceがもらえたりする。
	typedef u32 NodeIndex;

	// 探索結果をシリアルに並べたもの。この順番は入れ替わらないものとする。
	// SearchResults results;
	// NodeIndex index;
	// auto result = results[index];
	// のように、indexが局面のIDであるものとする。
	typedef std::vector<SearchResult> SearchResults;

	// PackedSfenから、SearchResultIndexへのmap。このindexを使って、↑のSearchResults型のinstanceの要素にアクセスする。
	// unordered_mapに直接SearchResultを持たせたいのだが、そうすると、このコンテナに対してinsert()したときなどにコンテナ要素のアドレスが変わってしまうので
	// アドレスで扱うわけにいかない。そこで、object IDのようなものでこの部分を抽象化する必要がある。
	// ここでは、SearchResultsをvectorにして、このコンテナの何番目の要素であるか、というのをSearchResultIndex型で示すというようにしてある。
	typedef std::unordered_map<PackedSfen, NodeIndex, PackedSfenHash> SfenToIndex;

	// ファイルに保存しておく探索結果の 1 record
	// 32 + 4 + 2 = 38 byte
	struct SearchResultRecord
	{
		PackedSfen	 packed_sfen;	// 局面をPositionクラスでpackしたもの。32 bytes固定長。
		SearchResult result;
	};

	// =================================================================
	//                        MCTSの探索木
	// =================================================================

	// MCTSで用いるNode構造体
	struct MctsNode {
		u32		move_count;				// このノードの訪問回数。
		u32		move_count_from_this;	// このノードから訪問した回数
		double	value_win;				// このノードの勝率(手番側から見たもの)
	};

	// ↑をシリアル化に並べた型
	typedef std::vector<MctsNode> MctsNodes;

	// 子ノードへのedgeと子ノードを表現する構造体
	struct MctsChild
	{
		// 子ノードへ到達するための指し手
		Move move;

		// 子ノードのindex(これを元にSearchResult*が確定する)
		NodeIndex index;
	};

	// あるNodeの子node一覧を表現する型
	typedef std::vector<MctsChild> MctsChildren;

	// ↑の何番目に格納されているのかを表現する型
	typedef u32 MctsChildrenIndex;

	// =================================================================
	//                        Nodeの管理用
	// =================================================================

	// 探索したノードを記録しておくclass。
	// ファイルへのserialize/deserializeもできる。
	// 純粋な探索の結果なので、これは価値があるはずで、SearchResultRecordを繰り返すだけの単純なデータフォーマットにしておく。
	class NodeManager
	{
	public:
		// ファイルから読み込み。
		void Deserialize(const std::string& filename) {}

		// ファイルに保存。
		void Serialize(const std::string& filename) {}

		// 要素ひとつ追加する
		void append(const PackedSfen& ps , const SearchResult& result) {
			// この要素を持っていてはならない。
			ASSERT_LV3(sfen_to_index.find(ps) == sfen_to_index.end());

			NodeIndex index = (NodeIndex)search_results.size();
			search_results.emplace_back(result);
			sfen_to_index[ps] = index;
		}

		// indexを指定してSearchResult*を得る
		// このpointerはappend()やgo_search()などが呼び出されるまでは有効。
		// index == UINT32_MAXのときは、nullptrが返る。
		SearchResult* get_search_result(NodeIndex index)
		{
			if (index == UINT32_MAX)
				return nullptr;
			return &search_results[(size_t)index];
		}

		// indexを指定してMctsNode*を得る。
		// このpointerはappend()やgo_search()などが呼び出されるまでは有効。
		// index == UINT32_MAXのときは、nullptrが返る。
		MctsNode* get_node(NodeIndex index)
		{
			if (index == UINT32_MAX)
				return nullptr;
			return &mcts_nodes[(size_t)index];
		}

		// 指定されたsfenに対応するindexを得る。
		// 格納されていなければUINT32_MAXが返る。
		NodeIndex get_index(const PackedSfen& ps) {
			auto it = sfen_to_index.find(ps);
			if (it == sfen_to_index.end())
				return UINT32_MAX;
			return it->second;
		}

		// sfenで指定した局面を探索して、この構造体の集合のなかに追加する。
		SearchResult* go_search(std::string& sfen , u64 nodes_limit)
		{
			// SetupStatesは破壊したくないのでローカルに確保
			StateListPtr states(new StateList(1));

			Position pos;

			// sfen文字列、Positionコマンドのparserで解釈させる。
			istringstream is(sfen);
			position_cmd(pos, is, states);

			sync_cout << "Position: " << sfen << sync_endl;

			// 探索時にnpsが表示されるが、それはこのglobalなTimerに基づくので探索ごとにリセットを行なうようにする。
			Time.reset();
			Search::LimitsType limits;
			limits.nodes = nodes_limit; // これだけ思考する
			limits.silent = true;       // silent modeで

			Threads.start_thinking(pos, states, limits);
			Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。

			sync_cout << "nodes_searched = " << Threads.nodes_searched() << sync_endl;

			// TODO : 探索結果は何とかして取り出すべし。


			return nullptr;
		}

	private:

		// packed sfenから、NodeIndexへのmap
		SfenToIndex sfen_to_index;

		// SearchResultをシリアルに並べた型
		SearchResults search_results;

		// MctsNodeをシリアルに並べた型
		MctsNodes mcts_nodes;
	};

	// MctsChildrenをcacheしておく。
	// MCTSの探索時に毎回、指し手生成をして1手進めて、探索済みのchild nodeであるかを確認して1手戻す..みたいなことを繰り返したくないため)
	class MctsChildrenCacheManager
	{
		// あるNodeIndexに対応する、(そのnodeの)MctsChildrenを返す。
		// cacheされていなければnullptrが返る。
		// ここで返されたpointerは、次にappend()かclear()が呼び出されるまで有効。
		MctsChildren* get_children(NodeIndex index)
		{
			auto it = node_index_to_children_index.find(index);
			if (it == node_index_to_children_index.end())
				return nullptr;
			return &childrens[it->second];
		}

		// あるNodeIndexに対応するMctsChildrenを格納する。
		void append(NodeIndex index,const MctsChildren& children)
		{
			// すでに格納されているなら、これ以上重複して格納することはできない。
			ASSERT_LV3(get_children(index) == nullptr);

			auto children_index = (MctsChildrenIndex)children.size();
			childrens.emplace_back(children);
			node_index_to_children_index[index] = children_index;
		}

		// cacheしているMctsChildrenをクリアする。
		void clear()
		{
			childrens.clear();
			node_index_to_children_index.clear();
		}

	private:
		// cacheされているMctsChildren集合。
		std::vector<MctsChildren> childrens;

		// ↑の何番目に格納されているのかの対応関係のmap
		std::unordered_map<NodeIndex, MctsChildrenIndex> node_index_to_children_index;
	};

	// =================================================================
	//                   MCTSの探索rootを表現する
	// =================================================================

	// 探索Root集合の型
	typedef std::vector<std::string> RootSfens;

	// positionコマンドに渡す形式で書かれているsfenファイルを読み込み、そのsfen集合を返す。
	// 重複局面は除去される。
	// all_node   : そこまでの手順(経由した各局面)も含めてすべて読み込む。
	// ignore_ply : 手数を無視する。(sfen化するときに手数をtrimする)
	// 
	// file formatは、
	// "startpos move xxxx xxxx"
	// "[sfen文字列] moves xxxx xxxx"
	RootSfens ReadPositionFile(const string& filename, bool all_node, bool ignore_ply)
	{
		std::unordered_set<std::string> sfens;

		TextFileReader reader;
		reader.Open(filename);

		std::string line, token, sfen;
		while (!reader.ReadLine(line).is_eof())
		{
			// line : この1行がpositionコマンドに渡す文字列と同等のもの
			std::istringstream is(line);
			is >> token;
			if (token == "startpos")
			{
				sfen = SFEN_HIRATE;
				is >> token; // "moves"を消費する
			}
			else {
				// "sfen"は書いてなくても可。
				if (token != "sfen")
					sfen += token + " ";
				while (is >> token && token != "moves")
					sfen += token + " ";
			}

			// 新しく渡す局面なので古いものは捨てて新しいものを作る。
			auto states = StateListPtr(new StateList(1));
			Position pos;
			pos.set(sfen, &states->back(), Threads.main());

			// 返す局面集合に追加する関数
			auto insert = [&] {
				std::string s = pos.sfen();
				if (sfens.count(s) == 0)
					sfens.insert(s);
			};

			// 開始局面をsfen化したものを格納
			if (all_node)
				insert();

			// 指し手のparser
			Move m;
			while (is >> token && (m = USI::to_move(pos, token)) != MOVE_NONE)
			{
				// 1手進めるごとにStateInfoが積まれていく。これは千日手の検出のために必要。
				states->emplace_back();
				if (m == MOVE_NULL) // do_move に MOVE_NULL を与えると死ぬので
					pos.do_null_move(states->back());
				else
					pos.do_move(m, states->back());

				if (all_node)
					insert();
			}

			// all_node == falseならば、最後の局面だけ返す。
			if (!all_node)
				insert();
		}

		// vectorに変換して返す。
		std::vector<std::string> sfens_vector;
		for (auto s : sfens)
		{
			if (ignore_ply)
				s = StringExtension::trim_number(s);
			sfens_vector.emplace_back(s);
		}

		return sfens_vector;
	}

	// =================================================================
	//             mcts定跡生成コマンド本体
	// =================================================================

	// MCTSで定跡を生成する本体
	class MctsMakeBook
	{
	public:

		// MCTSをやって定跡を生成する。
		void make_book(Position& pos, istringstream& is)
		{
			// loop_maxに達するか、"stop"が来るまで回る

			// この回数だけrootから探索したら終了する。
			u64 loop_max = 10000000;

			// 一つの局面(leaf node)でのPlayoutの回数
			u64 playout = 10000;

			// 探索root集合
			string root_filename = "mctsbook_root_sfen.txt";

			// 定跡ファイル名
			string book_filename = "mctsbook_user_book.db";

			// 定跡のMCTS探索中のメモリ状態をserializeしたやつ
			string mcts_tree_filename = "mctsbook_mcts_tree.db";

			// 定跡のleaf nodeのsfen
			string search_result_filename = "mctsbook_search_result.db";


			// 定跡ファイルの保存間隔。デフォルト、30分ごと。
			TimePoint book_save_interval = 60 * 30;

			// 最大手数
			int max_ply = 384;

			string token;
			while (is >> token)
			{
				if (token == "loop")
					is >> loop_max;
				else if (token == "playout")
					is >> playout;
				else if (token == "root_filename")
					is >> root_filename;
				else if (token == "book_filename")
					is >> book_filename;
				else if (token == "tree_filename")
					is >> mcts_tree_filename;
				else if (token == "search_result_filename")
					is >> search_result_filename;
				else if (token == "book_save_interval")
					is >> book_save_interval;
				else if (token == "max_ply")
					is >> max_ply;
			}

			cout << "makebook mcts command" << endl
				<< "  root filename       = " << root_filename << endl
				<< "  book_filename       = " << book_filename << endl
				<< "  mcts_tree_filename  = " << mcts_tree_filename << endl
				<< "  loop                = " << loop_max << endl
				<< "  playout             = " << playout << endl
				<< "  book_save_interval  = " << book_save_interval << endl
				<< "  max_ply             = " << max_ply << endl
				;

			cout << "read root file.." << endl;

			// そこまでの手順も含めてすべて読み込み、sfenにする。
			auto roots = ReadPositionFile(root_filename, true, true);
			cout << "  root sfens size()  = " << roots.size() << endl;

#if 0
			// デバッグ用に読み込まれたsfenを出力する。
			for (auto sfen : roots)
				cout << "sfen " << sfen << endl;
#endif



#if 0

			// 定跡DBを書き出す。
			auto save_book = [&]()
			{
				std::lock_guard<std::mutex> lk(book_mutex);

				sync_cout << "savebook ..start : " << filename << sync_endl;
				book.write_book(filename);
				sync_cout << "savebook ..done." << sync_endl;
			};

			// 定跡保存用のtimer
			Timer time;

			for (u64 loop_counter = 0 ; loop_counter < loop_max ; ++loop_counter)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));

				// 進捗ここで出力したほうがいいかも？

				// 定期的に定跡ファイルを保存する
				if (time.elapsed() >= book_save_interval * 1000)
				{
					save_book();
					time.reset();
				}
			}

			// 最後にも保存しないといけない。
			save_book();
#endif

			cout << "makebook mcts , done." << endl;
		}

#if 0
		// 各探索用スレッドのentry point
		void Worker(size_t thread_id)
		{
			//sync_cout << "thread_id = " << thread_id << sync_endl;

			WinProcGroup::bindThisThread(thread_id);

			Position pos;
			StateInfo si;
			pos.set_hirate(&si,Threads[thread_id]); // 平手の初期局面から

			UctSearch(pos);
		}

		// 並列UCT探索
		// pos : この局面から探索する。
		void UctSearch(Position& pos)
		{
			// 定跡DBにこの局面が登録されているか調べる。
			auto book_pos = book.find(pos);

		}


		// 指定された局面から、終局までplayout(対局)を試みる。
		// 返し値 :
		//   1 : 開始手番側の勝利
		//   0 : 引き分け
		//  -1 : 開始手番側でないほうの勝利
		int Playout(Position& pos)
		{
			// 開始時の手番
			auto rootColor = pos.side_to_move();

			// do_move()時に使うStateInfoの配列
			auto states = make_unique<StateInfo[]>((size_t)(max_ply + 1));

			// 自己対局
			while (pos.game_ply() <= max_ply)
			{
				// 宣言勝ち。現在の手番側の勝ち。
				if (pos.DeclarationWin())
					return pos.side_to_move() == rootColor ? 1 : -1;

				// 探索深さにランダム性を持たせることによって、毎回異なる棋譜になるようにする。
				//int search_depth = depth_min + (int)prng.rand(depth_max - depth_min + 1);
				//auto pv = Learner::search(pos, search_depth, 1);

				//Move m = pv.second[0];
				Move m;

				// 指し手が存在しなかった。現在の手番側の負け。
				if (m == MOVE_NONE)
					return pos.side_to_move() == rootColor ? -1 : 1;

				pos.do_move(m, states[pos.game_ply()]);
			}

			// 引き分け
			return 0;
		}

	private:
		// 定跡DB本体
		MemoryBook book;

		// 最大手数
		int max_ply;

		// Learner::search()の探索深さを変えるための乱数
		AsyncPRNG prng;

		// MemoryBookのsaveに対するmutex。
		mutex book_mutex;
#endif
	};
}

namespace Book
{
	// 2019年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2021(Position& pos, istringstream& is, const string& token)
	{
		if (token == "mcts")
		{
			MctsMakeBook mcts;
			mcts.make_book(pos, is);
			return 1;
		}

		return 0;
	}
}

#endif // defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
