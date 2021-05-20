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
#include <limits>
#include <cmath>

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

	// 訪問回数をカウントする型
	// 42億を超えることはしばらくないだろうからとりあえずこれで。
	typedef u32 MoveCountType;

	// MCTSのnodeを表現する。
	struct MctsNode
	{
		// ======= 探索部を呼び出した結果 =======

		float		value;	// 探索した時のこの局面の評価値(手番側から見た値)
		Move16		move;	// 探索した時のこの局面の最善手(MOVE_RESIGN,MOVE_WINもありうる)

		// =======      MCTSした結果      =======

		MoveCountType	move_count;				// このノードの訪問回数。
		MoveCountType	move_count_from_this;	// このノードから訪問した回数
		double	value_win;				// このノードの勝率(手番側から見たもの) 累積するのでdoubleでないと精度足りない。

		// このnodeの初期化。新しくnodeを生成する時は、これを呼び出して初期化しなければならない。
		void init(float value_ , Move16 move_)
		{
			value                = value_;
			move                 = move_;

			move_count           = 0;
			move_count_from_this = 0;
			value_win            = 0.0;
		}
	};

	// unordered_mapに直接SearchResultを持たせたいのだが、そうすると、このコンテナに対してinsert()したときなどにコンテナ要素のアドレスが変わってしまうので
	// アドレスで扱うわけにいかない。そこで、object IDのようなものでこの部分を抽象化する必要がある。
	// ここでは、SearchResultをvectorにして、このコンテナの何番目の要素であるか、というのをNodeIndex型で示すというようにしてある。

	// ↓これを引換券にして、↑のinstanceがもらえたりする。
	typedef u32 NodeIndex;

	// NodeIndexの無効値
	constexpr NodeIndex NULL_NODE = std::numeric_limits<NodeIndex>::max();

	// 探索結果DBの1record。
	// 探索結果DBのdefaultファイル名 : "mctsbook_mcts_tree.db"
	// 32 + (4+2)+(4+4+8) = 54 byte
	struct MctsTreeRecord
	{
		PackedSfen	packed_sfen;	// 局面をPositionクラスでpackしたもの。32 bytes固定長。
		MctsNode	node;
	};

	// =================================================================
	//                        MCTSの探索木
	// =================================================================

	// 子ノードへのedgeと子ノードを表現する構造体
	struct MctsChild
	{
		// 子ノードへ到達するための指し手
		Move move;

		// 子ノードのindex(これを元にSearchResult*が確定する)
		NodeIndex index;

		MctsChild(Move move_, NodeIndex index_) :
			move(move_), index(index_) {}
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
	class NodeManager
	{
	public:

		// 読み込んでいるMctsNodeのclear
		void clear_nodes()
		{
			node_index_to_children_index.clear();
			childrens.clear();
		}

		// MtsChildrenのcacheをclear
		void clear_cache()
		{
			node_index_to_children_index.clear();
			childrens.clear();
		}

		// ファイルから読み込み。
		void Deserialize(const std::string& filename) {
			clear_nodes();
			clear_cache();

			cout << "deserialize from " << filename << endl;

			SystemIO::BinaryReader reader;
			if (reader.open(filename).is_not_ok())
			{
				cout << "Error! : file not found , filename = " << filename << endl;
				return;
				// まあ、初回はファイルは存在しないのでこれは無視する。
			}

			// ファイルサイズ
			size_t file_size = reader.get_size();

			// 格納されているレコード数
			size_t record_num = file_size / sizeof(MctsTreeRecord);

			// レコード数は確定しているので、この数の分だけ事前に配列を確保したほうが高速化する。
			mcts_nodes.reserve(record_num);
			sfen_to_index.reserve(record_num);

			// 1/100読み込みごとに"."を出力する。そのためのcounter。
			size_t progress = 0;

			for (size_t i = 0; i < record_num; ++i)
			{
				MctsTreeRecord record;
				if (reader.read(&record, sizeof(MctsTreeRecord)).is_not_ok())
				{
					cout << "Error! : file read error , filename = " << filename << " , record at " << i << endl;
					return; // 途中までは読み込めているかも知れないし、これも無視する。
				}

				// このrecordの内容をこのclassで保持している構造体に追加。

				// 重複していないことは保証されているはずなのでここではチェックしない。
				// (定跡ファイルのmergeについてはまた別途実装する(かも))
				size_t last_index = mcts_nodes.size();
				sfen_to_index[record.packed_sfen] = (NodeIndex)last_index;
				mcts_nodes.emplace_back(record.node);

				if (100 * i / record_num >= progress)
				{
					++progress;
					cout << ".";
				}
			}
			cout << "done" << endl;
		}

		// ファイルに保存。
		void Serialize(const std::string& filename) {

			cout << "serialize to " << filename << endl;

			SystemIO::BinaryWriter writer;
			if (writer.open(filename).is_not_ok())
			{
				cout << "Error! : file open error , filename = " << filename << endl;
				return;
			}

			// 格納すべきレコード数
			size_t record_num = mcts_nodes.size();

			// 1/100読み込みごとに"."を出力する。そのためのcounter。
			size_t progress = 0;

			// mcts_nodes.size()分だけループをぶん回す場合、それに対応するpacked sfenが取得しにくいので、
			// unordered_mapであるsfen_to_index側をiteratorで回す。こちらならpacked_sfenが取得できる。
			size_t i = 0;
			for (const auto &it : sfen_to_index)
			{
				MctsTreeRecord record;
				record.packed_sfen = it.first;
				record.node = mcts_nodes[it.second];

				writer.write(&record, sizeof(MctsTreeRecord));

				// 進捗の表示
				if (100 * i / record_num >= progress)
				{
					++progress;
					cout << ".";
				}
				++i;
			}
			cout << "done" << endl;
		}

		// 要素ひとつ追加する
		void append(const PackedSfen& ps , const MctsNode& node) {
			// この要素を持っていてはならない。
			ASSERT_LV3(sfen_to_index.find(ps) == sfen_to_index.end());

			NodeIndex index = (NodeIndex)mcts_nodes.size();
			mcts_nodes.emplace_back(node);
			sfen_to_index[ps] = index;
		}

		// indexを指定してMctsNode*を得る。
		// このpointerはappend()やgo_search()などが呼び出されるまでは有効。
		// index == NULL_NODE のときは、nullptrが返る。
		MctsNode* get_node(NodeIndex index)
		{
			if (index == NULL_NODE)
				return nullptr;
			return &mcts_nodes[(size_t)index];
		}

		// 指定されたsfenに対応するindexを得る。
		// 格納されていなければNULL_NODEが返る。
		NodeIndex get_index(const PackedSfen& ps) {
			auto it = sfen_to_index.find(ps);
			if (it == sfen_to_index.end())
				return NULL_NODE;
			return it->second;
		}

		// sfenで指定した局面を探索して、この構造体の集合のなかに追加する。
		NodeIndex* go_search(std::string& sfen , u64 nodes_limit)
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

		// === MctsChildrenのcache ===

		// あるNodeIndexに対応する、(そのnodeの)MctsChildrenを返す。
		// cacheされていなければchildrenを生成して返す。
		// ここで返されたpointerは、次にclear_children()かget_children()が呼び出されるまで有効。
		// ChildNodeは、まだ存在しないものに関しては、NULLNODEになっている。
		MctsChildren* get_children(Position& pos , NodeIndex node_index)
		{
			auto it = node_index_to_children_index.find(node_index);
			if (it == node_index_to_children_index.end())
			{
				PackedSfen sfen;

				// すべての合法手を生成する。
				MoveList<LEGAL_ALL> ml(pos);

				MctsChildren children;
				children.reserve(ml.size());

				for (auto m : ml)
				{
					// この指し手で一手進めて、その局面のNodeIndexを取得する。
					StateInfo si;
					pos.do_move(m,si);

					pos.sfen_pack(sfen);
					NodeIndex index = get_index(sfen);
					// なければindex == NULL_NODEになる
					pos.undo_move(m);

					children.emplace_back(MctsChild(m,index));
				}
				NodeIndex lastIndex = (NodeIndex)childrens.size();
				childrens.emplace_back(children);
				node_index_to_children_index[node_index] = lastIndex;
			}
			return &childrens[it->second];
		}

	private:

		// PackedSfenから、NodeIndexへのmap
		std::unordered_map<PackedSfen, NodeIndex, PackedSfenHash> sfen_to_index;

		// MctsNodeをシリアルに並べた型
		// NodeIndex index;
		// mcts_nodes[index] = xx
		// のようにアクセスできる。
		std::vector<MctsNode> mcts_nodes;

		// === MctsChildrenのcache ===

		// MctsChildrenをcacheしておく。
		// MCTSの探索時に毎回、指し手生成をして1手進めて、探索済みのchild nodeであるかを確認して1手戻す..みたいなことを繰り返したくないため)

		// ↓の何番目に格納されているのかの対応関係のmap
		std::unordered_map<NodeIndex, MctsChildrenIndex> node_index_to_children_index;

		// cacheされているMctsChildren集合。
		std::vector<MctsChildren> childrens;

		// あるNodeIndexに対応するMctsChildrenを格納する。
		void append_children(NodeIndex index, const MctsChildren& children)
		{
			// すでに格納されているなら、これ以上重複して格納することはできない。
			//ASSERT_LV3(get_children(index) == nullptr);

			auto children_index = (MctsChildrenIndex)children.size();
			childrens.emplace_back(children);
			node_index_to_children_index[index] = children_index;
		}

	};

	// =================================================================
	//                   MCTSの探索rootを表現する
	// =================================================================

	// 探索Root集合の型
	typedef std::vector<std::string> RootSfens;

	// positionコマンドに渡す形式で書かれているsfenファイルを読み込み、そのsfen集合を返す。
	// 重複局面は除去される。
	//   all_node   : そこまでの手順(経由した各局面)も含めてすべて読み込む。
	//   ignore_ply : 手数を無視する。(sfen化するときに手数をtrimする)
	// 
	// file formatは、
	// "startpos move xxxx xxxx"
	// "[sfen文字列] moves xxxx xxxx"
	RootSfens ReadPositionFile(const string& filename, bool all_node, bool ignore_ply)
	{
		std::unordered_set<std::string> sfens;

		SystemIO::TextReader reader;
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
			// 定跡ファイル名
			//string book_filename = "mctsbook_user_book.db";
			// → やねうら王で使う定跡DBのファイル
			// これは専用の変換コマンドを別途用意。

			// loop_maxに達するか、"stop"が来るまで回る

			// この回数だけrootから探索したら終了する。
			u64 loop_max = 10000000;

			// 一つの局面(leaf node)でのPlayoutの回数
			u64 playout = 10000;

			// 探索root集合
			string root_filename = "mctsbook_root_sfen.txt";

			// 定跡のMCTS探索中のメモリ状態をserializeしたやつ
			string mcts_tree_filename = "mctsbook_mcts_tree.db";

			// rootのうち、最小手数と最大手数。
			// root_min_ply <= game_ply <= root_max_ply に該当する局面しかrootの対象としない。
			int root_min_ply = 1;
			int root_max_ply = 256;

			// 最大手数
			int max_ply = 384;

			// 途中セーブ機能はないが、1時間で終わるぐらいの量にして、
			// 延々とこのコマンドを呼び出すことを想定。

			Parser::ArgumentParser parser;
			parser.add_argument("loop"         ,loop_max);
			parser.add_argument("playout"      , playout);
			parser.add_argument("root_filename", root_filename);
			parser.add_argument("tree_filename", mcts_tree_filename);
			parser.add_argument("root_min_ply" , root_min_ply);
			parser.add_argument("root_max_ply" , root_max_ply);
			parser.add_argument("max_ply"      , max_ply);
			parser.parse_args(is);

			cout << "makebook mcts command" << endl
				<< "  root filename       = " << root_filename << endl
				<< "  mcts_tree_filename  = " << mcts_tree_filename << endl
				<< "  loop                = " << loop_max << endl
				<< "  playout             = " << playout << endl
				<< "  playout             = " << playout << endl
				<< "  root_min_ply        = " << root_min_ply << endl
				<< "  root_max_ply        = " << root_max_ply << endl
				<< "  max_ply             = " << max_ply << endl
				;

			cout << "read root file.." << endl;

			// そこまでの手順も含めてすべて読み込み、sfenにする。
			auto roots = ReadPositionFile(root_filename, true, true);
			cout << "  root sfens size()  = " << roots.size() << endl;

			// treeの復元
			node_manager.Deserialize(mcts_tree_filename);

			// 定跡生成部本体
			for (size_t sfen_no = 0 ; sfen_no < roots.size() ; ++sfen_no)
			{
				// 探索開始局面(root)のsfenを出力
				auto root_sfen = roots[sfen_no];
				cout << "root sfen [" << (sfen_no+1) << "/" << roots.size() << "] : " << root_sfen << endl;

				// 探索開始局面から、loopで指定された回数だけMCTSでのplayoutを行う。
				for (u64 loop = 0; loop < loop_max; ++loop)
				{
					uct_main(root_sfen);
				}
			}

			// treeの保存
			node_manager.Serialize(mcts_tree_filename);

			cout << "makebook mcts , done." << endl;
		}

		// 与えられたsfen文字列の局面からMCTSしていく。
		void uct_main(const string& root_sfen)
		{
			Position pos;
			StateInfo si;
			pos.set(root_sfen, &si, Threads.main());

			// ここからMCTSしていく。
			uct_search(pos);
		}

		// 与えられた局面から最良の子nodeを選択していき、leaf nodeに到達したら探索を呼び出す。
		void uct_search(Position& pos)
		{
			PackedSfen packed_sfen;
			pos.sfen_pack(packed_sfen);

			NodeIndex node_index = node_manager.get_index(packed_sfen);
			if (node_index == NULL_NODE)
			{
				// 未展開のnodeであったか。playoutする
			}

			MctsNode* this_node = node_manager.get_node(node_index);

			// ↑↑でplayoutしているので、this_node == nullptrはありえない
			ASSERT_LV3(this_node != nullptr);

			MctsChildren* children = node_manager.get_children(pos, node_index);

			if (children == nullptr)
			{
				// 合法手がない。詰みなのでは…。
				return;
			}

			// === UCB1で最良ノードを選択する。===

			// value_winの最大値
			double ucb1_max_value = std::numeric_limits<double>::min();

			// 最良の子のindex。children[best_child]がbestな子
			size_t best_child = std::numeric_limits<size_t>::max();

			// このノードのmove_count
			u32 M = this_node->move_count_from_this;

			for (size_t i = 0; i < children->size(); ++i)
			{
				// ChildNodeへのedge
				MctsChild child = children->at(i);

				MctsNode* child_node = child.index != NULL_NODE ? node_manager.get_node(child.index) : nullptr;

				// 子のvalue_win。未展開のnodeなら親nodeの価値を使う。
				double child_value_win = child_node ? child_node->value_win : this_node->value_win;

				// 子のmove_count
				MoveCountType child_move_count = child_node ? child_node->move_count : 0;
				// 子のmove_countは分母に来るので、0割防止に微小な数を加算しておく。
				double N = child_move_count ? child_move_count : 0.01;

				// === UCB1値を計算する ===

				// 典型的なucb1の計算
				double child_ucb1 = child_value_win / N + sqrt(2.0 * log(M) / N );

				// ucb1値を更新したらそれを記憶しておく。
				if (ucb1_max_value < child_ucb1)
				{
					ucb1_max_value = child_ucb1;
					best_child = i;
				}
			}
			// best childが決定した

			// TODO:あとでかく
		}

		// "makebook mcts"コマンドで生成したtreeをやねうら王形式の定跡DBに変換する。
		void convert_book(Position& pos, istringstream& is)
		{
			// 定跡のMCTS探索中のメモリ状態をserializeしたやつ
			string mcts_tree_filename = "mctsbook_mcts_tree.db";

			// 定跡ファイル名
			// → やねうら王で使う定跡DBのファイル
			string book_filename = "mctsbook_user_book.db";

			// この回数以上訪問しているnodeでなければ定跡に書き出さない。
			u32 min_move_count = 1000;

			Parser::ArgumentParser parser;
			parser.add_argument("tree_filename" , mcts_tree_filename);
			parser.add_argument("book_filename" , book_filename);
			parser.add_argument("min_move_count", min_move_count);
			parser.parse_args(is);

			cout << "makebook mcts_convert command" << endl
				<< "  mcts_tree_filename  = " << mcts_tree_filename << endl
				<< "  book_filename       = " << book_filename << endl
				<< "  min_move_count      = " << min_move_count
				;

			// treeの読み込み
			node_manager.Deserialize(mcts_tree_filename);

			// 全件のなかから、条件の合致するnodeだけ書き出す。


			// 書きかけ
		}


	private:
		NodeManager node_manager;
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
		if (token == "mcts_convert")
		{
			// やねうら王で使う定跡ファイルの形式に変換する(書き出す)コマンド
			MctsMakeBook mcts;
			mcts.convert_book(pos, is);
			return 1;
		}


		return 0;
	}
}

#endif // defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
