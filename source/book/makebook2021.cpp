#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) &&  defined(YANEURAOU_ENGINE_DEEP)
// いまのところ、ふかうら王のみ対応。
// 探索させた局面のすべての指し手の訪問回数(≒評価値)が欲しいので、この定跡生成ルーチンはNNUEとは相性良くない。

// ------------------------------
//  スーパーテラショック定跡手法
// ------------------------------
/*
	スーパーテラショックコマンド
		> makebook s_tera

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
		book.db
		→　このあと、このファイルをテラショック化コマンドでテラショック定跡化して使う。

		例)
		> makebook build_tree book.db user_book1.db


	高速化のために、局面をSFEN文字列ではなく、局面のhash値で一致しているかどうかをチェックするので
	局面のhash値は衝突しないように128bit以上のhashを用いる。(べき)
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
#include "../mate/mate.h"

using namespace std;
using namespace Book;

// positionコマンドのparserを呼び出したい。
extern void position_cmd(Position& pos, istringstream& is, StateListPtr& states);

namespace dlshogi {
	// 探索結果を返す。
	//   Threads.start_thinking(pos, states , limits);
	//   Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。
	// のようにUSIのgoコマンド相当で探索したあと、rootの各候補手とそれに対応する評価値を返す。
	extern std::vector < std::pair<Move, float>> GetSearchResult();
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

namespace MakeBook2021
{
	// Node(局面)からその子ノードへのedgeを表現する。
	struct Child
	{
		// 子ノードへ至るための指し手
		// メモリを節約するためにMove16にする。
		Move16 move;

		// その時の評価値(親局面から見て)
		int16_t eval;

		// Node*を持たせても良いが、メモリ勿体ない。
		// HASH_KEYで管理しておき、実際にdo_move()してHASH_KEYを見ることにする。

		Child() {}
		Child(Move m, Value e) : move(m), eval((int16_t)e) {}
	};

	// 局面を1つ表現する構造体
	struct Node
	{
		// 局面のSFEN文字列
		string sfen;

		// 子ノードへのEdgeの集合。
		vector<Child> children;

		// この局面の評価値(この定跡生成ルーチンで探索した時の)
		// 未探索ならVALUE_NONE
		// childrenのなかで最善手を指した時のevalが、この値。
		Value search_value = VALUE_NONE;

		// ↑のsearch_valueは循環を経て(千日手などで)得られたスコアであるかのフラグ。
		// これがfalseであり、かつ、search_value!=VALUE_NONEであるなら、
		// このNodeをleaf node扱いして良い。(探索で得られたスコアなので)
		bool is_cyclic = false;

		// この局面での最善手。保存する時などには使わないけど、PV表示したりする時にあれば便利。
		Move bestmove = MOVE_NONE;
	};

	// 局面管理用クラス。
	// HASH_KEYからそのに対応するNode構造体を取り出す。(あるいは格納する)
	// Node構造体には、その局面の全合法手とその指し手を指した時の評価値が格納されている。
	class PositionManager
	{
	public:

		// 定跡ファイルを読み込む
		void read_book(string filename)
		{
			MemoryBook book;
			book.read_book(filename);

			Position pos;

			cout << "book examination";

			// それぞれの局面を列挙
			StateInfo si;
			u64 count = 0;
			book.foreach([&](string sfen, BookMovesPtr ptr) {
				pos.set(sfen,&si,Threads.main());

				store(sfen, pos, ptr);

				// 進捗を表現するのに1000局面ごとに"."を出力。
				if (++count % 1000 == 0)
					cout << ".";
				});

			cout << endl << "readbook DB has completed." << endl;
		}

		// SFEN文字列とそれに対応する局面の情報(定跡の指し手、evalの値等)を
		// このクラスの構造体に保存する。
		void store(string sfen, Position& pos, const BookMovesPtr& ptr)
		{
			// Node一つ作る。
			Node& node = *create_node(pos);

			node.sfen = sfen;

			// この局面の定跡の各指し手に対応する評価値を保存していく。
			unordered_map<Move, Value> move_to_eval;
			(*ptr).foreach([&](BookMove& bm) {
				Move move = pos.to_move(bm.move);
				move_to_eval[move] = (Value)bm.value;
				});

			// 全合法手の生成と、この局面の定跡の各手の情報を保存。
			MoveList<LEGAL_ALL> ml(pos);
			for (auto move : ml)
			{
				// 指し手moveがBookMovesのなかにあるなら、その評価値も取得して登録
				// 愚直にやると計算量はO(N^2)になるのでmoveに対応するevalのmapを使う。
				Value eval = VALUE_NONE;
				auto it = move_to_eval.find(move);
				if (it != move_to_eval.end())
					eval = it->second;
				node.children.emplace_back(Child(move,eval));
			}
		}

		// Nodeを新規に生成して、そのNode*を返す。
		Node* create_node(Position& pos)
		{
			auto key = pos.long_key();
			auto r = hashkey_to_node.emplace(key, Node());
			auto& node = r.first->second;
			return &node;
		}

		// 引数で指定されたkeyに対応するNode*を返す。
		// 見つからなければnullptrが返る。
		Node* probe(HASH_KEY key)
		{
			auto it = hashkey_to_node.find(key);
			return it == hashkey_to_node.end() ? nullptr : &it->second;
		}

		// 指定された局面の情報を表示させてみる。(デバッグ用)
		void dump(Position& pos)
		{
			auto key = pos.long_key();

			auto* node = probe(key);
			if (node == nullptr)
			{
				cout << "Position HashKey , Not Found." << endl;
			}
			else
			{
				cout << "sfen      : " << node->sfen << endl;
				cout << "child_num : " << node->children.size() << endl;

				for (auto& child : node->children)
				{
					string eval = (child.eval == VALUE_NONE) ? "none" : to_string(child.eval);
					cout << " move : " << child.move << " eval : " << eval << endl;
				}
			}
		}

		// 指定された局面からPVを辿って表示する。(デバッグ用)
		// search_start()を呼び出してPVが確定している必要がある。
		void dump_pv(Position& pos)
		{
			auto key = pos.long_key();

			auto* node = probe(key);
			if (node == nullptr)
				return; // これ以上辿れなかった。

			Move m = node->bestmove;
			cout << m << " ";

			// bestmoveで進められるんか？
			StateInfo si;
			pos.do_move(m,si);
			dump_pv(pos);
			pos.undo_move(m);
		}

		// posで指定された局面から"PV line = ... , Value = .."の形でPV lineを出力する。
		// vは引数で渡す。
		void dump_pv_line(Position& pos , Value v)
		{
			cout << "PV line = ";
			dump_pv(pos);
			cout << ", Value = " << v << endl;
		}

		// 持っているNode情報をすべて定跡DBに保存する。
		// path : 保存する定跡ファイルのpath
		void save_book(string path)
		{
			MemoryBook book;
			for (auto& it : hashkey_to_node)
			{
				auto& node = it.second;
				auto sfen = node.sfen;

				BookMovesPtr bms(new BookMoves);

				for (auto& child : node.children)
				{
					// 値が未定。この指し手は書き出す価値がない。
					if (child.eval == VALUE_NONE)
						continue;

					// テラショック化するからponder moveいらんやろ
					bms->push_back(BookMove(child.move, /*ponder*/ MOVE_NONE , child.eval,/*depth*/32,/*move_count*/ 1));
				}

				book.append(sfen,bms);
			}

			book.write_book(path);
		}

		// 各Nodeのsearch_valueを初期化(VALUE_NONE)にする。
		// 探索の前に初期化しておけば、訪問済みのNodeかどうかがわかる。
		void clear_all_search_value()
		{
			for (auto& it : hashkey_to_node)
			{
				auto& node = it.second;

				node.search_value = VALUE_NONE;
				node.is_cyclic = false;
			}
		}

	protected:
		// その局面のHASH_KEYからNode構造体へのmap。
		unordered_map<HASH_KEY, Node> hashkey_to_node;
	};


	// 探索モード。SearchOptionで用いる。
	enum class SearchType
	{
		AlphaBeta,       // 通常のnaiveなαβ探索
		AlphaBetaRange,  // leaf nodeのevalが区間[alpha,beta) であるものをすべて列挙するためのαβ探索
		AlphaBetaOnKif,  // 与えられた棋譜上の局面でかつ、区間[alpha,beta) であるものをすべて列挙するためのαβ探索
	};

	// 探索オプション
	// search_startを呼び出す時に用いる。
	struct SearchOption
	{
		// 探索モード
		SearchType search_type;

		// この局面数だけ思考するごとにファイルに保存する。
		u64 book_save_interval;

		// 定跡を保存するファイル名。
		string write_book_name;

		// ↑のファイル、一時保存していくときのナンバリング
		int write_book_number = 0;

		// 1局面の探索ノード数
		u64 nodes_limit;

		// ranged alpha beta探索を行う時の、best_valueとの差。
		// best_value = 70のときにsearch_delta = 10であれば、60～70の評価値となるleaf nodeを
		// 選出して、それを延長(思考対象として思考)する。
		int search_delta;

		// ranged alpha beta探索した時のleaf nodeで棋譜上に出現したleaf nodeを選出して延長する。
		// その時の、best_valueとの差。
		// best_value = 70ときにsearch_delta_on_kif = 30であれば、40～70の評価値となるleaf nodeでかつ
		// 棋譜上に出現した局面だけを思考対象とする。
		int search_delta_on_kif;

		// 勝率から評価値に変換する時の定数
		float eval_coef;
	};

	// あとで思考すべきNode
	struct SearchNode
	{
		// あとで探索すべき局面のNode
		Node* node;

		// そこで探索すべき指し手(この指し手で1手進めた局面を探索する)
		Move16 move;

		SearchNode() {}
		SearchNode(Node* node_, Move16 move_) :node(node_), move(move_) {}

		// このクラスの指している局面のSFENと指し手を出力する。(デバッグ用)
		void print() const
		{
			cout << node->sfen << " : " << move << endl;
		}

		// 文字列化する
		string to_string() const
		{
			return "SFEN = " + node->sfen + " , move = " + to_usi_string(move);
		}
	};

	// あとで探索すべきNode集。
	typedef vector<SearchNode> SearchNodes;

	// スーパーテラショック定跡手法で定跡を生成するルーチン
	// "makebook stera read_book book1.db write_book book2.db"
	class SuperTeraBook
	{
	public:
		void make_book(Position& pos, istringstream& is)
		{
			cout << endl;
			cout << "makebook stera command : " << endl;

			// HashKey、64bit超えを推奨。(これのbit数が少ないとhash衝突してしまう)
			// コンパイルするときに、config.hの HASH_KEY_BITS のdefineを128か256にすることを推奨。
			if (HASH_KEY_BITS < 128)
				cout << "[WARNING!] It is recommended that HASH_KEY_BITS is 128-bits or higher." << endl;

			string read_book_name = "book/read_book.db";
			string write_book_name = "book/write_book.db";

			// 探索開始rootとなるsfenの集合。ファイルがなければ平手の初期局面をrootとする。
			string root_sfens_name = "root_sfens.txt";

			// 探索に追加したい棋譜
			// USIの"position"コマンドの、"position"の後続に来る文字列(これが1つの棋譜)が複数行(複数棋譜)書かれているようなファイル。
			string kif_sfens_name = "kif_sfens.txt";

			// ↓の局面数を思考するごとにsaveする。
			u64 book_save_interval = 1000;

			// 1局面に対する探索ノード数
			u64 nodes_limit = 3000;
			
			// 探索局面数
			u64 think_limit = 10000;

			// ranged alpha beta探索を行う時の、best_valueとの差。
			// best_value = 70のときにsearch_delta = 10であれば、60～70の評価値となるleaf nodeを
			// 選出して、それを延長(思考対象として思考)する。
			// 0を指定すると、この工程を常にskipする。
			int search_delta = 10;

			// ranged alpha beta探索した時のleaf nodeで棋譜上に出現したleaf nodeを選出して延長する。
			// その時の、best_valueとの差。
			// best_value = 70ときにsearch_delta_on_kif = 30であれば、40～70の評価値となるleaf nodeでかつ
			// 棋譜上に出現した局面だけを思考対象とする。
			// 0を指定すると、この工程を常にskipする。
			int search_delta_on_kif = 30;

			// このコマンドの流れとしては、
			// 1) 通常のalpha-beta探索でbest valueを確定させる
			// 2) ranged alpha-beta探索 区間[best_value - search_delta , best_value+1)
			//    でleaf nodeを列挙。(それが思考対象局面)
			// 3) ranged alpha-beta探索 区間[best_value - search_delta_on_kif , best_value+1)
			//    で列挙されたleaf nodeかつ、与えられたKIF上に出現した局面を列挙。(それが思考対象局面)
			// root_sfenで与えた局面に対して、1)～3)を繰り返して、think_limitの局面数だけ思考したら終了。

			Parser::ArgumentParser parser;
			parser.add_argument("read_book"           , read_book_name);
			parser.add_argument("write_book"          , write_book_name);
			parser.add_argument("book_save_interval"  , book_save_interval);
			parser.add_argument("nodes_limit"         , nodes_limit);
			parser.add_argument("think_limit"         , think_limit);
			parser.add_argument("search_delta"        , search_delta);
			parser.add_argument("search_delta_on_kif" , search_delta_on_kif);
			parser.add_argument("root_sfens_name"     , root_sfens_name);
			parser.add_argument("kif_sfens_name"      , kif_sfens_name);

			parser.parse_args(is);

			cout << "ReadBook  DB file    : " << read_book_name       << endl;
			cout << "WriteBook DB file    : " << write_book_name      << endl;
			cout << "book_save_interval   : " << book_save_interval   << "[" << "[positions]" << "]" << endl;
			cout << "nodes_limit          : " << nodes_limit          << endl;
			cout << "think_limit          : " << think_limit          << endl;
			cout << "search_delta         : " << search_delta         << endl;
			cout << "search_delta_on_kif  : " << search_delta_on_kif  << endl;
			cout << "root_sfens_name      : " << root_sfens_name      << endl;
			cout << "kif_sfens_name       : " << kif_sfens_name       << endl;
			cout << endl;

			// 定跡ファイルの読み込み。
			pm.read_book(read_book_name);

			// root sfen集合の読み込み
			vector<string> root_sfens;
			if (SystemIO::ReadAllLines(root_sfens_name, root_sfens, true).is_not_ok())
			{
				// ここで読み込むroot集合のsfenファイルは、
				// sfen XXX moves YY ZZ..
				// startpos moves VV WW..
				// のようなUSIの"position"コマンドとしてやってくるような棋譜形式であれば
				// いずれでも解釈できるものとする。

				// なければ無いで良いが..
				cout << "Warning , root sfens file = " << root_sfens_name << " , is not found." << endl;

				// なければ平手の初期局面を設定しておいてやる。
				root_sfens.emplace_back("sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
			}

			vector<string> kif_sfens;
			if (SystemIO::ReadAllLines(kif_sfens_name, kif_sfens, true).is_not_ok())
			{
				// なければ無いで良いが..
				cout << "Warning , kif file = " << kif_sfens_name << " , is not found." << endl;

				// この時、棋譜に出現する局面を探索する工程はスキップする
				search_delta_on_kif = 0;
			}
			else {
				// 棋譜に出現した局面を this->kif_hashに追加する。
				// (これはコマンド実行時の最初に1度行えば良い)
				vector<StateInfo> si(1024 /* above MAX_PLY */);
				for (auto kif : kif_sfens)
					set_root(pos, kif, si, true);
			}

			// 与えられたパラメーターをsearch_optionに反映。
			search_option.book_save_interval   = book_save_interval;
			search_option.write_book_name      = write_book_name;
			search_option.nodes_limit          = nodes_limit;
			search_option.search_delta         = search_delta;
			search_option.search_delta_on_kif  = search_delta_on_kif;
			search_option.eval_coef            = (float)Options["Eval_Coef"];

			// USIオプションを探索用に設定する。
			set_usi_options();

			// 準備完了したのであとは
			// think limitに達するまで延々と思考する。

			while (this->think_count < think_limit)
			{
				// 与えられたroot局面集合から。
				for (auto& root_sfen : root_sfens)
				{
					// rootとなる局面のsfenを出力しておく。
					cout << "[Step 1.] search root = " << root_sfen << endl;

					// 指定されたrootから定跡を生成する。
					make_book_from_root(pos, root_sfen);
				}
			}

			// 定跡ファイルの書き出し(最終)
			save_book(write_book_name);

			// コマンドが完了したことを出力。
			cout << "makebook stera command has finished." << endl;
		}

		// root_sfenで与えられたsfen(もしくは"moves"以下指し手がついているような棋譜の場合、その末尾の局面)をposに設定する。
		//
		// root_sfenとして、
		// "startpos"
		// "sfen XXX moves YY ZZ.."
		// "startpos moves VV WW.."
		// のようなUSIの"position"コマンドとしてやってくるような棋譜形式であれば
		// いずれでも解釈できるものとする。
		// 
		// append_to_kif = trueのとき、その局面のhash keyがkif_hashにappendされる。
		void set_root(Position& pos, const string& root_sfen, vector<StateInfo>& si , bool append_to_kif = false)
		{
			// -- makebook2015.cppから拝借

			// issから次のtokenを取得する
			auto feed_next = [](istringstream& iss)
			{
				string token = "";
				iss >> token;
				return token;
			};

			// "sfen"に後続するsfen文字列をissからfeedする
			auto feed_sfen = [&feed_next](istringstream& iss)
			{
				stringstream sfen;

				// ループではないが条件外であるときにbreakでreturnのところに行くためのhack
				while(true)
				{
					string token;

					// 盤面を表すsfen文字列
					sfen << feed_next(iss);

					// 手番
					token = feed_next(iss);
					if (token != "w" && token != "b")
						break;
					sfen << " " << token;

					// 手駒
					sfen << " " << feed_next(iss);

					// 初期局面からの手数
					sfen <<  " " << feed_next(iss);

					break;
				}
				return sfen.str();
			};

			// append_to_kif == trueならば現在の局面のhash keyを、kif_hashに追加する。
			auto append_to_kif_hash = [&]() {
				if (append_to_kif)
				{
					HASH_KEY key = pos.long_key();
					if (kif_hash.find(key) == kif_hash.end())
						kif_hash.insert(key);
				}
			};

			si.clear();
			si.emplace_back(StateInfo()); // このあとPosition::set()かset_hirate()を呼び出すので一つは必要。

			istringstream iss(root_sfen);
			string token;
			bool hirate = true;
			do {
				token = feed_next(iss);
				if (token == "sfen")
				{
					// 駒落ちなどではsfen xxx movesとなるのでこれをfeedしなければならない。
					auto sfen = feed_sfen(iss);
					pos.set(sfen,&si[0],Threads.main());
					hirate = false;
				}
			} while (token == "startpos" || token == "moves" || token == "sfen");

			if (hirate)
				pos.set_hirate(&si[0],Threads.main());

			append_to_kif_hash();

			// moves以降は1手ずつ進める
			while (token != "")
			{
				Move move = USI::to_move(pos, token);
				if (move == MOVE_NONE)
					break;

				if (!pos.pseudo_legal_s<true>(move) || !pos.legal(move))
				{
					cout << "Error : illegal move : sfen = " << root_sfen << " , move = " << token << endl;
					// もうだめぽ
					token = "";
				}
				si.emplace_back(StateInfo());
				pos.do_move(move, si.back());

				// 棋譜hashに追加。
				append_to_kif_hash();

				iss >> token;
			}

			// 局面posは指し手で進めて、与えられたsfen(棋譜)の末尾の局面になっている。
		}

		// 現在のrootから定跡を生成する。
		void make_book_from_root(Position& pos, const string& root_sfen)
		{
			// 局面の初期化
			vector<StateInfo> si(1024 /* above MAX_PLY */);
			set_root(pos, root_sfen , si);

			// ----------------------------------------
			//      Normal Alpha-Beta Search
			// ----------------------------------------

			// まず通常のαβ探索を行う。

			cout << "[Step 2.] normal alpha beta search [-INF,INF)" << endl;
			search_option.search_type = SearchType::AlphaBeta;
			Value best_value = search_start(pos, -VALUE_INFINITE, VALUE_INFINITE);

			// 探索で得られたPV lineの表示。
			pm.dump_pv_line(pos,best_value);

			// ----------------------------------------
			//      Ranged Alpha-Beta Search
			// ----------------------------------------

			// 区間[best_value - delta , best_value + 1) で探索しなおす。
			// 条件に合致するleaf nodeをすべて列挙して、search_nodes(思考対象局面)に追加する。
			// delta = 10ぐらいで十分だと思う。
			// 
			// ranged alpha-beta searchでは、alpha値のupdateは行わない。常に最初に渡された区間内を探索。
			// 前回のbestvalueを与えているのでここからはみ出すことはないはず。

			if (search_option.search_delta != 0)
			{
				cout << "[Step 3.] ranged alpha beta search ["
					 << (best_value - search_option.search_delta) << ", " << best_value << "]" << endl;
				search_option.search_type = SearchType::AlphaBetaRange;
				search_start(pos, best_value - search_option.search_delta, best_value + 1);
				// 探索結果にさきほどのleaf nodeが現れないことがある。(局面の循環などで)
				// 仕方ないので、deltaを少し広げる。
				if (search_nodes.size() == 0)
					search_option.search_delta += 5;
				think_nodes(pos);
			}

			// ----------------------------------------
			//    Ranged Alpha-Beta Search With Kif
			// ----------------------------------------

			// 区間[best_value - delta , best_value + 1) で探索しなおし、leaf nodeでかつ棋譜上に出現した局面を
			// search_nodes(思考対象局面)に追加する。
			// delta = 30ぐらいでいいと思う。

			if (search_option.search_delta_on_kif != 0)
			{
				cout << "[Step 4.] alpha beta search on kif ["
					 <<  (best_value - search_option.search_delta_on_kif) << "," << best_value  << "]" << endl;
				search_option.search_type = SearchType::AlphaBetaOnKif;
				search_start(pos, best_value - search_option.search_delta_on_kif, best_value + 1);
				think_nodes(pos);
			}
		}

		// search_nodesを思考対象局面として、
		// それらについて思考する。
		void think_nodes(Position& pos)
		{
			// 思考対象局面の数を出力
			cout << "think nodes = " << search_nodes.size() << endl;

			// 列挙された局面を思考対象として思考させる。
			for (auto& p : search_nodes)
				think(pos, &p);
		}

		// 探索させて、そのNodeを追加する。
		// 結果はpmにNode追加して書き戻される。
		Node* think(Position& pos, const SearchNode* s_node)
		{
			string sfen = s_node->node->sfen;
			Move16 m16 = s_node->move;
			Move move = pos.to_move(m16);

			// 思考中の局面を出力してやる。
			cout << "thinking : " << sfen << " , move = " << move << endl;

			return think_sub(pos , sfen + " moves " + to_usi_string(m16));
		}

		// SFEN文字列を指定して、その局面について思考させる。
		// 結果はpmにNode追加して書き戻される。
		Node* think(Position& pos, string sfen)
		{
			// 思考中の局面を出力してやる。
			cout << "thinking : " << sfen << endl;

			return think_sub(pos,sfen);
		}

		// 探索の入り口。
		Value search_start(Position& pos, Value alpha, Value beta)
		{
			// RootNode
			Node* node = pm.probe(pos.long_key());
			if (node == nullptr)
			{
				// これが存在しない時は、生成しなくてはならない。
				node = think(pos,pos.sfen());
			}

			// 探索前に各nodeのsearch_valueはクリアする必要がある。
			pm.clear_all_search_value();

			// 思考対象局面もクリアする必要がある。
			search_nodes.clear();

			// αβ探索ルーチンへ
			return search(pos, node, alpha, beta , 1);
		}

		// αβ探索
		// plyはrootからの手数。0ならroot。
		Value search(Position& pos, Node* node, Value alpha, Value beta , int ply)
		{
			// このnodeで得られたsearch_scoreは、循環が絡んだスコアなのか
			node->is_cyclic = true;

			// 手数制限による引き分け

			int game_ply = pos.game_ply();
			if (game_ply >= 512)
				return VALUE_DRAW; // 最大手数で引き分け。

			// 千日手の処理

			auto draw_type = pos.is_repetition(game_ply /* 千日手判定のために遡れる限り遡って良い */);
			if (draw_type != REPETITION_NONE)
				// draw_value()はデフォルトのままでいいや。すなわち千日手引き分けは VALUE_ZERO。
				return draw_value(draw_type, pos.side_to_move());

			// Mate distance pruning.
			//
			// この処理を入れておかないと長いほうの詰み手順に行ったりするし、
			// 詰みまで定跡を掘るにしても長い手順のほうを悪い値にしたい。
			// 
			// また、best value = -INFの時に、そこから - deltaすると範囲外の数値になってしまうので
			// それの防止でもある。

			alpha = std::max(mated_in(ply    ), alpha);
			beta  = std::min(mate_in (ply + 1), beta );
			if (alpha >= beta)
				return alpha;

			node->is_cyclic = false;

			ASSERT_LV3(node != nullptr);

			Move bestmove = MOVE_NONE;
			Value value;
			// 引数で渡されたalphaの値
			Value old_alpha = alpha;

			// すでに探索済みでかつ千日手が絡まないスコアが記入されている。
			if (node->search_value != VALUE_NONE)
				return node->search_value;

			// 現局面で詰んでますが？
			// こんなNodeまで掘ったら駄目でしょ…。
			if (node->children.size() == 0 /*pos.is_mated()*/)
				return mated_in(0);

			// 宣言勝ちの判定
			if (pos.DeclarationWin())
				// 次の宣言で勝ちになるから1手詰め相当。
				return mate_in(1);

			// 1手詰めもついでに判定しておく。
			if (Mate::mate_1ply(pos))
				return mate_in(1);

			// 上記のいずれかの条件に合致した場合、このnodeからさらに掘ることは考えられないので
			// このnodeでは何も情報を記録せずに即座にreturnして良い。

			// main-loop

			// 各合法手で1手進める。
			for (auto& child : node->children)
			{
				Move16 m16 = child.move;
				Move m = pos.to_move(m16);
				bool is_cyclic = false;

				// 今回の指し手によって、到達したleaf nodeのうち、
				// alpha <= eval < beta なnodeのものだけ残していく。
				size_t search_nodes_index = search_nodes.size();

				// 指し手mで進めた時のhash key。
				HASH_KEY key_next = pos.long_key_after(m);
				Node* next_node = pm.probe(key_next);
				if (next_node == nullptr)
				{
					// 見つからなかったので、この子nodeのevalがそのまま評価値
					// これがVALUE_NONEだとしたら、その定跡ファイルの作りが悪いのでは。
					// 何らかの値は入っているべき。
					// 
					// すなわち、定跡の各局面で全合法手に対して何らかの評価値はついているべき。
					// 評価値がついていない時点で、この局面、探索しなおしたほうが良いと思う。
					value = (Value)child.eval;
					if (value == VALUE_NONE)
						continue;

					// AlphaBetaOnKifの時)
					// いまnext_node == nullptrだが、この時のkey_nextが、
					// 棋譜上に出現している(kif_hash上に存在する)なら、
					// この局面を追加する。さもなくば追加しない。
					// 
					// AlphaBeta , RangedAlphaBetaの時)
					// 未探索のleafをここで無条件で追加して良い。
					//
					bool append_cancel = (search_option.search_type == SearchType::AlphaBetaOnKif) // AlphaBetaOnKif探索モードにおいて
						              && (kif_hash.find(key_next) == kif_hash.end()); // 棋譜上に出現しないleaf nodeである

					if (!append_cancel)
						search_nodes.emplace_back(SearchNode(node, m16));

				}
				else {
					// alpha-beta searchなので 1手進めてalphaとbetaを入れ替えて探索。
					StateInfo si;
					pos.do_move(m, si);

					if (search_option.search_type == SearchType::AlphaBeta)
						// 通常のalpha-beta探索
						value = -search(pos, next_node, -beta, -alpha, ply+1);
					else
						// alpha値は更新しない感じのalpha-beta探索
						// 区間[alpha,beta)に収まるleaf nodeをすべて展開したいので。
						value = -search(pos, next_node, -beta, -old_alpha, ply+1);

					// 子ノードのスコアが循環が絡んだスコアであるなら、このnodeにも伝播すべき。
					is_cyclic = next_node->is_cyclic;
					pos.undo_move(m);

					// なぜかvalue_none
					if (value == VALUE_NONE)
						continue;
				}

				// alpha値を更新するのか？
				if (alpha < value)
				{
					// alpha値を更新するのに用いた子nodeに関するis_cyclicだけをこのノードに伝播させる。
					node->is_cyclic |= is_cyclic;

					// beta以上ならbeta cutが生じる。
					if (beta <= value)
						return value;

					// update alpha
					alpha = value;
					bestmove = m;
				}
				else
				{
					// 今回、alpha値を更新しなかった。

					// SearchType::AlphaBetaの場合)
					// 
					// 今回得たleaf nodeの集合は不要な集合であったと言える。
					// そこで、search_nodesを巻き戻す。
					// 
					// SearchType::AlphaBetaRange
					// SearchType::AlphaBetaOnKif
					// の場合)
					// 引数で渡されたalpha値(old_alpha)未満のvalueであるleaf nodeは思考対象としない。
					// そこで、value < old_alphaならsearch_nodesを巻き戻す。

					if (search_option.search_type == SearchType::AlphaBeta || value < old_alpha)
						search_nodes.resize(search_nodes_index);
				}
			}

			// このループを回ったということは、search_valueがVALUE_NONE(未代入状態)であったか、
			// VALUE_NONEではないがcyclicな状態であったかのいずれかだから、今回の結果を代入してしまって良い。

			node->search_value = alpha;
			node->bestmove = bestmove;

			// bestmoveが見つかったので、それが1番目ではないなら入れ替えておくべきか？
			// まあ、普通に探索した場合、1番目の指し手が1番目に来ているだろうから、まあいいか…。

			return alpha;
		}

		// 現在の局面について思考して、その結果をpmにNode作って返す。
		Node* think_sub(Position& pos,string sfen)
		{
			// ================================
			//        Limitsの設定
			// ================================

			Search::LimitsType limits = Search::Limits;

			// ノード数制限
			limits.nodes = search_option.nodes_limit;

			// すべての合法手を生成する
			limits.generate_all_legal_moves = true;

			// 探索中にPVの出力を行わない。
			limits.silent = true;

			// ================================
			//           思考開始
			// ================================

			// SetupStatesは破壊したくないのでローカルに確保
			StateListPtr states(new StateList(1));

			// sfen文字列、Positionコマンドのparserで解釈させる。
			istringstream is("sfen " + sfen);
			position_cmd(pos, is, states);

			// すでにあるのでskip
			Node* n = pm.probe(pos.long_key());
			if (n != nullptr)
				return n;

			// 思考部にUSIのgoコマンドが来たと錯覚させて思考させる。
			Threads.start_thinking(pos, states , limits);
			Threads.main()->wait_for_search_finished();
			
			// ================================
			//        探索結果の取得
			// ================================

			auto search_result = dlshogi::GetSearchResult();

			// 新規にNodeを作成してそこに書き出す

			Node& node = *pm.create_node(pos);
			node.sfen = pos.sfen();

			// 合法手の数
			size_t num = search_result.size();
			node.children.reserve(num);

			for (auto& r : search_result)
			{
				Move m = r.first;

				// 勝率
				float wp = r.second;

				// 勝率を[centi-pawn]に変換
				Value cp = Eval::dlshogi::value_to_cp((float)wp,search_option.eval_coef);

				// Nodeのchildrenに追加。
				node.children.emplace_back(Child(m, cp));
			}

			// Childのなかで一番評価値が良い指し手がbestmoveであり、
			// その時の評価値がこのnodeの評価値。

			Value best_value = -VALUE_INFINITE;
			size_t max_index = -1;
			for (size_t i = 0; i < num; ++i)
			{
				Value eval = (Value)node.children[i].eval;
				if (best_value < eval)
				{
					best_value = eval;
					max_index = i;
				}
			}
			// 普通は合法手、ひとつは存在するはずなのだが…。
			if (max_index != -1)
			{
				node.bestmove = pos.to_move(node.children[max_index].move);

				// この指し手をchildrentの先頭に持ってきておく。(alpha-beta探索で早い段階で枝刈りさせるため)
				std::swap(node.children[0], node.children[max_index]);
			}

			// ================================
			//   定跡ファイルの定期的な書き出し
			// ================================

			if (++think_count % search_option.book_save_interval == 0)
			{
				string path = search_option.write_book_name
					// ゼロサプライして6桁にする。
					// "write_book.db.000001"
					// みたいな感じ。
					+ "." +StringExtension::to_string_with_zero(search_option.write_book_number++,6);

				save_book(path);
			}

			return &node;
		}

		// 定跡をファイルに保存する。
		void save_book(const string& path )
		{
			cout << "save book , path = " << path << endl;
			pm.save_book(path);
		}

		// USIオプションを探索用に設定する。
		void set_usi_options()
		{
			// 定跡にhitされると困るので定跡なしに
			Options["BookFile"] = "no_book";

			// rootでdf-pnで詰みが見つかると候補手が得られないので
			// とりまオフにしておく。
			Options["RootMateSearchNodesLimit"] = "0";
		}

	private:

		// 局面管理クラス
		PositionManager pm;

		// 探索時のオプション
		SearchOption search_option;

		// あとで思考すべき局面集
		// search()の時に列挙していく。
		SearchNodes search_nodes;

		// 思考した局面数のカウンター
		u64 think_count = 0;

		// 棋譜上に出現した局面のhash key
		unordered_set<HASH_KEY> kif_hash;
	};

}

namespace Book
{
	// 2019年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2021(Position& pos, istringstream& is, const string& token)
	{
		// makebook steraコマンド
		// スーパーテラショック定跡手法での定跡生成。
		if (token == "stera")
		{
			MakeBook2021::SuperTeraBook st;
			st.make_book(pos, is);
			return 1;
		}

		return 0;
	}
}

#endif // defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
