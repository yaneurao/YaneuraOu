#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) &&  defined(YANEURAOU_ENGINE_DEEP)
// いまのところ、ふかうら王のみ対応。
// 探索させた局面のすべての指し手の訪問回数(≒評価値)が欲しいので、この定跡生成ルーチンはNNUEとは相性良くない。

// ------------------------------
//  スーパーテラショック定跡手法
// ------------------------------
/*
	スーパーテラショック定跡コマンド
	// ふかうら王のみ使えるコマンド。

	> makebook s_tera

	パラメーター)

		read_book         : 前回書きだした定跡ファイル名(デフォルトでは "book/read_book.db")
		write_book        : 今回書き出す定跡ファイル名  (デフォルトでは "book/write_book.db")
		root_sfens_name   : 探索開始局面集のファイル名。デフォルトでは"book/root_sfens.txt"。このファイルがなければ平手の初期局面から。
			// 平手の初期局面をrootとすると後手番の定跡があまり生成されないので(先手の初手は26歩か34歩を主な読み筋としてしまうので)
			// 初期局面から1手指した局面をこのファイルとして与えたほうがいいかも？
			// あと駒落ちの定跡を生成するときは同様に駒落ちの初期局面と、そこから1手指した局面ぐらいをこのファイルで指定すると良いかも。
		kif_sfens_name    : 棋譜のファイル名。この棋譜上の局面かつPVまわりを思考させることもできる。
		book_save_interval: この局面数だけ思考するごとに定跡ファイルを書き出す。
			// "book/read_book.db.000001"のようなファイルに通しナンバー付きで書き出していく。
			// このコマンドが終了するときにも書き出すので、数時間に1度保存される程度のペースでも良いと思う。
		nodes_limit       : 1局面について思考するnode数。30knps出るPCで30000を指定すると1局面1秒。
		think_limit       : この局面数思考したらこのコマンドを終了する。
		search_delta_on_kif : ranged alpha beta searchの時に棋譜上に出現したleaf nodeに加点するスコア。
							そのleaf nodeが選ばれやすくなる。

	それぞれのfileフォーマット

	入力)
		book/root_sfens.txt

		// 探索開始局面
		例)
		sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
		↑のようなsfen形式か、
		startpos moves 7g7f ...
		のようなUSIプロトコルの"position"コマンドとして送られる形式でも可。
		// この場合、それぞれの局面が探索開始対象となる。

		book/kif_sfens.txt
		読み込ませたい棋譜。この棋譜上の局面かつ、PVまわりを思考する。
		例)
		startpos moves 7g7f ...
		// このファイルはなければなくとも良い。

		book/read_book.db
		前回出力された出力ファイル(book/write_book.db)をread_book.dbとrenameしたもの。
		// 配置するとその続きから掘ってくれる。
		(初回はこのファイルは不要)

	出力)
		book/write_book.db
		→　このあと、このファイルをテラショック化コマンドでテラショック定跡化して使う。

		例)
		> makebook build_tree book/write_book.db book/user_book1.db

	注意点)

		高速化のために、局面をSFEN文字列ではなく、局面のhash値で一致しているかどうかをチェックするので
		局面のhash値は衝突しないように128bit以上のhashを用いる。(べき)

		→　具体的には、config.hでHASH_KEY_BITSのdefineを128にしてビルドする。

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
#include "../engine/dlshogi-engine/dlshogi_min.h"

using namespace std;
using namespace Book;
using namespace Concurrent; // concurrent library from misc.h

// positionコマンドのparserを呼び出したい。
extern void position_cmd(Position& pos, istringstream& is, StateListPtr& states);

namespace MakeBook2021
{
	// ValueとDepthが一体になった型
	struct ValueDepth
	{
		ValueDepth() {}
		ValueDepth(Value value_) : value((s16)value_), depth(0) {}
		ValueDepth(Value value_, u16 depth_) : value((s16)value_), depth((u16)depth_) {}

		// 評価値
		s16 value;

		// rootからの手数
		u16 depth;

		// --- 算術演算子

		const ValueDepth operator+() const { return *this; }
		const ValueDepth operator-() const { return ValueDepth(Value(-value), depth); }

		// --- 比較演算子その1

		constexpr bool operator == (const ValueDepth& rhs) const { return value == rhs.value && depth == rhs.depth; }
		constexpr bool operator != (const ValueDepth& rhs) const { return !(*this == rhs); }

		// --- static const.

		static ValueDepth mate_in(u16 ply) { return ValueDepth(::mate_in(ply), ply); }
		static ValueDepth mated_in(u16 ply) { return ValueDepth(::mated_in(ply), ply); }

	};

	// --- 比較演算子その2

	constexpr bool operator<(const ValueDepth& t1, const ValueDepth& t2)
	{
		// valueが同じ時は、plyが大きい方が良い評価値扱いをする。
		// →　そうしてしまうと先後協力して飛車を動かして手数を伸ばした千日手にすることになって、探索が終わらなくなるので駄目だった。

		return t1.value < t2.value
			/*|| (t1.value == t2.value && t1.depth < t2.depth)*/;
	}
	constexpr bool operator>(const ValueDepth& t1, const ValueDepth& t2) { return t2 < t1; }
	constexpr bool operator<=(const ValueDepth& t1, const ValueDepth& t2) { return !(t1 > t2); }
	constexpr bool operator>=(const ValueDepth& t1, const ValueDepth& t2) { return !(t1 < t2); }

	// Node(局面)からその子ノードへのedgeを表現する。
	struct Child
	{
		// 子ノードへ至るための指し手
		// メモリを節約するためにMove16にする。
		Move16 move;

		// その時の評価値(親局面から見て)
		ValueDepth eval;

		// Node*を持たせても良いが、メモリ勿体ない。
		// HASH_KEYで管理しておき、実際にdo_move()してHASH_KEYを見ることにする。

		Child() {}
		Child(Move m, ValueDepth e) : move(m), eval(e) {}
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
		ValueDepth search_value;

		// ↑のsearch_valueは循環を経て(千日手などで)得られたスコアであるかのフラグ。
		// これがfalseであり、かつ、search_value!=VALUE_NONEであるなら、
		// このNodeをleaf node扱いして良い。(探索で得られたスコアなので)
		// 何手前からの循環であるか。(最大でも127)
		// 0なら、循環ではない。
		s8 cyclic;

		// この局面での最善手。保存する時などには使わないけど、PV表示したりする時にあれば便利。
		Move best_move;

		// ↑search_value、best_moveの性質。
		Bound bound;

		// このクラスのメンバー変数の世代。
		// search_value,cyclic,best_move は、generationが合致しなければ無視する。
		// 全域に渡って、Nodeの↑の3つの変数をクリアするのが大変なのでgenerationカウンターで管理する。
		u32 generation = 0;

		// childrenのなかから、bestを先頭に持ってくる。
		void set_best_move(Move16 best_move)
		{
			auto it = std::find_if(children.begin(), children.end(), [&](const Child& c) { return c.move == best_move; });
			if (it == children.end())
				return; // そんな馬鹿な…

			// 見つかった以上は、要素が1つ以上存在するはずで、children[0]が存在することは保証されているから
			// それを指定してswapすることはできる。
			std::swap(children[0], *it);
		}
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

			cout << "book examination" << endl;

			Tools::ProgressBar progress(book.size());
			u64 counter = 0;

			// それぞれの局面を列挙
			StateInfo si;
			u64 count = 0;
			book.foreach([&](string sfen, BookMovesPtr ptr) {
				pos.set(sfen, &si, Threads.main());

				store(sfen, pos, ptr);

				// 進捗の出力
				progress.check(++counter);
				});
		}

		// SFEN文字列とそれに対応する局面の情報(定跡の指し手、evalの値等)を
		// このクラスの構造体に保存する。
		// ただしすべての合法手の評価値がついていない局面は登録しない。
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
			for (auto m : ml)
			{
				Move move = m.move;
				// 指し手moveがBookMovesのなかにあるなら、その評価値も取得して登録
				// 愚直にやると計算量はO(N^2)になるのでmoveに対応するevalのmapを使う。
				Value eval;
				auto it = move_to_eval.find(move);
				if (it != move_to_eval.end())
					eval = it->second;
				else
				{
					// 合法手すべてに評価値がついていないとまずい。
					// このnodeはなかったことにしてしまう。
					sync_cout << "remove node , sfen = " << sfen << " , missing move = " << move << sync_endl;
					remove_node(pos);
					return;
				}
				node.children.emplace_back(Child(move, ValueDepth(eval, 0)));
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

		// Nodeを削除する。
		void remove_node(Position& pos)
		{
			auto key = pos.long_key();
			hashkey_to_node.erase(key);
		}

		// probe()とcreate_node()だけmutexを用いたいので、
		// create_nodeを呼び出す前後でこのmutexをlockすればprobe()から守られる。
		std::mutex* get_mutex() { return &mutex_; }

		// [ASYNC] 引数で指定されたkeyに対応するNode*を返す。
		// 見つからなければnullptrが返る。
		Node* probe(HASH_KEY key)
		{
			std::unique_lock<std::mutex> lk(mutex_);
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
					string eval = (child.eval.value == VALUE_NONE) ? "none" : to_string(child.eval.value) + " depth : " + to_string(child.eval.depth);
					cout << " move : " << child.move << " eval : " << eval << endl;
				}
			}
		}

		// 指定された局面からPVを辿って表示する。(デバッグ用)
		// search_start()を呼び出してPVが確定している必要がある。
		void dump_pv(Position& pos, int ply = 0)
		{
			// 千日手などで循環する可能性があるのでその防止。
			if (ply >= MAX_PLY)
				return;

			auto key = pos.long_key();

			auto* node = probe(key);
			if (node == nullptr)
				return; // これ以上辿れなかった。

			Move m = node->best_move;
			cout << m << " ";

			// best_moveで進められるんか？
			StateInfo si;
			pos.do_move(m, si);
			dump_pv(pos, ply + 1);
			pos.undo_move(m);
		}

		// posで指定された局面から"PV line = ... , Value = .."の形でPV lineを出力する。
		// vは引数で渡す。
		void dump_pv_line(Position& pos, ValueDepth v)
		{
			std::cout << IO_LOCK;
			cout << "PV line = ";
			dump_pv(pos);
			cout << ", Value = " << v.value << " , Depth = " << v.depth << endl;
			std::cout << IO_UNLOCK;
		}

		// 持っているNode情報をすべて定跡DBに保存する。
		// path : 保存する定跡ファイルのpath
		void save_book(string path)
		{
			sync_cout << "making save file in memory" << sync_endl;

			// 進捗出力用
			Tools::ProgressBar progress(hashkey_to_node.size());
			u64 counter = 0;

			MemoryBook book;
			for (auto& it : hashkey_to_node)
			{
				auto& node = it.second;
				auto sfen = node.sfen;

				BookMovesPtr bms(new BookMoves);

				for (auto& child : node.children)
				{
					// 値が未定。この指し手は書き出す価値がない。
					if (child.eval.value == VALUE_NONE)
						continue;

					// テラショック化するからponder moveいらんやろ
					bms->push_back(BookMove(child.move, /*ponder*/ MOVE_NONE, child.eval.value,/*depth*/ child.eval.depth,/*move_count*/ 1));
				}

				book.append(sfen, bms);

				progress.check(++counter);
			}

			book.write_book(path);
		}

		// 持っているNode情報をすべて定跡DBに保存する。
		// メモリ上の定跡はテラショック化されているものとする。
		// path : 保存する定跡ファイルのpath
		void save_tera_book(string path)
		{
			sync_cout << "making save file in memory" << sync_endl;

			// 進捗出力用
			Tools::ProgressBar progress(hashkey_to_node.size());
			u64 counter = 0;

			Position pos;
			MemoryBook book;
			for (auto& it : hashkey_to_node)
			{
				auto& node = it.second;
				auto sfen = node.sfen;

				BookMovesPtr bms(new BookMoves);

				for (auto& child : node.children)
				{
					// 値が未定。この指し手は書き出す価値がない。
					// child.evalには、その先のleaf nodeでの評価値が反映されているものとする。
					if (child.eval.value == VALUE_NONE)
						continue;

					Move16 move = child.move;
					Move16 ponder = MOVE_NONE;

					// このnodeから1手進めて、次の局面のbestを広い、ponderを確定させる。
					StateInfo si;
					pos.set(sfen, &si, Threads.main());
					HASH_KEY next_key = pos.long_key_after(pos.to_move(move));
					auto it = hashkey_to_node.find(next_key);
					if (it != hashkey_to_node.end())
						ponder = it->second.best_move;

					// depthとしてchild.eval.plyを埋めておく。
					bms->push_back(BookMove(move, ponder, child.eval.value, /* depth */child.eval.depth,/*move_count*/ 1));
				}

				book.append(sfen, bms);

				progress.check(++counter);
			}

			book.write_book(path);
		}

		// 各Nodeのsearch_valueを初期化(VALUE_NONE)にする。
		// 探索の前に初期化しておけば、訪問済みのNodeかどうかがわかる。
		// → generationで管理することにしたので、実際はこの関数は使わない。
		void clear_all_search_value()
		{
			for (auto& it : hashkey_to_node)
			{
				auto& node = it.second;

				node.search_value = ValueDepth(VALUE_NONE, 0);
				node.cyclic = 0;
			}
		}

	protected:
		// その局面のHASH_KEYからNode構造体へのmap。
		unordered_map<HASH_KEY, Node> hashkey_to_node;

		std::mutex mutex_;
	};

	// 探索中のnodeを表現する。
	struct SearchingNodes
	{
		// [ASYNC] nodeのkeyを追加する
		void append(HASH_KEY key)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			keys.emplace(key);
		}

		// [ASYNC] nodeのkeyを保持しているか。保持していればtrue。
		bool contains(HASH_KEY key)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			return keys.find(key) != keys.end();
		}

		// [ASYNC] nodeのkeyを削除する。
		void remove(HASH_KEY key)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			if (keys.find(key) != keys.end()) // ここcontains()を用いるとmutex二重にlockされてstaleする。
				keys.erase(key);
		}

		// [ASYNC] keyを丸ごとclearする。
		void clear()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			keys.clear();
		}

		// [ASYNC] 要素の数を返す。
		size_t size()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			return keys.size();
		}

	private:
		std::mutex mutex_;
		std::unordered_set<HASH_KEY> keys;
	};

	// 探索モード
	enum SearchMode
	{
		NormalAlphaBeta, // 通常のAlphaBeta探索
		TeraShockSearch, // テラショック化する時のAlphaBeta探索
	};

	// 探索オプション
	// search_startを呼び出す時に用いる。
	struct SearchOption
	{
		// この局面数だけ思考するごとにファイルに保存する。
		u64 book_save_interval;

		// 定跡を保存するファイル名。
		string write_book_name;

		// ↑のファイル、一時保存していくときのナンバリング
		int write_book_number = 0;

		// 1局面の探索ノード数
		u64 nodes_limit;

		// ranged alpha betaを連続して行う回数。
		size_t ranged_alpha_beta_loop;

		// ranged alpha beta searchの時に棋譜上に出現したleaf nodeに加点するスコア。
		// そのleaf nodeが選ばれやすくなる。
		int search_delta_on_kif;

		// 勝率から評価値に変換する時の定数
		float eval_coef;

		// 世代カウンター。
		// Nodeのメンバー変数にアクセスする時に世代が一致しない変数は無効扱いする。
		u32 generation = 0;

		// 探索モード
		// search関数の挙動が変わる。
		SearchMode search_mode = SearchMode::NormalAlphaBeta;
	};

	// あとで思考すべきNode
	struct SearchNode
	{
		// あとで探索すべき局面のNode
		Node* node;

		// そこで探索すべき指し手(この指し手で1手進めた局面を探索する)
		Move16 move;

		// nodeからmoveで1手進めた局面のhash key。
		HASH_KEY key;

		SearchNode() {}
		SearchNode(Node* node_, Move16 move_, HASH_KEY key_) :node(node_), move(move_), key(key_) {}

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

		// operator==を設定しとかないとunordered_setで使えない。
		bool operator==(const SearchNode& rhs) const
		{
			return node == rhs.node && move == rhs.move && key == rhs.key;
		}
	};
}

	// これ、templateの特殊化なのでnamespace MakeBook2021のなかで定義できない。
	// この提案はrejectされたのかな…。→　http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3730.html
	template <>
	struct std::hash<MakeBook2021::SearchNode> {
		size_t operator()(const MakeBook2021::SearchNode& sn) const {
			// nodeと指し手の組み合わせがuniqueっぽくなってほしい。
			return (size_t)((size_t)sn.node ^ ((u32)sn.move.to_u16() << 16));
		}
	};

namespace MakeBook2021 {

	// 探索node type
	enum NodeType { PV, NonPV };

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

			// 探索開始rootとなるsfenの集合。ファイルがなければ平手の初期局面(先手、後手)をrootとする。
			string root_sfens_name = "book/root_sfens.txt";

			// 探索に追加したい棋譜
			// USIの"position"コマンドの、"position"の後続に来る文字列(これが1つの棋譜)が複数行(複数棋譜)書かれているようなファイル。
			string kif_sfens_name = "book/kif_sfens.txt";

			// 1局面に対する探索ノード数
			// 1局面1秒ぐらい
			u64 nodes_limit = 30000;
			//u64 nodes_limit = 100;
			
			// ↓の局面数を思考するごとにsaveする。
			// 15分に1回ぐらいで良いような？
			// 定跡ファイルが大きくなってきたら数時間に1回でいいと思う。
			u64 book_save_interval = 30000/*nps*/ / nodes_limit * 30*60 /* 30分 */;

			// 探索局面数
			u64 think_limit = 10000000;

			// 1つのroot局面に対して、何回ranged alpha searchを連続して行うのか。
			// このloop回数分は、Nodeの値を信じるかどうかを判定するためのgenerationが
			// 変わらないので探索効率が良い。
			u64 ranged_alpha_beta_loop = 100;

			// ranged alpha beta searchの時に棋譜上に出現したleaf nodeに加点するスコア。
			// そのleaf nodeが選ばれやすくなる。
			int search_delta_on_kif = 100;

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
			parser.add_argument("ranged_alpha_beta_loop", ranged_alpha_beta_loop);
			parser.add_argument("search_delta_on_kif" , search_delta_on_kif);
			parser.add_argument("root_sfens_name"     , root_sfens_name);
			parser.add_argument("kif_sfens_name"      , kif_sfens_name);

			parser.parse_args(is);

			cout << "ReadBook  DB file      : " << read_book_name         << endl;
			cout << "WriteBook DB file      : " << write_book_name        << endl;
			cout << "book_save_interval     : " << book_save_interval     << "[positions]" << endl;
			cout << "nodes_limit            : " << nodes_limit            << endl;
			cout << "think_limit            : " << think_limit            << endl;
			cout << "ranged_alpha_beta_loop : " << ranged_alpha_beta_loop << endl;
			cout << "search_delta_on_kif    : " << search_delta_on_kif    << endl;
			cout << "root_sfens_name        : " << root_sfens_name        << endl;
			cout << "kif_sfens_name         : " << kif_sfens_name         << endl;
			cout << endl;

			// 定跡ファイルの読み込み。
			pm.read_book(read_book_name);

			// root sfen集合の読み込み
			// このroot sfenとして棋譜を与えて棋譜の局面も全部掘りたい気はするが、そうしたい時とそうしたくない時があるので
			// それやるなら棋譜の全部のsfenを列挙するconverterみたいなの作った方がいいと思う。
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

				// なければ平手の初期局面と後手の初期局面を設定しておいてやる。
				// 初手86歩みたいな指し手は掘っても仕方がないと思うが、
				// そのへんをカスタマイズしたければ、root_sfens.txtを用意してそこで指定すればいい。

				root_sfens = BookTools::get_next_sfens("startpos");

				// 先手の初期局面
				root_sfens.emplace_back("startpos");
			}

			cout << "[Step 1] extract leaf nodes on kif." << endl;

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
			search_option.book_save_interval     = book_save_interval;
			search_option.write_book_name        = write_book_name;
			search_option.nodes_limit            = nodes_limit;
			search_option.ranged_alpha_beta_loop = ranged_alpha_beta_loop;
			search_option.search_delta_on_kif    = search_delta_on_kif;
			search_option.eval_coef              = (float)Options["Eval_Coef"];
			search_option.search_mode            = SearchMode::NormalAlphaBeta;

			// 各種変数の初期化(この関数を二度呼び出す時のために一応..)
			search_nodes.clear();
			searching_nodes.clear();
			banned_nodes.clear();

			// USIオプションを探索用に設定する。
			set_usi_options();

			// 準備完了したのであとは
			// think limitに達するまで延々と思考する。

			// rootに対応するNode自体は必要。これは特別扱いする。
			cout << "[Step 2] root nodes check." << endl;
			for (auto& root_sfen : root_sfens)
				create_root_node(pos, root_sfen);

			// 指定局面数に達するまでループ
			while (this->think_count < think_limit)
			{
				// 指定されたrootから定跡を生成する。
				make_book_from_roots(pos, root_sfens);
			}

			// 定跡ファイルの書き出し(最終)
			save_book(write_book_name);

			// コマンドが完了したことを出力。
			cout << "makebook stera command has finished." << endl;
		}

		// スーパーテラショック定跡手法で生成した定跡を、テラショック化する。
		// これにより、通常の思考エンジンで使える定跡になる。
		void stera_convert(Position& pos, istringstream& is)
		{
			cout << endl;
			cout << "makebook stera_convert command : " << endl;

			string read_book_name = "book/read_book.db";
			string write_book_name = "book/write_tera_book.db";

			// 探索開始rootとなるsfenの集合。ファイルがなければ平手、駒落ちの初期局面をrootとする。
			string root_sfens_name = "book/root_sfens.txt";

			Parser::ArgumentParser parser;
			parser.add_argument("read_book"           , read_book_name);
			parser.add_argument("write_book"          , write_book_name);
			parser.add_argument("root_sfens_name"     , root_sfens_name);

			parser.parse_args(is);

			cout << "ReadBook  DB file      : " << read_book_name         << endl;
			cout << "WriteBook DB file      : " << write_book_name        << endl;
			cout << "root_sfens_name        : " << root_sfens_name        << endl;

			// 定跡ファイルの読み込み。
			pm.read_book(read_book_name);

			// root sfen集合の読み込み
			vector<string> root_sfens;
			if (SystemIO::ReadAllLines(root_sfens_name, root_sfens, true).is_not_ok())
			{
				// なければ無いで良いが..
				cout << "Warning , root sfens file = " << root_sfens_name << " , is not found." << endl;

				// 平手、駒落ちの初期局面に対して…
				for (auto root_sfen : BookTools::get_start_sfens())
				{
					// その局面と
					root_sfens.emplace_back(root_sfen);

					// ↓そこから１手指した局面も含めておく。

					auto next_sfens = BookTools::get_next_sfens(root_sfen);
					root_sfens.insert(root_sfens.begin(),next_sfens.begin(),next_sfens.end());
				}
			}

			sync_cout << "root_sfens.size() == " << root_sfens.size() << sync_endl;

			vector<StateInfo> si(1024);
			for (auto root_sfen : root_sfens)
			{
				BookTools::feed_position_string(pos, root_sfen, si);

				// Node is not found in Book DB , skipped.
				if (pm.probe(pos.long_key()) == nullptr)
					continue;

				sync_cout << "[Step 1] set root , root sfen = " << root_sfen << sync_endl;

				sync_cout << "[Step 2] alpha-beta search from root." << sync_endl;

				// generationを変更。
				++search_option.generation;
				search_option.search_mode = TeraShockSearch;

				search_start(pos, -VALUE_INFINITE, VALUE_INFINITE);
			}

			// 定跡ファイルの書き出し(最終)
			save_tera_book(write_book_name);

			// コマンドが完了したことを出力。
			cout << "makebook stera_convert command has finished." << endl;
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
			// append_to_kif == trueならば現在の局面のhash keyを、kif_hashに追加する。
			auto append_to_kif_hash = [&](Position& pos) {
				if (append_to_kif)
				{
					HASH_KEY key = pos.long_key();
					if (kif_hash.find(key) == kif_hash.end())
						kif_hash.emplace(key);
				}
			};

			BookTools::feed_position_string(pos, root_sfen, si, append_to_kif_hash );

			// 局面posは指し手で進めて、与えられたsfen(棋譜)の末尾の局面になっている。
		}

		// 現在のrootから定跡を生成する。
		void make_book_from_roots(Position& pos, const vector<string>& root_sfens)
		{
			// ----------------------------------------
			//      Ranged Alpha-Beta Search
			// ----------------------------------------

			sync_cout << "[Step 3] ranged alpha beta search (-INF,INF)" << sync_endl;

			// 探索スレッドの起動
			auto th = std::thread([&] {
				// 与えられたroot局面集合から。
				for (auto& root_sfen : root_sfens)
				{
					// rootとなる局面のsfenを出力しておく。
					sync_cout << "[Step 4] search root = " << root_sfen << sync_endl;

					// 世代カウンターをインクリメント
					// 探索前に各nodeのsearch_valueはクリアする必要があるが、
					// generationが合致していなければ無視すれば良いのでクリアしないことにする。
					// (このカウンターがlap aroundした時には困るがu32なので、まあ…)
					//
					// これ、search_start()のごとに行ったほうが正確なのだが、そうすると
					// 毎回全域を探索することになって効率がすこぶる悪い。
					++search_option.generation;

					for (size_t i = 0; i < search_option.ranged_alpha_beta_loop; ++i)
					{
						// 局面の初期化
						vector<StateInfo> si(1024 /* above MAX_PLY */);
						Position root_pos;
						set_root(root_pos, root_sfen, si);

						// ここ、aspiration searchにしたほうが効率が良いのだろうが、
						// そこまで変わらない＆面倒なのでこのまま。
						ValueDepth best_value = search_start(root_pos, -VALUE_INFINITE, VALUE_INFINITE);

						// 初回のみ探索で得られたPV lineの表示。
						if (i == 0)
							pm.dump_pv_line(root_pos, best_value);

						// best valueを記録したnodeを思考するためのqueueに追加。

						// 無い。おそらく初期局面ですべての指し手がqueueに入っている、みたいな状況。
						if (search_pv.size() == 0)
						{
							// 空の指し手を積んでおく。
							// こうしないとpopする回数と数が合わなくてdead lockになる。
							search_nodes.push(SearchNode(nullptr,MOVE_NONE,HASH_KEY()));
							continue;
						}

						const auto& next = search_pv.back();

						sync_cout << "leaf node , sfen = " << next.node->sfen << " , move = " << next.move << sync_endl;

						// 思考中なのでここはleaf nodeとして回避せよ。
						searching_nodes.append(next.key);

						// 思考するためのqueueに追加。
						search_nodes.push(next);
					}
				}
			});

			// ↑のスレッドが局面を作って、queueに積んでくれるはず。
			Timer time;
			for (size_t i = 0; i < search_option.ranged_alpha_beta_loop * root_sfens.size(); ++i)
			{
				// 思考するための局面queueから取り出す。
				auto s_node = search_nodes.pop();
				// 空の指し手(該当がなかった)
				if (s_node.move == MOVE_NONE)
					continue;

				time.reset();
				bool already_exists,banned_node=false;
				think(pos,&s_node,already_exists);
				if (already_exists)
				{
					// 統計を取ってみて、2回以上ここに来たらsearching_nodesからのremoveはしないことにする。
					// おそらく千日手なので。

					if (!banned_nodes.contains(s_node))
						banned_nodes.emplace(s_node);

				} else {
					sync_cout << "think time = " << time.elapsed() << "[ms] , queue.size() = " << search_nodes.size() << sync_endl;
				}

				// 探索が終わったので、いま以降、このleaf nodeに来ても大丈夫！
				searching_nodes.remove(s_node.key);
			}

			// worker threadは自動的に終了するはずなのだが？
			th.join();

			// 参考のためbanされているnodeの数を出力しておく。
			if (banned_nodes.size())
				sync_cout << "banned nodes.size() = " << banned_nodes.size() << sync_endl;
		}

		// rootに対応するNodeを生成する。
		Node* create_root_node(Position& pos,string rootSfen)
		{
			StateInfo si;
			pos.set(rootSfen, &si, Threads.main());

			// RootNode
			Node* node = pm.probe(pos.long_key());
			if (node == nullptr)
			{
				// これが存在しない時は、生成しなくてはならない。
				bool already_exists;
				node = think(pos,rootSfen,already_exists);
			}
			return node;
		}

		// 探索の入り口。
		ValueDepth search_start(Position& pos, Value alpha, Value beta)
		{
			// RootNode
			Node* node = pm.probe(pos.long_key());

			// →　存在は保証されている。
			ASSERT_LV3(node != nullptr);

			// 探索PVのクリア
			search_pv.clear();

			node_searched = 0;
			search_timer.reset();

			u32 game_ply = pos.game_ply();

			// αβ探索ルーチンへ
			switch (search_option.search_mode)
			{
			case SearchMode::NormalAlphaBeta:
				return search<PV , SearchMode::NormalAlphaBeta>(pos, node, ValueDepth(alpha,game_ply), ValueDepth(beta,game_ply) , 1);

			case SearchMode::TeraShockSearch:
			{
				cout << endl; // progressの出力のため
				auto v = search<PV, SearchMode::TeraShockSearch>(pos, node, ValueDepth(alpha, game_ply), ValueDepth(beta, game_ply), 1);
				cout << endl; // progressを出力のため
				return v;
			}

			default: UNREACHABLE; break;
			}
		}

		// αβ探索
		// plyはrootからの手数。0ならroot。
		template <NodeType nodeType , SearchMode searchMode>
		ValueDepth search(Position& pos, Node* node, ValueDepth alpha, ValueDepth beta , int ply)
		{
			// ノートパソコンでnps 160k～600knpsほど。
			// デスクトップ機でnps 300k～1300knpsほど。
			if (searchMode == SearchMode::TeraShockSearch && (node_searched++ % 1000000 == 0))
				cout << ".";
			//sync_cout << "search nps = " << 1000 * node_searched / (search_timer.elapsed() + 1) << sync_endl;

			// すでに探索済みでかつ千日手が絡まないスコアが記入されている。
			// (generationが合致した時のみ)
			if (    nodeType == NonPV
				&&  node->generation == search_option.generation
				//&&  node->cyclic == 0
				// ↑この条件はわりときついかも…。
				&&  node->search_value.value != VALUE_NONE)
			{
				if (   node->bound == Bound::BOUND_EXACT
					||(node->bound == Bound::BOUND_LOWER && node->search_value >= beta /* beta cut*/)
					||(node->bound == Bound::BOUND_UPPER && node->search_value <= alpha /* 更新する可能性がない */)
					)
					return node->search_value;
			}

			// TeraShock searchの時は、PVであっても、cyclic == 0であれば枝刈りしていいと思う。
			// (通常searchの時は、PV leafが書き換わるのでこれをやるなら、前回のPV lineをクリアする必要がある)
			if (   searchMode == SearchMode::TeraShockSearch
				&& nodeType == PV
				&& node->generation == search_option.generation
				&& node->cyclic == 0
				&& node->search_value.value != VALUE_NONE)
			{
				if (   node->bound == Bound::BOUND_EXACT
					||(node->bound == Bound::BOUND_LOWER && node->search_value >= beta /* beta cut*/)
					||(node->bound == Bound::BOUND_UPPER && node->search_value <= alpha /* 更新する可能性がない */)
					)
					return node->search_value;
			}

			// =========================================
			//            nodeの初期化
			// =========================================

			// 手数制限による引き分け

			int game_ply = pos.game_ply();
			if (game_ply >= 512)
			{
				// このnodeで得られたsearch_scoreは、循環が絡んだスコアなのか
				node->cyclic = 127; // 循環は絡んでいないが経路には依存しうるので最大にしておく。
				return VALUE_DRAW; // 最大手数で引き分け。
			}

			// Mate distance pruning.
			//
			// この処理を入れておかないと長いほうの詰み手順に行ったりするし、
			// 詰みまで定跡を掘るにしても長い手順のほうを悪い値にしたい。
			// 
			// また、best value = -INFの時に、そこから - deltaすると範囲外の数値になってしまうので
			// それの防止でもある。

			// →　これ、plyは経路に依存していて、経路に依存した枝刈りなので
			// cyclic扱いしておく。

			alpha = std::max(ValueDepth::mated_in(ply    ), alpha);
			beta  = std::min(ValueDepth::mate_in (ply + 1), beta );
			if (alpha >= beta)
			{
				node->cyclic = ply; // 循環は絡んでいないが経路には依存しうる
				return alpha;
			}

			// 千日手の処理

			int found_ply;
			auto draw_type = pos.is_repetition(game_ply /* 千日手判定のために遡れる限り遡って良い */ , found_ply);
			if (draw_type != REPETITION_NONE)
			{
				// 何手前からの循環であるか
				node->cyclic = found_ply;

				// draw_value()はデフォルトのままでいいや。すなわち千日手引き分けは VALUE_ZERO。
				return draw_value(draw_type, pos.side_to_move());
			}

			node->cyclic = 0;

			ASSERT_LV3(node != nullptr);

			// 現局面で詰んでますが？
			// こんなNodeまで掘ったら駄目でしょ…。
			if (node->children.size() == 0 /*pos.is_mated()*/)
				return ValueDepth::mated_in(0);

			// 宣言勝ちの判定
			if (pos.DeclarationWin())
				// 次の宣言で勝ちになるから1手詰め相当。
				return ValueDepth::mate_in(1);

			// 1手詰めもついでに判定しておく。
			if (Mate::mate_1ply(pos))
				return ValueDepth::mate_in(1);

			// 上記のいずれかの条件に合致した場合、このnodeからさらに掘ることは考えられないので
			// このnodeでは何も情報を記録せずに即座にreturnして良い。

			// ===================================
			//             main-loop
			// ===================================

			// search_pvを巻き戻すためにこの関数の入り口でのsearch_pvの要素数を記録しておく。
			size_t search_pv_index1 = search_pv.size();

			Move best_move = MOVE_NONE;
			// alphaとは無縁に、bestな値は求めておく必要がある。
			ValueDepth best_value = -VALUE_INFINITE;

			ValueDepth old_alpha = alpha;

			// 各合法手で1手進める。
			for (auto& child : node->children)
			{
				ValueDepth value;

				Move16 m16 = child.move;
				Move m = pos.to_move(m16);
				s8 cyclic = 0;

				// search_pvを巻き戻すために1手進める前のsearch_pvの要素数を記録しておく。
				size_t search_pv_index2 = search_pv.size();

				// 指し手mで進めた時のhash key。
				const HASH_KEY key_next = pos.long_key_after(m);

				Node* next_node = pm.probe(key_next);

				// 探索中のleaf nodeは無いことにする。
				if (searching_nodes.contains(key_next))
					continue;

				SearchNode sn(node, m, key_next);
				// この局面でこの指し手mは千日手になるのでbanされている。
				if (banned_nodes.contains(sn))
					continue;

				// PVに追加
				search_pv.emplace_back(sn);

				if (next_node == nullptr)
				{
					// 見つからなかったので、この子nodeのevalがそのまま評価値
					// これがVALUE_NONEだとしたら、その定跡ファイルの作りが悪いのでは。
					// 何らかの値は入っているべき。
					// 
					// すなわち、定跡の各局面で全合法手に対して何らかの評価値はついているべき。
					// 評価値がついていない時点で、この局面、探索しなおしたほうが良いと思う。
					value = child.eval;

					ASSERT_LV3(value.value != VALUE_NONE);
					// →　不成を生成しないモードで定跡を作ってしまっていると思われ。
					// 定跡読み込み時に削除しているはずなのだが。

					// ======================================
					//  leaf nodeに対するペナルティ/ボーナス
					// ======================================

					// 棋譜上に出現する局面がleaf nodeならば、ここに加点(このleaf nodeに行きたいため)
					if (kif_hash.find(key_next) != kif_hash.end())
						value.value += search_option.search_delta_on_kif;

				}
				else {
					// alpha-beta searchなので 1手進めてalphaとbetaを入れ替えて探索。
					StateInfo si;
					pos.do_move(m, si);

					// テラショック化する時の探索。
					// 1) 探索Windowは全域。(alpha-betaのWindowでの枝刈りをしてはならない)
					//  →　Windowが全域になっているのでβcutは発生しないはず。
					// 2) 探索結果のvalueをこのchild.evalに反映させる必要がある。(定跡DBに書き出す時に用いるため)

					// 1)
					ValueDepth new_alpha = (searchMode == NormalAlphaBeta) ? - beta : -VALUE_INFINITE;
					ValueDepth new_beta  = (searchMode == NormalAlphaBeta) ? -alpha :  VALUE_INFINITE;

					bool skipNonPV = (alpha == -VALUE_INFINITE) && nodeType == PV;

					// 直後にPVで探索しなおすことになるのでNonPVでの探索はskipする。
					if (skipNonPV)
					{
						value = ValueDepth(VALUE_ZERO); // alpha == -VALUE_INFINITEなので次のifの条件を確実に満たす。
					}
					else {
						value = -search<NonPV, searchMode>(pos, next_node, new_alpha, new_beta, ply + 1);
						value.depth++;
					}

					// alpha値を更新するなら、PVとして探索しなおす。
					// テラショック化の時はalpha == valueでもPVとして探索しなおす。
					if (nodeType == PV && (alpha < value || ((searchMode == TeraShockSearch) && alpha == value)))
					{
						value = -search<PV, searchMode>(pos, next_node, new_alpha, new_beta, ply + 1);
						value.depth++;
					}

					// 2)
					if (searchMode == TeraShockSearch)
						child.eval = value;

					// 子ノードのスコアが循環が絡んだスコアであるなら、このnodeにも伝播すべき。
					cyclic = next_node->cyclic;
					pos.undo_move(m);
				}

				// ======================================
				//              α値の更新
				// ======================================

				// update best value
				if (best_value < value)
				{
					best_value = value;
					best_move = m;
				}

				// alpha値を更新するのか？
				// テラショック化の時は、alpha == valueのケースもupdateする。
				// なぜなら、同じ評価値をつけた上位の指し手(引き分けで上位の数手がVALUE_ZEROを想定)に対しては
				// PVで探索したいし、その探索の時のcyclicをこのnodeのcyclicに伝播されて欲しいから。
				if (alpha < value || ((searchMode == TeraShockSearch) && alpha == value))
				{
					// --- nodeの更新

					// alpha値を更新するのに用いた最後の子nodeに関するcyclic - 1 をこのノードに伝播させる。
					// そこまでにcyclicな子がいても最終的にこのnodeのalphaを更新しないなら、その指し手は選択しないわけだから
					// このnodeを循環ノードとみなさなくて良いと思う。
					// 
					// TODO :  ここ注意深く考えないと、GHI問題に遭遇する。
					if (alpha < value)
						node->cyclic = 0; // ベストな指し手以外のcyclicは忘れる。

					node->cyclic = std::max(node->cyclic, (s8)(cyclic - 1));

					// update alpha
					alpha = value;

					// beta以上ならbeta cutが生じる。
					// (親nodeでalpha値を更新しないので)
					if (beta <= value)
						break;

					// alphaを更新したので、このノードで得たPVは今回のPVの末尾のもの(leaf node)だけで良い。
					// best valueを記録するときのleaf nodeだけ欲しいのでそれ以外の手順は消して問題ない。
					if (search_pv.size() > search_pv_index1 + 1)
					{
						search_pv[search_pv_index1] = search_pv.back();
						search_pv.resize(search_pv_index1 + 1);
					}
				}
				else {
					// alphaを更新しなかったので今回の指し手によるPVはすべて巻き戻して良い。
					// (それがPVであることはない)
					search_pv.resize(search_pv_index2);
				}
			}

			// 今回得られたbest_valueの性質
			node->bound = (best_value <= old_alpha) ? Bound::BOUND_UPPER
						: (best_value >= beta     ) ? Bound::BOUND_LOWER
													: Bound::BOUND_EXACT;

			node->best_move = best_move;
			node->search_value = alpha;

			// 世代の更新(このnodeは探索しなおすので)
			node->generation = search_option.generation;

			// best_move、それが1番目ではないなら次回の探索に備えて入れ替えておくべき。
			// 循環は完全に排除しているので、いまの祖先にこのnodeが含まれることはない。
			node->set_best_move(best_move);

			return best_value;
		}

		// 探索させて、そのNodeを追加する。
		// 結果はpmにNode追加して書き戻される。
		// 思考する前にNodeが存在していた時は、already_exists == trueになる。
		Node* think(Position& pos, const SearchNode* s_node,bool& already_exists)
		{
			string sfen = s_node->node->sfen;
			Move16 m16 = s_node->move;
			Move move = pos.to_move(m16);

			// 思考中の局面を出力してやる。
			sync_cout << "thinking : " << sfen << " , move = " << move << sync_endl;

			return think_sub(pos , "sfen " + sfen + " moves " + to_usi_string(m16),already_exists);
		}

		// SFEN文字列を指定して、その局面について思考させる。
		// 結果はpmにNode追加して書き戻される。
		// sfen : positionコマンドで送れる形の文字列
		// 例) "sfen xxxx" , "startpos" , "startpos moves xxxx" , "sfen xxxx moves yyyy..."
		// 思考する前にNodeが存在していた時は、already_exists == trueになる。
		Node* think(Position& pos, string sfen , bool& already_exists)
		{
			// 思考中の局面を出力してやる。
			cout << "thinking : " << sfen << endl;

			return think_sub(pos,sfen,already_exists);
		}

		// 現在の局面について思考して、その結果をpmにNode作って返す。
		// sfen : positionコマンドで送れる形の文字列
		// 例) "sfen xxxx" , "startpos" , "startpos moves xxxx" , "sfen xxxx moves yyyy..."
		// 思考する前にNodeが存在していた時は、already_exists == trueになる。
		Node* think_sub(Position& pos,string sfen,bool& already_exists)
		{
			// ================================
			//        Limitsの設定
			// ================================

			Search::LimitsType limits = Search::Limits;

			// ノード数制限
			limits.nodes = search_option.nodes_limit;

			// 探索中にPVの出力を行わない。
			limits.silent = true;

			// ================================
			//           思考開始
			// ================================

			// SetupStatesは破壊したくないのでローカルに確保
			StateListPtr states(new StateList(1));

			// sfen文字列、Positionコマンドのparserで解釈させる。
			istringstream is(sfen);
			position_cmd(pos, is, states);

			// すでにあるのでskip
			Node* n = pm.probe(pos.long_key());

			// すでに思考したあとの局面であった。
			already_exists = (n != nullptr);
			if (already_exists)
				return n;

			// 思考部にUSIのgoコマンドが来たと錯覚させて思考させる。
			Threads.start_thinking(pos, states , limits);
			Threads.main()->wait_for_search_finished();
			
			// ================================
			//        探索結果の取得
			// ================================

			auto search_result = dlshogi::GetSearchResult();
			Node* node_;

			// 新規にNodeを作成してそこに書き出す
			{
				// nodeを追加するので、create_node()付近でmutexをlockしなければならない。
				// これはthinkが終わった瞬間に行うだけなので、全体時間からするとlockされている時間は
				// 極わずかだから、問題ない。
				std::unique_lock<std::mutex> lk(*pm.get_mutex());

				node_ = pm.create_node(pos);
				Node& node = *node_;

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
					Value cp = Eval::dlshogi::value_to_cp((float)wp, search_option.eval_coef);
					ValueDepth vp(cp, 0);

					// Nodeのchildrenに追加。
					node.children.emplace_back(Child(m, vp));
				}

				// Childのなかで一番評価値が良い指し手がbest_moveであり、
				// その時の評価値がこのnodeの評価値。

				Value best_value = -VALUE_INFINITE;
				size_t max_index = SIZE_MAX; /* not found flag */
				for (size_t i = 0; i < num; ++i)
				{
					Value eval = (Value)node.children[i].eval.value;
					if (best_value < eval)
					{
						best_value = eval;
						max_index = i;
					}
				}

				// 普通は合法手、ひとつは存在するはずなのだが…。
				if (max_index != SIZE_MAX)
				{
					node.best_move = pos.to_move(node.children[max_index].move);

					// この指し手をchildrentの先頭に持ってきておく。(alpha-beta探索で早い段階で枝刈りさせるため)
					std::swap(node.children[0], node.children[max_index]);
				}
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

			return node_;
		}

		// 定跡をファイルに保存する。
		void save_book(const string& path)
		{
			cout << "save book , path = " << path << endl;
			pm.save_book(path);
		}

		void save_tera_book(const string& path)
		{
			cout << "save tera-book , path = " << path << endl;
			pm.save_tera_book(path);
		}

		// USIオプションを探索用に設定する。
		void set_usi_options()
		{
			// 定跡にhitされると困るので定跡なしに
			Options["BookFile"] = string("no_book");

			// rootでdf-pnで詰みが見つかると候補手が得られないので
			// とりまオフにしておく。
			Options["RootMateSearchNodesLimit"] = std::to_string(0);

			// 全合法手を生成する。
			Options["GenerateAllLegalMoves"] = true;

			// ↓isreadyを送ってやらないと↑が反映しない。
			is_ready();
		}

	private:

		// 局面管理クラス
		PositionManager pm;

		// 探索時のオプション
		SearchOption search_option;

		// alpha-beta探索した時のPV
		// best_valueを記録したleaf nodeしか要らない。
		// それは、search_pv.back()であることが保証されている。
		vector<SearchNode> search_pv;

		// 思考した局面数
		// (一定間隔ごとに定跡をファイルに保存しないといけないのでそのためのカウンター)
		size_t think_count = 0;

		// 棋譜上に出現した局面のhash key
		unordered_set<HASH_KEY> kif_hash;

		// ranged alpha beta searchで探索中なので避けるべきnodeのhash key
		SearchingNodes searching_nodes;

		// 何度もPV leaf nodeがここになる。おそらく、千日手による局面だと思う。
		// これは探索の時に除外する。
		ConcurrentSet<SearchNode> banned_nodes;

		// 探索すべき局面が詰まっているQueue。
		Concurrent::ConcurrentQueue<SearchNode> search_nodes;

		// 探索したノード数
		u64 node_searched;

		// ↑の計測用タイマー
		Timer search_timer;
	};

	// 定跡DBからsfenと書かれている行だけを抽出するコマンド。
	void extract_sfen_from_db(Position& pos, istringstream& is)
	{
		cout << "extract sfen from db command." << endl;

		string read_book_name  = "book/read_book.db";
		string write_sfen_name = "book/write_sfen.txt";

		Parser::ArgumentParser parser;
		parser.add_argument("read_book"           , read_book_name);
		parser.add_argument("write_sfen"          , write_sfen_name);
		parser.parse_args(is);

		cout << "ReadBook DB file      : " << read_book_name         << endl;
		cout << "Write  SFEN file      : " << write_sfen_name        << endl;

		SystemIO::TextReader reader;
		if (reader.Open(read_book_name).is_not_ok())
		{
			cout << "read file not found , file = " << read_book_name << endl;
			return;
		}

		size_t filesize = reader.GetSize();
		Tools::ProgressBar progress(filesize);

		SystemIO::TextWriter writer;
		if (writer.Open(write_sfen_name).is_not_ok())
		{
			cout << "write file open failed , file = " << write_sfen_name << endl;
			return;
		}

		string line;
		while (reader.ReadLine(line).is_ok())
		{
			progress.check(reader.GetFilePos());

			// sfenで始まる行だけを書き出す。(これだけならgrepでいいような気も…)
			if (StringExtension::StartsWith(line, "sfen"))
				writer.WriteLine(line);
		}

		writer.Close();
		reader.Close();

		// コマンドが完了したことを出力。
		cout << "extract sfen from db command has finished." << endl;
	}

	// USIのPositionコマンドで指定できる形で書かれた棋譜ファイルなどを読み込み、その各局面をsfen形式で書き出すコマンド
	void extract_sfen_from_sfen(Position& pos, istringstream& is)
	{
		cout << "extract sfen from sfen command." << endl;

		string read_sfen_name  = "book/read_sfen.txt";
		string write_sfen_name = "book/write_sfen.txt";

		Parser::ArgumentParser parser;
		parser.add_argument("read_sfen"           , read_sfen_name);
		parser.add_argument("write_sfen"          , write_sfen_name);
		parser.parse_args(is);

		cout << "ReadBook DB file      : " << read_sfen_name         << endl;
		cout << "Write  SFEN file      : " << write_sfen_name        << endl;

		SystemIO::TextReader reader;
		if (reader.Open(read_sfen_name).is_not_ok())
		{
			cout << "read file not found , file = " << read_sfen_name << endl;
			return;
		}

		size_t filesize = reader.GetSize();
		Tools::ProgressBar progress(filesize);

		SystemIO::TextWriter writer;
		if (writer.Open(write_sfen_name).is_not_ok())
		{
			cout << "write file open failed , file = " << write_sfen_name << endl;
			return;
		}

		// 重複除去のためのset
		unordered_set<string> written_sfens;

		string line;
		while (reader.ReadLine(line).is_ok())
		{
			progress.check(reader.GetFilePos());

			// この行の棋譜の各局面をsfenにして書き出す。

			std::vector<StateInfo> si(1024);
			BookTools::feed_position_string(pos, line , si,[&](Position& pos){
				// 各局面でcallbackがかかるので、sfen化したものを書き出す。
				auto sfen = pos.sfen();
				if (written_sfens.find(sfen) == written_sfens.end())
				{
					written_sfens.emplace(sfen);
					writer.WriteLine(pos.sfen());
				}
			});
		}

		writer.Close();
		reader.Close();

		// コマンドが完了したことを出力。
		cout << "extract sfen from sfen command has finished." << endl;
	}
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

		// makebook stera_convert
		// スーパーテラショック定跡コマンド("makebook stera")で生成した
		// 定跡をテラショック化(思考エンジンで使う形の定跡に)変換する。
		if (token == "stera_convert")
		{
			MakeBook2021::SuperTeraBook st;
			st.stera_convert(pos, is);
			return 1;
		}

		// 定跡DBからsfenと書かれている行だけを抽出するコマンド。
		// 定跡DBに対してその局面をスーパーテラショック定跡の棋譜として与えて
		// その棋譜周りを掘っていくために用いる。
		if (token == "extract_sfen_from_db")
		{
			MakeBook2021::extract_sfen_from_db(pos, is);
			return 1;
		}

		// USIのPositionコマンドで指定できる形で書かれた棋譜ファイルなどを読み込み、その各局面をsfen形式で書き出すコマンド
		if (token == "extract_sfen_from_sfen")
		{
			MakeBook2021::extract_sfen_from_sfen(pos, is);
			return 1;
		}

		return 0;
	}
}

#endif // defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
