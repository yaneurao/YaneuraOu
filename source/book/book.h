#ifndef _BOOK_H_INCLUDED_
#define _BOOK_H_INCLUDED_

#include "../types.h"
#include "../position.h"
#include "../misc.h"
#include "../usi.h"

#include <unordered_map>

namespace Search { struct LimitsType; };


// 定跡処理関連のnamespace
namespace Book
{
	// 将棋ソフト用の標準定跡ファイルフォーマットの提案 : http://yaneuraou.yaneu.com/2016/02/05/standard-shogi-book-format/

	// ある局面における指し手1つ(定跡の局面での指し手を格納するのに用いる)
	// これの集合体が、ある局面の指し手集合。その集合体が、定跡ツリー。
	struct BookMove
	{
		// ここでの指し手表現は32bit型の指し手(Move)ではなく、16bit型の指し手(Move16)なので、
		// Position::do_move()などにはPosition::to_move()を用いて32bit化してから用いること。

		Move16 move;   // この局面での指し手
		Move16 ponder; // その指し手を指したときの予想される相手の指し手(指し手が無いときはnoneと書くことになっているので、このとMOVE_NONEになっている)

		// ↓定跡DBに以下は書かれているとは限らない。optionalな項目

		int value;     // moveを指したときの局面の評価値
		int depth;     // moveの探索深さ

		uint64_t move_count; // 何らかの棋譜集において、この指し手が採択された回数。

		// --- MCTSで作った定跡の場合これを記録しておき、これを見て指し手を選択する ---

		double    win; // moveを採択してこの局面の手番側が勝利した回数(端数ありうる)
		double   draw; // moveを採択して引き分けた回数。(端数はないはずだが、使用目的によってはあるかも)

		// bestMoveを採択してこの局面の手番側が負けた回数
		// これは計算して出すので、関数。
		double lose() const {
			return move_count - (win + draw);
		}

		// ----------------------------------------------------------------------------

		BookMove(Move16 move_, Move16 ponder_, int value_, int depth_, uint64_t move_count_)
			: move(move_), ponder(ponder_), value(value_), depth(depth_), move_count(move_count_) , win(0),draw(0){}

		BookMove(Move16 move_, Move16 ponder_, int value_, int depth_, uint64_t move_count_ , double win_ , double draw_)
			: move(move_), ponder(ponder_), value(value_), depth(depth_), move_count(move_count_) , win(win_),draw(draw_){}

		// 定跡フォーマットの、ある局面の指し手を書いてある一行を引数lineに渡して、
		// BookMoveのオブジェクトを構築するbuilder。
		static BookMove from_string(std::string line);

		// このoperatorは、bestMoveが等しいかを判定して返す。
		bool operator == (const BookMove& rhs) const
		{
			return move == rhs.move;
		}

		// std::sort()で出現回数に対して降順ソートされて欲しいのでこう定義する。
		// また出現回数が同じ時は、評価値順に降順ソートされて欲しいので…。
		bool operator < (const BookMove& rhs) const {
			return (move_count != rhs.move_count) ? (move_count > rhs.move_count ) : (value > rhs.value);
		}
	};

	static std::ostream& operator<<(std::ostream& os, BookMove c);

	// BookMovesで返し値などに使いたいのでこれを定義しておく。
	typedef std::vector<BookMove>::iterator BookMoveIterator;

	// 定跡のある局面での指し手の候補集合
	// メモリ上ではこれをshared_ptrでくるんで保持している。
	//  → BookMovesPtr
	// ある局面が定跡に登録されている局面であるかを調べる関数が
	// 指し手を返すときもこのshared_ptrでくるんだものを返す。
	//
	// ※　このクラスのメンバ関数で、[ASYNC]とコメントに書かれているものは、
	// thread safeであることが保証されている。
	struct BookMoves
	{
		BookMoves() {}

		// copy constructor
		// std::recursive_mutexを持っているので暗黙のコピーは不可。自前でコピーしてやる。
		BookMoves(const BookMoves& bm) { sorted = bm.sorted; moves = bm.moves; }

		// [ASYNC] BookMoveを一つ追加する
		// ただし、その局面ですでに同じmoveの指し手が登録されている場合、
		//   overwrite == true の時は、上書き動作となる。このとき、BookMove::num,win,loseは、合算した値となる。
		//   overwrite == falseの時は、上書きされない。
		void insert(const BookMove& book_move,bool overwrite);

		// [ASYNC] BookMoveを一つ追加する。
		// 動作はinsert()とほぼ同じだが、この局面に同じ指し手は存在しないことがわかっている時に用いる。
		// こちらのほうが、同一の指し手が含まれるかのチェックをしない分だけ高速。
		void push_back(const BookMove& book_move);

		// [ASYNC] ある指し手と同じBookMoveのエントリーを探して、あればそのポインターを返す。
		std::shared_ptr<BookMove> find_move(const Move16 move) const;

		// [ASYNC] この局面に登録されている指し手の数を返す。
		// 指し手の数は、MAX_MOVESであるので、intの範囲で十分であるのだが、
		// std::vector.size()とコンパチでないと色んなところで警告が出てうざいのでsize_tにしておく。
		size_t size() const { return moves.size(); }

		// [ASYNC] 指し手を出現回数、評価値順に並び替える。
		// ※　より正確に言うなら、BookMoveのoperator <()で定義されている順。
		// すでに並び替わっているなら、何もしない。
		// 書き出す寸前とか、読み込んで、定跡にhitした直後とかにsort_moves()を呼び出せば良いという考え。
		void sort_moves();

		// [ASYNC] このクラスの持つ指し手集合に対して、それぞれの局面を列挙する時に用いる
		void foreach(std::function<void(BookMove&)> f);

		// -------------------------------------------------------------------

		// 以下、std::vector と同じメンバ。これらは、thread safeではない。
		// シングルスレッドで実行する文脈でだけ使うべし。

		const BookMove& operator[](size_t s) const { return moves[s]; }
		const BookMoveIterator begin() { return moves.begin(); }
		const BookMoveIterator end() { return moves.end(); }
		BookMoveIterator erase(BookMoveIterator start, BookMoveIterator last) { return moves.erase(start, last); }
		void clear() { moves.clear(); }

		// -------------------------------------------------------------------

	private:
		// 候補となる指し手の集合
		std::vector<BookMove> moves;

		// ↑のmovesがsort済みであるかのフラグ。insertなどに対してfalseに変更しておき、
		// sort()を呼び出されたら、sort()して、このフラグをtrueに変更する。
		// なるべく遅延してsortしたいため。
		bool sorted = false;

		// このrecordを操作するときのrecursive_mutex
		std::recursive_mutex mutex_;
	};

	typedef std::shared_ptr<BookMoves> BookMovesPtr;

	// sfen文字列からBookMovesPtrへの写像。(これが定跡データがメモリ上に存在するときの構造)
	typedef std::unordered_map<std::string /* sfen */, BookMovesPtr > BookType;

	// メモリ上にある定跡ファイル
	// ・sfen文字列をkeyとして、局面の指し手へ変換するのが主な役割。(このとき重複した指し手は除外するものとする)
	// ・on the flyが指定されているときは実際はメモリ上にはないがこれを透過的に扱う。
	struct MemoryBook
	{
		// [ASYNC] 定跡として登録されているかを調べて返す。
		// ・見つからなかった場合、nullptrが返る。
		// ・read_book()のときにon_the_flyが指定されていれば実際にはメモリ上には定跡データが存在しないので
		// ファイルを調べに行き、BookMovesPtrをメモリ上に作って、それをくるんだBookMovesPtrを返す。
		BookMovesPtr find(const Position& pos);

		// [ASYNC] 定跡を内部に読み込む。
		// ・Aperyの定跡ファイルは"book/book.bin"だと仮定。(これはon the fly読み込みに非対応なので丸読みする)
		// ・やねうら王の定跡ファイルは、on_the_flyが指定されているとメモリに丸読みしない。
		//      Options["BookOnTheFly"]がtrueのときはon the flyで読み込むのでそれ用。
		// 　　定跡作成時などはこれをtrueにしてはいけない。(メモリに読み込まれないため)
		// ・同じファイルを二度目は読み込み動作をskipする。
		// ・filenameはpathとして"book/"を補完しないので生のpathを指定する。
		Tools::Result read_book(const std::string& filename, bool on_the_fly = false);

		// [ASYNC] 定跡ファイルの書き出し
		// ・sort = 書き出すときにsfen文字列で並び替えるのか。(書き出しにかかる時間増)
		// →　必ずソートするように変更した。
		// ・ファイルへの書き出しは、*thisを書き換えないという意味においてconst性があるので関数にconstを付与しておく。
		// また、事前にis_ready()は呼び出されているものとする。
		Tools::Result write_book(const std::string& filename /*, bool sort = false*/) const;

		// [ASYNC] Aperyの定跡ファイルを読み込む
		// ・この関数はread_bookの下請けとして存在する。外部から直接呼び出すのは定跡のコンバートの時ぐらい。
		Tools::Result read_apery_book(const std::string& filename);

		// --------------------------------------------------------------------------
		//     以下のメンバは、普段は外部から普段は直接アクセスすべきではない 
		//
		// 定跡を書き換えてwrite_book()で書き出すような作業を行なうときだけアクセスする。
		// --------------------------------------------------------------------------

		// [ASYNC] book_body.find()のwrapper。book_body.find()ではなく、こちらのfindを呼び出して用いること。
		BookMovesPtr find(const std::string& sfen) const;

		// [ASYNC] メモリに保持している定跡に局面を一つ追加する。
		// book_body[sfen] = ptr;
		// と等価。すでに登録されているとしたら、それは置き換わる。
		void append(const std::string& sfen, const Book::BookMovesPtr& ptr);

		// [ASYNC] book_bodyの局面sfenに対してBookMoveを一つ追加するヘルパー関数。
		// 同じ局面に対して繰り返し、この関数を呼ぶぐらいなら、BookMovesに直接push_back()してから、このクラスのappend()を呼ぶべき。
		// overwrite : このフラグがtrueならば、その局面ですでに同じmoveの指し手が登録されている場合は上書き動作。(このとき採択回数num,win,loseは合算する)
		//             このフラグがfalseならば、その局面ですでに同じmoveの指し手が登録されている場合は何もしない。
		void insert(const std::string& sfen, const BookMove& bp , bool overwrite = true);

		// [ASYNC] このクラスの持つ定跡DBに対して、それぞれの局面を列挙する時に用いる
		void foreach(std::function<void(std::string /*sfen*/, BookMovesPtr)> f);

	protected:

		// メモリ上に読み込まれた定跡本体
		// book_body.find()の直接呼び出しは禁止
		// (Options["IgnoreBookPly"]==trueのときにplyの部分を削ってメモリに読み込んでいるため、一致しないから)
		// このクラス(MemoryBookクラス)のfind()メソッドを用いること。
		BookType book_body;

		// ↑を操作するときのmutex
		std::recursive_mutex mutex_;

		// 末尾のスペース、"\t","\r","\n"を除去する。
		// Options["IgnoreBookPly"] == trueのときは、さらに数字も除去する。
		// sfen文字列の末尾にある手数を除去する目的。
		std::string trim(std::string input) const;

		// メモリに丸読みせずにfind()のごとにファイルを調べにいくのか。
		// これは思考エンジン設定のOptions["BookOnTheFly"]の値を反映したもの。
		// ただし、read_book()のタイミングで定跡ファイルのopenに失敗したならfalseのままである。
		// このフラグがtrueのときは、定跡ファイルのopen自体には成功していることが保証される。
		bool on_the_fly = false;

		// 前回読み込み時のOptions["IgnoreBookPly"]の値を格納しておく。
		// これが異なるならファイルの読み直しが必要になる。
		bool ignoreBookPly = false;

		// 上のon_the_fly == trueのときに、開いている定跡ファイルのファイルハンドル
		std::fstream fs;

		// read_book()のときに読み込んだbookの名前
		// ・on_the_fly == trueのときは、読み込む予定のファイルの名前。
		// ・二度目のread_book()の呼び出しのときにすでに読み込んである(or ファイルをopenしてある)かどうかの
		// 判定のためにファイル名を内部的に保持してある。
		std::string book_name;
		std::string pure_book_name; // book_nameからフォルダ名を取り除いたもの。
	};

#if defined (ENABLE_MAKEBOOK_CMD)
	// USI拡張コマンド。"makebook"。定跡ファイルを作成する。
	// フォーマット等についてはdoc/解説.txt を見ること。
	extern void makebook_cmd(Position& pos, std::istringstream& is);
#endif

	// 思考エンジンにおいて定跡の指し手の選択をする部分を切り出したもの。
	struct BookMoveSelector
	{
		// extra_option()で呼び出すと、定跡関係のオプション項目をオプション(OptionMap)に追加する。
		void init(USI::OptionsMap & o);

		// 定跡ファイルの読み込み。
		// ・Search::clear()からこの関数を呼び出す。
		// ・Search::clear()は、USIのisreadyコマンドのときに呼び出されるので
		// 　定跡をメモリに丸読みするのであればこのタイミングで行なう。
		// ・Search::clear()が呼び出されたときのOptions["BookOnTheFly"]の値をcaptureして使う。(ことになる)
		void read_book() { memory_book.read_book(get_book_name(), (bool)Options["BookOnTheFly"]); }

		// --- 定跡の指し手の選択

		// 現在の局面が定跡に登録されているかを調べる。
		// ・定跡にhitした場合は、trueが返るので、このままrootMoves[0]を指すようにすれば良い。
		// ・定跡の指し手の選択は、思考エンジンのオプション設定に従う。
		// ・定跡のなかにPonderの指し手(bestmoveの次の指し手)がもしあればそれはrootMoves[0].pv[1]に返る。
		// ・ただしrootMoves[0].pv[1]が合法手である保証はない。合法手でなければGUI側が弾くと思う。
		// ・limit.silent == falseのときには画面に何故その指し手が選ばれたのか理由を出力する。
		// ・この関数自体はthread safeなのでread_book()したあとは非同期に呼び出して問題ない。
		// 　ただし、on_the_flyのときは、ディスクアクセスが必要で、その部分がthread safeではないので
		//   on_the_fly == falseでなければ、非同期にこの関数を呼び出してはならない。
		// ・Options["USI_OwnBook"]==trueにすることでエンジン側の定跡を有効化されていないなら、
		// 　probe()には常に失敗する。(falseが返る)
		bool probe(Thread& th , Search::LimitsType& limit);

		// 現在の局面が定跡に登録されているかを調べる。
		// ・pos.RootMovesを持っていないときに、現在の局面が定跡にhitするか調べてhitしたらその指し手を返す。
		// ・定跡の指し手の選択は、思考エンジンのオプション設定に従う。
		// ・定跡にhitしなかった場合はMOVE_NONEが返る。
		// ・画面には何も表示しない。
		// ・この関数自体はthread safeなのでread_book()したあとは非同期に呼び出して問題ない。
		// 　ただし、on_the_flyのときは、ディスクアクセスが必要で、その部分がthread safeではないので
		//   on_the_fly == falseでなければ、非同期にこの関数を呼び出してはならない。
		Move probe(Position& pos);

	protected:
		// メモリに読み込んだ定跡ファイル
		MemoryBook memory_book;

		// 読み込んだ定跡ファイル名
		std::string book_name;

		// 定跡ファイル名を返す。
		// Option["BookDir"]が定跡ファイルの入っているフォルダなのでこれを連結した定跡ファイルのファイル名を返す。
		std::string get_book_name() const { return Path::Combine((std::string)Options["BookDir"], (std::string)Options["BookFile"]); }

		// probe()の下請け
		// forceHit == trueのときは、設定オプションの値を無視して強制的に定跡にhitさせる。(BookPvMovesの実装で用いる)
		bool probe_impl(Position& rootPos, bool silent, Move16& bestMove, Move16& ponderMove , bool forceHit = false);

		// 定跡のpv文字列を生成して返す。
		// m : 局面posで進める指し手
		// depth : 残りdepth
		std::string pv_builder(Position& pos, Move16 m , int depth);

		AsyncPRNG prng;
	};

}

#endif // #ifndef INCLUDED_
