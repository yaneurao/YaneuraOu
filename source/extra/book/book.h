#ifndef _BOOK_H_
#define _BOOK_H_

#include "../../types.h"
#include "../../position.h"
#include "../../misc.h"
#include "../../usi.h"

#include <unordered_map>
#include <fstream>

namespace Search { struct LimitsType; };


// 定跡処理関連のnamespace
namespace Book
{
	// ある局面における指し手(定跡の局面での指し手を格納するのに用いる)
	struct BookPos
	{
		Move bestMove; // この局面での指し手
		Move nextMove; // その指し手を指したときの予想される相手の指し手
		int value;     // bestMoveを指したときの局面の評価値
		int depth;     // bestMoveの探索深さ
		uint64_t num;  // 何らかの棋譜集において、この指し手が採択された回数。
		float prob;    // ↑のnumをパーセンテージで表現したもの。(read_bookしたときには反映される。ファイルには書き出していない。)

		BookPos(Move best, Move next, int v, int d, uint64_t n) : bestMove(best), nextMove(next), value(v), depth(d), num(n) {}
		bool operator == (const BookPos& rhs) const { return bestMove == rhs.bestMove; }

		// std::sort()で出現回数に対して降順ソートされて欲しいのでこう定義する。
		// また出現回数が同じ時は、評価値順に降順ソートされて欲しいので…。
		bool operator < (const BookPos& rhs) const {
			return (num != rhs.num) ? (num > rhs.num ) : (value > rhs.value);
		}
	};

	static std::ostream& operator<<(std::ostream& os, BookPos c);

	// ある局面での指し手の集合がPosMoveList。
	// メモリ上ではこれをshared_ptrでくるんで保持する。
	// ある局面が定跡に登録されている局面であるかを調べる関数が
	// 指し手を返すときもこのshared_ptrでくるんだものを返す。
	typedef std::vector<BookPos> PosMoveList;
	typedef std::shared_ptr<PosMoveList> PosMoveListPtr;

	// sfen文字列からPosMoveListへの写像。(これが定跡データがメモリ上に存在するときの構造)
	typedef std::unordered_map<std::string /* sfen */, PosMoveListPtr > BookType;

	// PosMoveListPtrに対してBookPosを一つ追加するヘルパー関数。
	// (その局面ですでに同じbestMoveの指し手が登録されている場合は上書き動作となる)
	static void insert_book_pos(PosMoveListPtr ptr, const BookPos& bp);

	// メモリ上にある定跡ファイル
	// ・sfen文字列をkeyとして、局面の指し手へ変換するのが主な役割。(このとき重複した指し手は除外するものとする)
	// ・on the flyが指定されているときは実際はメモリ上にはないがこれを透過的に扱う。
	struct MemoryBook
	{
		// 定跡として登録されているかを調べて返す。
		// ・見つからなかった場合、nullptrが返る。
		// ・read_book()のときにon_the_flyが指定されていれば実際にはメモリ上には定跡データが存在しないので
		// ファイルを調べに行き、PosMoveListをメモリ上に作って、それをくるんだPosMoveListPtrを返す。
		PosMoveListPtr find(const Position& pos);

		// 定跡を内部に読み込む。
		// ・Aperyの定跡ファイルは"book/book.bin"だと仮定。(これはon the fly読み込みに非対応なので丸読みする)
		// ・やねうら王の定跡ファイルは、on_the_flyが指定されているとメモリに丸読みしない。
		//      Options["BookOnTheFly"]がtrueのときはon the flyで読み込むのでそれ用。
		// 　　定跡作成時などはこれをtrueにしてはいけない。(メモリに読み込まれないため)
		// ・同じファイルを二度目は読み込み動作をskipする。
		// ・filenameはpathとして"book/"を補完しないので生のpathを指定する。
		// ・返し値は正常終了なら0。さもなくば非0。
		int read_book(const std::string& filename, bool on_the_fly = false);

		// 定跡ファイルの書き出し
		// ・sort = 書き出すときにsfen文字列で並び替えるのか。(書き出しにかかる時間増)
		// →　必ずソートするように変更した。
		// ・ファイルへの書き出しは、*thisを書き換えないという意味においてconst性があるので関数にconstを付与しておく。
		// ・返し値は正常終了なら0。さもなくば非0。
		// また、事前にis_ready()は呼び出されているものとする。
		int write_book(const std::string& filename /*, bool sort = false*/) const;

		// Aperyの定跡ファイルを読み込む
		// ・この関数はread_bookの下請けとして存在する。外部から直接呼び出すのは定跡のコンバートの時ぐらい。
		// ・返し値は正常終了なら0。さもなくば非0。
		int read_apery_book(const std::string& filename);

		// --- 以下のメンバ、普段は外部から普段は直接アクセスすべきではない。
		// 定跡を書き換えてwrite_book()で書き出すような作業を行なうときだけアクセスする。

		// メモリ上に読み込まれた定跡本体
		BookType book_body;

		// book_bodyに対してBookPosを一つ追加するヘルパー関数。
		// overwrite : このフラグがtrueならば、その局面ですでに同じbestMoveの指し手が登録されている場合は上書き動作
		void insert(const std::string sfen, const BookPos& bp , bool overwrite = true);

	protected:

		// 末尾のスペース、"\t","\r","\n"を除去する。
		// Options["IgnoreBookPly"] == trueのときは、さらに数字も除去する。
		// sfen文字列の末尾にある手数を除去する目的。
		std::string trim(std::string input);

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

#ifdef ENABLE_MAKEBOOK_CMD
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
		bool probe_impl(Position& rootPos, bool silent, Move& bestMove, Move& ponderMove , bool forceHit = false);

		// 定跡のpv文字列を生成して返す。
		// m : 局面posで進める指し手
		// depth : 残りdepth
		std::string pv_builder(Position& pos, Move m , int depth);

		AsyncPRNG prng;
	};

}

#endif // #ifndef _BOOK_H_
