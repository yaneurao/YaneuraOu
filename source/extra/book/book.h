#ifndef _BOOK_H_
#define _BOOK_H_

#include "../../shogi.h"
#include "../../position.h"
#include <unordered_map>

struct PRNG;
namespace Search { struct LimitsType; };

// 定跡処理関係
namespace Book
{
	struct MemoryBook;
	// 定跡ファイルの読み込み(book.db)など。
	// 同じファイルを二度目は読み込み動作をskipする。
	// on_the_flyが指定されているとメモリに丸読みしない。
	// 定跡作成時などはこれをtrueにしてはいけない。(メモリに読み込まれないため)
	extern int read_book(const std::string& filename, MemoryBook& book, bool on_the_fly = false);


	// 局面における指し手(定跡を格納するのに用いる)
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
		bool operator < (const BookPos& rhs) const { return num > rhs.num; } // std::sortで降順ソートされて欲しいのでこう定義する。
	};

	// ある局面での指し手の集合
	typedef std::vector<BookPos> PosMoveList;
	typedef std::shared_ptr<PosMoveList> PosMoveListPtr;

	// sfen文字列からPosMoveListへの集合。(これが定跡)
	typedef std::unordered_map<std::string /* sfen */, PosMoveListPtr > BookType;

	// メモリ上にある定跡ファイル
	// sfen文字列をkeyとして、局面の指し手へ変換。(重複した指し手は除外するものとする)
	struct MemoryBook
	{
		// 定跡として登録されているかを調べて返す。
		// readのときにon_the_flyが指定されていればファイルを調べに行く。
		// 見つからなかった場合、nullptrが返る。
		PosMoveListPtr find(const Position& pos);

		// 定跡を内部に読み込む。
		// 定跡ファイル : set_book_name()で事前に渡したファイル名のファイル。
		// Aperyの定跡ファイルは"book/book.bin"だと仮定。(これはon the fly読み込みに非対応なので丸読みする)
		// やねうら王の定跡ファイルは、Options["BookOnTheFly"]がtrueのときはon the flyで読み込むので
		// このタイミングでは実際にはメモリに読み込まない。
		void read_book(const std::string& book_name)
		{ Book::read_book("book/" + book_name, *this , (bool)Options["BookOnTheFly"]); }

		// --- 以下のメンバ、普段は直接アクセスすべきではない。

		// 定跡本体
		BookType book_body;

		// 読み込んだbookの名前
		// (読み込む前にこの名前を設定してはいけない)
		std::string book_name;

		// 内部に読み込んだ定跡のクリア
		void clear() { book_body.clear(); }

		// メモリに丸読みせずにfind()のごとにファイルを調べにいくのか。
		bool on_the_fly = false;

		// 上のon_the_fly == trueのときに、開いている定跡ファイルのファイルハンドル
		std::fstream fs;
	};

	extern int read_apery_book(const std::string& filename, MemoryBook& book);

	// 定跡ファイルの書き出し
	// sort = 書き出すときにsfen文字列で並び替えるのか。(書き出しにかかる時間増)
	extern int write_book(const std::string& filename, const MemoryBook& book, bool sort = false);

	// MemoryBookとかPosMoveListPrtとかにBookPosを一つ追加。(その局面ですでに同じbestMoveの指し手が登録されている場合は上書き動作)
	extern void insert_book_pos(MemoryBook& book, const std::string sfen, const BookPos& bp);
	extern void insert_book_pos(PosMoveListPtr ptr, const BookPos& bp);

#ifdef ENABLE_MAKEBOOK_CMD
	// USI拡張コマンド。"makebook"。定跡ファイルを作成する。
	// フォーマット等についてはdoc/解説.txt を見ること。
	extern void makebook_cmd(Position& pos, std::istringstream& is);
#endif

	// 定跡の指し手の選択をする部分を切り出したもの。
	struct BookMoveSelector
	{
		// extra_option()で呼び出すと、定跡関係のオプション項目をオプション(OptionMap)に追加。
		void init(USI::OptionsMap & o);

		// 定跡ファイルの読み込み。Search::clear()で呼び出す。
		void read_book() { memory_book.read_book(book_name); }

		// --- 定跡の指し手の選択

		// 定跡にhitした場合は、trueが返るので、このままrootMoves[0]を指すようにすれば良い。
		bool probe(Thread& th , Search::LimitsType& limit ,  PRNG& prng);

		// pos.RootMovesを持っていないときに、現在の局面が定跡にhitするか調べてhitしたらその指し手を返す。
		// 画面には何も表示しない。
		Move probe(Position& pos, PRNG& prng);

		// メモリに読み込んだ定跡ファイル
		MemoryBook memory_book;

		// 読み込む予定の定跡ファイル名
		std::string book_name;

	private:
		// probe()の下請け
		bool probe_impl(Position& rootPos, PRNG& prng, bool silent, Move& bestMove, Move& ponderMove);

	};

}

#endif // #ifndef _BOOK_H_
