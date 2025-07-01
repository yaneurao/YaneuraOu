#ifndef POLICYBOOK_H_INCLUDED
#define POLICYBOOK_H_INCLUDED

#include "../config.h"

#if defined(USE_POLICY_BOOK)

#include <vector>
#include "../types.h"
#include "../extra/key128.h"
#include "../misc.h"

namespace YaneuraOu {

static_assert(HASH_KEY_BITS == 128 , "HASH_KEY_BITS must be 128");

#define POLICY_BOOK_DB_NAME           "eval/policy_book.db"
#define POLICY_BOOK_DB_BIN_NAME       "eval/policy_book.db.bin"
#define POLICY_BOOK_LEARN_DB_BIN_NAME "eval/policy_book-learn.db.bin"

// POLICY_BOOK_DB_NAME のDBファイルの先頭行。
#define POLICY_BOOK_HEADER            "#YANEURAOU-POLICY-DB2024 1.01"

// ============================================================
//				Policy Book
// ============================================================

struct MoveFreq
{
	Move16 move16; // 指し手
	u16    freq;   // 出現回数

	MoveFreq() { move16 = Move16::none(); freq = 0; }
	MoveFreq(Move16 m , u16 f) : move16(m), freq(f) {}
};

// MoveFreqのfreqが32bitになっている構造体。内部的に集計に用いる。
struct MoveFreq32
{
	Move16 move16; // 指し手
	u32    freq;   // 出現回数

	MoveFreq32() { move16 = Move16::none(); freq = 0; }
	MoveFreq32(Move16 m, u32 f) : move16(m), freq(f) {}
	MoveFreq32(std::pair<Move16,u32> p) : move16(p.first), freq(p.second) {}

//	explicit operator MoveFreq() const { return MoveFreq(move16, u16(freq)); }
	// → これ用意すると、overflowしている値が代入されてしまう。
};

constexpr int POLICY_BOOK_NUM = 3;

// MoveFreq32がPOLICY_BOOK_NUMだけある構造体(内部で集計にのみ使用する)
struct alignas(16) MoveFreq32Record
{
	MoveFreq32 move_freq32[POLICY_BOOK_NUM];

	// この局面でのvalue。FLT_MAXなら、不明。
	float value = FLT_MAX;

	// freqの和がUINT16_MAXに収まるようにする。
	// 返し値 : freqの和
	u16 overflow_check();
};

// PolicyBookの1つのrecord
struct alignas(32) PolicyBookEntry
{
	HASH_KEY key;
	MoveFreq move_freq[POLICY_BOOK_NUM];

	// この局面でのvalue。FLT_MAXなら、不明。
	float value = FLT_MAX;

	// MoveFreq32Record構造体の値をこの構造体のmove_freqに代入する。
	void from_move_freq32rec(const MoveFreq32Record& mf32r);
};

class PolicyBook
{
public:
	// PolicyBookを読み込む。
	// ".db.bin"形式があれば、それを読み込む。
	// ".db.bin"ファイルがなければ、".db"形式を読み込み、".db.bin"を書き出す。
	// デフォルトの読み込み時の動作。
	Tools::Result read_book();

	// PolicyBookを読み込む。(".db"形式)
	Tools::Result read_book_db(std::string path = POLICY_BOOK_DB_NAME);
	// PolicyBookを読み込む。(".db.bin"形式)
	Tools::Result read_book_db_bin(std::string path = POLICY_BOOK_DB_BIN_NAME);

	// PolicyBookを書き出す。(".db.bin"形式)
	Tools::Result write_book_db_bin(std::string path = POLICY_BOOK_DB_BIN_NAME);

	// PolicyBook同士のmerge
	void merge_book(const PolicyBook& book);

	// PolicyBookの重複レコードなどを掃除する。
	void garbage_book();

	// "position "コマンドのposition以降の文字列を渡して、それを
	// POLICY_BOOK_LEARN_DB_BIN_NAMEにappendで書き出す。
	void append_sfen_to_db_bin(const std::string& sfen);

	// ファイルから読み込んだか？
	bool is_loaded() const { return book_body.size() != 0; }

	// Policy Bookのなかに指定されたkeyの局面があるか。
	// 見つからなければnullptrが返る。
	PolicyBookEntry* probe_policy_book(HASH_KEY key);

protected:
	// book_bodyをsortする。
	// ※ probeは二分探索を用いるため、sortされていないとprobeできない。
	void sort_book();

private:
	// PolicyBook本体
	std::vector<PolicyBookEntry> book_body;
};

} // namespace YaneuraOu

#endif // defined(USE_POLICY_BOOK)
#endif
