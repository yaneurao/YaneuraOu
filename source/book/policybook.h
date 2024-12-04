#ifndef POLICYBOOK_H_INCLUDED
#define POLICYBOOK_H_INCLUDED

#include "../config.h"

#if defined(USE_POLICY_BOOK)

#include <vector>
#include "../types.h"
#include "../extra/key128.h"

static_assert(HASH_KEY_BITS == 128 , "HASH_KEY_BITS must be 128");

// ============================================================
//				Policy Book
// ============================================================
struct MoveRatio
{
	Move16 move16;
	u16 ratio;// 2^16 - 1 を100%とする割合の表現
};
constexpr int POLICY_BOOK_NUM = 4;
struct PolicyBookEntry
{
	HASH_KEY key;
	MoveRatio move_ratio[POLICY_BOOK_NUM];
	// 128bit(=16bytes) + 16bytes = 32bytes
};

class PolicyBook
{
public:
	// PolicyBookを読み込む。
	void read_book();

	// Policy Bookのなかに指定されたkeyの局面があるか。
	// 見つからなければnullptrが返る。
	PolicyBookEntry* probe_policy_book(HASH_KEY key);

private:
	// PolicyBook本体
	std::vector<PolicyBookEntry> book_body;
};
#endif

#endif
