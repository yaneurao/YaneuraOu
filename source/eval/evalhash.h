#ifndef EVALHASH_H_INCLUDED
#define EVALHASH_H_INCLUDED

#include "../types.h"

// シンプルなHashTableの実装。Sizeは2のべき乗。
// 評価値のcacheに用いる。
template <typename T>
struct HashTable
{
	// 配列のresize。単位は[MB]
	void resize(size_t mbSize)
	{
		size_t newClusterCount = mbSize * 1024 * 1024 / sizeof(T);
		newClusterCount = (size_t)1 << MSB64(newClusterCount); // msbだけ取り、2**nであることを保証する

		if (newClusterCount != size)
		{
			release();
			size = newClusterCount;
			entries_ = (T*)aligned_malloc(size * sizeof(T), alignof(T));
		}
	}
	void release() { if (entries_) { aligned_free(entries_); entries_ = nullptr; } }
	~HashTable() { release(); }

	T* operator[] (const Key k) { return entries_ + (static_cast<size_t>(k) & (size - 1)); }
	void clear() { memclear("eHash", entries_,  size * sizeof(T)); }

private:

	size_t size = 0;
	T* entries_ = nullptr;
};

#endif // EVALHASH_H_INCLUDED