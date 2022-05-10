#include "fast_alloc.h"

#if defined(USE_FAST_ALLOC)

#include <cstring>				// std::memset()

// globalになっていて、"USI_Hash"でこれを確保。
// ふかうら王では、ここから使う、的な…。
FastAlloc FAST_ALLOC;

// 最初に巨大メモリを確保する。
// 最初に呼び出すこと。
// 以前にmemory_alloc()で確保したメモリは開放される。
// mb の単位は [MB]。
void FastAlloc::memory_alloc(size_t mb)
{
    size_t new_chunks_size = 1024*1024/32*mb;
    // →　32で先に割らないと、mbが(非常に)大きいとオーバーフローする。

    // 要求されたサイズが、いますでに確保しているサイズと異なるなら、
    // 以前のものを開放して、再度確保する。
    if (chunks_size != new_chunks_size)
    {
        memory_free();
        // mallocはあとで32byteのalignされたメモリを確保できるやつに書き換える。
		size_t alloc_size = new_chunks_size*sizeof(MemoryChunk);
        chunks            = (MemoryChunk*)largeMemory.alloc(alloc_size, 64, false);
        chunks_size       = new_chunks_size;

		// メモリのclearを通じて、実際にアクセスしておいたほうが
		// OSが物理メモリに割り当てるのでそのあと高速にアクセスできる。
		Tools::memclear("USI_Hash" , chunks, alloc_size);
	}

    // いずれにせよ、poolのゼロ初期化は必要。
    std::memset(pool, 0, sizeof(pool));
}

// 確保した巨大メモリの開放
void FastAlloc::memory_free()
{
    if (chunks)
    {
        largeMemory.free();
        chunks = nullptr;
    }
}

// 確保しておいた巨大メモリからメモリを割り当てる。
void* FastAlloc::alloc(size_t s)
{
    ASSERT_LV3(s <= MAX_ALLOC_CHUNK * 32 - 8);

    // poolに空きがあるか。

    // s[byte]の確保によって消費するchunkの数。
    size_t chunk_block_num = ((s + 8) + (CHUNK_SIZE - 1))/CHUNK_SIZE;
    // →　s + 8 の 8 は、chunk header size。
    //    32の倍数に繰り上げてから CHUNK_SIZE(32)で割って、消費するchunkの数を算出している。

    // poolの操作に対するlock。
    // 確保するchunkの数によって、lockするmutexが異なる。
    std::lock_guard<std::mutex> lock(mutexes[chunk_block_num]);

    // chunk_block_numのchunkがpoolに返還されているか。
    // 返還されていれば、poolから返す。
    size_t chunk_index     = pool[chunk_block_num];
    // ※　返還されていれば、chunk_index != 0。

    void* ptr;

    if (s == 0)
    {
        // 0 byteのalloc。メモリのallocatorである以上、これは考慮されていないといけない。

        // 一つ前のアドレスを返す。(どうせ実際にはアクセスしないと思うのでこれで良し)
        ptr = chunks - 1;

    } else if (chunk_index == 0)
    {
        // このサイズのchunk blockは、poolに返還されていないので、
        // 巨大メモリの次に使うchunkの位置から確保して返す。

        // これはpool[0]に次に使うべきindexが書かれている。(chunks[pool[0]]を使っていく)
        // pool[0]を操作するのでmutexes[0]をlockする。(ことになっている)

        std::lock_guard<std::mutex> lock(mutexes[0]);
        size_t& chunk_next_index = pool[0];

        // 巨大メモリ、溢れてしまわないか？
        if (chunk_next_index + chunk_block_num <= chunks_size)
        {
            // 溢れないので、ここから返す。

            MemoryChunk& chunk = chunks[chunk_next_index];

            // chunkのheaderにブロックサイズの情報を記入しておく。(開放する時にこれがないと困るため)
            chunk.chunk_block_size = chunk_block_num;
 
            // chunk上で実データを配置する先頭アドレスをこの関数の返し値として返す。
            ptr = & chunk.t;

            // 次に新規にchunkを確保する時は、chunk_next_indexのところから。
            chunk_next_index += chunk_block_num;

        } else {

            // 巨大メモリを使い切ってしまうので確保できない。
            ptr = nullptr;
        }

    } else {

        // poolから返す。一つ返すのでそのchainを繋ぎ変える必要がある。
        // chunk.next_chunk_indexに次の同じサイズの空きchunkのchain(ポインタ)が格納されているものとする。
        MemoryChunk& chunk = chunks[chunk_index - 1];

        uint64_t next_chunk_index = chunk.next_chunk_index;

        // 空きブロックは数珠つなぎになっているはずなので、これをpoolに戻す。
        pool[chunk_block_num] = next_chunk_index;

        // 先頭から8バイト加算したところが実データーなので、そこのアドレスを返す。
        ptr = (int8_t*)&chunk.t;
    }
    return ptr;
}

// alloc()で確保したメモリを開放する。
void FastAlloc::free(void* p)
{
    if (p == nullptr || p == chunks - 1)
        return ;

    // 返してもらったメモリをpoolに格納しておく。

    // chunkの先頭アドレスは headerのサイズである8を引いたところ。
    MemoryChunk* chunk = (MemoryChunk*)((int8_t*)p - 8);
    size_t chunk_block_num = chunk->chunk_block_size;

    // この瞬間に、このpのchunkが別スレッドからfree()されてしまう心配はない。
    // なぜなら、同じアドレスpに対して複数のスレッドがfree()することはないから。
    // (2度freeされていることになってしまう)

    // poolの操作を行うのでlockする。
    std::lock_guard<std::mutex> lock(mutexes[chunk_block_num]);

    // poolに格納されていた空きchunkの先頭index。これを付け替える。
    size_t old_chunk_index = pool[chunk_block_num];
    chunk->next_chunk_index = old_chunk_index;

    size_t chunk_index = chunk - chunks + 1;
    pool[chunk_block_num] = chunk_index;
}

// 使用率を1000分率で返す。
int FastAlloc::hashfull() const
{
    // まだ確保していない。
    if (chunks == nullptr)
        return 0;

    return (int)(1000 * pool[0] / chunks_size);
}

// 残り空きメモリ量を返す。
size_t FastAlloc::rest() const
{
	return (chunks_size - pool[0]) * sizeof(MemoryChunk);
}

#endif // defined(USE_FAST_ALLOC)
