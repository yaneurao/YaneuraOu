#ifndef FAST_ALLOC_H_INCLUDED
#define FAST_ALLOC_H_INCLUDED

#include "../config.h"

#if defined(USE_FAST_ALLOC)

#include <mutex>
#include "../misc.h"

// ----------------------------
//    高速なmemory allocator
// ----------------------------

// 1つのメモリブロック。これは32byteから成る。
struct MemoryChunk
{
    // 例えば、MemoryChunkをN個、連続したメモリとして確保して扱う場合、
    // 1つ目のchunkにのみ、chunk_block_sizeとnext_chunk_indexが存在して(これがchunk headerに相当する)、
    // 2つ目以降のchunkはすべて実際のデータ(データの置き場として使える)である。

    // chunkが何個連続しているか。
    uint64_t chunk_block_size : 16;
    // 例えば、これが4なら、これは連続したメモリに配置された4ブロックから成るchunk。
    // すなわち、全体として4×32 = 128 byteのchunk。
    // chunk headerが8 byteあるから、実際にデータの置き場として使えるのは120 byte。

    // 同じサイズのchunkで、空いている次のchunkへのindex。
    // これは、chunkがpoolに格納されている時に、芋づる式に(chainになっていて)
    // 同じサイズの空きchunkを辿ることができるようになっている。
    // chunkがpoolに格納されていない時は、この変数は未使用。
    uint64_t next_chunk_index : 48;

    // 残り24byteは実データ。
    uint8_t t[24];
};

// 1つのchunkは32byteからなる。
static_assert(sizeof(MemoryChunk) == 32, "sizeof(MemoryChunk) must be 32.");

// 高速なmemory allocator
// 
// これは、ふかうら王で、
//   1. メモリを高速にnew/deleteする必要があった。
//   2. USI_Hashで事前に確保したmemory poolから割り当てたかった。
// ので開発した。
class FastAlloc
{
public:
    // allocした時に与えられるchunkの個数の最大値。
    // 1024*32 - 8 [byte] のalloc()を行う想定なので 1024。
    static const size_t MAX_ALLOC_CHUNK = 1024;

    // Chunkのサイズ[byte]
    const size_t CHUNK_SIZE = sizeof(MemoryChunk);

    FastAlloc()          { chunks = nullptr; chunks_size = 0; }
	~FastAlloc(){ memory_free();}

    /*
        CHUNKは、chunk_indexで指し示す。

        memory_ptrが確保された巨大メモリの先頭アドレスだとすると、
        void* ptr = (uint8_t*)memory_ptr + (chunk_index - 1)* 32;
        が、chunk_indexに対応するchunkの先頭アドレス。
        
        chunk_indexが1の時、巨大メモリの先頭アドレスに一致するが、なぜ0の時に一致するようにしないのかと言うと、
        chunk_indexが0の時は、これをnullptrのように見立てたい(見做したい)からだ。
    */

    // 最初に巨大メモリを確保する。
    // 最初に呼び出すこと。
    // 以前にmemory_alloc()で確保したメモリは開放される。
    // mb の単位は [MB]。
    void memory_alloc(size_t mb);

    // 確保した巨大メモリの開放
    void memory_free();

    // 確保しておいた巨大メモリからメモリを割り当てる。
    void* alloc(size_t s);

    // alloc()で確保したメモリを開放する。
    void free(void* p);

	// 使用率を1000分率で返す。
    int hashfull() const;

	// 残り空きメモリ量を返す。
	size_t rest() const;

protected:
    // 確保された巨大メモリ。ここから切り取って使っていく。
    // 注意)
    // 　size_t index; をポインタ代わりにして
    // 　　auto chunk& = chunks[index] のようにして使いたいのだが、
    // 　その時に index == 0をnullの意味で使いたい。
    // 　そうすると、chunks[0]が未使用のままになってしまうので、
    // 　　auto chunk& = chunks[index - 1]
    // 　のように用いる。(実際のindexに + 1 したものがここに格納されていると考える)
    MemoryChunk* chunks;

    // 確保されたchunksのchunkの数。
    size_t chunks_size;

    // 32[byte]単位で見た、空きメモリの先頭。
    // 
    // 例えば64[byte]の空きchunkなら、chunk_ptr[ pool[64/32] - 1] が poolしている空きメモリ(の一つ)という解釈。
    // pool[chunk_block_num] が0なら、そのサイズのchunkの空きをpoolしていない。
    // 
    // pool[0] == chunk_next_index は、特別な意味で、alloc()された時にそれに該当するchunkの空きを
    // poolしていなかった時に、chunk_ptr[pool[0]]から確保する意味。
    size_t pool[MAX_ALLOC_CHUNK];

    // ↑のにアクセスする時のmutex。
    // 
    // pool[n]用のbarrierはmutexes[n]。
    // pool[0]は特殊な意味で使っているが、これも同様に、
    // pool[0]用のbarrierはmutexes[0]。
    std::mutex mutexes[MAX_ALLOC_CHUNK];
    // pool[N]を操作する時には、その直前と、操作完了後までを
    // mutexes[N]によってlockすることによって、pool[N]の操作自体にatomic性を持たせる。
};

// globalになっていて、"USI_Hash"でこれを確保。
// ふかうら王では、ここから使う、的な…。
extern FastAlloc FAST_ALLOC;

#endif
#endif
