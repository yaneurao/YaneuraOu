#include "tt.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#if STOCKFISH
#include <numeric>
#include <vector>

#include "bitboard.h"
#endif

#include "memory.h"
#include "misc.h"
#include "thread.h"

#if !STOCKFISH
#include "engine.h"

// やねうら王独自拡張

#include "extra/key128.h"
#include "testcmd/unit_test.h"
#endif

namespace YaneuraOu {

// ============================================================
//                   置換表エントリー
// ============================================================

// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
// CPUのcache lineに一発で載るというミラクル。
/// ※ cache line sizeは、IntelだとPentium4やPentiumMからでPentiumⅢ(3)までは32byte。
///    そこ以降64byte。AMDだとK8のときには既に64byte。

// TTEntry struct is the 8 bytes transposition table entry, defined as:
//
// depth       8 bit
// pv node     1 bit
// bound type  2 bit
// generation  5 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// The 21-bit keys are packed separately in Cluster::keys.
//
// We use `bool(depth8)` as the cheap internal occupancy check, corresponding
// to `depth == DEPTH_NONE` externally, so we offset the internal depth by DEPTH_NONE.
//
// Pv, bound and generation are packed in a single byte.

// TTEntry 構造体は以下のように定義された8バイトのトランスポジションテーブルエントリです:
//
// depth       8 bit
// pv node     1 bit
// bound type  2 bit
// generation  5 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// これらのフィールドは、メモリが順次アクセスされるときに最も高速であるため、
// TT::probe()によってアクセスされる順序と同じ順序で配置されています。
// 同様に、save()内の保存順序もこの順序に一致しています。
//
// bool(depth8)を安価な内部占有チェックとして使います。
// 外部的にはdepth == DEPTH_NONEに対応するため、内部のdepthはDEPTH_NONEだけ下駄履きして保存します。
//
// pv, bound, generationは1バイトに詰め込まれています。

// ■ 各メンバーの意味
//
// depth       8 bit : 格納されているvalue値の探索深さ
// move       16 bit : このnodeの最善手(指し手16bit ≒ Move16 , Moveの上位16bitは無視される)
// generation  5 bit : このエントリーにsave()された時のTTの世代カウンターの値
// pv node     1 bit : PV nodeで調べた値であるかのフラグ
// bound type  2 bit : 格納されているvalue値の性質(fail low/highした時の値であるだとか)
// value      16 bit : このnodeでのsearch()の返し値
// evaluation 16 bit : このnodeでのevaluate()の返し値
//
// pv node , bound type , generation をあわせると 1 + 2 + 5 bit = 8 bitとなる。
// TTEntryは、この3つを合わせた変数として generation8 が格納されている。
//

static constexpr u8 GENERATION_BITS = 5;                            // genBound8のうち、generationに使うbit数。
static constexpr u8 GENERATION_MASK = (1 << GENERATION_BITS) - 1;   // genBound8からgenerationだけを取り出すためのマスク。
static constexpr u8 BOUND_SHIFT     = GENERATION_BITS;              // Boundを格納するbit位置。generationの直後に置く。
static constexpr u8 BOUND_MASK      = 0b11 << BOUND_SHIFT;          // genBound8からBoundの2bitだけを取り出すためのマスク。
static constexpr u8 PV_SHIFT        = BOUND_SHIFT + 2;              // PV nodeフラグを格納するbit位置。Boundの直後に置く。
static constexpr u8 PV_MASK         = 1 << PV_SHIFT;                // genBound8からPV nodeフラグだけを取り出すためのマスク。

struct TTEntry {

    // Extract data from the TT entry. We have to convert TT internal bitfields
    // to external types.
	// TTエントリからデータを取り出す。TT内部のビットフィールドを
	// 外部で使用する型に変換する必要がある。

	TTData read() const {
#if STOCKFISH
        return TTData{Move(move16),
#else
		// やねうら王では、32bit Moveを設定する必要があるが、
		// この時点では上位16bitは不明であるため、0埋めしておく。
		return TTData{Move(u32(Move16(move16).to_u16())),
#endif
				Value(value16),
				Value(eval16),
				Depth(DEPTH_NONE + depth8),
				Bound((genBound8 & BOUND_MASK) >> BOUND_SHIFT),
				bool(genBound8 & PV_MASK)};
	}

    // Check if the TT entry is occupied
	// このEntryが使われているか？
	bool is_occupied() const { return bool(depth8); }

    // Return TT entry age
    // TT entryの世代を返す
	u8 relative_age(const u8 curr_generation) const;

   private:
    friend class TranspositionTable;

    friend struct TTWriter;

    RelaxedAtomic<u8>   depth8;
    RelaxedAtomic<u8>   genBound8;
#if STOCKFISH
    RelaxedAtomic<Move> move16;
#else
    RelaxedAtomic<Move16> move16;
#endif
    RelaxedAtomic<i16>  value16;
    RelaxedAtomic<i16>  eval16;
};
// A TranspositionTable is an array of Cluster, of size clusterCount. Each
// cluster consists of ClusterSize number of TTEntry. Each non-empty TTEntry
// contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is
// prefetched when possible.

// TranspositionTableは、clusterCount個のClusterからなる配列である。
// 各Clusterは、ClusterSize個のTTEntryで構成される。空でない各TTEntryには、
// ちょうど1つの局面に関する情報が格納される。最高の性能を得るためには、
// Clusterのサイズがキャッシュラインのサイズを割り切れるようにすべきである。
// これは、可能な場合にキャッシュラインがプリフェッチされるためである。

static constexpr int ClusterSize = 3;
static constexpr int KeyBits     = 21;
static constexpr u64 KeyMask     = (1ULL << KeyBits) - 1;



struct Cluster {

    // keys contains the low 21 bits of the Zobrist keys of the 3 TT entries in this cluster,
    // packed together into 63 bits of an u64
	// keysには、このCluster内の3つのTTEntryのZobristキーの下位21ビットが格納される。
	// これらは合計63ビットにパックされ、1つのu64に格納される。
    RelaxedAtomic<u64> keys;
    TTEntry            entry[ClusterSize];
};
static_assert(sizeof(Cluster) == 32, "Suboptimal Cluster size");

// Populates the TTEntry and its key with a new node's data, possibly overwriting
// an old position. The update is non-atomic and can be racy. We convert external
// types to internal bitfields.
// TTEntryとそのキーに新しいノードのデータを格納する。この際、古い局面のデータを
// 上書きする可能性がある。更新は非アトミックであり、競合が発生する可能性がある。
// 外部で使用する型を内部のビットフィールドに変換する。


void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, u8 curr_generation) {

#if !STOCKFISH
	// value,evalが適切な範囲であるか(やねうら王独自追加)
	ASSERT_LV3(-VALUE_INFINITE <   v  && v < VALUE_INFINITE  ||  v == VALUE_NONE);
	ASSERT_LV3(-VALUE_MAX_EVAL <= ev && ev <= VALUE_MAX_EVAL || ev == VALUE_NONE);
#endif

    const int shift   = KeyBits * index;
    const u64 keys    = cluster->keys;
    const u64 keyDiff = ((keys >> shift) ^ k) & KeyMask;
    auto&     entry   = cluster->entry[index];

	// Preserve the old ttmove if we don't have a new one
	// 新しいttmoveがない場合、古いttmoveを保持します

	if (m || keyDiff)
	#if STOCKFISH
        entry.move16 = m;
	#else
        entry.move16 = m.to_move16();
    #endif

	// Overwrite less valuable entries (cheapest checks first)
	// より価値の低いエントリを上書きします（最も簡単にできるチェックを先に行う）

	if (b == BOUND_EXACT || keyDiff || d - DEPTH_NONE + 2 * pv > entry.depth8 - 4
		|| entry.relative_age(curr_generation))
	{
		assert(d > DEPTH_NONE);
		assert(d - DEPTH_NONE < 256);
		assert(curr_generation <= GENERATION_MASK); // TT::new_search() plays nice

        cluster->keys   = keys ^ (keyDiff << shift);
        entry.depth8    = u8(d - DEPTH_NONE);
        entry.genBound8 = u8(curr_generation | b << BOUND_SHIFT | u8(pv) << PV_SHIFT);
        entry.value16   = i16(v);
        entry.eval16    = i16(ev);
    }

    // Secondary aging. Important for elementary mate finding.
    // (*Scaler) Secondary aging on entries relevant to singular extensions
    // generally scales poorly and requires VVLTC verification.

	// 二次的なエイジング。基本的な詰みの発見に重要である。
	// (*Scaler) Singular Extensionに関連するエントリに対する二次的なエイジングは、
	// 一般にスケーリング特性が悪く、VVLTCによる検証が必要である。

    else if (entry.depth8 + DEPTH_NONE >= 5
             && Bound((entry.genBound8 & BOUND_MASK) >> BOUND_SHIFT) != BOUND_EXACT)
    {
        auto v16 = entry.value16;

        // Guard against racy underflows, default to "unoccupied"
        if (std::abs(v16) < VALUE_INFINITE && is_decisive(v16))
            entry.depth8 = std::max(int(entry.depth8) - 1, 0);
    }

}
// Returns this entry's age. We count generations like clocks count hours,
// i.e. we require 0 - 1 == 31. Unsigned subtraction guarantees the required
// borrowing regardless of the upper pv/bound bits.

// このエントリの世代差（age）を返す。時計が時刻を数えるのと同様に、世代を循環させて
// 数える。つまり、0 - 1 == 31 となる必要がある。符号なし整数の減算により、
// 上位のpv/boundビットに関係なく、必要な桁借りが保証される。

u8 TTEntry::relative_age(const u8 curr_generation) const {
	return (curr_generation - genBound8) & GENERATION_MASK;
}


// TTWriter is but a very thin wrapper around the pointer
// TTWriterはポインタを包む非常に薄いラッパーに過ぎません

TTWriter::TTWriter(Cluster* cl, int idx) :
    cluster(cl),
    index(idx) {}

// Flag a TTEntry as useless (decrementing the stored depth by the given penalty)
// TTEntryを有用でないものとしてマークする（保存されているdepthから指定された
// ペナルティ値を減算する）。
void TTWriter::penalize(int penalty) {
    auto& entry = cluster->entry[index];
    // Guard against racy underflows, default to "unoccupied"
    entry.depth8 = std::max(int(entry.depth8) - penalty, 0);
}


// Sets the size of the transposition table, measured in megabytes.
// Transposition table consists of clusters and each cluster consists
// of ClusterSize number of TTEntry.

// 置換表のサイズをメガバイト単位で設定する。
// 置換表は複数のClusterから構成され、各Clusterは
// ClusterSize個のTTEntryから構成される。

void TranspositionTable::resize(usize mbSize, ThreadPool& threads) {
#if STOCKFISH
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

    usize ttBytes = clusterCount * sizeof(Cluster);

    // Request 1GB pages if we'd get at least eight per NUMA node, to avoid
    // memory oversubscription
    bool hugePageHint = ttBytes >= threads.numa_nodes() * HugePageSize * 8;

    table = static_cast<Cluster*>(aligned_large_pages_alloc_with_hint(ttBytes, hugePageHint));


#else

	// mbSizeの単位は[MB]なので、ここでは1MBの倍数単位のメモリが確保されるが、
	// 仕様上は、1MBの倍数である必要はない。
	usize newClusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

	// clusterCountは偶数でなければならない。
	// この理由については、TranspositionTable::get_cluster()のコメントを見よ。
	// しかし、1024 * 1024 / sizeof(Cluster)の部分、sizeof(Cluster)==64なので、
	// これを掛け算するから2の倍数である。
	ASSERT_LV3((newClusterCount & 1) == 0);

	// 同じサイズなら確保しなおす必要はない。

	// Stockfishのコード、問答無用で確保しなおしてゼロクリアしているが、
	// ゼロクリアの時間も馬鹿にならないのであまり良いとは言い難い。

	if (newClusterCount == clusterCount)
		return;

	aligned_large_pages_free(table);

	clusterCount = newClusterCount;

	// tableはCacheLineSizeでalignされたメモリに配置したいので、CacheLineSize-1だけ余分に確保する。
	// callocではなくmallocにしないと初回の探索でTTにアクセスするとき、特に巨大なTTだと
	// 極めて遅くなるので、mallocで確保して自前でゼロクリアすることでこれを回避する。
	// cf. Explicitly zero TT upon resize. : https://github.com/official-stockfish/Stockfish/commit/2ba47416cbdd5db2c7c79257072cd8675b61721f

	// Large Pageを確保する。ランダムメモリアクセスが5%程度速くなる。

	table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

#endif

	if (!table)
	{
		std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
		exit(EXIT_FAILURE);
	}

#if STOCKFISH
	clear(threads);

	// →　Stockfish、ここでclear()呼び出しているが、Search::clear()からTT.clear()を呼び出すので
	// 二重に初期化していることになると思う。
#endif
}

// Initializes the entire transposition table to zero, in a multi-threaded way
// トランスポジションテーブル全体をマルチスレッドでゼロに初期化します。

void TranspositionTable::clear(ThreadPool& threads) {

#if !STOCKFISH
#if defined(TANUKI_MATE_ENGINE) || defined(YANEURAOU_MATE_ENGINE)
	// MateEngineではこの置換表は用いないのでクリアもしない。
	return;
#endif
#endif

	generation8 = 0;

#if STOCKFISH
	const usize threadCount = threads.num_threads();

    std::vector<usize> threadToNuma = threads.get_bound_thread_to_numa_node();

    std::vector<usize> order(threadCount);
    std::iota(order.begin(), order.end(), 0);

    // To promote good NUMA distribution (esp. with huge pages), we permute
    // threads so that all threads in a NUMA node clear a contiguous region.
    if (threadToNuma.size() == threadCount)
    {
        std::stable_sort(order.begin(), order.end(), [&threadToNuma](usize t1, usize t2) {
            return threadToNuma.at(t1) < threadToNuma.at(t2);
        });
    }

    for (usize i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(order[i], [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const usize stride = clusterCount / threadCount;
            const usize start  = stride * i;
            const usize len    = i + 1 != threadCount ? stride : clusterCount - start;

            std::memset(static_cast<void*>(&table[start]), 0, len * sizeof(Cluster));
        });
    }

    for (usize i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);

#else

	auto size = clusterCount * sizeof(Cluster);

	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishのここにあったコードは、独自の置換表を実装した時にも使いたいため、tt.cppに移動させた。
	Tools::memclear(threads, "USI_Hash", table, size);
#endif
}

// Returns an approximation of the hashtable occupation during a search.
// The hash is x permill full, as per UCI protocol. Only counts entries
// which are younger than maxAge.

// 探索中のハッシュテーブルの使用率の概算値を返す。
// UCIプロトコルの仕様に従い、使用率は千分率で表す。
// ageがmaxAgeより小さいエントリのみをカウントする。

int TranspositionTable::hashfull(int maxAge) const {
	int cnt = 0;
	for (int i = 0; i < 1000; ++i)
		for (int j = 0; j < ClusterSize; ++j)
			cnt += table[i].entry[j].is_occupied()
			&& table[i].entry[j].relative_age(generation8) <= maxAge;

	return cnt / ClusterSize;
}

// Must be called at the beginning of each root search to track entry aging
// エントリのエイジングを管理するため、各ルート探索の開始時に呼び出す必要がある。
void TranspositionTable::new_search() {

	++generation8;

	// Don't overflow into the other bits of TTEntry::genBound8
	// TTEntry::genBound8の他のビットへ溢れないようにします。
	generation8 &= GENERATION_MASK;
}


// The current age, used when writing new data to the TT
// 現在のage。TTに新しいデータを書き込む際に使用する。
u8 TranspositionTable::generation() const { return generation8; }

// Looks up the current position in the transposition table. Calling probe(key)
// returns true if the key is found (which may be a collision) and has non-null
// data. Otherwise, it returns false and a pointer to an empty or least valuable
// TTEntry to be replaced later. The value of an entry is its depth minus 8 times
// its relative age.

// 置換表から現在の局面を検索する。probe(key)を呼び出すと、キーが見つかり
// （ハッシュ衝突の可能性もある）、かつデータが空でなければtrueを返す。
// それ以外の場合はfalseを返し、後で置換するための、空のTTEntryまたは
// 最も価値の低いTTEntryへのポインタを返す。エントリの価値は、そのdepthから
// 相対的なageの8倍を引いた値である。

// 🌈 やねうら王独自拡張
//    probe()してhitしたときに ttData.moveは Move16のままなので ttData.move32(pos)を用いて取得する必要がある。
//    そこで、probe()の第2引数にPositionを渡すようにして、Move16ではなくMoveに変換されたTTDataを返すことにする。

#if STOCKFISH
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key) const {
    Cluster* const cluster = get_cluster(key);

#else
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key, const Position& pos) const {
    Cluster* const cluster = get_cluster(key, pos.side_to_move());
#endif

    auto&          tte     = cluster->entry;
    const u64      key21   = key & KeyMask;  // Use the low 21 bits as key inside the cluster

    constexpr u64 KeyLsb  = 1ULL | (1ULL << KeyBits) | (1ULL << 2 * KeyBits);
    constexpr u64 KeyMsb  = KeyLsb << (KeyBits - 1);
    const u64     keys    = cluster->keys;
    const u64     keyXors = keys ^ (KeyLsb * key21);
    // This bit twiddling trick will produce a one in the last (=most significant) bit of
    // the first 21bit block matching key21. The bits before that first matching block will
    // be all zeros, but after the first matching block there could be garbage.
    const u64 matches = (keyXors - KeyLsb) & ~keyXors & KeyMsb;

    if (matches)
    {
        static_assert(KeyBits == 21, "Index computation below assumes KeyBits == 21");
        const int i = (int(lsb(matches)) >> 4) - 1;
        // This gap is the main place for read races.
        // After `read()` completes that copy is final, but may be self-inconsistent.
		// このギャップが、読み取り競合の主な発生場所です。
		// `read()`が完了した後、そのコピーは最終的なものですが、自己矛盾している可能性があります。

#if STOCKFISH
        return {tte[i].is_occupied(), tte[i].read(), TTWriter(cluster, i)};
#else
		auto ttData = tte[i].read();
		if (ttData.move)
		{
			// TTEntryにMoveが登録されていて、それを32bit化しようとして失敗したら、
			// 置換表にhitしなかったという扱いにする。(99.99%ありえないのだが..)
			Move move = pos.to_move(ttData.move.to_move16());
			if (move)
			{
				ttData.move = move;
				return { tte[i].is_occupied(), ttData, TTWriter(cluster, i) };
			}
		} else
			// Move::none()が書き出されているentryはそのまま返す。
	        return {tte[i].is_occupied(), ttData, TTWriter(cluster, i)};

	    // 最初の候補の32bit化に失敗した場合だけ、後続を調べる。
	    // (救済措置。非常にレアケースではあるが..)
	    for (int j = i + 1; j < ClusterSize; ++j)
	    {
	        if (((keys >> (KeyBits * j)) & KeyMask) != key21)
	            continue;

	        auto data = tte[j].read();
	        if (data.move)
	        {
	            Move move = pos.to_move(data.move.to_move16());
	            if (!move)
	                continue;

	            data.move = move;
	        }

	        return {tte[j].is_occupied(), data, TTWriter(cluster, j)};
		}
#endif
    }

	// Find an entry to be replaced according to the replacement strategy
	// 置換戦略に従って、置き換えるエントリを見つけます

    int replace      = 0;
    int replaceValue = tte[0].depth8 - 8 * tte[0].relative_age(generation8);
    for (int i = 1; i < ClusterSize; ++i)
    {
        const int value = tte[i].depth8 - 8 * tte[i].relative_age(generation8);
        if (replaceValue > value)
        {
            replace      = i;
            replaceValue = value;
        }
    }

    return {false, TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false},
            TTWriter(cluster, replace)};
}

// The hash function; its only external use is memory prefetching
// ハッシュ関数。外部からはメモリのプリフェッチにのみ使用される。

// keyを元にClusterのindexを求めて、そのCluster*を返す。
// ※　ここで渡されるkeyのbit 0は局面の手番フラグ(Position::side_to_move())であると仮定している。

#if STOCKFISH
Cluster* TranspositionTable::get_cluster(const Key key) const {
#else
Cluster* TranspositionTable::get_cluster(const Key key, Color side_to_move) const {
#endif

#if STOCKFISH

    return &table[mul_hi64(key, clusterCount)];
	// 💡 mul_hi64は、64bit * 64bitの掛け算をして下位64bitを取得する関数。

	// key(64bit) × clusterCount / 2^64 の値は 0 ～ clusterCount - 1 である。
    // 掛け算が必要にはなるが、こうすることで custerCountを2^Nで確保しないといけないという制約が外れる。
    // cf. Allow for general transposition table sizes. : https://github.com/official-stockfish/Stockfish/commit/2198cd0524574f0d9df8c0ec9aaf14ad8c94402b

#else

	/*
		📓

		やねうら王では、cluster indexのbit0(先後フラグ) に手番が反映される必要がある。
		このときclusterCountが奇数だと、(index & ~(u64)1) | side_to_move のようにしたときに、
		(clusterCount - 1)が上限であるべきなのにclusterCountになりかねない。

		そこでclusterCountは偶数であるという制約を課す。
	*/
	ASSERT_LV3((clusterCount & 1) == 0);

	// 💡 key * clusterCount / 2^64 をするので、indexは 0 ～ clusterCount-1 の範囲となる。
	uint64_t index = mul_hi64((u64)key, clusterCount);

	// indexは0～ clusterCount -1の範囲にある。このbit 0を手番に変更する。
	// ⚠ Colorの実体はuint8で、0,1の値しか取らないものとする。
	return &table[(index & ~1) | side_to_move];

#endif
}

#if !STOCKFISH
// ----------------------------------
//			UnitTest
// ----------------------------------

void TranspositionTable::UnitTest(Test::UnitTester& unittest, IEngine& engine)
{
	auto section1 = unittest.section("TT");
	{
		TranspositionTable tt;

		// 1024[MB]確保
		tt.resize(1024,engine.get_threads());
		tt.clear(engine.get_threads());

		auto section2 = unittest.section("probe()");
		Position pos;
		StateInfo si;
		pos.set_hirate(&si);

		u64 posKey = u64(pos.key());
		auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);
		for (int i = 0; i < 10; ++i)
		{
			Value v = Value(i*100-500);
			bool pv = (i % 2) != 0;
			Bound b = BOUND_LOWER;
			Depth d = 16 + i;
			Move m = make_move(SQ_77, SQ_76, BLACK, PAWN);
			Value ev = Value(i*200-1000);
			int g = 5;
			ttWriter.write(posKey, v, pv, b, d, m, ev, g);

			auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);
			bool ok = true;
			ok &= ttHit;
			if (ttHit)
			{
				ok &= ttData.value == v;
				ok &= ttData.is_pv == pv;
				ok &= ttData.bound == b;
				ok &= ttData.depth == d;
				ok &= ttData.move  == m;
				ok &= ttData.eval  == ev;
			}
			unittest.test("write & probe", ok);
		}
	}
}
#endif

} // namespace YaneuraOu
