#include "tt.h"

//#include <cassert>
//#include <cstdlib>
//#include <cstring>
//#include <iostream>
//#include <thread>
//#include <vector>

#include "memory.h"
#include "misc.h"
#include "thread.h"
#include "engine.h"

// やねうら王独自拡張
#include "extra/key128.h"
#include "testcmd/unit_test.h"

namespace YaneuraOu {

// ============================================================
//                   置換表エントリー
// ============================================================

// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
// CPUのcache lineに一発で載るというミラクル。
/// ※ cache line sizeは、IntelだとPentium4やPentiumMからでPentiumⅢ(3)までは32byte。
///    そこ以降64byte。AMDだとK8のときには既に64byte。

// TTEntry struct is the 10 bytes transposition table entry, defined as below:
//
// key        16 bit
// depth       8 bit
// generation  5 bit
// pv node     1 bit
// bound type  2 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// These fields are in the same order as accessed by TT::probe(), since memory is fastest sequentially.
// Equally, the store order in save() matches this order.

// TTEntry 構造体は以下のように定義された10バイトのトランスポジションテーブルエントリです:
//
// key        16 bit
// depth       8 bit
// generation  5 bit
// pv node     1 bit
// bound type  2 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// これらのフィールドは、メモリが順次アクセスされるときに最も高速であるため、
// TT::probe()によってアクセスされる順序と同じ順序で配置されています。
// 同様に、save()内の保存順序もこの順序に一致しています。

// ■ 各メンバーの意味
//
// key        16 bit : hash keyの下位16bit(bit0は除くのでbit16..1)
// depth       8 bit : 格納されているvalue値の探索深さ
// move       16 bit : このnodeの最善手(指し手16bit ≒ Move16 , Moveの上位16bitは無視される)
// generation  5 bit : このエントリーにsave()された時のTTの世代カウンターの値
// pv node     1 bit : PV nodeで調べた値であるかのフラグ
// bound type  2 bit : 格納されているvalue値の性質(fail low/highした時の値であるだとか)
// value      16 bit : このnodeでのsearch()の返し値
// evaluation 16 bit : このnodeでのevaluate()の返し値
//
// generation , pv node , bound type をあわせると 5 + 1 + 2 bit = 8 bitとなる。
// TTEntryは、この3つを合わせた変数として generation8 が格納されている。
//
// ■ 補足
//
// Stockfishではkey16は16bit固定であるが、これをやねうら王では、HASH_KEYが64bit,128bit,256bitのときに、それぞれ16bit,64bit,64bitに拡張している。

struct TTEntry {

    // Convert internal bitfields to external types
    // 内部ビットフィールドを外部型に変換します

    TTData read() const {
        return TTData{
          Move(u32(move16.to_u16())),         Value(value16),         Value(eval16),
          Depth(depth8 + DEPTH_ENTRY_OFFSET), Bound(genBound8 & 0x3), bool(genBound8 & 0x4)};
    }

    // このEntryが使われているか？
    bool is_occupied() const;

#if STOCKFISH
    void save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
#else
    void
    save(TTE_KEY_TYPE k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
#endif

    // The returned age is a multiple of TranspositionTable::GENERATION_DELTA
    // 返されるエイジは、TranspositionTable::GENERATION_DELTA の倍数です
    // ⇨ 相対的なageに変換して返す。

    uint8_t relative_age(const uint8_t generation8) const;

   private:
    friend class TranspositionTable;

    TTE_KEY_TYPE key;
    uint8_t      depth8;
    uint8_t      genBound8;
    Move16       move16;
    int16_t      value16;
    int16_t      eval16;
};


// `genBound8` is where most of the details are. We use the following constants to manipulate 5 leading generation bits
// and 3 trailing miscellaneous bits.
// These bits are reserved for other things.

// genBound8には大部分の詳細が含まれています。
// 次の定数を使用して、5ビットの先頭世代ビットと3ビットの末尾のその他のビットを操作します。
// これらのビットは他の用途のために予約されています。
// ⇨ generation8の下位↓bitは、generation用ではなく、別の情報を格納するのに用いる。
//   (PV nodeかどうかのフラグとBoundに用いている。)

static constexpr unsigned GENERATION_BITS = 3;

// increment for generation field
// 世代フィールドをインクリメント
// ⇨ 次のgenerationにするために加算する定数。2の↑乗。

static constexpr int GENERATION_DELTA = (1 << GENERATION_BITS);

// cycle length
// サイクル長
// ⇨ generationを加算していき、1周して戻ってくるまでの長さ。

static constexpr int GENERATION_CYCLE = 255 + GENERATION_DELTA;

// mask to pull out generation number
// TTEntryから世代番号を抽出するためのマスク

static constexpr int GENERATION_MASK = (0xFF << GENERATION_BITS) & 0xFF;


// DEPTH_ENTRY_OFFSET exists because 1) we use `bool(depth8)` as the occupancy check, but
// 2) we need to store negative depths for QS. (`depth8` is the only field with "spare bits":
// we sacrifice the ability to store depths greater than 1<<8 less the offset, as asserted in `save`.)

// DEPTH_ENTRY_OFFSETが存在する理由は、
// 1) `bool(depth8)`を使用してエントリの占有状態を確認しますが、
// 2) QSのために負の深さを保存する必要があるためです。(`depth8`は「予備のビット」を持つ唯一のフィールドです。
// その結果、オフセットを引いた値が1<<8より大きな深さを保存する能力を犠牲にしています。このことは`save`で検証されます。)
// ※ QS = 静止探索

bool TTEntry::is_occupied() const { return bool(depth8); }


// Populates the TTEntry with a new node's data, possibly
// overwriting an old position. The update is not atomic and can be racy.

// TTEntryに新しいノードのデータを格納し、古い局面を上書きする可能性があります。
// この更新はアトミックではなく、競合が発生する可能性があります。

// ⇨ 置換表のエントリーに対して与えられたデータを保存する。上書き動作
//   v    : 探索のスコア
//   eval : 評価関数 or 静止探索の値
//   m    : ベストな指し手(指し手16bit ≒ Move16 , Moveの上位16bitは無視される)
//   gen  : TT.generation()
// 引数のgenは、Stockfishにはないが、やねうら王では学習時にスレッドごとに別の局面を探索させたいので
// スレッドごとに異なるgenerationの値を指定したくてこのような作りになっている。

void TTEntry::save(TTE_KEY_TYPE k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {

	// Preserve the old ttmove if we don't have a new one
	// 新しいttmoveがない場合、古いttmoveを保持します

	if (m || k != key)
		move16 = m.to_move16();

	// Overwrite less valuable entries (cheapest checks first)
	// より価値の低いエントリを上書きします（最も簡単にできるチェックを先に行う）

	if (b == BOUND_EXACT || k != key || d - DEPTH_ENTRY_OFFSET + 2 * pv > depth8 - 4
		|| relative_age(generation8))
	{
		assert(d > DEPTH_ENTRY_OFFSET);
		assert(d < 256 + DEPTH_ENTRY_OFFSET);

		key       = TTE_KEY_TYPE(k);
		depth8    = uint8_t(d - DEPTH_ENTRY_OFFSET);
		genBound8 = uint8_t(generation8 | uint8_t(pv) << 2 | b);
		value16   = int16_t(v);
		eval16    = int16_t(ev);

		// value,evalが適切な範囲であるか(やねうら王独自追加)
		ASSERT_LV3(-VALUE_INFINITE <   v  && v < VALUE_INFINITE  ||  v == VALUE_NONE);
		ASSERT_LV3(-VALUE_MAX_EVAL <= ev && ev <= VALUE_MAX_EVAL || ev == VALUE_NONE);
		
	}
	// depthが高くてBOUND_EXACTでないときは、BOUND_EXACTと差別化するためにdepthを1引いておく。
	else if (depth8 + DEPTH_ENTRY_OFFSET >= 5 && Bound(genBound8 & 0x3) != BOUND_EXACT)
		depth8--;
}


uint8_t TTEntry::relative_age(const uint8_t generation8) const {
	// Due to our packed storage format for generation and its cyclic
	// nature we add GENERATION_CYCLE (256 is the modulus, plus what
	// is needed to keep the unrelated lowest n bits from affecting
	// the result) to calculate the entry age correctly even after
	// generation8 overflows into the next cycle.

	// 世代のパックされた保存形式とその循環的な性質により、
	// 世代エイジを正しく計算するために、GENERATION_CYCLEを加えます
	//  （256がモジュロとなり、関係のない下位nビットが
	// 結果に影響を与えないようにするために必要な値も加えます）。
	// これにより、generation8が次のサイクルにオーバーフローした後でも、
	// エントリのエイジを正しく計算できます。

	// ■ 補足情報
	//
	// generationは256になるとオーバーフローして0になるのでそれをうまく処理できなければならない。
	// a,bが8bitであるとき ( 256 + a - b ) & 0xff　のようにすれば、オーバーフローを考慮した引き算が出来る。
	// このテクニックを用いる。
	// いま、
	//   a := generationは下位3bitは用いていないので0。
	//   b := genBound8は下位3bitにはBoundが入っているのでこれはゴミと考える。
	// ( 256 + a - b + c) & 0xfc として c = 7としても結果に影響は及ぼさない、かつ、このゴミを無視した計算が出来る。

	return (GENERATION_CYCLE + generation8 - genBound8) & GENERATION_MASK;
}


// TTWriter is but a very thin wrapper around the pointer
// TTWriterはポインタを包む非常に薄いラッパーに過ぎません

TTWriter::TTWriter(TTEntry* tte) :
	entry(tte) {}

#if STOCKFISH
void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {
    entry->save(k, v, pv, b, d, m, ev, generation8);
}
#else

void TTWriter::write(
  const Key k_, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {

#if HASH_KEY_BITS <= 64
    const TTE_KEY_TYPE k = TTE_KEY_TYPE(k_);
#else
    const TTE_KEY_TYPE k = TTE_KEY_TYPE(k_.extract64<1>());
#endif

    entry->save(k, v, pv, b, d, m, ev, generation8);
}
#endif


// A TranspositionTable is an array of Cluster, of size clusterCount. Each cluster consists of ClusterSize number
// of TTEntry. Each non-empty TTEntry contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is prefetched when possible.

// TranspositionTableは、clusterCountのサイズを持つClusterの配列です。
// 各クラスターはClusterSize個のTTEntryで構成されます。
// 各非空のTTEntryは、正確に1つの局面に関する情報を含んでいます。
// クラスターのサイズは、パフォーマンスを最大化するためにキャッシュラインのサイズを割り切れるべきです。
// キャッシュラインは可能な場合にプリフェッチされます。

// ■ 補足情報
//
// StockfishではClusterSize == 3固定だが、やねうら王では、ビルドオプションで変更できるようになっている。
//
// 1クラスターにおけるTTEntryの数
// TT_CLUSTER_SIZE == 2のとき、TTEntry 10bytes×3つ + 2(padding) =  32bytes
// TT_CLUSTER_SIZE == 3のとき、TTEntry 16bytes×2つ + 0(padding) =  32bytes
// TT_CLUSTER_SIZE == 4のとき、TTEntry 16bytes×4つ + 0(padding) =  64bytes
// TT_CLUSTER_SIZE == 6のとき、TTEntry 16bytes×6つ + 0(padding) =  96bytes
// TT_CLUSTER_SIZE == 8のとき、TTEntry 16bytes×8つ + 0(padding) = 128bytes

static constexpr int ClusterSize = TT_CLUSTER_SIZE;

struct Cluster {
	TTEntry entry[ClusterSize];

#if TT_CLUSTER_SIZE == 3
	char    padding[2];  // Pad to 32 bytes
						 // 全体を32byteぴったりにするためのpadding
#endif
};

// static_assert(sizeof(Cluster) == 32, "Suboptimal Cluster size");
static_assert((sizeof(Cluster) % 32) == 0, "Unexpected Cluster size");

// Sets the size of the transposition table,
// measured in megabytes. Transposition table consists
// of clusters and each cluster consists of ClusterSize number of TTEntry.

// トランスポジションテーブルのサイズをメガバイト単位で設定します。
// トランスポジションテーブルはクラスターで構成されており、
// 各クラスターはClusterSize個のTTEntryで構成されます。

void TranspositionTable::resize(size_t mbSize, ThreadPool& threads) {
#if STOCKFISH
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

#else

	// mbSizeの単位は[MB]なので、ここでは1MBの倍数単位のメモリが確保されるが、
	// 仕様上は、1MBの倍数である必要はない。
	size_t newClusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

	// clusterCountは偶数でなければならない。
	// この理由については、TTEntry::first_entry()のコメントを見よ。
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
#endif

	// tableはCacheLineSizeでalignされたメモリに配置したいので、CacheLineSize-1だけ余分に確保する。
	// callocではなくmallocにしないと初回の探索でTTにアクセスするとき、特に巨大なTTだと
	// 極めて遅くなるので、mallocで確保して自前でゼロクリアすることでこれを回避する。
	// cf. Explicitly zero TT upon resize. : https://github.com/official-stockfish/Stockfish/commit/2ba47416cbdd5db2c7c79257072cd8675b61721f

	// Large Pageを確保する。ランダムメモリアクセスが5%程度速くなる。

	table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

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

// Initializes the entire transposition table to zero,
// in a multi-threaded way.

// トランスポジションテーブル全体をマルチスレッドでゼロに初期化します。

void TranspositionTable::clear(ThreadPool& threads) {

#if defined(TANUKI_MATE_ENGINE) || defined(YANEURAOU_MATE_ENGINE)
	// MateEngineではこの置換表は用いないのでクリアもしない。
	return;
#endif

	generation8 = 0;

	// Stockfishのコード
#if 0
	const size_t threadCount = threads.num_threads();

	for (size_t i = 0; i < threadCount; ++i)
	{
		threads.run_on_thread(i, [this, i, threadCount]() {
			// Each thread will zero its part of the hash table
			const size_t stride = clusterCount / threadCount;
			const size_t start = stride * i;
			const size_t len = i + 1 != threadCount ? stride : clusterCount - start;

			std::memset(&table[start], 0, len * sizeof(Cluster));
			});
	}

	for (size_t i = 0; i < threadCount; ++i)
		threads.wait_on_thread(i);
#endif

	auto size = clusterCount * sizeof(Cluster);

	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishのここにあったコードは、独自の置換表を実装した時にも使いたいため、tt.cppに移動させた。
	Tools::memclear(threads, "USI_Hash", table, size);
}

// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which match the current generation.

// 検索中のハッシュテーブルの占有率を概算して返します。
// ハッシュはUCIプロトコルに従って、xパーミルで満たされています。
// 現在の世代と一致するエントリのみをカウントします。

int TranspositionTable::hashfull(int maxAge) const {
	int maxAgeInternal = maxAge << GENERATION_BITS;
	int cnt = 0;
	for (int i = 0; i < 1000; ++i)
		for (int j = 0; j < ClusterSize; ++j)
			cnt += table[i].entry[j].is_occupied()
			&& table[i].entry[j].relative_age(generation8) <= maxAgeInternal;

	return cnt / ClusterSize;
}

void TranspositionTable::new_search() {

	// increment by delta to keep lower bits as is
	// 下位ビットをそのままにして、デルタでインクリメントします

	generation8 += GENERATION_DELTA;
}


uint8_t TranspositionTable::generation() const { return generation8; }

// Looks up the current position in the transposition
// table. It returns true if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable TTEntry
// to be replaced later. The replace value of an entry is calculated as its depth
// minus 8 times its relative age. TTEntry t1 is considered more valuable than
// TTEntry t2 if its replace value is greater than that of t2.

// 現在の局面をトランスポジションテーブルで検索します。局面が見つかった場合、trueを返します。
// そうでない場合、falseと、後で置き換えるための空または最も価値の低いTTEntryへのポインタを返します。
// エントリの置き換え値は、その深さから相対的なエイジの8倍を引いたものとして計算されます。
// TTEntry t1は、t2の置き換え値より大きい場合、t2よりも価値があると見なされます。

// 🌈 やねうら王独自拡張
//    probe()してhitしたときに ttData.moveは Move16のままなので ttData.move32(pos)を用いて取得する必要がある。
//    そこで、probe()の第2引数にPositionを渡すようにして、Move16ではなくMoveに変換されたTTDataを返すことにする。

std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key, const Position& pos) const {

    TTEntry* const tte = first_entry(key, pos.side_to_move());

#if HASH_KEY_BITS <= 64
    const TTE_KEY_TYPE key_for_ttentry = TTE_KEY_TYPE(key);
#else
    const TTE_KEY_TYPE key_for_ttentry = TTE_KEY_TYPE(key.extract64<1>());
#endif

	// Use the low 16 bits as key inside the cluster
	// クラスター内で下位16ビットをキーとして使用します

	for (int i = 0; i < ClusterSize; ++i)
		if (tte[i].key == key_for_ttentry)

			// This gap is the main place for read races.
			// After `read()` completes that copy is final, but may be self-inconsistent.

			// このギャップが、読み取り競合の主な発生場所です。
			// `read()`が完了した後、そのコピーは最終的なものですが、自己矛盾している可能性があります。

		{
			auto ttData = tte[i].read();
			if (ttData.move)
			{
				// TTEntryにMoveが登録されていて、それを32bit化しようとして失敗したら、
				// 置換表にhitしなかったという扱いにする。
				Move move = pos.to_move(ttData.move.to_move16());
				if (!move)
					continue;
				ttData.move = move;
			}
			return { tte[i].is_occupied(), ttData, TTWriter(&tte[i]) };
		}

	// Find an entry to be replaced according to the replacement strategy
	// 置換戦略に従って、置き換えるエントリを見つけます

	TTEntry* replace = tte;
	for (int i = 1; i < ClusterSize; ++i)
		if (replace->depth8 - replace->relative_age(generation8)
			> tte[i].depth8 - tte[i].relative_age(generation8))
			replace = &tte[i];

	return { false,
			TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_ENTRY_OFFSET, BOUND_NONE, false},
			TTWriter(replace) };
}

// keyを元にClusterのindexを求めて、その最初のTTEntry*を返す。内部実装用。
// ※　ここで渡されるkeyのbit 0は局面の手番フラグ(Position::side_to_move())であると仮定している。

TTEntry* TranspositionTable::first_entry(const Key& key_, Color side_to_move) const {

#if STOCKFISH

    return &table[mul_hi64(key, clusterCount)].entry[0];
	// 💡 mul_hi64は、64bit * 64bitの掛け算をして下位64bitを取得する関数。

	// key(64bit) × clusterCount / 2^64 の値は 0 ～ clusterCount - 1 である。
    // 掛け算が必要にはなるが、こうすることで custerCountを2^Nで確保しないといけないという制約が外れる。
    // cf. Allow for general transposition table sizes. : https://github.com/official-stockfish/Stockfish/commit/2198cd0524574f0d9df8c0ec9aaf14ad8c94402b

#else

	// ⚠ Key128, Key256ならば、これで key_.extract64<0>() の意味になる。
	const Key64 key = Key64(key_);

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
	return &table[(index & ~1) | side_to_move].entry[0];

#endif
}

#if defined(EVAL_LEARN)
// スレッド数が変更になった時にThread.set()から呼び出される。
// これに応じて、スレッドごとに保持しているTTを初期化する。
void TranspositionTable::init_tt_per_thread()
{
	// スレッド数
	size_t thread_size = Threads.size();

	// エンジン終了時にThreads.set(0)で全スレッド終了させるコードが書いてあるので、
	// そのときに、Threads.size() == 0の状態で呼び出される。
	// ここで抜けないと、このあとゼロ除算することになる。
	if (thread_size == 0)
		return;

	// 1スレッドあたりのクラスター数(端数切捨て)
	// clusterCountは2の倍数でないと駄目なので、端数を切り捨てるためにLSBを0にする。
	size_t clusterCountPerThread = (clusterCount / thread_size) & ~(size_t)1;

	ASSERT_LV3((clusterCountPerThread & 1) == 0);

	// これを、自分が確保したglobalな置換表用メモリから切り分けて割当てる。
	for (size_t i = 0; i < thread_size; ++i)
	{
		auto& tt = Threads[i]->tt;
		tt.clusterCount = clusterCountPerThread;
		tt.table = this->table + clusterCountPerThread * i;
	}
}
#endif

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

		Key posKey = pos.key();
		auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);
		for (int i = 0; i < 10; ++i)
		{
			Value v = Value(i*100-500);
			bool pv = (i % 2) != 0;
			Bound b = BOUND_LOWER;
			Depth d = 16 + i;
			Move m = make_move(SQ_77, SQ_76, BLACK, PAWN);
			Value ev = Value(i*200-1000);
			int g = 8 * 5; /* 8の倍数でないと駄目 */
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

} // namespace YaneuraOu

//}  // namespace Stockfish
