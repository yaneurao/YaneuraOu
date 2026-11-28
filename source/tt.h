#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include <tuple>

#include "misc.h"
#include "memory.h"
#include "types.h"

namespace YaneuraOu {

class ThreadPool;

// cf.【決定版】コンピュータ将棋のHASHの概念について詳しく : http://yaneuraou.yaneu.com/2018/11/18/%E3%80%90%E6%B1%BA%E5%AE%9A%E7%89%88%E3%80%91%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E5%B0%86%E6%A3%8B%E3%81%AEhash%E3%81%AE%E6%A6%82%E5%BF%B5%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/

struct Cluster;

// There is only one global hash table for the engine and all its threads.
// For chess in particular, we even allow racy updates between threads to and
// from the TT, as taking the time to synchronize access would cost thinking
// time and thus Elo. As a hash table, collisions are possible and may cause
// chess playing issues (bizarre blunders, faulty mate reports, etc). Fixing
// these also loses Elo; however such risk decreases with larger TT size.
//
// We clearly separate TTData, a local copy of an entry, from TTWriter, which
// writes to the global table.

// 各エンジンはハッシュテーブルを1つ保持し、そのエンジンの全探索スレッドで共有する。
// やねうら王では置換表はEngineのメンバーであり、プロセス全体のグローバル変数ではない。
// 特にチェスでは、TTへのアクセスを同期するための時間が思考時間を消費し、
// 結果としてEloを低下させるため、スレッド間でTTを読み書きする際の競合を許容している。
// ハッシュテーブルである以上、衝突は起こり得るし、それによってチェスの対局上の
// 問題（不可解な悪手、誤った詰みの報告など）が生じる可能性がある。
// これらの問題を修正することもEloを低下させる。ただし、TTのサイズが大きくなるほど、
// そのようなリスクは減少する。
//
// TTData（エントリのローカルコピー）と、グローバルテーブルへの書き込みを行う
// TTWriterは、明確に分離している。

// A copy of the data already in an entry (possibly collided). Probes and reads
// are racy and non-atomic, possibly resulting in inconsistent data.

// エントリにすでに格納されているデータのコピー（衝突している可能性もある）。
// TTの探索と読み取りは競合を許容しており、アトミックではないため、
// データの整合性が失われる可能性がある。

// ■ 補足
// 
// moveはMove(32bit)ではあるが、TTEntryにはMove16(16bit)でしか格納されていない。
// そのため、TT.probe()で取り出すときにこの16bitを32bitに拡張して返す。

struct TTData {
    Move  move;
    Value value, eval;
    Depth depth;
    Bound bound;
    bool  is_pv;

    TTData() = delete;

    // clang-format off
    TTData(Move m, Value v, Value ev, Depth d, Bound b, bool pv) :
        move(m),
        value(v),
        eval(ev),
        depth(d),
        bound(b),
        is_pv(pv) {}
    // clang-format on
};


// This is used to make racy, non-atomic writes to the global TT. Writes are
// not "guaranteed": for chess reasons, we may decide the new data is less
// important than the old.

// これは、グローバルTTに対して競合を許容した非アトミックな書き込みを行うために使用する。
// 書き込みは「保証」されない。チェスの観点から、新しいデータは古いデータよりも
// 重要度が低いと判断し、書き込みを行わない場合がある。
struct TTWriter {
public:
    void write(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, u8 generation8);
    void penalize(int penalty);  // decrement stored depth by the penalty
    // 格納された深さからpenaltyを減算する。

private:
	friend class TranspositionTable;

	// コンストラクタで渡されたcl,idx。最後、cluster[index]にwriteで書き戻す。
	Cluster* cluster;
    int      index;

    TTWriter(Cluster* cl, int idx);
};

// ============================================================
//               置換表本体
// ============================================================

// ClusterSize個のTTEntryからなるClusterを、clusterCount個確保する。
// キーから求めた同じClusterに複数の局面が割り当てられることがある。
// その中から保存したキーの一部が一致するエントリ、または置換対象を探す。
class TranspositionTable {

   public:
    ~TranspositionTable() { aligned_large_pages_free(table); }

	// Set TT size in MiB
	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。[MiB]単位。

#if STOCKFISH
    void resize(usize mbSize, ThreadPool& threads);
#else
    void resize(usize mbSize, ThreadPool& threads, bool force = false);  // Set TT size in MiB
#endif

	// Re-initialize memory, multithreaded
	// メモリを再初期化、マルチスレッド対応

	// 置換表のエントリーの全クリア
	// 並列化してクリアするので高速。

    void clear(ThreadPool& threads);

    // Must be called at the beginning of each root search to track entry aging
    // エントリのエイジングを追跡するために、各ルート検索の開始時にこれを呼び出す必要があります。
    // ⇨ 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)

    void new_search();

    // The current age, used when writing new data to the TT
    // 新しいデータをTTに書き込む際に使用される現在のエイジ

    u8 generation() const;

    // Approximate what fraction of entries (permille) have been written to
    // during this root search.

    // maxAge以下の世代差を持つエントリの割合を千分率で概算する。
    // maxAgeの既定値0では、現在の探索世代に書き込まれたエントリを数える。

    int hashfull(int maxAge = 0) const;

    // `probe(key)` is the primary method: given a board position, we lookup
    //  its entry in the table, and return a tuple of:
    //   1) whether the entry already had data on this position
    //   2) a copy of the prior data, if any (may be self-inconsistent due to races)
    //   3) a writer object to the entry
	//
	// probeは主なメソッドであり、局面を与えられるとテーブル内のエントリを検索し、以下のタプルを返します。
	//   1) そのエントリがすでにこの局面のデータを持っていたかどうか
	//   2) 以前のデータのコピー(あれば)。読み取り競合により自己矛盾している可能性があります
	//   3) このエントリへのライターオブジェクト

	// 置換表のなかから与えられたkeyに対応するentryを探す。
	// ヒットの有無、データのコピー、書き戻し先を指すTTWriterを返す。
	// ヒットしなかった場合も、置換候補へ書き込むためのTTWriterを返す。
	// Positionのキー幅によらず、TTでは下位64bitをキーとして使用する。
	//
	// ⇨ このprobe()でTTの内部状態が変更されないことは保証されている。(されるようになった)

#if STOCKFISH
    std::tuple<bool, TTData, TTWriter> probe(const Key key) const;
#else
	// やねうら王では、probeするときに pos も渡す必要がある。
	// 置換表に格納された16bitの指し手を、局面の駒情報を使って32bitのMoveへ拡張するため。
	// TTクラスタの先後分離には、posではなくkeyのbit0を使う。
	std::tuple<bool, TTData, TTWriter> probe(const Key key, const Position& pos) const;
#endif

	// The hash function; its only external use is memory prefetching
	// ハッシュ関数です。外部での唯一の使用目的はメモリのプリフェッチです。

	/*
		📓 get_cluster()とは？

		keyを元にClusterのindexを求めて、そのCluster* を返す。

		Stockfishとは違い、keyのbit0に格納した手番をCluster indexのbit0に埋めることで、
		手番が異なれば、異なる
		TT Clusterになることを保証するため。

		これは、将棋では駒の移動が上下対称ではないので、先手の指し手が(TT raceで)
		後手番の局面でTT.probeで返ってくると、pseudo-legalの判定で余計なチェックが
		必要になって嫌だからである。
	*/ 

    Cluster* get_cluster(const Key key) const;

#if !STOCKFISH
	static void UnitTest(Test::UnitTester& unittest, IEngine& engine);
#endif

private:
	friend struct TTEntry;

	// この置換表が保持しているクラスター数。
	// Stockfishはresize()ごとに毎回新しく置換表を確保するが、やねうら王では
	// そこは端折りたいので、毎回は確保しない。そのため前回サイズがここに格納されていないと
	// 再確保すべきかどうかの判定ができない。ゆえに0で初期化しておく。
#if STOCKFISH
    usize    clusterCount;
#else
    usize    clusterCount = 0;
#endif

	// 確保されているClusterの先頭(alignされている)
	// Stockfishと同じくnullptrで初期化し、未確保であることを示す。
    Cluster* table = nullptr;

	// 世代カウンター。new_search()で進め、TTWriter::write()で保存する。
	u8 generation8 = 0;
};

} // namespace YaneuraOu

#endif // #ifndef TT_H_INCLUDED
