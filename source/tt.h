#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

//#include <cstddef>
//#include <cstdint>
#include "types.h"
#include "misc.h"
#include "memory.h"

struct Key128;
struct Key256;

// cf.【決定版】コンピュータ将棋のHASHの概念について詳しく : http://yaneuraou.yaneu.com/2018/11/18/%E3%80%90%E6%B1%BA%E5%AE%9A%E7%89%88%E3%80%91%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E5%B0%86%E6%A3%8B%E3%81%AEhash%E3%81%AE%E6%A6%82%E5%BF%B5%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/

//class ThreadPool;
struct TTEntry;
struct Cluster;

// There is only one global hash table for the engine and all its threads. For chess in particular, we even allow racy
// updates between threads to and from the TT, as taking the time to synchronize access would cost thinking time and
// thus elo. As a hash table, collisions are possible and may cause chess playing issues (bizarre blunders, faulty mate
// reports, etc). Fixing these also loses elo; however such risk decreases quickly with larger TT size.
//
// probe is the primary method: given a board position, we lookup its entry in the table, and return a tuple of:
//   1) whether the entry already has this position
//   2) a copy of the prior data (if any) (may be inconsistent due to read races)
//   3) a writer object to this entry
// The copied data and the writer are separated to maintain clear boundaries between local vs global objects.

//エンジンとそのすべてのスレッドに対して、グローバルなハッシュテーブルは1つだけ存在します。
// 特にチェスにおいては、TT（トランスポジションテーブル）間でのスレッド間の競合的な更新も許可しており、
// アクセスを同期化するための時間を費やすと思考時間が減少し、それに伴いEloレーティングも下がるためです。
//
// ハッシュテーブルであるため、衝突が発生する可能性があり、
// それが原因でチェスプレイに問題が生じる場合があります（奇妙なミスや誤ったチェックメイト報告など）。
// これらを修正することもEloレーティングを失うことにつながりますが、大きなTTサイズではそのリスクは急速に減少します。
//
// probeは主なメソッドであり、ボードの局面を与えられると、テーブル内のエントリを検索し、以下のタプルを返します：
//
// そのエントリがすでにこの局面を持っているかどうか
// 以前のデータのコピー（あれば）（読み取り競合により不整合がある可能性があります）
// このエントリへのライターオブジェクト
// コピーされたデータとライターは、ローカルオブジェクトとグローバルオブジェクトの境界を明確にするために分離されています。


// A copy of the data already in the entry (possibly collided). `probe` may be racy, resulting in inconsistent data.

// すでにエントリに存在するデータのコピー（衝突している可能性があります）。
// `probe` は競合が発生することがあり、不整合なデータを返す可能性があります。

// ■ 補足
// 
// moveはMove(32bit)ではあるが、TTEntryにはMove16(16bit)でしか格納されていない。
// そのため、TT.probe()で取り出すときにこの16bitを32bitに拡張して返す。

struct TTData {
	Move   move;
	Value  value, eval;
	Depth  depth;
	Bound  bound;
	bool   is_pv;
};

// This is used to make racy writes to the global TT.

// これはグローバルTTへの競合的な書き込みを行うために使用されます。

struct TTWriter {
public:
	// TTのTTEntryに書き込む。
	void write(Key    k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
	void write(Key128 k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
	void write(Key256 k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);

private:
	friend class TranspositionTable;
	TTEntry* entry;
	TTWriter(TTEntry* tte);
};

// ============================================================
//               やねうら王独自拡張
// ============================================================

// やねうら王では、TTClusterSizeを変更できて、これが2の時は、TTEntryに格納するhash keyは64bit。(Stockfishのように)3の時は16bit。
#if TT_CLUSTER_SIZE == 3
typedef uint16_t TTE_KEY_TYPE;
#else // TT_CLUSTER_SIZEが2,4,6,8の時は64bit。5,7は選択する意味がないと思うので考えない。
typedef uint64_t TTE_KEY_TYPE;
#endif

// ============================================================
//               置換表本体
// ============================================================

// TT_ENTRYをClusterSize個並べて、クラスターをつくる。
// このクラスターのTT_ENTRYは同じhash keyに対する保存場所である。(保存場所が被ったときに後続のTT_ENTRYを使う)
// このクラスターが、clusterCount個だけ確保されている。
class TranspositionTable {

public:
	~TranspositionTable() { aligned_large_pages_free(table); }

	// Set TT size
	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。[MB]単位。

	void resize(size_t mbSize /*, ThreadPool& threads */);

	// Re-initialize memory, multithreaded
	// メモリを再初期化、マルチスレッド対応

	// 置換表のエントリーの全クリア
	// 並列化してクリアするので高速。
	// 備考)
	// LEARN版のときは、
	// 単一スレッドでメモリをクリアする。(他のスレッドは仕事をしているので..)
	// 教師生成を行う時は、対局の最初にスレッドごとのTTに対して、
	// このclear()が呼び出されるものとする。
	// 例) th->tt.clear();

	void clear();

	// Approximate what fraction of entries (permille) have been written to during this root search
	// このルート探索中に書き込まれたエントリの割合（パーミル単位）を概算します。
	// ⇨ 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)

	int hashfull(int maxAge = 0) const;

	// This must be called at the beginning of each root search to track entry aging
	// エントリのエイジングを追跡するために、各ルート検索の開始時にこれを呼び出す必要があります。
	// ⇨ 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)

	// USE_GLOBAL_OPTIONSが有効のときは、このタイミングで、Options["Threads"]の値を
	// キャプチャして、探索スレッドごとの置換表と世代カウンターを用意する。
	// ⇨ 下位3bitはPV nodeかどうかのフラグとBoundに用いている。
	void new_search();
	
	// The current age, used when writing new data to the TT
	// 新しいデータをTTに書き込む際に使用される現在のエイジ

	uint8_t generation() const;

	// The main method, whose retvals separate local vs global objects
	// メインメソッドで、その戻り値はローカルオブジェクトとグローバルオブジェクトを区別します

	// 置換表のなかから与えられたkeyに対応するentryを探す。
	// 見つかったならfound == trueにしてそのTT_ENTRY*を返す。
	// 見つからなかったらfound == falseで、このとき置換表に書き戻すときに使うと良いTT_ENTRY*を返す。
	// ※ KeyとしてKey(64 bit)以外に 128,256bitのhash keyにも対応。(やねうら王独自拡張)
	//
	// ⇨ このprobe()でTTの内部状態が変更されないことは保証されている。(されるようになった)
	//
	// ■ 備考
	//
	// Stockfishのprobe()を_probe()とrename。
	// そして、以下の3つのprobe()を用意して、この_probe()を下請けとして呼び出すように変更。

	std::tuple<bool, TTData, TTWriter> probe(const Key     key, const Position& pos) const;
	std::tuple<bool, TTData, TTWriter> probe(const Key128& key, const Position& pos) const;
	std::tuple<bool, TTData, TTWriter> probe(const Key256& key, const Position& pos) const;

	// This is the hash function; its only external use is memory prefetching.
	// これはハッシュ関数です。外部での唯一の使用目的はメモリのプリフェッチです。

	// ⇨ keyを元にClusterのindexを求めて、その最初のTTEntry*を返す。
	// 　ここで渡されるkeyのbit 0は局面の手番フラグ(Position::side_to_move())であると仮定している。
	// 
	// ■ 備考
	//
	// Stockfishのfirst_entry()を_first_entry()とrename。
	// そして、以下の3つのfirst_entry()を用意して、この_first_entry()を下請けとして呼び出すように変更。

	TTEntry* first_entry(const Key     key) const;
	TTEntry* first_entry(const Key128& key) const;
	TTEntry* first_entry(const Key256& key) const;

#if defined(EVAL_LEARN)
	// 学習用の実行ファイルでは、スレッド数が変更になったときに各ThreadごとのTTに
	// メモリを再割り当てする必要がある。
	void init_tt_per_thread();
#endif

	static void UnitTest(Test::UnitTester& unittest);

private:
	friend struct TTEntry;

	// keyを元にClusterのindexを求めて、その最初のTTEntry*を返す。内部実装用。
	// ※　ここで渡されるkeyのbit 0は局面の手番フラグ(Position::side_to_move())であると仮定している。

	TTEntry* _first_entry(const Key    key) const;
	std::tuple<bool, TTData, TTWriter> _probe(const Key key, const TTE_KEY_TYPE key_for_ttentry, const Position& pos) const;

	// この置換表が保持しているクラスター数。
	// Stockfishはresize()ごとに毎回新しく置換表を確保するが、やねうら王では
	// そこは端折りたいので、毎回は確保しない。そのため前回サイズがここに格納されていないと
	// 再確保すべきかどうかの判定ができない。ゆえに0で初期化しておく。
	size_t clusterCount = 0;

	// 確保されているClusterの先頭(alignされている)
	// Stockfishではmemをコンストラクタで初期化していないが、初期化しておいたほうが
	// 不用意に使った場合に確実にアクセス保護違反で落ちるので都合が良い。
	Cluster* table = nullptr;

	// Size must be not bigger than TTEntry::genBound8
	// サイズはTTEntry::genBound8を超えてはなりません。
	// ⇨ 世代カウンター。new_search()のごとに8ずつ加算する。TTEntry::save()で用いる。
	uint8_t generation8;
};

extern TranspositionTable TT;

#endif // #ifndef TT_H_INCLUDED
