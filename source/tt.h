#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include "types.h"
#include "misc.h"

// cf.【決定版】コンピュータ将棋のHASHの概念について詳しく : http://yaneuraou.yaneu.com/2018/11/18/%E3%80%90%E6%B1%BA%E5%AE%9A%E7%89%88%E3%80%91%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E5%B0%86%E6%A3%8B%E3%81%AEhash%E3%81%AE%E6%A6%82%E5%BF%B5%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/

// --------------------
//       置換表
// --------------------

/// 置換表エントリー
/// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
/// CPUのcache lineに一発で載るというミラクル。
///
/// key        16 bit : hash keyの下位16bit(bit0は除くのでbit16..1)
/// depth       8 bit : 格納されているvalue値の探索深さ
/// move       16 bit : このnodeの最善手(指し手16bit ≒ Move16 , Moveの上位16bitは無視される)
/// generation  5 bit : 世代カウンター
/// pv node     1 bit : PV nodeで調べた値であるかのフラグ
/// bound type  2 bit : 格納されているvalue値の性質(fail low/highした時の値であるだとか)
/// value      16 bit : このnodeでのsearch()の返し値
/// eval value 16 bit : このnodeでのevaluate()の返し値
struct TTEntry {

	Move16 move() const { return Move16(move16); }
	Value value() const { return (Value)value16; }
	Value eval() const { return (Value)eval16; }
	Depth depth() const { return (Depth)depth8 + DEPTH_OFFSET; }
	bool is_pv() const { return (bool)(genBound8 & 0x4); }
	Bound bound() const { return (Bound)(genBound8 & 0x3); }

	// 置換表のエントリーに対して与えられたデータを保存する。上書き動作
	//   v    : 探索のスコア
	//   ev   : 評価関数 or 静止探索の値
	//   pv   : PV nodeであるか
	//   d    : その時の探索深さ
	//   m    : ベストな指し手
	void save(Key k, Value v, bool pv , Bound b, Depth d, Move m, Value ev);

private:
	friend struct TranspositionTable;

	// hash keyの下位bit16(bit0は除く)
	// Stockfishの最新版[2020/11/03]では、key16はhash_keyの下位16bitに変更になったが(取り出しやすいため)
	// やねうら王ではhash_keyのbit0を先後フラグとして用いるので、bit16..1を使う。
	// hash keyの上位bitは、TTClusterのindexの算出に用いるので、下位を格納するほうが理にかなっている。
	uint16_t key16;

	// 指し手(の下位16bit。Moveの上位16bitには移動させる駒種などが格納される)
	uint16_t move16;

	// このnodeでのsearch()の値
	int16_t value16;

	// このnodeでのevaluate()の値
	int16_t eval16;

	// entryのgeneration上位5bit + PVであるか1bit + Bound下位2bitのpackしたもの。
	// generationはエントリーの世代を表す。TranspositionTableで新しい探索ごとに+8されていく。
	uint8_t genBound8;

	// そのときの残り深さ(これが大きいものほど価値がある)
	// 1バイトに収めるために、DepthをONE_PLYで割ったものを格納する。
	// 符号付き8bitだと+127までしか表現できないので、符号なしにして、かつ、
	// DEPTH_NONEが-6なのでこの分だけ下駄履きさせてある。(+6して格納してある)
	uint8_t depth8;
};

// --- 置換表本体
// TT_ENTRYをClusterSize個並べて、クラスターをつくる。
// このクラスターのTT_ENTRYは同じhash keyに対する保存場所である。(保存場所が被ったときに後続のTT_ENTRYを使う)
// このクラスターが、clusterCount個だけ確保されている。
struct TranspositionTable {

	// 1クラスターにおけるTTEntryの数
	// TTEntry 10bytes×3つ + 2(padding) = 32bytes
	static constexpr int ClusterSize = 3;

	struct Cluster {
		TTEntry entry[ClusterSize];
		u8 padding[2]; // 全体を32byteぴったりにするためのpadding
	};

	static_assert(sizeof(Cluster) == 32, "Unexpected Cluster size");

public:
	//~TranspositionTable() { aligned_ttmem_free(mem); }
	// メモリの開放は、LargeMemoryクラスが勝手にやってくれるので、やねうら王では、
	// このclassのデストラクタでメモリを明示的に開放しなくて良い。

	// 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)
	// USE_GLOBAL_OPTIONSが有効のときは、このタイミングで、Options["Threads"]の値を
	// キャプチャして、探索スレッドごとの置換表と世代カウンターを用意する。
	void new_search() { generation8 += 8; } // 下位3bitはPV nodeかどうかのフラグとBoundに用いている。

	// 置換表のなかから与えられたkeyに対応するentryを探す。
	// 見つかったならfound == trueにしてそのTT_ENTRY*を返す。
	// 見つからなかったらfound == falseで、このとき置換表に書き戻すときに使うと良いTT_ENTRY*を返す。
	TTEntry* probe(const Key key, bool& found) const;

	// probe()の、置換表を一切書き換えないことが保証されている版。
	// ConsiderationMode時のPVの出力時は置換表をprobe()したいが、hitしないときに空きTTEntryを作る挙動が嫌なので、
	// こちらを用いる。(やねうら王独自拡張)
	TTEntry* read_probe(const Key key, bool& found) const;

	// 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)
	int hashfull() const;

	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
	void resize(size_t mbSize);

	// 置換表のエントリーの全クリア
	// 並列化してクリアするので高速。
	// 備考)
	// LEARN版のときは、
	// 単一スレッドでメモリをクリアする。(他のスレッドは仕事をしているので..)
	// 教師生成を行う時は、対局の最初にスレッドごとのTTに対して、
	// このclear()が呼び出されるものとする。
	// 例) th->tt.clear();
	void clear();

	// keyを元にClusterのindexを求めて、その最初のTTEntry*を返す。
	TTEntry* first_entry(const Key key) const {
		// Stockfishのコード
		// mul_hi64は、64bit * 64bitの掛け算をして下位64bitを取得する関数。
		//return &table[mul_hi64(key, clusterCount)].entry[0];

		// key(64bit) × clusterCount / 2^64 の値は 0 ～ clusterCount - 1 である。
		// 掛け算が必要にはなるが、こうすることで custerCountを2^Nで確保しないといけないという制約が外れる。
		// cf. Allow for general transposition table sizes. : https://github.com/official-stockfish/Stockfish/commit/2198cd0524574f0d9df8c0ec9aaf14ad8c94402b

		// ※　以下、やねうら王独自拡張

		// やねうら王では、keyのbit0(先後フラグ)がindexのbit0に反映される必要がある。
		// このときclusterCountが奇数だと、(index & ~(u64)1) | (key & 1) のようにしたときに、
		// (clusterCount - 1)が上限であるべきなのにclusterCountになりかねない。
		// そこでclusterCountは偶数であるという制約を課す。
		ASSERT_LV3((clusterCount & 1) == 0);

		// indexのbit0は、keyのbit0(先後フラグ)が反映されなければならない。
		// →　次のindexの計算ではbit0を潰して計算するためにkeyを2で割ってからmul_hi64()している。

		// (key/2) * clusterCount / 2^64 をするので、indexは 0 ～ (clusterCount/2)-1 の範囲となる。
		uint64_t index = mul_hi64((u64)key >> 1, clusterCount);

		// indexは0～(clusterCount/2)-1の範囲にあるのでこれを2倍すると、0～clusterCount-2の範囲。
		// clusterCountは偶数で、ここにkeyのbit0がbit-orされるので0～clusterCount-1が得られる。
		return &table[(index << 1) | ((u64)key & 1)].entry[0];
	}

#if defined(EVAL_LEARN)
	// スレッド数が変更になった時にThread.set()から呼び出される。
	// これに応じて、スレッドごとに保持しているTTを初期化する。
	void init_tt_per_thread();
#endif

private:
	friend struct TTEntry;

	// この置換表が保持しているクラスター数。
	// Stockfishはresize()ごとに毎回新しく置換表を確保するが、やねうら王では
	// そこは端折りたいので、毎回は確保しない。そのため前回サイズがここに格納されていないと
	// 再確保すべきかどうかの判定ができない。ゆえに0で初期化しておく。
	size_t clusterCount = 0;

	// 確保されているClusterの先頭(alignされている)
	// Stockfishではmemをコンストラクタで初期化していないが、初期化しておいたほうが
	// 不用意に使った場合に確実にアクセス保護違反で落ちるので都合が良い。
	Cluster* table = nullptr;

	// 確保されたメモリの先頭(alignされていない)
	//void* mem;
	// →　やねうら王では、LargeMemoryで確保するのでこれは不要

	// 世代カウンター。new_search()のごとに8ずつ加算する。TTEntry::save()で用いる。
	uint8_t generation8;

	// --- やねうら王独自拡張

	// 置換表テーブルのメモリ確保用のhelpper
	LargeMemory tt_memory;
};

// global object。探索部からこのinstanceを参照する。
extern TranspositionTable TT;

#endif // #ifndef TT_H_INCLUDED
