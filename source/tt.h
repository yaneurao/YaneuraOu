#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include "types.h"

// cf.【決定版】コンピュータ将棋のHASHの概念について詳しく : http://yaneuraou.yaneu.com/2018/11/18/%E3%80%90%E6%B1%BA%E5%AE%9A%E7%89%88%E3%80%91%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E5%B0%86%E6%A3%8B%E3%81%AEhash%E3%81%AE%E6%A6%82%E5%BF%B5%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/

// --------------------
//       置換表
// --------------------

/// 置換表エントリー
/// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
/// CPUのcache lineに一発で載るというミラクル。
///
/// key        16 bit : hash keyの上位16bit
/// move       16 bit : このnodeの最善手
/// value      16 bit : このnodeでのsearch()の返し値
/// eval value 16 bit : このnodeでのevaluate()の返し値
/// generation  5 bit : 世代カウンター
/// pv node     1 bit : PV nodeで調べた値であるかのフラグ
/// bound type  2 bit : 格納されているvalue値の性質(fail low/highした時の値であるだとか)
/// depth       8 bit : 格納されているvalue値の探索深さ
struct TTEntry {

	Move move() const { return (Move)move16; }
	Value value() const { return (Value)value16; }
	Value eval() const { return (Value)eval16; }
	Depth depth() const { return (Depth)(depth8 * int(ONE_PLY)) + DEPTH_NONE; }
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

	// hash keyの上位16bit。下位48bitはこのエントリー(のアドレス)に一致しているということは
	// そこそこ合致しているという前提のコード
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

	// TTEntryはこのサイズでalignされたメモリに配置する。(される)
	static const int CacheLineSize = 64;

	// 1クラスターにおけるTTEntryの数
	// TTEntry 10bytes×3つ + 2(padding) = 32bytes
	static constexpr int ClusterSize = 3;

	struct Cluster {
		TTEntry entry[ClusterSize];
		u8 padding[2]; // 全体を32byteぴったりにするためのpadding
	};

	static_assert(sizeof(Cluster) == CacheLineSize / 2, "Cluster size incorrect");

public:

	// Stockfishではmemをコンストラクタで初期化していないが、初期化しておいたほうが
	// 不用意に使った場合にアクセス保護違反で落ちるので都合が良い。
	TranspositionTable() { mem = nullptr; clusterCount = 0; }
	~TranspositionTable() { free(mem); }

	// 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)
	// USE_GLOBAL_OPTIONSが有効のときは、このタイミングで、Options["Threads"]の値を
	// キャプチャして、探索スレッドごとの置換表と世代カウンターを用意する。
	void new_search() { generation8 += 8; } // 下位3bitはPV nodeかどうかのフラグとBoundに用いている。

	// 置換表のなかから与えられたkeyに対応するentryを探す。
	// 見つかったならfound == trueにしてそのTT_ENTRY*を返す。
	// 見つからなかったらfound == falseで、このとき置換表に書き戻すときに使うと良いTT_ENTRY*を返す。
	TTEntry* probe(const Key key, bool& found) const;

	// 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)
	int hashfull() const;

	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
	void resize(size_t mbSize);

	// 置換表のエントリーの全クリア
	void clear();

	// keyの下位bitをClusterのindexにしてその最初のTTEntry*を返す。
	TTEntry* first_entry(const Key key) const {
		// 下位32bit × clusterCount / 2^32 なので、この値は 0 ～ clusterCount - 1 である。
		// 掛け算が必要にはなるが、こうすることで custerCountを2^Nで確保しないといけないという制約が外れる。
		// cf. Allow for general transposition table sizes. : https://github.com/official-stockfish/Stockfish/commit/2198cd0524574f0d9df8c0ec9aaf14ad8c94402b

		// return &table[(uint32_t(key) * uint64_t(clusterCount)) >> 32].entry[0];
		// →　(key & 1)が保存される必要性があるので(ここが先後フラグなので)、もうちょい工夫する。
		// このときclusterCountが奇数だと、最後の(clusterCount & ~1) | (key & 1) の値が、
		// (clusterCount - 1)が上限であるべきなのにclusterCountになりかねない。
		// そこでclusterCountは偶数であるという制約を課す。
		ASSERT_LV3((clusterCount & 1) == 0);

		// cf. 置換表の128GB制限を取っ払う冴えない方法 : http://yaneuraou.yaneu.com/2018/05/03/%E7%BD%AE%E6%8F%9B%E8%A1%A8%E3%81%AE128gb%E5%88%B6%E9%99%90%E3%82%92%E5%8F%96%E3%81%A3%E6%89%95%E3%81%86%E5%86%B4%E3%81%88%E3%81%AA%E3%81%84%E6%96%B9%E6%B3%95/
		// Stockfish公式で対応されるまでデフォルトでは無効にしておく。
#if defined (IS_64BIT) && defined(USE_SSE2) && defined(USE_HUGE_HASH)

		// cf. 128 GB TT size limitation : https://github.com/official-stockfish/Stockfish/issues/1349
		uint64_t highProduct;
		//		_umul128(key + (key << 32) , clusterCount, &highProduct);
		_umul128(key << 16, clusterCount, &highProduct);

		// この計算ではhighProductに第1パラメーターの上位bit周辺が色濃く反映されることに注意。
		// 上のStockfishのissuesに書かれている修正案は、あまりよろしくない。
		// TTEntry::key16はKeyの上位16bitの一致を見ているので、この16bitを計算に用いないか何らかの工夫が必要。
		// また、第1パラメーターをkeyにすると著しく勝率が落ちる。singular用のhash keyの性質が悪いのだと思う。
		// singluar用のhash keyは、bit16...31にexcludedMoveをxorしてあるのでこのbitを第1パラメーターの上位bit付近に
		// 反映させないとhash key衝突してしまう。そこで、お行儀はあまりよくないが、(key + (key << 32))を用いることにする。
		return &table[(highProduct & ~1) | (key & 1)].entry[0];
#else
		// また、(uint32_t(key) * uint64_t(clusterCount)だとkeyのbit0を使ってしまうので(31bitしか
		// indexを求めるのに使っていなくて精度がやや落ちるので)、(key >> 1)を使う。
		// ただ、Hashが小さいときにkeyの下位から取ると、singularのときのkeyとして
		// excludedMoveはbit16..31にしか反映させないので、あまりいい性質でもないような…。
		return &table[(((uint32_t(key >> 1) * uint64_t(clusterCount)) >> 32) & ~1) | (key & 1)].entry[0];
#endif
	}

private:
	friend struct TTEntry;

	// この置換表が保持しているクラスター数。
	size_t clusterCount;

	// 確保されているクラスターの先頭(alignされている)
	Cluster* table;

	// 確保されたメモリの先頭(alignされていない)
	void* mem;

	// 世代カウンター。new_search()のごとに8ずつ加算する。TTEntry::save()で用いる。
	uint8_t generation8;
};

extern TranspositionTable TT;

#endif // #ifndef TT_H_INCLUDED
