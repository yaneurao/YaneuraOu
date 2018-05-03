#ifndef _TT_H_
#define _TT_H_

#include "shogi.h"

// --------------------
//       置換表
// --------------------

// 置換表エントリー
// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
// CPUのcache lineに一発で載るというミラクル。
struct TTEntry {

	Move move() const { return (Move)move16; }
	Value value() const { return (Value)value16; }
	void set_value(Value v) { value16 = v; }

#if !defined (NO_EVAL_IN_TT)
	// この局面でevaluate()を呼び出したときの値
	Value eval() const { return (Value)eval16; }
#endif

	Depth depth() const { return (Depth)(depth8 * int(ONE_PLY)); }
	Bound bound() const { return (Bound)(genBound8 & 0x3); }

	uint8_t generation() const { return genBound8 & 0xfc; }
	void set_generation(uint8_t g) { genBound8 = bound() | g; }

	// 置換表のエントリーに対して与えられたデータを保存する。上書き動作
	//   v    : 探索のスコア
	//   eval : 評価関数 or 静止探索の値
	//   m    : ベストな指し手
	//   gen  : TT.generation()
	void save(Key k, Value v, Bound b, Depth d, Move m,
#if !defined (NO_EVAL_IN_TT)
		Value eval,
#endif
		uint8_t gen)
	{
		// ASSERT_LV3((-VALUE_INFINITE < v && v < VALUE_INFINITE) || v == VALUE_NONE);

		// 置換表にVALUE_INFINITE以上の値を書き込んでしまうのは本来はおかしいが、
		// 実際には置換表が衝突したときにqsearch()から書き込んでしまう。
		//
		// 例えば、3手詰めの局面で、置換表衝突により1手詰めのスコアが返ってきた場合、VALUE_INFINITEより
		// 大きな値を書き込む。
		//
		// 逆に置換表をprobe()したときにそのようなスコアが返ってくることがある。
		// しかしこのようなスコアは、mate distance pruningで補正されるので問題ない。
		// (ように、探索部を書くべきである。)
		//
		// Stockfishで、VALUE_INFINITEを32001(int16_tの最大値よりMAX_PLY以上小さな値)にしてあるのは
		// そういった理由から。


		// このif式だが、
		// A = m!=MOVE_NONE
		// B = (k >> 48) != key16)
		// として、ifが成立するのは、
		// a)  A && !B
		// b)  A &&  B
		// c) !A &&  B
		// の3パターン。b),c)は、B == trueなので、その下にある次のif式が成立して、この局面のhash keyがkey16に格納される。
		// a)は、B == false すなわち、(k >> 48) == key16であり、この局面用のentryであるから、その次のif式が成立しないとしても
		// 整合性は保てる。
		// a)のケースにおいても、指し手の情報は格納しておいたほうがいい。
		// これは、このnodeで、TT::probeでhitして、その指し手は試したが、それよりいい手が見つかって、枝刈り等が発生しているような
		// ケースが考えられる。ゆえに、今回の指し手のほうが、いまの置換表の指し手より価値があると考えられる。

		if (m != MOVE_NONE || (k >> 48) != key16)
			move16 = (uint16_t)m;

		// このエントリーの現在の内容のほうが価値があるなら上書きしない。
		// 1. hash keyが違うということはTT::probeでここを使うと決めたわけだから、このEntryは無条件に潰して良い
		// 2. hash keyが同じだとしても今回の情報のほうが残り探索depthが深い(新しい情報にも価値があるので
		// 　少しの深さのマイナスなら許容)
		// 3. BOUND_EXACT(これはPVnodeで探索した結果で、とても価値のある情報なので無条件で書き込む)
		// 1. or 2. or 3.
		if (  (k >> 48) != key16
			|| (d / ONE_PLY > depth8 - 4)
			/*|| g != generation() // probe()において非0のkeyとマッチした場合、その瞬間に世代はrefreshされている。　*/
			|| b == BOUND_EXACT
			)
		{
			key16 = (uint16_t)(k >> 48);
			value16 = (int16_t)v;
#if !defined (NO_EVAL_IN_TT)
			eval16 = (int16_t)eval;
#endif
			genBound8 = (uint8_t)(gen | b);
			depth8 = (int8_t)(d / ONE_PLY);
		}
	}

private:
	friend struct TranspositionTable;

	// hash keyの上位16bit。下位48bitはこのエントリー(のアドレス)に一致しているということは
	// そこそこ合致しているという前提のコード
	uint16_t key16;

	// 指し手
	uint16_t move16;

	// このnodeでの探索の結果スコア
	int16_t value16;

#if !defined (NO_EVAL_IN_TT)
	// 評価関数の評価値
	int16_t eval16;
#endif

	// entryのgeneration上位6bit + Bound下位2bitのpackしたもの。
	// generationはエントリーの世代を表す。TranspositionTableで新しい探索ごとに+4されていく。
	uint8_t genBound8;

	// そのときの残り深さ(これが大きいものほど価値がある)
	// 1バイトに収めるために、DepthをONE_PLYで割ったものを格納する。
	// qsearch()でも置換表に登録するので少し負になっている数もありうる。
	int8_t depth8;
};

// --- 置換表本体
// TT_ENTRYをClusterSize個並べて、クラスターをつくる。
// このクラスターのTT_ENTRYは同じhash keyに対する保存場所である。(保存場所が被ったときに後続のTT_ENTRYを使う)
// このクラスターが、clusterCount個だけ確保されている。
struct TranspositionTable {

	// 置換表のなかから与えられたkeyに対応するentryを探す。
	// 見つかったならfound == trueにしてそのTT_ENTRY*を返す。
	// 見つからなかったらfound == falseで、このとき置換表に書き戻すときに使うと良いTT_ENTRY*を返す。
	// GlobalOptions.use_per_thread_tt == trueのときはスレッドごとに置換表の異なるエリアに属するTTEntryを
	// 渡す必要があるので、引数としてthread_idを渡す。
	TTEntry* probe(const Key key, bool& found
#if defined(USE_GLOBAL_OPTIONS)
		, size_t thread_id = -1
#endif
		) const;

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

	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
	void resize(size_t mbSize);

	// 置換表のエントリーの全クリア
	void clear() { memset(table, 0, clusterCount * sizeof(Cluster)); }

	// 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)
	// USE_GLOBAL_OPTIONSが有効のときは、このタイミングで、Options["Threads"]の値を
	// キャプチャして、探索スレッドごとの置換表と世代カウンターを用意する。
	void new_search() {
		generation8 += 4;

#if defined(USE_GLOBAL_OPTIONS)
		size_t m = Options["Threads"];
		if (m != max_thread)
		{
			max_thread = m;
			// スレッドごとの世代カウンター用の配列もこのタイミングで確保。
			a_generation8.resize(m);
		}
#endif
	} // 下位2bitはTTEntryでBoundに使っているので4ずつ加算。

	// 世代を返す。これはTTEntry.save()のときに使う。
	uint8_t generation() const { return generation8; }

#if defined(USE_GLOBAL_OPTIONS)

	// --- スレッドIDごとにgenerationを持っているとき用の処理。

	uint8_t generation(size_t thread_id) const {
		if (GlobalOptions.use_per_thread_tt)
			return a_generation8[thread_id];
		else
			return generation8;
	}

	void new_search(size_t thread_id) {
		if (GlobalOptions.use_per_thread_tt)
			a_generation8[thread_id] += 4;
		else
			generation8 += 4;
	}

#endif

	// 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)
	int hashfull() const;

	TranspositionTable() { mem = nullptr; clusterCount = 0; }
	~TranspositionTable() { free(mem); }

private:

	// TTEntryはこのサイズでalignされたメモリに配置する。(される)
	static const int CacheLineSize = 64;

#if !defined (NO_EVAL_IN_TT)
	// 1クラスターにおけるTTEntryの数
	// TTEntry 10bytes×3つ + 2(padding) = 32bytes
	static const int ClusterSize = 3;
#else
	// TTEntry 8bytes×4つ = 32bytes
	static const int ClusterSize = 4;
#endif

#if defined(USE_GLOBAL_OPTIONS)
	// スレッド数
	// スレッドごとに置換表を分けたいときのために現在のスレッド数を保持しておき、
	// 異なるエリアのなかのTTEntryを返すようにする。
	static size_t max_thread;

	// スレッドごとに世代を持っている必要がある。
	std::vector<u8> a_generation8;
#endif

	struct Cluster {
		TTEntry entry[ClusterSize];
#if !defined (NO_EVAL_IN_TT)
		u8 padding[2]; // 全体を32byteぴったりにするためのpadding
#endif
	};

	static_assert(sizeof(Cluster) == CacheLineSize / 2, "Cluster size incorrect");

	// この置換表が保持しているクラスター数。2の累乗。
	size_t clusterCount;

	// 確保されているクラスターの先頭(alignされている)
	Cluster* table;

	// 確保されたメモリの先頭(alignされていない)
	void* mem;

	uint8_t generation8; // TT_ENTRYのset_gen()で書き込む
};

// 詰みのスコアは置換表上は、このnodeからあと何手で詰むかというスコアを格納する。
// しかし、search()の返し値は、rootからあと何手で詰むかというスコアを使っている。
// (こうしておかないと、do_move(),undo_move()するごとに詰みのスコアをインクリメントしたりデクリメントしたり
// しないといけなくなってとても面倒くさいからである。)
// なので置換表に格納する前に、この変換をしなければならない。
// 詰みにまつわるスコアでないなら関係がないので何の変換も行わない。
// ply : root node からの手数。(ply_from_root)
inline Value value_to_tt(Value v, int ply) {

	ASSERT_LV3(-VALUE_INFINITE < v && v < VALUE_INFINITE);

	return  v >= VALUE_MATE_IN_MAX_PLY ? v + ply
		  : v <= VALUE_MATED_IN_MAX_PLY ? v - ply : v;
}

// value_to_tt()の逆関数
// ply : root node からの手数。(ply_from_root)
inline Value value_from_tt(Value v, int ply) {

	return  v == VALUE_NONE ? VALUE_NONE
		: v >= VALUE_MATE_IN_MAX_PLY ? v - ply
		: v <= VALUE_MATED_IN_MAX_PLY ? v + ply : v;
}

// PV lineをコピーする。
// pv に move(1手) + childPv(複数手,末尾MOVE_NONE)をコピーする。
// 番兵として末尾はMOVE_NONEにすることになっている。
inline void update_pv(Move* pv, Move move, Move* childPv) {

	for (*pv++ = move; childPv && *childPv != MOVE_NONE; )
		*pv++ = *childPv++;
	*pv = MOVE_NONE;
}

extern TranspositionTable TT;

#endif // _TT_H_
