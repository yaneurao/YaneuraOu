#include "tt.h"
#include "misc.h"

TranspositionTable TT; // 置換表をglobalに確保。

#if defined(USE_GLOBAL_OPTIONS)
size_t TranspositionTable::max_thread;
#endif


// 置換表のサイズを確保しなおす。
void TranspositionTable::resize(size_t mbSize) {

	// mbSizeの単位は[MB]なので、ここでは1MBの倍数単位のメモリが確保されるが、
	// 仕様上は、1MBの倍数である必要はない。
	size_t newClusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

	// clusterCountは偶数でなければならない。
	// この理由については、TTEntry::first_entry()のコメントを見よ。
	// しかし、1024 * 1024 / sizeof(Cluster)の部分、sizeof(Cluster)==64なので、
	// これを掛け算するから2の倍数である。
	ASSERT_LV3((newClusterCount & 1) == 0);

	// 同じサイズなら確保しなおす必要はない。
	if (newClusterCount == clusterCount)
		return;

	clusterCount = newClusterCount;

	free(mem);

	// tableはCacheLineSizeでalignされたメモリに配置したいので、CacheLineSize-1だけ余分に確保する。
	// callocではなくmallocにしないと初回の探索でTTにアクセスするとき、特に巨大なTTだと
	// 極めて遅くなるので、mallocで確保して、ゼロクリアすることでこれを回避する。
	// cf. Explicitly zero TT upon resize. : https://github.com/official-stockfish/Stockfish/commit/2ba47416cbdd5db2c7c79257072cd8675b61721f
	mem = malloc(clusterCount * sizeof(Cluster) + CacheLineSize - 1);

	if (!mem)
	{
		std::cout << "info string Error : Failed to allocate " << mbSize
			<< "MB for transposition table. ClusterCount = " << newClusterCount << std::endl;
		my_exit();
	}

	table = (Cluster*)((uintptr_t(mem) + CacheLineSize - 1) & ~(CacheLineSize - 1));
}


TTEntry* TranspositionTable::probe(const Key key, bool& found
#if defined(USE_GLOBAL_OPTIONS)
	, size_t thread_id
#endif
	
	) const
{
	ASSERT_LV3(clusterCount != 0);

#if defined(USE_GLOBAL_OPTIONS)
	if (!GlobalOptions.use_hash_probe)
	{
		// 置換表にhitさせないモードであるなら、見つからなかったことにして
		// つねに確保しているメモリの先頭要素を返せば良い。(ここに書き込まれたところで問題ない)
		return found = false, first_entry(0);
	}
#endif

	// 最初のTT_ENTRYのアドレス(このアドレスからTT_ENTRYがClusterSize分だけ連なっている)
	// keyの下位bitをいくつか使って、このアドレスを求めるので、自ずと下位bitはいくらかは一致していることになる。
	TTEntry* tte;
	u8 gen8;

#if !defined(USE_GLOBAL_OPTIONS)

	tte = first_entry(key);
	gen8 = generation();

#else

	if (GlobalOptions.use_per_thread_tt)
	{
		// スレッドごとに置換表の異なるエリアを渡す必要がある。
		// 置換表にはclusterCount個のクラスターがあるのでこれをスレッドの個数で均等に割って、
		// そのthread_id番目のblockを使わせてあげる、的な考え。
		//
		// ただしkeyのbit0は手番bitであり、これはそのままindexのbit0に反映されている必要がある。
		//
		// また、blockは2の倍数になるように下丸めしておく。
		// ・上丸めするとblock*max_thread > clusterCountになりかねない)
		// ・2の倍数にしておかないと、(key % block)にkeyのbit0を反映させたときにこの値がblockと同じ値になる。
		//   (各スレッドが使えるのは、( 0～(block-1) ) + (thread_id * block)のTTEntryなので、これはまずい。

		size_t block = (clusterCount / max_thread) & ~1;

		// Stockfish公式で置換表サイズ128GB超えに対応するまでデフォルトで無効にしておく。
#if defined (IS_64BIT) && defined(USE_SSE2) && defined(USE_HUGE_HASH)
		uint64_t highProduct;
		_umul128(key + (key << 32), block, &highProduct);
		size_t index = (highProduct & ~1) | (key & 1);
#else
		size_t index = (((uint32_t(key >> 1) * uint64_t(block)) >> 32) & ~1) | (key & 1);
		// uint32_t(key)*block / 2^32 は、0～(block-1)の値。
		// keyのbit0は、先後フラグなのでindexのbit0に反映されなければならない。
#endif

		tte = &table[index + thread_id * block].entry[0];

	}	else {
		tte = first_entry(key);
	}
	gen8 = generation(thread_id);

#endif

	// 上位16bitが合致するTT_ENTRYを探す
	const uint16_t key16 = key >> 48;

	// クラスターのなかから、keyが合致するTT_ENTRYを探す
	for (int i = 0; i < ClusterSize; ++i)
	{
		// returnする条件
		// 1. 空のエントリーを見つけた(そこまではkeyが合致していないので、found==falseにして新規TT_ENTRYのアドレスとして返す)
		// 2. keyが合致しているentryを見つけた。(found==trueにしてそのTT_ENTRYのアドレスを返す)

		// Stockfishのコードだと、1.が成立したタイミングでもgenerationのrefreshをしているが、
		// save()のときにgenerationを書き出すため、このケースにおいてrefreshは必要ない。

		// 1.
		if (!tte[i].key16)
			return found = false, &tte[i];

		// 2.
		if (tte[i].key16 == key16)
		{
#if defined(USE_GLOBAL_OPTIONS)
			// 置換表とTTEntryの世代が異なるなら、信用できないと仮定するフラグ。
			if (GlobalOptions.use_strict_generational_tt)
				if (tte[i].generation() != gen8)
					return found = false, &tte[i];
#endif

			tte[i].set_generation(gen8); // Refresh
			return found = true, &tte[i];
		}
	}

	// 空きエントリーも、探していたkeyが格納されているentryが見当たらなかった。
	// クラスター内のどれか一つを潰す必要がある。

	TTEntry* replace = tte;
	for (int i = 1; i < ClusterSize; ++i)

		// ・深い探索の結果であるものほど価値があるので残しておきたい。depth8 × 重み1.0
		// ・generationがいまの探索generationに近いものほど価値があるので残しておきたい。geration(4ずつ増える)×重み 2.0
		// 以上に基いてスコアリングする。
		// 以上の合計が一番小さいTTEntryを使う。

		if (replace->depth8 - ((259 + gen8 - replace->genBound8) & 0xFC) * 2
		  >   tte[i].depth8 - ((259 + gen8 -   tte[i].genBound8) & 0xFC) * 2)
			replace = &tte[i];

	// generationは256になるとオーバーフローして0になるのでそれをうまく処理できなければならない。
	// a,bが8bitであるとき ( 256 + a - b ) & 0xff　のようにすれば、オーバーフローを考慮した引き算が出来る。
	// このテクニックを用いる。
	// いま、
	//   a := generationは下位2bitは用いていないので0。
	//   b := genBound8は下位2bitにはBoundが入っているのでこれはゴミと考える。
	// ( 256 + a - b + c) & 0xfc として c = 3としても結果に影響は及ぼさない、かつ、このゴミを無視した計算が出来る。

	return found = false, replace;
}

int TranspositionTable::hashfull() const
{
	// すべてのエントリーにアクセスすると時間が非常にかかるため、先頭から1000エントリーだけ
	// サンプリングして使用されているエントリー数を返す。
	int cnt = 0;
	for (int i = 0; i < 1000 / ClusterSize; ++i)
	{
		const auto tte = &table[i].entry[0];
		for (int j = 0; j < ClusterSize; ++j)
			if ((tte[j].generation() == generation8))
				++cnt;
	}
	return cnt;
}
