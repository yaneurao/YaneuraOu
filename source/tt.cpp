#include "tt.h"

TranspositionTable TT; // 置換表をglobalに確保。

#if defined(USE_GLOBAL_OPTIONS)
size_t TranspositionTable::max_thread;
#endif


// 置換表のサイズを確保しなおす。
void TranspositionTable::resize(size_t mbSize) {

	size_t newClusterCount = size_t(1) << MSB64((mbSize * 1024 * 1024) / sizeof(Cluster));

	// 同じサイズなら確保しなおす必要はない。
	if (newClusterCount == clusterCount)
		return;

	clusterCount = newClusterCount;

	free(mem);

	// tableはCacheLineSizeでalignされたメモリに配置したいので、CacheLineSize-1だけ余分に確保する。
	mem = calloc(clusterCount * sizeof(Cluster) + CacheLineSize - 1, 1);

	if (!mem)
	{
		std::cout << "info string Error : Failed to allocate " << mbSize
			<< "MB for transposition table. ClusterCount = " << newClusterCount << std::endl;
		exit(EXIT_FAILURE);
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
		size_t index = (((size_t)key % block) & ~1 ) | ((size_t)key & 1);
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
		// 1. keyが合致しているentryを見つけた。(found==trueにしてそのTT_ENTRYのアドレスを返す)
		// 2. 空のエントリーを見つけた(そこまではkeyが合致していないので、found==falseにして新規TT_ENTRYのアドレスとして返す)

		// Stockfishのコードだと、1.が成立したタイミングでもgenerationのrefreshをしているが、
		// save()のときにgenerationを書き出すため、このケースにおいてrefreshは必要ない。

		// 1.
		if (!tte[i].key16)
			return found = false, &tte[i];

		// 2.
		if (tte[i].key16 == key16)
		{
#if defined(USE_GLOBAL_OPTIONS)
			// 置換表とTTEntryの世代が異なるなら、値は信用できないと仮定するフラグ。
			if (GlobalOptions.use_strict_generational_tt)
				if (tte[i].generation() != gen8)
					tte[i].set_value(VALUE_NONE);
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
