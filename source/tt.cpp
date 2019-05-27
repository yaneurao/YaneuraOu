#include "misc.h"
#include "thread.h"
#include "tt.h"

TranspositionTable TT; // 置換表をglobalに確保。

// 置換表のエントリーに対して与えられたデータを保存する。上書き動作
//   v    : 探索のスコア
//   eval : 評価関数 or 静止探索の値
//   m    : ベストな指し手
//   gen  : TT.generation()
// 引数のgenは、Stockfishにはないが、やねうら王では学習時にスレッドごとに別の局面を探索させたいので
// スレッドごとに異なるgenerationの値を指定したくてこのような作りになっている。
void TTEntry::save(Key k, Value v, bool pv , Bound b, Depth d, Move m , Value ev)
{
	// assert(d / ONE_PLY * ONE_PLY == d);
	// →　ONE_PLY == 1である現状、このassert要らんやろ。

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

	if (m || (k >> 48) != key16)
		move16 = (uint16_t)m;

	// このエントリーの現在の内容のほうが価値があるなら上書きしない。
	// 1. hash keyが違うということはTT::probeでここを使うと決めたわけだから、このEntryは無条件に潰して良い
	// 2. hash keyが同じだとしても今回の情報のほうが残り探索depthが深い(新しい情報にも価値があるので
	// 　少しの深さのマイナスなら許容)
	// 3. BOUND_EXACT(これはPVnodeで探索した結果で、とても価値のある情報なので無条件で書き込む)
	// 1. or 2. or 3.
	if (  (k >> 48) != key16
		|| d / ONE_PLY + 10 > depth8
		/*|| g != generation() // probe()において非0のkeyとマッチした場合、その瞬間に世代はrefreshされている。　*/
		|| b == BOUND_EXACT
		)
	{
		key16 = (uint16_t)(k >> 48);
		value16 = (int16_t)v;
		eval16    = (int16_t)ev;
		genBound8 = (uint8_t)(TT.generation8 | uint8_t(pv) << 2 | b);
		ASSERT_LV3((d - DEPTH_NONE) / ONE_PLY >= 0);
		depth8 = (uint8_t)((d - DEPTH_NONE) / ONE_PLY); // DEPTH_NONEだけ下駄履きさせてある。
	}
}

// 置換表のサイズを確保しなおす。
void TranspositionTable::resize(size_t mbSize) {

#if defined(MATE_ENGINE)
	// MateEngineではこの置換表は用いないので確保しない。
	return;
#endif

	// 探索が終わる前に次のresizeが来ると落ちるので探索の終了を待つ。
	Threads.main()->wait_for_search_finished();

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

	clusterCount = newClusterCount;

	free(mem);

	// tableはCacheLineSizeでalignされたメモリに配置したいので、CacheLineSize-1だけ余分に確保する。
	// callocではなくmallocにしないと初回の探索でTTにアクセスするとき、特に巨大なTTだと
	// 極めて遅くなるので、mallocで確保して自前でゼロクリアすることでこれを回避する。
	// cf. Explicitly zero TT upon resize. : https://github.com/official-stockfish/Stockfish/commit/2ba47416cbdd5db2c7c79257072cd8675b61721f
	mem = malloc(clusterCount * sizeof(Cluster) + CacheLineSize - 1);

	if (!mem)
	{
		std::cout << "info string Error : Failed to allocate " << mbSize
			<< "MB for transposition table. ClusterCount = " << newClusterCount << std::endl;
		my_exit();
	}

	table = (Cluster*)((uintptr_t(mem) + CacheLineSize - 1) & ~(CacheLineSize - 1));

	// clear();

	// →　Stockfish、ここでclear()呼び出しているが、Search::clear()からTT.clear()を呼び出すので
	// 二重に初期化していることになると思う。
}

void TranspositionTable::clear()
{
#if defined(MATE_ENGINE)
	// MateEngineではこの置換表は用いないのでクリアもしない。
	return;
#endif

	auto size = clusterCount * sizeof(Cluster);

	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishのここにあったコードは、独自の置換表を実装した時にも使いたいため、tt.cppに移動させた。
	memclear(table, size);

}

TTEntry* TranspositionTable::probe(const Key key, bool& found) const
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
	TTEntry* const tte = first_entry(key);

	// 上位16bitが合致するTT_ENTRYを探す
	// 下位bitは、tteのアドレスが一致してることから、だいたい合ってる。
	const uint16_t key16 = key >> 48;

	// クラスターのなかから、keyが合致するTT_ENTRYを探す
	for (int i = 0; i < ClusterSize; ++i)
	{
		// returnする条件
		// 1. 空のエントリーを見つけた(そこまではkeyが合致していないので、found==falseにして新規TT_ENTRYのアドレスとして返す)
		// 2. keyが合致しているentryを見つけた。(found==trueにしてそのTT_ENTRYのアドレスを返す)

		// Stockfishのコードだと、1.が成立したタイミングでもgenerationのrefreshをしているが、
		// save()のときにgenerationを書き出すため、このケースにおいてrefreshは必要ない。
		// (しかしソースコードをStockfishに合わせておくことに価値があると思うので、Stockfishに合わせておく)

		if (!tte[i].key16 || tte[i].key16 == key16)
		{
			tte[i].genBound8 = uint8_t(generation8 | (tte[i].genBound8 & 0x7)); // Refresh

			return found = (bool)tte[i].key16, &tte[i];
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

		if (replace->depth8 - ((263 + generation8 - replace->genBound8) & 0xF8)
		  >   tte[i].depth8 - ((263 + generation8 -   tte[i].genBound8) & 0xF8))
			replace = &tte[i];

	// generationは256になるとオーバーフローして0になるのでそれをうまく処理できなければならない。
	// a,bが8bitであるとき ( 256 + a - b ) & 0xff　のようにすれば、オーバーフローを考慮した引き算が出来る。
	// このテクニックを用いる。
	// いま、
	//   a := generationは下位3bitは用いていないので0。
	//   b := genBound8は下位3bitにはBoundが入っているのでこれはゴミと考える。
	// ( 256 + a - b + c) & 0xfc として c = 7としても結果に影響は及ぼさない、かつ、このゴミを無視した計算が出来る。

	return found = false, replace;
}

int TranspositionTable::hashfull() const
{
	// すべてのエントリーにアクセスすると時間が非常にかかるため、先頭から1000エントリーだけ
	// サンプリングして使用されているエントリー数を返す。
	int cnt = 0;
	for (int i = 0; i < 1000 / ClusterSize; ++i)
		for (int j = 0; j < ClusterSize; ++j)
			cnt += (table[i].entry[j].genBound8 & 0xF8) == generation8;

	// return cnt;でも良いが、そうすると最大で999しか返らず、置換表使用率が100%という表示にならない。
	return cnt * 1000 / (ClusterSize * (1000 / ClusterSize));
}
