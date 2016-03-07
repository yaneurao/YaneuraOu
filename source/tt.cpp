#include "tt.h"

TranspositionTable TT; // 置換表をglobalに確保。

#ifdef NEW_TT

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
    std::cerr << "Failed to allocate " << mbSize
      << "MB for transposition table." << std::endl;
    exit(EXIT_FAILURE);
  }

  table = (Cluster*)((uintptr_t(mem) + CacheLineSize - 1) & ~(CacheLineSize - 1));
}


TTEntry* TranspositionTable::probe(const Key key, bool& found) const {

  ASSERT_LV3(clusterCount != 0);

  // 最初のTT_ENTRYのアドレス(このアドレスからTT_ENTRYがClusterSize分だけ連なっている)
  // keyの下位bitをいくつか使って、このアドレスを求めるので、自ずと下位bitはいくらかは一致していることになる。
  TTEntry* const tte = &table[(size_t)(key) & (clusterCount - 1)].entry[0];

  // 上位16bitが合致するTT_ENTRYを探す
  const uint16_t key16 = key >> 48;

  // クラスターのなかから、keyが合致するTT_ENTRYを探す
  for (int i = 0; i < ClusterSize; ++i)
  {
    // returnする条件
    // 1. keyが合致しているentryを見つけた。(found==trueにしてそのTT_ENTRYのアドレスを返す)
    // 2. 空のエントリーを見つけた(そこまではkeyが合致していないので、found==falseにして新規TT_ENTRYのアドレスとして返す)
    if (!tte[i].key16)
      return found = false, &tte[i];

    if (tte[i].key16 == key16)
    {
      tte[i].set_generation(generation8); // Refresh
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

    if (replace->depth8 - ((259 + generation8 - replace->genBound8) & 0xFC) * 2 * ONE_PLY
      >   tte[i].depth8 - ((259 + generation8 - tte[i].genBound8  ) & 0xFC) * 2 * ONE_PLY)
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

#else
// 昔のStockfish風の実装

#include <cstring>
#include <iostream>

#include "bitboard.h"
#include "tt.h"

//: TranspositionTable TT; // Our global transposition table
// 置換表はglobalに配置。
// →　これはあかん。Gameクラスに移動させる。


/// TranspositionTable::set_size() sets the size of the transposition table,
/// measured in megabytes. Transposition table consists of a power of 2 number
/// of clusters and each cluster consists of ClusterSize number of TTEntry.

// TranspositionTable::set_size()は、置換表のサイズをMB(メガバイト)単位で設定する。
// 置換表は2の累乗のclusterで構成されており、それぞれのclusterはTTEnteryのClusterSizeで決まる。
// (1つのClusterは、TTEntry::ClusterSize×16バイト)
// mbSizeはClusterSize[GB]までのようだ。ClusterSize = 16だから16GBまで。

//:void TranspositionTable::set_size(uint64_t mbSize) {
void TranspositionTable::resize(uint64_t mbSize) {

  // size()で取得したいので保存しておく。
  hashSize = mbSize;

  // MB単位で指定してあるので左20回シフト

  // TTEntryの数が32bitでindexできる範囲を超えているとまずい。
  // TTEntryが128bit(16バイト)なので64GBが置換表用のサイズの上限。
  // ToDo : このassert本当に要るのか？64GB以上確保できてもいいように思うのだが…。

  ASSERT_LV3(MSB64((mbSize << 20) / sizeof(TTEntry)) < 32);

  // mbSize[MB] / clusterのサイズのうち、最上位の1になっているbit以外は端数とみなして0にした個数だけ
  // clusterを確保する。

  //:  uint32_t size = ClusterSize << msb((mbSize << 20) / sizeof(TTEntry[ClusterSize]));
  uint64_t size = size_t(ClusterSize) << MSB64((mbSize << 20) / sizeof(TTEntry[ClusterSize]));

  // 現在確保中の置換表用のメモリと等しいならば再確保は行わない
  if (hashMask == size - ClusterSize)
    return;

  // hashMaskは、sizeがClusterSize×010....0bみたいな数なので、ここからClusterSizeを引き算すると
  // 0011....1100bみたいな数になる。これをmaskとして使う。
  // 逆に、確保してあるTTEntryの数は、hashMask + ClusterSizeである。
  // そうか。bit0,bit1は無視されるのか、このhashは..

  hashMask = size - ClusterSize;
  // 前回確保していたメモリを開放
  free(mem);
  // 確保しなおす。callocを使ってあるのはゼロクリアするため。
  // あと、CACHE_LINE_SIZEは、64。余分に確保して、64でアラインされたアドレスを得るため。
  mem = calloc((size_t)(size * sizeof(TTEntry) + CACHE_LINE_SIZE - 1), 1);
  // x86環境ではsize_tは32bitなのでsize_tに明示的にcastしないとbit切り下げの警告がでる。

  if (!mem)
  {
    std::cerr << "Failed to allocate " << mbSize
      << "MB for transposition table." << std::endl;
    exit(EXIT_FAILURE);
  }

  // memから64バイトでアラインされたアドレスを得て、それをtableとして使う。
  table = (TTEntry*)((uintptr_t(mem) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
}


/// TranspositionTable::clear() overwrites the entire transposition table
/// with zeroes. It is called whenever the table is resized, or when the
/// user asks the program to clear the table (from the UCI interface).

// TranspositionTable::clear()は、ゼロで置換表全体をクリアする。
// テーブルがリサイズされたときや、ユーザーがUCI interface経由でテーブルのクリアを
// 要求したときに行われる。
#include "misc.h"

void TranspositionTable::clear() {

  if (table == nullptr)
  {
    sync_cout << "info string hash table is null." << sync_endl;
    return;
  }

  // これ時間がかかるのでisreadyに対して行うようにしたほうがいいのでは…。

  // hashMask + ClusterSizeになっている理由はTranspositionTable::set_sizeにある説明を読むこと。
  //:  std::memset(table, 0, (hashMask + ClusterSize) * sizeof(TTEntry));

  // 32bit超えるとこれでクリアできないので手でやる。

  for (size_t i = 0; i < (hashMask + ClusterSize) * sizeof(TTEntry) / 8; ++i)
    *((int64_t*)table + i) = 0;


  // 0だとnew_search()で1にされるが、使用率計測のときに1世代前もカウントするので1000と表示されてしまい気分が悪い。
  gen = 1;
}


/// TranspositionTable::probe() looks up the current position in the
/// transposition table. Returns a pointer to the TTEntry or NULL if
/// position is not found.

// 置換表を調べる。置換表のエントリーへのポインター(TTEntry*)が返る。
// エントリーが登録されていなければNULLが返る。

const TTEntry* TranspositionTable::probe(const Key key) const {

  // 最初のエントリーを取得
  const TTEntry* tte = first_entry(key);

  // 上位32bitが、置換表に登録されているhashの32bitと一致するのかを確認する。
  //:  uint32_t key32 = key >> 32;
  uint64_t key32 = key >> 16; // こっそり48bitに変更しとこか..

                              // ClusterSize分だけrehashされると解釈。
  for (unsigned i = 0; i < ClusterSize; ++i, ++tte)
    if (tte->key() == key32)
    {
      ((TTEntry*)tte)->set_generation(gen); // このタイミングでrefreshしておく。
      return tte;
    }
  // 見つかったならそれを返す。さもなくばnull
  // これ、もうちょっとbitを見ないと危険だが、まあ…。

  // 手番のhashをLSBに持たせるならこのrehashはまずいのか…。
  // ClusterSize == 4なら手番を16ぐらいにしとかないと…。

  //:  return NULL;
  return nullptr;
}


/// TranspositionTable::store() writes a new entry containing position key and
/// valuable information of current position. The lowest order bits of position
/// key are used to decide in which cluster the position will be placed.
/// When a new entry is written and there are no empty entries available in the
/// cluster, it replaces the least valuable of the entries. A TTEntry t1 is considered
/// to be more valuable than a TTEntry t2 if t1 is from the current search and t2
/// is from a previous search, or if the depth of t1 is bigger than the depth of t2.

// TranspositionTable::store()は現在の局面の、局面ハッシュキーと価値のある情報を新しいエントリーに書き込む。
// 局面ハッシュキーの下位のオーダーのbitは、どこにclusterを格納すべきかを決定するのに使われる。
// 新しいエントリーが書き込まれるときに、利用できる空のエントリーがcluster上になければ
// もっとも価値の低いエントリーを置き換える。
// ２つのTTEntry t1,t2があるとして、t1が現在の探索で、t2が以前の探索(世代が古い)だとか、
// t1の探索深さのほうがt2の探索深さより深いだとかする場合は、t1のほうがt2より価値があると考えられる。

// 置換表に値を格納する。
// key : この局面のハッシュキー。
// v : この局面の探索の結果得たスコア
// b : このスコアの性質。
//  BOUND_NONE →　探索していない(DEPTH_NONE)ときに、最善手か、静的評価スコアだけ置換表に突っ込みたいときに使う。
//  BOUND_LOWER →　fail-low
//  BOUND_UPPER → fail-high
//  BOUND_EXACT →　正確なスコア
// d : このスコア・指し手を得たときの残り探索深さ
// m : 最善手
// statV : 静的評価(この局面で評価関数を呼び出して得た値)
void TranspositionTable::store(const Key key, Value v, Bound b, Depth d, Move m, Value statV) {

  int c1, c2, c3;
  TTEntry *tte, *replace;

  // 局面ハッシュキーの上位32bitはcluster(≒置換表のエントリー)のなかで用いる。
  // また下位bitをclusterへのindexとして用いる。
  //  uint32_t key32 = key >> 32; // Use the high 32 bits as key inside the cluster
  uint64_t key32 = key >> 16; // 怖いので48bitに変更しよう..

                              // clusterの先頭のTTEntryを得る
  tte = replace = first_entry(key);

  // clusterのそれぞれの要素について..
  for (unsigned i = 0; i < ClusterSize; ++i, ++tte)
  {
    // 空のエントリーもしくはkeyが一致するエントリーであるなら..
    // ToDo : これtte->key() == 0なら、空のエントリーという扱いになっているが、
    // これ、偶然、keyの上位32bitが0になる場合、空のエントリーとみなされて上書きされてしまうではないのか…。
    // レアケースなのでいいのか？
    if (!tte->key() || tte->key() == key32) // Empty or overwrite old
    {
      // いま、指し手がないならば指し手を何にせよ潰さずに保存しておく。
      // ToDo : これ、偶然keyの上位32bitが0な局面が書かれていると、そこのmoveが使われてしまうことになるのでは…。
      // ここで非合法手になってしまうのどうなんだろう…。
      if (!m)
        m = tte->move(); // Preserve any existing ttMove

      // ともかく、空か、この局面用の古いエントリーが見つかったのでそこを使う。

      // ToDo : 反復深化では同じエントリーは基本的には以前の残り探索深さより深いはずなので上書きして問題ない..のか..？
      // 別経路でこの局面に突入したときに…どうなんだろう…。
      // 探索深さが置換表のエントリー上に書かれていた探索深さより浅い探索においては
      // 値を保存するためにこの関数を呼びださなければいいわけか。

      replace = tte;
      break;
    }

    // Implement replace strategy
    // 一番価値の低いところと置換する

    // replaceはClusterのなかでいま一番価値が低いとされるTTEntry
    // これ、都度replace->XXXと比較になっているのが少し気持ち悪いが、綺麗なコードなのでまあいいのか…。

    // いま置換予定のentryをreplaceポインターが指しているものとして、
    // replaceのgenerationが、現在のgenerationと一致するならば +2
    // tteのgenerationが現在のgeneration、もしくはtteがBOUND_EXACTなら-2
    // tteの探索深さのほうがreplaceの探索深さより浅いなら +1
    c1 = (replace->generation() == gen ? 2 : 0);
    c2 = (tte->generation() == gen || tte->bound() == BOUND_EXACT ? -2 : 0);
    c3 = (tte->depth() < replace->depth() ? 1 : 0);

    if (c1 + c2 + c3 > 0)
      replace = tte;

    // ケース1) generationがreplace,tteともに最新である場合。
    //   replaceのほうが探索深さが同じか浅いならreplace(最初に見つかったほう)を置き換える。
    //  (なるべく先頭に近いところを置き換えたほうがprobeが早く終るため)
    // ケース2) generationがreplaceは最新だが、tteがBOUND_EXACTである場合。
    //   replaceを置き換える。
    // ToDo : このロジックどうなん？
    // 探索深さが深いほうという重みが低いので、探索深さが深かろうとも、世代が古いと
    // 上書きされてしまう。探索深さが深いときの重みをもう少し大きくして、また、
    // 世代差2までは、加点するなどしてバランスをとったほうがいいのでは…。

  }

  // 置換対象が決まったのでそこに対して値を上書きしてしまう。
  // このgenerationは、このTranspositionTableが保持しているメンバー変数。
  replace->save(key32, v, b, d, m, gen, statV);
}


// 使用率を計算する関数。Stockfishにはないが、参考のために作った。
// 1000分率で返す。
int TranspositionTable::hashfull() const
{
  TTEntry* ptr = table;
  int sum = 0;
  // 置換表のクラスター数が1000より小さいことは想定しなくていいだろう。
  // 計測回数もったいない気がするので回数1/5にして5倍しておくか。
  const int ratio = 5;
  for (size_t i = 0; i < 1000 * ClusterSize / ratio; i += ClusterSize)
  {
    // いまのgenerationとひとつ前のgenerationを足したものを使用率とする。
    if (ptr[i].generation() == gen ||
      ptr[i].generation() == ((gen - 1) & 0xff))
    {
      sum++;
    }
  }
  return sum*ratio;
}

#endif
