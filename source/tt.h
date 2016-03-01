#ifndef _TT_H_
#define _TT_H_

#include "shogi.h"
#include "extra/key128.h"

// --------------------
//       置換表
// --------------------

#ifdef NEW_TT
// 置換表エントリー
// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
// CPUのcache lineに一発で載るというミラクル。
struct TTEntry {

  Move move() const { return (Move)move16; }
  Value value() const { return (Value)value16; }
  Value eval() const { return (Value)eval16; }
  Depth depth() const { return (Depth)depth8 * (int)ONE_PLY; }
  Bound bound() const { return (Bound)(genBound8 & 0x3); }

  uint8_t generation() const { return genBound8 & 0xfc; }
  void set_generation(uint8_t g) { genBound8 = bound() | g; }

  // 置換表のエントリーに対して与えられたデータを保存する。上書き動作
  //   v    : 探索のスコア
  //   eval : 評価関数 or 静止探索の値
  //   m    : ベストな指し手
  //   gen  : TT.generation()
  void save(Key k, Value v, Bound b,Depth d, Move m,Value eval,uint8_t gen)
  {
    ASSERT_LV3((-VALUE_INFINITE < v && v < VALUE_INFINITE) || v == VALUE_NONE);

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

    if (m!=MOVE_NONE || (k >> 48) != key16)
      move16 = (uint16_t)m;

    // このエントリーの現在の内容のほうが価値があるなら上書きしない。
    // 1. hash keyが違うということはTT::probeでここを使うと決めたわけだから、このEntryは無条件に潰して良い
    // 2. hash keyが同じだとしても今回の情報のほうが残り探索depthが深い(新しい情報にも価値があるので
    // 　少しの深さのマイナスなら許容)
    // 3. BOUND_EXACT(これはPVnodeで探索した結果で、とても価値のある情報なので無条件で書き込む)
    // 1. or 2. or 3.
    if ((k >> 48) != key16
      || (d > depth() - 2 * ONE_PLY) // ここ、2と4とどちらがいいか。あとで比較する。
      || b == BOUND_EXACT
      )
    {
      key16     = (int16_t)(k >> 48);
      value16   = (int16_t)v;
      eval16    = (int16_t)eval;
      genBound8 = (uint8_t)(gen | b);
      depth8    = (int8_t)d / (int)ONE_PLY;
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

  // 評価関数の評価値
  int16_t eval16;

  // entryのgeneration上位6bit + Bound下位2bitのpackしたもの。
  // generationはエントリーの世代を表す。TranspositionTableで新しい探索ごとに+4されていく。
  uint8_t genBound8;

  // そのときの残り深さ(これが大きいものほど価値がある)
  // 1バイトに収めるために、DepthをONE_PLYで割ったものを格納する。
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
  TTEntry* probe(const Key key, bool& found) const;

  // 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
  void resize(size_t mbSize);

  // 置換表のエントリーの全クリア
  void clear() { memset(table, 0, clusterCount*sizeof(Cluster)); }

  // 新しい探索ごとにこの関数を呼び出す。(generationを加算する。)
  void new_search() { generation8 += 4; } // 下位2bitはTTEntryでBoundに使っているので4ずつ加算。

  // 世代を返す。これはTTEntry.save()のときに使う。
  uint8_t generation() const { return generation8; }

  // 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)
  int hashfull() const;

  TranspositionTable() { mem = nullptr; clusterCount = 0; resize(16);/*デバッグ時ようにデフォルトで16MB確保*/ }
  ~TranspositionTable() { free(mem); }

private:
  // TTEntryはこのサイズでalignされたメモリに配置する。(される)
  static const int CacheLineSize = 64;

  // 1クラスターにおけるTTEntryの数
  static const int ClusterSize = 3;

  struct Cluster {
    TTEntry entry[ClusterSize];
    int8_t padding[2]; // 全体を32byteぴったりにするためのpadding
  };
  static_assert(sizeof(Cluster) == CacheLineSize /2 , "Cluster size incorrect");

  // この置換表が保持しているクラスター数。2の累乗。
  size_t clusterCount;

  // 確保されているクラスターの先頭(alignされている)
  Cluster* table;

  // 確保されたメモリの先頭(alignされていない)
  void* mem;

  uint8_t generation8; // TT_ENTRYのset_gen()で書き込む
};

#else
// 昔のStockfish風の置換表


/// The TTEntry is the 128 bit transposition table entry, defined as below:
// TTEntryは、128bitの置換表上の1エントリーで、以下のように定義されている。
///
// 局面のハッシュキーの上位16bit
/// key: 32 bit

// 指し手
/// move: 16 bit

// BOUND_LOWERとかUPPERとか
/// bound type: 8 bit

// 置換表エントリーの世代
/// generation: 8 bit

// 探索をした結果の評価値
/// value: 16 bit

// そのときの探索深さ
/// depth: 16 bit

// 静的評価値
/// static value: 16 bit

// ？　現在、未使用。以前のソースで使ってあった。
/// static margin: 16 bit
//　→　ここ使わないなら、valueを32bitに変更する。

struct TTEntry {

  // このクラスのインスタンスはcallocで確保されるので、コンストラクタが呼び出されることはない。
  // TTEntry*を得て、そこに対してsave()などで値を書き込んだりする。

  // k : 局面のハッシュキーの上位16bit
  // v : 探索したときの結果の評価値
  // b : BOUND
  // d : 残り探索depth
  // m : このnodeのベストの指し手
  // g : このentryの世代
  // ev : 静的評価値
  //:  void save(uint32_t k, Value v, Bound b, Depth d, Move m, int g, Value ev) {
  void save(uint64_t k, Value v, Bound b, Depth d, Move m, int g, Value ev) {

    key32 = (uint32_t)k;
    key16 = (uint16_t)(k >> 32);
    // kは48bitに変更。
    // 上位から見て16bit(0) - 16bit(key16) - 32bit(key32)
    // となっているはず。

    move16 = (uint16_t)m;
    bound8 = (uint8_t)b;
    generation8 = (uint8_t)g;
    value16 = (int16_t)v;
    depth16 = (int16_t)d;
    evalValue = (int16_t)ev;
  }

  // 世代の設定用
  void set_generation(uint8_t g) { generation8 = g; }

  // --- 以下、getter

  // 局面のhashkeyの上位32bit
  // 置換表サイズが小さいときはもう少しbitを詰めておかないと危険かも知れない…。
  //:  uint32_t key() const      { return key32; }

  // 上位48bit
  uint64_t key() const { return (uint64_t)key32 | ((uint64_t)key16 << 32); }


  // 探索残り深さ
  Depth depth() const { return (Depth)depth16; }

  // このnodeのベストの指し手
  Move move() const { return (Move)move16; }

  // 探索の結果の評価値
  Value value() const { return (Value)value16; }

  Value eval() const { return (Value)evalValue; }

  // BOUND_LOWERとかUPPERとか
  Bound bound() const { return (Bound)bound8; }

  // 世代(8bit)
  int generation() const { return (int)generation8; }

private:
  // 局面のハッシュキーの上位32bit
  uint32_t key32;
  uint16_t key16; // 怖いのでさらに16bit。

                  // 指し手16bit
  uint16_t move16;

  // BOUNDを表現する1バイト、置換表のentryの世代用カウンター
  uint8_t bound8, generation8;

  // 評価値、そのときの探索残り深さ
  // その局面での静的評価スコア
  int16_t value16, depth16, evalValue;
};


/// A TranspositionTable consists of a power of 2 number of clusters and each
/// cluster consists of ClusterSize number of TTEntry. Each non-empty entry
/// contains information of exactly one position. The size of a cluster should
/// not be bigger than a cache line size. In case it is less, it should be padded
/// to guarantee always aligned accesses.

// 置換表
// TranspositionTableは、2の累乗のclusterからなる。それぞれのclusterは、
// TTEntry(置換表のエントリーひとつを表す構造体)のClusterSizeの数によって決まる。
// それぞれのclusterはcache lineサイズを上回るべきではない。
// 逆に、それを下回るケースにおいては、alignされたアクセスを常に保証するために
// パディングされるべきである。

const int CACHE_LINE_SIZE = 64;

class TranspositionTable {

  // 一つのclusterは、16バイト(= sizeof(TTEntry))×ClusterSize = 64バイト。
  // Clusterは、rehashのための連続したTTEntryの塊のこと。
  static const unsigned ClusterSize = 4; // A cluster is 64 Bytes

public:
  TranspositionTable() { mem = nullptr; resize(1024); }
  ~TranspositionTable() { free(mem); }

  // 置換表を新しい探索のために掃除する。(generationを進める)
  void new_search() { ++gen; }

  // 置換表を調べる。置換表のエントリーへのポインター(TTEntry*)が返る。
  // エントリーが登録されていなければNULLが返る。
  const TTEntry* probe(const Key key) const;

  // TranspositionTable::first_entry()は、与えられた局面(のハッシュキー)に該当する
  // 置換表上のclusterの最初のエントリーへのポインターを返す。
  // 引数として渡されたkey(ハッシュキー)の下位ビットがclusterへのindexとして使われる。
  TTEntry* first_entry(const Key key) const;

  // 置換表上のtteのエントリーの世代を現在の世代(this->generation)にする。
  void refresh(const TTEntry* tte) const;

  // TranspositionTable::set_size()は、置換表のサイズをMB(メガバイト)単位で設定する。
  // 置換表は2の累乗のclusterで構成されており、それぞれのclusterはTTEnteryのClusterSizeで決まる。
  // (1つのClusterは、TTEntry::ClusterSize×16バイト)
  //:void set_size(uint64_t mbSize);
  // →　Stockfish 2014/10で名前が変わっていたので関数名だけ変更しておく。
  void resize(uint64_t mbSize);

  // 置換表のサイズを取得する。(学習時にサイズをランダマイズさせたいため)
  uint64_t size() const { return hashSize; }

  // TranspositionTable::clear()は、ゼロで置換表全体をクリアする。
  // テーブルがリサイズされたときや、ユーザーがUCI interface経由でテーブルのクリアを
  // 要求したときに行われる。
  void clear();

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
  void store(const Key key, Value v, Bound type, Depth d, Move m, Value statV);

  // 使用率を計算する関数。Stockfishにはないが、参考のために作った。
  // 1000分率で返す。
  int hashfull() const;

  int generation() { return gen; }

private:

  // 置換表のindexのmask用。
  // table[hashMask & 局面のhashkey] がその局面の最初のentry
  // hashMask + 1 が確保されたTTEntryの数
  //:  uint32_t hashMask;
  uint64_t hashMask;

  // 置換表の先頭を示すポインター(確保したメモリを64バイトでアラインしたもの)
  TTEntry* table;

  // 置換表のために確保した生のメモリへのポインター。開放するときに必要。
  void* mem;

  // resize()で指定されたサイズ
  uint64_t hashSize;

  // 置換表のEntryの、いま使っている世代。
  // これをroot局面が進むごとにインクリメントしていく。
  uint8_t gen; // Size must be not bigger than TTEntry::generation8
};

//: extern TranspositionTable TT;
// globalな置換表
// →　globalに確保しないように変更。


/// TranspositionTable::first_entry() returns a pointer to the first entry of
/// a cluster given a position. The lowest order bits of the key are used to
/// get the index of the cluster.

// TranspositionTable::first_entry()は、与えられた局面(のハッシュキー)に該当する
// 置換表上のclusterの最初のエントリーへのポインターを返す。
// 引数として渡されたkey(ハッシュキー)の下位ビットがclusterへのindexとして使われる。

inline TTEntry* TranspositionTable::first_entry(const Key key) const {

  //:  return table + ((uint32_t)key & hashMask);
  return table + ((uint64_t)key & hashMask);
}


/// TranspositionTable::refresh() updates the 'generation' value of the TTEntry
/// to avoid aging. It is normally called after a TT hit.

// TranspositionTable::refresh()は、TTEntryが年をとる(世代が進む)のを回避するため
// generationの値を更新する。普通、置換表にhitしたあとに呼び出される。
// TranspositionTable::generationの値が引数で指定したTTEntryの世代として設定される。

inline void TranspositionTable::refresh(const TTEntry* tte) const {

  const_cast<TTEntry*>(tte)->set_generation(gen);
}
#endif




// 詰みのスコアは置換表上は、このnodeからあと何手で詰むかというスコアを格納する。
// しかし、search()の返し値は、rootからあと何手で詰むかというスコアを使っている。
// (こうしておかないと、do_move(),undo_move()するごとに詰みのスコアをインクリメントしたりデクリメントしたり
// しないといけなくなってとても面倒くさいからである。)
// なので置換表に格納する前に、この変換をしなければならない。
// 詰みにまつわるスコアでないなら関係がないので何の変換も行わない。
// ply : root node からの手数。(ply_from_root)
inline Value value_to_tt(Value v, int ply) {

  ASSERT_LV3(-VALUE_INFINITE < v && v < VALUE_INFINITE);

  return  v >= VALUE_MATE_IN_MAX_PLY  ? v + ply
        : v <= VALUE_MATED_IN_MAX_PLY ? v - ply : v;
}

// value_to_tt()の逆関数
// ply : root node からの手数。(ply_from_root)
inline Value value_from_tt(Value v, int ply) {

  return  v == VALUE_NONE ? VALUE_NONE
    : v >= VALUE_MATE_IN_MAX_PLY  ? v - ply
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
