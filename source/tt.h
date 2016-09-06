#ifndef _TT_H_
#define _TT_H_

#include "shogi.h"
#include "extra/key128.h"

// --------------------
//       置換表
// --------------------

// 置換表エントリー
// 本エントリーは10bytesに収まるようになっている。3つのエントリーを並べたときに32bytesに収まるので
// CPUのcache lineに一発で載るというミラクル。
struct TTEntry {

  Move move() const { return (Move)move16; }
  Value value() const { return (Value)value16; }
#ifndef  NO_EVAL_IN_TT
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
  void save(Key k, Value v, Bound b,Depth d, Move m,
#ifndef NO_EVAL_IN_TT
    Value eval,
#endif
    uint8_t gen)
  {
    ASSERT_LV3((-VALUE_INFINITE < v && v < VALUE_INFINITE) || v == VALUE_NONE);

	// ToDo: 探索部によってはVALUE_INFINITEを書き込みたいのかも知れない…。うーん。
	//  ASSERT_LV3((-VALUE_INFINITE <= v && v <= VALUE_INFINITE) || v == VALUE_NONE);

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
      || (d / ONE_PLY > depth8 - 4)
    /*|| g != generation() // probe()において非0のkeyとマッチした場合、その瞬間に世代はrefreshされている。　*/
      || b == BOUND_EXACT
      )
    {
      key16     = (uint16_t)(k >> 48);
      value16   = (int16_t)v;
#ifndef NO_EVAL_IN_TT
      eval16    = (int16_t)eval;
#endif
      genBound8 = (uint8_t)(gen | b);
      depth8    = (int8_t)(d / ONE_PLY);
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

#ifndef NO_EVAL_IN_TT
  // 評価関数の評価値
  int16_t eval16;
#endif

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

  // keyの下位bitをClusterのindexにしてその最初のTTEntry*を返す。
  TTEntry* first_entry(const Key key) const {
    return &table[(size_t)key & (clusterCount - 1)].entry[0];
  }

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

#ifndef  NO_EVAL_IN_TT
  // 1クラスターにおけるTTEntryの数
  // TTEntry 10bytes×3つ + 2(padding) = 32bytes
  static const int ClusterSize = 3;
#else
  // TTEntry 8bytes×4つ = 32bytes
  static const int ClusterSize = 4;
#endif

  struct Cluster {
    TTEntry entry[ClusterSize];
#ifndef  NO_EVAL_IN_TT
    int8_t padding[2]; // 全体を32byteぴったりにするためのpadding
#endif
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
