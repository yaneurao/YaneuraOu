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
  Value eval() const { return (Value)eval16; }
  Depth depth() const { return (Depth)depth8; }
  Bound bound() const { return (Bound)(genBound8 & 0x3); }

  uint8_t generation() const { return genBound8 & 0xfc; }
  void set_generation(uint8_t g) { genBound8 = bound() | g; }

  // 置換表のエントリーに対して与えられたデータを保存する。上書き動作
  //    ToDo:genBound更新
  void save(Key k, Value v, Move m)
  {
    // hash keyの上位16bitが格納されているkeyと一致していないなら
    // 何も考えずに指し手を上書き(新しいデータのほうが価値があるので)
    if ((k >> 48) != key16)
      move16 = (uint16_t)m;

    // かきかけ


    // このエントリーの現在の内容のほうが価値があるなら上書きしない。
    if ((k >> 48) != key16 )
    {
      key16   = (int16_t)(k >> 48);
      value16 = (int16_t)v;
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
  uint16_t value16;

  // 評価関数の評価値
  uint16_t eval16;

  // entryのgeneration上位6bit + Bound下位2bitのpackしたもの。
  // generationはエントリーの世代を表す。TranspositionTableで新しい探索ごとに+4されていく。
  uint8_t genBound8;

  // そのときの残り深さ(これが大きいものほど価値がある)
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

  // 置換表の使用率を1000分率で返す。(USIプロトコルで統計情報として出力するのに使う)
  int hashfull() const;

  TranspositionTable() { mem = nullptr; clusterCount = 0; }
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

extern TranspositionTable TT;

#endif // _TT_H_
