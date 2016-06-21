#ifndef _MOVE_PICKER_2016Q2_H_
#define _MOVE_PICKER_2016Q2_H_

#include "../shogi.h"

// -----------------------
//   MovePicker
// -----------------------

#ifdef USE_MOVE_PICKER_2016Q2

#include "../search.h"

// -----------------------
//  history , counter move
// -----------------------

// Pieceを升sqに移動させるときの値(T型)
// CM : CounterMove用フラグ
template<typename T, bool CM = false>
struct Stats {

  // このtableの要素の最大値
  static const Value Max = Value(1 << 28);

  // tableの要素の値を取り出す
  const T* operator[](Square to) const {
    ASSERT_LV4(is_ok(to));
    return table[to];
  }
  T* operator[](Square to) {
    ASSERT_LV4(is_ok(to));
    return table[to];
  }

  // tableのclear
  void clear() { memset(table, 0, sizeof(table)); }

  // tableに指し手を格納する。(Tの型がMoveのとき)
  void update(Piece pc, Square to, Move m)
  {
    ASSERT_LV4(is_ok(to));
    table[to][pc] = m;
  }

  // tableに値を格納する(Tの型がValueのとき)
  void update(Piece pc, Square to, Value v) {

    // USE_DROPBIT_IN_STATSが有効なときはpcとして +32したものを駒打ちとして格納する。
    // なので is_ok(pc)というassertは書けない。

    ASSERT_LV4(is_ok(to));

    // abs(v) <= 324に制限する。

    //v = max((Value)-324, v);
    //v = min((Value)+324, v);

    // ToDo : ↑と↓と、どちらが良いのか..
    if (abs(int(v) >= 324))
      return ;

    table[to][pc] -= table[to][pc] * abs(int(v)) / (CM ? 936 : 324);
    table[to][pc] += int(v) * 32;
  }

private:
  // Pieceを升sqに移動させるときの値
  // ※　Stockfishとは添字が逆順だが、将棋ではPIECE_NBのほうだけが2^Nなので仕方がない。
  // NULL_MOVEのときは、[color][NO_PIECE]を用いる
#ifndef USE_DROPBIT_IN_STATS
  T table[SQ_NB_PLUS1][PIECE_NB];
#else
  T table[SQ_NB_PLUS1][(int)PIECE_NB*2];
#endif
};

// Statsは、pcをsqの升に移動させる指し手に対してT型の値を保存する。
// TがMoveのときは、指し手に対する指し手、すなわち、"応手"となる。
// TがValueのときは指し手に対するスコアとなる。これがhistory table(HistoryStatsとCounterMoveStats)
// このStats<CounterMoveStats>は、直前の指し手に対する、あらゆる指し手に対するスコアである。

typedef Stats<Move            > MoveStats;
typedef Stats<Value, false    > HistoryStats;
typedef Stats<Value, true     > CounterMoveStats;
typedef Stats<CounterMoveStats> CounterMoveHistoryStats;


enum Stages : int;

// 指し手オーダリング器
struct MovePicker
{
  // このクラスは指し手生成バッファが大きいので、コピーして使うような使い方は禁止。
  MovePicker(const MovePicker&) = delete;
  MovePicker& operator=(const MovePicker&) = delete;

  // 通常探索から呼び出されるとき用。
  MovePicker(const Position& pos_, Move ttMove_, Depth depth_, Search::Stack*ss_);

  // 静止探索から呼び出される時用。recapSq = 直前に動かした駒の行き先の升(取り返される升)
  MovePicker(const Position& pos_, Move ttMove_, Depth depth_, Square recapSq);
  
  // 通常探索時にProbCutの処理から呼び出されるの専用。threshold_ = 直前に取られた駒の価値。これ以下の捕獲の指し手は生成しない。
  MovePicker(const Position& pos_, Move ttMove_, Value threshold_);

  // 次の指し手をひとつ返す
  // 指し手が尽きればMOVE_NONEが返る。
  Move next_move();

private:

  // 次のstageにするため、必要なら指し手生成器で指し手を生成する。
  void generate_next_stage();

  // range-based forを使いたいので。
  ExtMove* begin() { return moves; }
  ExtMove* end() { return endMoves; }

  // beginからendのなかでベストのスコアのものを先頭(begin)に移動させる。
  Move pick_best(ExtMove* begin, ExtMove* end);

  // CAPTUREの指し手をオーダリング
  void score_captures();

  // QUIETの指し手をスコアリングする。
  void score_quiets();

  // 王手回避の指し手をスコアリングする。
  void score_evasions();

  const Position& pos;

  // node stack
  Search::Stack* ss;

  // コンストラクタで渡された、前の局面の指し手に対する応手
  Move countermove;

  // コンストラクタで渡された探索深さ
  Depth depth;

  // RECAPUTREの指し手で移動させる先の升
  Square recaptureSquare;

  // 置換表の指し手
  Move ttMove;

  // killer move 2個 + counter move 1個 + ttMove(QUIETS時) = 3個
  // これはオーダリングしないからExtMoveである必要はない。
  ExtMove killers[4];

  // ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
  Value threshold;

  // 指し手生成の段階
  Stages stage;

  // BadCaptureの終端(これはメモリの前方に向かって使っていく)。
  ExtMove *endBadCaptures = moves + MAX_MOVES - 1;
  // 指し手生成バッファと、次に返す指し手、生成された指し手の末尾
  ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;
};
#endif // USE_MOVE_PICKER_2016Q2

#endif // _MOVE_PICKER_2016Q2_H_
