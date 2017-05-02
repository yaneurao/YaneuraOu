#ifndef _MOVE_PICKER_2017Q2_H_
#define _MOVE_PICKER_2017Q2_H_

#include "../shogi.h"

// -----------------------
//   MovePicker
// -----------------------

#ifdef USE_MOVE_PICKER_2017Q2

// -----------------------
//  history , counter move
// -----------------------

// fromの駒をtoに移動させることに対するhistory
// 駒打ちはfromが特殊な値になっていて、盤上のfromとは区別される。
struct HistoryStats
{
	// 値の最大値
	static const int Max = Value(1 << 28);

	int get(Color c, Move m) const { return table[from_sq(m) + (is_drop(m) ? SQ_NB:0)][to_sq(m)][c]; }
	void clear() { std::memset(table, 0, sizeof(table)); }

	void update(Color c, Move m, int v)
	{
		Square from = from_sq(m);
		Square to = to_sq(m);

		// 駒打ちを分類すべきだと思うので、駒種に応じてfromの位置を調整する。
		if (is_drop(m))
			from += SQ_NB;

		ASSERT_LV3(from < SQ_NB_PLUS1 + 7);

		const int D = 324;

		ASSERT_LV3(abs(int(v)) <= D); // 下記の公式に対する一貫性チェック
		
		table[from][to][c] -= table[from][to][c] * abs(int(v)) / D;
		table[from][to][c] += int(v) * 32;
	}
private:
	// table[from][to][color]となっているが、fromはSQ_NB_PLUS1 + 打ち駒の7種
	int table[SQ_NB_PLUS1 + 7][SQ_NB_PLUS1][COLOR_NB];
};


// Pieceを升sqに移動させるときの値(T型)
// 移動先のみ。移動元に関しては情報を持っていない。
template<typename T>
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
  void update(Piece pc, Square to, int v) {

    // USE_DROPBIT_IN_STATSが有効なときはpcとして +32したものを駒打ちとして格納する。
    // なので is_ok(pc)というassertは書けない。

    ASSERT_LV4(is_ok(to));


	const int D = 936;

	ASSERT_LV3(abs(int(v)) <= D); // 下記の公式に対する一貫性チェック

	table[to][pc] -= table[to][pc] * abs(int(v)) / D;
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
typedef Stats<int             > CounterMoveStats;
typedef Stats<CounterMoveStats> CounterMoveHistoryStats;

enum Stages : int;
namespace Search { struct Stack; }

// 指し手オーダリング器
struct MovePicker
{
	// このクラスは指し手生成バッファが大きいので、コピーして使うような使い方は禁止。
	MovePicker(const MovePicker&) = delete;
	MovePicker& operator=(const MovePicker&) = delete;

	// 通常探索時にProbCutの処理から呼び出されるの専用。threshold_ = 直前に取られた駒の価値。これ以下の捕獲の指し手は生成しない。
	MovePicker(const Position& pos_, Move ttMove_, Value threshold_);

	// 静止探索から呼び出される時用。recapSq = 直前に動かした駒の行き先の升(取り返される升)
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, Square recapSq);

	// 通常探索から呼び出されるとき用。
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, Search::Stack*ss_);


	// 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
	// 指し手が尽きればMOVE_NONEが返る。
	// 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
	Move next_move(bool skipQuiets = false);

private:
	// 指し手のオーダリング用
	// GenType == CAPTURES : 捕獲する指し手のオーダリング
	// GenType == QUIETS   : 捕獲しない指し手のオーダリング
	// GenType == EVASIONS : 王手回避の指し手のオーダリング
	template<MOVE_GEN_TYPE> void score();

	// range-based forを使いたいので。
	ExtMove* begin() { return moves; }
	ExtMove* end() { return endMoves; }

	const Position& pos;

	// node stack
	Search::Stack* ss;

	// これはオーダリングしないからExtMoveである必要はない。
	Move killers[2];

	// コンストラクタで渡された、前の局面の指し手に対する応手
	Move countermove;

	// コンストラクタで渡された探索深さ
	Depth depth;

	// 置換表の指し手
	Move ttMove;

	// RECAPUTREの指し手で移動させる先の升
	Square recaptureSquare;

	// ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
	Value threshold;

	// 指し手生成の段階
	int stage;

	// 次に返す指し手 , 生成された指し手の末尾 , BadCaptureの終端(これは、movesの先頭から再利用していく)
	ExtMove *cur, *endMoves, *endBadCaptures;

	// 指し手生成バッファ
	ExtMove moves[MAX_MOVES];

#ifdef MUST_CAPTURE_SHOGI_ENGINE
	// 合法な駒を捕獲する指し手が1手でもあるのか
	bool mustCapture;

	// ↑のフラグを更新する
	void checkMustCapture();

	// 本来のnext_move()を以下の関数に移動させて、
	// next_move()はmustCaptureのチェックを行なう関数と差し替える。
	Move next_move2();
#endif

};
#endif // USE_MOVE_PICKER_2016Q3

#endif // _MOVE_PICKER_2016Q3_H_
