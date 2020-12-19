#ifndef _MOVE_PICKER_2017Q2_H_
#define _MOVE_PICKER_2017Q2_H_

#include "../../shogi.h"

// -----------------------
//   MovePicker
// -----------------------

#ifdef USE_MOVE_PICKER_2017Q2

// -----------------------
//		history
// -----------------------

// StatBoardsは汎用的な2次元配列であり、様々な統計情報を格納するために用いる。
template<int Size1, int Size2, typename T = s16>
struct StatBoards : public std::array<std::array<T, Size2>, Size1> {

	// ある値でこの2次元配列を丸ごと埋める。
	void fill(const T& v) {
		T* p = &(*this)[0][0];
		std::fill(p, p + sizeof(*this) / sizeof(*p), v);
	}

	// bonus値に基づくupdate
	// Dはbonusの絶対値の上限。
	void update(T& entry, int bonus, const int D) {

		// bonusの絶対値をD以内に保つことで、このentryの値の範囲を[-32 * D , 32 * D]に保つ。
		ASSERT_LV3(abs(bonus) <= D);

		// オーバーフローしていないことを保証する。
		ASSERT_LV3(abs(32 * D) < INT16_MAX);

		entry += bonus * 32 - entry * abs(bonus) / D;

		ASSERT_LV3(abs(entry) <= 32 * D);
	}
};

// ButterflyBoardsは、2つのテーブル(1つの手番ごとに1つ)があり、「指し手の移動元と移動先」によってindexされる。
// cf. http://chessprogramming.wikispaces.com/Butterfly+Boards
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
// 簡単に言うと、fromの駒をtoに移動させることに対するhistory。
// やねうら王では、ここで用いられるfromは、駒打ちのときに特殊な値になっていて、盤上のfromとは区別される。
// そのため、(SQ_NB + 7)まで移動元がある。
typedef StatBoards<int(SQ_NB + 7) * int(SQ_NB), COLOR_NB> ButterflyBoards;

/// PieceToBoardsは、指し手の[to][piece]の情報によってaddressされる。
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
// 2つ目の添字のほう、USE_DROPBIT_IN_STATSを考慮したほうがいいのだが、
// 以前計測したときには効果がなかったのでそのコードは削除した。
typedef StatBoards<SQ_NB, PIECE_NB> PieceToBoards;

// ButterflyHistoryは、 現在の探索中にquietな指し手がどれくらい成功/失敗したかを記録し、
// reductionと指し手オーダリングの決定のために用いられる。
// ButterflyBoardsをこの情報の格納のために用いる。
struct ButterflyHistory : public ButterflyBoards {

	void update(Color c, Move m, int bonus) {
		StatBoards::update((*this)[from_to(m)][c], bonus, 324);
	}
};

/// PieceToHistoryは、ButterflyHistoryに似ているが、PieceToBoardsに基づく。
struct PieceToHistory : public PieceToBoards {

	void update(Piece pc, Square to, int bonus) {
		StatBoards::update((*this)[to][pc], bonus, 936);
	}
};

// CounterMoveStatは、直前の指し手の[to][piece]でindexされるcounter moves(応手)である。
// cf. http://chessprogramming.wikispaces.com/Countermove+Heuristic
// ※　Stockfishとは、1,2番目の添字を入れ替えてあるので注意。
typedef StatBoards<SQ_NB, PIECE_NB, Move> CounterMoveStat;

// CounterMoveHistoryStatは、CounterMoveStatに似ているが、指し手の代わりに、
// full history(ButterflyBoardsの代わりに用いられるPieceTo boards)を格納する。
// ※　Stockfishとは、1,2番目の添字を入れ替えてあるので注意。
typedef StatBoards<SQ_NB, PIECE_NB, PieceToHistory> ContinuationHistory;


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
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, const ButterflyHistory* , Square recapSq);

	// 通常探索から呼び出されるとき用。
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, const ButterflyHistory* , const PieceToHistory** , Search::Stack*ss_);


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
