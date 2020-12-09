#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include "config.h"
#if defined(USE_MOVE_PICKER)

#include <array>
//#include <limits>
//#include <type_traits>

//#include "movegen.h"
//#include "position.h"
#include "types.h"

// -----------------------
//		history
// -----------------------

/// StatsEntryはstat tableの値を格納する。これは、大抵数値であるが、指し手やnestされたhistoryでさえありうる。
/// 多次元配列であるかのように呼び出し側でstats tablesを用いるために、
/// 生の値を用いる代わりに、history updateを行なうentry上でoperator<<()を呼び出す。

// T : このEntryの実体
// D : abs(entry) <= Dとなるように制限される。
template<typename T, int D>
class StatsEntry {

	T entry;

public:
	void operator=(const T& v) { entry = v; }
	T* operator&() { return &entry; }
	T* operator->() { return &entry; }
	operator const T&() const { return entry; }

	// このStatsEntry(Statsの1要素)に対して"<<"演算子でbonus値の加算が出来るようにしておく。
	// 値が範囲外にならないように制限してある。
	void operator<<(int bonus) {
		ASSERT_LV3(abs(bonus) <= D); // 範囲が[-D,D]であるようにする。
		// オーバーフローしないことを保証する。
		static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");

		// この式は、Stockfishのコードそのまま。
		// 試行錯誤の結果っぽくて、数学的な根拠はおそらくない。

		entry += bonus - entry * abs(bonus) / D;

		//
		// この式は、
		// 1) bouns == D (最大値)のとき、右辺が bonus - entry になって、entry == bonud == Dとなる。
		//     すなわち、絶対にDは超えない。
		// 2) bonus = entry * k (kは1に近い定数とする) のとき、
		//     右辺は　k・entry - entry*(k・entry)/D = k・entry ( 1 - entry/D ) となり、entry/D ≒ 0とみなせるとき
		//      = k・entry = bonus となり、単なるbonusをentryに加算している意味になる。
		//
		// つまり、entryにbonusを加算するのだけど、その結果がDを超えないようにちょっと減算してくれるような式になっている。
		//
		// 性質)
		// ・自然にゼロ方向に引っ張られる
		// ・絶対値がDを超えないように注意しながらentryにbonusを加算する
		// 

		ASSERT_LV3(abs(entry) <= D);
	}
};

/// Statsは、様々な統計情報を格納するために用いられる汎用的なN-次元配列である。
/// 1つ目のtemplate parameterであるTは、配列の基本的な型を示し、2つ目の
/// template parameterであるDは、<< operatorで値を更新するときに、値を[-D,D]の範囲に
/// 制限する。最後のparameter(SizeとSizes)は、配列の次元に用いられる。
template <typename T, int D, int Size, int... Sizes>
struct Stats : public std::array<Stats<T, D, Sizes...>, Size>
{
	typedef Stats<T, D, Size, Sizes...> stats;

	void fill(const T& v) {

		// For standard-layout 'this' points to first struct member
		ASSERT_LV3(std::is_standard_layout<stats>::value);
		
		typedef StatsEntry<T, D> entry;
		entry* p = reinterpret_cast<entry*>(this);
		std::fill(p, p + sizeof(*this) / sizeof(entry), v);
	}
};

template <typename T, int D, int Size>
struct Stats<T, D, Size> : public std::array<StatsEntry<T, D>, Size> {};

// stats tableにおいて、Dを0にした場合、このtemplate parameterは用いないという意味。
enum StatsParams { NOT_USED = 0 };
enum StatsType { NoCaptures, Captures };

// ButterflyHistoryは、 現在の探索中にquietな指し手がどれくらい成功/失敗したかを記録し、
// reductionと指し手オーダリングの決定のために用いられる。
// 添字は[from_to][color]の順。
// cf. http://chessprogramming.wikispaces.com/Butterfly+Boards
// 簡単に言うと、fromの駒をtoに移動させることに対するhistory。
// やねうら王では、ここで用いられるfromは、駒打ちのときに特殊な値になっていて、盤上のfromとは区別される。
// そのため、(SQ_NB + 7)まで移動元がある。
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
typedef Stats<int16_t, 10692, int(SQ_NB + 7) * int(SQ_NB) , COLOR_NB> ButterflyHistory;


// より高い探索深さにおいて、LowPlyHistoryは、root付近の成功したquietな指し手と
// PV(ttPV)にある/あったquietな指し手を記録します。LowPlyHistoryは、新しい探索の
// たびにクリアされて、反復深化の間に埋められます。
//
// Stockfishと異なり、from_to()は、int(SQ_NB + 7) * int(SQ_NB)の間の値が返るのでこの部分のサイズ変更してある。
// [ply][from_to] これは逆順にしたほうが少し高速化が図れるのだが、Stockfishでこの順に依存するコードがあったので、
// とりあえずStockfishと同じ順にしてある。
constexpr int MAX_LPH = 4; // これ以下のdepthのときにこのテーブルを用いる。
typedef Stats<int16_t, 10692, MAX_LPH,int(SQ_NB + 7)* int(SQ_NB)> LowPlyHistory;

/// CounterMoveHistoryは、直前の指し手の[to][piece]によってindexされるcounter moves(応手)を格納する。
/// cf. http://chessprogramming.wikispaces.com/Countermove+Heuristic
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
typedef Stats<Move, NOT_USED, SQ_NB , PIECE_NB> CounterMoveHistory;

/// CapturePieceToHistoryは、指し手の[to][piece][captured piece type]で示される。
// ※　Stockfishとは、添字の順番を変更してあるので注意。
//    Stockfishでは、[piece][to][captured piece type]の順。
typedef Stats<int16_t, 10692, SQ_NB, PIECE_NB , PIECE_TYPE_NB> CapturePieceToHistory;

/// PieceToHistoryは、ButterflyHistoryに似たものだが、指し手の[to][piece]で示される。
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
//     Stockfishでは[piece][to]の順。
typedef Stats<int16_t, 29952, SQ_NB , PIECE_NB> PieceToHistory;

/// ContinuationHistoryは、与えられた2つの指し手のhistoryを組み合わせたもので、
// 普通、1手前によって与えられる現在の指し手(によるcombined history)
// このnested history tableは、ButterflyBoardsの代わりに、PieceToHistoryをベースとしている。
// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
typedef Stats<PieceToHistory, NOT_USED, SQ_NB , PIECE_NB> ContinuationHistory;


// -----------------------
//   MovePicker
// -----------------------

// 指し手オーダリング器
//
// MovePickerクラスは、現在の局面から、(呼び出し)一回につきpseudo legalな指し手を一つ取り出すのに用いる。
// 最も重要なメソッドはnext_move()であり、これは、新しいpseudo legalな指し手を呼ばれるごとに返し、
// (返すべき)指し手が無くなった場合には、MOVE_NONEを返す。
// alpha beta探索の効率を改善するために、MovePickerは最初に(早い段階のnext_move()で)カットオフ(beta cut)が
// 最も出来そうな指し手を返そうとする。
//
class MovePicker
{
	// 生成順に次の1手を取得するのか、オーダリング上、ベストな指し手を取得するのかの定数
	// (このクラスの内部で用いる。)
	enum PickType { Next, Best };

public:
	// このクラスは指し手生成バッファが大きいので、コピーして使うような使い方は禁止。
	MovePicker(const MovePicker&) = delete;
	MovePicker& operator=(const MovePicker&) = delete;

	// 通常探索(search)のProbCutの処理から呼び出されるの専用。
	// threshold_ = 直前に取られた駒の価値。これ以下の捕獲の指し手は生成しない。
	MovePicker(const Position& pos_, Move ttMove_, Value threshold_ ,
		const CapturePieceToHistory* cph);

	// 静止探索(qsearch)から呼び出される時用。
	// recapSq = 直前に動かした駒の行き先の升(取り返される升)
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, const ButterflyHistory* mh ,
		const CapturePieceToHistory* cph , 
		const PieceToHistory** ch,
		Square recapSq);

	// 通常探索(search)から呼び出されるとき用。
	// cm = counter move , killers_p = killerの指し手へのポインタ
	MovePicker(const Position& pos_, Move ttMove_, Depth depth_, const ButterflyHistory* mh,
		const LowPlyHistory* lph,
		const CapturePieceToHistory* cph ,
		const PieceToHistory** ch,
		Move cm,
		const Move* killers_p,
		int ply);


	// 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
	// 指し手が尽きればMOVE_NONEが返る。
	// 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
	Move next_move(bool skipQuiets = false);

private:
	template <PickType T, typename Pred> Move select(Pred);

	// 指し手のオーダリング用
	// GenType == CAPTURES : 捕獲する指し手のオーダリング
	// GenType == QUIETS   : 捕獲しない指し手のオーダリング
	// GenType == EVASIONS : 王手回避の指し手のオーダリング
	template<MOVE_GEN_TYPE> void score();

	// range-based forを使いたいので。
	// 現在の指し手から終端までの指し手が返る。
	ExtMove* begin() { return cur; }
	ExtMove* end() { return endMoves; }

	const Position& pos;

	// コンストラクタで渡されたhistroyのポインタを保存しておく変数。
	const ButterflyHistory* mainHistory;
	const LowPlyHistory* lowPlyHistory;
	const CapturePieceToHistory* captureHistory;
	const PieceToHistory** continuationHistory;

	// 置換表の指し手(コンストラクタで渡される)
	Move ttMove;

	// refutations[0] : killer[0]
	// refutations[1] : killer[1]
	// refutations[2] : counter move(コンストラクタで渡された、前の局面の指し手に対する応手)
	// cur           : 次に返す指し手
	// endMoves      : 生成された指し手の末尾
	// endBadCapture : BadCaptureの終端(captureの指し手を試すフェイズでのendMovesから後方に向かって悪い捕獲の指し手を移動させていく時に用いる)
	ExtMove refutations[3] , *cur, *endMoves, *endBadCaptures;

	// 指し手生成の段階
	int stage;


	// RECAPUTREの指し手で移動させる先の升
	Square recaptureSquare;

	// ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
	Value threshold;

	// コンストラクタで渡された探索深さ
	Depth depth;

	// コンストラクタで渡されたply
	int ply;

	// 指し手生成バッファ
	// 最大合法手の数 = 593 , これを要素数が16の倍数になるようにpaddingすると608。
	// メモリアドレスが32byteの倍数になるようにcurを使いたいので+3して、611。
	ExtMove moves[MAX_MOVES + 11];
};

#endif // defined(USE_MOVE_PICKER)

#endif // #ifndef MOVEPICK_H_INCLUDED

