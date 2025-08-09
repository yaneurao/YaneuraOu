#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <array>
//#include <cassert>
#include <cmath>
//#include <cstdint>
//#include <cstdlib>
#include <limits>
//#include <type_traits>

//#include "movegen.h"
#include "types.h"
#include "misc.h"
#include "position.h"

namespace YaneuraOu {

// -----------------------
//		history
// -----------------------

// 歩の陣形に対するhistory
constexpr int PAWN_HISTORY_SIZE        = 512;    // has to be a power of 2
constexpr int CORRECTION_HISTORY_SIZE  = 32768;  // has to be a power of 2
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE     = 5;

static_assert((PAWN_HISTORY_SIZE & (PAWN_HISTORY_SIZE - 1)) == 0,
              "PAWN_HISTORY_SIZE has to be a power of 2");

static_assert((CORRECTION_HISTORY_SIZE & (CORRECTION_HISTORY_SIZE - 1)) == 0,
              "CORRECTION_HISTORY_SIZE has to be a power of 2");

inline int pawn_history_index(const Position& pos) {
    return pos.pawn_key() & (PAWN_HISTORY_SIZE - 1);
}

inline int pawn_correction_history_index(const Position& pos) {
    return pos.pawn_key() & (CORRECTION_HISTORY_SIZE - 1);
}

inline int minor_piece_index(const Position& pos) {
    return pos.minor_piece_key() & (CORRECTION_HISTORY_SIZE - 1);
}

template<Color c>
inline int non_pawn_index(const Position& pos) {
    return pos.non_pawn_key(c) & (CORRECTION_HISTORY_SIZE - 1);
}

// StatsEntry is the container of various numerical statistics. We use a class
// instead of a naked value to directly call history update operator<<() on
// the entry. The first template parameter T is the base type of the array,
// and the second template parameter D limits the range of updates in [-D, D]
// when we update values with the << operator

// StatsEntry は各種数値統計のコンテナです。
// 生の値ではなくクラスを使うことで、エントリに対して直接 << 演算子で履歴を更新できます。
// 最初のテンプレートパラメータ T は配列の基本型を表し、
// 2 番目のテンプレートパラメータ D は << 演算子による更新時の範囲を [-D, D] に制限します。

// T : このEntryの実体
// D : abs(entry) <= Dとなるように制限される。
template<typename T, int D>
class StatsEntry {

	//static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");
	//static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");

	T entry;

public:
	StatsEntry& operator=(const T& v) {
		entry = v;
		return *this;
	}
	operator const T&() const { return entry; }

	// このStatsEntry(Statsの1要素)に対して"<<"演算子でbonus値の加算が出来るようにしておく。
	// 値が範囲外にならないように制限してある。
	void operator<<(int bonus) {

		// Make sure that bonus is in range [-D, D]
		// bonusが[-D,D]の範囲に収まるようにする

		int clampedBonus = std::clamp(bonus, -D, D);
		entry += clampedBonus - entry * std::abs(clampedBonus) / D;

		// ※　この式は、Stockfishのコードそのまま。
		// 試行錯誤の結果っぽくて、数学的な根拠はおそらくない。

		// 解説)
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

		ASSERT_LV3(std::abs(entry) <= D);
	}
};

enum StatsType {
	NoCaptures,
	Captures
};

template<typename T, int D, std::size_t... Sizes>
using Stats = MultiArray<StatsEntry<T, D>, Sizes...>;


// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see https://www.chessprogramming.org/Butterfly_Boards

// ButterflyHistory は、現在の探索中に quiet moves がどれくらい成功または失敗したかを記録し、
// reductionや move ordering の判断に使用されます。
// これは 2 つのテーブル（各手番ごとに1つ）を使用し、手の移動元と移動先のマスでインデックス付けします。
// 詳細は https://www.chessprogramming.org/Butterfly_Boards を参照してください。
// 💡 簡単に言うと、fromの駒をtoに移動させることに対するhistory。

#if STOCKFISH
using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, int(SQUARE_NB)* int(SQUARE_NB)>;
#else
using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, Move::FROM_TO_SIZE>;
#endif

/*
	 📓 上記のような配列、将棋では添字順を入れ替えて、history[c][to]をhistory[to][c]としたい。
	     これは、配列の末尾側サイズは2のべき乗になっている方がアドレッシングが簡単で高速なコードが生成されるため。

	     しかし逆順にしているのを忘れてしまい、Stockfishのコードを参考にする時に色々面倒である。

		 やねうら王V8.60まではwrapper classを用意していたが、管理が複雑になり、Stockfishとの差分が
		 大きくなるのでやめることにした。

		 move.from_to()を呼び出した時、Stockfishでは 0～SQUARE_NB*SQUARE_NB-1までの値だが、
	     やねうら王では、0 ～ ((SQUARE_NB+7) * SQUARE_NB - 1)であることに注意。
	     ⇨ 後者のサイズとして、Move::FROM_TO_SIZEを用いると良い。

	     また、やねうら王では、ここのfrom_toで用いられるfromは、駒打ちのときに特殊な値になっていて、
	     盤上のfromとは区別される。そのため、(SQUARE_NB + 7)まで移動元がある。

		 例) from = SQUARE_NB     の時、歩打ち
			 from = SQUARE_NB + 1 の時、香打ち
				 …
		 注) 打ち駒に関して、先手と後手の歩打ちを区別する必要はない。
 　　		   なぜなら、このButterflyHistoryではその指し手の手番(Color)の区別をしているから。
*/

// LowPlyHistory is adressed by play and move's from and to squares, used
// to improve move ordering near the root

// LowPlyHistoryはプレイおよび手の「from」と「to」のマスで管理され、
// ルート付近での手順の順序を改善するために使用されます。

using LowPlyHistory =
#if STOCKFISH
Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, int(SQUARE_NB)* int(SQUARE_NB)>;
#else
Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, Move::FROM_TO_SIZE>;
#endif

// CapturePieceToHistory is addressed by a move's [piece][to][captured piece type]
// CapturePieceToHistoryは、指し手の [piece][to][captured piece type]で示される。

using CapturePieceToHistory = Stats<std::int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// PieceToHistory is like ButterflyHistory but is addressed by a move's [piece][to]
// PieceToHistoryは、ButterflyHistoryに似たものだが、指し手の[piece][to]で示される。

using PieceToHistory = Stats<std::int16_t, 30000, PIECE_NB, SQUARE_NB>;

// ContinuationHistory is the combined history of a given pair of moves, usually
// the current one given a previous one. The nested history table is based on
// PieceToHistory instead of ButterflyBoards.

// ContinuationHistoryは、与えられた2つの指し手のhistoryを組み合わせたもので、
// 普通、1手前によって与えられる現在の指し手(によるcombined history)
// このnested history tableは、ButterflyBoardsの代わりに、PieceToHistoryをベースとしている。

using ContinuationHistory = MultiArray<PieceToHistory, PIECE_NB, SQUARE_NB>;

// PawnHistory is addressed by the pawn structure and a move's [piece][to]
// PawnHistoryは、pawn structureと指し手の[piece][to]で示される。
// ※　歩の陣形に対するhistory。

using PawnHistory = Stats<std::int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;

// Correction histories record differences between the static evaluation of
// positions and their search score. It is used to improve the static evaluation
// used by some search heuristics.
// see https://www.chessprogramming.org/Static_Evaluation_Correction_History

// Correction History(修正履歴)は、局面の静的評価と探索スコアとの差異を記録する。
// これは、一部の探索ヒューリスティックで使用される静的評価を改善するために用いられる。

enum CorrHistType {
    Pawn,          // By color and pawn structure
    Minor,         // By color and positions of minor pieces (Knight, Bishop)
    NonPawn,       // By non-pawn material positions and color
    PieceTo,       // By [piece][to] move
    Continuation,  // Combined history of move pairs
};
/*
	📓 CorrHistType(Correction History Type)

		Pawn         : 歩の陣形(先後の区別はする)に対するもの
		Minor        : minor piece(小駒。先後の区別はする)に対するもの。将棋では、香、桂、銀、金とその成り駒。
		NonPawn      : 歩以外の陣形(先後の区別はする)
		PieceTo      : 移動させる駒と移動先に対するもの。
		Continuation : 指し手のペアの組み合わせhistory。
*/

namespace Detail {

template<CorrHistType>
struct CorrHistTypedef {
    using type = Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_SIZE, COLOR_NB>;
};

template<>
struct CorrHistTypedef<PieceTo> {
    using type = Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<Continuation> {
    using type = MultiArray<CorrHistTypedef<PieceTo>::type, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<NonPawn> {
    using type =
      Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_SIZE, COLOR_NB, COLOR_NB>;
};

}

template<CorrHistType T>
using CorrectionHistory = typename Detail::CorrHistTypedef<T>::type;

using TTMoveHistory = StatsEntry<std::int16_t, 8192>;

} // namespace YaneuraOu

#endif // #ifndef HISTORY_H_INCLUDED

