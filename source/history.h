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
#include "position.h"

// -----------------------
//		history
// -----------------------

#if defined(ENABLE_PAWN_HISTORY)
// 歩の陣形に対するhistory
constexpr int PAWN_HISTORY_SIZE = 512;
inline int pawn_structure(const Position& pos) { return pos.pawn_key() & (PAWN_HISTORY_SIZE - 1); }
#endif
constexpr int LOW_PLY_HISTORY_SIZE     = 4;

// StatsEntryはstat tableの値を格納する。これは、大抵数値であるが、指し手やnestされたhistoryでさえありうる。
// 多次元配列であるかのように呼び出し側でstats tablesを用いるために、
// 生の値を用いる代わりに、history updateを行なうentry上でoperator<<()を呼び出す。

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

		// オーバーフローしないことを保証する。
		static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");

		// この式は、Stockfishのコードそのまま。
		// 試行錯誤の結果っぽくて、数学的な根拠はおそらくない。

		// Make sure that bonus is in range [-D, D]
		// bonusが[-D,D]の範囲に収まるようにする

		int clampedBonus = std::clamp(bonus, -D, D);
		entry += clampedBonus - entry * std::abs(clampedBonus) / D;

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

// Statsは、様々な統計情報を格納するために用いられる汎用的なN-次元配列である。
// 1つ目のtemplate parameterであるTは、配列の基本的な型を示し、2つ目の
// template parameterであるDは、<< operatorで値を更新するときに、値を[-D,D]の範囲に
// 制限する。最後のparameter(SizeとSizes)は、配列の次元に用いられる。
template <typename T, int D, int Size, int... Sizes>
struct Stats : public std::array<Stats<T, D, Sizes...>, Size>
{
    using stats = Stats<T, D, Size, Sizes...>;

	void fill(const T& v) {

		// For standard-layout 'this' points to first struct member
		ASSERT_LV3(std::is_standard_layout_v<stats>);

	    using entry = StatsEntry<T, D>;
		entry* p = reinterpret_cast<entry*>(this);
		std::fill(p, p + sizeof(*this) / sizeof(entry), v);
	}
};

template <typename T, int D, int Size>
struct Stats<T, D, Size> : public std::array<StatsEntry<T, D>, Size> {};

// In stats table, D=0 means that the template parameter is not used
// stats tableにおいて、Dを0にした場合、このtemplate parameterは用いないという意味。
enum StatsParams { NOT_USED = 0 };

enum StatsType { NoCaptures, Captures };

// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see www.chessprogramming.org/Butterfly_Boards (~11 elo)

// ButterflyHistoryは、 現在の探索中にquietな指し手がどれくらい成功/失敗したかを記録し、
// reductionと指し手オーダリングの決定のために用いられる。
// cf. http://chessprogramming.wikispaces.com/Butterfly+Boards
// 簡単に言うと、fromの駒をtoに移動させることに対するhistory。

//using ButterflyHistory = Stats<int16_t, 7183, COLOR_NB, int(SQUARE_NB) * int(SQUARE_NB)>;

// ↑ このような配列、将棋では添字順を入れ替えて、history[c][to]をhistory[to][c]としたい。
// すなわち、こう
// using ButterflyHistory = Stats<int16_t, 7183, int(SQUARE_NB + 7) * int(SQUARE_NB) , COLOR_NB>;
// 書きたい。
// 
// これは、配列の末尾側サイズは2のべき乗になっている方がアドレッシングが簡単で高速なコードが生成されるため。
// 
// しかし逆順にしているのを忘れてしまい、Stockfishのコードを参考にする時に色々面倒である。
// 
// そこで、以下⇓のようなwrapper classを書いて、このopeator()を通じてアクセスを行うことにする。
// これにより、添字の順番はStockfishと同じ形になり、かつ、コンパイル時に引数の型チェックがなされる。
//
// ■ 備考
//
// move.from_to()を呼び出した時、Stockfishでは 0～SQUARE_NB*SQUARE_NB-1までの値だが、
// やねうら王では、0 ～ ((SQUARE_NB+7) * SQUARE_NB - 1)であることに注意。
// ⇨ 後者のサイズとして、Move::FROM_TO_SIZEを用いると良い。

struct ButterflyHistory
{
	using T = int16_t;             // StatsEntryの型
	static constexpr int D = 7183; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (Color c, int from_to) const {
        return stats[from_to][c];
    }

	StatsEntry<T, D>& operator() (Color c, int from_to) {
        return stats[from_to][c];
    }
	void fill(T t) { stats.fill(t); }

private:
	// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
	// やねうら王の実際の格納配列(stats)では、添字は[from_to][color]の順。

	// また、やねうら王では、ここのfrom_toで用いられるfromは、駒打ちのときに特殊な値になっていて、
	// 盤上のfromとは区別される。そのため、(SQUARE_NB + 7)まで移動元がある。
	// 例) from = SQUARE_NB     の時、歩打ち
	//     from = SQUARE_NB + 1 の時、香打ち
	//         …
	// 注) 打ち駒に関して、先手と後手の歩打ちを区別する必要はない。
	// 　　なぜなら、このButterflyHistoryではその指し手の手番(Color)の区別をしているから。
	// 
	Stats<T, D , Move::FROM_TO_SIZE, COLOR_NB> stats;
};


// LowPlyHistory is adressed by play and move's from and to squares, used
// to improve move ordering near the root

// LowPlyHistoryはプレイおよび手の「from」と「to」のマスで管理され、
// ルート付近での手順の順序を改善するために使用されます。

struct LowPlyHistory
{
	using T = int16_t;             // StatsEntryの型
	static constexpr int D = 7183; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (int ply, int from_to) const {
		return stats[from_to][ply];
	}

	StatsEntry<T, D>& operator() (int ply, int from_to) {
		return stats[from_to][ply];
	}
	void fill(T t) { stats.fill(t); }

	//using LowPlyHistory = Stats<int16_t, 7183, LOW_PLY_HISTORY_SIZE, int(SQUARE_NB)* int(SQUARE_NB)>;
	// ⇨ Stockfishのコードだと、末尾が2の冪にならないので並び順を変更する。
	Stats<int16_t, D, Move::FROM_TO_SIZE, LOW_PLY_HISTORY_SIZE> stats;
};

// CapturePieceToHistory is addressed by a move's [piece][to][captured piece type]
// CapturePieceToHistoryは、指し手の [piece][to][captured piece type]で示される。

//using CapturePieceToHistory = Stats<int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

struct CapturePieceToHistory
{
	using T = int16_t;          // StatsEntryの型
	static constexpr int D = 10692; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (Piece moved_pc, Square sq, PieceType captured_pt) const {
        return stats[sq][moved_pc][captured_pt];
    }

	StatsEntry<T, D>& operator() (Piece moved_pc, Square sq, PieceType captured_pt) {
        return stats[sq][moved_pc][captured_pt];
    }

	void fill(T t) { stats.fill(t); }

private:
	// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
	//    やねうら王の実際の格納配列(stats)では、[to][piece][captured piece type]の順。
	Stats<T, D , SQUARE_NB, PIECE_NB , PIECE_TYPE_NB> stats;
};


// PieceToHistory is like ButterflyHistory but is addressed by a move's [piece][to]
// PieceToHistoryは、ButterflyHistoryに似たものだが、指し手の[piece][to]で示される。

//using PieceToHistory = Stats<int16_t, 29952, PIECE_NB, SQUARE_NB>;

struct PieceToHistory
{
	using T = int16_t;          // StatsEntryの型
	static constexpr int D = 29952; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (Piece pc , Square to) const {
        return stats[to][pc];
    }

	StatsEntry<T, D>& operator() (Piece pc , Square to) {
        return stats[to][pc];
    }

	void fill(T t) { stats.fill(t); }

private:
	// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
	//    やねうら王の実際の格納配列(stats)では、[to][piece]の順。
	Stats<T, D , SQUARE_NB, PIECE_NB> stats;
};


// ContinuationHistory is the combined history of a given pair of moves, usually
// the current one given a previous one. The nested history table is based on
// PieceToHistory instead of ButterflyBoards.
// (~63 elo)
// ContinuationHistoryは、与えられた2つの指し手のhistoryを組み合わせたもので、
// 普通、1手前によって与えられる現在の指し手(によるcombined history)
// このnested history tableは、ButterflyBoardsの代わりに、PieceToHistoryをベースとしている。
//using ContinuationHistory = Stats<PieceToHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

struct ContinuationHistory
{
	using T = PieceToHistory;      // StatsEntryの型
	static constexpr int D = NOT_USED; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (Piece pc , Square to) const {
        return stats[to][pc];
    }

	StatsEntry<T, D>& operator() (Piece pc , Square to) {
        return stats[to][pc];
    }

	void fill(int16_t t) {
		// Stockfish 16のthread.cppにあった初期化コード
		for(auto& to :stats)
			for(auto& h : to)
				h->fill(t);
	}

private:
	// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
	//    やねうら王の実際の格納配列(stats)では、[to][piece]の順。
	Stats<T, D, SQUARE_NB, PIECE_NB> stats;
};

#if defined(ENABLE_PAWN_HISTORY)
// PawnHistory is addressed by the pawn structure and a move's [piece][to]
// 歩の陣形に対するhistory。
//using PawnHistory = Stats<int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;

struct PawnHistory
{
	using T = int16_t;         // StatsEntryの型
	static constexpr int D = 8192; // StatsEntryの範囲

	// 必ず以下のアクセッサを通してアクセスすること。
	// ※ 引数の順番は、Stockfishの配列の添字の順番と合わせてある。

	const StatsEntry<T, D>& operator() (int pawn_key, Piece pc , Square to) const {
        return stats[pawn_key][to][pc];
    }

	StatsEntry<T, D>& operator() (int pawn_key, Piece pc , Square to) {
        return stats[pawn_key][to][pc];
    }

	void fill(T t) {
		stats.fill(t);
	}

private:
	// ※　Stockfishとは、添字の順番を入れ替えてあるので注意。
	//    やねうら王の実際の格納配列(stats)では、
	// 	     Stockfishでは、 [pawn_key][pc][to]の順。
	//		 やねうら王では、[pawn_key][to][pc]の順。
	Stats<T, D, PAWN_HISTORY_SIZE, SQUARE_NB, PIECE_NB> stats;
};

#endif

#endif // #ifndef HISTORY_H_INCLUDED

