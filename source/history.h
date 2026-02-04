#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <array>
#include <atomic>
//#include <cassert>
#include <cmath>
//#include <cstdint>
//#include <cstdlib>
#include <limits>
#include <type_traits>  // IWYU pragma: keep
// 📝 IWYU pragmaとは、Include What You Use（IWYU） というincludeを整理してくれるツールに対する指示コメント。
//     keepというのは人間の判断で必要(暗黙的に使っている)ので消すなという意味。

#include "memory.h"
#include "misc.h"
#include "position.h"

namespace YaneuraOu {

// -----------------------
//		history
// -----------------------

// 歩の陣形に対するhistoryのサイズ
constexpr int PAWN_HISTORY_BASE_SIZE   = 8192;  // has to be a power of 2
constexpr int UINT_16_HISTORY_SIZE     = std::numeric_limits<uint16_t>::max() + 1;
// correction historyのサイズ
constexpr int CORRHIST_BASE_SIZE       = UINT_16_HISTORY_SIZE;
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE     = 5;

static_assert((PAWN_HISTORY_BASE_SIZE & (PAWN_HISTORY_BASE_SIZE - 1)) == 0,
              "PAWN_HISTORY_BASE_SIZE has to be a power of 2");

static_assert((CORRHIST_BASE_SIZE & (CORRHIST_BASE_SIZE - 1)) == 0,
              "CORRHIST_BASE_SIZE has to be a power of 2");

inline uint16_t pawn_correction_history_index(const Position& pos) { return uint16_t(pos.pawn_key()); }
// 🌈 Stockfishではhash keyはuint64_tなので、そのままuint16_tにできるが、やねうら王では、
//     Keyという構造体なので暗黙で64bitにすることはできない。ゆえに明示的なcastが必要となる。

inline uint16_t minor_piece_index(const Position& pos) { return uint16_t(pos.minor_piece_key()); }

template<Color c>
inline uint16_t non_pawn_index(const Position& pos) {
    return uint16_t(pos.non_pawn_key(c));
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
// Atomic : これがtrueなら、atomic版になる。
template<typename T, int D, bool Atomic = false>
struct StatsEntry {

	static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");

   private:
	// Atomic == Falseの時は普通のT、さもなくばstd::atomic<T>
    std::conditional_t<Atomic, std::atomic<T>, T> entry;


   public:
    void operator=(const T& v) {
        if constexpr (Atomic)
            entry.store(v, std::memory_order_relaxed);
        else
            entry = v;
    }

    operator T() const {
        if constexpr (Atomic)
            return entry.load(std::memory_order_relaxed);
        else
            return entry;
    }

	// このStatsEntry(Statsの1要素)に対して"<<"演算子でbonus値の加算が出来るようにしておく。
	// 値が範囲外にならないように制限してある。
	void operator<<(int bonus) {

		// Make sure that bonus is in range [-D, D]
		// bonusが[-D,D]の範囲に収まるようにする

		int clampedBonus = std::clamp(bonus, -D, D);

        T   val          = *this;
        *this            = val + clampedBonus - val * std::abs(clampedBonus) / D;

        assert(std::abs(T(*this)) <= D);

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
		// 📝 指数移動平均(EMA)みたいなことがしたい。
		//     しかし、EMAだと制限Dを超えてしまうことがある。Dの範囲を守りながらEMAするには上記のような更新式になる。

		ASSERT_LV3(std::abs(entry) <= D);
	}
};

enum StatsType {
	NoCaptures,
	Captures
};

template<typename T, int D, std::size_t... Sizes>
using Stats = MultiArray<StatsEntry<T, D>, Sizes...>;

template<typename T, int D, std::size_t... Sizes>
using AtomicStats = MultiArray<StatsEntry<T, D, true>, Sizes...>;

// DynStats is a dynamically sized array of Stats, used for thread-shared histories
// which should scale with the total number of threads. The SizeMultiplier gives
// the per-thread allocation count of T.

// DynStats は Stats の動的サイズ配列で、
// 全スレッド数に応じてスケールする必要がある、スレッド間で共有される履歴情報のために使われます。
// SizeMultiplier は、スレッド1本あたりに確保される T の個数を表します。

// 📓 TをSizeMultiplier分確保。
template<typename T, int SizeMultiplier>
struct DynStats {
	// s : スレッド数
    explicit DynStats(size_t s) {
		// この配列のサイズ
        size = s * SizeMultiplier;

		// 実際に確保される領域(T型でsize個)
        data = make_unique_large_page<T[]>(size);
    }

    // Sets all values in the range to 0
	// すべての要素を0にする。
	// 📓 並列で呼び出すことを想定している。threadIdxがthread id、numaTotalがthreadの総数。 
    void clear_range(int value, size_t threadIdx, size_t numaTotal) {
        size_t start = uint64_t(threadIdx) * size / numaTotal;
        assert(start < size);

		// 最後のスレッドは、末尾まで。さもなくば、次のスレッドの開始点まで。
        size_t end = threadIdx + 1 == numaTotal ? size : uint64_t(threadIdx + 1) * size / numaTotal;

        while (start < end)
            data[start++].fill(value);
    }

	// 確保されている配列のサイズ
	size_t get_size() const { return size; }

	// 配列の要素へのaccessor
	T&     operator[](size_t index) {
        assert(index < size);
        return data.get()[index];
    }
    const T& operator[](size_t index) const {
        assert(index < size);
        return data.get()[index];
    }

   private:
	// 確保されている配列のsize
    size_t            size;

	// 確保されている配列の実体
    LargePagePtr<T[]> data;
};


// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see https://www.chessprogramming.org/Butterfly_Boards

// ButterflyHistory は、現在の探索中に quiet moves がどれくらい成功または失敗したかを記録し、
// reductionや move ordering の判断に使用されます。
// これは 2 つのテーブル（各手番ごとに1つ）を使用し、手の移動元と移動先のマスでインデックス付けします。
// 詳細は https://www.chessprogramming.org/Butterfly_Boards を参照してください。
// 💡 簡単に言うと、fromの駒をtoに移動させることに対するhistory。

using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, UINT_16_HISTORY_SIZE>;

/*
	 📓 上記のような配列、将棋では添字順を入れ替えて、history[c][to]をhistory[to][c]としたい。
	     これは、配列の末尾側サイズは2のべき乗になっている方がアドレッシングが簡単で高速なコードが生成されるため。

	     しかし逆順にしているのを忘れてしまい、Stockfishのコードを参考にする時に色々面倒である。

		 やねうら王V8.60まではwrapper classを用意していたが、管理が複雑になり、Stockfishとの差分が
		 大きくなるのでやめることにした。

		 move.from_to()を呼び出した時、Stockfishでは 0～SQUARE_NB*SQUARE_NB-1までの値だが、
	     やねうら王では、0 ～ ((SQUARE_NB+7) * SQUARE_NB - 1)であることに注意。
	     ⇨ 後者のサイズとして、Move::FROM_TO_SIZEを用いると良い。
		 ⇨ move::from_to()が削除された。move::raw()でuint16_tが返る。from_toを使う必要がなくなった。

	     また、やねうら王では、ここのfrom_toで用いられるfromは、駒打ちのときに特殊な値になっていて、
	     盤上のfromとは区別される。そのため、(SQUARE_NB + 7)まで移動元がある。

		 例) from = SQUARE_NB     の時、歩打ち
			 from = SQUARE_NB + 1 の時、香打ち
				 …
		 注) 打ち駒に関して、先手と後手の歩打ちを区別する必要はない。
 　　		   なぜなら、このButterflyHistoryではその指し手の手番(Color)の区別をしているから。
*/

// LowPlyHistory is addressed by ply and move's from and to squares, used
// to improve move ordering near the root

// LowPlyHistoryはplyおよび手の「from」と「to」のマスで管理され、
// ルート付近での手順の順序を改善するために使用されます。
// 💡 ply とは 探索開始局面からの手数のこと。

using LowPlyHistory = Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, UINT_16_HISTORY_SIZE>;

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

using PawnHistory =
  DynStats<AtomicStats<std::int16_t, 8192, PIECE_NB, SQUARE_NB>, PAWN_HISTORY_BASE_SIZE>;

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

// Correction Historyのひとまとめにしたもの。
// 歩、小駒、後手の大駒、先手の大駒
template<typename T, int D>
struct CorrectionBundle {
    StatsEntry<T, D, true> pawn;
    StatsEntry<T, D, true> minor;
    StatsEntry<T, D, true> nonPawnWhite;
    StatsEntry<T, D, true> nonPawnBlack;

	// メンバーにまとめて代入するoperator
    void operator=(T val) {
        pawn         = val;
        minor        = val;
        nonPawnWhite = val;
        nonPawnBlack = val;
    }
};

namespace Detail {

template<CorrHistType>
struct CorrHistTypedef {
    using type =
      DynStats<Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB>, CORRHIST_BASE_SIZE>;
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
    using type = DynStats<Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB, COLOR_NB>,
                          CORRHIST_BASE_SIZE>;
};

}

using UnifiedCorrectionHistory =
  DynStats<MultiArray<CorrectionBundle<std::int16_t, CORRECTION_HISTORY_LIMIT>, COLOR_NB>,
           CORRHIST_BASE_SIZE>;


template<CorrHistType T>
using CorrectionHistory = typename Detail::CorrHistTypedef<T>::type;

using TTMoveHistory = StatsEntry<std::int16_t, 8192>;


// Set of histories shared between groups of threads. To avoid excessive
// cross-node data transfer, histories are shared only between threads
// on a given NUMA node. The passed size must be a power of two to make
// the indexing more efficient.

// スレッドのグループ間で共有される履歴情報の集合です。
// 過度なノード間データ転送を避けるため、履歴は
// 同一の NUMA ノード上にあるスレッド間でのみ共有されます。
// 渡される size は、インデックス計算をより効率的にするため、
// 2 の冪である必要があります。

struct SharedHistories {
    SharedHistories(size_t threadCount) :
        correctionHistory(threadCount),
        pawnHistory(threadCount) {
        assert((threadCount & (threadCount - 1)) == 0 && threadCount != 0);
        sizeMinus1         = correctionHistory.get_size() - 1;
        pawnHistSizeMinus1 = pawnHistory.get_size() - 1;
    }

    size_t get_size() const { return sizeMinus1 + 1; }

    auto& pawn_entry(const Position& pos) {
        return pawnHistory[pos.pawn_key() & pawnHistSizeMinus1];
    }
    const auto& pawn_entry(const Position& pos) const {
        return pawnHistory[pos.pawn_key() & pawnHistSizeMinus1];
    }

    auto& pawn_correction_entry(const Position& pos) {
        return correctionHistory[pos.pawn_key() & sizeMinus1];
    }
    const auto& pawn_correction_entry(const Position& pos) const {
        return correctionHistory[pos.pawn_key() & sizeMinus1];
    }

    auto& minor_piece_correction_entry(const Position& pos) {
        return correctionHistory[pos.minor_piece_key() & sizeMinus1];
    }
    const auto& minor_piece_correction_entry(const Position& pos) const {
        return correctionHistory[pos.minor_piece_key() & sizeMinus1];
    }

    template<Color c>
    auto& nonpawn_correction_entry(const Position& pos) {
        return correctionHistory[pos.non_pawn_key(c) & sizeMinus1];
    }
    template<Color c>
    const auto& nonpawn_correction_entry(const Position& pos) const {
        return correctionHistory[pos.non_pawn_key(c) & sizeMinus1];
    }

    UnifiedCorrectionHistory correctionHistory;
    PawnHistory              pawnHistory;

   private:
    size_t sizeMinus1, pawnHistSizeMinus1;
};


} // namespace YaneuraOu

#endif // #ifndef HISTORY_H_INCLUDED

