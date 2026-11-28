#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <type_traits>

#include "types.h"
#include "misc.h"

#if !STOCKFISH
#include <iosfwd>
#endif

namespace YaneuraOu {

#if !STOCKFISH
struct Bitboard;
#endif

namespace Bitboards {

std::string pretty(Bitboard b);

#if !STOCKFISH
	// Bitboard関連のテーブル初期化のための関数
	void init();
#endif

} // namespace YaneuraOu::Bitboards

#if STOCKFISH
#ifdef USE_AVX512
// clang-format off
inline const __m512i AllSquares = _mm512_set_epi8(
    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
    40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
// clang-format on
#endif

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFF;
constexpr Bitboard Rank2BB = Rank1BB << (8 * 1);
constexpr Bitboard Rank3BB = Rank1BB << (8 * 2);
constexpr Bitboard Rank4BB = Rank1BB << (8 * 3);
constexpr Bitboard Rank5BB = Rank1BB << (8 * 4);
constexpr Bitboard Rank6BB = Rank1BB << (8 * 5);
constexpr Bitboard Rank7BB = Rank1BB << (8 * 6);
constexpr Bitboard Rank8BB = Rank1BB << (8 * 7);
#else
// --------------------
//     Bitboard
// --------------------

// Bitboardをゼロクリアするコンストラクタに指定する引数
// 例) Bitboard(ZERO) のように指定するとゼロクリアされたBitboardが出来上がる。
enum BitboardZero{ ZERO };

// Bitboardクラスは、コンストラクタでの初期化が保証できないので(オーバーヘッドがあるのでやりたくないので)
// GCC 7.1.0以降で警告が出るのを回避できない。ゆえに、このクラスではこの警告を抑制する。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// Bitboard本体クラス
// StockfishのBitboardはu64だが、将棋版は2つの64bitに81升を格納する128bitクラス。
// 下位wordのbit0～62がSquare 0～62、上位wordのbit0～17がSquare 63～80に対応する。
// 下位wordのbit63は未使用なので、格納上のbit位置をそのままSquareへ変換してはならない。
// +、-、<<、>>は各64bit単位で計算し、word間の桁上がり・桁借り・bit移動は行わない。

struct alignas(16) Bitboard
{
#if defined (USE_SSE2)

	union
	{
		// 64bitずつとして扱うとき用
		// SSE4.1以上なら、このメンバを用いずに 変数m の方を用いて、一貫して128bitレジスタとして扱ったほうが良いと思う。

		u64 p[2];

		// SSEで取り扱い時用
		// 下位wordのbit0～62がSQ_11～SQ_79、上位wordのbit0～17がSQ_81～SQ_99を表す。
		// wordをまたぐときのSquareとの対応は、クラス冒頭の説明を参照すること。
		//
		// ただしbit63は未使用。これは、ここを余らせることで香の利きや歩の打てる場所を求めやすくする。
		// Aperyを始めとするmagic bitboard派によって考案された。

		// ここから上位/下位64bitを取り出すのは、メンバのextract64()を使うべし。

		__m128i m;
	};

#else // no SSE
	u64 p[2];
#endif

#if defined (USE_SSE2)
	// SSE2が使えるときは代入等においてはSSE2を使ったコピーがなされて欲しい。

	Bitboard& operator = (const Bitboard& rhs) { _mm_store_si128(&this->m, rhs.m); return *this; }

	Bitboard(const Bitboard& bb) { _mm_store_si128(&this->m, bb.m); }
#endif

	// --- ctor

	// 初期化しない。このとき中身は不定。
	Bitboard() {}

	// ゼロ初期化     : Bitboard x(0); // 全升が0であるBitboard
	// 全升を1で初期化 : Bitboard x(1); // 全升が1であるBitboard。ただしp[0]の63bit目は0。
	// のように用いる。
	// StockfishのBitboard(1)は最下位の1bitだけだが、将棋版のBitboard(1)は全81升。
	// 単一の升を表すときはsquare_bb(sq)を使う。
	Bitboard(const int N)
	{
		// templateではないが、最適化されるはず。
		ASSERT_LV3(N == 0 || N == 1);

		if (N == 0)
		{
#if defined (USE_SSE2)
			m = _mm_setzero_si128();
#else
			p[0] = p[1] = 0;
#endif
		}
		else if (N == 1)
		{
			// 全升が1であるBitboard
			// p[0]の63bit目は0

#if defined (USE_SSE2)
			m = _mm_set_epi64x(UINT64_C(0x000000000003FFFF), UINT64_C(0x7FFFFFFFFFFFFFFF));
#else
			p[0] = UINT64_C(0x7FFFFFFFFFFFFFFF);
			p[1] = UINT64_C(0x000000000003FFFF);
#endif
		}
	}

	// p[0],p[1]の値を直接指定して初期化する。
    // SIMDと同じメモリ配置の2ワードを初期化し、C++17のconstexprにも使えるようにする。
    constexpr Bitboard(u64 p0, u64 p1) : p{p0, p1} {}

	// sqの升が1のBitboardとして初期化する。
	Bitboard(Square sq);

	// 値を直接代入する。
	void set(u64 p0, u64 p1);

	// --- property

	// Stockfishのソースとの互換性がよくなるようにboolへの暗黙の型変換書いておく。
	operator bool() const;

	// bit test命令
	// if (lhs & rhs)とか(lhs & sq) と書くべきところを
	// if (lhs.test(rhs)) とか(lhs.test(ssq)) 書くことでSSE命令を用いて高速化する。

	bool test(Bitboard rhs) const;
	bool test(Square sq) const { return test(Bitboard(sq)); }

	// p[n]を取り出す。SSE4の命令が使えるときはそれを使う。
	template <int n> u64 extract64() const;

	// p[n]を取り出す。nがtemplate引数でないバージョン。
	u64 extract64(int n) const { return n == 0 ? extract64<0>() : extract64<1>(); }

	// p[n]に値を設定する。SSE4の命令が使えるときはそれを使う。
	template <int n> Bitboard& insert64(u64 u);

	// p[0]とp[1]をbitwise orしたものを返す。toU()相当。
	u64 merge() const { return extract64<0>() | extract64<1>(); }

	// p[0]とp[1]とで bitwise and したときに被覆しているbitがあるか。
	// merge()したあとにpext()を使うときなどに被覆していないことを前提とする場合にそのassertを書くときに使う。
	bool cross_over() const { return extract64<0>() & extract64<1>(); }

	// 指定した升(Square)が Bitboard のどちらの u64 変数の要素に属するか。
	// 本ソースコードのように縦型Bitboardにおいては、香の利きを求めるのにBitboardの
	// 片側のp[x]を調べるだけで済むので、ある升がどちらに属するかがわかれば香の利きは
	// そちらを調べるだけで良いというAperyのアイデア。
	constexpr static int part(Square sq) { return static_cast<int>(SQ_79 < sq); }

	// --- operator

	// 下位bitから1bit拾ってそのbit位置を返す。
	// 絶対に1bitはnon zeroと仮定
	// while (bb) {
	//     Square to = bb.pop_lsb();
	//     make_move(from, to);
	// }
	// のように用いる。
	Square pop_lsb();

	// 最下位の1bitの升を返す。このBitboardの値は変えない。
	Square lsb() const { u64 q0 = extract64<0>();  return (q0 != 0) ? Square(LSB64(q0)) : Square(LSB64(extract64<1>()) + 63); }

	// 1のbitを数えて返す。
	int popcount() const { return (int)(POPCNT64(extract64<0>()) + POPCNT64(extract64<1>())); }

	// 代入型演算子

#if defined (USE_SSE2)
	Bitboard& operator |= (const Bitboard& b1) { this->m = _mm_or_si128 (m, b1.m); return *this; }
	Bitboard& operator &= (const Bitboard& b1) { this->m = _mm_and_si128(m, b1.m); return *this; }
	Bitboard& operator ^= (const Bitboard& b1) { this->m = _mm_xor_si128(m, b1.m); return *this; }
	Bitboard& operator += (const Bitboard& b1) { this->m = _mm_add_epi64(m, b1.m); return *this; }
	Bitboard& operator -= (const Bitboard& b1) { this->m = _mm_sub_epi64(m, b1.m); return *this; }

	// 左シフト(縦型Bitboardでは左1回シフトで1段下の升に移動する)
	// ※　シフト演算子は歩の利きを求めるためだけに使う。
	Bitboard& operator <<= (int shift) { /*ASSERT_LV3(shift == 1);*/ m = _mm_slli_epi64(m, shift); return *this; }

	// 右シフト(縦型Bitboardでは右1回シフトで1段上の升に移動する)
	Bitboard& operator >>= (int shift) { /*ASSERT_LV3(shift == 1);*/ m = _mm_srli_epi64(m, shift); return *this; }

#else
	Bitboard& operator |= (const Bitboard& b1) { this->p[0] |= b1.p[0]; this->p[1] |= b1.p[1]; return *this; }
	Bitboard& operator &= (const Bitboard& b1) { this->p[0] &= b1.p[0]; this->p[1] &= b1.p[1]; return *this; }
	Bitboard& operator ^= (const Bitboard& b1) { this->p[0] ^= b1.p[0]; this->p[1] ^= b1.p[1]; return *this; }
	Bitboard& operator += (const Bitboard& b1) { this->p[0] += b1.p[0]; this->p[1] += b1.p[1]; return *this; }
	Bitboard& operator -= (const Bitboard& b1) { this->p[0] -= b1.p[0]; this->p[1] -= b1.p[1]; return *this; }

	Bitboard& operator <<= (int shift) { /*ASSERT_LV3(shift == 1);*/ this->p[0] <<= shift; this->p[1] <<= shift; return *this; }
	Bitboard& operator >>= (int shift) { /*ASSERT_LV3(shift == 1);*/ this->p[0] >>= shift; this->p[1] >>= shift; return *this; }

#endif

	// 比較演算子

	bool operator == (const Bitboard& rhs) const;
	bool operator != (const Bitboard& rhs) const { return !(*this == rhs); }

	// 2項演算子

	Bitboard operator & (const Bitboard& rhs) const { return Bitboard(*this) &= rhs; }
	Bitboard operator | (const Bitboard& rhs) const { return Bitboard(*this) |= rhs; }
	Bitboard operator ^ (const Bitboard& rhs) const { return Bitboard(*this) ^= rhs; }
	Bitboard operator + (const Bitboard& rhs) const { return Bitboard(*this) += rhs; }
	Bitboard operator - (const Bitboard& rhs) const { return Bitboard(*this) -= rhs; }
	Bitboard operator << (const int i) const { return Bitboard(*this) <<= i; }
	Bitboard operator >> (const int i) const { return Bitboard(*this) >>= i; }

	// 非代入型演算子

#if defined (USE_SSE2)
	// and_not演算
	// *this = (~*this) & b1;
	// ただし、notする時に、将棋盤の81升以外のところもnotされるので注意。
	// 自分自身は書き換えない。
	Bitboard andnot(const Bitboard& b1) const { Bitboard b0; b0.m = _mm_andnot_si128(m, b1.m); return b0; }
#else
	Bitboard andnot(const Bitboard& b1) const { Bitboard b0; b0.p[0] = ~p[0] & b1.p[0]; b0.p[1] = ~p[1] & b1.p[1]; return b0; }
#endif

	// 128bit全体の16byteの順序を反転したBitboardを返す。
	// 2つの64bitの位置も入れ替える。各64bit内だけのバイト反転ではない。
	// Stockfishの同名自由関数は64bitの8byteを反転するが、将棋版は16byte全体を反転する。
	// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
	Bitboard byteswap_bb() const;

	// SSE2のunpackを実行して返す。
	// hi_out = _mm_unpackhi_epi64(lo_in,hi_in);
	// lo_out = _mm_unpacklo_epi64(lo_in,hi_in);
	static void unpack(const Bitboard hi_in,const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out);

	// 2組のBitboardを、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
	// 128bit整数とみなして1引き算したBitboardを返す。
	static void decrement(const Bitboard hi_in,const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out);

	// このbitboardを128bitレジスタとみなして1減算したBitboardを返す。方向利きの計算で用いる。
	Bitboard decrement() const;

	// 2bit以上あるかどうかを判定する。
	// Stockfish版は任意のBitboardに使えるが、将棋版は!cross_over()が前提。
	// 2つの64bitをORして数えるため、同じbit位置が両方に立つ盤面には使えない。
	bool more_than_one() const;

	// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
	Square operator*() { return pop_lsb(); }
	void operator++() {}

	// T(Square sq)と呼び出されるので各升に対して処理ができる。
	// コードは展開されるのでわりと大きくなるから注意。
	// 使用例) target.foreach([&](Square to) { mlist++->move = make_move(from,to) + OurPt(Us,Pt); })
	template <typename T> FORCE_INLINE void foreach(T t) const
	{
		u64 p0 = this->extract64<0>();
		while (p0) t( (Square)YaneuraOu::pop_lsb(p0));

		u64 p1 = this->extract64<1>();
		while (p1) t( (Square)(YaneuraOu::pop_lsb(p1) + 63));
	}

	// Bitboard bbに対して、1であるbitのSquareがsqに入ってきて、このときにTを呼び出す。
	// bb.p[0]のbitに対してはT(sq,0)と呼び出す。bb.p[1]のbitに対してはT(sq,1)と呼び出す。
	// 使用例) bb.foreach_part([&](Square sq, int part){ ... } );
	// bbの内容は破壊しない。
	// コードは展開されるのでわりと大きくなるから注意。
	template <typename T> FORCE_INLINE void foreach_part(T t) const
	{
		u64 p0 = this->extract64<0>();
		while (p0) { t( (Square)(YaneuraOu::pop_lsb(p0)     ),0); }

		u64 p1 = this->extract64<1>();
		while (p1) { t( (Square)(YaneuraOu::pop_lsb(p1) + 63),1); }
	}

	// UnitTest
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);
};

// 抑制していた警告を元に戻す。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <int n>
inline u64 Bitboard::extract64() const
{
	static_assert(n == 0 || n == 1, "");
	return p[n];
}

template <int n>
inline Bitboard& Bitboard::insert64(u64 u)
{
	static_assert(n == 0 || n == 1, "");
	p[n] = u;
	return *this;
}

// --------------------
//     Bitboard定数
// --------------------

namespace BB_Table
{
	// 各筋を表現するBitboard定数
	inline constexpr Bitboard File1BB = Bitboard(UINT64_C(0x1ff) << (9 * 0), 0);
	inline constexpr Bitboard File2BB = Bitboard(UINT64_C(0x1ff) << (9 * 1), 0);
	inline constexpr Bitboard File3BB = Bitboard(UINT64_C(0x1ff) << (9 * 2), 0);
	inline constexpr Bitboard File4BB = Bitboard(UINT64_C(0x1ff) << (9 * 3), 0);
	inline constexpr Bitboard File5BB = Bitboard(UINT64_C(0x1ff) << (9 * 4), 0);
	inline constexpr Bitboard File6BB = Bitboard(UINT64_C(0x1ff) << (9 * 5), 0);
	inline constexpr Bitboard File7BB = Bitboard(UINT64_C(0x1ff) << (9 * 6), 0);
	inline constexpr Bitboard File8BB = Bitboard(0, 0x1ff << (9 * 0));
	inline constexpr Bitboard File9BB = Bitboard(0, 0x1ff << (9 * 1));

	// 各段を表現するBitboard定数
	inline constexpr Bitboard Rank1BB = Bitboard(UINT64_C(0x40201008040201) << 0, 0x201 << 0);
	inline constexpr Bitboard Rank2BB = Bitboard(UINT64_C(0x40201008040201) << 1, 0x201 << 1);
	inline constexpr Bitboard Rank3BB = Bitboard(UINT64_C(0x40201008040201) << 2, 0x201 << 2);
	inline constexpr Bitboard Rank4BB = Bitboard(UINT64_C(0x40201008040201) << 3, 0x201 << 3);
	inline constexpr Bitboard Rank5BB = Bitboard(UINT64_C(0x40201008040201) << 4, 0x201 << 4);
	inline constexpr Bitboard Rank6BB = Bitboard(UINT64_C(0x40201008040201) << 5, 0x201 << 5);
	inline constexpr Bitboard Rank7BB = Bitboard(UINT64_C(0x40201008040201) << 6, 0x201 << 6);
	inline constexpr Bitboard Rank8BB = Bitboard(UINT64_C(0x40201008040201) << 7, 0x201 << 7);
	inline constexpr Bitboard Rank9BB = Bitboard(UINT64_C(0x40201008040201) << 8, 0x201 << 8);

	// 各筋を表現するBitboard配列
	extern const Bitboard FILE_BB[FILE_NB];

	// 各段を表現するBitboard配列
	extern const Bitboard RANK_BB[RANK_NB];

	// sqの升だけが1であるBitboard。SQ_NBの要素は空。
	extern Bitboard SquareBB[SQ_NB_PLUS1];
}
#endif

#if STOCKFISH
constexpr Bitboard square_bb(Square s) {
    assert(is_ok(s));
    return 1ULL << s;
}
#else
// Stockfish版は盤上の升だけを受け付けるが、将棋版はSQ_NBでも空のBitboardを返す。
inline Bitboard square_bb(Square s) { return BB_Table::SquareBB[s]; }
#endif

// Overloads of bitwise operators between a Bitboard and a Square for testing
// whether a given bit is set in a bitboard, and for setting and clearing bits.
// 指定した升のビットの検査・設定・反転に使う、BitboardとSquare間のビット演算子。
#if STOCKFISH
constexpr Bitboard  operator&(Bitboard b, Square s) { return b & square_bb(s); }
constexpr Bitboard  operator|(Bitboard b, Square s) { return b | square_bb(s); }
constexpr Bitboard  operator^(Bitboard b, Square s) { return b ^ square_bb(s); }
constexpr Bitboard& operator|=(Bitboard& b, Square s) { return b |= square_bb(s); }
constexpr Bitboard& operator^=(Bitboard& b, Square s) { return b ^= square_bb(s); }

constexpr Bitboard operator&(Square s, Bitboard b) { return b & s; }
constexpr Bitboard operator|(Square s, Bitboard b) { return b | s; }
constexpr Bitboard operator^(Square s, Bitboard b) { return b ^ s; }

constexpr Bitboard operator|(Square s1, Square s2) { return square_bb(s1) | s2; }

constexpr bool more_than_one(Bitboard b) { return b & (b - 1); }
#else
Bitboard operator&(const Bitboard& b, Square s);
Bitboard operator|(const Bitboard& b, Square s);
Bitboard operator^(const Bitboard& b, Square s);
#endif

#if !STOCKFISH
// 単項演算子
// →　NOTで書くと、使っていないbit(p[0]のbit63)がおかしくなるのでBitboard(1)でxorする。
// Stockfishの整数型の~と異なり、将棋版は81升のbitだけを反転し、盤外bitは変更しない。
Bitboard operator ~ (const Bitboard& a);

// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
const Bitboard begin(const Bitboard& b);
const Bitboard end(const Bitboard&);

// Bitboardの1の升を'*'、0の升を'.'として表示する。デバッグ用。
std::ostream& operator<<(std::ostream& os, const Bitboard& board);
#endif

// rank_bb() and file_bb() return a bitboard representing all the squares on
// the given file or rank.
// 指定した筋または段にある全升を表すBitboardを返す。
#if STOCKFISH
constexpr Bitboard rank_bb(Rank r) { return Rank1BB << (8 * r); }

constexpr Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }

constexpr Bitboard file_bb(File f) { return FileABB << f; }

constexpr Bitboard file_bb(Square s) { return file_bb(file_of(s)); }
#else
inline Bitboard rank_bb(Rank r) { return BB_Table::RANK_BB[r]; }

inline Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }

inline Bitboard file_bb(File f) { return BB_Table::FILE_BB[f]; }

inline Bitboard file_bb(Square s) { return file_bb(file_of(s)); }
#endif

#if STOCKFISH
// Moves a bitboard one or two steps as specified by dir
// 指定した方向へBitboardを1升または2升移動する。
inline constexpr Bitboard shift(Bitboard b, Direction dir) {
    return dir == NORTH         ? b << 8
         : dir == SOUTH         ? b >> 8
         : dir == NORTH + NORTH ? b << 16
         : dir == SOUTH + SOUTH ? b >> 16
         : dir == EAST          ? (b & ~FileHBB) << 1
         : dir == WEST          ? (b & ~FileABB) >> 1
         : dir == NORTH_EAST    ? (b & ~FileHBB) << 9
         : dir == NORTH_WEST    ? (b & ~FileABB) << 7
         : dir == SOUTH_EAST    ? (b & ~FileHBB) >> 7
         : dir == SOUTH_WEST    ? (b & ~FileABB) >> 9
                                : 0;
}
#else
// ForwardRanksBBの定義)
//    c側の香の利き = 飛車の利き & ForwardRanksBB[c][rank_of(sq)]
//
// すなわち、
// color == BLACKのとき、n段目よりWHITE側(1からn-1段目)を表現するBitboard。
// color == WHITEのとき、n段目よりBLACK側(n+1から9段目)を表現するBitboard。
// このアイデアはAperyのもの。
namespace BB_Table { extern const Bitboard ForwardRanksBB[COLOR_NB][RANK_NB]; }

// 先手から見て1段目からr段目までを表現するBB(US==WHITEなら、9段目から数える)
static const Bitboard rank1_n_bb(Color US, const Rank r)
{
	ASSERT_LV2(is_ok(r));
	return BB_Table::ForwardRanksBB[US][(US == BLACK ? r + 1 : 7 - r)];
}

// 敵陣を表現するBitboard。
namespace BB_Table { extern const Bitboard EnemyField[COLOR_NB]; }
static const Bitboard enemy_field(Color Us) { return BB_Table::EnemyField[Us]; }

// 歩が打てる筋が1になっているBitboardを返す。
// pawns : いま歩を打とうとしている手番側の歩のBitboard
//
// ここで返すBitboardは、
// C == BLACKの時は、1段目は0(歩が打てないから)、
// C == WHITEの時は、9段目は0(歩が打てないから)を保証する。
template <Color C>
static Bitboard pawn_drop_mask(const Bitboard& pawns) {
	// Quigy[WCSC31]の手法 : cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal.pdf
	// 先後で番兵以外を共通化することでさらに命令数を減らすことができる。
	// 先手は仮想的な十段目を番兵とする。
	const Bitboard s = C == BLACK ? BB_Table::Rank9BB << 1 : BB_Table::Rank9BB;
	// 歩の立っていない筋では番兵が立つ。
	const Bitboard t = s - pawns & s;
	// 先手だと0b1000000000 - 0b10 = 0b111111110で1段目、後手は0b100000000 - 0b000000001 = 0b011111111で9段目が空く。
	return t - (t >> 8);
}


#endif

// Returns the squares attacked by pawns of the given color
// from the squares in the given bitboard.
// 指定した手番の歩が、渡されたBitboardの各升から利かせる升を返す。
#if STOCKFISH
template<Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? shift(b, NORTH_WEST) | shift(b, NORTH_EAST)
                      : shift(b, SOUTH_WEST) | shift(b, SOUTH_EAST);
}
#else
// Stockfishの同名関数はチェスの斜め2方向。将棋版は縦型Bitboard上で前方1升だけを返す。
// 端の段からのシフトを盤面座標で補正する関数ではないので、歩を置ける升を渡す。
// color = BLACKのとき、51の歩は49の升に移動するので、注意すること。
// (51の升にいる先手の歩は存在しないので、歩の移動に用いる分には問題ないが。)
template <Color C>
inline Bitboard pawn_attacks_bb(Bitboard bb)
{
	// Apery型の縦型Bitboardにおいては歩の利きはbit shiftで済む。
	ASSERT_LV3(is_ok(C));
	return
		C == BLACK ? (bb >> 1) :
		C == WHITE ? (bb << 1) :
		Bitboard(ZERO);
}

// ↑の非template版。Colorを受け取るこのオーバーロードは将棋独自。
inline Bitboard pawn_attacks_bb(Color c, const Bitboard& bb)
{
	return (c == BLACK) ? pawn_attacks_bb<BLACK>(bb) : pawn_attacks_bb<WHITE>(bb);
}
#endif

#if STOCKFISH
constexpr Bitboard pawn_single_push_bb(Color c, Bitboard b) {
    return shift(b, c == WHITE ? NORTH : SOUTH);
}

inline constexpr auto PawnPairBB = []() {
    std::array<Bitboard, SQUARE_NB> result{};
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        Bitboard file  = file_bb(s);
        Bitboard files = file | shift(file, EAST) | shift(file, WEST);
        result[s]      = files & ~(Rank1BB | Rank8BB) & ~square_bb(s);
    }
    return result;
}();

// Returns the squares that can host a pawn forming a "pawn pair" with a pawn
// on s: own file plus adjacent files, restricted to ranks 2-7, excluding s.
// The geometry is color-independent.
// sの歩と「pawn pair」を作れる升。自分の筋と隣接筋の2～7段目に限定し、s自身を除く。
// この配置条件は先後に依存しない。
constexpr Bitboard pawn_pair_bb(Square s) { return PawnPairBB[s]; }

constexpr int edge_distance(File f) { return std::min(f, File(FILE_H - f)); }

template<typename T>
constexpr int constexpr_popcount(T v) {
    static_assert(std::is_integral_v<T>, "constexpr_popcount is undefined for non-integral types");

    if constexpr (sizeof(T) <= 8)
    {
        u64 b = static_cast<std::make_unsigned_t<T>>(v);

        b = b - ((b >> 1) & 0x5555555555555555ULL);
        b = (b & 0x3333333333333333ULL) + ((b >> 2) & 0x3333333333333333ULL);
        b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0FULL;

        return static_cast<int>((b * 0x0101010101010101ULL) >> 56);
    }
    else
    {
        int result = 0;

        for (; v; v >>= static_cast<T>(1))
            if (v & static_cast<T>(1))
                ++result;

        return result;
    }
}

// Counts the number of non-zero bits in a bitboard.
// Bitboard内で1になっているビットの数を返す。
inline int popcount(Bitboard b) {

#ifdef _MSC_VER

    return int(_mm_popcnt_u64(b));

#else  // Assumed gcc or compatible compiler
    // GCCまたは互換コンパイラーを想定する。

    return __builtin_popcountll(b);

#endif
}

// Returns the least significant bit in a non-zero bitboard.
// 非ゼロのBitboardで最下位のビットに対応する升を返す。
inline Square lsb(Bitboard b) {
    assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return Square(__builtin_ctzll(b));

#elif defined(_MSC_VER)
    #ifdef _WIN64  // MSVC, WIN64

    unsigned long idx;
    _BitScanForward64(&idx, b);
    return Square(idx);

    #else  // MSVC, WIN32
    unsigned long idx;

    if (b & 0xffffffff)
    {
        _BitScanForward(&idx, i32(b));
        return Square(idx);
    }

    _BitScanForward(&idx, i32(b >> 32));
    return Square(idx + 32);
    #endif
#else  // Compiler is neither GCC nor MSVC compatible
    // GCC互換でもMSVC互換でもないコンパイラー。
    #error "Compiler not supported."
#endif
}

// Returns the most significant bit in a non-zero bitboard.
// 非ゼロのBitboardで最上位のビットに対応する升を返す。
inline Square msb(Bitboard b) {
    assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return Square(63 ^ __builtin_clzll(b));

#elif defined(_MSC_VER)
    #ifdef _WIN64  // MSVC, WIN64

    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return Square(idx);

    #else  // MSVC, WIN32

    unsigned long idx;

    if (b >> 32)
    {
        _BitScanReverse(&idx, i32(b >> 32));
        return Square(idx + 32);
    }

    _BitScanReverse(&idx, i32(b));
    return Square(idx);
    #endif
#else  // Compiler is neither GCC nor MSVC compatible
    // GCC互換でもMSVC互換でもないコンパイラー。
    #error "Compiler not supported."
#endif
}
#else
// Returns the least significant bit in a non-zero bitboard.
// 0でないビットボードにおける最下位ビットを返す。
// 📝 Stockfishとの互換性のために用意
// Stockfishのlsb(Bitboard)は盤上のSquareを返すが、この関数はu64の生のbit番号0～63を返す。
// 128bit盤面の升を得るときはBitboard::lsb()を使う。TT等の生整数用処理と混同しないこと。
inline i8 lsb(u64 b) {
	assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return i8(__builtin_ctzll(b));

#elif defined(_MSC_VER)
    #ifdef _WIN64  // MSVC, WIN64

    unsigned long idx;
    _BitScanForward64(&idx, b);
    return i8(idx);

    #else  // MSVC, WIN32
    unsigned long idx;

    if (b & 0xffffffff)
    {
        _BitScanForward(&idx, i32(b));
        return i8(idx);
    }

    _BitScanForward(&idx, i32(b >> 32));
    return i8(idx + 32);
    #endif
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}

// Returns the most significant bit in a non-zero bitboard.
// 0でないビットボードにおける最上位ビットを返す。
// 📝 Stockfishとの互換性のために用意
// Stockfishのmsb(Bitboard)は盤上のSquareを返すが、この関数はu64の生のbit番号0～63を返す。
// 128bitの将棋盤全体を調べる関数ではない。
inline i8 msb(u64 b) {
    assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return i8(63 ^ __builtin_clzll(b));

#elif defined(_MSC_VER)
    #ifdef _WIN64  // MSVC, WIN64

    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return i8(idx);

    #else  // MSVC, WIN32

    unsigned long idx;

    if (b >> 32)
    {
        _BitScanReverse(&idx, i32(b >> 32));
        return i8(idx + 32);
    }

    _BitScanReverse(&idx, i32(b));
    return i8(idx);
    #endif
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}
#endif

// Returns the bitboard of the least significant
// square of a non-zero bitboard. It is equivalent to square_bb(lsb(bb)).
// 0でないBitboardの最下位の升だけを立てたBitboardを返す。
// Stockfishではsquare_bb(lsb(bb))と等価で、将棋側では2つの64bit領域を考慮する。
// ※　Stockfishとの互換性のために用意。

#if STOCKFISH
constexpr Bitboard least_significant_square_bb(Bitboard b) {
    assert(b);
    return b & -b;
}
#else
// Stockfish版は非ゼロを要求するが、将棋版は空のBitboardを渡しても空を返す。
inline Bitboard least_significant_square_bb(Bitboard b) {
	//return b & -b;
	// →　Stockfishはチェスが64升だから盤面のBitboardが64bit整数に収まるのだが、
	//    やねうら王では一工夫必要。

	u64 q0 = b.extract64<0>();
	u64 q1 = b.extract64<1>();
	return (q0 != 0) ? Bitboard(q0 & -q0 , 0) : Bitboard(0, q1 & -q1);
}
#endif

#if STOCKFISH
// Finds and clears the least significant bit in a non-zero bitboard.
// 非ゼロのBitboardの最下位ビットに対応する升を返し、そのビットを消す。
inline Square pop_lsb(Bitboard& b) {
    assert(b);
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}
#endif

} // namespace YaneuraOu

#endif  // #ifndef BITBOARD_H_INCLUDED
