#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <string>

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

#if !STOCKFISH
	// Bitboard関連のテーブル初期化のための関数
	void init();
#endif
std::string pretty(Bitboard b);

} // namespace YaneuraOu::Bitboards

// --------------------
//     Bitboard
// --------------------

// Bitboardをゼロクリアするコンストラクタに指定する引数
// 例) Bitboard(ZERO) のように指定するとゼロクリアされたBitboardが出来上がる。
enum BitboardZero{ ZERO };

// Bitboardクラスは、コンストラクタでの初期化が保証できないので(オーバーヘッドがあるのでやりたくないので)
// GCC 7.1.0以降で警告が出るのを回避できない。ゆえに、このクラスではこの警告を抑制する。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// Bitboard本体クラス

struct alignas(16) Bitboard
{
#if defined (USE_SSE2)

	union
	{
		// 64bitずつとして扱うとき用
		// SSE4.1以上なら、このメンバを用いずに 変数m の方を用いて、一貫して128bitレジスタとして扱ったほうが良いと思う。

		u64 p[2];

		// SSEで取り扱い時用
		// bit0がSQ_11,bit1がSQ_12,…,bit81がSQ_99を表現する。(縦型Bitboard)
		// このbit位置がSquare型と対応する。
		//
		// ただしbit63は未使用。これは、ここを余らせることで香の利きや歩の打てる場所を求めやすくする。
		// Aperyを始めとするmagic bitboard派によって考案された。

		// ここから上位/下位64bitを取り出すのは、メンバのextract()を使うべし。

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
	// ALL_BBで初期化 : Bitboard x(1); // 全升が1であるBitboard。ただしp[0]の63bit目は0。
	// のように用いる。
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

	// p[0],p[1]の値を直接指定しての初期化。(Bitboard定数の初期化のときのみ用いる)
	Bitboard(u64 p0, u64 p1);

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
	// while(to = bb.pop())
	//  make_move(from,to);
	// のように用いる。
	Square pop();

	// このBitboardの値を変えないpop()
	Square pop_c() const { u64 q0 = extract64<0>();  return (q0 != 0) ? Square(LSB64(q0)) : Square(LSB64(extract64<1>()) + 63); }

	// 1のbitを数えて返す。
	int pop_count() const { return (int)(POPCNT64(extract64<0>()) + POPCNT64(extract64<1>())); }

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

	// byte単位で入れ替えたBitboardを返す。
	// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
	Bitboard byte_reverse() const;

	// SSE2のunpackを実行して返す。
	// hi_out = _mm_unpackhi_epi64(lo_in,hi_in);
	// lo_out = _mm_unpacklo_epi64(lo_in,hi_in);
	static void unpack(const Bitboard hi_in,const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out);

	// 2組のBitboardを、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
	// 128bit整数とみなして1引き算したBitboardを返す。
	static void decrement(const Bitboard hi_in,const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out);

	// このbitboardを128bitレジスタとみなして1減算したBitboardを返す。方向利きの計算で用いる。
	Bitboard decrement() const;

	// 2bit以上あるかどうかを判定する。縦横斜め方向に並んだ駒が2枚以上であるかを判定する。この関係にないと駄目。
	// この関係にある場合、Bitboard::merge()によって被覆しないことがBitboardのレイアウトから保証されている。
	bool more_than_one() const;

	// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
	Square operator*() { return pop(); }
	void operator++() {}

	// T(Square sq)と呼び出されるので各升に対して処理ができる。
	// コードは展開されるのでわりと大きくなるから注意。
	// 使用例) target.foreach([&](Square to) { mlist++->move = make_move(from,to) + OurPt(Us,Pt); })
	template <typename T> FORCE_INLINE void foreach(T t) const
	{
		u64 p0 = this->extract64<0>();
		while (p0) t( (Square)pop_lsb(p0));

		u64 p1 = this->extract64<1>();
		while (p1) t( (Square)(pop_lsb(p1) + 63));
	}

	// Bitboard bbに対して、1であるbitのSquareがsqに入ってきて、このときにTを呼び出す。
	// bb.p[0]のbitに対してはT(sq,0)と呼び出す。bb.p[1]のbitに対してはT(sq,1)と呼び出す。
	// 使用例) bb.foreach_part([&](Square sq, int part){ ... } );
	// bbの内容は破壊しない。
	// コードは展開されるのでわりと大きくなるから注意。
	template <typename T> FORCE_INLINE void foreach_part(T t) const
	{
		u64 p0 = this->extract64<0>();
		while (p0) { t( (Square)(pop_lsb(p0)     ),0); }

		u64 p1 = this->extract64<1>();
		while (p1) { t( (Square)(pop_lsb(p1) + 63),1); }
	}

	// UnitTest
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);
};

// 抑制していた警告を元に戻す。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
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

// Overloads of bitwise operators between a Bitboard and a Square for testing
// whether a given bit is set in a bitboard, and for setting and clearing bits.
// 指定した升のビットの検査・設定・反転に使う、BitboardとSquare間のビット演算子。
Bitboard operator|(const Bitboard& b, Square s);
Bitboard operator&(const Bitboard& b, Square s);
Bitboard operator^(const Bitboard& b, Square s);

// 単項演算子
// →　NOTで書くと、使っていないbit(p[0]のbit63)がおかしくなるのでALL_BBでxorしないといけない。
Bitboard operator ~ (const Bitboard& a);

// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
const Bitboard begin(const Bitboard& b);
const Bitboard end(const Bitboard&);

// Bitboardの1の升を'*'、0の升を'.'として表示する。デバッグ用。
std::ostream& operator<<(std::ostream& os, const Bitboard& board);

// --------------------
//     Bitboard定数
// --------------------

namespace BB_Table
{
	// 各筋を表現するBitboard定数
	extern const Bitboard FILE1_BB;
	extern const Bitboard FILE2_BB;
	extern const Bitboard FILE3_BB;
	extern const Bitboard FILE4_BB;
	extern const Bitboard FILE5_BB;
	extern const Bitboard FILE6_BB;
	extern const Bitboard FILE7_BB;
	extern const Bitboard FILE8_BB;
	extern const Bitboard FILE9_BB;

	// 各段を表現するBitboard定数
	extern const Bitboard RANK1_BB;
	extern const Bitboard RANK2_BB;
	extern const Bitboard RANK3_BB;
	extern const Bitboard RANK4_BB;
	extern const Bitboard RANK5_BB;
	extern const Bitboard RANK6_BB;
	extern const Bitboard RANK7_BB;
	extern const Bitboard RANK8_BB;
	extern const Bitboard RANK9_BB;

	// 各筋を表現するBitboard配列
	extern const Bitboard FILE_BB[FILE_NB];

	// 各段を表現するBitboard配列
	extern const Bitboard RANK_BB[RANK_NB];

	// 全升が0であるBitboard
	//extern Bitboard ZERO_BB;
	// 廃止 →　代わりにBitboard<0>()を用いた方が、メモリ参照がなくて速いのでこの定数は廃止。

	// 全升が1であるBitboard
	// p[0]の63bit目は0
	//extern Bitboard ALL_BB;
	// 廃止 →　代わりにBitboard<1>()を用いる。	
}

// rank_bb() and file_bb() return a bitboard representing all the squares on
// the given file or rank.
// 指定した筋または段にある全升を表すBitboardを返す。
static const Bitboard file_bb(File f) { return BB_Table::FILE_BB[f]; }
static const Bitboard rank_bb(Rank r) { return BB_Table::RANK_BB[r]; }

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

	const Bitboard left(0x4020100804020100ULL, 0x0000000000020100ULL);

	// 9段目だけ1にしたbitboardから、歩の升を引き算すると、桁借りで上位bit(9段目)が0になる。これを敷衍するという考えかた。
	Bitboard t = left - pawns;

#if 0
	// 8回シフトで1段目に移動する。
	t = (t & left) >> 8;

	// tを 9段目が1になっているleftが引き算すると、
	// 1) 1段目が1の時、9段目が0、1段目から8段目が1のBitboardになる。
	// 2) 1段目が0の時、9段目が1、1段目から8段目は0のBitboardになる。
	// ここにleftとxorすると、
	// 1') 1段目が1の時、9段目が1、1段目から8段目が1のBitboardになる。= 1段目から9段目が1。
	// 2') 1段目が0の時、9段目が0、1段目から8段目は0のBitboardになる。= 1段目から9段目が0。
	// というように、tの内容が敷衍された。

	return left ^ (left - t);
#endif
	// ↑これだと先後に関わらず1段目から9段目が1になる。
	// 先手の時は1段目を1にしたくないし、後手の時は9段目を1にしたくない。
	// そこで少し調整する必要がある。

	if (C == BLACK)
	{
		// 2段目に移動させて、それを9段目まで敷衍させる。
		t = (t & left) >> 7;
		return left ^ (left - t);
	}
	else {
		// 1段目に移動させて、それを8段目まで敷衍させる。
		t = (t & left) >> 8;
		return left.andnot(left - t);
	}
}


// 歩を複数配置したBitboardに対して、その歩の利きのBitboardを返す。
// color = BLACKのとき、51の歩は49の升に移動するので、注意すること。
// (51の升にいる先手の歩は存在しないので、歩の移動に用いる分には問題ないが。)
template <Color C>
inline Bitboard pawnBbEffect(const Bitboard& bb)
{
	// Apery型の縦型Bitboardにおいては歩の利きはbit shiftで済む。
	ASSERT_LV3(is_ok(C));
	return
		C == BLACK ? (bb >> 1) :
		C == WHITE ? (bb << 1) :
		Bitboard(ZERO);
}

// ↑の非template版
inline Bitboard pawnBbEffect(Color c, const Bitboard& bb)
{
	return (c == BLACK) ? pawnBbEffect<BLACK>(bb) : pawnBbEffect<WHITE>(bb);
}

// Returns the squares attacked by pawns of the given color
// from the squares in the given bitboard.
// 指定した手番の歩が、渡されたBitboardの各升から利かせる升を返す。

// pawn_attacks_bb()は、手番C側の利き。
// b : 手番C側の歩のBitboardを渡す。

template<Color C>
inline Bitboard pawn_attacks_bb(Bitboard b)
{
	return pawnBbEffect<C>(b);
}


// Returns the least significant bit in a non-zero bitboard.
// 0でないビットボードにおける最下位ビットを返す。
// 📝 Stockfishとの互換性のために用意
#if STOCKFISH
inline Square lsb(Bitboard b) {
#else
inline i8 lsb(u64 b) {
#endif
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
#if STOCKFISH
inline Square msb(Bitboard b) {
#else
inline i8 msb(u64 b) {
#endif
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


// Returns the bitboard of the least significant
// square of a non-zero bitboard. It is equivalent to square_bb(lsb(bb)).
// 0でないBitboardの最下位の升だけを立てたBitboardを返す。
// Stockfishではsquare_bb(lsb(bb))と等価で、将棋側では2つの64bit領域を考慮する。
// ※　Stockfishとの互換性のために用意。

inline Bitboard least_significant_square_bb(Bitboard b) {
	//return b & -b;
	// →　Stockfishはチェスが64升だから盤面のBitboardが64bit整数に収まるのだが、
	//    やねうら王では一工夫必要。

	u64 q0 = b.extract64<0>();
	u64 q1 = b.extract64<1>();
	return (q0 != 0) ? Bitboard(q0 & -q0 , 0) : Bitboard(0, q1 & -q1);
}

} // namespace YaneuraOu

#endif // #ifndef BITBOARD_H_INCLUDED
