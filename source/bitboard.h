#ifndef _BITBOARD_H_
#define _BITBOARD_H_

#include "types.h"

namespace Bitboards
{
	// Bitboard関連のテーブル初期化のための関数
	extern void init();
}

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
	static void UnitTest(Test::UnitTester&);
};

// 抑制していた警告を元に戻す。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

// p[n]を取り出す。SSE4の命令が使えるときはそれを使う。
template <int n>
inline u64 Bitboard::extract64() const
{
	static_assert(n == 0 || n == 1, "");
#if defined(USE_SSE41)
	return (u64)(_mm_extract_epi64(m, n));
#else
	return p[n];
#endif
}

template <int n>
inline Bitboard& Bitboard::insert64(u64 u)
{
	static_assert(n == 0 || n == 1, "");
#if defined(USE_SSE41)
	m = _mm_insert_epi64(m, u, n);
#else
	p[n] = u;
#endif
	return *this;
}

// Square型との演算子
extern Bitboard operator|(const Bitboard& b, Square s);
extern Bitboard operator&(const Bitboard& b, Square s);
extern Bitboard operator^(const Bitboard& b, Square s);

// 単項演算子
// →　NOTで書くと、使っていないbit(p[0]のbit63)がおかしくなるのでALL_BBでxorしないといけない。
extern Bitboard operator ~ (const Bitboard& a);

// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
extern const Bitboard begin(const Bitboard& b);
extern const Bitboard end(const Bitboard&);

// Bitboardの1の升を'*'、0の升を'.'として表示する。デバッグ用。
extern std::ostream& operator<<(std::ostream& os, const Bitboard& board);

// --------------------
//     Bitboard256
// --------------------

// Bitboard 2つを256bit registerで扱う。
// これをうまく用いると飛車、角の利きがmagic bitboardなしで求められる。
// Qugiy[WCSC31]のアイデアの応用。

struct alignas(32) Bitboard256
{
	// Bitboard 2つ分。
#if defined (USE_AVX2)
	union
	{
		// 64bitずつとして扱うとき用
		u64 p[4];

		__m256i m;
	};
#else // no SSE
	u64 p[4];
#endif

	Bitboard256() {}
#if defined (USE_AVX2)
	Bitboard256& operator = (const Bitboard256& rhs) { _mm256_store_si256(&this->m, rhs.m); return *this; }
	Bitboard256(const Bitboard256& bb) { _mm256_store_si256(&this->m, bb.m); }

	// 同じBitboardを2つに複製し、それをBitboard256とする。
	Bitboard256(const Bitboard& b1) { m = _mm256_broadcastsi128_si256(b1.m); }

	// 2つのBitboardを合わせたBitboard256を作る。
	Bitboard256(const Bitboard& b1, const Bitboard& b2) {
		// m = _mm256_set_epi64x(b2.p[1],b2.p[0],b1.p[1],b1.p[0]);
		m = _mm256_castsi128_si256(b1.m);        // 256bitにcast(上位は0)。これはcompiler向けの命令。
		m = _mm256_inserti128_si256(m, b2.m, 1); // 上位128bitにb2.mを代入
	}
#else
	Bitboard256(const Bitboard& b1, const Bitboard& b2) { p[0] = b1.p[0]; p[1] = b1.p[1]; p[2] = b2.p[0]; p[3]=b2.p[1]; }
	Bitboard256(const Bitboard& b1) { p[0] = p[2] = b1.p[0]; p[1] = p[3] = b1.p[1]; }
#endif

#if defined (USE_AVX2)
	Bitboard256& operator |= (const Bitboard256& b1) { this->m = _mm256_or_si256( m, b1.m); return *this; }
	Bitboard256& operator &= (const Bitboard256& b1) { this->m = _mm256_and_si256(m, b1.m); return *this; }
	Bitboard256& operator ^= (const Bitboard256& b1) { this->m = _mm256_xor_si256(m, b1.m); return *this; }
	Bitboard256& operator += (const Bitboard256& b1) { this->m = _mm256_add_epi64(m, b1.m); return *this; }
	Bitboard256& operator -= (const Bitboard256& b1) { this->m = _mm256_sub_epi64(m, b1.m); return *this; }

	// 左シフト(縦型Bitboardでは左1回シフトで1段下の升に移動する)
	// ※　シフト演算子は歩の利きを求めるためだけに使う。
	Bitboard256& operator <<= (int shift) { /*ASSERT_LV3(shift == 1);*/ m = _mm256_slli_epi64(m, shift); return *this; }

	// 右シフト(縦型Bitboardでは右1回シフトで1段上の升に移動する)
	Bitboard256& operator >>= (int shift) { /*ASSERT_LV3(shift == 1);*/ m = _mm256_srli_epi64(m, shift); return *this; }

	// and_not演算
	// *this = (~*this) & b1;
	// ただし、notする時に、将棋盤の81升以外のところもnotされるので注意。
	// 自分自身は書き換えない。
	Bitboard256 andnot(const Bitboard256& b1) const { Bitboard256 b0; b0.m = _mm256_andnot_si256(m, b1.m); return b0; }

#else
	Bitboard256& operator |= (const Bitboard256& b1) { this->p[0] |= b1.p[0]; this->p[1] |= b1.p[1]; this->p[2] |= b1.p[2]; this->p[3] |= b1.p[3]; return *this; }
	Bitboard256& operator &= (const Bitboard256& b1) { this->p[0] &= b1.p[0]; this->p[1] &= b1.p[1]; this->p[2] &= b1.p[2]; this->p[3] &= b1.p[3]; return *this; }
	Bitboard256& operator ^= (const Bitboard256& b1) { this->p[0] ^= b1.p[0]; this->p[1] ^= b1.p[1]; this->p[2] ^= b1.p[2]; this->p[3] ^= b1.p[3]; return *this; }
	Bitboard256& operator += (const Bitboard256& b1) { this->p[0] += b1.p[0]; this->p[1] += b1.p[1]; this->p[2] += b1.p[2]; this->p[3] += b1.p[3]; return *this; }
	Bitboard256& operator -= (const Bitboard256& b1) { this->p[0] -= b1.p[0]; this->p[1] -= b1.p[1]; this->p[2] -= b1.p[2]; this->p[3] -= b1.p[3]; return *this; }

	Bitboard256& operator <<= (int shift) { /*ASSERT_LV3(shift == 1);*/ this->p[0] <<= shift; this->p[1] <<= shift; this->p[2] <<= shift; this->p[3] <<= shift; return *this; }
	Bitboard256& operator >>= (int shift) { /*ASSERT_LV3(shift == 1);*/ this->p[0] >>= shift; this->p[1] >>= shift; this->p[2] >>= shift; this->p[3] >>= shift; return *this; }

	Bitboard256 andnot(const Bitboard256& b1) const { Bitboard256 b0; b0.p[0] = ~p[0] & b1.p[0]; b0.p[1] = ~p[1] & b1.p[1]; b0.p[2] = ~p[2] & b1.p[2]; b0.p[3] = ~p[3] & b1.p[3]; return b0; }

#endif

	// 比較演算子

	bool operator == (const Bitboard256& rhs) const;
	bool operator != (const Bitboard256& rhs) const { return !(*this == rhs); }

	// 2項演算子

	Bitboard256 operator & (const Bitboard256& rhs) const { return Bitboard256(*this) &= rhs; }
	Bitboard256 operator | (const Bitboard256& rhs) const { return Bitboard256(*this) |= rhs; }
	Bitboard256 operator ^ (const Bitboard256& rhs) const { return Bitboard256(*this) ^= rhs; }
	Bitboard256 operator + (const Bitboard256& rhs) const { return Bitboard256(*this) += rhs; }
	Bitboard256 operator - (const Bitboard256& rhs) const { return Bitboard256(*this) -= rhs; }
	Bitboard256 operator << (const int i) const { return Bitboard256(*this) <<= i; }
	Bitboard256 operator >> (const int i) const { return Bitboard256(*this) >>= i; }

	// その他の操作

	// このBitboard256をBitboard2つに分離する。(デバッグ用)
	void toBitboard(Bitboard& b1, Bitboard& b2) const { b1 = Bitboard(p[0], p[1]); b2 = Bitboard(p[2], p[3]); }

	// byte単位で入れ替えたBitboardを返す。
	// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
	Bitboard256 byte_reverse() const;

	// SSE2のunpackを実行して返す。
	// hi_out = _mm256_unpackhi_epi64(lo_in,hi_in);
	// lo_out = _mm256_unpacklo_epi64(lo_in,hi_in);
	static void unpack(const Bitboard256 hi_in,const Bitboard256 lo_in, Bitboard256& hi_out, Bitboard256& lo_out);

	// 2組のBitboard256を、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
	// 128bit整数とみなして1引き算したBitboardを返す。
	static void decrement(const Bitboard256 hi_in,const Bitboard256 lo_in, Bitboard256& hi_out, Bitboard256& lo_out);

	// 保持している2つの盤面を重ね合わせた(ORした)Bitboardを返す。
	Bitboard merge() const;

	// UnitTest
	static void UnitTest(Test::UnitTester&);
};

inline bool Bitboard256::operator == (const Bitboard256& rhs) const
{
#if defined (USE_AVX2)
	__m256i neq = _mm256_xor_si256(this->m, rhs.m);
	return /*_mm256_test_all_zeros*/ _mm256_testz_si256(neq, neq) ? true : false;
#else
	return (this->p[0] == rhs.p[0]) && (this->p[1] == rhs.p[1]) && (this->p[2] == rhs.p[2]) && (this->p[3] == rhs.p[3]);
	// return (this->p[0] ^ rhs.p[0]) | (this->p[1] ^ rhs.p[1]) | (this->p[2] ^ rhs.p[2]) | (this->p[3] == rhs.p[3]);
	// の方が速いかも？
#endif
}

// Bitboard256の1の升を'*'、0の升を'.'として表示する。デバッグ用。
std::ostream& operator<<(std::ostream& os, const Bitboard256& board);

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
inline const Bitboard rank1_n_bb(Color US, const Rank r)
{
	ASSERT_LV2(is_ok(r));
	return BB_Table::ForwardRanksBB[US][(US == BLACK ? r + 1 : 7 - r)];
}

// 敵陣を表現するBitboard。
namespace BB_Table { extern const Bitboard EnemyField[COLOR_NB]; }
inline const Bitboard enemy_field(Color Us) { return BB_Table::EnemyField[Us]; }

// 2升に挟まれている升を返すためのテーブル(その2升は含まない)
// この配列には直接アクセスせずにbetween_bb()を使うこと。
// 配列サイズが大きくてcache汚染がひどいのでシュリンクしてある。
namespace BB_Table {
	extern Bitboard BetweenBB[785];
	extern u16 BetweenIndex[SQ_NB_PLUS1][SQ_NB_PLUS1];
}

// 2升に挟まれている升を表すBitboardを返す。sq1とsq2が縦横斜めの関係にないときはBitboard(ZERO)が返る。
inline const Bitboard between_bb(Square sq1, Square sq2) { return BB_Table::BetweenBB[BB_Table::BetweenIndex[sq1][sq2]]; }

// 2升を通過する直線を返すためのテーブル
// 2つ目のindexは[0]:右上から左下、[1]:横方向、[2]:左上から右下、[3]:縦方向の直線。
// この配列には直接アクセスせず、line_bb()を使うこと。
namespace BB_Table { extern Bitboard LineBB[SQ_NB][4]; }

// 2升を通過する直線を返すためのBitboardを返す。sq1とsq2が縦横斜めの関係にないときに呼び出してはならない。
inline const Bitboard line_bb(Square sq1, Square sq2)
{
	static_assert(Effect8::DIRECT_RU == 0 && Effect8::DIRECT_LD == 7 , "");
	auto directions = Effect8::directions_of(sq1, sq2);
	ASSERT_LV3(directions != 0);
	static const int a[8] = { 0 , 1 , 2 , 3 , 3 , 2 , 1 , 0 };
	return BB_Table::LineBB[sq1][a[(int)Effect8::pop_directions(directions)]];
}

#if 0
// →　高速化のために、Effect8::directions_ofを使って実装しているのでコメントアウト。(shogi.hにおいて)
inline bool aligned(Square s1, Square s2, Square s3) {
	return LineBB[s1][s2] & s3;
}
#endif

// sqの升にいる敵玉に王手となるc側の駒ptの候補を得るテーブル。第2添字は(pr-1)を渡して使う。
// 直接アクセスせずに、check_candidate_bb()、around24_bb()を用いてアクセスすること。
namespace BB_Table { extern Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING - 1][COLOR_NB]; }

// sqの升にいる敵玉に王手となるus側の駒ptの候補を得る
// pr == ROOKは無条件全域なので代わりにHORSEで王手になる領域を返す。
// pr == KINGで呼び出してはならない。それは、around24_bb()のほうを用いる。
inline const Bitboard check_candidate_bb(Color us, PieceType pr, Square sq)
{
	ASSERT_LV3(PAWN<= pr && pr < KING && sq <= SQ_NB && is_ok(us));
	return BB_Table::CheckCandidateBB[sq][pr - 1][us];
}

#if defined(LONG_EFFECT_LIBRARY) // 詰みルーチンでは使わなくなったが、LONG_EFFECT_LIBRARYのなかで使っている。

namespace BB_Table { extern Bitboard CheckCandidateKingBB[SQ_NB_PLUS1]; }

// ある升の24近傍のBitboardを返す。
inline const Bitboard around24_bb(Square sq)
{
	ASSERT_LV3(sq <= SQ_NB);
	return BB_Table::CheckCandidateKingBB[sq];
}
#endif

// 歩が打てる筋が1になっているBitboardを返す。
// pawns : いま歩を打とうとしている手番側の歩のBitboard
//
// ここで返すBitboardは、
// C == BLACKの時は、1段目は0(歩が打てないから)、
// C == WHITEの時は、9段目は0(歩が打てないから)を保証する。
template <Color C>
inline Bitboard pawn_drop_mask(const Bitboard& pawns) {
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


// --------------------
// 利きのためのテーブル
// --------------------

// 利きのためのライブラリ
// 注意) ここのテーブルを直接参照せず、kingEffect()など、利きの関数を経由して用いること。

// --- 近接駒の利き

// 具体的なPiece名を指定することがほとんどなので1本の配列になっているメリットがあまりないので配列を分ける。
// 外部から直接アクセスしないようにnamespaceに入れておく。
namespace BB_Table
{
	extern Bitboard KingEffectBB[SQ_NB_PLUS1];
	extern Bitboard GoldEffectBB[SQ_NB_PLUS1][COLOR_NB];
	extern Bitboard SilverEffectBB[SQ_NB_PLUS1][COLOR_NB];
	extern Bitboard KnightEffectBB[SQ_NB_PLUS1][COLOR_NB];
	extern Bitboard PawnEffectBB[SQ_NB_PLUS1][COLOR_NB];

	// 盤上の駒をないものとして扱う、遠方駒の利き。香、角、飛
	extern Bitboard LanceStepEffectBB[SQ_NB_PLUS1][COLOR_NB];
	extern Bitboard BishopStepEffectBB[SQ_NB_PLUS1];
	extern Bitboard RookStepEffectBB[SQ_NB_PLUS1];
}

// =====================
//   大駒・小駒の利き
// =====================

// --------------------
//     近接駒の利き
// --------------------

// 歩の利き
// c側の升sqに置いた歩の利き。
template <Color c>
inline Bitboard pawnEffect(Square sq)
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return BB_Table::PawnEffectBB[sq][c];
}

// 歩の利き、非template版。
inline Bitboard pawnEffect(Color c, Square sq)
{
	return (c == BLACK) ? pawnEffect<BLACK>(sq) : pawnEffect<WHITE>(sq);
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

// 桂の利き
// これは遮断されることはないのでOccupiedBitboard不要。
inline Bitboard knightEffect(const Color c, const Square sq)
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return BB_Table::KnightEffectBB[sq][c];
}

// ↑のtemplate版。
template <Color C>
inline Bitboard knightEffect(const Square sq)
{
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return BB_Table::KnightEffectBB[sq][C];
}

// 銀の利き
inline Bitboard silverEffect(const Color c, const Square sq)
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return BB_Table::SilverEffectBB[sq][c];
}

// ↑のtemplate版
template <Color C>
inline Bitboard silverEffect(const Square sq)
{
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return BB_Table::SilverEffectBB[sq][C];
}

// 金の利き
inline Bitboard goldEffect(const Color c, const Square sq) {
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return BB_Table::GoldEffectBB[sq][c];
}

// ↑のtemplate版
template <Color C>
inline Bitboard goldEffect(const Square sq) {
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return BB_Table::GoldEffectBB[sq][C];
}

// 王の利き
// 王の利きは先後の区別はない。
inline Bitboard kingEffect(const Square sq)
{
	ASSERT_LV3(sq <= SQ_NB);
	return BB_Table::KingEffectBB[sq];
}

// --------------------
//  遠方駒のpseudoな利き
// --------------------
//
//  遠方駒で、盤上には他に駒がないものとして求める利き。(pseudo-attack)
//  関数名に"Step"とついているのは、pseudo-attackを意味する。
// 

// 盤上の駒を考慮しない香の利き
inline Bitboard lanceStepEffect(Color c, Square sq) {
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return BB_Table::LanceStepEffectBB[sq][c];
}
// ↑のtemplate版。
template <Color C>
inline Bitboard lanceStepEffect(Square sq) {
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return BB_Table::LanceStepEffectBB[sq][C];
}

// 盤上の駒を考慮しない角の利き
inline Bitboard bishopStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return BB_Table::BishopStepEffectBB[sq];
}

// 盤上の駒を考慮しない飛車の利き
inline Bitboard rookStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return BB_Table::RookStepEffectBB[sq];
}

// 盤上の駒を考慮しない馬の利き
inline Bitboard horseStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	// わざわざ用意するほどでもないので玉の利きと合成しておく。
	return BB_Table::BishopStepEffectBB[sq] | BB_Table::KingEffectBB[sq];
}

// 盤上の駒を考慮しない龍の利き
inline Bitboard dragonStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	// わざわざ用意するほどでもないので玉の利きと合成しておく。
	return BB_Table::RookStepEffectBB[sq] | BB_Table::KingEffectBB[sq];
}

// 盤上の駒を考慮しないQueenの動き。
inline Bitboard queenStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return rookStepEffect(sq) | bishopStepEffect(sq);
}

// 縦横十字の利き 利き長さ=1升分。
inline Bitboard cross00StepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return rookStepEffect(sq) & kingEffect(sq);
}

// 斜め十字の利き 利き長さ=1升分。
inline Bitboard cross45StepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return bishopStepEffect(sq) & kingEffect(sq);
}

// --------------------
//      遠方駒の利き
// --------------------
//
// 遠方駒で、盤上の駒の状態を考慮しながら利きを求める。
//

// 香 : occupied bitboardを考慮しながら香の利きを求める
template <Color C>
inline Bitboard lanceEffect(Square sq, const Bitboard& occupied)
{
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);

	// Bitboard 128bitのまま操作する。
#if 0
// これは、Qugiyのアイデア。
// Quigy[WCSC31]の手法 : cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal.pdf

	if (C == WHITE)
	{
		// 9段目が0、その他の升が1になっているmask。
		const Bitboard mask(0x3fdfeff7fbfdfeffULL, 0x000000000001feffULL);

		// 駒が存在しない升が1となるmaskを作る。ただし9段目は0固定。
		Bitboard em = occupied.andnot(mask);

		// emに歩を利きを足すと2進数の足し算の繰り上がりによって利きが届く升まで1になっていく。(縦型Bitboard特有)
		// 
		// 【図解】
		//  0 0 1 ... 1 : 駒が存在しない升が1、駒がある場所が0となっているmask。
		//          ↑最下位bitがsqの升を意味しているとする。
		//
		//            : sqの升に1加算する
		// 
		//  0 1 0 ... 0 : 2進数の足し算をした結果、左のようになる。
		//    ↑利きが届いた一番先の升が1。そこまでの桁は0になる。
		//
		// この結果を元のmaskとxorすることにより、bitの値が変化した場所(差分)が検出できる。
		// bitが変化した場所は、利きが届いたということだから、それこそが求める利きであった。

		// 同様の考え方で、角の2方向と飛車の左と下方向の利きは求まる。
		// 角の残り2方向は、byte reverseして同じ考え方を適用できる。
		// 飛車の右方向もbyte reverseして同じ考え方を適用できる。
		// 飛車の上方向はbyte reverseでは解決しないので先手の香の利きと合成する。(飛車の下方向も、後手の香の利きを合成した方が手っ取り早い)

		Bitboard t = em + pawnEffect<C>(sq);

		// tとemの差分が香の利き
		return t ^ em;

	}
	else {

		// step effectなのでここで返ってくるBitboardのsqの升は0であることが保証されている。
		const Bitboard se = lanceStepEffect<C>(sq);

		// 香の利きがあるかも知れない範囲に対して、駒がある升だけ1にする。
		Bitboard mocc = se & occupied;

		// 1を上方向に8升上書きコピーしてやる。これで、駒がある升より上は1になる。
		// 1になっていない升が、香の利きが通っている升ということになる。
		mocc |= mocc >> 1;
		mocc |= mocc >> 2;
		mocc |= mocc >> 4;
		mocc >>= 1;

		return mocc.andnot(se);
	}
#endif

#if 1
	// Qugiyのアルゴリズムを1～7筋と8～9筋に分けたもの。
	// こうすることで飛車の縦利きのコードがちょっと短くなる。
	if (C == WHITE)
	{
		// 後手の香

		if (Bitboard::part(sq) == 0)
		{
			// 香がp[0]に属する
			/*
			u64 mask = 0x3fdfeff7fbfdfeffULL;
			u64 em = ~occupied.p[0] & mask;
			u64 t = em + pawnEffect<C>(sq).p[0];
			return Bitboard(t ^ em , 0);
			*/
			// ↑Qugiyのアルゴリズムをpart()を用いて書いたもの。
			// ↓sqの次の升(＝歩の利き)に1加算する代わりに全体から1引く考え方。
			//    こっちの方が、定数が一つ消せて良いと思う。

			u64 mask = lanceStepEffect<WHITE>(sq).template extract64<0>();
			u64 em = occupied.extract64<0>() & mask;
			u64 t = em - 1; // 1引き算すれば、桁借りが上位桁が1のところまで波及する。
			return Bitboard((em ^ t) & mask, 0);
		}
		else {
			// 香がp[1]に属する
			/*
			u64 mask = 0x000000000001feffULL;
			u64 em =  ~occupied.p[1] & mask;
			u64 t = em + pawnEffect<C>(sq).p[1];
			return Bitboard(0 , t ^ em );
			*/

			u64 mask = lanceStepEffect<WHITE>(sq).template extract64<1>();
			u64 em = occupied.extract64<1>() & mask;
			u64 t = em - 1;
			return Bitboard(0, (em ^ t) & mask);
		}
	}
	else {
		// 先手の香

		if (Bitboard::part(sq) == 0)
		{
			// 香がp[0]に属する
			u64 se = lanceStepEffect<C>(sq).template extract64<0>();
			u64 mocc = se & occupied.extract64<0>();
			mocc |= mocc >> 1;
			mocc |= mocc >> 2;
			mocc |= mocc >> 4;
			mocc >>= 1;
			return Bitboard(~mocc & se, 0);
		}
		else {
			// 香がp[1]に属する
			u64 se = lanceStepEffect<C>(sq).template extract64<1>();
			u64 mocc = se & occupied.extract64<1>();
			mocc |= mocc >> 1;
			mocc |= mocc >> 2;
			mocc |= mocc >> 4;
			mocc >>= 1;
			return Bitboard(0, ~mocc & se);
		}
	}
#endif
}

// 香の利き、非template版。
inline Bitboard lanceEffect(Color c, Square sq, const Bitboard& occupied)
{
	return (c == BLACK) ? lanceEffect<BLACK>(sq, occupied) : lanceEffect<WHITE>(sq, occupied);
}

// 飛車の縦の利き(これはPEXTを用いていないのでどんな環境でも遅くはない)
// 香の利きを求めるQugiyのコードを応用。
inline Bitboard rookFileEffect(Square sq, const Bitboard& occupied)
{
	ASSERT_LV3(sq <= SQ_NB);

	if (Bitboard::part(sq) == 0)
	{
		// 飛車がp[0]に属する

		// 後手の香の利き
		u64 mask = lanceStepEffect<WHITE>(sq).template extract64<0>();
		u64 em = occupied.extract64<0>() & mask;
		u64 t = em - 1; // 1引き算すれば、桁借りが上位桁が1のところまで波及する。

		// 先手の香の利き
		u64 se = lanceStepEffect<BLACK>(sq).template extract64<0>();
		u64 mocc = se & occupied.extract64<0>();
		mocc |= mocc >> 1;
		mocc |= mocc >> 2;
		mocc |= mocc >> 4;
		mocc >>= 1;

		// 後手の香の利きと先手の香の利きを合成
		return Bitboard((em ^ t) & mask | (~mocc & se), 0);
	}
	else {
		// 飛車がp[1]に属する
		// ↑の処理と同様。

		u64 mask = lanceStepEffect<WHITE>(sq).template extract64<1>();
		u64 em = occupied.extract64<1>() & mask;
		u64 t = em - 1;

		u64 se = lanceStepEffect<BLACK>(sq).template extract64<1>();
		u64 mocc = se & occupied.extract64<1>();
		mocc |= mocc >> 1;
		mocc |= mocc >> 2;
		mocc |= mocc >> 4;
		mocc >>= 1;

		return Bitboard(0, (em ^ t) & mask | (~mocc & se));
	}
}

// ==== 飛車と角の利き ===

// 飛車の横の利き
extern Bitboard rookRankEffect(Square sq, const Bitboard& occupied);

// 飛車の利き
inline Bitboard rookEffect(const Square sq, const Bitboard& occupied) {
	// 縦の利きと横の利きを合成する。
	return rookRankEffect(sq, occupied) | rookFileEffect(sq, occupied);
}

// 角の利き
extern Bitboard bishopEffect(const Square sq, const Bitboard& occupied);

// 馬の利き
inline Bitboard horseEffect(Square sq, const Bitboard& occupied)
{
	return bishopEffect(sq, occupied) | kingEffect(sq);
}

// 龍の利き
inline Bitboard dragonEffect(Square sq, const Bitboard& occupied)
{
	return rookEffect(sq, occupied) | kingEffect(sq);
}

// 角と飛車の利きはセットで用いることが多いので、
// Queen(角+飛)の利きを(AVX512等で)求める関数を用意して、
// rookStepEffectとbishopStepEffectで分解してしまう方が速いかも知れない。

// === 大駒の部分利き(SEEなどで用いる) ===

// sqの升から各方向への利き
// 右上、右、右下、上方向は、byte_reverse()してあるので、普通の利きではないから注意。
// 6方向しか使っていないので詰めてある。
namespace BB_Table { extern Bitboard QUGIY_STEP_EFFECT[Effect8::DIRECT_NB - 2][SQ_NB_PLUS1]; }

// 方向利き
template <Effect8::Direct D>
Bitboard rayEffect(Square sq, const Bitboard& occupied)
{
	// DIRECT_Uはbyte_reverse()しても正しく求められないからlanceEffectを用いる。
	// DIRECT_Dも、lanceEffectにこれ専用の高速なコードが書いてあるのでそれを用いる。
	if (D == Effect8::DIRECT_U) return lanceEffect<BLACK>(sq, occupied);
	if (D == Effect8::DIRECT_D) return lanceEffect<WHITE>(sq, occupied);

	ASSERT_LV3(D < Effect8::DIRECT_NB);

	// 6方向しか使っていないのでテーブルを詰めてある。
	const int dd = D > Effect8::DIRECT_D ? D - 2 : D;
	const Bitboard mask = BB_Table::QUGIY_STEP_EFFECT[dd][sq];

	// 右上、右、右下、上方向(Squareとしてみた時に、値が減る方向)
	constexpr bool reverse =
		   D == Effect8::DIRECT_RU
		|| D == Effect8::DIRECT_R
		|| D == Effect8::DIRECT_RD;

	Bitboard bb = occupied;

	if (reverse)
		bb = bb.byte_reverse();

	// 利きに関係する升のoccupiedだけを抽出
	bb &= mask;
	// 1減算することにより、利きが通る升までが変化する。
	Bitboard bb_minus_one = bb.decrement();
	// 変化したbitを抽出してmaskすれば出来上がり。
	bb = (bb ^ bb_minus_one) & mask;

	if (reverse)
		bb = bb.byte_reverse();

	return bb;
}

// sqの升から指定した方向dへの利き。盤上の駒も考慮する。
extern Bitboard directEffect(Square sq, Effect8::Direct d, const Bitboard& occupied);

// --------------------
//   汎用性のある利き
// --------------------

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。(step effect)
// pc == QUEENだと馬+龍の利きが返る。盤上には駒は何もないものとして考える。
extern Bitboard effects_from(Piece pc, Square sq);

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。
// pc == QUEENだと馬+龍の利きが返る。
extern Bitboard effects_from(Piece pc, Square sq, const Bitboard& occ);

// --------------------
//   Stockfishとの互換性のために用意
// --------------------

/// pawn_attacks_bb() returns the squares attacked by pawns of the given color
/// from the squares in the given bitboard.

// pawn_attacks_bb()は、手番C側の利き。
// b : 手番C側の歩のBitboardを渡す。

template<Color C>
inline Bitboard pawn_attacks_bb(Bitboard b)
{
	return pawnBbEffect<C>(b);
}

inline Bitboard pawn_attacks_bb(Color c, Square s)
{
	return pawnEffect(c, s);
}


/// attacks_bb(Square) returns the pseudo attacks of the give piece type
/// assuming an empty board.

// attacks_bb<PC>(s)は、駒PCをsの升に置いた時の利き。
// ※　StockfishではPCのところ、PieceType Ptとなっているので注意
// 盤上には駒はないものとする。

template<Piece PC>
inline Bitboard attacks_bb(Square s) {

	return effects_from(PC, s);
}

/// attacks_bb(Square, Bitboard) returns the attacks by the given piece
/// assuming the board is occupied according to the passed Bitboard.
/// Sliding piece attacks do not continue passed an occupied square.

// attacks_bb<PC>(s)は、駒PCをsの升に置いた時の利き。
// ※　StockfishではPCのところ、PieceType Ptとなっているので注意
// occupied : 盤上の駒

template<Piece PC>
inline Bitboard attacks_bb(Square s, Bitboard occupied) {

	return effects_from(PC, s, occupied);
}


#endif // #ifndef _BITBOARD_H_
