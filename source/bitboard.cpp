#include "bitboard.h"

#if !STOCKFISH
#include <iostream>
#include <ostream>

#include "misc.h"
#include "attacks.h"
#include "testcmd/unit_test.h"
#endif

namespace YaneuraOu {

using namespace Attacks;

// ----- Bitboard const

// 外部から直接アクセスしないようにnamespaceに入れておく。
namespace BB_Table
{
	const Bitboard FILE1_BB = Bitboard(UINT64_C(0x1ff) << (9 * 0), 0);
	const Bitboard FILE2_BB = Bitboard(UINT64_C(0x1ff) << (9 * 1), 0);
	const Bitboard FILE3_BB = Bitboard(UINT64_C(0x1ff) << (9 * 2), 0);
	const Bitboard FILE4_BB = Bitboard(UINT64_C(0x1ff) << (9 * 3), 0);
	const Bitboard FILE5_BB = Bitboard(UINT64_C(0x1ff) << (9 * 4), 0);
	const Bitboard FILE6_BB = Bitboard(UINT64_C(0x1ff) << (9 * 5), 0);
	const Bitboard FILE7_BB = Bitboard(UINT64_C(0x1ff) << (9 * 6), 0);
	const Bitboard FILE8_BB = Bitboard(0, 0x1ff << (9 * 0));
	const Bitboard FILE9_BB = Bitboard(0, 0x1ff << (9 * 1));

	const Bitboard RANK1_BB = Bitboard(UINT64_C(0x40201008040201) << 0, 0x201 << 0);
	const Bitboard RANK2_BB = Bitboard(UINT64_C(0x40201008040201) << 1, 0x201 << 1);
	const Bitboard RANK3_BB = Bitboard(UINT64_C(0x40201008040201) << 2, 0x201 << 2);
	const Bitboard RANK4_BB = Bitboard(UINT64_C(0x40201008040201) << 3, 0x201 << 3);
	const Bitboard RANK5_BB = Bitboard(UINT64_C(0x40201008040201) << 4, 0x201 << 4);
	const Bitboard RANK6_BB = Bitboard(UINT64_C(0x40201008040201) << 5, 0x201 << 5);
	const Bitboard RANK7_BB = Bitboard(UINT64_C(0x40201008040201) << 6, 0x201 << 6);
	const Bitboard RANK8_BB = Bitboard(UINT64_C(0x40201008040201) << 7, 0x201 << 7);
	const Bitboard RANK9_BB = Bitboard(UINT64_C(0x40201008040201) << 8, 0x201 << 8);

	const Bitboard FILE_BB[FILE_NB] = { FILE1_BB,FILE2_BB,FILE3_BB,FILE4_BB,FILE5_BB,FILE6_BB,FILE7_BB,FILE8_BB,FILE9_BB };
	const Bitboard RANK_BB[RANK_NB] = { RANK1_BB,RANK2_BB,RANK3_BB,RANK4_BB,RANK5_BB,RANK6_BB,RANK7_BB,RANK8_BB,RANK9_BB };


	// sqの升が1であるbitboard
	Bitboard SquareBB[SQ_NB_PLUS1];

	const Bitboard ForwardRanksBB[COLOR_NB][RANK_NB] =
	{
		{
			Bitboard(0),
			RANK1_BB,
			RANK1_BB | RANK2_BB,
			RANK1_BB | RANK2_BB | RANK3_BB,
			RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
			~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB),
			~(RANK9_BB | RANK8_BB | RANK7_BB),
			~(RANK9_BB | RANK8_BB),
			~RANK9_BB
		},{
			~RANK1_BB,
			~(RANK1_BB | RANK2_BB),
			~(RANK1_BB | RANK2_BB | RANK3_BB),
			~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
			RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB,
			RANK9_BB | RANK8_BB | RANK7_BB,
			RANK9_BB | RANK8_BB,
			RANK9_BB,
			Bitboard(0)
		}
	};

	// 敵陣を表現するBitboard。
	const Bitboard EnemyField[COLOR_NB] = {
		RANK1_BB | RANK2_BB | RANK3_BB ,
		RANK7_BB | RANK8_BB | RANK9_BB
	};

	// SquareからSquareWithWallへの変換テーブル
	SquareWithWall sqww_table[SQ_NB_PLUS1];

}



using namespace BB_Table;

// ----------------------------------------------------------------------------------------------

// Prints an ASCII representation of a bitboard to the output stream.
// Useful for debugging.
// 標準出力でのデバッグに使う、BitboardのASCII表現を出力する。
// StockfishのBitboards::pretty()に対応し、将棋側では従来どおりostreamへ直接書き込む。
std::ostream& operator<<(std::ostream& os, const Bitboard& board)
{
	for (Rank rank = RANK_1; rank <= RANK_9; ++rank)
	{
		for (File file = FILE_9; file >= FILE_1; --file)
			os << ((board & (file | rank)) ? " *" : " .");
		os << std::endl;
	}
	// 連続して表示させるときのことを考慮して改行を最後に入れておく。
	os << std::endl;
	return os;
}

// Bitboard関連の各種テーブルの初期化。
void Bitboards::init()
{
	// ------------------------------------------------------------
	//        Bitboard関係のテーブルの初期化
	// ------------------------------------------------------------

	// 1) SquareWithWallテーブルの初期化。

	for (auto sq : SQ)
		sqww_table[sq] = SquareWithWall(SQWW_11 + (i32)file_of(sq) * SQWW_L + (i32)rank_of(sq) * SQWW_D);


	// 2) Square型のsqの指す升が1であるBitboardがSquareBB。これをまず初期化する。

	for (auto sq : SQ)
	{
		Rank r = rank_of(sq);
		File f = file_of(sq);
		SquareBB[sq] = Bitboard(
			(f <= FILE_7) ? ((uint64_t)1 << (f * 9 + r)) : 0,
			(f >= FILE_8) ? ((uint64_t)1 << ((f - FILE_8) * 9 + r)) : 0
		);
	}

}

// --- Bitboardの実装

Bitboard::Bitboard(u64 p0, u64 p1) :
#if defined(USE_SSE2)
	// この命令、引数の順に注意。
	m( _mm_set_epi64x(p1,p0))
#else
	p { p0 , p1 }
#endif
{}

// 値を直接代入する。
void Bitboard::set(u64 p0, u64 p1)
{
#if defined(USE_SSE2)
	m = _mm_set_epi64x(p1,p0);
#else
	p[0] = p0; p[1] = p1;
#endif
}

Bitboard::operator bool() const
{
#if defined(USE_SSE41)
	return !_mm_testz_si128(m, m);
#else
	return (this->merge() ? true : false);
#endif
}

bool Bitboard::test(Bitboard rhs) const
{
#if defined(USE_SSE41)
	return !_mm_testz_si128(m, rhs.m);
#else
	return (*this & rhs);
#endif
}

Square Bitboard::pop()
{
	u64 q0 = extract64<0>();  Square sq;
	if (q0 != 0) { sq = Square(pop_lsb(q0)); insert64<0>(q0); }
	else { u64 q1 = extract64<1>();  sq = Square(pop_lsb(q1) + 63); insert64<1>(q1); }
	return sq;
}

bool Bitboard::operator == (const Bitboard& rhs) const
{
#if defined (USE_SSE41)
	// 以下のようにすると2命令で済むらしい。
	// testing equality between two __m128i variables
	// cf.http://stackoverflow.com/questions/26880863/sse-testing-equality-between-two-m128i-variables
	__m128i neq = _mm_xor_si128(this->m, rhs.m);
	return _mm_test_all_zeros(neq, neq) ? true : false;
#else
	return (this->p[0] == rhs.p[0]) && (this->p[1] == rhs.p[1]);
#endif
}

Bitboard::Bitboard(Square sq) { *this = SquareBB[sq]; }

// Square型との演算子
Bitboard operator|(const Bitboard& b, Square s) { return b | SquareBB[s]; }
Bitboard operator&(const Bitboard& b, Square s) { return b & SquareBB[s]; }
Bitboard operator^(const Bitboard& b, Square s) { return b ^ SquareBB[s]; }

// 単項演算子
// →　NOTで書くと、使っていないbit(p[0]のbit63)がおかしくなるのでBitboard(1)でxorしないといけない。
Bitboard operator ~ (const Bitboard& a) { return a ^ Bitboard(1); }

// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
const Bitboard begin(const Bitboard& b) { return b; }
const Bitboard end(const Bitboard&) { return Bitboard(ZERO); }

// このbitboardを128bitレジスタとみなして1減算したBitboardを返す。方向利きの計算で用いる。
Bitboard Bitboard::decrement() const
{
#if defined(USE_SSE2)

#if defined(USE_SSE41)
	// _mm_setzero_si128()は同一レジスタのXORとなる。
	// 同一レジスタのXORは依存性が切れる
	// SandyBridge以降Skylake系までなら実行ユニットも使用しない
	__m128i t2 = _mm_cmpeq_epi64(m, _mm_setzero_si128());
	// alignrを使ってcmpeqの結果(p[0] == 0 ? -1 : 0)を上位64bit側にシフトしつつ、下位を-1で埋める
	t2 = _mm_alignr_epi8(t2, _mm_set1_epi64x(-1LL), 8);
	__m128i t1 = _mm_add_epi64(m, t2);
#else // SSE2用のコード
	// p[0]--;
	__m128i c  = _mm_set_epi64x(0, 1);
	__m128i t1 = _mm_sub_epi64(m, c);

	// if (p[0] MSB == 1) p[1]--;
	__m128i t2 = _mm_srli_epi64(t1, 63); // MSBをbit0に持ってきて、byte shiftで上位64bit側に移動させて減算
	t2 = _mm_slli_si128(t2, 8);
	t1 = _mm_sub_epi64(t1, t2);
#endif

	Bitboard bb;
	bb.m = t1;
	return bb;
#else // no SSE
	return Bitboard(p[0] - 1, p[0] == 0 ? p[1] - 1 : p[1]);
#endif
}

// byte単位で入れ替えたBitboardを返す。
// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
Bitboard Bitboard::byte_reverse() const
{
#if defined(USE_SSSE3)
	const __m128i shuffle = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	Bitboard b0;
	// _mm_shuffle_epi8はSSSE3で実装された命令らしい。
	b0.m = _mm_shuffle_epi8(m, shuffle);
	return b0;
#else
	Bitboard b0;
	b0.p[0] = bswap64(p[1]);
	b0.p[1] = bswap64(p[0]);
	return b0;
#endif
}

// SSE2のunpackを実行して返す。
// hi_out = _mm_unpackhi_epi64(lo_in,hi_in);
// lo_out = _mm_unpacklo_epi64(lo_in,hi_in);
void Bitboard::unpack(const Bitboard hi_in, const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out)
{
#if defined(USE_SSE2)
	hi_out.m = _mm_unpackhi_epi64(lo_in.m , hi_in.m);
	lo_out.m = _mm_unpacklo_epi64(lo_in.m , hi_in.m);
#else
	hi_out.p[0] = lo_in.p[1];
	hi_out.p[1] = hi_in.p[1];

	lo_out.p[0] = lo_in.p[0];
	lo_out.p[1] = hi_in.p[0];
#endif
}

// 2組のBitboardを、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
// 128bit整数とみなして1引き算したBitboardを返す。
void Bitboard::decrement(const Bitboard hi_in,const Bitboard lo_in, Bitboard& hi_out, Bitboard& lo_out)
{
#if defined(USE_SSE41)

	// loが0の時だけ1減算するときにhiからの桁借りが生じるので、
	// hi += (lo == 0) ? -1 : 0;
	// みたいな処理で良い。
	hi_out.m = _mm_add_epi64(hi_in.m, _mm_cmpeq_epi64(lo_in.m, _mm_setzero_si128()));

	//  1減算する
	lo_out.m = _mm_add_epi64(lo_in.m, _mm_set1_epi64x(-1LL)); // cmpeqとどっちがいいか？
#else
	// bool型はtrueだと(暗黙の型変換で)1だとみなされる。
	hi_out.p[0] = hi_in.p[0] - (lo_in.p[0] == 0);
	hi_out.p[1] = hi_in.p[1] - (lo_in.p[1] == 0);

	lo_out.p[0] = lo_in.p[0] - 1;
	lo_out.p[1] = lo_in.p[1] - 1;
#endif
}

// 2bit以上あるかどうかを判定する。縦横斜め方向に並んだ駒が2枚以上であるかを判定する。この関係にないと駄目。
// この関係にある場合、Bitboard::merge()によって被覆しないことがBitboardのレイアウトから保証されている。
bool Bitboard::more_than_one() const
{
	ASSERT_LV2(!cross_over());
	return POPCNT64(merge()) > 1;
}

// UnitTest
void Bitboard::UnitTest(Test::UnitTester& tester, IEngine& engine)
{
	//Bitboard b(SQ_75);
	//std::cout << lanceEffect<WHITE>(SQ_71, b) << std::endl;
	//std::cout << lanceEffect<WHITE>(SQ_72, b) << std::endl;

	auto section1 = tester.section("Bitboard");
#if defined(ENABLE_TEST_CMD)
    Attacks::test_sliding_attacks(tester);
#endif

	{
		// SQの升のbitを立てて、それがちゃんと読み取れるかのテスト
		bool all_ok = true;
		for (Rank r = RANK_1; r <= RANK_9; ++r)
			for (File f = FILE_1; f <= FILE_9; ++f)
			{
				Square sq = f | r;
				Bitboard b(sq);
				all_ok &= b.test(sq);    // そのbitが立っているか
				all_ok &= b.pop() == sq; // 1の立っているbitをもらえば、その升であるか。
			}
		tester.test("sq occupied", all_ok);
	}
	{
		// ByteReverseがちゃんと機能しているかのテスト

		Bitboard b(0x0123456789abcdef, 0xfedcba9876543210);
		Bitboard r = b.byte_reverse();

		tester.test("byte_reverse", r.extract64<0>() == 0x1032547698badcfe && r.extract64<1>() == 0xefcdab8967452301);
	}
	{
		// 9段目が0、そこ以外が1のmask(香の利きを求めるコードのなかで使っている)
		Bitboard mask(0x3fdfeff7fbfdfeffULL , 0x000000000001feffULL);

		tester.test("RANK9_BB", RANK9_BB == ~mask);
	}

	{
		// 駒の利き

		bool all_ok = true;

		// 何も駒のない盤面上に駒ptを55に置いた時の利きの数。
		int p0_table[] = {
			0,1,4,2,5,16,16,6,	// Empty、歩、香、桂、銀、角、飛、金
			8,6,6,6,6,20,20,6,	// 玉、と、…
		};
		// 駒が敷き詰められた盤面上で駒ptを55に置いた時の利きの数。
		int p1_table[] = {
			0,1,1,2,5, 4, 4,6,	// Empty、歩、香、桂、銀、角、飛
			8,6,6,6,6, 8, 8,6,	// 玉、と、…
		};

		for (Color c : COLOR)
			for (PieceType pt = PAWN ; pt < PIECE_TYPE_NB ; ++pt )
			{
				Piece pc = make_piece(c, pt);
				Bitboard bb0 = attacks_bb(pc, SQ_55, Bitboard(ZERO));
				int p0 = bb0.pop_count();
				Bitboard bb1 = attacks_bb(pc, SQ_55, Bitboard(1));
				int p1 = bb1.pop_count();
				bool ok0 = (p0 == p0_table[(int)pt]);
				if (!ok0)
				{
					std::cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(ZERO) , pop_count = " << p0 << std::endl;
					std::cout << bb0 << std::endl;
				}
				bool ok1 = (p1 == p1_table[(int)pt]);
				if (!ok1)
				{
					std::cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(1) , pop_count = " << p1 << std::endl;
					std::cout << bb1 << std::endl;
				}
				all_ok &= ok0 & ok1;
			}
		tester.test("attacks_bb", all_ok);
	}

	{
		// 方向利き

		bool all_ok = true;
		Bitboard occ(SQ_77);
		Bitboard zero(ZERO);
		all_ok &= rayEffect<Effect8::DIRECT_LD>(SQ_55, occ) == between_bb(SQ_55, SQ_88);
		all_ok &= rayEffect<Effect8::DIRECT_LD>(SQ_55, zero) == QUGIY_STEP_EFFECT[Effect8::DIRECT_LD - 2][SQ_55];

		Bitboard occ2(SQ_33);
		all_ok &= rayEffect<Effect8::DIRECT_RU>(SQ_55, occ2) == between_bb(SQ_55, SQ_22);

		tester.test("rayEffect", all_ok);
	}

	{
		// SSE unpack

		const Bitboard bb_0000(0, 0);
		const Bitboard bb_ff00((u64)0, (u64)-1LL); // Bitboard(low64,high64)と書くようになっている。
		const Bitboard bb_00ff((u64)-1LL, (u64)0);
		const Bitboard bb_ffff((u64)-1LL, (u64)-1LL);

		bool all_ok = true;
		Bitboard hi_out, lo_out;

		Bitboard::unpack(bb_ffff, bb_0000, hi_out, lo_out);

		all_ok &= hi_out == bb_ff00;
		all_ok &= lo_out == bb_ff00;

		Bitboard::unpack(bb_ff00, bb_ff00, hi_out, lo_out);

		all_ok &= hi_out == bb_ffff;
		all_ok &= lo_out == bb_0000;

		Bitboard::unpack(bb_ff00, bb_00ff, hi_out, lo_out);

		all_ok &= hi_out == bb_ff00;
		all_ok &= lo_out == bb_00ff;

		tester.test("unpack", all_ok );
	}

	{
		// Bitboard::decrement() method

		const Bitboard all_bb((u64)-1LL, (u64)-1LL);
		const Bitboard all_bb_minus_one((u64)-2LL, (u64)-2LL);
		const Bitboard zero_bb(0, 0);

		bool all_ok = true;
		Bitboard hi_out, lo_out;

		// これなら桁借りは生じない。
		Bitboard::decrement(zero_bb, all_bb, hi_out, lo_out);

		all_ok &= hi_out == zero_bb;
		all_ok &= lo_out == all_bb_minus_one;

		// これは桁借りが生じる。
		Bitboard::decrement(all_bb, zero_bb, hi_out, lo_out);

		all_ok &= hi_out == all_bb_minus_one;
		all_ok &= lo_out == all_bb;

		tester.test("decrement(static function)", all_ok );
	}
	{
		const Bitboard zero_bb(0, 0);
		const Bitboard minus_one((u64)-1LL, (u64)-1LL);
		const Bitboard one_bb(1, 0);
		bool all_ok = true;

		all_ok &= zero_bb.decrement() == minus_one;
		all_ok &= one_bb.decrement()  == zero_bb;

		tester.test("decrement(method)", all_ok);
	}
	{
		// pawn_attacks_bbのテスト
		tester.test("pawn_attacks_bb<BLACK>",pawn_attacks_bb<BLACK>(RANK7_BB) == RANK6_BB);
		tester.test("pawn_attacks_bb<WHITE>",pawn_attacks_bb<WHITE>(RANK7_BB) == RANK8_BB);

		tester.test("attacks_bb(B_KNIGHT)",attacks_bb(B_KNIGHT, SQ_77) == (Bitboard(SQ_85) | Bitboard(SQ_65)));
		tester.test("attacks_bb(W_KNIGHT)",attacks_bb(W_KNIGHT, SQ_77) == (Bitboard(SQ_89) | Bitboard(SQ_69)));
	}
	{
		bool all_ok = true;
		Bitboard bb;

		bb = Bitboard(SQ_22) | Bitboard(SQ_77);
		all_ok &= least_significant_square_bb(bb) == Bitboard(SQ_22);
		bb = Bitboard(SQ_99) | Bitboard(SQ_87);
		all_ok &= least_significant_square_bb(bb) == Bitboard(SQ_87);

		tester.test("least_significant_square_bb",all_ok);
	}
}

} // namespace YaneuraOu
