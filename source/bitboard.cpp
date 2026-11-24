#include "bitboard.h"

#if !STOCKFISH
#include <ostream>
#endif

namespace YaneuraOu {

// Returns an ASCII representation of a bitboard suitable
// to be printed to standard output. Useful for debugging.
// 標準出力でのデバッグに使う、BitboardのASCII表現を返す。
std::string Bitboards::pretty(Bitboard b) {
#if STOCKFISH

    std::string s = "+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            s += b & make_square(f, r) ? "| X " : "|   ";

        s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";

        if (r == RANK_1)
            break;
    }
    s += "  a   b   c   d   e   f   g   h\n";

    return s;
#else
    // Stockfish版は8×8の盤を罫線・座標付きで表示する。将棋版は9×9を * / . で表示する。
    // 既存の表示形式との互換性のため、罫線・座標を付けず末尾に空行を置く。

    std::string s;

    for (Rank r = RANK_1; r <= RANK_9; ++r)
    {
        for (File f = FILE_9; f >= FILE_1; --f)
            s += b & (f | r) ? " *" : " .";

        s += '\n';
    }
    // 連続して表示したときの区切りとなる空行も、従来の表示と同じように付ける。
    s += '\n';

    return s;
#endif
}


#if !STOCKFISH
// ----- Bitboard const

// 外部から直接アクセスしないようにnamespaceに入れておく。
namespace BB_Table
{
	const Bitboard File1BB = Bitboard(UINT64_C(0x1ff) << (9 * 0), 0);
	const Bitboard File2BB = Bitboard(UINT64_C(0x1ff) << (9 * 1), 0);
	const Bitboard File3BB = Bitboard(UINT64_C(0x1ff) << (9 * 2), 0);
	const Bitboard File4BB = Bitboard(UINT64_C(0x1ff) << (9 * 3), 0);
	const Bitboard File5BB = Bitboard(UINT64_C(0x1ff) << (9 * 4), 0);
	const Bitboard File6BB = Bitboard(UINT64_C(0x1ff) << (9 * 5), 0);
	const Bitboard File7BB = Bitboard(UINT64_C(0x1ff) << (9 * 6), 0);
	const Bitboard File8BB = Bitboard(0, 0x1ff << (9 * 0));
	const Bitboard File9BB = Bitboard(0, 0x1ff << (9 * 1));

	const Bitboard Rank1BB = Bitboard(UINT64_C(0x40201008040201) << 0, 0x201 << 0);
	const Bitboard Rank2BB = Bitboard(UINT64_C(0x40201008040201) << 1, 0x201 << 1);
	const Bitboard Rank3BB = Bitboard(UINT64_C(0x40201008040201) << 2, 0x201 << 2);
	const Bitboard Rank4BB = Bitboard(UINT64_C(0x40201008040201) << 3, 0x201 << 3);
	const Bitboard Rank5BB = Bitboard(UINT64_C(0x40201008040201) << 4, 0x201 << 4);
	const Bitboard Rank6BB = Bitboard(UINT64_C(0x40201008040201) << 5, 0x201 << 5);
	const Bitboard Rank7BB = Bitboard(UINT64_C(0x40201008040201) << 6, 0x201 << 6);
	const Bitboard Rank8BB = Bitboard(UINT64_C(0x40201008040201) << 7, 0x201 << 7);
	const Bitboard Rank9BB = Bitboard(UINT64_C(0x40201008040201) << 8, 0x201 << 8);

	const Bitboard FILE_BB[FILE_NB] = { File1BB,File2BB,File3BB,File4BB,File5BB,File6BB,File7BB,File8BB,File9BB };
	const Bitboard RANK_BB[RANK_NB] = { Rank1BB,Rank2BB,Rank3BB,Rank4BB,Rank5BB,Rank6BB,Rank7BB,Rank8BB,Rank9BB };


	// sqの升が1であるbitboard
	Bitboard SquareBB[SQ_NB_PLUS1];

	const Bitboard ForwardRanksBB[COLOR_NB][RANK_NB] =
	{
		{
			Bitboard(0),
			Rank1BB,
			Rank1BB | Rank2BB,
			Rank1BB | Rank2BB | Rank3BB,
			Rank1BB | Rank2BB | Rank3BB | Rank4BB,
			~(Rank9BB | Rank8BB | Rank7BB | Rank6BB),
			~(Rank9BB | Rank8BB | Rank7BB),
			~(Rank9BB | Rank8BB),
			~Rank9BB
		},{
			~Rank1BB,
			~(Rank1BB | Rank2BB),
			~(Rank1BB | Rank2BB | Rank3BB),
			~(Rank1BB | Rank2BB | Rank3BB | Rank4BB),
			Rank9BB | Rank8BB | Rank7BB | Rank6BB,
			Rank9BB | Rank8BB | Rank7BB,
			Rank9BB | Rank8BB,
			Rank9BB,
			Bitboard(0)
		}
	};

	// 敵陣を表現するBitboard。
	const Bitboard EnemyField[COLOR_NB] = {
		Rank1BB | Rank2BB | Rank3BB ,
		Rank7BB | Rank8BB | Rank9BB
	};

	// SquareからSquareWithWallへの変換テーブル
	SquareWithWall sqww_table[SQ_NB_PLUS1];

}



using namespace BB_Table;

// ----------------------------------------------------------------------------------------------

// operator<<はBitboards::pretty()の表示内容をストリームへ出力する。
std::ostream& operator<<(std::ostream& os, const Bitboard& board)
{
    return os << Bitboards::pretty(board) << std::flush;
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

Square Bitboard::pop_lsb()
{
	u64 q0 = extract64<0>();  Square sq;
	if (q0 != 0) { sq = Square(YaneuraOu::pop_lsb(q0)); insert64<0>(q0); }
	else { u64 q1 = extract64<1>();  sq = Square(YaneuraOu::pop_lsb(q1) + 63); insert64<1>(q1); }
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

Bitboard::Bitboard(Square sq) { *this = square_bb(sq); }

// Square型との演算子
Bitboard operator|(const Bitboard& b, Square s) { return b | square_bb(s); }
Bitboard operator&(const Bitboard& b, Square s) { return b & square_bb(s); }
Bitboard operator^(const Bitboard& b, Square s) { return b ^ square_bb(s); }

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

// 128bit全体の16byteの順序を反転したBitboardを返す。
// 2つの64bitの位置も入れ替える。各64bit内だけのバイト反転ではない。
// Stockfishの同名自由関数は64bitの8byteを反転するが、将棋版は16byte全体を反転する。
// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
Bitboard Bitboard::byteswap_bb() const
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

// 2bit以上あるかどうかを判定する。
// Stockfish版は任意のBitboardに使えるが、将棋版は!cross_over()が前提。
// 2つの64bitをORして数えるため、同じbit位置が両方に立つ盤面には使えない。
bool Bitboard::more_than_one() const
{
	ASSERT_LV2(!cross_over());
	return POPCNT64(merge()) > 1;
}

#endif

} // namespace YaneuraOu
