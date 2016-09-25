#ifndef _BITBOARD_H_
#define _BITBOARD_H_

#include "shogi.h"

// --------------------
//     Bitboard
// --------------------

namespace Bitboards{
  void init(); // Bitboard関連のテーブル初期化のための関数
}

// Bitboard本体クラス

struct alignas(16) Bitboard
{
#ifdef  USE_SSE2
  union
  {
    // 64bitずつとして扱いとき用
    uint64_t p[2];

    // SSEで取り扱い時用
    // bit0がSQ_11,bit1がSQ_12,…,bit81がSQ_99を表現する。
    // このbit位置がSquare型と対応する。
    // ただしbit63は未使用。これは、ここを余らせることで飛車の利きをpext1回で求めるためのhack。
    // Aperyを始めとするmagic bitboard派によって考案された。
    __m128i m;
  };
#else // no SSE
  u64 p[2];
#endif

#ifdef  USE_SSE2
  // SSE2が使えるときは代入等においてはSSE2を使ったコピーがなされて欲しい。

  Bitboard& operator = (const Bitboard& rhs) { _mm_store_si128(&this->m, rhs.m); return *this; }

  Bitboard(const Bitboard& bb) { _mm_store_si128(&this->m, bb.m); }
#endif

  // --- ctor

  // 初期化しない。このとき中身は不定。
  Bitboard() {}

  // p[0],p[1]の値を直接指定しての初期化。(Bitboard定数の初期化のときのみ用いる)
  Bitboard(uint64_t p0, uint64_t p1) { p[0] = p0; p[1] = p1; }

  // sqの升が1のBitboardとして初期化する。
  Bitboard(Square sq);
  
  // 値を直接代入する。
  void set(uint64_t p0, uint64_t p1) { p[0] = p0; p[1] = p1; }

  // --- property

  // Stockfishのソースとの互換性がよくなるようにboolへの暗黙の型変換書いておく。
  operator bool() const {
#ifdef USE_SSE41
    return !(_mm_testz_si128(m, _mm_set1_epi8(static_cast<char>(0xffu))));
#else
    return (this->merge() ? true : false);
#endif
  }

  // p[0]とp[1]をorしたものを返す。toU()相当。
  uint64_t merge() const { return p[0] | p[1]; }

  // p[0]とp[1]とで and したときに被覆しているbitがあるか。
  // merge()したあとにpext()を使うときなどに被覆していないことを前提とする場合にそのassertを書くときに使う。
  bool cross_over() const { return p[0] & p[1]; }

  // 指定した升(Square)が Bitboard のどちらの u64 変数の要素に属するか。
  // 本ソースコードのように縦型Bitboardにおいては、香の利きを求めるのにBitboardの
  // 片側のp[x]を調べるだけで済むので、ある升がどちらに属するかがわかれば香の利きは
  // そちらを調べるだけで良いというAperyのアイデア。
  constexpr static int part(const Square sq) { return static_cast<int>(SQ_79 < sq); }

  // --- operator

  // 下位bitから1bit拾ってそのbit位置を返す。
  // 絶対に1bitはnon zeroと仮定
  // while(to = bb.pop())
  //  make_move(from,to);
  // のように用いる。
  FORCE_INLINE Square pop() { return (p[0] != 0) ? Square(pop_lsb(p[0])) : Square(pop_lsb(p[1]) + 63); }

  // このBitboardの値を変えないpop()
  FORCE_INLINE Square pop_c() const { return (p[0] != 0) ? Square(LSB64(p[0])) : Square(LSB64(p[1]) + 63); }

  // pop()をp[0],p[1]に分けて片側ずつする用
  FORCE_INLINE Square pop_from_p0() { ASSERT_LV3(p[0] != 0);  return Square(pop_lsb(p[0])); }
  FORCE_INLINE Square pop_from_p1() { ASSERT_LV3(p[1] != 0);  return Square(pop_lsb(p[1]) + 63); }

  // 1のbitを数えて返す。
  int pop_count() const { return (int)(POPCNT64(p[0]) + POPCNT64(p[1])); }

  // 代入型演算子

#ifdef USE_SSE2
  Bitboard& operator |= (const Bitboard& b1) { this->m = _mm_or_si128( m, b1.m); return *this; }
  Bitboard& operator &= (const Bitboard& b1) { this->m = _mm_and_si128(m, b1.m); return *this; }
  Bitboard& operator ^= (const Bitboard& b1) { this->m = _mm_xor_si128(m, b1.m); return *this; }
  Bitboard& operator += (const Bitboard& b1) { this->m = _mm_add_epi64(m, b1.m); return *this; }
  Bitboard& operator -= (const Bitboard& b1) { this->m = _mm_sub_epi64(m, b1.m); return *this; }

  // 左シフト(縦型Bitboardでは左1回シフトで1段下の升に移動する)
  // ※　シフト演算子は歩の利きを求めるためだけに使う。
  Bitboard& operator <<= (int shift) { ASSERT_LV3(shift == 1); m = _mm_slli_epi64(m, shift); return *this; }

  // 右シフト(縦型Bitboardでは右1回シフトで1段上の升に移動する)
  Bitboard& operator >>= (int shift) { ASSERT_LV3(shift == 1); m = _mm_srli_epi64(m, shift); return *this; }

#else
  Bitboard& operator |= (const Bitboard& b1) { this->p[0] |= b1.p[0]; this->p[1] |= b1.p[1]; return *this; }
  Bitboard& operator &= (const Bitboard& b1) { this->p[0] &= b1.p[0]; this->p[1] &= b1.p[1]; return *this; }
  Bitboard& operator ^= (const Bitboard& b1) { this->p[0] ^= b1.p[0]; this->p[1] ^= b1.p[1]; return *this; }
  Bitboard& operator += (const Bitboard& b1) { this->p[0] += b1.p[0]; this->p[1] += b1.p[1]; return *this; }
  Bitboard& operator -= (const Bitboard& b1) { this->p[0] -= b1.p[0]; this->p[1] -= b1.p[1]; return *this; }

  Bitboard& operator <<= (int shift) { ASSERT_LV3(shift == 1); this->p[0] <<= shift; this->p[1] <<= shift; return *this; }
  Bitboard& operator >>= (int shift) { ASSERT_LV3(shift == 1); this->p[0] >>= shift; this->p[1] >>= shift; return *this; }
#endif


  // 比較演算子

  bool operator == (const Bitboard& rhs) const {
#ifdef USE_SSE41
    // 以下のようにすると2命令で済むらしい。
    // testing equality between two __m128i variables
    // cf.http://stackoverflow.com/questions/26880863/sse-testing-equality-between-two-m128i-variables
    __m128i neq = _mm_xor_si128(this->m, rhs.m);
    return _mm_test_all_zeros(neq, neq) ? true : false;
#else
    return (this->p[0] == rhs.p[0]) && (this->p[1] == rhs.p[1]);
#endif
  }

  bool operator != (const Bitboard& rhs) const { return !(*this == rhs); }

  // 2項演算子

  Bitboard operator & (const Bitboard& rhs) const { return Bitboard(*this) &= rhs; }
  Bitboard operator | (const Bitboard& rhs) const { return Bitboard(*this) |= rhs; }
  Bitboard operator ^ (const Bitboard& rhs) const { return Bitboard(*this) ^= rhs; }
  Bitboard operator + (const Bitboard& rhs) const { return Bitboard(*this) += rhs; }
  Bitboard operator << (const int i) const { return Bitboard(*this) <<= i; }
  Bitboard operator >> (const int i) const { return Bitboard(*this) >>= i; }

  // range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
  Square operator*() { return pop(); }
  void operator++() {}
};

// sqの升が1であるbitboard
extern Bitboard SquareBB[SQ_NB_PLUS1];
inline Bitboard::Bitboard(Square sq) { *this = SquareBB[sq]; }

// 全升が1であるBitboard
// p[0]の63bit目は0
extern Bitboard ALL_BB;

// 全升が0であるBitboard
extern Bitboard ZERO_BB;

// Square型との演算子
inline Bitboard operator|(const Bitboard& b, Square s) { return b | SquareBB[s]; }
inline Bitboard operator&(const Bitboard& b, Square s) { return b & SquareBB[s]; }
inline Bitboard operator^(const Bitboard& b, Square s) { return b ^ SquareBB[s]; }

// 単項演算子
// →　NOTで書くと、使っていないbit(p[0]のbit63)がおかしくなるのでALL_BBでxorしないといけない。
inline Bitboard operator ~ (const Bitboard& a) { return a ^ ALL_BB; }

// range-forで回せるようにするためのhack(少し遅いので速度が要求されるところでは使わないこと)
inline const Bitboard begin(const Bitboard& b) { return b; }
inline const Bitboard end(const Bitboard& b) { return ZERO_BB; }

// Bitboardの1の升を'*'、0の升を'.'として表示する。デバッグ用。
std::ostream& operator<<(std::ostream& os, const Bitboard& board);

// --------------------
//     Bitboard定数
// --------------------

// 各筋を表現するBitboard定数
extern Bitboard FILE1_BB;
extern Bitboard FILE2_BB;
extern Bitboard FILE3_BB;
extern Bitboard FILE4_BB;
extern Bitboard FILE5_BB;
extern Bitboard FILE6_BB;
extern Bitboard FILE7_BB;
extern Bitboard FILE8_BB;
extern Bitboard FILE9_BB;

// 各段を表現するBitboard定数
extern Bitboard RANK1_BB;
extern Bitboard RANK2_BB;
extern Bitboard RANK3_BB;
extern Bitboard RANK4_BB;
extern Bitboard RANK5_BB;
extern Bitboard RANK6_BB;
extern Bitboard RANK7_BB;
extern Bitboard RANK8_BB;
extern Bitboard RANK9_BB;

// 各筋を表現するBitboard配列
extern Bitboard FILE_BB[FILE_NB];

// 各段を表現するBitboard配列
extern Bitboard RANK_BB[RANK_NB];

// InFrontBBの定義)
//    c側の香の利き = 飛車の利き & InFrontBB[c][rank_of(sq)]
//
// すなわち、
// color == BLACKのとき、n段目よりWHITE側(1からn-1段目)を表現するBitboard。
// color == WHITEのとき、n段目よりBLACK側(n+1から9段目)を表現するBitboard。
// このアイデアはAperyのもの。
extern Bitboard InFrontBB[COLOR_NB][RANK_NB];
  
// 先手から見て1段目からr段目までを表現するBB(US==WHITEなら、9段目から数える)
inline const Bitboard rank1_n_bb(const Color US, const Rank r) { ASSERT_LV2(is_ok(r));  return InFrontBB[US][(US == BLACK ? r + 1 : 7 - r)]; }

// 敵陣を表現するBitboard。
inline const Bitboard enemy_field(const Color US) { return rank1_n_bb(US, RANK_3); }

// 歩が打てる筋を得るためのBitboard mask
extern Bitboard PAWN_DROP_MASK_BB[0x200][COLOR_NB];

// 2升に挟まれている升を返すためのテーブル(その2升は含まない)
extern Bitboard BetweenBB[SQ_NB_PLUS1][SQ_NB_PLUS1];

// 2升に挟まれている升を表すBitboardを返す。sq1とsq2が縦横斜めの関係にないときはZERO_BBが返る。
inline const Bitboard between_bb(Square sq1, Square sq2) { return BetweenBB[sq1][sq2]; }

// 2升を通過する直線を返すためのテーブル
extern Bitboard LineBB[SQ_NB_PLUS1][SQ_NB_PLUS1];

// 2升を通過する直線を返すためのBitboardを返す。sq1とsq2が縦横斜めの関係にないときはZERO_BBが返る。
inline const Bitboard line_bb(Square sq1, Square sq2) { return LineBB[sq1][sq2]; }

#if 0
// →　高速化のために、Effect8::directions_ofを使って実装しているのでコメントアウト。(shogi.hにおいて)
inline bool aligned(Square s1, Square s2, Square s3) {
	return LineBB[s1][s2] & s3;
}
#endif

// sqの升にいる敵玉に王手となるc側の駒ptの候補を得るテーブル。第2添字は(pr-1)を渡して使う。
extern Bitboard CheckCandidateBB[SQ_NB_PLUS1][HDK][COLOR_NB];

// sqの升にいる敵玉に王手となるus側の駒ptの候補を得る
// pr == ROOKは無条件全域なので代わりにHORSEで王手になる領域を返す。
// pr == KINGはsqの24近傍を返す。(ただしこれは王手生成では使わない)
inline const Bitboard check_candidate_bb(Color us, Piece pr, Square sq) { ASSERT_LV3(PAWN<= pr && pr <= HDK); return CheckCandidateBB[sq][pr - 1][us]; }

// ある升の24近傍のBitboardを返す。
inline const Bitboard around24_bb(Square sq) { return check_candidate_bb(BLACK, HDK, sq); }

// --------------------
//  Bitboard用の駒定数
// --------------------

// Bitboardの配列用の定数
// StepEffect[]の添字で使う。
enum PieceTypeBitboard
{
  PIECE_TYPE_BITBOARD_PAWN,
  PIECE_TYPE_BITBOARD_LANCE,
  PIECE_TYPE_BITBOARD_KNIGHT,
  PIECE_TYPE_BITBOARD_SILVER,
  PIECE_TYPE_BITBOARD_BISHOP,
  PIECE_TYPE_BITBOARD_ROOK,
  PIECE_TYPE_BITBOARD_GOLD,
  PIECE_TYPE_BITBOARD_HDK, // Horse , Dragon , King

  PIECE_TYPE_BITBOARD_NB = 8, // ビットボードを示すときにのみ使う定数

  // 以下、StepEffectで使う特殊な定数
  PIECE_TYPE_BITBOARD_QUEEN = 8,  // 馬+龍
  PIECE_TYPE_BITBOARD_CROSS00 = 9,   // 十字方向に1升
  PIECE_TYPE_BITBOARD_CROSS45 = 10,  // ×字方向に1升

  PIECE_TYPE_BITBOARD_NB2 = 16, // StepEffectに使う特殊な定数
};

// --------------------
// 利きのためのテーブル
// --------------------

// 利きのためのライブラリ
// Bitboardを用いるとソースコードが長くなるが、ソースコードのわかりやすさ、速度において現実的なのであえて使う。

// 近接駒の利き
// 3番目の添字がPIECE_TYPE_BITBOARD_LANCE,PIECE_TYPE_BITBOARD_BISHOP,PIECE_TYPE_BITBOARD_ROOK
// のときは、盤上の駒の状態を無視した(盤上に駒がないものとする)香・角・飛の利き。
// また、PIECE_TYPE_BITBOARD_QUEEN,PIECE_TYPE_BITBOARD_CROSS00,PIECE_TYPE_BITBOARD_CROSS45
// は、馬+龍,十字方向に1升,×字方向に1升となる。
extern Bitboard StepEffectsBB[SQ_NB_PLUS1][COLOR_NB][PIECE_TYPE_BITBOARD_NB2];

// --- 香の利き
extern Bitboard LanceEffect[COLOR_NB][SQ_NB_PLUS1][128];

// 指定した位置の属する file の bit を shift し、
// index を求める為に使用する。(from Apery)
extern int Slide[SQ_NB_PLUS1];

// --- 角の利き
extern Bitboard BishopEffect[20224+1];
extern Bitboard BishopEffectMask[SQ_NB_PLUS1];
extern int BishopEffectIndex[SQ_NB_PLUS1];

// --- 飛車の利き
extern Bitboard RookEffect[495616+1];
extern Bitboard RookEffectMask[SQ_NB_PLUS1];
extern int RookEffectIndex[SQ_NB_PLUS1];

// Haswellのpext()を呼び出す。occupied = occupied bitboard , mask = 利きの算出に絡む升が1のbitboard
// この関数で戻ってきた値をもとに利きテーブルを参照して、遠方駒の利きを得る。
inline uint64_t occupiedToIndex(const Bitboard& occupied, const Bitboard& mask) { return PEXT64(occupied.merge(), mask.merge()); }

// --------------------
//   大駒・小駒の利き
// --------------------

// --- 近接駒

// 王の利き
inline Bitboard kingEffect(const Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_HDK]; }

// 歩の利き
inline Bitboard pawnEffect(const Color color, const Square sq) { return StepEffectsBB[sq][color][PIECE_TYPE_BITBOARD_PAWN]; }

// 桂の利き
inline Bitboard knightEffect(const Color color, const Square sq) { return StepEffectsBB[sq][color][PIECE_TYPE_BITBOARD_KNIGHT]; }

// 銀の利き
inline Bitboard silverEffect(const Color color, const Square sq) { return StepEffectsBB[sq][color][PIECE_TYPE_BITBOARD_SILVER]; }

// 金の利き
inline Bitboard goldEffect(const Color color, const Square sq) { return StepEffectsBB[sq][color][PIECE_TYPE_BITBOARD_GOLD]; }

// --- 遠方仮想駒(盤上には駒がないものとして求める利き)

// 盤上の駒を無視するQueenの動き。
inline Bitboard queenStepEffect(const Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_QUEEN]; }

// 十字の利き 利き長さ=1升分。
inline Bitboard cross00StepEffect(Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_CROSS00]; }

// 斜め十字の利き 利き長さ=1升分。
inline Bitboard cross45StepEffect(Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_CROSS45]; }

// 盤上の駒を考慮しない香の利き
inline Bitboard lanceStepEffect(Color c, Square sq) { return StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_LANCE];}

// 盤上の駒を考慮しない角の利き
inline Bitboard bishopStepEffect(Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_BISHOP]; }

// 盤上の駒を考慮しない飛車の利き
inline Bitboard rookStepEffect(Square sq) { return StepEffectsBB[sq][BLACK][PIECE_TYPE_BITBOARD_ROOK];}

// --- 遠方駒(盤上の駒の状態を考慮しながら利きを求める)

// 香 : occupied bitboardを考慮しながら香の利きを求める
inline Bitboard lanceEffect(const Color c,const Square sq, const Bitboard& occupied) {
  const int index = (occupied.p[Bitboard::part(sq)] >> Slide[sq]) & 127;
  return LanceEffect[c][sq][index];
}

// 角 : occupied bitboardを考慮しながら角の利きを求める
inline Bitboard bishopEffect(const Square sq, const Bitboard& occupied) {
  const Bitboard block(occupied & BishopEffectMask[sq]);
  return BishopEffect[BishopEffectIndex[sq] + occupiedToIndex(block, BishopEffectMask[sq])];
}

// 馬 : occupied bitboardを考慮しながら香の利きを求める
inline Bitboard horseEffect(const Square sq, const Bitboard& occupied) { return bishopEffect(sq, occupied) | kingEffect(sq); }

// 飛 : occupied bitboardを考慮しながら香の利きを求める
inline Bitboard rookEffect(const Square sq, const Bitboard& occupied)
{
  const Bitboard block(occupied & RookEffectMask[sq]);
  return RookEffect[RookEffectIndex[sq] + occupiedToIndex(block, RookEffectMask[sq])];
}

// 龍 : occupied bitboardを考慮しながら香の利きを求める
inline Bitboard dragonEffect(const Square sq, const Bitboard& occupied){ return rookEffect(sq, occupied) | kingEffect(sq); }

// 上下にしか利かない飛車の利き
inline Bitboard rookEffectFile(const Square sq, const Bitboard& occupied) {
  const int index = (occupied.p[Bitboard::part(sq)] >> Slide[sq]) & 127;
  return LanceEffect[BLACK][sq][index] | LanceEffect[WHITE][sq][index];
}

// --------------------
//   汎用性のある利き
// --------------------

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。
// pc == QUEENだと馬+龍の利きが返る。
Bitboard effects_from(Piece pc, Square sq, const Bitboard& occ);

// --------------------
//   Bitboard tools
// --------------------

// 2bit以上あるかどうかを判定する。縦横斜め方向に並んだ駒が2枚以上であるかを判定する。この関係にないと駄目。
// この関係にある場合、Bitboard::merge()によって被覆しないことがBitboardのレイアウトから保証されている。
inline bool more_than_one(const Bitboard& bb) { ASSERT_LV2(!bb.cross_over()); return POPCNT64(bb.merge()) > 1; }

// shift()は、与えられた方向に添ってbitboardを1升ずつ移動させる。主に歩に対して用いる。
// SQ_Uを指定したときに、51の升は49の升に移動するので、注意すること。(51の升にいる先手の歩は存在しないので、
// 歩の移動に用いる分には問題ないはずではあるが。)

template<Square D>
inline Bitboard shift(Bitboard b) {
	ASSERT_LV3(D == SQ_U || D == SQ_D);

	// Apery型の縦型Bitboardにおいては歩の利きはbit shiftで済む。
	return  D == SQ_U ? b >> 1 : D == SQ_D ? b << 1
		: ZERO_BB;
}


#endif // #ifndef _BITBOARD_H_