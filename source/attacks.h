#ifndef ATTACKS_H_INCLUDED
#define ATTACKS_H_INCLUDED

#include <cassert>
#include <utility>

#include "misc.h"
#include "types.h"
#include "bitboard.h"

namespace YaneuraOu::Attacks {

void init();

#if !STOCKFISH
// Initialize attack tables after Bitboards::init().
// 利きテーブルを初期化するinit()は、Bitboards::init()のあとに呼ぶ。
// StockfishにはBitboards::init()がないが、将棋版では先に基本表の初期化が必要。
#if defined(ENABLE_TEST_CMD)
void test_sliding_attacks(Test::UnitTester& tester);
#endif

namespace SimdSliding {
struct alignas(64) BishopMasks { Bitboard ray[4]; };
static_assert(sizeof(BishopMasks) == 64);
extern BishopMasks bishopMasks[SQ_NB_PLUS1];

inline Bitboard bishop(Square sq, const Bitboard& occupied) {
#if defined(USE_AVX512)
    // Keep four directional masks in one cache line, in their final lane order.
    // 4方向のマスクを計算時のレーン順に並べ、1キャッシュラインに収める。
    const __m512i mask = _mm512_load_si512(&bishopMasks[sq]);
    const __m512i reverseUpper = _mm512_setr_epi64(
      0x0706050403020100LL, 0x0f0e0d0c0b0a0908LL,
      0x0706050403020100LL, 0x0f0e0d0c0b0a0908LL,
      0x08090a0b0c0d0e0fLL, 0x0001020304050607LL,
      0x08090a0b0c0d0e0fLL, 0x0001020304050607LL);
    const __m512i o = _mm512_and_si512(mask, _mm512_shuffle_epi8(
      _mm512_broadcast_i64x2(occupied.m), reverseUpper));
    // Compute independent 64-bit prefixes. Keep a high prefix only if its low word is empty.
    // 64bitごとに独立した範囲を求め、下位ワードが空のときだけ上位の結果を残す。
    // Odd bits shifted into low lanes are harmless because those lanes are always enabled.
    // 奇数bitが下位レーンへ移っても、下位レーンは常に有効なので結果に影響しない。
    const __mmask8 empty = _mm512_cmpeq_epi64_mask(o, _mm512_setzero_si512());
    const __mmask8 active = __mmask8(0x55) | _kshiftli_mask8(empty, 1);
    const __m512i decremented = _mm512_add_epi64(o, _mm512_set1_epi64(-1));
    const __m512i changed = _mm512_maskz_ternarylogic_epi64(active, o, decremented, mask, 0x28);
    const __m512i rays = _mm512_shuffle_epi8(changed, reverseUpper);
    const __m256i pairs = _mm256_or_si256(_mm512_castsi512_si256(rays),
                                         _mm512_extracti64x4_epi64(rays, 1));
    Bitboard result;
    result.m = _mm_or_si128(_mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1));
    return result;
#elif defined(USE_AVX2)
    const auto& masks = bishopMasks[sq];
    const __m256i forwardMask = _mm256_load_si256(reinterpret_cast<const __m256i*>(&masks.ray[0]));
    const __m256i backwardMask = _mm256_load_si256(reinterpret_cast<const __m256i*>(&masks.ray[2]));
    const __m256i reverse = _mm256_setr_epi64x(
      0x08090a0b0c0d0e0fLL, 0x0001020304050607LL,
      0x08090a0b0c0d0e0fLL, 0x0001020304050607LL);
    const __m256i o = _mm256_broadcastsi128_si256(occupied.m);
    const __m256i forward = _mm256_and_si256(forwardMask, o);
    const __m256i backward = _mm256_and_si256(backwardMask, _mm256_shuffle_epi8(o, reverse));
    const auto prefix = [](__m256i o, __m256i mask) {
        // Zero-filled low words are always active; each high word tests its own low word.
        // 0で埋めた下位ワードは常に有効とし、上位ワードは対応する下位ワードの空判定を使う。
        const __m256i active = _mm256_cmpeq_epi64(_mm256_slli_si256(o, 8), _mm256_setzero_si256());
        const __m256i changed = _mm256_xor_si256(o, _mm256_add_epi64(o, _mm256_set1_epi64x(-1)));
        return _mm256_and_si256(_mm256_and_si256(changed, mask), active);
    };
    const __m256i rays = _mm256_or_si256(prefix(forward, forwardMask),
      _mm256_shuffle_epi8(prefix(backward, backwardMask), reverse));
    Bitboard result;
    result.m = _mm_or_si128(_mm256_castsi256_si128(rays), _mm256_extracti128_si256(rays, 1));
    return result;
#else
    // Use the same directional masks without requiring SIMD instructions.
    // SIMD命令を必要としない経路でも、同じ方向別マスクを使う。
    const auto prefix = [](const Bitboard& o, const Bitboard& mask) {
        const u64 low = o.extract64<0>() & mask.extract64<0>();
        const u64 high = o.extract64<1>() & mask.extract64<1>();
        return Bitboard((low ^ (low - 1)) & mask.extract64<0>(),
          (high ^ (high - 1)) & mask.extract64<1>() & (u64(0) - u64(low == 0)));
    };
    const auto& masks = bishopMasks[sq];
    const Bitboard reversed = occupied.byteswap_bb();
    const Bitboard forward = prefix(occupied, masks.ray[0]) | prefix(occupied, masks.ray[1]);
    const Bitboard backward = prefix(reversed, masks.ray[2]) | prefix(reversed, masks.ray[3]);
    return forward | backward.byteswap_bb();
#endif
}

} // namespace SimdSliding
#endif

#if STOCKFISH
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
#else
// 2升を通過する直線。[0]:右上から左下、[1]:横、[2]:左上から右下、[3]:縦。
// StockfishのLineBB[s1][s2]とは添字の意味が異なり、将棋版はLineBB[升][方向]。
// この配列には直接アクセスせず、line_bb()を使うこと。
extern Bitboard LineBB[SQ_NB][4];

// 2升に挟まれている升を返すためのテーブル(その2升は含まない)。
// StockfishのBetweenBB[s1][s2]は終点込みの直接表。将棋版は両端を除く圧縮表。
// 同一のBitboardを共有して縮小してあり、between_bb()から参照する。
extern Bitboard BetweenBB[638];
extern u16 BetweenIndex[SQ_NB_PLUS1][SQ_NB_PLUS1];
#endif

// horizontal/vertical/diagonal line passing through s1 and s2
// s1とs2を通る縦・横・斜めの直線。
inline Bitboard line_bb(Square s1, Square s2) {
#if STOCKFISH
    assert(is_ok(s1) && is_ok(s2));
    return LineBB[s1][s2];
#else
    // Stockfish版は同一直線にない有効な2升や同じ升の組なら空を返す。
    // 将棋版は異なる2升が縦横斜めの関係にあることが前提で、それ以外の入力では空を保証しない。
    // この前提で、4方向へ圧縮した表を参照する。
    static_assert(Effect8::DIRECT_RU == 0 && Effect8::DIRECT_LD == 7, "");
    auto directions = Effect8::directions_of(s1, s2);
    ASSERT_LV3(directions != 0);
    static const int a[8] = {0, 1, 2, 3, 3, 2, 1, 0};
    return LineBB[s1][a[(int)Effect8::pop_directions(directions)]];
#endif
}

// horizontal/vertical/diagonal line segment from s1 to s2,
// excluding s1 but including s2
// Stockfishでは、s1を含まずs2を含む、s1からs2までの縦・横・斜めの線分。
// 将棋版とは端点の扱いが異なる。下記#elseの仕様を参照。
inline Bitboard between_bb(Square s1, Square s2) {
#if STOCKFISH
    assert(is_ok(s1) && is_ok(s2));
    return BetweenBB[s1][s2];
#else
    // Stockfishとの仕様差: 将棋版はs1・s2の両端を含まない。
    // 縦横斜めの関係にない組やs1 == s2では空。Stockfish版はその場合もs2だけを返す。
    // 将棋版はどちらかがSQ_NBでも空を返すが、Stockfish版は両方とも盤上の升を要求する。
    // 終点の駒を数えないピン計算や合駒生成が依存するため、単純に終点込みへ変えてはならない。
    return BetweenBB[BetweenIndex[s1][s2]];
#endif
}

#if !STOCKFISH
// --------------------
// 利きのためのテーブル
// --------------------

// 利きのためのライブラリ
// 注意) ここのテーブルを直接参照せず、kingEffect()など、利きの関数を経由して用いること。

// --- 近接駒の利き

// 具体的なPiece名を指定することがほとんどなので1本の配列になっているメリットがあまりないので配列を分ける。
// 利き関連のテーブルはAttacks名前空間にまとめる。
extern Bitboard KingEffectBB[SQ_NB_PLUS1];
extern Bitboard GoldEffectBB[SQ_NB_PLUS1][COLOR_NB];
extern Bitboard SilverEffectBB[SQ_NB_PLUS1][COLOR_NB];
extern Bitboard KnightEffectBB[SQ_NB_PLUS1][COLOR_NB];
extern Bitboard PawnEffectBB[SQ_NB_PLUS1][COLOR_NB];

// 盤上の駒をないものとして扱う、遠方駒の利き。香、角、飛
extern Bitboard LanceStepEffectBB[SQ_NB_PLUS1][COLOR_NB];
extern Bitboard BishopStepEffectBB[SQ_NB_PLUS1];
extern Bitboard RookStepEffectBB[SQ_NB_PLUS1];


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
	return PawnEffectBB[sq][c];
}

// 歩の利き、非template版。
inline Bitboard pawnEffect(Color c, Square sq)
{
	return (c == BLACK) ? pawnEffect<BLACK>(sq) : pawnEffect<WHITE>(sq);
}

// 桂の利き
// これは遮断されることはないのでOccupiedBitboard不要。
inline Bitboard knightEffect(const Color c, const Square sq)
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return KnightEffectBB[sq][c];
}

// ↑のtemplate版。
template <Color C>
inline Bitboard knightEffect(const Square sq)
{
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return KnightEffectBB[sq][C];
}

// 銀の利き
inline Bitboard silverEffect(const Color c, const Square sq)
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return SilverEffectBB[sq][c];
}

// ↑のtemplate版
template <Color C>
inline Bitboard silverEffect(const Square sq)
{
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return SilverEffectBB[sq][C];
}

// 金の利き
inline Bitboard goldEffect(const Color c, const Square sq) {
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);
	return GoldEffectBB[sq][c];
}

// ↑のtemplate版
template <Color C>
inline Bitboard goldEffect(const Square sq) {
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return GoldEffectBB[sq][C];
}

// 王の利き
// 王の利きは先後の区別はない。
inline Bitboard kingEffect(const Square sq)
{
	ASSERT_LV3(sq <= SQ_NB);
	return KingEffectBB[sq];
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
	return LanceStepEffectBB[sq][c];
}
// ↑のtemplate版。
template <Color C>
inline Bitboard lanceStepEffect(Square sq) {
	ASSERT_LV3(is_ok(C) && sq <= SQ_NB);
	return LanceStepEffectBB[sq][C];
}

// 盤上の駒を考慮しない角の利き
inline Bitboard bishopStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return BishopStepEffectBB[sq];
}

// 盤上の駒を考慮しない飛車の利き
inline Bitboard rookStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	return RookStepEffectBB[sq];
}

// 盤上の駒を考慮しない馬の利き
inline Bitboard horseStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	// わざわざ用意するほどでもないので玉の利きと合成しておく。
	return BishopStepEffectBB[sq] | KingEffectBB[sq];
}

// 盤上の駒を考慮しない龍の利き
inline Bitboard dragonStepEffect(Square sq) {
	ASSERT_LV3(sq <= SQ_NB);
	// わざわざ用意するほどでもないので玉の利きと合成しておく。
	return RookStepEffectBB[sq] | KingEffectBB[sq];
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

	// Qugiyのアルゴリズムを1～7筋と8～9筋に分けたもの。
	// こうすることで飛車の縦利きのコードがちょっと短くなる。
	// Qugiy[WCSC31]の手法 : cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal.pdf
	// 1本の筋は片側の64bit内に収まるので、part()で該当wordだけを扱う。
	if (C == WHITE)
	{
		// 後手の香はbit番号が増える方向。対象範囲の障害物bitをemとして1を引く。
		// em      : ...1 0 0 0
		// em - 1  : ...0 1 1 1
		// 両者のXORは最初の障害物まで1となるので、maskで対象の筋と向きだけを残す。
		// 障害物がなければ符号なし減算で全bitが1となり、盤端までの利きが得られる。

		if (Bitboard::part(sq) == 0)
		{
			// 香がp[0]に属する
			u64 mask = lanceStepEffect<WHITE>(sq).template extract64<0>();
			u64 em = occupied.extract64<0>() & mask;
			u64 t = em - 1; // 1引き算すれば、桁借りが上位桁が1のところまで波及する。
			return Bitboard((em ^ t) & mask, 0);
		}
		else {
			// 香がp[1]に属する

			u64 mask = lanceStepEffect<WHITE>(sq).template extract64<1>();
			u64 em = occupied.extract64<1>() & mask;
			u64 t = em - 1;
			return Bitboard(0, (em ^ t) & mask);
		}
	}
	else {
		// 先手の香はbit番号が減る方向。対象範囲の最上位の障害物bitまでを残す。
		// mocc | 1は障害物がない場合にもMSB64()を使えるようにするための番兵。
		// 最後にseでマスクし、対象範囲の外や番兵による不要なbitを落とす。

		if (Bitboard::part(sq) == 0)
		{
			// 香がp[0]に属する
			u64 se = lanceStepEffect<C>(sq).template extract64<0>();
			u64 mocc = se & occupied.extract64<0>();
			// 香が当たる駒より上の升に対応するビットを0、それ以外を1にする
			mocc = ~uint64_t{0} << MSB64(mocc | 1);
			return Bitboard(mocc & se, 0);
		}
		else {
			// 香がp[1]に属する
			u64 se = lanceStepEffect<C>(sq).template extract64<1>();
			u64 mocc = se & occupied.extract64<1>();
			mocc = ~uint64_t{0} << MSB64(mocc | 1);
			return Bitboard(0, mocc & se);
		}
	}
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
		mocc = ~uint64_t{0} << MSB64(mocc | 1);

		// 後手の香の利きと先手の香の利きを合成
		return Bitboard(((em ^ t) & mask) | (mocc & se), 0);
	}
	else {
		// 飛車がp[1]に属する
		// ↑の処理と同様。

		u64 mask = lanceStepEffect<WHITE>(sq).template extract64<1>();
		u64 em = occupied.extract64<1>() & mask;
		u64 t = em - 1;

		u64 se = lanceStepEffect<BLACK>(sq).template extract64<1>();
		u64 mocc = se & occupied.extract64<1>();
		mocc = ~uint64_t{0} << MSB64(mocc | 1);

		return Bitboard(0, ((em ^ t) & mask) | (mocc & se));
	}
}

// ==== 飛車と角の利き ===

// 飛車の横の利き
Bitboard rookRankEffect(Square sq, const Bitboard& occupied);

// 飛車の利き
inline Bitboard rookEffect(const Square sq, const Bitboard& occupied) {
	// 縦の利きと横の利きを合成する。
	return rookRankEffect(sq, occupied) | rookFileEffect(sq, occupied);
}

// 角の利き
Bitboard bishopEffect(const Square sq, const Bitboard& occupied);

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

// 角と飛車の利きを同時に求める呼び出しは、both_attacks_bb()にまとめる。

// === 大駒の部分利き(SEEなどで用いる) ===

// sqの升から各方向への利き
// 右上、右、右下、上方向は、byteswap_bb()してあるので、普通の利きではないから注意。
// 6方向しか使っていないので詰めてある。
extern Bitboard QUGIY_STEP_EFFECT[Effect8::DIRECT_NB - 2][SQ_NB_PLUS1];


// 方向利き
template <Effect8::Direct D>
Bitboard rayEffect(Square sq, const Bitboard& occupied)
{
	// DIRECT_Uはbyteswap_bb()しても正しく求められないからlanceEffectを用いる。
	// DIRECT_Dも、lanceEffectにこれ専用の高速なコードが書いてあるのでそれを用いる。
	if (D == Effect8::DIRECT_U) return lanceEffect<BLACK>(sq, occupied);
	if (D == Effect8::DIRECT_D) return lanceEffect<WHITE>(sq, occupied);


	ASSERT_LV3(D < Effect8::DIRECT_NB);

	// 6方向しか使っていないのでテーブルを詰めてある。
	const int dd = D > Effect8::DIRECT_D ? D - 2 : D;
	const Bitboard mask = QUGIY_STEP_EFFECT[dd][sq];

	// 右上、右、右下、上方向(Squareとしてみた時に、値が減る方向)
	constexpr bool reverse =
		   D == Effect8::DIRECT_RU
		|| D == Effect8::DIRECT_R
		|| D == Effect8::DIRECT_RD;

	Bitboard bb = occupied;

	if (reverse)
		bb = bb.byteswap_bb();

	// 利きに関係する升のoccupiedだけを抽出
	bb &= mask;
	// 1減算することにより、利きが通る升までが変化する。
	Bitboard bb_minus_one = bb.decrement();
	// 変化したbitを抽出してmaskすれば出来上がり。
	bb = (bb ^ bb_minus_one) & mask;

	if (reverse)
		bb = bb.byteswap_bb();

	return bb;
}

// sqの升から指定した方向dへの利き。盤上の駒も考慮する。
Bitboard directEffect(Square sq, Effect8::Direct d, const Bitboard& occupied);

// sqの升にいる敵玉に王手となるc側の駒ptの候補を得るテーブル。第2添字は(pr-1)を渡して使う。
// 直接アクセスせずに、check_candidate_bb()、around24_bb()を用いてアクセスすること。
extern Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING - 1][COLOR_NB];


// sqの升にいる敵玉に王手となるus側の駒ptの候補を得る
// pr == ROOKは無条件全域なので代わりにHORSEで王手になる領域を返す。
// pr == KINGで呼び出してはならない。それは、around24_bb()のほうを用いる。
static const Bitboard check_candidate_bb(Color us, PieceType pr, Square sq)
{
	ASSERT_LV3(PAWN<= pr && pr < KING && sq <= SQ_NB && is_ok(us));
	return CheckCandidateBB[sq][pr - 1][us];
}

#if defined(LONG_EFFECT_LIBRARY) // 詰みルーチンでは使わなくなったが、LONG_EFFECT_LIBRARYのなかで使っている。

extern Bitboard CheckCandidateKingBB[SQ_NB_PLUS1];


// ある升の24近傍のBitboardを返す。
static const Bitboard around24_bb(Square sq)
{
	ASSERT_LV3(sq <= SQ_NB);
	return CheckCandidateKingBB[sq];
}
#endif
#endif

// Returns the pseudo attacks of the given piece type
// assuming an empty board.
// 盤上に他の駒がないものとして、指定した駒種の利きを返す。
#if STOCKFISH
sf_always_inline Bitboard attacks_bb(PieceType pt, Square s, Color c = COLOR_NB) {

    assert((pt != PAWN || c < COLOR_NB) && is_ok(s));
    return pt == PAWN ? PseudoAttacks[c][s] : PseudoAttacks[pt][s];
}
#else
// Stockfish版はPieceTypeと、歩の場合にColorを渡す。将棋版は先後を含むPieceを渡す。
// 将棋では香・桂・銀・金などにも向きがあるため、PieceTypeだけでは向きが確定しない。
// B_GOLDS / W_GOLDSは、従来どおり金の利きとして扱う。
Bitboard attacks_bb(Piece pc, Square s);
#endif

// Returns the attacks by the given piece
// assuming the board is occupied according to the passed Bitboard.
// Sliding piece attacks do not continue past an occupied square.
// 渡されたBitboardを盤上の占有状態として、指定した駒の利きを返す。
// 遠方駒の利きは、駒がある升を越えては進まない。
#if STOCKFISH
sf_always_inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {

    assert(pt != PAWN && is_ok(s));

    [[maybe_unused]] const auto [bishop, rook] = dual_magic(s).both_attacks_bb(occupied);

    switch (pt)
    {
    case BISHOP :
        return bishop;
    case ROOK :
        return rook;
    case QUEEN :
        return bishop | rook;
    default :
        return PseudoAttacks[pt][s];
    }
}
#else
// Stockfish版はPieceTypeを指定するが、将棋版は先後を含むPieceを指定する。
// Stockfishの占有あり版は歩を対象外とするが、将棋版は歩にも対応する。
// 盤上の駒による遠方駒の利きの遮断も考慮する。
Bitboard attacks_bb(Piece pc, Square s, const Bitboard& occupied);
#endif

inline std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
#if STOCKFISH
    return dual_magic(s).both_attacks_bb(occupied);
#else
    // 角、飛車の順で返す。既存のSIMD計算とマスク配置は維持する。
    // Stockfish版の入力は盤上の升。将棋版はSQ_NBでも両方が空のBitboardとなる。
    return {bishopEffect(s, occupied), rookEffect(s, occupied)};
#endif
}

} // namespace YaneuraOu::Attacks

#endif  // #ifndef ATTACKS_H_INCLUDED
