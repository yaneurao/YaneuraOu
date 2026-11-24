#include "attacks.h"

#include "misc.h"

#if !STOCKFISH
#include "extra/long_effect.h"
#include "mate/mate.h"
#endif

namespace YaneuraOu::Attacks {

#if STOCKFISH
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
#else
// 配列を縮小しているので、升の組を方向または圧縮indexへ変換して参照する。
Bitboard LineBB[SQ_NB][4];
Bitboard BetweenBB[785];
u16 BetweenIndex[SQ_NB_PLUS1][SQ_NB_PLUS1];
#endif


#if !STOCKFISH
// 玉、金、銀、桂、歩の利き
Bitboard KingEffectBB[SQ_NB_PLUS1];
Bitboard GoldEffectBB[SQ_NB_PLUS1][COLOR_NB];
Bitboard SilverEffectBB[SQ_NB_PLUS1][COLOR_NB];
Bitboard KnightEffectBB[SQ_NB_PLUS1][COLOR_NB];
Bitboard PawnEffectBB[SQ_NB_PLUS1][COLOR_NB];

// 盤上の駒をないものとして扱う、遠方駒の利き。香、角、飛
Bitboard LanceStepEffectBB[SQ_NB_PLUS1][COLOR_NB];
Bitboard BishopStepEffectBB[SQ_NB_PLUS1];
Bitboard RookStepEffectBB[SQ_NB_PLUS1];


Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING - 1][COLOR_NB];

#if defined(LONG_EFFECT_LIBRARY) // 詰みルーチンでは使わなくなったが、LONG_EFFECT_LIBRARYのなかで使っている。
Bitboard CheckCandidateKingBB[SQ_NB_PLUS1];
#endif


// ----------------------------------------------------------------------------------------------
// 飛車・角の利きのためのテーブル
// ----------------------------------------------------------------------------------------------

Bitboard QUGIY_ROOK_MASK[SQ_NB_PLUS1][2];

// sqの升から各方向への利き
// 右上、右、右下、上方向は、byte_reverse()してあるので、普通の利きではないから注意。
// 6方向しか使っていないの、ちょっともったいないので詰める。
Bitboard QUGIY_STEP_EFFECT[Effect8::DIRECT_NB - 2][SQ_NB_PLUS1];

SimdSliding::BishopMasks SimdSliding::bishopMasks[SQ_NB_PLUS1];

using namespace BB_Table;
#endif

void init()
{
	// 1) direct_tableの初期化

	for (auto sq1 : SQ)
		for (auto dir = Effect8::DIRECT_ZERO; dir < Effect8::DIRECT_NB; ++dir)
		{
			// dirの方角に壁にぶつかる(盤外)まで延長していく。このとき、sq1から見てsq2のDirectionsは (1 << dir)である。
			auto delta = Effect8::DirectToDeltaWW(dir);
			for (auto sq2 = to_sqww(sq1) + delta; is_ok(sq2); sq2 += delta)
				Effect8::direc_table[sq1][sqww_to_sq(sq2)] = Effect8::to_directions(dir);
		}


    // 2) Build shared directional masks from board coordinates.
    // 盤の座標から共通の方向別マスクを作る。
    for (int index = 0; index <= SQ_NB; ++index) {
        auto& bishop = SimdSliding::bishopMasks[index];
        for (auto& ray : bishop.ray) ray = Bitboard(ZERO);
        if (index == SQ_NB) continue;
        const Square sq = Square(index);
        const auto ray = [sq](int df, int dr) {
            Bitboard b(ZERO);
            for (int f = int(file_of(sq)) + df, r = int(rank_of(sq)) + dr;
                 0 <= f && f < 9 && 0 <= r && r < 9; f += df, r += dr)
                b |= Bitboard(File(f) | Rank(r));
            return b;
        };
        bishop.ray[0] = ray(1, -1);
        bishop.ray[1] = ray(1, 1);
        bishop.ray[2] = ray(-1, -1).byte_reverse();
        bishop.ray[3] = ray(-1, 1).byte_reverse();
    }

	// 3) Qugiyの飛車のBitboardテーブルの初期化

	for (File f = FILE_1; f <= FILE_9; ++f) {
		for (Rank r = RANK_1; r <= RANK_9; ++r) {

			Bitboard left(ZERO), right(ZERO);

			// SQの升から左方向
			for (File f2 = (File)(f + 1); f2 <= FILE_9; ++f2)
				left |= Bitboard(f2 | r);

			// SQの升から右方向
			for (File f2 = (File)(f - 1); f2 >= FILE_1; --f2)
				right |= Bitboard(f2 | r);

			Bitboard right_rev = right.byte_reverse();

			Bitboard hi, lo;
			Bitboard::unpack(right_rev, left, hi, lo);

			QUGIY_ROOK_MASK[f | r][0] = lo;
			QUGIY_ROOK_MASK[f | r][1] = hi;
		}
	}



	// 4) Qugiyのアルゴリズムで用いる、sqの升から各方向への利きテーブルの初期化
	// DIRECT_UとDIRECT_Dを用いないので、これをskipする。
	int dd = 0;
	for (auto d = Effect8::DIRECT_ZERO; d < Effect8::DIRECT_NB ; d++)
	{
		if (d == Effect8::DIRECT_D || d == Effect8::DIRECT_U)
			continue;

		auto delta = Effect8::DirectToDeltaWW(d);

		// 右上、右、右下、上方向(Squareとしてみた時に、値が減る方向)
		bool reverse = sqww_to_sq(to_sqww(SQ_55) + delta) < SQ_55;

		for (auto sq : SQ)
		{
			Bitboard bb(ZERO);

			// 壁に突き当たるまで進む
			for (auto sq2 = to_sqww(sq) + delta; is_ok(sq2); sq2 += delta)
				bb |= Bitboard(sqww_to_sq(sq2));

			if (reverse)
				bb = bb.byte_reverse();

			QUGIY_STEP_EFFECT[dd][sq] = bb;
		}

		dd++;
	}

	// 5) 近接駒(+盤上の利きを考慮しない駒)のテーブルの初期化。
	// なるべく他の駒の利きに依存しない形で初期化する。

	// 歩と香のstep effect
	// ※　step effectとは、障害物がないときの利きのこと。
	for (auto c : COLOR)
		for (auto sq : SQ)
		{
			// 香
			// 香のstep effectは、筋のBitboardとsqより前の段を表現するBitboardを用いて、他の駒の利きに依存せず初期化できる。

			LanceStepEffectBB[sq][c] = FILE_BB[file_of(sq)] & ForwardRanksBB[c][rank_of(sq)];

			// 歩

			// 備考) 歩の利きは駒が敷き詰められている時の香の利きとして定義できる。
			//  PawnEffectBB[sq][c] = lanceEffect(c, sq, Bitboard(1));
			// → QugiyのアルゴリズムでlanceEffectの計算にpawnEffectを用いるようになったのでこれができなくなった。

			// 歩の利きは何段目であるか。
			Rank r = (Rank)(rank_of(sq) + (c == BLACK ? -1 : +1));
			// その段数が1～9段目に収まるなら..
			PawnEffectBB[sq][c] = (RANK_1 <= r && r <= RANK_9) ? Bitboard(file_of(sq) | r) : Bitboard(ZERO);
		}

	// 備考) ここでlanceEffectが使えるようになったので、以降、rookEffectが使える。
	// (rookEffectは縦利きを求めるのにlanceEffectに依存している)
	// ※　bishopEffectは、上でQugiyのテーブルを初期化した時以降使える。
	// 以下では、rookEffect,bishopEffect,lanceEffectを用いて残りの駒の利きを定義する。

	// 先後の区別のないstep effect (玉、角、飛)
	for (auto sq : SQ)
	{
		// 玉
		// 玉の利きは駒が敷き詰められている時の角と飛車の利きの重ね合わせとして定義できる。
		KingEffectBB[sq] = bishopEffect(sq, Bitboard(1)) | rookEffect(sq, Bitboard(1));

		// 角
		// 角のstep effectは、盤上に駒がない時の角の利き。
		BishopStepEffectBB[sq] = bishopEffect(sq, Bitboard(0));

		// 飛
		// 飛車のstep effectは、盤上に駒がない時の飛車の利き。
		RookStepEffectBB[sq]   = rookEffect(sq, Bitboard(0));
	}

	// 先後の区別のあるstep effect (桂、銀、金)
	for (auto c : COLOR)
		for (auto sq : SQ)
		{
			// 桂
			// 桂の利きは、歩の利きの地点に長さ1の角の利きを作って、前方のみ残す。
			Bitboard tmp(ZERO);
			Bitboard pawn = lanceEffect(c, sq, Bitboard(1));

			if (pawn)
			{
				Square sq2 = pawn.pop();
				Bitboard pawn2 = lanceEffect(c, sq2, Bitboard(1)); // さらに1つ前
				if (pawn2)
					tmp = bishopEffect(sq2, Bitboard(1)) & RANK_BB[rank_of(pawn2.pop())];
			}
			KnightEffectBB[sq][c] = tmp;

			// 銀
			// 銀の利きは長さ1の角の利きと長さ1の香の利きの合成として定義できる。
			SilverEffectBB[sq][c] = lanceEffect(c, sq, Bitboard(1)) | bishopEffect(sq, Bitboard(1));

			// 金
			// 金の利きは長さ1の角と飛車の利き。ただし、角のほうは相手側の歩の行き先の段でmaskしてしまう。
			Bitboard e_pawn = lanceEffect(~c, sq, Bitboard(1));
			Bitboard mask(ZERO);
			if (e_pawn)
				mask = RANK_BB[rank_of(e_pawn.pop())];
			GoldEffectBB[sq][c]= (bishopEffect(sq, Bitboard(1)) & ~mask) | rookEffect(sq, Bitboard(1));

			// --- 以下のbitboard、あまり頻繁に用いないので他のbitboardを合成して代用する。

			// 盤上の駒がないときのqueenの利き
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_QUEEN] = bishopEffect(sq, Bitboard(ZERO)) | rookEffect(sq, Bitboard(ZERO));

			// 長さ1の十字
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS00] = rookEffect(sq, Bitboard(1));

			// 長さ1の斜め
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS45] = bishopEffect(sq, Bitboard(1));
		}

	// 6) BetweenBB , LineBBの初期化
	{
		u16 between_index = 1;
		// BetweenBB[0] == Bitboard(ZERO)であることを保証する。

		for (auto s1 : SQ)
			for (auto s2 : SQ)
			{
				// 十字方向か、斜め方向かだけを判定して、例えば十字方向なら
				// rookEffect(sq1,Bitboard(s2)) & rookEffect(sq2,Bitboard(s1))
				// のように初期化したほうが明快なコードだが、この初期化をそこに依存したくないので愚直にやる。

				// これについてはあとで設定する。
				if (s1 >= s2)
					continue;

				// 方角を用いるテーブルの初期化
				if (Effect8::directions_of(s1, s2))
				{
					Bitboard bb(ZERO);
					// 間に挟まれた升を1に
					Square delta = (s2 - s1) / dist(s1, s2);
					for (Square s = s1 + delta; s != s2; s += delta)
						bb |= s;

					// Bitboard(ZERO)なら、このindexとしては0を指しておけば良いので書き換える必要ない。
					if (!bb)
						continue;

					BetweenIndex[s1][s2] = between_index;
					BetweenBB[between_index++] = bb;
				}
			}

		ASSERT_LV1(between_index == 785);

		// 対称性を考慮して、さらにシュリンクする。
		for (auto s1 : SQ)
			for (auto s2 : SQ)
				if (s1 > s2)
					BetweenIndex[s1][s2] = BetweenIndex[s2][s1];

	}
	for (auto s1 : SQ)
		for (int d = 0; d < 4; ++d)
		{
			// BishopEffect0 , RookRankEffect , BishopEffect1 , RookFileEffectを用いて初期化したほうが
			// 明快なコードだが、この初期化をそこに依存したくないので愚直にやる。

			const Square deltas[4] = { SQ_RU , SQ_R , SQ_RD , SQ_U };
			const Square delta = deltas[d];
			Bitboard bb = Bitboard(s1);

			// 壁に当たるまでs1から-delta方向に延長
			for (Square s = s1; dist(s, s - delta) <= 1; s -= delta) bb |= (s - delta);

			// 壁に当たるまでs1から+delta方向に延長
			for (Square s = s1; dist(s, s + delta) <= 1; s += delta) bb |= (s + delta);

			LineBB[s1][d] = bb;
		}


	// 7) 王手となる候補の駒のテーブル初期化(王手の指し手生成に必要)

#define FOREACH_KING(BB, EFFECT ) { for(auto sq : BB){ target|= EFFECT(sq); } }
#define FOREACH(BB, EFFECT ) { for(auto sq : BB){ target|= EFFECT(them,sq); } }
#define FOREACH_BR(BB, EFFECT ) { for(auto sq : BB) { target|= EFFECT(sq,Bitboard(ZERO)); } }

	for (auto Us : COLOR)
		for (auto ksq : SQ)
		{
			const Color them = ~Us;
			auto enemyGold = goldEffect(them, ksq) & enemy_field(Us);
			Bitboard target(ZERO);

			// 歩で王手になる可能性のあるものは、敵玉から２つ離れた歩(不成での移動) + ksqに敵の金をおいた範囲(enemyGold)に成りで移動できる
			FOREACH(pawnEffect(them, ksq), pawnEffect);
			FOREACH(enemyGold, pawnEffect);
			CheckCandidateBB[ksq][PAWN - 1][Us] = target & ~Bitboard(ksq);

			// 香で王手になる可能性のあるものは、ksqに敵の香をおいたときの利き。(盤上には何もないものとする)
			// と、王が1から3段目だと成れるので王の両端に香を置いた利きも。
			target = lanceStepEffect(them, ksq);
			if (enemy_field(Us) & ksq)
			{
				if (file_of(ksq) != FILE_1)
					target |= lanceStepEffect(them, ksq + SQ_R);
				if (file_of(ksq) != FILE_9)
					target |= lanceStepEffect(them, ksq + SQ_L);
			}
			CheckCandidateBB[ksq][LANCE - 1][Us] = target;

			// 桂で王手になる可能性のあるものは、ksqに敵の桂をおいたところに移動できる桂(不成) + ksqに金をおいた範囲(enemyGold)に成りで移動できる桂
			target = Bitboard(ZERO);
			FOREACH(knightEffect(them, ksq) | enemyGold, knightEffect);
			CheckCandidateBB[ksq][KNIGHT - 1][Us] = target & ~Bitboard(ksq);

			// 銀も同様だが、2,3段目からの引き成りで王手になるパターンがある。(4段玉と5段玉に対して)
			target = Bitboard(ZERO);
			FOREACH(silverEffect(them, ksq), silverEffect);
			FOREACH(enemyGold, silverEffect); // 移動先が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			FOREACH(goldEffect(them, ksq), enemy_field(Us) & silverEffect); // 移動元が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			CheckCandidateBB[ksq][SILVER - 1][Us] = target & ~Bitboard(ksq);

			// 金
			target = Bitboard(ZERO);
			FOREACH(goldEffect(them, ksq), goldEffect);
			CheckCandidateBB[ksq][GOLD - 1][Us] = target & ~Bitboard(ksq);

			// 角
			target = Bitboard(ZERO);
			FOREACH_BR(bishopEffect(ksq, Bitboard(ZERO)), bishopEffect);
			FOREACH_BR(kingEffect(ksq) & enemy_field(Us), bishopEffect); // 移動先が敵陣 == 成れる == 王の動き
			FOREACH_BR(kingEffect(ksq), enemy_field(Us) & bishopEffect); // 移動元が敵陣 == 成れる == 王の動き
			CheckCandidateBB[ksq][BISHOP - 1][Us] = target & ~Bitboard(ksq);

			// 飛・龍は無条件全域。
			// ROOKのところには馬のときのことを格納

			// 馬
			target = Bitboard(ZERO);
			FOREACH_BR(horseEffect(ksq, Bitboard(ZERO)), horseEffect);
			CheckCandidateBB[ksq][ROOK - 1][Us] = target & ~Bitboard(ksq);

#if 0
			// 王(24近傍が格納される)
			target = Bitboard(ZERO);
			FOREACH_KING(kingEffect(ksq), kingEffect);
			CheckCandidateKingBB[ksq] = target & ~Bitboard(ksq);
#endif
		}

	// 以下はBitboardとは関係はないが、Bitboardが初期化されていないと初期化できないので
	// ここから初期化しておいてやる。

	// 8) LONG_EFFECT_LIBRARYの初期化

#if defined (LONG_EFFECT_LIBRARY)
	LongEffect::init();
#endif

	// 9) 1手詰めテーブルの初期化
#if defined (USE_MATE_1PLY)
	Mate::init();
#endif

}

#if !STOCKFISH

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。(step effect)
// 盤上には他の駒がないものとして考える。B_GOLDS / W_GOLDSは金の利き。
Bitboard attacks_bb(Piece pc, Square s)
{
	switch (pc)
	{
	case B_PAWN:   return pawnEffect     <BLACK>(s);
	case B_LANCE:  return lanceStepEffect<BLACK>(s);
	case B_KNIGHT: return knightEffect   <BLACK>(s);
	case B_SILVER: return silverEffect   <BLACK>(s);
	case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: case B_GOLDS: return goldEffect<BLACK>(s);

	case W_PAWN:   return pawnEffect     <WHITE>(s);
	case W_LANCE:  return lanceStepEffect<WHITE>(s);
	case W_KNIGHT: return knightEffect   <WHITE>(s);
	case W_SILVER: return silverEffect   <WHITE>(s);
	case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: case W_GOLDS: return goldEffect<WHITE>(s);

		//　先後同じ移動特性の駒
	case B_BISHOP: case W_BISHOP: return bishopStepEffect(s);
	case B_ROOK:   case W_ROOK:   return rookStepEffect  (s);
	case B_HORSE:  case W_HORSE:  return horseStepEffect (s);
	case B_DRAGON: case W_DRAGON: return dragonStepEffect(s);
	case B_KING:   case W_KING:   return kingEffect      (s);
	//case B_QUEEN:  case W_QUEEN:  return bishopStepEffect(s) | rookStepEffect(s); // 角+飛でいいや。(馬+龍は王の利きを2回合成して損)
	case NO_PIECE: case PIECE_WHITE: return Bitboard(ZERO); // これも入れておかないと初期化が面倒になる。

	default: ASSERT_LV3(false); sf_unreachable(); return Bitboard(1);
	}
}

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。
Bitboard attacks_bb(Piece pc, Square s, const Bitboard& occupied)
{
	switch (pc)
	{
	case B_PAWN:   return pawnEffect  <BLACK>(s);
	case B_LANCE:  return lanceEffect <BLACK>(s, occupied);
	case B_KNIGHT: return knightEffect<BLACK>(s);
	case B_SILVER: return silverEffect<BLACK>(s);
	case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: case B_GOLDS: return goldEffect<BLACK>(s);

	case W_PAWN:   return pawnEffect  <WHITE>(s);
	case W_LANCE:  return lanceEffect <WHITE>(s, occupied);
	case W_KNIGHT: return knightEffect<WHITE>(s);
	case W_SILVER: return silverEffect<WHITE>(s);
	case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: case W_GOLDS: return goldEffect<WHITE>(s);

		//　先後同じ移動特性の駒
	case B_BISHOP: case W_BISHOP: return bishopEffect(s, occupied);
	case B_ROOK:   case W_ROOK:   return rookEffect  (s, occupied);
	case B_HORSE:  case W_HORSE:  return horseEffect (s, occupied);
	case B_DRAGON: case W_DRAGON: return dragonEffect(s, occupied);
	case B_KING:   case W_KING:   return kingEffect  (s     );
	//case B_QUEEN:  case W_QUEEN:  return bishopEffect (s, occupied) | rookEffect(s, occupied); // 角+飛でいいや。(馬+龍は王の利きを2回合成して損)
	case NO_PIECE: case PIECE_WHITE: return Bitboard(ZERO); // これも入れておかないと初期化が面倒になる。

	default: ASSERT_LV3(false); sf_unreachable(); return Bitboard(1);
	}
}

// Qugiyのアルゴリズムによる、飛車と角の利きの実装。
// magic bitboard tableが不要になる。

// 飛車の横の利き
Bitboard rookRankEffect(Square sq, const Bitboard& occupied)
{
#if defined(USE_AVX512)
    // Shuffle directly into low/high directional words and back, without intermediate interleaving.
    // 中間の交互配置を作らず、方向別の下位・上位ワードへ直接並べ替え、結果も直接戻す。
    const __m128i lowOrder = _mm_setr_epi8(0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8);
    const __m128i highOrder = _mm_setr_epi8(8,9,10,11,12,13,14,15,7,6,5,4,3,2,1,0);
    const __m128i restoreHigh = _mm_setr_epi8(15,14,13,12,11,10,9,8,0,1,2,3,4,5,6,7);
    const __m128i maskLo = QUGIY_ROOK_MASK[sq][0].m;
    const __m128i maskHi = QUGIY_ROOK_MASK[sq][1].m;
    const __m128i oLo = _mm_and_si128(_mm_shuffle_epi8(occupied.m, lowOrder), maskLo);
    const __m128i oHi = _mm_and_si128(_mm_shuffle_epi8(occupied.m, highOrder), maskHi);
    const __m128i resultLo = _mm_ternarylogic_epi64(
      oLo, _mm_add_epi64(oLo, _mm_set1_epi64x(-1)), maskLo, 0x28);
    const __m128i resultHi = _mm_maskz_ternarylogic_epi64(
      _mm_cmpeq_epi64_mask(oLo, _mm_setzero_si128()),
      oHi, _mm_add_epi64(oHi, _mm_set1_epi64x(-1)), maskHi, 0x28);
    Bitboard result;
    result.m = _mm_or_si128(_mm_shuffle_epi8(resultLo, lowOrder),
                            _mm_shuffle_epi8(resultHi, restoreHigh));
    return result;
#else
	// Qugiyのアルゴリズムを忠実にBitboardで実装。
	Bitboard hi, lo , t1, t0;

	const Bitboard mask_lo = QUGIY_ROOK_MASK[sq][0];
	const Bitboard mask_hi = QUGIY_ROOK_MASK[sq][1];

	// occupiedを逆順にする

	// reversed byte occupied bitboard
	Bitboard rocc = occupied.byte_reverse();

	// roccとoccを2枚並べて、その上位u64をhi、下位u64をloに集める。
	// occ側は(先手から見て)左方向への利き、roccは右方向への利き。
	Bitboard::unpack(rocc , occupied, hi, lo);

	// 飛車のstep effectでmaskして…
	hi &= mask_hi;
	lo &= mask_lo;

	// 1減算することにより、利きが通る升までが変化する。
	Bitboard::decrement(hi, lo , t1, t0);

	// xorで変化した升を抽出して、step effectでmaskすれば完成
	t1 = (t1 ^ hi) & mask_hi;
	t0 = (t0 ^ lo) & mask_lo;

	// unpackしていたものを元の状態に戻す(unpackの逆変換はunpack)
	Bitboard::unpack(t1 , t0 , hi , lo);

	// byte_reverseして元の状態に戻して、重ね合わせる。
	// hiの方には、右方向の利き、loは左方向の利きが得られている。
	return hi.byte_reverse() | lo;
#endif
}

// 角の利き
Bitboard bishopEffect(const Square sq, const Bitboard& occupied)
{
    return SimdSliding::bishop(sq, occupied);
}

// === 大駒の部分利き(SEEなどで用いる) ===

// sqの升から指定した方向dへの利き。盤上の駒も考慮する。
Bitboard directEffect(Square sq, Effect8::Direct d, const Bitboard& occupied)
{
	switch (d)
	{
	case Effect8::DIRECT_RU : return rayEffect<Effect8::DIRECT_RU>(sq, occupied);
	case Effect8::DIRECT_R  : return rayEffect<Effect8::DIRECT_R >(sq, occupied);
	case Effect8::DIRECT_RD : return rayEffect<Effect8::DIRECT_RD>(sq, occupied);
	case Effect8::DIRECT_U  : return rayEffect<Effect8::DIRECT_U >(sq, occupied);
	case Effect8::DIRECT_D  : return rayEffect<Effect8::DIRECT_D >(sq, occupied);
	case Effect8::DIRECT_LU : return rayEffect<Effect8::DIRECT_LU>(sq, occupied);
	case Effect8::DIRECT_L  : return rayEffect<Effect8::DIRECT_L >(sq, occupied);
	case Effect8::DIRECT_LD : return rayEffect<Effect8::DIRECT_LD>(sq, occupied);
	default: ASSERT_LV3(false); sf_unreachable(); return Bitboard(ZERO);
	}
}


#endif // !STOCKFISH

} // namespace YaneuraOu::Attacks
