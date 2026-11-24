#include "attacks.h"

#include <array>

#include "misc.h"

#if !STOCKFISH
#include "extra/long_effect.h"
#include "mate/mate.h"
#endif

namespace YaneuraOu::Attacks {

#if STOCKFISH
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard inverse_RayPassBB[SQUARE_NB][SQUARE_NB];
Bitboard DiagBB[SQUARE_NB];
Bitboard AntiDiagBB[SQUARE_NB];
#else
// StockfishのLineBBは[升][升]、将棋版は[升][方向]。
// StockfishのBetweenBBは終点込みだが、将棋版は両端を除く圧縮表。
// 配列を縮小しているので、升の組を方向または圧縮indexへ変換して参照する。
Bitboard LineBB[SQ_NB][4];
Bitboard BetweenBB[638];
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

DualMagicTables DualMagics;

// sqの升から各方向への利き
// 右上、右、右下、上方向は、byteswap_bb()してあるので、普通の利きではないから注意。
// 6方向しか使っていないの、ちょっともったいないので詰める。
Bitboard QUGIY_STEP_EFFECT[Effect8::DIRECT_NB - 2][SQ_NB_PLUS1];


using namespace BB_Table;
#endif

#if STOCKFISH
static constexpr Bitboard line_mask(Square sq, Direction d1, Direction d2) {
    Bitboard mask = 0;
    for (Direction d : {d1, d2})
    {
        Square s = sq;
        while (Bitboard dest = safe_destination(s, d))
        {
            mask |= dest;
            s += d;
        }
    }
    return mask;
}
#else
// 指定した升から、壁に突き当たるまで一方向へ延長したマスクを返す。
static Bitboard ray_mask(Square sq, Effect8::Direct d) {
    Bitboard mask(ZERO);
    const auto delta = Effect8::DirectToDeltaWW(d);
    for (auto s = to_sqww(sq) + delta; is_ok(s); s += delta)
        mask |= square_bb(sqww_to_sq(s));
    return mask;
}

// Stockfishのline_mask()に対応。将棋では壁付き座標と8方向のDirectを使う。
static Bitboard line_mask(Square sq, Effect8::Direct d1, Effect8::Direct d2) {
    Bitboard mask(ZERO);
    for (Effect8::Direct d : {d1, d2})
        mask |= ray_mask(sq, d);
    return mask;
}

static void init_directions() {
	// 1) direc_tableの初期化

	for (auto sq1 : SQ)
		for (auto dir = Effect8::DIRECT_ZERO; dir < Effect8::DIRECT_NB; ++dir)
		{
			// dirの方角に壁にぶつかる(盤外)まで延長していく。このとき、sq1から見てsq2のDirectionsは (1 << dir)である。
			auto delta = Effect8::DirectToDeltaWW(dir);
			for (auto sq2 = to_sqww(sq1) + delta; is_ok(sq2); sq2 += delta)
				Effect8::direc_table[sq1][sqww_to_sq(sq2)] = Effect8::to_directions(dir);
		}
}

static void init_dual_magics() {
    // 2) Build shared directional masks from board coordinates.
    // 盤の座標から共通の方向別マスクを作る。
    for (int index = 0; index <= SQ_NB; ++index) {
        auto& bishop = DualMagics.bishop[index];
        for (auto& ray : bishop.ray) ray = Bitboard(ZERO);
        if (index == SQ_NB) continue;
        const Square sq = Square(index);
        bishop.ray[0] = ray_mask(sq, Effect8::DIRECT_LU);
        bishop.ray[1] = ray_mask(sq, Effect8::DIRECT_LD);
        bishop.ray[2] = ray_mask(sq, Effect8::DIRECT_RU).byteswap_bb();
        bishop.ray[3] = ray_mask(sq, Effect8::DIRECT_RD).byteswap_bb();
    }

	// 3) Qugiyの飛車のBitboardテーブルの初期化

	for (File f = FILE_1; f <= FILE_9; ++f) {
		for (Rank r = RANK_1; r <= RANK_9; ++r) {

			// SQの升から左方向
            const Bitboard left = ray_mask(f | r, Effect8::DIRECT_L);

			// SQの升から右方向
            const Bitboard right = ray_mask(f | r, Effect8::DIRECT_R);

			Bitboard right_rev = right.byteswap_bb();

			Bitboard hi, lo;
			Bitboard::unpack(right_rev, left, hi, lo);

			DualMagics.rook[f | r][0] = lo;
			DualMagics.rook[f | r][1] = hi;
		}
	}
}

static void init_sliding_attacks() {
    init_dual_magics();

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
            Bitboard bb = ray_mask(sq, d);

			if (reverse)
				bb = bb.byteswap_bb();

			QUGIY_STEP_EFFECT[dd][sq] = bb;
		}

		dd++;
	}
}

static void init_pseudo_attacks() {
	// 5) 近接駒(+盤上の利きを考慮しない駒)のテーブルの初期化。
	// なるべく他の駒の利きに依存しない形で初期化する。

	// 歩と香のstep effect
	// ※　step effectとは、障害物がないときの利きのこと。
	for (auto c : COLOR)
		for (auto sq : SQ)
		{
			// 香
			// 香のstep effectは、筋のBitboardとsqより前の段を表現するBitboardを用いて、他の駒の利きに依存せず初期化できる。

			LanceStepEffectBB[sq][c] = file_bb(sq) & ForwardRanksBB[c][rank_of(sq)];

			// 歩

			// 備考) 歩の利きは駒が敷き詰められている時の香の利きとして定義できる。
			//  PawnEffectBB[sq][c] = lanceEffect(c, sq, Bitboard(1));
			// → QugiyのアルゴリズムでlanceEffectの計算にpawnEffectを用いるようになったのでこれができなくなった。

			// 歩の利きは何段目であるか。
			Rank r = (Rank)(rank_of(sq) + (c == BLACK ? -1 : +1));
			// その段数が1～9段目に収まるなら..
			PawnEffectBB[sq][c] = (RANK_1 <= r && r <= RANK_9) ? square_bb(file_of(sq) | r) : Bitboard(ZERO);
		}

	// 備考) ここでlanceEffectが使えるようになったので、以降、rookEffectが使える。
	// (rookEffectは縦利きを求めるのにlanceEffectに依存している)
	// ※　bishopEffectは、方向別マスクを初期化した時以降使える。
	// 以下では、rookEffect,bishopEffect,lanceEffectを用いて残りの駒の利きを定義する。

	// 先後の区別のないstep effect (玉、角、飛)
	for (auto sq : SQ)
	{
		// 玉
		// 玉の利きは駒が敷き詰められている時の角と飛車の利きの重ね合わせとして定義できる。
        const auto [bishopAttacks, rookAttacks] = both_attacks_bb(sq, Bitboard(1));
		KingEffectBB[sq] = bishopAttacks | rookAttacks;

		// 角
		// 角のstep effectは、盤上に駒がない時の角の利き。
        const auto [bishopEmpty, rookEmpty] = both_attacks_bb(sq, Bitboard(0));
		BishopStepEffectBB[sq] = bishopEmpty;

		// 飛
		// 飛車のstep effectは、盤上に駒がない時の飛車の利き。
		RookStepEffectBB[sq]   = rookEmpty;
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
				Square sq2 = pawn.pop_lsb();
				Bitboard pawn2 = lanceEffect(c, sq2, Bitboard(1)); // さらに1つ前
				if (pawn2)
					tmp = bishopEffect(sq2, Bitboard(1)) & rank_bb(pawn2.pop_lsb());
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
				mask = rank_bb(e_pawn.pop_lsb());
			GoldEffectBB[sq][c]= (bishopEffect(sq, Bitboard(1)) & ~mask) | rookEffect(sq, Bitboard(1));

			// --- 以下のbitboard、あまり頻繁に用いないので他のbitboardを合成して代用する。

			// 盤上の駒がないときのqueenの利き
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_QUEEN] = bishopEffect(sq, Bitboard(ZERO)) | rookEffect(sq, Bitboard(ZERO));

			// 長さ1の十字
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS00] = rookEffect(sq, Bitboard(1));

			// 長さ1の斜め
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS45] = bishopEffect(sq, Bitboard(1));
		}
}

static void init_lines() {
	// 6) BetweenBB , LineBBの初期化
    // Stockfishと異なり、BetweenBBの各値は両端を含めず、BetweenIndexを対称にする。
	{
		u16 between_index = 1;
		// BetweenBB[0] == Bitboard(ZERO)であることを保証する。

		for (auto s1 : SQ)
			for (auto s2 : SQ)
			{
				// 十字方向か、斜め方向かだけを判定して、例えば十字方向なら
				// rookEffect(sq1,square_bb(s2)) & rookEffect(sq2,square_bb(s1))
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

                    // 端点が異なっても同じ升集合なら、既存のBitboardを共有する。
                    // 重複判定は起動時だけ行い、探索中の表引きは変更しない。
                    u16 index = 1;
                    while (index < between_index && BetweenBB[index] != bb)
                        ++index;
                    if (index == between_index)
                    {
                        ASSERT_LV1(between_index < std::size(BetweenBB));
                        BetweenBB[between_index++] = bb;
                    }
                    BetweenIndex[s1][s2] = index;
				}
			}

		ASSERT_LV1(between_index == std::size(BetweenBB));

		// 対称性を考慮して、さらにシュリンクする。
		for (auto s1 : SQ)
			for (auto s2 : SQ)
				if (s1 > s2)
					BetweenIndex[s1][s2] = BetweenIndex[s2][s1];

	}
    // 利き関数に依存せず空盤上の直線マスクを作り、始点の升も含める。
    static constexpr std::array<std::array<Effect8::Direct, 2>, 4> directions = {{
      {Effect8::DIRECT_RU, Effect8::DIRECT_LD},
      {Effect8::DIRECT_R, Effect8::DIRECT_L},
      {Effect8::DIRECT_RD, Effect8::DIRECT_LU},
      {Effect8::DIRECT_U, Effect8::DIRECT_D}
    }};
    for (auto s1 : SQ)
        for (int d = 0; d < 4; ++d)
            LineBB[s1][d] = line_mask(s1, directions[d][0], directions[d][1]) | s1;
}

static void init_check_candidates() {
	// 7) 王手となる候補の駒のテーブル初期化(王手の指し手生成に必要)

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
			CheckCandidateBB[ksq][PAWN - 1][Us] = target & ~square_bb(ksq);

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
			CheckCandidateBB[ksq][KNIGHT - 1][Us] = target & ~square_bb(ksq);

			// 銀も同様だが、2,3段目からの引き成りで王手になるパターンがある。(4段玉と5段玉に対して)
			target = Bitboard(ZERO);
			FOREACH(silverEffect(them, ksq), silverEffect);
			FOREACH(enemyGold, silverEffect); // 移動先が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			FOREACH(goldEffect(them, ksq), enemy_field(Us) & silverEffect); // 移動元が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			CheckCandidateBB[ksq][SILVER - 1][Us] = target & ~square_bb(ksq);

			// 金
			target = Bitboard(ZERO);
			FOREACH(goldEffect(them, ksq), goldEffect);
			CheckCandidateBB[ksq][GOLD - 1][Us] = target & ~square_bb(ksq);

			// 角
			target = Bitboard(ZERO);
			FOREACH_BR(bishopEffect(ksq, Bitboard(ZERO)), bishopEffect);
			FOREACH_BR(kingEffect(ksq) & enemy_field(Us), bishopEffect); // 移動先が敵陣 == 成れる == 王の動き
			FOREACH_BR(kingEffect(ksq), enemy_field(Us) & bishopEffect); // 移動元が敵陣 == 成れる == 王の動き
			CheckCandidateBB[ksq][BISHOP - 1][Us] = target & ~square_bb(ksq);

			// 飛・龍は無条件全域。
			// ROOKのところには馬のときのことを格納

			// 馬
			target = Bitboard(ZERO);
			FOREACH_BR(horseEffect(ksq, Bitboard(ZERO)), horseEffect);
			CheckCandidateBB[ksq][ROOK - 1][Us] = target & ~square_bb(ksq);
		}

#undef FOREACH
#undef FOREACH_BR
}

#endif

void init() {
#if STOCKFISH

    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
    {
        for (PieceType pt : {BISHOP, ROOK})
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            {
                if (PseudoAttacks[pt][s1] & s2)
                {
                    LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) | s1 | s2;
                    BetweenBB[s1][s2] =
                      (attacks_bb(pt, s1, square_bb(s2)) & attacks_bb(pt, s2, square_bb(s1)));
                    inverse_RayPassBB[s2][s1] =
                      attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2);
                }
                BetweenBB[s1][s2] |= s2;
            }

        DiagBB[s1]     = line_mask(s1, NORTH_EAST, SOUTH_WEST) | s1;
        AntiDiagBB[s1] = line_mask(s1, NORTH_WEST, SOUTH_EAST) | s1;
    }
#else
    // 各テーブルの依存関係に従って初期化する。
    init_directions();
    init_sliding_attacks();
    init_pseudo_attacks();
    init_lines();
    init_check_candidates();

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
    return dual_magic(sq).rook_rank_attacks_bb(occupied);
}

// 角の利き
Bitboard bishopEffect(const Square sq, const Bitboard& occupied)
{
    return dual_magic(sq).bishop_attacks_bb(occupied);
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
