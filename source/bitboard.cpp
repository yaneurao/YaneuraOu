#include "bitboard.h"
#include "extra/long_effect.h"
#include "mate/mate.h"
#include "testcmd/unit_test.h"
#include "misc.h"

#include <sstream>

using namespace std;
namespace YaneuraOu {

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

	// LineBBは、王手の指し手生成からしか使っていないが、move_pickerからQUIET_CHECKS呼び出しているので…。
	// そして、配列シュリンクした。
	Bitboard LineBB[SQ_NB][4];

	Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING - 1][COLOR_NB];

#if defined(LONG_EFFECT_LIBRARY) // 詰みルーチンでは使わなくなったが、LONG_EFFECT_LIBRARYのなかで使っている。
	Bitboard CheckCandidateKingBB[SQ_NB_PLUS1];
#endif

	Bitboard BetweenBB[785];
	u16 BetweenIndex[SQ_NB_PLUS1][SQ_NB_PLUS1];

	// SquareからSquareWithWallへの変換テーブル
	SquareWithWall sqww_table[SQ_NB_PLUS1];

	// ----------------------------------------------------------------------------------------------
	// 飛車・角の利きのためのテーブル
	// ----------------------------------------------------------------------------------------------

	Bitboard QUGIY_ROOK_MASK[SQ_NB_PLUS1][2];
	Bitboard256 QUGIY_BISHOP_MASK[SQ_NB_PLUS1][2];

	// sqの升から各方向への利き
	// 右上、右、右下、上方向は、byte_reverse()してあるので、普通の利きではないから注意。
	// 6方向しか使っていないの、ちょっともったいないので詰める。
	Bitboard QUGIY_STEP_EFFECT[Effect8::DIRECT_NB - 2][SQ_NB_PLUS1];
}

using namespace BB_Table;

// ----------------------------------------------------------------------------------------------

// Bitboardを表示する(USI形式ではない) デバッグ用
std::ostream& operator<<(std::ostream& os, const Bitboard& board)
{
	for (Rank rank = RANK_1; rank <= RANK_9; ++rank)
	{
		for (File file = FILE_9; file >= FILE_1; --file)
			os << ((board & (file | rank)) ? " *" : " .");
		os << endl;
	}
	// 連続して表示させるときのことを考慮して改行を最後に入れておく。
	os << endl;
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
		sqww_table[sq] = SquareWithWall(SQWW_11 + (int32_t)file_of(sq) * SQWW_L + (int32_t)rank_of(sq) * SQWW_D);


	// 2) direct_tableの初期化

	for (auto sq1 : SQ)
		for (auto dir = Effect8::DIRECT_ZERO; dir < Effect8::DIRECT_NB; ++dir)
		{
			// dirの方角に壁にぶつかる(盤外)まで延長していく。このとき、sq1から見てsq2のDirectionsは (1 << dir)である。
			auto delta = Effect8::DirectToDeltaWW(dir);
			for (auto sq2 = to_sqww(sq1) + delta; is_ok(sq2); sq2 += delta)
				Effect8::direc_table[sq1][sqww_to_sq(sq2)] = Effect8::to_directions(dir);
		}


	// 3) Square型のsqの指す升が1であるBitboardがSquareBB。これをまず初期化する。

	for (auto sq : SQ)
	{
		Rank r = rank_of(sq);
		File f = file_of(sq);
		SquareBB[sq] = Bitboard(
			(f <= FILE_7) ? ((uint64_t)1 << (f * 9 + r)) : 0,
			(f >= FILE_8) ? ((uint64_t)1 << ((f - FILE_8) * 9 + r)) : 0
		);
	}

	// 4) Qugiyの飛車のBitboardテーブルの初期化

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

	// 5) Qugiyの角のBitboardテーブルの初期化

	// 4方向
	const Effect8::Direct bishop_direct[4] ={
		Effect8::Direct::DIRECT_LU , // 左上
		Effect8::Direct::DIRECT_LD , // 左下
		Effect8::Direct::DIRECT_RU , // 右上
		Effect8::Direct::DIRECT_RD   // 右下
	};

	for (File f = FILE_1; f <= FILE_9; ++f) {
		for (Rank r = RANK_1; r <= RANK_9; ++r) {

			// 対象升から
			Square sq = f | r;

			// 角の左上、左下、右上、右下それぞれへのstep effect
			Bitboard step_effect[4];

			// 4方向の利きをループで求める
			for (int i = 0; i < 4; ++i)
			{
				Bitboard bb(ZERO);

				auto delta = Effect8::DirectToDeltaWW(bishop_direct[i]);
				// 壁に突き当たるまで進む
				for (auto sq2 = to_sqww(sq) + delta; is_ok(sq2); sq2 += delta)
					bb |= Bitboard(sqww_to_sq(sq2));

				step_effect[i] = bb;
			}

			// 右上、右下はbyte reverseしておかないとうまく求められない。(先手の香の利きがうまく求められないのと同様)

			step_effect[2] = step_effect[2].byte_reverse();
			step_effect[3] = step_effect[3].byte_reverse();

			for (int i = 0; i < 2; ++i)
				QUGIY_BISHOP_MASK[sq][i] = Bitboard256(
					Bitboard(step_effect[0].extract64(i), step_effect[2].extract64(i)),
					Bitboard(step_effect[1].extract64(i), step_effect[3].extract64(i))
				);
		}
	}

	// Qugiyのアルゴリズムで用いる、sqの升から各方向への利きテーブルの初期化
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

	// 6. 近接駒(+盤上の利きを考慮しない駒)のテーブルの初期化。
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

	// 7) BetweenBB , LineBBの初期化
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


	// 8) 王手となる候補の駒のテーブル初期化(王手の指し手生成に必要)

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

	// 9. LONG_EFFECT_LIBRARYの初期化

#if defined (LONG_EFFECT_LIBRARY)
	LongEffect::init();
#endif

	// 10. 1手詰めテーブルの初期化
#if defined (USE_MATE_1PLY)
	Mate::init();
#endif

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

// --- Bitboard256

// Bitboard256の1の升を'*'、0の升を'.'として表示する。デバッグ用。
std::ostream& operator<<(std::ostream& os, const Bitboard256& board)
{
	Bitboard b1, b2;
	board.toBitboard(b1, b2);

	auto print_rank = [&](const Bitboard& b,Rank r) {
		for (File f = FILE_9; f >= FILE_1; --f)
			os << (b.test(f | r) ? " *" : " .");
	};

	for (Rank r = RANK_1; r <= RANK_9; ++r)
	{
		// Bitboardを2列表示する。
		print_rank(b1,r);
		os << ' ';
		print_rank(b2,r);
		os << endl;
	}
	return os;
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

// byte単位で入れ替えたBitboardを返す。
// 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
Bitboard256 Bitboard256::byte_reverse() const
{
#if defined(USE_AVX2)
	const __m256i shuffle = _mm256_set_epi8
		(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,
		 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	Bitboard256 b0;
	b0.m = _mm256_shuffle_epi8(m, shuffle);
	return b0;
#else
	Bitboard256 b0;
	b0.p[0] = bswap64(p[1]);
	b0.p[1] = bswap64(p[0]);
	b0.p[2] = bswap64(p[3]);
	b0.p[3] = bswap64(p[2]);
	return b0;
#endif
}

// SSE2のunpackを実行して返す。
// hi_out = _mm256_unpackhi_epi64(lo_in,hi_in);
// lo_out = _mm256_unpacklo_epi64(lo_in,hi_in);
void Bitboard256::unpack(const Bitboard256 hi_in,const Bitboard256 lo_in, Bitboard256& hi_out, Bitboard256& lo_out)
{
#if defined(USE_AVX2)
	hi_out.m = _mm256_unpackhi_epi64(lo_in.m , hi_in.m);
	lo_out.m = _mm256_unpacklo_epi64(lo_in.m , hi_in.m);
#else
	hi_out.p[0] = lo_in.p[1];
	hi_out.p[1] = hi_in.p[1];
	hi_out.p[2] = lo_in.p[3];
	hi_out.p[3] = hi_in.p[3];

	lo_out.p[0] = lo_in.p[0];
	lo_out.p[1] = hi_in.p[0];
	lo_out.p[2] = lo_in.p[2];
	lo_out.p[3] = hi_in.p[2];
#endif
}

// 2組のBitboard256を、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
// 128bit整数とみなして1引き算したBitboardを返す。
void Bitboard256::decrement(const Bitboard256 hi_in,const Bitboard256 lo_in, Bitboard256& hi_out, Bitboard256& lo_out)
{
#if defined(USE_AVX2)

	// loが0の時だけ1減算するときにhiからの桁借りが生じるので、
	// hi += (lo == 0) ? -1 : 0;
	// みたいな処理で良い。
	hi_out.m = _mm256_add_epi64(hi_in.m, _mm256_cmpeq_epi64(lo_in.m, _mm256_setzero_si256()));

	//  1減算する
	lo_out.m = _mm256_add_epi64(lo_in.m, _mm256_set1_epi64x(-1LL)); // cmpeqとどっちがいいか？
#else
	// bool型はtrueだと(暗黙の型変換で)1だとみなされる。
	hi_out.p[0] = hi_in.p[0] - (lo_in.p[0] == 0);
	hi_out.p[1] = hi_in.p[1] - (lo_in.p[1] == 0);
	hi_out.p[2] = hi_in.p[2] - (lo_in.p[2] == 0);
	hi_out.p[3] = hi_in.p[3] - (lo_in.p[3] == 0);

	lo_out.p[0] = lo_in.p[0] - 1;
	lo_out.p[1] = lo_in.p[1] - 1;
	lo_out.p[2] = lo_in.p[2] - 1;
	lo_out.p[3] = lo_in.p[3] - 1;
#endif
}

// 保持している2つの盤面を重ね合わせた(OR)Bitboardを返す。
Bitboard Bitboard256::merge() const
{
#if defined(USE_AVX2)
	Bitboard b;
	b.m = _mm_or_si128(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
	return b;
#else
	Bitboard b;
	b.p[0] = p[0] | p[2];
	b.p[1] = p[1] | p[3];
	return b;
#endif
}

// 駒の利きに関するnamespace

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。(step effect)
// pc == QUEENだと馬+龍の利きが返る。盤上には駒は何もないものとして考える。
Bitboard effects_from(Piece pc, Square sq)
{
	switch (pc)
	{
	case B_PAWN:   return pawnEffect     <BLACK>(sq);
	case B_LANCE:  return lanceStepEffect<BLACK>(sq);
	case B_KNIGHT: return knightEffect   <BLACK>(sq);
	case B_SILVER: return silverEffect   <BLACK>(sq);
	case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: case B_GOLDS: return goldEffect<BLACK>(sq);

	case W_PAWN:   return pawnEffect     <WHITE>(sq);
	case W_LANCE:  return lanceStepEffect<WHITE>(sq);
	case W_KNIGHT: return knightEffect   <WHITE>(sq);
	case W_SILVER: return silverEffect   <WHITE>(sq);
	case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: case W_GOLDS: return goldEffect<WHITE>(sq);

		//　先後同じ移動特性の駒
	case B_BISHOP: case W_BISHOP: return bishopStepEffect(sq);
	case B_ROOK:   case W_ROOK:   return rookStepEffect  (sq);
	case B_HORSE:  case W_HORSE:  return horseStepEffect (sq);
	case B_DRAGON: case W_DRAGON: return dragonStepEffect(sq);
	case B_KING:   case W_KING:   return kingEffect      (sq);
	//case B_QUEEN:  case W_QUEEN:  return bishopStepEffect(sq) | rookStepEffect(sq); // 角+飛でいいや。(馬+龍は王の利きを2回合成して損)
	case NO_PIECE: case PIECE_WHITE: return Bitboard(ZERO); // これも入れておかないと初期化が面倒になる。

	default: UNREACHABLE; return Bitboard(1);
	}
}

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。
Bitboard effects_from(Piece pc, Square sq, const Bitboard& occ)
{
	switch (pc)
	{
	case B_PAWN:   return pawnEffect  <BLACK>(sq);
	case B_LANCE:  return lanceEffect <BLACK>(sq, occ);
	case B_KNIGHT: return knightEffect<BLACK>(sq);
	case B_SILVER: return silverEffect<BLACK>(sq);
	case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: case B_GOLDS: return goldEffect<BLACK>(sq);

	case W_PAWN:   return pawnEffect  <WHITE>(sq);
	case W_LANCE:  return lanceEffect <WHITE>(sq, occ);
	case W_KNIGHT: return knightEffect<WHITE>(sq);
	case W_SILVER: return silverEffect<WHITE>(sq);
	case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: case W_GOLDS: return goldEffect<WHITE>(sq);

		//　先後同じ移動特性の駒
	case B_BISHOP: case W_BISHOP: return bishopEffect(sq, occ);
	case B_ROOK:   case W_ROOK:   return rookEffect  (sq, occ);
	case B_HORSE:  case W_HORSE:  return horseEffect (sq, occ);
	case B_DRAGON: case W_DRAGON: return dragonEffect(sq, occ);
	case B_KING:   case W_KING:   return kingEffect  (sq     );
	//case B_QUEEN:  case W_QUEEN:  return bishopEffect (sq, occ) | rookEffect(sq, occ); // 角+飛でいいや。(馬+龍は王の利きを2回合成して損)
	case NO_PIECE: case PIECE_WHITE: return Bitboard(ZERO); // これも入れておかないと初期化が面倒になる。

	default: UNREACHABLE; return Bitboard(1);
	}
}

// Qugiyのアルゴリズムによる、飛車と角の利きの実装。
// magic bitboard tableが不要になる。

// 飛車の横の利き
Bitboard rookRankEffect(Square sq, const Bitboard& occupied)
{
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
}

// 角の利き
Bitboard bishopEffect(const Square sq, const Bitboard& occupied)
{
	const Bitboard256 mask_lo =  QUGIY_BISHOP_MASK[sq][0];
	const Bitboard256 mask_hi =  QUGIY_BISHOP_MASK[sq][1];

	// occupiedを2枚並べたBitboard256を用意する。
	const Bitboard256 occ2(occupied);

	// occupiedを(byte単位で)左右反転させたBitboardを2枚並べたBitboard256を用意する。
	const Bitboard256 rocc2(occupied.byte_reverse());

	Bitboard256 hi, lo ,t1 , t0;
	Bitboard256::unpack(rocc2, occ2, hi, lo);

	hi &= mask_hi;
	lo &= mask_lo;

	Bitboard256::decrement(hi, lo, t1, t0);

	// xorで変化した升を抽出して、step effectでmaskすれば完成
	t1 = (t1 ^ hi) & mask_hi;
	t0 = (t0 ^ lo) & mask_lo;

	// unpackしていたものを元の状態に戻す(unpackの逆変換はunpack)
	Bitboard256::unpack(t1 , t0 , hi , lo);

	// byte_reverseして元の状態に戻して、重ね合わせる。
	// hiの方には、右方向の利き、loは左方向の利きが得られている。
	return (hi.byte_reverse() | lo).merge();
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
	default: UNREACHABLE; return Bitboard(ZERO);
	}
}

// UnitTest
void Bitboard::UnitTest(Test::UnitTester& tester, IEngine& engine)
{
	//Bitboard b(SQ_75);
	//cout << lanceEffect<WHITE>(SQ_71, b) << endl;
	//cout << lanceEffect<WHITE>(SQ_72, b) << endl;

	auto section1 = tester.section("Bitboard");

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
				Bitboard bb0 = effects_from(pc, SQ_55, Bitboard(ZERO));
				int p0 = bb0.pop_count();
				Bitboard bb1 = effects_from(pc, SQ_55, Bitboard(1));
				int p1 = bb1.pop_count();
				bool ok0 = (p0 == p0_table[(int)pt]);
				if (!ok0)
				{
					cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(ZERO) , pop_count = " << p0 << endl;
					cout << bb0 << endl;
				}
				bool ok1 = (p1 == p1_table[(int)pt]);
				if (!ok1)
				{
					cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(1) , pop_count = " << p1 << endl;
					cout << bb1 << endl;
				}
				all_ok &= ok0 & ok1;
			}
		tester.test("effects_from", all_ok);
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

		tester.test("attacks_bb<B_KNIGHT>",attacks_bb<B_KNIGHT>(SQ_77) == (Bitboard(SQ_85) | Bitboard(SQ_65)));
		tester.test("attacks_bb<W_KNIGHT>",attacks_bb<W_KNIGHT>(SQ_77) == (Bitboard(SQ_89) | Bitboard(SQ_69)));
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

// UnitTest
void Bitboard256::UnitTest(Test::UnitTester& tester, IEngine& engine)
{
	auto section1 = tester.section("Bitboard256");
	{
		// 2つのBitboardを合体させて、分離させて一致するかのテスト

		bool all_ok = true;

		Bitboard b1, b2 , b3 ,b4;

		for (auto sq : SQ)
		{
			b1 = Bitboard(sq);
			b2 = Bitboard(SQ_99 - sq);

			Bitboard256 b256(b1, b2);
			b256.toBitboard(b3, b4);
			all_ok &= b1 == b3 && b2 == b4;
		}

		tester.test("toBitboard", all_ok);
	}
	{
		// ByteReverseがちゃんと機能しているかのテスト

		Bitboard256 b(Bitboard(0x0123456789abcdef, 0x123456789abcdef0), Bitboard(0x23456789abcdef01, 0x3456789abcdef012));
		Bitboard256 r = b.byte_reverse();

		tester.test("byte_reverse",
			r.p[0] == 0xf0debc9a78563412 && r.p[1] == 0xefcdab8967452301 &&
			r.p[2] == 0x12f0debc9a785634 && r.p[3] == 0x01efcdab89674523
		);
	}
}

} // namespace YaneuraOu
