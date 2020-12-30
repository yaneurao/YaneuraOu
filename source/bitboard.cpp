#include "bitboard.h"
#include "extra/long_effect.h"
#include "mate/mate.h"

#include <sstream>

using namespace std;

// ----- Bitboard const

Bitboard ALL_BB = Bitboard(UINT64_C(0x7FFFFFFFFFFFFFFF), UINT64_C(0x3FFFF));
Bitboard ZERO_BB = Bitboard(0, 0);

Bitboard FILE1_BB = Bitboard(UINT64_C(0x1ff) << (9 * 0), 0);
Bitboard FILE2_BB = Bitboard(UINT64_C(0x1ff) << (9 * 1), 0);
Bitboard FILE3_BB = Bitboard(UINT64_C(0x1ff) << (9 * 2), 0);
Bitboard FILE4_BB = Bitboard(UINT64_C(0x1ff) << (9 * 3), 0);
Bitboard FILE5_BB = Bitboard(UINT64_C(0x1ff) << (9 * 4), 0);
Bitboard FILE6_BB = Bitboard(UINT64_C(0x1ff) << (9 * 5), 0);
Bitboard FILE7_BB = Bitboard(UINT64_C(0x1ff) << (9 * 6), 0);
Bitboard FILE8_BB = Bitboard(0, 0x1ff << (9 * 0));
Bitboard FILE9_BB = Bitboard(0, 0x1ff << (9 * 1));

Bitboard RANK1_BB = Bitboard(UINT64_C(0x40201008040201) << 0, 0x201 << 0);
Bitboard RANK2_BB = Bitboard(UINT64_C(0x40201008040201) << 1, 0x201 << 1);
Bitboard RANK3_BB = Bitboard(UINT64_C(0x40201008040201) << 2, 0x201 << 2);
Bitboard RANK4_BB = Bitboard(UINT64_C(0x40201008040201) << 3, 0x201 << 3);
Bitboard RANK5_BB = Bitboard(UINT64_C(0x40201008040201) << 4, 0x201 << 4);
Bitboard RANK6_BB = Bitboard(UINT64_C(0x40201008040201) << 5, 0x201 << 5);
Bitboard RANK7_BB = Bitboard(UINT64_C(0x40201008040201) << 6, 0x201 << 6);
Bitboard RANK8_BB = Bitboard(UINT64_C(0x40201008040201) << 7, 0x201 << 7);
Bitboard RANK9_BB = Bitboard(UINT64_C(0x40201008040201) << 8, 0x201 << 8);

Bitboard FILE_BB[FILE_NB] = { FILE1_BB,FILE2_BB,FILE3_BB,FILE4_BB,FILE5_BB,FILE6_BB,FILE7_BB,FILE8_BB,FILE9_BB };
Bitboard RANK_BB[RANK_NB] = { RANK1_BB,RANK2_BB,RANK3_BB,RANK4_BB,RANK5_BB,RANK6_BB,RANK7_BB,RANK8_BB,RANK9_BB };

Bitboard ForwardRanksBB[COLOR_NB][RANK_NB] = {
  { ZERO_BB, RANK1_BB, RANK1_BB | RANK2_BB, RANK1_BB | RANK2_BB | RANK3_BB, RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
  ~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB), ~(RANK9_BB | RANK8_BB | RANK7_BB), ~(RANK9_BB | RANK8_BB), ~RANK9_BB },
  { ~RANK1_BB, ~(RANK1_BB | RANK2_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
  RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB, RANK9_BB | RANK8_BB | RANK7_BB, RANK9_BB | RANK8_BB, RANK9_BB, ZERO_BB }
};

// 敵陣を表現するBitboard。
Bitboard EnemyField[COLOR_NB] = { RANK1_BB | RANK2_BB | RANK3_BB , RANK7_BB | RANK8_BB | RANK9_BB };


// ----- Bitboard tables

// sqの升が1であるbitboard
Bitboard SquareBB[SQ_NB_PLUS1];

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


// 歩が打てる筋を得るためのmask
u64 PAWN_DROP_MASKS[SQ_NB];

// LineBBは、王手の指し手生成からしか使っていないが、move_pickerからQUIET_CHECKS呼び出しているので…。
// そして、配列シュリンクした。
Bitboard LineBB[SQ_NB][4];

Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING-1][COLOR_NB];
Bitboard CheckCandidateKingBB[SQ_NB_PLUS1];

u8 Slide[SQ_NB_PLUS1] = {
  1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
  10, 10, 10, 10, 10, 10, 10, 10, 10,
  19, 19, 19, 19, 19, 19, 19, 19, 19,
  28, 28, 28, 28, 28, 28, 28, 28, 28,
  37, 37, 37, 37, 37, 37, 37, 37, 37,
  46, 46, 46, 46, 46, 46, 46, 46, 46,
  55, 55, 55, 55, 55, 55, 55, 55, 55,
  1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
  10, 10, 10, 10, 10, 10, 10, 10, 10,
  0 , // SQ_NB用
};

Bitboard BetweenBB[785];
u16 BetweenIndex[SQ_NB_PLUS1][SQ_NB_PLUS1];

// SquareからSquareWithWallへの変換テーブル
SquareWithWall sqww_table[SQ_NB_PLUS1];

// ----------------------------------------------------------------------------------------------
// 飛車・角の利きのためのテーブル
// ----------------------------------------------------------------------------------------------

// 飛車の縦の利き
u64      RookFileEffect[RANK_NB + 1][128];

#if defined(USE_OLD_YANEURAOU_EFFECT)

// magic bitboardを用いない時の実装

// 角の利き
Bitboard BishopEffect[2][1856 + 1];
Bitboard BishopEffectMask[2][SQ_NB_PLUS1];
int BishopEffectIndex[2][SQ_NB_PLUS1];

// 飛車の横の利き
Bitboard RookRankEffect[FILE_NB + 1][128];

#else


// ----------------------------------
//  Magic Bitboard Table from Apery
//  https://github.com/HiraokaTakuya/apery/blob/master/src/bitboard.cpp
// ----------------------------------

// 各マスのrookが利きを調べる必要があるマスの数
const int RookBlockBits[SQ_NB_PLUS1] = {
	14, 13, 13, 13, 13, 13, 13, 13, 14,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	13, 12, 12, 12, 12, 12, 12, 12, 13,
	14, 13, 13, 13, 13, 13, 13, 13, 14,
	0
};

// 各マスのbishopが利きを調べる必要があるマスの数
const int BishopBlockBits[SQ_NB_PLUS1] = {
	7,  6,  6,  6,  6,  6,  6,  6,  7,
	6,  6,  6,  6,  6,  6,  6,  6,  6,
	6,  6,  8,  8,  8,  8,  8,  6,  6,
	6,  6,  8, 10, 10, 10,  8,  6,  6,
	6,  6,  8, 10, 12, 10,  8,  6,  6,
	6,  6,  8, 10, 10, 10,  8,  6,  6,
	6,  6,  8,  8,  8,  8,  8,  6,  6,
	6,  6,  6,  6,  6,  6,  6,  6,  6,
	7,  6,  6,  6,  6,  6,  6,  6,  7,
	0
};

// Magic Bitboard で利きを求める際のシフト量
// RookShiftBits[17], RookShiftBits[53] はマジックナンバーが見つからなかったため、
// シフト量を 1 つ減らす。(テーブルサイズを 2 倍にする。)
// この方法は issei_y さんに相談したところ、教えて頂いた方法。
// PEXT Bitboardを使用する際はシフト量を減らす必要が無い。
const int RookShiftBits[SQ_NB_PLUS1] = {
	50, 51, 51, 51, 51, 51, 51, 51, 50,
#if defined (USE_BMI2)
	51, 52, 52, 52, 52, 52, 52, 52, 51,
#else
	51, 52, 52, 52, 52, 52, 52, 52, 50, // [17]: 51 -> 50
#endif
	51, 52, 52, 52, 52, 52, 52, 52, 51,
	51, 52, 52, 52, 52, 52, 52, 52, 51,
	51, 52, 52, 52, 52, 52, 52, 52, 51,
#if defined (USE_BMI2)
	51, 52, 52, 52, 52, 52, 52, 52, 51,
#else
	51, 52, 52, 52, 52, 52, 52, 52, 50, // [53]: 51 -> 50
#endif
	51, 52, 52, 52, 52, 52, 52, 52, 51,
	51, 52, 52, 52, 52, 52, 52, 52, 51,
	50, 51, 51, 51, 51, 51, 51, 51, 50,
	0
};

// Magic Bitboard で利きを求める際のシフト量
const int BishopShiftBits[SQ_NB_PLUS1] = {
	57, 58, 58, 58, 58, 58, 58, 58, 57,
	58, 58, 58, 58, 58, 58, 58, 58, 58,
	58, 58, 56, 56, 56, 56, 56, 58, 58,
	58, 58, 56, 54, 54, 54, 56, 58, 58,
	58, 58, 56, 54, 52, 54, 56, 58, 58,
	58, 58, 56, 54, 54, 54, 56, 58, 58,
	58, 58, 56, 56, 56, 56, 56, 58, 58,
	58, 58, 58, 58, 58, 58, 58, 58, 58,
	57, 58, 58, 58, 58, 58, 58, 58, 57,
	0
};

#if defined (USE_BMI2)
#else
const u64 RookMagic[SQ_NB_PLUS1] = {
	UINT64_C(0x140000400809300),  UINT64_C(0x1320000902000240), UINT64_C(0x8001910c008180),
	UINT64_C(0x40020004401040),   UINT64_C(0x40010000d01120),   UINT64_C(0x80048020084050),
	UINT64_C(0x40004000080228),   UINT64_C(0x400440000a2a0a),   UINT64_C(0x40003101010102),
	UINT64_C(0x80c4200012108100), UINT64_C(0x4010c00204000c01), UINT64_C(0x220400103250002),
	UINT64_C(0x2600200004001),    UINT64_C(0x40200052400020),   UINT64_C(0xc00100020020008),
	UINT64_C(0x9080201000200004), UINT64_C(0x2200201000080004), UINT64_C(0x80804c0020200191),
	UINT64_C(0x45383000009100),   UINT64_C(0x30002800020040),   UINT64_C(0x40104000988084),
	UINT64_C(0x108001000800415),  UINT64_C(0x14005000400009),   UINT64_C(0xd21001001c00045),
	UINT64_C(0xc0003000200024),   UINT64_C(0x40003000280004),   UINT64_C(0x40021000091102),
	UINT64_C(0x2008a20408000d00), UINT64_C(0x2000100084010040), UINT64_C(0x144080008008001),
	UINT64_C(0x50102400100026a2), UINT64_C(0x1040020008001010), UINT64_C(0x1200200028005010),
	UINT64_C(0x4280030030020898), UINT64_C(0x480081410011004),  UINT64_C(0x34000040800110a),
	UINT64_C(0x101000010c0021),   UINT64_C(0x9210800080082),    UINT64_C(0x6100002000400a7),
	UINT64_C(0xa2240800900800c0), UINT64_C(0x9220082001000801), UINT64_C(0x1040008001140030),
	UINT64_C(0x40002220040008),   UINT64_C(0x28000124008010c),  UINT64_C(0x40008404940002),
	UINT64_C(0x40040800010200),   UINT64_C(0x90000809002100),   UINT64_C(0x2800080001000201),
	UINT64_C(0x1400020001000201), UINT64_C(0x180081014018004),  UINT64_C(0x1100008000400201),
	UINT64_C(0x80004000200201),   UINT64_C(0x420800010000201),  UINT64_C(0x2841c00080200209),
	UINT64_C(0x120002401040001),  UINT64_C(0x14510000101000b),  UINT64_C(0x40080000808001),
	UINT64_C(0x834000188048001),  UINT64_C(0x4001210000800205), UINT64_C(0x4889a8007400201),
	UINT64_C(0x2080044080200062), UINT64_C(0x80004002861002),   UINT64_C(0xc00842049024),
	UINT64_C(0x8040000202020011), UINT64_C(0x400404002c0100),   UINT64_C(0x2080028202000102),
	UINT64_C(0x8100040800590224), UINT64_C(0x2040009004800010), UINT64_C(0x40045000400408),
	UINT64_C(0x2200240020802008), UINT64_C(0x4080042002200204), UINT64_C(0x4000b0000a00a2),
	UINT64_C(0xa600000810100),    UINT64_C(0x1410000d001180),   UINT64_C(0x2200101001080),
	UINT64_C(0x100020014104e120), UINT64_C(0x2407200100004810), UINT64_C(0x80144000a0845050),
	UINT64_C(0x1000200060030c18), UINT64_C(0x4004200020010102), UINT64_C(0x140600021010302)
};

const u64 BishopMagic[SQ_NB_PLUS1] = {
	UINT64_C(0x20101042c8200428), UINT64_C(0x840240380102),     UINT64_C(0x800800c018108251),
	UINT64_C(0x82428010301000),   UINT64_C(0x481008201000040),  UINT64_C(0x8081020420880800),
	UINT64_C(0x804222110000),     UINT64_C(0xe28301400850),     UINT64_C(0x2010221420800810),
	UINT64_C(0x2600010028801824), UINT64_C(0x8048102102002),    UINT64_C(0x4000248100240402),
	UINT64_C(0x49200200428a2108), UINT64_C(0x460904020844),     UINT64_C(0x2001401020830200),
	UINT64_C(0x1009008120),       UINT64_C(0x4804064008208004), UINT64_C(0x4406000240300ca0),
	UINT64_C(0x222001400803220),  UINT64_C(0x226068400182094),  UINT64_C(0x95208402010d0104),
	UINT64_C(0x4000807500108102), UINT64_C(0xc000200080500500), UINT64_C(0x5211000304038020),
	UINT64_C(0x1108100180400820), UINT64_C(0x10001280a8a21040), UINT64_C(0x100004809408a210),
	UINT64_C(0x202300002041112),  UINT64_C(0x4040a8000460408),  UINT64_C(0x204020021040201),
	UINT64_C(0x8120013180404),    UINT64_C(0xa28400800d020104), UINT64_C(0x200c201000604080),
	UINT64_C(0x1082004000109408), UINT64_C(0x100021c00c410408), UINT64_C(0x880820905004c801),
	UINT64_C(0x1054064080004120), UINT64_C(0x30c0a0224001030),  UINT64_C(0x300060100040821),
	UINT64_C(0x51200801020c006),  UINT64_C(0x2100040042802801), UINT64_C(0x481000820401002),
	UINT64_C(0x40408a0450000801), UINT64_C(0x810104200000a2),   UINT64_C(0x281102102108408),
	UINT64_C(0x804020040280021),  UINT64_C(0x2420401200220040), UINT64_C(0x80010144080c402),
	UINT64_C(0x80104400800002),   UINT64_C(0x1009048080400081), UINT64_C(0x100082000201008c),
	UINT64_C(0x10001008080009),   UINT64_C(0x2a5006b80080004),  UINT64_C(0xc6288018200c2884),
	UINT64_C(0x108100104200a000), UINT64_C(0x141002030814048),  UINT64_C(0x200204080010808),
	UINT64_C(0x200004013922002),  UINT64_C(0x2200000020050815), UINT64_C(0x2011010400040800),
	UINT64_C(0x1020040004220200), UINT64_C(0x944020104840081),  UINT64_C(0x6080a080801c044a),
	UINT64_C(0x2088400811008020), UINT64_C(0xc40aa04208070),    UINT64_C(0x4100800440900220),
	UINT64_C(0x48112050),         UINT64_C(0x818200d062012a10), UINT64_C(0x402008404508302),
	UINT64_C(0x100020101002),     UINT64_C(0x20040420504912),   UINT64_C(0x2004008118814),
	UINT64_C(0x1000810650084024), UINT64_C(0x1002a03002408804), UINT64_C(0x2104294801181420),
	UINT64_C(0x841080240500812),  UINT64_C(0x4406009000004884), UINT64_C(0x80082004012412),
	UINT64_C(0x80090880808183),   UINT64_C(0x300120020400410),  UINT64_C(0x21a090100822002)
};
#endif

// これらは一度値を設定したら二度と変更しない。
// 本当は const 化したい。
#if defined (USE_BMI2)
Bitboard RookAttack[495616 + 1 /* SQ_NB対応*/];
#else
Bitboard RookAttack[512000 + 1 /* SQ_NB対応*/];
#endif

int RookAttackIndex[SQ_NB_PLUS1];
Bitboard RookBlockMask[SQ_NB_PLUS1];
Bitboard BishopAttack[20224 + 1 /* SQ_NB対応*/];
int BishopAttackIndex[SQ_NB_PLUS1];
Bitboard BishopBlockMask[SQ_NB_PLUS1];

namespace {

	// square のマスにおける、障害物を調べる必要がある場所を調べて Bitboard で返す。
	Bitboard rookBlockMaskCalc(const Square square) {
		Bitboard result = FILE_BB[file_of(square)] ^ RANK_BB[rank_of(square)];
		if (file_of(square) != FILE_9) result &= ~FILE9_BB;
		if (file_of(square) != FILE_1) result &= ~FILE1_BB;
		if (rank_of(square) != RANK_9) result &= ~RANK9_BB;
		if (rank_of(square) != RANK_1) result &= ~RANK1_BB;
		return result;
	}

	// square のマスにおける、障害物を調べる必要がある場所を調べて Bitboard で返す。
	Bitboard bishopBlockMaskCalc(const Square square) {
		const Rank rank = rank_of(square);
		const File file = file_of(square);
		Bitboard result = ZERO_BB;
		for (auto sq : SQ)
		{
			const Rank r = rank_of(sq);
			const File f = file_of(sq);
			if (abs(rank - r) == abs(file - f))
				result |= sq;
		}
		result &= ~(RANK9_BB | RANK1_BB | FILE9_BB | FILE1_BB);
		result &= ~Bitboard(square);

		return result;
	}

	// Rook or Bishop の利きの範囲を調べて bitboard で返す。
	// occupied  障害物があるマスが 1 の bitboard
	Bitboard attackCalc(const Square square, const Bitboard& occupied, const bool isBishop) {

		// 飛車と角の利きの方角
		const SquareWithWall deltaArray[2][4] = {
			{ SQWW_U, SQWW_D , SQWW_L , SQWW_R },
			{ SQWW_LU , SQWW_LD , SQWW_RD , SQWW_RU}
		};

		Bitboard result = ZERO_BB;
		for (SquareWithWall delta : deltaArray[isBishop]) {

			// 壁に当たるまでsqを利き方向に伸ばしていく
			for (auto sq = to_sqww(square) + delta; is_ok(sq) ; sq += delta)
			{
				auto s = sqww_to_sq(sq);  // まだ障害物に当っていないのでここまでは利きが到達している
				result |= s;
				if (occupied & s) // sqの地点に障害物があればこのrayは終了。
					break;
			}
		}

		return result;
	}

	// index, bits の情報を元にして、occupied の 1 のbit を いくつか 0 にする。
	// index の値を, occupied の 1のbit の位置に変換する。
	// index   [0, 1<<bits) の範囲のindex
	// bits    bit size
	// blockMask   利きのあるマスが 1 のbitboard
	// result  occupied
	Bitboard indexToOccupied(const int index, const int bits, const Bitboard& blockMask) {
		Bitboard tmpBlockMask = blockMask;
		Bitboard result = ZERO_BB;;
		for (int i = 0; i < bits; ++i) {
			const Square sq = tmpBlockMask.pop();
			if (index & (1 << i))
				result |= sq;
		}
		return result;
	}

	void initAttacks(const bool isBishop)
	{
		auto* attacks = (isBishop ? BishopAttack : RookAttack);
		auto* attackIndex = (isBishop ? BishopAttackIndex : RookAttackIndex);
		auto* blockMask = (isBishop ? BishopBlockMask : RookBlockMask);
		auto* shift = (isBishop ? BishopShiftBits : RookShiftBits);
#if defined (USE_BMI2)
#else
		auto* magic = (isBishop ? BishopMagic : RookMagic);
#endif
		int index = 0;
		for (Square sq = SQ_11; sq < SQ_NB; ++sq) {
			blockMask[sq] = (isBishop ? bishopBlockMaskCalc(sq) : rookBlockMaskCalc(sq));
			attackIndex[sq] = index;

			const int num1s = (isBishop ? BishopBlockBits[sq] : RookBlockBits[sq]);
			for (int i = 0; i < (1 << num1s); ++i) {
				const Bitboard occupied = indexToOccupied(i, num1s, blockMask[sq]);
#if defined (USE_BMI2)
				attacks[index + occupiedToIndex(occupied & blockMask[sq], blockMask[sq])] = attackCalc(sq, occupied, isBishop);
#else
				attacks[index + occupiedToIndex(occupied, magic[sq], shift[sq])] = attackCalc(sq, occupied, isBishop);
#endif
			}
			index += 1 << (64 - shift[sq]);
		}

		// 駒(飛車・角)がSQ_NBの時には利きは発生してはならない。
		blockMask[SQ_NB] = ZERO_BB; // 駒はない扱い (マスク後、ZERO_BBになる)
		attackIndex[SQ_NB] = index; // そうするとindexの先頭を指すはず
		attacks[index] = ZERO_BB;   // そこにはZERO_BBが書き込まれていると。
	}

	// Apery型の遠方駒の利きの処理で用いるテーブルの初期化
	void init_apery_attack_tables()
	{
		// 飛車の利きテーブルの初期化
		initAttacks(false);

		// 角の利きテーブルの初期化
		initAttacks(true);
	}

} // of nameless namespace


#endif // defined(USE_OLD_YANEURAOU_EFFECT)

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

// 盤上sqに駒pc(先後の区別あり)を置いたときの利き。
Bitboard effects_from(Piece pc, Square sq, const Bitboard& occ)
{
  switch (pc)
  {
  case B_PAWN: return pawnEffect(BLACK, sq);
  case B_LANCE: return lanceEffect(BLACK, sq, occ);
  case B_KNIGHT: return knightEffect(BLACK, sq);
  case B_SILVER: return silverEffect(BLACK, sq);
  case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: return goldEffect(BLACK, sq);

  case W_PAWN: return pawnEffect(WHITE, sq);
  case W_LANCE: return lanceEffect(WHITE, sq, occ);
  case W_KNIGHT: return knightEffect(WHITE, sq);
  case W_SILVER: return silverEffect(WHITE, sq);
  case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: return goldEffect(WHITE, sq);

    //　先後同じ移動特性の駒
  case B_BISHOP: case W_BISHOP: return bishopEffect(sq, occ);
  case B_ROOK:   case W_ROOK:   return rookEffect(sq, occ);
  case B_HORSE:  case W_HORSE:  return horseEffect(sq, occ);
  case B_DRAGON: case W_DRAGON: return dragonEffect(sq, occ);
  case B_KING:   case W_KING:   return kingEffect(sq);
  case B_QUEEN:  case W_QUEEN:  return horseEffect(sq, occ) | dragonEffect(sq, occ);
  case NO_PIECE: case PIECE_WHITE: return ZERO_BB; // これも入れておかないと初期化が面倒になる。

  default: UNREACHABLE; return ALL_BB;
  }
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
		SquareBB[sq].p[0] = (f <= FILE_7) ? ((uint64_t)1 << (f * 9 + r)) : 0;
		SquareBB[sq].p[1] = (f >= FILE_8) ? ((uint64_t)1 << ((f - FILE_8) * 9 + r)) : 0;
	}


	// 4) 遠方利きのテーブルの初期化
	//  thanks to Apery (Takuya Hiraoka)

	// 引数のindexをbits桁の2進数としてみなす。すなわちindex(0から2^bits-1)。
	// 与えられたmask(1の数がbitsだけある)に対して、1のbitのいくつかを(indexの値に従って)0にする。
	auto indexToOccupied = [](const int index, const int bits, const Bitboard& mask_)
	{
		auto mask = mask_;
		auto result = ZERO_BB;
		for (int i = 0; i < bits; ++i)
		{
			const Square sq = mask.pop();
			if (index & (1 << i))
			result ^= sq;
		}
		return result;
	};

	// Rook or Bishop の利きの範囲を調べて bitboard で返す。
	// occupied  障害物があるマスが 1 の bitboard
	// n = 0 右上から左下 , n = 1 左上から右下
	auto effectCalc = [](const Square square, const Bitboard& occupied, int n)
	{
		auto result = ZERO_BB;

		// 角の利きのrayと飛車の利きのray
		const SquareWithWall deltaArray[2][2] = { { SQWW_RU, SQWW_LD },{ SQWW_RD, SQWW_LU} };
		for (auto delta : deltaArray[n])
		{
			// 壁に当たるまでsqを利き方向に伸ばしていく
			for (auto sq = to_sqww(square) + delta; is_ok(sq); sq += delta)
			{
				result ^= sqww_to_sq(sq); // まだ障害物に当っていないのでここまでは利きが到達している

				if (occupied & sqww_to_sq(sq)) // sqの地点に障害物があればこのrayは終了。
					break;
			}
		}
		return result;
	};

	// pieceをsqにおいたときに利きを得るのに関係する升を返す
	auto calcBishopEffectMask = [](Square sq, int n)
	{
		Bitboard result = ZERO_BB;

		// 外周は角の利きには関係ないのでそこは除外する。
		for (Rank r = RANK_2; r <= RANK_8; ++r)
			for (File f = FILE_2; f <= FILE_8; ++f)
			{
				auto dr = rank_of(sq) - r;
				auto df = file_of(sq) - f;
				// dr == dfとdr != dfとをnが0,1とで切り替える。
				if (abs(dr) == abs(df)
					&& (!!((int)dr == (int)df) ^ n ))
						result ^= (f | r);
			}

		// sqの地点は関係ないのでクリアしておく。
		result &= ~Bitboard(sq);

		return result;
	};
  
	// 5. 飛車の縦方向の利きテーブルの初期化
	// ここでは飛車の利きを使わずに初期化しないといけない。

	for (Rank rank = RANK_1; rank <= RANK_9; ++rank)
	{
		// sq = SQ_11 , SQ_12 , ... , SQ_19
		Square sq = FILE_1 | rank;

		const int num1s = 7;
		for (int i = 0; i < (1 << num1s); ++i)
		{
			// iはsqに駒をおいたときに、その筋の2段～8段目の升がemptyかどうかを表現する値なので
			// 1ビットシフトして、1～9段目の升を表現するようにする。
			int ii = i << 1;
			Bitboard bb = ZERO_BB;
			for (int r = rank_of(sq) - 1; r >= RANK_1; --r)
			{
				bb |= file_of(sq) | (Rank)r;
				if (ii & (1 << r))
					break;
			}
			for (int r = rank_of(sq) + 1; r <= RANK_9; ++r)
			{
				bb |= file_of(sq) | (Rank)r;
				if (ii & (1 << r))
					break;
			}
			RookFileEffect[rank][i] = bb.p[0];
			// RookEffectFile[RANK_NB][x] には値を代入していないがC++の規約によりゼロ初期化されている。
		}
	}

#if defined(USE_OLD_YANEURAOU_EFFECT)

	// 角の利きテーブルの初期化
	for (int n : { 0 , 1 })
	{
		int index = 0;
		for (auto sq : SQ)
		{
			// sqの升に対してテーブルのどこを見るかのindex
			BishopEffectIndex[n][sq] = index;

			// sqの地点にpieceがあるときにその利きを得るのに関係する升を取得する
			auto& mask = BishopEffectMask[n][sq];
			mask = calcBishopEffectMask(sq, n);

			// p[0]とp[1]が被覆していると正しく計算できないのでNG。
			// Bitboardのレイアウト的に、正しく計算できるかのテスト。
			// 縦型Bitboardであるならp[0]のbit63を余らせるようにしておく必要がある。
			ASSERT_LV3(!(mask.cross_over()));

			// sqの升用に何bit情報を拾ってくるのか
			const int bits = mask.pop_count();

			// 参照するoccupied bitboardのbit数と、そのbitの取りうる状態分だけ..
			const int num = 1 << bits;

			for (int i = 0; i < num; ++i)
			{
				Bitboard occupied = indexToOccupied(i, bits, mask);
				// 初期化するテーブル
				BishopEffect[n][index + occupiedToIndex(occupied & mask, mask)] = effectCalc(sq, occupied, n);
			}
			index += num;
		}

		// 盤外(SQ_NB)に駒を配置したときに利きがZERO_BBとなるときのための処理
		BishopEffectIndex[n][SQ_NB] = index;

		// 何番まで使ったか出力してみる。(確保する配列をこのサイズに収めたいので)
		// cout << index << endl;
	}

	// 飛車の横の利き
	for (File file = FILE_1 ; file <= FILE_9 ; ++file )
	{
		// sq = SQ_11 , SQ_21 , ... , SQ_NBまで
		Square sq = file | RANK_1;
		
		const int num1s = 7;
		for (int i = 0; i < (1 << num1s); ++i)
		{
			int ii = i << 1;
			Bitboard bb = ZERO_BB;
			for (int f = file_of(sq) - 1; f >= FILE_1; --f)
			{
				bb |= (File)f | rank_of(sq);
				if (ii & (1 << f))
					break;
			}
			for (int f = file_of(sq) + 1; f <= FILE_9; ++f)
			{
				bb |= (File)f | rank_of(sq);
				if (ii & (1 << f))
					break;
			}
			RookRankEffect[file][i] = bb;
			// RookRankEffect[FILE_NB][x] には値を代入していないがC++の規約によりゼロ初期化されている。
		}
	}

#else

	// Apery型の遠方駒の利きの処理で用いるテーブルの初期化
	init_apery_attack_tables();

#endif

	// 6. 近接駒(+盤上の利きを考慮しない駒)のテーブルの初期化。
	// 上で初期化した、香・馬・飛の利きを用いる。

	for (auto sq : SQ)
	{
		// 玉は長さ1の角と飛車の利きを合成する
		KingEffectBB[sq] = bishopEffect(sq, ALL_BB) | rookEffect(sq, ALL_BB);
	}

	for (auto c : COLOR)
		for(auto sq : SQ)
			// 障害物がないときの香の利き
			// これを最初に初期化しないとlanceEffect()が使えない。
			LanceStepEffectBB[sq][c] = rookEffect(sq, ZERO_BB) & ForwardRanksBB[c][rank_of(sq)];

	for (auto c : COLOR)
		for (auto sq : SQ)
		{
			// 歩は長さ1の香の利きとして定義できる
			PawnEffectBB[sq][c] = lanceEffect(c, sq, ALL_BB);

			// 桂の利きは、歩の利きの地点に長さ1の角の利きを作って、前方のみ残す。
			Bitboard tmp = ZERO_BB;
			Bitboard pawn = lanceEffect(c, sq, ALL_BB);
			if (pawn)
			{
				Square sq2 = pawn.pop();
				Bitboard pawn2 = lanceEffect(c, sq2, ALL_BB); // さらに1つ前
				if (pawn2)
					tmp = bishopEffect(sq2, ALL_BB) & RANK_BB[rank_of(pawn2.pop())];
			}
			KnightEffectBB[sq][c] = tmp;

			// 銀は長さ1の角の利きと長さ1の香の利きの合成として定義できる。
			SilverEffectBB[sq][c] = lanceEffect(c, sq, ALL_BB) | bishopEffect(sq, ALL_BB);

			// 金は長さ1の角と飛車の利き。ただし、角のほうは相手側の歩の行き先の段でmaskしてしまう。
			Bitboard e_pawn = lanceEffect(~c, sq, ALL_BB);
			Bitboard mask = ZERO_BB;
			if (e_pawn)
				mask = RANK_BB[rank_of(e_pawn.pop())];
			GoldEffectBB[sq][c]= (bishopEffect(sq, ALL_BB) & ~mask) | rookEffect(sq, ALL_BB);

			// 障害物がないときの角と飛車の利き
			BishopStepEffectBB[sq] = bishopEffect(sq, ZERO_BB);
			RookStepEffectBB[sq]   = rookEffect(sq, ZERO_BB);

			// --- 以下のbitboard、あまり頻繁に呼び出さないので他のbitboardを合成して代用する。

			// 盤上の駒がないときのqueenの利き
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_QUEEN] = bishopEffect(sq, ZERO_BB) | rookEffect(sq, ZERO_BB);

			// 長さ1の十字
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS00] = rookEffect(sq, ALL_BB);

			// 長さ1の斜め
			// StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS45] = bishopEffect(sq, ALL_BB);
		}

	// 7) 二歩用のテーブル初期化

	for (auto sq : SQ)
		PAWN_DROP_MASKS[sq] = ~FILE_BB[SquareToFile[sq]].p[Bitboard::part(sq)];

	// 8) BetweenBB , LineBBの初期化
	{
		u16 between_index = 1;
		// BetweenBB[0] == ZERO_BBであることを保証する。

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
					Bitboard bb = ZERO_BB;
					// 間に挟まれた升を1に
					Square delta = (s2 - s1) / dist(s1, s2);
					for (Square s = s1 + delta; s != s2; s += delta)
						bb |= s;

					// ZERO_BBなら、このindexとしては0を指しておけば良いので書き換える必要ない。
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


	// 9) 王手となる候補の駒のテーブル初期化(王手の指し手生成に必要。やねうら王nanoでは削除予定)

#define FOREACH_KING(BB, EFFECT ) { for(auto sq : BB){ target|= EFFECT(sq); } }
#define FOREACH(BB, EFFECT ) { for(auto sq : BB){ target|= EFFECT(them,sq); } }
#define FOREACH_BR(BB, EFFECT ) { for(auto sq : BB) { target|= EFFECT(sq,ZERO_BB); } }

	for (auto Us : COLOR)
		for (auto ksq : SQ)
		{
			Color them = ~Us;
			auto enemyGold = goldEffect(them, ksq) & enemy_field(Us);
			Bitboard target = ZERO_BB;

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
			target = ZERO_BB;
			FOREACH(knightEffect(them, ksq) | enemyGold, knightEffect);
			CheckCandidateBB[ksq][KNIGHT - 1][Us] = target & ~Bitboard(ksq);

			// 銀も同様だが、2,3段目からの引き成りで王手になるパターンがある。(4段玉と5段玉に対して)
			target = ZERO_BB;
			FOREACH(silverEffect(them, ksq), silverEffect);
			FOREACH(enemyGold, silverEffect); // 移動先が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			FOREACH(goldEffect(them, ksq), enemy_field(Us) & silverEffect); // 移動元が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
			CheckCandidateBB[ksq][SILVER - 1][Us] = target & ~Bitboard(ksq);

			// 金
			target = ZERO_BB;
			FOREACH(goldEffect(them, ksq), goldEffect);
			CheckCandidateBB[ksq][GOLD - 1][Us] = target & ~Bitboard(ksq);

			// 角
			target = ZERO_BB;
			FOREACH_BR(bishopEffect(ksq, ZERO_BB), bishopEffect);
			FOREACH_BR(kingEffect(ksq) & enemy_field(Us), bishopEffect); // 移動先が敵陣 == 成れる == 王の動き
			FOREACH_BR(kingEffect(ksq), enemy_field(Us) & bishopEffect); // 移動元が敵陣 == 成れる == 王の動き
			CheckCandidateBB[ksq][BISHOP - 1][Us] = target & ~Bitboard(ksq);

			// 飛・龍は無条件全域。
			// ROOKのところには馬のときのことを格納

			// 馬
			target = ZERO_BB;
			FOREACH_BR(horseEffect(ksq, ZERO_BB), horseEffect);
			CheckCandidateBB[ksq][ROOK - 1][Us] = target & ~Bitboard(ksq);

			// 王(24近傍が格納される)
			target = ZERO_BB;
			FOREACH_KING(kingEffect(ksq), kingEffect);
			CheckCandidateKingBB[ksq] = target & ~Bitboard(ksq);
		}

	// 以下はBitboardとは関係はないが、Bitboardが初期化されていないと初期化できないので
	// ここから初期化しておいてやる。

	// 10. LONG_EFFECT_LIBRARYの初期化

#if defined (LONG_EFFECT_LIBRARY)
	LongEffect::init();
#endif

	// 11. 1手詰めテーブルの初期化
#if defined (USE_MATE_1PLY)
	Mate::init();
#endif

}
