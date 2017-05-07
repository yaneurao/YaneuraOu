#include <algorithm>
#include <sstream>
#include <iostream>

#include "shogi.h"
#include "bitboard.h"
#include "extra/long_effect.h"
#include "extra/mate1ply.h"

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

Bitboard InFrontBB[COLOR_NB][RANK_NB] = {
  { ZERO_BB,RANK1_BB, RANK1_BB | RANK2_BB , RANK1_BB | RANK2_BB | RANK3_BB , RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
  ~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB) , ~(RANK9_BB | RANK8_BB | RANK7_BB),~(RANK9_BB | RANK8_BB),~RANK9_BB },
  { ~RANK1_BB , ~(RANK1_BB | RANK2_BB) , ~(RANK1_BB | RANK2_BB | RANK3_BB),~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
  RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB , RANK9_BB | RANK8_BB | RANK7_BB , RANK9_BB | RANK8_BB , RANK9_BB , ZERO_BB }
};

// ----- Bitboard tables

// sqの升が1であるbitboard
Bitboard SquareBB[SQ_NB_PLUS1];

// 近接駒の利き
Bitboard StepEffectsBB[SQ_NB_PLUS1][COLOR_NB][16];

// 香の利き
Bitboard LanceEffect[COLOR_NB][SQ_NB_PLUS1][128];

// 角の利き
Bitboard BishopEffect[20224+1];
Bitboard BishopEffectMask[SQ_NB_PLUS1];
int BishopEffectIndex[SQ_NB_PLUS1];

// 飛車の利き
Bitboard RookEffect[495616+1];
Bitboard RookEffectMask[SQ_NB_PLUS1];
int RookEffectIndex[SQ_NB_PLUS1];

// 歩が打てる筋を得るためのBitboard
// bit0..9筋に歩が打てないなら1 , bit1..8筋に , … , bit8..1筋に歩が打てないなら1
// というbit列をindexとして、歩の打てるBitboardを返すためのテーブル。
Bitboard PAWN_DROP_MASK_BB[0x200][COLOR_NB];

Bitboard BetweenBB[SQ_NB_PLUS1][SQ_NB_PLUS1];
Bitboard LineBB[SQ_NB_PLUS1][SQ_NB_PLUS1];
Bitboard CheckCandidateBB[SQ_NB_PLUS1][KING][COLOR_NB];

// SquareからSquareWithWallへの変換テーブル
SquareWithWall sqww_table[SQ_NB_PLUS1];

int Slide[SQ_NB_PLUS1] = {
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
  auto effectCalc = [](const Square square, const Bitboard& occupied, const Piece piece)
  {
    auto result = ZERO_BB;

    // 角の利きのrayと飛車の利きのray
    const SquareWithWall deltaArray[2][4] = { { SQWW_RU, SQWW_RD, SQWW_LU, SQWW_LD },{ SQWW_U, SQWW_D, SQWW_R, SQWW_L } };
    for (auto delta : deltaArray[(piece == BISHOP) ? 0 : 1])
      // 壁に当たるまでsqを利き方向に伸ばしていく
      for (auto sq = to_sqww(square) + delta; is_ok(sq) ; sq += delta)
      {
        result ^= sqww_to_sq(sq); // まだ障害物に当っていないのでここまでは利きが到達している

        if (occupied & sqww_to_sq(sq)) // sqの地点に障害物があればこのrayは終了。
          break;
      }
    return result;
  };

  // pieceをsqにおいたときに利きを得るのに関係する升を返す
  auto calcEffectMask = [](Square sq, Piece piece)
  {
    Bitboard result;
    if (piece == BISHOP) {

      result = ZERO_BB;

      for (Rank r = RANK_2; r <= RANK_8; ++r)
        for (File f = FILE_2; f <= FILE_8; ++f)
          // 外周は角の利きには関係ないのでそこは除外する。
          if (abs(rank_of(sq) - r) == abs(file_of(sq) - f))
            result ^= (f | r);
    } else {

      ASSERT_LV3(piece == ROOK);

      result = RANK_BB[rank_of(sq)] ^ FILE_BB[file_of(sq)];

      // 外周に居ない限り、その外周升は利きの計算には関係ない。
      if (file_of(sq) != FILE_1) { result &= ~FILE1_BB; }
      if (file_of(sq) != FILE_9) { result &= ~FILE9_BB; }
      if (rank_of(sq) != RANK_1) { result &= ~RANK1_BB; }
      if (rank_of(sq) != RANK_9) { result &= ~RANK9_BB; }
    }

    // sqの地点は関係ないのでクリアしておく。
    result &= ~Bitboard(sq);

    return result;
  };
  
  // 角と飛車の利きテーブルの初期化
  for (Piece pc : {BISHOP,ROOK} )
  {
    // 初期化するテーブルのアドレス
    Bitboard* effects = (pc == BISHOP) ? BishopEffect : RookEffect;

    // sqの升に対してテーブルのどこを引くかのindex
    int* effectIndex = (pc == BISHOP) ? BishopEffectIndex : RookEffectIndex;

    // 利きを得るために関係する升
    Bitboard* masks = (pc == BISHOP) ? BishopEffectMask : RookEffectMask;

    int index = 0;

    for (auto sq : SQ)
    {
      effectIndex[sq] = index;

      // sqの地点にpieceがあるときにその利きを得るのに関係する升を取得する
      masks[sq] = calcEffectMask(sq, pc);

      // p[0]とp[1]が被覆していると正しく計算できないのでNG。
      // Bitboardのレイアウト的に、正しく計算できるかのテスト。
      // 縦型Bitboardであるならp[0]のbit63を余らせるようにしておく必要がある。
      ASSERT_LV3(!(masks[sq].p[0] & masks[sq].p[1]));

      // sqの升用に何bit情報を拾ってくるのか
      const int bits = masks[sq].pop_count();

      // 参照するoccupied bitboardのbit数と、そのbitの取りうる状態分だけ..
      const int num = 1 << bits;

      for (int i = 0; i < num; ++i)
      {
        Bitboard occupied = indexToOccupied(i, bits, masks[sq]);
        effects[index + occupiedToIndex(occupied & masks[sq], masks[sq])] = effectCalc(sq, occupied, pc);
      }
      index += num;
    }

    // 盤外(SQ_NB)に駒を配置したときに利きがZERO_BBとなるときのための処理
    effectIndex[SQ_NB] = index;
  }

  
  // 5. 香の利きテーブルの初期化
  // 上で初期化した飛車の利きを用いる。

  for (auto c : COLOR)
    for (auto sq : SQ)
    {
      const Bitboard blockMask = FILE_BB[file_of(sq)] & ~(RANK1_BB | RANK9_BB);
      const int num1s = 7;
      for (int i = 0; i < (1 << num1s); ++i) {
        Bitboard occupied = indexToOccupied(i, num1s, blockMask);
        LanceEffect[c][sq][i] = rookEffect(sq, occupied) & InFrontBB[c][rank_of(sq)];
      }
    }

  // 6. 近接駒(+盤上の利きを考慮しない駒)のテーブルの初期化。
  // 上で初期化した、香・馬・飛の利きを用いる。

  for (auto c : COLOR)
    for (auto sq : SQ)
    {
      // 歩は長さ1の香の利きとして定義できる
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_PAWN] = lanceEffect(c, sq, ALL_BB);

      // 障害物がないときの香の利き
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_LANCE] = lanceEffect(c, sq, ZERO_BB);

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
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_KNIGHT] = tmp;

      // 銀は長さ1の角の利きと長さ1の香の利きの合成として定義できる。
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_SILVER] = lanceEffect(c, sq, ALL_BB) | bishopEffect(sq, ALL_BB);

      // 金は長さ1の角と飛車の利き。ただし、角のほうは相手側の歩の行き先の段でmaskしてしまう。
      Bitboard e_pawn = lanceEffect(~c, sq, ALL_BB);
      Bitboard mask = ZERO_BB;
      if (e_pawn)
        mask = RANK_BB[rank_of(e_pawn.pop())];
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_GOLD] = (bishopEffect(sq, ALL_BB) & ~mask) | rookEffect(sq, ALL_BB);

      // 障害物がないときの角と飛車の利き
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_BISHOP] = bishopEffect(sq, ZERO_BB);
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_ROOK] = rookEffect(sq, ZERO_BB);

      // 玉は長さ1の角と飛車の利きを合成する
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_HDK] = bishopEffect(sq, ALL_BB) | rookEffect(sq, ALL_BB);

      // 盤上の駒がないときのqueenの利き
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_QUEEN] = bishopEffect(sq, ZERO_BB) | rookEffect(sq, ZERO_BB);

      // 長さ1の十字
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS00] = rookEffect(sq, ALL_BB);

      // 長さ1の斜め
      StepEffectsBB[sq][c][PIECE_TYPE_BITBOARD_CROSS45] = bishopEffect(sq, ALL_BB);
    }

  // 7) 二歩用のテーブル初期化

  for (int i = 0; i <= 0x1ff; ++i)
  {
    Bitboard b = ZERO_BB;
    for (int k = 0; k < 9; ++k)
      if ((i & (1 << k)) == 0)
        b |= FILE_BB[k];

    PAWN_DROP_MASK_BB[i][BLACK] = b & rank1_n_bb(WHITE, RANK_8); // 2～9段目まで
    PAWN_DROP_MASK_BB[i][WHITE] = b & rank1_n_bb(BLACK, RANK_8); // 1～8段目まで
  }

  // 8) BetweenBB , LineBBの初期化
  
  for (auto s1 : SQ)
    for (auto s2 : SQ)
    {
      // 方角を用いるテーブルの初期化
      if (Effect8::directions_of(s1,s2))
      {
        // 間に挟まれた升を1に
        Square delta = (s2 - s1) / dist(s1 , s2);
        for (Square s = s1 + delta; s != s2; s += delta)
          BetweenBB[s1][s2] |= s;

        // 間に挟まれてない升も1に
        LineBB[s1][s2] = BetweenBB[s1][s2];

        // 壁に当たるまでs1から-delta方向に延長
        for (Square s = s1; dist(s, s + delta) <= 1; s -= delta) LineBB[s1][s2] |= s;

        // 壁に当たるまでs2から+delta方向に延長
        for (Square s = s2; dist(s, s - delta) <= 1; s += delta) LineBB[s1][s2] |= s;
      }
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
      Bitboard target;

      // 歩で王手になる可能性のあるものは、敵玉から２つ離れた歩(不成での移動) + ksqに敵の金をおいた範囲(enemyGold)に成りで移動できる
      target = ZERO_BB;
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
      FOREACH(silverEffect(them, ksq) , silverEffect);
      FOREACH(enemyGold, silverEffect); // 移動先が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
      FOREACH(goldEffect(them, ksq), enemy_field(Us) & silverEffect); // 移動元が敵陣 == 成れる == 金になるので、敵玉の升に敵の金をおいた利きに成りで移動すると王手になる。
      CheckCandidateBB[ksq][SILVER - 1][Us] = target & ~Bitboard(ksq);

      // 金
      target = ZERO_BB;
      FOREACH(goldEffect(them, ksq), goldEffect);
      CheckCandidateBB[ksq][GOLD - 1][Us] = target & ~Bitboard(ksq);

      // 角
      target = ZERO_BB;
      FOREACH_BR(bishopEffect(ksq,ZERO_BB), bishopEffect);
      FOREACH_BR(kingEffect(ksq) & enemy_field(Us), bishopEffect); // 移動先が敵陣 == 成れる == 王の動き
      FOREACH_BR(kingEffect(ksq) , enemy_field(Us) & bishopEffect); // 移動元が敵陣 == 成れる == 王の動き
      CheckCandidateBB[ksq][BISHOP - 1][Us] = target & ~Bitboard(ksq);

      // 飛・龍は無条件全域。
      // ROOKのところには馬のときのことを格納

      // 馬
      target = ZERO_BB;
      FOREACH_BR(horseEffect(ksq, ZERO_BB), horseEffect);
      CheckCandidateBB[ksq][ROOK - 1][Us] = target & ~Bitboard(ksq);
      
      // 王(24近傍が格納される)
      target = ZERO_BB;
      FOREACH_KING(kingEffect(ksq) , kingEffect);
      CheckCandidateBB[ksq][KING - 1][Us] = target & ~Bitboard(ksq);
    }

  // 10. LONG_EFFECT_LIBRARYの初期化
  
#ifdef LONG_EFFECT_LIBRARY
  LongEffect::init();
#endif

  // 11. 1手詰めテーブルの初期化
#ifdef USE_MATE_1PLY
  Mate1Ply::init();
#endif
}

