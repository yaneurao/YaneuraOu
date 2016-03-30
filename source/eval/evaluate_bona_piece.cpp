#include "../evaluate.h"
#include "../position.h"

using namespace std;

namespace Eval
{
#ifndef EVAL_NO_USE

  int PieceValue[PIECE_NB] =
  {
    0, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue,GoldValue,
    KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue,0,

    0, -PawnValue, -LanceValue, -KnightValue, -SilverValue, -BishopValue, -RookValue,-GoldValue,
    -KingValue, -ProPawnValue, -ProLanceValue, -ProKnightValue, -ProSilverValue, -HorseValue, -DragonValue,0,
  };

  int PieceValueCapture[PIECE_NB] =
  {
    VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
    BishopValue * 2, RookValue * 2, GoldValue * 2, KingValue , // SEEで使うので大きな値にしておく。
    ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
    HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,
    // KingValueの値は使わない
    VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
    BishopValue * 2, RookValue * 2, GoldValue * 2, KingValue , // SEEで使うので大きな値にしておく。
    ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
    HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,
  };

  int ProDiffPieceValue[PIECE_NB] =
  {
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
  };

  ExtBonaPiece kpp_board_index[PIECE_NB] = {
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
    { f_pawn, e_pawn },
    { f_lance, e_lance },
    { f_knight, e_knight },
    { f_silver, e_silver },
    { f_bishop, e_bishop },
    { f_rook, e_rook },
    { f_gold, e_gold },
    { f_king, e_king },
    { f_gold, e_gold }, // 成歩
    { f_gold, e_gold }, // 成香
    { f_gold, e_gold }, // 成桂
    { f_gold, e_gold }, // 成銀
    { f_horse, e_horse }, // 馬
    { f_dragon, e_dragon }, // 龍
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO }, // 金の成りはない

                                          // 後手から見た場合。fとeが入れ替わる。
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
    { e_pawn, f_pawn },
    { e_lance, f_lance },
    { e_knight, f_knight },
    { e_silver, f_silver },
    { e_bishop, f_bishop },
    { e_rook, f_rook },
    { e_gold, f_gold },
    { e_king, f_king },
    { e_gold, f_gold }, // 成歩
    { e_gold, f_gold }, // 成香
    { e_gold, f_gold }, // 成桂
    { e_gold, f_gold }, // 成銀
    { e_horse, f_horse }, // 馬
    { e_dragon, f_dragon }, // 龍
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO }, // 金の成りはない
  };

  ExtBonaPiece kpp_hand_index[COLOR_NB][KING] = {
    {
      { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
      { f_hand_pawn, e_hand_pawn },
      { f_hand_lance, e_hand_lance },
      { f_hand_knight, e_hand_knight },
      { f_hand_silver, e_hand_silver },
      { f_hand_bishop, e_hand_bishop },
      { f_hand_rook, e_hand_rook },
      { f_hand_gold, e_hand_gold },
    },
    {
      { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
      { e_hand_pawn, f_hand_pawn },
      { e_hand_lance, f_hand_lance },
      { e_hand_knight, f_hand_knight },
      { e_hand_silver, f_hand_silver },
      { e_hand_bishop, f_hand_bishop },
      { e_hand_rook, f_hand_rook },
      { e_hand_gold, f_hand_gold },
    },
  };

  // BonaPieceの内容を表示する。手駒ならH,盤上の駒なら升目。例) HP3 (3枚目の手駒の歩)
  std::ostream& operator<<(std::ostream& os, BonaPiece bp)
  {
    if (bp < fe_hand_end)
    {
      for (auto c : COLOR)
        for (Piece pc = PAWN; pc < KING; ++pc)
        {
          // この駒種の上限(e.g. 歩 = 18)
          int kind_num = kpp_hand_index[c][pc].fw - kpp_hand_index[c][pc].fb;
          int start = kpp_hand_index[c][pc].fb;
          if (start <= bp && bp < start+kind_num * 2)
          {
            bool is_black = bp < start + kind_num;
            if (!is_black) bp = (BonaPiece)(bp - kind_num);
#ifdef PRETTY_JP
            os << "手" << (is_black ? "先" : "後") << pretty(pc) << int(bp - start + 1); // ex.手先歩3
#else
            os << "H" << (is_black ? "B" : "W") << pc << int(bp - kpp_hand_index[c][pc].fb + 1); // ex.HBP3
#endif
            break;
          }
        }
    } else {
      for (auto pc : Piece())
        if (kpp_board_index[pc].fb <= bp && bp < kpp_board_index[pc].fb + SQ_NB)
        {
#ifdef PRETTY_JP
          os << Square(bp - kpp_board_index[pc].fb) << pretty(pc); // ex.32P
#else
          os << Square(bp - kpp_board_index[pc].fb) << pc; // ex.32P
#endif
          break;
        }
    }

    return os;
  }
#endif

}