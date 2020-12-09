#include "../evaluate.h"
#include "../position.h"

using namespace std;

namespace Eval
{
#if defined (USE_PIECE_VALUE)

    // 駒の価値
  int PieceValue[PIECE_NB] =
  {
    0, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue,GoldValue,
    KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue,0,

    0, -PawnValue, -LanceValue, -KnightValue, -SilverValue, -BishopValue, -RookValue,-GoldValue,
    -KingValue, -ProPawnValue, -ProLanceValue, -ProKnightValue, -ProSilverValue, -HorseValue, -DragonValue,0,
  };

  // KINGの価値はゼロとしておく。KINGを捕獲する指し手は非合法手なので、これがプラスとして評価されたとしても
  // そのあとlegal()で引っ掛かり、実際はその指し手で進めないからこれで良い。
  int CapturePieceValue[PIECE_NB] =
  {
    VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
    BishopValue * 2, RookValue * 2, GoldValue * 2, 0 /* SEEやfutilityで用いるため王の価値は0にしておかないといけない */ ,
    ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
    HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,
 
    VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
    BishopValue * 2, RookValue * 2, GoldValue * 2, 0 ,
    ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
    HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,
  };

    // 成った時の価値の上昇分
  int ProDiffPieceValue[PIECE_NB] =
  {
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
  };
#endif


#if defined(USE_EVAL_LIST)
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

    // 通常の場合（金と小駒の成り駒を区別しない場合）
    #if !defined (DISTINGUISH_GOLDS)
    { f_gold, e_gold }, // 成歩
    { f_gold, e_gold }, // 成香
    { f_gold, e_gold }, // 成桂
    { f_gold, e_gold }, // 成銀

    // 金と小駒の成り駒を区別する場合
    #else
    { f_pro_pawn, e_pro_pawn },     // 成歩
    { f_pro_lance, e_pro_lance },   // 成香
    { f_pro_knight, e_pro_knight }, // 成桂
    { f_pro_silver, e_pro_silver }, // 成銀
    #endif

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

    // 通常の場合（金と小駒の成り駒を区別しない場合）
    #if !defined (DISTINGUISH_GOLDS)
    { e_gold, f_gold }, // 成歩
    { e_gold, f_gold }, // 成香
    { e_gold, f_gold }, // 成桂
    { e_gold, f_gold }, // 成銀

    // 金と小駒の成り駒を区別する場合
    #else
    { e_pro_pawn, f_pro_pawn },     // 成歩
    { e_pro_lance, f_pro_lance },   // 成香
    { e_pro_knight, f_pro_knight }, // 成桂
    { e_pro_silver, f_pro_silver }, // 成銀
    #endif

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
        for (PieceType pc = PAWN; pc < KING; ++pc)
        {
          // この駒種の上限(e.g. 歩 = 18)
          int kind_num = kpp_hand_index[c][pc].fw - kpp_hand_index[c][pc].fb;
          int start = kpp_hand_index[c][pc].fb;
          if (start <= bp && bp < start+kind_num * 2)
          {
            bool is_black = bp < start + kind_num;
            if (!is_black) bp = (BonaPiece)(bp - kind_num);
        #if defined (PRETTY_JP)
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
        #if defined (PRETTY_JP)
          os << Square(bp - kpp_board_index[pc].fb) << pretty(pc); // ex.32P
        #else
          os << Square(bp - kpp_board_index[pc].fb) << pc; // ex.32P
        #endif
          break;
        }
    }

    return os;
  }

#endif // defined(USE_EVAL_LIST)

} // namespace Eval