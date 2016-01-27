#include <fstream>
#include <iostream>

#include "evaluate.h"
#include "position.h"

// KKPファイル名
#define KKP_BIN "kkp32ap.bin"

// KPPファイル名
#define KPP_BIN "kpp16ap.bin"

using namespace std;

namespace Eval {

  typedef int16_t ValueKpp;
  typedef int32_t ValueKkp;

#ifdef EVAL_KPP
  // KPP
  ValueKpp kpp[SQ_NB_PLUS1][fe_end][fe_end];

  // KKP
  // fe_endまでにしてしまう。これによりPiece番号がKPPとKKPとで共通になる。
  // さらに、2パラ目のKはInv(K2)を渡すものとすればkppと同じInv(K2)で済む。
  // [][][fe_end]のところはKK定数にしてあるものとする。
  ValueKkp kkp[SQ_NB_PLUS1][SQ_NB_PLUS1][fe_end + 1];

  // 評価関数ファイルを読み込む
  void load_eval()
  {
    fstream fs;
    size_t size;

    fs.open(KPP_BIN, ios::in | ios::binary);
    if (fs.fail())
      goto Error;
    size = SQ_NB_PLUS1 * (int)fe_end * (int)fe_end * (int)sizeof(ValueKpp);
    fs.read((char*)&kpp, size);
    if (fs.fail())
      goto Error;
    fs.close();
    size = SQ_NB_PLUS1 * (int)SQ_NB_PLUS1 * ((int)(fe_end)+1) * (int)sizeof(ValueKkp);
    fs.open(KKP_BIN, ios::in | ios::binary);
    if (fs.fail())
      goto Error;
    fs.read((char*)&kkp, size);
    if (fs.fail())
      goto Error;
    fs.close();

    return;

  Error:;
    cout << "\ninfo string open evaluation file failed.\n";
    // 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
  }
#else
  void load_eval() {}
#endif

  // Bona6の駒割りを初期値に。それぞれの駒の価値。
  enum {
    PawnValue = 86,
    LanceValue = 227,
    KnightValue = 256,
    SilverValue = 365,
    GoldValue = 439,
    BishopValue = 563,
    RookValue = 629,
    ProPawnValue = 540,
    ProLanceValue = 508,
    ProKnightValue = 517,
    ProSilverValue = 502,
    HorseValue = 826,
    DragonValue = 942,
    KingValue = 15000,
  };

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
    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue,
    VALUE_ZERO, HorseValue - BishopValue, DragonValue - RookValue,

    VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue,
    VALUE_ZERO, HorseValue - BishopValue, DragonValue - RookValue,
  };

  // 駒得だけの評価関数
  // 手番側から見た評価値
  Value material(const Position& pos)
  {
    int v = VALUE_ZERO;

    for(auto i : SQ)
      v = v + PieceValue[pos.piece_on(i)];

    // 手駒も足しておく
    for (auto c : COLOR)
      for (auto pc = PAWN; pc < PIECE_HAND_NB; ++pc)
        v += (c == BLACK ? 1 : -1) * Value(hand_count(pos.hand_of(c), pc) * PieceValue[pc]);

    return (Value)v;
  }

#ifdef EVAL_KPP
  // pos.st->BKPP,WKPP,KPPを初期化する。Position::set()で一度だけ呼び出される。(以降は差分計算)
  Value compute_eval(const Position& pos)
  {
    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto& pos_ = *const_cast<Position*>(&pos);
    auto list = pos_.eval_list().piece_list();

    int i, j;
    BonaPiece k0, k1;
    int32_t sumBKPP, sumWKPP, sumKKP;

    sumBKPP = 0;
    sumWKPP = 0;
    sumKKP = kkp[sq_bk0][sq_wk1][fe_end];

    for (i = 0; i < PIECE_NO_NB; i++)
    {
      k0 = list[i].fb;

      k1 = list[i].fw;
      sumKKP += kkp[sq_bk0][sq_wk1][k0];

      for (j = 0; j <= i; j++)
      {
        sumBKPP += kpp[sq_bk0][k0][list[j].fb];
        sumWKPP -= kpp[sq_wk1][k1][list[j].fw];
      }
    }

    auto& info = *pos.state();
    info.sumBKPP = Value(sumBKPP);
    info.sumWKPP = Value(sumWKPP);
    info.sumKKP = Value(sumKKP);

    const int FV_SCALE = 32;

    // KKP配列の32bit化に伴い、KKP用だけ512倍しておく。(それくらいの計算精度はあるはず..)
    // 最終的なKKP = sumKKP / (FV_SCALE * FV_SCALE_KKP)
    const int FV_SCALE_KKP = 512;

//    cout << "sumBKPP ... = " << sumBKPP << " , " << sumWKPP << " , " << sumKKP << endl;

    return (Value)((sumBKPP + sumWKPP + sumKKP/ FV_SCALE_KKP)/FV_SCALE);
  }

  // 評価関数
  Value eval(const Position& pos)
  {
    auto score = compute_eval(pos) + pos.state()->materialValue;

    return pos.side_to_move() == BLACK ? score : -score;
  }
#else
  Value compute_eval(const Position& pos) { return VALUE_ZERO; }
  Value eval(const Position& pos) { return VALUE_ZERO; }
#endif

  // BonaPieceの内容を表示する。手駒ならH,盤上の駒なら升目。例) HP3 (3枚目の手駒の歩)
  std::ostream& operator<<(std::ostream& os, BonaPiece bp)
  {
    // まだ

    return os;
  }

}
