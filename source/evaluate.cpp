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

  ValueKpp kpp2[SQ_NB_PLUS1][fe_end][fe_end];
  ValueKkp kkp2[SQ_NB_PLUS1][SQ_NB_PLUS1][fe_end + 1];

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

    for (i = 0; i < PIECE_NO_KING; i++)
    {
      k0 = list[i].fb;
      k1 = list[i].fw;
      sumKKP += kkp[sq_bk0][sq_wk1][k0];

      for (j = 0; j < i; j++)
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
    ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));

    return pos.side_to_move() == BLACK ? score : -score;
  }

  // 現在の局面の評価値の内訳を表示する。
  void print_eval_stat(Position& pos)
  {
    cout << "--- EVAL STAT\n";

    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto list = pos.eval_list().piece_list();

    int i, j;
    BonaPiece k0, k1;

    // 38枚の駒を表示
    for (i = 0; i < PIECE_NO_KING; ++i)
      cout << int(list[i].fb) << " = " << list[i].fb << endl;

    int32_t sumBKPP, sumWKPP, sumKKP;

    cout << "KKC : " << sq_bk0 << " " << Inv(sq_wk1) << " = " << kkp[sq_bk0][sq_wk1][fe_end] << "\n";

    sumBKPP = sumWKPP = 0;
    sumKKP = kkp[sq_bk0][sq_wk1][fe_end];

    for (i = 0; i < PIECE_NO_KING; i++)
    {
      k0 = list[i].fb;
      k1 = list[i].fw;

      cout << "KKP : " << sq_bk0 << " " << Inv(sq_wk1) << " " << k0 << " = " << kkp[sq_bk0][sq_wk1][k0] << "\n";
      sumKKP += kkp[sq_bk0][sq_wk1][k0];

      for (j = 0; j <= i; j++)
      {
        cout << "BKPP : " << sq_bk0 << " " << k0 << " " << list[j].fb << " = " << kpp[sq_bk0][k0][list[j].fb] << "\n";
        cout << "WKPP : " << sq_wk1 << " " << k1 << " " << list[j].fw << " = " << kpp[sq_wk1][k1][list[j].fw] << "\n";

        sumBKPP += kpp[sq_bk0][k0][list[j].fb];
        sumWKPP += kpp[sq_wk1][k1][list[j].fw];

        //        cout << "sumWKPP = " << sumWKPP << " sumBKPP " << sumBKPP << " sumWKPP " << sumWKPP << endl;

        // i==jにおいて0以外やったらあかんで!!
        ASSERT(!(i == j && kpp[sq_bk0][k0][list[j].fb] != 0));
      }
    }

    cout << "Material = " << pos.state()->materialValue << endl;
    cout << "sumWKPP = " << sumWKPP << " sumBKPP " << sumBKPP << " sumWKPP " << sumWKPP << endl;
    cout << "---\n";
  }

#else
  Value compute_eval(const Position& pos) { return VALUE_ZERO; }
  Value eval(const Position& pos) { return VALUE_ZERO; }
  void print_eval_stat(Position& pos) {}
#endif

  // BonaPieceの内容を表示する。手駒ならH,盤上の駒なら升目。例) HP3 (3枚目の手駒の歩)
  std::ostream& operator<<(std::ostream& os, BonaPiece bp)
  {
    if (bp < fe_hand_end)
    {
      for (auto c : COLOR)
        for (Piece pc = PAWN; pc < KING; ++pc)
          if (kpp_hand_index[c][pc].fb <= bp && bp < kpp_hand_index[c][pc].fw)
          {
#ifdef PRETTY_JP
            os << "H" << pretty(pc) << int(bp - kpp_hand_index[c][pc].fb + 1); // ex.HP3
#else
            os << "H" << pc << int(bp - kpp_hand_index[c][pc].fb + 1); // ex.HP3
#endif
            break;
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

}
