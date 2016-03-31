#include "../shogi.h"

#ifdef EVAL_KPP

#include <fstream>
#include <iostream>

#include "../evaluate.h"
#include "../position.h"

using namespace std;

namespace Eval
{

// KKPファイル名
#define KKP_BIN "eval/kkp32ap.bin"

// KPPファイル名
#define KPP_BIN "eval/kpp16ap.bin"

  typedef int16_t ValueKpp;
  typedef int32_t ValueKkp;

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

    // 手駒の添字、コンバートするときにひとつ間違えてた。(๑´ڡ`๑)
    for (int k = 0; k < SQ_NB_PLUS1; ++k)
      for (int i = 1; i < fe_end; ++i)
        for (int j = 1; j < fe_end; ++j)
        {
          int i2 = i < fe_hand_end ? i - 1 : i;
          int j2 = j < fe_hand_end ? j - 1 : j;
          kpp2[k][i][j] = kpp[k][i2][j2];
        }
    for (int k1 = 0; k1 < SQ_NB_PLUS1; ++k1)
      for (int k2 = 0; k2 < SQ_NB_PLUS1; ++k2)
        for (int j = 1; j < fe_end + 1; ++j)
        {
          int j2 = j < fe_hand_end ? j - 1 : j;
          kkp2[k1][k2][j] = kkp[k1][k2][j2];
        }
    memcpy(kkp, kkp2, sizeof(kkp));
    memcpy(kpp, kpp2, sizeof(kpp));

    return;

  Error:;
    cout << "\ninfo string open evaluation file failed.\n";
    // 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
  }

  // KKPのスケール
  const int FV_SCALE_KKP = 512;
  
  // KPP,KPのスケール
  const int FV_SCALE = 32;

  // 駒割り以外の全計算
  // pos.st->BKPP,WKPP,KPPを初期化する。Position::set()で一度だけ呼び出される。(以降は差分計算)
  Value compute_eval(const Position& pos)
  {
    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto& pos_ = *const_cast<Position*>(&pos);
    auto list = pos_.eval_list()->piece_list();

    int i, j;
    BonaPiece k0, k1;
    int32_t sumBKPP, sumWKPP, sumKKP;

    sumKKP = kkp[sq_bk0][sq_wk1][fe_end];
    sumBKPP = 0;
    sumWKPP = 0;

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
    info.sumKKP = Value(sumKKP);
    info.sumBKPP = Value(sumBKPP);
    info.sumWKPP = Value(sumWKPP);

    // KKP配列の32bit化に伴い、KKP用だけ512倍しておく。(それくらいの計算精度はあるはず..)
    // 最終的なKKP = sumKKP / (FV_SCALE * FV_SCALE_KKP)

    return Value((sumBKPP + sumWKPP + sumKKP/ FV_SCALE_KKP)/FV_SCALE);
  }

  Value calc_diff_kpp(const Position& pos)
  {
    // 過去に遡って差分を計算していく。
    auto st = pos.state();

    // すでに計算されている。rootか？
    int sumKKP, sumBKPP, sumWKPP;
    if (st->sumKKP != VALUE_NONE)
    {
      sumKKP = st->sumKKP;
      sumBKPP = st->sumBKPP;
      sumWKPP = st->sumWKPP;
      goto CALC_DIFF_END;
    }

    // 遡るのは一つだけ
    // ひとつずつ遡りながらsumKPPがVALUE_NONEでないところまで探してそこからの差分を計算することは出来るが
    // レアケースだし、StateInfoにEvalListを持たせる必要が出てきて、あまり得しない。
    auto now = st;
    auto prev = st->previous;

    if (prev->sumKKP == VALUE_NONE)
    {
      // 全計算
      compute_eval(pos);
      sumKKP = now->sumKKP;
      sumBKPP = now->sumBKPP;
      sumWKPP = now->sumWKPP;
      goto CALC_DIFF_END;
    }

    // この差分を求める
    {
      sumKKP = prev->sumKKP;
      sumBKPP = prev->sumBKPP;
      sumWKPP = prev->sumWKPP;
      int k0, k1, k2, k3;

      auto sq_bk0 = pos.king_square(BLACK);
      auto sq_wk1 = Inv(pos.king_square(WHITE));

      auto now_list = pos.eval_list()->piece_list();

      int i, j;
      auto& dp = now->dirtyPiece;

      // 移動させた駒は最大2つある。その数
      int k = dp.dirty_num;

      auto dirty = dp.pieceNo[0];
      if (dirty >= PIECE_NO_KING) // 王と王でないかで場合分け
      {
        if (dirty == PIECE_NO_BKING)
        {
          // ----------------------------
          // 先手玉が移動したときの計算
          // ----------------------------

          // 現在の玉の位置に移動させて計算する。
          // 先手玉に関するKKP,KPPは全計算なので一つ前の値は関係ない。

          sumBKPP = 0;

          // このときKKPは差分で済まない。
          sumKKP = Eval::kkp[sq_bk0][sq_wk1][fe_end];

          // 片側まるごと計算
          for (i = 0; i < PIECE_NO_KING; i++)
          {
            k0 = now_list[i].fb;
            sumKKP += Eval::kkp[sq_bk0][sq_wk1][k0];

            for (j = 0; j < i; j++)
              sumBKPP += Eval::kpp[sq_bk0][k0][now_list[j].fb];
          }

          // もうひとつの駒がないならこれで計算終わりなのだが。
          if (k == 2)
          {
            // この駒についての差分計算をしないといけない。
            k1 = dp.piecePrevious[1].fw;
            k3 = dp.pieceNow[1].fw;

            dirty = dp.pieceNo[1];
            // BKPPはすでに計算済みなのでWKPPのみ。
            // WKは移動していないのでこれは前のままでいい。
            for (i = 0; i < dirty; ++i)
            {
              sumWKPP += Eval::kpp[sq_wk1][k1][now_list[i].fw];
              sumWKPP -= Eval::kpp[sq_wk1][k3][now_list[i].fw];
            }
            for (++i; i < PIECE_NO_KING; ++i)
            {
              sumWKPP += Eval::kpp[sq_wk1][k1][now_list[i].fw];
              sumWKPP -= Eval::kpp[sq_wk1][k3][now_list[i].fw];
            }
          }

        } else {
          // ----------------------------
          // 後手玉が移動したときの計算
          // ----------------------------
          ASSERT_LV3(dirty == PIECE_NO_WKING);

          sumWKPP = 0;
          sumKKP = Eval::kkp[sq_bk0][sq_wk1][fe_end];

          for (i = 0; i < PIECE_NO_KING; i++)
          {
            k0 = now_list[i].fb; // これ、KKPテーブルにk1側も入れておいて欲しい気はするが..
            k1 = now_list[i].fw;
            sumKKP += Eval::kkp[sq_bk0][sq_wk1][k0];

            for (j = 0; j < i; j++)
              sumWKPP -= Eval::kpp[sq_wk1][k1][now_list[j].fw];
          }

          if (k == 2)
          {
            k0 = dp.piecePrevious[1].fb;
            k2 = dp.pieceNow[1].fb;

            dirty = dp.pieceNo[1];
            for (i = 0; i < dirty; ++i)
            {
              sumBKPP -= Eval::kpp[sq_bk0][k0][now_list[i].fb];
              sumBKPP += Eval::kpp[sq_bk0][k2][now_list[i].fb];
            }
            for (++i; i < PIECE_NO_KING; ++i)
            {
              sumBKPP -= Eval::kpp[sq_bk0][k0][now_list[i].fb];
              sumBKPP += Eval::kpp[sq_bk0][k2][now_list[i].fb];
            }
          }
        }

      } else {
        // ----------------------------
        // 玉以外が移動したときの計算
        // ----------------------------

#define ADD_BWKPP(W0,W1,W2,W3) { \
          sumBKPP -= Eval::kpp[sq_bk0][W0][now_list[i].fb]; \
          sumWKPP += Eval::kpp[sq_wk1][W1][now_list[i].fw]; \
          sumBKPP += Eval::kpp[sq_bk0][W2][now_list[i].fb]; \
          sumWKPP -= Eval::kpp[sq_wk1][W3][now_list[i].fw]; \
}

        if (k == 1)
        {
          // 移動した駒が一つ。

          k0 = dp.piecePrevious[0].fb;
          k1 = dp.piecePrevious[0].fw;
          k2 = dp.pieceNow[0].fb;
          k3 = dp.pieceNow[0].fw;

          // KKP差分
          sumKKP -= Eval::kkp[sq_bk0][sq_wk1][k0];
          sumKKP += Eval::kkp[sq_bk0][sq_wk1][k2];

          // KP値、要らんのでi==dirtyを除く
          for (i = 0; i < dirty; ++i)
            ADD_BWKPP(k0, k1, k2, k3);
          for (++i; i < PIECE_NO_KING; ++i)
            ADD_BWKPP(k0, k1, k2, k3);

        } else if (k == 2) {

          // 移動する駒が王以外の2つ。
          PieceNo dirty2 = dp.pieceNo[1];
          if (dirty > dirty2) swap(dirty, dirty2);
          // PIECE_NO_ZERO <= dirty < dirty2 < PIECE_NO_KING
          // にしておく。

          k0 = dp.piecePrevious[0].fb;
          k1 = dp.piecePrevious[0].fw;
          k2 = dp.pieceNow[0].fb;
          k3 = dp.pieceNow[0].fw;

          int m0, m1, m2, m3;
          m0 = dp.piecePrevious[1].fb;
          m1 = dp.piecePrevious[1].fw;
          m2 = dp.pieceNow[1].fb;
          m3 = dp.pieceNow[1].fw;

          // KKP差分
          sumKKP -= Eval::kkp[sq_bk0][sq_wk1][k0];
          sumKKP += Eval::kkp[sq_bk0][sq_wk1][k2];
          sumKKP -= Eval::kkp[sq_bk0][sq_wk1][m0];
          sumKKP += Eval::kkp[sq_bk0][sq_wk1][m2];

          // KPP差分
          for (i = 0; i < dirty; ++i)
          {
            ADD_BWKPP(k0, k1, k2, k3);
            ADD_BWKPP(m0, m1, m2, m3);
          }
          for (++i; i < dirty2; ++i)
          {
            ADD_BWKPP(k0, k1, k2, k3);
            ADD_BWKPP(m0, m1, m2, m3);
          }
          for (++i; i < PIECE_NO_KING; ++i)
          {
            ADD_BWKPP(k0, k1, k2, k3);
            ADD_BWKPP(m0, m1, m2, m3);
          }

          sumBKPP -= Eval::kpp[sq_bk0][k0][m0];
          sumWKPP += Eval::kpp[sq_wk1][k1][m1];
          sumBKPP += Eval::kpp[sq_bk0][k2][m2];
          sumWKPP -= Eval::kpp[sq_wk1][k3][m3];

        }
      }
    }

    now->sumKKP = sumKKP;
    now->sumBKPP = sumBKPP;
    now->sumWKPP = sumWKPP;

    // 差分計算終わり
  CALC_DIFF_END:;
    return (Value)((sumBKPP + sumWKPP + sumKKP / FV_SCALE_KKP) / FV_SCALE);
  }

  // 評価関数
  Value evaluate(const Position& pos)
  {
    // 差分計算
    auto score = calc_diff_kpp(pos) + pos.state()->materialValue;

    // 非差分計算
//    auto score = compute_eval(pos) + pos.state()->materialValue;

    ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));

    // 差分計算と非差分計算との計算結果が合致するかのテスト。(さすがに重いのでコメントアウトしておく)
    //    ASSERT_LV5(score == compute_eval(pos) + pos.state()->materialValue);

    return pos.side_to_move() == BLACK ? score : -score;
  }

  // 現在の局面の評価値の内訳を表示する。
  void print_eval_stat(Position& pos)
  {
    cout << "--- EVAL STAT\n";

    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto list = pos.eval_list()->piece_list();

    int i, j;
    BonaPiece k0, k1;

    // 38枚の駒を表示
    for (i = 0; i < PIECE_NO_KING; ++i)
      cout << int(list[i].fb) << " = " << list[i].fb << " , " << int(list[i].fw) << " =  " << list[i].fw << endl;

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
    cout << "sumKKP = " << sumKKP << " sumBKPP " << sumBKPP << " sumWKPP " << sumWKPP << endl;
    cout << "---\n";
  }

}

#endif // EVAL_KPP
