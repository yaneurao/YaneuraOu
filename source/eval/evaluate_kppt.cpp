#include "../shogi.h"

// Apery WCSC26の評価関数バイナリを読み込むための仕組み
#ifdef EVAL_KPPT

#include <fstream>
#include <iostream>

#include "../evaluate.h"
#include "../position.h"

using namespace std;

namespace Eval
{

// KKファイル名
#define KK_BIN "/KK_synthesized.bin"
  
// KKPファイル名
#define KKP_BIN "/KKP_synthesized.bin"

// KPPファイル名
#define KPP_BIN "/KPP_synthesized.bin"

  // 手番込みの評価値。[0]が手番に無縁な部分。[1]が手番があるときの上乗せ
  //  (これは先手から見たものではなく先後に依存しないボーナス)。
  // 先手から見て、先手の手番があるときの評価値 =  [0] + [1]
  // 先手から見て、先手の手番がないときの評価値 =  [0] - [1]
  // 後手から見て、後手の手番があるときの評価値 = -[0] + [1]
  typedef std::array<int32_t, 2> ValueKk;
  typedef std::array<int16_t, 2> ValueKpp;
  typedef std::array<int32_t, 2> ValueKkp;

  // 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
  // ここではやらない。ゆえに片側の玉や、駒落ちの盤面には対応出来ない。

  // KK
  ValueKk kk[SQ_NB][SQ_NB];

  // KPP
  ValueKpp kpp[SQ_NB][fe_end][fe_end];

  // KKP
  ValueKkp kkp[SQ_NB][SQ_NB][fe_end];

  // 評価関数ファイルを読み込む
  void load_eval()
  {
    {
      // KK
      std::ifstream ifsKK((string)Options["EvalDir"] + KK_BIN, std::ios::binary);
      if (ifsKK) ifsKK.read(reinterpret_cast<char*>(kk), sizeof(kk));
      else goto Error;

      // KKP
      std::ifstream ifsKKP((string)Options["EvalDir"] + KKP_BIN, std::ios::binary);
      if (ifsKKP) ifsKKP.read(reinterpret_cast<char*>(kkp), sizeof(kkp));
      else goto Error;

      // KPP
      std::ifstream ifsKPP((string)Options["EvalDir"] + KPP_BIN, std::ios::binary);
      if (ifsKPP) ifsKPP.read(reinterpret_cast<char*>(kpp), sizeof(kpp));
      else goto Error;
    }
    
    return;

  Error:;
    // 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
    cout << "\ninfo string Error! open evaluation file failed.\n";
    exit(EXIT_FAILURE);
  }
  
  // KP,KPP,KKPのスケール
  const int FV_SCALE = 32;

  // 駒割り以外の全計算
  // pos.st->BKPP,WKPP,KPPを初期化する。Position::set()で一度だけ呼び出される。(以降は差分計算)
  // 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
  Value compute_eval(const Position& pos)
  {
    Square sq_bk = pos.king_square(BLACK);
    Square sq_wk = pos.king_square(WHITE);
    const auto* ppkppb = kpp[sq_bk];
    const auto* ppkppw = kpp[Inv(sq_wk)];

    auto& pos_ = *const_cast<Position*>(&pos);

    auto list_fb = pos_.eval_list()->piece_list_fb();
    auto list_fw = pos_.eval_list()->piece_list_fw();

    int i, j;
    BonaPiece k0, k1,l0,l1;

    // 評価値の合計
    EvalSum sum;


#if defined(USE_SSE2)
    // sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
    sum.m[0] = _mm_setzero_si128();
#else
    sum.p[0][0] = sum.p[0][1] = sum.p[1][0] = sum.p[1][1] = 0;
#endif

    // KK
    sum.p[2] = kk[sq_bk][sq_wk];

    for (i = 0; i < PIECE_NO_KING; ++i)
    {
      k0 = list_fb[i];
      k1 = list_fw[i];
      const auto* pkppb = ppkppb[k0];
      const auto* pkppw = ppkppw[k1];
      for (j = 0; j < i; ++j)
      {
        l0 = list_fb[j];
        l1 = list_fw[j];

#if defined(SSE2)
        // SSEによる実装

        // pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
        __m128i tmp;
        tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
        tmp = _mm_cvtepi16_epi32(tmp);
        sum.m[0] = _mm_add_epi32(sum.m[0], tmp);
#else
        sum.p[0] += pkppb[l0];
        sum.p[1] += pkppw[l1];
#endif
      }
      sum.p[2] += kkp[sq_bk][sq_wk][k0];
    }

    auto& info = *pos.state();
    info.sum = sum;

    sum.p[2][0] += pos.state()->materialValue * FV_SCALE;

    return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
  }

#ifdef USE_EVAL_HASH
  EvaluateHashTable g_evalTable;
#endif

  Value calc_diff_kpp(const Position& pos)
  {
    // 過去に遡って差分を計算していく。
    auto st = pos.state();

    // すでに計算されている。rootか？
    EvalSum sum;
    if (st->sum.p[2][0] != INT_MAX)
    {
      sum = st->sum;
      goto CALC_DIFF_END;
    }

    {
#ifdef USE_EVAL_HASH
      // evaluate hash tableにはあるかも。

      const Key keyExcludeTurn = pos.state()->key() & ~1; // 手番を消した局面hash key
      EvalSum entry = *g_evalTable[keyExcludeTurn];       // atomic にデータを取得する必要がある。
      entry.decode();
      if (entry.key == keyExcludeTurn)
      {
        // あった！
        sum = st->sum = entry;
        goto CALC_DIFF_END;
      }
#endif

      // 遡るのは一つだけ
      // ひとつずつ遡りながらsumKPPがVALUE_NONEでないところまで探してそこからの差分を計算することは出来るが
      // レアケースだし、StateInfoにEvalListを持たせる必要が出てきて、あまり得しない。
      auto now = st;
      auto prev = st->previous;

      if (prev->sum.p[2][0] == INT_MAX)
      {
        // 全計算
        return  compute_eval(pos);
      }

      // この差分を求める
      {
        sum = prev->sum;

        int k0, k1, k2, k3;

        auto sq_bk0 = pos.king_square(BLACK);
        auto sq_wk0 = pos.king_square(WHITE);
        auto sq_wk1 = Inv(pos.king_square(WHITE));

        auto now_list_fb = pos.eval_list()->piece_list_fb();
        auto now_list_fw = pos.eval_list()->piece_list_fw();

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

            // BKPP
            sum.p[0][0] = 0;
            sum.p[0][1] = 0;

            // このときKKPは差分で済まない。
            sum.p[2] = Eval::kk[sq_bk0][sq_wk0];

            // 片側まるごと計算
            for (i = 0; i < PIECE_NO_KING; i++)
            {
              k0 = now_list_fb[i];
              sum.p[2] += Eval::kkp[sq_bk0][sq_wk0][k0];

              for (j = 0; j < i; j++)
                sum.p[0] += Eval::kpp[sq_bk0][k0][now_list_fb[j]];
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
                sum.p[1] -= Eval::kpp[sq_wk1][k1][now_list_fw[i]];
                sum.p[1] += Eval::kpp[sq_wk1][k3][now_list_fw[i]];
              }
              for (++i; i < PIECE_NO_KING; ++i)
              {
                sum.p[1] -= Eval::kpp[sq_wk1][k1][now_list_fw[i]];
                sum.p[1] += Eval::kpp[sq_wk1][k3][now_list_fw[i]];
              }
            }

          } else {
            // ----------------------------
            // 後手玉が移動したときの計算
            // ----------------------------
            ASSERT_LV3(dirty == PIECE_NO_WKING);

            // WKPP
            sum.p[1][0] = 0;
            sum.p[1][1] = 0;
            sum.p[2] = Eval::kk[sq_bk0][sq_wk0];

            for (i = 0; i < PIECE_NO_KING; i++)
            {
              k0 = now_list_fb[i]; // これ、KKPテーブルにk1側も入れておいて欲しい気はするが..
              k1 = now_list_fw[i];
              sum.p[2] += Eval::kkp[sq_bk0][sq_wk0][k0];

              for (j = 0; j < i; j++)
                sum.p[1] += Eval::kpp[sq_wk1][k1][now_list_fw[j]];
            }

            if (k == 2)
            {
              k0 = dp.piecePrevious[1].fb;
              k2 = dp.pieceNow[1].fb;

              dirty = dp.pieceNo[1];
              for (i = 0; i < dirty; ++i)
              {
                sum.p[0] -= Eval::kpp[sq_bk0][k0][now_list_fb[i]];
                sum.p[0] += Eval::kpp[sq_bk0][k2][now_list_fb[i]];
              }
              for (++i; i < PIECE_NO_KING; ++i)
              {
                sum.p[0] -= Eval::kpp[sq_bk0][k0][now_list_fb[i]];
                sum.p[0] += Eval::kpp[sq_bk0][k2][now_list_fb[i]];
              }
            }
          }

        } else {
          // ----------------------------
          // 玉以外が移動したときの計算
          // ----------------------------

#define ADD_BWKPP(W0,W1,W2,W3) { \
          sum.p[0] -= Eval::kpp[sq_bk0][W0][now_list_fb[i]]; \
          sum.p[1] -= Eval::kpp[sq_wk1][W1][now_list_fw[i]]; \
          sum.p[0] += Eval::kpp[sq_bk0][W2][now_list_fb[i]]; \
          sum.p[1] += Eval::kpp[sq_wk1][W3][now_list_fw[i]]; \
}

          if (k == 1)
          {
            // 移動した駒が一つ。

            k0 = dp.piecePrevious[0].fb;
            k1 = dp.piecePrevious[0].fw;
            k2 = dp.pieceNow[0].fb;
            k3 = dp.pieceNow[0].fw;

            // KKP差分
            sum.p[2] -= Eval::kkp[sq_bk0][sq_wk0][k0];
            sum.p[2] += Eval::kkp[sq_bk0][sq_wk0][k2];

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
            sum.p[2] -= Eval::kkp[sq_bk0][sq_wk0][k0];
            sum.p[2] += Eval::kkp[sq_bk0][sq_wk0][k2];
            sum.p[2] -= Eval::kkp[sq_bk0][sq_wk0][m0];
            sum.p[2] += Eval::kkp[sq_bk0][sq_wk0][m2];

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

            sum.p[0] -= Eval::kpp[sq_bk0][k0][m0];
            sum.p[1] -= Eval::kpp[sq_wk1][k1][m1];
            sum.p[0] += Eval::kpp[sq_bk0][k2][m2];
            sum.p[1] += Eval::kpp[sq_wk1][k3][m3];

          }
        }
      }

      now->sum = sum;
      // 差分計算終わり

#ifdef USE_EVAL_HASH
    // せっかく計算したのでevaluate hash tableに保存しておく。
      sum.key = keyExcludeTurn;
      sum.encode();
      *g_evalTable[keyExcludeTurn] = sum;
#endif
    }

  CALC_DIFF_END:;
    sum.p[2][0] += pos.state()->materialValue * FV_SCALE;
    return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
 }

  // 評価関数
  Value evaluate(const Position& pos)
  {
    // 差分計算
    auto score = calc_diff_kpp(pos);

    // 非差分計算
//    auto score = compute_eval(pos);

    ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));

    // 差分計算と非差分計算との計算結果が合致するかのテスト。(さすがに重いのでコメントアウトしておく)
    // ASSERT_LV5(score == compute_eval(pos));

    return score;
  }

  void evaluate_with_no_return(const Position& pos)
  {
    // まだ評価値が計算されていないなら
    if (pos.state()->sum.p[2][0] == INT_MAX)
      evaluate(pos);
  }


  // null move後のevaluate()
  // 手番を反転させたときの評価値を返す。
  Value evaluate_nullmove(const Position& pos)
  {
    auto sum = pos.state()->sum;
    if (sum.p[2][0] != INT_MAX)
    {
      // 計算済みなので現在の手番から計算して計算終了。
      sum.p[2][0] += pos.state()->materialValue * FV_SCALE;
      return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
    }
    return compute_eval(pos);
  }

  // 現在の局面の評価値の内訳を表示する。
  void print_eval_stat(Position& pos)
  {
    cout << "--- EVAL STAT\n";

    Square sq_bk = pos.king_square(BLACK);
    Square sq_wk = pos.king_square(WHITE);
    const auto* ppkppb = kpp[sq_bk];
    const auto* ppkppw = kpp[Inv(sq_wk)];

    auto& pos_ = *const_cast<Position*>(&pos);

    auto list_fb = pos_.eval_list()->piece_list_fb();
    auto list_fw = pos_.eval_list()->piece_list_fw();

    int i, j;
    BonaPiece k0, k1, l0, l1;

    // 38枚の駒を表示
    for (i = 0; i < PIECE_NO_KING; ++i)
      cout << int(list_fb[i]) << " = " << list_fb[i] << " , " << int(list_fw[i]) << " =  " << list_fw[i] << endl;

    // 評価値の合計
    EvalSum sum;

#if defined(USE_SSE2)
    // sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
    sum.m[0] = _mm_setzero_si128();
#else
    sum.p[0][0] = sum.p[0][1] = sum.p[1][0] = sum.p[1][1] = 0;
#endif

    // KK
    sum.p[2] = kk[sq_bk][sq_wk];
    cout << "KKC : " << sq_bk << " " << sq_wk << " = " << kk[sq_bk][sq_wk][0] << " + " << kk[sq_bk][sq_wk][1] << "\n";

    for (i = 0; i < PIECE_NO_KING; ++i)
    {
      k0 = list_fb[i];
      k1 = list_fw[i];
      const auto* pkppb = ppkppb[k0];
      const auto* pkppw = ppkppw[k1];
      for (j = 0; j < i; ++j)
      {
        l0 = list_fb[j];
        l1 = list_fw[j];

#if defined(USE_SSE2)
        // SSEによる実装

        // pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
        __m128i tmp;
        tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
        tmp = _mm_cvtepi16_epi32(tmp);
        sum.m[0] = _mm_add_epi32(sum.m[0], tmp);

        cout << "BKPP : " << sq_bk << " " << k0 << " " << l0 << " = " << pkppb[l0][0] << " + " << pkppb[l0][1] << "\n";
        cout << "WKPP : " << sq_wk << " " << k1 << " " << l1 << " = " << pkppw[l1][0] << " + " << pkppw[l1][1] << "\n";

#else
        sum.p[0] += pkppb[l0];
        sum.p[1] += pkppw[l1];
#endif
      }
      sum.p[2] += kkp[sq_bk][sq_wk][k0];

      cout << "KKP : " << sq_bk << " " << sq_wk << " " << k0 << " = " << kkp[sq_bk][sq_wk][k0][0] << " + " << kkp[sq_bk][sq_wk][k0][1] << "\n";

    }

    cout << "Material = " << pos.state()->materialValue << endl;
    cout << sum;
    cout << "---\n";

  }

}

#endif // EVAL_KPP
