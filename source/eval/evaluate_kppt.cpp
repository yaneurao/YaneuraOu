#include "../shogi.h"

// Apery WCSC26の評価関数バイナリを読み込むための仕組み。
//
// このコードを書くに当たって、Apery、Silent Majorityのコードを非常に参考にさせていただきました。
// Special thanks to Takuya Hiraoka and Jangia , I am very impressed by their devouring enthusiasm.
//

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
  // なので、この関数の最適化は頑張らない。
  Value compute_eval(const Position& pos)
  {
    Square sq_bk = pos.king_square(BLACK);
    Square sq_wk = pos.king_square(WHITE);
    const auto* ppkppb = kpp[    sq_bk ];
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

#if defined(USE_SSE41)
        // SSEによる実装

        // pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
        __m128i tmp;
        tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
        // この命令SSE4.1の命令のはず..
        tmp = _mm_cvtepi16_epi32(tmp);
        sum.m[0] = _mm_add_epi32(sum.m[0], tmp);
#else
        sum.p[0] += pkppb[l0];
        sum.p[1] += pkppw[l1];
#endif
      }
      sum.p[2] += kkp[sq_bk][sq_wk][k0];
    }

    auto st = pos.state();
    sum.p[2][0] += st->materialValue * FV_SCALE;

    st->sum = sum;

    return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
  }

  // 先手玉が移動したときに先手側の差分
  std::array<s32, 2> do_a_black(const Position& pos, const ExtBonaPiece ebp) {
    const Square sq_bk = pos.king_square(BLACK);
    const auto* list0 = pos.eval_list()->piece_list_fb();

    const auto* pkppb = kpp[sq_bk][ebp.fb];
    std::array<s32, 2> sum = { { pkppb[list0[0]][0], pkppb[list0[0]][1] } };
    for (int i = 1; i < PIECE_NO_KING ; ++i) {
      sum[0] += pkppb[list0[i]][0];
      sum[1] += pkppb[list0[i]][1];
    }
    return sum;
  }

  // 後手玉が移動したときの後手側の差分
  std::array<s32, 2> do_a_white(const Position& pos, const ExtBonaPiece ebp) {
    const Square sq_wk = pos.king_square(WHITE);
    const auto* list1 = pos.eval_list()->piece_list_fw();

    const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];
    std::array<s32, 2> sum = { { pkppw[list1[0]][0], pkppw[list1[0]][1] } };
    for (int i = 1; i < PIECE_NO_KING ; ++i) {
      sum[0] += pkppw[list1[i]][0];
      sum[1] += pkppw[list1[i]][1];
    }
    return sum;
  }

  // 玉以外の駒が移動したときの差分
  EvalSum do_a_pc(const Position& pos, const ExtBonaPiece ebp) {
    const Square sq_bk = pos.king_square(BLACK);
    const Square sq_wk = pos.king_square(WHITE);
    const auto list0 = pos.eval_list()->piece_list_fb();
    const auto list1 = pos.eval_list()->piece_list_fw();

    EvalSum sum;
    sum.p[2][0] = kkp[sq_bk][sq_wk][ebp.fb][0];
    sum.p[2][1] = kkp[sq_bk][sq_wk][ebp.fb][1];

    const auto* pkppb = kpp[    sq_bk ][ebp.fb];
    const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];
#if defined (USE_SSE41)
    sum.m[0] = _mm_set_epi32(0, 0, *reinterpret_cast<const s32*>(&pkppw[list1[0]][0]), *reinterpret_cast<const s32*>(&pkppb[list0[0]][0]));
    sum.m[0] = _mm_cvtepi16_epi32(sum.m[0]);
    for (int i = 1; i < PIECE_NO_KING; ++i) {
      __m128i tmp;
      tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const s32*>(&pkppw[list1[i]][0]), *reinterpret_cast<const s32*>(&pkppb[list0[i]][0]));
      tmp = _mm_cvtepi16_epi32(tmp);
      sum.m[0] = _mm_add_epi32(sum.m[0], tmp);
    }
#else
    sum.p[0][0] = pkppb[list0[0]][0];
    sum.p[0][1] = pkppb[list0[0]][1];
    sum.p[1][0] = pkppw[list1[0]][0];
    sum.p[1][1] = pkppw[list1[0]][1];
    for (int i = 1; i < PIECE_NO_KING ; ++i) {
      sum.p[0] += pkppb[list0[i]];
      sum.p[1] += pkppw[list1[i]];
    }
#endif

    return sum;
  }


#ifdef USE_EVAL_HASH
  EvaluateHashTable g_evalTable;
#endif

  void evaluateBody(const Position& pos)
  {
    // 過去に遡って差分を計算していく。

    auto now = pos.state();
    auto prev = now->previous;

    // nodeごとにevaluate()は呼び出しているので絶対に差分計算できるはず。
    // 一つ前のnodeでevaluate()されているはず。
    if (!prev->sum.evaluated())
    {
      // 全計算
      compute_eval(pos);
      return;
      // 結果は、pos->state().sumから取り出すべし。
    }

    // 遡るnodeは一つだけ
    // ひとつずつ遡りながらsumKPPがVALUE_NONEでないところまで探してそこからの差分を計算することは出来るが
    // 現状、探索部では毎node、evaluate()を呼び出すから問題ない。

    auto& dp = now->dirtyPiece;

    // 移動させた駒は最大2つある。その数
    int moved_piece_num = dp.dirty_num;

    auto list0 = pos.eval_list()->piece_list_fb();
    auto list1 = pos.eval_list()->piece_list_fw();

    auto dirty = dp.pieceNo[0];

    // 移動させた駒は王か？
    if (dirty >= PIECE_NO_KING)
    {
      // 前のnodeの評価値からの増分を計算していく。
      // (直接この変数に加算していく)
      // この意味においてdiffという名前は少々不適切ではあるが。
      EvalSum diff = prev->sum;

      auto sq_bk = pos.king_square(BLACK);
      auto sq_wk = pos.king_square(WHITE);

      diff.p[2] = kk[sq_bk][sq_wk];
      diff.p[2][0] += now->materialValue * FV_SCALE;

      // 後手玉の移動(片側分のKPPを丸ごと求める)
      if (dirty == PIECE_NO_WKING)
      {
        const auto ppkppw = kpp[Inv(sq_wk)];

        // ΣWKPP = 0
        diff.p[1][0] = 0;
        diff.p[1][1] = 0;

        for (int i = 0; i < PIECE_NO_KING; ++i)
        {
          const int k1 = list1[i];
          const auto* pkppw = ppkppw[k1];
          for (int j = 0; j < i; ++j)
          {
            const int l1 = list1[j];
            diff.p[1] += pkppw[l1];
          }

          // KKPのWK分。BKは移動していないから、BK側には影響ない。
          diff.p[2][0] -= kkp[Inv(sq_wk)][Inv(sq_bk)][k1][0];
          diff.p[2][1] += kkp[Inv(sq_wk)][Inv(sq_bk)][k1][1];
        }

        // 動かした駒が２つ
        if (moved_piece_num == 2)
        {
          // 瞬間的にeval_listの移動させた駒の番号を変更してしまう。
          // こうすることで前nodeのpiece_listを持たなくて済む。

          const int listIndex_cap = dp.pieceNo[1];
          diff.p[0] += do_a_black(pos, dp.pieceNow[1]);
          list0[listIndex_cap] = dp.piecePrevious[1].fb;
          diff.p[0] -= do_a_black(pos, dp.piecePrevious[1]);
          list0[listIndex_cap] = dp.pieceNow[1].fb;
        }

      } else {

        // 先手玉の移動
        // さきほどの処理と同様。

        const auto* ppkppb = kpp[sq_bk];
        diff.p[0][0] = 0;
        diff.p[0][1] = 0;

        for (int i = 0; i < PIECE_NO_KING; ++i)
        {
          const int k0 = list0[i];
          const auto* pkppb = ppkppb[k0];
          for (int j = 0; j < i; ++j) {
            const int l0 = list0[j];
            diff.p[0] += pkppb[l0];
          }
          diff.p[2] += kkp[sq_bk][sq_wk][k0];
        }

        if (moved_piece_num == 2) {
          const int listIndex_cap = dp.pieceNo[1];
          diff.p[1] += do_a_white(pos, dp.pieceNow[1]);
          list1[listIndex_cap] = dp.piecePrevious[1].fw;
          diff.p[1] -= do_a_white(pos, dp.piecePrevious[1]);
          list1[listIndex_cap] = dp.pieceNow[1].fw;
        }
      }

      // sumの計算が終わったのでpos.state()->sumに反映させておく。(これがこの関数の返し値に相当する。)
      now->sum = diff;

    } else {

      // 王以外の駒が移動したケース
      // 今回の差分を計算して、そこに加算する。

      const int listIndex = dp.pieceNo[0];

      auto diff = do_a_pc(pos, dp.pieceNow[0]);
      if (moved_piece_num == 1) {

        // 動いた駒が1つ。
        list0[listIndex] = dp.piecePrevious[0].fb;
        list1[listIndex] = dp.piecePrevious[0].fw;
        diff -= do_a_pc(pos, dp.piecePrevious[0]);

      } else {

        // 動いた駒が2つ。

        auto sq_bk = pos.king_square(BLACK);
        auto sq_wk = pos.king_square(WHITE);

        diff += do_a_pc(pos, dp.pieceNow[1]);
        diff.p[0] -= kpp[sq_bk][dp.pieceNow[0].fb][dp.pieceNow[1].fb];
        diff.p[1] -= kpp[Inv(sq_wk)][dp.pieceNow[0].fw][dp.pieceNow[1].fw];

        const PieceNo listIndex_cap = dp.pieceNo[1];
        list0[listIndex_cap] = dp.piecePrevious[1].fb;
        list1[listIndex_cap] = dp.piecePrevious[1].fw;

        list0[listIndex] = dp.piecePrevious[0].fb;
        list1[listIndex] = dp.piecePrevious[0].fw;
        diff -= do_a_pc(pos, dp.piecePrevious[0]);
        diff -= do_a_pc(pos, dp.piecePrevious[1]);

        diff.p[0] += kpp[sq_bk][dp.piecePrevious[0].fb][dp.piecePrevious[1].fb];
        diff.p[1] += kpp[Inv(sq_wk)][dp.piecePrevious[0].fw][dp.piecePrevious[1].fw];
        list0[listIndex_cap] = dp.pieceNow[1].fb;
        list1[listIndex_cap] = dp.pieceNow[1].fw;
      }

      list0[listIndex] = dp.pieceNow[0].fb;
      list1[listIndex] = dp.pieceNow[0].fw;

      // 前nodeからの駒割りの増分を加算。
      diff.p[2][0] += (now->materialValue - prev->materialValue) * FV_SCALE;

      now->sum = diff + prev->sum;
    }

  }

  // 評価関数
  Value evaluate(const Position& pos)
  {
    auto st = pos.state();
    auto &sum = st->sum;

    // すでに計算済(Null Moveなどで)であるなら、それを返す。
    if (sum.evaluated())
      return Value(sum.sum(pos.side_to_move()) / FV_SCALE);

#ifdef USE_EVAL_HASH
    // evaluate hash tableにはあるかも。

    const Key keyExcludeTurn = st->key() & ~1; // 手番を消した局面hash key
    EvalSum entry = *g_evalTable[keyExcludeTurn];       // atomic にデータを取得する必要がある。
    entry.decode();
    if (entry.key == keyExcludeTurn)
    {
      // あった！
      st->sum = entry;
      return Value(entry.sum(pos.side_to_move()) / FV_SCALE);
    }
#endif

    // 評価関数本体を呼び出して求める。
    evaluateBody(pos);

#ifdef USE_EVAL_HASH
    // せっかく計算したのでevaluate hash tableに保存しておく。
    st->sum.key = keyExcludeTurn;
    st->sum.encode();
    *g_evalTable[keyExcludeTurn] = st->sum;
#endif

    ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
    // 差分計算と非差分計算との計算結果が合致するかのテスト。(さすがに重いのでコメントアウトしておく)
    // ASSERT_LV5(Value(st->sum.sum(pos.side_to_move()) / FV_SCALE) == compute_eval(pos));

#if 0
    if (!(Value(st->sum.sum(pos.side_to_move()) / FV_SCALE) == compute_eval(pos)))
    {
      st->sum.p[0][0] = VALUE_NOT_EVALUATED;
      evaluateBody(pos);
    }
#endif

    return Value(st->sum.sum(pos.side_to_move()) / FV_SCALE);
  }

  void evaluate_with_no_return(const Position& pos)
  {
    // まだ評価値が計算されていないなら
    if (!pos.state()->sum.evaluated())
      evaluate(pos);
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

#if defined(USE_SSE41)
        // SSEによる実装

        // pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
        __m128i tmp;
        tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
        // この命令SSE4.1の命令のはず
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

#endif // EVAL_KPPT
