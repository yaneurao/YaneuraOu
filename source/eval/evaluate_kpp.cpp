#include "../shogi.h"

#ifdef EVAL_KPP

#include <fstream>
#include <iostream>

#include "../evaluate.h"
#include "../position.h"

using namespace std;

// eval cacheを用いるか。
//#define USE_EHASH

namespace Eval
{

// KKPファイル名
#define KKP_BIN "/kkp32ap.bin"

// KPPファイル名
#define KPP_BIN "/kpp16ap.bin"

  typedef int16_t ValueKpp;
  typedef int32_t ValueKkp;

  // KPP
  ValueKpp kpp[SQ_NB_PLUS1][fe_end][fe_end];

  // KKP
  // fe_endまでにしてしまう。これによりPiece番号がKPPとKKPとで共通になる。
  // さらに、2パラ目のKはInv(K2)を渡すものとすればkppと同じInv(K2)で済む。
  // [][][fe_end]のところはKK定数にしてあるものとする。
  ValueKkp kkp[SQ_NB_PLUS1][SQ_NB_PLUS1][fe_end + 1];

  void init() {}

  // 評価関数ファイルを読み込む
  void load_eval()
  {
    fstream fs;
    size_t size;

    fs.open((string)Options["EvalDir"] + KPP_BIN, ios::in | ios::binary);
    if (fs.fail())
      goto Error;
    size = SQ_NB_PLUS1 * (int)fe_end * (int)fe_end * (int)sizeof(ValueKpp);
    fs.read((char*)&kpp, size);
    if (fs.fail())
      goto Error;
    fs.close();
    size = SQ_NB_PLUS1 * (int)SQ_NB_PLUS1 * ((int)(fe_end)+1) * (int)sizeof(ValueKkp);
    fs.open((string)Options["EvalDir"] + KKP_BIN, ios::in | ios::binary);
    if (fs.fail())
      goto Error;
    fs.read((char*)&kkp, size);
    if (fs.fail())
      goto Error;
    fs.close();

    {
      // 手駒の添字、コンバートするときにひとつ間違えてた。(๑´ڡ`๑)
      //ValueKpp kpp2[SQ_NB_PLUS1][fe_end][fe_end];
      //ValueKkp kkp2[SQ_NB_PLUS1][SQ_NB_PLUS1][fe_end + 1];

      ValueKkp* kkp2 = new ValueKkp[SQ_NB_PLUS1*(int)SQ_NB_PLUS1*(int)(fe_end + 1)];
      ValueKpp* kpp2 = new ValueKpp[SQ_NB_PLUS1*(int)fe_end*(int)fe_end];
      #define KKP2(k1,k2,p) kkp2[k1 * (int)SQ_NB_PLUS1*(int)(fe_end + 1) + k2 * (int)(fe_end + 1) + p ]
      #define KPP2(k,p1,p2) kpp2[k * (int)fe_end*(int)fe_end + p1 * (int)fe_end + p2 ]
      memset(kkp2, 0, sizeof(kkp));
      memset(kpp2, 0, sizeof(kpp));

      for (int k1 = 0; k1 < SQ_NB_PLUS1; ++k1)
        for (int k2 = 0; k2 < SQ_NB_PLUS1; ++k2)
          for (int j = 1; j < fe_end + 1; ++j)
          {
            int j2 = j < fe_hand_end ? j - 1 : j;
            KKP2(k1, k2, j) = kkp[k1][k2][j2];
          }

      for (int k = 0; k < SQ_NB_PLUS1; ++k)
        for (int i = 1; i < fe_end; ++i)
          for (int j = 1; j < fe_end; ++j)
          {
            int i2 = i < fe_hand_end ? i - 1 : i;
            int j2 = j < fe_hand_end ? j - 1 : j;
            KPP2(k,i,j) = kpp[k][i2][j2];
          }

      memcpy(kkp, kkp2, sizeof(kkp));
      memcpy(kpp, kpp2, sizeof(kpp));

      delete[] kkp2;
      delete[] kpp2;
    }

    return;

  Error:;
    // 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
    cout << "\ninfo string Error! open evaluation file failed.\n";
    exit(EXIT_FAILURE);
  }

  // KKPのスケール
  const int FV_SCALE_KKP = 512;
  
  // KPP,KPのスケール
  const int FV_SCALE = 32;

#ifdef USE_EHASH
  // 評価値をcacheしておくための仕組み
  struct alignas(32) EvalHash
  {
    union {
      struct {
        HASH_KEY key;
        Value sumKKP;
        Value sumBKPP;
        Value sumWKPP;
        // 8 + 4 + 4 + 4 = 20bytes
        u8 padding[12];
      };
      ymm m;
    };

    EvalHash(){}

    // copyがatomicでないと困るのでAVX2の命令でコピーしてしまう。
    EvalHash& operator = (const EvalHash& rhs) { this->m = rhs.m; return *this; }
  };
  
  // あまり大きくしすぎるとCPU cacheの汚染がひどくなる。
  const u32 EHASH_SIZE = 1024 * 128;
  EvalHash ehash[EHASH_SIZE];
#endif

  // 駒割り以外の全計算
  // pos.st->BKPP,WKPP,KPPを初期化する。Position::set()で一度だけ呼び出される。(以降は差分計算)
  Value compute_eval(const Position& pos)
  {
    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto& pos_ = *const_cast<Position*>(&pos);
    auto list_fb = pos_.eval_list()->piece_list_fb();
    auto list_fw = pos_.eval_list()->piece_list_fw();

    int i, j;
    BonaPiece k0, k1;
    int32_t sumBKPP, sumWKPP, sumKKP;

    sumKKP = kkp[sq_bk0][sq_wk1][fe_end];
    sumBKPP = 0;
    sumWKPP = 0;

    for (i = 0; i < PIECE_NO_KING; i++)
    {
      k0 = list_fb[i];
      k1 = list_fw[i];
      sumKKP += kkp[sq_bk0][sq_wk1][k0];

      for (j = 0; j < i; j++)
      {
        sumBKPP += kpp[sq_bk0][k0][list_fb[j]];
        sumWKPP -= kpp[sq_wk1][k1][list_fw[j]];
      }
    }

    auto& info = *pos.state();
    info.sumKKP = Value(sumKKP);
    info.sumBKPP = Value(sumBKPP);
    info.sumWKPP = Value(sumWKPP);

#ifdef USE_EHASH

    // eval cacheに保存しておく。
    EvalHash e;
    e.sumKKP = Value(sumKKP);
    e.sumBKPP = Value(sumBKPP);
    e.sumWKPP = Value(sumWKPP);
    e.key = info.key();
    ehash[e.key & (EHASH_SIZE - 1)] = e;

#endif

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

	if (st->sumKKP != VALUE_NOT_EVALUATED)
    {
      sumKKP = st->sumKKP;
      sumBKPP = st->sumBKPP;
      sumWKPP = st->sumWKPP;
      goto CALC_DIFF_END;
    }

#ifdef USE_EHASH
    {
		HASH_KEY key = st->key();
		auto e = ehash[key & (EHASH_SIZE - 1)];
		if (e.key == key)
		{
			// hitしたのでこのまま返す
			st->sumKKP = sumKKP = e.sumKKP;
			st->sumBKPP = sumBKPP = e.sumBKPP;
			st->sumWKPP = sumWKPP = e.sumWKPP;
        
			goto CALC_DIFF_END;
		}
	}
#endif

	{
		// 遡るのは一つだけ
		// ひとつずつ遡りながらsumKPPがVALUE_NONEでないところまで探してそこからの差分を計算することは出来るが
		// レアケースだし、StateInfoにEvalListを持たせる必要が出てきて、あまり得しない。
		auto now = st;
		auto prev = st->previous;

		if (prev->sumKKP == VALUE_NOT_EVALUATED)
		{
#ifdef USE_EHASH
			HASH_KEY key2 = prev->key();
			auto e = ehash[key2 & (EHASH_SIZE - 1)];
			if (e.key == key2)
			{
				// hitしたのでここからの差分計算を行なう。
				prev->sumKKP = e.sumKKP;
				prev->sumBKPP = e.sumBKPP;
				prev->sumWKPP = e.sumWKPP;
				ASSERT_LV3(e.sumKKP != VALUE_NOT_EVALUATED);
			}
			else
#endif
			{
				// 全計算
				compute_eval(pos);
				sumKKP = now->sumKKP;
				sumBKPP = now->sumBKPP;
				sumWKPP = now->sumWKPP;
				goto CALC_DIFF_END;
			}
		}

		// この差分を求める
		{
			sumKKP = prev->sumKKP;
			sumBKPP = prev->sumBKPP;
			sumWKPP = prev->sumWKPP;
			int k0, k1, k2, k3;

			auto sq_bk0 = pos.king_square(BLACK);
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

					sumBKPP = 0;

					// このときKKPは差分で済まない。
					sumKKP = Eval::kkp[sq_bk0][sq_wk1][fe_end];

					// 片側まるごと計算
					for (i = 0; i < PIECE_NO_KING; i++)
					{
						k0 = now_list_fb[i];
						sumKKP += Eval::kkp[sq_bk0][sq_wk1][k0];

						for (j = 0; j < i; j++)
							sumBKPP += Eval::kpp[sq_bk0][k0][now_list_fb[j]];
					}

					// もうひとつの駒がないならこれで計算終わりなのだが。
					if (k == 2)
					{
						// この駒についての差分計算をしないといけない。
						k1 = dp.changed_piece[1].old_piece.fw;
						k3 = dp.changed_piece[1].new_piece.fw;

						dirty = dp.pieceNo[1];
						// BKPPはすでに計算済みなのでWKPPのみ。
						// WKは移動していないのでこれは前のままでいい。
						for (i = 0; i < dirty; ++i)
						{
							sumWKPP += Eval::kpp[sq_wk1][k1][now_list_fw[i]];
							sumWKPP -= Eval::kpp[sq_wk1][k3][now_list_fw[i]];
						}
						for (++i; i < PIECE_NO_KING; ++i)
						{
							sumWKPP += Eval::kpp[sq_wk1][k1][now_list_fw[i]];
							sumWKPP -= Eval::kpp[sq_wk1][k3][now_list_fw[i]];
						}
					}

				}
				else {
					// ----------------------------
					// 後手玉が移動したときの計算
					// ----------------------------
					ASSERT_LV3(dirty == PIECE_NO_WKING);

					sumWKPP = 0;
					sumKKP = Eval::kkp[sq_bk0][sq_wk1][fe_end];

					for (i = 0; i < PIECE_NO_KING; i++)
					{
						k0 = now_list_fb[i]; // これ、KKPテーブルにk1側も入れておいて欲しい気はするが..
						k1 = now_list_fw[i];
						sumKKP += Eval::kkp[sq_bk0][sq_wk1][k0];

						for (j = 0; j < i; j++)
							sumWKPP -= Eval::kpp[sq_wk1][k1][now_list_fw[j]];
					}

					if (k == 2)
					{
						k0 = dp.changed_piece[1].old_piece.fb;
						k2 = dp.changed_piece[1].new_piece.fb;

						dirty = dp.pieceNo[1];
						for (i = 0; i < dirty; ++i)
						{
							sumBKPP -= Eval::kpp[sq_bk0][k0][now_list_fb[i]];
							sumBKPP += Eval::kpp[sq_bk0][k2][now_list_fb[i]];
						}
						for (++i; i < PIECE_NO_KING; ++i)
						{
							sumBKPP -= Eval::kpp[sq_bk0][k0][now_list_fb[i]];
							sumBKPP += Eval::kpp[sq_bk0][k2][now_list_fb[i]];
						}
					}
				}

			}
			else {
				// ----------------------------
				// 玉以外が移動したときの計算
				// ----------------------------

#define ADD_BWKPP(W0,W1,W2,W3) { \
          sumBKPP -= Eval::kpp[sq_bk0][W0][now_list_fb[i]]; \
          sumWKPP += Eval::kpp[sq_wk1][W1][now_list_fw[i]]; \
          sumBKPP += Eval::kpp[sq_bk0][W2][now_list_fb[i]]; \
          sumWKPP -= Eval::kpp[sq_wk1][W3][now_list_fw[i]]; \
}

				if (k == 1)
				{
					// 移動した駒が一つ。

					k0 = dp.changed_piece[0].old_piece.fb;
					k1 = dp.changed_piece[0].old_piece.fw;
					k2 = dp.changed_piece[0].new_piece.fb;
					k3 = dp.changed_piece[0].new_piece.fw;

					// KKP差分
					sumKKP -= Eval::kkp[sq_bk0][sq_wk1][k0];
					sumKKP += Eval::kkp[sq_bk0][sq_wk1][k2];

					// KP値、要らんのでi==dirtyを除く
					for (i = 0; i < dirty; ++i)
						ADD_BWKPP(k0, k1, k2, k3);
					for (++i; i < PIECE_NO_KING; ++i)
						ADD_BWKPP(k0, k1, k2, k3);

				}
				else if (k == 2) {

					// 移動する駒が王以外の2つ。
					PieceNo dirty2 = dp.pieceNo[1];
					if (dirty > dirty2) swap(dirty, dirty2);
					// PIECE_NO_ZERO <= dirty < dirty2 < PIECE_NO_KING
					// にしておく。

					k0 = dp.changed_piece[0].old_piece.fb;
					k1 = dp.changed_piece[0].old_piece.fw;
					k2 = dp.changed_piece[0].new_piece.fb;
					k3 = dp.changed_piece[0].new_piece.fw;

					int m0, m1, m2, m3;
					m0 = dp.changed_piece[1].old_piece.fb;
					m1 = dp.changed_piece[1].old_piece.fw;
					m2 = dp.changed_piece[1].new_piece.fb;
					m3 = dp.changed_piece[1].new_piece.fw;

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

#ifdef USE_EHASH
		// せっかく計算したのでehashに保存しておく。
		{
			EvalHash e;
			e.sumKKP = Value(sumKKP);
			e.sumBKPP = Value(sumBKPP);
			e.sumWKPP = Value(sumWKPP);
			e.key = key;
			ehash[key & (EHASH_SIZE - 1)] = e;
		}
#endif
	}

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

  void evaluate_with_no_return(const Position& pos)
  {
    if (pos.state()->sumKKP == VALUE_NOT_EVALUATED)
      evaluate(pos);
  }

  // 現在の局面の評価値の内訳を表示する。
  void print_eval_stat(Position& pos)
  {
    cout << "--- EVAL STAT\n";

    Square sq_bk0 = pos.king_square(BLACK);
    Square sq_wk1 = Inv(pos.king_square(WHITE));

    auto list_fb = pos.eval_list()->piece_list_fb();
    auto list_fw = pos.eval_list()->piece_list_fw();

    int i, j;
    BonaPiece k0, k1;

    // 38枚の駒を表示
    for (i = 0; i < PIECE_NO_KING; ++i)
      cout << int(list_fb[i]) << " = " << list_fb[i] << " , " << int(list_fw[i]) << " =  " << list_fw[i] << endl;

    int32_t sumBKPP, sumWKPP, sumKKP;

    cout << "KKC : " << sq_bk0 << " " << Inv(sq_wk1) << " = " << kkp[sq_bk0][sq_wk1][fe_end] << "\n";

    sumBKPP = sumWKPP = 0;
    sumKKP = kkp[sq_bk0][sq_wk1][fe_end];

    for (i = 0; i < PIECE_NO_KING; i++)
    {
      k0 = list_fb[i];
      k1 = list_fw[i];

      cout << "KKP : " << sq_bk0 << " " << Inv(sq_wk1) << " " << k0 << " = " << kkp[sq_bk0][sq_wk1][k0] << "\n";
      sumKKP += kkp[sq_bk0][sq_wk1][k0];

      for (j = 0; j <= i; j++)
      {
        cout << "BKPP : " << sq_bk0 << " " << k0 << " " << list_fb[j] << " = " << kpp[sq_bk0][k0][list_fb[j]] << "\n";
        cout << "WKPP : " << sq_wk1 << " " << k1 << " " << list_fw[j] << " = " << kpp[sq_wk1][k1][list_fw[j]] << "\n";

        sumBKPP += kpp[sq_bk0][k0][list_fb[j]];
        sumWKPP += kpp[sq_wk1][k1][list_fw[j]];

        //        cout << "sumWKPP = " << sumWKPP << " sumBKPP " << sumBKPP << " sumWKPP " << sumWKPP << endl;

        // i==jにおいて0以外やったらあかんで!!
        ASSERT(!(i == j && kpp[sq_bk0][k0][list_fb[j]] != 0));
      }
    }

    cout << "Material = " << pos.state()->materialValue << endl;
    cout << "sumKKP = " << sumKKP << " sumBKPP " << sumBKPP << " sumWKPP " << sumWKPP << endl;
    cout << "---\n";
  }

}

#endif // EVAL_KPP
