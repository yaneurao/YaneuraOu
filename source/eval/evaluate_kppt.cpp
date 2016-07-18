#include "../shogi.h"

// Apery WCSC26の評価関数バイナリを読み込むための仕組み。
//
// このコードを書くに当たって、Apery、Silent Majorityのコードを非常に参考にさせていただきました。
// Special thanks to Takuya Hiraoka and Jangia , I was very impressed by their devouring enthusiasm.
//

#ifdef EVAL_KPPT

#include <fstream>
#include <iostream>
#include <unordered_set>

#include "../evaluate.h"
#include "../position.h"
#include "../misc.h"

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

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_MSC_VER)

  // 評価関数パラメーターを他プロセスと共有するための機能。

  // KK
	ValueKk(*kk_)[SQ_NB][SQ_NB];

	// KPP
	ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

	// KKP
	ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk (*kk_)
#define kpp (*kpp_)
#define kkp (*kkp_)

// memory mapped fileを介して共有するデータ
	struct SharedEval
	{
		ValueKk kk_[SQ_NB][SQ_NB];
		ValueKpp kpp_[SQ_NB][fe_end][fe_end];
		ValueKkp kkp_[SQ_NB][SQ_NB][fe_end];

		// 参照カウント(プロセス間で共有している数)
		int shared_count;
	};

#else

  // 通常の評価関数テーブル

  // KK
	ALIGNED(32) ValueKk kk[SQ_NB][SQ_NB];

	// KPP
	ALIGNED(32) ValueKpp kpp[SQ_NB][fe_end][fe_end];

	// KKP
	ALIGNED(32) ValueKkp kkp[SQ_NB][SQ_NB][fe_end];

#endif

// 評価関数の学習関連。
#ifdef EVAL_LEARN

	// あるBonaPieceを相手側から見たときの値
	BonaPiece inv_piece[fe_end];

	// 盤面上のあるBonaPieceをミラーした位置にあるものを返す。
	BonaPiece mir_piece[fe_end];

	// 学習時にkppテーブルに値を書き出すためのヘルパ関数。
	// この関数を用いると、ミラー関係にある箇所などにも同じ値を書き込んでくれる。次元下げの一種。
	void kpp_write(Square k1, BonaPiece p1, BonaPiece p2, ValueKpp value)
	{
		// '~'をinv記号,'M'をmirror記号だとして
		//   [  k1 ][  p1 ][  p2 ]
		//   [  k1 ][  p2 ][  p1 ]
		//   [M(k1)][M(p1)][M(p2)]
		//   [M(k1)][M(p2)][M(p1)]
		// は、同じ値であるべきなので、その4箇所に書き出す。

		BonaPiece mp1 = mir_piece[p1];
		BonaPiece mp2 = mir_piece[p2];
		Square  mk1 = Mir(k1);

		// Apery(WCSC26)の評価関数、玉が5筋にいるきにmirrorした値が一致しないので
		// assertから除外しておく。

		ASSERT_LV3(kpp[k1][p1][p2] == kpp[k1][p2][p1]);
		ASSERT_LV3(kpp[k1][p1][p2] == kpp[mk1][mp1][mp2] || file_of(k1) == FILE_5);
		ASSERT_LV3(kpp[k1][p1][p2] == kpp[mk1][mp2][mp1] || file_of(k1) == FILE_5);

		kpp[k1][p1][p2]
			= kpp[k1][p2][p1]
			= kpp[mk1][mp1][mp2]
			= kpp[mk1][mp2][mp1]
			= value;
	}


	// 学習時にkkpテーブルに値を書き出すためのヘルパ関数。
	// この関数を用いると、ミラー関係にある箇所などにも同じ値を書き込んでくれる。次元下げの一種。
	void kkp_write(Square k1, Square k2, BonaPiece p1, ValueKkp value)
	{
		// '~'をinv記号,'M'をmirror記号だとして
		//   [  k1 ][  k2 ][  p1 ]
		//   [ ~k2 ][ ~k1 ][ ~p1 ] (1)
		//   [M k1 ][M k2 ][M p1 ]
		//   [M~k2 ][M~k1 ][M~p1 ] (2)
		// は、同じ値であるべきなので、その4箇所に書き出す。
		// ただし、(1)[0],(2)[0]は[k1][k2][p1][0]とは符号が逆なので注意。

		BonaPiece ip1 = inv_piece[p1];
		BonaPiece mp1 = mir_piece[p1];
		BonaPiece mip1 = mir_piece[ip1];
		Square  mk1 = Mir(k1);
		Square  ik1 = Inv(k1);
		Square mik1 = Mir(ik1);
		Square  mk2 = Mir(k2);
		Square  ik2 = Inv(k2);
		Square mik2 = Mir(ik2);

		ASSERT_LV3(kkp[k1][k2][p1][0] == -kkp[ik2][ik1][ip1][0]);
		ASSERT_LV3(kkp[k1][k2][p1][1] == +kkp[ik2][ik1][ip1][1]);
		ASSERT_LV3(kkp[k1][k2][p1] == kkp[mk1][mk2][mp1]);
		ASSERT_LV3(kkp[ik2][ik1][ip1] == kkp[mik2][mik1][mip1]);

		kkp[k1][k2][p1]
			= kkp[mk1][mk2][mp1]
			= value;

		kkp[ik2][ik1][ip1][0]
			= kkp[mik2][mik1][mip1][0]
			= -value[0];

		kkp[ik2][ik1][ip1][1]
			= kkp[mik2][mik1][mip1][1]
			= +value[1];
	}

	typedef std::array<float, 2> ValueKkFloat;
	typedef std::array<float, 2> ValueKppFloat;
	typedef std::array<float, 2> ValueKkpFloat;

	// KK
	ValueKkFloat(*kk_grad_)[SQ_NB][SQ_NB];   // 学習時の目的関数の勾配
	ValueKkFloat(*kk_grad2_)[SQ_NB][SQ_NB];  // AdaGrad用のg2

	// KPP
	ValueKppFloat(*kpp_grad_)[SQ_NB][fe_end][fe_end];
	ValueKppFloat(*kpp_grad2_)[SQ_NB][fe_end][fe_end];

	// KKP
	ValueKkpFloat(*kkp_grad_)[SQ_NB][SQ_NB][fe_end];
	ValueKkpFloat(*kkp_grad2_)[SQ_NB][SQ_NB][fe_end];

	#define kk_grad (*kk_grad_)
	#define kk_grad2 (*kk_grad2_)
	#define kpp_grad (*kpp_grad_)
	#define kpp_grad2 (*kpp_grad2_)
	#define kkp_grad (*kkp_grad_)
	#define kkp_grad2 (*kkp_grad2_)

	// 学習のときの勾配配列の初期化
	void init_grad()
	{
		if (kk_grad == nullptr)
		{
			u64 size;

			size = u64(SQ_NB)*u64(SQ_NB);
			kk_grad_ = (ValueKkFloat(*)[SQ_NB][SQ_NB])new ValueKkFloat[size];
			kk_grad2_ = (ValueKkFloat(*)[SQ_NB][SQ_NB])new ValueKkFloat[size];
			memset(kk_grad_, 0, sizeof(ValueKkFloat) * size);
			memset(kk_grad2_, 0, sizeof(ValueKkFloat) * size);

			size = u64(SQ_NB)*u64(fe_end)*u64(fe_end);
			kpp_grad_ = (ValueKppFloat(*)[SQ_NB][fe_end][fe_end])new ValueKppFloat[size];
			kpp_grad2_ = (ValueKppFloat(*)[SQ_NB][fe_end][fe_end])new ValueKppFloat[size];
			memset(kpp_grad_, 0, sizeof(ValueKppFloat) * size);
			memset(kpp_grad2_, 0, sizeof(ValueKppFloat) * size);

			size = u64(SQ_NB)*u64(SQ_NB)*u64(fe_end);
			kkp_grad_ = (ValueKkpFloat(*)[SQ_NB][SQ_NB][fe_end])new ValueKkpFloat[size];
			kkp_grad2_ = (ValueKkpFloat(*)[SQ_NB][SQ_NB][fe_end])new ValueKkpFloat[size];
			memset(kkp_grad_, 0, sizeof(ValueKkpFloat) * size);
			memset(kkp_grad2_, 0, sizeof(ValueKkpFloat) * size);
		}
	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	void add_grad(Position& pos, Color rootColor, double delta_grad)
	{
		// 勾配配列を確保するメモリがもったいないのでとりあえずfloatでいいや。

		// 手番を考慮しない値
		auto f = (rootColor == BLACK) ? float(delta_grad) : -float(delta_grad);

		// 手番を考慮する値
		auto g = (rootColor == pos.side_to_move()) ? float(delta_grad) : -float(delta_grad);

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

		int i, j;
		BonaPiece k0, k1, l0, l1;

		// KK
		kk_grad[sq_bk][sq_wk] += ValueKkFloat{ f , g };

		for (i = 0; i < PIECE_NO_KING; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

				kpp_grad[sq_bk][k0][l0] += ValueKppFloat{ f ,  g };
				kpp_grad[Inv(sq_wk)][k1][l1] += ValueKppFloat{ -f ,  g };
			}
			kkp_grad[sq_bk][sq_wk][k0] += ValueKkpFloat{ f , g };
		}

	}

// update_weights()の更新式を以下のなかから一つ選択すべし。

// AdaGradによるupdate
//#define USE_ADA_GRAD_UPADTE
// SGDによるupdate
#define USE_SGD_UPDATE

	// 現在の勾配をもとにSGDかAdaGradか何かする。m = 教師データの件数。
	void update_weights(u64 m)
	{

#ifdef USE_ADA_GRAD_UPADTE
		// 学習率η = 0.01。勾配が一定な場合、1万回でη×199ぐらい。
		// cf. [AdaGradのすすめ](http://qiita.com/ak11/items/7f63a1198c345a138150)
		const float eta = 0.01f * 32 / 600; // 32 == Eval::FV_SCALE , 1/ 600 == dsigmoidのときに割ってなかった係数。
#endif

#ifdef USE_SGD_UPDATE
		const float eta = 0.01f * 32 / 600; // 32 == Eval::FV_SCALE , 1/ 600 == dsigmoidのときに割ってなかった係数。
#endif

		// g2[i] += g * g;
		// w[i] -= eta * g / sqrt(g2[i]);
		// g = 0 // mini-batchならこのタイミングで勾配配列クリア。

		ASSERT_LV3(m != 0);

		for (auto k1 : SQ)
			for (auto k2 : SQ)
			{
				auto g = kk_grad[k1][k2];

#ifdef USE_ADA_GRAD_UPADTE
				if (g[0] == 0 && g[1] == 0)
					continue;

				auto gg = ValueKkFloat{ g[0] * g[0] / m, g[1] * g[1] / m };
				kk_grad2[k1][k2] += gg;

				auto g2 = kk_grad2[k1][k2];
				if (abs(g2[0]) < 0.1f|| abs(g2[1]) < 0.1f)
					continue;

				kk_grad[k1][k2] = ValueKkFloat{ 0,0 };

				// kk[k1][k2] -= eta * g / sqrt(g2);

				auto e = ValueKk{ (int32_t)(-eta * g[0] / sqrt(g2[0])) , (int32_t)(-eta * g[1] / sqrt(g2[1])) };
				kk[k1][k2] += e;
#endif

#ifdef USE_SGD_UPDATE
				auto e = ValueKk{ (int32_t)(-eta * g[0] / m) , (int32_t)(-eta * g[1] / m) };
				kk[k1][k2] += e;
#endif

				ASSERT_LV3(abs(kk[k1][k2][0]) < INT16_MAX * 4);
				ASSERT_LV3(abs(kk[k1][k2][1]) < INT16_MAX * 4);
			}

		for (auto k : SQ)
			for (auto p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (auto p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
				{
					auto g = kpp_grad[k][p1][p2];

#ifdef USE_ADA_GRAD_UPADTE
					if (g[0] == 0 && g[1] == 0)
						continue;

					auto gg = ValueKppFloat{ g[0] * g[0] / m, g[1] * g[1] / m };
					kpp_grad2[k][p1][p2] += gg;

					auto g2 = kpp_grad2[k][p1][p2];
					if (abs(g2[0]) < 0.1f || abs(g2[1]) < 0.1f)
						continue;

					kpp_grad[k][p1][p2] = ValueKppFloat{ 0,0 };

					auto v = ValueKpp{ int16_t(kpp[k][p1][p2][0] - eta * g[0] / sqrt(g2[0])) , int16_t(kpp[k][p1][p2][1] - eta * g[1] / sqrt(g2[1])) };
					kpp_write(k, p1, p2, v);
#endif

#ifdef USE_SGD_UPDATE
					auto v = ValueKpp{ int16_t(kpp[k][p1][p2][0] - eta * g[0] / m) , int16_t(kpp[k][p1][p2][1] - eta * g[1] / m) };
					kpp_write(k, p1, p2, v);
#endif

					ASSERT_LV3(abs(kpp[k][p1][p2][0]) < INT16_MAX / 2);
					ASSERT_LV3(abs(kpp[k][p1][p2][1]) < INT16_MAX / 2);
				}

		for (auto k1 : SQ)
			for (auto k2 : SQ)
				for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					auto g = kkp_grad[k1][k2][p];

#ifdef USE_ADA_GRAD_UPADTE
					if (g[0] == 0 && g[1] == 0)
						continue;

					auto gg = ValueKkFloat{ g[0] * g[0] / m, g[1] * g[1] / m };
					kkp_grad2[k1][k2][p] += gg;

					auto g2 = kkp_grad2[k1][k2][p];
					if (abs(g2[0]) < 0.1f || abs(g2[1]) < 0.1f)
						continue;

					kkp_grad[k1][k2][p] = ValueKkFloat{ 0,0 };

					// kk[k1][k2] -= eta * g / sqrt(g2);

					auto v = ValueKkp{ int32_t(kkp[k1][k2][p][0] - eta * g[0] / sqrt(g2[0])) , int32_t(kkp[k1][k2][p][1] - eta * g[1] / sqrt(g2[1])) };
					kkp_write(k1, k2, p, v);
#endif

#ifdef USE_SGD_UPDATE
					auto v = ValueKkp{ int32_t(kkp[k1][k2][p][0] - eta * g[0] / m) , int32_t(kkp[k1][k2][p][1] - eta * g[1] / m) };
					kkp_write(k1, k2, p, v);
#endif

					ASSERT_LV3(abs(kkp[k1][k2][p][0]) < INT16_MAX / 4);
					ASSERT_LV3(abs(kkp[k1][k2][p][1]) < INT16_MAX / 4);
				}
	}


	// 学習のためのテーブルの初期化
	void eval_learn_init()
	{
		// fとeとの交換
		int t[] = {
		  f_hand_pawn - 1 , e_hand_pawn - 1 ,
		  f_hand_lance - 1 , e_hand_lance - 1 ,
		  f_hand_knight - 1 , e_hand_knight - 1 ,
		  f_hand_silver - 1 , e_hand_silver - 1 ,
		  f_hand_gold - 1 , e_hand_gold - 1 ,
		  f_hand_bishop - 1 , e_hand_bishop - 1 ,
		  f_hand_rook - 1 , e_hand_rook - 1 ,
		  f_pawn             , e_pawn            ,
		  f_lance            , e_lance           ,
		  f_knight           , e_knight          ,
		  f_silver           , e_silver          ,
		  f_gold             , e_gold            ,
		  f_bishop           , e_bishop          ,
		  f_horse            , e_horse           ,
		  f_rook             , e_rook            ,
		  f_dragon           , e_dragon          ,
		};

		// 未初期化の値を突っ込んでおく。
		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
		{
			inv_piece[p] = (BonaPiece)-1;

			// mirrorは手駒に対しては機能しない。元の値を返すだけ。
			mir_piece[p] = (p < f_pawn) ? p : (BonaPiece)-1;
		}

		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
		{
			for (int i = 0; i < 32 /* t.size() */; i += 2)
			{
				if (t[i] <= p && p < t[i + 1])
				{
					Square sq = (Square)(p - t[i]);

					// 見つかった!!
					BonaPiece q = (p < fe_hand_end) ? BonaPiece(sq + t[i + 1]) : (BonaPiece)(Inv(sq) + t[i + 1]);
					inv_piece[p] = q;
					inv_piece[q] = p;

					/*
					ちょっとトリッキーだが、pに関して盤上の駒は
						p >= fe_hand_end
					のとき。

					このpに対して、nを整数として(上のコードのiは偶数しかとらない)、
						a)  t[2n + 0] <= p < t[2n + 1] のときは先手の駒
						b)  t[2n + 1] <= p < t[2n + 2] のときは後手の駒
				  　である。

					ゆえに、a)の範囲にあるpをq = Inv(p-t[2n+0]) + t[2n+1] とすると180度回転させた升にある後手の駒となる。
					そこでpとqをswapさせてinv_piece[ ]を初期化してある。
					*/

					// 手駒に関してはmirrorなど存在しない。
					if (p < fe_hand_end)
						continue;

					BonaPiece r1 = (BonaPiece)(Mir(sq) + t[i]);
					mir_piece[p] = r1;
					mir_piece[r1] = p;

					BonaPiece p2 = (BonaPiece)(sq + t[i + 1]);
					BonaPiece r2 = (BonaPiece)(Mir(sq) + t[i + 1]);
					mir_piece[p2] = r2;
					mir_piece[r2] = p2;

					break;
				}
			}
		}

		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
			if (inv_piece[p] == (BonaPiece)-1
				|| mir_piece[p] == (BonaPiece)-1
				)
			{
				// 未初期化のままになっている。上のテーブルの初期化コードがおかしい。
				ASSERT(false);
			}

#if 0
		// 評価関数のミラーをしても大丈夫であるかの事前検証
		// 値を書き込んだときにassertionがあるので、ミラーしてダメである場合、
		// そのassertに引っかかるはず。

		// AperyのWCSC26の評価関数、kppのp1==0とかp1==20(後手の0枚目の歩)とかの
		// ところにゴミが入っていて、これを回避しないとassertに引っかかる。

		std::unordered_set<BonaPiece> s;
		vector<int> a = {
			f_hand_pawn - 1,e_hand_pawn - 1,
			f_hand_lance - 1, e_hand_lance - 1,
			f_hand_knight - 1, e_hand_knight - 1,
			f_hand_silver - 1, e_hand_silver - 1,
			f_hand_gold - 1, e_hand_gold - 1,
			f_hand_bishop - 1, e_hand_bishop - 1,
			f_hand_rook - 1, e_hand_rook - 1,
		};
		for (auto b : a)
			s.insert((BonaPiece)b);

		// さらに出現しない升の盤上の歩、香、桂も除外(Aperyはここにもゴミが入っている)
		for (Rank r = RANK_1; r <= RANK_2; ++r)
			for (File f = FILE_1; f <= FILE_9; ++f)
			{
				if (r == RANK_1)
				{
					// 1段目の歩
					BonaPiece b1 = BonaPiece(f_pawn + (f | r));
					s.insert(b1);
					s.insert(inv_piece[b1]);

					// 1段目の香
					BonaPiece b2 = BonaPiece(f_lance + (f | r));
					s.insert(b2);
					s.insert(inv_piece[b2]);
				}

				// 1,2段目の桂
				BonaPiece b = BonaPiece(f_knight + (f | r));
				s.insert(b);
				s.insert(inv_piece[b]);
			}

		cout << "\nchecking kpp_write()..";
		for (auto sq : SQ)
		{
			cout << sq << ' ';
			for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
					if (!s.count(p1) && !s.count(p2))
						kpp_write(sq, p1, p2, kpp[sq][p1][p2]);
		}
		cout << "\nchecking kkp_write()..";

		for (auto sq1 : SQ)
		{
			cout << sq1 << ' ';
			for (auto sq2 : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
					if (!s.count(p1))
						kkp_write(sq1, sq2, p1, kkp[sq1][sq2][p1]);
		}
		cout << "..done!" << endl;
#endif
	}

#endif // EVAL_LEARN

	// 評価関数ファイルを読み込む
	void load_eval_impl()
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

#if 0
			// Aperyの評価関数バイナリ、kppのp=0のところでゴミが入っている。
			// 駒落ちなどではここを利用したいので0クリアすべき。
			const ValueKpp kpp_zero = { 0,0 };
			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				{
					kpp[sq][p1][0] = kpp_zero;
					kpp[sq][0][p1] = kpp_zero;
				}

			const ValueKkp kkp_zero = { 0,0 };
			for (auto sq1 : SQ)
				for (auto sq2 : SQ)
					kkp[sq1][sq2][0] = kkp_zero;
#endif

#if 0
			// Aperyの評価関数バイナリ、kkptは意味があるけどkpptはあまり意味がないので
			// 手番価値をクリアする実験用のコード

			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
					for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
						kpp[sq][p1][p2][1] = 0;

#endif

#ifdef EVAL_LEARN
			eval_learn_init();
#endif

		}

		// 読み込みは成功した。

		return;

	Error:;
		// 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
		sync_cout << "\ninfo string Error! open evaluation file failed.\n" << sync_endl;
		exit(EXIT_FAILURE);
	}



#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_MSC_VER)
	// 評価関数の共有を行うための大掛かりな仕組み

#include <windows.h>

	void load_eval()
	{
		// エンジンのバージョンによって評価関数は一意に定まるものとする。
		auto mapped_file_name = TEXT("YANEURAOU_KPPT_MMF" ENGINE_VERSION);
		auto mutex_name = TEXT("YANEURAOU_KPPT_MUTEX" ENGINE_VERSION);

		// プロセス間の排他用mutex
		auto hMutex = CreateMutex(NULL, FALSE, mutex_name);

		// ファイルマッピングオブジェクトの処理をプロセス間で排他したい。
		WaitForSingleObject(hMutex, INFINITE);
		{

			// ファイルマッピングオブジェクトの作成
			auto hMap = CreateFileMapping(INVALID_HANDLE_VALUE,
				NULL,
				PAGE_READWRITE, // | /**SEC_COMMIT/**/ /*SEC_RESERVE/**/,
				0, sizeof(SharedEval),
				mapped_file_name);

			bool already_exists = (GetLastError() == ERROR_ALREADY_EXISTS);

			// ビュー
			auto shared_eval_ptr = (SharedEval *)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedEval));

			kk_ = &(shared_eval_ptr->kk_);
			kkp_ = &(shared_eval_ptr->kkp_);
			kpp_ = &(shared_eval_ptr->kpp_);


			if (!already_exists)
			{
				// 新規作成されてしまった

				// このタイミングで評価関数バイナリを読み込む
				load_eval_impl();

				sync_cout << "info string created shared eval memory." << sync_endl;

			} else {

				// 評価関数バイナリを読み込む必要はない。ファイルマッピングが成功した時点で
				// 評価関数バイナリは他のプロセスによって読み込まれていると考えられる。

				sync_cout << "info string use shared eval memory." << sync_endl;
			}
		}
		ReleaseMutex(hMutex);

		// 終了時に本当ならば
		// 1) ::ReleaseMutex()
		// 2) ::UnmapVieOfFile()
		// が必要であるが、1),2)がプロセスが解体されるときに自動でなされるので、この処理は特に入れない。
	}
#else

	// 評価関数のプロセス間共有を行わないときは、普通に
	// load_eval_impl()を呼び出すだけで良い。
	void load_eval()
	{
		load_eval_impl();
	}

#endif


	// KP,KPP,KKPのスケール
	const int FV_SCALE = 32;

	// 駒割り以外の全計算
	// pos.st->BKPP,WKPP,KPPを初期化する。Position::set()で一度だけ呼び出される。(以降は差分計算)
	// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
	// なので、この関数の最適化は頑張らない。
	Value compute_eval(const Position& pos)
	{
#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_MSC_VER)
		// shared memoryを用いているときには、is_ready()で評価関数を読み込み、
		// 初期化してからしかcompute_eval()を呼び出すことは出来ない。
		ASSERT_LV1(kk_ != nullptr);
#endif

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

		int i, j;
		BonaPiece k0, k1, l0, l1;

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
		for (int i = 1; i < PIECE_NO_KING; ++i) {
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
		for (int i = 1; i < PIECE_NO_KING; ++i) {
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

		const auto* pkppb = kpp[sq_bk][ebp.fb];
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
		for (int i = 1; i < PIECE_NO_KING; ++i) {
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
		//
		// root nodeではprevious == nullptrであるが、root nodeではPosition::set()でcompute_eval()
		// を呼び出すので通常この関数が呼び出されることはないのだが、学習関係でこれが出来ないと
		// コードが書きにくいのでEVAL_LEARNのときは、このチェックをする。
		if (
#ifdef EVAL_LEARN
			prev == nullptr ||
#endif
			!prev->sum.evaluated())
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

			// ΣKKPは最初から全計算するしかないので初期化する。
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

					// 後手から見たKKP。後手から見ているのでマイナス
					diff.p[2][0] -= kkp[Inv(sq_wk)][Inv(sq_bk)][k1][0];
					// 後手から見たKKP手番。後手から見るのでマイナスだが、手番は先手から見たスコアを格納するのでさらにマイナスになって、プラス。
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
