#ifndef _EVALUATE_LEARN_CPP_
#define _EVALUATE_LEARN_CPP_

// KPPT評価関数の学習時用のコード

#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

#include "learn.h"

#include "../evaluate.h"
#include "../eval/evaluate_kppt.h"
#include "../eval/kppt_evalsum.h"
#include "../position.h"
#include "../misc.h"

using namespace std;

namespace Eval
{
	// 絶対値を抑制するマクロ
#define SET_A_LIMIT_TO(X,MIN,MAX)  \
	X[0] = std::min(X[0],(MAX));     \
	X[0] = std::max(X[0],(MIN));     \
	X[1] = std::min(X[1],(MAX));     \
	X[1] = std::max(X[1],(MIN));

	typedef std::array<int32_t, 2> ValueKk;
	typedef std::array<int16_t, 2> ValueKpp;
	typedef std::array<int32_t, 2> ValueKkp;

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

	// kpp_write()するときの一番若いアドレス(index)を返す。
	u64 get_kpp_index(Square k1, BonaPiece p1, BonaPiece p2)
	{
		BonaPiece mp1 = mir_piece[p1];
		BonaPiece mp2 = mir_piece[p2];
		Square  mk1 = Mir(k1);

		const auto q0 = &kpp[0][0][0];
		auto q1 = &kpp[k1][p1][p2] - q0;
		auto q2 = &kpp[k1][p2][p1] - q0;
		auto q3 = &kpp[mk1][mp1][mp2] - q0;
		auto q4 = &kpp[mk1][mp2][mp1] - q0;

		return std::min({ q1, q2, q3, q4 });
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

		ASSERT_LV3(kkp[k1][k2][p1] == kkp[mk1][mk2][mp1]);

		kkp[k1][k2][p1]
			= kkp[mk1][mk2][mp1]
			= value;

	// kkpに関して180度のflipは入れないほうが良いのでは..
#if 1
		ASSERT_LV3(kkp[k1][k2][p1][0] == -kkp[ik2][ik1][ip1][0]);
		ASSERT_LV3(kkp[k1][k2][p1][1] == +kkp[ik2][ik1][ip1][1]);
		ASSERT_LV3(kkp[ik2][ik1][ip1] == kkp[mik2][mik1][mip1]);

		kkp[ik2][ik1][ip1][0]
			= kkp[mik2][mik1][mip1][0]
			= -value[0];

		kkp[ik2][ik1][ip1][1]
			= kkp[mik2][mik1][mip1][1]
			= +value[1];
#endif
	}

#if 0
	// kkp_write()するときの一番若いアドレス(index)を返す。
	u64 get_kkp_index(Square k1, Square k2 , BonaPiece p1)
	{
		BonaPiece ip1 = inv_piece[p1];
		BonaPiece mp1 = mir_piece[p1];
		BonaPiece mip1 = mir_piece[ip1];
		Square  mk1 = Mir(k1);
		Square  ik1 = Inv(k1);
		Square mik1 = Mir(ik1);
		Square  mk2 = Mir(k2);
		Square  ik2 = Inv(k2);
		Square mik2 = Mir(ik2);

		const auto q0 = &kkp[0][0][0];
		auto q1 = &kkp[k1][k2][p1] - q0;
		auto q2 = &kkp[mk1][mk2][mp1] - q0;

		return std::min({ q1, q2 });
	}
#endif

	typedef std::array<float, 2> ValueKkFloat;
	typedef std::array<float, 2> ValueKppFloat;
	typedef std::array<float, 2> ValueKkpFloat;

	// 勾配等の配列
	struct WeightKk
	{
		ValueKkFloat w;   // 元の重み
		ValueKkFloat g;   // トータルの勾配

#ifdef USE_SGD_UPDATE
		u32   count;      // この特徴の出現回数
#endif

	#ifdef		USE_ADA_GRAD_UPDATE
		ValueKkFloat g2;  // AdaGradのg2
	#endif

		void add_grad(ValueKkFloat delta)
		{
			g += delta;
#ifdef USE_SGD_UPDATE
			count++;
#endif
		}
	};

	struct WeightKpp
	{
		ValueKppFloat w;   // 元の重み
		ValueKppFloat g;  // トータルの勾配
#ifdef USE_SGD_UPDATE
		u32   count;      // この特徴の出現回数
#endif

#ifdef		USE_ADA_GRAD_UPDATE
		ValueKppFloat g2;  // AdaGradのg2
#endif

		void add_grad(ValueKppFloat delta)
		{
			g += delta;
#ifdef USE_SGD_UPDATE
			count++;
#endif
		}
	};

	struct WeightKkp
	{
		ValueKkpFloat w;   // 元の重み
		ValueKkpFloat g;   // トータルの勾配

#ifdef USE_SGD_UPDATE
		u32   count;       // この特徴の出現回数
#endif

#ifdef		USE_ADA_GRAD_UPDATE
		ValueKkpFloat g2;  // AdaGradのg2
#endif

		void add_grad(ValueKkpFloat delta)
		{
			g += delta;
#ifdef USE_SGD_UPDATE
			count++;
#endif
		}
	};

	WeightKk(*kk_w_)[SQ_NB][SQ_NB];
	WeightKpp(*kpp_w_)[SQ_NB][fe_end][fe_end];
	WeightKkp(*kkp_w_)[SQ_NB][SQ_NB][fe_end];

#define kk_w (*kk_w_)
#define kpp_w (*kpp_w_)
#define kkp_w (*kkp_w_)

	// 学習のときの勾配配列の初期化
	void init_grad()
	{
		if (kk_w_ == nullptr)
		{
			u64 size;

			size = u64(SQ_NB)*u64(SQ_NB);
			kk_w_ = (WeightKk(*)[SQ_NB][SQ_NB])new WeightKk[size];
			memset(kk_w_, 0, sizeof(WeightKk) * size);
#ifdef RESET_TO_ZERO_VECTOR
			cout << "[RESET_TO_ZERO_VECTOR]";
			memset(kk_, 0, sizeof(ValueKk) * size);
#endif

			size = u64(SQ_NB)*u64(fe_end)*u64(fe_end);
			kpp_w_ = (WeightKpp(*)[SQ_NB][fe_end][fe_end])new WeightKpp[size];
			memset(kpp_w_, 0, sizeof(WeightKpp) * size);
#ifdef RESET_TO_ZERO_VECTOR
			memset(kpp_, 0, sizeof(ValueKpp) * size);
#endif

			size = u64(SQ_NB)*u64(SQ_NB)*u64(fe_end);
			kkp_w_ = (WeightKkp(*)[SQ_NB][SQ_NB][fe_end])new WeightKkp[size];
			memset(kkp_w_, 0, sizeof(WeightKkp) * size);
#ifdef RESET_TO_ZERO_VECTOR
			memset(kkp_, 0, sizeof(ValueKkp) * size);
#endif

			// 重みのコピー
			for (auto k1 : SQ)
				for (auto k2 : SQ)
					kk_w[k1][k2].w = { float(kk[k1][k2][0]) , float(kk[k1][k2][1]) };

			for (auto k : SQ)
				for (auto p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
					for (auto p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
						kpp_w[k][p1][p2].w = { float(kpp[k][p1][p2][0]) , float(kpp[k][p1][p2][1]) };

			for (auto k1 : SQ)
				for (auto k2 : SQ)
					for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
						kkp_w[k1][k2][p].w = { float(kkp[k1][k2][p][0]) , float(kkp[k1][k2][p][1]) };

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
		kk_w[sq_bk][sq_wk].add_grad( ValueKkFloat{ f , g } );

		for (i = 0; i < PIECE_NO_KING; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

				// kpp配列に関してはミラー(左右判定)とフリップ(180度回転)の次元下げを行う。

				// kpp_w[sq_bk][k0][l0].add_grad(ValueKppFloat{ f ,  g });
				// kpp_w[Inv(sq_wk)][k1][l1].add_grad(ValueKppFloat{ -f ,  g });

				((WeightKpp*)kpp_w_)[get_kpp_index(sq_bk, k0, l0)].add_grad(ValueKppFloat{ f ,  g });
				((WeightKpp*)kpp_w_)[get_kpp_index(Inv(sq_wk), k1, l1)].add_grad(ValueKppFloat{ -f ,  g });

#if 0
				// ミラーもやめると…？
				kpp_w[sq_bk][k0][l0].add_grad(ValueKppFloat{ f ,  g });
				kpp_w[sq_bk][l0][k0].add_grad(ValueKppFloat{ f ,  g });
				kpp_w[Inv(sq_wk)][k1][l1].add_grad(ValueKppFloat{ -f ,  g });
				kpp_w[Inv(sq_wk)][l1][k1].add_grad(ValueKppFloat{ -f ,  g });
#endif
			}

			// ((WeightKkp*)kkp_w_)[get_kkp_index(sq_bk,sq_wk,k0)].add_grad( ValueKkpFloat{ f , g });

			// kkpは次元下げ、ミラーも含めてやらないことにする。どうせ教師の数は足りているし、
			// 右と左とでは居飛車、振り飛車的な戦型選択を暗に含むからミラーするのが良いとは限らない。
			// 180度回転も先手後手とは非対称である可能性があるのでここのフリップも入れない。

			kkp_w[sq_bk][sq_wk][k0].add_grad(ValueKkpFloat{ f , g });
		}

	}


// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights()
	{
		// kkpの一番大きな値を表示させることで学習が進んでいるかのチェックに用いる。
#ifdef DISPLAY_STATS_IN_UPDATE_WEIGHTS
		float max_kkp = 0.0f;
#endif

#ifdef USE_ADA_GRAD_UPDATE
		// 学習率η = 0.01として勾配が一定な場合、1万回でη×199ぐらい。
		// cf. [AdaGradのすすめ](http://qiita.com/ak11/items/7f63a1198c345a138150)
		const float eta = 2.0f;
		// 初回更新量はeta。そこから小さくなっていく。

		// 最初のほうの更新量を抑制する項を独自に追加しておく。
		const float epsilon = 1.0f;
#endif

#ifdef USE_SGD_UPDATE

		// SGDの場合、勾配自動調整ではないので、損失関数に合わせて適宜調整する必要がある。

		#if defined (LOSS_FUNCTION_IS_CROSS_ENTOROPY)
			const float eta = 32;
		#elif defined (LOSS_FUNCTION_IS_WINNING_PERCENTAGE)
			const float eta = 10 * 32;
		#endif
#endif
		// 手番用の学習率。これはηより小さめで良いはず。(小さめの値がつくべきところなので)
		const float eta2 = eta * LEARN_ETA2_RATE;

		// g2[i] += g * g;
		// w[i] -= η * g / sqrt(g2[i]);
		// g = 0 // mini-batchならこのタイミングで勾配配列クリア。

		for (auto k1 : SQ)
			for (auto k2 : SQ)
			{
				auto& w = kk_w[k1][k2];
#ifdef DISPLAY_STATS_IN_UPDATE_WEIGHTS
				max_kkp = max(max_kkp, abs(w.w[0]));
#endif

#ifdef USE_ADA_GRAD_UPDATE

				if (w.g[0] == 0 && w.g[1] == 0)
					goto NEXT_KK;

				auto gg = ValueKkFloat{ w.g[0] * w.g[0] , w.g[1] * w.g[1] };
				w.g2 += gg;

				// AdaGradの更新式
				//    w -= eta * g / sqrt(g2);
				// これ、g2が小さいときにおかしくなるので、
				//    w -= eta * g / sqrt(g2 + ε);
				// のようにε項を追加しておく。

				w.w = ValueKkFloat{ w.w[0] -eta * w.g[0] / sqrt(w.g2[0] + epsilon) ,w.w[1] -eta2 * w.g[1] / sqrt(w.g2[1] + epsilon) };

#endif

#ifdef USE_SGD_UPDATE
				// 勾配はこの特徴の出現したサンプル数で割ったもの。
				if (w.count == 0)
					goto NEXT_KK;
				w.g[0] /= w.count;
				w.g[1] /= w.count;
				w.count = 0;
				w.w = ValueKkFloat { w.w[0] -eta * w.g[0] , w.w[1] -eta2 * w.g[1] };
#endif
				// 絶対値を抑制する。
				SET_A_LIMIT_TO(w.w , float(INT16_MIN / 2), float(INT16_MAX / 2));

				kk[k1][k2] = { (s32)w.w[0], (s32)w.w[1] };

			NEXT_KK:;

				w.g = { 0.0f,0.0f };
			}

		for (auto k : SQ)
			for (auto p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (auto p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
				{
					auto& w = kpp_w[k][p1][p2];

#ifdef USE_ADA_GRAD_UPDATE
					if (w.g[0] == 0 && w.g[1] == 0)
						goto NEXT_KPP;

					auto gg = ValueKppFloat{ w.g[0] * w.g[0] , w.g[1] * w.g[1] };
					w.g2 += gg;
					w.w = ValueKppFloat{ w.w[0] - eta * w.g[0] / sqrt(w.g2[0] + epsilon) , w.w[1] - eta2 * w.g[1] / sqrt(w.g2[1] + epsilon) };

#endif

#ifdef USE_SGD_UPDATE
					if (w.count == 0)
						goto NEXT_KPP;
					w.g[0] /= w.count;
					w.g[1] /= w.count;
					w.count = 0;
					w.w = ValueKppFloat{ w.w[0] - eta * w.g[0] , w.w[1] - eta2 * w.g[1] };
#endif

					// 絶対値を抑制する。
					SET_A_LIMIT_TO(w.w , (float)(INT16_MIN / 2), (float)(INT16_MAX / 2));

					kpp_write(k, p1, p2, ValueKpp{ (s16)w.w[0], (s16)w.w[1] });

//					kpp[k][p1][p2] = ValueKpp{ (s16)w.w[0], (s16)w.w[1] };

					//kpp[k][p1][p2] = ValueKpp{ (s16)w.w[0], (s16)w.w[1] };
					//kpp[k][p2][p1] = ValueKpp{ (s16)w.w[0], (s16)w.w[1] };

				NEXT_KPP:;

					w.g = { 0.0f,0.0f };
				}

		for (auto k1 : SQ)
			for (auto k2 : SQ)
				for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					auto& w = kkp_w[k1][k2][p];

					// cout << "\n" << w.g[0] << " & " << w.g[1];

#ifdef USE_ADA_GRAD_UPDATE
					if (w.g[0] == 0 && w.g[1] == 0)
						goto NEXT_KKP;

					auto gg = ValueKkpFloat{ w.g[0] * w.g[0] , w.g[1] * w.g[1] };
					w.g2 += gg;
					w.w = ValueKkpFloat{ w.w[0] - eta * w.g[0] / sqrt(w.g2[0] + epsilon) , w.w[1] - eta2 * w.g[1] / sqrt(w.g2[1] + epsilon) };

//					cout << endl << w.g[0] << "," << w.g[1] << " : " << w.g2[0] << "," << w.g2[1];

#endif

#ifdef USE_SGD_UPDATE
					if (w.count == 0)
						goto NEXT_KKP;
					w.g[0] /= w.count;
					w.g[1] /= w.count;
					w.count = 0;
					w.w = ValueKkpFloat{ w.w[0] - eta * w.g[0]  , w.w[1] - eta2 * w.g[1] };
#endif

					// 絶対値を抑制する。
					SET_A_LIMIT_TO(w.w, (float)(INT16_MIN / 2), (float)(INT16_MAX / 2));

					//	kkp_write(k1, k2, p, ValueKkp{s32(w.w[0]),s32(w.w[1])});

					// kkpは上で書いた理由で次元下げをしない。
					kkp[k1][k2][p] = ValueKkp{ s32(w.w[0]),s32(w.w[1]) };

				NEXT_KKP:;
					w.g = { 0.0f,0.0f };
				}

#ifdef DISPLAY_STATS_IN_UPDATE_WEIGHTS
		cout << " , max_kkp = " << max_kkp << " ";
#endif
	}


	// 学習のためのテーブルの初期化
	void eval_learn_init()
	{
		// fとeとの交換
		int t[] = {
			f_hand_pawn - 1    , e_hand_pawn - 1   ,
			f_hand_lance - 1   , e_hand_lance - 1  ,
			f_hand_knight - 1  , e_hand_knight - 1 ,
			f_hand_silver - 1  , e_hand_silver - 1 ,
			f_hand_gold - 1    , e_hand_gold - 1   ,
			f_hand_bishop - 1  , e_hand_bishop - 1 ,
			f_hand_rook - 1    , e_hand_rook - 1   ,
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


	void save_eval(std::string suffix)
	{
		{
			// KK
			std::ofstream ofsKK((string)Options["EvalDir"] + KK_BIN + suffix, std::ios::binary);
			if (!ofsKK.write(reinterpret_cast<char*>(kk), sizeof(kk)))
				goto Error;

			// KKP
			std::ofstream ofsKKP((string)Options["EvalDir"] + KKP_BIN + suffix, std::ios::binary);
			if (!ofsKKP.write(reinterpret_cast<char*>(kkp), sizeof(kkp)))
				goto Error;

			// KPP
			std::ofstream ofsKPP((string)Options["EvalDir"] + KPP_BIN + suffix, std::ios::binary);
			if (!ofsKPP.write(reinterpret_cast<char*>(kpp), sizeof(kpp)))
				goto Error;

			cout << "save_eval() finished." << endl;

			return;
		}

	Error:;
		cout << "Error : save_eval() failed" << endl;
	}

} // namespace Eval

#endif // EVAL_LEARN

#endif
