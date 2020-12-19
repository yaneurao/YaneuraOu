// KPP_KKPT評価関数の学習時用のコード
// KPPTの学習用コードのコピペから少しいじった感じ。

#include "../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_NABLA)

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../learn/learn.h"
#include "../../learn/learning_tools.h"

#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"

#include "../evaluate_io.h"
#include "../evaluate_common.h"

#include "evaluate_nabla.h"

// --- 以下、定義

namespace Eval
{
	using namespace EvalLearningTools;

	// bugなどにより間違って書き込まれた値を補正する。
	void correct_eval()
	{
		// kppのp1==p2のところ、値はゼロとなっていること。
		// (差分計算のときにコードの単純化のために参照はするけど学習のときに使いたくないので)
		// kppのp1==p2のときはkkpに足しこまれているという考え。
		{
			const ValueKpp kpp_zero = 0;
			float sum = 0;
			for (auto sq : SQ)
				for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					sum += abs(kpp_ksq_pcpc(sq,p,p));
					kpp_ksq_pcpc(sq,p,p) = kpp_zero;
				}
			//	cout << "info string sum kp = " << sum << endl;
		}

		// 以前Aperyの評価関数バイナリ、kppのp=0のところでゴミが入っていた。
		// 駒落ちなどではここを利用したいので0クリアすべき。
		{
			const ValueKkp kkp_zero = { 0,0 };
			for (auto sq1 : SQ)
				for (auto sq2 : SQ)
					kkp[sq1][sq2][0] = kkp_zero;

			const ValueKpp kpp_zero = 0;
			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				{
					kpp_ksq_pcpc(sq,p1,BONA_PIECE_ZERO) = kpp_zero;
					kpp_ksq_pcpc(sq,BONA_PIECE_ZERO,p1) = kpp_zero;
				}
		}

#if defined(USE_KK_INVERSE_WRITE)
		// KKの先後対称性から、kk[bk][wk][0] == -kk[Inv(wk)][Inv(bk)][0]である。
		// bk == Inv(wk)であるときも、この式が成立するので、このとき、kk[bk][wk][0] == 0である。
		{
			for (auto sq : SQ)
				kk[sq][Inv(sq)][0] = 0;
		}
#endif
	}


	// 評価関数学習用の構造体

	// KK,KKPのWeightを保持している配列
	// 直列化してあるので1次元配列
	std::vector<Weight2> weights;

	// KPPは手番なしなので手番なし用の1次元配列。
	std::vector<Weight> weights_kpp;

	// 学習配列のデザイン
	namespace
	{
		KK g_kk;
		KKP g_kkp;
		KPP g_kpp;
	}

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// bugなどにより誤って書き込まれた値を補正する。
		correct_eval();

		// 学習で使用するテーブル類の初期化
		EvalLearningTools::init();
			
		// 学習配列のデザイン
		g_kk.set(SQ_NB, Eval::fe_end, 0);
		g_kkp.set(SQ_NB, Eval::fe_end, g_kk.max_index());
		g_kpp.set(SQ_NB, Eval::fe_end, g_kkp.max_index());

		// 学習用配列の確保
		u64 size = g_kkp.max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

		u64 size_kpp = g_kpp.size();
		weights_kpp.resize(size_kpp);
		memset(&weights_kpp[0], 0, sizeof(Weight) * weights_kpp.size());

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);

	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad , const std::array<bool, 4>& freeze)
	{
		const bool freeze_kpp = freeze[2];

		// LearnFloatTypeにatomicつけてないが、2つのスレッドが、それぞれx += yと x += z を実行しようとしたとき
		// 極稀にどちらか一方しか実行されなくともAdaGradでは問題とならないので気にしないことにする。
		// double型にしたときにWeight.gが破壊されるケースは多少困るが、double型の下位4バイトが破壊されたところで
		// それによる影響は小さな値だから実害は少ないと思う。
		
		// 勾配に多少ノイズが入ったところで、むしろ歓迎！という意味すらある。
		// (cf. gradient noise)

		// Aperyに合わせておく。
		delta_grad /= 32.0 /*FV_SCALE*/;

		// 勾配
		std::array<LearnFloatType,2> g =
		{
			// 手番を考慮しない値
			(rootColor == BLACK             ) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad),

			// 手番を考慮する値
			(rootColor == pos.side_to_move()) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad)
		};

		// 180度盤面を回転させた位置関係に対する勾配
		std::array<LearnFloatType,2> g_flip = { -g[0] , g[1] };

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);

		auto& pos_ = *const_cast<Position*>(&pos);

		// piece_listが最新の状態になっていない可能性があるので更新を適用する。
		auto& eval_list = *const_cast<EvalList*>(pos.eval_list());
		auto& dp = pos.state()->dirtyPiece;
		if (!dp.updated())
			dp.do_update(eval_list);

#if !defined (USE_EVAL_MAKE_LIST_FUNCTION)

		auto list_fb = eval_list.piece_list_fb();
		auto list_fw = eval_list.piece_list_fw();
		int length = eval_list.length();
#else
		// -----------------------------------
		// USE_EVAL_MAKE_LIST_FUNCTIONが定義されているときは
		// ここでeval_listをコピーして、組み替える。
		// -----------------------------------

		const int length = 40;

		// バッファを確保してコピー
		BonaPiece list_fb[length];
		BonaPiece list_fw[length];
		memcpy(list_fb, eval_list.piece_list_fb(), sizeof(BonaPiece) * length);
		memcpy(list_fw, eval_list.piece_list_fw(), sizeof(BonaPiece) * length);

		// ユーザーは、この関数でBonaPiece番号の自由な組み換えを行なうものとする。
		make_list_function(pos, list_fb, list_fw);
#endif

		// KK
		weights[g_kk.fromKK(sq_bk,sq_wk).toIndex()].add_grad(g);

		for (int i = 0; i < length ; ++i)
		{
			BonaPiece k0 = list_fb[i];
			BonaPiece k1 = list_fw[i];

			// KPP
			if (!freeze_kpp)
			{
				// このループではk0 == l0は出現しない。(させない)
				// それはKPであり、KKPの計算に含まれると考えられるから。
				for (int j = 0; j < i; ++j)
				{
					BonaPiece l0 = list_fb[j];
					BonaPiece l1 = list_fw[j];

					weights_kpp[g_kpp.fromKPP(sq_bk     , k0, l0).toRawIndex()].add_grad(g[0]);
					weights_kpp[g_kpp.fromKPP(Inv(sq_wk), k1, l1).toRawIndex()].add_grad(g_flip[0]);
				}
			}

			// KKP
			weights[g_kkp.fromKKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	void update_weights(u64 epoch , const std::array<bool, 4>& freeze)
	{
		u64 vector_length = g_kpp.max_index();

		const bool freeze_kk = freeze[0];
		const bool freeze_kkp = freeze[1];
		const bool freeze_kpp = freeze[2];

		// KPPを学習させないなら、KKPのmaxまでだけで良い。あとは数が少ないからまあいいや。
		if (freeze_kpp)
			vector_length = g_kkp.max_index();

		// epochに応じたetaを設定してやる。
		Weight::calc_eta(epoch);

		// ゼロ定数 手番つき、手番なし
		const auto zero_t = std::array<LearnFloatType, 2> {0, 0};
		const auto zero = LearnFloatType(0);
		
		// 並列化を効かせたいので直列化されたWeight配列に対してループを回す。

#pragma omp parallel
		{

#if defined(_OPENMP)
			// Windows環境下でCPUが２つあるときに、論理64コアまでしか使用されないのを防ぐために
			// ここで明示的にCPUに割り当てる
			int thread_index = omp_get_thread_num();    // 自分のthread numberを取得
			WinProcGroup::bindThisThread(thread_index);
#endif

#pragma omp for schedule(dynamic,20000)
			for (s64 index_ = 0; index_ < (s64)vector_length; ++index_)
			{
				// OpenMPではループ変数は符号型変数でなければならないが
				// さすがに使いにくい。
				u64 index = (u64)index_;

				// 自分が更新すべきやつか？
				// 次元下げしたときのindexの小さいほうが自分でないならこの更新は行わない。
				if (!min_index_flag[index])
					continue;

				if (g_kk.is_ok(index) && !freeze_kk)
				{
					KK x = g_kk.fromIndex(index);

					// 次元下げ
					KK a[KK_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					// 次元下げで得た情報を元に、それぞれのindexを得る。
					u64 ids[KK_LOWER_COUNT];
					for (int i = 0; i < KK_LOWER_COUNT; ++i)
						ids[i] = a[i].toIndex();

					// それに基いてvの更新を行い、そのvをlowerDimensionsそれぞれに書き出す。
					// ids[0]==ids[1]==ids[2]==ids[3]みたいな可能性があるので、gは外部で合計する。
					std::array<LearnFloatType, 2> g_sum = zero_t;

					// inverseした次元下げに関しては符号が逆になるのでadjust_grad()を経由して計算する。
					for (int i = 0; i < KK_LOWER_COUNT; ++i)
						g_sum += a[i].apply_inverse_sign(weights[ids[i]].get_grad());
					
					// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
					if (is_zero(g_sum))
						continue;

					auto& v = kk[a[0].king0()][a[0].king1()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					for (int i = 1; i < KK_LOWER_COUNT; ++i)
						kk[a[i].king0()][a[i].king1()] = a[i].apply_inverse_sign(v);
					
					// mirrorした場所が同じindexである可能性があるので、gのクリアはこのタイミングで行なう。
					// この場合、毎回gを通常の2倍加算していることになるが、AdaGradは適応型なのでこれでもうまく学習できる。
					for (auto id : ids)
						weights[id].set_grad(zero_t);

				}
				else if (g_kkp.is_ok(index) && !freeze_kkp)
				{
					// KKの処理と同様

					KKP x = g_kkp.fromIndex(index);

					KKP a[KKP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					u64 ids[KKP_LOWER_COUNT];
					for (int i = 0; i < KKP_LOWER_COUNT; ++i)
						ids[i] = a[i].toIndex();

					std::array<LearnFloatType, 2> g_sum = zero_t;
					for (int i = 0; i <KKP_LOWER_COUNT; ++i)
						g_sum += a[i].apply_inverse_sign(weights[ids[i]].get_grad());
					
					if (is_zero(g_sum))
						continue;

					auto& v = kkp[a[0].king0()][a[0].king1()][a[0].piece()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					for (int i = 1; i < KKP_LOWER_COUNT; ++i)
						kkp[a[i].king0()][a[i].king1()][a[i].piece()] = a[i].apply_inverse_sign(v);
					
					for (auto id : ids)
						weights[id].set_grad(zero_t);

				}
				else if (g_kpp.is_ok(index) && !freeze_kpp)
				{
					KPP x = g_kpp.fromIndex(index);

					KPP a[KPP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					u64 ids[KPP_LOWER_COUNT];
					for (int i = 0; i < KPP_LOWER_COUNT; ++i)
						ids[i] = a[i].toRawIndex();

					// KPPに関してはinverseの次元下げがないので、inverseの判定は不要。

					// KPPTとの違いは、ここに手番がないというだけ。
					LearnFloatType g_sum = zero;
					for (auto id : ids)
						g_sum += weights_kpp[id].get_grad();

					if (g_sum == 0)
						continue;

					auto& v = kpp_ksq_pcpc(a[0].king(),a[0].piece0(),a[0].piece1());
					weights_kpp[ids[0]].set_grad(g_sum);
					weights_kpp[ids[0]].updateFV(v);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					for (int i = 1; i < KPP_LOWER_COUNT; ++i)
						kpp_ksq_pcpc(a[i].king(),a[i].piece0(),a[i].piece1()) = v;
#else
					// 三角配列の場合、KPP::toLowerDimensionsで、piece0とpiece1を入れ替えたものは返らないので
					// (同じindexを指しているので)、自分で入れ替えてkpp配列にvの値を反映させる。
					kpp_ksq_pcpc(a[0].king(),a[0].piece1(),a[0].piece0()) = v;
#if KPP_LOWER_COUNT == 2
					kpp_ksq_pcpc(a[1].king(),a[1].piece0(),a[1].piece1()) = v;
					kpp_ksq_pcpc(a[1].king(),a[1].piece1(),a[1].piece0()) = v;
#endif
#endif

					for (auto id : ids)
						weights_kpp[id].set_grad(zero);
				}
			}
		}
	}

	// kppのパラメーターをnabla型に展開する。
	// まず、歩を展開するコードから。
	// 変換コマンドを用意するの面倒なので、save_eval()の直前で明示的に呼び出して使う。要らなくなったらコメントアウトして欲しい。
	/*
	変換コマンド例)
		evaldir rezero_nabla_epoch5
		isready
		evalsavedir rezero_nabla_epoch5
		test evalsave

		※　注 : fe_endの拡張は済んでいるものとする。
	*/
	// 注意 : 現状、このルーチン、何らかバグっていて変換するとめっちゃ弱くなる。
	// どこか計算間違っているのでは…。
	void expand_kpp_to_nabla()
	{
		cout << "expand kpp to nabla1()";

		// 歩の位置関係を拡張する。
		// fe_old_end～fe_old_end + 4096までの領域をこの拡張領域として割り当てる。

		// 1024刻みで、
		// 1) 先手の1～5筋の歩を表現したものとする。
		// 2) 先手の5～9筋の歩を表現したものとする。
		// 3) 後手の5～9筋の歩
		// 4) 後手の1～9筋の歩
		//
		// 例えば先手であれば、7段、6段、5段、それ以外。という4通り。
		// これを5筋分で4*4*4*4*4 = 1024通りとして表現する。

		// よく考えたらKKPのほうのPも拡張されるのでKKPのほうも変更しなければならない…。

#if 0
		// デバッグのために出力して確認する。
		for (BonaPiece p = fe_old_end; p < fe_end; ++p)
		{
			std::vector<BPInfo> a;
			map_from_p(p, a);

			// raw_index
			int r = ((int)p - fe_old_end);

			cout << (int)p << " : (" << r << ") = ";
			for (auto q : a)
				cout << q.p << " , ";
			cout << endl;
		}
#endif

		// 正確に等価に変換するの激ムズなのでだいたい変換したあと、あとは学習に任せる。

#if 1
		// KKP拡張。
		std::vector<BPInfo> a;
		for (auto sq1 : SQ)
			for (auto sq2 : SQ)
				for (BonaPiece p = fe_old_end; p < fe_old_end + 4096; ++p)
				{
					map_from_nabla_pawn_to_bp(p, a);
					ValueKkp sum = { 0,0 };

					for (auto q1 : a)
						for (auto q2 : a)
						{
							ValueKkp v = { 0,0 };
							if (q1.p == q2.p)
							{
								// KKP

								// よく考えたら、5筋の歩、KKPの両側に載ってくるのでは…。これはまずい…。半分ずつ載せるべきか？
								// そのときの端数はどうなるのだ…。よく考えないと破綻してしまう。

								v = kkp[sq1][sq2][q1.p];
								// qが5筋の駒であるなら、半分だけ加算。

								if (q1.f == FILE_5)
								{
									// 元のpが右側用のやつなら、端数繰り上げ。
									if (q1.right)
									{
										// 端数繰り上げで2で割る

										// C++11で負の数の剰余は0または負となるように規定されているのでこれで正しい。
										v[0] = (v[0] / 2) + (v[0] % 2);
										v[1] = (v[1] / 2) + (v[1] % 2);
									}
									else
									{
										// 端数切り捨てで2で割る
										v = v / 2;
									}
									// これで右側用と左側用のkkpの値を足せば元のkkpの値と合致する。
								}

							}
							else {
								// あと、aの中の任意の2駒P,PによるKPPがこのKKPに載ってこないとおかしい。
								// 重複除去のため、p1 < p2の制約を入れる
								if (q1.p < q2.p)
								{
									// 手番なしなのでこれでいいのか..
									v[0] = kpp_ksq_pcpc(sq1     , q1.p           , q2.p)
										 - kpp_ksq_pcpc(Inv(sq2), inv_piece(q1.p), inv_piece(q2.p));
								}

							}
							sum += v;
						}

					kkp[sq1][sq2][p] = sum;
				}
#endif

#if 1
		// KPP拡張。
		std::vector<BPInfo> a1, a2;
		for (auto sq : SQ)
			for (BonaPiece p1 = fe_old_end; p1 < fe_old_end + 4096; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_old_end + 4096; ++p2)
				{
					if (p1 >= fe_old_end && p2 >= fe_old_end)
					{
						// p1 > p2 のときは、p1 < p2でp1,p2を入れ替えたものとして書き込まれる。
						// p1 == p2は、これはKPであり、KKPとして扱われるべき。
						if (p1 >= p2)
							continue;

						map_from_nabla_pawn_to_bp(p1, a1);
						map_from_nabla_pawn_to_bp(p2, a2);

						ValueKpp sum = 0;
						for (auto q1 : a1)
							for (auto q2 : a2)
							{
								ValueKpp v = 0;

								// -- 重複要素の削除など色々必要。

								// これはKKPのほうに載ってきている。
								if (q1.p == q2.p)
									continue;

								v = kpp_ksq_pcpc(sq, q1.p, q2.p);

								// 先手の5筋の歩と後手の5筋の歩
								// これは4倍の重複加算になってしまう..
								// TODO:端数の調整が必要。
								if (q1.f == FILE_5)
									v /= 2;
								if (q2.f == FILE_5)
									v /= 2;

								sum += v;
							}

							kpp_ksq_pcpc(sq, p1, p2) = sum;
							kpp_ksq_pcpc(sq, p2, p1) = sum;
						}
						else if (p1 >= fe_old_end && p2 < fe_old_end)
						{
							// これはこれでむずいのでは..

							map_from_nabla_pawn_to_bp(p1, a1);

							ValueKpp sum = 0;
							for (auto q1 : a1)
							{
								auto v = kpp_ksq_pcpc(sq, p2 , q1.p );

								// 5筋の歩は特別扱い。
								if (q1.f == FILE_5)
								{
									// 元のpが右側用のやつなら、端数繰り上げ。
									if (q1.right)
									{
										// 端数繰り上げで2で割る

										// C++11で負の数の剰余は0または負となるように規定されているのでこれで正しい。
										v = (v / 2) + (v % 2);
									}
									else
									{
										// 端数切り捨てで2で割る
										v = v / 2;
									}
								}
							}

							// p1,p2を入れ替えたところにも書き出しておく。
							kpp_ksq_pcpc(sq, p1, p2) = sum;
							kpp_ksq_pcpc(sq, p2, p1) = sum;
						}
					}
#endif
		cout << "..done" << endl;
	}

	// KKP/KPPのzeroである箇所をそこは未学習であると仮定して、何らかの値で埋めていく。
	void fill_kpp_where_zero()
	{
		auto exam_kkp = [] {
			u64 count_total = 0;
			// v[0]==0の箇所の数、v[1]==0の箇所の数、v[0]==v[1]==0の箇所の数。
			u64 count1 = 0, count2 = 0, count3 = 0;
			for (auto sq1 : SQ)
				for (auto sq2 : SQ)
					for (BonaPiece p = fe_nabla_pawn; p < fe_end; ++p)
					{
						auto& v = kkp[sq1][sq2][p];

						// 集計用
						count_total++;
						if (v[0] == 0)
							count1++;
						if (v[1] == 0)
							count2++;
						if (v[0] == 0 && v[1] == 0)
							count3++;
					}

			cout << "kkp count where v[0] == 0 is " << count1 << "/" << count_total << endl;
			cout << "kkp count where v[1] == 0 is " << count2 << "/" << count_total << endl;
			cout << "kkp count where v    == 0 is " << count3 << "/" << count_total << endl;
		};

		auto fill_kkp = [] {
			cout << "fill kkp where v[0]==0 and/or v[1]==0" << endl;
			for (auto sq1 : SQ)
				for (auto sq2 : SQ)
					for (BonaPiece p = BONA_PIECE_ZERO ; p < fe_end; ++p)
					{
						// KKPのp==BONA_PIECE_ZEROは0であって欲しいので余計なことをしない。
						if (p == BONA_PIECE_ZERO)
							continue;

						auto& v = kkp[sq1][sq2][p];
						for (int i = 0; i<2; ++i)
							if (v[i] == 0)
							{
								// KKPのそれぞれのKを片方ずつ任意の升に移動させたときの平均値でfillする。
								// 値がゼロのところは未学習なので無視

								s32 sum = 0;
								int c = 0;
								for (auto sq3 : SQ)
								{
									auto v2 = kkp[sq1][sq3][p][i];
									if (v2)
									{
										c++;
										sum += v2;
									}
									auto v3 = kkp[sq1][sq3][p][i];
									if (v3)
									{
										c++;
										sum += v3;
									}
								}

								if (c)
								{
									sum = sum / c;
									v[i] = sum;
								}
							}
					}
		};

		auto exam_kpp = [] {
			u64 count_total = 0;
			// v[0]==0の箇所の数
			u64 count1 = 0;
			for (auto sq1 : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO ; p1 < fe_end; ++p1)
					for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
					{
						auto& v = kpp_ksq_pcpc(sq1,p1,p2);

						// 集計用
						count_total++;
						if (v == 0)
							count1++;
					}

			cout << "kpp count where v    == 0 is " << count1 << "/" << count_total << endl;
		};

		auto fill_kpp = [] {
			cout << "fill kpp where v==0" << endl;

			for (auto sq1 : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end ; ++p1)
					for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
					{
						// KPPのp1==p2のところは、0であるべき。ここを書き換えてはならない。
						if (p1 == p2 || p1 == BONA_PIECE_ZERO || p2 == BONA_PIECE_ZERO)
							continue;

						auto& v = kpp_ksq_pcpc(sq1, p1, p2);

						if (v == 0)
						{
							s32 sum = 0;
							int c = 0;

							// KPPのKを任意の升に移動させたときのKPP値の平均でfillする
							// ただし値が0のところは未学習ということなので無視。
							for (auto sq2 : SQ)
							{
								auto v2 = kpp_ksq_pcpc(sq2, p1 , p2);
								if (v2)
								{
									c++;
									sum += v2;
								}
							}

							if (c)
							{
								sum = sum / c;
								v = sum;
							}
						}
					}
		};

#if 1
		// KKP
		cout << "fill_kkp_where_zero" << endl;
		exam_kkp();
		fill_kkp();
		exam_kkp();
#endif

#if 1
		// KPP
		cout << "fill_kpp_where_zero" << endl;
		exam_kpp();
		fill_kpp();
		exam_kpp();
#endif

		cout << "..done" << endl;
	}


	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
		// これ、あかんかった…。
		//expand_kpp_to_nabla();

		// KKP,KPPのゼロである要素を何らか埋める
		//fill_kpp_where_zero();
		//return;

		{
			auto eval_dir = Path::Combine((std::string)Options["EvalSaveDir"], dir_name);

			std::cout << "save_eval() start. folder = " << eval_dir << std::endl;

			// すでにこのフォルダがあるならmkdir()に失敗するが、
			// 別にそれは構わない。なければ作って欲しいだけ。
			// また、EvalSaveDirまでのフォルダは掘ってあるものとする。

			MKDIR(eval_dir);

			// EvalIOを利用して評価関数ファイルに書き込む。
			// 読み込みのときのinputとoutputとを入れ替えるとファイルに書き込める。EvalIo::eval_convert()マジ優秀。
			auto make_name = [&](std::string filename) { return Path::Combine(eval_dir, filename); };
			auto input = EvalIO::EvalInfo::build_nabla((void*)kk, (void*)kkp, (void*)kpp);
			auto output = EvalIO::EvalInfo::build_nabla(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));

			// 評価関数の実験のためにfe_endをKPPT32から変更しているかも知れないので現在のfe_endの値をもとに書き込む。
			input.fe_end = output.fe_end = Eval::fe_end;

			if (!EvalIO::eval_convert(input, output, nullptr))
				goto Error;

			std::cout << "save_eval() finished. folder = " << eval_dir << std::endl;
			return;
		}

	Error:;
		std::cout << "Error : save_eval() failed" << std::endl;
	}

	// 現在のetaを取得する。
	double get_eta() {
		return Weight::eta;
	}

}

#endif // EVAL_LEARN
