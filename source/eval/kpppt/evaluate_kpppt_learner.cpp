// KPP_KKPT評価関数の学習時用のコード
// KPPTの学習用コードのコピペから少しいじった感じ。

#include "../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_KPPPT)

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../learn/learn.h"
#include "../../learn/learning_tools.h"

#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"

#include "../evaluate_io.h"
#include "evaluate_kpppt.h"

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
			const ValueKpp kpp_zero = { 0,0 };
			float sum = 0;
			for (auto sq : SQ)
				for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					sum += abs(kpp[sq][p][p][0]) + abs(kpp[sq][p][p][1]);
					kpp[sq][p][p] = kpp_zero;
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

			const ValueKpp kpp_zero = { 0,0 };
			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				{
					kpp[sq][p1][0] = kpp_zero;
					kpp[sq][0][p1] = kpp_zero;
				}
		}


#if  defined(USE_KK_INVERSE_WRITE)
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

	// KPPは手番ありなので手番あり用の1次元配列。
	std::vector<Weight2> weights_kpp;

	// KPPPも手番ありなので手番あり用の1次元配列。
	// 手番ありに変更するかも知れないので、配列をkppと分けておく。
	std::vector<Weight2> weights_kppp;

	// 学習配列のデザイン
	namespace
	{
		KK g_kk;
		KKP g_kkp;
		KPP g_kpp;
		KPPP g_kppp;
	}

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// bugなどにより間違って書き込まれた値を補正する。
		correct_eval();

		// 学習配列のデザイン
		g_kk.set(SQ_NB, Eval::fe_end, 0);
		g_kkp.set(SQ_NB, Eval::fe_end, g_kk.max_index());
		g_kpp.set(SQ_NB, Eval::fe_end, g_kkp.max_index());
		g_kppp.set(Eval::KPPP_KING_SQ, Eval::fe_end, g_kpp.max_index());
		ASSERT(g_kppp.get_triangle_fe_end() == Eval::kppp_triangle_fe_end);

		// 学習で使用するテーブル類の初期化
		EvalLearningTools::init();
			
		// 学習用配列の確保

		// KK,KKP
		u64 size = g_kkp.max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。

		// KPP
		u64 size_kpp = g_kpp.size();
		
		// 三角配列を用いる場合、このassert式は成り立たない…。
		//ASSERT_LV1(size_kpp == size_of_kpp / sizeof(ValueKpp));

		weights_kpp.resize(size_kpp);

		// KPPP
		u64 size_kppp = g_kppp.size();
		ASSERT_LV1(size_kppp == size_of_kppp / sizeof(ValueKppp));
		weights_kppp.resize(size_kppp);

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);

	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad , const std::array<bool, 4>& freeze)
	{
		const bool freeze_kpp = freeze[2];
		const bool freeze_kppp = freeze[3];

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
		Square inv_sq_wk = Inv(sq_wk);

		auto& pos_ = *const_cast<Position*>(&pos);

		// バッファを確保してコピー
		BonaPiece list_fb[PIECE_NUMBER_KING];
		BonaPiece list_fw[PIECE_NUMBER_KING];
		memcpy(list_fb, pos_.eval_list()->piece_list_fb(), sizeof(BonaPiece) * (int)PIECE_NUMBER_KING);
		memcpy(list_fw, pos_.eval_list()->piece_list_fw(), sizeof(BonaPiece) * (int)PIECE_NUMBER_KING);

		// ユーザーは、この関数でBonaPiece番号の自由な組み換えを行なうものとする。
		// make_list_function(pos, list_fb, list_fw);

		// step 1. bk,inv_sq_wkで場合分けする。

		Square sq_bk_for_kppp = sq_bk;
		Square sq_wk_for_kppp = inv_sq_wk;

		// KPPPのbk,wk側が有効なのか
		bool kppp_bk_enable = (int)rank_of(sq_bk_for_kppp) >= KPPP_KING_RANK;
		bool kppp_wk_enable = (int)rank_of(sq_wk_for_kppp) >= KPPP_KING_RANK;

		// step 2. ミラーの処理が必要である。
		// このミラーの処理が入るだけでごっつい難しい…。

		if (kppp_bk_enable)
		{
#if defined(USE_KPPPT_MIRROR)
			if (file_of(sq_bk_for_kppp) >= FILE_6)
			{
				//  KK   は、sq_bk,sq_wk               に依存するので、これらをまとめてミラーしているからこのコードで合ってる。
				//  KKP  は、sq_bk,sq_wk,    Bから見たPに依存するので、これらをまとめてミラーしているからこのコードで合ってる。
				//  BKPP は、sq_bk,          Bから見たPに依存するので、この2つを同時にミラーしているからこれで合ってる。
				//  BKPPPは、sq_bk_for_kppp、Bから見たPに依存して、これらまとめてミラーしているからこのコードで合ってる。
				sq_bk_for_kppp = sq_bk = Mir(sq_bk);
				sq_wk = Mir(sq_wk);

				for (int i = 0; i < PIECE_NUMBER_KING ; ++i)
					list_fb[i] = EvalLearningTools::mir_piece(list_fb[i]);
			}
			// 玉の位置を詰める
			ASSERT_LV3((int)file_of(sq_bk_for_kppp) <= 4); // 0～4のはずなのだが
			sq_bk_for_kppp = (Square)((int)file_of(sq_bk_for_kppp) + ((int)rank_of(sq_bk_for_kppp) - KPPP_KING_RANK) * 5);
#else
			sq_bk_for_kppp = (Square)((int)file_of(sq_bk_for_kppp) + ((int)rank_of(sq_bk_for_kppp) - KPPP_KING_RANK) * 9);
#endif

			ASSERT_LV3((u64)sq_bk_for_kppp < KPPP_KING_SQ);
		}
#if KPPP_NULL_KING_SQ == 1
		else {
			sq_bk_for_kppp = (Square)KPPP_KING_SQ;
		}
#endif

		if (kppp_wk_enable)
		{
#if defined(USE_KPPPT_MIRROR)
			if (file_of(sq_wk_for_kppp) >= FILE_6)
			{
				// WKPP は、inv_sq_wk     , Wから見たPに依存するので、これらまとめてミラーしているので合ってる。
				// WKPPPは、sq_wk_for_kppp, Wから見たPに依存するので、これらまとめてミラーしているからこのコードで合ってる。
				sq_wk_for_kppp = inv_sq_wk = Mir(inv_sq_wk);

				for (int i = 0; i < PIECE_NUMBER_KING ; ++i)
					list_fw[i] = EvalLearningTools::mir_piece(list_fw[i]);
			}

			// 玉の位置を詰める
			ASSERT_LV3((int)file_of(sq_wk_for_kppp) <= 4); // 0～4のはずなのだが
			sq_wk_for_kppp = (Square)((int)file_of(sq_wk_for_kppp) + ((int)rank_of(sq_wk_for_kppp) - KPPP_KING_RANK) * 5);
#else
			sq_wk_for_kppp = (Square)((int)file_of(sq_wk_for_kppp) + ((int)rank_of(sq_wk_for_kppp) - KPPP_KING_RANK) * 9);
#endif
			ASSERT_LV3((u64)sq_wk_for_kppp < KPPP_KING_SQ);
		}
#if KPPP_NULL_KING_SQ == 1
		else {
			sq_wk_for_kppp = (Square)KPPP_KING_SQ;
		}
#endif

		// step 3. list_fb,list_fwを昇順に並び替える。
		my_insertion_sort(list_fb, 0, PIECE_NUMBER_KING);
		my_insertion_sort(list_fw, 0, PIECE_NUMBER_KING);

		int i, j, k;
		BonaPiece k0, k1, l0, l1, m0, m1;

		// KK
		weights[g_kk.fromKK(sq_bk,sq_wk).toIndex()].add_grad(g);

		// KPPPは、Kが対象範囲にないと適用できないので、
		// case 0. KPPPなし
		// case 1. 先手のみKPPPあり
		// case 2. 後手のみKPPPあり
		// case 3. 両方あり
		// の4つに場合分けされる。

		int kppp_case = (kppp_bk_enable ? 1 : 0) + (kppp_wk_enable ? 2 : 0);
		if (freeze_kppp)
			kppp_case = 0;

		for (i = 0; i < PIECE_NUMBER_KING; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];

			// このループではk0 == l0は出現しない。(させない)
			// それはKPであり、KKPの計算に含まれると考えられるから。
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

				if (!freeze_kpp)
				{
					weights_kpp[g_kpp.fromKPP(sq_bk    , k0, l0).toRawIndex()].add_grad(g);
					weights_kpp[g_kpp.fromKPP(inv_sq_wk, k1, l1).toRawIndex()].add_grad(g_flip);
				}

				// KPPP
				switch (kppp_case)
				{
				case 0: break;
				case 1:
					// BKPPP
					for (k = 0; k < j; ++k)
					{
						m0 = list_fb[k];
						weights_kppp[g_kppp.fromKPPP((int)sq_bk_for_kppp, k0, l0, m0).toRawIndex()].add_grad(g);
					}
					break;
				case 2:
					// WKPPP
					for (k = 0; k < j; ++k)
					{
						m1 = list_fw[k];
						weights_kppp[g_kppp.fromKPPP((int)sq_wk_for_kppp, k1, l1, m1).toRawIndex()].add_grad(g_flip);
					}
					break;
				case 3:
					// KPPP
					for (k = 0; k < j; ++k)
					{
						m0 = list_fb[k];
						m1 = list_fw[k];

						weights_kppp[g_kppp.fromKPPP((int)sq_bk_for_kppp, k0, l0, m0).toRawIndex()].add_grad(g);
						weights_kppp[g_kppp.fromKPPP((int)sq_wk_for_kppp, k1, l1, m1).toRawIndex()].add_grad(g_flip);
					}
					break;
				}
			}

			// KKP
			weights[g_kkp.fromKKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	// freeze_kk   : kkは学習させないフラグ
	// freeze_kkp  : kkpは学習させないフラグ
	// freeze_kpp  : kppは学習させないフラグ
	// freeze_kppp : kpppは学習させないフラグ
	void update_weights(u64 epoch, const std::array<bool, 4>& freeze)
	{
		u64 vector_length = g_kppp.max_index();

		const bool freeze_kk = freeze[0];
		const bool freeze_kkp = freeze[1];
		const bool freeze_kpp = freeze[2];
		const bool freeze_kppp = freeze[3];
			
		// epochに応じたetaを設定してやる。
		Weight::calc_eta(epoch);

		// ゼロ定数 手番つき
		const auto zero_t = std::array<LearnFloatType, 2> {0, 0};
		
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
				// ただし、KPPP配列に関しては無条件でokである。
				// "index < g_kpp.max_index()"のほうを先に書かないと、"min_index_flag[index]"のindexが配列の範囲外となる。

				if (index < g_kpp.max_index() && !min_index_flag[index])
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

					std::array<LearnFloatType, 2> g_sum = zero_t;
					for (auto id : ids)
						g_sum += weights_kpp[id].get_grad();

					if (is_zero(g_sum))
						continue;

					auto& v = kpp[a[0].king()][a[0].piece0()][a[0].piece1()];
					weights_kpp[ids[0]].set_grad(g_sum);
					weights_kpp[ids[0]].updateFV(v);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					for (int i = 1; i < KPP_LOWER_COUNT; ++i)
						kpp[a[i].king()][a[i].piece0()][a[i].piece1()] = v;
#else
					// 三角配列の場合、KPP::toLowerDimensionsで、piece0とpiece1を入れ替えたものは返らないので
					// (同じindexを指しているので)、自分で入れ替えてkpp配列にvの値を反映させる。
					kpp[a[0].king()][a[0].piece1()][a[0].piece0()] = v;
#if KPP_LOWER_COUNT == 2
					kpp[a[1].king()][a[1].piece0()][a[1].piece1()] = v;
					kpp[a[1].king()][a[1].piece1()][a[1].piece0()] = v;
#endif
#endif

					for (auto id : ids)
						weights_kpp[id].set_grad(zero_t);
				}
				else if (g_kppp.is_ok(index) && !freeze_kppp)
				{

#if KPPP_LOWER_COUNT >= 2 // このコード、汎用的ではあるが、遅い。
					KPPP x = g.kppp.fromIndex(index);

					KPPP a[KPPP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					u64 ids[KPPP_LOWER_COUNT];
					for (int i = 0; i < KPPP_LOWER_COUNT; ++i)
						ids[i] = a[i].toRawIndex();

					std::array<LearnFloatType, 2> g_sum = zero_t;
					for (auto id : ids)
						g_sum += weights_kppp[id].get_grad();

					if (is_zero(g_sum))
						continue;

					auto& v = kppp_ksq_pcpcpc(a[0].king(),a[0].piece0(),a[0].piece1(),a[0].piece2());
					weights_kppp[ids[0]].set_grad(g_sum);
					weights_kppp[ids[0]].updateFV(v);

					for (int i = 1; i < KPPP_LOWER_COUNT; ++i)
						kppp_ksq_pcpcpc(a[i].king(),a[i].piece0(),a[i].piece1(),a[i].piece2()) = v;

					for (auto id : ids)
						weights_kppp[id].set_grad(zero_t);
#else

					// 次元下げなしとわかっていれば、もっと単純化された速いコードが書ける。
					u64 raw_index = index - g_kppp.min_index();
					if (!is_zero(weights_kppp[raw_index].get_grad()))
					{
						auto& v = ((ValueKppp*)kppp)[raw_index];
						weights_kppp[raw_index].updateFV(v);
						weights_kppp[raw_index].set_grad(zero_t);
					}
#endif
				}
			}
		}
	}

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
		{
			auto eval_dir = path_combine((std::string)Options["EvalSaveDir"], dir_name);

			std::cout << "save_eval() start. folder = " << eval_dir << std::endl;

			// すでにこのフォルダがあるならmkdir()に失敗するが、
			// 別にそれは構わない。なければ作って欲しいだけ。
			// また、EvalSaveDirまでのフォルダは掘ってあるものとする。

			MKDIR(eval_dir);

			// EvalIOを利用して評価関数ファイルに書き込む。
			// 読み込みのときのinputとoutputとを入れ替えるとファイルに書き込める。EvalIo::eval_convert()マジ優秀。
			auto make_name = [&](std::string filename) { return path_combine(eval_dir, filename); };
			auto input = EvalIO::EvalInfo::build_kpppt32((void*)kk, (void*)kkp, (void*)kpp , (void*)kppp , size_of_kppp);
			auto output = EvalIO::EvalInfo::build_kpppt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN), make_name(KPPP_BIN) , size_of_kppp);

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
