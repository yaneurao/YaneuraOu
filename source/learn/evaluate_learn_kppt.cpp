#ifndef _EVALUATE_LEARN_KPPT_CPP_
#define _EVALUATE_LEARN_KPPT_CPP_

// KPPT評価関数の学習時用のコード
// tanuki-さんの学習部のコードをかなり参考にさせていただきました。

#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_KPPT)

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "learn.h"
#include "learning_tools.h"
#include "../eval/evaluate_io.h"

#include "../evaluate.h"
#include "../eval/evaluate_kppt.h"
#include "../eval/kppt_evalsum.h"
#include "../eval/evaluate_io.h"
#include "../position.h"
#include "../misc.h"

using namespace std;

namespace Eval
{
	// 学習のときの勾配配列の初期化
	void init_grad(double eta1, u64 eta_epoch, double eta2, u64 eta2_epoch, double eta3);

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad,bool without_kpp);

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights(u64 epoch,bool without_kpp);

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name);
}

// --- 以下、定義

namespace Eval
{
	using namespace EvalLearningTools;

	// 評価関数学習用の構造体

	// KK,KKP,KPPのWeightを保持している配列
	// 直列化してあるので1次元配列
	std::vector<Weight2> weights;

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// 学習で使用するテーブル類の初期化
		EvalLearningTools::init();
			
		// 学習用配列の確保
		u64 size = KPP::max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);
	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad , bool without_kpp)
	{
		// LearnFloatTypeにatomicつけてないが、2つのスレッドが、それぞれx += yと x += z を実行しようとしたとき
		// 極稀にどちらか一方しか実行されなくともAdaGradでは問題とならないので気にしないことにする。
		// double型にしたときにWeight.gが破壊されるケースは多少困るが、double型の下位4バイトが破壊されたところで
		// それによる影響は小さな値だから実害は少ないと思う。
		
		// 勾配に多少ノイズが入ったところで、むしろ歓迎！という意味すらある。
		// (cf. gradient noise)

		// Aperyに合わせておく。
		delta_grad /= 32.0 /*FV_SCALE*/;

		// 勾配
		array<LearnFloatType,2> g =
		{
			// 手番を考慮しない値
			(rootColor == BLACK             ) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad),

			// 手番を考慮する値
			(rootColor == pos.side_to_move()) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad)
		};

		// 180度盤面を回転させた位置関係に対する勾配
		array<LearnFloatType,2> g_flip = { -g[0] , g[1] };

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);

		auto& pos_ = *const_cast<Position*>(&pos);

#if !defined (USE_EVAL_MAKE_LIST_FUNCTION)

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

#else
		// -----------------------------------
		// USE_EVAL_MAKE_LIST_FUNCTIONが定義されているときは
		// ここでeval_listをコピーして、組み替える。
		// -----------------------------------

		// バッファを確保してコピー
		BonaPiece list_fb[40];
		BonaPiece list_fw[40];
		memcpy(list_fb, pos_.eval_list()->piece_list_fb(), sizeof(BonaPiece) * 40);
		memcpy(list_fw, pos_.eval_list()->piece_list_fw(), sizeof(BonaPiece) * 40);

		// ユーザーは、この関数でBonaPiece番号の自由な組み換えを行なうものとする。
		make_list_function(pos, list_fb, list_fw);
#endif

		// KK
		weights[KK(sq_bk,sq_wk).toIndex()].add_grad(g);

		// flipした位置関係にも書き込む
		//kk_w[Inv(sq_wk)][Inv(sq_bk)].g += g_flip;

		for (int i = 0; i < PIECE_NO_KING; ++i)
		{
			BonaPiece k0 = list_fb[i];
			BonaPiece k1 = list_fw[i];

			if (!without_kpp)
			{
				// このループではk0 == l0は出現しない。(させない)
				// それはKPであり、KKPの計算に含まれると考えられるから。
				for (int j = 0; j < i; ++j)
				{
					BonaPiece l0 = list_fb[j];
					BonaPiece l1 = list_fw[j];

					weights[KPP(sq_bk, k0, l0).toIndex()].add_grad(g);
					weights[KPP(Inv(sq_wk), k1, l1).toIndex()].add_grad(g_flip);
				}
			}

			// KKP
			weights[KKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	// without_kpp : kppは学習させないフラグ
	void update_weights(u64 epoch, bool without_kpp)
	{
		u64 vector_length = KPP::max_index();

		// KPPを学習させないなら、KKPのmaxまでだけで良い。
		if (without_kpp)
			vector_length = KKP::max_index();

		// epochに応じたetaを設定してやる。
		Weight::calc_eta(epoch);

		const auto zero = array<LearnFloatType, 2>{ 0, 0 };

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
			for (s64 index_ = 0; (u64)index_ < vector_length; ++index_)
			{
				// OpenMPではループ変数は符号型変数でなければならないが
				// さすがに使いにくい。
				u64 index = (u64)index_;

				// 自分が更新すべきやつか？
				// 次元下げしたときのindexの小さいほうが自分でないならこの更新は行わない。
				if (!min_index_flag[index])
					continue;

				if (KK::is_ok(index))
				{
					// KKは次元下げしていないので普通にupdate
					KK x = KK::fromIndex(index);
					weights[index].updateFV(kk[x.king0()][x.king1()]);
					weights[index].set_grad(zero);
				}
				else if (KKP::is_ok(index))
				{
					// KKPは次元下げがあるので..
					KKP x = KKP::fromIndex(index);
					KKP a[KKP_LOWER_COUNT];
					x.toLowerDimensions(/*out*/a);

					// a[0] == indexであることは保証されている。
#if KKP_LOWER_COUNT==2
					u64 ids[2] = { /*a[0].toIndex()*/ index , a[1].toIndex() };
					// 勾配を合計して、とりあえずa[0]に格納し、
					// それに基いてvの更新を行い、そのvをlowerDimensionsそれぞれに書き出す。
					// id[0]==id[1]==id[2]==id[3]みたいな可能性があるので、gは外部で合計する
					array<LearnFloatType, 2> g_sum = { 0,0 };
					for (auto id : ids)
						g_sum += weights[id].get_grad();

					// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
					if (is_zero(g_sum))
						continue;

					//cout << a[0].king0() << a[0].king1() << a[0].piece() << g_sum << endl;

					auto& v = kkp[a[0].king0()][a[0].king1()][a[0].piece()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					kkp[a[1].king0()][a[1].king1()][a[1].piece()] = v;

					// mirrorした場所が同じindexである可能性があるので、gのクリアはこのタイミングで行なう。
					// この場合、毎回gを通常の2倍加算していることになるが、AdaGradは適応型なのでこれでもうまく学習できる。
					for (auto id : ids)
						weights[id].set_grad(zero);

#elif KKP_LOWER_COUNT==4
					u64 ids1[2] = { /*a[0].toIndex()*/ index , a[1].toIndex() };
					u64 ids2[2] = { a[2].toIndex() ,a[3].toIndex() };

					// ids2はinverseしたものだから符号が逆になるので注意。
					array<LearnFloatType, 2> g_sum = { 0,0 };
					for (auto id : ids1) g_sum += weights[id].get_grad();
					for (auto id : ids2) g_sum -= weights[id].get_grad();

					// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
					if (is_zero(g_sum))
						continue;

					//cout << a[0].king0() << a[0].king1() << a[0].piece() << g_sum << endl;

					auto& v = kkp[a[0].king0()][a[0].king1()][a[0].piece()];
					weights[ids1[0]].set_grad(g_sum);
					weights[ids1[0]].updateFV(v);

					kkp[a[1].king0()][a[1].king1()][a[1].piece()] = v;
					kkp[a[2].king0()][a[2].king1()][a[2].piece()] = -v;
					kkp[a[3].king0()][a[3].king1()][a[3].piece()] = -v;

					for (auto id : ids1) weights[id].g = { 0,0 };
					for (auto id : ids2) weights[id].g = { 0,0 };

#endif

				}
				else if (KPP::is_ok(index))
				{
					KPP x = KPP::fromIndex(index);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					KPP a[4];
					x.toLowerDimensions(/*out*/a);
					u64 ids[4] = { /*a[0].toIndex()*/ index , a[1].toIndex() , a[2].toIndex() , a[3].toIndex() };
#else
					// 3角配列を用いる場合、次元下げは2つ。
					KPP a[2];
					x.toLowerDimensions(/*out*/a);
					u64 ids[2] = { /*a[0].toIndex()*/ index , a[1].toIndex() };
#endif
					array<LearnFloatType, 2> g_sum = { 0,0 };
					for (auto id : ids)
						g_sum += weights[id].get_grad();

					//// KPPの手番は動かさないとき。
					//g_sum[1] = 0;

					if (is_zero(g_sum))
						continue;

					//cout << a[0].king() << a[0].piece0() << a[0].piece1() << g_sum << endl;

					auto& v = kpp[a[0].king()][a[0].piece0()][a[0].piece1()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					for (int i = 1; i < 4; ++i)
						kpp[a[i].king()][a[i].piece0()][a[i].piece1()] = v;
#else
					// 三角配列の場合、KPP::toLowerDimensionsで、piece0とpiece1を入れ替えたものは返らないので
					// (同じindexを指しているので)、自分で入れ替えてkpp配列にvの値を反映させる。
					kpp[a[0].king()][a[0].piece1()][a[0].piece0()] = v;
					kpp[a[1].king()][a[1].piece0()][a[1].piece1()] = v;
					kpp[a[1].king()][a[1].piece1()][a[1].piece0()] = v;
#endif

					for (auto id : ids)
						weights[id].set_grad(zero);
				}
				else
				{
					ASSERT_LV3(false);
				}
			}
		}
	}

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
		{
			auto eval_dir = path_combine((string)Options["EvalSaveDir"], dir_name);

			cout << "save_eval() start. folder = " << eval_dir << endl;

			// すでにこのフォルダがあるならmkdir()に失敗するが、
			// 別にそれは構わない。なければ作って欲しいだけ。
			// また、EvalSaveDirまでのフォルダは掘ってあるものとする。

			MKDIR(eval_dir);

			// EvalIOを利用して評価関数ファイルに書き込む。
			// 読み込みのときのinputとoutputとを入れ替えるとファイルに書き込める。EvalIo::eval_convert()マジ優秀。
			auto make_name = [&](std::string filename) { return path_combine(eval_dir, filename); };
			auto input = EvalIO::EvalInfo::build_kppt32((void*)kk, (void*)kkp, (void*)kpp);
			auto output = EvalIO::EvalInfo::build_kppt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));

			// 評価関数の実験のためにfe_endをKPPT32から変更しているかも知れないので現在のfe_endの値をもとに書き込む。
			input.fe_end = output.fe_end = Eval::fe_end;

			if (!EvalIO::eval_convert(input, output, nullptr))
				goto Error;

			cout << "save_eval() finished. folder = " << eval_dir << endl;
			return;
		}

	Error:;
		cout << "Error : save_eval() failed" << endl;
	}

	// 現在のetaを取得する。
	double get_eta() {
		return Weight::eta;
	}

}

#endif // EVAL_LEARN
#endif // _EVALUATE_LEARN_KPPT_CPP_
