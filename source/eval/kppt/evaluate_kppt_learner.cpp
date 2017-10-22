// KPPT評価関数の学習時用のコード
// tanuki-さんの学習部のコードをかなり参考にさせていただきました。

#include "../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_KPPT)

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

#include "evaluate_kppt.h"

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

	// KK,KKP,KPPのWeightを保持している配列
	// 直列化してあるので1次元配列
	std::vector<Weight2> weights;

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
		u64 size = g_kpp.max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

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
		weights[g_kk.fromKK(sq_bk,sq_wk).toIndex()].add_grad(g);

		for (int i = 0; i < PIECE_NUMBER_KING; ++i)
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

					weights[g_kpp.fromKPP(sq_bk     , k0, l0).toIndex()].add_grad(g);
					weights[g_kpp.fromKPP(Inv(sq_wk), k1, l1).toIndex()].add_grad(g_flip);
				}
			}

			// KKP
			weights[g_kkp.fromKKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	void update_weights(u64 epoch, const std::array<bool, 4>& freeze)
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

		// 手番つきのゼロ
		const auto zero_t = std::array<LearnFloatType, 2>{ 0, 0 };

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
				// OpenMPではループ変数は符号型変数でなければならないが、さすがに使いにくい。
				// ※　この制限はOpenMP 2.5までの制限で、OpenMP 3.0では解除されている。
				//   Visual C++ 2017はOpenMP 3.0に対応していない。PPLを推奨している模様。
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
					for (int i = 0; i <KK_LOWER_COUNT; ++i)
						g_sum += a[i].apply_inverse_sign(weights[ids[i]].get_grad());
					
					// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
					if (is_zero(g_sum))
						continue;

					auto& v = kk[a[0].king0()][a[0].king1()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

					for (int i = 1; i< KK_LOWER_COUNT; ++i)
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
						ids[i] = a[i].toIndex();

					// KPPに関してはinverseの次元下げがないので、inverseの判定は不要。

					std::array<LearnFloatType, 2> g_sum = zero_t;
					for (auto id : ids)
						g_sum += weights[id].get_grad();

					if (is_zero(g_sum))
						continue;

					auto& v = kpp[a[0].king()][a[0].piece0()][a[0].piece1()];
					weights[ids[0]].set_grad(g_sum);
					weights[ids[0]].updateFV(v);

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
						weights[id].set_grad(zero_t);
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
			auto input = EvalIO::EvalInfo::build_kppt32((void*)kk, (void*)kkp, (void*)kpp);
			auto output = EvalIO::EvalInfo::build_kppt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));

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
