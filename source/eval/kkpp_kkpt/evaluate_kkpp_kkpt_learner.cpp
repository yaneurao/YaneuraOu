// KKPP_KKPT評価関数の学習時用のコード

#include "../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_KKPP_KKPT)

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

#include "evaluate_kkpp_kkpt.h"

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

			const ValueKkpp kkpp_zero = 0;
			for (int sq = 0; sq < KKPP_EVAL_KING_SQ ; ++sq)
			{
				for (auto p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					sum += abs(kkpp_ksq_pcpc(sq, p, p));
					kkpp_ksq_pcpc(sq, p, p) = kkpp_zero;
				}
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
					kpp_ksq_pcpc(sq,p1,BONA_PIECE_ZERO)  = kpp_zero;
					kpp_ksq_pcpc(sq, BONA_PIECE_ZERO,p1) = kpp_zero;
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

	// KKPPも手番なしなので手番なし用の1次元配列。
	std::vector<Weight> weights_kkpp;

	// 学習配列のKK,KKP,KPP,KKPPの内部デザイン。
	namespace {
		KK g_kk;
		KKP g_kkp;
		KPP g_kpp;
		KKPP g_kkpp;
	}

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// bugなどにより誤って書き込まれた値を補正する。
		correct_eval();

		// 学習で使用するテーブル類の初期化
		EvalLearningTools::init();

		// 学習配列のKK,KKP,KPP,KKPPの内部デザイン。
		g_kk.set(SQ_NB, Eval::fe_end, 0);
		g_kkp.set(SQ_NB, Eval::fe_end, g_kk.max_index());
		g_kpp.set(SQ_NB, Eval::fe_end, g_kkp.max_index());
		g_kkpp.set(KKPP_LEARN_KING_SQ, Eval::fe_end, g_kpp.max_index());

		// 学習用配列の確保
		u64 size = g_kkp.max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

		u64 size_kpp = g_kpp.size();
		weights_kpp.resize(size_kpp);
		memset(&weights_kpp[0], 0, sizeof(Weight) * weights_kpp.size());

		u64 size_kkpp = g_kkpp.size();
		weights_kkpp.resize(size_kkpp);
		memset(&weights_kkpp[0], 0, sizeof(Weight) * weights_kkpp.size());

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);

	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad , const std::array<bool, 4>& freeze)
	{
		const bool freeze_kpp = freeze[2];
		const bool freeze_kkpp = freeze[3];

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

		// KKPPのことを考慮しないといけない。

		int encoded_eval_kk = encode_to_eval_kk(sq_bk, sq_wk);
		if (encoded_eval_kk != -1)
		{
			// KKPPで計算する(先手玉・後手玉の位置的にKKPPで計算できる状況)

			// バッファを確保してコピー。(どうせsortが必要なので、そのためにはコピーしておかなければならない)
			// なお、list_fwは用いないので不要。
			BonaPiece list_fb[PIECE_NUMBER_KING];
			memcpy(list_fb, pos.eval_list()->piece_list_fb() , sizeof(BonaPiece) * (int)PIECE_NUMBER_KING);

			// ただし、file_of(sq_bk) > FILE_5なら、ここでミラーしてから、学習配列の勾配を加算。
			if (file_of(sq_bk) > FILE_5)
			{
				sq_bk = Mir(sq_bk);
				sq_wk = Mir(sq_wk);

				for (int i = 0 ; i < PIECE_NUMBER_KING; ++i)
					list_fb[i] = mir_piece(list_fb[i]);
			}

			// KKPPの学習配列は、piece0 > piece1でなければならないので、ここで
			// piece_listのsortしておく。

			my_insertion_sort(list_fb, 0, PIECE_NUMBER_KING);

			int encoded_learn_kk = encode_to_learn_kk(sq_bk, sq_wk);
			ASSERT_LV3(encoded_learn_kk != -1);

			// 準備は整った。以下、compute_eval()と似た感じの構成。

			// KK
			weights[g_kk.fromKK(sq_bk, sq_wk).toIndex()].add_grad(g);

			for (int i = 0; i < PIECE_NUMBER_KING; ++i)
			{
				BonaPiece k0 = list_fb[i];

				// KKPP
				if (!freeze_kkpp)
					for (int j = 0; j < i; ++j)
					{
						BonaPiece l0 = list_fb[j];
						weights_kkpp[g_kkpp.fromKKPP(encoded_learn_kk, k0, l0).toRawIndex()].add_grad(g[0]);
					}

				// KKP
				weights[g_kkp.fromKKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
			}

			// 上記のループ、AVX2を使わないコードだとすこぶる明快。

		}
		else {

			// 通常のKPP

			auto list_fb = pos.eval_list()->piece_list_fb();
			auto list_fw = pos.eval_list()->piece_list_fw();

			// KK
			weights[g_kk.fromKK(sq_bk, sq_wk).toIndex()].add_grad(g);

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

						weights_kpp[g_kpp.fromKPP(    sq_bk , k0, l0).toRawIndex()].add_grad(g[0]);
						weights_kpp[g_kpp.fromKPP(Inv(sq_wk), k1, l1).toRawIndex()].add_grad(g_flip[0]);
					}
				}

				// KKP
				weights[g_kkp.fromKKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
			}
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	// epoch       : 世代カウンター(0から始まる)
	void update_weights(u64 epoch , const std::array<bool, 4>& freeze)
	{
		u64 vector_length = g_kkpp.max_index();

		const bool freeze_kk = freeze[0];
		const bool freeze_kkp = freeze[1];
		const bool freeze_kpp = freeze[2];
		const bool freeze_kkpp = freeze[3];

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
				// ただし、KKPP配列に関しては無条件でokである。
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
					for (int i = 0; i < KKP_LOWER_COUNT; ++i)
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

					auto& v = kpp_ksq_pcpc(a[0].king(), a[0].piece0(), a[0].piece1());
					weights_kpp[ids[0]].set_grad(g_sum);
					weights_kpp[ids[0]].updateFV(v);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
					for (int i = 1; i < KPP_LOWER_COUNT; ++i)
						kpp_ksq_pcpc(a[i].king(), a[i].piece0(), a[i].piece1()) = v;
#else
					// 三角配列の場合、KPP::toLowerDimensionsで、piece0とpiece1を入れ替えたものは返らないので
					// (同じindexを指しているので)、自分で入れ替えてkpp配列にvの値を反映させる。
					kpp_ksq_pcpc(a[0].king(), a[0].piece1(), a[0].piece0()) = v;
#if KPP_LOWER_COUNT == 2
					kpp_ksq_pcpc(a[1].king(), a[1].piece0(), a[1].piece1()) = v;
					kpp_ksq_pcpc(a[1].king(), a[1].piece1(), a[1].piece0()) = v;
#endif
#endif

					for (auto id : ids)
						weights_kpp[id].set_grad(zero);
				}
				else if (g_kkpp.is_ok(index) && !freeze_kkpp)
				{
					// ミラーの次元下げとpiece0(),piece1()の入れ替えは学習配列に勾配を加算するときに行われているとみなせるので
					// ここでは次元下げをする必要がない。

					u64 raw_index = index - g_kkpp.min_index();
					LearnFloatType g_sum = weights_kkpp[raw_index].get_grad();

					if (g_sum == 0)
						continue;

					KKPP x = g_kkpp.fromRawIndex(raw_index);

					Square bk, wk;
					decode_from_learn_kk(x.king(), bk, wk);
					int encoded_eval_kk = encode_to_eval_kk(bk, wk);
					ASSERT_LV3(encoded_eval_kk != -1);

					auto& v = kkpp_ksq_pcpc(encoded_eval_kk, x.piece0(), x.piece1());
					weights_kkpp[raw_index].updateFV(v);

					// 自前で、ミラーしたところと、piece0(),piece1()を入れ替えたところに書き出す
					int mir_encoded_eval_kk = encode_to_eval_kk(Mir(bk),Mir(wk));
					ASSERT_LV3(mir_encoded_eval_kk != -1);

					kkpp_ksq_pcpc(    encoded_eval_kk ,           x.piece1() ,           x.piece0() ) = v;
					kkpp_ksq_pcpc(mir_encoded_eval_kk , mir_piece(x.piece0()), mir_piece(x.piece1())) = v;
					kkpp_ksq_pcpc(mir_encoded_eval_kk , mir_piece(x.piece1()), mir_piece(x.piece0())) = v;

					// inverseしたところには書き出さない。
					// inverseの次元下げ入れたほうが良いかも知れない…。

					weights_kkpp[raw_index].set_grad(zero);
				}
			}
		}
	}

	// SkipLoadingEval trueにしてKPP_KKPT型の評価関数を読み込む。このときkkpp配列はゼロで埋められた状態になる。
	// この状態から、kpp配列をkkpp配列に展開するコード。
	// save_eval()の手前でこの関数を呼びたすようにして、ビルドしたのちに、
	// isreadyコマンドで評価関数ファイルを読み込み、"test evalsave"コマンドでsave_eval()を呼び出すと変換したものが書き出されるという考え。
	// 誰もが使えたほうが良いのかも知れないがコマンド型にするのが面倒なのでとりあえずこれで凌ぐ。
	/*
	変換時のコマンド例)
		EvalDir rezero_kpp_kkpt_epoch5
		SkipLoadingEval true
		isready
		EvalSaveDir rezero_kkpp_kkpt_epoch5
		test evalsave
	*/
	void expand_kpp_to_kkpp()
	{
		EvalLearningTools::init_mir_inv_tables();

		std::cout << "expand_kpp_to_kkpp..";

		for (auto bk : SQ) // 先手玉
			for (auto wk : SQ) // 後手玉
			{
				int k = encode_to_eval_kk((Square)bk, (Square)wk);
				if (k == -1)
					continue;

				for (auto p0 = BONA_PIECE_ZERO; p0 < Eval::fe_end; ++p0)
					for (auto p1 = BONA_PIECE_ZERO; p1 < Eval::fe_end; ++p1)
					{
						int bkpp = kpp_ksq_pcpc(    bk ,           p0 ,           p1 );
						int wkpp = kpp_ksq_pcpc(Inv(wk), inv_piece(p0), inv_piece(p1));

						// これを合わせたものがkkppテーブルに書き込まれるべき。
						kkpp_ksq_pcpc(k, p0, p1) = bkpp - wkpp;
					}
			}

		std::cout << "done." << std::endl;
	}

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
		// KPP_KKPT型の評価関数から変換するとき。
		//expand_kpp_to_kkpp();

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
			auto input = EvalIO::EvalInfo::build_kkpp_kkpt32((void*)kk, (void*)kkp, (void*)kpp, (void*)kkpp, size_of_kkpp);
			auto output = EvalIO::EvalInfo::build_kkpp_kkpt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN), make_name(KKPP_BIN), size_of_kkpp);

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
