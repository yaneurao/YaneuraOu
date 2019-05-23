// KPP_KKPT評価関数の学習時用のコード
// KPPTの学習用コードのコピペから少しいじった感じ。

#include "../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_HELICES)

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../learn/learn.h"
#include "../../learn/learning_tools.h"

#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"

#include "../evaluate_io.h"
#include "evaluate_helices.h"

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
					sum += abs(kpp[sq][p][p]);
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

			const ValueKpp kpp_zero = 0;
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

	// KPPは手番なしなので手番なし用の1次元配列。
	std::vector<Weight> weights_kpp;

	// KPPPも手番なしなので手番なし用の1次元配列。
	// 手番ありに変更するかも知れないので、配列をkppと分けておく。
	std::vector<Weight> weights_kppp;

	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta1, u64 eta1_epoch, double eta2, u64 eta2_epoch, double eta3)
	{
		// bugなどにより間違って書き込まれた値を補正する。
		correct_eval();

		// 学習で使用するテーブル類の初期化

		KPPP g_kppp(Eval::KPPP_KING_SQ,Eval::kppp_fe_end);
		ASSERT(g_kppp.get_triangle_fe_end() == Eval::kppp_triangle_fe_end);
		EvalLearningTools::init();
			
		// 学習用配列の確保

		// KK,KKP
		u64 size = KKP::max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight2) * weights.size());

		// KPP
		u64 size_kpp = KPP::max_index() - KPP::min_index();

		// 三角配列を用いる場合、このassert式は成り立たない…。
		//ASSERT_LV1(size_kpp == size_of_kpp / sizeof(ValueKpp));

		weights_kpp.resize(size_kpp);
		memset(&weights_kpp[0], 0, sizeof(Weight) * weights_kpp.size());

		// KPPP
		u64 size_kppp = g_kppp.max_index() - g_kppp.min_index();
		ASSERT_LV1(size_kppp == Eval::kppp_triangle_fe_end * (u64)KPPP_KING_SQ );
		weights_kppp.resize(size_kppp);
		memset(&weights_kppp[0], 0, sizeof(Weight) * weights_kppp.size());

		// 学習率の設定
		Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);

	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad, const std::array<bool, 4>& freeze)
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
		std::array<LearnFloatType, 2> g =
		{
			// 手番を考慮しない値
			(rootColor == BLACK) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad),

			// 手番を考慮する値
			(rootColor == pos.side_to_move()) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad)
		};

		// 180度盤面を回転させた位置関係に対する勾配
		std::array<LearnFloatType, 2> g_flip = { -g[0] , g[1] };

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);
		Square sq_wk_inv = Inv(sq_wk);

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
		weights[KK(sq_bk, sq_wk).toIndex()].add_grad(g);

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

					weights_kpp[KPP(sq_bk    , k0, l0).toRawIndex()].add_grad(g[0]);
					weights_kpp[KPP(sq_wk_inv, k1, l1).toRawIndex()].add_grad(g_flip[0]);
				}
			}

			// KKP
			weights[KKP(sq_bk, sq_wk, k0).toIndex()].add_grad(g);
		}

		// KPPP

		// 桂・銀・金のBonaPieceを0～kppp_fe_end-1にmappingして、Σkpppを計算する。
		const int kppp_piece_number_nb = KPPP_PIECE_NUMBER_NB;
		BonaPiece kppp_list_fb[kppp_piece_number_nb];
		BonaPiece kppp_list_fw[kppp_piece_number_nb];
		for (int i = 0; i < kppp_piece_number_nb; ++i)
		{
			// PIECE_NUMBER_KNIGHTからPIECE_NUMBER_SILVER , PIECE_NUMBER_GOLDがあるはず。
			kppp_list_fb[i] = to_kppp_bona_piece(list_fb[i + PIECE_NUMBER_KNIGHT]);
			kppp_list_fw[i] = to_kppp_bona_piece(list_fw[i + PIECE_NUMBER_KNIGHT]);
		}

		// KPPPクラスを用いるときに昇順ソートされていなければならない。
		my_insertion_sort(kppp_list_fb, 0, kppp_piece_number_nb);
		my_insertion_sort(kppp_list_fw, 0, kppp_piece_number_nb);

		// KPPPの3つのPであるp0,p1,p2のfb(先手から見たもの)、fw(後手から見たもの)を
		// p0_fb,p0_fw,p1_fb,p1_fw,p2_fb,f2_fwのように名前をつける。

		KPPP g_kppp(KPPP_KING_SQ,kppp_fe_end);

		// 桂・銀・金 = 12駒
		for (int i = 2; i < kppp_piece_number_nb; ++i)
		{
			auto p0_fb = kppp_list_fb[i];
			auto p0_fw = kppp_list_fw[i];

			for (int j = 1; j < i; ++j)
			{
				auto p1_fb = kppp_list_fb[j];
				auto p1_fw = kppp_list_fw[j];

				for (int k = 0; k < j; ++k)
				{
					auto p2_fb = kppp_list_fb[k];
					auto p2_fw = kppp_list_fw[k];

					weights_kppp[g_kppp.fromKPPP(sq_bk    , p0_fb, p1_fb, p2_fb).toRawIndex()].add_grad(g[0]);
					weights_kppp[g_kppp.fromKPPP(sq_wk_inv, p0_fw, p1_fw, p2_fw).toRawIndex()].add_grad(g_flip[0]);
				}
			}
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
		KPPP g_kppp(Eval::KPPP_KING_SQ,Eval::kppp_fe_end);
		u64 vector_length = g_kppp.max_index();

		const bool freeze_kk = freeze[0];
		const bool freeze_kkp = freeze[1];
		const bool freeze_kpp = freeze[2];
		const bool freeze_kppp = freeze[3];
			
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
				// ただし、KPPP配列に関しては無条件でokである。
				// "index < KPPP::min_index()"のほうを先に書かないと、"min_index_flag[index]"のindexが配列の範囲外となる。

				if (index < KPP::max_index() && !min_index_flag[index])
					continue;

				if (KK::is_ok(index) && !freeze_kk)
				{
					KK x = KK::fromIndex(index);

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
				else if (KKP::is_ok(index) && !freeze_kkp)
				{
					// KKの処理と同様

					KKP x = KKP::fromIndex(index);

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
				else if (KPP::is_ok(index) && !freeze_kpp)
				{
					KPP x = KPP::fromIndex(index);

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
						weights_kpp[id].set_grad(zero);
				}
				else if (g_kppp.is_ok(index) && !freeze_kppp)
				{

					// このコード、汎用的ではあるが、遅い。
					KPPP x = g_kppp.fromIndex(index);

					// ミラーしたものが得られないので(KPPPのPの値がKPPとかのBonaPieceと異なるため、ミラーがmir_pieceで出来ないので)、自前でミラーの次元下げを敢行する。
					// 6筋～9筋ならupdateをskipして良い。(1～5筋のmirrorのときに書き出されるはず)
					if (file_of((Square)x.king()) >= FILE_6)
						continue;

					const int MY_KPPP_LOWER_COUNT = 2;
					KPPP a[MY_KPPP_LOWER_COUNT];
					// x.toLowerDimensions(/*out*/a);

					a[0] = x;

					// ミラーしてsortする。細かいことだが、おそらくpiece0 > piece1 > piece2なはずなので、
					// piece2 , piece1 , piece0の順でp_listを作りsortしたほうがsortの並べ替えの回数が少なくなるはず。
					BonaPiece p_list[3] = { kppp_mir_piece(x.piece2()) ,kppp_mir_piece(x.piece1()), kppp_mir_piece(x.piece0()) };
					my_insertion_sort(p_list, 0, 3);
					a[1] = g_kppp.fromKPPP(Mir((Square)x.king()), p_list[2], p_list[1], p_list[0]);

					u64 ids[MY_KPPP_LOWER_COUNT];
					for (int i = 0; i < MY_KPPP_LOWER_COUNT; ++i)
						ids[i] = a[i].toRawIndex();

					LearnFloatType g_sum = zero;
					for (auto id : ids)
						g_sum += weights_kppp[id].get_grad();

					if (g_sum == 0)
						continue;

					auto& v = kppp_ksq_pcpcpc(a[0].king(),a[0].piece0(),a[0].piece1(),a[0].piece2());
					weights_kppp[ids[0]].set_grad(g_sum);
					weights_kppp[ids[0]].updateFV(v);

					for (int i = 0; i < MY_KPPP_LOWER_COUNT; ++i)
					{
						// 正方配列なのでp0,p1,p2を入れ替えた場所にも書き出す。
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece0(), a[i].piece1(), a[i].piece2()) = v;
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece0(), a[i].piece2(), a[i].piece1()) = v;
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece1(), a[i].piece0(), a[i].piece2()) = v;
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece1(), a[i].piece2(), a[i].piece0()) = v;
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece2(), a[i].piece0(), a[i].piece1()) = v;
						kppp_ksq_pcpcpc(a[i].king(), a[i].piece2(), a[i].piece1(), a[i].piece0()) = v;
					}

					for (auto id : ids)
						weights_kppp[id].set_grad(zero);
				}
			}
		}
	}

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name)
	{
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
			auto input = EvalIO::EvalInfo::build_kppp_kkpt32((void*)kk, (void*)kkp, (void*)kpp , (void*)kppp , size_of_kppp);
			auto output = EvalIO::EvalInfo::build_kppp_kkpt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN), make_name(KPPP_BIN) , size_of_kppp);

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
