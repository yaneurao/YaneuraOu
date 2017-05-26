#ifndef _EVALUATE_KPPT_LEARN_CPP_
#define _EVALUATE_KPPT_LEARN_CPP_

// KPPT評価関数の学習時用のコード
// tanuki-さんの学習部のコードをかなり参考にさせていただきました。

#include "../shogi.h"

#if defined(EVAL_LEARN)

#include "learn.h"

#include "../evaluate.h"
#include "../eval/evaluate_kppt.h"
#include "../eval/kppt_evalsum.h"
#include "../position.h"
#include "../misc.h"

using namespace std;

// ----------------------
//  学習のときの浮動小数
// ----------------------

// これをdoubleにしたほうが計算精度は上がるが、重み配列絡みのメモリが倍必要になる。
// 現状、ここをfloatにした場合、評価関数ファイルに対して、重み配列はその4.5倍のサイズ。(KPPTで4.5GB程度)
// double型にしても収束の仕方にほとんど差異がなかったのでfloatに固定する。
typedef float LearnFloatType;


namespace Eval
{
	// 学習のときの勾配配列の初期化
	void init_grad(double eta);

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad);

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights(u64 epoch);

	// 学習のためのテーブルの初期化
	void eval_learn_init();

	// 評価関数パラメーターをファイルに保存する。
	void save_eval(std::string dir_name);

	// あるBonaPieceを相手側から見たときの値
	BonaPiece inv_piece[fe_end];

	// 盤面上のあるBonaPieceをミラーした位置にあるものを返す。
	BonaPiece mir_piece[fe_end];
}

// --- 以下、定義

namespace Eval
{
	// 勾配等を格納している学習用の配列
#if defined(_MSC_VER)
#pragma pack(push,2)
#elif defined(__GNUC__)
#pragma pack(2)
#endif
	struct Weight
	{
		// AdaGradの学習率η(eta)。
		// updateFV()が呼び出されるまでに設定されているものとする。
		static double eta;

		// mini-batch 1回分の勾配の累積値
		array<LearnFloatType,2> g;

		// AdaGradのg2
		array<LearnFloatType,2> g2;

		// vの小数部上位8bit。(vをfloatで持つのもったいないのでvの補助bitとして8bitで持つ)
		array<s8,2> v8;

		// 合計 4*2 + 4*2 + 1*2 = 18 bytes(LearnFloatType == floatのとき)
		// 1GBの評価関数パラメーターに対してその4.5倍のサイズのWeight配列が確保できれば良い。
		// ただし、構造体のアライメントが4バイト単位になっているとsizeof(Weight)==20なコードが生成されるので
		// pragma pack(2)を指定しておく。

		// AdaGradでupdateする
		// この関数を実行しているときにgの値やメンバーが書き変わらないことは
		// 呼び出し側で保証されている。atomic演算である必要はない。
		template <typename T>
		void updateFV(array<T,2>& v)
		{
			// AdaGradの更新式
			//   勾配ベクトルをg、更新したいベクトルをv、η(eta)は定数として、
			//     g2 = g2 + g^2
			//     v = v - ηg/sqrt(g2)

			constexpr double epsilon = 0.000001;
			for (int i = 0; i < 2; ++i)
			{
				if (g[i] == 0)
					continue;

				g2[i] += g[i] * g[i];

				// v8は小数部8bitを含んでいるのでこれを復元する。
				// 128倍にすると、-1を保持できなくなるので127倍にしておく。
				// -1.0～+1.0を-127～127で保持している。
				// std::round()限定なら-0.5～+0.5の範囲なので255倍でも良いが、
				// どんな丸め方をするかはわからないので余裕を持たせてある。

				double V = v[i] + ((double)v8[i] / 127);

				V -= eta * (double)g[i] / sqrt((double)g2[i] + epsilon);

				// Vの値をINT16の範囲に収まるように制約を課す。
				V = min((double)INT16_MAX * 3 / 4, V);
				V = max((double)INT16_MIN * 3 / 4, V);

				v[i] = (T)round(V);
				v8[i] = (s8)((V - v[i]) * 127);

				// この要素に関するmini-batchの1回分の更新が終わったのでgをクリア
				//g[i] = 0;
				// これは呼び出し側で行なうことにする。
			}
		}
	};
#if defined(_MSC_VER)
#pragma pack(pop)
#elif defined(__GNUC__)
#pragma pack(0)
#endif


	double Weight::eta;

	// --- 以下のKK,KKP,KPPは、Weight配列を直列化したときのindexを計算したりするヘルパー。

	struct KK
	{
		KK() {}
		KK(Square king0, Square king1) : king0_(king0), king1_(king1) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KKの最小値、最大値。
		static u64 min_index() { return 0;  }
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)SQ_NB; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKKのオブジェクトを生成するbuilder
		static KK fromIndex(u64 index)
		{
			index -= min_index();
			Square king1 = (Square)(index % SQ_NB);
			index /= SQ_NB;
			Square king0 = (Square)(index  /* % SQ_NB */ );
			ASSERT_LV3(king0 < SQ_NB);
			return KK(king0,king1);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king0() const { return king0_; }
		Square king1() const { return king1_; }

		// 低次元の配列のindexを得る。
		// KKはミラーの次元下げを行わないので、そのままの値。
		void toLowerDimensions(/*out*/KK kk_[1]) const {
			kk_[0] = KK(king0_, king1_);
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const {
			return min_index() + (u64)king0_ * (u64)SQ_NB + (u64)king1_;
		}

	private:
		Square king0_, king1_;
	};

	struct KKP
	{
		KKP() {}
		KKP(Square king0, Square king1,BonaPiece p) : king0_(king0), king1_(king1),piece_(p) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KKPの最小値、最大値。
		static u64 min_index() { return KK::max_index(); }
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKKPのオブジェクトを生成するbuilder
		static KKP fromIndex(u64 index)
		{
			index -= min_index();
			BonaPiece piece = (BonaPiece)(index % fe_end);
			index /= fe_end;
			Square king1 = (Square)(index % SQ_NB);
			index /= SQ_NB;
			Square king0 = (Square)(index  /* % SQ_NB */);
			ASSERT_LV3(king0 < SQ_NB);
			return KKP(king0, king1, piece);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king0() const { return king0_; }
		Square king1() const { return king1_; }
		BonaPiece piece() const { return piece_; }

		// 低次元の配列のindexを得る。ミラーしたものがkkp_[1]に返る。
		void toLowerDimensions(/*out*/ KKP kkp_[2]) const {
			kkp_[0] = KKP(king0_, king1_, piece_);
			kkp_[1] = KKP(Mir(king0_), Mir(king1_), mir_piece[piece_]);
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const {
			return min_index() + ((u64)king0_ * (u64)SQ_NB + (u64)king1_) * (u64)fe_end + (u64)piece_;
		}

	private:
		Square king0_, king1_;
		BonaPiece piece_;
	};

	struct KPP
	{
		KPP() {}
		KPP(Square king, BonaPiece p0, BonaPiece p1) : king_(king), piece0_(p0), piece1_(p1) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KPPの最小値、最大値。
		static u64 min_index() { return KKP::max_index(); }
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)fe_end*(u64)fe_end; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKPPのオブジェクトを生成するbuilder
		static KPP fromIndex(u64 index)
		{
			index -= min_index();
			BonaPiece piece1 = (BonaPiece)(index % fe_end);
			index /= fe_end;
			BonaPiece piece0 = (BonaPiece)(index % fe_end);
			index /= fe_end;
			Square king = (Square)(index  /* % SQ_NB */);
			ASSERT_LV3(king < SQ_NB);
			return KPP(king,piece0,piece1);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king() const { return king_; }
		BonaPiece piece0() const { return piece0_; }
		BonaPiece piece1() const { return piece1_; }

		// 低次元の配列のindexを得る。p1,p2を入れ替えたもの、ミラーしたものなどが返る。
		void toLowerDimensions(/*out*/ KPP kpp_[4]) const {
			kpp_[0] = KPP(king_, piece0_, piece1_);
			kpp_[1] = KPP(king_, piece1_, piece0_);
			kpp_[2] = KPP(Mir(king_) , mir_piece[piece0_], mir_piece[piece1_]);
			kpp_[3] = KPP(Mir(king_) , mir_piece[piece1_], mir_piece[piece0_]);
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const {
			return min_index() + ((u64)king_ * (u64)fe_end + (u64)piece0_) * (u64)fe_end + (u64)piece1_;
		}

	private:
		Square king_;
		BonaPiece piece0_ , piece1_;
	};

	// 評価関数学習用の構造体

	// KK,KKP,KPPのWeightを保持している配列
	// 直列化してあるので1次元配列
	std::vector<Weight> weights;

	// 次元下げしたときに、そのなかの一番小さなindexになることがわかっているindexに対してtrueとなっているフラグ配列。
	std::vector<bool> min_index_flag;


	// 学習のときの勾配配列の初期化
	// 引数のetaは、AdaGradのときの定数η(eta)。
	void init_grad(double eta)
	{
		// 学習用配列の確保
		u64 size = KPP::max_index();
		weights.resize(size); // 確保できるかは知らん。確保できる環境で動かしてちょうだい。
		memset(&weights[0], 0, sizeof(Weight) * size);

		// 次元下げ用フラグ配列の初期化
		min_index_flag.resize(size);
#pragma omp parallel for schedule(guided)
		for (u64 index = 0; index < size; ++index)
		{
			if (KK::is_ok(index))
			{
				min_index_flag[index] = true;
				// indexからの変換と逆変換によって元のindexに戻ることを確認しておく。
				// 起動時に1回しか実行しない処理なのでASSERT_LV1で書いておく。
				ASSERT_LV1(KK::fromIndex(index).toIndex() == index);
				// 次元下げの1つ目の要素が元のindexと同一であることを確認しておく。
				KK a[1];
				KK::fromIndex(index).toLowerDimensions(a);
				ASSERT_LV1(a[0].toIndex() == index);
			}
			else if (KKP::is_ok(index))
			{
				KKP x = KKP::fromIndex(index);
				KKP a[2];
				x.toLowerDimensions(a);
				u64 id[2] = { a[0].toIndex(),a[1].toIndex() };
				min_index_flag[index] = ( min({ id[0],id[1] }) == index );
				ASSERT_LV1(id[0] == index);
			}
			else if (KPP::is_ok(index))
			{
				KPP x = KPP::fromIndex(index);
				KPP a[4];
				x.toLowerDimensions(a);
				u64 id[4] = { a[0].toIndex() , a[1].toIndex() , a[2].toIndex() , a[3].toIndex() };
				min_index_flag[index] = ( min({ id[0],id[1],id[2],id[3] }) == index );
				ASSERT_LV1(KPP::fromIndex(index).toIndex() == index);
				ASSERT_LV1(id[0] == index);
			}
			else
			{
				ASSERT_LV3(false);
			}
		}

		// 学習率の設定
		if (eta != 0)
			Weight::eta = eta;
		else
			Weight::eta = 30.0; // default値
	}

	// 現在の局面で出現している特徴すべてに対して、勾配値を勾配配列に加算する。
	// 現局面は、leaf nodeであるものとする。
	void add_grad(Position& pos, Color rootColor, double delta_grad)
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
			(rootColor == BLACK) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad),

			// 手番を考慮する値
			(rootColor == pos.side_to_move()) ? LearnFloatType(delta_grad) : -LearnFloatType(delta_grad)
		};

		// 180度盤面を回転させた位置関係に対する勾配
		array<LearnFloatType,2> g_flip = { -g[0] , +g[1] };

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);

		auto& pos_ = *const_cast<Position*>(&pos);

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

		// KK
		weights[KK(sq_bk,sq_wk).toIndex()].g += g;

		// flipした位置関係にも書き込む
		//kk_w[Inv(sq_wk)][Inv(sq_bk)].g += g_flip;

		for (int i = 0; i < PIECE_NO_KING; ++i)
		{
			BonaPiece k0 = list_fb[i];
			BonaPiece k1 = list_fw[i];

			// このループではk0 == l0は出現しない。(させない)
			// それはKPであり、KKPの計算に含まれると考えられるから。
			for (int j = 0; j < i; ++j)
			{
				BonaPiece l0 = list_fb[j];
				BonaPiece l1 = list_fw[j];

				weights[KPP(sq_bk, k0, l0).toIndex()].g += g;
				weights[KPP(Inv(sq_wk), k1, l1).toIndex()].g += g_flip;
			}

			// KKP
			weights[KKP(sq_bk, sq_wk, k0).toIndex()].g += g;
		}
	}

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights(u64 epoch)
	{
		u64 vector_length = KPP::max_index();

		// 並列化を効かせたいので直列化されたWeight配列に対してループを回す。
#pragma omp parallel for schedule(guided)
		for (u64 index = 0; index < vector_length; ++index)
		{
			// 自分が更新すべきやつか？
			// 次元下げしたときのindexの小さいほうが自分でないならこの更新は行わない。
			if (!min_index_flag[index])
				continue;

			if (KK::is_ok(index))
			{
				// KKは次元下げしていないので普通にupdate
				KK x = KK::fromIndex(index);
				weights[index].updateFV(kk[x.king0()][x.king1()]);
				weights[index].g = { 0,0 };
			}
			else if (KKP::is_ok(index))
			{
				// KKPは次元下げがあるので..
				KKP x = KKP::fromIndex(index);
				KKP a[2];
				x.toLowerDimensions(/*out*/a);

				// a[0] == indexであることは保証されている。
				u64 ids[2] = { /*a[0].toIndex()*/ index , a[1].toIndex() };

				// 勾配を合計して、とりあえずa[0]に格納し、
				// それに基いてvの更新を行い、そのvをlowerDimensionsそれぞれに書き出す。
				// id[0]==id[1]==id[2]==id[3]みたいな可能性があるので、gは外部で合計する
				array<LearnFloatType, 2> g_sum = { 0,0 };
				for (auto id : ids)
					g_sum += weights[id].g;

				// 次元下げを考慮して、その勾配の合計が0であるなら、一切の更新をする必要はない。
				if (is_zero(g_sum))
					continue;

				//cout << a[0].king0() << a[0].king1() << a[0].piece() << g_sum << endl;

				auto& v = kkp[a[0].king0()][a[0].king1()][a[0].piece()];
				weights[ids[0]].g = g_sum;
				weights[ids[0]].updateFV(v);

				kkp[a[1].king0()][a[1].king1()][a[1].piece()] = v;

				// mirrorした場所が同じindexである可能性があるので、gのクリアはこのタイミングで行なう。
				// この場合、毎回gを通常の2倍加算していることになるが、AdaGradは適応型なのでこれでもうまく学習できる。
				for (auto id : ids)
					weights[id].g = { 0,0 };
			}
			else if (KPP::is_ok(index))
			{
				KPP x = KPP::fromIndex(index);
				KPP a[4];
				x.toLowerDimensions(/*out*/a);

				u64 ids[4] = { /*a[0].toIndex()*/ index , a[1].toIndex() , a[2].toIndex() , a[3].toIndex() };

				array<LearnFloatType, 2> g_sum = { 0,0 };
				for (auto id : ids)
					g_sum += weights[id].g;

				if (is_zero(g_sum))
					continue;

				//cout << a[0].king() << a[0].piece0() << a[0].piece1() << g_sum << endl;

				auto& v = kpp[a[0].king()][a[0].piece0()][a[0].piece1()];
				weights[ids[0]].g = g_sum;
				weights[ids[0]].updateFV(v);

				for (int i = 1; i<4; ++i)
					kpp[a[i].king()][a[i].piece0()][a[i].piece1()] = v;

				for (auto id : ids)
					weights[id].g = { 0 , 0 };
			}
			else
			{
				ASSERT_LV3(false);
			}
		}
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
			inv_piece[p] = BONA_PIECE_NOT_INIT;

			// mirrorは手駒に対しては機能しない。元の値を返すだけ。
			mir_piece[p] = (p < f_pawn) ? p : BONA_PIECE_NOT_INIT;
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
			if (inv_piece[p] == BONA_PIECE_NOT_INIT
				|| mir_piece[p] == BONA_PIECE_NOT_INIT
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

			// KK
			std::ofstream ofsKK(path_combine(eval_dir, KK_BIN), std::ios::binary);
			if (!ofsKK.write(reinterpret_cast<char*>(kk), sizeof(kk)))
				goto Error;

			// KKP
			std::ofstream ofsKKP(path_combine(eval_dir, KKP_BIN), std::ios::binary);
			if (!ofsKKP.write(reinterpret_cast<char*>(kkp), sizeof(kkp)))
				goto Error;

			// KPP
			std::ofstream ofsKPP(path_combine(eval_dir, KPP_BIN), std::ios::binary);
			if (!ofsKPP.write(reinterpret_cast<char*>(kpp), sizeof(kpp)))
				goto Error;

			cout << "save_eval() finished. folder = " << eval_dir << endl;

			return;
		}

	Error:;
		cout << "Error : save_eval() failed" << endl;
	}

}

#endif // EVAL_LEARN
#endif // _EVALUATE_LEARN_CPP_
