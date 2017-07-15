﻿#ifndef _LEARN_H_
#define _LEARN_H_

#include "../shogi.h"

#if defined(EVAL_LEARN)

// =====================
//  学習時の設定
// =====================

// 以下のいずれかを選択すれば、そのあとの細々したものは自動的に選択される。
// いずれも選択しない場合は、そのあとの細々したものをひとつひとつ設定する必要がある。

// elmo方式での学習設定。これをデフォルト設定とする。
// 標準の雑巾絞りにするためにはlearnコマンドで "lambda 1"を指定してやれば良い。
#define LEARN_ELMO_METHOD

// やねうら王2017GOKU用のデフォルトの学習設定
// ※　このオプションは実験中なので使わないように。
// #define LEARN_YANEURAOU_2017_GOKU


// ----------------------
//        更新式
// ----------------------

// AdaGrad。これが安定しているのでお勧め。
// #define ADA_GRAD_UPDATE

// 勾配の符号だけ見るSGD。省メモリで済むが精度は…。
// #define SGD_UPDATE

// RMSProp風のAdaGrad
// #define ADA_PROP_UPDATE


// ----------------------
//    学習時の設定
// ----------------------

// mini-batchサイズ。
// この数だけの局面をまとめて勾配を計算する。
// 小さくするとupdate_weights()の回数が増えるので収束が速くなる。勾配が不正確になる。
// 大きくするとupdate_weights()の回数が減るので収束が遅くなる。勾配は正確に出るようになる。
// 多くの場合において、この値を変更する必要はないと思う。

#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)

// ファイルから1回に読み込む局面数。これだけ読み込んだあとshuffleする。
// ある程度大きいほうが良いが、この数×34byte×3倍ぐらいのメモリを消費する。10M局面なら340MB*3程度消費する。
// THREAD_BUFFER_SIZE(=10000)の倍数にすること。

#define LEARN_SFEN_READ_SIZE (1000 * 1000 * 10)

// 学習時の評価関数の保存間隔。この局面数だけ学習させるごとに保存。
// 当然ながら、保存間隔を長くしたほうが学習時間は短くなる。
// フォルダ名は 0/ , 1/ , 2/ ...のように保存ごとにインクリメントされていく。
// デフォルトでは10億局面に1回。
#define LEARN_EVAL_SAVE_INTERVAL (1000000000ULL)


// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
// 詳しい説明は、learner.cppを見ること。
// いわゆる、普通の「雑巾絞り」
//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY

// 目的関数が交差エントロピーだが、勝率の関数を通さない版
// #define LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE

// elmo(WCSC27)の方式
// #define LOSS_FUNCTION_IS_ELMO_METHOD

// ※　他、色々追加するかも。


// ----------------------
// 学習に関するデバッグ設定
// ----------------------

// 学習時のrmseの出力をこの回数に1回に減らす。
// rmseの計算は1スレッドで行なうためそこそこ時間をとられるので出力を減らすと効果がある。
#define LEARN_RMSE_OUTPUT_INTERVAL 1


// ----------------------
// ゼロベクトルからの学習
// ----------------------

// 評価関数パラメーターをゼロベクトルから学習を開始する。
// ゼロ初期化して棋譜生成してゼロベクトルから学習させて、
// 棋譜生成→学習を繰り返すとプロの棋譜に依らないパラメーターが得られる。(かも)
// (すごく時間かかる)

//#define RESET_TO_ZERO_VECTOR


// ----------------------
//  学習のときの浮動小数
// ----------------------

// これをdoubleにしたほうが計算精度は上がるが、重み配列絡みのメモリが倍必要になる。
// 現状、ここをfloatにした場合、評価関数ファイルに対して、重み配列はその4.5倍のサイズ。(KPPTで4.5GB程度)
// double型にしても収束の仕方にほとんど差異がなかったのでfloatに固定する。

// floatを使う場合
typedef float LearnFloatType;

// doubleを使う場合
//typedef double LearnFloatType;

// float16を使う場合
//#include "half_float.h"
//typedef HalfFloat::float16 LearnFloatType;


// ----------------------
//  省メモリ化
// ----------------------

// Weight配列(のうちのKPP)に三角配列を用いて省メモリ化する。
// これを用いると、学習用の重み配列は評価関数ファイルの2.5倍程度で済むようになる。

#define USE_TRIANGLE_WEIGHT_ARRAY


// ======================
//  教師局面生成時の設定
// ======================

// ----------------------
//  引き分けを書き出す
// ----------------------

// 引き分けに至ったとき、それを教師局面として書き出す
// これをするほうが良いかどうかは微妙。
// #define LEARN_GENSFEN_USE_DRAW_RESULT


// ======================
//       configure
// ======================

// ----------------------
//  elmo(WCSC27)の方法での学習
// ----------------------

#if defined( LEARN_ELMO_METHOD )
#define LOSS_FUNCTION_IS_ELMO_METHOD
#define ADA_GRAD_UPDATE
#endif

// ----------------------
//  やねうら王2017GOKUの方法
// ----------------------

#if defined(LEARN_YANEURAOU_2017_GOKU)

// 損失関数、比較実験中。
//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY
//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE
#define LOSS_FUNCTION_IS_ELMO_METHOD
//#define LOSS_FUNCTION_IS_YANE_ELMO_METHOD

#define ADA_GRAD_UPDATE
//#define SGD_UPDATE
//#define ADA_PROP_UPDATE
#endif


// ----------------------
// Learnerで用いるstructの定義
// ----------------------
#include "../position.h"

namespace Learner
{
	// PackedSfenと評価値が一体化した構造体
	// オプションごとに書き出す内容が異なると教師棋譜を再利用するときに困るので
	// とりあえず、以下のメンバーはオプションによらずすべて書き出しておく。
	struct PackedSfenValue
	{
		// 局面
		PackedSfen sfen;

		// Learner::search()から返ってきた評価値
		s16 score;

		// PVの初手
		u16 move;

		// 初期局面からの局面の手数。
		u16 gamePly;

		// この局面の手番側が、ゲームを最終的に勝っているなら1。負けているなら-1。
		// 引き分けに至った場合は、0。
		// 引き分けは、教師局面生成コマンドgensfenにおいて、
		// LEARN_GENSFEN_DRAW_RESULTが有効なときにだけ書き出す。
		s8 game_result;

		// 教師局面を書き出したファイルを他の人とやりとりするときに
		// この構造体サイズが不定だと困るため、paddingしてどの環境でも必ず40bytesになるようにしておく。
		u8 padding;

		// 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
	};

	// 読み筋とそのときの評価値を返す型
	// Learner::search() , Learner::qsearch()で用いる。
	typedef std::pair<Value, std::vector<Move> > ValueAndPV;

	// いまのところ、やねうら王2017Earlyしか、このスタブを持っていないが
	// EVAL_LEARNをdefineするなら、このスタブが必須。
	extern Learner::ValueAndPV  search(Position& pos, int depth , size_t multiPV = 1);
	extern Learner::ValueAndPV qsearch(Position& pos);

}

#endif

#endif // ifndef _LEARN_H_