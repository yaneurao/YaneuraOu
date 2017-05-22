#ifndef _LEARN_H_
#define _LEARN_H_

// =====================
//  学習時の設定
// =====================

// 以下のいずれかを選択すれば、そのあとの細々したものは自動的に選択される。
// いずれも選択しない場合は、そのあとの細々したものをひとつひとつ設定する必要がある。

// デフォルトの学習設定
// #define LEARN_DEFAULT

// elmo方式での学習設定。
// #define LEARN_ELMO_METHOD

// やねうら王2017GOKU用のデフォルトの学習設定
// ※　このオプションは実験中なので使わないように。
#define LEARN_YANEURAOU_2017_GOKU

// ----------------------
//  重みベクトルの更新式
// ----------------------

// update_weights()の更新式を以下のなかから一つ選択すべし。
// 詳しい説明は、evaluate_kppt_learn.cppを見ること。

// 1) SGDによるupdate
//  これは普通のSGD

//#define USE_SGD_UPDATE


// 2) AdaGradによるupdate
//  これは普通のAdaGrad

//#define USE_ADA_GRAD_UPDATE


// 3) Adamによるupdate
//  これは普通のAdam 評価関数ファイルの16倍ぐらいWeight用のメモリが必要。

//#define USE_ADAM_UPDATE



// ----------------------
//    学習時の設定
// ----------------------

// mini-batchサイズ。
// この数だけの局面をまとめて勾配を計算する。
// 小さくするとupdate_weights()の回数が増えるので収束が速くなる。勾配が不正確になる。
// 大きくするとupdate_weights()の回数が減るので収束が遅くなる。勾配は正確に出るようになる。

#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)

// ファイルから1回に読み込む局面数。これだけ読み込んだあとshuffleする。
// ある程度大きいほうが良いが、この数×34byte×3倍ぐらいのメモリを消費する。10M局面なら340MB*3程度消費する。
// THREAD_BUFFER_SIZE(=10000)の倍数にすること。

#define LEARN_SFEN_READ_SIZE (1000 * 1000 * 10)

// 学習時の評価関数の保存間隔。この局面数だけ学習させるごとに保存。
#define LEARN_EVAL_SAVE_INTERVAL (80000000ULL)


// KKP,KPPの評価値、ミラーを考慮するか(ミラーの位置にある評価値を同じ値にする)
// #define USE_KKP_MIRROR_WRITE
// #define USE_KPP_MIRROR_WRITE

// KKPの評価値、フリップを考慮するか(盤面を180度回転させた位置にある評価値を同じ値にする)
// #define USE_KKP_FLIP_WRITE

// 評価関数ファイルを出力するときに指数移動平均(EMA)を用いた平均化を行なう。
// #define LEARN_USE_EMA

// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY

// 目的関数が交差エントロピーだが、勝率の関数を通さない版
// #define LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE

// elmo(WCSC27)の方式
// #define LOSS_FUNCTION_IS_ELMO_METHOD

// ※　他、色々追加するかも。


// ----------------------
//    浅い探索の選択
// ----------------------

// 浅い探索の値としてevaluate()を用いる
//#define USE_EVALUATE_FOR_SHALLOW_VALUE

// 浅い探索の値としてqsearch()を用いる。
//#define USE_QSEARCH_FOR_SHALLOW_VALUE


// ※　他、色々追加するかも。


// ----------------------
// ゼロベクトルからの学習
// ----------------------

// 評価関数パラメーターをゼロベクトルから学習を開始する。
// ゼロ初期化して棋譜生成してゼロベクトルから学習させて、
// 棋譜生成→学習を繰り返すとプロの棋譜に依らないパラメーターが得られる。(かも)
// (すごく時間かかる)

//#define RESET_TO_ZERO_VECTOR


// ----------------------
//        置換表
// ----------------------

// 置換表を用いない。(やねうら王2017Earlyのみ対応)
// これをオンにすると通常対局時にも置換表を参照しなくなってしまうので棋譜からの学習を行う実行ファイルでのみオンにする。
// 棋譜からの学習時にはオンにしたほうがよさげ。
// 理由) 置換表にhitするとPV leafが評価値の局面ではなくなってしまう。

//#define DISABLE_TT_PROBE


// ----------------------
// 評価関数ファイルの保存
// ----------------------

// 保存するときのフォルダ番号を、この局面数ごとにインクリメントしていく。
// (Windows環境限定)
// 例) "0/KK_synthesized.bin" →　"1/KK_synthesized.bin"
// ※　Linux環境でファイルの上書きが嫌ならconfig.hのMKDIRのコードを有効にしてください。
// 現状、10億局面ずつ。
#define EVAL_FILE_NAME_CHANGE_INTERVAL (u64)1000000000

// evalファイルの保存は1度のみにする。
//#define EVAL_SAVE_ONLY_ONCE

// ----------------------
// 学習に関するデバッグ設定
// ----------------------

// 学習時にsfenファイルを1万局面読み込むごとに'.'を出力する。
//#define DISPLAY_STATS_IN_THREAD_READ_SFENS

// 学習時のrmseとタイムスタンプの出力をこの回数に1回に減らす
#define LEARN_RMSE_OUTPUT_INTERVAL 1
#define LEARN_TIMESTAMP_OUTPUT_INTERVAL 10

// ----------------------
//  学習のときの浮動小数
// ----------------------

// これをdoubleにしたほうが計算精度は上がるが、重み配列絡みのメモリが倍必要になる。
typedef float LearnFloatType;


// =====================
// 教師局面生成時の設定
// =====================

// これはgensfenコマンドに関する設定。
// これらは、configureの設定では変化しない。

// search()のleaf nodeまでの手順が合法手であるかを検証する。
//#define TEST_LEGAL_LEAF

// 棋譜を生成するときに一定手数の局面まで定跡を用いる機能
// これはOptions["BookMoves"]の値が反映される。この値が0なら、定跡を用いない。
// 用いる定跡は、Options["BookFile"]が反映される。

// ときどき合法手のなかからランダムに1手選ぶ。(Apery方式)
//#define USE_RANDOM_LEGAL_MOVE

// タイムスタンプの出力をこの回数に一回に抑制する。
// スレッドを論理コアの最大数まで酷使するとコンソールが詰まるので…。
#define GEN_SFENS_TIMESTAMP_OUTPUT_INTERVAL 1

// ----------------------
// configureの内容を反映
// ----------------------

#if defined (LEARN_DEFAULT)
#define USE_SGD_UPDATE
#define USE_KPP_MIRROR_WRITE
#undef LEARN_MINI_BATCH_SIZE
#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)
#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE
#define USE_QSEARCH_FOR_SHALLOW_VALUE
#undef EVAL_FILE_NAME_CHANGE_INTERVAL
#define EVAL_FILE_NAME_CHANGE_INTERVAL 1000000000
#define USE_RANDOM_LEGAL_MOVE
#endif

// ----------------------
//  elmoの方法での学習
// ----------------------

#if defined( LEARN_ELMO_METHOD )

// elmoでは、ゲームの勝敗を学習時に利用するのでこのフラグを書き出す必要がある。
#define GENSFEN_SAVE_GAME_RESULT

#define USE_ADA_GRAD_UPDATE
#define USE_KPP_MIRROR_WRITE
#define USE_KKP_FLIP_WRITE
#define USE_KKP_MIRROR_WRITE

#undef LEARN_MINI_BATCH_SIZE
#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)
#define LOSS_FUNCTION_IS_ELMO_METHOD
#define USE_QSEARCH_FOR_SHALLOW_VALUE
#undef EVAL_FILE_NAME_CHANGE_INTERVAL
#define EVAL_FILE_NAME_CHANGE_INTERVAL 1000000000
#define USE_RANDOM_LEGAL_MOVE

#endif

// ----------------------
//  やねうら王2017GOKUの方法
// ----------------------

#if defined(LEARN_YANEURAOU_2017_GOKU)

#define GENSFEN_SAVE_GAME_RESULT

#define USE_ADA_GRAD_UPDATE
//#define USE_ADAM_UPDATE

#define USE_KPP_MIRROR_WRITE
#undef LEARN_MINI_BATCH_SIZE
#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)

// 損失関数、あとでよく考える。
#define LOSS_FUNCTION_IS_ELMO_METHOD
//#define LOSS_FUNCTION_IS_YANE_ELMO_METHOD

#define USE_QSEARCH_FOR_SHALLOW_VALUE
#undef EVAL_FILE_NAME_CHANGE_INTERVAL
#define EVAL_FILE_NAME_CHANGE_INTERVAL 1000000000
#define USE_RANDOM_LEGAL_MOVE

// 出力を減らして高速化。
//#undef LEARN_RMSE_OUTPUT_INTERVAL
//#define LEARN_RMSE_OUTPUT_INTERVAL 10

// #define EVAL_SAVE_ONLY_ONCE

#endif

// ----------------------
// 設定内容に基づく定数文字列
// ----------------------

// 更新式に応じた文字列。(デバッグ用に出力する。)
#if defined(USE_SGD_UPDATE)
#define LEARN_UPDATE "SGD"
#elif defined(USE_YANE_SGD_UPDATE)
#define LEARN_UPDATE "YaneSGD"
#elif defined(USE_ADA_GRAD_UPDATE)
#define LEARN_UPDATE "AdaGrad"
#elif defined(USE_ADAM_UPDATE)
#define LEARN_UPDATE "Adam"
#endif

#if defined(LOSS_FUNCTION_IS_WINNING_PERCENTAGE)
#define LOSS_FUNCTION "WINNING_PERCENTAGE"
#elif defined(LOSS_FUNCTION_IS_CROSS_ENTOROPY)
#define LOSS_FUNCTION "CROSS_ENTOROPY"
#elif defined(LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE)
#define LOSS_FUNCTION "CROSS_ENTOROPY_FOR_VALUE"
#elif defined(LOSS_FUNCTION_IS_ELMO_METHOD)
#define LOSS_FUNCTION "ELMO_METHOD(WCSC27)"
#elif defined(LOSS_FUNCTION_IS_YANE_ELMO_METHOD)
#define LOSS_FUNCTION "YANE_ELMO_METHOD(WCSC27)"
#endif

// rmseの観測用
#if 0
#undef LEARN_RMSE_OUTPUT_INTERVAL
#define LEARN_RMSE_OUTPUT_INTERVAL 1
#define LEARN_SFEN_NO_SHUFFLE
#undef LEARN_SFEN_READ_SIZE
#define LEARN_SFEN_READ_SIZE 100000 
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
		PackedSfen sfen;
		s16 score; // PV leafでの評価値

		u16 move; // PVの初手

		// 初期局面からの局面の手数。
		u16 gamePly;

		// この局面の手番側が、ゲームを最終的に勝っているならtrue。負けているならfalse。
		// 引き分けに至った場合は、局面自体書き出さない。
		bool isWin;
	};
}

#endif // ifndef _LEARN_H_