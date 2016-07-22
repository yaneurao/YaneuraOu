#ifndef _LEARN_H_
#define _LEARN_H_

//
//    学習時の各種設定
//

// ----------------------
//     configure
// ----------------------

// 以下のいずれかを選択すれば、そのあとの細々したものは自動的に選択される。
// 以下のいずれも選択しない場合は、そのあとの細々したものをひとつひとつ設定する必要がある。


// デフォルトの学習設定

#define LEARN_DEFAULT


// やねうら王2016Late用デフォルトの学習設定。
//
// 置換表を無効化するので通常対局は出来ない。learnコマンド用の実行ファイル専用。
//                       ~~~~~~~~~~~~~~~~~~

//#define LEARN_YANEURAOU_2016_LATE


// ----------------------
//  重みベクトルの更新式
// ----------------------

// update_weights()の更新式を以下のなかから一つ選択すべし。

// 1) YaneGradによるupdate
//  これはAdaGradを改良したもの。
// 詳しい説明は、evaluate_kppt_learn.cppを見ること。

//#define USE_YANE_GRAD_UPDATE


// 2) AdaGradによるupdate
//  これは普通のAdaGrad
// 詳しい説明は、evaluate_kppt_learn.cppを見ること。

//#define USE_ADA_GRAD_UPDATE


// 3) SGDによるupdate
//  これは普通のSGD
// 詳しい説明は、evaluate_kppt_learn.cppを見ること。

//#define USE_SGD_UPDATE


// ----------------------
//  更新式に関する係数
// ----------------------

//
// すべての更新式共通
//

// 評価項目の手番の学習率
// 手番じゃないほう×0.25ぐらいが良いのではないかと思っている。(値も手番じゃないほうの1/4ぐらいだろうし)
#define LEARN_ETA2_RATE 0.25f

//
// AdaGradのとき
//

// AdaGradの学習率η
#define ADA_GRAD_ETA 5.0f

//
// YaneGradのとき
// 

// YaneGradのα。Adam風に過去の履歴がある程度抑制されたほうが良いと思う。
#define YANE_GRAD_ALPHA 0.99f

// YaneGradの学習率η
#define YANE_GRAD_ETA 5.0f

// YaneGradのε
#define YANE_GRAD_EPSILON 1.0f

//
// SGDのとき
// 

// SDGの学習率η
#define SGD_ETA 32.0f


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

#define LEARN_READ_SFEN_SIZE (1000 * 1000 * 5)

// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY


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

// 置換表を用いない。(やねうら王Mid2016のみ対応)
// これをオンにすると通常対局時にも置換表を参照しなくなってしまうので棋譜からの学習を行う実行ファイルでのみオンにする。
// 棋譜からの学習時にはオンにしたほうがよさげ。
// 理由) 置換表にhitするとPV learが評価値の局面ではなくなってしまう。

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


// ----------------------
// 学習に関するデバッグ設定
// ----------------------

// kkpの一番大きな値を表示させることで学習が進んでいるかのチェックに用いる。
//#define DISPLAY_STATS_IN_UPDATE_WEIGHTS

// 学習時にsfenファイルを1万局面読み込むごとに'.'を出力する。
//#define DISPLAY_STATS_IN_THREAD_READ_SFENS

// 学習時のrmseとタイムスタンプの出力をこの回数に1回に減らす
#define LEARN_RMSE_OUTPUT_INTERVAL 1
#define LEARN_TIMESTAMP_OUTPUT_INTERVAL 3


// ----------------------
//   棋譜生成時の設定
// ----------------------

// これはgensfenコマンドに関する設定。
// これらは、configureの設定では変化しない。

// packされたsfenを書き出す
#define WRITE_PACKED_SFEN

// search()のleaf nodeまでの手順が合法手であるかを検証する。
#define TEST_LEGAL_LEAF

// packしたsfenをunpackして元の局面と一致するかをテストする。
// →　十分テストしたのでもう大丈夫やろ…。
//#define TEST_UNPACK_SFEN

// 棋譜を生成するときに一定手数の局面まで定跡を用いる機能
// これはOptions["BookMoves"]の値が反映される。この値が0なら、定跡を用いない。
// 用いる定跡は、Options["BookFile"]が反映される。

// 2駒の入れ替えを5手に1回ぐらいの確率で行なう。
#define USE_SWAPPING_PIECES

// その局面で探索した評価値がこれ以上になった時点でその対局は終了する。
#define GEN_SFENS_EVAL_LIMIT 3000

// タイムスタンプの出力をこの回数に一回に抑制する。
// スレッドを論理コアの最大数まで酷使するとコンソールが詰まるので…。
#define GEN_SFENS_TIMESTAMP_OUTPUT_INTERVAL 1

// ----------------------
// configureの内容を反映
// ----------------------

#ifdef LEARN_DEFAULT
#define USE_SGD_UPDATE
#undef SGD_ETA
#define SGD_ETA 32.0f
#undef LEARN_MINI_BATCH_SIZE
#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)
#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE
#define USE_QSEARCH_FOR_SHALLOW_VALUE
#undef EVAL_FILE_NAME_CHANGE_INTERVAL
#define EVAL_FILE_NAME_CHANGE_INTERVAL 1000000000
#endif

#ifdef LEARN_YANEURAOU_2016_LATE
#define USE_YANE_GRAD_UPDATE
#undef YANE_GRAD_ALPHA
#define YANE_GRAD_ALPHA 0.99f
#undef YANE_GRAD_ETA
#define YANE_GRAD_ETA 5.0f
#undef YANE_GRAD_EPSILON
#define YANE_GRAD_EPSILON 1.0f
#undef LEARN_MINI_BATCH_SIZE
#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)
#define LOSS_FUNCTION_IS_CROSS_ENTOROPY
#define USE_QSEARCH_FOR_SHALLOW_VALUE
#define DISABLE_TT_PROBE
#undef EVAL_FILE_NAME_CHANGE_INTERVAL
#define EVAL_FILE_NAME_CHANGE_INTERVAL 1000000000
#endif


#endif // ifndef _LEARN_H_