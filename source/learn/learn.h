#ifndef _LEARN_H_
#define _LEARN_H_

// 学習時の各種設定


// ----------------------
//  重みベクトルの更新式
// ----------------------

// update_weights()の更新式を以下のなかから一つ選択すべし。

// 1) AdaGradによるupdate
#define USE_ADA_GRAD_UPDATE

// 2) SGDによるupdate
//#define USE_SGD_UPDATE

// ----------------------
// ゼロベクトルからの学習
// ----------------------

// 評価関数パラメーターをゼロベクトルから学習を開始する。
// ゼロ初期化して棋譜生成してゼロベクトルから学習させて、
// 棋譜生成→学習を繰り返すとプロの棋譜に依らないパラメーターが得られる。(かも)
// (すごく時間かかる)
//#define RESET_TO_ZERO_VECTOR

// ----------------------
//    浅い探索の選択
// ----------------------

// 浅い探索の値としてevaluate()を用いる
//#define USE_EVALUATE_FOR_SHALLOW_VALUE

// 浅い探索の値としてqsearch()を用いる。
#define USE_QSEARCH_FOR_SHALLOW_VALUE


// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
#define LOSS_FUNCTION_IS_CROSS_ENTOROPY

// 他、色々追加するかも。


// ----------------------
//   棋譜生成時の設定
// ----------------------

// これはgensfenコマンドに関する設定。

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



#endif
