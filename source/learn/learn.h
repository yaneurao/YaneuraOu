#ifndef _LEARN_H_
#define _LEARN_H_

//
//    学習時の各種設定
//

// ----------------------
//  重みベクトルの更新式
// ----------------------

// update_weights()の更新式を以下のなかから一つ選択すべし。

// 1) YaneGradによるupdate
//  これはAdaGradを改良したもの。
// 詳しい説明は、evaluate_kppt_learn.cppを見ること。

#define USE_YANE_GRAD_UPDATE


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
#define YANE_GRAD_EPSILON 5.0f

//
// SGDのとき
// 

// SDGの学習率η
#define SGD_ETA 32.0f


// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
// 詳しい説明は、learner.cppを見ること。

#define LOSS_FUNCTION_IS_CROSS_ENTOROPY


// ※　他、色々追加するかも。


// ----------------------
//    浅い探索の選択
// ----------------------

// 浅い探索の値としてevaluate()を用いる
//#define USE_EVALUATE_FOR_SHALLOW_VALUE

// 浅い探索の値としてqsearch()を用いる。
#define USE_QSEARCH_FOR_SHALLOW_VALUE


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
// 評価関数ファイルの保存
// ----------------------

// 保存するときのファイル名の末尾につける番号を、この局面数ごとにインクリメントしていく。
// 例) "KK_synthesized.bin0" → "KK_synthesized.bin1" → "KK_synthesized.bin2" → ...
#define EVAL_FILE_NAME_CHANGE_INTERVAL 500000000


// ----------------------
// 学習に関するデバッグ設定
// ----------------------

// kkpの一番大きな値を表示させることで学習が進んでいるかのチェックに用いる。
//#define DISPLAY_STATS_IN_UPDATE_WEIGHTS

// 学習時にsfenファイルを1万局面読み込むごとに'.'を出力する。
//#define DISPLAY_STATS_IN_THREAD_READ_SFENS


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

// その局面で探索した評価値がこれ以上になった時点でその対局は終了する。
#define GEN_SFENS_EVAL_LIMIT 3000

#endif
