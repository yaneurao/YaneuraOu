#ifndef _2017_GOKU_PARAMETERS_
#define _2017_GOKU_PARAMETERS_

// パラメーターの説明に "fixed"と書いてあるパラメーターはランダムパラメーター化するときでも変化しない。
// 「前提depth」は、これ以上ならその枝刈りを適用する(かも)の意味。
// 「適用depth」は、これ以下ならその枝刈りを適用する(かも)の意味。

// 現在の値から、min～maxの範囲で、+step,-step,+0を試す。
// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_DEPTH
// 元の値 = 150
// [PARAM] min:100,max:240,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 147;

// 
// 元の値 = 200
// [PARAM] min:100,max:240,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 195;


// 静止探索でのfutility pruning
// 元の値 = 128
// [PARAM] min:50,max:160,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 145;

// futility pruningの適用depth。
// この制限自体が要らない可能性がある。→　そうでもなかった。
// 元の値 = 7
// [PARAM] min:5,max:15,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 9;

// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。
// 元の値 = 7
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 12;

// 親nodeでのfutility margin
// 元の値 = 256
// [PARAM] min:100,max:300,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1 = 256;

// staticEvalから減算するmargin
// 元の値 = 200
// [PARAM] min:0,max:300,step:25,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN2 = 248;

// depthが2乗されるので影響大きい
// 元の値 = 35
// [PARAM] min:20,max:50,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 = 40;

// depthが2乗されるので影響大きい
// 元の値 = 35
// [PARAM] min:20,max:60,step:3,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2 = 51;

//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// 元の値 = 823
// [PARAM] min:500,max:1500,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 818;

// 元の値 = 67
// [PARAM] min:50,max:100,step:8,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 67;

// 元の値 = 35
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN = 31;

// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 12
// [PARAM] min:4,max:16,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 14;


//
// probcut
//

// probcutの前提depth
// 元の値 = 5
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;

// probcutのmargin
// 元の値 = 200
// [PARAM] min:100,max:300,step:3,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN = 194;


//
// singular extension
//

// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 10秒設定だと6か8あたりに局所解があるようだ。
// 元の値 = 8
// [PARAM] min:4,max:13,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 7;

// singular extensionのmarginを計算するときの係数
// rBeta = std::max(ttValue - PARAM_SINGULAR_MARGIN * depth / (8 * ONE_PLY), -VALUE_MATE);
// 元の値 = 256
// [PARAM] min:128,max:400,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 194;

// singular extensionで浅い探索をするときの深さに関する係数
// このパラメーター、長い時間でないと調整できないし下手に調整すべきではない。
// 元の値 = 16
// [PARAM] min:8,max:32,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH_ALPHA = 20;


//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ。適用depth。
// 元の値 = 16
// [PARAM] min:8,max:32,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 16;

// historyによる枝刈りをする深さ。適用depth。
// これ、将棋ではそこそこ上げたほうが長い時間では良さげ。
// 元の値 = 3
// [PARAM] min:2,max:32,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 9;


// historyの値によってreductionするときの係数
// これ、元のが (hist - 8000) / 20000みたいな意味ありげな値なので下手に変更しないほうが良さげ。
// 元の値 = 4000
// [PARAM] min:2000,max:8000,step:100,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 4000;


//
// Internal iterative deeping
// 

// 置換表に指し手が登録されていないときに浅い探索をするときの深さに関する係数
// 元の値 = 256
// [PARAM] min:128,max:384,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_IID_MARGIN_ALPHA = 251;


//
// razoring pruning
// 

// この値は、未使用。razoringはdepth < ONE_PLYでは行わないため。
// 元の値 = 0
// [PARAM] min:0,max:0,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN1 = 483;

// 以下、変更しても計測できるほどの差ではないようなので元の値にしておく。
// 元の値 = 570
// [PARAM] min:400,max:700,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN2 = 570;

// 元の値 = 603
// [PARAM] min:400,max:700,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN3 = 603;

// 元の値 = 554
// [PARAM] min:400,max:700,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN4 = 554;


//
// LMR reduction table
//

// 元の値 = 131
// [PARAM] min:64,max:256,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_ALPHA = 135;


//
// futility move count table
//

// どうも、元の値ぐらいが最適値のようだ…。

// 元の値 = 240
// [PARAM] min:150,max:400,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_ALPHA0 = 240;

// 元の値 = 500
// [PARAM] min:300,max:600,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_ALPHA1 = 492;

// 元の値 = 740
// [PARAM] min:500,max:2000,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_BETA0 = 740;

// 元の値 = 1000
// [PARAM] min:500,max:2000,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_BETA1 = 1000;


//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// 元の値 = 64
// 将棋では駒打ちがあるから少し増やしたほうがいいかも。
// →　そうでもなかった。固定しておく。
// historyの計算でこの64に基づいた値を使っている箇所があるからのような気がする。よく考える。
// [PARAM] min:32,max:128,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_QUIET_SEARCH_COUNT = 64;


// 静止探索での1手詰め
// 元の値 = 1
// →　1スレ2秒で対技巧だと有りのほうが強かったので固定しておく。
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_QSEARCH_MATE1 = 1;

// 通常探索での1手詰め
// →　よくわからないが1スレ2秒で対技巧だと無しのほうが強かった。
//     1スレ3秒にすると有りのほうが強かった。やはり有りのほうが良いのでは..
// 元の値 = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SEARCH_MATE1 = 1;

// 1手詰めではなくN手詰めを用いる
// 元の値 = 1
// [PARAM] min:1,max:5,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_WEAK_MATE_PLY = 1;


// aspiration searchの増加量
// 元の値 = 15
// [PARAM] min:12,max:40,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_ASPIRATION_SEARCH_DELTA = 16;


// 評価関数での手番の価値
// 元の値 = 20
// [PARAM] min:10,max:50,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_EVAL_TEMPO = 20;


#endif
