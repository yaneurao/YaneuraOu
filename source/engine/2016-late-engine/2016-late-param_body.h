#ifndef _2016_LATE_PARAMETERS_
#define _2016_LATE_PARAMETERS_

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
// [PARAM] min:100,max:240,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 143;

// 
// 元の値 = 200
// [PARAM] min:100,max:240,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 200;


// 静止探索でのfutility pruning
// 元の値 = 128
// [PARAM] min:50,max:160,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 145;

// futility pruningの適用depth。
// 元の値 = 7
// 7より8のほうが良さげなので固定。
// [PARAM] min:5,max:13,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 10;

// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。
// 元の値 = 7
// [PARAM] min:5,max:13,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 13;

// 親nodeでのfutility margin
// 元の値 = 256
// [PARAM] min:100,max:300,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1 = 246;

// これ、あまり下手にいじると他のパラメーターに影響がありすぎるので固定。
// 元の値 = 8
// [PARAM] min:6,max:12,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH1 = 8;

// これ、あまり下手にいじると他のパラメーターに影響がありすぎるので固定。
// 元の値 = 7
// [PARAM] min:6,max:12,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH2 = 7;

// depthが2乗されるので影響大きい
// 元の値 = 35
// [PARAM] min:20,max:50,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 = 41;

// depthが2乗されるので影響大きい
// 元の値 = 35
// [PARAM] min:20,max:60,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2 = 51;


//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// 元の値 = 823
// [PARAM] min:500,max:1500,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 823;

// 元の値 = 67
// [PARAM] min:50,max:100,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 53;

// 元の値 = 35
// [PARAM] min:10,max:60,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN = 32;

// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 12
// [PARAM] min:4,max:15,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 13;


//
// probcut
//

// probcutの前提depth
// 元の値 = 5
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 4;

// probcutのmargin
// 元の値 = 200
// [PARAM] min:100,max:300,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN = 216;


//
// singular extension
//

// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 10秒設定だと6か8あたりに局所解があるようだ。
// 元の値 = 8
// [PARAM] min:4,max:13,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 8;

// singular extensionのmarginを計算するときの係数
// rBeta = std::max(ttValue - PARAM_SINGULAR_MARGIN * depth / (8 * ONE_PLY), -VALUE_MATE);
// 元の値 = 256
// [PARAM] min:128,max:400,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 200;

// singular extensionで浅い探索をするときの深さに関する係数
// このパラメーター、長い時間でないと調整できないし下手に調整すべきではない。
// 元の値 = 16
// [PARAM] min:8,max:32,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH_ALPHA = 16;


//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ。適用depth。
// 元の値 = 16
// [PARAM] min:8,max:32,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 17;

// historyによる枝刈りをする深さ。適用depth。
// これ、将棋ではそこそこ上げたほうが長い時間では良さげ。
// 元の値 = 3
// [PARAM] min:2,max:32,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 9;


// historyの値によってreductionするときの係数
// これ、元のが (hist - 8000) / 20000みたいな意味ありげな値なので下手に変更しないほうが良さげ。
// 元の値 = 8000
// [PARAM] min:4000,max:15000,step:40,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 8000;


//
// Internal iterative deeping
// 

// historyの値によってreductionするときの係数
// 元の値 = 256
// [PARAM] min:128,max:384,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_IID_MARGIN_ALPHA = 261;


//
// razoring pruning
// 

// 元の値 = 483
// [PARAM] min:400,max:700,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN1 = 483;

// 元の値 = 570
// [PARAM] min:400,max:700,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN2 = 555;

// 元の値 = 603
// [PARAM] min:400,max:700,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN3 = 593;

// 元の値 = 554
// [PARAM] min:400,max:700,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_RAZORING_MARGIN4 = 539;


//
// LMR reduction table
//

// 元の値 = 128
// [PARAM] min:64,max:256,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_ALPHA = 124;


//
// futility move count table
//

// どうも、元の値ぐらいが最適値のようだ…。

// 元の値 = 240
// [PARAM] min:150,max:400,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_ALPHA0 = 240;

// 元の値 = 290
// [PARAM] min:150,max:400,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_ALPHA1 = 288;

// 元の値 = 773
// [PARAM] min:500,max:2000,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_BETA0 = 773;

// 元の値 = 1045
// [PARAM] min:500,max:2000,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MOVE_COUNT_BETA1 = 1041;


//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// 元の値 = 64
// 将棋では駒打ちがあるから少し増やしたほうがいいかも。
// →　そうでもなかった。固定しておく。
// [PARAM] min:32,max:128,step:1,interval:1,time_rate:1,fixed
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


#endif
