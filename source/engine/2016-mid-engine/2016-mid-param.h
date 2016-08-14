#ifndef _2016_MID_PARAMETERS_
#define _2016_MID_PARAMETERS_

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_DEPTH
// 180ぐらいが最適値かも。
// 元の値 = 150
// [PARAM] min:100,max:240,step:3,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 150;

// 元の値 = 200
// [PARAM] min:100,max:240,step:3,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 200;

// 静止探索でのfutility pruning
// 元の値 = 128
// [PARAM] min:50,max:150,step:3,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 128;

// futility pruningが適用されるdepth。これ以下のdepthに対して適用される。
// 元の値 = 7
// [PARAM] min:5,max:13,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 7;

// 親nodeでのfutilityを行なうdepthとそのmarginと、seeが負の指し手の枝刈りをするdepth

// 元の値 = 7
// [PARAM] min:5,max:13,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 7;

// 元の値 = 256
// [PARAM] min:100,max:200,step:3,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN = 256;

// 元の値 = 4
// [PARAM] min:2,max:10,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH = 4;


//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// 元の値 = 823
// [PARAM] min:500,max:1500,step:16,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 823;

// 元の値 = 67
// [PARAM] min:50,max:100,step:3,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 67;

// null moveの前提depthと、beta値を上回ったときにreturnするdepth
// 元の値 = 12
// [PARAM] min:4,max:10,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 12;


//
// probcut
//

// probcutの前提depth
// 元の値 = 5
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;


//
// singular extension
//

// singular extensionの前提depth
// 元の値 = 8
// [PARAM] min:6,max:13,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 8;

// singular extensionのmarginを計算するときの係数
// Value rBeta = ttValue - (PARAM_SINGULAR_MARGIN / 8) * depth / ONE_PLY;

// 元の値 = 16
// [PARAM] min:2,max:20,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 16;

// singular extensionで浅い探索をするときの深さに関する係数
// depth * PARAM_SINGULAR_SEARCH_DEPTH / 256

// 元の値 = 128
// [PARAM] min:64,max:192,step:32,interval:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH = 128;


//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ
// 元の値 = 16
// [PARAM] min:8,max:32,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 16;

// historyによる枝刈りをする深さ

// 元の値 = 4
// [PARAM] min:2,max:32,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 4;

// historyの値によってreductionするときの係数
// 元の値 = 20000
// [PARAM] min:5000,max:30000,step:256,interval:1,time_rate:1
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 20000;


//
// Internal iterative deeping
// 

// historyの値によってreductionするときの係数
// 元の値 = 256
// [PARAM] min:128,max:384,step:16,interval:1,time_rate:1
PARAM_DEFINE PARAM_IID_MARGIN_ALPHA = 256;


//
// razoring pruning
// 

// 元の値 = 483, 570, 603, 554
// 4つ目、大きめの値が最適値かも。

// [PARAM] min:400,max:700,step:8,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN1 = 483;

// [PARAM] min:400,max:700,step:8,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN2 = 570;

// [PARAM] min:400,max:700,step:8,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN3 = 603;

// [PARAM] min:400,max:700,step:8,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN4 = 554;

//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// 元の値 = 64
// 将棋では駒打ちがあるから少し増やしたほうがいいかも。
// [PARAM] min:32,max:128,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_QUIET_SEARCH_COUNT = 64;


//
// history of changed parameters
//
/*
ここに過去の変更履歴が自動的に書き込まれる。
右側にある「←」は値を減らしたときの勝率。「→」は値を増やしたときの勝率。
[HISTORY]
*/

#endif
