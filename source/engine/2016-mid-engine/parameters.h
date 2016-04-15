#ifndef _2016_MID_PARAMETERS_
#define _2016_MID_PARAMETERS_

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_DEPTH
// [PARAM] min:50,max:120,step:5
PARAM_DEFINE PARAM_FUTILITY_MARGIN_DEPTH = 90;

// 静止探索でのfutility pruning
// [PARAM] min:50,max:150,step:5
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 128;

// futility pruningが適用されるdepth。これ以下のdepthに対して適用される。
// [PARAM] min:5,max:13,step:1
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 7;

// 親nodeでのfutilityを行なうdepthとそのmarginと、seeが負の指し手の枝刈りをするdepth

// [PARAM] min:5,max:13,step:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 7;

// [PARAM] min:100,max:200,step:10
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN = 170;

// [PARAM] min:2,max:10,step:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH = 4;

//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// [PARAM] min:500,max:1500,step:16
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 823;

// [PARAM] min:50,max:100,step:4
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 67;

// null moveの前提depthと、beta値を上回ったときにreturnするdepth
// [PARAM] min:4,max:10,step:1
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 7;

//
// probcut
//

// probcutの前提depth
// [PARAM] min:3,max:10,step:1
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;

//
// singular extension
//

// singular extensionの前提depth
// [PARAM] min:6,max:13,step:1
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 10;

// singular extensionのmarginを計算するときの係数
// Value rBeta = ttValue - PARAM_SINGULAR_MARGIN * depth / ONE_PLY;

// [PARAM] min:2,max:20,step:2
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 8;

// singular extensionで浅い探索をするときの深さに関する係数
// depth * PARAM_SINGULAR_SEARCH_DEPTH / 256

// [PARAM] min:64,max:192,step:32
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH = 128;

//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ
// [PARAM] min:8,max:32,step:1
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 16;

// historyによる枝刈りをする深さ
// [PARAM] min:8,max:32,step:1
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 4;

// historyの値によってreductionするときの係数
// [PARAM] min:4000,max:32000,step:256
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 14980;

//
// razoring pruning
// 


// return (Value)(PARAM_RAZORING_MARGIN + PARAM_RAZORING_ALPHA * static_cast<int>(d));

// [PARAM] min:64,max:1024,step:32
PARAM_DEFINE PARAM_RAZORING_MARGIN = 512;

// [PARAM] min:4,max:32,step:2
PARAM_DEFINE PARAM_RAZORING_ALPHA = 16;

//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// [PARAM] min:32,max:128,step:4
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
