#ifndef _2016_MID_PARAMETERS_
#define _2016_MID_PARAMETERS_

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_DEPTH
// 元の値 = 90
// [PARAM] min:50,max:120,step:5,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 90;

// 静止探索でのfutility pruning
// 元の値 = 128
// [PARAM] min:50,max:150,step:5,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 128;

// futility pruningが適用されるdepth。これ以下のdepthに対して適用される。
// 元の値 = 7
// [PARAM] min:5,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 7;

// 親nodeでのfutilityを行なうdepthとそのmarginと、seeが負の指し手の枝刈りをするdepth

// 元の値 = 7
// [PARAM] min:5,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 7;

// 元の値 = 170
// [PARAM] min:100,max:200,step:10,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN = 170;

// 元の値 = 4
// [PARAM] min:2,max:10,step:1,interval:2,time_rate:1
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
// [PARAM] min:50,max:100,step:4,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 67;

// null moveの前提depthと、beta値を上回ったときにreturnするdepth
// 元の値 = 12
// [PARAM] min:4,max:10,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 12;

//
// probcut
//

// probcutの前提depth
// 元の値 = 5
// [PARAM] min:3,max:10,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;

//
// singular extension
//

// singular extensionの前提depth
// 元の値 = 10
// [PARAM] min:6,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 10;

// singular extensionのmarginを計算するときの係数
// Value rBeta = ttValue - PARAM_SINGULAR_MARGIN * depth / ONE_PLY;

// 元の値 = 8
// [PARAM] min:2,max:20,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 8;

// singular extensionで浅い探索をするときの深さに関する係数
// depth * PARAM_SINGULAR_SEARCH_DEPTH / 256

// 元の値 = 128
// [PARAM] min:64,max:192,step:32,interval:2,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH = 128;

//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ
// 元の値 = 16
// [PARAM] min:8,max:32,step:1,interval:3,time_rate:3
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 16;

// historyによる枝刈りをする深さ

// 元の値 = 4
// [PARAM] min:8,max:32,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 4;

// historyの値によってreductionするときの係数
// 元の値 = 14980
// [PARAM] min:4000,max:32000,step:256,interval:1,time_rate:1
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 14980;

//
// razoring pruning
// 


// return (Value)(PARAM_RAZORING_MARGIN + PARAM_RAZORING_ALPHA * static_cast<int>(d));

// 元の値 = 512
// [PARAM] min:64,max:1024,step:32,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN = 512;

// 元の値 = 16
// [PARAM] min:4,max:32,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_ALPHA = 16;

//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// 元の値 = 64
// [PARAM] min:32,max:128,step:4,interval:2,time_rate:2
PARAM_DEFINE PARAM_QUIET_SEARCH_COUNT = 72;

//
// history of changed parameters
//
/*
ここに過去の変更履歴が自動的に書き込まれる。
右側にある「←」は値を減らしたときの勝率。「→」は値を増やしたときの勝率。
[HISTORY]
PARAM_REDUCTION_BY_HISTORY : 14724 → 14980(53.44%) : ←←(値14212,勝率50.41%,3744局,有意36.28%) ,←(値14468,勝率47.17%,1872局,有意97.60%) ,→(値14980,勝率53.44%,624局,有意4.08%) ,→→(値15236,勝率49.96%,1872局,有意51.14%) ,
PARAM_SINGULAR_MARGIN : 6 → 6(50.00%) : ←←(値2,勝率49.26%,3744局,有意73.58%) ,←(値4,勝率51.19%,1872局,有意20.31%) ,→(値8,勝率49.51%,1872局,有意63.43%) ,→→(値10,勝率51.03%,1872局,有意23.70%) ,
PARAM_NULL_MOVE_DYNAMIC_BETA : 67 → 67(50.00%) : ←←(値59,勝率50.71%,3744局,有意27.14%) ,←(値63,勝率45.71%,624局,有意98.44%) ,→(値71,勝率48.06%,1872局,有意91.14%) ,→→(値75,勝率48.11%,1872局,有意90.61%) ,
PARAM_NULL_MOVE_DYNAMIC_ALPHA : 823 → 823(50.00%) : ←←(値791,勝率50.08%,1872局,有意47.72%) ,←(値807,勝率45.48%,624局,有意98.84%) ,→(値839,勝率50.95%,1872局,有意25.47%) ,→→(値855,勝率50.54%,1872局,有意35.40%) ,
PARAM_FUTILITY_AT_PARENT_NODE_MARGIN : 160 → 150(54.05%) : ←←(値140,勝率49.51%,1872局,有意63.45%) ,←(値150,勝率54.05%,624局,有意2.10%) ,→(値170,勝率50.25%,1872局,有意43.19%) ,→→(値180,勝率50.29%,1872局,有意42.07%) ,
PARAM_FUTILITY_MARGIN_QUIET : 123 → 133(53.03%) : ←←(値113,勝率50.66%,1872局,有意32.34%) ,←(値118,勝率50.55%,3744局,有意31.98%) ,→(値128,勝率49.59%,3744局,有意63.72%) ,→→(値133,勝率53.03%,1872局,有意1.71%) ,
PARAM_FUTILITY_MARGIN_ALPHA : 95 → 95(50.00%) : ←←(値85,勝率49.35%,1872局,有意67.61%) ,←(値90,勝率50.00%,1872局,有意50.00%) ,→(値100,勝率50.65%,1872局,有意32.36%) ,→→(値105,勝率48.05%,1872局,有意91.44%) ,

*/

#endif
