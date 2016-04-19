#ifndef _2016_MID_PARAMETERS_
#define _2016_MID_PARAMETERS_

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_DEPTH
// [PARAM] min:50,max:120,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 87;

// 静止探索でのfutility pruning
// [PARAM] min:50,max:150,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 130;

// futility pruningが適用されるdepth。これ以下のdepthに対して適用される。
// [PARAM] min:5,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 8;

// 親nodeでのfutilityを行なうdepthとそのmarginと、seeが負の指し手の枝刈りをするdepth

// [PARAM] min:5,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 7;

// [PARAM] min:100,max:200,step:10,interval:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN = 170;

// [PARAM] min:2,max:10,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH = 5;

//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// [PARAM] min:500,max:1500,step:16,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 823;

// [PARAM] min:50,max:100,step:4,interval:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 69;

// null moveの前提depthと、beta値を上回ったときにreturnするdepth
// [PARAM] min:4,max:10,step:1,interval:3,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 6;

//
// probcut
//

// probcutの前提depth
// [PARAM] min:3,max:10,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;

//
// singular extension
//

// singular extensionの前提depth
// [PARAM] min:6,max:13,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 10;

// singular extensionのmarginを計算するときの係数
// Value rBeta = ttValue - PARAM_SINGULAR_MARGIN * depth / ONE_PLY;

// [PARAM] min:2,max:20,step:2,interval:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 8;

// singular extensionで浅い探索をするときの深さに関する係数
// depth * PARAM_SINGULAR_SEARCH_DEPTH / 256

// [PARAM] min:64,max:192,step:32,interval:2,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH = 96;

//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ
// [PARAM] min:8,max:32,step:1,interval:3,time_rate:3
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 16;

// historyによる枝刈りをする深さ
// [PARAM] min:2,max:32,step:1,interval:2,time_rate:1
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 4;

// historyの値によってreductionするときの係数
// [PARAM] min:4000,max:32000,step:256,interval:1,time_rate:1
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 14980;

//
// razoring pruning
// 


// return (Value)(PARAM_RAZORING_MARGIN + PARAM_RAZORING_ALPHA * static_cast<int>(d));

// [PARAM] min:64,max:1024,step:8,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN = 512;

// [PARAM] min:4,max:32,step:1,interval:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_ALPHA = 15;

//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// [PARAM] min:32,max:128,step:4,interval:2,time_rate:2
PARAM_DEFINE PARAM_QUIET_SEARCH_COUNT = 64;

//
// history of changed parameters
//
/*
ここに過去の変更履歴が自動的に書き込まれる。
右側にある「←」は値を減らしたときの勝率。「→」は値を増やしたときの勝率。
[HISTORY]
PARAM_QUIET_SEARCH_COUNT : 64 → 64(50.00%) : ←←(値56,勝率49.78%,1728局,有意55.91%) ,←(値60,勝率49.69%,1728局,有意58.23%) ,→(値68,勝率48.71%,3456局,有意85.68%) ,→→(値72,勝率46.30%,576局,有意96.44%) ,
PARAM_PRUNING_BY_HISTORY_DEPTH : 4 → 4(50.00%) : ←←(値8,勝率0.00%,0局,有意50.00%) ,←(値8,勝率46.10%,576局,有意97.10%) ,→(値8,勝率47.97%,3456局,有意95.29%) ,→→(値8,勝率48.70%,3456局,有意85.72%) ,
PARAM_SINGULAR_SEARCH_DEPTH : 128 → 96(52.22%) : ←←(値64,勝率49.02%,3456局,有意78.91%) ,←(値96,勝率52.22%,1728局,有意6.79%) ,→(値160,勝率49.43%,1728局,有意65.02%) ,→→(値192,勝率46.96%,1728局,有意97.97%) ,
PARAM_SINGULAR_EXTENSION_DEPTH : 10 → 10(50.00%) : ←←(値8,勝率49.05%,3456局,有意78.17%) ,←(値9,勝率49.24%,1728局,有意69.39%) ,→(値11,勝率47.05%,1728局,有意97.66%) ,→→(値12,勝率49.65%,1728局,有意59.41%) ,
PARAM_PROBCUT_DEPTH : 5 → 5(50.00%) : ←←(値3,勝率38.90%,576局,有意100.00%) ,←(値4,勝率46.78%,1728局,有意98.49%) ,→(値6,勝率47.25%,1728局,有意96.76%) ,→→(値7,勝率46.10%,576局,有意97.10%) ,
PARAM_NULL_MOVE_RETURN_DEPTH : 7 → 6(52.18%) : ←←(値5,勝率49.03%,1728局,有意74.34%) ,←(値6,勝率52.18%,1728局,有意7.20%) ,→(値8,勝率47.73%,3456局,有意96.92%) ,→→(値9,勝率49.60%,1728局,有意60.54%) ,
PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH : 4 → 4(50.00%) : ←←(値2,勝率50.31%,1728局,有意41.76%) ,←(値3,勝率51.15%,1728局,有意21.94%) ,→(値5,勝率51.30%,3456局,有意14.24%) ,→→(値6,勝率44.60%,576局,有意99.55%) ,
PARAM_FUTILITY_AT_PARENT_NODE_DEPTH : 7 → 7(50.00%) : ←←(値5,勝率49.44%,3456局,有意67.74%) ,←(値6,勝率49.42%,1728局,有意65.07%) ,→(値8,勝率50.31%,1728局,有意41.78%) ,→→(値9,勝率49.85%,3456局,有意54.84%) ,
PARAM_FUTILITY_RETURN_DEPTH : 7 → 7(50.00%) : ←←(値5,勝率48.91%,3456局,有意81.59%) ,←(値6,勝率47.50%,3456局,有意98.03%) ,→(値8,勝率50.26%,1728局,有意42.94%) ,→→(値9,勝率50.31%,1728局,有意41.76%) ,
PARAM_RAZORING_ALPHA : 16 → 16(50.00%) : ←←(値12,勝率49.50%,3456局,有意66.01%) ,←(値14,勝率50.98%,1728局,有意25.62%) ,→(値18,勝率49.87%,1728局,有意53.56%) ,→→(値20,勝率50.44%,3456局,有意35.76%) ,
PARAM_RAZORING_MARGIN : 512 → 512(50.00%) : ←←(値496,勝率46.30%,0局,有意96.44%) ,←(値504,勝率48.76%,1728局,有意79.76%) ,→(値520,勝率45.52%,0局,有意98.54%) ,→→(値528,勝率50.98%,3456局,有意21.10%) ,

PARAM_REDUCTION_BY_HISTORY : 14980 → 14980(50.00%) : ← 48.94%(14724) in 1728 tries , 有意確率 = 76.20% , → 50.04%(15236) in 1728 tries , 有意確率 = 48.81%
PARAM_SINGULAR_MARGIN : 8 → 8(50.00%) : ← 49.91%(6) in 1728 tries , 有意確率 = 52.37% , → 50.27%(10) in 1728 tries , 有意確率 = 42.92%
PARAM_NULL_MOVE_DYNAMIC_BETA : 67 → 67(50.00%) : ← 48.77%(63) in 1728 tries , 有意確率 = 79.67% , → 51.09%(71) in 3456 tries , 有意確率 = 18.45%
PARAM_NULL_MOVE_DYNAMIC_ALPHA : 823 → 823(50.00%) : ← 50.77%(807) in 3456 tries , 有意確率 = 26.34% , → 50.40%(839) in 1728 tries , 有意確率 = 39.41%
PARAM_FUTILITY_AT_PARENT_NODE_MARGIN : 170 → 170(50.00%) : ← 49.38%(160) in 1728 tries , 有意確率 = 66.16% , → 50.22%(180) in 1728 tries , 有意確率 = 44.08%

PARAM_FUTILITY_MARGIN_QUIET : 128 → 133(50.46%) : ← 49.00%(123) in 3456 tries  , → 50.46%(133) in 3456 tries
PARAM_FUTILITY_MARGIN_ALPHA : 90 → 85(50.60%) : ← 50.60%(85) in 3456 tries  , → 49.09%(95) in 3456 tries

*/

#endif
