#ifndef _2016_MID_PARAMETERS_
#define _2016_MID_PARAMETERS_

//
// 書式
// 

// min  : 最小値
// max  : 最大値
// step : 更新の時の幅
// interval   : この数字が2なら2周に1回だけ調整
//    めったに最適値が変動しないパラメーターはここを大きくして大丈夫。
// trial_rate : 通常、試行は500局だが、この数字が2だと1000局やる。
//    勝率に変化が少ないパラメーターはここを大きくすべき。
// time_rate  : rtime(1手の思考時間)に倍率を掛ける。
//		深いdepthでしかしないような枝刈りの効果を調べるときに用いる。

//
// futility pruning
//

// 深さに比例したfutility pruning
// depth手先で評価値が変動する幅が = depth * PARAM_FUTILITY_MARGIN_ALPHA
// [PARAM] min:50,max:120,step:4,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA = 87;

// 静止探索でのfutility pruning
// [PARAM] min:50,max:150,step:5,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 123;

// futility pruningが適用されるdepth。これ以下のdepthに対して適用される。
// [PARAM] min:5,max:13,step:1,interval:2,trial_rate:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 9;

// 親nodeでのfutilityを行なうdepthとそのmarginと、seeが負の指し手の枝刈りをするdepth

// [PARAM] min:5,max:13,step:1,interval:2,trial_rate:2,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 7;

// [PARAM] min:100,max:200,step:10,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN = 190;

// [PARAM] min:2,max:10,step:1,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH = 4;

//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// [PARAM] min:500,max:1500,step:8,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 815;

// [PARAM] min:50,max:100,step:4,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 59;

// null moveの前提depthと、beta値を上回ったときにreturnするdepth,time_rate:1
// [PARAM] min:4,max:10,step:1,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 6;

//
// probcut
//

// probcutの前提depth
// [PARAM] min:3,max:10,step:1,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 5;

//
// singular extension
//

// singular extensionの前提depth
// [PARAM] min:6,max:13,step:1,interval:2,trial_rate:1,time_rate:5
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 10;

// singular extensionのmarginを計算するときの係数
// Value rBeta = ttValue - PARAM_SINGULAR_MARGIN * depth / ONE_PLY;

// [PARAM] min:2,max:20,step:1,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 5;

// singular extensionで浅い探索をするときの深さに関する係数
// depth * PARAM_SINGULAR_SEARCH_DEPTH / 256

// [PARAM] min:64,max:230,step:28,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_SINGULAR_SEARCH_DEPTH = 192;

//
// pruning by move count,history,etc..
//

// move countによる枝刈りをする深さ
// [PARAM] min:8,max:32,step:1,interval:1,trial_rate:1,time_rate:5
PARAM_DEFINE PARAM_PRUNING_BY_MOVE_COUNT_DEPTH = 15;

// historyによる枝刈りをする深さ
// [PARAM] min:2,max:32,step:1,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 4;

// historyの値によってreductionするときの係数
// [PARAM] min:4000,max:32000,step:256,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 14980;

//
// razoring pruning
// 


// return (Value)(PARAM_RAZORING_MARGIN + PARAM_RAZORING_ALPHA * static_cast<int>(d));

// [PARAM] min:64,max:1024,step:16,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_RAZORING_MARGIN = 512;

// [PARAM] min:4,max:32,step:2,interval:2,trial_rate:2,time_rate:1
PARAM_DEFINE PARAM_RAZORING_ALPHA = 16;

//
// etc..
// 

// この個数までquietの指し手を登録してhistoryなどを増減させる。
// [PARAM] min:32,max:128,step:4,interval:1,trial_rate:1,time_rate:1
PARAM_DEFINE PARAM_QUIET_SEARCH_COUNT = 56;

//
// history of changed parameters
//
/*
ここに過去の変更履歴が自動的に書き込まれる。
右側にある「←」は値を減らしたときの勝率。「→」は値を増やしたときの勝率。
[HISTORY]
PARAM_QUIET_SEARCH_COUNT : 56 → 56(50.00%) : ← 46.29%(52) , → 46.88%(60)
PARAM_RAZORING_MARGIN : 496 → 512(51.96%) : ← 49.03%(480) , → 51.96%(512)
PARAM_REDUCTION_BY_HISTORY : 14980 → 14980(50.00%) : ← 49.02%(14724) , → 49.80%(15236)
PARAM_PRUNING_BY_HISTORY_DEPTH : 3 → 4(51.36%) : ← 51.27%(2) , → 51.36%(4)
PARAM_PRUNING_BY_MOVE_COUNT_DEPTH : 16 → 15(51.28%) : ← 51.28%(15) , → 49.90%(17)
PARAM_SINGULAR_SEARCH_DEPTH : 169 → 192(53.42%) : ← 46.59%(141) , → 53.42%(197)
PARAM_SINGULAR_MARGIN : 6 → 5(54.47%) : ← 54.47%(5) , → 53.63%(7)
PARAM_PROBCUT_DEPTH : 5 → 5(50.00%) : ← 46.43%(4) , → 49.80%(6)
PARAM_NULL_MOVE_RETURN_DEPTH : 5 → 6(50.60%) : ← 47.38%(4) , → 50.60%(6)
PARAM_NULL_MOVE_DYNAMIC_BETA : 63 → 59(51.76%) : ← 51.76%(59) , → 50.39%(67)
PARAM_NULL_MOVE_DYNAMIC_ALPHA : 823 → 815(54.12%) : ← 54.12%(815) , → 50.49%(831)
PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH : 4 → 4(50.00%) : ← 43.36%(3) , → 44.97%(5)
PARAM_FUTILITY_AT_PARENT_NODE_MARGIN : 190 → 190(50.00%) : ← 47.67%(180) , → 47.95%(200)
PARAM_FUTILITY_MARGIN_QUIET : 123 → 123(50.00%) : ← 46.85%(118) , → 48.34%(128)
PARAM_FUTILITY_MARGIN_ALPHA : 91 → 87(51.27%) : ← 51.27%(87) , → 50.29%(95)
PARAM_RAZORING_ALPHA : 14 → 16(51.73%) : ← 50.10%(12) , → 51.73%(16)
PARAM_FUTILITY_AT_PARENT_NODE_DEPTH : 8 → 7(51.22%) : ← 51.22%(7) , → 49.85%(9)
PARAM_FUTILITY_RETURN_DEPTH : 8 → 9(51.82%) : ← 49.51%(7) , → 51.82%(9)
PARAM_QUIET_SEARCH_COUNT : 60 → 56(50.20%) : ← 50.20%(56) , → 48.93%(64)
PARAM_RAZORING_MARGIN : 496 → 496(50.00%) : ← 47.06%(480) , → 49.80%(512)
PARAM_REDUCTION_BY_HISTORY : 15236 → 14980(54.40%) : ← 54.40%(14980) , → 49.22%(15492)
PARAM_PRUNING_BY_HISTORY_DEPTH : 4 → 3(51.19%) : ← 51.19%(3) , → 50.78%(5)
PARAM_PRUNING_BY_MOVE_COUNT_DEPTH : 17 → 16(53.89%) : ← 53.89%(16) , → 49.90%(18)
PARAM_SINGULAR_SEARCH_DEPTH : 169 → 169(50.00%) : ← 48.03% , → 48.04%
PARAM_SINGULAR_MARGIN : 6 → 6(50.00%) : ← 47.95% , → 49.21%
PARAM_SINGULAR_EXTENSION_DEPTH : 10 → 10(50.00%) : ← 50.00% , → 48.72%
PARAM_PROBCUT_DEPTH : 5 → 5(50.00%) : ← 48.13% , → 47.83%
PARAM_NULL_MOVE_RETURN_DEPTH : 6 → 5(50.69%) : ← 50.69% , → 49.90%
PARAM_NULL_MOVE_DYNAMIC_BETA : 63 → 63(50.00%) : ← 47.76% , → 47.95%
PARAM_NULL_MOVE_DYNAMIC_ALPHA : 823 → 823(50.00%) : ← 49.60% , → 44.47%
PARAM_FUTILITY_AT_PARENT_NODE_SEE_DEPTH : 4 → 4(50.00%) : ← 49.51% , → 46.64%
PARAM_FUTILITY_AT_PARENT_NODE_MARGIN : 180 → 190(53.13%) : ← 47.85% , → 53.13%
PARAM_FUTILITY_MARGIN_QUIET : 128 → 123(51.66%) : ← 51.66% , → 49.31%
PARAM_FUTILITY_MARGIN_ALPHA : 95 → 91(50.89%) : ← 50.89% , → 49.40%
*/

#endif
