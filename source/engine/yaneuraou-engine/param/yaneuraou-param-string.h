// yaneuraou-param.h : 元ファイル
// param_conv.pyというPythonのスクリプトにより、以下のファイルは自動生成されます。
// 1) yaneuraou - param - extern.h
// 2) yaneuraou - param - array.h
// 3) yaneuraou - param - string.h
// 1),2),3)のファイルは手で編集しないでください。
// 教師局面生成用のパラメーター
// 低depthで強くする ≒ 低depth時の枝刈りを甘くする。
//  →　教師生成時と学習時の探索部の性質が違うのはNNUE型にとってよくないようなのだが、
//     これはたぶん許容範囲。
// パラメーターの説明に "fixed"と書いてあるパラメーターはランダムパラメーター化するときでも変化しない。
// 「前提depth」は、これ以上ならその枝刈りを適用する(かも)の意味。
// 「適用depth」は、これ以下ならその枝刈りを適用する(かも)の意味。
// 現在の値から、min～maxの範囲で、+step,-step,+0を試す。
// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。
//
// futility pruning
//
// 深さに比例したfutility pruning
// 元の値 = 223
// [PARAM] min:100,max:300,step:2,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_MARGIN_ALPHA1",

// 
// 元の値 = 170
// [PARAM] min:100,max:240,step:2,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_MARGIN_BETA",

// 静止探索でのfutility pruning
// 元の値 = 128
// [PARAM] min:50,max:160,step:5,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_MARGIN_QUIET",

// futility pruningの適用depth。
// この制限自体が要らない可能性がある。→　そうでもなかった。
// 元の値 = 8
// [PARAM] min:5,max:15,step:1,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_RETURN_DEPTH",

// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。
// 元の値 = 7
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_AT_PARENT_NODE_DEPTH",

// 親nodeでのfutility margin
// 元の値 = 283
// [PARAM] min:100,max:300,step:5,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1",

// 元の値 = 29
// [PARAM] min:20,max:50,step:1,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1",

// lmrのときのdepthの上限値。(これを超えるdepthは、↓この値とみなす)
// 元の値 = 18
// [PARAM] min:10,max:30,step:3,interval:1,time_rate:1,fixed
"PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2",

// lmrのときのseeの値。
// 元の値 = 221
// [PARAM] min:0,max:300,step:10,interval:1,time_rate:1,fixed
"PARAM_LMR_SEE_MARGIN1",

//
// null move dynamic pruning
//
// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ
// 元の値 = 982
// [PARAM] min:500,max:1500,step:5,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_DYNAMIC_ALPHA",

// 元の値 = 85
// [PARAM] min:50,max:100,step:8,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_DYNAMIC_BETA",

// 元の値 = 192
// [PARAM] min:50,max:400,step:50,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_DYNAMIC_GAMMA",

// 元の値 = 22977
// [PARAM] min:0,max:50000,step:5000,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_MARGIN0",

// 元の値 = 30
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_MARGIN1",

// 元の値 = 28
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_MARGIN2",

// 元の値 = 84
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_MARGIN3",

// 元の値 = 182
// [PARAM] min:0,max:400,step:30,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_MARGIN4",

// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 13
// [PARAM] min:4,max:16,step:1,interval:1,time_rate:1,fixed
"PARAM_NULL_MOVE_RETURN_DEPTH",

//
// ProbCut
//
// probcutの前提depth
// 元の値 = 4
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1,fixed
"PARAM_PROBCUT_DEPTH",

// probcutのmargin
//    式 = beta + PARAM_PROBCUT_MARGIN1 - improving * PARAM_PROBCUT_MARGIN2
//   improvingの効果怪しいので抑え気味にしておく。
// 元の値 = 176
// [PARAM] min:100,max:300,step:3,interval:3,time_rate:1
"PARAM_PROBCUT_MARGIN1",

// 元の値 = 46
// [PARAM] min:20,max:80,step:2,interval:3,time_rate:1
"PARAM_PROBCUT_MARGIN2",

//
// singular extension
//
// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 10秒設定だと6か8あたりに局所解があるようだ。
// 元の値 = 8
// [PARAM] min:4,max:13,step:1,interval:1,time_rate:1,fixed
"PARAM_SINGULAR_EXTENSION_DEPTH",

// singular extensionのmarginを計算するときの係数
// rBeta = std::max(ttValue - PARAM_SINGULAR_MARGIN * depth / (64 * ONE_PLY), -VALUE_MATE);
// 元の値 = 128
// [PARAM] min:64,max:400,step:4,interval:1,time_rate:1,fixed
"PARAM_SINGULAR_MARGIN",

// singular extensionで浅い探索をするときの深さに関する係数
// このパラメーター、長い時間でないと調整できないし下手に調整すべきではない。
// 元の値 = 16
// [PARAM] min:8,max:32,step:2,interval:1,time_rate:1,fixed
"PARAM_SINGULAR_SEARCH_DEPTH_ALPHA",

//
// pruning by history
//
// historyによる枝刈りをする深さ。適用depth。
// Stockfish10からこの値を大きくしすぎると良くないようだ。
// 元の値 = 4
// [PARAM] min:2,max:16,step:1,interval:1,time_rate:1,fixed
"PARAM_PRUNING_BY_HISTORY_DEPTH",

// historyの値によってreductionするときの係数
// これ、元のが (hist - 8000) / 20000みたいな意味ありげな値なので下手に変更しないほうが良さげ。
// 元の値 = 4000
// [PARAM] min:2000,max:8000,step:100,interval:1,time_rate:1,fixed
"PARAM_REDUCTION_BY_HISTORY",

//
// razoring pruning
// 
// 以下、変更しても計測できるほどの差ではないようなので元の値にしておく。
// 元の値 = 510
// [PARAM] min:400,max:700,step:10,interval:1,time_rate:1,fixed
"PARAM_RAZORING_MARGIN",

//
// etc..
// 
// 静止探索での1手詰め
// 元の値 = 1
// →　1スレ2秒で対技巧だと有りのほうが強かったので固定しておく。
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
"PARAM_QSEARCH_MATE1",

// 通常探索での1手詰め
// →　よくわからないが1スレ2秒で対技巧だと無しのほうが強かった。
//     1スレ3秒にすると有りのほうが強かった。やはり有りのほうが良いのでは..
// 元の値 = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
"PARAM_SEARCH_MATE1",

// 1手詰めではなくN手詰めを用いる
// 元の値 = 1
// [PARAM] min:1,max:5,step:2,interval:1,time_rate:1,fixed
"PARAM_WEAK_MATE_PLY",

// aspiration searchの増加量
// 元の値 = 17
// [PARAM] min:12,max:40,step:1,interval:2,time_rate:1,fixed
"PARAM_ASPIRATION_SEARCH_DELTA",

// 評価関数での手番の価値
// 元の値 = 20
// [PARAM] min:10,max:50,step:5,interval:2,time_rate:1,fixed
"PARAM_EVAL_TEMPO",

