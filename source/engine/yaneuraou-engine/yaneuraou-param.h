#ifndef YANEURAOU_PARAM_H_INCLUDED
#define YANEURAOU_PARAM_H_INCLUDED

// yaneuraou-param.h : 元ファイル
// param_conv.pyというPythonのスクリプトにより、以下のファイルは自動生成されます。
// 1) yaneuraou - param - extern.h
// 2) yaneuraou - param - array.h
// 3) yaneuraou - param - string.h
// 1),2),3)のファイルは手で編集しないでください。


//#if  defined(GENSFEN2019)
// 教師局面生成用のパラメーター
// 低depthで強くする ≒ 低depth時の枝刈りを甘くする。
//#include "yaneuraou-param_gen.h"
//  →　教師生成時と学習時の探索部の性質が違うのはNNUE型にとってよくないようなのだが、
//     これはたぶん許容範囲。

//#else

// パラメーターの説明に "fixed"と書いてあるパラメーターはランダムパラメーター化するときでも変化しない。
// 「前提depth」は、これ以上ならその枝刈りを適用する(かも)の意味。
// 「適用depth」は、これ以下ならその枝刈りを適用する(かも)の意味。

// 現在の値から、min～maxの範囲で、+step,-step,+0を試す。
// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。

//
// futility pruning
//

// 深さに比例したfutility pruning
// 元の値 = 168 , step = 20
// [PARAM] min:100,max:300,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA1 = 168;

// 

// 元の値 = 138 , step = 20
// [PARAM] min:100,max:240,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 138;


// 静止探索でのfutility pruning
// 元の値 = 118 , step = 20
// [PARAM] min:50,max:200,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 118;

// futility pruningの適用depth。
// この制限自体が要らない可能性がある。→　そうでもなかった。
// 元の値 = 8 , step = 1
// [PARAM] min:5,max:15,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 8;

// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。
// 元の値 = 11
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 11;

// 親nodeでのfutility margin
// 元の値 = 122 , step = 30
// [PARAM] min:100,max:400,step:20,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1 = 122;

// 元の値 = 25 , step = 4
// [PARAM] min:15,max:50,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 = 25;

// 元の値 = 20 , step = 4
// [PARAM] min:10,max:30,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2 = 20;

// lmrのときのseeの値。
// 元の値 = 203 ,step = 40
// [PARAM] min:0,max:300,step:20,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_SEE_MARGIN1 = 203;


// Reductionの計算式に出てくる定数
// 元の値 = 1463 ,step = 32
// [PARAM] min:0,max:1024,step:16,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_ALPHA = 1463;

// Reductionの計算式に出てくる定数
// このパラメーター怖くて調整できない。
// 元の値 = 1010 , step = 128
// [PARAM] min:600,max:1500,step:128,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_BETA = 1010;

//
// null move dynamic pruning
//

// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ

// 256倍されていることに注意。
// 元の値 = 1024 , step = 128
// [PARAM] min:500,max:1500,step:64,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_ALPHA = 1024;

// 256倍されていることに注意。85なら85/256 = 1/3とほぼ等価。
// 元の値 = 85 , step = 24
// [PARAM] min:50,max:120,step:3,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_BETA = 85;

// 元の値 = 147 , step = 40
// [PARAM] min:50,max:400,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_GAMMA = 147;


// 元の値 = 14695 , step = 4000
// [PARAM] min:0,max:50000,step:500,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN0 = 14695;

// 元の値 = 15 , step = 2
// [PARAM] min:10,max:60,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN1 = 15;

// 元の値 = 22 , step = 2
// [PARAM] min:10,max:60,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN2 = 22;

// 元の値 = 15 , step = 4
// [PARAM] min:5,max:40,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN3 = 15;

// 元の値 = 198 , step = 50
// [PARAM] min:0,max:400,step:30,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN4 = 198;



// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 13
// 他のNULL_MOVEの値が悪いと、この枝刈りを適用しないほうが強くなるわけで、
// このdepthがどんどん高い値に発散してしまうので注意。
// [PARAM] min:4,max:16,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 13;


//
// ProbCut
//

// probcutの前提depth
// 元の値 = 4 , step = 1
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_DEPTH = 4;

// probcutのmargin
//    式 = beta + PARAM_PROBCUT_MARGIN1 - improving * PARAM_PROBCUT_MARGIN2
//   improvingの効果怪しいので抑え気味にしておく。
// 元の値 = 179 , step = 20
// [PARAM] min:100,max:300,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN1 = 179;

// 元の値 = 46 , step = 10
// [PARAM] min:20,max:80,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN2 = 46;


//
// singular extension
//

// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 元の値 = 4 , step = 1
// [PARAM] min:2,max:13,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 4;

// singular extensionのmarginを計算するときの係数(Stockfishのコードから256倍にscalingしてある)
// +64すると自己対局だとすごく強くなって見えるかもしれないが、まやかしである。
// 元の値 = 768 , step = 64
// [PARAM] min:0,max:1024,step:64,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN = 768;


//
// pruning by history
//

// historyによる枝刈りをする深さ。適用depth。
// Stockfish10からこの値を大きくしすぎると良くないようだ。
// 元の値 = 5 , step = 1
// [PARAM] min:2,max:16,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 5;


// historyの値によってreductionするときの係数
// 元の値 = 4334 , step = 500
// [PARAM] min:2000,max:8000,step:250,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_BY_HISTORY = 4334;


//
// etc..
// 

// 静止探索での1手詰め
// 元の値 = 1
// →　1スレ2秒で対技巧だと有りのほうが強かったので固定しておく。
// NNUEだと、これ無しのほうが良い可能性がある。
// いったん無しでやって最後に有りに変更して有効か見る。
// 2スレ1,2秒程度だと無しと有意差がなかったが、4秒～8秒では、有りのほうが+R30ぐらい強い。
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_QSEARCH_MATE1 = 1;

// 通常探索での1手詰め
// →　よくわからないが1スレ2秒で対技巧だと無しのほうが強かった。
//     1スレ3秒にすると有りのほうが強かった。やはり有りのほうが良いのでは..
// 元の値 = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SEARCH_MATE1 = 1;

// 1手詰めではなくN手詰めを用いる
// ※　3手,5手はコストに見合わないようだ。
// 元の値 = 1
// [PARAM] min:1,max:5,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_WEAK_MATE_PLY = 1;


// qsearch()でnull moveのときもevaluate()を呼び出す。
// この値が0(false)ならば、null moveのときはeval = 前局面にEval::Tempoを加算した値 とする。
// 計測できる差にならない。
// PARAM_EVAL_TEMPOを変動させていると(適正値から離れていると)、
// evaluate()を呼び出したほうが良いことになってしまうのでこれが1のときのほうが良いことになってしまうので注意。
// 元の値 = 1 , step = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_QSEARCH_FORCE_EVAL = 0;


// aspiration searchの増加量。
// 古い評価関数では20ぐらいがベストだったが、NNUEでは17がベストのようだ。
// 元の値 = 16 , step = 1
// [PARAM] min:12,max:40,step:1,interval:2,time_rate:1, fixed
PARAM_DEFINE PARAM_ASPIRATION_SEARCH_DELTA = 16;


// MovePickerの quietのvalue計算用の係数
// 注意 : この調整をONにするためには movepick.cpp の
// 　以下の変数を使ってあるところの #if を有効化("#if 1"を"#if 0"に変更)
// 　する必要がある。

// 元の値 = 32 , step = 8
// これだけ " 2 * " と係数が掛かっているので倍にして考える必要がある。
// 32は大きすぎるのではないかと…。
// [PARAM] min:10,max:50,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE MOVE_PICKER_Q_PARAM1 = 32;

// 元の値 = 32 , step = 8
// [PARAM] min:10,max:50,step:8,interval:1,time_rate:1,fixed
PARAM_DEFINE MOVE_PICKER_Q_PARAM2 = 32;

// 元の値 = 32 , step = 8
// [PARAM] min:10,max:50,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE MOVE_PICKER_Q_PARAM3 = 32;

// 元の値 = 16 , step = 4
// [PARAM] min:10,max:50,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE MOVE_PICKER_Q_PARAM4 = 16;

// 元の値 = 16 , step = 4
// [PARAM] min:10,max:50,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE MOVE_PICKER_Q_PARAM5 = 16;


// ABテスト用
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE AB_TEST1 = 1;

// ABテスト用
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE AB_TEST2 = 1;



//#endif // defined(GENSFEN2019)
#endif

