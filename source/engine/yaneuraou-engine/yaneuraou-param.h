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
// 元の値 = 126 , step = 20
// [PARAM] min:100,max:300,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA1 = 126;

// 深さに比例したfutility pruning
// 元の値 = 42 , step = 4
// [PARAM] min:10,max:200,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_ALPHA2 = 42;
// 

// 元の値 = 138 , step = 20
// [PARAM] min:100,max:240,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 138;


// 静止探索でのfutility pruning
// 1つ前のバージョンの値 = 118。
// 元の値 = 200 , step = 20
// [PARAM] min:50,max:200,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 200;

// futility pruningの適用depth。
// この制限自体が要らない可能性がある。→　そうでもなかった。
// 元の値 = 9 , step = 1
// [PARAM] min:5,max:15,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_RETURN_DEPTH = 9;



// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。
// 元の値 = 13
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 13;

// 親nodeでのfutility margin
// 元の値 = 115 , step = 10
// [PARAM] min:100,max:400,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1 = 115;

// 元の値 = 122 , step = 5
// [PARAM] min:100,max:400,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_ALPHA = 122;


// 元の値 = 27 , step = 2
// [PARAM] min:15,max:50,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 = 27;

// lmrのときのseeの値。
// 元の値 = 185 ,step = 40
// [PARAM] min:0,max:300,step:20,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_SEE_MARGIN1 = 185;

// Reductionsテーブルの初期化用
// 元の値 = 2037 ,step = 8
// [PARAM] min:1500,max:2500,step:8,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTIONS_PARAM1 = 2037;

// Reductionの計算式に出てくる定数
// 元の値 = 1560 ,step = 32
// [PARAM] min:0,max:1024,step:128,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_ALPHA = 1560;

// Reductionの計算式に出てくる定数
// このパラメーター怖くて調整できない。
// 元の値 = 791 , step = 128
// [PARAM] min:300,max:1500,step:128,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_BETA = 791;

// Reductionの計算式に出てくる定数
// 元の値 = 945 , step = 128
// [PARAM] min:300,max:1500,step:128,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTION_GAMMA = 945;

//
// null move dynamic pruning
//

// 元の値 = 152 , step = 40
// [PARAM] min:50,max:400,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_GAMMA = 152;



// 元の値 = 24 , step = 2
// Stockfishの前バージョンではこの値は15。
// [PARAM] min:10,max:60,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN1 = 24;

// 元の値 = 281 , step = 50
// Stockfishの前バージョンではこの値は198。
// [PARAM] min:0,max:400,step:30,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN4 = 281;



// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 14
// 他のNULL_MOVEの値が悪いと、この枝刈りを適用しないほうが強くなるわけで、
// このdepthがどんどん高い値に発散してしまうので注意。
// [PARAM] min:4,max:16,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 14;


//
// ProbCut
//

// probcutのmargin
//    式 = beta + PARAM_PROBCUT_MARGIN1 - improving * PARAM_PROBCUT_MARGIN2
//   improvingの効果怪しいので抑え気味にしておく。
// 元の値 = 168 , step = 20
// [PARAM] min:100,max:300,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN1 = 168;

// 元の値 = 70 , step = 10
// [PARAM] min:20,max:80,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN2 = 70;

// 前のバージョンのStockfishではこの値は481。
// 元の値 = 416 , step = 10
// [PARAM] min:20,max:80,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN3 = 416;

//
// singular extension
//

// singular extensionのsingular betaを計算するときのマージン
// 元の値 = 64 , step = 8
// [PARAM] min:0,max:1024,step:8,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN1 = 64;

// singular extensionのsingular betaを計算するときの係数
// 自己対局だとすごく強くなって見えるかもしれないが、まやかしである。
// 元の値 = 57 , step = 8
// [PARAM] min:0,max:1024,step:8,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN2 = 57;

//
// LMR
//

// LMRのパラメーター
// 元の値 = 51 , step = 4
// [PARAM] min:0,max:128,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_MARGIN1 = 51;

// 元の値 = 10 , step = 1
// [PARAM] min:0,max:128,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_MARGIN2 = 10;

// 元の値 = 700 , step = 1
// [PARAM] min:0,max:1024,step:100,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_MARGIN3 = 700;


//
// pruning by history
//

// historyによる枝刈りをする深さ。適用depth。
// Stockfish10からこの値を大きくしすぎると良くないようだ。
// 元の値 = 6 , step = 1
// [PARAM] min:2,max:16,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PRUNING_BY_HISTORY_DEPTH = 6;


// update_all_stats()で、静止探索時のquietMoveとみなすbestvalueとbetaの差(PAWN_VALUEより少し小さな値)
// 元の値 = 90 , step = 5
// [PARAM] min:10,max:200,step:10,interval:5,time_rate:1,fixed
PARAM_DEFINE PARAM_UPDATE_ALL_STATS_EVAL_TH = 90;

//
// mate..
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


//
// misc
//

// fail lowを引き起こしたcounter moveにbonus与える時のevalのmargin値。
// 元の値 = 653 , step = 10
// [PARAM] min:10,max:200,step:10,interval:5,time_rate:1,fixed
PARAM_DEFINE PARAM_COUNTERMOVE_FAILLOW_MARGIN = 653;


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
// 元の値 = 10 , step = 1
// [PARAM] min:6,max:40,step:1,interval:2,time_rate:1, fixed
PARAM_DEFINE PARAM_ASPIRATION_SEARCH_DELTA = 10;


//
// move picker
//

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

