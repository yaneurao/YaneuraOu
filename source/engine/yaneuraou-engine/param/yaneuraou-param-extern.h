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
// 元の値 = 224 , step = 20
// [PARAM] min:100,max:300,step:3,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_MARGIN_ALPHA1;

// 
// 元の値 = 170 , step = 20
// [PARAM] min:100,max:240,step:2,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_MARGIN_BETA;

// 静止探索でのfutility pruning
// 元の値 = 155 , step = 10
// [PARAM] min:50,max:200,step:2,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_MARGIN_QUIET;

// futility pruningの適用depth。
// この制限自体が要らない可能性がある。→　そうでもなかった。
// 元の値 = 8 , step = 1
// [PARAM] min:5,max:15,step:1,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_RETURN_DEPTH;

// 親nodeでのfutilityの適用depth。
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。
// 元の値 = 7
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_AT_PARENT_NODE_DEPTH;

// 親nodeでのfutility margin
// 元の値 = 266 , step = 20
// [PARAM] min:100,max:400,step:3,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1;

// 元の値 = 30
// [PARAM] min:20,max:50,step:2,interval:2,time_rate:1,fixed
extern int PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1;

// lmrのときのdepthの上限値。(これを超えるdepthは、↓この値とみなす)
// 元の値 = 18 , step = 1
// [PARAM] min:10,max:30,step:1,interval:1,time_rate:1,fixed
extern int PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2;

// lmrのときのseeの値。
// 元の値 = 221 ,step = 40
// [PARAM] min:0,max:300,step:3,interval:1,time_rate:1,fixed
extern int PARAM_LMR_SEE_MARGIN1;

// Reductionの計算式に出てくる定数
// 元の値 = 503 ,step = 16
// [PARAM] min:0,max:1024,step:3,interval:1,time_rate:1,fixed
extern int PARAM_REDUCTION_ALPHA;

// Reductionの計算式に出てくる定数
// このパラメーター怖くて調整できない。
// 元の値 = 915 , step = 128
// [PARAM] min:600,max:1500,step:128,interval:1,time_rate:1,fixed
extern int PARAM_REDUCTION_BETA;

//
// null move dynamic pruning
//
// null move dynamic pruningのときの
//  Reduction = (α + β * depth ) / 256 + ...みたいなαとβ
// 元の値 = 1015 , step = 50
// [PARAM] min:500,max:1500,step:5,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_DYNAMIC_ALPHA;

// 元の値 = 85 , step = 15
// [PARAM] min:50,max:120,step:3,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_DYNAMIC_BETA;

// 元の値 = 191 , step = 40
// [PARAM] min:50,max:400,step:5,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_DYNAMIC_GAMMA;

// 元の値 = 22977 , step = 8000
// [PARAM] min:0,max:50000,step:500,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_MARGIN0;

// 元の値 = 30
// [PARAM] min:10,max:60,step:2,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_MARGIN1;

// 元の値 = 28 , step = 2
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_MARGIN2;

// 元の値 = 84 , step = 4
// [PARAM] min:10,max:100,step:1,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_MARGIN3;

// 元の値 = 168 , step = 50
// [PARAM] min:0,max:400,step:5,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_MARGIN4;

// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 元の値 = 13
// 他のNULL_MOVEの値が悪いと、この枝刈りを適用しないほうが強くなるわけで、
// このdepthがどんどん高い値に発散してしまうので注意。
// [PARAM] min:4,max:16,step:1,interval:1,time_rate:1,fixed
extern int PARAM_NULL_MOVE_RETURN_DEPTH;

//
// ProbCut
//
// probcutの前提depth
// 元の値 = 4
// [PARAM] min:3,max:10,step:1,interval:1,time_rate:1,fixed
extern int PARAM_PROBCUT_DEPTH;

// probcutのmargin
//    式 = beta + PARAM_PROBCUT_MARGIN1 - improving * PARAM_PROBCUT_MARGIN2
//   improvingの効果怪しいので抑え気味にしておく。
// 元の値 = 183
// [PARAM] min:100,max:300,step:3,interval:1,time_rate:1,fixed
extern int PARAM_PROBCUT_MARGIN1;

// 元の値 = 49 , step = 5
// [PARAM] min:20,max:80,step:1,interval:1,time_rate:1,fixed
extern int PARAM_PROBCUT_MARGIN2;

//
// singular extension
//
// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 10秒設定だと6か8あたりに局所解があるようだ。
// 元の値 = 7 , step = 1
// [PARAM] min:4,max:13,step:1,interval:2,time_rate:1,fixed
extern int PARAM_SINGULAR_EXTENSION_DEPTH;

// singular extensionのmarginを計算するときの係数
// 元の値 = 4 , step = 1
// [PARAM] min:0,max:10,step:1,interval:2,time_rate:1,fixed
extern int PARAM_SINGULAR_MARGIN;

//
// pruning by history
//
// historyによる枝刈りをする深さ。適用depth。
// Stockfish10からこの値を大きくしすぎると良くないようだ。
// 元の値 = 4 , step = 1
// [PARAM] min:2,max:16,step:1,interval:1,time_rate:1,fixed
extern int PARAM_PRUNING_BY_HISTORY_DEPTH;

// historyの値によってreductionするときの係数
// 元の値 = 5278 , step = 500
// [PARAM] min:2000,max:8000,step:250,interval:1,time_rate:1,fixed
extern int PARAM_REDUCTION_BY_HISTORY;

//
// razoring pruning
// 
// 以下、変更しても計測できるほどの差ではないようだが。
// 元の値 = 510 , step = 100
// [PARAM] min:400,max:700,step:3,interval:1,time_rate:1,fixed
extern int PARAM_RAZORING_MARGIN;

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
extern int PARAM_QSEARCH_MATE1;

// 通常探索での1手詰め
// →　よくわからないが1スレ2秒で対技巧だと無しのほうが強かった。
//     1スレ3秒にすると有りのほうが強かった。やはり有りのほうが良いのでは..
// 元の値 = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
extern int PARAM_SEARCH_MATE1;

// 1手詰めではなくN手詰めを用いる
// ※　3手,5手はコストに見合わないようだ。
// 元の値 = 1
// [PARAM] min:1,max:5,step:2,interval:1,time_rate:1,fixed
extern int PARAM_WEAK_MATE_PLY;

// qsearch()でnull moveのときもevaluate()を呼び出す。
// この値が0(false)ならば、null moveのときはeval = 前局面にEval::Tempoを加算した値 とする。
// 計測できる差にならない。
// PARAM_EVAL_TEMPOを変動させていると(適正値から離れていると)、
// evaluate()を呼び出したほうが良いことになってしまうのでこれが1のときのほうが良いことになってしまうので注意。
// 元の値 = 1 , step = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
extern int PARAM_QSEARCH_FORCE_EVAL;

// aspiration searchの増加量。
// 古い評価関数では20ぐらいがベストだったが、NNUEでは17がベストのようだ。
// 元の値 = 17 , step = 1
// [PARAM] min:12,max:40,step:1,interval:2,time_rate:1, fixed
extern int PARAM_ASPIRATION_SEARCH_DELTA;

// 評価関数での手番の価値
// 元の値 = 20 , step = 2
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int PARAM_EVAL_TEMPO;

// MovePickerの quietのvalue計算用の係数
// 元の値 = 32 , step = 4
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int MOVE_PICKER_Q_PARAM1;

// 元の値 = 32 , step = 4
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int MOVE_PICKER_Q_PARAM2;

// 元の値 = 32 , step = 4
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int MOVE_PICKER_Q_PARAM3;

// 元の値 = 16 , step = 4
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int MOVE_PICKER_Q_PARAM4;

// 元の値 = 16 , step = 4
// [PARAM] min:10,max:50,step:1,interval:1,time_rate:1,fixed
extern int MOVE_PICKER_Q_PARAM5;

// ABテスト用
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
extern int AB_TEST1;

// ABテスト用
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
extern int AB_TEST2;

//#endif // defined(GENSFEN2019)
