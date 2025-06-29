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

// Reductionsテーブルの初期化用
// 重要度　★★★★★
// 元の値 = Stockfish 17.1 : 2796 , step = 10
// [PARAM] min:1500,max:3500,step:2,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_REDUCTIONS_PARAM1 = 2796;


//
// futility pruning
//

// 深さに比例したfutility pruning


// 重要度　★★★★☆
// 元の値 = 138 , step = 30
// [PARAM] min:100,max:240,step:30,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_MARGIN_BETA = 108;



// 静止探索でのfutility pruning
// 重要度　★★★★☆
// 元の値 = Stockfish 14 : 200 , Stockfish 16 : 200 , Stockfish 17 : 280 , 306 , Stockfish 17.1 : 376,  step = 20
// [PARAM] min:50,max:300,step:30,interval:1,time_rate:1,
PARAM_DEFINE PARAM_FUTILITY_MARGIN_QUIET = 376;


// 親nodeでのfutilityの適用depth。
// 重要度　★★★☆☆
// この枝刈り、depthの制限自体が要らないような気がする。→　そうでもなかった。→こんなdepthいじらんほうがマシ
// 元の値 = Stockfish 14 : 12 , Stockfish 16 : 13  , Stockfish 16.1 : 15 , Stockfish 17: 12
// [PARAM] min:5,max:20,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_DEPTH = 12;


// 重要度　★★★★★
// このパラメーター、lmrDepth * lmrDepthに比例するので、影響がすごく大きい。
// 調整には気をつけること。
// 元の値 = Stockfish 14 : 31 , Stockfish 16 : 26 , Stockfish 17 : 24,25, Stockfish 17.1 : 27, step = 2
// [PARAM] min:15,max:50,step:2,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 = 27;


// lmrのときのseeの値。
// 重要度　★★★★☆
// Stockfishの7,8割ぐらいの値にするのがよさげ。
// 元の値 = Stockfish 17 : 167 , 162, Stockfish 17.1 : 158, step = 40
// [PARAM] min:0,max:300,step:10,interval:1,time_rate:1
PARAM_DEFINE PARAM_LMR_SEE_MARGIN1 = 158;



//
// null move dynamic pruning
//

// 重要度　★★★☆☆
// 元の値 = Stockfish 14 : 173 , Stockfish 16 : 152 , Stockfish 17 : 209,235 , Stockfish 17.1 : 213, step = 10
// [PARAM] min:50,max:400,step:10,interval:1,time_rate:1,
PARAM_DEFINE PARAM_NULL_MOVE_DYNAMIC_GAMMA = 213;

// 重要度　★★★☆☆
// 元の値 = Stockfish 17 : 23,21 , Stockfish 17.1 : 19,  step = 2
// [PARAM] min:10,max:60,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN1 = 19;

// 元の値 = Stockfish 17 : 400,421 , Stockfish 17.1 : 389, step = 50
// 重要度　★★★☆☆
// [PARAM] min:0,max:800,step:50,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_MARGIN2 = 389;



// null moveでbeta値を上回ったときに、これ以下ならreturnするdepth。適用depth。
// 重要度　★★★☆☆
// 元の値 = 16
// 他のNULL_MOVEの値が悪いと、この枝刈りを適用しないほうが強くなるわけで、
// このdepthがどんどん高い値に発散してしまうので注意。
// この値は、低くなるのが正しいチューニングだと思う。
// [PARAM] min:4,max:20,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_NULL_MOVE_RETURN_DEPTH = 16;


//
// ProbCut
//

// probcutのmargin
// 重要度　★★★☆☆
//    式 = beta + PARAM_PROBCUT_MARGIN1 - improving * PARAM_PROBCUT_MARGIN2A - opponentWorsening * PARAM_PROBCUT_MARGIN2B
//   improvingの効果怪しいので抑え気味にしておく。
// 元の値 = Stockfish 17 : 189,187 , Stockfish 17.1 : 201, step = 20
// [PARAM] min:100,max:300,step:5,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN1 = 201;

// 重要度　★★★☆☆
// 元の値 = 53 , Stockfish 17.1 = 58, step = 10
// [PARAM] min:20,max:100,step:5,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN2A = 58;


// 前のバージョンのStockfishではこの値は481。
// 重要度　★★★☆☆
// 元の値 = Stockfish 17 : 379,417 , Stockfish 17.1 : 400,  step = 10
// [PARAM] min:20,max:500,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_PROBCUT_MARGIN3 = 400;

//
// singular extension
//

// singular extensionの前提depth。
// これ変更すると他のパラメーターががらっと変わるので固定しておく。
// 元の値 = Stockfish 14 : 4 , Stockfish 16 : 4 , step = 1
// [PARAM] min:2,max:13,step:1,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_EXTENSION_DEPTH = 4;

// singular extensionのsingular betaを計算するときのマージン
// 重要度　★★★★☆
// 元の値 = Stockfish 16 : 64 , Stockfish 17 : 54,56 , step = 8
// [PARAM] min:16,max:1024,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN1 = 56;

// singular extensionのsingular betaを計算するときの係数
// 重要度　★★★★☆
// 自己対局だとすごく強くなって見えるかもしれないが、まやかしである。
// 元の値 = Stockfish 16 : 57 , Stockfish 17 : 77,79 , step = 8
// [PARAM] min:0,max:1024,step:8,interval:2,time_rate:1,fixed
PARAM_DEFINE PARAM_SINGULAR_MARGIN2 = 79;

//
// LMR
//

// LMRのパラメーター
// 重要度　★★★☆☆
// 元の値 = 51 , step = 4
// [PARAM] min:0,max:128,step:4,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_MARGIN1 = 51;

// 重要度　★★☆☆☆
// →　重要なパラメーターではあるが、下手にいじらないほうがよさげ。
// 元の値 = 10 , step = 1
// min:0,max:128,step:1,interval:1,time_rate:1,
// [PARAM] min:13,max:14,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_LMR_MARGIN2 = 13;

//
// in qsearch
//

// 重要度　★★☆☆☆
// →　重要なパラメーターではあるが、下手にいじらないほうがよさげ。
// 元の値 = Stockfish 14 : 95 , Stockfish 16 : 90 , Stockfish 17 : 82 ,83 , Stockfish 17.1 : 74, step = 10
// min:0,max:128,step:1,interval:1,time_rate:1,
// [PARAM] min:50,max:200,step:10,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_BAD_ENOUGH_SEE_VALUE = 74;


//
// misc
//


// aspiration searchの増加量。
// 重要度　★★☆☆☆
// →　調整が難しいパラメーター。下手にいじらないほうがよさげ。
// 古い評価関数では20ぐらいがベストだったが、NNUEでは17がベストのようだ。評価関数の精度向上とともに徐々に小さくなってきている。
// FV_SCALEを上げるなら、ここもう少し下げたほうがいいような…。
// 元の値 = Stockfish 16 : 10 , Stockfish 17 : 5 , step = 1
// [PARAM] min:4,max:30,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_ASPIRATION_SEARCH1 = 10;

// aspiration searchの定数。
// 重要度　★★☆☆☆
// 元の値 = Stockfish 16 : 15335, Stockfish 17 : 13797,13461 , Stockfish 17.1 : 11134 , step = 1000
// [PARAM] min:10000,max:20000,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_ASPIRATION_SEARCH2 = 11134;


// MovePicker

// move pickerでsortする係数 (super sort使用時)
// 重要度　★★★★★
// 元の値 = 3560 , step = 1
// [PARAM] min:0,max:6000,step:500,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_MOVEPICKER_SORT_ALPHA1 = 3560;

// move pickerでsortする係数 (super sort使用しない時)
// 重要度　★★★★★
// 元の値 = 3130 , step = 1
// [PARAM] min:0,max:6000,step:500,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_MOVEPICKER_SORT_ALPHA2 = 3330;

//
// mate..
// 

// 静止探索での1手詰め
// 重要度　★★☆☆☆
// 元の値 = 1
// →　1スレ2秒で対技巧だと有りのほうが強かったので固定しておく。
// NNUEだと、これ無しのほうが良い可能性がある。
// いったん無しでやって最後に有りに変更して有効か見る。
// 2スレ1,2秒程度だと無しと有意差がなかったが、4秒～8秒では、有りのほうが+R30ぐらい強い。
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_QSEARCH_MATE1 = 1;

// 通常探索での1手詰め
// 重要度　★★☆☆☆
// →　よくわからないが1スレ2秒で対技巧だと無しのほうが強かった。
//     1スレ3秒にすると有りのほうが強かった。やはり有りのほうが良いのでは..
// 元の値 = 1
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE PARAM_SEARCH_MATE1 = 1;

// ABテスト用
// 重要度　★☆☆☆☆
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE AB_TEST1 = 1;

// ABテスト用
// 重要度　★☆☆☆☆
// [PARAM] min:0,max:1,step:1,interval:1,time_rate:1,fixed
PARAM_DEFINE AB_TEST2 = 1;




//#endif // defined(GENSFEN2019)
#endif

