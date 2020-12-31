#ifndef __DLSHOGI_TYPES_H_INCLUDED__
#define __DLSHOGI_TYPES_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include <cfloat>

// dlshogiのソースコードを参考にさせていただいています。
// DeepLearningShogi GitHub : https://github.com/TadaoYamaoka/DeepLearningShogi

namespace dlshogi
{
	// Virtual Loss (Best Parameter)
	// Virtual Lossが何かわからない人は「MCTS Virtual Loss」とかでググれ。
	constexpr int VIRTUAL_LOSS = 1;

	// nodeの訪問回数を表現する型。
	// AWSのA100×8で3Mnpsぐらい出ると思うのだが、その場合、1430秒ほどで
	// u32の範囲を超えてしまう。(先にメモリが枯渇するだろうが)
	// ※　dlshogiでは、intなので715秒。
	// とりあえず、あとで変更できるように設計しておく。
	typedef u32 NodeCountType;

	// 勝った回数の記録用の型
	// floatだとIEE754においてfractionが23bitしかないのでdoubleにしたほうがよさげ。
	typedef float WinCountType;

	// 詰み探索で詰みの場合のvalue_winの定数
	// Node構造体のvalue_winがこの値をとる。
	// value_winはfloatで表現されているので、これの最大値を詰みを見つけた時の定数としておく。
	// 探索で取りえない範囲の値であれば何でも良い。(例えば、FLT_MINは、0より大きく1より小さいので偶然取り得る可能性はある)
	// -FLT_MAXは、floatの取りうる最小値として規格上、保証されている。
	constexpr WinCountType VALUE_WIN  =   FLT_MAX;
	constexpr WinCountType VALUE_LOSE =  -FLT_MAX;
	// 千日手の場合のvalue_winの定数
	constexpr WinCountType VALUE_DRAW =   FLT_MAX / 2;

	// UctSearcher::UctSearch()の返し値として使う。
	// 探索の結果を評価関数の呼び出しキューに追加したか、破棄したか。

	// 評価関数の呼び出しキューに追加した。
	constexpr float QUEUING   =  FLT_MAX;

	// 他のスレッドがすでにこのnodeの評価関数の呼び出しをしたあとであった(処理はまだ完了していない)ので、
	// 何もせずにリターンしたことを示す。
	constexpr float DISCARDED = -FLT_MAX;


}

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // __DLSHOGI_TYPES_H_INCLUDED__
