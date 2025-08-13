#ifndef __DLSHOGI_TYPES_H_INCLUDED__
#define __DLSHOGI_TYPES_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../types.h"
#include <cfloat> // FLT_MAX

// dlshogiのソースコードを参考にさせていただいています。
// DeepLearningShogi GitHub : https://github.com/TadaoYamaoka/DeepLearningShogi

namespace dlshogi {

// Virtual Loss (Best Parameter)
// Virtual Lossが何かわからない人は「MCTS Virtual Loss」とかでググれ。
constexpr int VIRTUAL_LOSS = 1;
	
// nodeの訪問回数を表現する型。
// AWSのA100×8で3Mnpsぐらい出ると思うのだが、その場合、1430秒ほどで
// u32の範囲を超えてしまう。(先にメモリが枯渇するだろうが)
// ※　dlshogiでは、intなので715秒。
// とりあえず、あとで変更できるように設計しておく。
typedef uint32_t NodeCountType;

// 勝った回数(勝率)の記録用の型
// floatだとIEE754においてfractionが23bitしかないのでdoubleにしたほうがよさげだが、
// 少しnpsが下がるので、10M nodeぐらいしか読ませないとわかっている時は、floatでもいいかも。
#if defined(WIN_TYPE_DOUBLE)
typedef double WinType;
#else
typedef float WinType;
#endif

// 子ノードの数を表現する型
// 将棋では合法手は593手とされており、メモリがもったいないので、16bit整数で持つ。
typedef uint16_t ChildNumType;

// 詰み探索で詰みの場合の定数
// Moveの最上位byteでWin/Lose/Drawの状態を表現する。(32bit型のMoveでは、ここは使っていないはずなので)
constexpr uint32_t VALUE_WIN  = 0x1000000;
constexpr uint32_t VALUE_LOSE = 0x2000000;
// 千日手の場合のvalue_winの定数
constexpr uint32_t VALUE_DRAW = 0x4000000;

// ノード未展開を表す定数
// move_countがこの値であれば、ノードは未展開。
constexpr NodeCountType NOT_EXPANDED = std::numeric_limits<NodeCountType>::max();


// UctSearcher::UctSearch()の返し値として使う。
// 探索の結果を評価関数の呼び出しキューに追加したか、破棄したか。

// 評価関数の呼び出しキューに追加した。
constexpr float QUEUING   =  FLT_MAX;

// 他のスレッドがすでにこのnodeの評価関数の呼び出しをしたあとであった(処理はまだ完了していない)ので、
// 何もせずにリターンしたことを示す。
constexpr float DISCARDED = -FLT_MAX;
} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // __DLSHOGI_TYPES_H_INCLUDED__
