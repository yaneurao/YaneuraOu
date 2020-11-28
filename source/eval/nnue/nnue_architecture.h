// Input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

// Defines the network structure
// 入力特徴量とネットワーク構造が定義されたヘッダをincludeする

#if defined(EVAL_NNUE_HALFKP256)

// 標準NNUE型。NNUE評価関数のデフォルトは、halfKP256

#include "architectures/halfkp_256x2-32-32.h"

#elif defined(EVAL_NNUE_KP256)

// KP256型
#include "architectures/k-p_256x2-32-32.h"

#elif defined(EVAL_NNUE_HALFKPE9)

// halfKPE9型
#include "architectures/halfkpe9_256x2-32-32.h"

#else

// どれも定義されていなかったので標準NNUE型にしておく。
#include "architectures/halfkp_256x2-32-32.h"

#endif

namespace Eval::NNUE {

	static_assert(kTransformedFeatureDimensions % kMaxSimdWidth == 0, "");
	static_assert(Network::kOutputDimensions == 1, "");
	static_assert(std::is_same<Network::OutputType, std::int32_t>::value, "");

	// Trigger for full calculation instead of difference calculation
	// 差分計算の代わりに全計算を行うタイミングのリスト
	constexpr auto kRefreshTriggers = RawFeatures::kRefreshTriggers;

}  // namespace Eval::NNUE

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
