// Input features and network structure used in NNUE evaluation function
// NNUE評価関数で用いる入力特徴量とネットワーク構造

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <cstdint>
#include <type_traits>

#include "../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

// 後段構成はStockfishと同じNetworkArchitectureという型で表す。
// 重み全体を保持するNetworkはnetwork.hで定義する。

// Defines the network structure
// 入力特徴量とネットワーク構造が定義されたヘッダをincludeする

#if defined(NNUE_ARCHITECTURE_HEADER)

// 動的に生成されたファイルがある。
#include NNUE_ARCHITECTURE_HEADER

#elif defined(EVAL_NNUE_HALFKP256)

// 標準NNUE型。NNUE評価関数のデフォルトは、halfKP256

#include "architectures/halfkp_256x2-32-32.h"

#elif defined(EVAL_NNUE_KP256)

// kp型
#include "architectures/kp_256x2-32-32.h"

#elif defined(EVAL_NNUE_HALFKPE9)

// halfkpe9型
#include "architectures/halfkpe9_256x2-32-32.h"

#elif defined(EVAL_NNUE_HALFKP_VM_256X2_32_32)

// halfkpvm_256x2-32-32型
#include "architectures/halfkpvm_256x2-32-32.h"

#else

// どれも定義されていなかったので標準NNUE型にしておく。
#include "architectures/halfkp_256x2-32-32.h"

#endif

namespace YaneuraOu {
namespace Eval::NNUE {

	static_assert(kTransformedFeatureDimensions % MaxSimdWidth == 0, "");
	static_assert(NetworkArchitecture::OutputDimensions == 1, "");
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
	static_assert(std::is_same<NetworkArchitecture::OutputType, i64>::value, "");
#else
	static_assert(std::is_same<NetworkArchitecture::OutputType, i32>::value, "");
#endif

#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
    namespace Features {
    template<Side AssociatedKing> class HalfKA_hm2;
    }
    // Multi-ply updates currently support this single feature set only.
    // 複数手の差分更新は、現在この単一特徴量の組み合わせに対応する。
    inline constexpr bool kUseMultiPly = std::is_same_v<
      RawFeatures, Features::FeatureSet<Features::HalfKA_hm2<Features::Side::kFriend>>>;
#else
    inline constexpr bool kUseMultiPly = false;
#endif

	// Trigger for full calculation instead of difference calculation
	// 差分計算の代わりに全計算を行うタイミングのリスト
	constexpr auto kRefreshTriggers = RawFeatures::kRefreshTriggers;

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
