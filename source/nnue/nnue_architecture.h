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

// Shogi SFNN layer-stack settings, after selecting the architecture.
// 将棋用SFNNのLayerStack設定。構成ヘッダーを読み込んだ後に、型IDと既定値を揃える。
#if defined(EVAL_SFNN)
#define NNUE_SFNN_KING_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3 1
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9 2
#define NNUE_SFNN_KING_BUCKET_TYPE_K21K21 3
#define NNUE_SFNN_KING_BUCKET_TYPE_K29K29 4
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z 5
#define NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z 6
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3Z 7

#define NNUE_SFNN_HAND_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND4 1
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND16 2
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64 3
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64Z 4
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND256 5
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024 6

#ifndef NNUE_SFNN_HAND_BUCKETS
#define NNUE_SFNN_HAND_BUCKETS 1
#endif
#ifndef NNUE_SFNN_HAND_BUCKET_TYPE
#if NNUE_SFNN_HAND_BUCKETS == 1
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#elif NNUE_SFNN_HAND_BUCKETS == 4
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND4
#elif NNUE_SFNN_HAND_BUCKETS == 16
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND16
#elif NNUE_SFNN_HAND_BUCKETS == 64
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND64
#elif NNUE_SFNN_HAND_BUCKETS == 256
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND256
#elif NNUE_SFNN_HAND_BUCKETS == 1024
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024
#else
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#endif
#endif
#ifndef NNUE_SFNN_KING_BUCKETS
#define NNUE_SFNN_KING_BUCKETS 9
#endif
#ifndef NNUE_SFNN_KING_BUCKET_TYPE
#if NNUE_SFNN_KING_BUCKETS == 9
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K3K3
#elif NNUE_SFNN_KING_BUCKETS == 81
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K9K9
#elif NNUE_SFNN_KING_BUCKETS == 169
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z
#elif NNUE_SFNN_KING_BUCKETS == 441
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K21K21
#elif NNUE_SFNN_KING_BUCKETS == 841
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K29K29
#else
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_NONE
#endif
#endif
#ifndef NNUE_SFNN_PROGRESS_BUCKETS
#define NNUE_SFNN_PROGRESS_BUCKETS 1
#endif
#endif

namespace YaneuraOu::Eval::NNUE {

namespace Features {
// Type checks for lazy updates and Finny tables need only this declaration.
// 遅延更新・Finny Tablesの型判定には前方宣言だけで足りる。
template<Side AssociatedKing> class HalfKA_hm2;
}

#if defined(EVAL_SFNN)
	constexpr int kLayerStacks = LayerStacks;
#else
	constexpr int kLayerStacks = 1;
#endif

	static_assert(kTransformedFeatureDimensions % MaxSimdWidth == 0, "");
	static_assert(NetworkArchitecture::OutputDimensions == 1, "");
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
	static_assert(std::is_same_v<NetworkArchitecture::OutputType, i64>, "");
#else
	static_assert(std::is_same_v<NetworkArchitecture::OutputType, i32>, "");
#endif

#if defined(USE_LAZY_EVALUATE) && USE_LAZY_EVALUATE > 1
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

} // namespace YaneuraOu::Eval::NNUE

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
