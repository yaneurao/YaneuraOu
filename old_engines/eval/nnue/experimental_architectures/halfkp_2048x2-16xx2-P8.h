// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義

#include "../features/feature_set.h"
#include "../features/half_kp.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/clipped_relu.h"
#include "../layers/sum.h"

namespace Eval {

namespace NNUE {

// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
    Features::HalfKP<Features::Side::kFriend>>;

	// 並列化数
	constexpr int ParallelNumber = 8;

	// 変換後の入力特徴量の次元数
	constexpr IndexType kTransformedFeatureDimensions = 2048/ParallelNumber;
	// 隠れ層 2層目以降の次元数
	constexpr int kHiddenLayerDimentions = 16;

	namespace Layers {

		// ネットワーク構造の定義
		using InputLayer1 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer1_1 = ClippedReLU<AffineTransform<InputLayer1, kHiddenLayerDimentions>>;
		using HiddenLayer1_2 = ClippedReLU<AffineTransform<HiddenLayer1_1, kHiddenLayerDimentions>>;

		using InputLayer2 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer2_1 = ClippedReLU<AffineTransform<InputLayer2, kHiddenLayerDimentions>>;
		using HiddenLayer2_2 = ClippedReLU<AffineTransform<HiddenLayer2_1, kHiddenLayerDimentions>>;

		using InputLayer3 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer3_1 = ClippedReLU<AffineTransform<InputLayer3, kHiddenLayerDimentions>>;
		using HiddenLayer3_2 = ClippedReLU<AffineTransform<HiddenLayer3_1, kHiddenLayerDimentions>>;

		using InputLayer4 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer4_1 = ClippedReLU<AffineTransform<InputLayer4, kHiddenLayerDimentions>>;
		using HiddenLayer4_2 = ClippedReLU<AffineTransform<HiddenLayer4_1, kHiddenLayerDimentions>>;

		using InputLayer5 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer5_1 = ClippedReLU<AffineTransform<InputLayer5, kHiddenLayerDimentions>>;
		using HiddenLayer5_2 = ClippedReLU<AffineTransform<HiddenLayer5_1, kHiddenLayerDimentions>>;

		using InputLayer6 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer6_1 = ClippedReLU<AffineTransform<InputLayer6, kHiddenLayerDimentions>>;
		using HiddenLayer6_2 = ClippedReLU<AffineTransform<HiddenLayer6_1, kHiddenLayerDimentions>>;

		using InputLayer7 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer7_1 = ClippedReLU<AffineTransform<InputLayer7, kHiddenLayerDimentions>>;
		using HiddenLayer7_2 = ClippedReLU<AffineTransform<HiddenLayer7_1, kHiddenLayerDimentions>>;

		using InputLayer8 = InputSlice<kTransformedFeatureDimensions * 2>;
		using HiddenLayer8_1 = ClippedReLU<AffineTransform<InputLayer8, kHiddenLayerDimentions>>;
		using HiddenLayer8_2 = ClippedReLU<AffineTransform<HiddenLayer8_1, kHiddenLayerDimentions>>;

		using OutputLayer = Sum<
			AffineTransform<HiddenLayer1_2, 1> , AffineTransform<HiddenLayer2_2, 1> ,
			AffineTransform<HiddenLayer3_2, 1> , AffineTransform<HiddenLayer4_2, 1> ,
			AffineTransform<HiddenLayer5_2, 1> , AffineTransform<HiddenLayer6_2, 1> ,
			AffineTransform<HiddenLayer7_2, 1> , AffineTransform<HiddenLayer8_2, 1>
			>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace NNUE

}  // namespace Eval
