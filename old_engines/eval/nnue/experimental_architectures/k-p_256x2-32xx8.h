// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義

#include "../features/feature_set.h"
#include "../features/k.h"
#include "../features/p.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/clipped_relu.h"

namespace Eval {

namespace NNUE {

// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<Features::K, Features::P>;

// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 256;

namespace Layers {

// ネットワーク構造の定義
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = ClippedReLU<AffineTransform<InputLayer, 32>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using HiddenLayer3 = ClippedReLU<AffineTransform<HiddenLayer2, 32>>;
using HiddenLayer4 = ClippedReLU<AffineTransform<HiddenLayer3, 32>>;
using HiddenLayer5 = ClippedReLU<AffineTransform<HiddenLayer4, 32>>;
using HiddenLayer6 = ClippedReLU<AffineTransform<HiddenLayer5, 32>>;
using HiddenLayer7 = ClippedReLU<AffineTransform<HiddenLayer6, 32>>;
using HiddenLayer8 = ClippedReLU<AffineTransform<HiddenLayer7, 32>>;
using HiddenLayer9 = ClippedReLU<AffineTransform<HiddenLayer8, 32>>;
using OutputLayer = AffineTransform<HiddenLayer9, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace NNUE

}  // namespace Eval
