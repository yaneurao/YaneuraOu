// NNUE評価関数で用いる入力特徴量とネットワーク構造の定義

#ifndef NNUE_HALFKPE9_256X2_32_32_H_INCLUDED
#define NNUE_HALFKPE9_256X2_32_32_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kpe9.h"

#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/clipped_relu.h"

namespace Eval {

namespace NNUE {

	// 評価関数で用いる入力特徴量
	using RawFeatures = Features::FeatureSet<
	    Features::HalfKPE9<Features::Side::kFriend>>;
	
	// 変換後の入力特徴量の次元数
	constexpr IndexType kTransformedFeatureDimensions = 256;
	
	namespace Layers {
	
	// ネットワーク構造の定義
	using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
	using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 32>>;
	using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
	using OutputLayer = AffineTransform<HiddenLayer2, 1>;

}  // namespace Layers

using Network = Layers::OutputLayer;

}  // namespace NNUE

}  // namespace Eval

#endif // #ifndef NNUE_HALFKPE9_256X2_32_32_H_INCLUDED
