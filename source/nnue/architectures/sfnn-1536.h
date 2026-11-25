// SFNN without PSQT 1536 architecture

#ifndef CLASSIC_NNUE_SFNN_1536_H_INCLUDED
#define CLASSIC_NNUE_SFNN_1536_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka_hm2.h"

#include "sfnn_network.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
    Features::HalfKA_hm2<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 1536;

// Number of networks stored in the evaluation file
constexpr int LayerStacks = 9;

// 各層の次元数
constexpr IndexType kInputDims   = kTransformedFeatureDimensions;
constexpr IndexType kHidden1Dims = 15;
constexpr IndexType kHidden2Dims = 32;

using Fc0Layer = Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1>;
using NetworkBase = SfnnNetwork<Fc0Layer, kInputDims, kHidden1Dims, kHidden2Dims>;

struct Network : NetworkBase {
	static std::string GetStructureString() {
		return "SFNN-1536";
	}
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif // CLASSIC_NNUE_SFNN_1536_H_INCLUDED
