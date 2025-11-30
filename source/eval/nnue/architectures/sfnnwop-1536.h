// SFNN without PSQT 1536 architecture

#ifndef CLASSIC_NNUE_SFNNWOP_1536_H_INCLUDED
#define CLASSIC_NNUE_SFNNWOP_1536_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka_hm.h"

#include <cstring>

#include "../layers/input_slice.h"
#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 評価関数で用いる入力特徴量
using RawFeatures = Features::FeatureSet<
    Features::HalfKA_hm<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 1536;

// Number of networks stored in the evaluation file
constexpr int LayerStacks = 9;

namespace Layers {

// Define network structure
// ネットワーク構造の定義
using InputLayer = InputSlice<kTransformedFeatureDimensions>;
using FC0 = AffineTransformSparseInputExplicit<kTransformedFeatureDimensions, 15 + 1>;
using AC0 = ClippedReLUExplicit<15 + 1>;
using AC0Sqr = SqrClippedReLU<15 + 1>;
using FC1 = AffineTransformExplicit<15 * 2, 32>;
using AC1 = ClippedReLUExplicit<32>;
using FC2 = AffineTransformExplicit<32, 1>;

}  // namespace Layers

// 評価ファイルに埋め込まれているハッシュ値に合わせるラッパー
struct Network {
	using OutputType = std::int32_t;
	static constexpr IndexType kOutputDimensions = 1;

	struct alignas(kCacheLineSize) Buffer {
		alignas(kCacheLineSize) typename Layers::FC0::OutputBuffer fc_0_out;
		alignas(kCacheLineSize) typename Layers::AC0Sqr::OutputType ac_sqr_0_out[CeilToMultiple<IndexType>(15 * 2, 32)];
		alignas(kCacheLineSize) typename Layers::AC0::OutputBuffer ac_0_out;
		alignas(kCacheLineSize) typename Layers::FC1::OutputBuffer fc_1_out;
		alignas(kCacheLineSize) typename Layers::AC1::OutputBuffer ac_1_out;
		alignas(kCacheLineSize) typename Layers::FC2::OutputBuffer fc_2_out;
	};

	static constexpr std::size_t kBufferSize = sizeof(Buffer);

	static constexpr std::uint32_t GetHashValue() { return 0x6333718Au; }
	static std::string GetStructureString() { return "SFNN-1536"; }

	Tools::Result ReadParameters(std::istream& stream) {
		bool result = fc_0.ReadParameters(stream).is_ok()
			&& ac_0.ReadParameters(stream).is_ok()
			&& ac_sqr_0.ReadParameters(stream).is_ok()
			&& fc_1.ReadParameters(stream).is_ok()
			&& ac_1.ReadParameters(stream).is_ok()
			&& fc_2.ReadParameters(stream).is_ok();
		return result ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	bool WriteParameters(std::ostream& stream) const {
		return fc_0.WriteParameters(stream)
			&& ac_0.WriteParameters(stream)
			&& ac_sqr_0.WriteParameters(stream)
			&& fc_1.WriteParameters(stream)
			&& ac_1.WriteParameters(stream)
			&& fc_2.WriteParameters(stream);
	}

	const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);

		fc_0.Propagate(transformedFeatures, buf.fc_0_out);
		ac_sqr_0.Propagate(buf.fc_0_out, buf.ac_sqr_0_out);
		ac_0.Propagate(buf.fc_0_out, buf.ac_0_out);
		std::memcpy(buf.ac_sqr_0_out + 15, buf.ac_0_out,
			15 * sizeof(typename Layers::AC0::OutputType));
		fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
		ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
		fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);

		// add shortcut term
		buf.fc_2_out[0] += buf.fc_0_out[15];

		return buf.fc_2_out;
	}

private:
	Layers::FC0 fc_0;
	Layers::AC0 ac_0;
	Layers::AC0Sqr ac_sqr_0;
	Layers::FC1 fc_1;
	Layers::AC1 ac_1;
	Layers::FC2 fc_2;
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif  // CLASSIC_NNUE_SFNNWOP_1536_H_INCLUDED
