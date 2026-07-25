// Common SFNN without PSQT network implementation.
// Architecture headers should only select types and constants.

#ifndef CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED
#define CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "../layers/affine_transform_common_shard_input_explicit.h"
#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace YaneuraOu {
namespace Eval::NNUE {

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims, bool PackedTail>
struct SfnnNetworkBuffer;

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(kCacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, false> {
	alignas(kCacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(kCacheLineSize) typename Layers::SqrClippedReLU<Hidden1Dims + 1>::OutputType
	    ac_sqr_0_out[CeilToMultiple<IndexType>(Hidden1Dims * 2, 32)];
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden1Dims * 2, Hidden2Dims>::OutputBuffer fc_1_out;
	alignas(kCacheLineSize) typename Layers::ClippedReLUExplicit<Hidden2Dims>::OutputBuffer ac_1_out;
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(kCacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, true> {
	alignas(kCacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(kCacheLineSize) typename Layers::AffineTransformExplicit<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType InputDims, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct SfnnNetwork {
	Fc0Layer fc_0;
	Layers::ClippedReLUExplicit<Hidden1Dims + 1> ac_0;
	Layers::SqrClippedReLU<Hidden1Dims + 1> ac_sqr_0;
	Layers::AffineTransformExplicit<Hidden1Dims * 2, Hidden2Dims> fc_1;
	Layers::ClippedReLUExplicit<Hidden2Dims> ac_1;
	Layers::AffineTransformExplicit<Hidden2Dims, 1> fc_2;

	using OutputType = std::int32_t;
	static constexpr IndexType kOutputDimensions = 1;
	static constexpr IndexType kInputDims = InputDims;
	static constexpr IndexType kHidden1Dims = Hidden1Dims;
	static constexpr IndexType kHidden2Dims = Hidden2Dims;

#if defined(USE_AVX512)
	static constexpr bool kUsePackedTail = kHidden2Dims == 64;
#else
	static constexpr bool kUsePackedTail = false;
#endif

	using Buffer = SfnnNetworkBuffer<Fc0Layer, kHidden1Dims, kHidden2Dims, kUsePackedTail>;
	static constexpr std::size_t kBufferSize = sizeof(Buffer);

	static constexpr std::uint32_t GetHashValue() {
		return 0x6333718Au;
	}

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

	static typename decltype(ac_0)::OutputType ClippedReLUValue(std::int32_t value) {
		const auto shifted = value >> kWeightScaleBits;
		if (shifted <= 0)
			return 0;
		if (shifted >= 127)
			return 127;
		return static_cast<typename decltype(ac_0)::OutputType>(shifted);
	}

	static typename decltype(ac_sqr_0)::OutputType SqrClippedReLUValue(std::int32_t value) {
		const auto sqr = (static_cast<long long>(value) * value) >> (2 * kWeightScaleBits + 7);
		return static_cast<typename decltype(ac_sqr_0)::OutputType>(sqr >= 127 ? 127 : sqr);
	}

	void MakeHidden1Input(const typename decltype(fc_0)::OutputType* input,
	                      typename decltype(ac_sqr_0)::OutputType* output) const {
		ac_sqr_0.PropagatePair(input, output, output + kHidden1Dims);
		std::fill(output + kHidden1Dims * 2,
		          output + CeilToMultiple<IndexType>(kHidden1Dims * 2, 32),
		          typename decltype(ac_sqr_0)::OutputType{0});
	}

	static void MakeHidden1InputPacked(const typename decltype(fc_0)::OutputType* input,
	                                   std::uint32_t* output) {
		constexpr IndexType kPackedWords = CeilToMultiple<IndexType>(kHidden1Dims * 2, 8) / 4;
		if constexpr (kHidden1Dims == 7) {
			const auto s0 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[0]));
			const auto s1 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[1]));
			const auto s2 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[2]));
			const auto s3 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[3]));
			const auto s4 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[4]));
			const auto s5 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[5]));
			const auto s6 = static_cast<std::uint32_t>(SqrClippedReLUValue(input[6]));
			const auto c0 = static_cast<std::uint32_t>(ClippedReLUValue(input[0]));
			const auto c1 = static_cast<std::uint32_t>(ClippedReLUValue(input[1]));
			const auto c2 = static_cast<std::uint32_t>(ClippedReLUValue(input[2]));
			const auto c3 = static_cast<std::uint32_t>(ClippedReLUValue(input[3]));
			const auto c4 = static_cast<std::uint32_t>(ClippedReLUValue(input[4]));
			const auto c5 = static_cast<std::uint32_t>(ClippedReLUValue(input[5]));
			const auto c6 = static_cast<std::uint32_t>(ClippedReLUValue(input[6]));
			output[0] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
			output[1] = s4 | (s5 << 8) | (s6 << 16) | (c0 << 24);
			output[2] = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24);
			output[3] = c5 | (c6 << 8);
		} else {
			std::memset(output, 0, kPackedWords * sizeof(std::uint32_t));
			auto bytes = reinterpret_cast<typename decltype(ac_sqr_0)::OutputType*>(output);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[i] = SqrClippedReLUValue(input[i]);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[kHidden1Dims + i] = ClippedReLUValue(input[i]);
		}
	}

	template <typename BufferType>
	const OutputType* PropagateTail(BufferType& buf) const {
		if constexpr (kUsePackedTail) {
			std::uint32_t ac_sqr_0_packed[CeilToMultiple<IndexType>(kHidden1Dims * 2, 8) / 4];
			MakeHidden1InputPacked(buf.fc_0_out, ac_sqr_0_packed);
			fc_1.PropagateClippedReLUPackedToOutput(ac_sqr_0_packed, fc_2, buf.fc_2_out);
		} else {
			MakeHidden1Input(buf.fc_0_out, buf.ac_sqr_0_out);
			fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
			ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
			fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);
		}

		buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims];
		return buf.fc_2_out;
	}

	const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
		fc_0.Propagate(transformedFeatures, buf.fc_0_out);
		return PropagateTail(buf);
	}

#if defined(USE_AVX512)
	template <typename AccumulationType>
	const OutputType* PropagateFromAccumulator(const AccumulationType& accumulation,
	                                           Color sideToMove,
	                                           char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
		fc_0.template PropagateSfnnFromAccumulator<kInputDims>(accumulation, sideToMove, buf.fc_0_out);
		return PropagateTail(buf);
	}
#endif
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // CLASSIC_NNUE_SFNN_NETWORK_H_INCLUDED
