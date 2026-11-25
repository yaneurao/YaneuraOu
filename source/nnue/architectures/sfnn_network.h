// Common SFNN without PSQT network implementation.
// Architecture headers should only select types and constants.

#ifndef NNUE_SFNN_NETWORK_H_INCLUDED
#define NNUE_SFNN_NETWORK_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
#include "sfnn_network_16.h"
#endif
#include "../layers/affine_transform_common_shard_input_explicit.h"
#include "../layers/affine_transform_sparse_input.h"
#include "../layers/clipped_relu.h"
#include "../layers/sqr_clipped_relu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace YaneuraOu::Eval::NNUE {

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims, bool PackedTail>
struct SfnnNetworkBuffer;

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(CacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, false> {
	alignas(CacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(CacheLineSize) typename Layers::SqrClippedReLU<Hidden1Dims>::OutputType
	    ac_sqr_0_out[ceil_to_multiple<IndexType>(Hidden1Dims * 2, 32)];
	alignas(CacheLineSize) typename Layers::AffineTransform<Hidden1Dims * 2, Hidden2Dims>::OutputBuffer fc_1_out;
	alignas(CacheLineSize) typename Layers::ClippedReLU<Hidden2Dims>::OutputBuffer ac_1_out;
	alignas(CacheLineSize) typename Layers::AffineTransform<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType Hidden1Dims, IndexType Hidden2Dims>
struct alignas(CacheLineSize) SfnnNetworkBuffer<Fc0Layer, Hidden1Dims, Hidden2Dims, true> {
	alignas(CacheLineSize) typename Fc0Layer::OutputBuffer fc_0_out;
	alignas(CacheLineSize) typename Layers::AffineTransform<Hidden2Dims, 1>::OutputBuffer fc_2_out;
};

template <typename Fc0Layer, IndexType InputDims, IndexType Hidden1Dims, IndexType Hidden2Dims,
          bool UseShortcut = (Hidden1Dims % 8 == 7)>
struct SfnnNetwork {
	Fc0Layer fc_0;
	Layers::ClippedReLU<Hidden1Dims> ac_0;
	Layers::SqrClippedReLU<Hidden1Dims> ac_sqr_0;
	Layers::AffineTransform<Hidden1Dims * 2, Hidden2Dims> fc_1;
	Layers::ClippedReLU<Hidden2Dims> ac_1;
	Layers::AffineTransform<Hidden2Dims, 1> fc_2;

	using OutputType = i32;
	static constexpr IndexType OutputDimensions = 1;
	static constexpr IndexType TransformedFeatureDimensions = InputDims;
	static constexpr IndexType kHidden1Dims = Hidden1Dims;
	static constexpr IndexType FC_1_OUTPUTS = Hidden2Dims;
	static constexpr bool kUseShortcut = UseShortcut;
	// Stockfishと同じくfc_0の物理的な出力数。ショートカットがある場合はその1出力も含む。
	static constexpr IndexType FC_0_OUTPUTS = kHidden1Dims + (kUseShortcut ? 1 : 0);

	static_assert(kHidden1Dims % 8 == 0 || kHidden1Dims % 8 == 7,
	              "SFNN H1 must be 8n without shortcut, or 8n-1 with shortcut.");
	static_assert(kUseShortcut == (kHidden1Dims % 8 == 7),
	              "SFNN shortcut is enabled only when H1 is 8n-1.");

#if defined(USE_AVX512)
	static constexpr bool kUsePackedTail = FC_1_OUTPUTS == 64;
#else
	static constexpr bool kUsePackedTail = false;
#endif

	using Buffer = SfnnNetworkBuffer<Fc0Layer, kHidden1Dims, FC_1_OUTPUTS, kUsePackedTail>;
	static constexpr usize BufferSize = sizeof(Buffer);

	static constexpr u32 get_hash_value() {
		return 0x6333718Au;
	}

	bool read_parameters(std::istream& stream) {
		return fc_0.read_parameters(stream)
			&& ac_0.read_parameters(stream)
			&& ac_sqr_0.read_parameters(stream)
			&& fc_1.read_parameters(stream)
			&& ac_1.read_parameters(stream)
			&& fc_2.read_parameters(stream);
	}

	bool write_parameters(std::ostream& stream) const {
		return fc_0.write_parameters(stream)
			&& ac_0.write_parameters(stream)
			&& ac_sqr_0.write_parameters(stream)
			&& fc_1.write_parameters(stream)
			&& ac_1.write_parameters(stream)
			&& fc_2.write_parameters(stream);
	}

	static typename decltype(ac_0)::OutputType clipped_relu_value(i32 value) {
		const auto shifted = value >> WeightScaleBits;
		if (shifted <= 0)
			return 0;
		if (shifted >= 127)
			return 127;
		return static_cast<typename decltype(ac_0)::OutputType>(shifted);
	}

	static typename decltype(ac_sqr_0)::OutputType sqr_clipped_relu_value(i32 value) {
		const auto sqr = (static_cast<long long>(value) * value) >> (2 * WeightScaleBits + 7);
		return static_cast<typename decltype(ac_sqr_0)::OutputType>(sqr >= 127 ? 127 : sqr);
	}

	void make_hidden1_input(const typename decltype(fc_0)::OutputType* input,
	                      typename decltype(ac_sqr_0)::OutputType* output) const {
		ac_sqr_0.propagate_pair(input, output, output + kHidden1Dims);
		std::fill(output + kHidden1Dims * 2,
		          output + ceil_to_multiple<IndexType>(kHidden1Dims * 2, 32),
		          typename decltype(ac_sqr_0)::OutputType{0});
	}

	static void make_hidden1_input_packed(const typename decltype(fc_0)::OutputType* input,
	                                   u32* output) {
		constexpr IndexType kPackedWords = ceil_to_multiple<IndexType>(kHidden1Dims * 2, 8) / 4;
		if constexpr (kHidden1Dims == 7) {
			const auto s0 = static_cast<u32>(sqr_clipped_relu_value(input[0]));
			const auto s1 = static_cast<u32>(sqr_clipped_relu_value(input[1]));
			const auto s2 = static_cast<u32>(sqr_clipped_relu_value(input[2]));
			const auto s3 = static_cast<u32>(sqr_clipped_relu_value(input[3]));
			const auto s4 = static_cast<u32>(sqr_clipped_relu_value(input[4]));
			const auto s5 = static_cast<u32>(sqr_clipped_relu_value(input[5]));
			const auto s6 = static_cast<u32>(sqr_clipped_relu_value(input[6]));
			const auto c0 = static_cast<u32>(clipped_relu_value(input[0]));
			const auto c1 = static_cast<u32>(clipped_relu_value(input[1]));
			const auto c2 = static_cast<u32>(clipped_relu_value(input[2]));
			const auto c3 = static_cast<u32>(clipped_relu_value(input[3]));
			const auto c4 = static_cast<u32>(clipped_relu_value(input[4]));
			const auto c5 = static_cast<u32>(clipped_relu_value(input[5]));
			const auto c6 = static_cast<u32>(clipped_relu_value(input[6]));
			output[0] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
			output[1] = s4 | (s5 << 8) | (s6 << 16) | (c0 << 24);
			output[2] = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24);
			output[3] = c5 | (c6 << 8);
		} else {
			std::memset(output, 0, kPackedWords * sizeof(u32));
			auto bytes = reinterpret_cast<typename decltype(ac_sqr_0)::OutputType*>(output);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[i] = sqr_clipped_relu_value(input[i]);
			for (IndexType i = 0; i < kHidden1Dims; ++i)
				bytes[kHidden1Dims + i] = clipped_relu_value(input[i]);
		}
	}

	template <typename BufferType>
	const OutputType* propagate_tail(BufferType& buf) const {
		if constexpr (kUsePackedTail) {
			u32 ac_sqr_0_packed[ceil_to_multiple<IndexType>(kHidden1Dims * 2, 8) / 4];
			make_hidden1_input_packed(buf.fc_0_out, ac_sqr_0_packed);
			fc_1.propagate_clipped_relu_packed_to_output(ac_sqr_0_packed, fc_2, buf.fc_2_out);
		} else {
			make_hidden1_input(buf.fc_0_out, buf.ac_sqr_0_out);
			fc_1.propagate(buf.ac_sqr_0_out, buf.fc_1_out);
			ac_1.propagate(buf.fc_1_out, buf.ac_1_out);
			fc_2.propagate(buf.ac_1_out, buf.fc_2_out);
		}

		if constexpr (kUseShortcut)
			buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims];
		return buf.fc_2_out;
	}

	const OutputType* propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
		fc_0.propagate(transformedFeatures, buf.fc_0_out);
		return propagate_tail(buf);
	}

#if defined(USE_AVX512)
	template <typename AccumulationType>
	const OutputType* propagate_from_accumulator(const AccumulationType& accumulation,
	                                           Color sideToMove,
	                                           char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);
#if defined(NNUE_SFNN_FUSED_FT)
		if constexpr (FC_0_OUTPUTS == 16
		              && std::is_same_v<Fc0Layer, Layers::AffineTransformSparseInput<TransformedFeatureDimensions, 16>>)
			fc_0.template propagate_sfnn16_from_accumulator<TransformedFeatureDimensions>(accumulation, sideToMove, buf.fc_0_out);
		else
			fc_0.template propagate_sfnn_from_accumulator<TransformedFeatureDimensions>(accumulation, sideToMove, buf.fc_0_out);
#else
		fc_0.template propagate_sfnn_from_accumulator<TransformedFeatureDimensions>(accumulation, sideToMove, buf.fc_0_out);
#endif
		return propagate_tail(buf);
	}
#endif
};

} // namespace YaneuraOu::Eval::NNUE

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // NNUE_SFNN_NETWORK_H_INCLUDED
