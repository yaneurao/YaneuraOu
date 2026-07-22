// Definition of layer AffineTransformGroupedInputExplicit of NNUE evaluation function
// SFNN grouped L1用。入力をgroupに分け、各出力は対応するgroupだけを見る。

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_GROUPED_INPUT_EXPLICIT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_GROUPED_INPUT_EXPLICIT_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "affine_transform.h" // For affine_transform_unaligned
#include "simd.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// AffineTransform layer for grouped dense input (Explicit Dimensions).
//
// The on-disk nn.bin layout is kept compatible with the ordinary SFNN fc_0:
//   bias[OutputDimensions], weight[OutputDimensions][pad(InputDimensions)].
// During ReadParameters(), only the in-group weights are kept in memory.
template <IndexType InputDimensions, IndexType OutputDimensions, IndexType GroupCount>
class AffineTransformGroupedInputExplicit {
   public:
	using InputType = std::uint8_t;
	using OutputType = std::int32_t;

	static constexpr IndexType kInputDimensions        = InputDimensions;
	static constexpr IndexType kOutputDimensions       = OutputDimensions;
	static constexpr IndexType kGroupCount             = GroupCount;
	static constexpr IndexType kGroupInputDimensions   = kInputDimensions / kGroupCount;
	static constexpr IndexType kGroupOutputDimensions  = kOutputDimensions / kGroupCount;
	static constexpr IndexType kPaddedInputDimensions  = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);
	static constexpr IndexType kPaddedGroupInputDimensions =
	    CeilToMultiple<IndexType>(kGroupInputDimensions, kMaxSimdWidth);
	static constexpr IndexType kPaddedOutputDimensions =
	    CeilToMultiple<IndexType>(kOutputDimensions, kMaxSimdWidth);

	static_assert(kGroupCount > 1, "Grouped affine requires more than one group.");
	static_assert(kInputDimensions % kGroupCount == 0, "InputDimensions must be divisible by GroupCount.");
	static_assert(kOutputDimensions % kGroupCount == 0, "OutputDimensions must be divisible by GroupCount.");
	static_assert(kGroupInputDimensions % kMaxSimdWidth == 0,
	              "Each input group must be aligned to SIMD width to avoid reading across groups.");

	using OutputBuffer = OutputType[kPaddedOutputDimensions];

	static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= prevHash >> 1;
		hash_value ^= prevHash << 31;
		return hash_value;
	}

	static std::string GetStructureString() {
		return "AffineTransformGroupedInput[" + std::to_string(kOutputDimensions) + "<-"
		       + std::to_string(kInputDimensions) + ",g" + std::to_string(kGroupCount) + "]";
	}

	Tools::Result ReadParameters(std::istream& stream) {
		for (std::size_t i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);

		WeightType zero{};
		for (std::size_t i = 0; i < kOutputDimensions * kPaddedGroupInputDimensions; ++i)
			weights_[i] = zero;

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			const IndexType group      = out / kGroupOutputDimensions;
			const IndexType local_out  = out - group * kGroupOutputDimensions;
			const IndexType input_base = group * kGroupInputDimensions;
			const IndexType weight_base =
			    (group * kGroupOutputDimensions + local_out) * kPaddedGroupInputDimensions;

			for (IndexType in = 0; in < kPaddedInputDimensions; ++in) {
				const WeightType w = read_little_endian<WeightType>(stream);
				if (input_base <= in && in < input_base + kGroupInputDimensions)
					weights_[weight_base + (in - input_base)] = w;
			}
		}

		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kOutputDimensions * sizeof(BiasType));

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			const IndexType group      = out / kGroupOutputDimensions;
			const IndexType local_out  = out - group * kGroupOutputDimensions;
			const IndexType input_base = group * kGroupInputDimensions;
			const IndexType weight_base =
			    (group * kGroupOutputDimensions + local_out) * kPaddedGroupInputDimensions;

			for (IndexType in = 0; in < kPaddedInputDimensions; ++in) {
				const WeightType w =
				    (input_base <= in && in < input_base + kGroupInputDimensions)
				        ? weights_[weight_base + (in - input_base)]
				        : WeightType{};
				stream.write(reinterpret_cast<const char*>(&w), sizeof(w));
			}
		}

		return !stream.fail();
	}

#if defined(USE_AVX512)
	template <IndexType HalfDimensions, typename AccumulationType>
	void PropagateSfnnFromAccumulator(const AccumulationType& accumulation,
	                                  Color sideToMove,
	                                  OutputType* output) const {
		static_assert(kInputDimensions == HalfDimensions);
		static_assert(kGroupCount % 2 == 0);
		static_assert(kGroupInputDimensions % 64 == 0);
		static_assert((HalfDimensions / 2) % 64 == 0);

		constexpr IndexType kGroupsPerPerspective = kGroupCount / 2;
		constexpr IndexType kNumChunksPerGroup    = kGroupInputDimensions / 64;

		const __m512i zero = _mm512_setzero_si512();
		const __m512i one  = _mm512_set1_epi16(127 * 2);
		const Color perspectives[2] = {sideToMove, ~sideToMove};
		constexpr int shift =
#if defined(USE_SSE2)
		    7;
#else
		    6;
#endif

		for (IndexType group = 0; group < kGroupCount; ++group) {
			const IndexType perspective_index = group / kGroupsPerPerspective;
			const IndexType group_in_perspective = group - perspective_index * kGroupsPerPerspective;
			const IndexType base_chunk = group_in_perspective * kNumChunksPerGroup;
			const auto perspective = perspectives[perspective_index];
			const auto acc0 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][0]);
			const auto acc1 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][HalfDimensions / 2]);

			for (IndexType local_out = 0; local_out < kGroupOutputDimensions; ++local_out) {
				__m512i sum = _mm512_setzero_si512();
				const auto weight_vectors = reinterpret_cast<const __m512i*>(
				    weights_ + (group * kGroupOutputDimensions + local_out) *
				                   kPaddedGroupInputDimensions);

				for (IndexType j = 0; j < kNumChunksPerGroup; ++j) {
					const IndexType chunk = base_chunk + j;
					const __m512i sum0a =
					    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 0], one), zero), shift);
					const __m512i sum0b =
					    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 1], one), zero), shift);
					const __m512i sum1a =
					    _mm512_min_epi16(acc1[chunk * 2 + 0], one);
					const __m512i sum1b =
					    _mm512_min_epi16(acc1[chunk * 2 + 1], one);
					const __m512i pa = _mm512_mulhi_epi16(sum0a, sum1a);
					const __m512i pb = _mm512_mulhi_epi16(sum0b, sum1b);
					const __m512i transformed = _mm512_packus_epi16(pa, pb);
					Simd::m512_add_dpbusd_epi32(sum, transformed, weight_vectors[j]);
				}

				const IndexType out = group * kGroupOutputDimensions + local_out;
				output[out] = Simd::m512_hadd(sum, biases_[out]);
			}
		}

		for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
			output[out] = OutputType{};
	}
#endif

	void Propagate(const InputType* input, OutputType* output) const {
#if defined(USE_AVX512)
		if constexpr (kGroupInputDimensions % 64 == 0) {
			constexpr IndexType kNumChunks = kGroupInputDimensions / 64;

			for (IndexType group = 0; group < kGroupCount; ++group) {
				__m512i acc[kGroupOutputDimensions];
				for (IndexType local_out = 0; local_out < kGroupOutputDimensions; ++local_out)
					acc[local_out] = _mm512_setzero_si512();

				const auto input_vector =
				    reinterpret_cast<const __m512i*>(input + group * kGroupInputDimensions);
				const __m512i* weight_vectors[kGroupOutputDimensions];
				for (IndexType local_out = 0; local_out < kGroupOutputDimensions; ++local_out)
					weight_vectors[local_out] = reinterpret_cast<const __m512i*>(
					    weights_ + (group * kGroupOutputDimensions + local_out) *
					                   kPaddedGroupInputDimensions);

				for (IndexType j = 0; j < kNumChunks; ++j) {
					const __m512i in = input_vector[j];
					for (IndexType local_out = 0; local_out < kGroupOutputDimensions; ++local_out)
						Simd::m512_add_dpbusd_epi32(acc[local_out], in, weight_vectors[local_out][j]);
				}

				for (IndexType local_out = 0; local_out < kGroupOutputDimensions; ++local_out) {
					const IndexType out = group * kGroupOutputDimensions + local_out;
					output[out] = Simd::m512_hadd(acc[local_out], biases_[out]);
				}
			}

			for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
				output[out] = OutputType{};
			return;
		}
#endif

		for (IndexType group = 0; group < kGroupCount; ++group) {
			affine_transform_unaligned<kGroupInputDimensions,
			                           kPaddedGroupInputDimensions,
			                           kGroupOutputDimensions>(
			    output + group * kGroupOutputDimensions,
			    weights_ + group * kGroupOutputDimensions * kPaddedGroupInputDimensions,
			    biases_ + group * kGroupOutputDimensions,
			    input + group * kGroupInputDimensions);
		}

		for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
			output[out] = OutputType{};
	}

   private:
	using BiasType   = OutputType;
	using WeightType = std::int8_t;

	alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
	alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedGroupInputDimensions];
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_GROUPED_INPUT_EXPLICIT_H_INCLUDED
