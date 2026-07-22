// Definition of layer AffineTransformCommonShardInputExplicit of NNUE evaluation function
// SFNN common+shard L1用。全出力がcommonを見る一方、各出力groupは対応するshardだけを見る。

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_COMMON_SHARD_INPUT_EXPLICIT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_COMMON_SHARD_INPUT_EXPLICIT_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include "simd.h"

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

// AffineTransform layer for common+shard dense input (Explicit Dimensions).
//
// The on-disk nn.bin layout is kept compatible with the ordinary SFNN fc_0:
//   bias[OutputDimensions], weight[OutputDimensions][pad(InputDimensions)].
// During ReadParameters(), only common weights and the selected shard weights
// for each output are kept in memory.
template <IndexType InputDimensions,
          IndexType OutputDimensions,
          IndexType CommonDimensions,
          IndexType ShardDimensions,
          IndexType ShardGroupCount>
class AffineTransformCommonShardInputExplicit {
   public:
	using InputType  = std::uint8_t;
	using OutputType = std::int32_t;

	static constexpr IndexType kInputDimensions        = InputDimensions;
	static constexpr IndexType kOutputDimensions       = OutputDimensions;
	static constexpr IndexType kCommonDimensions       = CommonDimensions;
	static constexpr IndexType kShardDimensions        = ShardDimensions;
	static constexpr IndexType kShardGroupCount        = ShardGroupCount;
	static constexpr IndexType kShardTotalDimensions   = kShardDimensions * kShardGroupCount;
	static constexpr IndexType kEffectiveInputDimensions = kCommonDimensions + kShardDimensions;
	static constexpr IndexType kGroupOutputDimensions  = kOutputDimensions / kShardGroupCount;
	static constexpr IndexType kPaddedInputDimensions  = CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);
	static constexpr IndexType kPaddedEffectiveInputDimensions =
	    CeilToMultiple<IndexType>(kEffectiveInputDimensions, kMaxSimdWidth);
	static constexpr IndexType kPaddedOutputDimensions =
	    CeilToMultiple<IndexType>(kOutputDimensions, kMaxSimdWidth);

	static_assert(kShardGroupCount > 1, "Common+shard affine requires more than one shard group.");
	static_assert(kShardDimensions > 0, "ShardDimensions must be positive.");
	static_assert(kCommonDimensions + kShardTotalDimensions == kInputDimensions,
	              "CommonDimensions + ShardDimensions * ShardGroupCount must be InputDimensions.");
	static_assert(kOutputDimensions % kShardGroupCount == 0,
	              "OutputDimensions must be divisible by ShardGroupCount.");
	static_assert(kCommonDimensions % 64 == 0, "CommonDimensions must be a multiple of 64.");
	static_assert(kShardDimensions % 64 == 0, "ShardDimensions must be a multiple of 64.");

	using OutputBuffer = OutputType[kPaddedOutputDimensions];

	static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
		std::uint32_t hash_value = 0xCC03DAE4u;
		hash_value += kOutputDimensions;
		hash_value ^= prevHash >> 1;
		hash_value ^= prevHash << 31;
		return hash_value;
	}

	static std::string GetStructureString() {
		return "AffineTransformCommonShardInput[" + std::to_string(kOutputDimensions) + "<-"
		       + std::to_string(kInputDimensions) + ",c" + std::to_string(kCommonDimensions)
		       + ",s" + std::to_string(kShardDimensions) + "x"
		       + std::to_string(kShardGroupCount) + "]";
	}

	Tools::Result ReadParameters(std::istream& stream) {
		for (std::size_t i = 0; i < kOutputDimensions; ++i)
			biases_[i] = read_little_endian<BiasType>(stream);

		WeightType zero{};
		for (std::size_t i = 0; i < kOutputDimensions * kPaddedEffectiveInputDimensions; ++i)
			weights_[i] = zero;

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			const IndexType group      = out / kGroupOutputDimensions;
			const IndexType shard_base = kCommonDimensions + group * kShardDimensions;
			const IndexType weight_base = out * kPaddedEffectiveInputDimensions;

			for (IndexType in = 0; in < kPaddedInputDimensions; ++in) {
				const WeightType w = read_little_endian<WeightType>(stream);
				if (in < kCommonDimensions) {
					weights_[weight_base + in] = w;
				} else if (shard_base <= in && in < shard_base + kShardDimensions) {
					weights_[weight_base + kCommonDimensions + (in - shard_base)] = w;
				}
			}
		}

		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kOutputDimensions * sizeof(BiasType));

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			const IndexType group       = out / kGroupOutputDimensions;
			const IndexType shard_base  = kCommonDimensions + group * kShardDimensions;
			const IndexType weight_base = out * kPaddedEffectiveInputDimensions;

			for (IndexType in = 0; in < kPaddedInputDimensions; ++in) {
				WeightType w{};
				if (in < kCommonDimensions) {
					w = weights_[weight_base + in];
				} else if (shard_base <= in && in < shard_base + kShardDimensions) {
					w = weights_[weight_base + kCommonDimensions + (in - shard_base)];
				}
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
		static_assert((HalfDimensions / 2) % 64 == 0);

		const __m512i zero = _mm512_setzero_si512();
		const __m512i one  = _mm512_set1_epi16(127 * 2);
		const Color perspectives[2] = {sideToMove, ~sideToMove};
		constexpr IndexType kChunksPerPerspective = (HalfDimensions / 2) / 64;
		constexpr int shift =
#if defined(USE_SSE2)
		    7;
#else
		    6;
#endif

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			__m512i sum = _mm512_setzero_si512();
			const auto weight_vectors =
			    reinterpret_cast<const __m512i*>(weights_ + out * kPaddedEffectiveInputDimensions);

			for (IndexType j = 0; j < kCommonDimensions / 64; ++j) {
				const __m512i transformed =
				    transform_accumulator_chunk<HalfDimensions>(accumulation, perspectives,
				                                                zero, one, shift, j,
				                                                kChunksPerPerspective);
				Simd::m512_add_dpbusd_epi32(sum, transformed, weight_vectors[j]);
			}

			const IndexType group = out / kGroupOutputDimensions;
			const IndexType shard_chunk_base = (kCommonDimensions + group * kShardDimensions) / 64;
			const IndexType weight_chunk_base = kCommonDimensions / 64;
			for (IndexType j = 0; j < kShardDimensions / 64; ++j) {
				const __m512i transformed =
				    transform_accumulator_chunk<HalfDimensions>(accumulation, perspectives,
				                                                zero, one, shift,
				                                                shard_chunk_base + j,
				                                                kChunksPerPerspective);
				Simd::m512_add_dpbusd_epi32(sum, transformed, weight_vectors[weight_chunk_base + j]);
			}

			output[out] = Simd::m512_hadd(sum, biases_[out]);
		}

		for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
			output[out] = OutputType{};
	}
#endif

	void Propagate(const InputType* input, OutputType* output) const {
#if defined(USE_AVX512)
		{
			const auto common_input_vectors = reinterpret_cast<const __m512i*>(input);

			for (IndexType out = 0; out < kOutputDimensions; ++out) {
				__m512i sum = _mm512_setzero_si512();
				const auto weight_vectors =
				    reinterpret_cast<const __m512i*>(weights_ + out * kPaddedEffectiveInputDimensions);

				for (IndexType j = 0; j < kCommonDimensions / 64; ++j)
					Simd::m512_add_dpbusd_epi32(sum, common_input_vectors[j], weight_vectors[j]);

				const IndexType group = out / kGroupOutputDimensions;
				const auto shard_input_vectors =
				    reinterpret_cast<const __m512i*>(input + kCommonDimensions + group * kShardDimensions);
				const IndexType weight_chunk_base = kCommonDimensions / 64;
				for (IndexType j = 0; j < kShardDimensions / 64; ++j)
					Simd::m512_add_dpbusd_epi32(sum, shard_input_vectors[j], weight_vectors[weight_chunk_base + j]);

				output[out] = Simd::m512_hadd(sum, biases_[out]);
			}

			for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
				output[out] = OutputType{};
			return;
		}
#endif

		for (IndexType out = 0; out < kOutputDimensions; ++out) {
			const IndexType group = out / kGroupOutputDimensions;
			const IndexType shard_base = kCommonDimensions + group * kShardDimensions;
			const IndexType weight_base = out * kPaddedEffectiveInputDimensions;
			OutputType sum = biases_[out];

			for (IndexType in = 0; in < kCommonDimensions; ++in)
				sum += OutputType(input[in]) * OutputType(weights_[weight_base + in]);

			for (IndexType in = 0; in < kShardDimensions; ++in)
				sum += OutputType(input[shard_base + in])
				       * OutputType(weights_[weight_base + kCommonDimensions + in]);

			output[out] = sum;
		}

		for (IndexType out = kOutputDimensions; out < kPaddedOutputDimensions; ++out)
			output[out] = OutputType{};
	}

   private:
	using BiasType   = OutputType;
	using WeightType = std::int8_t;

#if defined(USE_AVX512)
	template <IndexType HalfDimensions, typename AccumulationType>
	static __m512i transform_accumulator_chunk(const AccumulationType& accumulation,
	                                           const Color perspectives[2],
	                                           __m512i zero,
	                                           __m512i one,
	                                           int shift,
	                                           IndexType transformedChunk,
	                                           IndexType chunksPerPerspective) {
		const IndexType perspective_index = transformedChunk / chunksPerPerspective;
		const IndexType chunk = transformedChunk - perspective_index * chunksPerPerspective;
		const auto perspective = perspectives[perspective_index];
		const auto acc0 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][0]);
		const auto acc1 = reinterpret_cast<const __m512i*>(&accumulation[perspective][0][HalfDimensions / 2]);

		const __m512i sum0a =
		    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 0], one), zero), shift);
		const __m512i sum0b =
		    _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0[chunk * 2 + 1], one), zero), shift);
		const __m512i sum1a = _mm512_min_epi16(acc1[chunk * 2 + 0], one);
		const __m512i sum1b = _mm512_min_epi16(acc1[chunk * 2 + 1], one);
		const __m512i pa = _mm512_mulhi_epi16(sum0a, sum1a);
		const __m512i pb = _mm512_mulhi_epi16(sum0b, sum1b);
		return _mm512_packus_epi16(pa, pb);
	}
#endif

	alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
	alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedEffectiveInputDimensions];
};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_COMMON_SHARD_INPUT_EXPLICIT_H_INCLUDED
