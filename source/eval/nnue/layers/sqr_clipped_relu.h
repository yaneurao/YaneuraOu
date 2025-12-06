// Definition of layer SqrClippedReLU of NNUE evaluation function
// NNUE評価関数の層SqrClippedReLUの定義

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"
#include <algorithm>
#include <cstdint>

namespace YaneuraOu {
namespace Eval::NNUE::Layers {

	// SqrClippedReLUレイヤー
	template <IndexType InputDimensions>
	class SqrClippedReLU {
	public:
		// Input/output type
		// 入出力の型
		using InputType = std::int32_t;
		using OutputType = std::uint8_t;

		// Number of input/output dimensions
		// 入出力の次元数
		static constexpr IndexType kInputDimensions = InputDimensions;
		static constexpr IndexType kOutputDimensions = kInputDimensions;
		static constexpr IndexType PaddedOutputDimensions =
			CeilToMultiple<IndexType>(kOutputDimensions, 32);

		using OutputBuffer = OutputType[PaddedOutputDimensions];

		// Hash value embedded in the evaluation file
		// 評価関数ファイルに埋め込むハッシュ値
		static constexpr std::uint32_t GetHashValue(std::uint32_t prevHash) {
			std::uint32_t hash_value = 0x538D24C7u;
			hash_value += prevHash;
			return hash_value;
		}

		// 入力層からこの層までの構造を表す文字列
		static std::string GetStructureString() {
			return "SqrClippedReLU[" +
				std::to_string(kOutputDimensions) + "]";
		}

		// Read network parameters
		// パラメータを読み込む
		Tools::Result ReadParameters(std::istream& /*stream*/) {
			return Tools::ResultCode::Ok;
		}

		// パラメータを書き込む
		bool WriteParameters(std::ostream& /*stream*/) const {
			return true;
		}

		// Forward propagation
		// 順伝播
		void Propagate(const InputType* input, OutputType* output) const {

#if defined(USE_SSE2)
			constexpr IndexType NumChunks = kInputDimensions / 16;
			static_assert(kWeightScaleBits == 6);

			const auto in = reinterpret_cast<const __m128i*>(input);
			const auto out = reinterpret_cast<__m128i*>(output);
			for (IndexType i = 0; i < NumChunks; ++i)
			{
				__m128i words0 =
					_mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
				__m128i words1 =
					_mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

				// We shift by WeightScaleBits * 2 = 12 and divide by 128
				// which is an additional shift-right of 7, meaning 19 in total.
				// MulHi strips the lower 16 bits so we need to shift out 3 more to match.
				words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
				words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

				_mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
			}
			constexpr IndexType Start = NumChunks * 16;
#else
			constexpr IndexType Start = 0;
#endif

			for (IndexType i = Start; i < kInputDimensions; ++i) {
				output[i] = static_cast<OutputType>(
					std::min(127ll, ((long long)(input[i]) * input[i]) >> (2 * kWeightScaleBits + 7)));
			}
		}

	private:
		friend class Trainer<SqrClippedReLU>;

	};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
