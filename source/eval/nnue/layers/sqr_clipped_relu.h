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
			if constexpr (PaddedOutputDimensions > kOutputDimensions) {
				std::fill(output + kOutputDimensions, output + PaddedOutputDimensions, OutputType{0});
			}
		}

		// SqrClippedReLUとClippedReLUをまとめて計算する。
		// SFNNの後段入力は [sqr(input[0..n-2]), clip(input[0..n-2])] という連結なので、
		// 1回の入力変換から両方の出力を作る。
		void PropagatePair(const InputType* input, OutputType* sqrOut, OutputType* clipOut) const {
			static_assert(kWeightScaleBits == 6);
			[[maybe_unused]] constexpr int SimdShiftAmount = 2 * kWeightScaleBits + 7 - 16;

#if defined(USE_AVX512)
			if constexpr (kInputDimensions == 16)
			{
				const __m512i x = _mm512_load_si512(reinterpret_cast<const __m512i*>(input));
				const __m256i w = _mm512_cvtsepi32_epi16(x);

				const __m256i sq = _mm256_srli_epi16(_mm256_mulhi_epi16(w, w), SimdShiftAmount);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(sqrOut), _mm256_cvtsepi16_epi8(sq));

				const __m256i cl = _mm256_srli_epi16(
					_mm256_max_epi16(w, _mm256_setzero_si256()), kWeightScaleBits);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(clipOut), _mm256_cvtsepi16_epi8(cl));
				return;
			}
#endif
#if defined(USE_AVX2)
			if constexpr (kInputDimensions == 16)
			{
				const auto in = reinterpret_cast<const __m256i*>(input);
				__m256i w = _mm256_packs_epi32(_mm256_load_si256(&in[0]), _mm256_load_si256(&in[1]));
				w         = _mm256_permute4x64_epi64(w, 0b11011000);

				const __m256i sq = _mm256_srli_epi16(_mm256_mulhi_epi16(w, w), SimdShiftAmount);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(sqrOut),
				                 _mm_packs_epi16(_mm256_castsi256_si128(sq),
				                                 _mm256_extracti128_si256(sq, 1)));

				const __m256i cl = _mm256_srli_epi16(
					_mm256_max_epi16(w, _mm256_setzero_si256()), kWeightScaleBits);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(clipOut),
				                 _mm_packs_epi16(_mm256_castsi256_si128(cl),
				                                 _mm256_extracti128_si256(cl, 1)));
				return;
			}
#endif
#if defined(USE_SSE2)
			if constexpr (kInputDimensions % 8 == 0)
			{
				constexpr IndexType NumChunks = kInputDimensions / 8;
				const auto in = reinterpret_cast<const __m128i*>(input);

				// 出力領域は重なりうるので、sqr側をすべて書いてからclip側を書く。
				for (IndexType i = 0; i < NumChunks; ++i)
				{
					const __m128i w = _mm_packs_epi32(_mm_load_si128(&in[i * 2 + 0]),
					                                  _mm_load_si128(&in[i * 2 + 1]));
					const __m128i sq = _mm_srli_epi16(_mm_mulhi_epi16(w, w), SimdShiftAmount);
					_mm_storel_epi64(reinterpret_cast<__m128i*>(sqrOut + i * 8),
					                 _mm_packs_epi16(sq, sq));
				}
				for (IndexType i = 0; i < NumChunks; ++i)
				{
					const __m128i w = _mm_packs_epi32(_mm_load_si128(&in[i * 2 + 0]),
					                                  _mm_load_si128(&in[i * 2 + 1]));
					const __m128i cl = _mm_srli_epi16(_mm_max_epi16(w, _mm_setzero_si128()),
					                                  kWeightScaleBits);
					_mm_storel_epi64(reinterpret_cast<__m128i*>(clipOut + i * 8),
					                 _mm_packs_epi16(cl, cl));
				}
				return;
			}
#endif
			for (IndexType i = 0; i < kInputDimensions; ++i)
				sqrOut[i] = static_cast<OutputType>(
					std::min(127ll, ((long long)(input[i]) * input[i]) >> (2 * kWeightScaleBits + 7)));
			for (IndexType i = 0; i < kInputDimensions; ++i)
				clipOut[i] = static_cast<OutputType>(
					std::max(0, std::min(127, input[i] >> kWeightScaleBits)));
		}

	};

}  // namespace Eval::NNUE::Layers
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
