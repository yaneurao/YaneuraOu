// header used in NNUE evaluation function
// NNUE評価関数で用いるheader

#ifndef CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED
#define CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
//#include "../../misc.h"
#include "../../memory.h"

namespace YaneuraOu {
namespace Eval::NNUE {

	#define EvalFileDefaultName "nn.bin"

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
#if defined(YANEURAOU_ENGINE_NNUE_SFNNwoP1536)
	constexpr std::uint32_t kHashValue = 0x3c203b32u;
#else
	constexpr std::uint32_t kHashValue =
	    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();
#endif

	// 入力特徴量変換器
	extern LargePagePtr<FeatureTransformer> feature_transformer;

	// 評価関数
#if defined(YANEURAOU_ENGINE_NNUE_SFNNwoP1536)
	constexpr int kLayerStacks = LayerStacks;
	extern AlignedPtr<Network> network[kLayerStacks];
#else
	constexpr int kLayerStacks = 1;
	extern AlignedPtr<Network> network;
#endif

	// 評価関数ファイル名
	extern const char* const kFileName;

	// 評価関数の構造を表す文字列を取得する
	std::string GetArchitectureString();

	// ヘッダを読み込む
	Tools::Result ReadHeader(std::istream& stream,
	    std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out = nullptr);

	// ヘッダを書き込む
	bool WriteHeader(std::ostream& stream,
	    std::uint32_t hash_value, const std::string& architecture);

	// 評価関数パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream);

	// 評価関数パラメータを書き込む
	bool WriteParameters(std::ostream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
