// header used in NNUE evaluation function
// NNUE評価関数で用いるheader

#ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
#define NNUE_EVALUATE_NNUE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
//#include "../../misc.h"
#include "../../memory.h"

// 評価関数のソースコードへの埋め込みをする時は、EVAL_EMBEDDINGをdefineして、
// ⇓この2つのシンボルを正しく定義するembedded_nnue.cppを書けば良い。
#if defined(EVAL_EMBEDDING)
	extern const char*  gEmbeddedNNUEData;
	extern const size_t gEmbeddedNNUESize;
#else
	const char   gEmbeddedNNUEData[1] = {0x0};
	const size_t gEmbeddedNNUESize = 1;
#endif

namespace Eval::NNUE {

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
	constexpr std::uint32_t kHashValue =
	    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();

	// 入力特徴量変換器
	extern LargePagePtr<FeatureTransformer> feature_transformer;

	// 評価関数
	extern AlignedPtr<Network> network;

	// 評価関数ファイル名
	extern const char* const kFileName;

	// 評価関数の構造を表す文字列を取得する
	std::string GetArchitectureString();

	// ヘッダを読み込む
	Tools::Result ReadHeader(std::istream& stream,
	    std::uint32_t* hash_value, std::string* architecture);

	// ヘッダを書き込む
	bool WriteHeader(std::ostream& stream,
	    std::uint32_t hash_value, const std::string& architecture);

	// 評価関数パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream);

	// 評価関数パラメータを書き込む
	bool WriteParameters(std::ostream& stream);

}  // namespace Eval::NNUE

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
