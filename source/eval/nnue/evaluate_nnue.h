// header used in NNUE evaluation function
// NNUE評価関数で用いるheader

#ifndef CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED
#define CLASSIC_NNUE_EVALUATE_NNUE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
#include "../../misc.h"
#include "../../memory.h"
#include "../../shm.h"

#if defined(SFNNwoPSQT)
#define NNUE_SFNN_KING_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3 1
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9 2
#define NNUE_SFNN_KING_BUCKET_TYPE_K21K21 3
#define NNUE_SFNN_KING_BUCKET_TYPE_K29K29 4
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z 5
#define NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z 6

#ifndef NNUE_SFNN_HAND_BUCKETS
#define NNUE_SFNN_HAND_BUCKETS 1
#endif
#ifndef NNUE_SFNN_KING_BUCKETS
#define NNUE_SFNN_KING_BUCKETS 9
#endif
#ifndef NNUE_SFNN_KING_BUCKET_TYPE
#if NNUE_SFNN_KING_BUCKETS == 9
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K3K3
#elif NNUE_SFNN_KING_BUCKETS == 81
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K9K9
#elif NNUE_SFNN_KING_BUCKETS == 169
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z
#elif NNUE_SFNN_KING_BUCKETS == 441
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K21K21
#elif NNUE_SFNN_KING_BUCKETS == 841
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_K29K29
#else
#define NNUE_SFNN_KING_BUCKET_TYPE NNUE_SFNN_KING_BUCKET_TYPE_NONE
#endif
#endif
#ifndef NNUE_SFNN_PROGRESS_BUCKETS
#define NNUE_SFNN_PROGRESS_BUCKETS 1
#endif
#endif

namespace YaneuraOu {
class Position;

namespace Eval::NNUE {

	#define EvalFileDefaultName "nn.bin"

#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress {

	// SFNNのLayerStack選択に使う進行度計算パラメーター。
	// nn.bin内ではFeatureTransformerの直後にこのセクションを置く。
	struct Parameters {
		static constexpr int kProgressValueCount = 256;
		static constexpr int kWeightCount = int(SQ_NB) * int(Eval::fe_end);

		static constexpr std::uint32_t GetHashValue() {
			return 0x6f50524fu; // "oPRO" : NNUE progress parameter section
		}

		Tools::Result ReadParameters(std::istream& stream);
		bool WriteParameters(std::ostream& stream) const;

		int Value0To255(const Position& pos) const;
		int BucketIndex(const Position& pos, int bucket_count) const;

		std::int32_t bias_q16_ = 0;
		std::int32_t weights_q16_[SQ_NB][Eval::fe_end] = {};
	};

} // namespace Progress
#endif

	// Hash value of evaluation function structure
	// 評価関数の構造のハッシュ値
#if defined(SFNNwoPSQT)
	constexpr std::uint32_t kSfnnBaseHashValue = 0x3c203b32u;
#if NNUE_SFNN_PROGRESS_BUCKETS != 1
	constexpr std::uint32_t kHashValue =
	    kSfnnBaseHashValue ^ Progress::Parameters::GetHashValue();
#else
	constexpr std::uint32_t kHashValue = kSfnnBaseHashValue;
#endif
	constexpr int kLayerStacks = LayerStacks;
#else
	constexpr std::uint32_t kHashValue =
	    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();
	constexpr int kLayerStacks = 1;
#endif

	// NNUE評価関数パラメーターを格納する統合構造体。
	// 全メンバーが生配列で構成されており trivially copyable であるため、
	// プロセス間共有メモリに直接配置できる。
	struct NnueNetworks {
		FeatureTransformer feature_transformer;
#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
		Progress::Parameters progress;
#endif
		Network network[kLayerStacks];
	};
	static_assert(std::is_trivially_copyable_v<NnueNetworks>,
		"NnueNetworks must be trivially copyable for shared memory support");

	// NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
	extern SystemWideSharedConstant<NnueNetworks> shared_networks;

	// 共有メモリ上のNnueNetworksへのconst参照を返すヘルパー。
	// 評価関数の呼び出しで毎回使われるので、インライン化する。
	inline const NnueNetworks& networks() { return *shared_networks; }

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

// NnueNetworks のコンテンツハッシュ。共有メモリの名前に使われる。
// 同一の評価関数パラメーターを持つプロセス同士で自動的にメモリが共有される。
template<>
struct std::hash<YaneuraOu::Eval::NNUE::NnueNetworks> {
	std::size_t operator()(const YaneuraOu::Eval::NNUE::NnueNetworks& n) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&n), sizeof(n)));
	}
};

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_EVALUATE_NNUE_H_INCLUDED
