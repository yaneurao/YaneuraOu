// Stockfishとのinclude構造・型名の互換性を維持するための共通入口。
// 評価関数の種類に応じて、やねうら王の実装・型の別名・ダミー実装を提供する。

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

#include "../config.h"
#include "../evaluate.h"

// 探索・Engine・評価実装で共用するネットワーク定義。
// 旧NNUE/SFNN以外では、重みを持たないダミーの参照口を提供する。
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
#include "../misc.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"

#if defined(EVAL_SFNN)
#define NNUE_SFNN_KING_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3 1
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9 2
#define NNUE_SFNN_KING_BUCKET_TYPE_K21K21 3
#define NNUE_SFNN_KING_BUCKET_TYPE_K29K29 4
#define NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z 5
#define NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z 6
#define NNUE_SFNN_KING_BUCKET_TYPE_K3K3Z 7

#define NNUE_SFNN_HAND_BUCKET_TYPE_NONE 0
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND4 1
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND16 2
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64 3
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND64Z 4
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND256 5
#define NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024 6

#ifndef NNUE_SFNN_HAND_BUCKETS
#define NNUE_SFNN_HAND_BUCKETS 1
#endif
#ifndef NNUE_SFNN_HAND_BUCKET_TYPE
#if NNUE_SFNN_HAND_BUCKETS == 1
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#elif NNUE_SFNN_HAND_BUCKETS == 4
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND4
#elif NNUE_SFNN_HAND_BUCKETS == 16
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND16
#elif NNUE_SFNN_HAND_BUCKETS == 64
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND64
#elif NNUE_SFNN_HAND_BUCKETS == 256
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND256
#elif NNUE_SFNN_HAND_BUCKETS == 1024
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024
#else
#define NNUE_SFNN_HAND_BUCKET_TYPE NNUE_SFNN_HAND_BUCKET_TYPE_NONE
#endif
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
// 入出力の結果型は宣言だけで足りる。結果を使う実装側でextra/tools.hを読み込む。
namespace Tools { struct Result; }
class Position;

namespace Eval::NNUE {

#if defined(ENABLE_SFNN_16BIT_WEIGHT)
#if !defined(EVAL_SFNN)
#error ENABLE_SFNN_16BIT_WEIGHT requires an SFNN architecture
#endif
	#define EvalFileDefaultName "nn16.bin"
#else
	#define EvalFileDefaultName "nn.bin"
#endif

#if defined(EVAL_SFNN) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress {

	// SFNNのLayerStack選択に使う進行度計算パラメーター。
	// EvalDir/progress.bin: headerless little-endian doubles, without bias.
	struct Parameters {
		static constexpr int kWeightCount = int(SQ_NB) * int(Eval::fe_end);

		Tools::Result read_parameters(std::istream& stream);

		i64 sum_q16(const Position& pos, AccumulatorState* state = nullptr) const;
		int bucket_index(const Position& pos, int bucket_count, AccumulatorState* state = nullptr) const;

		i32 weights_q16[SQ_NB][Eval::fe_end] = {};
	};

} // namespace Progress
#endif

#if defined(EVAL_SFNN)
	constexpr int kLayerStacks = LayerStacks;
#else
	constexpr int kLayerStacks = 1;
#endif

	// NNUE評価関数パラメーターを格納する統合構造体。
	// Stockfishと同じく、後段構成のNetworkArchitectureと重み全体のNetworkを区別する。
	// 全メンバーが生配列で構成されており trivially copyable であるため、
	// プロセス間共有メモリに直接配置できる。
	struct Network {
        // Hash value of evaluation function structure
        // 評価関数の構造のハッシュ値。既存の将棋用nn.binの値を維持する。
#if defined(EVAL_SFNN)
        static constexpr u32 hash = 0x3c203b32u;
#else
        static constexpr u32 hash =
          FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();
#endif

		FeatureTransformer featureTransformer;
#if defined(EVAL_SFNN) && NNUE_SFNN_PROGRESS_BUCKETS != 1
		Progress::Parameters progress;
#endif
		NetworkArchitecture network[kLayerStacks];

		// Content hash used to identify shared network parameters.
		// 共有するネットワークパラメーターを識別するための内容ハッシュ。
		usize get_content_hash() const {
			return static_cast<usize>(hash_bytes(reinterpret_cast<const char*>(this), sizeof(*this)));
		}
	};
	static_assert(std::is_trivially_copyable_v<Network>,
		"Network must be trivially copyable for shared memory support");

    // Stockfishの固定OutputScaleとは異なり、エンジンオプションで変更できる評価値の除数。
    extern int FV_SCALE;

	// 評価関数ファイル名
	extern const char* const kFileName;

	// 評価関数の構造を表す文字列を取得する
	std::string get_architecture_string();

	// ヘッダを読み込む
	Tools::Result read_header(std::istream& stream,
	    u32* hashValue, std::string* architecture, u32* version_out = nullptr);

	// ヘッダを書き込む
	bool write_header(std::ostream& stream,
	    u32 hashValue, const std::string& architecture);

	// 評価関数パラメータを読み込む
	// On a sidecar read failure, replace failed_file with its path.
    // 補助ファイルの読み込みに失敗したとき、failed_fileにそのパスを設定する。
    Tools::Result read_parameters(LazyNumaReplicatedSystemWide<Network>& network,
                                const OptionsMap& options, std::istream& stream,
                                std::string* failed_file = nullptr);

	// 評価関数パラメータを書き込む
	bool write_parameters(const Network& network, std::ostream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

// Networkのコンテンツハッシュ。共有メモリの名前に使われる。
// 同一の評価関数パラメーターを持つプロセス同士で自動的にメモリが共有される。
template<>
struct std::hash<YaneuraOu::Eval::NNUE::Network> {
	YaneuraOu::usize operator()(const YaneuraOu::Eval::NNUE::Network& n) const noexcept {
		return n.get_content_hash();
	}
};
#else
namespace YaneuraOu::Eval {

// NNUEの重みを持たないエンジンでも、同じ形でネットワークを参照できるようにする。
class NetworkStorage {
   public:
    explicit NetworkStorage(NumaReplicationContext&) {}
    const NetworkStorage& operator[](const NumaReplicatedAccessToken&) const { return *this; }
};

} // namespace YaneuraOu::Eval
#endif

#endif
