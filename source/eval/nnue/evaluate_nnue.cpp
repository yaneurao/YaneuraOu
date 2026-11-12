// NNUE評価関数の計算に関するコード

#include "../../config.h"
#include "../../extra/tools.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../../incbin/incbin.h"

#include "../../types.h"
#include "../../evaluate.h"
#include "../../position.h"
#include "../../memory.h"
#include "../../usi.h"

#if defined(USE_EVAL_HASH)
#include "../evalhash.h"
#endif

#include "evaluate_nnue.h"
#include "nnue_common.h"

namespace YaneuraOu::Eval::NNUE {
extern int FV_SCALE;
}
 
// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL) || defined(EVAL_SFNN)
using namespace YaneuraOu;
void add_options_(OptionsMap& options, ThreadPool& threads);

namespace {
YaneuraOu::OptionsMap* options_ptr;
YaneuraOu::ThreadPool* threads_ptr;
}

// 📌 旧Options、旧Threadsとの互換性のための共通のマクロ 📌
#define Options (*options_ptr)
#define Threads (*threads_ptr)

namespace YaneuraOu::Eval {
void add_options(OptionsMap& options, ThreadPool& threads) {
    options_ptr = &options;
    threads_ptr = &threads;
    add_options_(options, threads);
}
}
// ============================================================

// 評価関数を読み込み済みであるか
bool        eval_loaded   = false;
std::string last_eval_dir = "None";

// 📌 この評価関数で追加したいエンジンオプションはここで追加する。
void add_options_(OptionsMap& options, ThreadPool& threads) {

#if defined(NNUE_EMBEDDING_OFF)
    const char* default_eval_dir = "eval";
#else
	// メモリから読み込む。
    const char* default_eval_dir = "<internal>";
#endif
    Options.add("EvalDir", Option(default_eval_dir, [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    if (last_eval_dir != eval_dir)
                    {
                        // 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
                        last_eval_dir = eval_dir;
                        eval_loaded   = false;
                    }
                    return std::nullopt;
                }));

#if defined(__EMSCRIPTEN__)
    // 📝 wasmでは評価関数ファイル名をJS側から指定できるようにする。
    //     JS側はEmscriptenの仮想FSにファイルを書き出したあと、
    //     "setoption name EvalFile value ..."を送ってくる。
    // ⚠ load_eval()はwasmのとき、EvalDirが"<internal>"でもOptions["EvalFile"]を
    //     参照するので、埋め込み版でもこのOptionは必要。
    Options.add("EvalFile", Option(EvalFileDefaultName, [](const Option&) {
                    // 評価関数ファイル名の変更に際して、読み込みフラグをクリアする。
                    eval_loaded = false;
                    return std::nullopt;
                }));
#endif

    // NNUEのFV_SCALEの値
    Options.add("FV_SCALE", Option(16, 1, 128, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FV_SCALE = int(o);
                    return std::nullopt;
                }));

}
#endif

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.

// デフォルトの効率的に更新可能なニューラルネットワーク（NNUE）ファイルの
// データをエンジンのバイナリに埋め込むためのマクロ
// （Dale Weiler 氏の incbin.h を使用）。
// このマクロを使うことで、以下の3つの変数が宣言されます：
//     const unsigned char        gEmbeddedNNUEData[];  // 埋め込まれたデータへのポインタ
//     const unsigned char *const gEmbeddedNNUEEnd;     // データの終端を示すマーカー
//     const unsigned int         gEmbeddedNNUESize;    // 埋め込まれたファイルのサイズ
// なお、この方法は Microsoft Visual Studio では動作しません。

#if defined(__EMSCRIPTEN__) && !defined(NNUE_EMBEDDING_OFF)
// 📝 WebAssemblyではincbin.hは使えない。incbin.hはモジュールレベルのinline asmで
//     データシンボルを置くが、wasmはデータシンボルをtextセクションに置けないため
//     "data symbols must live in a data section"となる。-fltoではアセンブルが
//     リンク時まで遅延されるので、コンパイルは通りwasm-ldがSIGABRTで落ちる。
//     代わりに、パラメーターを文字列リテラルとして持つembedded_nnue.cpp
//     (tools/build/wasm_build.jsが生成)をリンクする。
extern const char*       gEmbeddedNNUEData;
extern const YaneuraOu::usize gEmbeddedNNUESize;
#elif !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#else
const unsigned char        gEmbeddedNNUEData[1] = { 0x0 };
const unsigned char* const gEmbeddedNNUEEnd = &gEmbeddedNNUEData[1];
const unsigned int         gEmbeddedNNUESize = 1;
#endif

// NNUEの埋め込みデータ型

namespace {

	struct EmbeddedNNUE {
		EmbeddedNNUE(const unsigned char* embeddedData,
			const unsigned char* embeddedEnd,
			const unsigned int   embeddedSize) :
			data(embeddedData),
			end(embeddedEnd),
			size(embeddedSize) {
		}
		const unsigned char* data;
		const unsigned char* end;
		const unsigned int   size;
	};

	EmbeddedNNUE get_embedded() {
#if defined(__EMSCRIPTEN__) && !defined(NNUE_EMBEDDING_OFF)
		// embedded_nnue.cppは先頭ポインタとサイズしか提供しないので、終端はここで計算する。
		const auto* data = reinterpret_cast<const unsigned char*>(gEmbeddedNNUEData);
		return EmbeddedNNUE(data, data + gEmbeddedNNUESize,
			static_cast<unsigned int>(gEmbeddedNNUESize));
#else
		return EmbeddedNNUE(gEmbeddedNNUEData, gEmbeddedNNUEEnd, gEmbeddedNNUESize);
#endif
	}
}


namespace YaneuraOu {
namespace Eval {
namespace NNUE {

	int FV_SCALE = 16; // 水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

#if defined(EVAL_SFNN) && NNUE_SFNN_PROGRESS_BUCKETS != 1
namespace Progress {
namespace {

	constexpr double kQ16Scale = 65536.0;
	constexpr int kProgressThresholdCount = NNUE_SFNN_PROGRESS_BUCKETS - 1;

	std::array<i64, kProgressThresholdCount> make_thresholds_q16() {
		std::array<i64, kProgressThresholdCount> thresholds{};
		for (int i = 1; i < NNUE_SFNN_PROGRESS_BUCKETS; ++i) {
			const double p = double(i) / double(NNUE_SFNN_PROGRESS_BUCKETS);
			const double scaled = std::round(std::log(p / (1.0 - p)) * kQ16Scale);
			const double clamped = std::clamp(
			    scaled,
			    double((std::numeric_limits<i64>::min)()),
			    double((std::numeric_limits<i64>::max)()));
			thresholds[i - 1] = i64(clamped);
		}
		return thresholds;
	}

	const std::array<i64, kProgressThresholdCount>& thresholds_q16() {
		static const auto thresholds = make_thresholds_q16();
		return thresholds;
	}

	int bucket_from_sum_q16(i64 sum_q16) {
		const auto& thresholds = thresholds_q16();
		const auto it = std::upper_bound(thresholds.begin(), thresholds.end(), sum_q16);
		return int(it - thresholds.begin());
	}

} // namespace

Tools::Result Parameters::ReadParameters(std::istream& stream) {
	static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
	static_assert(kWeightCount == 81 * 1548);
	for (auto& row : weights_q16_)
		for (auto& weight : row) {
			const auto bits = read_little_endian<std::uint64_t>(stream);
			if (stream.fail()) return Tools::ResultCode::FileReadError;
			double value;
			std::memcpy(&value, &bits, sizeof(value));
			// Check the IEEE exponent directly, also with -ffast-math enabled.
			if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
				return Tools::ResultCode::FileMismatch;
			weight = i32(std::clamp(std::round(value * kQ16Scale),
			    double((std::numeric_limits<i32>::min)()),
			    double((std::numeric_limits<i32>::max)())));
		}
	return stream.peek() == std::ios::traits_type::eof()
	    ? Tools::ResultCode::Ok : Tools::ResultCode::FileMismatch;
}

i64 Parameters::SumQ16(const Position& pos) const {
	const auto sq_bk = pos.square<KING>(BLACK);
	const auto sq_wk = Inv(pos.square<KING>(WHITE));

	auto* st = pos.state();
	i64 sum_q16 = 0;
	if (st->nnue_progress_valid
	    && st->nnue_progress_key == pos.key()
	    && st->nnue_progress_sq_bk == sq_bk
	    && st->nnue_progress_sq_wk == sq_wk) {
		sum_q16 = st->nnue_progress_sum;
	} else {
		const auto& list0 = pos.eval_list()->piece_list_fb();
		const auto& list1 = pos.eval_list()->piece_list_fw();

		for (int i = 0; i < PIECE_NUMBER_KING; ++i) {
			sum_q16 += weights_q16_[sq_bk][list0[i]];
			sum_q16 += weights_q16_[sq_wk][list1[i]];
		}

		st->nnue_progress_key = pos.key();
		st->nnue_progress_sum = sum_q16;
		st->nnue_progress_sq_bk = sq_bk;
		st->nnue_progress_sq_wk = sq_wk;
		st->nnue_progress_valid = true;
	}

	return sum_q16;
}

int Parameters::BucketIndex(const Position& pos, int bucket_count) const {
	if (bucket_count <= 1)
		return 0;

	ASSERT_LV1(bucket_count == NNUE_SFNN_PROGRESS_BUCKETS);
	return bucket_from_sum_q16(SumQ16(pos));
}

} // namespace Progress
#endif

    // NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
    SystemWideSharedConstant<NnueNetworks> shared_networks;

    // 評価関数ファイル名
    const char* const kFileName = EvalFileDefaultName;

#if defined(ENABLE_SFNN_16BIT_WEIGHT)
    static std::string Nn16Architecture(u32 qb) {
        return "ModelType=SFNNWithoutPsqt;Features=" + FeatureTransformer::GetStructureString()
            + ",Network=SFNN-" + std::to_string(Network::kInputDims)
            + "{LayerStack=" + std::to_string(kLayerStacks)
            + "};FCWeightBits=16;QB=" + std::to_string(qb)
            + ";H1=" + std::to_string(Network::kHidden1Dims)
            + ";Skip=" + std::to_string(int(Network::kUseShortcut))
            + ";H2=" + std::to_string(Network::kHidden2Dims);
    }
#endif

    // 評価関数の構造を表す文字列を取得する
    std::string GetArchitectureString() {
        const std::string base = "Features=" + FeatureTransformer::GetStructureString() +
			",Network=" + Network::GetStructureString();
#if defined(EVAL_SFNN)
		return "ModelType=SFNNWithoutPsqt;" + base + "{LayerStack=" + std::to_string(kLayerStacks) + "}";
#else
		return base;
#endif
    }

namespace {
	namespace Detail {

		// 評価関数パラメータを読み込む（参照版）
		template <typename T>
		Tools::Result ReadParameters(std::istream& stream, T& obj) {
			u32 header;
			stream.read(reinterpret_cast<char*>(&header), sizeof(header));
			if (!stream) return Tools::ResultCode::FileReadError;
			// hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。
			if (header != T::GetHashValue())
				sync_cout << "info string Warning : nn.bin hash mismatch." << sync_endl;
			return obj.ReadParameters(stream);
		}

		// 評価関数パラメータを書き込む（参照版）
		template <typename T>
		bool WriteParameters(std::ostream& stream, const T& obj) {
			constexpr u32 header = T::GetHashValue();
			stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
			return obj.WriteParameters(stream);
		}

	}  // namespace Detail

	// テンポラリにパラメータを読み込み、共有メモリに配置する。
	// 同じパラメータを持つ他プロセスが既に共有メモリを作成済みなら、そちらを参照する。
	Tools::Result LoadAndShare(std::istream& stream, [[maybe_unused]] std::string* failed_file) {
		// テンポラリ領域にパラメータを読み込む
		auto tmp = make_unique_large_page<NnueNetworks>();

		u32 hash_value;
		std::string architecture;
		Tools::Result result = ReadHeader(stream, &hash_value, &architecture, nullptr);
		if (result.is_not_ok()) return result;
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
        const auto qb = read_little_endian<u32>(stream);
        if (!stream) return Tools::ResultCode::FileReadError;
        for (auto& network : tmp->network)
            if (!network.SetWeightScale(qb)) {
                sync_cout << "info string Invalid nn16 QB: " << qb << sync_endl;
                return Tools::ResultCode::FileMismatch;
            }
        if (hash_value != kHashValue || architecture != Nn16Architecture(qb)) {
            sync_cout << "info string nn16 architecture mismatch: " << architecture
                      << " expected " << Nn16Architecture(qb) << sync_endl;
            return Tools::ResultCode::FileMismatch;
        }
#endif
		if (hash_value != kHashValue) {
			sync_cout << "info string Warning: NNUE hash mismatch: expected " << kHashValue
				<< " got " << hash_value
				<< " arch_in_file=" << architecture
				<< " arch_expected=" << GetArchitectureString()
				<< sync_endl;
		}

		result = Detail::ReadParameters<FeatureTransformer>(stream, tmp->feature_transformer);
		if (result.is_not_ok()) {
			sync_cout << "info string NNUE feature params read failed: " << result.to_string() << sync_endl;
			return result;
		}
#if defined(EVAL_SFNN) && NNUE_SFNN_PROGRESS_BUCKETS != 1
		const std::string eval_dir = Options["EvalDir"];
		const auto progress_path = CommandLine::get_binary_directory() / path_from_utf8(eval_dir) / "progress.bin";
		std::ifstream progress_stream(progress_path, std::ios::binary);
		sync_cout << "info string loading progress file : " << progress_path << sync_endl;
		result = progress_stream.is_open()
		    ? tmp->progress.ReadParameters(progress_stream)
		    : Tools::Result(Tools::ResultCode::FileNotFound);
		if (result.is_not_ok()) {
			if (failed_file) *failed_file = progress_path.u8string();
			sync_cout << "info string NNUE progress params read failed: " << progress_path
			          << " : " << result.to_string() << sync_endl;
			return result;
		}
#endif
		for (int i = 0; i < kLayerStacks; ++i) {
			result = Detail::ReadParameters<Network>(stream, tmp->network[i]);
			if (result.is_not_ok()) {
				sync_cout << "info string NNUE network params read failed at stack " << i << ": " << result.to_string() << sync_endl;
				return result;
			}
		}

		if (!stream || stream.peek() != std::ios::traits_type::eof())
			return Tools::ResultCode::FileCloseError;

		// 共有メモリに配置（同一ハッシュの共有メモリが既に存在すればそちらを参照）
		shared_networks = SystemWideSharedConstant<NnueNetworks>(*tmp);

		return Tools::ResultCode::Ok;
	}

	}  // namespace
    // ヘッダを読み込む
    Tools::Result ReadHeader(std::istream& stream,
        u32* hash_value, std::string* architecture, u32* version_out) {
        u32 version = 0, size = 0;
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (!stream) return Tools::ResultCode::FileReadError;
		if (version_out)
			*version_out = version;
        if (version != kVersion) {
			sync_cout << "info string NNUE header version mismatch: expected " << kVersion
				<< " got " << version << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
        architecture->resize(size);
        stream.read(&(*architecture)[0], size);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    // ヘッダを書き込む
    bool WriteHeader(std::ostream& stream,
        u32 hash_value, const std::string& architecture) {
        stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
        const u32 size = static_cast<u32>(architecture.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(architecture.data(), size);
        return !stream.fail();
    }

    	// 評価関数パラメータを読み込む
    Tools::Result ReadParameters(std::istream& stream, std::string* failed_file) {
        return LoadAndShare(stream, failed_file);
    }
    // 評価関数パラメータを書き込む
    bool WriteParameters(std::ostream& stream) {
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
        const auto qb = networks().network[0].weight_scale;
        if (!WriteHeader(stream, kHashValue, Nn16Architecture(qb))) return false;
        for (int i = 0; i < 4; ++i)
            stream.put(static_cast<char>((qb >> (8 * i)) & 255));
#else
        if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
#endif
        if (!Detail::WriteParameters<FeatureTransformer>(stream, networks().feature_transformer)) return false;
        for (int i = 0; i < kLayerStacks; ++i) {
            if (!Detail::WriteParameters<Network>(stream, networks().network[i])) return false;
        }
        return !stream.fail();
    }

    // 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossible(const Position& pos, AccumulatorState* state) {
        networks().feature_transformer.UpdateAccumulatorIfPossible(pos, state);
    }

#if defined(EVAL_SFNN)
    static_assert(NNUE_SFNN_HAND_BUCKETS == 1 || NNUE_SFNN_HAND_BUCKETS == 4
        || NNUE_SFNN_HAND_BUCKETS == 16 || NNUE_SFNN_HAND_BUCKETS == 64
        || NNUE_SFNN_HAND_BUCKETS == 256 || NNUE_SFNN_HAND_BUCKETS == 1024,
        "unsupported NNUE_SFNN_HAND_BUCKETS");
    static_assert(
        (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_NONE && NNUE_SFNN_HAND_BUCKETS == 1)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND4 && NNUE_SFNN_HAND_BUCKETS == 4)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND16 && NNUE_SFNN_HAND_BUCKETS == 16)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND64 && NNUE_SFNN_HAND_BUCKETS == 64)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND64Z && NNUE_SFNN_HAND_BUCKETS == 64)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND256 && NNUE_SFNN_HAND_BUCKETS == 256)
        || (NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024 && NNUE_SFNN_HAND_BUCKETS == 1024),
        "NNUE_SFNN_HAND_BUCKET_TYPE does not match NNUE_SFNN_HAND_BUCKETS");
    static_assert(NNUE_SFNN_KING_BUCKETS == 1 || NNUE_SFNN_KING_BUCKETS == 9
        || NNUE_SFNN_KING_BUCKETS == 81 || NNUE_SFNN_KING_BUCKETS == 169
        || NNUE_SFNN_KING_BUCKETS == 441 || NNUE_SFNN_KING_BUCKETS == 841,
        "unsupported NNUE_SFNN_KING_BUCKETS");
    static_assert(
        (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_NONE && NNUE_SFNN_KING_BUCKETS == 1)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K3K3 && NNUE_SFNN_KING_BUCKETS == 9)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K9K9 && NNUE_SFNN_KING_BUCKETS == 81)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K21K21 && NNUE_SFNN_KING_BUCKETS == 441)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K29K29 && NNUE_SFNN_KING_BUCKETS == 841)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z && NNUE_SFNN_KING_BUCKETS == 81)
        || (NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z && NNUE_SFNN_KING_BUCKETS == 169),
        "NNUE_SFNN_KING_BUCKET_TYPE does not match NNUE_SFNN_KING_BUCKETS");
    static_assert(NNUE_SFNN_PROGRESS_BUCKETS == 1 || NNUE_SFNN_PROGRESS_BUCKETS == 2
        || NNUE_SFNN_PROGRESS_BUCKETS == 3 || NNUE_SFNN_PROGRESS_BUCKETS == 4
        || NNUE_SFNN_PROGRESS_BUCKETS == 8 || NNUE_SFNN_PROGRESS_BUCKETS == 16
        || NNUE_SFNN_PROGRESS_BUCKETS == 32,
        "unsupported NNUE_SFNN_PROGRESS_BUCKETS");
    static_assert(kLayerStacks == NNUE_SFNN_HAND_BUCKETS * NNUE_SFNN_KING_BUCKETS * NNUE_SFNN_PROGRESS_BUCKETS,
        "LayerStacks must match the SFNN bucket product");

    // レイヤースタックの選択。双方の玉の段に応じて9通りに分岐させる。
    static int king3_by_king3_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        int f_rank = int(pos.square<KING>(stm)) % 9;
        int e_rank = int(pos.square<KING>(~stm)) % 9;

        if (stm == BLACK)
            e_rank = 8 - e_rank;
        else
            f_rank = 8 - f_rank;

        return (f_rank / 3) * 3 + e_rank / 3;
    }

    // レイヤースタックの選択。双方の玉の段に応じて81通りに分岐させる。
    static int king9_by_king9_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        int f_rank = int(stm == BLACK ? rank_of(f_king) : rank_of(Inv(f_king)));
        int e_rank = int(stm == BLACK ? rank_of(Inv(e_king)) : rank_of(e_king));
        if (f_rank < 0) f_rank = 0;
        if (f_rank > 8) f_rank = 8;
        if (e_rank < 0) e_rank = 0;
        if (e_rank > 8) e_rank = 8;
        return f_rank * 9 + e_rank;
    }

    static int file3_bucket(int file) {
        if (file < 0) file = 0;
        if (file > 8) file = 8;
        return file / 3;
    }

    static int king9_zone_single_bucket(Square sq) {
        int rank = int(rank_of(sq));
        int file = int(file_of(sq));
        if (rank < 0) rank = 0;
        if (rank > 8) rank = 8;

        if (rank < 3) return 0;
        if (rank < 6) return 1;
        if (rank == 6) return 2;
        return 3 + (rank - 7) * 3 + file3_bucket(file);
    }

    // レイヤースタックの選択。玉1つを9 zoneに分け、双方の玉で81通りに分岐させる。
    static int king9_zone_by_king9_zone_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_sq = stm == BLACK ? f_king : Inv(f_king);
        const auto e_sq = stm == BLACK ? Inv(e_king) : e_king;
        return king9_zone_single_bucket(f_sq) * 9 + king9_zone_single_bucket(e_sq);
    }

    static int king13_zone_single_bucket(Square sq) {
        int rank = int(rank_of(sq));
        int file = int(file_of(sq));
        if (rank < 0) rank = 0;
        if (rank > 8) rank = 8;

        if (rank < 7) return rank;
        return 7 + (rank - 7) * 3 + file3_bucket(file);
    }

    // レイヤースタックの選択。玉1つを13 zoneに分け、双方の玉で169通りに分岐させる。
    static int king13_zone_by_king13_zone_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_sq = stm == BLACK ? f_king : Inv(f_king);
        const auto e_sq = stm == BLACK ? Inv(e_king) : e_king;
        return king13_zone_single_bucket(f_sq) * 13 + king13_zone_single_bucket(e_sq);
    }

    static int king21_single_bucket(Square sq) {
        int rank = int(rank_of(sq));
        int file = int(file_of(sq));
        if (rank < 0) rank = 0;
        if (rank > 8) rank = 8;
        if (file < 0) file = 0;
        if (file > 8) file = 8;

        if (rank < 3) return 0;
        if (rank < 6) return 1;
        if (rank == 6) return 2;
        return 3 + (rank - 7) * 9 + file;
    }

    // レイヤースタックの選択。玉1つを21通りに分け、双方の玉で441通りに分岐させる。
    static int king21_by_king21_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_sq = stm == BLACK ? f_king : Inv(f_king);
        const auto e_sq = stm == BLACK ? Inv(e_king) : e_king;
        return king21_single_bucket(f_sq) * 21 + king21_single_bucket(e_sq);
    }

    static int king29_single_bucket(Square sq) {
        int rank = int(rank_of(sq));
        int file = int(file_of(sq));
        if (rank < 0) rank = 0;
        if (rank > 8) rank = 8;
        if (file < 0) file = 0;
        if (file > 8) file = 8;

        if (rank < 3) return 0;
        if (rank < 6) return 1;
        return 2 + (rank - 6) * 9 + file;
    }

    // レイヤースタックの選択。玉1つを29通りに分け、双方の玉で841通りに分岐させる。
    static int king29_by_king29_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_sq = stm == BLACK ? f_king : Inv(f_king);
        const auto e_sq = stm == BLACK ? Inv(e_king) : e_king;
        return king29_single_bucket(f_sq) * 29 + king29_single_bucket(e_sq);
    }

    static int hand4_single_bucket(Hand hand) {
        return hand_count(hand, BISHOP) > 0 ? 1 : 0;
    }

    static int hand16_single_bucket(Hand hand) {
        int bucket = 0;
        if (hand_count(hand, PAWN) > 0)
            bucket |= 1;
        if (hand_count(hand, BISHOP) > 0)
            bucket |= 2;
        return bucket;
    }

    static int hand64_single_bucket(Hand hand) {
        int bucket = 0;
        if (hand_count(hand, PAWN) + hand_count(hand, LANCE) + hand_count(hand, KNIGHT) > 0)
            bucket |= 1;
        if (hand_count(hand, GOLD) + hand_count(hand, SILVER) + hand_count(hand, ROOK) > 0)
            bucket |= 2;
        if (hand_count(hand, BISHOP) > 0)
            bucket |= 4;
        return bucket;
    }

    static int hand64z_single_bucket(Hand hand) {
        const int score =
              hand_count(hand, PAWN)
            + (hand_count(hand, LANCE) + hand_count(hand, KNIGHT)) * 2
            + (hand_count(hand, SILVER) + hand_count(hand, GOLD)) * 3
            + (hand_count(hand, BISHOP) + hand_count(hand, ROOK)) * 5;

        int bucket = (score + 3) / 4;
        if (bucket < 0) bucket = 0;
        if (bucket > 7) bucket = 7;
        return bucket;
    }

    static int hand256_single_bucket(Hand hand) {
        int bucket = 0;
        if (hand_count(hand, PAWN) + hand_count(hand, LANCE) + hand_count(hand, KNIGHT) > 0)
            bucket |= 1;
        if (hand_count(hand, SILVER) + hand_count(hand, GOLD) > 0)
            bucket |= 2;
        if (hand_count(hand, BISHOP) > 0)
            bucket |= 4;
        if (hand_count(hand, ROOK) > 0)
            bucket |= 8;
        return bucket;
    }

    static int hand1024_single_bucket(Hand hand) {
        int bucket = 0;
        if (hand_count(hand, PAWN) > 0)
            bucket |= 1;
        if (hand_count(hand, LANCE) + hand_count(hand, KNIGHT) > 0)
            bucket |= 2;
        if (hand_count(hand, SILVER) + hand_count(hand, GOLD) > 0)
            bucket |= 4;
        if (hand_count(hand, BISHOP) > 0)
            bucket |= 8;
        if (hand_count(hand, ROOK) > 0)
            bucket |= 16;
        return bucket;
    }

    static int hand4_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand4_single_bucket(pos.hand_of(stm)) * 2
            + hand4_single_bucket(pos.hand_of(~stm));
    }

    static int hand16_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand16_single_bucket(pos.hand_of(stm)) * 4
            + hand16_single_bucket(pos.hand_of(~stm));
    }

    // 手番側/非手番側の手駒有無を3bitずつ見て、64通りに分岐させる。
    static int hand64_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand64_single_bucket(pos.hand_of(stm)) * 8
            + hand64_single_bucket(pos.hand_of(~stm));
    }

    // 手番側/非手番側の手駒点zoneを8段階ずつに分け、64通りに分岐させる。
    static int hand64z_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand64z_single_bucket(pos.hand_of(stm)) * 8
            + hand64z_single_bucket(pos.hand_of(~stm));
    }

    static int hand256_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand256_single_bucket(pos.hand_of(stm)) * 16
            + hand256_single_bucket(pos.hand_of(~stm));
    }

    static int hand1024_bucket(const Position& pos) {
        const auto stm = pos.side_to_move();
        return hand1024_single_bucket(pos.hand_of(stm)) * 32
            + hand1024_single_bucket(pos.hand_of(~stm));
    }

    static int progress_bucket(const Position& pos) {
#if NNUE_SFNN_PROGRESS_BUCKETS == 1
        return 0;
#else
        return networks().progress.BucketIndex(pos, NNUE_SFNN_PROGRESS_BUCKETS);
#endif
    }

    static int stack_index_for_nnue(const Position& pos) {
#if NNUE_SFNN_HAND_BUCKETS == 1 && NNUE_SFNN_KING_BUCKETS == 9 && NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K3K3 && NNUE_SFNN_PROGRESS_BUCKETS == 1
        return king3_by_king3_bucket(pos);
#else
        int idx = 0;

#if NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND4
        idx = hand4_bucket(pos);
#elif NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND16
        idx = hand16_bucket(pos);
#elif NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND64
        idx = hand64_bucket(pos);
#elif NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND64Z
        idx = hand64z_bucket(pos);
#elif NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND256
        idx = hand256_bucket(pos);
#elif NNUE_SFNN_HAND_BUCKET_TYPE == NNUE_SFNN_HAND_BUCKET_TYPE_HAND1024
        idx = hand1024_bucket(pos);
#endif

#if NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K3K3
        idx = idx * 9 + king3_by_king3_bucket(pos);
#elif NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K9K9
        idx = idx * 81 + king9_by_king9_bucket(pos);
#elif NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K9K9Z
        idx = idx * 81 + king9_zone_by_king9_zone_bucket(pos);
#elif NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K13K13Z
        idx = idx * 169 + king13_zone_by_king13_zone_bucket(pos);
#elif NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K21K21
        idx = idx * 441 + king21_by_king21_bucket(pos);
#elif NNUE_SFNN_KING_BUCKET_TYPE == NNUE_SFNN_KING_BUCKET_TYPE_K29K29
        idx = idx * 841 + king29_by_king29_bucket(pos);
#endif

#if NNUE_SFNN_PROGRESS_BUCKETS != 1
        idx = idx * NNUE_SFNN_PROGRESS_BUCKETS + progress_bucket(pos);
#endif

        if (idx < 0) idx = 0;
        if (idx >= kLayerStacks) idx = kLayerStacks - 1;
        return idx;
#endif
    }
#endif

    // 評価値を計算する
    static Value ComputeScore(const Position& pos, AccumulatorState* state, bool refresh = false) {
        auto& accumulator = accumulator_of(state);
#if defined(EVAL_NNUE)
        if (!refresh && accumulator.computed_score) {
            return accumulator.score;
        }
#endif

        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
#if defined(EVAL_SFNN)
        const auto bucket = stack_index_for_nnue(pos);
#if defined(USE_AVX512) && defined(NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE)
        networks().feature_transformer.EnsureAccumulator(pos, refresh, state);
        const auto output = networks().network[bucket].PropagateFromAccumulator(
            accumulator.accumulation, pos.side_to_move(), buffer);
#else
        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
        networks().feature_transformer.Transform(pos, transformed_features, refresh, state);
        const auto output = networks().network[bucket].Propagate(transformed_features, buffer);
#endif
#else
        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
        networks().feature_transformer.Transform(pos, transformed_features, refresh, state);
        const auto output = networks().network[0].Propagate(transformed_features, buffer);
#endif

        // VALUE_MAX_EVALより大きな値が返ってくるとaspiration searchがfail highして
        // 探索が終わらなくなるのでVALUE_MAX_EVAL以下であることを保証すべき。

        // この現象が起きても、対局時に秒固定などだとそこで探索が打ち切られるので、
        // 1つ前のiterationのときの最善手がbestmoveとして指されるので見かけ上、
        // 問題ない。このVALUE_MAX_EVALが返ってくるような状況は、ほぼ詰みの局面であり、
        // そのような詰みの局面が出現するのは終盤で形勢に大差がついていることが多いので
        // 勝敗にはあまり影響しない。

        // しかし、教師生成時などdepth固定で探索するときに探索から戻ってこなくなるので
        // そのスレッドの計算時間を無駄にする。またdepth固定対局でtime-outするようになる。

#if defined(ENABLE_SFNN_16BIT_WEIGHT)
        auto score = static_cast<Value>(std::clamp<i64>(
            output[0] / FV_SCALE, -VALUE_MAX_EVAL, VALUE_MAX_EVAL));
#else
        auto score = static_cast<Value>(output[0] / FV_SCALE);
#endif

        // 1) ここ、下手にclipすると学習時には影響があるような気もするが…。
        // 2) accumulator.scoreは、差分計算の時に用いないので書き換えて問題ない。
        score = std::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

#if defined(EVAL_NNUE)
        accumulator.score = score;
        accumulator.computed_score = true;
#endif
        return score;
    }

}  // namespace NNUE

#if defined(USE_EVAL_HASH)

// HashTableに評価値を保存するために利用するクラス
struct alignas(16) ScoreKeyValue {
#if defined(USE_SSE2)
    ScoreKeyValue() = default;
    ScoreKeyValue(const ScoreKeyValue & other) {
        static_assert(sizeof(ScoreKeyValue) == sizeof(__m128i),
            "sizeof(ScoreKeyValue) should be equal to sizeof(__m128i)");
        _mm_store_si128(&as_m128i, other.as_m128i);
    }
    ScoreKeyValue& operator=(const ScoreKeyValue & other) {
        _mm_store_si128(&as_m128i, other.as_m128i);
        return *this;
    }
#endif

    // evaluate hashでatomicに操作できる必要があるのでそのための操作子
    void encode() {
#if defined(USE_SSE2)
        // ScoreKeyValue は atomic にコピーされるので key が合っていればデータも合っている。
#else
        key ^= score;
#endif
    }
    // decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
    void decode() { encode(); }

    union {
        struct {
            std::uint64_t key;
            std::uint64_t score;
        };
#if defined(USE_SSE2)
        __m128i as_m128i;
#endif
    };
};

// evaluateしたものを保存しておくHashTable(俗にいうehash)

struct EvaluateHashTable : HashTable<ScoreKeyValue> {};

EvaluateHashTable g_evalTable;
void EvalHash_Resize(size_t mbSize) { g_evalTable.resize(mbSize); }
void EvalHash_Clear() { g_evalTable.clear(); };

// prefetchする関数も用意しておく。
void prefetch_evalhash(const u64 key) {
    constexpr auto mask = ~((u64)0x1f);
    prefetch((void*)((u64)g_evalTable[key] & mask));
}
#endif

// 評価関数ファイルを読み込む
void load_eval() {
    // 評価関数パラメーターを読み込み済みであるなら帰る。
    if (eval_loaded)
        return;

    {
        const std::string dir_name = Options["EvalDir"];
    #if !defined(__EMSCRIPTEN__)
		const std::string file_name = NNUE::kFileName;
#else
		// WASM
        const std::string file_name = Options["EvalFile"];
    #endif
        std::string failed_file = file_name;
        const Tools::Result result = [&] {
            if (dir_name != "<internal>") {
            #if !defined(__EMSCRIPTEN__)
                auto abs_eval_path = CommandLine::get_binary_directory() / path_from_utf8(dir_name);
            #else
                // 📝 wasmでは評価関数ファイルはEmscriptenの仮想FS上にある。
                //     CommandLine::get_binary_directory()はホスト側のパスを返すので前置せず、
                //     EvalDirをそのまま仮想FS上のパスとして扱う。
                auto abs_eval_path = path_from_utf8(dir_name);
            #endif
                const auto file_path = abs_eval_path / path_from_utf8(file_name);
                failed_file = file_path.u8string();
                std::ifstream stream(file_path, std::ios::binary);
                sync_cout << "info string loading eval file : " << file_path << sync_endl;
				if (!stream.is_open())
					return Tools::Result(Tools::ResultCode::FileNotFound);

                return NNUE::ReadParameters(stream, &failed_file);
            }
            else {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, usize n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };

			    const auto embedded = get_embedded(/* embeddedType */);

                MemoryBuffer buffer(
                              const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                              usize(embedded.size));

                std::istream stream(&buffer);
                sync_cout << "info string loading eval file : <internal>" << sync_endl;

                return NNUE::ReadParameters(stream, &failed_file);
            }
        }();

        //      ASSERT(result);

        if (result.is_not_ok())
        {
            // 読み込みエラーのとき終了してくれないと困る。
            sync_cout << "Error! : failed to read " << failed_file << " : " << result.to_string() << sync_endl;
            Tools::exit();
        }

		// 評価関数ファイルの読み込みが完了した。
		eval_loaded = true;
    }
}


// 評価関数。差分計算ではなく全計算する。
// Position::set()で一度だけ呼び出される。(以降は差分計算)
// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
// なので、この関数の最適化は頑張らない。
Value compute_eval(const Position& pos) {
#if defined(EVAL_SFNN)
    NNUE::AccumulatorState state;
    return NNUE::ComputeScore(pos, &state, true);
#else
    return NNUE::ComputeScore(pos, pos.state(), true);
#endif
}

// 評価関数
Value evaluate(const Position& pos) {
#if defined(EVAL_SFNN)
    // Standalone evaluation must not change a search worker's stack.
    // 探索外の評価では、探索Workerのスタックを変更しない。
    NNUE::AccumulatorState state;
    return NNUE::ComputeScore(pos, &state);
#else
    const auto& accumulator = pos.state()->accumulator;
    if (accumulator.computed_score) {
        return accumulator.score;
    }

#if defined(USE_GLOBAL_OPTIONS)
    // GlobalOptionsでeval hashを用いない設定になっているなら
    // eval hashへの照会をskipする。
    if (!GlobalOptions.use_eval_hash) {
        ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
        return NNUE::ComputeScore(pos, pos.state());
    }
#endif

#if defined(USE_EVAL_HASH)
    // evaluate hash tableにはあるかも。
    const HASH_KEY key = pos.state()->key();
    ScoreKeyValue entry = *g_evalTable[key];
    entry.decode();
    if (entry.key == key) {
        // あった！
        return Value(entry.score);
    }
#endif

    Value score = NNUE::ComputeScore(pos, pos.state());
#if defined(USE_EVAL_HASH)
    // せっかく計算したのでevaluate hash tableに保存しておく。
    entry.key = key;
    entry.score = score;
    entry.encode();
    *g_evalTable[key] = entry;
#endif

    return score;
#endif
}

#if defined(EVAL_SFNN)
Value evaluate(const Position& pos, NNUE::AccumulatorStack& stack) {
    return NNUE::ComputeScore(pos, &stack.latest());
}
#endif

// 差分計算ができるなら進める
#if defined(EVAL_NNUE)
void evaluate_with_no_return(const Position& pos) {
    NNUE::UpdateAccumulatorIfPossible(pos, pos.state());
}
#endif

// 現在の局面の評価値の内訳を表示する
void print_eval_stat(Position& /*pos*/) {
    std::cout << "--- EVAL STAT: not implemented" << std::endl;
}

} // namespace Eval
} // namespace YaneuraOu

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))
