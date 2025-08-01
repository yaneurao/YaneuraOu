// NNUE評価関数の計算に関するコード

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <fstream>

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

namespace YaneuraOu::Eval::NNUE {
extern int FV_SCALE;
}
 
// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL)
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

#if defined(EVAL_LEARN)
    // isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
    // test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
    // このコマンドの実行前に異常終了してしまう。
    // そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
    // test evalconvertコマンドを叩く。
    Options("SkipLoadingEval", Option(false));
#endif

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

#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
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

	//EmbeddedNNUE get_embedded(EmbeddedNNUEType type) {
	//	if (type == EmbeddedNNUEType::BIG)
	//		return EmbeddedNNUE(gEmbeddedNNUEBigData, gEmbeddedNNUEBigEnd, gEmbeddedNNUEBigSize);
	//	else
	//		return EmbeddedNNUE(gEmbeddedNNUESmallData, gEmbeddedNNUESmallEnd, gEmbeddedNNUESmallSize);
	//}

	// ⇨  StockfishはNNUEとして大きなnetworkと小さなnetworkがある。

	EmbeddedNNUE get_embedded() {
		return EmbeddedNNUE(gEmbeddedNNUEData, gEmbeddedNNUEEnd, gEmbeddedNNUESize);
	}
}


namespace YaneuraOu {
namespace Eval {
namespace NNUE {

	int FV_SCALE = 16; // 水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

    // 入力特徴量変換器
	LargePagePtr<FeatureTransformer> feature_transformer;

    // 評価関数
    AlignedPtr<Network> network;

    // 評価関数ファイル名
    const char* const kFileName = EvalFileDefaultName;

    // 評価関数の構造を表す文字列を取得する
    std::string GetArchitectureString() {
        return "Features=" + FeatureTransformer::GetStructureString() +
			",Network=" + Network::GetStructureString();
    }

    namespace {
        namespace Detail {

            // 評価関数パラメータを初期化する
            template <typename T>
            void Initialize(AlignedPtr<T>& pointer) {
				pointer = make_unique_aligned<T>();
            }

			template <typename T>
			void Initialize(LargePagePtr<T>& pointer) {
				// →　メモリはLarge Pageから確保することで高速化する。
				pointer = make_unique_large_page<T>();
			}

            // 評価関数パラメータを読み込む
            template <typename T>
            Tools::Result ReadParameters(std::istream& stream, const AlignedPtr<T>& pointer) {
                std::uint32_t header;
                stream.read(reinterpret_cast<char*>(&header), sizeof(header));
				if (!stream)                     return Tools::ResultCode::FileReadError;
				if (header != T::GetHashValue()) return Tools::ResultCode::FileMismatch;
                return pointer->ReadParameters(stream);
            }

			// 評価関数パラメータを読み込む
			template <typename T>
			Tools::Result ReadParameters(std::istream& stream, const LargePagePtr<T>& pointer) {
				std::uint32_t header;
				stream.read(reinterpret_cast<char*>(&header), sizeof(header));
				if (!stream)                     return Tools::ResultCode::FileReadError;
				if (header != T::GetHashValue()) return Tools::ResultCode::FileMismatch;
				return pointer->ReadParameters(stream);
			}

			// 評価関数パラメータを書き込む
            template <typename T>
            bool WriteParameters(std::ostream& stream, const AlignedPtr<T>& pointer) {
                constexpr std::uint32_t header = T::GetHashValue();
                stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
                return pointer->WriteParameters(stream);
            }

			// 評価関数パラメータを書き込む
			template <typename T>
			bool WriteParameters(std::ostream& stream, const LargePagePtr<T>& pointer) {
				constexpr std::uint32_t header = T::GetHashValue();
				stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
				return pointer->WriteParameters(stream);
			}

        }  // namespace Detail

        // 評価関数パラメータを初期化する
        void Initialize() {
            Detail::Initialize<FeatureTransformer>(feature_transformer);
            Detail::Initialize<Network>(network);
        }

    }  // namespace

    // ヘッダを読み込む
    Tools::Result ReadHeader(std::istream& stream,
        std::uint32_t* hash_value, std::string* architecture) {
        std::uint32_t version, size;
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (!stream || version != kVersion) return Tools::ResultCode::FileMismatch;
        architecture->resize(size);
        stream.read(&(*architecture)[0], size);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    // ヘッダを書き込む
    bool WriteHeader(std::ostream& stream,
        std::uint32_t hash_value, const std::string& architecture) {
        stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
        const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(architecture.data(), size);
        return !stream.fail();
    }

    // 評価関数パラメータを読み込む
    Tools::Result ReadParameters(std::istream& stream) {
        std::uint32_t hash_value;
        std::string architecture;
		Tools::Result result = ReadHeader(stream, &hash_value, &architecture);
        if (result.is_not_ok()) return result;
        if (hash_value != kHashValue) return Tools::ResultCode::FileMismatch;
		result = Detail::ReadParameters<FeatureTransformer>(stream, feature_transformer); if (result.is_not_ok()) return result;
		result = Detail::ReadParameters<Network>(stream, network);             if (result.is_not_ok()) return result;
        return (stream && stream.peek() == std::ios::traits_type::eof()) ? Tools::ResultCode::Ok : Tools::ResultCode::FileCloseError;
    }

    // 評価関数パラメータを書き込む
    bool WriteParameters(std::ostream& stream) {
        if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
        if (!Detail::WriteParameters<FeatureTransformer>(stream, feature_transformer)) return false;
        if (!Detail::WriteParameters<Network>(stream, network)) return false;
        return !stream.fail();
    }

    // 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossible(const Position& pos) {
        feature_transformer->UpdateAccumulatorIfPossible(pos);
    }

    // 評価値を計算する
    static Value ComputeScore(const Position& pos, bool refresh = false) {
        auto& accumulator = pos.state()->accumulator;
        if (!refresh && accumulator.computed_score) {
            return accumulator.score;
        }

        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
        feature_transformer->Transform(pos, transformed_features, refresh);
        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
        const auto output = network->Propagate(transformed_features, buffer);

        // VALUE_MAX_EVALより大きな値が返ってくるとaspiration searchがfail highして
        // 探索が終わらなくなるのでVALUE_MAX_EVAL以下であることを保証すべき。

        // この現象が起きても、対局時に秒固定などだとそこで探索が打ち切られるので、
        // 1つ前のiterationのときの最善手がbestmoveとして指されるので見かけ上、
        // 問題ない。このVALUE_MAX_EVALが返ってくるような状況は、ほぼ詰みの局面であり、
        // そのような詰みの局面が出現するのは終盤で形勢に大差がついていることが多いので
        // 勝敗にはあまり影響しない。

        // しかし、教師生成時などdepth固定で探索するときに探索から戻ってこなくなるので
        // そのスレッドの計算時間を無駄にする。またdepth固定対局でtime-outするようになる。

        auto score = static_cast<Value>(output[0] / FV_SCALE);

        // 1) ここ、下手にclipすると学習時には影響があるような気もするが…。
        // 2) accumulator.scoreは、差分計算の時に用いないので書き換えて問題ない。
        score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

        accumulator.score = score;
        accumulator.computed_score = true;
        return accumulator.score;
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
void prefetch_evalhash(const Key key) {
    constexpr auto mask = ~((u64)0x1f);
    prefetch((void*)((u64)g_evalTable[key] & mask));
}
#endif

// 評価関数ファイルを読み込む
void load_eval() {
    // 評価関数パラメーターを読み込み済みであるなら帰る。
    if (eval_loaded)
        return;

	// 初期化もここでやる。
	NNUE::Initialize();

#if defined(EVAL_LEARN)
    if (!Options["SkipLoadingEval"])
#endif
    {
        const std::string dir_name = Options["EvalDir"];
    #if !defined(__EMSCRIPTEN__)
		const std::string file_name = NNUE::kFileName;
#else
		// WASM
        const std::string file_name = options["EvalFile"];
    #endif
        const Tools::Result result = [&] {
            if (dir_name != "<internal>") {
                auto full_dir_name = Path::Combine(Directory::GetCurrentFolder(), dir_name);
                sync_cout << "info string EvalDirectory = " << full_dir_name << sync_endl;

                const std::string file_path = Path::Combine(dir_name, file_name);
                std::ifstream stream(file_path, std::ios::binary);
                sync_cout << "info string loading eval file : " << file_path << sync_endl;
				if (!stream.is_open())
					return Tools::Result(Tools::ResultCode::FileNotFound);

                return NNUE::ReadParameters(stream);
            }
            else {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };

			    const auto embedded = get_embedded(/* embeddedType */);

                MemoryBuffer buffer(
                              const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                              size_t(embedded.size));

                std::istream stream(&buffer);
                sync_cout << "info string loading eval file : <internal>" << sync_endl;

                return NNUE::ReadParameters(stream);
            }
        }();

        //      ASSERT(result);

        if (result.is_not_ok())
        {
            // 読み込みエラーのとき終了してくれないと困る。
            sync_cout << "Error! : failed to read " << file_name << " : " << result.to_string() << sync_endl;
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
    return NNUE::ComputeScore(pos, true);
}

// 評価関数
Value evaluate(const Position& pos) {
    const auto& accumulator = pos.state()->accumulator;
    if (accumulator.computed_score) {
        return accumulator.score;
    }

#if defined(USE_GLOBAL_OPTIONS)
    // GlobalOptionsでeval hashを用いない設定になっているなら
    // eval hashへの照会をskipする。
    if (!GlobalOptions.use_eval_hash) {
        ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
        return NNUE::ComputeScore(pos);
    }
#endif

#if defined(USE_EVAL_HASH)
    // evaluate hash tableにはあるかも。
    const Key key = pos.state()->key();
    ScoreKeyValue entry = *g_evalTable[key];
    entry.decode();
    if (entry.key == key) {
        // あった！
        return Value(entry.score);
    }
#endif

    Value score = NNUE::ComputeScore(pos);
#if defined(USE_EVAL_HASH)
    // せっかく計算したのでevaluate hash tableに保存しておく。
    entry.key = key;
    entry.score = score;
    entry.encode();
    *g_evalTable[key] = entry;
#endif

    return score;
}

// 差分計算ができるなら進める
void evaluate_with_no_return(const Position& pos) {
    NNUE::UpdateAccumulatorIfPossible(pos);
}

// 現在の局面の評価値の内訳を表示する
void print_eval_stat(Position& /*pos*/) {
    std::cout << "--- EVAL STAT: not implemented" << std::endl;
}

} // namespace Eval
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
