// NNUE評価関数で用いるheader

#ifndef _EVALUATE_NNUE_H_
#define _EVALUATE_NNUE_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_feature_transformer.h"
#include "nnue_architecture.h"
#include "../../misc.h"

#include <memory>

namespace Eval {

namespace NNUE {

// 評価関数の構造のハッシュ値
constexpr std::uint32_t kHashValue =
    FeatureTransformer::GetHashValue() ^ Network::GetHashValue();

// メモリ領域の解放を自動化するためのデリータ
template <typename T>
struct AlignedDeleter {

    void operator()(T* ptr) const {

        // Tクラスのデストラクタ
        ptr->~T();

        // LargeMemoryの開放
        //const_cast<AlignedDeleter*>(this)->memory.free();

        // →　これ、このタイミングで開放せずとも、large_memory()->alloc()で次に確保するときに前のは開放されるし、
        // このクラスのデストラクタでも開放されるから、余計なことしないことにする。
        //
        // ここで上のように開放すると、shared_ptr.reset()のときに前のやつが所有権を失うのでそのタイミングでLargeMemoryが開放されてしまうので
        //   pointer.reset(reinterpret_cast<T*>(pointer.get_deleter().large_memory()->alloc(sizeof(T), alignof(T), true)));
        // のように書けなくなる。
    }

    // SharedPtrのポインタの開放とともに開放されて欲しいLargeMemoryは、ここから割り当てると良い。
    LargeMemory* large_memory() { return &memory; }

private:
    LargeMemory memory;
};

template <typename T>
using AlignedPtr = std::unique_ptr<T, AlignedDeleter<T>>;

// 入力特徴量変換器
extern AlignedPtr<FeatureTransformer> feature_transformer;

// 評価関数
extern AlignedPtr<Network> network;

// 評価関数ファイル名
extern const char* const kFileName;

// 評価関数の構造を表す文字列を取得する
std::string GetArchitectureString();

// ヘッダを読み込む
bool ReadHeader(std::istream& stream,
    std::uint32_t* hash_value, std::string* architecture);

// ヘッダを書き込む
bool WriteHeader(std::ostream& stream,
    std::uint32_t hash_value, const std::string& architecture);

// 評価関数パラメータを読み込む
bool ReadParameters(std::istream& stream);

// 評価関数パラメータを書き込む
bool WriteParameters(std::ostream& stream);

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
