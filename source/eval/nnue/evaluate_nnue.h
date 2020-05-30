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

    // このクラスのoperator()でaligned_ttmem_free()したいアドレスを渡す。
    void set_mem(void* mem)
    {
        this->mem = mem;
    }
  void operator()(T* ptr) const {
        // Tクラスのデストラクタ
    ptr->~T();

        // aligned_ttmem_alloc()で確保してたポインタ
        if (mem)
            aligned_ttmem_free(mem);
  }
    void* mem = nullptr;
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
