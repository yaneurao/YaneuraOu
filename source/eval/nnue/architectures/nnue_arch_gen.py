# NNUE architecture header generator
#
#  NNUE評価関数のarchitecture headerを動的に生成するPythonで書かれたスクリプト。
# 

import argparse
import os

def dedent4(text: str) -> str:
    # 各行の先頭4文字（スペース4つ）を削除して結合し直す
    # 行が4文字未満、あるいはスペースでない場合を考慮して lstrip でも可
    return "\n".join(line[4:] if line.startswith("    ") else line 
                        for line in text.strip("\n").splitlines())


print("NNUE architecture header generator by yaneurao V1.02 , 2026/01/31")

parser = argparse.ArgumentParser(description="NNUEのarchitecture headerを生成する。")
parser.add_argument('arch', type=str, nargs='?', default="halfkp_256x2-32-32", help="architectureを指定する。例) halfkp_1024x2-8-64, YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_16_32とか")
parser.add_argument('out_dir', type=str, nargs='?', default="", help="出力先のフォルダを指定する。例) /source/eval/nnue/architectures/")

args = parser.parse_args()

arch    : str = args.arch
out_dir : str = args.out_dir

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

# makefileで指定したエディション名そのままかも知れないので削除。
arch = strip_prefix_ci(arch, "YANEURAOU_ENGINE_")
arch = strip_prefix_ci(arch, "NNUE_")

arch_upper_for_validation = arch.replace('-', '_').upper()
if "SFNNWOP" in arch_upper_for_validation:
    print("Error! : SFNNWOP architecture names are no longer supported. Use SFNN1536 or SFNN_..._k3k3 / SFNN_..._king3_by_king3.")
    raise SystemExit(1)

if "LS9" in arch_upper_for_validation.split('_'):
    print("Error! : ls9 is no longer supported. Use k3k3 or king3_by_king3.")
    raise SystemExit(1)

# 出力ファイル名
filename = arch + ".h"

# 出力file path
out_path = os.path.join(out_dir, filename)

print(f"output file path  : {out_path}")

# 大文字化して、'-'を'_'に置換したアーキテクチャ名
arch   = arch.replace('-','_')
arch   = arch.upper()

print(f"architecture name : {arch}")

# if os.path.exists(out_path):
#     print("Warning : file already exists. stop.")
#     exit()
#  🤔 ファイルがすでに存在していても上書きしたほうがいいと思う。

arches = arch.split('_')
if len(arches) <= 3 :
    # アーキテクチャ名は、アンダースコアは3つ以上ないと駄目。
    print("Error! : architecture name must be like halfkp_256x2-32-32 or kp_256x2-32-32 halfkpvm_256x2_32_32")
    raise SystemExit(1)

# 📝 SFNN_halfkahm2_1536-15-32-k3k3のように指定されていれば、SFNNのheaderを生成する。
SFNN = False
layer_stack_name = ""
if arches[0].startswith("SFNN"):
    SFNN = True
    if len(arches) < 6:
        print("Error! : SFNN architecture name must be like SFNN_halfkahm2_1536-15-32-k3k3")
        raise SystemExit(1)

    layer_stack_spec = "_".join(arches[5:])
    if layer_stack_spec == "K3K3" or layer_stack_spec == "KING3_BY_KING3":
        layer_stack_name = "K3K3"
        layer_stack_count = "9"
    else:
        print("Error! : SFNN layer stack must be k3k3 or king3_by_king3")
        raise SystemExit(1)

    arches = [arches[1], arches[2], arches[3], arches[4], layer_stack_count]

# ============================================================
#                        includes
# ============================================================

if SFNN:
    header = f"""
    // SFNN without PSQT 1536 architecture

    #ifndef CLASSIC_NNUE_SFNN_{arch}_H_INCLUDED
    #define CLASSIC_NNUE_SFNN_{arch}_H_INCLUDED
    """
else:
    header = f"""
    // Definition of input features and network structure used in NNUE evaluation function
    // NNUE評価関数で用いる入力特徴量とネットワーク構造の定義
    #ifndef NNUE_{arch}_H_INCLUDED
    #define NNUE_{arch}_H_INCLUDED
    """

# ============================================================
#                     input features
# ============================================================

# アーキテクチャ名のアンダースコアでsplitした1つ目は入力特徴量。
# 現在サポートしている入力特徴量は、
#   halfkp
#   kp
#   ka2
#   halfkpe9
#   halfkpvm
#   halfka1
#   halfkahm1
#   halfka2
#   halfkahm2

input_feature = arches[0].lower()

print(f"input feature     : {input_feature}")

header += f"""
    #include "../features/feature_set.h"
    """

if input_feature == "halfkp":

    header += f"""
    #include "../features/half_kp.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKP<Features::Side::kFriend>>;
    """

elif input_feature == "kp":

    header += f"""
    #include "../features/k.h"
    #include "../features/p.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<Features::K, Features::P>;
    """

elif input_feature == "ka2":

    header += f"""
    #include "../features/k.h"
    #include "../features/a2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<Features::K, Features::A2>;
    """

elif input_feature == "halfkpe9":

    header += f"""
    #include "../features/half_kpe9.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKPE9<Features::Side::kFriend>>;
    """

elif input_feature == "halfkpvm":

    header += f"""
    #include "../features/half_kp_vm.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKP_vm<Features::Side::kFriend>>;
    """

elif input_feature == "halfka1":

    header += f"""
    #include "../features/half_ka1.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA1<Features::Side::kFriend>>;
    """

elif input_feature == "halfkahm1":

    header += f"""
    #include "../features/half_ka_hm1.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA_hm1<Features::Side::kFriend>>;
    """

elif input_feature == "halfka2":

    header += f"""
    #include "../features/half_ka2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA2<Features::Side::kFriend>>;
    """

elif input_feature == "halfkahm2":

    header += f"""
    #include "../features/half_ka_hm2.h"
    """

    raw_features = f"""
        using RawFeatures = Features::FeatureSet<
            Features::HalfKA_hm2<Features::Side::kFriend>>;
    """

else:
    # 知らない入力特徴量だった。
    print(f"Error : input feature {input_feature} is not supported.")
    raise SystemExit(1)

if SFNN:
    header += """
    #include <cstring>

    #include "../layers/affine_transform_explicit.h"
    #include "../layers/affine_transform_sparse_input_explicit.h"
    #include "../layers/clipped_relu_explicit.h"
    #include "../layers/sqr_clipped_relu.h"

    namespace YaneuraOu {
    namespace Eval::NNUE {

    // Input features used in evaluation function
    // 評価関数で用いる入力特徴量
    """

else:    

    header += """
    #include "../layers/input_slice.h"
    #include "../layers/affine_transform.h"
    #include "../layers/affine_transform_sparse_input.h"
    #include "../layers/clipped_relu.h"

    namespace YaneuraOu {
    namespace Eval::NNUE {

    // Input features used in evaluation function
    // 評価関数で用いる入力特徴量
    """

header += raw_features

# ============================================================
#                     hidden layers
# ============================================================

# レイヤ情報
# 例えば、"256x2_32_32" ならば ["256x2","32","32"]のように分解される。
#   (SFNNで) "1536-15-32-k3k3" なら ["1536","15","32","9"]のように分解される。(はず)
layers = arches[1:]
layers[0] = layers[0].lower()

if SFNN:
    if len(layers) != 4:
        print(f"Error : layers must be like 1536-15-32-k3k3 , layers = {layers}.")
        raise SystemExit(1)

    print(f"layers feature    : {layers}")

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {layers[0]};

        // Number of networks stored in the evaluation file
        constexpr int LayerStacks = {layers[3]};

        // 各層の次元数
        constexpr IndexType kInputDims   = kTransformedFeatureDimensions;
        constexpr IndexType kHidden1Dims = {layers[1]};
        constexpr IndexType kHidden2Dims = {layers[2]};                              
    """

else:

    if len(layers) != 3 or len(layers[0].split('x')) != 2:
        print(f"Error : layers must be like 256x2-32-32 , layers = {layers}.")
        raise SystemExit(1)

    first_layer = layers[0].split('x')

    print(f"layers feature    : {layers}")

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {first_layer[0]};

        namespace Layers {{

            // Define network structure
            // ネットワーク構造の定義
            using InputLayer = InputSlice<kTransformedFeatureDimensions * {first_layer[1]}>;
            using HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, {layers[1]}>>;
            using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, {layers[2]}>>;
            using OutputLayer = AffineTransform<HiddenLayer2, 1>;

        }}  // namespace Layers
    """

# ============================================================
#                     output layer
# ============================================================

if SFNN:
    # `sfnn-1536.h`からそのままコピペ。
    header += f"""
        struct Network {{

            // Define network structure
            // ネットワーク構造の定義
            Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1> fc_0;
            Layers::ClippedReLUExplicit<kHidden1Dims + 1> ac_0;
            Layers::SqrClippedReLU<kHidden1Dims + 1> ac_sqr_0;

            Layers::AffineTransformExplicit<kHidden1Dims * 2, kHidden2Dims> fc_1;
            Layers::ClippedReLUExplicit<kHidden2Dims> ac_1;
            
        Layers::AffineTransformExplicit<kHidden2Dims, 1> fc_2;

            using OutputType = std::int32_t;
            static constexpr IndexType kOutputDimensions = 1;

            // Hash値などは適宜実装
            static constexpr std::uint32_t GetHashValue() {{
                return 0x6333718Au;
            }}

            static std::string GetStructureString() {{
                return "{'SFNN-1536' if input_feature == 'halfkahm2' and layers == ['1536', '15', '32', '9'] and layer_stack_name == 'K3K3' else arch}";
            }}

            Tools::Result ReadParameters(std::istream& stream) {{
                bool result = fc_0.ReadParameters(stream).is_ok()
                    && ac_0.ReadParameters(stream).is_ok()
                    && ac_sqr_0.ReadParameters(stream).is_ok()
                    && fc_1.ReadParameters(stream).is_ok()
                    && ac_1.ReadParameters(stream).is_ok()
                    && fc_2.ReadParameters(stream).is_ok();
                return result ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
            }}

            bool WriteParameters(std::ostream& stream) const {{
                return fc_0.WriteParameters(stream)
                    && ac_0.WriteParameters(stream)
                    && ac_sqr_0.WriteParameters(stream)
                    && fc_1.WriteParameters(stream)
                    && ac_1.WriteParameters(stream)
                    && fc_2.WriteParameters(stream);
            }}

            struct alignas(kCacheLineSize) Buffer {{
                alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
                alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
                alignas(kCacheLineSize) typename decltype(ac_sqr_0)::OutputType ac_sqr_0_out[CeilToMultiple<IndexType>(kHidden1Dims * 2, 32)];
                alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
                alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
                alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
            }};

            static constexpr std::size_t kBufferSize = sizeof(Buffer);

            const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, char* buffer) const {{
                auto& buf = *reinterpret_cast<Buffer*>(buffer);
                std::memset(buf.ac_sqr_0_out, 0, sizeof(buf.ac_sqr_0_out));

                fc_0.Propagate(transformedFeatures, buf.fc_0_out);
                ac_0.Propagate(buf.fc_0_out, buf.ac_0_out);
                ac_sqr_0.Propagate(buf.fc_0_out, buf.ac_sqr_0_out);
                std::memcpy(buf.ac_sqr_0_out + kHidden1Dims, buf.ac_0_out,
                    kHidden1Dims * sizeof(typename decltype(ac_0)::OutputType));
                fc_1.Propagate(buf.ac_sqr_0_out, buf.fc_1_out);
                ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
                fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);

                // add shortcut term
                buf.fc_2_out[0] += buf.fc_0_out[kHidden1Dims];

                return buf.fc_2_out;
            }}
        }};

    }}  // namespace Eval::NNUE
    }}  // namespace YaneuraOu

    #endif // CLASSIC_NNUE_{arch}_H_INCLUDED
    """

    # 💡 GetStructureString()で異なる文字列を返すと別のアーキテクチャとみなされてしまう。

else:
    header += f"""
        using Network = Layers::OutputLayer;

    }} // namespace Eval::NNUE
    }} // namespace YaneuraOu

    #endif // #ifndef NNUE_{arch}_H_INCLUDED
    """

with open(out_path, "w", encoding = 'utf-8') as f:
    f.write(dedent4(header))

print("..done!")
