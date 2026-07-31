# NNUE architecture header generator
#
#  NNUE評価関数のarchitecture headerを動的に生成するPythonで書かれたスクリプト。
# 

import argparse
import os
import subprocess
import sys

def dedent4(text: str) -> str:
    # 各行の先頭4文字（スペース4つ）を削除して結合し直す
    # 行が4文字未満、あるいはスペースでない場合を考慮して lstrip でも可
    return "\n".join(line[4:] if line.startswith("    ") else line 
                        for line in text.strip("\n").splitlines())


print("NNUE architecture header generator by yaneurao V1.03 , 2026/07/20")

parser = argparse.ArgumentParser(description="NNUEのarchitecture headerを生成する。")
parser.add_argument('arch', type=str, nargs='?', default="halfkp_256x2-32-32", help="architectureを指定する。例) halfkp_1024x2-8-64, YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_16_32とか")
parser.add_argument('out_dir', type=str, nargs='?', default=None, help="出力先のフォルダを指定する。省略時はこのスクリプトと同じフォルダ。")
parser.add_argument('--write-dummy-nn', type=str, default="", help="指定pathに、このarchitecture用のdummy nn.binを生成する。")
parser.add_argument('--dummy-mode', type=str, choices=("random-small", "zero"), default="random-small", help="dummy nn.binの初期化方式。デフォルトはrandom-small。")
parser.add_argument('--dummy-seed', type=int, default=20260722, help="random-small用の乱数seed。")

args = parser.parse_args()

arch    : str = args.arch
out_dir : str = args.out_dir or os.path.dirname(os.path.abspath(__file__))
dummy_nn_path : str = args.write_dummy_nn
original_arch : str = arch

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

SQ_NB = 81
FILE_NB = 9
FE_END = 1548
F_KING = FE_END
E_KING = F_KING + SQ_NB
FE_END2 = E_KING + SQ_NB

FEATURE_INFO = {
    "halfkp": ("HalfKP(Friend)", 0x5D69D5B8, SQ_NB * FE_END),
    "kp": ("K+P", 0, SQ_NB * 2 + FE_END),
    "ka2": ("K+A2", 0, SQ_NB * 2 + E_KING),
    "halfkpe9": ("HalfKPE9(Friend)", 0x5D69D5B8, SQ_NB * FE_END * 3 * 3),
    "halfkpvm": ("HalfKP_vm(Friend)", 0x0B6B1D9A, 5 * FILE_NB * FE_END),
    "halfka1": ("HalfKA1(Friend)", 0x5F134CB8, SQ_NB * FE_END2),
    "halfkahm1": ("HalfKA_hm1(Friend)", 0x7F134CB8, 5 * FILE_NB * FE_END2),
    "halfka2": ("HalfKA2(Friend)", 0x5F234CB8, SQ_NB * E_KING),
    "halfkahm2": ("HalfKA_hm2(Friend)", 0x7F234CB8, 5 * FILE_NB * E_KING),
}

# makefileで指定したエディション名そのままかも知れないので削除。
arch = strip_prefix_ci(arch, "YANEURAOU_ENGINE_")
arch = strip_prefix_ci(arch, "NNUE_")

arch_upper_for_validation = arch.replace('-', '_').upper()
if "SFNNWOP" in arch_upper_for_validation:
    print("Error! : SFNNWOP architecture names are no longer supported. Use SFNN1536, SFNN_... without suffix, or SFNN_..._k3k3 / SFNN_..._king3_by_king3.")
    raise SystemExit(1)

if "LS9" in arch_upper_for_validation.split('_'):
    print("Error! : ls9 is no longer supported. Use no suffix, hand4/16/64/64z/256/1024, k3k3, k9k9, k9k9z, k13k13z, k21k21, k29k29, or their long names.")
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
#     SFNN_ka2_3072_7_64_c1024_s256x8_k3k3 のように、cN_sMxG を置くと
#     fc_0を common N + shard M x G に分割する。
#     SFNN_halfka2_1024_7_64 のようにsuffixを省略すると、単一LayerStackになる。
#     SFNN_halfka2_1024_7_64_hand64z のように、hand64zを指定すると
#     手番側/非手番側の手駒点を8段階ずつに分けた64 bucketを用いる。
#     hand4 / hand16 / hand64 / hand256 / hand1024も同様に、手番側/非手番側の手駒状態でbucketを用いる。
#     SFNN_halfka2_1024_7_64_k9k9 / k9k9z / k13k13z / k21k21 / k29k29 のように指定すると、
#     手番側/非手番側の玉位置でbucketを分ける。
#     SFNN_halfka2_1024_7_64_hand16_k3k3 / hand64z_k9k9 / hand64z_k13k13z / hand64z_k21k21 / hand64z_k29k29 のように、
#     hand bucketと複合できる。
#     SFNN_halfka2_1024_7_64_k3k3_progress8 のようにprogress2/3/4/8/16/32とも複合できる。
SFNN = False
layer_stack_name = ""
layer_stack_count = ""
layer_stack_hand_buckets = "1"
layer_stack_hand_bucket_type = "0"
layer_stack_king_buckets = "1"
layer_stack_king_bucket_type = "0"
layer_stack_progress_buckets = "1"
sfnn_group_count = "1"
sfnn_common_dims = "0"
sfnn_shard_dims = "0"
sfnn_common_shard = False

def parse_sfnn_layer_stack_spec(layer_stack_spec):
    if layer_stack_spec == "":
        return "NONE", "1", "1", "0", "1", "0", "1"

    normalized = layer_stack_spec
    for long_name, short_name in {
            "KING3_BY_KING3": "K3K3",
            "KING9_BY_KING9": "K9K9",
            "KING9Z_BY_KING9Z": "K9K9Z",
            "KING9ZONE_BY_KING9ZONE": "K9K9Z",
            "KING13Z_BY_KING13Z": "K13K13Z",
            "KING13ZONE_BY_KING13ZONE": "K13K13Z",
            "KING21_BY_KING21": "K21K21",
            "KING29_BY_KING29": "K29K29",
        }.items():
        normalized = normalized.replace(long_name, short_name)

    hand_buckets = 1
    king_buckets = 1
    progress_buckets = 1
    hand_name = ""
    hand_type = 0
    king_name = ""
    king_type = 0
    progress_name = ""

    hand_map = {
        "HAND4": (4, 1),
        "HAND16": (16, 2),
        "HAND64": (64, 3),
        "HAND64Z": (64, 4),
        "HAND256": (256, 5),
        "HAND1024": (1024, 6),
    }
    king_map = {
        "K3K3": (9, 1),
        "K9K9": (81, 2),
        "K21K21": (21 * 21, 3),
        "K29K29": (29 * 29, 4),
        "K9K9Z": (81, 5),
        "K13K13Z": (13 * 13, 6),
    }
    progress_values = {2, 3, 4, 8, 16, 32}

    for token in [t for t in normalized.split("_") if t]:
        if token in hand_map:
            if hand_buckets != 1:
                print(f"Error! : duplicate SFNN hand bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            hand_name = token
            hand_buckets, hand_type = hand_map[token]
        elif token in king_map:
            if king_buckets != 1:
                print(f"Error! : duplicate SFNN king bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            king_name = token
            king_buckets, king_type = king_map[token]
        elif token.startswith("PROGRESS"):
            raw = token[len("PROGRESS"):]
            if not raw.isdigit() or int(raw) not in progress_values:
                print(f"Error! : progress bucket must be progress2/3/4/8/16/32 , got {token}.")
                raise SystemExit(1)
            if progress_buckets != 1:
                print(f"Error! : duplicate SFNN progress bucket in {layer_stack_spec}.")
                raise SystemExit(1)
            progress_name = token
            progress_buckets = int(raw)
        else:
            print(f"Error! : unknown SFNN layer stack token {token} in {layer_stack_spec}.")
            print("Error! : SFNN layer stack tokens are hand4/16/64/64z/256/1024, k3k3/k9k9/k9k9z/k13k13z/k21k21/k29k29, and progress2/3/4/8/16/32.")
            raise SystemExit(1)

    canonical = "_".join([name for name in [hand_name, king_name, progress_name] if name])
    if canonical == "":
        canonical = "NONE"

    layer_count = hand_buckets * king_buckets * progress_buckets
    return canonical, str(layer_count), str(hand_buckets), str(hand_type), str(king_buckets), str(king_type), str(progress_buckets)

if arches[0].startswith("SFNN"):
    SFNN = True
    if len(arches) < 5:
        print("Error! : SFNN architecture name must be like SFNN_halfka2_1024_7_64, SFNN_halfkahm2_1536-15-32-k3k3, or SFNN_ka2_3072_7_64_c1024_s256x8_k3k3")
        raise SystemExit(1)

    layer_stack_start = 5
    if len(arches) > 5 and arches[5].startswith("C"):
        common_raw = arches[5][1:]
        if not common_raw.isdigit():
            print(f"Error! : SFNN common token must be like c0 or c1024 , got {arches[5]}.")
            raise SystemExit(1)
        if len(arches) <= 6 or not arches[6].startswith("S"):
            print("Error! : SFNN common+shard architecture requires shard token like s256x8.")
            raise SystemExit(1)
        shard_spec = arches[6][1:]
        shard_parts = shard_spec.split("X")
        if (len(shard_parts) != 2 or not shard_parts[0].isdigit()
                or not shard_parts[1].isdigit() or int(shard_parts[0]) <= 0
                or int(shard_parts[1]) <= 1):
            print(f"Error! : SFNN shard token must be like s256x8 , got {arches[6]}.")
            raise SystemExit(1)
        sfnn_common_dims = common_raw
        sfnn_shard_dims = shard_parts[0]
        sfnn_group_count = shard_parts[1]
        sfnn_common_shard = True
        layer_stack_start = 7
    layer_stack_spec = "_".join(arches[layer_stack_start:]) if len(arches) > layer_stack_start else ""
    (layer_stack_name, layer_stack_count, layer_stack_hand_buckets,
        layer_stack_hand_bucket_type,
        layer_stack_king_buckets, layer_stack_king_bucket_type,
        layer_stack_progress_buckets) = parse_sfnn_layer_stack_spec(layer_stack_spec)

    arches = [arches[1], arches[2], arches[3], arches[4], layer_stack_count]

# ============================================================
#                        includes
# ============================================================

if SFNN:
    header = f"""
    // SFNN without PSQT architecture

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

raw_feature_name, raw_feature_hash, raw_feature_dims = FEATURE_INFO.get(input_feature, ("", 0, 0))

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
    #include "sfnn_network.h"

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

    if not sfnn_group_count.isdigit():
        print(f"Error : SFNN group count must be an integer , group = {sfnn_group_count}.")
        raise SystemExit(1)

    if sfnn_common_shard:
        transformed_dims = int(layers[0])
        hidden1_out_dims = int(layers[1]) + 1
        common_dims = int(sfnn_common_dims)
        shard_dims = int(sfnn_shard_dims)
        group_count = int(sfnn_group_count)
        if common_dims + shard_dims * group_count != transformed_dims:
            print(f"Error : common+shard SFNN requires common + shard * group == transformed dimensions. common={common_dims}, shard={shard_dims}, group={group_count}, dims={transformed_dims}.")
            raise SystemExit(1)
        if hidden1_out_dims % group_count != 0:
            print(f"Error : common+shard SFNN requires hidden1+1 divisible by group count. hidden1+1={hidden1_out_dims}, group={group_count}.")
            raise SystemExit(1)
        if common_dims % 64 != 0:
            print(f"Error : common+shard SFNN requires common dimensions to be a multiple of 64. common={common_dims}.")
            raise SystemExit(1)
        if shard_dims % 64 != 0:
            print(f"Error : common+shard SFNN requires shard dimensions to be a multiple of 64. shard={shard_dims}.")
            raise SystemExit(1)

    print(f"layers feature    : {layers}")

    small_sfnn_ft_macro = ""
    if int(layers[0]) < 128:
        small_sfnn_ft_macro = "#define NNUE_SMALL_SFNN_FT"

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {layers[0]};

        // 小幅SFNN専用。従来幅のSFNNでは定義せず、既存の高速経路をそのまま使う。
        {small_sfnn_ft_macro}

        // Number of networks stored in the evaluation file
        constexpr int LayerStacks = {layers[3]};

        #define NNUE_SFNN_HAND_BUCKETS {layer_stack_hand_buckets}
        #define NNUE_SFNN_HAND_BUCKET_TYPE {layer_stack_hand_bucket_type}
        #define NNUE_SFNN_KING_BUCKETS {layer_stack_king_buckets}
        #define NNUE_SFNN_KING_BUCKET_TYPE {layer_stack_king_bucket_type}
        #define NNUE_SFNN_PROGRESS_BUCKETS {layer_stack_progress_buckets}

        // Number of groups for the first affine layer of SFNN.
        // common+shard fc_0でのみ2以上になる。
        constexpr IndexType kHidden1GroupCount = {sfnn_group_count};

        // common+shard fc_0 settings. kHidden1ShardDimensions is per shard.
        constexpr bool kHidden1UsesCommonShard = {"true" if sfnn_common_shard else "false"};
        constexpr IndexType kHidden1CommonDimensions = {sfnn_common_dims};
        constexpr IndexType kHidden1ShardDimensions = {sfnn_shard_dims};

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
    fc_0_type = "Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1>"
    group_count = int(sfnn_group_count)
    if sfnn_common_shard:
        fc_0_type = "Layers::AffineTransformCommonShardInputExplicit<kInputDims, kHidden1Dims + 1, kHidden1CommonDimensions, kHidden1ShardDimensions, kHidden1GroupCount>"
    group_input_dims = int(sfnn_shard_dims) if sfnn_common_shard else 0
    hidden1_output_dims = int(layers[1]) + 1
    enable_common_shard_sfnn_accumulator_propagate = (
        sfnn_common_shard and group_count % 2 == 0 and group_input_dims % 64 == 0
    )
    enable_sparse_sfnn_accumulator_propagate = (
        False
        and not sfnn_common_shard
        and hidden1_output_dims == 8
        and int(layers[0]) % 128 == 0
    )
    sfnn_accumulator_propagate_macro = ""
    if enable_common_shard_sfnn_accumulator_propagate:
        sfnn_accumulator_propagate_macro = "#define NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE"
    elif enable_sparse_sfnn_accumulator_propagate:
        sfnn_accumulator_propagate_macro = "#define NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE"

    structure_string = (
        "SFNN-1536"
        if input_feature == "halfkahm2"
        and layers == ["1536", "15", "32", "9"]
        and layer_stack_name == "K3K3"
        else arch
    )

    header += f"""
        {sfnn_accumulator_propagate_macro}

        using Fc0Layer = {fc_0_type};
        using NetworkBase = SfnnNetwork<Fc0Layer, kInputDims, kHidden1Dims, kHidden2Dims>;

        struct Network : NetworkBase {{
            static std::string GetStructureString() {{
                return "{structure_string}";
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

if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(out_path, "w", encoding = 'utf-8') as f:
    f.write(dedent4(header))

print("..done!")

if dummy_nn_path:
    dummy_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnue_dummy_gen.py")
    subprocess.run([
        sys.executable,
        dummy_script,
        original_arch,
        dummy_nn_path,
        "--dummy-mode",
        args.dummy_mode,
        "--dummy-seed",
        str(args.dummy_seed),
    ], check=True)
