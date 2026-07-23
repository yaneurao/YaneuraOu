# NNUE architecture header generator
#
#  NNUE評価関数のarchitecture headerを動的に生成するPythonで書かれたスクリプト。
# 

import argparse
import os
import random
import struct

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

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

U32_MASK = 0xFFFFFFFF
NNUE_FILE_VERSION = 0x7AF32F16
SFNN_HASH_VALUE = 0x3C203B32
SFNN_FEATURE_TRANSFORMER_HASH = 0x5F134AB8
SFNN_NETWORK_HASH = 0x6333718A
LEB128_MAGIC = b"COMPRESSED_LEB128"

SQ_NB = 81
FILE_NB = 9
FE_END = 1548
F_KING = FE_END
E_KING = F_KING + SQ_NB
FE_END2 = E_KING + SQ_NB

FEATURE_INFO = {
    "halfkp": ("HalfKP(Friend)", 0x5D69D5B8, SQ_NB * FE_END),
    "kp": ("K+P", 0xD3CEE169 ^ ((0x764CFB4B << 1) & U32_MASK) ^ (0x764CFB4B >> 31), SQ_NB * 2 + FE_END),
    "ka2": ("K+A2", 0xD3CEE169 ^ ((0xA20DCB9B << 1) & U32_MASK) ^ (0xA20DCB9B >> 31), SQ_NB * 2 + E_KING),
    "halfkpe9": ("HalfKPE9(Friend)", 0x5D69D5B8, SQ_NB * FE_END * 3 * 3),
    "halfkpvm": ("HalfKP_vm(Friend)", 0x0B6B1D9A, 5 * FILE_NB * FE_END),
    "halfka1": ("HalfKA1(Friend)", 0x5F134CB8, SQ_NB * FE_END2),
    "halfkahm1": ("HalfKA_hm1(Friend)", 0x7F134CB8, 5 * FILE_NB * FE_END2),
    "halfka2": ("HalfKA2(Friend)", 0x5F234CB8, SQ_NB * E_KING),
    "halfkahm2": ("HalfKA_hm2(Friend)", 0x7F234CB8, 5 * FILE_NB * E_KING),
}

def u32(value: int) -> int:
    return value & U32_MASK

def ceil_to_multiple(n: int, base: int) -> int:
    return (n + base - 1) // base * base

def feature_transformer_hash(raw_feature_hash: int, output_dimensions: int, *, sfnn: bool) -> int:
    if sfnn:
        return SFNN_FEATURE_TRANSFORMER_HASH
    return u32(raw_feature_hash ^ output_dimensions)

def input_slice_hash(output_dimensions: int, offset: int = 0) -> int:
    return u32(0xEC42E90D ^ output_dimensions ^ (offset << 10))

def affine_hash(prev_hash: int, output_dimensions: int) -> int:
    return u32((0xCC03DAE4 + output_dimensions) ^ (prev_hash >> 1) ^ u32(prev_hash << 31))

def clipped_relu_hash(prev_hash: int) -> int:
    return u32(0x538D24C7 + prev_hash)

def normal_network_hash(transformed_dims: int, first_layer_multiplier: int, hidden1: int, hidden2: int) -> int:
    h = input_slice_hash(transformed_dims * first_layer_multiplier)
    h = affine_hash(h, hidden1)
    h = clipped_relu_hash(h)
    h = affine_hash(h, hidden2)
    h = clipped_relu_hash(h)
    h = affine_hash(h, 1)
    return h

def write_u32(stream, value: int) -> None:
    stream.write(struct.pack("<I", u32(value)))

def write_i32_zeros(stream, count: int) -> None:
    stream.write(b"\x00\x00\x00\x00" * count)

def write_header(stream, hash_value: int, architecture: str) -> None:
    encoded = architecture.encode("utf-8")
    write_u32(stream, NNUE_FILE_VERSION)
    write_u32(stream, hash_value)
    write_u32(stream, len(encoded))
    stream.write(encoded)

def write_zero_bytes(stream, count: int) -> None:
    chunk = b"\x00" * min(count, 1 << 20)
    while count:
        n = min(count, len(chunk))
        stream.write(chunk[:n])
        count -= n

def write_random_small_bytes(stream, count: int, rng: random.Random, *, negative_byte: int) -> None:
    table = bytes((0, 1, negative_byte)[i % 3] for i in range(256))
    chunk_size = 1 << 20
    while count:
        n = min(count, chunk_size)
        stream.write(rng.randbytes(n).translate(table))
        count -= n

def write_int8_values(stream, count: int, rng: random.Random, mode: str) -> None:
    if mode == "zero":
        write_zero_bytes(stream, count)
    else:
        write_random_small_bytes(stream, count, rng, negative_byte=0xFF)

def write_int16_values(stream, count: int, rng: random.Random, mode: str) -> None:
    if mode == "zero":
        write_zero_bytes(stream, count * 2)
        return

    chunk_values = 1 << 19
    patterns = (b"\x00\x00", b"\x01\x00", b"\xff\xff")
    table = bytes((0, 1, 2)[i % 3] for i in range(256))
    while count:
        n = min(count, chunk_values)
        selector = rng.randbytes(n).translate(table)
        out = bytearray(n * 2)
        for i, s in enumerate(selector):
            out[i * 2:i * 2 + 2] = patterns[s]
        stream.write(out)
        count -= n

def write_sleb128_block_small(stream, count: int, rng: random.Random, mode: str) -> None:
    # zero/random-smallの -1,0,+1 はsigned LEB128で必ず1byteになる。
    if count > U32_MASK:
        raise ValueError(f"LEB128 block is too large: {count} bytes")
    stream.write(LEB128_MAGIC)
    write_u32(stream, count)
    if mode == "zero":
        write_zero_bytes(stream, count)
    else:
        write_random_small_bytes(stream, count, rng, negative_byte=0x7F)

def write_feature_transformer(stream, input_dims: int, transformed_dims: int, raw_feature_hash: int, rng: random.Random, mode: str, *, sfnn: bool) -> None:
    write_u32(stream, feature_transformer_hash(raw_feature_hash, transformed_dims if sfnn else transformed_dims * 2, sfnn=sfnn))
    if sfnn:
        write_sleb128_block_small(stream, transformed_dims, rng, "zero")
        write_sleb128_block_small(stream, transformed_dims * input_dims, rng, mode)
    else:
        write_int16_values(stream, transformed_dims, rng, "zero")
        write_int16_values(stream, transformed_dims * input_dims, rng, mode)

def write_affine_explicit(stream, input_dims: int, output_dims: int, rng: random.Random, mode: str) -> None:
    write_i32_zeros(stream, output_dims)
    write_int8_values(stream, output_dims * ceil_to_multiple(input_dims, 32), rng, mode)

def write_sfnn_network(stream, transformed_dims: int, hidden1: int, hidden2: int, group_count: int, rng: random.Random, mode: str) -> None:
    del group_count  # file layoutはcommon+shardでもdense fc_0互換。
    write_u32(stream, SFNN_NETWORK_HASH)
    write_affine_explicit(stream, transformed_dims, hidden1 + 1, rng, mode)
    write_affine_explicit(stream, hidden1 * 2, hidden2, rng, mode)
    write_affine_explicit(stream, hidden2, 1, rng, mode)

def write_normal_network(stream, transformed_dims: int, first_layer_multiplier: int, hidden1: int, hidden2: int, rng: random.Random, mode: str) -> None:
    write_u32(stream, normal_network_hash(transformed_dims, first_layer_multiplier, hidden1, hidden2))
    write_affine_explicit(stream, transformed_dims * first_layer_multiplier, hidden1, rng, mode)
    write_affine_explicit(stream, hidden1, hidden2, rng, mode)
    write_affine_explicit(stream, hidden2, 1, rng, mode)

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
#     SFNN_ka2_3072_7_64_c1024_s256x8_k3k3 のように、cN_sMxG を置くと
#     fc_0を common N + shard M x G に分割する。
#     SFNN_halfka2_1024_7_64_hand64 のように、hand64を指定すると
#     手番側/非手番側の手駒点を8段階ずつに分けた64 bucketを用いる。
#     hand256 / hand1024も同様に、手番側/非手番側の手駒状態で256/1024 bucketを用いる。
#     SFNN_halfka2_1024_7_64_k9k9 のように指定すると、
#     手番側/非手番側の玉の段を9段階ずつに分けた81 bucketを用いる。
#     SFNN_halfka2_1024_7_64_hand64_k3k3 / hand64_k9k9 のように、hand64と複合できる。
SFNN = False
layer_stack_name = ""
layer_stack_count = ""
layer_stack_hand_buckets = "1"
layer_stack_king_buckets = "1"
sfnn_group_count = "1"
sfnn_common_dims = "0"
sfnn_shard_dims = "0"
sfnn_common_shard = False
if arches[0].startswith("SFNN"):
    SFNN = True
    if len(arches) < 6:
        print("Error! : SFNN architecture name must be like SFNN_halfkahm2_1536-15-32-k3k3 or SFNN_ka2_3072_7_64_c1024_s256x8_k3k3")
        raise SystemExit(1)

    layer_stack_start = 5
    if arches[5].startswith("C"):
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
    if len(arches) <= layer_stack_start:
        print("Error! : SFNN architecture name must end with k3k3, k9k9, hand64, or their long names.")
        raise SystemExit(1)

    layer_stack_spec = "_".join(arches[layer_stack_start:])
    if layer_stack_spec == "K3K3" or layer_stack_spec == "KING3_BY_KING3":
        layer_stack_name = "K3K3"
        layer_stack_count = "9"
        layer_stack_king_buckets = "9"
    elif layer_stack_spec == "K9K9" or layer_stack_spec == "KING9_BY_KING9":
        layer_stack_name = "K9K9"
        layer_stack_count = "81"
        layer_stack_king_buckets = "81"
    elif layer_stack_spec == "HAND64":
        layer_stack_name = "HAND64"
        layer_stack_count = "64"
        layer_stack_hand_buckets = "64"
    elif layer_stack_spec == "HAND256":
        layer_stack_name = "HAND256"
        layer_stack_count = "256"
        layer_stack_hand_buckets = "256"
    elif layer_stack_spec == "HAND1024":
        layer_stack_name = "HAND1024"
        layer_stack_count = "1024"
        layer_stack_hand_buckets = "1024"
    elif layer_stack_spec == "HAND64_K3K3" or layer_stack_spec == "HAND64_KING3_BY_KING3":
        layer_stack_name = "HAND64_K3K3"
        layer_stack_count = str(64 * 9)
        layer_stack_hand_buckets = "64"
        layer_stack_king_buckets = "9"
    elif layer_stack_spec == "HAND64_K9K9" or layer_stack_spec == "HAND64_KING9_BY_KING9":
        layer_stack_name = "HAND64_K9K9"
        layer_stack_count = str(64 * 81)
        layer_stack_hand_buckets = "64"
        layer_stack_king_buckets = "81"
    elif layer_stack_spec == "HAND256_K3K3" or layer_stack_spec == "HAND256_KING3_BY_KING3":
        layer_stack_name = "HAND256_K3K3"
        layer_stack_count = str(256 * 9)
        layer_stack_hand_buckets = "256"
        layer_stack_king_buckets = "9"
    elif layer_stack_spec == "HAND256_K9K9" or layer_stack_spec == "HAND256_KING9_BY_KING9":
        layer_stack_name = "HAND256_K9K9"
        layer_stack_count = str(256 * 81)
        layer_stack_hand_buckets = "256"
        layer_stack_king_buckets = "81"
    elif layer_stack_spec == "HAND1024_K3K3" or layer_stack_spec == "HAND1024_KING3_BY_KING3":
        layer_stack_name = "HAND1024_K3K3"
        layer_stack_count = str(1024 * 9)
        layer_stack_hand_buckets = "1024"
        layer_stack_king_buckets = "9"
    elif layer_stack_spec == "HAND1024_K9K9" or layer_stack_spec == "HAND1024_KING9_BY_KING9":
        layer_stack_name = "HAND1024_K9K9"
        layer_stack_count = str(1024 * 81)
        layer_stack_hand_buckets = "1024"
        layer_stack_king_buckets = "81"
    else:
        print("Error! : SFNN layer stack must be k3k3, k9k9, hand64/256/1024, or hand*_k3k3/k9k9")
        raise SystemExit(1)

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
    #include <cstring>

    #include "../layers/affine_transform_explicit.h"
    #include "../layers/affine_transform_common_shard_input_explicit.h"
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

    header += f"""
        // Number of input feature dimensions after conversion
        // 変換後の入力特徴量の次元数
        constexpr IndexType kTransformedFeatureDimensions = {layers[0]};

        // Number of networks stored in the evaluation file
        constexpr int LayerStacks = {layers[3]};

        #define NNUE_SFNN_HAND_BUCKETS {layer_stack_hand_buckets}
        #define NNUE_SFNN_KING_BUCKETS {layer_stack_king_buckets}

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
    # `sfnn-1536.h`からそのままコピペ。
    fc_0_type = "Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1>"
    common_shard_sfnn_macro = ""
    common_shard_sfnn_accumulator_propagate = ""
    group_count = int(sfnn_group_count)
    if sfnn_common_shard:
        fc_0_type = "Layers::AffineTransformCommonShardInputExplicit<kInputDims, kHidden1Dims + 1, kHidden1CommonDimensions, kHidden1ShardDimensions, kHidden1GroupCount>"
    group_input_dims = int(sfnn_shard_dims) if sfnn_common_shard else 0
    enable_common_shard_sfnn_accumulator_propagate = (
        sfnn_common_shard and group_count % 2 == 0 and group_input_dims % 64 == 0
    )
    if enable_common_shard_sfnn_accumulator_propagate:
        common_shard_sfnn_macro = "#define NNUE_HAS_COMMON_SHARD_SFNN_ACCUMULATOR_PROPAGATE"
        common_shard_sfnn_accumulator_propagate = """
            #if defined(USE_AVX512)
            template <typename AccumulationType>
            const OutputType* PropagateFromAccumulator(const AccumulationType& accumulation,
                                                       Color sideToMove,
                                                       char* buffer) const {
                auto& buf = *reinterpret_cast<Buffer*>(buffer);
                std::memset(buf.ac_sqr_0_out, 0, sizeof(buf.ac_sqr_0_out));

                fc_0.PropagateSfnnFromAccumulator<kInputDims>(accumulation, sideToMove, buf.fc_0_out);
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
            }
            #endif
        """

    header += f"""
        {common_shard_sfnn_macro}

        struct Network {{

            // Define network structure
            // ネットワーク構造の定義
            {fc_0_type} fc_0;
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
{common_shard_sfnn_accumulator_propagate}
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

def normal_network_structure_string(transformed_dims: int, first_layer_multiplier: int, hidden1: int, hidden2: int) -> str:
    input_dims = transformed_dims * first_layer_multiplier
    s = f"InputSlice[{input_dims}(0:{input_dims})]"
    s = f"AffineTransformSparseInput[{hidden1}<-{input_dims}]({s})"
    s = f"ClippedReLU[{hidden1}]({s})"
    s = f"AffineTransform[{hidden2}<-{hidden1}]({s})"
    s = f"ClippedReLU[{hidden2}]({s})"
    s = f"AffineTransform[1<-{hidden2}]({s})"
    return s

def write_dummy_nn(path: str) -> None:
    rng = random.Random(args.dummy_seed)
    mode = args.dummy_mode
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as stream:
        if SFNN:
            transformed_dims = int(layers[0])
            hidden1 = int(layers[1])
            hidden2 = int(layers[2])
            layer_stacks = int(layers[3])
            group_count = int(sfnn_group_count)
            network_name = (
                "SFNN-1536"
                if input_feature == "halfkahm2"
                and layers == ["1536", "15", "32", "9"]
                and layer_stack_name == "K3K3"
                else arch
            )
            architecture_string = (
                f"ModelType=SFNNWithoutPsqt;Features={raw_feature_name}"
                f"[{raw_feature_dims}->{transformed_dims}x2],Network={network_name}"
                f"{{LayerStack={layer_stacks}}}"
            )

            write_header(stream, SFNN_HASH_VALUE, architecture_string)
            write_feature_transformer(stream, raw_feature_dims, transformed_dims, raw_feature_hash, rng, mode, sfnn=True)
            for _ in range(layer_stacks):
                write_sfnn_network(stream, transformed_dims, hidden1, hidden2, group_count, rng, mode)
        else:
            transformed_dims = int(first_layer[0])
            first_layer_multiplier = int(first_layer[1])
            hidden1 = int(layers[1])
            hidden2 = int(layers[2])
            ft_hash = feature_transformer_hash(raw_feature_hash, transformed_dims * 2, sfnn=False)
            net_hash = normal_network_hash(transformed_dims, first_layer_multiplier, hidden1, hidden2)
            architecture_string = (
                f"Features={raw_feature_name}[{raw_feature_dims}->{transformed_dims}x2],"
                f"Network={normal_network_structure_string(transformed_dims, first_layer_multiplier, hidden1, hidden2)}"
            )

            write_header(stream, ft_hash ^ net_hash, architecture_string)
            write_feature_transformer(stream, raw_feature_dims, transformed_dims, raw_feature_hash, rng, mode, sfnn=False)
            write_normal_network(stream, transformed_dims, first_layer_multiplier, hidden1, hidden2, rng, mode)

    print(f"dummy nn.bin path : {path}")
    print(f"dummy mode        : {mode}")
    print(f"dummy seed        : {args.dummy_seed}")
    print(f"dummy nn.bin size : {os.path.getsize(path)} bytes")

if dummy_nn_path:
    write_dummy_nn(dummy_nn_path)
