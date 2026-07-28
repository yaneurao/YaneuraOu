#!/usr/bin/env python3
# Generate a dummy nn.bin for an NNUE/SFNN architecture.

import argparse
import os
import random
import struct

U32_MASK = 0xFFFFFFFF
NNUE_FILE_VERSION = 0x7AF32F16
SFNN_HASH_VALUE = 0x3C203B32
SFNN_FEATURE_TRANSFORMER_HASH = 0x5F134AB8
SFNN_NETWORK_HASH = 0x6333718A
PROGRESS_HASH_VALUE = 0x6F50524F
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

def strip_prefix_ci(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.upper().startswith(prefix) else text

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

def write_i32_values(stream, count: int, rng: random.Random, mode: str) -> None:
    if mode == "zero":
        write_i32_zeros(stream, count)
        return

    chunk_values = 1 << 18
    patterns = (b"\x00\x00\x00\x00", b"\x01\x00\x00\x00", b"\xff\xff\xff\xff")
    table = bytes((0, 1, 2)[i % 3] for i in range(256))
    while count:
        n = min(count, chunk_values)
        selector = rng.randbytes(n).translate(table)
        out = bytearray(n * 4)
        for i, s in enumerate(selector):
            out[i * 4:i * 4 + 4] = patterns[s]
        stream.write(out)
        count -= n

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

def write_sfnn_network(stream, transformed_dims: int, hidden1: int, hidden2: int, rng: random.Random, mode: str) -> None:
    write_u32(stream, SFNN_NETWORK_HASH)
    write_affine_explicit(stream, transformed_dims, hidden1 + 1, rng, mode)
    write_affine_explicit(stream, hidden1 * 2, hidden2, rng, mode)
    write_affine_explicit(stream, hidden2, 1, rng, mode)

def write_progress_parameters(stream, rng: random.Random, mode: str) -> None:
    write_u32(stream, PROGRESS_HASH_VALUE)
    write_i32_zeros(stream, 1)
    write_i32_values(stream, SQ_NB * FE_END, rng, mode)

def parse_sfnn_layer_stack_spec(layer_stack_spec: str):
    if layer_stack_spec == "":
        return 1, 1, 1, 1

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
    hand_map = {
        "HAND4": 4,
        "HAND16": 16,
        "HAND64": 64,
        "HAND64Z": 64,
        "HAND256": 256,
        "HAND1024": 1024,
    }
    king_map = {
        "K3K3": 9,
        "K9K9": 81,
        "K9K9Z": 81,
        "K13K13Z": 13 * 13,
        "K21K21": 21 * 21,
        "K29K29": 29 * 29,
    }
    progress_values = {2, 3, 4, 8, 16, 32}

    for token in [t for t in normalized.split("_") if t]:
        if token in hand_map:
            if hand_buckets != 1:
                raise ValueError(f"duplicate SFNN hand bucket in {layer_stack_spec}")
            hand_buckets = hand_map[token]
        elif token in king_map:
            if king_buckets != 1:
                raise ValueError(f"duplicate SFNN king bucket in {layer_stack_spec}")
            king_buckets = king_map[token]
        elif token.startswith("PROGRESS"):
            raw = token[len("PROGRESS"):]
            if not raw.isdigit() or int(raw) not in progress_values:
                raise ValueError(f"progress bucket must be progress2/3/4/8/16/32, got {token}")
            if progress_buckets != 1:
                raise ValueError(f"duplicate SFNN progress bucket in {layer_stack_spec}")
            progress_buckets = int(raw)
        else:
            raise ValueError(f"unsupported SFNN layer stack token: {token}")

    return hand_buckets * king_buckets * progress_buckets, hand_buckets, king_buckets, progress_buckets

def write_normal_network(stream, transformed_dims: int, first_layer_multiplier: int, hidden1: int, hidden2: int, rng: random.Random, mode: str) -> None:
    write_u32(stream, normal_network_hash(transformed_dims, first_layer_multiplier, hidden1, hidden2))
    write_affine_explicit(stream, transformed_dims * first_layer_multiplier, hidden1, rng, mode)
    write_affine_explicit(stream, hidden1, hidden2, rng, mode)
    write_affine_explicit(stream, hidden2, 1, rng, mode)

def parse_arch(arch: str):
    arch = strip_prefix_ci(arch, "YANEURAOU_ENGINE_")
    arch = strip_prefix_ci(arch, "NNUE_")
    arch = arch.replace("-", "_").upper()
    parts = arch.split("_")
    if len(parts) <= 3:
        raise ValueError(f"invalid architecture: {arch}")

    if parts[0].startswith("SFNN"):
        if len(parts) < 5:
            raise ValueError(f"invalid SFNN architecture: {arch}")
        layer_stack_start = 5
        if len(parts) > 5 and parts[5].startswith("C"):
            layer_stack_start = 7
        layer_stack_spec = "_".join(parts[layer_stack_start:]) if len(parts) > layer_stack_start else ""
        layer_stack_count, hand_buckets, king_buckets, progress_buckets = parse_sfnn_layer_stack_spec(layer_stack_spec)
        return {
            "arch": arch,
            "sfnn": True,
            "feature": parts[1].lower(),
            "transformed_dims": int(parts[2]),
            "hidden1": int(parts[3]),
            "hidden2": int(parts[4]),
            "layer_stacks": layer_stack_count,
            "hand_buckets": hand_buckets,
            "king_buckets": king_buckets,
            "progress_buckets": progress_buckets,
            "network_name": "SFNN-1536" if parts[1].lower() == "halfkahm2" and parts[2:5] == ["1536", "15", "32"] and layer_stack_count == 9 else arch,
        }

    first_layer = parts[1].lower().split("x")
    if len(first_layer) != 2:
        raise ValueError(f"invalid NNUE first layer: {parts[1]}")
    return {
        "arch": arch,
        "sfnn": False,
        "feature": parts[0].lower(),
        "transformed_dims": int(first_layer[0]),
        "first_layer_multiplier": int(first_layer[1]),
        "hidden1": int(parts[2]),
        "hidden2": int(parts[3]),
    }

def normal_network_structure_string(transformed_dims: int, first_layer_multiplier: int, hidden1: int, hidden2: int) -> str:
    input_dims = transformed_dims * first_layer_multiplier
    s = f"InputSlice[{input_dims}(0:{input_dims})]"
    s = f"AffineTransformSparseInput[{hidden1}<-{input_dims}]({s})"
    s = f"ClippedReLU[{hidden1}]({s})"
    s = f"AffineTransform[{hidden2}<-{hidden1}]({s})"
    s = f"ClippedReLU[{hidden2}]({s})"
    s = f"AffineTransform[1<-{hidden2}]({s})"
    return s

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dummy NNUE/SFNN nn.bin.")
    parser.add_argument("arch")
    parser.add_argument("output")
    parser.add_argument("--dummy-mode", choices=("random-small", "zero"), default="random-small")
    parser.add_argument("--dummy-seed", type=int, default=20260722)
    args = parser.parse_args()

    spec = parse_arch(args.arch)
    feature_name, feature_hash, feature_dims = FEATURE_INFO.get(spec["feature"], ("", 0, 0))
    if not feature_name:
        raise ValueError(f"unsupported feature: {spec['feature']}")

    rng = random.Random(args.dummy_seed)
    dir_name = os.path.dirname(args.output)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(args.output, "wb") as stream:
        if spec["sfnn"]:
            architecture_string = (
                f"ModelType=SFNNWithoutPsqt;Features={feature_name}"
                f"[{feature_dims}->{spec['transformed_dims']}x2],Network={spec['network_name']}"
                f"{{LayerStack={spec['layer_stacks']}}}"
            )
            hash_value = SFNN_HASH_VALUE
            if spec["progress_buckets"] != 1:
                hash_value = u32(hash_value ^ PROGRESS_HASH_VALUE)
            write_header(stream, hash_value, architecture_string)
            write_feature_transformer(stream, feature_dims, spec["transformed_dims"], feature_hash, rng, args.dummy_mode, sfnn=True)
            if spec["progress_buckets"] != 1:
                write_progress_parameters(stream, rng, args.dummy_mode)
            for _ in range(spec["layer_stacks"]):
                write_sfnn_network(stream, spec["transformed_dims"], spec["hidden1"], spec["hidden2"], rng, args.dummy_mode)
        else:
            ft_hash = feature_transformer_hash(feature_hash, spec["transformed_dims"] * 2, sfnn=False)
            net_hash = normal_network_hash(spec["transformed_dims"], spec["first_layer_multiplier"], spec["hidden1"], spec["hidden2"])
            architecture_string = (
                f"Features={feature_name}[{feature_dims}->{spec['transformed_dims']}x2],"
                f"Network={normal_network_structure_string(spec['transformed_dims'], spec['first_layer_multiplier'], spec['hidden1'], spec['hidden2'])}"
            )
            write_header(stream, ft_hash ^ net_hash, architecture_string)
            write_feature_transformer(stream, feature_dims, spec["transformed_dims"], feature_hash, rng, args.dummy_mode, sfnn=False)
            write_normal_network(stream, spec["transformed_dims"], spec["first_layer_multiplier"], spec["hidden1"], spec["hidden2"], rng, args.dummy_mode)

    print(f"dummy nn.bin path : {args.output}")
    print(f"dummy mode        : {args.dummy_mode}")
    print(f"dummy seed        : {args.dummy_seed}")
    print(f"dummy nn.bin size : {os.path.getsize(args.output)} bytes")

if __name__ == "__main__":
    main()
