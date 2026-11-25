# Shared architecture rules for NNUE/SFNN build generators.
# NNUE/SFNNのヘッダー生成とダミー評価関数生成で共用する構成規則。

from typing import NamedTuple

# Board and BonaPiece constants for the default shogi feature encoding.
# 標準の将棋用特徴量で使う盤面とBonaPieceの定数。
U32_MASK = 0xFFFFFFFF
SQ_NB = 81
FILE_NB = 9
FE_END = 1548
F_KING = FE_END
E_KING = F_KING + SQ_NB
FE_END2 = E_KING + SQ_NB


class NnueFeatureSpec(NamedTuple):
    headers: tuple
    cpp_types: str
    name: str
    hash_value: int
    dimensions: int


# Keep C++ feature selection and dummy-file metadata in the same entry.
# C++の特徴量選択とダミー評価関数のメタデータを、同じ項目で管理する。
NNUE_FEATURES = {
    "halfkp": NnueFeatureSpec(
        ("half_kp.h",), "Features::HalfKP<Features::Side::kFriend>",
        "HalfKP(Friend)", 0x5D69D5B8, SQ_NB * FE_END),
    "kp": NnueFeatureSpec(
        ("k.h", "p.h"), "Features::K, Features::P",
        "K+P", 0xD3CEE169 ^ ((0x764CFB4B << 1) & U32_MASK) ^ (0x764CFB4B >> 31), SQ_NB * 2 + FE_END),
    "ka2": NnueFeatureSpec(
        ("k.h", "a2.h"), "Features::K, Features::A2",
        "K+A2", 0xD3CEE169 ^ ((0xA20DCB9B << 1) & U32_MASK) ^ (0xA20DCB9B >> 31), SQ_NB * 2 + E_KING),
    "halfkpe9": NnueFeatureSpec(
        ("half_kpe9.h",), "Features::HalfKPE9<Features::Side::kFriend>",
        "HalfKPE9(Friend)", 0x5D69D5B8, SQ_NB * FE_END * 3 * 3),
    "halfkpvm": NnueFeatureSpec(
        ("half_kp_vm.h",), "Features::HalfKP_vm<Features::Side::kFriend>",
        "HalfKP_vm(Friend)", 0x0B6B1D9A, 5 * FILE_NB * FE_END),
    "halfka1": NnueFeatureSpec(
        ("half_ka1.h",), "Features::HalfKA1<Features::Side::kFriend>",
        "HalfKA1(Friend)", 0x5F134CB8, SQ_NB * FE_END2),
    "halfkahm1": NnueFeatureSpec(
        ("half_ka_hm1.h",), "Features::HalfKA_hm1<Features::Side::kFriend>",
        "HalfKA_hm1(Friend)", 0x7F134CB8, 5 * FILE_NB * FE_END2),
    "halfka2": NnueFeatureSpec(
        ("half_ka2.h",), "Features::HalfKA2<Features::Side::kFriend>",
        "HalfKA2(Friend)", 0x5F234CB8, SQ_NB * E_KING),
    "halfkahm2": NnueFeatureSpec(
        ("half_ka_hm2.h",), "Features::HalfKA_hm2<Features::Side::kFriend>",
        "HalfKA_hm2(Friend)", 0x7F234CB8, 5 * FILE_NB * E_KING),
}

# Bucket tables contain (number of buckets, C++ bucket type ID).
# バケット表は（バケット数、既存アーキテクチャの型ID）の組。型IDは互換性維持のため変更しない。

SFNN_BUCKET_ALIASES = {
    "KING3_BY_KING3": "K3K3",
    "KING3Z_BY_KING3Z": "K3K3Z",
    "KING3ZONE_BY_KING3ZONE": "K3K3Z",
    "KING9_BY_KING9": "K9K9",
    "KING9Z_BY_KING9Z": "K9K9Z",
    "KING9ZONE_BY_KING9ZONE": "K9K9Z",
    "KING13Z_BY_KING13Z": "K13K13Z",
    "KING13ZONE_BY_KING13ZONE": "K13K13Z",
    "KING21_BY_KING21": "K21K21",
    "KING29_BY_KING29": "K29K29",
}

SFNN_HAND_BUCKETS = {
    "HAND4": (4, 1),
    "HAND16": (16, 2),
    "HAND64": (64, 3),
    "HAND64Z": (64, 4),
    "HAND256": (256, 5),
    "HAND1024": (1024, 6),
}

SFNN_KING_BUCKETS = {
    "K3K3": (9, 1),
    "K3K3Z": (9, 7),
    "K9K9": (81, 2),
    "K21K21": (441, 3),
    "K29K29": (841, 4),
    "K9K9Z": (81, 5),
    "K13K13Z": (169, 6),
}

SFNN_PROGRESS_BUCKETS = frozenset({2, 3, 4, 8, 16, 32})


def normalize_sfnn_bucket_spec(spec: str) -> str:
    for long_name, short_name in SFNN_BUCKET_ALIASES.items():
        spec = spec.replace(long_name, short_name)
    return spec


def strip_nnue_arch_prefixes(arch: str) -> str:
    # Keep spelling and separators intact: the header filename uses them.
    # ヘッダーのファイル名に使うので、大文字小文字や区切り文字はそのまま残す。
    for prefix in ("YANEURAOU_ENGINE_", "NNUE_"):
        if arch.upper().startswith(prefix):
            arch = arch[len(prefix):]
    return arch


def sfnn_hidden1_output_dims(hidden1_dims: int) -> int:
    # The shortcut occupies one extra fc_0 output, including H1=7 and H1=15.
    # H1=7・15などのショートカットは、fc_0の追加出力を1つ使う。
    if hidden1_dims % 8 == 7:
        return hidden1_dims + 1
    if hidden1_dims % 8 == 0:
        return hidden1_dims
    raise ValueError(f"SFNN H1 must be 8n with no shortcut, or 8n-1 with shortcut: {hidden1_dims}")


class SfnnBucketSpec(NamedTuple):
    canonical: str
    layer_stacks: int
    hand_buckets: int
    hand_type: int
    king_buckets: int
    king_type: int
    progress_buckets: int


class SfnnBucketError(ValueError):
    # Callers keep their own CLI diagnostics and exception formats.
    # 診断メッセージや例外の形式は各ツール側で従来どおりに整える。
    def __init__(self, kind: str, token: str):
        super().__init__(kind, token)
        self.kind = kind
        self.token = token


def parse_sfnn_buckets(spec: str) -> SfnnBucketSpec:
    # Parse and validate once for both generators; keep the trainer's bucket order.
    # 両生成器で同じ解析・検証を使い、学習器と共有するバケットの並びを維持する。
    hand_buckets = king_buckets = progress_buckets = 1
    hand_type = king_type = 0
    hand_name = king_name = progress_name = ""
    for token in filter(None, normalize_sfnn_bucket_spec(spec).split("_")):
        if token in SFNN_HAND_BUCKETS:
            if hand_buckets != 1:
                raise SfnnBucketError("hand", token)
            hand_name = token
            hand_buckets, hand_type = SFNN_HAND_BUCKETS[token]
        elif token in SFNN_KING_BUCKETS:
            if king_buckets != 1:
                raise SfnnBucketError("king", token)
            king_name = token
            king_buckets, king_type = SFNN_KING_BUCKETS[token]
        elif token.startswith("PROGRESS"):
            raw = token[len("PROGRESS"):]
            if not raw.isdigit() or int(raw) not in SFNN_PROGRESS_BUCKETS:
                raise SfnnBucketError("progress_value", token)
            if progress_buckets != 1:
                raise SfnnBucketError("progress", token)
            progress_name = token
            progress_buckets = int(raw)
        else:
            raise SfnnBucketError("unknown", token)

    canonical = "_".join(filter(None, (hand_name, king_name, progress_name))) or "NONE"
    return SfnnBucketSpec(
        canonical, hand_buckets * king_buckets * progress_buckets,
        hand_buckets, hand_type, king_buckets, king_type, progress_buckets,
    )
