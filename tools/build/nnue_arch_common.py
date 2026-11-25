# Shared SFNN bucket definitions for architecture and dummy-network generation.
# ヘッダー生成とダミー評価関数生成で共用するSFNNバケットの定義。

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
