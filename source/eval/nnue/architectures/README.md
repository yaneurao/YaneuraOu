# NNUE Architecture Headers

このディレクトリにはNNUE/SFNN評価関数のarchitecture headerを置く。
`nnue_arch_gen.py`で生成されるSFNN headerは、LayerStackのbucket数を
`LayerStacks`、手駒bucket数を`NNUE_SFNN_HAND_BUCKETS`、玉位置bucket数を
`NNUE_SFNN_KING_BUCKETS`として出力する。

学習器側も、ここに書かれている入力特徴量、network形状、bucket番号と完全に同じものを
使う必要がある。どれか1つでもずれると、学習した`nn.bin`と探索時の評価関数が一致しない。

## Name Format

通常NNUEは次の形で指定する。

```text
NNUE_<feature>_<FT>x2_<H1>_<H2>
<feature>_<FT>x2_<H1>_<H2>
```

例:

```text
NNUE_halfkp_256x2_32_32
halfkp_512x2-16-32
kp_256x2-32-32
```

`FT`はFeatureTransformerの片側出力次元数、`x2`は手番側視点と相手側視点の2本の
accumulatorを後段に渡すことを表す。したがって通常NNUEの後段入力は`FT * 2`次元になる。
`H1`と`H2`は後段の2つの隠れ層の次元数。

SFNNは次の形で指定する。

```text
SFNN_<feature>_<FT>_<H1>_<H2>[_cC_sSxG][_<layer_stack>]
```

例:

```text
SFNN_halfka2_1024_7_64_k3k3
SFNN_halfka2_1024_7_64_hand64_k3k3
SFNN_ka2_8192_7_64_c0_s1024x8_k3k3
SFNN_halfka2_3072_7_64_c1024_s256x8_k3k3
```

`feature`は入力特徴量、`FT`はSFNN後段に渡す変換後特徴量の次元数、`H1`と`H2`は
SFNN後段の次元数。通常NNUEと違って、SFNNの名前には`x2`を書かない。
`layer_stack`は局面ごとに切り替える後段networkのbucket方式。
`cC_sSxG`はSFNNの`fc_0`を軽くするためのcommon+shard分割指定で、詳細は後述する。

## Normal NNUE

通常NNUEの構造は次の通り。

```text
sparse input features
  -> FeatureTransformer
  -> accumulator[side_to_move] + accumulator[opponent]
  -> InputSlice<FT * 2>
  -> AffineTransformSparseInput<FT * 2, H1>
  -> ClippedReLU
  -> AffineTransform<H1, H2>
  -> ClippedReLU
  -> AffineTransform<H2, 1>
```

`halfkp_256x2-32-32`なら、FeatureTransformer出力は片側256次元で、後段には
`256 * 2 = 512`次元が入力される。そのあと`512 -> 32 -> 32 -> 1`で評価値を出す。

通常NNUEではLayerStackを使わない。入力特徴量を変えたい場合は`halfkp`、`kp`、
`ka2`などの`feature`部分を変える。後段を太くしたい場合は`FT`、`H1`、`H2`を変える。

## SFNN

SFNNは通常NNUEと同じくFeatureTransformerとaccumulatorを使うが、変換後特徴量と
後段networkが通常NNUEと異なる。

`SFNN_*`でビルドすると`SFNNwoPSQT`が定義され、FeatureTransformerでは
`USE_ELEMENT_WISE_MULTIPLY`が有効になる。この経路では、2視点分のaccumulatorを
element-wise multiplyで混ぜて、最終的に`FT`次元の変換後特徴量を後段に渡す。
そのため、SFNNのarchitecture名は`1024_7_64`のようになり、通常NNUEの
`256x2_32_32`のような`x2`表記を持たない。

```text
sparse input features
  -> FeatureTransformer
  -> transformed features
  -> selected LayerStack network
       fc_0: FT -> H1 + 1
       ClippedReLU(fc_0[0..H1))
       SqrClippedReLU(fc_0[0..H1))
       concat: H1 * 2
       fc_1: H1 * 2 -> H2
       ClippedReLU
       fc_2: H2 -> 1
       output += fc_0[H1]
```

`H1 + 1`の最後の1次元は、`fc_2`の出力に加算されるshortcut項である。
`SFNNwoPSQT`という古い名前が残っている箇所があるが、このarchitecture headerでは
このshortcut項を含む。

SFNNでは`LayerStacks`個の後段networkを`nn.bin`に持ち、局面ごとに1つを選んで使う。
例えば`k3k3`なら9個、`hand64_k3k3`なら576個の後段networkを持つ。
FeatureTransformerはLayerStackごとに増えないが、`fc_0`以降の後段パラメータは
基本的にLayerStack数に比例して増える。

## SFNN Common+Shard fc_0

`cC_sSxG`を指定すると、SFNNの`fc_0`をcommon部分とshard部分に分ける。

```text
SFNN_halfka2_3072_7_64_c1024_s256x8_k3k3
```

この例では、FeatureTransformer出力3072次元を次のように見る。

```text
common = 1024
shard  = 256 * 8
total  = 1024 + 256 * 8 = 3072
```

全ての`fc_0`出力はcommon 1024次元を見る。一方、各出力groupは対応するshard
256次元だけを見る。`H1 + 1 = 8`、`G = 8`なら、各出力は
`common 1024 + shard 256 = 1280`次元だけを使う。

`c0_s1024x8`のように`common = 0`も指定できる。この場合は、共通入力なしで、
各出力groupが対応するshardだけを見る。

`cC_sSxG`には制約がある。

```text
C + S * G == FT
(H1 + 1) % G == 0
C は64の倍数
S は64の倍数
G > 1
```

## Input Feature Summary

`nnue_arch_gen.py`が生成できる入力特徴量は次の通り。
次元数の記号は、`SQ_NB = 81`、`FILE_NB = 9`、`FE_END = 1548`、
`E_KING = FE_END + SQ_NB = 1629`、`FE_END2 = E_KING + SQ_NB = 1710`。

| feature | RawFeatures | 入力次元数 | 概要 |
| --- | --- | ---: | --- |
| `halfkp` | `HalfKP(Friend)` | `81 * 1548 = 125388` | 関連玉の位置と、玉以外の駒のBonaPieceの組み合わせ。標準的なNNUE特徴量。 |
| `kp` | `K + P` | `81 * 2 + 1548 = 1710` | 玉位置`K`と、玉以外の駒`P`を足し合わせる軽量特徴量。 |
| `ka2` | `K + A2` | `81 * 2 + 1629 = 1791` | 玉位置`K`と、玉を含む全駒`A2`を足し合わせる軽量特徴量。 |
| `halfkpe9` | `HalfKPE9(Friend)` | `81 * 1548 * 9 = 1128492` | `HalfKP`に先後の利き数bucketを掛けた特徴量。 |
| `halfkpvm` | `HalfKP_vm(Friend)` | `5 * 9 * 1548 = 69660` | `HalfKP`を玉の筋で左右反転して圧縮した特徴量。 |
| `halfka1` | `HalfKA1(Friend)` | `81 * 1710 = 138510` | 関連玉と、両玉を区別して含む全駒BonaPieceの組み合わせ。 |
| `halfkahm1` | `HalfKA_hm1(Friend)` | `5 * 9 * 1710 = 76950` | `halfka1`を玉の筋で左右反転して圧縮した特徴量。 |
| `halfka2` | `HalfKA2(Friend)` | `81 * 1629 = 131949` | 関連玉と、玉を含む全駒`A2`の組み合わせ。 |
| `halfkahm2` | `HalfKA_hm2(Friend)` | `5 * 9 * 1629 = 73305` | `halfka2`を玉の筋で左右反転して圧縮した特徴量。 |

`Friend`は手番側、`Enemy`は相手側を意味する。通常は`Friend`特徴量だけをheaderに書くが、
評価時には手番側視点と相手側視点のaccumulatorを使う。

## Input Feature Details

### halfkp

`halfkp`は、関連玉の位置と玉以外の駒を掛け合わせる。

```text
index = king_square * FE_END + bona_piece
```

玉と各駒の関係を直接表現できるので、通常NNUEの標準的な特徴量として使いやすい。
一方、関連玉が動いたときは、その玉に対応するaccumulatorを全再計算する必要がある。

### kp

`kp`は`K`と`P`のFeatureSetである。

```text
K: 先手玉/後手玉の位置
P: 玉以外の駒のBonaPiece
```

`halfkp`のように「玉位置 x 駒」の直積を持たないので、入力次元数は非常に小さい。
玉が動いても全再計算になりにくいが、表現力は`halfkp`より低い。

### ka2

`ka2`は`K`と`A2`のFeatureSetである。
`A2`は玉を含む全駒のBonaPieceで、後手玉planeを自玉planeにマージするv2 encodingを使う。

`kp`より盤上情報は多いが、`halfka2`のような「関連玉 x 全駒」の直積は持たない。
大きなFeatureTransformerを使うSFNN実験では、`halfka2`よりFT更新が軽い候補になる。

### halfka1

`halfka1`は、関連玉の位置と、両玉を区別して含む全駒BonaPieceを掛け合わせる。
`halfkp`より全駒情報が多いが、入力次元数も大きい。

現在は、玉planeを圧縮した`halfka2`のほうが使いやすいことが多い。

### halfka2

`halfka2`は、関連玉の位置と`A2`を掛け合わせる。

```text
index = king_square * E_KING + a2_piece
```

玉を含む全駒情報を、関連玉との直積として表現できる。`SFNN_halfka2_...`系の
主な入力特徴量である。関連玉が動いたときは全再計算になる。

### halfkpvm / halfkahm1 / halfkahm2

これらは玉が6筋から9筋にいる場合、盤面を左右反転して1筋から4筋側に寄せる。
玉の筋方向の対称性を使うので、関連玉の筋は9通りではなく5通り相当になる。

```text
dimensions = 5 * FILE_NB * piece_dimensions
```

パラメータ数を減らせる一方、左右非対称な情報はnetwork側で表現しにくくなる。

### halfkpe9

`halfkpe9`は`halfkp`に利き数bucketを掛けた特徴量である。
先手の利き数と後手の利き数をそれぞれ0、1、2以上の3段階に丸め、`3 * 3 = 9`
通りを使う。持ち駒は利き数0として扱う。

入力次元数が`halfkp`の9倍になるので、表現力は上がるが、メモリ量と更新コストが重い。

## Architecture Choices

`halfkp_256x2_32_32`は、従来型の標準NNUEとして扱いやすい。
比較基準や小さめの評価関数を作るときに向く。

`halfkp_1024x2_8_64`のようにFTを太くして後段を細くする形は、FeatureTransformer側に
表現力を寄せる設計である。FT更新コストと後段計算量のバランスを見る必要がある。

`SFNN_halfka2_1024_7_64_k3k3`は、`halfka2`入力、1024次元FT、`7 -> 64`の後段、
玉段による9 LayerStackを持つSFNNである。SFNN系の基本形として使いやすい。

`SFNN_halfka2_1024_7_64_hand64`のようなhand bucket系は、手駒状態によって後段networkを
切り替える。終盤寄りの手駒が多い局面と、序盤寄りの手駒が少ない局面で別の後段を
使えるが、bucket数に比例して後段パラメータが増え、bucketごとの教師密度は下がる。

`SFNN_ka2_8192_7_64_c0_s1024x8_k3k3`のような大きなFT + shard splitは、
FTの表現力を増やしつつ、後段`fc_0`の計算量を抑えるための実験用構成である。
実測NPSはaccumulatorのコピー、メモリアクセス、SIMD実装の有無にも左右されるため、
MAC数だけでは決まらない。

## LayerStack Name

SFNN architecture名の末尾でLayerStackの分岐方式を指定する。

```text
SFNN_halfka2_1024_7_64_k3k3
SFNN_halfka2_1024_7_64_k9k9
SFNN_halfka2_1024_7_64_hand64
SFNN_halfka2_1024_7_64_hand256
SFNN_halfka2_1024_7_64_hand1024
SFNN_halfka2_1024_7_64_hand64_k3k3
SFNN_halfka2_1024_7_64_hand256_k9k9
```

`king3_by_king3`は`k3k3`、`king9_by_king9`は`k9k9`の別名として使える。
同様に、`hand64_king3_by_king3`なども対応する。

## King Buckets

玉位置bucketは、手番側玉と非手番側玉を「手番側から見た向き」にそろえて計算する。

`stm`を手番、`f_king`を手番側玉、`e_king`を非手番側玉とする。

```cpp
f_rank = stm == BLACK ? rank_of(f_king)      : rank_of(Inv(f_king));
e_rank = stm == BLACK ? rank_of(Inv(e_king)) : rank_of(e_king);
```

`rank_of()`の値は0..8。

### k3k3

`k3k3`は、手番側玉の段を3区分、非手番側玉の段を3区分に分ける。
合計は`3 * 3 = 9` bucket。

```cpp
f_index = [0,0,0,3,3,3,6,6,6][f_rank];
e_index = [0,0,0,1,1,1,2,2,2][e_rank];
king_bucket = f_index + e_index; // 0..8
```

### k9k9

`k9k9`は、手番側玉の段を9区分、非手番側玉の段を9区分に分ける。
合計は`9 * 9 = 81` bucket。

```cpp
king_bucket = f_rank * 9 + e_rank; // 0..80
```

`k9k9`は`k3k3`より細かい局面分類になるが、LayerStack数が9倍になる。
教師密度、ファイルサイズ、メモリ使用量とのトレードオフを確認すること。

## Hand Buckets

手駒bucketは、手番側の手駒bucketを上位側、非手番側の手駒bucketを下位側に置く。

### hand64

`hand64`は、手駒を点数化して片側8段階に分ける。
合計は`8 * 8 = 64` bucket。

```cpp
score =
    pawn_count
  + (lance_count + knight_count) * 2
  + (silver_count + gold_count) * 3
  + (bishop_count + rook_count) * 5;

single = min((score + 3) / 4, 7); // 0..7
hand_bucket = stm_single * 8 + non_stm_single; // 0..63
```

点数範囲は次の通り。

```text
single 0: score 0
single 1: score 1..4
single 2: score 5..8
single 3: score 9..12
single 4: score 13..16
single 5: score 17..20
single 6: score 21..24
single 7: score 25以上
```

`hand64`は手駒量による粗い進行度分類である。手駒の種類を細かく区別しないので、
bucket数を抑えながら序盤/中盤/終盤の違いを後段に渡したい場合に使う。

### hand256

`hand256`は、手駒の種類グループの有無を4bitで表す。
片側16通り、合計は`16 * 16 = 256` bucket。

```cpp
single = 0;
if (pawn_count + lance_count + knight_count > 0) single |= 1;
if (silver_count + gold_count > 0)               single |= 2;
if (bishop_count > 0)                            single |= 4;
if (rook_count > 0)                              single |= 8;

hand_bucket = stm_single * 16 + non_stm_single; // 0..255
```

`hand256`は、手駒量ではなく手駒の種類を見たい場合のbucketである。
歩香桂を同じ軽い駒グループ、金銀を同じ金駒グループとしてまとめる。

### hand1024

`hand1024`は、手駒の種類グループの有無を5bitで表す。
片側32通り、合計は`32 * 32 = 1024` bucket。

```cpp
single = 0;
if (pawn_count > 0)                  single |= 1;
if (lance_count + knight_count > 0)  single |= 2;
if (silver_count + gold_count > 0)   single |= 4;
if (bishop_count > 0)                single |= 8;
if (rook_count > 0)                  single |= 16;

hand_bucket = stm_single * 32 + non_stm_single; // 0..1023
```

`hand1024`は`hand256`より歩の有無を独立に扱う。手駒状態の分類は細かくなるが、
LayerStack数が増えるため、教師密度が足りないと過学習しやすい。

## Combined Buckets

手駒bucketと玉位置bucketを組み合わせるときは、手駒bucketを上位側に置く。

```cpp
final_bucket = hand_bucket * king_bucket_count + king_bucket;
```

例:

```text
hand64_k3k3     : 64 * 9     = 576 buckets
hand64_k9k9     : 64 * 81    = 5184 buckets
hand256_k3k3    : 256 * 9    = 2304 buckets
hand256_k9k9    : 256 * 81   = 20736 buckets
hand1024_k3k3   : 1024 * 9   = 9216 buckets
hand1024_k9k9   : 1024 * 81  = 82944 buckets
```

手駒bucketを使わない場合は`hand_bucket = 0`相当、玉位置bucketを使わない場合は
`king_bucket_count = 1`、`king_bucket = 0`相当として扱う。

## Generated Headers

`nnue_arch_gen.py`はarchitecture名からheaderを生成する。

```bash
python3 source/eval/nnue/architectures/nnue_arch_gen.py \
  SFNN_halfka2_1024_7_64_hand64_k3k3
```

第2引数を指定すると、出力先ディレクトリを変更できる。

```bash
python3 source/eval/nnue/architectures/nnue_arch_gen.py \
  SFNN_halfka2_1024_7_64_hand64_k3k3 \
  /tmp/nnue_arch
```

探索速度の概算だけ見たい場合は、dummyの`nn.bin`を生成できる。

```bash
python3 source/eval/nnue/architectures/nnue_arch_gen.py \
  SFNN_halfka2_1024_7_64_hand64_k3k3 \
  --write-dummy-nn /tmp/nn.bin
```

dummyのデフォルトは`random-small`である。完全に0で初期化したい場合は
`--dummy-mode zero`を指定する。

## Implementation

探索時のbucket計算は`source/eval/nnue/evaluate_nnue.cpp`の
`stack_index_for_nnue()`に実装されている。

architecture headerの生成は`source/eval/nnue/architectures/nnue_arch_gen.py`が行う。
新しい入力特徴量やLayerStack方式を追加するときは、生成器、探索時bucket計算、
学習器側のbucket計算を同時に更新すること。
