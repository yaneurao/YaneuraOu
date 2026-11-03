# 調査・実験メモの目次

探索部の不具合調査、Stockfishからの移植検討、SFNNの高速化実験、評価関数のファイル形式に関する資料です。
各資料は記載日のソース・モデル・環境を対象にしています。現在の実装状況や、別環境での性能を保証するものではありません。

## 目的から探す

- **探索部の不具合と対処順を知りたい** → [探索部不具合の処理台帳](search-audit-20261029.txt)
- **Stockfishの変更を取り込むべきか調べたい** → [全280コミットの移植可否レビュー](stockfish-port-review-20261029.txt)
- **USE_FINNY_RESETの効果と計測の問題を確認したい** → [計測監査・再比較](finny-reset-measurement-audit-20261030.txt)
- **追加の高速化案と不採用の理由を知りたい** → [Finny reset追加最適化の実験記録](finny-reset-experiments-20261030.txt)
- **評価関数ファイルの形式を知りたい** → [16bit重み](nn16.md) / [進行度ファイル](progress-format.md)

## 探索部・Stockfishの調査

調査日：2026-10-29。まず処理台帳を読み、個々のStockfishコミットの根拠を確認するときにレビューと一覧を使います。

| ファイル | 内容・用途 |
| --- | --- |
| [search-audit-20261029.txt](search-audit-20261029.txt) | 不具合の処理台帳。SA-01〜SA-18、性能候補PC-01の内容、推奨処理順、再現条件、調査の訂正を記録。再現済みの問題と未再現の候補を区別しています。 |
| [stockfish-port-review-20261029.txt](stockfish-port-review-20261029.txt) | 2026-04-29〜2026-10-29の期間に抽出したStockfish全280コミットを個別に精査した詳細レビュー。移植可否、対応するやねうら王の実装、理由、依存関係、必要な検証を記載。 |
| [search-audit-20261029-stockfish.tsv](search-audit-20261029-stockfish.tsv) | 同じ280コミットの一覧。SF-001〜SF-280のID、コミット、判定、理由などを収録。検索・比較用です。 |

調査時点では、これらの資料に基づく本体の修正は行っていません。対策を進める際は、現行コードと台帳の状態を確認してください。

## SFNN・Finny resetの高速化実験

調査日：2026-10-30。主な測定条件はRyzen 9 9950X、AVX512VNNI、32スレッド、`SFNN_halfka2_2304_15_64_progress8`、評価関数`evalTuyo12-20261025`です。

**USE_FINNY_RESETの結果は、計測監査・再比較を優先して読んでください。** 旧報の「約94%高速化」を安定した効果として扱った結論は撤回されています。再比較は約+19.3%でしたが、各版2回の当該条件での結果であり、旧測定でOFFが極端に遅かった直接原因は未確定です。

| ファイル | 内容・用途 |
| --- | --- |
| [finny-reset-measurement-audit-20261030.txt](finny-reset-measurement-audit-20261030.txt) | **USE_FINNY_RESETについて最初に読む資料。** ビルドフラグ、バイナリ、集計、外部時計、CPU時間、ページ状態などの調査と、新規ビルドによる再比較を記録。確認できた事実と未確定の原因を区別しています。 |
| [finny-reset-measurement-audit-20261030.json](finny-reset-measurement-audit-20261030.json) | 上記監査の機械可読な証跡。ビルド条件、測定値、ハッシュ、診断結果などを詳しく確認するためのデータです。 |
| [finny-reset-long-20261030.txt](finny-reset-long-20261030.txt) | USE_FINNY_RESETの旧OFF/ON比較。約94%という結果を記録していますが、その解釈は後続監査で撤回。旧ログと訂正の経緯を確認するために保存しています。 |
| [finny-reset-experiments-20261030.txt](finny-reset-experiments-20261030.txt) | AVX512による特徴インデックスの差分判定と、FT重みの可逆圧縮の実験記録。実装案、正しさ検証、短時間測定、追加調整、不採用の判断、再現方法を記載。 |
| [finny-mask-long-20261030.txt](finny-mask-long-20261030.txt) | 差分判定AVX512版を長時間で再比較した結果。12組の比較で平均約2.5%低下し、この条件では採用しないと判断した記録。 |
| [finny-reset-experiments-20261030.patch](finny-reset-experiments-20261030.patch) | 不採用にした差分判定AVX512版・FT可逆圧縮版の実験実装を保存したパッチ。採用済みコードではありません。適用条件・手順は実験記録を参照してください。 |

本文中の`build/`以下の生ログ・バイナリはGit管理対象外です。このフォルダの資料だけでは、すべての生ログが揃うわけではありません。測定用スクリプトもこのフォルダに保存しています。使い方は[実験用ベンチマーク](#実験用ベンチマーク)を参照してください。

## 評価関数のファイル形式

| ファイル | 内容・用途 |
| --- | --- |
| [nn16.md](nn16.md) | SFNNの16bit重み版。ビルドフラグ、`nn16.bin`の配置、ファイル形式、量子化倍率、整数演算、回帰テストの説明。 |
| [progress-format.md](progress-format.md) | 進行度bucketを使うSFNNの`progress.bin`形式。配置、重みの並びと型、読込時の整数化、bucket境界、`nn.bin`との関係、旧形式との非互換性を説明。本文は英語です。 |

## 実験用ベンチマーク

以下の4本は、Finny reset・FT幅2304などの最適化を調べるための実験用です。Python 3.11以上（`hashlib.file_digest`を使用）、Linuxの`taskset`などを前提とし、ビルドする場合はmakeとClangも必要です。一般用途の`bench.py`とは対象と条件が異なります。

| ファイル | 比較対象・用途 | 主な設定・制約 |
| --- | --- | --- |
| [bench_finny_reset.py](bench_finny_reset.py) | `USE_FINNY_RESET`なしの`baseline`と、ありの`finny_reset`をビルド・検証・比較。 | `--arch`、`--eval-dir`必須。既定は1スレッド、CPU 2に固定、各版4回。`--threads`と`--pin`を変更可能。 |
| [bench_finny_reset_verify.py](bench_finny_reset_verify.py) | 過去の測定を監査して作った再比較用。全翻訳単位のビルド引数と実マクロを照合し、外部時計・CPU時間・メモリ情報も保存。 | アーキテクチャは`SFNN_halfka2_2304_15_64_progress8`、AVX512VNNI、clang++、32スレッド、CPU 0–31、各版2回に固定。 |
| [bench_sfnn_experiments.py](bench_sfnn_experiments.py) | FTの8/12/24レジスタ版、固定差分更新の特化、差分相殺、FTと後段の融合の6構成を比較。 | `--arch`、`--eval-dir`必須。AVX512VNNI・clang++に固定。共通でlazy evaluate・Finny tables・Finny resetを有効化。 |
| [bench_finny_mask_long.py](bench_finny_mask_long.py) | 検証済みバイナリ2版の長時間比較。ビルドは行わず、過去の正当性検証をハッシュ照合後に再利用。 | `--candidate mask`または`finny_reset`。既定は32スレッド、CPU 0–31、各版12回。`--reference`と`--bin-dir`に既存の結果・バイナリが必要。 |

### 実行例

```sh
# モデルのアーキテクチャを指定できるFinny reset比較
python3 tools/memo/bench_finny_reset.py --build \
  --arch SFNN_halfka2_2304_15_64_progress8 \
  --eval-dir ../inbox/evalTuyo12-20261025 \
  --threads 1 --pin 2 --rounds 2

# 監査付きの32スレッド比較（このアーキテクチャとCPU配置に固定）
python3 tools/memo/bench_finny_reset_verify.py --build \
  --eval-dir ../inbox/evalTuyo12-20261025
```

共通の測定対象は`bench`既定の4局面で、ノード数は`--nodes`で**1局面あたり**の値を指定します。評価値倍率は`FV_SCALE=24`です。`--warmup-nodes`は測定前に別の`bench`を実行するための指定ですが、`bench`ごとに探索スレッドが作り直されるため、**Finnyキャッシュが温まった状態を引き継ぐものではありません**。

`bench_finny_reset.py`の共通ビルドフラグは既定で`-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES`、比較版だけ`-DUSE_FINNY_RESET`を追加します。`--extra-cppflags`で共通フラグを変更できます。正当性は`test nn test_accumulator`と評価値チェックサムで確認します。

`bench_finny_reset_verify.py`も同じフラグ差で比較し、1サンプルごとにプロセスを終了させます。`--build`なしで再利用する場合は、出力先に既存の`off`・`on`バイナリと`build-manifest.json`が必要です。他の3本の詳しい引数は`--help`を参照してください。

各スクリプトは`--output`以下に生ログと`results.json`などを保存します。既定の保存先は順に`build/finny-reset/`、`build/finny-reset-verified-20261030/`、`build/sfnn-experiments/`、`build/finny-mask-long-20261030/`です。別の測定結果を残すには出力先を分けてください。

実験の結果・採否・計測上の訂正は、この目次の各資料から確認できます。`mask`版などの不採用実装は現行ソースに常時含まれるものではありません。実験フラグを指定しただけで意図した比較が成立するとは限らないため、対応する実装・パッチがあることを確認してください。
