# 補助スクリプト

速度比較、自己対局のためのスクリプトです。特記しない限り、コマンド例はリポジトリのルート（`source/`と`tools/`がある場所）から実行します。

Pythonの世代と実行環境はファイルごとに異なります。旧`readme.txt`の「すべてPython 2.7用」という説明は現在の構成には当てはまりません。以下は実装を確認して整理した説明であり、全環境での実行確認を示すものではありません。

## 目的から探す

- ビルド用スクリプト（Linux・Windows・Android・WebAssembly） → [tools/build/](../build/README.md)
- 通常のNPS比較 → [bench.py](#通常のベンチマーク)
- Finny reset・SFNN最適化の再現実験 → [tools/memo/](../memo/README.md#実験用ベンチマーク)
- 旧自己対局環境を使う → [自己対局](#自己対局旧環境向け)

## 通常のベンチマーク

[bench.py](bench.py)はPython 3用です。1つまたは2つのエンジンで`bench`を繰り返し実行し、NPSの平均・ばらつき・差を表示します。エンジンごとに評価関数ディレクトリを指定できます。

依存パッケージは`pyyaml`、`pexpect`、`pandas`、`statsmodels`、`cpuid`です。Windowsでは`PopenSpawn`、それ以外では`pexpect.spawn`を使います。

```sh
python3 tools/script/bench.py --loop 4 --log build/bench.log \
  ./engine-base /path/to/eval-base ./engine-test /path/to/eval-test
```

`engine1 eval1`だけなら単独測定です。`--cmd`には`bench`以降の引数をまとめて渡せます。ログの既定名は実行ディレクトリの`bench.log`です。ログの親ディレクトリは事前に作成してください。

このスクリプトはCPU affinityを指定せず、2版比較では毎回engine1→engine2の順に実行します。表示名`P(speedup > 0)`の値は実装上はt検定のp値であり、「高速である確率」ではありません。

## 自己対局（旧環境向け）

次の2本はPython 2構文のままの旧自己対局ツールです。Python 3ではそのまま実行できません。現在のエンジンや環境での動作確認済みツールとしては扱わず、旧環境の再現・改修の資料として参照してください。

| ファイル | 内容・前提 |
| --- | --- |
| [engine_invoker1.py](engine_invoker1.py) | 外部の`local-game-serverV350.exe`を起動して並列対局を実行。`engine-config1.txt`・`engine-config2.txt`を実行ディレクトリに生成し、勝敗とレーティング差を集計。Windowsのパスと実行ファイルを前提にしています。 |
| [engine_invoker5.py](engine_invoker5.py) | 対局管理をPython側で実装。USIエンジンを直接複数起動し、定跡局面、先後交代、時間設定、棋譜出力、勝敗集計を扱います。`cmd.exe /c start /NODE`とWindowsの非同期パイプ処理を使うWindows専用実装です。 |

`engine_invoker1.py`の引数順は次のとおりです。

```text
HOMEPATH engine1 evaldir1 engine2 evaldir2 threads loop numa { rtime1 ... rtimeN }
```

この版は`<home>/exe/local-game-serverV350.exe`と`<home>/book/records1.sfen`を参照します。相手側の評価関数に`0`、`1`、…という連番サブディレクトリがあれば、順番に比較します。

`engine_invoker5.py`の主な引数は`--home`、`--engine1`、`--eval1`、`--engine2`、`--eval2`、`--cores`、`--loop`、`--engine_threads`、`--hash1`、`--hash2`、`--time`です。実行ファイルは`<home>/exe/`、評価関数は`<home>/eval/`、開始局面集は`<home>/book/records2016_10818.sfen`を参照します。

`--time`は`r100`（ランダム時間）、`b1000`（秒読み）、`t300000/i3000`（持ち時間＋加算）、`d6`（深さ指定）など。時間値はミリ秒で、カンマ区切りで複数条件、ピリオド区切りでエンジンごとに異なる条件を指定します。`--cores / --engine_threads`が並列対局数になります。

`engine_invoker5.py`は実行ディレクトリに日時付きの`.sfen`を出力し、`--cpu 256`では通信ログも有効にします。現行のUSIオプションや対局規則に追従しているとは限りません。

## このフォルダ以外の関連ツール

- [nnue_arch_gen.py](../../source/eval/nnue/architectures/nnue_arch_gen.py)：NNUE/SFNNのアーキテクチャ名からC++ヘッダーを生成。使い方は[アーキテクチャのREADME](../../source/eval/nnue/architectures/README.md)を参照。
- 旧`readme.txt`にあった`eval_bin_to_cpp_literal.py`は、現在のリポジトリにはありません。
- Dockerでのエンジンビルドは[tools/docker/README.md](../docker/README.md)、調査・実験結果は[tools/memo/README.md](../memo/README.md)を参照。
