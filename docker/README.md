# Docker build guide

このディレクトリには、やねうら王とふかうら王を Docker でビルド・起動するための設定を置いています。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `Dockerfile.yaneuraou` | 通常のやねうら王をビルドします。デフォルトは NNUE 版です。 |
| `Dockerfile.fukauraou-cpu` | GPU なしで動く、ONNX Runtime CPU 版のふかうら王をビルドします。 |
| `Dockerfile.fukauraou-tensorrt` | NVIDIA GPU 用の TensorRT 版ふかうら王をビルドします。 |
| `compose.yaneuraou.yaml` | VSCode Docker 拡張などから、やねうら王を起動しやすくする compose ファイルです。 |
| `compose.fukauraou-cpu.yaml` | GPU なし環境用のふかうら王 CPU 版 compose ファイルです。 |
| `compose.fukauraou-tensorrt.yaml` | NVIDIA GPU 環境用のふかうら王 TensorRT 版 compose ファイルです。 |
| `*.dockerignore` | Docker build に不要なファイルを build context から除外します。 |

## 事前準備

Docker build はリポジトリの root directory から実行してください。

```bash
cd /path/to/YaneuraOu
mkdir -p eval book
```

評価関数ファイルや定跡ファイルは image には含めません。実行時に volume mount します。

デフォルトの mount 先は以下です。

| ホスト側 | コンテナ側 | 用途 |
| --- | --- | --- |
| `./eval` | `/opt/yaneuraou/eval` | 評価関数、ONNX model |
| `./book` | `/opt/yaneuraou/book` | 定跡ファイル |

通常のやねうら王 NNUE 版では、例えば `./eval/nn.bin` を配置します。定跡ファイルを使う場合は `./book` に配置します。

```bash
cp /path/to/nn.bin eval/nn.bin
cp /path/to/standard_book.db book/standard_book.db
```

ふかうら王では、デフォルトで `EvalDir=eval`, `DNN_Model=model.onnx` なので、`./eval/model.onnx` を配置します。別名の model を使う場合は、USI の `setoption name DNN_Model value ...` で指定してください。

```bash
cp /path/to/model.onnx eval/model.onnx
```

## やねうら王

`compose.yaneuraou.yaml` には、Makefile に列挙されている通常系 edition の service を用意しています。service 名を選ぶだけで、ビルドする edition が決まります。

| service | `YANEURAOU_EDITION` |
| --- | --- |
| `yaneuraou` | `YANEURAOU_ENGINE_NNUE` |
| `yaneuraou-nnue` | `YANEURAOU_ENGINE_NNUE` |
| `yaneuraou-nnue-kp256` | `YANEURAOU_ENGINE_NNUE_KP256` |
| `yaneuraou-nnue-halfkpe9` | `YANEURAOU_ENGINE_NNUE_HALFKPE9` |
| `yaneuraou-nnue-halfkp-512x2-16-32` | `YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32` |
| `yaneuraou-nnue-halfkp-1024x2-8-32` | `YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_32` |
| `yaneuraou-nnue-halfkp-1024x2-8-64` | `YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_64` |
| `yaneuraou-nnue-halfkp-vm-256x2-32-32` | `YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32` |
| `yaneuraou-nnue-sfnn1536` | `YANEURAOU_ENGINE_SFNN1536` |
| `yaneuraou-kppt` | `YANEURAOU_ENGINE_KPPT` |
| `yaneuraou-kpp-kkpt` | `YANEURAOU_ENGINE_KPP_KKPT` |
| `yaneuraou-material` | `YANEURAOU_ENGINE_MATERIAL` |
| `tanuki-mate` | `TANUKI_MATE_ENGINE` |
| `yaneuraou-mate` | `YANEURAOU_MATE_ENGINE` |
| `user-engine` | `USER_ENGINE` |

ビルドだけ行う場合:

```bash
docker compose -f docker/compose.yaneuraou.yaml build yaneuraou
```

ビルドして、そのまま起動する場合:

```bash
docker compose -f docker/compose.yaneuraou.yaml run --rm --build yaneuraou
```

起動するとエンジンのコンソールに入ります。そこで標準入力から USI command を入力して動作確認します。

```text
usi
isready
bench
quit
```

SFNN1536 版をビルドして、そのまま起動する場合:

```bash
docker compose -f docker/compose.yaneuraou.yaml run --rm --build yaneuraou-sfnn1536
```

起動後は同様に `usi`, `isready`, `bench`, `quit` を入力して確認します。

SFNN1536, KPPT, KPP_KKPT など大きな評価関数ファイルを読み込むため、`compose.yaneuraou.yaml` では `/dev/shm` を `1g` に設定しています。`docker compose run` には `--shm-size` を指定できないので、変更したい場合は compose ファイルの `shm_size` を変更してください。

## ふかうら王 CPU 版

GPU がない環境ではこちらを使います。ONNX Runtime CPU 版を取得して `YANEURAOU_ENGINE_DEEP_ORT_CPU` でビルドします。

ビルドだけ行う場合:

```bash
docker compose -f docker/compose.fukauraou-cpu.yaml build
```

ビルドして、そのまま起動する場合:

```bash
docker compose -f docker/compose.fukauraou-cpu.yaml run --rm --build fukauraou-cpu
```

起動後、エンジンのコンソールで以下を入力して動作確認します。

```text
usi
isready
bench
quit
```

デフォルトでは ONNX Runtime `1.23.0` を使います。変更する場合:

```bash
docker compose -f docker/compose.fukauraou-cpu.yaml build --build-arg ONNXRUNTIME_VERSION=1.23.2 fukauraou-cpu
```

## ふかうら王 TensorRT 版

NVIDIA GPU、NVIDIA driver、NVIDIA Container Toolkit が必要です。GPU がない環境では起動できないか、`isready` 時に失敗します。

ビルドだけ行う場合:

```bash
docker compose -f docker/compose.fukauraou-tensorrt.yaml build
```

ビルドして、そのまま起動する場合:

```bash
docker compose -f docker/compose.fukauraou-tensorrt.yaml run --rm --build fukauraou-tensorrt
```

起動後、エンジンのコンソールで以下を入力して動作確認します。

```text
usi
isready
bench
quit
```

TensorRT 版は初回 model load 時に、ONNX model から TensorRT serialized engine を生成します。`./eval` に `.serialized` ファイルが作られるので、`./eval` は書き込み可能にしてください。

GPU がない環境や、WSL2 から NVIDIA GPU が見えていない環境では、起動時に以下のようなエラーになることがあります。

```text
nvidia-container-cli: initialization error: WSL environment detected but no adapters were found
```

この場合、TensorRT 版のコンテナは起動できません。GPU なし環境では CPU 版を使ってください。

```bash
docker compose -f docker/compose.fukauraou-cpu.yaml run --rm --build fukauraou-cpu
```

## docker build を直接使う場合

compose を使わずに直接 build することもできます。

```bash
docker build -f docker/Dockerfile.yaneuraou -t yaneuraou:local .
docker build -f docker/Dockerfile.fukauraou-cpu -t fukauraou-cpu:local .
docker build -f docker/Dockerfile.fukauraou-tensorrt -t fukauraou-tensorrt:local .
```

直接起動する場合:

```bash
docker run --rm -it -v "${PWD}/eval:/opt/yaneuraou/eval" -v "${PWD}/book:/opt/yaneuraou/book" yaneuraou:local

docker run --rm -it -v "${PWD}/eval:/opt/yaneuraou/eval" -v "${PWD}/book:/opt/yaneuraou/book" fukauraou-cpu:local

docker run --rm -it --gpus all -v "${PWD}/eval:/opt/yaneuraou/eval" -v "${PWD}/book:/opt/yaneuraou/book" fukauraou-tensorrt:local
```

起動後、エンジンのコンソールで以下を入力して動作確認します。

```text
usi
isready
bench
quit
```

Windows の `cmd.exe` で直接 `docker run` する場合は、上記の `${PWD}` を `%cd%` に置き換えてください。PowerShell、bash、Git Bash では `${PWD}` のまま使えます。

## VSCode Docker 拡張から使う場合

VSCode の Docker 拡張から GUI で実行しやすいように、compose ファイルを用途ごとに分けています。

- 通常のやねうら王: `docker/compose.yaneuraou.yaml`
- GPU なしのふかうら王: `docker/compose.fukauraou-cpu.yaml`
- NVIDIA GPU 用のふかうら王: `docker/compose.fukauraou-tensorrt.yaml`

GUI から起動する前に、必要な評価関数ファイルや ONNX model を `./eval` に配置してください。

## 注意

- Dockerfile は root directory に置かず、`docker/` 以下にまとめています。
- `*.dockerignore` により、`.git`, `obj`, `eval`, `book`, `inbox` などは Docker build context から除外されます。
- `TARGET_CPU=AVX2` でビルドした image は、AVX2 非対応 CPU では動きません。互換性重視なら `TARGET_CPU=OTHER` でビルドしてください。
- DirectML 版は Windows 向けです。Docker では CPU 版と TensorRT 版を用意しています。
