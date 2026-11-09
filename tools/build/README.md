# ビルド用スクリプト

Linux・Windows・Android・WebAssembly向けのビルド補助ツールです。コマンド例はリポジトリのルート（`source/`と`tools/`がある場所）から実行します。

## 目次

- [Linux・Windows](#ビルド)
- [PGO](#pgo)
- [Android](#android)
- [WebAssembly](#webassembly)

## ビルド

各スクリプトに列挙されたエディション・コンパイラ・CPU・makeターゲットの組み合わせをビルドします。引数を省略すると多数の組み合わせが対象になるので、単一構成を作る場合は条件を指定してください。

| ファイル | 実行環境・役割 | 出力先（リポジトリルート基準） |
| --- | --- | --- |
| [build.sh](build.sh) | LinuxのBash向け。`make`、Clang/GCCなどを使う一括ビルド。GitHub Actionsでも使用。 | `build/<OS>/<edition別ディレクトリ>/`。OSの既定値は`linux`。 |
| [mingw_gcc.sh](mingw_gcc.sh) | Linux上からMinGW-w64でWindows向けにクロスビルド。既定コンパイラは`x86_64-w64-mingw32-g++-posix`と`i686-w64-mingw32-g++-posix`。 | `build/mingw/<edition別ディレクトリ>/` |
| [msys2_build.sh](msys2_build.sh) | WindowsのMSYS2 / MinGW向け。`mingw32-make`、Clang/GCCを使用。 | `build/windows/<edition別ディレクトリ>/` |
| [msys2_build32.sh](msys2_build32.sh) | MSYS2 / MinGWの32bit版。`TARGET_CPU=NO_SSE`でビルド。 | `build/windows/<edition別ディレクトリ>/` |
| [msys2_build.ps1](msys2_build.ps1) | PowerShellから`msys2_shell.cmd`を呼ぶWindowsビルド。構成ごとに一時ディレクトリを使用し、PowerShell 7以上では複数構成を並列処理。 | `build/windows/<edition別ディレクトリ>/` |

### Bash版の指定方法

| オプション | 意味 | 対応スクリプト |
| --- | --- | --- |
| `-e` | エディションの選択 | 上記Bash版すべて |
| `-c` | コンパイラの選択 | 上記Bash版すべて |
| `-t` | makeターゲット（`normal`、`tournament`など）の選択 | 上記Bash版すべて |
| `-a` | CPUターゲットの選択 | `build.sh`、`mingw_gcc.sh`、`msys2_build.sh` |
| `-x` | makeへ渡す追加の変数指定 | 上記Bash版すべて |
| `-o` | 出力ディレクトリ名に使うOS名 | `build.sh` |

選択肢は各ファイル内の配列で定義されています。複数指定はカンマ区切りで、選択パターンにはワイルドカードも使えます。シェルによる展開を避けるため、`'*NNUE*'`のように引用してください。任意のアーキテクチャ名をそのまま受け付ける仕組みではなく、配列にない構成は`source/Makefile`を直接使ってビルドします。

```sh
bash tools/build/build.sh \
  -e YANEURAOU_ENGINE_NNUE -c clang++ -t normal -a AVX2

bash tools/build/mingw_gcc.sh \
  -e YANEURAOU_ENGINE_NNUE -c x86_64-w64-mingw32-g++-posix -t normal -a AVX2
```

これらのBash版は`source/`で`make clean`を実行します。同じソースツリーで別のビルドを同時に走らせる用途には向きません。

`msys2_build.sh`の出力名にはCPU名が含まれないため、CPUを複数選ぶと同じ名前の実行ファイルとログが上書きされます。CPUごとの成果物を残す場合は、CPU名が出力名に含まれるPowerShell版を使うか、構成ごとに退避してください。

### PowerShell版の指定方法

`msys2_build.ps1`は`-Edition`、`-Compiler`、`-Target`、`-Cpu`で対象を絞り、`-Extra`でmakeの追加変数を渡します。MSYS2と必要なツールチェーンが導入され、`msys2_shell.cmd`にPATHが通っている必要があります。導入例はスクリプト冒頭にもあります。

```powershell
.\tools\build\msys2_build.ps1 -Edition YANEURAOU_ENGINE_NNUE -Compiler clang++ -Target normal -Cpu AVX2
```

## PGO

`source/Makefile`の`pgo`ターゲットで、計測用ビルド、benchによる訓練、プロファイルの集約、本番ビルドを順番に実行します。**すべて`tournament`でビルド**します。Clang/GCC、Python 3.8以上、および生成したエンジンを実行できる環境が必要です。Clangでは対応する`llvm-profdata`も必要です。

次の例は`source/`から実行します。`PGO_EVAL_DIR`には使用する評価関数のフォルダを指定してください。エディションとCPU、追加フラグは通常のmakeと同様です。

```sh
make -j16 pgo \
  COMPILER=clang++ TARGET_CPU=AVX512VNNI \
  YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfkahm2_2304_15_64_k9k9_progress8 \
  PGO_EVAL_DIR=../../inbox/SFNN_halfkahm2_2304_15_64_k9k9_progress8-20261109 \
  EXTRA_CPPFLAGS="-DUSE_LAZY_EVALUATE -DUSE_FINNY_TABLES -DUSE_FINNY_RESET -DNNUE_FT_SPARSE8"
```

最後に通常の`TARGET`へPGO済みの実行ファイルを出力します。訓練やビルドに失敗した場合、既存の実行ファイルは置き換えません。定跡は訓練時に無効にします。モデルの`eval_options.txt`は通常どおり読み込まれます。

| 変数 | 既定値・役割 |
| --- | --- |
| `PGO_EVAL_DIR` | `../eval`。訓練用エンジンの`EvalDir`。`source/`からの相対パスまたは絶対パス。 |
| `PGO_BENCH` | `1024 1 14 default depth`。`bench`の後ろに渡す引数。既定はHash 1024MiB、1スレッド、既定4局面を深さ14まで探索。 |
| `PGO_DIR` | `$(OBJDIR)/pgo`。各実行のオブジェクト・プロファイル・訓練ログの保存先。`source/`からの相対パスに空白がない場所を指定。 |
| `PGO_PROFDATA` | Clang用の`llvm-profdata`。未指定ならClangと同じディレクトリ、バージョン付きの名前、PATHから検索。例：`llvm-profdata-14`。 |
| `PGO_TIMEOUT` | `600`。エンジン起動から訓練終了までの上限秒数。ビルド時間は含まない。 |

例えば独自のSFEN局面集を使う場合は、`PGO_BENCH="1024 1 150000 /path/to/training.sfen nodes"`を追加します。これは各局面を15万ノードまで探索する指定です。局面集のパスはbenchの仕様により空白を含まないものを使ってください。

GCCでは`COMPILER=g++`を指定します。Clangのprofrawは`llvm-profdata`で集約し、GCCのgcdaは同じオブジェクト・実行ファイルパスで利用します。通常ビルドのオブジェクトを流用せず、ソースの`touch`も行いません。ビルドの補助処理は[pgo.py](pgo.py)にあります。

訓練を手動で行う場合は、同じビルド設定で`make profgen`、計測用エンジンの実行、`make profuse`の順でも使えます。`profgen`は計測用エンジンを`TARGET`へ出力します。`pgo`と`profgen`は毎回新しい作業ディレクトリを作るので、古いプロファイルは混ざりません。`profuse`は直前の`profgen`のデータを使い、コンパイラ・エディション・CPU・追加フラグが異なる場合はエラーにします。

## Android

| ファイル | 内容 |
| --- | --- |
| [android_build.sh](android_build.sh) | BashからAndroid NDKの`ndk-build`を実行。`-e`でエディション、`-x`で追加のmake変数を指定。 |
| [android_build.ps1](android_build.ps1) | PowerShellから`ndk-build.cmd`を実行。`-Edition`でエディション、`-Extra`で追加変数を指定。 |
| [jni/Android.mk](jni/Android.mk) | NDK用のソース一覧、エディション別の定義、ABI別のコンパイル設定。 |
| [jni/Application.mk](jni/Application.mk) | NDKの共通設定。`APP_ABI := all`、`APP_STL := c++_static`、`APP_PLATFORM = android-19`。 |

NDKの実行ファイルにPATHを通してから実行します。成果物とビルドログは`build/android/<edition別ディレクトリ>/`に保存します。

```sh
bash tools/build/android_build.sh -e YANEURAOU_ENGINE_NNUE
```

対象エディションやABIの設定は各ファイルに列挙されています。使用するNDKがその設定・ABIに対応しているか確認してください。

## WebAssembly

| ファイル | 内容 |
| --- | --- |
| [wasm_build.js](wasm_build.js) | Node.jsで動く生成本体。Emscriptenの`em++`とmakeを使い、エンジン、USI起動用JS/TS、型定義、gzip/Brotli圧縮ファイルを生成。 |
| [wasm_docker_build.js](wasm_docker_build.js) | Dockerイメージ`emscripten/emsdk:3.1.43`をpullし、コンテナ内で生成本体を実行。引数のパッケージ名を引き継ぎます。 |
| [wasm_build.sh](wasm_build.sh) | Bash用のDocker起動ラッパー。パッケージ指定を渡さず、定義された全構成を生成。 |
| [wasm_build.cmd](wasm_build.cmd) | Windowsのコマンドプロンプト用Dockerラッパー。パッケージ指定を渡さず、定義された全構成を生成。 |

生成本体とNode.js版Dockerラッパーは、ルートで実行してください。成果物は`build/wasm/<パッケージ名>/`、エンジン本体はその下の`lib/`に置かれます。

```sh
# ローカルにNode.jsとEmscriptenのビルド環境がある場合
node tools/build/wasm_build.js halfkp.noeval

# Node.jsとDockerを使う場合
node tools/build/wasm_docker_build.js halfkp.noeval
```

パッケージ名は`halfkp`、`k-p`、`halfkp.noeval`、`halfkpe9.noeval`、`halfkpvm.noeval`、`k-p.noeval`、`material`、`material9`、`yaneuraou-mate`、`tanuki-mate`です。生成本体では引数なしで全構成、引数ありで指定名に一致した構成を生成します。

`halfkp`と`k-p`は評価関数のC++データをダウンロードし、`source/eval/nnue/embedded_nnue.cpp`へ展開します。ダウンロード先は`.dl/`です。`noeval`は評価関数を埋め込まない構成です。

## 関連資料

- [速度比較・自己対局](../script/README.md)
- [Dockerでのエンジンビルド](../docker/README.md)
- [調査・実験メモ](../memo/README.md)
