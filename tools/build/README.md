# ビルド用スクリプト

Linux・Windows・Android・WebAssembly向けのビルド補助ツールです。コマンド例はリポジトリのルート（`source/`と`tools/`がある場所）から実行します。

## 目次

- [Linux・Windows](#ビルド)
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
