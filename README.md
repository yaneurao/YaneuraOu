[![Make CI (MSYS2 for Windows)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-msys2.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-msys2.yml)
[![Make CI (DeepLearning for Windows)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-deep-windows.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-deep-windows.yml)
[![Make CI (MinGW for Windows)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-mingw.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-mingw.yml)
[![Make CI (for Ubuntu Linux)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make.yml)
[![Make CI (DeepLearning for Ubuntu Linux)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-deep-ubuntu.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-deep-ubuntu.yml)
[![NDK CI (for Android)](https://github.com/yaneurao/YaneuraOu/actions/workflows/ndk.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/ndk.yml)
[![Make CI (for MacOS)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-macos.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-macos.yml)
[![Make CI (for WebAssembly)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-wasm.yml/badge.svg?event=push)](https://github.com/yaneurao/YaneuraOu/actions/workflows/make-wasm.yml)

# About this project

YaneuraOu is the World's Strongest Shogi engine(AI player) , WCSC29 1st winner , educational and USI compliant engine.

やねうら王は、WCSC29(世界コンピュータ将棋選手権/2019年)において優勝した世界最強の将棋の思考エンジンです。教育的でUSIプロトコルに準拠しています。

- [WCSC29、やねうら王優勝しました！](http://yaneuraou.yaneu.com/2019/05/06/wcsc29%E3%80%81%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E5%84%AA%E5%8B%9D%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F%EF%BC%81/)

# やねうら王エンジンの大会での戦績

- 2021年 第2回世界将棋AI電竜戦TSEC 『水匠』総合優勝。(探索部やねうら王)
- 2020年 第1回 世界コンピュータ将棋オンライン大会(WCSO1) 『水匠』優勝。(探索部やねうら王)
- 2019年 世界コンピュータ将棋選手権(WCSC29) 『やねうら王 with お多福ラボ2019』優勝。
  - 決勝の上位8チームすべてがやねうら王の思考エンジンを採用。
- 2018年 世界コンピュータ将棋選手権(WCSC28) 『Hefeweizen』優勝
- 2017年 世界コンピュータ将棋選手権(WCSC27) 『elmo』優勝
- 2017年 第5回将棋電王トーナメント(SDT5) 『平成将棋合戦ぽんぽこ』優勝

# やねうら王の特徴

- USIプロトコルに準拠した思考エンジンです。
- 入玉宣言勝ち、トライルール等にも対応しています。
- Ponder(相手番で思考する)に対応しています。
- 秒読み、フィッシャールールなど様々な持時間に対応しています。
- 256スレッドのような超並列探索に対応しています。
- 定跡にやねうら王標準定跡フォーマットを採用しています。
- 置換表の上限サイズは33TB(実質的に無限)まで対応しています。

# やねうら王の解説記事

|記事内容|リンク|レベル|
|-|-|-|
|やねうら王のインストール手順について | [やねうら王のインストール手順](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王のインストール手順)|入門|
|ふかうら王のインストール手順について | [ふかうら王のインストール手順](https://github.com/yaneurao/YaneuraOu/wiki/ふかうら王のインストール手順)|中級|
|やねうら王のお勧めエンジン設定について | [やねうら王のお勧めエンジン設定](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王のお勧めエンジン設定)|入門|
|ふかうら王のお勧めエンジン設定について | [ふかうら王のお勧めエンジン設定](https://github.com/yaneurao/YaneuraOu/wiki/ふかうら王のお勧めエンジン設定)|入門|
|やねうら王のエンジンオプションについて | [思考エンジンオプション](https://github.com/yaneurao/YaneuraOu/wiki/思考エンジンオプション)|入門~中級|
|やねうら王詰将棋エンジンについて| [やねうら王詰将棋エンジン](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王詰将棋エンジン)|入門~中級|
|やねうら王のよくある質問|[よくある質問](https://github.com/yaneurao/YaneuraOu/wiki/よくある質問)|初級~中級|
|やねうら王の隠し機能 | [隠し機能](https://github.com/yaneurao/YaneuraOu/wiki/隠し機能)|中級~上級|
|やねうら王の定跡を作る | [定跡の作成](https://github.com/yaneurao/YaneuraOu/wiki/定跡の作成)|中級~上級|
|やねうら王のUSI拡張コマンドについて | [USI拡張コマンド](https://github.com/yaneurao/YaneuraOu/wiki/USI拡張コマンド)|開発者向け|
|やねうら王のビルド手順について | [やねうら王のビルド手順](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王のビルド手順)|開発者向け|
|ふかうら王のビルド手順について | [ふかうら王のビルド手順](https://github.com/yaneurao/YaneuraOu/wiki/ふかうら王のビルド手順)|開発者向け|
|やねうら王のソースコード解説 |[やねうら王のソースコード解説](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王のソースコード解説)|開発者向け|
|AWSでやねうら王を動かす| [AWSでやねうら王](https://github.com/yaneurao/YaneuraOu/wiki/AWSでやねうら王)|中級~開発者|
|大会に参加する時の設定|[大会に参加する時の設定](https://github.com/yaneurao/YaneuraOu/wiki/大会に参加する時の設定)|開発者|
|やねうら王の学習コマンド|[やねうら王の学習コマンド](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王の学習コマンド)|開発者|
|ふかうら王の学習手順|[ふかうら王の学習手順](https://github.com/yaneurao/YaneuraOu/wiki/ふかうら王の学習手順)|開発者|
|USI対応エンジンの自己対局|[USI対応エンジンの自己対局](https://github.com/yaneurao/YaneuraOu/wiki/USI対応エンジンの自己対局)|中級～開発者|
|パラメーター自動調整フレームワーク|[パラメーター自動調整フレームワーク](https://github.com/yaneurao/YaneuraOu/wiki/パラメーター自動調整フレームワーク)|開発者|
|探索部の計測資料|[探索部の計測資料](探索部の計測資料)|開発者|
|廃止したコマンド・オプションなど| [過去の資料](https://github.com/yaneurao/YaneuraOu/wiki/過去の資料)|開発者|
|やねうら王の更新履歴|[やねうら王の更新履歴](https://github.com/yaneurao/YaneuraOu/wiki/やねうら王の更新履歴)|開発者|


# 現在進行中のサブプロジェクト

|プロジェクト名|進捗|
|-|-|
|やねうら王通常探索エンジン|2022年も引き続き、このエンジン部を改良していきます。|
| YaneuraOu The Cluster | やねうら王のクラスター化エンジン。2022年夏にはお披露目できそう。|
| やねうら王詰将棋エンジンV2 | 省メモリで長手数の詰将棋が解ける詰将棋用のエンジン。|
| floodgateV2 | floodgateに取って代わる対局場です |

# 過去のサブプロジェクト

過去のサブプロジェクトである、やねうら王nano , mini , classic、王手将棋、取る一手将棋、協力詰めsolver、連続自己対戦フレームワークなどはこちらからどうぞ。

- [過去のサブプロジェクト](https://github.com/yaneurao/YaneuraOu/wiki/%E9%81%8E%E5%8E%BB%E3%81%AE%E3%82%B5%E3%83%96%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88)

# やねうら王ニュース記事一覧

やねうら王公式ブログの関連記事の見出し一覧です。

各エンジンオプションの解説、定跡ファイルのダウンロード、定跡の生成手法などについての詳しい資料があります。初心者から開発者まで、知りたいことが全部詰まっています。

# ライセンス

やねうら王プロジェクトのソースコードはStockfishをそのまま用いている部分が多々あり、Apery/SilentMajorityを参考にしている部分もありますので、やねうら王プロジェクトは、それらのプロジェクトのライセンス(GPLv3)に従うものとします。

「リゼロ評価関数ファイル」については、やねうら王プロジェクトのオリジナルですが、一切の権利は主張しませんのでご自由にお使いください。

# やねうら王プロジェクト関連リンク

やねうら王関連の最新情報がキャッチできる主要なサイトです。

|サイト | リンク|
|-----|-----|
|やねうら王公式ブログ | https://yaneuraou.yaneu.com/|
|やねうら王mini 公式 (解説記事等)| http://yaneuraou.yaneu.com/YaneuraOu_Mini/|
|やねうら王Twitter | https://twitter.com/yaneuraou|
|やねうら王公式ちゃんねる(YouTube) | https://www.youtube.com/c/yanechan|

上記のやねうら王公式ブログでは、コンピュータ将棋に関する情報を大量に発信していますので、やねうら王に興味がなくとも、コンピュータ将棋の開発をしたいなら、非常に参考になると思います。

# 質問箱

やねうら王関連の質問は、以下のブログ記事のコメント欄にお願いします。
https://yaneuraou.yaneu.com/2022/05/19/yaneuraou-question-box/
