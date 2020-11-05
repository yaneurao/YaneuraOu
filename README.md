![Make CI (MSYS2 for Windows)](https://github.com/yaneurao/YaneuraOu/workflows/Make%20CI%20(MSYS2%20for%20Windows)/badge.svg?event=push)
![Make CI (MinGW for Windows)](https://github.com/yaneurao/YaneuraOu/workflows/Make%20CI%20(MinGW%20for%20Windows)/badge.svg?event=push)
![Make CI (for Ubuntu Linux)](https://github.com/yaneurao/YaneuraOu/workflows/Make%20CI%20(for%20Ubuntu%20Linux)/badge.svg?event=push)
![NDK CI (for Android)](https://github.com/yaneurao/YaneuraOu/workflows/NDK%20CI%20(for%20Android)/badge.svg?event=push)

# About this project

YaneuraOu is the World's Strongest Shogi engine(AI player) , WCSC29 1st winner , educational and USI compliant engine.

やねうら王は、将棋の思考エンジンとして世界最強で、WCSC29(世界コンピュータ将棋選手権/2019年)において優勝しました。教育的でUSIプロトコル準拠の思考エンジンです。


# お知らせ

2019/05/05 WCSC29(世界コンピュータ将棋選手権/2019年)に『やねうら王 with お多福ラボ2019』として出場し、見事優勝を果たしました。(WCSC初参加、優勝)　このあと一年間は世界最強を名乗ります。(｀･ω･´)ｂ
- [WCSC29、やねうら王優勝しました！](http://yaneuraou.yaneu.com/2019/05/06/wcsc29%E3%80%81%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E5%84%AA%E5%8B%9D%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F%EF%BC%81/)


# やねうら王エンジンの大会での戦績

- 2017年 世界コンピュータ将棋選手権(WCSC27) 『elmo』優勝
- 2017年 第5回将棋電王トーナメント(SDT5) 『平成将棋合戦ぽんぽこ』優勝
- 2018年 世界コンピュータ将棋選手権(WCSC28) 『Hefeweizen』優勝
- 2019年 世界コンピュータ将棋選手権(WCSC29) 『やねうら王 with お多福ラボ2019』優勝。
  - 決勝の上位8チームすべてがやねうら王の思考エンジンを採用。


# やねうら王の使い方について

  - セットアップ手順 : [やねうら王 セットアップ質問スレッド](http://yaneuraou.yaneu.com/2017/05/04/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B-%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E8%B3%AA%E5%95%8F%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89/)
  - エンジンオプションの解説 : [docs/USI拡張コマンド.txt](docs/USI%E6%8B%A1%E5%BC%B5%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89.txt)
  - 評価関数の学習コマンド、定跡の生成コマンド、ビルド方法等の技術的文書 : [docs/解説.txt](/docs/%E8%A7%A3%E8%AA%AC.txt)


# 現在進行中のサブプロジェクト

## やねうら王エンジン (やねうら王 with お多福ラボ 2019)

2019年も引き続き、このエンジン部を改良していきます。

## やねうら王詰め将棋solver

《tanuki-さんが開発中》 長手数の詰将棋が解けるsolverです。

# 過去のサブプロジェクト

過去のサブプロジェクトである、やねうら王nano , mini , classic、王手将棋、取る一手将棋、協力詰めsolver、連続自己対戦フレームワークなどはこちらからどうぞ。

- [過去のサブプロジェクト](/docs/README2017.md)

# やねうら王評価関数ファイル

- やねうら王2018 KPPT型 - Apery(WCSC26)、Apery(SDT4)＝「浮かむ瀬」の評価関数バイナリがそのまま使えます。
- やねうら王2018 KPP_KKPT型 - [過去のサブプロジェクト](/docs/README2017.md)のKPP_KKPT型ビルド用評価関数のところにあるものが使えます。
- やねうら王2018 NNUE型 - tanuki-(SDT5,WCSC28,WCSC29),NNUEkaiなどの評価関数が使えます。

# やねうら王 ニュース

やねうら王公式ブログに書いた関連記事の見出し一覧です。各エンジンオプションの解説、定跡ファイルのダウンロード、定跡の生成手法などもこちらにあります。

  - [やねうら王ニュース目次一覧](docs/news.md)

# WCSC(世界コンピュータ将棋選手権)に参加される開発者の方へ

やねうら王をライブラリとして用いて参加される場合、このやねうら王のGitHub上にあるすべてのファイルおよび、このトップページから直リンしているファイルすべてが使えます。

# ライセンス

やねうら王プロジェクトのソースコードはStockfishをそのまま用いている部分が多々あり、Apery/SilentMajorityを参考にしている部分もありますので、やねうら王プロジェクトは、それらのプロジェクトのライセンス(GPLv3)に従うものとします。

「リゼロ評価関数ファイル」については、やねうら王プロジェクトのオリジナルですが、一切の権利は主張しませんのでご自由にお使いください。
