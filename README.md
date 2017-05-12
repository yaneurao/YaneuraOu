# About this project

YaneuraOu mini is a shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2017

やねうら王miniは、将棋の思考エンジンで、Bonanza6より強く、教育的で短いコード(2500行程度)で書かれたUSIプロトコル準拠の思考エンジンで、VC++2017でコンパイル可能です。

[やねうら王mini 公式サイト (解説記事、開発者向け情報等)](http://yaneuraou.yaneu.com/YaneuraOu_Mini/)

[やねうら王公式 ](http://yaneuraou.yaneu.com/)

## やねうら王シリーズの遊び方

[このプロジェクトのexeフォルダ](https://github.com/yaneurao/YaneuraOu/tree/master/exe)の対象フォルダ配下にある、XXX-readme.txtをご覧ください。

- 質問等は以下の記事のコメント欄でお願いします。Twitterでの個別質問にはお答え出来ません。
	- [やねうら王セットアップ質問スレッド](http://yaneuraou.yaneu.com/2017/05/04/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B-%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E8%B3%AA%E5%95%8F%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89/)

# 現在進行中のサブプロジェクト

## やねうら王2017 Early

2017年5月5日完成。この思考エンジンを用いたelmoがWCSC27で優勝しました。elmo(WCSC27)や蒼天幻想ナイツ・オブ・タヌキ(WCSC27出場)の評価関数を用いるとXeon 24コアでR4000程度の模様。

## やねうら王2017 GOKU

《開発計画中》

## やねうら王詰め将棋solver

《tanuki-さんが開発中》

長手数の詰将棋が解けるsolverです。


# 過去のサブプロジェクト

過去のサブプロジェクトである、やねうら王nano , mini , classic、王手将棋、取る一手将棋、協力詰めsolver、連続自己対戦フレームワークなどはこちらからどうぞ。

- [過去のサブプロジェクト](/docs/README2017.md)

## やねうら王評価関数ファイル

やねうら王2016Mid用/2016Late用/2017Early用

- [真やねうら王の評価関数ファイル](https://drive.google.com/open?id=0ByIGrGAuSfHHVVh0bEhxRHNpcGc) (Apery20161007の評価関数から追加学習させたものです。) 詳しい情報は[こちら。](http://yaneuraou.yaneu.com/2016/10/17/%E7%9C%9F%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/)
- [やねうら王で使える評価関数ファイル28バリエーション公開しました](http://yaneuraou.yaneu.com/2016/07/22/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%A7%E4%BD%BF%E3%81%88%E3%82%8B%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB28%E3%83%90%E3%83%AA%E3%82%A8%E3%83%BC%E3%82%B7/)
- また、Apery(WCSC26)、Apery(SDT4)＝「浮かむ瀬」の評価関数バイナリがそのまま使えます。

## 定跡集

やねうら王2016Mid以降で使える、各種定跡集。
ダウンロードしたあと、zipファイルになっているのでそれを解凍して、やねうら王の実行ファイルを配置しているフォルダ配下のbookフォルダに放り込んでください。

コンセプトおよび定跡フォーマットについて : [やねうら大定跡はじめました](http://yaneuraou.yaneu.com/2016/07/10/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E5%A4%A7%E5%AE%9A%E8%B7%A1%E3%81%AF%E3%81%98%E3%82%81%E3%81%BE%E3%81%97%E3%81%9F/)

-[やねうら大定跡V1.01](https://drive.google.com/open?id=0Bzbi5rbfN85NbWxfazMzamFVZm8)
-[真やねうら定跡](https://drive.google.com/open?id=0ByIGrGAuSfHHcXRrc2FmdHVmRzA)

## 世界コンピュータ将棋選手権および2017年に開催される第5回将棋電王トーナメントに参加される開発者の方へ

やねうら王をライブラリとして用いて参加される場合、このやねうら王のGitHub上にあるすべてのファイルおよび、このトップページから直リンしているファイルすべてが使えます。
ただし、真やねうら王の評価関数ファイルを用いる場合は、Aperyライブラリの申請が必要かも知れません。詳しくは大会のルールを参照してください。

## ライセンス

やねうら王プロジェクトのソースコードはStockfishをそのまま用いている部分が多々あり、Apery/SilentMajorityを参考にしている部分もありますので、やねうら王プロジェクトは、それらのプロジェクトのライセンス(GPLv3)に従うものとします。

また、「真やねうら王の評価関数ファイル」は、Aperyの評価関数バイナリから追加学習させたものですので、その著作権は、Aperyの開発者の平岡拓也氏に帰属し、ライセンスや取扱いは元のライセンスに従うものとします。また、やねうら王プロジェクト側はこのファイルの著作権を主張しません。
