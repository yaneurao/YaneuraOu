# About this project

YaneuraOu mini is a shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2017

やねうら王miniは、将棋の思考エンジンで、Bonanza6より強く、教育的で短いコード(2500行程度)で書かれたUSIプロトコル準拠の思考エンジンで、VC++2017でコンパイル可能です。

[やねうら王mini 公式サイト (解説記事、開発者向け情報等)](http://yaneuraou.yaneu.com/YaneuraOu_Mini/)

[やねうら王公式 ](http://yaneuraou.yaneu.com/)

## やねうら王シリーズの遊び方

[このプロジェクトのexeフォルダ](https://github.com/yaneurao/YaneuraOu/tree/master/exe)の対象フォルダ配下にある、XXX-readme.txtをご覧ください。

- 質問等は以下の記事のコメント欄でお願いします。Twitterでの個別質問にはお答え出来ません。
	- [やねうら王セットアップ質問スレッド](http://yaneuraou.yaneu.com/2017/05/04/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B-%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E8%B3%AA%E5%95%8F%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89/)

# Sub-projects

## やねうら王nano

やねうら王nanoは1500行程度で書かれた将棋AIの基本となるプログラムです。探索部は150行程度で、非常にシンプルなコードで、αβ以外の枝刈りを一切していません。(R2000程度)

## やねうら王nano plus

やねうら王nano plusは、探索部300行程度で、オーダリングなどを改善した非常にシンプルでかつそこそこ強い思考エンジンです。(R2500程度)

## やねうら王mini

やねうら王miniは、やねうら王nano plusを並列化して、将棋ソフトとしての体裁を整えたものです。Bonanza6より強く、教育的かつ短いコードで書かれています。全体で3000行程度、探索部500行程度。(R2700程度)

## やねうら王classic

やねうら王classicは、やねうら王miniのソースコードを改良する形で、Apery(WCSC 2015)ぐらいの強さを目指しました。入玉宣言機能も追加しました。(R3000程度)

## やねうら王classic-tce

やねうら王classic-tceは、やねうら王classicのソースコードに持ち時間制御(秒読み、フィッシャールールに対応)、ponderの機能を追加したものです。(R3250程度)

## やねうら王2016 Mid

やねうら王 思考エンジン 2016年Mid版。Hyperopt等を用いて各種ハイパーパラメーターの調整の自動化を行ない自動調整します。長い持ち時間に対して強化しました。Apery(WCSC26)の評価関数バイナリを読み込めるようにしました。(R3450程度)

## やねうら王2016 Late

第4回将棋電王トーナメント出場バージョン。「真やねうら王」

NDFの学習メソッドを用い、Hyperopt等を用いて各種パラメーターの調整を行い、技巧(2015)を超えた強さになりました。(R3650程度)
→　第4回将棋電王トーナメントは無事終えました。真やねうら王は3位に入賞しました。応援してくださった皆様、本当にありがとうございました。

## やねうら王2017 Early

2017年5月3日完成。蒼天幻想ナイツ・オブ・タヌキ(WCSC27出場)の評価関数を用いるとXeon 24コアでR4000程度の模様。

## やねうら王2017 Late

《開発計画中》

## やねうら王 王手将棋エディション

王手すると勝ちという変則ルールの将棋。世界最強の王手将棋になりました。(R4250)

## やねうら王 取る一手将棋エディション

合法な取る手がある場合は、必ず取らないといけないという変則ルールの将棋。人類では勝てない最強の取る一手将棋を目指します。

## 連続自動対局フレームワーク

連続自動対局を自動化できます。 python版のスクリプトも用意。今後は、python版のほうに注力します。

## やねうら王協力詰めsolver

『寿限無3』(49909手)も解ける協力詰めsolver →　[解説ページ](http://yaneuraou.yaneu.com/2016/01/02/%E5%8D%94%E5%8A%9B%E8%A9%B0%E3%82%81solver%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%99/)

## やねうら王詰め将棋solver (気が向いたら製作します)

長手数の詰将棋が解けるsolverです。

## やねうら王評価関数バイナリ

やねうら王 王手将棋エディション用

- [王手将棋用評価関数ファイルV1](https://drive.google.com/file/d/0Bzbi5rbfN85NOEF6QWFienZrSDg/) [解説記事](http://yaneuraou.yaneu.com/2016/11/21/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E5%B0%82%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv1%E3%81%8C%E5%87%BA%E6%9D%A5%E3%81%BE%E3%81%97%E3%81%9F/)
- [王手将棋用評価関数ファイルV2](https://drive.google.com/open?id=0Bzbi5rbfN85Nci02T3hkWm1yQlE) [解説記事](http://yaneuraou.yaneu.com/2016/11/22/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv2%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)
- [王手将棋用評価関数ファイルV3](https://drive.google.com/open?id=0Bzbi5rbfN85NVGJ3eHNtaHZhLXc) [解説記事](http://yaneuraou.yaneu.com/2016/11/23/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv3%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)
- [王手将棋用評価関数ファイルV4](https://drive.google.com/open?id=0Bzbi5rbfN85NcTIzaFVKU0ZfNU0) [解説記事](http://yaneuraou.yaneu.com/2016/11/23/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv4%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)
- [王手将棋用評価関数ファイルV5](https://drive.google.com/open?id=0Bzbi5rbfN85Na3ZOeE5zNUZpNkE) [解説記事](http://yaneuraou.yaneu.com/2016/11/24/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv5%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)
- [王手将棋用評価関数ファイルV6](https://drive.google.com/open?id=0Bzbi5rbfN85NeWxUWUFfMFdZSjQ) [解説記事](http://yaneuraou.yaneu.com/2016/11/29/%E7%8E%8B%E6%89%8B%E5%B0%86%E6%A3%8B%E7%94%A8%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%ABv6%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)

やねうら王2016Mid用/2016Late用/2017Early用

- [真やねうら王の評価関数ファイル](https://drive.google.com/open?id=0ByIGrGAuSfHHVVh0bEhxRHNpcGc) (Apery20161007の評価関数から追加学習させたものです。) 詳しい情報は[こちら。](http://yaneuraou.yaneu.com/2016/10/17/%E7%9C%9F%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/)
- [やねうら王で使える評価関数ファイル28バリエーション公開しました](http://yaneuraou.yaneu.com/2016/07/22/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%A7%E4%BD%BF%E3%81%88%E3%82%8B%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB28%E3%83%90%E3%83%AA%E3%82%A8%E3%83%BC%E3%82%B7/)
- また、Apery(WCSC26)、Apery(SDT4)＝「浮かむ瀬」の評価関数バイナリがそのまま使えます。

やねうら王nano,nano-plus,classic,classic-tce用
- CSAのライブラリの[ダウンロードページ](http://www.computer-shogi.org/library/)からダウンロードできます。

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
