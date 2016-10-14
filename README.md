# About this project

YaneuraOu mini is a shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2015 Update3

やねうら王miniは、将棋の思考エンジンで、Bonanza6より強く、教育的で短いコード(2500行程度)で書かれたUSIプロトコル準拠の思考エンジンで、VC++2015 Update3でコンパイル可能です。

[やねうら王mini 公式サイト (解説記事、開発者向け情報等)](http://yaneuraou.yaneu.com/YaneuraOu_Mini/)

[やねうら王公式 ](http://yaneuraou.yaneu.com/)

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

## やねうら王2017 Early【非公開の予定】

2017年開発するかも。

## 連続自動対局フレームワーク

連続自動対局を自動化できます。 

## やねうら王協力詰めsolver
	
『寿限無3』(49909手)も解ける協力詰めsolver →　[解説ページ](http://yaneuraou.yaneu.com/2016/01/02/%E5%8D%94%E5%8A%9B%E8%A9%B0%E3%82%81solver%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%99/)

## やねうら王詰め将棋solver (気が向いたら製作します)

長手数の詰将棋が解けるsolverです。

## やねうら王シリーズの遊び方

[このプロジェクトのexeフォルダ](https://github.com/yaneurao/YaneuraOu/tree/master/exe)にある、XXX-readme.txtをご覧ください。

## やねうら王評価関数バイナリ

やねうら王2016Mid用/2016Late用 

- [真やねうら王の評価関数ファイル](https://drive.google.com/open?id=0ByIGrGAuSfHHVVh0bEhxRHNpcGc)
- [やねうら王で使える評価関数ファイル28バリエーション公開しました](http://yaneuraou.yaneu.com/2016/07/22/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%A7%E4%BD%BF%E3%81%88%E3%82%8B%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB28%E3%83%90%E3%83%AA%E3%82%A8%E3%83%BC%E3%82%B7/)
- また、Apery(WCSC26)、Apery(SDT4)＝「浮かむ瀬」の評価関数バイナリがそのまま使えます。

やねうら王nano,nano-plus,classic,classic-tce用
- CSAのライブラリの[ダウンロードページ](http://www.computer-shogi.org/library/)からダウンロードできます。

## 定跡集

やねうら王2016Mid以降で使える、各種定跡集。
ダウンロードしたあと、zipファイルになっているのでそれを解凍して、やねうら王2016Midの実行ファイルを配置しているフォルダ配下のbookフォルダに放り込んでください。

コンセプトおよび定跡フォーマットについて : [やねうら大定跡はじめました](http://yaneuraou.yaneu.com/2016/07/10/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E5%A4%A7%E5%AE%9A%E8%B7%A1%E3%81%AF%E3%81%98%E3%82%81%E3%81%BE%E3%81%97%E3%81%9F/)

-[やねうら大定跡V1.01](https://drive.google.com/open?id=0Bzbi5rbfN85NbWxfazMzamFVZm8)  
-[真やねうら定跡](https://drive.google.com/open?id=0ByIGrGAuSfHHcXRrc2FmdHVmRzA)

## 世界コンピュータ将棋選手権および2017年に開催される第5回将棋電王トーナメントに参加される開発者の方へ

やねうら王をライブラリとして用いて参加される場合、このやねうら王のGitHub上にあるすべてのファイルおよび、このトップページから直リンしているファイルすべてが使えます。


