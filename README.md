shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2015

■　やねうら王nano

やねうら王nanoは1000行程度で書かれた将棋AIのお手本的となるプログラムです。(予定)

■　やねうら王mini (作業中2016年1月下旬完成予定)

やねうら王miniは、将棋の思考エンジンです。Bonanza6より強く、教育的かつ短いコードで書かれています。(2500行程度) USI準拠の思考エンジンで、VC++2015でコンパイル可能です。

やねうら王mini 公式サイト (解説記事等) : http://yaneuraou.yaneu.com/YaneuraOu_Mini/

やねうら王公式 : http://yaneuraou.yaneu.com/

■　やねうら王classic

やねうら王classicは、ソースコード4000行程度でAperyと同等の棋力を実現するプロジェクトです。(予定)

■  やねうら王2016

やねうら王 思考エンジン 2016年版(非公開)

■　連続自動対局フレームワーク

連続自動対局を自動化できます。

■  やねうら王協力詰めsolver

『寿限無3』(49909手)も解ける協力詰めsolver
解説ページ : http://yaneuraou.yaneu.com/2016/01/02/%E5%8D%94%E5%8A%9B%E8%A9%B0%E3%82%81solver%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%99/



■　俺の作業メモ(2016/01/22 8:40現在)

【作業中】・Position::do_move()のときのcaptureとnon_captureのときの利きの差分更新処理

【完成】・Position::do_move()のdropのときの利きの差分更新処理

【完成】・Long Effect Library

【完成】・利きの初期化処理

【完成】・自動対局サーバーの開始局面のランダマイズ機能

【完成】・自動対局サーバー機能、書けた。

【完成】・32bit環境用のコード、ちゃんと動くようになった。(手元に32bit環境がないので実際の環境で試してはいない。)

・1手詰め判定ルーチン、もう少し作業が残っている。

・評価関数の差分計算まだ。探索部書いてから書く。

・探索部手付かず。あとで書く。

・GitHubにアップロードしようと思ってフォルダ整理したときに、新しい形式に変換した評価関数バイナリの入ったフォルダを誤って消してしまった。お、、おい…。また変換部を書いてもってくる。泣きそう。