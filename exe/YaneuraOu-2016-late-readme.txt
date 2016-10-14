
・やねうら王2016 Lateとは？

やねうら王2016 Lateとは、第4回将棋電王トーナメント(2016)に「真やねうら王」として出場したソフトです。

NDFの学習メソッドを用い、Hyperopt等を用いて各種パラメーターの調整を行い、技巧(2015)を超えた強さになりました。(R3650程度)


・やねうら王2016 Lateの遊び方

    将棋所かShogiGUIから思考エンジンとして登録して使ってください。
    実行ファイルはWindows 64bit版、CPUはAVX2用にコンパイルされています。
	またAVX2有りの場合、Skylake以降だと(Skylake以前のAVX2と比較して)
	10%程度さらに高速化されます。

    以下のようにファイルを配置します。

    YaneuraOu-2016-late.exe      : やねうら王2016 Late本体
    YaneuraOu-2016-late_ja.txt   : これを用いると思考エンジンの設定項目の表示が日本語化される。
    book/standard_book.db   : 基本定跡
	book/yaneura_book1.db   : やねうら大定跡(これは別途ダウンロードが必要)
	book/yaneura_book3.db   : 真やねうら定跡(これは別途ダウンロードが必要)
		※　やねうら大定跡のダウンロードは https://github.com/yaneurao/YaneuraOu の
			末尾のところにダウンロードリンクがあります。

	※　AVX2に対応していないCPUの場合、以下のなかから
　　　　　　ターゲット環境のCPUのものを選択して、それを
	    YaneuraOu-2016-late.exeとリネームしてお使いください。

		YaneuraOu-2016-late-sse42.exe : SSE4.2以降用
		YaneuraOu-2016-late-sse4.exe  : SSE4  以降用
		YaneuraOu-2016-late-sse2.exe  : SSE2  以降用
		YaneuraOu-2016-late-nosse.exe : SSEなし

    eval/KK_synthesized.bin        : 3駒関係の評価関数で用いるファイル(KK)
    eval/KKP_synthesized.bin       : 3駒関係の評価関数で用いるファイル(KKP)
    eval/KPP_synthesized.bin       : 3駒関係の評価関数で用いるファイル(KPP)

        evalフォルダに入れる評価関数バイナリ(上記の3ファイル)は、以下のところからダウンロード出来ます。

			https://github.com/yaneurao/YaneuraOu
			の「やねうら王評価関数バイナリ」のところ。

		また、Apery(WCSC26),Apery(2016 = 浮かむ瀬)の
		評価関数ファイルもそのまま使えます。
		これについては、Aperyの公式サイトからダウンロード出来ます。

			Aperyの公式サイト
			http://hiraokatakuya.github.io/apery/
			の
			第26回世界コンピュータ将棋選手権バージョン (for Windows 64bit)
			をクリックしてダウンロードしたファイルを解凍して、
			20160307/
			のフォルダのなかに入っているものを、やねうら王の実行ファイル配下のevalフォルダにコピーします。

    ・入玉宣言勝ちに対応しています。
    ・Ponder(相手番で思考する)に対応しています。
    ・秒読み、フィッシャールールに対応しています。
    ・最小思考時間設定に対応しています。
    ・スレッド数は思考エンジン設定で選べます。
    ・定跡の指し手がランダムに選択されます。
    ・置換表サイズは、思考エンジン設定のところで設定した値に従います。

	その他は、docs/
			https://github.com/yaneurao/YaneuraOu/tree/master/docs
	配下にある、
		解説.txt
		USI拡張コマンド.txt
	なども併せてご覧ください。
