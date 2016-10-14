
・やねうら王2016 Midとは？

やねうら王2016 Midはやねうら王classic-tceに対してApery(WCSC26)の評価関数バイナリの
読み込みに対応させたものです。(R3400程度)

・やねうら王2016 Midの遊び方

    将棋所かShogiGUIから思考エンジンとして登録して使ってください。
    実行ファイルはWindows 64bit版、CPUはAVX2用にコンパイルされています。

    以下のようにファイルを配置します。

    YaneuraOu-2016-mid.exe      : やねうら王2016 Mid本体
    YaneuraOu-2016-mid_ja.txt   : これを用いると思考エンジンの設定項目の表示が日本語化される。
    book/standard_book.db   : 基本定跡
	book/yaneura_book1.db   : やねうら大定跡(これは別途ダウンロードが必要)
		※　やねうら大定跡のダウンロードは https://github.com/yaneurao/YaneuraOu の
			末尾のところにダウンロードリンクがあります。

	※　AVX2に対応していないCPUの場合、以下のなかから
　　　　　　ターゲット環境のCPUのものを選択して、それを
	    YaneuraOu-2016-mid.exeとリネームしてお使いください。

		YaneuraOu-2016-mid-sse42.exe : SSE4.2以降用
		YaneuraOu-2016-mid-sse4.exe  : SSE4  以降用
		YaneuraOu-2016-mid-sse2.exe  : SSE2  以降用
		YaneuraOu-2016-mid-nosse.exe : SSEなし

    eval/KK_synthesized.bin        : 3駒関係の評価関数で用いるファイル(KK)
    eval/KKP_synthesized.bin       : 3駒関係の評価関数で用いるファイル(KKP)
    eval/KPP_synthesized.bin       : 3駒関係の評価関数で用いるファイル(KPP)

        evalフォルダに入れる評価関数バイナリ(上記の3ファイル)は、以下のところからダウンロード出来ます。

			やねうら王で使える評価関数ファイル28バリエーション公開しました
			http://yaneuraou.yaneu.com/2016/07/22/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%A7%E4%BD%BF%E3%81%88%E3%82%8B%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB28%E3%83%90%E3%83%AA%E3%82%A8%E3%83%BC%E3%82%B7/

		また、Apery(WCSC26)の評価関数ファイルもそのまま使えます。
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
