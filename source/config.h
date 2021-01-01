#ifndef _CONFIG_H_INCLUDED
#define _CONFIG_H_INCLUDED

//
//  やねうら王プロジェクト
//  公式サイト :  http://yaneuraou.yaneu.com/yaneuraou_mini/
//

// 思考エンジンのバージョンとしてUSIプロトコルの"usi"コマンドに応答するときの文字列。
// ただし、この値を数値として使用することがあるので数値化できる文字列にしておく必要がある。
#define ENGINE_VERSION "6.00"


// --------------------
//  思考エンジンの種類
// --------------------


// やねうら王の思考エンジンとしてリリースする場合、以下から選択。(どれか一つは必ず選択しなければならない)
// ここに書かれていないエンジンもあるのでMakefileを見ること。
// オリジナルの思考エンジンをユーザーが作成する場合は、USER_ENGINE を defineして 他のエンジンのソースコードを参考に
//  engine/user-engine/ フォルダの中身を書くべし。

//#define YANEURAOU_ENGINE_DEEP            // ふかうら王
//#define YANEURAOU_ENGINE_NNUE            // やねうら王 通常探索部 NNUE評価関数
//#define YANEURAOU_ENGINE_KPPT            // やねうら王 通常探索部 KPPT評価関数
//#define YANEURAOU_ENGINE_KPP_KKPT        // やねうら王 通常探索部 KPP_KKPT評価関数
//#define YANEURAOU_ENGINE_MATERIAL        // やねうら王 通常探索部 駒得評価関数
//#define TANUKI_MATE_ENGINE               // tanuki- 詰め将棋solver  (2017/05/06～)
//#define YANEURAOU_MATE_ENGINE            // やねうら王 詰将棋solver (2020/12/29～)
//#define USER_ENGINE                      // ユーザーの思考エンジン


// --------------------
//  CPUの種類、CPU拡張命令の使用の有無
// --------------------


// USE_AVX512VNNI : AVX-512かつ、VNNI命令対応(Cascade Lake以降)でサポートされた命令を使うか。
// USE_AVX512 : AVX-512(サーバー向けSkylake以降)でサポートされた命令を使うか。
// USE_AVX2   : AVX2(Haswell以降)でサポートされた命令を使うか。pextなど。
// USE_SSE42  : SSE4.2でサポートされた命令を使うか。popcnt命令など。
// USE_SSE41  : SSE4.1でサポートされた命令を使うか。_mm_testz_si128など。
// USE_SSE2   : SSE2  でサポートされた命令を使うか。
// NO_SSE     : SSEは使用しない。
// (Windowsの64bit環境だと自動的にSSE2は使えるはず)
// noSSE ⊂ SSE2 ⊂ SSE4.1 ⊂ SSE4.2 ⊂ AVX2 ⊂  AVX-512

// Visual Studioのプロジェクト設定で「構成のプロパティ」→「C / C++」→「コード生成」→「拡張命令セットを有効にする」
// のところの設定の変更も忘れずに。

// ターゲットCPUのところだけdefineしてください。(残りは自動的にdefineされます。)

//#define USE_AVX512VNNI
//#define USE_AVX512
//#define USE_AVX2
//#define USE_SSE42
//#define USE_SSE41
//#define USE_SSSE3
//#define USE_SSE2
//#define NO_SSE

// BMI2命令を使う/使わない

// AVX2環境でBMI2が使えるときに、BMI2対応命令を使うのか(ZEN/ZEN2ではBMI2命令は使わないほうが速い)
// AMDのRyzenシリーズ(ZEN1/ZEN2)では、BMI2命令遅いので、これを定義しないほうが速い。
//#define USE_BMI2


// ---------------------
//  探索に関する設定
// ---------------------


// Position::see()を用いるか。これはSEE(Static Exchange Evaluation : 静的取り合い評価)の値を返す関数。
// #define USE_SEE


// Timerクラスに、今回の思考時間を計算する機能を追加するか。
// #define USE_TIME_MANAGEMENT


// MovePickerを使うのか(Stockfish風のコード)
// ※　通常探索部以外では使わないので、切り離せるようになっている。
// #define USE_MOVE_PICKER


// 入玉時の宣言勝ち機能を用いるか
// これをdefineすると、"EnteringKingRule"というオプションが自動追加される。
// ※　Search::Limits.enteringKingRule に↑のオプションの値が反映される。
//     Position::DeclarationWin()は、宣言勝ち判定を行うときに、それを見る。
// #define USE_ENTERING_KING_WIN


// 全合法手を生成する機能を追加するか。
// これをdefineすると、"GenerateAllLegalMoves"というオプションが自動追加される。
// ※　Search::Limits.generate_all_legal_moves に↑のオプションの値が反映されるので、
//     それを見て、trueならば、指し手生成の時に歩の不成も含めて生成すれば良い。
// #define USE_GENERATE_ALL_LEGAL_MOVES


// PV(読み筋)を表示するときに置換表の指し手をかき集めてきて表示するか。
// 自前でPVを管理してRootMoves::pvを更新するなら、この機能を使う必要はない。
// これはPVの更新が不要なので実装が簡単だが、Ponderの指し手を返すためには
// PVが常に正常に更新されていないといけないので最近はこの方法は好まれない。
// ただしShogiGUIの解析モードでは思考エンジンが出力した最後の読み筋を記録するようなので、
// 思考を途中で打ち切るときに、fail low/fail highが起きていると、中途半端なPVが出力され、それが棋譜に残る。
// かと言って、そのときにPVの出力をしないと、最後に出力されたPVとbest moveとは異なる可能性があるので、
// それはよろしくない。検討モード用の思考オプションを用意すべき。
// #define USE_TT_PV
// →　ConsiderationMode というエンジンオプションを用意したので、この機能は無効化する。


// ---------------------
//  評価関数関連の設定
// ---------------------


// 評価関数を使うオプション。
// これを定義するなら、Eval::init() , Eval::compute_eval() , Eval::print_eval_stat()などを
// 評価関数側で定義しないといけない。これらの評価関数を用いないなら、これは定義しないこと。
//#define USE_EVAL


// 協力詰め用思考エンジンなどで評価関数を使わないときにまで評価関数用のテーブルを
// 確保するのはもったいないので、そのテーブルを確保するかどうかを選択するためのオプション。
// 評価関数を用いるなら、どれか一つを選択すべし。

// 「○」がついているもの..実装済み
// 「△」がついているもの..参考実装。
// 「×」がついているもの..実装予定なし
// 「？」がついているもの..実装するかも
// 「！」がついているもの..かつて実装していたがサポートを終了したもの。

// #define EVAL_MATERIAL  // ○  駒得のみの評価関数 ※4
// #define EVAL_PP        // ×  ツツカナ型 2駒関係(開発予定なし)
// #define EVAL_KPP       // ！  Bonanza型 3駒関係、手番なし
// #define EVAL_KPPT      // ○  Bonanza型 3駒関係、手番つき(Apery WCSC26相当)
// #define EVAL_KPP_KKPT  // ○  KK手番あり + KKP手番あり + KPP手番なし(Ponanza WCSC26相当？)
// #define EVAL_KPP_PPT   // ×  PP手番あり + KKP手番あり + KPP手番なし(実装、途中まで)※1
// #define EVAL_KPPP_KKPT // △  KKP手番あり + KPP手番なし + KPPP(4駒関係)手番なし。→　※2,※3
// #define EVAL_KPPPT     // △  KPPP(4駒関係)手番あり。→　実装したけどいまひとつだったので差分計算実装せず。※2,※3
// #define EVAL_PPET      // ×  技巧型 2駒+利き+手番(実装予定なし)
// #define EVAL_KKPPT     // ○  KKPP型 4駒関係 手番あり。(55将棋、56将棋でも使えそう)※3
// #define EVAL_KKPP_KKPT // ○  KKPP型 4駒関係 手番はKK,KKPTにのみあり。※3
// #define EVAL_DL        //     Deep Learning系の評価関数。dlshogiを参考に。※5

// ※1 : KPP_PPTは、差分計算が面倒で割に合わないことが判明したのでこれを使うぐらいならKPP_KKPTで十分だという結論。
// ※2 : 実装したけどいまひとつだったので差分計算実装せず。そのため遅すぎて、実質使い物にならない。ソースコードの参考用。
// ※3 : このシンボルの値として対象とする王の升の数を指定する。例えばEVAL_KPPPTを27とdefineすると玉が自陣(3*9升 = 27)に
//       いるときのみがKPPPTの評価対象となる。(そこ以外に玉があるときは普通のKPPT)
// ※4 : MATERIAL_LEVELというシンボルで評価関数のタイプを選択できる。
//       #define MATERIAL_LEVEL 1 なら、駒得のみの評価関数
//       #define MATERIAL_LEVEL 2 なら…
//       → eval/material/evaluate_material.cppに定義があるのでそちらを見ること。
// ※5 : // あとで

// 駒の価値のテーブルを使うか。(Apery WCSC26の定義)
// Eval::PieceValueなどが使えるようになる。
//#define USE_PIECE_VALUE

// do_move()のときに移動した駒の管理をして差分計算
// 1. 駒番号を管理しているpiece_listについて
//   これはPosition::eval_list()で取得可能。
// 2. 移動した駒の管理について
//   これは、Position::state()->dirtyPiece。
//   最大で2個。
// 3. 駒番号が何の駒であるかが決まっている。(PieceNumber型)
//#define USE_EVAL_LIST


// 評価関数を計算したときに、それをHashTableに記憶しておく機能。KPPT評価関数においてのみサポート。
// #define USE_EVAL_HASH


// 評価関数パラメーターを共有メモリを用いて他プロセスのものと共有する。
// 少ないメモリのマシンで思考エンジンを何十個も立ち上げようとしたときにメモリ不足になるので
// 評価関数をshared memoryを用いて他のプロセスと共有する機能。(対応しているのはいまのところKPPT評価関数のみ。かつWindows限定)
// #define USE_SHARED_MEMORY_IN_EVAL


// 評価関数で金と小駒の成りを区別する
// 駒の特徴量はBonaPiece。これはBonanzaに倣っている。
// このオプションを有効化すると、金と小駒の成りを区別する。(Bonanzaとは異なる特徴量になる)
// #define DISTINGUISH_GOLDS


// ---------------------
//  機械学習関連の設定
// ---------------------

// 評価関数を教師局面から学習させるときに使うときのモード
//#define EVAL_LEARN


// 教師生成用の特殊コマンド"gensfen2019"を使えるようにするモード。
// 教師生成用の探索パラメーターも別途用意するといいかも。
//#define GENSFEN2019


// sfenを256bitにpackする機能、unpackする機能を有効にする。
// これをdefineするとPosition::packe_sfen(),unpack_sfen()が使えるようになる。
// ※　機械学習関連で局面の読み書きをする時に使う。
// #define USE_SFEN_PACKER


// 置換表のprobeに必ず失敗する設定
// ※　 自己生成棋譜からの学習でqsearch()のPVが欲しいときに
// 置換表にhitして枝刈りされたときにPVが得られないの悔しいので。
// #define USE_FALSE_PROBE_IN_TT


// ---------------------
//  詰将棋ルーチン関係の設定
// ---------------------

// 1手詰め判定ルーチンを用いるか。
// LONG_EFFECT_LIBRARYが有効なときは、利きを利用した高速な一手詰め。
// LONG_EFFECT_LIBRARYが無効なときは、Bonanza6風の一手詰め。
//#define USE_MATE_1PLY


// MateSolver(奇数手詰めルーチン)を用いるか。
// これをdefineするとMateSolverが使えるようになる。
// ※　同時にUSE_MATE_1PLYがdefineされていないといけない。
//#define USE_MATE_SOLVER


// df-pn詰将棋ルーチンを用いるか。
// これをdefineするとMate::Dfpnクラスが使えるようになる。(開発中)
// ※　同時にUSE_MATE_1PLYがdefineされていないといけない。
//#define USE_MATE_DFPN


// ---------------------
//  定跡に関する設定
// ---------------------


// 定跡を作るコマンド("makebook")を有効にする。
// #define ENABLE_MAKEBOOK_CMD


// ---------------------
//  高速化に関する設定
// ---------------------


// トーナメント(大会)用のビルド。最新CPU(いまはAVX2)用でEVAL_HASH大きめ。EVAL_LEARN、TEST_CMD使用不可。ASSERTなし。GlobalOptionsなし。
// #define FOR_TOURNAMENT

// sortが少し高速化されるらしい。
// 安定ソートではないので並び順が以前のとは異なるから、benchコマンドの探索ノード数は変わる。
// CPU targetによって実装が変わるのでCPUによってbenchコマンドの探索ノード数は変わる。
//#define USE_SUPER_SORT


// ---------------------
// ふかうら王(dlshogi互換エンジン)に関する設定。
// ---------------------


// ふかうら王で、ONNXRUNTIMEを用いて推論を行うときは、これをdefineすること。
// ※　GPUがなくても動作する。
//#define ONNXRUNTIME

// ふかうら王でTensorRTを使う時はこちら。
//#define TENSOR_RT


// ---------------------
// 探索パラメーターの自動調整用
// ---------------------


// 探索パラメーターのチューニングを行うモード
//
// 実行時に"param/yaneuraou-param.h" からパラメーターファイルを読み込むので
// "source/engine/yaneuraou-engine/yaneuraou-param.h"をそこに配置すること。
//#define TUNING_SEARCH_PARAMETERS


// ---------------------
// デバッグに関する設定
// ---------------------


// assertのレベルを6段階で。
//  ASSERT_LV 0 : assertなし(全体的な処理が速い)
//  ASSERT_LV 1 : 軽量なassert
//  　　　…
//  ASSERT_LV 5 : 重度のassert(全体的な処理が遅い)
// あまり重度のassertにすると、探索性能が落ちるので時間当たりに調べられる局面数が低下するから
// そのへんのバランスをユーザーが決めれるようにこの仕組みを導入。

//#define ASSERT_LV 3

// ASSERTのリダイレクト
// ASSERTに引っかかったときに、それを"Error : x=1"のように標準出力に出力する。

//#define USE_DEBUG_ASSERT


// USI拡張コマンドの"test"コマンドを有効にする。
// 非常にたくさんのテストコードが書かれているのでコードサイズが膨らむため、
// 思考エンジンとしてリリースするときはコメントアウトしたほうがいいと思う。

//#define ENABLE_TEST_CMD


// ---------------------
// その他、オプション機能
// ---------------------


// 長い利き(遠方駒の利き)のライブラリを用いるか。
// 超高速1手詰め判定などではこのライブラリが必要。
// do_move()のときに利きの差分更新を行なうので、do_move()は少し遅くなる。(その代わり、利きが使えるようになる)
//#define LONG_EFFECT_LIBRARY


// やねうら王の従来の遠方駒の利きを求めるコードを用いる。
// これをundefするとApery型の利きのコードを用いる。(そっちのほうがPEXTが使えない環境だと速い)
// 互換性維持および、55将棋のように盤面を変形させるときに、magic tableで用いるmagic numberを求めたくないときに用いる。

// #define USE_OLD_YANEURAOU_EFFECT


// position.hのStateInfoに直前の指し手、移動させた駒などの情報を保存しておくのか
// これが保存されていると詰将棋ルーチンなどを自作する場合においてそこまでの手順を表示するのが簡単になる。
// (Position::moves_from_start_pretty()などにより、わかりやすい手順が得られる。
// ただし通常探索においてはやや遅くなるので思考エンジンとしてリリースするときには無効にしておくこと。

//#define KEEP_LAST_MOVE


// GlobalOptionという、EVAL_HASHを有効/無効を切り替えたり、置換表の有効/無効を切り替えたりする
// オプションのための変数が使えるようになる。スピードが1%ぐらい遅くなるので大会用のビルドではオフを推奨。
// #define USE_GLOBAL_OPTIONS


// USIプロトコルでgameoverコマンドが送られてきたときに gameover_handler()を呼び出す。
// #define USE_GAMEOVER_HANDLER


// "Threads"オプション が 8以下の設定の時でも強制的に bindThisThread()を呼び出して、指定されたNUMAで動作するようにする。
// "ThreadIdOffset"オプションと併用して、狙ったNUMAで動作することを強制することができる。
//#define FORCE_BIND_THIS_THREAD


// PVの出力時の千日手に関する出力をすべて"rep_draw"に変更するオプション。
// GUI側が、何らかの都合で"rep_draw"のみしか処理できないときに用いる。
// #define PV_OUTPUT_DRAW_ONLY


// ニコニコ生放送の電王盤用
// 電王盤はMultiPV非対応なので定跡を送るとき、"multipv"をつけずに1番目の候補手を送信する必要がある。
// #define NICONICO


// ===============================================================
// ここ以降では、↑↑↑で設定した内容に基づき必要なdefineを行う。
// ===============================================================

// 通例hash keyは64bitだが、これを128にするとPosition::state()->long_key()から128bit hash keyが
// 得られるようになる。研究時に局面が厳密に合致しているかどうかを判定したいときなどに用いる。
// 実験用の機能なので、128bit,256bitのhash keyのサポートはAVX2のみ。
#define HASH_KEY_BITS 64
//#define HASH_KEY_BITS 128
//#define HASH_KEY_BITS 256

// 通常探索時の最大探索深さ
constexpr int MAX_PLY_NUM = 246;

// デバッグ時の標準出力への局面表示などに日本語文字列を用いる。

#define PRETTY_JP

// --------------------
// release configurations
// --------------------

// --- 通常の思考エンジンとして実行ファイルを公開するとき用の設定集

#if defined(YANEURAOU_ENGINE_KPPT) || defined(YANEURAOU_ENGINE_KPP_KKPT) || defined(YANEURAOU_ENGINE_NNUE) || defined(YANEURAOU_ENGINE_MATERIAL)

	#define ENGINE_NAME "YaneuraOu"

	// 通常のやねうら王探索部(Stockfishっぽいやつ)を用いる。
	#define YANEURAOU_ENGINE

	#define USE_PIECE_VALUE
	#define USE_SEE
	#define USE_EVAL_LIST
	#define USE_MATE_1PLY
	#define USE_MATE_SOLVER
	#define USE_MATE_DFPN
	#define USE_TIME_MANAGEMENT
	#define USE_MOVE_PICKER
	#define USE_EVAL
	#define USE_GENERATE_ALL_LEGAL_MOVES
	#define USE_ENTERING_KING_WIN

	#if defined(YANEURAOU_ENGINE_KPPT) || defined(YANEURAOU_ENGINE_KPP_KKPT)
		// EvalHashを用いるのは3駒型のみ。それ以外は差分計算用の状態が大きすぎてhitしたところでどうしようもない。
		#define USE_EVAL_HASH

	// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
	#define USE_SHARED_MEMORY_IN_EVAL
	#endif

	// 学習機能を有効にするオプション。
	// 教師局面の生成、定跡コマンド(makebook thinkなど)を用いる時には、これを
	// 有効化してコンパイルしなければならない。
	//#define EVAL_LEARN

	// デバッグ絡み
	//#define ASSERT_LV 3
	//#define USE_DEBUG_ASSERT

	#define ENABLE_TEST_CMD
	// 学習絡みのオプション
	#define USE_SFEN_PACKER

	// 定跡生成絡み
	#define ENABLE_MAKEBOOK_CMD

	// パラメーターの自動調整絡み
	#define USE_GAMEOVER_HANDLER
	//#define LONG_EFFECT_LIBRARY

	// GlobalOptionsは有効にしておく。
	#define USE_GLOBAL_OPTIONS

	// -- 各評価関数ごとのconfiguration

	#if defined(YANEURAOU_ENGINE_MATERIAL)

		#define EVAL_MATERIAL
		// 駒割のみの評価関数ではサポートされていない機能をundefする。
		#undef EVAL_LEARN

		// 実験用評価関数
		// 駒得評価関数の拡張扱いをする。
		#if MATERIAL_LEVEL >= 002
			// evaluate()のために利きが必要。
			#define LONG_EFFECT_LIBRARY
		#endif
	#endif

	#if defined(YANEURAOU_ENGINE_KPPT)
		#define EVAL_KPPT
	#endif

	#if defined(YANEURAOU_ENGINE_KPP_KKPT)
		#define EVAL_KPP_KKPT
	#endif

	#if defined(YANEURAOU_ENGINE_NNUE)
		#define EVAL_NNUE

		// 学習のためにOpenBLASを使う
		// "../openblas/lib/libopenblas.dll.a"をlibとして追加すること。
		//#define USE_BLAS

		// NNUEの使いたい評価関数アーキテクチャの選択
		//
		// EVAL_NNUE_HALFKP256  : 標準NNUE型(評価関数ファイル60MB程度)
		// EVAL_NNUE_KP256      : KP256(評価関数1MB未満)
		// EVAL_NNUE_HALFKPE9   : 標準NNUE型のおよそ9倍(540MB程度)

		 //#undef EVAL_NNUE_KP256
		 //#define EVAL_NNUE_HALFKPE9

		// #define EVAL_NNUE_HALFKP256
		// #define EVAL_NNUE_KP256
		// #define EVAL_NNUE_HALFKPE9
	#endif

#endif // defined(YANEURAOU_ENGINE_KPPT) || ...

// --- Deep Learning系のエンジン ふかうら王(dlshogi互換エンジン)

#if defined(YANEURAOU_ENGINE_DEEP)

	#define ENGINE_NAME "FukauraOu"
	#define EVAL_DEEP "dlshogi-denryu2020"
	#define USE_EVAL
	#define USE_TIME_MANAGEMENT
	#define USE_GENERATE_ALL_LEGAL_MOVES
	#define USE_ENTERING_KING_WIN
	#define USE_MATE_1PLY
	#define USE_MATE_SOLVER
	#define USE_MATE_DFPN
	#define USE_PIECE_VALUE
	#define ENABLE_TEST_CMD

//#define USE_KIF_CONVERT_TOOLS

	//#define ASSERT_LV 3
#endif

// --- tanuki-詰将棋エンジンとして実行ファイルを公開するとき用の設定集

#if defined(TANUKI_MATE_ENGINE)
	#define ENGINE_NAME "tanuki- mate solver"
	#define KEEP_LAST_MOVE
	#undef  MAX_PLY_NUM
	#define MAX_PLY_NUM 2000
	#define USE_MATE_1PLY
	//#define LONG_EFFECT_LIBRARY
	#define USE_KEY_AFTER
	#define ENABLE_TEST_CMD
#endif

// --- やねうら王詰将棋エンジンとして実行ファイルを公開するとき用の設定集

#if defined(YANEURAOU_MATE_ENGINE)
	#define ENGINE_NAME "YaneuraOu mate solver"
	#undef  MAX_PLY_NUM
	#define MAX_PLY_NUM 2000
	#define USE_MATE_1PLY
	//#define LONG_EFFECT_LIBRARY
	#define USE_MATE_SOLVER
	#define USE_MATE_DFPN
	#define USE_PIECE_VALUE
	#define ENABLE_TEST_CMD
#endif


// --- ユーザーの自作エンジンとして実行ファイルを公開するとき用の設定集

#if defined(USER_ENGINE)
	#define ENGINE_NAME "YaneuraOu user engine"
	#define USE_SEE
	#define USE_EVAL
	#define EVAL_MATERIAL
	#define USE_PIECE_VALUE
#endif

// --------------------
//   for tournament
// --------------------

// トーナメント(大会)用に、対局に不要なものをすべて削ぎ落とす。
#if defined(FOR_TOURNAMENT)
	#undef ASSERT_LV
	#undef EVAL_LEARN
	#undef ENABLE_TEST_CMD
	#undef USE_GLOBAL_OPTIONS
	#undef KEEP_LAST_MOVE
#endif

// --------------------
//   for learner
// --------------------

// 学習時にはEVAL_HASHを無効化しておかないと、rmseの計算のときなどにeval hashにhitしてしまい、
// 正しく計算できない。そのため、EVAL_HASHを動的に無効化するためのオプションを用意する。
#if defined(EVAL_LEARN)
	#define USE_GLOBAL_OPTIONS
#endif

// --------------------
//   GlobalOptions
// --------------------

#if defined(USE_GLOBAL_OPTIONS)

struct GlobalOptions_
{
	// eval hashを有効/無効化する。
	// (USE_EVAL_HASHがdefineされていないと有効にはならない。)
	bool use_eval_hash;

	// 置換表のprobe()を有効化/無効化する。E
	// (無効化するとTT.probe()が必ずmiss hitするようになる)
	bool use_hash_probe;

	GlobalOptions_()
	{
		use_eval_hash = use_hash_probe = true;
	}
};

extern GlobalOptions_ GlobalOptions;

#endif

// --------------------
//      configure
// --------------------

// --- assertion tools

// DEBUGビルドでないとassertが無効化されてしまうので無効化されないASSERT
// 故意にメモリアクセス違反を起こすコード。
// USE_DEBUG_ASSERTが有効なときには、ASSERTの内容を出力したあと、3秒待ってから
// アクセス違反になるようなコードを実行する。
#if !defined (USE_DEBUG_ASSERT)
#define ASSERT(X) { if (!(X)) *(int*)1 = 0; }
#else
#include <iostream>
#include <chrono>
#include <thread>
#define ASSERT(X) { if (!(X)) { std::cout << "\nError : ASSERT(" << #X << "), " << __FILE__ << "(" << __LINE__ << "): " << __func__ << std::endl; \
 std::this_thread::sleep_for(std::chrono::microseconds(3000)); *(int*)1 =0;} }
#endif

// ASSERT LVに応じたassert
#ifndef ASSERT_LV
#define ASSERT_LV 0
#endif

#define ASSERT_LV_EX(L, X) { if (L <= ASSERT_LV) ASSERT(X); }
#define ASSERT_LV1(X) ASSERT_LV_EX(1, X)
#define ASSERT_LV2(X) ASSERT_LV_EX(2, X)
#define ASSERT_LV3(X) ASSERT_LV_EX(3, X)
#define ASSERT_LV4(X) ASSERT_LV_EX(4, X)
#define ASSERT_LV5(X) ASSERT_LV_EX(5, X)

// --- declaration of unreachablity

// switchにおいてdefaultに到達しないことを明示して高速化させる

// デバッグ時は普通にしとかないと変なアドレスにジャンプして原因究明に時間がかかる。
#if defined(_MSC_VER)
#define UNREACHABLE ASSERT_LV3(false); __assume(0);
#elif defined(__GNUC__)
#define UNREACHABLE ASSERT_LV3(false); __builtin_unreachable();
#else
#define UNREACHABLE ASSERT_LV3(false);
#endif

// --- alignment tools

// 構造体などのアライメントを揃えるための宣言子

#if defined(_MSC_VER)
#define ALIGNED(X) __declspec(align(X))
#elif defined(__GNUC__)
#define ALIGNED(X) __attribute__ ((aligned(X)))
#else
#define ALIGNED(X)
#endif

// --- output for Japanese notation

// PRETTY_JPが定義されているかどうかによって三項演算子などを使いたいので。
#if defined (PRETTY_JP)
constexpr bool pretty_jp = true;
#else
constexpr bool pretty_jp = false;
#endif


// --- hash key bits

#if HASH_KEY_BITS <= 64
#define HASH_KEY Key64
#elif HASH_KEY_BITS <= 128
#define HASH_KEY Key128
#else
#define HASH_KEY Key256
#endif

// --- lastMove

// KIF形式に変換するときにPositionクラスにその局面へ至る直前の指し手が保存されていないと
// "同"金のように出力できなくて困る。
#if defined (USE_KIF_CONVERT_TOOLS)
	#define KEEP_LAST_MOVE
#endif

// ----------------------------
//      CPU environment
// ----------------------------

// ターゲットが64bitOSかどうか
#if (defined(_WIN64) && defined(_MSC_VER)) || (defined(__GNUC__) && defined(__x86_64__)) || defined(IS_64BIT)
	constexpr bool Is64Bit = true;
	#ifndef IS_64BIT
		#define IS_64BIT
	#endif
#else
	constexpr bool Is64Bit = false;
#endif

// TARGET_CPU、Makefileのほうで"ZEN2"のようにダブルコーテーション有りの文字列として定義されているはずだが、
// それが定義されていないならここでUSE_XXXオプションから推定する。
#if !defined(TARGET_CPU)
	#if defined(USE_BMI2)
	#define BMI2_STR "BMI2"
	#else
	#define BMI2_STR ""
	#endif

	#if defined(USE_AVX512VNNI)
	#define TARGET_CPU "AVX512VNNI" BMI2_STR
	#elif defined(USE_AVX512)
	#define TARGET_CPU "AVX512" BMI2_STR
	#elif defined(USE_AVX2)
	#define TARGET_CPU "AVX2" BMI2_STR
	#elif defined(USE_SSE42)
	#define TARGET_CPU "SSE4.2"
	#elif defined(USE_SSE41)
	#define TARGET_CPU "SSE4.1"
	#elif defined(USE_SSSE3)
	#define TARGET_CPU "SSSE3"
	#elif defined(USE_SSE2)
	#define TARGET_CPU "SSE2"
	#else
	#define TARGET_CPU "noSSE"
	#endif
#endif

// 上位のCPUをターゲットとするなら、その下位CPUの命令はすべて使えるはずなので…。

#if defined (USE_AVX512VNNI)
#define USE_AVX512
#endif

#if defined (USE_AVX512)
#define USE_AVX2
#endif

#if defined (USE_AVX2)
#define USE_SSE42
#endif

#if defined (USE_SSE42)
#define USE_SSE41
#endif

#if defined (USE_SSE41)
#define USE_SSSE3
#endif

#if defined (USE_SSSE3)
#define USE_SSE2
#endif


// --------------------
//    for 32bit OS
// --------------------

#if !defined(IS_64BIT)

// 32bit環境ではメモリが足りなくなるので以下の2つは強制的にオフにしておく。

#undef USE_EVAL_HASH
//#undef USE_SHARED_MEMORY_IN_EVAL

// 機械学習用の配列もメモリ空間に収まりきらないのでコンパイルエラーとなるから
// これもオフにしておく。
#undef EVAL_LEARN

#endif

// ----------------------------
//     evaluate function
// ----------------------------

// -- 評価関数の種類によりエンジン名に使用する文字列を変更する。
#if defined(EVAL_MATERIAL)
	#if defined(MATERIAL_LEVEL)
	// MATERIAL_LEVELの番号を"Level"として出力してやる。
		#define EVAL_TYPE_NAME "MaterialLv" << MATERIAL_LEVEL
	#else
	// 適切な評価関数がないので単にEVAL_MATERIALを指定しているだけだから、EVAL_TYPE_NAMEとしては空欄でいいかと。
		#define EVAL_TYPE_NAME ""
	#endif
#elif defined(EVAL_KPPT)
	#define EVAL_TYPE_NAME "KPPT"
#elif defined(EVAL_KPP_KKPT)
	#define EVAL_TYPE_NAME "KPP_KKPT"
#elif defined(EVAL_NNUE_KP256)
	#define EVAL_TYPE_NAME "NNUE KP256"
#elif defined(EVAL_NNUE_HALFKPE9)
	#define EVAL_TYPE_NAME "NNUE halfKPE9"
	// hafeKPE9には利きが必要
	#define LONG_EFFECT_LIBRARY
	#define USE_BOARD_EFFECT_PREV
#elif defined(EVAL_NNUE) // それ以外のNNUEなので標準NNUE halfKP256だと思われる。
	#define EVAL_TYPE_NAME "NNUE"
#elif defined(EVAL_DEEP)
	#if defined(ONNXRUNTIME)
		#if defined(ORT_CPU)
			#define EVAL_TYPE_NAME "ORT_CPU-" << EVAL_DEEP
		#elif defined(ORT_DML)
			#define EVAL_TYPE_NAME "ORT_DML-" << EVAL_DEEP
		#elif defined(ORT_MKL)
			#define EVAL_TYPE_NAME "ORT_MKL-" << EVAL_DEEP
		#else
			#define EVAL_TYPE_NAME "ORT-" << EVAL_DEEP
		#endif
	#elif defined(TENSOR_RT)
		#define EVAL_TYPE_NAME "TensorRT-" << EVAL_DEEP
	#endif

#else
	#define EVAL_TYPE_NAME ""
#endif


// -- 評価関数の種類により、盤面の利きの更新ときの処理が異なる。(このタイミングで評価関数の差分計算をしたいので)

// 盤面上の利きを更新するときに呼び出したい関数。(評価関数の差分更新などのために差し替え可能にしておく。)

// color = 手番 , sq = 升 , e = 利きの加算量
#define ADD_BOARD_EFFECT(color_,sq_,e1_) { board_effect[color_].e[sq_] += (uint8_t)e1_; }
// e1 = color側の利きの加算量 , e2 = ~color側の利きの加算量
#define ADD_BOARD_EFFECT_BOTH(color_,sq_,e1_,e2_) { board_effect[color_].e[sq_] += (uint8_t)e1_; board_effect[~color_].e[sq_] += (uint8_t)e2_; }

// ↑の関数のundo_move()時用。こちらは、評価関数の差分更新を行わない。(評価関数の値を巻き戻すのは簡単であるため)
#define ADD_BOARD_EFFECT_REWIND(color_,sq_,e1_) { board_effect[color_].e[sq_] += (uint8_t)e1_; }
#define ADD_BOARD_EFFECT_BOTH_REWIND(color_,sq_,e1_,e2_) { board_effect[color_].e[sq_] += (uint8_t)e1_; board_effect[~color_].e[sq_] += (uint8_t)e2_; }

#endif // ifndef _CONFIG_H_INCLUDED

