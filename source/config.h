#ifndef _CONFIG_H_INCLUDED
#define _CONFIG_H_INCLUDED

//
//  やねうら王プロジェクト
//  公式サイト :  http://yaneuraou.yaneu.com/yaneuraou_mini/
//

// 思考エンジンのバージョンとしてUSIプロトコルの"usi"コマンドに応答するときの文字列。
// ただし、この値を数値として使用することがあるので数値化できる文字列にしておく必要がある。
#define ENGINE_VERSION "4.86"

// --------------------
//  思考エンジンの種類
// --------------------

// やねうら王の思考エンジンとしてリリースする場合、以下から選択。(どれか一つは必ず選択しなければならない)
// オリジナルの思考エンジンをユーザーが作成する場合は、USER_ENGINE を defineして 他のエンジンのソースコードを参考に
//  engine/user-engine/ フォルダの中身を書くべし。

#if !defined (USE_MAKEFILE)

#define YANEURAOU_2018_OTAFUKU_ENGINE    // やねうら王2018 with お多福Lab。(開発中2018/01/01～)
//#define MATE_ENGINE                      // 詰め将棋solverとしてリリースする場合。(開発中2017/05/06～)
//#define USER_ENGINE                      // ユーザーの思考エンジン

#else

// Makefileを使ってビルドをするときは、Makefile側で選択する。

#endif

// --------------------
// コンパイル時設定
// --------------------

// --- ターゲットCPUの選択

#if !defined(USE_MAKEFILE)

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

//#define USE_AVX512
#define USE_AVX2
//#define USE_SSE42
//#define USE_SSE41
//#define USE_SSE2
//#define NO_SSE

#else

// Makefileを使ってbuildするときは、
// $ make avx2
// のようにしてビルドすれば自動的にAVX2用がビルドされます。

#endif


// 通例hash keyは64bitだが、これを128にするとPosition::state()->long_key()から128bit hash keyが
// 得られるようになる。研究時に局面が厳密に合致しているかどうかを判定したいときなどに用いる。
// 実験用の機能なので、128bit,256bitのhash keyのサポートはAVX2のみ。
#define HASH_KEY_BITS 64
//#define HASH_KEY_BITS 128
//#define HASH_KEY_BITS 256


// 通常探索時の最大探索深さ
#define MAX_PLY_NUM 127

// --- デバッグ時の標準出力への局面表示などに日本語文字列を用いる。

#define PRETTY_JP

//
// 以下、デフォルトではdefineしていないので、必要に応じてdefineすること。
//

// --- assertのレベルを6段階で。
//  ASSERT_LV 0 : assertなし(全体的な処理が速い)
//  ASSERT_LV 1 : 軽量なassert
//  　　　…
//  ASSERT_LV 5 : 重度のassert(全体的な処理が遅い)
// あまり重度のassertにすると、探索性能が落ちるので時間当たりに調べられる局面数が低下するから
// そのへんのバランスをユーザーが決めれるようにこの仕組みを導入。

//#define ASSERT_LV 3

// --- ASSERTのリダイレクト
// ASSERTに引っかかったときに、それを"Error : x=1"のように標準出力に出力する。

//#define USE_DEBUG_ASSERT


// --- USI拡張コマンドの"test"コマンドを有効にする。
// 非常にたくさんのテストコードが書かれているのでコードサイズが膨らむため、
// 思考エンジンとしてリリースするときはコメントアウトしたほうがいいと思う。

//#define ENABLE_TEST_CMD

// --- StateInfoに直前の指し手、移動させた駒などの情報を保存しておくのか
// これが保存されていると詰将棋ルーチンなどを自作する場合においてそこまでの手順を表示するのが簡単になる。
// (Position::moves_from_start_pretty()などにより、わかりやすい手順が得られる。
// ただし通常探索においてはやや遅くなるので思考エンジンとしてリリースするときには無効にしておくこと。

//#define KEEP_LAST_MOVE

// 協力詰め用思考エンジンなどで評価関数を使わないときにまで評価関数用のテーブルを
// 確保するのはもったいないので、そのテーブルを確保するかどうかを選択するためのオプション。
// 評価関数を用いるなら、どれか一つを選択すべし。

// 「○」がついているもの..実装済み
// 「△」がついているもの..参考実装。
// 「×」がついているもの..実装予定なし
// 「？」がついているもの..実装するかも
// 「！」がついているもの..かつて実装していたがサポートを終了したもの。

// #define EVAL_NO_USE    // ！　評価関数なし。※4
// #define EVAL_MATERIAL  // ○  駒得のみの評価関数
// #define EVAL_PP        // ×  ツツカナ型 2駒関係(開発予定なし)
// #define EVAL_KPP       // ！  Bonanza型 3駒関係、手番なし
// #define EVAL_KPPT      // ○  Bonanza型 3駒関係、手番つき(Apery WCSC26相当)
// #define EVAL_KPP_KKPT  // ○  KK手番あり + KKP手番あり + KPP手番なし(Ponanza WCSC26相当？)
// #define EVAL_KPP_KKPT_FV_VAR // ○ KPP_KKPTと同一。※5
// #define EVAL_KPP_PPT   // ×  PP手番あり + KKP手番あり + KPP手番なし(実装、途中まで)※1
// #define EVAL_KPPP_KKPT // △  KKP手番あり + KPP手番なし + KPPP(4駒関係)手番なし。→　※2,※3
// #define EVAL_KPPPT     // △  KPPP(4駒関係)手番あり。→　実装したけどいまひとつだったので差分計算実装せず。※2,※3
// #define EVAL_PPET      // ×  技巧型 2駒+利き+手番(実装予定なし)
// #define EVAL_KKPPT     // ○  KKPP型 4駒関係 手番あり。(55将棋、56将棋でも使えそう)※3
// #define EVAL_KKPP_KKPT // ○  KKPP型 4駒関係 手番はKK,KKPTにのみあり。※3
// #define EVAL_NABLA     // ○  ∇(ナブラ) 評価関数(現状、非公開)
// #define EVAL_HELICES   // ？  螺旋評価関数

// ※1 : KPP_PPTは、差分計算が面倒で割に合わないことが判明したのでこれを使うぐらいならKPP_KKPTで十分だという結論。
// ※2 : 実装したけどいまひとつだったので差分計算実装せず。そのため遅すぎて、実質使い物にならない。ソースコードの参考用。
// ※3 : このシンボルの値として対象とする王の升の数を指定する。例えばEVAL_KPPPTを27とdefineすると玉が自陣(3*9升 = 27)に
//       いるときのみがKPPPTの評価対象となる。(そこ以外に玉があるときは普通のKPPT)
// ※4 : 以前、EVAL_NO_USEという評価関数なしのものが選択できるようになっていたが、
//       需要がほとんどない上に、ソースコードがifdefの嵐になるので読みづらいのでバッサリ削除した。
//		代わりにEVAL_MATERIALを使うと良い。追加コストはほぼ無視できる。
// ※5 : 可変長EvalListを用いるリファレンス実装。KPP_KKPT型に比べてわずかに遅いが、拡張性が非常に高く、
//      極めて美しい実装なので、今後、評価関数の拡張は、これをベースにやっていくことになると思う。


// 評価関数を教師局面から学習させるときに使うときのモード
// #define EVAL_LEARN

// Eval::compute_eval()やLearner::add_grad()を呼び出す前にEvalListの組み換えを行なう機能を提供する。
// 評価関数の実験に用いる。詳しくは、Eval::make_list_functionに書いてある説明などを読むこと。
// #define USE_EVAL_MAKE_LIST_FUNCTION

// この機能は、やねうら王の評価関数の開発/実験用の機能で、いまのところ一般ユーザーには提供していない。
// 評価関数番号を指定するとその評価関数を持ち、その評価関数ファイルの読み込み/書き出しに自動的に対応して、
// かつ評価関数の旧形式からの変換が"test convert"コマンドで自動的に出来るようになるという、わりかし凄い機能
// #define EVAL_EXPERIMENTAL 0001

// 長い利き(遠方駒の利き)のライブラリを用いるか。
// 超高速1手詰め判定などではこのライブラリが必要。
// do_move()のときに利きの差分更新を行なうので、do_move()は少し遅くなる。(その代わり、利きが使えるようになる)
//#define LONG_EFFECT_LIBRARY

// 1手詰め判定ルーチンを用いるか。
// LONG_EFFECT_LIBRARYが有効なときは、利きを利用した高速な一手詰め。
// LONG_EFFECT_LIBRARYが無効なときは、Bonanza6風の一手詰め。
//#define USE_MATE_1PLY

// Position::see()を用いるか。これはSEE(Static Exchange Evaluation : 静的取り合い評価)の値を返す関数。
// #define USE_SEE

// PV(読み筋)を表示するときに置換表の指し手をかき集めてきて表示するか。
// 自前でPVを管理してRootMoves::pvを更新するなら、この機能を使う必要はない。
// これはPVの更新が不要なので実装が簡単だが、Ponderの指し手を返すためには
// PVが常に正常に更新されていないといけないので最近はこの方法は好まれない。
// ただしShogiGUIの解析モードでは思考エンジンが出力した最後の読み筋を記録するようなので、
// 思考を途中で打ち切るときに、fail low/fail highが起きていると、中途半端なPVが出力され、それが棋譜に残る。
// かと言って、そのときにPVの出力をしないと、最後に出力されたPVとbest moveとは異なる可能性があるので、
// それはよろしくない。検討モード用の思考オプションを用意すべき。
// #define USE_TT_PV

// 定跡を作るコマンド("makebook")を有効にする。
// #define ENABLE_MAKEBOOK_CMD

// 入玉時の宣言勝ちを用いるか
// #define USE_ENTERING_KING_WIN

// TimeMangementクラスに、今回の思考時間を計算する機能を追加するか。
// #define USE_TIME_MANAGEMENT

// 置換表のなかでevalを持たない
// #define NO_EVAL_IN_TT

// ONE_PLY == 1にするためのモード。これを指定していなければONE_PLY == 2
// #define ONE_PLY_EQ_1

// オーダリングに使っているStatsの配列のなかで駒打ちのためのbitを持つ。
// #define USE_DROPBIT_IN_STATS

// 指し手生成のときに上位16bitにto(移動後の升)に来る駒を格納する。
// #define KEEP_PIECE_IN_GENERATE_MOVES

// 評価関数を計算したときに、それをHashTableに記憶しておく機能。KPPT評価関数においてのみサポート。
// #define USE_EVAL_HASH

// sfenを256bitにpackする機能、unpackする機能を有効にする。
// これをdefineするとPosition::packe_sfen(),unpack_sfen()が使えるようになる。
// #define USE_SFEN_PACKER

// 置換表のprobeに必ず失敗する設定
// 自己生成棋譜からの学習でqsearch()のPVが欲しいときに
// 置換表にhitして枝刈りされたときにPVが得られないの悔しいので
// #define USE_FALSE_PROBE_IN_TT

// 評価関数パラメーターを共有メモリを用いて他プロセスのものと共有する。
// 少ないメモリのマシンで思考エンジンを何十個も立ち上げようとしたときにメモリ不足になるので
// 評価関数をshared memoryを用いて他のプロセスと共有する機能。(対応しているのはいまのところKPPT評価関数のみ。かつWindows限定)
// #define USE_SHARED_MEMORY_IN_EVAL

// USIプロトコルでgameoverコマンドが送られてきたときに gameover_handler()を呼び出す。
// #define USE_GAMEOVER_HANDLER

// EVAL_HASHで使用するメモリとして大きなメモリを確保するか。
// これをONすると数%高速化する代わりに、メモリ使用量が1GBほど増える。
// #define USE_LARGE_EVAL_HASH

// GlobalOptionという、EVAL_HASHを有効/無効を切り替えたり、置換表の有効/無効を切り替えたりする
// オプションのための変数が使えるようになる。スピードが1%ぐらい遅くなるので大会用のビルドではオフを推奨。
// #define USE_GLOBAL_OPTIONS

// トーナメント(大会)用のビルド。最新CPU(いまはAVX2)用でEVAL_HASH大きめ。EVAL_LEARN、TEST_CMD使用不可。ASSERTなし。GlobalOptionsなし。
// #define FOR_TOURNAMENT

// 棋譜の変換などを行なうツールセット。CSA,KIF,KIF2(KI2)形式などの入出力を担う。
// これをdefineすると、extra/kif_converter/ フォルダにある棋譜や指し手表現の変換を行なう関数群が使用できるようになる。
// #define USE_KIF_CONVERT_TOOLS

// 128GB超えの置換表を使いたいとき用。
// このシンボルを定義するとOptions["Hash"]として131072(=128*1024[MB]。すなわち128GB)超えの置換表が扱えるようになる。
// Stockfishのコミュニティではまだ議論中なのでデフォルトでオフにしておく。
// cf. 128 GB TT size limitation : https://github.com/official-stockfish/Stockfish/issues/1349
// #define USE_HUGE_HASH

// --------------------
// release configurations
// --------------------

// --- 通常の思考エンジンとして実行ファイルを公開するとき用の設定集

// やねうら王2018 with お多福ラボ
#if defined(YANEURAOU_2018_OTAFUKU_ENGINE) || defined(YANEURAOU_2018_OTAFUKU_ENGINE_KPPT) || defined(YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT) || defined(YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
#define ENGINE_NAME "YaneuraOu 2018 Otafuku"
#if defined(YANEURAOU_2018_OTAFUKU_ENGINE)
#define EVAL_KPPT
//#define EVAL_KPP_KKPT
#endif
#if defined(YANEURAOU_2018_OTAFUKU_ENGINE_KPPT)
#define EVAL_KPPT
#endif
#if defined(YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT)
#define EVAL_KPP_KKPT
#endif
#if defined(YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
#define EVAL_MATERIAL
#endif
#if !defined(YANEURAOU_2018_OTAFUKU_ENGINE)
#define YANEURAOU_2018_OTAFUKU_ENGINE
#endif

#if !defined(YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
#define USE_EVAL_HASH
#endif
#define USE_SEE
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1

// デバッグ絡み
//#define ASSERT_LV 3
//#define USE_DEBUG_ASSERT

#define ENABLE_TEST_CMD
// 学習絡みのオプション
#define USE_SFEN_PACKER
// 学習機能を有効にするオプション。
#if !defined(YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
#define EVAL_LEARN
#endif

// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#if !defined(YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
#define USE_SHARED_MEMORY_IN_EVAL
#endif

// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY

// GlobalOptionsは有効にしておく。
#define USE_GLOBAL_OPTIONS
#endif

// NNUE評価関数を積んだtanuki-エンジン
#if defined(YANEURAOU_2018_TNK_ENGINE)
#define ENGINE_NAME "YaneuraOu 2018 T.N.K."
#define EVAL_NNUE

#define USE_EVAL_HASH
#define USE_SEE
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1

// デバッグ絡み
//#define ASSERT_LV 3
//#define USE_DEBUG_ASSERT

#define ENABLE_TEST_CMD
// 学習絡みのオプション
#define USE_SFEN_PACKER
// 学習機能を有効にするオプション。
#define EVAL_LEARN

// 学習のためにOpenBLASを使う
// "../openblas/lib/libopenblas.dll.a"をlibとして追加すること。
//#define USE_BLAS


// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
//#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY

// GlobalOptionsは有効にしておく。
#define USE_GLOBAL_OPTIONS

// 探索部はYANEURAOU_2018_OTAFUKU_ENGINEを使う。
#define YANEURAOU_2018_OTAFUKU_ENGINE
#endif

// --- 詰将棋エンジンとして実行ファイルを公開するとき用の設定集

#if defined(MATE_ENGINE)
#define ENGINE_NAME "YaneuraOu mate solver"
#define KEEP_LAST_MOVE
#undef  MAX_PLY_NUM
#define MAX_PLY_NUM 2000
#define USE_SEE
#define USE_MATE_1PLY
#define EVAL_MATERIAL
//#define LONG_EFFECT_LIBRARY
#define USE_KEY_AFTER
#define ENABLE_TEST_CMD
#endif

// --- ユーザーの自作エンジンとして実行ファイルを公開するとき用の設定集

#if defined(USER_ENGINE)
#define ENGINE_NAME "YaneuraOu user engine"
#define EVAL_KPP
#endif

// --------------------
//   for tournament
// --------------------

// トーナメント(大会)用に、対局に不要なものをすべて削ぎ落とす。
#if defined(FOR_TOURNAMENT)
#undef ASSERT_LV
#undef EVAL_LEARN
#undef ENABLE_TEST_CMD
#define USE_LARGE_EVAL_HASH
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

	// 置換表のprobe()を有効化/無効化する。
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

// --- Dropbit

// USE_DROPBIT_IN_STATSがdefineされているときは、Moveの上位16bitに格納するPieceとして駒打ちは +32(PIECE_DROP)　にする。
#ifdef USE_DROPBIT_IN_STATS
#define PIECE_DROP 32
#else
#define PIECE_DROP 0
#endif

// --- lastMove

// KIF形式に変換するときにPositionクラスにその局面へ至る直前の指し手が保存されていないと
// "同"金のように出力できなくて困る。
#ifdef USE_KIF_CONVERT_TOOLS
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

#if defined(USE_AVX512)
#define TARGET_CPU "AVX-512"
#elif defined(USE_AVX2)
#define TARGET_CPU "AVX2"
#elif defined(USE_SSE42)
#define TARGET_CPU "SSE4.2"
#elif defined(USE_SSE41)
#define TARGET_CPU "SSE4.1"
#elif defined(USE_SSE2)
#define TARGET_CPU "SSE2"
#else
#define TARGET_CPU "noSSE"
#endif

// 上位のCPUをターゲットとするなら、その下位CPUの命令はすべて使えるはずなので…。

#ifdef USE_AVX512
#define USE_AVX2
#endif

#ifdef USE_AVX2
#define USE_SSE42
#endif

#ifdef USE_SSE42
#define USE_SSE41
#endif

#ifdef USE_SSE41
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
#define EVAL_TYPE_NAME "Material"
#elif defined(EVAL_KPPT)
#define EVAL_TYPE_NAME "KPPT"
#elif defined(EVAL_KPP_KKPT)
#define EVAL_TYPE_NAME "KPP_KKPT"
#elif defined(EVAL_NNUE)
#define EVAL_TYPE_NAME "NNUE"
#else
#define EVAL_TYPE_NAME ""
#endif

// -- do_move()のときに移動した駒の管理をして差分計算

// 1. 駒番号を管理しているpiece_listについて
//   これはPosition::eval_list()で取得可能。
// 2. 移動した駒の管理について
//   これは、Position::state()->dirtyPiece。
//   FV38だと最大で2個。
//   FV_VARだとn個(可変)。
// 3. FV38だとある駒番号が何の駒であるかが決まっている。(PieceNumber型)
// 4. FV_VARだとある駒番号が何の駒であるかは定まっていない。必要な駒だけがeval_list()に格納されている。
//    駒落ちの場合、BonaPieceZeroは使われない。(必要ない駒はeval_list()に格納されていないため。)
//    また、銀10枚のように特定の駒種の駒を増やすことにも対応できる。(EvalList::MAX_LENGTHを変更する必要はあるが。)
// 5. FV38かFV_VARかどちらかを選択しなければならない。
//    本来なら、そのどちらも用いないようにも出来ると良いのだが、ソースコードがぐちゃぐちゃになるのでそれはやらないことにした。
// 6. FV_VAR方式は、Position::do_move()ではpiece_listは更新されず、dirtyPieceの情報のみが更新される。
//    ゆえに、evaluate()ではdirtyPieceの情報に基づき、piece_listの更新もしなければならない。
//    →　DirtyPiece::do_update()などを見ること。
//    また、inv_piece()を用いるので、評価関数の初期化タイミングでEvalLearningTools::init_mir_inv_tables()を呼び出して
//    inv_piece()が使える状態にしておかなければならない。
// 7. FV_VAR方式のリファレンス実装として、EVAL_KPP_KKPT_FV_VARがあるので、そのソースコードを見ること。

// あらゆる局面でP(駒)の数が増えないFV38と呼ばれる形式の差分計算用。
#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_NNUE)
#define USE_FV38
#endif

// P(駒)の数が増えたり減ったりするタイプの差分計算用
// FV38とは異なり、可変長piece_list。
#if defined(EVAL_MATERIAL)
#define USE_FV_VAR
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

