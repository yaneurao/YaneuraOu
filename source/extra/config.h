#ifndef _CONFIG_H_
#define _CONFIG_H_

// --------------------
// コンパイル時設定
// --------------------

// --- ターゲットCPUの選択

#ifndef USE_MAKEFILE

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
// ※　やねうら王nanoではこの機能は削除する予定。
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

// #define EVAL_NO_USE    // 評価関数を用いないとき。
// #define EVAL_MATERIAL  // 駒得のみの評価関数
// #define EVAL_PP        // ツツカナ型 2駒関係(開発するかも)
// #define EVAL_KPP       // Bonanza型 3駒関係
// #define EVAL_KPPT      // Bonanza型 3駒関係、手番つき(Apery WCSC26相当)
// #define EVAL_KKPT      // KKP手番あり + KPP手番なし(Ponanza WCSC26相当)
// #define EVAL_PPET      // 技巧型 2駒+利き+手番(開発するかも/しないかも)
// #define EVAL_KKPPT     // KKPPT型 4駒関係ね手番つき(55将棋、56将棋で用いる)
// #define EVAL_PPAT      // 3駒 + Piece-Piece-and Pawn型

// KPPT評価関数の学習に使うときのモード
// #define EVAL_LEARN

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
// #define USE_TT_PV

// 定跡を作るコマンド("makebook")を有効にする。
// #define ENABLE_MAKEBOOK_CMD

// 標準で用意されているMovePickerを用いるか
// #define USE_MOVE_PICKER_2016Q2

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

// 探索スレッドごとにCounterMoveHistoryを持つ。
// #define PER_THREAD_COUNTERMOVEHISTORY

// 探索StackごとにHistoryを持つ。
// #define PER_STACK_HISTORY

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

// --------------------
// release configurations
// --------------------

// --- 通常の思考エンジンとして実行ファイルを公開するとき用の設定集

#ifdef YANEURAOU_NANO_ENGINE
#define ENGINE_NAME "YaneuraOu nano"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_TT_PV
#define KEEP_LAST_MOVE
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_NANO_PLUS_ENGINE
#define ENGINE_NAME "YaneuraOu nano plus"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_TT_PV
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_MINI_ENGINE
#define ENGINE_NAME "YaneuraOu mini"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_CLASSIC_ENGINE
#define ENGINE_NAME "YaneuraOu classic"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_CLASSIC_TCE_ENGINE
#define ENGINE_NAME "YaneuraOu classic-tce"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_2016_MID_ENGINE
#define ENGINE_NAME "YaneuraOu 2016 Mid"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q2
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define ENABLE_TEST_CMD
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
#endif

#ifdef YANEURAOU_2016_LATE_ENGINE
#define ENGINE_NAME "YaneuraOu 2016 Late"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q3
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define PER_THREAD_COUNTERMOVEHISTORY
#define PER_STACK_HISTORY

#define ENABLE_TEST_CMD
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY
#endif

#ifdef YANEURAOU_2017_EARLY_ENGINE
#define ENGINE_NAME "YaneuraOu 2017 Early"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2017Q2
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define PER_THREAD_COUNTERMOVEHISTORY
#define PER_STACK_HISTORY

// デバッグ絡み
//#define ASSERT_LV 3
//#define USE_DEBUG_ASSERT

#define ENABLE_TEST_CMD
// 学習絡みのオプション
#define USE_SFEN_PACKER
#define EVAL_LEARN
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY
#endif

#ifdef MUST_CAPTURE_SHOGI_ENGINE
#define ENGINE_NAME "YaneuraOu MustCaptureShogi"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q3
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define PER_THREAD_COUNTERMOVEHISTORY
#define PER_STACK_HISTORY

// デバッグ絡み
#define ASSERT_LV 3
#define USE_DEBUG_ASSERT

#define ENABLE_TEST_CMD
// 学習絡みのオプション
#define USE_SFEN_PACKER
#define EVAL_LEARN
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY
#endif


#ifdef RANDOM_PLAYER_ENGINE
#define ENGINE_NAME "YaneuraOu random player"
#define EVAL_NO_USE
#define ASSERT_LV 3
#endif

#ifdef LOCAL_GAME_SERVER
#define ENGINE_NAME "YaneuraOu Local Game Server"
#define EVAL_NO_USE
#define ASSERT_LV 3 // ローカルゲームサーバー、host側の速度はそれほど要求されないのでASSERT_LVを3にしておく。
#define KEEP_LAST_MOVE
#define EVAL_NO_USE
#define USE_ENTERING_KING_WIN
#endif

// --- 協力詰めエンジンとして実行ファイルを公開するとき用の設定集

#ifdef HELP_MATE_ENGINE
#define ENGINE_NAME "YaneuraOu help mate solver"
#define KEEP_LAST_MOVE
#undef  MAX_PLY_NUM
#define MAX_PLY_NUM 65000
#undef HASH_KEY_BITS
#define HASH_KEY_BITS 128
#define EVAL_NO_USE
#endif

// --- 詰将棋エンジンとして実行ファイルを公開するとき用の設定集

#ifdef MATE_ENGINE
#define ENGINE_NAME "YaneuraOu mate solver"
#define KEEP_LAST_MOVE
#undef  MAX_PLY_NUM
#define MAX_PLY_NUM 2000
#define USE_MATE_1PLY
#define EVAL_NO_USE
#define LONG_EFFECT_LIBRARY
#endif

// --- ユーザーの自作エンジンとして実行ファイルを公開するとき用の設定集

#ifdef USER_ENGINE
#define ENGINE_NAME "YaneuraOu user engine"
#define EVAL_KPP
#endif

// --------------------
//      include
// --------------------

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <map>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>   // このあとMutexをtypedefするので
#include <condition_variable>
#include <cstring>  // std::memcpy()
#include <cmath>    // log(),std::round()
#include <climits>  // INT_MAX
#include <ctime>    // std::ctime()
#include <random>   // random_device
#include <cstddef>  // offsetof

// --------------------
//   diable warnings
// --------------------

// うざいので無効化するwarning

// for MSVC or Intel on Windows

#if defined(_MSC_VER)
// C4800 : 'unsigned int': ブール値を 'true' または 'false' に強制的に設定します
// →　static_cast<bool>(...)において出る。
#pragma warning(disable : 4800)

// C4996 : 'ctime' : This function or variable may be unsafe.Consider using ctime_s instead.
#pragma warning(disable : 4996)
#endif

// C4102 : ラベルは 1 度も参照されません。
#pragma warning(disable : 4102)


// for GCC
#if defined(__GNUC__)
#endif

// for Clang
//#pragma clang diagnostic ignored "-Wunused-value"     // 未使用な変数に対する警告
//#pragma clang diagnostic ignored "-Wnull-dereference" // *(int*)0 = 0; のようにnullptrに対する参照に対する警告
//#pragma clang diagnostic ignored "-Wparentheses"      // while (m = mp.next()) {.. } みたいな副作用についての警告
//#pragma clang diagnostic ignored "-Wmicrosoft"        // 括弧のなかからの gotoでの警告


// --------------------
//      configure
// --------------------

// --- assertion tools

// DEBUGビルドでないとassertが無効化されてしまうので無効化されないASSERT
// 故意にメモリアクセス違反を起こすコード。
#ifndef USE_DEBUG_ASSERT
#define ASSERT(X) { if (!(X)) *(int*)1 =0; }
#else
#define ASSERT(X) { if (!(X)) { std::cout << "\nError : ASSERT(" << #X << ")\n"; *(int*)1 =0;} }
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

// --- for linux

#if !defined(_MSC_VER)
// stricmpはlinux系では存在しないらしく、置き換える。
#define _stricmp strcasecmp

// あと、getline()したときにテキストファイルが'\r\n'だと
// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
// そのため、fstreamに対してgetline()を呼び出すときは、
// std::getline()ではなく単にgetline()と書いて、この関数を使うべき。
inline bool getline(std::fstream& fs, std::string& s)
{
	bool b = (bool)std::getline(fs, s);
	if (s.size() && s[s.size() - 1] == '\r')
		s.erase(s.size() - 1);
	return b;
}

#endif

// --- output for Japanese notation

// PRETTY_JPが定義されているかどうかによって三項演算子などを使いたいので。
#ifdef PRETTY_JP
const bool pretty_jp = true;
#else
const bool pretty_jp = false;
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

// ----------------------------
//      CPU environment
// ----------------------------

// ターゲットが64bitOSかどうか
#if (defined(_WIN64) && defined(_MSC_VER)) || (defined(__GNUC__) && defined(__x86_64__))
const bool Is64Bit = true;
#define IS_64BIT
#else
const bool Is64Bit = false;
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

// ----------------------------
//     mutex wrapper
// ----------------------------

// std::mutexをもっと速い実装に差し替えたい時のためにwrapしておく。
typedef std::mutex Mutex;
typedef std::condition_variable ConditionVariable;


// ----------------------------
//     mkdir wrapper
// ----------------------------

#if defined(_MSC_VER)

// Windows用

#include <codecvt>	// mkdirするのにwstringが欲しいのでこれが必要

// フォルダを作成する。日本語は使っていないものとする。
// カレントフォルダ相対で指定する。
// 成功すれば0、失敗すれば非0が返る。
inline int MKDIR(std::string dir_name)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
	return _wmkdir(cv.from_bytes(dir_name).c_str());
}
#elif defined(_LINUX)
// linux環境において、この_LINUXというシンボルはmakefileにて定義されるものとする。

// Linux用のmkdir実装。
#include "sys/stat.h"

inline int MKDIR(std::string dir_name)
{
	return ::mkdir(dir_name.c_str(), 0777);
}

#else

// Linux環境かどうかを判定するためにはmakefileを分けないといけなくなってくるな..
// linuxでフォルダ掘る機能は、とりあえずナシでいいや..。評価関数ファイルの保存にしか使ってないし…。
inline int MKDIR(std::string dir_name)
{
	return 0;
}

#endif


// ----------------------------
//     evaluate function
// ----------------------------

// -- 評価関数の種類によりエンジン名に使用する文字列を変更する。
#if defined(EVAL_MATERIAL)
#define EVAL_TYPE_NAME "Material"
#elif defined(EVAL_PP)
#define EVAL_TYPE_NAME "PP"
#elif defined(EVAL_KPP)
#define EVAL_TYPE_NAME "KPP"
#elif defined(EVAL_PPE)
#define EVAL_TYPE_NAME "PPE"
#elif defined(EVAL_KPPT)
#define EVAL_TYPE_NAME "KPPT"
#else
#define EVAL_TYPE_NAME ""
#endif

// PP,KPP,KPPT,PPEならdo_move()のときに移動した駒の管理をして差分計算
// また、それらの評価関数は駒割りの計算(EVAL_MATERIAL)に依存するので、それをdefineしてやる。
#if defined(EVAL_PP) || defined(EVAL_KPP) || defined(EVAL_KPPT) || defined(EVAL_PPE)
#define USE_EVAL_DIFF
#endif

// AVX2を用いたKPPT評価関数は高速化できるので特別扱い。
// Skylake以降でないとほぼ効果がないが…。
#if defined(EVAL_KPPT) && defined(USE_AVX2)
#define USE_FAST_KPPT
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

#endif // _CONFIG_H_
