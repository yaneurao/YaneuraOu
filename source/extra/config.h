#ifndef _CONFIG_H_
#define _CONFIG_H_

// 新しい置換表(次のバージョンで古いほうの置換表実装は消す)
#define NEW_TT

// --------------------
// コンパイル時設定
// --------------------

// --- ターゲットCPUの選択

// AVX2(Haswell以降)でサポートされた命令を使うか。
// このシンボルをdefineしなければ、pext命令をソフトウェアでエミュレートする。
// 古いCPUのPCで開発をしたていて、遅くてもいいからともかく動いて欲しいときにそうすると良い。

#define USE_AVX2

// SSE4.2以降でサポートされた命令を使うか。
// このシンボルをdefineしなければ、popcnt命令をソフトウェアでエミュレートする。
// 古いCPUのPCで開発をしたていて、遅くてもいいからともかく動いて欲しいときにそうすると良い。

#define USE_SSE42


// 通例hash keyは64bitだが、これを128にするとPosition::state()->long_key()から128bit hash keyが
// 得られるようになる。研究時に局面が厳密に合致しているかどうかを判定したいときなどに用いる。
// ※　やねうら王nanoではこの機能は削除する予定。
#define HASH_KEY_BITS 64
//#define HASH_KEY_BITS 128
//#define HASH_KEY_BITS 256


// 通常探索時の最大探索深さ
#define MAX_PLY_NUM 128

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

// #define EVAL_NO_USE   // 評価関数を用いないとき。
// #define EVAL_MATERIAL // 駒得のみの評価関数
// #define EVAL_PP       // ツツカナ型 2駒関係
// #define EVAL_KPP      // Bonanza型 3駒関係
// #define EVAL_PPE      // 技巧型 2駒+利き

// 長い利き(遠方駒の利き)のライブラリを用いるか。
// 超高速1手詰め判定などではこのライブラリが必要。
// do_move()のときに利きの差分更新を行なうので、do_move()は少し遅くなる。(その代わり、利きが使えるようになる)
//#define LONG_EFFECT_LIBRARY

// 超高速1手詰め判定ルーチンを用いるか。
//#define MATE_1PLY

// Position::see()を用いるか。これはSEE(Static Exchange Evaluation : 静的取り合い評価)の値を返す関数。
//#define USE_SEE

// PV(読み筋)を表示するときに置換表の指し手をかき集めてきて表示するか。
// 自前でPVを管理してRootMoves::pvを更新するなら、この機能を使う必要はない。
// これはPVの更新が不要なので実装が簡単だが、Ponderの指し手を返すためには
// PVが常に正常に更新されていないといけないので最近はこの方法は好まれない。
// #define USE_TT_PV

// 定跡を作るコマンド("makebook")を有効にする。
// #define ENABLE_MAKEBOOK_CMD

// 標準で用意されているMovePickerを用いるか
// #define USE_MOVE_PICKER


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
#endif

#ifdef YANEURAOU_NANO_PLUS_ENGINE
#define ENGINE_NAME "YaneuraOu nano plus"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_TT_PV
#define USE_SEE
#define USE_MOVE_PICKER
#define LONG_EFFECT_LIBRARY
#define MATE_1PLY
#endif

#ifdef YANEURAOU_MINI_ENGINE
#define ENGINE_NAME "YaneuraOu mini"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER
#define LONG_EFFECT_LIBRARY
#define MATE_1PLY
#endif

#ifdef YANEURAOU_CLASSIC_ENGINE
#define ENGINE_NAME "YaneuraOu classic"
// 開発中なのでassertを有効に。
#define ASSERT_LV 3
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER
#define LONG_EFFECT_LIBRARY
#define MATE_1PLY
#endif

#ifdef YANEURAOU_2016_ENGINE
#define ENGINE_NAME "YaneuraOu 2016"
#define LONG_EFFECT_LIBRARY
#undef USE_TT_PV
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
#define MATE_1PLY
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

// --------------------
//   diable warnings
// --------------------

// うざいので無効化するwarning

// for MSVC or Intel on Windows

#if defined(_MSC_VER)
// C4800 : 'unsigned int': ブール値を 'true' または 'false' に強制的に設定します
// →　static_cast<bool>(...)において出る。
#pragma warning(disable : 4800)
#endif

// for Clang
//#pragma clang diagnostic ignored "-Wunused-value"     // 未使用な変数に対する警告
//#pragma clang diagnostic ignored "-Wnull-dereference" // *(int*)0 = 0; のようにnullptrに対する参照に対する警告
//#pragma clang diagnostic ignored "-Wparentheses"      // while (m = mp.next()) {.. } みたいな副作用についての警告
//#pragma clang diagnostic ignored "-Wmicrosoft"        // 括弧のなかからの gotoでの警告

// for GCC
// かきかけ

// --------------------
//      configure
// --------------------

// --- assertion tools

// DEBUGビルドでないとassertが無効化されてしまうので無効化されないASSERT
// 故意にメモリアクセス違反を起こすコード。
#ifndef USE_DEBUG_ASSERT
#define ASSERT(X) { if (!(X)) *(int*)1 =0; }
#else
#define ASSERT(X) { if (!(X)) std::cout << "\nError : ASSERT(" << #X << ")\n"; }
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

// --------------------
//      configure
// --------------------

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


// --- 32-bit OS or 64-bit OS

// ターゲットが64bitOSかどうか
#if defined(_WIN64) && defined(_MSC_VER)
const bool Is64Bit = true;
#else
const bool Is64Bit = false;
#endif


// --- Long Effect Library

// 1手詰め判定は、LONG_EFFECT_LIBRARYに依存している。
#ifdef MATE_1PLY
#define LONG_EFFECT_LIBRARY
#endif


// --- evaluate function

// -- 評価関数の種類によりエンジン名に使用する文字列を変更する。
#if defined(EVAL_MATERIAL)
#define EVAL_TYPE_NAME "Material"
#elif defined(EVAL_PP)
#define EVAL_TYPE_NAME "PP"
#elif defined(EVAL_KPP)
#define EVAL_TYPE_NAME "KPP"
#elif defined(EVAL_PPE)
#define EVAL_TYPE_NAME "PPE"
#else
#define EVAL_TYPE_NAME ""
#endif

// PP,KPP,PPEならdo_move()のときに移動した駒の管理をして差分計算
// また、それらの評価関数は駒割りの計算(EVAL_MATERIAL)に依存するので、それをdefineしてやる。
#if defined(EVAL_PP) || defined(EVAL_KPP) || defined(EVAL_PPE)
#define USE_EVAL_DIFF
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
