#ifndef _CONFIG_H_
#define _CONFIG_H_

// --------------------
// release configurations
// --------------------

// --- 通常の思考エンジンとして実行ファイルを公開するとき用の設定集

#ifdef YANEURAOU_NANO_ENGINE
#define ENGINE_NAME "YaneuraOu nano"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT_LIBRARY
#endif

#ifdef YANEURAOU_MINI_ENGINE
#define ENGINE_NAME "YaneuraOu mini"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT_LIBRARY
#endif

#ifdef YANEURAOU_CLASSIC_ENGINE
#define ENGINE_NAME "YaneuraOu classic"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#endif

#ifdef YANEURAOU_2016_ENGINE
#define ENGINE_NAME "YaneuraOu 2016"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#define LONG_EFFECT_LIBRARY
#endif

#ifdef RANDOM_PLAYER_ENGINE
#define ENGINE_NAME "YaneuraOu random player"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT_LIBRARY
#endif

#ifdef LOCAL_GAME_SERVER
#define ENGINE_NAME "YaneuraOu Local Game Server"
#undef ASSERT_LV
#define ASSERT_LV 3 // ローカルゲームサーバー、host側の速度はそれほど要求されないのでASSERT_LVを3にしておく。
#define KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT_LIBRARY
#endif

// --- 協力詰めエンジンとして実行ファイルを公開するとき用の設定集

#ifdef HELP_MATE_ENGINE
#define ENGINE_NAME "YaneuraOu help mate solver"
#undef ASSERT_LV
#define KEEP_LAST_MOVE
#undef  MAX_PLY_NUM
#define MAX_PLY_NUM 65000
#undef HASH_KEY_BITS
#define HASH_KEY_BITS 128
#undef MATE_1PLY
#undef LONG_EFFECT_LIBRARY
#endif

// --- 詰将棋エンジンとして実行ファイルを公開するとき用の設定集

#ifdef MATE_ENGINE
#define ENGINE_NAME "YaneuraOu mate solver"
//#undef ASSERT_LV
#define KEEP_LAST_MOVE
#undef  MAX_PLY_NUM
#define MAX_PLY_NUM 2000
#define MATE_1PLY
#define LONG_EFFECT_LIBRARY
#endif

// --- ユーザーの自作エンジンとして実行ファイルを公開するとき用の設定集

#ifdef USER_ENGINE
#define ENGINE_NAME "YaneuraOu user engine"
#endif

// --------------------
// include & configure
// --------------------

// --- includes

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <map>
#include <iostream>


// --- diable warnings

// うざいので無効化するwarning

// C4800 : 'unsigned int': ブール値を 'true' または 'false' に強制的に設定します
// →　static_cast<bool>(...)において出る。
#pragma warning(disable : 4800)


// --- assertion tools

// DEBUGビルドでないとassertが無効化されてしまうので無効化されないASSERT
// 故意にメモリアクセス違反を起こすコード。
#define ASSERT(X) { if (!(X)) *(int*)0 =0; }

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

// -- 評価関数の種類により、盤面の利きの更新ときの処理が異なる。(このタイミングで評価関数の差分計算をしたいので)

// 盤面上の利きを更新するときに呼び出したい関数。(評価関数の差分更新などのために差し替え可能にしておく。)

// color = 手番 , sq = 升 , e = 利きの加算量
#define ADD_BOARD_EFFECT(color_,sq_,e1_) { board_effect[color_].e[sq_] += (uint8_t)e1_; }
// e1 = color側の利きの加算量 , e2 = ~color側の利きの加算量
#define ADD_BOARD_EFFECT_BOTH(color_,sq_,e1_,e2_) { board_effect[color_].e[sq_] += (uint8_t)e1_; board_effect[~color_].e[sq_] += (uint8_t)e2_; }


#endif // _CONFIG_H_
