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
#undef LONG_EFFECT
#endif

#ifdef YANEURAOU_MINI_ENGINE
#define ENGINE_NAME "YaneuraOu mini"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT
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
#endif

#ifdef RANDOM_PLAYER_ENGINE
#define ENGINE_NAME "YaneuraOu random player"
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#undef LONG_EFFECT
#undef USE_EVAL_TABLE
#endif

#ifdef LOCAL_GAME_SERVER
#define ENGINE_NAME "YaneuraOu Local Game Server"
#undef ASSERT_LV
#define ASSERT_LV 3
#define KEEP_LAST_MOVE
#undef USE_EVAL_TABLE
#endif

// --- 協力詰めエンジンとして実行ファイルを公開するとき用の設定集

#ifdef HELP_MATE_ENGINE
#define ENGINE_NAME "YaneuraOu help mate solver"
#undef ASSERT_LV
#define KEEP_LAST_MOVE
#undef  MAX_PLY_
#define MAX_PLY_ 65000
#undef HASH_KEY_BITS
#define HASH_KEY_BITS 128
#undef USE_EVAL_TABLE
#undef MATE_1PLY
#undef LONG_EFFECT
#endif

// --- 詰将棋エンジンとして実行ファイルを公開するとき用の設定集
#ifdef MATE_ENGINE
#define ENGINE_NAME "YaneuraOu mate solver"
#undef ASSERT_LV
#define KEEP_LAST_MOVE
#undef  MAX_PLY_
#define MAX_PLY_ 2000
#undef USE_EVAL_TABLE
#define MATE_1PLY
#define LONG_EFFECT
#endif

// --- ユーザーの自作エンジンとして実行ファイルを公開するとき用の設定集

#ifdef USER_ENGINE
#define ENGINE_NAME "YaneuraOu user engine"
#endif

// --------------------
// include & configure
// --------------------

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <map>
#include <iostream>

// --- うざいので無効化するwarning
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

// --- switchにおいてdefaultに到達しないことを明示して高速化させる
#ifdef _DEBUG
// デバッグ時は普通にしとかないと変なアドレスにジャンプして原因究明に時間がかかる。
#define UNREACHABLE ASSERT_LV1(false);
#elif defined(_MSC_VER)
#define UNREACHABLE ASSERT_LV1(false); __assume(0);
#elif defined(__GNUC__)
#define UNREACHABLE __builtin_unreachable();
#else
#define UNREACHABLE ASSERT_LV1(false);
#endif

// PRETTY_JPが定義されているかどうかによって三項演算子などを使いたいので。
#ifdef PRETTY_JP
const bool pretty_jp = true;
#else
const bool pretty_jp = false;
#endif

#if HASH_KEY_BITS <= 64
#define HASH_KEY Key64
#elif HASH_KEY_BITS <= 128
#define HASH_KEY Key128
#else
#define HASH_KEY Key256
#endif

// ターゲットが64bitOSかどうか
#if defined(_WIN64) && defined(_MSC_VER)
const bool Is64Bit = true;
#else
const bool Is64Bit = false;
#endif

#endif // _CONFIG_H_

