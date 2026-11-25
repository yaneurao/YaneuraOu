// Constants used in NNUE evaluation function
// NNUE評価関数で用いる定数など

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "../config.h"

#if defined(EVAL_NNUE) || defined(EVAL_SFNN)
#include "../misc.h"
#include "../types.h"

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_LASX)
    #include <lasxintrin.h>
    #include <lsxintrin.h>

#elif defined(USE_LSX)
    #include <lsxintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

#if defined(USE_WASM_SIMD)
#include "wasm_simd.h"
#endif

namespace YaneuraOu::Eval::NNUE {

using BiasType   = i16;
// StockfishのFT重み型はi8だが、将棋用nn.binはi16を使う。SPARSE8でもファイル形式は変えない。
using WeightType = i16;
using IndexType  = u32;

// Version of the evaluation file
// 評価関数ファイルのバージョン。既存の将棋用nn.binとの互換性を保つ。
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
constexpr u32 Version = 0x7AF32F17u;
#else
constexpr u32 Version = 0x7AF32F16u;
#endif

// Constant used in evaluation value calculation
// 評価値の計算で利用する定数。可変のFV_SCALEはnetwork.hで宣言する。
constexpr int WeightScaleBits = 6;

// Size of cache line (in bytes)
// キャッシュラインのサイズ（バイト単位）
constexpr usize CacheLineSize = 64;

constexpr usize MaxSimdWidth = 32;

// Type of input feature after conversion
// 変換後の入力特徴量の型
using TransformedFeatureType = u8;

// Round n up to be a multiple of base
// n以上で最小のbaseの倍数を求める。
template<typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
}


// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read if
// necessary to return a result with the byte ordering of the compiling machine.
// little-endianで整数を読み込み、実行環境のバイト順に変換して返す。
template<typename IntType>
inline IntType read_little_endian(std::istream& stream) {
    IntType result;

    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = 0;

        stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
        for (usize i = 0; i < sizeof(IntType); ++i)
            v = (v << 8) | u[sizeof(IntType) - i - 1];

        std::memcpy(&result, &v, sizeof(IntType));
    }

    return result;
}


// Utility to write an integer (signed or unsigned, any size)
// to a stream in little-endian order. We swap the byte order before the write if
// necessary to always write in little-endian order, independently of the byte
// ordering of the compiling machine.
// 整数をlittle-endianへ変換して書き込む。実行環境のバイト順には依存しない。
template<typename IntType>
inline void write_little_endian(std::ostream& stream, IntType value) {

    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(&value), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = value;

        usize i = 0;
        // if constexpr to silence the warning about shift by 8
        // 1byte型では8bitシフトを実体化せず、警告を避ける。
        if constexpr (sizeof(IntType) > 1)
        {
            for (; i + 1 < sizeof(IntType); ++i)
            {
                u[i] = u8(v);
                v >>= 8;
            }
        }
        u[i] = u8(v);

        stream.write(reinterpret_cast<char*>(u), sizeof(IntType));
    }
}


// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
// little-endianの整数をまとめて読み込み、配列outへ格納する。
template<typename IntType>
inline void read_little_endian(std::istream& stream, IntType* out, usize count) {
    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            out[i] = read_little_endian<IntType>(stream);
}


// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
// 配列valuesの整数をまとめてlittle-endianで書き込む。
template<typename IntType>
inline void write_little_endian(std::ostream& stream, const IntType* values, usize count) {
    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(values), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            write_little_endian<IntType>(stream, values[i]);
}

} // namespace YaneuraOu::Eval::NNUE
#endif // defined(EVAL_NNUE) || defined(EVAL_SFNN)

#endif // #ifndef NNUE_COMMON_H_INCLUDED
