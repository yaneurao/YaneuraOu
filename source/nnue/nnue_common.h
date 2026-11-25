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
#include "../extra/tools.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../misc.h"
#include "../types.h"

#if defined(USE_WASM_SIMD)
#include "wasm_simd.h"
#endif

// HACK: Use _mm256_loadu_si256() instead of _mm256_load_si256. Otherwise a binary
//       compiled with older g++ crashes because the output memory is not aligned
//       even though alignas is specified.

// HACK : _mm256_loadu_si256()を_mm256_load_si256の代わりに使え。さもなくば、
//       古いg++でコンパイルされた実行ファイルはクラッシュする。なぜなら、
//       alignasが指定されているのにalignされていないコードを生成しやがるからだ。

#if defined(USE_AVX2)
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) && !defined(__clang__)
#define _mm256_loadA_si256  _mm256_loadu_si256
#define _mm256_storeA_si256 _mm256_storeu_si256
#else
#define _mm256_loadA_si256  _mm256_load_si256
#define _mm256_storeA_si256 _mm256_store_si256
#endif
#endif

#if defined(USE_AVX512)
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) && !defined(__clang__)
#define _mm512_loadA_si512   _mm512_loadu_si512
#define _mm512_storeA_si512  _mm512_storeu_si512
#else
#define _mm512_loadA_si512   _mm512_load_si512
#define _mm512_storeA_si512  _mm512_store_si512
#endif
#endif

namespace YaneuraOu::Eval::NNUE {

  // Version of the evaluation file
  // 評価関数ファイルのバージョンを表す定数
  // 既存の将棋用nn.binとの互換性を保つため、StockfishのVersionの値には合わせない。
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
  constexpr u32 Version = 0x7AF32F17u;
#else
  constexpr u32 Version = 0x7AF32F16u;
#endif

  // LEB128圧縮データを識別するためのmagic string
  constexpr const char        Leb128MagicString[] = "COMPRESSED_LEB128";
  constexpr const usize Leb128MagicStringSize = sizeof(Leb128MagicString) - 1;

  // Constant used in evaluation value calculation
  // 評価値の計算で利用する定数
  extern int FV_SCALE; // default 16。水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

  constexpr int WeightScaleBits = 6;

  // Size of cache line (in bytes)
  // キャッシュラインのサイズ（バイト単位）
  constexpr usize CacheLineSize = 64;

  // SIMD width (in bytes)
  // SIMD幅（バイト単位）
  // Stockfishでは削除済みだが、将棋のFT差分更新ではSIMDチャンク幅として使用する。
  #if defined(USE_AVX2)
  constexpr usize kSimdWidth = 32;
  #elif defined(USE_SSE2)
  constexpr usize kSimdWidth = 16;
  #elif defined(USE_MMX)
  constexpr usize kSimdWidth = 8;

  #elif defined(USE_NEON)
  constexpr usize kSimdWidth = 16;
  #elif defined(USE_WASM_SIMD)
  constexpr usize kSimdWidth = 16;
  #endif
  constexpr usize MaxSimdWidth = 32;

  // Type of input feature after conversion
  // 変換後の入力特徴量の型
  using TransformedFeatureType = u8;

  // インデックスの型
  using IndexType = u32;

  // Round n up to be a multiple of base
  // n以上で最小のbaseの倍数を求める
  template <typename IntType>
  constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
  }

// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read if
// necessary to return a result with the byte ordering of the compiling machine.
// ストリームから整数をlittle-endianで読み込み、実行環境のバイト順で返す。
// little-endian環境では直接読み込み、それ以外では読み込み後にバイト順を変換する。
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

  // signed LEB128で圧縮された整数を読み込む
  template<typename IntType>
  inline void read_leb_128(std::istream& stream, IntType* out, usize count) {

	  // Check the presence of our LEB128 magic string
	  char leb128MagicString[Leb128MagicStringSize];
	  stream.read(leb128MagicString, Leb128MagicStringSize);
	  assert(strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) == 0);

	  static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");

	  const u32 BUF_SIZE = 4096;
	  u8        buf[BUF_SIZE];

	  auto bytes_left = read_little_endian<u32>(stream);

	  u32 buf_pos = BUF_SIZE;
	  for (usize i = 0; i < count; ++i)
	  {
		  IntType result = 0;
		  usize  shift = 0;
		  do
		  {
			  if (buf_pos == BUF_SIZE)
			  {
				  stream.read(reinterpret_cast<char*>(buf), std::min(bytes_left, BUF_SIZE));
				  buf_pos = 0;
			  }

			  u8 byte = buf[buf_pos++];
			  --bytes_left;
			  result |= (byte & 0x7f) << shift;
			  shift += 7;

			  if ((byte & 0x80) == 0)
			  {
				  out[i] = (sizeof(IntType) * 8 <= shift || (byte & 0x40) == 0)
					  ? result
					  : result | ~((1 << shift) - 1);
				  break;
			  }
		  } while (shift < sizeof(IntType) * 8);
	  }

	  assert(bytes_left == 0);
  }
} // namespace YaneuraOu::Eval::NNUE

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // #ifndef NNUE_COMMON_H_INCLUDED
