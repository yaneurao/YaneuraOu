// Constants used in NNUE evaluation function
// NNUE評価関数で用いる定数など

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <cstring>		// std::memcpy()

#include "../../config.h"

#if defined(EVAL_NNUE)

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

namespace Eval::NNUE {



  // Version of the evaluation file
  // 評価関数ファイルのバージョンを表す定数
  constexpr std::uint32_t kVersion = 0x7AF32F16u;

  // Constant used in evaluation value calculation
  // 評価値の計算で利用する定数
  constexpr int FV_SCALE = 16;
  constexpr int kWeightScaleBits = 6;

  // Size of cache line (in bytes)
  // キャッシュラインのサイズ（バイト単位）
  constexpr std::size_t kCacheLineSize = 64;

  // SIMD width (in bytes)
  // SIMD幅（バイト単位）
  #if defined(USE_AVX2)
  constexpr std::size_t kSimdWidth = 32;
  #elif defined(USE_SSE2)
  constexpr std::size_t kSimdWidth = 16;
  #elif defined(USE_MMX)
  constexpr std::size_t kSimdWidth = 8;

  #elif defined(USE_NEON)
  constexpr std::size_t kSimdWidth = 16;
  #endif
  constexpr std::size_t kMaxSimdWidth = 32;

  // unique number for each piece type on each square
  // 将棋では、BonaPieceで事足りるのでここでは定義しない。
  #if 0 // チェス用 for chess
  enum {
    PS_NONE     =  0,
    PS_W_PAWN   =  1,
    PS_B_PAWN   =  1 * SQUARE_NB + 1,
    PS_W_KNIGHT =  2 * SQUARE_NB + 1,
    PS_B_KNIGHT =  3 * SQUARE_NB + 1,
    PS_W_BISHOP =  4 * SQUARE_NB + 1,
    PS_B_BISHOP =  5 * SQUARE_NB + 1,
    PS_W_ROOK   =  6 * SQUARE_NB + 1,
    PS_B_ROOK   =  7 * SQUARE_NB + 1,
    PS_W_QUEEN  =  8 * SQUARE_NB + 1,
    PS_B_QUEEN  =  9 * SQUARE_NB + 1,
    PS_W_KING   = 10 * SQUARE_NB + 1,
    PS_END      = PS_W_KING, // pieces without kings (pawns included)
    PS_B_KING   = 11 * SQUARE_NB + 1,
    PS_END2     = 12 * SQUARE_NB + 1
  };
  extern const uint32_t kpp_board_index[PIECE_NB][COLOR_NB];

  #endif

  // Type of input feature after conversion
  // 変換後の入力特徴量の型
  using TransformedFeatureType = std::uint8_t;

  // インデックスの型
  using IndexType = std::uint32_t;

  // 学習用クラステンプレートの前方宣言
  template <typename Layer>
  class Trainer;

  // Round n up to be a multiple of base
  // n以上で最小のbaseの倍数を求める
  template <typename IntType>
  constexpr IntType CeilToMultiple(IntType n, IntType base) {
  return (n + base - 1) / base * base;
  }

  // read_little_endian() is our utility to read an integer (signed or unsigned, any size)
  // from a stream in little-endian order. We swap the byte order after the read if
  // necessary to return a result with the byte ordering of the compiling machine.
  template <typename IntType>
  inline IntType read_little_endian(std::istream& stream) {

      IntType result;
      std::uint8_t u[sizeof(IntType)];
      typename std::make_unsigned<IntType>::type v = 0;

      stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
      for (std::size_t i = 0; i < sizeof(IntType); ++i)
          v = (v << 8) | u[sizeof(IntType) - i - 1];

      std::memcpy(&result, &v, sizeof(IntType));
      return result;
  }
}  // namespace Eval::NNUE

#endif  // defined(EVAL_NNUE)

#endif // #ifndef NNUE_COMMON_H_INCLUDED
