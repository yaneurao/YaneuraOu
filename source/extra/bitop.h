#ifndef _BITOP_H_
#define _BITOP_H_

//
//   bit operation library
//

// ターゲット環境でSSE,AVX,AVX2が搭載されていない場合はこれらの命令をsoftware emulationにより実行する。
// software emulationなので多少遅いが、SSE,AVX,AVX2の使えない環境でそれに合わせたコードを書く労力が省ける。

// USE_SSE42   : SSE4.2以降で使える命令を使う
//                 POPCNT /
// USE_AVX2    : AVX2以降で使える命令を使う(Haswell以降からサポートされている)
//                 PEXT   / MOVEMASK

#ifdef USE_AVX2
const bool use_avx2 = true;

// for SSE,AVX,AVX2
#include <immintrin.h>

// for AVX2 : hardwareによるpext実装
#define PEXT32(a,b) _pext_u32(a,b)
#define PEXT64(a,b) _pext_u64(a,b)

// Byteboardの直列化で使うAVX2命令
struct ymm
{
  __m256i m;

  ymm(const void* p) :m(_mm256_load_si256((__m256i*)p)) {}
  ymm(const uint8_t t) { for (int i = 0; i < 32; ++i) m.m256i_u8[i] = t; }
  ymm() {}

  // MSBが1なら1にする
  uint32_t to_uint32() const { return _mm256_movemask_epi8(m); }

  ymm& operator |= (const ymm& b1) { m = _mm256_or_si256(m, b1.m); return *this; }
  ymm& operator &= (const ymm& b1) { m = _mm256_and_si256(m, b1.m); return *this; }
  ymm operator & (const ymm& rhs) const { return ymm(*this) &= rhs; }
  ymm operator | (const ymm& rhs) const { return ymm(*this) |= rhs; }

  // packed byte単位で比較してthisのほうが大きければMSBを1にする。(このあとto_uint32()で直列化するだとか)
  ymm cmp(const ymm& rhs) const { ymm t; t.m = _mm256_cmpgt_epi8(m, rhs.m); return t; }
};

// 24近傍で8近傍に利く長い利きの方向。
//static const ymm ymm_direct_around8 = ymm_zero;

#else
const bool use_avx2 = false;

// for non-AVX2 : software emulationによるpext実装(やや遅い。とりあえず動くというだけ。)
inline uint64_t pext(uint64_t val, uint64_t mask)
{
  uint64_t res = 0;
  for (uint64_t bb = 1; mask; bb += bb) {
    if ((int64_t)val & (int64_t)mask & -(int64_t)mask)
      res |= bb;
    // マスクを1bitずつ剥がしていく実装なので処理時間がbit長に依存しない。
    // ゆえに、32bit用のpextを別途用意する必要がない。
    mask &= mask - 1;
  }
  return res;
}

inline uint32_t PEXT32(uint32_t a, uint32_t b) { return (uint32_t)pext(a, b); }
inline uint64_t PEXT64(uint64_t a, uint64_t b) { return pext(a, b); }

struct ymm
{
  union {
    uint8_t m8[32];
    uint64_t m64[4];
  };

  ymm(const void* p) { const auto p64 = (uint64_t*)p;  m64[0] = p64[0]; m64[1] = p64[1]; m64[2] = p64[2]; m64[3] = p64[3]; }
  ymm(const uint8_t t) { for (int i = 0; i < 32; ++i) m8[i] = t; }
  ymm() {}

  // MSBを直列化する
  uint32_t to_uint32() const {
    uint32_t r = 0;
    for (int i = 0; i < 32; ++i)
      r |= uint32_t(m8[i] >> 7) << i;
    return r;
  }

  ymm& operator |= (const ymm& b1) {
    m64[0] = m64[0] | b1.m64[0];
    m64[1] = m64[1] | b1.m64[1];
    m64[2] = m64[2] | b1.m64[2];
    m64[3] = m64[3] | b1.m64[3];
    return *this;
  }
  ymm& operator &= (const ymm& b1) {
    m64[0] = m64[0] & b1.m64[0];
    m64[1] = m64[1] & b1.m64[1];
    m64[2] = m64[2] & b1.m64[2];
    m64[3] = m64[3] & b1.m64[3];
    return *this;
  }
  ymm operator & (const ymm& rhs) const { return ymm(*this) &= rhs; }
  ymm operator | (const ymm& rhs) const { return ymm(*this) |= rhs; }

  // packed byte単位で比較してthisのほうが大きければMSBを1にする。(このあとto_uint32()で直列化するだとか)
  ymm cmp(const ymm& rhs) const {
    ymm t;
    for (int i = 0; i < 32; ++i)
      t.m8[i]= (m8[i] > rhs.m8[i]) ? 0xff : 0;
    return t;
  }
};

#endif

#ifdef USE_SSE42
const bool use_sse42 = true;

// for SSE4.2
#include <intrin.h>
#define POPCNT8(a) __popcnt8(a)
#define POPCNT32(a) __popcnt32(a)
#define POPCNT64(a) __popcnt64(a)

#else
const bool use_sse42 = false;

// software emulationによるpopcnt(やや遅い)
inline int32_t POPCNT8(uint32_t a) {
  a = (a & UINT32_C(0x55)) + (a >> 1 & UINT32_C(0x55));
  a = (a & UINT32_C(0x33)) + (a >> 2 & UINT32_C(0x33));
  a = (a & UINT32_C(0x0f)) + (a >> 4 & UINT32_C(0x0f));
  return (int32_t)a;
}
inline int32_t POPCNT32(uint32_t a) {
  a = (a & UINT32_C(0x55555555)) + (a >> 1 & UINT32_C(0x55555555));
  a = (a & UINT32_C(0x33333333)) + (a >> 2 & UINT32_C(0x33333333));
  a = (a & UINT32_C(0x0f0f0f0f)) + (a >> 4 & UINT32_C(0x0f0f0f0f));
  a = (a & UINT32_C(0x00ff00ff)) + (a >> 8 & UINT32_C(0x00ff00ff));
  a = (a & UINT32_C(0x0000ffff)) + (a >> 16 & UINT32_C(0x0000ffff));
  return (int32_t)a;
}
inline int32_t POPCNT64(uint64_t a) {
  a = (a & UINT64_C(0x5555555555555555)) + (a >> 1 & UINT64_C(0x5555555555555555));
  a = (a & UINT64_C(0x3333333333333333)) + (a >> 2 & UINT64_C(0x3333333333333333));
  a = (a & UINT64_C(0x0f0f0f0f0f0f0f0f)) + (a >> 4 & UINT64_C(0x0f0f0f0f0f0f0f0f));
  a = (a & UINT64_C(0x00ff00ff00ff00ff)) + (a >> 8 & UINT64_C(0x00ff00ff00ff00ff));
  a = (a & UINT64_C(0x0000ffff0000ffff)) + (a >> 16 & UINT64_C(0x0000ffff0000ffff));
  return (int32_t)a + (int32_t)(a >> 32);
}
#endif

static const ymm ymm_zero = ymm(uint8_t(0));
static const ymm ymm_one = ymm(uint8_t(1));


#endif

