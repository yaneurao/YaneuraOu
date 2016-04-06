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


// ----------------------------
//      type define(uint)
// ----------------------------
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// ----------------------------
//      64bit environment
// ----------------------------

// 64bit環境のときにはIS_64BITがdefinedになるのでこのシンボルが定義されていなければ
// 32bit環境用のemulation codeを書く。
#if defined(_WIN64) && defined(_MSC_VER)
#define IS_64BIT
#endif

// ----------------------------
//      PEXT(AVX2の命令)
// ----------------------------

#ifdef USE_AVX2
const bool use_avx2 = true;

// for SSE,AVX,AVX2
#include <immintrin.h>

// for AVX2 : hardwareによるpext実装
#define PEXT32(a,b) _pext_u32(a,b)
#ifdef IS_64BIT
#define PEXT64(a,b) _pext_u64(a,b)
#else
// PEXT32を2回使った64bitのPEXTのemulation
#define PEXT64(a,b) ( uint64_t(PEXT32(uint32_t((a)>>32),uint32_t((b)>>32)) << POPCNT32(b)) | uint64_t(PEXT32(uint32_t(a),uint32_t(b))) )
#endif

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

#endif

// ----------------------------
//     POPCNT(SSE4.2の命令)
// ----------------------------

#ifdef USE_SSE42
const bool use_sse42 = true;

// for SSE4.2
#include <intrin.h>
#define POPCNT8(a) __popcnt8(a)
#ifdef IS_64BIT
#define POPCNT32(a) __popcnt32(a)
#define POPCNT64(a) __popcnt64(a)
#else
// 32bit版だと、何故かこれ関数名に"32"がついてない。
#define POPCNT32(a) __popcnt(uint32_t(a))
// 32bit環境では32bitのpop_count 2回でemulation。
#define POPCNT64(a) (POPCNT32((a)>>32) + POPCNT32(a))
#endif

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

// ----------------------------
//     BSF(bitscan forward)
// ----------------------------

#ifdef IS_64BIT
// 1である最下位のbitのbit位置を得る。0を渡してはならない。
inline int LSB32(uint32_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanForward(&index, v); return index; }
inline int LSB64(uint64_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanForward64(&index, v); return index; }

// 1である最上位のbitのbit位置を得る。0を渡してはならない。
inline int MSB32(uint32_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanReverse(&index, v); return index; }
inline int MSB64(uint64_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanReverse64(&index, v); return index; }
#else
// 32bit環境では64bit版を要求されたら2回に分けて実行。
inline int LSB32(uint32_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanForward(&index, v); return index; }
inline int LSB64(uint64_t v) { ASSERT_LV3(v != 0); return uint32_t(v) ? LSB32(uint32_t(v)) : 32 + LSB32(uint32_t(v >> 32)); }

inline int MSB32(uint32_t v) { ASSERT_LV3(v != 0); unsigned long index; _BitScanReverse(&index, v); return index; }
inline int MSB64(uint64_t v) { ASSERT_LV3(v != 0); return uint32_t(v >> 32) ? 32 + MSB32(uint32_t(v >> 32)) : MSB32(uint32_t(v)); }
#endif

// ----------------------------
//  ymm(256bit register class)
// ----------------------------

#ifdef USE_AVX2

// Byteboardの直列化で使うAVX2命令
struct alignas(32) ymm
{
  __m256i m;

  // アライメント揃っていないところからの読み込みに対応させるためにloadではなくloaduのほうを用いる。
  ymm(const void* p) :m(_mm256_loadu_si256((__m256i*)p)) {}
  ymm(const uint8_t t) { for (int i = 0; i < 32; ++i) m.m256i_u8[i] = t; }
  ymm() {}

  // MSBが1なら1にする
  uint32_t to_uint32() const { return _mm256_movemask_epi8(m); }

  ymm& operator |= (const ymm& b1) { m = _mm256_or_si256(m, b1.m); return *this; }
  ymm& operator &= (const ymm& b1) { m = _mm256_and_si256(m, b1.m); return *this; }
  ymm operator & (const ymm& rhs) const { return ymm(*this) &= rhs; }
  ymm operator | (const ymm& rhs) const { return ymm(*this) |= rhs; }

  // packed byte単位で符号つき比較してthisのほうが大きければMSBを1にする。(このあとto_uint32()で直列化するだとか)
  // ※　AVX2には符号つき比較する命令しかない…。
  ymm cmp(const ymm& rhs) const { ymm t; t.m = _mm256_cmpgt_epi8(m, rhs.m); return t; }

  // packed byte単位で比較して、等しいなら0xffにする。(このあとto_uint32()で直列化するだとか)
  // ※　AVXにはnot equal命令がない…。
  ymm eq(const ymm& rhs) const { ymm t; t.m = _mm256_cmpeq_epi8(m, rhs.m); return t; }
};

#else

struct ymm
{
  union {
    uint8_t m8[32];
    uint16_t m16[16];
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
      t.m8[i] = ((int8_t)m8[i] > (int8_t)rhs.m8[i]) ? 0xff : 0;
    return t;
  }

  // packed byte単位で比較して、等しいなら0xffにする。(このあとto_uint32()で直列化するだとか)
  ymm eq(const ymm& rhs) const {
    ymm t;
    for (int i = 0; i < 32; ++i)
      t.m8[i] = (m8[i] == rhs.m8[i]) ? 0xff : 0;
    return t;
  }
};

#endif

extern ymm ymm_zero;  // all packed bytes are 0.
extern ymm ymm_one;   // all packed bytes are 1.

// ----------------------------
//    custom allocator
// ----------------------------

// C++11では、std::stack<StateInfo>がalignasを無視するために、代わりにstack相当のものを自作。
template <typename T> struct aligned_stack {
  void push(const T& t) { auto ptr = (T*)_mm_malloc(sizeof(T), alignof(T)); *ptr = t; container.push_back(ptr); }
  T& top() const { return **container.rbegin(); }
  void clear() { for (auto ptr : container) _mm_free(ptr); container.clear(); }
  ~aligned_stack() { clear(); }
private:
  std::vector<T*> container;
};

// ----------------------------
//    pop_lsb
// ----------------------------

// 1である最下位bitを1bit取り出して、そのbit位置を返す。0を渡してはならない。
// sizeof(T)<=4 なら LSB32(b)で済むのだが、これをコンパイル時に評価させるの、どう書いていいのかわからん…。
// デフォルトでLSB32()を呼ぶようにしてuint64_tのときだけ64bit版を用意しておく。
template <typename T> int pop_lsb(T& b) {  int index = LSB32(b);  b = T(b & (b - 1)); return index; }
inline int pop_lsb(uint64_t & b) { int index = LSB64(b);  b &= b - 1; return index; }

#endif
