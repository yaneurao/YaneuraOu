#include "wasm_simd.h"
#if defined(USE_WASM_SIMD)

namespace emscripten_wasm_simd {

template<int n, int m, int n_stride>
void affine(const int8_t A[m][n_stride], const uint8_t x[n], const int32_t b[m], int32_t y[m]) {
  //
  // Dot product of two SIMD vectors
  // SIMDベクトル2本の内積。
  //
  [[maybe_unused]] auto dot_i8x16 = [](__i8x16 a, __i8x16 b) -> __i32x4 {
    __i16x8 a_lo = wasm_i16x8_extend_low_i8x16(a);
    __i16x8 a_hi = wasm_i16x8_extend_high_i8x16(a);
    __i16x8 b_lo = wasm_i16x8_extend_low_i8x16(b);
    __i16x8 b_hi = wasm_i16x8_extend_high_i8x16(b);
    return wasm_i32x4_add(wasm_i32x4_dot_i16x8(a_lo, b_lo), wasm_i32x4_dot_i16x8(a_hi, b_hi));
  };

  //
  // Horizontal sum
  // 各ベクトルの隣り合う要素を加算し、4レーンにまとめる。
  //
  [[maybe_unused]] auto hadd = [&](__i32x4 x0, __i32x4 x1) -> __i32x4 {
    return wasm_i32x4_add(wasm_i32x4_shuffle(x0, x1, 0, 2, 4, 6), wasm_i32x4_shuffle(x0, x1, 1, 3, 5, 7));
  };

  // Combine four partial dot products into the four output lanes.
  // 4本の部分内積を水平加算し、4つの出力レーンにまとめる。
  [[maybe_unused]] auto haddx4 = [&](__i32x4 x0, __i32x4 x1, __i32x4 x2, __i32x4 x3) -> __i32x4 {
    return hadd(hadd(x0, x1), hadd(x2, x3));
  };

  //
  // Four dot products (using horizontal sums)
  // 水平加算を用いて4出力分の内積をまとめて計算する。
  //
  [[maybe_unused]] auto dot4 = [&](const int8_t* a, const uint8_t* x, const int32_t* b, int32_t* out) {
    __i32x4 z0 = wasm_i32x4_splat(0);
    __i32x4 z1 = wasm_i32x4_splat(0);
    __i32x4 z2 = wasm_i32x4_splat(0);
    __i32x4 z3 = wasm_i32x4_splat(0);
    for (int j = 0; j < n; j += 16) {
      __i8x16 xv = wasm_v128_load(&x[j]);
      z0 = wasm_i32x4_add(z0, dot_i8x16(wasm_v128_load(&a[j + 0 * n_stride]), xv));
      z1 = wasm_i32x4_add(z1, dot_i8x16(wasm_v128_load(&a[j + 1 * n_stride]), xv));
      z2 = wasm_i32x4_add(z2, dot_i8x16(wasm_v128_load(&a[j + 2 * n_stride]), xv));
      z3 = wasm_i32x4_add(z3, dot_i8x16(wasm_v128_load(&a[j + 3 * n_stride]), xv));
    }
    __i32x4 z = wasm_i32x4_add(wasm_v128_load(&b[0]), haddx4(z0, z1, z2, z3));
    wasm_v128_store(&out[0], z);
  };

  //
  // Affine
  // アフィン変換。
  //

  // 2048x8
  if constexpr (n % 16 == 0 && m % 4 == 0) {
    for (int i = 0; i < m; i += 4) {
      dot4(&A[i][0], &x[0], &b[i], &y[i]);
    }

  // 8x32, 32x1 (it doesn't worth optimizing such small kernels)
  } else {
    for (int i = 0; i < m; i++) {
      y[i] = b[i];
      for (int j = 0; j < n; j++) {
        y[i] += A[i][j] * x[j];
      }
    }
  }
}

// Explicit instantiation
// 明示的なインスタンス化。
template void affine< 512, 32,  512>(const int8_t A[32][ 512], const uint8_t x[ 512], const int32_t b[32], int32_t y[32]);
template void affine<  32, 32,   32>(const int8_t A[32][  32], const uint8_t x[  32], const int32_t b[32], int32_t y[32]);
template void affine<  32,  1,   32>(const int8_t A[ 1][  32], const uint8_t x[  32], const int32_t b[ 1], int32_t y[ 1]);

} // namespace emscripten_wasm_simd
#endif
