#ifndef _KEY128_H_
#define _KEY128_H_

#include "../types.h"

// --------------------
//     拡張hash key
// --------------------

// 関数の引数に直接書くと(rand関数呼び出しの)評価順序が既定されていないので困る。C++11では左からのはずなのだがVC++2015ではそうなっていない。
#if HASH_KEY_BITS <= 64
#define SET_HASH(x,p0,p1,p2,p3) { x = (p0); auto dummy_func = [](u64,u64,u64){}; dummy_func(p1,p2,p3); }
#elif HASH_KEY_BITS <= 128
#define SET_HASH(x,p0,p1,p2,p3) { Key _K0=(p0); Key _K1=(p1); x.set(_K0, _K1); auto dummy_func = [](u64,u64){}; dummy_func(p2,p3); }
#else
#define SET_HASH(x,p0,p1,p2,p3) { Key _K0=(p0); Key _K1=(p1); Key _K2=(p2); Key _K3=(p3); x.set(_K0, _K1, _K2, _K3); }
#endif

// 置換表で用いるためにPositionクラスから得られるhash keyを64bit版の他に128,256bitに変更することが出来るのでそのための構造体。

// 64bit版
typedef uint64_t Key64;


// 実験用の機能なので、128bit,256bitのhash keyのサポートはAVX2のみ。
#if defined (USE_AVX2)

// 128bit版
struct alignas(16) Key128
{
	union {
		__m128i m;
		u64 _u64[2];
	};

	Key128() {}
	Key128(const Key128& bb) { _mm_store_si128(&this->m, bb.m); }
	Key128& operator = (const Key128& rhs) { _mm_store_si128(&this->m, rhs.m); return *this; }

	void set(Key k0, Key k1) { _u64[0] = k0; _u64[1] = k1; }
	operator Key() const { return _u64[0]; }

	uint64_t p(int i) const { return _u64[i]; }

	Key128& operator += (const Key128& b1) { this->m = _mm_add_epi64(m, b1.m); return *this; }
	Key128& operator -= (const Key128& b1) { this->m = _mm_sub_epi64(m, b1.m); return *this; }
	Key128& operator ^= (const Key128& b1) { this->m = _mm_xor_si128(m, b1.m); return *this; }

	Key128& operator *= (int64_t i) { _u64[0] *= i; _u64[1] *= i; return *this; }
	bool operator == (const Key128& rhs) const {
		return (_mm_testc_si128(_mm_cmpeq_epi8(this->m, rhs.m), _mm_set1_epi8(static_cast<char>(0xffu))) ? true : false);
	}
	bool operator != (const Key128& rhs) const { return !(*this == rhs); }

	Key128 operator * (const int64_t i) const { return Key128(*this) *= i; }

	Key128 operator + (const Key128& rhs) const { return Key128(*this) += rhs; }
	Key128 operator ^ (const Key128& rhs) const { return Key128(*this) ^= rhs; }

};

// 256bit版
struct alignas(32) Key256
{
	union {
		__m256i m;
		u64 _u64[4];
	};

	Key256() {}
	Key256(const Key256& bb) { _mm256_store_si256(&this->m, bb.m); }
	Key256& operator = (const Key256& rhs) { _mm256_store_si256(&this->m, rhs.m); return *this; }

	void set(Key k0, Key k1, Key k2, Key k3) { _u64[0] = k0; _u64[1] = k1; _u64[2] = k2; _u64[3] = k3; }
	operator Key() const { return _u64[0]; }

	uint64_t p(int i) const { return _u64[i]; }

	Key256& operator += (const Key256& b1) { this->m = _mm256_add_epi64(m, b1.m); return *this; }
	Key256& operator -= (const Key256& b1) { this->m = _mm256_sub_epi64(m, b1.m); return *this; }
	Key256& operator ^= (const Key256& b1) { this->m = _mm256_xor_si256(m, b1.m); return *this; }

	Key256& operator *= (int64_t i) { _u64[0] *= i; _u64[1] *= i; _u64[2] *= i; _u64[3] *= i; return *this; }
	bool operator == (const Key256& rhs) const {
		return (_mm256_testc_si256(_mm256_cmpeq_epi8(this->m, rhs.m), _mm256_set1_epi8(static_cast<char>(0xffu))) ? true : false);
	}
	bool operator != (const Key256& rhs) const { return !(*this == rhs); }

	Key256 operator * (const int64_t i) const { return Key256(*this) *= i; }

	Key256 operator + (const Key256& rhs) const { return Key256(*this) += rhs; }
	Key256 operator ^ (const Key256& rhs) const { return Key256(*this) ^= rhs; }

};

#endif

#endif // _KEY128_H_
