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

	// kを下位64bitに格納する(上位64bitは0)
	Key128(const Key& k) { set(k,0); }
	Key128(const Key128& bb) { _mm_store_si128(&this->m, bb.m); }
	Key128& operator = (const Key128& rhs) { _mm_store_si128(&this->m, rhs.m); return *this; }

	// 下位64bitをk0、上位64bitをk1にする。
	void set(Key k0, Key k1) {
#if defined(USE_SSE2)
	m = _mm_set_epi64x(k1,k0);
#else
	_u64[0] = k0; _u64[1] = k1;
#endif
	}

	operator Key() const { return _u64[0]; }

	uint64_t p(int i) const { return _u64[i]; }

	// _u64[n]を取り出す。SSE4の命令が使えるときはそれを使う。
	// n == 0なら下位64bit、n == 1なら上位64bitが取り出される。
	template <int n>
	u64 extract64() const
	{
		static_assert(n == 0 || n == 1, "");
	#if defined(USE_SSE41)
		return (u64)(_mm_extract_epi64(m, n));
	#else
		return _u64[n];
	#endif
	}

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

// std::unorded_map<Key128,string>みたいなのを使うときにoperator==とhash化が必要。

template <>
struct std::hash<Key128> {
	size_t operator()(const Key128& k) const {
		// 下位bit返すだけで良いのでは？
		return (size_t)(k._u64[0]);
	}
};

// 256bit版
struct alignas(32) Key256
{
	union {
		__m256i m;
		u64 _u64[4];
	};

	Key256() {}
	Key256(const Key& k) { set(k, 0, 0, 0); }
	Key256(const Key256& bb) { _mm256_store_si256(&this->m, bb.m); }
	Key256& operator = (const Key256& rhs) { _mm256_store_si256(&this->m, rhs.m); return *this; }

	// 下位64bitから順にk0, k1, k2, k3に設定する。
	void set(Key k0, Key k1, Key k2, Key k3) {
#if defined(USE_SSE2)
	m = _mm256_set_epi64x(k3,k2,k1,k0);
#else
	_u64[0] = k0; _u64[1] = k1; _u64[2] = k2; _u64[3] = k3;
#endif
	}

	operator Key() const { return _u64[0]; }

	uint64_t p(int i) const { return _u64[i]; }

	// p[n]を取り出す。SSE4の命令が使えるときはそれを使う。
	template <int n>
	u64 extract64() const
	{
		static_assert(n == 0 || n == 1 || n == 2 || n == 3 , "");
	#if defined(USE_SSE41)
		return (u64)(_mm256_extract_epi64(m, n));
	#else
		return _u64[n];
	#endif
	}

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

template <>
struct std::hash<Key256> {
	size_t operator()(const Key256& k) const {
		// 下位bit返すだけで良いのでは？
		return (size_t)(k._u64[0]);
	}
};

#endif

#endif // _KEY128_H_
