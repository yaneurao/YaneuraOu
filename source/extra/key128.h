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
using Key64 = uint64_t;

// 128bit版
struct alignas(16) Key128
{
#if defined (USE_SSE2)
	union {
		__m128i m;
		u64 p[2];
	};
#else
		u64 p[2];
#endif

	Key128() {}

	// kを下位64bitに格納する(上位64bitは0)
	Key128(const Key& k) { set(k, 0); }

#if defined (USE_SSE2)
	Key128(const Key128& bb) { _mm_store_si128(&this->m, bb.m); }
	Key128& operator = (const Key128& rhs) { _mm_store_si128(&this->m, rhs.m); return *this; }
#else
	// 定義不要？
#endif

	// 下位64bitをk0、上位64bitをk1にする。
	void set(Key k0, Key k1) {
#if defined(USE_SSE2)
	m = _mm_set_epi64x(k1,k0);
#else
	p[0] = k0; p[1] = k1;
#endif
	}

	// これ暗黙で変換するの便利だが、バグに気づかずに危ない意味があるな…。
	//operator Key() const { return extract64<0>(); }

	// _u64[n]を取り出す。SSE4の命令が使えるときはそれを使う。
	// n == 0なら下位64bit、n == 1なら上位64bitが取り出される。
	template <int n>
	u64 extract64() const
	{
		static_assert(n == 0 || n == 1, "");
	#if defined(USE_SSE41)
		return (u64)(_mm_extract_epi64(m, n));
	#else
		return p[n];
	#endif
	}

#if defined (USE_SSE2)
	Key128& operator += (const Key128& b1) { this->m = _mm_add_epi64(m, b1.m); return *this; }
	Key128& operator -= (const Key128& b1) { this->m = _mm_sub_epi64(m, b1.m); return *this; }
	Key128& operator ^= (const Key128& b1) { this->m = _mm_xor_si128(m, b1.m); return *this; }
#else
	Key128& operator += (const Key128& b1) { this->p[0] += b1.p[0]; this->p[1] += b1.p[1]; return *this; }
	Key128& operator -= (const Key128& b1) { this->p[0] -= b1.p[0]; this->p[1] -= b1.p[1]; return *this; }
	Key128& operator ^= (const Key128& b1) { this->p[0] ^= b1.p[0]; this->p[1] ^= b1.p[1]; return *this; }
#endif

	Key128& operator *= (int64_t i) { p[0] *= i; p[1] *= i; return *this; }
	bool operator == (const Key128& rhs) const {
#if defined (USE_SSE41)
		__m128i neq = _mm_xor_si128(this->m, rhs.m);
		return _mm_test_all_zeros(neq, neq) ? true : false;
#else
		return p[0]==rhs.p[0] && p[1]==rhs.p[1];
#endif
	}
	bool operator != (const Key128& rhs) const { return !(*this == rhs); }

	Key128 operator * (const int64_t i) const { return Key128(*this) *= i; }

	Key128 operator + (const Key128& rhs) const { return Key128(*this) += rhs; }
	Key128 operator ^ (const Key128& rhs) const { return Key128(*this) ^= rhs; }

	// sortなどで使うために比較演算子を定義しておく。
    bool operator < (const Key128& rhs) const {
        if (this->p[0] != rhs.p[0]) {
            return this->p[0] < rhs.p[0];
        }
        return this->p[1] < rhs.p[1];
    }
};

// std::unorded_map<Key128,string>みたいなのを使うときにoperator==とhash化が必要。

template <>
struct std::hash<Key128> {
	size_t operator()(const Key128& k) const {
		// 下位bit返すだけで良いのでは？
		return (size_t)(k.extract64<0>());
	}
};

static std::ostream& operator << (std::ostream& os, const Key128& k)
{
	// 上位bitから出力する。(数字ってそういうものであるから…)
	os <<  k.extract64<1>() << ":" << k.extract64<0>();
	return os;
}

// 256bit版
struct alignas(32) Key256
{
#if defined(USE_AVX2)
	union {
		__m256i m;
		u64 p[4];
	};
#else
		u64 p[4];
#endif

	Key256() {}
	Key256(const Key& k) { set(k, 0, 0, 0); }
#if defined(USE_AVX2)
	Key256(const Key256& bb) { _mm256_store_si256(&this->m, bb.m); }
	Key256& operator = (const Key256& rhs) { _mm256_store_si256(&this->m, rhs.m); return *this; }
#endif

	// 下位64bitから順にk0, k1, k2, k3に設定する。
	void set(Key k0, Key k1, Key k2, Key k3) {
#if defined(USE_AVX2)
	m = _mm256_set_epi64x(k3, k2, k1, k0);
#else
	p[0] = k0; p[1] = k1; p[2] = k2; p[3] = k3;
#endif
	}

	//operator Key() const { return extract64<0>(); }

	// p[n]を取り出す。SSE4の命令が使えるときはそれを使う。
	template <int n>
	u64 extract64() const
	{
		static_assert(n == 0 || n == 1 || n == 2 || n == 3 , "");
	#if defined(USE_AVX2) && defined(IS_64BIT)
		return (u64)(_mm256_extract_epi64(m, n));
		// ⇨ gcc/clangだと32bit環境で、この命令が定義されていなくてコンパイルエラーになる。
		//		コンパイラ側のバグっぽい。仕方ないので、この命令を使うのは64bit環境の時のみにする。
	#else
		return p[n];
	#endif
	}

#if defined(USE_AVX2)
	Key256& operator += (const Key256& b1) { this->m = _mm256_add_epi64(m, b1.m); return *this; }
	Key256& operator -= (const Key256& b1) { this->m = _mm256_sub_epi64(m, b1.m); return *this; }
	Key256& operator ^= (const Key256& b1) { this->m = _mm256_xor_si256(m, b1.m); return *this; }
#else
	Key256& operator += (const Key256& b1) { this->p[0] += b1.p[0]; this->p[1] += b1.p[1]; this->p[2] += b1.p[2]; this->p[3] += b1.p[3]; return *this; }
	Key256& operator -= (const Key256& b1) { this->p[0] -= b1.p[0]; this->p[1] -= b1.p[1]; this->p[2] -= b1.p[2]; this->p[3] -= b1.p[3]; return *this; }
	Key256& operator ^= (const Key256& b1) { this->p[0] ^= b1.p[0]; this->p[1] ^= b1.p[1]; this->p[2] ^= b1.p[2]; this->p[3] ^= b1.p[3]; return *this; }
#endif
	Key256& operator *= (int64_t i) { p[0] *= i; p[1] *= i; p[2] *= i; p[3] *= i; return *this; }
	bool operator == (const Key256& rhs) const {
#if defined(USE_AVX2)
		return (_mm256_testc_si256(_mm256_cmpeq_epi8(this->m, rhs.m), _mm256_set1_epi8(static_cast<char>(0xffu))) ? true : false);
#else
		return p[0]==rhs.p[0] && p[1]==rhs.p[1] && p[2]==rhs.p[2] && p[3]==rhs.p[3];
#endif
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
		return (size_t)(k.extract64<0>());
	}
};

static std::ostream& operator << (std::ostream& os, const Key256& k)
{
	// 上位bitから出力する。(数字ってそういうものであるから…)
	os <<  k.extract64<3>() << ":" << k.extract64<2>() <<  k.extract64<1>() << ":" << k.extract64<0>();
	return os;
}

// HASH_KEYをKeyに変換する。
static Key hash_key_to_key(const Key     key) { return key               ; }
static Key hash_key_to_key(const Key128& key) { return key.extract64<0>(); }
static Key hash_key_to_key(const Key256& key) { return key.extract64<0>(); }

#endif // _KEY128_H_
