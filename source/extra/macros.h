#ifndef _MACROS_H_
#define _MACROS_H_

// --------------------
//    マクロ集
// --------------------

// --- enum用のマクロ

// +,-,*など標準的なoperatorを標準的な方法で定義するためのマクロ
// enumで定義されている型に対して用いる。Stockfishのアイデア。

#define ENABLE_BASE_OPERATORS_ON(T)													\
	constexpr T operator+(const T d1, const T d2) { return T(int(d1) + int(d2)); }  \
	constexpr T operator-(const T d1, const T d2) { return T(int(d1) - int(d2)); }  \
	constexpr T operator-(const T d) { return T(-int(d)); }                         \
	inline T& operator+=(T& d1, const T d2) { return d1 = d1 + d2; }				\
	inline T& operator-=(T& d1, const T d2) { return d1 = d1 - d2; }				\

// インクリメント用
#define ENABLE_INCR_OPERATORS_ON(T)													\
inline T& operator++(T& d) { return d = T(int(d) + 1); }							\
inline T& operator--(T& d) { return d = T(int(d) - 1); }

#define ENABLE_FULL_OPERATORS_ON(T)													\
	ENABLE_BASE_OPERATORS_ON(T)														\
	constexpr T operator*(const int i, const T d) { return T(i * int(d)); }         \
	constexpr T operator*(const T d, const int i) { return T(int(d) * i); }         \
	inline T& operator*=(T& d, const int i) { return d = T(int(d) * i); }			\
	inline T& operator++(T& d) { return d = T(int(d) + 1); }						\
	inline T& operator--(T& d) { return d = T(int(d) - 1); }						\
	inline T operator++(T& d,int) { T prev = d; d = T(int(d) + 1); return prev; }	\
	inline T operator--(T& d,int) { T prev = d; d = T(int(d) - 1); return prev; }	\
	constexpr T operator/(T d, int i) { return T(int(d) / i); }                     \
	constexpr int operator/(T d1, T d2) { return int(d1) / int(d2); }               \
	inline T& operator/=(T& d, int i) { return d = T(int(d) / i); }

ENABLE_FULL_OPERATORS_ON(Color)

// StockfishではFileとRankはINCR_OPERATORだが、やねうら王では File同士の加算などができてほしいのでFULL_OPERATORに変うする。
ENABLE_FULL_OPERATORS_ON(File)
ENABLE_FULL_OPERATORS_ON(Rank)

ENABLE_FULL_OPERATORS_ON(Square)
ENABLE_FULL_OPERATORS_ON(SquareWithWall)
ENABLE_FULL_OPERATORS_ON(Piece)
ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_BASE_OPERATORS_ON(PieceType)
ENABLE_FULL_OPERATORS_ON(PieceNumber)
ENABLE_FULL_OPERATORS_ON(Value)
ENABLE_FULL_OPERATORS_ON(Hand)
ENABLE_FULL_OPERATORS_ON(Move)
ENABLE_FULL_OPERATORS_ON(Eval::BonaPiece)
ENABLE_FULL_OPERATORS_ON(Effect8::Direct)

// enumに対してint型との加算と減算を提供するマクロ。Value型など一部の型はこれがないと不便。(やねうら王独自拡張)

#define ENABLE_ADD_SUB_OPERATORS_ON(T)						\
constexpr T operator+(T v, int i) { return T(int(v) + i); } \
constexpr T operator-(T v, int i) { return T(int(v) - i); } \
inline T& operator+=(T& v, int i) { return v = v + i; }		\
inline T& operator-=(T& v, int i) { return v = v - i; }

ENABLE_ADD_SUB_OPERATORS_ON(Value)


// enumに対して標準的なビット演算を定義するマクロ(やねうら王独自拡張)
#define ENABLE_BIT_OPERATORS_ON(T)													\
  inline T operator&(const T d1, const T d2) { return T(int(d1) & int(d2)); }		\
  inline T& operator&=(T& d1, const T d2) { return d1 = T(int(d1) & int(d2)); }		\
  constexpr T operator|(const T d1, const T d2) { return T(int(d1) | int(d2)); }	\
  inline T& operator|=(T& d1, const T d2) { return d1 = T(int(d1) | int(d2)); }		\
  constexpr T operator^(const T d1, const T d2) { return T(int(d1) ^ int(d2)); }	\
  inline T& operator^=(T& d1, const T d2) { return d1 = T(int(d1) ^ int(d2)); }		\
  constexpr T operator~(const T d1) { return T(~int(d1)); }

#if defined(LONG_EFFECT_LIBRARY)
// LONG_EFFECT_LIBRARYでHandKind使ってる箇所がある。そのうち修正する。
ENABLE_FULL_OPERATORS_ON(HandKind)
ENABLE_BIT_OPERATORS_ON(HandKind)
#endif


// enumに対してrange forで回せるようにするためのhack(やねうら王独自拡張)
// (速度低下があるかも知れないので速度の要求されるところでは使わないこと)
#define ENABLE_RANGE_OPERATORS_ON(X,ZERO,NB)     \
  inline X operator*(X x) { return x; }          \
  inline X begin(X) { return ZERO; }             \
  inline X end(X) { return NB; }

ENABLE_RANGE_OPERATORS_ON(Square, SQ_ZERO, SQ_NB)
ENABLE_RANGE_OPERATORS_ON(Color, COLOR_ZERO, COLOR_NB)
ENABLE_RANGE_OPERATORS_ON(File, FILE_ZERO, FILE_NB)
ENABLE_RANGE_OPERATORS_ON(Rank, RANK_ZERO, RANK_NB)
ENABLE_RANGE_OPERATORS_ON(Piece, NO_PIECE, PIECE_NB)

// for(auto sq : Square())ではなく、for(auto sq : SQ) のように書くためのhack
#define SQ Square()
#define COLOR Color()
// FILE,RANKはfstreamでのdefineと被るので定義できない。
// PIECEはマクロ引数として使いたいので定義しない。

// 他のファイルでこの定義使いたいので、生かしておく。(やねうら王独自拡張)
//#undef ENABLE_FULL_OPERATORS_ON
//#undef ENABLE_INCR_OPERATORS_ON
//#undef ENABLE_BASE_OPERATORS_ON
//#undef ENABLE_ADD_SUB_OPERATORS_ON
//#undef ENABLE_BIT_OPERATORS_ON


// --- N回ループを展開するためのマクロ

// N 回ループを展開させる。t は lambda で書く。(Aperyのコードを参考にしています)
// 
// 使い方)
//   Unroller<5>()([&](const int i){std::cout << i << " ";});
// と書くと5回展開されて、
//   0 1 2 3 4
// と出力される。

template <int N> struct Unroller {
    template <typename T> FORCE_INLINE void operator () (T t) {
        Unroller<N-1>()(t);
        t(N-1);
    }
};
template <> struct Unroller<0> {
    template <typename T> FORCE_INLINE void operator () (T) {}
};

#endif
