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
ENABLE_FULL_OPERATORS_ON(File)
ENABLE_FULL_OPERATORS_ON(Rank)
ENABLE_FULL_OPERATORS_ON(Square)
ENABLE_FULL_OPERATORS_ON(SquareWithWall)
ENABLE_FULL_OPERATORS_ON(Piece)
ENABLE_FULL_OPERATORS_ON(PieceNumber)
ENABLE_FULL_OPERATORS_ON(Value)
ENABLE_FULL_OPERATORS_ON(Depth)
ENABLE_FULL_OPERATORS_ON(Hand)
ENABLE_FULL_OPERATORS_ON(HandKind)
ENABLE_FULL_OPERATORS_ON(Move)
ENABLE_FULL_OPERATORS_ON(Eval::BonaPiece)
ENABLE_FULL_OPERATORS_ON(Effect8::Direct)


// enumに対してint型との加算と減算を提供するマクロ。Value型など一部の型はこれがないと不便。

#define ENABLE_ADD_SUB_OPERATORS_ON(T)						\
constexpr T operator+(T v, int i) { return T(int(v) + i); } \
constexpr T operator-(T v, int i) { return T(int(v) - i); } \
inline T& operator+=(T& v, int i) { return v = v + i; }		\
inline T& operator-=(T& v, int i) { return v = v - i; }

ENABLE_ADD_SUB_OPERATORS_ON(Value)


// enumに対して標準的なビット演算を定義するマクロ
#define ENABLE_BIT_OPERATORS_ON(T)													\
  inline T operator&(const T d1, const T d2) { return T(int(d1) & int(d2)); }		\
  inline T& operator&=(T& d1, const T d2) { return d1 = T(int(d1) & int(d2)); }		\
  constexpr T operator|(const T d1, const T d2) { return T(int(d1) | int(d2)); }	\
  inline T& operator|=(T& d1, const T d2) { return d1 = T(int(d1) | int(d2)); }		\
  constexpr T operator^(const T d1, const T d2) { return T(int(d1) ^ int(d2)); }	\
  inline T& operator^=(T& d1, const T d2) { return d1 = T(int(d1) ^ int(d2)); }		\
  constexpr T operator~(const T d1) { return T(~int(d1)); }

ENABLE_BIT_OPERATORS_ON(HandKind)


// enumに対してrange forで回せるようにするためのhack(速度低下があるかも知れないので速度の要求されるところでは使わないこと)
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


// --- N回ループを展開するためのマクロ
// AperyのUnrollerのtemplateによる実装は模範的なコードなのだが、lambdaで書くと最適化されないケースがあったのでマクロで書く。

#define UNROLLER1(Statement_) { const int i = 0; Statement_; }
#define UNROLLER2(Statement_) { UNROLLER1(Statement_); const int i = 1; Statement_;}
#define UNROLLER3(Statement_) { UNROLLER2(Statement_); const int i = 2; Statement_;}
#define UNROLLER4(Statement_) { UNROLLER3(Statement_); const int i = 3; Statement_;}
#define UNROLLER5(Statement_) { UNROLLER4(Statement_); const int i = 4; Statement_;}
#define UNROLLER6(Statement_) { UNROLLER5(Statement_); const int i = 5; Statement_;}

// --- bitboardに対するforeach

// Bitboardのそれぞれの升に対して処理を行なうためのマクロ。
// p[0]側とp[1]側との両方で同じコードが生成されるので生成されるコードサイズに注意。
// BB_自体は破壊されない。(このあとemptyであることを仮定しているなら間違い)

#define FOREACH_BB(BB_, SQ_, Statement_)		\
	do {										\
		u64 p0_ = BB_.extract64<0>();			\
		while (p0_) {							\
			SQ_ = (Square)pop_lsb(p0_);			\
			Statement_;							\
		}										\
		u64 p1_ = BB_.extract64<1>();			\
		while (p1_) {							\
			SQ_ = (Square)(pop_lsb(p1_) + 63);	\
			Statement_;							\
		}										\
	} while (false)


#endif
