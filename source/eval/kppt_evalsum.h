#ifndef _KPPT_EVAL_SUM_H_
#define _KPPT_EVAL_SUM_H_

#include "../shogi.h"
#include <array>

// KPPTで使うためのヘルパクラス

namespace Eval {

	// std::array<T,2>に対して 基本的な演算を提供する。
	template <typename Tl, typename Tr>
	FORCE_INLINE std::array<Tl, 2> operator += (std::array<Tl, 2>& lhs, const std::array<Tr, 2>& rhs) {
		lhs[0] += rhs[0];
		lhs[1] += rhs[1];
		return lhs;
	}
	template <typename Tl, typename Tr>
	FORCE_INLINE std::array<Tl, 2> operator -= (std::array<Tl, 2>& lhs, const std::array<Tr, 2>& rhs) {
		lhs[0] -= rhs[0];
		lhs[1] -= rhs[1];
		return lhs;
	}

	// 与えられたarrayが0ベクトルであるかどうかを判定する。
	template <typename T>
	bool is_zero(const std::array<T, 2> v)
	{
		return v[0] == T(0) && v[1] == T(0);
	}

	// デバッグ用出力
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::array<T, 2> v)
	{
		os << "{ " << v[0] << "," << v[1] << " } ";
		return os;
	}

	//
	// 手番つきの評価値を足していくときに使うclass
	//

	// EvalSum sum;
	// に対して
	// sum.p[0] = ΣBKKP
	// sum.p[1] = ΣWKPP
	// sum.p[2] = ΣKK
	// (それぞれに手番は加味されているものとする)
	// sum.sum() == ΣBKPP - ΣWKPP + ΣKK

	// EvalSumクラスは、コンストラクタでの初期化が保証できないので(オーバーヘッドがあるのでやりたくないので)
	// GCC 7.1.0以降で警告が出るのを回避できない。ゆえに、このクラスではこの警告を抑制する。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

	struct alignas(32) EvalSum {

#if defined(USE_AVX2)
		EvalSum(const EvalSum& es) {
		  _mm256_store_si256(&mm, es.mm);
		}
		EvalSum& operator = (const EvalSum& rhs) {
		  _mm256_store_si256(&mm, rhs.mm);
		  return *this;
		}
#elif defined(USE_SSE2)
		EvalSum(const EvalSum& es) {
		  _mm_store_si128(&m[0], es.m[0]);
		  _mm_store_si128(&m[1], es.m[1]);
		}
		EvalSum& operator = (const EvalSum& rhs) {
		  _mm_store_si128(&m[0], rhs.m[0]);
		  _mm_store_si128(&m[1], rhs.m[1]);
		  return *this;
		}
#endif

		EvalSum() {}

		// この局面の手番は c側にあるものとする。c側から見た評価値を返す。
		int32_t sum(const Color c) const
		{
			// NDF(2014)の手番評価の手法。
			// cf. http://www.computer-shogi.org/wcsc24/appeal/NineDayFever/NDF.txt

			// 手番に依存しない評価値合計
			// p[1][0]はΣWKPPなので符号はマイナス。
			const int32_t scoreBoard = p[0][0] - p[1][0] + p[2][0];
			// 手番に依存する評価値合計
			const int32_t scoreTurn  = p[0][1] + p[1][1] + p[2][1];

			// この関数は手番側から見た評価値を返すのでscoreTurnは必ずプラス

			return (c == BLACK ? scoreBoard : -scoreBoard) + scoreTurn;
		}

		EvalSum& operator += (const EvalSum& rhs)
		{
#if defined(USE_AVX2)
			mm = _mm256_add_epi32(mm, rhs.mm);
#elif defined(USE_SSE2)
			m[0] = _mm_add_epi32(m[0], rhs.m[0]);
			m[1] = _mm_add_epi32(m[1], rhs.m[1]);
#else
			p[0][0] += rhs.p[0][0];
			p[0][1] += rhs.p[0][1];
			p[1][0] += rhs.p[1][0];
			p[1][1] += rhs.p[1][1];
			p[2][0] += rhs.p[2][0];
			p[2][1] += rhs.p[2][1];
#endif
			return *this;
		}
		EvalSum& operator -= (const EvalSum& rhs)
		{
#if defined(USE_AVX2)
			mm = _mm256_sub_epi32(mm, rhs.mm);
#elif defined(USE_SSE2)
			m[0] = _mm_sub_epi32(m[0], rhs.m[0]);
			m[1] = _mm_sub_epi32(m[1], rhs.m[1]);
#else
			p[0][0] -= rhs.p[0][0];
			p[0][1] -= rhs.p[0][1];
			p[1][0] -= rhs.p[1][0];
			p[1][1] -= rhs.p[1][1];
			p[2][0] -= rhs.p[2][0];
			p[2][1] -= rhs.p[2][1];
#endif
			return *this;
		}
		EvalSum operator + (const EvalSum& rhs) const { return EvalSum(*this) += rhs; }
		EvalSum operator - (const EvalSum& rhs) const { return EvalSum(*this) -= rhs; }

		// evaluate hashでatomicに操作できる必要があるのでそのための操作子
		void encode()
		{
#if defined USE_AVX2
			// EvalSum は atomic にコピーされるので key が合っていればデータも合っている。
#else
			key ^= data[0] ^ data[1] ^ data[2];
#endif
	    }
		// decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
		void decode() { encode(); }

		// 評価値が計算済みであるかを判定する。
		// このEvalSumに値が入っていないときはp[0][0]をVALUE_NOT_EVALUATEDを設定することになっている。
		bool evaluated() const { return p[0][0] != VALUE_NOT_EVALUATED; }

		union {
			// array<.. , 3>でいいが、この下のstructに合わせてpaddingしておく。
			std::array<std::array<int32_t, 2>, 4> p;
      
			struct {
			u64 data[3];
			u64 key; // EVAL_HASH用。pos.key() >> 1 したもの。
			};
#if defined(USE_AVX2)
			__m256i mm;
			__m128i m[2];
#elif defined(USE_SSE2)
	      __m128i m[2];
#endif
			};
		};

	// 抑制していた警告を元に戻す。
#if defined (__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

	// 出力用　デバッグ用。
	static std::ostream& operator<<(std::ostream& os, const EvalSum& sum)
	{
		os << "sum BKPP = " << sum.p[0][0] << " + " << sum.p[0][1] << std::endl;
		os << "sum WKPP = " << sum.p[1][0] << " + " << sum.p[1][1] << std::endl;
		os << "sum KK   = " << sum.p[2][0] << " + " << sum.p[2][1] << std::endl;
		return os;
	}

#ifdef USE_EVAL_HASH
	// シンプルなHashTableの実装。
	// Sizeは2のべき乗。
	template <typename T, size_t Size>
	struct HashTable
	{
		HashTable() { clear(); }
		T* operator [] (const Key k) { return entries_ + (static_cast<size_t>(k) & (Size - 1)); }
		void clear() { memset(entries_, 0, sizeof(T)*Size); }

		// Size が 2のべき乗であることのチェック
		static_assert((Size & (Size - 1)) == 0, "");

	private:
		T entries_[Size];
	};

	// evaluateしたものを保存しておくHashTable(俗にいうehash)

#if !defined(USE_LARGE_EVAL_HASH)
	// 134MB(魔女のAVX2以外の時の設定)
	struct EvaluateHashTable : HashTable<EvalSum, 0x400000> {};
#else
	// prefetch有りなら大きいほうが良いのでは…。
	// →　あまり変わらないし、メモリもったいないのでデフォルトでは↑の設定で良いか…。
	// 1GB(魔女のAVX2の時の設定)
	struct EvaluateHashTable : HashTable<EvalSum, 0x2000000> {};
#endif

	extern EvaluateHashTable g_evalTable;
#endif

} // namespace Eval

#endif // EVAL_SUM_H
