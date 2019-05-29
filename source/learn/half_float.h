﻿#ifndef __HALF_FLOAT_H__
#define __HALF_FLOAT_H__

// Half Float Library by yaneurao
// (16-bit float)

// 16bit型による浮動小数点演算
// コンパイラの生成するfloat型のコードがIEEE 754の形式であると仮定して、それを利用する。

#include "../types.h"

namespace HalfFloat
{
	// IEEE 754 float 32 format is :
	//   sign(1bit) + exponent(8bits) + fraction(23bits) = 32bits
	//
	// Our float16 format is :
	//   sign(1bit) + exponent(5bits) + fraction(10bits) = 16bits
	union float32_converter
	{
		s32 n;
		float f;
	};


	// 16-bit float
	struct float16
	{
		// --- constructors

		float16() {}
		float16(s16 n) { from_float((float)n);  }
		float16(s32 n) { from_float((float)n); }
		float16(float n) { from_float(n); }
		float16(double n) { from_float((float)n); }

		// build from a float
		void from_float(float f) { *this = to_float16(f); }

		// --- implicit converters

		operator s32() const { return (s32)to_float(*this); }
		operator float() const { return to_float(*this); }
		operator double() const { return double(to_float(*this)); }

		// --- operators

		float16 operator += (float16 rhs) { from_float(to_float(*this) + to_float(rhs)); return *this; }
		float16 operator -= (float16 rhs) { from_float(to_float(*this) - to_float(rhs)); return *this; }
		float16 operator *= (float16 rhs) { from_float(to_float(*this) * to_float(rhs)); return *this; }
		float16 operator /= (float16 rhs) { from_float(to_float(*this) / to_float(rhs)); return *this; }
		float16 operator + (float16 rhs) const { return float16(*this) += rhs; }
		float16 operator - (float16 rhs) const { return float16(*this) -= rhs; }
		float16 operator * (float16 rhs) const { return float16(*this) *= rhs; }
		float16 operator / (float16 rhs) const { return float16(*this) /= rhs; }
		float16 operator - () const { return float16(-to_float(*this)); }
		bool operator == (float16 rhs) const { return this->v_ == rhs.v_; }
		bool operator != (float16 rhs) const { return !(*this == rhs); }

		static void UnitTest() { unit_test(); }

	private:

		// --- entity

		u16 v_;

		// --- conversion between float and float16

		static float16 to_float16(float f)
		{
			float32_converter c;
			c.f = f;
			u32 n = c.n;

			// The sign bit is MSB in common.
			u16 sign_bit = (n >> 16) & 0x8000;

			// The exponent of IEEE 754's float 32 is biased +127 , so we change this bias into +15 and limited to 5-bit.
			u16 exponent = (((n >> 23) - 127 + 15) & 0x1f) << 10;

			// The fraction is limited to 10-bit.
			u16 fraction = (n >> (23-10)) & 0x3ff;

			float16 f_;
			f_.v_ = sign_bit | exponent | fraction;

			return f_;
		}

		static float to_float(float16 v)
		{
			u32 sign_bit = (v.v_ & 0x8000) << 16;
			u32 exponent = ((((v.v_ >> 10) & 0x1f) - 15 + 127) & 0xff) << 23;
			u32 fraction = (v.v_ & 0x3ff) << (23 - 10);

			float32_converter c;
			c.n = sign_bit | exponent | fraction;
			return c.f;
		}

		// unit testになってないが、一応計算が出来ることは確かめた。コードはあとでなおす(かも)。
		static void unit_test()
		{
			float16 a, b, c, d;
			a = 1;
			std::cout << (float)a << std::endl;
			b = -118.625;
			std::cout << (float)b << std::endl;
			c = 2.5;
			std::cout << (float)c << std::endl;
			d = a + c;
			std::cout << (float)d << std::endl;

			c *= 1.5;
			std::cout << (float)c << std::endl;

			b /= 3;
			std::cout << (float)b << std::endl;

			float f1 = 1.5;
			a += f1;
			std::cout << (float)a << std::endl;

			a += f1 * (float)a;
			std::cout << (float)a << std::endl;
		}

	};

}

#endif // __HALF_FLOAT_H__
