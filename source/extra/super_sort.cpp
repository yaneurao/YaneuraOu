#include "../config.h"
#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
#include "../types.h"
#include "../misc.h"

#include <memory>
#include <immintrin.h>

// TokusiNさん作のSuperSort
// std::sortの7倍速で動作する外部ソート
// https://github.com/TokusiN/SuperSort
// を改造しました。

// 参考にしたもの
//		https://twitter.com/yuk__to/status/1125456987976679424
//		Qhapaq(WCSC29): https://github.com/yaneurao/YaneuraOu/compare/master...yuk-to:qhapaq_2019_WCSC
//
//		https://twitter.com/toku51n/status/1137600797166821382
//      > int64のソーティングオーダーとdoubleのソーティングオーダーは正の数の時に一致して負の数の時に反転するので、
//      > 履かせる下駄を0x4000...にするか0xC000...にするかを変えるだけで昇順、降順を切り替えられます。
//
//		https://twitter.com/toku51n/status/1137602360287432704
//		> AVX2はint64のmin, max命令が存在しないので、ビットを使い切っている場合でも下駄を履かせてからdoubleでソートする方が恐らく速いです。
//		> avx512なら直接ソート可能ですが、avx512を使うことによる速度低下をレジスタのサイズと数によって補えるかどうかは不明・・・

namespace {
	void insertion_sort32(int64_t* begin, int64_t* end);
	void insertion_sort64(int64_t* begin, int64_t* end);

	void SuperSortDAligned(double* array, size_t num);
	void SuperSortD32(double* arr, double* dst = NULL);
	void SuperSortD48(double* arr, double* dst = NULL);
	void SuperSortD64(double* arr, double* dst = NULL);

	void MergeD(double* src1, size_t size1, double* src2, size_t size2, double* dst);
	void SuperSortRecD(double* src, double* dst, double* org, size_t num);
} // namespace


// MovePickerから呼び出す用
// startは32byteでalignされているものとする。
void partial_super_sort(ExtMove* start, ExtMove* end , int limit)
{
	size_t num_max = end - start;

	//auto mid = std::partition(start, end , [limit](const ExtMove& mov) { return mov.value >= limit; });

#if defined(IS_64BIT)
	// →　64bitの値とみなしてstd::partitionしたほうが速い。
	std::int64_t limit64 = static_cast<int64_t>(limit) << 32;
	auto mid = (ExtMove*)std::partition((int64_t*)start, (int64_t*)end, [limit64](const int64_t& mov) { return mov >= limit64; });
#else
	auto mid = std::partition(start, end, [limit](const ExtMove& mov) { return mov.value >= limit; });
#endif

	size_t num = mid - start; // これを16の倍数になるか、endMovesまでの個数で繰り上げる
	if (num < 8)
	{
		// 要素数 7以下は insertion_sortのほうが速い。
		// 要素数 8以上になると32要素のSuperSortのほうが速い。
		// (すべてCPU cacheに載っていることが前提なので、新たなバッファを用いる場合SuperSortもう少し不利かも)
		// SuperSortはAVX2を酷使するのでもう少し大きな値でないと使いたくはないが、
		// 32要素目まで全部ソートされるのでその利点はある。

#if defined(IS_64BIT)
		insertion_sort64((int64_t*)start, (int64_t*)start + num);
#else
		// AVX2かつ32bit。そんな環境、いまどきあるんか…。
		insertion_sort32(start, start + num);
#endif
		return;
	}

	if (num_max < 32)
		num = num_max;	// 全件ソートしちゃいなよ。
	else if (num < 32)
		num = 32;		// num_max >= 32なので後方に変なものをpaddingされると困るから32個目までソートしてしまう
	else
		// 要素数が16の倍数になるようにalign(paddingされると困るので)。ただしnum_maxを超えてはならない。
		num = std::min(Math::align(num,16) , num_max);

	// AVX2を用いて一気にソートする個数
	size_t alignedsize;
	if (num < 32)
		alignedsize = 32;
	else
		// 要素数が16の倍数になるようにalign
		alignedsize = Math::align(num, 16);
	
	// arrが32バイトでalignされているか
	// 前提条件として、それは満たしているものとする。
	bool isAligned = (((size_t)start) & 31) == 0;
	ASSERT_LV3(isAligned);

	size_t i;
	constexpr std::uint64_t ONE = 3;
	constexpr std::int64_t half_64bit = ONE << 62;

	// 下駄を履かせる。これにより降順ソートとみなせる。0xC000...を加算している。
	int64_t* startInt64 = reinterpret_cast<int64_t*>(start);
	for (i = 0; i < num; i++)
		startInt64[i] += half_64bit;

#if 0
	// →　安定ソートになったほうがいいかも。
	// 指し手生成で最初に歩の指し手を生成してて、同じスコアなら歩の指し手のほうが価値が高いことが多いという問題はあるのか。
	
	constexpr std::int32_t half_32bit = ONE << (62 - 32);
	for (i = 0; i < num; i++)
	{
		// start[i]の上位32bit
		int32_t& s = *((int32_t*)start + i * 2 + 1);
		s = (s * 1024) - (int32_t)i + half_32bit;
	}
#endif

	// 後ろにalignedsizeになるところまでpaddingしておく。指し手生成バッファはMAX_MOVES+αまで後ろを使っていいはずで…。
	// 最大値をpaddingしておく。(これは昇順の並び替えによって末尾にくるはず)
	double* startDouble = reinterpret_cast<double*>(start);
	for (/* i = num */ ; i < alignedsize; i++)
		startDouble[i] = std::numeric_limits<float>::max();

	if (alignedsize > 64)
		SuperSortDAligned(startDouble, alignedsize / 16);
	else if (alignedsize == 32)
		SuperSortD32(startDouble);
	else if (alignedsize == 48)
		SuperSortD48(startDouble);
	else
		SuperSortD64(startDouble);

	// 履かせた下駄を取る処理を入れていないが、まあええやろ…。このあと使わへんしな。
	//for (i = 0; i < num; i++)
	//	start[i] -= half_64bit;
}

namespace {

	void insertion_sort32(ExtMove* begin, ExtMove* end) {

		for (ExtMove* sortedEnd = begin, *p = begin + 1; p < end; ++p)
			{
				ExtMove tmp = *p, * q;
				*p = *++sortedEnd;
				for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
					* q = *(q - 1);
				*q = tmp;
			}
	}

	void insertion_sort64(int64_t* begin, int64_t* end) {

		// ExtMoveをint64_tとみなしてlittle-endianの仮定のもとソートしてしまう。
		for (int64_t* sortedEnd = begin, *p = begin + 1; p < end; ++p)
		{
			int64_t tmp = *p, * q;
			*p = *++sortedEnd;
			for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
				* q = *(q - 1);
			*q = tmp;
		}
	}

	// 以下、元のSuperSortD.cppのコードほぼそのまま

	// 比較器
	auto Comparator = [](__m256d& lo, __m256d& hi) {
		__m256d t;
		t = _mm256_min_pd(lo, hi);
		hi = _mm256_max_pd(lo, hi);
		lo = t;
	};
	auto Swap01 = [](__m256d& lo, __m256d& hi) {
		__m256d t;
		t = _mm256_shuffle_pd(lo, hi, 0);
		hi = _mm256_shuffle_pd(lo, hi, 15);
		lo = t;
	};
	auto Swap02 = [](__m256d& lo, __m256d& hi) {
		__m256d t;
		t = _mm256_permute2f128_pd(lo, hi, 0x20);
		hi = _mm256_permute2f128_pd(lo, hi, 0x31);
		lo = t;
	};

#define Merge1616() {\
	m4 = _mm256_permute4x64_pd(m4, 0x1B);\
	m5 = _mm256_permute4x64_pd(m5, 0x1B);\
	m6 = _mm256_permute4x64_pd(m6, 0x1B);\
	m7 = _mm256_permute4x64_pd(m7, 0x1B);\
	Comparator(m0, m7);\
	Comparator(m1, m6);\
	Comparator(m2, m5);\
	Comparator(m3, m4);\
	Comparator(m0, m2);\
	Comparator(m1, m3);\
	Comparator(m4, m6);\
	Comparator(m5, m7);\
	Comparator(m0, m1);\
	Comparator(m2, m3);\
	Comparator(m4, m5);\
	Comparator(m6, m7);\
	Swap02(m0, m2);\
	Swap02(m1, m3);\
	Swap02(m4, m6);\
	Swap02(m5, m7);\
	Swap01(m0, m1);\
	Swap01(m2, m3);\
	Swap01(m4, m5);\
	Swap01(m6, m7);\
	Comparator(m0, m2);\
	Comparator(m1, m3);\
	Comparator(m4, m6);\
	Comparator(m5, m7);\
	Comparator(m0, m1);\
	Comparator(m2, m3);\
	Comparator(m4, m5);\
	Comparator(m6, m7);\
	Swap02(m0, m2);\
	Swap02(m1, m3);\
	Swap02(m4, m6);\
	Swap02(m5, m7);\
	Swap01(m0, m1);\
	Swap01(m2, m3);\
	Swap01(m4, m5);\
	Swap01(m6, m7);\
}

	void SuperSortD32(double* arr, double* dst)
	{
		__m256d m0, m1, m2, m3, m4, m5, m6, m7 /* , ms, mt */;

		if (!dst)
		{
			dst = arr;
		}
		// 規定のメモリにロード
		m0 = _mm256_load_pd((arr + 0));
		m1 = _mm256_load_pd((arr + 4));
		m2 = _mm256_load_pd((arr + 8));
		m3 = _mm256_load_pd((arr + 12));
		m4 = _mm256_load_pd((arr + 16));
		m5 = _mm256_load_pd((arr + 20));
		m6 = _mm256_load_pd((arr + 24));
		m7 = _mm256_load_pd((arr + 28));
		// 4並列でバッチャー奇偶マージソートを実行
		Comparator(m0, m1);
		Comparator(m2, m3);
		Comparator(m4, m5);
		Comparator(m6, m7);
		Comparator(m0, m2);
		Comparator(m1, m3);
		Comparator(m4, m6);
		Comparator(m5, m7);
		Comparator(m1, m2);
		Comparator(m5, m6);
		Comparator(m0, m4);
		Comparator(m1, m5);
		Comparator(m2, m6);
		Comparator(m3, m7);
		Comparator(m2, m4);
		Comparator(m3, m5);
		Comparator(m1, m2);
		Comparator(m3, m4);
		Comparator(m5, m6);
		// 0と1、2と3をスワップ
		m4 = _mm256_permute4x64_pd(m4, 0xB1);
		m5 = _mm256_permute4x64_pd(m5, 0xB1);
		m6 = _mm256_permute4x64_pd(m6, 0xB1);
		m7 = _mm256_permute4x64_pd(m7, 0xB1);

		Comparator(m0, m7);
		Comparator(m1, m6);
		Comparator(m2, m5);
		Comparator(m3, m4);
		// m0の0とm7の1、m0の2とm7の3、・・・をスワップ
		Swap01(m0, m7);
		Swap01(m1, m6);
		Swap01(m2, m5);
		Swap01(m3, m4);

		// バイトニック列をソート
		auto SortBitnic = [&]() {
			Comparator(m0, m4);
			Comparator(m1, m5);
			Comparator(m2, m6);
			Comparator(m3, m7);
			Comparator(m0, m2);
			Comparator(m1, m3);
			Comparator(m4, m6);
			Comparator(m5, m7);
			Comparator(m0, m1);
			Comparator(m2, m3);
			Comparator(m4, m5);
			Comparator(m6, m7);
		};
		SortBitnic();
		m4 = _mm256_permute4x64_pd(m4, 0x1B);
		m5 = _mm256_permute4x64_pd(m5, 0x1B);
		m6 = _mm256_permute4x64_pd(m6, 0x1B);
		m7 = _mm256_permute4x64_pd(m7, 0x1B);
		Comparator(m0, m7);
		Comparator(m1, m6);
		Comparator(m2, m5);
		Comparator(m3, m4);
		Swap01(m0, m4);
		Swap01(m1, m5);
		Swap01(m2, m6);
		Swap01(m3, m7);
		Comparator(m0, m4);
		Comparator(m1, m5);
		Comparator(m2, m6);
		Comparator(m3, m7);
		Swap02(m0, m7);
		Swap02(m1, m6);
		Swap02(m2, m5);
		Swap02(m3, m4);
		SortBitnic();
		// ソート完了
		Swap02(m0, m2);
		Swap02(m1, m3);
		Swap02(m4, m6);
		Swap02(m5, m7);
		Swap01(m0, m1);
		Swap01(m2, m3);
		Swap01(m4, m5);
		Swap01(m6, m7);
		// メモリの詰め替え完了
		// ストア
		_mm256_store_pd((dst + 0), m0);
		_mm256_store_pd((dst + 4), m4);
		_mm256_store_pd((dst + 8), m2);
		_mm256_store_pd((dst + 12), m6);
		_mm256_store_pd((dst + 16), m1);
		_mm256_store_pd((dst + 20), m5);
		_mm256_store_pd((dst + 24), m3);
		_mm256_store_pd((dst + 28), m7);

	}
	void SuperSortD48(double* arr, double* dst)
	{
		if (!dst)
		{
			dst = arr;
		}
		SuperSortD32(arr);
		SuperSortD32(arr + 16, dst + 16);
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;


		m0 = _mm256_load_pd(arr + 0);
		m1 = _mm256_load_pd(arr + 4);
		m2 = _mm256_load_pd(arr + 8);
		m3 = _mm256_load_pd(arr + 12);
		m4 = _mm256_load_pd(dst + 16 + 0);
		m5 = _mm256_load_pd(dst + 16 + 4);
		m6 = _mm256_load_pd(dst + 16 + 8);
		m7 = _mm256_load_pd(dst + 16 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		_mm256_store_pd(dst + 16 + 0, m4);
		_mm256_store_pd(dst + 16 + 4, m5);
		_mm256_store_pd(dst + 16 + 8, m6);
		_mm256_store_pd(dst + 16 + 12, m7);
	}

	void SuperSortD64(double* arr, double* dst)
	{
		if (!dst)
		{
			dst = arr;
		}
		SuperSortD32(arr);
		SuperSortD32(arr + 32);
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;

		m0 = _mm256_load_pd(arr + 0);
		m1 = _mm256_load_pd(arr + 4);
		m2 = _mm256_load_pd(arr + 8);
		m3 = _mm256_load_pd(arr + 12);
		m4 = _mm256_load_pd(arr + 32 + 0);
		m5 = _mm256_load_pd(arr + 32 + 4);
		m6 = _mm256_load_pd(arr + 32 + 8);
		m7 = _mm256_load_pd(arr + 32 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		_mm256_store_pd(arr + 32 + 0, m4);
		_mm256_store_pd(arr + 32 + 4, m5);
		_mm256_store_pd(arr + 32 + 8, m6);
		_mm256_store_pd(arr + 32 + 12, m7);
		m0 = _mm256_load_pd(arr + 16 + 0);
		m1 = _mm256_load_pd(arr + 16 + 4);
		m2 = _mm256_load_pd(arr + 16 + 8);
		m3 = _mm256_load_pd(arr + 16 + 12);
		m4 = _mm256_load_pd(arr + 48 + 0);
		m5 = _mm256_load_pd(arr + 48 + 4);
		m6 = _mm256_load_pd(arr + 48 + 8);
		m7 = _mm256_load_pd(arr + 48 + 12);
		Merge1616();
		_mm256_store_pd(dst + 48 + 0, m4);
		_mm256_store_pd(dst + 48 + 4, m5);
		_mm256_store_pd(dst + 48 + 8, m6);
		_mm256_store_pd(dst + 48 + 12, m7);
		m4 = _mm256_load_pd(arr + 32 + 0);
		m5 = _mm256_load_pd(arr + 32 + 4);
		m6 = _mm256_load_pd(arr + 32 + 8);
		m7 = _mm256_load_pd(arr + 32 + 12);
		Merge1616();
		_mm256_store_pd(dst + 16 + 0, m0);
		_mm256_store_pd(dst + 16 + 4, m1);
		_mm256_store_pd(dst + 16 + 8, m2);
		_mm256_store_pd(dst + 16 + 12, m3);
		_mm256_store_pd(dst + 32 + 0, m4);
		_mm256_store_pd(dst + 32 + 4, m5);
		_mm256_store_pd(dst + 32 + 8, m6);
		_mm256_store_pd(dst + 32 + 12, m7);
	}

	void MergeD(double* src1, size_t size1, double* src2, size_t size2, double* dst)
	{
		size_t i, j;
		i = j = 1;
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;

		m0 = _mm256_load_pd(src1 + 0);
		m1 = _mm256_load_pd(src1 + 4);
		m2 = _mm256_load_pd(src1 + 8);
		m3 = _mm256_load_pd(src1 + 12);
		m4 = _mm256_load_pd(src2 + 0);
		m5 = _mm256_load_pd(src2 + 4);
		m6 = _mm256_load_pd(src2 + 8);
		m7 = _mm256_load_pd(src2 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		src1 += 16;
		src2 += 16;
		dst += 16;
		while (1)
		{
			if (src1[0] > src2[0])
			{
				m0 = _mm256_load_pd(src2 + 0);
				m1 = _mm256_load_pd(src2 + 4);
				m2 = _mm256_load_pd(src2 + 8);
				m3 = _mm256_load_pd(src2 + 12);
				src2 += 16;
				j++;
				Merge1616();
				_mm256_store_pd(dst + 0, m0);
				_mm256_store_pd(dst + 4, m1);
				_mm256_store_pd(dst + 8, m2);
				_mm256_store_pd(dst + 12, m3);
				dst += 16;
				if (j == size2)
				{
					while (i < size1)
					{
						m0 = _mm256_load_pd(src1 + 0);
						m1 = _mm256_load_pd(src1 + 4);
						m2 = _mm256_load_pd(src1 + 8);
						m3 = _mm256_load_pd(src1 + 12);
						src1 += 16;
						i++;
						Merge1616();
						_mm256_store_pd(dst + 0, m0);
						_mm256_store_pd(dst + 4, m1);
						_mm256_store_pd(dst + 8, m2);
						_mm256_store_pd(dst + 12, m3);
						dst += 16;
					}
					break;
				}
			}
			else
			{
				m0 = _mm256_load_pd(src1 + 0);
				m1 = _mm256_load_pd(src1 + 4);
				m2 = _mm256_load_pd(src1 + 8);
				m3 = _mm256_load_pd(src1 + 12);
				src1 += 16;
				i++;
				Merge1616();
				_mm256_store_pd(dst + 0, m0);
				_mm256_store_pd(dst + 4, m1);
				_mm256_store_pd(dst + 8, m2);
				_mm256_store_pd(dst + 12, m3);
				dst += 16;
				if (i == size1)
				{
					while (j < size2)
					{
						m0 = _mm256_load_pd(src2 + 0);
						m1 = _mm256_load_pd(src2 + 4);
						m2 = _mm256_load_pd(src2 + 8);
						m3 = _mm256_load_pd(src2 + 12);
						src2 += 16;
						j++;
						Merge1616();
						_mm256_store_pd(dst + 0, m0);
						_mm256_store_pd(dst + 4, m1);
						_mm256_store_pd(dst + 8, m2);
						_mm256_store_pd(dst + 12, m3);
						dst += 16;
					}
					break;
				}
			}
		}
		_mm256_store_pd(dst + 0, m4);
		_mm256_store_pd(dst + 4, m5);
		_mm256_store_pd(dst + 8, m6);
		_mm256_store_pd(dst + 12, m7);
	}
	void SuperSortRecD(double* src, double* dst, double* org, size_t num)
	{
		if (num > 4)
		{
			SuperSortRecD(dst, src, org, num / 2);
			SuperSortRecD(dst + num / 2 * 16, src + num / 2 * 16, org + num / 2 * 16, num - num / 2);
			MergeD(src, num / 2, src + num / 2 * 16, num - num / 2, dst);
		}
		else
		{
			if (num == 4)
			{
				SuperSortD64(org, dst);
			}
			else if (num == 3)
			{
				SuperSortD48(org, dst);
			}
			else
			{
				SuperSortD32(org, dst);
			}
		}
	}

	void SuperSortDAligned(double* array, size_t num)
	{
		// ここでsort用のbufferを確保してしまう。
		// SuperSortの元のコードとはこの点が異なる。

		double buf[MAX_MOVES+11];
		// MAX_MOVES以上の最小の16の倍数が608。32byteでalignするときのためにさらに3つ余分に確保。611

		double* aligned_buf = reinterpret_cast<double*>(Math::align((size_t)buf, 32)); // 32でalignされているものとする。

		SuperSortRecD(aligned_buf, array, array, num);
	}
}
// namespace

#endif // defined(USE_SUPER_SORT) && defined(USE_AVX2)

