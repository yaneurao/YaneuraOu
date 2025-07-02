#include "unpack.cuh"

namespace YaneuraOu {

#if defined(TRT_NN_FP16)
typedef __half DType;
typedef short FType;
constexpr FType ftype_one = 0x3c00;
#else
typedef float DType;
typedef int FType;
constexpr FType ftype_one = 0x3f800000;
#endif

constexpr int features1_size = 62;
constexpr int features2_size = 57;

__global__ void unpack_features1_kernel(char *p1, FType *x1, int max_tid) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= max_tid) return;
	int x1_offset = tid * 81;
#pragma unroll
	for (int i = 0; i < 81; ++i) {
		// p1[j / 8] >> (j % 8)で下位1bitに設定する値を持ってくる
		// 下位1bitのマスクを行い、符号を負にすることで1の場合1byteの全bitを1にする
		// (FP16) 0x3c00 と論理積を取ることでfloat16の1.0にする
		// (FP32) 0x3f800000 と論理積を取ることでfloatの1.0にする
		int j = x1_offset + i;
		x1[j] = (-(FType)((p1[j >> 3] >> (j & 7)) & 1)) & ftype_one;
	}
}

__global__ void unpack_features2_kernel(char *p2, FType *x2, int max_tid) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= max_tid) return;
	int x2_offset = tid * 81;
	FType v = (-(FType)((p2[tid >> 3] >> (tid & 7)) & 1)) & ftype_one;
#pragma unroll
	for (int i = 0; i < 81; ++i) {
		x2[x2_offset + i] = v;
	}
}

void unpack_features1(const int batch_size, PType* p1, DType* x1, cudaStream_t stream)
{
	unpack_features1_kernel<<<batch_size, features1_size, 0, stream>>>((char*)p1, (FType*)x1, batch_size * features1_size);
}

void unpack_features2(const int batch_size, PType* p2, DType* x2, cudaStream_t stream)
{
	unpack_features2_kernel<<<batch_size, features2_size, 0, stream>>>((char*)p2, (FType*)x2, batch_size * features2_size);
}

} // namespace YaneuraOu
