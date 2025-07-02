#pragma once

#include "../../config.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "nn_types.h"

namespace YaneuraOu {

#if defined(TRT_NN_FP16)
typedef unsigned char PType;
typedef __half DType;
#else
typedef unsigned char PType;
typedef float DType;
#endif

void unpack_features1(const int batch_size, PType* p1, DType* x1, cudaStream_t stream);
void unpack_features2(const int batch_size, PType* p2, DType* x2, cudaStream_t stream);

} // namespace YaneuraOu
