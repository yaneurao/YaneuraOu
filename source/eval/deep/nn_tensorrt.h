#ifndef __NN_TENSORRT_H_INCLUDED__
#define __NN_TENSORRT_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined(TENSOR_RT)

// TensorRT版

// Cudaのheader
#include <cuda_runtime.h>

// TensorRTのheader
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "nn.h"

// CUDA APIの返し値のエラーチェックを行うヘルパ
// エラーが発生すれば、その旨を出力して終了する。
void checkCudaErrors(cudaError_t status);

namespace Eval::dlshogi
{
	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			if (obj)
			{
				obj->destroy();
			}
		}
	};

	template <typename T>
	using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

	class NNTensorRT : public NN {
	public:
		virtual ~NNTensorRT() { release(); }

		// モデルファイルの読み込み。
		virtual Tools::Result load(const std::string& model_path , int gpu_id , int max_batch_size);

		// 推論
		virtual void forward(const int batch_size, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2);

	private:

		// host(GPU)側で確保されたbufferに対するポインタ
		NN_Input1* x1_dev;
		NN_Input2* x2_dev;
		NN_Output_Policy* y1_dev;
		NN_Output_Value * y2_dev;

		// x1_dev,x2_dev,y1_dev,y2_devをひとまとめにしたもの。
		std::vector<void*> inputBindings;

		// ↑で確保されたメモリを開放する。
		void release();

		// モデルファイルの読み込み。load()から呼び出される。内部的に用いる。
		Tools::Result load_model(const std::string& model_path);

		// load()が呼び出された時の値
		int gpu_id;
		int max_batch_size;

		// build,forwardで必要

		InferUniquePtr<nvinfer1::ICudaEngine> engine;
		InferUniquePtr<nvinfer1::IExecutionContext> context;
		nvinfer1::Dims inputDims1;
		nvinfer1::Dims inputDims2;

		// 初回のみ、ビルドが必要。シリアライズされたファイルを生成する。
		void build(const std::string& onnx_filename);
	};

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP) && defined(TENSOR_RT)
#endif // ndef __NN_TENSORRT_H_INCLUDED__
