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
		// TensorRT 7 までは protected だったデストラクタが、TensorRT 8 からは public となり、 destroy() メソッドの使用は非推奨となった。
		// destroy() メソッドは TensorRT 10.0 にて削除される見込み。
		//
		// Deprecated And Removed Features
		// The following features are deprecated in TensorRT 8.0.0:
		// - Interface functions that provided a destroy function are deprecated in TensorRT 8.0. The destructors will be exposed publicly in order for the delete operator to work as expected on these classes.
		// - Destructors for classes with destroy() methods were previously protected. They are now public, enabling use of smart pointers for these classes. The destroy() methods are deprecated.
		// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/deprecated.html
		// https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-800-ea/release-notes/tensorrt-8.html#rel_8-0-0-EA
		template <typename T>
		void operator()(T* obj) const
		{
			if (obj)
			{
#if NV_TENSORRT_MAJOR >= 8
				delete obj;
#else
				obj->destroy();
#endif
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
		virtual void forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2);

		// 使用可能なデバイス数を取得する。
		static int get_device_count();

		// 現在のスレッドとGPUを紐付ける。
		// ※　CUDAの場合、cudaSetDevice()を呼び出す。必ず、そのスレッドの探索開始時(forward()まで)に一度はこれを呼び出さないといけない。
		virtual void set_device(int gpu_id);

	private:

		// host(GPU)側で確保されたbufferに対するポインタ
		PType* p1_dev;
		PType* p2_dev;
		NN_Input1* x1_dev;
		NN_Input2* x2_dev;
		NN_Output_Policy* y1_dev;
		NN_Output_Value * y2_dev;

		// x1_dev,x2_dev,y1_dev,y2_devをひとまとめにしたもの。
		std::vector<void*> infer_inputBindings;

		// ↑で確保されたメモリを開放する。
		void release();

		// モデルファイルの読み込み。load()から呼び出される。内部的に用いる。
		Tools::Result load_model(const std::string& model_path);

		// load()が呼び出された時の値
		int gpu_id;
		int max_batch_size;

		// build,forwardで必要
		u32 unpack_size1;
		u32 unpack_size2;
		InferUniquePtr<nvinfer1::ICudaEngine> infer_engine;
		InferUniquePtr<nvinfer1::IExecutionContext> infer_context;
		nvinfer1::Dims inputDims1;
		nvinfer1::Dims inputDims2;

		// 初回のみ、ビルドが必要。シリアライズされたファイルを生成する。
		void build(const std::string& onnx_filename);
	};

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP) && defined(TENSOR_RT)
#endif // ndef __NN_TENSORRT_H_INCLUDED__
