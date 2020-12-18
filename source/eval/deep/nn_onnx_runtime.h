#ifndef __NN_ONNX_RUNTIME_H_INCLUDED__
#define __NN_ONNX_RUNTIME_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)
#if defined(ONNXRUNTIME)

// ONNX Runtimeを使う場合。
	// "docs/解説.txt"を確認すること。

	#include <onnxruntime_cxx_api.h>

#include "nn.h"
#include "nn_types.h"

namespace Eval::dlshogi
{
	// ONNXRUNTIME用
	class NNOnnxRuntime : public NN
	{
	public:
		// モデルファイルの読み込み。
		virtual Tools::Result load(const std::string& model_path , int gpu_id , int batch_size);

		// NNによる推論
		virtual void forward(const int batch_size, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2);

	private:
		Ort::Env env;
		std::unique_ptr<Ort::Session> session;
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	};

} // namespace Eval::dlshogi

#endif // defined(ONNXRUNTIME)
#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __NN_ONNX_RUNTIME_H_INCLUDED__

