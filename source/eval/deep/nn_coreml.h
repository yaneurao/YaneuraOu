#ifndef __NN_COREML_H_INCLUDED__
#define __NN_COREML_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)
#if defined(COREML)

// Core MLを使う場合。

#include "nn.h"
#include "nn_types.h"

namespace Eval::dlshogi
{
	// Core ML用
	class NNCoreML : public NN
	{
	public:
		// モデルファイルの読み込み。
		virtual Tools::Result load(const std::string& model_path , int gpu_id , int batch_size);

		// NNによる推論
		virtual void forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2);

		// 使用可能なデバイス数を取得する。
		static int get_device_count();

	private:
		void* model; // Objective-Cの型を見せないため
		DType* input_buf;

	};

} // namespace Eval::dlshogi

#endif // defined(COREML)
#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __NN_COREML_H_INCLUDED__

