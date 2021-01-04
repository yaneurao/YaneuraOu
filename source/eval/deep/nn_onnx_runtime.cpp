#include "nn_onnx_runtime.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined(ONNXRUNTIME)

//#include "dlshogi_types.h"

#if defined(ORT_DML)
#include <dml_provider_factory.h>
// →　ここでDirectML.h が見つからないとエラーが出るなら Windows SDKの最新版をインストールすること。
//	   https://github.com/microsoft/DirectML/issues/1
#else
#include <cpu_provider_factory.h>
#endif
#include "../../usi.h"

using namespace std;
using namespace Tools;

namespace Eval::dlshogi
{
	// モデルファイルの読み込み。
	Result NNOnnxRuntime::load(const std::string& model_filename , int gpu_id , int batch_size)
	{
		Ort::SessionOptions session_options;
		session_options.DisableMemPattern();
		session_options.SetExecutionMode(ORT_SEQUENTIAL);
#if defined(ORT_MKL)
		session_options.SetInterOpNumThreads((int)Options["InterOpNumThreads"]);
		session_options.SetIntraOpNumThreads((int)Options["IntraOpNumThreads"]);
#endif
#if defined(ORT_DML)
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, gpu_id));
#else
	    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, true));
#endif
		// Windows環境ではwstringでファイル名を渡す必要があるようだが？
		std::wstring onnx_filename = MultiByteToWideChar(model_filename);
		//std::string onnx_filename(filename);

		session.reset(new Ort::Session(env, onnx_filename.c_str(), session_options));

		return ResultCode::Ok;
	}

	// NNによる推論
	void NNOnnxRuntime::forward(const int batch_size, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
	{
		// input

		std::array<int64_t, 4> input_shape1 { batch_size, (size_t)COLOR_NB * MAX_FEATURES1_NUM, 9, 9 };
		std::array<int64_t, 4> input_shape2 { batch_size, MAX_FEATURES2_NUM, 9, 9 };

		std::array<Ort::Value, 2> input_values{
			Ort::Value::CreateTensor<float>(memory_info, (float*)x1, batch_size * sizeof(NN_Input1), input_shape1.data(), input_shape1.size()),
			Ort::Value::CreateTensor<float>(memory_info, (float*)x2, batch_size * sizeof(NN_Input2), input_shape2.data(), input_shape2.size())
		};

		// output

		std::array<int64_t, 2> output_shape1{ batch_size, MAX_MOVE_LABEL_NUM * (size_t)SQ_NB };
		std::array<int64_t, 2> output_shape2{ batch_size, 1 };

		std::array<Ort::Value, 2> output_values{
			Ort::Value::CreateTensor<float>(memory_info, (float*)y1, batch_size * MAX_MOVE_LABEL_NUM * (size_t)SQ_NB, output_shape1.data(), output_shape1.size()),
			Ort::Value::CreateTensor<float>(memory_info, (float*)y2, batch_size, output_shape2.data(), output_shape2.size())
		};

		// names
		const char* input_names[] = { "input1", "input2" };
		const char* output_names[] = { "output_policy", "output_value" };

		// run
		session->Run(Ort::RunOptions{ nullptr }, input_names, input_values.data(), input_values.size(), output_names, output_values.data(), output_values.size());
	}

} // namespace Eval::dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP) && defined(ONNXRUNTIME)

