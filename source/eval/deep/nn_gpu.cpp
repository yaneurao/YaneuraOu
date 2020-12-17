#include "nn_gpu.h"

#if defined(YANEURAOU_ENGINE_DEEP)

//#include "dlshogi_types.h"

#if defined (ONNXRUNTIME)
	#include <dml_provider_factory.h>
	// →　ここでDirectML.h が見つからないとエラーが出るなら Windows SDKの最新版をインストールすること。
	//	   https://github.com/microsoft/DirectML/issues/1
#endif

using namespace std;
using namespace Tools;

namespace Eval::dlshogi
{
	// forwardに渡すメモリの確保
	void* NN::alloc(size_t size)
	{
		void* ptr;
#if defined (ONNXRUNTIME)
		ptr = (void*)new u8[size];
#else
		checkCudaErrors(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#endif

		return ptr;
	}

	// alloc()で確保したメモリの開放
	void NN::free(void*ptr)
	{

#if defined (ONNXRUNTIME)
		delete[] (u8*)ptr;
#else
		checkCudaErrors(cudaFreeHost(ptr));
#endif

	}
	
	// モデルファイル名を渡すとそれに応じたNN派生クラスをbuildして返してくれる。デザパタで言うところのbuilder。
	std::shared_ptr<NN> NN::build_nn(const std::string& model_path , int gpu_id)
	{
		shared_ptr<NN> nn;
#if defined (ONNXRUNTIME)
		nn = std::make_unique<NNOnnxRuntime>();
#else
		if (model_filename.find("onnx") != string::npos)
			nn = std::make_unique<NNTensorRT>();
		else if (model_path[gpu_id].find("wideresnet15") != string::npos)
			nn = std::make_unique<NNWideResnet15>();
		else if (model_path[gpu_id].find("fused_wideresnet10") != string::npos)
			nn = std::make_unique<NNFusedWideResnet10>();
		else if (model_path[gpu_id].find("senet10") != string::npos)
			nn = std::make_unique<NNSENet10>();
		else
			nn = std::make_unique<NNWideResnet10>();
#endif

		if (nn->load(model_path , gpu_id).is_not_ok())
		{
			sync_cout << "Error! : read error , model path = " << model_path << sync_endl;
		}

		return nn;
	}

	// ---- NNOnnxRuntime

	// モデルファイルの読み込み。
	Result NNOnnxRuntime::load(const std::string& model_filename , int gpu_id)
	{
		Ort::SessionOptions session_options;
		session_options.DisableMemPattern();
		session_options.SetExecutionMode(ORT_SEQUENTIAL);
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, gpu_id));

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


#endif // defined(YANEURAOU_ENGINE_DEEP)
