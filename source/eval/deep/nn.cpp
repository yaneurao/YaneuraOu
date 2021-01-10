#include "nn.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#if defined (ONNXRUNTIME)
	#include "nn_onnx_runtime.h"
#elif defined (TENSOR_RT)
	#include <cuda_runtime.h> // cudaHostAlloc()
	#include "nn_tensorrt.h"
#endif

#include "../../misc.h"

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
#elif defined (TENSOR_RT)
		checkCudaErrors(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#endif
		return ptr;
	}

	// alloc()で確保したメモリの開放
	void NN::free(void*ptr)
	{

#if defined (ONNXRUNTIME)
		delete[] (u8*)ptr;
#elif defined (TENSOR_RT)
		checkCudaErrors(cudaFreeHost(ptr));
#endif
	}
	
	// 使用可能なデバイス数を取得する。
	int NN::get_device_count() {
#if defined(ONNXRUNTIME)
		return NNOnnxRuntime::get_device_count();
#elif defined(TENSOR_RT)
		return NNTensorRT::get_device_count();
#endif
	}

	// モデルファイル名を渡すとそれに応じたNN派生クラスをbuildして返してくれる。デザパタで言うところのbuilder。
	std::shared_ptr<NN> NN::build_nn(const std::string& model_path , int gpu_id , int batch_size)
	{
		std::shared_ptr<NN> nn;

#if defined (ONNXRUNTIME)

		nn = std::make_unique<NNOnnxRuntime>();

#elif defined (TENSOR_RT)

		nn = std::make_unique<NNTensorRT>();

		// ファイル名に応じて、他のフォーマットに対応させるはずだったが、
		// TensorRTの場合、モデルファイル側にその情報があるので
		// ここで振り分ける必要はなさげ。
#endif

		sync_cout << "info string Start loading the model file, path = " << model_path << ", gpu_id = " << gpu_id << ", batch_size = " << batch_size << sync_endl;
		if (!nn)
		{
			sync_cout << "Error! : unknown model type." << sync_endl;
			return nullptr;
		}

		if (nn->load(model_path , gpu_id , batch_size).is_not_ok())
		{
			sync_cout << "Error! : read error , model path = " << model_path << sync_endl;
			return nullptr;
		}
		sync_cout << "info string The model file has been loaded, path = " << model_path
			<< ", gpu_id = " << gpu_id
			<< ", batch_size = " << batch_size
			<< sync_endl;

		return nn;
	}

} // namespace Eval::dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP)
