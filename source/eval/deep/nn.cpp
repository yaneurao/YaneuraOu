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
	
	// モデルファイル名を渡すとそれに応じたNN派生クラスをbuildして返してくれる。デザパタで言うところのbuilder。
	std::shared_ptr<NN> NN::build_nn(const std::string& model_path , int gpu_id , int batch_size)
	{
		shared_ptr<NN> nn;
#if defined (ONNXRUNTIME)
		nn = std::make_unique<NNOnnxRuntime>();
#elif defined (TENSOR_RT)
		// フォルダ名に"onnx"と入ってると誤爆するのでそれを回避する必要がある。
		auto file_name = Path::GetFileName(model_path);
		if (file_name.find("onnx") != string::npos)
			nn = std::make_unique<NNTensorRT>();

		// あとのことは知らん。なんぞこれ。

		//else if (model_path[gpu_id].find("wideresnet15") != string::npos)
		//	nn = std::make_unique<NNWideResnet15>();
		//else if (model_path[gpu_id].find("fused_wideresnet10") != string::npos)
		//	nn = std::make_unique<NNFusedWideResnet10>();
		//else if (model_path[gpu_id].find("senet10") != string::npos)
		//	nn = std::make_unique<NNSENet10>();
		//else
		//	nn = std::make_unique<NNWideResnet10>();
#endif

		sync_cout << "info string Start loading the model file, path = " << model_path << sync_endl;
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
		sync_cout << "info string The model file has been loaded." << sync_endl;

		return nn;
	}

} // namespace Eval::dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP)
