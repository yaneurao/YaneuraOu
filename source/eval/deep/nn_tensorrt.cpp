#include "nn_tensorrt.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined (TENSOR_RT)

#include <regex>
#include "unpack.cuh"
//#include "dlshogi_types.h"

namespace {
	void FatalError(const std::string& s) {
		std::cerr << s << "\nAborting...\n";
		cudaDeviceReset();
		Tools::exit();
	}

	class Logger : public nvinfer1::ILogger
	{
		const char* error_type(Severity severity)
		{
			switch (severity)
			{
			case Severity::kINTERNAL_ERROR: return "[F] ";
			case Severity::kERROR         : return "[E] ";
			case Severity::kWARNING       : return "[W] ";
			case Severity::kINFO          : return "[I] ";
			case Severity::kVERBOSE       : return "[V] ";
			default: ASSERT(false);         return "";
			}
		}
		void log(Severity severity, const char* msg) noexcept
		{
			if (severity == Severity::kINTERNAL_ERROR) {
				std::cerr << error_type(severity) << msg << std::endl;
			}
		}
	} gLogger;

	constexpr long long int operator"" _MiB(long long unsigned int val)
	{
		return val * (1 << 20);
	}
}

//inline void checkCUDNN(cudnnStatus_t status) {
//	if (status != CUDNN_STATUS_SUCCESS) {
//		std::stringstream _error;
//		_error << "CUDNN failure\nError: " << cudnnGetErrorString(status);
//		FatalError(_error.str());
//	}
//}

// CUDA APIの返し値のエラーチェックを行うヘルパ
// エラーが発生すれば、その旨を出力して終了する。
void checkCudaErrors(cudaError_t status) {
	if (status != 0) {
		sync_cout << "Error! : Cuda failure , Error = " << cudaGetErrorString(status) << sync_endl;
		FatalError(cudaGetErrorString(status));
	}
}

//inline void checkCublasErrors(cublasStatus_t status) {
//	if (status != 0) {
//		std::stringstream _error;
//		_error << "Cublas failure\nError code " << status;
//		FatalError(_error.str());
//	}
//}


using namespace std;
using namespace Tools;

namespace Eval::dlshogi
{
	// モデルファイルの読み込み。
	Tools::Result NNTensorRT::load(const std::string& model_path, int gpu_id, int max_batch_size)
	{
		this->gpu_id = gpu_id;
		this->max_batch_size = max_batch_size;

		// Create host and device buffers
		// host(GPU側)に同じだけメモリを確保しておいて、CPU側からそこに転送する。
		set_device(gpu_id);

		checkCudaErrors(cudaMalloc((void**)&p1_dev, sizeof(PType)            * ((max_batch_size * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB) + 7) >> 3)));
		checkCudaErrors(cudaMalloc((void**)&p2_dev, sizeof(PType)            * ((max_batch_size * ((int)MAX_FEATURES2_NUM) + 7) >> 3)));
		checkCudaErrors(cudaMalloc((void**)&x1_dev, sizeof(NN_Input1)        * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&x2_dev, sizeof(NN_Input2)        * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&y1_dev, sizeof(NN_Output_Policy) * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&y2_dev, sizeof(NN_Output_Value)  * max_batch_size));

		infer_inputBindings = { x1_dev, x2_dev, y1_dev, y2_dev };

		return load_model(model_path);
	}

	void NNTensorRT::release()
	{
		// load()でメモリ確保を行った場合、inputBindings.size() == 4のはず。
		if (infer_inputBindings.size())
		{
			// 安全のため、GPU IDをスレッドと関連付けてから開放する。
			// ※　これは本来しなくても良いと思うのだが、ドライバー側の実装次第では
			//     何か地雷を踏む可能性がなくはないので安全側に倒しておく。

			// メモリを確保した時のCUDAデバイスを設定する。
			set_device(gpu_id);

			// メモリの開放
			checkCudaErrors(cudaFree(p1_dev));
			checkCudaErrors(cudaFree(p2_dev));
			checkCudaErrors(cudaFree(x1_dev));
			checkCudaErrors(cudaFree(x2_dev));
			checkCudaErrors(cudaFree(y1_dev));
			checkCudaErrors(cudaFree(y2_dev));
			infer_inputBindings.resize(0);

		}
	}

	// 使用可能なデバイス数を取得する。
	int NNTensorRT::get_device_count() {
		int device_count = 0;
		checkCudaErrors(cudaGetDeviceCount(&device_count));
		return device_count;
	}

	// 現在のスレッドとGPUを紐付ける。
	// ※　CUDAの場合、cudaSetDevice()を呼び出す。必ず、そのスレッドの探索開始時(forward()まで)に一度はこれを呼び出さないといけない。
	void NNTensorRT::set_device(int gpu_id)
	{
		// 存在しないCUDAデバイスに設定しようとした場合、例えば↓のようなエラーを起こす。
		// Error! : Cuda failure , Error = invalid device ordinal
		checkCudaErrors(cudaSetDevice(gpu_id));
	}

	// 初回のみビルドが必要。
	void NNTensorRT::build(const std::string& onnx_filename)
	{
		auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
		if (!builder)
		{
			FatalError("createInferBuilder");
		}

		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		if (!network)
		{
			FatalError("createNetworkV2");
		}

		auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!config)
		{
			FatalError("createBuilderConfig");
		}

		auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
		if (!parser)
		{
			FatalError("createParser");
		}

		auto parsed = parser->parseFromFile(onnx_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
		if (!parsed)
		{
			FatalError("parseFromFile");
		}

		builder->setMaxBatchSize(max_batch_size);
		config->setMaxWorkspaceSize(64_MiB);

		// 教師局面なくてcalibration_cache用意できないのでコメントアウトしておく。(yane)
#if 0
		std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
		if (builder->platformHasFastInt8())
		{
			// キャリブレーションキャッシュがある場合のみINT8を使用
			std::string calibration_cache_filename = std::string(onnx_filename) + ".calibcache";
			std::ifstream calibcache(calibration_cache_filename);
			if (calibcache.is_open())
			{
				calibcache.close();

				config->setFlag(nvinfer1::BuilderFlag::kINT8);
				calibrator.reset(new Int8EntropyCalibrator2(onnx_filename.c_str(), 1));
				config->setInt8Calibrator(calibrator.get());
			}
			else if (builder->platformHasFastFp16())
			{
				config->setFlag(nvinfer1::BuilderFlag::kFP16);
			}
		}
		else
#endif
		if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}

#if defined(TRT_NN_FP16)
		network->getInput(0)->setType(nvinfer1::DataType::kHALF);
		network->getInput(1)->setType(nvinfer1::DataType::kHALF);
		network->getOutput(0)->setType(nvinfer1::DataType::kHALF);
		network->getOutput(1)->setType(nvinfer1::DataType::kHALF);
#endif

		ASSERT_LV3(network->getNbInputs() == 2);
		nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
		ASSERT_LV3(inputDims[0].nbDims == 4);
		ASSERT_LV3(inputDims[1].nbDims == 4);

		ASSERT_LV3(network->getNbOutputs() == 2);

		// Optimization Profiles
		auto profile = builder->createOptimizationProfile();
		const auto dims1 = inputDims[0].d;
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
		const auto dims2 = inputDims[1].d;
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
		config->addOptimizationProfile(profile);

		// TensorRT 8 より nvinfer1::IBuilder::buildSerializedNetwork() が追加され、 nvinfer1::IBuilder::buildEngineWithConfig() は非推奨となった。
		// nvinfer1::IBuilder::buildEngineWithConfig() は TensorRT 10.0 にて削除される見込み。
		// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/deprecated.html
		// https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-800-ea/release-notes/tensorrt-8.html#rel_8-0-0-EA
#if NV_TENSORRT_MAJOR >= 8
		auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
		if (!serializedEngine)
		{
			FatalError("buildSerializedNetwork");
		}
		auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
		infer_engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
		if (!infer_engine)
		{
			FatalError("deserializeCudaEngine");
		}
		// 一旦シリアライズ化されたエンジンはデシリアライズを行った上で捨てているが、
		// この後またすぐにファイル書き出し用にシリアライズを行っているので、手順改善の余地あり。
		// // auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
#else
		engine.reset(builder->buildEngineWithConfig(*network, *config));
		if (!engine)
		{
			FatalError("buildEngineWithConfig");
		}
#endif
	}

	Tools::Result NNTensorRT::load_model(const string& filename)
	{
		// 前に読み込んでいたものがあるなら、それを開放する。
		infer_engine.reset();

		// シリアライズされたファイルがあるなら、それを代わりに読み込む。

		// デバイス情報の取得
		//const int  cBufLen = 256;
		cudaDeviceProp device_prop;
		//char pciBusId[cBufLen];
		checkCudaErrors(cudaGetDeviceProperties(&device_prop, gpu_id));
		//checkCudaErrors(cudaDeviceGetPCIBusId(pciBusId, cBufLen, gpu_id));

		// ファイル名 + "." + (GPU_ID + "." +) DEVICE_NAME + "." + (PCI_BUS_ID + "." +) MAX_BATCH_SIZE + ".TRT" + NV_TENSORRT_VERSION + ".serialized"
		// GPU_ID は個体に固有・固定ではない。（構成変更時に限らず、リブートしたらIDが変わることもある）
		// 複数のCUDAデバイスが存在した時、全てのCUDAデバイスが同一とは限らない。

		//sync_cout << "info string gpu_id = " << gpu_id << ", device_name = " << device_prop.name << ", pci_bus_id = " << pciBusId << sync_endl;

		std::string serialized_filename =
			filename + "." +
			// std::to_string(gpu_id) + "." +
			std::regex_replace(std::string(device_prop.name), std::regex("[^A-Za-z0-9._-]"), std::string("_")) + "." +
			// std::regex_replace(std::string(pciBusId), std::regex("[^A-Za-z0-9._-]"), std::string("_")) + "." +
			std::to_string(max_batch_size) + "." +
			"TRT" + std::to_string(getInferLibVersion()) + "." +
#if defined(TRT_NN_FP16)
			"FP16." +
#endif
			"serialized";

		sync_cout << "info string serialized filename = " << serialized_filename << sync_endl;

		//std::ifstream seriarizedFile(serialized_filename, std::ios::binary);
		// →　遅いのでReadFileToMemory()を用いる。これで一発で読み込める。

		std::unique_ptr<u8[]> modelPtr;
		size_t modelSize = 0;
		auto result = SystemIO::ReadFileToMemory(serialized_filename, [&](size_t size) {
			modelPtr = make_unique<u8[]>(size); modelSize = size; return modelPtr.get();
		});

		if (result.is_ok())
		{
			auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
			infer_engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelPtr.get(), modelSize));

			// ドライバのバージョンが異なるなどが原因で、デシリアライズに失敗することがある。その場合はやりなおす。
			if (!infer_engine)
				sync_cout << "info string Warning! TensorRT : Failed to deserialize the model file. filename = " << serialized_filename << sync_endl;
		}

		// デシリアライズされたファイルがなかったか、デシリアライズに失敗している。
		if (!infer_engine)
		{
			// 初回のみビルドが必要。
			// シリアライズされたファイルを生成する。
			sync_cout << "info string TensorRT : build the model file." << sync_endl;

			build(filename);

			// serializing a model
			auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(infer_engine->serialize());
			if (!serializedEngine)
			{
				//throw std::runtime_error("Engine serialization failed");
				return Tools::ResultCode::SomeError;
			}
			std::ofstream engineFile(serialized_filename, std::ios::binary);
			if (!engineFile)
			{
				//throw std::runtime_error("Cannot open engine file");
				return Tools::ResultCode::FileOpenError;
			}
			engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
			if (engineFile.fail())
			{
				//throw std::runtime_error("Cannot open engine file");
				return Tools::ResultCode::FileOpenError;
			}
		}

		infer_context = InferUniquePtr<nvinfer1::IExecutionContext>(infer_engine->createExecutionContext());
		if (!infer_context)
		{
			//throw std::runtime_error("createExecutionContext");
				return Tools::ResultCode::FileWriteError;
		}

		inputDims1 = infer_engine->getBindingDimensions(0);
		inputDims2 = infer_engine->getBindingDimensions(1);

		return Tools::ResultCode::Ok;
	}

	void NNTensorRT::forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
	{
		inputDims1.d[0] = batch_size;
		inputDims2.d[0] = batch_size;
		infer_context->setBindingDimensions(0, inputDims1);
		infer_context->setBindingDimensions(1, inputDims2);
#if defined(UNPACK_CUDA)
		checkCudaErrors(cudaMemcpyAsync(p1_dev, p1, sizeof(PType) * ((batch_size * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB) + 7) >> 3), cudaMemcpyHostToDevice, cudaStreamPerThread));
		checkCudaErrors(cudaMemcpyAsync(p2_dev, p2, sizeof(PType) * ((batch_size * ((int)MAX_FEATURES2_NUM) + 7) >> 3), cudaMemcpyHostToDevice, cudaStreamPerThread));
		unpack_features1(batch_size, p1_dev, (DType*)x1_dev, cudaStreamPerThread);
		unpack_features2(batch_size, p2_dev, (DType*)x2_dev, cudaStreamPerThread);
#else
		checkCudaErrors(cudaMemcpyAsync(x1_dev, x1, sizeof(NN_Input1) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
		checkCudaErrors(cudaMemcpyAsync(x2_dev, x2, sizeof(NN_Input2) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
#endif
		const bool status = infer_context->enqueue(batch_size, infer_inputBindings.data(), cudaStreamPerThread, nullptr);
		ASSERT_LV3(status);
		checkCudaErrors(cudaMemcpyAsync(y1, y1_dev, sizeof(NN_Output_Policy) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
		checkCudaErrors(cudaMemcpyAsync(y2, y2_dev, sizeof(NN_Output_Value ) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
		checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
	}

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP) && defined (TENSOR_RT)

