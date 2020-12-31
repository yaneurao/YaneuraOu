#include "nn_tensorrt.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined (TENSOR_RT)

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
		void log(Severity severity, const char* msg)
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

		checkCudaErrors(cudaMalloc((void**)&x1_dev, sizeof(NN_Input1)        * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&x2_dev, sizeof(NN_Input2)        * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&y1_dev, sizeof(NN_Output_Policy) * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&y2_dev, sizeof(NN_Output_Value)  * max_batch_size));

		inputBindings = { x1_dev, x2_dev, y1_dev, y2_dev };

		return load_model(model_path);
	}

	void NNTensorRT::release()
	{
		// load()でメモリ確保を行った場合、inputBindings.size() == 4のはず。
		if (inputBindings.size())
		{
			checkCudaErrors(cudaFree(x1_dev));
			checkCudaErrors(cudaFree(x2_dev));
			checkCudaErrors(cudaFree(y1_dev));
			checkCudaErrors(cudaFree(y2_dev));
			inputBindings.resize(0);
		}
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

		engine.reset(builder->buildEngineWithConfig(*network, *config));
		if (!engine)
		{
			FatalError("buildEngineWithConfig");
		}
	}

	Tools::Result NNTensorRT::load_model(const string& filename)
	{
		// シリアライズされたファイルがあるなら、それを代わりに読み込む。

		// ファイル名 + "." + GPU_ID + "." + serialized"
		std::string serialized_filename = filename + "." + std::to_string(gpu_id) + "." + std::to_string(max_batch_size) + ".serialized";
		std::ifstream seriarizedFile(serialized_filename, std::ios::binary);

		if (seriarizedFile.is_open())
		{
			// deserializing a model
			seriarizedFile.seekg(0, std::ios_base::end);
			const size_t modelSize = seriarizedFile.tellg();
			seriarizedFile.seekg(0, std::ios_base::beg);
			std::unique_ptr<char[]> blob(new char[modelSize]);
			seriarizedFile.read(blob.get(), modelSize);
			auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
			engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(blob.get(), modelSize, nullptr));
		}
		else
		{
			// 初回のみビルドが必要。
			// シリアライズされたファイルを生成する。
			build(filename);

			// serializing a model
			auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
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

		context = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
		if (!context)
		{
			//throw std::runtime_error("createExecutionContext");
				return Tools::ResultCode::FileWriteError;
		}

		inputDims1 = engine->getBindingDimensions(0);
		inputDims2 = engine->getBindingDimensions(1);

		return Tools::ResultCode::Ok;
	}

	void NNTensorRT::forward(const int batch_size, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
	{
		inputDims1.d[0] = batch_size;
		inputDims2.d[0] = batch_size;
		context->setBindingDimensions(0, inputDims1);
		context->setBindingDimensions(1, inputDims2);

		checkCudaErrors(cudaMemcpy(x1_dev, x1, sizeof(NN_Input1) * batch_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(x2_dev, x2, sizeof(NN_Input2) * batch_size, cudaMemcpyHostToDevice));
		const bool status = context->executeV2(inputBindings.data());
		ASSERT_LV3(status);
		checkCudaErrors(cudaMemcpy(y1, y1_dev, sizeof(NN_Output_Policy) * batch_size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(y2, y2_dev, sizeof(NN_Output_Value ) * batch_size, cudaMemcpyDeviceToHost));
	}

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP) && defined (TENSOR_RT)

