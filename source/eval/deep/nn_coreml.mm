#include "nn_coreml.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined(COREML)

//#include "dlshogi_types.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include "../../usi.h"

using namespace std;
using namespace Tools;


/// Model Prediction Input Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetInput : NSObject<MLFeatureProvider>

/// input as 1 × 119 × 9 × 9 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInput:(MLMultiArray *)input NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetOutput : NSObject<MLFeatureProvider>

/// output_policy as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_policy;

/// output_value as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_value;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value NS_DESIGNATED_INITIALIZER;

@end

@implementation DlShogiResnetInput

- (instancetype)initWithInput:(MLMultiArray *)input {
	self = [super init];
	if (self) {
		_input = input;
	}
	return self;
}

- (NSSet<NSString *> *)featureNames {
	return [NSSet setWithArray:@[@"input"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
	if ([featureName isEqualToString:@"input"]) {
		return [MLFeatureValue featureValueWithMultiArray:_input];
	}
	return nil;
}

@end

@implementation DlShogiResnetOutput

- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value {
	self = [super init];
	if (self) {
		_output_policy = output_policy;
		_output_value = output_value;
	}
	return self;
}

- (NSSet<NSString *> *)featureNames {
	return [NSSet setWithArray:@[@"output_policy", @"output_value"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
	if ([featureName isEqualToString:@"output_policy"]) {
		return [MLFeatureValue featureValueWithMultiArray:_output_policy];
	}
	if ([featureName isEqualToString:@"output_value"]) {
		return [MLFeatureValue featureValueWithMultiArray:_output_value];
	}
	return nil;
}

@end

namespace Eval::dlshogi
{
	// モデルファイルの読み込み。
	Result NNCoreML::load(const std::string& model_filename , int gpu_id , int batch_size)
	{
		fixed_batch_size = batch_size;
		MLModelConfiguration* config = [MLModelConfiguration new];
		// 使用デバイス
		// MLComputeUnitsCPUOnly = 0,
		// MLComputeUnitsCPUAndGPU = 1,
		// MLComputeUnitsAll = 2
		// Allで損をする事例は見つかっていないが、選べるようにすることも考えられる。
		config.computeUnits = MLComputeUnitsAll;

		// model_filenameはxxx.mlmodelを指す。このモデルをコンパイルし、mlmodelcを生成し、それをロードする。
		NSString* model_path_ns = [NSString stringWithCString:model_filename.c_str() encoding:NSUTF8StringEncoding];
		NSString* modelc_path_ns = [NSString stringWithFormat:@"%@c", model_path_ns];

		
		NSFileManager *file_manager = [NSFileManager defaultManager];

		NSError *error = nil;
		// 自己対局などで複数プロセス・スレッドがほぼ同時にこの区間に入るとファイル作成について競合の恐れがある。
		if (![file_manager fileExistsAtPath:modelc_path_ns]) {
			if ([file_manager fileExistsAtPath:model_path_ns]) {
				sync_cout << "Compiling model" << sync_endl;
				NSURL* modelc_tmp_path = [MLModel compileModelAtURL:[NSURL fileURLWithPath:model_path_ns] error:&error];
				if (!modelc_tmp_path) {
					sync_cout << [[NSString stringWithFormat:@"info string Failed to compile model, %@", error] UTF8String] << sync_endl;
					Tools::exit();
				}
				// 既にmlmodelcがある場合や、書き込み権限がなければ失敗する。
				BOOL ret = [file_manager moveItemAtURL:modelc_tmp_path toURL:[NSURL fileURLWithPath:modelc_path_ns] error:&error];
				if (!ret) {
					sync_cout << [[NSString stringWithFormat:@"info string Failed to move compiled model from %@ to %@, %@", modelc_tmp_path, modelc_path_ns, error] UTF8String] << sync_endl;
					Tools::exit();
				}
			} else {
				sync_cout << [[NSString stringWithFormat:@"info string Model %@ does not exist", model_path_ns] UTF8String] << sync_endl;
				Tools::exit();
			}
		} else {
			sync_cout << "info string Loading already compiled model" << sync_endl;
		}

		MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:modelc_path_ns] configuration:config error:&error];

		if (!model) {
			sync_cout << [[NSString stringWithFormat:@"info string Failed to load model, %@", error] UTF8String] << sync_endl;
			Tools::exit();
		}

		if (![model init]) {
			sync_cout << "info string Failed to initialize model" << sync_endl;
			Tools::exit();
		}

		// 所有権をARCからプログラマに移す
		this->model = (void*)CFBridgingRetain(model);

		input_buf = new DType[(sizeof(NN_Input1) + sizeof(NN_Input2)) / sizeof(DType) * batch_size];

		return ResultCode::Ok;
	}

	// 使用可能なデバイス数を取得する。
	int NNCoreML::get_device_count()
	{
		// eGPUの場合は複数個もありうる？
		return 1;
	}

	// NNによる推論
	void NNCoreML::forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
	{
		if (batch_size > fixed_batch_size) {
			sync_cout << "info string batch_size > fixed_batch_size" << sync_endl;
			Tools::exit();
		}
		@autoreleasepool { // Core ML内部で確保されるバッファの解放に必要
			NSError *error = nil;
			// 所有権を移さない(プログラマのまま)
			MLModel* model = (__bridge MLModel*)(this->model);

			// x1: [batch_size, 62 (MAX_FEATURES1_NUM * COLOR_NB), 9, 9], x2: [batch_size, 57 (MAX_FEATURES2_NUM), 9, 9]として与えられたものを、[batch_size, 119, 9, 9]に詰め替える
			// fixed_batch_sizeに関わらず、意味のある部分だけ更新
			for (int i = 0; i < batch_size; i++) {
				memcpy(&input_buf[(sizeof(NN_Input1) + sizeof(NN_Input2)) / sizeof(DType) * i], &x1[i], sizeof(NN_Input1));
				memcpy(&input_buf[(sizeof(NN_Input1) + sizeof(NN_Input2)) / sizeof(DType) * i + sizeof(NN_Input1) / sizeof(DType)], &x2[i], sizeof(NN_Input2));
			}

			MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input_buf shape:@[[NSNumber numberWithInt:fixed_batch_size], @((size_t)COLOR_NB * MAX_FEATURES1_NUM + MAX_FEATURES2_NUM), @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(((size_t)COLOR_NB * MAX_FEATURES1_NUM + MAX_FEATURES2_NUM) * 9 * 9), @(9 * 9), @9, @1] deallocator:NULL error:&error];
			if (error) {
				sync_cout << [[NSString stringWithFormat:@"info string CoreML inference array allocation failed, %@", error] UTF8String] << sync_endl;
				Tools::exit();
			}

			DlShogiResnetInput *input_ = [[DlShogiResnetInput alloc] initWithInput:model_input];
			id<MLFeatureProvider> out_features = [model predictionFromFeatures:input_ options:[[MLPredictionOptions alloc] init] error:&error];
			if (error) {
				sync_cout << [[NSString stringWithFormat:@"info string CoreML inference failed, %@", error] UTF8String] << sync_endl;
				Tools::exit();
			}

			DlShogiResnetOutput *model_output = [[DlShogiResnetOutput alloc] initWithOutput_policy:(MLMultiArray *)[out_features featureValueForName:@"output_policy"].multiArrayValue output_value:(MLMultiArray *)[out_features featureValueForName:@"output_value"].multiArrayValue];

			// 出力は動的確保された領域に書き出されるため、これを引数で指定されたバッファにコピー
			memcpy(y1, model_output.output_policy.dataPointer, batch_size * MAX_MOVE_LABEL_NUM * (size_t)SQ_NB * sizeof(DType));
			memcpy(y2, model_output.output_value.dataPointer, batch_size * sizeof(DType));
		}
	}

	NNCoreML::~NNCoreML() {
		// 所有権をARCに返す
		MLModel *model = CFBridgingRelease(this->model);
		// スコープを外れるので解放される
	}

} // namespace Eval::dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP) && defined(COREML)

