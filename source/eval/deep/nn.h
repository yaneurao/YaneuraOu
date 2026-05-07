#ifndef __NN_H_INCLUDED__
#define __NN_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "nn_types.h"

namespace YaneuraOu {
namespace Eval::dlshogi {

	// ニューラルネットの基底クラス
	class NN
	{
	public:
		// forwardに渡すメモリの確保
		// size個分だけDTypeを確保して返してくれる。
		// TODO : 派生クラス側でoverrideするかも。
		void* alloc(size_t size);

		// alloc()で確保したメモリの開放
		// TODO : 派生クラス側でoverrideするかも。
		void free(void* ptr);

		// NNによる推論。
		virtual void forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2) = 0;

		// TensorRTのように複数の推論slotを使える実装では、slot_idを指定して推論する。
		// それ以外の実装では従来のforward()に委譲する。
		virtual void forward(const int slot_id, const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
		{
			(void)slot_id;
			forward(batch_size, p1, p2, x1, x2, y1, y2);
		}

		// モデルファイルの読み込み。
		virtual Tools::Result load(const std::string& model_path, int gpu_id , int batch_size) = 0;

		// TensorRTのoptimization profile数を指定する。対応しない実装では無視する。
		virtual void set_profile_count(int profile_count) { (void)profile_count; }

		// 確保済みの推論slot数。
		virtual int slot_capacity() const { return 1; }

		// 指定数の推論slotを準備する。対応しない実装では何もしない。
		virtual void prepare_slots(int slot_count) { (void)slot_count; }

		// 使用可能なデバイス数を取得する。
		static int get_device_count();

		// 現在のスレッドとGPUを紐付ける。
		// ※　CUDAの場合、cudaSetDevice()を呼び出す。必ず、そのスレッドの探索開始時(forward()まで)に一度はこれを呼び出さないといけない。
		virtual void set_device(int gpu_id) {};

		// モデルファイル名を渡すとそれに応じたNN派生クラスをbuildして返してくれる。デザパタで言うところのbuilder。
		static std::shared_ptr<NN> build_nn(const std::string& model_path, int gpu_id, int batch_size, int profile_count = 1);

		// 派生クラス側のデストラクタ呼び出されてほしいのでこれ用意しとく。
		virtual ~NN() {}
	};

} // namespace Eval::dlshogi
} // namespace YaneuraOu

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __NN_H_INCLUDED__
