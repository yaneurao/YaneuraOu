#ifndef __FUKAURAOU_ENGINE_H_INCLUDED__
#define __FUKAURAOU_ENGINE_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../engine.h"
#include "../../book/book.h"
#include "../../usioption.h"
#include "../../numa.h"

#include "dlshogi_types.h"
#include "dlshogi_searcher.h"
#include "SearchOptions.h"

namespace dlshogi {

/*
	この探索部は、dlshogiのソースコードを参考にさせていただいています。🙇
	DeepLearningShogi GitHub : https://github.com/TadaoYamaoka/DeepLearningShogi
*/

class FukauraOuEngine;

class FukauraOuWorker : public YaneuraOu::Search::Worker {

	FukauraOuWorker(OptionsMap&               options,
                    ThreadPool&               threads,
                    size_t                    threadIdx,
                    NumaReplicatedAccessToken numaAccessToken,
                    DlshogiSearcher&          searcher,
                    FukauraOuEngine&          engine);

	// "go"コマンドで呼び出される。
    virtual void start_searching() override;

	// dlshogiの探索部本体
    DlshogiSearcher& searcher;

	// ふかうら王のEngine本体
	FukauraOuEngine& engine;
};

class FukauraOuEngine: public YaneuraOu::Engine {
   public:
    FukauraOuEngine();

    // エンジンoptionを生やす。
    virtual void add_options() override;

    // "isready"コマンド応答。
    virtual void isready() override;

    // エンジン作者名の変更。
    virtual std::string get_engine_author() const override;

    // 定跡の指し手を選択するモジュール。
    YaneuraOu::Book::BookMoveSelector book;

	// dlshogiの探索部本体
    DlshogiSearcher searcher;

   protected:
    // NNの設定を生やす。add_options()時に呼び出される。
    void add_nn_options();

	// "isready"タイミングで行うGPUの初期化。
	void init_gpu();

	// "Max_GPU","Disabled_GPU"と"UCT_Threads"の設定値から、各GPUのスレッド数の設定を返す。
    std::vector<int> FukauraOuEngine::get_thread_settings();

};  // class FukauraOuEngine

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __FUKAURAOU_ENGINE_H_INCLUDED__
