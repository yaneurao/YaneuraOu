#include "../../config.h"

#if defined(USER_ENGINE)

/*

	ユーザーエンジンを製作するサンプル

	これを参考に、あなただけのユーザーエンジンを作ってみてください。

*/

#include "../../types.h"
#include "../../extra/all.h"

namespace YaneuraOu {

namespace Eval {

	// 評価関数

	// 評価関数パラメーターが読み込まれているかのチェック。
	void Networks::verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const
	{
		sync_cout << "Networks::verify, evalFilePath = " << evalfilePath << sync_endl;
	}

	// 評価関数パラメーターを読み込む。
	void Networks::load(const std::string& evalfilePath) {
		sync_cout << "Networks::load, evalFilePath = " << evalfilePath << sync_endl;
	}

	// 評価関数パラメーターを保存する。
	bool Networks::save(const std::string& evalfilePath) const
	{
		sync_cout << "Networks::save , filename = " << evalfilePath << sync_endl;
		return false;
	}
}

class UserEngine : public Engine
{
	// "isready"のタイミングのcallback。時間のかかる初期化処理はここで行う。
	virtual void isready() override
	{
		sync_cout << "Engine::isready" << sync_endl;
	}

	// エンジンに追加オプションを設定したいときは、この関数を定義する。
	virtual void extra_option() override
	{
		sync_cout << "Engine::extra_option" << sync_endl;

		// 試しに、Optionを生やしてみる。
		options.add("HogeOption", Option("hogehoge"));
	}

	// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使う。
	virtual void user(std::istringstream& is) override
	{
		sync_cout << "Engine::user_cmd" << sync_endl;
	}
};

namespace Search {

	// このworker(探索用の1つのスレッド)の初期化
	// 📝 これは、"usinewgame"のタイミングで、すべての探索スレッド(エンジンオプションの"Threads"で決まる)に対して呼び出される。
	void Worker::clear()
	{
		sync_cout << "Worker::clear" << sync_endl;
	}

	// Workerによる探索の開始
	// 📝　メインスレッドに対して呼び出される。
	//     そのあと非メインスレッドに対してstart_searching()を呼び出すのは、threads.start_searching()を呼び出すと良い。
	void Worker::start_searching()
	{
		sync_cout << "Worker::start_searching , position sfen = " << rootPos.sfen() << ", threadIdx = " << threadIdx << sync_endl;

		if (is_mainthread())
		{
			threads.start_searching();  // start non-main threads

			// 1秒後にcheck_time()を呼び出してみる。
			Sleep(1000);
			main_manager()->check_time(*this);
		}
	}

	// 探索中に、main threadから一定間隔ごとに呼び出して
	// ここで残り時間のチェックを行う。(ことになっている)
	void SearchManager::check_time(Search::Worker& worker)
	{
		sync_cout << "SearchManager::check_time" << sync_endl;
	}

} // namespace Search
} // namespace YaneuraOu

using namespace YaneuraOu;

// 自作のエンジンのentry point
void engine_main()
{
	// ここで作ったエンジン
	UserEngine engine;

	// USIコマンドの応答部
	USIEngine usi;
	usi.set_engine(engine);

	// USIコマンドの応答のためのループ
	usi.loop();
}

#endif // USER_ENGINE
