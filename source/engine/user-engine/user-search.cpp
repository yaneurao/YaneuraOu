#include "../../config.h"

#if defined(USER_ENGINE)

/*

	ユーザーエンジンを製作するサンプル

	これを参考に、あなただけのユーザーエンジンを作ってみてください。

*/

#include "../../types.h"
#include "../../extra/all.h"

namespace YaneuraOu {

namespace Search {

	class UserWorker : public Worker
	{
	public:

		UserWorker(OptionsMap& options, ThreadPool& threads, size_t threadIdx, NumaReplicatedAccessToken numaAccessToken):
			// 基底classのconstructorの呼び出し
			Worker(options,threads,threadIdx,numaAccessToken){ }

		// このworker(探索用の1つのスレッド)の初期化
		// 📝 これは、"usinewgame"のタイミングで、すべての探索スレッド(エンジンオプションの"Threads"で決まる)に対して呼び出される。
		virtual void clear() override
		{
			sync_cout << "UserWorker::clear" << sync_endl;
		}

		// Workerによる探索の開始
		// 📝　メインスレッドに対して呼び出される。
		//     そのあと非メインスレッドに対してstart_searching()を呼び出すのは、threads.start_searching()を呼び出すと良い。
		virtual void start_searching() override
		{
			sync_cout << "UserWorker::start_searching , position sfen = " << rootPos.sfen() << ", threadIdx = " << threadIdx << sync_endl;

			if (is_mainthread())
			{
				threads.start_searching();  // start non-main threads

				Tools::sleep(1000);

				// bestmoveとして投了する。
				sync_cout << "bestmove resign" << sync_endl;
			}
		}

	};


} // namespace Search

class UserEngine : public Engine
{
    // "usi"コマンドに対して出力するエンジン名と作者。
    virtual std::string get_engine_name() const override { return "my engine"; }
    virtual std::string get_engine_author() const override { return "myself"; }

	// 💡 ↓のように"usi"出力を丸ごとカスタマイズもできる。
	#if 0
	virtual void usi() override {
        sync_cout << "id user-engine\n"
                  << "author a user" << sync_endl;
	}
	#endif

	// "isready"のタイミングのcallback。時間のかかる初期化処理はここで行う。
	virtual void isready() override
	{
		sync_cout << "UserEngine::isready" << sync_endl;

		// Engine classのisready()でスレッド数の反映処理などがあるので、そちらに委譲してやる。
		Engine::isready();
	}

	// エンジンに追加オプションを設定したいときは、この関数を定義する。
	virtual void add_options() override
	{
		// 基底classのadd_options()を呼び出して"Threads", "NumaPolicy"など基本的なオプションを生やす。
		Engine::add_options();

		sync_cout << "UserEngine::add_options" << sync_endl;

		// 試しに、Optionを生やしてみる。
		options.add("HogeOption", Option("hogehoge"));
	}

	// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使う。
	virtual void user(std::istringstream& is) override
	{
		sync_cout << "UserEngine::user_cmd" << sync_endl;
	}

	// スレッド数を反映させる関数
	virtual void resize_threads() override
	{
		// 💡 Engine::resize_threads()を参考に書くと良いでしょう。

		// 📌 探索の終了を待つ
		threads.wait_for_search_finished();

		// 📌 スレッド数のリサイズ

		// 💡　難しいことは考えずにコピペして使ってください。"Search::UserWorker"と書いてあるところに、
		//      あなたの作成したWorker派生classの名前を書きます。
		auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken)
			{ return std::make_unique<Search::UserWorker>(options, threads, threadIdx, numaAccessToken); };
                threads.set(numaContext.get_numa_config(), options,
                            options["Threads"], worker_factory);

		// 📌 NUMAの設定

		// スレッドの用いる評価関数パラメーターが正しいNUMAに属するようにする
		threads.ensure_network_replicated();
	}
};

} // namespace YaneuraOu

using namespace YaneuraOu;

namespace {

	// 自作のエンジンのentry point
	void engine_main()
	{
		// ここで作ったエンジン
		UserEngine engine;

		// USIコマンドの応答部
		USIEngine usi;
		usi.set_engine(engine); // エンジン実装を差し替える。

		// USIコマンドの応答のためのループ
		usi.loop();
	}

	// このentry pointを登録しておく。
	static EngineFuncRegister r(engine_main, "UserEngine", 0);

} // namespace


#endif // USER_ENGINE
