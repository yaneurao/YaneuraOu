#include "../../config.h"

#if defined(USER_ENGINE)

#include "../../types.h"
#include "../../extra/all.h"

namespace YaneuraOu {

#if 0
// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, std::istringstream& is)
{
}

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void extra_option(OptionsMap & o)
{
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::search()
{
  // 例)
  //  for (auto th : Threads.slaves) th->start_searching();
  //  Thread::search();
  //  for (auto th : Threads.slaves) th->wait_for_search_finished();
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search()
{
}

#endif


namespace Eval {

	// 評価関数

	void Networks::verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const
	{
		sync_cout << "Networks::verify, evalFilePath = " << evalfilePath << sync_endl;
	}

	void Networks::load(const std::string& evalfilePath) {
		sync_cout << "Networks::load, evalFilePath = " << evalfilePath << sync_endl;
	}

	bool Networks::save(const std::string& evalfilePath) const
	{
		sync_cout << "Networks::save , filename = " << evalfilePath << sync_endl;
		return false;
	}
}

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

#endif // USER_ENGINE
