#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(YANEURAOU_2017_EARLY_ENGINE)

#include "multi_think.h"

extern void is_ready();

void MultiThink::go_think()
{
	// 評価関数の読み込み等
	is_ready();

	// ループ上限はset_loop_max()で設定されているものとする。
	loop_count = 0;

	// threadをOptions["Threads"]の数だけ生成して思考開始。
	std::vector<std::thread> threads;
	auto thread_num = (size_t)Options["Threads"];

	// worker threadの終了フラグの確保
	thread_finished.resize(thread_num);
	
	// worker threadの起動
	for (size_t i = 0; i < thread_num; ++i)
	{
		thread_finished[i] = 0;
		threads.push_back(std::thread([i, this]
		{ 
			// プロセッサの全スレッドを使い切る。
			WinProcGroup::bindThisThread(i);

			// オーバーライドされている処理を実行
			this->thread_worker(i);

			// スレッドが終了したので終了フラグを立てる
			this->thread_finished[i] = 1;
		}));
	}

	// すべてのthreadの終了待ちを
	// for (auto& th : threads)
	//  th.join();
	// のように書くとスレッドがまだ仕事をしている状態でここに突入するので、
	// その間、callback_func()が呼び出せず、セーブできなくなる。
	// そこで終了フラグを自前でチェックする必要がある。

	// すべてのスレッドが終了したかを判定する関数
	auto threads_done = [&]()
	{
		// ひとつでも終了していなければfalseを返す
		for (auto& f : thread_finished)
			if (!f)
				return false;
		return true;
	};

	// コールバック関数が設定されているならコールバックする。
	auto do_a_callback = [&]()
	{
		if (callback_func)
			callback_func();
	};


	for (u64 i = 0 ; ; )
	{
		// 全スレッドが終了していたら、ループを抜ける。
		if (threads_done())
			break;

		sleep(1000);

		// callback_secondsごとにcallback_func()が呼び出される。
		if (++i == callback_seconds)
		{
			do_a_callback();
			i = 0;
		}
	}

	// 最後の保存。
	std::cout << std::endl << "finalize..";
	do_a_callback();

	// 終了したフラグは立っているがスレッドの終了コードの実行中であるということはありうるので
	// join()でその終了を待つ必要がある。
	for (auto& th : threads)
		th.join();

	std::cout << "..all works..done!!" << std::endl;
}


#endif // defined(EVAL_LEARN)
