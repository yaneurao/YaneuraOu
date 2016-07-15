#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

#include "multi_think.h"

using namespace std;

extern void is_ready();

// いまのところ、やねうら王2016Midしか、このスタブを持っていない。
namespace Learner
{
  extern pair<Value, vector<Move> >  search(Position& pos, Value alpha, Value beta, int depth);
  extern pair<Value, vector<Move> > qsearch(Position& pos, Value alpha, Value beta);
}

// 通常探索をして、その結果を返す。
pair<Value, vector<Move> >  MultiThink::search(Position& pos, Value alpha, Value beta, int depth)
{
  return Learner::search(pos, alpha, beta, depth);
}

// 静止探索をして、その結果を返す。
pair<Value, vector<Move> > MultiThink::qsearch(Position& pos, Value alpha, Value beta)
{
  return Learner::qsearch(pos, alpha, beta);
}

void MultiThink::go_think()
{
  // 評価関数の読み込み等
  is_ready();
  
  // ループ上限はset_loop_max()で設定されているものとする。
  loop_count = 0;

  // threadをOptions["Threads"]の数だけ生成して思考開始。
  vector<std::thread> threads;
  auto thread_num = (size_t)Options["Threads"];

  // worker threadの終了フラグの確保
  thread_finished.reset(new volatile bool[thread_num]);

  // worker threadの起動
  for (size_t i = 0; i < thread_num; ++i)
  {
    thread_finished.get()[i] = false;
    threads.push_back(std::thread([i, this] { this->thread_worker(i); this->thread_finished.get()[i] = true; }));
  }

  // すべてのthreadの終了待ちを
  // for (auto& th : threads)
  //  th.join();
  // のように書くとスレッドがまだ仕事をしている状態でここに突入するので、
  // その間、callback_func()が呼び出せず、セーブできなくなる。
  // そこで終了フラグを自前でチェックする必要がある。

  while (true)
  {
    // 5秒ごとにスレッドの終了をチェックする。
    const int check_interval = 5;

    for (int i = 0; i < callback_seconds/check_interval; ++i)
    {
      this_thread::sleep_for(chrono::seconds(check_interval));

      // すべてのスレッドが終了したか
      for (size_t i = 0; i < thread_num; ++i)
        if (!thread_finished.get()[i])
          goto NEXT;

      // すべてのthread_finished[i]に渡って、trueなのですべてのthreadが終了している。
      goto FINISH;

    NEXT:;
    }

    // callback_secondsごとにcallback_func()が呼び出される。
    if (callback_func)
      callback_func();
  }
FINISH:;
  

  // 最後の保存。
  if (callback_func)
  {
	  cout << endl << "finalize..";
	  callback_func();
  }

  // 終了したフラグは立っているがスレッドの終了コードの実行中であるということはありうるので
  // join()でその終了を待つ必要がある。
  for (size_t i = 0; i < thread_num; ++i)
	  threads[i].join();

  cout << "..all works..done!!" << endl;
}


#endif // defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)
