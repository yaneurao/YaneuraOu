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
  for (size_t i = 0; i < thread_num; ++i)
  {
    threads.push_back(std::thread([i,this] { this->thread_worker(i);  }));
  }

  if (callback_func)
  {
    while (true)
    {
      for (int i = 0; i < callback_seconds ; ++i)
      {
        // 1秒おきに終了チェック
        this_thread::sleep_for(chrono::seconds(1));
        if (loop_count == loop_max)
          goto Exit;
      }
      callback_func();
    }
  Exit:;
  }
  // すべてのthreadの終了待ち
  for (auto& th : threads)
  {
    th.join();
  }

  // 最後の保存。
  cout << "finalize.." << endl;
  callback_func();
  cout << "makebook..done!!" << endl;
}


#endif // defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)
