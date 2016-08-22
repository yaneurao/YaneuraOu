#ifndef _MULTI_THINK_
#define _MULTI_THINK_

#include "../shogi.h"

#if defined(EVAL_LEARN)

#include "../misc.h"

// 棋譜からの学習や、自ら思考させて定跡を生成するときなど、
// 複数スレッドが個別にSearch::think()を呼び出したいときに用いるヘルパクラス。
// このクラスを派生させて用いる。
struct MultiThink
{
  // 発生する乱数は再現性があるように同じものにしておく。
  // この動作が気に要らないなら、set_prng()を用いるべし。
  MultiThink() : prng(20160101){}

  // マスタースレッドからこの関数を呼び出すと、スレッドがそれぞれ思考して、
  // 思考終了条件を満たしたところで制御を返す。
  // [要件]
  // 1) thread_worker()のオーバーライド
  // 2) set_loop_max()でループ回数の設定
  // 3) 定期的にcallbackされる関数を設定する(必要なら)
  //   callback_funcとcallback_interval
  void go_think();

  // go_think()したときにスレッドを生成して呼び出されるthread worker
  // これをoverrideして用いる。
  virtual void thread_worker(size_t thread_id) = 0;

  // go_think()したときにcallback_seconds[秒]ごとにcallbackされる。
  std::function<void()> callback_func;
  int callback_seconds = 600;

  // workerが処理する(Search::think()を呼び出す)回数を設定する。
  void set_loop_max(u64 loop_max_) { loop_max = loop_max_; }

  // [ASYNC] ループカウンターの値を取り出して、取り出し後にループカウンターを加算する。
  // もしループカウンターがloop_maxに達していたらUINT64_MAXを返す。
  u64 get_next_loop_count() {
    std::unique_lock<Mutex> lk(loop_mutex);
    if (loop_count >= loop_max)
      return UINT64_MAX;
    return loop_count++;
  }

  // [ASYNC] 通常探索をして、その結果を返す。
  std::pair<Value, std::vector<Move> >  search(Position& pos, Value alpha, Value beta, int depth);

  // [ASYNC] 静止探索をして、その結果を返す。
  std::pair<Value, std::vector<Move> > qsearch(Position& pos, Value alpha, Value beta);

  // worker threadがI/Oにアクセスするときのmutex
  Mutex io_mutex;

protected:

  // [ASYNC] 乱数を一つ取り出す。
  template<typename T> T rand() {
    std::unique_lock<Mutex> lk(rand_mutex);
    return T(prng.rand64());
  }

  // [ASYNC] 0からn-1までの乱数を返す。(一様分布ではないが現実的にはこれで十分)
  uint64_t rand(size_t n) {
    std::unique_lock<Mutex> lk(rand_mutex);
    return prng.rand<uint64_t>() % n;
  }

  // 乱数の再初期化をしたいときに用いる。
  void set_prng(PRNG prng_)
  {
    prng = prng_;
  }

private:
  // workerが処理する(Search::think()を呼び出す)回数
  volatile u64 loop_max;
  // workerが処理した(Search::think()を呼び出した)回数
  volatile u64 loop_count = 0;

  // ↑の変数を変更するときのmutex
  Mutex loop_mutex;

  // 乱数発生器本体
  PRNG prng;

  // ↑の乱数を取得するときのmutex
  Mutex rand_mutex;

  // スレッドの終了フラグ。
  std::shared_ptr<volatile bool> thread_finished;

};

#endif // defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

#endif
