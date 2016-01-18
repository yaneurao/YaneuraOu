#ifndef _THREAD_H_
#define _THREAD_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "position.h"
#include "search.h"

// --------------------
// 探索時に用いるスレッド
// --------------------

// std::mutexをもっと速い実装に差し替えたい時のためにwrapしておく。
typedef std::mutex Mutex;
typedef std::condition_variable ConditionVariable;

// スレッドの基底クラス。std::threadのwrapper。
struct ThreadBase : public std::thread
{
  // idle_loop()で待機しているスレッドに通知して処理を進められる状態にする。
  void notify_one();

  // idle_loop()で待機しているスレッドを終了させる。
  void terminate();

  // 派生クラス側でoverrideして使う。sleepCondition待ちにして待機させておく。
  // exitフラグが立ったら終了するように書く。
  // 継承を使うとvtableが必要になってalignするのが難しくなるので使わない。
  void idle_loop() {}

  ThreadBase() { exit = false; }

  // bの状態がtrueになるのを待つ。
  // 他のスレッドからは、この待機を解除するには、bをtrueにしてからnotify_one()を呼ぶこと。
  void wait(std::atomic_bool& b) {
    std::unique_lock<Mutex> lk(mutex);
    sleepCondition.wait(lk, [&] { return bool(b); });
  }

  // bの状態がfalseになるのを待つ。
  // 他のスレッドからは、この待機を解除するには、bをtrueにしてからnotify_one()を呼ぶこと。
  void wait_while(std::atomic_bool& b) {
    std::unique_lock<Mutex> lk(mutex);
    sleepCondition.wait(lk, [&] { return !b; });
  }

protected:

  // idle_loop()で待機しているときに待つ対象
  ConditionVariable sleepCondition;
  
  // notify_one()するときに使うmutex
  Mutex mutex;

  std::atomic_bool exit;
};

// 探索時に用いる、それぞれのスレッド
// これを思考スレッド数だけ確保する。
struct Thread : public ThreadBase
{
  Thread();

  // slave用のidle_loop。
  void idle_loop();

  // slaveは、main threadから
  // for(auto th : Threads.slavle) th->search_start();のようにされると
  // この関数が呼び出される。
  // MainThreadのほうからはこの関数は呼び出されない。
  // MainThread::think()が呼び出される。
  void search();

  // スレッドidが返る。
  // MainThreadなら0、slaveなら1,2,3,...
  size_t thread_id() const { return idx; }
  
  // main threadであるならtrueを返す。
  bool is_main() const { return idx == 0; }

  // MainThreadがslaveを起こして、Thread::search()を開始させるときに呼び出す。
  void search_start() { searching = true; notify_one(); }

  // このスレッドのsearchingフラグがfalseになるのを待つ。(MainThreadがslaveの探索が終了するのを待機するのに使う)
  void join() { wait_while(searching); }

  // 探索開始局面(alignasが利くように先頭に書いておく)
  Position rootPos;

  // 探索開始局面で思考対象とする指し手の集合。
  // goコマンドで渡されていなければ、全合法手(ただし歩の不成などは除く)とする。
  std::vector<Search::RootMove> rootMoves;

protected:
  // 探索中であるかを表すフラグ
  std::atomic_bool searching;

  // thread id
  size_t idx;
};

// 探索時のmainスレッド(これがmasterであり、これ以外はslaveとみなす)
struct MainThread : public Thread
{
  // スレッドが思考を停止するのを待つ
  void join() { wait_while(thinking); }

  void idle_loop();

  // 思考を開始する。search.cppで定義されているthink()が呼び出される。
  void think();

  // 思考中であることを表すフラグ。
  // これをtrueにしてからnotify_one()でスレッドを起こすと思考を開始する。
  std::atomic_bool thinking;
};

struct Slaves
{
  std::vector<Thread*>::iterator begin() const;
  std::vector<Thread*>::iterator end() const;
};

// 思考で用いるスレッドの集合体
// 継承はあまり使いたくないが、for(auto* th:Threads) ... のようにして回せて便利なのでこうしておく。
struct ThreadPool : public std::vector<Thread*>
{
  // 起動時に呼び出される。Main
  void init();

  // 終了時に呼び出される
  void exit();

  // mainスレッドを取得する。これはthis[0]がそう。
  MainThread* main() { return static_cast<MainThread*>(at(0)); }

  // mainスレッドに思考を開始させる。
  void start_thinking(const Position& pos, const Search::LimitsType& limits, Search::StateStackPtr& states);

  // 今回、goコマンド以降に探索したノード数
  int64_t nodes_searched() { int64_t nodes = 0; for (auto*th : *this) nodes += th->rootPos.nodes_searched(); return nodes; }

  // main()以外のスレッド
  Slaves slaves;

  // USIプロトコルで指定されているスレッド数を反映させる。
  void read_usi_options();
};

// ThreadPoolのglobalな実体
extern ThreadPool Threads;

#endif // _THREAD_H_
