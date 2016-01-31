#include "thread.h"

#include "misc.h"

ThreadPool Threads;

using namespace std;
using namespace Search;

namespace {

  // std::thread派生型であるT型のthreadを一つ作って、そのidle_loopを実行するためのマクロ。
  // 生成されたスレッドはidle_loop()で仕事が来るのを待機している。
  template<typename T> T* new_thread() {

    std::thread* th = new (_mm_malloc(sizeof(T),alignof(T))) T();

    // Tの基底クラスはstd::threadなのでスライシングされて正しく代入されるはず。
    *th = std::thread([&] {((T*)th)->idle_loop(); });
    
    return (T*)th;
  }

  // new_thread()の逆。エンジン終了時に呼び出される。
  void delete_thread(ThreadBase *th) {
    th->terminate();
    _mm_free(th);
  }
}

void ThreadBase::notify_one() {

  std::unique_lock<Mutex> lk(mutex);

  // idle_loopでsleepCondition待ちになっているはずなので、これに通知してidle_loop内での処理を進める。
  sleepCondition.notify_one();
}

void ThreadBase::terminate() {

  mutex.lock();
  exit = true; // 探索は終わっているはずなのでこのフラグをセットして待機する。
  mutex.unlock();

  notify_one();
  join();
}

Thread::Thread()
{
  searching = false;
  idx = Threads.size(); // スレッド番号(MainThreadが0。slaveは1から順番に)
}


// slave用のidle_loop。searching == trueにして、notify_one()で起こされるとThread::search(false)として呼び出す。
// search.cppのほうでこの関数を定義して探索関係の処理を書くと良い。
void Thread::idle_loop() {

  while (!exit)
  {
    std::unique_lock<Mutex> lk(mutex);

    while (!searching && !exit)
      sleepCondition.wait(lk);

    lk.unlock();

    if (!exit && searching)
    {
      search();
      searching = false;
      notify_one(); // searchingの変化を待っているMainThreadを起こしてやる
    }
  }
}

// 探索するときのmasterスレッド。探索開始するまで待っている。
void MainThread::idle_loop()
{
  while (!exit)
  {
    std::unique_lock<Mutex> lk(mutex);

    thinking = false;

    // thinkかexitかどちらかがtrueになるのを待つ
    while (!thinking && !exit)
    {
      sleepCondition.notify_one(); // UIスレッドがsleepCondition待ちになっているのを起こすために
      sleepCondition.wait(lk);
    }

    lk.unlock();

    // ここに抜けてきて!exitでなければthink==true
    // また、think==trueでもexit==trueなら抜けなければならない。ゆえにthink()を呼び出す条件としては!exitとなる。
    if (!exit)
      think();
  }
}


std::vector<Thread*>::iterator Slaves::begin() const { return Threads.begin() + 1; }
std::vector<Thread*>::iterator Slaves::end() const { return Threads.end(); }

void ThreadPool::init() {
  // MainThreadを一つ生成して、そのあとusi_optionを呼び出す。
  // そのなかにスレッド数が書いてあるので足りなければその数だけスレッドが生成される。

  push_back(new_thread<MainThread>());
  read_usi_options();
}

void ThreadPool::exit()
{
  for (auto th : *this)
    delete_thread(th);

  clear(); // 念のため使わなくなったポインタを開放しておく。
}

// USIプロトコルで指定されているスレッド数を反映させる。
void ThreadPool::read_usi_options() {

  // MainThreadが生成されてからしかworker threadを生成してはいけない作りになっているので
  // USI::Optionsの初期化のタイミングでこの関数を呼び出すわけにはいかない。
  // ゆえにUSI::Optionを引数に取らず、USI::Options["Threads"]から値をもらうようにする。
  size_t requested = Options["Threads"];
  ASSERT_LV1(requested > 0);

  // 足りなければ生成
  while (size() < requested)
    push_back(new_thread<Thread>());

  // 余っていれば解体
  while (size() > requested)
  {
    delete_thread(back());
    pop_back();
  }
}


void ThreadPool::start_thinking(const Position& pos, const Search::LimitsType& limits, Search::StateStackPtr& states)
{
  // 思考中であれば停止するまで待つ。
  main()->join();

  Signals.stop = false;

  main()->rootMoves.clear();
  main()->rootPos = pos;
  Limits = limits;

  // statesが呼び出し元から渡されているならこの所有権をSearch::SetupStatesに移しておく。
  if (states.get())
    SetupStates = std::move(states);

  // 初期局面では合法手すべてを生成してそれをrootMovesに設定しておいてやる。
  // このとき、歩の不成などの指し手は除く。(そのほうが勝率が上がるので)
  // また、goコマンドでsearchmovesが指定されているなら、そこに含まれていないものは除く。
  for (auto m : MoveList<LEGAL>(pos))
    if (limits.searchmoves.empty()
      || count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
      main()->rootMoves.push_back(RootMove(m));

  // おまけでslaveの初期局面も同じにしておいてやる。
  for (auto th : *this)
    if (th != main())
    {
      th->rootPos = Position(main()->rootPos);
      th->rootMoves = main()->rootMoves;
    }

  // Positionクラスに対して、それを探索しているスレッドを設定しておいてやる。
  for (auto th : *this)
    th->rootPos.set_this_thread(th);

  main()->thinking = true;
  main()->notify_one(); // mainスレッドでの思考を開始させる。このときthinkingフラグは先行してtrueになっていなければならない。

}