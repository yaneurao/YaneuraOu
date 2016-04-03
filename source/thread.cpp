#include "thread.h"

#include "misc.h"

ThreadPool Threads;

using namespace std;
using namespace Search;

namespace {

  // std::thread派生型であるT型のthreadを一つ作って、そのidle_loopを実行するためのマクロ。
  // 生成されたスレッドはidle_loop()で仕事が来るのを待機している。
  template<typename T> T* new_thread() {
    T* th = new (_mm_malloc(sizeof(T),alignof(T))) T();
    return (T*)th;
  }

  // new_thread()の逆。エンジン終了時に呼び出される。
  void delete_thread(Thread *th) {
    th->terminate();
    _mm_free(th);
  }
}

Thread::Thread()
{
  resetCalls = exit = false;

  // maxPlyを更新しない思考エンジンでseldepthの出力がおかしくなるのを防止するために
  // ここでとりあえず初期化しておいてやる。
  maxPly = callsCnt = 0;

  idx = Threads.size();  // スレッド番号(MainThreadが0。slaveは1から順番に)

  // スレッドを一度起動してworkerのほう待機状態にさせておく
  std::unique_lock<Mutex> lk(mutex);
  searching = true;
  nativeThread = std::thread(&Thread::idle_loop, this);
  sleepCondition.wait(lk, [&] {return !searching; });
}

// std::threadの終了を待つ(デストラクタに相当する)
// Threadクラスがnewするときにalignasが利かないので自前でnew_thread(),delete_thread()を
// 呼び出しているのでデストラクタで書くわけにはいかない。
void Thread::terminate() {

  mutex.lock();
  exit = true; // 探索は終わっているはずなのでこのフラグをセットして待機する。
  sleepCondition.notify_one();
  mutex.unlock();
  nativeThread.join();
}

// 探索するときのmaster,slave用のidle_loop。探索開始するまで待っている。
void Thread::idle_loop() {

  while (!exit)
  {
    std::unique_lock<Mutex> lk(mutex);

    searching = false;

    while (!searching && !exit)
    {
      sleepCondition.notify_one(); // 他のスレッドがこのスレッドを待機待ちしてるならそれを起こす
      sleepCondition.wait(lk);
    }

    lk.unlock();

    // !exitで抜けてきたということはsearch == trueというわけだから探索する。
    // exit == true && search == trueというケースにおいてはsearch()を呼び出してはならないので
    // こういう書き方をしてある。
    if (!exit)
      search();
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
  // 逆順で解体する必要がある。
  while (size())
  {
    delete_thread(back());
    pop_back();
  }
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

void ThreadPool::init_for_slave(const Position& pos, const Search::LimitsType& limits)
{
  // 初期局面では合法手すべてを生成してそれをrootMovesに設定しておいてやる。
  // このとき、歩の不成などの指し手は除く。(そのほうが勝率が上がるので)
  // また、goコマンドでsearchmovesが指定されているなら、そこに含まれていないものは除く。
  for (auto m : MoveList<LEGAL>(pos))
    if (limits.searchmoves.empty()
      || count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
      main()->rootMoves.push_back(RootMove(m));

  // おまけでslaveの初期局面も同じにしておいてやる。
  for (auto th : *this)
  {
    if (th != main())
    {
      th->rootPos = Position(main()->rootPos);
      th->rootMoves = main()->rootMoves;
    }
    // Positionクラスに対して、それを探索しているスレッドを設定しておいてやる。
    th->rootPos.set_this_thread(th);
  }
}

void ThreadPool::start_thinking(const Position& pos, const Search::LimitsType& limits, Search::StateStackPtr& states)
{
  // 思考中であれば停止するまで待つ。
  main()->wait_for_search_finished();

  Signals.stop = false;

  main()->rootMoves.clear();
  main()->rootPos = pos;
  Limits = limits;
  
  // statesが呼び出し元から渡されているならこの所有権をSearch::SetupStatesに移しておく。
  if (states.get())
    SetupStates = std::move(states);

  init_for_slave(pos, limits);

  main()->start_searching();
}