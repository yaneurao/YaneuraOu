#ifndef _THREAD_H_
#define _THREAD_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "position.h"
#include "search.h"
#include "move_picker.h"

// --------------------
// 探索時に用いるスレッド
// --------------------

// 探索時に用いる、それぞれのスレッド
// これを思考スレッド数だけ確保する。
// ただしメインスレッドはこのclassを継承してMainThreadにして使う。
struct Thread
{
	// slaveは、main threadから
	// for(auto th : Threads.slavle) th->start_searching();のようにされると
	// この関数が呼び出される。
	// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
	// この関数を呼び出したいときは、Thread::search()とすること。
	virtual void search();

	// スレッド起動後、この関数が呼び出される。
	void idle_loop();

	// ------------------------------
	//      同期待ちのwait等
	// ------------------------------

	// MainThreadがslaveを起こして、Thread::search()を開始させるときに呼び出す。
	// resume == falseなら(探索中断～再開時でないなら)searchingフラグをtrueにする。
	// これはstopコマンドかponderhitコマンドを受信したときにスレッドが(すでに思考を終えて)
	// 寝てるのを起こすときに使う。
	void start_searching(bool resume = false)
	{
		std::unique_lock<Mutex> lk(mutex);
		if (!resume)
			searching = true;
		sleepCondition.notify_one();
	}

	// 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)
	void wait_for_search_finished()
	{
		std::unique_lock<Mutex> lk(mutex);
		sleepCondition.wait(lk, [&] { return !searching; });
	}

	// bの状態がtrueになるのを待つ。
	// 他のスレッドからは、この待機を解除するには、bをtrueにしてからnotify_one()を呼ぶこと。
	void wait(std::atomic_bool& condition) {
		std::unique_lock<Mutex> lk(mutex);
		sleepCondition.wait(lk, [&] { return bool(condition); });
	}

	// ------------------------------
	//       プロパティ
	// ------------------------------

	// スレッドidが返る。
	// MainThreadなら0、slaveなら1,2,3,...
	size_t thread_id() const { return idx; }

	// main threadであるならtrueを返す。
	bool is_main() const { return idx == 0; }

	// ------------------------------
	//       探索に必要なもの
	// ------------------------------

	// 探索開始局面
	Position rootPos;

	// 探索開始局面で思考対象とする指し手の集合。
	// goコマンドで渡されていなければ、全合法手(ただし歩の不成などは除く)とする。
	std::vector<Search::RootMove> rootMoves;

	// このスレッドでMultiPVを用いているとして、rootMovesの(0から数えて)何番目のPVの指し手を探索中であるか
	// MultiPVでないときはこの変数の値は0。
	size_t PVIdx;

	// rootから最大、何手目まで探索したか(選択深さの最大)
	int selDepth;

	// このスレッドが探索したノード数(≒Position::do_move()を呼び出した回数)
	std::atomic<uint64_t> nodes;

	// 反復深化の深さ
	Depth rootDepth;

	// このスレッドに関して、終了した反復深化の深さ
	Depth completedDepth;

	// ある種のMovePickerではオーダリングのために、
	// スレッドごとにhistoryとcounter movesのtableを持たないといけない。
#if defined ( USE_MOVE_PICKER_2017Q2 )
	CounterMoveStat counterMoves;
	ButterflyHistory history;
#endif

#if defined( PER_THREAD_COUNTERMOVEHISTORY )
	// コア数が多いか、長い持ち時間においては、スレッドごとにCounterMoveHistoryを確保したほうが良い。
	// cf. https://github.com/official-stockfish/Stockfish/commit/5c58d1f5cb4871595c07e6c2f6931780b5ac05b5
	CounterMoveHistoryStat counterMoveHistory;
#endif

	// ------------------------------
	//       constructor ..
	// ------------------------------

	Thread();
	void terminate();

protected:

	Mutex mutex;

	// idle_loop()で待機しているときに待つ対象
	ConditionVariable sleepCondition;

	// exitフラグが立ったら終了する。
	bool exit;

	// 探索中であるかを表すフラグ
	bool searching;

	// thread id
	size_t idx;

	// wrapしているstd::thread
	std::thread nativeThread;
};
  

// 探索時のmainスレッド(これがmasterであり、これ以外はslaveとみなす)
struct MainThread: public Thread
{
	// この関数はvirtualになっていてthink()が呼び出される。
	// MainThread::think()から呼び出すべきは、Thread::search()
	virtual void search() { think(); }

	// 思考時間の終わりが来たかをチェックする。
	void check_time();

	// 思考を開始する。engine/*/*_search.cpp等で定義されているthink()が呼び出される。
	void think();

	// 反復深化のときにPVがあまり変化がないなら探索が安定しているということだから
	// 短めの思考時間で指す機能のためのフラグ。
	bool easyMovePlayed;

	// root nodeでfail lowが起きているのか
	bool failedLow;

	// 反復深化においてbestMoveが変わった回数。nodeの安定性の指標として使う。
	double bestMoveChanges;

	// 前回の探索時のスコア。
	// 次回の探索のときに何らか使えるかも。
	Value previousScore;

	// check_time()で用いるカウンター。
	// デクリメントしていきこれが0になるごとに思考をストップするのか判定する。
	int callsCnt = 0;
};

struct Slaves
{
	std::vector<Thread*>::iterator begin() const;
	std::vector<Thread*>::iterator end() const;
};

// 思考で用いるスレッドの集合体
// 継承はあまり使いたくないが、for(auto* th:Threads) ... のようにして回せて便利なのでこうしておく。
struct ThreadPool: public std::vector<Thread*>
{
	// 起動時に呼び出される。Main
	void init();

	// 終了時に呼び出される
	void exit();

	// mainスレッドを取得する。これはthis[0]がそう。
	MainThread* main() { return static_cast<MainThread*>(at(0)); }

	// mainスレッドに思考を開始させる。
	void start_thinking(const Position& pos, Search::StateStackPtr& states , const Search::LimitsType& limits);

	// 今回、goコマンド以降に探索したノード数
	uint64_t nodes_searched() { uint64_t nodes = 0; for (auto*th : *this) nodes += th->nodes.load(std::memory_order_relaxed); return nodes; }

	// main()以外のスレッド
	Slaves slaves;

	// 探索中にこれがtrueになったら探索を即座に終了すること。
	std::atomic_bool stop;

	// USIプロトコルで指定されているスレッド数を反映させる。
	void read_usi_options();
};

// ThreadPoolのglobalな実体
extern ThreadPool Threads;

#endif // _THREAD_H_
