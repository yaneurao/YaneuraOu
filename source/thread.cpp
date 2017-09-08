#include "thread.h"

ThreadPool Threads;		// Global object

namespace {

	// std::thread派生型であるT型のthreadを一つ作って、そのidle_loopを実行するためのマクロ。
	// Threadクラスがalignされていることを要求するのでこうやって生成している。
	// (C++のnewがalignasを無視するのがおかしいのだが…)
	// 生成されたスレッドはidle_loop()で仕事が来るのを待機している。
	template<typename T> T* new_thread(size_t n) {
		void* dst = aligned_malloc(sizeof(T), alignof(T));
		// 確保に成功したならゼロクリアしておく。
		if (dst)
			std::memset(dst, 0, sizeof(T));

		T* th = new (dst) T(n);
		return (T*)th;
	}

	// new_thread()の逆。スレッド解体時に呼び出される。
	void delete_thread(Thread *th) {
		th->~Thread();
		aligned_free(th);
	}
}

Thread::Thread(size_t n) : idx(n)
{
	exit = false;

	// selDepthを更新しない思考エンジンでseldepthの出力がおかしくなるのを防止するために
	// ここでとりあえず初期化しておいてやる。
	selDepth = 0;

	// 探索したノード数
	nodes = 0;

	// スレッドを一度起動してworkerのほう待機状態にさせておく
	std::unique_lock<Mutex> lk(mutex);
	searching = true;
	stdThread = std::thread(&Thread::idle_loop, this);
	cv.wait(lk, [&] {return !searching; });
}

// std::threadの終了を待つ
Thread::~Thread() {

	mutex.lock();
	exit = true; // 探索は終わっているはずなのでこのフラグをセットして待機する。
	cv.notify_one();
	mutex.unlock();
	stdThread.join();
}


// 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)
void Thread::wait_for_search_finished()
{
	std::unique_lock<Mutex> lk(mutex);
	cv.wait(lk, [&] { return !searching; });
}

// 探索するときのmaster,slave用のidle_loop。探索開始するまで待っている。
void Thread::idle_loop() {

	WinProcGroup::bindThisThread(idx);

	while (!exit)
	{
		std::unique_lock<Mutex> lk(mutex);

		searching = false;

		while (!searching && !exit)
		{
			cv.notify_one(); // 他のスレッドがこのスレッドを待機待ちしてるならそれを起こす
			cv.wait(lk);
		}

		if (exit)
			return;

		lk.unlock();

		// exit == falseということはsearch == trueというわけだから探索する。
		search();
	}
}

// MainThreadを一つ生成して、そのあとrequestedで要求された分だけスレッドを生成する。(MainThreadも含めて数える)
void ThreadPool::init(size_t requested)
{
	push_back(new_thread<MainThread>(0));
	set(requested);
}

void ThreadPool::exit()
{
	// 探索の終了を待つ
	main()->wait_for_search_finished();
	set(0);
}

// スレッド数を変更する。
void ThreadPool::set(size_t requested)
{
	// スレッドが足りなければ生成
	while (size() < requested)
		push_back(new_thread<Thread>(size()));

	// スレッドが余っていれば解体
	while (size() > requested)
		delete_thread(back()), pop_back();
}

void ThreadPool::start_thinking(const Position& pos, Search::StateStackPtr& states , const Search::LimitsType& limits)
{
	// 思考中であれば停止するまで待つ。
	main()->wait_for_search_finished();

	// ponderに関して、StockfishではstopOnPonderhitというのがあるが、やねうら王にはこのフラグはない。
	Threads.stop = false;

	Search::Limits = limits;

	Search::RootMoves rootMoves;

	// 初期局面では合法手すべてを生成してそれをrootMovesに設定しておいてやる。
	// このとき、歩の不成などの指し手は除く。(そのほうが勝率が上がるので)
	// また、goコマンドでsearchmovesが指定されているなら、そこに含まれていないものは除く。
	
	// あと宣言勝ちできるなら、その指し手を先頭に入れておいてやる。
	// (ただし、トライルールのときはMOVE_WINではないので、トライする指し手はsearchmovesに含まれていなければ
	// 指しては駄目な手なのでrootMovesに追加しない。)
	if (pos.DeclarationWin() == MOVE_WIN)
		rootMoves.push_back(Search::RootMove(MOVE_WIN));

	for (auto m : MoveList<LEGAL>(pos))
		if (limits.searchmoves.empty()
			|| std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
			rootMoves.push_back(Search::RootMove(m));

	// statesが呼び出し元から渡されているならこの所有権をSearch::SetupStatesに移しておく。
	if (states.get())
		Search::SetupStates = std::move(states);

	// Position::set()によってst->previosがクリアされるので事前にコピーして保存する。
	StateInfo tmp = Search::SetupStates->top();

	auto sfen = pos.sfen();
	for (auto th : *this)
	{
		th->nodes = 0;
		th->rootDepth = th->completedDepth = DEPTH_ZERO;
		th->rootMoves = rootMoves;
		th->rootPos.set(sfen,th);
	}

	// Position::set()によってクリアされていた、st->previousを復元する。
	Search::SetupStates->top() = tmp;

	main()->start_searching();
}
