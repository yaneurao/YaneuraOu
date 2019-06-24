#include "thread.h"
#include "usi.h"

ThreadPool Threads;		// Global object

void* Thread::operator new(size_t s) { return aligned_malloc(s, alignof(Thread)); }
void Thread::operator delete(void*p) noexcept { aligned_free(p); }

Thread::Thread(size_t n) : idx(n) , stdThread(&Thread::idle_loop, this)
{
	// スレッドはsearching == trueで開始するので、このままworkerのほう待機状態にさせておく
	wait_for_search_finished();
}

// std::threadの終了を待つ
Thread::~Thread()
{
	// 探索中にスレッドオブジェクトが解体されることはない。
	ASSERT_LV3(!searching);

	// 探索は終わっているのでexitフラグをセットしてstart_searching()を呼べば終了するはず。
	exit = true;
	start_searching();
	stdThread.join();
}

// このクラスが保持している探索で必要なテーブル(historyなど)をクリアする。
void Thread::clear()
{
	counterMoves.fill(MOVE_NONE);
	mainHistory.fill(0);
	captureHistory.fill(0);

	// ここは、未初期化のときに[SQ_ZERO][NO_PIECE]を指すので、ここを-1で初期化しておくことによって、
	// history > 0 を条件にすれば自ずと未初期化のときは除外されるようになる。
	for (auto& to : continuationHistory)
		for (auto& h : to)
			h->fill(0);

	continuationHistory[SQ_ZERO][NO_PIECE]->fill(Search::CounterMovePruneThreshold - 1);
}

// 待機していたスレッドを起こして探索を開始させる
void Thread::start_searching()
{
	std::lock_guard<Mutex> lk(mutex);
	searching = true;
	cv.notify_one(); // idle_loop()で回っているスレッドを起こす。(次の処理をさせる)
}

// 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)
void Thread::wait_for_search_finished()
{
	std::unique_lock<Mutex> lk(mutex);
	cv.wait(lk, [&] { return !searching; });
}

// 探索するときのmaster,slave用のidle_loop。探索開始するまで待っている。
void Thread::idle_loop() {

	// NUMA環境では、8スレッド未満だと異なるNUMAに割り振られることがあり、パフォーマンス激減するのでその回避策。
	// ・8スレッド未満のときはOSに任せる
	// ・8スレッド以上のときは、自前でbindThisThreadを行なう。
	// cf. Upon changing the number of threads, make sure all threads are bound : https://github.com/official-stockfish/Stockfish/commit/1c50d8cbf554733c0db6ab423b413d75cc0c1928

	if (Options["Threads"] >= 8)
		WinProcGroup::bindThisThread(idx);
		// このifを有効にすると何故かNUMA環境のマルチスレッド時に弱くなることがある気がする。
		// (長い時間対局させ続けると安定するようなのだが…)
		// 上の投稿者と条件が何か違うのだろうか…。
		// 前のバージョンのソフトが、こちらのNUMAの割当を阻害している可能性が微レ存。

	while (true)
	{
		std::unique_lock<Mutex> lk(mutex);
		searching = false;
		cv.notify_one(); // 他のスレッドがこのスレッドを待機待ちしてるならそれを起こす
		cv.wait(lk, [&] { return searching; });

		if (exit)
			return;

		lk.unlock();

		// exit == falseということはsearch == trueというわけだから探索する。
		search();
	}
}

// スレッド数を変更する。
void ThreadPool::set(size_t requested)
{
	if (size() > 0) { // いったんすべてのスレッドを解体(NUMA対策)
		main()->wait_for_search_finished();

		while (size() > 0)
			delete back(), pop_back();
	}

	if (requested > 0) { // 要求された数だけのスレッドを生成
		push_back(new MainThread(0));

		while (size() < requested)
			push_back(new Thread(size()));
		clear();

		// Reallocate the hash with the new threadpool size
		//TT.resize(Options["Hash"]);

		// →　新しいthreadpoolのサイズで置換表用のメモリを確保しなおしたほうが
		//  良いらしいのだが、大きなメモリの置換表だと確保に時間がかかるのでやりたくない。
	}
}

// ThreadPool::clear()は、threadPoolのデータを初期値に設定する。
void ThreadPool::clear() {

	for (Thread* th : *this)
		th->clear();

	main()->callsCnt = 0;
	main()->previousScore = VALUE_INFINITE;
	main()->previousTimeReduction = 1.0;
}

// ilde_loop()で待機しているmain threadを起こして即座にreturnする。
// main threadは他のスレッドを起こして、探索を開始する。
void ThreadPool::start_thinking(const Position& pos, StateListPtr& states ,
								const Search::LimitsType& limits , bool ponderMode)
{
	// 思考中であれば停止するまで待つ。
	main()->wait_for_search_finished();

	// ponderに関して、StockfishではstopOnPonderhitというのがあるが、やねうら王にはこのフラグはない。
	/* main()->stopOnPonderhit = */ stop = false;
	main()->ponder = ponderMode;
	Search::Limits = limits;
	Search::RootMoves rootMoves;

	// 初期局面では合法手すべてを生成してそれをrootMovesに設定しておいてやる。
	// このとき、歩の不成などの指し手は除く。(そのほうが勝率が上がるので)
	// また、goコマンドでsearchmovesが指定されているなら、そこに含まれていないものは除く。
	
	// あと宣言勝ちできるなら、その指し手を先頭に入れておいてやる。
	// (ただし、トライルールのときはMOVE_WINではないので、トライする指し手はsearchmovesに含まれていなければ
	// 指しては駄目な手なのでrootMovesに追加しない。)
#if defined (USE_ENTERING_KING_WIN)
	if (pos.DeclarationWin() == MOVE_WIN)
		rootMoves.emplace_back(MOVE_WIN);
#endif

#if !defined(MATE_ENGINE) && !defined(FOR_TOURNAMENT) 
	// 全合法手を生成するオプションが有効ならば。
	if (limits.generate_all_legal_moves)
	{
		for (auto m : MoveList<LEGAL_ALL>(pos))
			if (limits.searchmoves.empty()
				|| std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
				rootMoves.emplace_back(m);
	} else
#endif
	{   // トーナメントモードなら、歩の不成は生成しない。
		for (auto m : MoveList<LEGAL>(pos))
			if (limits.searchmoves.empty()
				|| std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
				rootMoves.emplace_back(m);
	}

	// 所有権の移動後、statesが空になるので、探索を停止させ、
	// "go"をstate.get() == NULLである新しいpositionをセットせずに再度呼び出す。
	ASSERT_LV3(states.get() || setupStates.get());

	// statesが呼び出し元から渡されているならこの所有権をSearch::SetupStatesに移しておく。
	// このstatesは、positionコマンドに対して用いたStateInfoでなければならない。(CheckInfoが異なるため)
	// 引数で渡されているstatesは、そうなっているものとする。
	if (states.get())
		setupStates = std::move(states);

	// Position::set()によってst->previosがクリアされるので事前にコピーして保存する。
	// cf. Fix incorrect StateInfo : https://github.com/official-stockfish/Stockfish/commit/232c50fed0b80a0f39322a925575f760648ae0a5
	StateInfo tmp = setupStates->back();

	auto sfen = pos.sfen();
	for (Thread* th : *this)
	{
		/* th->shuffleExts = */ th->nodes = /* th->tbHits = */ th->nmpMinPly = 0;
		th->rootDepth = th->completedDepth = DEPTH_ZERO;
		th->rootMoves = rootMoves;

		// setupStatesを渡して、これをコピーしておかないと局面を遡れない。
		th->rootPos.set(sfen, &setupStates->back(),th);
	}

#if defined(USE_FV_VAR)
	// Position::set()によって、dirtyPieceはevalListに反映されていることになるのだが、
	// 次にst->previousを復元してしまうと、その更新したことを示すフラグを潰してしまう。ここでフラグを立て直す。
	tmp.dirtyPiece.updated_ = true;
#endif

	// Position::set()によってクリアされていた、st->previousを復元する。
	setupStates->back() = tmp;

	main()->start_searching();
}
