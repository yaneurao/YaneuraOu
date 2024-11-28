#include <algorithm> // std::count
#include <cmath>     // std::abs
#include <unordered_map>

#include "thread.h"
#include "usi.h"
#include "tt.h"

ThreadPool Threads;		// Global object

Thread::Thread(size_t n) : idx(n) , stdThread(&Thread::idle_loop, this)
{
#if !defined(__EMSCRIPTEN__)
	// スレッドはsearching == trueで開始するので、このままworkerのほう待機状態にさせておく
	wait_for_search_finished();
#else
	// yaneuraou.wasm
	// wait_for_search_finished すると、ブラウザのメインスレッドをブロックしデッドロックが発生するため、コメントアウト。
	//
	// 新しいスレッドが cv を設定するのを待ってから、ブラウザに処理をパスしたいが、
	// 新しいスレッド用のworkerを作成するためには、いったんブラウザに処理をパスする必要がある。
	//
	// https://bugzilla.mozilla.org/show_bug.cgi?id=1049079
	//
	// threadStarted という変数を設けて全てのスレッドが開始するまでリトライするようにする
	//
	// 参考：https://github.com/lichess-org/stockfish.wasm/blob/a022fa1405458d1bc1ba22fe813bace961859102/src/thread.cpp#L38
#endif
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
#if defined(USE_MOVE_PICKER)
	mainHistory.fill(0);
	captureHistory.fill(-758);
#if defined(ENABLE_PAWN_HISTORY)
	pawnHistory.fill(-1158);
	pawnCorrectionHistory.fill(0);
	materialCorrectionHistory.fill(0);
	majorPieceCorrectionHistory.fill(0);
	minorPieceCorrectionHistory.fill(0);
	nonPawnCorrectionHistory[WHITE].fill(0);
	nonPawnCorrectionHistory[BLACK].fill(0);

	for (auto& to : continuationCorrectionHistory)
		for (auto& h : to)
			h->fill(0);
#endif

	// ここは、未初期化のときに[NO_PIECE][SQ_ZERO]を指すので、ここを-1で初期化しておくことによって、
	// history > 0 を条件にすれば自ずと未初期化のときは除外されるようになる。

	// ほとんどの履歴エントリがいずれにせよ後で負になるため、
	// 開始値を「正しい」方向に少しシフトさせるため、-71で埋めている。
	// この効果は、深度が深くなるほど薄れるので、長時間思考させる時には
	// あまり意味がないが、無駄ではないらしい。
	// Tweak history initialization : https://github.com/official-stockfish/Stockfish/commit/7d44b43b3ceb2eebc756709432a0e291f885a1d2

	for (bool inCheck : { false, true })
		for (StatsType c : { NoCaptures, Captures })
			//for (auto& to : continuationHistory[inCheck][c])
			//	for (auto& h : to)
			//		h->fill(-675);

			// ↑この初期化コードは、ContinuationHistory::fill()に移動させた。

			continuationHistory[inCheck][c].fill(-645);

#endif
}

// 待機していたスレッドを起こして探索を開始させる
void Thread::start_searching()
{
	mutex.lock();
	searching = true;
    mutex.unlock(); // Unlock before notifying saves a few CPU-cycles
	cv.notify_one(); // idle_loop()で回っているスレッドを起こす。(次の処理をさせる)
}

// 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)
void Thread::wait_for_search_finished()
{
	std::unique_lock<std::mutex> lk(mutex);
	cv.wait(lk, [&] { return !searching; });
}

// 探索するときのmaster,slave用のidle_loop。探索開始するまで待っている。
void Thread::idle_loop() {

	// NUMA環境では、8スレッド未満だと異なるNUMAに割り振られることがあり、パフォーマンス激減するのでその回避策。
	// ・8スレッド未満のときはOSに任せる
	// ・8スレッド以上のときは、自前でbindThisThreadを行なう。
	// cf. Upon changing the number of threads, make sure all threads are bound : https://github.com/official-stockfish/Stockfish/commit/1c50d8cbf554733c0db6ab423b413d75cc0c1928
	// その後、
	// ・fishtestを8スレッドで行うから、9以上にしてくれとのことらしい。
	// cf. NUMA for 9 threads or more : https://github.com/official-stockfish/Stockfish/commit/bc3b148d5712ef9ea00e74d3ff5aea10a4d3cabe

#if !defined(FORCE_BIND_THIS_THREAD)
	// "Threads"というオプションがない時は、強制的にbindThisThread()しておいていいと思う。(使うスレッド数がここではわからないので..)
	if (Options.count("Threads")==0 || Options["Threads"] > 8)
#endif
		WinProcGroup::bindThisThread(idx);
		// このifを有効にすると何故かNUMA環境のマルチスレッド時に弱くなることがある気がする。
		// (長い時間対局させ続けると安定するようなのだが…)
		// 上の投稿者と条件が何か違うのだろうか…。
		// 前のバージョンのソフトが、こちらのNUMAの割当を阻害している可能性が微レ存。

	while (true)
	{
		std::unique_lock<std::mutex> lk(mutex);
		searching = false;
#if defined(__EMSCRIPTEN__)
		// yaneuraOu.wasm
		threadStarted = true;
#endif
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
	// bindThreadの都合があるので、スレッドをいったんすべて解体して、再度作り直す。
	// ただし、__EMSCRIPTEN__の時は、スレッド数がrequestedと同じならスレッドを作り直さず、
	// また、いったんすべて解放する処理も端折る。


#if defined(__EMSCRIPTEN__)
	// yaneuraou.wasm
	// ブラウザのメインスレッドをブロックしないようstockfish.wasmと同様の実装に修正
	if (size() == requested)
		return;
#endif

	if (threads.size() > 0) { // いったんすべてのスレッドを解体(NUMA対策)
		main()->wait_for_search_finished();

#if !defined(__EMSCRIPTEN__)
		while (threads.size() > 0)
			delete threads.back(), threads.pop_back();
#else
		// yaneuraou.wasm
		while (threads.size() > requested)
			delete threads.back(), threads.pop_back();
#endif
	}

	if (requested > 0) { // 要求された数だけのスレッドを生成
#if !defined(__EMSCRIPTEN__)
		threads.push_back(new MainThread(0));

		while (size() < requested)
			threads.push_back(new Thread(size()));
#else
		// yaneuraou.wasm
		while (size() < requested)
			threads.push_back(size() ? new Thread(size()) : new MainThread(0));
#endif
		clear();

		// Reallocate the hash with the new threadpool size
		// 新しいスレッドプールのサイズで置換表用のメモリを再確保する。

		// →　大きなメモリの置換表だと確保に時間がかかるのでやりたくない。
		// 　　isreadyの応答でやるべき。

		//TT.resize(size_t(Options["USI_Hash"]));
	}

#if defined(EVAL_LEARN)
	// 学習用の実行ファイルでは、スレッド数が変更になったときに各ThreadごとのTTに
	// メモリを再割り当てする必要がある。
	TT.init_tt_per_thread();
#endif

}

// ThreadPool::clear()は、threadPoolのデータを初期値に設定する。
void ThreadPool::clear() {

	for (Thread* th : threads)
		th->clear();

	main()->callsCnt = 0;
	main()->bestPreviousScore        = VALUE_INFINITE;
	main()->bestPreviousAverageScore = VALUE_INFINITE;
	main()->previousTimeReduction    = 1.0;
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
	increaseDepth = true;
	main()->ponder = ponderMode;
	main()->time_to_return_bestmove = false;
	Search::Limits = limits;
	Search::RootMoves rootMoves;

	// 初期局面では合法手すべてを生成してそれをrootMovesに設定しておいてやる。
	// このとき、歩の不成などの指し手は除く。(そのほうが勝率が上がるので)
	// また、goコマンドでsearchmovesが指定されているなら、そこに含まれていないものは除く。

	// あと宣言勝ちできるなら、その指し手を先頭に入れておいてやる。
	// (ただし、トライルールのときはMOVE_WINではないので、トライする指し手はsearchmovesに含まれていなければ
	// 指しては駄目な手なのでrootMovesに追加しない。)
#if defined (USE_ENTERING_KING_WIN)
	if (pos.DeclarationWin() == Move::win())
		rootMoves.emplace_back(Move::win());
#endif

	// 全合法手を生成するオプションが有効ならば。
	if (limits.generate_all_legal_moves)
	{
		for (auto m : MoveList<LEGAL_ALL>(pos))
			if (limits.searchmoves.empty()
				|| std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
				rootMoves.emplace_back(m);

	} else {

		for (auto m : MoveList<LEGAL>(pos))
			if (limits.searchmoves.empty()
				|| std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
				rootMoves.emplace_back(m);
	}

	// After ownership transfer 'states' becomes empty, so if we stop the search
	// and call 'go' again without setting a new position states.get() == nullptr.
	// 所有権の移動後、statesが空になるので、探索を停止させ、
	// "go"をstate.get() == nullptrである新しいpositionをセットせずに再度呼び出す。

	ASSERT_LV3(states.get() || setupStates.get());

	// statesが呼び出し元から渡されているならこの所有権をSearch::SetupStatesに移しておく。
	// このstatesは、positionコマンドに対して用いたStateInfoでなければならない。(CheckInfoが異なるため)
	// 引数で渡されているstatesは、そうなっているものとする。
	if (states.get())
		setupStates = std::move(states);	// Ownership transfer, states is now empty

	// We use Position::set() to set root position across threads. But there are
	// some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
	// be deduced from a fen string, so set() clears them and they are set from
	// setupStates->back() later. The rootState is per thread, earlier states are shared
	// since they are read-only.

	// Position::set()によってst->previosがクリアされるので事前にコピーして保存する。
	// これは、rootStateの役割。これはスレッドごとに持っている。
	// cf. Fix incorrect StateInfo : https://github.com/official-stockfish/Stockfish/commit/232c50fed0b80a0f39322a925575f760648ae0a5

	auto sfen = pos.sfen();
	for (Thread* th : *this)
	{
		// th->nodes = th->tbHits = th->nmpMinPly = th->bestMoveChanges = 0;
		// Stockfish12のこのコード、bestMoveChangesがatomic型なのでそこからint型に代入してることになってコンパイラが警告を出す。
		// ↓のように書いたほうが良い。
		th->nodes = th->bestMoveChanges = /* th->tbHits = */ th->nmpMinPly = 0;

		th->rootDepth = th->completedDepth = 0;

		// 以上の初期化、探索スレッド側でやるべきだと思うが、しかしth->nodesなどはmain threadが探索ノード数の
		// 出力のために積算するので、main threadが積算する時にはすでに他のスレッドのth->nodesがゼロ初期化されている状態でないと
		// おかしい値になってしまう。
		//
		// そこで、main threadが開始する時点では、すべてのスレッドのスレッド初期化が完了していることを保証しなければならない。
		// ゆえにここに書くしかないのである。

		th->rootMoves = rootMoves;
		th->rootPos.set(sfen, &th->rootState,th);
		th->rootState = setupStates->back();
	    //th->rootSimpleEval = Eval::simple_eval(pos, pos.side_to_move());
		// →　やねうら王では使っていない。
	}

	main()->start_searching();
}


// 探索終了時に、一番良い探索ができていたスレッドを選ぶ。
Thread* ThreadPool::get_best_thread() const {

	// 深くまで探索できていて、かつそっちの評価値のほうが優れているならそのスレッドの指し手を採用する
	// 単にcompleteDepthが深いほうのスレッドを採用しても良さそうだが、スコアが良いほうの探索深さのほうが
	// いい指し手を発見している可能性があって楽観合議のような効果があるようだ。

	Thread* bestThread = threads.front();

	std::unordered_map<Move, int64_t, Move::MoveHash> votes(
		2 * std::min(size(), bestThread->rootMoves.size()));

	Value minScore = VALUE_NONE;

	// Find minimum score of all threads
	for (Thread* th : threads)
		minScore = std::min(minScore, th->rootMoves[0].score);

	// Vote according to score and depth, and select the best thread
    auto thread_value = [minScore](Thread* th) {
            return (th->rootMoves[0].score - minScore + 14) * int(th->completedDepth);
        };

    for (Thread* th : threads)
        votes[th->rootMoves[0].pv[0]] += thread_value(th);

    for (Thread* th : threads)
        if (std::abs(bestThread->rootMoves[0].score) >= VALUE_TB_WIN_IN_MAX_PLY)
        {
            // Make sure we pick the shortest mate / TB conversion or stave off mate the longest
            if (th->rootMoves[0].score > bestThread->rootMoves[0].score)
                bestThread = th;
        }
        else if (   th->rootMoves[0].score >= VALUE_TB_WIN_IN_MAX_PLY
                 || (   th->rootMoves[0].score > VALUE_TB_LOSS_IN_MAX_PLY
                     && (   votes[th->rootMoves[0].pv[0]] > votes[bestThread->rootMoves[0].pv[0]]
                         || (   votes[th->rootMoves[0].pv[0]] == votes[bestThread->rootMoves[0].pv[0]]
                             &&   thread_value(th) * int(th->rootMoves[0].pv.size() > 2)
                                > thread_value(bestThread) * int(bestThread->rootMoves[0].pv.size() > 2)))))
            bestThread = th;

    return bestThread;
}

/// Start non-main threads
// 探索を開始する(main thread以外)

void ThreadPool::start_searching() {

	for (Thread* th : threads)
		if (th != threads.front())
			th->start_searching();
}


/// Wait for non-main threads
// main threadがそれ以外の探索threadの終了を待つ。

void ThreadPool::wait_for_search_finished() const {

	for (Thread* th : threads)
		if (th != threads.front())
			th->wait_for_search_finished();
}

// main thread以外の探索スレッドがすべて終了しているか。
// すべて終了していればtrueが返る。
bool ThreadPool::search_finished() const
{
	for (Thread* th : threads)
		if (th != threads.front())
			if (th->is_searching())
				return false;

	return true;
}
