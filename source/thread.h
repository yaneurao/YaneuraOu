#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
//#include <cstddef>
//#include <cstdint>
#include <mutex>
#include <vector>

#include "movepick.h"
#include "position.h"
#include "search.h"
#include "thread_win32_osx.h"
//#include "types.h"

#if defined(EVAL_LEARN)
// 学習用の実行ファイルでは、スレッドごとに置換表を持ちたい。
#include "tt.h"
#endif

// --------------------
// 探索時に用いるスレッド
// --------------------

// 探索時に用いる、それぞれのスレッド
// これを思考スレッド数だけ確保する。
// ただしメインスレッドはこのclassを継承してMainThreadにして使う。
class Thread
{
	// exitフラグやsearchingフラグの状態を変更するときのmutex
	std::mutex mutex;

	// idle_loop()で待機しているときに待つ対象
	std::condition_variable cv;

	// thread id。main threadなら0。slaveなら1から順番に値が割当てられる。
	size_t idx;

	// exit      : このフラグが立ったら終了する。
	// searching : 探索中であるかを表すフラグ。プログラムを簡素化するため、事前にtrueにしてある。
	bool exit = false , searching = true;

	// stack領域を増やしたstd::thread
	NativeThread stdThread;

public:

	// ThreadPoolで何番目のthreadであるかをコンストラクタで渡すこと。この値は、idx(スレッドID)となる。
	explicit Thread(size_t n);
	virtual ~Thread();

	// slaveは、main threadから
	//   for(auto th : Threads) th->start_searching();
	// のようにされるとこの関数が呼び出される。
	// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
	// この関数を呼び出したいときは、Thread::search()とすること。
	virtual void search();

	// このクラスが保持している探索で必要なテーブル(historyなど)をクリアする。
	void clear();

	// スレッド起動後、この関数が呼び出される。
	void idle_loop();

	// ------------------------------
	//      同期待ちのwait等
	// ------------------------------

	// Thread::search()を開始させるときに呼び出す。
	void start_searching();

	// 探索が終わるのを待機する。(searchingフラグがfalseになるのを待つ)
	void wait_for_search_finished();

	// Threadの自身のスレッド番号を返す。0 origin。
	size_t id() const { return idx; }

	// === やねうら王独自拡張 ===

	// 探索中であるかを返す。
	bool is_searching() const { return searching; }

	// ------------------------------
	//       探索に必要なもの
	// ------------------------------

	// pvIdx    : このスレッドでMultiPVを用いているとして、rootMovesの(0から数えて)何番目のPVの指し手を
	//      探索中であるか。MultiPVでないときはこの変数の値は0。
	// pvLast   : tbRank絡み。将棋では関係ないので用いない。
	size_t pvIdx /*,pvLast*/;

	// nodes     : このスレッドが探索したノード数(≒Position::do_move()を呼び出した回数)
	// bestMoveChanges : 反復深化においてbestMoveが変わった回数。nodeの安定性の指標として用いる。全スレ分集計して使う。
	std::atomic<uint64_t> nodes,/* tbHits,*/ bestMoveChanges;

	// selDepth  : rootから最大、何手目まで探索したか(選択深さの最大)
	// nmpMinPly : null moveの前回の適用ply
	// nmpColor  : null moveの前回の適用Color
	// state     : 探索で組合せ爆発が起きているか等を示す状態
	int selDepth, nmpMinPly;

	// bestValue :
	// search()で、そのnodeでbestMoveを指したときの(探索の)評価値
	// Stockfishではevaluate()の遅延評価のためにThreadクラスに持たせることになった。
	// cf. Reduce use of lazyEval : https://github.com/official-stockfish/Stockfish/commit/7b278aab9f61620b9dba31896b38aeea1eb911e2
	// optimism  : 楽観値
	// → やねうら王では導入せず
	Value bestValue /*, optimism[COLOR_NB]*/ ;

	// 探索開始局面
	Position rootPos;

	// rootでのStateInfo
	// Position::set()で書き換えるのでスレッドごとに保持していないといけない。
	StateInfo rootState;

	// 探索開始局面で思考対象とする指し手の集合。
	// goコマンドで渡されていなければ、全合法手(ただし歩の不成などは除く)とする。
	Search::RootMoves rootMoves;

	// rootDepth      : 反復深化の深さ
	//					Lazy SMPなのでスレッドごとにこの変数を保有している。
	//
	// completedDepth : このスレッドに関して、終了した反復深化の深さ
	//
	Depth rootDepth, completedDepth;

#if defined(__EMSCRIPTEN__)
	// yaneuraou.wasm
	std::atomic_bool threadStarted;
#endif

	// aspiration searchのrootでの beta - alpha
	Value rootDelta;

	// ↓Stockfishでは思考開始時に評価関数から設定しているが、やねうら王では使っていないのでコメントアウト。
	//Value rootSimpleEval;

#if defined(USE_MOVE_PICKER)
	// 近代的なMovePickerではオーダリングのために、スレッドごとにhistoryとcounter movesなどのtableを持たないといけない。
	CounterMoveHistory counterMoves;
	ButterflyHistory mainHistory;
	CapturePieceToHistory captureHistory;

	// コア数が多いか、長い持ち時間においては、ContinuationHistoryもスレッドごとに確保したほうが良いらしい。
	// cf. https://github.com/official-stockfish/Stockfish/commit/5c58d1f5cb4871595c07e6c2f6931780b5ac05b5
	// 添字の[2][2]は、[inCheck(王手がかかっているか)][capture_stage]
	// →　この改造、レーティングがほぼ上がっていない。悪い改造のような気がする。
	ContinuationHistory continuationHistory[2][2];

#if defined(ENABLE_PAWN_HISTORY)
	PawnHistory pawnHistory;
#endif

#endif

	// Stockfish10ではスレッドごとにcontemptを保持するように変わった。
	//Score contempt;

	// ------------------------------
	//   やねうら王、独自追加
	// ------------------------------

	// スレッドidが返る。Stockfishにはないメソッドだが、
	// スレッドごとにメモリ領域を割り当てたいときなどに必要となる。
	// MainThreadなら0、slaveなら1,2,3,...
	size_t thread_id() const { return idx; }

#if defined(EVAL_LEARN)
	// 学習用の実行ファイルでは、スレッドごとに置換表を持ちたい。
	TranspositionTable tt;
#endif
};


// 探索時のmainスレッド(これがmasterであり、これ以外はslaveとみなす)
struct MainThread: public Thread
{
	// constructorはThreadのものそのまま使う。
	using Thread::Thread;

	// 探索を開始する時に呼び出される。
	void search() override;

	// Thread::search()を呼び出す。
	// ※　Stockfish、MainThreadがsearch()をoverrideする設計になっているの、良くないと思う。
	//     そのため、MainThreadに対して外部からThread::search()を呼び出させることが出来ない。
	//     仕方ないのでこれを回避するために抜け道を用意しておく。
	void thread_search() { Thread::search(); }

	// 思考時間の終わりが来たかをチェックする。
	void check_time();

	// previousTimeReduction : 反復深化の前回のiteration時のtimeReductionの値。
	double previousTimeReduction;

	// 前回の探索時のスコアとその平均。
	// 次回の探索のときに何らか使えるかも。
	Value bestPreviousScore;
	Value bestPreviousAverageScore;

	// 時間まぎわのときに探索を終了させるかの判定に用いるための、
	// 反復深化のiteration、前4回分のScore
	Value iterValue[4];

	// check_time()で用いるカウンター。
	// デクリメントしていきこれが0になるごとに思考をストップするのか判定する。
	int callsCnt;

	//bool stopOnPonderhit;
	// →　やねうら王では、このStockfishのponderの仕組みを使わない。(もっと上手にponderの時間を活用したいため)

	// ponder : "go ponder" コマンドでの探索中であるかを示すフラグ
	std::atomic_bool ponder;

	// -------------------
	// やねうら王独自追加
	// -------------------

	// 将棋所のコンソールが詰まるので出力を抑制するために、前回の出力時刻を
	// 記録しておき、そこから一定時間経過するごとに出力するという方式を採る。
	TimePoint lastPvInfoTime;

	// Ponder用の指し手
	// Stockfishは置換表からponder moveをひねり出すコードになっているが、
	// 前回iteration時のPVの2手目の指し手で良いのではなかろうか…。
	Move ponder_candidate;

	// "Position"コマンドで1つ目に送られてきた文字列("startpos" or sfen文字列)
	std::string game_root_sfen;

	// "Position"コマンドで"moves"以降にあった、rootの局面からこの局面に至るまでの手順
	std::vector<Move> moves_from_game_root;

	// Stochastic Ponderのときに↑を2手前に戻すので元の"position"コマンドと"go"コマンドの文字列を保存しておく。
	std::string last_position_cmd_string = "position startpos";
	std::string last_go_cmd_string;
	// Stochastic Ponderのために2手前に戻してしまっているかのフラグ
	bool position_is_dirty = false;

	// goコマンドの"wait_stop"フラグと関連して、↓と出力したかのフラグ。
	// "info string time to return bestmove."
	bool time_to_return_bestmove;
};


// 思考で用いるスレッドの集合体
// 継承はあまり使いたくないが、for(auto* th:Threads) ... のようにして回せて便利なのでこうしてある。
//
// このクラスにコンストラクタとデストラクタは存在しない。
// Threads(スレッドオブジェクト)はglobalに配置するし、スレッドの初期化の際には
// スレッドが保持する思考エンジンが使う変数等がすべてが初期化されていて欲しいからである。
// スレッドの生成はset(options["Threads"])で行い、スレッドの終了はset(0)で行なう。
struct ThreadPool
{
	// mainスレッドに思考を開始させる。
	void start_thinking(const Position& pos, StateListPtr& states , const Search::LimitsType& limits , bool ponderMode = false);

	// set()で生成したスレッドの初期化
	void clear();

	// スレッド数を変更する。
	// 終了時は明示的にset(0)として呼び出すこと。(すべてのスレッドの終了を待つ必要があるため)
	void set(size_t requested);

	// mainスレッドを取得する。これはthis[0]がそう。
	MainThread* main() const { return static_cast<MainThread*>(threads.front()); }

	// 今回、goコマンド以降に探索したノード数
	// →　これはPosition::do_move()を呼び出した回数。
	// ※　dlshogiエンジンで、探索ノード数が知りたい場合は、
	// 　dlshogi::nodes_visited()を呼び出すこと。
	uint64_t nodes_searched() { return accumulate(&Thread::nodes); }

	// 探索終了時に、一番良い探索ができていたスレッドを選ぶ。
	Thread* get_best_thread() const;

	// 探索を開始する(main thread以外)
	void start_searching();

	// main threadがそれ以外の探索threadの終了を待つ。
	void wait_for_search_finished() const;

	// stop          : 探索中にこれがtrueになったら探索を即座に終了すること。
	// increaseDepth : 一定間隔ごとに反復深化の探索depthが増えて行っているかをチェックするためのフラグ
	//                 増えて行ってないなら、同じ深さを再度探索するのに用いる。
	std::atomic_bool stop , increaseDepth;

	auto cbegin() const noexcept { return threads.cbegin(); }
	auto begin() noexcept { return threads.begin(); }
	auto end() noexcept { return threads.end(); }
	auto cend() const noexcept { return threads.cend(); }
	auto size() const noexcept { return threads.size(); }
	auto empty() const noexcept { return threads.empty(); }
	// thread_pool[n]のようにでアクセスしたいので…。
	auto operator[](size_t i) const noexcept { return threads[i];}

	// === やねうら王独自拡張 ===

	// main thread以外の探索スレッドがすべて終了しているか。
	// すべて終了していればtrueが返る。
	bool search_finished() const;

private:

	// 現局面までのStateInfoのlist
	StateListPtr setupStates;

	// vector<Thread*>からこのclassを継承させるのはやめて、このメンバーとして持たせるようにした。
	std::vector<Thread*> threads;

	// Threadクラスの特定のメンバー変数を足し合わせたものを返す。
	uint64_t accumulate(std::atomic<uint64_t> Thread::* member) const {

		uint64_t sum = 0;
		for (Thread* th : threads)
			sum += (th->*member).load(std::memory_order_relaxed);
		return sum;
	}
};

// ThreadPoolのglobalな実体
extern ThreadPool Threads;

#endif // #ifndef THREAD_H_INCLUDED

