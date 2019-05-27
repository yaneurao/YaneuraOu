#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "movepick.h"
#include "position.h"
#include "search.h"
#include "thread_win32.h"

// --------------------
// 探索時に用いるスレッド
// --------------------

// 探索時に用いる、それぞれのスレッド
// これを思考スレッド数だけ確保する。
// ただしメインスレッドはこのclassを継承してMainThreadにして使う。
class Thread
{
	// exitフラグやsearchingフラグの状態を変更するときのmutex
	Mutex mutex;

	// idle_loop()で待機しているときに待つ対象
	ConditionVariable cv;

	// thread id。main threadなら0。slaveなら1から順番に値が割当てられる。
	size_t idx;

	// exit      : このフラグが立ったら終了する。
	// searching : 探索中であるかを表すフラグ。プログラムを簡素化するため、事前にtrueにしてある。
	bool exit = false , searching = true;

	// wrapしているstd::thread
	std::thread stdThread;

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

	// ------------------------------
	//       探索に必要なもの
	// ------------------------------

	// pvIdx    : このスレッドでMultiPVを用いているとして、rootMovesの(0から数えて)何番目のPVの指し手を
	//      探索中であるか。MultiPVでないときはこの変数の値は0。
	// pvLast   : tbRank絡み。将棋では関係ないので用いない。
	size_t pvIdx /*,pvLast*/;

	// selDepth  : rootから最大、何手目まで探索したか(選択深さの最大)
	// nmpMinPly : null moveの前回の適用ply
	// nmpColor  : null moveの前回の適用Color
	int selDepth ,nmpMinPly;
	Color nmpColor;

	// nodes     : このスレッドが探索したノード数(≒Position::do_move()を呼び出した回数)
 	// bestMoveChanges : 反復深化においてbestMoveが変わった回数。nodeの安定性の指標として用いる。全スレ分集計して使う。
	std::atomic<uint64_t> nodes,/* tbHits,*/ bestMoveChanges;


	// 探索開始局面
	Position rootPos;

	// 探索開始局面で思考対象とする指し手の集合。
	// goコマンドで渡されていなければ、全合法手(ただし歩の不成などは除く)とする。
	Search::RootMoves rootMoves;

	// rootDepth      : 反復深化の深さ
	//					Lazy SMPなのでスレッドごとにこの変数を保有している。
	// 
	// completedDepth : このスレッドに関して、終了した反復深化の深さ
	//
	Depth rootDepth, completedDepth;

	// 近代的なMovePickerではオーダリングのために、スレッドごとにhistoryとcounter movesのtableを持たないといけない。
	CounterMoveHistory counterMoves;
	ButterflyHistory mainHistory;
	CapturePieceToHistory captureHistory;

	// コア数が多いか、長い持ち時間においては、ContinuationHistoryもスレッドごとに確保したほうが良いらしい。
	// cf. https://github.com/official-stockfish/Stockfish/commit/5c58d1f5cb4871595c07e6c2f6931780b5ac05b5
	ContinuationHistory continuationHistory;

	// Stockfish10ではスレッドごとにcontemptを保持するように変わった。
	//Score contempt;

	// ------------------------------
	//   やねうら王、独自追加
	// ------------------------------

	// PositionクラスのEvalListにalignasを指定されていて、Positionクラスを保持するこのThreadクラスをnewするが、
	// そのときにalignasを無視されるのでcustom allocatorを定義しておいてやる。
	void* operator new(std::size_t s);
	void operator delete(void*p) noexcept;

	// スレッドidが返る。Stockfishにはないメソッドだが、
	// スレッドごとにメモリ領域を割り当てたいときなどに必要となる。
	// MainThreadなら0、slaveなら1,2,3,...
	size_t thread_id() const { return idx; }

};
  

// 探索時のmainスレッド(これがmasterであり、これ以外はslaveとみなす)
struct MainThread: public Thread
{
	// constructorはThreadのものそのまま使う。
	using Thread::Thread;

	// 探索を開始する時に呼び出される。
	void search() override;
	
	// 思考時間の終わりが来たかをチェックする。
	void check_time();

	// previousTimeReduction : 反復深化の前回のiteration時のtimeReductionの値。
	double previousTimeReduction;

	// 前回の探索時のスコア。
	// 次回の探索のときに何らか使えるかも。
	Value previousScore;

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
};


// 思考で用いるスレッドの集合体
// 継承はあまり使いたくないが、for(auto* th:Threads) ... のようにして回せて便利なのでこうしてある。
//
// このクラスにコンストラクタとデストラクタは存在しない。
// Threads(スレッドオブジェクト)はglobalに配置するし、スレッドの初期化の際には
// スレッドが保持する思考エンジンが使う変数等がすべてが初期化されていて欲しいからである。
// スレッドの生成はset(options["Threads"])で行い、スレッドの終了はset(0)で行なう。
struct ThreadPool: public std::vector<Thread*>
{
	// mainスレッドに思考を開始させる。
	void start_thinking(const Position& pos, StateListPtr& states , const Search::LimitsType& limits , bool ponderMode = false);

	// set()で生成したスレッドの初期化
	void clear();

	// スレッド数を変更する。
	// 終了時は明示的にset(0)として呼び出すこと。(すべてのスレッドの終了を待つ必要があるため)
	void set(size_t requested);

	// mainスレッドを取得する。これはthis[0]がそう。
	MainThread* main() { return static_cast<MainThread*>(at(0)); }

	// 今回、goコマンド以降に探索したノード数
	uint64_t nodes_searched() { return accumulate(&Thread::nodes); }

	// stop   : 探索中にこれがtrueになったら探索を即座に終了すること。
	std::atomic_bool stop;
	
private:

	// 現局面までのStateInfoのlist
	StateListPtr setupStates;

	// Threadクラスの特定のメンバー変数を足し合わせたものを返す。
	uint64_t accumulate(std::atomic<uint64_t> Thread::* member) const {

		uint64_t sum = 0;
		for (Thread* th : *this)
			sum += (th->*member).load(std::memory_order_relaxed);
		return sum;
	}
};

// ThreadPoolのglobalな実体
extern ThreadPool Threads;

#endif // #ifndef THREAD_H_INCLUDED

