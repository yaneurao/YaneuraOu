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
class Thread
{
	// exitフラグやsearchingフラグの状態を変更するときのmutex
	Mutex mutex;

	// idle_loop()で待機しているときに待つ対象
	ConditionVariable cv;

	// thread id。main threadなら0。slaveなら1から順番に値が割当てられる。
	size_t idx;

	// このフラグが立ったら終了する。
	bool exit = false;

	// 探索中であるかを表すフラグ。プログラムを簡素化するため、事前にtrueにしてある。
	bool searching = true;

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
	//       プロパティ
	// ------------------------------

	// スレッドidが返る。Stockfishにはないメソッドだが、
	// スレッドごとにメモリ領域を割り当てたいときなどに必要となる。
	// MainThreadなら0、slaveなら1,2,3,...
	size_t thread_id() const { return idx; }

	// ------------------------------
	//       探索に必要なもの
	// ------------------------------

	// 探索開始局面
	Position rootPos;

	// 探索開始局面で思考対象とする指し手の集合。
	// goコマンドで渡されていなければ、全合法手(ただし歩の不成などは除く)とする。
	Search::RootMoves rootMoves;

	// このスレッドでMultiPVを用いているとして、rootMovesの(0から数えて)何番目のPVの指し手を探索中であるか
	// MultiPVでないときはこの変数の値は0。
	size_t PVIdx;

	// rootから最大、何手目まで探索したか(選択深さの最大)
	int selDepth;

	// このスレッドが探索したノード数(≒Position::do_move()を呼び出した回数)
	std::atomic<uint64_t> nodes;

	// 反復深化の深さ
	// Lazy SMPなのでスレッドごとにこの変数を保有している。
	Depth rootDepth;

	// このスレッドに関して、終了した反復深化の深さ
	Depth completedDepth;

	// 近代的なMovePickerではオーダリングのために、スレッドごとにhistoryとcounter movesのtableを持たないといけない。
	CounterMoveStat counterMoves;
	ButterflyHistory mainHistory;

	// コア数が多いか、長い持ち時間においては、ContinuationHistoryもスレッドごとに確保したほうが良いらしい。
	// cf. https://github.com/official-stockfish/Stockfish/commit/5c58d1f5cb4871595c07e6c2f6931780b5ac05b5
	ContinuationHistory counterMoveHistory;

	// PositionクラスのEvalListにalignasを指定されていて、Positionクラスを保持するこのThreadクラスをnewするが、
	// そのときにalignasを無視されるのでcustom allocatorを定義しておいてやる。
	void* operator new(std::size_t s);
	void operator delete(void*p) noexcept;
};
  

// 探索時のmainスレッド(これがmasterであり、これ以外はslaveとみなす)
struct MainThread: public Thread
{
	// constructorはThreadのものそのまま使う。
	using Thread::Thread;

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
	int callsCnt;
};


// 思考で用いるスレッドの集合体
// 継承はあまり使いたくないが、for(auto* th:Threads) ... のようにして回せて便利なのでこうしてある。
struct ThreadPool: public std::vector<Thread*>
{
	// このクラスにコンストラクタとデストラクタは存在しない。

	// Threads(スレッドオブジェクト)はglobalに配置するし、スレッドの初期化の際には
	// スレッドが保持する思考エンジンが使う変数等がすべてが初期化されていて欲しいからである。

	// 起動時に一度だけ呼び出す。そのときにMainThreadが生成される。
	// requested : 生成するスレッドの数(MainThreadも含めて数える)
	// スレッド数が変更になったときは、set()のほうを呼び出すこと。
	void init(size_t requested);

	// 終了時に呼び出される
	void exit();

	// mainスレッドに思考を開始させる。
	void start_thinking(const Position& pos, StateListPtr& states , const Search::LimitsType& limits , bool ponderMode = false);

	// スレッド数を変更する。
	void set(size_t requested);

	// mainスレッドを取得する。これはthis[0]がそう。
	MainThread* main() { return static_cast<MainThread*>(at(0)); }

	// 今回、goコマンド以降に探索したノード数
	uint64_t nodes_searched() { return accumulate(&Thread::nodes); }

	// stop   : 探索中にこれがtrueになったら探索を即座に終了すること。
	// ponder : "go ponder" コマンドでの探索中であるかを示すフラグ
	// stopOnPonderhit : Stockfishのこのフラグは、やねうら王では用いない。(もっと上手にponderの時間を活用したいため)
	// received_go_ponder : Stockfishにはこのコードはないが、試合開始後、"go ponder"が一度でも送られてきたかのフラグ。これにより思考時間を自動調整する。
	// 本来は、Options["Ponder"]で設定すべきだが(UCIではそうなっている)、USIプロトコルだとGUIが勝手に設定するので、思考エンジン側からPonder有りのモードなのかどうかが取得できない。
	// ゆえに、このようにして判定している。
	// 備考) ponderのフラグを変更するのはUSIコマンドで"ponderhit"などが送られてきたときであり、探索スレッドからは、探索中は
	//       ponderの値はreadonlyであるから複雑な同期処理は必要がない。
	//       (途中で値が変更されるのは、ponder == trueで探索が始まり、途中でfalseに変更されるケースのみ)
	//       そこで単にatomic_boolにしておけば十分である。
	// 
	std::atomic_bool stop , ponder /*, stopOnPonderhit*/ , received_go_ponder;


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

#endif // _THREAD_H_
