#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

//#include <cstdint>
//#include <vector>

#include "config.h"

#include "history.h"
#include "misc.h"
//#include "nnue/network.h"
//#include "nnue/nnue_accumulator.h"
#include "numa.h"
#include "position.h"
//#include "score.h"
//#include "syzygy/tbprobe.h"
//#include "timeman.h"
#include "timeman.h"
//#include "types.h"

namespace YaneuraOu {

// NNUE以外のエンジンであるなら、空のNetworksを定義しておく。
#if !defined(YANEURAOU_ENGINE_NNUE)
namespace Eval::NNUE { struct Networks{}; }
#endif

// -----------------------
//      探索用の定数
// -----------------------

// Different node types, used as a template parameter
// テンプレートパラメータとして使用される異なるノードタイプ
enum NodeType {
	NonPV,
	PV,
	Root
};

class TranspositionTable;
class ThreadPool;
class OptionsMap;

// 探索関係
namespace Search {

// -----------------------
//  探索のときに使うStack
// -----------------------

// Stack struct keeps track of the information we need to remember from nodes
// shallower and deeper in the tree during the search. Each search thread has
// its own array of Stack objects, indexed by the current ply.

// Stack構造体は、検索中にツリーの浅いノードや深いノードから記憶する必要がある情報を管理します。
// 各検索スレッドは、現在の深さ（ply）に基づいてインデックスされた、独自のStackオブジェクトの配列を持っています。

struct Stack {
	Move*           pv;					// PVへのポインター。RootMovesのvector<Move> pvを指している。
	PieceToHistory* continuationHistory;// historyのうち、counter moveに関するhistoryへのポインタ。実体はThreadが持っている。
	int             ply;				// rootからの手数。rootならば0。
	Move            currentMove;		// そのスレッドの探索においてこの局面で現在選択されている指し手
	Move            excludedMove;		// singular extension判定のときに置換表の指し手をそのnodeで除外して探索したいのでその除外する指し手
	Value           staticEval;			// 評価関数を呼び出して得た値。NULL MOVEのときに親nodeでの評価値が欲しいので保存しておく。
	int             statScore;			// 一度計算したhistoryの合計値をcacheしておくのに用いる。
	int             moveCount;			// このnodeでdo_move()した生成した何手目の指し手か。(1ならおそらく置換表の指し手だろう)
	bool            inCheck;			// この局面で王手がかかっていたかのフラグ
	bool            ttPv;				// 置換表にPV nodeで調べた値が格納されていたか(これは価値が高い)
	bool            ttHit;				// 置換表にhitしたかのフラグ
	int             cutoffCnt;			// cut off(betaを超えたので枝刈りとしてreturn)した回数。
	int             reduction;          // このnodeでのreductionの量
	bool            isPvNode;           // PV nodeであるかのフラグ。
	int             quietMoveStreak;    // quietの指し手が親nodeからこのnodeまでに何連続したか。
};


// RootMove struct is used for moves at the root of the tree. For each root move
// we store a score and a PV (really a refutation in the case of moves which
// fail low). Score is normally set at -VALUE_INFINITE for all non-pv moves.

// root(探索開始局面)での指し手として使われる。それぞれのroot moveに対して、
// その指し手で進めたときのscore(評価値)とPVを持っている。(PVはfail lowしたときには信用できない)
// scoreはnon-pvの指し手では-VALUE_INFINITEで初期化される。
struct RootMove
{
	// pv[0]には、このコンストラクタの引数で渡されたmを設定する。
	explicit RootMove(Move m) : pv(1, m) {}

	// Called in case we have no ponder move before exiting the search,
	// for instance, in case we stop the search during a fail high at root.
	// We try hard to have a ponder move to return to the GUI,
	// otherwise in case of 'ponder on' we have nothing to think about.

	// 探索を終了する前にponder moveがない場合に呼び出されます。
	// 例えば、rootでfail highが発生して探索を中断した場合などです。
	// GUIに返すponder moveをできる限り準備しようとしますが、
	// そうでない場合、「ponder on」の際に考えるべきものが何もなくなります。

	bool extract_ponder_from_tt(const TranspositionTable& tt, Position& pos, Move ponder_candidate);

	// std::count(),std::find()などで指し手と比較するときに必要。
	bool operator==(const Move& m) const { return pv[0] == m; }

	// sortするときに必要。std::stable_sort()で降順になって欲しいので比較の不等号を逆にしておく。
	// 同じ値のときは、previousScoreも調べる。
	bool operator<(const RootMove& m) const {
		return m.score != score ? m.score < score
								: m.previousScore < previousScore;
	}

	// この指し手のためにどれだけのnodeを探索したか。
	// 💡 反復深化がもう1回回りそうかの判定に用いる。
    uint64_t effort        = 0;

	// 今回の(反復深化の)iterationでの探索結果のスコア
	Value score			   = -VALUE_INFINITE;

	// 前回の(反復深化の)iterationでの探索結果のスコア
	// 次のiteration時の探索窓の範囲を決めるときに使う。
	Value previousScore    = -VALUE_INFINITE;

	// aspiration searchの時に用いる。previousScoreの移動平均。
	Value averageScore	   = -VALUE_INFINITE;

	// aspiration searchの時に用いる。二乗平均スコア。
	Value meanSquaredScore = - VALUE_INFINITE * VALUE_INFINITE;

	// USIに出力する用のscore
	// 🤔 (usiScoreではなく)Stockfishの変数名のままuciScoreにしておくことで
	//     ソースコードの差分を減らすことにする。
	Value uciScore		   = -VALUE_INFINITE;

	// usiScoreはlowerboundになっているのか。
	bool scoreLowerbound   = false;

	// usiScoreはupperboundになっているのか。
	bool scoreUpperbound   = false;

	// このスレッドがrootから最大、何手目まで探索したか(選択深さの最大)
	int selDepth           = 0;

	// 💡 チェスのtablebase絡みの変数。将棋では未使用。
	// int tbRank          = 0;
	// Value tbScore;

	// この指し手で進めたときのpv
	std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;

// goコマンドでの探索時に用いる、持ち時間設定などが入った構造体
// "ponder"のフラグはここに含まれず、Threads.ponderにあるので注意。
struct LimitsType {

	// Init explicitly due to broken value-initialization of non POD in MSVC
	// PODでない型をmemsetでゼロクリアすると破壊してしまうので明示的に初期化する。
	LimitsType() {

		time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] /* = npmsec */ = movetime = TimePoint(0);
		/* movestogo =*/ depth = mate = perft = infinite = 0;
		nodes                                            = 0;
		ponderMode                                       = false;

		// --- やねうら王で、将棋用に追加したメンバーの初期化。

		byoyomi[WHITE] = byoyomi[BLACK] = TimePoint(0);
		rtime = 0;
	}

	// 時間制御を行うのか。
	// 詰み専用探索、思考時間0、探索深さが指定されている、探索ノードが指定されている、思考時間無制限
	// であるときは、時間制御に意味がないのでやらない。
	bool use_time_management() const {
		//return time[WHITE] || time[BLACK];
		// →　将棋だと秒読みの処理があるので両方のtime[c]が0であっても持ち時間制御が不要とは言えない。
		return !(mate | movetime | depth | nodes | perft | infinite);
	}

	// root(探索開始局面)で、探索する指し手集合。特定の指し手を除外したいときにここから省く
	std::vector<std::string> searchmoves;

	// time[]    : 残り時間(ms換算で)
	// inc[]     : 1手ごとに増加する時間(フィッシャールール)
	// npmsec    : 探索node数を思考経過時間の代わりに用いるモードであるかのフラグ(from UCI)
	// 　　→　将棋と相性がよくないのでこの機能をサポートしないことにする。
	// movetime  : 思考時間固定(0以外が指定してあるなら) : 単位は[ms]
	// startTime : 探索開始時刻。"go"コマンドを受け取った時のnow()。なるべく早くに格納しておき、時差をなくす。
	//             💡 この時刻は、USIEngineの"go"のhandlerで設定される。
    // startTimeFromPonderhit
	//           : 📌 やねうら王独自追加。
	//             ponderhitからの経過時間。
	//             TimeManagement::reset()かreset_for_ponderhit()が呼び出された時刻。
    TimePoint time[COLOR_NB], inc[COLOR_NB] /*, npmsec*/, movetime, startTime, startTimeFromPonderhit;

	// movestogo: この手数で引き分け。
	//			📌 USIプロトコルではサポートしない。エンジンオプションで設定すべき。
	// depth    : 探索深さ固定(0以外を指定してあるなら)
	// mate     : 詰み専用探索(USIの'go mate'コマンドを使ったとき)
	//		詰み探索モードのときは、ここに詰みの手数が指定されている。
	//		その手数以内の詰みが見つかったら探索を終了する。
	//		※　Stockfishの場合、この変数は先後分として将棋の場合の半分の手数が格納されているので注意。
	//		USIプロトコルでは、この値に詰将棋探索に使う時間[ms]を指定することになっている。
	//		時間制限なしであれば、INT32_MAXが入っている。
	// perft    : perft(performance test)中であるかのフラグ。非0なら、perft時の深さが入る。
	// infinite : 思考時間無制限かどうかのフラグ。非0なら無制限。
	int                      /* movestogo,*/ depth, mate, perft, infinite;

	// 今回のgoコマンドでの指定されていた"nodes"(探索ノード数)の値。
	// これは、USIプロトコルで規定されているものの将棋所では送ってこない。ShogiGUIはたぶん送ってくる。
	// goコマンドで"nodes"が指定されていない場合は、"エンジンオプションの"NodesLimit"の値。
	uint64_t                 nodes;

	// ponderが有効なのか？
	bool                     ponderMode;

	// -- やねうら王が将棋用に追加したメンバー

	// 秒読み(ms換算で)
	TimePoint byoyomi[COLOR_NB];

	// "go rtime 100"とすると100～300msぐらい考える。
	TimePoint rtime;
};

/*
	📌  すべてのWorkerの基底classに相当する最小限のWorker 📌

	💡  それぞれの変数・メソッドの意味については、
         やねうら王探索部のWorker(YaneuraOuWorker)のコメントを確認すること。

	📝  エンジンを自作する時は、このclassを派生させて、このclassのfactoryをThreadPoolに渡す。
		例として、USER_ENGINE である、user-engine.cpp のソースコードを見ると良い。
*/

class Worker;
typedef std::function<std::unique_ptr<Worker>(size_t /*threadIdx*/, NumaReplicatedAccessToken /*numaAccessToken*/)> WorkerFactory;

class Worker
{
public:

	Worker(OptionsMap& options, ThreadPool& threads, size_t threadIdx, NumaReplicatedAccessToken numaAccessToken);

#if 0
	// Called at instantiation to initialize reductions tables.
    // Reset histories, usually before a new game.
	// インスタンス化時に呼び出され、リダクションテーブルを初期化する。
    // 通常、新しい対局の前に履歴をリセットする。
#endif
    // 📌 やねうら王では、このworkerの初期化は(派生classで)ここに書く。
	// 💡 これは、"usinewgame"に対して呼び出されることが保証されている。(つまり各対局の最初に呼び出される。)
	//     "usinewgame" ⇨ ThreadPool::resize_threads() ⇨ ThreadPool.clear() ⇨  各Threadに所属するWorker.clear()
    virtual void clear() {}

#if 0
	// Called when the program receives the UCI 'go' command.
    // It searches from the root position and outputs the "bestmove".
    // プログラムが UCI の 'go' コマンドを受け取ったときに呼び出される。
    // ルート局面から探索を行い、"bestmove" を出力する。
#endif
	// 📌 探索の処理を(派生classで)ここに書く。
	// 📝 このメソッドはmain threadから呼び出される。
	//    そのあと、sub threadの探索を開始するには、このメソッドのなかから
	//    threads.start_searching()を呼び出す。
	//    そうすると、sub threadから、このstart_searching()が呼び出される。
	//    並列探索の具体例としては、YaneuraOuWorker::start_searching()を見ること。
	virtual void start_searching(){}

	// メインスレッドであるならtrueを返す。
	bool is_mainthread() const { return threadIdx == 0; }

	// 探索開始局面
    Position rootPos;

    // rootPosに対するStateInfo
    StateInfo rootState;

    // Rootの指し手
    RootMoves rootMoves;

    // 探索した深さ
	// 🤔 これは派生class側で持つべき。
    //Depth rootDepth, completedDepth;

    // aspiration searchのroot delta
    // 🤔 これは派生class側で持つべき。
    //Value rootDelta;

	// threadのindex(0からの連番), 0がmain thread
    // 📑コンストラクタで渡されたもの
    size_t threadIdx;

    // 今回の"go"コマンドで渡された思考条件
    LimitsType limits;

	virtual void ensure_network_replicated(){}

protected:
	// ⚠ do_move～undo_null_moveは、派生class側でのみ定義する。
	//     これを仮想関数にしてしまうと、呼び出しのoverheadが気になる。
	#if 0
	void do_move(Position& pos, const Move move, StateInfo& st);
	void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck);
	void do_null_move(Position& pos, StateInfo& st);
	void undo_move(Position& pos, const Move move);
	void undo_null_move(Position& pos);
	#endif

	// nodes           : 探索したnode数。do_move()で(自分で)カウントする。
    // tbHits          : tablebaseにhitした回数。将棋では使わない。
    // bestMoveChanges : bestMoveが反復深化のなかで変化した回数。これは、Worker派生classのほうで必要なら用意する。
    std::atomic<uint64_t> nodes /*, tbHits, bestMoveChanges*/;

	// エンジンOption管理
    // 💡コンストラクタで渡されたもの
    const OptionsMap& options;

    // thread管理
    // 💡コンストラクタで渡されたもの
	ThreadPool& threads;

	// このWorker threadに対応るNumaのtoken
    // 💡コンストラクタで渡されたもの
    NumaReplicatedAccessToken numaAccessToken;

	friend class YaneuraOu::ThreadPool;
	friend class SearchManager;
};

// 📌 やねうら王では、SharedStateを用いない。
// 
//     EngineとWorkerと評価関数とを自由に組み合わせられるようにするには、
//     このStockfishの設計だと難しい。

#if 0
// The UCI stores the uci options, thread pool, and transposition table.
// This struct is used to easily forward data to the Search::Worker class.

// UCIは、UCIオプション、スレッドプール、トランスポジションテーブルを保持する。
// この構造体は、Search::Workerクラスへデータを簡単に渡すために使われる。

struct SharedState {
	SharedState(const OptionsMap& optionsMap,
		ThreadPool& threadPool,
		TranspositionTable& transpositionTable,
		const LazyNumaReplicated<Eval::Evaluator>& nets
	) :
		options(optionsMap),
		threads(threadPool),
		tt(transpositionTable),
		networks(nets)
	{
	}

	const OptionsMap& options;
	ThreadPool& threads;
	TranspositionTable& tt;

	const LazyNumaReplicated<Eval::NNUE::Networks>& networks;
};
#endif

} // namespace Search
} // namespace YaneuraOu


#endif // SEARCH_H_INCLUDED

