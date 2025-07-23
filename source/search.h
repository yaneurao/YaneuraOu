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
#include "score.h"
//#include "syzygy/tbprobe.h"
//#include "timeman.h"
#include "timeman.h"
//#include "types.h"

namespace YaneuraOu {

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

// 💡 ここにあった"struct Stack"は、
//     engine/yaneuraou-engine/yaneuraou-search.h に移動させた。


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

#if STOCKFISH
	// 💡 チェスのtablebase絡みの変数。将棋では未使用。
	int tbRank          = 0;
	Value tbScore;
#endif

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
        /* movestogo =*/depth = mate = perft = infinite = 0;
        nodes                                           = 0;
        ponderMode                                      = false;

        // --- やねうら王で、将棋用に追加したメンバーの初期化。

        byoyomi[WHITE] = byoyomi[BLACK] = TimePoint(0);
        rtime                           = 0;
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
#if STOCKFISH
    TimePoint time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
#else
    TimePoint time[COLOR_NB], inc[COLOR_NB] /*, npmsec*/, movetime, startTime;
#endif

    // movestogo: あと何手で引き分けとなるか。
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
#if STOCKFISH
    int movestogo, depth, mate, perft, infinite;
#else
    int /* movestogo,*/ depth, mate, perft, infinite;
#endif

    // 今回のgoコマンドでの指定されていた"nodes"(探索ノード数)の値。
    // これは、USIプロトコルで規定されているものの将棋所では送ってこない。ShogiGUIはたぶん送ってくる。
    // goコマンドで"nodes"が指定されていない場合は、"エンジンオプションの"NodesLimit"の値。
    uint64_t nodes;

    // "go"コマンドに"ponder"が付随していたかのフラグ。
    // 💡 ponder探索中であるかのフラグは、別途SearchManager::ponderが持っている。
    //     そちらは、"stop"か"ponderhit"が来るとfalseになるが、こちらは、変化しない。
    bool ponderMode;

#if STOCKFISH
#else
    // 🌈 やねうら王が将棋用に追加したメンバー

    // 秒読み(ms換算で)
    TimePoint byoyomi[COLOR_NB];

    // "go rtime 100"とすると100～300msぐらい考える。
    TimePoint rtime;
#endif
};

/*
	📌  読み筋を表現する構造体  📌

	読み筋の出力は、USIEngineに実装されている。(on_update_no_movesなど)
	UpdateContextがそのlistenerになっていて、読み筋を出力したい時は、UpdateContext経由で
	読み筋出力を呼び出す。

	UpdateContextのlistenerを変更することでEngine側は、読み筋の出力の抑制などができる。
	(benchmarkや教師生成の時は抑制したいので…)	
*/

// PVの短いやつ
struct InfoShort {
    int   depth;
    Score score;
};

// PVの長いやつ
// 📝 MultiPVの場合、MultiPVのある1つの候補手を出力する。
struct InfoFull: InfoShort {
    // 選択的な探索深さ
    int selDepth;

    // "multipv"の値。
    size_t multiPV;

#if STOCKFISH
    // 💡勝率はやねうら王では使わない
    std::string_view wdl;
#endif

    // boundを文字列化したもの
    std::string_view bound;

    // 経過時間
    size_t timeMs;

    // 探索したnode数
    size_t nodes;

    // NPS
    size_t nps;

    // 💡tbHitsもやねうら王では使わない。(tb = tablebases)
    //size_t           tbHits;

    // PVを文字列化したもの
    std::string_view pv;

    // hashfullを文字列化したもの
    int hashfull;
};

// 反復深化のIteration中のPV出力
struct InfoIteration {
    // 探索深さ
    int depth;
    // 現在探索中の指し手を文字列化したもの
    std::string_view currmove;
    // 現在探索中の指し手のナンバー
    size_t currmovenumber;
};

// 📌 読み筋を出力する時に呼び出すlistener
// 🤔 StockfishではSearchManagerで定義されているが、
//     やねうら王ではnamespace Searchで定義しておく。
// 📝 UpdateInfoは、"info string ..."にそのまま出力する。
//    やねうら王独自拡張。

// Infoを更新した時のcallback。このcallbackを行うと標準出力に出力する。
using UpdateShort    = std::function<void(const InfoShort&)>;
using UpdateFull     = std::function<void(const InfoFull&)>;
using UpdateIter     = std::function<void(const InfoIteration&)>;
using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;
using UpdateInfo     = std::function<void(std::string_view)>;

// 読み筋を出力するための関数を呼び出すlistener
struct UpdateContext {
    UpdateShort    onUpdateNoMoves;  // root局面で指し手がない時に用いる。
    UpdateFull     onUpdateFull;     // PVを出力する時に用いる。
    UpdateIter     onIter;           // 反復深化で現在探索中の指し手。
    UpdateBestmove onBestmove;       // bestmoveを出力する時に用いる。
    UpdateInfo     onUpdateString;   // "info string "でそのまま出力する。
};

// Search::Worker is the class that does the actual search.
// It is instantiated once per thread, and it is responsible for keeping track
// of the search history, and storing data required for the search.

// Search::Worker は実際の探索を行うクラスです。
// このクラスはスレッドごとに1つインスタンス化され、探索履歴を管理し、
// 探索に必要なデータを保持する役割を担います。

/*
	📌  すべてのWorkerの基底classに相当する最小限のWorker 📌

	💡  やねうら王では、Search::Workerは最小限にして、このclassを派生して
	     それぞれの思考エンジンを実装するように変更している。

		 それぞれの変数・メソッドの意味については、
         やねうら王探索部のWorker(YaneuraOuWorker)のコメントも確認すること。

	📝  エンジンを自作する時は、このclassを派生させて、このclassのfactoryをThreadPoolに渡す。
		例として、USER_ENGINE である、user-engine.cpp のソースコードを見ると良い。
*/

class Worker;
typedef std::function<std::unique_ptr<Worker>(size_t /*threadIdx*/, NumaReplicatedAccessToken /*numaAccessToken*/)> WorkerFactory;

class Worker
{
public:

	Worker(OptionsMap& options, ThreadPool& threads, size_t threadIdx, NumaReplicatedAccessToken numaAccessToken);

	// Called at instantiation to initialize reductions tables.
    // Reset histories, usually before a new game.
	// インスタンス化時に呼び出され、リダクションテーブルを初期化する。
    // 通常、新しい対局の前に履歴をリセットする。

	// 📌 やねうら王では、このworkerの初期化は(派生classで)ここに書く。
	// 💡 これは、"usinewgame"に対して呼び出されることが保証されている。(つまり各対局の最初に呼び出される。)
	//     "usinewgame" ⇨ ThreadPool::resize_threads() ⇨ ThreadPool.clear() ⇨  各Threadに所属するWorker.clear()
    virtual void clear() {}

	// Called when the program receives the UCI 'go' command.
    // It searches from the root position and outputs the "bestmove".
    // プログラムが UCI の 'go' コマンドを受け取ったときに呼び出される。
    // ルート局面から探索を行い、"bestmove" を出力する。

	// 📌 やねうら王では、探索の処理を(派生classで)ここに書く。
	// 📝 このメソッドはmain threadから呼び出される。
	//    そのあと、sub threadの探索を開始するには、このメソッドのなかから
	//    threads.start_searching()を呼び出す。
	//    そうすると、sub threadから、このstart_searching()が呼び出される。
	//    並列探索の具体例としては、YaneuraOuWorker::start_searching()を見ること。
	virtual void start_searching(){}

	// 🌈 start_searching()より前にUI threadから呼び出される。
    /* 📓 start_searching() のなかでmain threadがlimits.ponderを初期化しようにも、
	       start_searching()が呼び出された時にはUI threadは、次のUSIコマンドを受け取りのUSI loopに
	       復帰していてstart_searching()内でlimits.ponderを初期化するより前に"ponderhit"を
	       受信してしまう可能性がある。
	       よって、start_searching()より前のタイミングで、UI threadからblock呼び出しで
	       呼び出されるようなevent handlerが必要となり、それが、このpre_start_searching()である。
	*/
	virtual void pre_start_searching() {}

	// メインスレッドであるならtrueを返す。
	bool is_mainthread() const { return threadIdx == 0; }

	// 評価関数パラメーターが各Numaにコピーされるようにする。
	virtual void ensure_network_replicated() {}

	// 📝 やねうら王では、以下は、派生class(YaneuraOuWorker)側で実装する。
#if STOCKFISH
    // Public because they need to be updatable by the stats
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;

    CapturePieceToHistory captureHistory;
    ContinuationHistory   continuationHistory[2][2];
    PawnHistory           pawnHistory;

    CorrectionHistory<Pawn>         pawnCorrectionHistory;
    CorrectionHistory<Minor>        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation> continuationCorrectionHistory;

    TTMoveHistory ttMoveHistory;
#endif

protected:

	// 📝 やねうら王では、派生class(YaneuraOuWorker)側で実装する。
	// ⚠ do_move～undo_null_moveは、派生class側でのみ定義する。
	//     これを仮想関数にしてしまうと、呼び出しのoverheadが気になる。

#if STOCKFISH
    //void iterative_deepening();

	void do_move(Position& pos, const Move move, StateInfo& st);
	void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck);
	void do_null_move(Position& pos, StateInfo& st);
	void undo_move(Position& pos, const Move move);
	void undo_null_move(Position& pos);

    // This is the main search function, for both PV and non-PV nodes
    template<NodeType nodeType>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    // Quiescence search function, which is called by the main search
    template<NodeType nodeType>
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta);

    Depth reduction(bool i, Depth d, int mn, int delta) const;

    // Pointer to the search manager, only allowed to be called by the main thread
    SearchManager* main_manager() const {
        assert(threadIdx == 0);
        return static_cast<SearchManager*>(manager.get());
    }

    TimePoint elapsed() const;
    TimePoint elapsed_time() const;

    Value evaluate(const Position&);
#endif

    // 今回の"go"コマンドで渡された思考条件
	LimitsType limits;

	// 📝 派生class側で
    // size_t pvIdx, pvLast;

	// nodes           : 探索したnode数。do_move()で(自分で)カウントする。
    // tbHits          : tablebaseにhitした回数。将棋では使わない。
    // bestMoveChanges : bestMoveが反復深化のなかで変化した回数。📝 派生classのほうで。
    std::atomic<uint64_t> nodes /*, tbHits, bestMoveChanges*/;

	// 📝 派生class側で。
#if STOCKFISH
    int selDepth, nmpMinPly;

    Value optimism[COLOR_NB];
#endif

	// 🤔 外部からrootMovesにアクセスしたいことがあるので、やねうら王では
    //     このへんはpublicにしておく。
public:

	// 探索開始局面
    Position rootPos;

    // rootPosに対するStateInfo
    StateInfo rootState;

    // Rootの指し手
    RootMoves rootMoves;

protected:

	// 📝 派生class側で
#if STOCKFISH
    // 探索した深さ。
    //Depth rootDepth, completedDepth;

    // aspiration searchのroot delta
    //Value rootDelta;
#endif

	// threadのindex(0からの連番), 0がmain thread
    // 📑コンストラクタで渡されたもの
    size_t threadIdx;

	// このWorker threadに対応るNumaのtoken
    // 💡 コンストラクタで渡されたもの
    NumaReplicatedAccessToken numaAccessToken;

	// 📝 派生class側で
    //// The main thread has a SearchManager, the others have a NullSearchManager
    //std::unique_ptr<ISearchManager> manager;

	// 📝 tablebaseは将棋では使わない。
    //Tablebases::Config tbConfig;

	// エンジンOption管理
    // 💡 コンストラクタで渡されたもの
    const OptionsMap& options;

    // thread管理
    // 💡 コンストラクタで渡されたもの
	ThreadPool& threads;

	// 置換表
	// 📝 派生class側で。
	// 🤔 エンジン種別ごとに異なる置換表実装を行う余地を残すため、
	//     やねうら王ではWorker classは置換表を持たせない。
#if STOCKFISH
    TranspositionTable& tt;
#endif

#if defined(EVAL_SFNN)
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;

    // Used by NNUE
    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;
#endif

	friend class YaneuraOu::ThreadPool;
#if STOCKFISH
    friend class SearchManager;
#endif
};

// 📌 やねうら王では、SharedStateを用いない。
// 
//     EngineとWorkerと評価関数とを自由に組み合わせられるようにするには、
//     このStockfishの設計だと難しい。

#if STOCKFISH
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

