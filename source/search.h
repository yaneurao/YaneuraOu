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

#if defined(USE_MOVE_PICKER)

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

#endif

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

	// 今回の(反復深化の)iterationでの探索結果のスコア
	Value score			= -VALUE_INFINITE;

	// 前回の(反復深化の)iterationでの探索結果のスコア
	// 次のiteration時の探索窓の範囲を決めるときに使う。
	Value previousScore = -VALUE_INFINITE;

	// aspiration searchの時に用いる。previousScoreの移動平均。
	Value averageScore	= -VALUE_INFINITE;

	// aspiration searchの時に用いる。二乗平均スコア。
	Value meanSquaredScore = - VALUE_INFINITE * VALUE_INFINITE;

	// USIに出力する用のscore
	Value usiScore		= -VALUE_INFINITE;

	// usiScoreはlowerboundになっているのか。
	bool scoreLowerbound = false;

	// usiScoreはupperboundになっているのか。
	bool scoreUpperbound = false;

	// このスレッドがrootから最大、何手目まで探索したか(選択深さの最大)
	int selDepth = 0;

	// チェスの定跡絡みの変数。将棋では未使用。
	// int tbRank = 0;
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
		time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
		/*movestogo = */depth = mate = perft = infinite = 0;
		nodes = 0;

		// --- やねうら王で、将棋用に追加したメンバーの初期化。

		byoyomi[WHITE] = byoyomi[BLACK] = TimePoint(0);
		max_game_ply = 100000;
		rtime = 0;

		// 入玉に関して
		enteringKingRule = EKR_NONE;
		enteringKingPoint[BLACK] = 28; // Position::set()でupdate_entering_point()が呼び出されて設定される。
		enteringKingPoint[WHITE] = 27; // Position::set()でupdate_entering_point()が呼び出されて設定される。

		silent = consideration_mode = outout_fail_lh_pv = false;
		pv_interval = 0;
		generate_all_legal_moves = true;
		wait_stop = false;
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
	std::vector<Move> searchmoves;

	// time[]   : 残り時間(ms換算で)
	// inc[]    : 1手ごとに増加する時間(フィッシャールール)
	// npmsec   : 探索node数を思考経過時間の代わりに用いるモードであるかのフラグ(from UCI)
	// 　　→　将棋と相性がよくないのでこの機能をサポートしないことにする。
	// movetime : 思考時間固定(0以外が指定してあるなら) : 単位は[ms]
	TimePoint time[COLOR_NB] , inc[COLOR_NB] , npmsec , movetime /* , startTime; */;

	// depth    : 探索深さ固定(0以外を指定してあるなら)
	// mate     : 詰み専用探索(USIの'go mate'コマンドを使ったとき)
	//		詰み探索モードのときは、ここに詰みの手数が指定されている。
	//		その手数以内の詰みが見つかったら探索を終了する。
	//		※　Stockfishの場合、この変数は先後分として将棋の場合の半分の手数が格納されているので注意。
	//		USIプロトコルでは、この値に詰将棋探索に使う時間[ms]を指定することになっている。
	//		時間制限なしであれば、INT32_MAXが入っている。
	// perft    : perft(performance test)中であるかのフラグ。非0なら、perft時の深さが入る。
	// infinite : 思考時間無制限かどうかのフラグ。非0なら無制限。
	int /* movestogo,*/ depth, mate, perft, infinite;

	// 今回のgoコマンドでの指定されていた"nodes"(探索ノード数)の値。
	// これは、USIプロトコルで規定されているものの将棋所では送ってこない。ShogiGUIはたぶん送ってくる。
	// goコマンドで"nodes"が指定されていない場合は、"エンジンオプションの"NodesLimit"の値。
	uint64_t nodes;

	// -- やねうら王が将棋用に追加したメンバー

	// 秒読み(ms換算で)
	TimePoint byoyomi[COLOR_NB];

	// この手数で引き分けとなる。256なら256手目を指したあとに引き分け。
	// USIのoption["MaxMovesToDraw"]の値。0が設定されていたら、引き分けなしだからmax_game_ply = 100000が代入されることになっている。
	// (残り手数を計算する時に桁あふれすると良くないのでint_maxにはしていない)
	// この値が0なら引き分けルールはなし(無効)。
	// ※　この変数の値が設定されるタイミングは、"go"コマンドに対してなので、
	//     "go"コマンドが呼び出される前にはこの値は不定であるから用いないこと。
	/*
		初手(76歩とか)が1手目である。1手目を指す前の局面はPosition::game_ply() == 1である。
		そして256手指された時点(257手目の局面で指す権利があること。サーバーから257手目の局面はやってこないものとする)で引き分けだとしたら
		257手目(を指す前の局面)は、game_ply() == 257である。これが、引き分け扱いということになる。

		pos.game_ply() > limits.max_game_ply

		　で(かつ、詰みでなければ)引き分けということになる。

		この引き分けの扱いについては、以下の記事が詳しい。
		多くの将棋ソフトで256手ルールの実装がバグっている件
		https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/
	*/
	int max_game_ply;

	// "go rtime 100"とすると100～300msぐらい考える。
	TimePoint rtime;

	// 入玉ルール設定
	EnteringKingRule enteringKingRule;
	// 駒落ち対応入玉ルーの時に、この点数以上であれば入玉宣言可能。
	// 例) 27点法の2枚落ちならば、↓の[BLACK(下手 = 後手)]には 27 , ↓の[WHITE(上手 = 先手)]には 28-10 = 18 が代入されている。
	int enteringKingPoint[COLOR_NB];

	// 画面に出力しないサイレントモード(プロセス内での連続自己対戦のとき用)
	// このときPVを出力しない。
	bool silent;

	// 検討モード用のPVを出力するのか
	// ※ やねうら王のみ , ふかうら王は未対応。
	bool consideration_mode;

	// fail low/highのときのPVを出力するのか
	bool outout_fail_lh_pv;

	// PVの出力間隔(探索のときにMainThread::search()内で初期化する)
	TimePoint pv_interval;

	// 合法手を生成する時に全合法手を生成するのか(歩の不成など)
	// エンジンオプションのGenerateAllLegalMovesの値がこのフラグに反映される。
	// 
	// Position::pseudo_legal()も、このフラグに応じてどこまでをpseudo-legalとみなすかが変わる。
	// (このフラグがfalseなら歩の不成は非合法手扱い)
	bool generate_all_legal_moves;

	// "go"コマンドに"wait_stop"がついていたかのフラグ。
	// これがついていると、stopが送られてくるまで思考しつづける。
	// 本来の"bestmove"を返すタイミングになると、"info string time to return bestmove."と出力する。
	// この機能は、Clusterのworkerで、持時間制御はworker側にさせたいが、思考は継続させたい時に用いる。
	bool wait_stop;

#if defined(TANUKI_MATE_ENGINE)
	std::vector<Move16> pv_check;
#endif
};

extern LimitsType Limits;

// 探索部の初期化。
void init();

// 探索部のclear。
// 置換表のクリアなど時間のかかる探索の初期化処理をここでやる。isreadyに対して呼び出される。
void clear();

// pv(読み筋)をUSIプロトコルに基いて出力する。
// pos   : 局面
// tt    : このスレッドに属する置換表
// depth : 反復深化のiteration深さ。
std::string pv(const Position& pos, const TranspositionTable& tt, Depth depth);



// The UCI stores the uci options, thread pool, and transposition table.
// This struct is used to easily forward data to the Search::Worker class.

// UCIは、UCIオプション、スレッドプール、トランスポジションテーブルを保持する。
// この構造体は、Search::Workerクラスへデータを簡単に渡すために使われる。

struct SharedState {
	SharedState(const OptionsMap& optionsMap,
		ThreadPool& threadPool,
		TranspositionTable& transpositionTable
		// ,const LazyNumaReplicated<Eval::NNUE::Networks>& nets
	) :
		options(optionsMap),
		threads(threadPool),
		tt(transpositionTable)
		// ,networks(nets)
	{
	}

	const OptionsMap& options;
	ThreadPool& threads;
	TranspositionTable& tt;
	//const LazyNumaReplicated<Eval::NNUE::Networks>& networks;
};


class Worker;

// Null Object Pattern, implement a common interface for the SearchManagers.
// A Null Object will be given to non-mainthread workers.

// Nullオブジェクトパターン：SearchManagerの共通インターフェースを実装する。
// メインスレッドでないワーカーにはNullオブジェクトが与えられる。

class ISearchManager {
public:
	virtual ~ISearchManager() {}
	virtual void check_time(Search::Worker&) = 0;
};

// Engineが持つべき読み筋の情報(簡単版)
struct InfoShort {
	int   depth;
	Value score;
};

// Engineが持つべき読み筋の情報(完全版)
struct InfoFull : InfoShort {
	int              selDepth;
	size_t           multiPV;
	std::string_view wdl;
	std::string_view bound;
	size_t           timeMs;
	size_t           nodes;
	size_t           nps;
	size_t           tbHits;
	std::string_view pv;
	int              hashfull;
};

// Engineが持つべき反復深化の情報
struct InfoIteration {
	int              depth;
	std::string_view currmove;
	size_t           currmovenumber;
};


// SearchManager manages the search from the main thread. It is responsible for
// keeping track of the time, and storing data strictly related to the main thread.
class SearchManager : public ISearchManager {
public:
	using UpdateShort = std::function<void(const InfoShort&)>;
	using UpdateFull = std::function<void(const InfoFull&)>;
	using UpdateIter = std::function<void(const InfoIteration&)>;
	using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;

	struct UpdateContext {
		UpdateShort    onUpdateNoMoves;
		UpdateFull     onUpdateFull;
		UpdateIter     onIter;
		UpdateBestmove onBestmove;
	};


	SearchManager(const UpdateContext& updateContext) :
		updates(updateContext) {
	}

	void check_time(Search::Worker& worker) override;

	void pv(Search::Worker& worker,
		const ThreadPool& threads,
		const TranspositionTable& tt,
		Depth                     depth);

	//Stockfish::TimeManagement tm;
	double                    originalTimeAdjust;
	int                       callsCnt;
	std::atomic_bool          ponder;

	std::array<Value, 4> iterValue;
	double               previousTimeReduction;
	Value                bestPreviousScore;
	Value                bestPreviousAverageScore;
	bool                 stopOnPonderhit;

	size_t id;

	const UpdateContext& updates;
};

class NullSearchManager : public ISearchManager {
public:
	void check_time(Search::Worker&) override {}
};


// Search::Worker is the class that does the actual search.
// It is instantiated once per thread, and it is responsible for keeping track
// of the search history, and storing data required for the search.
class Worker {
public:
	Worker(SharedState&, std::unique_ptr<ISearchManager>, size_t, NumaReplicatedAccessToken);

	// Called at instantiation to initialize reductions tables.
	// Reset histories, usually before a new game.
	void clear();

	// Called when the program receives the UCI 'go' command.
	// It searches from the root position and outputs the "bestmove".
	void start_searching();

	bool is_mainthread() const { return threadIdx == 0; }

	void ensure_network_replicated();

	// Public because they need to be updatable by the stats
	ButterflyHistory mainHistory;
	LowPlyHistory    lowPlyHistory;

	CapturePieceToHistory captureHistory;
	ContinuationHistory   continuationHistory[2][2];
	//PawnHistory           pawnHistory;

	//CorrectionHistory<Pawn>         pawnCorrectionHistory;
	//CorrectionHistory<Minor>        minorPieceCorrectionHistory;
	//CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
	//CorrectionHistory<Continuation> continuationCorrectionHistory;

	TTMoveHistory ttMoveHistory;

private:
	void iterative_deepening();

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

	LimitsType limits;

	size_t                pvIdx, pvLast;
	std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;
	int                   selDepth, nmpMinPly;

	Value optimism[COLOR_NB];

	Position  rootPos;
	StateInfo rootState;
	RootMoves rootMoves;
	Depth     rootDepth, completedDepth;
	Value     rootDelta;

	size_t                    threadIdx;
	NumaReplicatedAccessToken numaAccessToken;

	// Reductions lookup table initialized at startup
	std::array<int, MAX_MOVES> reductions;  // [depth or moveNumber]

	// The main thread has a SearchManager, the others have a NullSearchManager
	std::unique_ptr<ISearchManager> manager;

	//Tablebases::Config tbConfig;

	const OptionsMap& options;
	ThreadPool& threads;
	TranspositionTable& tt;
	//const LazyNumaReplicated<Eval::NNUE::Networks>& networks;

	// Used by NNUE
	//Eval::NNUE::AccumulatorStack  accumulatorStack;
	//Eval::NNUE::AccumulatorCaches refreshTable;

	//friend class Stockfish::ThreadPool;
	friend class SearchManager;
};


// Continuation Historyに対するBonus値の配列の型
struct ConthistBonus {
	int index;
	int weight;
};

} // namespace Search

} // namespace YaneuraOu

#endif // SEARCH_H_INCLUDED

