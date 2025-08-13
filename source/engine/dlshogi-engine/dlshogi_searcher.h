#ifndef __DLSHOGI_SEARCHER_H_INCLUDED__
#define __DLSHOGI_SEARCHER_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"
#include "../../book/book.h"
#include "../../book/policybook.h"
#include "../../mate/mate.h"
#include "../../timeman.h"
#include "dlshogi_types.h"
#include "SearchOptions.h"
#include "PvMateSearch.h"

// dlshogiの探索部で構造体化・クラス化されていないものを集めたもの。

namespace dlshogi {

	struct Node;
	class NodeTree;
	class NodeGarbageCollector;
	class UctSearcher;
	class UctSearcherGroup;
	class SearchInterruptionChecker;
	class DlshogiSearcher;
    class FukauraOuEngine;

	// 探索したノード数など探索打ち切り条件を保持している。
	// ※　dlshogiのstruct po_info_t。
	//     dlshogiではglobalになっていた探索条件に関する変数もこの構造体に含めている。
	struct SearchLimits
	{
		// --- ↓今回のgoコマンド後に決定した値

		// 探索を打ち切る探索ノード数
		// 探索したノード数
		// 0 なら制限されていない。これが0以外の時は、↓↓の項目が有効。
		NodeCountType nodes_limit = 0;

		// 探索している時間を計測する用のtimer
		// 探索開始時にresetされる。
		// →　やねうら王ではtime_managerで行うので削除。
		//Timer begin_time;

		// 持ち時間制御用
		TimeManagement time_manager;

		// 思考時間の延長が可能なのか。
		// ※　思考時間が固定ではなく、ノード数固定でもないなら、探索の状況によっては時間延長してもよい。
		//bool extend_time;
		// →　やねうら王のtime managerを用いるので不要。

		// 思考時間固定
		// movetimeが反映される。これはgoコマンドの時に付与されているmovetimeの値。
		// 0なら、movetimeによる思考時間制限はない。
		TimePoint movetime;

		// "go infinite"されているか。("stop"がくるまで思考を継続する)
		bool infinite;

		// -- 探索開始局面の情報

		// 今回の探索のrootColor
		// ※　draw_valueなどの実装のためにこここに持っておく必要がある。
		Color root_color;

		// 探索開始局面
		// hashfullの計測用にここに持っておく必要がある。
		Node* current_root;

		// --- ↓これだけ探索中に書き換える。(増えていく) 
		
		// 現在の探索ノード数
		// これは"go","go ponder"に対してゼロに初期化される。
		// 前回、rootの局面に対して訪問していた回数は考慮されない。
		std::atomic<NodeCountType> nodes_searched;

		// 探索停止フラグ。これがtrueになれば探索は強制打ち切り。
		// →　やねうら王では使わない。Threads.stopかsearch_limits.interruptionを使う。
		//std::atomic<bool> uct_search_stop;

		// 最後にPVを出力した時刻(begin_time相対)
		TimePoint last_pv_print;

		// ponder mode("go ponder"コマンド)でsearch()が呼び出されているかのフラグ。
		// これがtrueのときは探索は"bestmove"を返すのを"stop"が来るまで待機しなければならない。
		// →　やねうら王では、Threads.main()->ponderを用いるのでこのフラグは使わない。
		//bool pondering;

		// 中断用フラグ
		// これがtrueになると全スレッドは探索を終了する。
		// この停止のためのチェックは、SearchInterruptionCheckerが行う。
		std::atomic<bool> interruption;

		// 現在のrootの対局開始からの手数
		int game_ply;

		// "go ponder"中であるか。
		std::atomic_bool ponder;
	};


	// ノードのlock用。
	// 子ノードの構造体にstd::mutex(sizeof(std::mutex)==80)を持たせると子ノードがメモリを消費しすぎるので
	// 局面のhash keyからindexを計算して、↓の配列の要素を使う。

	class MutexPool
	{
	public:
		static const uint64_t MUTEX_NUM = 65536; // must be 2^n
		static_assert((MUTEX_NUM & (MUTEX_NUM - 1)) == 0);

		// ある局面に対応するmutexを返す。
		std::mutex& get_mutex(const Position* pos)
		{
			return get_mutex(pos->key());
		}

	private:
		// ある局面のHASH_KEY(Position::key()にて取得できる)に対応するmutexを返す。
		// →　これ外から呼び出して使わなくていいっぽいのでとりまprivateにしておく。
		std::mutex& get_mutex(const Key key)
		{
			return mutexes[key & (MUTEX_NUM - 1)];
		}

	private:
		std::mutex mutexes[MUTEX_NUM];
	};

	// UCT探索部
	// ※　dlshogiでは、この部分、class化されていない。
	class DlshogiSearcher
	{
	public:
        DlshogiSearcher(FukauraOuEngine& engine);
		~DlshogiSearcher(){ FinalizeUctSearch(); }

		// 探索で使うエンジンオプションを生やす。
		void add_options(OptionsMap& options);

		// GPUの初期化、各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
		// "isready"に対して呼び出される。
		// スレッドの生成ついでに、詰将棋探索系の初期化もここで行う。
		// thread_settings : 各GPU用のスレッド数
        void InitGPU(const std::string& model_path , std::vector<int> thread_settings, int policy_value_batch_maxsize);

		// 対局開始時に呼び出されるハンドラ
		void NewGame();

		// 対局終了時に呼び出されるハンドラ
		void GameOver();

		// 投了の閾値設定
		//   resign_threshold : 1000分率で勝率を指定する。この値になると投了する。
		void SetResignThreshold(const int resign_threshold);

		// 千日手の価値設定
		//   value_black           : この値を先手の千日手の価値とみなす。(千分率)
		//   value_white           : この値を後手の千日手の価値とみなす。(千分率)
		void SetDrawValue(const int value_black, const int value_white);

		// 引き分けのスコアを返す。
		// これは、root color , side_to_move に依存する。
		float draw_value(Color side_to_move) const {
			return (search_limits.root_color == BLACK)
				? (side_to_move == BLACK ? search_options.draw_value_black : 1 - search_options.draw_value_black)
				: (side_to_move == WHITE ? search_options.draw_value_white : 1 - search_options.draw_value_white);
		}

		// 1手にかける時間取得[ms]
		//TimePoint GetTimeLimit() const;
		// →　search_limits.time_limitから取得すればいいか…。

		// 勝率から評価値に変換する際の係数を設定する。
		// ここで設定した値は、そのままsearch_options.eval_coefに反映する。
		// 変換部の内部的には、ここで設定した値が1/1000倍されて計算時に使用される。
		// デフォルトは 756。
		void SetEvalCoef(const int eval_coef);

		// PV lineの詰み探索の設定
		// threads : スレッド数
		// nodes   : 1局面で詰探索する最大ノード数。
		void SetPvMateSearch(const int threads, /*const int depth,*/ const int nodes);

		// (歩の不成、敵陣2段目の香の不成など)全合法手を生成するのか。
		void SetGetnerateAllLegalMoves(bool flag) { search_options.generate_all_legal_moves = flag; }

		// UCT探索の初期設定
		void InitializeUctSearch();

		//  UCT探索の終了処理
		void TerminateUctSearch();

		// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
		//    limits.nodes        : 探索を打ち切るnode数   　→  search_limits.nodes_limitに反映する。
		//    limits.movetime     : 思考時間固定時の指定     →　search_limits.movetimeに反映する。
		//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limits.max_moves_to_drawに反映する。
		// などなど。
		// その他、"go"コマンドで渡された残り時間等から、今回の思考時間を算出し、search_limits.time_managerに反映する。
		void SetLimits(const Position& pos, const Search::LimitsType& limits);

		// 終了させるために、search_groupsを開放する。
		void FinalizeUctSearch();

		// UCT探索を停止させる。
		// search_limits.uct_search_stop == trueになる。
		//void StopUctSearch();

		// UCTアルゴリズムによる着手生成
		// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
		// ※　事前にSetLimits()で探索条件を設定しておくこと。
		//   pos            : 探索開始局面
		//   game_root_sfen : ゲーム開始局面のsfen文字列
		//   moves          : ゲーム開始局面からの手順
		//   ponderMove     : [Out] ponderの指し手(ないときはMOVE_NONEが代入される)
		//   返し値 : この局面でのbestな指し手
		// ponderの場合は、呼び出し元で待機すること。
		Move UctSearchGenmove(Position& pos, const std::string& game_root_sfen , const std::vector<Move>& moves, Move& ponderMove);

		// NNに渡すモデルPathの設定。
		//void SetModelPaths(const std::vector<std::string>& paths);

		// 最後にPVを出力した時刻をリセットする。
		void ResetLastPvPrint() { search_limits.last_pv_print = 0; }

		// NodeTreeを取得。
		NodeTree* get_node_tree() const { return tree.get(); }

		// 並列探索を行う。
		//   rootPos   : 探索開始局面
		//   thread_id : スレッドID
		void parallel_search(const Position& rootPos , size_t thread_id);

		// -- public variables..

		// プレイアウト情報。
		// 探索打ち切り条件などはここに格納されている。
		SearchLimits search_limits;

		// エンジンオプションで設定された値
		SearchOptions search_options;

		// RootNodeを展開する時に使うmutex。
		// UCTSearcher::ParallelUctSearch()で用いる。
		std::mutex mutex_expand;

		// 定跡の指し手を選択するモジュール
		Book::BookMoveSelector book;

		//  探索停止の確認
		// SearchInterruptionCheckerから呼び出される。
		void InterruptionCheck(const Position& rootPos);

		// PV表示の確認
		// SearchInterruptionCheckerから呼び出される。
		void OutputPvCheck();

		// 子ノードで使うstd::mutexを返す。
		//   pos : 子ノードの局面になっていること。
		// Nodeのchildrenの書き換えの時などにこれをlockすることになっている。
		std::mutex& get_node_mutex(const Position* pos) { return node_mutexes.get_mutex(pos); }
		//std::mutex& get_child_node_mutex(const HASH_KEY posKey) { return child_node_mutexes.get_mutex(posKey); }

		// 探索開始局面。これはこの局面の探索中には消失しないのでglobalに参照して良い。
		Position pos_root;

		// root局面でdf-pnが詰みを見つけているときは、これがMove::none()以外になる。
		Move rootMateMove;

#if defined(USE_POLICY_BOOK)
		// PolicyBook本体
		PolicyBook policy_book;
#endif

		// 🌈 コンストラクタで渡されたEngine&
        FukauraOuEngine& engine;

	private:

		// Root Node(探索開始局面)を展開する。
		// generate_all : 歩の不成なども生成する。
		void ExpandRoot(const Position* pos , bool generate_all);

		//  思考時間延長の確認
		//    返し値 : 探索を延長したほうが良さそうならtrue
		//bool ExtendTime();

		// 全スレッドでの探索開始
		void StartThreads();
		// 探索スレッドの終了(main thread以外)
		void TeminateThreads();

		// UCT探索を行う、GPUに対応するスレッドグループの集合
		std::unique_ptr<UctSearcherGroup[]> search_groups;
		// ↑の配列サイズ
		size_t search_groups_size = 0;

		// 前回の探索開始局面などを保持するためのtree
		std::unique_ptr<NodeTree> tree;
		
		// ガーベジコレクタ
		std::unique_ptr<NodeGarbageCollector> gc;

		// 探索停止チェック用
		std::unique_ptr<SearchInterruptionChecker> interruption_checker;

		// PVの出力と、ベストの指し手の取得
		std::tuple<Move /*bestMove*/, float /* best_wp */, Move /* ponderMove */> get_and_print_pv();

		// 読み筋
		typedef std::vector<Move> PV;
		// MultiPVの時のそれぞれ読み筋
		typedef std::vector<PV> PVs;

		// スレッドIDから対応するgpu_idへのmapper
		std::vector<int> thread_id_to_gpu_id;
		std::vector<UctSearcher*> thread_id_to_uct_searcher;

		// 前回のInitGPU時のthread_settings
        std::vector<int>          last_thread_settings;

		// ノードのlockに使うmutex
		MutexPool node_mutexes;

		// PV lineの詰探索用
		std::vector<PvMateSearcher> pv_mate_searchers;
	};

	// 探索の終了条件を満たしたかを調べるスレッド
	// dlshogiにはないが、これがないと正確な時刻で探索を終了させることが困難だと思った。
	// GCと同じような作りになっている。
	// スレッドの生成は、やねうら王フレームワーク側で行うものとする。(探索用スレッドを1つ使う)
	// また、PVの表示チェックと探索終了チェックは、DlshogiSearcher側に実装してあるものとする。
	class SearchInterruptionChecker
	{
	public:
		SearchInterruptionChecker(DlshogiSearcher* ds) : ds(ds) {}

		// この間隔ごとに探索停止のチェック、PVの出力のチェックを行う。
		static const int kCheckIntervalMs = 10;

		// ガーベジ用のスレッドが実行するworker
		// 探索開始時にこの関数を呼び出す。
		void Worker(const Position& rootPos);

	private:
		DlshogiSearcher* ds;
	};

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __DLSHOGI_SEARCHER_H_INCLUDED__
