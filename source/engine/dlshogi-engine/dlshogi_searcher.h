#ifndef __DLSHOGI_SEARCHER_H_INCLUDED__
#define __DLSHOGI_SEARCHER_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"
#include "../../book/book.h"
#include "../../mate/mate.h"
#include "dlshogi_types.h"

// dlshogiの探索部で構造体化・クラス化されていないものを集めたもの。

namespace dlshogi
{
	struct Node;
	class NodeTree;
	class NodeGarbageCollector;
	class UctSearcher;
	class UctSearcherGroup;
	class SearchInterruptionChecker;

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
		Timer time_manager;

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
		// ※　draw_value_from_blackの実装のためにこここに持っておく必要がある。
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
	};

	// エンジンオプションで設定された定数。
	// ※　dlshogiでは構造体化されていない。
	struct SearchOptions
	{
		// PUCTの定数

		// KataGoの論文に説明がある。Leela Chess Zeroでも実装されている。
		// https://blog.janestreet.com/accelerating-self-play-learning-in-go/

		float c_fpu_reduction      = 0.27f;
		float c_fpu_reduction_root = 0.00f;

		// AlphaZeroの論文の疑似コードで使われているPuctの式に出てくる。
		// https://science.sciencemag.org/content/362/6419/1140/tab-figures-data

		float         c_init       = 1.44f;
		NodeCountType c_base       = 28288;
		float         c_init_root  = 1.16f;
		NodeCountType c_base_root  = 25617;

		// --- 千日手の価値
		// →これらの値を直接使わずに、get_draw_value_black(),get_draw_value_white()を用いること。

		// エンジンオプションの"Draw_Value_Black","Draw_Value_White","Draw_Value_From_Black"の値。

		float draw_value_black = 0.5f; // 先手が千日手にした時の価値(≒評価値)。1.0fにすると勝ちと同じ扱い。0.0fにすると負けと同じ扱い。
		float draw_value_white = 0.5f; // 後手が千日手にした時の価値(≒評価値)。

		// エンジンオプションの"Draw_Value_From_Black"の値。
		// これがtrueでかつrootColor == WHITEなら、draw_value_blackとdraw_value_whiteを入れ替えて考える。
		bool draw_value_from_black = false; 

		// 投了する勝率。これを下回った時に投了する。
		// エンジンオプションの"Resign_Threshold"を1000で割った値
		float RESIGN_THRESHOLD = 0.0f;

		// ノードを再利用するかの設定。
		// エンジンオプションの"ReuseSubtree"の値。
		bool reuse_subtree = true;

		// PVの出力間隔
		// エンジンオプションの"PV_Interval"の値。
		TimePoint pv_interval = 0;

		// 決着つかずで引き分けとなる手数
		// エンジンオプションの"MaxMovesToDraw"の値。
		int max_moves_to_draw = 512;

		// 予測読みの設定
		// エンジンオプションの"USI_Ponder"の値。
		// これがtrueのときだけ "bestmove XXX ponder YYY" のようにponderの指し手を返す。
		// ※　dlshogiは変数名pondering_mode。
		bool usi_ponder = false;

		// エンジンオプションの "UCT_NodeLimit" の値。
		// これは、Nodeを作る数の制限。これはメモリ使用量に影響する。
		// 探索したノード数とは異なる。
		NodeCountType uct_node_limit;

		// エンジンオプションの"MultiPV"の値。
		size_t multi_pv;

		// デバッグ用のメッセージの出力を行うかのフラグ。
		// エンジンオプションの"DebugMessage"の値。
		bool debug_message = false;

		// (歩の不成、敵陣2段目の香の不成など)全合法手を生成するのか。
		bool generate_all_legal_moves = false;

		// leaf node(探索の末端の局面)での奇数手詰みルーチンを呼び出す時の手数
		// 0 = 奇数手詰めを呼び出さない。
		// エンジンオプションの"MateSearchPly"の値。
		int mate_search_ply;

	};

	// ノードのlock用。
	// 子ノードの構造体にstd::mutex(sizeof(std::mutex)==80)を持たせると子ノードがメモリを消費しすぎるので
	// 局面のhash keyからindexを計算して、↓の配列の要素を使う。

	class NodeMutexs
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
		std::mutex& get_mutex(const HASH_KEY key)
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
		DlshogiSearcher();
		~DlshogiSearcher(){ FinalizeUctSearch(); }

		// エンジンオプションの"USI_Ponder"の値をセットする。
		// "bestmove XXX ponder YYY"
		// のようにponderの指し手を返すようになる。
		void SetPonderingMode(bool flag);

		// GPUの初期化、各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
		// "isready"に対して呼び出される。
		void InitGPU(const std::vector<std::string>& model_paths, std::vector<int> new_thread, std::vector<int> policy_value_batch_maxsizes);

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
		//   draw_value_from_black : エンジンオプションの"Draw_Value_From_Black"の値。
		void SetDrawValue(const int value_black, const int value_white,bool draw_value_from_black);

		// 先手の引き分けのスコアを返す。
		float draw_value_black() const {
			return (search_options.draw_value_from_black && search_limits.root_color == WHITE)
				? search_options.draw_value_white // draw_value_from_blackかつ後手番なら、後手の引き分けのスコアを返す
				: search_options.draw_value_black;
		}

		// 後手の引き分けのスコアを返す。
		float draw_value_white() const {
			return (search_options.draw_value_from_black && search_limits.root_color == WHITE)
				? search_options.draw_value_black
				: search_options.draw_value_white;
		}

		// 1手にかける時間取得[ms]
		//TimePoint GetTimeLimit() const;
		// →　search_limits.time_limitから取得すればいいか…。

		//  ノード再利用の設定
		//    flag : 探索したノードの再利用をするのか
		void SetReuseSubtree(bool flag);

		// PV表示間隔設定[ms]
		void SetPvInterval(const TimePoint interval);

		// DebugMessageの出力。
		// エンジンオプションの"DebugMessage"の値をセットする。
		// search_options.debug_messageに反映される。
		void SetDebugMessage(bool flag);

		// (歩の不成、敵陣2段目の香の不成など)全合法手を生成するのか。
		void SetGetnerateAllLegalMoves(bool flag) { search_options.generate_all_legal_moves = flag; }

		// UCT探索の初期設定
		//    node_limit : 探索ノード数の制限 0 = 無制限
		//  →　これ、SetLimitsで反映するから、ここでは設定しない。
		void InitializeUctSearch(NodeCountType  node_limit);

		//  UCT探索の終了処理
		void TerminateUctSearch();

		// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
		//    limits.nodes        : 探索を打ち切るnode数   　→  search_limits.nodes_limitに反映する。
		//    limits.movetime     : 思考時間固定時の指定     →　search_limits.movetimeに反映する。
		//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limits.max_moves_to_drawに反映する。
		// などなど。
		// その他、"go"コマンドで渡された残り時間等から、今回の思考時間を算出し、search_limits.time_managerに反映する。
		void SetLimits(const Position* pos, const Search::LimitsType& limits);

		// 終了させるために、search_groupsを開放する。
		void FinalizeUctSearch();

		// UCT探索を停止させる。
		// search_limits.uct_search_stop == trueになる。
		void StopUctSearch();

		// UCTアルゴリズムによる着手生成
		// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
		// ※　事前にSetLimits()で探索条件を設定しておくこと。
		//   pos           : 探索開始局面
		//   gameRootSfen  : 対局開始局面のsfen文字列(探索開始局面ではない)
		//   moves         : 探索開始局面からの手順
		//   ponderMove    : [Out] ponderの指し手(ないときはMOVE_NONEが代入される)
		//   start_threads    : この関数を呼び出すと全スレッドがParallelUctSearch()を呼び出して探索を開始する。
		//   teminate_threads :	ponderが解除されるのを待機して、そのあとstart_threadsで開始した全スレッドが終了するのを待機する。
		//                    // start_threadsを呼び出さずにteminate_threads()だけ呼び出すことがある。
		// 返し値 : この局面でのbestな指し手
		// この関数は、定跡にhitした時や宣言勝ちなどで、実際の探索を行わない場合でもteminate_threads()を呼び出してから
		// リターンすることは保証する。
		Move UctSearchGenmove(Position* pos, const std::string& gameRootSfen, const std::vector<Move>& moves, Move& ponderMove,
			const std::function<void()>& start_threads,
			const std::function<void()>& terminate_threads
			);

		// NNに渡すモデルPathの設定。
		void SetModelPaths(const std::vector<std::string>& paths);

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
		void InterruptionCheck();

		// PV表示の確認
		// SearchInterruptionCheckerから呼び出される。
		void OutputPvCheck();

		// 子ノードで使うstd::mutexを返す。
		//   pos : 子ノードの局面になっていること。
		// Nodeのchildrenの書き換えの時などにこれをlockすることになっている。
		std::mutex& get_node_mutex(const Position* pos) { return node_mutexes.get_mutex(pos); }
		//std::mutex& get_child_node_mutex(const HASH_KEY posKey) { return child_node_mutexes.get_mutex(posKey); }

	private:

		// Root Node(探索開始局面)を展開する。
		// generate_all : 歩の不成なども生成する。
		void ExpandRoot(const Position* pos , bool generate_all);

		//  思考時間延長の確認
		//    返し値 : 探索を延長したほうが良さそうならtrue
		//bool ExtendTime();

		// UCT探索を行う、GPUに対応するスレッドグループの集合
		std::unique_ptr<UctSearcherGroup[]> search_groups;

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

		// ノードのlockに使うmutex
		NodeMutexs node_mutexes;

		// 探索とは別スレッドでの詰み探索用

		// 奇数手詰め
		Mate::MateSolver mate_solver;
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
		const int kCheckIntervalMs = 10;

		// ガーベジ用のスレッドが実行するworker
		// 探索開始時にこの関数を呼び出す。
		void Worker();

	private:
		DlshogiSearcher* ds;
	};


} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __DLSHOGI_SEARCHER_H_INCLUDED__
