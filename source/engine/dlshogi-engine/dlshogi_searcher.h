#ifndef __DLSHOGI_SEARCHER_H_INCLUDED__
#define __DLSHOGI_SEARCHER_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"
#include "dlshogi_types.h"

// dlshogiの探索部で構造体化・クラス化されていないものを集めたもの。

namespace dlshogi
{
	class NodeTree;
	class NodeGarbageCollector;
	class UctSearcher;
	class UctSearcherGroup;

	// 探索したノード数など探索打ち切り条件を保持している。
	// ※　dlshogiのstruct po_info_t。
	//     dlshogiではglobalになっていた探索条件に関する変数もこの構造体に含めている。
	struct SearchLimit
	{
		// --- ↓今回のgoコマンド後に決定した値

		// 探索を打ち切る探索ノード数
		// 探索したノード数
		// 0 なら制限されていない。これが0以外の時は、↓↓の項目が有効。
		NodeCountType node_limit = 0;
		
		// 先後に対する残りの持ち時間
		TimePoint remaining_time[COLOR_NB];

		// 1手の最小思考時間
		// 0 なら設定されておらず。
		TimePoint minimum_time;

		// 今回の思考時間目安。
		// 0なら、思考時間制限はない。
		// 直接取得せずにDlshogiSearcher::GetTimeLimit()を用いて取得すること。
		TimePoint time_limit;

		// 思考時間の延長が可能なのか。
		// ※　思考時間が固定ではなく、ノード数固定でもないなら、探索の状況によっては時間延長してもよい。
		bool extend_time;

		// 今回の探索のrootColor
		// ※　draw_value_from_blackの実装のためにこここに持っておく必要がある。
		Color root_color;

		// 探索開始局面
		// hashfullの計測用にここに持っておく必要がある。
		Node* current_root;

		// --- ↓これだけ探索中に書き換える。(増えていく) 
		
		// 現在の探索ノード数
		std::atomic<NodeCountType> node_searched;

		// 探索停止フラグ。これがtrueになれば探索は強制打ち切り。
		std::atomic<bool> uct_search_stop;

		// 探索している時間を計測する用のtimer
		// 探索開始時にresetされる。
		Timer begin_time;

		// 最後にPVを出力した時刻(begin_time相対)
		TimePoint last_pv_print;

		// 中断用フラグ
		// これがtrueになると全スレッドは探索を終了する。
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
		int draw_ply = 512;

		// 予測読みの設定
		// エンジンオプションの"USI_Ponder"の値。
		bool pondering_mode = false;

		// エンジンオプションの "UCT_NodeLimit" の値。
		// これは、Nodeを作る数の制限。これはメモリ使用量に影響する。
		// 探索したノード数とは異なる。
		NodeCountType uct_node_limit;

		// エンジンオプションの"MultiPV"の値。
		size_t multi_pv;
	};

	// UCT探索部
	// ※　dlshogiでは、この部分、class化されていない。
	class DlshogiSearcher
	{
	public:
		DlshogiSearcher();
		~DlshogiSearcher(){ FinalizeUctSearch(); }

		// pondering modeの設定
		// "USI_Ponder"の値が反映する。
		void SetPonderingMode(bool flag);

		// 各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
		// "isready"に対して呼び出される。
		void SetThread(std::vector<int> thread_nums, std::vector<int> policy_value_batch_maxsizes);

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
			return (search_options.draw_value_from_black && search_limit.root_color == WHITE)
				? search_options.draw_value_white // draw_value_from_blackかつ後手番なら、後手の引き分けのスコアを返す
				: search_options.draw_value_black;
		}

		// 後手の引き分けのスコアを返す。
		float draw_value_white() const {
			return (search_options.draw_value_from_black && search_limit.root_color == WHITE)
				? search_options.draw_value_black
				: search_options.draw_value_white;
		}

		// 1手にかける時間取得[ms]
		TimePoint GetTimeLimit() const;

		//  ノード再利用の設定
		//    flag : 探索したノードの再利用をするのか
		void SetReuseSubtree(bool flag);

		// PV表示間隔設定[ms]
		void SetPvInterval(const TimePoint interval);

		// UCT探索の初期設定
		//    node_limit : 探索ノード数の制限 0 = 無制限
		//  →　これ、SetLimitsで反映するから、ここでは設定しない。
		void InitializeUctSearch(NodeCountType  node_limit);

		//  UCT探索の終了処理
		void TerminateUctSearch();

		// 探索条件をPosition抜きで設定する。
		// "go"コマンドに対して今回の探索条件を設定してやる。
		//    limits.nodes        : 探索を打ち切るnode数   　→  search_limit.node_limitに反映する。
		//    limits.movetime     : 思考時間固定時の指定     →　search_limit.time_limitに反映する。
		//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limit.draw_plyに反映する。
		void SetLimits(const Search::LimitsType& limits);

		// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
		void SetLimits(const Position* pos, const Search::LimitsType& limits);

		// 終了させるために、search_groupsを開放する。
		void FinalizeUctSearch();

		// UCT探索を停止させる。
		// search_limit.uct_search_stop == trueになる。
		void StopUctSearch();

		// UCTアルゴリズムによる着手生成
		// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
		//   pos           : 探索開始局面
		//   gameRootSfen  : 対局開始局面のsfen文字列(探索開始局面ではない)
		//   moves         : 探索開始局面からの手順
		//   ponderMove    : ponderの指し手 [Out]
		//   ponder        : ponder modeで呼び出されているのかのフラグ。
		//            このフラグがtrueであるなら、この関数はMOVE_NONEしか返さない。
		//   start_threads : この関数を呼び出すと全スレッドがParallelUctSearch()を呼び出して探索を開始するものとする。
		// 返し値 : この局面でのbestな指し手
		// ※　事前にSetLimits()で探索条件を設定しておくこと。
		Move UctSearchGenmove(Position* pos, const std::string& gameRootSfen, const std::vector<Move>& moves, Move& ponderMove, bool ponder,
			const std::function<void()>& start_threads);

		// NNに渡すモデルPathの設定。
		void SetModelPaths(const std::vector<std::string>& paths);

		// NodeTreeを取得。
		NodeTree* get_node_tree() const { return tree.get(); }

		// 並列探索を行う。
		//   rootPos   : 探索開始局面
		//   thread_id : スレッドID
		void parallel_search(const Position& rootPos , size_t thread_id);

		// -- public variables..

		// デバッグ用のメッセージの出力を行うかのフラグ。
		bool debug_message = false;

		// ponder modeでsearch()が呼び出されているかのフラグ。
		bool pondering;

		// プレイアウト情報。
		// 探索打ち切り条件などはここに格納されている。
		SearchLimit search_limit;

		// エンジンオプションで設定された値
		SearchOptions search_options;

		// RootNodeを展開する時に使うmutex。
		// UCTSearcher::ParallelUctSearch()で用いる。
		std::mutex mutex_expand;

		//  探索停止の確認
		//	  返し値 : 探索停止条件を満たしていればtrue
		bool InterruptionCheck();

	private:

		// Root Node(探索開始局面)を展開する。
		void ExpandRoot(const Position* pos);

		//  思考時間延長の確認
		//    返し値 : 探索を延長したほうが良さそうならtrue
		bool ExtendTime();

		// UCT探索を行う、GPUに対応するスレッドグループの集合
		std::unique_ptr<UctSearcherGroup[]> search_groups;

		// 前回の探索開始局面などを保持するためのtree
		std::unique_ptr<NodeTree> tree;
		
		// ガーベジコレクタ
		std::unique_ptr<NodeGarbageCollector> gc;

		// PVの出力と、ベストの指し手の取得
		std::tuple<Move /*bestMove*/, float /* best_wp */, Move /* ponderMove */> get_and_print_pv();

		// 読み筋
		typedef std::vector<Move> PV;
		// MultiPVの時のそれぞれ読み筋
		typedef std::vector<PV> PVs;

		// スレッドIDから対応するgpu_idへのmapper
		std::vector<int> thread_id_to_gpu_id;
		std::vector<UctSearcher*> thread_id_to_uct_searcher;
	};


} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __DLSHOGI_SEARCHER_H_INCLUDED__
