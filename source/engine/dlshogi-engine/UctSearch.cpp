#include "UctSearch.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "Node.h"
#include "dlshogi_searcher.h"
#include "PrintInfo.h"
#include "misc/fastmath.h"  // FastLog()

#include "../../thread.h"
#include "../../usi.h"

#include <limits>           // max<T>()

// 完全なログ出力をしてdlshogiと比較する時用。
//#define LOG_PRINT

#if defined(LOG_PRINT)
#include <sstream>
#endif

using namespace Eval::dlshogi;
using namespace std;

#define LOCK_EXPAND grp->get_dlsearcher()->mutex_expand.lock();
#define UNLOCK_EXPAND grp->get_dlsearcher()->mutex_expand.unlock();


#if defined(LOG_PRINT)
struct MoveIntFloat
{
	MoveIntFloat(Move move, int label, float nnrate) : move(move),label(label),nnrate(nnrate){}

	bool operator < (const MoveIntFloat& rhs) const {
		return nnrate < rhs.nnrate;
	}
	
	std::string to_string()
	{
		return to_usi_string(move) + " " + std::to_string(label) + " " + std::to_string(nnrate);
		//return move.toUSI() + " " + std::to_string(label) + " " + std::to_string(nnrate);
	}

	Move move;
	int label;
	float nnrate;
};

struct MoveMoveLabel
{
	MoveMoveLabel(Move move, int label) : move(move), label(label) {}
	bool operator < (const MoveMoveLabel& rhs) const {
		return label < rhs.label;
	}

	Move move;
	int label;
};

class MyLogger
{
public:
	MyLogger() {
		fs.open("my_log_yane.txt");
	}

	~MyLogger()
	{
		fs.close();
	}

	void print(const std::string& s)
	{
		fs << s << endl;
	}
	void print(std::vector<MoveIntFloat>& m)
	{
		//std::sort(m.begin(), m.end());

		for (auto ml : m)
			fs << ml.to_string() << endl;

		fs.flush();
	}

	ofstream fs;
};
MyLogger logger;
#endif

namespace dlshogi
{
	// atomicな加算。
	template <typename T>
	inline void atomic_fetch_add(std::atomic<T>* obj, T arg) {
		T expected = obj->load();
		while (!atomic_compare_exchange_weak(obj, &expected, expected + arg))
			;
	}

	// Virtual Lossの加算
	inline void AddVirtualLoss(ChildNode* child, Node* current)
	{
		current->move_count += VIRTUAL_LOSS;
		child  ->move_count += VIRTUAL_LOSS;
	}

	// Virtual Lossの減算
	inline void SubVirtualLoss(ChildNode* child, Node* current)
	{
		current->move_count -= VIRTUAL_LOSS;
		child  ->move_count -= VIRTUAL_LOSS;
	}

	// 探索結果の更新
	// 1) resultをcurrentとchildのwinに加算
	// 2) VIRTUAL_LOSSが1でないときは、currnetとchildのmove_countに (1 - VIRTUAL_LOSS) を加算。
	inline void UpdateResult(ChildNode* child, WinCountType result, Node* current)
	{
		atomic_fetch_add(&current->win, result);
		if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
		atomic_fetch_add(&child->win   , result);
		if constexpr (VIRTUAL_LOSS != 1) child->move_count   += 1 - VIRTUAL_LOSS;
	}

	// --------------------------------------------------------------------
	//  DlshogiSearcher : dlshogiの探索部で、globalになっていたものをクラス化したもの。
	// --------------------------------------------------------------------

	DlshogiSearcher::DlshogiSearcher()
	{
		search_groups = std::make_unique<UctSearcherGroup[]>(max_gpu);
		gc = std::make_unique<NodeGarbageCollector>();
	}

	// エンジンオプションの"USI_Ponder"の値をセットする。
	// "bestmove XXX ponder YYY"
	// のようにponderの指し手を返すようになる。
	void DlshogiSearcher::SetPonderingMode(bool flag)
	{
		search_options.usi_ponder = flag;
	}

	// 各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
	// "isready"に対して呼び出される。
	void DlshogiSearcher::SetThread(std::vector<int> new_thread, std::vector<int> policy_value_batch_maxsizes)
	{
		ASSERT_LV3(new_thread.size() == max_gpu);
		ASSERT_LV3(policy_value_batch_maxsizes.size() == max_gpu);

		// この時、前回設定と異なるなら、スレッドの再確保を行う必要がある。

		// トータルのスレッド数
		size_t total_thread_num = 0;
		for (int i = 0; i < max_gpu; ++i)
			total_thread_num += new_thread[i];

		// まず、数だけ確保
		// やねうら王のThreadPoolクラスは、前回と異なるスレッド数であれば自動的に再確保される。
		Threads.set(total_thread_num);

		for (int i = 0; i < max_gpu; i++) {
			if (new_thread[i] > 0) {
				int policy_value_batch_maxsize = policy_value_batch_maxsizes[i];

				//	DNN_Batch_Size2～8は、0が設定されている場合、DNN_Batch_Size1の値が採用される。(そのスレッド数が1以上であるとき)
				if (policy_value_batch_maxsize == 0)
					policy_value_batch_maxsize = policy_value_batch_maxsizes[0];

				search_groups[i].Initialize(new_thread[i],/* gpu_id = */i, policy_value_batch_maxsize);
			}
		}

		// -- やねうら王独自拡張

		// 何番目のスレッドがどのUctSearcherGroupのインスタンスに割り当たるのかのmapperだけ設定しておいてやる。
		// Thread::search()からは、このmapperを使って、UctSearchGroup::search()を呼び出す。

		thread_id_to_uct_searcher.clear();

		for (int i = 0; i < max_gpu; ++i)
			for(int j = 0;j < new_thread[i];++j)
				thread_id_to_uct_searcher.push_back(search_groups[i].get_uct_searcher(j));

		// GC用のスレッドにもスレッド番号を連番で与えておく。
		// (WinProcGroup::bindThisThread()用)
		gc->set_thread_id(thread_id_to_uct_searcher.size());
	}

	// 対局開始時に呼び出されるハンドラ
	void DlshogiSearcher::NewGame()
	{
	}

	// 対局終了時に呼び出されるハンドラ
	void DlshogiSearcher::GameOver()
	{
	}

	// 投了の閾値設定
	//   resign_threshold : 1000分率で勝率を指定する。この値になると投了する。
	void DlshogiSearcher::SetResignThreshold(const int resign_threshold)
	{
		search_options.RESIGN_THRESHOLD = (float)resign_threshold / 1000.0f;
	}

	// 千日手の価値設定
	//   value_black           : この値を先手の千日手の価値とみなす。(千分率)
	//   value_white           : この値を後手の千日手の価値とみなす。(千分率)
	//   draw_value_from_black : エンジンオプションの"Draw_Value_From_Black"の値。
	void DlshogiSearcher::SetDrawValue(const int value_black, const int value_white,bool draw_value_from_black)
	{
		search_options.draw_value_black = (float)value_black / 1000.0f;
		search_options.draw_value_white = (float)value_white / 1000.0f;
		search_options.draw_value_from_black = draw_value_from_black;
	}

	// →　これは、エンジンオプションの"MaxMovesToDraw"を自動的にDlshogiSearcher::SetLimits()で設定するので不要。
	//// 引き分けとする手数の設定
	//void SetDrawPly(const int ply)
	//{
	//	draw_ply = ply;
	//}

	//  ノード再利用の設定
	//    flag : 探索したノードの再利用をするのか
	void DlshogiSearcher::SetReuseSubtree(bool flag)
	{
		search_options.reuse_subtree = flag;
	}

	//---あとで

	// PV表示間隔設定
	void DlshogiSearcher::SetPvInterval(const TimePoint interval)
	{
		search_options.pv_interval = interval;
	}

	//  UCT探索の初期設定
	void DlshogiSearcher::InitializeUctSearch(NodeCountType uct_node_limit)
	{
		search_options.uct_node_limit = uct_node_limit;

		if (!tree) tree = std::make_unique<NodeTree>(gc.get());
		//search_groups = std::make_unique<UctSearcherGroup[]>(max_gpu);
		// →　これもっと早い段階で行わないと間に合わない。コンストラクタに移動させる。

		// dlshogiにはないが、dlshogiでglobalだった変数にアクセスするために、
		// UctSearcherGroupは、DlshogiSearcher*を持たなければならない。
		for (int i = 0; i < max_gpu; ++i)
			search_groups[i].set_dlsearcher(this);
	}

	//  UCT探索の終了処理
	void DlshogiSearcher::TerminateUctSearch()
	{
		// やねうら王では、スレッドはThreadPoolクラスで生成しているのでここは関係ない。

	//#ifdef THREAD_POOL
	//	if (search_groups) {
	//		for (int i = 0; i < max_gpu; i++)
	//			search_groups[i].Term();
	//	}
	//#endif
	}

	// 探索条件をPosition抜きで設定する。
	// "go"コマンドに対して今回の探索条件を設定してやる。
	//    limits.nodes        : 探索を打ち切るnode数   　→  search_limit.node_limitに反映する。
	//    limits.movetime     : 思考時間固定時の指定     →　search_limit.time_limitに反映する。
	//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limit.draw_plyに反映する。
	void DlshogiSearcher::SetLimits(const Search::LimitsType& limits)
	{
		// 探索を打ち切るnode数
		search_limit.node_limit = (NodeCountType)limits.nodes;

		// 思考時間固定の時の指定
		search_limit.time_limit = limits.movetime;

		// 引き分けになる手数の設定。
		search_options.draw_ply = limits.max_game_ply;
	}

	// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
	void DlshogiSearcher::SetLimits(const Position* pos, const Search::LimitsType& limits)
	{
		// dlshogiのコード
#if 0
		// ノード数固定ならばそれを設定。
		// このときタイムマネージメント不要
		if (limits.nodes) {
			search_limit.node_limit = (NodeCountType)(u64)limits.nodes;
			return;
		}

		const Color color = pos->side_to_move();
		const int divisor = 14 + std::max(0, 30 - pos->game_ply());

		// 自分側の残り時間を divisorで割り算して、それを今回の目安時間とする。
		search_limit.remaining_time[BLACK] = limits.time[BLACK];
		search_limit.remaining_time[WHITE] = limits.time[WHITE];

		search_limit.time_limit = search_limit.remaining_time[color] / divisor + limits.inc[color];

		// 最小思考時間の設定。(これ以上考える)
		search_limit.minimum_time = limits.movetime;

		// 無制限に思考するなら、NodeCountTypeの最大値を設定しておく。
		search_limit.node_limit = limits.infinite ? std::numeric_limits<NodeCountType>::max() : (NodeCountType)(u64)limits.nodes;

		// 思考時間が固定ではなく、ノード数固定でもないなら、探索の状況によっては時間延長してもよい。
		search_limit.extend_time = limits.movetime == 0 && limits.nodes == 0;
#endif

		// やねうら王の time managerをそのまま用いる。
		search_limit.time_manager.init(limits,pos->side_to_move(),pos->game_ply());

	}

	// ノード数固定にしたい時は、USIの"go nodes XXX"ででき、これは、SetLimits()で反映するので↓は不要。

	//// 1手のプレイアウト回数を固定したモード
	//// 0の場合は無効
	//void SetConstPlayout(const int playout)
	//{
	//	const_playout = playout;
	//}

	// 終了させるために、search_groupsを開放する。
	void DlshogiSearcher::FinalizeUctSearch()
	{
		search_groups.release();
		tree.release(); // treeの開放を行う時にGCが必要なのでCGをあとから開放
		gc.release();
	}

	// あとで
	// std::tuple<Move, float, Move> get_and_print_pv()

	// UCTアルゴリズムによる着手生成
	// 並列探索を開始して、PVを表示したりしつつ、指し手ひとつ返す。
	//   pos           : 探索開始局面
	//   gameRootSfen  : 対局開始局面のsfen文字列(探索開始局面ではない)
	//   moves         : 探索開始局面からの手順
	//   ponderMove    : ponderの指し手 [Out]
	//   ponder        : ponder mode("go ponder")で呼び出されているのかのフラグ。
	//   start_threads : この関数を呼び出すと全スレッドがParallelUctSearch()を呼び出して探索を開始するものとする。
	// 返し値 : この局面でのbestな指し手
	// ※　事前にSetLimits()で探索条件を設定しておくこと。
	Move DlshogiSearcher::UctSearchGenmove(Position *pos, const std::string& gameRootSfen, const std::vector<Move>& moves, Move &ponderMove,
		bool ponder,const std::function<void()>& start_threads)
	{
		// 探索停止フラグをreset。
		// →　やねうら王では使わない。Threads.stopかsearch_limit.interruptionを使う。
		//search_limit.uct_search_stop = false;

		// わからん。あとで。
		//init_search_begin_time = false;

		// 探索開始時にタイマーをリセットして経過時間を計測する。
		search_limit.begin_time.reset();

		// 中断フラグのリセット
		search_limit.interruption = false;

		// ゲーム木を現在の局面にリセット
		tree->ResetToPosition(gameRootSfen, moves);

		// ルート局面をグローバル変数に保存
		//pos_root =  pos;

		// 探索開始局面
		const Node* current_root = tree->GetCurrentHead();
		search_limit.current_root = tree->GetCurrentHead();

		// "go ponder"で呼び出されているかのフラグの設定
		search_limit.pondering = ponder;

		// 探索ノード数のクリア
		search_limit.node_searched = 0;

		// UCTの初期化。
		// 探索開始局面の初期化
		ExpandRoot(pos);

		//create_test_node(tree.GetCurrentHead());

		// 詰みのチェック

		// ExpandRoot()が呼び出されている以上、子ノードの初期化は完了していて、この局面の合法手の数がchild_numに
		// 代入されているはず。これが0であれば、合法手がないということだから、詰みの局面であり、探索ができることは何もない。
		const int child_num = current_root->child_num;
		if (child_num == 0) {
			return MOVE_NONE;
		}

		// この局面で子ノードから伝播されて、勝ちが確定している。
		if (current_root->value_win == VALUE_WIN)
		{
			if (!pos->in_check())
			{
				// 詰みを発見しているはず
				Move move = pos->mate1ply();

				// 長手数の詰みはあとで。

				if (move)
					return move;

				// これ、子ノードのいずれかは勝ちのはずだからその子ノードを選択したほうが良いのでは…。
			}
		}

		// 前回、この現在の探索局面を何回訪問したのか
		NodeCountType pre_simulated = current_root->move_count;

		// 探索時間とプレイアウト回数の予測値を出力
		if (debug_message)
			UctPrint::PrintPlayoutLimits(search_limit.time_limit, search_limit.node_limit);

		// スレッドの開始と終了
		start_threads();

		// ponderモードでは指し手自体は返さない。
		// →　やねうら王では、stopが来るまで待機して返す。
		//  dlshogiの実装はいったんUCT探索を終了させるようになっているのでコメントアウト。
		//if (pondering)
		//	return MOVE_NONE;

		// あとで
		// 探索の延長判定
		// →　これは探索の停止判定で行うから削除

		// PVの取得と表示
		auto best = UctPrint::get_best_move_multipv(current_root , search_limit , search_options);
		ponderMove = best.ponder;

		// 探索にかかった時間を求める
		const TimePoint finish_time = search_limit.begin_time.elapsed();
		//search_limit.remaining_time[pos->side_to_move()] -= finish_time;

		// デバッグ用のメッセージ出力
		if (debug_message)
		{
			// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
			UctPrint::PrintPlayoutInformation(current_root, &search_limit, finish_time, pre_simulated);
		}

		return best.move;
	}

	// Root Node(探索開始局面)を展開する。
	void DlshogiSearcher::ExpandRoot(const Position* pos)
	{
		Node* current_head = tree->GetCurrentHead();
		if (current_head->child_num == 0) {
			current_head->ExpandNode(pos);
		}
	}

	//  探索停止の確認
	//	  返し値 : 探索停止条件を満たしていればtrue
	bool DlshogiSearcher::InterruptionCheck()
	{
		// "go ponder"で呼び出されている場合、あらゆる制限は無視する。
		// NodesLimitとかは有効であるほうが便利かもしれないが、そういうときにgo ponderで呼び出さないと思う。
		if (search_limit.pondering)
			return false;

		auto& s = search_limit;
		auto& o = search_options;

		// 探索depth固定
		// →　PV掘らないとわからないので実装しない。

		// 探索ノード固定(NodesLimitで設定する)
		//   ※　この時、時間制御は行わない
		if (s.node_limit)
			return s.node_searched >= s.node_limit;

		// 探索時間固定
		// "go movetime XXX"のように指定するのでGUI側が対応していないかもしれないが。
		if (s.time_limit)
			return s.begin_time.elapsed() > s.time_limit;

		// hashfull
		if ((NodeCountType)s.current_root->move_count >= o.uct_node_limit)
			return true;

		// -- 時間制御

		// 消費時間が短い場合は打ち止めしない
		const auto elapsed = search_limit.begin_time.elapsed();

		// 最小思考時間を使っていない。
		if (elapsed < s.time_manager.minimum())
			return false;

		// 最適時間の半分未満であるならもうちょっと考える。
		if (elapsed - 100 < s.time_manager.optimum())
			return false;

		// optimum_timeの 100 [ms]前なので指す。
		return true;

		// 以下、計測しつつ、ちゃんと調整する。
		// あとで

#if 0
		// 最適時間の半分未満であるならもうちょっと考える。
		if (elapsed / 2 < s.time_manager.optimum())
			return false;

		NodeCountType max_searched = 0, second = 0;
		const Node* current_root = tree->GetCurrentHead();
		const int child_num = current_root->child_num;
		const ChildNode* uct_child = current_root->child.get();

		// 探索回数が最も多い手と次に多い手を求める
		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].move_count > max_searched) {
				second = max_searched;
				max_searched = uct_child[i].move_count;
			}
			else if (uct_child[i].move_count > second) {
				second = uct_child[i].move_count;
			}
		}

		// 残りの探索を全て次善手に費やしても optimum_timeまでに
		// 最善手を超えられない場合は探索を打ち切る

		//   rest_po : 残り何回ぐらいplayoutできそうか。
		// elapsed == 0のとき0除算になるので分母を +1 してある。

		// ここ、maximum_timeを利用して、もう少しうまく書くべき。

		s64 rest_po = (s64)(search_limit.node_searched * (s.time_manager.optimum() - elapsed) / (elapsed + 1) );
		rest_po = std::max(rest_po, (s64)0);

		if (max_searched - second > rest_po) {
			//cout << "info string interrupt_no_movechange" << endl;
			return true;
		}
		else {
			return false;
		}
#endif
	}

	//  思考時間延長の確認
	bool DlshogiSearcher::ExtendTime()
	{
		// あとでよく見る。

		// 1番探索した指し手のノード数、2番目に探索した指し手のノード数
		NodeCountType max = 0, second = 0;
		float max_eval = 0, second_eval = 0;
		const Node* current_root = tree->GetCurrentHead();
		const int child_num = current_root->child_num;
		const ChildNode *uct_child = current_root->child.get();

		// 探索回数が最も多い手と次に多い手を求める
		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].move_count > max) {
				second = max;
				max = uct_child[i].move_count;
				max_eval = uct_child[i].win / uct_child[i].move_count;
			}
			else if (uct_child[i].move_count > second) {
				second = uct_child[i].move_count;
				second_eval = uct_child[i].win / uct_child[i].move_count;
			}
		}

		// 最善手の探索回数がが次善手の探索回数の1.5倍未満
		// もしくは、勝率が逆なら探索延長
		if (max < second * 1.5 || max_eval < second_eval) {
			return true;
		}
		else {
			return false;
		}
	}

	void DlshogiSearcher::SetModelPaths(const std::vector<std::string>& paths)
	{
		ASSERT_LV3(paths.size() == max_gpu);
		ASSERT_LV3(search_groups != nullptr);

		for (int i = 0; i < max_gpu; ++i)
		{
			string path = paths[i];

			// paths[1]..paths[7]は、空であればpaths[0]と同様であるものとする。
			if (i > 0 && path == "")
				path = paths[0];

			search_groups[i].InitGPU(path, i);
		}
	}

	// 並列探索を行う。
	//   rootPos   : 探索開始局面
	//   thread_id : スレッドID
	// ※ やねうら王独自拡張
	void DlshogiSearcher::parallel_search(const Position& rootPos, size_t thread_id)
	{
		// このrootPosはスレッドごとに用意されているからコピー可能。

		thread_id_to_uct_searcher[thread_id]->ParallelUctSearch(rootPos);
	}

	// --------------------------------------------------------------------
	//  UCTSearcherGroup : UctSearcherをGPU一つ利用する分ずつひとまとめにしたもの。
	// --------------------------------------------------------------------

	// 初期化
	// "isready"に対して呼び出される。
	// スレッド生成は、やねうら王フレームワーク側で行う。
	//   new_thread                 : このインスタンスが確保するUctSearcherの数
	//   gpu_id                     : このインスタンスに紐付けられているGPU ID
	//   policy_value_batch_maxsize : このインスタンスが生成したスレッドがNNのforward()を呼び出す時のbatchsize
	void UctSearcherGroup::Initialize(const int new_thread , const int gpu_id, const int policy_value_batch_maxsize)
	{
		this->gpu_id = gpu_id;
		if (searchers.size() != new_thread)
		{
			searchers.clear();
			searchers.reserve(new_thread); // いまから追加する要素数はわかっているので事前に確保しておく。

			for (int i = 0; i < new_thread; ++i)
				searchers.emplace_back(this, i, policy_value_batch_maxsize);
		}

		this->policy_value_batch_maxsize = policy_value_batch_maxsize;
	}

	// やねうら王では探索スレッドはThreadPoolが管理しているのでこれらは不要。
	/*
	// スレッド開始
	void
	UCTSearcherGroup::Run()
	{
		if (threads > 0) {
			// 探索用スレッド
			for (int i = 0; i < threads; i++) {
				searchers[i].Run();
			}
		}
	}

	// スレッド終了待機
	void
	UCTSearcherGroup::Join()
	{
		if (threads > 0) {
			// 探索用スレッド
			for (int i = 0; i < threads; i++) {
				searchers[i].Join();
			}
		}
	}

	#ifdef THREAD_POOL
	// スレッド終了
	void
	UCTSearcherGroup::Term()
	{
		if (threads > 0) {
			// 探索用スレッド
			for (int i = 0; i < threads; i++) {
				searchers[i].Term();
			}
		}
	}
	#endif

	
	*/

	// --------------------------------------------------------------------
	//  UCTSearcher : UctSearcherを行うスレッド一つを表現する。
	// --------------------------------------------------------------------

	// NodeTreeを取得
	NodeTree* UctSearcher::get_node_tree() const { return grp->get_dlsearcher()->get_node_tree(); }

	// Evaluateを呼び出すリスト(queue)に追加する。
	void UctSearcher::QueuingNode(const Position *pos, Node* node)
	{
#if defined(LOG_PRINT)
		logger.print("sfen "+pos->sfen(0));
#endif		

		//cout << "QueuingNode:" << index << ":" << current_policy_value_queue_index << ":" << current_policy_value_batch_index << endl;
		//cout << pos->toSFEN() << endl;

		/* if (current_policy_value_batch_index >= policy_value_batch_maxsize) {
			std::cout << "error" << std::endl;
		}*/

		// 現在の局面に出現している特徴量を設定する。
		// current_policy_value_batch_indexは、UctSearchThreadごとに持っているのでlock不要

		make_input_features(*pos, &features1[current_policy_value_batch_index], &features2[current_policy_value_batch_index]);

		// 現在のNodeと手番を保存しておく。
		policy_value_batch[current_policy_value_batch_index] = { node, pos->side_to_move() };

	#ifdef MAKE_BOOK
		policy_value_book_key[current_policy_value_batch_index] = Book::bookKey(*pos);
	#endif

		current_policy_value_batch_index++;
		// これが、policy_value_batch_maxsize分だけ溜まったら、nn->forward()を呼び出す。
	}


	//  並列処理で呼び出す関数
	//  UCTアルゴリズムを反復する
	void UctSearcher::ParallelUctSearch(const Position& rootPos)
	{
		Node* current_root = get_node_tree()->GetCurrentHead();
		DlshogiSearcher* ds = grp->get_dlsearcher();

		// ルートノードを評価。これは最初にevaledでないことを見つけたスレッドが行えば良い。
		LOCK_EXPAND;
		if (!current_root->evaled) {
			current_policy_value_batch_index = 0;
			QueuingNode(&rootPos, current_root);
			EvalNode();
		}
		UNLOCK_EXPAND;

		// いずれか一つのスレッドが時間を監視する
		// このスレッドがそれを担当するかを判定する。
		//bool monitoring_thread = false;
		//if (!init_search_begin_time.exchange(true)) {
		//	search_begin_time.restart();
		//	last_pv_print = 0;
		//	monitoring_thread = true;
		//}

		// 単にメインスレッドが時刻を監視したりPVを出力したりする。
		bool main_thread = (thread_id == 0);
		auto& last_pv_print = ds->search_limit.last_pv_print;
		auto  pv_interval   = ds->search_options.pv_interval;

		if (main_thread)
			last_pv_print = 0;

		// 探索経路のバッチ
		vector<vector<NodeTrajectory>> trajectories_batch;
		vector<vector<NodeTrajectory>> trajectories_batch_discarded;
		trajectories_batch.reserve(policy_value_batch_maxsize);
		trajectories_batch_discarded.reserve(policy_value_batch_maxsize);

		// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
		do {
			trajectories_batch.clear();
			trajectories_batch_discarded.clear();
			current_policy_value_batch_index = 0;

			// バッチサイズ分探索を繰り返す
			for (int i = 0; i < policy_value_batch_maxsize; i++) {

				// 盤面のコピー

				// rootPosはスレッドごとに用意されたもので、呼び出し元にインスタンスが存在しているので、
				// 単純なコピーで問題ない。
				Position pos;
				memcpy(&pos, &rootPos, sizeof(Position));

				// 1回プレイアウトする
				trajectories_batch.emplace_back();
				float result = UctSearch(&pos, current_root, 0, trajectories_batch.back());

				if (result != DISCARDED) {
					// 探索回数を1回増やす
					atomic_fetch_add(&ds->search_limit.node_searched, 1);
				}
				else {
					// 破棄した探索経路を保存
					trajectories_batch_discarded.emplace_back(trajectories_batch.back());
				}

				// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
				if (result == DISCARDED || result != QUEUING) {
					trajectories_batch.pop_back();
				}
			}

			// 評価
			EvalNode();

			// 破棄した探索経路のVirtual Lossを戻す
			for (auto& trajectories : trajectories_batch_discarded) {
				for (int i = (int)trajectories.size() - 1; i >= 0; i--) {
					NodeTrajectory& current_next = trajectories[i];
					Node* current        = current_next.node;
					ChildNode* uct_child = current->child.get();
					const u16 next_index = current_next.index;
					SubVirtualLoss(&uct_child[next_index], current);
				}
			}

			// バックアップ
			float result = 0.0f;
			for (auto& trajectories : trajectories_batch) {
				for (int i = (int)trajectories.size() - 1; i >= 0; i--) {
					NodeTrajectory& current_next = trajectories[i];
					Node* current = current_next.node;
					const u16 next_index = current_next.index;
					ChildNode* uct_child = current->child.get();
					if (i == (int)trajectories.size() - 1) {
						const Node* child_node = uct_child[next_index].node.get();
						const float value_win = child_node->value_win;
						// 他スレッドの詰みの伝播によりvalue_winがVALUE_WINまたはVALUE_LOSEに上書きされる場合があるためチェックする
						if (value_win == VALUE_WIN)
							result = 0.0f;
						else if (value_win == VALUE_LOSE)
							result = 1.0f;
						else
							result = 1.0f - value_win;
					}
					UpdateResult(&uct_child[next_index], result, current);
					result = 1.0f - result;
				}
			}

			// PV表示
			if (main_thread && ds->search_options.pv_interval > 0) {
				const auto elapsed_time = ds->search_limit.begin_time.elapsed();
				// いずれかのスレッドが1回だけ表示する
				if (elapsed_time > last_pv_print + pv_interval) {
					const auto prev_last_pv_print = last_pv_print;
					last_pv_print = elapsed_time;

					// PV表示
					//get_and_print_pv();
					UctPrint::get_best_move_multipv(current_root , ds->search_limit , ds->search_options );
				}
			}

			// 探索を打ち切るか確認
			// この確認は、main_threadのみが行う。
			if (main_thread)
				ds->search_limit.interruption = ds->InterruptionCheck();

			// 探索打ち切り
			//   Threads.stop              : "stop"コマンドが送られてきたか、main threadがすでに指し手を返したので待機中である。
			//   search_limit.interruption : 探索の打ち切り条件を満たしたとmain threadが判定した。
			if ( Threads.stop || ds->search_limit.interruption)
				break;

		} while (true);

	}

	// UCT探索を行う関数
	// 1回の呼び出しにつき, 1プレイアウトする。
	// (leaf nodeで呼び出すものとする)
	//   pos          : UCT探索を行う開始局面
	//   current      : UCT探索を行う開始局面
	//   depth        : 
	//   trajectories : 探索開始局面(tree.GetCurrentHead())から、currentに至る手順。あるNodeで何番目のchildを選択したかというpair。
	//
	// 返し値 : currentの局面の期待勝率を返すが、以下の特殊な定数を取ることがある。
	//   QUEUING      : 評価関数を呼び出した。(呼び出しはqueuingされていて、完了はしていない)
	//   DISCARDED    : 他のスレッドがすでにこのnodeの評価関数の呼び出しをしたあとであったので、何もせずにリターンしたことを示す。
	// 
	float UctSearcher::UctSearch(Position* pos, Node* current, const int depth, std::vector<NodeTrajectory>& trajectories)
	{
		// policy計算中のため破棄する(他のスレッドが同じノードを先に展開中である場合)
		if (!current->evaled)
			return DISCARDED;

		auto ds = grp->get_dlsearcher();
		auto& options = ds->search_options;

		// 探索開始局面ではないなら、このnodeが詰みか千日手でないかのチェツクを行う。
		if (current != get_node_tree()->GetCurrentHead()) {
			// 詰みのチェック
			if (current->value_win == VALUE_WIN) {
				// 詰み、もしくはRepetitionWinかRepetitionSuperior
				return 0.0f;  // 反転して値を返すため0を返す
			}
			else if (current->value_win == VALUE_LOSE) {
				// 自玉の詰み、もしくはRepetitionLoseかRepetitionInferior
				return 1.0f; // 反転して値を返すため1を返す
			}

			// 千日手チェック
			if (current->value_win == VALUE_DRAW) {
				if (pos->side_to_move() == BLACK) {
					// 白が選んだ手なので、白の引き分けの価値を返す
					return ds->draw_value_white();
				}
				else {
					// 黒が選んだ手なので、黒の引き分けの価値を返す
					return ds->draw_value_black();
				}
			}

			// 詰みのチェック
			if (current->child_num == 0) {
				return 1.0f; // 反転して値を返すため1を返す
			}
		}

		float result;
		u16 next_index;
		//double score;
		// →　この変数、使ってない

		// 初回に訪問した場合、子ノードを展開する（メモリを節約するためExpandNodeでは候補手のみ準備して子ノードは展開していない）
		ChildNode* uct_child = current->child.get();

		// 現在見ているノードをロック
		// これは、このNode(current)の展開(child[i].node = new Node(); ... )を行う時にLockすることになっている。
		current->Lock();

		// 子ノードのなかからUCB値最大の手を求める
		next_index = SelectMaxUcbChild(pos, current, depth);


#if defined(LOG_PRINT)
		logger.print("do_move = " + to_usi_string(uct_child[next_index].move));
#endif

		// 選んだ手を着手
		StateInfo st;
		pos->do_move(uct_child[next_index].move, st);

		// Virtual Lossを加算
		AddVirtualLoss(&uct_child[next_index], current);

		// ノードの展開の確認
		// この子ノードがまだ展開されていないなら、この子ノードを展開する。
		if (!uct_child[next_index].node) {
			// ノードの展開
			Node* child_node = uct_child[next_index].CreateChildNode();
			//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

			// ノードを展開したので、もうcurrentは書き換えないからunlockして良い。

			// 現在見ているノードのロックを解除
			current->UnLock();

			// 経路を記録
			trajectories.emplace_back(current, next_index);

			// 千日手チェック
			switch (pos->is_repetition(16))
			{
				case REPETITION_WIN     : // 連続王手の千日手で反則勝ち
				case REPETITION_SUPERIOR: // 優等局面は勝ち扱い
					// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
					child_node->value_win = VALUE_WIN;
					result = 0.0f;
					break;

				case REPETITION_LOSE    : // 連続王手の千日手で反則負け
				case REPETITION_INFERIOR: // 劣等局面は負け扱い
					// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
					child_node->value_win = VALUE_LOSE;
					result = 1.0f;
					break;

				case REPETITION_DRAW    : // 引き分け
					child_node->value_win = VALUE_DRAW;
					// 現在の局面が先手番であるとしたら、この指し手は後手が選んだ指し手による千日手成立なので後手の引き分けのスコアを用いる。
					result = (pos->side_to_move() == BLACK) ? ds->draw_value_white() : ds->draw_value_black();
					break;

				case REPETITION_NONE    : // 繰り返しはない
				{
					// 詰みチェック

					// TODO : あとでN手詰め実装する。

#if 1
					bool isMate =
						(!pos->in_check() && pos->mate1ply() != MOVE_NONE) // 1手詰め
						|| (pos->DeclarationWin() != MOVE_NONE)            // 宣言勝ち
						;
#else
					bool isMate = false;
#endif

					// 詰みの場合、ValueNetの値を上書き
					if (isMate) {
						child_node->value_win = VALUE_WIN;
						result = 0.0f;
					}
					else {
						// 候補手を展開する（千日手や詰みの場合は候補手の展開が不要なため、タイミングを遅らせる）
						child_node->ExpandNode(pos);
						if (child_node->child_num == 0) {
							// 詰み
							child_node->value_win = VALUE_LOSE;
							result = 1.0f;
						}
						else
						{
							// ノードをキューに追加
							QueuingNode(pos, child_node);

							// このとき、まだEvalNodeが完了していないのでchild_node->evaledはまだfalseのまま
							// にしておく必要がある。

							return QUEUING;
						}
					}

					break;
				}

				default: UNREACHABLE;
			}

			// ノードの展開は終わって、詰みであるなら、value_winにそれが反映されている。
			child_node->evaled = true;

		}
		else {
			// 現在見ているノードのロックを解除
			current->UnLock();

			// 経路を記録
			trajectories.emplace_back(current, next_index);

			// 手番を入れ替えて1手深く読む
			result = UctSearch(pos, uct_child[next_index].node.get() , depth + 1, trajectories);
		}

		if (result == QUEUING)
			return result;
		else if (result == DISCARDED) {
			// Virtual Lossはバッチ完了までそのままにする
			return result;
		}

		// 探索結果の反映
		// currentとchildのwinに、resultを加算。
		UpdateResult(&uct_child[next_index], result, current);

		// 手番を入れ替えて再帰的にUctSearch()を呼び出した結果がresultなので、ここで返す期待勝率は1.0 - resultになる。
		return 1.0f - result;
	}

	//  UCBが最大となる子ノードのインデックスを返す関数
	//    pos     : 調べたい局面
	//    current : 調べたい局面
	//    depth   : root(探索開始局面から)からの手数。0ならposとcurrentがrootであることを意味する。 
	//  current->value_winに、子ノードを調べた結果が代入される。
	int UctSearcher::SelectMaxUcbChild(const Position *pos, Node* current, const int depth)
	{
		// 子ノード一覧
		const ChildNode *uct_child = current->child.get();
		// 子ノードの数
		const int child_num = current->child_num;

		int max_child = 0;

		// move_countの合計
		const NodeCountType sum = current->move_count;
		float q, u, max_value;
		float ucb_value;
		int child_win_count = 0;

		max_value = -FLT_MAX;

		auto ds = grp->get_dlsearcher();
		auto& options = ds->search_options;

		float fpu_reduction = (depth > 0 ? options.c_fpu_reduction : options.c_fpu_reduction_root) * sqrtf((float)current->visited_nnrate);

		// UCB値最大の手を求める
		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].node) {
				const Node* child_node = uct_child[i].node.get();
				const float child_value_win = child_node->value_win;
				if (child_value_win == VALUE_WIN) {
					child_win_count++;
					// 負けが確定しているノードは選択しない
					continue;
				}
				else if (child_value_win == VALUE_LOSE) {
					// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
					current->value_win = VALUE_WIN;
				}
			}

			float win = uct_child[i].win;
			NodeCountType move_count = uct_child[i].move_count;

			if (move_count == 0) {
				// 未探索のノードの価値に、親ノードの価値を使用する
				if (current->win > 0)
					q = std::max(0.0f, current->win / current->move_count - fpu_reduction);
				else
					q = 0.0f;
				u = sum == 0 ? 1.0f : (float)sqrt((double)sum); /* ここdoubleにしておかないと精度足りないように思う */
			}
			else {
				q = win / move_count;
				u = (float)(sqrt((double)sum) / (1 + move_count));
			}

			const float rate = uct_child[i].nnrate;

			const float c = depth > 0 ?
				FastLog((sum + options.c_base + 1.0f) / options.c_base) + options.c_init :
				FastLog((sum + options.c_base_root + 1.0f) / options.c_base_root) + options.c_init_root;
			ucb_value = q + c * u * rate;

			if (ucb_value > max_value) {
				max_value = ucb_value;
				max_child = i;
			}
		}

		if (child_win_count == child_num) {
			// 子ノードがすべて勝ちのため、自ノードを負けにする
			current->value_win = VALUE_LOSE;
		}

		// for FPU reduction
		if (uct_child[max_child].node) {
			atomic_fetch_add(&current->visited_nnrate, uct_child[max_child].nnrate);
		}

		return max_child;
	}

	// 評価関数を呼び出す。
	// batchに積まれていた入力特徴量をまとめてGPUに投げて、結果を得る。
	void UctSearcher::EvalNode()
	{
		// 何もデータが積まれていないなら呼び出してはならない。
		if (current_policy_value_batch_index == 0)
			return;

		// batchに積まれているデータの個数
		const int policy_value_batch_size = current_policy_value_batch_index;

#if defined(LOG_PRINT)
		// 入力特徴量
		std::stringstream ss;
		for (int i = 0; i < sizeof(NN_Input1) / sizeof(DType); ++i)
			ss << ((DType*)features1)[i] << ",";
		ss << endl << "Input2" << endl;
		for (int i = 0; i < sizeof(NN_Input2) / sizeof(DType); ++i)
			ss << ((DType*)features2)[i] << ",";
		logger.print(ss.str());
#endif

		// predict
		// policy_value_batch_sizeの数だけまとめて局面を評価する
		grp->nn_forward(policy_value_batch_size, features1, features2, y1, y2);

		//cout << *y2 << endl;

		const NN_Output_Policy *logits = y1;
		const NN_Output_Value  *value  = y2;

		for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
			Node* node  = policy_value_batch[i].node;
			Color color = policy_value_batch[i].color;

			node->Lock();

			const int child_num = node->child_num;
			ChildNode *uct_child = node->child.get();

			// 合法手それぞれに対する遷移確率
			std::vector<float> legal_move_probabilities;
			// いまからemplace_backしていく回数がchild_numであることはわかっているので事前に要素を確保しておく。
			legal_move_probabilities.reserve(child_num);

#if defined(LOG_PRINT)
			// あとで消す
			vector<int> move_labels;
			vector<MoveMoveLabel> moves;
			for (int j = 0; j < child_num; j++) {
				Move move = uct_child[j].move;
				const int move_label = make_move_label(move, color);
				moves.emplace_back(move, move_label);
			}
			// move label順でsortして、再現性を持たせる。
			std::sort(moves.begin(), moves.end());
			for (int j = 0; j < child_num; j++)
			{
				uct_child[j].move = moves[j].move;
				move_labels.push_back(make_move_label(moves[j].move,color));
			}
#endif

			for (int j = 0; j < child_num; j++) {
				Move move = uct_child[j].move;
				const int move_label = make_move_label(move, color);
				const float logit = (*logits)[move_label];
				legal_move_probabilities.emplace_back(logit);

				// デバッグ用に出力させてみる。
				//cout << uct_child[j].move << " " << move_label << " " << logit << endl;
			}

			// Boltzmann distribution
			softmax_temperature_with_normalize(legal_move_probabilities);

			for (int j = 0; j < child_num; j++) {
				uct_child[j].nnrate = legal_move_probabilities[j];
			}

			node->value_win = *value;

#if defined(LOG_PRINT)
			std::vector<MoveIntFloat> m;
			for (int j = 0; j < child_num; ++j)
				m.emplace_back(uct_child[j].move, move_labels[j], uct_child[j].nnrate);
			logger.print(m);
			logger.print("NN value = " + std::to_string(node->value_win));
			static int visit_count = 0;
			++visit_count;
			logger.print("visit = " + std::to_string(visit_count));
#endif


			// あとで
	#ifdef MAKE_BOOK
			// 定跡作成時は、事前確率に定跡の遷移確率も使用する
			constexpr float alpha = 0.5f;
			const Key& key = policy_value_book_key[i];
			const auto itr = bookMap.find(key);
			if (itr != bookMap.end()) {
				const auto& entries = itr->second;
				// countから分布を作成
				std::map<u16, u16> count_map;
				int sum = 0;
				for (const auto& entry : entries) {
					count_map.insert(std::make_pair(entry.fromToPro, entry.count));
					sum += entry.count;
				}
				// policyと定跡から作成した分布の加重平均
				for (int j = 0; j < child_num; ++j) {
					const Move& move = uct_child[j].move;
					const auto itr2 = count_map.find((u16)move.proFromAndTo());
					const float bookrate = itr2 != count_map.end() ? (float)itr2->second / sum : 0.0f;
					uct_child[j].nnrate = (1.0f - alpha) * uct_child[j].nnrate + alpha * bookrate;
				}
			}
	#endif
			node->evaled = true;
			node->UnLock();
		}
	}
}


#endif // defined(YANEURAOU_ENGINE_DEEP)
