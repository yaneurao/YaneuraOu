#include "dlshogi_searcher.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "dlshogi_types.h"
#include "UctSearch.h"
#include "PrintInfo.h"

#include "../../search.h"
#include "../../thread.h"

namespace dlshogi
{
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

	// GPUの初期化、各UctSearchThreadGroupに属するそれぞれのスレッド数と、各スレッドごとのNNのbatch sizeの設定
	// "isready"に対して呼び出される。
	void DlshogiSearcher::InitGPU(const std::vector<std::string>& model_paths , std::vector<int> new_thread, std::vector<int> policy_value_batch_maxsizes)
	{
		ASSERT_LV3(model_paths.size() == max_gpu);
		ASSERT_LV3(new_thread.size() == max_gpu);
		ASSERT_LV3(policy_value_batch_maxsizes.size() == max_gpu);

		// この時、前回設定と異なるなら、スレッドの再確保を行う必要がある。

		// トータルのスレッド数go
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

				std::string path = model_paths[i];

				// paths[1]..paths[7]は、空であればpaths[0]と同様であるものとする。
				if (i > 0 && path == "")
					path = model_paths[0];

				search_groups[i].Initialize(path , new_thread[i],/* gpu_id = */i, policy_value_batch_maxsize);
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

	// PV表示間隔設定
	void DlshogiSearcher::SetPvInterval(const TimePoint interval)
	{
		search_options.pv_interval = interval;
	}

	// DebugMessageの出力。
	// エンジンオプションの"DebugMessage"の値をセットする。
	// search_options.debug_messageに反映される。
	void DlshogiSearcher::SetDebugMessage(bool flag)
	{
		search_options.debug_message = flag;
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

	// 探索の"go"コマンドの前に呼ばれ、今回の探索の打ち切り条件を設定する。
	//    limits.nodes        : 探索を打ち切るnode数   　→  search_limit.node_limitに反映する。
	//    limits.movetime     : 思考時間固定時の指定     →　search_limit.time_limitに反映する。
	//    limits.max_game_ply : 引き分けになる手数の設定 →  search_limit.draw_plyに反映する。
	// などなど。
	// その他、"go"コマンドで渡された残り時間等から、今回の思考時間を算出し、search_limit.time_managerに反映する。
	void DlshogiSearcher::SetLimits(const Position* pos, const Search::LimitsType& limits)
	{
		auto& s = search_limit;
		auto& o = search_options;

		// go infiniteされているのか
		s.infinite = limits.infinite;

		// 探索を打ち切るnode数
		s.node_limit = (NodeCountType)limits.nodes;

		// 思考時間固定の時の指定
		s.movetime = limits.movetime;

		// 引き分けになる手数の設定。
		o.draw_ply = limits.max_game_ply;

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

		// 持ち時間制御は、やねうら王の time managerをそのまま用いる。
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
		search_limit.time_manager.reset();

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
			// 投了しておく。
			return MOVE_RESIGN;
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
		if (search_options.debug_message)
			UctPrint::PrintPlayoutLimits(search_limit.time_manager , search_limit.node_limit);

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

		// この時点で探索スレッドをすべて停止させないと
		// Virtual Lossを元に戻す前にbestmoveを選出してしまう。

		// PVの取得と表示
		auto best = UctPrint::get_best_move_multipv(current_root , search_limit , search_options);
		ponderMove = best.ponder;

		// 探索にかかった時間を求める
		const TimePoint finish_time = search_limit.time_manager.elapsed();
		//search_limit.remaining_time[pos->side_to_move()] -= finish_time;

		// デバッグ用のメッセージ出力
		if (search_options.debug_message)
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
	void DlshogiSearcher::InterruptionCheck()
	{
		// "go ponder"で呼び出されている場合、あらゆる制限は無視する。
		// NodesLimitとかは有効であるほうが便利かもしれないが、そういうときにgo ponderで呼び出さないと思う。
		if (search_limit.pondering)
			return;

		auto& s = search_limit;
		auto& o = search_options;

		// すでに中断することが確定しているのでここでは何もしない。
		if (s.interruption || s.time_manager.search_end)
			return;

		// リミットなしなので"stop"が来るまで停止しない。
		if (s.infinite)
			return;

		// 探索depth固定
		// →　PV掘らないとわからないので実装しない。

		if (
			// 探索ノード固定(NodesLimitで設定する)
			//   ※　この時、時間制御は行わない
			(s.node_limit && s.node_searched >= s.node_limit) ||

			// hashfull
			((NodeCountType)s.current_root->move_count >= o.uct_node_limit)
			)
		{
			// これは、時間制御の対象外。
			// ただちに中断すべき。

			s.interruption = true;
			return;
		}
			
		// -- 時間制御

		// 以下の思考時間は、ponderhitからの時間で考える。
		const auto elapsed = s.time_manager.elapsed_from_ponderhit();

		// 探索時間固定
		// "go movetime XXX"のように指定するのでGUI側が対応していないかもしれないが。
		// 探索時間固定なのにponderしていることはないと思うので、ponder後の時間経過を見て停止させて良いと思う。
		if (s.movetime && elapsed > s.movetime)
		{
			// 時間固定なので、それを超過していたら、ただちに中断すべき。
			s.interruption = true;
			return;
		}

		const Node* current_root = tree->GetCurrentHead();
		const int child_num = current_root->child_num;
		if (child_num == 1)
		{
			// one replyなので最小時間だけ使い切れば良い。
			s.time_manager.search_end = s.time_manager.minimum();
			return;
		}

		// 最小思考時間を使っていない。
		if (elapsed < s.time_manager.minimum())
			return ;

		// 最大思考時間を超過している。
		if (elapsed > s.time_manager.maximum())
		{
			// この場合も余計な時間制御は不要。ただちに中断すべき。
			s.interruption = true;
			return;
		}

		// 最適時間の1/3は必ず使うべき。
		if (elapsed < s.time_manager.optimum() * 1/3)
			return ;

		// 最適時間のrate倍を基準時間と考える。
		// なぜなら、残り時間を使っても1番目の指し手が2番目の指し手を超えないことが判明して探索を打ち切ることがあるから(頻度はそれほど高くない)
		// 平均的にoptimum()の時間が使えていない。そこでoptimumに係数をかけて、それを基準に考える。
		const float rate = 1.3f;
		auto optimum = std::min((TimePoint)(s.time_manager.optimum() * rate), s.time_manager.maximum());

		if (elapsed + 300 > optimum)
		{
			// あと300msほどで超えそうなので、もう中断準備に入る。

			// "ponderhit"しているときは、そこからの経過時間を丸める。
			// "ponderhit"していないときは開始からの経過時間を丸める。
			// そのいずれもtime_manager.elapsed_from_ponderhit()から計算して良い。

			s.time_manager.search_end = s.time_manager.round_up(elapsed);
			return;
		}

		// 残りの探索を全て次善手に費やしても optimum_timeまでに
		// 最善手を超えられない場合は探索を打ち切る。
		// TODO : 勝率の差とかも考慮したほうが良いのではないだろうか…。

		NodeCountType max_searched = 0, second = 0;
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

		// e       : ponder前からの経過時間
		// rest_po : 残り何回ぐらいplayoutできそうか。

		// nps = node_searched / (e + 1) 
		// ※　0のとき0除算になるので分母を +1 してある。
		// rest_po = nps × 残り時間

		auto e = s.time_manager.elapsed();
		s64 rest_po = (s64)(search_limit.node_searched * (optimum - e) / (e + 1) );
		rest_po = std::max(rest_po, (s64)0);

		if (max_searched - second > rest_po) {
			if (o.debug_message)
				sync_cout << "info string interrupt_no_movechange , max_searched = " << max_searched << " , second = " << second
						  << " , rest po = " << rest_po << sync_endl;

			// だめぽ。時間を秒のギリギリまで使ったら、もう探索を切り上げる。

			// "ponderhit"しているときは、そこからの経過時間を丸める。
			// "ponderhit"していないときは開始からの経過時間を丸める。
			// そのいずれもtime_manager.elapsed_from_ponderhit()から計算して良い。
			s.time_manager.search_end = s.time_manager.round_up(elapsed);
		}
	}

#if 0
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
#endif

	// 並列探索を行う。
	//   rootPos   : 探索開始局面
	//   thread_id : スレッドID
	// ※ やねうら王独自拡張
	void DlshogiSearcher::parallel_search(const Position& rootPos, size_t thread_id)
	{
		// このrootPosはスレッドごとに用意されているからコピー可能。

		thread_id_to_uct_searcher[thread_id]->ParallelUctSearch(rootPos);
	}

}

#endif // defined(YANEURAOU_ENGINE_DEEP)
