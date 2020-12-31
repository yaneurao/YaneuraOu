#include "UctSearch.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "Node.h"
#include "dlshogi_searcher.h"
#include "PrintInfo.h"
#include "misc/fastmath.h"  // FastLog()

#include "../../thread.h"
#include "../../usi.h"
#include "../../mate/mate.h"

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
	// 1) result(child->value_win = NNが返してきたvalue ← 期待勝率)をcurrentとchildのwinに加算
	// 2) VIRTUAL_LOSSが1でないときは、currnetとchildのmove_countに (1 - VIRTUAL_LOSS) を加算。
	inline void UpdateResult(ChildNode* child, WinCountType result, Node* current)
	{
		atomic_fetch_add(&current->win, result);
		if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
		atomic_fetch_add(&child->win   , result);
		if constexpr (VIRTUAL_LOSS != 1) child->move_count   += 1 - VIRTUAL_LOSS;
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
	void UctSearcherGroup::Initialize(const std::string& model_path , const int new_thread , const int gpu_id, const int policy_value_batch_maxsize)
	{
		// gpu_idは呼び出しごとに変更される可能性はないと仮定してよい。
		// (固定で確保しているので)
		this->gpu_id = gpu_id;

		// モデルpath名に変更があるなら、それを読み直す。
		// ※　先にNNが構築されていないと、このあとNNからalloc()できないのでUctSearcherより先に構築する。
		// batch sizeに変更があった場合も、このbatch size分だけGPU側にメモリを確保したいので、この時もNNのインスタンスを作りなおす。
		if (this->model_path != model_path || policy_value_batch_maxsize != this->policy_value_batch_maxsize)
	{
			std::lock_guard<std::mutex> lk(mutex_gpu);

			// 以前のやつを先に開放しないと次のを確保するメモリが足りないかも知れない。
			if (nn)
				nn.reset();

			nn = NN::build_nn(model_path, gpu_id, policy_value_batch_maxsize);

			// 次回、このmodel_pathかalloced_policy_value_batch_maxsizeに変更があれば、再度NNをbuildする。
			this->model_path = model_path;
	}

		// スレッド数に変更があるか、batchサイズが前回から変更があったならばUctSearcherのインスタンス自体を生成しなおす。
		if (searchers.size() != new_thread || policy_value_batch_maxsize != this->policy_value_batch_maxsize)
		{
			searchers.clear();
			searchers.reserve(new_thread); // いまから追加する要素数はわかっているので事前に確保しておく。

			for (int i = 0; i < new_thread; ++i)
				searchers.emplace_back(this, i, policy_value_batch_maxsize);

		this->policy_value_batch_maxsize = policy_value_batch_maxsize;
	}
	}

	// やねうら王では探索スレッドはThreadPoolが管理しているのでこれらは不要。
#if 0
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

#endif

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
		policy_value_batch[current_policy_value_batch_index] = { node, pos->side_to_move() /* , pos->key() */};

	#ifdef MAKE_BOOK
		policy_value_book_key[current_policy_value_batch_index] = Book::bookKey(*pos);
	#endif

		current_policy_value_batch_index++;
		// これが、policy_value_batch_maxsize分だけ溜まったら、nn->forward()を呼び出す。
	}

	// UCTアルゴリズム(UctSearch())を反復的に実行する。
	// 探索用のすべてのスレッドが並列的にこの関数を実行をする。
	// この関数とUctSearch()、SelectMaxUcbChild()が探索部本体と言えると思う。
	void UctSearcher::ParallelUctSearch(const Position& rootPos)
	{
		Node* current_root = get_node_tree()->GetCurrentHead();
		DlshogiSearcher* ds = grp->get_dlsearcher();
		auto& search_limits = ds->search_limits;
		auto stop = [&]() { return Threads.stop || search_limits.interruption; };

		// ルートノードを評価。これは最初にevaledでないことを見つけたスレッドが行えば良い。
		LOCK_EXPAND;
		if (!current_root->evaled) {
			current_policy_value_batch_index = 0;
			QueuingNode(&rootPos, current_root);
			EvalNode();
		}
		UNLOCK_EXPAND;

		// 探索経路のバッチ
		vector<vector<NodeTrajectory>> trajectories_batch;
		vector<vector<NodeTrajectory>> trajectories_batch_discarded;
		trajectories_batch.reserve(policy_value_batch_maxsize);
		trajectories_batch_discarded.reserve(policy_value_batch_maxsize);

		// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
		while ( ! stop() )
		{
			trajectories_batch.clear();
			trajectories_batch_discarded.clear();
			current_policy_value_batch_index = 0;

			// バッチサイズ分探索を繰り返す
			for (int i = 0; i < policy_value_batch_maxsize && !stop(); i++) {

				// 盤面のコピー

				// rootPosはスレッドごとに用意されたもので、呼び出し元にインスタンスが存在しているので、
				// 単純なコピーで問題ない。
				Position pos;
				memcpy(&pos, &rootPos, sizeof(Position));

				// 1回プレイアウトする
				trajectories_batch.emplace_back();
				float result = UctSearch(&pos, current_root, 0, trajectories_batch.back());

				if (result != DISCARDED)
				{
				  atomic_fetch_add(&search_limits.nodes_searched, 1);
					//  →　ここで加算するとnpsの計算でまだEvalNodeしてないものまで加算されて
					// 大きく見えてしまうのでもう少しあとで加算したいところだが…。
				}
				else {
					// 破棄した探索経路を保存
					trajectories_batch_discarded.emplace_back(std::move(trajectories_batch.back()));
				}

				// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
				if (result == DISCARDED || result != QUEUING) {
					trajectories_batch.pop_back();
				}

			}

			// 評価
			// 探索終了の合図が来ているなら、なるべくただちに終了したいので、EvalNode()もskipして
			// virtual lossを戻して終わる。
			if (!stop())
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
			// 通った経路(rootからleaf node)までのmove_countを加算するなどの処理。
			// AlphaZeroの論文で、"Backup"と呼ばれている。

			// leaf nodeでの期待勝率(NNの返してきたvalue)。
			// ただし詰みを発見している場合はVALUE_WIN or VALUE_LOSEであるので
			// これをそれぞれ1.0f,0.0fに変換して保持する。
			// これをleaf nodeからrootに向かって、伝播していく。(Node::winに加算していく)
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
							// 親nodeでは、自分の手番から見た期待勝率なので値が反転する。
							result = 1.0f - value_win;
					}
					UpdateResult(&uct_child[next_index], result, current);
					result = 1.0f - result;
				}
			}
		}

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
		auto& mutex = ds->get_node_mutex(pos);
		mutex.lock();

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
			mutex.unlock();
			// →　CreateChildNode()で新しく作られたNodeは、evaledがfalseのままになっているので
			// 　　他の探索スレッドがここに到達した場合、DISCARDする。
			//     この新しく作られたNodeは、EvalNode()のなかで最後にevaled = trueに変更される。

			// 経路を記録
			trajectories.emplace_back(current, next_index);

			// 千日手チェック

			RepetitionState rep;

			// この局面の手数が最大手数を超えているなら千日手扱いにする。
			if (options.max_moves_to_draw < pos->game_ply())
				rep = REPETITION_DRAW;
			else
				rep = pos->is_repetition(16);

			switch (rep)
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

					bool isMate =
						// Mate::mate_odd_ply()は自分に王手がかかっていても詰みを読めるはず…。
						(options.mate_search_ply && mate_solver.mate_odd_ply(*pos,options.mate_search_ply,options.generate_all_legal_moves) != MOVE_NONE) // N手詰め
						|| (pos->DeclarationWin() != MOVE_NONE)            // 宣言勝ち
						;

					// 詰みの場合、ValueNetの値を上書き
					if (isMate) {
						child_node->value_win = VALUE_WIN;
						result = 0.0f;
					}
					else {
						// 候補手を展開する（千日手や詰みの場合は候補手の展開が不要なため、タイミングを遅らせる）
						child_node->ExpandNode(pos,options.generate_all_legal_moves);
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
			mutex.unlock();

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
		const float sum_win = current->win;
		float q, u, max_value;
		float ucb_value;
		int child_win_count = 0;

		max_value = -FLT_MAX;

		auto ds = grp->get_dlsearcher();
		auto& options = ds->search_options;

		// ループの外でsqrtしておく。
		const float sqrt_sum = (float)sqrt(sum);
		const float c = depth > 0 ?
			FastLog((sum + options.c_base + 1.0f) / options.c_base) + options.c_init :
			FastLog((sum + options.c_base_root + 1.0f) / options.c_base_root) + options.c_init_root;
		float fpu_reduction = (depth > 0 ? options.c_fpu_reduction : options.c_fpu_reduction_root) * sqrtf((float)current->visited_nnrate);
		const float parent_q = sum_win > 0 ? std::max(0.0f, sum_win / sum - fpu_reduction) : 0.0f;
		const float init_u = sum == 0 ? 1.0f : sqrt_sum;

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

			const float win = uct_child[i].win;
			const NodeCountType move_count = uct_child[i].move_count;

			if (move_count == 0) {
				// 未探索の子ノードの価値に、親ノードでのpoの勝率(win/move_count)を使用する
				q = parent_q;
				u = init_u;
			}
			else {
				q = win / move_count;
				u = sqrt_sum / (1 + move_count);
			}

			const float rate = uct_child[i].nnrate;

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
		auto ds = grp->get_dlsearcher();

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
		}
	}
}


#endif // defined(YANEURAOU_ENGINE_DEEP)
