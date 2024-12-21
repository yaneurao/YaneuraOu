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

// -------------------------------------------------------------------------------------------
//  LOG_PRINT
//   ※　たぶんあとで消す
// -------------------------------------------------------------------------------------------

#if defined(LOG_PRINT)
#include <sstream>
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
		fs << s << std::endl;
	}
	void print(std::vector<MoveIntFloat>& m)
	{
		//std::sort(m.begin(), m.end());

		for (auto ml : m)
			fs << ml.to_string() << std::endl;

		fs.flush();
	}

	std::ofstream fs;
};
MyLogger logger;
#endif

// -------------------------------------------------------------------------------------------

using namespace Eval::dlshogi;
using namespace std;

#define LOCK_EXPAND grp->get_dlsearcher()->mutex_expand.lock();
#define UNLOCK_EXPAND grp->get_dlsearcher()->mutex_expand.unlock();

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
	inline void UpdateResult(ChildNode* child, float result, Node* current)
	{
		atomic_fetch_add(&current->win, (WinType)result);
		if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
		atomic_fetch_add(&child  ->win, (WinType)result);
		if constexpr (VIRTUAL_LOSS != 1) child  ->move_count += 1 - VIRTUAL_LOSS;
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
		if (searchers.size() != (size_t)new_thread || policy_value_batch_maxsize != this->policy_value_batch_maxsize)
		{
			searchers.clear();
			searchers.reserve(new_thread); // いまから追加する要素数はわかっているので事前に確保しておく。

			for (int i = 0; i < new_thread; ++i)
				searchers.emplace_back(this, i, policy_value_batch_maxsize);

			this->policy_value_batch_maxsize = policy_value_batch_maxsize;
		}

		for (int i = 0; i < new_thread; ++i) {
			searchers[i].DummyForward();
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
	void UctSearcher::QueuingNode(const Position *pos, Node* node, float* value_win)
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

		make_input_features(*pos, current_policy_value_batch_index, packed_features1, packed_features2);

		// 現在のNodeと手番を保存しておく。
		policy_value_batch[current_policy_value_batch_index] = { node, pos->side_to_move() ,
#if defined(USE_POLICY_BOOK)
			pos->hash_key() ,
#endif
			value_win};

	#ifdef MAKE_BOOK
		policy_value_book_key[current_policy_value_batch_index] = Book::bookKey(*pos);
	#endif

		current_policy_value_batch_index++;
		// これが、policy_value_batch_maxsize分だけ溜まったら、nn->forward()を呼び出す。
	}

	// leaf node用の詰め将棋ルーチンの初期化(alloc)を行う。
	// ※　SetLimits()が"go"に対してしか呼び出されていないからmax_moves_to_drawは未確定なので
	//     ここでそれを用いた設定をするわけにはいかない。
	void UctSearcher::InitMateSearcher(const SearchOptions& options)
	{
		// -- leaf nodeでdf-pn solverを用いる時はメモリの確保が必要

		// 300 nodeと5手詰めがだいたい等価(時間的に)
		// でもマルチスレッドだとメモリアクセスが足を引っ張るようで50nodeぐらいでないと…。
		// 50 nodeデフォルトでいいや。強さこれで5手詰めとほぼ変わらないし、nps 5%ほど速いので…。
		// 10000/* nodes */ * 16 /* bytes */ * 10 /* 平均分岐数 */ / (1024 * 1024) = 1.525[MB] お、、おう…。1万nodeぐらいまでは1MBあればいけるか。
		mate_solver.alloc(1);
	}

	// "go"に対して探索を開始する時に呼び出す。
	// "go"に対してしかmax_moves_to_drawは未確定なので、それが確定してから呼び出す。
	void UctSearcher::SetMateSearcher(const SearchOptions& options)
	{
		// -- leaf nodeでdf-pn solverを用いる時

		// 引き分けの手数の設定
		mate_solver.set_max_game_ply(options.max_moves_to_draw);

	}

	// 初期化時に呼び出す。
	// policy_value_batch_maxsize と同数のダミーデータを作成し、推論を行う。
	// ORT-TensorRT では、 最大バッチサイズ(policy_value_batch_maxsize) と 最小バッチサイズ(1) で
	// それぞれ推論を実行してTensorRT推論エンジンを暖気する必要がある。
	void UctSearcher::DummyForward()
	{
		// 適当なダミー局面を生成するルーチン
		auto dummy_sfen = [](u32 value) {
			Piece board[SQ_NB];
			for (size_t sq = SQ_ZERO; sq < SQ_NB; ++sq)
				board[sq] = NO_PIECE;
			board[FILE_1 | ((((value >>  0) & 1) == 0) ? RANK_9 : RANK_8)] = B_LANCE;
			board[FILE_9 | ((((value >>  1) & 1) == 0) ? RANK_9 : RANK_8)] = B_LANCE;
			board[FILE_2 | RANK_9] = B_KNIGHT;
			board[FILE_8 | RANK_9] = B_KNIGHT;
			board[FILE_3 | ((((value >>  2) & 1) == 0) ? RANK_9 : RANK_8)] = B_SILVER;
			board[FILE_7 | ((((value >>  3) & 1) == 0) ? RANK_9 : RANK_8)] = B_SILVER;
			board[FILE_4 | ((((value >>  4) & 1) == 0) ? RANK_9 : RANK_8)] = B_GOLD;
			board[FILE_6 | ((((value >>  5) & 1) == 0) ? RANK_9 : RANK_8)] = B_GOLD;
			board[FILE_5 | ((((value >>  6) & 1) == 0) ? RANK_9 : RANK_8)] = B_KING;
			board[FILE_2 | RANK_8] = B_ROOK;
			board[FILE_8 | RANK_8] = B_BISHOP;
			board[FILE_1 | ((((value >>  7) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_9 | ((((value >>  8) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_2 | ((((value >>  9) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_8 | ((((value >> 10) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_3 | ((((value >> 11) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_7 | ((((value >> 12) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_4 | ((((value >> 13) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_6 | ((((value >> 14) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_5 | ((((value >> 15) & 1) == 0) ? RANK_7 : RANK_6)] = B_PAWN;
			board[FILE_9 | ((((value >> 16) & 1) == 0) ? RANK_1 : RANK_2)] = W_LANCE;
			board[FILE_1 | ((((value >> 17) & 1) == 0) ? RANK_1 : RANK_2)] = W_LANCE;
			board[FILE_8 | RANK_1] = W_KNIGHT;
			board[FILE_2 | RANK_1] = W_KNIGHT;
			board[FILE_7 | ((((value >> 18) & 1) == 0) ? RANK_1 : RANK_2)] = W_SILVER;
			board[FILE_3 | ((((value >> 19) & 1) == 0) ? RANK_1 : RANK_2)] = W_SILVER;
			board[FILE_6 | ((((value >> 20) & 1) == 0) ? RANK_1 : RANK_2)] = W_GOLD;
			board[FILE_4 | ((((value >> 21) & 1) == 0) ? RANK_1 : RANK_2)] = W_GOLD;
			board[FILE_5 | ((((value >> 22) & 1) == 0) ? RANK_1 : RANK_2)] = W_KING;
			board[FILE_8 | RANK_2] = W_ROOK;
			board[FILE_2 | RANK_2] = W_BISHOP;
			board[FILE_9 | ((((value >> 23) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_1 | ((((value >> 24) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_8 | ((((value >> 25) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_2 | ((((value >> 26) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_7 | ((((value >> 27) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_3 | ((((value >> 28) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_6 | ((((value >> 29) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_4 | ((((value >> 30) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;
			board[FILE_5 | ((((value >> 31) & 1) == 0) ? RANK_3 : RANK_4)] = W_PAWN;

			std::ostringstream ss;
			int emptyCnt;
			for (Rank r = RANK_1; r <= RANK_9; ++r)
			{
				for (File f = FILE_9; f >= FILE_1; --f)
				{
					for (emptyCnt = 0; f >= FILE_1 && board[f | r] == NO_PIECE; --f)
						++emptyCnt;
					if (emptyCnt)
						ss << emptyCnt;
					if (f >= FILE_1)
						ss << board[f | r];
				}
				if (r < RANK_9)
					ss << '/';
			}
			ss << " b - " << 1;

			return ss.str();
		};

		if (policy_value_batch_maxsize < 1)
			return;

		// ダミー局面推論開始時間
		TimePoint tpforwardbegin = now();
		// ダミー局面設定
		Position pos;
		StateInfo si;
		for (int i = 0; i < policy_value_batch_maxsize; ++i) {
			pos.set(dummy_sfen((u32)i), &si, Threads.main());
			make_input_features(pos, i, packed_features1, packed_features2);
		}
		// このスレッドとGPUとを紐付ける。
		grp->set_device();
		// 最大バッチサイズ(policy_value_batch_maxsize) と 最小バッチサイズ(1) でそれぞれ推論を実行しておく
		grp->nn_forward(policy_value_batch_maxsize, packed_features1, packed_features2, features1, features2, y1, y2);
		grp->nn_forward(1, packed_features1, packed_features2, features1, features2, y1, y2);
		// ダミー局面推論終了時間
		TimePoint tpforwardend = now();

		sync_cout << "info string engine forward test. batch_size = " << policy_value_batch_maxsize << ", Processing time = " << tpforwardend - tpforwardbegin << "ms." << sync_endl;
	}

	// UCTアルゴリズムによる並列探索の各スレッドのEntry Point
	// ※　Thread::search()から呼び出す。
	void UctSearcher::ParallelUctSearchStart(const Position& rootPos)
	{
		// このスレッドとGPUとを紐付ける。
		grp->set_device();

		// 詰み探索部の"go"コマンド時の初期化
		auto& options = grp->get_dlsearcher()->search_options;
		SetMateSearcher(options);

		// 並列探索の開始
		ParallelUctSearch(rootPos);
	}

	// UCTアルゴリズム(UctSearch())を反復的に実行する。
	// 探索用のすべてのスレッドが並列的にこの関数を実行をする。
	// この関数とUctSearch()、SelectMaxUcbChild()が探索部本体と言えると思う。
	void UctSearcher::ParallelUctSearch(const Position& rootPos)
	{
		DlshogiSearcher* ds = grp->get_dlsearcher();
		auto& search_limits = ds->search_limits;
		auto stop = [&]() { return Threads.stop || search_limits.interruption; };

		// ↓ dlshogiのコードここから ↓

		Node* current_root = get_node_tree()->GetCurrentHead();

		// ルートノードを評価。これは最初にevaledでないことを見つけたスレッドが行えば良い。
		LOCK_EXPAND;
		if (!current_root->IsEvaled()) {
			current_policy_value_batch_index = 0;
			float value_win; // EvalNode()した時に、ここにvalueが書き戻される。ダミーの変数。
			QueuingNode(&rootPos, current_root, &value_win);
			EvalNode();
		}
		UNLOCK_EXPAND;

		// 探索経路のバッチ
		vector<NodeVisitor> visitor_batch;
		vector<NodeTrajectories> trajectories_batch_discarded;
		visitor_batch.reserve(policy_value_batch_maxsize);
		trajectories_batch_discarded.reserve(policy_value_batch_maxsize);

		// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
		while ( ! stop() )
		{
			visitor_batch.clear();
			trajectories_batch_discarded.clear();
			current_policy_value_batch_index = 0;

			// バッチサイズ分探索を繰り返す
			// stop()になったらなるべく早く終わりたいので終了判定のところに "&& !stop"を書いておく。
			// ※　VirtualLossを無くすなどして、stop()になったら直ちにリターンすべきだが、
			//    1回のbatch sizeはGPU側で0.1秒程度で完了できる量にすると思うので、普通のGPUでは誤差か。
			for (int i = 0; i < policy_value_batch_maxsize && !stop(); i++) {

				// 盤面のコピー

				// rootPosはスレッドごとに用意されたもので、呼び出し元にインスタンスが存在しているので、
				// 単純なコピーで問題ない。
				Position pos;
				std::memcpy(&pos, &rootPos, sizeof(Position));

				// 1回プレイアウトする
				visitor_batch.emplace_back();
				const float result = UctSearch(&pos, nullptr, current_root, visitor_batch.back());

				if (result != DISCARDED)
				{
					atomic_fetch_add(&search_limits.nodes_searched, (NodeCountType)1);
					//  →　ここで加算するとnpsの計算でまだEvalNodeしてないものまで加算されて
					// 大きく見えてしまうのでもう少しあとで加算したいところだが…。
				}
				else {
					// 破棄した探索経路を保存
					trajectories_batch_discarded.emplace_back(std::move(visitor_batch.back().trajectories));
				}

				// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
				if (result == DISCARDED || result != QUEUING) {
					visitor_batch.pop_back();
				}

			}

			// 評価
			EvalNode();

			// 破棄した探索経路のVirtual Lossを戻す
			for (auto& trajectories : trajectories_batch_discarded) {
				for (auto it = trajectories.rbegin(); it != trajectories.rend(); ++it)
				{
					NodeTrajectory& current_next  = *it;
					Node* current				  = current_next.node;
					ChildNode* uct_child		  = current->child.get();
					const ChildNumType next_index = current_next.index;

					SubVirtualLoss(&uct_child[next_index], current);
				}
			}

			// バックアップ
			// 通った経路(rootからleaf node)までのmove_countを加算するなどの処理。
			// AlphaZeroの論文で、"Backup"と呼ばれている。

			// leaf nodeでの期待勝率(NNの返してきたvalue)。
			// これをleaf nodeからrootに向かって、伝播していく。(Node::winに加算していく)
			for (auto& visitor : visitor_batch) {
				// leaf nodeの一つ上のnode用にvisitor.value_winから取り出す。
				float result = 1.0f - visitor.value_win;

				auto& trajectories = visitor.trajectories;
				for (auto it = trajectories.rbegin(); it != trajectories.rend() ; ++it)
				{
					auto& current_next            = *it;
					Node* current                 = current_next.node;
					const ChildNumType next_index = current_next.index;
					ChildNode* uct_child          = current->child.get();

					UpdateResult(&uct_child[next_index], result, current);

					// Value Networkの返した期待勝率を手番ごとに反転させて伝播する。
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
	//   visitor      : 探索開始局面(tree.GetCurrentHead())から、currentに至る手順。あるNodeで何番目のchildを選択したかという情報。
	//
	// 返し値 : currentの局面の期待勝率を返すが、以下の特殊な定数を取ることがある。
	//   QUEUING      : 評価関数を呼び出した。(呼び出しはqueuingされていて、完了はしていない)
	//   DISCARDED    : 他のスレッドがすでにこのnodeの評価関数の呼び出しをしたあとであったので、何もせずにリターンしたことを示す。
	//
	float UctSearcher::UctSearch(Position* pos, ChildNode* parent , Node* current, NodeVisitor& visitor)
	{
#if defined(USE_POLICY_BOOK)
		// PolicyBookからvalueを与えられていたら、それをそのまま返す。
		// ただし、root局面では、いまから探索しないといけないので、それはしない。
		if (parent!=nullptr && current->policy_book_value != FLT_MAX)
			return 1.0f - current->policy_book_value;
#endif

		auto ds = grp->get_dlsearcher();
		auto& options = ds->search_options;

		// ↓dlshogiのコード、ここから↓

		float result;

		// 初回に訪問した場合、子ノードを展開する（メモリを節約するためExpandNodeでは候補手のみ準備して子ノードは展開していない）
		ChildNode* uct_child = current->child.get();

		// ここまでの手順
		auto& trajectories = visitor.trajectories;

		// 現在見ているノードをロック
		// これは、このNode(current)の展開(child[i].node = new Node(); ... )を行う時にLockすることになっている。
		auto& mutex = ds->get_node_mutex(pos);
		mutex.lock();

		// 子ノードへのポインタ配列が初期化されていない場合、初期化する
		if (!current->child_nodes) current->InitChildNodes();

		// 子ノードのなかからUCB値最大の手を求める
		const ChildNumType next_index = SelectMaxUcbChild(parent, current);

#if defined(LOG_PRINT)
		logger.print("do_move = " + to_usi_string(uct_child[next_index].move));
#endif

		// 選んだ手を着手
		StateInfo st;
		pos->do_move(uct_child[next_index].getMove(), st);

		// Virtual Lossを加算
		// ※　ノードの訪問回数をmove_countに加算。
		// 　　この時点では勝率は加算していないのでこの指し手の勝率が相対的に低く見えるようになる。
		AddVirtualLoss(&uct_child[next_index], current);

		// ノードの展開の確認
		// この子ノードがまだ展開されていないなら、この子ノードを展開する。
		if (!current->child_nodes[next_index]) {
			// ノードの作成
			Node* child_node = current->CreateChildNode(next_index);
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

			// この局面で詰んでいる可能性がある。その時はmatedのスコアを返すべき。
			// 詰んでいないなら引き分けのスコアを返すべき。
			//
			// 関連)
			//    多くの将棋ソフトで256手ルールの実装がバグっている件
			//    https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/
			//
			// デフォルトではleaf nodeで5手詰めは呼び出しているので、その5手詰めで普通1手詰めが漏れることはないので、
			// 256手ルールの256手目で1手詰めがあるのにそれを逃して257手目の局面に到達することはありえない…ということであれば、
			// これは問題とはない。(leaf nodeで5手詰めを呼び出さないエンジン設定にした時に困るが。)

			if (options.max_moves_to_draw < pos->game_ply())
				rep = pos->is_mated() ? REPETITION_LOSE /* 負け扱い */ : REPETITION_DRAW;
			else
				rep = pos->is_repetition(16);

			switch (rep)
			{
				case REPETITION_WIN     : // 連続王手の千日手で反則勝ち
				case REPETITION_SUPERIOR: // 優等局面は勝ち扱い
					// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
					uct_child[next_index].SetWin();
					result = 0.0f;
					break;

				case REPETITION_LOSE    : // 連続王手の千日手で反則負け
				case REPETITION_INFERIOR: // 劣等局面は負け扱い
					// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
					uct_child[next_index].SetLose();
					result = 1.0f;
					break;

				case REPETITION_DRAW    : // 引き分け
					uct_child[next_index].SetDraw();
					// 引き分け時のスコア(これはroot colorに依存する)
					result = 1 - ds->draw_value(pos->side_to_move());
					break;

				case REPETITION_NONE    : // 繰り返しはない
				{
					// 詰みチェック

#if !defined(LOG_PRINT)

					bool isMate = false;

					// 浅いdfpnによる詰み探索

					// 0なら詰み探索無効
					if (options.leaf_dfpn_nodes_limit) {

						// Mate::mate_odd_ply()は自分に王手がかかっていても詰みを読めるが遅い。
						// leaf nodeでもdf-pn mate solverを用いることにする。

						// Move::none()(詰み不明) , Move::null()(不詰)ではない 。これらはis_ok(m) == false
						Move mate_move = mate_solver.mate_dfpn(*pos, options.leaf_dfpn_nodes_limit);
						if (mate_move == Move::null())
						{
							// 不詰を証明したので、このnodeでは詰み探索をしたことを記録しておく。
							// (そうするとPvMateでmate探索が端折れる)
							child_node->dfpn_proven_unsolvable = true;
						}
						else {
							isMate = mate_move.is_ok();
						}
					}
					if (!isMate)
						// 宣言勝ち
						isMate = pos->DeclarationWin() != Move::none();

#else
					// mateが絡むとdlshogiと異なるノードを探索してしまうのでログ調査する時はオフにする。
					bool isMate = (pos->DeclarationWin() != Move::none());            // 宣言勝ち
#endif

					// 詰みの場合、ValueNetの値を上書き
					if (isMate) {
						// 親nodeでnext_indexの子を選択した時に即詰みがあったので、この子ノードを勝ち扱いにして、
						// 今回の期待勝率は0%に設定する。
						uct_child[next_index].SetWin();
						result = 0.0f;
					}
					else {
						// 候補手を展開する（千日手や詰みの場合は候補手の展開が不要なため、タイミングを遅らせる）
						child_node->ExpandNode(pos,options.generate_all_legal_moves);
						if (child_node->child_num == 0) {
							// 詰み
							uct_child[next_index].SetLose();
							result = 1.0f;
						}
						else
						{
							// ノードをキューに追加
							QueuingNode(pos, child_node , &visitor.value_win);

							// このとき、まだEvalNodeが完了していないのでchild_node->evaledはまだfalseのまま
							// にしておく必要がある。

							return QUEUING;
						}
					}

					break;
				}

				default: UNREACHABLE;
			}

			// ノードの展開は終わって、詰みであるなら、SetWin()/SetLose()等でそれがChildNodeに反映されている。
			child_node->SetEvaled();

		}
		else {
			// 現在見ているノードのロックを解除
			mutex.unlock();

			// 経路を記録
			trajectories.emplace_back(current, next_index);

			Node* next_node = current->child_nodes[next_index].get();

			// policy計算中のため破棄する(他のスレッドが同じノードを先に展開した場合)
			if (!next_node->IsEvaled())
				return DISCARDED;

			if (uct_child[next_index].IsWin()) {
				// 詰み、もしくはRepetitionWinかRepetitionSuperior
				result = 0.0f;  // 反転して値を返すため0を返す
			}
			else if (uct_child[next_index].IsLose()) {
				// 自玉の詰み、もしくはRepetitionLoseかRepetitionInferior
				result = 1.0f; // 反転して値を返すため1を返す
			}
			// 千日手チェック
			else if (uct_child[next_index].IsDraw()) {
				// 反転して値を返すため、1から引き算して返す。
				result = 1 - ds->draw_value(pos->side_to_move());
			}
			// 詰みのチェック
			else if (next_node->child_num == 0) {
				result = 1.0f; // 反転して値を返すため1を返す
			}
			else {
				// 手番を入れ替えて1手深く読む
				result = UctSearch(pos, &uct_child[next_index], next_node, visitor);
			}
		}

		if (result == QUEUING)
			return result;
		else if (result == DISCARDED) {
			// Virtual Lossはバッチ完了までそのままにする
			// →　そうしないとまた同じNodeに辿り着いてしまう。
			return result;
		}

		// 探索結果の反映
		// currentとchildのwinに、resultを加算。
		UpdateResult(&uct_child[next_index], result, current);

		// 手番を入れ替えて再帰的にUctSearch()を呼び出した結果がresultなので、ここで返す期待勝率は1.0 - resultになる。
		return 1.0f - result;
	}

	//  UCBが最大となる子ノードのインデックスを返す関数
	//    parent  : 調べたい局面の親局面のcurrentに至るedge。(current == rootであるなら、nullptrを設定しておく)
	//    current : 調べたい局面
	//  返し値 : currentの局面においてUCB最大の子のindex
	//  currentノードがすべて勝ちなら、親ノードは負けなので、parent->SetLose()を呼び出す。
	ChildNumType UctSearcher::SelectMaxUcbChild(ChildNode* parent, Node* current)
	{
		// nodeはlockされているので、atomicは不要なはず。

		auto ds = grp->get_dlsearcher();
		auto& options = ds->search_options;

		// ↓dlshogiのコード、ここから↓

		// 子ノード一覧
		const ChildNode *uct_child = current->child.get();
		// 子ノードの数
		const ChildNumType child_num = current->child_num;

		// 最大となる子のindex
		ChildNumType max_child = 0;

		// move_countの合計
		const NodeCountType sum = current->move_count;

		// ExpandNode()は終わっているわけだから、move_count != NOT_EXPANDED
		ASSERT_LV3(sum != NOT_EXPANDED);

		// 勝率の集計用
		const WinType sum_win = current->win;

		float q, u, max_value;

		max_value = -FLT_MAX;

		// ループの外でsqrtしておく。
		// sumは、double型で計算しているとき、sqrt()してからfloatにしたほうがいいか？
		const float sqrt_sum = sqrtf((float)sum);
		const float c = parent == nullptr ?
			FastLog((sum + options.c_base_root + 1.0f) / options.c_base_root) + options.c_init_root :
			FastLog((sum + options.c_base + 1.0f) / options.c_base) + options.c_init;
		const float fpu_reduction = (parent == nullptr ? options.c_fpu_reduction_root : options.c_fpu_reduction) * sqrtf(current->visited_nnrate);
		const float parent_q = sum_win > 0 ? std::max(0.0f, (float)(sum_win / sum) - fpu_reduction) : 0.0f;
		const float init_u = sum == 0 ? 1.0f : sqrt_sum;

		// すべて指し手をbitwise andして、すべての指し手にIsWin()かIsLose()のフラグが立っていないかを見る。
		// すべての子ノードがIsWin()なら、このノードは負け確定。
		// すべての子ノードがIsDraw()なら、このノードは引き分け確定。
		u32 and_move = 0xffffffff;

		// UCB値最大の手を求める
		for (ChildNumType i = 0; i < child_num; i++) {

			Move move = uct_child[i].move;

			// 負けが確定しているノードは選択しない
			if (ChildNode::IsMoveWin(move))
				continue;

			if (ChildNode::IsMoveLose(move)) {
				// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
				if (parent != nullptr)
					parent->SetWin();
				return i;
			}

			// --- 引き分けについて

			//	1. 子ノードすべてが負け
			//	2. 子ノードすべてが引き分け
			//	3. 子ノードすべてが引き分け or 負け

			// 引き分け とは、 2. or (3. and (not 1.))である。
			// 上の2つのifより↓を先に書くと、3.の条件が抜けてしまい、2. になってしまうので誤り。

			and_move &= move.to_u32();

			// 引き分けに関しては特にスキップしない。普通に計算する。

			const WinType       win        = uct_child[i].win;
			const NodeCountType move_count = uct_child[i].move_count;

			if (move_count == 0) {
				// 未探索の子ノードの価値に、親ノードの価値を使用する。
				// ※　親ノードの価値 = 親ノードでのpoの勝率(win/move_count)
				q = parent_q;
				u = init_u;
			}
			else {
				q = (float)(win / move_count);
				u = sqrt_sum / (1 + move_count);
			}

			// policy networkの値
			const float rate = uct_child[i].nnrate;

			// MCTSとの組み合わせの時には、UCBの代わりにp-UCB値を用いる。
			//
			// 親ノードでi番目の指し手を指した局面を子ノードと呼ぶ。
			// 　子ノードのvalue networkの値           : v(s_i)    ==> 変数 q
			// 　親ノードの指し手iのpolicy networkの値 :   p_i     ==> 変数 rate
			// 　親nodeの訪問数                        :   n       ==> 変数 sum
			// 　子ノードの訪問数                      :   n_i     ==> 変数 move_count
			//
			//   (論文によく出てくるp-UCBの式は、)
			//         p-UCB = v(s_i) + p_i・c・sqrt(n)/(1+n_i)
			//
			//   ※　v(s_i)は、初回はvalue networkの値を使うが、そのあとは、win / move_count のほうがより正確な期待勝率なのでそれを用いる。
			//   ※　sqrt(n) ==> 変数sqrt_sum // 高速化のためループの外で計算している
			//

			const float ucb_value = q + c * u * rate;

			if (ucb_value > max_value) {
				max_value = ucb_value;
				max_child = i;
			}
		}

		if (ChildNode::IsMoveWin((Move)and_move)) {
			// 子ノードがすべて勝ちのため、自ノードを負けにする
			// このときmax_child == 0(初期値)。
			if (parent != nullptr)
				parent->SetLose();

		} else if (ChildNode::IsMoveDraw((Move)and_move)) {

			// 子ノードがすべて引き分けのため、自ノードを引き分けにする
			// このときmax_childは普通に計算される。
			if (parent != nullptr)
				parent->SetDraw();

		} else {

			// for FPU reduction
			atomic_fetch_add(&current->visited_nnrate, uct_child[max_child].nnrate);
		}

		return max_child;
	}

	// 評価関数を呼び出す。
	// batchに積まれていた入力特徴量をまとめてGPUに投げて、結果を得る。
	void UctSearcher::EvalNode()
	{
		// 何もデータが積まれていないならこのあとforwardを呼び出してはならないので帰る。
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
		grp->nn_forward(policy_value_batch_size, packed_features1, packed_features2, features1, features2, y1, y2);

		//cout << *y2 << endl;

		const NN_Output_Policy *logits = y1;
		const NN_Output_Value  *value  = y2;

		for (int i = 0; i < policy_value_batch_size; i++, logits++, value++)
		{
			      Node*        node      = policy_value_batch[i].node;
			const Color        color     = policy_value_batch[i].color;
			const ChildNumType child_num = node->child_num;
			      ChildNode *  uct_child = node->child.get();

#if defined(USE_POLICY_BOOK)
				  HASH_KEY     key       = policy_value_batch[i].key;
			auto* policy_book_entry      = grp->get_dlsearcher()->policy_book.probe_policy_book(key);
#endif

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

			for (ChildNumType j = 0; j < child_num; j++) {
				Move move = uct_child[j].move;
				const int move_label = make_move_label(move, color);
				const float logit = (*logits)[move_label];
				legal_move_probabilities.emplace_back(logit);

				// デバッグ用に出力させてみる。
				//cout << uct_child[j].move << " " << move_label << " " << logit << endl;
			}

			// Boltzmann distribution
			softmax_temperature_with_normalize(legal_move_probabilities);

#if !defined(USE_POLICY_BOOK)
			for (ChildNumType j = 0; j < child_num; j++) {
				uct_child[j].nnrate = legal_move_probabilities[j];
			}
#else
			if (policy_book_entry == nullptr)
			{
				for (ChildNumType j = 0; j < child_num; j++) {
					uct_child[j].nnrate = legal_move_probabilities[j];
				}
			} else {
				// Policy Bookに従う。

				// 評価値の書かれている局面であるか？
				float v = policy_book_entry->value;
				if (v != FLT_MAX)
					node->policy_book_value = v;

				u32 total = 0;
				size_t k1;
				for (k1 = 0; k1 < POLICY_BOOK_NUM; ++k1)
				{
					if (policy_book_entry->move_freq[k1].move16 == Move16::none())
						break;
					total += policy_book_entry->move_freq[k1].freq;
				}
				// ⇨ k1個だけ有効なmoveがあることがわかった。

				// 元のPolicyの按分率
				// 
				// 定跡の質により変化させる。
				// totalが1000回    ⇨ さすがに信用していいのでは。100%
				// totalが 100回    ⇨ 90%ぐらい信用できるか
				// totalが  10回    ⇨ 80%
				// totalが それ以下 ⇨ 70%
				// みたいな感じにする。
				// 0 < total <= u32_maxであることは保証されている。
				//
				// 注意 : 電竜戦の開始4手の玉の屈伸の棋譜を利用したときに、あれをPolicyとされてしまうと困る。
				//  (PolicyBookを作るときに除外する必要がある)

				float book_policy_ratio = 0.7f + 0.1f * std::clamp(log10f(float(total)), 0.0f, 3.0f);

				for (ChildNumType j = 0; j < child_num; j++) {

					uct_child[j].nnrate = (1.0f - book_policy_ratio) * legal_move_probabilities[j];

					// PolicyBookに出現していた指し手であれば、それで按分する。
					for (size_t k2 = 0 ; k2 < k1; ++k2)
					{
						if (uct_child[j].move.to_move16() == policy_book_entry->move_freq[k2].move16)
						{
							uct_child[j].nnrate += book_policy_ratio * policy_book_entry->move_freq[k2].freq / total;
							break;
						}
					}
				}
			}
#endif

			// valueの値はここに返すことになっている。
			*policy_value_batch[i].value_win = *value;

#if defined(LOG_PRINT)
			std::vector<MoveIntFloat> m;
			for (int j = 0; j < child_num; ++j)
				m.emplace_back(uct_child[j].move, move_labels[j], uct_child[j].nnrate);
			logger.print(m);
			logger.print("NN value = " + std::to_string(*value));
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
			node->SetEvaled();
		}
	}


	// 訪問回数が最大の子ノードを選択
	unsigned int select_max_child_node(const Node* uct_node)
	{
		const ChildNode* uct_child = uct_node->child.get();

		unsigned int select_index = 0;
		NodeCountType max_count = 0;
		const int child_num = uct_node->child_num;
		NodeCountType child_win_count = 0;
		NodeCountType child_lose_count = 0;

		for (int i = 0; i < child_num; i++) {
			if (uct_child[i].IsWin()) {
				// 負けが確定しているノードは選択しない
				if (child_win_count == NodeCountType(i) && uct_child[i].move_count > max_count) {
					// すべて負けの場合は、探索回数が最大の手を選択する
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				child_win_count++;
				continue;
			}
			else if (uct_child[i].IsLose()) {
				// 子ノードに一つでも負けがあれば、勝ちなので選択する
				if (child_lose_count == 0 || uct_child[i].move_count > max_count) {
					// すべて勝ちの場合は、探索回数が最大の手を選択する
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				child_lose_count++;
				continue;
			}

			if (child_lose_count == 0 && uct_child[i].move_count > max_count) {
				select_index = i;
				max_count = uct_child[i].move_count;
			}
		}

		return select_index;
	}

	// 訪問回数が上からN個の子ノードを返す。
	// N個ない時は、残りが-1で埋まる。
	void select_nth_child_node(const Node* uct_node, int n, int(&indices)[MAX_MOVES])
	{
		const ChildNode* uct_child = uct_node->child.get();
		const int child_num = uct_node->child_num;

		int index = 0;
		for (int i = 0; i < child_num; i++) {
			// 勝ち負けが確定していない子に対して。
			// あと、訪問回数が10回以上であること。
			if (!(uct_child[i].IsWin() || uct_child[i].IsLose() || uct_child[i].move_count < 10))
				indices[index++] = i;
		}
		std::partial_sort(indices, indices + std::min(n, index) , indices + index, [&uct_child](int a, int b) {
			return uct_child[a].move_count > uct_child[b].move_count;
			});

		// 残りをn個まで-1でpaddingする。
		for (int i = index; i < n; ++i)
			indices[i] = -1;
	}

}


#endif // defined(YANEURAOU_ENGINE_DEEP)
