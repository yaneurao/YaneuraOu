#ifndef __NODE_H_INCLUDED__
#define __NODE_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include <thread>
#include "../../position.h"
#include "dlshogi_types.h"

namespace dlshogi
{
	struct Node;
	class NodeGarbageCollector;

	// 子ノードを表現する。
	// より正確に言うなら、あるNodeから子ノードへのedge(辺)を表現する。
	// あるノードから実際に子ノードにアクセスするとランダムアクセスになってしまうので
	// それが許容できないから、ある程度の情報をedgeがcacheするという考え。
	// UctNodeが親ノードで、基本的には合法手の数だけChildNodeを持つ。
	// ※　dlshogiのchild_node_t
	struct ChildNode
	{
		ChildNode() : move_count(0), win(0.0f) , nnrate(0.0f) {}

		ChildNode(Move move)
			: move(move), move_count(0), win(0.0f), nnrate(0.0f){}

		// ムーブコンストラクタ
		ChildNode(ChildNode&& o) noexcept
			: move(o.move), move_count(0), win(0.0f), nnrate(0.0f), node(std::move(o.node)) {}

		// ムーブ代入演算子
		ChildNode& operator=(ChildNode&& o) noexcept {
			move = o.move;
			move_count = (int)o.move_count;
			win = (WinCountType)o.win;
			node = std::move(o.node);
			return *this;
		}

		Node* CreateChildNode() {
			node = std::make_unique<Node>();
			return node.get();
		}

		// --- public variables

		// 親局面(Node)で、このedgeに至るための指し手
		Move move;

		// このedgeの訪問回数。
		// Node::move_countと同じ意味。
		std::atomic<NodeCountType> move_count;

		// このedgeの勝った回数。Node::winと同じ意味。
		// ※　あるNodeの期待勝率 = win / move_count の計算式で算出する。
		std::atomic<WinCountType> win;

		// Policy Networkが返してきた、moveが選ばれる確率を正規化したもの。
		float nnrate;

		// 子ノードへのポインタ
		// これがnullptrであれば、この子ノードはまだexpand(展開)されていないという意味。
		//
		// TODO : ここに持つともったいないから、親nodeに最小限だけ持たせたほうが良いような…。
		std::unique_ptr<Node> node;
	};

	// 局面一つを表現する構造体
	// dlshogiのuct_node_t
	struct Node
	{
		Node() : move_count(0), win(0.0f), evaled(false) , value_win(0.0f) , visited_nnrate(0.0f) , child_num(0) {}

		// 子ノード1つのみで初期化する。
		void CreateSingleChildNode(const Move move)
		{
			child_num = 1;
			child = std::make_unique<ChildNode[]>(1);
			child[0].move = move;
		}

		// 候補手の展開
		// pos          : thisに対応する展開する局面
		// generate_all : 歩の不成なども生成する。
		void ExpandNode(const Position* pos, bool generate_all)
		{
			// 全合法手を生成する。

			if (generate_all)
				// 歩の不成などを含めて生成する。
				expand_node<LEGAL_ALL>(pos);
			else
				// 歩の不成は生成しない。
				expand_node<LEGAL>(pos);
		}

		// 引数のmoveで指定した子ノード以外の子ノードをすべて開放する。
		// 前回探索した局面からmoveの指し手を選んだ局面の以外の情報を開放するのに用いる。
		// moveを指した子ノードが見つかった場合はそのNode*を返す。
		// 見つからなかった場合は、新しくNodeを作成してそのNode*を返す。
		//
		// ある局面から2手先の局面が送られてきた時に、2手前から現局面に遷移する以外の指し手を
		// 削除したいので、そのためにこの関数がある。
		//
		// ※　その時のガーベジコレクションは別スレッドで行われる。
		// 子ノードが一つも見つからない時は、新しいノードを作成する。
		Node* ReleaseChildrenExceptOne(NodeGarbageCollector* gc, Move move);

		// --- public members..

		// このノードの訪問回数
		std::atomic<NodeCountType> move_count;

		// このノードを訪れて勝った回数
		// 実際にはplayoutまで行わずにValue Networkの返し値から期待勝率を求めるので
		// 端数が発生するから浮動小数点数になっている。
		// UctSearcher::UctSearch()で子ノードを辿った時に、その子ノードの期待勝率がここに加算される。
		// これは累積されるので、このノードの期待勝率は、 win / move_count で求める。
		std::atomic<WinCountType> win;

		// このノードがexpand(展開)されたあと、評価関数を呼び出しをするが、それが完了しているかのフラグ。
		// ノードはexpandされた時点で評価関数は呼び出されるので、これがfalseであれば、まだ評価関数の評価中であることを意味する。
		// これがtrueになっていれば、評価関数からの値が反映されている。
		// 評価関数にはPolicy Networkも含むので、このフラグがtrueになっていれば、各ChildNode(child*)のnnrateに値が反映されていることも保証される。
		std::atomic<bool> evaled;

		// Eval()したときに、NNから返ってきたこの局面のvalue(期待勝率)の値。
		// ただし詰み探索などで子ノードから伝播した場合、以下の定数をとることがある。
		// ・このノードで勝ちなら      VALUE_WIN   // 子ノードで一つでもVALUE_LOSEがあればその指し手を選択するので       VALUE_WIN
		// ・このノードで負けなら      VALUE_LOSE  // 子ノードがすべてVALUE_WINであればどうやってもこの局面では負けなのでVALUE_LOSE
		// ・このノードで引き分けなら、VALUE_DRAW
		// 備考) RepetitionWin (連続王手の千日手による反則勝ち) , RepetitionSuperior(優等局面)の場合も、VALUE_WINに含まれる。
		//       RepetitionLose(連続王手の千日手による反則負け) , RepetitionSuperior(劣等局面)の場合も、VALUE_LOSEに含まれる。
		// この変数は、UctSearcher::SelectMaxUcbChild()を呼び出した時に、子ノードを調べて、その結果が代入される。
		// TODO : この変数、この構造体に持たせる必要ないのでは…。
		std::atomic<WinCountType> value_win;

		// 訪問した子ノードのnnrateを累積(加算)したもの。
		// 訪問ごとに加算している。// 目的はよくわからん…。
		std::atomic<WinCountType> visited_nnrate;

		// 子ノードの数
		// 将棋では合法手は593手とされており、メモリがもったいないので、16bit整数で持つ。
		u16 child_num;

		// 子ノード(に至るedge)
		// child_numの数だけ、ChildNodeをnewして保持している。
		std::unique_ptr<ChildNode[]> child;

	private:
		// このnodeのchildを展開する時にこのlockが必要なので、Lock()～Unlock()を用いること。
		std::mutex mutex;

		// ExpandNode()の下請け。生成する指し手の種類を指定できる。
		template <MOVE_GEN_TYPE T>
		void expand_node(const Position* pos)
		{
			MoveList<T> ml(*pos);

			// 子ノードの数 = 生成された指し手の数
			child_num = (u16)ml.size();

			child = std::make_unique<ChildNode[]>(child_num);
			auto* child_node = child.get();
			for (auto m : ml)
				(child_node++)->move = m.move;
		}
	};

	// 前回探索した局面から2手進んだ局面かを判定するための情報を保持しておくためのNodeTree。
	// 1つのゲームに対して1つのインスタンス。
	class NodeTree
	{
	public:

		NodeTree(NodeGarbageCollector* gc) : gc(gc){}

		// デストラクタではゲーム木を開放する。
		~NodeTree() { DeallocateTree(); }

		// 局面(Position)を渡して、node tree内からこの局面を探す。
		// もし見つかれば、node treeの再利用を試みる。
		// 新しい位置が古い位置と同じゲームであるかどうかを返す。
		//   sfen  : 今回探索するrootとなる局面のsfen文字列
		//   moves : 初期局面からposに至る指し手
		// ※　位置が完全に異なる場合、または以前よりも短い指し手がmovesとして与えられている場合は、falseを返す
		bool ResetToPosition(const std::string& rootSfen, const std::vector<Move>& moves);

		// 現在の探索開始局面の取得
		Node* GetCurrentHead() const { return current_head; }

	private:
		// game_root_nodeをrootとするゲーム木を開放する。
		void DeallocateTree();

		// 探索開始局面(現在のroot局面)
		Node* current_head = nullptr;

		// ゲーム木のroot node = ゲームの開始局面
		// dlshogiでは、gamebegin_node_という変数名
		std::unique_ptr<Node> game_root_node;

		// ゲーム木のroot nodeのsfen文字列
		std::string game_root_sfen;

		// dlshogiではGCはglobalになっているが、NodeTreeからも使えるようにしておく。
		NodeGarbageCollector* gc;
	};

	// 定期的に走るガーベジコレクタ。
	// 別スレッドで開放していく。
	// ※　dlshogiでは"Node.cpp"に書いてあったが、こちらに移動させた。
	// ※　探索スレッドからスレッドを割り当てるとシンプルなコードになるのだが、
	//    GCに時間がかかることがあり、bestmoveを返すときに全スレッドの終了を待機するので
	//    それは良くないアイデアであった。
	class NodeGarbageCollector
	{
	public:
		// コンストラクタでGCスレッドを開始する。
		NodeGarbageCollector() : current_thread_id(-1), gc_thread(std::thread([this]() { Worker(); })) {}

		// この間隔ごとにGCを走らせる。
		const int kGCIntervalMs = 100;

		// GC対象に追加する。ここから辿れるNode,ChildNodeはすべて開放する。
		// また、Nodeは循環していないものとする。
		void AddToGcQueue(std::unique_ptr<Node> node) {
			if (!node) return;

			std::lock_guard<std::mutex> lock(gc_mutex);
			subtrees_to_gc.emplace_back(std::move(node));
		}

		~NodeGarbageCollector() {
			// stopフラグを変更して、GCスレッドが停止するのを待つ
			stop.store(true);
			gc_thread.join();
		}

		// --- やねうら王独自拡張

		// GC用のスレッドのスレッドIDを設定する。
		// これは、WinProcGroup::bindThisThread()を呼び出す時のID。
		void set_thread_id(size_t thread_id) { next_thread_id = (int)thread_id; }

	private:
		// ガーベジ用のスレッドがkGCIntervalMs[ms]ごとに実行するガーベジ本体。
		void GarbageCollect()
		{
			while (!stop.load()) {

				// Node will be released in destructor when mutex is not locked.
				std::unique_ptr<Node> node_to_gc;
				{
					// Lock the mutex and move last subtree from subtrees_to_gc_ into
					// node_to_gc.
					std::lock_guard<std::mutex> lock(gc_mutex);
					if (subtrees_to_gc.empty()) return;
					node_to_gc = std::move(subtrees_to_gc.back());
					// node_to_gcにmoveしておけば、このスコープを抜けるときにまずgc_mutexがunlockされて、
					// そのさらに一つ外のスコープでnode_to_gcのデストラクタが呼び出されて、
					// unique_ptrなどで保持しているノードが数珠つなぎに開放される。LC0の手法。

					subtrees_to_gc.pop_back();
				}
			}
		}

		// ガーベジ用のスレッドが実行するworker
		void Worker() {
			// stop == trueになるまで、kGCIntervalMs[ms]ごとにGarbageCollect()を実行。
			while (!stop.load()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
				GarbageCollect();

				// --- やねうら王独自拡張

				// bindThisThreadして欲しいなら、それを行う。
				if (current_thread_id != next_thread_id)
				{
					current_thread_id = next_thread_id.load();
					WinProcGroup::bindThisThread(current_thread_id);
				}
			};
		}

		// subtrees_to_gc を変更する時のmutex
		mutable std::mutex gc_mutex;

		// GC対象のTree。ここから数珠つなぎに開放していく。
		// 一度にそんなにたくさん積まれないので、そこまで大きなコンテナにはならない。
	    std::vector<std::unique_ptr<Node>> subtrees_to_gc;

		std::atomic<bool> stop{ false };
		std::thread gc_thread;

		// --- やねうら王独自拡張

		// 現在のスレッドIDを保持しておいて、設定されたものが異なるなら、新しいIDで
		// WinProcGroup::bindThisThread()を行う。

		std::atomic<int> current_thread_id;
		std::atomic<int> next_thread_id;
	};

}

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __NODE_H_INCLUDED__
