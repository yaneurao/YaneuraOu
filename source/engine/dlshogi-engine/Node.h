#ifndef __NODE_H_INCLUDED__
#define __NODE_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include <thread>
#include "../../position.h"
#include "../../movegen.h"
#include "dlshogi_types.h"

namespace dlshogi {

using namespace YaneuraOu;

struct Node;
class NodeGarbageCollector;

// 子ノード(に至るEdge(辺))を表現する。
// あるノードから実際に子ノードにアクセスするとランダムアクセスになってしまうので
// それが許容できないから、ある程度の情報をedgeがcacheするという考え。
// Nodeが親ノードを表現していて、基本的には合法手の数だけ、このChildNodeを持つ。
// ※　dlshogiのchild_node_t
struct ChildNode {
    ChildNode() :
        move_count(0),
        win((WinType) 0),
        nnrate(0.0f) {}

    ChildNode(Move move) :
        move(move),
        move_count(0),
        win((WinType) 0),
        nnrate(0.0f) {}

    // ムーブコンストラクタ
    ChildNode(ChildNode&& o) noexcept :
        move(o.move),
        move_count(0),
        win((WinType) o.win),
        nnrate(o.nnrate) {}

    // ムーブ代入演算子
    ChildNode& operator=(ChildNode&& o) noexcept {
        move       = o.move;
        move_count = (NodeCountType) o.move_count;
        win        = (WinType) o.win;
        nnrate     = (float) o.nnrate;
        return *this;
    }

    // --- public variables

    // 指し手(Move)が、SetWin()されているかの判定
    // ChildNodes::IsMoveWin(move)のように用いる。
    // これはその子ノードへの指し手moveによって、相手が勝ちになるかという判定なので、
    // これがtrueであれば現局面は負けの局面ということである。
    static bool IsMoveWin(Move m) { return m.to_u32() & VALUE_WIN; }
    static bool IsMoveLose(Move m) { return m.to_u32() & VALUE_LOSE; }
    static bool IsMoveDraw(Move m) { return m.to_u32() & VALUE_DRAW; }

    // メモリ節約のため、moveの最上位バイトでWin/Lose/Drawの状態を表す
    // (32bit型のMoveでは、ここは使っていないはずなので)
    bool IsWin() const { return IsMoveWin(move); }
    bool IsLose() const { return IsMoveLose(move); }
    bool IsDraw() const { return IsMoveDraw(move); }
    void SetWin() { move = Move(move.to_u32() | VALUE_WIN); }
    void SetLose() { move = Move(move.to_u32() | VALUE_LOSE); }
    void SetDraw() { move = Move(move.to_u32() | VALUE_DRAW); }
    // →　SetDraw()したときに、win = DRAW_VALUEにしたほうが良くないかな…。

    // 親局面(Node)で、このedgeに至るための指し手
    /*
		上位8bitをWin/Loseのフラグに使っているので、値比較するときには注意すること。
		ただし、上位bitが汚れていても、Position::do_move()するのは合法。

		⚠ dlshogiの以下の行は、フラグをクリアするコードに変更するとフラグが立っている
			ノードの再利用がなされて(そこがExpandNodeで
			展開されていないのにrootになりucb scoreが計算できないからまずい)

			> if (uct_child.move == move) {

			> if (root_node->child[i].move == move && root_node->child_nodes[i] && 
	*/
    Move move;
	// moveの上位bitにあるフラグをクリアして返す。
    Move getMove() const { return Move(move.to_u32() & 0xffffff); }

    // Policy Networkが返してきた、moveが選ばれる確率を正規化したもの。
    float nnrate;

    // このedgeの訪問回数。
    // Node::move_countと同じ意味。
    std::atomic<NodeCountType> move_count;

    // このedgeの勝った回数。Node::winと同じ意味。
    // ※　このChildNodeの着手moveによる期待勝率 = win / move_count の計算式で算出する。
    std::atomic<WinType> win;
};

// 局面一つを表現する構造体
// dlshogiのuct_node_t
struct Node {
    Node() :
        move_count(NOT_EXPANDED),
        win(0),
        visited_nnrate(0.0f),
        child_num(0),
        dfpn_checked(0),
        dfpn_proven_unsolvable(0) /*, dfpn_mate_ply(0)*/ {}

    // 子ノード作成
    Node* CreateChildNode(int i) { return (child_nodes[i] = std::make_unique<Node>()).get(); }

    // 子ノード1つのみで初期化する。
    void CreateSingleChildNode(const Move move) {
        child_num = 1;
        child     = std::make_unique<ChildNode[]>(1);
        child[0]  = move;
    }

    // 候補手の展開
    // pos          : thisに対応する展開する局面
    // generate_all : 歩の不成なども生成する。
    void ExpandNode(const Position* pos, bool generate_all) {
        // 全合法手を生成する。

        if (generate_all)
            // 歩の不成などを含めて生成する。
            expand_node<LEGAL_ALL>(pos);
        else
            // 歩の不成は生成しない。
            expand_node<LEGAL>(pos);
    }

    // 子ノードへのポインタ配列の初期化
    void InitChildNodes() { child_nodes = std::make_unique<std::unique_ptr<Node>[]>(child_num); }

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
    Node* ReleaseChildrenExceptOne(NodeGarbageCollector* gc, const Move move);

    // このノードがexpand(展開)されたあと、評価関数を呼び出しをするが、それが完了しているかのフラグ。
    // ※　実際は、フラグ用の変数がもったいないので、move_countを使いまわしている。
    // ノードはexpandされた時点で評価関数は呼び出されるので、これがfalseであれば、まだ評価関数の評価中であることを意味する。
    // これがtrueになっていれば、評価関数からの値が反映されている。
    // 評価関数にはPolicy Networkも含むので、このフラグがtrueになっていれば、各ChildNode(child*)のnnrateに値が反映されていることも保証される。
    bool IsEvaled() const { return move_count != NOT_EXPANDED; }

    // Evalした時にEvaledフラグをtrueにする関数
    // ※　実際は、フラグ用の変数がもったいないので、move_countを使いまわしている。
    void SetEvaled() { move_count = 0; }

    // --- public members..

    // このノードの訪問回数
    std::atomic<NodeCountType> move_count;

    // このノードを訪れて勝った回数
    // 実際にはplayoutまで行わずにValue Networkの返し値から期待勝率を求めるので
    // 端数が発生するから浮動小数点数になっている。
    // UctSearcher::UctSearch()で子ノードを辿った時に、その子ノードの期待勝率がここに加算される。
    // これは累積されるので、このノードの期待勝率は、 win / move_count で求める。
    std::atomic<WinType> win;

    // 訪問した子ノードのnnrateを累積(加算)したもの。
    // 訪問ごとに加算している。
    // fpu reductionで用いる。
    // ※　visited_nnrateはfpu_reductionが1を超えると意味のない値なのでfloatでも精度的に問題ないらしい。
    std::atomic<float> visited_nnrate;

    // 子ノードの数
    ChildNumType child_num;

    // 子ノード(に至るedge)
    // child_numの数だけ、ChildNodeをnewして保持している。
    std::unique_ptr<ChildNode[]> child;

    // 子ノードへのポインタ配列
    // もったいないので必要になってからnewする。
    // 展開した子ノード以外はnullptrのまま。
    std::unique_ptr<std::unique_ptr<Node>[]> child_nodes;

#if defined(USE_POLICY_BOOK)
    // PolicyBookから与えられたvalue
    // なければ FLT_MAX
    float policy_book_value = FLT_MAX;
#endif

    // 詰み関連のフラグ
    bool dfpn_checked;            // df-pn調べ済み
    bool dfpn_proven_unsolvable;  // df-pnで詰まないことが証明されている。
    //s16 dfpn_mate_ply : 15; // 詰み手数。0なら詰みなし。プラスならこの局面からの詰みまでの手数。マイナスなら、詰まされるまでの手数。

    // dfpn_checkedのフラグの状態を変更する時のmutex
    static std::mutex mtx_dfpn;

   private:
    // ExpandNode()の下請け。生成する指し手の種類を指定できる。
    template<GenType T>
    void expand_node(const Position* pos) {
        MoveList<T> ml(*pos);

        child            = std::make_unique<ChildNode[]>(ml.size());
        auto* child_node = child.get();
        for (auto m : ml)
            (child_node++)->move = m;

        // 子ノードの数 = 生成された指し手の数
        child_num = (ChildNumType) ml.size();
    }
};

// 前回探索した局面から2手進んだ局面かを判定するための情報を保持しておくためのNodeTree。
// 1つのゲームに対して1つのインスタンス。
class NodeTree {
   public:
    NodeTree(NodeGarbageCollector* gc) :
        gc(gc) {}

    // デストラクタではゲーム木を開放する。
    ~NodeTree() { DeallocateTree(); }

    // ゲーム開始局面からの手順を渡して、node tree内からこの局面を探す。
    // もし見つかれば、node treeの再利用を試みる。
    // 新しい位置が古い位置と同じゲームであるかどうかを返す。
    //   game_root_sfen : ゲーム開始局面のsfen文字列
    //   moves          : game_root_sfenの局面からposに至る指し手
    // ※　位置が完全に異なる場合、または以前よりも短い指し手がmovesとして与えられている場合は、falseを返す
    bool ResetToPosition(const std::string& game_root_sfen, const std::vector<Move>& moves);

    // 現在の探索開始局面の取得
    Node* GetCurrentHead() const { return current_head; }

   private:
    // game_root_nodeをrootとするゲーム木を開放する。
    void DeallocateTree();

    // 探索開始局面(現在のroot局面)
    Node* current_head = nullptr;

    // ゲーム木のroot node = ゲームの開始局面
    // ※　dlshogiでは、gamebegin_node_という変数名
    std::unique_ptr<Node> game_root_node;

    // ゲーム開始局面
    // ※　dlshogiではhistory_starting_pos_key_というKey型の変数
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
class NodeGarbageCollector {
   public:
    // コンストラクタでGCスレッドを開始する。
    NodeGarbageCollector() :
        current_thread_id(-1),
        gc_thread(std::thread([this]() { Worker(); })) {}

    // この間隔ごとにGCを走らせる。
    const int kGCIntervalMs = 100;

    // GC対象に追加する。ここから辿れるNode,ChildNodeはすべて開放する。
    // また、Nodeは循環していないものとする。
    // また、node == nullptrなら何もせずにreturnする。
    void AddToGcQueue(std::unique_ptr<Node> node) {
        if (!node)
            return;

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
    // worker thread自体は、やねうら王フレームワーク側(ThreadPoolクラス)で作成してもらうのではなく
    // コンストラクタで起動させ、デストラクタで終了する感じ。
    void set_thread_id(size_t thread_id) { next_thread_id = (int) thread_id; }

   private:
    // ガーベジ用のスレッドがkGCIntervalMs[ms]ごとに実行するガーベジ本体。
    void GarbageCollect() {
        while (!stop.load())
        {

            // Node will be released in destructor when mutex is not locked.
            std::unique_ptr<Node> node_to_gc;
            {
                // Lock the mutex and move last subtree from subtrees_to_gc_ into
                // node_to_gc.
                std::lock_guard<std::mutex> lock(gc_mutex);
                if (subtrees_to_gc.empty())
                    return;
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
        while (!stop.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
            GarbageCollect();

            // --- やねうら王独自拡張

            // bindThisThreadして欲しいなら、それを行う。
            if (current_thread_id != next_thread_id)
            {
                current_thread_id = next_thread_id.load();

                //WinProcGroup::bindThisThread(current_thread_id);
                // TODO : あとで binderどうにかする。
            }
        };
    }

    // subtrees_to_gc を変更する時のmutex
    mutable std::mutex gc_mutex;

    // GC対象のTree。ここから数珠つなぎに開放していく。
    // 一度にそんなにたくさん積まれないので、そこまで大きなコンテナにはならない。
    std::vector<std::unique_ptr<Node>> subtrees_to_gc;

    // gc_threadの停止フラグ。trueになったら、gc_threadはWorker()から抜けて終了する。
    std::atomic<bool> stop{false};
    // GC用のthread
    std::thread gc_thread;

    // --- やねうら王独自拡張

    // 現在のスレッドIDを保持しておいて、設定されたものが異なるなら、新しいIDで
    // WinProcGroup::bindThisThread()を行う。

    std::atomic<int> current_thread_id;
    std::atomic<int> next_thread_id;
};

}  // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __NODE_H_INCLUDED__
