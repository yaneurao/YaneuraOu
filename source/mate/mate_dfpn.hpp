#include "mate.h"
#if defined(USE_MATE_DFPN)

/*
	並列化df-pn詰将棋ルーチン

	古来からあるhash tableを用いた実装や、df-pn論文にあるような実装をやめて、
	df(disproof number)とpn(proof number)を方策として持つ最良優先探索として実装する。
	※　pn-searchと呼ばれるアルゴリズムに近い。

	また、合流を処理しないことにする。

	これらの意味で、この詰将棋ルーチンをdf-pnと呼べるかは疑問だが、従来のものより以下のメリットがある。
	・MCTSの実装風に書けてわかりやすい
	・探索を並列化しやすい
	・合流を扱わないので、メモリが無限にあれば、原理上、どんな詰将棋も必ず詰むか詰まないかを判定できる
	・hash tableを用いないので、詰み・不詰を間違うことがない

	デメリットとして
	・メモリを大量に消費する
	ことが挙げられるが、大会用のPCには1TBぐらいメモリが載っているので問題とならない。

	また、一般的に、df-pnはnpsがあまり出ないことで知られているが、生成した指し手を保存しておくことによって、
	二度目の指し手生成を省略するので従来のdf-pnと比較して遅くはないはず。

	メモリ使用量についても可能な限り省メモリで済むようなデータ構造にしてある。

	なお、並列化については、Node構造体に std::mutexをもたせて、子ノードの展開の時にlockしないといけないことになるが
	sizeof(std::mutex)==80もあるので、これ持ちたくない。

	// →　いったん並列化を諦めよう。まずはシングルスレッド用で高速なやつを作るべき。

	そもそも、Node構造体は無限に増えていくので、そんなところに同期化のためのデータを保持しているのが設計上の誤り。
	mutex事前に65536個ほどどこかに確保しておいて、Positionのhash keyの下位16bitを使って、そのmutex選んで使うなどすれば良い。
*/

//#define DFPN64

#if defined(DFPN64) || defined(DFPN32)

#include <mutex>
#include "../position.h"
#include "../thread.h"
#include "mate_move_picker.h"

#if defined (DFPN64)
// Node数が64bitで表現できる数まで扱える版
namespace Mate::Dfpn64
{
	// ===================================
	//   ノードを表現する構造体
	// ===================================

	// あるノード(局面)を表現する最小の構造体。
	// NodeCountTypeは、u32を想定していたが、オーダリングのために大きな数を扱いたいのでu64に変更することにした。

	template <typename NodeCountType , bool MoveOrdering>
	struct alignas(32) Node
	{
		// dn,pnの∞を表現する定数
		// これがdn == DNPN_MATE以上が詰んでいるスコア、1手詰めのスコアが、dn == DNPN_INF - 1 , N手詰めは dn == DNPN-INF - N  …というように定義する。
		// 2000手以上の詰将棋はないので、2000も引いておけば十分かと。
		static constexpr NodeCountType DNPN_INF  = std::numeric_limits<NodeCountType>::max();
		static constexpr NodeCountType DNPN_MATE = std::numeric_limits<NodeCountType>::max() - 2000;

		// 子のノード数が未初期化であることを意味する定数。
		static constexpr u8 CHILDNUM_NOT_INIT = std::numeric_limits<u8>::max();

		//   子ノード配列の先頭のNode*を表現している。
		Node* children;

		//   ここにdn,pnも持たせるとメモリ少し余分に消費するがランダムアクセスが減るので高速化する。
		//   dn : disproof number : 詰みにくさを表現する(大きいほど不詰を証明するのにたくさんのノードを調べなくてはならない)
		//   pn : proof number    : 詰みやすさを表現する(大きいほど詰みを証明するのにたくさんのノードを調べなくてはならない)
		// 
		//   df-pnでANDノードのdn,pnの役割を入れ替えて考えれば、ORノードとANDノードとで場合分けが無くせるのだが、
		//   PVを取得する時や、1手詰めを考える時などにそこで場合分けが必要となるから、この実装ではやらない。
		NodeCountType pn, dn;

		// このノードに至るための直前の指し手
		Move lastMove;

		// Parent nodeの時は、そのノードの子の数が格納されている。
		// 詰将棋においてたかだか100いくつしかないので1byteで十分。
		// 16bitでも十分なのだが…。

		// 子ノードの数。
		// 子ノードが未展開のときはこの変数の値はCHILDNUM_NOT_INITになる。
		// 子ノードを展開しようとしたが、子ノードが存在しなかった場合は、0になる。
		u8 child_num;

		// 千日手など循環が、このノードを含めたsub-nodeで生じたか。
		// (この場合、pn,dnは経路依存と言うことになる。)
		bool repeated;

		u8 padding[2]; // これを含めてこの構造体が32 bytesになるようにしておく。

		// このnodeを初期化する。
		template <bool or_node>
		void init(ExtMove m)
		{
			lastMove = m.move;

			if (MoveOrdering)
			{
				// m.valueは攻め方(or node)から見た駒割の値(-30000～+30000)なので+30000して、1～60000の範囲の値にする。
				pn = -m.value + 30000; // 攻め方から見た駒割の符号反転 + 30000
				dn = +m.value + 30000; // 攻め方から見た駒割           + 30000
			}
			else {
				pn = dn = 1;
			}

			children = nullptr;
			child_num = CHILDNUM_NOT_INIT;
		}

		// このnodeを初期化する
		// この指し手で詰む時の、このnodeの初期化
		template <bool or_node>
		void init_mate_move(Move move)
		{
			this->lastMove = move;
			this->template set_mate<true>();

			children = nullptr;
			child_num = CHILDNUM_NOT_INIT;
		}

		// 詰みの時のpnとdnの値を設定する。
		// ply : 現局面から何手後に詰むのか。
		template <bool or_node>
		void set_mate(int ply = 0)
		{
			// or nodeであれば、pn = 0 , dn = Mate だが、
			// and nodeでは、これが逆になる。
			this->pn = or_node ? 0                             : NodeCountType(DNPN_INF - ply);
			this->dn = or_node ? NodeCountType(DNPN_INF - ply): 0;
		}

		// 詰まされた時のpnとdnの値を設定する。
		template <bool or_node>
		void set_mated(int ply = 0)
		{
			this->pn = or_node ? NodeCountType(DNPN_INF - ply): 0;
			this->dn = or_node ? 0                             :NodeCountType(DNPN_INF - ply);
		}

		// 詰まない時のpnとdnの値を設定する。
		void set_nomate(int ply = 0)
		{
			this->pn = NodeCountType(DNPN_INF - ply);
			this->dn = 0; // 不詰が証明されている。
		}

		// この構造体のchild_num(子ノードの数)とnode(これは子ノードへのポインタ相当),を設定してやる。
		void set_child(u8 child_num , Node* children = nullptr)
		{
			this->child_num = child_num;
			this->children = children;
		}
	};

#endif // defined (DFPN64)

#if defined (DFPN32)

// Node数が32bitで表現できる数までしか扱えない版
namespace Mate::Dfpn32
{
	// ===================================
	//   ノードを表現する構造体
	// ===================================

	// あるノード(局面)を表現する最小の構造体。
	// NodeCountTypeは、u32を想定していたが、オーダリングのために大きな数を扱いたいのでu64に変更することにした。
	// →　高速化のために低nodes_limitの探索用を作った。
	// 探索できるノード数に制限がある。

	template <typename NodeCountType , bool MoveOrdering>
	struct alignas(16) Node
	{
		// NodeCountTypeで初期化されていない値を表現する定数
		static constexpr NodeCountType NodeCountTypeNull = std::numeric_limits<NodeCountType>::max();

		// dn,pnの∞を表現する定数
		// これがdn == DNPN_MATE以上が詰んでいるスコア、1手詰めのスコアが、dn == DNPN_INF - 1 , N手詰めは dn == DNPN-INF - N  …というように定義する。
		// 2000手以上の詰将棋はないので、2000も引いておけば十分かと。
		static constexpr NodeCountType DNPN_INF  = std::numeric_limits<NodeCountType>::max();
		static constexpr NodeCountType DNPN_MATE = std::numeric_limits<NodeCountType>::max() - 2000;

		// 子のノード数が未初期化であることを意味する定数。
		static constexpr u8 CHILDNUM_NOT_INIT = std::numeric_limits<u8>::max();

		// this->childrenなど、あるNodeCountTypeの値がnullptrを意味するものか判定する。
		static bool is_nullptr(NodeCountType n) { return n == NodeCountTypeNull; }

		//   子ノード配列の先頭のNode*を表現している。
		// NodeManagerに問い合わせると、これに対するNode*がもらえる。
		NodeCountType children;

		//   ここにdn,pnも持たせるとメモリ少し余分に消費するがランダムアクセスが減るので高速化する。
		//   dn : disproof number : 詰みにくさを表現する(大きいほど不詰を証明するのにたくさんのノードを調べなくてはならない)
		//   pn : proof number    : 詰みやすさを表現する(大きいほど詰みを証明するのにたくさんのノードを調べなくてはならない)
		// 
		//   df-pnでANDノードのdn,pnの役割を入れ替えて考えれば、ORノードとANDノードとで場合分けが無くせるのだが、
		//   PVを取得する時や、1手詰めを考える時などにそこで場合分けが必要となるから、この実装ではやらない。
		NodeCountType pn, dn;

		// このノードに至るための直前の指し手
		// Moveは21bitであることが保証されているので、23bitだけ使う。残り1byteは、child_numの格納のために用いる。
		u32 lastMove  : 23;

		// 千日手など循環が、このノードを含めたsub-nodeで生じたか。
		// (この場合、pn,dnは経路依存と言うことになる。)
		u32 repeated  : 1;

		// このノードの子の数が格納されている。
		// 詰将棋においてたかだか100いくつしかないので1byteで十分。
		// 子ノードの数。
		// 子ノードが未展開のときはこの変数の値はCHILDNUM_NOT_INITになる。
		// 子ノードを展開しようとしたが、子ノードが存在しなかった場合は、0になる。
		u32 child_num :  8;

		// 以上、この構造体は16 bytes。
		//static_assert(sizeof(Node<u32,false>) == 16);

		// このnodeを初期化する。
		template <bool or_node>
		void init(ExtMove m)
		{
			lastMove = m.move;

			if (MoveOrdering)
			{
				// m.valueは攻め方(or node)から見た駒割の値(-30000～+30000)なので+30000して、1～60000の範囲の値にする。
				pn = -m.value + 30000; // 攻め方から見た駒割の符号反転 + 30000
				dn = +m.value + 30000; // 攻め方から見た駒割           + 30000
			}
			else {
				pn = dn = 1;
			}

			// これはnullptrを意味する。
			children = NodeCountTypeNull;
			child_num = CHILDNUM_NOT_INIT;
		}

		// このnodeを初期化する
		// この指し手で詰む時の、このnodeの初期化
		template <bool or_node>
		void init_mate_move(Move move ,int ply = 0)
		{
			this->lastMove = move;
			this->template set_mate<true>(ply);

			// これはnullptrを意味する。
			children = NodeCountTypeNull;
			child_num = CHILDNUM_NOT_INIT;
		}

		// 詰みの時のpnとdnの値を設定する。
		// ply : 現局面から何手後に詰むのか。
		template <bool or_node>
		void set_mate(int ply = 0)
		{
			// or nodeであれば、pn = 0 , dn = Mate だが、
			// and nodeでは、これが逆になる。
			this->pn = or_node ? 0                             : NodeCountType(DNPN_INF - ply);
			this->dn = or_node ? NodeCountType(DNPN_INF - ply): 0;
		}

		// 詰まされた時のpnとdnの値を設定する。
		template <bool or_node>
		void set_mated(int ply = 0)
		{
			this->pn = or_node ? NodeCountType(DNPN_INF - ply): 0;
			this->dn = or_node ? 0                             :NodeCountType(DNPN_INF - ply);
		}

		// 詰まない時のpnとdnの値を設定する。
		void set_nomate(int ply = 0)
		{
			this->pn = NodeCountType(DNPN_INF - ply);
			this->dn = 0; // 不詰が証明されている。
		}

		// この構造体のchild_num(子ノードの数)とnode(これは子ノードへのポインタ相当),を設定してやる。
		void set_child(u8 child_num , NodeCountType children = NodeCountTypeNull)
		{
			this->child_num = child_num;
			this->children = children;
		}
	};

#endif // defined (DFPN32)

	// ===================================
	// 子ノード、ノードのメモリマネージャー
	// ===================================

	// Nodeのためのバッファを表現する。
	// これは先頭からリニアに使っていく。GCは考えない。
	template <typename NodeCountType , bool MoveOrdering>
	struct NodeManager
	{
		// このクラスで扱うノード型
		typedef Node<NodeCountType,MoveOrdering> NodeType;

		NodeManager() :node_index(0), nodes_num(0) {}

		// メモリ確保。size_個分Nodeが確保される。
		void alloc(size_t size_) {
			// 前のを開放してからでないと新しいのを確保するメモリがないかも知れない。
			release();

			this->nodes_num = (NodeCountType)size_;
			nodes = std::make_unique<NodeType[]>(size_);
		}

		// 確保していたメモリを開放する。
		void release() { nodes.reset(); }

		// 確保されているNodeのバッファの数を返す。alloc()で確保した個数。
		NodeCountType size() const { return nodes_num; }

		// Nodeをsize個分確保して、その先頭のアドレスを返す。
		// 確保できない時はnullptrが返る。
		NodeType* new_node(size_t size = 1)
		{
			//std::lock_guard<std::mutex> lk(mutex);
			// 並列化対応はまたの機会に…。

			if (is_out_of_memory())
				return nullptr;

			NodeType* node = &nodes[node_index];
			node_index += (NodeCountType)size;
			return node;
		}

		// 内部カウンターのリセット。
		// 次回のnew_node()でまた1番目の要素が返るようになる。
		// 新しい局面の探索の開始時に呼び出すと良い。
		void reset_counter() { node_index = 0; }

		// alloc()で確保されたバッファを使い切ったのか。
		bool is_out_of_memory() const {
			// 次にnew_node()で王手の組み合わせMaxCheckMoves分が確保できない時はメモリを使い切ったと判断してtrueを返す。
			return node_index + MaxCheckMoves >= nodes_num;
		}

		// hash使用率を1000分率で返す。
		int hashfull() const { return (int)((u64)node_index * 1000 / nodes_num); }


#if defined(DFPN32)
		// -----------------------------------------------------------------
		// ChildNodeではNode*を持ちたくないので(8バイト消費するのが嫌)、
		// Node*をnode番号に変換して保持している。以下は、その設定/取得を行う関数。
		// -----------------------------------------------------------------

		// あるnodeのnode->children に対応する Node* を取得する。
		NodeType* get_children(const NodeType* node) const
		{
			return node_index_to_node(node->children);
		}

		// あるnodeのnode->childrenに、Node*を代入する。
		void set_children(NodeType* node, const NodeType* next) const
		{
			node->children = node_to_node_index(next);
		}

		// 指定されたNode番号のNode*を返す。
		// n が NodeCountTypeNull であれば、nullptrを返す。
		NodeType* node_index_to_node(NodeCountType n) const {
			return ! NodeType::is_nullptr(n) ? &nodes[n] : nullptr;
		}

		// Node*からNode番号を返す。
		// 相互変換であるので、nullptrの時にはNodeCountTypeNullを返す。
		NodeCountType node_to_node_index(const NodeType*node) const
		{
			// メモリ上にリニアに確保されているので、これで何番目のNodeであるかが返る。
			return (node == nullptr) ? NodeType::NodeCountTypeNull : NodeCountType(node - nodes.get());
		}
#endif

#if defined(DFPN64)
		// DFPN32の時と同一のinterfaceにするために必要。

		NodeType* get_children(const NodeType* node) const { return node->children; }
		void set_children(NodeType* node, const NodeType* next) const { node->children = next; }
		NodeType* node_index_to_node(NodeType* n) const { return n; }
		NodeType* node_to_node_index(NodeType* node) const { return node; }
#endif

	private:

		// 確保されたnode用のメモリ本体
		// これを先頭から使っていく。
		std::unique_ptr<Node<NodeCountType,MoveOrdering>[]> nodes;

		// 確保しているNode用のbufferの数。
		NodeCountType nodes_num;

		// 次に返すべきnode用のカウンター
		std::atomic<NodeCountType> node_index;

		// ↑を返す時に必要となるlock
		//std::mutex mutex;
	};


	// ===================================
	//   class DfpnImp , df-pnの実装本体
	// ===================================

	// df-pn詰将棋ルーチン本体
	// 事前にメモリ確保やら何やらしないといけないのでクラス化してある。
	template <typename NodeCountType , bool MoveOrdering , bool WithHash >
	class MateDfpnPn : public Mate::Dfpn::MateDfpnSolverInterface
	{
	public:
		// このクラスで扱うノード型
		typedef Node<NodeCountType,MoveOrdering> NodeType;

		// このクラスを用いるには、この関数を呼び出して事前にDfpn用のメモリを確保する必要がある。
		// size_mb [MB] だけ探索用のメモリを確保する。
		virtual void alloc(size_t size_mb) {
			// 前に使用していたメモリを先に開放しないと次に確保できないかも知れない。
			release();

			// 使っていいメモリサイズ。
			size_t size = size_mb * 1024 * 1024;
			node_manager.alloc(size / sizeof(Node<NodeCountType,MoveOrdering>));
		}

		// 探索ノード数の上限を指定してメモリを確保する。
		// alloc()かalloc_by_nodes_limit()か、どちらかを呼び出してメモリを確保すること！
		virtual void alloc_by_nodes_limit(size_t nodes_limit)
		{
			// 前に使用していたメモリを先に開放しないと次に確保できないかも知れない。
			release();

			node_manager.alloc(nodes_limit);
		}

		// Hash Tableの設定。
		// 全スレッドで共用しているようなhash table
		// ※　詰み/不詰を証明済みの局面についてcacheしておくためのテーブル
		//    Node32bitWithHashのような"WithHash"とついているインスタンスに対して有効。
		virtual void set_hash_table(MateHashTable* hash_table)
		{
			this->hash_table = hash_table;
		}

		// alloc()で確保していたメモリを開放する。
		void release()
		{
			node_manager.release();
		}

		// 最大探索深さ。これを超えた局面は不詰扱いとする。
		// Position::game_ply()がこれを超えた時点で不詰扱い。
		// 0を指定すると制限なし。デフォルトは0。
		virtual void set_max_game_ply(int max_game_ply)
		{
			this->max_game_ply = max_game_ply;
		}

		// 詰み探索をしてnodes_limit内のノード数で解ければその初手が返る。
		// 不詰が証明できれば、MOVE_NULL、解がわからなかった場合は、MOVE_NONEが返る。
		// nodes_limit : ノード制限。0を指定するとノード制限なし。(ただしメモリの制限から解けないことはある)
		virtual Move mate_dfpn(const Position& pos_, u64 nodes_limit)
		{
			// const剥がし。ここからreturnする時には、元の局面になっているのでconst性は崩れていないという解釈
			auto& pos = *const_cast<Position*>(&pos_);

			nodes_searched = 0;
			this->nodes_limit = (NodeCountType)nodes_limit;
			bestmove = MOVE_NONE;

			// カウンターのリセットをしておかないと新しいメモリが使えない。
			node_manager.reset_counter();
			out_of_memory = false;

			// RootNodeを展開する。
			ExpandRoot(pos);

			// あとはrootから良さげなところを最良優先探索するのを繰り返すだけで解けるのでは…。
			ParallelSearch(pos);

			// 詰んだ
			if (current_root->pn == 0 && current_root->dn >= NodeType::DNPN_MATE)
			{
#if 0
				std::cout << "solved! ply = " << get_mate_ply() << ", nodes_searched = " << nodes_searched << std::endl;
				//dump_tree(current_root);
#endif

				// pick_the_bestは引数書き換えるのでコピーしとく。
				auto node = current_root;
				return pick_the_best<true, true, false>(node);
			}

			// 不詰が証明された
			if (current_root->pn >= NodeType::DNPN_MATE && current_root->dn == 0)
				return MOVE_NULL;

			// 制限ノード数では解けなかった。
			// もしくはout of memory
			return MOVE_NONE;
		}

		// mate_dfpn()がMOVE_NULL,MOVE_NONE以外を返した場合にその手順を取得する。
		// ※　最短手順である保証はない。
		virtual std::vector<Move> get_pv() const { return get_pv<true,false>(); }

		// 現在の探索中のPVを出力する。
		virtual std::vector<Move> get_current_pv() const { return get_pv<true,true>(); }
		
		// 不詰の証明をするPVの取得。
		// mate_dfpn()がMOVE_NULL(不詰)を返した時に、そのPVを取得する。
		// (不詰の手順。攻め方はなるべく長め(最長ではない)、受け方はなるべく短め(最短ではない)に逃れる)
		std::vector<Move> get_unproof_pv() const { return get_pv<false,false>(); }

		// 詰んだとして、その手数を取得する。
		int get_mate_ply() const {
			if (current_root->pn != 0 || current_root->dn < NodeType::DNPN_MATE)
				return -1; // 詰んでない。
			return int(NodeType::DNPN_INF - current_root->dn);
		}

		// 今回の探索ノード数を取得する。
		virtual u64 get_nodes_searched() const {
			return (u64)nodes_searched;
		}

		// mate_dfpn()でMOVE_NONE以外が返ってきた時にメモリが不足しているかを返す。
		virtual bool is_out_of_memory() const { return node_manager.is_out_of_memory(); }

		// hash使用率を1000分率で返す。
		virtual int hashfull() const { return node_manager.hashfull(); }

	protected:

		// 詰み手順、不詰の手順を得る。
		// proof   : trueなら詰みの手順、falseなら不詰の手順。
		// current : trueなら現在探索中のベスト
		template <bool proof , bool current>
		std::vector<Move> get_pv() const {
			std::vector<Move> pv;

			auto node = current_root;
			// pnが0の子を辿るだけ。
			bool or_node = true;
			while (node) {
				// これが詰み手順だとして、
				// ただしorノードではdnは最小であってほしい。(これが詰みまでの距離を表現しているので)
				// andノードではdnは最大であってほしい。
				Move move = or_node ? pick_the_best<true,proof,current>(node) : pick_the_best<false,proof,current>(node);
				if (move == MOVE_NONE)
					break;

				pv.push_back(move);
				or_node ^= true; // 次のnodeでは反転させる
			}
			return pv;
		}

		// 解図できたとき(mate_dfpn()を呼び出してMOVE_NONE以外が返ってきた時に)
		// あるノードでベストな指し手を得る。
		//  or_node : nodeは、開始局面とその2手先、4手先、…の局面であるか。
		//  proof   : 詰みの時のPVがほしい時。これをfalseにすると、不詰が証明されている時に、そのなかの長そうなpvが得られる。
		//  current : trueなら現在探索中のベスト
		// 返し値   : そのベストな指し手。nodeは、その指し手で進めた時の次のNodeを指すポインター。
		template <bool or_node, bool proof,bool current>
		Move pick_the_best(NodeType*& node) const
		{
			NodeType* children = node_manager.get_children(node);
			if ( children == nullptr || node->child_num == 0)
			{
				node = nullptr;
				return MOVE_NONE; // これ以上辿れない
			}

			// 子ノードの数
			int child_num = node->child_num;

			int selected_index = 0;
			if (proof)
			{
				for (int i = 1;  i < child_num ; ++i)
					if (or_node)
					{
						if (current)
						{
							// pn最小で、pnの値が同じならdn最大を選びたい。
							if (children[i].pn < children[selected_index].pn
								|| (children[i].pn == children[selected_index].pn && children[i].dn > children[selected_index].dn))
								selected_index = i;
						}
						else {
							// pn = 0 , dn >= DNPN_MATEは証明されているので、pn = 0のところを辿るだけだが、
							// 自分(or_node)は dnが大きめ(短い手順の詰み)を選びたいし、相手(!or_node)はdnが小さめ(長い手順の詰み)を選びたい。
							if (children[i].dn > children[selected_index].dn)
								selected_index = i;
						}
					}
					else {
						if (current)
						{
							// pn最小で、pnの値が同じならdn最小を(受け方は選ぶであろうから)選ぶ。
							if (children[i].pn < children[selected_index].pn
								|| (children[i].pn == children[selected_index].pn && children[i].dn < children[selected_index].dn))
								selected_index = i;
						}
						else {
							// pn >= DNPN_MATE , dn = 0は証明されているので、pn = 0のところを辿るだけだが、
							// 受け方(!or_node)はdnが小さめ(長い手順の詰み)を選びたい。
							if (children[i].dn < children[selected_index].dn)
								selected_index = i;
						}
					}
			}
			else {
				// 不詰の証明においては、攻め方(or_node)はなるべく最長手順を選びたい(pn小さい)が、
				// 受け方(!or_node)は、なるべく最短手順(pn大きい)で逃れたい。
				for (int i = 1;  i < child_num ; ++i)
					if (or_node)
					{
						if (children[i].pn > children[selected_index].pn)
							selected_index = i;
					}
					else {
						if (children[i].pn < children[selected_index].pn)
							selected_index = i;
					}
			}

			auto& selected = children[selected_index];
			node = &selected;

			return (Move)selected.lastMove;
		}

		// node*以下の詰みツリーを全部出力する。(デバッグ用)
		void dump_tree(NodeType* node , std::string pv = "")
		{
			NodeType* children = node_manager.get_nodeptr(node);
			if (children == nullptr)
			{
				std::cout << pv << std::endl;
				return;
			}
			int child_num = node->child_num;
			if (child_num == 0)
			{
				std::cout << pv << std::endl;
				return;
			}

			bool found = false;
			for (int i = 0; i < child_num; ++i)
			{
				if (node->children[i].pn == 0)
				{
					found = true;
					//auto pv2 = pv + to_usi_string(children[i].move) + " ";
					auto pv2 = pv + "[" + std::to_string(node->dn) + "] " + to_usi_string(children[i].move);
					NodeType* next = node->children[i].children;
					dump_tree(next, pv2);
				}
			}

			if (!found)
				std::cout << "mate not found : " << pv << std::endl;
		}

		// 並列化する時は、ここがthreadのエントリーポイントとなる。
		void ParallelSearch(Position& pos)
		{
			ParallelSearch<true>(pos, current_root , NodeType::DNPN_INF , NodeType::DNPN_INF);
		}

		// 並列探索部本体
		//   second_pn : 親nodeの2番目に小さなpn ←これをこのnodeのpnが上回ったら親nodeに戻りたい。
		//   second_dn : 親nodeの2番目に小さなdn ←これをこのnodeのdnが上回ったら親nodeに戻りたい。
		template <bool or_node>
		void ParallelSearch(Position& pos , NodeType* node , NodeCountType second_pn , NodeCountType second_dn)
		{
			ASSERT_LV3(node->pn <= second_pn && node->dn <= second_dn);

			// or nodeでpnがsecond_pnを上回ると、２つ上のnodeで、second_pnであった子ノードを選んだほうが良いことになる。
			// and nodeでdnがsecond_dnを上回ると、以下同様。
			 while (
				    node->pn <= second_pn
				 && node->dn <= second_dn
				 && node->pn
				 && node->dn
				 && !out_of_memory
				 //&& (!nodes_limit || nodes_searched < nodes_limit)
				 && (!nodes_limit ||
						 ( MoveOrdering && (nodes_limit - nodes_searched > (std::max(current_root->pn, node->pn) >>16) )) ||
						 (!MoveOrdering && (nodes_limit - nodes_searched > (std::max(current_root->pn, node->pn)))))
				 // pnはMoveOrdering有りだと 2**16 されていることに注意。
				 // 残り探索ノード数がpnを上回ると証明不可。不詰は証明できるかもしれないが、不詰の証明はあまり価値がないのでこの状況下ならできなくていいと思う。
				 // ↑この枝刈りは、やねうらお考案。leaf nodeから呼び出すときに3%ぐらいnps上がる。
				 && !stop // スレッド停止命令が来たら即座に終了する。
				)
			{
#if 0
				 // デバッグ用に局面を出力してやる。
				 std::cout << pos << std::endl;
#endif

				u8 child_num = node->child_num;
				if (child_num == NodeType::CHILDNUM_NOT_INIT)
				{
					ExpandNode<or_node>(pos, node);
					// 今回はこれを展開しただけで良しとする。

					continue;
				}

				NodeCountType second_pn2 = second_pn;
				NodeCountType second_dn2 = second_dn;
				auto best_child = select_the_best_child<or_node>(node, second_pn2, second_dn2);

				// 一手進めて子ノードに行く
				StateInfo si;
				Move m = pos.to_move(best_child->lastMove);
				pos.do_move(m, si);

				// 再帰的に呼び出す。
				ParallelSearch<!or_node>(pos, best_child ,second_pn2 , second_dn2);

				// 子ノードから返ってきたので、子ノードのdn,pnを集計する。
				SummarizeNode<or_node>(node);

				pos.undo_move(m);

				// 千日手が絡まずに詰み or 不詰を証明したので、置換表に保存する。
				if (WithHash && (node->dn == 0 || node->pn == 0) && !node->repeated)
				{
					auto key = pos.state()->board_key();
					auto entry = hash_table->first_entry(key);

					// たぶん指し手mでこれを証明したはずなので、これを登録しておく。
					bool is_mate = node->pn == 0; /* pn == 0はわかったから、その手数(== DNPN_INF - dn)を保存したい */
					auto ply = is_mate ? (u32)(NodeType::DNPN_INF - node->dn) : (u32)(NodeType::DNPN_INF - node->pn);
					entry->save(key, root_color , pos.state()->hand , is_mate , ply, m);

#if 0
					// デバッグ用に証明済みの局面を出力してみる。
					std::cout << pos << std::endl
							  << " is_mate = " << is_mate << std::endl
							  << " ply     = " << ply << std::endl
							  << " move    = " << m << std::endl;
#endif
				}

			 }
		}

		// あるnodeの子ノードのなかから、一番良さげなNodeを選択する。
		// OR ノードであれば、一番pnが小さい子を選ぶ。(詰みを証明しやすそうなので)
		// ANDノードであれば、一番dnが小さい子を選ぶ。(不詰を証明しやすそうなので)
		// second_pn , second_dn : 2番目によさげな子ノードのpn,dnの値
		template <bool or_node>
		NodeType* select_the_best_child(NodeType* node,NodeCountType& second_pn,NodeCountType& second_dn)
		{
			auto children  = node_manager.get_children(node);
			u32 child_num = node->child_num;

			u32 selected_index = 0;
			NodeCountType pn2 = second_pn;
			NodeCountType dn2 = second_dn;

			if (or_node)
			{
				// 攻め方は、一番詰やすそうな(pn最小)のところを選ぶ。
				for (u32 i = 1; i < child_num; ++i)
					if (children[i].pn < children[selected_index].pn)
						selected_index = i;

				// 2つ目に小さなpnを探す。selected_indexを除いて最小を探す。
				// ※　次のノードのpnが、second_pn2を上回ったら、この2番目の子を調べたい。
				for (u32 i = 0; i < child_num; ++i)
					if (i != selected_index)
					{
						if (children[i].pn < pn2)
							pn2 = children[i].pn;

						// dnは、子ノードのdnの和になるから、次に進む子ノードのdnをdn_nextとして、残りの子ノードのdnの和が dn_sumが
						// dn_next + dn_sum > second_dn になったら子ノードの探索を終わりたいので、
						// dn_next > second_dn - dn_sum がその条件。
						// そこで、
						// second_dn2 = second_dn - dn_sum
						// に設定してやる。

						// ∞は除外しておく。
						if (children[i].dn < NodeType::DNPN_MATE)
							dn2 -= children[i].dn;
					}

				// dn2から引きすぎてマイナスの値になっている可能性があるが、その場合、dn2は無符号型なのですごく大きな数になっている。
				dn2 = std::min(dn2, second_dn);
			}
			else {
				// 受け方は、一番詰みにくそうな(dn最小)のところを選ぶ
				for (u32 i = 1; i < child_num; ++i)
					if (children[i].dn < children[selected_index].dn)
						selected_index = i;

				for (u32 i = 0; i < child_num; ++i)
					if (i != selected_index)
					{
						if (children[i].dn < dn2)
							dn2 = children[i].dn;

						if (children[i].pn < NodeType::DNPN_MATE)
							pn2 -= children[i].pn;
					}

				pn2 = std::min(pn2, second_pn);
			}

			// 引数に戻しとく。
			second_pn = pn2;
			second_dn = dn2;

			return &children[selected_index];
		}

		// あるnodeにぶら下がっている子ノードのpn,dnを集計して、このnodeのpn,dnに反映させる。
		template <bool or_node>
		void SummarizeNode(NodeType* node)
		{
			NodeCountType pn = or_node ? NodeType::DNPN_INF : 0                 ;
			NodeCountType dn = or_node ? 0                  : NodeType::DNPN_INF;

			u32 child_num = node->child_num;
			ASSERT_LV3(child_num != NodeType::CHILDNUM_NOT_INIT);
			NodeType* children = node_manager.get_children(node); // nullptrでありうるが、そのときはchild_num == 0;

			for (u32 i = 0; i < child_num ; ++i)
			{
				// 子ノードのpn,dn
				NodeCountType p = children[i].pn;
				NodeCountType d = children[i].dn;

				// 子ノードは、1手先の∞なので(手数に応じてpn,dnが減るように)1減算しておく。

				if (p >= NodeType::DNPN_MATE) --p;
				if (d >= NodeType::DNPN_MATE) --d;

				// and nodeでdnとpnの役割を入れ替えるなら、この場合分けは不要になるが…。
				if (or_node)
				{
					// 通常のdf-pnは、or nodeの
					// ・pnは、子ノードのpnの最小値
					// ・dnは、子ノードのdnの和
					// だが、ここでは詰み局面までの手数に応じた∞について取り扱うためにこれを拡張する。

					pn = std::min(pn, p);

					// 1) 有限と∞(MATE score)に関しては、∞のほうの値とする。
					// 2) 攻め方は、∞のなかで手数が一番短いものを選びたいため、両方が∞の時は、大きいほうを選ぶ。
					// 1),2)より、いずれかがMATE scoreなら、max()をとれば良い。

					if (dn >= NodeType::DNPN_MATE || d >= NodeType::DNPN_MATE)
						dn = std::max(dn, d);
					else
						dn += d; // ∞が絡まないならば単純な和
				}
				else {
					// pnに関してもor nodeのdn同様に処理しておけば、不詰を証明する時に
					// 攻め方の最長手順での攻め筋を調べるのに都合が良い。
					if (pn >= NodeType::DNPN_MATE || p >= NodeType::DNPN_MATE)
						pn = std::max(pn, p);
					else
						pn += p; // ∞が絡まないならば単純な和

					dn = std::min(dn, d);
				}
			}
			node->pn = pn;
			node->dn = dn;

			if (WithHash)
			{
				// 詰み or 不詰を証明しているので、repeatedな情報を用いたのかが問題となる。
				if (node->pn == 0 || node->dn == 0)
				{
					// or_nodeについて)
					//   pn == 0 なら、pn == 0の子ノードからrepeatedが伝播する。
					//   dn == 0 なら、全子ノードのrepeated の OR が伝播する。

					// and_nodeについて)
					//   dn == 0 なら、dn == 0の子ノードからrepeatedが伝播する。
					//   pn == 0 なら、全子ノードのrepeated の OR が伝播する。


					if (or_node)
						if (node->pn == 0) {
							for (u32 i = 0; i < child_num; ++i)
								if (children[i].pn == 0) {
									node->repeated = children[i].repeated;
									break;
								}
						}
						else {
							bool repeated = false;
							for (u32 i = 0; i < child_num; ++i)
								repeated |= children[i].repeated;
							node->repeated = repeated;
						}
					else // !or_node
						if (node->dn == 0) {
							for (u32 i = 0; i < child_num; ++i)
								if (children[i].dn == 0) {
									node->repeated = children[i].repeated;
									break;
								}
						}
						else {
							bool repeated = false;
							for (u32 i = 0; i < child_num; ++i)
								repeated |= children[i].repeated;
							node->repeated = repeated;
						}
				}
				else {
					// dn,pnが0ではないなら、repeatedであるかは問題とならないので何も伝播しない。
					//node->repeated = false;
				}
			}

		}

		// root nodeを展開する。
		void ExpandRoot(Position& pos)
		{
			// ORノード(詰ます側の手番)

			current_root = new_node();
			// さすがにbufferが0ってことはないから、ここでnullptrが返るとしたら初期化するコードを書き忘れている。
			// node_manager.reset_counter() , child_manager.reseet_counter()を呼び出しましょう。
			ASSERT_LV3(current_root != nullptr);

			// 開始局面は or nodeでござる。
			constexpr bool or_node = true;
			current_root->child_num = NodeType::CHILDNUM_NOT_INIT;
			ExpandNode<or_node>(pos, current_root);

			root_game_ply =  pos.game_ply();
			root_color    =  pos.side_to_move();
			root_pos      = &pos;
		}

		// Nodeを展開する(子ノードをぶら下げる)
		// 前提条件)
		//    1.  node != nullptr                      // 呼び出し元ですでに確保されている
		//    2.  node->child_num != CHILDNUM_NOT_INIT // まだ子が展開されていない
		// node->pn, node->dnの計算もしてリターンする。
		template <bool or_node>
		void ExpandNode(Position& pos, NodeType* node)
		{
			pos.in_check() ? ExpandNode<or_node,true>(pos, node) : ExpandNode<or_node,false>(pos, node);

			nodes_searched++;
		}

		template <bool or_node , bool INCHECK>
		void ExpandNode(Position& pos, NodeType* node)
		{
			ASSERT_LV3(node != nullptr && node->child_num == NodeType::CHILDNUM_NOT_INIT);

			if (WithHash)
			{
				// 置換表に登録されていれば、その結論を用いる。

				auto key = pos.state()->board_key();
				auto entry = hash_table->first_entry(key);

				// 置換表の値で証明されたか？
				bool proven = false;

				entry->lock();
				if (   entry->board_key  == (key & 0xffffffffffff) /* 48bit/4 = fが12個 */
					&& entry->root_color == root_color
					)
				{
#if 0
					// デバッグのために出力してみる。
					std::cout << pos << std::endl
						      << entry->get_move() << std::endl
							  << entry->get_hand() << std::endl;
#endif

					// board_keyは一致した。
					if (entry->is_mate) // pn == 0 , 詰み
					{
						//  or_node : 手番側(攻め方)の手駒が登録局面より優越している(or同じ)なら、この結論が使えるはず。
						// !or_node : 手番側(受け方)の手駒が登録局面より劣等している(or同じ)なら、この結論が使えるはず。
						//  詰みが証明されている局面の情報があるとして、
						// 　攻め方は、それより手駒が同じか多ければ同様に詰む。
						//   受け方は、それより手駒が同じか少なければ同様に詰む(詰まされる)。
						if (   ( or_node && hand_is_equal_or_superior(pos.state()->hand , entry->get_hand()))
							|| (!or_node && hand_is_equal_or_superior(entry->get_hand() , pos.state()->hand))
							)
						{
							node->template set_mate<true /* or nodeから見て詰む */>(entry->ply);
							proven = true;

#if 0
							// 登録されている内容が正しいか検証。
							if (or_node)
							{
								Mate::Dfpn::MateDfpnSolver dfpn(Mate::Dfpn::DfpnSolverType::Node64bit);
								dfpn.alloc(1024);
								Move move = dfpn.mate_dfpn(pos, 1000000);
								if (move == MOVE_NONE || move == MOVE_NULL)
								{
									// 詰むはずなのだが？
									std::cout << pos << std::endl;
								}
							}
#endif
						}
					}
					else { // dn == 0 , 不詰
						//  or_node : 手番側(攻め方)の手駒が登録局面より劣等している(or同じ)なら、この結論が使えるはず。
						// !or_node : 手番側(受け方)の手駒が登録局面より優越している(or同じ)なら、この結論が使えるはず。
						//  不詰が証明されている局面の情報があるとして、
						// 　攻め方は、それより手駒が同じか少なければ同様に詰まない。
						//   受け方は、それより手駒が同じか多ければ同様に詰まない
						if (   ( or_node && hand_is_equal_or_superior(entry->get_hand() , pos.state()->hand))
							|| (!or_node && hand_is_equal_or_superior(pos.state()->hand , entry->get_hand()))
							)
						{
							node->set_nomate(entry->ply);
							proven = true;

#if 0
							// 登録されている内容が正しいか検証。
							Mate::Dfpn::MateDfpnSolver dfpn(Mate::Dfpn::DfpnSolverType::Node64bit);
							dfpn.alloc(1024);
							if (dfpn.mate_dfpn(pos, 1000000) != MOVE_NULL)
							{
								// 詰まないはずなのだが？
								std::cout << pos << std::endl;
							}
#endif
						}
					}
				}
				// unlock()する前に取り出しておく。
				Move move = entry->get_move();

				// 置換表へのアクセスはこれ以上行わないので早い段階でunlockしておく。
				entry->unlock();

				if (proven)
				{
					// root nodeであれば子ノード一つ作って、この指し手を保存しておきたい。
					// ※　あとで初手がわからなくなるので…。
					if (node == current_root)
					{
						NodeType* child = new_node(1);
						// メモリ確保に失敗ぽ。
						if (child == nullptr)
							return;

						child->template init_mate_move<or_node>(move);
						node->set_child(1,node_manager.node_to_node_index(child));
					}
					else {
						// root node以外なら子なし扱いでもいいのでは…。
						// 厳密な読み筋がいま欲しいわけではあるまい。
						node->set_child(0);
					}

					// 置換表には循環を含む結論は書き出されていないので、そこから得た情報ならrepeatedはfalse。
					node->repeated = false;
					return;
				}

			}

			// orノード  : 手数がmax_game_plyを超えているなら不詰扱い。
			// 　　　　　  max_game_plyが0の時は無制限なのでこの判定はしない。
			// andノード : 手数がmax_game_plyを超えているなら不詰扱い。ただし、現局面で詰んでいる or 禁則手しか残っていないなら
			// 　　　　　　この限りではない。
			// なのでandノードでは、このチェックは、指し手生成しないと正確な判定ができない。
			// よって、このあとでチェックを行う。
			if (or_node && max_game_ply && max_game_ply < pos.game_ply())
			{
				// ここで次の一手は指せない。(例:256手ルールにおける257手目の局面)
				node->set_child(0);
				node->set_nomate();
				if (WithHash)
					node->repeated = true;
				return;
			}

			int plies_from_root = pos.game_ply() - root_game_ply;
			auto rep = mate_repetition(pos, plies_from_root, MAX_REPETITION_PLY, or_node);

			// node->init()がand nodeならpn,dnを入れ替えてくれるので場合分け不要。

			// or node
			switch (rep)
			{
			case MateRepetitionState::Mate:
				node->set_child(0);
				node->template set_mate<or_node>();
				if (WithHash)
					node->repeated = true;
				return;

			case MateRepetitionState::Mated:
				node->set_child(0);
				node->template set_mated<or_node>();
				if (WithHash)
					node->repeated = true;
				return;

			case MateRepetitionState::Unknown:
				break;
			}

#if 1
			// このnodeで1手詰めであるなら、その1手だけの子ノードを生成して、このノードにぶら下げておく。
			// mate_1ply()、王手がかかっていると呼び出せないので、王手がかかっていない時だけ。
			if (or_node && !pos.in_check())
			{
				Move mate_move = Mate::mate_1ply(pos);
				if (mate_move != MOVE_NONE)
				{
					// 詰んだ
					NodeType* child = new_node(1);
					// メモリ確保に失敗ぽ。
					if (child == nullptr)
						return;

					child->template init_mate_move<or_node>(mate_move);

					node->set_child(1,node_manager.node_to_node_index(child));
					node->template set_mate<or_node>(1 /* 次の局面で詰むので */);

					if (WithHash)
						node->repeated = false;
					
					return;
				}
			}
#endif

			// 歩の不成も含めて生成しておく。(詰め将棋ルーチンでそれが詰ませられないと悔しいので)
			auto mp = MovePicker<or_node, INCHECK, true , MoveOrdering>(pos);
			u32 child_num = (u32)mp.size();

			if (!or_node && max_game_ply && max_game_ply < pos.game_ply() && child_num)
			{
				// ここで次の一手は指せないが、指し手はあるので詰んでいないことは言える。
				// (例:256手ルールにおける257手目の局面)
				// なのでand nodeでは、これで逃れている。= 不詰。
				// 仮にこの指し手で連続王手の千日手(禁則)になるとしても、
				// ここで詰んでいないので手番が来た時点で引き分けが成立するから不詰。
				node->set_child(0);
				node->set_nomate();
				if (WithHash)
					node->repeated = true;
				return;
			}


			node->child_num = child_num;

			if (child_num == 0)
			{
				// 攻め方で指し手がない == 王手が続かない == 詰まない = pn ∞ , dn 0に。
				// 受け方で指し手がない == 詰んでいる = pn 0 , dn ∞に
				node->set_child(0);
				node->template set_mated<or_node>();
			}
			else {
				NodeType* children = new_node(child_num);
				// メモリ確保に失敗ぽ。
				if (children == nullptr)
					return;

				// 忘れないうちにぶら下げておく。
				node->set_child(child_num , node_manager.node_to_node_index(children) );

				for (auto m : mp)
					(children++)->template init<or_node>(m);

				// 子の数が決まったのでdn,pnを集計してからリターンする。
				// 今回生成した子の数ではあるが、指し手オーダリングをするなら、新規ノードのdn=pn=1ではない仕様かも知れないので。
				SummarizeNode<or_node>(node);
			}
		}

		// 新しいnodeを一つ確保して返す。確保できない時はnullptrが返る。
		NodeType* new_node(size_t size = 1) {
			auto node = node_manager.new_node(size);
			if (node == nullptr)
				out_of_memory = true;
			return node;
		}

	private:
		// 探索開始局面
		NodeType* current_root;
		Position* root_pos;
		int root_game_ply; // 探索開始局面のgame_ply。これは、千日手検出の時に必要となる。
		Color root_color;  // 探索開始局面の手番

		// 探索の時に指定されたノード制限
		NodeCountType nodes_limit;

		// 探索したノード数
		std::atomic<NodeCountType> nodes_searched;

		// new_node()などに失敗した
		std::atomic<bool> out_of_memory;

		// current_rootの局面での詰みの指し手
		Move bestmove;

		// この手数に達したら、引き分け扱い(不詰)とする。
		// set_max_game_ply()で設定された値。
		// 0は制限なし。
		int max_game_ply;

		// 詰み/不詰を証明済みの局面をcacheしておくtable
		MateHashTable* hash_table;

	private:
		// Node,Childのcustom allocatorみたいなもん。
		NodeManager<NodeCountType,MoveOrdering> node_manager;
	};

} // namespace Mate::Dfpn64 or Mate::Dfpn32

#endif //defined(DFPN64) || defined(DFPN32)

#endif // defined(USE_DFPN)
