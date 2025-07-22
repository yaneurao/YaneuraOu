#ifndef MATE_H_INCLUDED
#define MATE_H_INCLUDED

#include "../types.h"
#if defined (USE_MATE_1PLY)

#include <vector>
#include <memory> // std::unique_ptr<>
#include <atomic> // std::atomic<>

#include "../thread.h"

namespace YaneuraOu {

namespace Mate {

	// Mate関連で使うテーブルの初期化
	// ※　Bitboard::init()から呼び出される。
	void init();

	// 現局面で1手詰めであるかを判定する。1手詰めであればその指し手を返す。
	// ただし1手詰めであれば確実に詰ませられるわけではなく、簡単に判定できそうな近接王手による
	// 1手詰めのみを判定する。(要するに判定に漏れがある。)
	// 詰みがある場合は、その指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// 王手がかかっている局面で呼び出してはならない。(この関数は、王手を回避しながらの詰みを探すわけではない)
	Move mate_1ply(const Position& pos);

	// mate_1ply()の手番をtemplateにしたやつ。
	template <Color Us>
	Move mate_1ply_imp(const Position& pos);

	// 利きのある場所への取れない近接王手からの3手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	Move weak_mate_3ply(const Position& pos, int ply);

	// Mate::MateRepetition() で千日手判定を行った時の返し値
	enum class MateRepetitionState
	{
		Mate   , // 詰み扱い
		Mated  , // 詰まされた扱い
		Unknown, // そのいずれでもない
	};

	// MateSolverやMateDfpnSolverで千日手判定のために遡る最大手数
	constexpr int MAX_REPETITION_PLY = 16;

	// 連続王手などの千日手判定を行う。
	// plies_from_root : 詰み探索開始局面からの手数
	// repetition_ply  : 千日手判定を行う最大手数 (16ぐらいでいいと思う)
	// or_node         : 攻め方のnodeであればtrue。受け方のnodeであればfalse。
	MateRepetitionState mate_repetition(const Position& pos, const int plies_from_root , const int repetition_ply, bool or_node);


#if defined(USE_MATE_SOLVER)|| defined(USE_MATE_DFPN)

	// =========================
	// 詰み探索で用いる置換表
	// =========================

	// 詰み探索で用いる置換表のEntry
	// これは、詰み/不詰を証明した局面を記録しておくためのもの。
	// このentryに一切hitしなかったとしても、解くのに問題はない。(解く効率が悪くなるだけ)
	// なので、書き込みの時に前の内容はつねに上書きする。
	struct alignas(16) MateHashEntry
	{
		// Positionクラスが返す盤面のkeyの下位48bit
		// Position::board_key()で取得できる
		// board_key()の上位16bitは、この格納されているアドレスの計算に用いるので
		// 実質的に64bitすべてのbitが合致しないとこのEntryを選択されない。
		// ※　StockfishのTTと似た構造にする。
		u64 board_key : 48;

		// 詰み探索の開始局面の手番。
		// root color。
		// ※　例) 詰み探索の開始局面が先手で、不詰を証明したのなら、
		//     後手は詰まないという意味であるが先手は詰まされるかもしれない。
		u64 root_color : 1;

		// 現局面の手番。
		// やねうら王の実装では、Hash Keyの下位1bitは手番のはず。
		Color side_to_move() const { return (Color)(board_key & 1); }

		// 現在の手番
		// ↓これは現在のboard_keyのbit0に格納されていると考えられるので用いない。
		//u64 side_to_move : 1;

		// is_mate == true (1)ならば、plyは詰みまでの手数(DNPN_INF - dn)の値を表現している。(このときpn = 0)
		// is_mate == false(0)ならば、plyは不詰までの手数(DNPN_INF - pn)の値を表現している。(このときdn = 0)
		u64 is_mate   : 1; 

		// 詰みに関するスコア
		// ミクロコスモス(1525手)より長い詰将棋はいまのところ存在しないし、解かせることはないと思われるので11bit(2047まで表現できる)あれば十分。
		u64 ply       : 11;

		u64 padding1  : 2;

		// その時の最善手のgetterとsetter
		Move get_move() const { return (Move)(move16 + (move8 << 16)); }
		void set_move(Move move) { move16 = move.to_u16(); move8 = move.to_u32() >> 16; }

		// このEntryに格納されている手駒のgetter
		// save()する時に指定した手駒が返る。
		Hand get_hand() const { return (Hand)hand; }

		// このentryをlockする。
		void lock();

		// このentryをunlockする。
		void unlock();

		// このEntryに保存する。
		void save(Key board_key, Color root_color, /*Color side_to_move,*/ Hand hand, bool is_mate, u32 ply, Move move);

		//void probe(Key board_key, Color root_color);
		// →　直接entry操作するので要らないや。

	private:
		u16 move16;
		u8  move8;
		// その時の手番側の手駒(手駒の優越判定に用いる)
		u32 hand;

		// このentryのlock用
		// cluster->entries[0].mutexがlockされてたら、entries[1]側もlockされていると解釈する。
		std::atomic<bool> mutex;

		// 以上、16byte
	};

	// 詰み探索で用いる置換表本体
	//
	// これは、奇数手詰め、df-pnなど共通で使える。
	// 詰み、不詰みという結論だけを書き込む用。
	// このクラスが保持しているデータ構造であるMateHashEntryには、
	// 探索開始局面の手番(root_color)でフラグがあるので先後を混同することはない。
	// 複数スレッドから一つのMateHashTableを参照して使うのでlock～unlockは必要。
	// 複数のスレッドから、そんなに同一の局面をlockすることはないので、ここではCAS lockを用いる。
	class MateHashTable
	{
	public:

		// 与えられたboard_keyを持つMateHashEntryの先頭アドレスを返す。
		// (現状、1つしか該当するエントリーはない)
		// 取得したあと、lock()～unlock()して用いること。
		MateHashEntry* first_entry(const Key board_key, Color side_to_move) const;

		// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
		// このあと呼び出し側でclear()を呼び出す必要がある。
		void resize(size_t mbSize);

		// 置換表のエントリーの全クリア
		// 連続対局の時はクリアしなくともいいような気はするが…。
		void clear(ThreadPool& threads);

	private:
		// 置換表の先頭アドレス
		MateHashEntry* table = nullptr;

		// エントリーの数
		size_t entryCount = 0;
	};

#endif // defined(USE_MATE_SOLVER)|| defined(USE_MATE_DFPN)

	// n手詰め
#if defined(USE_MATE_SOLVER)

	// 詰み探索。内部的にメモリを確保しているわけではないので、
	// 詰み探索が必要な時に気軽にこのインスタンスを生成して問題ない。
	class MateSolver
	{
	public:
		// 奇数手詰め
		// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
		// ply     : 最大で調べる手数
		// gen_all : 歩の不成も生成するのか
		Move mate_odd_ply(Position& pos, const int ply, bool gen_all);

		// 最大探索深さ。これを超えた局面は不詰扱いとする。
		// Position::game_ply()がこれを超えた時点で不詰扱い。
		// 0を指定すると制限なし。デフォルトは0。
		void set_max_game_ply(int max_game_ply) { this->max_game_ply = max_game_ply; }

	private:

		// mate_odd_ply()の王手がかかっているかをtemplateにしたやつ。
		// ※　dlshogiのmateMoveInOddPlyReturnMove()を参考にさせていただいています。
		// INCHECK : 王手がかかっているか
		// GEN_ALL : 歩の不成も生成するのか
		template <bool INCHECK, bool GEN_ALL>
		Move mate_odd_ply(Position& pos, const int ply);

		// 偶数手詰め
		// 前提) 手番側が王手されていること。
		// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
		// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
		// ply     : 最大で調べる手数
		// gen_all : 歩の不成も生成するのか。
		Move mated_even_ply(Position& pos, const int ply, bool gen_all);

		// mated_even_ply()のtemplate版。
		// GEN_ALL : 歩の不成も生成するのか。
		template <bool GEN_ALL>
		Move mated_even_ply(Position& pos, const int ply);

		// 3手詰めチェック
		// 手番側が王手でないこと
		// mate_even_ply()から内部的に呼び出される。
		// INCHECK : 王手がかかっているか
		// GEN_ALL : 歩の不成も生成するのか。
		template <bool INCHECK, bool GEN_ALL>
		Move mate_3ply(Position& pos);

	private:
		// 探索開始時のgame_plyを保存しておく。
		// 千日手判定のためにこの局面以前に遡る時と、そうでない時とで処理が異なるので。
		int root_game_ply;

		// 最大探索深さ。これを超えた局面は不詰扱いとする。
		// Position::game_ply()がこれを超えた時点で不詰扱い。
		// 0を指定すると制限なし。デフォルトは0。
		int max_game_ply = 0;
	};

#endif // defined(USE_MATE_SOLVER)
}

#if defined(USE_MATE_DFPN)

namespace Mate::Dfpn
{
	// df-pn詰将棋ルーチンのinterface
	// 事前にメモリ確保やら何やらしないといけないのでクラス化してある。
	// 古来からあるdf-pnとは異なる、新しいコンセプトによる実装。
	class MateDfpnSolverInterface
	{
	public:
		// このクラスを用いるには、この関数を呼び出して事前にdf-pn用のメモリを確保する必要がある。
		// size_mb [MB] だけ探索用のメモリを確保する。
		virtual void alloc(size_t size_mb) = 0;

		// 探索ノード数の上限を指定してメモリを確保する。
		// alloc()かalloc_by_nodes_limit()か、どちらかを呼び出してメモリを確保すること！
		virtual void alloc_by_nodes_limit(size_t nodes_limit) = 0;

		// Hash Tableの設定。
		// 全スレッドで共用しているようなhash table
		// ※　詰み/不詰を証明済みの局面についてcacheしておくためのテーブル
		//    Node32bitWithHashのような"WithHash"とついているインスタンスに対して有効。
		virtual void set_hash_table(MateHashTable* hash_table) = 0;

		// 探索して初手を返す。
		// nodes_limit内に解ければその初手が返る。
		// 不詰が証明できれば、MOVE_NULL、解がわからなかった場合は、MOVE_NONEが返る。
		// nodes_limit : ノード制限。0を指定するとノード制限なし。(ただしメモリの制限から解けないことはある)
		virtual Move mate_dfpn(const Position& pos , u64 nodes_limit)= 0;

		// 最大探索深さ。これを超えた局面は不詰扱いとする。
		// Position::game_ply()がこれを超えた時点で不詰扱い。
		// 0を指定すると制限なし。デフォルトは0。
		virtual void set_max_game_ply(int max_game_ply) = 0;

		// mate_dfpn()がMOVE_NULL,MOVE_NONE以外を返した場合にその手順を取得する。
		// ※　最短手順である保証はない。
		virtual std::vector<Move> get_pv() const = 0;

		// 現在の探索中のPVを出力する。
		virtual std::vector<Move> get_current_pv() const = 0;

		// 解けた時に今回の探索ノード数を取得する。
		virtual u64 get_nodes_searched() const = 0;

		// 解けた時に今回の詰み手数を取得する。
		virtual int get_mate_ply() const= 0;

		// 探索を終了させる。
		// これで停止させる場合は、次回、明示的に dfpn_stop(false);としてから詰み探索を呼び出す必要がある。
		virtual void dfpn_stop(const bool stop) { this->stop = stop; };

		// mate_dfpn()でMOVE_NONE以外が返ってきた時にメモリが不足しているかを返す。
		virtual bool is_out_of_memory() const= 0;

		// hash使用率を1000分率で返す。
		virtual int hashfull() const = 0;

		virtual ~MateDfpnSolverInterface() {}

	protected:
		// 停止フラグ。これがtrueになると停止する。
		bool stop = false;
	};

	// DfpnのSolverの種類
	enum class DfpnSolverType
	{
		None,              // あとで設定する時に使う。

		// ガーベジなし。
		Node32bit        , // nodes_limit < 2^32 の時に使うやつ 省メモリ版
		Node16bitOrdering, // nodes_limit < 2^16 の時に使うやつ 省メモリ版 かつ orderingあり
		Node64bit        , // nodes_limit < 2^64 の時に使うやつ 
		Node48bitOrdering, // nodes_limit < 2^48 の時に使うやつ            かつ orderingあり

		// ガーベジなし、Hash対応
		Node32bitWithHash,
		Node16bitOrderingWithHash,
		Node64bitWithHash,
		Node48bitOrderingWithHash,

		// ガーベジあり

		// 未実装。気が向いたら実装するが、ふかうら王で使わないと思われるのであまり気が進まない…。
	};

	// MateDfpnSolverInterfaceの入れ物。
	class MateDfpnSolver : public MateDfpnSolverInterface
	{
	public:
		MateDfpnSolver(DfpnSolverType t);

		// Solverのtypeをあとから変更する。
		void ChangeSolverType(DfpnSolverType t);

		// このクラスを用いるには、この関数を呼び出して事前にdf-pn用のメモリを確保する必要がある。
		// size_mb [MB] だけ探索用のメモリを確保する。
		virtual void alloc(size_t size_mb) { impl->alloc(size_mb); }

		// 探索ノード数の上限を指定してメモリを確保する。
		// alloc()かalloc_by_nodes_limit()か、どちらかを呼び出してメモリを確保すること！
		virtual void alloc_by_nodes_limit(size_t nodes_limit) { impl->alloc_by_nodes_limit(nodes_limit); }

		// Hash Tableの設定。
		// 全スレッドで共用しているようなhash table
		// ※　詰み/不詰を証明済みの局面についてcacheしておくためのテーブル
		//    Node32bitWithHashのような"WithHash"とついているインスタンスに対して有効。
		virtual void set_hash_table(Mate::MateHashTable* table) { impl->set_hash_table(table); }

		// 探索して初手を返す。
		// nodes_limit内に解ければその初手が返る。
		// 不詰が証明できれば、MOVE_NULL、解がわからなかった場合は、MOVE_NONEが返る。
		// nodes_limit : ノード制限。0を指定するとノード制限なし。(ただしメモリの制限から解けないことはある)
		virtual Move mate_dfpn(const Position& pos, u64 nodes_limit) { return impl->mate_dfpn(pos, nodes_limit); }

		// 最大探索深さ。これを超えた局面は不詰扱いとする。
		// Position::game_ply()がこれを超えた時点で不詰扱い。
		// 0を指定すると制限なし。デフォルトは0。
		virtual void set_max_game_ply(int max_game_ply) { impl->set_max_game_ply(max_game_ply); }

		// mate_dfpn()がMOVE_NULL,MOVE_NONE以外を返した場合にその手順を取得する。
		// ※　最短手順である保証はない。
		virtual std::vector<Move> get_pv() const { return impl->get_pv(); }

		// 現在の探索中のPVを出力する。
		virtual std::vector<Move> get_current_pv() const { return impl->get_current_pv(); }

		// 解けた時に今回の探索ノード数を取得する。
		virtual u64 get_nodes_searched() const { return impl->get_nodes_searched(); }

		// 解けた時に今回の詰み手数を取得する。
		virtual int get_mate_ply() const { return impl->get_mate_ply(); }

		// mate_dfpn()でMOVE_NONE以外が返ってきた時にメモリが不足しているかを返す。
		virtual bool is_out_of_memory() const { return impl->is_out_of_memory(); }

		// hash使用率を1000分率で返す。
		virtual int hashfull() const { return impl->hashfull(); }

	private:
		std::unique_ptr<MateDfpnSolverInterface> impl;
	};

} // namespace Mate::Dfpn
#endif // defined(USE_MATE_DFPN)

} // namespace YaneuraOu
#endif // defined (USE_MATE_1PLY)

#endif // MATE_H_INCLUDED
