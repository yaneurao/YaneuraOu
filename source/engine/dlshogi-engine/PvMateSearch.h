#ifndef __PV_MATE_SEARCH_H__
#define __PV_MATE_SEARCH_H__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include <thread>
#include <set>

#include "../../mate/mate.h"
#include "Node.h"

namespace dlshogi {

	class DlshogiSearcher;

	//	PV lineの詰探索
	class PvMateSearcher
	{
	public:
		// nodes : 最大探索ノード数
		PvMateSearcher(const int nodes, DlshogiSearcher* dl_searcher);

		PvMateSearcher(PvMateSearcher&& o) noexcept :
			th(o.th), dfpn(std::move(o.dfpn)), dl_searcher(o.dl_searcher), nodes_limit(o.nodes_limit),
			ready_th(o.ready_th), term_th(o.term_th)
		{} // 未使用
		// ⇨　エンジンオプションの PV_Mate_Search_Threads を途中で変更しない限りは…。

		// コンストラクタで渡されたnodesを返す。
		int get_nodes_limit() const { return nodes_limit;  }

		// 詰み探索スレッドを開始する。
		void Run();

		// 詰み探索スレッドに停止信号を送る。
		void Stop(const bool stop = true);

		// 詰み探索スレッドの終了を待機する。
		void Join();

		// thread poolの仕組みで待機させていたスレッドを終了させる。
		void Term();

		~PvMateSearcher() { Term(); }

	private:
		// uct_nodeからPVを辿り、詰み探索をしていない局面を見つけたら
		// df-pnで詰探索を1回行う。
		// 詰み探索を行う局面がないか、詰探索を1回したならば、returnする。
		void SearchInner(Position& pos, Node* uct_node, ChildNode* child_node, bool root);

		// BFS型の詰み探索をする。depth = (df-pnを呼び出す)残り探索depth。dfpnを呼び出した回数を返す。
		//void SearchInnerBFS(Position& pos, Node* uct_node, ChildNode* child_node, bool root, int depth);

		//// 探索済みノードを表現する
		//static std::set<ChildNode*> searched;
		//// ⇑にアクセスするときのmutex
		//static std::mutex mtx_searched;
		// ⇨ dlshogiではこれで各局面がdf-pn探索済みかを管理しているが、
		//   ふかうら王では、Node側に1bit flagを持たせることにする。

		// 探索用のスレッド
		std::thread* th;

		// PV lineのmate solver
		Mate::Dfpn::MateDfpnSolver dfpn;
		// 1局面の詰探索のノード数の上限
		int nodes_limit;

		// 停止フラグ
		std::atomic<bool> stop;

		// globalにアクセスすると行儀悪いので..
		DlshogiSearcher* dl_searcher;

		// thread id。この番号によって挙動を変えたい時に用いる。
		// int thread_id;

		// ===============
		// thread pool関連
		// ===============

		// Stop()～Join()ではthreadを待機させておき、次のRunでそのthreadを再利用する。
		// Term()で本当に終了させる。

		// --- スレッドプール用

		// ⇓⇓の２つのboolフラグにアクセスする時のmutex。
		std::mutex mtx_th;
		// poolしているthreadをwaitするために使う。
		std::condition_variable cond_th;

		// これをtrueにして⇑のCVでスレッド起こすとpoolしていたスレッドが再開する。
		bool ready_th;

		// poolしていたthreadを本当に終了させるかのフラグ。
		bool term_th;
	};

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __PV_MATE_SEARCH__

