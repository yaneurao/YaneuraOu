#ifndef YANEURAOU_SEARCH_H_INCLUDED
#define YANEURAOU_SEARCH_H_INCLUDED

#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

#include "../../engine.h"
#include "../../search.h"
#include "../../book/book.h"
#include "../../tt.h"

namespace YaneuraOu {

// やねうら王 Engine
// 📌 Stockfishから拡張して、やねうら王はエンジンを自由に差し替えられるようになっているので、
//     自分のEngineを定義するには、Engine classから派生させる。
class YaneuraOuEngine : public Engine
{
public:

	// 思考エンジンの追加オプションを設定する。
	virtual void add_options() override;

	// 置換表
	TranspositionTable tt;

	// 定跡の指し手を選択するモジュール
	Book::BookMoveSelector book;
};

namespace Search {

// やねうら王の探索Worker
// 📌 Stockfishから拡張して、やねうら王はエンジンを自由に差し替えられるようになっているので、
//     自分のWorkerを定義するには、Search::Worker classから派生させる。
class YaneuraOuWorker : public Worker
{
public:
	// 💡 コンストラクタでWorkerのコンストラクタを初期化しないといけないので、
	//     少なくともWorkerのコンストラクタと同じ引数が必要。
	YaneuraOuWorker(OptionsMap& options, ThreadPool& threads, size_t threadIdx, NumaReplicatedAccessToken numaAccessToken,
		// 追加でYaneuraOuEngineからもらいたいもの
		TranspositionTable& tt);

	// 評価関数のパラメーターが各NUMAにコピーされているようにする。
	virtual void ensure_network_replicated() override;

	// 探索の開始時に呼び出される。
	virtual void start_searching() override;

	// 反復深化
	// 💡 並列探索しているmain thread以外のthreadのentry point
	void iterative_deepening() {}

	// 並列探索において一番良い思考をしたthreadの選出。
    // 💡 Stockfishでは ThreadPool::get_best_thread()に相当するもの。
    YaneuraOuWorker* get_best_thread() const;

	// コンストラクタでもらった置換表
	TranspositionTable& tt;

	Depth     rootDepth, completedDepth;
};

} // namespace Search

} // namespace YaneuraOu

#endif

#endif // YANEURAOU_SEARCH_H_INCLUDED
