#ifndef YANEURAOU_SEARCH_H_INCLUDED
#define YANEURAOU_SEARCH_H_INCLUDED

#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

#include "../../engine.h"
#include "../../search.h"
#include "../../book/book.h"
#include "../../tt.h"
#include "../../score.h"

namespace YaneuraOu {

namespace Search {

// PVの短いやつ
struct InfoShort {
    int   depth;
    Score score;
};

// PVの長いやつ
struct InfoFull: InfoShort {
	// 選択的な探索深さ
	int              selDepth;

	// MultiPVの設定数
    size_t           multiPV;

	// 💡勝率はやねうら王では使わない
    //std::string_view wdl;

	// boundを文字列化したもの
	std::string_view bound;

	// 経過時間
    size_t           timeMs;

	// 探索したnode数
    size_t           nodes;

	// NPS
    size_t           nps;

	// 💡tbHitsもやねうら王では使わない。(tb = tablebases)
    //size_t           tbHits;

	// PVを文字列化したもの
	std::string_view pv;

	// hashfullを文字列化したもの
    int              hashfull;
};

// 反復深化のIteration中のPV出力
struct InfoIteration {
	// 探索深さ
    int              depth;
	// 現在探索中の指し手を文字列化したもの
    std::string_view currmove;
	// 現在探索中の指し手のナンバー
    size_t           currmovenumber;
};

// 残り時間チェックを行ったり、main threadからのみアクセスされる探索manager
// 💡 Stockfishの同名のclassとほぼ同じ内容。
//     YaneuraOuEngineの1インスタンスに対して、SearchManagerが1インスタンスあれば良いので、
//     やねうら王では、YaneuraOuEngineのメンバーとして持たせることにする。
class SearchManager {
   public:
	// Infoを更新した時のcallback。このcallbackを行うと標準出力に出力する。
    using UpdateShort    = std::function<void(const InfoShort&)>;
    using UpdateFull     = std::function<void(const InfoFull&)>;
    using UpdateIter     = std::function<void(const InfoIteration&)>;
    using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;

	// PVを設定した時にupdateするためのcallback集。
	struct UpdateContext {
        UpdateShort    onUpdateNoMoves; // root局面で指し手がない時のhandler
        UpdateFull     onUpdateFull;
        UpdateIter     onIter;
        UpdateBestmove onBestmove;
    };

    SearchManager(const UpdateContext& updateContext) :
        updates(updateContext) {}

    //void check_time(Search::Worker& worker) override;

	#endif

	// 現在のPVをUpdateContext::onUpdateFull()で登録する。
    void pv(Search::Worker&           worker,
            const ThreadPool&         threads,
            const TranspositionTable& tt,
            Depth                     depth);

	//Stockfish::TimeManagement tm;
    // 持ち時間管理
    TimeManagement            tm;

	double                    originalTimeAdjust;
    int                       callsCnt;
    std::atomic_bool          ponder;

    std::array<Value, 4> iterValue;
    double               previousTimeReduction;
    Value                bestPreviousScore;
    Value                bestPreviousAverageScore;
    bool                 stopOnPonderhit;

    size_t id;

    const UpdateContext& updates;
};
}

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

	// 探索manager
    Search::SearchManager manager;

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
		TranspositionTable& tt,
		YaneuraOuEngine& engine
	);

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

	// SearchManager*を取得する。
	// 💡 Stockfishとの互換性のために用意。
	SearchManager* main_manager() { return &manager; }


	// コンストラクタでもらった置換表
	TranspositionTable& tt;

	// コンストラクタでもらったengine
	YaneuraOuEngine& engine;

	// コンストラクタでもらったSearchManager
	SearchManager& manager;

	Depth     rootDepth, completedDepth;
};

} // namespace Search

} // namespace YaneuraOu


#endif // YANEURAOU_SEARCH_H_INCLUDED
