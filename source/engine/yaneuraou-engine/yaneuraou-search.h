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

// 📌 Skill .. 手加減のための仕組み 📌
//    やねうら王では実装しない。

#if 0
// Skill structure is used to implement strength limit. If we have a UCI_Elo,
// we convert it to an appropriate skill level, anchored to the Stash engine.
// This method is based on a fit of the Elo results for games played between
// Stockfish at various skill levels and various versions of the Stash engine.
// Skill 0 .. 19 now covers CCRL Blitz Elo from 1320 to 3190, approximately
// Reference: https://github.com/vondele/Stockfish/commit/a08b8d4e9711c2

// Skill 構造体は強さ制限を実装するために使われる。
// UCI_Elo が指定されている場合、Stash エンジンを基準として
// 適切なスキルレベルに変換する。
// この方法は、さまざまなスキルレベルの Stockfish と
// さまざまなバージョンの Stash エンジンとの対局結果に基づいている。
// 現在、Skill 0 から 19 は、おおよそ CCRL Blitz の Elo 1320 から 3190 をカバーする。
// 参考: https://github.com/vondele/Stockfish/commit/a08b8d4e9711c2

// 💡 Skill構造体は強さの制限の実装に用いられる。
//    (わざと手加減して指すために用いる) →　やねうら王では未使用

struct Skill {
    // Lowest and highest Elo ratings used in the skill level calculation
    constexpr static int LowestElo  = 1320;
    constexpr static int HighestElo = 3190;

	// skill_level : 手加減のレベル。20未満であれば手加減が有効。0が一番弱い。(R2000以上下がる)
	// uci_elo     : 0以外ならば、そのelo ratingになるように調整される。
    Skill(int skill_level, int uci_elo) {
        if (uci_elo)
        {
            double e = double(uci_elo - LowestElo) / (HighestElo - LowestElo);
            level = std::clamp((((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438), 0.0, 19.0);
        }
        else
            level = double(skill_level);
    }

	// 手加減が有効であるか。
    bool enabled() const { return level < 20.0; }

	// SkillLevelがNなら探索深さもNぐらいにしておきたいので、
	// depthがSkillLevelに達したのかを判定する。
	bool time_to_pick(Depth depth) const { return depth == 1 + int(level); }

	// 手加減が有効のときはMultiPV = 4で探索
	Move pick_best(const RootMoves&, size_t multiPV);

    double level;
    Move   best = Move::none();
};

#else

// 💡 やねうら王ではSkillLevelを実装しない。
struct Skill {
    // dummy constructor
    Skill(int, int) {}

    // 常にfalseを返す。つまり、手加減の無効化。
    bool enabled() { return false; }
    bool time_to_pick(Depth) const { return true; }
    Move pick_best(const RootMoves&, size_t multiPV) { return Move::none(); }
    Move best = Move::none();
};

#endif


// 残り時間チェックを行ったり、main threadからのみアクセスされる探索manager
// 💡 Stockfishの同名のclassとほぼ同じ内容。Stockfishのsearch.hにあるSearchManagerを参考にすること。
// 🤔 YaneuraOuEngineの1インスタンスに対して、SearchManagerが1インスタンスあれば良いので、
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

    void check_time(Search::Worker& worker) {}

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

	// 💡 timeReductionは読み筋が安定しているときに時間を短縮するための係数。
	//     ここで保存しているのは、前回の反復深化のiterationの時のtimeReductionの値。
    double               previousTimeReduction;

    Value                bestPreviousScore;
    Value                bestPreviousAverageScore;
    bool                 stopOnPonderhit;

    size_t id;

    const UpdateContext& updates;

	// 📌 やねうら王独自 📌

	// 前回のPV出力した時刻。PVが詰まるのを抑制するためのもの。
	// 💡 startTimeからの経過時間。
	TimePoint lastPvInfoTime;

	// ponder用の指し手
    // 📝 やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのPV上の(相手の)指し手を用いるという独自仕様。
    //     Stockfish本家もこうするべきだと思う。
    Move ponder_candidate;
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

	// "isready"のタイミングでの初期化処理。
	virtual void isready() override;

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
// 💡 このclassのコードは、Stockfishのsearch.hにあるWorker classを参考にすること。
class YaneuraOuWorker: public Worker {
   public:
    // 💡 コンストラクタでWorkerのコンストラクタを初期化しないといけないので、
    //     少なくともWorkerのコンストラクタと同じ引数が必要。
    YaneuraOuWorker(OptionsMap&               options,
                    ThreadPool&               threads,
                    size_t                    threadIdx,
                    NumaReplicatedAccessToken numaAccessToken,
                    // 追加でYaneuraOuEngineからもらいたいもの
                    TranspositionTable& tt,
                    YaneuraOuEngine&    engine);

    // 評価関数のパラメーターが各NUMAにコピーされているようにする。
    virtual void ensure_network_replicated() override;

    // "go"コマンドでの探索の開始時にmain threadから呼び出される。
    virtual void start_searching() override;

    // "usinewgame"に対して呼び出される。対局前の初期化。
    virtual void clear() override;

    // 反復深化
    // 💡 並列探索のentry point。
    //     start_searching()から呼び出される。
    void iterative_deepening();

    // 探索本体
    // 💡 最初、iterative_deepening()のなかから呼び出される。
    template<NodeType nodeType>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    // 静止探索
    // 💡 search()から、残りdepthが小さくなった時に呼び出される。
    template<NodeType nodeType>
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {}

    // 📌 do_move～undo_move
    // 💡 do_moveするときにWorker::nodesをインクリメントする。

    void do_move(Position& pos, const Move move, StateInfo& st);
    void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck);
    void do_null_move(Position& pos, StateInfo& st);
    void undo_move(Position& pos, const Move move);
    void undo_null_move(Position& pos);

    // 📌 Stockfishのsearch.hで定義されているWorkerが持っているメンバ変数 📌

    // 近代的なMovePickerではオーダリングのために、スレッドごとにhistoryとcounter movesなどのtableを持たないといけない。
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;

    CapturePieceToHistory captureHistory;

    // コア数が多いか、長い持ち時間においては、ContinuationHistoryもスレッドごとに確保したほうが良いらしい。
    // cf. https://github.com/official-stockfish/Stockfish/commit/5c58d1f5cb4871595c07e6c2f6931780b5ac05b5
    // 添字の[2][2]は、[inCheck(王手がかかっているか)][capture_stage]
    // →　この改造、レーティングがほぼ上がっていない。悪い改造のような気がする。
    ContinuationHistory continuationHistory[2][2];

// TODO : あとで
#if 0
    PawnHistory           pawnHistory;

	CorrectionHistory<Pawn>         pawnCorrectionHistory;
    CorrectionHistory<Minor>        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation> continuationCorrectionHistory;
#endif

    TTMoveHistory ttMoveHistory;

    // MultiPVの時の現在探索中のPVのindexと、PVの末尾
    size_t pvIdx, pvLast;

    // nodes           : 探索node数これはWorker classのほうにある。
    // tbHits          : tablebaseにhitした回数。将棋では使わない。
    // bestMoveChanges : bestMoveが反復深化のなかで変化した回数
    std::atomic<uint64_t> /* nodes, tbHits,*/ bestMoveChanges;

    // selDepth : 選択探索の深さ。
    // 💡depthとPV lineに対するUSI infoで出力するselDepth。
    int selDepth, nmpMinPly;

    // aspiration searchで使う。
    Depth rootDepth, completedDepth;
    Value rootDelta;

    // Reductions lookup table initialized at startup
    // 起動時に初期化されるreductionsの参照表
    // 💡 reductionとは、残り探索深さを減らすこと。
    std::array<int, MAX_MOVES> reductions;  // [depth or moveNumber]

    // 📌 以下、やねうら王独自追加 📌

    // WorkerのポインタをYaneuraOuWorkerのポインタにupcastする。
    // 💡 このWorkerから派生させるようなclass設計だと必要になるので用意した。
    YaneuraOuWorker* toYaneuraOuWorker(std::unique_ptr<Worker>& worker) {
        return dynamic_cast<YaneuraOuWorker*>(worker.get());
    }

    // SearchManager*を取得する。
    // 💡 Stockfishとの互換性のために用意。
    SearchManager* main_manager() { return &manager; }

    // 並列探索において一番良い思考をしたthreadの選出。
    // 💡 Stockfishでは ThreadPool::get_best_thread()に相当するもの。
    YaneuraOuWorker* get_best_thread() const;

    // 📌 コンストラクタでもらったやつ 📌

    // 置換表
    TranspositionTable& tt;

    // Engine本体
    YaneuraOuEngine& engine;

    // SearchManager
    SearchManager& manager;
};

struct ConthistBonus {
    int index;
    int weight;
};

} // namespace Search

} // namespace YaneuraOu

#endif // defined(YANEURAOU_ENGINE)
#endif // YANEURAOU_SEARCH_H_INCLUDED
