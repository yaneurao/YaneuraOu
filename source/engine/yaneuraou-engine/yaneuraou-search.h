#ifndef YANEURAOU_SEARCH_H_INCLUDED
#define YANEURAOU_SEARCH_H_INCLUDED

#include "../../config.h"

#if defined (YANEURAOU_ENGINE)

#include "../../engine.h"
#include "../../search.h"
#include "../../book/book.h"
#include "../../tt.h"
#include "../../score.h"

namespace YaneuraOu {

namespace Search {

class YaneuraOuWorker;

// 🌈 やねうら王 探索部の設定
//     エンジンオプションで設定した設定値
struct SearchOptions
{
    SearchOptions() {
        max_moves_to_draw        = 100000;
        pv_interval              = 300;
        consideration_mode       = false;
        outout_fail_lh_pv        = true;
        generate_all_legal_moves = false;
        enteringKingRule         = EKR_27_POINT;
        lastPvInfoTime           = 0;
        computed_pv_interval     = 0;
    }

    // この構造体メンバーに対応するエンジンオプションを生やす
    void add_options(OptionsMap& options);

    // この手数で引き分けとなる。256なら256手目を指したあとに引き分け。
    // 📝 options["MaxMovesToDraw"]の設定値。
    int max_moves_to_draw;

    // PVの出力の抑制のために前回出力時間からの間隔を指定できる。単位は[ms]
    // 📝 options["PvInterval"]の設定値。
    // ⚠ 探索中は、こちらの値を使うのではなく、computed_pv_intervalを使う。
    TimePoint pv_interval;

    // 検討モード用のPVを出力するのか
    // 📝 options["ConsiderationMode"]の設定値。
    bool consideration_mode;

    // fail low/highの時にPVを出力するか。
    // 📝 options["OutputFailLHPV"]の設定値。
    bool outout_fail_lh_pv;

    // 合法手を生成する時に全合法手を生成するのか(歩の不成など)
    // エンジンオプションのGenerateAllLegalMovesの値がこのフラグに反映される。
    // 📝 options["GenerateAllLegalMoves"]の設定値。
    bool generate_all_legal_moves;

    // 入玉ルール設定
    // 📝 options["EnteringKingRule"]の値。
    EnteringKingRule enteringKingRule;

    // 📌 ここ以降は、SearchManagerで用いるメンバ変数 📌

    // 前回のPV出力した時刻。PVが詰まるのを抑制するためのもの。
    // lastPvInfoTime       : 出力した時のnow()の値。
    // computed_pv_interval : 実際のPVの出力間隔[ms]。
    //                      📝 options["PvInterval"]とoptions["ConsiderationMode"]から決定したもの。
    //                      ⚠ "go infinite"された時や、ConsiderationMode == trueなら、0 が設定される。
    TimePoint lastPvInfoTime;
    TimePoint computed_pv_interval;
};

// 📌 Skill .. 手加減のための仕組み 📌
//    やねうら王では実装しない。

#if STOCKFISH
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
    // スキルレベルの計算に用いるEloレーティングの下限と上限。
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
    Move pick_best(const RootMoves&, usize multiPV);

    double level;
    Move   best = Move::none();
};

#else

// 💡 やねうら王ではSkillLevelを実装しない。
struct Skill {
    // dummy constructor
    // 互換用の、何もしないコンストラクター。
    Skill(int, int) {}

    // 常にfalseを返す。つまり、手加減の無効化。
    bool enabled() { return false; }
    bool time_to_pick(Depth) const { return true; }
    Move pick_best(const RootMoves&, usize multiPV) { return Move::none(); }
    Move best = Move::none();
};

#endif


// 残り時間チェックを行ったり、main threadからのみアクセスされる探索manager
// 💡 Stockfishの同名のclassとほぼ同じ内容。Stockfishのsearch.hにあるSearchManagerを参考にすること。
// 🤔 YaneuraOuEngineの1インスタンスに対して、SearchManagerが1インスタンスあれば良いので、
//     やねうら王では、YaneuraOuEngineのメンバーとして持たせることにする。
// SearchManager manages the search from the main thread. It is responsible for
// keeping track of the time, and storing data strictly related to the main thread.
// SearchManagerはメインスレッドの探索を管理する。
// 時間の追跡と、メインスレッド専用のデータの保持を担当する。
class SearchManager {
   public:
    // 📝 やねうら王では、これはnamespace Searchで定義しておく。
#if STOCKFISH
    // Infoを更新した時のcallback。このcallbackを行うと標準出力に出力する。
    using UpdateShort    = std::function<void(const InfoShort&)>;
    using UpdateFull     = std::function<void(const InfoFull&)>;
    using UpdateIter     = std::function<void(const InfoIteration&)>;
    using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;
    using UpdateStart    = std::function<void()>;

    // PVを設定した時に出力するためのlistener
    struct UpdateContext {
        UpdateShort    onUpdateNoMoves;  // root局面で指し手がない時のhandler
        UpdateFull     onUpdateFull;
        UpdateIter     onIter;
        UpdateBestmove onBestmove;
        UpdateStart    onStart;
    };
#endif

    SearchManager(const UpdateContext& updateContext) :
        updates(updateContext) {}

    // 指し手をGUIに返す時刻になったかをチェックする。
    void check_time(YaneuraOuWorker& worker);

    // 現在のPV(読み筋)をUpdateContext::onUpdateFull()で登録する。
    // tt      : このスレッドに属する置換表
    // depth   : 反復深化のiteration深さ。
    void output_pv(Search::YaneuraOuWorker&   worker,
                   const ThreadPool&         threads,
                   const TranspositionTable& tt,
                   Depth                     depth);

    // 🌈 start_searching()より前にUI threadから
    //		Worker::pre_start_searching()が呼び出され、virtualなので派生classの
    //      YaneuraOuWorker::pre_start_searching()が呼び出され、そこから委譲される。
    //     ponderフラグなどの初期化はここで行う。
    void pre_start_searching(YaneuraOuWorker& worker);

    /*
		📓 持ち時間管理

		tm                 :     持ち時間制御用。
		originalTimeAdjust :     持ち時間制御のためのパラメーター。やねうら王では使用しない。
		callsCnt           :     main threadが一定のnode数を探索するごとにcheck_time()を呼び出したいので、そのためのカウンター。
	*/
    TimeManagement tm;
#if STOCKFISH
    double originalTimeAdjust;
#endif
    int callsCnt;

    // "go ponder"実行中であるかのフラグ
    // 💡 "ponderhit"が来るとfalseになる。
    // 📓 "ponderhit"をUI threadが受信 → Engine::set_ponderhit() → YaneuraOuEngine::set_ponderhit()
    //     という流れでこのフラグが変更される。
    std::atomic_bool ponder;

    std::array<Value, 4> iterValue;

    // 💡 timeReductionは読み筋が安定しているときに時間を短縮するための係数。
    //     ここで保存しているのは、前回の反復深化のiterationの時のtimeReductionの値。
    double previousTimeReduction;

    // 前回の探索時のbestScore,bestAverageScore。
    // 📝 aspiration searchの初期値として用いる。
    Value bestPreviousScore;
    Value bestPreviousAverageScore;

    // ("go ponder"で思考を開始していて)次に"ponderhit"を受信したら
    // 探索を即座に終了させていいところまで探索が進んでいるフラグ。
    bool stopOnPonderhit;

    const UpdateContext& updates;

    // 🌈 やねうら王独自 🌈

    /*
		 📓 やねうら王ではThread, ThreadPoolを完全に抽象化しているため、
			 ThreadPoolはincreaseDepthを持たず、このMainManagerが持っている。
	*/
    std::atomic<bool> increaseDepth;

    // やねうら王探索部で用いるオプション一覧
    SearchOptions search_options;

    // ponder用の指し手
    // 📝 やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのPV上の(相手の)指し手を用いるという独自仕様。
    //     Stockfish本家もこうするべきだと思う。
    Move ponder_candidate;

    // 前回のgamePly。今回と手番が異なるかを検出するのに用いる。
    // 📝 Stochastic Ponderの場合、手番が異なることになる。
    //     この時、bestPreviousScore、bestPreviousAverageScoreを反転させる必要がある。
    int lastGamePly;
};

/*
   やねうら王 Engine(やねうら王の通常探索部)

   📓 Stockfishから拡張して、やねうら王はエンジンを自由に差し替えられるようになっているので、
      自分のEngineを定義するには、Engine classから派生させる。

      このclassがStockfishのEngine classに相当する。
      エンジン共通で必要なものは、IEngine/Engine(これが、それぞれエンジンのinterfaceとエンジン基底class)に移動させた。
*/
class YaneuraOuEngine: public Engine {
   public:
    // 📌 StockfishのEngine classに合わせる 📌

    // 📝 やねうら王では、namespace Searchに書いてあるので不要。
#if STOCKFISH
    using InfoShort = Search::InfoShort;
    using InfoFull  = Search::InfoFull;
    using InfoIter  = Search::InfoIteration;
#endif

#if defined(EVAL_SFNN) && defined(ENABLE_NUMA_NETWORK_REPLICATION)
    YaneuraOuEngine();
    ~YaneuraOuEngine() override;
#else
    YaneuraOuEngine(/* std::optional<std::string> path = std::nullopt */) :
        manager(updateContext) {}
#endif

    // 📝 やねうら王では、CommandLine::gから取得できるので使わない。
    // const std::string binaryDirectory;

    // TODO : あとで整理する

    //NumaReplicationContext numaContext;

    // 📝 やねうら王では、base classであるEngine classが持っている。
#if STOCKFISH
    Position     pos;
    StateListPtr states;

    OptionsMap options;
    ThreadPool threads;
#endif

    // 📝 以下は、Engine classにある

	// 置換表
    // TranspositionTable tt;

    // TODO : あとで
    //LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;

    // Search::UpdateContext updateContext;

    // TODO : あとで
    //std::function<void(std::string_view)> onVerifyNetwork;

	// historyの集めたやつ。
    std::map<NumaIndex, SharedHistories>  sharedHists;

    // 🌈 やねうら王独自 🌈

    // 思考エンジンの追加オプションを設定する。
    virtual void add_options() override;

    // "isready"のタイミングでの初期化処理。
    virtual void isready() override;

    // "ponderhit"に対するhandler。
    virtual void set_ponderhit(bool b) override;

    // Threadのresizeするときのevent。
    virtual void resize_threads() override;

    // 置換表のresize event。
    virtual void set_tt_size(size_t mb) override;

	// 置換表の使用率を返す。
    virtual int get_hashfull(int maxAge) const override;

    // USI拡張コマンド "qsearch_psv" の実体。
    // .psv(PsvRecord列)の各局面をqsearch PVのleaf nodeで置換して書き出す。
    virtual bool qsearch_psv(const std::string& inputPath,
                             const std::string& outputPath,
                             size_t             workerCount,
                             std::string&       message) override;

	// 現在の局面の評価値の詳細を出力する。
    virtual void trace_eval() const override;

	// 現在の局面の評価値を出力する。
    virtual Value evaluate() const override;

    // StockfishのThreadPool::clear()にあったもの。
    void clear();

    // 定跡の指し手を選択するモジュール
    Book::BookMoveSelector book;

    // 探索manager
    // 📝 やねうら王では、Engine派生classがSearchMangerを持っている。
    Search::SearchManager manager;

    // Stockfishとの互換性のために用意。
    Search::SearchManager* main_manager() { return &manager; }

   private:
    friend class YaneuraOuWorker;
    // Worker間で共有するが、別EngineのSPSA設定とは独立したreductions表。
    std::array<int, MAX_MOVES> reductions{};

};

// やねうら王の探索Worker
// 📌 Stockfishから拡張して、やねうら王はエンジンを自由に差し替えられるようになっているので、
//     自分のWorkerを定義するには、Search::Worker classから派生させる。
// 💡 このclassのコードは、Stockfishのsearch.hにあるWorker classを参考にすること。
// Search::Worker is the class that does the actual search.
// It is instantiated once per thread, and it is responsible for keeping track
// of the search history, and storing data required for the search.
// Search::Workerは実際の探索を担当するクラスで、スレッドごとに1つ生成される。
// 探索履歴と探索に必要なデータを保持する。
// やねうら王では、この役割を派生クラスYaneuraOuWorkerが担う。
class YaneuraOuWorker: public Worker {
   public:
    // 💡 コンストラクタでWorkerのコンストラクタを初期化しないといけないので、
    //     少なくともWorkerのコンストラクタと同じ引数が必要。
    YaneuraOuWorker(SharedState&        sharedState,
					const ThreadIds&	ids,
                    // 追加でYaneuraOuEngineからもらいたいもの
                    YaneuraOuEngine&    engine);

    // Reset histories, usually before a new game.
    // 通常は新しい対局の前に、履歴をリセットする。
    virtual void clear() override;

    // Called when the program receives the UCI 'go' command. It searches
    // from the root position and outputs the "bestmove".
    // goコマンドを受信したときに呼び出され、ルート局面から探索してbestmoveを出力する。
    // やねうら王ではUSIのgoを扱う。補助スレッドでは探索のみを行う。
    virtual void start_searching() override;

	// 📝 これは、やねうら王ではbase classで定義されている。
    //bool is_mainthread() const { return threadIdx == 0; }

    // 評価関数のパラメーターが各NUMAにコピーされているようにする。
    virtual void ensure_network_replicated() override;

    // qsearch<PV>()をTT hitなし扱いで行い、この呼び出し中に得られたPVを返す。
    // 返されたPVを進めた局面が、qsearchで到達したleaf nodeになる。
    Value qsearch_pv(Position& pos, PVMoves& pv);

    // 📌 Stockfishのsearch.hで定義されているWorkerが持っているメンバ変数 📌

	// Public because they need to be updatable by the stats
    // stats によって更新可能である必要があるため、public

    // 近代的なMovePickerではオーダリングのために、スレッドごとにhistoryとcounter movesなどのtableを持たないといけない。
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;

    CapturePieceToHistory captureHistory;

    CorrectionHistory<Continuation> continuationCorrectionHistory;

    TTMoveHistory    ttMoveHistory;
    SharedHistories& sharedHistory;
    // 添字は[inCheck][capture_stage]。実体はNUMAノード内で共有する。
    ContinuationHistory (&continuationHistory)[2][2];
    const std::array<int, MAX_MOVES>& reductions;

   private:
    // 反復深化
    // 💡 並列探索のentry point。
    //     start_searching()から呼び出される。
    bool iterative_deepening();

    // 📌 do_move～undo_move
    // 📝 do_move()は、Worker::nodesをインクリメントする。
    //     do_null_move()は、nodesはインクリメントしない。
    // 💡 givesCheckはこの指し手moveで王手になるか。
    //     givesCheckとcapture_stage()の結果を呼び出し元から渡し、再計算を避ける。

    void
    do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck,
            const bool capture, Stack* const ss);
    void do_null_move(Position& pos, StateInfo& st, Stack* const ss);

	void undo_move(Position& pos, const Move move);
    void undo_null_move(Position& pos);

	// 探索本体
    // This is the main search function, for both PV and non-PV nodes
    // これは PV ノードおよび非 PV ノードの両方に対応するメインの探索関数
    // 💡 最初、iterative_deepening()のなかから呼び出される。
    template<NodeType nodeType>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, const bool cutNode);

    // 静止探索
    // Quiescence search function, which is called by the main search
    // メイン探索から呼ばれる静止探索関数
    // 💡 search()から、残りdepthが小さくなった時に呼び出される。
#if STOCKFISH
    template<NodeType nodeType>
#else
    // PSV変換用のTT読取無効化。通常探索は既定のtrueを使う。
    template<NodeType nodeType, bool ReadTT = true>
#endif
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta);

	// LMRのreductionの値を計算する。
    // ⚠ この関数は、Stockfish 17(2024.11)で、1024倍して返すことになった。
    //   i     : improving , 評価値が2手前から上がっているかのフラグ。
    //                       上がっていないなら悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
    //   d     : depth
    //   mn    : move_count
    //   dは探索深さ、mnは指し手番号。
    int reduction(bool i, Depth d, int mn) const;

	// Pointer to the search manager, only allowed to be called by the main thread
    // 検索マネージャへのポインタ。メインスレッドからのみ呼び出すことが許可されています。

#if STOCKFISH
    SearchManager* main_manager() const {
        assert(threadIdx == 0);
        assert(manager.get() != nullptr);
        return manager.get();
    }
#else
    // 💡 Stockfishとの互換性のために用意。
    SearchManager* main_manager() const { return &manager; }
#endif

	// 時間経過。
	// 💡 やねうら王では、SearchManagerがTimeManagement tmを持っていて、
	//     このclassがelapsed()を持っているのでそちらを用いる。
#if STOCKFISH
    TimePoint elapsed() const;
#endif

	// 評価関数
    Value evaluate(const Position& pos);

	// "go"で渡された探索条件。
	// 💡 やねうら王ではbase classが持っている。
    //LimitsType limits;

    // MultiPVの時の現在探索中のPVのindexと、PVの末尾
    usize pvIdx, pvLast;

    // nodes           : 探索node数これはbase classのほうにある。
    // tbHits          : tablebaseにhitした回数。将棋では使わない。
    // bestMoveChanges : bestMoveが反復深化のなかで変化した回数
    RelaxedAtomic<u64> /* nodes, tbHits,*/ bestMoveChanges;

    // selDepth : 選択探索の深さ。
    // 💡depthとPV lineに対するUSI infoで出力するselDepth。
    int selDepth, nmpMinPly;

	// 探索時に評価値に楽観的バイアスを与えるために用いるパラメーター。
	Value optimism[COLOR_NB];

#if STOCKFISH
	// 探索開始局面とrootでのStateInfo
	// 📝 やねうら王では、Engine classが持っている。
    Position  rootPos;
    StateInfo rootState;
    RootMoves rootMoves;
#endif

    // 現在探索中、または最後に開始した反復深化の深さ。
    Depth rootDepth;
    // 詰みを探すために一部の枝刈りを緩めるモード。
    bool seekMate = false;

    // 現在のMultiPV候補について、前回の反復深化で得たPV。
    // 次の反復深化でこのPV lineを辿っている間は、IIRとquiet shallow pruningを抑制する。
    PVMoves lastIterationIdxPV;

#if STOCKFISH
	// 📓 やねうら王では、Engine classが持っている。
	// threadIdx     : threadのindex(0からの連番), 0がmain thread
	// numaThreadIdx : NUMA内のthread index。
	// numaTotal     : NUMA内のtotal thread数。numaTotalThreads。
	// 📑この3つはコンストラクタで渡されたもの

	usize                     threadIdx, numaThreadIdx, numaTotal;
    NumaReplicatedAccessToken numaAccessToken;
#endif


#if STOCKFISH
	// やねうら王では、これらは基底classであるWorkerが持っている。

    // The main thread has a SearchManager, the others have a nullptr
    // メインスレッドはSearchManagerを持ち、他のスレッドはnullptrを持つ。
    std::unique_ptr<SearchManager> manager;

    // 📝 tablebaseは将棋では使わない。
    Tablebases::Config tbConfig;

    const OptionsMap&                               options;
    ThreadPool&                                     threads;
    TranspositionTable&                             tt;
	// TODO : あとで考える。
    //const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;

#endif

	// 💡 やねうら王では、SearchManagerは、NullObject patternをやめて、単に参照で持つ。
    // 🤔 Stockfishも、main threadからしか呼び出さないのだから、これでいいと思うのだが…。
    SearchManager& manager;

	// start_searching()より前にUI threadから呼び出される。
    // 📝 より詳しい説明は、Worker::pre_start_searching()のコメントを読むこと。
    virtual void pre_start_searching() override;

	// NNUEの評価関数の計算用

#if STOCKFISH
	// TODO : あとでよく考える。

	// NNUE評価関数のパラメーターがNumaごとにコピーされるようにする。
	const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;

	// Used by NNUE
	// NNUEで使う

	// NNUE評価関数のnetworkのL1層を保持しているstack
    Eval::NNUE::AccumulatorStack  accumulatorStack;
	// NNUE評価関数の差分計算用
    Eval::NNUE::AccumulatorCaches refreshTable;
#endif

#if STOCKFISH
    friend class Stockfish::ThreadPool;
#else
	// 🌈 以下、やねうら王独自追加 🌈

#if defined(EVAL_SFNN)
    Eval::NNUE::AccumulatorStack accumulatorStack;
#endif

	// WorkerのポインタをYaneuraOuWorkerのポインタにupcastする。
    // 💡 このWorkerから派生させるようなclass設計だと必要になるので用意した。
    YaneuraOuWorker* toYaneuraOuWorker(LargePagePtr<Worker>& worker) {
        //return dynamic_cast<YaneuraOuWorker*>(worker.get());
		// ⚠  RTTIが無効(コンパイル時に-frttiを指定している)ので
		//     dytnamic_castは使えない。
		//     このupcastができることはわかっているのでstatic_castを行う。
        return static_cast<YaneuraOuWorker*>(worker.get());
    }

    // Engine本体
	// 📝 コンストラクタで受け取ったもの。
    YaneuraOuEngine& engine;
#endif

	friend class SearchManager;
};

// ContinuationHistoryに対するbonusの係数
struct ConthistBonus {
    int index;
    int weight;
};

} // namespace Search

} // namespace YaneuraOu

#endif // defined(YANEURAOU_ENGINE)
#endif // YANEURAOU_SEARCH_H_INCLUDED
