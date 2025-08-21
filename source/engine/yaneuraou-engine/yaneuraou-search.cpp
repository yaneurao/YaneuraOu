#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

// -----------------------
//  やねうら王 標準探索部
// -----------------------

// 計測資料置き場 : https://github.com/yaneurao/YaneuraOu/wiki/%E6%8E%A2%E7%B4%A2%E9%83%A8%E3%81%AE%E8%A8%88%E6%B8%AC%E8%B3%87%E6%96%99

// -----------------------
//   includes
// -----------------------

#include "../../search.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>	// std::log(),std::pow(),std::round()
#include <cstring>	// memset()

#include "yaneuraou-search.h"
#include "../../position.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"
#include "../../book/book.h"
#include "../../movepick.h"
#include "../../usi.h"
#include "../../learn/learn.h"
#include "../../mate/mate.h"
#include "../../tune.h"

namespace YaneuraOu {

using namespace Search;
using namespace Eval;  // Eval::PieceValue

// -------------------
// 🌈 やねうら王独自追加
// -------------------

// 🌈 tune.pyによって自動的にTUNEする変数宣言がここ以下に追加される。
// 📝 tune.pyとは、パラメーター自動調整フレームワークのスクリプトである。
//     https://github.com/yaneurao/YaneuraOu-ScriptCollection/tree/main/SPSA
//                            %%TUNE_DECLARATION%%


// この構造体メンバーに対応するエンジンオプションを生やす
void SearchOptions::add_options(OptionsMap& options) {
    // 引き分けまでの最大手数。256手ルールのときに256を設定すると良い。0なら無制限。
    /*
		📓 0が設定されていたら、引き分けなしだから100000が代入されることになっている。
		    (残り手数を計算する時に桁あふれすると良くないのでint_maxにはしていない)

			初手(76歩とか)が1手目である。
			1手目を指す前の局面はPosition::game_ply() == 1である。

			そして256手指された時点(257手目の局面で指す権利があること。
			サーバーから257手目の局面はやってこないものとする)で引き分けだとしたら
			257手目(を指す前の局面)は、game_ply() == 257である。
			これが、引き分け扱いということになる。

			pos.game_ply() > max_moves_to_draw
		　	で(かつ、詰みでなければ)引き分けということになる。

			この引き分けの扱いについては、以下の記事が詳しい。
			多くの将棋ソフトで256手ルールの実装がバグっている件
			https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/

	*/
    options.add("MaxMovesToDraw", Option(0, 0, 100000, [&](const Option& o) {
                    // これ0の時、何らか設定しておかないと探索部でこの手数を超えた時に
                    // 引き分け扱いにしてしまうので、無限大みたいな定数の設定が必要。
                    max_moves_to_draw = int(o);
                    if (max_moves_to_draw == 0)
                        max_moves_to_draw = 100000;
                    return std::nullopt;
                }));

    //  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
    options.add("PvInterval", Option(300, 0, 100000000, [&](const Option& o) {
                    pv_interval = s64(o);
                    return std::nullopt;
                }));

    // 検討モード用のPVを出力するモード
    options.add("ConsiderationMode", Option(false, [&](const Option& o) {
                    consideration_mode = o;
                    return std::nullopt;
                }));

    // fail low/highのときにPVを出力するかどうか。
    options.add("OutputFailLHPV", Option(true, [&](const Option& o) {
                    outout_fail_lh_pv = o;
                    return std::nullopt;
                }));

    // すべての合法手を生成するのか
    options.add("GenerateAllLegalMoves", Option(false, [&](const Option& o) {
                    generate_all_legal_moves = o;
                    return std::nullopt;
                }));

    // 入玉ルール
    options.add("EnteringKingRule",
                Option(EKR_STRINGS, EKR_STRINGS[EKR_27_POINT], [&](const Option& o) {
                    enteringKingRule = to_entering_king_rule(o);
                    return std::nullopt;
                }));
}


void SearchManager::pre_start_searching(YaneuraOuWorker& worker) {
    // 🤔 StockfishのThreadPool::start_thinking()にあった以下の初期化をこちらに移動させた。

    stopOnPonderhit /* = stop = abortedSearch */ = false;
    ponder                                       = worker.limits.ponderMode;
    increaseDepth                                = true;
}

// 思考エンジンの追加オプションを設定する。
// 💡 Stockfishでは、Engine::Engine()で行っている。
void YaneuraOuEngine::add_options() {

	// 📌 基本設定(base classのadd_options()を呼び出してやる)

	Engine::add_options();

	// 📌 この探索部が用いるオプションの追加。

#if STOCKFISH
    options.add(  //
        "USI_Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
            set_tt_size(o);
            return std::nullopt;
        }));
#else
	// 🌈 やねうら王では、default値を1024に変更。
    options.add(  //
        "USI_Hash", Option(1024, 1, MaxHashMB, [this](const Option& o) {
            set_tt_size(o);
            return std::nullopt;
        }));
#endif

	// その局面での上位N個の候補手を調べる機能
    // ⇨　これMAX_MOVESで十分。
    options.add("MultiPV", Option(1, 1, MAX_MOVES));

    options.add("DrawValueBlack", Option(-2, -30000, 30000));
    options.add("DrawValueWhite", Option(-2, -30000, 30000));

    // 投了スコア
    options.add("ResignValue", Option(99999, 0, 99999));

	// 📌 SearchOptionsが用いるオプションの追加

	manager.search_options.add_options(options);

#if defined(EVAL_LEARN)
    // 評価関数の学習を行なうときは、評価関数の保存先のフォルダを変更できる。
    // デフォルトではevalsave。このフォルダは事前に用意されているものとする。
    // このフォルダ配下にフォルダを"0/","1/",…のように自動的に掘り、そこに評価関数ファイルを保存する。
    options.add("EvalSaveDir", Option("evalsave"));
#endif

	// 📌 TimeManagementが用いるオプションの追加

    manager.tm.add_options(options);

	// 📌 定跡が用いるオプションの追加

    book.add_options(options);

    // 💡  以下の設定のうち、"isready"のタイミングでoptionsから値を取得するものに関しては
    //      event handlerは設定しない。

    // 引き分けを受け入れるスコア
    // 歩を100とする。例えば、この値を -100にすると引き分けの局面は評価値が -100とみなされる。

    // 千日手での引き分けを回避しやすくなるように、デフォルト値を-2。
    // ちなみに、-2にしてあるのは、
    //  int draw_value = Options["DrawValueBlack"] * PawnValue / 100; でPawnValueが100より小さいので
    // 1だと切り捨てられてしまうからである。

    // Stockfishでは"Contempt"というエンジンオプションであったが、先後の区別がつけられないし、
    // 分かりづらいので変更した。


    // 📌 Stockfishにはあるが、やねうら王ではサポートしないオプション
#if STOCKFISH
	// 弱くするために調整する。20なら手加減なし。0が最弱。
	options.add("Skill Level", Option(20, 0, 20));

	options.add("Move Overhead", Option(10, 0, 5000));

	// nodes as timeモード。
	// ミリ秒あたりのノード数を設定する。goコマンドでbtimeが、ここで設定した値に掛け算されたノード数を探索の上限とする。
	// 0を指定すればnodes as timeモードではない。
	// 例) 600knpsのPC動作をシミュレートするならば600を指定する。
	// 📝 やねうら王では、この機能、サポートしないことにする。GUIがどうせサポートしていないので…。
	options.add("nodestime", Option(0, 0, 10000));

	options.add("UCI_Chess960", Option(false));

	options.add("UCI_LimitStrength", Option(false));

	// Eloレーティングを指定して棋力調整するためのエンジンオプション。
	options.add("UCI_Elo",
		Option(Stockfish::Search::Skill::LowestElo, Stockfish::Search::Skill::LowestElo,
			Stockfish::Search::Skill::HighestElo));

	options.add("UCI_ShowWDL", Option(false));

	// 💡 Syzygyとは、終盤のTablebases関連。将棋では使えない。

	options.add(  //
		"SyzygyPath", Option("", [](const Option& o) {
			Tablebases::init(o);
			return std::nullopt;
			}));

	options.add("SyzygyProbeDepth", Option(1, 1, 100));

	options.add("Syzygy50MoveRule", Option(true));

	options.add("SyzygyProbeLimit", Option(7, 0, 7));
#endif

	// 🌈 tune.pyによってここ以下に自動的にエンジンオプションが追加される。
	//                      %%TUNE_OPTIONS%%


	// もしTUNE()マクロで新たにパラメーターを追加したなら、それを反映させる。
	Tune::init(options);
}

// "isready"のタイミングでの初期化処理。
void YaneuraOuEngine::isready() {

	// 🌈 やねうら王独自オプションの内容を設定などに反映させる。

#if defined(USE_CLASSIC_EVAL)
    // 📌 旧評価関数は、isreadyに対して呼び出す。
    //     評価関数パラメーターの読み込み処理が必要。
    Engine::run_heavy_job([] { Eval::load_eval(); });

    // 初期化タイミングがなかったので、
    // このタイミングで平手の局面に初期化しておく。
    // 💡 これをしておかないとデバッグの時に、"isready"のあと
    //     "position"コマンドを送信しないと局面が不正で落ちて面倒。
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(StartSFEN, &states->back());
#endif

	// 📌 基本設定(base classのisready()を呼び出してやる)
	//Engine::isready();
	// 🤔 ttのclearに先立ち、ThreadPoolがスレッドを用意してくれていないといけない。

	// エンジン設定のスレッド数を反映させる。
    resize_threads();

	// 置換表のclear
	tt.clear(threads);

	// StockfishのThreadPool::clear()にあったコード。
	clear();

	// 定跡の読み込み
    book.read_book();

	// 🌈 tune.pyによってここ以下に自動的にエンジンオプションが追加される。
    //                      %%TUNE_ISREADY%%


    sync_cout << "readyok" << sync_endl;
}

// StockfishのThreadPool::clear()にあったもの。
// 📝 isready()から呼び出される。対局開始時に実行される。
void YaneuraOuEngine::clear()
{
    // 🤔 以下の初期化は、StockfishのThreadPool::clear()にあったもの。
    //     やねうら王では、これはEngine派生classで行う。

    // These two affect the time taken on the first move of a game:
    // これら2つは、ゲームの最初の手にかかる時間に影響する。

    main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
    main_manager()->previousTimeReduction    = 0.85;

    main_manager()->callsCnt          = 0;
    main_manager()->bestPreviousScore = VALUE_INFINITE;
#if STOCKFISH
    main_manager()->originalTimeAdjust = -1;
#else
    main_manager()->lastGamePly = 0;
#endif
    main_manager()->tm.clear();
}


// 🌈 "ponderhit"に対する処理。
void YaneuraOuEngine::set_ponderhit(bool b) {

	// 📝 ponderhitしたので、やねうら王では、
    //     現在時刻をtimer classに保存しておく必要がある。
	// 💡 ponderフラグを変更する前にこちらを先に実行しないと
	//     ponderフラグを見てponderhitTimeを参照して間違った計算をしてしまう。
    manager.tm.ponderhitTime = now();

	manager.ponder           = b;
}

// スレッド数を反映させる関数
void YaneuraOuEngine::resize_threads() {
    // 💡 Engine::resize_threads()を参考に書く。

    // 📌 探索の終了を待つ
    threads.wait_for_search_finished();

    // 📌 スレッド数のリサイズ

    auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken) {
        return std::make_unique<Search::YaneuraOuWorker>(

			// Worker基底classが渡して欲しいもの。
			options, threads, threadIdx, numaAccessToken,

			// 追加でYaneuraOuEngineからもらいたいもの
			tt, *this);
    };

    threads.set(numaContext.get_numa_config(), options, options["Threads"], worker_factory);

	// 置換表の割り当て
	set_tt_size(options["USI_Hash"]);
 
    // 📌 NUMAの設定

    // スレッドの用いる評価関数パラメーターが正しいNUMAに属するようにする
    threads.ensure_network_replicated();
}

// 置換表の割り当て
void YaneuraOuEngine::set_tt_size(size_t mb){
	wait_for_search_finished();
	tt.resize(mb, threads);
}

// 置換表の使用率を返す。
// 🌈 やねうら王では派生class側でget_hashfull()を実装する。
int YaneuraOuEngine::get_hashfull(int maxAge) const
{
	return tt.hashfull(maxAge);
}

// utility functions

void YaneuraOuEngine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
#if STOCKFISH
	p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());
#else
    p.set(pos.sfen(),&trace_states->back());
#endif
    verify_networks();

    //sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
	// TODO あとで
}

// 現在の局面の評価値を出力する。
Value YaneuraOuEngine::evaluate() const {
	verify_networks();
    return Eval::evaluate(pos); // cpに変換するか？まあいいか…。
}

// 並列探索において一番良い思考をしたthreadの選出。
// 💡 Stockfishでは ThreadPool::get_best_thread()に相当するもの。
YaneuraOuWorker* YaneuraOuWorker::get_best_thread() const {

    auto& threads = this->threads.threads;

    Thread* bestThread = threads.front().get();
    Value   minScore   = VALUE_NONE;

    std::unordered_map<Move, int64_t, Move::MoveHash> votes(
      2 * std::min(threads.size(), bestThread->worker->rootMoves.size()));

    // Find the minimum score of all threads
    for (auto&& th : threads)
        minScore = std::min(minScore, th->worker->rootMoves[0].score);

    // Vote according to score and depth, and select the best thread
    auto thread_voting_value = [minScore](Thread* th) {
        // Workerから派生させているのでdynamic_castしてしまう。
		// ⚠ RTTIを有効化していないので単にstatic_castにする。
        auto worker = static_cast<Search::YaneuraOuWorker*>(th->worker.get());
        return (th->worker->rootMoves[0].score - minScore + 14) * int(worker->completedDepth);
    };

    for (auto&& th : threads)
        votes[th->worker->rootMoves[0].pv[0]] += thread_voting_value(th.get());

    for (auto&& th : threads)
    {
        const auto bestThreadScore = bestThread->worker->rootMoves[0].score;
        const auto newThreadScore  = th->worker->rootMoves[0].score;

        const auto& bestThreadPV = bestThread->worker->rootMoves[0].pv;
        const auto& newThreadPV  = th->worker->rootMoves[0].pv;

        const auto bestThreadMoveVote = votes[bestThreadPV[0]];
        const auto newThreadMoveVote  = votes[newThreadPV[0]];

        const bool bestThreadInProvenWin = is_win(bestThreadScore);
        const bool newThreadInProvenWin  = is_win(newThreadScore);

        const bool bestThreadInProvenLoss =
          bestThreadScore != -VALUE_INFINITE && is_loss(bestThreadScore);
        const bool newThreadInProvenLoss =
          newThreadScore != -VALUE_INFINITE && is_loss(newThreadScore);

        // We make sure not to pick a thread with truncated principal variation
        const bool betterVotingValue =
          thread_voting_value(th.get()) * int(newThreadPV.size() > 2)
          > thread_voting_value(bestThread) * int(bestThreadPV.size() > 2);

        if (bestThreadInProvenWin)
        {
            // Make sure we pick the shortest mate / TB conversion
            if (newThreadScore > bestThreadScore)
                bestThread = th.get();
        }
        else if (bestThreadInProvenLoss)
        {
            // Make sure we pick the shortest mated / TB conversion
            if (newThreadInProvenLoss && newThreadScore < bestThreadScore)
                bestThread = th.get();
        }
        else if (newThreadInProvenWin || newThreadInProvenLoss
                 || (!is_loss(newThreadScore)
                     && (newThreadMoveVote > bestThreadMoveVote
                         || (newThreadMoveVote == bestThreadMoveVote && betterVotingValue))))
            bestThread = th.get();
    }

    // Threadに対してworkerが得られるから、これはYaneuraOuWorker*なのでdynamic_castして返す。
    // ⚠ RTTIを有効化していないので単にstatic_castにする。
    //return dynamic_cast<YaneuraOuWorker*>(bestThread->worker.get());
    return static_cast<YaneuraOuWorker*>(bestThread->worker.get());
}

// -----------------------------------------------------------
// 📌 ここ以降はStockfishのsearch.cppを参考にしながら書く。 📌
// -----------------------------------------------------------


#if STOCKFISH
namespace Stockfish {
// 💡 このファイルの冒頭でnamespace YaneuraOu { と書いているのでコメントアウト。

// 将棋では、Tablebasesは用いない。
namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap& options,
	const Search::LimitsType& limits,
	Stockfish::Position& pos,
	Stockfish::Search::RootMove& rootMove,
	Value& v);

using namespace Search;
// 💡 冒頭で書いたのでコメントアウト。
#endif

namespace {

// -----------------------
//   探索用の評価値補整
// -----------------------

// 💡 あるnodeで生成した指し手にbonusを与えるために、そのnodeで生成した指し手を良い順に保存しておく配列のcapacity。
constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

int correction_value(const YaneuraOuWorker& w, const Position& pos, const Stack* const ss) {
    const Color us    = pos.side_to_move();
    const auto  m     = (ss - 1)->currentMove;
    const auto  pcv   = w.pawnCorrectionHistory[pawn_correction_history_index(pos)][us];
    const auto  micv  = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto  wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto  bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const auto  cntcv =
      m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                 : 0;

    return 8867 * pcv + 8136 * micv + 10757 * (wnpcv + bnpcv) + 7232 * cntcv;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
// correctionHistory の値を raw staticEval に加え、
// 評価値がテーブルベースの範囲に入らないことを保証する。
Value to_corrected_static_eval(const Value v, const int cv) {
	return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position&          pos,
                               Stack* const             ss,
                               Search::YaneuraOuWorker& workerThread,
                               const int                bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    static constexpr int nonPawnWeight = 165;

    workerThread.pawnCorrectionHistory[pawn_correction_history_index(pos)][us] << bonus;
    workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 153 / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us]
      << bonus * nonPawnWeight / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us]
      << bonus * nonPawnWeight / 128;

    if (m.is_ok())
        (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
          << bonus * 153 / 128;
}


// Add a small random component to draw evaluations to avoid 3-fold blindness
// 3回同一局面になる盲点（3-fold blindness）を回避するため、評価を引き分け方向に誘導する小さなランダム成分を追加する

// 💡 引き分け時の評価値VALUE_DRAW(0)の代わりに±1の乱数みたいなのを与える。
//     nodes : 現在の探索node数(乱数のseed代わりに用いる)

// 📝 チェスでは、引き分けが0.5勝扱いなので引き分け回避のための工夫がしてあって、
//     以下のようにvalue_drawに揺らぎを加算することによって探索を固定化しない(同じnodeを
//     探索しつづけて千日手にしてしまうのを回避)工夫がある。
//     将棋の場合、普通の千日手と連続王手の千日手と劣等局面による千日手(循環？)とかあるのでこれ導入するのちょっと嫌。
//     ⇨  TODO : もうちょっとどうにかする。
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply /*, int r50c */);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(const Position&                pos,
                             Stack*                         ss,
                             Search::YaneuraOuWorker& workerThread,
                             Move                           move,
                             int                            bonus);

// 📝 32は、quietsSearched、quietsSearchedの最大数。そのnodeで生成したQUIETS/CAPTURESの指し手を良い順に保持してある。
//     bonusを加点するときにこれらの指し手に対して行う。
//     Stockfish 16ではquietsSearchedの配列サイズが[64]から[32]になった。
//     将棋ではハズレのquietの指し手が大量にあるので、それがベストとは限らない。
// 📊 比較したところ、64より32の方がわずかに良かったので、とりあえず32にしておく。(V7.73mとV7.73m2との比較)

void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::YaneuraOuWorker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            TTMove);


}  // namespace

// 💡 やねうら王では、Workerを派生させて書くことにしたので、このコードは、派生classであるYaneuraOuWorkerのコンストラクタで書く。
#if STOCKFISH
Search::Worker::Worker(SharedState& sharedState,
	std::unique_ptr<ISearchManager> sm,
	size_t                          threadId,
	NumaReplicatedAccessToken       token) :
	// Unpack the SharedState struct into member variables
	threadIdx(threadId),
	numaAccessToken(token),
	manager(std::move(sm)),
	options(sharedState.options),
	threads(sharedState.threads),
	tt(sharedState.tt),
	networks(sharedState.networks),
	refreshTable(networks[token]) {
	clear();
}
#endif

Search::YaneuraOuWorker::YaneuraOuWorker(OptionsMap&               options,
                                         ThreadPool&               threads,
                                         size_t                    threadIdx,
                                         NumaReplicatedAccessToken numaAccessToken,
										 TranspositionTable&       tt,
										 YaneuraOuEngine&          engine) :
    Search::Worker(options, threads, threadIdx, numaAccessToken), tt(tt),
		engine(engine), manager(engine.manager) {

    //clear();

	// 🤔 ThreadPool::resize_thread()→ThreadPool::set()でThreadPool::clear()が呼び出されて、
	//     そのなかでWorker::clear()が呼び出されるから、ここで呼び出す必要はないと思う。
}

void Search::YaneuraOuWorker::ensure_network_replicated() {
	// TODO : あとで評価関数を実装する。
	#if 0
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
	#endif
}

void Search::YaneuraOuWorker::pre_start_searching() {

	if (is_mainthread())
        // 🌈 Stockfishでthread.cppにあった初期化の一部はSearchManager::pre_start_searching()に移動させた。
        main_manager()->pre_start_searching(*this);

    // 📝 StockfishではThreadPool::start_thinking()で行っているが、
    //     やねうら王では、派生classのpre_start_thinking()以降で行う。
    nmpMinPly       = 0;
    bestMoveChanges = 0;
    rootDepth = completedDepth = 0;

    // 各WorkerのPosition::set_ekr()を呼び出して入玉ルールを反映させる必要がある。
    auto& search_options = main_manager()->search_options;
    rootPos.set_ekr(search_options.enteringKingRule);

	// 🌈 入玉宣言ができるならrootMovesに追加する。
	//    これは、main threadでだけ行えば良い。(main threadに属するWorkerのrootMovesにさえ追加されていれば良いので)
	if (is_mainthread())
    {
        // 🌈  宣言勝ちできるなら、rootMovesに追加する。
		// ⚠  ↑のset_erk()をしたあとでないとPosition::DeclarationWin()が正常に機能しない。
        auto bestMove = rootPos.DeclarationWin();
        if (bestMove != Move::none())
        {
            // 🤔 searchmovesが指定されていて
            //     そこに宣言勝ちがない時に宣言勝ちはできるのか…？
            //     できないと不便な気は少しするから、
            //     宣言勝ちできるならばつねにMove::win()を追加しておく。

            auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
            if (it_move == rootMoves.end()) // 追加されていない
                rootMoves.emplace_back(Move::win());
        }
    }
}

void Search::YaneuraOuWorker::start_searching() {

#if defined(USE_SFNN)
    // 探索の初回evaluate()では局面の差分更新ができないので
    // accumulatorの初回フラグをセットする。
    accumulatorStack.reset();
#endif

    // Non-main threads go directly to iterative_deepening()
    // メインスレッド以外は直接 iterative_deepening() へ進む

    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    // 📌 ここ以下のコードは、main threadで"go"に対して実行される。
    //     "go"のごとに初期化しないといけないものはここで行う。

    // 📌 今回の思考時間の設定。
    //     これは、ponderhitした時にponderhitにパラメーターが付随していれば
    //     再計算するする必要性があるので、いずれにせよ呼び出しておく必要がある。
    // 💡 やねうら王では、originalTimeAdjustは用いない。

#if STOCKFISH
    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
#else
    auto& mainManager = *main_manager();

    mainManager.tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                        mainManager.search_options.max_moves_to_draw);
#endif

    // 📌 置換表のTTEntryの世代を進める。
    // 📝 sub threadが動く前であるこのタイミングで置換表の世代を進めるべきである。
    //     cf. Call TT.new_search() earlier. : https://github.com/official-stockfish/Stockfish/commit/ebc563059c5fc103ca6d79edb04bb6d5f182eaf5

    // 置換表の世代カウンターを進める(クリアではない)
    tt.new_search();

#if STOCKFISH
#else
    // 🌈 やねうら王固有の初期化 🌈

    auto& search_options = mainManager.search_options;

    // PVが詰まるのを抑制するために、前回出力時刻を記録しておく。
    search_options.lastPvInfoTime = limits.startTime;

    // PVの出力間隔[ms]
    // go infiniteはShogiGUIなどの検討モードで動作させていると考えられるので
    // この場合は、PVを毎回出力しないと読み筋が出力されないことがある。
    search_options.computed_pv_interval =
      (limits.infinite || search_options.consideration_mode) ? 0 : search_options.pv_interval;

    // 🌈 引き分けのスコア

    // 引き分け時の値として現在の手番に応じた値を設定してやる。
    Color us         = rootPos.side_to_move();
    int   draw_value = (int) ((us == BLACK ? options["DrawValueBlack"] : options["DrawValueWhite"])
                            * Eval::PawnValue / 100);

    // 探索のleaf nodeでは、相手番(root_color != side_to_move)である場合、 +draw_valueではなく、-draw_valueを設定してやらないと非対称な探索となって良くない。
    // 例) 自分は引き分けを勝ち扱いだと思って探索しているなら、相手は、引き分けを負けとみなしてくれないと非対称になる。
    drawValueTable[REPETITION_DRAW][us]  = +draw_value;
    drawValueTable[REPETITION_DRAW][~us] = -draw_value;

    // 今回、通常探索をしたかのフラグ
    // このフラグがtrueなら(定跡にhitしたり1手詰めを発見したりしたので)探索をスキップした。
    bool search_skipped = true;

    // 🌈 ponder用の指し手の初期化
    //    やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのPV上の(相手の)指し手を用いるという独自仕様。
    //     Stockfish本家もこうするべきだと思う。
    mainManager.ponder_candidate = Move::none();

    // 探索せずに指し手を返すときの指し手
    Book::ProbeResult probeResult;

#if defined(SHOGI24)
    // ---------------------
    //    将棋倶楽部24対策
    // ---------------------

    // 相手玉が取れるなら取る。
    //
    // 相手玉が取れる局面は、(直前で王手放置があったということだから)非合法局面で、
    // 将棋所ではそのような局面からの対局開始はできないが、ShogiGUIでは対局開始できる。
    //
    // また、将棋倶楽部24で王手放置の局面を作ることができるので、
    // 相手玉が取れることがある。
    //
    // ゆえに、取れるなら取る指し手を指せたほうが良い。
    //
    // 参考動画 : https://www.youtube.com/watch?v=8nwJcKH0x0c

    auto their_king = rootPos.square<KING>(~us);
    auto our_piece  = rootPos.attackers_to(their_king) & rootPos.pieces(us);
    // 敵玉に利いている自駒があるなら、それを移動させて勝ち。
    if (our_piece)
    {
        Square from = our_piece.pop();
        Square to   = their_king;
        Move16 m16  = make_move16(from, to);
        Move   m    = rootPos.to_move(m16);

        // 玉を取る指し手はcapturesで生成されていない可能性がある。
        // 仕方がないので、rootMoves[0]を書き換えることにする。

        // 玉で玉を取る手はrootMovesに含まれないので、場合によっては、それしか指し手がない場合に、
        // rootMoves.size() == 0だけど、玉で玉を取る指し手だけがあることは起こり得る。
        // (この理由から、玉を取る判定は、合法手がない判定より先にしなければならない)

        probeMove.bestmove  = m;
        probeMove.bestscore = mate_in(1);
        goto SKIP_SEARCH;
    }
#endif

    // ✋ 独自追加ここまで。
#endif

    // ---------------------
    // 合法手がないならここで投了
    // ---------------------

    if (rootMoves.empty())
    {
        // rootで指し手がない = (将棋だと)詰みの局面である

        probeResult.bestmove  = Move::resign();
        probeResult.bestscore = mated_in(1);

		// 💡 このあとrootMoves[0]にアクセスして、アクセス違反になるのを防ぐため。
        rootMoves.emplace_back(Move::none());

#if STOCKFISH
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
        // 💡 チェスだと王手されていないなら引き分けだが、将棋だとつねに負け。
#else
		// やねうら王では、このあと、probeResultを用いる時用にPVを出力するので、そこで行う。
#endif

        goto SKIP_SEARCH;
    }

    // ---------------------
    //    宣言勝ち判定
    // ---------------------

    {
        // 宣言勝ちならその指し手を選択。
        // 王手がかかっていても、回避しながらトライすることもあるので王手がかかっていようが
        // Position::DeclarationWin()で判定して良い。
        // 1手詰めは、ここでは判定しない。
        // (MultiPVのときに1手詰めを見つけたからと言って探索を終了したくないから。)

        auto bestMove = rootPos.DeclarationWin();
        if (bestMove != Move::none())
        {
            probeResult.bestmove  = bestMove;
            probeResult.bestscore = mate_in(1);
            goto SKIP_SEARCH;
        }
    }

    // ---------------------
    //     定跡の選択部
    // ---------------------

    probeResult = engine.book.probe(rootPos, main_manager()->updates);
    if (probeResult.bestmove)
        goto SKIP_SEARCH;

    // ---------------------
    //    通常の思考処理
    // ---------------------

    threads.start_searching();  // start non-main threads
    // 📝 main以外のすべてのthreadを開始する。
    //    main以外のthreadがstart_searching()を開始する。
    //    start_searching()の先頭には、main thread以外であれば即座に
    //    iterative_deepning()を呼び出すようになっているので、これにより並列探索が開始できる。

    iterative_deepening();  // main thread start searching
    // 💡 main threadも並列探索に加わる。

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.

    // 最大深さに到達したとき、threads.stop が発生せずにここに到達することがある。
    // しかし、ポンダリング中や無限探索中の場合、UCIプロトコルでは
    // GUI が "stop" または "ponderhit" コマンドを送るまで
    // best move を出力すべきではないとされている。
    // したがって、ここでは単純に GUI からこれらのコマンドが送られてくるのを待つ。

    // 📝 最大depth深さに到達したときに、ここまで実行が到達するが、
    //     まだthreads.stopが生じていない。しかし、ponder中や、go infiniteによる探索の場合、
    //     USI(UCI)プロトコルでは、"stop"や"ponderhit"コマンドをGUIから送られてくるまでbest moveを出力してはならない。
    //     それゆえ、単にここでGUIからそれらのいずれかのコマンドが送られてくるまで待つ。
    //
    //     1. "stop"が送られてきたらThreads.stop == trueになる。
    //     2. "ponderhit"が送られてきたらThreads.ponder == falseになる
    //       ので、それを待つ。
    //
    //     ちなみにStockfishのほう、ここのコードに長らく同期上のバグがあった。
    //     やねうら王のほうは、かなり早くからこの構造で書いていた。後にStockfishがこの書き方に追随した。

    // 普通に探索したのでskipしたかのフラグをfalseにする。
    // 💡やねうら王独自
    search_skipped = false;

SKIP_SEARCH:

    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {
        // Busy wait for a stop or a ponder reset
        // stop か ponder reset を待つ間のビジーウェイト

        //	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
        // (思考のためには計算資源を使っていないので。)
        Tools::sleep(1);
        // ⚠ Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。


        /*
			📓 ここでPVを出力したほうがいいかも？

				ponder中/go infinite中であっても、ここに抜けてきている以上、
				全探索スレッドの停止が確認できた時点でPVは出力したほうが良いと思う。

				"go infinite"の場合、詰みを発見してもそれがponderフラグの解除を待ってからだと、
				PVを返すのが遅れる。("stop"が来るまで返せない)

				// 　ここですべての探索スレッドが停止しているならば最終PVを出力してやる。
				if (!output_final_pv_done
					&& Threads.search_finished()) // 全探索スレッドが探索を完了している
					output_final_pv();
		*/
    }

    // Stop the threads if not already stopped (also raise the stop if
    // "ponderhit" just reset threads.ponder)
    // まだ停止していなければスレッドを停止する（"ponderhit" により threads.ponder が
    // リセットされた場合も stop を発生させる）
    threads.stop = true;

    // Wait until all threads have finished
    // すべてのスレッドが終了するのを待つ
    // 💡 開始していなければいないで構わない。

    threads.wait_for_search_finished();

// 💡 やねうら王では、npmsecをサポートしない。
#if STOCKFISH
    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    // 'nodes as time' モードでプレイしている場合、終了する前に
    // 使用可能なノード数から探索済みノード数を差し引く。

    // 📝 'nodes as time'モードとは、時間としてnodesを用いるモード
    //     時間切れの場合、負の数になりうる。
    // ⚠ 将棋の場合、秒読みがあるので秒読みも考慮しないといけない。
    //     Time.availableNodes += Limits.inc[us] + Limits.byoyomi[us] - Threads.nodes_searched();
    //     みたいな処理が必要か？将棋と相性が良くないのでこの機能、無効化する。

    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);
#endif

    // 📌 指し手をGUIに返す 📌

    // Lazy SMPの結果を取り出す

#if STOCKFISH
    // 並列探索したうちのbestな結果を保持しているthread
    // まずthisを入れておいて、定跡を返す時などはthisのままにするコードを適用する。
    Worker* bestThread = this;

    Skill skill =
      Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
    // TODO : Skillの導入はあとで検討する。
    //  🤔  それにしてもオプションが3つも増えるの嫌だな…。

#else
    YaneuraOuWorker* bestThread = this;

    Skill skill = Skill(20, 0);
#endif

    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
        && rootMoves[0].pv[0] != Move::none())
#if STOCKFISH
        bestThread = threads.get_best_thread()->worker.get();
#else
        // 💡 やねうら王では、get_best_thread()は、ThreadPoolからこのclassに移動させた。
        bestThread = get_best_thread();
#endif

    // 次回の探索のときに何らか使えるのでベストな指し手の評価値を保存しておく。
    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;
#if !STOCKFISH
	// 次回に手番が今回と異なるかを検出するためにgame_ply()を保存しておく。
    main_manager()->lastGamePly = rootPos.game_ply();
#endif

#if STOCKFISH
    // Send again PV info if we have a new best thread
    // 新しいベストスレッドがあれば、再度PV情報を送信する
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);

    // 🤔 こんなにPV出力するの好きじゃないので省略。
    //     ただし、一度もPVを出力していないなら、出力すべきだと思う。

#else
    // この時点で一度もPVを出力していないなら出力する。
    // 💡 一度も出力していない場合、lastPvInfoTimeは、"go"された時刻であるstartTimeになっている。
    if (search_options.lastPvInfoTime == limits.startTime)
    {
        if (search_skipped)
        {
            // search_skippedのときは、自前でPVを構築する。
            // 💡 このとき、rootMovesの情報を使わないようにしたい。

            InfoFull info;
            info.depth     = 0;
            info.selDepth  = 0;
            info.multiPV   = 1;
            info.score     = probeResult.bestscore;
            TimePoint time = std::max(TimePoint(1), mainManager.tm.elapsed_time());
            info.timeMs    = time;
            info.nodes     = 0;
            info.nps       = 0;
            std::string pv = probeResult.bestmove.to_usi_string();
            if (probeResult.pondermove)
                pv += " " + probeResult.pondermove.to_usi_string();
            info.pv       = pv;
            info.hashfull = tt.hashfull();
            mainManager.updates.onUpdateFull(info);
        }
        else
        {
            main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);
        }
    }

    // 🌈 投了スコアが設定されていて、歩の価値を100として正規化した値がそれを下回るなら投了。
    //    ただし定跡の指し手にhitした場合などはrootMoves[0].score == -VALUE_INFINITEになっているのでそれは除外。
    auto resign_value = (int) options["ResignValue"];
    if (bestThread->rootMoves[0].score != -VALUE_INFINITE
        && USIEngine::to_cp(bestThread->rootMoves[0].score) <= -resign_value)
    {
        // 探索がskipされた扱いにして、resignを積む。
        search_skipped = true;
        probeResult    = Book::ProbeResult(Move::resign());
    };

#endif

    // デバッグ用に(ギリギリまで思考できているかを確認するために)経過時間を出力してみる。
    /*
    auto& tm = main_manager()->tm;
    sync_cout << "info string elapsed time           = " << tm.elapsed_time() << "\n"
              << "info string elapsed_from_ponderhit = " << now() - tm.ponderhitTime << sync_endl;
	*/

    std::string ponder;

#if STOCKFISH
    if (bestThread->rootMoves[0].pv.size() > 1
        || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
#else

	std::string bestmove;
    if (search_skipped)
    {
        bestmove = probeResult.bestmove.to_usi_string();
        if (probeResult.pondermove)
	        ponder   = probeResult.pondermove.to_usi_string();
	}
    else
    {
		// 🌈 extract_ponder_from_tt()に
		//     ponder_candidateを渡して、ponderの指し手をひねり出す。
        if (bestThread->rootMoves[0].pv.size() > 1
            || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos,
                                                               main_manager()->ponder_candidate))
            ponder = USIEngine::move(bestThread->rootMoves[0].pv[1]);

        bestmove = USIEngine::move(bestThread->rootMoves[0].pv[0]);
    }

	/*
		📓
			USIプロトコルにおいて、bestmoveの時に返すponderは、常に返して問題ない。

			エンジンオプションの"USI_Ponder"がtrueであれば、GUIは次に相手の手番で"go ponder"を送信して
			思考エンジンにponderで思考させる。

			つまり、"USI_Ponder"は思考エンジンのためではなく、GUI側のためのオプションである。
	*/

#endif

    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has been
// consumed, the user stops the search, or the maximum search depth is reached.

// メインの反復深化ループ。search() を繰り返し呼び出し、
// 設定された思考時間を使い切るか、ユーザーが探索を停止するか、
// 最大探索深度に到達するまで処理を続ける。

// 📝 探索本体。並列化している場合、ここが各threadの探索のentry point。
//     Lazy SMPなので、置換表を共有しながらそれぞれのスレッドが勝手に探索しているだけ。

void Search::YaneuraOuWorker::iterative_deepening() {

    // もし自分がメインスレッドであるならmainThreadにmain_managerのポインタを代入。
    // 自分がサブスレッドのときは、これはnullptrになる。
    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

#if STOCKFISH
#else
    // やねうら王では探索オプションは、main_managerが持っている。
    SearchOptions& search_options = main_manager()->search_options;

    // 各WorkerのPosition::set_ekr()を呼び出して入玉ルールを反映させる必要がある。
    rootPos.set_ekr(search_options.enteringKingRule);
#endif

    Move pv[MAX_PLY + 1];

    // 探索の安定性を評価するために前回のiteration時のbest PVを記録しておく。
    Depth lastBestMoveDepth = 0;
    Value lastBestScore     = -VALUE_INFINITE;
    auto  lastBestPV        = std::vector{Move::none()};

    // alpha,beta         : aspiration searchの窓の範囲(alpha,beta)
    // delta              : apritation searchで窓を動かす大きさdelta
    // us                 : この局面の手番側
    // timeReduction      : 読み筋が安定しているときに時間を短縮するための係数。
    // totBestMoveChanges : 直近でbestMoveが変化した回数の統計。読み筋の安定度の目安にする。
    // iterIdx            : 反復深化の時に1回ごとのbest valueを保存するための配列へのindex。0から3までの値をとる。
    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int    delta, iterIdx                        = 0;

    // 📝 Stockfish 14の頃は、反復深化のiterationが浅いうちはaspiration searchを使わず
    //     探索窓を (-VALUE_INFINITE , +VALUE_INFINITE)としていたが、Stockfish 16では、
    //     浅いうちからaspiration searchを使うようになったので、alpha,betaの初期化はここでやらなくなった。

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.

    // (ss-7)から(ss+2)へのアクセスを許可するために、追加のサイズでスタックを割り当てます
    // (ss-7)はupdate_continuation_histories(ss-1, ...)のために必要であり、これは(ss-6)にアクセスします
    // (ss+2)はstatScoreとkillersの初期化のために必要です

    // 💡 continuationHistoryのため、(ss-7)から(ss+2)までにアクセスしたいので余分に確保しておく。

    // 📝 stackは、先頭10個を初期化しておけば十分。
    //     そのあとはsearch()の先頭でss+1,ss+2を適宜初期化していく。
    //     RootNodeはss->ply == 0がその条件。(ss->plyはsearch_thread_initで初期化されている)

    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    // counterMovesをnullptrに初期化するのではなくNO_PIECEのときの値を番兵として用いる。
    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel

        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    // Stack(探索用の構造体)上のply(手数)は事前に初期化しておけば探索時に代入する必要がない。
    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    // 最善応手列(Principal Variation)
    ss->pv = pv;

    if (mainThread)
    {
#if !STOCKFISH
		// 🌈 Stochastic Ponderでは前回と別手番になるので、このとき、
		//     bestPreviousScoreとpreviousAverageScoreを反転させる必要がある。
		if ((mainThread->lastGamePly - rootPos.game_ply()) & 1)
		{
            if (mainThread->bestPreviousScore != VALUE_INFINITE)
	            mainThread->bestPreviousScore *= -1;

            if (mainThread->bestPreviousAverageScore != VALUE_INFINITE)
	            mainThread->bestPreviousAverageScore *= -1;
		}
#endif

        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    // MultiPV
    // 💡 bestmoveとしてしこの局面の上位N個を探索する機能

    size_t multiPV = size_t(options["MultiPV"]);

#if STOCKFISH
    Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
    // 🤔 ↑これでエンジンオプション2つも増えるのやだな…。気が向いたらサポートすることにする。
#else
    Skill skill = Skill(20, 0);
#endif

    // When playing with strength handicap enable MultiPV search that we will
    // use behind-the-scenes to retrieve a set of possible moves.

    // 強さハンディキャップ付きでプレイするときは、MultiPV探索を有効にする。
    // これにより、裏側で複数の候補手を取得できるようにする。

    // 📝 SkillLevelが有効(設定された値が20未満)のときは、MultiPV = 4で探索。

    if (skill.enabled())
        multiPV = std::max(multiPV, size_t(4));

    // 💡 multiPVの値は、この局面での合法手の数を上回ってはならない。
    multiPV = std::min(multiPV, rootMoves.size());

    // ---------------------
    //   反復深化のループ
    // ---------------------

    // 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
    // これが増えていない時、同じ深さを再度探索していることになる。(fail highし続けている)
    // 💡 あまり同じ深さでつっかえている時は、aspiration windowの幅を大きくしてやるなどして回避する必要がある。
    int searchAgainCounter = 0;

    // 💡 lowPlyHistoryは、試合開始時に1回だけではなく、"go"の度に初期化したほうが強い。
    lowPlyHistory.fill(89);

    // Iterative deepening loop until requested to stop or the target depth is reached
    // 要求があるか、または目標深度に達するまで反復深化ループを実行します

    // 📝 rootDepthはこのthreadの反復深化での探索中の深さ。
    //     limits.depth("go"コマンドの時に"depth"として指定された探索深さ)が指定されている時は、
    //     main threadのrootDepthがそれを超えた時点でこのループを抜ける。
    //     (main threadが抜けるとthreads.stop == trueになるのでそのあとsub threadは勝手にこのループを抜ける)

    while (++rootDepth < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth > limits.depth))
    {
        /*
		📓 Stockfish9にはslave threadをmain threadより先行させるコードがここにあったが、
			Stockfish10で廃止された。

			これにより短い時間(低いrootDepth)では探索効率が悪化して弱くなった。
			これは、rootDepthが小さいときはhelper threadがほとんど探索に寄与しないためである。
			しかしrootDepthが高くなってきたときには事情が異なっていて、main threadよりdepth + 3とかで
			調べているhelper threadがあったとしても、探索が打ち切られる直前においては、
			それはmain threadの探索に寄与しているとは言い難いため、無駄になる。

			折衷案として、rootDepthが低い時にhelper threadをmain threadより先行させる(高いdepthにする)
			コード自体は入れたほうがいいかも知れない。
		*/

        // ------------------------
        // Lazy SMPのための初期化
        // ------------------------

        // Age out PV variability metric
        // PV変動メトリックを古く(期限切れに)する

        // 📝 bestMoveが変化した回数を記録しているが、反復深化の世代が一つ進むので、
        //     古い世代の情報として重みを低くしていく。

        if (mainThread)
            totBestMoveChanges /= 2;

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.

        // 最初のPVラインを探索する前に前回イテレーションのスコアを保存し、
        // （新しい）PV 以外のすべての手のスコアを -VALUE_INFINITE に設定する。

        // 💡 aspiration window searchのために反復深化の前回のiterationのスコアをコピーしておく

        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        // 🤔 将棋ではこれ使わなくていいような？

        //size_t pvFirst = 0;
        //pvLast         = 0;

        // 💡 探索深さを増やすかのフラグがfalseなら、同じ深さを探索したことになるので、
        //     searchAgainCounterカウンターを1増やす
#if STOCKFISH
        if (!threads.increaseDepth)
#else
        if (!main_manager()->increaseDepth)
#endif
            searchAgainCounter++;

        // MultiPV loop. We perform a full root search for each PV line
        // MultiPVのloop。MultiPVのためにこの局面の候補手をN個選出する。
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            // 📝 chessではtbRankの処理が必要らしい。将棋では関係なさげなのでコメントアウト。
            //     tbRankが同じ値のところまでしかsortしなくて良いらしい。
            //     (そこ以降は、明らかに悪い指し手なので)

#if STOCKFISH
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }
#else

            // 🤔 将棋だとtbRankは常に同じとみなせるので、
            //     pvLastはrootMoves.size()になるまで
            //     インクリメントされるから、次のように単純化できる。

            size_t pvFirst = pvIdx;
            pvLast         = rootMoves.size();

#endif

            // Reset UCI info selDepth for each depth and each PV line
            // それぞれのdepthとPV lineに対するUSI infoで出力するselDepth
            selDepth = 0;

            // ------------------------
            // Aspiration window search
            // ------------------------

            /*
				📓 探索窓を狭めてこの範囲で探索して、この窓の範囲のscoreが返ってきたらラッキー、みたいな探索。

					探索が浅いときは (-VALUE_INFINITE,+VALUE_INFINITE)の範囲で探索する。
					探索深さが一定以上あるなら前回の反復深化のiteration時の最小値と最大値
					より少し幅を広げたぐらいの探索窓をデフォルトとする。

					Reset aspiration window starting size
					aspiration windowの開始サイズをリセットする

					aspiration windowの幅
					精度の良い評価関数ならばこの幅を小さくすると探索効率が上がるのだが、
					精度の悪い評価関数だとこの幅を小さくしすぎると再探索が増えて探索効率が低下する。
					やねうら王のKPP評価関数では35～40ぐらいがベスト。
					やねうら王のKPPT(Apery WCSC26)ではStockfishのまま(18付近)がベスト。
					もっと精度の高い評価関数を用意すべき。
					この値はStockfish10では20に変更された。
					Stockfish 12(NNUEを導入した)では17に変更された。
					Stockfish 12.1では16に変更された。
					Stockfish 16では10に変更された。
					Stockfish 17では 5に変更された。

				💡 将棋ではStockfishより少し高めが良さそう。
			*/

            // Reset aspiration window starting size
            // aspiration windowの開始サイズをリセットする。
            delta     = 5 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 11131;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);

#if 0
            // Adjust optimism based on root move's averageScore
            // ルート手の averageScore に基づいて楽観度を調整する
            optimism[us]  = 136 * avg / (std::abs(avg) + 93);
            optimism[~us] = -optimism[us];
#endif

			// 🤔 このoptimismは、StockfishのNNUE評価関数で何やら使っているようなのだが…。
            //     TODO : あとで検討する。

            // Start with a small aspiration window and, in the case of a fail
            // high/low, re-search with a bigger window until we don't fail
            // high/low anymore.

            // 小さなaspiration windowで開始して、fail high/lowのときに、fail high/lowにならないようになるまで
            // 大きなwindowで再探索する。

            // fail highした回数
            // 💡 fail highした回数分だけ探索depthを下げてやるほうが強いらしい。
            int failedHighCnt = 0;

            while (true)
            {
                // Adjust the effective depth searched, but ensure at least one
                // effective increment for every four searchAgain steps (see issue #2717).

                // 実際に探索した深さを調整するが、
                // searchAgain ステップ4回ごとに少なくとも1回は有効な増分があるようにする
                // （issue #2717 を参照）。

                // fail highするごとにdepthを下げていく処理
                Depth adjustedDepth =
                  std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.

                // 最善手を先頭に持ってくる。これは非常に重要であり、
                // 安定ソートアルゴリズムを使う必要がある。
                // なぜなら、最初の手および新しい最善手以外のすべての値は
                // -VALUE_INFINITE に設定されるため、
                // 新しいPVが先頭に来る以外は、すべての手の順序を維持したいからである。
                // MultiPV探索の場合、すでに探索済みのPVラインは保持される点に注意。

                // 📝 それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
                //    一つ目の指し手以外は-VALUE_INFINITEが返る仕様なので並べ替えのために安定ソートを
                //    用いないと前回の反復深化の結果によって得た並び順を変えてしまうことになるのでまずい。

                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.

                // 探索が停止されている場合は直ちに break する。
                // RootMoves は前回のイテレーションを参照しているが依然として有効なので、
                // ソートは安全である。

                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs like Fritz 19, only start
                // at nodes > 10M (rather than depth N, which can be reached quickly)

                // fail high / lowの時には、再探索前に何らかの情報を出力する。
                // ただし、Fritz 19 などの GUI がフリーズするのを避けるため、
                // 深さ N ではなく、10M ノードを超えてから出力を始める。

                // 💡 将棋所のコンソールが詰まるのを予防するために出力を少し抑制する。
                //    また、go infiniteのときは、検討モードから使用しているわけで、PVは必ず出力する。

                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > 10000000

#if !SOTCKFISH
                    // 🌈 以下やねうら王独自拡張
                    && (rootDepth < 3
                        || search_options.lastPvInfoTime + search_options.computed_pv_interval
                             <= now())
                    // outout_fail_lh_pvがfalseならfail high/fail lowのときのPVを出力しない。
                    && search_options.outout_fail_lh_pv
#endif
                )
#if STOCKFISH
                    main_manager()->pv(*this, threads, tt, rootDepth);
#else
                {
                    main_manager()->pv(*this, threads, tt, rootDepth);
                    // 最後にPVを出力した時刻を格納しておく。
                    search_options.lastPvInfoTime = now();
                }
#endif

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = alpha;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    beta = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += delta / 3;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Sort the PV lines searched so far and update the GUI
            // これまでに探索したPVラインをソートし、GUIを更新する

            // 💡 MultiPVの候補手をスコア順に再度並び替えておく。
            //    (二番目だと思っていたほうの指し手のほうが評価値が良い可能性があるので…)

            std::stable_sort(rootMoves.begin() /* + pvFirst */, rootMoves.begin() + pvIdx + 1);

            if (mainThread
                && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000)
                // A thread that aborted search can have mated-in/TB-loss PV and
                // score that cannot be trusted, i.e. it can be delayed or refuted
                // if we would have had time to fully search other root-moves. Thus
                // we suppress this output and below pick a proven score/PV for this
                // thread (from the previous iteration).

                // 探索を中断したスレッドは、詰み直前のPVやTB損失のPVおよび
                // 信頼できないスコアを持つ可能性がある。
                // つまり、他のルートムーブを完全に探索する時間があれば、
                // 遅延したり反証されたりするかもしれない。
                // したがって、この出力を抑制し、以下でこのスレッドに対して
                // （前回の反復から）証明済みのスコア／PVを選択する。

                && !(threads.abortedSearch && is_loss(rootMoves[0].uciScore))
#if !STOCKFISH
                // PVの出力間隔を超えている。
                && search_options.lastPvInfoTime + search_options.computed_pv_interval <= now()
#endif
            )
#if STOCKFISH
                main_manager()->pv(*this, threads, tt, rootDepth);
#else
            {
                main_manager()->pv(*this, threads, tt, rootDepth);
                // 最後にPVを出力した時刻を格納しておく。
                search_options.lastPvInfoTime = now();
            }
#endif


            if (threads.stop)
                break;

        }  // multi pv loop

        if (!threads.stop)
            completedDepth = rootDepth;

        // We make sure not to pick an unproven mated-in score,
        // in case this thread prematurely stopped search (aborted-search).

        // このスレッドが探索を早期に停止した（中断探索）場合に備えて、
        // 証明されていない詰みスコアを選ばないように注意している。

        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score))
        {
            // Bring the last best move to the front for best thread selection.
            // 最後に得られた最善手を先頭に移動し、最適なスレッド選択のために備える。
            // 💡 move_to_front()は、見つけたものを先頭に移動させ、元の先頭からそこまでは1つ後方にずらす。

            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](
                                                const auto& rm) { return rm == lastBestPV[0]; });
            rootMoves[0].pv    = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;
        }
        else if (rootMoves[0].pv[0] != lastBestPV[0])
        {
            lastBestPV        = rootMoves[0].pv;
            lastBestScore     = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }

        // 🤔 探索深さが、mateの手数の2倍以上になったら、それより短い詰みが
        //     見つかることは稀だし、探索を停止させて良いと思う。(やねうら王独自)
        // 💡 multi_pvのときは一つのpvで詰みを見つけただけでは停止するのは良くないので
        //     早期終了はmultiPV == 1のときのみ行なう。

        if (multiPV == 1)
        {
            // 勝ちを読みきっているのに将棋所の表示が追いつかずに、将棋所がフリーズしていて、その間の時間ロスで
            // 時間切れで負けることがある。
            // mateを読みきったとき、そのmateの2.5倍以上、iterationを回しても仕方ない気がするので探索を打ち切るようにする。
            // ⇨ あと、rootで1手詰め呼び出さなくしたので、その影響もあって、VALUE_MATE == bestValueになることはあるから、この時、
            //   rootDepth == 1で探索を終了されては困る。もう少し先まで調べて欲しいので、+2しておく。
            if (!limits.mate && bestValue >= VALUE_MATE_IN_MAX_PLY
                && (VALUE_MATE - bestValue + 2) * 5 / 2 < (Value) (rootDepth))
                break;

            // 詰まされる形についても同様。こちらはmateの2.5倍以上、iterationを回したなら探索を打ち切る。
            if (!limits.mate && bestValue <= VALUE_MATED_IN_MAX_PLY
                && (bestValue - (-VALUE_MATE) + 2) * 5 / 2 < (Value) (rootDepth))
                break;
        }

        if (!mainThread)
            continue;

        // 🌈 ponder用の指し手として、2手目の指し手を保存しておく。
        //     これがmain threadのものだけでいいかどうかはよくわからないが。
        //     とりあえず、無いよりマシだろう。(やねうら王独自拡張)

        if (rootMoves[0].pv.size() > 1)
            mainThread->ponder_candidate = rootMoves[0].pv[1];

        // Have we found a "mate in x"?
        // x手詰めを発見したのか？

        // 💡 UCIでは"go mate 5"のようにmateのあと手数が送られてくる仕様。
        //     USIでは"go mate"のあとは思考時間がやってくるので、早期リタイアできない。
#if STOCKFISH
        if (limits.mate && rootMoves[0].score == rootMoves[0].usiScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;
#endif

        // If the skill level is enabled and time is up, pick a sub-optimal best move
        // スキルレベルが有効で、かつ時間切れの場合、最適でないベストムーブを選ぶ。

        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Use part of the gained time from a previous stable move for the current move
        // 直前の安定した手で得た時間の一部を現在の手に使う。

        for (auto&& th : threads)
        {
            auto yw = toYaneuraOuWorker(th->worker);
            totBestMoveChanges += yw->bestMoveChanges;
            yw->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        // 次の反復を行う時間はあるか？今すぐ探索を止められるか？
#if STOCKFISH
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
#else
        // 📝 やねうら王の場合、search_endが設定されている時はもう終了が確定しているので
        //     この終了チェックを行うのは無駄である。
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit
            && !mainThread->tm.search_end)
#endif
        {
            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (11.396 + 2.035 * (mainThread->bestPreviousAverageScore - bestValue)
               + 0.968 * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5786, 1.6752);

            // If the bestMove is stable over several iterations, reduce time accordingly
            // bestMove が複数回のイテレーションで安定している場合、それに応じて時間を短縮する

            double k      = 0.527;
            double center = lastBestMoveDepth + 11;
            timeReduction = 0.8 + 0.84 / (1.077 + std::exp(-k * (completedDepth - center)));
            double reduction =
              (1.4540 + mainThread->previousTimeReduction) / (2.1593 * timeReduction);
            double bestMoveInstability = 0.9929 + 1.8519 * totBestMoveChanges / threads.size();

            double totalTime =
              mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;

            // Cap used time in case of a single legal move for a better viewer experience
            // 視聴者体験を向上させるため、合法手が1つだけの場合に使用時間を上限で制限する

            if (rootMoves.size() == 1)
                totalTime = std::min(500.0, totalTime);
	            // 🤔 やねうら王ではここ0でも良いような…？

#if STOCKFISH
            auto elapsedTime = elapsed();
#else
            // 📝 やねうら王では、MainManagerのTimer classのほうから経過時間をもらう。
            auto elapsedTime = mainThread->tm.elapsed_time();
#endif

            // ⚠ やねうら王では、このへん仕組みが異なる。
            //     秒単位で切り上げたいので、tm.search_endまで
            //     持ち時間を使い切りたい。

            if (completedDepth >= 10 && nodesEffort >= 97056 && elapsedTime > totalTime * 0.6540
                && !mainThread->ponder)
#if STOCKFISH
                threads.stop = true;
#else
                mainThread->tm.set_search_end(elapsedTime);
#endif

            // Stop the search if we have exceeded the totalTime or maximum
            // totalTime または maximum を超えた場合、探索を停止する

#if STOCKFISH
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                // 停止条件を満たした

                // 📝 将棋の場合、フィッシャールールではないのでこの時点でも最小思考時間分だけは
                //     思考を継続したほうが得なので、思考自体は継続して、キリの良い時間になったら
                //     check_time()にて停止する。

                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".

                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.5138;

#else
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    // 停止条件を満たしてはいるのだが、秒の切り上げは行う。
                    mainThread->tm.set_search_end(elapsedTime);
            }
            else
                main_manager()->increaseDepth =
                  mainThread->ponder || elapsedTime <= totalTime * 0.5138;
#endif
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return;

    mainThread->previousTimeReduction = timeReduction;

    // If the skill level is enabled, swap the best PV line with the sub-optimal one
    if (skill.enabled())
        std::swap(rootMoves[0],
                  *std::find(rootMoves.begin(), rootMoves.end(),
                             skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));
}

void Search::YaneuraOuWorker::do_move(Position & pos, const Move move, StateInfo& st,
                                        Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), ss);
}

void YaneuraOuWorker::do_move(Position&  pos,
                            const Move move,
                            StateInfo& st,
                              const bool givesCheck,
							  Stack* const ss) {

    bool capture = pos.capture_stage(move);

#if defined(EVAL_SFNN)

    // accumulatorStackを用いる実装。

    DirtyPiece dp = pos.do_move(move, st, givesCheck , &tt);
    nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);

#else

	pos.do_move(move, st, givesCheck, &tt);
    nodes.fetch_add(1, std::memory_order_relaxed);

#endif

    if (ss != nullptr)
    {
		// currentMove(現在探索中の指し手)の更新
        ss->currentMove         = move;

#if STOCKFISH
        ss->continuationHistory = &continuationHistory[ss->inCheck][capture][dp.pc][move.to_sq()];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[dp.pc][move.to_sq()];
#else
		// やねうら王とStockfishでは、DirtyPieceの構造が違うし、
		// やねうら王では、移動後の駒(成りの指し手なら成り駒)を使いたい。
        Piece dp_pc = pos.moved_piece(move);
        ss->continuationHistory =
          &continuationHistory[ss->inCheck][capture][dp_pc][move.to_sq()];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[dp_pc][move.to_sq()];
#endif
    }
}

void YaneuraOuWorker::do_null_move(Position& pos, StateInfo& st) { pos.do_null_move(st, tt); }

void YaneuraOuWorker::undo_move(Position& pos, const Move move) {

	pos.undo_move(move);

#if defined(EVAL_SFNN)
    //accumulatorStack.pop();
#endif
}

void YaneuraOuWorker::undo_null_move(Position& pos) { pos.undo_null_move(); }


// Reset histories, usually before a new game
// 履歴をリセットする。通常は新しいゲームの前に実行される。
void YaneuraOuWorker::clear() {

    mainHistory.fill(64);
    captureHistory.fill(-753);
    pawnHistory.fill(-1275);
    pawnCorrectionHistory.fill(5);
    minorPieceCorrectionHistory.fill(0);
    nonPawnCorrectionHistory.fill(0);

	ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(8);

    //     ほとんどの履歴エントリがいずれにせよ後で負になるため、
    //     開始値を「正しい」方向に少しシフトさせるため、負の数で埋めている。
    //     この効果は、深度が深くなるほど薄れるので、長時間思考させる時には
    //     あまり意味がないが、無駄ではないらしい。
    //     cf. Tweak history initialization : https://github.com/official-stockfish/Stockfish/commit/7d44b43b3ceb2eebc756709432a0e291f885a1d2

	for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : continuationHistory[inCheck][c])
                for (auto& h : to)
                    h.fill(-494);

	// reductions tableの初期化(これはWorkerごとが持つように変更された)
    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(2782 / 128.0 * std::log(i));

	// 📝 lowPlyHistoryの初期化は、対局ごとではなく、局面ごと("go"のごと)に変更された。

#if defined(EVAL_SFNN)
    refreshTable.clear(networks[numaAccessToken]);
#endif
}

// -----------------------
//      通常探索
// -----------------------

// 💡  search()は、最初、iterative_deepening()のなかから呼び出される。
//     これは仮想関数にはなっていないので、仮想関数呼び出しのoverheadはない。

// Main search function for both PV and non-PV nodes
// PV , non-PV node共用のメインの探索関数。

// cutNode : LMRで悪そうな指し手に対してreduction量を増やすnode

template<NodeType nodeType>
Value YaneuraOuWorker::search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
{
    // -----------------------
    //     nodeの種類
    // -----------------------

	// PV nodeであるか
	// 💡 root nodeは必ずPV nodeに含まれる。

    constexpr bool PvNode   = nodeType != NonPV;

	// root nodeであるか
    constexpr bool rootNode = nodeType == Root;

	// allNodeであるか。
    // 📝 allNodeとは、PvNodeでもなくcutNodeでもないnodeのこと。
    //     allNodeとは、ゲーム木探索で、全ての子ノードを評価する必要があるノードのこと。

    const bool     allNode  = !(PvNode || cutNode);

    // Dive into quiescence search when the depth reaches zero
    // 残り探索深さが1手未満であるなら現在の局面のまま静止探索を呼び出す
    if (depth <= 0)
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return qsearch<nt>(pos, ss, alpha, beta);
    }

    // Limit the depth if extensions made it too large
    // 拡張によって深さが大きくなりすぎた場合、深さを制限します

    depth = std::min(depth, MAX_PLY - 1);

	// 📝 次の指し手で引き分けに持ち込めてかつ、betaが引き分けのスコアより低いなら
    //     早期枝刈りが実施できる。
    // 🤔 将棋だとあまり千日手が起こらないので効果がなさげ。採用しない。

#if STOCKFISH
    // Check if we have an upcoming move that draws by repetition
    // 直近の手が繰り返しによる引き分けになるかを確認します

	if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }
#endif

	ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    ASSERT_LV3(PvNode || (alpha == beta - 1));
    ASSERT_LV3(0 < depth && depth < MAX_PLY);
    // IIDを含め、PvNodeではcutNodeで呼び出さない。
    ASSERT_LV3(!(PvNode && cutNode));

	// -----------------------
    //     変数宣言
    // -----------------------

	// pv : このnodeからのPV line(読み筋)
    // st : do_move()するときに必要

	Move      pv[MAX_PLY + 1];
	StateInfo st;

	// posKey       : このnodeのhash key

	Key        posKey;

    // move			: MovePickerから1手ずつもらうときの一時変数
    // excludedMove	: singular extemsionのときに除外する指し手
    // bestMove		: このnodeのbest move

	Move  move, excludedMove, bestMove;

	// extension	: 延長する深さ
	// newDepth		: 新しいdepth(残り探索深さ)

	Depth extension, newDepth;

    // bestValue	: このnodeのbestな探索スコア
    // value		: 探索スコアを受け取る一時変数
    // eval			: このnodeの静的評価値(の見積り)
    // maxValue     : table base probeに用いる。📌 将棋だと用いない。
    // probCutBeta  : prob cutする時のbetaの値。
#if STOCKFISH
	Value bestValue, value, eval, maxValue, probCutBeta;
#else
	Value bestValue, value, eval, probCutBeta;
#endif
	// givesCheck			: moveによって王手になるのか
	// improving			: 直前のnodeから評価値が上がってきているのか
	//   このフラグを各種枝刈りのmarginの決定に用いる
	//   cf. Tweak probcut margin with 'improving' flag : https://github.com/official-stockfish/Stockfish/commit/c5f6bd517c68e16c3ead7892e1d83a6b1bb89b69
	//   cf. Use evaluation trend to adjust futility margin : https://github.com/official-stockfish/Stockfish/commit/65c3bb8586eba11277f8297ef0f55c121772d82c
	// priorCapture         : 1つ前の局面は駒を取る指し手か？
	// opponentWorsening    : 相手の状況が悪化しているかのフラグ

	bool  givesCheck, improving, priorCapture, opponentWorsening;

	// capture              : moveが駒を捕獲する指し手もしくは歩を成る手であるか
    // ttCapture			: 置換表の指し手がcaptureする指し手であるか

	bool  capture, ttCapture;

	// priorReduction       : 1手前の局面でのreductionの量
	// movedPiece           :  moveによって移動させる駒

	int   priorReduction;
    Piece movedPiece;

	// capturesSearched : このnodeで生成した、MovePickerから取得した駒を捕獲する指し手を順番に格納する  (+歩の成り)
    // quietsSearched   : このnodeで生成した、MovePickerから取得した駒を捕獲しない指し手を順番に格納する(-歩の成り)

    SearchedList capturesSearched;
    SearchedList quietsSearched;

	// -----------------------
    // Step 1. Initialize node
	// Step 1. ノードの初期化
    // -----------------------

	//     nodeの初期化

    ss->inCheck        = pos.checkers();
    priorCapture       = pos.captured_piece();
    Color us           = pos.side_to_move();
    ss->moveCount      = 0;
    bestValue          = -VALUE_INFINITE;
#if STOCKFISH
    maxValue           =  VALUE_INFINITE;
    // 📝 将棋ではtable probe使っていないのでmaxValueは関係ない。
#else
    // やねうら王探索で追加した思考エンジンオプション
    auto& search_options = main_manager()->search_options;
#endif

#if defined(USE_CLASSIC_EVAL) && defined(USE_LAZY_EVALUATE)
	// 📝 次のnodeに行くまでにevaluate()かevaluate_with_no_return()を呼び出すことを保証して
	//     evaluate内の差分計算を促さなければならない。
    bool evaluated = false;
    auto evaluate  = [&](Position& pos) {
        evaluated = true;
        return this->evaluate(pos);
    };
    auto do_move = [&](Position & pos, Move move, StateInfo st, bool givesCheck, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, givesCheck, ss);
    };

	// 🤔 同じ名前で呼び分けできないので、
	//     こちらを名前を do_move_ にする。
    auto do_move_ = [&](Position & pos, Move move, StateInfo st, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, ss);
    };
    auto do_null_move = [&](Position& pos, StateInfo st) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_null_move(pos, st);
    };
#else
    auto do_move_ = [&](Position& pos, Move move, StateInfo st, Stack* ss) { this->do_move(pos, move, st, ss); };
#endif

	// 📌 Timerの監視

    // Check for the available remaining time
    // 残りの利用可能な時間を確認します
    // 💡 これはメインスレッドのみが行なう。
	if (is_mainthread())
        main_manager()->check_time(*this);


    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
	// selDepth情報をGUIに送信するために使用します（selDepthは1からカウントし、plyは0からカウントします）
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

	// -----------------------
    //  RootNode以外での処理
    // -----------------------

    if (!rootNode)
    {
        // -----------------------
        // Step 2. Check for aborted search and immediate draw
        // Step 2. 中断された探索および即時引き分けを確認します
        // -----------------------

		// 探索の中断と、引き分けについてチェックする
        // 連続王手による千日手、および通常の千日手、優等局面・劣等局面。

        // 連続王手による千日手に対してdraw_value()は、詰みのスコアを返すので、rootからの手数を考慮したスコアに変換する必要がある。
        // そこで、value_from_tt()で変換してから返すのが正解。

        // 教師局面生成時には、これをオフにしたほうが良いかも知れない。
        // ただし、そのときであっても連続王手の千日手の判定は有効にしておく。

        // →　優等局面・劣等局面はrootより遡って判定しない。
        // (USIで出力する評価値がVALUE_SUPERIORになるのはちょっと嫌だし、
        // 　優等局面に突入するからと言って即詰みを逃がすのもちょっと嫌)
        // cf. https://github.com/yaneurao/YaneuraOu/issues/264

		// 最大手数を超えている、もしくは千日手、停止命令が来ている。

#if STOCKFISH
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos)
                                                        : value_draw(nodes);
#else

		auto draw_type = pos.is_repetition(ss->ply);
        if (draw_type != REPETITION_NONE)
        { 
            if (draw_type == REPETITION_DRAW)
				// 通常の千日手の時はゆらぎを持たせる。
				// 💡 引き分けのスコアvは abs(v±1) <= VALUE_MAX_EVALであることが保証されているので、
				//     value_from_tt()での変換は不要。
                return draw_value(draw_type, pos.side_to_move()) + value_draw(nodes);
            else
	            return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);
        }

		// 📌 将棋では手数を超えたら無条件で引き分け扱い。
        if (threads.stop.load(std::memory_order_relaxed) || ss->ply >= MAX_PLY
            || pos.game_ply() > search_options.max_moves_to_draw)
            return draw_value(REPETITION_DRAW, pos.side_to_move()) + value_draw(nodes);
#endif

		/*
		📝 備考

			256手ルールで
			1. 256手目の局面で判定を行う場合は、
				「詰まされていない、かつ、連続王手の千日手が成立していない」ならば、
				引き分けとしてreturnして良い。

			2. 257手目の局面で判定を行う場合は、
				この局面に到達したということは、256手目の局面で合法手があったということだから、
				引き分けとしてreturnして良い。

			ということになる。1.の方式でやったほうが、256手目の局面で指し手生成とか
			1手ずつ試していくのとかが丸ごと端折れるので探索効率は良いのだが、コードが複雑になる。

			2.の方式でやっていいならば、257手目の局面なら単にreturnするだけで済む。
			探索効率は少し悪いが、コードは極めてシンプルになる。

			また、256手ルールで256手目である指し手を指して、257手目の局面で連続王手の千日手が
			成立するときはこれは非合法手扱いをすべきだと思う。

			だから、2.の方式で判定するときは、この連続王手の千日手判定を先にやってから、
			257手目の局面であるかの判定を行う必要がある。

			上記のコードは、そうなっている。
		*/

		// -----------------------
		// Step 3. Mate distance pruning.
		// Step 3. 詰みまでの手数による枝刈り
		// -----------------------

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Same logic but with reversed
        // signs apply also in the opposite condition of being mated instead of giving
        // mate. In this case, return a fail-high score.

		// ステップ3. 詰みまでの手数による枝刈り。たとえ次の手でメイトしても、スコアは最大で
        // mate_in(ss->ply + 1)となります。しかし、もしalphaがすでにそれ以上であれば、
        // ツリー上でより短い詰みが見つかったことを意味し、これ以上探索する必要はありません。
        // 現在のalphaを上回ることはできないからです。
        // 逆に、詰みされる場合も同様のロジックが適用されますが、符号が逆になります。
        // この場合、fail highを返します。

		/*
		   📝 備考
        
			   rootから5手目の局面だとして、このnodeのスコアが5手以内で
			   詰ますときのスコアを上回ることもないし、
			   5手以内で詰まさせるときのスコアを下回ることもない。
			   そこで、alpha , betaの値をまずこの範囲に補正したあと、
			   alphaがbeta値を超えているならbeta cutする。
		*/

        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

	// -----------------------
    //  探索Stackの初期化
    // -----------------------

	// rootからの手数
    ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

	// 前の指し手で移動させた先の升目
    // null moveのときはis_ok() == falseなのでSQ_NONEとする。
    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore       = 0;
    (ss + 2)->cutoffCnt = 0;

	// -----------------------
    // Step 4. Transposition table lookup.
    // Step 4. 置換表の参照
    // -----------------------

	// このnodeで探索から除外する指し手。ss->excludedMoveのコピー。
    excludedMove                   = ss->excludedMove;

	/*
		📝
			excludedMoveがある(singular extension時)は、
			前回の全探索の置換表の値を上書きする部分探索のスコアは
			欲しくないので、excluded moveがある場合には異なるhash keyを用いて
			異なるTTEntryを読み書きすべきだと思うが、
			Stockfish 16で、同じTTEntryを用いるようになった。
			(ただしexcluded moveがある時に探索した結果はTTEntryにsaveしない)
			つまり、probeして情報だけ利用する感じのようだ。情報は使えるということなのだろうか…。

			posKey = excludedMove == Move::none() ? pos.hash_key() : pos.hash_key() ^ HASH_KEY(make_key(excludedMove));
			↑このときpos.key()のbit0を破壊することは許されないので、make_key()でbit0はクリアしておく。
			excludedMoveがMove::none()の時はkeyを変更してはならない。

			↓Stockfish 16で異なるTTEntryを使わないようになって次のように単純化された。
			   cf. https://github.com/official-stockfish/Stockfish/commit/8d3457a9966f8c744ab7f8536be408196ccd8af9
	*/

	/*
		📓
			excluded moveについて詳しく。

			singular extensionとは、置換表から拾ってきた指し手だけがすこぶるよろしい指し手である時、
			一本道の変化だから、この指し手はもっと延長してあげようということである。駒のただ捨てなどで
			指し手を引き伸ばすような水平線効果を抑える役割もある。(たぶん)

			だから、置換表の指し手を除外して同じnodeで探索しなおす必要がある。
			この時の探索における置換表に元あった指し手をexcluded moveと呼ぶ。

			つまり、この時の探索結果は、excluded moveを除外して得られた探索結果なので、
			同じTTEntry(置換表のエントリー)に書き出すのはおかしいわけである。

			だからexcluded moveがある時は、局面のhash keyを、このexcluded moveを
			考慮したhash keyに変更して別のTTEntryを用いるようにしていた。

			そのコードが上の pos.hash_key() ^ HASH_KEY(make_key(excludedMove) の部分である。
			(make_keyはexcludedMoveをseedとする疑似乱数を生成する)

			ところが、これをStockfishの上のcommitは、廃止するというのである。

			メリットとしては、make_keyで全然違うTTEntryを見に行くとCPUのcacheにmiss hitするので、
			そこで遅くなるのだが、同じTTEntryを見に行くなら、間違いなくCPU cacheにhitするというものである。
			また、元エントリーの値のうち、staticEval(evaluate()した時の値)ぐらいは使えるんじゃね？ということである。

			デメリットとしては、この時の探索結果をそのTTEntryに保存してしまうとそれはexcluded moveがない時の
			探索結果としては正しくないので、このような保存はできないということである。
			それにより、次回も同じexcluded moveでsingular extensionする時に今回の探索結果が活用できない
			というのはある。

			そのどちらが得なのかということのようである。
	*/

    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);

    // Need further processing of the saved data
    // 保存されていたデータのさらなる処理が必要です

	ss->ttHit    = ttHit;

	/*
		📝
			置換表の指し手
			置換表にhitしなければMove::none()
			RootNodeであるなら、(MultiPVなどでも)現在注目している1手だけがベストの指し手と仮定できるから、
			それが置換表にあったものとして指し手を進める。
	
		⚠
	
		    TTにMove::win()も指し手として書き出す場合は、
		    tte->move()にはMove::win()も含まれている可能性がある。
		    この時、pos.to_move(MOVE_WIN) == Move::win()なので、ttMove == Move::win()となる。
		    ⇨ 現状のやねうら王では、Move::win()はTTに書き出さない。
	*/

	ttData.move  = rootNode ? rootMoves[pvIdx].pv[0]
                 : ttHit    ? ttData.move
                            : Move::none();

	// 置換表上のスコア
    // 💡 置換表にhitしなければVALUE_NONE

    // singular searchとIIDとのスレッド競合を考慮して、ttValue , ttMoveの順で取り出さないといけないらしい。
    // cf. More robust interaction of singular search and iid : https://github.com/official-stockfish/Stockfish/commit/16b31bb249ccb9f4f625001f9772799d286e2f04

	ttData.value =
          ttHit ? value_from_tt(ttData.value, ss->ply /*, pos.rule50_count()*/) : VALUE_NONE;

	// 📝 置換表の指し手にpseudo_legalではない指し手が混じっていたら、
	//     それは先後の局面を間違えた置換表Entryに書き出してしまっているバグ。
	ASSERT_LV3(pos.legal_promote(ttData.move));

    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);

	/*
		置換表の指し手がcaptureであるか。
		置換表の指し手がcaptureなら高い確率でこの指し手がベストなので、他の指し手を
		そんなに読まなくても大丈夫。なので、このnodeのすべての指し手のreductionを増やす。

		ここ、capture_or_promotion()とかcapture_or_pawn_promotion()とか色々変えてみたが、
		現在では、capture()にするのが良いようだ。[2022/04/13]
		→　捕獲する指し手で一番小さい価値上昇は歩の捕獲(+ 2*PAWN_VALUE)なのでこれぐらいの差になるもの
			歩の成り、香の成り、桂の成り　ぐらいは調べても良さそうな…。
		→ Stockfishでcapture_stage()になっているところはそれに倣うことにした。[2023/11/05]
	*/

    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // At this point, if excluded, skip straight to step 6, static eval. However,
    // to save indentation, we list the condition in all code between here and there.
	// この時点で、除外された場合は、ステップ6の静的評価に直接進みます。
    // しかし、インデントを減らすために、ここからそこまでのコード内の条件をすべて記載しています。

	// 📝 補足
    //
    //    置換表にhitしなかった時は、PV nodeのときだけttPvとして扱う。
    //    これss->ttPVに保存してるけど、singularの判定等でsearchをss+1ではなくssで呼び出すことがあり、
    //    そのときにss->ttPvが破壊される。なので、破壊しそうなときは直前にローカル変数に保存するコードが書いてある。

    // At non-PV nodes we check for an early TT cutoff
    // 非PVノードでは、早期のTTカットオフを確認する

	// 📝  PV nodeでは置換表の指し手では枝刈りしない。
	//      PV nodeはごくわずかしかないし、重要な変化だから枝刈りしたくない。

	if (!PvNode
		&& !excludedMove
		&& ttData.depth > depth - (ttData.value <= beta) // 置換表に登録されている探索深さのほうが深くて
        && is_valid(ttData.value)   // Can happen when !ttHit or when access race in probe()
							        // !ttHitの場合やprobe()でのアクセス競合時に発生する可能性がありうる。
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5))
    /*
		📝 解説
		
			ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
			ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
			今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
			このままこの値でreturnして良い。
	*/
    {
        // If ttMove is quiet, update move sorting heuristics on TT hit
        // ttMoveがquietの指し手である場合、置換表ヒット時に指し手のソート用ヒューリスティクスを更新します。

        if (ttData.move && ttData.value >= beta)
        {
			/*
				📝 備考
		
					置換表の指し手でbeta cutが起きたのであれば、この指し手をkiller等に登録する。
					捕獲する指し手か成る指し手であればこれは(captureで生成する指し手なので)killerを更新する価値はない。
		
					ただし置換表の指し手には、hash衝突によりpseudo-leaglでない指し手である可能性がある。
					update_quiet_stats()で、この指し手の移動元の駒を取得してCounter Moveとするが、
					それがこの局面の手番側の駒ではないことがあるのでゆえにここでpseudo_legalのチェックをして、
					Counter Moveに先手の指し手として後手の指し手が登録されるような事態を回避している。
					その時に行われる誤ったβcut(枝刈り)は許容できる。(non PVで生じることなのでそこまで探索に対して悪い影響がない)
					cf. https://yaneuraou.yaneu.com/2021/08/17/about-the-yaneuraou-bug-that-appeared-in-the-long-match/

					やねうら王ではttMoveがMove::win()であることはありうるので注意が必要。
					is_ok(m)==falseの時、Position::to_move(m)がmをそのまま帰すことは保証されている。
					そのためttMoveがMove::win()でありうる。これはstatのupdateをされると困るのでis_ok()で弾く必要がある。
					is_ok()は、ttMove == Move::none()の時はfalseなのでこの条件を省略できる。
					⇨　Move::win()書き出すの、筋が良くないし、入玉自体が超レアケースなので棋力に影響しないし、これやめることにする。

					If ttMove is quiet, update move sorting heuristics on TT hit (~2 Elo)
					置換表にhitした時に、ttMoveがquietの指し手であるなら、指し手並び替えheuristics(quiet_statsのこと)を更新する。
			*/

            // Bonus for a quiet ttMove that fails high
            // fail highしたquietなquietな(駒を取らない)ttMove(置換表の指し手)に対するボーナス

            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                       std::min(127 * depth - 74, 1063));

            // Extra penalty for early quiet moves of the previous ply
            // 1手前の早い時点のquietの指し手に対する追加のペナルティ

			// 💡 1手前がMove::null()であることを考慮する必要がある。

			if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 3 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2128);
        }

        // Partial workaround for the graph history interaction problem
        // For high rule50 counts don't produce transposition table cutoffs.

		// グラフ履歴の相互作用問題に対する部分的な回避策
        // rule50カウントが高い場合は、トランスポジションテーブルによるカットオフを行いません。

        // ⚠ 将棋では関係のないルールなので無視して良いが、pos.rule50_count < 90 が(チェスの)通常の状態なので、
        //     if成立時のreturnはしなければならない。

        // 🤔 比較してみたが、これを有効にするとR10ぐらい弱くなるっぽい。
        //     全体を調整してからまた考える。
#if 0
		if (pos.rule50_count() < 91)
        {
            // TODO : 将棋でもこの処理必要なのか？

            if (depth >= 8 && ttData.move
#if STOCKFISH
				&& pos.pseudo_legal(ttData.move)
#else
                && pos.pseudo_legal(ttData.move, search_options.generate_all_legal_moves)
#endif
				&& pos.legal(ttData.move)
                && !is_decisive(ttData.value))
            {
#if STOCKFISH
                pos.do_move(ttData.move, st);
                Key nextPosKey                             = pos.key();
#else
                pos.do_move(ttData.move, st);
                HASH_KEY nextPosKey                        = pos.hash_key();
#endif
				auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey, pos);
                pos.undo_move(ttData.move);

                // Check that the ttValue after the tt move would also trigger a cutoff
                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            }
            else
                return ttData.value;
        }
#else
        return ttData.value;
#endif        
    }

	// -----------------------
    //     宣言勝ち
    // -----------------------

    // Step 5. Tablebases probe
    // ⚠ StockfishのStep 5.のコードはtablebase(終盤データベース)で将棋には関係ないので割愛。

#if STOCKFISH
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore   wdl = Tablebases::probe_wdl(pos, &err);

            // Force check of time on the next occasion
            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                tbHits.fetch_add(1, std::memory_order_relaxed);

                int drawScore = tbConfig.useRule50 ? 1 : 0;

                Value tbValue = VALUE_TB - ss->ply;

                // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to score
                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                   std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                   tt.generation());

                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }
#else

	// これは将棋にはないが、将棋には代わりに宣言勝ちというのがある。
    // 宣言勝ちと1手詰めだと1手詰めの方が圧倒的に多いので、まず1手詰め判定を行う。

    // 🌈 以下は、やねうら王独自のコード 🌈

    // -----------------------
    //    1手詰みか？
    // -----------------------

	/*
		📝 ttHitしている時は、すでに1手詰め・宣言勝ちの判定は完了しているはずなので行わない。
			excludedMoveがある時はttHitしてttMoveがあるはずだが、置換表壊されるケースがあるので
			念のためチェックはしないといけない。

			!rootnodeではなく!PvNodeの方がいいかも？
			(PvNodeでは置換表の指し手を信用してはいけないという方針なら)

			excludedMoveがある時には本当は、それを除外して詰み探索をする必要があるが、
			詰みがある場合は、singular extensionの判定の前までにbeta cutするので、結局、
			詰みがあるのにexcludedMoveが設定されているということはありえない。
			よって、「excludedMoveは設定されていない」時だけ詰みがあるかどうかを調べれば良く、
			この条件を詰み探索の事前条件に追加することができる。
		
			ただし、excludedMoveがある時、singular extensionの事前条件を満たすはずで、
			singular extensionはttMoveが存在することがその条件に含まれるから、
			ss->ttHit == trueになっているはず。
		
			RootNodeでは1手詰め判定、ややこしくなるのでやらない。(RootMovesの入れ替え等が発生するので)
			置換表にhitしたときも1手詰め判定はすでに行われていると思われるのでこの場合もはしょる。
			depthの残りがある程度ないと、1手詰めはどうせこのあとすぐに見つけてしまうわけで1手詰めを
			見つけたときのリターン(見返り)が少ない。
			ただ、静止探索で入れている以上、depth == 1でも1手詰めを判定したほうがよさげではある。
	*/

	if (!rootNode && !ttHit
		// TODO : この条件必要なのか？
        && !excludedMove
    )
    {
        if (!ss->inCheck)
        {
            move = Mate::mate_1ply(pos);

            if (move != Move::none())
            {
                /*
					🤔 1手詰めスコアなので確実にvalue > alphaなはず。
					    1手詰めは次のnodeで詰むという解釈

					⚠ このとき、bestValueをvalue_to_tt()で変換してはならない。
					    mate_in()はrootからの計算された詰みのスコアであるから、このまま置換表に格納してよい。
                
                		あるいは、VALUE_MATE - 1をvalue_to_tt()で変換したものを置換表に格納するかだが、
					    その場合、returnで返す値を用意するのに再度変換が必要になるのでそういう書き方は良くない。
				*/

                bestValue = mate_in(ss->ply + 1);

                ASSERT_LV3(pos.legal_promote(move));
                if (!excludedMove)
                    ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT,
                                    std::min(MAX_PLY - 1, depth + 6), move, VALUE_NONE,
                                    tt.generation());

				// ⚠ excludedMoveがあるときは置換表に書き出さないルールになっているので、
                //     この↑の条件式が必要なので注意。

				/*
				   📝 
                　	 【計測資料 39.】 mate1plyの指し手を見つけた時に置換表の指し手でbeta
					  cutする時と同じ処理をする。

                      兄弟局面でこのmateの指し手がよい指し手ある可能性があるので ここでttMoveでbeta
                      cutする時と同様の処理を行うと短い時間ではわずかに強くなるっぽいのだが
                      長い時間で計測できる差ではなかったので削除。

					💡
						1手詰めを発見した時に、save()でdepthをどのように設定すべきか問題について。

						即詰みは絶対であり、MAX_PLYの深さで探索した時の結果と同じであるから、
						以前はMAX_PLYにしていたのだが、よく考えたら、即詰みがあるなら上位ノードで
						枝刈りが発生してこのノードにはほぼ再訪問しないと考えられるのでこんなものが
						置換表に残っている価値に乏しく、また、MAX_PLYにしてしまうと、
						TTEntryのreplacement strategy上、depthが大きなTTEntryはかなりの優先度になり
						いつまでもreplacementされない。

						こんな情報、lostしたところで1手詰めならmate1ply()で一手も進めずに得られる情報であり、
						最優先にreplaceすべきTTEntryにも関わらずである。

						かと言ってDEPTH_NONEにするとtt->depth()が 0 になってしまい、枝刈りがされなくなる。
						そこで、depth + 6 ぐらいがベストであるようだ。
				*/

				return bestValue;
            }
        }
	}

	// -----------------------
    //      宣言勝ちか？
    // -----------------------

	// 置換表にhitしていないときは宣言勝ちの判定をまだやっていないということなので今回やる。
    // PvNodeでは置換表の指し手を信用してはいけないので毎回やる。
    if (!ttData.move || PvNode)
    {
        // 💡 王手がかかってようがかかってまいが、宣言勝ちの判定は正しい。
        //     (トライルールのとき王手を回避しながら入玉することはありうるので)
        //     トライルールのときここで返ってくるのは16bitのmoveだが、置換表に格納するには問題ない。
        move = pos.DeclarationWin();
        if (move != Move::none())
        {
            bestValue = mate_in(
                ss->ply + 1);  // 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈

            ASSERT_LV3(pos.legal_promote(move));
            /*
				if (!excludedMove)
					ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT,
						std::min(MAX_PLY - 1, depth + 6),
						is_ok(move) ? move : MOVE_NONE, // MOVE_WINはMOVE_NONEとして書き出す。
						VALUE_NONE, TT.generation());
			*/

			/*
				📝  is_ok(m)は、MOVE_WINではない通常の指し手(トライルールの時の51玉のような指し手)は
					 その指手を置換表に書き出すという処理。
            
					 probe()でttData.move == MOVE_WINのケースを完全に考慮するのは非常に難しい。
             　		 MOVE_WINの値は、置換表に書き出さないほうがいいと思う。
            
					 また、置換表にわざと書き込まないことによって、次にこのnodeに訪問したときに
					 再度入玉判定を行い、枝刈りされることを期待している。
			*/

            return bestValue;
        }
        // 🤔 1手詰めと宣言勝ちがなかったのでこの時点でもsave()したほうがいいような気がしなくもない。
    }

#endif // STOCKFISH

	// -----------------------
    // Step 6. Static evaluation of the position
    // Step 6. 局面の静的な評価
    // -----------------------

    Value      unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue      = correction_value(*this, pos, ss);

	if (ss->inCheck)
    {
        // Skip early pruning when in check
        // 王手がかかっているときは、early pruning(早期枝刈り)をスキップする

		ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
        goto moves_loop;
    }
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    /*
		📝  excludedMoveがあるときは、この局面の情報をTTに保存してはならない。
			 (同一局面で異なるexcludedMoveを持つ局面が同じhashkeyを持つので情報の一貫性がなくなる。)
			 ⇨ 異なるexcludedMoveに対して異なるhashkeyを持てばいいのだが
			   例: auto posKey = pos.hash_key() ^ make_key(excludedMove);
			 (以前のStockfishのバージョンではそのようにしていた)
			  現在のコードのほうが強いようで、そのコードは取り除かれた。
	*/
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        // TTに格納されている値に関して何も仮定はしない

        // 💡 置換表にhitしたなら、評価値が記録されているはずだから、それを取り出しておく。
        //     あとで置換表に書き込むときにこの値を使えるし、各種枝刈りはこの評価値をベースに行なうから。

		unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

#if !STOCKFISH
#if defined(YANEURAOU_ENGINE_NNUE)
#if defined(USE_LAZY_EVALUATE)
        // 🌈 これ書かないとR70ぐらい弱くなる。
        else if (PvNode)
        {
			unadjustedStaticEval = evaluate(pos);

			/*
				 🤔 : ここでevaluate() が必須な理由がよくわからない。
				       Stockfishには無いのに…。なぜなのか…。

				lazy_evaluate()に変える                        ⇨ 弱い(V8.38dev-n7)
				unadjustedStaticEvalに代入せずに単にevaluate() ⇨ 特に弱くはなさそう(V838dev_n8)
				結論的には、NNUEのevaluate_with_no_return()とevaluate()の挙動が違うのが原因っぽい。
				これはあきませんわ…。
			*/
        }
#else
        // lazy evalを使わないなら、この時点でどうにかしておく。
        else
			unadjustedStaticEval = evaluate(pos);
#endif
#endif
#endif

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // ttValue can be used as a better position evaluation
        // ttValue は、より良い局面評価として使用できる

		/*
			📝 ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。

				1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
				  evalとしてttValueを採用して良い。

				2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
				  evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。
		*/

        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // Static evaluation is saved as it was before adjustment by correction history
        // 静的評価は、補正履歴による調整が行われる前の状態で保存される。

		/*
		  📝 excludedMoveがある時は、これを置換表に保存するのは危ない。
              cf . Add / remove leaves from search tree ttPv : https://github.com/official-stockfish/Stockfish/commit/c02b3a4c7a339d212d5c6f75b3b89c926d33a800
              上の方にある else if (excludedMove) でこの条件は除外されている。
		*/

        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation());

		// どうせ毎node評価関数を呼び出すので、evalの値にそんなに価値はないのだが、mate_1ply()を
        // 実行したという証にはなるので意味がある。
    }

	// -----------------------
    //   evalベースの枝刈り
    // -----------------------

    // Use static evaluation difference to improve quiet move ordering
    // 静的評価の差を利用して、静かな手の順序付けを改善します。

	/*
		📝 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
		    王手のときはここにはこない。(上のinCheckのなかでMOVES_LOOPに突入。)

			is_ok()はMove::null()かのチェック。
			1手前でMove::null()ではなく、王手がかかっておらず、駒を取る指し手ではなかったなら…。
	*/

    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        int bonus =
            std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -1979, 1561) + 630;
        mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 935 / 1024;

		// TODO : これ必要なのか？あとで検証する。
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << bonus * 1428 / 1024;
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.

	// improvingフラグを設定します。これは、現在の静的評価が前回の自分の手番での
    // 静的評価より大きい場合にtrueとなります（前回の手で王手を受けていた場合、
    // 王手を受けていない局面まで遡って評価します）。
    // それ以外の場合はfalseとなります。このimprovingフラグは、さまざまな枝刈り手法で使用されます。

	/*
		📝 improvingは、評価値が2手前の局面から上がって行っているのかのフラグ
		    上がって行っているなら枝刈りを甘くする。

		    VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
		    枝刈りを甘くして調べないといけないのでimproving扱いとする。

		💡 VALUE_NONE == 32002なのでこれより大きなstaticEvalの値であることはない。
	*/

    improving = ss->staticEval > (ss - 2)->staticEval;

	/*
		📝 opponentWorseningは、相手の状況が悪化しているかのフラグ。
	
		💡 ss->staticEval == - (ss-1)->staticEval であるのが普通だが、
		    左辺のほうが大きい(相手の評価値が悪化している)ならば、
		    相手の評価値が悪くなっていっていることを意味している。
	*/

    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

	// 1手前のreductionに応じた残りdepthの調整

    if (priorReduction >= (depth < 10 ? 1 : 3) && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 177)
        depth--;

	// -----------------------
    // Step 7. Razoring
    // -----------------------

    // If eval is really low, skip search entirely and return the qsearch value.
    // For PvNodes, we must have a guard against mates being returned.

	// 評価値が非常に低い場合、検索を完全にスキップして qsearch の値を返します。
    // PvNode では、チェックメイトが返されるのを防ぐためのガードが必要です。

    if (!PvNode && eval < alpha - 495 - 290 * depth * depth)
        return qsearch<NonPV>(pos, ss, alpha, beta);

	// -----------------------
    // Step 8. Futility pruning: child node
    // Step 8. Futility枝刈り : 子ノード
    // -----------------------

    // The depth condition is important for mate finding.
    // depthの条件は詰みを発見するために重要である。

	/*
		📝 このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
			evalからこれを引いてbetaより大きいなら、beta cutが出来る。

			ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
			ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。

			Stockfish9までは、futility pruningを、root node以外に適用していたが、
			Stockfish10でnonPVにのみの適用に変更になった。
	*/

    {
        // futility margin
        // 💡 depth(残り探索深さ)に応じたfutility margin。

		auto futility_margin = [&](Depth d) {
            Value futilityMult = 90 - 20 * (cutNode && !ss->ttHit);

            return futilityMult * d                      //
                 - improving * futilityMult * 2          //
                 - opponentWorsening * futilityMult / 3  //
                 + (ss - 1)->statScore / 356             //
                 + std::abs(correctionValue) / 171290;
        };

        if (!ss->ttPv && depth < 14 && eval - futility_margin(depth) >= beta && eval >= beta
            && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
            return beta + (eval - beta) / 3;
    }

	// -----------------------
    // Step 9. Null move search with verification search
    // Step 9. 検証探索を伴うnull move探索
    // -----------------------

    //  🖊 evalがbetaを超えているので1手パスしてもbetaは超えそう。だからnull moveを試す
    if (cutNode && ss->staticEval >= beta - 19 * depth + 403
	    && !excludedMove
#if STOCKFISH
        && pos.non_pawn_material(us)
    // 💡 盤上にpawn以外の駒がある ≒ pawnだけの終盤ではない。
    // 🤔 将棋でもこれに相当する条件が必要かも。
#endif
        && ss->ply >= nmpMinPly && !is_loss(beta)
        // 同じ手番側に連続してnull moveを適用しない
    )
    {
        ASSERT_LV3((ss - 1)->currentMove != Move::null());

        // Null move dynamic reduction based on depth
        // (残り探索)深さと評価値に基づくnull moveの動的なreduction

        Depth R = 7 + depth / 3;

        ss->currentMove                   = Move::null();
        ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

        // 💡  null moveなので、王手はかかっていなくて駒取りでもない。
        //     よって、continuationHistory[0(王手かかってない)][0(駒取りではない)][NO_PIECE][SQ_ZERO]
        //
        // 📃 王手がかかっている局面では ⇑の方にある goto moves_loop; によってそっちに行ってるので、
        //     ここでは現局面で手番側に王手がかかっていない = 直前の指し手(非手番側)は王手ではない ことがわかっている。
        //     do_null_move()は、この条件を満たす必要がある。

        do_null_move(pos, st);

        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        // 証明されていないmate scoreやTB scoreはreturnで返さない。

        if (nullValue >= beta && !is_win(nullValue))
        {
            // 1手パスしてもbetaを上回りそうであることがわかったので
            // これをもう少しちゃんと検証しなおす。

            if (nmpMinPly || depth < 16)
                return nullValue;

            ASSERT_LV3(!nmpMinPly);  // Recursive verification is not allowed
                                     // 再帰的な検証は認めていない。

            // Do verification search at high depths, with null move pruning disabled
            // until ply exceeds nmpMinPly.
            //
            // 💡 null move枝刈りを無効化して、plyがnmpMinPlyを超えるまで
            //     高いdepthで検証のための探索を行う。

            nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            // 📝 nullMoveせずに(現在のnodeと同じ手番で)同じ深さで探索しなおして本当にbetaを超えるか検証する。
            //     cutNodeにしない。

            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

	// ここでimproving計算しなおす。

    improving |= ss->staticEval >= beta;

	// -----------------------
    // Step 10. Internal iterative reductions
    // Step 10. 内部反復リダクション
    // -----------------------

	// At sufficient depth, reduce depth for PV/Cut nodes without a TTMove.
    // (*Scaler) Especially if they make IIR less aggressive.
    // 十分な探索深さがある場合、置換表（TTMove）に手がないPVノードやCutノードについては探索深さを削減する。
    //（*Scaler）特に、IIR のアグレッシブさが抑えられる場合に適用されます。

    if (!allNode && depth >= 6 && !ttData.move && priorReduction <= 3)
        depth--;

#if OLD_CODE
    // 🌈 以前のコードのほうが強い可能性がある。

	if (    PvNode
		&& !ttData.move)
		depth -= 3;

	// Use qsearch if depth <= 0
	// (depthをreductionした結果、)もしdepth <= 0ならqsearchを用いる

	if (depth <= 0)
		return qsearch<PV>(pos, ss, alpha, beta);

	// For cutNodes, if depth is high enough, decrease depth by 2 if there is no ttMove,
	// or by 1 if there is a ttMove with an upper bound.

	// カットノードの場合、深さが十分にある場合は、ttMoveがない場合に深さを2減らし、
	// ttMoveが上限値を持つ場合は深さを1減らします。

	if (cutNode && depth >= 7 && (!ttData.move || ttData.bound == BOUND_UPPER))
		depth -= 1 + !ttData.move;
#endif
        
	// -----------------------
    // Step 11. ProbCut
    // -----------------------

    // If we have a good enough capture (or queen promotion) and a reduced search
    // returns a value much above beta, we can (almost) safely prune the previous move.

	// 十分に良い駒取り（またはクイーン昇格）があり、
    // (残り探索深さを)削減された探索でbetaを大幅に上回る値が返される場合、
    // 直前の手を（ほぼ）安全に枝刈りできます。

	// probCutに使うbeta値。
    probCutBeta = beta + 215 - 60 * improving;

	if (depth >= 3
        && !is_decisive(beta)
        // If value from transposition table is lower than probCutBeta, don't attempt
        // probCut there
        // 置換表から得た値が probCutBeta より低い場合は、そこで probCut を試みない
        && !(is_valid(ttData.value) && ttData.value < probCutBeta))
    {
        ASSERT_LV3(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

#if STOCKFISH
        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
#else
        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory,
                      search_options.generate_all_legal_moves);
#endif

        Depth dynamicReduction = (ss->staticEval - beta) / 300;
        Depth probCutDepth     = std::max(depth - 5 - dynamicReduction, 0);

		// 💡 試行回数は2回(cutNodeなら4回)までとする。(よさげな指し手を3つ試して駄目なら駄目という扱い)
        //     cf. Do move-count pruning in probcut : https://github.com/official-stockfish/Stockfish/commit/b87308692a434d6725da72bbbb38a38d3cac1d5f

        while ((move = mp.next_move()) != Move::none())
        {
            ASSERT_LV3(move.is_ok());
            ASSERT_LV5(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

            if (move == excludedMove || !pos.legal(move))
                continue;

            //assert(pos.capture_stage(move));
            // ⚠ moveとして歩の成りも返ってくるが、これがcapture_stage()と一致するとは限らない。
            //     MovePickerはprob cutの時に、
            //    (GenerateAllLegalMovesオプションがオンであっても)歩の成らずは返してこないことを保証すべき。

            movedPiece = pos.moved_piece(move);

#if STOCKFISH
            do_move(pos, move, st, ss);
#else
            do_move_(pos, move, st, ss);
#endif

            // Perform a preliminary qsearch to verify that the move holds
            // この指し手がよさげであることを確認するための予備的なqsearch

            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            // If the qsearch held, perform the regular search
            // qsearch が維持された場合、通常の探索を実行する

            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth,
                                       !cutNode);

            undo_move(pos, move);

            if (value >= probCutBeta)
            {
                // Save ProbCut data into transposition table
                // ProbCutのdataを置換表に保存する。

				ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               probCutDepth + 1, move, unadjustedStaticEval, tt.generation());

                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        } // end of while
    }

moves_loop:  // When in check, search starts here
			 // 王手がかかっている局面では、探索はここから始まる。

	// -----------------------
    // Step 12. A small Probcut idea
    // Step 12. 小さなProbcutのアイデア
    // -----------------------

    probCutBeta = beta + 417;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value))
        return probCutBeta;

	// -----------------------
    // 🚀 moves loopに入る前の準備
    // -----------------------

	// continuationHistory[0]  = Counter Move History    : ある指し手が指されたときの応手
    // continuationHistory[1]  = Follow up Move History  : 2手前の自分の指し手の継続手
    // continuationHistory[3]  = Follow up Move History2 : 4手前からの継続手

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,
                  &pawnHistory, ss->ply
#if !STOCKFISH
                  ,
                  search_options.generate_all_legal_moves
#endif
    );

    value = bestValue;

	// 調べた指し手の数(合法手に限る)
    int moveCount = 0;

	// -----------------------
    // Step 13. Loop through all pseudo-legal moves until no moves remain
    //			or a beta cutoff occurs.
    // Step 13. 擬似合法手をすべてループし、手がなくなるか
    //          もしくはbetaカットオフが発生するまで繰り返します。
    // -----------------------

	// 💡 MovePickerが返す指し手はpseudo-legalであることは保証されているが、
	//     do_move()までにはlegalかどうかの判定が必要。

    while ((move = mp.next_move()) != Move::none())
    {
        ASSERT_LV3(move.is_ok());
        ASSERT_LV5(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

        if (move == excludedMove)
            continue;

        // Check for legality
        // 指し手の合法性のチェック

        /*
		   📝 root nodeなら、rootMovesになければlegalではないのでこのチェックは不要だが、
			   root nodeは全体から見ると極わずかなのでそのチェックを端折るほうが良いようだ。

			   非合法手はほとんど含まれていないから、以前はこの判定はdo_move()の直前まで
			   遅延させたほうが得だったが、do_move()するまでの枝刈りが増えてきたので、
			   ここでやったほうが良いようだ。
		*/

        if (!pos.legal(move))
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root
        // Move List. In MultiPV mode we also skip PV moves that have been already
        // searched and those of lower "TB rank" if we are in a TB root position.

        // ルートで "searchmoves" オプションに従い、Root Move Listにリストされていない手をスキップします。
        // MultiPVモードでは、既に検索されたPVの手や、TBルート位置にいる場合のTBランクが低い手も
        // スキップします。

        // 💡 root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。

        if (rootNode
#if STOCKFISH
            && !std::count(rootMoves.begin() + pvLast,
        // 📝 将棋ではこの処理不要なのでやねうら王ではpvLastは使わない。
#else
            && !std::count(rootMoves.begin() + pvIdx,
#endif
                           rootMoves.end(), move))
            continue;

        // do_move()した指し手の数のインクリメント
        ss->moveCount = ++moveCount;

// 🤔 Stockfish本家のこの読み筋の出力、細かすぎるので時間をロスする。しないほうがいいと思う。
#if STOCKFISH

		if (rootNode && is_mainthread() && nodes > 10000000)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        }
#endif

        // 💡 次のnodeのpvをクリアしておく。

        if (PvNode)
            (ss + 1)->pv = nullptr;

        // -----------------------
        //      extension
        //      探索の延長
        // -----------------------

        // 今回延長する残り探索深さ
        extension = 0;

        // 今回捕獲する駒
        capture    = pos.capture_stage(move);

        // 今回移動させる駒(移動後の駒)
        movedPiece = pos.moved_piece(move);

        // 今回の指し手で王手になるかどうか
        givesCheck = pos.gives_check(move);

        // quietの指し手の連続回数
        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;

        // Calculate new depth for this move
        // 今回の指し手に関して新しいdepth(残り探索深さ)を計算する。
        newDepth = depth - 1;

        int delta = beta - alpha;

        // ⚠ reduction()では、depthを1024倍した値が返ってきている。
        Depth r = reduction(improving, depth, moveCount, delta);

        // Increase reduction for ttPv nodes (*Scaler)
        // Smaller or even negative value is better for short time controls
        // Bigger value is better for long time controls

        // ttPv ノードに対する減少量を増やす（*Scaler）
        // 短い持ち時間制限では、より小さい値、あるいは負の値のほうが望ましい
        // 長い持ち時間制限では、より大きい値のほうが望ましい

        if (ss->ttPv)
            r += 931;

        // -----------------------
        // Step 14. Pruning at shallow depth
        // Step 14. (残り探索深さが)浅い深さでの枝刈り
        // -----------------------

        // Depth conditions are important for mate finding.
        // (残り探索)深さの条件は詰みを見つける上で重要である

        // 📊 【計測資料 7.】 浅い深さでの枝刈りを行なうときに王手がかかっていないことを条件に入れる/入れない

        if (!rootNode
            //&& pos.non_pawn_material(us)
            && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
            // movecountがFutilityMoveCountの閾値を超えた場合、quietの手をスキップします

            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // Reduced depth of the next LMR search
            // 次のLMR探索における減らさたあとの深さ

            // rは1024倍されているので注意。
            int lmrDepth = newDepth - r / 1024;

            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist =
                  captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                // 駒を取る指し手に対するfutility枝刈り

                if (!givesCheck && lmrDepth < 7 && !ss->inCheck)
                {
                    // 🤔 StockfishのPieceValue[]は、
                    //     やねうら王ではCapturePieceValue[] + CapturePieceValue[capturedPiece]
                    //       = CapturePieceValuePlusPromote()
                    //     のほうがより正確な評価ではないか？


                    Value futilityValue = ss->staticEval + 232 + 224 * lmrDepth
                                        + PieceValue[capturedPiece] + 131 * captHist / 1024;

                    if (futilityValue <= alpha)
                        continue;
                }

                // SEE based pruning for captures and checks
                // 駒取りや王手に対するSEE（静的交換評価）に基づく枝刈り

                int margin = std::clamp(158 * depth + captHist / 31, 0, 283 * depth);
                if (!pos.see_ge(move, -margin))
                {
#if STOCKFISH
                    bool mayStalemateTrap =
                      depth > 2 && alpha < 0 && pos.non_pawn_material(us) == PieceValue[movedPiece]
                      && PieceValue[movedPiece] >= RookValue
                      // it can't be stalemate if we moved a piece adjacent to the king
                      && !(attacks_bb<KING>(pos.square<KING>(us)) & move.from_sq())
                      && !mp.can_move_king_or_pawn();

                    // avoid pruning sacrifices of our last piece for stalemate
                    if (!mayStalemateTrap)
                        continue;
#else

                    /*
						🤔 Stockfishは、StalemateTrapっぽかったら、この枝刈りをしないことになっているが、
						    将棋では関係ないので、単にcontinue。
					*/

                    continue;

#endif
				}
            }
            else
            {
                int history = (*contHist[0])[movedPiece][move.to_sq()]
                            + (*contHist[1])[movedPiece][move.to_sq()]
                            + pawnHistory[pawn_history_index(pos)][movedPiece][move.to_sq()];

                // Continuation history based pruning
                // Continuation historyに基づいた枝刈り(historyの値が悪いものに関してはskip)

                if (history < -4361 * depth)
                    continue;

                history += 71 * mainHistory[us][move.from_to()] / 32;

                lmrDepth += history / 3233;

                Value baseFutility = (bestMove ? 46 : 230);
                Value futilityValue =
                  ss->staticEval + baseFutility + 131 * lmrDepth + 91 * (ss->staticEval > alpha);

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning
                // scales well with respect to time and threads

                // Futility枝刈り: 親ノード
                // (*Scaler): 一般的に、より頻繁な無駄枝刈りは
                // 時間およびスレッドに対して適切にスケールする

                // 📝 親nodeの時点で子nodeを展開する前にfutilityの対象となりそうなら枝刈りしてしまう。
                // 🤔 パラメーター調整の係数を調整したほうが良いのかも知れないが、
                // 　  ここ、そんなに大きなEloを持っていないので、調整しても…。

                if (!ss->inCheck && lmrDepth < 11 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue) && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }

                /*
				   ⚠ 以下のLMRまわり、棋力に極めて重大な影響があるので枝刈りを入れるかどうかを含めて慎重に調整すべき。
				       将棋ではseeが負の指し手もそのあと詰むような場合があるから、あまり無碍にも出来ないようだが…。
				*/

                // 📊【計測資料 20.】SEEが負の指し手を枝刈りする/しない

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE
                // 負のSEEを持つ指し手を枝刈りする
                // 💡 lmrDepthの2乗に比例するのでこのパラメーターの影響はすごく大きい。

                if (!pos.see_ge(move, -26 * lmrDepth * lmrDepth))
                    continue;
            }
        }

		// -----------------------
        // Step 15. Extensions
        // Step 15. (探索の)延長
        // -----------------------

        // Singular extension search. If all moves but one
        // fail low on a search of (alpha-s, beta-s), and just one fails high on
        // (alpha, beta), then that move is singular and should be extended. To
        // verify this we do a reduced search on the position excluding the ttMove
        // and if the result is lower than ttValue minus a margin, then we will
        // extend the ttMove. Recursive singular search is avoided.

		// シンギュラー延長探索。もし (alpha-s, beta-s) の探索で 1 手を除くすべての手が fail low となり、
        // (alpha, beta) の探索でただ 1 手だけが fail high となった場合、
        // その手はシンギュラー（特異）と判断し、延長すべきです。
        // これを検証するため、ttMove を除外した局面で縮小探索を行い、
        // その結果が ttValue からマージンを引いた値より低ければ、ttMove を延長します。
        // 再帰的なシンギュラー探索は回避されます。

        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.

        // （*Scaler）一般的に、より高い singularBeta（すなわち ttValue に近い値）と、
        // より低い延長マージンはスケーリングに適しています。

		/*
			📃 Stockfishの実装だとmargin = 2 * depthだが、
				将棋だと1手以外はすべてそれぐらい悪いことは多々あり、
				ほとんどの指し手がsingularと判定されてしまう。
				これでは効果がないので、1割ぐらいの指し手がsingularとなるぐらいの係数に調整する。
		
			💡 singular延長で強くなるのは、あるnodeで1手だけが特別に良い場合、
				相手のプレイヤーもそのnodeではその指し手を選択する可能性が高く、
				それゆえ、相手のPVもそこである可能性が高いから、そこを相手よりわずかにでも
				読んでいて詰みを回避などできるなら、その相手に対する勝率は上がるという理屈。

				いわば、0.5手延長が自己対戦で(のみ)強くなるのの拡張。
				そう考えるとベストな指し手のスコアと2番目にベストな指し手のスコアとの差に
				応じて1手延長するのが正しいのだが、2番目にベストな指し手のスコアを
				小さなコストで求めることは出来ないので…。

			📝 ifの条件式に !excludedMove があるのは、再帰的なsingular延長を除外するため。
		*/

		// singular延長をするnodeであるか。
        if (!rootNode && move == ttData.move && !excludedMove && depth >= 6 + ss->ttPv
            && is_valid(ttData.value) && !is_decisive(ttData.value) && (ttData.bound & BOUND_LOWER)
            && ttData.depth >= depth - 3)
        {
            /*
				💡 このnodeについてある程度調べたことが置換表によって証明されている。(ttMove == moveなのでttMove != Move::none())
				    (そうでないとsingularの指し手以外に他の有望な指し手がないかどうかを調べるために
					null window searchするときに大きなコストを伴いかねないから。)
			*/

            //  📍 このmargin値は評価関数の性質に合わせて調整されるべき。

            Value singularBeta  = ttData.value - (56 + 79 * (ss->ttPv && !PvNode)) * depth / 58;
            Depth singularDepth = newDepth / 2;

            // 💡 move(ttMove)の指し手を以下のsearch()での探索から除外。

            ss->excludedMove = move;

            // 📝 局面はdo_move()で進めずにこのnodeから浅い探索深さで探索しなおす。
            //     浅いdepthでnull windowなので、すぐに探索は終わるはず。

            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            // 💡 置換表の指し手以外がすべてfail lowしているならsingular延長確定。
            //    (延長され続けるとまずいので何らかの考慮は必要)

            if (value < singularBeta)
            {
                int corrValAdj   = std::abs(correctionValue) / 249096;
                int doubleMargin = 4 + 205 * PvNode - 223 * !ttCapture - corrValAdj
                                 - 959 * ttMoveHistory / 131072 - (ss->ply > rootDepth) * 45;
                int tripleMargin = 80 + 276 * PvNode - 249 * !ttCapture + 86 * ss->ttPv - corrValAdj
                                 - (ss->ply * 2 > rootDepth * 3) * 53;

                // 📝 2重延長を制限して探索の組合せ爆発を回避する必要がある。

                extension =
                  1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);

                depth++;
            }

            // Multi-cut pruning
            // Our ttMove is assumed to fail high based on the bound of the TT entry,
            // and if after excluding the ttMove with a reduced search we fail high
            // over the original beta, we assume this expected cut-node is not
            // singular (multiple moves fail high), and we can prune the whole
            // subtree by returning a softbound.

            // マルチカット枝刈り
            // 私たちのttMoveはfail highすると想定されており、
            // 今、ttMoveなしの(この局面でttMoveの指し手を候補手から除外した)、
            // reduced search(探索深さを減らした探索)でもfail highしました。
            // したがって、この予想されるカットノードはsingular(1つだけ傑出した指し手)ではないと想定し、
            // 複数の手がfail highすると考え、softboundを返すことで全サブツリーを枝刈りすることができます。

            /*
				📓 訳注
            
				 cut-node  : αβ探索において早期に枝刈りできるnodeのこと。
							 つまり、searchの引数で渡されたbetaを上回ることがわかったのでreturnできる(これをbeta cutと呼ぶ)
							 できるようなnodeのこと。
            
				 softbound : lowerbound(下界)やupperbound(上界)のように真の値がその値より大きい(小さい)
							 ことがわかっているような値のこと。

			*/

            else if (value >= beta && !is_decisive(value))
                return value;

            // Negative extensions
            // If other moves failed high over (ttValue - margin) without the
            // ttMove on a reduced search, but we cannot do multi-cut because
            // (ttValue - margin) is lower than the original beta, we do not know
            // if the ttMove is singular or can do a multi-cut, so we reduce the
            // ttMove in favor of other moves based on some conditions:

            // 負の延長
            // もしttMoveを使用せずに(ttValue - margin)以上で他の手がreduced search
            // (簡略化した探索)で高いスコアを出したが、(ttValue - margin)が元のbetaよりも
            // 低いためにマルチカットを行えない場合、
            // ttMoveがsingularかマルチカットが可能かはわからないので、
            // いくつかの条件に基づいて他の手を優先してttMoveを減らします：


            // If the ttMove is assumed to fail high over current beta
            // ttMove が現在の beta を超えて fail high すると想定される場合

            else if (ttData.value >= beta)
                extension = -3;

            // If we are on a cutNode but the ttMove is not assumed to fail high
            // over current beta
            // 現在のノードがカットノードであるが、ttMoveが現在のbetaを超えて
            // fail highすると思われない場合

            else if (cutNode)
                extension = -2;

            /*
			  ⚠  王手延長に関して、Stockfishのコード、ここに持ってくる時には気をつけること！
			       将棋では王手はわりと続くのでそのまま持ってくるとやりすぎの可能性が高い。

			  📓 Stockfishで削除されたが、王手延長自体は何らかあった方が良い可能性はあるので条件を調整してはどうか。
					Remove check extension : https://github.com/official-stockfish/Stockfish/commit/96837bc4396d205536cdaabfc17e4885a48b0588
			*/
        }

		// -----------------------
        // Step 16. Make the move
        // Step 16. 指し手で進める
        // -----------------------

		// 指し手で1手進める
        do_move(pos, move, st, givesCheck, ss);

        // Add extension to new depth
        // 求まった延長する手数を新しいdepthに加算

		newDepth += extension;

		uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

        // Decrease reduction for PvNodes (*Scaler)
        // Pv Nodesに対してreductionを減らす(*Scaler)

		if (ss->ttPv)
            r -= 2510 + PvNode * 963 + (ttData.value > alpha)* 916
               + (ttData.depth >= depth) * (943 + cutNode * 1180);

        // These reduction adjustments have no proven non-linear scaling
        // これらの減少量調整には、非線形スケーリングの有効性が証明されていません

        r += 679 - 6 * msb(depth);  // Base reduction offset to compensate for other tweaks
								    // 他の調整を補正するための基準リダクションオフセット
        r -= moveCount * (67 - 2 * msb(depth));
        r -= std::abs(correctionValue) / 27160;

        // Increase reduction for cut nodes
        // カットノードのreductionを増やす

		/*
			💡 cut nodeにおいてhistoryの値が悪い指し手に対してはreduction量を増やす。
			    PVnodeではIID時でもcutNode == trueでは呼ばないことにしたので、
			        if (cutNode)という条件式は暗黙に && !PvNode を含む。

			📊 【計測資料 18.】cut nodeのときにreductionを増やすかどうか。
		*/

        if (cutNode)
            r += 2998 + 2 * msb(depth) + (948 + 14 * msb(depth)) * !ttData.move;

        // Increase reduction if ttMove is a capture
        // ttMove が捕獲する指し手なら、reductionを増やす

        if (ttCapture)
            r += 1402 - 39 * msb(depth);

        // Increase reduction if next ply has a lot of fail high
        // 次の手でfail highが多い場合、reductionを増やす

        if ((ss + 1)->cutoffCnt > 2)
            r += 925 + 33 * msb(depth) + allNode * (701 + 224 * msb(depth));

        r += (ss + 1)->quietMoveStreak * 51;

        // For first picked move (ttMove) reduce reduction
        // 最初に選ばれた指し手（ttMove）ではreductionを減らす

        if (move == ttData.move)
            r -= 2121 + 28 * msb(depth);

        if (capture)
            ss->statScore = 782 * int(PieceValue[pos.captured_piece()]) / 128
                          + captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())];
        else
            // 📊【計測資料 11.】statScoreの計算でcontHist[3]も調べるかどうか。
            // 🤔 contHist[5]も/2とかで入れたほうが良いのでは…。誤差か…？
            ss->statScore = 2 * mainHistory[us][move.from_to()]
                          + (*contHist[0])[movedPiece][move.to_sq()]
                          + (*contHist[1])[movedPiece][move.to_sq()];

        // Decrease/increase reduction for moves with a good/bad history
        // 良い/悪い履歴を持つ手に対して、reductionを減らす/増やす

        r -= ss->statScore * (729 - 12 * msb(depth)) / 8192;

		// -----------------------
        // Step 17. Late moves reduction / extension (LMR)
        // Step 17. 遅い指し手の削減／延長（LMR)
        // -----------------------

		/*
			📓 reduction(削減)とは、depthを減らした探索のこと。
		        late move(遅い指し手)とは、このnodeでMovePickerで生成されたのが、あとのほうの指し手のこと。
		        Late Move Reductionは、遅い指し手の削減を意味していて、LMRと略される。
		*/

        if (depth >= 2 && moveCount > 1)
        {
            /*
			  💡 depthを減らして探索させて、その指し手がfail highしたら元のdepthで再度探索する。
			      moveCountが大きいものなどは探索深さを減らしてざっくり調べる。
			      alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。
			*/

            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth.
            // To prevent problems when the max value is less than the min value,
            // std::clamp has been replaced by a more robust implementation.

			// 一般的には LMR 探索の深さを newDepth で上限にしたいが、
            // reduction が負の場合、この指し手には最初の指し手の深さを超えて
            // 限定的な探索延長を許す。
            // max 値が min 値より小さくなる問題を防ぐため、
            // std::clamp はより堅牢な実装に置き換えられた。
            /*
                📓 C++の仕様上、std::clamp(x, min, max)は、min > maxの時に未定義動作であるから、
			        clamp()を用いるのではなく、max()とmin()を組み合わせて書かないといけない。
			*/

            Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 1 + PvNode)) + PvNode;

            ss->reduction = newDepth - d;
            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;

            // Do a full-depth search when reduced LMR search fails high
            // 深さを減らした LMR 探索がfail highを出した場合は、full depth(元の探索深さ)で探索を行う

			// (*Scaler) Usually doing more shallower searches
            // doesn't scale well to longer TCs
            // （*Scaler）通例、より浅い探索を増やしても
            // 長い持ち時間制限ではうまくスケールしない

            if (value > alpha)
            {
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                // LMRの結果に基づいて完全な探索深さを調整します -
                // 結果が十分に良ければ深く探索し、十分に悪ければ浅く探索します。

                const bool doDeeperSearch = d < newDepth && value > (bestValue + 43 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                // Post LMR continuation history updates
                // LMR後のcontinuation historyの更新

                update_continuation_histories(ss, movedPiece, move.to_sq(), 1412);
            }
            else if (value > alpha && value < bestValue + 9)
                newDepth--;
        }

		// -----------------------
        // Step 18. Full-depth search when LMR is skipped
        // Step 18. LMRがスキップされた場合の完全な探索
        // -----------------------

        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            // ttMoveが存在しない場合、削減を増やします。

            if (!ttData.move)
                r += 1199 + 35 * msb(depth);

            if (depth <= 4)
                r += 1150;

            // Note that if expected reduction is high, we reduce search depth here
            // 期待される削減が大きい場合、ここで探索深さを1減らすことに注意してください。

            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > 3200) - (r > 4600 && newDepth > 2), !cutNode);

		}

        // For PV nodes only, do a full PV search on the first move or after a fail high,
        // otherwise let the parent node fail low with value <= alpha and try another move.
        // PVノードの場合のみ、最初の手やfail highの後に完全なPV探索を行います。
        // それ以外の場合、親ノードが value <= alpha でfail lowし、別の手を試みます。

		/*
		   📓 PV nodeにおいては、full depth searchがfail highしたならPV nodeとしてsearchしなおす。
               ただし、value >= betaなら、正確な値を求めることにはあまり意味がないので、
		       これはせずにbeta cutしてしまう。
		*/

        if (PvNode && (moveCount == 1 || value > alpha))
        {
            // 次のnodeのPVポインターはこのnodeのpvバッファを指すようにしておく。
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();

            // Extend move from transposition table if we are about to dive into qsearch.
            // qsearchに入ろうとしている場合、置換表からの手を延長します。

            if (move == ttData.move && rootDepth > 8)
                newDepth = std::max(newDepth, 1);

			// 📝 full depthで探索するときはcutNodeにしてはいけない。
            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

		// -----------------------
        // Step 19. Undo move
        // Step 19. 1手戻す
        // -----------------------

		undo_move(pos, move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

		// -----------------------
        // Step 20. Check for a new best move
        // -----------------------

        // Finished searching the move. If a stop occurred, the return value of
        // the search cannot be trusted, and we return immediately without updating
        // best move, principal variation nor transposition table.
		// 指し手の探索が終了しました。もし停止が発生した場合、探索の返り値は信頼できないため、
        // 最善手、主要変化、トランスポジションテーブルを更新せずに直ちに戻ります。

        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

		// -----------------------
        //  root node用の特別な処理
        // -----------------------

        if (rootNode)
        {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

			/*
				📓
					effort           : このRootMoveのために探索したnode数
					averageScore     : rootの平均スコア。
					meanSquaredScore : rootの二乗平均スコア。

					これらは、aspiration searchのときにもう一回iterationが回るかの判定に用いる。
			*/

            // rootでこのRootMovesの指し手に対して探索したnode数を加算。
			rm.effort += nodes - nodeCount;

			rm.averageScore =
              rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                  ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                  : value * std::abs(value);

            // PV move or new best move?
            // PVの指し手か、新しいbest moveか？

            if (moveCount == 1 || value > alpha)
            {
                // 💡 root nodeにおいてPVの指し手または、α値を更新した場合、スコアをセットしておく。
                //    (iterationの終わりでsortするのでそのときに指し手が入れ替わる。)

                rm.score = rm.uciScore = value;
                rm.selDepth            = selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;

                if (value >= beta)
                {
                    rm.scoreLowerbound = true;
                    rm.uciScore        = beta;
                }
                else if (value <= alpha)
                {
                    rm.scoreUpperbound = true;
                    rm.uciScore        = alpha;
                }

				// 📝 PVは変化するはずなのでいったんリセットしている。
                rm.pv.resize(1);

				// 📝 1手進めたのだから、何らかPVを持っているはず。
                ASSERT_LV3((ss + 1)->pv);

				// 📝 RootでPVが変わるのは稀なのでここがちょっとぐらい重くても問題ない。
                //     新しく変わった指し手の後続のpvをRootMoves::pvにコピーしてくる。

                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.

				// 各イテレーションで最善手がどのくらいの頻度で変化したかを記録します。
                // この情報は時間管理に使用されます。MultiPV モードでは、
                // 最初の PV ラインに対してのみこれを行うよう注意する必要があります。

				// ⚠ !thisThread->pvIdx という条件を入れておかないとMultiPVで
                //     time managementがおかしくなる。

                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;
            }
            else
                // All other moves but the PV, are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved - just the PV is pushed up.

				// PV以外のすべての手は最低値に設定されます。
                // これはソート時に問題とならないです。
                // なぜなら、ソートは安定しており、リスト内の手の位置は保持されているからです
                // - PVだけが上に押し上げられます。

                rm.score = -VALUE_INFINITE;

				/*
					📓 root nodeにおいてα値を更新しなかったのであれば、この指し手のスコアを
						-VALUE_INFINITEにしておく。
						こうしておかなければ、stable_sort() しているにもかかわらず、
						前回の反復深化のときの値との 大小比較してしまい指し手の順番が
						入れ替わってしまうことによるオーダリング性能の低下がありうる。
				*/

        }

		// -----------------------
        //  alpha値の更新処理
        // -----------------------

        // In case we have an alternative move equal in eval to the current bestmove,
        // promote it to bestmove by pretending it just exceeds alpha (but not beta).

		// 評価値が現在の最善手と等しい代替手がある場合、
        // その手を最善手に昇格させます。この際、α（アルファ）を少しだけ超えるが、
        // β（ベータ）は超えないように見せかけます。

        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                   && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)  // Update pv even in fail-high case
					                      // fail-highのときにもPVをupdateする。
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta)
                {
                    // (* Scaler) Especially if they make cutoffCnt increment more often.
                    //（* Scaler）特に cutoffCnt のインクリメントをより頻繁に行わせる場合に

                    ss->cutoffCnt += (extension < 2) || PvNode;
                    ASSERT_LV3(value >= beta);  // Fail high

					/*
						📓 value >= beta なら fail high(beta cut)
							また、non PVであるなら探索窓の幅が0なのでalphaを更新した時点で、
							value >= betaが言えて、beta cutである。
					*/
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                // 少なくとも1つのスコアの改善が見られた場合、他の手(の探索深さ)を削減します。

                if (depth > 2 && depth < 16 && !is_decisive(value))
                    depth -= 2;

				ASSERT_LV3(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
                                // alpha値を更新! つねに alpha < beta

				/*
					💬 このとき相手からの詰みがあるかどうかを
					    調べるなどしたほうが良いならここに書くべし。
				*/
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.

		// その手が以前に探索された他の手よりも悪い場合、
        // 後でその統計を更新するために記憶しておきます。

        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                // 探索した、駒を捕獲する指し手
                capturesSearched.push_back(move);
			else
                // 探索した、駒を捕獲しない指し手
                quietsSearched.push_back(move);
        }
    } // end of while

	// -----------------------
    // Step 21. Check for mate and stalemate
    // -----------------------

    // All legal moves have been searched and if there are no legal moves, it
    // must be a mate or a stalemate. If we are in a singular extension search then
    // return a fail low score.

	// すべての合法手が探索されており、合法手が存在しない場合、
	// それは詰みかステイルメイトである。
	// シンギュラー延長探索中であれば、fail low スコアを返す。

	// 📝 将棋ではステイルメイトは存在しないので、合法手がなければ負け。

	// このStockfishのassert、合法手を生成しているので重すぎる。良くない。
    ASSERT_LV5(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL_ALL>(pos).size());

    // Adjust best value for fail high cases
    // fail highの場合に最良値を調整する

    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

	// ⚠ Stockfishでは、ここのコードは以下のようになっているが、これは、
    //     自玉に王手がかかっておらず指し手がない場合は、stalemateで引き分けだから。

#if STOCKFISH
    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;
#else
    // ⚠ ⇓ここ⇓、↑Stockfishのコード↑をそのままコピペしてこないように注意！

	// 🤔 (将棋では)合法手がない == 詰まされている なので、rootの局面からの手数で詰まされたという評価値を返す。
    //     ただし、singular extension中のときは、ttMoveの指し手が除外されているので単にalphaを返すべき。
    if (!moveCount)
        bestValue = excludedMove ? alpha : mated_in(ss->ply);
#endif

	// If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    // alphaよりも大きな探索値を生み出す手がある場合、探索された手の統計を更新します

	else if (bestMove)
    {
        // 💡 quietな(駒を捕獲しない)best moveなのでkillerとhistoryとcountermovesを更新する。

		update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 811 : -848);
    }

    // Bonus for prior quiet countermove that caused the fail low
    // fail lowを引き起こした1手前のquiet countermoveに対するボーナス

	/*
		📓 bestMoveがない == fail lowしているケースなので、
			fail lowを引き起こした前nodeでのcounter moveに対してボーナスを加点する。

		📊 【計測資料 15.】search()でfail lowしているときにhistoryのupdateを行なう条件
	*/
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -215;
        bonusScale += std::min(-(ss - 1)->statScore / 103, 337);
        bonusScale += std::min(64 * depth, 552);
        bonusScale += 177 * ((ss - 1)->moveCount > 8);
        bonusScale += 141 * (!ss->inCheck && bestValue <= ss->staticEval - 94);
        bonusScale += 141 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 76);

        bonusScale = std::max(bonusScale, 0);

        const int scaledBonus = std::min(155 * depth - 88, 1416) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * 397 / 32768);

        mainHistory[~us][((ss - 1)->currentMove).from_to()] << scaledBonus * 224 / 32768;

		// TODO : これで合ってるか？あとで検証する。
        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << scaledBonus * 1127 / 32768;
    }

    // Bonus for prior capture countermove that caused the fail low
    // 前のfail lowを引き起こしたcapture countermoveに対するボーナス

    else if (priorCapture && prevSq != SQ_NONE)
    {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 1042;
    }

    // ⚠ 将棋ではtable probeを使っていないので、maxValueは使わない。
    //     ゆえにStockfishのここのコードは不要。(maxValueでcapする必要がない)
#if STOCKFISH
    if (PvNode)
        bestValue = std::min(bestValue, maxValue);
#endif

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.

	// もし良い指し手が見つからず(bestValueがalphaを更新せず)、前の局面はttPvを選んでいた場合は、
    // 前の相手の手がおそらく良い手であり、新しい局面が探索木に追加される。
    // (ttPvをtrueに変更してTTEntryに保存する)

    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

	// -----------------------
    //  置換表に保存する
    // -----------------------

    // Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.

	// 収集した情報をトランスポジションテーブルに書き込みます。
    // 静的評価は、修正履歴が適用される前の状態で保存されることに注意してください。

	/*
		📓 betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから
			真の値はまだ大きいと考えられる。

			すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
			さもなくば、(PvNodeなら)枝刈りはしていないので、
			これが正確な値であるはずだから、BOUND_EXACTを返す。

			また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、
			BOUND_UPPERという扱いにする。

			ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の
			詰みがあるかも知れないから、すなわち、スコアは変動するかも知れないので、
			BOUND_UPPERという扱いをする。
	*/

    if (!excludedMove && !(rootNode && pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation());

    // Adjust correction history
	// correction historyの調整

	if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && ((bestValue < ss->staticEval && bestValue < beta)  // negative correction & no fail high
            || (bestValue > ss->staticEval && bestMove)))     // positive correction & no fail low
    {
        auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                                -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *this, bonus);
    }

	// 👉 qsearch()内の末尾にあるassertの文の説明を読むこと。
	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

    return bestValue;
}


// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search

// 静止関数。これはメインの探索関数から深さ0で呼び出されるか、さらに深さを減らしながら再帰的に呼び出される。
// depth <= 0 の場合、本来は静的評価だけを使う「べき」だが、戦術的な手が静的評価を惑わせることがある。
// この地平線効果に対抗するため、戦術的な手だけを対象としたこの静止探索を実装している。
// 詳細は https://www.chessprogramming.org/Horizon_Effect
// および https://www.chessprogramming.org/Quiescence_Search を参照。

template<NodeType nodeType>
Value Search::YaneuraOuWorker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    /*
		📓 チェスと異なり将棋では、手駒があるため、王手を無条件で延長するとかなりの長手数、王手が続くことがある。
			手駒が複数あると、その組み合わせをすべて延長してしまうことになり、組み合わせ爆発を容易に起こす。
	
			この点は、Stockfishを参考にする時に、必ず考慮しなければならない。
	
			ここでは、以下の対策をする。
			1. qsearch(静止探索)ではcaptures(駒を取る指し手)とchecks(王手の指し手)のみをMovePickerで生成
			2. 王手の指し手は、depthがDEPTH_QS_CHECKS(== 0)の時だけ生成。
			3. capturesの指し手は、depthがDEPTH_QS_RECAPTURES(== -5)以下なら、直前に駒が移動した升に移動するcaptureの手だけを生成。(取り返す手)
			4. captureでも歩損以上の損をする指し手は延長しない。
			5. 連続王手の千日手は検出する
			これらによって、王手ラッシュや連続王手で追い回して千日手(実際は反則負け)に至る手順を排除している。
	
			ただし、置換表の指し手に関してはdepth < DEPTH_QS_CHECKS でも王手の指し手が交じるので、
			置換表の指し手のみで循環するような場合、探索が終わらなくなる。
	
			そこで、
			6. depth < -16 なら、置換表の指し手を無視する
			のような対策が必要だと思う。
			 →　引き分け扱いすることにした。
	*/

    // -----------------------
    //     変数宣言
    // -----------------------

    // PV nodeであるか。
    // 📝 ここがRoot nodeであることはないので、そのケースは考えなくて良い。

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    ASSERT_LV3(PvNode || (alpha == beta - 1));

    // 🤔 Stockfishではここで上記のように千日手に突入できるかのチェックがあるようだが
    //     将棋でこれをやっても強くならないので導入しない。
#if STOCKFISH
    // Check if we have an upcoming move that draws by repetition
    // 反復による引き分けとなる可能性のある次の手があるかを確認する

    // 💡 このコードの原理としては、次の一手で千日手局面に持ち込めるなら、
    //     少なくともこの局面は引き分けであるから、
    //     betaが引き分けのスコアより低いならbeta cutできるというもの。

    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }
#endif

    // PV求める用のbuffer
    // 💡 これnonPVでは使わないので、参照しておらず削除される。
    Move pv[MAX_PLY + 1];

    // make_move()のときに必要
    StateInfo st;

    // この局面のhash key
	Key posKey;

    // move				: MovePickerからもらった現在の指し手
    // bestMove			: この局面でのベストな指し手
    Move move, bestMove;

    // bestValue		: best moveに対する探索スコア(alphaとは異なる)
    // value			: 現在のmoveに対する探索スコア
    // futilityBase		: futility pruningの基準となる値
    Value bestValue, value, futilityBase;

    // pvHit			: 置換表から取り出した指し手が、PV nodeでsaveされたものであった。
    // givesCheck		: MovePickerから取り出した指し手で王手になるか
    // capture          : 駒を捕獲する指し手か
    bool pvHit, givesCheck, capture;

    // このnodeで何手目の指し手であるか
    int moveCount;

    // -----------------------
    // Step 1. Initialize node
    // Step 1. ノードの初期化
    // -----------------------

    if (PvNode)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }

    bestMove                    = Move::none();
    ss->inCheck                 = pos.checkers();
    moveCount                   = 0;

#if defined(USE_CLASSIC_EVAL) && defined(USE_LAZY_EVALUATE)
    bool evaluated = false;
    auto evaluate  = [&](Position& pos) {
        evaluated = true;
        return this->evaluate(pos);
    };
    auto do_move = [&](Position& pos, Move move, StateInfo st, bool givesCheck, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, givesCheck, ss);
    };
#endif

#if STOCKFISH
#else
    // やねうら王探索で追加した思考エンジンオプション
    auto& search_options = main_manager()->search_options;
#endif

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    // selDepth情報をGUIに送信するために使用します（selDepthは1からカウントし、plyは0からカウントします）。

    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    // -----------------------
    // Step 2. Check for an immediate draw or maximum ply reached
    // Step 2. 即座に引き分けになるか、最大のply(手数)に達していないかを確認します。
    // -----------------------

    // 千日手チェックは、MovePickerでcaptures(駒を取る指し手しか生成しない)なら、
    // 千日手チェックしない方が強いようだ。
    // ただし、MovePickerで、TTの指し手に対してもcapturesであるという制限をかけないと
    // TTの指し手だけで無限ループ(MAX_PLYまで再帰的に探索が進む)になり弱くなるので、注意が必要。

#if STOCKFISH
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

#else

    // ⚠ Stockfishはis_draw()で千日手判定をしているが、
    //     やねうら王では劣等局面の判定があるので is_repetition()で判定しなくてはならない。

    // 🌈 やねうら王独自改良 🌈

    // 現局面の手番側のColor
    Color us = pos.side_to_move();

    auto draw_type = pos.is_repetition(ss->ply);
    if (draw_type != REPETITION_NONE)
    /*
			📓 なぜvalue_from_tt()が必要なのか？

			draw_value()では、優等局面、劣等局面の時にVALUE_MATE , -VALUE_MATEが返ってくる。
			これは1手詰めのスコアと同じ意味である。

			これは現在の手数に準じたスコアに変換する必要がある。
			すなわち、mate_in(VALUE_MATE, ss->ply), mated_in(VALUE_MATE, ss->ply)と同じスコアに
			なるように変換しなければならない。
			
			そのため、value_from_tt()が必要なのである。
	*/
    {
        if (draw_type == REPETITION_DRAW)
            // 通常の千日手の時はゆらぎを持たせる。
            // 💡 引き分けのスコアvは abs(v±1) <= VALUE_MAX_EVALであることが保証されているので、
            //     value_from_tt()での変換は不要。
            return draw_value(draw_type, pos.side_to_move()) + value_draw(nodes);
		else
	        return value_from_tt(draw_value(draw_type, us), ss->ply);
    }

// TODO : あとで検討する。
#if 0
	// 16手以内の循環になってないのにqsearchで16手も延長している場合、
    // 置換表の指し手だけで長い循環になっている可能性が高く、
    // これは引き分け扱いにしてしまう。
    if (depth <= -16)
        return draw_value(REPETITION_DRAW, us);
#endif

    // 最大手数の到達
    if (ss->ply >= MAX_PLY || pos.game_ply() > search_options.max_moves_to_draw)
        return draw_value(REPETITION_DRAW, us) + value_draw(nodes);

    ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

#endif

    // -----------------------
    // Step 3. Transposition table lookup
    // Step 3. 置換表のlookup
    // -----------------------

    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);

    // Need further processing of the saved data
    // 保存されたデータのさらなる処理が必要です

    ss->ttHit   = ttHit;
    ttData.move = ttHit ? ttData.move : Move::none();

    ttData.value =
#if STOCKFISH
      ttHit ? value_from_tt(ttData.value, ss->ply , pos.rule50_count()) : VALUE_NONE;
#else
      ttHit ? value_from_tt(ttData.value, ss->ply ) : VALUE_NONE;
#endif      
    pvHit = ttHit && ttData.is_pv;

    // 📌 やねうら王では置換表に先後間違えて書き出すバグを生じうるので、このassert追加する。
    ASSERT_LV3(pos.legal_promote(ttData.move));

    // At non-PV nodes we check for an early TT cutoff
    // non-PV nodeにおいて、置換表による早期枝刈りをチェックします

    /*
		📓 nonPVでは置換表の指し手で枝刈りする
		    PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)
	*/

    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        // 置換表から取り出したときに他スレッドが値を潰している可能性がありうる
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        /*
				💡 ↑ここは、↓この意味。
				&& (ttData.value >= beta ? (ttData.bound & BOUND_LOWER)
										 : (ttData.bound & BOUND_UPPER)))
			*/
        return ttData.value;
    /*
			📓 ttData.valueが下界(真の評価値はこれより大きい)もしくはジャストな値で、
			    かつttData.value >= beta超えならbeta cutできる。

				ttData.valueが上界(真の評価値はこれより小さい)だが、
				tte->depth()のほうがdepthより深ければ
				より深い探索結果として十分信頼できるので、
				この値を信頼して、この値でreturnして良い。
		*/

    // -----------------------
    // Step 4. Static evaluation of the position
    // Step 4. この局面の静止評価
    // -----------------------

    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
    {
        /*
			📓	bestValueはalphaとは違う。
				王手がかかっているときは-VALUE_INFINITEを初期値として、
				すべての指し手を生成してこれを上回るものを探すので
				alphaとは区別しなければならない。
		*/
        bestValue = futilityBase = -VALUE_INFINITE;
	}
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);

		if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            // TT（置換表）に保存されている値については、決して何も仮定しないこと。

			// 📝 置換表に評価値が格納されているとは限らないのでその場合は評価関数の呼び出しが必要。
            //     bestValueの初期値としてこの局面のevaluate()の値を使う。これを上回る指し手があるはずなのだが..。

            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
#if defined(USE_CLASSIC_EVAL)
			else if (PvNode) {
				// 🌈 やねうら王独自
				unadjustedStaticEval = evaluate(pos);
				// ⇨ NNUEだとこれ入れたほうが強い可能性が…。
			}
#endif

			ss->staticEval = bestValue =
				to_corrected_static_eval(unadjustedStaticEval, correctionValue);

			// ttValue can be used as a better position evaluation
            // ttValueは、より良い局面評価として使用できる

			/*
				📓 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
				    精度で探索されたものであるなら、それをbestValueの初期値として使う。
			
				    ただし、mate valueは変更しない方が良いので、!is_decisive(ttData.value) は、
				    そのための条件。
			*/

            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            // -----------------------
            //  🌈 一手詰め判定
            // -----------------------

            // 置換表にhitした場合は、すでに詰みを調べたはずなので
            // 置換表にhitしなかったときにのみ調べる。

			ASSERT_LV3(!ss->inCheck && !ss->ttHit);
            if (true)
            {
                // ■ 備考
                //
                // ⇨ このqsearch()での1手詰めチェックは、いまのところ、入れたほうが良いようだ。
                //    play_time = b1000 ,  1631 - 55 - 1314(55.38% R37.54) [2016/08/19]
                //    play_time = b6000 ,  538 - 23 - 439(55.07% R35.33) [2016/08/19]

                // 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈
                move = Mate::mate_1ply(pos);
                if (move != Move::none())
                {
                    bestValue = mate_in(ss->ply + 1);

					// WRITE_QSEARCH_MATE1PLY_TO_TT
                    if (0)
                    {
                        // 🤔 このnodeに再訪問することはまずないだろうから、置換表に保存する価値はない可能性があるが。

						ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT, DEPTH_QS, move,
                                       unadjustedStaticEval, tt.generation());

                    }

                    return bestValue;
                }
            }

			// 📌 ここからStockfishの元のコード 📌

            unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue =
				to_corrected_static_eval(unadjustedStaticEval, correctionValue);

#if 0       // 以前のコード
            unadjustedStaticEval =
              (ss - 1)->currentMove != Move::null()
                ? evaluate(pos)
                : -(ss - 1)->staticEval;
#endif

        }

        // Stand pat. Return immediately if static value is at least beta
        // Stand pat。静的評価値が少なくともベータ値に達している場合は直ちに返します

		/*
			📓 現在のbestValueは、この局面で何も指さないときのスコア。
			    recaptureすると損をする変化もあるのでこのスコアを基準に考える。

				王手がかかっていないケースにおいては、この時点での静的なevalの値が
				betaを上回りそうならこの時点で帰る。
		*/

        if (bestValue >= beta)
        {
            if (!is_decisive(bestValue))
                // bestValueを少しbetaのほうに寄せる。
                bestValue = (bestValue + beta) / 2;

            if (!ss->ttHit)
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                               DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval,
                               tt.generation());
            return bestValue;
        }

		/*
			📓 王手がかかっていなくてPvNodeでかつ、bestValueがalphaより
				大きいならそれをalphaの初期値に使う。
				王手がかかっているなら全部の指し手を調べたほうがいい。
		*/ 
		if (bestValue > alpha)
            alpha = bestValue;

		// 💡 futilityの基準となる値をbestValueにmargin値を加算したものとして、
        //     これを下回るようであれば枝刈りする。

		futilityBase = ss->staticEval + 352;

    }

	// -----------------------
    //     1手ずつ調べる
    // -----------------------

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory,
                                        (ss - 2)->continuationHistory};

	// 📓 取り合いの指し手だけ生成する
    //     searchから呼び出された場合、直前の指し手がMove::null()であることがありうる。
	//     この場合、SQ_NONEを設定する。

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare to search
    // the moves. We presently use two stages of move generator in quiescence search:
    // captures, or evasions only when in check.

	// 現在の局面に対して MovePicker オブジェクトを初期化し、指し手の探索を準備します。
    // 現在、静止探索では2段階の指し手生成を使用しています：
    // captures(駒を取る指し手)、またはevasions(王手の回避)のみです。

    MovePicker mp(pos, ttData.move, DEPTH_QS, &mainHistory, &lowPlyHistory, &captureHistory,
                  contHist, &pawnHistory, ss->ply
#if !STOCKFISH
                  ,
                  search_options.generate_all_legal_moves
#endif
    );

	// -----------------------
    // Step 5. Loop through all pseudo-legal moves until no moves remain or a beta cutoff occurs.
    // Step 5. 疑似合法手をすべてループ処理し、手が残らなくなるか、ベータカットオフが発生するまで続けます。
    // -----------------------

    while ((move = mp.next_move()) != Move::none())
    {
        //assert(move.is_ok());

		// 🤔 MovePickerで生成された指し手はpseudo_legalであるはず。
        ASSERT_LV3(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

		/*
			合法手かどうかのチェック
		
			📓 指し手の合法性の判定は直前まで遅延させたほうが得だと思われていたのだが
			   (これが非合法手である可能性はかなり低いので他の判定によりskipされたほうが得)
			   Stockfish14から、静止探索でも、早い段階でlegal()を呼び出すようになった。
		*/

        if (!pos.legal(move))
            continue;

		//  局面を進める前の枝刈り

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

		// -----------------------
        // Step 6. Pruning
        // Step 6. 枝刈り
        // -----------------------

		/*
			📓 moveが王手にならない指し手であり、1手前で相手が移動した駒を
				取り返す指し手でもなく、今回捕獲されるであろう駒による評価値の上昇分を
			 加算してもalpha値を超えそうにないならこの指し手は枝刈りしてしまう。
		*/

		if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            // futility枝刈りとmove countに基づく枝刈り

            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
#if STOCKFISH            
				// TODO : ここの最適化、optimizerに任せるか？
                && move.type_of() != PROMOTION
				// 📝 この最後の条件、入れたほうがいいのか？
				// 📊 入れない方が良さげ。(V7.74taya-t50 VS V7.74taya-t51)
#endif                
				)
            {
				// 💡 MoveCountに基づく枝刈り

				if (moveCount > 2)
                    continue;

				/*
                    🤔 moveが成りの指し手なら、その成ることによる価値上昇分も
					    ここに乗せたほうが正しい見積りになるはず。
					📊 【計測資料 14.】 futility pruningのときにpromoteを考慮するかどうか。
				*/

                Value futilityValue = futilityBase +
                                    PieceValue[pos.piece_on(move.to_sq())];
                                // ⚠　これ、加算した結果、s16に収まらない可能性があるが、
                                //      計算はs32で行って、そのあと、この値を用いないからセーフ。

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.

				// 静的評価値とキャプチャする駒の価値を合わせたものがalphaより大幅に低い場合、
                // この手を枝刈りすることができます。

				// 💡 ここ、わりと棋力に影響する。下手なことするとR30ぐらい変わる。

				if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough
                // we can prune this move.

				// 静的交換評価が十分に低い場合、
                // この手を枝刈りできます。

				// 💡 futilityBaseはこの局面のevalにmargin値を加算しているのだが、
                //     それがalphaを超えないのは、悪い手だろうから枝刈りしてしまう。

				if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::min(alpha, futilityBase);
                    continue;
                }
            }

            // Continuation history based pruning
            // 継続履歴に基づく枝刈り

			/*
				📓 Stockfish12でqsearch() にも導入された。

				    駒を取らない王手回避の指し手はよろしくない可能性が高いのでこれは枝刈りしてしまう。
					成りでない && seeが負の指し手はNG。王手回避でなくとも、同様。
			*/

            if (!capture
                && (*contHist[0])[pos.moved_piece_after(move)][move.to_sq()]
                       + pawnHistory[pawn_history_index(pos)][pos.moved_piece(move)][move.to_sq()]
                     <= 5868)
                continue;

            // Do not search moves with bad enough SEE values
            // SEEが十分悪い指し手は探索しない。

			/*
				🤔　無駄な王手ラッシュみたいなのを抑制できる？
					 これ-90だとPawnValue == 90なので歩損は許してしまう。

					 歩損する指し手は延長しないようにするほうがいいか？
					 captureの時の歩損は、歩で取る、同角、同角みたいな局面なのでそこにはあまり意味なさげ。
			*/

			if (!pos.see_ge(move, -74))
                continue;
        }

		// -----------------------
        // Step 7. Make and search the move
        // Step 7. 指し手で進め探索する
        // -----------------------

		// 📝 1手動かして、再帰的にqsearch()を呼ぶ

        do_move(pos, move, st, givesCheck, ss);

        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move(pos, move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

        // -----------------------
		// Step 8. Check for a new best move
        // Step 8. 新しいbest moveをチェックする
        // -----------------------

		// bestValue(≒alpha値)を更新するのか

        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
							 // fail-highの場合もPVは更新する。
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                                   // alpha値の更新はこのタイミングで良い。
				                   // 💡 なぜなら、このタイミング以外だと枝刈りされるから。(else以下を読むこと)
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

	// -----------------------
    // Step 9. Check for mate
    // Step 9. 詰みの確認
    // -----------------------

	// All legal moves have been searched. A special case: if we are
    // in check and no legal moves were found, it is checkmate.

	// すべての合法手を探索しました。特別なケース: もし現在王手を受けていて、
    // かつ合法手が見つからなかった場合、それはチェックメイト（詰み）です。

	/*
		📓 王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである。
		    どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
		    わかるわけだし、そもそもこのnodeが詰みだとわかるとこのnodeに再訪問する確率は極めて低く、
		    置換表に保存しても置換表を汚すだけでほとんど得をしない。(レアケースなのでほとんど損もしないが)
		 
		    ※　計測したところ、置換表に保存したほうがわずかに強かったが、有意差ではなさげだし、
		    Stockfish10のコードが保存しないコードになっているので保存しないことにする。

		📊 【計測資料 26.】 qsearchで詰みのときに置換表に保存する/しない。

		📝  チェスでは王手されていて、合法手がない時に詰みだが、将棋では、合法手がなければ詰みなので ss->inCheckの条件は不要かと思ったら、
		     qsearch()で王手されていない時は、captures(駒を捕獲する指し手)とchecks(王手の指し手)の指し手しか生成していないから、
		     moveCount==0だから詰みとは限らない。
	
			 王手されている局面なら、evasion(王手回避手)を生成するから、moveCount==0なら詰みと確定する。
			 しかし置換表にhitした時にはbestValueの初期値は -VALUE_INFINITEではないので、そう考えると
			 ここは(Stockfishのコードのように)bestValue == -VALUE_INFINITEとするのではなくmoveCount == 0としたほうが良いように思うのだが…。
			 →　置換表にhitしたのに枝刈りがなされていない時点で有効手があるわけで詰みではないことは言えるのか…。
			 cf. https://yaneuraou.yaneu.com/2022/04/22/yaneuraous-qsearch-is-buggy/

		🤔
			 if (ss->inCheck && bestValue == -VALUE_INFINITE)
			↑Stockfishのコード。↓こう変更したほうが良いように思うが計測してみると大差ない。
			 Stockfishも12年前は↑ではなく↓この書き方だったようだ。moveCountが除去された時に変更されてしまったようだ。
			 cf. https://github.com/official-stockfish/Stockfish/commit/452f0d16966e0ec48385442362c94a810feaacd9
			 moveCountが再度導入されたからには、Stockfishもここは、↓の書き方に戻したほうが良いと思う。
	*/


#if STOCKFISH        
	if (ss->inCheck && bestValue == -VALUE_INFINITE)
    {
        assert(!MoveList<LEGAL>(pos).size());
#else
	if (ss->inCheck && moveCount == 0)
    {
		// 💡 合法手は存在しないはずだから指し手生成してもすぐに終わるだろうから
        //     このassertはそんなに遅くはない。
        ASSERT_LV5(!MoveList<LEGAL_ALL>(pos).size());
#endif

		return mated_in(ss->ply);  // Plies to mate from the root
                                   // rootから詰みまでの手数。
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;

	// 💡 盤面にkingとpawnしか残ってないときに特化したstalemate判定。
	//     将棋では用いない。
#if STOCKFISH
    Color us = pos.side_to_move();
    if (!ss->inCheck && !moveCount && !pos.non_pawn_material(us)
        && type_of(pos.captured_piece()) >= ROOK)
    {
        if (!((us == WHITE ? shift<NORTH>(pos.pieces(us, PAWN))
                           : shift<SOUTH>(pos.pieces(us, PAWN)))
              & ~pos.pieces()))  // no pawn pushes available
        {
            pos.state()->checkersBB = Rank1BB;  // search for legal king-moves only
            if (!MoveList<LEGAL>(pos).size())   // stalemate
                bestValue = VALUE_DRAW;
            pos.state()->checkersBB = 0;
        }
    }
#endif

    // Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.

	// 収集した情報をトランスポジションテーブルに保存します。
    // 静的評価は、修正履歴による調整が行われる前の状態で保存されます。

	// 📝 詰みではなかったのでこの情報を書き出す。
    //   　qsearch()の結果は信用ならないのでBOUND_EXACTで書き出すことはない。

    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());

	/*
		📓 置換表には abs(value) < VALUE_INFINITEの値しか書き込まないし、この関数もこの範囲の値しか返さない。
		   しかし置換表が衝突した場合はそうではない。3手詰めの局面で、置換表衝突により1手詰めのスコアが
		   返ってきた場合がそれである。
	
		    ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));
		    ⇨ このnodeはrootからss->ply手進めた局面なのでここでss->plyより短い詰みがあるのはおかしいが、
		    この関数はそんな値を返してしまう。しかしこれは通常探索ならば次のnodeでの
		    mate distance pruningで補正されるので問題ない。
		    また、VALUE_INFINITEはint16_tの最大値よりMAX_PLY以上小さいなのでオーバーフローの心配はない。
	
		  よってsearch(),qsearch()のassertは次のように書くべきである。
	*/

	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

    return bestValue;
}

// LMRのreductionの値を計算する。
Depth Search::YaneuraOuWorker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 731 / rootDelta + !i * reductionScale * 216 / 512 + 1089;
}

// 📝 やねうら王では、下記のelapsed(), elapsed_time()は用いない。
//     やねうら王はTimer classを持っているので、そちらを用いる。
#if STOCKFISH
// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.

// elapsed() は探索開始から経過した時間を返す。ただし、'nodestime' オプションが有効な場合、
// 探索したノード数を代わりに返す。この関数は、時間制限や探索ノード数などの
// 事前に定めた閾値に基づいて探索を停止すべきかを確認するために呼び出される。

// elapsed_time() returns the actual time elapsed since the start of the search.
// This function is intended for use only when printing PV outputs, and not used
// for making decisions within the search algorithm itself.

// elapsed_time() は探索開始から実際に経過した時間を返す。
// この関数はPV出力の表示時のみ使用され、探索アルゴリズム内の意思決定には使用されない。

TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }
#endif

Value Search::YaneuraOuWorker::evaluate(const Position& pos) {

#if defined(EVAL_SFNN)
	// 最新のStockfishのコード

    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);

#else
	return Eval::evaluate(pos);
#endif
}

namespace {

// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.

// メイトスコアまたはTB（テーブルベース）スコアを
// 「ルートからメイトまでの手数」から「現在の局面からメイトまでの手数」に調整する。
// 通常のスコアは変更しない。
// この関数は、トランスポジションテーブルに値を保存する前に呼び出される。

/*
	📓 詰みのスコアは置換表上は、このnodeからあと何手で詰むかというスコアを格納する。
	    しかし、search()の返し値は、rootからあと何手で詰むかというスコアを使っている。
	   (こうしておかないと、do_move(),undo_move()するごとに詰みのスコアをインクリメントしたりデクリメントしたり
	    しないといけなくなってとても面倒くさいからである。)

		なので置換表に格納する前に、この変換をしなければならない。
	    詰みにまつわるスコアでないなら関係がないので何の変換も行わない。

		ply : root node からの手数。
*/

Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }
// 🤔 これ足した結果がabs(x) < VALUE_INFINITEであることを確認すべきだと思う。

// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.

// value_to_tt() の逆の処理：トランスポジションテーブルから取得した
// メイトスコアまたはTB（テーブルベース）スコア（現在の局面からメイト/敗北までの手数を示す）を
// 「ルートからメイト/敗北（TB勝利/敗北）までの手数」に調整する。
// ただし、50手ルールやグラフ履歴との相互作用に関連する誤ったメイト/TBスコアを避けるため、
// 代わりに最大の非TBスコアを返す。

// 📓 value_to_tt()の逆関数
//     ply : root node からの手数。

Value value_from_tt(Value v, int ply
#if STOCKFISH
    , int r50c
#endif
) {

    if (!is_valid(v))
        return VALUE_NONE;

    // handle TB win or better
    if (is_win(v))
    {
		// 📌 将棋ではTablebase関係ないのでコメントアウト
#if STOCKFISH
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 /* - r50c */)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 /* - r50c */)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
#endif

        return v - ply;
    }

    // handle TB loss or worse
    if (is_loss(v))
    {
		// 📌 将棋ではTablebase関係ないのでコメントアウト

#if STOCKFISH
        // Downgrade a potentially false mate score.
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
#endif
        return v + ply;
    }

    return v;
}

// Adds current move and appends child pv[]
// 現在の指し手を追加し、子pv[] を連結する。

/*
	📓 PV lineをコピーする。
        pv に move(1手 現在の指し手) + childPv(複数手,末尾Move::none())をコピーする。
	    番兵として末尾はMove::none()にすることになっている。
*/

void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

// -----------------------
//     Statsのupdate
// -----------------------

// Updates stats at the end of search() when a bestMove is found
// update_all_stats()は、bestmoveが見つかったときにそのnodeの探索の終端で呼び出される。

/*
	統計情報一式を更新する。
	prevSq           : 直前の指し手の駒の移動先。直前の指し手がMove::none()の時はSQ_NONE
	quietsSearched   : このnodeで生成したquietな指し手、良い順
	capturesSearched : このnodeで生成したcaptureの指し手、良い順
*/

void update_all_stats(const Position&          pos,
                      Stack*                   ss,
                      Search::YaneuraOuWorker& workerThread,
                      Move                     bestMove,
                      Square                   prevSq,
                      SearchedList&            quietsSearched,
                      SearchedList&            capturesSearched,
                      Depth                    depth,
                      Move                     ttMove) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

#if STOCKFISH
    int bonus        = std::min(170 * depth - 87, 1598) + 332 * (bestMove == ttMove);
    int quietMalus   = std::min(743 * depth - 180, 2287) - 33 * quietsSearched.size();
    int captureMalus = std::min(708 * depth - 148, 2287) - 29 * capturesSearched.size();
#else
	// 🤔 size()はsize_tでintに代入すると警告が出るので修正しておく。
    int bonus        = std::min(170 * depth - 87, 1598) + 332 * (bestMove == ttMove);
    int quietMalus   = std::min(743 * depth - 180, 2287) - 33 * int(quietsSearched.size());
    int captureMalus = std::min(708 * depth - 148, 2287) - 29 * int(capturesSearched.size());
#endif
	/*
		📓 Stockfish 14ではcapture_or_promotion()からcapture()に変更された。[2022/3/23]
			Stockfish 16では、capture()からcapture_stage()に変更された。[2023/10/15]
	*/ 

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 978 / 1024);

        // Decrease stats for all non-best quiet moves
        // 最善でないquietの指し手すべての統計を減少させる

		for (Move move : quietsSearched)
            update_quiet_histories(pos, ss, workerThread, move, -quietMalus * 1115 / 1024);
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        // 最善手が捕獲する指し手だった場合、その統計を増加させる

        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.

	// quietな初期の手が、前の手でトランスポジションテーブル（TT）の手ではなく、
    // かつ反証された場合に追加のペナルティを与えます。

    // 📝 (ss-1)->ttHit : 一つ前のnodeで置換表にhitしたか
    // 💡 Move::null()の場合、Stockfishでは65(移動後の升がSQ_NONEであることを保証している。やねうら王もそう変更した。)

	if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      -captureMalus * 622 / 1024);

    // Decrease stats for all non-best capture moves
    // 最善の捕獲する指し手以外のすべての手の統計を減少させます

    for (Move move : capturesSearched)
    {
        /*
		   🤔 ここ、moved_piece_before() で、捕獲前の駒の価値で考えたほうがいいか？

		       → MovePickerでcaptureHistoryを用いる時に
			      moved_piece_afterの方で表引きしてるので、それに倣う必要がある。
		*/

        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -captureMalus * 1431 / 1024;
    }
}


// Updates histories of the move pairs formed by moves
// at ply -1, -2, -3, -4, and -6 with current move.

// update_continuation_histories() は、形成された手のペアの履歴を更新します。
// 1,2,4,6手前の指し手と現在の指し手との指し手ペアによってcontinuationHistoryを更新する。

/*
	📓 1手前に対する現在の指し手 ≒ counterMove  (応手)
		2手前に対する現在の指し手 ≒ followupMove (継続手)
		4手前に対する現在の指し手 ≒ followupMove (継続手)
		⇨　Stockfish 10で6手前も見るようになった。
		⇨　Stockfish 16で3手前も見るようになった。
*/
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 1108}, {2, 652}, {3, 273}, {4, 572}, {5, 126}, {6, 449}}};

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        // 王手がかかっている場合のみ、最初の2つのcontinuation historiesを更新する

        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << (bonus * weight / 1024) + 80 * (i < 2);
    }
}

// Updates move sorting heuristics
// 手のソートのヒューリスティックを更新します

// 💡 新しいbest moveが見つかったときに指し手の並べ替えheuristicsを更新する。
//     具体的には駒を取らない指し手のstat tables等を更新する。

// 📝  move      : これが良かった指し手

void update_quiet_histories(
  const Position& pos, Stack* ss, Search::YaneuraOuWorker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;  // Untuned to prevent duplicate effort
	                                                        // 重複した処理を防ぐためにチューニングされていない

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
	    workerThread.lowPlyHistory[ss->ply][move.from_to()] << (bonus * 771 / 1024) + 40;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(),
                                  bonus * (bonus > 0 ? 979 : 842) / 1024);

    int pIndex = pawn_history_index(pos);
    workerThread.pawnHistory[pIndex][pos.moved_piece_after(move)][move.to_sq()]
      << (bonus * (bonus > 0 ? 704 : 439) / 1024) + 70;
}

} // namespace

// 📌 Skill class、やねうら王ではサポートしない。
#if STOCKFISH
// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'. Idea by Heinz van Saanen.

// 手加減が有効であるなら、best moveを'level'に依存する統計ルールに基づくRootMovesの集合から選ぶ。
// Heinz van Saanenのアイデア。
Move Skill::pick_best(const RootMoves& rootMoves, size_t multiPV) {
    static PRNG rng(now());  // PRNG sequence should be non-deterministic
							 // 乱数ジェネレーターは非決定的であるべき。

    // RootMoves are already sorted by score in descending order
	// RootMovesはすでにscoreで降順にソートされている。

	Value  topScore = rootMoves[0].score;
    int    delta    = std::min(topScore - rootMoves[multiPV - 1].score, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms, both dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one is
    // random. Then we choose the move with the resulting highest score.

	// best moveを選ぶ。それぞれの指し手に対して弱さに依存する2つのterm(用語)を追加する。
	// 1つは、決定的で、弱いレベルでは大きくなるもので、1つはランダムである。
	// 次に得点がもっとも高い指し手を選択する。

	for (size_t i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
		// これが魔法の公式

		int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}
#endif


// Used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.

// デバッグ情報の出力、そしてより重要なのは、
// 利用可能な時間を使い切ったことを検出し、探索を停止するために使われる。

void SearchManager::check_time(Search::YaneuraOuWorker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    // ノード数を基準にする場合、チェック頻度がノード数の0.1%未満にならないようにする

    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    // GUIから指示があるまで、ポンダリングを停止すべきではない
    // 💡 ponderフラグが立っていたら、"go ponder"の最中なので
    //     "stop"か"ponderhit"が来るまでは停止する必要がない。

    if (ponder)
        return;

	if (
    // Later we rely on the fact that we can at least use the mainthread previous
    // root-search score and PV in a multithreaded environment to prove mated-in scores.

    // 後で、少なくともメインスレッドの直前の
    // ルート探索のスコアとPVをマルチスレッド環境で利用して、
    // 詰みのスコアを証明できるという事実に依存している。

    /*
				📓 time managementを行う時に、

					1. 今回使える最大時間(tm.maximum())を超過
					2. stopOnPonderhitがtrue
					3. movetimeを超過
					4. nodesが指定されていてそれを超過
					5. 🌈 終了時刻(tm.search_end)がTimeManagementによって設定されていてそれを超過

					しているなら探索を即座に停止させる。

				やねうら王の場合、3,4,5 は即座に停止させていいと思う。
				1. は、ponderhitしている場合、"go"からtm.maximum()を超過しているならば思考終了させて良いが、
				   そのとき、秒ぎりぎりまで思考したほうが得。(つまり、即座に停止させるのは少し損。)
				2.のケースも、秒ぎりぎりまでは思考したほうが得。

				🤔 ponderhitした時刻から計算して、秒単位で切り上げた時刻まで思考させたい。
			*/

#if STOCKFISH
      worker.completedDepth >= 1
      && ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit)) || (worker.limits.movetime && elapsed >= worker.limits.movetime)
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = worker.threads.abortedSearch = true;
#else
      worker.completedDepth >= 1)
    {
        if (
          // 3.
          (worker.limits.movetime && elapsed >= worker.limits.movetime)
          // 4.
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)
          // 5.
          || (tm.search_end && tm.search_end <= elapsed))

            worker.threads.stop = worker.threads.abortedSearch = true;

        else if (!tm.search_end && worker.limits.use_time_management() &&
				 // 1.
				 (elapsed > tm.maximum()
				 // 2
				  || stopOnPonderhit))
            tm.set_search_end(elapsed);
    }


#endif
}

// 📌 Tablebase関係の処理。将棋では用いないのでコメントアウト。
#if STOCKFISH
// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
// Keeps the search based PV for as long as it is verified to maintain the game
// outcome, truncates afterwards. Finally, extends to mate the PV, providing a
// possible continuation (but not a proven mating line).
void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Position&                 pos,
                      RootMove&                 rootMove,
                      Value&                    v) {

    auto t_start      = std::chrono::steady_clock::now();
    int  moveOverhead = int(options["Move Overhead"]);
    bool rule50       = bool(options["Syzygy50MoveRule"]);

    // Do not use more than moveOverhead / 2 time, if time management is active
    auto time_abort = [&t_start, &moveOverhead, &limits]() -> bool {
        auto t_end = std::chrono::steady_clock::now();
        return limits.use_time_management()
            && 2 * std::chrono::duration<double, std::milli>(t_end - t_start).count()
                 > moveOverhead;
    };

    std::list<StateInfo> sts;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB.
    auto& stRoot = sts.emplace_back();
    pos.do_move(rootMove.pv[0], stRoot);
    int ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive score
    while (size_t(ply) < rootMove.pv.size())
    {
        Move& pvMove = rootMove.pv[ply];

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
            legalMoves.emplace_back(m);

        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves);
        RootMove&          rm     = *std::find(legalMoves.begin(), legalMoves.end(), pvMove);

        if (legalMoves[0].tbRank != rm.tbRank)
            break;

        ply++;

        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);

        // Do not allow for repetitions or drawing moves along the PV in TB regime
        if (config.rootInTB && ((rule50 && pos.is_draw(ply)) || pos.is_repetition(ply)))
        {
            pos.undo_move(pvMove);
            ply--;
            break;
        }

        // Full PV shown will thus be validated and end in TB.
        // If we cannot validate the full PV in time, we do not show it.
        if (config.rootInTB && time_abort())
            break;
    }

    // Resize the PV to the correct part
    rootMove.pv.resize(ply);

    // Step 2, now extend the PV to mate, as if the user explored syzygy-tables.info
    // using top ranked moves (minimal DTZ), which gives optimal mates only for simple
    // endgames e.g. KRvK.
    while (!(rule50 && pos.is_draw(0)))
    {
        if (time_abort())
            break;

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
        {
            auto&     rm = legalMoves.emplace_back(m);
            StateInfo tmpSI;
            pos.do_move(m, tmpSI);
            // Give a score of each move to break DTZ ties restricting opponent mobility,
            // but not giving the opponent a capture.
            for (const auto& mOpp : MoveList<LEGAL>(pos))
                rm.tbRank -= pos.capture(mOpp) ? 100 : 1;
            pos.undo_move(m);
        }

        // Mate found
        if (legalMoves.size() == 0)
            break;

        // Sort moves according to their above assigned rank.
        // This will break ties for moves with equal DTZ in rank_root_moves.
        std::stable_sort(
          legalMoves.begin(), legalMoves.end(),
          [](const Search::RootMove& a, const Search::RootMove& b) { return a.tbRank > b.tbRank; });

        // The winning side tries to minimize DTZ, the losing side maximizes it
        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves, true);

        // If DTZ is not available we might not find a mate, so we bail out
        if (!config.rootInTB || config.cardinality > 0)
            break;

        ply++;

        Move& pvMove = legalMoves[0].pv[0];
        rootMove.pv.push_back(pvMove);
        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);
    }

    // Finding a draw in this function is an exceptional case, that cannot happen when rule50 is false or
    // during engine game play, since we have a winning score, and play correctly
    // with TB support. However, it can be that a position is draw due to the 50 move
    // rule if it has been been reached on the board with a non-optimal 50 move counter
    // (e.g. 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter rounding
    // cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score can be
    // displayed if the engine did not find a drawing move yet, but eventually search
    // will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w - - 96 1 )
    if (pos.is_draw(0))
        v = VALUE_DRAW;

    // Undo the PV moves
    for (auto it = rootMove.pv.rbegin(); it != rootMove.pv.rend(); ++it)
        pos.undo_move(*it);

    // Inform if we couldn't get a full extension in time
    if (time_abort())
        sync_cout
          << "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
          << sync_endl;
}
#endif


void SearchManager::pv(Search::YaneuraOuWorker&  worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) {

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    size_t     pvIdx     = worker.pvIdx;
    size_t     multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
#if STOCKFISH
    uint64_t tbHits = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);
#endif

    for (size_t i = 0; i < multiPV; ++i)
    {
        bool updated = rootMoves[i].score != -VALUE_INFINITE;

        if (depth == 1 && !updated && i > 0)
            continue;

        Depth d = updated ? depth : std::max(1, depth - 1);
        Value v = updated ? rootMoves[i].uciScore : rootMoves[i].previousScore;

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

#if STOCKFISH
        bool tb = worker.tbConfig.rootInTB && std::abs(v) <= VALUE_TB;
        v       = tb ? rootMoves[i].tbScore : v;
#endif

#if STOCKFISH
        bool isExact = i != pvIdx || tb || !updated;  // tablebase- and previous-scores are exact

        // Potentially correct and extend the PV, and in exceptional cases v
        // 必要に応じてPVを修正・延長し、例外的な場合にはvも処理する

        if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
            && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);
#else
        bool isExact =
          i != pvIdx /* || tb */ || !updated;  // tablebase- and previous-scores are exact
                                               // tablebaseのスコアおよび以前のスコアは正確である

#endif

        std::string pv;
#if STOCKFISH
        for (Move m : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";
#else
		// 🌈 やねうら王では、consideration_modeのときは置換表からPVをかき集める。
        if (worker.main_manager()->search_options.consideration_mode)
        {
            Move      moves[MAX_PLY + 1];
            StateInfo si[MAX_PLY];
            int       ply = 0;

            while (ply < MAX_PLY)
            {
                // 千日手はそこで終了。ただし初手はPVを出力。
                // 千日手がベストのとき、置換表を更新していないので
                // 置換表上はMove::none()がベストの指し手になっている可能性があるので早めに検出する。
                auto rep = pos.is_repetition(ply);
                if (rep != REPETITION_NONE && ply >= 1)
                {
                    // 千日手でPVを打ち切るときはその旨を表示
                    pv += to_usi_string(rep) + ' ';
                    break;
                }

                Move m;

                // まず、rootMoves.pvを辿れるところまで辿る。
                // rootMoves[i].pv[0]は宣言勝ちの指し手(Move::win())の可能性があるので注意。
                if (ply < int(rootMoves[i].pv.size()))
                    m = rootMoves[i].pv[ply];
                else
                {
                    // 次の手を置換表から拾う。
                    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
                    // 置換表になかった
                    if (!ttHit)
                        break;

                    m = ttData.move;

                    // leaf nodeはわりと高い確率でMove::none()
                    if (m == Move::none())
                        break;

                    // 置換表にはpseudo_legalではない指し手が含まれるのでそれを弾く。
                    if (!(pos.pseudo_legal_s<true>(m) && pos.legal(m)))
                        break;
                }
                // leaf node末尾にMove::resign()があることはないが、
                // 詰み局面で呼び出されると1手先がmove resignなので、これでdo_move()するのは
                // 非合法だから、do_move()せずにループを抜ける。
                if (!m.is_ok())
                {
                    pv += USIEngine::move(m) + ' ';
                    break;
                }

                moves[ply] = m;
                pv += USIEngine::move(m) + ' ';

                pos.do_move(m, si[ply]);
                ++ply;
            }
            while (ply > 0)
                pos.undo_move(moves[--ply]);
		}
        else
            for (Move m : rootMoves[i].pv)
                pv += USIEngine::move(m) + ' ';
#endif

        // Remove last whitespace
        // 最後の空白を削除する

        if (!pv.empty())
            pv.pop_back();

#if STOCKFISH
        auto wdl = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
#endif
        auto bound = rootMoves[i].scoreLowerbound
                     ? "lowerbound"
                     : (rootMoves[i].scoreUpperbound ? "upperbound" : "");

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
#if STOCKFISH
        info.score = {v, pos};  // 📝 StockfishではValue,Position&からScore型に変換する。
        info.wdl   = wdl;
#else
        info.score = v;
#endif

        if (!isExact)
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
#if STOCKFISH
        info.tbHits = tbHits;
#endif
        info.pv       = pv;
        info.hashfull = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI,
// otherwise in case of 'ponder on' we have nothing to think about.

// 探索を終了する前にponder moveがない場合に呼び出されます。
// 例えば、rootでfail highが発生して探索を中断した場合などです。
// GUIに返すponder moveをできる限り準備しようとしますが、
// そうでない場合、「ponder on」の際に考えるべきものが何もなくなります。

bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt,
                                              Position&                 pos,
                                              Move                      ponder_candidate) {
    StateInfo st;

    ASSERT_LV3(pv.size() == 1);

    // 💡 Stockfishでは if (pv[0] == Move::none()) となっているが、
    //     詰みの局面が"ponderhit"で返ってくることがあるので、
    //     ここでのpv[0] == Move::resign()であることがありうる。
    //     だから、やねうら王では、ここは、is_ok()で判定する。

    if (!pv[0].is_ok())
        return false;

    pos.do_move(pv[0], st);

    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
    if (ttHit)
    {
        Move m = ttData.move;
        //if (MoveList<LEGAL>(pos).contains(ttData.move))
        // ⇨ Stockfishのこのコード、pseudo_legalとlegalで十分なのではないか？
        if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
            pv.push_back(m);
    }
    // 置換表にもなかったので以前のiteration時のpv[1]をほじくり返す。
    // 🌈 やねうら王独自改良
    else if (ponder_candidate)
    {
        Move m = ponder_candidate;
        if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
            pv.push_back(m);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

}  // namespace YaneuraOu

using namespace YaneuraOu;
namespace {

// 自作のエンジンのentry point
void engine_main() {

	// ここで作ったエンジン
    YaneuraOuEngine engine;

    // USIコマンドの応答部
    USIEngine usi;
    usi.set_engine(engine);  // エンジン実装を差し替える。

    // USIコマンドの応答のためのループ
    usi.loop();
}

// このentry pointを登録しておく。
static EngineFuncRegister r(engine_main, "YaneuraOuEngine", 0);
}

#endif // YANEURAOU_ENGINE
