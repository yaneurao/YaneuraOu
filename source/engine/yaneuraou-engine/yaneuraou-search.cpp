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

namespace YaneuraOu {

using namespace Search;

// -------------------
// やねうら王独自追加
// -------------------

// パラメーターの調整を行うのか
#if defined(TUNING_SEARCH_PARAMETERS)
    // ハイパーパラメーターを調整するときは終了時にその時のパラメーターを書き出す。
    #define ENABLE_OUTPUT_GAME_RESULT

    // パラメーターをランダムに少し変化させる。
    // 探索パラメーターにstep分のランダム値を加えて対戦させるとき用。
    // 試合が終わったときに勝敗と、そのときに用いたパラメーター一覧をファイルに出力する。
    #define USE_RANDOM_PARAMETERS

    #define PARAM_DEFINE int
    #include "yaneuraou-param.h"
#else
    // 変更しないとき
    #define PARAM_DEFINE constexpr int
    #include "yaneuraou-param.h"

#endif

// 実行時に読み込むパラメーターファイルを配置するフォルダとその名前
#define PARAM_FILE "param/yaneuraou-param.h"

#if defined(ENABLE_OUTPUT_GAME_RESULT)
// 変更したパラメーター一覧と、リザルト(勝敗)を書き出すためのファイルハンドル
static std::fstream result_log;
#endif


// 探索パラメーターの自動調整のためのフレームワーク。
// これ、今後使うかどうかわからないので、いったんコメントアウト。
// TODO : あとで
#if 0

// パラメーターのランダム化のときには、
// USIの"gameover"コマンドに対して、それをログに書き出す。
void gameover_handler([[maybe_unused]] const std::string& cmd)
{
#if defined(ENABLE_OUTPUT_GAME_RESULT)
	result_log << cmd << std::endl << std::flush;
#endif
}

#if defined(YANEURAOU_ENGINE_NNUE)
void init_fv_scale() {
	Eval::NNUE::FV_SCALE = (int)Options["FV_SCALE"];
}
#endif

// 探索パラメーターを動的に読み込む機能。
void init_param()
{
	// -----------------------
	//   parameters.hの動的な読み込み
	// -----------------------

        #if defined(TUNING_SEARCH_PARAMETERS)
	{
		std::vector<std::string> param_names = {
			// このheader fileは、yaneuraou-param.hからparam_conv.pyによって自動生成される。
            #include "param/yaneuraou-param-string.h"
		};

		std::vector<int*> param_vars = {
			// このheader fileは、yaneuraou-param.hからparam_conv.pyによって自動生成される。
            #include "param/yaneuraou-param-array.h"
		};

		std::fstream fs;

		std::string path = Path::Combine(Directory::GetCurrentFolder(), PARAM_FILE);

		fs.open(path.c_str(), std::ios::in);
		if (fs.fail())
		{
			std::cout << "info string Error! : can't read " << path << std::endl;
			return;
		}

		size_t count = 0;
		std::string line, last_line;

		// bufのなかにある部分文字列strの右側にある数値を読む。
		auto get_num = [](const std::string& buf, const std::string& str)
			{
				auto pos = buf.find(str);
				ASSERT_LV3(pos != std::string::npos);

				auto s = buf.substr(pos + str.size());
				if (s.empty() || !(('0' <= s[0] && s[0] <= '9') || s[0] == '-' || s[0] == ' '))
				{
					std::cout << "Error : Parse Error " << buf << "   ==>   " << s << std::endl;
					return 0;
				}

				return stoi(s);
				// ここで落ちてたら、paramファイルとして、変な文をparseしている。
			};

		std::vector<bool> founds(param_vars.size());

		while (!fs.eof())
		{
			getline(fs, line);
			if (line.find("PARAM_DEFINE") != std::string::npos)
			{
				for (size_t i = 0; i < param_names.size(); ++i)
				{
					auto pos = line.find(param_names[i]);
					if (pos != std::string::npos)
					{
						char c = line[pos + param_names[i].size()];
						// ここ、パラメーター名のあと、スペースか"="か来るのを確認しておかないと
						// "PARAM_T1" が "PARAM_T10" に誤爆する。
						if (!(c == '\t' || c == ' ' || c == '='))
							continue;

						count++;

						// "="の右側にある数値を読む。
						*param_vars[i] = get_num(line, "=");

						// 見つかった
						founds[i] = true;

#if defined(USE_RANDOM_PARAMETERS)
						// PARAM_DEFINEの一つ前の行には次のように書いてあるはずなので、
						// USE_RANDOM_PARAMETERSのときは、このstepをプラスかマイナス方向に加算してやる。
						// ただし、fixedと書いてあるパラメーターに関しては除外する。
						// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。

						// [PARAM] min:100,max:240,step:3,interval:1,time_rate:1,fixed

						// "fixed"と書かれているパラメーターはないものとして扱う。
						if (last_line.find("fixed") != std::string::npos)
						{
							param_names[i] = "FIXED";
							goto NEXT;
						}

						static PRNG rand;
						int param_step = get_num(last_line, "step:");
						int param_min = get_num(last_line, "min:");
						int param_max = get_num(last_line, "max:");
						int param_interval = get_num(last_line, "interval:");

						// 現在の値
						int v = *param_vars[i];

						// とりうる値の候補
						std::vector<int> a;

						for (int j = 0; j <= param_interval; ++j)
						{
							// j==0のときは同じ値であり、これはのちに除外される。
							a.push_back(std::max(v - param_step * j, param_min));
							a.push_back(std::min(v + param_step * j, param_max));
						}

						// 重複除去。
						// 1) std::unique()は隣接要素しか削除しないので事前にソートしている。
						// 2) std::unique()では末尾にゴミが残るのでそれをerase()で消している。
						std::sort(a.begin(), a.end());
						a.erase(std::unique(a.begin(), a.end()), a.end());

						// 残ったものから1つをランダムに選択
						if (a.size() == 0)
						{
							std::cout << "Error : param is out of range -> " << line << std::endl;
						}
						else {
							*param_vars[i] = a[rand.rand(a.size())];
						}
#endif

						//            cout << param_names[i] << " = " << *param_vars[i] << endl;
						goto NEXT;
					}
				}
				std::cout << "Error : param not found! in yaneuraou-param.h -> " << line << std::endl;

			NEXT:;
			}
			last_line = line; // 1つ前の行を記憶しておく。
		}
		fs.close();

		// 読み込んだパラメーターの数が合致しないといけない。
		// 見つかっていなかったパラメーターを表示させる。
		if (count != param_names.size())
		{
			for (size_t i = 0; i < founds.size(); ++i)
				if (!founds[i])
					std::cout << "Error : param not found in " << path << " -> " << param_names[i] << std::endl;
		}

#if defined(ENABLE_OUTPUT_GAME_RESULT)
		{
			if (!result_log.is_open())
				result_log.open(Options["PARAMETERS_LOG_FILE_PATH"], std::ios::app);
			// 今回のパラメーターをログファイルに書き出す。
			for (size_t i = 0; i < param_names.size(); ++i)
			{
				if (param_names[i] == "FIXED")
					continue;

				result_log << param_names[i] << ":" << *param_vars[i] << ",";
			}
			result_log << std::endl << std::flush;
		}
#endif

		// Evalのパラメーター初期化
		// 上のコードでパラメーターが変更された可能性があるのでこのタイミングで再度呼び出す。
		Eval::init();
	}

#endif
}
#endif

// 思考エンジンの追加オプションを設定する。
// 💡 Stockfishでは、Engine::Engine()で行っている。
void YaneuraOuEngine::add_options() {
    // 📌　定跡設定

    book.init(options);

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

    options.add("DrawValueBlack", Option(-2, -30000, 30000));
    options.add("DrawValueWhite", Option(-2, -30000, 30000));

    //  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
    options.add("PvInterval", Option(300, 0, 100000000));

    // 投了スコア
    options.add("ResignValue", Option(99999, 0, 99999));

    //
    //   パラメーターの外部からの自動調整
    //

    #if defined(EVAL_LEARN)
    // 評価関数の学習を行なうときは、評価関数の保存先のフォルダを変更できる。
    // デフォルトではevalsave。このフォルダは事前に用意されているものとする。
    // このフォルダ配下にフォルダを"0/","1/",…のように自動的に掘り、そこに評価関数ファイルを保存する。
    options.add("EvalSaveDir", Option("evalsave"));
    #endif

    #if defined(ENABLE_OUTPUT_GAME_RESULT)

        #if defined(TUNING_SEARCH_PARAMETERS)
    sync_cout << "info string warning!! TUNING_SEARCH_PARAMETERS." << sync_endl;
        #elif defined(USE_RANDOM_PARAMETERS)
    sync_cout << "info string warning!! USE_RANDOM_PARAMETERS." << sync_endl;
        #else
    sync_cout << "info string warning!! ENABLE_OUTPUT_GAME_RESULT." << sync_endl;
        #endif

    // パラメーターのログの保存先のfile path
        options.add("PARAMETERS_LOG_FILE_PATH"] , Option("param_log.txt"));
    #endif

        // 検討モード用のPVを出力するモード
        options.add("ConsiderationMode", Option(true));

        // fail low/highのときにPVを出力するかどうか。
        options.add("OutputFailLHPV", Option(true));

    #if defined(YANEURAOU_ENGINE_NNUE)
        // NNUEのFV_SCALEの値
        options.add("FV_SCALE", Option(16, 1, 128));
    #endif


        // 📌 Stockfishにはあるが、やねうら王ではサポートしないオプション
    #if 0
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
}

// "isready"のタイミングでの初期化処理。
void YaneuraOuEngine::isready() {

	// 📌 やねうら王独自オプションの内容を設定などに反映させる。

	// 検討モード用のPVを出力するのか。
    global_options.consideration_mode = options["ConsiderationMode"];

    // fail low/highのときにPVを出力するかどうか。
    global_options.outout_fail_lh_pv  = options["OutputFailLHPV"];

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
        auto worker = dynamic_cast<Search::YaneuraOuWorker*>(th->worker.get());
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
    return dynamic_cast<YaneuraOuWorker*>(bestThread->worker.get());
}

// -----------------------------------------------------------
// 📌 ここからStockfishのsearch.cppを参考にしながら書く。 📌
// -----------------------------------------------------------

//namespace YaneuraOu /*Stockfish*/ {

    // 💡 将棋では、Tablebasesは用いないのでコメントアウト。
    #if 0
namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap& options,
	const Search::LimitsType& limits,
	Stockfish::Position& pos,
	Stockfish::Search::RootMove& rootMove,
	Value& v);
    #endif

//using namespace Search;
// 💡 冒頭で書いたのでコメントアウト。

namespace {

// -----------------------
//   探索用の評価値補整
// -----------------------

// 💡 あるnodeで生成した指し手にbonusを与えるために、そのnodeで生成した指し手を良い順に保存しておく配列のcapacity。
constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;


// TODO : correction historyについては、のちほど導入を検討する。
    #if 0
// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

	int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
		const Color us = pos.side_to_move();
		const auto  m = (ss - 1)->currentMove;
		const auto  pcv = w.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us];
		const auto  micv = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
		const auto  wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
		const auto  bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
		const auto  cntcv =
			m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
			: 0;

		return 7696 * pcv + 7689 * micv + 9708 * (wnpcv + bnpcv) + 6978 * cntcv;
	}

	// Add correctionHistory value to raw staticEval and guarantee evaluation
	// does not hit the tablebase range.
	Value to_corrected_static_eval(const Value v, const int cv) {
		return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
	}

	void update_correction_history(const Position& pos,
		Stack* const    ss,
		Search::Worker& workerThread,
		const int       bonus) {
		const Move  m = (ss - 1)->currentMove;
		const Color us = pos.side_to_move();

		static constexpr int nonPawnWeight = 172;

		workerThread.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us]
			<< bonus * 111 / 128;
		workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 151 / 128;
		workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us]
			<< bonus * nonPawnWeight / 128;
		workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us]
			<< bonus * nonPawnWeight / 128;

		if (m.is_ok())
			(*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
			<< bonus * 141 / 128;
	}
}
    #endif

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
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);

// 📝 32は、quietsSearched、quietsSearchedの最大数。そのnodeで生成したQUIETS/CAPTURESの指し手を良い順に保持してある。
//     bonusを加点するときにこれらの指し手に対して行う。
//     Stockfish 16ではquietsSearchedの配列サイズが[64]から[32]になった。
//     将棋ではハズレのquietの指し手が大量にあるので、それがベストとは限らない。
// 🌈　比較したところ、64より32の方がわずかに良かったので、とりあえず32にしておく。(V7.73mとV7.73m2との比較)

void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            TTMove,
                      int             moveCount);


}  // namespace

// 💡 やねうら王では、Workerを派生させて書くことにしたので、このコードは、派生classであるYaneuraOuWorkerのコンストラクタで書く。
    #if 0
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

    // 💡 Worker::clear()が呼び出される。
    clear();
}

void Search::YaneuraOuWorker::ensure_network_replicated() {
	// TODO : あとで評価関数を実装する。
	#if 0
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
	#endif
}

void Search::YaneuraOuWorker::start_searching() {

    // TODO : あとで
    //accumulatorStack.reset();

    // Non-main threads go directly to iterative_deepening()
    // メインスレッド以外は直接 iterative_deepening() へ進む

    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    // 📌 今回の思考時間の設定。
    //     これは、ponderhitした時にponderhitにパラメーターが付随していれば
    //     再計算するする必要性があるので、いずれにせよ呼び出しておく必要がある。
    // 💡 やねうら王では、originalTimeAdjustは用いない。

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options
                            /*  , main_manager()->originalTimeAdjust */);

    // 📌 置換表のTTEntryの世代を進める。
    // 📝 sub threadが動く前であるこのタイミングで置換表の世代を進めるべきである。
    //     cf. Call TT.new_search() earlier. : https://github.com/official-stockfish/Stockfish/commit/ebc563059c5fc103ca6d79edb04bb6d5f182eaf5

    // 置換表の世代カウンターを進める(クリアではない)
    tt.new_search();

    // 📌 やねうら王固有の初期化 📌

    // PVが詰まるのを抑制するために、前回出力時刻を記録しておく。
    main_manager()->lastPvInfoTime = 0;

    // PVの出力間隔[ms]
    // go infiniteはShogiGUIなどの検討モードで動作させていると考えられるので
    // この場合は、PVを毎回出力しないと読み筋が出力されないことがある。
    global_options.pv_interval =
      (limits.infinite || global_options.consideration_mode) ? 0 : (int) options["PvInterval"];

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

    // ponder用の指し手の初期化
    // やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのPV上の(相手の)指し手を用いるという独自仕様。
    // Stockfish本家もこうするべきだと思う。
    main_manager()->ponder_candidate = Move::none();


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

    auto their_king = rootPos.king_square(~us);
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

        if (rootMoves.size() == 0)
            rootMoves.emplace_back(m);
        else
            rootMoves[0].pv[0] = m;

        rootMoves[0].score = rootMoves[0].usiScore = mate_in(1);

        goto SKIP_SEARCH;
    }
#endif

    // ✋ 独自追加ここまで。

    // ---------------------
    // 合法手がないならここで投了
    // ---------------------

    if (rootMoves.empty())
    {
        // rootで指し手がない = (将棋だと)詰みの局面である

        // 💡 投了の指し手と評価値をrootMoves[0]に積んでおけばUSI::pv()が良きに計らってくれる。
        //     読み筋にresignと出力されるが、将棋所、ShogiGUIともにバグらないのでこれで良しとする。
        rootMoves.emplace_back(Move::none());

        //main_manager()->updates.onUpdateNoMoves(
        //  {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
        // 💡 チェスだと王手されていないなら引き分けだが、将棋だとつねに負け。
        main_manager()->updates.onUpdateNoMoves({0, -VALUE_MATE});

// TODO : あとで考える。
#if 0
		// 📌 やねうら王独自
		// 評価値を用いないなら代入しなくて良いのだが(Stockfishはそうなっている)、
        // このあと、↓USI::pv()を呼び出したいので、scoreをきちんと設定しておいてやる。
        rootMoves[0].score = rootMoves[0].usiScore = mated_in(0);
#endif

        goto SKIP_SEARCH;
    }

    // ---------------------
    //     定跡の選択部
    // ---------------------

    if (engine.book.probe(rootMoves, limits))
        goto SKIP_SEARCH;

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
            // root movesの集合に突っ込んであるはず。
            // このときMultiPVが利かないが、ここ真面目にMultiPVして指し手を返すのは
            // プログラムがくちゃくちゃになるのでいまはこれは仕様としておく。

            // トライルールのとき、その指し手がgoコマンドで指定された指し手集合に含まれることを
            // 保証しないといけないのでrootMovesのなかにこの指し手が見つからないなら指すわけにはいかない。

            // 入玉宣言の条件を満たしているときは、
            // goコマンドを処理したあとのthreads.cppでMove::win()は追加されているはず。

            auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
            if (it_move != rootMoves.end())
            {
                std::swap(rootMoves[0], *it_move);

                // 1手詰めのときのスコアにしておく。
                rootMoves[0].score = rootMoves[0].usiScore = mate_in(1);
                ;

                goto SKIP_SEARCH;
            }
        }
    }

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
    //     "stop"が送られてきたらThreads.stop == trueになる。
    //     "ponderhit"が送られてきたらThreads.ponder == falseになるので、それを待つ。(stopOnPonderhitは用いない)
    //      "go infinite"に対してはstopが送られてくるまで待つ。
    //      ちなみにStockfishのほう、ここのコードに長らく同期上のバグがあった。
    //     やねうら王のほうは、かなり早くからこの構造で書いていた。後にStockfishがこの書き方に追随した。

    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {
        // Busy wait for a stop or a ponder reset
        // stop か ponder reset を待つ間のビジーウェイト

        //	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
        // (思考のためには計算資源を使っていないので。)
        Tools::sleep(1);
        // ⚠ Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。

// TODO : あとで
#if 0
		// === やねうら王独自改良 ===

		// 最終的なPVを出力する。
		// ponder中/go infinite中であっても、ここに抜けてきている以上、全探索スレッドの停止が確認できた時点でPVは出力すべき。
		// "go infinite"の場合、詰みを発見してもそれがponderフラグの解除を待ってからだと、PVを返すのが遅れる。("stop"が来るまで返せない)
		// Stockfishもこうなっている。この作り、良くないように思うので、改良した。

		// 　ここですべての探索スレッドが停止しているならば最終PVを出力してやる。
        if (!output_final_pv_done
            && Threads.search_finished() /* 全探索スレッドが探索を完了している */)
            output_final_pv();
#endif
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
#if 0
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

    // 普通に探索したのでskipしたかのフラグをfalseにする。
    // 💡やねうら王独自
    search_skipped = false;

SKIP_SEARCH:;
    // TODO あとで検討する。
    //output_final_pv();

    // 📌 指し手をGUIに返す 📌

    // Lazy SMPの結果を取り出す

    // 並列探索したうちのbestな結果を保持しているthread
    // まずthisを入れておいて、定跡を返す時などはthisのままにするコードを適用する。
    YaneuraOuWorker* bestThread = this;

    Skill skill =
      //  Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
      Skill(/*(int)Options["SkillLevel"]*/ 20, 0);
    // TODO : Skillの導入はあとで検討する。
    //  🤔  それにしてもオプションが3つも増えるの嫌だな…。

    if (
      int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
      && rootMoves[0].pv[0] != Move::none() && !search_skipped
      // ⚠ "&& !search_skipped"は、やねうら王独自追加。
      //     これを追加しておかないと、定跡にhitしたりして、main threadのrootMovesに積んだりしても、
      //     bestThreadがmain threadではないものを指してしまい、期待した指し手がbestmoveとして出力されなくなる。
    )
        //bestThread = threads.get_best_thread()->worker.get();
        // 💡 やねうら王では、get_best_thread()は、ThreadPoolからこのclassに移動させた。
        bestThread = get_best_thread();

    // 次回の探索のときに何らか使えるのでベストな指し手の評価値を保存しておく。
    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    // 投了スコアが設定されていて、歩の価値を100として正規化した値がそれを下回るなら投了。(やねうら王独自拡張)
    // ただし定跡の指し手にhitした場合などはrootMoves[0].score == -VALUE_INFINITEになっているのでそれは除外。
    auto resign_value = (int) options["ResignValue"];
    if (bestThread->rootMoves[0].score != -VALUE_INFINITE
        && USIEngine::to_cp(bestThread->rootMoves[0].score) <= -resign_value)
        bestThread->rootMoves[0].pv[0] = Move::resign();

#if 0
    // Send again PV info if we have a new best thread
	// 新しいベストスレッドがあれば、再度PV情報を送信する
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);
#endif
    // 💡 ↑こんなにPV出力するの好きじゃないので省略。

    // サイレントモードでないならbestな指し手を出力
    // 📌 サイレントモードは、やねうら王独自拡張
    if (!global_options.silent)
    {
        std::string ponder;

        // 🌈 extract_ponder_from_tt()にponder_candidateを渡すのは、やねうら王独自拡張。
        if (bestThread->rootMoves[0].pv.size() > 1
            || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos,
                                                               main_manager()->ponder_candidate))
            ponder = USIEngine::move(bestThread->rootMoves[0].pv[1] /*, rootPos.is_chess960()*/);

        auto bestmove = USIEngine::move(bestThread->rootMoves[0].pv[0] /*, rootPos.is_chess960()*/);
        main_manager()->updates.onBestmove(bestmove, ponder);
    }
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
          &this->continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
                                                          // TODO : あとで
        //(ss - i)->continuationCorrectionHistory = &this->continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval = VALUE_NONE;
    }

    // Stack(探索用の構造体)上のply(手数)は事前に初期化しておけば探索時に代入する必要がない。
    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    // 最善応手列(Principal Variation)
    ss->pv = pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    // MultiPV
    // 💡 bestmoveとしてしこの局面の上位N個を探索する機能

    size_t multiPV = size_t(options["MultiPV"]);

    //Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
    // 🤔 ↑これでエンジンオプション2つも増えるのやだな…。気が向いたらサポートすることにする。
    Skill skill = Skill(/*(int)Options["SkillLevel"]*/ 20, 0);

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

    // PV出力用のtimer
    // 📌 やねうら王独自
    Timer time(limits.startTime);

    // 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
    // これが増えていない時、同じ深さを再度探索していることになる。(fail highし続けている)
    // 💡 あまり同じ深さでつっかえている時は、aspiration windowの幅を大きくしてやるなどして回避する必要がある。
    int searchAgainCounter = 0;

	// 💡 lowPlyHistoryは、試合開始時に1回だけではなく、"go"の度に初期化したほうが強い。
    lowPlyHistory.fill(86);

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
        if (!threads.increaseDepth)
            searchAgainCounter++;

        // MultiPV loop. We perform a full root search for each PV line
        // MultiPVのloop。MultiPVのためにこの局面の候補手をN個選出する。
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            // 📝 chessではtbRankの処理が必要らしい。将棋では関係なさげなのでコメントアウト。
            //     tbRankが同じ値のところまでしかsortしなくて良いらしい。
            //     (そこ以降は、明らかに悪い指し手なので)

#if 0
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }
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
            delta = PARAM_ASPIRATION_SEARCH1 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 11134;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);

#if 0
            // Adjust optimism based on root move's averageScore
            // ルート手の averageScore に基づいて楽観度を調整する
            //optimism[ us]  = 137 * avg / (std::abs(avg) + 91);
            //optimism[~us] = -optimism[us];
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

                if (
                  mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                  && nodes > 10000000

				// TODO : あとで考える。

                  // 📌 以下やねうら王独自拡張
                  && (rootDepth < 3
                      || mainThread->lastPvInfoTime + global_options.pv_interval <= time.elapsed())
                  // silent modeや検討モードなら出力を抑制する。
                  && !global_options.silent
                  // ただし、outout_fail_lh_pvがfalseならfail high/fail lowのときのPVを出力しない。
                  && global_options.outout_fail_lh_pv

                )
                    main_manager()->pv(*this, threads, tt, rootDepth);

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = (alpha + beta) / 2;
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

                && !(threads.abortedSearch && is_loss(rootMoves[0].usiScore)))
                main_manager()->pv(*this, threads, tt, rootDepth);



            if (threads.stop)
                break;

		} // multi pv loop

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
            rootMoves[0].score = rootMoves[0].usiScore = lastBestScore;
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
#if 0
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
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
			// TODO : あとで。
			//        やねうら王では、このへん仕組みが異なる。
			//         Time.search_endまで持ち時間を使い切りたい。
			//         !Threads.stop && Time.search_end == 0

            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (11.396 + 2.035 * (mainThread->bestPreviousAverageScore - bestValue)
               + 0.968 * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5786, 1.6752);

            // If the bestMove is stable over several iterations, reduce time accordingly
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
				// TODO : やねうら王ではここ0で良いような？

            auto elapsedTime = time.elapsed();

            if (completedDepth >= 10 && nodesEffort >= 97056 && elapsedTime > totalTime * 0.6540
                && !mainThread->ponder)
                threads.stop = true;

            // Stop the search if we have exceeded the totalTime or maximum
			// totalTime または maximum を超えた場合、探索を停止する

			if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                // 停止条件を満たした

                // 📝 将棋の場合、フィッシャールールではないのでこの時点でも最小思考時間分だけは
                //     思考を継続したほうが得なので、思考自体は継続して、キリの良い時間になったら
				//     check_time()にて停止する。

				// If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".

#if 0
				if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
#endif

				auto& tm = mainThread->tm;
                if (mainThread->ponder)
					// ponder中なら、終了時刻はponderhit後から計算して、Time.minimum()。
					tm.search_end = tm.minimum();
                else
                    tm.search_end = std::max(tm.round_up(time.elapsed_from_ponderhit()), tm.minimum());
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.5138;
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


void YaneuraOuWorker::do_move(Position& pos, const Move move, StateInfo& st) {
    do_move(pos, move, st, pos.gives_check(move));
}

void YaneuraOuWorker::do_move(Position&  pos,
                              const Move move,
                              StateInfo& st,
                              const bool givesCheck) {

#if defined(USE_ACCUMULATOR_STACK)

    // accumulatorStackを用いる実装。

    DirtyPiece dp = pos.do_move(move, st, givesCheck /*, &tt */);
    // 📝　やねうら王では、TTのprefetchをしないので、ttを渡す必要がない。
    nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);

#else

    pos.do_move(move, st, givesCheck);
    nodes.fetch_add(1, std::memory_order_relaxed);

#endif
}

void YaneuraOuWorker::do_null_move(Position& pos, StateInfo& st) {
    pos.do_null_move(st /*, tt*/);
    // 📝　やねうら王では、TTのprefetchをしないので、ttを渡す必要がない。
}

void YaneuraOuWorker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);

#if defined(USE_ACCUMULATOR_STACK)
    //accumulatorStack.pop();
#endif
}

void YaneuraOuWorker::undo_null_move(Position& pos) { pos.undo_null_move(); }


// Reset histories, usually before a new game
// 履歴をリセットする。通常は新しいゲームの前に実行される。
void YaneuraOuWorker::clear() {
	// TODO : あとで

    mainHistory.fill(67);
    captureHistory.fill(-688);

	// 📝 lowPlyHistoryの初期化は、対局ごとではなく、局面ごと("go"のごと)に変更された。

#if 0
    pawnHistory.fill(-1287);
    pawnCorrectionHistory.fill(5);
    minorPieceCorrectionHistory.fill(0);
    nonPawnCorrectionHistory.fill(0);
#endif
    ttMoveHistory = 0;

#if 0
    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(8);
#endif

	// 📝 ここは、未初期化のときに[NO_PIECE][SQ_ZERO]を指すので、ここを-1で初期化しておくことによって、
    //     history > 0 を条件にすれば自ずと未初期化のときは除外されるようになる。

    //     ほとんどの履歴エントリがいずれにせよ後で負になるため、
    //     開始値を「正しい」方向に少しシフトさせるため、負の数で埋めている。
    //     この効果は、深度が深くなるほど薄れるので、長時間思考させる時には
    //     あまり意味がないが、無駄ではないらしい。
    //     cf. Tweak history initialization : https://github.com/official-stockfish/Stockfish/commit/7d44b43b3ceb2eebc756709432a0e291f885a1d2

	for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : continuationHistory[inCheck][c])
                for (auto& h : to)
                    h.fill(-473);

	// reductions tableの初期化

    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(2796 / 128.0 * std::log(i));

#if 0
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
    // 🤔 将棋だとあまり千日手が起こらないので効果がなさげ。

#if 0
    // Check if we have an upcoming move that draws by repetition
    // 直近の手が繰り返しによる引き分けになるかを確認します

	if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(this->nodes);
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

    Key   posKey;

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

	Value bestValue, value, eval, /* maxValue,*/ probCutBeta;

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

	auto thisThread    = this;
    ss->inCheck        = pos.checkers();
    priorCapture       = pos.captured_piece();
    Color us           = pos.side_to_move();
    ss->moveCount      = 0;
    bestValue          = -VALUE_INFINITE;
    //maxValue           =  VALUE_INFINITE;
    // 📝 将棋ではtable probe使っていないのでmaxValueは関係ない。

	//     Timerの監視

    // Check for the available remaining time
    // 残りの利用可能な時間を確認します
    // 💡 これはメインスレッドのみが行なう。
	if (is_mainthread())
        main_manager()->check_time(*thisThread);


    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
	// selDepth情報をGUIに送信するために使用します（selDepthは1からカウントし、plyは0からカウントします）
    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

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

		auto draw_type = pos.is_repetition(ss->ply);
        if (draw_type != REPETITION_NONE)
            return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);

		// 最大手数を超えている、もしくは停止命令が来ている。

		#if 0
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos)
                                                        : value_draw(thisThread->nodes);
		#endif

		// 📌 将棋では手数を超えたら無条件で引き分け扱い。
		if (threads.stop.load(std::memory_order_relaxed)
            || ss->ply >= MAX_PLY || pos.game_ply() > global_options.max_game_ply
			)
            return draw_value(REPETITION_DRAW, pos.side_to_move());

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
    ss->isPvNode        = PvNode;
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

	ttData.move  = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
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

	// 🚧 工事中 🚧

#if 0
	if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta)
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5))
    {
        // If ttMove is quiet, update move sorting heuristics on TT hit
        if (ttData.move && ttData.value >= beta)
        {
            // Bonus for a quiet ttMove that fails high
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                       std::min(125 * depth - 77, 1157));

            // Extra penalty for early quiet moves of the previous ply
            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 3 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2301);
        }

        // Partial workaround for the graph history interaction problem
        // For high rule50 counts don't produce transposition table cutoffs.
        if (pos.rule50_count() < 90)
        {
            if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) && pos.legal(ttData.move)
                && !is_decisive(ttData.value))
            {
                do_move(pos, ttData.move, st);
                Key nextPosKey                             = pos.key();
                auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
                undo_move(pos, ttData.move);

                // Check that the ttValue after the tt move would also trigger a cutoff
                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            }
            else
                return ttData.value;
        }
    }

    // Step 5. Tablebases probe
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
                thisThread->tbHits.fetch_add(1, std::memory_order_relaxed);

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

    // Step 6. Static evaluation of the position
    Value      unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue      = correction_value(*thisThread, pos, ss);
    if (ss->inCheck)
    {
        // Skip early pruning when in check
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
        goto moves_loop;
    }
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // ttValue can be used as a better position evaluation
        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // Static evaluation is saved as it was before adjustment by correction history
        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation());
    }

    // Use static evaluation difference to improve quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture && !ttHit)
    {
        int bonus = std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -1858, 1492) + 661;
        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 1057 / 1024;
        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            thisThread->pawnHistory[pawn_structure_index(pos)][pos.piece_on(prevSq)][prevSq]
              << bonus * 1266 / 1024;
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    improving = ss->staticEval > (ss - 2)->staticEval;

    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    if (priorReduction >= 3 && !opponentWorsening)
        depth++;
    if (priorReduction >= 1 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 175)
        depth--;

    // Step 7. Razoring
    // If eval is really low, skip search entirely and return the qsearch value.
    // For PvNodes, we must have a guard against mates being returned.
    if (!PvNode && eval < alpha - 486 - 325 * depth * depth)
        return qsearch<NonPV>(pos, ss, alpha, beta);

    // Step 8. Futility pruning: child node
    // The depth condition is important for mate finding.
    {
        auto futility_margin = [&](Depth d) {
            Value futilityMult = 93 - 20 * (cutNode && !ss->ttHit);

            return futilityMult * d                      //
                 - improving * futilityMult * 2          //
                 - opponentWorsening * futilityMult / 3  //
                 + (ss - 1)->statScore / 376             //
                 + std::abs(correctionValue) / 168639;
        };

        if (!ss->ttPv && depth < 14 && eval - futility_margin(depth) >= beta && eval >= beta
            && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
            return beta + (eval - beta) / 3;
    }

    // Step 9. Null move search with verification search
    if (cutNode && (ss - 1)->currentMove != Move::null() && eval >= beta
        && ss->staticEval >= beta - 19 * depth + 389 && !excludedMove && pos.non_pawn_material(us)
        && ss->ply >= thisThread->nmpMinPly && !is_loss(beta))
    {
        assert(eval - beta >= 0);

        // Null move dynamic reduction based on depth
        Depth R = 7 + depth / 3;

        ss->currentMove                   = Move::null();
        ss->continuationHistory           = &thisThread->continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[NO_PIECE][0];

        do_null_move(pos, st);

        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        if (nullValue >= beta && !is_win(nullValue))
        {
            if (thisThread->nmpMinPly || depth < 16)
                return nullValue;

            assert(!thisThread->nmpMinPly);  // Recursive verification is not allowed

            // Do verification search at high depths, with null move pruning disabled
            // until ply exceeds nmpMinPly.
            thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            thisThread->nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    improving |= ss->staticEval >= beta + 94;

    // Step 10. Internal iterative reductions
    // For PV nodes without a ttMove as well as for deep enough cutNodes, we decrease depth.
    // (*Scaler) Especially if they make IIR less aggressive.
    if (!allNode && depth >= 6 && !ttData.move)
        depth--;

    // Step 11. ProbCut
    // If we have a good enough capture (or queen promotion) and a reduced search
    // returns a value much above beta, we can (almost) safely prune the previous move.
    probCutBeta = beta + 201 - 58 * improving;
    if (depth >= 3
        && !is_decisive(beta)
        // If value from transposition table is lower than probCutBeta, don't attempt
        // probCut there
        && !(is_valid(ttData.value) && ttData.value < probCutBeta))
    {
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &thisThread->captureHistory);
        Depth      probCutDepth = std::max(depth - 5, 0);

        while ((move = mp.next_move()) != Move::none())
        {
            assert(move.is_ok());

            if (move == excludedMove || !pos.legal(move))
                continue;

            assert(pos.capture_stage(move));

            movedPiece = pos.moved_piece(move);

            do_move(pos, move, st);

            ss->currentMove = move;
            ss->continuationHistory =
              &this->continuationHistory[ss->inCheck][true][movedPiece][move.to_sq()];
            ss->continuationCorrectionHistory =
              &this->continuationCorrectionHistory[movedPiece][move.to_sq()];

            // Perform a preliminary qsearch to verify that the move holds
            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            // If the qsearch held, perform the regular search
            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth,
                                       !cutNode);

            undo_move(pos, move);

            if (value >= probCutBeta)
            {
                // Save ProbCut data into transposition table
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               probCutDepth + 1, move, unadjustedStaticEval, tt.generation());

                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        }
    }

moves_loop:  // When in check, search starts here

    // Step 12. A small Probcut idea
    probCutBeta = beta + 400;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value))
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    MovePicker mp(pos, ttData.move, depth, &thisThread->mainHistory, &thisThread->lowPlyHistory,
                  &thisThread->captureHistory, contHist, &thisThread->pawnHistory, ss->ply);

    value = bestValue;

    int moveCount = 0;

    // Step 13. Loop through all pseudo-legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        // Check for legality
        if (!pos.legal(move))
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root
        // Move List. In MultiPV mode we also skip PV moves that have been already
        // searched and those of lower "TB rank" if we are in a TB root position.
        if (rootNode
            && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
                           thisThread->rootMoves.begin() + thisThread->pvLast, move))
            continue;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > 10000000)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + thisThread->pvIdx});
        }
        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture_stage(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);

        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;

        // Calculate new depth for this move
        newDepth = depth - 1;

        int delta = beta - alpha;

        Depth r = reduction(improving, depth, moveCount, delta);

        // Increase reduction for ttPv nodes (*Scaler)
        // Smaller or even negative value is better for short time controls
        // Bigger value is better for long time controls
        if (ss->ttPv)
            r += 968;

        // Step 14. Pruning at shallow depth.
        // Depth conditions are important for mate finding.
        if (!rootNode && pos.non_pawn_material(us) && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // Reduced depth of the next LMR search
            int lmrDepth = newDepth - r / 1024;

            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist =
                  thisThread->captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                if (!givesCheck && lmrDepth < 7 && !ss->inCheck)
                {
                    Value futilityValue = ss->staticEval + 232 + 224 * lmrDepth
                                        + PieceValue[capturedPiece] + 131 * captHist / 1024;
                    if (futilityValue <= alpha)
                        continue;
                }

                // SEE based pruning for captures and checks
                int seeHist = std::clamp(captHist / 31, -137 * depth, 125 * depth);
                if (!pos.see_ge(move, -158 * depth - seeHist))
                {
                    bool mayStalemateTrap =
                      depth > 2 && alpha < 0 && pos.non_pawn_material(us) == PieceValue[movedPiece]
                      && PieceValue[movedPiece] >= RookValue
                      // it can't be stalemate if we moved a piece adjacent to the king
                      && !(attacks_bb<KING>(pos.square<KING>(us)) & move.from_sq())
                      && !mp.can_move_king_or_pawn();

                    // avoid pruning sacrifices of our last piece for stalemate
                    if (!mayStalemateTrap)
                        continue;
                }
            }
            else
            {
                int history =
                  (*contHist[0])[movedPiece][move.to_sq()]
                  + (*contHist[1])[movedPiece][move.to_sq()]
                  + thisThread->pawnHistory[pawn_structure_index(pos)][movedPiece][move.to_sq()];

                // Continuation history based pruning
                if (history < -4229 * depth)
                    continue;

                history += 68 * thisThread->mainHistory[us][move.from_to()] / 32;

                lmrDepth += history / 3388;

                Value baseFutility = (bestMove ? 46 : 230);
                Value futilityValue =
                  ss->staticEval + baseFutility + 117 * lmrDepth + 102 * (ss->staticEval > alpha);

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning
                // scales well with respect to time and threads
                if (!ss->inCheck && lmrDepth < 12 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue)
                        && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE
                if (!pos.see_ge(move, -27 * lmrDepth * lmrDepth))
                    continue;
            }
        }

        // Step 15. Extensions
        // Singular extension search. If all moves but one
        // fail low on a search of (alpha-s, beta-s), and just one fails high on
        // (alpha, beta), then that move is singular and should be extended. To
        // verify this we do a reduced search on the position excluding the ttMove
        // and if the result is lower than ttValue minus a margin, then we will
        // extend the ttMove. Recursive singular search is avoided.

        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.

        if (!rootNode && move == ttData.move && !excludedMove
            && depth >= 6 - (thisThread->completedDepth > 27) + ss->ttPv && is_valid(ttData.value)
            && !is_decisive(ttData.value) && (ttData.bound & BOUND_LOWER)
            && ttData.depth >= depth - 3)
        {
            Value singularBeta  = ttData.value - (58 + 76 * (ss->ttPv && !PvNode)) * depth / 57;
            Depth singularDepth = newDepth / 2;

            ss->excludedMove = move;
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            if (value < singularBeta)
            {
                int corrValAdj   = std::abs(correctionValue) / 248400;
                int doubleMargin = -4 + 244 * PvNode - 206 * !ttCapture - corrValAdj
                                 - 997 * ttMoveHistory / 131072
                                 - (ss->ply > thisThread->rootDepth) * 47;
                int tripleMargin = 84 + 269 * PvNode - 253 * !ttCapture + 91 * ss->ttPv - corrValAdj
                                 - (ss->ply * 2 > thisThread->rootDepth * 3) * 54;

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
            else if (value >= beta && !is_decisive(value))
                return value;

            // Negative extensions
            // If other moves failed high over (ttValue - margin) without the
            // ttMove on a reduced search, but we cannot do multi-cut because
            // (ttValue - margin) is lower than the original beta, we do not know
            // if the ttMove is singular or can do a multi-cut, so we reduce the
            // ttMove in favor of other moves based on some conditions:

            // If the ttMove is assumed to fail high over current beta
            else if (ttData.value >= beta)
                extension = -3;

            // If we are on a cutNode but the ttMove is not assumed to fail high
            // over current beta
            else if (cutNode)
                extension = -2;
        }

        // Step 16. Make the move
        do_move(pos, move, st, givesCheck);

        // Add extension to new depth
        newDepth += extension;

        // Update the current move (this must be done after singular extension search)
        ss->currentMove = move;
        ss->continuationHistory =
          &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];
        uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

        // Decrease reduction for PvNodes (*Scaler)
        if (ss->ttPv)
            r -= 2437 + PvNode * 926 + (ttData.value > alpha) * 901
               + (ttData.depth >= depth) * (943 + cutNode * 1180);

        // These reduction adjustments have no proven non-linear scaling

        r += 316;  // Base reduction offset to compensate for other tweaks
        r -= moveCount * 66;
        r -= std::abs(correctionValue) / 28047;

        // Increase reduction for cut nodes
        if (cutNode)
            r += 2864 + 966 * !ttData.move;

        // Increase reduction if ttMove is a capture
        if (ttCapture)
            r += 1210 + (depth < 8) * 963;

        // Increase reduction if next ply has a lot of fail high
        if ((ss + 1)->cutoffCnt > 2)
            r += 1036 + allNode * 848;

        r += (ss + 1)->quietMoveStreak * 50;

        // For first picked move (ttMove) reduce reduction
        if (move == ttData.move)
            r -= 2006;

        if (capture)
            ss->statScore =
              826 * int(PieceValue[pos.captured_piece()]) / 128
              + thisThread->captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())]
              - 5030;
        else
            ss->statScore = 2 * thisThread->mainHistory[us][move.from_to()]
                          + (*contHist[0])[movedPiece][move.to_sq()]
                          + (*contHist[1])[movedPiece][move.to_sq()] - 3206;

        // Decrease/increase reduction for moves with a good/bad history
        r -= ss->statScore * 826 / 8192;

        // Step 17. Late moves reduction / extension (LMR)
        if (depth >= 2 && moveCount > 1)
        {
            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth.
            // To prevent problems when the max value is less than the min value,
            // std::clamp has been replaced by a more robust implementation.
            Depth d = std::max(1, std::min(newDepth - r / 1024,
                                           newDepth + !allNode + (PvNode && !bestMove)))
                    + (ss - 1)->isPvNode;

            ss->reduction = newDepth - d;
            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;

            // Do a full-depth search when reduced LMR search fails high
            // (*Scaler) Usually doing more shallower searches
            // doesn't scale well to longer TCs
            if (value > alpha && d < newDepth)
            {
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                const bool doDeeperSearch    = value > (bestValue + 42 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1508);
            }
            else if (value > alpha && value < bestValue + 9)
                newDepth--;
        }

        // Step 18. Full-depth search when LMR is skipped
        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            if (!ttData.move)
                r += 1128;

            r -= ttMoveHistory / 8;

            // Note that if expected reduction is high, we reduce search depth here
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > 3564) - (r > 4969 && newDepth > 2), !cutNode);
        }

        // For PV nodes only, do a full PV search on the first move or after a fail high,
        // otherwise let the parent node fail low with value <= alpha and try another move.
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();

            // Extend move from transposition table if we are about to dive into qsearch.
            if (move == ttData.move && thisThread->rootDepth > 8)
                newDepth = std::max(newDepth, 1);

            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        // Step 19. Undo move
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 20. Check for a new best move
        // Finished searching the move. If a stop occurred, the return value of
        // the search cannot be trusted, and we return immediately without updating
        // best move, principal variation nor transposition table.
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
        {
            RootMove& rm =
              *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);

            rm.effort += nodes - nodeCount;

            rm.averageScore =
              rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                  ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                  : value * std::abs(value);

            // PV move or new best move?
            if (moveCount == 1 || value > alpha)
            {
                rm.score = rm.uciScore = value;
                rm.selDepth            = thisThread->selDepth;
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

                rm.pv.resize(1);

                assert((ss + 1)->pv);

                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.
                if (moveCount > 1 && !thisThread->pvIdx)
                    ++thisThread->bestMoveChanges;
            }
            else
                // All other moves but the PV, are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved - just the PV is pushed up.
                rm.score = -VALUE_INFINITE;
        }

        // In case we have an alternative move equal in eval to the current bestmove,
        // promote it to bestmove by pretending it just exceeds alpha (but not beta).
        int inc = (value == bestValue && ss->ply + 2 >= thisThread->rootDepth
                   && (int(nodes) & 15) == 0 && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta)
                {
                    // (* Scaler) Especially if they make cutoffCnt increment more often.
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);  // Fail high
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                if (depth > 2 && depth < 16 && !is_decisive(value))
                    depth -= 2;

                assert(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }

    // Step 21. Check for mate and stalemate
    // All legal moves have been searched and if there are no legal moves, it
    // must be a mate or a stalemate. If we are in a singular extension search then
    // return a fail low score.

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    // Adjust best value for fail high cases
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

    // If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    else if (bestMove)
    {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, moveCount);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 800 : -879);
    }

    // Bonus for prior quiet countermove that caused the fail low
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -220;
        bonusScale += std::min(-(ss - 1)->statScore / 103, 323);
        bonusScale += std::min(73 * depth, 531);
        bonusScale += 174 * ((ss - 1)->moveCount > 8);
        bonusScale += 144 * (!ss->inCheck && bestValue <= ss->staticEval - 104);
        bonusScale += 128 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 82);

        bonusScale = std::max(bonusScale, 0);

        const int scaledBonus = std::min(159 * depth - 94, 1501) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * 412 / 32768);

        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()]
          << scaledBonus * 203 / 32768;

        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            thisThread->pawnHistory[pawn_structure_index(pos)][pos.piece_on(prevSq)][prevSq]
              << scaledBonus * 1040 / 32768;
    }

    // Bonus for prior capture countermove that caused the fail low
    else if (priorCapture && prevSq != SQ_NONE)
    {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        thisThread->captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 1080;
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    // Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.
    if (!excludedMove && !(rootNode && thisThread->pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation());

    // Adjust correction history
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && ((bestValue < ss->staticEval && bestValue < beta)  // negative correction & no fail high
            || (bestValue > ss->staticEval && bestMove)))     // positive correction & no fail low
    {
        auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                                -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *thisThread, bonus);
    }

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
#endif
    return bestValue;
}

void SearchManager::pv(Search::Worker&           worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) {}


// TODO : あとで

namespace {

// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.
Value value_from_tt(Value v, int ply /*, int r50c */) {

    if (!is_valid(v))
        return VALUE_NONE;

    // handle TB win or better
    if (is_win(v))
    {
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 /* - r50c */)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 /* - r50c * /)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        return v - ply;
    }

    // handle TB loss or worse
    if (is_loss(v))
    {
        // Downgrade a potentially false mate score.
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        return v + ply;
    }

    return v;
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

bool Search::RootMove::extract_ponder_from_tt(const TranspositionTable& tt,
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
	// 🌠 やねうら王独自改良
	else if (ponder_candidate)
    {
        Move m = ponder_candidate;
        if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
            pv.push_back(m);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}


#if 0
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




}  // namespace YaneuraOu




#endif // YANEURAOU_ENGINE
