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
// 💡 やねうら王ではdraw valueはテーブルを参照するため、その表引きの時に条件分岐が入るのが嫌だから、このコード使わないことにする。
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position&      pos,
                      Stack*               ss,
                      Search::Worker&      workerThread,
                      Move                 bestMove,
                      Square               prevSq,
                      ValueList<Move, 32>& quietsSearched,
                      ValueList<Move, 32>& capturesSearched,
                      Depth                depth,
                      Move                 TTMove,
                      int                  moveCount);

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

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options
			/*  , main_manager()->originalTimeAdjust */);
			// 💡 やねうら王では、originalTimeAdjustは用いない。

	tt.new_search();

    if (rootMoves.empty())
    {
        // rootで指し手がない = (将棋だと)詰みの局面である

		rootMoves.emplace_back(Move::none());
        //main_manager()->updates.onUpdateNoMoves(
        //  {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
		// 💡 チェスだと王手されていないなら引き分けだが、将棋だとつねに負け。
		main_manager()->updates.onUpdateNoMoves({0, -VALUE_MATE });
    }
    else
    {
        threads.start_searching();  // start non-main threads
        iterative_deepening();      // main thread start searching
    }

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

	while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {}  // Busy wait for a stop or a ponder reset
		// stop か ponder reset を待つ間のビジーウェイト

    // Stop the threads if not already stopped (also raise the stop if
    // "ponderhit" just reset threads.ponder)
    // まだ停止していなければスレッドを停止する（"ponderhit" により threads.ponder が
    // リセットされた場合も stop を発生させる）
	threads.stop = true;

    // Wait until all threads have finished
    // すべてのスレッドが終了するのを待つ
    threads.wait_for_search_finished();

	// 💡 やねうら王では、npmsecをサポートしない。
	#if 0
    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    // 'nodes as time' モードでプレイしている場合、終了する前に
    // 使用可能なノード数から探索済みノード数を差し引く。
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);
	#endif

    YaneuraOuWorker* bestThread = this;
    //Skill   skill =
    //  Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate /* && !skill.enabled() */
        && rootMoves[0].pv[0] != Move::none())
        //bestThread = threads.get_best_thread()->worker.get();
		// 💡 やねうら王では、get_best_thread()は、ThreadPoolからこのclassに移動させた。
        bestThread = get_best_thread();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    // Send again PV info if we have a new best thread
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);

    std::string ponder;

    if (bestThread->rootMoves[0].pv.size() > 1
        || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos, Move::none() /* TODO あとで*/))
        ponder = USIEngine::move(bestThread->rootMoves[0].pv[1] /*, rootPos.is_chess960()*/);

    auto bestmove = USIEngine::move(bestThread->rootMoves[0].pv[0] /*, rootPos.is_chess960()*/);
    main_manager()->updates.onBestmove(bestmove, ponder);
}



void SearchManager::pv(Search::Worker&           worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) {}



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

}  // namespace YaneuraOu


#endif // YANEURAOU_ENGINE
