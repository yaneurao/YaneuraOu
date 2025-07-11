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

// 💡 引き分け時の評価値VALUE_DRAW(0)の代わりに±1の乱数みたいなのを与える。
//     nodes : 現在の探索node数(乱数のseed代わりに用いる)

// 📝 チェスでは、引き分けが0.5勝扱いなので引き分け回避のための工夫がしてあって、
//     以下のようにvalue_drawに揺らぎを加算することによって探索を固定化しない(同じnodeを
//     探索しつづけて千日手にしてしまうのを回避)工夫がある。
//     将棋の場合、普通の千日手と連続王手の千日手と劣等局面による千日手(循環？)とかあるのでこれ導入するのちょっと嫌。
//     ⇨  TODO : もうちょっとどうにかする。
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);

// 📝 32は、quietsSearched、quietsSearchedの最大数。そのnodeで生成したQUIETS/CAPTURESの指し手を良い順に保持してある。
//     bonusを加点するときにこれらの指し手に対して行う。
//     Stockfish 16ではquietsSearchedの配列サイズが[64]から[32]になった。
//     将棋ではハズレのquietの指し手が大量にあるので、それがベストとは限らない。
// 🌈　比較したところ、64より32の方がわずかに良かったので、とりあえず32にしておく。(V7.73mとV7.73m2との比較)

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
		// 📝 main以外のすべてのthreadを開始する。
		//    main以外のthreadがstart_searching()を開始する。
		//    start_searching()の先頭には、main thread以外であれば即座に
		//    iterative_deepning()を呼び出すようになっているので、これにより並列探索が開始できる。

		iterative_deepening();      // main thread start searching
		// 💡 main threadも並列探索に加わる。
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
        (ss - i)->staticEval                    = VALUE_NONE;
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
	// 🧠 ↑これでエンジンオプション2つも増えるのやだな…。気が向いたらサポートすることにする。
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

	// 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
    // これが増えていない時、同じ深さを再度探索していることになる。(fail highし続けている)
    // 💡 あまり同じ深さでつっかえている時は、aspiration windowの幅を大きくしてやるなどして回避する必要がある。
    int searchAgainCounter = 0;

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

		// 🧠 将棋ではこれ使わなくていいような？

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
            delta     = PARAM_ASPIRATION_SEARCH1 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 11134;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta,  VALUE_INFINITE);

			#if 0
            // Adjust optimism based on root move's averageScore
            // ルート手の averageScore に基づいて楽観度を調整する
            //optimism[ us]  = 137 * avg / (std::abs(avg) + 91);
            //optimism[~us] = -optimism[us];
			#endif
            // 🧠 このoptimismは、StockfishのNNUE評価関数で何やら使っているようなのだが…。
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

					// 🚧 作業中 🚧

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
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread
                && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000)
                // A thread that aborted search can have mated-in/TB-loss PV and
                // score that cannot be trusted, i.e. it can be delayed or refuted
                // if we would have had time to fully search other root-moves. Thus
                // we suppress this output and below pick a proven score/PV for this
                // thread (from the previous iteration).
                && !(threads.abortedSearch && is_loss(rootMoves[0].uciScore)))
                main_manager()->pv(*this, threads, tt, rootDepth);

            if (threads.stop)
                break;
        }

        if (!threads.stop)
            completedDepth = rootDepth;

        // We make sure not to pick an unproven mated-in score,
        // in case this thread prematurely stopped search (aborted-search).
        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score))
        {
            // Bring the last best move to the front for best thread selection.
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

        if (!mainThread)
            continue;

        // Have we found a "mate in x"?
        if (limits.mate && rootMoves[0].score == rootMoves[0].uciScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;

        // If the skill level is enabled and time is up, pick a sub-optimal best move
        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Use part of the gained time from a previous stable move for the current move
        for (auto&& th : threads)
        {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
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
            if (rootMoves.size() == 1)
                totalTime = std::min(500.0, totalTime);

            auto elapsedTime = elapsed();

            if (completedDepth >= 10 && nodesEffort >= 97056 && elapsedTime > totalTime * 0.6540
                && !mainThread->ponder)
                threads.stop = true;

            // Stop the search if we have exceeded the totalTime or maximum
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
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
