#include "usi.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <sstream>
#include <string_view>
#include <filesystem>
#include <utility>
#include <variant>
#include <vector>

#include "benchmark.h"
#include "engine.h"
#include "memory.h"
#include "movegen.h"
#include "position.h"
#include "score.h"
#include "search.h"
#include "types.h"
#include "usioption.h"

#if !STOCKFISH
#include "extra/system_io.h"
#include "extra/tools.h"
#include "testcmd/unit_test.h"

#if defined(__EMSCRIPTEN__)
// yaneuraou.wasm
#include <emscripten.h>
#endif
#endif


namespace YaneuraOu {

using Time = std::chrono::steady_clock;
using ms   = std::chrono::milliseconds;

// benchmark用のコマンドその2
constexpr auto BenchmarkCommand = "speedtest";

#if !STOCKFISH

// Engine開発者が用いる"test"コマンド。
namespace Test {
void test_cmd(IEngine& engine, std::istringstream& is);
}

// 定跡を作るコマンド
#if defined(ENABLE_MAKEBOOK_CMD)
namespace Book {
void makebook(IEngine& engine, std::istringstream& is);
}
#endif

#endif

/*
	📓 複数のlambda型を1つにまとめるtemplate。
        以下例のようにvariant型からの(デザパタの)visitorパターンを簡単に書ける。

		std::variant<int, std::string> v = ...;

		std::visit(overload{
			[](int i) { std::cout << "int: " << i; },
			[](const auto& s) { std::cout << "string: " << s; }
		}, v);
*/

template<typename... Ts>
struct overload : Ts... {
	using Ts::operator()...;
};

template<typename... Ts>
overload(Ts...) -> overload<Ts...>;

// string_viewで受け取ったものを"\n"で複数行に分割して、それをinfo stringとして出力する。
void USIEngine::print_info_string(std::string_view str) {
	sync_cout_start();
	for (auto& line : split(str, "\n"))
		if (!is_whitespace(line))
			std::cout << "info string " << line << '\n';
	sync_cout_end();
}

#if STOCKFISH
USIEngine::USIEngine(CommandLine cli_) :
    engine(cli_.argc > 0 ? std::optional{path_from_utf8(cli_.argv[0])} : std::nullopt),
    cli(std::move(cli_)) {
    // 📌  やねうら王では、CommandLine::gが持つようになったのでこのclassには持たせない。

    engine.get_options().add_info_listener([](const std::optional<std::string>& str) {
        if (str.has_value())
            print_info_string(*str);
    });

    // 📝 Stockfishでは"uci"が来る前に"info string"でオプション内容を出力している。
    //     やねうら王では、この機能、サポートしない。

    // すべての読み筋出力listenerを初期化する。
    init_search_update_listeners();
    // 📝 やねうら王では、外部からEngine派生classをset_engine()でセットするので、
    //     そのタイミングで呼び出さないと駄目。
}
#else

void USIEngine::set_engine(IEngine& _engine) {
    engine.set_engine(_engine);

    // "engine_option_profile.txt"で、どのオプション群を生やすかを決める。
    // これは"usi"応答より前に反映される必要がある。
    engine.get_options().read_engine_option_profile("engine_option_profile.txt");

    // ⚠ やねうら王では、Engineのコンストラクタではoptionを生やさない設計に変更した。
    //     よって、派生classのadd_options()をここで明示的に呼び出してoptionを生やす必要がある。
    engine.add_options();

    // 📝 旧評価関数は、起動時にEval::add_options()が呼び出されることを
    //     期待するコードになっているので呼び出して初期化してやる。
    //     また、その時にエンジンオプションを追加する。

#if defined(USE_CLASSIC_EVAL)
    Eval::add_options(engine.get_options(), engine.get_threads());
#endif

	// optionの値が変更された時に、その結果文字列を出力するためのhandlerを設定してやる。
    engine.get_options().add_info_listener([](const std::optional<std::string>& str) {
        if (str.has_value())
            print_info_string(*str);
    });

#if defined(ENGINE_OPTIONS)
    // 🌈 ENGINE_OPTIONSマクロで指定されたエンジンオプションを反映させる。
    //     上書き対象のオプションがすべて生えたあとに行う必要があるので、ここで呼び出す。
    engine.get_options().set_engine_options(ENGINE_OPTIONS);
#endif

    // 📝 セットされたEngineに対してlisterを設定する必要がある。
    //     Stockfishは、USIEngineのコンストラクタで行っているが、
    //     やねうら王ではEngineの差し替えができるのでこのタイミング。
    init_search_update_listeners();
}

#endif

// すべての読み筋出力listenerを初期化する。
void USIEngine::init_search_update_listeners() {
    engine.set_on_iter([](const auto& i) { on_iter(i); });
    engine.set_on_update_no_moves([](const auto& i) { on_update_no_moves(i); });
#if STOCKFISH
    engine.set_on_update_full(
      [this](const auto& i) { on_update_full(i, engine.get_options()["UCI_ShowWDL"]); });
#else
    engine.set_on_update_full([](const auto& i) { on_update_full(i); });
#endif
#if !STOCKFISH
    engine.set_on_update_string([](const auto& i) { on_update_string(i); });
#endif
    engine.set_on_start([]() {});
    engine.set_on_bestmove([](const auto& bm, const auto& p) { on_bestmove(bm, p); });
    engine.set_on_verify_network([](const auto& s) { print_info_string(s); });
}

#if !STOCKFISH
// 標準入力から1行もらう。Ctrl+Zが来れば"quit"が来たものとする。
// また先行入力でqueueに積んでおくことができる。(次のread_command()で取り出される)
std::string USIEngine::read_command()
{
	std::string cmd;
	if (pending_commands.empty())
	{
		if (!std::getline(std::cin, cmd)) // 入力が来るかEOFがくるまでここで待機する。
			cmd = "quit";
	} else {
		// 積んであるコマンドがあるならそれを実行する。
		// 尽きれば"quit"だと解釈してdoループを抜ける仕様にすることはできるが、
		// そうしてしまうとgoコマンド(これはノンブロッキングなので)の最中にquitが送られてしまう。
		// ただ、
		// YaneuraOu-mid.exe bench,quit
		// のようなことは出来るのでPGOの役には立ちそうである。
		cmd = pending_commands.front();
		pending_commands.pop();
	}
	return cmd;
}
#endif

// USI応答部ループ
void USIEngine::loop()
{
    set_console_utf8();
#if STOCKFISH
    std::string cmd; // diff:ignore

    for (int i = 1; i < cli.argc; ++i)
        cmd += std::string(cli.argv[i]) + " ";

    do
    {
        // Wait for an input or an end-of-file (EOF) indication.
        // 入力またはファイル終端を待つ。
        if (cli.argc <= 1 // diff:ignore
            && !getline(std::cin, cmd))  // Wait for an input or an end-of-file (EOF) indication
            cmd = "quit";
    } while (!usi_cmdexec(cmd) && cli.argc <= 1); // diff:ignore
#else
    // 起動時コマンドは、マルチインスタンス時に注入先を選べるよう、
    // 選択されたEngineのentry point側からコマンドキューへ積む。

#if !defined(__EMSCRIPTEN__)

	// USIコマンドの処理
	while (true)
	{
		// 標準入力から1行取得。
		std::string cmd = read_command();

		// "quit"が来たらwhileを抜ける
		if (usi_cmdexec(cmd))
			break;
	}

	// エンジンの終了を待つ。
	engine.wait_for_search_finished();

#else
	// yaneuraOu.wasm
	// ここでループしてしまうと、ブラウザのメインスレッドがブロックされてしまう。
#endif
#endif
} // diff:ignore

// USIEngine::loop()の下請け。
// 📝 wasm版対応のため、関数を分離する必要があった。
bool USIEngine::usi_cmdexec(const std::string& cmd) { // diff:ignore

    currentCmd = cmd;
    std::istringstream is(cmd);
    std::string token; // diff:ignore
    is >> token;

#if STOCKFISH
    if (token == "quit" || token == "stop")
#else
    if (token == "quit" || token == "stop" || token == "gameover")
    /*
	📓 USIプロトコルにはUCIプロトコルから、
            gameover win | lose | draw
        が追加されているが、stopと同じ扱いをして良いと思う。

		あと、将棋所では、通信対局では、こちらのgo ponderのあと相手が投了すると、
		stopを送らずにgameoverが送られてくるため、これをハンドルしておかないと
		ponderingが停止しない。

		gameoverに対してはbestmoveは返すべきではないのかも知れないが、
        それを言えばstopにだって…。
	*/
#endif
        // "stop"コマンドが来るとEngine.stop()が呼び出され、その結果threads.stop = trueとなる。
        engine.stop();

    // The GUI sends 'ponderhit' to tell that the user has played the expected move.
    // So, 'ponderhit' is sent if pondering was done on the same move that the user
    // has played. The search should continue, but should also switch from pondering
    // to the normal search.

    // GUIは「ponderhit」を送信して、ユーザーが予想通りの手を指したことを通知する。
    // つまり、ユーザーが実際に指した手と同じ手についてポンダリング（先読み）が
    // 行われていた場合、「ponderhit」が送られる。
    // 探索は継続すべきだが、ポンダリングから通常の探索に切り替える必要がある。

    else if (token == "ponderhit")
#if STOCKFISH
        engine.set_ponderhit(false);
#else
    {
        // Stochastic Ponder中にhitした。
        if (stochastic_ponder_rewound)
        {
			// 思考をいったん停止。このときbestmoveを出力されると困るので抑制してから。
            auto on_bestmove = engine.get_on_bestmove();
            engine.set_on_bestmove([](auto,auto) {});
	        engine.stop();
            engine.wait_for_search_finished();
            engine.set_on_bestmove(std::move(on_bestmove));

			// 1手前の局面で思考させていたので、現在の局面にする必要がある。

            std::istringstream iss1(last_position_cmd_string);
            iss1 >> token; // 先頭の"position"を捨てる。
            position(iss1);
            stochastic_ponder_rewound = false;

			std::istringstream iss2(last_go_cmd_string);
            iss2 >> token; // 先頭の"go"を捨てる。
            // ponderの位置に依存せず制限を読み直し、通常探索として再開する。
            // parse_limits()で開始時刻も今回のponderhit時刻へ更新する。
            auto limits = parse_limits(iss2);
            limits.ponderMode = false;
            engine.go(limits);
		}
		else
	        engine.set_ponderhit(false);
    }
#endif

    // 起動時いきなりこれが飛んでくるので速攻応答しないとタイムアウトになる。
    else if (token == "usi")
#if STOCKFISH
    {
        sync_cout << "id name " << engine_info(true) << "\n"
                    << engine.get_options() << sync_endl;

        sync_cout << "uciok" << sync_endl;
    }
#else
        engine.usi();
#endif

    // オプションを設定する
    else if (token == "setoption")
        setoption(is);

    else if (token == "go")
    {
        // send info strings after the go command is sent for old GUIs and python-chess
        // 古いGUIやpython-chessのために、goコマンド送信後にinfo文字列を送信する。

#if STOCKFISH
        print_info_string(engine.numa_config_information_as_string());
        print_info_string(engine.thread_allocation_information_as_string());
        /*
		📓 以下のようなメッセージを出力する。要らんと思う..。
        	info string Available processors : 0 - 31
        	info string Using 4 thread
	*/
#else
		// Stochastic Ponderのために"go"コマンド行を保存しておく。
        last_go_cmd_string = cmd;
#endif
        go(is);
    }

    else if (token == "position")
#if STOCKFISH
        position(is);
#else
    {
		// Stochastic Ponderのために保存しておく。
        last_position_cmd_string = cmd;
        position(is);
    }
#endif

#if STOCKFISH
    else if (token == "ucinewgame")
        engine.search_clear();
#else
    else if (token == "usinewgame")
        engine.usinewgame();
#endif

    // 思考エンジンの準備が出来たかの確認
    else if (token == "isready")
#if STOCKFISH
        sync_cout << "readyok" << sync_endl;
#else
        isready();
#endif

    // Add custom non-UCI commands, mainly for debugging purposes.
    // These commands must not be used during a search!

    // 独自の非UCIコマンドを追加する（主にデバッグ目的）。
    // これらのコマンドは探索中に使用してはならない！

    // 📝 flipは盤面を180°回転させたsfenを出力する。
    else if (token == "flip")
#if STOCKFISH
    {
        if (auto err = engine.flip())
            terminate_on_critical_error(err->what());
    }
#else
        engine.flip();
#endif

    // ベンチコマンド(これは常に使える)
    else if (token == "bench")
        bench(is);

    else if (token == BenchmarkCommand)
        benchmark(is);

    // 現在の局面を視覚的に表示する。
    else if (token == "d")
        sync_cout << engine.visualize() << sync_endl;

    // 現在の局面の評価値を表示する。
    else if (token == "eval")
        engine.trace_eval();

    // コンパイルに使用したコンパイラを表示する。
    else if (token == "compiler")
        sync_cout << compiler_info() << sync_endl;

    // 評価関数パラメーターをファイルに保存する。
    // export_net filename
    else if (token == "export_net")
    {
#if STOCKFISH
        std::optional<std::filesystem::path> file;
        std::string                          filename;

        if (is >> filename)
            file = path_from_utf8(filename);

        engine.save_network(file);
#else
    std::string file;
    is >> file;
    engine.save_network(file);
#endif
    }

#if STOCKFISH
    else if (token == "--help" || token == "help" || token == "--license" || token == "license")
            sync_cout
              << "\nStockfish is a powerful chess engine for playing and analyzing."
                 "\nIt is released as free software licensed under the GNU GPLv3 License."
                 "\nStockfish is normally used with a graphical user interface (GUI) and implements"
                 "\nthe Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc."
                 "\nFor any further information, visit https://github.com/official-stockfish/Stockfish#readme"
                 "\nor read the corresponding README.md and Copying.txt files distributed along with this program.\n"
              << sync_endl;

    else if (!token.empty() && token[0] != '#')
        sync_cout << "Unknown command: '" << cmd << "'. Type help for more information." << sync_endl;
#else

	// --------------------------------
    // 📌 以下、やねうら王独自拡張 📌
    // --------------------------------

    // fileの内容をUSIコマンドとして実行する。
    else if (token == "f")
        enqueue_command_from_file(is);

	// evaluate()を呼び出して生の評価値の出力。(デバッグ用)
	else if (token == "e")
        sync_cout << "eval = " << engine.evaluate() << sync_endl;

    // この局面での指し手をすべて出力
    else if (token == "moves")
        moves();

    // オプションを取得する
    else if (token == "getoption")
        getoption(is);

    // UnitTest
    else if (token == "unittest")
        unittest(is);

    // config.hで設定した値などについて出力する。
    else if (token == "config")
        sync_cout << config_info() << sync_endl;

    // .psv(PsvRecord列)の局面をqsearch PVのleafに置換する。
    else if (token == "qsearch_psv")
        qsearch_psv(is);

#if defined(ENABLE_MAKEBOOK_CMD)
	// 定跡コマンド
	else if (token == "makebook")
        Book::makebook(engine, is);
#endif

    // 指し手生成祭りの局面をセットする。
    else if (token == "matsuri")
    {
        std::istringstream iss("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1");

		// "position"コマンドが実行されたものとして処理する。
        position(iss);

        sync_cout << "The matsuri position has been set." << sync_endl;
    }

    // ログファイルの書き出しのon
    // 🤔 Stockfishの方は、エンジンオプションでログの出力ファイル名を指定できるのだが、
    //     ログ自体はホスト側で記録することが多いので、ファイル名は固定でいいや…。
    else if (token == "log")
        start_logger("io_log.txt");

#if defined(ENABLE_TEST_CMD)
    // テストコマンド
    else if (token == "test")
        Test::test_cmd(engine, is);
#endif

    // ユーザーによる実験用コマンド。Engine::user_cmd()が呼び出される。
    else if (token == "user")
        engine.user(is);


    // エンジンオプションの簡易変更機能
    else
    {
        /*
			簡略表現として、
			> threads 1
			    のように指定したとき、
			> setoption name Threads value 1
			    と等価なようにしておく。
			また、
			> threads = 1
			のように、 = が入っている表記も可能とする。
		*/

        if (!token.empty())
        {
            std::string value;
            is >> value;
            if (value == "=") // skip '='
                is >> value;

			// ⚠ set_option_if_exists()のなかでoptionが変更されて、結果文字列が戻ってきて、
			//    それを出力するhandlerが呼び出されて、そのなかでsync_cout ～ sync_endlで出力するので
			//    ここで直接 set_option_if_exists()の戻り値をsync_cout～sync_endlで出力しようとすると
			//    二重ロックになる。ゆえにいったん変数に代入している。

			auto ss = engine.get_options().set_option_if_exists(token, value);
            sync_cout << ss << sync_endl;

        }
    }
#endif

#if STOCKFISH
	// diffを抑制するため
		} while (token != "quit" && cli.argc <= 1);  // The command-line arguments are one-shot
#endif

    // "quit"に対しては、この関数はtrueを返す。
    return token == "quit"; // diff:ignore
}


// コマンドラインを解析して、Search::LimitsTypeに反映させて返す。
Search::LimitsType USIEngine::parse_limits(std::istream& is) {

	Search::LimitsType limits;
	std::string        token;

	// 📝 "go"が呼び出された時にこの関数が呼び出される。
	//     なるべく早くに探索開始時刻をlimits.startTimeに格納しておかないと
	//     サーバー時刻との差が生じてしまうから、早めに格納する。

    limits.startTime = now();  // The search starts as early as possible
							   // 探索はできるだけ早く開始される

    // エンジンオプションによる探索制限(0なら無制限)
    // このあと、depthもしくはnodesが指定されていたら、その値で上書きされる。(この値は無視される)

#if !STOCKFISH
    const auto& options = engine.get_options();
    limits.depth = options.count("DepthLimit") ? (int) options["DepthLimit"] : 0;
    limits.nodes = options.count("NodesLimit") ? (u64) options["NodesLimit"] : 0;
#endif

	while (is >> token)
    {

		/* 📓 searchmovesとは？
		
			探索すべき指し手。(探索開始局面から特定の初手だけ探索させるとき)

			これ、Stockfishのコードでこうなっているからそのままにしてあるが、
			これを指定しても定跡の指し手としてはこれ以外を指したりする問題はある。
			またふかうら王ではこのオプションをサポートしていない。
			ゆえに、非対応扱いで考えて欲しい。
		*/

		if (token == "searchmoves")  // Needs to be the last command on the line
			                         // この行の最後のコマンドである必要がある
		{
			// 残りの指し手すべてをsearchMovesに突っ込む。
			while (is >> token)
#if STOCKFISH
				limits.searchmoves.push_back(to_lower(token));
#else
				// チェスUCIではsearchmovesを小文字化しても問題ないが、将棋USIでは
				// "P*5e"のように打ち駒の駒種を大文字で表すので、大文字小文字を保持する。
				limits.searchmoves.push_back(token);
#endif
            break;
		}

		// 先手、後手の残り時間。[ms]
        else if (token == "wtime")
			is >> limits.time[WHITE];
		else if (token == "btime")
			is >> limits.time[BLACK];

		// フィッシャールール時における時間
        else if (token == "winc")
			is >> limits.inc[WHITE];
		else if (token == "binc")
			is >> limits.inc[BLACK];

#if STOCKFISH
		// あと何手で引き分けとなるか。
		// 🤔 やねうら王ではサポートしない。GUIが対応していないし…。
		else if (token == "movestogo")
			is >> limits.movestogo;
#else
        // 秒読み設定。
        // 📝 USIでは"byoyomi"として指定できる。
        else if (token == "byoyomi")
        {
            TimePoint t = 0;
            is >> t;

            // USIプロトコルで送られてきた秒読み時間より少なめに思考する設定
            // ※　通信ラグがあるときに、ここで少なめに思考しないとタイムアップになる可能性があるので。

            // t = std::max(t - Options["ByoyomiMinus"], Time::point(0));

            // USIプロトコルでは、これが先手後手同じ値だと解釈する。
            limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = t;
        }

        // ランダム時間の思考を行う、"rtime"。
        // 例) "go rtime 100"だと100～300[ms]思考する。
		// 📌 やねうら王独自
		else if (token == "rtime")
            is >> limits.rtime;

#endif
        // この探索深さで探索を打ち切る
        else if (token == "depth")
			is >> limits.depth;

		// この探索ノード数で探索を打ち切る
		else if (token == "nodes")
			is >> limits.nodes;

		// 持ち時間固定(将棋だと対応しているGUIが無いが..)
		else if (token == "movetime")
			is >> limits.movetime;

        // 詰み探索。
		// 📝 UCIではこのあとには手数が入っており、その手数以内に詰むかどうかを判定するが、
        //     USIでは、ここは探索のための時間制限に変更となっている。
#if STOCKFISH
		else if (token == "mate")
			is >> limits.mate;
#else
        else if (token == "mate")
        {
            // USIでは引数を省略した場合も無制限として扱う。
            if (!(is >> token))
            {
                limits.mate = INT32_MAX;
                break;
            }

            if (token == "infinite")
                limits.mate = INT32_MAX;
            else
            {
                // USIのmateの数値は探索時間（ミリ秒）。
                std::istringstream mateLimit(token);
                if (!(mateLimit >> limits.mate) || !mateLimit.eof())
                    terminate_on_critical_error("Invalid argument for 'mate'");
            }
        }
#endif

		/* 📓 perftとは？
		       パフォーマンステストの略。
               合法手N手で到達できる局面の数を求める。
		*/
		else if (token == "perft")
			is >> limits.perft;

		// 時間無制限。
		else if (token == "infinite")
			limits.infinite = 1;

		// ponderモードでの思考。
        else if (token == "ponder")
			limits.ponderMode = true;

        if (is.fail())
            terminate_on_critical_error("Invalid argument for '" + token + "'");
    }

    // Sanitize negative time values, zero for stm means no TM
    // 負の残り時間と固定思考時間を補正する。手番側の残り時間がゼロなら時間管理を使わない。
    for (auto* t : {&limits.time[WHITE], &limits.time[BLACK], &limits.movetime})
        if (*t < 0)
            *t = 1;

    return limits;
}

// --------------------
// USI関係のコマンド処理
// --------------------

// Called when the engine receives the "go" UCI command. The function sets the
// thinking time and other parameters from the input string then stars with a search

// go()は、思考エンジンがUSIコマンドの"go"を受け取ったときに呼び出される。
// この関数は、入力文字列から思考時間とその他のパラメーターをセットし、探索を開始する。

void USIEngine::go(std::istringstream& is)
{
    Search::LimitsType limits = parse_limits(is);

	if (limits.perft)
		perft(limits);
	else
#if STOCKFISH
		engine.go(limits);
#else
	{
        stochastic_ponder_rewound = false;
        // Stochastic Ponderでは、moves節の最後の指し手だけを除いて1手戻す。
        // 履歴がなければ元の局面のまま通常のponderを行う。
        if (limits.ponderMode && engine.get_options().count("Stochastic_Ponder")
            && engine.get_options()["Stochastic_Ponder"])
        {
            std::istringstream command(last_position_cmd_string);
            std::string token, previousPosition;
            bool moves = false;
            usize lastMove = std::string::npos;
            while (command >> token)
            {
                if (moves)
                    lastMove = previousPosition.size();
                previousPosition += token + ' ';
                if (token == "moves")
                    moves = true;
            }

            if (lastMove != std::string::npos)
            {
                previousPosition.resize(lastMove);
                std::istringstream previous(previousPosition);
                previous >> token;  // Skip "position".
                                    // 先頭の"position"を読み飛ばす。
                position(previous);
                stochastic_ponder_rewound = true;
            }
        }

		engine.go(limits);
	}
#endif
}

// "bench"コマンドの応答部。
void USIEngine::bench(std::istream& args) {

    std::string token;
    u64         num, nodes = 0, cnt = 1;
    u64         nodesSearched = 0;
    const auto& options       = engine.get_options();

    engine.set_on_update_full([&](const auto& i) {
        nodesSearched = i.nodes;
#if STOCKFISH
        on_update_full(i, options["UCI_ShowWDL"]);
#else
        on_update_full(i);
#endif
    });

#if !STOCKFISH
	// USIEngine::isready()を呼び出してやらかないと"engine_options.txt"などの読み込みが行われない。
    isready();
#endif

    std::vector<std::string> list = Benchmark::setup_bench(engine.sfen(), args);

    num = count_if(list.begin(), list.end(), [](const std::string& s) { return s.find("go ") == 0 || s.find("eval") == 0; });

    TimePoint elapsed = now();

    for (const auto& cmd : list)
    {
        currentCmd = cmd; // diff:ignore
        std::istringstream is(cmd);
        is >> token;

        if (token == "go" || token == "eval")
        {
            std::cerr << "\nPosition: " << cnt++ << '/' << num << " (" << engine.sfen() << ")"
                      << std::endl;
            if (token == "go")
            {
                Search::LimitsType limits = parse_limits(is);

#if !STOCKFISH
                // benchではPV出力間隔を無効化し、最後のinfo nodesを探索終了時点の値に揃える。
                limits.disablePvInterval = true;
#endif

                if (limits.perft)
                    nodesSearched = perft(limits);
                else
                {
                    engine.go(limits);
                    engine.wait_for_search_finished();
                }

                nodes += nodesSearched;
                nodesSearched = 0;
            }
            else
                engine.trace_eval();
        }
        else if (token == "setoption")
            setoption(is);
        else if (token == "position")
            position(is);
        else if (token == "ucinewgame")
						// 💡 Stockfishとの互換性維持のため"usinewgame"と変更していない。
						//     どうせ内部でしか使わない符号みたいなものなので…。
        {
			engine.search_clear();  // search_clear may take a while
                                    // search_clear は時間がかかることがある

			elapsed = now();
        }
    }

    elapsed = now() - elapsed + 1;  // Ensure positivity to avoid a 'divide by zero'
                                    // ゼロ除算を避けるために正の値であることを保証する

    dbg_print();

    std::cerr << "\n==========================="    //
              << "\nTotal time (ms) : " << elapsed  //
              << "\nNodes searched  : " << nodes    //
              << "\nNodes/second    : " << 1000 * nodes / elapsed << std::endl;

    // reset callback, to not capture a dangling reference to nodesSearched
    // コールバックをリセットする。nodesSearched へのダングリング参照を捕捉しないようにするため。

#if STOCKFISH
    engine.set_on_update_full([&](const auto& i) { on_update_full(i, options["UCI_ShowWDL"]); });
#else
    engine.set_on_update_full([&](const auto& i) { on_update_full(i); });
#endif
}

void USIEngine::benchmark(std::istream& args) {

	// Probably not very important for a test this long, but include for completeness and sanity.
    // このように長いテストではおそらくあまり重要ではないが、完全性と安心のために含めておく。
    static constexpr int NUM_WARMUP_POSITIONS = 3;

    std::string token;
    u64         cnt = 1;

    engine.set_on_update_full([](const auto&) {});
    engine.set_on_iter([](const auto&) {});
    engine.set_on_update_no_moves([](const auto&) {});
    engine.set_on_bestmove([](const auto&, const auto&) {});
    engine.set_on_verify_network([](const auto&) {});

#if !STOCKFISH
	// USIEngine::isready()を呼び出してやらかないと"engine_options.txt"などの読み込みが行われない。
    isready();
#endif

    Benchmark::BenchmarkSetup setup = Benchmark::setup_benchmark(args);

    const auto numGoCommands = count_if(setup.commands.begin(), setup.commands.end(),
                                             [](const std::string& s) { return s.find("go ") == 0; });

    // Set options once at the start.
    auto ss = std::istringstream("name Threads value " + std::to_string(setup.threads));
    setoption(ss);
#if STOCKFISH
	ss = std::istringstream("name Hash value " + std::to_string(setup.ttSize));
    setoption(ss);
    ss = std::istringstream("name UCI_Chess960 value false");
    setoption(ss);
#else
    ss = std::istringstream("name USI_Hash value " + std::to_string(setup.ttSize));
    setoption(ss);
#endif

    // Warmup
    for (const auto& cmd : setup.commands)
    {
        currentCmd = cmd; // diff:ignore
        std::istringstream is(cmd);
        is >> token;

        if (token == "go")
        {
            // One new line is produced by the search, so omit it here
            std::cerr << "\rWarmup position " << cnt++ << '/' << NUM_WARMUP_POSITIONS;

            Search::LimitsType limits = parse_limits(is);

            // Run with silenced network verification
            engine.go(limits);
            engine.wait_for_search_finished();
        }
        else if (token == "position")
            position(is);
        else if (token == "ucinewgame")
            engine.search_clear();  // search_clear may take a while

        if (cnt > NUM_WARMUP_POSITIONS)
            break;
    }

    std::cerr << "\n";

    cnt = 1;

    int           numHashfullReadings = 0;
    constexpr int hashfullAges[]      = {0, 999};  // Only normal hashfull and touched hash.
    constexpr int hashfullAgeCount    = std::size(hashfullAges);
    int           totalHashfull[hashfullAgeCount] = {0};
    int           maxHashfull[hashfullAgeCount]   = {0};

    auto updateHashfullReadings = [&]() {
        numHashfullReadings += 1;

        for (int i = 0; i < hashfullAgeCount; ++i)
        {
            const int hashfull = engine.get_hashfull(hashfullAges[i]);
            maxHashfull[i]     = std::max(maxHashfull[i], hashfull);
            totalHashfull[i] += hashfull;
        }
    };

    engine.search_clear();  // search_clear may take a while

#if STOCKFISH || defined(YANEURAOU_ENGINE)
    Time::time_point elapsed;
#endif

    Time::duration totalTime(0);
    u64 nodes = 0, nodesSearched = 0;

    engine.set_on_update_full([&](const Engine::InfoFull& i) { nodesSearched = i.nodes; });

#if STOCKFISH || defined(YANEURAOU_ENGINE)
    engine.set_on_start([&elapsed, &nodesSearched]() {
        elapsed       = Time::now();
        nodesSearched = 0;
    });

    engine.set_on_bestmove(
      [&totalTime, &elapsed, &nodes, &nodesSearched](const auto&, const auto&) {
          totalTime += Time::now() - elapsed;
          nodes += nodesSearched;
      });
#endif

    for (const auto& cmd : setup.commands)
    {
        currentCmd = cmd; // diff:ignore
        std::istringstream is(cmd);
        is >> token;

        if (token == "go")
        {
            // One new line is produced by the search, so omit it here
            std::cerr << "\rPosition " << cnt++ << '/' << numGoCommands;

            Search::LimitsType limits = parse_limits(is);

#if !STOCKFISH && !defined(YANEURAOU_ENGINE)
            // 開始通知を送らない構成では、goから探索終了待ちまでを測る。
            nodesSearched = 0;
            const auto elapsed = Time::now();
#endif

            // Run with silenced network verification
            engine.go(limits);
            engine.wait_for_search_finished();

#if !STOCKFISH && !defined(YANEURAOU_ENGINE)
            totalTime += Time::now() - elapsed;
            nodes += nodesSearched;
#endif
            updateHashfullReadings();
        }
        else if (token == "position")
            position(is);
        else if (token == "ucinewgame")
            engine.search_clear();  // search_clear may take a while
    }

    // Ensure positivity to avoid a 'divide by zero'
    // NPSの計算でゼロ除算しないよう、合計時間を最低1ミリ秒とする。
    const auto totalTimeMs = std::max<i64>(std::chrono::duration_cast<ms>(totalTime).count(), 1LL);

    dbg_print();

    std::cerr << "\n";

    static_assert(std::size(hashfullAges) == 2 && hashfullAges[0] == 0 && hashfullAges[1] == 999,
                  "Hardcoded for display. Would complicate the code needlessly in the current state.");

    std::string threadBinding = engine.thread_binding_information_as_string();
    if (threadBinding.empty())
        threadBinding = "none";

    // clang-format off

    std::cerr << "==========================="
              << "\nVersion                    : "
              << engine_version_info()
              // "\nCompiled by                : "
              << compiler_info()
              << "Large pages                : " << (has_large_pages() ? "yes" : "no")
              << "\nUser invocation            : " << BenchmarkCommand << " "
              << setup.originalInvocation << "\nFilled invocation          : " << BenchmarkCommand
              << " " << setup.filledInvocation
              << "\nAvailable processors       : " << engine.get_numa_config_as_string()
              << "\nThread count               : " << setup.threads
              << "\nThread binding             : " << threadBinding
              << "\nTT size [MiB]              : " << setup.ttSize
              << "\nHash max, avg [per mille]  : "
              << "\n    single search          : " << maxHashfull[0] << ", "
              << totalHashfull[0] / numHashfullReadings
              << "\n    single game            : " << maxHashfull[1] << ", "
              << totalHashfull[1] / numHashfullReadings
              << "\nTotal nodes searched       : " << nodes
              << "\nTotal search time [s]      : " << totalTimeMs / 1000.0
              << "\nNodes/second               : " << 1000 * nodes / totalTimeMs << std::endl;

    // clang-format on

    init_search_update_listeners();

}

// "setoption"コマンド応答。
void USIEngine::setoption(std::istringstream& is) {
    engine.wait_for_search_finished();
    engine.get_options().setoption(is);
}

u64 USIEngine::perft(const Search::LimitsType& limits) {
#if STOCKFISH
    auto result = engine.perft(engine.fen(), limits.perft, engine.get_options()["UCI_Chess960"]);
#else
    auto result = engine.perft(engine.sfen(), limits.perft /*, engine.get_options()["UCI_Chess960"]*/);
#endif
    if (auto err = std::get_if<PositionSetError>(&result))
        terminate_on_critical_error(err->what());

    auto nodes = std::get<u64>(result);
    sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;
    return nodes;
}

// "position"コマンドのhandler
void USIEngine::position(std::istringstream& is) {
	std::string token, sfen;

    is >> token;

    if (token == "startpos")
    {
        // 初期局面として初期局面のFEN形式の入力が与えられたとみなして処理する。
        sfen = StartSFEN;
        is >> token;  // Consume the "moves" token, if any
					  // もしあるなら"moves"トークンを消費する。
    }
    // 局面がfen形式で指定されているなら、その局面を読み込む。
    // UCI(チェスプロトコル)ではなくUSI(将棋用プロトコル)だとここの文字列は"fen"ではなく"sfen"
#if STOCKFISH
    else if (token == "fen")
        while (is >> token && token != "moves")
            sfen += token + " ";
    else
        return;
#else
    // 💡 この"sfen"という文字列は省略可能にしたいので
    //     Stockfishのコードを少し工夫して書き換える。
    else
    {
        // "sfen"なら吸い込むが、"sfen"でないなら、それを局面文字列の一部とみなす。
        if (token != "sfen")
            sfen += token + " ";

        while (is >> token && token != "moves")
            sfen += token + " ";
    }
#endif

    std::vector<std::string> moves;

	// 指し手のリストをパースする(あるなら)
    while (is >> token)
        moves.push_back(token);

    auto err = engine.set_position(sfen, moves);
    if (err.has_value())
        terminate_on_critical_error(err->what());
}

#if STOCKFISH

namespace {

struct WinRateParams {
    double a;
    double b;
};

WinRateParams win_rate_params(const Position& pos) {

    int material = pos.count<PAWN>() + 3 * pos.count<KNIGHT>() + 3 * pos.count<BISHOP>()
                 + 5 * pos.count<ROOK>() + 9 * pos.count<QUEEN>();

    // The fitted model only uses data for material counts in [17, 78], and is anchored at count 58.
    double m = std::clamp(material, 17, 78) / 58.0;

    // Return a = p_a(material) and b = p_b(material), see github.com/official-stockfish/WDL_model
    constexpr double as[] = {-142.72052667, 372.35176398, -340.71073572, 415.23490212};
    constexpr double bs[] = {5.93832785, 15.61267078, -30.57816876, 69.63866711};

    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

    return {a, b};
}

// The win rate model is 1 / (1 + exp((a - eval) / b)), where a = p_a(material) and b = p_b(material).
// It fits the LTC fishtest statistics rather accurately.
int win_rate_model(Value v, const Position& pos) {

    auto [a, b] = win_rate_params(pos);

    // Return the win rate in per mille units, rounded to the nearest integer.
    return int(0.5 + 1000 / (1 + std::exp((a - double(v)) / b)));
}
}

#endif


// Score構造体の内容をUSI形式のscoreとして出力する。
std::string USIEngine::format_score(const Score& s) {
#if STOCKFISH
    constexpr int TB_CP = 20000;
#endif
    const auto    format =
      overload{[](Score::Mate mate) -> std::string {
#if STOCKFISH
                   auto m = (mate.plies > 0 ? (mate.plies + 1) : mate.plies) / 2;
        // 📝 UCIだと先後1手ずつで mate Xと出力しているらしく、2で割ってある。
#else
                   auto m = mate.plies;
#endif
                   return std::string("mate ") + std::to_string(m);
               },
#if STOCKFISH
               [](Score::Tablebase tb) -> std::string {
                   return std::string("cp ") + std::to_string((tb.win ? TB_CP : -TB_CP) - tb.plies);
               },
#endif
               [](Score::InternalUnits units) -> std::string {
                   return std::string("cp ") + std::to_string(units.value);
               }};

    return s.visit(format);
}

// Turns a Value to an integer centipawn number,
// without treatment of mate and similar special scores.
// 詰みやそれに類似した特別なスコアの処理なしに、Valueを整数のセントポーン数に変換する。
#if STOCKFISH
int USIEngine::to_cp(Value v, const Position& pos) {

    // In general, the score can be defined via the WDL as
    // (log(1/L - 1) - log(1/W - 1)) / (log(1/L - 1) + log(1/W - 1)).
    // Based on our win_rate_model, this simply yields v / a.

    auto [a, b] = win_rate_params(pos);

    return int(std::round(100 * int(v) / a));
}

std::string USIEngine::wdl(Value v, const Position& pos) {
    std::stringstream ss;

    int wdl_w = win_rate_model(v, pos);
    int wdl_l = win_rate_model(-v, pos);
    int wdl_d = 1000 - wdl_w - wdl_l;
    ss << wdl_w << " " << wdl_d << " " << wdl_l;

    return ss.str();
}

#else

// → やねうら王の場合、PawnValue = 90なので Value = 90なら 100として出力する必要がある。
constexpr int NormalizeToPawnValue = Eval::PawnValue;

int USIEngine::to_cp(Value v) { return 100 * v / NormalizeToPawnValue; }

#endif

// Square型をUSI文字列に変換する
std::string USIEngine::square(Square s) {
	return std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
}

// 指し手をUSI文字列に変換する。
#if STOCKFISH
std::string USIEngine::move(Move m, bool chess960) {
    if (m == Move::none())
        return "(none)";

    if (m == Move::null())
        return "0000";

    Square from = m.from_sq();
    Square to   = m.to_sq();

    if (m.type_of() == CASTLING && !chess960)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    std::string move = square(from) + square(to);

    if (m.type_of() == PROMOTION)
        move += " pnbrqk"[m.promotion_type()];

    return move;
}

#else

std::string USIEngine::move(Move m /*, bool chess960*/) { return USIEngine::move(m.to_move16()); }

std::string USIEngine::move(Move16 m){

	std::stringstream ss;
	if (!m.is_ok())
	{
		ss << ((m.to_u16() == MOVE_RESIGN) ? "resign":
			   (m.to_u16() == MOVE_WIN)    ? "win"   :
			   (m.to_u16() == MOVE_NULL)   ? "null"  :
			   (m.to_u16() == MOVE_NONE)   ? "none"  :
			"");
	}
	else if (m.is_drop())
	{
		ss << m.move_dropped_piece();
		ss << '*';
		ss << m.to_sq();
	}
	else {
		ss << m.from_sq();
		ss << m.to_sq();
		if (m.is_promote())
			ss << '+';
	}
	return ss.str();
}

// 読み筋をUSI文字列化して返す。
// " 7g7f 8c8d" のように返る。
std::string USIEngine::move(const std::vector<Move>& moves)
{
	std::ostringstream oss;
	for (const auto& move : moves) {
		oss << " " << move;
	}
	return oss.str();
}
#endif

// string全体を小文字化して返す。
std::string USIEngine::to_lower(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });

	return str;
}

// USIの指し手文字列をMove型の変換する。
// 合法手でなければMove::noneを返すようになっている。
// 💡 合法でない指し手の場合、エラーである旨を出力する。
#if STOCKFISH
Move USIEngine::to_move(const Position& pos, std::string str) {
    str = to_lower(str);
    // ⇨  将棋(USIプロトコル)では、駒打ちの時に大文字と小文字の区別があるので、
    //	  小文字化して比較することはできない。

    // 全合法手のなかからusi文字列に変換したときにstrと一致する指し手を探してそれを返す
    for (const auto& m : MoveList<ALL>(pos))
        if (str == move(m, pos.is_chess960()))
            return m;
    // 📝 のコードは大変美しいコードではあるが、
    //     棋譜を大量に読み込むときに時間がかかるうるのでもっと高速な実装をする。

    return Move::none();
}
#else
Move USIEngine::to_move(const Position& pos, std::string str) {

	if (str == "resign")
		return Move::resign();

	if (str == "win")
		return Move::win();

	// パス(null move)入力への対応 {UCI: "0000", GPSfish: "pass"}
	if (str == "0000" || str == "null" || str == "pass")
		return Move::null();

	// usi文字列を高速にmoveに変換するやつがいるがな..
	Move move = pos.to_move(to_move16(str));

	// 現在の局面に至る手順として歩の不成が与えられることはあるので、
	// pseudo_legal_s<true>()で判定する。
	if (pos.pseudo_legal_s<true>(move) && pos.legal(move))
		return move;

	// 入力に非合法手が含まれていた。エラーとして出力すべき。
	sync_cout << "info string Error! : Illegal Input Move : " << str << sync_endl;

	return Move::none();
}

// USI形式から指し手への変換。本来この関数は要らないのだが、
// 棋譜を大量に読み込む都合、この部分をそこそこ高速化しておきたい。
// やねうら王、独自追加。
Move16 USIEngine::to_move16(const std::string& str)
{
	Move16 move = Move16::none();

	{
		// さすがに3文字以下の指し手はおかしいだろ。
		if (str.length() <= 3)
			goto END;

		Square to = usi_to_sq(str[2], str[3]);
		if (!is_ok(to))
			goto END;

		bool promote = str.length() == 5 && str[4] == '+';
		bool drop = str[1] == '*';

		if (!drop)
		{
			Square from = usi_to_sq(str[0], str[1]);
			if (is_ok(from))
				move = promote ? make_move_promote16(from, to) : make_move16(from, to);
		}
		else
		{
			for (int i = 1; i <= 7; ++i)
				if (PieceToCharBW[i] == str[0])
				{
					move = make_move_drop16((PieceType)i, to);
					break;
				}
		}
	}

END:
	return move;
}

// USIの指し手文字列などに使われている盤上の升を表す文字列をSquare型に変換する
// 変換できなかった場合はSQ_NBが返る。高速化のために用意した。
Square USIEngine::usi_to_sq(char f, char r) {
    File file = toFile(f);
    Rank rank = toRank(r);

    if (is_ok(file) && is_ok(rank))
        return file | rank;

    return SQ_NB;
}

// USIプロトコルのマス目文字列をSquare型に変換する。
// 変換できない文字である場合、SQ_NBを返す。
Square USIEngine::to_square(const std::string& str) {
    if (str.size() != 2)
        return SQ_NB;
    return usi_to_sq(str[0], str[1]);
}

#endif

void USIEngine::on_update_no_moves(const Engine::InfoShort& info) {
    sync_cout << "info depth " << info.depth << " score " << format_score(info.score) << sync_endl;
}

#if STOCKFISH
void USIEngine::on_update_full(const Engine::InfoFull& info, bool showWDL) {
#else
void USIEngine::on_update_full(const Engine::InfoFull& info) {
#endif

	std::stringstream ss;

    ss << "info";

#if STOCKFISH
	ss << " depth " << info.depth                 //
       << " seldepth " << info.selDepth           //
       << " multipv " << info.multiPV             //
       << " score " << format_score(info.score);  //
#else

	ss << " depth " << info.depth;

	// selDepthは、非0の時のみ出力する。(定跡にhitして出力されると見づらい)
    if (info.selDepth)
        ss << " seldepth " << info.selDepth;

	ss << " multipv " << info.multiPV             //
       << " score " << format_score(info.score);  //
#endif

    if (!info.bound.empty())
        ss << " " << info.bound;

#if STOCKFISH
    if (showWDL)
        ss << " wdl " << info.wdl;
#endif

    ss << " nodes " << info.nodes        //
       << " nps " << info.nps            //
       << " hashfull " << info.hashfull  //
#if STOCKFISH
       << " tbhits " << info.tbHits  //
#endif
       << " time " << info.timeMs  //
       << " pv " << info.pv;       //

    sync_cout << ss.str() << sync_endl;
}

void USIEngine::on_iter(const Engine::InfoIter& info) {

	std::stringstream ss;

    ss << "info";
    ss << " depth " << info.depth                     //
       << " currmove " << info.currmove               //
       << " currmovenumber " << info.currmovenumber;  //

    sync_cout << ss.str() << sync_endl;
}

void USIEngine::on_bestmove(std::string_view bestmove, std::string_view ponder) {

    sync_cout << "bestmove " << bestmove;
    if (!ponder.empty())
        std::cout << " ponder " << ponder;
    std::cout << sync_endl;
}

void USIEngine::terminate_on_critical_error(const std::string& message) {
    sync_cout << "info string CRITICAL ERROR: Command `" << currentCmd
              << "` failed. Reason: " << message << '\n'
              << sync_endl;
    std::exit(1);
}

#if !STOCKFISH
void USIEngine::on_update_string(std::string_view info) { sync_cout << "info string " << info << sync_endl; }

// 🌈 以下、やねうら王独自 🌈

// cpからValueへ。to_cp()の逆変換。
Value USIEngine::cp_to_value(int v) {
    return Value((std::abs(v) < VALUE_MATE_IN_MAX_PLY) ? (NormalizeToPawnValue * v / 100) : v);
}


// "isready"コマンドのhandler
void USIEngine::isready() {

    // 探索中は初期化・設定の再読込を行わず、そのまま応答する。
    // 終了待ちに入ると、同じ入力スレッドで後続のstopを処理できなくなる。
    if (engine.is_searching())
    {
        sync_cout << "readyok" << sync_endl;
        return;
    }

	auto& options = engine.get_options();

	// 🌈 カレントフォルダに"engine_options.txt"があれば
    //    それをオプションとしてOptions[]の値をオーバーライドする機能。
    options.read_engine_options("engine_options.txt");

    // 🌈 EvalDirにある"eval_options.txt"も読み込みたい。

	// "EvalDir"オプションが生えているなら..
	if (options.count("EvalDir"))
    {
        const auto eval_options_path = path_from_utf8(options["EvalDir"]) / "eval_options.txt";
        options.read_engine_options(eval_options_path.u8string());
    }

	// Engineの派生classのisready()を呼び出す。
    engine.isready();
}

// "moves"コマンドのhandler
void USIEngine::moves() {
    auto& pos = engine.get_position();
    for (auto m : MoveList<LEGAL_ALL>(pos))
        std::cout << m << ' ';
    std::cout << std::endl;
}

// "getoption"コマンドのhandler
// オプションの値を取得する。
void USIEngine::getoption(std::istringstream& is) {
    auto& options = engine_options();

    // getoption オプション名
    std::string option_name;
    is >> option_name;
    sync_cout << options.get_option(option_name) << sync_endl;
}

// USI拡張コマンド "qsearch_psv" のhandler。
// input.psvの各PsvRecord局面をqsearchのPV leaf nodeで置換し、
// output.psvへ同じPSV形式で書き出す処理をEngine側へ委譲する。
void USIEngine::qsearch_psv(std::istringstream& is) {
    std::string inputPath, outputPath;
    size_t      workerCount = 0;

    is >> inputPath >> outputPath;
    if (!(is >> workerCount))
        workerCount = 0;

    std::string message;
    const bool  ok = engine.qsearch_psv(inputPath, outputPath, workerCount, message);

    if (!message.empty())
        sync_cout << "info string " << message << sync_endl;

    if (!ok)
        sync_cout << "info string qsearch_psv failed" << sync_endl;
}

// "unittest"コマンドのhandler
void USIEngine::unittest(std::istringstream& is) { Test::UnitTest(is, engine); }


// コマンドラインに書かれているUSIコマンドをコマンドキューに積む。
void USIEngine::enqueue_command_line_commands(const CommandLine& cli)
{
	// ファイルからコマンドの指定
	if (cli.argc >= 3 && std::string(cli.argv[1]) == "file")
	{
		std::vector<std::string> cmds0;
		SystemIO::ReadAllLines(cli.argv[2], cmds0);

		// queueに変換する。
		for (auto c : cmds0)
			pending_commands.push(c);

	} else {

		std::string cmd;

		// 引数として指定されたものを一つのコマンドとして実行する機能
		// ただし、','が使われていれば、そこでコマンドが区切れているものとして解釈する。

		for (int i = 1; i < cli.argc; ++i)
		{
			std::string s = cli.argv[i];

			// sから前後のスペースを除去しないといけない。
			while (!s.empty() && s.back() == ' ') s.pop_back();
			while (!s.empty() && s.front() == ' ') s = s.substr(1, s.size() - 1);

			if (s != ",")
				cmd += s + " ";
			else
			{
				pending_commands.push(cmd);
				cmd = "";
			}
		}
		if (cmd.size() != 0)
			pending_commands.push(cmd);
	}
}


// startup.txtなどのファイルに書かれているUSIコマンドをコマンドキューに積む。
void USIEngine::enqueue_startup_file_commands(const std::string& filename) {
    std::vector<std::string> lines;
    if (SystemIO::ReadAllLines(filename, lines).is_ok())
    {
        for (auto& line : lines)
            pending_commands.push(line);
    }
}

// 単一エンジン起動用に、コマンドラインとstartup.txtのUSIコマンドをコマンドキューに積む。
void USIEngine::enqueue_startup_commands(const CommandLine& cli) {
    enqueue_command_line_commands(cli);
    enqueue_startup_file_commands("startup.txt");
}

// ファイルからUSIコマンドをコマンドキューに積む。
void USIEngine::enqueue_command_from_file(std::istringstream& is) {
    std::string filename = "";
    is >> filename;
    if (!filename.empty())
    {
        filename += ".txt";
        sync_cout << "USI Commands from File = " << filename << sync_endl;

        std::vector<std::string> lines;
        if (SystemIO::ReadAllLines(filename, lines).is_ok())
            for (auto& line : lines)
                pending_commands.push(line);
        else
            sync_cout << "Error : File Not Found." << sync_endl;
    }
}


// ----------------------------------
//      USI拡張コマンド "test"
// ----------------------------------

#if defined(ENABLE_TEST_CMD)

// USI拡張コマンドのうち、開発上のテスト関係のコマンド。
// 思考エンジンの実行には関係しない。

namespace Test {
// 通常のテスト用コマンド。コマンドを処理した時 trueが返る。
bool normal_test_cmd(IEngine& engine, std::istringstream& is, const std::string& token);

// 詰み関係のテスト用コマンド。コマンドを処理した時 trueが返る。
bool mate_test_cmd(IEngine& engine, std::istringstream& is, const std::string& token);

void test_cmd(IEngine& engine, std::istringstream& is) {
    std::string token;
    is >> token;

    // デザパタのDecoratorの呼び出しみたいな感じで書いていく。

    // 通常のテスト用コマンド
    if (normal_test_cmd(engine, is, token))
        return;

    // 詰み関係のテスト用コマンド
    if (mate_test_cmd(engine, is, token))
        return;

    sync_cout << "info string Error! : unknown command = " << token << sync_endl;
}
}

#endif  // defined(ENABLE_TEST_CMD)



// namespace USI内のUnitTest。
void USIEngine::UnitTest(Test::UnitTester& tester, IEngine& engine) {
    auto section1 = tester.section("USI");

    Position  pos;
    StateInfo si;

    // 平手初期化
    auto hirate_init = [&] { pos.set_hirate(&si); };

    // SFEN文字列でのPosition初期化
    auto sfen_init = [&](const std::string& sfen) { pos.set(sfen, &si); };

    {
        auto section2 = tester.section("to_move()");
        {
            //auto section3 = tester.section("unpromoted pawn move");

            sfen_init("2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1");

            auto moves =
              "6a5b 1g1f 4a3b 1f1e 3a2b 2g2f 2c2d 7g7f 3b2c 2f2e 2d2e 2h2e P*2d 2e9e 7a8b 8h6f 8c8d 3i3h 5b6b 7i6h 6b7b 9e9f 7b8c 6h7g 7c7d 7f7e 7d7e 7g8f 8b7c 1e1d 1c1d 8f7e P*7d 7e8f 5a5b P*7b 5b6b 7b7a+ 6b7a P*7e 7a6b 7e7d 7c7d 8f9e P*7e 9f8f 8d8e 8f9f 5c5d 5g5f 6c6d 5f5e 6b6c 5e5d 6c5d 8i7g 6d6e 6f5g 2c3d 6i7h 3d4e 4g4f P*5f 5g6h 4e5e 3h4g 6e6f 4f4e 9c9d 9e9d 8c8d 6g6f P*9e 9d9c 8d9d P*5g 9e9f P*2c 2b2c 5g5f 5e4e P*4f 4e4d 4f4e 4d4e 9c8b 7e7f 7g6e 7d7e P*4f 4e4d 4f4e 4d4e 7h6g R*8i 5i4h 8i9i+ P*4f 4e4d 4f4e 4d4e 6e7c+ 4c4d 7c7d 9d8d 7d8d 7e8d 4g4f 4e4f 6h4f L*4e 6g5g 4e4f 5g4f 5d4c 5f5e P*5c G*2b 2c3d L*5i 4d4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f N*4d 4h3h 4d3f 3g3f 4e4f P*2g S*4g 3h2h 4g3f P*4d 4c4d P*4h 2d2e P*3g 3f2g 2h2g 2e2f 2g2f 3d3e 2f2e G*2d 2e1f 1d1e 1f1g P*2f N*3f 3e3f 3g3f B*2g S*3h 2g3f+ 1g2h N*3e S*1h 2f2g+ 1h2g 3e2g+ 3h2g 3f6c N*3f 4d3d 3f2d P*2f 2g2f S*2g 2h3i P*2h G*3g 2h2i+ 3i2i P*2h 2i3i N*5g 3g4f 5g4i+ 3i4i G*6h P*7i 9i7i G*5h 6h5h 4i5h 7i7h G*6h G*6g 5h4g 7h6h N*3f G*5g 4g3g 6h4h 3g2g 4h4f P*3g 3d4e S*3e 4f4g P*4i 4e5f 4i4h 4g4h 5i5g 5f5g 1i1e 5g5h 2g1f 5h5i P*4f 4h4i 1e1c+ 4i5h 1f1e 5h4g G*3i 4g5h 8b7a+ 6c6d 1e1d 6d5e 8g8f 8e8f P*8g 8f8g";
            // ↑この局面、最後の8f8gが歩の不成だが、これがUSI::to_move()で非合法手扱いされないかをテストする。
            // cf. https://github.com/yaneurao/YaneuraOu/issues/190

            std::istringstream is(moves);
            std::string        token;
            bool          fail = false;

            StateInfo si[512];
            while (is >> token)
            {
                Move m = USIEngine::to_move(pos, token);
                if (m == Move::none())
                    fail = true;

                pos.do_move(m, si[pos.game_ply()]);
            }

            tester.test("pawn's unpromoted move", !fail);
        }
    }
}

// TODO : 🚧 工事中 🚧

#if 0

// ユーザーの実験用に開放している関数。
// USI拡張コマンドで"user"と入力するとこの関数が呼び出される。
// "user"コマンドの後続に指定されている文字列はisのほうに渡される。
void user_test(Position& pos, std::istringstream& is);

#if defined(ENABLE_TEST_CMD)
	void generate_moves_cmd(Position& pos);
#endif

#if defined(USE_MATE_DFPN)
// "mate"コマンド
void mate_cmd(Position& pos, std::istream& is);
#endif


// "gameover"コマンドに対するハンドラ
#if defined(USE_GAMEOVER_HANDLER) || defined(YANEURAOU_ENGINE_DEEP)
void gameover_handler(const string& cmd);
#endif

// --------------------
//   USI parse helper
// --------------------

// "ponderhit"に"go"で使うようなwtime,btime,winc,binc,byoyomiが書けるような拡張。(やねうら王独自拡張。USI拡張プロトコル)
// 何かトークンを処理したらこの関数はtrueを返す。
bool parse_ponderhit(istringstream& is, Search::LimitsType& limits)
{
	string token;
	bool token_processed = false;

	while (is >> token)
	{
		// 何かトークンを処理したらこの関数はtrueを返す。
		token_processed = true;

		// 先手、後手の残り時間。[ms]
		if (token == "wtime")     is >> limits.time[WHITE];
		else if (token == "btime")     is >> limits.time[BLACK];

		// フィッシャールール時における時間
		else if (token == "winc")      is >> limits.inc[WHITE];
		else if (token == "binc")      is >> limits.inc[BLACK];

		// "go rtime 100"だと100～300[ms]思考する。
		else if (token == "rtime")     is >> limits.rtime;

		// 秒読み設定。
		else if (token == "byoyomi") {
			TimePoint t = 0;
			is >> t;

			// USIプロトコルでは、これが先手後手同じ値だと解釈する。
			limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = t;
		}
	}
	return token_processed;
}


// --------------------
// 🌈 USI拡張コマンド 🌈
// --------------------

// --------------------
//   やねうら王独自
// --------------------


// スコアを歩の価値を100として正規化して出力する。
//   MATEではないスコアなら"cp x"のように出力する。
//   MATEのスコアなら、"mate x"のように出力する。
std::string USIEngine::value(Value v)
{
	ASSERT_LV3(-VALUE_INFINITE < v && v < VALUE_INFINITE);

	std::stringstream ss;

	// 置換表上、値が確定していないことがある。
	if (v == VALUE_NONE)
		ss << "none";
	else if (std::abs(v) < VALUE_MATE_IN_MAX_PLY)
		//s << "cp " << v * 100 / int(Eval::PawnValue);
		ss << "cp " << USIEngine::to_cp(v);
	/*
    else if (abs(v) <= VALUE_TB)
    {
        const int ply = VALUE_TB - std::abs(v);  // recompute ss->ply
        ss << "cp " << (v > 0 ? 20000 - ply : -20000 + ply);
    }
	*/
	else if (v == -VALUE_MATE)
		// USIプロトコルでは、手数がわからないときには "mate -"と出力するらしい。
		// 手数がわからないというか詰んでいるのだが…。これを出力する方法がUSIプロトコルで定められていない。
		// ここでは"-0"を出力しておく。
		// ※　ShogiGUIだと、これで"+詰"と出力されるようである。
		ss << "mate -0";
	else
		ss << "mate " << (v > 0 ? VALUE_MATE - v : -VALUE_MATE - v);

	return ss.str();
}



#endif

#if defined(__EMSCRIPTEN__)
// --------------------
// EMSCRIPTEN support
// --------------------

namespace {
// wasm_keep_alive()で保持するインスタンス。プロセス終了まで解放しない。(理由はusi.hを参照)
std::unique_ptr<IEngine>   wasm_engine;
std::unique_ptr<USIEngine> wasm_usi;
}  // namespace

void wasm_keep_alive(std::unique_ptr<IEngine> engine, std::unique_ptr<USIEngine> usi) {
    wasm_engine = std::move(engine);
    wasm_usi    = std::move(usi);
}

USIEngine* wasm_usi_instance() { return wasm_usi.get(); }

int USIEngine::exec_command_from_js(const std::string& cmd) {
    // 💡 usi_cmdexec()の戻り値("quit"ならtrue)は使わない。
    //     "quit"はwasm_pre.jsが先にModule.terminate()で処理するため、ここには届かない。
    usi_cmdexec(cmd);

    return 0;
}

// USI応答部 emscriptenインターフェース
// 💡 wasm_pre.jsからModule.ccall("usi_command", ...)で呼び出される。
EMSCRIPTEN_KEEPALIVE extern "C" int usi_command(const char* c_cmd) {
    USIEngine* usi = wasm_usi_instance();

    // engine_main()が未実行。JS側に再送してもらう。
    if (usi == nullptr)
        return 1;

    return usi->exec_command_from_js(std::string(c_cmd));
}
#endif

#endif // !STOCKFISH

} // namespace YaneuraOu
