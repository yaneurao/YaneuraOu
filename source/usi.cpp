#include "types.h"
#include "usi.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "misc.h"
#include "testcmd/unit_test.h"
#include "benchmark.h"
#include "engine.h"

#if !defined(YANEURAOU_ENGINE_DEEP)
#include "tt.h"
#endif

#if defined(__EMSCRIPTEN__)
// yaneuraou.wasm
#include <emscripten.h>
#endif

#include <sstream>
#include <queue>

using namespace std;
namespace YaneuraOu {

namespace Test {
	void test_cmd(IEngine& engine, std::istringstream& is);
}

// benchmark用のコマンドその2
constexpr auto BenchmarkCommand = "speedtest";

// 初期局面
//constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// ⇨  Position::SFEN_HIRATEに移動。

/*
  複数のlambda型を1つにまとめるtemplate。
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
	{
		if (!is_whitespace(line))
		{
			std::cout << "info string " << line << '\n';
		}
	}
	sync_cout_end();
}

USIEngine::USIEngine(/*int argc, char** argv*/)
	// :
	//engine(argv[0]),
	// 📌 やねうら王では、Engine engineは、あとからset_engine()で渡すように変更した。
	//cli(argc, argv)
	// 📌  やねうら王では、CommandLine::gが持つようになったのでこのclassには持たせない。
{

	//engine.get_options().add_info_listener([](const std::optional<std::string>& str) {
	//	if (str.has_value())
	//		print_info_string(*str);
	//	});
	// 📝 Stockfishでは"uci"が来る前に"info string"でオプション内容を出力している。
	//     やねうら王では、この機能、サポートしない。

	// すべての読み筋出力listenerを初期化する。
    init_search_update_listeners();
}

// すべての読み筋出力listenerを初期化する。
void USIEngine::init_search_update_listeners() {
    engine.set_on_iter([](const auto& i) { on_iter(i); });
    engine.set_on_update_no_moves([](const auto& i) { on_update_no_moves(i); });
    engine.set_on_update_full(
      [this](const auto& i) { on_update_full(i, engine.get_options()["UCI_ShowWDL"]); });
    engine.set_on_bestmove([](const auto& bm, const auto& p) { on_bestmove(bm, p); });
    engine.set_on_verify_networks([](const auto& s) { print_info_string(s); });
}

void USIEngine::set_engine(IEngine& _engine)
{
	engine.set_engine(_engine);

	// ⚠ やねうら王では、Engineのコンストラクタではoptionを生やさない設計に変更した。
	//     よって、add_options()をここで明示的に呼び出してoptionを生やす必要がある。
	engine.add_options();
}


// USI応答部ループ
void USIEngine::loop()
{
	// コマンドラインと"startup.txt"に書かれているUSIコマンドをstd_inputに積む。
	//enqueue_startup_command();

#if !defined(__EMSCRIPTEN__)

	// USIコマンドの処理
	while (true)
	{
		// 標準入力から1行取得。
		string cmd = std_input.input();

		// "quit"が来たらwhileを抜ける
		if (usi_cmdexec(cmd))
			break;
	}

#else
	// yaneuraOu.wasm
	// ここでループしてしまうと、ブラウザのメインスレッドがブロックされてしまう。
#endif
}

// コマンドラインを解析して、Search::LimitsTypeに反映させて返す。
Search::LimitsType USIEngine::parse_limits(std::istream& is) {
	Search::LimitsType limits;
	std::string        token;

	limits.startTime = now();  // The search starts as early as possible

	while (is >> token)
		if (token == "searchmoves")  // Needs to be the last command on the line
			while (is >> token)
				limits.searchmoves.push_back(to_lower(token));

		else if (token == "wtime")
			is >> limits.time[WHITE];
		else if (token == "btime")
			is >> limits.time[BLACK];
		else if (token == "winc")
			is >> limits.inc[WHITE];
		else if (token == "binc")
			is >> limits.inc[BLACK];
		//else if (token == "movestogo")
		//	is >> limits.movestogo;
		else if (token == "depth")
			is >> limits.depth;
		else if (token == "nodes")
			is >> limits.nodes;
		else if (token == "movetime")
			is >> limits.movetime;
		else if (token == "mate")
			is >> limits.mate;
		else if (token == "perft")
			is >> limits.perft;
		else if (token == "infinite")
			limits.infinite = 1;
		else if (token == "ponder")
			limits.ponderMode = true;

	return limits;
}


// Called when the engine receives the "go" UCI command. The function sets the
// thinking time and other parameters from the input string then stars with a search

// go()は、思考エンジンがUSIコマンドの"go"を受け取ったときに呼び出される。
// この関数は、入力文字列から思考時間とその他のパラメーターをセットし、探索を開始する。
// ignore_ponder : これがtrueなら、"ponder"という文字を無視する。
void USIEngine::go(std::istringstream& is)
{
	Search::LimitsType limits = parse_limits(is);

	if (limits.perft)
		perft(limits);
	else
		engine.go(limits);
}

#if 0
auto& options = engine_options();

// "isready"コマンド受信前に"go"コマンドが呼び出されている。
if (!engine.eval_loaded)
{
	sync_cout << "info string Error! go cmd before isready cmd." << sync_endl;
	return;
}

Search::LimitsType limits;
string token;
bool ponderMode = false;

auto main_thread = Threads.main();

if (!states)
{
	// 前回から"position"コマンドを処理せずに再度goが呼び出された。
	// 前回、ponderでStochastic Ponderのために局面を壊してしまっている可能性があるので復元しておく。
	// (これがStochastic Ponderの一番簡単な実装)
	// Stochastic Ponderのために局面を2手前に戻して、そのあと現在の局面に対するコマンド("d"など)を実行すると
	// それは2手前の局面が表示されるが、それは仕様であるものとする。(これを修正するとプログラムのフローが複雑になる)
	istringstream iss(main_thread->last_position_cmd_string);
	iss >> token; // "position"
	position(*const_cast<Position*>(&pos), iss, states);
}

// 思考開始時刻の初期化。なるべく早い段階でこれをしておかないとサーバー時間との誤差が大きくなる。
Time.reset();

// 終局(引き分け)になるまでの手数
// 引き分けになるまでの手数。(Options["MaxMovesToDraw"]として与えられる。エンジンによってはこのオプションを持たないこともある。)
// 0のときは制限なしだが、これをint_maxにすると残り手数を計算するときに桁があふれかねないので100000を設定。

int max_game_ply = 0;
if (options.count("MaxMovesToDraw"))
max_game_ply = (int)options["MaxMovesToDraw"];

// これ0の時、何らか設定しておかないと探索部でこの手数を超えた時に引き分け扱いにしてしまうので、無限大みたいな定数の設定が必要。
limits.max_game_ply = (max_game_ply == 0) ? 100000 : max_game_ply;

#if defined (USE_ENTERING_KING_WIN)
// 入玉ルール
limits.enteringKingRule = to_entering_king_rule(Options["EnteringKingRule"]);
#endif

// すべての合法手を生成するのか
limits.generate_all_legal_moves = options["GenerateAllLegalMoves"];

// エンジンオプションによる探索制限(0なら無制限)
// このあと、depthもしくはnodesが指定されていたら、その値で上書きされる。(この値は無視される)

limits.depth = options.count("DepthLimit") ? (int)options["DepthLimit"] : 0;
limits.nodes = options.count("NodesLimit") ? (u64)options["NodesLimit"] : 0;

while (is >> token)
{
	// 探索すべき指し手。(探索開始局面から特定の初手だけ探索させるとき)
	// これ、Stockfishのコードでこうなっているからそのままにしてあるが、
	// これを指定しても定跡の指し手としてはこれ以外を指したりする問題はある。
	// またふかうら王ではこのオプションをサポートしていない。
	// ゆえに、非対応扱いで考えて欲しい。
	if (token == "searchmoves")
		// 残りの指し手すべてをsearchMovesに突っ込む。
		while (is >> token)
			limits.searchmoves.push_back(USIEngine::to_move(pos, token));

	// 先手、後手の残り時間。[ms]
	else if (token == "wtime")     is >> limits.time[WHITE];
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

		// USIプロトコルで送られてきた秒読み時間より少なめに思考する設定
		// ※　通信ラグがあるときに、ここで少なめに思考しないとタイムアップになる可能性があるので。

		// t = std::max(t - Options["ByoyomiMinus"], Time::point(0));

		// USIプロトコルでは、これが先手後手同じ値だと解釈する。
		limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = t;
	}
	// この探索深さで探索を打ち切る
	else if (token == "depth")     is >> limits.depth;

	// この探索ノード数で探索を打ち切る
	else if (token == "nodes")     is >> limits.nodes;

	// 持ち時間固定(将棋だと対応しているGUIが無いかもしれないが..)
	else if (token == "movetime")  is >> limits.movetime;

	// 詰み探索。"UCI"プロトコルではこのあとには手数が入っており、その手数以内に詰むかどうかを判定するが、
	// "USI"プロトコルでは、ここは探索のための時間制限に変更となっている。
	else if (token == "mate") {
		is >> token;
		if (token == "infinite")
			limits.mate = INT32_MAX;
		else
			// USIプロトコルでは、UCIと異なり、ここは手数ではなく、探索に使う時間[ms]が指定されている。
			limits.mate = stoi(token);
	}

	// パフォーマンステスト(Stockfishにある、合法手N手で到達できる局面を求めるやつ)
	// このあとposition～goコマンドを使うとパフォーマンステストモードに突入し、ここで設定した手数で到達できる局面数を求める
	else if (token == "perft")		is >> limits.perft;

	// 時間無制限。
	else if (token == "infinite")	limits.infinite = 1;

	// ponderモードでの思考。
	else if (token == "ponder" && !ignore_ponder) {
		ponderMode = true;

		if (options["Stochastic_Ponder"] && main_thread->moves_from_game_root.size() >= 1)
		{
			// 1手前の局面(相手番)に戻して、ponderとして思考する。
			// Threads.main()->moves_from_game_root に保存されているので大丈夫。

			auto m = main_thread->moves_from_game_root.back();
			main_thread->moves_from_game_root.pop_back();
			const_cast<Position*>(&pos)->undo_move(m);
			states->pop_back();
			main_thread->position_is_dirty = true;
		}
	}

	// --- やねうら王独自拡張

	// "wait_stop"指定。
	else if (token == "wait_stop")
		limits.wait_stop = true;

#if defined(TANUKI_MATE_ENGINE)
	// MateEngineのデバッグ用コマンド: 詰将棋の特定の変化に対する解析を効率的に行うことが出来る。
	//	cf.https ://github.com/yaneurao/YaneuraOu/pull/115

	else if (token == "matedebug") {
		string token = "";
		Move16 m;
		limits.pv_check.clear();
		while (is >> token && (m = USI::to_move16(token)).to_u16() != MOVE_NONE) {
			limits.pv_check.push_back(m);
		}
	}
#endif

}

// goコマンド、デバッグ時に使うが、そのときに"go btime XXX wtime XXX byoyomi XXX"と毎回入力するのが面倒なので
// デフォルトで1秒読み状態で呼び出されて欲しい。
//if (limits.byoyomi[BLACK] == 0 && limits.inc[BLACK] == 0 && limits.time[BLACK] == 0 && limits.rtime == 0)
//	limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = 1000;

// →　これやると、パラメーターなしで"go ponder"されて"ponderhit"したときに、byoyomi 1秒と錯覚する。

Threads.start_thinking(pos, states, limits, ponderMode);

#endif




bool USIEngine::usi_cmdexec(const std::string& cmd)
{
	istringstream is(cmd);
	string token;
	is >> skipws >> token;

	if (token == "quit" || token == "stop" || token == "gameover")
	{
		// USIプロトコルにはUCIプロトコルから、
		// gameover win | lose | draw
		// が追加されているが、stopと同じ扱いをして良いと思う。
		// これハンドルしておかないとponderが停止しなくて困る。
		// gameoverに対してbestmoveは返すべきではないのかも知れないが、
		// それを言えばstopにだって…。

#if defined(USE_GAMEOVER_HANDLER) || defined(YANEURAOU_ENGINE_DEEP)

		if (token == "gameover")
			// "gameover"コマンドに対するハンドラを呼び出したいのか？
			gameover_handler(cmd);

#endif
		// "stop"コマンドが来るとEngine.stop()が呼び出され、その結果threads.stop = trueとなる。
		engine.stop();

		// "quit"に対しては、この関数はtrueを返す。
		if (token == "quit")
			return true;
	}
#if 0
	else if (token == "ponderhit")
	{
		if (Options["Stochastic_Ponder"])
		{
			// Stochastic Ponder hit

			// まず探索スレッドを停止させる。
			// ただしこの時にbestmoveを返してはならないので、これはSearch::Limits.silentで抑制する。
			auto org = Search::Limits.silent;
			Search::Limits.silent = true;
			Threads.stop = true;
			// 終了を待機しないとsilentの解除ができない。
			Threads.main()->wait_for_search_finished();
			Search::Limits.silent = org;

			// 前回と同様のgoコマンドをそのまま送る。ただし"ponder"の文字は無視する。
			// last_go_cmd_stringには先頭に"go"の文字があるが、それはgo_cmdのなかで無視されるので気にしなくて良い。
			istringstream iss(Threads.main()->last_go_cmd_string);
			go_cmd(pos, iss, states, true);
		}
		else {
			// ponderhitに追加パラメーターがあるか？(USI拡張プロトコル)

#if defined(USE_TIME_MANAGEMENT)
			bool token_processed = parse_ponderhit(is);
			// 追加パラメーターを処理したなら今回の思考時間を再計算する。
			if (token_processed)
				Time.reinit();
#endif

			// 通常のponder
			Time.reset_for_ponderhit();     // ponderhitから計測しなおすべきである。
			Threads.main()->ponder = false; // 通常探索に切り替える。
		}
	}


	// 与えられた局面について思考するコマンド
	else if (token == "go") {
		Threads.main()->last_go_cmd_string = cmd;       // Stochastic_Ponderで使うので保存しておく。
		go_cmd(pos, is, states);
	}

	// (思考などに使うための)開始局面(root)を設定する
	else if (token == "position") {
		Threads.main()->last_position_cmd_string = cmd; // 保存しておく。
		position_cmd(pos, is, states);
	}

	// "usinewgame"はゲーム中にsetoptionなどを送らないことを宣言するためのものだが、
	// 我々はこれに関知しないので単に無視すれば良い。
	// やねうら王では、時間のかかる初期化はisreadyの応答でやっている。
	// Stockfishでは、Search::clear() (時間のかかる処理)をここで呼び出しているようだが。
	// そもそもで言うと、"usinewgame"に対してはエンジン側は何ら応答を返さないので、
	// GUI側は、エンジン側が処理中なのかどうかが判断できない。
	// なのでここで長い時間のかかる処理はすべきではないと思うのだが。
	else if (token == "usinewgame") return;

	// 以下、デバッグのためのカスタムコマンド(非USIコマンド)
	// 探索中には使わないようにすべし。

	// 現在の局面について評価関数を呼び出して、その値を返す。
	else if (token == "eval") cout << "eval = " << Eval::compute_eval(pos) << endl;
	else if (token == "evalstat") Eval::print_eval_stat(pos);

	// この実行ファイルをコンパイルしたコンパイラの情報を出力する。
	else if (token == "compiler") sync_cout << compiler_info() << sync_endl;

	// -- 以下、やねうら王独自拡張のカスタムコマンド

	// config.hで設定した値などについて出力する。
	else if (token == "config") sync_cout << config_info() << sync_endl;


	// 指し手生成祭りの局面をセットする。
	else if (token == "matsuri") pos.set("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1", &states->back(), Threads.main());

	// "position sfen"の略。
	else if (token == "sfen") position_cmd(pos, is, states);

	// ログファイルの書き出しのon
	// 備考)
	// Stockfishの方は、エンジンオプションでログの出力ファイル名を指定できるのだが、
	// ログ自体はホスト側で記録することが多いので、ファイル名は固定でいいや…。
	else if (token == "log") start_logger("io_log.txt");

#if defined(EVAL_LEARN)
	// テスト用にqsearch(),search()を直接呼ぶコマンド
	else if (token == "qsearch") qsearch_cmd(pos);
	else if (token == "search") search_cmd(pos, is);
#endif


	// この局面の手番側がどちらであるかを返す。BLACK or WHITE
	else if (token == "side") cout << (pos.side_to_move() == BLACK ? "black" : "white") << endl;

	// この局面が詰んでいるかの判定
	else if (token == "mated") cout << pos.is_mated() << endl;

	// この局面のhash keyの値を出力
	else if (token == "key") cout << hex << pos.state()->hash_key() << dec << endl;

	// 探索の終了を待機するコマンド("stop"は送らずに。goコマンドの終了を待機できて便利。)
	else if (token == "wait") Threads.main()->wait_for_search_finished();

	// 一定時間待機するコマンド。("quit"の前に一定時間待ちたい時などに用いる。sleep 1000 == 1秒待つ)
	else if (token == "sleep") { u64 ms; is >> ms; Tools::sleep(ms); }

#if defined(MATE_1PLY) && defined(LONG_EFFECT_LIBRARY)
	// この局面での1手詰め判定
	else if (token == "mate1") cout << pos.mate1ply() << endl;
#endif

#if defined (ENABLE_MAKEBOOK_CMD) && (defined(EVAL_LEARN) || defined(YANEURAOU_ENGINE_DEEP))
	// 定跡を作るコマンド
	else if (token == "makebook") Book::makebook_cmd(pos, is);
#endif

#if defined (EVAL_LEARN)
	else if (token == "gensfen") Learner::gen_sfen(pos, is);
	else if (token == "learn") Learner::learn(pos, is);

#if defined (GENSFEN2019)
	// 開発中の教師局面生成コマンド
	else if (token == "gensfen2019") Learner::gen_sfen2019(pos, is);
#endif

#endif

#endif

	// 起動時いきなりこれが飛んでくるので速攻応答しないとタイムアウトになる。
	else if (token == "usi")
        engine.usi();

	// オプションを設定する
	else if (token == "setoption")
		setoption(is);

	else if (token == "go")
	{
		// send info strings after the go command is sent for old GUIs and python-chess
		// 古いGUIやpython-chessのために、goコマンド送信後にinfo文字列を送信する。

		//print_info_string(engine.numa_config_information_as_string());
		//print_info_string(engine.thread_allocation_information_as_string());

		// ⇨  以下のようなメッセージを出力する。要らんと思う..。
		//		info string Available processors : 0 - 31
		//		info string Using 4 thread

		go(is);
	}

	// ベンチコマンド(これは常に使える)
	else if (token == "bench")
		bench(is);

	else if (token == "position")
		position(is);

	else if (token == "usinewgame")
		engine.usinewgame();

	// 思考エンジンの準備が出来たかの確認
	else if (token == "isready")
		engine.isready();

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
		//std::pair<std::optional<std::string>, std::string> files[2];

		//if (is >> std::skipws >> files[0].second)
		//	files[0].first = files[0].second;

		//if (is >> std::skipws >> files[1].second)
		//	files[1].first = files[1].second;

		std::string file;
		is >> std::skipws >> file;
		engine.save_network(file);
	}

	// 📌 以下、やねうら王独自拡張 📌

	// fileの内容をUSIコマンドとして実行する。
	else if (token == "f")
		enqueue_command_from_file(is);

	// この局面での指し手をすべて出力
	else if (token == "moves")
		moves();

	// オプションを取得する
	else if (token == "getoption")
		getoption(is);

	// UnitTest
	else if (token == "unittest")
		unittest(is);

#if defined (ENABLE_TEST_CMD)
		// テストコマンド
	else if (token == "test") Test::test_cmd(engine, is);
#endif

	// ユーザーによる実験用コマンド。Engine::user_cmd()が呼び出される。
	else if (token == "user")
		engine.user(is);


	// エンジンオプションの簡易変更機能
	else {
		/*
			簡略表現として、
			> threads 1
			    のように指定したとき、
			> setoption name Threads value 1
			    と等価なようにしておく。
		*/

		if (!token.empty())
		{
			string value;
			is >> value;
			sync_cout << engine.get_options().set_option_if_exists(token, value) << sync_endl;
		}
	}
	return false;
}




// Square型をUSI文字列に変換する
std::string USIEngine::square(Square s) {
	return std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
}


// 指し手をUSI文字列に変換する。
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

// string全体を小文字化して返す。
std::string USIEngine::to_lower(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), [](auto c) { return std::tolower(c); });

	return str;
}

// USIの指し手文字列をMove型の変換する。
// 合法手でなければMove::noneを返すようになっている。
// 💡 合法でない指し手の場合、エラーである旨を出力する。
Move USIEngine::to_move(const Position& pos, std::string str)
{
	//str = to_lower(str);
	// ⇨  将棋(USIプロトコル)では、駒打ちの時に大文字と小文字の区別があるので、
	//	  小文字化して比較することはできない。

	// 全合法手のなかからusi文字列に変換したときにstrと一致する指し手を探してそれを返す
	//for (const auto& m : MoveList<LEGAL_ALL>(pos))
	//	if (str == move(m))
	//		return m;
	// ↑のコードは大変美しいコードではあるが、棋譜を大量に読み込むときに時間がかかるうるのでもっと高速な実装をする。

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
Move16 USIEngine::to_move16(const string& str)
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
Square USIEngine::usi_to_sq(char f, char r)
{
	File file = toFile(f);
	Rank rank = toRank(r);

	if (is_ok(file) && is_ok(rank))
		return file | rank;

	return SQ_NB;
}


// USIプロトコルのマス目文字列をSquare型に変換する。
// 変換できない文字である場合、SQ_NBを返す。
Square      USIEngine::to_square(const std::string& str)
{
	if (str.size() != 2)
		return SQ_NB;
	return usi_to_sq(str[0], str[1]);
}

// "bench"コマンドの応答部。
void USIEngine::bench(std::istream& args) {
#if 0
	std::string token;
	uint64_t    num, nodes = 0, cnt = 1;
	uint64_t    nodesSearched = 0;
	const auto& options = engine.get_options();

	engine.set_on_update_full([&](const auto& i) {
		nodesSearched = i.nodes;
		on_update_full(i, options["UCI_ShowWDL"]);
		});

	std::vector<std::string> list = Benchmark::setup_bench(engine.fen(), args);

	num = count_if(list.begin(), list.end(),
		[](const std::string& s) { return s.find("go ") == 0 || s.find("eval") == 0; });

	TimePoint elapsed = now();

	for (const auto& cmd : list)
	{
		std::istringstream is(cmd);
		is >> std::skipws >> token;

		if (token == "go" || token == "eval")
		{
			std::cerr << "\nPosition: " << cnt++ << '/' << num << " (" << engine.fen() << ")"
				<< std::endl;
			if (token == "go")
			{
				Search::LimitsType limits = parse_limits(is);

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
		{
			engine.search_clear();  // search_clear may take a while
			elapsed = now();
		}
	}

	elapsed = now() - elapsed + 1;  // Ensure positivity to avoid a 'divide by zero'

	dbg_print();

	std::cerr << "\n==========================="    //
		<< "\nTotal time (ms) : " << elapsed  //
		<< "\nNodes searched  : " << nodes    //
		<< "\nNodes/second    : " << 1000 * nodes / elapsed << std::endl;

	// reset callback, to not capture a dangling reference to nodesSearched
	engine.set_on_update_full([&](const auto& i) { on_update_full(i, options["UCI_ShowWDL"]); });

#endif

}

void USIEngine::benchmark(std::istream& args) {
	// Probably not very important for a test this long, but include for completeness and sanity.
	static constexpr int NUM_WARMUP_POSITIONS = 3;

	std::string token;
	uint64_t    nodes = 0, cnt = 1;
	uint64_t    nodesSearched = 0;

#if 0
	engine.set_on_update_full([&](const Engine::InfoFull& i) { nodesSearched = i.nodes; });

	engine.set_on_iter([](const auto&) {});
	engine.set_on_update_no_moves([](const auto&) {});
	engine.set_on_bestmove([](const auto&, const auto&) {});
	engine.set_on_verify_networks([](const auto&) {});

	Benchmark::BenchmarkSetup setup = Benchmark::setup_benchmark(args);

	const int numGoCommands = count_if(setup.commands.begin(), setup.commands.end(),
		[](const std::string& s) { return s.find("go ") == 0; });

	TimePoint totalTime = 0;

	// Set options once at the start.
	auto ss = std::istringstream("name Threads value " + std::to_string(setup.threads));
	setoption(ss);
	ss = std::istringstream("name Hash value " + std::to_string(setup.ttSize));
	setoption(ss);
	ss = std::istringstream("name UCI_Chess960 value false");
	setoption(ss);

	// Warmup
	for (const auto& cmd : setup.commands)
	{
		std::istringstream is(cmd);
		is >> std::skipws >> token;

		if (token == "go")
		{
			// One new line is produced by the search, so omit it here
			std::cerr << "\rWarmup position " << cnt++ << '/' << NUM_WARMUP_POSITIONS;

			Search::LimitsType limits = parse_limits(is);

			TimePoint elapsed = now();

			// Run with silenced network verification
			engine.go(limits);
			engine.wait_for_search_finished();

			totalTime += now() - elapsed;

			nodes += nodesSearched;
			nodesSearched = 0;
		}
		else if (token == "position")
			position(is);
		else if (token == "ucinewgame")
		{
			engine.search_clear();  // search_clear may take a while
		}

		if (cnt > NUM_WARMUP_POSITIONS)
			break;
	}

	std::cerr << "\n";

	cnt = 1;
	nodes = 0;

	int           numHashfullReadings = 0;
	constexpr int hashfullAges[] = { 0, 999 };  // Only normal hashfull and touched hash.
	int           totalHashfull[std::size(hashfullAges)] = { 0 };
	int           maxHashfull[std::size(hashfullAges)] = { 0 };

	auto updateHashfullReadings = [&]() {
		numHashfullReadings += 1;

		for (int i = 0; i < static_cast<int>(std::size(hashfullAges)); ++i)
		{
			const int hashfull = engine.get_hashfull(hashfullAges[i]);
			maxHashfull[i] = std::max(maxHashfull[i], hashfull);
			totalHashfull[i] += hashfull;
		}
		};

	engine.search_clear();  // search_clear may take a while

	for (const auto& cmd : setup.commands)
	{
		std::istringstream is(cmd);
		is >> std::skipws >> token;

		if (token == "go")
		{
			// One new line is produced by the search, so omit it here
			std::cerr << "\rPosition " << cnt++ << '/' << numGoCommands;

			Search::LimitsType limits = parse_limits(is);

			TimePoint elapsed = now();

			// Run with silenced network verification
			engine.go(limits);
			engine.wait_for_search_finished();

			totalTime += now() - elapsed;

			updateHashfullReadings();

			nodes += nodesSearched;
			nodesSearched = 0;
		}
		else if (token == "position")
			position(is);
		else if (token == "ucinewgame")
		{
			engine.search_clear();  // search_clear may take a while
		}
	}

	totalTime = std::max<TimePoint>(totalTime, 1);  // Ensure positivity to avoid a 'divide by zero'

	dbg_print();

	std::cerr << "\n";

	static_assert(
		std::size(hashfullAges) == 2 && hashfullAges[0] == 0 && hashfullAges[1] == 999,
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
		<< "\nTotal search time [s]      : " << totalTime / 1000.0
		<< "\nNodes/second               : " << 1000 * nodes / totalTime << std::endl;

	// clang-format on

	init_search_update_listeners();

#endif
}


// "setoption"コマンド応答。
void USIEngine::setoption(istringstream& is)
{
	engine.wait_for_search_finished();
	engine_options().setoption(is);
}

std::uint64_t USIEngine::perft(const Search::LimitsType& limits) {
	auto nodes = engine.perft(engine.sfen(), limits.perft /*, engine.get_options()["UCI_Chess960"]*/);
	sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;
	return nodes;
}

// ----------------------------------
//      USI拡張コマンド "test"
// ----------------------------------

#if defined(ENABLE_TEST_CMD)

// USI拡張コマンドのうち、開発上のテスト関係のコマンド。
// 思考エンジンの実行には関係しない。

namespace Test
{
	// 通常のテスト用コマンド。コマンドを処理した時 trueが返る。
	bool normal_test_cmd(IEngine& engine, std::istringstream& is, const std::string& token);

	// 詰み関係のテスト用コマンド。コマンドを処理した時 trueが返る。
	bool mate_test_cmd(IEngine& engine, std::istringstream& is, const std::string& token);

	void test_cmd(IEngine& engine, std::istringstream& is)
	{
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

#endif // defined(ENABLE_TEST_CMD)

//
// あとで整理する
//


// ユーザーの実験用に開放している関数。
// USI拡張コマンドで"user"と入力するとこの関数が呼び出される。
// "user"コマンドの後続に指定されている文字列はisのほうに渡される。
void user_test(Position& pos, std::istringstream& is);

#if defined(ENABLE_TEST_CMD)
	void generate_moves_cmd(Position& pos);
#endif

#if defined(USE_MATE_DFPN)
// "mate"コマンド
void mate_cmd(Position& pos, istream& is);
#endif

// ----------------------------------
//      USI拡張コマンド "makebook"
// ----------------------------------

// 定跡を作るコマンド
#if defined (ENABLE_MAKEBOOK_CMD) && (defined(EVAL_LEARN) || defined(YANEURAOU_ENGINE_DEEP))
namespace Book { void makebook_cmd(Position& pos, istringstream& is); }
#endif

// ----------------------------------
//      USI拡張コマンド "learn"
// ----------------------------------

// 棋譜を自動生成するコマンド
#if defined (EVAL_LEARN)
namespace Learner
{
  // 教師局面の自動生成
  void gen_sfen(Position& pos, istringstream& is);

  // 生成した棋譜からの学習
  void learn(Position& pos, istringstream& is);

#if defined(GENSFEN2019)
  // 開発中の教師局面の自動生成コマンド
  void gen_sfen2019(Position& pos, istringstream& is);
#endif

  // 読み筋と評価値のペア。Learner::search(),Learner::qsearch()が返す。
  typedef std::pair<Value, std::vector<Move> > ValuePV;

  ValuePV qsearch(Position& pos);
  ValuePV search(Position& pos, int depth_, size_t multiPV = 1 , u64 nodesLimit = 0 );

}
#endif


// "gameover"コマンドに対するハンドラ
#if defined(USE_GAMEOVER_HANDLER) || defined(YANEURAOU_ENGINE_DEEP)
void gameover_handler(const string& cmd);
#endif

// --------------------
// USI関係のコマンド処理
// --------------------

// "position"コマンドのhandler
void USIEngine::position(std::istringstream& is)
{
	string token, sfen;

	is >> token;

	if (token == "startpos")
	{
		// 初期局面として初期局面のFEN形式の入力が与えられたとみなして処理する。
		sfen = StartSFEN;
		is >> token; // もしあるなら"moves"トークンを消費する。
	}
	// 局面がfen形式で指定されているなら、その局面を読み込む。
	// UCI(チェスプロトコル)ではなくUSI(将棋用プロトコル)だとここの文字列は"fen"ではなく"sfen"
	else {
		// 💡 この"sfen"という文字列は省略可能にしたいので
		//     Stockfishのコードを少し工夫して書き換える。

		// "sfen"なら吸い込むが、"sfen"でないなら、それを局面文字列の一部とみなす。
		if (token != "sfen")
			sfen += token + " ";

		while (is >> token && token != "moves")
			sfen += token + " ";
	}

	std::vector<std::string> moves;

	// 指し手のリストをパースする(あるなら)
	while (is >> token)
	{
		moves.push_back(token);
	}

	engine.set_position(sfen, moves);
}


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


#if defined(EVAL_LEARN)
void qsearch_cmd(Position& pos)
{
	cout << "qsearch : ";
	auto pv = Learner::qsearch(pos);
	cout << "Value = " << pv.first << " , PV = ";
	for (auto m : pv.second)
		cout << m << " ";
	cout << endl;
}

void search_cmd(Position& pos, istringstream& is)
{
	string token;
	int depth = 1;
	int multi_pv = (int)Options["MultiPV"];
	while (is >> token)
	{
		if (token == "depth")
			is >> depth;
		if (token == "multipv")
			is >> multi_pv;
	}

	cout << "search depth = " << depth << " , multi_pv = " << multi_pv << " : ";
	auto pv = Learner::search(pos, depth, multi_pv);
	cout << "Value = " << pv.first << " , PV = ";
	for (auto m : pv.second)
		cout << m << " ";
	cout << endl;
}

#endif

// --------------------
//   USI拡張コマンド
// --------------------

// "unittest"コマンド
void USIEngine::unittest(std::istringstream& is)
{
	Test::UnitTest(is, engine);
}

// getoptionコマンド応答(USI独自拡張)
// オプションの値を取得する。
void USIEngine::getoption(istringstream& is)
{
	auto& options = engine_options();

	// getoption オプション名
	string option_name = "";
	is >> option_name;
	sync_cout << options.get_option(option_name) << sync_endl;
}

#if 0
// isreadyコマンド処理部
void USIEngine::isready()
{
	// 対局ごとに"isready","usinewgame"の両方が来る。
	// "isready"が起動後に1度だけしか来ないようなGUI実装は、
	// 実装上の誤りであるから修正すべきである。)

	// 少なくとも将棋のGUI(将棋所、ShogiGUI、将棋神やねうら王)では、
	// "isready"が毎回来るようなので、"usinewgame"のほうは無視して、
	// "isready"に応じて評価関数、定跡、探索部を初期化する。

	auto& options = engine_options();

	// EvalDirにある"eval_options.txt"を読み込む。
	// ここに評価関数に応じた設定を書いておくことができる。

	options.read_engine_options(Path::Combine(options["EvalDir"], "eval_options.txt"));

	// yaneuraou.wasm
	// ブラウザのメインスレッドをブロックしないよう、Keep Alive処理をコメントアウト
#if !defined(__EMSCRIPTEN__)
	// --- Keep Alive的な処理 ---

	// "isready"を受け取ったあと、"readyok"を返すまで5秒ごとに改行を送るように修正する。(keep alive的な処理)
	// →　これ、よくない仕様であった。
	// cf. USIプロトコルでisready後の初期化に時間がかかる時にどうすれば良いのか？
	//     http://yaneuraou.yaneu.com/2020/01/05/usi%e3%83%97%e3%83%ad%e3%83%88%e3%82%b3%e3%83%ab%e3%81%a7isready%e5%be%8c%e3%81%ae%e5%88%9d%e6%9c%9f%e5%8c%96%e3%81%ab%e6%99%82%e9%96%93%e3%81%8c%e3%81%8b%e3%81%8b%e3%82%8b%e6%99%82%e3%81%ab%e3%81%a9/
	// cf. isready後のkeep alive用改行コードの送信について
	//		http://yaneuraou.yaneu.com/2020/03/08/isready%e5%be%8c%e3%81%aekeep-alive%e7%94%a8%e6%94%b9%e8%a1%8c%e3%82%b3%e3%83%bc%e3%83%89%e3%81%ae%e9%80%81%e4%bf%a1%e3%81%ab%e3%81%a4%e3%81%84%e3%81%a6/

	// これを送らないと、将棋所、ShogiGUIでタイムアウトになりかねない。
	// ワーカースレッドを一つ生成して、そいつが5秒おきに改行を送信するようにする。
	// このあと重い処理を行うのでスレッドの起動が遅延する可能性があるから、先にスレッドを生成して、そのスレッドが起動したことを
	// 確認してから処理を行う。

	// スレッドが起動したことを通知するためのフラグ
	auto thread_started = false;

	// この関数を抜ける時に立つフラグ(スレッドを停止させる用)
	auto thread_end = false;

	// 定期的な改行送信用のスレッド
	auto th = std::thread([&] {
		// スレッドが起動した
		thread_started = true;

		int count = 0;
		while (!thread_end)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			if (++count >= 50 /* 5秒 */)
			{
				count = 0;
				sync_cout << sync_endl; // 改行を送信する。

				// 定跡の読み込み部などで"info string.."で途中経過を出力する場合、
				// sync_cout ～ sync_endlを用いて送信しないと、この改行を送るタイミングとかち合うと
				// 変なところで改行されてしまうので注意。
			}
		}
		});
	SCOPE_EXIT({ thread_end = true; th.join(); });

	// スレッド起動待ち
	while (!thread_started)
		Tools::sleep(100);

	// --- Keep Alive的な処理ここまで ---
#endif

	// スレッドを先に生成しないとUSI_Hashで確保したメモリクリアの並列化が行われなくて困る。

#if defined(YANEURAOU_ENGINE_DEEP)

	// ここ、max_gpu == 8固定として扱っている。あとで修正する。(かも)
	int threads_num =
		(int)Options["UCT_Threads1"] + (int)Options["UCT_Threads2"] + (int)Options["UCT_Threads3"] + (int)Options["UCT_Threads4"] +
		(int)Options["UCT_Threads5"] + (int)Options["UCT_Threads6"] + (int)Options["UCT_Threads7"] + (int)Options["UCT_Threads8"];

	Threads.set(std::max(threads_num,1));
#else
	//Threads.set(size_t(options["Threads"]));
#endif

#if defined (USE_EVAL_HASH)
	Eval::EvalHash_Resize(Options["EvalHash"]);
#endif

	// Engine側のisready callback
	engine.isready();

#if 0
	// 評価関数の読み込み

#if defined(YANEURAOU_ENGINE_DEEP)

	// 毎回、load_eval()は呼び出すものとする。
	// モデルファイル名に変更がなければ、再読み込みされないような作りになっているならばこの実装のほうがシンプル。
	Eval::load_eval();
	engine.engine.eval_loaded = true;

#else

	// 評価関数の読み込みなど時間のかかるであろう処理はこのタイミングで行なう。
	// 起動時に時間のかかる処理をしてしまうと将棋所がタイムアウト判定をして、思考エンジンとしての認識をリタイアしてしまう。
	engine.verify_networks();
#endif

	// isreadyに対してはreadyokを返すまで次のコマンドが来ないことは約束されているので
	// このタイミングで各種変数の初期化もしておく。

#if defined(YANEURAOU_ENGINE_DEEP)
	// ふかうら王では置換表は用いない。
#else
	//TT.resize(size_t(options["USI_Hash"]));
#endif

	//Search::clear();

#if defined (USE_EVAL_HASH)
	Eval::EvalHash_Clear();
#endif

	//Threads.stop = false;

	// Positionコマンドが送られてくるまで評価値の全計算をしていないの気持ち悪いのでisreadyコマンドに対して
	// evalの値を返せるようにこのタイミングで平手局面で初期化してしまう。

#endif

	sync_cout << "readyok" << sync_endl;

}
#endif


// "moves"コマンドのhandler
void USIEngine::moves()
{
	auto& pos = engine.get_position();
	for (auto m : MoveList<LEGAL_ALL>(pos))
		cout << Move(m) << ' ';
	cout << endl;
}

// --------------------
//   やねうら王独自
// --------------------

// コマンドラインと"startup.txt"に書かれているUSIコマンドをstd_inputに積む。
void USIEngine::enqueue_startup_command()
{
	// コマンドラインから積まれたコマンドをstd_inputに積んでやる。
	std_input.parse_args(CommandLine::g);

	// "startup.txt"というファイルがあれば、この内容を実行してやる。
	// そのため、std_inputにそこに書かれているコマンドを積んでやる。
	const string startup = "startup.txt";
	vector<string> lines;
	if (SystemIO::ReadAllLines(startup, lines).is_ok())
	{
		for (auto& line : lines)
			std_input.push(line);
	}
}

// ファイルからUSIコマンドをstd_inputに積む。
void USIEngine::enqueue_command_from_file(std::istringstream& is)
{
	string filename = "";
	is >> filename;
	if (!filename.empty())
	{
		filename += ".txt";
		sync_cout << "USI Commands from File = " << filename << sync_endl;

		vector<string> lines;
		if (SystemIO::ReadAllLines(filename, lines).is_ok())
			for (auto& line : lines)
				std_input.push(line);
		else
			sync_cout << "Error : File Not Found." << sync_endl;
	}
}

// Score構造体の内容をUSI形式のscoreとして出力する。
std::string USIEngine::format_score(const Score& s) {
    constexpr int TB_CP = 20000;
    const auto    format =
      overload{[](Score::Mate mate) -> std::string {
                   auto m = (mate.plies > 0 ? (mate.plies + 1) : mate.plies) / 2;
                   return std::string("mate ") + std::to_string(m);
               },
               //[](Score::Tablebase tb) -> std::string {
               //    return std::string("cp ")
               //         + std::to_string((tb.win ? TB_CP - tb.plies : -TB_CP - tb.plies));
               //},
               [](Score::InternalUnits units) -> std::string {
                   return std::string("cp ") + std::to_string(units.value);
               }};

    return s.visit(format);
}


// → やねうら王の場合、PawnValue = 90なので Value = 90なら 100として出力する必要がある。
// Stockfish 16ではこの値は328になっている。
constexpr int NormalizeToPawnValue = Eval::PawnValue;

/// Turns a Value to an integer centipawn number,
/// without treatment of mate and similar special scores.
// 詰みやそれに類似した特別なスコアの処理なしに、Valueを整数のセントポーン数に変換する。
int USIEngine::to_cp(Value v) {

  return 100 * v / NormalizeToPawnValue;
}

// cpからValueへ。⇑の逆変換。
Value USIEngine::cp_to_value(int v)
{
	return Value((std::abs(v) < VALUE_MATE_IN_MAX_PLY) ? (NormalizeToPawnValue * v / 100) : v);
}

// スコアを歩の価値を100として正規化して出力する。
//   MATEではないスコアなら"cp x"のように出力する。
//   MATEのスコアなら、"mate x"のように出力する。
// ⚠ USE_PIECE_VALUEが定義されていない時は正規化しようがないのでこの関数は呼び出せない。
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

// namespace USI内のUnitTest。
void USIEngine::UnitTest(Test::UnitTester& tester, IEngine& engine)
{
	auto section1 = tester.section("USI");

	Position pos;
	StateInfo si;

	// 平手初期化
	auto hirate_init = [&] { pos.set_hirate(&si); };

	// SFEN文字列でのPosition初期化
	auto sfen_init = [&](const string& sfen) { pos.set(sfen, &si); };

	// いまから、global_optionsを書き換えるので、あとで元に戻す必要がある。
	auto options_backup = global_options;
	SCOPE_EXIT({ global_options = options_backup; });

	{
		auto section2 = tester.section("to_move()");
		{
			//auto section3 = tester.section("unpromoted pawn move");

			sfen_init("2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1");

			// いま不成を生成するオプションがオフであると仮定する。
			global_options.generate_all_legal_moves = false;

			auto moves = "6a5b 1g1f 4a3b 1f1e 3a2b 2g2f 2c2d 7g7f 3b2c 2f2e 2d2e 2h2e P*2d 2e9e 7a8b 8h6f 8c8d 3i3h 5b6b 7i6h 6b7b 9e9f 7b8c 6h7g 7c7d 7f7e 7d7e 7g8f 8b7c 1e1d 1c1d 8f7e P*7d 7e8f 5a5b P*7b 5b6b 7b7a+ 6b7a P*7e 7a6b 7e7d 7c7d 8f9e P*7e 9f8f 8d8e 8f9f 5c5d 5g5f 6c6d 5f5e 6b6c 5e5d 6c5d 8i7g 6d6e 6f5g 2c3d 6i7h 3d4e 4g4f P*5f 5g6h 4e5e 3h4g 6e6f 4f4e 9c9d 9e9d 8c8d 6g6f P*9e 9d9c 8d9d P*5g 9e9f P*2c 2b2c 5g5f 5e4e P*4f 4e4d 4f4e 4d4e 9c8b 7e7f 7g6e 7d7e P*4f 4e4d 4f4e 4d4e 7h6g R*8i 5i4h 8i9i+ P*4f 4e4d 4f4e 4d4e 6e7c+ 4c4d 7c7d 9d8d 7d8d 7e8d 4g4f 4e4f 6h4f L*4e 6g5g 4e4f 5g4f 5d4c 5f5e P*5c G*2b 2c3d L*5i 4d4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f 4e4f 3f4f P*4e 4f3f N*4d 4h3h 4d3f 3g3f 4e4f P*2g S*4g 3h2h 4g3f P*4d 4c4d P*4h 2d2e P*3g 3f2g 2h2g 2e2f 2g2f 3d3e 2f2e G*2d 2e1f 1d1e 1f1g P*2f N*3f 3e3f 3g3f B*2g S*3h 2g3f+ 1g2h N*3e S*1h 2f2g+ 1h2g 3e2g+ 3h2g 3f6c N*3f 4d3d 3f2d P*2f 2g2f S*2g 2h3i P*2h G*3g 2h2i+ 3i2i P*2h 2i3i N*5g 3g4f 5g4i+ 3i4i G*6h P*7i 9i7i G*5h 6h5h 4i5h 7i7h G*6h G*6g 5h4g 7h6h N*3f G*5g 4g3g 6h4h 3g2g 4h4f P*3g 3d4e S*3e 4f4g P*4i 4e5f 4i4h 4g4h 5i5g 5f5g 1i1e 5g5h 2g1f 5h5i P*4f 4h4i 1e1c+ 4i5h 1f1e 5h4g G*3i 4g5h 8b7a+ 6c6d 1e1d 6d5e 8g8f 8e8f P*8g 8f8g";
			// ↑この局面、最後の8f8gが歩の不成だが、これがUSI::to_move()で非合法手扱いされないかをテストする。
			// cf. https://github.com/yaneurao/YaneuraOu/issues/190

			istringstream is(moves);
			string token;
			bool fail = false;

			StateInfo si[512];
			while (is >> token)
			{
				Move m = USIEngine::to_move(pos,token);
				if (m == Move::none())
					fail = true;

				pos.do_move(m, si[pos.game_ply()]);
			}

			tester.test("pawn's unpromoted move",!fail);

		}

	}
}

#if defined(__EMSCRIPTEN__)
// --------------------
// EMSCRIPTEN support
// --------------------
static StateListPtr states(new StateList(1));

// USI応答部 emscriptenインターフェース
EMSCRIPTEN_KEEPALIVE extern "C" int usi_command(const char *c_cmd) {
	std::string cmd(c_cmd);

	static Position pos;
	string token;

	for (Thread* th : Threads) {
		if (!th->threadStarted)
			return 1;
	}

	usi_cmdexec(pos, states, cmd);

	return 0;
}
#endif

void USIEngine::on_update_no_moves(const Engine::InfoShort& info) {
    sync_cout << "info depth " << info.depth << " score " << format_score(info.score) << sync_endl;
}

void USIEngine::on_update_full(const Engine::InfoFull& info, bool showWDL) {
    std::stringstream ss;

    ss << "info";
    ss << " depth " << info.depth                 //
       << " seldepth " << info.selDepth           //
       << " multipv " << info.multiPV             //
       << " score " << format_score(info.score);  //

    //if (showWDL)
    //    ss << " wdl " << info.wdl;

    if (!info.bound.empty())
        ss << " " << info.bound;

    ss << " nodes " << info.nodes        //
       << " nps " << info.nps            //
       << " hashfull " << info.hashfull  //
       //<< " tbhits " << info.tbHits      //
       << " time " << info.timeMs        //
       << " pv " << info.pv;             //

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


} // namespace YaneuraOu
