#ifndef USI_H_INCLUDED
#define USI_H_INCLUDED

#include <iostream>
#include <string>
#include <string_view>

#include "engine.h"
#include "misc.h"
#include "search.h"

#if !STOCKFISH
#if defined(__EMSCRIPTEN__)
#include <memory>
#endif
#include <queue>
#endif

namespace YaneuraOu {

// --------------------
//     USI関連
// --------------------

#if STOCKFISH
class Position;
class Move;
class Score;
enum Square : u8;
using Value = int;

constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#endif

// USIEngine本体。
// このclassがEngineを内包していて、USIメッセージのハンドラを処理して
// 内包しているEngineに対して司令を送る。
class USIEngine {
public:

#if STOCKFISH
	// 📝 やねうら王では、CommandLine::gを見れば良いので、argc,argvは引数として渡さないことにした。
	//     また、ここでやっている初期化は不要になったのでコンストラクタ自体を削除。
	USIEngine(CommandLine cli);
#else
	 // USIEngine classから使うEngine。
	 // 📌 やねうら王独自。エンジンの実装を変更できるように、
	 //     IEngine(エンジン interface)を渡し、エンジンを動的に切り替えたり、
	 //     複数の異なるエンジンから成るUSIEngineを同時に使うことができるようにする。
	 void set_engine(IEngine& _engine);
#endif

	// main threadをUSIメッセージの受信のために待機させる。
	// "quit"コマンドが送られてくるまでこのループは抜けない。
	// 📝 wasmではブラウザのメインスレッドをブロックできないため、この関数は何もしない。
	//     代わりにwasm_pre.jsからusi_command()経由でexec_command_from_js()が呼び出される。
	void loop();

	// --------------------
	// USI関係の記法変換部
	// --------------------

	// 詰みやそれに類似した特別なスコアの処理なしに、Valueを整数のセントポーン数に変換する。
#if STOCKFISH
	static int         to_cp(Value v, const Position& pos);
    // 📝 やねうら王では、Position&は不要。
#else
	static int   to_cp(Value v);
#endif

    // USIプロトコルで用いるscoreにScore構造体の内容を変換する。
    static std::string format_score(const Score& s);

	// USIプロトコルで使うマス目文字列に変換する。
    static std::string square(Square s);

    // USIプロトコルで使う指し手文字列に変換する。
#if STOCKFISH
    static std::string move(Move m, bool chess960 = false);
#else
    static std::string move(Move m);
#endif

#if STOCKFISH
    // 勝率文字列に変換する。
    // 📌 将棋では評価値をcpで出力するので不要。
    static std::string wdl(Value v, const Position& pos);
#endif

	// string全体を小文字化して返す。
    static std::string to_lower(std::string str);

    // USIの指し手文字列をMove型の変換する。
    // 合法手でなければMove::noneを返すようになっている。
    // 💡 合法でない指し手の場合、エラーである旨を出力する。
    static Move to_move(const Position& pos, std::string str);

#if !STOCKFISH
	// USI形式から指し手への変換。本来この関数は要らないのだが、
    // 棋譜を大量に読み込む都合、この部分をそこそこ高速化しておきたい。
    static Move16 to_move16(const std::string& str);
#endif

	// "go"の後続文字列を解析して、それを反映させたSearch::LimitsTypeを返す。
    Search::LimitsType parse_limits(std::istream& is);

    // エンジンオプション設定を取得する
    auto& engine_options() { return engine.get_options(); }

#if !STOCKFISH
    // --------------------
    // 🌈 やねうら王独自 🌈
    // --------------------

	// cpからValueへ。to_cpの逆変換。
	static Value cp_to_value(int v);

	// スコアを歩の価値を100として正規化して出力する。
	// MATEではないスコアなら"cp x"のように出力する。
	// MATEのスコアなら、"mate x"のように出力する。
	// 
	static std::string value(Value v);

    // USIの指し手文字列などに使われている盤上の升を表す文字列をSquare型に変換する
    // 変換できなかった場合はSQ_NBが返る。高速化のために用意した。
    static Square usi_to_sq(char f, char r);

    // USIプロトコルのマス目文字列をSquare型に変換する。
    // 変換できない文字である場合、SQ_NBを返す。
    static Square to_square(const std::string& str);

    // Move16をUSIプロトコルで使う文字列に変換する。
    static std::string move(Move16 m /*, bool chess960*/);

    // vector<Move>をUSIプロトコルで使う文字列に変換する。
    static std::string move(const std::vector<Move>& moves);

    // コマンドラインに書かれているUSIコマンドをコマンドキューに積む。
    // マルチインスタンス時に注入先を選べるよう、USIEngine::loop()からは呼び出さない。
    void enqueue_command_line_commands(const CommandLine& cli);

    // startup.txtなどのファイルに書かれているUSIコマンドをコマンドキューに積む。
    void enqueue_startup_file_commands(const std::string& filename);

    // 単一エンジン起動用に、コマンドラインとstartup.txtのUSIコマンドをコマンドキューに積む。
    void enqueue_startup_commands(const CommandLine& cli);

	// このclassのUnitTest。
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);

#if defined(__EMSCRIPTEN__)
	// yaneuraou.wasm : JS側から渡されたUSIコマンドを1つ実行する。
	// 戻り値はそのままwasm_pre.jsに返る。
	//   0 : 受理した。
	//   1 : まだ実行できない。あとで再送してもらう。
	int exec_command_from_js(const std::string& cmd);
#endif
#endif

private:

	// 内包している思考エンジン
#if STOCKFISH
	// main関数にコマンドラインから渡された引数
	Engine engine;

	CommandLine cli;
	// 🌈 やねうら王では、CommandLine::gを用いるから、このclassが保持する必要がない。
#else
	// 🌈 やねうら王ではengineを切り替えられるようにIEngineをくるんだ
	//     EngineWrapperというclassを用いる。
	EngineWrapper engine;

    // 先行入力されたUSIコマンド。尽きれば標準入力から入力する。
    std::queue<std::string> pending_commands;

    // キューを優先して1行取得する。標準入力がEOFならquitを返す。
    std::string read_command();

#endif

    std::string currentCmd;

	// string_viewを"\n"で複数行に分割して、それを"info string .."の形で出力する。
	static void print_info_string(std::string_view str);

	// --------------------
	// USI command handlers
	// --------------------

	// USIプロトコルのコマンドに対応するhandler
	// USIプロトコルのコマンド名がそのまま関数名になっている。

	void          go(std::istringstream& is);
	void          bench(std::istream& args);
	void          benchmark(std::istream& args);
	void          position(std::istringstream& is);
	void          setoption(std::istringstream& is);
	u64 perft(const Search::LimitsType&);

#if !STOCKFISH
	// 🌈 やねうら王独自拡張 🌈

	void isready();
    void moves();
    void getoption(std::istringstream& is);
    void qsearch_psv(std::istringstream& is);
    void unittest(std::istringstream& is);
#endif

	// 読み筋を出力するevent handler
	// 📝 Engine class(およびその派生class)から、読み筋を出力したいタイミングで
	//     updateContext経由で呼び出される。
	// 🌈 on_update_info_string()はやねうら王独自拡張。

	static void on_update_no_moves(const Engine::InfoShort& info);
#if STOCKFISH
	static void on_update_full(const Engine::InfoFull& info, bool showWDL);
#else
    static void on_update_full(const Engine::InfoFull& info /*, bool showWDL*/);
#endif

	static void on_iter(const Engine::InfoIter& info);
	static void on_bestmove(std::string_view bestmove, std::string_view ponder);
#if !STOCKFISH
	static void on_update_string(std::string_view info);
#endif

    // すべての読み筋出力listenerを初期化する。
	// 📝 set_engine()のタイミングでEngine側のset_on_XXXを呼び出して
	//     上記のhandlerを登録してやる。
	//     engine側は、読み筋の出力を抑制したい時やカスタマイズしたい時に
	//     このlistenerを変更して対応する。
    void init_search_update_listeners();

    // 致命的なエラーが出たときにその内容を出力してプログラムを終了する。
    [[noreturn]] void terminate_on_critical_error(const std::string& message);

#if !STOCKFISH

	// 🌈 やねうら王独自拡張 🌈

    // USIコマンドを1行実行する。
    // "quit"が来たら、trueを返す。
    bool usi_cmdexec(const std::string& cmd);

	// ファイルからUSIコマンドをコマンドキューに積む。
	void enqueue_command_from_file(std::istringstream& is);

	// option名とvalueを指定して、そのoption名があるなら、そのoptionの値を変更する。
	void set_option_if_exists(const std::string& option_name, const std::string& option_value);

	/*
		📓 Stochastic Ponderの実装について

		Stochastic Ponderはdlshogiで最初に実装された機能。
		やねうら王では、以下のように実装している。このため、Stockfishと差異が生じる。

		1. GUIから送られてきた"position"コマンドのコマンドラインを丸ごと保存しておく。
		2. GUIから送られてきた"go ponder"コマンドのコマンドラインを丸ごと保存しておく。
		3. GUIから"go ponder"が送られてきたときに、1.のmoves節にある最後の指し手を除き、その局面から探索する。
		   指し手履歴がない場合は、元の局面で通常のponderを行う。
		4. GUIから"ponderhit"が送られてきたときに、2.で保存しておいたコマンド列から"ponder"の文字列を取り除き(無視して)実行する。
	*/

	// 上記1.
	std::string last_position_cmd_string = "position startpos";

	// 上記2.
	std::string last_go_cmd_string;

    // 今回のponderで実際に1手戻した場合だけ、ponderhit時に元の局面へ戻す。
    bool stochastic_ponder_rewound = false;
#endif
};

#if !STOCKFISH
#if defined(__EMSCRIPTEN__)
// --------------------
// 🌈 yaneuraou.wasm 用のヘルパー 🌈
// --------------------

/*
	📝 wasmではUSIEngine::loop()が即座にreturnするので、各エンジンのentry pointである
	    engine_main()もすぐに抜ける。engine_main()はEngineとUSIEngineを
	    std::unique_ptrのlocal変数として持っているため、そのまま抜けると両者が解体され、
	    以後JS側からusi_command()を呼んでも実体が無い。
	    (解体時にThread::~Thread()がjoinするため、メインスレッドが停止する危険もある。)

	    そこでengine_main()の末尾でこの関数にmoveし、プロセス終了までインスタンスを保持する。
*/
void wasm_keep_alive(std::unique_ptr<IEngine> engine, std::unique_ptr<USIEngine> usi);

// wasm_keep_alive()で保持しているUSIEngineを返す。未登録ならnullptr。
USIEngine* wasm_usi_instance();
#endif
#endif

} // namespace YaneuraOu

#endif // #ifndef USI_H_INCLUDED
