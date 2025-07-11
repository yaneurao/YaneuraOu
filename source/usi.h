﻿#ifndef USI_H_INCLUDED
#define USI_H_INCLUDED

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>

#include "types.h"
#include "engine.h"
#include "position.h"
#include "testcmd/unit_test.h"

namespace YaneuraOu {

// --------------------
//     USI関連
// --------------------

// USIEngine本体。
// このclassがEngineを内包していて、USIメッセージのハンドラを処理して
// 内包しているEngineに対して司令を送る。
class USIEngine {
public:
	// 📝 やねうら王では、CommandLine::gを見れば良いので、argc,argvは引数として渡さないことにした。
	USIEngine(/*int argc, char** argv */);

	// USIEngine classから使うEngine。
	// 📌 やねうら王独自。エンジンの実装を変更できるように、
	//     IEngine(エンジン interface)を渡し、エンジンを動的に切り替えたり、
	//     複数の異なるエンジンから成るUSIEngineを同時に使うことができるようにする。
	void set_engine(IEngine& _engine);

	// main threadをUSIメッセージの受信のために待機させる。
	// "quit"コマンドが送られてくるまでこのループは抜けない。
	void loop();

	// --------------------
	// USI関係の記法変換部
	// --------------------

#if defined(USE_PIECE_VALUE)

	// 詰みやそれに類似した特別なスコアの処理なしに、Valueを整数のセントポーン数に変換する。

	//static int         to_cp(Value v, const Position& pos);
	static int   to_cp(Value v);

	// cpからValueへ。⇑の逆変換。
	// 📌　やねうら王独自
	static Value cp_to_value(int v);

	// スコアを歩の価値を100として正規化して出力する。
	//   MATEではないスコアなら"cp x"のように出力する。
	//   MATEのスコアなら、"mate x"のように出力する。
	// ⚠ USE_PIECE_VALUEが定義されていない時は正規化しようがないのでこの関数は呼び出せない。
	// 📌　やねうら王独自
	static std::string value(Value v);

#endif

	//static std::string format_score(const Score& s);
	// 📌 やねうら王では使わない

	// USIプロトコルで使うマス目文字列に変換する。
	static std::string square(Square s);

	// USIプロトコルで使う指し手文字列に変換する。
	static std::string move(Move m  /*, bool chess960*/);

	// 勝率文字列に変換する。
	// 📌 将棋では評価値をcpで出力するので不要。
	// static std::string wdl(Value v, const Position& pos);

	// string全体を小文字化して返す。
	static std::string to_lower(std::string str);

	// USIの指し手文字列をMove型の変換する。
	// 合法手でなければMove::noneを返すようになっている。
	// 💡 合法でない指し手の場合、エラーである旨を出力する。
	static Move        to_move(const Position& pos, std::string str);

	// USI形式から指し手への変換。本来この関数は要らないのだが、
	// 棋譜を大量に読み込む都合、この部分をそこそこ高速化しておきたい。
	// 📌 やねうら王、独自
	static Move16      to_move16(const std::string& str);

	// コマンドラインを解析して、Search::LimitsTypeに反映させて返す。
	static Search::LimitsType parse_limits(std::istream& is);

	// USIの指し手文字列などに使われている盤上の升を表す文字列をSquare型に変換する
	// 変換できなかった場合はSQ_NBが返る。高速化のために用意した。
	// 📌　やねうら王独自
	static Square      usi_to_sq(char f, char r);

	// USIプロトコルのマス目文字列をSquare型に変換する。
	// 変換できない文字である場合、SQ_NBを返す。
	// 📌　やねうら王独自
	static Square      to_square(const std::string& str);

	// Move16をUSIプロトコルで使う文字列に変換する。
	// 📌　やねうら王独自
	static std::string move(Move16 m/*, bool chess960*/);

	// vector<Move>をUSIプロトコルで使う文字列に変換する。
	// 📌　やねうら王独自
	static std::string move(const std::vector<Move>& moves);

	// USIコマンドを積むことができる標準入力
	// 💡 ここにUSIコマンドを積むとそれが実行される。
	// 📌 やねうら王独自
	StandardInput std_input;

	// --------------------
	//    Properties
	// --------------------

	// エンジンオプション設定を取得する
	OptionsMap& engine_options() { return engine.get_options(); }

	// このclassのUnitTest。
	// 📌 やねうら王独自
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);

private:
	// 内包している思考エンジン
	//Engine	    engine;
	// 📌 やねうら王ではengineを切り替えられるようにIEngineをくるんだEngineWrapperというclassを用いる。
	EngineWrapper engine;

	// main関数にコマンドラインから渡された引数
	//CommandLine cli;
	// 📌 やねうら王では、CommandLine::gを用いるから、このclassが保持する必要がない。

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
	std::uint64_t perft(const Search::LimitsType&);

	// -- やねうら王独自拡張

	void          unittest(std::istringstream& is);
	void          getoption(std::istringstream& is);
	void          isready();
	void          moves();

	// event handlerが成功した時に呼び出されるhandler

	//static void on_update_no_moves(const Engine::InfoShort& info);
	//static void on_update_full(const Engine::InfoFull& info, bool showWDL);
	//static void on_iter(const Engine::InfoIter& info);
	//static void on_bestmove(std::string_view bestmove, std::string_view ponder);

	//void init_search_update_listeners();

	// --- やねうら王独自拡張

	// コマンドラインと"startup.txt"に書かれているUSIコマンドをstd_inputに積む。
	void enqueue_startup_command();

	// ファイルからUSIコマンドをstd_inputに積む。
	void enqueue_command_from_file(std::istringstream& is);

	// option名とvalueを指定して、そのoption名があるなら、そのoptionの値を変更する。
	void set_option_if_exists(const std::string& option_name, const std::string& option_value);

	// USIコマンドを1行実行する。
	// "quit"が来たら、trueを返す。
	bool usi_cmdexec(const std::string& cmd);

};

} // namespace YaneuraOu

#endif // #ifndef USI_H_INCLUDED
