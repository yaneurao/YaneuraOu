//#include <iostream>
//#include "bitboard.h"
//#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "position.h"
#include "usi.h"
#include "misc.h"

using namespace YaneuraOu;

namespace
{
	// ファイルの中身を出力する。
	void print_file(const std::string& path)
	{
		SystemIO::TextReader reader;
		if (reader.Open(path).is_not_ok())
			return;

		std::string line;
		while (reader.ReadLine(line).is_ok())
			sync_cout << line << sync_endl;
	}
}

// ----------------------------------------
//  main()
// ----------------------------------------

// main関数は、namespaceに入れてはならない。(それをするとWindowsアプリ扱いされてしまう)
int main(int argc, char* argv[])
{
	// CommandLineにはglobal objectがあるので、これに設定しておく。
	CommandLine::g.set_arg(argc, argv);

	#if 0
	// 起動直後にソフト名と作者の出力。
    //std::cout << engine_info() << std::endl;
	#endif
	// 📌 やねうら王ではMultiEngineを採用しており、
	//     このタイミングではエンジン名が確定しないから出力できない。

	// 起動時に説明書きを出力。(やねうら王独自拡張)
	print_file("startup_info.txt");

	// -- 全体的な初期化

	Bitboards::init();
	Position::init();

	// 自作Engineのentry point(これはEngineFuncRegisterを用いて登録されている。)
	YaneuraOu::run_engine_entry();

	return 0;
}
