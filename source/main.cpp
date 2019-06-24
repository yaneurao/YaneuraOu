//#include <iostream>
//#include "bitboard.h"
//#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "usi.h"

// ----------------------------------------
//  main()
// ----------------------------------------

int main(int argc, char* argv[])
{
	// --- 全体的な初期化
	USI::init(Options);
	Bitboards::init();
	Position::init();
	Search::init();
	Threads.set(Options["Threads"]);
	//Search::clear();
	Eval::init();

	// USIコマンドの応答部
	USI::loop(argc, argv);

	// 生成して、待機させていたスレッドの停止
	Threads.set(0);

	return 0;
}
