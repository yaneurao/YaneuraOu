#include "benchmark.h"
#include "numa.h"

#include <fstream>
#include <iostream>

// ----------------------------------
//  USI拡張コマンド "bench"(ベンチマーク)
// ----------------------------------
namespace {

// benchmark用デフォルトの局面集
const std::vector<std::string> Defaults =
{
	// 初期局面に近い曲面。
	"lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",

	// 読めば読むほど後手悪いような局面
	"l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",

	// 57同銀は詰み、みたいな。
	// 読めば読むほど先手が悪いことがわかってくる局面。
	"6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",

	// 指し手生成祭りの局面
	// cf. http://d.hatena.ne.jp/ak11/20110508/p1
	"l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
};

// speedtestコマンドでテストする用。
// やねうら王ではDefaultsと同じ局面にしておく。
const std::vector<std::vector<std::string>> BenchmarkPositions = { {Defaults} };

} // namespace

namespace YaneuraOu::Benchmark {

// Builds a list of UCI commands to be run by bench. There
// are five parameters: TT size in MB, number of search threads that
// should be used, the limit value spent for each position, a file name
// where to look for positions in FEN format, and the type of the limit:
// depth, perft, nodes and movetime (in milliseconds). Examples:
//
// bench                            : search default positions up to depth 13
// bench 64 1 15                    : search default positions up to depth 15 (TT = 64MB)
// bench 64 1 100000 default nodes  : search default positions for 100K nodes each
// bench 64 4 5000 current movetime : search current position with 4 threads for 5 sec
// bench 16 1 5 blah perft          : run a perft 5 on positions in file "blah"

// benchで実行するUCIコマンドのリストを構築します。
// 引数は以下の5つです：TTサイズ（MB単位）、使用するサーチスレッド数、
// 各局面に費やす制限値、FEN形式の局面を読み込むファイル名、
// そして制限の種類（depth、perft、nodes、movetime（ミリ秒））。
// 使用例：
// bench                            : デフォルトの局面を深さ13まで探索
// bench 64 1 15                    : デフォルトの局面を深さ15まで探索（TT=64MB）
// bench 64 1 100000 default nodes  : デフォルトの局面を各局面100,000ノードで探索
// bench 64 4 5000 current movetime : 現在の局面を4スレッドで5秒間探索
// bench 16 1 5 blah perft          : ファイル"blah"内の局面に対してperft 5を実行

/*
	📓 やねうら王では、制限の種類をデフォルトで、depthからmovetimeに変更しているので、
		上の例のようにdepthで深さ15まで探索するには

			bench 64 1 15 default depth

		のように指定する必要がある。
*/

std::vector<std::string> setup_bench(const std::string& currentFen, std::istream& is) {
	std::vector<std::string> fens, list;
	std::string              go, token;

#if STOCKFISH
	// Assign default values to missing arguments
	std::string ttSize    = (is >> token) ? token : "16";
	std::string threads   = (is >> token) ? token : "1";
	std::string limit     = (is >> token) ? token : "13";
	std::string fenFile   = (is >> token) ? token : "default";
	std::string limitType = (is >> token) ? token : "depth";
#else
	// 🌈 やねうら王では、デフォルトで1分間(時間固定)のbenchにしておく。

	std::string ttSize    = (is >> token) ? token : "1024";
    std::string threads   = (is >> token) ? token : "1";
    std::string limit     = (is >> token) ? token : "15000";
    std::string fenFile   = (is >> token) ? token : "default";
    std::string limitType = (is >> token) ? token : "movetime";
#endif

	go = limitType == "eval" ? "eval" : "go " + limitType + " " + limit;

	if (fenFile == "default")
		fens = Defaults;

	else if (fenFile == "current")
		fens.push_back(currentFen);

	else
	{
		std::string   fen;
		std::ifstream file(fenFile);

		if (!file.is_open())
		{
			std::cerr << "Unable to open file " << fenFile << std::endl;
			exit(EXIT_FAILURE);
		}

		while (getline(file, fen))
			if (!fen.empty())
				fens.push_back(fen);

		file.close();
	}

	list.emplace_back("setoption name Threads value " + threads);
#if STOCKFISH
	list.emplace_back("setoption name Hash value " + ttSize);
#else
    list.emplace_back("setoption name USI_Hash value " + ttSize);
#endif
	// 🤔 どうせ内部的にしか使わない符号みたいなものなので"usinewgame"に変更しないことにする。
    list.emplace_back("ucinewgame");

	for (const std::string& fen : fens)
		if (fen.find("setoption") != std::string::npos)
			list.emplace_back(fen);
		else
		{
#if STOCKFISH
			list.emplace_back("position fen " + fen);
#else
            list.emplace_back("position sfen " + fen);
#endif
			list.emplace_back(go);
		}

	return list;
}

BenchmarkSetup setup_benchmark(std::istream& is) {

	// TT_SIZE_PER_THREAD is chosen such that roughly half of the hash is used all positions
	// for the current sequence have been searched.

	// TT_SIZE_PER_THREAD は、現在のシーケンスのすべての局面を探索した際に
	// ハッシュの約半分が使用されるように選定されています。

	static constexpr int TT_SIZE_PER_THREAD = 128;

	static constexpr int DEFAULT_DURATION_S = 150;

	BenchmarkSetup setup{};

	// Assign default values to missing arguments
	int desiredTimeS;

	if (!(is >> setup.threads))
		setup.threads = get_hardware_concurrency();
	else
		setup.originalInvocation += std::to_string(setup.threads);

	if (!(is >> setup.ttSize))
		setup.ttSize = TT_SIZE_PER_THREAD * setup.threads;
	else
		setup.originalInvocation += " " + std::to_string(setup.ttSize);

	if (!(is >> desiredTimeS))
		desiredTimeS = DEFAULT_DURATION_S;
	else
		setup.originalInvocation += " " + std::to_string(desiredTimeS);

	setup.filledInvocation += std::to_string(setup.threads) + " " + std::to_string(setup.ttSize)
		+ " " + std::to_string(desiredTimeS);

	auto getCorrectedTime = [&](int ply) {
		// time per move is fit roughly based on LTC games
		// seconds = 50/{ply+15}
		// ms = 50000/{ply+15}
		// with this fit 10th move gets 2000ms
		// adjust for desired 10th move time
		return 50000.0 / (static_cast<double>(ply) + 15.0);
		};

	float totalTime = 0;
	for (const auto& game : BenchmarkPositions)
	{
		setup.commands.emplace_back("ucinewgame");
		int ply = 1;
		for (int i = 0; i < static_cast<int>(game.size()); ++i)
		{
			const float correctedTime = float(getCorrectedTime(ply));
			totalTime += correctedTime;
			ply += 1;
		}
	}

	float timeScaleFactor = static_cast<float>(desiredTimeS * 1000) / totalTime;

	for (const auto& game : BenchmarkPositions)
	{
		setup.commands.emplace_back("ucinewgame");
		int ply = 1;
		for (const std::string& fen : game)
		{
#if STOCKFISH
			setup.commands.emplace_back("position fen " + fen);
#else
            setup.commands.emplace_back("position sfen " + fen);
#endif
			const int correctedTime = static_cast<int>(getCorrectedTime(ply) * timeScaleFactor);
			setup.commands.emplace_back("go movetime " + std::to_string(correctedTime));

			ply += 1;
		}
	}

	return setup;
}

} // namespace YaneuraOu
