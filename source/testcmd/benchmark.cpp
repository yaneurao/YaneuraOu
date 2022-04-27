#include "../types.h"

#include <sstream>
#include "../tt.h"
#include "../search.h"
#include "../thread.h"
#include "../usi.h"

#if defined(YANEURAOU_ENGINE_DEEP)
// dlshogiではnodeのカウントの仕方が異なるので、nodes_searched()を別途用意する。
#include "../engine/dlshogi-engine/dlshogi_min.h"
#endif

using namespace std;

// ----------------------------------
//  USI拡張コマンド "bench"(ベンチマーク)
// ----------------------------------

// benchmark用デフォルトの局面集
const vector<string> BenchSfen =
{
	// 初期局面に近い曲面。
	"sfen lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",

	// 読めば読むほど後手悪いような局面
	"sfen l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",

	// 57同銀は詰み、みたいな。
	// 読めば読むほど先手が悪いことがわかってくる局面。
	"sfen 6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",

	// 指し手生成祭りの局面
	// cf. http://d.hatena.ne.jp/ak11/20110508/p1
	"sfen l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
};

void bench_cmd(Position& current, istringstream& is)
{
	// Optionsを書き換えるのであとで復元する。
	auto oldOptions = Options;

	std::string token;
	Search::LimitsType limits;
	vector<std::string> fens;

	// hashはデフォルト1024にしておかないと置換表あふれるな。
//	std::string ttSize = "1024", threads  ="1", limit ="17" , fenFile ="default", limitType = "depth";
	// →　固定depthにすると探索部の改良に左右されすぎる。固定timeの方がいいと思う。1局面15秒設定。
	std::string ttSize = "1024", threads  ="1", limit ="15" , fenFile ="default", limitType = "time";

	string* positional_args[] = { &ttSize, &threads, &limit, &fenFile, &limitType };

	// "benchmark hash 1024 threads 4 limit 3000 type nodes file sfen.txt"のようにも書きたい。

	// 解析中の引数の位置
	int p = 0;
	while (is >> token)
	{
		if (token == "hash")
			is >> ttSize;
		else if (token == "threads")
			is >> threads;
		else if (token == "limit")
			is >> limit;
		else if (token == "file")
			is >> fenFile;
		else if (token == "type")
			is >> limitType;
		else
		{
			// 解釈できなかったものは、位置固定の引数と解釈する
			if (p < /*positional_args.size()*/ 5)
				*positional_args[p++] = token;
		}
	}

	if (ttSize == "d")
	{
		// デバッグ用の設定(毎回入力するのが面倒なので)
		ttSize = "1024";
		threads = "1";
		fenFile = "default";
		limitType = "depth";
		limit = "6";
	}

	// "Threads"があるとは仮定できない
	if (Options.count("Threads"))
		Options["Threads"] = threads;

	// これふかうら王だわ
	if (Options.count("UCT_Threads1"))
		Options["UCT_Threads1"] = threads;

	if (limitType == "time")
		limits.movetime = (TimePoint)1000 * stoi(limit); // movetime is in ms

	else if (limitType == "nodes")
		limits.nodes = stoll(limit);

	else if (limitType == "mate")
		limits.mate = stoi(limit);

	else
		// depth limit
		limits.depth = stoi(limit);

	if (Options.count("USI_Hash"))
		Options["USI_Hash"] = ttSize;

	// 定跡にhitされるとベンチマークにならない。
	if (Options.count("BookFile"))
		Options["BookFile"] = string("no_book");

	// ベンチマークモードにしておかないとPVの出力のときに置換表を漁られて探索に影響がある。
	limits.bench = true;

	// すべての合法手を生成するのか
	limits.generate_all_legal_moves = Options["GenerateAllLegalMoves"];

	// Optionsの影響を受けると嫌なので、その他の条件を固定しておく。
	limits.enteringKingRule = EKR_NONE;

	// テスト用の局面
	// "default"=デフォルトの局面、"current"=現在の局面、それ以外 = ファイル名とみなしてそのsfenファイルを読み込む
	if (fenFile == "default")
		fens = BenchSfen;
	else if (fenFile == "current")
		fens.push_back(current.sfen());
	else
		SystemIO::ReadAllLines(fenFile, fens);

	// 評価関数の読み込み等
	is_ready();

//	TT.clear();
	// → is_ready()のなかでsearch::clear()が呼び出されて、そのなかでTT.clear()しているのでこの初期化は不要。

	// トータルの探索したノード数
#if !defined(YANEURAOU_ENGINE_DEEP)
	int64_t nodes_searched = 0;
	// main threadが探索したノード数
	int64_t nodes_searched_main = 0;
#else
	int64_t nodes_visited = 0;
	//int64_t nodes_visited_main = 0;
	// →　ふかうら王、main threadだけ分けて集計していない。
#endif

	// ベンチの計測用タイマー
	Timer time;
	time.reset();

	// bench条件を出力。
	sync_cout << "Benchmark" << endl
			  << "    hash    : " << ttSize << endl
			  << "    threads : " << threads << endl
			  << "    limit   : " << limitType << " " << limit << endl
			  << "    sfen    : " << fenFile << sync_endl;

	Position pos;
	for (size_t i = 0; i < fens.size(); ++i)
	{
		// SetupStatesは破壊したくないのでローカルに確保
		StateListPtr states(new StateList(1));

		// sfen文字列、Positionコマンドのparserで解釈させる。
		istringstream is(fens[i]);
		position_cmd(pos, is, states);

		sync_cout << "\nPosition: " << (i + 1) << '/' << fens.size() << sync_endl;

		// 探索時にnpsが表示されるが、それはこのglobalなTimerに基づくので探索ごとにリセットを行なうようにする。
		Time.reset();

		Threads.start_thinking(pos, states , limits);

		Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。

#if !defined(YANEURAOU_ENGINE_DEEP)
		nodes_searched      += Threads.nodes_searched();
		nodes_searched_main += Threads.main()->nodes.load(std::memory_order_relaxed);
#else
		// ふかうら王の時は、訪問ノード数を集計する。
		nodes_visited      += dlshogi::nodes_visited();
#endif
	}

	auto elapsed = time.elapsed() + 1; // 0除算の回避のため

	sync_cout << "\n==========================="
		<< "\nTotal time (ms) : " << elapsed
#if !defined(YANEURAOU_ENGINE_DEEP)
		<< "\nNodes searched  : " << nodes_searched
		<< "\nNodes_searched/second    : " << 1000 * nodes_searched / elapsed
#else
		<< "\nNodes visited  : " << nodes_visited
		<< "\nNodes_visited/second     : " << 1000 * nodes_visited  / elapsed
#endif
		;

#if !defined(YANEURAOU_ENGINE_DEEP)
	if (stoi(threads) > 1)
		cout
		<< "\nNodes searched       (main thread) : " << nodes_searched_main
		<< "\nNodes searched/second(main thread) : " << 1000 * nodes_searched_main / elapsed;
#endif

	// 終了したことを出力しないと他のスクリプトから呼び出した時に終了判定にこまる。
	cout << "\n==========================="
		 << "\nThe bench command has completed." << sync_endl;

	// Optionsを書き換えたので復元。
	// 値を代入しないとハンドラが起動しないのでこうやって復元する。
	for (auto& s : oldOptions)
		Options[s.first] = std::string(s.second);
}


