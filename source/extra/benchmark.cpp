#include "../shogi.h"

#include <sstream>
#include "../tt.h"
#include "../search.h"
#include "../thread.h"

using namespace std;

// ----------------------------------
//  USI拡張コマンド "bench"(ベンチマーク)
// ----------------------------------

// benchmark用デフォルトの局面集
// これを増やすなら、下のほうの fens.assign のところの局面数も増やすこと。
static const char* BenchSfen[] = {

	// 初期局面に近い曲面。
	//"lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",

	// 読めば読むほど後手悪いような局面
	"l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",

	// 57同銀は詰み、みたいな。
	// 読めば読むほど先手が悪いことがわかってくる局面。
	"6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",

	// 指し手生成祭りの局面
	// cf. http://d.hatena.ne.jp/ak11/20110508/p1
	"l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
};

void bench_cmd(Position& current, istringstream& is)
{
	// Optionsを書き換えるのであとで復元する。
	auto oldOptions = Options;

	string token;
	Search::LimitsType limits;
	vector<string> fens;

	// →　デフォルト1024にしておかないと置換表あふれるな。
	std::string ttSize = (is >> token) ? token : "1024";

	string threads = (is >> token) ? token : "1";
	string limit = (is >> token) ? token : "17";

	string fenFile = (is >> token) ? token : "default";
	string limitType = (is >> token) ? token : "depth";

	if (ttSize == "d")
	{
		// デバッグ用の設定(毎回入力するのが面倒なので)
		ttSize = "1024";
		threads = "1";
		fenFile = "default";
		limitType = "depth";
		limit = "6";
	}

	if (limitType == "time")
		limits.movetime = 1000 * stoi(limit); // movetime is in ms

	else if (limitType == "nodes")
		limits.nodes = stoll(limit);

	else if (limitType == "mate")
		limits.mate = stoi(limit);

	else
		limits.depth = stoi(limit);

	Options["Hash"] = ttSize;
	Options["Threads"] = threads;

#if defined(YANEURAOU_2018_OTAFUKU_ENGINE) || defined(YANEURAOU_2018_GOKU_ENGINE)
	// 定跡にhitされるとベンチマークにならない。
	Options["BookFile"] = "no_book";
#endif

	// ベンチマークモードにしておかないとPVの出力のときに置換表を漁られて探索に影響がある。
	limits.bench = true;

	TT.clear();

	// Optionsの影響を受けると嫌なので、その他の条件を固定しておく。
	limits.enteringKingRule = EKR_NONE;

	// テスト用の局面
	// "default"=デフォルトの局面、"current"=現在の局面、それ以外 = ファイル名とみなしてそのsfenファイルを読み込む
	if (fenFile == "default")
		fens.assign(BenchSfen, BenchSfen + 3);
	else if (fenFile == "current")
		fens.push_back(current.sfen());
	else
		read_all_lines(fenFile, fens);

	// 評価関数の読み込み等
	is_ready();

	// トータルの探索したノード数
	int64_t nodes = 0;

	// main threadが探索したノード数
	int64_t nodes_main = 0;

	// ベンチの計測用タイマー
	Timer time;
	time.reset();

	Position pos;
	for (size_t i = 0; i < fens.size(); ++i)
	{
		// SetupStatesは破壊したくないのでローカルに確保
		StateListPtr states(new StateList(1));

		pos.set(fens[i] ,&states->back() , Threads.main());

		sync_cout << "\nPosition: " << (i + 1) << '/' << fens.size() << sync_endl;

		// 探索時にnpsが表示されるが、それはこのglobalなTimerに基づくので探索ごとにリセットを行なうようにする。
		Time.reset();

		Threads.start_thinking(pos, states , limits);
		Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。

		nodes += Threads.nodes_searched();
		nodes_main += Threads.main()->nodes.load(std::memory_order_relaxed);
	}

	auto elapsed = time.elapsed() + 1; // 0除算の回避のため

	sync_cout << "\n==========================="
		<< "\nTotal time (ms) : " << elapsed
		<< "\nNodes searched  : " << nodes
		<< "\nNodes/second    : " << 1000 * nodes / elapsed;

	if ((int)Options["Threads"] > 1)
		cout
		<< "\nNodes searched(main thread) : " << nodes_main
		<< "\nNodes/second  (main thread) : " << 1000 * nodes_main / elapsed;

	cout << sync_endl;

	// Optionsを書き換えたので復元。
	// 値を代入しないとハンドラが起動しないのでこうやって復元する。
	for (auto& s : oldOptions)
		Options[s.first] = std::string(s.second);
}


