#include "../../config.h"

#if defined(YANEURAOU_MATE_ENGINE)

#include <sstream>

#include "../../usi.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../mate/mate.h"

using namespace std;
using namespace Mate::Dfpn;

namespace
{
	// Solver本体
	MateDfpnSolver solver(DfpnSolverType::None);

	std::vector<std::string> solver_types = { "32bitNodeSolver" , "64bitNodeSolver" };
}

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	//  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
	//  0なら出力なし。
	o["PvInterval"] << Option(1000, 0, 100000);

	o["SolverType"] << Option(solver_types, solver_types[0]);

	// 探索ノード制限。0なら無制限。
	o["NodesLimit"] << Option(0, 0, INT64_MAX);
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
	// Sovler種別
	auto solver_type = (string)Options["SolverType"];
	if (solver_type == solver_types[0])
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node32bit);
	else if (solver_type == solver_types[1])
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node64bit);
	else
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::None);

	u64 mem = Options["USI_Hash"];
	sync_cout << "info string DfPn memory allocation , USI_Hash = " << mem << " [MB]" << sync_endl;
	solver.alloc(mem);
}

// 探索開始時に呼び出される。
void MainThread::search()
{
	// 思考エンジンからの返し値
	// 詰将棋ルーチンからMOVE_RESIGNが返ってくることはないので、この値が変化していたら返し値があったことを意味する。
	atomic<Move> move = MOVE_RESIGN;

	// 探索ノード数制限
	u64 nodes_limit = Options["NodesLimit"];

	// 詰将棋の探索用スレッド
	auto thread = std::thread([&]()
		{
			move = solver.mate_dfpn(rootPos, nodes_limit);
		});

	Timer time;
	time.reset(); // 探索開始からの経過時間を記録しておく。
	TimePoint lastPvOutput = 0; // 前回のPV出力時刻
	TimePoint pvInterval = Options["PvInterval"]; // PV出力間隔

	// 読み筋の出力するヘルパ
	auto print_pv = [&]() {
		auto elapsed = time.elapsed();
		u64 nodes_searched = solver.get_nodes_searched();

		// nps算出
		u64 nps = nodes_searched * 1000 / elapsed;

		sync_cout << "info time " << elapsed << " nodes " << nodes_searched << " nps " << nps
			      << " hashfull " << solver.hashfull() << " pv" << USI::move(solver.get_current_pv()) << sync_endl;
	};

	// 時間切れ判定するヘルパ
	auto time_up = [&]() { return Search::Limits.mate && time.elapsed() >= Search::Limits.mate; };

	// 探索の終了を待つ
	while (!Threads.stop && !time_up() && move == MOVE_RESIGN)
	{
		Tools::sleep(100);

		auto elapsed = time.elapsed();
		if (pvInterval && elapsed > lastPvOutput + pvInterval)
		{
			print_pv();
			lastPvOutput = time.elapsed();
		}
	}

	thread.join();

	// 最後に必ず1回PVを出力する。
	print_pv();

	if (time_up())
	{
		sync_cout << "checkmate timeout" << sync_endl;
	}
	else if (move == MOVE_NONE)
	{
		if (solver.is_out_of_memory())
			sync_cout << "info string Out Of Memory." << sync_endl;
		else if (solver.get_nodes_searched() >= nodes_limit)
			sync_cout << "info string Exceeded NodesLimit." << sync_endl;

		sync_cout << "checkmate none" << sync_endl; // 不明
	}
	else if (move == MOVE_NULL)
	{
		// 不詰が証明された
		sync_cout << "checkmate nomate" << sync_endl;
	}
	else {
		auto pv = solver.get_pv();
		sync_cout << "checkmate" << USI::move(pv) << sync_endl;
	}
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search()
{
}

#endif // YANEURAOU_MATE_ENGINE
