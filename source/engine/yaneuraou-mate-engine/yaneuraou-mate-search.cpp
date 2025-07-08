#include "../../config.h"

#if defined(YANEURAOU_MATE_ENGINE)

#include <sstream>

#include "../../usi.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../mate/mate.h"

using namespace std;
namespace YaneuraOu {

using namespace Mate::Dfpn;

namespace
{
	// Solver本体
	MateDfpnSolver solver(DfpnSolverType::None);
}

// エンジンに追加オプションを設定したいときは、この関数を定義すること。
// Engineのコンストラクタからコールバックされる。
void Engine::extra_option()
{
	//  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
	//  0なら出力なし。
	options.add("PvInterval", Option(1000, 0, 100000));

	options.add("SolverType", Option("32bitNodeSolver", "32bitNodeSolver 64bitNodeSolver"));

	// 探索ノード制限。0なら無制限。
	options.add("NodesLimit", Option(0, 0, INT64_MAX));
}

// "isready"のタイミングのcallback。時間のかかる初期化処理はここで行うこと。
void  Engine::isready()
{
	// Sovler種別
	auto solver_type = (string)options["SolverType"];
	if (solver_type == "32bitNodeSolver")
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node32bit);
	else if (solver_type == "64bitNodeSolver")
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node64bit);
	else
		solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::None);

	u64 mem = options["USI_Hash"];
	sync_cout << "info string DfPn memory allocation , USI_Hash = " << mem << " [MB]" << sync_endl;
	solver.alloc(mem);
}

namespace Search {

// このworker(探索用の1つのスレッド)の初期化
// 📝 これは、"usinewgame"のタイミングで、すべての探索スレッド(エンジンオプションの"Threads"で決まる)に対して呼び出される。
void Worker::clear() {}

void Worker::start_searching()
{
	// 思考エンジンからの返し値
	// 詰将棋ルーチンからMove::resign()が返ってくることはないので、この値が変化していたら返し値があったことを意味する。
	atomic<Move> move = Move::resign();

	// 探索ノード数制限
	u64 nodes_limit = options["NodesLimit"];

	// 詰将棋の探索用スレッド
	auto thread = std::thread([&]()
		{
			move = solver.mate_dfpn(rootPos, nodes_limit);
		});

	Timer time;
	time.reset(); // 探索開始からの経過時間を記録しておく。
	TimePoint lastPvOutput = 0; // 前回のPV出力時刻
	TimePoint pvInterval = options["PvInterval"]; // PV出力間隔

	// 読み筋の出力するヘルパ
	auto print_pv = [&]() {
		auto elapsed = time.elapsed();
		u64 nodes_searched = solver.get_nodes_searched();

		// nps算出
		u64 nps = nodes_searched * 1000 / elapsed;

		sync_cout << "info time " << elapsed << " nodes " << nodes_searched << " nps " << nps
			      << " hashfull " << solver.hashfull() << " pv" << USIEngine::move(solver.get_current_pv()) << sync_endl;
	};

	

	// 時間切れ判定するヘルパ
	// 将棋ではgo mateのあとmateに使う秒数が入ってきている。
	auto time_up = [&]() { return limits.mate && time.elapsed() >= limits.mate; };

	// 探索の終了を待つ
	while (!threads.stop && !time_up() && move.load() == Move::resign())
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
	else if (move.load() == Move::none())
	{
		if (solver.is_out_of_memory())
			sync_cout << "info string Out Of Memory." << sync_endl;
		else if (solver.get_nodes_searched() >= nodes_limit)
			sync_cout << "info string Exceeded NodesLimit." << sync_endl;

		sync_cout << "checkmate none" << sync_endl; // 不明
	}
	else if (move.load() == Move::null())
	{
		// 不詰が証明された
		sync_cout << "checkmate nomate" << sync_endl;
	}
	else {
		auto pv = solver.get_pv();
		sync_cout << "checkmate" << USIEngine::move(pv) << sync_endl;
	}
}

// 探索中に、main threadから一定間隔ごとに呼び出して
// ここで残り時間のチェックを行う。(ことになっている)
void SearchManager::check_time(Search::Worker& worker)
{
	sync_cout << "SearchManager::check_time" << sync_endl;
}

} // namespace Search
} // namespace YaneuraOu

#endif // YANEURAOU_MATE_ENGINE
