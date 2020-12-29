#include "../config.h"

#if defined(ENABLE_TEST_CMD)

// ----------------------------------
//      詰み関係のtestコマンド
// ----------------------------------

// "test genmate ..."のように"test"コマンドの後続コマンドとして書く。

#include <sstream>

#include "mate.h"

#include "../position.h"
#include "../usi.h"
#include "../thread.h"
#include "../search.h"

#if defined (EVAL_LEARN)
#include "../learn/learn.h"  // Learner::search()が自己対局のために必要。
#endif

using namespace std;

namespace {

	// ----------------------------------
	//      "test genmate" command
	// ----------------------------------

	// N手詰みの局面を生成する。
	void gen_mate(Position& pos, std::istringstream& is)
	{
#if !defined (EVAL_LEARN)
		cout << "Error! genmate command is only for EVAL_LEARN" << endl;
		return;
#else
		// default 5万局面
		uint64_t loop_max = 50000;

		// この手数で詰む詰みの局面を探す
		// min_ply以上、max_ply以下の詰みを探す。
		int min_ply = 3;
		int max_ply = 3;

		string filename = "mate.sfen";

		string token;
		while (is >> token)
		{
			if (token == "loop")
				is >> loop_max;
			else if (token == "min_ply")
				is >> min_ply;
			else if (token == "max_ply")
				is >> max_ply;
			else if (token == "filename")
				is >> filename;
		}

		size_t thread_num = Options["Threads"];

		cout << "gen_mate command" << endl
			<< "  loop_max         = " << loop_max << endl
			<< "  min_ply          = " << min_ply << endl
			<< "  max_ply          = " << max_ply << endl
			<< "  output filename  = " << filename << endl
			<< "  Threads          = " << thread_num << endl
			;

		ofstream fs(filename);

		// 最小手数と最大手数が逆転している
		if (min_ply > max_ply)
		{
			cout << "Error! min_ply > max_ply" << endl;
			return;
		}

		// 偶数手の詰将棋はない
		if (min_ply % 2 == 0 || max_ply % 2 == 0)
		{
			cout << "Error! min_ply , max_ply must be odd." << endl;
			return;
		}

		const int MAX_PLY = 256; // 256手までテスト

		// 並列化しないと時間かかる。
		// Options["Threads"]の数だけ実行する。

		atomic<u64> generated_count = 0;
		std::mutex mutex;

		auto worker = [&](size_t thread_id) {
			WinProcGroup::bindThisThread(thread_id);

			PRNG prng;

			// StateInfoを最大手数分だけ確保
			auto states = std::make_unique<StateInfo[]>(MAX_PLY + 1);

			Position pos;
			Mate::MateSolver solver;

			while (true)
			{
				pos.set_hirate(&(states[0]), Threads[thread_id]);

				for (int ply = 0; ply < MAX_PLY; ++ply)
				{
					MoveList<LEGAL_ALL> mg(pos);
					if (mg.size() == 0)
						break;

					// 探索深さにランダム性を持たせることによって、毎回異なる棋譜になるようにする。
					int search_depth = 3 + (int)prng.rand(3); // depth 3～5
					auto pv = Learner::search(pos, search_depth , 1);

					Move m = pv.second[0];
					if (m == MOVE_NONE)
						break;

					pos.do_move(m, states[ply + 1]);

					// ここまでの手順が絡んで千日手模様の逃れられたりすると嫌なので
					// sfen化してから、それが解けるか試す。
					StateInfo si;
					Position newPos;
					string sfen = pos.sfen();
					newPos.set(sfen, &si, pos.this_thread());

					// mate_min_ply - 2で詰まなくて、
					// mate_max_plyで詰むことを確認すれば良いはず。
					if (solver.mate_odd_ply(newPos, min_ply - 2, true) == MOVE_NONE
						&& solver.mate_odd_ply(newPos, max_ply, true) != MOVE_NONE)
					{
						// 発見した。

						// 局面図を出力してみる。
						//sync_cout << pos << sync_endl;

						// 初手を出力してみる。
						//cout << Mate::mate_odd_ply(pos, max_ply, true) << endl;

						{
							std::lock_guard<std::mutex> lk(mutex);

							//sync_cout << "sfen = " << sfen << sync_endl;
							fs << sfen << endl;
							fs.flush();
						}

						// 生成した数
						if (++generated_count >= loop_max)
							return;

						break; // ここで次の対局へ
					}

					//moves[ply] = m;
				}
			}
		};

		auto threads = make_unique<std::thread[]>(thread_num);
		for (size_t i = 0; i < thread_num; ++i)
			threads[i] = std::thread(worker, i);

		while (generated_count < loop_max)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(5000));
			sync_cout << generated_count << sync_endl;
		}

		for (size_t i = 0; i < thread_num; ++i)
			threads[i].join();

		sync_cout << "..done" << sync_endl;

#endif // defined(EVAL_LEARN)

	}

	// ----------------------------------
	//      "test matebench" command
	// ----------------------------------

	// 詰みルーチンに関するbenchをとる。
	// このbenchには、"test genmate"コマンドで生成した詰み局面を利用すると良い。
	void mate_bench(Position& pos, std::istringstream& is)
	{

#if !(defined (USE_MATE_SOLVER) || defined(USE_MATE_DFPN))
		cout << "Error! define USE_MATE_SOLVER or USE_MATE_DFPN" << endl;
#else

		// 読み込む問題数
		size_t num = 10000;

		// 問題を何回ずつ解かせるのか。
		// sfen文字列からPositionにセットする時間がロスするので、何度かずつ同じを局面を繰り返し解かせる時に用いる。
		size_t loop = 1;

		// 詰み局面のファイル名(genmateコマンドで生成したものを指定すると良い)
		string filename = "mate.sfen";

		// mate_odd_plyで読む手数
		int mate_ply = 3;

		// ノード数制限
		u32 nodes_limit = 10000;

		// dfpn用のメモリ[MB]
		u32 dfpn_mem = 1024;

		// デバッグ用に、解けなかった問題を出力する。
		bool verbose = false;

		int test_mode = 3; // mate_odd_plyとdfpnと両方のテストを行う。

		std::string token;
		while (is >> token)
		{
			if (token == "num")
				is >> num;
			else if (token == "loop")
				is >> loop;
			else if (token == "file")
				is >> filename;
			else if (token == "mate_ply")
				is >> mate_ply;
			else if (token == "nodes")
				is >> nodes_limit;
			else if (token == "verbose")
				verbose = true;
			else if (token == "mode")
				is >> test_mode;

#if defined(USE_MATE_DFPN)
			else if (token == "dfpn_mem")
				is >> dfpn_mem;
#endif
		}

		std::cout << "mate bench :" << std::endl
			<< " num (number of problems) = " << num << endl
			<< " sfen file name           = " << filename << endl
			<< " mate_ply                 = " << mate_ply << endl
			<< " nodes(nodes limit)       = " << nodes_limit << endl
			<< " verbose                  = " << verbose << endl
			<< " mode                     = " << test_mode << endl
#if defined(USE_MATE_DFPN)
			<< " dfpn_mem [MB]            = " << dfpn_mem << endl
#endif
			;

#if defined(USE_MATE_DFPN)
		Mate::Dfpn::MateDfpnSolver dfpn(Mate::Dfpn::DfpnSolverType::Node32bit);
		dfpn.alloc(dfpn_mem);

		Mate::Dfpn::MateDfpnSolver dfpn2(Mate::Dfpn::DfpnSolverType::Node16bitOrdering);
		dfpn2.alloc(dfpn_mem);

		Mate::Dfpn::MateDfpnSolver dfpn3(Mate::Dfpn::DfpnSolverType::Node48bitOrdering);
		dfpn3.alloc(dfpn_mem);
#endif

		ifstream f(filename);
		unique_ptr<vector<string>> problems = make_unique<vector<string>>();
		problems->reserve(num);

		cout << "read problems.." << endl;
		for (size_t i = 0; i < num; ++i)
		{
			string sfen;
			if (!getline(f, sfen) || f.eof())
				break;
			problems->emplace_back(sfen);

			// 読み込み経過を出力
			if (i % 10000 == 0 && i > 0)
				cout << ".";
		}
		auto problem_num = problems->size();
		cout << "..number of problems read = " << problem_num << endl;

		if (problem_num == 0)
			return;

		string sfen2;
		u64 nodes_searched;

		auto bench = [&](function<Move()> solver, string test_name) {
			Timer timer;
			timer.reset();

			cout << "===== " << test_name << " =====" << endl
				<< "test start" << endl;

			// 解けた問題数
			u32 solved = 0;
			nodes_searched = 0;

			TimePoint last_pv = 0;

			size_t size = min(num, problem_num);
			for (size_t i = 0; i < size; ++i)
			{
				auto elapsed = timer.elapsed();
				// pvは一定間隔で出力する。
				if (last_pv + 2000 < elapsed)
				{
					last_pv = elapsed;

					// mateを呼び出した回数。
					size_t times = i * loop;
					cout << " number of times = " << times << " , elapsed = " << elapsed
						<< " , called per second = " << (double)times * 1000 / (elapsed + 0.00000001)
						<< " nps = " << nodes_searched * 1000 / elapsed << endl;
				}

				auto sfen = (*problems)[i];
				sfen2 = sfen;
				StateInfo si;
				pos.set(sfen, &si, Threads.main());
				Move move;
				for (size_t j = 0; j < loop; ++j)
				{
					move = solver();
					if (move)
						solved++;
					else if (verbose && j == 0)
						// 解けなかった時にその問題を出力する。
						cout << "unsolved : line = " << i + 1 << endl <<
						"sfen " << sfen << endl; // このままの形式でどこかに貼り付けたり、やねうら王に局面設定したりできる。

				}
			}

			auto elapsed = timer.elapsed();
			size_t times = size * loop;
			cout << " number of times = " << times << " , elapsed = " << elapsed
				<< " , solved per second = " << (double)times * 1000 / (elapsed + 0.00000001) << endl;
			cout << " solved = " << solved << " , accuracy = " << (100.0 * solved / times) << "%" << endl;

			cout << test_name << " test end" << endl;

		};

#if defined(USE_MATE_SOLVER)
		// 2進数にしてbit0が立ってたら奇数詰めを呼び出す
		if (test_mode & 1)
			bench([&]() { Mate::MateSolver solver; return solver.mate_odd_ply(pos, mate_ply, true); }, "mate_odd_ply");
#endif

#if defined(USE_MATE_DFPN)
		// 2進数にしてbit1が立ってたらdfpnを呼び出す
		// 速度比較用。
		if (test_mode & 2)
			bench([&]() { auto m = dfpn.mate_dfpn(pos, nodes_limit); nodes_searched += dfpn.get_nodes_searched(); return m; }, "mate_dfpn");

		if (test_mode & 4)
			bench([&]() { auto m = dfpn2.mate_dfpn(pos, nodes_limit); nodes_searched += dfpn2.get_nodes_searched(); return m; }, "mate_dfpn2");

		if (test_mode & 8)
			bench([&]() { auto m = dfpn3.mate_dfpn(pos, nodes_limit); nodes_searched += dfpn3.get_nodes_searched(); return m; }, "mate_dfpn3");

#endif

		// 他にも詰将棋ルーチンを追加(or 改良)した時に、ここに追加していく。

#endif // !(defined (USE_MATE_SOLVER) || defined(USE_MATE_DFPN))
	}

	// ----------------------------------
	//      "test mate_dfpn" command
	// ----------------------------------

	// 現在の局面に対してdf-pn詰め将棋ルーチンを呼び出す。
	void mate_dfpn(Position& pos, std::istringstream& is)
	{
#if !defined(USE_MATE_DFPN)
		cout << "Error! : define USE_MATE_DFPN" << endl;
		return;
#else

		// default 1000万ノード(どうせ先にメモリが足りなくなる)
		size_t nodes = 10000000;

		// df-pn用メモリ
		size_t mem = 1024;

		string token;
		while (is >> token)
		{
			if (token == "nodes")
				is >> nodes;
			else if (token == "mem")
				is >> mem;
		}

		cout << "df-pn mate :" << endl
			<< " nodes = " << nodes
			<< " mem   = " << mem << "[MB]" << endl;

		Mate::Dfpn::MateDfpnSolver dfpn(Mate::Dfpn::DfpnSolverType::Node64bit);
		dfpn.alloc(mem);

		Timer time;
		cout << "start mate." << endl;
		time.reset();
		Move m = dfpn.mate_dfpn(pos, (u32)nodes);
		cout << "time = " << time.elapsed() << endl;
		if (m != MOVE_NONE && m != MOVE_NULL)
		{
			auto nodes_searched = dfpn.get_nodes_searched();
			cout << "solved! , nodes_searched = " << nodes_searched << endl;

			auto pv = dfpn.get_pv();
			for (auto m : pv)
				cout << m << " ";

			cout << endl;
		}
		else {
			if (m == MOVE_NULL)
				cout << "solved! this is unmate." << endl;
			else if (dfpn.is_out_of_memory())
				cout << "out of memory" << endl;
			else if (dfpn.get_nodes_searched() >= nodes)
				cout << "unsolved , nodes limit." << endl;
			else
				cout << "unsolved. why?" << endl;
		}

#endif
		}

	// ----------------------------------
	//      "test matebench2" command
	// ----------------------------------

	// tanuki-詰将棋ルーチンのbench
	// そのうち削除するかも。

#if defined (TANUKI_MATE_ENGINE) || defined(YANEURAOU_MATE_ENGINE)
	// 詰将棋エンジンテスト用局面集
	static const char* TestMateEngineSfen[] = {
		// http://www.ne.jp/asahi/tetsu/toybox/shogi/kifu.htm
		"3sks3/9/4+P4/9/9/+B8/9/9/9 b S2rb4gs4n4l17p 1",
		// http://www.ne.jp/asahi/tetsu/toybox/shogi/kifu.htm
		"7nl/7k1/6p2/6S1p/9/9/9/9/9 b GS2r2b3g2s3n3l16p 1",
		// http://www.ne.jp/asahi/tetsu/toybox/shogi/kifu.htm
		"4k4/9/PPPPPPPPP/9/9/9/9/9/9 b B4L2rb4g4s4n9p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2Bukamuse_6700K%2Bcatshogi%2B20170430143005.csa&move_to=102
		"l2g5/2s3g2/3k1p2p/P2pp2P1/1pP4s1/p1+B6/NP1P+nPS1P/K1G4+p1/L6NL b RBGNLPrs3p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2Bcoduck_pi2_600MHz_1c%2BShogiNet%2B20170430110007.csa&move_to=100
		"6lnk/6+Rbl/2n4pp/7s1/1p2P2NP/p1P2PPP1/1P4GS1/6GK1/LNr5L b B2G2S6Pp 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BSM_1_25_Xeon_E5_2698_v4_40c%2BSILENT_MAJORITY_1.25_6950X%2B20170430103005.csa&move_to=195
		"lnks5/1pg1s4/2p5p/p4+r3/P1g6/1Nn6/BKN1P3P/9/LG2s4 w GSL2Prbl9p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2Bcatshogi%2Bgps_l%2B20170430070003.csa&move_to=134
		"l7l/2+Rbk4/3rp4/2p3pPs/p2P1p2p/2P1G4/P1N1PPN2/2GK2G2/L7L b B2S6Pgs2n 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2Bcatshogi%2BGikouAperyEvalMix_SeoTsume_i5-33%2B20170430063002.csa&move_to=127
		"l5g1l/2s+B5/p2ppp2p/5kpP1/3n5/6Pp1/P3PP1lP/2+nr2SS1/3N1GKRL w G2Pbgsn3p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BGc_at_Cortex-A53_4c%2BSaturday_Crush_4770K%2B20170430023007.csa&move_to=99
		"l4g2l/7k1/p1+Pp3pp/5ss1P/3Pp1gP1/P3SL3/N2GPK3/1+rP6/+p6RL w BG2N2Pbsn3p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BSM_1_25_Xeon_E5_2698_v4_40c%2Bukamuse_i7%2B20170430013007.csa&move_to=116
		"l2s3nl/3g1p+R+R1/p1k5p/2pPp4/1p1p5/5Sp2/PPP1PP2P/3G5/L1K4NL b BG2S2Pbg2np 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BGc_at_Cortex-A53_4c%2Bsonic%2B20170430013003.csa&move_to=149
		"ln7/2gk1S+S2/2+rpPp2G/2p5p/PP4P2/3B4P/K1SP3PN/1Sg2P+np1/L+r6L w L2Pbgn3p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BTest_NB10.5_i5_6200U%2BGikouAperyEvalMix_SeoTsume_i5-33%2B20170430010007.csa&move_to=121
		"6p1l/1+R1G2g2/5pns1/pp1pk3p/2p3P2/P7P/1L1PSP+b2/1SG1K2P1/L5G1L w N2Prbs2n3p 1",
		// http://wdoor.c.u-tokyo.ac.jp/shogi/view/index.cgi?csa=http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST/2017/04/30/wdoor%2Bfloodgate-300-10F%2BInoue%2Byeu%2B20170430003006.csa&move_to=144
		"lng3+R2/2kgs4/ppp6/1B1pp4/7B1/2P2pLp1/PP1PP3P/1S1K2p2/LN5GL b RG2SP2n3p 1",
	};
#endif

	// MATE ENGINEのテスト。(ENGINEに対して局面図を送信する)
	void mate_bench2(Position& pos, std::istringstream& is)
	{
#if !defined (TANUKI_MATE_ENGINE) && !defined(YANEURAOU_MATE_ENGINE)
		cout << "Error! : define TANUKI_MATE_ENGINE or YANEURAOU_MATE_ENGINE" << endl;
#else
		string token;

		// →　デフォルト1024にしておかないと置換表あふれるな。
		string ttSize = (is >> token) ? token : "1024";

		Options["USI_Hash"] = ttSize;

		Search::LimitsType limits;

		// ベンチマークモードにしておかないとPVの出力のときに置換表を漁られて探索に影響がある。
		limits.bench = true;

		// 探索制限
		limits.nodes = 0;
		limits.mate = 100000; // 100秒

		// Optionsの影響を受けると嫌なので、その他の条件を固定しておく。
		limits.enteringKingRule = EKR_NONE;

		// 評価関数の読み込み等
		is_ready();

		// トータルの探索したノード数d
		int64_t nodes = 0;

		// main threadが探索したノード数
		int64_t nodes_main = 0;

		// ベンチの計測用タイマー
		Timer time;
		time.reset();

		for (const char* sfen : TestMateEngineSfen) {
			Position pos;
			StateListPtr st(new StateList(1));
			pos.set(sfen, &st->back(), Threads.main());

			sync_cout << "\nPosition: " << sfen << sync_endl;

			// 探索時にnpsが表示されるが、それはこのglobalなTimerに基づくので探索ごとにリセットを行なうようにする。
			Time.reset();

			Threads.start_thinking(pos, st , limits);
			Threads.main()->wait_for_search_finished(); // 探索の終了を待つ。

			nodes += Threads.nodes_searched();
			nodes_main += Threads.main()->rootPos.this_thread()->nodes.load(memory_order_relaxed);
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

#endif // !defined (TANUKI_MATE_ENGINE) && !defined(YANEURAOU_MATE_ENGINE)
	}

} // namespace


// ----------------------------------
//      "test" command Decorator
// ----------------------------------

namespace Test
{
	// 詰み関係のテストコマンド。コマンドを処理した時 trueが返る。
	bool mate_test_cmd(Position& pos , std::istringstream& is, const std::string& token)
	{
		if (token == "genmate") gen_mate(pos, is);            // N手詰みの局面を生成する。
		else if (token == "matebench") mate_bench(pos, is);   // 詰みルーチンに関するbenchをとる。
		else if (token == "matebench2") mate_bench2(pos, is);      // MATE ENGINEのテスト。(ENGINEに対して局面図を送信する)
		else if (token == "dfpn")       mate_dfpn(pos, is);        // 現在の局面に対してdf-pn詰め将棋ルーチンを呼び出す。
		//else if (token == "matesolve") mate_solve(pos, is);      // 現在の局面に対してN手詰みルーチンを呼び出す。
		else return false;									  // どのコマンドも処理することがなかった
			
		// いずれかのコマンドを処理した。
		return true;
	}

}


#endif // defined(ENABLE_TEST_CMD)
