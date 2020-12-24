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
		Mate::Dfpn::MateDfpnSolver dfpn;
		dfpn.alloc(dfpn_mem);
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

		auto bench = [&](function<Move()> solver, string test_name) {
			Timer timer;
			timer.reset();

			cout << "===== " << test_name << " =====" << endl
				<< "test start" << endl;

			// 解けた問題数
			u32 solved = 0;

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
						<< " , solved per second = " << (double)times * 1000 / (elapsed + 0.00000001) << endl;
				}

				auto sfen = (*problems)[i];
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
		if (test_mode & 2)
			bench([&]() { return dfpn.mate_dfpn(pos, nodes_limit); }, "mate_dfpn");
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

		// default 50万ノード
		size_t nodes = 500000;

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

		Mate::Dfpn::MateDfpnSolver dfpn;
		dfpn.alloc(mem);

		Move m = dfpn.mate_dfpn(pos, (u32)nodes);
		if (m != MOVE_NONE && m != MOVE_NULL)
		{
			auto nodes_searched = dfpn.get_node_searched();
			cout << "solved! , nodes_searched = " << nodes_searched << endl;

			auto pv = dfpn.get_pv();
			for (auto m : pv)
				cout << m << " ";

			cout << endl;
		}
		else {
			cout << "unsolved." << endl;
		}

#endif
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
		else if (token == "matedfpn") mate_dfpn(pos, is);     // 現在の局面に対してdf-pn詰め将棋ルーチンを呼び出す。
		else return false;									  // どのコマンドも処理することがなかった
			
		// いずれかのコマンドを処理した。
		return true;
	}

}


#endif // defined(ENABLE_TEST_CMD)
