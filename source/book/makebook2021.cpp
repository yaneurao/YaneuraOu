#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) && defined(EVAL_LEARN)

// -----------------------
// MCTSで定跡を生成する
// -----------------------

#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>

#include "../usi.h"
#include "../misc.h"
#include "../thread.h"
#include "../book/book.h"
#include "../learn/learn.h"

using namespace std;
using namespace Book;

namespace {

	// MCTSで定跡を生成する本体
	class MctsMakeBook
	{
	public:

		// MCTSをやって定跡を生成する。
		void make_book(Position& pos, istringstream& is)
		{
			// loop_maxに達するか、"stop"が来るまで回る

			// この回数だけ探索したら終了する。
			u64 loop_max = 10000;

			// depth_min～depth_maxの間のランダムな深さでsearch()を呼び出す。
			int depth_min = 6;
			int depth_max = 8;

			// 定跡ファイル名
			string filename = "book2021.db";

			// 定跡ファイルの保存間隔。デフォルト、30分ごと。
			TimePoint book_save_interval = 60 * 30;

			// 最大手数
			int max_ply = 384;

			string token;
			while (is >> token)
			{
				if (token == "loop")
					is >> loop_max;
				else if (token == "depth_min")
					is >> depth_min;
				else if (token == "depth_max")
					is >> depth_max;
				else if (token == "filename")
					is >> filename;
				else if (token == "book_save_interval")
					is >> book_save_interval;
				else if (token == "max_ply")
					is >> max_ply;
			}

			size_t thread_num = Options["Threads"];

			cout << "gen_mate command" << endl
				<< "  Threads            = " << thread_num << endl
				<< "  loop_max           = " << loop_max << endl
				<< "  depth_min          = " << depth_min << endl
				<< "  depth_max          = " << depth_max << endl
				<< "  book filename      = " << filename << endl
				<< "  max_ply            = " << max_ply << endl
				<< "  book_save_interval = " << book_save_interval << endl
				;

			// depth_minとdepth_maxの大小関係が逆転している
			if (depth_min > depth_max)
			{
				cout << "Error! depth_min > depth_max" << endl;
				return;
			}

			// 定跡ファイルの読み込み。
			// 新規に作るかも知れないので、存在しなくとも構わない。
			book.read_book(filename, false);

			this->loop_max       = loop_max;
			this->nodes_searched = 0;
			this->depth_min      = depth_min;
			this->depth_max      = depth_max;
			this->max_ply        = max_ply;

			// スレッドを開始する。
			auto threads = make_unique<std::thread[]>(thread_num);
			for (size_t i = 0; i < thread_num; ++i)
				threads[i] = std::thread(&MctsMakeBook::Worker,this, i);

			// 定跡DBを書き出す。
			auto save_book = [&]()
			{
				std::lock_guard<std::mutex> lk(book_mutex);

				sync_cout << "savebook ..start : " << filename << sync_endl;
				book.write_book(filename);
				sync_cout << "savebook ..done." << sync_endl;
			};

			// 定跡保存用のtimer
			Timer time;

			while (nodes_searched < loop_max)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));

				// 進捗ここで出力したほうがいいかも？

				// 定期的に定跡ファイルを保存する
				if (time.elapsed() >= book_save_interval * 1000)
				{
					save_book();
					time.reset();
				}
			}

			// スレッドの終了を待機する。
			for (size_t i = 0; i < thread_num; ++i)
				threads[i].join();

			// 最後にも保存しないといけない。
			save_book();

			cout << "makebook mcts , done." << endl;
		}

		// 各探索用スレッドのentry point
		void Worker(size_t thread_id)
		{
			//sync_cout << "thread_id = " << thread_id << sync_endl;

			WinProcGroup::bindThisThread(thread_id);

			Position pos;
			StateInfo si;
			pos.set_hirate(&si,Threads[thread_id]); // 平手の初期局面から

			UctSearch(pos);
		}

		// 並列UCT探索
		// pos : この局面から探索する。
		void UctSearch(Position& pos)
		{
			// 定跡DBにこの局面が登録されているか調べる。
			auto book_pos = book.find(pos);

		}

		// 指定された局面から、終局までplayout(対局)を試みる。
		// 返し値 :
		//   1 : 開始手番側の勝利
		//   0 : 引き分け
		//  -1 : 開始手番側でないほうの勝利
		int Playout(Position& pos)
		{
			// 開始時の手番
			auto rootColor = pos.side_to_move();

			// do_move()時に使うStateInfoの配列
			auto states = make_unique<StateInfo[]>((size_t)(max_ply + 1));

			// 自己対局
			while (pos.game_ply() <= max_ply)
			{
				// 宣言勝ち。現在の手番側の勝ち。
				if (pos.DeclarationWin())
					return pos.side_to_move() == rootColor ? 1 : -1;

				// 探索深さにランダム性を持たせることによって、毎回異なる棋譜になるようにする。
				int search_depth = depth_min + (int)prng.rand(depth_max - depth_min + 1);
				auto pv = Learner::search(pos, search_depth, 1);

				Move m = pv.second[0];

				// 指し手が存在しなかった。現在の手番側の負け。
				if (m == MOVE_NONE)
					return pos.side_to_move() == rootColor ? -1 : 1;

				pos.do_move(m, states[pos.game_ply()]);
			}

			// 引き分け
			return 0;
		}


	private:
		// 定跡DB本体
		MemoryBook book;

		// 探索したノード数
		atomic<u64> nodes_searched;

		// この回数だけ探索したら終了する。
		u64 loop_max;

		// depth_min～depth_maxの間のランダムな深さでsearch()を呼び出す。
		int depth_min;
		int depth_max;

		// 最大手数
		int max_ply;

		// Learner::search()の探索深さを変えるための乱数
		AsyncPRNG prng;

		// MemoryBookのsaveに対するmutex。
		mutex book_mutex;
	};
}

namespace Book
{
	// 2019年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2021(Position& pos, istringstream& is, const string& token)
	{
		if (token == "mcts")
		{
			MctsMakeBook mcts;
			mcts.make_book(pos, is);
			return 1;
		}

		return 0;
	}
}

#endif //defined (ENABLE_MAKEBOOK_CMD) && defined(EVAL_LEARN)

