#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) && defined(YANEURAOU_ENGINE_DEEP)

// -----------------------
// MCTSで定跡を生成する
// -----------------------

/*
	dlshogiの定跡生成部を参考にしつつも独自の改良を加えた。

	大きな改良点) 親ノードに訪問回数を伝播する。

		dlshogiはこれをやっていない。
		これを行うと、局面に循環があると訪問回数が発散してしまうからだと思う。
		// これだと訪問回数を親ノードに伝播しないと、それぞれの局面で少ないPlayoutで事前に思考しているに過ぎない。

		そこで、親ノードに訪問回数は伝播するが、局面の合流は処理しないようにする。

		この場合、合流が多い変化では同じ局面を何度も探索することになりパフォーマンスが落ちるが、
		それを改善するためにleaf nodeでの思考結果はcacheしておき、同じ局面に対しては探索部を２回
		呼び出さないようにして解決する。

		また合流を処理しないため、同一の局面であっても経路によって異なるNodeとなるが、書き出す定跡ファイルとは別に
		この経路情報を別のファイルに保存しておくので、前回の定跡の生成途中から再開できる。
*/

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

			// 一つの局面(leaf node)でのPlayoutの回数
			u64 playout = 10000;

			// 定跡ファイル名
			string filename = "book2021.db";

			// 定跡cacheファイル名
			string cache_filename = "book_cache.db";

			// 定跡ファイルの保存間隔。デフォルト、30分ごと。
			TimePoint book_save_interval = 60 * 30;

			// 最大手数
			int max_ply = 384;

			string token;
			while (is >> token)
			{
				if (token == "loop")
					is >> loop_max;
				else if (token == "playout")
					is >> playout;
				else if (token == "filename")
					is >> filename;
				else if (token == "cache_filename")
					is >> cache_filename;
				else if (token == "book_save_interval")
					is >> book_save_interval;
				else if (token == "max_ply")
					is >> max_ply;
			}

			cout << "makebook mcts command" << endl
				<< "  loop_max           = " << loop_max << endl
				<< "  playout            = " << playout << endl
				<< "  book filename      = " << filename << endl
				<< "  cache filename     = " << cache_filename << endl
				<< "  book_save_interval = " << book_save_interval << endl
				<< "  max_ply            = " << max_ply << endl
				;

			// 定跡ファイルの読み込み。
			// 新規に作るかも知れないので、存在しなくとも構わない。
			book.read_book(filename, false);

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

			for (u64 loop_counter = 0 ; loop_counter < loop_max ; ++loop_counter)
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
				//int search_depth = depth_min + (int)prng.rand(depth_max - depth_min + 1);
				//auto pv = Learner::search(pos, search_depth, 1);

				//Move m = pv.second[0];
				Move m;

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

#endif // defined (ENABLE_MAKEBOOK_CMD) && defined(YANEURAOU_ENGINE_DEEP)

