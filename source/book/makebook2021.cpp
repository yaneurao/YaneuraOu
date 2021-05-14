#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
// いまのところ、ふかうら王のみ対応。気が向いたら、NNUEにも対応させるが、NNUEの評価関数だとMCTS、あまり相性良くない気も…。

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

		その他、いくつかのテクニックを導入することでMCTSで定跡を生成する上での問題点を解決している。

*/

/*
	それぞれのfileフォーマット

	入力)
		root_sfen.txt
		// 探索開始局面
		例)
		sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
		↑のようなsfen形式か、
		startpos moves 7g7f ...
		のようなUSIプロトコルの"position"コマンドとして送られる形式でも可。
		// この場合、それぞれの局面が探索開始対象となる。

	出力)
		leaf_sfen.txt
			探索済leaf nodeのsfenとそのvalue,指し手
			例)
				sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 // sfen
				0.5,10000,7g7f,100
				// この局面の探索した時のvalue(float値,手番側から見た値),playoutの回数,1つ目の指し手(ないときはresign),1つ目の指し手の探索回数

		mctsbook.serialized
			探索tree(内部状態)をそのまま書き出したの。

		user_book_mcts.db
			やねうら王で定跡ファイルとして使えるファイル

*/

#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_set>

#include "../usi.h"
#include "../misc.h"
#include "../thread.h"
#include "../book/book.h"
#include "../learn/learn.h"

using namespace std;
using namespace Book;

namespace {

	// =================================================================
	//                        探索済みのNode
	// =================================================================

	// 探索部を呼び出してある局面で探索した結果
	struct SearchResult
	{
		float value;    // この局面の評価値(手番側から見た値)
		Move16 move;    // この局面の最善手(無い場合はMOVE_NONE)
		u64 move_count; // 探索したノード数
	};

	// 探索したノードを記録しておく構造体。
	// ファイルへのserialize/deserializeもできる。
	class SearchNodes
	{
	public:
		// ファイルから読み込み。
		void Deserialize(const std::string& filename) {}

		// ファイルに保存。
		void Serialize(const std::string& filename) {}

		// 要素ひとつ追加する
		void append(const SearchResult& result) {}

		// 指定されたsfenが格納されているかを調べる。
		// sfenは末尾に手数は書かれていないものとする。
		SearchResult* find(const std::string& sfen) { return nullptr; }

	private:
		// 探索した結果
		std::unordered_map<std::string /*sfen*/, SearchResult> nodes;
	};

	// =================================================================
	//                        MCTSの探索木
	// =================================================================

	// MCTSで用いるNode構造体
	struct MctsNode {

	};

	// MCTSの探索木
	class MctsTree {


	};

	// =================================================================
	//                   MCTSの探索rootを表現する
	// =================================================================

	// 探索Root集合の型
	typedef std::vector<std::string> RootSfens;

	// positionコマンドに渡す形式で書かれているsfenファイルを読み込み、そのsfen集合を返す。
	// 重複局面は除去される。
	// all_node   : そこまでの手順(経由した各局面)も含めてすべて読み込む。
	// ignore_ply : 手数を無視する。(sfen化するときに手数をtrimする)
	// 
	// file formatは、
	// "startpos move xxxx xxxx"
	// "[sfen文字列] moves xxxx xxxx"
	RootSfens ReadPositionFile(const string& filename, bool all_node, bool ignore_ply)
	{
		std::unordered_set<std::string> sfens;

		TextFileReader reader;
		reader.Open(filename);

		std::string line, token, sfen;
		while (!reader.ReadLine(line).is_eof())
		{
			// line : この1行がpositionコマンドに渡す文字列と同等のもの
			std::istringstream is(line);
			is >> token;
			if (token == "startpos")
			{
				sfen = SFEN_HIRATE;
				is >> token; // "moves"を消費する
			}
			else {
				// "sfen"は書いてなくても可。
				if (token != "sfen")
					sfen += token + " ";
				while (is >> token && token != "moves")
					sfen += token + " ";
			}

			// 新しく渡す局面なので古いものは捨てて新しいものを作る。
			auto states = StateListPtr(new StateList(1));
			Position pos;
			pos.set(sfen, &states->back(), Threads.main());

			// 返す局面集合に追加する関数
			auto insert = [&] {
				std::string s = pos.sfen();
				if (sfens.count(s) == 0)
					sfens.insert(s);
			};

			// 開始局面をsfen化したものを格納
			if (all_node)
				insert();

			// 指し手のparser
			Move m;
			while (is >> token && (m = USI::to_move(pos, token)) != MOVE_NONE)
			{
				// 1手進めるごとにStateInfoが積まれていく。これは千日手の検出のために必要。
				states->emplace_back();
				if (m == MOVE_NULL) // do_move に MOVE_NULL を与えると死ぬので
					pos.do_null_move(states->back());
				else
					pos.do_move(m, states->back());

				if (all_node)
					insert();
			}

			// all_node == falseならば、最後の局面だけ返す。
			if (!all_node)
				insert();
		}

		// vectorに変換して返す。
		std::vector<std::string> sfens_vector;
		for (auto s : sfens)
		{
			if (ignore_ply)
				s = StringExtension::trim_number(s);
			sfens_vector.emplace_back(s);
		}

		return sfens_vector;
	}

	// =================================================================
	//             mcts定跡生成コマンド本体
	// =================================================================

	// MCTSで定跡を生成する本体
	class MctsMakeBook
	{
	public:

		// MCTSをやって定跡を生成する。
		void make_book(Position& pos, istringstream& is)
		{
			// loop_maxに達するか、"stop"が来るまで回る

			// この回数だけrootから探索したら終了する。
			u64 loop_max = 10000000;

			// 一つの局面(leaf node)でのPlayoutの回数
			u64 playout = 10000;

			// 探索root集合
			string root_filename = "root_sfen.txt";

			// 定跡ファイル名
			string book_filename = "user_book_mcts.db";

			// 定跡cacheファイル名
			string serialized_filename = "mctsbook.serialized";

			// 定跡のleaf nodeのsfen
			string leaf_filename = "leaf_sfen.txt";


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
				else if (token == "root_filename")
					is >> root_filename;
				else if (token == "book_filename")
					is >> book_filename;
				else if (token == "serialized_filename")
					is >> serialized_filename;
				else if (token == "leaf_filename")
					is >> leaf_filename;
				else if (token == "book_save_interval")
					is >> book_save_interval;
				else if (token == "max_ply")
					is >> max_ply;
			}

			cout << "makebook mcts command" << endl
				<< "  root filename       = " << root_filename << endl
				<< "  loop                = " << loop_max << endl
				<< "  playout             = " << playout << endl
				<< "  book_filename       = " << book_filename << endl
				<< "  serialized_filename = " << serialized_filename << endl
				<< "  book_save_interval  = " << book_save_interval << endl
				<< "  max_ply             = " << max_ply << endl
				;

			cout << "read root file.." << endl;

			// そこまでの手順も含めてすべて読み込み、sfenにする。
			auto roots = ReadPositionFile(root_filename, true, true);
			cout << "  root sfens size()  = " << roots.size() << endl;

#if 0
			// デバッグ用に読み込まれたsfenを出力する。
			for (auto sfen : roots)
				cout << "sfen " << sfen << endl;
#endif



#if 0

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
#endif

			cout << "makebook mcts , done." << endl;
		}

#if 0
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
#endif
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

#endif // defined (ENABLE_MAKEBOOK_CMD) && (/*defined(EVAL_LEARN) ||*/ defined(YANEURAOU_ENGINE_DEEP))
