#include "../shogi.h"

// 学習関係のルーチン
// 1) 棋譜の自動生成
//   → gensfenコマンド
// 2) 生成した棋譜からの評価関数パラメーターの学習
//   → learnコマンド
// 3) 定跡の自動生成
//   → makebook thinkコマンド
// 4) 局後自動検討モード
//   →　考え中
// etc..


#if defined(EVAL_LEARN)

#include "learn.h"

#include <sstream>
#include <fstream>
#include <unordered_set>

#include "../misc.h"
#include "../thread.h"
#include "../position.h"
#include "../extra/book/book.h"
#include "../tt.h"
#include "multi_think.h"

// これ、Windows専用なの？よくわからん…。
#if defined(_MSC_VER)
#include <filesystem>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

extern Book::MemoryBook book;
extern void is_ready();

namespace Learner
{
// いまのところ、やねうら王2017Early/王手将棋しか、このスタブを持っていない。
extern pair<Value, vector<Move> > qsearch(Position& pos);
extern pair<Value, vector<Move> >  search(Position& pos, int depth);

// -----------------------------------
//    局面のファイルへの書き出し
// -----------------------------------

// Sfenを書き出して行くためのヘルパクラス
struct SfenWriter
{
	// 書き出すファイル名と生成するスレッドの数
	SfenWriter(string filename, int thread_num)
	{
		sfen_buffers_pool.reserve(thread_num * 10);
		sfen_buffers.resize(thread_num);

		fs.open(filename, ios::out | ios::binary | ios::app);

		finished = false;
	}

	~SfenWriter()
	{
		finished = true;
		file_worker_thread.join();
	}

	// 各スレッドについて、この局面数ごとにファイルにflushする。
	const size_t SFEN_WRITE_SIZE = 5000;

#ifdef  WRITE_PACKED_SFEN

	// 局面と評価値をペアにして1つ書き出す(packされたsfen形式で)
	void write(size_t thread_id, const PackedSfen& sfen, s16 value
#ifdef GENSFEN_SAVE_FIRST_MOVE
		// PVの初手
		,Move move
#endif
	)
	{
		// スレッドごとにbufferを持っていて、そこに追加する。
		// bufferが溢れたら、ファイルに書き出す。

		// このバッファはスレッドごとに用意されている。
		auto& buf = sfen_buffers[thread_id];

		// 初回とスレッドバッファを書き出した直後はbufがないので確保する。
		if (!buf)
		{
			buf = shared_ptr<vector<PackedSfenValue>>(new vector<PackedSfenValue>());
			buf->reserve(SFEN_WRITE_SIZE);
		}

		PackedSfenValue ps;
		ps.sfen = sfen;
		ps.score = value;
#ifdef GENSFEN_SAVE_FIRST_MOVE
		ps.move = (u16)move;
#endif

		// スレッドごとに用意されており、一つのスレッドが同時にこのwrite()関数を呼び出さないので
		// この時点では排他する必要はない。
		buf->push_back(ps);

		if (buf->size() >= SFEN_WRITE_SIZE)
		{
			// sfen_buffers_poolに積んでおけばあとはworkerがよきに計らってくれる。

			// sfen_buffers_poolの内容を変更するときはmutexのlockが必要。
			std::unique_lock<Mutex> lk(mutex);
			sfen_buffers_pool.push_back(buf);

			// この瞬間、参照を剥がしておかないとこのスレッドとwrite workerのほうから同時に
			// shared_ptrの参照カウントをいじることになってまずい。
			buf = nullptr;

			// buf == nullptrにしておけば次回にこの関数が呼び出されたときにバッファは確保される。
		}
	}
#else
	// 局面を1行書き出す
	void write(size_t thread_id, string line)
	{
		// スレッドごとにbufferを持っていて、そこに追加する。
		// bufferが溢れたら、ファイルに書き出す。

		auto& buf = sfen_buffers[thread_id];

		auto buf_reserve = [&]()
		{
			buf = shared_ptr<vector<string>>(new vector<string>());
			buf->reserve(SFEN_WRITE_SIZE);
		};

		if (!buf)
			buf_reserve();

		buf->push_back(line);
		if (buf->size() >= SFEN_WRITE_SIZE)
		{
			// sfen_buffers_poolに積んでおけばあとはworkerがよきに計らってくれる。
			{
				std::unique_lock<Mutex> lk(mutex);
				sfen_buffers_pool.push_back(buf);
				buf = nullptr;
			}
			buf_reserve();
		}
	}
#endif

	// バッファに残っている分をファイルに書き出す。
	void finalize(size_t thread_id)
	{
		auto& buf = sfen_buffers[thread_id];
		if (buf->size() != 0)
		{
			std::unique_lock<Mutex> lk(mutex);
			sfen_buffers_pool.push_back(buf);
			buf = nullptr;
		}
	}

	// write_workerスレッドを開始する。
	void start_file_write_worker()
	{
		file_worker_thread = std::thread([&] { this->file_write_worker(); });
	}

	// ファイルに書き出すの専用スレッド
	void file_write_worker()
	{
		auto output_status = [&]()
		{
			// 現在時刻も出力
			auto now = std::chrono::system_clock::now();
			auto tp = std::chrono::system_clock::to_time_t(now);

			cout << endl << sfen_write_count << " sfens , at " << std::ctime(&tp);

			// flush()はこのタイミングで十分。
			fs.flush();
		};

		while (!finished || sfen_buffers_pool.size())
		{
#ifdef  WRITE_PACKED_SFEN
			vector<shared_ptr<vector<PackedSfenValue>>> buffers;
#else
			vector<shared_ptr<vector<string>>> buffers;
#endif
			{
				std::unique_lock<Mutex> lk(mutex);

				// まるごとコピー
				buffers = sfen_buffers_pool;
				sfen_buffers_pool.clear();
			}

			// 何も取得しなかったならsleep()
			if (!buffers.size())
				sleep(100);
			else
			{
				// バッファがどれだけ積まれているのかデバッグのために出力
				//cout << "[" << buffers.size() << "]";

				for (auto ptr : buffers)
				{
#ifdef  WRITE_PACKED_SFEN
					fs.write((const char*)&((*ptr)[0]), sizeof(PackedSfenValue) * ptr->size());
#else
					for (auto line : *ptr)
						fs << line;
#endif

//					cout << "[" << ptr->size() << "]";
					sfen_write_count += ptr->size();

					// 棋譜を書き出すごとに'.'を出力。
					cout << ".";

					// 40回×GEN_SFENS_TIMESTAMP_OUTPUT_INTERVALごとに処理した局面数を出力
					if ((++time_stamp_count % (u64(40) * GEN_SFENS_TIMESTAMP_OUTPUT_INTERVAL)) == 0)
						output_status();
				}
			}
		}

		// 終了前にもう一度、タイムスタンプを出力。
		output_status();
	}

private:

	fstream fs;

	// ファイルに書き込む用のthread
	std::thread file_worker_thread;
	// 終了したかのフラグ
	atomic<bool> finished;

	// タイムスタンプの出力用のカウンター
	u64 time_stamp_count = 0;

	// ファイルに書き出す前のバッファ
#ifdef  WRITE_PACKED_SFEN
	vector<shared_ptr<vector<PackedSfenValue>>> sfen_buffers;
	vector<shared_ptr<vector<PackedSfenValue>>> sfen_buffers_pool;
#else
	vector<shared_ptr<vector<string>>> sfen_buffers;
	vector<shared_ptr<vector<string>>> sfen_buffers_pool;
#endif

	// sfen_buffers_poolにアクセスするときに必要なmutex
	Mutex mutex;

	// 書きだした局面の数
	u64 sfen_write_count = 0;

};

// -----------------------------------
//  棋譜を生成するworker(スレッドごと)
// -----------------------------------

// 複数スレッドでsfenを生成するためのクラス
struct MultiThinkGenSfen: public MultiThink
{
  MultiThinkGenSfen(int search_depth_, int search_depth2_, SfenWriter& sw_)
	  : search_depth(search_depth_), search_depth2(search_depth2_), sw(sw_)
  {
    // 乱数を時刻で初期化しないとまずい。
    // (同じ乱数列だと同じ棋譜が生成されかねないため)
    set_prng(PRNG());

	hash.resize(GENSFEN_HASH_SIZE);
  }

  virtual void thread_worker(size_t thread_id);
  void start_file_write_worker() { sw.start_file_write_worker(); }

  //  search_depth = 通常探索の探索深さ
  int search_depth;
  int search_depth2;

  // 生成する局面の評価値の上限
  int eval_limit;

  // sfenの書き出し器
  SfenWriter& sw;

  // 同一局面の書き出しを制限するためのhash
  // hash_indexを求めるためのmaskに使うので、2**Nでなければならない。
  static const u64 GENSFEN_HASH_SIZE = 64 * 1024 * 1024;

  vector<HASH_KEY> hash; // 64MB*sizeof(HASH_KEY) = 512MB
};

//  thread_id    = 0..Threads.size()-1
void MultiThinkGenSfen::thread_worker(size_t thread_id)
{
	StateInfo state[MAX_PLY + 8]; // StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
	Move m = MOVE_NONE;

	// 定跡の指し手を用いるモード
	int book_ply = Options["BookMoves"];

	// 規定回数回になるまで繰り返し
	while (true)
	{
		// Positionに対して従属スレッドの設定が必要。
		// 並列化するときは、Threads (これが実体が vector<Thread*>なので、
		// Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
		auto th = Threads[thread_id];

		auto& pos = th->rootPos;
		pos.set_hirate();
		pos.set_this_thread(th);

		// ply : 初期局面からの手数
		for (int ply = 0; ply < MAX_PLY - 20; ++ply)
		{
			// 詰んでいるなら次の対局に
			if (pos.is_mated())
				break;

			// 定跡を使用するのか？
			if (pos.game_ply() <= book_ply)
			{
				auto it = book.find(pos);
				if (it != book.end() && it->second.size() != 0)
				{
					// 定跡にhitした。it->second->size()!=0をチェックしておかないと
					// 指し手のない定跡が登録されていたときに困る。

					const auto& move_list = it->second;

					const auto& move = move_list[(size_t)rand(move_list.size())];
					auto bestMove = move.bestMove;
					// この指し手に不成があってもLEGALであるならこの指し手で進めるべき。
					if (pos.pseudo_legal(bestMove) && pos.legal(bestMove))
					{
						// この指し手で1手進める。
						m = bestMove;

						// 定跡の局面であっても、一定確率でランダムムーブは行なう。
						goto RANDOM_MOVE;
					}
				}
			}

			{
				// search_depth～search_depth2 手読みの評価値とPV(最善応手列)
				// 探索窓を狭めておいても問題ないはず。

				int depth = search_depth + (int)rand(search_depth2 - search_depth + 1);

				auto pv_value1 = Learner::search(pos, depth);

				auto value1 = pv_value1.first;
				auto pv1 = pv_value1.second;

				// 評価値の絶対値がこの値以上の局面については
				// その局面を学習に使うのはあまり意味がないのでこの試合を終了する。
				if (abs(value1) >= eval_limit)
				{
#if 0
					sync_cout << pos << "eval limit = " << eval_limit << " over , move = " << pv1[0] << sync_endl;
#endif
					break;
				}

				// 何らかの千日手局面に突入したので局面生成を終了する。
				auto draw_type = pos.is_repetition();
				if (draw_type != REPETITION_NONE)
				{
#if 0
					sync_cout << pos << "repetition , move = " << pv1[0] << sync_endl;
#endif
					break;
				}

#if 0
				// 0手読み(静止探索のみ)の評価値とPV(最善応手列)
				auto pv_value2 = qsearch(pos);
				auto value2 = pv_value2.first;
				auto pv2 = pv_value2.second;
#endif

				// 上のように、search()の直後にqsearch()をすると、search()で置換表に格納されてしまって、
				// qsearch()が置換表にhitして、search()と同じ評価値が返るので注意。

				// 局面のsfen,3手読みでの最善手,0手読みでの評価値
				// これをファイルか何かに書き出すと良い。
				//      cout << pos.sfen() << "," << value1 << "," << value2 << "," << endl;

				// PVの指し手でleaf nodeまで進めて、そのleaf nodeでevaluate()を呼び出した値を用いる。
				auto evaluate_leaf = [&](Position& pos , auto& pv)
				{
					auto rootColor = pos.side_to_move();

					int ply2 = ply;
					for (auto m : pv)
					{
						// デバッグ用の検証として、途中に非合法手が存在しないことを確認する。
						// NULL_MOVEはこないものとする。
#ifdef TEST_LEGAL_LEAF
						// 非合法手はやってこないはずなのだが。
						if (!pos.pseudo_legal(m) || !pos.legal(m))
						{
							cout << pos << m << endl;
							ASSERT_LV3(false);
						}
#endif
						pos.do_move(m, state[ply2++]);
						
						// 毎ノードevaluate()を呼び出さないと、evaluate()の差分計算が出来ないので注意！
						Eval::evaluate(pos);
						//						cout << "move = m " << m << " , evaluate = " << Eval::evaluate(pos) << endl;
					}

					// leafに到達
					//      cout << pos;

					auto v = Eval::evaluate(pos);
					// evaluate()は手番側の評価値を返すので、
					// root_colorと違う手番なら、vを反転させて返さないといけない。
					if (rootColor != pos.side_to_move())
						v = -v;

					// 巻き戻す。
					// C++x14にもなって、いまだreverseで回すforeachすらないのか…。
					//  for (auto it : boost::adaptors::reverse(pv))

					for (auto it = pv.rbegin(); it != pv.rend(); ++it)
						pos.undo_move(*it);

					return v;
				};

				// leaf nodeでのroot colorから見たevaluate()の値を取得。
				// auto leaf_value = evaluate_leaf(pos , pv1);

				// →　置換表にhitしたとき、PVが途中で枝刈りされてしまうので、
				// 駒の取り合いの途中の変な局面までのPVしかないと、局面の評価値として
				// 適切ではない気がする。

				auto leaf_value = value1;

#if 0
				//				cout << pv_value1.first << " , " << leaf_value << endl;
				// dbg_hit_on(pv_value1.first == leaf_value);
				// Total 150402 Hits 127195 hit rate (%) 84.569

				// qsearch()中に置換表の指し手で枝刈りされたのか..。
				// これ、教師としては少し気持ち悪いので、そういう局面を除外する。
				// これによって教師が偏るということはないと思うが..
				if (pv_value1.first != leaf_value)
					goto NEXT_MOVE;

				// →　局面が偏るのが怖いので実験してからでいいや。
#endif

#if 0
				//				dbg_hit_on(pv1.size() >= search_depth);
				// Total 101949 Hits 101794 hit rate (%) 99.847
				// 置換表にヒットするなどしてPVが途中で切れるケースは全体の0.15%程度。
				// このケースを捨てたほうがいいかも知れん。

				if ((int)pv1.size() < search_depth)
					goto NEXT_MOVE;
#endif

				// depth 0の場合、pvが得られていないのでdepth 2で探索しなおす。
				if (search_depth <= 0)
				{
					pv_value1 = Learner::search(pos, 2);
					pv1 = pv_value1.second;
				}

				// 16手目までの局面、類似局面ばかりなので
				// 学習に用いると過学習になりかねないから書き出さない。
				if (ply < 16)
					goto SKIP_WRITE;

				// 同一局面を書き出したところか？
				// これ、複数のPCで並列して生成していると同じ局面が含まれることがあるので
				// 読み込みのときにも同様の処理をしたほうが良い。
				{
					auto key = pos.key();
					auto hash_index = (size_t)(key & (GENSFEN_HASH_SIZE - 1));
					auto key2 = hash[hash_index];
					if (key == key2)
						goto SKIP_WRITE;
					hash[hash_index] = key; // 今回のkeyに入れ替えておく。
				}

				{
					// 局面を書き出そうと思ったら規定回数に達していた。
					// get_next_loop_count()内でカウンターを加算するので
					// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
					auto loop_count = get_next_loop_count();
					if (loop_count == UINT64_MAX)
						goto FINALIZE;

					// 100k局面に1回ぐらい置換表の世代を進める。
					if ((loop_count % 100000) == 0)
						TT.new_search();

#ifdef WRITE_PACKED_SFEN
					PackedSfen sfen;
					// packを要求されているならpackされたsfenとそのときの評価値を書き出す。
					pos.sfen_pack(sfen);
					// このwriteがスレッド排他を行うので、ここでの排他は不要。
#ifndef GENSFEN_SAVE_FIRST_MOVE
					sw.write(thread_id, sfen, leaf_value);
#else
					// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
					ASSERT_LV3(pv_value1.second.size() >= 1);
					Move pv_move1 = pv_value1.second[0];
					sw.write(thread_id, sfen, leaf_value, pv_move1);
#endif

#if 0
					// デバッグ用に局面を表示させてみる。
					sync_cout << pos << "leaf value = " << leaf_value << sync_endl;
#endif

#ifdef TEST_UNPACK_SFEN

					// sfenのpack test
					// pack()してunpack()したものが元のsfenと一致するのかのテスト。

			  //      pos.sfen_pack(data);
					auto sfen = pos.sfen_unpack(data);
					auto pos_sfen = pos.sfen();

					// 手数の部分の出力がないので異なる。末尾の数字を消すと一致するはず。
					auto trim = [](std::string& s)
					{
						while (true)
						{
							auto c = *s.rbegin();
							if (c < '0' || '9' < c)
								break;
							s.pop_back();
						}
					};
					trim(sfen);
					trim(pos_sfen);

					if (sfen != pos_sfen)
					{
						cout << "Error: sfen packer error\n" << sfen << endl << pos_sfen << endl;
					}
#endif

#else // WRITE_PACKED_SFEN

					{
						// C++のiostreamに対するスレッド排他は自前で行なう必要がある。
						std::unique_lock<Mutex> lk(io_mutex);

						// sfenとそのときの評価値を書き出す。
						string line = pos.sfen() + "," + to_string(value1) + "\n";
						sw.write(thread_id, line);
					}
#endif

#if 0
					// デバッグ用に局面と読み筋を表示させてみる。
					cout << pos;
					cout << "search() PV = ";
					for (auto pv_move : pv1)
						cout << pv_move << " ";
					cout << endl;

					// 静止探索のpvは存在しないことがある。(駒の取り合いがない場合など)　その場合は、現局面がPVのleafである。
					cout << "qsearch() PV = ";
					for (auto pv_move : pv2)
						cout << pv_move << " ";
					cout << endl;

#endif
				}

			SKIP_WRITE:;

				// 何故かPVが得られなかった(置換表などにhitして詰んでいた？)ので次に行く。
				if (pv1.size() == 0)
					break;
				
				// search_depth手読みの指し手で局面を進める。
				m = pv1[0];
			}

#ifdef      USE_SWAPPING_PIECES
			// 2駒をときどき入れ替える機能

			// このイベントは、王手がかかっていない局面において一定の確率でこの指し手が発生する。
			// 王手がかかっていると王手回避しないといけないので良くない。
			// 二枚とも歩が選ばれる可能性がそこそこあるため、1/5に設定しておく。
			// また、レアケースながら盤上に王しかいないケースがある。
			// これは、6駒以上という条件を入れておく。
			if (rand(5) == 0 && !pos.in_check() && pos.pieces(pos.side_to_move()).pop_count() >= 6)
			{
				for (int retry = 0; retry < 10; ++retry)
				{
					// 手番側の駒を2駒入れ替える。

					// 与えられたBitboardからランダムに1駒を選び、そのSquareを返す。
					auto get_one = [this](Bitboard pieces)
					{
						// 駒の数
						int num = pieces.pop_count();

						// 何番目かの駒
						int n = (int)rand(num) + 1;
						Square sq = SQ_NB;
						for (int i = 0; i < n; ++i)
							sq = pieces.pop();
						return sq;
					};

					// この升の2駒を入れ替える。
					auto pieces = pos.pieces(pos.side_to_move());

					auto sq1 = get_one(pieces);
					// sq1を除くbitboard
					auto sq2 = get_one(pieces ^ sq1);

					// sq2は王しかいない場合、SQ_NBになるから、これを調べておく
					// この指し手に成功したら、それはdo_moveの代わりであるから今回、do_move()は行わない。

					if (sq2 != SQ_NB
						&& pos.do_move_by_swapping_pieces(sq1, sq2))
					{
#if 0
						// 検証用のassert
						if (!is_ok(pos))
							cout << pos << sq1 << sq2;
#endif
						goto DO_MOVE_FINISH;
					}
				}
			}
#endif

		RANDOM_MOVE:;
#ifdef USE_RANDOM_LEGAL_MOVE

			// 合法手のなかからランダムに1手選ぶフェーズ
			// plyが小さいときは高い確率で。そのあとはあまり選ばれなくて良い。
			// 24手超えてランダムムーブを選ぶと角のただ捨てなどの指し手が入って終局してしまう。
			if (ply <= 16 && rand(4 + ply / 10) == 0)
			{
				// mateではないので合法手が1手はあるはず…。
				MoveList<LEGAL> list(pos);
				m = list.at((size_t)rand(list.size()));

				// 玉の2手指しのコードを入れていたが、合法手から1手選べばそれに相当するはずで
				// コードが複雑化するだけだから不要だと判断した。
			}
#endif

		DO_MOVE:;
			pos.do_move(m, state[ply]);

			// 差分計算を行なうために毎node evaluate()を呼び出しておく。
			Eval::evaluate(pos);
		}
	}
FINALIZE:;
	sw.finalize(thread_id);
}

// -----------------------------------
//    棋譜を生成するコマンド(master thread)
// -----------------------------------

// 棋譜を生成するコマンド
void gen_sfen(Position& pos, istringstream& is)
{
  // スレッド数(これは、USIのsetoptionで与えられる)
  u32 thread_num = Options["Threads"];

  // 生成棋譜の個数 default = 80億局面(Ponanza仕様)
  u64 loop_max = 8000000000UL;

  // 評価値がこの値になったら生成を打ち切る。
  int eval_limit = 2000;

  // 探索深さ
  int search_depth = 3;
  int search_depth2 = INT_MIN;

  // 書き出すファイル名
  string filename =
#ifdef WRITE_PACKED_SFEN
	  "generated_kifu.bin"
#else
	  "generated_kifu.sfen"
#endif
	  ;

  string token;
  while (true)
  {
	  token = "";
	  is >> token;
	  if (token == "")
		  break;

	  if (token == "depth")
		  is >> search_depth;
	  else if (token == "depth2")
		  is >> search_depth2;
	  else if (token == "loop")
		  is >> loop_max;
	  else if (token == "file")
		  is >> filename;
	  else if (token == "eval_limit")
		  is >> eval_limit;
	  else
		  cout << "Error! : Illegal token " << token << endl;
  }

  // search depth2が設定されていないなら、search depthと同じにしておく。
  if (search_depth2 == INT_MIN)
	  search_depth2 = search_depth;

  std::cout << "gen_sfen : "
	  << "search_depth = " << search_depth << " to " << search_depth2
	  << " , loop_max = " << loop_max
	  << " , eval_limit = " << eval_limit
	  << " , thread_num (set by USI setoption) = " << thread_num
	  << " , book_moves (set by USI setoption) = " << Options["BookMoves"]
	  << " , filename = " << filename
	  << endl;

  // Options["Threads"]の数だけスレッドを作って実行。
  {
	  SfenWriter sw(filename,thread_num);
	  MultiThinkGenSfen multi_think(search_depth, search_depth2, sw);
	  multi_think.set_loop_max(loop_max);
	  multi_think.eval_limit = eval_limit;
	  multi_think.start_file_write_worker();
	  multi_think.go_think();

	  // SfenWriterのデストラクタでjoinするので、joinが終わってから終了したというメッセージを
	  // 表示させるべきなのでここをブロックで囲む。
  }

  std::cout << "gen_sfen finished." << endl;
}

// -----------------------------------
// 生成した棋譜から学習させるコマンド(learn)
// -----------------------------------

// 普通のシグモイド関数
double sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

// 評価値を勝率[0,1]に変換する関数
double winning_percentage(Value value)
{
	// この600.0という定数は、ponanza定数。(ponanzaがそうしているらしいという意味で)
	// ゲームの進行度に合わせたものにしたほうがいいかも知れないけども、その効果のほどは不明。
	return sigmoid(static_cast<int>(value) / 600.0);
}

// 普通のシグモイド関数の導関数。
double dsigmoid(double x)
{
	// シグモイド関数
	//    f(x) = 1/(1+exp(-x))
	// に対して1階微分は、
	//    f'(x) = df/dx = f(x)・{ 1 - f(x) }
	// となる。

	return sigmoid(x) * (1.0 - sigmoid(x));
}

// 誤差を計算する関数(rmseの計算用)
// これは尺度が変わるといけないので目的関数を変更しても共通の計算にしておく。勝率の差の二乗和。
double calc_error(Value record_value, Value value)
{
	double diff = winning_percentage(value) - winning_percentage(record_value);
	return diff * diff;
}

// 目的関数が勝率の差の二乗和のとき
#ifdef LOSS_FUNCTION_IS_WINNING_PERCENTAGE
// 勾配を計算する関数
double calc_grad(Value deep, Value shallow)
{
	// 勝率の差の2乗が目的関数それを最小化する。
	// 目的関数 J = 1/2m Σ ( win_rate(shallow) - win_rate(deep) ) ^2
	// ただし、σはシグモイド関数で、評価値を勝率の差に変換するもの。
	// mはサンプルの件数。shallowは浅い探索(qsearch())のときの評価値。deepは深い探索のときの評価値。
	// また、Wを特徴ベクトル(評価関数のパラメーター)、Xi,Yiを教師とすると
	// shallow = W*Xi   // *はアダマール積で、Wの転置・X の意味
	// f(Xi) = win_rate(W*Xi)
	// σ(i番目のdeep) = Yi とおくと、
	// J = m/2 Σ ( f(Xi) - Yi )^2
	// とよくある式になる。
	// Wはベクトルで、j番目の要素をWjと書くとすると、連鎖律から
	// ∂J/∂Wj =            ∂J/∂f     ・  ∂f/∂W   ・ ∂W/∂Wj
	//          =  1/m Σ ( f(Xi) - y )  ・  f'(Xi)    ・    1

	// 1/mはあとで掛けるとして、勾配の値としてはΣの中身を配列に保持しておけば良い。
	// f'(Xi) = win_rate'(shallow) = sigmoid'(shallow/600) = dsigmoid(shallow / 600) / 600
	// この末尾の /600 は学習率で調整するから書かなくていいか..
	// また1/mという係数も、Adam , AdaGradのような勾配の自動調整機能を持つ更新式を用いるなら不要。
	// ゆえにメモリ上に保存しておく必要はない。

	double p = winning_percentage(deep);
	double q = winning_percentage(shallow);
	return (q - p) * dsigmoid(double(shallow) / 600.0);
}
#endif

#ifdef LOSS_FUNCTION_IS_CROSS_ENTOROPY
double calc_grad(Value deep, Value shallow)
{
	// 交差エントロピーを用いた目的関数

	// 交差エントロピーの概念と性質については、
	// http://nnadl-ja.github.io/nnadl_site_ja/chap3.html#the_cross-entropy_cost_function
	// http://postd.cc/visual-information-theory-3/
	// などを参考に。

	// 目的関数の設計)
	// pの分布をqの分布に近づけたい → pとqの確率分布間の交差エントロピーの最小化問題と考える。
	// J = H(p,q) = - Σ p(x) log(q(x)) = -p log q - (1-p) log(1-q)
	//                 x

	// pは定数、qはWiの関数(q = σ(W・Xi) )としてWiに対する偏微分を求める。
	// ∂J/∂Wi = -p・q'/q - (1-p)(1-q)'/(1-q)
	//          = ...
	//          = q - p.

	double p = winning_percentage(deep);
	double q = winning_percentage(shallow);

	return q - p;
}
#endif

#ifdef LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE
double calc_grad(Value deep, Value shallow)
{
	// 勝率の関数を通さない版
	// これ、EVAL_LIMITを低くしておかないと、終盤の形に対して評価値を一致させようとして
	// evalがevalの範囲を超えかねない。
	return shallow - deep;
}
#endif

// 目的関数として他のバリエーションも色々用意するかも..


// Sfenの読み込み機
struct SfenReader
{
	SfenReader(int thread_num)
	{
		packed_sfens.resize(thread_num);
		total_read = 0;
		total_done = 0;
		next_update_weights = 0;
		save_count = 0;
		end_of_files = false;

		// 比較実験がしたいので乱数を固定化しておく。
		prng = PRNG(20160720);

		hash.resize(READ_SFEN_HASH_SIZE);
	}
	~SfenReader()
	{
		file_worker_thread.join();
	}

	// mseの計算用に1万局面ほど読み込んでおく。
	void read_for_mse()
	{
		Position& pos = Threads.main()->rootPos;
		for (int i = 0; i < 10000; ++i)
		{
			PackedSfenValue ps;
			if (!read_to_thread_buffer(0, ps))
			{
				cout << "Error! read packed sfen , failed." << endl;
				break;
			}
			sfen_for_mse.push_back(ps);

			// hash keyを求める。
			pos.set_from_packed_sfen(ps.sfen);
			sfen_for_mse_hash.insert(pos.key());
		}
	}


	// 各スレッドがバッファリングしている局面数 0.1M局面。40HTで4M局面
	const size_t THREAD_BUFFER_SIZE = 10 * 1000;

	// ファイル読み込み用のバッファ(これ大きくしたほうが局面がshuffleが大きくなるので局面がバラけていいと思うが
	// あまり大きいとメモリ消費量も上がる。
	// SFEN_READ_SIZEはTHREAD_BUFFER_SIZEの倍数であるものとする。
	const size_t SFEN_READ_SIZE = LEARN_SFEN_READ_SIZE;

	// [ASYNC] スレッドが局面を一つ返す。なければfalseが返る。
	bool read_to_thread_buffer(size_t thread_id, PackedSfenValue& ps)
	{
		// スレッドバッファに局面が残っているなら、それを1つ取り出して返す。
		auto& thread_ps = packed_sfens[thread_id];

		// バッファに残りがなかったらread bufferから充填するが、それすらなかったらもう終了。
		if ((thread_ps == nullptr || thread_ps->size() == 0) // バッファが空なら充填する。
			&& !read_to_thread_buffer_impl(thread_id))
			return false;

		// read_to_thread_buffer_impl()がtrueを返したというこは、
		// スレッドバッファへの局面の充填が無事完了したということなので
		// thread_ps->rbegin()は健在。

		ps = *(thread_ps->rbegin());
		thread_ps->pop_back();

		return true;
	}

	// rmseを計算して表示する。
	void calc_rmse()
	{
		// 置換表にhitされてもかなわんので、このタイミングで置換表の世代を新しくする。
		// 置換表を無効にしているなら関係ないのだが。
		TT.new_search();

		// thread_idは0に固定。(main threadで行わせるため)
		const int thread_id = 0;
		auto& pos = Threads[thread_id]->rootPos;

		double sum_error = 0;
		double sum_error2 = 0;

		int i = 0;
		for (auto& ps : sfen_for_mse)
		{
//			auto sfen = pos.sfen_unpack(ps.data);
//			pos.set(sfen);

			pos.set_from_packed_sfen(ps.sfen);

			auto th = Threads[thread_id];
			pos.set_this_thread(th);

			// 浅い探索の評価値

#ifdef USE_QSEARCH_FOR_SHALLOW_VALUE
			const int depth = 0; // qsearch()相当
#endif
#ifdef USE_EVALUATE_FOR_SHALLOW_VALUE
			const int depth = -1; // evaluate()相当
#endif
			auto r = Learner::search(pos,depth);
			auto shallow_value = r.first;

			// これPVに行ってleaf nodeで、w(floatで計算されている)に基いて
			// eval()の値を計算する評価関数を呼び出したほうが正確だと思う。
			
			// 深い探索の評価値
			auto deep_value = (Value)ps.score;

			// 誤差の計算
			sum_error += calc_error(shallow_value, deep_value);
			sum_error2 += abs(shallow_value - deep_value);

#if 0
			{
				// 検証用にlogを書き出してみる。
				static fstream log;
				if (!log.is_open())
					log.open("rmse_log.txt", ios::out);
				log << this->total_done << ": [" << i++ << "]" << " = " << shallow_value << " , " << deep_value << endl;
			}
#endif
		}

		auto rmse = std::sqrt(sum_error / sfen_for_mse.size());
		auto mean_error = sum_error2 / sfen_for_mse.size();
		cout << endl << "rmse = " << rmse << " , mean_error = " << mean_error << endl;
	}

	// [ASYNC] スレッドバッファに局面を10000局面ほど読み込む。
	bool read_to_thread_buffer_impl(size_t thread_id)
	{
		while (true)
		{
			{
				std::unique_lock<Mutex> lk(mutex);
				// ファイルバッファから充填できたなら、それで良し。
				if (packed_sfens_pool.size() != 0)
				{
					// 充填可能なようなので充填して終了。

					packed_sfens[thread_id] = *packed_sfens_pool.rbegin();
					packed_sfens_pool.pop_back();

					total_read += THREAD_BUFFER_SIZE;

					return true;
				}
			}

			// もうすでに読み込むファイルは無くなっている。もうダメぽ。
			if (end_of_files)
				return false;

			// file workerがpacked_sfens_poolに充填してくれるのを待っている。
			// mutexはlockしていないのでいずれ充填してくれるはずだ。
			sleep(1);
		}

	}
	
	// 局面ファイルをバックグラウンドで読み込むスレッドを起動する。
	void start_file_read_worker()
	{
		file_worker_thread = std::thread([&] { this->file_read_worker(); });
	}

	// ファイルの読み込み専用スレッド用
	void file_read_worker()
	{
		auto open_next_file = [&]()
		{
			if (fs.is_open())
				fs.close();

			// もう無い
			if (filenames.size() == 0)
				return false;

			// 次のファイル名ひとつ取得。
			string filename = *filenames.rbegin();
			filenames.pop_back();

			fs.open(filename, ios::in | ios::binary);
			cout << endl << "open filename = " << filename << " ";

#if 0
			// 棋譜の先頭2M局面ほど、探索の初期化が十分なされていないせいか、
			// あまりいい探索結果ではないので、これを捨てる。
			fs.seekp(sizeof(PackedSfenValue) * 2*1000*1000,ios::beg);
#endif

			return true;
		};

		while (true)
		{
			// バッファが減ってくるのを待つ。
			// このsize()の読み取りはread onlyなのでlockしなくていいだろう。
			while (packed_sfens_pool.size() >= SFEN_READ_SIZE / THREAD_BUFFER_SIZE)
				sleep(100);

			vector<PackedSfenValue> sfens;
			sfens.reserve(SFEN_READ_SIZE);

			// ファイルバッファにファイルから読み込む。
			while (sfens.size() < SFEN_READ_SIZE)
			{
				PackedSfenValue p;
				if (fs.read((char*)&p, sizeof(PackedSfenValue)))
				{
					sfens.push_back(p);
				} else
				{
					// 読み込み失敗
					if (!open_next_file())
					{
						// 次のファイルもなかった。あぼーん。
						cout << "..end of files.\n";
						end_of_files = true;
						return;
					}
				}
			}

#ifndef LEARN_SFEN_NO_SHUFFLE
			// この読み込んだ局面データをshuffleする。
			// random shuffle by Fisher-Yates algorithm
			{
				auto size = sfens.size();
				for (size_t i = 0; i < size; ++i)
					swap(sfens[i], sfens[(size_t)(prng.rand(size - i) + i)]);
			}
#endif

			// これをTHREAD_BUFFER_SIZEごとの細切れにする。それがsize個あるはず。
			// SFEN_READ_SIZEはTHREAD_BUFFER_SIZEの倍数であるものとする。
			ASSERT_LV3((SFEN_READ_SIZE % THREAD_BUFFER_SIZE)==0);

			auto size = size_t(SFEN_READ_SIZE / THREAD_BUFFER_SIZE);
			vector<shared_ptr<vector<PackedSfenValue>>> ptrs;
			ptrs.reserve(size);

			for (size_t i = 0; i < size; ++i)
			{
				shared_ptr<vector<PackedSfenValue>> ptr(new vector<PackedSfenValue>());
				ptr->resize(THREAD_BUFFER_SIZE);
				memcpy(&((*ptr)[0]), &sfens[i * THREAD_BUFFER_SIZE], sizeof(PackedSfenValue) * THREAD_BUFFER_SIZE);

				ptrs.push_back(ptr);
			}

			// sfensの用意が出来たので、折を見てコピー
			{
				std::unique_lock<Mutex> lk(mutex);

				// shared_ptrをコピーするだけなのでこの時間は無視できるはず…。
				// packed_sfens_poolの内容を変更するのでmutexのlockが必要。

				for (size_t i = 0; i < size; ++i)
					packed_sfens_pool.push_back(ptrs[i]);

				// mutexをlockしている間にshared_ptrのデストラクタを呼び出さないと
				// 参照カウントを複数スレッドから変更することになってまずい。
				ptrs.clear();
			}
		}
	}

	// sfenファイル群
	vector<string> filenames;

	// 読み込んだ局面数(ファイルからメモリ上のバッファへ)
	atomic<u64> total_read;

	// 処理した局面数
	atomic<u64> total_done;

	// total_readがこの値を超えたらupdate_weights()してmseの計算をする。
	u64 next_update_weights;

	u64 save_count;

	// rmseの計算用の局面であるかどうかを判定する。
	// (rmseの計算用の局面は学習のために使うべきではない。)
	bool is_for_rmse(Key key) const
	{
		return sfen_for_mse_hash.count(key) != 0;
	}

	// 同一局面の読み出しを制限するためのhash
	// 6400万局面って多すぎるか？そうでもないか..
	// hash_indexを求めるためのmaskに使うので、2**Nでなければならない。
	static const u64 READ_SFEN_HASH_SIZE = 64 * 1024 * 1024;
	vector<HASH_KEY> hash; // 64MB*8 = 512MB

protected:

	// fileをバックグラウンドで読み込みしているworker thread
	std::thread file_worker_thread;

	// 局面の読み込み時にshuffleするための乱数
	PRNG prng;

	// ファイル群を読み込んでいき、最後まで到達したか。
	atomic<bool> end_of_files;


	// sfenファイルのハンドル
	fstream fs;

	// 各スレッド用のsfen
	vector<shared_ptr<vector<PackedSfenValue>>> packed_sfens;

	// packed_sfens_poolにアクセスするときのmutex
	Mutex mutex;

	// sfenのpool。fileから読み込むworker threadはここに補充する。
	// 各worker threadはここから自分のpacked_sfens[thread_id]に充填する。
	// ※　mutexをlockしてアクセスすること。
	vector<shared_ptr<vector<PackedSfenValue>>> packed_sfens_pool;

	// mse計算用のバッファ
	vector<PackedSfenValue> sfen_for_mse;
	// mse計算用の局面を学習に用いないためにhash keyを保持しておく。
	std::unordered_set<Key> sfen_for_mse_hash;
};


// 複数スレッドでsfenを生成するためのクラス
struct LearnerThink: public MultiThink
{
	LearnerThink(SfenReader& sr_):sr(sr_),updating_weight(false) {}
	virtual void thread_worker(size_t thread_id);

	// 局面ファイルをバックグラウンドで読み込むスレッドを起動する。
	void start_file_read_worker() { sr.start_file_read_worker(); }

	// 評価関数パラメーターをファイルに保存
	void save();

	// sfenの読み出し器
	SfenReader& sr;

	// 学習の反復回数のカウンター
	u64 epoch = 0;

	// ミニバッチサイズのサイズ。必ずこのclassを使う側で設定すること。
	u64 mini_batch_size = 1000*1000;

	// weightのupdate中であるか。(このとき、sleepする)
	// あるいはsleepさせたいときはこれをTrueにする。
	atomic<bool> updating_weight;
};

void LearnerThink::thread_worker(size_t thread_id)
{
	auto& pos = Threads[thread_id]->rootPos;

	while (true)
	{
		// mseの表示(これはthread 0のみときどき行う)
		// ファイルから読み込んだ直後とかでいいような…。
		if (thread_id == 0 && sr.next_update_weights <= sr.total_done)
		{
			// 一応、他のスレッド停止させる。
			updating_weight = true;

			// 現在時刻を出力
			static u64 sfens_output_count = 0;
			if ((sfens_output_count++ % LEARN_TIMESTAMP_OUTPUT_INTERVAL) == 0)
			{
				auto now = std::chrono::system_clock::now();
				auto tp = std::chrono::system_clock::to_time_t(now);
				cout << endl << sr.total_done << " sfens , at " << std::ctime(&tp);
			} else {
				// これぐらいは出力しておく。
				cout << '.' << flush;
			}

			// このタイミングで勾配をweight配列に反映。勾配の計算も1M局面ごとでmini-batch的にはちょうどいいのでは。

			Eval::update_weights(mini_batch_size, ++epoch);

			// 8000万局面ごとに1回保存、ぐらいの感じで。

			// ただし、update_weights(),calc_rmse()している間の時間経過は無視するものとする。
			if (++sr.save_count * mini_batch_size >= LEARN_EVAL_SAVE_INTERVAL)
			{
				sr.save_count = 0;

				// この間、gradientの計算が進むと値が大きくなりすぎて困る気がするので他のスレッドを停止させる。
				save();
			}

			// rmseを計算する。1万局面のサンプルに対して行う。
			// 40コアでやると100万局面ごとにupdate_weightsするとして、特定のスレッドが
			// つきっきりになってしまうのあまりよくないような気も…。
			static u64 rmse_output_count = 0;
			if ((++rmse_output_count % LEARN_RMSE_OUTPUT_INTERVAL) == 0)
			{
				// この計算自体も並列化すべきのような…。
				// この計算をしているときにあまり処理が進みすぎると困るので停止させておくか…。
				sr.calc_rmse();
			}

			// 次回、この一連の処理は、
			// total_read + LEARN_MINI_BATCH_SIZE <= total_read
			// となったときにやって欲しい。
			sr.next_update_weights = sr.total_done + mini_batch_size;

			// 他のスレッド再開。
			updating_weight = false;
		}

		PackedSfenValue ps;
	RetryRead:;
		if (!sr.read_to_thread_buffer(thread_id, ps))
			break;
		
#if 0
		auto sfen = pos.sfen_unpack(ps.data);
		pos.set(sfen);
#endif
		// ↑sfenを経由すると遅いので専用の関数を作った。
		pos.set_from_packed_sfen(ps.sfen);
		{
			auto key = pos.key();
			// rmseの計算用に使っている局面なら除外する。
			if (sr.is_for_rmse(key))
				goto RetryRead;

			// 直近で用いた局面も除外する。
			auto hash_index = size_t(key & (sr.READ_SFEN_HASH_SIZE - 1));
			auto key2 = sr.hash[hash_index];
			if (key == key2)
				goto RetryRead;
			sr.hash[hash_index] = key; // 今回のkeyに入れ替えておく。
		}

		// このインクリメントはatomic
		sr.total_done++;

		auto th = Threads[thread_id];
		pos.set_this_thread(th);

		// 読み込めたので試しに表示してみる。
		//		cout << pos << value << endl;

		// 浅い探索(qsearch)の評価値
#ifdef USE_QSEARCH_FOR_SHALLOW_VALUE
		auto r = Learner::qsearch(pos);
		// 置換表を無効化しているのでPV leafでevaluate()を呼び出したときの値と同じはず..
		// (詰みのスコアでないなら)
		auto shallow_value = r.first;
#endif
#ifdef USE_EVALUATE_FOR_SHALLOW_VALUE
		auto shallow_value = Eval::evaluate(pos);
#endif

		// qsearchではなくevaluate()の値をそのまま使う場合。
		//			auto shallow_value = Eval::evaluate(pos);

		// 深い探索の評価値
		auto deep_value = (Value)ps.score;

		// 勾配
		double dj_dw = calc_grad(deep_value, shallow_value);
		
		// 現在、leaf nodeで出現している特徴ベクトルに対する勾配(∂J/∂Wj)として、jd_dwを加算する。

		// mini batchのほうが勾配が出ていいような気がする。
		// このままleaf nodeに行って、勾配配列にだけ足しておき、あとでrmseの集計のときにAdaGradしてみる。

		auto rootColor = pos.side_to_move();

#ifdef		USE_QSEARCH_FOR_SHALLOW_VALUE

		auto pv = r.second;

		// PVの初手が異なる場合は学習に用いないほうが良いのでは…。
		// 全然違うところを探索した結果だとそれがノイズに成りかねない。
		// 評価値の差が大きすぎるところも学習対象としないほうがいいかも…。
#ifdef GENSFEN_SAVE_FIRST_MOVE

#if 0
		// これやると13%程度の局面が学習対象から外れてしまう。善悪は微妙。
		if (pv.size() >= 1 && (u16)pv[0] != ps.move)
		{
//			dbg_hit_on(false);
			continue;
		}
#endif

#if 0
		// 評価値の差が大きすぎるところも学習対象としないほうがいいかも…。
		// →　勝率の関数を通すのでまあいいか…。30%ぐらいの局面が学習対象から外れてしまうしな…。
		if (abs((s16)r.first - ps.score) >= Eval::PawnValue * 4)
		{
//			dbg_hit_on(false);
			continue;
		}
#endif

//		dbg_hit_on(true);

#endif

		int ply = 0;
		StateInfo state[MAX_PLY]; // qsearchのPVがそんなに長くなることはありえない。
		for (auto m : pv)
		{
			// 非合法手はやってこないはずなのだが。
			if (!pos.pseudo_legal(m) || !pos.legal(m))
			{
				cout << pos << m << endl;
				ASSERT_LV3(false);
			}
			pos.do_move(m, state[ply++]);
		}

		// leafに到達
		Eval::add_grad(pos,rootColor,dj_dw);

		// 局面を巻き戻す
		for (auto it = pv.rbegin(); it != pv.rend(); ++it)
			pos.undo_move(*it);
#endif

#ifdef USE_EVALUATE_FOR_SHALLOW_VALUE
		// 現局面でevaluate()するので現局面がleafだと考えられる。
		Eval::add_grad(pos, rootColor, dj_dw);
#endif

		// weightの更新中であれば、そこはparallel forで回るので、こっちのスレッドは中断しておく。
		// 保存のときに回られるのも勾配だけが更新され続けるので良くない。
		while (updating_weight)
			sleep(0);

	}
}

void LearnerThink::save()
{
	// 定期的に保存
	// 10億局面ごとにファイル名の拡張子部分を"0","1","2",..のように変えていく。
	// (あとでそれぞれの評価関数パラメーターにおいて勝率を比較したいため)
#ifndef EVAL_SAVE_ONLY_ONCE
	u64 change_name_size = (u64)EVAL_FILE_NAME_CHANGE_INTERVAL;
	Eval::save_eval(std::to_string(sr.total_read / change_name_size));

	// sr.total_readは、処理した件数ではないので、ちょっとオーバーしている可能性はある。

#else
	// 1度だけの保存のときはサブフォルダを掘らない。
	Eval::save_eval("");
#endif
}

// 生成した棋譜からの学習
void learn(Position& pos, istringstream& is)
{
	auto thread_num = (int)Options["Threads"];
	SfenReader sr(thread_num);

	LearnerThink learn_think(sr);
	vector<string> filenames;

	// mini_batch_size デフォルトで1M局面。これを大きくできる。
	auto mini_batch_size = LEARN_MINI_BATCH_SIZE;

	// ループ回数(この回数だけ棋譜ファイルを読み込む)
	int loop = 1;

	// 棋譜ファイル格納フォルダ(ここから相対pathで棋譜ファイルを取得)
	string base_dir;

	string target_dir;
	
	float eta = 0.0f;

	// ファイル名が後ろにずらずらと書かれていると仮定している。
	while (true)
	{
		string option;
		is >> option;

		if (option == "")
			break;

		// mini-batchの局面数を指定
		if (option == "bat")
		{
			is >> mini_batch_size;
			mini_batch_size *= 10000; // 単位は万

		}

		// 棋譜が格納されているフォルダを指定して、根こそぎ対象とする。
		else if (option == "targetdir")
		{
			is >> target_dir;

#if !defined(_MSC_VER)
			cout << "ERROR! : targetdir , this function is only for Windows." << endl;
#endif
		}

		// ループ回数の指定
		else if (option == "loop")      is >> loop;

		// 棋譜ファイル格納フォルダ(ここから相対pathで棋譜ファイルを取得)
		else if (option == "basedir")   is >> base_dir;

		// ミニバッチのサイズ
		else if (option == "batchsize") is >> mini_batch_size;

		// 学習率
		else if (option == "eta")       is >> eta;

		// さもなくば、それはファイル名である。
		else
			filenames.push_back(option);
	}

	cout << "learn command , ";

	// OpenMP無効なら警告を出すように。
#ifndef _OPENMP
	cout << "Warning! OpenMP disabled." << endl;
#endif

#if defined(_MSC_VER)
	// ディレクトリ操作がWindows専用っぽいので…。

	// 学習棋譜ファイルの表示
	if (target_dir != "")
	{
		// このフォルダを根こそぎ取る。base_dir相対にしておく。
		namespace sys = std::tr2::sys;
		string kif_base_dir = path_combine(base_dir, target_dir);
		sys::path p(kif_base_dir); // 列挙の起点
		std::for_each(sys::directory_iterator(p), sys::directory_iterator(),
			[&](const sys::path& p) {
			if (sys::is_regular_file(p))
				filenames.push_back(path_combine(target_dir , p.filename().generic_string()));
		});
	}
#endif

	cout << "learn from ";
	for (auto s : filenames)
		cout << s << " , ";

	cout << "\nbase dir        : " << base_dir;
	cout << "\ntarget dir      : " << target_dir;
	cout << "\nloop            : " << loop;

	// ループ回数分だけファイル名を突っ込む。
	for (int i = 0; i < loop; ++i)
		// sfen reader、逆順で読むからここでreverseしておく。すまんな。
		for (auto it = filenames.rbegin(); it != filenames.rend(); ++it)
			sr.filenames.push_back(path_combine(base_dir, *it));

	cout << "\nGradient Method : " << LEARN_UPDATE;
	cout << "\nLoss Function   : " << LOSS_FUNCTION;
	cout << "\nmini-batch size : " << mini_batch_size;
	cout << "\nlearning rate   : " << eta;

	// -----------------------------------
	//            各種初期化
	// -----------------------------------

	cout << "\ninit..";

	// 評価関数パラメーターの読み込み
	is_ready();

	cout << "\ninit_grad..";

	// 評価関数パラメーターの勾配配列の初期化
	Eval::init_grad(eta);

#ifdef _OPENMP
	omp_set_num_threads((int)Options["Threads"]);
#endif

	cout << "\ninit done." << endl;

	// 局面ファイルをバックグラウンドで読み込むスレッドを起動
	// (これを開始しないとmseの計算が出来ない。)
	learn_think.start_file_read_worker();

	learn_think.mini_batch_size = mini_batch_size;

	// mse計算用にデータ1万件ほど取得しておく。
	sr.read_for_mse();

#ifdef LEARN_UPDATE_EVERYTIME
	// 毎回updateするなら学習率の初期化のために最初に呼び出しておく必要がある。
	Eval::update_weights(mini_batch_size, 1);
#endif

	// この時点で一度rmseを計算(0 sfenのタイミング)
	// sr.calc_rmse();

	// -----------------------------------
	//   評価関数パラメーターの学習の開始
	// -----------------------------------

	// 学習開始。
	learn_think.go_think();

	// 最後に一度保存。
	learn_think.save();
}


} // namespace Learner

#endif // EVAL_LEARN
