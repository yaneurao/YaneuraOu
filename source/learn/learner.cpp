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

// ----------------------
// 学習時に関する、あまり大した意味のない設定項目はここ。
// ----------------------

// タイムスタンプの出力をこの回数に一回に抑制する。
// スレッドを論理コアの最大数まで酷使するとコンソールが詰まるので調整用。
#define GEN_SFENS_TIMESTAMP_OUTPUT_INTERVAL 1

// 学習時にsfenファイルを1万局面読み込むごとに'.'を出力する。
//#define DISPLAY_STATS_IN_THREAD_READ_SFENS

// ----------------------
// 設定内容に基づく定数文字列
// ----------------------

// 更新式に応じた文字列。(デバッグ用に出力する。)
// 色々更新式を実装したがAdaGradが速度面、メモリ面においてベストという結論になった。
#if defined(ADA_GRAD_UPDATE)
#define LEARN_UPDATE "AdaGrad"
#elif defined(SGD_UPDATE)
#define LEARN_UPDATE "SGD"
#endif

#if defined(LOSS_FUNCTION_IS_WINNING_PERCENTAGE)
#define LOSS_FUNCTION "WINNING_PERCENTAGE"
#elif defined(LOSS_FUNCTION_IS_CROSS_ENTOROPY)
#define LOSS_FUNCTION "CROSS_ENTOROPY"
#elif defined(LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE)
#define LOSS_FUNCTION "CROSS_ENTOROPY_FOR_VALUE"
#elif defined(LOSS_FUNCTION_IS_ELMO_METHOD)
#define LOSS_FUNCTION "ELMO_METHOD(WCSC27)"
#elif defined(LOSS_FUNCTION_IS_YANE_ELMO_METHOD)
#define LOSS_FUNCTION "YANE_ELMO_METHOD(WCSC27)"
#endif

// -----------------------------------
//    以下、実装部。
// -----------------------------------


#include <sstream>
#include <fstream>
#include <unordered_set>

#include "../misc.h"
#include "../thread.h"
#include "../position.h"
#include "../extra/book/book.h"
#include "../tt.h"
#include "multi_think.h"

#if defined(_MSC_VER)
// C++のfilesystemは、C++17以降か、MSVCでないと使えないようだ。
// windows.hを使うようにしたが、msys2のg++だとうまくフォルダ内のファイルが取得できない。
// 仕方ないのでdirent.hを用いる。
#include <filesystem>
#elif defined(__GNUC__)
#include <dirent.h>
#endif


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// これは探索部で定義されているものとする。
extern Book::BookMoveSelector book;
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
		fs.close();
	}

	// 各スレッドについて、この局面数ごとにファイルにflushする。
	const size_t SFEN_WRITE_SIZE = 5000;

	// 局面と評価値をペアにして1つ書き出す(packされたsfen形式で)
	void write(size_t thread_id, const PackedSfenValue& psv)
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

		// スレッドごとに用意されており、一つのスレッドが同時にこのwrite()関数を呼び出さないので
		// この時点では排他する必要はない。
		buf->push_back(psv);

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

	// バッファに残っている分をファイルに書き出す。
	void finalize(size_t thread_id)
	{
		std::unique_lock<Mutex> lk(mutex);

		auto& buf = sfen_buffers[thread_id];

		// buf==nullptrであるケースもあるのでそのチェックが必要。
		if (buf && buf->size() != 0)
			sfen_buffers_pool.push_back(buf);

		buf = nullptr;
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
			cout << endl << sfen_write_count << " sfens , at " << now_string();

			// flush()はこのタイミングで十分。
			fs.flush();
		};

		while (!finished || sfen_buffers_pool.size())
		{
			vector<shared_ptr<vector<PackedSfenValue>>> buffers;
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
					fs.write((const char*)&((*ptr)[0]), sizeof(PackedSfenValue) * ptr->size());

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
	vector<shared_ptr<vector<PackedSfenValue>>> sfen_buffers;
	vector<shared_ptr<vector<PackedSfenValue>>> sfen_buffers_pool;

	// sfen_buffers_poolにアクセスするときに必要なmutex
	Mutex mutex;

	// 書きだした局面の数
	u64 sfen_write_count = 0;

};

// -----------------------------------
//  棋譜を生成するworker(スレッドごと)
// -----------------------------------

// 複数スレッドでsfenを生成するためのクラス
struct MultiThinkGenSfen : public MultiThink
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

	vector<Key> hash; // 64MB*sizeof(HASH_KEY) = 512MB
};

//  thread_id    = 0..Threads.size()-1
void MultiThinkGenSfen::thread_worker(size_t thread_id)
{
	const int MAX_PLY2 = 400;

	StateInfo state[MAX_PLY2 + 20]; // StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
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

		// 探索部で定義されているBookMoveSelectorのメンバを参照する。
		auto& book = ::book.memory_book;

		// 1局分の局面を保存しておき、終局のときに勝敗を含めて書き出す。
		// 書き出す関数は、この下にあるflush_psv()である。
		vector<PackedSfenValue> a_psv;

		// a_psvに積まれている局面をファイルに書き出す。
		// lastTurnIsWin : a_psvに積まれている最終局面の次の局面での勝敗
		// 返し値 : もう書き出せないので終了する場合にtrue。
		auto flush_psv = [&](bool lastTurnIsWin)
		{
			bool isWin = lastTurnIsWin;

			// 終局の局面(の一つ前)から初手に向けて、各局面に関して、対局の勝敗の情報を付与しておく。
			// a_psvに保存されている局面は(手番的に)連続しているものとする。
			for (auto it = a_psv.rbegin(); it != a_psv.rend(); ++it)
			{
				isWin = !isWin;
				it->isWin = isWin;

				// 局面を書き出そうと思ったら規定回数に達していた。
				// get_next_loop_count()内でカウンターを加算するので
				// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
				auto loop_count = get_next_loop_count();
				if (loop_count == UINT64_MAX)
					return true;

				// 100k局面に1回ぐらい置換表の世代を進める。
				if ((loop_count % 100000) == 0)
					TT.new_search();

				// 局面を一つ書き出す。
				sw.write(thread_id, *it);

#if 0
				pos.set_from_packed_sfen(it->sfen);
				cout << pos << "Win : " << it->isWin << " , " << it->score << endl;
#endif
			}
			return false;
		};


		// ply : 初期局面からの手数
		for (int ply = 0; ply < MAX_PLY2 ; ++ply)
		{
			// 詰んでいるなら次の対局に
			// 基本的にはここに来る前にsearch()でeval_limitを超えたスコアが返ってくるはず。
			if (pos.is_mated())
			{
				// この局面では負けなので引数はfalseを渡す
				if (flush_psv(false))
					goto FINALIZE;
				break;
			}

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
				// これをもって勝敗がついたという扱いをする。
				if (abs(value1) >= eval_limit)
				{
//					sync_cout << pos << "eval limit = " << eval_limit << " over , move = " << pv1[0] << sync_endl;

					// この局面でvalue1 >= eval_limitならば、(この局面の手番側の)勝ちである。
					if (flush_psv(value1 >= eval_limit))
						goto FINALIZE;
					break;
				}

				// 何らかの千日手局面に突入したので局面生成を終了する。
				auto draw_type = pos.is_repetition();
				if (draw_type != REPETITION_NONE)
				{
					// 引き分けのときは、局面を書き出さない。
					// (勝敗情報が書き出せないため。)
//					sync_cout << pos << "repetition , move = " << pv1[0] << sync_endl;
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
				auto evaluate_leaf = [&](Position& pos , vector<Move>& pv)
				{
					auto rootColor = pos.side_to_move();

					int ply2 = ply;
					for (auto m : pv)
					{
						// デバッグ用の検証として、途中に非合法手が存在しないことを確認する。
						// NULL_MOVEはこないものとする。

						// 十分にテストしたのでコメントアウトで良い。
#if 0
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
				//	cout << pv_value1.first << " , " << leaf_value << endl;
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
				//	dbg_hit_on(pv1.size() >= search_depth);
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
				{
					a_psv.clear();
					goto SKIP_SAVE;
				}

				// 同一局面を書き出したところか？
				// これ、複数のPCで並列して生成していると同じ局面が含まれることがあるので
				// 読み込みのときにも同様の処理をしたほうが良い。
				{
					auto key = pos.key();
					auto hash_index = (size_t)(key & (GENSFEN_HASH_SIZE - 1));
					auto key2 = hash[hash_index];
					if (key == key2)
					{
						// スキップするときはこれ以前に関する
						// 勝敗の情報がおかしくなるので保存している局面をクリアする。
						// どのみち、hashが合致した時点でそこ以前の局面も合致している可能性が高いから
						// 書き出す価値がない。
						a_psv.clear();
						goto SKIP_SAVE;
					}
					hash[hash_index] = key; // 今回のkeyに入れ替えておく。
				}

				// 局面の一時保存。
				{
					a_psv.emplace_back(PackedSfenValue());
					auto &psv = a_psv.back();
					
					// packを要求されているならpackされたsfenとそのときの評価値を書き出す。
					// 最終的な書き出しは、勝敗がついてから。
					pos.sfen_pack(psv.sfen);
					psv.score = leaf_value;
					psv.gamePly = ply;

					// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
					ASSERT_LV3(pv_value1.second.size() >= 1);
					Move pv_move1 = pv_value1.second[0];
					psv.move = pv_move1;
				}

			SKIP_SAVE:;

				// 何故かPVが得られなかった(置換表などにhitして詰んでいた？)ので次に行く。
				if (pv1.size() == 0)
					break;
				
				// search_depth手読みの指し手で局面を進める。
				m = pv1[0];
			}

		RANDOM_MOVE:;

//#if defined ( USE_RANDOM_LEGAL_MOVE )
			// これは効果があるので、常に有効で良い。
#if 1

			// 合法手のなかからランダムに1手選ぶフェーズ
			// plyが小さいときは高い確率で。そのあとはあまり選ばれなくて良い。
			// 24手超えてrandom moveを選ぶと角のただ捨てなどの指し手が入って終局してしまう。
			// まあ、それは書き出さないからいいか…。
			// 20手目までに5回ぐらいrandom moveしてくれればいいか。
			if (ply <= 20 && rand(4) == 0)
			{
				// mateではないので合法手が1手はあるはず…。
				MoveList<LEGAL> list(pos);
				m = list.at((size_t)rand(list.size()));

				// 玉の2手指しのコードを入れていたが、合法手から1手選べばそれに相当するはずで
				// コードが複雑化するだけだから不要だと判断した。

				// ゲームの勝敗から指し手を評価しようとするとき、
				// 今回のrandom moveがあるので、ここ以前には及ばないようにする。
				a_psv.clear(); // 保存していた局面のクリア
			}
#endif

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
void gen_sfen(Position&, istringstream& is)
{
	// スレッド数(これは、USIのsetoptionで与えられる)
	u32 thread_num = Options["Threads"];

	// 生成棋譜の個数 default = 80億局面(Ponanza仕様)
	u64 loop_max = 8000000000UL;

	// 評価値がこの値になったら生成を打ち切る。
	int eval_limit = 3000;

	// 探索深さ
	int search_depth = 3;
	int search_depth2 = INT_MIN;

	// 書き出すファイル名
	string filename = "generated_kifu.bin";

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
		SfenWriter sw(filename, thread_num);
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
double winning_percentage(double value)
{
	// この600.0という定数は、ponanza定数。(ponanzaがそうしているらしいという意味で)
	// ゲームの進行度に合わせたものにしたほうがいいかも知れないけども、その効果のほどは不明。
	return sigmoid(value / 600.0);
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

// 目的関数が勝率の差の二乗和のとき
#if defined (LOSS_FUNCTION_IS_WINNING_PERCENTAGE)
// 勾配を計算する関数
double calc_grad(Value deep, Value shallow, PackedSfenValue& psv)
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

#if defined (LOSS_FUNCTION_IS_CROSS_ENTOROPY)
double calc_grad(Value deep, Value shallow, PackedSfenValue& psv)
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

#if defined ( LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE )
double calc_grad(Value deep, Value shallow , PackedSfenValue& psv)
{
	// 勝率の関数を通さない版
	// これ、EVAL_LIMITを低くしておかないと、終盤の形に対して評価値を一致させようとして
	// evalがevalの範囲を超えかねない。
	return shallow - deep;
}
#endif

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )

// elmo(WCSC27)で使われている定数。要調整。
// elmoのほうは式を内分していないので値が違う。
// learnコマンドでこの値を設定できる。
double ELMO_LAMBDA = 0.33; // elmoの定数(0.5)相当

// isWin : この手番側が最終的に勝利したかどうか
double calc_grad(Value deep, Value shallow , const PackedSfenValue& psv)
{
	// elmo(WCSC27)方式
	// 実際のゲームの勝敗で補正する。

	const double eval_winrate = winning_percentage(shallow);
	const double teacher_winrate = winning_percentage(deep);

	// 期待勝率を勝っていれば1、負けていれば 0として補正項として用いる。
	const double t = (psv.isWin) ? 1.0 : 0.0;

	// 実際の勝率を補正項として使っている。
	// これがelmo(WCSC27)のアイデアで、現代のオーパーツ。
	const double dsig = (1 - ELMO_LAMBDA) * (eval_winrate - t) + ELMO_LAMBDA * (eval_winrate - teacher_winrate);

	return dsig;
}

// 学習時の交差エントロピーの計算
// elmo式の勝敗項と勝率項との個別の交差エントロピーが引数であるcross_entropy_evalとcross_entropy_winに返る。
void calc_cross_entropy(Value deep, Value shallow, const PackedSfenValue& psv,
	double& cross_entropy_eval, double& cross_entropy_win)
{
	const double p /* teacher_winrate */ = winning_percentage(deep);
	const double q /* eval_winrate    */ = winning_percentage(shallow);
	const double t = (psv.isWin) ? 1.0 : 0.0;

	constexpr double epsilon = 0.000001;
	cross_entropy_eval = ELMO_LAMBDA *
		(-p * std::log(q + epsilon) - (1.0 - p) * std::log(1.0 - q + epsilon));
	cross_entropy_win = (1.0 - ELMO_LAMBDA) *
		(-t * std::log(q + epsilon) - (1.0 - t) * std::log(1.0 - q + epsilon));
}

#endif


#if defined( LOSS_FUNCTION_IS_YANE_ELMO_METHOD )

// isWin : この手番側が最終的に勝利したかどうか
double calc_grad(Value deep, Value shallow, PackedSfenValue& psv)
{
	// elmo(WCSC27)方式
	// 実際のゲームの勝敗で補正する。

	const double eval_winrate = winning_percentage(shallow);
	const double teacher_winrate = winning_percentage(deep);

	// 期待勝率を勝っていれば1、負けていれば 0として補正項として用いる。
	const double t = (psv.isWin) ? 1.0 : 0.0;

	// gamePly == 0なら  λ = 0.8ぐらい。(勝敗の影響を小さめにする)
	// gamePly == ∞なら λ = 0.4ぐらい。(元のelmo式ぐらいの値になる)
	const double LAMBDA = 0.8 - (0.8-0.4)*(double)std::min((int)psv.gamePly, 100)/100.0;

	const double dsig = (1 - LAMBDA) * (eval_winrate - t) + LAMBDA * (eval_winrate - teacher_winrate);

	return dsig;
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

	// mseの計算用に1000局面ほど読み込んでおく。
	void read_for_mse()
	{
		Position& pos = Threads.main()->rootPos;
		for (int i = 0; i < 1000; ++i)
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
		double sum_error3 = 0;

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
		double cross_entropy_eval, cross_entropy_win;
		double sum_cross_entropy_eval = 0, sum_cross_entropy_win = 0;
#endif

		// 平手の初期局面のeval()の値を表示させて、揺れを見る。
		pos.set_hirate();
		cout << endl << "hirate eval = " << Eval::evaluate(pos);

		int i = 0;
		for (auto& ps : sfen_for_mse)
		{
//			auto sfen = pos.sfen_unpack(ps.data);
//			pos.set(sfen);

			pos.set_from_packed_sfen(ps.sfen);

			auto th = Threads[thread_id];
			pos.set_this_thread(th);

			// 浅い探索の評価値
			// qsearch()だとそこそこ時間がかかるのでevaluate()にしておく。
			// 目安としてはこれでいいだろう。
			// EvalHashは事前に無効化してある。
			auto shallow_value = Eval::evaluate(pos);
			
			// 深い探索の評価値
			auto deep_value = (Value)ps.score;

			// --- 誤差の計算

			auto dsig = calc_grad(deep_value, shallow_value, ps);
			// rmse的なもの
			sum_error += dsig*dsig;
			// 勾配の絶対値を足したもの
			sum_error2 += abs(dsig);
			// 評価値の差の絶対値を足したもの
			sum_error3 += abs(shallow_value - deep_value);


			// --- 交差エントロピーの計算

			// とりあえずelmo methodの時だけ勝率項と勝敗項に関して
			// 交差エントロピーを計算して表示させる。

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
			calc_cross_entropy(deep_value, shallow_value, ps , cross_entropy_eval, cross_entropy_win);
			sum_cross_entropy_eval += cross_entropy_eval;
			sum_cross_entropy_win  += cross_entropy_win;
#endif

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

		// rmse = root mean square error : 平均二乗誤差
		// mae  = mean absolute error    : 平均絶対誤差
		auto dsig_rmse = std::sqrt(sum_error / sfen_for_mse.size());
		auto dsig_mae = sum_error2 / sfen_for_mse.size();
		auto eval_mae = sum_error3 / sfen_for_mse.size();
		cout << " , dsig rmse = " << dsig_rmse << " , dsig mae = " << dsig_mae
			<< " , eval mae = " << eval_mae
#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
			<< " , cross_entropy_eval = " << sum_cross_entropy_eval
			<< " , cross_entropy_win = " << sum_cross_entropy_win
#endif
			<<  endl;
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
	vector<Key> hash; // 64MB*8 = 512MB

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
				cout << endl << sr.total_done << " sfens , at " << now_string() << flush;
			} else {
				// これぐらいは出力しておく。
				cout << '.' << flush;
			}

			// このタイミングで勾配をweight配列に反映。勾配の計算も1M局面ごとでmini-batch的にはちょうどいいのでは。

			Eval::update_weights(/* ++epoch */);

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
		auto r = Learner::qsearch(pos);
		auto shallow_value = r.first;

		// qsearchではなくevaluate()の値をそのまま使う場合。
		// auto shallow_value = Eval::evaluate(pos);

		// 深い探索の評価値
		auto deep_value = (Value)ps.score;

		// 勾配
		double dj_dw = calc_grad(deep_value, shallow_value , ps);

		// 現在、leaf nodeで出現している特徴ベクトルに対する勾配(∂J/∂Wj)として、jd_dwを加算する。

		// mini batchのほうが勾配が出ていいような気がする。
		// このままleaf nodeに行って、勾配配列にだけ足しておき、あとでrmseの集計のときにAdaGradしてみる。

		auto rootColor = pos.side_to_move();

		auto pv = r.second;

		// PVの初手が異なる場合は学習に用いないほうが良いのでは…。
		// 全然違うところを探索した結果だとそれがノイズに成りかねない。
		// 評価値の差が大きすぎるところも学習対象としないほうがいいかも…。

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
			
			// leafでevaluate()は呼ばないので差分計算していく必要はないのでevaluate()を呼び出す必要はない。
			//	Eval::evaluate(pos);
		}

		// leafに到達したのでこの局面に出現している特徴に勾配を加算しておく。
		// 勾配に基づくupdateはのちほど行なう。
		Eval::add_grad(pos,rootColor,dj_dw);

		// 局面を巻き戻す
		for (auto it = pv.rbegin(); it != pv.rend(); ++it)
			pos.undo_move(*it);

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
void learn(Position&, istringstream& is)
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

	// 0であれば、デフォルト値になる。
	double eta = 0.0;

	// あとで復元するために保存しておく。
	auto oldGlobalOptions = GlobalOptions;
	// eval hashにhitするとrmseなどの計算ができなくなるのでオフにしておく。
	GlobalOptions.use_eval_hash = false;
	// 置換表にhitするとそこで以前の評価値で枝刈りがされることがあるのでオフにしておく。
	GlobalOptions.use_hash_probe = false;

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

			//			cout << "ERROR! : targetdir , this function is only for Windows." << endl;
		}

		// ループ回数の指定
		else if (option == "loop")      is >> loop;

		// 棋譜ファイル格納フォルダ(ここから相対pathで棋譜ファイルを取得)
		else if (option == "basedir")   is >> base_dir;

		// ミニバッチのサイズ
		else if (option == "batchsize") is >> mini_batch_size;

		// 学習率
		else if (option == "eta")       is >> eta;

#if defined (LOSS_FUNCTION_IS_ELMO_METHOD) || defined (LOSS_FUNCTION_IS_YANE_ELMO_METHOD)
		// LAMBDA
		else if (option == "lambda")    is >> ELMO_LAMBDA;
#endif

		// さもなくば、それはファイル名である。
		else
			filenames.push_back(option);
	}

	cout << "learn command , ";

	// OpenMP無効なら警告を出すように。
#ifndef _OPENMP
	cout << "Warning! OpenMP disabled." << endl;
#endif

	// 学習棋譜ファイルの表示
	if (target_dir != "")
	{
		string kif_base_dir = path_combine(base_dir, target_dir);

		// このフォルダを根こそぎ取る。base_dir相対にしておく。
#if defined(_MSC_VER)
		namespace sys = std::tr2::sys;
		sys::path p(kif_base_dir); // 列挙の起点
		std::for_each(sys::directory_iterator(p), sys::directory_iterator(),
			[&](const sys::path& p) {
			if (sys::is_regular_file(p))
				filenames.push_back(path_combine(target_dir, p.filename().generic_string()));
		});
#elif defined(__GNUC__)

		auto ends_with = [](std::string const & value, std::string const & ending)
		{
			if (ending.size() > value.size()) return false;
			return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
		};

		// 仕方ないのでdirent.hを用いて読み込む。
		DIR *dp;       // ディレクトリへのポインタ
		dirent* entry; // readdir() で返されるエントリーポイント

		dp = opendir(kif_base_dir.c_str());
		if (dp != NULL)
		{
			do {
				entry = readdir(dp);
				// ".bin"で終わるファイルのみを列挙
				if (entry != NULL && ends_with(entry->d_name, ".bin"))
				{
					//cout << entry->d_name << endl;
					filenames.push_back(path_combine(target_dir, entry->d_name));
				}
			} while (entry != NULL);
			closedir(dp);
		}
#endif
	}

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
#if defined (LOSS_FUNCTION_IS_ELMO_METHOD) || defined (LOSS_FUNCTION_IS_YANE_ELMO_METHOD)
	cout << "\nLAMBDA          : " << ELMO_LAMBDA;
#endif

	// -----------------------------------
	//            各種初期化
	// -----------------------------------

	cout << "\ninit..";

	// 評価関数パラメーターの読み込み
	is_ready();

	cout << "\ninit_grad..";

	// 評価関数パラメーターの勾配配列の初期化
	Eval::init_grad(eta);

#if 0
	// 平手の初期局面に対して1.0の勾配を与えてみるテスト。
	pos.set_hirate();
	cout << Eval::evaluate(pos) << endl;
	//Eval::print_eval_stat(pos);
	Eval::add_grad(pos, BLACK, 32.0);
	Eval::update_weights(1);
	pos.state()->sum.p[2][0] = VALUE_NOT_EVALUATED;
	cout << Eval::evaluate(pos) << endl;
	//Eval::print_eval_stat(pos);
#endif

#if defined( _OPENMP )
	omp_set_num_threads((int)Options["Threads"]);
#endif

	cout << "\ninit done." << endl;

	// 局面ファイルをバックグラウンドで読み込むスレッドを起動
	// (これを開始しないとmseの計算が出来ない。)
	learn_think.start_file_read_worker();

	learn_think.mini_batch_size = mini_batch_size;

	// mse計算用にデータ1万件ほど取得しておく。
	sr.read_for_mse();

	// この時点で一度rmseを計算(0 sfenのタイミング)
	// sr.calc_rmse();

	// -----------------------------------
	//   評価関数パラメーターの学習の開始
	// -----------------------------------

	// 学習開始。
	learn_think.go_think();

	// 最後に一度保存。
	learn_think.save();

	// GlobalOptionsの復元。
	GlobalOptions = oldGlobalOptions;
}


} // namespace Learner

#endif // EVAL_LEARN
