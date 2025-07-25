﻿#include "../config.h"

// 学習関係のルーチン
//
// 1) 棋譜の自動生成
//   → "gensfen"コマンド
// 2) 生成した棋譜からの評価関数パラメーターの学習
//   → "learn"コマンド
//   → 教師局面のshuffleもこのコマンドの拡張として行なう。
//   例) "learn shuffle"
// 3) 定跡の自動生成
//   → "makebook think"コマンド
//   → extra/book/book.cppで実装
// 4) 局後自動検討モード
//   →　GUIが補佐すべき問題なのでエンジンでは関与しないことにする。
// etc..

#if defined(EVAL_LEARN)

#include "learn.h"

// 学習用のevaluate絡みのheader
#include "../eval/evaluate_common.h"

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
#endif

// -----------------------------------
//    以下、実装部。
// -----------------------------------

#include <sstream>
#include <fstream>
#include <unordered_set>
#include <iomanip>
#include <list>
#include <cmath>	// std::exp(),std::pow(),std::log()
#include <cstring>	// memcpy()

#if defined (_OPENMP)
#include <omp.h>
#endif

#include "../misc.h"
#include "../thread.h"
#include "../position.h"
#include "../book/book.h"
#include "../tt.h"
#include "../mate/mate.h"
#include "multi_think.h"

#if defined(EVAL_NNUE)
#include "../eval/nnue/evaluate_nnue_learner.h"
#include <shared_mutex>
#endif

using namespace std;
namespace YaneuraOu {

// これは探索部で定義されているものとする。
extern Book::BookMoveSelector book;

// atomic<T>に対する足し算、引き算の定義
// Apery/learner.hppにあるatomicAdd()に合わせてある。
template <typename T>
T operator += (std::atomic<T>& x, const T rhs)
{
	T old = x.load(std::memory_order_consume);
	// このタイミングで他スレッドから値が書き換えられることは許容する。
	// 値が破壊されなければ良しという考え。
	T desired = old + rhs;
	while (!x.compare_exchange_weak(old, desired, std::memory_order_release, std::memory_order_consume))
		desired = old + rhs;
	return desired;
}
template <typename T>
T operator -= (std::atomic<T>& x, const T rhs) { return x += -rhs; }

namespace Learner
{

// 局面の配列 : PSVector は packed sfen vector の略。
typedef std::vector<PackedSfenValue> PSVector;

// -----------------------------------
//    局面のファイルへの書き出し
// -----------------------------------

// Sfenを書き出して行くためのヘルパクラス
struct SfenWriter
{
	// 書き出すファイル名と生成するスレッドの数
	SfenWriter(string filename, int thread_num)
	{
		sfen_buffers_pool.reserve((size_t)thread_num * 10);
		sfen_buffers.resize(thread_num);

		// 追加学習するとき、評価関数の学習後も生成される教師の質はあまり変わらず、教師局面数を稼ぎたいので
		// 古い教師も使うのが好ましいのでこういう仕様にしてある。
		fs.open(filename, ios::out | ios::binary | ios::app);
		filename_ = filename;

		finished = false;
	}

	~SfenWriter()
	{
		finished = true;
		file_worker_thread.join();
		fs.close();

		// file_worker_threadがすべて書き出したあとなのでbufferはすべて空のはずなのだが..
		for (auto p : sfen_buffers) { ASSERT_LV1(p == nullptr); }
		ASSERT_LV1(sfen_buffers_pool.empty());
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
			buf = new PSVector();
			buf->reserve(SFEN_WRITE_SIZE);
		}

		// スレッドごとに用意されており、一つのスレッドが同時にこのwrite()関数を呼び出さないので
		// この時点では排他する必要はない。
		buf->push_back(psv);

		if (buf->size() >= SFEN_WRITE_SIZE)
		{
			// sfen_buffers_poolに積んでおけばあとはworkerがよきに計らってくれる。

			// sfen_buffers_poolの内容を変更するときはmutexのlockが必要。
			std::unique_lock<std::mutex> lk(mutex);
			sfen_buffers_pool.push_back(buf);

			buf = nullptr;
			// buf == nullptrにしておけば次回にこの関数が呼び出されたときにバッファは確保される。
		}
	}

	// 自分のスレッド用のバッファに残っている分をファイルに書き出すためのバッファに移動させる。
	void finalize(size_t thread_id)
	{
		std::unique_lock<std::mutex> lk(mutex);

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
			sync_cout << endl << sfen_write_count << " sfens , at " << Tools::now_string() << sync_endl;

			// flush()はこのタイミングで十分。
			fs.flush();
		};

		while (!finished || sfen_buffers_pool.size())
		{
			vector<PSVector*> buffers;
			{
				std::unique_lock<std::mutex> lk(mutex);

				// まるごとコピー
				buffers = sfen_buffers_pool;
				sfen_buffers_pool.clear();
			}

			// 何も取得しなかったならsleep()
			if (!buffers.size())
				Tools::sleep(100);
			else
			{
				for (auto ptr : buffers)
				{
					fs.write(reinterpret_cast<const char*>(ptr->data()), sizeof(PackedSfenValue) * ptr->size());

					sfen_write_count += ptr->size();

#if 1
					// 処理した件数をここに加算していき、save_everyを超えたら、ファイル名を変更し、このカウンターをリセットする。
					save_every_counter += ptr->size();
					if (save_every_counter >= save_every)
					{
						save_every_counter = 0;
						// ファイル名を変更。

						fs.close();

						// ファイルにつける連番
						int n = (int)(sfen_write_count / save_every);
						// ファイル名を変更して再度openする。上書き考慮してios::appをつけておく。(運用によっては、ないほうがいいかも..)
						string filename = filename_ + "_" + std::to_string(n);
						fs.open(filename, ios::out | ios::binary | ios::app);
						cout << endl << "output sfen file = " << filename << endl;
					}
#endif

					// 棋譜を書き出すごとに'.'を出力。
					std::cout << ".";

					// 40回ごとに処理した局面数を出力
					// 最後、各スレッドの教師局面の余りを書き出すので中途半端な数が表示されるが、まあいいか…。
					// スレッドを論理コアの最大数まで酷使するとコンソールが詰まるのでもう少し間隔甘くてもいいと思う。
					if ((++time_stamp_count % 40) == 0)
						output_status();

					// このメモリは不要なのでこのタイミングで開放しておく。
					delete ptr;
				}
			}
		}

		// 終了前にもう一度、タイムスタンプを出力。
		output_status();
	}

	// この単位でファイル名を変更する。
	u64 save_every = UINT64_MAX;

private:

	fstream fs;

	// コンストラクタで渡されたファイル名
	std::string filename_;

	// 処理した件数をここに加算していき、save_everyを超えたら、ファイル名を変更し、このカウンターをリセットする。
	u64 save_every_counter = 0;

	// ファイルに書き込む用のthread
	std::thread file_worker_thread;
	// すべてのスレッドが終了したかのフラグ
	atomic<bool> finished;

	// タイムスタンプの出力用のカウンター
	u64 time_stamp_count = 0;

	// ファイルに書き出す前のバッファ
	// sfen_buffersは各スレッドに対するバッファ
	// sfen_buffers_poolは書き出しのためのバッファ。
	// 前者のバッファに局面をSFEN_WRITE_SIZEだけ積んだら、後者に積み替える。
	std::vector<PSVector*> sfen_buffers;
	std::vector<PSVector*> sfen_buffers_pool;

	// sfen_buffers_poolにアクセスするときに必要なmutex
	std::mutex mutex;

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
		hash.resize(GENSFEN_HASH_SIZE);

		// PCを並列化してgensfenするときに同じ乱数seedを引いていないか確認用の出力。
		std::cout << prng << std::endl;
	}

	virtual void thread_worker(size_t thread_id);
	void start_file_write_worker() { sw.start_file_write_worker(); }

	//  search_depth = 通常探索の探索深さ
	int search_depth;
	int search_depth2;

	// 生成する局面の評価値の上限
	int eval_limit;

	// ランダムムーブを行なう最小ply
	int random_move_minply;
	// ランダムムーブを行なう最大ply
	int random_move_maxply;
	// 1局のなかでランダムムーブを行なう回数
	int random_move_count;
	// Aperyのようにランダムムーブのときに1/Nの確率で玉を動かす。
	// また玉を動かしたときは1/Nの確率で相手番で1回ランダムムーブする。
	// AperyはN=2。ここ0を指定するとこの機能を無効化する。
	int random_move_like_apery;

	// ランダムムーブの代わりにmulti pvを使うとき用。
	// random_multi_pvは、MultiPVのときの候補手の数。
	// 候補手の指し手を採択するとき、1位の指し手の評価値とN位の指し手の評価値との差が
	// random_multi_pv_diffの範囲でなければならない。
	// random_multi_pv_depthはMultiPVのときの探索深さ。
	int random_multi_pv;
	int random_multi_pv_diff;
	int random_multi_pv_depth;

	// 書き出す局面のply(初期局面からの手数)の最小、最大。
	int write_minply;
	int write_maxply;

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
	// とりあえず、書き出す手数の最大のところで引き分け扱いになるものとする。
	const int MAX_PLY2 = write_maxply;

	// StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
	std::vector<StateInfo> states((size_t)MAX_PLY2 + MAX_PLY /* == search_depth + α */);
	StateInfo si;

	// 今回の指し手。この指し手で局面を進める。
	Move m = Move::none();

	// 終了フラグ
	bool quit = false;

	// 規定回数回になるまで繰り返し
	while (!quit)
	{
		// Positionに対して従属スレッドの設定が必要。
		// 並列化するときは、Threads (これが実体が vector<Thread*>なので、
		// Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
		auto th = Threads[thread_id];

		auto& pos = th->rootPos;
		pos.set_hirate(&si);

		// 自分スレッド用の置換表があるはずなので自分の置換表だけをクリアする。
		th->tt.clear();

		// 探索部で定義されているBookMoveSelectorのメンバを参照する。
		auto& book = YaneuraOu::book;

		// 1局分の局面を保存しておき、終局のときに勝敗を含めて書き出す。
		// 書き出す関数は、この下にあるflush_psv()である。
		PSVector a_psv;
		a_psv.reserve(MAX_PLY2 + MAX_PLY);

		// a_psvに積まれている局面をファイルに書き出す。
		// lastTurnIsWin : a_psvに積まれている最終局面の次の局面での勝敗
		// 勝ちのときは1。負けのときは-1。引き分けのときは0を渡す。
		// 返し値 : もう規定局面数に達したので終了する場合にtrue。
		auto flush_psv = [&](s8 lastTurnIsWin)
		{
			s8 isWin = lastTurnIsWin;

			// 終局の局面(の一つ前)から初手に向けて、各局面に関して、対局の勝敗の情報を付与しておく。
			// a_psvに保存されている局面は(手番的に)連続しているものとする。
			for (auto it = a_psv.rbegin(); it != a_psv.rend(); ++it)
			{
				// isWin == 0(引き分け)なら -1を掛けても 0(引き分け)のまま
				isWin = - isWin;
				it->game_result = isWin;

				// 局面を書き出そうと思ったら規定回数に達していた。
				// get_next_loop_count()内でカウンターを加算するので
				// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
				auto loop_count = get_next_loop_count();
				if (loop_count == UINT64_MAX)
				{
					// 終了フラグを立てておく。
					quit = true;
					return;
				}

				// 局面を一つ書き出す。
				sw.write(thread_id, *it);

#if 0
				pos.set_from_packed_sfen(it->sfen);
				cout << pos << "Win : " << it->isWin << " , " << it->score << endl;
#endif
			}
		};

		// ply手目でランダムムーブをするかどうかのフラグ
		vector<bool> random_move_flag;
		{
			// ランダムムーブを入れるならrandom_move_maxply手目までに絶対にrandom_move_count回入れる。
			// そこそこばらけて欲しい。
			// どれくらいがベストなのかはよくわからない。色々条件を変えて実験中。
			
			// a[0] = 0 , a[1] = 1, ... みたいな配列を作って、これを
			// Fisher-Yates shuffleして先頭のN個を取り出せば良い。
			// 実際には、N個欲しいだけなので先頭N個分だけFisher-Yatesでshuffleすれば良い。

			vector<int> a;
			a.reserve((size_t)random_move_maxply);

			// random_move_minply , random_move_maxplyは1 originで指定されるが、
			// ここでは0 originで扱っているので注意。
			for (int i = std::max(random_move_minply - 1 , 0) ; i < random_move_maxply; ++i)
				a.push_back(i);

			// Apery方式のランダムムーブの場合、insert()がrandom_move_count回呼び出される可能性があるので
			// それを考慮したサイズだけ確保しておく。
			random_move_flag.resize((size_t)random_move_maxply + random_move_count);

			// a[]のsize()を超える回数のランダムムーブは適用できないので制限する。
			for (int i = 0 ; i < std::min(random_move_count, (int)a.size()) ; ++i)
			{
				swap(a[i], a[prng.rand((u64)a.size() - i) + i]);
				random_move_flag[a[i]] = true;
			}
		}

		// random moveを行なった回数をカウントしておくカウンター
		// random_move_minply == -1のときに、連続してランダムムーブを行なうので、このときに用いる。
		int random_move_c = 0;

		// ply : 初期局面からの手数
		for (int ply = 0; ; ++ply)
		{
			//cout << pos << endl;

			// 今回の探索depth
			// gotoで飛ぶので先に宣言しておく。
			int depth = search_depth + (int)prng.rand(search_depth2 - search_depth + 1);

			// 長手数に達したのか
			if (ply >= MAX_PLY2)
			{
#if defined (LEARN_GENSFEN_USE_DRAW_RESULT)
				// 勝敗 = 引き分けとして書き出す。
				// こうしたほうが自分が入玉したときに、相手の入玉を許しにくい(かも)
				flush_psv(0);
#endif
				break;
			}

			// 全駒されて詰んでいたりしないか？
			if (pos.is_mated())
			{
				// (この局面の一つ前の局面までは書き出す)
				flush_psv(-1);
				break;
			}

			// 宣言勝ち
			if (pos.DeclarationWin() != Move::none())
			{
				// (この局面の一つ前の局面までは書き出す)
				flush_psv(1);
				break;
			}

			// 定跡
			if ((m = book.probe(pos)) != Move::none())
			{
				// 定跡にhitした。
				// その指し手はmに格納された。

				// 定跡の局面は学習には用いない。
				a_psv.clear();

				if (random_move_minply != -1)
					// 定跡の局面であっても、一定確率でランダムムーブは行なう。
					goto RANDOM_MOVE;
				else
					// random_move_minplyとして-1が指定されているときは定跡を抜けるところまでは定跡に従って指す。
					// 巨大定跡を用いて、ConsiderBookMoveCount trueとして定跡を抜けた局面を無数に用意しておき、
					// そこから5回ランダムムーブを行なう、などの用途に用いる。
					goto DO_MOVE;
			}

			{
				// search_depth～search_depth2 手読みの評価値とPV(最善応手列)
				// 探索窓を狭めておいても問題ないはず。

				auto pv_value1 = search(pos, depth);

				auto value1 = pv_value1.first;
				auto& pv1 = pv_value1.second;

				// 評価値の絶対値がこの値以上の局面については
				// その局面を学習に使うのはあまり意味がないのでこの試合を終了する。
				// これをもって勝敗がついたという扱いをする。

				// 1手詰め、宣言勝ちならば、ここでmate_in(2)が返るのでeval_limitの上限値と同じ値になり、
				// このif式は必ず真になる。resignについても同様。

				if (abs(value1) >= eval_limit)
				{
//					sync_cout << pos << "eval limit = " << eval_limit << " over , move = " << pv1[0] << sync_endl;

					// この局面でvalue1 >= eval_limitならば、(この局面の手番側の)勝ちである。
					flush_psv((value1 >= eval_limit) ? 1 : -1);
					break;
				}

				// おかしな指し手の検証
				if (pv1.size() > 0
					&& (pv1[0] == Move::resign() || pv1[0] == Move::win() || pv1[0] == Move::none())
					)
				{
					// MOVE_WINは、この手前で宣言勝ちの局面であるかチェックしているので
					// ここで宣言勝ちの指し手が返ってくることはないはず。
					// また、MOVE_RESIGNのときvalue1は1手詰めのスコアであり、eval_limitの最小値(-31998)のはずなのだが…。
					cout << "Error! : " << pos.sfen() << m << value1 << endl;
					break;
				}

				// 各千日手に応じた処理。

				s8 is_win = 0;
				bool game_end = false;
				auto draw_type = pos.is_repetition();
				switch (draw_type)
				{
				case REPETITION_WIN      : is_win =  1; game_end = true; break;
				case REPETITION_DRAW     : is_win =  0; game_end = true; break;
				case REPETITION_LOSE     : is_win = -1; game_end = true; break;

				// case REPETITION_SUPERIOR: break;
				// case REPETITION_INFERIOR: break;
					// これらは意味があるので無視して良い。
				default: break;
				}

				if (game_end)
				{
#if defined	(LEARN_GENSFEN_USE_DRAW_RESULT)
					// 引き分けを書き出すとき
					flush_psv(is_win);
#endif
					break;
				}

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
#if 1
						// 非合法手はやってこないはずなのだが。
						// 宣言勝ちとmated()でないことは上でテストしているので
						// 読み筋としてMOVE_WINとMOVE_RESIGNが来ないことは保証されている。(はずだが…)
						if (!pos.pseudo_legal(m) || !pos.legal(m))
						{
							cout << "Error! : " << pos.sfen() << m << endl;
						}
#endif
						pos.do_move(m, states[ply2++]);
						
						// 毎ノードevaluate()を呼び出さないと、evaluate()の差分計算が出来ないので注意！
						// depthが8以上だとこの差分計算はしないほうが速いと思われる。
						if (depth < 8)
							Eval::evaluate_with_no_return(pos);
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

#if 0
				dbg_hit_on(pv_value1.first == leaf_value);
				// gensfen depth 3 eval_limit 32000
				// Total 217749 Hits 203579 hit rate (%) 93.490
				// gensfen depth 6 eval_limit 32000
				// Total 78407 Hits 69190 hit rate (%) 88.245
				// gensfen depth 6 eval_limit 3000
				// Total 53879 Hits 43713 hit rate (%) 81.132

				// 置換表の指し手で枝刈りされるなどの問題。
				// これ、教師としては少し気持ち悪いが…。
#endif

				// depth 0の場合、pvが得られていないのでdepth 2で探索しなおす。
				if (search_depth <= 0)
				{
					pv_value1 = search(pos, 2);
					pv1 = pv_value1.second;
				}

				// 初期局面周辺はは類似局面ばかりなので
				// 学習に用いると過学習になりかねないから書き出さない。
				// →　比較実験すべき
				if (ply < write_minply - 1)
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

					// PV lineのleaf nodeでのroot colorから見たevaluate()の値を取得。
					// search()の返し値をそのまま使うのとこうするのとの善悪は良くわからない。
					psv.score = evaluate_leaf(pos, pv1);
					psv.gamePly = ply;

					// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
					ASSERT_LV3(pv_value1.second.size() >= 1);
					Move pv_move1 = pv_value1.second[0];
					psv.move = pv_move1.to_u16();
				}

			SKIP_SAVE:;

				// 何故かPVが得られなかった(置換表などにhitして詰んでいた？)ので次の対局に行く。
				// かなりのレアケースなので無視して良いと思う。
				if (pv1.size() == 0)
					break;
				
				// search_depth手読みの指し手で局面を進める。
				m = pv1[0];
			}

		RANDOM_MOVE:;

			// 合法手のなかからランダムに1手選ぶフェーズ
			if (
				// 1. random_move_minplyからrandom_move_maxplyの間でrandom_move_count回のランダムムーブを行なうモード
				(random_move_minply != -1 && ply < (int)random_move_flag.size() && random_move_flag[ply]) ||
				// 2. 定跡を抜けたあとにまとめてrandom_move_count回のランダムムーブを行なうモード
				(random_move_minply == -1 && random_move_c < random_move_count))
			{
				++random_move_c;

				// mateではないので合法手が1手はあるはず…。
				if (random_multi_pv == 0)
				{
					// 普通のランダムムーブ

					MoveList<LEGAL> list(pos);

					// ここをApery方式にするのとの善悪はよくわからない。
					if (random_move_like_apery == 0
						|| prng.rand(random_move_like_apery) != 0
					)
					{
						// 普通に合法手から1手選択
						m = list.at((size_t)prng.rand((u64)list.size()));
					}
					else {
						// 玉が動かせるなら玉を動かす
						Move moves[8]; // 8近傍
						Move* p = &moves[0];
						for (auto& m : list)
							if (type_of(pos.moved_piece_after(m)) == KING)
								*(p++) = m;
						size_t n = p - &moves[0];
						if (n != 0)
						{
							// 玉を動かす指し手
							m = moves[prng.rand(n)];

							// Apery方式ではこのとき1/2の確率で相手もランダムムーブ
							if (prng.rand(2) == 0)
							{
								// random_move_flag[ply]の次のところに"1"を追加するのがシンプルなhackか。
								random_move_flag.insert(random_move_flag.begin() + ply + 1, 1, true);
							}
						}
						else
							// 普通に合法手から1手選択
							m = list.at((size_t)prng.rand((u64)list.size()));
					}

					// 玉の2手指しのコードを入れていたが、合法手から1手選べばそれに相当するはずで
					// コードが複雑化するだけだから不要だと判断した。
				}
				else {
					// ロジックが複雑になるので、すまんがここで再度MultiPVで探索する。
					Learner::search(pos, random_multi_pv_depth, random_multi_pv);
					// rootMovesの上位N手のなかから一つ選択

					auto& rm = pos.this_thread()->rootMoves;

					u64 s = min((u64)rm.size(), (u64)random_multi_pv);
					for (u64 i = 1; i < s; ++i)
					{
						// rm[0]の評価値との差がrandom_multi_pv_diffの範囲でなければならない。
						// rm[x].scoreは、降順に並んでいると仮定できる。 
						if (rm[0].score > rm[i].score + random_multi_pv_diff)
						{
							s = i;
							break;
						}
					}

					m = rm[prng.rand(s)].pv[0];

					// まだ1局面も書き出していないのに終局してたので書き出し処理は端折って次の対局に。
					if (!m.is_ok())
						break;
				}

				// ゲームの勝敗から指し手を評価しようとするとき、
				// 今回のrandom moveがあるので、ここ以前には及ばないようにする。
				a_psv.clear(); // 保存していた局面のクリア
			}

		DO_MOVE:;
			pos.do_move(m, states[ply]);

			// 差分計算を行なうために毎node evaluate()を呼び出しておく。
			Eval::evaluate_with_no_return(pos);

		} // for (int ply = 0; ; ++ply)
	
	} // while(!quit)
	
	sw.finalize(thread_id);
}

// -----------------------------------
//    棋譜を生成するコマンド(master thread)
// -----------------------------------

// 棋譜を生成するコマンド
void gen_sfen(Position&, istringstream& is)
{
	// スレッド数(これは、USIのsetoptionで与えられる)
	u32 thread_num = (u32)Options["Threads"];

	// 生成棋譜の個数 default = 80億局面(Ponanza仕様)
	u64 loop_max = 8000000000UL;

	// 評価値がこの値になったら生成を打ち切る。
	int eval_limit = 3000;

	// 探索深さ
	int search_depth = 3;
	int search_depth2 = int_min;

	// ランダムムーブを行なう最小plyと最大plyと回数
	int random_move_minply = 1;
	int random_move_maxply = 24;
	int random_move_count = 5;
	// ランダムムーブをAperyのように玉を主に動かす機能
	// これを例えば3にすると1/3の確率で玉を動かす。
	int random_move_like_apery = 0;
	// ランダムムーブの代わりにmultipvで探索してそのなかからランダムに選ぶときはrandom_multi_pv = 1以上の数にする。
	int random_multi_pv       = 0;
	int random_multi_pv_diff  = 32000;
	int random_multi_pv_depth = int_min;

	// 書き出す局面のply(初期局面からの手数)の最小、最大。
	int write_minply = 16;
	int write_maxply = 400;

	// 書き出すファイル名
	string output_file_name = "generated_kifu.bin";

	string token;

	// eval hashにhitすると初期局面付近の評価値として、hash衝突して大きな値を書き込まれてしまうと
	// eval_limitが小さく設定されているときに初期局面で毎回eval_limitを超えてしまい局面の生成が進まなくなる。
	// そのため、eval hashは無効化する必要がある。
	// あとeval hashのhash衝突したときに、変な値の評価値が使われ、それを教師に使うのが気分が悪いというのもある。
	bool use_eval_hash = false;

	// この単位でファイルに保存する。
	// ファイル名は file_1.bin , file_2.binのように連番がつく。
	u64 save_every = UINT64_MAX;

	// ファイル名の末尾にランダムな数値を付与する。
	bool random_file_name = false;

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
		else if (token == "output_file_name")
			is >> output_file_name;
		else if (token == "eval_limit")
		{
			is >> eval_limit;
			// 最大値を1手詰みのスコアに制限する。(そうしないとループを終了しない可能性があるので)
			eval_limit = std::min(eval_limit, (int)mate_in(2));
		}
		else if (token == "random_move_minply")
			is >> random_move_minply;
		else if (token == "random_move_maxply")
			is >> random_move_maxply;
		else if (token == "random_move_count")
			is >> random_move_count;
		else if (token == "random_move_like_apery")
			is >> random_move_like_apery;
		else if (token == "random_multi_pv")
			is >> random_multi_pv;
		else if (token == "random_multi_pv_diff")
			is >> random_multi_pv_diff;
		else if (token == "random_multi_pv_depth")
			is >> random_multi_pv_depth;
		else if (token == "write_minply")
			is >> write_minply;
		else if (token == "write_maxply")
			is >> write_maxply;
		else if (token == "use_eval_hash")
			is >> use_eval_hash;
		else if (token == "save_every")
			is >> save_every;
		else if (token == "random_file_name")
			is >> random_file_name;
		else
			cout << "Error! : Illegal token " << token << endl;
	}

#if defined(USE_GLOBAL_OPTIONS)
	// あとで復元するために保存しておく。
	auto oldGlobalOptions = GlobalOptions;
	GlobalOptions.use_eval_hash = use_eval_hash;
#endif

	// search depth2が設定されていないなら、search depthと同じにしておく。
	if (search_depth2 == int_min)
		search_depth2 = search_depth;
	if (random_multi_pv_depth == int_min)
		random_multi_pv_depth = search_depth;

	if (random_file_name)
	{
		// output_file_nameにこの時点でランダムな数値を付与してしまう。
		PRNG r;
		// 念のため乱数振り直しておく。
		for(int i=0;i<10;++i)
			r.rand(1);
		auto to_hex = [](u64 u){
			std::stringstream ss;
			ss << std::hex << u;
			return ss.str();
		};
		// 64bitの数値で偶然かぶると嫌なので念のため64bitの数値２つくっつけておく。
		output_file_name = output_file_name + "_" + to_hex(r.rand<u64>()) + to_hex(r.rand<u64>());
	}

	std::cout << "gensfen : " << endl
		<< "  search_depth = " << search_depth << " to " << search_depth2 << endl
		<< "  loop_max = " << loop_max << endl
		<< "  eval_limit = " << eval_limit << endl
		<< "  thread_num (set by USI setoption) = " << thread_num << endl
		<< "  book_moves (set by USI setoption) = " << Options["BookMoves"] << endl
		<< "  random_move_minply     = " << random_move_minply << endl
		<< "  random_move_maxply     = " << random_move_maxply << endl
		<< "  random_move_count      = " << random_move_count << endl
		<< "  random_move_like_apery = " << random_move_like_apery << endl
		<< "  random_multi_pv        = " << random_multi_pv << endl
		<< "  random_multi_pv_diff   = " << random_multi_pv_diff << endl
		<< "  random_multi_pv_depth  = " << random_multi_pv_depth << endl
		<< "  write_minply           = " << write_minply << endl
		<< "  write_maxply           = " << write_maxply << endl
		<< "  output_file_name       = " << output_file_name << endl
		<< "  use_eval_hash          = " << use_eval_hash << endl
		<< "  save_every             = " << save_every << endl
		<< "  random_file_name       = " << random_file_name << endl;

	// Options["Threads"]の数だけスレッドを作って実行。
	{
		SfenWriter sw(output_file_name, thread_num);
		sw.save_every = save_every;

		MultiThinkGenSfen multi_think(search_depth, search_depth2, sw);
		multi_think.set_loop_max(loop_max);
		multi_think.eval_limit = eval_limit;
		multi_think.random_move_minply = random_move_minply;
		multi_think.random_move_maxply = random_move_maxply;
		multi_think.random_move_count = random_move_count;
		multi_think.random_move_like_apery = random_move_like_apery;
		multi_think.random_multi_pv = random_multi_pv;
		multi_think.random_multi_pv_diff = random_multi_pv_diff;
		multi_think.random_multi_pv_depth = random_multi_pv_depth;
		multi_think.write_minply = write_minply;
		multi_think.write_maxply = write_maxply;
		multi_think.start_file_write_worker();
		multi_think.go_think();

		// SfenWriterのデストラクタでjoinするので、joinが終わってから終了したというメッセージを
		// 表示させるべきなのでここをブロックで囲む。
	}

	std::cout << "gensfen finished." << endl;

#if defined(USE_GLOBAL_OPTIONS)
	// GlobalOptionsの復元。
	GlobalOptions = oldGlobalOptions;
#endif

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
double calc_grad(Value deep, Value shallow, const PackedSfenValue& psv)
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
double calc_grad(Value deep, Value shallow, const PackedSfenValue& psv)
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
// 0.33は、elmo(WCSC27)で使われていた定数(0.5)相当
double ELMO_LAMBDA = 0.33;
double ELMO_LAMBDA2 = 0.33;
double ELMO_LAMBDA_LIMIT = 32000;

double calc_grad(Value deep, Value shallow , const PackedSfenValue& psv)
{
	// elmo(WCSC27)方式
	// 実際のゲームの勝敗で補正する。

	const double eval_winrate = winning_percentage(shallow);
	const double teacher_winrate = winning_percentage(deep);

	// 期待勝率を勝っていれば1、負けていれば 0、引き分けなら0.5として補正項として用いる。
	// game_result = 1,0,-1なので1足して2で割る。
	const double t = double(psv.game_result + 1) / 2;

	// 深い探索での評価値がELMO_LAMBDA_LIMITを超えているならELMO_LAMBDAではなくELMO_LAMBDA2を適用する。
	const double lambda = (abs(deep) >= ELMO_LAMBDA_LIMIT) ? ELMO_LAMBDA2 : ELMO_LAMBDA;

	// 実際の勝率を補正項として使っている。
	// これがelmo(WCSC27)のアイデアで、現代のオーパーツ。
	const double grad = (1 - lambda) * (eval_winrate - t) + lambda * (eval_winrate - teacher_winrate);

	return grad;
}

// 学習時の交差エントロピーの計算
// elmo式の勝敗項と勝率項との個別の交差エントロピーが引数であるcross_entropy_evalとcross_entropy_winに返る。
void calc_cross_entropy(Value deep, Value shallow, const PackedSfenValue& psv,
	double& cross_entropy_eval, double& cross_entropy_win, double& cross_entropy,
	double& entropy_eval, double& entropy_win, double& entropy)
{
	const double p /* teacher_winrate */ = winning_percentage(deep);
	const double q /* eval_winrate    */ = winning_percentage(shallow);
	const double t = double(psv.game_result + 1) / 2;

	constexpr double epsilon = 0.000001;

	// 深い探索での評価値がELMO_LAMBDA_LIMITを超えているならELMO_LAMBDAではなくELMO_LAMBDA2を適用する。
	const double lambda = (abs(deep) >= ELMO_LAMBDA_LIMIT) ? ELMO_LAMBDA2 : ELMO_LAMBDA;

	const double m = (1.0 - lambda) * t + lambda * p;

	cross_entropy_eval =
		(-p * std::log(q + epsilon) - (1.0 - p) * std::log(1.0 - q + epsilon));
	cross_entropy_win =
		(-t * std::log(q + epsilon) - (1.0 - t) * std::log(1.0 - q + epsilon));
	entropy_eval =
		(-p * std::log(p + epsilon) - (1.0 - p) * std::log(1.0 - p + epsilon));
	entropy_win =
		(-t * std::log(t + epsilon) - (1.0 - t) * std::log(1.0 - t + epsilon));

	cross_entropy =
		(-m * std::log(q + epsilon) - (1.0 - m) * std::log(1.0 - q + epsilon));
	entropy =
		(-m * std::log(m + epsilon) - (1.0 - m) * std::log(1.0 - m + epsilon));
}

#endif


// 目的関数として他のバリエーションも色々用意するかも..

double calc_grad(Value shallow, const PackedSfenValue& psv) {
	return calc_grad((Value)psv.score, shallow, psv);
}

// Sfenの読み込み機
struct SfenReader
{
	SfenReader(int thread_num)
	{
		packed_sfens.resize(thread_num);
		total_read = 0;
		total_done = 0;
		last_done = 0;
		next_update_weights = 0;
		save_count = 0;
		end_of_files = false;
		no_shuffle = false;
		stop_flag = false;

		hash.resize(READ_SFEN_HASH_SIZE);
	}

	~SfenReader()
	{
		if (file_worker_thread.joinable())
			file_worker_thread.join();

		for (auto p : packed_sfens)
			delete p;
		for (auto p : packed_sfens_pool)
			delete p;
	}

	// mseなどの計算用に用いる局面数
	// mini-batch size = 1Mが標準的なので、その0.2%程度なら時間的には無視できるはず。
	// 指し手一致率の計算でdepth = 1でsearch()をするので、単純比較はできないが…。
	const u64 sfen_for_mse_size = 2000;

	// mseなどの計算用に局面を読み込んでおく。
	void read_for_mse()
	{
		auto th = Threads.main();
		Position& pos = th->rootPos;
		for (u64 i = 0; i < sfen_for_mse_size; ++i)
		{
			PackedSfenValue ps;
			if (!read_to_thread_buffer(0, ps))
			{
				cout << "Error! read packed sfen , failed." << endl;
				break;
			}
			sfen_for_mse.push_back(ps);

			// hash keyを求める。
			StateInfo si;
			if (pos.set_from_packed_sfen(ps.sfen, &si, th).is_not_ok())
			{
				// 運悪くrmse計算用のsfenとして、不正なsfenを引いてしまっていた。
				cout << "Error! : illegal packed sfen " << pos.sfen() << endl;
				return;
			}
			sfen_for_mse_hash.insert(pos.key());
		}
	}

	void read_validation_set(const string file_name, int eval_limit)
	{
		ifstream fs(file_name, ios::binary);

		while (fs)
		{
			PackedSfenValue p;
			if (fs.read((char*)&p, sizeof(PackedSfenValue)))
			{
				if (eval_limit < abs(p.score) || abs(p.score) == VALUE_SUPERIOR)
					continue;
#if !defined (LEARN_GENSFEN_USE_DRAW_RESULT)
				if (p.game_result == 0)
					continue;
#endif

				sfen_for_mse.push_back(p);
			} else {
				break;
			}
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
		
		// バッファを使いきったのであれば自らdeleteを呼び出してこのバッファを開放する。
		if (thread_ps->size() == 0)
		{
			delete thread_ps;
			thread_ps = nullptr;
		}

		return true;
	}

	// [ASYNC] スレッドバッファに局面をある程度読み込む。
	bool read_to_thread_buffer_impl(size_t thread_id)
	{
		while (true)
		{
			{
				std::unique_lock<std::mutex> lk(mutex);
				// ファイルバッファから充填できたなら、それで良し。
				if (packed_sfens_pool.size() != 0)
				{
					// 充填可能なようなので充填して終了。

					packed_sfens[thread_id] = packed_sfens_pool.front();
					packed_sfens_pool.pop_front();

					total_read += THREAD_BUFFER_SIZE;

					return true;
				}
			}

			// もうすでに読み込むファイルは無くなっている。もうダメぽ。
			if (end_of_files)
				return false;

			// file workerがpacked_sfens_poolに充填してくれるのを待っている。
			// mutexはlockしていないのでいずれ充填してくれるはずだ。
			Tools::sleep(1);
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
			// もう無い
			if (filenames.size() == 0)
				return false;

			// 次のファイル名ひとつ取得。
			string filename = *filenames.rbegin();
			filenames.pop_back();

			auto result = binary_reader.Open(filename);
			cout << "open filename = " << filename << endl;
			ASSERT(result.is_ok());

			return true;
		};

		open_next_file();

		while (true)
		{
			// バッファが減ってくるのを待つ。
			// このsize()の読み取りはread onlyなのでlockしなくていいだろう。
			while (!stop_flag && packed_sfens_pool.size() >= SFEN_READ_SIZE / THREAD_BUFFER_SIZE)
				Tools::sleep(100);
			if (stop_flag)
				return;

			PSVector sfens(SFEN_READ_SIZE);
			// 次にこの位置から読み込む。
			size_t sfens_read_offset = 0;

			// ファイルバッファにファイルから読み込む。
			while (sfens_read_offset < SFEN_READ_SIZE)
			{
				size_t expected_size_of_read_bytes = (SFEN_READ_SIZE - sfens_read_offset) * sizeof(PackedSfenValue);
				size_t actual_size_of_read_bytes = 0;
				auto result = binary_reader.Read(&sfens[sfens_read_offset], expected_size_of_read_bytes, &actual_size_of_read_bytes);
				if (!(result.is_ok() || result.is_eof())) {
					cout << endl << "Failed to read a file." << endl;
					end_of_files = true;
					return;
				}

				sfens_read_offset += actual_size_of_read_bytes / sizeof(PackedSfenValue);
				if (sfens_read_offset < SFEN_READ_SIZE) {
					// ファイルの終端に達した等、必要な量を読み込むことができなかった。
					// 次のファイルを読み込む。
					if (!open_next_file())
					{
						// 次のファイルもなかった。あぼーん。
						cout << "..end of files." << endl;
						end_of_files = true;
						return;
					}
				}
			}

			// この読み込んだ局面データをshuffleする。
			// random shuffle by Fisher-Yates algorithm

			if (!no_shuffle)
			{
				auto size = sfens.size();
				for (size_t i = 0; i < size; ++i)
					swap(sfens[i], sfens[(size_t)(prng.rand((u64)size - i) + i)]);
			}

			// これをTHREAD_BUFFER_SIZEごとの細切れにする。それがsize個あるはず。
			// SFEN_READ_SIZEはTHREAD_BUFFER_SIZEの倍数であるものとする。
			ASSERT_LV3((SFEN_READ_SIZE % THREAD_BUFFER_SIZE)==0);

			auto size = size_t(SFEN_READ_SIZE / THREAD_BUFFER_SIZE);
			std::vector<PSVector*> ptrs;
			ptrs.reserve(size);

			for (size_t i = 0; i < size; ++i)
			{
				// このポインターのdeleteは、受け側で行なう。
				PSVector* ptr = new PSVector();
				ptr->resize(THREAD_BUFFER_SIZE);
				memcpy(&((*ptr)[0]), &sfens[i * THREAD_BUFFER_SIZE], sizeof(PackedSfenValue) * THREAD_BUFFER_SIZE);

				ptrs.push_back(ptr);
			}

			// sfensの用意が出来たので、折を見てコピー
			{
				std::unique_lock<std::mutex> lk(mutex);

				// ポインタをコピーするだけなのでこの時間は無視できるはず…。
				// packed_sfens_poolの内容を変更するのでmutexのlockが必要。

				for (size_t i = 0; i < size; ++i)
					packed_sfens_pool.push_back(ptrs[i]);
			}
		}
	}

	// sfenファイル群
	vector<string> filenames;

	// 読み込んだ局面数(ファイルからメモリ上のバッファへ)
	atomic<u64> total_read;

	// 処理した局面数
	atomic<u64> total_done;

	// 前回までに処理した件数
	u64 last_done;

	// total_readがこの値を超えたらupdate_weights()してmseの計算をする。
	u64 next_update_weights;

	u64 save_count;

	// 局面読み込み時のシャッフルを行わない。
	bool no_shuffle;

	bool stop_flag;

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

	// mse計算用のtest局面
	PSVector sfen_for_mse;

protected:

	// fileをバックグラウンドで読み込みしているworker thread
	std::thread file_worker_thread;

	// 局面の読み込み時にshuffleするための乱数
	PRNG prng;

	// ファイル群を読み込んでいき、最後まで到達したか。
	atomic<bool> end_of_files;


	// sfenファイルのハンドル
	SystemIO::BinaryReader binary_reader;

	// 各スレッド用のsfen
	// (使いきったときにスレッドが自らdeleteを呼び出して開放すべし。)
	std::vector<PSVector*> packed_sfens;

	// packed_sfens_poolにアクセスするときのmutex
	std::mutex mutex;

	// sfenのpool。fileから読み込むworker threadはここに補充する。
	// 各worker threadはここから自分のpacked_sfens[thread_id]に充填する。
	// ※　mutexをlockしてアクセスすること。
	std::list<PSVector*> packed_sfens_pool;

	// mse計算用の局面を学習に用いないためにhash keyを保持しておく。
	std::unordered_set<Key> sfen_for_mse_hash;
};

// 複数スレッドでsfenを生成するためのクラス
struct LearnerThink: public MultiThink
{
	LearnerThink(SfenReader& sr_):sr(sr_),stop_flag(false), save_only_once(false)
	{
#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
		learn_sum_cross_entropy_eval = 0.0;
		learn_sum_cross_entropy_win = 0.0;
		learn_sum_cross_entropy = 0.0;
		learn_sum_entropy_eval = 0.0;
		learn_sum_entropy_win = 0.0;
		learn_sum_entropy = 0.0;
#endif
#if defined(EVAL_NNUE)
		newbob_scale = 1.0;
		newbob_decay = 1.0;
		newbob_num_trials = 2;
		best_loss = std::numeric_limits<double>::max();
		latest_loss_sum = 0.0;
		latest_loss_count = 0;
#endif
	}

	virtual void thread_worker(size_t thread_id);

	// 局面ファイルをバックグラウンドで読み込むスレッドを起動する。
	void start_file_read_worker() { sr.start_file_read_worker(); }

	// 評価関数パラメーターをファイルに保存
	bool save(bool is_final=false);

	// sfenの読み出し器
	SfenReader& sr;

	// 学習の反復回数のカウンター
	u64 epoch = 0;

	// ミニバッチサイズのサイズ。必ずこのclassを使う側で設定すること。
	u64 mini_batch_size = 1000*1000;

	bool stop_flag;

	// 割引率
	double discount_rate;

	// 序盤を学習対象から外すオプション
	int reduction_gameply;

	// kk/kkp/kpp/kpppを学習させないオプション
	std::array<bool,4> freeze;

	// 教師局面の深い探索の評価値の絶対値がこの値を超えていたらその教師局面を捨てる。
	int eval_limit;

	// 評価関数の保存するときに都度フォルダを掘るかのフラグ。
	// trueだとフォルダを掘らない。
	bool save_only_once;

	// --- lossの計算

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
	// 学習用データのロスの計算用
	atomic<double> learn_sum_cross_entropy_eval;
	atomic<double> learn_sum_cross_entropy_win;
	atomic<double> learn_sum_cross_entropy;
	atomic<double> learn_sum_entropy_eval;
	atomic<double> learn_sum_entropy_win;
	atomic<double> learn_sum_entropy;
#endif

#if defined(EVAL_NNUE)
	shared_timed_mutex nn_mutex;
	double newbob_scale;
	double newbob_decay;
	int newbob_num_trials;
	double best_loss;
	double latest_loss_sum;
	u64 latest_loss_count;
	std::string best_nn_directory;
#endif

	u64 eval_save_interval;
	u64 loss_output_interval;
	u64 mirror_percentage;

	// ロスの計算。
	// done : 今回対象とした局面数
	void calc_loss(size_t thread_id , u64 done);

	// ↑のlossの計算をタスクとして定義してやり、それを実行する
	TaskDispatcher task_dispatcher;
};

void LearnerThink::calc_loss(size_t thread_id, u64 done)
{

#if defined(EVAL_NNUE)
	std::cout << "PROGRESS: " << Tools::now_string() << ", ";
	std::cout << sr.total_done << " sfens";
	std::cout << ", iteration " << epoch;
	std::cout << ", eta = " << Eval::get_eta() << ", ";
#endif

#if !defined(LOSS_FUNCTION_IS_ELMO_METHOD)
	double sum_error = 0;
	double sum_error2 = 0;
	double sum_error3 = 0;
#endif

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
	// 検証用データのロスの計算用
	atomic<double> test_sum_cross_entropy_eval,test_sum_cross_entropy_win,test_sum_cross_entropy;
	atomic<double> test_sum_entropy_eval,test_sum_entropy_win,test_sum_entropy;
	test_sum_cross_entropy_eval = 0;
	test_sum_cross_entropy_win = 0;
	test_sum_cross_entropy = 0;
	test_sum_entropy_eval = 0;
	test_sum_entropy_win = 0;
	test_sum_entropy = 0;

	// 学習時のnorm
	atomic<double> sum_norm;
	sum_norm = 0;
#endif

	// 深い探索のpvの初手と、search(1)のpvの初手の指し手が一致した回数。
	atomic<int> move_accord_count;
	move_accord_count = 0;

	// 平手の初期局面のeval()の値を表示させて、揺れを見る。
	auto th = Threads[thread_id];
	auto& pos = th->rootPos;
	StateInfo si;
	pos.set_hirate(&si);
	std::cout << "hirate eval = " << Eval::evaluate(pos);

	//Eval::print_eval_stat(pos);

	// ここ、並列化したほうが良いのだがslaveの前の探索が終わってなかったりしてちょっと面倒。
	// taskを呼び出すための仕組みを作ったのでそれを用いる。

	// Apery式並列タスク実行
	std::atomic_int64_t global_position_index;
	global_position_index = 0;

	// スレッド一つにつき一つのタスクを作る。
	int num_tasks = (int)Options["Threads"];
	task_dispatcher.task_reserve(num_tasks);

	atomic<int> num_finished_tasks;
	num_finished_tasks = 0;

	for (int task_index = 0; task_index < num_tasks; ++task_index) {
		// TaskDispatcherを用いて各スレッドに作業を振る。
		// そのためのタスクの定義。
		// ↑で使っているposをcaptureされるとたまらんのでcaptureしたい変数は一つずつ指定しておく。
		auto task = [&test_sum_cross_entropy_eval, &test_sum_cross_entropy_win, &test_sum_cross_entropy,
			&test_sum_entropy_eval, &test_sum_entropy_win, &test_sum_entropy,
			&sum_norm, &num_finished_tasks, &move_accord_count,
			&global_position_index, this](size_t thread_id)
		{
			// 複数のプロセスでlearnコマンドを実行した場合、NUMAノード0しか使われなくなる問題への対処
			WinProcGroup::bindThisThread(thread_id);

			// 各タスク内のローカルな総和
			double local_test_sum_cross_entropy_eval = 0.0;
			double local_test_sum_cross_entropy_win = 0.0;
			double local_test_sum_cross_entropy = 0.0;
			double local_test_sum_entropy_eval = 0.0;
			double local_test_sum_entropy_win = 0.0;
			double local_test_sum_entropy = 0.0;
			double local_sum_norm = 0.0;
			int local_move_accord_count = 0;

			size_t num_sfens = sr.sfen_for_mse.size();
			for (size_t position_index = global_position_index++; position_index < num_sfens;
				position_index = global_position_index++) {
				auto th = Threads[thread_id];
				auto& pos = th->rootPos;
				StateInfo si;
				auto& ps = sr.sfen_for_mse[position_index];
				if (pos.set_from_packed_sfen(ps.sfen, &si, th).is_not_ok())
				{
					// 運悪くrmse計算用のsfenとして、不正なsfenを引いてしまっていた。
					cout << "Error! : illegal packed sfen " << pos.sfen() << endl;
				}

				// 浅い探索の評価値
				// evaluate()の値を用いても良いのだが、ロスを計算するときにlearn_cross_entropyと
				// 値が比較しにくくて困るのでqsearch()を用いる。
				// EvalHashは事前に無効化してある。(そうしないと毎回同じ値が返ってしまう)
				auto r = qsearch(pos);

				auto shallow_value = r.first;
				{
					const auto rootColor = pos.side_to_move();
					const auto pv = r.second;
					std::vector<StateInfo> states(pv.size());
					for (size_t i = 0; i < pv.size(); ++i)
					{
						pos.do_move(pv[i], states[i]);
						Eval::evaluate_with_no_return(pos);
					}
					shallow_value = (rootColor == pos.side_to_move()) ? Eval::evaluate(pos) : -Eval::evaluate(pos);
					for (auto it = pv.rbegin(); it != pv.rend(); ++it)
						pos.undo_move(*it);
				}

				// 深い探索の評価値
				auto deep_value = (Value)ps.score;

				// 注) このコードは、learnコマンドでeval_limitを指定しているときのことを考慮してない。

				// --- 誤差の計算

#if !defined(LOSS_FUNCTION_IS_ELMO_METHOD)
				auto grad = calc_grad(deep_value, shallow_value, ps);

				// rmse的なもの
				sum_error += grad * grad;
				// 勾配の絶対値を足したもの
				sum_error2 += abs(grad);
				// 評価値の差の絶対値を足したもの
				sum_error3 += abs(shallow_value - deep_value);
#endif

				// --- 交差エントロピーの計算

				// とりあえずelmo methodの時だけ勝率項と勝敗項に関して
				// 交差エントロピーを計算して表示させる。

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
				double test_cross_entropy_eval, test_cross_entropy_win, test_cross_entropy;
				double test_entropy_eval, test_entropy_win, test_entropy;
				calc_cross_entropy(deep_value, shallow_value, ps, test_cross_entropy_eval, test_cross_entropy_win, test_cross_entropy, test_entropy_eval, test_entropy_win, test_entropy);
				// 交差エントロピーの合計は定義的にabs()をとる必要がない。
				local_test_sum_cross_entropy_eval += test_cross_entropy_eval;
				local_test_sum_cross_entropy_win += test_cross_entropy_win;
				local_test_sum_cross_entropy += test_cross_entropy;
				local_test_sum_entropy_eval += test_entropy_eval;
				local_test_sum_entropy_win += test_entropy_win;
				local_test_sum_entropy += test_entropy;
				local_sum_norm += (double)abs(shallow_value);
#endif

				// 教師の指し手と浅い探索のスコアが一致するかの判定
				{
					auto r = search(pos, 1);
					if (r.second[0].to_u16() == ps.move)
						++local_move_accord_count;
				}
			}

			// グローバルな総和にまとめて足し合わせる。
			test_sum_cross_entropy_eval += local_test_sum_cross_entropy_eval;
			test_sum_cross_entropy_win += local_test_sum_cross_entropy_win;
			test_sum_cross_entropy += local_test_sum_cross_entropy;
			test_sum_entropy_eval += local_test_sum_entropy_eval;
			test_sum_entropy_win += local_test_sum_entropy_win;
			test_sum_entropy += local_test_sum_entropy;
			sum_norm += local_sum_norm;
			move_accord_count += local_move_accord_count;

			// タスクが一つ終了した。
			++num_finished_tasks;
		};

		// 定義したタスクをslaveに投げる。
		task_dispatcher.push_task_async(task);
	}

	// 自分自身もslaveとして参加する
	task_dispatcher.on_idle(thread_id);

	// すべてのtaskの完了を待つ
	while (num_finished_tasks < num_tasks)
		Tools::sleep(1);

#if !defined(LOSS_FUNCTION_IS_ELMO_METHOD)
	// rmse = root mean square error : 平均二乗誤差
	// mae  = mean absolute error    : 平均絶対誤差
	auto dsig_rmse = std::sqrt(sum_error / (sfen_for_mse.size() + epsilon));
	auto dsig_mae = sum_error2 / (sfen_for_mse.size() + epsilon);
	auto eval_mae = sum_error3 / (sfen_for_mse.size() + epsilon);
	cout << " , dsig rmse = " << dsig_rmse << " , dsig mae = " << dsig_mae
		<< " , eval mae = " << eval_mae;
#endif

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
#if defined(EVAL_NNUE)
	latest_loss_sum += test_sum_cross_entropy - test_sum_entropy;
	latest_loss_count += sr.sfen_for_mse.size();
#endif

	// learn_cross_entropyは、機械学習の世界ではtrain cross entropyと呼ぶべきかも知れないが、
	// 頭文字を略するときに、lceと書いて、test cross entropy(tce)と区別出来たほうが嬉しいのでこうしてある。

	if (sr.sfen_for_mse.size() && done)
	{
		cout
			<< " , test_cross_entropy_eval = "  << test_sum_cross_entropy_eval / sr.sfen_for_mse.size()
			<< " , test_cross_entropy_win = "   << test_sum_cross_entropy_win / sr.sfen_for_mse.size()
			<< " , test_entropy_eval = "        << test_sum_entropy_eval / sr.sfen_for_mse.size()
			<< " , test_entropy_win = "         << test_sum_entropy_win / sr.sfen_for_mse.size()
			<< " , test_cross_entropy = "       << test_sum_cross_entropy / sr.sfen_for_mse.size()
			<< " , test_entropy = "             << test_sum_entropy / sr.sfen_for_mse.size()
			<< " , norm = "						<< sum_norm
			<< " , move accuracy = "			<< (move_accord_count * 100.0 / sr.sfen_for_mse.size()) << "%";
		if (done != static_cast<u64>(-1))
		{
			cout
				<< " , learn_cross_entropy_eval = " << learn_sum_cross_entropy_eval / done
				<< " , learn_cross_entropy_win = "  << learn_sum_cross_entropy_win / done
				<< " , learn_entropy_eval = "       << learn_sum_entropy_eval / done
				<< " , learn_entropy_win = "        << learn_sum_entropy_win / done
				<< " , learn_cross_entropy = "      << learn_sum_cross_entropy / done
				<< " , learn_entropy = "            << learn_sum_entropy / done;
		}
		cout << endl;
	}
	else {
		cout << "Error! : sr.sfen_for_mse.size() = " << sr.sfen_for_mse.size() << " ,  done = " << done << endl;
	}

	// 次回のために0クリアしておく。
	learn_sum_cross_entropy_eval = 0.0;
	learn_sum_cross_entropy_win = 0.0;
	learn_sum_cross_entropy = 0.0;
	learn_sum_entropy_eval = 0.0;
	learn_sum_entropy_win = 0.0;
	learn_sum_entropy = 0.0;
#else
	<< endl;
#endif
}


void LearnerThink::thread_worker(size_t thread_id)
{
#if defined(_OPENMP)
	omp_set_num_threads((int)Options["Threads"]);
#endif

	auto th = Threads[thread_id];
	auto& pos = th->rootPos;

	// qsearch()を呼び出した回数。
	// ある程度呼び出すと置換表が汚れてくると思うので、クリアする。
	// 置換表は、自分のスレッド用の置換表が用意されている。(Thread.tt)
	u64 qsearch_count = 0;

	while (true)
	{
		// mseの表示(これはthread 0のみときどき行う)
		// ファイルから読み込んだ直後とかでいいような…。

#if defined(EVAL_NNUE)
		// 更新中に評価関数を使わないようにロックする。
		shared_lock<shared_timed_mutex> read_lock(nn_mutex, defer_lock);
		if (sr.next_update_weights <= sr.total_done ||
		    (thread_id != 0 && !read_lock.try_lock()))
#else
		if (sr.next_update_weights <= sr.total_done)
#endif
		{
			if (thread_id != 0)
			{
				// thread_id == 0以外は、待機。

				if (stop_flag)
					break;

				// rmseの計算などを並列化したいのでtask()が積まれていればそれを処理する。
				task_dispatcher.on_idle(thread_id);
				continue;
			}
			else
			{
				// thread_id == 0だけが以下の更新処理を行なう。

				// 初回はweight配列の更新は行わない。
				if (sr.next_update_weights == 0)
				{
					sr.next_update_weights += mini_batch_size;
					continue;
				}

#if !defined(EVAL_NNUE)
				// 現在時刻を出力。毎回出力する。
				std::cout << sr.total_done << " sfens , at " << Tools::now_string() << std::endl;

				// このタイミングで勾配をweight配列に反映。勾配の計算も1M局面ごとでmini-batch的にはちょうどいいのでは。
				Eval::update_weights(epoch , freeze);

				// デバッグ用にepochと現在のetaを表示してやる。
				std::cout << "epoch = " << epoch << " , eta = " << Eval::get_eta() << std::endl;
#else
				{
					// パラメータの更新

					// 更新中に評価関数を使わないようにロックする。
					lock_guard<shared_timed_mutex> write_lock(nn_mutex);
					Eval::NNUE::UpdateParameters(epoch);
				}
#endif
				++epoch;

				// 10億局面ごとに1回保存、ぐらいの感じで。

				// ただし、update_weights(),calc_rmse()している間の時間経過は無視するものとする。
				if (++sr.save_count * mini_batch_size >= eval_save_interval)
				{
					sr.save_count = 0;

					// この間、gradientの計算が進むと値が大きくなりすぎて困る気がするので他のスレッドを停止させる。
					const bool converged = save();
					if (converged)
					{
						stop_flag = true;
						sr.stop_flag = true;
						break;
					}
				}

				// rmseを計算する。1万局面のサンプルに対して行う。
				// 40コアでやると100万局面ごとにupdate_weightsするとして、特定のスレッドが
				// つきっきりになってしまうのあまりよくないような気も…。
				static u64 loss_output_count = 0;
				if (++loss_output_count * mini_batch_size >= loss_output_interval)
				{
					loss_output_count = 0;

					// 今回処理した件数
					u64 done = sr.total_done - sr.last_done;

					// lossの計算
					calc_loss(thread_id , done);

#if defined(EVAL_NNUE)
					Eval::NNUE::CheckHealth();
#endif

					// どこまで集計したかを記録しておく。
					sr.last_done = sr.total_done;
				}

				// 次回、この一連の処理は、次回、mini_batch_sizeだけ処理したときに再度やって欲しい。
				sr.next_update_weights += mini_batch_size;

				// main thread以外は、このsr.next_update_weightsの更新を待っていたので
				// この値が更新されると再度動き始める。				
			}
		}

		PackedSfenValue ps;
	RetryRead:;
		if (!sr.read_to_thread_buffer(thread_id, ps))
		{
			// 自分のスレッド用の局面poolを使い尽くした。
			// 局面がもうほとんど残っていないということだから、
			// 他のスレッドもすべて終了させる。

			stop_flag = true;
			break;
		}

		// 評価値が学習対象の値を超えている。
		// この局面情報を無視する。
		if (eval_limit < abs(ps.score) || abs(ps.score) == VALUE_SUPERIOR)
			goto RetryRead;

#if !defined (LEARN_GENSFEN_USE_DRAW_RESULT)
		if (ps.game_result == 0)
			goto RetryRead;
#endif

		// 序盤局面に関する読み飛ばし
		if (ps.gamePly < prng.rand(reduction_gameply))
			goto RetryRead;

#if 0
		auto sfen = pos.sfen_unpack(ps.data);
		pos.set(sfen);
#endif
		// ↑sfenを経由すると遅いので専用の関数を作った。
		StateInfo si;
		const bool mirror = prng.rand(100) < mirror_percentage;
		if (pos.set_from_packed_sfen(ps.sfen,&si,th,mirror).is_not_ok())
		{
			// 変なsfenを掴かまされた。デバッグすべき！
			// 不正なsfenなのでpos.sfen()で表示できるとは限らないが、しないよりマシ。
			cout << "Error! : illegal packed sfen = " << pos.sfen() << endl;
			goto RetryRead;
		}
#if !defined(EVAL_NNUE)
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
#endif

		// 全駒されて詰んでいる可能性がある。
		// また宣言勝ちの局面はPVの指し手でleafに行けないので学習から除外しておく。
		// (そのような教師局面自体を書き出すべきではないのだが古い生成ルーチンで書き出しているかも知れないので)
		if (pos.is_mated() || pos.DeclarationWin() != Move::none())
			goto RetryRead;

		// 読み込めたので試しに表示してみる。
		//		cout << pos << value << endl;

		// 浅い探索(qsearch)の評価値
		auto r = qsearch(pos);

		if ((++qsearch_count % 1000) == 0)
		{
			// qsearch()を1000回呼び出すごとに置換表をクリアする。
			// qsearch()で汚れる置換表はたかだか知れてるとは思うが、
			// 定期的にクリアはしたほうが良いと思われる。
			th->tt.clear();
		}

		auto pv = r.second;

		// 深い探索の評価値
		auto deep_value = (Value)ps.score;

		// mini batchのほうが勾配が出ていいような気がする。
		// このままleaf nodeに行って、勾配配列にだけ足しておき、あとでrmseの集計のときにAdaGradしてみる。

		auto rootColor = pos.side_to_move();

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

		// 現在の局面に対して勾配を加算するヘルパー関数。
		auto pos_add_grad = [&]() {
			// shallow_valueとして、leafでのevaluateの値を用いる。
			// qsearch()の戻り値をshallow_valueとして用いると、
			// PVが途中で途切れている場合、勾配を計算するのにevaluate()を呼び出した局面と、
			// その勾配を与える局面とが異なることになるので、これはあまり好ましい性質ではないと思う。
			// 置換表をオフにはしているのだが、1手詰みなどはpv配列を更新していないので…。

			Value shallow_value = (rootColor == pos.side_to_move()) ? Eval::evaluate(pos) : -Eval::evaluate(pos);

#if defined ( LOSS_FUNCTION_IS_ELMO_METHOD )
			// 学習データに対するロスの計算
			double learn_cross_entropy_eval, learn_cross_entropy_win, learn_cross_entropy;
			double learn_entropy_eval, learn_entropy_win, learn_entropy;
			calc_cross_entropy(deep_value, shallow_value, ps, learn_cross_entropy_eval, learn_cross_entropy_win, learn_cross_entropy, learn_entropy_eval, learn_entropy_win, learn_entropy);
			learn_sum_cross_entropy_eval += learn_cross_entropy_eval;
			learn_sum_cross_entropy_win += learn_cross_entropy_win;
			learn_sum_cross_entropy += learn_cross_entropy;
			learn_sum_entropy_eval += learn_entropy_eval;
			learn_sum_entropy_win += learn_entropy_win;
			learn_sum_entropy += learn_entropy;
#endif

#if !defined(EVAL_NNUE)
			// 勾配
			double dj_dw = calc_grad(deep_value, shallow_value, ps);

			// 現在、leaf nodeで出現している特徴ベクトルに対する勾配(∂J/∂Wj)として、jd_dwを加算する。

			// PV終端でなければ割引率みたいなものを適用。
			if (discount_rate != 0 && ply != (int)pv.size())
				dj_dw *= discount_rate;

			// leafに到達したのでこの局面に出現している特徴に勾配を加算しておく。
			// 勾配に基づくupdateはのちほど行なう。
			Eval::add_grad(pos, rootColor, dj_dw, freeze);
#else
			const double example_weight =
			    (discount_rate != 0 && ply != (int)pv.size()) ? discount_rate : 1.0;
			Eval::NNUE::AddExample(pos, rootColor, ps, example_weight);
#endif

			// 処理が終了したので処理した件数のカウンターをインクリメント
			sr.total_done++;
		};

		StateInfo state[MAX_PLY]; // qsearchのPVがそんなに長くなることはありえない。
		for (auto m : pv)
		{
			// 非合法手はやってこないはずなのだが。
			if (!pos.pseudo_legal(m) || !pos.legal(m))
			{
				cout << pos << m << endl;
				ASSERT_LV3(false);
			}

			// 各PV上のnodeでも勾配を加算する場合の処理。
			// discount_rateが0のときはこの処理は行わない。
			if (discount_rate != 0)
				pos_add_grad();

			pos.do_move(m, state[ply++]);
			
			// leafでのevaluateの値を用いるので差分更新していく。
			Eval::evaluate_with_no_return(pos);
		}

		// PVの終端局面に達したので、ここで勾配を加算する。
		pos_add_grad();

		// 局面を巻き戻す
		for (auto it = pv.rbegin(); it != pv.rend(); ++it)
			pos.undo_move(*it);

#if 0
		// rootの局面にも勾配を加算する場合
		shallow_value = (rootColor == pos.side_to_move()) ? Eval::evaluate(pos) : -Eval::evaluate(pos);
		dj_dw = calc_grad(deep_value, shallow_value, ps);
		Eval::add_grad(pos, rootColor, dj_dw , without_kpp);
#endif

	}
}


// 評価関数ファイルの書き出し。
bool LearnerThink::save(bool is_final)
{
	// 保存前にcheck sumを計算して出力しておく。(次に読み込んだときに合致するか調べるため)
	std::cout << "Check Sum = " << std::hex << Eval::calc_check_sum() << std::dec << std::endl;

	// 保存ごとにファイル名の拡張子部分を"0","1","2",..のように変えていく。
	// (あとでそれぞれの評価関数パラメーターにおいて勝率を比較したいため)

	if (save_only_once)
	{
		// EVAL_SAVE_ONLY_ONCEが定義されているときは、
		// 1度だけの保存としたいのでサブフォルダを掘らない。
		Eval::save_eval("");
	}
	else if (is_final) {
		Eval::save_eval("final");
		return true;
	}
	else {
		static int dir_number = 0;
		const std::string dir_name = std::to_string(dir_number++);
		Eval::save_eval(dir_name);
#if defined(EVAL_NNUE)
		if (newbob_decay != 1.0 && latest_loss_count > 0) {
			static int trials = newbob_num_trials;
			const double latest_loss = latest_loss_sum / latest_loss_count;
			latest_loss_sum = 0.0;
			latest_loss_count = 0;
			cout << "loss: " << latest_loss;
			if (latest_loss < best_loss) {
				cout << " < best (" << best_loss << "), accepted" << endl;
				best_loss = latest_loss;
				best_nn_directory = Path::Combine((std::string)Options["EvalSaveDir"], dir_name);
				trials = newbob_num_trials;
			} else {
				cout << " >= best (" << best_loss << "), rejected" << endl;
				if (best_nn_directory.empty()) {
					cout << "WARNING: no improvement from initial model" << endl;
				} else {
					cout << "restoring parameters from " << best_nn_directory << endl;
					Eval::NNUE::RestoreParameters(best_nn_directory);
				}
				if (--trials > 0 && !is_final) {
					cout << "reducing learning rate scale from " << newbob_scale
					     << " to " << (newbob_scale * newbob_decay)
					     << " (" << trials << " more trials)" << endl;
					newbob_scale *= newbob_decay;
					Eval::NNUE::SetGlobalLearningRateScale(newbob_scale);
				}
			}
			if (trials == 0) {
				cout << "converged" << endl;
				return true;
			}
		}
#endif
	}
	return false;
}

// shuffle_files() , shuffle_files_quick()の下請けで、書き出し部分。
// output_file_name : 書き出すファイル名
// prng : 乱数
// afs  : それぞれの教師局面ファイルのfstream
// a_count : それぞれのファイルに内在する教師局面の数。
void shuffle_write(const string& output_file_name , PRNG& prng , vector<fstream>& afs , vector<u64>& a_count)
{
	u64 total_sfen_count = 0;
	for (auto c : a_count)
		total_sfen_count += c;

	// 書き出した局面数
	u64 write_sfen_count = 0;

	// 進捗をこの局面数ごとに画面に出力する。
	const u64 buffer_size = 10000000;

	auto print_status = [&]()
	{
		// 10M局面ごと、もしくは、すべての書き出しが終わったときに進捗を出力する
		if (((write_sfen_count % buffer_size) == 0) ||
			(write_sfen_count == total_sfen_count))
			cout << write_sfen_count << " / " << total_sfen_count << endl;
	};


	cout << endl <<  "write : " << output_file_name << endl;

	fstream fs(output_file_name, ios::out | ios::binary);

	// 教師局面の合計
	u64 sum = 0;
	for (auto c : a_count)
		sum += c;

	while (sum != 0)
	{
		auto r = prng.rand(sum);

		// fs[0]のファイルに格納されている局面 ... fs[1]のファイルに格納されている局面 ...
		// のようにひと続きになっているものと考えて、rがどのファイルに格納されている局面を指しているかを確定させる。
		// ファイルの中身はシャッフルされているので、そのファイルから次の要素を1つ取ってくれば良い。
		// それぞれのファイルにはa_count[x]ずつ局面が残っているので、この処理は以下のように書ける。

		u64 n = 0;
		while (a_count[n] <= r)
			r -= a_count[n++];

		// これでnが確定した。忘れないうちに残り件数を減らしておく。

		--a_count[n];
		--sum;

		PackedSfenValue psv;
		// これ、パフォーマンスあんまりよくないまでまとめて読み書きしたほうが良いのだが…。
		if (afs[n].read((char*)&psv, sizeof(PackedSfenValue)))
		{
			fs.write((char*)&psv, sizeof(PackedSfenValue));
			++write_sfen_count;
			print_status();
		}
	}
	print_status();
	fs.close();
	cout << "done!" << endl;
}

// 教師局面のシャッフル "learn shuffle"コマンドの下請け。
// output_file_name : シャッフルされた教師局面が書き出される出力ファイル名
void shuffle_files(const vector<string>& filenames , const string& output_file_name , u64 buffer_size )
{
	// 出力先のフォルダは
	// tmp/               一時書き出し用

	// テンポラリファイルはbuffer_size局面ずつtmp/フォルダにいったん書き出す。
	// 例えば、buffer_size = 20Mならば 20M*40bytes = 800MBのバッファが必要。
	// メモリが少ないPCでは、ここを減らすと良いと思う。
	// ただし、あまりファイル数が増えるとOSの制限などから同時にopen出来なくなる。
	// Windowsだと1プロセス512という制約があったはずなので、ここでopen出来るのが500として、
	// 現在の設定で500ファイル×20M = 10G = 100億局面が限度。

	PSVector buf;
	buf.resize(buffer_size);
	// ↑のバッファ、どこまで使ったかを示すマーカー
	u64 buf_write_marker = 0;

	// 書き出すファイル名(連番なのでインクリメンタルカウンター)
	u64 write_file_count = 0;

	// シャッフルするための乱数
	PRNG prng;

	// テンポラリファイルの名前を生成する
	auto make_filename = [](u64 i)
	{
		return "tmp/" + to_string(i) + ".bin";
	};

	// 書き出したtmp/フォルダのファイル、それぞれに格納されている教師局面の数
	vector<u64> a_count;

	auto write_buffer = [&](u64 size)
	{
		// buf[0]～buf[size-1]までをshuffle
		for (u64 i = 0; i < size; ++i)
			swap(buf[i], buf[(u64)(prng.rand(size - i) + i)]);

		// ファイルに書き出す
		fstream fs;
		fs.open(make_filename(write_file_count++), ios::out | ios::binary);
		fs.write((char*)buf.data(), size * sizeof(PackedSfenValue));
		fs.close();
		a_count.push_back(size);

		buf_write_marker = 0;
		cout << ".";
	};

	Directory::CreateFolder("tmp");

	// 10M局面の細切れファイルとしてシャッフルして書き出す。
	for (auto filename : filenames)
	{
		fstream fs(filename, ios::in | ios::binary);
		cout << endl << "open file = " << filename;
		while (fs.read((char*)&buf[buf_write_marker], sizeof(PackedSfenValue)))
			if (++buf_write_marker == buffer_size)
				write_buffer(buffer_size);

		// sizeof(PackedSfenValue)単位で読み込んでいき、
		// 最後に残っている端数は無視する。(fs.readで失敗するのでwhileを抜ける)
		// (最後に残っている端数は、教師生成時に途中で停止させたために出来た中途半端なデータだと思われる。)

	}

	if (buf_write_marker != 0)
		write_buffer(buf_write_marker);

	// シャッフルされたファイルがwrite_file_count個だけ書き出された。
	// 2pass目として、これをすべて同時にオープンし、ランダムに1つずつ選択して1局面ずつ読み込めば
	// これにてシャッフルされたことになる。

	// シャツフルする元ファイル+tmpファイル+書き出すファイルで元ファイルの3倍のストレージ容量が必要になる。
	// 100億局面400GBなのでシャッフルするために1TBのSSDでは足りない。
	// tmpに書き出しが終わったこのタイミングで元ファイルを消す(あるいは手で削除してしまう)なら、
	// 元ファイルの2倍程度のストレージ容量で済む。
	// だから、元ファイルを消すためのオプションを用意すべきかも知れない。

	// ファイルの同時openをしている。これがFOPEN_MAXなどを超える可能性は高い。
	// その場合、buffer_sizeを調整して、ファイルの数を減らすよりない。

	vector<fstream> afs;
	for (u64 i = 0; i < write_file_count; ++i)
		afs.emplace_back(fstream(make_filename(i),ios::in | ios::binary));

	// 下請け関数に丸投げして終わり。
	shuffle_write(output_file_name, prng, afs, a_count);
}

// 教師局面のシャッフル "learn shuffleq"コマンドの下請け。
// こちらは1passで書き出す。
// output_file_name : シャッフルされた教師局面が書き出される出力ファイル名
void shuffle_files_quick(const vector<string>& filenames, const string& output_file_name)
{
	// 読み込んだ局面数
	u64 read_sfen_count = 0;

	// シャッフルするための乱数
	PRNG prng;

	// ファイルの数
	size_t file_count = filenames.size();

	// filenamesのファイルそれぞれに格納されている教師局面の数
	vector<u64> a_count(file_count);

	// それぞれのファイルの教師局面の数をカウントする。
	vector<fstream> afs(file_count);

	for (size_t i = 0; i < file_count ; ++i)
	{
		auto filename = filenames[i];
		auto& fs = afs[i];

		fs.open(filename, ios::in | ios::binary);
		fs.seekg(0, fstream::end);
		u64 eofPos = (u64)fs.tellg();
		fs.clear(); // これをしないと次のseekに失敗することがある。
		fs.seekg(0, fstream::beg);
		u64 begPos = (u64)fs.tellg();
		u64 file_size = eofPos - begPos;
		u64 sfen_count = file_size / sizeof(PackedSfenValue);
		a_count[i] = sfen_count;

		// 各ファイルに格納されていたsfenの数を出力する。
		cout << filename << " = " << sfen_count << " sfens." << endl;
	}

	// それぞれのファイルのファイルサイズがわかったので、
	// これらをすべて同時にオープンし(すでにオープンされている)、
	// ランダムに1つずつ選択して1局面ずつ読み込めば
	// これにてシャッフルされたことになる。

	// 下請け関数に丸投げして終わり。
	shuffle_write(output_file_name, prng, afs, a_count);
}

// 教師局面のシャッフル "learn shufflem"コマンドの下請け。
// メモリに丸読みして指定ファイル名で書き出す。
void shuffle_files_on_memory(const vector<string>& filenames,const string output_file_name)
{
	PSVector buf;

	for (auto filename : filenames)
	{
		std::cout << "read : " << filename << std::endl;
		SystemIO::ReadFileToMemory(filename, [&buf](u64 size) {
			ASSERT_LV1((size % sizeof(PackedSfenValue)) == 0);
			// バッファを拡充して、前回の末尾以降に読み込む。
			u64 last = buf.size();
			buf.resize(last + size / sizeof(PackedSfenValue));
			return (void*)&buf[last];
		});
	}

	// buf[0]～buf[size-1]までをshuffle
	PRNG prng;
	u64 size = (u64)buf.size();
	std::cout << "shuffle buf.size() = " << size << std::endl;
	for (u64 i = 0; i < size; ++i)
		swap(buf[i], buf[(u64)(prng.rand(size - i) + i)]);

	std::cout << "write : " << output_file_name << endl;

	// 書き出すファイルが2GBを超えるとfstream::write一発では書き出せないのでwrapperを用いる。
	SystemIO::WriteMemoryToFile(output_file_name, (void*)&buf[0], (u64)sizeof(PackedSfenValue)*(u64)buf.size());

	std::cout << "..shuffle_on_memory done." << std::endl;
}

void convert_bin(const vector<string>& filenames , const string& output_file_name)
{
	std::fstream fs;
	auto th = Threads.main();
	auto &tpos = th->rootPos;
	// plain形式の雑巾をやねうら王用のpackedsfenvalueに変換する
	fs.open(output_file_name, ios::app | ios::binary);

	for (auto filename : filenames) {
		std::cout << "convert " << filename << " ... ";
		std::string line;
		ifstream ifs;
		ifs.open(filename);
		PackedSfenValue p;
		p.gamePly = 1; // apery形式では含まれない。一応初期化するべし
		while (std::getline(ifs, line)) {
			std::stringstream ss(line);
			std::string token;
			std::string value;
			ss >> token;
			if (token == "sfen") {
				StateInfo si;
				tpos.set(line.substr(5), &si, Threads.main());
				tpos.sfen_pack(p.sfen);
			}
			else if (token == "move") {
				ss >> value;
				p.move = USI::to_move16(value).to_u16();
			}
			else if (token == "score") {
				ss >> p.score;
			}
			else if (token == "ply") {
				int temp;
				ss >> temp;
				p.gamePly = u16(temp); // 此処のキャストいらない？
			}
			else if (token == "result") {
				int temp;
				ss >> temp;
				p.game_result = s8(temp); // 此処のキャストいらない？
			}
			else if (token == "e") {
				fs.write((char*)&p, sizeof(PackedSfenValue));
				// debug
				/*
				std::cout<<tpos<<std::endl;
				std::cout<<to_usi_string(Move(p.move))<<","<<p.score<<","<<int(p.gamePly)<<","<<int(p.game_result)<<std::endl;
				*/
			}
		}
		std::cout << "done" << std::endl;
		ifs.close();
	}
	std::cout << "all done" << std::endl;
	fs.close();
}
  
void convert_plain(const vector<string>& filenames , const string& output_file_name)
{
	Position tpos;
	std::ofstream ofs;
	ofs.open(output_file_name, ios::app);
	for (auto filename : filenames) {
		std::cout << "convert " << filename << " ... ";

		// ひたすらpackedsfenvalueをテキストに変換する
		std::fstream fs;
		fs.open(filename, ios::in | ios::binary);
		PackedSfenValue p;
		while (true)
		{
			if (fs.read((char*)&p, sizeof(PackedSfenValue))) {
				// plain textとして書き込む
				ofs << "sfen " << tpos.sfen_unpack(p.sfen) << std::endl;
				ofs << "move " << to_usi_string(Move(p.move)) << std::endl;
				ofs << "score " << p.score << std::endl;
				ofs << "ply " << int(p.gamePly) << std::endl;
				ofs << "result " << int(p.game_result) << std::endl;
				ofs << "e" << std::endl;
			}
			else {
				break;
			}
		}
		fs.close();
		std::cout << "done" << std::endl;
	}
	ofs.close();
	std::cout << "all done" << std::endl;
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
	double eta1 = 0.0;
	double eta2 = 0.0;
	double eta3 = 0.0;
	u64 eta1_epoch = 0; // defaultではeta2は適用されない
	u64 eta2_epoch = 0; // defaultではeta3は適用されない

#if defined(USE_GLOBAL_OPTIONS)
	// あとで復元するために保存しておく。
	auto oldGlobalOptions = GlobalOptions;
	// eval hashにhitするとrmseなどの計算ができなくなるのでオフにしておく。
	GlobalOptions.use_eval_hash = false;
	// 置換表にhitするとそこで以前の評価値で枝刈りがされることがあるのでオフにしておく。
	GlobalOptions.use_hash_probe = false;
#endif

	// --- 教師局面をシャッフルするだけの機能

	// 通常シャッフル
	bool shuffle_normal = false;
	u64 buffer_size = 20000000;
	// それぞれのファイルがシャッフルされていると仮定しての高速シャッフル
	bool shuffle_quick = false;
	// メモリにファイルを丸読みしてシャッフルする機能。(要、ファイルサイズのメモリ)
	bool shuffle_on_memory = false;
	// packed sfenの変換。plainではsfen(string), 評価値(整数), 指し手(例：7g7f, string)、結果(負け-1、勝ち1、引き分け0)からなる
	bool use_convert_plain = false;
	// plain形式の教師をやねうら王のbinに変換する
	bool use_convert_bin = false;
	// それらのときに書き出すファイル名(デフォルトでは"shuffled_sfen.bin")
	string output_file_name = "shuffled_sfen.bin";

	// 教師局面の深い探索での評価値の絶対値が、この値を超えていたらその局面は捨てる。
	int eval_limit = 32000;

	// 評価関数ファイルの保存は終了間際の1回に限定するかのフラグ。
	bool save_only_once = false;

	// 教師局面を先読みしている分に関してシャッフルする。(1000万局面単位ぐらいのシャッフル)
	// 事前にシャッフルされているファイルを渡すならオンにすれば良い。
	bool no_shuffle = false;

#if defined (LOSS_FUNCTION_IS_ELMO_METHOD)
	// elmo lambda
	ELMO_LAMBDA = 0.33;
	ELMO_LAMBDA2 = 0.33;
	ELMO_LAMBDA_LIMIT = 32000;
#endif

	// 割引率。これを0以外にすると、PV終端以外でも勾配を加算する。(そのとき、この割引率を適用する)
	double discount_rate = 0;

	// if (gamePly < rand(reduction_gameply)) continue;
	// のようにして、序盤を学習対象から程よく除外するためのオプション
	// 1にしてあるとrand(1)==0なので、何も除外されない。
	int reduction_gameply = 1;

	// KK/KKP/KPP/KPPPを学習させないオプション項目
	array<bool,4> freeze = {};

#if defined(EVAL_NNUE)
	u64 nn_batch_size = 1000;
	double newbob_decay = 1.0;
	int newbob_num_trials = 2;
	string nn_options;
#endif

	u64 eval_save_interval = LEARN_EVAL_SAVE_INTERVAL;
	u64 loss_output_interval = 0;
	u64 mirror_percentage = 0;

	string validation_set_file_name;

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
		else if (option == "targetdir") is >> target_dir;

		// ループ回数の指定
		else if (option == "loop")      is >> loop;

		// 棋譜ファイル格納フォルダ(ここから相対pathで棋譜ファイルを取得)
		else if (option == "basedir")   is >> base_dir;

		// ミニバッチのサイズ
		else if (option == "batchsize") is >> mini_batch_size;

		// 学習率
		else if (option == "eta")        is >> eta1;
		else if (option == "eta1")       is >> eta1; // alias
		else if (option == "eta2")       is >> eta2;
		else if (option == "eta3")       is >> eta3;
		else if (option == "eta1_epoch") is >> eta1_epoch;
		else if (option == "eta2_epoch") is >> eta2_epoch;

		// 割引率
		else if (option == "discount_rate") is >> discount_rate;

		// KK/KKP/KPP/KPPPの学習なし。
		else if (option == "freeze_kk")    is >> freeze[0];
		else if (option == "freeze_kkp")   is >> freeze[1];
		else if (option == "freeze_kpp")   is >> freeze[2];

#if defined (LOSS_FUNCTION_IS_ELMO_METHOD)
		// LAMBDA
		else if (option == "lambda")       is >> ELMO_LAMBDA;
		else if (option == "lambda2")      is >> ELMO_LAMBDA2;
		else if (option == "lambda_limit") is >> ELMO_LAMBDA_LIMIT;

#endif
		else if (option == "reduction_gameply") is >> reduction_gameply;

		// シャッフル関連
		else if (option == "shuffle")	shuffle_normal = true;
		else if (option == "buffer_size") is >> buffer_size;
		else if (option == "shuffleq")	shuffle_quick = true;
		else if (option == "shufflem")	shuffle_on_memory = true;
		else if (option == "output_file_name") is >> output_file_name;

		else if (option == "eval_limit") is >> eval_limit;
		else if (option == "save_only_once") save_only_once = true;
		else if (option == "no_shuffle") no_shuffle = true;

#if defined(EVAL_NNUE)
		else if (option == "nn_batch_size") is >> nn_batch_size;
		else if (option == "newbob_decay") is >> newbob_decay;
		else if (option == "newbob_num_trials") is >> newbob_num_trials;
		else if (option == "nn_options") is >> nn_options;
#endif
		else if (option == "eval_save_interval") is >> eval_save_interval;
		else if (option == "loss_output_interval") is >> loss_output_interval;
		else if (option == "mirror_percentage") is >> mirror_percentage;
		else if (option == "validation_set_file_name") is >> validation_set_file_name;
		
		// 雑巾のconvert関連
		else if (option == "convert_plain") use_convert_plain = true;
		else if (option == "convert_bin") use_convert_bin = true;
		// さもなくば、それはファイル名である。
		else
			filenames.push_back(option);
	}
	if (loss_output_interval == 0)
		loss_output_interval = LEARN_RMSE_OUTPUT_INTERVAL * mini_batch_size;

	cout << "learn command , ";

	// OpenMP無効なら警告を出すように。
#if !defined(_OPENMP)
	cout << "Warning! OpenMP disabled." << endl;
#endif

	// 学習棋譜ファイルの表示
	if (target_dir != "")
	{
		string kif_base_dir = Path::Combine(base_dir, target_dir);
		
		// このフォルダのファイルを根こそぎ取る。base_dir相対にしておく。
		filenames = Directory::EnumerateFiles(kif_base_dir, ".bin");
	}

	cout << "learn from ";
	for (auto s : filenames)
		cout << s << " , ";
	cout << endl;
	if (!validation_set_file_name.empty())
	{
		cout << "validation set  : " << validation_set_file_name << endl;
	}

	cout << "base dir        : " << base_dir   << endl;
	cout << "target dir      : " << target_dir << endl;

	// シャッフルモード
	if (shuffle_normal)
	{
		cout << "buffer_size     : " << buffer_size << endl;
		cout << "shuffle mode.." << endl;
		shuffle_files(filenames,output_file_name , buffer_size);
		return;
	}
	if (shuffle_quick)
	{
		cout << "quick shuffle mode.." << endl;
		shuffle_files_quick(filenames, output_file_name);
		return;
	}
	if (shuffle_on_memory)
	{
		cout << "shuffle on memory.." << endl;
		shuffle_files_on_memory(filenames,output_file_name);
		return;
	}
	if (use_convert_plain)
	{
	  	is_ready(true);
		cout << "convert_plain.." << endl;
		convert_plain(filenames,output_file_name);
		return;
		
	}
	if (use_convert_bin)
	{
	  	is_ready(true);
		cout << "convert_bin.." << endl;
		convert_bin(filenames,output_file_name);
		return;
		
	}

	cout << "loop              : " << loop << endl;
	cout << "eval_limit        : " << eval_limit << endl;
	cout << "save_only_once    : " << (save_only_once ? "true" : "false") << endl;
	cout << "no_shuffle        : " << (no_shuffle ? "true" : "false") << endl;

	// ループ回数分だけファイル名を突っ込む。
	for (int i = 0; i < loop; ++i)
		// sfen reader、逆順で読むからここでreverseしておく。すまんな。
		for (auto it = filenames.rbegin(); it != filenames.rend(); ++it)
			sr.filenames.push_back(Path::Combine(base_dir, *it));
			
#if !defined(EVAL_NNUE)
	cout << "Gradient Method   : " << LEARN_UPDATE      << endl;
#endif
	cout << "Loss Function     : " << LOSS_FUNCTION     << endl;
	cout << "mini-batch size   : " << mini_batch_size   << endl;
#if defined(EVAL_NNUE)
	cout << "nn_batch_size     : " << nn_batch_size     << endl;
	cout << "nn_options        : " << nn_options        << endl;
#endif
	cout << "learning rate     : " << eta1 << " , " << eta2 << " , " << eta3 << endl;
	cout << "eta_epoch         : " << eta1_epoch << " , " << eta2_epoch << endl;
#if defined(EVAL_NNUE)
	if (newbob_decay != 1.0) {
		cout << "scheduling        : newbob with decay = " << newbob_decay
		     << ", " << newbob_num_trials << " trials" << endl;
	} else {
		cout << "scheduling        : default" << endl;
	}
#endif
	cout << "discount rate     : " << discount_rate     << endl;

	// reduction_gameplyに0を設定されるとrand(0)が0除算になってしまうので1に補正。
	reduction_gameply = max(reduction_gameply, 1);
	cout << "reduction_gameply : " << reduction_gameply << endl;

#if defined (LOSS_FUNCTION_IS_ELMO_METHOD)
	cout << "LAMBDA            : " << ELMO_LAMBDA       << endl;
	cout << "LAMBDA2           : " << ELMO_LAMBDA2      << endl;
	cout << "LAMBDA_LIMIT      : " << ELMO_LAMBDA_LIMIT << endl;
#endif
	cout << "mirror_percentage : " << mirror_percentage << endl;
	cout << "eval_save_interval  : " << eval_save_interval << " sfens" << endl;
	cout << "loss_output_interval: " << loss_output_interval << " sfens" << endl;

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
	cout << "freeze_kk/kkp/kpp      : " << freeze[0] << " , " << freeze[1] << " , " << freeze[2] << endl;
#endif

	// -----------------------------------
	//            各種初期化
	// -----------------------------------

	cout << "init.." << endl;

	// 評価関数パラメーターの読み込み
	is_ready(true);

#if !defined(EVAL_NNUE)
	cout << "init_grad.." << endl;

	// 評価関数パラメーターの勾配配列の初期化
	Eval::init_grad(eta1,eta1_epoch,eta2,eta2_epoch,eta3);
#else
	cout << "init_training.." << endl;
	Eval::NNUE::InitializeTraining(eta1,eta1_epoch,eta2,eta2_epoch,eta3);
	Eval::NNUE::SetBatchSize(nn_batch_size);
	Eval::NNUE::SetOptions(nn_options);
	if (newbob_decay != 1.0 && !Options["SkipLoadingEval"]) {
		learn_think.best_nn_directory = std::string(Options["EvalDir"]);
	}
#endif

#if 0
	// 平手の初期局面に対して1.0の勾配を与えてみるテスト。
	pos.set_hirate();
	cout << Eval::evaluate(pos) << endl;
	//Eval::print_eval_stat(pos);
	Eval::add_grad(pos, BLACK, 32.0 , false);
	Eval::update_weights(1);
	pos.state()->sum.p[2][0] = VALUE_NOT_EVALUATED;
	cout << Eval::evaluate(pos) << endl;
	//Eval::print_eval_stat(pos);
#endif

	cout << "init done." << endl;

	// その他、オプション設定を反映させる。
	learn_think.discount_rate = discount_rate;
	learn_think.eval_limit = eval_limit;
	learn_think.save_only_once = save_only_once;
	learn_think.sr.no_shuffle = no_shuffle;
	learn_think.freeze = freeze;
	learn_think.reduction_gameply = reduction_gameply;
#if defined(EVAL_NNUE)
	learn_think.newbob_scale = 1.0;
	learn_think.newbob_decay = newbob_decay;
	learn_think.newbob_num_trials = newbob_num_trials;
#endif
	learn_think.eval_save_interval = eval_save_interval;
	learn_think.loss_output_interval = loss_output_interval;
	learn_think.mirror_percentage = mirror_percentage;

	// 局面ファイルをバックグラウンドで読み込むスレッドを起動
	// (これを開始しないとmseの計算が出来ない。)
	learn_think.start_file_read_worker();

	learn_think.mini_batch_size = mini_batch_size;

	if (validation_set_file_name.empty()) {
		// mse計算用にデータ1万件ほど取得しておく。
		sr.read_for_mse();
	} else {
		// base_dirの指定を"validation_set_file_name"オプションにも反映させるべきか..
		//validation_set_file_name = Path::Combine(base_dir, validation_set_file_name);
		sr.read_validation_set(validation_set_file_name, eval_limit);
	}

	// この時点で一度rmseを計算(0 sfenのタイミング)
	// sr.calc_rmse();
#if defined(EVAL_NNUE)
	if (newbob_decay != 1.0) {
		learn_think.calc_loss(0, -1);
		learn_think.best_loss = learn_think.latest_loss_sum / learn_think.latest_loss_count;
		learn_think.latest_loss_sum = 0.0;
		learn_think.latest_loss_count = 0;
		cout << "initial loss: " << learn_think.best_loss << endl;
	}
#endif

	// -----------------------------------
	//   評価関数パラメーターの学習の開始
	// -----------------------------------

	// 学習開始。
	learn_think.go_think();

	// 最後に一度保存。
	learn_think.save(true);

#if defined(USE_GLOBAL_OPTIONS)
	// GlobalOptionsの復元。
	GlobalOptions = oldGlobalOptions;
#endif
}


} // namespace Learner
} // namespace YaneuraOu

#if defined(EVAL_LEARN) && defined(GENSFEN2019)

//
// 教師局面の生成ルーチン2019年度版
//

#include <sstream>
#include <unordered_set>
#include "multi_think.h"

using namespace std;

namespace YaneuraOu {
namespace {

	// C#のstring.Split()みたいなの
	vector<string> split(const string &s, char delim) {
		vector<string> elems;
		stringstream ss(s);
		string item;
		while (getline(ss, item, delim)) {
		if (!item.empty()) {
				elems.push_back(item);
			}
		}
		return elems;
	}
} // namespace

namespace Learner {

	// -----------------------------------
	//  棋譜を生成するworker(スレッドごと)
	// -----------------------------------

	// 複数スレッドでsfenを生成するためのクラス
	struct MultiThinkGenSfen2019 : public MultiThink
	{
		// hash_size : NodeInfoを格納するためのhash sizeを指定する。単位は[MB]。
		// メモリに余裕があるなら大きめの値を指定するのが好ましい。
		MultiThinkGenSfen2019(SfenWriter& sw_ , int search_depth_ , u64 nodes_limit_ , const string& book_file_name_)
		: sw(sw_) , search_depth(search_depth_) , nodes_limit(nodes_limit_) , book_file_name(book_file_name_){}

		// コンストラクタとは別に初期化用のコード。(write_maxplyなどを設定後に呼び出す)
		// このタイミングで定跡ファイルから読み込む
		void init()
		{
			// PCを並列化してgensfenするときに同じ乱数seedを引いていないか確認用の出力。
			std::cout << endl << prng << std::endl;

			cout << "read book" << endl;
			if (SystemIO::ReadAllLines(book_file_name, my_book).is_not_ok())
			{
				cout << endl << "info string Error! read book error!";
				// 定跡ファイルがないと、開始局面に困るのでこの時点でexitする。				
				exit(0);
			}
			else
			{
				// 丸読みして、局面に落とし込む＆重複除去する
				cout << "..done" << endl;

				parse_book_file();
			}
		}

		virtual void thread_worker(size_t thread_id);
		void start_file_write_worker() { sw.start_file_write_worker(); }

		// 読み込んだ定跡ファイルをparseして各局面を取得する。
		void parse_book_file();

		// 開始局面をランダムに一つ選択する。
		void set_start_pos(Position&pos, Thread& th , StateInfo* si);

		// 1手進める関数
		void do_move(Position& pos , Move move, StateInfo* states)
		{
			ASSERT_LV3(move.is_ok() && pos.pseudo_legal(move) && pos.legal(move));

			pos.do_move(move, states[pos.game_ply()]);

			ASSERT_LV3(pos.pos_is_ok());

			//			Eval::evaluate_with_no_return(pos);
			Eval::evaluate(pos);

		};

		// 生成する局面の評価値の上限
		int eval_limit;

		// 書き出す局面のply(初期局面からの手数)の最大。
		int write_minply;
		int write_maxply;

		// 探索ノード数
		u64 nodes_limit;

		// 探索depth
		int search_depth;

		// 定跡ファイル名
		string book_file_name;

		// sfenの書き出し器
		SfenWriter& sw;

		// 定跡
		vector<string> my_book;

		// 定跡の各局面
		vector<PackedSfenValue> my_book_sfens;
	};

	void MultiThinkGenSfen2019::parse_book_file()
	{
		// -- 定跡の各局面

		// unordered_setで用いるhashとequal関数

		struct PackedSfenValueHash {
			size_t operator()(const PackedSfenValue & s) const {
				// packされたバイナリの全部の値をxorして返す程度でいいや…。
				size_t tmp = 0;
				for(int i=0;i<(int)(sizeof(PackedSfen) / sizeof(size_t)) ;++i)
					tmp ^= ((size_t*)&s.sfen.data)[i];
				return tmp;
			}
		};
		struct PackedSfenValueEqual {
			bool operator()(const PackedSfenValue &left, const PackedSfenValue&right) const
			{
				// 局面が一致すればあとは無視する。
				return memcmp(&left.sfen, &right.sfen, sizeof(PackedSfen)) == 0;
			}
		};

		// unordered_setを用いて局面の重複除去を行う。
		unordered_set<PackedSfenValue, PackedSfenValueHash , PackedSfenValueEqual> book_sfens;

		// -- 1手進める関数

		Position pos;
		auto th = Threads.main();

		const int MAX_PLY2 = write_maxply;
		std::vector<StateInfo> states_(MAX_PLY2 + MAX_PLY /* == search_depth + α */);
		StateInfo* const states = &states_[0];

		Move move;
		u64 count = 0; // 局面数
		u64 line_number = 0; // 定跡ファイル行番号

		auto my_do_move = [&move, &pos, &states ,&count , &line_number , &book_sfens ]()
		{
			ASSERT_LV3(move.is_ok() && pos.pseudo_legal(move) && pos.legal(move));

			ASSERT_LV3(pos.game_ply() != 0);
			pos.do_move(move, states[pos.game_ply()]);

			ASSERT_LV3(pos.pos_is_ok());

			// 評価値使わないので、評価関数の計算しなくていいや。
//			Eval::evaluate(pos);

			// 局面の保存(手数も保存しておかないといけない)
			PackedSfenValue ps;
			pos.sfen_pack(ps.sfen);
			ps.gamePly = pos.game_ply();
			ASSERT_LV3(ps.gamePly != 0);

			// すでに挿入済であればこの局面は無視する。
			if (book_sfens.find(ps) != book_sfens.end())
				return;

			book_sfens.insert(ps);
			++count;
		};

		auto out_status = [&count,&line_number]
		{
			cout << count << " positions , line_number = " << line_number << endl;
		};

		ASSERT_LV3(Search::Limits.enteringKingRule = EKR_27_POINT);

		for (auto book_line : my_book)
		{
			if ((++line_number % 1000) == 0)
				out_status();

			auto book_moves = split(book_line, ' ');

			pos.set_hirate(&states[0]);

			// "startpos moves"を読み飛ばしてそこ以降の指し手文字列で指し手を進める
			for (int book_move_index = 2; book_move_index < (int)book_moves.size()
					&& pos.game_ply() <= MAX_PLY2 - 32 /* あまり直前の局面だと即シミュレーションが終了してしまうので… */
					; ++book_move_index)
			{
				// /* 詰みの局面もゴミでしかない。1手詰め、宣言勝ちの局面も除外。*/
				if (pos.is_mated()
					|| (!pos.checkers() && Mate::mate_1ply(pos) != Move::none())
					|| pos.DeclarationWin() != Move::none()
					)
					break;

				// 定跡の指し手で一手進める
				auto book_move = book_moves[book_move_index];
				move = USI::to_move(pos, book_move);
				// 信用できない定跡の場合、このチェックが必要。
				if (!move.is_ok() || !pos.pseudo_legal(move) || !pos.legal(move))
					break;

				my_do_move();

#if 1
				// 32手目までとする。
				// ・Apery(SDT5)は手数制限をしていないらしい。
				// ・tanuki-(2018)は、手数制限をしているらしい。
				// 手数制限をしないと終盤の局面に偏ってしまうように思うのだが…。
				if (pos.game_ply() > 32)
					break;
#endif
			}
		}

		// vectorに局面をcopy
		my_book_sfens.clear();
		for(auto& it : book_sfens)
			my_book_sfens.push_back(it);

		out_status();
	}

	void MultiThinkGenSfen2019::set_start_pos(Position&pos, Thread& th , StateInfo* states)
	{
	Retry:;

		// 定跡の局面を一つ取り出す
		auto& ps = my_book_sfens[prng.rand(my_book_sfens.size())];
		ASSERT_LV3(ps.gamePly != 0);
		pos.set_from_packed_sfen(ps.sfen , &states[0 /* ここは確実に空いてる */] , &th , /*mirror = */ false , ps.gamePly);

		// ランダムムーブで1手進める
		// 実現確率が高い局面の周辺局面ということならランダムムーブ1手がベスト

		// ランダムムーブの手数
		const int random_move_ply = 2;

		for(int i=0;i< random_move_ply;++i)
		{
			Move move = Move::none();
			MoveList<LEGAL> legal_moves(pos);
			if (legal_moves.size() == 0)
				goto Retry;
				// なぜか合法手がないので局面の選択に戻る。

#if 0
			// 1/2の確率で玉を移動させる指し手を選択する。(Apery(SDT5)のアイデア)
			// 玉が移動している局面を開始局面にしたほうがhalfKPなどでは0になる要素が減って良いと考えられる。
			if (prng.rand(2) == 0)
			{
				vector<Move> moves;
				for (auto m : legal_moves)
				{
					if (!is_drop(m.move) && type_of(pos.piece_on(from_sq(m.move))) == KING)
						moves.push_back(m);
				}

				if (moves.size())
				{
					// 玉を移動させる指し手があったので、このなかから指し手を採用する。
					move = moves.at(prng.rand(moves.size()));
				}
			}
#endif

			// 玉を移動する指し手ではなかったので全合法手のなかから指し手を選択する。
			if (move == Move::none())
				move = Move(legal_moves.at(prng.rand(legal_moves.size())));

			do_move(pos, move, states);

			// 詰みの局面、1手詰めの局面を除外
			if (pos.is_mated()
				|| (!pos.checkers() && Mate::mate_1ply(pos) != Move::none())
				|| pos.DeclarationWin() != Move::none()
				)
				goto Retry;
		}

		// 局面の生成に成功したのでこれにて終了。
	}

	//  thread_id    = 0..Threads.size()-1
	void MultiThinkGenSfen2019::thread_worker(size_t thread_id)
	{
		// とりあえず、書き出す手数の最大のところで引き分け扱いになるものとする。
		const int MAX_PLY2 = write_maxply;

		// StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
		// leaf nodeに行くのであれば、search_depth分ぐらいは必要。
		std::vector<StateInfo> states_(MAX_PLY2 + MAX_PLY /* == search_depth + α */);
		StateInfo* const states = &states_[0];

		// Positionに対して従属スレッドの設定が必要。
		// 並列化するときは、Threads (これが実体が vector<Thread*>なので、
		// Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
		auto& th = *Threads[thread_id];

		auto& pos = th.rootPos;

		// 終了フラグ
		bool quit = false;

		Move move;

		// 1局分の局面を保存しておき、終局のときに勝敗を含めて書き出す。
		PSVector a_psv;
		a_psv.reserve(MAX_PLY2 + MAX_PLY);

		// 対局シミュレーションのループ
		// 規定回数の局面を書き出すまで繰り返し
		while (!quit)
		{
			// -- 1局分スタート

			// 自分スレッド用の置換表があるはずなので自分の置換表だけをクリアする。
			th.tt.clear();

			// 局面の初期化
			set_start_pos(pos, th , states);

			// 局面バッファのクリア
			a_psv.clear();

			Value lastValue = VALUE_NONE;

			/* 本局の探索ノード数。平均5%のゆらぎ。これで指し手をある程度ばらつかせる。
				本局を通じたNodes数なので、シミュレーションの精度への影響はない。
				あまり大きくすると勝敗項に対するノイズになりかねないので自重して10%に留める。
			*/
			// u64 nodes = nodes_limit + (nodes_limit * prng.rand(100) / 1000);

			// →　ノイズになるのでノードは固定しておき、置換表をスレッド間で共有することにより揺らぎをもたせる。
			u64 nodes = nodes_limit;

			// 対局シミュレーションのループ
			while (pos.game_ply() < MAX_PLY2
				&& !pos.is_mated() && pos.DeclarationWin() == Move::none()
				&& pos.is_repetition() != REPETITION_DRAW /* 千日手 */)
			{
				// -- 普通に探索してその指し手で局面を進める。

				// NodesLimitで制限しているのでdepthは24ぐらいで問題ない。
				// しかし、ここをあまり大きくすると詰み周りの局面で延長がかかって、探索が終わらなくなる。(´ω｀)
				auto pv_value = search(pos, search_depth , /*multi_pv*/1 , nodes );

				lastValue = pv_value.first;
				auto& pv = pv_value.second;

				// eval_limitの値を超えていれば勝ち(or 負け)として扱うのでここで対局シミュレーションを終了。
				if (abs(lastValue) > eval_limit)
					break;

				// --- 局面の一時保存
					
				// 初期局面周辺は類似局面ばかりなので学習に用いると過学習になりかねない。

				if (write_minply <= pos.game_ply())
				{
					a_psv.emplace_back(PackedSfenValue());
					auto &psv = a_psv.back();

					// packを要求されているならpackされたsfenとそのときの評価値を書き出す。
					// 最終的な書き出しは、勝敗がついてから。
					pos.sfen_pack(psv.sfen);

					// PV leafのevaluate()の値とどちらが良いかはよくわからない。
					// PV leafの値だと詰みかけの局面で駒を捨ててて自分不利に見えるのが少し嫌。
					psv.score = (s16)lastValue;
					psv.gamePly = (u16)pos.game_ply();

					// この局面の手番を仮で入れる。この値はファイルに書き出すまでに書き換える。
					psv.game_result = (s8)pos.side_to_move();
					
					// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
					psv.move = pv[0].to_u16();
				}

				// search_depth手読みの指し手で局面を進める。
				// is_mated()ではないので、pv[0]として合法手が存在するはずなのだが..
				move = pv[0];
				do_move(pos,move,states);

			} // 対局シミュレーション終わり
			
			// lastValue == VALUE_NONEの場合は一度も探索していないということであり、
			// 書き出す局面がないはずであるから、以下の処理で問題ない。
			// ただ、その状態でこのwhileループに突入しているのがおかしくて…。
			ASSERT_LV3(lastValue != VALUE_NONE);

			// 勝利した側
			Color win;
			//RepetitionState repetition_state = pos.is_repetition(20);

			if (pos.is_mated()) {
				// 負け
				// 詰まされた
				win = ~pos.side_to_move();
			}
			else if (pos.DeclarationWin() != Move::none()) {
				// 勝ち
				// 入玉勝利
				win = pos.side_to_move();
			}
			else if (lastValue > eval_limit) {
				// 勝ち
				win = pos.side_to_move();
			}
			else if (lastValue < -eval_limit) {
				// 負け
				win = ~pos.side_to_move();
			}
			else {
				// それ以外は引き分け等なので書き出さない
				// 千日手も同様。
				continue;
			}

			// 各局面に関して、対局の勝敗の情報を付与しておく。
			// a_psvに保存されている局面は(手番的に)連続しているものとする。
			// 終局の局面(現在の局面)は書き出されていないことに注意すべき。
			for (auto& psv : a_psv)
			{
				// 局面を書き出そうと思ったら規定回数に達していた。
				// get_next_loop_count()内でカウンターを加算するので
				// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
				auto loop_count = get_next_loop_count();
				if (loop_count == UINT64_MAX)
				{
					// 終了フラグを立てておく。
					quit = true;
					break;
				}

				// この局面の手番側が仮でgame_resultに入っている。
				// 最後の局面の手番側の勝利であれば1 , 負けであれば -1 を入れる。
				auto stm = (Color)psv.game_result;
				psv.game_result = (stm == win) ? 1 : -1;

				//cout << (int)psv.game_result << endl;

				// 局面を一つ書き出す。
				sw.write(thread_id, psv);
			}

		} // while(!quit)

		sw.finalize(thread_id);
	}

	// gensfen2019コマンド本体
	void gen_sfen2019([[maybe_unused]] Position& pos, [[maybe_unused]] istringstream& is)
	{
		// スレッド数(これは、USIのsetoptionで与えられる)
		u32 thread_num = (u32)Options["Threads"];
		
		// 生成棋譜の個数 default = 80億局面(Ponanza仕様)
		u64 loop_max = 8000000000UL;

		// 評価値がこの値を超えたら生成を打ち切る。
		// デフォルトのこの値だと超えることはないので、評価値での打ち切りは無し。
		int eval_limit = 32000;

		// 探索深さ
		// NodesLimitで制限するが王手延長で延長されると探索終わらないので何らかの上限が必要。
		int search_depth = 24;
		
		// 探索ノード数
		u64 nodes_limit = 10000;

		// 書き出す局面のply(初期局面からの手数)の最小、最大。
		// 重複局面を除去するので初手から書き出して良いと思う。
		// ここの手数、あまり大きくすると入玉局面ばかりになり、引き分けになる確率が高いので無駄なシミュレーションになる。
		// ※　tanuki-(WCSC28)ではwrite_maxply == 400
		int write_minply = 1;
		int write_maxply = 300;

		// 使用する定跡ファイル。
		// この定跡ファイルの各局面から1局面を選んでランダムムーブで1手進めてから対局シミュレーションを開始する。
		string book_file_name = "book/flood2018.sfen";

		// 教師局面を書き出すファイル名
		string output_file_name = "generated_kifu.bin";

		string token;

		// eval hashにhitすると初期局面付近の評価値として、hash衝突して大きな値を書き込まれてしまうと
		// eval_limitが小さく設定されているときに初期局面で毎回eval_limitを超えてしまい局面の生成が進まなくなる。
		// そのため、eval hashは無効化する必要がある。
		// あとeval hashのhash衝突したときに、変な値の評価値が使われ、それを教師に使うのが気分が悪いというのもある。
		bool use_eval_hash = false;

		// この単位でファイルに保存する。
		// ファイル名は file_1.bin , file_2.binのように連番がつく。
		u64 save_every = UINT64_MAX;

		// ファイル名の末尾にランダムな数値を付与する。
		bool random_file_name = false;

		while (true)
		{
			token = "";
			is >> token;
			if (token == "")
				break;

			if (token == "loop")
				is >> loop_max;
			else if (token == "output_file_name")
				is >> output_file_name;
			else if (token == "eval_limit")
				is >> eval_limit;
			else if (token == "search_depth")
				is >> search_depth;
			else if (token == "write_minply")
				is >> write_minply;
			else if (token == "write_maxply")
				is >> write_maxply;
			else if (token == "nodes_limit")
				is >> nodes_limit;
			else if (token == "use_eval_hash")
				is >> use_eval_hash;
			else if (token == "save_every")
				is >> save_every;
			else if (token == "random_file_name")
				is >> random_file_name;
			else if (token == "book_file_name")
				is >> book_file_name;
			else
				cout << "Error! : Illegal token " << token << endl;
		}

#if defined(USE_GLOBAL_OPTIONS)
		// あとで復元するために保存しておく。
		auto oldGlobalOptions = GlobalOptions;
		GlobalOptions.use_eval_hash = use_eval_hash;
#endif

		if (random_file_name)
		{
			// output_file_nameにこの時点でランダムな数値を付与してしまう。
			PRNG r;
			// 念のため乱数振り直しておく。
			for (int i = 0; i<10; ++i)
				r.rand(1);
			auto to_hex = [](u64 u) {
				std::stringstream ss;
				ss << std::hex << u;
				return ss.str();
			};
			// 64bitの数値で偶然かぶると嫌なので念のため64bitの数値２つくっつけておく。
			output_file_name = output_file_name + "_" + to_hex(r.rand<u64>()) + to_hex(r.rand<u64>());
		}

		std::cout << "gensfen2019 : " << endl
			<< "  search_depth = " << search_depth << endl
			<< "  nodes_limit = " << nodes_limit << endl
			<< "  loop_max = " << loop_max << endl
			<< "  eval_limit = " << eval_limit << endl
			<< "  thread_num (set by USI setoption) = " << thread_num << endl
			<< "  write_minply            = " << write_minply << endl
			<< "  write_maxply            = " << write_maxply << endl
			<< "  output_file_name        = " << output_file_name << endl
			<< "  use_eval_hash           = " << use_eval_hash << endl
			<< "  save_every              = " << save_every << endl
			<< "  random_file_name        = " << random_file_name << endl
			<< "  book_file_name          = " << book_file_name << endl
			;

		// Options["Threads"]の数だけスレッドを作って実行。
		{
			SfenWriter sw(output_file_name, thread_num);
			sw.save_every = save_every;

			MultiThinkGenSfen2019 multi_think( sw , search_depth , nodes_limit , book_file_name);
			multi_think.set_loop_max(loop_max);
			multi_think.eval_limit = eval_limit;
			multi_think.write_minply = write_minply;
			multi_think.write_maxply = write_maxply;
			multi_think.start_file_write_worker();
			multi_think.go_think();

			// SfenWriterのデストラクタでjoinするので、joinが終わってから終了したというメッセージを
			// 表示させるべきなのでここをブロックで囲む。
		}

		std::cout << "gensfen2019 finished." << endl;

#if defined(USE_GLOBAL_OPTIONS)
		// GlobalOptionsの復元。
		GlobalOptions = oldGlobalOptions;
#endif

	}

} // namespace Learner
} // namespace YaneuraOu

#endif // defined(EVAL_LEARN) && defined(GENSFEN2019)

#endif // EVAL_LEARN
