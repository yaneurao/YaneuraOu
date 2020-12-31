#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

// -----------------------
//   やねうら王 標準探索部
// -----------------------

// 計測資料置き場 : https://github.com/yaneurao/YaneuraOu/blob/master/docs/%E8%A8%88%E6%B8%AC%E8%B3%87%E6%96%99.txt

// -----------------------
//   includes
// -----------------------

#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>	// std::log(),std::pow(),std::round()
#include <cstring>	// memset()

#include "../../position.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"
#include "../../book/book.h"
#include "../../movepick.h"
#include "../../usi.h"
#include "../../learn/learn.h"
#include "../../mate/mate.h"

// -------------------
// やねうら王独自追加
// -------------------

// パラメーターの調整を行うのか
#if defined(TUNING_SEARCH_PARAMETERS)
	// ハイパーパラメーターを調整するときは終了時にその時のパラメーターを書き出す。
	#define ENABLE_OUTPUT_GAME_RESULT

	// パラメーターをランダムに少し変化させる。
	// 探索パラメーターにstep分のランダム値を加えて対戦させるとき用。
	// 試合が終わったときに勝敗と、そのときに用いたパラメーター一覧をファイルに出力する。
	#define USE_RANDOM_PARAMETERS

	#define PARAM_DEFINE int
	#include "yaneuraou-param.h"
#else
	// 変更しないとき
	#define PARAM_DEFINE constexpr int
	#include "yaneuraou-param.h"

#endif

// 実行時に読み込むパラメーターファイルを配置するフォルダとその名前
#define PARAM_FILE "param/yaneuraou-param.h"

#if defined(ENABLE_OUTPUT_GAME_RESULT)
// 変更したパラメーター一覧と、リザルト(勝敗)を書き出すためのファイルハンドル
static std::fstream result_log;
#endif

// 定跡の指し手を選択するモジュール
Book::BookMoveSelector book;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	//   定跡設定

	book.init(o);

	// 弱くするために調整する。20なら手加減なし。0が最弱。
	o["SkillLevel"] << Option(20, 0, 20);

	// 引き分けを受け入れるスコア
	// 歩を100とする。例えば、この値を100にすると引き分けの局面は評価値が -100とみなされる。

	// 千日手での引き分けを回避しやすくなるように、デフォルト値を2に変更した。[2017/06/03]
	// ちなみに、2にしてあるのは、
	//  int contempt = Options["Contempt"] * PawnValue / 100; でPawnValueが100より小さいので
	// 1だと切り捨てられてしまうからである。

	o["Contempt"] << Option(2, -30000, 30000);

	// Contemptの設定値を先手番から見た値とするオプション。Stockfishからの独自拡張。
	// 先手のときは千日手を狙いたくなくて、後手のときは千日手を狙いたいような場合、
	// このオプションをオンにすれば、Contemptをそういう解釈にしてくれる。
	// この値がtrueのときは、Contemptを常に先手から見たスコアだとみなす。

	o["ContemptFromBlack"] << Option(false);


	//  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
	o["PvInterval"] << Option(300, 0, 100000);

	// 投了スコア
	o["ResignValue"] << Option(99999, 0, 99999);

#if 0
	// nodes as timeモード。
	// ミリ秒あたりのノード数を設定する。goコマンドでbtimeが、ここで設定した値に掛け算されたノード数を探索の上限とする。
	// 0を指定すればnodes as timeモードではない。
	// 例) 600knpsのPC動作をシミュレートするならば600を指定する。
	o["nodestime"] << Option(0, 0, 999999999);
#endif

	//
	//   パラメーターの外部からの自動調整
	//

#if defined(EVAL_LEARN)
	// 評価関数の学習を行なうときは、評価関数の保存先のフォルダを変更できる。
	// デフォルトではevalsave。このフォルダは事前に用意されているものとする。
	// このフォルダ配下にフォルダを"0/","1/",…のように自動的に掘り、そこに評価関数ファイルを保存する。
	o["EvalSaveDir"] << Option("evalsave");
#endif

#if defined(ENABLE_OUTPUT_GAME_RESULT)

#if defined(TUNING_SEARCH_PARAMETERS)
	sync_cout << "info string warning!! TUNING_SEARCH_PARAMETERS." << sync_endl;
#elif defined(USE_RANDOM_PARAMETERS)
	sync_cout << "info string warning!! USE_RANDOM_PARAMETERS." << sync_endl;
#else
	sync_cout << "info string warning!! ENABLE_OUTPUT_GAME_RESULT." << sync_endl;
#endif

	// パラメーターのログの保存先のfile path
	o["PARAMETERS_LOG_FILE_PATH"] << Option("param_log.txt");
#endif

	// 検討モード用のPVを出力するモード
	o["ConsiderationMode"] << Option(false);

	// fail low/highのときにPVを出力するかどうか。
	o["OutputFailLHPV"] << Option(true);
}

// パラメーターのランダム化のときには、
// USIの"gameover"コマンドに対して、それをログに書き出す。
void gameover_handler(const std::string& cmd)
{
#if defined(ENABLE_OUTPUT_GAME_RESULT)
	result_log << cmd << std::endl << std::flush;
#endif
}

// "isready"に対して探索パラメーターを動的にファイルから読み込んだりして初期化するための関数。
void init_param();

// -----------------------
//   やねうら王2019探索部
// -----------------------

using namespace Search;
using namespace Eval;

//namespace Search {
//
//	LimitsType Limits;
//}
// →　これはやねうら王ではtypes.cppのほうに書くようにする。

namespace {

	// -----------------------
	//      探索用の定数
	// -----------------------

	// 探索しているnodeの種類
	// Rootはここでは用意しない。Rootに特化した関数を用意するのが少し無駄なので。
	enum NodeType { NonPV, PV };

	// 置換表にどれくらいhitしているかという統計情報
	constexpr uint64_t TtHitAverageWindow = 4096;
	constexpr uint64_t TtHitAverageResolution = 1024;

	// Razor and futility margins

	// Razor marginはdepthに依存しない形で良いことが証明された。(Stockfish10)
	// これconstexprにできないので、RazorMarginという変数に代入せずにPARAM_RAZORING_MARGINを直接使うことにする。
//	constexpr int RazorMargin = PARAM_RAZORING_MARGIN;

	// depth(残り探索深さ)に応じたfutility margin。
	Value futility_margin(Depth d, bool improving) {
		return Value(PARAM_FUTILITY_MARGIN_ALPHA1/*224*/ * (d - improving));
	}

	// 【計測資料 30.】　Reductionのコード、Stockfish 9と10での比較

	// 探索深さを減らすためのReductionテーブル。起動時に初期化する。
	int Reductions[MAX_MOVES]; // [depth or moveNumber]

	// 残り探索深さをこの深さだけ減らす。d(depth)とmn(move_count)
	// i(improving)とは、評価値が2手前から上がっているかのフラグ。上がっていないなら
	// 悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
	Depth reduction(bool i, Depth d, int mn) {
		int r = Reductions[d] * Reductions[mn];
		return (r + PARAM_REDUCTION_ALPHA /* 503*/ ) / 1024 + (!i && r > PARAM_REDUCTION_BETA /*915*/);
	}

	// 【計測資料 29.】　Move CountベースのFutiliy Pruning、Stockfish 9と10での比較

	// 残り探索depthが少なくて、王手がかかっていなくて、王手にもならないような指し手を
	// 枝刈りしてしまうためのmoveCountベースのfutility pruningで用いる。
	// improving : 1手前の局面から評価値が上昇しているのか
	// depth     : 残り探索depth
	// 返し値    : 返し値よりmove_countが大きければfutility pruningを実施
	//
	// この " 3 + "のところ、パラメーター調整をしたほうが良いかも知れないが、
	// こんな細かいところいじらないほうがよさげ。(他のところでバランスを取ることにする)
	constexpr int futility_move_count(bool improving, int depth) {
		return (3 + depth * depth) / (2 - improving);
	}

	// depthに基づく、historyとstatsのupdate bonus
	int stat_bonus(Depth depth) {
		int d = depth;

		// historyとstatsの更新時のボーナス値。depthに基づく。

		// [Stockfish10]
		//return d > 17 ? 0 : 29 * d * d + 138 * d - 134;

		// [Stockfish12]
		return d > 13 ? 29 : 17 * d * d + 134 * d - 134;
	}

	// チェスでは、引き分けが0.5勝扱いなので引き分け回避のための工夫がしてあって、
	// 以下のようにvalue_drawに揺らぎを加算することによって探索を固定化しない(同じnodeを
	// 探索しつづけて千日手にしてしまうのを回避)工夫がある。
	// 将棋の場合、普通の千日手と連続王手の千日手と劣等局面による千日手(循環？)とかあるので
	// これ導入するのちょっと嫌。

	// // Add a small random component to draw evaluations to avoid 3fold-blindness
	//	Value value_draw(Thread* thisThread) {
	//	  return VALUE_DRAW + Value(2 * (thisThread->nodes & 1) - 1);
	//  }

	// Skill構造体は強さの制限の実装に用いられる。
	// (わざと手加減して指すために用いる)
	struct Skill {
		// 引数のlは、SkillLevel(手加減のレベル)。
		// 20未満であれば手加減が有効。0が一番弱い。(R2000以上下がる)
		explicit Skill(int l) : level(l) {}

		// 手加減が有効であるか。
		bool enabled() const { return level < 20; }

		// SkillLevelがNなら探索深さもNぐらいにしておきたいので、
		// depthがSkillLevelに達したのかを判定する。
		bool time_to_pick(Depth depth) const { return depth == 1 + level; }

		// 手加減が有効のときはMultiPV = 4で探索
		Move pick_best(size_t multiPV);

		// SkillLevel
		int level;

		Move best = MOVE_NONE;
	};


	// Breadcrumbs(パンくず)は、指定されたスレッドで探索されたノードをマークするために使用される。
	// スレッドが極めて多い時に同一ノードを探索するのを回避するためのhack。
	// スレッド競合でTTが破壊されるのを防ぐため？

	struct Breadcrumb {
		std::atomic<Thread*> thread;
		std::atomic<Key> key;
	};
	std::array<Breadcrumb, 1024> breadcrumbs;

	// ThreadHolding構造体は、指定されたノードでどのスレッドがパンくずを残したかを追跡する。
	// フリーノードはコンストラクタによって moves ループに入るとマークされ、デストラクタに
	// よってそのループから出るとマークされなくなる。
	struct ThreadHolding {
		explicit ThreadHolding(Thread* thisThread, Key posKey, int ply) {
			location = ply < 8 ? &breadcrumbs[posKey & (breadcrumbs.size() - 1)] : nullptr;
			otherThread = false;
			owning = false;
			if (location)
			{
				// See if another already marked this location, if not, mark it ourselves
				Thread* tmp = (*location).thread.load(std::memory_order_relaxed);
				if (tmp == nullptr)
				{
					(*location).thread.store(thisThread, std::memory_order_relaxed);
					(*location).key.store(posKey, std::memory_order_relaxed);
					owning = true;
				}
				else if (tmp != thisThread
					&& (*location).key.load(std::memory_order_relaxed) == posKey)
					otherThread = true;
			}
		}

		~ThreadHolding() {
			if (owning) // Free the marked location
				(*location).thread.store(nullptr, std::memory_order_relaxed);
		}

		bool marked() { return otherThread; }

	private:
		Breadcrumb* location;
		bool otherThread, owning;
	};


	template <NodeType NT>
	Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

	template <NodeType NT>
	Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth = 0);

	Value value_to_tt(Value v, int ply);
	Value value_from_tt(Value v, int ply);
	void update_pv(Move* pv, Move move, Move* childPv);
	void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
	void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus, int depth);
	void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
		Move* quietsSearched, int quietCount, Move* capturesSearched, int captureCount, Depth depth);


	// perftとはperformance testのこと。
	// 開始局面から深さdepthまで全合法手で進めるときの総node数を数えあげる。

	// perft() is our utility to verify move generation. All the leaf nodes up
	// to the given depth are generated and counted, and the sum is returned.
	template<bool Root>
	uint64_t perft(Position& pos, Depth depth) {

		StateInfo st;
		uint64_t cnt, nodes = 0;
		const bool leaf = (depth == 2);

		for (const auto& m : MoveList<LEGAL_ALL>(pos))
		{
			if (Root && depth <= 1)
				cnt = 1, nodes++;
			else
			{
				pos.do_move(m, st);
				cnt = leaf ? MoveList<LEGAL_ALL>(pos).size() : perft<false>(pos, depth - 1);
				nodes += cnt;
				pos.undo_move(m);
			}
			if (Root)
				sync_cout << USI::move(m) << ": " << cnt << sync_endl;
		}
		return nodes;
	}

} // namespace 


// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
// スレッド数が変更された時にも呼び出される。
void Search::init()
{
	// Stockfishでは、ここにReduction配列の初期化コードがあるが、
	// そういうのは、"isready"応答(Search::clear())でやるべき。
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void Search::clear()
{
	// 前の探索の終了を待たないと、"bestmove"応答を受け取る前に次のisreadyコマンドを送ってくる不埒なGUIがあるとも限らない。
	// 実際、"bestmove"を受け取るのを待つコードを書くと受信側のコードが複雑化するので、ここで一つ前の探索の終了を待ってあげるのは良いコード。
	Threads.main()->wait_for_search_finished();

	// -----------------------
	//   探索パラメーターの初期化
	// -----------------------

	// 探索パラメーターを動的に調整する場合は、
	// このタイミングでファイルから探索パラメーターを読み込む。
	// エンジンを終了させずに連続対局を行うときに"isready"コマンドに対して
	// 探索パラメーターをファイルから読み直して欲しいのでここで行う。

	init_param();

	// -----------------------
	//   テーブルの初期化
	// -----------------------

	// LMRで使うreduction tableの初期化

	// この初期化処理、起動時に1度でも良いのだが、探索パラメーターの調整を行なうときは、
	// init_param()のあとに行なうべきなので、ここで初期化することにする。

	// pvとnon pvのときのreduction定数
	// 0.05とか変更するだけで勝率えらく変わる

	// Threads.size()が式に含まれているのは、スレッド数が少ない時は枝刈りを甘くしたほうが得であるため。
	// スレッド数が多い時に枝刈りが甘いと同じnodeを探索するスレッドばかりになって非効率。

	// スレッド数の取得
	const size_t thread_size =
#if !defined(EVAL_LEARN)
		Threads.size();
#else
		// EVAL_LEARN版では1スレッドで探索を行うので、スレッド数は1の時の基準の枝刈りであって欲しい。
		// ゆえに、EVAL_LEARN版ではスレッド数を1とみなす。
		1;
#endif

	for (int i = 1; i < MAX_MOVES; ++i)
		Reductions[i] = int((21.3 + 2 * std::log(thread_size)) * std::log(i + 0.25 * std::log(i)));

	// -----------------------
	//   定跡の読み込み
	// -----------------------

	book.read_book();

	// -----------------------
	//   置換表のクリアなど
	// -----------------------

	//	Time.availableNodes = 0;
	TT.clear();
	Threads.clear();
	//	Tablebases::init(Options["SyzygyPath"]); // Free up mapped files
}

// 探索開始時に(goコマンドなどで)呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::search()
{
	// ---------------------
	// やねうら王固有の初期化
	// ---------------------

	// 今回、通常探索をしたかのフラグ
	// このフラグがtrueなら探索をスキップした。
	bool search_skipped = true;

	// root nodeにおける自分の手番
	auto us = rootPos.side_to_move();

	// 検討モード用のPVを出力するのか。
	Limits.consideration_mode = Options["ConsiderationMode"];

	// fail low/highのときにPVを出力するかどうか。
	Limits.outout_fail_lh_pv = Options["OutputFailLHPV"];

	// PVが詰まるのを抑制するために、前回出力時刻を記録しておく。
	lastPvInfoTime = 0;

	// ponder用の指し手の初期化
	// やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのbestmoveを用いるという独自仕様。
	// Stockfish本家もこうするべきだと思う。
	ponder_candidate = MOVE_NONE;

	// --- contempt factor(引き分けのスコア)

	// Option["Contempt"]とOption["ContemptFromBlack"]をdrawValueTableに反映させる。

	// Contempt: 引き分けを受け入れるスコア。歩を100とする。例えば、この値を100にすると引き分けの局面は
	// 評価値が - 100とみなされる。(互角と思っている局面であるなら引き分けを選ばずに他の指し手を選ぶ)
	// contempt_from_blackがtrueのときは、Contemptを常に先手から見たスコアだとみなす。

	int contempt = (int)(Options["Contempt"] * PawnValue / 100);
	if (!Options["ContemptFromBlack"])
	{
		// contemptの値を現在の手番側(us)から見た値とみなす。
		drawValueTable[REPETITION_DRAW][ us] = VALUE_ZERO - Value(contempt);
		drawValueTable[REPETITION_DRAW][~us] = VALUE_ZERO + Value(contempt);
	}
	else {
		// contemptの値を、現在の手番ではなく先手から見た値だとみなす。
		drawValueTable[REPETITION_DRAW][BLACK] = VALUE_ZERO - Value(contempt);
		drawValueTable[REPETITION_DRAW][WHITE] = VALUE_ZERO + Value(contempt);
	}

	// PVの出力間隔[ms]
	// go infiniteはShogiGUIなどの検討モードで動作させていると考えられるので
	// この場合は、PVを毎回出力しないと読み筋が出力されないことがある。
	Limits.pv_interval = (Limits.infinite || Limits.consideration_mode) ? 0 : (int)Options["PvInterval"];

	// ---------------------
	// perft(performance test)
	// ---------------------

	if (Limits.perft)
	{
		nodes = perft<true>(rootPos, Limits.perft);
		sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;
		return;
	}

	// ---------------------
	// 合法手がないならここで投了
	// ---------------------

	// 現局面で詰んでいる。
	if (rootMoves.empty())
	{
		// 投了の指し手と評価値をrootMoves[0]に積んでおけばUSI::pv()が良きに計らってくれる。
		// 読み筋にresignと出力されるが、将棋所、ShogiGUIともにバグらないのでこれで良しとする。
		rootMoves.push_back(RootMove(MOVE_RESIGN));
		rootMoves[0].score = mated_in(0);

		if (!Limits.silent)
			sync_cout << USI::pv(rootPos, 1 , -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

		goto SKIP_SEARCH;
	}

	// ---------------------
	//     定跡の選択部
	// ---------------------

	if (book.probe(*this, Limits))
		goto SKIP_SEARCH;

	// ---------------------
	//    宣言勝ち判定
	// ---------------------

	{
		// 宣言勝ちならその指し手を選択。
		// 王手がかかっていても、回避しながらトライすることもあるので王手がかかっていようが
		// Position::DeclarationWin()で判定して良い。
		// 1手詰めは、ここでは判定しない。
		// (MultiPVのときに1手詰めを見つけたからと言って探索を終了したくないから。)

		auto bestMove = rootPos.DeclarationWin();
		if (bestMove != MOVE_NONE)
		{
			// root movesの集合に突っ込んであるはず。
			// このときMultiPVが利かないが、ここ真面目にMultiPVして指し手を返すのは
			// プログラムがくちゃくちゃになるのでいまはこれは仕様としておく。

			// トライルールのとき、その指し手がgoコマンドで指定された指し手集合に含まれることを
			// 保証しないといけないのでrootMovesのなかにこの指し手が見つからないなら指すわけにはいかない。

			// 入玉宣言の条件を満たしているときは、
			// goコマンドを処理したあとのthreads.cppでMOVE_WINは追加されているはず。

			auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
			if (it_move != rootMoves.end())
			{
				std::swap(rootMoves[0], *it_move);

				// 1手詰めのときのスコアにしておく。
				rootMoves[0].score = mate_in(/*ss->ply*/ 1 + 1);;

				// rootで宣言勝ちのときにもそのPVを出力したほうが良い。
				if (!Limits.silent)
					sync_cout << USI::pv(rootPos, 1 , -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

				goto SKIP_SEARCH;
			}
		}
	}

	// ---------------------
	//    将棋倶楽部24対策
	// ---------------------

	// 将棋倶楽部24に参戦させるのであれば、王手放置の局面が送られてくるので
	// ここで王手放置の局面であるかを判定して、そうであれば相手の玉を取る指し手を生成しなくてはならない。

	// 将棋のルールに反する局面だと言えるし、コードが汚くなるので書きたくない。

	// ---------------------
	//    通常の思考処理
	// ---------------------

	// --- 今回の思考時間の設定。

	Time.init(Limits, us, rootPos.game_ply());

	// --- 置換表のTTEntryの世代を進める。

	// main threadが開始されてからだと、slaveがすでに少し探索している。
	// それらは古い世代で置換表に書き込んでしまう。
	// よってslaveが動く前であるこのタイミングで置換表の世代を進めるべきである。
	// cf. Call TT.new_search() earlier.  : https://github.com/official-stockfish/Stockfish/commit/ebc563059c5fc103ca6d79edb04bb6d5f182eaf5

	TT.new_search();

	// Stockfishでは評価関数の正常性のチェック、ここにあるが…。
	// isreadyに対する応答でやっているのでここはコメントアウトしておく。
	//Eval::NNUE::verify();

	// ---------------------
	// 各スレッドがsearch()を実行する
	// ---------------------

	Threads.start_searching(); // main以外のthreadを開始する
	Thread::search();          // main thread(このスレッド)も探索に参加する。

	// -- 探索の終了

	// 普通に探索したのでskipしたかのフラグをfalseにする。
	search_skipped = false;
SKIP_SEARCH:;

	// 最大depth深さに到達したときに、ここまで実行が到達するが、
	// まだThreads.stopが生じていない。しかし、ponder中や、go infiniteによる探索の場合、
	// USI(UCI)プロトコルでは、"stop"や"ponderhit"コマンドをGUIから送られてくるまでbest moveを出力してはならない。
	// それゆえ、単にここでGUIからそれらのいずれかのコマンドが送られてくるまで待つ。
	// "stop"が送られてきたらThreads.stop == trueになる。
	// "ponderhit"が送られてきたらThreads.ponder == falseになるので、それを待つ。(stopOnPonderhitは用いない)
	// "go infinite"に対してはstopが送られてくるまで待つ。
	// ちなみにStockfishのほう、ここのコードに長らく同期上のバグがあった。
	// やねうら王のほうは、かなり早くからこの構造で書いていた。最近のStockfishではこの書き方に追随した。
	while (!Threads.stop && (ponder || Limits.infinite))
	{
		//	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
		// (思考のためには計算資源を使っていないので。)
		Tools::sleep(1);

		// Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。
	}

	Threads.stop = true;

	// 各スレッドが終了するのを待機する(開始していなければいないで構わない)
	Threads.wait_for_search_finished();

#if 0
	// nodes as time(時間としてnodesを用いるモード)のときは、利用可能なノード数から探索したノード数を引き算する。
	// 時間切れの場合、負の数になりうる。
	// 将棋の場合、秒読みがあるので秒読みも考慮しないといけない。
	if (Limits.npmsec)
		Time.availableNodes += Limits.inc[us] + Limits.byoyomi[us] - Threads.nodes_searched();
	// →　将棋と相性がよくないのでこの機能をサポートしないことにする。
#endif

	// ---------------------
	// Lazy SMPの結果を取り出す
	// ---------------------

	Thread* bestThread = this;

	// 並列して探索させていたスレッドのうち、ベストのスレッドの結果を選出する。
	if (Options["MultiPV"] == 1
		&& !Limits.depth
		&& !Skill((int)Options["SkillLevel"]).enabled()
		//&& rootMoves[0].pv[0] != MOVE_NONE // やねうら王では投了の局面でMOVE_NONEを突っ込まないのでこのチェックは不要。
		&& !search_skipped                   // 定跡などの指し手を指させるためのこのチェックが必要。
		)
		// やねうら王では、詰んでいるときなどMOVE_RESIGNを積むので、
		// rootMoves[0].pv[0]がMOVE_RESIGNでありうる。
		// このとき、main thread以外のth->rootMoves[0]は、指し手がなくアクセス違反になるので
		// アクセスしてはならない。
		// search_skipped のときは、bestThread == mainThreadとしておき、
		// bestThread->rootMoves[0].pv[0]とpv[1]の指し手を出力すれば良い。

		bestThread = Threads.get_best_thread();


	// 次回の探索のときに何らか使えるのでベストな指し手の評価値を保存しておく。
	bestPreviousScore = bestThread->rootMoves[0].score;

	// ベストな指し手として返すスレッドがmain threadではないのなら、
	// その読み筋は出力していなかったはずなのでここで読み筋を出力しておく。
	// ただし、これはiterationの途中で停止させているので中途半端なPVである可能性が高い。
	// 検討モードではこのPVを出力しない。
	// →　いずれにせよ、mateを見つけた時に最終的なPVを出力していないと、詰みではないscoreのPVが最終的な読み筋としてGUI上に
	//     残ることになるからよろしくない。PV自体は必ず出力すべきなのでは。
	if (/*bestThread != this &&*/ !Limits.silent && !Limits.consideration_mode)
		sync_cout << USI::pv(bestThread->rootPos, bestThread->completedDepth, -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

	// ---------------------
	// 指し手をGUIに返す
	// ---------------------

	// 投了スコアが設定されていて、歩の価値を100として正規化した値がそれを下回るなら投了。
	// ただし定跡の指し手にhitした場合などはrootMoves[0].score == -VALUE_INFINITEになっているのでそれは除外。
	auto resign_value = (int)Options["ResignValue"];
	if (bestThread->rootMoves[0].score != -VALUE_INFINITE
		&& bestThread->rootMoves[0].score * 100 / PawnValue <= -resign_value)
		bestThread->rootMoves[0].pv[0] = MOVE_RESIGN;

	// サイレントモードでないならbestな指し手を出力
	if (!Limits.silent)
	{
		// sync_cout～sync_endlで全体を挟んでいるのでここを実行中に他スレッドの出力が割り込んでくる余地はない。

		// ベストなスレッドの指し手を返す。
		sync_cout << "bestmove " << bestThread->rootMoves[0].pv[0];

		// ponderの指し手の出力。
		// pvにはbestmoveのときの読み筋(PV)が格納されているので、ponderとしてpv[1]があればそれを出力してやる。
		// また、pv[1]がない場合(rootでfail highを起こしたなど)、置換表からひねり出してみる。
		if (bestThread->rootMoves[0].pv.size() > 1 || bestThread->rootMoves[0].extract_ponder_from_tt(rootPos, ponder_candidate))
			std::cout << " ponder " << bestThread->rootMoves[0].pv[1];

		std::cout << sync_endl;
	}
}

// ----------------------------------------------------------------------------------------------------------
//                        探索スレッドごとに個別の置換表へのアクセス
// ----------------------------------------------------------------------------------------------------------

// 以下のTT.probe()は、学習用の実行ファイルではスレッドごとに持っているTTのほうにアクセスして欲しいので、
// TTのマクロを定義して無理やりそっちにアクセスするように挙動を変更する。
#if defined(EVAL_LEARN)
#define TT (thisThread->tt)
// Threadのメンバにttという変数名で、スレッドごとのTranspositionTableを持っている。
// そちらを参照するように変更する。
#endif

// ----------------------------------------------------------------------------------------------------------

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// Lazy SMPなので、置換表を共有しながらそれぞれのスレッドが勝手に探索しているだけ。
void Thread::search()
{
	// ---------------------
	//      variables
	// ---------------------

	// continuationHistoryのため、(ss-7)から(ss+2)までにアクセスしたいので余分に確保しておく。
	Stack stack[MAX_PLY + 10], *ss = stack + 7;
	Move  pv[MAX_PLY + 1];

	// bestValue  : このnodeでbestMoveを指したときの(探索の)評価値
	// alpha,beta : aspiration searchの窓の範囲(alpha,beta)
	// delta      : apritation searchで窓を動かす大きさdelta
	Value bestValue, alpha, beta, delta;

	// 探索の安定性を評価するために前回のiteration時のbest moveを記録しておく。
	Move  lastBestMove = MOVE_NONE;
	Depth lastBestMoveDepth = 0;

	// もし自分がメインスレッドであるならmainThreadにそのポインタを入れる。
	// 自分がスレーブのときはnullptrになる。
	MainThread* mainThread = (this == Threads.main() ? Threads.main() : nullptr);

	// timeReduction      : 読み筋が安定しているときに時間を短縮するための係数。
	// Stockfish9までEasyMoveで処理していたものが廃止され、Stockfish10からこれが導入された。
	// totBestMoveChanges : 直近でbestMoveが変化した回数の統計。読み筋の安定度の目安にする。
	double timeReduction = 1.0 , totBestMoveChanges = 0;;

	// この局面の手番側
	Color us = rootPos.side_to_move();

	// 反復深化の時に1回ごとのbest valueを保存するための配列へのindex
	// 0から3までの値をとる。
	int iterIdx = 0;

	// 先頭10個を初期化しておけば十分。そのあとはsearch()の先頭でss+1,ss+2を適宜初期化していく。
	memset(ss - 7, 0, 10 * sizeof(Stack));

	// counterMovesをnullptrに初期化するのではなくNO_PIECEのときの値を番兵として用いる。
	for (int i = 7; i > 0; i--)
		(ss - i)->continuationHistory = &this->continuationHistory[0][0][SQ_ZERO][NO_PIECE]; // Use as a sentinel
	ss->pv = pv;

	// 反復深化のiterationが浅いうちはaspiration searchを使わない。
	// 探索窓を (-VALUE_INFINITE , +VALUE_INFINITE)とする。
	bestValue = delta = alpha = -VALUE_INFINITE;
	beta = VALUE_INFINITE;

	if (mainThread)
	{
		if (mainThread->bestPreviousScore == VALUE_INFINITE)
			for (int i = 0; i < 4; ++i)
				mainThread->iterValue[i] = VALUE_ZERO;
		else
			for (int i = 0; i < 4; ++i)
				mainThread->iterValue[i] = mainThread->bestPreviousScore;
	}

	// lowPlyHistoryのコピー(世代を一つ新しくする)
	std::copy(&lowPlyHistory[2][0], &lowPlyHistory.back().back() + 1, &lowPlyHistory[0][0]);
	std::fill(&lowPlyHistory[MAX_LPH - 2][0], &lowPlyHistory.back().back() + 1, 0);


	// --- MultiPV

	// bestmoveとしてしこの局面の上位N個を探索する機能
	size_t multiPV = Options["MultiPV"];

	// SkillLevelの実装
	Skill skill((int)Options["SkillLevel"]);

	// 強さの手加減が有効であるとき、MultiPVを有効にして、その指し手のなかから舞台裏で指し手を探す。
	// ※　SkillLevelが有効(設定された値が20未満)のときは、MultiPV = 4で探索。
	if (skill.enabled())
		multiPV = std::max(multiPV, (size_t)4);

	// この局面での指し手の数を上回ってはいけない
	multiPV = std::min(multiPV, rootMoves.size());

	// 最近の置換表の平均ヒット率の初期化。
	ttHitAverage = TtHitAverageWindow * TtHitAverageResolution / 2;

	// Contemptの処理は、やねうら王ではMainThread::search()で行っているのでここではやらない。
	// Stockfishもそうすべきだと思う。
	//int ct = int(Options["Contempt"]) * PawnValueEg / 100; // From centipawns

	// ---------------------
	//   反復深化のループ
	// ---------------------

	// 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
	// これが増えていない時、同じ深さを再度探索する。
	int searchAgainCounter = 0;

	// 1つ目のrootDepthはこのthreadの反復深化での探索中の深さ。
	// 2つ目のrootDepth (Threads.main()->rootDepth)は深さで探索量を制限するためのもの。
	// main threadのrootDepthがLimits.depthを超えた時点で、
	// slave threadはこのループを抜けて良いのでこういう書き方になっている。
	while (++rootDepth < MAX_PLY
		&& !Threads.stop
		&& !(Limits.depth && mainThread && rootDepth > Limits.depth))
	{
		// Stockfish9にはslave threadをmain threadより先行させるコードがここにあったが、
		// Stockfish10で廃止された。
		
		// これにより短い時間(低いrootDepth)では探索効率が悪化して弱くなった。
		// これは、rootDepthが小さいときはhelper threadがほとんど探索に寄与しないためである。
		// しかしrootDepthが高くなってきたときには事情が異なっていて、main threadよりdepth + 3とかで
		// 調べているhelper threadがあったとしても、探索が打ち切られる直前においては、
		// それはmain threadの探索に寄与しているとは言い難いため、無駄になる。

		// 折衷案として、rootDepthが低い時にhelper threadをmain threadより先行させる(高いdepthにする)
		// コード自体は入れたほうがいいかも知れない。

		// ------------------------
		// Lazy SMPのための初期化
		// ------------------------

		// bestMoveが変化した回数を記録しているが、反復深化の世代が一つ進むので、
		// 古い世代の情報として重みを低くしていく。
		if (mainThread)
			totBestMoveChanges /= 2;

		// aspiration window searchのために反復深化の前回のiterationのスコアをコピーしておく
		for (RootMove& rm : rootMoves)
			rm.previousScore = rm.score;

		// 将棋ではこれ使わなくていいような？

		//size_t pvFirst = 0;
		//pvLast = 0;

		// 探索深さが増えているかのフラグがfalseならカウンターを1増やす
		if (!Threads.increaseDepth)
			searchAgainCounter++;

		// MultiPVのためにこの局面の候補手をN個選出する。
		for (pvIdx = 0; pvIdx < multiPV && !Threads.stop; ++pvIdx)
		{
			// chessではtbRankの処理が必要らしい。将棋では関係なさげなのでコメントアウト。
			// tbRankが同じ値のところまでしかsortしなくて良いらしい。
			// (そこ以降は、明らかに悪い指し手なので)

			//if (pvIdx == pvLast)
			//{
			//	pvFirst = pvLast;
			//	for (pvLast++; pvLast < rootMoves.size(); pvLast++)
			//		if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
			//			break;
			//}

			// それぞれのdepthとPV lineに対するUSI infoで出力するselDepth
			selDepth = 0;

			// ------------------------
			// Aspiration window search
			// ------------------------

			// 探索窓を狭めてこの範囲で探索して、この窓の範囲のscoreが返ってきたらラッキー、みたいな探索。

			// 探索が浅いときは (-VALUE_INFINITE,+VALUE_INFINITE)の範囲で探索する。
			// 探索深さが一定以上あるなら前回の反復深化のiteration時の最小値と最大値
			// より少し幅を広げたぐらいの探索窓をデフォルトとする。

			// Reset aspiration window starting size
			// aspiration windowの開始サイズをリセットする			

			// この値は 5～10ぐらいがベスト？ Stockfish7～10では、5。Stockfish 12では4
			if (rootDepth >= 4)
			{
				Value prev = rootMoves[pvIdx].previousScore;

				// aspiration windowの幅
				// 精度の良い評価関数ならばこの幅を小さくすると探索効率が上がるのだが、
				// 精度の悪い評価関数だとこの幅を小さくしすぎると再探索が増えて探索効率が低下する。
				// やねうら王のKPP評価関数では35～40ぐらいがベスト。
				// やねうら王のKPPT(Apery WCSC26)ではStockfishのまま(18付近)がベスト。
				// もっと精度の高い評価関数を用意すべき。
				// この値はStockfish10では20に変更された。
				// Stockfish 12(NNUEを導入した)では17に変更された。
				delta = Value(PARAM_ASPIRATION_SEARCH_DELTA);

				alpha = std::max(prev - delta, -VALUE_INFINITE);
				beta  = std::min(prev + delta, VALUE_INFINITE);

				// Adjust contempt based on root move's previousScore (dynamic contempt)
				// contemptの値をrootの指し手のpreviousScoreを基に調整する。(動的なcontempt)

				// ※　contemptは千日手を受け入れるスコア。
				//     勝ってるほうは千日手にはしたくないし、負けてるほうは千日手やむなしという…。

				//int dct = ct + (113 - ct / 2) * prev / (abs(prev) + 147);
				//contempt = (us == WHITE ? make_score(dct, dct / 2)
				//	                      : -make_score(dct, dct / 2));

			}

			// 小さなaspiration windowで開始して、fail high/lowのときに、fail high/lowにならないようになるまで
			// 大きなwindowで再探索する。

			// fail highした回数
			// fail highした回数分だけ探索depthを下げてやるほうが強いらしい。
			failedHighCnt = 0;

			while (true)
			{
				// fail highするごとにdepthを下げていく処理
				Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - searchAgainCounter);
				bestValue = ::search<PV>(rootPos, ss, alpha, beta, adjustedDepth, false);

				// それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
				// 一つ目の指し手以外は-VALUE_INFINITEが返る仕様なので並べ替えのために安定ソートを
				// 用いないと前回の反復深化の結果によって得た並び順を変えてしまうことになるのでまずい。

				stable_sort(rootMoves.begin() + pvIdx, rootMoves.end());

				if (Threads.stop)
					break;

				// main threadでfail low/highが起きたなら読み筋をGUIに出力する。
				// ただし出力を散らかさないように思考開始から3秒経ってないときは抑制する。
				if (mainThread
					// MultiPVのとき、fail low/highのときにはfail low/highしたときのPVは出力しない。
					// (Stockfishがこういうコードになっている。)
					// MultiPVなのだから、別の指し手を指したときの読み筋自体はつねに出力されているわけで、
					// fail low/highしたときの読み筋は役に立たないであろうという考え。
					&& multiPV == 1
					&& (bestValue <= alpha || bestValue >= beta)
					&& Time.elapsed() > 3000
					// 将棋所のコンソールが詰まるのを予防するために出力を少し抑制する。
					// また、go infiniteのときは、検討モードから使用しているわけで、PVは必ず出力する。
					&& (rootDepth < 3 || mainThread->lastPvInfoTime + Limits.pv_interval <= Time.elapsed())
					// silent modeや検討モードなら出力を抑制する。
					&& !Limits.silent
					// ただし、outout_fail_lh_pvがfalseならfail high/fail lowのときのPVを出力しない。
					&&  Limits.outout_fail_lh_pv
					)
				{
					// 最後に出力した時刻を記録しておく。
					mainThread->lastPvInfoTime = Time.elapsed();
					sync_cout << USI::pv(rootPos, rootDepth, alpha, beta) << sync_endl;
				}

				// aspiration窓の範囲外
				if (bestValue <= alpha)
				{
					// fails low

					// betaをalphaにまで寄せてしまうと今度はfail highする可能性があるので
					// betaをalphaのほうに少しだけ寄せる程度に留める。
					beta = (alpha + beta) / 2;
					alpha = std::max(bestValue - delta, -VALUE_INFINITE);

					failedHighCnt = 0;
					// fail lowを起こしていて、いま探索を中断するのは危険。

					// やねうら王ではstopOnPonderhit使ってないのでこれはコメントアウト。
#if 0
					//if (mainThread)
						//	mainThread->stopOnPonderhit = false;
						// →　探索終了時刻が確定していてもこの場合、延長できるなら延長したい気はするが…。
#endif

				}
				else if (bestValue >= beta)
				{
					// fails high

					// このときalphaは動かさないほうが良いらしい。
					// cf. Simplify aspiration window : https://github.com/official-stockfish/Stockfish/commit/a6ae2d3a31e93000e65bdfd8f0b6d9a3e6b8ce1b
					beta = std::min(bestValue + delta, VALUE_INFINITE);

					++failedHighCnt;
				}
				else
					// 正常な探索結果なのでこれにてaspiration window searchは終了
					break;

				// delta を等比級数的に大きくしていく
				delta += delta / 4 + 5;

				ASSERT_LV3(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
			}

			// MultiPVの候補手をスコア順に再度並び替えておく。
			// (二番目だと思っていたほうの指し手のほうが評価値が良い可能性があるので…)

			stable_sort(rootMoves.begin() /* + pvFirst */, rootMoves.begin() + pvIdx + 1);

			// メインスレッド以外はPVを出力しない。
			// また、silentモードの場合もPVは出力しない。
			if (mainThread && !Limits.silent && !Threads.stop)
			{
				// 停止するときにもPVを出力すべき。(少なくともnode数などは出力されるべき)
				// (そうしないと正確な探索node数がわからなくなってしまう)

				// ただし、反復深化のiterationを途中で打ち切る場合、PVが途中までしか出力されないので困る。
				// かと言ってstopに対してPVを出力しないと、PvInterval = 300などに設定されていて短い時間で
				// 指し手を返したときに何も読み筋が出力されない。
				// →　これは、ここで出力するべきではない。best threadの選出後に出力する。

				if (/* Threads.stop || */
					// MultiPVのときは最後の候補手を求めた直後とする。
					// ただし、時間が3秒以上経過してからは、MultiPVのそれぞれの指し手ごと。
					((pvIdx + 1 == multiPV || Time.elapsed() > 3000)
						&& (rootDepth < 3 || mainThread->lastPvInfoTime + Limits.pv_interval <= Time.elapsed())))
				{
						mainThread->lastPvInfoTime = Time.elapsed();
						sync_cout << USI::pv(rootPos, rootDepth, alpha, beta) << sync_endl;
					}
				}

		} // multi PV

		  // ここでこの反復深化の1回分は終了したのでcompletedDepthに反映させておく。
		if (!Threads.stop)
			completedDepth = rootDepth;

		if (rootMoves[0].pv[0] != lastBestMove) {
			lastBestMove = rootMoves[0].pv[0];
			lastBestMoveDepth = rootDepth;
		}

		if (!mainThread)
			continue;

		//
		// main threadのときは探索の停止判定が必要
		//

		// -- やねうら王独自の処理ここから↓↓↓

		// x手詰めを発見したのか？

		// multi_pvのときは一つのpvで詰みを見つけただけでは停止するのは良くないので
		// 早期終了はmultiPV == 1のときのみ行なう。

		if (multiPV == 1)
		{
			// go mateで詰み探索として呼び出されていた場合、その手数以内の詰みが見つかっていれば終了。
			if (Limits.mate
				&& bestValue >= VALUE_MATE_IN_MAX_PLY
				&& VALUE_MATE - bestValue <= Limits.mate)
				Threads.stop = true;

			// 勝ちを読みきっているのに将棋所の表示が追いつかずに、将棋所がフリーズしていて、その間の時間ロスで
			// 時間切れで負けることがある。
			// mateを読みきったとき、そのmateの倍以上、iterationを回しても仕方ない気がするので探索を打ち切るようにする。
			if (!Limits.mate
				&& bestValue >= VALUE_MATE_IN_MAX_PLY
				&& (VALUE_MATE - bestValue) * 5/2 < (Value)(rootDepth))
				break;

			// 詰まされる形についても同様。こちらはmateの2.5倍以上、iterationを回したなら探索を打ち切る。
			if (!Limits.mate
				&& bestValue <= VALUE_MATED_IN_MAX_PLY
				&& (bestValue - (-VALUE_MATE)) * 5/2 < (Value)(rootDepth))
				break;
		}

		// ponder用の指し手として、2手目の指し手を保存しておく。
		// これがmain threadのものだけでいいかどうかはよくわからないが。
		// とりあえず、無いよりマシだろう。
		if (mainThread->rootMoves[0].pv.size() > 1)
			mainThread->ponder_candidate = mainThread->rootMoves[0].pv[1];

		// -- やねうら王独自の処理ここまで↑↑↑

		// もしSkillLevelが有効であり、時間いっぱいになったなら、準最適なbest moveを選ぶ。
		if (skill.enabled() && skill.time_to_pick(rootDepth))
			skill.pick_best(multiPV);

		// 残り時間的に、次のiterationに行って良いのか、あるいは、探索をいますぐここでやめるべきか？
		if (Limits.use_time_management())
		{
			// まだ停止が確定していない
			if (!Threads.stop && Time.search_end == 0)
			{
				// 1つしか合法手がない(one reply)であるだとか、利用できる時間を使いきっているだとか、

				double fallingEval = (318 + 6 * (mainThread->bestPreviousScore - bestValue)
					+ 6 * (mainThread->iterValue[iterIdx] - bestValue)) / 825.0;
				fallingEval = std::clamp(fallingEval, 0.5, 1.5);

				// If the bestMove is stable over several iterations, reduce time accordingly
				// もしbestMoveが何度かのiterationにおいて安定しているならば、思考時間もそれに応じて減らす

				timeReduction = lastBestMoveDepth + 9 < completedDepth ? 1.92 : 0.95;
				double reduction = (1.47 + mainThread->previousTimeReduction) / (2.32 * timeReduction);

				// Use part of the gained time from a previous stable move for the current move
				for (Thread* th : Threads)
				{
					totBestMoveChanges += th->bestMoveChanges;
					th->bestMoveChanges = 0;
				}

				double bestMoveInstability = 1 + 2 * totBestMoveChanges / Threads.size();

				// 合法手が1手しかないときはtotalTime = 0となり、即指しする計算式。
				double totalTime = rootMoves.size() == 1 ? 0 :
					Time.optimum() * fallingEval * reduction * bestMoveInstability;

				// bestMoveが何度も変更になっているならunstablePvFactorが大きくなる。
				// failLowが起きてなかったり、1つ前の反復深化から値がよくなってたりするとimprovingFactorが小さくなる。
				// Stop the search if we have only one legal move, or if available time elapsed

				if (Time.elapsed() > totalTime)
				{
					// 停止条件を満たした

					// 将棋の場合、フィッシャールールではないのでこの時点でも最小思考時間分だけは
					// 思考を継続したほうが得なので、思考自体は継続して、キリの良い時間になったらcheck_time()にて停止する。

					// ponder中なら、終了時刻はponderhit後から計算して、Time.minimum()。
					if (mainThread->ponder)
						Time.search_end = Time.minimum();
					else
					{
						// "ponderhit"しているときは、そこからの経過時間を丸める。
						// "ponderhit"していないときは開始からの経過時間を丸める。
						// そのいずれもTime.elapsed_from_ponderhit()で良い。
						Time.search_end = std::max(Time.round_up(Time.elapsed_from_ponderhit()), Time.minimum());
					}
				}
				// 前回からdepthが増えたかのチェック。
				// depthが増えて行っていないなら、同じ深さで再度探索する。
				else if (Threads.increaseDepth
					&& !mainThread->ponder
					&&  Time.elapsed() > totalTime * 0.58)
					Threads.increaseDepth = false;
				else
					Threads.increaseDepth = true;
			}
		}

		// ここで、反復深化の1回前のスコアをiterValueに保存しておく。
		// iterIdxは0..3の値をとるようにローテーションする。
		mainThread->iterValue[iterIdx] = bestValue;
		iterIdx = (iterIdx + 1) & 3;

	} // iterative deeping , 反復深化の1回分の終了

	if (!mainThread)
		return;

	mainThread->previousTimeReduction = timeReduction;

	// もしSkillLevelが有効なら、最善応手列を準最適なそれと入れ替える。
	if (skill.enabled())
		std::swap(rootMoves[0], *std::find(rootMoves.begin(), rootMoves.end(),
			skill.best ? skill.best : skill.pick_best(multiPV)));
}

// -----------------------
//      通常探索
// -----------------------

namespace {

	// cutNode = LMRで悪そうな指し手に対してreduction量を増やすnode
	template <NodeType NT>
	Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
	{
		// -----------------------
		//     nodeの種類
		// -----------------------

		// PV nodeであるか(root nodeはPV nodeに含まれる)
		constexpr bool PvNode = NT == PV;

		// root nodeであるか
		const bool rootNode = PvNode && ss->ply == 0;

		// 次の最大探索深さ。
		// これを超える深さでsearch()を再帰的に呼び出さない。
		// 延長されすぎるのを回避する。
		const Depth maxNextDepth = rootNode ? depth : depth + 1;


		// 【計測資料 34.】cuckooコード Stockfishの2倍のサイズのcuckoo配列で実験

#if defined(CUCKOO)
		// この局面から数手前の局面に到達させる指し手があるなら、それによって千日手になるので
		// このnodeで千日手スコアを即座に返すことで早期枝刈りを実施することができるらしい。
		
		Value ValueDraw = draw_value(REPETITION_DRAW, pos.side_to_move());
		if (/* pos.rule50_count() >= 3
			&&*/ alpha < ValueDraw
			&& !rootNode
			&& pos.has_game_cycle(ss->ply))
		{
			//alpha = value_draw(depth, pos.this_thread());
			alpha = ValueDraw;
			if (alpha >= beta)
				return alpha;

			/*
				将棋では、1手あれば現局面よりプラスになる指し手がほぼ確実に存在するであろうから、
				4+2n手前の局面に戻る指し手があるからと言って、draw_valueを返すのは、もったいない意味が。
				
				手番の価値(Eval::Turn)を返すのはありかな？

				あと、連続王手による千日手到達に関してはdraw_value返すのはやめたほうが…。
				これは、rating下がりうる。戻る指し手が王手ではないことを条件に含めないと。
			*/

		}
#endif

		// 残り探索深さが1手未満であるなら静止探索を呼び出す
		if (depth <= 0)
			return qsearch<NT>(pos, ss, alpha, beta);

		ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
		ASSERT_LV3(PvNode || (alpha == beta - 1));
		ASSERT_LV3(0 < depth && depth < MAX_PLY);
		// IIDを含め、PvNodeではcutNodeで呼び出さない。
		ASSERT_LV3(!(PvNode && cutNode));

		// -----------------------
		//     変数宣言
		// -----------------------

		// pv               : このnodeからのPV line(読み筋)
		// capturesSearched : 駒を捕獲する指し手(+歩の成り)
		// quietsSearched   : 駒を捕獲しない指し手(-歩の成り)
		// この[32]と[64]のところ、値を変えても強さにあまり影響なかったので固定化する。
		Move pv[MAX_PLY + 1], capturesSearched[32], quietsSearched[64];

		// do_move()するときに必要
		StateInfo st;

		// TTのprobe()の返し値
		TTEntry* tte;

		// このnodeのhash key
		Key posKey;

		// ttMove				: 置換表の指し手
		// move					: MovePickerから1手ずつもらうときの一時変数
		// excludedMove			: singular extemsionのときに除外する指し手
		// bestMove				: このnodeのbest move
		Move ttMove, move, excludedMove, bestMove;

		// extension			: 延長する深さ
		// newDepth				: 新しいdepth(残り探索深さ)
		Depth extension, newDepth;

		// bestValue			: このnodeのbestな探索スコア
		// value				: 探索スコアを受け取る一時変数
		// ttValue				: 置換表上のスコア
		// eval					: このnodeの静的評価値(の見積り)
		// maxValue             : table base probeに用いる。将棋だと関係ない。
		// probCutBeta          : prob cutする時のbetaの値。
		Value bestValue, value, ttValue, eval /*, maxValue */, probCutBeta;

		// formerPv				: このnode、以前は(置換表を見る限りは)PV nodeだったのに、今回はPV nodeではない。
		// givesCheck			: moveによって王手になるのか
		// improving			: 直前のnodeから評価値が上がってきているのか
		//   このフラグを各種枝刈りのmarginの決定に用いる
		//   cf. Tweak probcut margin with 'improving' flag : https://github.com/official-stockfish/Stockfish/commit/c5f6bd517c68e16c3ead7892e1d83a6b1bb89b69
		//   cf. Use evaluation trend to adjust futility margin : https://github.com/official-stockfish/Stockfish/commit/65c3bb8586eba11277f8297ef0f55c121772d82c
		// didLMR				: LMRを行ったのフラグ
		// priorCapture         : 1つ前の局面は駒を取る指し手か？
		bool formerPv ,givesCheck, improving, didLMR, priorCapture;

		// captureOrPawnPromotion : moveが駒を捕獲する指し手もしくは歩を成る手であるか
		// doFullDepthSearch	: LMRのときにfail highが起きるなどしたので元の残り探索深さで探索することを示すフラグ
		// moveCountPruning		: moveCountによって枝刈りをするかのフラグ(quietの指し手を生成しない)
		// ttCapture			: 置換表の指し手がcaptureする指し手であるか
		// pvExact				: PvNodeで置換表にhitして、しかもBOUND_EXACT
		// singularQuietLMR     :
		bool captureOrPawnPromotion, doFullDepthSearch, moveCountPruning,
			ttCapture, singularQuietLMR;

		// moveによって移動させる駒
		Piece movedPiece;

		// 調べた指し手を残しておいて、statsのupdateを行なうときに使う。
		// moveCount			: 調べた指し手の数(合法手に限る)
		// captureCount			: 調べた駒を捕獲する指し手の数(capturesSearched[]用のカウンター)
		// quietCount			: 調べた駒を捕獲しない指し手の数(quietsSearched[]用のカウンター)
		int moveCount, captureCount, quietCount;

		// -----------------------
		// Step 1. Initialize node
		// -----------------------

		//     nodeの初期化

		Thread* thisThread = pos.this_thread();
		ss->inCheck = pos.checkers();
		priorCapture = pos.captured_piece();
		Color us = pos.side_to_move();
		moveCount = captureCount = quietCount = ss->moveCount = 0;
		bestValue = -VALUE_INFINITE;
		//	maxValue = VALUE_INFINITE;

			//  Timerの監視

			// これはメインスレッドのみが行なう。
		if (thisThread == Threads.main())
			static_cast<MainThread*>(thisThread)->check_time();

		//  USIで出力するためのselDepthの更新

		// seldepthをGUIに出力するために、PVnodeであるならmaxPlyを更新してやる。
		// selDepthは1から数える。plyは0から数える。
		if (PvNode && thisThread->selDepth < ss->ply + 1)
			thisThread->selDepth = ss->ply + 1;

		// -----------------------
		//  RootNode以外での処理
		// -----------------------

		if (!rootNode)
		{
			// -----------------------
			// Step 2. Check for aborted search and immediate draw
			// -----------------------
			// 探索の中断と、引き分けについてチェックする

			// 連続王手による千日手、および通常の千日手、優等局面・劣等局面。

			// 連続王手による千日手に対してdraw_value()は、詰みのスコアを返すので、rootからの手数を考慮したスコアに変換する必要がある。
			// そこで、value_from_tt()で変換してから返すのが正解。

			// 教師局面生成時には、これをオフにしたほうが良いかも知れない。
			// ただし、そのときであっても連続王手の千日手は有効にしておく。
			auto draw_type = pos.is_repetition(/*ss->ply*/);
			if (draw_type != REPETITION_NONE)
				return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);

			// 最大手数を超えている、もしくは停止命令が来ている。
			if (Threads.stop.load(std::memory_order_relaxed)
				|| (ss->ply >= MAX_PLY
					|| pos.game_ply() > Limits.max_game_ply))
				return draw_value(REPETITION_DRAW, pos.side_to_move());

			// -----------------------
			// Step 3. Mate distance pruning.
			// -----------------------
			// 詰みまでの手数による枝刈り

			// rootから5手目の局面だとして、このnodeのスコアが5手以内で
			// 詰ますときのスコアを上回ることもないし、
			// 5手以内で詰まさせるときのスコアを下回ることもない。
			// そこで、alpha , betaの値をまずこの範囲に補正したあと、
			// alphaがbeta値を超えているならbeta cutする。

			alpha = std::max(mated_in(ss->ply), alpha);
			beta = std::min(mate_in(ss->ply + 1), beta);
			if (alpha >= beta)
				return alpha;
		}

		// -----------------------
		//  探索Stackの初期化
		// -----------------------

		// rootからの手数
		ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

		(ss + 1)->ply = ss->ply + 1;
		(ss + 1)->ttPv = false;
		(ss + 1)->excludedMove = bestMove = MOVE_NONE;

		// 2手先のkillerの初期化。
		(ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

		// 前の指し手で移動させた先の升目
		// TODO : null moveのときにprevSq == 1 == SQ_12になるのどうなのか…。
		Square prevSq = to_sq((ss - 1)->currentMove);

		// statScoreを現nodeの孫nodeのためにゼロ初期化。
		// statScoreは孫nodeの間でshareされるので、最初の孫だけがstatScore = 0で開始する。
		// そのあと、孫は前の孫が計算したstatScoreから計算していく。
		// このように計算された親局面のstatScoreは、LMRにおけるreduction rulesに影響を与える。
		if (!rootNode)
			(ss + 2)->statScore = 0;

		// -----------------------
		// Step 4. Transposition table lookup.
		// -----------------------

		// 置換表のlookup。前回の全探索の置換表の値を上書きする部分探索のスコアは
		// 欲しくないので、excluded moveがある場合には異なる局面キーを用いる。

		// このnodeで探索から除外する指し手。ss->excludedMoveのコピー。
		excludedMove = ss->excludedMove;

		// excludedMoveがある(singular extension時)は、異なるentryにアクセスするように。
		// ただし、このときpos.key()のbit0を破壊することは許されないので、make_key()でbit0はクリアしておく。
		// excludedMoveがMOVE_NONEの時はkeyを変更してはならない。
		posKey = excludedMove == MOVE_NONE ? pos.key() : pos.key() ^ make_key(excludedMove);

		tte = TT.probe(posKey, ss->ttHit);

		// 置換表上のスコア
		// 置換表にhitしなければVALUE_NONE

		// singular searchとIIDとのスレッド競合を考慮して、ttValue , ttMoveの順で取り出さないといけないらしい。
		// cf. More robust interaction of singular search and iid : https://github.com/official-stockfish/Stockfish/commit/16b31bb249ccb9f4f625001f9772799d286e2f04

		ttValue = ss->ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;

		// 置換表の指し手
		// 置換表にhitしなければMOVE_NONE
		// RootNodeであるなら、(MultiPVなどでも)現在注目している1手だけがベストの指し手と仮定できるから、
		// それが置換表にあったものとして指し手を進める。

		ttMove = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
				: ss->ttHit ? pos.to_move(tte->move()) : MOVE_NONE;

		// 置換表にhitしなかった時は、PV nodeのときだけttPvとして扱う。
		// これss->ttPVに保存してるけど、singularの判定等でsearchをss+1ではなくssで呼び出すことがあり、
		// そのときにss->ttPvが破壊される。なので、破壊しそうなときは直前にローカル変数に保存するコードが書いてある。

		if (!excludedMove)
			ss->ttPv = PvNode || (ss->ttHit && tte->is_pv());

		formerPv = ss->ttPv && !PvNode;

		// depthが高くてplyが低いときは、lowPlyHistoryを更新する。
		// 直前の指し手がcaptureでなければ(それは良い手のはずだから)、bonusをちょこっと加算。
		if (ss->ttPv
			&& depth > 12
			&& ss->ply - 1 < MAX_LPH
			&& !priorCapture
			&& is_ok((ss - 1)->currentMove))
			thisThread->lowPlyHistory[ss->ply - 1][from_to((ss - 1)->currentMove)] << stat_bonus(depth - 5);

		// thisThread->ttHitAverageは、ttHit(置換表にhitしたかのフラグ)の実行時の平均を近似するために用いられる。
		// 移動平均のようなものを算出している。
		thisThread->ttHitAverage = (TtHitAverageWindow - 1) * thisThread->ttHitAverage / TtHitAverageWindow
			+ TtHitAverageResolution * ss->ttHit;

		// 置換表の値による枝刈り

		if (!PvNode        // PV nodeでは置換表の指し手では枝刈りしない(PV nodeはごくわずかしかないので..)
			&& ss->ttHit         // 置換表の指し手がhitして
			&& tte->depth() >= depth   // 置換表に登録されている探索深さのほうが深くて
			&& ttValue != VALUE_NONE   // (VALUE_NONEだとすると他スレッドからTTEntryが読みだす直前に破壊された可能性がある)
			&& (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
				: (tte->bound() & BOUND_UPPER))
			// ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
			// ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
			// 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
			// このままこの値でreturnして良い。
			)
		{
			// 置換表の指し手でbeta cutが起きたのであれば、この指し手をkiller等に登録する。
			// ただし、捕獲する指し手か成る指し手であればこれは(captureで生成する指し手なので)killerを更新する価値はない。
			if (ttMove)
			{
				if (ttValue >= beta)
				{
					// 【計測資料 8.】 capture()とcaputure_or_pawn_promotion()の比較

					if (!pos.capture(ttMove))
						update_quiet_stats(pos, ss, ttMove, stat_bonus(depth), depth);

					// Extra penalty for early quiet moves of the previous ply
					// 1手前の早い時点のquietの指し手に対する追加のペナルティ
					// 1手前は置換表の指し手であるのでNULL MOVEではありえない。

					// Stockfish 10～12相当のコード
					// prioirCaptureのとこは、"pos.capture_or_pawn_promotion((ss - 1)->currentMove"
					// とするほうが正しい気がするが、強さは変わらず。

					if ((ss - 1)->moveCount <= 2 && !priorCapture)
						update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -stat_bonus(depth + 1));

				}
				// fails lowのときのquiet ttMoveに対するペナルティ
				// 【計測資料 9.】capture_or_promotion(),capture_or_pawn_promotion(),capture()での比較
#if 1
				// Stockfish 10～12相当のコード
				else if (!pos.capture_or_promotion(ttMove))
#else
				else if (!pos.capture_or_pawn_promotion(ttMove))
					//		else if (!pos.capture(ttMove))
#endif
				{
					int penalty = -stat_bonus(depth);
					thisThread->mainHistory[from_to(ttMove)][us] << penalty;
					update_continuation_histories(ss, pos.moved_piece_after(ttMove), to_sq(ttMove), penalty);
				}
			}

			//if (pos.rule50_count() < 90)
			//	return ttValue;

			return ttValue;
		}

		// -----------------------
		//     宣言勝ち
		// -----------------------

		// Step 5. Tablebases probe
		// chessだと終盤データベースというのがある。
		// これは将棋にはないが、将棋には代わりに宣言勝ちというのがある。
		// ここは、やねうら王独自のコード。

		{
			// 宣言勝ちの指し手が置換表上に登録されていることがある
			// ただしPV nodeではこれを信用しない。
			if (ttMove == MOVE_WIN && !PvNode)
			{
				return mate_in(ss->ply + 1);
			}

			// 置換表にhitしていないときは宣言勝ちの判定をまだやっていないということなので今回やる。
			// PvNodeでは置換表の指し手を信用してはいけないので毎回やる。
			if (!ttMove || PvNode)
			{
				// 王手がかかってようがかかってまいが、宣言勝ちの判定は正しい。
				// (トライルールのとき王手を回避しながら入玉することはありうるので)
				// トライルールのときここで返ってくるのは16bitのmoveだが、置換表に格納するには問題ない。
				Move m = pos.DeclarationWin();
				if (m != MOVE_NONE)
				{
					bestValue = mate_in(ss->ply + 1); // 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈
					tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, BOUND_EXACT,
						MAX_PLY, m, ss->staticEval);

					// 読み筋にMOVE_WINも出力するためには、このときpv配列を更新したほうが良いが
					// ここから更新する手段がない…。

					return bestValue;
				}
			}
		}

		// -----------------------
		//    1手詰みか？
		// -----------------------

		if (PARAM_SEARCH_MATE1)
		{
			// RootNodeでは1手詰め判定、ややこしくなるのでやらない。(RootMovesの入れ替え等が発生するので)
			// 置換表にhitしたときも1手詰め判定はすでに行われていると思われるのでこの場合もはしょる。
			// depthの残りがある程度ないと、1手詰めはどうせこのあとすぐに見つけてしまうわけで1手詰めを
			// 見つけたときのリターン(見返り)が少ない。
			// ただ、静止探索で入れている以上、depth == 1でも1手詰めを判定したほうがよさげではある。
			if (!rootNode && !ss->ttHit && !ss->inCheck)
			{
				if (PARAM_WEAK_MATE_PLY == 1)
				{
					move = Mate::mate_1ply(pos);

					if (move != MOVE_NONE)
					{
						// 1手詰めスコアなので確実にvalue > alphaなはず。
						// 1手詰めは次のnodeで詰むという解釈
						bestValue = mate_in(ss->ply + 1);

						// staticEvalの代わりに詰みのスコア書いてもいいのでは..
						tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, BOUND_EXACT,
							MAX_PLY, move, /* ss->staticEval */ bestValue);

						// ■　【計測資料 39.】 mate1plyの指し手を見つけた時に置換表の指し手でbeta cutする時と同じ処理をする。

						// 兄弟局面でこのmateの指し手がよい指し手ある可能性があるので
						// ここでttMoveでbeta cutする時と同様の処理を行うと短い時間ではわずかに強くなるっぽいのだが
						// 長い時間で計測できる差ではなかったので削除。

						return bestValue;
					}

				} else {

					move = Mate::weak_mate_3ply(pos,PARAM_WEAK_MATE_PLY);
					if (move != MOVE_NONE)
					{
						// N手詰めかも知れないのでPARAM_WEAK_MATE_PLY手詰めのスコアを返す。
						bestValue = mate_in(ss->ply + PARAM_WEAK_MATE_PLY);

						tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, BOUND_EXACT,
							MAX_PLY, move, /* ss->staticEval */ bestValue);

						return bestValue;
					}
				}

			}
			// 1手詰めがなかったのでこの時点でもsave()したほうがいいような気がしなくもない。
		}

		// -----------------------
		// Step 6. Evaluate the position statically
		// -----------------------

		//  局面を評価値によって静的に評価

		CapturePieceToHistory& captureHistory = thisThread->captureHistory;

		if (ss->inCheck)
		{
			// Skip early pruning when in check
			// 王手がかかっているときは、early pruning(早期枝刈り)をスキップする

			ss->staticEval = eval = VALUE_NONE;
			improving = false;
			goto moves_loop;
		}
		else if (ss->ttHit)
		{
			// Never assume anything about values stored in TT
			// TTに格納されている値に関して何も仮定はしない

			ss->staticEval = eval = tte->eval();

			// 置換表にhitしたなら、評価値が記録されているはずだから、それを取り出しておく。
			// あとで置換表に書き込むときにこの値を使えるし、各種枝刈りはこの評価値をベースに行なうから。

			if (eval == VALUE_NONE)
				ss->staticEval = eval = evaluate(pos);

			// 引き分けっぽい評価値であるなら、いくぶん揺らす。
			// (千日手回避のため)
			// 将棋ではこの処理どうなのかな…。

			//if (eval == VALUE_DRAW)
			//	eval = value_draw(thisThread);

			// Can ttValue be used as a better position evaluation?
			// ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。
			// 1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
			//   evalとしてttValueを採用して良い。
			// 2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
			//   evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。

			if (ttValue != VALUE_NONE
				&& (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
				eval = ttValue;

		}
		else
		{
			if ((ss - 1)->currentMove != MOVE_NULL)
				ss->staticEval = eval = evaluate(pos);
			else
				ss->staticEval = eval = -(ss - 1)->staticEval + 2 * PARAM_EVAL_TEMPO;
				// 手番の価値、PARAM_EVAL_TEMPOと仮定している。
				// 将棋では終盤の手番の価値があがるので、ここは進行度に比例する値にするだとか、
				// 評価関数からもらうだとか何とかしたほうがいい気はする。

			// 評価関数を呼び出したので置換表のエントリーはなかったことだし、何はともあれそれを保存しておく。
			// ※　bonus分だけ加算されているが静止探索の値ということで…。
			tte->save(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, MOVE_NONE, eval);

			// どうせ毎node評価関数を呼び出すので、evalの値にそんなに価値はないのだが、mate1ply()を
			// 実行したという証にはなるので意味がある。
		}

		// -----------------------
		//   evalベースの枝刈り
		// -----------------------

		// 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
		// 王手のときはここにはこない。(上のinCheckのなかでMOVES_LOOPに突入。)

		// -----------------------
		// Step 7. Razoring (skipped when in check) : ~1 Elo
		// -----------------------

		//  Razoring (王手がかかっているときはスキップする)

		// 【計測資料 24.】RazoringをStockfish 8と9とで比較

		// 残り探索深さが少ないときに、その手数でalphaを上回りそうにないとき用の枝刈り。

		if (!rootNode // The required rootNode PV handling is not available in qsearch
			&&  depth == 1
			&&  eval <= alpha - PARAM_RAZORING_MARGIN /* == RazorMargin == 510 */)
			return qsearch<NT>(pos, ss, alpha, beta);

		// 残り探索深さが1,2手のときに、alpha - razor_marginを上回るかだけ簡単に
		// (qsearchを用いてnull windowで)調べて、上回りそうにないなら
		// このnodeの探索はここ終了してリターンする。


		// 評価値が2手前の局面から上がって行っているのかのフラグ
		// 上がって行っているなら枝刈りを甘くする。
		// ※ VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
		//   枝刈りを甘くして調べないといけないのでimproving扱いとする。
		improving = (ss - 2)->staticEval == VALUE_NONE
			? ss->staticEval > (ss - 4)->staticEval || (ss - 4)->staticEval == VALUE_NONE
			: ss->staticEval > (ss - 2)->staticEval;

			//	  || ss->staticEval == VALUE_NONE
			// この条件は一つ上の式に暗黙的に含んでいる。
			// ※　VALUE_NONE == 32002なのでこれより大きなstaticEvalの値であることはないので。


		// -----------------------
		// Step 8. Futility pruning: child node (skipped when in check) : ~50 Elo
		// -----------------------

		//   Futility pruning : 子ノード (王手がかかっているときはスキップする)

		// このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
		// evalからこれを引いてbetaより大きいなら、beta cutが出来る。
		// ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
		// ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。

		// Stockfish9までは、futility pruningを、root node以外に適用していたが、
		// Stockfish10でnonPVにのみの適用に変更になった。

		if (!PvNode
			&&  depth < PARAM_FUTILITY_RETURN_DEPTH/*7*/
			&&  eval - futility_margin(depth, improving) >= beta
			&& eval < VALUE_KNOWN_WIN) // 詰み絡み等だとmate distance pruningで枝刈りされるはずで、ここでは枝刈りしない。
			return eval;
		// 次のようにするより、単にevalを返したほうが良いらしい。
		//	 return eval - futility_margin(depth);
		// cf. Simplify futility pruning return value : https://github.com/official-stockfish/Stockfish/commit/f799610d4bb48bc280ea7f58cd5f78ab21028bf5

		// -----------------------
		// Step 9. Null move search with verification search (is omitted in PV nodes) : ~40 Elo
		// -----------------------

		//  検証用の探索つきのnull move探索。PV nodeではやらない。

		//  evalの見積りがbetaを超えているので1手パスしてもbetaは超えそう。
		if (!PvNode
			&& (ss - 1)->currentMove != MOVE_NULL
			&& (ss - 1)->statScore < PARAM_NULL_MOVE_MARGIN0/*22977*/
			&&  eval >= beta
			&&  eval >= ss->staticEval
			&&  ss->staticEval >= beta - PARAM_NULL_MOVE_MARGIN1 /*30*/ * depth - PARAM_NULL_MOVE_MARGIN2 /*28*/ * improving
									+ PARAM_NULL_MOVE_MARGIN3 /*84*/ * ss->ttPv + PARAM_NULL_MOVE_MARGIN4/*168*/
			&& !excludedMove
			//		&&  pos.non_pawn_material(us)  // これ終盤かどうかを意味する。将棋でもこれに相当する条件が必要かも。
			&& (ss->ply >= thisThread->nmpMinPly || us != thisThread->nmpColor)
			// 同じ手番側に連続してnull moveを適用しない
			)
		{
			ASSERT_LV3(eval - beta >= 0);

			// 残り探索深さと評価値によるnull moveの深さを動的に減らす
			Depth R = ((PARAM_NULL_MOVE_DYNAMIC_ALPHA/*1015*/ + PARAM_NULL_MOVE_DYNAMIC_BETA/*85*/ * depth) / 256
				+ std::min(int(eval - beta) / PARAM_NULL_MOVE_DYNAMIC_GAMMA/*191*/, 3));

			ss->currentMove = MOVE_NULL;
			// null moveなので、王手はかかっていなくて駒取りでもない。
			// よって、continuationHistory[0(王手かかってない)][0(駒取りではない)][SQ_ZERO][NO_PIECE]
			ss->continuationHistory = &thisThread->continuationHistory[0][0][SQ_ZERO][NO_PIECE];

			pos.do_null_move(st);

			Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, !cutNode);

			pos.undo_null_move();

			if (nullValue >= beta)
			{
				// 1手パスしてもbetaを上回りそうであることがわかったので
				// これをもう少しちゃんと検証しなおす。

				// 証明されていないmate scoreはreturnで返さない。
				if (nullValue >= VALUE_TB_WIN_IN_MAX_PLY)
					nullValue = beta;

				if (thisThread->nmpMinPly || (abs(beta) < VALUE_KNOWN_WIN && depth < PARAM_NULL_MOVE_RETURN_DEPTH/*13*/ ))
					return nullValue;

				ASSERT_LV3(!thisThread->nmpMinPly); // 再帰的な検証は認めていない。

				// null move枝刈りを無効化してus側の手番で、plyがnmpMinPlyを超えるまで
				// 高いdepthで検証のための探索を行う。
				thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4 ;
				thisThread->nmpColor = us;

				// nullMoveせずに(現在のnodeと同じ手番で)同じ深さで探索しなおして本当にbetaを超えるか検証する。cutNodeにしない。
				Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

				thisThread->nmpMinPly = 0;

				if (v >= beta)
					return nullValue;
			}
		}

		// -----------------------
		// Step 10. ProbCut (skipped when in check) : ~10 Elo
		// -----------------------

		// probCutに使うbeta値。
		probCutBeta = beta + PARAM_PROBCUT_MARGIN1/*183*/ - PARAM_PROBCUT_MARGIN2/*49*/ * improving;

		// ProbCut(王手のときはスキップする)

		// もし、このnodeで非常に良いcaptureの指し手があり(例えば、SEEの値が動かす駒の価値を上回るようなもの)
		// 探索深さを減らしてざっくり見てもbetaを非常に上回る値を返すようなら、このnodeをほぼ安全に枝刈りすることが出来る。

		if (!PvNode
			&&  depth > PARAM_PROBCUT_DEPTH/*4*/
			&&  abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
			
			// if value from transposition table is lower than probCutBeta, don't attempt probCut
			// there and in further interactions with transposition table cutoff depth is set to depth - 3
			// because probCut search has depth set to depth - 4 but we also do a move before it
			// so effective depth is equal to depth - 3

			// もし置換表から取り出したvalueがprobCutBetaより小さいなら、そこではprobCutを試みず、
			// 置換表との相互作用では、cutoff depthをdepth - 3に設定されます。
			// なぜなら、probCut searchはdepth - 4に設定されていますが、我々はその前に指すので、
			// 実効的な深さはdepth - 3と同じになるからです。

			&& !(ss->ttHit
				&& tte->depth() >= depth - (PARAM_PROBCUT_DEPTH-1)
				&& ttValue != VALUE_NONE
				&& ttValue < probCutBeta))
		{
			// if ttMove is a capture and value from transposition table is good enough produce probCut
			// cutoff without digging into actual probCut search
			// もしttMoveが捕獲する指し手であり、置換表から取り出した値が十分に良い場合、実際のprobCut searchをせずに
			// probCutしてしまう。

			if (ss->ttHit
				&& tte->depth() >= depth - (PARAM_PROBCUT_DEPTH - 1)
				&& ttValue != VALUE_NONE
				&& ttValue >= probCutBeta
				&& ttMove
				&& pos.capture_or_promotion(ttMove))
				return probCutBeta;

			ASSERT_LV3(probCutBeta < VALUE_INFINITE);

			MovePicker mp(pos, ttMove, probCutBeta - ss->staticEval, &captureHistory);
			int probCutCount = 0;
			bool ttPv = ss->ttPv; // このあとの探索でss->ttPvを潰してしまうのでtte->save()のときはこっちを用いる。
			ss->ttPv = false;

			// 試行回数は2回(cutNodeなら4回)までとする。(よさげな指し手を3つ試して駄目なら駄目という扱い)
			// cf. Do move-count pruning in probcut : https://github.com/official-stockfish/Stockfish/commit/b87308692a434d6725da72bbbb38a38d3cac1d5f
			while ((move = mp.next_move()) != MOVE_NONE
				&& probCutCount < 2 + 2 * cutNode)
			{
				if (move != excludedMove && pos.legal(move))
				{
					ASSERT_LV3(pos.capture_or_pawn_promotion(move));
					ASSERT_LV3(depth > PARAM_PROBCUT_DEPTH);
					// Stockfish 12のコード、ここ"depth >= 5"と書いてある。
					// なぜにifの条件式に倣って"depth > 4"と書かないのか…。

					captureOrPawnPromotion = true;
					probCutCount++;

					ss->currentMove = move;
					ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
																			  [captureOrPawnPromotion]
																			  [to_sq(move)]
																			  [pos.moved_piece_after(move)];

					pos.do_move(move, st);

					// Perform a preliminary qsearch to verify that the move holds
					// この指し手がよさげであることを確認するための予備的なqsearch

					value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

					// If the qsearch held, perform the regular search
					// よさげであったので、普通に探索する

					if (value >= probCutBeta)
						value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, depth - PARAM_PROBCUT_DEPTH, !cutNode);

					pos.undo_move(move);

					if (value >= probCutBeta)
					{
						// if transposition table doesn't have equal or more deep info write probCut data into it
						// もし置換表が、等しいかより深く探索した情報ではないなら、probCutの情報をそこに書く

						if (!(ss->ttHit
							&& tte->depth() >= depth - (PARAM_PROBCUT_DEPTH - 1)
							&& ttValue != VALUE_NONE))
							tte->save(posKey, value_to_tt(value, ss->ply), ttPv,
								BOUND_LOWER,
								depth - (PARAM_PROBCUT_DEPTH - 1), move, ss->staticEval);
						return value;
				}
			}
			}

			// ss->ttPvはprobCutの探索で書き換えてしまったかも知れないので復元する。
			ss->ttPv = ttPv;
		}

		// -----------------------
		// Step 11. If the position is not in TT, decrease depth by 2
		// -----------------------

		// 局面がTTになかったのなら、探索深さを2下げる。
		// ※　このあとも置換表にヒットしないであろうから、ここを浅めで探索しておく。
		// (次に他のスレッドがこの局面に来たときには置換表にヒットするのでそのときにここの局面の
		//   探索が完了しているほうが助かるため)

		if (PvNode
			&& depth >= 6
			&& !ttMove)
			depth -= 2;

		// 王手がかかっている局面では、探索はここから始まる。
	moves_loop:

		// このノードでまだ評価関数を呼び出していないなら、呼び出して差分計算しないといけない。
		// (やねうら王独自仕様)
		// do_move()で行っている評価関数はこの限りではないが、NNUEでも
		// このタイミングで呼び出したほうが高速化するようなので呼び出す。
		Eval::evaluate_with_no_return(pos);

		// continuationHistory[0]  = Counter Move History    : ある指し手が指されたときの応手
		// continuationHistory[1]  = Follow up Move History  : 2手前の自分の指し手の継続手
		// continuationHistory[3]  = Follow up Move History2 : 4手前からの継続手
		const PieceToHistory* contHist[] = { (ss - 1)->continuationHistory	, (ss - 2)->continuationHistory,
												nullptr						, (ss - 4)->continuationHistory ,
												nullptr						, (ss - 6)->continuationHistory };

		// 直前の指し手で動かした(移動後の)駒 : やねうら王独自
		Piece prevPc = pos.piece_on(prevSq);

		// 1手前の指し手(1手前のtoとPiece)に対応するよさげな応手を統計情報から取得。
		Move countermove = thisThread->counterMoves[prevSq][prevPc];

		MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory,
			&thisThread->lowPlyHistory,
			&captureHistory,
			contHist,
			countermove,
			ss->killers,
			ss->ply);

		value = bestValue;

		singularQuietLMR = moveCountPruning = false;

		// 置換表の指し手がcaptureOrPromotionであるか。
		// 置換表の指し手がcaptureOrPromotionなら高い確率でこの指し手がベストなので、他の指し手を
		// そんなに読まなくても大丈夫。なので、このnodeのすべての指し手のreductionを増やす。

		ttCapture = ttMove && pos.capture_or_pawn_promotion(ttMove);

		// このnodeを探索中であると印をつけておく。
		// (極めて多いスレッドで探索するときに同一ノードを探索するのを回避するため)
		ThreadHolding th(thisThread, posKey, ss->ply);

		// -----------------------
		// Step 12. Loop through moves
		// -----------------------

		//  一手ずつ調べていく

		//  指し手がなくなるか、beta cutoffが発生するまで、すべての疑似合法手を調べる。

		while ((move = mp.next_move(moveCountPruning)) != MOVE_NONE)
		{
			ASSERT_LV3(is_ok(move));

			if (move == excludedMove)
				continue;

			// root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。
			if (rootNode && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
				thisThread->rootMoves.end(), move))
				continue;

			// Check for legality
			// moveの合法性をチェック。

			// root nodeはlegal()だとわかっているのでこのチェックは不要。
			// 非合法手はほとんど含まれていないから、以前はこの判定はdo_move()の直前まで遅延させたほうが得だったが、
			// do_move()するまでの枝刈りが増えてきたので、ここでやったほうが良いようだ。
			if (!rootNode && !pos.legal(move))
				continue;

			// do_move()した指し手の数のインクリメント
			ss->moveCount = ++moveCount;

			// Stockfish本家のこの読み筋の出力、細かすぎるので時間をロスする。しないほうがいいと思う。
#if 0
		// 3秒以上経過しているなら現在探索している指し手をGUIに出力する。
			if (rootNode && !Limits.silent && thisThread == Threads.main() && Time.elapsed() > 3000)
				sync_cout << "info depth " << depth
				<< " currmove " << move
				<< " currmovenumber " << moveCount + thisThread->pvIdx << sync_endl;
#endif

			// 次のnodeのpvをクリアしておく。
			if (PvNode)
				(ss + 1)->pv = nullptr;

			// -----------------------
			//      extension
			// -----------------------

		//
		// Extend checks
		//

			extension = 0;

			// 指し手で捕獲する指し手、もしくは歩の成りである。
			// 【検証資料 12.】extend checksのときのcaptureOrPawnPromotionをどう扱うか。

			// Stockfish12のコード
			//captureOrPawnPromotion = pos.capture(move);

			// こう変えるほうが強いようだ。
			captureOrPawnPromotion = pos.capture_or_pawn_promotion(move);
			//captureOrPawnPromotion = pos.capture_or_promotion(move);


			// 今回移動させる駒(移動後の駒)
			movedPiece = pos.moved_piece_after(move);

			// 今回の指し手で王手になるかどうか
			givesCheck = pos.gives_check(move);

			// -----------------------
			// Step 13. Pruning at shallow depth (~200 Elo)
			// -----------------------

			// 浅い深さでの枝刈り

			// 今回の指し手に関して新しいdepth(残り探索深さ)を計算する。
			newDepth = depth - 1;

			if (!rootNode
				// 【計測資料 7.】 浅い深さでの枝刈りを行なうときに王手がかかっていないことを条件に入れる/入れない
			//	&& pos.non_pawn_material(us)  // これに相当する処理、将棋でも必要だと思う。
				&& bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
			{
				// Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
				// move countベースの枝刈りを実行するかどうかのフラグ
				moveCountPruning = moveCount >= futility_move_count(improving, depth);

				// Reduced depth of the next LMR search
				// 次のLMR探索における軽減された深さ
				int lmrDepth = std::max(newDepth - reduction(improving, depth, moveCount), 0);

				if (!captureOrPawnPromotion
					&& !givesCheck)
				{

					// Countermoves based pruning (~20 Elo)
					// Countermovesに基づいた枝刈り(historyの値が悪いものに関してはskip) : ~20 Elo

					if (lmrDepth < PARAM_PRUNING_BY_HISTORY_DEPTH/*4*/ + ((ss - 1)->statScore > 0 || (ss - 1)->moveCount == 1)
						&& (*contHist[0])[to_sq(move)][movedPiece] < CounterMovePruneThreshold
						&& (*contHist[1])[to_sq(move)][movedPiece] < CounterMovePruneThreshold)
						// contHist[][]はStockfishと逆順なので注意。
						continue;

					// Futility pruning: parent node (~5 Elo)
					// 親nodeの時点で子nodeを展開する前にfutilityの対象となりそうなら枝刈りしてしまう。

					// パラメーター調整の係数を調整したほうが良いのかも知れないが、
					// ここ、そんなに大きなEloを持っていないので、調整しても無意味。

					if (lmrDepth < PARAM_FUTILITY_AT_PARENT_NODE_DEPTH/*7*/
						&& !ss->inCheck
						&&  ss->staticEval + PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1/*266*/ + PARAM_FUTILITY_MARGIN_BETA/*170*/ * lmrDepth <= alpha
						&& (*contHist[0])[to_sq(move)][movedPiece]
						 + (*contHist[1])[to_sq(move)][movedPiece]
						 + (*contHist[3])[to_sq(move)][movedPiece]
						 + (*contHist[5])[to_sq(move)][movedPiece] / 2 < 27376)
						continue;

					// ※　このLMRまわり、棋力に極めて重大な影響があるので枝刈りを入れるかどうかを含めて慎重に調整すべき。

					// Prune moves with negative SEE (~20 Elo)
					// 将棋ではseeが負の指し手もそのあと詰むような場合があるから、あまり無碍にも出来ないようだが…。

					// 【計測資料 20.】SEEが負の指し手を枝刈りする/しない

					if (!pos.see_ge(move, Value(-(PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1/*30*/
							- std::min(lmrDepth, PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2/*18*/)) * lmrDepth * lmrDepth)))
						continue;
				}
				else
				{
					// Capture history based pruning when the move doesn't give check
					if (  !givesCheck
						&& lmrDepth < 1
						&& captureHistory[to_sq(move)][movedPiece][type_of(pos.piece_on(to_sq(move)))] < 0)
						continue;

					// SEE based pruning
					if (!pos.see_ge(move, - Value(PARAM_LMR_SEE_MARGIN1 /*221*/) * depth)) // (~25 Elo)
						continue;
				}
			}

			// -----------------------
			// Step 14. Singular and Gives Check Extensions. : ~75 Elo
			// -----------------------

			// singular延長と王手延長。

			// Singular extension search (~70 Elo). If all moves but one fail low on a
			// search of (alpha-s, beta-s), and just one fails high on (alpha, beta),
			// then that move is singular and should be extended. To verify this we do
			// a reduced search on all the other moves but the ttMove and if the
			// result is lower than ttValue minus a margin, then we will extend the ttMove.

			// (alpha-s,beta-s)の探索(sはマージン値)において1手以外がすべてfail lowして、
			// 1手のみが(alpha,beta)においてfail highしたなら、指し手はsingularであり、延長されるべきである。
			// これを調べるために、ttMove以外の探索深さを減らして探索して、
			// その結果がttValue-s 以下ならttMoveの指し手を延長する。

			// Stockfishの実装だとmargin = 2 * depthだが、
			// 将棋だと1手以外はすべてそれぐらい悪いことは多々あり、
			// ほとんどの指し手がsingularと判定されてしまう。
			// これでは効果がないので、1割ぐらいの指し手がsingularとなるぐらいの係数に調整する。

			// note : 
			// singular延長で強くなるのは、あるnodeで1手だけが特別に良い場合、相手のプレイヤーもそのnodeでは
			// その指し手を選択する可能性が高く、それゆえ、相手のPVもそこである可能性が高いから、そこを相手よりわずかにでも
			// 読んでいて詰みを回避などできるなら、その相手に対する勝率は上がるという理屈。
			// いわば、0.5手延長が自己対戦で(のみ)強くなるのの拡張。
			// そう考えるとベストな指し手のスコアと2番目にベストな指し手のスコアとの差に応じて1手延長するのが正しいのだが、
			// 2番目にベストな指し手のスコアを小さなコストで求めることは出来ないので…。

			// singular延長をするnodeであるか。
			if (depth >= PARAM_SINGULAR_EXTENSION_DEPTH/*7*/
				&& move == ttMove
				&& !rootNode
				&& !excludedMove // 再帰的なsingular延長を除外する。
		   /* &&  ttValue != VALUE_NONE Already implicit in the next condition */
				&& abs(ttValue) < VALUE_KNOWN_WIN // 詰み絡みのスコアはsingular extensionはしない。(Stockfish 10～)
				&& (tte->bound() & BOUND_LOWER)
				&& tte->depth() >= depth - 3)
				// このnodeについてある程度調べたことが置換表によって証明されている。(ttMove == moveなのでttMove != MOVE_NONE)
				// (そうでないとsingularの指し手以外に他の有望な指し手がないかどうかを調べるために
				// null window searchするときに大きなコストを伴いかねないから。)
			{
				// このmargin値は評価関数の性質に合わせて調整されるべき。
				Value singularBeta = ttValue - ((formerPv + PARAM_SINGULAR_MARGIN/*4*/ ) * depth) / 2;
				Depth singularDepth = (depth - 1 + 3 * formerPv) / 2;

				// move(ttMove)の指し手を以下のsearch()での探索から除外

				ss->excludedMove = move;
				// 局面はdo_move()で進めずにこのnodeから浅い探索深さで探索しなおす。
				// 浅いdepthでnull windowなので、すぐに探索は終わるはず。
				value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
				ss->excludedMove = MOVE_NONE;

				// 置換表の指し手以外がすべてfail lowしているならsingular延長確定。
				if (value < singularBeta)
				{
					extension = 1;
					singularQuietLMR = !ttCapture;

					// singular extentionが生じた回数の統計を取ってみる。
					// dbg_hit_on(extension == 1);
				}

				// Multi-cut pruning
				// Our ttMove is assumed to fail high, and now we failed high also on a reduced
				// search without the ttMove. So we assume this expected Cut-node is not singular,
				// 今回のttMoveはfail highであろうし、そのttMoveなしでdepthを減らした探索においてもfail highした。
				// that multiple moves fail high, and we can prune the whole subtree by returning
				// a soft bound.
				// だから、この期待されるCut-nodeはsingularではなく、複数の指し手でfail highすると考えられる。
				// よって、hard beta boundを返すことでこの部分木全体を枝刈りする。
				else if (singularBeta >= beta)
					return singularBeta;

				// If the eval of ttMove is greater than beta we try also if there is another
				// move that pushes it over beta, if so also produce a cutoff.

				else if (ttValue >= beta)
				{
					ss->excludedMove = move;
					value = search<NonPV>(pos, ss, beta - 1, beta, (depth + 3) / 2, cutNode);
					ss->excludedMove = MOVE_NONE;

					if (value >= beta)
					return beta;
			}

			}

			// 王手延長 : ~2 Elo

			// 王手となる指し手でSEE >= 0であれば残り探索深さに1手分だけ足す。
			// また、moveCountPruningでない指し手(置換表の指し手とか)も延長対象。
			// これはYSSの0.5手延長に似たもの。
			// ※　将棋においてはこれはやりすぎの可能性も..

			// 【計測資料 33.】王手延長のコード、pos.blockers_for_king(~us) & from_sq(move)も延長する/しない
			
			// Stockfish9では、	&& !moveCountPruning が条件式に入っていた。
			// Stockfish10のコードは、敵側のpin駒を取る指し手か、駒得になる王手に限定して延長している。
			// pin駒を剥がす指し手は、こちらの利きはあるということなので2枚利いていることが多く、駒得でなくとも有効。

			// Check extension (~2 Elo)
			else if (givesCheck
				&& (pos.is_discovery_check_on_king(~us, move) || pos.see_ge(move)))
				extension = 1;

			// Last captures extension

			// 最後に捕獲した駒による延長
			// 捕獲した駒の終盤での価値がPawnより大きく、詰みに直結しそうなら延長する。
			// 将棋では駒は終盤で増えていくので関係なさげ。

			//else if (PieceValue[EG][pos.captured_piece()] > PawnValueEg
			//	&& pos.non_pawn_material() <= 2 * RookValueMg)
			//	extension = 1;

			// Late irreversible move extension
			// 終盤の不可逆な指し手による延長
			// 将棋では関係なさげなので知らね。将棋ではほとんどの駒が不可逆な動きをするから。
			
			//if (move == ttMove
			//	&& pos.rule50_count() > 80
			//	&& (captureOrPromotion || type_of(movedPiece) == PAWN))
			//	extension = 2;

			// -----------------------
			//   1手進める前の枝刈り
			// -----------------------

			// 再帰的にsearchを呼び出すとき、search関数に渡す残り探索深さ。
			// これはsingluar extensionの探索が終わってから決めなければならない。(singularなら延長したいので)
			newDepth += extension;

			// -----------------------
			//      1手進める
			// -----------------------

			// この時点で置換表をprefetchする。将棋においては、指し手に駒打ちなどがあって指し手を適用したkeyを
			// 計算するコストがわりとあるので、これをやってもあまり得にはならない。無効にしておく。

			// 投機的なprefetch
			//const Key nextKey = pos.key_after(move);
			//prefetch(TT.first_entry(nextKey));
			//Eval::prefetch_evalhash(nextKey);

			// 現在このスレッドで探索している指し手を保存しておく。
			ss->currentMove = move;
			ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck][captureOrPawnPromotion][to_sq(move)][movedPiece];

			// -----------------------
			// Step 15. Make the move
			// -----------------------

			// 指し手で1手進める
			pos.do_move(move, st, givesCheck);

			// -----------------------
			// Step 16. Reduced depth search (LMR, ~200 Elo). 
			// -----------------------
			// depthを減らした探索。LMR(Late Move Reduction)

			// If the move fails high it will be re - searched at full depth.
			// depthを減らして探索させて、その指し手がfail highしたら元のdepthで再度探索するという手法 

			// 【計測資料 32.】LMRのコード、Stockfish9と10の比較

			// moveCountが大きいものなどは探索深さを減らしてざっくり調べる。
			// alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。

			if (    depth >= 3
				&&  moveCount > 1 + 3 * rootNode
				&& (!captureOrPawnPromotion
					|| moveCountPruning
					|| ss->staticEval + CapturePieceValue[pos.captured_piece()] <= alpha
					|| cutNode
					|| thisThread->ttHitAverage < 432 * TtHitAverageResolution * TtHitAverageWindow / 1024))
			{
				// Reduction量
				Depth r = reduction(improving, depth, moveCount);

				// Decrease reduction if the ttHit running average is large
				if (thisThread->ttHitAverage > 537 * TtHitAverageResolution * TtHitAverageWindow / 1024)
					r--;

				// Increase reduction if other threads are searching this position
				// 他のスレッドがこの局面を探索中であるならreductionを増やす。
				// (どうせそのスレッドが探索するであろうから)
				if (th.marked())
					r++;

				// Decrease reduction if position is or has been on the PV (~10 Elo)
				// この局面がPV上であるならreductionを減らす
				if (ss->ttPv)
					r -= 2 ;

				// Increase reduction at root and non-PV nodes when the best move does not change frequently
				// best moveが頻繁に変更されていないならば局面が安定しているのだろうから、rootとnon-PVではreductionを増やす。
				if ((rootNode || !PvNode) && depth > 10 && thisThread->bestMoveChanges <= 2)
						r++;

				if (moveCountPruning && !formerPv)
					r++;

				// Decrease reduction if opponent's move count is high (~5 Elo)
				// 相手の指し手(1手前の指し手)のmove countが高い場合、reduction量を減らす。
				// 相手の指し手をたくさん読んでいるのにこちらだけreductionするとバランスが悪いから。

				// 【計測資料 4.】相手のmoveCountが高いときにreductionを減らす
				// →　古い計測なので当時はこのコードないほうが良かったが、Stockfish10では入れたほうが良さげ。

				if ((ss - 1)->moveCount > 13)
					r--;

				// Decrease reduction if ttMove has been singularly extended (~3 Elo)
				if (singularQuietLMR)
					r--;

				if (!captureOrPawnPromotion)
				{
					// Increase reduction if ttMove is a capture (~5 Elo)
					// 【計測資料 3.】置換表の指し手がcaptureのときにreduction量を増やす。

					if (ttCapture)
						r += 1;

					// Increase reduction at root if failing high
					r += rootNode ? thisThread->failedHighCnt * thisThread->failedHighCnt * moveCount / 512 : 0;

					// cut nodeにおいてhistoryの値が悪い指し手に対してはreduction量を増やす。
					// ※　PVnodeではIID時でもcutNode == trueでは呼ばないことにしたので、
					// if (cutNode)という条件式は暗黙に && !PvNode を含む。

					// Increase reduction for cut nodes (~10 Elo)
					// 【計測資料 18.】cut nodeのときにreductionを増やすかどうか。

					if (cutNode)
						r += 2;

					// Decrease reduction for moves that escape a capture. Filter out
					// castling moves, because they are coded as "king captures rook" and
					// hence break make_move(). (~2 Elo)
						  
					// 当たりを避ける手(捕獲から逃れる指し手)はreduction量を減らす。

					// do_move()したあとなのでtoの位置には今回移動させた駒が来ている。
					// fromの位置は空(NO_PIECE)の升となっている。

					// 例えばtoの位置に金(今回移動させた駒)があるとして、これをfromに動かす。
					// 仮にこれが歩で取られるならsee() < 0 となる。

					// ただ、手番つきの評価関数では駒の当たりは評価されているし、
					// 当たり自体は将棋ではチェスほど問題とならないので…。

#if 0
					// よく考えたら駒打ちに対してはreverse_move()が作れないし、
					// 普通の移動であったとしても、将棋だとそれが合法手とは限らないし..
					// これ、toにある駒がfromに利いてるかのチェックとか何かが必要なのではないかな..
					//
					// あと、この枝刈りをやるとして、将棋では、type_of(move) == NORMALに対して駒打ちは除外するように設計しないと
					// 駒打ちに対するreverse_move()を考えるとおかしいことになるので…。

					// 【計測資料 38.】 reverse_moveのSEEで、reduction

					else if (   type_of(move) == NORMAL
							&& !pos.see_ge(reverse_move(move)))
						r -= 2 + ss->ttPv - (type_of(movedPiece) == PAWN);
#endif

					// 【計測資料 11.】statScoreの計算でcontHist[3]も調べるかどうか。
					// contHist[5]も/2とかで入れたほうが良いのでは…。誤差か…？
					ss->statScore = thisThread->mainHistory[from_to(move)][us]
						+ (*contHist[0])[to_sq(move)][movedPiece]
						+ (*contHist[1])[to_sq(move)][movedPiece]
						+ (*contHist[3])[to_sq(move)][movedPiece]
						- PARAM_REDUCTION_BY_HISTORY/*5278*/; // 修正項

					// historyの値に応じて指し手のreduction量を増減する。

					// 【計測資料 1.】

					// Decrease/increase reduction by comparing opponent's stat score (~10 Elo)
					if (ss->statScore >= -105 && (ss - 1)->statScore < -103)
						r--;

					else if ((ss - 1)->statScore >= -122 && ss->statScore < -129)
						r++;

					// Decrease/increase reduction for moves with a good/bad history (~30 Elo)
					// ~30 Elo
					r -= ss->statScore / 14884;
				}
				else
				{
					// Unless giving check, this capture is likely bad
					if (!givesCheck
						&& ss->staticEval + CapturePieceValue[pos.captured_piece()] + 210 * depth <= alpha)
						r++;
				}

				// depth >= 3なのでqsearchは呼ばれないし、かつ、
				// moveCount > 1 すなわち、このnodeの2手目以降なのでsearch<NonPv>が呼び出されるべき。

				Depth d = std::clamp(newDepth - r, 1, newDepth);

				value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);

				//
				// ここにその他の枝刈り、何か入れるべき(かも)
				//

				// 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす。

				doFullDepthSearch = value > alpha && d != newDepth;

				didLMR = true;
			}
			else
			{
				// non PVか、PVでも2手目以降であればfull depth searchを行なう。
				doFullDepthSearch = !PvNode || moveCount > 1;

				didLMR = false;
			}


			// -----------------------
			// Step 17. Full depth search when LMR is skipped or fails high
			// -----------------------

			// Full depth search。LMRがskipされたか、LMRにおいてfail highを起こしたなら元の探索深さで探索する。

			// ※　静止探索は残り探索深さはdepth = 0として開始されるべきである。(端数があるとややこしいため)
			if (doFullDepthSearch)
			{
				value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

				if (didLMR && !captureOrPawnPromotion)
				{
					int bonus = value > alpha ? stat_bonus(newDepth)
						: -stat_bonus(newDepth);

					update_continuation_histories(ss, movedPiece, to_sq(move), bonus);
				}
			}

			// PV nodeにおいては、full depth searchがfail highしたならPV nodeとしてsearchしなおす。
			// ただし、value >= betaなら、正確な値を求めることにはあまり意味がないので、これはせずにbeta cutしてしまう。
			if (PvNode && (moveCount == 1 || (value > alpha && (rootNode || value < beta))))
			{
				// 次のnodeのPVポインターはこのnodeのpvバッファを指すようにしておく。
				(ss + 1)->pv = pv;
				(ss + 1)->pv[0] = MOVE_NONE;

				// full depthで探索するときはcutNodeにしてはいけない。
				value = -search<PV>(pos, ss + 1, -beta, -alpha,
									std::min(maxNextDepth, newDepth), false);
			}

			// -----------------------
			// Step 18. Undo move
			// -----------------------

			//      1手戻す

			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// -----------------------
			// Step 19. Check for a new best move
			// -----------------------

			// 指し手を探索するのを終了する。
			// 停止シグナルが来たときは、探索の結果の値は信頼できないので、
			// best moveの更新をせず、PVや置換表を汚さずに終了する。

			if (Threads.stop.load(std::memory_order_relaxed))
				return VALUE_ZERO;

			// -----------------------
			//  root node用の特別な処理
			// -----------------------

			if (rootNode)
			{
				RootMove& rm = *std::find(thisThread->rootMoves.begin(),
					thisThread->rootMoves.end(), move);

				// PVの指し手か、新しいbest moveか？
				if (moveCount == 1 || value > alpha)
				{
					// root nodeにおいてPVの指し手または、α値を更新した場合、スコアをセットしておく。
					// (iterationの終わりでsortするのでそのときに指し手が入れ替わる。)

					rm.score = value;
					rm.selDepth = thisThread->selDepth;
					rm.pv.resize(1);
					// PVは変化するはずなのでいったんリセット

					// 1手進めたのだから、何らかPVを持っているはずなのだが。
					ASSERT_LV3((ss + 1)->pv);

					// RootでPVが変わるのは稀なのでここがちょっとぐらい重くても問題ない。
					// 新しく変わった指し手の後続のpvをRootMoves::pvにコピーしてくる。
					for (Move* m = (ss + 1)->pv; *m != MOVE_NONE; ++m)
						rm.pv.push_back(*m);

					// bestMoveが何度変更されたかを記録しておく。
					// これが頻繁に行われるのであれば、思考時間を少し多く割り当てる。
					if (moveCount > 1)
						++thisThread->bestMoveChanges;

				} else {

					// root nodeにおいてα値を更新しなかったのであれば、この指し手のスコアを-VALUE_INFINITEにしておく。
					// こうしておかなければ、stable_sort()しているにもかかわらず、前回の反復深化のときの値との
					// 大小比較してしまい指し手の順番が入れ替わってしまうことによるオーダリング性能の低下がありうる。
					rm.score = -VALUE_INFINITE;
				}
			}

			// -----------------------
			//  alpha値の更新処理
			// -----------------------

			if (value > bestValue)
			{
				bestValue = value;

				if (value > alpha)
				{
					bestMove = move;

					// fail-highのときにもPVをupdateする。
					if (PvNode && !rootNode)
						update_pv(ss->pv, move, (ss + 1)->pv);

					// alpha値を更新したので更新しておく
					if (PvNode && value < beta)
					{
						alpha = value;

						// PvNodeでalpha値を更新した。
						// このとき相手からの詰みがあるかどうかを調べるなどしたほうが良いなら
						// ここに書くべし。

					}
					else
					{
						// value >= beta なら fail high(beta cut)

						// また、non PVであるなら探索窓の幅が0なのでalphaを更新した時点で、value >= betaが言えて、
						// beta cutである。

						ASSERT_LV3(value >= beta);

						// fail highのときには、負のstatScoreをリセットしたほうが良いらしい。
						// cf. Reset negative statScore on fail high : https://github.com/official-stockfish/Stockfish/commit/b88374b14a7baa2f8e4c37b16a2e653e7472adcc
						// →　その後、単に0にリセットしたほうが良いことが判明した。
						ss->statScore = 0;
						break;
					}
				}
			}

			if (move != bestMove)
			{
				// 探索した駒を捕獲する指し手を32手目まで
				if (captureOrPawnPromotion && captureCount < 32)
					capturesSearched[captureCount++] = move;

				// 探索した駒を捕獲しない指し手を64手目までquietsSearchedに登録しておく。
				// あとでhistoryなどのテーブルに加点/減点するときに使う。

				if (!captureOrPawnPromotion && quietCount < 64)
					quietsSearched[quietCount++] = move;
			}
		}
		// end of while

		// -----------------------
		// Step 20. Check for mate and stalemate
		// -----------------------

		// 詰みとステイルメイトをチェックする。

		// このStockfishのassert、合法手を生成しているので重すぎる。良くない。
		ASSERT_LV5(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

		// (将棋では)合法手がない == 詰まされている なので、rootの局面からの手数で詰まされたという評価値を返す。
		// ただし、singular extension中のときは、ttMoveの指し手が除外されているので単にalphaを返すべき。
		if (!moveCount)
			bestValue = excludedMove ? alpha : mated_in(ss->ply);

		// bestMoveがあるならこの指し手に基いてhistoryのupdateを行なう。
		else if (bestMove)

			// quietな(駒を捕獲しない)best moveなのでkillerとhistoryとcountermovesを更新する。

			update_all_stats(pos, ss, bestMove, bestValue, beta, prevSq,
				quietsSearched, quietCount, capturesSearched, captureCount, depth);


		// bestMoveがない == fail lowしているケース。
		// fail lowを引き起こした前nodeでのcounter moveに対してボーナスを加点する。
		// 【計測資料 15.】search()でfail lowしているときにhistoryのupdateを行なう条件

		else if ((depth >= 3 || PvNode)
				&& !priorCapture)
			update_continuation_histories(ss - 1, /*pos.piece_on(prevSq)*/prevPc, prevSq, stat_bonus(depth));

		// 将棋ではtable probe使っていないのでmaxValue関係ない。
		//if (PvNode)
		//	bestValue = std::min(bestValue, maxValue);

		// もし良い指し手が見つからず(bestValueがalphaを更新せず)、前の局面はttPvを選んでいた場合は、
		// 前の相手の手がおそらく良い手であり、新しい局面が探索木に追加される。
		// (ttPvをtrueに変更してTTEntryに保存する)

		if (bestValue <= alpha)
			ss->ttPv = ss->ttPv || ((ss - 1)->ttPv && depth > 3);

		// それ以外の場合は、カウンターの手が見つかり、その局面が探索木のleafであれば、
		// その局面を探索木から削除する。
		// (ttPvをfalseに変更してTTEntryに保存する)

		else if (depth > 3)
			ss->ttPv = ss->ttPv && (ss + 1)->ttPv;


		// -----------------------
		//  置換表に保存する
		// -----------------------

		// betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
		// すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
		// さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
		// また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
		// ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
		// すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。

		if (!excludedMove && !(rootNode && thisThread->pvIdx))
			tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
				bestValue >= beta ? BOUND_LOWER :
				PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
				depth, bestMove, ss->staticEval);

		// qsearch()内の末尾にあるassertの文の説明を読むこと。
		ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

		return bestValue;
	}

	// -----------------------
	//      静止探索
	// -----------------------

	// qsearch()は静止探索を行う関数で、search()でdepth(残り探索深さ)が0になったときに呼び出されるか、
	// このqseach()自身から再帰的にもっと低いdepthで呼び出される。

	template <NodeType NT>
	Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
	{
		// -----------------------
		//     変数宣言
		// -----------------------

		// PV nodeであるか。
		constexpr bool PvNode = NT == PV;

		ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
		ASSERT_LV3(PvNode || alpha == beta - 1);
		ASSERT_LV3(depth <= 0);

		// PV求める用のbuffer
		// (これnonPVでは不要なので、nonPVでは参照していないの削除される。)
		Move pv[MAX_PLY + 1];

		// make_move()のときに必要
		StateInfo st;

		// 置換表にhitしたときの置換表のエントリーへのポインタ
		TTEntry* tte;

		// この局面のhash key
		Key posKey;

		// ttMove			: 置換表に登録されていた指し手
		// move				: MovePickerからもらった現在の指し手
		// bestMove			: この局面でのベストな指し手
		Move ttMove, move, bestMove;

		// このnodeに関して置換表に登録するときのdepth(残り探索深さ)
		Depth ttDepth;

		// bestValue		: best moveに対する探索スコア(alphaとは異なる)
		// value			: 現在のmoveに対する探索スコア
		// ttValue			: 置換表に登録されていたスコア
		// futilityValue	: futility pruningに用いるスコア
		// futilityBase		: futility pruningの基準となる値
		// oldAlpha			: この関数が呼び出された時点でのalpha値
		Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;

		// pvHit			: 置換表から取り出した指し手が、PV nodeでsaveされたものであった。
		// givesCheck		: MovePickerから取り出した指し手で王手になるか
		// captureOrPawnPromotion : 駒を捕獲する指し手か、歩を成る指し手であるか
		bool pvHit, givesCheck , captureOrPawnPromotion;

		// このnodeで何手目の指し手であるか
		int moveCount;

		// -----------------------
		//     nodeの初期化
		// -----------------------

		if (PvNode)
		{
			// PV nodeではalpha値を上回る指し手が存在した場合は(そこそこ指し手を調べたので)置換表にはBOUND_EXACTで保存したいから、
			// そのことを示すフラグとして元の値が必要(non PVではこの変数は参照しない)
			// PV nodeでalpha値を上回る指し手が存在しなかった場合は、調べ足りないのかも知れないからBOUND_UPPERとしてbestValueを保存しておく。
			oldAlpha = alpha;

			// PvNodeのときしかoldAlphaを初期化していないが、PvNodeのときしか使わないのでこれは問題ない。

			(ss + 1)->pv = pv;
			ss->pv[0] = MOVE_NONE;
		}

		Thread* thisThread = pos.this_thread();

		// rootからの手数
		(ss + 1)->ply = ss->ply + 1;

		bestMove = MOVE_NONE;
		ss->inCheck = pos.checkers();
		moveCount = 0;

		// -----------------------
		//    最大手数へ到達したか？
		// -----------------------

		if (ss->ply >= MAX_PLY || pos.game_ply() > Limits.max_game_ply)
			return draw_value(REPETITION_DRAW, pos.side_to_move());

		ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

		// -----------------------
		//     置換表のprobe
		// -----------------------

		// 置換表に登録するdepthはあまりマイナスの値だとおかしいので、
		// 王手がかかっているときは、DEPTH_QS_CHECKS(=0)、王手がかかっていないときはDEPTH_QS_NO_CHECKS(-1)とみなす。
		ttDepth = ss->inCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
													  : DEPTH_QS_NO_CHECKS;

		posKey = pos.key();
		tte = TT.probe(posKey, ss->ttHit);
		ttValue = ss->ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;
		ttMove = ss->ttHit ? pos.to_move(tte->move()) : MOVE_NONE;
		pvHit = ss->ttHit && tte->is_pv();

		// nonPVでは置換表の指し手で枝刈りする
		// PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)
		if (!PvNode
			&& ss->ttHit
			&& tte->depth() >= ttDepth
			&& ttValue != VALUE_NONE // 置換表から取り出したときに他スレッドが値を潰している可能性があるのでこのチェックが必要
			&& (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
								: (tte->bound() & BOUND_UPPER)))
			// ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
			// ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
			// 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
			// このままこの値でreturnして良い。
		{
			return ttValue;
		}

		// -----------------------
		//     eval呼び出し
		// -----------------------

		// Evaluate the position statically

		if (ss->inCheck)
		{
			ss->staticEval = VALUE_NONE;

			// bestValueはalphaとは違う。
			// 王手がかかっているときは-VALUE_INFINITEを初期値として、すべての指し手を生成してこれを上回るものを探すので
			// alphaとは区別しなければならない。
			bestValue = futilityBase = -VALUE_INFINITE;

		} else {

			// -----------------------
			//      一手詰め判定
			// -----------------------

			// 置換表にhitした場合は、すでに詰みを調べたはずなので
			// 置換表にhitしなかったときにのみ調べる。

			if (PARAM_QSEARCH_MATE1 && !ss->ttHit)
			{
				// いまのところ、入れたほうが良いようだ。
				// play_time = b1000 ,  1631 - 55 - 1314(55.38% R37.54) [2016/08/19]
				// play_time = b6000 ,  538 - 23 - 439(55.07% R35.33) [2016/08/19]

				// 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈
				if (PARAM_WEAK_MATE_PLY == 1)
				{
					if (Mate::mate_1ply(pos) != MOVE_NONE)
						return mate_in(ss->ply + 1);
				}
				else
				{
					if (Mate::weak_mate_3ply(pos, PARAM_WEAK_MATE_PLY) != MOVE_NONE)
						// 1手詰めかも知れないがN手詰めの可能性があるのでNを返す。
						return mate_in(ss->ply + PARAM_WEAK_MATE_PLY);
				}
				// このnodeに再訪問することはまずないだろうから、置換表に保存する価値はない。

			}

			// 王手がかかっていないなら置換表の指し手を持ってくる

			if (ss->ttHit)
			{

				// 置換表に評価値が格納されているとは限らないのでその場合は評価関数の呼び出しが必要
				// bestValueの初期値としてこの局面のevaluate()の値を使う。これを上回る指し手があるはずなのだが..
				if ((ss->staticEval = bestValue = tte->eval()) == VALUE_NONE)
					ss->staticEval = bestValue = evaluate(pos);

				// 毎回evaluate()を呼ぶならtte->eval()自体不要なのだが、
				// 置換表の指し手でこのまま枝刈りできるケースがあるから難しい。
				// 評価関数がKPPTより軽ければ、tte->eval()をなくしても良いぐらいなのだが…。

				// 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
				// 精度で探索されたものであるなら、それをbestValueの初期値として使う。
				if (	ttValue != VALUE_NONE
					&& (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
					bestValue = ttValue;

			} else {

				// 置換表がhitしなかった場合、bestValueの初期値としてevaluate()を呼び出すしかないが、
				// NULL_MOVEの場合は前の局面での値を反転させると良い。(手番を考慮しない評価関数であるなら)
				// NULL_MOVEしているということは王手がかかっていないということであり、前の局面でevaluate()は呼び出しているから
				// StateInfo.sumは更新されていて、そのあとdo_null_move()ではStateInfoが丸ごとコピーされるから、現在のpos.state().sumは
				// 正しい値のはず。


				if (!PARAM_QSEARCH_FORCE_EVAL)
				{
			// Stockfish相当のコード
				ss->staticEval = bestValue =
					(ss - 1)->currentMove != MOVE_NULL ? evaluate(pos)
					: -(ss - 1)->staticEval + 2 * PARAM_EVAL_TEMPO;

				} else {

					// 評価関数の実行時間・精度によっては、こう書いたほうがいいかもという書き方。
					// 残り探索深さが大きい時は、こっちに切り替えるのはありかも…。
					// どちらが優れているかわからないので、optimizerに任せる。
				ss->staticEval = bestValue = evaluate(pos);
				}
			}

			// Stand pat.
			// 現在のbestValueは、この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
			// 王手がかかっていないケースにおいては、この時点での静的なevalの値がbetaを上回りそうならこの時点で帰る。
			if (bestValue >= beta)
			{
				// Stockfishではここ、pos.key()になっているが、posKeyを使うべき。
				if (!ss->ttHit)
					tte->save(posKey, value_to_tt(bestValue, ss->ply), false /* ss->ttHit == false */, BOUND_LOWER,
						DEPTH_NONE, MOVE_NONE, ss->staticEval);

				return bestValue;
			}

			// 王手がかかっていなくてPvNodeでかつ、bestValueがalphaより大きいならそれをalphaの初期値に使う。
			// 王手がかかっているなら全部の指し手を調べたほうがいい。
			if (PvNode && bestValue > alpha)
				alpha = bestValue;

			// futilityの基準となる値をbestValueにmargin値を加算したものとして、
			// これを下回るようであれば枝刈りする。
			futilityBase = bestValue + PARAM_FUTILITY_MARGIN_QUIET /*155*/;
		}

		// -----------------------
		//     1手ずつ調べる
		// -----------------------

		const PieceToHistory* contHist[] = { (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
												nullptr					  , (ss - 4)->continuationHistory,
												nullptr					  , (ss - 6)->continuationHistory };

		// 取り合いの指し手だけ生成する
		// searchから呼び出された場合、直前の指し手がMOVE_NULLであることがありうるが、
		// 静止探索の1つ目の深さではrecaptureを生成しないならこれは問題とならない。
		MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory,
										  &thisThread->captureHistory,
										  contHist,
										  to_sq((ss - 1)->currentMove));

		// このあとnodeを展開していくので、evaluate()の差分計算ができないと速度面で損をするから、
		// evaluate()を呼び出していないなら呼び出しておく。
		evaluate_with_no_return(pos);

		while ((move = mp.next_move()) != MOVE_NONE)
		{
			// MovePickerで生成された指し手はpseudo_legalであるはず。
			ASSERT_LV3(pos.pseudo_legal(move));

			// -----------------------
			//  局面を進める前の枝刈り
			// -----------------------

			givesCheck = pos.gives_check(move);
			captureOrPawnPromotion = pos.capture_or_pawn_promotion(move);

			moveCount++;

			//
			//  Futility pruning
			//

			// 自玉に王手がかかっていなくて、敵玉に王手にならない指し手であるとき、
			// 今回捕獲されるであろう駒による評価値の上昇分を
			// 加算してもalpha値を超えそうにないならこの指し手は枝刈りしてしまう。

			if (   bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				// cf. https://github.com/official-stockfish/Stockfish/commit/392b529c3f52103ad47ad096b86103c17758cb4f
				// !ss->inCheck
				&& !givesCheck
				&&  futilityBase > -VALUE_KNOWN_WIN
			//	&& !pos.advanced_pawn_push(move))
				)
			{
				//assert(type_of(move) != ENPASSANT); // Due to !pos.advanced_pawn_push

				// MoveCountに基づく枝刈り
				if (moveCount > 2)
					continue;

				// moveが成りの指し手なら、その成ることによる価値上昇分もここに乗せたほうが正しい見積りになるはず。
				// 【計測資料 14.】 futility pruningのときにpromoteを考慮するかどうか。
				futilityValue = futilityBase + (Value)CapturePieceValue[pos.piece_on(to_sq(move))]
								+ (is_promote(move) ? (Value)ProDiffPieceValue[pos.piece_on(from_sq(move))] : VALUE_ZERO);

				// futilityValueは今回捕獲するであろう駒の価値の分を上乗せしているのに
				// それでもalpha値を超えないというとってもひどい指し手なので枝刈りする。
				if (futilityValue <= alpha)
				{
					bestValue = std::max(bestValue, futilityValue);
					continue;
				}

				// futilityBaseはこの局面のevalにmargin値を加算しているのだが、それがalphaを超えないし、
				// かつseeがプラスではない指し手なので悪い手だろうから枝刈りしてしまう。

				if (futilityBase <= alpha && !pos.see_ge(move, VALUE_ZERO + 1))
				{
					bestValue = std::max(bestValue, futilityBase);
					continue;
				}
			}

			//
			//  Detect non-capture evasions
			//

			// 駒を取らない王手回避の指し手はよろしくない可能性が高いのでこれは枝刈りしてしまう。
			// 成りでない && seeが負の指し手はNG。王手回避でなくとも、同様。

			// ここ、わりと棋力に影響する。下手なことするとR30ぐらい変わる。

			// Stockfish 12のコード
			// Do not search moves with negative SEE values
			if (    bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				// !ss->inCheck
				// これ不要らしい。cf. https://github.com/official-stockfish/Stockfish/commit/392b529c3f52103ad47ad096b86103c17758cb4f
				&& !pos.see_ge(move))
				continue;


			// TODO : prefetchは、入れると遅くなりそうだが、many coreだと違うかも。
			// Speculative prefetch as early as possible
			//prefetch(TT.first_entry(pos.key_after(move)));

			// -----------------------
			//     局面を1手進める
			// -----------------------

			// 指し手の合法性の判定は直前まで遅延させたほうが得。
			// (これが非合法手である可能性はかなり低いので他の判定によりskipされたほうが得)
			if (!pos.legal(move))
			{
				moveCount--;
				continue;
			}

			// 現在このスレッドで探索している指し手を保存しておく。
			ss->currentMove = move;

			ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
																	  [captureOrPawnPromotion]
																	  [to_sq(move)]
																	  [pos.moved_piece_after(move)];


			// MoveCountベースの枝刈り
			// ※ Stockfish12でqsearch()にも導入された。
			// 成りは歩だけに限定してあるが、これが適切かどうかはよくわからない。(大抵計測しても誤差ぐらいでしかない…)
			if (!captureOrPawnPromotion
				//&& moveCount
				// →　これは誤りらしい
				// cf. https://github.com/official-stockfish/Stockfish/commit/a260c9a8a24a2630a900efc3821000c3481b0c5d
				&& bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				&& (*contHist[0])[to_sq(move)][pos.moved_piece_after(move)] < CounterMovePruneThreshold
				&& (*contHist[1])[to_sq(move)][pos.moved_piece_after(move)] < CounterMovePruneThreshold)
				continue;

			// 1手動かして、再帰的にqsearch()を呼ぶ
			pos.do_move(move, st, givesCheck);
			value = -qsearch<NT>(pos, ss + 1, -beta, -alpha, depth - 1);
			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// bestValue(≒alpha値)を更新するのか
			if (value > bestValue)
			{
				bestValue = value;

				if (value > alpha)
				{
					bestMove = move;

					if (PvNode)  // Update pv even in fail-high case
					// fail-highの場合もPVは更新する。
						update_pv(ss->pv, move, (ss + 1)->pv);

					if (PvNode && value < beta) // Update alpha here!
						// alpha値の更新はこのタイミングで良い。
						// なぜなら、このタイミング以外だと枝刈りされるから。(else以下を読むこと)
						alpha = value;

					else
						break; // Fail high
				}
			}
		}

		// -----------------------
		// 指し手を調べ終わった
		// -----------------------

		// 王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである。
		// どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
		// わかるわけだし、そもそもこのnodeが詰みだとわかるとこのnodeに再訪問する確率は極めて低く、
		// 置換表に保存しても置換表を汚すだけでほとんど得をしない。(レアケースなのでほとんど損もしないが)
		 
		// ※　計測したところ、置換表に保存したほうがわずかに強かったが、有意差ではなさげだし、
		// Stockfish10のコードが保存しないコードになっているので保存しないことにする。

		// 【計測資料 26.】 qsearchで詰みのときに置換表に保存する/しない。
		if (ss->inCheck && bestValue == -VALUE_INFINITE)
			return mated_in(ss->ply); // rootからの手数による詰みである。

		// 詰みではなかったのでこれを書き出す。
		tte->save(posKey, value_to_tt(bestValue, ss->ply), pvHit,
			bestValue >= beta ? BOUND_LOWER :
			PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
			ttDepth, bestMove, ss->staticEval);

		// 置換表には abs(value) < VALUE_INFINITEの値しか書き込まないし、この関数もこの範囲の値しか返さない。
		// しかし置換表が衝突した場合はそうではない。3手詰めの局面で、置換表衝突により1手詰めのスコアが
		// 返ってきた場合がそれである。

		// ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));

		// このnodeはrootからss->ply手進めた局面なのでここでss->plyより短い詰みがあるのはおかしいが、
		// この関数はそんな値を返してしまう。しかしこれは通常探索ならば次のnodeでの
		// mate distance pruningで補正されるので問題ない。
		// また、VALUE_INFINITEはint16_tの最大値よりMAX_PLY以上小さいなのでオーバーフローの心配はない。
		//
		// よってsearch(),qsearch()のassertは次のように書くべきである。

		ASSERT_LV3(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

		return bestValue;
	}


	// 詰みのスコアは置換表上は、このnodeからあと何手で詰むかというスコアを格納する。
	// しかし、search()の返し値は、rootからあと何手で詰むかというスコアを使っている。
	// (こうしておかないと、do_move(),undo_move()するごとに詰みのスコアをインクリメントしたりデクリメントしたり
	// しないといけなくなってとても面倒くさいからである。)
	// なので置換表に格納する前に、この変換をしなければならない。
	// 詰みにまつわるスコアでないなら関係がないので何の変換も行わない。
	// ply : root node からの手数。(ply_from_root)
	Value value_to_tt(Value v, int ply) {

		//		assert(v != VALUE_NONE);
		ASSERT_LV3(-VALUE_INFINITE < v && v < VALUE_INFINITE);

		return  v >= VALUE_TB_WIN_IN_MAX_PLY ? v + ply
			  : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
	}

	// value_to_tt()の逆関数
	// ply : root node からの手数。(ply_from_root)
	Value value_from_tt(Value v, int ply) {

		if (v == VALUE_NONE)
			return VALUE_NONE;

		if (v >= VALUE_TB_WIN_IN_MAX_PLY)
		{
			//if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c)
			//	return VALUE_MATE_IN_MAX_PLY - 1; // do not return a potentially false mate score

			return v - ply;
		}

		if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
		{
			//if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c)
			//	return VALUE_MATED_IN_MAX_PLY + 1; // do not return a potentially false mate score

			return v + ply;
		}

		return v;
	}

	// PV lineをコピーする。
	// pv に move(1手) + childPv(複数手,末尾MOVE_NONE)をコピーする。
	// 番兵として末尾はMOVE_NONEにすることになっている。
	void update_pv(Move* pv, Move move, Move* childPv) {

		for (*pv++ = move; childPv && *childPv != MOVE_NONE; )
			*pv++ = *childPv++;
		*pv = MOVE_NONE;
	}

	// -----------------------
	//     Statsのupdate
	// -----------------------

	// update_all_stats()は、bestmoveが見つかったときにそのnodeの探索の終端で呼び出される。
	// 統計情報一式を更新する。

	void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
		Move* quietsSearched, int quietCount, Move* capturesSearched, int captureCount, Depth depth) {

		int bonus1, bonus2;
		Color us = pos.side_to_move();
		Thread* thisThread = pos.this_thread();
		CapturePieceToHistory& captureHistory = thisThread->captureHistory;
		Piece moved_piece = pos.moved_piece_after(bestMove);
		PieceType captured = type_of(pos.piece_on(to_sq(bestMove)));

		bonus1 = stat_bonus(depth + 1);
		bonus2 = bestValue > beta + PawnValue ? bonus1               // larger bonus
			: stat_bonus(depth);   // smaller bonus

		if (!pos.capture_or_promotion(bestMove))
		{
			update_quiet_stats(pos, ss, bestMove, bonus2, depth);

			// Decrease all the non-best quiet moves
			for (int i = 0; i < quietCount; ++i)
			{
				thisThread->mainHistory[from_to(quietsSearched[i])][us] << -bonus2;
				// Stockfishは[Color][from_to]の順なので注意。

				update_continuation_histories(ss, pos.moved_piece_after(quietsSearched[i]), to_sq(quietsSearched[i]), -bonus2);
			}
		}
		else
			captureHistory[to_sq(bestMove)][moved_piece][captured] << bonus1;
			// Stockfishは[pc][to][captured]の順なので注意。

		// Extra penalty for a quiet early move that was not a TT move or main killer move in previous ply when it gets refuted
		// (ss-1)->ttHit : 一つ前のnodeで置換表にhitしたか
		if (((ss - 1)->moveCount == 1 + (ss - 1)->ttHit || ((ss - 1)->currentMove == (ss - 1)->killers[0]))
			&& !pos.captured_piece())
			update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -bonus1);

		// 捕獲する指し手でベストではなかったものをすべて減点する。
		for (int i = 0; i < captureCount; ++i)
		{
			moved_piece = pos.moved_piece_after(capturesSearched[i]);
			captured = type_of(pos.piece_on(to_sq(capturesSearched[i])));
			captureHistory[to_sq(capturesSearched[i])][moved_piece][captured] << -bonus1;
			// Stockfishは[pc][to][captured]の順なので注意。
		}
	}

	// update_continuation_histories() は、形成された手のペアの履歴を更新します。
	// 1,2,4,6手前の指し手と現在の指し手との指し手ペアによってcontinuationHistoryを更新する。
	// 1手前に対する現在の指し手 ≒ counterMove  (応手)
	// 2手前に対する現在の指し手 ≒ followupMove (継続手)
	// 4手前に対する現在の指し手 ≒ followupMove (継続手)
	// ※　Stockfish 10で6手前も見るようになった。
	void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus)
	{
		for (int i : {1, 2, 4, 6})
		{
			if (ss->inCheck && i > 2)
				break;
			if (is_ok((ss - i)->currentMove))
				(*(ss - i)->continuationHistory)[to][pc] << bonus;
		}
	}

	// update_quiet_stats()は、新しいbest moveが見つかったときに指し手の並べ替えheuristicsを更新する。
	// 具体的には駒を取らない指し手のstat tables、killer等を更新する。

	// move      = これが良かった指し手
	void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus , int depth)
	{
		//   killerのupdate

		// killer 2本しかないので[0]と違うならいまの[0]を[1]に降格させて[0]と差し替え
		if (ss->killers[0] != move)
		{
			ss->killers[1] = ss->killers[0];
			ss->killers[0] = move;
		}

		//   historyのupdate
		Color us = pos.side_to_move();

		Thread* thisThread = pos.this_thread();
		thisThread->mainHistory[from_to(move)][us] << bonus;
		update_continuation_histories(ss, pos.moved_piece_after(move), to_sq(move), bonus);

		// 将棋はチェスと違って元の場所に戻れない駒が多いのでreverse_move()を用いるhistory updateは
		// 大抵、悪影響しかない。
		// また、reverse_move()を用いるならば、ifの条件式に " && !is_drop(move)"が要ると思う。

		// Stockfish12のコード
		//if (type_of(pos.moved_piece_after(move)) != PAWN)
		//	thisThread->mainHistory[from_to(reverse_move(move))][us] << -bonus;

#if 1
		// 【計測資料 37.】 update_quiet_stats()で、歩以外に対してreverse_moveにペナルティ。

		if (type_of(pos.moved_piece_after(move)) != PAWN && !is_drop(move))
			thisThread->mainHistory[from_to(reverse_move(move))][us] << -bonus;
#endif

		if (is_ok((ss - 1)->currentMove))
		{
			// 直前に移動させた升(その升に移動させた駒がある。今回の指し手はcaptureではないはずなので)
			Square prevSq = to_sq((ss - 1)->currentMove);
			thisThread->counterMoves[prevSq][pos.piece_on(prevSq)] = move;
		}

		if (depth > 11 && ss->ply < MAX_LPH)
			thisThread->lowPlyHistory[ss->ply][from_to(move)] << stat_bonus(depth - 7);
	}

	// 手加減が有効であるなら、best moveを'level'に依存する統計ルールに基づくRootMovesの集合から選ぶ。
	// Heinz van Saanenのアイデア。
	Move Skill::pick_best(size_t multiPV) {

		const RootMoves& rootMoves = Threads.main()->rootMoves;
		static PRNG rng(now()); // 乱数ジェネレーターは非決定的であるべき。

		// RootMovesはすでにscoreで降順にソートされている。
		Value topScore = rootMoves[0].score;
		int delta = std::min(topScore - rootMoves[multiPV - 1].score, (Value)PawnValue);
		int weakness = 120 - 2 * level;
		int maxScore = -VALUE_INFINITE;

		// best moveを選ぶ。それぞれの指し手に対して弱さに依存する2つのterm(用語)を追加する。
		// 1つは、決定的で、弱いレベルでは大きくなるもので、1つはランダムである。
		// 次に得点がもっとも高い指し手を選択する。
		for (size_t i = 0; i < multiPV; ++i)
		{
			// これが魔法の公式
			int push = (weakness * int(topScore - rootMoves[i].score)
				+ delta * (rng.rand<unsigned>() % weakness)) / 128;

			if (rootMoves[i].score + push >= maxScore)
			{
				maxScore = rootMoves[i].score + push;
				best = rootMoves[i].pv[0];
			}
		}

		return best;
	}

} // namespace


// 残り時間をチェックして、時間になっていればThreads.stopをtrueにする。
// main threadからしか呼び出されないのでロジックがシンプルになっている。
void MainThread::check_time()
{
	// 4096回に1回ぐらいのチェックで良い。
	if (--callsCnt > 0)
		return;

	// Limits.nodesが指定されているときは、そのnodesの0.1%程度になるごとにチェック。
	// さもなくばデフォルトの値を使う。
	// このデフォルト値、ある程度小さくしておかないと、通信遅延分のマージンを削ったときに
	// ちょうど1秒を超えて計測2秒になり、損をしうるという議論があるようだ。
	// cf. Check the clock every 1024 nodes : https://github.com/official-stockfish/Stockfish/commit/8db75dd9ec05410136898aa2f8c6dc720b755eb8
	// Stockfish9から10になったときに4096→1024に引き下げられた。
	// main threadでしか判定しないからチェックに要するコストは微小だと思われる。
	callsCnt = Limits.nodes ? std::min(1024, int(Limits.nodes / 1024)) : 1024;

	// 1秒ごとにdbg_print()を呼び出す処理。
	// dbg_print()は、dbg_hit_on()呼び出しによる統計情報を表示する。

	static TimePoint lastInfoTime = now();
	TimePoint tick = now();

	// 1秒ごとに
	if (tick - lastInfoTime >= 1000)
	{
		lastInfoTime = tick;
		dbg_print();
	}

	// ponder中においては、GUIがstopとかponderhitとか言ってくるまでは止まるべきではない。
	if (ponder)
		return;

	// "ponderhit"時は、そこからの経過時間で考えないと、elapsed > Time.maximum()になってしまう。
	// elapsed_from_ponderhit()は、"ponderhit"していないときは"go"コマンドからの経過時間を返すのでちょうど良い。
	TimePoint elapsed = Time.elapsed_from_ponderhit();

	// 今回のための思考時間を完璧超えているかの判定。

	// 反復深化のループ内でそろそろ終了して良い頃合いになると、Time.search_endに停止させて欲しい時間が代入される。
	// (それまではTime.search_endはゼロであり、これは終了予定時刻が未確定であることを示している。)
	// ※　前半部分、やねうら王、独自実装。
	if ((Limits.use_time_management() &&
		(elapsed > Time.maximum() || (Time.search_end > 0 && elapsed > Time.search_end)))
		|| (Limits.movetime && elapsed >= Limits.movetime)
		|| (Limits.nodes && Threads.nodes_searched() >= (uint64_t)Limits.nodes)
		)
		Threads.stop = true;
}

// --- Stockfishの探索のコード、ここまで。

// 探索パラメーターの初期化
void init_param()
{
	// -----------------------
	//   parameters.hの動的な読み込み
	// -----------------------

#if defined (TUNING_SEARCH_PARAMETERS)
	{
		std::vector<std::string> param_names = {
			// このheader fileは、yaneuraou-param.hからparam_conv.pyによって自動生成される。
			#include "param/yaneuraou-param-string.h"
		};

		std::vector<int*> param_vars = {
			// このheader fileは、yaneuraou-param.hからparam_conv.pyによって自動生成される。
			#include "param/yaneuraou-param-array.h"
		};

		std::fstream fs;

		std::string path = Path::Combine(Directory::GetCurrentFolder(), PARAM_FILE);
		
		fs.open( path.c_str(), std::ios::in);
		if (fs.fail())
		{
			std::cout << "info string Error! : can't read " << path << std::endl;
			return;
		}

		size_t count = 0;
		std::string line, last_line;

		// bufのなかにある部分文字列strの右側にある数値を読む。
		auto get_num = [](const std::string& buf, const std::string& str)
		{
			auto pos = buf.find(str);
			ASSERT_LV3(pos != std::string::npos);

			auto s = buf.substr(pos + str.size());
			if (s.empty() || !(('0' <= s[0] && s[0] <= '9') || s[0] == '-' || s[0] == ' '))
			{
				std::cout << "Error : Parse Error " << buf << "   ==>   " << s << std::endl;
				return 0;
			}

			return stoi(s);
			// ここで落ちてたら、paramファイルとして、変な文をparseしている。
		};

		std::vector<bool> founds(param_vars.size());

		while (!fs.eof())
		{
			getline(fs, line);
			if (line.find("PARAM_DEFINE") != std::string::npos)
			{
				for (size_t i = 0; i < param_names.size(); ++i)
					{
					auto pos = line.find(param_names[i]);
					if (pos != std::string::npos)
					{
						char c = line[pos + param_names[i].size()];
						// ここ、パラメーター名のあと、スペースか"="か来るのを確認しておかないと
						// "PARAM_T1" が "PARAM_T10" に誤爆する。
						if (!(c == '\t' || c == ' ' || c == '='))
							continue;

						count++;

						// "="の右側にある数値を読む。
						*param_vars[i] = get_num(line, "=");

						// 見つかった
						founds[i] = true;

#if defined(USE_RANDOM_PARAMETERS)
						// PARAM_DEFINEの一つ前の行には次のように書いてあるはずなので、
						// USE_RANDOM_PARAMETERSのときは、このstepをプラスかマイナス方向に加算してやる。
						// ただし、fixedと書いてあるパラメーターに関しては除外する。
						// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。

						// [PARAM] min:100,max:240,step:3,interval:1,time_rate:1,fixed

						// "fixed"と書かれているパラメーターはないものとして扱う。
						if (last_line.find("fixed") != std::string::npos)
						{
							param_names[i] = "FIXED";
							goto NEXT;
						}

						static PRNG rand;
						int param_step = get_num(last_line, "step:");
						int param_min = get_num(last_line, "min:");
						int param_max = get_num(last_line, "max:");
						int param_interval = get_num(last_line, "interval:");

						// 現在の値
						int v = *param_vars[i];

						// とりうる値の候補
						std::vector<int> a;

						for (int j = 0; j <= param_interval; ++j)
						{
							// j==0のときは同じ値であり、これはのちに除外される。
							a.push_back(std::max(v - param_step * j, param_min));
							a.push_back(std::min(v + param_step * j, param_max));
						}

						// 重複除去。
						// 1) std::unique()は隣接要素しか削除しないので事前にソートしている。
						// 2) std::unique()では末尾にゴミが残るのでそれをerase()で消している。
						std::sort(a.begin(), a.end());
						a.erase(std::unique(a.begin(), a.end()), a.end());

						// 残ったものから1つをランダムに選択
						if (a.size() == 0)
						{
							std::cout << "Error : param is out of range -> " << line << std::endl;
						} else {
							*param_vars[i] = a[rand.rand(a.size())];
						}
#endif

						//            cout << param_names[i] << " = " << *param_vars[i] << endl;
						goto NEXT;
					}
				}
				std::cout << "Error : param not found! in yaneuraou-param.h -> " << line << std::endl;

			NEXT:;
			}
			last_line = line; // 1つ前の行を記憶しておく。
		}
		fs.close();

		// 読み込んだパラメーターの数が合致しないといけない。
		// 見つかっていなかったパラメーターを表示させる。
		if (count != param_names.size())
		{
			for (size_t i = 0; i < founds.size(); ++i)
				if (!founds[i])
					std::cout << "Error : param not found in " << path << " -> " << param_names[i] << std::endl;
		}

#if defined(ENABLE_OUTPUT_GAME_RESULT)
		{
			if (!result_log.is_open())
				result_log.open(Options["PARAMETERS_LOG_FILE_PATH"], std::ios::app);
			// 今回のパラメーターをログファイルに書き出す。
			for (size_t i = 0; i < param_names.size(); ++i)
			{
				if (param_names[i] == "FIXED")
					continue;

				result_log << param_names[i] << ":" << *param_vars[i] << ",";
			}
			result_log << std::endl << std::flush;
		}
#endif

		// Evalのパラメーター初期化
		// 上のコードでパラメーターが変更された可能性があるのでこのタイミングで再度呼び出す。
		Eval::init();

		}

#endif
	}

// --- 学習時に用いる、depth固定探索などの関数を外部に対して公開

#if defined (EVAL_LEARN)

namespace Learner
{
	// 学習用に、1つのスレッドからsearch,qsearch()を呼び出せるようなスタブを用意する。
	// いまにして思えば、AperyのようにSearcherを持ってスレッドごとに置換表などを用意するほうが
	// 良かったかも知れない。

	// 学習のための初期化。
	// Learner::search(),Learner::qsearch()から呼び出される。
	void init_for_search(Position& pos, Stack* ss)
	{

		// RootNodeはss->ply == 0がその条件。
		// ゼロクリアするので、ss->ply == 0となるので大丈夫…。

		memset(ss - 7, 0, 10 * sizeof(Stack));

		// Search::Limitsに関して
		// このメンバー変数はglobalなので他のスレッドに影響を及ぼすので気をつけること。
		{
			auto& limits = Search::Limits;

			// 探索を"go infinite"コマンド相当にする。(time managementされると困るため)
			limits.infinite = true;

			// PVを表示されると邪魔なので消しておく。
			limits.silent = true;

			// これを用いると各スレッドのnodesを積算したものと比較されてしまう。ゆえに使用しない。
			limits.nodes = 0;

			// depthも、Learner::search()の引数として渡されたもので処理する。
			limits.depth = 0;

			// 引き分け付近の手数で引き分けの値が返るのを防ぐために大きな値にしておく。
			limits.max_game_ply = 1 << 16;

			// 入玉ルールも入れておかないと引き分けになって決着つきにくい。
			limits.enteringKingRule = EnteringKingRule::EKR_27_POINT;
		}

		// DrawValueの設定
		{
			// スレッドごとに用意してないので
			// 他のスレッドで上書きされかねない。仕方がないが。
			// どうせそうなるなら、0にすべきだと思う。
			drawValueTable[REPETITION_DRAW][BLACK] = VALUE_ZERO;
			drawValueTable[REPETITION_DRAW][WHITE] = VALUE_ZERO;
		}

		// this_threadに関して。
		{
			auto th = pos.this_thread();

			th->completedDepth = 0;
			th->selDepth = 0;
			th->rootDepth = 0;

			// 探索ノード数のゼロ初期化
			th->nodes = 0;

			// history類を全部クリアする。この初期化は少し時間がかかるし、探索の精度はむしろ下がるので善悪はよくわからない。
			// th->clear();

			for (int i = 7; i > 0; i--)
				(ss - i)->continuationHistory = &th->continuationHistory[0][0][SQ_ZERO][NO_PIECE];

			// rootMovesの設定
			auto& rootMoves = th->rootMoves;

			rootMoves.clear();
			for (auto m : MoveList<LEGAL>(pos))
				rootMoves.push_back(Search::RootMove(m));

			ASSERT_LV3(!rootMoves.empty());

			// 学習用の実行ファイルではスレッドごとに置換表を持っているので
			// 探索前に自分(のスレッド用)の置換表の世代カウンターを回してやる。
			th->tt.new_search();

			// 最近の置換表の平均ヒット率の初期化。
			th->ttHitAverage = TtHitAverageWindow * TtHitAverageResolution / 2;
		}
	}

	// 読み筋と評価値のペア。Learner::search(),Learner::qsearch()が返す。
	typedef std::pair<Value, std::vector<Move> > ValueAndPV;

	// 静止探索。
	//
	// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。
	// 　また、Threads.stopが来ると探索を中断してしまうので、そのときのPVは正しくない。
	// 　search()から戻ったあと、Threads.stop == trueなら、その探索結果を用いてはならない。
	// 　あと、呼び出し前は、Threads.stop == falseの状態で呼び出さないと、探索を中断して返ってしまうので注意。
	//
	// 詰まされている場合は、PV配列にMOVE_RESIGNが返る。
	//
	// 引数でalpha,betaを指定できるようにしていたが、これがその窓で探索したときの結果を
	// 置換表に書き込むので、その窓に対して枝刈りが出来るような値が書き込まれて学習のときに
	// 悪い影響があるので、窓の範囲を指定できるようにするのをやめることにした。
	ValueAndPV qsearch(Position& pos)
	{
		Stack stack[MAX_PLY + 10], *ss = stack + 7;
		Move pv[MAX_PLY + 1];
		std::vector<Move> pvs;

		init_for_search(pos, ss);
		ss->pv = pv; // とりあえずダミーでどこかバッファがないといけない。

		// 詰まされているのか
		if (pos.is_mated())
		{
			pvs.push_back(MOVE_RESIGN);
			return ValueAndPV(mated_in(/*ss->ply*/ 0 + 1), pvs);
		}

		auto bestValue = ::qsearch<PV>(pos, ss, -VALUE_INFINITE, VALUE_INFINITE, 0);

		// 得られたPVを返す。
		for (Move* p = &ss->pv[0]; is_ok(*p); ++p)
			pvs.push_back(*p);

		return ValueAndPV(bestValue, pvs);
	}

	// 通常探索。深さdepth(整数で指定)。
	// 3手読み時のスコアが欲しいなら、
	//   auto v = search(pos,3);
	// のようにすべし。
	// v.firstに評価値、v.secondにPVが得られる。
	// multi pvが有効のときは、pos.this_thread()->rootMoves[N].pvにそのPV(読み筋)の配列が得られる。
	// multi pvの指定はこの関数の引数multiPVで行なう。(Options["MultiPV"]の値は無視される)
	// 
	// rootでの宣言勝ち判定はしないので(扱いが面倒なので)、ここでは行わない。
	// 呼び出し側で処理すること。
	//
	// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。
	// 　また、Threads.stopが来ると探索を中断してしまうので、そのときのPVは正しくない。
	// 　search()から戻ったあと、Threads.stop == trueなら、その探索結果を用いてはならない。
	// 　あと、呼び出し前は、Threads.stop == falseの状態で呼び出さないと、探索を中断して返ってしまうので注意。

	ValueAndPV search(Position& pos, int depth_, size_t multiPV /* = 1 */, u64 nodesLimit /* = 0 */)
	{
		std::vector<Move> pvs;

		Depth depth = depth_;
		if (depth < 0)
			return std::pair<Value, std::vector<Move>>(Eval::evaluate(pos), std::vector<Move>());

		if (depth == 0)
			return qsearch(pos);

		Stack stack[MAX_PLY + 10], *ss = stack + 7;
		Move pv[MAX_PLY + 1];

		init_for_search(pos, ss);

		ss->pv = pv; // とりあえずダミーでどこかバッファがないといけない。

		// this_threadに関連する変数の初期化
		auto th = pos.this_thread();
		auto& rootDepth = th->rootDepth;
		auto& pvIdx = th->pvIdx;
		auto& rootMoves = th->rootMoves;
		auto& completedDepth = th->completedDepth;
		auto& selDepth = th->selDepth;

		// bestmoveとしてしこの局面の上位N個を探索する機能
		//size_t multiPV = Options["MultiPV"];

		// この局面での指し手の数を上回ってはいけない
		multiPV = std::min(multiPV, rootMoves.size());

		// ノード制限にMultiPVの値を掛けておかないと、depth固定、MultiPVありにしたときに1つの候補手に同じnodeだけ思考したことにならない。
		nodesLimit *= multiPV;

		Value alpha = -VALUE_INFINITE;
		Value beta = VALUE_INFINITE;
		Value delta = -VALUE_INFINITE;
		Value bestValue = -VALUE_INFINITE;

		while (++rootDepth <= depth
			// node制限を超えた場合もこのループを抜ける
			// 探索ノード数は、この関数の引数で渡されている。
			&& !(nodesLimit /*node制限あり*/ && th->nodes.load(std::memory_order_relaxed) >= nodesLimit)
			)
		{
			for (RootMove& rm : rootMoves)
				rm.previousScore = rm.score;

			// MultiPV
			for (pvIdx = 0; pvIdx < multiPV && !Threads.stop; ++pvIdx)
			{
				// それぞれのdepthとPV lineに対するUSI infoで出力するselDepth
				selDepth = 0;

				// depth 4以上においてはaspiration searchに切り替える。
				if (rootDepth >= 4)
				{
					delta = Value(PARAM_ASPIRATION_SEARCH_DELTA);

					Value p = rootMoves[pvIdx].previousScore;

					alpha = std::max(p - delta, -VALUE_INFINITE);
					beta = std::min(p + delta, VALUE_INFINITE);
				}

				// aspiration search
				int failedHighCnt = 0;
				while (true)
				{
					Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt);
					bestValue = ::search<PV>(pos, ss, alpha, beta, adjustedDepth, false);

					stable_sort(rootMoves.begin() + pvIdx, rootMoves.end());
					//my_stable_sort(pos.this_thread()->thread_id(),&rootMoves[0] + pvIdx, rootMoves.size() - pvIdx);

					// fail low/highに対してaspiration windowを広げる。
					// ただし、引数で指定されていた値になっていたら、もうfail low/high扱いとしてbreakする。
					if (bestValue <= alpha)
					{
						beta = (alpha + beta) / 2;
						alpha = std::max(bestValue - delta, -VALUE_INFINITE);

						failedHighCnt = 0;
						//if (mainThread)
						//    mainThread->stopOnPonderhit = false;

					}
					else if (bestValue >= beta)
					{
						beta = std::min(bestValue + delta, VALUE_INFINITE);
						++failedHighCnt;
					}
					else
						break;

					delta += delta / 4 + 5;
					ASSERT_LV3(-VALUE_INFINITE <= alpha && beta <= VALUE_INFINITE);

					// 暴走チェック
					//ASSERT_LV3(th->nodes.load(std::memory_order_relaxed) <= 1000000 );
				}

				stable_sort(rootMoves.begin(), rootMoves.begin() + pvIdx + 1);
				//my_stable_sort(pos.this_thread()->thread_id() , &rootMoves[0] , pvIdx + 1);

			} // multi PV

			completedDepth = rootDepth;
		}

		// このPV、途中でNULL_MOVEの可能性があるかも知れないので排除するためにis_ok()を通す。
		// →　PVなのでNULL_MOVEはしないことになっているはずだし、
		//     MOVE_WINも突っ込まれていることはない。(いまのところ)
		for (Move move : rootMoves[0].pv)
		{
			if (!is_ok(move))
				break;
			pvs.push_back(move);
		}

		//sync_cout << rootDepth << sync_endl;

		// multiPV時を考慮して、rootMoves[0]のscoreをbestValueとして返す。
		bestValue = rootMoves[0].score;

		return ValueAndPV(bestValue, pvs);
	}

}
#endif

#endif // YANEURAOU_ENGINE
