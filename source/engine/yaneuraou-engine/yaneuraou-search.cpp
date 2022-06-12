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
	// 歩を100とする。例えば、この値を -100にすると引き分けの局面は評価値が -100とみなされる。

	// 千日手での引き分けを回避しやすくなるように、デフォルト値を-2。
	// ちなみに、-2にしてあるのは、
	//  int draw_value = Options["DrawValueBlack"] * PawnValue / 100; でPawnValueが100より小さいので
	// 1だと切り捨てられてしまうからである。

	// Stockfishでは"Contempt"というエンジンオプションであったが、先後の区別がつけられないし、
	// 分かりづらいので変更した。

	o["DrawValueBlack"] << Option(-2, -30000, 30000);
	o["DrawValueWhite"] << Option(-2, -30000, 30000);

	//  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
	o["PvInterval"]     << Option(300, 0, 100000);

	// 投了スコア
	o["ResignValue"]    << Option(99999, 0, 99999);

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
	o["ConsiderationMode"] << Option(true);

	// fail low/highのときにPVを出力するかどうか。
	o["OutputFailLHPV"] << Option(true);

#if defined(YANEURAOU_ENGINE_NNUE)
	// NNUEのFV_SCALEの値
	o["FV_SCALE"] << Option(16, 1, 128);
#endif
}

// パラメーターのランダム化のときには、
// USIの"gameover"コマンドに対して、それをログに書き出す。
void gameover_handler(const std::string& cmd)
{
#if defined(ENABLE_OUTPUT_GAME_RESULT)
	result_log << cmd << std::endl << std::flush;
#endif
}

#if defined(YANEURAOU_ENGINE_NNUE)
void init_fv_scale() {
	Eval::NNUE::FV_SCALE = (int)Options["FV_SCALE"];
}
#endif


// "isready"に対して探索パラメーターを動的にファイルから読み込んだりして初期化するための関数。
void init_param();

// "go"コマンドに"wait_stop"が指定されていて、かつ、いまbestmoveを返す準備ができたので
// それをGUIに通知して、"stop"が送られてくるのを待つ。
void output_time_to_return_bestmove()
{
	Threads.main()->time_to_return_bestmove = true;

	// ここでPVを出力しておきたいが、rootでのalpha,betaが確定しないので出力できない。

	sync_cout << "info string time to return bestmove." << sync_endl;
}

// -----------------------
//   やねうら王2022探索部
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

	// Different node types, used as a template parameter
	// 探索しているnodeの種類
	enum NodeType { NonPV, PV , Root};

	// Futility margin
	// RazoringはStockfish12で効果がないとされてしまい除去された。

	// depth(残り探索深さ)に応じたfutility margin。
	Value futility_margin(Depth d, bool improving) {
		return Value(PARAM_FUTILITY_MARGIN_ALPHA1/*168*/ * (d - improving));
	}

	// 【計測資料 30.】　Reductionのコード、Stockfish 9と10での比較

	// Reductions lookup table, initialized at startup
	// 探索深さを減らすためのReductionテーブル。起動時に初期化する。
	int Reductions[MAX_MOVES]; // [depth or moveNumber]

	// 残り探索深さをこの深さだけ減らす。d(depth)とmn(move_count)
	// i(improving)とは、評価値が2手前から上がっているかのフラグ。上がっていないなら
	// 悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
	// delta, rootDelta は、staticEvalとchildのeval(value)の差が一貫して低い時にreduction量を増やしたいので、
	// そのためのフラグ。(これがtrueだとreduction量が1増える)
	Depth reduction(bool i, Depth d, int mn, Value delta, Value rootDelta) {
		int r = Reductions[d] * Reductions[mn];
		return (r + PARAM_REDUCTION_ALPHA/*1463*/ - int(delta) * 1024 / int(rootDelta)) / 1024 + (!i && r > PARAM_REDUCTION_BETA/*1010*/);
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
	// 
	// 注意 : この関数は 0以下を返してはならない。例えば0を返すと、moveCount == 0でmoveCountPruningが trueになり、
	//   MovePicker::move_next(true)で呼び出されることとなり、QUIETの指し手が生成されずに、whileループを抜けて
	//   moveCount == 0 かつ、ループを抜けたので合法手がない判定になり、詰みの局面だと錯覚する。
	constexpr int futility_move_count(bool improving, int depth) {
		return (3 + depth * depth) / (2 - improving);
	}

	// History and stats update bonus, based on depth
	// depthに基づく、historyとstatsのupdate bonus
	int stat_bonus(Depth d) {
		return std::min((9 * d + 270) * d - 311, 2145);
	}

	// チェスでは、引き分けが0.5勝扱いなので引き分け回避のための工夫がしてあって、
	// 以下のようにvalue_drawに揺らぎを加算することによって探索を固定化しない(同じnodeを
	// 探索しつづけて千日手にしてしまうのを回避)工夫がある。
	// 将棋の場合、普通の千日手と連続王手の千日手と劣等局面による千日手(循環？)とかあるので
	// これ導入するのちょっと嫌。

	// Add a small random component to draw evaluations to avoid 3fold-blindness
	// 引き分け時の評価値VALUE_DRAW(0)の代わりに±1の乱数みたいなのを与える。
	Value value_draw(Thread* thisThread) {
		return VALUE_DRAW + Value(2 * (thisThread->nodes & 1) - 1);
	}

	// Skill structure is used to implement strength limit. If we have an uci_elo then
	// we convert it to a suitable fractional skill level using anchoring to CCRL Elo
	// (goldfish 1.13 = 2000) and a fit through Ordo derived Elo for match (TC 60+0.6)
	// results spanning a wide range of k values.
	//
	// Skill構造体は強さの制限の実装に用いられる。
	// (わざと手加減して指すために用いる)
	struct Skill {
		// skill_level : 手加減のレベル。20未満であれば手加減が有効。0が一番弱い。(R2000以上下がる)
		// uci_elo     : 0以外ならば、そのelo ratingになるように調整される。
		Skill(int skill_level, int uci_elo) {
			if (uci_elo)
				level = std::clamp(std::pow((uci_elo - 1346.6) / 143.4, 1 / 0.806), 0.0, 20.0);
			else
				level = double(skill_level);
		}

		// 手加減が有効であるか。
		bool enabled() const { return level < 20.0; }

		// SkillLevelがNなら探索深さもNぐらいにしておきたいので、
		// depthがSkillLevelに達したのかを判定する。
		bool time_to_pick(Depth depth) const { return depth == 1 + int(level); }

		// 手加減が有効のときはMultiPV = 4で探索
		Move pick_best(size_t multiPV);

		// SkillLevel
		double level;

		Move best = MOVE_NONE;
	};

	template <NodeType nodeType>
	Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

	template <NodeType nodeType>
	Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth = 0);

	Value value_to_tt(Value v, int ply);
	Value value_from_tt(Value v, int ply /*,int r50c */);
	void update_pv(Move* pv, Move move, Move* childPv);
	void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
	void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus);
	void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
		Move* quietsSearched, int quietCount, Move* capturesSearched, int captureCount, Depth depth);

	// perft() is our utility to verify move generation. All the leaf nodes up
	// to the given depth are generated and counted, and the sum is returned.
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

	// --- やねうら王独自拡張 ---

	// 指し手moveによってtoの地点の駒が捕獲できることがわかっている時の、駒を捕獲する価値
	// moveが成りの指し手である場合、その価値も上乗せして計算する。
	Value CapturePieceValueEg(const Position& pos, Move move)
	{
		return (Value)CapturePieceValue[pos.piece_on(to_sq(move))]
			+ (is_promote(move) ? (Value)ProDiffPieceValue[pos.piece_on(from_sq(move))] : VALUE_ZERO);
	}

} // namespace 


/// Search::init() is called at startup to initialize various lookup tables
// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
	// 時間がかかるものは、"isready"応答(Search::clear())でやるべき。

	// -----------------------
	//   テーブルの初期化
	// -----------------------

	// LMRで使うreduction tableの初期化
	//
	// pvとnon pvのときのreduction定数
	// 0.05とか変更するだけで勝率えらく変わる

	for (int i = 1; i < MAX_MOVES; ++i)
		Reductions[i] = int(20.81 * std::log(i));

	// Stockfish11.1で現在のスレッド数設定に依存する項が追加された。(以下コード)
	// このため、スレッド数が確定するisreadyタイミングで初期化する必要があるが、
	// Stockfish11.1のコードは、Search::init()でやっているがこれは起動時に呼び出されるので誤り。
	// つまりは、正しく実装されてないので、この項を追加する効果は怪しい。
	// やねうら王では追加しないでおく。
	//
	// ※　デフォルトのスレッド数設定が4だとして、 (std::log(4) / 2) という項を加えて強くなったにすぎない。

	//Reductions[i] = int((20.81 + std::log(Threads.size()) / 2) * std::log(i));
	// cf. Increase reductions with thread count : https://github.com/official-stockfish/Stockfish/commit/135caee606c86ade9e9c199ef469661c374eb9ba
}

/// Search::clear() resets search state to its initial value
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

	// -----------------------
	//   評価関数の定数を初期化
	// -----------------------

#if defined(YANEURAOU_ENGINE_NNUE)
	init_fv_scale();
#endif

}

/// MainThread::search() is started when the program receives the UCI 'go'
/// command. It searches from the root position and outputs the "bestmove".
// 探索開始時に(goコマンドなどで)呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::search()
{
	// ---------------------
	// やねうら王固有の初期化
	// ---------------------

	// root nodeにおける自分の手番
	Color us = rootPos.side_to_move();

	// --- 今回の思考時間の設定。
	// これは、ponderhitした時にponderhitにパラメーターが付随していれば
	// 再計算するする必要性があるので、いずれにせよ呼び出しておく必要がある。

	Time.init(Limits, us, rootPos.game_ply());

	// 今回、通常探索をしたかのフラグ(やねうら王独自拡張)
	// このフラグがtrueなら(定跡にhitしたり1手詰めを発見したりしたので)探索をスキップした。
	bool search_skipped = true;

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

	// 引き分け時の値として現在の手番に応じた値を設定してやる。

	int draw_value = (int)((us == BLACK ? Options["DrawValueBlack"] : Options["DrawValueWhite"]) * PawnValue / 100);

	// 探索のleaf nodeでは、相手番(root_color != side_to_move)である場合、 +draw_valueではなく、-draw_valueを設定してやらないと非対称な探索となって良くない。
	// 例) 自分は引き分けを勝ち扱いだと思って探索しているなら、相手は、引き分けを負けとみなしてくれないと非対称になる。
	drawValueTable[REPETITION_DRAW][ us] = Value(+draw_value);
	drawValueTable[REPETITION_DRAW][~us] = Value(-draw_value);

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
		sync_cout << "\nNodes searched: " << nodes << ", time " << Time.elapsed() << "ms.\n" << sync_endl;
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
		rootMoves.emplace_back(MOVE_RESIGN);
		// 評価値を用いないなら代入しなくて良いのだが(Stockfishはそうなっている)、
		// このあと、↓USI::pv()を呼び出したいので、scoreをきちんと設定しておいてやる。
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

	// ---------------------
	// Lazy SMPの結果を取り出す
	// ---------------------

	Thread* bestThread = this;

	// 最終的なPVを出力する。
	// ponder中/go infinite中であっても、ここに抜けてきている以上、全探索スレッドの停止が確認できた時点でPVは出力すべき。
	// "go infinite"の場合、詰みを発見してもそれがponderフラグの解除を待ってからだと、PVを返すのが遅れる。("stop"が来るまで返せない)
	// Stockfishもこうなっている。この作り、良くないように思うので、改良した。

	// final PVの出力をやり終えたのかのフラグ
	bool output_final_pv_done = false;

	// final PVの出力を行うlambda。
	auto output_final_pv = [&]()
	{
		if (!output_final_pv_done)
		{
			//Skill skill = Skill(Options["SkillLevel"], Options["USI_LimitStrength"] ? int(Options["USI_Elo"]) : 0);
			// ↑これでエンジンオプション2つも増えるのやだな…。気が向いたらサポートすることにする。

			Skill skill = Skill((int)Options["SkillLevel"], 0);

			// 並列して探索させていたスレッドのうち、ベストのスレッドの結果を選出する。
			if (   int(Options["MultiPV"]) == 1
				&& !Limits.depth
				&& !skill.enabled()
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

			// ベストな指し手として返すスレッドがmain threadではないのなら、
			// その読み筋は出力していなかったはずなのでここで読み筋を出力しておく。
			// ただし、これはiterationの途中で停止させているので中途半端なPVである可能性が高い。
			// 検討モードではこのPVを出力しない。
			// →　いずれにせよ、mateを見つけた時に最終的なPVを出力していないと、詰みではないscoreのPVが最終的な読み筋としてGUI上に
			//     残ることになるからよろしくない。PV自体は必ず出力すべきなのでは。
			if (/*bestThread != this &&*/ !Limits.silent && !Limits.consideration_mode)
				sync_cout << USI::pv(bestThread->rootPos, bestThread->completedDepth, -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

			output_final_pv_done = true;
		}
	};

	// ここで思考は完了したのでwait_stopの処理。
	// まだ思考が完了したことを通知していないならば。

	if (Limits.wait_stop && !Threads.main()->time_to_return_bestmove)
		output_time_to_return_bestmove();

	// 最大depth深さに到達したときに、ここまで実行が到達するが、
	// まだThreads.stopが生じていない。しかし、ponder中や、go infiniteによる探索の場合、
	// USI(UCI)プロトコルでは、"stop"や"ponderhit"コマンドをGUIから送られてくるまでbest moveを出力してはならない。
	// それゆえ、単にここでGUIからそれらのいずれかのコマンドが送られてくるまで待つ。
	// "stop"が送られてきたらThreads.stop == trueになる。
	// "ponderhit"が送られてきたらThreads.ponder == falseになるので、それを待つ。(stopOnPonderhitは用いない)
	// "go infinite"に対してはstopが送られてくるまで待つ。
	// ちなみにStockfishのほう、ここのコードに長らく同期上のバグがあった。
	// やねうら王のほうは、かなり早くからこの構造で書いていた。最近のStockfishではこの書き方に追随した。
	while (!Threads.stop && (ponder || Limits.infinite || Limits.wait_stop))
	{
		//	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
		// (思考のためには計算資源を使っていないので。)
		Tools::sleep(1);
		// →　Stockfishのコード、ここ、busy waitになっているが、さすがにそれは良くないと思う。

		// === やねうら王独自改良 ===
		// 　ここですべての探索スレッドが停止しているならば最終PVを出力してやる。
		if (!output_final_pv_done && Threads.search_finished() /* 全探索スレッドが探索を完了している */)
			output_final_pv();
	}

	Threads.stop = true;

	// 各スレッドが終了するのを待機する(開始していなければいないで構わない)
	Threads.wait_for_search_finished();

#if 0
	// When playing in 'nodes as time' mode, subtract the searched nodes from
	// the available ones before exiting.

	// nodes as time(時間としてnodesを用いるモード)のときは、利用可能なノード数から探索したノード数を引き算する。
	// 時間切れの場合、負の数になりうる。
	// 将棋の場合、秒読みがあるので秒読みも考慮しないといけない。

	if (Limits.npmsec)
		Time.availableNodes += Limits.inc[us] + Limits.byoyomi[us] - Threads.nodes_searched();
	// →　将棋と相性がよくないのでこの機能をサポートしないことにする。
#endif

	output_final_pv();

	// ---------------------
	// 指し手をGUIに返す
	// ---------------------

	// 次回の探索のときに何らか使えるのでベストな指し手の評価値を保存しておく。
	bestPreviousScore        = bestThread->rootMoves[0].score;
	bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

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

// 探索スレッド用の初期化(探索部と学習部と共通)
// やねうら王、独自拡張。
// ssにはstack+7が渡されるものとする。
void search_thread_init(Thread* th, Stack* ss , Move pv[])
{
	// 先頭10個を初期化しておけば十分。そのあとはsearch()の先頭でss+1,ss+2を適宜初期化していく。
	// RootNodeはss->ply == 0がその条件。
	// ゼロクリアするので、ss->ply == 0となるので大丈夫…。
	std::memset(ss - 7, 0, 10 * sizeof(Stack));

	// counterMovesをnullptrに初期化するのではなくNO_PIECEのときの値を番兵として用いる。
	for (int i = 7; i > 0; i--)
		(ss - i)->continuationHistory = &th->continuationHistory[0][0][SQ_ZERO][NO_PIECE]; // Use as a sentinel

	// Stack(探索用の構造体)上のply(手数)は事前に初期化しておけば探索時に代入する必要がない。
	for (int i = 0; i <= MAX_PLY + 2; ++i)
		(ss + i)->ply = i;

	// 最善応手列(Principal Variation)
	ss->pv = pv;

	// ---------------------
	//   移動平均を用いる統計情報の初期化
	// ---------------------


	// 千日手の時の動的なcontempt。これ、やねうら王では使わないことにする。
	//th->trend = VALUE_ZERO;

}


// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// Lazy SMPなので、置換表を共有しながらそれぞれのスレッドが勝手に探索しているだけ。
void Thread::search()
{
	// ---------------------
	//      variables
	// ---------------------

  // To allow access to (ss-7) up to (ss+2), the stack must be oversized.
  // The former is needed to allow update_continuation_histories(ss-1, ...),
  // which accesses its argument at ss-6, also near the root.
  // The latter is needed for statScore and killer initialization.

	// continuationHistoryのため、(ss-7)から(ss+2)までにアクセスしたいので余分に確保しておく。
	Stack stack[MAX_PLY + 10], *ss = stack + 7;
	Move  pv[MAX_PLY + 1];

	// alpha,beta : aspiration searchの窓の範囲(alpha,beta)
	// delta      : apritation searchで窓を動かす大きさdelta
	Value alpha, beta, delta;

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

	// 探索部、学習部のスレッドの共通初期化コード
	search_thread_init(this,ss,pv);

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

	// --- MultiPV

	// bestmoveとしてしこの局面の上位N個を探索する機能
	size_t multiPV = Options["MultiPV"];

	// SkillLevelの実装
	//Skill skill(Options["SkillLevel"], Options["USI_LimitStrength"] ? int(Options["USI_Elo"]) : 0);
	// ↑これでエンジンオプション2つも増えるのやだな…。気が向いたらサポートすることにする。
	Skill skill((int)Options["SkillLevel"], 0);

	// When playing with strength handicap enable MultiPV search that we will
	// use behind the scenes to retrieve a set of possible moves.

	// 強さの手加減が有効であるとき、MultiPVを有効にして、その指し手のなかから舞台裏で指し手を探す。
	// ※　SkillLevelが有効(設定された値が20未満)のときは、MultiPV = 4で探索。
	if (skill.enabled())
		multiPV = std::max(multiPV, (size_t)4);

	// この局面での指し手の数を上回ってはいけない
	multiPV = std::min(multiPV, rootMoves.size());

	//complexityAverage.set(202, 1);
	// →導入せず

	//trend = SCORE_ZERO;
	//optimism[ us] = Value(39);
	//optimism[~us] = -optimism[us];

	// Contemptの処理は、やねうら王ではMainThread::search()で行っているのでここではやらない。
	// Stockfishもそうすべきだと思う。
	//int ct = int(Options["Contempt"]) * PawnValueEg / 100; // From centipawns
	// →　Stockfishこのコードなくなった？[2021/09/24]

	// ---------------------
	//   反復深化のループ
	// ---------------------

	// 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
	// これが増えていない時、同じ深さを再度探索する。
	int searchAgainCounter = 0;
	// 反復深化のループで何度fail highしたかのカウンター

	// Iterative deepening loop until requested to stop or the target depth is reached

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

		// Save the last iteration's scores before first PV line is searched and
		// all the move scores except the (new) PV are set to -VALUE_INFINITE.

		// aspiration window searchのために反復深化の前回のiterationのスコアをコピーしておく
		for (RootMove& rm : rootMoves)
			rm.previousScore = rm.score;

		// 将棋ではこれ使わなくていいような？

		//size_t pvFirst = 0;
		//pvLast = 0;

		// 探索深さが増えているかのフラグがfalseならカウンターを1増やす
		if (!Threads.increaseDepth)
			searchAgainCounter++;

		// MultiPV loop. We perform a full root search for each PV line
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
				Value prev = rootMoves[pvIdx].averageScore;

				// aspiration windowの幅
				// 精度の良い評価関数ならばこの幅を小さくすると探索効率が上がるのだが、
				// 精度の悪い評価関数だとこの幅を小さくしすぎると再探索が増えて探索効率が低下する。
				// やねうら王のKPP評価関数では35～40ぐらいがベスト。
				// やねうら王のKPPT(Apery WCSC26)ではStockfishのまま(18付近)がベスト。
				// もっと精度の高い評価関数を用意すべき。
				// この値はStockfish10では20に変更された。
				// Stockfish 12(NNUEを導入した)では17に変更された。
				// Stockfish 12.1では16に変更された。
				delta = Value(PARAM_ASPIRATION_SEARCH_DELTA) + int(prev) * prev / 19178;

				alpha = std::max(prev - delta, -VALUE_INFINITE);
				beta  = std::min(prev + delta,  VALUE_INFINITE);

#if 0
				// Adjust trend and optimism based on root move's previousScore
				// trendと楽観の値をrootの指し手のpreviousScoreを基に調整する。(動的なcontempt)

				// ※　trendは千日手を受け入れるスコア。
				//     勝ってるほうは千日手にはしたくないし、負けてるほうは千日手やむなしという…。

				int tr = sigmoid(prev, 3, 8, 90, 125, 1);

				trend = (us == WHITE ?  make_score(tr, tr / 2)
					                 : -make_score(tr, tr / 2));

				int opt = sigmoid(prev, 8, 17, 144, 13966, 183);
				optimism[us] = Value(opt);
				optimism[~us] = -optimism[us];
#endif
			}

			// Start with a small aspiration window and, in the case of a fail
			// high/low, re-search with a bigger window until we don't fail
			// high/low anymore.

			// 小さなaspiration windowで開始して、fail high/lowのときに、fail high/lowにならないようになるまで
			// 大きなwindowで再探索する。

			// fail highした回数
			// fail highした回数分だけ探索depthを下げてやるほうが強いらしい。
			int failedHighCnt = 0;

			while (true)
			{
				// fail highするごとにdepthを下げていく処理
				Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - searchAgainCounter);
				bestValue = ::search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

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
					//	  mainThread->stopOnPonderhit = false;
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
				delta += delta / 4 + 2;

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

		// -- やねうら王独自の処理ここから↓↓↓

		// x手詰めを発見したのか？

		// multi_pvのときは一つのpvで詰みを見つけただけでは停止するのは良くないので
		// 早期終了はmultiPV == 1のときのみ行なう。

		if (multiPV == 1)
		{
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

		if (!mainThread)
			continue;

		//
		// main threadのときは探索の停止判定が必要
		//

		// ponder用の指し手として、2手目の指し手を保存しておく。
		// これがmain threadのものだけでいいかどうかはよくわからないが。
		// とりあえず、無いよりマシだろう。
		if (mainThread->rootMoves[0].pv.size() > 1)
			mainThread->ponder_candidate = mainThread->rootMoves[0].pv[1];

		// -- やねうら王独自の処理ここまで↑↑↑

		// もしSkillLevelが有効であり、時間いっぱいになったなら、準最適なbest moveを選ぶ。
		if (skill.enabled() && skill.time_to_pick(rootDepth))
			skill.pick_best(multiPV);

		// Use part of the gained time from a previous stable move for the current move
		for (Thread* th : Threads)
		{
			totBestMoveChanges += th->bestMoveChanges;
			th->bestMoveChanges = 0;
		}

		// 残り時間的に、次のiterationに行って良いのか、あるいは、探索をいますぐここでやめるべきか？
		if (Limits.use_time_management())
		{
			// まだ停止が確定していない
			if (!Threads.stop && Time.search_end == 0)
			{
				// 1つしか合法手がない(one reply)であるだとか、利用できる時間を使いきっているだとか、

				double fallingEval = (69 + 12 * (mainThread->bestPreviousAverageScore - bestValue)
										 +  6 * (mainThread->iterValue[iterIdx] - bestValue)) / 781.4;
				fallingEval = std::clamp(fallingEval, 0.5, 1.5);

				// If the bestMove is stable over several iterations, reduce time accordingly
				// もしbestMoveが何度かのiterationにおいて安定しているならば、思考時間もそれに応じて減らす

				timeReduction = lastBestMoveDepth + 10 < completedDepth ? 1.63 : 0.73;
				double reduction = (1.56 + mainThread->previousTimeReduction) / (2.20 * timeReduction);

				// rootでのbestmoveの不安定性。
				// bestmoveが不安定であるなら思考時間を増やしたほうが良い。
				double bestMoveInstability = 1.073 + std::max(1.0, 2.25 - 9.9 / rootDepth)
													* totBestMoveChanges / Threads.size();

#if 0
				int complexity = mainThread->complexityAverage.value();
				double complexPosition = std::clamp(1.0 + (complexity - 326) / 1618.1, 0.5, 1.5);
				double totalTime = Time.optimum() * fallingEval * reduction * bestMoveInstability * complexPosition;
				// →　やねうら王ではcomplexityを導入しないので、以前のコードにしておく。
#endif

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

	// search<>() is the main search function for both PV and non-PV nodes
	// search<>()は、PV , non-PV node共用のメインの探索関数。
	// cutNode = LMRで悪そうな指し手に対してreduction量を増やすnode
	template <NodeType nodeType>
	Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
	{
		// -----------------------
		//     nodeの種類
		// -----------------------

		// PV nodeであるか(root nodeはPV nodeに含まれる)
		constexpr bool PvNode = nodeType != NonPV;

		// root nodeであるか
		constexpr bool rootNode = nodeType == Root;
		
		// 次の最大探索深さ。
		// これを超える深さでsearch()を再帰的に呼び出さない。
		// 延長されすぎるのを回避する。
		const Depth maxNextDepth = rootNode ? depth : depth + 1;

		// Dive into quiescence search when the depth reaches zero
		// 残り探索深さが1手未満であるなら静止探索を呼び出す
		if (depth <= 0)
			return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

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
		HASH_KEY posKey;

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

		// givesCheck			: moveによって王手になるのか
		// improving			: 直前のnodeから評価値が上がってきているのか
		//   このフラグを各種枝刈りのmarginの決定に用いる
		//   cf. Tweak probcut margin with 'improving' flag : https://github.com/official-stockfish/Stockfish/commit/c5f6bd517c68e16c3ead7892e1d83a6b1bb89b69
		//   cf. Use evaluation trend to adjust futility margin : https://github.com/official-stockfish/Stockfish/commit/65c3bb8586eba11277f8297ef0f55c121772d82c
		// didLMR				: LMRを行ったのフラグ
		// priorCapture         : 1つ前の局面は駒を取る指し手か？
		bool givesCheck, improving, didLMR, priorCapture;

		// capture              : moveが駒を捕獲する指し手もしくは歩を成る手であるか
		// doFullDepthSearch	: LMRのときにfail highが起きるなどしたので元の残り探索深さで探索することを示すフラグ
		// moveCountPruning		: moveCountによって枝刈りをするかのフラグ(quietの指し手を生成しない)
		// ttCapture			: 置換表の指し手がcaptureする指し手であるか
		// pvExact				: PvNodeで置換表にhitして、しかもBOUND_EXACT
		bool capture, doFullDepthSearch, moveCountPruning, ttCapture;

		// moveによって移動させる駒
		Piece movedPiece;

		// 調べた指し手を残しておいて、statsのupdateを行なうときに使う。
		// moveCount			: 調べた指し手の数(合法手に限る)
		// captureCount			: 調べた駒を捕獲する指し手の数(capturesSearched[]用のカウンター)
		// quietCount			: 調べた駒を捕獲しない指し手の数(quietsSearched[]用のカウンター)
		// improvement			: improvingフラグのもうちょっと細かい版(int型なので)  improving = improvement > 0
		// complexity           : 局面の複雑性。
		// →この計算に psq_score()が必要なのだが、これ将棋で実装してないのでcomplexityの計算しないことにする。
		int moveCount, captureCount, quietCount, improvement /*, complexity*/;

		// -----------------------
		// Step 1. Initialize node
		// -----------------------

		//     nodeの初期化

		Thread* thisThread = pos.this_thread();
		ss->inCheck		   = pos.checkers();
		priorCapture	   = pos.captured_piece();
		Color us		   = pos.side_to_move();
		moveCount		   = captureCount = quietCount = ss->moveCount = 0;
		bestValue		   = -VALUE_INFINITE;
		//maxValue	       = VALUE_INFINITE;
		// →　将棋ではtable probe使っていないのでmaxValue関係ない。

		//  Timerの監視

		// Check for the available remaining time
		// これはメインスレッドのみが行なう。
		if (thisThread == Threads.main())
			static_cast<MainThread*>(thisThread)->check_time();

		// Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
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
			if (Threads.stop.load(std::memory_order_relaxed) || (ss->ply >= MAX_PLY))
				return draw_value(REPETITION_DRAW, pos.side_to_move());

			if (pos.game_ply() > Limits.max_game_ply)
			{
				// この局面で詰んでいる可能性がある。その時はmatedのスコアを返すべき。
				// 詰んでいないなら引き分けのスコアを返すべき。
				// 関連)
				//    多くの将棋ソフトで256手ルールの実装がバグっている件
				//    https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/

				return pos.is_mated() ? mated_in(ss->ply) : draw_value(REPETITION_DRAW, pos.side_to_move());
			}

			// -----------------------
			// Step 3. Mate distance pruning.
			// -----------------------


			// Even if we mate at the next move our score
			// would be at best mate_in(ss->ply+1), but if alpha is already bigger because
			// a shorter mate was found upward in the tree then there is no need to search
			// because we will never beat the current alpha. Same logic but with reversed
			// signs applies also in the opposite condition of being mated instead of giving
			// mate. In this case return a fail-high score.

			// 詰みまでの手数による枝刈り

			// rootから5手目の局面だとして、このnodeのスコアが5手以内で
			// 詰ますときのスコアを上回ることもないし、
			// 5手以内で詰まさせるときのスコアを下回ることもない。
			// そこで、alpha , betaの値をまずこの範囲に補正したあと、
			// alphaがbeta値を超えているならbeta cutする。

			alpha = std::max(mated_in(ss->ply    ), alpha);
			beta  = std::min(mate_in (ss->ply + 1), beta );
			if (alpha >= beta)
				return alpha;
		}
		else
			// root nodeなら
			thisThread->rootDelta = beta - alpha;

		// -----------------------
		//  探索Stackの初期化
		// -----------------------

		// rootからの手数
		ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

		(ss + 1)->ttPv			= false;
		(ss + 1)->excludedMove	= bestMove = MOVE_NONE;

		// 2手先のkillerの初期化。
		(ss + 2)->killers[0]	= (ss + 2)->killers[1] = MOVE_NONE;

		// Update the running average statistics for double extensions
		ss->doubleExtensions	= (ss - 1)->doubleExtensions;
		ss->depth				= depth;

		// 前の指し手で移動させた先の升目
		// TODO : null moveのときにprevSq == 1 == SQ_12になるのどうなのか…。
		Square prevSq			= to_sq((ss - 1)->currentMove);

		// Initialize statScore to zero for the grandchildren of the current position.
		// So statScore is shared between all grandchildren and only the first grandchild
		// starts with statScore = 0. Later grandchildren start with the last calculated
		// statScore of the previous grandchild. This influences the reduction rules in
		// LMR which are based on the statScore of parent position.

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
		posKey = excludedMove == MOVE_NONE ? pos.hash_key() : pos.hash_key() ^ HASH_KEY(make_key(excludedMove));

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
		// 注意)
		// tte->move()にはMOVE_WINも含まれている可能性がある。
		// この時、pos.to_move(MOVE_WIN) == MOVE_WINなので、ttMove == MOVE_WINとなる。

		ttMove = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
			  : ss->ttHit ? pos.to_move(tte->move()) : MOVE_NONE;
		ASSERT_LV3(pos.legal_promote(ttMove));

		// pos.to_move()でlegalityのcheckに引っかかったパターンなので置換表にhitしなかったことにする。
		if (tte->move().to_u16() && !ttMove)
			ss->ttHit = false;

		// 置換表の指し手がcaptureであるか。
		// 置換表の指し手がcaptureなら高い確率でこの指し手がベストなので、他の指し手を
		// そんなに読まなくても大丈夫。なので、このnodeのすべての指し手のreductionを増やす。

		// ここ、capture_or_promotion()とかcapture_or_pawn_promotion()とか色々変えてみたが、
		// 現在では、capture()にするのが良いようだ。[2022/04/13]

		ttCapture = ttMove && pos.capture(ttMove);

		// 置換表にhitしなかった時は、PV nodeのときだけttPvとして扱う。
		// これss->ttPVに保存してるけど、singularの判定等でsearchをss+1ではなくssで呼び出すことがあり、
		// そのときにss->ttPvが破壊される。なので、破壊しそうなときは直前にローカル変数に保存するコードが書いてある。

		if (!excludedMove)
			ss->ttPv = PvNode || (ss->ttHit && tte->is_pv());

	    // At non-PV nodes we check for an early TT cutoff
		// 置換表の値による枝刈り

		if (  !PvNode                  // PV nodeでは置換表の指し手では枝刈りしない(PV nodeはごくわずかしかないので..)
			&& ss->ttHit               // 置換表の指し手がhitして
			&& tte->depth() > depth - (thisThread->id() % 2 == 1)   // 置換表に登録されている探索深さのほうが深くて
									   // ↑ スレッドごとに探索がばらけるように。
			&& ttValue != VALUE_NONE   // (VALUE_NONEだとすると他スレッドからTTEntryが読みだす直前に破壊された可能性がある)
			&& (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
				                : (tte->bound() & BOUND_UPPER))
			// ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
			// ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
			// 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
			// このままこの値でreturnして良い。
			)
		{
			// If ttMove is quiet, update move sorting heuristics on TT hit (~1 Elo)

			// 置換表の指し手でbeta cutが起きたのであれば、この指し手をkiller等に登録する。
			// ただし、捕獲する指し手か成る指し手であればこれは(captureで生成する指し手なので)killerを更新する価値はない。

			// ただし置換表の指し手には、hash衝突によりpseudo-leaglでない指し手である可能性がある。
			// update_quiet_stats()で、この指し手の移動元の駒を取得してCounter Moveとするが、
			// それがこの局面の手番側の駒ではないことがあるのでゆえにここでpseudo_legalのチェックをして、
			// Counter Moveに先手の指し手として後手の指し手が登録されるような事態を回避している。
			// その時に行われる誤ったβcut(枝刈り)は許容できる。(non PVで生じることなのでそこまで探索に対して悪い影響がない)
			// cf. https://yaneuraou.yaneu.com/2021/08/17/about-the-yaneuraou-bug-that-appeared-in-the-long-match/

			// ttMoveがMOVE_WINであることはありうるので注意が必要。
			// is_ok(m)==falseの時、Position::to_move(m)がmをそのまま帰すことは保証されている。
			// そのためttMoveがMOVE_WINでありうる。これはstatのupdateをされると困るのでis_ok()で弾く必要がある。
			// is_ok()は、ttMove == MOVE_NONEの時はfalseなのでこの条件を省略できる。

			if (/* ttMove && */ is_ok(ttMove))
			{
				if (ttValue >= beta)
				{
					// Bonus for a quiet ttMove that fails high (~3 Elo)
					// fail highしたquietなquietな(駒を取らない)ttMove(置換表の指し手)に対するボーナス

					if (!ttCapture)
						update_quiet_stats(pos, ss, ttMove, stat_bonus(depth));

					// Extra penalty for early quiet moves of the previous ply (~0 Elo)
					// 1手前の早い時点のquietの指し手に対する追加のペナルティ
					// 1手前は置換表の指し手であるのでNULL MOVEではありえない。

					if ((ss - 1)->moveCount <= 2 && !priorCapture)
						update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -stat_bonus(depth + 1));

				}
				// Penalty for a quiet ttMove that fails low (~1 Elo)
				// fails lowのときのquiet ttMoveに対するペナルティ
				else if (!ttCapture)
				{
					int penalty = -stat_bonus(depth);
					thisThread->mainHistory[from_to(ttMove)][us] << penalty;
					update_continuation_histories(ss, pos.moved_piece_after(ttMove), to_sq(ttMove), penalty);
				}
			}

			// Partial workaround for the graph history interaction problem
			// For high rule50 counts don't produce transposition table cutoffs.
			// →　将棋では関係のないルールなので無視して良い。
			// 
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

					ASSERT_LV3(pos.legal_promote(m));
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
						ASSERT_LV3(pos.legal_promote(move));
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

						ASSERT_LV3(pos.legal_promote(move));
						tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, BOUND_EXACT,
							MAX_PLY, move, /* ss->staticEval */ bestValue);

						return bestValue;
					}
				}

			}
			// 1手詰めがなかったのでこの時点でもsave()したほうがいいような気がしなくもない。
		}

		// -----------------------
		// Step 6. Static evaluation of the position
		// -----------------------

		//  局面の静的な評価

		CapturePieceToHistory& captureHistory = thisThread->captureHistory;

		if (ss->inCheck)
		{
			// Skip early pruning when in check
			// 王手がかかっているときは、early pruning(早期枝刈り)をスキップする
			
			ss->staticEval = eval = VALUE_NONE;
			improving = false;
			improvement = 0;
			//complexity = 0;
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

			// 引き分けっぽい評価値であるなら、いくぶん揺らす。(0を±1として扱う)
			// (千日手回避のため)
			// 将棋ではこの処理どうなのかな…。
			// 
			// ※　そもそも千日手時のスコアがデフォルトではVALUE_DRAW==0ではないので
			// 　　このコードだとうまく動作しない。

	        // // Randomize draw evaluation
			// if (eval == VALUE_DRAW)
			//	   eval = value_draw(thisThread);

			// ttValue can be used as a better position evaluation (~4 Elo)
			// ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。
			// 1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
			//   evalとしてttValueを採用して良い。
			// 2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
			//   evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。

			if (    ttValue != VALUE_NONE
				&& (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
				eval = ttValue;

		}
		else
		{
			ss->staticEval = eval = evaluate(pos);

	        // Save static evaluation into transposition table
			// 評価関数を呼び出したので置換表のエントリーはなかったことだし、何はともあれそれを保存しておく。
			// ※　bonus分だけ加算されているが静止探索の値ということで…。
			//
			// また、excludedMoveがある時は、これを置換表に保存するのは危ない。
			// cf . Add / remove leaves from search tree ttPv : https://github.com/official-stockfish/Stockfish/commit/c02b3a4c7a339d212d5c6f75b3b89c926d33a800

			if (!excludedMove)
				tte->save(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, MOVE_NONE, eval);

			// どうせ毎node評価関数を呼び出すので、evalの値にそんなに価値はないのだが、mate1ply()を
			// 実行したという証にはなるので意味がある。
		}

		// -----------------------
		//   evalベースの枝刈り
		// -----------------------

		// Use static evaluation difference to improve quiet move ordering (~3 Elo)

		// 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
		// 王手のときはここにはこない。(上のinCheckのなかでMOVES_LOOPに突入。)

		if (is_ok((ss-1)->currentMove) && !(ss-1)->inCheck && !priorCapture)
		{
			int bonus = std::clamp(-16 * int((ss - 1)->staticEval + ss->staticEval), -2000, 2000);
			thisThread->mainHistory[from_to((ss-1)->currentMove)][~us] << bonus;
		}

		// Set up the improvement variable, which is the difference between the current
		// static evaluation and the previous static evaluation at our turn (if we were
		// in check at our previous move we look at the move prior to it). The improvement
		// margin and the improving flag are used in various pruning heuristics.
		//
		// 評価値が2手前の局面から上がって行っているのかのフラグ(improving)
		// 上がって行っているなら枝刈りを甘くする。
		// ※ VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
		//   枝刈りを甘くして調べないといけないのでimproving扱いとする。

		improvement =	  (ss-2)->staticEval != VALUE_NONE ? ss->staticEval - (ss-2)->staticEval
						: (ss-4)->staticEval != VALUE_NONE ? ss->staticEval - (ss-4)->staticEval
						:                                    175;
		// ※　VALUE_NONE == 32002なのでこれより大きなstaticEvalの値であることはない。

		// やねうら王ではpsq_score()を実装していないのでcomplexityは導入せず[2022/04/13]

		//complexity = abs(ss->staticEval - (us == WHITE ? eg_value(pos.psq_score()) : -eg_value(pos.psq_score())));

		//thisThread->complexityAverage.update(complexity);

		// improvingフラグは、improvementをbool化したもの。
		improving = improvement > 0;

		// -----------------------
		// Step 7. Razoring.
		// -----------------------

		// If eval is really low check with qsearch if it can exceed alpha, if it can't,
		// return a fail low.

		// eval が alpha よりもずっと下にある場合、qsearch が alpha よりも上に押し上げることが
		// できるかどうかをチェックし、もしできなければ fail low を返す。

		if (  !PvNode
			&& depth <= 7
			&& eval < alpha - 348 - 258 * depth * depth)
		{
			value = qsearch<NonPV>(pos, ss, alpha - 1, alpha);
			if (value < alpha)
				return value;
		}

		// -----------------------
		// Step 8. Futility pruning: child node (~25 Elo).
		//         The depth condition is important for mate finding.
		// -----------------------

		//   Futility pruning : 子ノード (王手がかかっているときはスキップする)

		// このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
		// evalからこれを引いてbetaより大きいなら、beta cutが出来る。
		// ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
		// ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。

		// Stockfish9までは、futility pruningを、root node以外に適用していたが、
		// Stockfish10でnonPVにのみの適用に変更になった。

		if (   !ss->ttPv
			&&  depth < PARAM_FUTILITY_RETURN_DEPTH/*8*/
			&&  eval - futility_margin(depth, improving) - (ss - 1)->statScore / 256 >= beta
			&&  eval >= beta
			&&  eval < VALUE_KNOWN_WIN + MAX_PLY * 2 /*26305*/) // larger than VALUE_KNOWN_WIN, but smaller than TB wins.

			// 詰み絡み等だとmate distance pruningで枝刈りされるはずで、ここでは枝刈りしない。
			// Stockfishでは、上の最後の条件は、
			//   &&  eval < 26305) // larger than VALUE_KNOWN_WIN, but smaller than TB wins.
			// となっているが、StockfishではVALUE_KNOWN_WIN == 10000で、これより十分大きく、
			// TB winより小さな値として26305となっている。
			// やねうら王では、VALUE_KNOWN_WINは mateと1000しか離れていないため、
			// VALUE_KNOWN_WIN + MAX_PLY * 2で代用する。(そこまではfutility pruningして良いという考え)
			// そこを超えるとmateのスコアになってくるので futilityで刈るのは危ない。
			
			return eval;
		// 次のようにするより、単にevalを返したほうが良いらしい。
		//	 return eval - futility_margin(depth);
		// cf. Simplify futility pruning return value : https://github.com/official-stockfish/Stockfish/commit/f799610d4bb48bc280ea7f58cd5f78ab21028bf5


		// -----------------------
		// Step 9. Null move search with verification search (~22 Elo)
		// -----------------------

		//  検証用の探索つきのnull move探索。PV nodeではやらない。

		//  evalの見積りがbetaを超えているので1手パスしてもbetaは超えそう。
		if (!PvNode
			&& (ss - 1)->currentMove != MOVE_NULL
			&& (ss - 1)->statScore < PARAM_NULL_MOVE_MARGIN0/*14695*/
			&&  eval >= beta
			&&  eval >= ss->staticEval
			&&  ss->staticEval >= beta - PARAM_NULL_MOVE_MARGIN1 /*15*/ * depth
								- improvement / PARAM_NULL_MOVE_MARGIN3 /* 15 */
								+ PARAM_NULL_MOVE_MARGIN4 /*198*/
								/* + complexity / 28 */
			&& !excludedMove
		//	&&  pos.non_pawn_material(us)  // これ終盤かどうかを意味する。将棋でもこれに相当する条件が必要かも。
			&& (ss->ply >= thisThread->nmpMinPly || us != thisThread->nmpColor)
			// 同じ手番側に連続してnull moveを適用しない
			)
		{
			ASSERT_LV3(eval - beta >= 0);

			// Null move dynamic reduction based on depth, eval and complexity of position
			// 残り探索深さと評価値によるnull moveの深さを動的に減らす
			Depth R = std::min(int(eval - beta) / PARAM_NULL_MOVE_DYNAMIC_GAMMA/*147*/, 5)
					+ (PARAM_NULL_MOVE_DYNAMIC_ALPHA/*1024*/ +  PARAM_NULL_MOVE_DYNAMIC_BETA/*85*/ * depth) / 256
					//- (complexity > 753)
					// →　やねうら王では complexityを導入せず
				;

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

				// Do not return unproven mate or TB scores
				// 証明されていないmate scoreはreturnで返さない。
				if (nullValue >= VALUE_TB_WIN_IN_MAX_PLY)
					nullValue = beta;

				if (thisThread->nmpMinPly || (abs(beta) < VALUE_KNOWN_WIN && depth < PARAM_NULL_MOVE_RETURN_DEPTH/*13*/ ))
					return nullValue;

				ASSERT_LV3(!thisThread->nmpMinPly); // 再帰的な検証は認めていない。

				// Do verification search at high depths, with null move pruning disabled
				// for us, until ply exceeds nmpMinPly.
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
		// Step 10. ProbCut (~4 Elo)
		// -----------------------

		// probCutに使うbeta値。
		probCutBeta = beta + PARAM_PROBCUT_MARGIN1/*179*/ - PARAM_PROBCUT_MARGIN2/*46*/ * improving;

		// ProbCut(王手のときはスキップする)

		// If we have a good enough capture and a reduced search returns a value
		// much above beta, we can (almost) safely prune the previous move.

		// もし、このnodeで非常に良いcaptureの指し手があり(例えば、SEEの値が動かす駒の価値を上回るようなもの)
		// 探索深さを減らしてざっくり見てもbetaを非常に上回る値を返すようなら、このnodeをほぼ安全に枝刈りすることが出来る。

		if (   !PvNode
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

			&& !(  ss->ttHit
				&& tte->depth() >= depth - (PARAM_PROBCUT_DEPTH-1)
				&& ttValue != VALUE_NONE
				&& ttValue < probCutBeta))
		{
			ASSERT_LV3(probCutBeta < VALUE_INFINITE);

			MovePicker mp(pos, ttMove, probCutBeta - ss->staticEval, depth - 3, &captureHistory);
			bool ttPv = ss->ttPv;  // このあとの探索でss->ttPvを潰してしまうのでtte->save()のときはこっちを用いる。
			//bool captureOrPromotion;
			// ↑このStockfishのコード、ここで宣言しなくてもいいと思う。[2022/04/13]
			ss->ttPv = false;

			// 試行回数は2回(cutNodeなら4回)までとする。(よさげな指し手を3つ試して駄目なら駄目という扱い)
			// cf. Do move-count pruning in probcut : https://github.com/official-stockfish/Stockfish/commit/b87308692a434d6725da72bbbb38a38d3cac1d5f
			while ((move = mp.next_move()) != MOVE_NONE)
			{
				// ↑Stockfishでは省略してあるけど、この"{"、省略するとbugの原因になりうる。

				ASSERT_LV3(pos.pseudo_legal(move) && pos.legal_promote(move));

				if (move != excludedMove && pos.legal(move))
				{
					// 現状、prob cutでは 歩の成りも含まれる。
					ASSERT_LV3(pos.capture_or_pawn_promotion(move) /* || promotion_type(move) == QUEEN */);

					const bool captureOrPromotion = true;

					ss->currentMove = move;
					ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck                ]
																			  [captureOrPromotion         ]
																			  [to_sq(move)                ]
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

						if (! (ss->ttHit
							&& tte->depth() >= depth - (PARAM_PROBCUT_DEPTH - 1)
							&& ttValue != VALUE_NONE))
						{
							ASSERT_LV3(pos.legal_promote(move));
							tte->save(posKey, value_to_tt(value, ss->ply), ttPv,
								BOUND_LOWER,
								depth - (PARAM_PROBCUT_DEPTH - 1), move, ss->staticEval);
						}
						return value;
					}
				}

			} // end of while


			// ss->ttPvはprobCutの探索で書き換えてしまったかも知れないので復元する。
			ss->ttPv = ttPv;
		}

		// -----------------------
		// Step 11. If the position is not in TT, decrease depth by 2 or 1 depending on node type (~3 Elo)
		// -----------------------

		// 局面がTTになかったのなら、探索深さを2下げる。
		// ※　このあとも置換表にヒットしないであろうから、ここを浅めで探索しておく。
		// (次に他のスレッドがこの局面に来たときには置換表にヒットするのでそのときにここの局面の
		//   探索が完了しているほうが助かるため)
		
		if (   PvNode
			&& depth >= 3
			&& !ttMove)
			depth -= 2;

		if (   cutNode
			&& depth >= 8
			&& !ttMove)
			depth--;

		// When in check, search starts here
		// 王手がかかっている局面では、探索はここから始まる。
	moves_loop:

		// このノードでまだ評価関数を呼び出していないなら、呼び出して差分計算しないといけない。
		// (やねうら王独自仕様)
		// do_move()で行っている評価関数はこの限りではないが、NNUEでも
		// このタイミングで呼び出したほうが高速化するようなので呼び出す。
		Eval::evaluate_with_no_return(pos);

		// -----------------------
		// Step 12. A small Probcut idea, when we are in check (~0 Elo)
		// -----------------------

		probCutBeta = beta + 481;
		if (   ss->inCheck
			&& !PvNode
			&& depth >= 2
			&& ttCapture
			&& (tte->bound() & BOUND_LOWER)
			&& tte->depth() >= depth - 3
			&& ttValue >= probCutBeta
			&& abs(ttValue) <= VALUE_KNOWN_WIN
			&& abs(beta) <= VALUE_KNOWN_WIN
			)
			return probCutBeta;


		// -----------------------
		// moves loopに入る前の準備
		// -----------------------

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
										  &captureHistory,
										  contHist,
										  countermove,
										  ss->killers);


		value = bestValue;

		moveCountPruning = false;

		// Indicate PvNodes that will probably fail low if the node was searched
		// at a depth equal or greater than the current depth, and the result of this search was a fail low.
		// ノードが現在のdepth以上で探索され、fail lowである時に、PvNodeがfail lowしそうであるかを示すフラグ。
		bool likelyFailLow =   PvNode
							&& ttMove
							&& (tte->bound() & BOUND_UPPER)
							&& tte->depth() >= depth;


		// -----------------------
		// Step 13. Loop through all pseudo-legal moves until no moves remain
		//			or a beta cutoff occurs.
		// -----------------------

		//  一手ずつ調べていく

		//  指し手がなくなるか、beta cutoffが発生するまで、すべての指し手を調べる。
		//  MovePickerが返す指し手はpseudo-legalであることは保証されている。
		// ※　do_move()までにはlegalかどうかの判定が必要。

		// moveCountPruningがtrueの時はnext_move()はQUIETの指し手を返さないので注意。
		while ((move = mp.next_move(moveCountPruning)) != MOVE_NONE)
		{
			ASSERT_LV3(pos.pseudo_legal(move) && pos.legal_promote(move));

			if (move == excludedMove)
				continue;

			// At root obey the "searchmoves" option and skip moves not listed in Root
			// Move List. As a consequence any illegal move is also skipped. In MultiPV
			// mode we also skip PV moves which have been already searched and those
			// of lower "TB rank" if we are in a TB root position.


			// root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。
			// Stockfishでは、2行目、
			//  thisThread->rootMoves.end() + thisThread->pvLast となっているが
			// 将棋ではこの処理不要なのでやねうら王ではpvLastは使わない。
			if (rootNode && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
										thisThread->rootMoves.end()   , move))
				continue;

			// Check for legality
			// moveの合法性をチェック。

			// root nodeなら、rootMovesになければlegalではない。
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

			capture    = pos.capture(move);

			// 今回移動させる駒(移動後の駒)
			movedPiece = pos.moved_piece_after(move);

			// 今回の指し手で王手になるかどうか
			givesCheck = pos.gives_check(move);

			// -----------------------
			// Step 14. Pruning at shallow depth (~98 Elo). Depth conditions are important for mate finding.
			// -----------------------

			// 浅い深さでの枝刈り

			// Calculate new depth for this move
			// 今回の指し手に関して新しいdepth(残り探索深さ)を計算する。
			newDepth = depth - 1;

			Value delta = beta - alpha;

			if (  !rootNode
				// 【計測資料 7.】 浅い深さでの枝刈りを行なうときに王手がかかっていないことを条件に入れる/入れない
			//	&& pos.non_pawn_material(us)  // これに相当する処理、将棋でも必要だと思う。
				&& bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
			{
				// Skip quiet moves if movecount exceeds our FutilityMoveCount threshold (~7 Elo)
				// move countベースの枝刈りを実行するかどうかのフラグ
				moveCountPruning = moveCount >= futility_move_count(improving, depth);

				// Reduced depth of the next LMR search
				// 次のLMR探索における軽減された深さ
				int lmrDepth = std::max(newDepth - reduction(improving, depth, moveCount, delta, thisThread->rootDelta), 0);

				if (   capture
					|| givesCheck)
				{

#if 0
					// Stockfishでは削除された枝刈り。これはあったほうが良いかも。
					// →　計測したら誤差だったのでコメントアウト
					// 
					// Capture history based pruning when the move doesn't give check
					if (  !givesCheck
						&& lmrDepth < 1
						&& captureHistory[to_sq(move)][movedPiece][type_of(pos.piece_on(to_sq(move)))] < 0)
						continue;
#endif

					// Futility pruning for captures (~0 Elo)
					if (   !pos.empty(to_sq(move))
						&& !givesCheck
						&& !PvNode
						&& lmrDepth < 6
						&& !ss->inCheck
						&& ss->staticEval + 281 + 179 * lmrDepth + CapturePieceValueEg(pos,move)
						 + captureHistory[to_sq(move)][movedPiece][type_of(pos.piece_on(to_sq(move)))] / 6 < alpha)
						// やねうら王、captureHistory[to][pc]で、Stockfishと逆順なので注意。
						continue;

					// SEE based pruning (~9 Elo)
					if (!pos.see_ge(move, - Value(PARAM_LMR_SEE_MARGIN1 /*203*/) * depth)) // (~25 Elo)
						continue;
				}
				else
				{
					// // Continuation history based pruning (~20 Elo)
					// Continuation historyに基づいた枝刈り(historyの値が悪いものに関してはskip) : ~20 Elo

					int history = (*contHist[0])[to_sq(move)][movedPiece]
								+ (*contHist[1])[to_sq(move)][movedPiece]
								+ (*contHist[3])[to_sq(move)][movedPiece];

					if (lmrDepth < PARAM_PRUNING_BY_HISTORY_DEPTH/*5*/
						&& history < -3875 * (depth - 1))
						// contHist[][]はStockfishと逆順なので注意。
						continue;

					// mainHistory[][]もStockfishと逆順なので注意。
					history += thisThread->mainHistory[from_to(move)][ us];

					// Futility pruning: parent node (~9 Elo)
					// 親nodeの時点で子nodeを展開する前にfutilityの対象となりそうなら枝刈りしてしまう。

					// パラメーター調整の係数を調整したほうが良いのかも知れないが、
					// ここ、そんなに大きなEloを持っていないので、調整しても無意味。

					if (   !ss->inCheck
						&& lmrDepth < PARAM_FUTILITY_AT_PARENT_NODE_DEPTH/*11*/
						&& ss->staticEval + PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1/*122*/ + PARAM_FUTILITY_MARGIN_BETA/*138*/ * lmrDepth
							+ history / 60 <= alpha )
						continue;

					// ※　このLMRまわり、棋力に極めて重大な影響があるので枝刈りを入れるかどうかを含めて慎重に調整すべき。

					// Prune moves with negative SEE (~3 Elo)
					// 将棋ではseeが負の指し手もそのあと詰むような場合があるから、あまり無碍にも出来ないようだが…。

					// 【計測資料 20.】SEEが負の指し手を枝刈りする/しない

					if (!pos.see_ge(move, Value( - PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1/*25*/ * lmrDepth * lmrDepth
						- PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2/*20*/ * lmrDepth )))
						continue;
				}
			}

			// -----------------------
			// Step 15. Extensions (~66 Elo)
			// -----------------------

			// We take care to not overdo to avoid search getting stuck.
			// 探索がstuckしないように、やりすぎに気をつける。
			// (rootDepthの2倍より延長しない)

			if (ss->ply < thisThread->rootDepth * 2)
			{

				// singular延長と王手延長。

				// Singular extension search (~58 Elo). If all moves but one fail low on a
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
				if (!rootNode
					&& depth >= PARAM_SINGULAR_EXTENSION_DEPTH /* 4 */ + 2 * (PvNode && tte->is_pv())
					&& move == ttMove
					&& !excludedMove // 再帰的なsingular延長を除外する。
				/*  &&  ttValue != VALUE_NONE Already implicit in the next condition */
					&& abs(ttValue) < VALUE_KNOWN_WIN // 詰み絡みのスコアはsingular extensionはしない。(Stockfish 10～)
					&& (tte->bound() & BOUND_LOWER)
					&& tte->depth() >= depth - 3)
					// このnodeについてある程度調べたことが置換表によって証明されている。(ttMove == moveなのでttMove != MOVE_NONE)
					// (そうでないとsingularの指し手以外に他の有望な指し手がないかどうかを調べるために
					// null window searchするときに大きなコストを伴いかねないから。)
				{
					// このmargin値は評価関数の性質に合わせて調整されるべき。
					Value singularBeta = ttValue - PARAM_SINGULAR_MARGIN/* == 3 * 256 */ * depth / 256;
					Depth singularDepth = (depth - 1) / 2;

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

#if 0
						// singular extentionが生じた回数の統計を取ってみる。
						dbg_hit_on(extension == 1);
#endif

						// Avoid search explosion by limiting the number of double extensions
						// 2重延長を制限することで探索の組合せ爆発を回避する。
						if (!PvNode
							&& value < singularBeta - 26
							&& ss->doubleExtensions <= 8)
							extension = 2;
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

					// If the eval of ttMove is greater than beta, we reduce it (negative extension)
					// ttMoveのevalがbetaより大きいなら、extensionを減らす(負の延長)

					else if (ttValue >= beta)
						extension = -2;

					// If the eval of ttMove is less than alpha and value, we reduce it (negative extension)
					else if (ttValue <= alpha && ttValue <= value)
						extension = -1;
				}

				// Check extensions (~1 Elo)
				// 王手延長

				//  注意 : 王手延長に関して、Stockfishのコード、ここに持ってくる時には気をつけること！
				// →　将棋では王手はわりと続くのでそのまま持ってくるとやりすぎの可能性が高い。
				// 
				// ※ Stockfish 14では depth > 6 だったのが、Stockfish 15でdepth > 9に変更されたが				
				//  それでもまだやりすぎの感はある。やねうら王では、延長の条件をさらに絞る。

				else if (givesCheck
					&& depth > 9
					&& abs(ss->staticEval) > 71
					// この条件、やねうら王で独自追加。
					// →　王手延長は、開き王手と駒損しない王手に限定する。
					&& (pos.is_discovery_check_on_king(~us, move) || pos.see_ge(move))
					)
					extension = 1;

				// Quiet ttMove extensions (~0 Elo)
				// PV nodeで quietなttは良い指し手のはずだから延長するというもの。

				else if (PvNode
					&& move == ttMove
					&& move == ss->killers[0]
					&& (*contHist[0])[to_sq(move)][movedPiece] >= 5491)
					extension = 1;
			}

			// -----------------------
			//   1手進める前の枝刈り
			// -----------------------

			// Add extension to new depth
			// 求まった延長する手数を新しいdepthに加算

			// 再帰的にsearchを呼び出すとき、search関数に渡す残り探索深さ。
			// これはsingluar extensionの探索が終わってから決めなければならない。(singularなら延長したいので)
			newDepth += extension;

			// doubleExtensionsは、前のノードで延長したかと本ノードで延長したかを加算した値
			ss->doubleExtensions = (ss - 1)->doubleExtensions + (extension == 2);

			// -----------------------
			//      1手進める
			// -----------------------

			// この時点で置換表をprefetchする。将棋においては、指し手に駒打ちなどがあって指し手を適用したkeyを
			// 計算するコストがわりとあるので、これをやってもあまり得にはならない。無効にしておく。

			// Speculative prefetch as early as possible
			// 投機的なprefetch
			//const Key nextKey = pos.key_after(move);
			//prefetch(TT.first_entry(nextKey));
			//Eval::prefetch_evalhash(nextKey);

			// Update the current move (this must be done after singular extension search)
			// 現在このスレッドで探索している指し手を更新しておく。(これは、singular extension探索のあとになされるべき)
			ss->currentMove = move;
			ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
																	  [capture    ]
																	  [to_sq(move)]
																	  [movedPiece ];

			// -----------------------
			// Step 16. Make the move
			// -----------------------

			// 指し手で1手進める
			pos.do_move(move, st, givesCheck);

			bool doDeeperSearch = false;

			// -----------------------
			// Step 17. Late moves reduction / extension (LMR, ~98 Elo)
			// -----------------------
			// depthを減らした探索。LMR(Late Move Reduction)

			// If the move fails high it will be re - searched at full depth.
			// depthを減らして探索させて、その指し手がfail highしたら元のdepthで再度探索するという手法

			// We use various heuristics for the sons of a node after the first son has
			// been searched. In general we would like to reduce them, but there are many
			// cases where we extend a son if it has good chances to be "interesting".

			// moveCountが大きいものなどは探索深さを減らしてざっくり調べる。
			// alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。

			if (   depth >= 2
				&& moveCount > 1 + (PvNode && ss->ply <= 1)
				&& (   !ss->ttPv
					|| !capture
					|| (cutNode && (ss-1)->moveCount > 1)))
			{
				// Reduction量
				Depth r = reduction(improving, depth, moveCount, delta, thisThread->rootDelta);

				// Decrease reduction if position is or has been on the PV
				// and node is not likely to fail low. (~3 Elo)
				// この局面がPV上にあり、fail lowしそうであるならreductionを減らす
				// (fail lowしてしまうとまた探索をやりなおさないといけないので)
				if (   ss->ttPv
					&& !likelyFailLow)
					r -= 2 ;

				// Decrease reduction if opponent's move count is high (~5 Elo)
				// 相手の指し手(1手前の指し手)のmove countが高い場合、reduction量を減らす。
				// 相手の指し手をたくさん読んでいるのにこちらだけreductionするとバランスが悪いから。

				// 【計測資料 4.】相手のmoveCountが高いときにreductionを減らす
				// →　古い計測なので当時はこのコードないほうが良かったが、Stockfish10では入れたほうが良さげ。

				// Decrease reduction if opponent's move count is high (~1 Elo)
				// 相手の(1手前の)move countが大きければ、reductionを減らす。
				// ※ この > 7 の 7は調整が必要かも。
				if ((ss - 1)->moveCount > 7)
					r--;

				// Increase reduction for cut nodes (~3 Elo)
				// cut nodeにおいてhistoryの値が悪い指し手に対してはreduction量を増やす。
				// ※　PVnodeではIID時でもcutNode == trueでは呼ばないことにしたので、
				// if (cutNode)という条件式は暗黙に && !PvNode を含む。

				// 【計測資料 18.】cut nodeのときにreductionを増やすかどうか。

				if (cutNode && move != ss->killers[0])
					r += 2;

				// Increase reduction if ttMove is a capture (~3 Elo)
				// 【計測資料 3.】置換表の指し手がcaptureのときにreduction量を増やす。

				if (ttCapture)
					r++;

				// Decrease reduction at PvNodes if bestvalue
				// is vastly different from static evaluation

				// PvNodesでbestValueが静的評価(評価関数の返し値)と大きく異るなら
				// reductionを減らす。
				if (PvNode && !ss->inCheck && abs(ss->staticEval - bestValue) > 250)
					r--;

				// Decrease reduction for PvNodes based on depth
				if (PvNode)
					r -= 1 + 15 / ( 3 + depth );


				// 【計測資料 11.】statScoreの計算でcontHist[3]も調べるかどうか。
				// contHist[5]も/2とかで入れたほうが良いのでは…。誤差か…？
				ss->statScore = thisThread->mainHistory[from_to(move)][us]
								+ (*contHist[0])[to_sq(move)][movedPiece]
								+ (*contHist[1])[to_sq(move)][movedPiece]
								+ (*contHist[3])[to_sq(move)][movedPiece]
								- PARAM_REDUCTION_BY_HISTORY/*4334*/; // 修正項


				// Decrease/increase reduction for moves with a good/bad history (~30 Elo)
				r -= ss->statScore / 15914;

				// In general we want to cap the LMR depth search at newDepth. But if reductions
				// are really negative and movecount is low, we allow this move to be searched
				// deeper than the first move (this may lead to hidden double extensions).

				int deeper =  r >= -1					? 0
							: moveCount <= 4			? 2
							: PvNode && depth > 4       ? 1
							: cutNode && moveCount <= 8 ? 1
							:							  0;

				// depth >= 3なのでqsearchは呼ばれないし、かつ、
				// moveCount > 1 すなわち、このnodeの2手目以降なのでsearch<NonPv>が呼び出されるべき。

				Depth d = std::clamp(newDepth - r, 1, newDepth + deeper);

				value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);

				//
				// ここにその他の枝刈り、何か入れるべき(かも)
				//

				// If the son is reduced and fails high it will be re-searched at full depth
				// 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす。

				doFullDepthSearch = value > alpha && d < newDepth;
				doDeeperSearch = value > (alpha + 78 + 11 * (newDepth - d));
				didLMR = true;
			}
			else
			{
				// non PVか、PVでも2手目以降であればfull depth searchを行なう。
				doFullDepthSearch = !PvNode || moveCount > 1;
				didLMR = false;
			}


			// -----------------------
			// Step 18. Full depth search when LMR is skipped or fails high
			// -----------------------

			// Full depth search。LMRがskipされたか、LMRにおいてfail highを起こしたなら元の探索深さで探索する。

			// ※　静止探索は残り探索深さはdepth = 0として開始されるべきである。(端数があるとややこしいため)
			if (doFullDepthSearch)
			{
				value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth + doDeeperSearch, !cutNode);

				// If the move passed LMR update its stats
				if (didLMR)
				{
					int bonus = value > alpha ?  stat_bonus(newDepth)
											  : -stat_bonus(newDepth);

					if (capture)
						bonus /= 6;

					update_continuation_histories(ss, movedPiece, to_sq(move), bonus);
				}
			}

			// For PV nodes only, do a full PV search on the first move or after a fail
			// high (in the latter case search only if value < beta), otherwise let the
			// parent node fail low with value <= alpha and try another move.

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
			// Step 19. Undo move
			// -----------------------

			//      1手戻す

			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// -----------------------
			// Step 20. Check for a new best move
			// -----------------------


			// Finished searching the move. If a stop occurred, the return value of
			// the search cannot be trusted, and we return immediately without
			// updating best move, PV and TT.

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

				// rootの平均スコアを求める。aspiration searchで用いる。
				rm.averageScore = rm.averageScore != -VALUE_INFINITE ? (2 * value + rm.averageScore) / 3 : value;

				// PV move or new best move?
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

					// We record how often the best move has been changed in each iteration.
					// This information is used for time management. In MultiPV mode,
					// we must take care to only do this for the first PV line.
					//
					// bestMoveが何度変更されたかを記録しておく。
					// これが頻繁に行われるのであれば、思考時間を少し多く割り当てる。
					//
					// !thisThread->pvIdx という条件を入れておかないとMultiPVで
					// time managementがおかしくなる。

					if (    moveCount > 1
						&& !thisThread->pvIdx)
						++thisThread->bestMoveChanges;

				} else {

					// All other moves but the PV are set to the lowest value: this
					// is not a problem when sorting because the sort is stable and the
					// move position in the list is preserved - just the PV is pushed up.

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

					// Update pv even in fail-high case
					// fail-highのときにもPVをupdateする。
					if (PvNode && !rootNode)
						update_pv(ss->pv, move, (ss + 1)->pv);

					// Update alpha! Always alpha < beta
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

						ASSERT_LV3(value >= beta); // Fail high
						break;
					}
				}
			}

			// If the move is worse than some previously searched move, remember it to update its stats later
			// もしその指し手が、以前に探索されたいくつかの指し手より悪い場合は、あとで統計を取る時のために記憶しておく。

			if (move != bestMove)
			{
				// 探索した駒を捕獲する指し手を32手目まで
				if (capture && captureCount < 32)
					capturesSearched[captureCount++] = move;

				// 探索した駒を捕獲しない指し手を64手目までquietsSearchedに登録しておく。
				// あとでhistoryなどのテーブルに加点/減点するときに使う。

				else if (!capture && quietCount < 64)
					quietsSearched[quietCount++] = move;
			}
		}
		// end of while

		// -----------------------
		// Step 21. Check for mate and stalemate
		// -----------------------

		// All legal moves have been searched and if there are no legal moves, it
		// must be a mate or a stalemate. If we are in a singular extension search then
		// return a fail low score.

		// 詰みとステイルメイトをチェックする。

		// このStockfishのassert、合法手を生成しているので重すぎる。良くない。
		ASSERT_LV5(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());


		// Stockfishでは、ここのコードは以下のようになっているが、これは、
		// 自玉に王手がかかっておらず指し手がない場合(stalemate)、引き分けだから。
		/*
		if (!moveCount)
			bestValue = excludedMove ? alpha :
						 ss->inCheck ? mated_in(ss->ply)
									 : VALUE_DRAW;
		*/

		// ※　ここ、Stockfishのコードをそのままコピペしてこないように注意！

		// (将棋では)合法手がない == 詰まされている なので、rootの局面からの手数で詰まされたという評価値を返す。
		// ただし、singular extension中のときは、ttMoveの指し手が除外されているので単にalphaを返すべき。
		if (!moveCount)
			bestValue = excludedMove ? alpha : mated_in(ss->ply);


	    // If there is a move which produces search value greater than alpha we update stats of searched moves
		// bestMoveがあるならこの指し手に基いてhistoryのupdateを行なう。
		else if (bestMove)

			// quietな(駒を捕獲しない)best moveなのでkillerとhistoryとcountermovesを更新する。

			update_all_stats(pos, ss, bestMove, bestValue, beta, prevSq,
				quietsSearched, quietCount, capturesSearched, captureCount, depth);


		// Bonus for prior countermove that caused the fail low

		// bestMoveがない == fail lowしているケース。
		// fail lowを引き起こした前nodeでのcounter moveに対してボーナスを加点する。
		// 【計測資料 15.】search()でfail lowしているときにhistoryのupdateを行なう条件

		else if (  (depth >= 4 || PvNode)
				&& !priorCapture)
		{
			//Assign extra bonus if current node is PvNode or cutNode
			//or fail low was really bad
			bool extraBonus =  PvNode
							|| cutNode
							|| bestValue < alpha - 70 * depth;

			// continuation historyのupdate。PvNodeかcutNodeならボーナスを2倍する。
			update_continuation_histories(ss - 1, /*pos.piece_on(prevSq)*/prevPc, prevSq, stat_bonus(depth) * (1 + extraBonus));
		}
		// 将棋ではtable probe使っていないのでmaxValue関係ない。
		// ゆえにStockfishのここのコードは不要。(maxValueでcapする必要がない)
		/*
		if (PvNode)
			bestValue = std::min(bestValue, maxValue);
		*/

		// If no good move is found and the previous position was ttPv, then the previous
		// opponent move is probably good and the new position is added to the search tree.

		// もし良い指し手が見つからず(bestValueがalphaを更新せず)、前の局面はttPvを選んでいた場合は、
		// 前の相手の手がおそらく良い手であり、新しい局面が探索木に追加される。
		// (ttPvをtrueに変更してTTEntryに保存する)

		if (bestValue <= alpha)
			ss->ttPv = ss->ttPv || ((ss - 1)->ttPv && depth > 3);

		// -----------------------
		//  置換表に保存する
		// -----------------------

	    // Write gathered information in transposition table
		// 集めた情報を置換表に書き込む

		// betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
		// すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
		// さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
		// また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
		// ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
		// すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。

		if (!excludedMove && !(rootNode && thisThread->pvIdx))
		{
			ASSERT_LV3(pos.legal_promote(bestMove));
			tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
				bestValue >= beta ? BOUND_LOWER :
				PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
				depth, bestMove, ss->staticEval);
		}

		// qsearch()内の末尾にあるassertの文の説明を読むこと。
		ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

		return bestValue;
	}

	// -----------------------
	//      静止探索
	// -----------------------

	// qsearch()は静止探索を行う関数で、search()でdepth(残り探索深さ)が0になったときに呼び出されるか、
	// このqseach()自身から再帰的にもっと低いdepthで呼び出される。

	template <NodeType nodeType>
	Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
	{
		// -----------------------
		//     変数宣言
		// -----------------------

		// PV nodeであるか。
		// ※　ここがRoot nodeであることはないので、そのケースは考えなくて良い。
		static_assert(nodeType != Root);
		constexpr bool PvNode = nodeType == PV;

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
		HASH_KEY posKey;

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
		Value bestValue, value, ttValue, futilityValue, futilityBase;

		// pvHit			: 置換表から取り出した指し手が、PV nodeでsaveされたものであった。
		// givesCheck		: MovePickerから取り出した指し手で王手になるか
		// capture          : 駒を捕獲する指し手か
		bool pvHit, givesCheck , capture;

		// このnodeで何手目の指し手であるか
		int moveCount;

		// -----------------------
		//     nodeの初期化
		// -----------------------

		if (PvNode)
		{
			(ss + 1)->pv = pv;
			ss->pv[0] = MOVE_NONE;
		}

		Thread* thisThread = pos.this_thread();
		bestMove           = MOVE_NONE;
		ss->inCheck        = pos.checkers();
		moveCount          = 0;

		// -----------------------
		//    最大手数へ到達したか？
		// -----------------------

	    // Check for an immediate draw or maximum ply reached

		if (ss->ply >= MAX_PLY)
			return draw_value(REPETITION_DRAW, pos.side_to_move());

		if (pos.game_ply() > Limits.max_game_ply)
		{
			// この局面で詰んでいる可能性がある。その時はmatedのスコアを返すべき。
			// 詰んでいないなら引き分けのスコアを返すべき。
			// 関連)
			//    多くの将棋ソフトで256手ルールの実装がバグっている件
			//    https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/

			return pos.is_mated() ? mated_in(ss->ply) : draw_value(REPETITION_DRAW, pos.side_to_move());
		}

		ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

		// -----------------------
		//     置換表のprobe
		// -----------------------

		// Decide whether or not to include checks: this fixes also the type of
		// TT entry depth that we are going to use. Note that in qsearch we use
		// only two types of depth in TT: DEPTH_QS_CHECKS or DEPTH_QS_NO_CHECKS.

		// 置換表に登録するdepthはあまりマイナスの値だとおかしいので、
		// 王手がかかっているときは、DEPTH_QS_CHECKS(=0)、王手がかかっていないときはDEPTH_QS_NO_CHECKS(-1)とみなす。
		ttDepth = ss->inCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
													      : DEPTH_QS_NO_CHECKS;

		// Transposition table lookup
		// 置換表のlookup

		posKey = pos.hash_key();
		tte = TT.probe(posKey, ss->ttHit);
		ttValue = ss->ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;
		ttMove  = ss->ttHit ? pos.to_move(tte->move()) : MOVE_NONE;
		pvHit   = ss->ttHit && tte->is_pv();

		ASSERT_LV3(pos.legal_promote(ttMove));

		// pos.to_move()でlegalityのcheckに引っかかったパターンなので置換表にhitしなかったことにする。
		if (tte->move().to_u16() && !ttMove)
			ss->ttHit = false;

		// nonPVでは置換表の指し手で枝刈りする
		// PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)
		if (  !PvNode
			&& ss->ttHit
			&& tte->depth() >= ttDepth
			&& ttValue != VALUE_NONE // Only in case of TT access race
									 // ↑置換表から取り出したときに他スレッドが値を潰している可能性があるのでこのチェックが必要
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
				// Never assume anything about values stored in TT

				// 置換表に評価値が格納されているとは限らないのでその場合は評価関数の呼び出しが必要
				// bestValueの初期値としてこの局面のevaluate()の値を使う。これを上回る指し手があるはずなのだが..

				if ((ss->staticEval = bestValue = tte->eval()) == VALUE_NONE)
					ss->staticEval = bestValue = evaluate(pos);

				// 毎回evaluate()を呼ぶならtte->eval()自体不要なのだが、
				// 置換表の指し手でこのまま枝刈りできるケースがあるから難しい。
				// 評価関数がKPPTより軽ければ、tte->eval()をなくしても良いぐらいなのだが…。

				// ttValue can be used as a better position evaluation (~7 Elo)

				// 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
				// 精度で探索されたものであるなら、それをbestValueの初期値として使う。

				if (	ttValue != VALUE_NONE
					&& (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
					bestValue = ttValue;

			} else {

				// In case of null move search use previous static eval with a different sign

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
														   : -(ss - 1)->staticEval;

					// 1手前の局面(相手の手番)において 評価値が 500だとしたら、
					// 自分の手番側から見ると -500 なので、 -(ss - 1)->staticEval はそういう意味。
					// ただ、1手パスしているのに同じ評価値のままなのかという問題はあるが、そこが下限で、
					// そこ以上の指し手があるはずという意味で、bestValueの初期値としてはそうなっている。

				} else {

					// 評価関数の実行時間・精度によっては、こう書いたほうがいいかもという書き方。
					// 残り探索深さが大きい時は、こっちに切り替えるのはありかも…。
					// どちらが優れているかわからないので、optimizerに任せる。
					ss->staticEval = bestValue = evaluate(pos);
				}
			}

			// Stand pat. Return immediately if static value is at least beta
			// 現在のbestValueは、この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
			// 王手がかかっていないケースにおいては、この時点での静的なevalの値がbetaを上回りそうならこの時点で帰る。
			if (bestValue >= beta)
			{
	            // Save gathered info in transposition table
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
			futilityBase = bestValue + PARAM_FUTILITY_MARGIN_QUIET /*118*/;
		}

		// -----------------------
		//     1手ずつ調べる
		// -----------------------

		const PieceToHistory* contHist[] = { (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
												nullptr					  , (ss - 4)->continuationHistory,
												nullptr					  , (ss - 6)->continuationHistory };

		// Initialize a MovePicker object for the current position, and prepare
		// to search the moves. Because the depth is <= 0 here, only captures,
		// queen promotions, and other checks (only if depth >= DEPTH_QS_CHECKS)
		// will be generated.

		// 取り合いの指し手だけ生成する
		// searchから呼び出された場合、直前の指し手がMOVE_NULLであることがありうるが、
		// 静止探索の1つ目の深さではrecaptureを生成しないならこれは問題とならない。
		Square prevSq = to_sq((ss - 1)->currentMove);
		MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory,
										  &thisThread->captureHistory,
										  contHist,
										  prevSq);

		// 王手回避の指し手のうちquiet(駒を捕獲しない)な指し手の数
		int quietCheckEvasions = 0;

		// このあとnodeを展開していくので、evaluate()の差分計算ができないと速度面で損をするから、
		// evaluate()を呼び出していないなら呼び出しておく。
		evaluate_with_no_return(pos);

		// Loop through the moves until no moves remain or a beta cutoff occurs

		while ((move = mp.next_move()) != MOVE_NONE)
		{
			// MovePickerで生成された指し手はpseudo_legalであるはず。
			ASSERT_LV3(pos.pseudo_legal(move) && pos.legal_promote(move));

			// Check for legality
			// 指し手の合法性の判定は直前まで遅延させたほうが得だと思われていたのだが
			// (これが非合法手である可能性はかなり低いので他の判定によりskipされたほうが得)
			// Stockfish14から、静止探索でも、早い段階でlegal()を呼び出すようになった。
			if (!pos.legal(move))
				continue;

			// -----------------------
			//  局面を進める前の枝刈り
			// -----------------------

			givesCheck = pos.gives_check(move);
			capture = pos.capture(move);

			moveCount++;

			//
			// Futility pruning and moveCount pruning (~5 Elo)
			//

			// 自玉に王手がかかっていなくて、敵玉に王手にならない指し手であるとき、
			// 今回捕獲されるであろう駒による評価値の上昇分を
			// 加算してもalpha値を超えそうにないならこの指し手は枝刈りしてしまう。

			if (    bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				&& !givesCheck
				&&  to_sq(move) != prevSq
				&&  futilityBase > -VALUE_KNOWN_WIN
			//	&&  type_of(move) != PROMOTION) // TODO : これ入れたほうがいいのか？
				)
			{
				//assert(type_of(move) != ENPASSANT); // Due to !pos.advanced_pawn_push

				// MoveCountに基づく枝刈り
				if (moveCount > 2)
					continue;

				// moveが成りの指し手なら、その成ることによる価値上昇分もここに乗せたほうが正しい見積りになるはず。
				// 【計測資料 14.】 futility pruningのときにpromoteを考慮するかどうか。
				futilityValue = futilityBase + CapturePieceValueEg(pos, move);

				// これ、加算した結果、s16に収まらない可能性があるが、計算はs32で行ってして、そのあと、この値を用いないからセーフ。

				// futilityValueは今回捕獲するであろう駒の価値の分を上乗せしているのに
				// それでもalpha値を超えないというとってもひどい指し手なので枝刈りする。
				if (futilityValue <= alpha)
				{
					bestValue = std::max(bestValue, futilityValue);
					ASSERT_LV3(bestValue != -VALUE_INFINITE);
					continue;
				}

				// futilityBaseはこの局面のevalにmargin値を加算しているのだが、それがalphaを超えないし、
				// かつseeがプラスではない指し手なので悪い手だろうから枝刈りしてしまう。

				if (futilityBase <= alpha && !pos.see_ge(move, VALUE_ZERO + 1))
				{
					bestValue = std::max(bestValue, futilityBase);
					ASSERT_LV3(bestValue != -VALUE_INFINITE);
					continue;
				}
			}

			//
			//  Detect non-capture evasions
			//

			// 駒を取らない王手回避の指し手はよろしくない可能性が高いのでこれは枝刈りしてしまう。
			// 成りでない && seeが負の指し手はNG。王手回避でなくとも、同様。

			// ここ、わりと棋力に影響する。下手なことするとR30ぐらい変わる。

			// Do not search moves with negative SEE values (~5 Elo)
			if (    bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				&& !pos.see_ge(move))
				continue;


			// TODO : prefetchは、入れると遅くなりそうだが、many coreだと違うかも。
			// Speculative prefetch as early as possible
			//prefetch(TT.first_entry(pos.key_after(move)));

			// -----------------------
			//     局面を1手進める
			// -----------------------

			// 現在このスレッドで探索している指し手を保存しておく。
			ss->currentMove = move;

			ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck                ]
																	  [capture                    ]
																	  [to_sq(move)                ]
																	  [pos.moved_piece_after(move)];


			// Continuation history based pruning (~2 Elo)
			// Continuation historyベースの枝刈り
			// ※ Stockfish12でqsearch()にも導入された。
			if (  !capture
				&& bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				&& (*contHist[0])[to_sq(move)][pos.moved_piece_after(move)] < CounterMovePruneThreshold
				&& (*contHist[1])[to_sq(move)][pos.moved_piece_after(move)] < CounterMovePruneThreshold)
				continue;

			// movecount pruning for quiet check evasions
			// quietな指し手による王手回避のためのmovecountによる枝刈り。

			// 王手回避でquietな指し手は良いとは思えないから、捕獲する指し手を好むようにする。
			// だから、もし、qsearchで2つのquietな王手回避に失敗したら、
			// そこ以降(captureから生成しているのでそこ以降もquietな指し手)も良くないと
			// 考えるのは理に適っている。

			if (  bestValue > VALUE_TB_LOSS_IN_MAX_PLY
				&& quietCheckEvasions > 1
				&& !capture
				&& ss->inCheck)
				continue;

			quietCheckEvasions += !capture && ss->inCheck;

			// Make and search the move
			// 1手動かして、再帰的にqsearch()を呼ぶ
			pos.do_move(move, st, givesCheck);
			value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha, depth - 1);
			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// Check for a new best move
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

		// All legal moves have been searched. A special case: if we're in check
		// and no legal moves were found, it is checkmate.

		// 王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである。
		// どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
		// わかるわけだし、そもそもこのnodeが詰みだとわかるとこのnodeに再訪問する確率は極めて低く、
		// 置換表に保存しても置換表を汚すだけでほとんど得をしない。(レアケースなのでほとんど損もしないが)
		 
		// ※　計測したところ、置換表に保存したほうがわずかに強かったが、有意差ではなさげだし、
		// Stockfish10のコードが保存しないコードになっているので保存しないことにする。

		// 【計測資料 26.】 qsearchで詰みのときに置換表に保存する/しない。

		// チェスでは王手されていて、合法手がない時に詰みだが、将棋では、合法手がなければ詰みなので ss->inCheckの条件は不要かと思ったら、
		// qsearch()で王手されていない時は、captures(駒を捕獲する指し手)とchecks(王手の指し手)の指し手しか生成していないから、
		// moveCount==0だから詰みとは限らない。
		// 
		// 王手されている局面なら、evasion(王手回避手)を生成するから、moveCount==0なら詰みと確定する。
		// しかし置換表にhitした時にはbestValueの初期値は -VALUE_INFINITEではないので、そう考えると
		// ここは(Stockfishのコードのように)bestValue == -VALUE_INFINITEとするのではなくmoveCount == 0としたほうが良いように思うのだが…。
		// →　置換表にhitしたのに枝刈りがなされていない時点で有効手があるわけで詰みではないことは言えるのか…。
		// cf. https://yaneuraou.yaneu.com/2022/04/22/yaneuraous-qsearch-is-buggy/

		// if (ss->inCheck && bestValue == -VALUE_INFINITE)
		// ↑Stockfishのコード。↓こう変更したほうが良いように思うが計測してみると大差ない。
		// Stockfishも12年前は↑ではなく↓この書き方だったようだ。moveCountが除去された時に変更されてしまったようだ。
		//  cf. https://github.com/official-stockfish/Stockfish/commit/452f0d16966e0ec48385442362c94a810feaacd9
		// moveCountが再度導入されたからには、Stockfishもここは、↓の書き方に戻したほうが良いと思う。
		if (ss->inCheck && moveCount == 0)
		{
			// 合法手は存在しないはずだから指し手生成しても速攻終わるはず。
			ASSERT_LV5(!MoveList<LEGAL>(pos).size());

			return mated_in(ss->ply); // Plies to mate from the root
									  // rootから詰みまでの手数。
		}

	    // Save gathered info in transposition table
		// 詰みではなかったのでこれを書き出す。
		// ※　qsearch()の結果は信用ならないのでBOUND_EXACTで書き出すことはない。
		ASSERT_LV3(pos.legal_promote(bestMove));
		tte->save(posKey, value_to_tt(bestValue, ss->ply), pvHit,
				  bestValue >= beta ? BOUND_LOWER : BOUND_UPPER,
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

	// value_to_tt() adjusts a mate or TB score from "plies to mate from the root" to
	// "plies to mate from the current position". Standard scores are unchanged.
	// The function is called before storing a value in the transposition table.

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
		// →　これ足した結果がabs(x) < VALUE_INFINITEであることを確認すべきでは…。

		return  v >= VALUE_TB_WIN_IN_MAX_PLY  ? v + ply
			  : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
	}

	// value_from_tt() is the inverse of value_to_tt(): it adjusts a mate or TB score
	// from the transposition table (which refers to the plies to mate/be mated from
	// current position) to "plies to mate/be mated (TB win/loss) from the root". However,
	// for mate scores, to avoid potentially false mate scores related to the 50 moves rule
	// and the graph history interaction, we return an optimal TB score instead.

	// value_to_tt()の逆関数
	// ply : root node からの手数。(ply_from_root)
	Value value_from_tt(Value v, int ply) {

		if (v == VALUE_NONE)
			return VALUE_NONE;

		if (v >= VALUE_TB_WIN_IN_MAX_PLY)  // TB win or better
		{
			//if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c)
			//	return VALUE_MATE_IN_MAX_PLY - 1; // do not return a potentially false mate score

			return v - ply;
		}

		if (v <= VALUE_TB_LOSS_IN_MAX_PLY) // TB loss or worse
		{
			//if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c)
			//	return VALUE_MATED_IN_MAX_PLY + 1; // do not return a potentially false mate score

			return v + ply;
		}

		return v;
	}

	// update_pv() adds current move and appends child pv[]

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

	// update_all_stats() updates stats at the end of search() when a bestMove is found

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
		bonus2 = bestValue > beta + PawnValue ? bonus1				// larger bonus
											  : stat_bonus(depth);	// smaller bonus

		// Stockfishではcapture_or_promotion()からcapture()に変更された。[2022/3/23]
		if (!pos.capture(bestMove))
		{
	        // Increase stats for the best move in case it was a quiet move
			update_quiet_stats(pos, ss, bestMove, bonus2);

			// Decrease stats for all non-best quiet moves
			for (int i = 0; i < quietCount; ++i)
			{
				thisThread->mainHistory[from_to(quietsSearched[i])][us] << -bonus2;
				// StockfishはmainHistory↑は、[Color][from_to]の順なので注意。

				update_continuation_histories(ss, pos.moved_piece_after(quietsSearched[i]), to_sq(quietsSearched[i]), -bonus2);
			}
		}
		else
			// Increase stats for the best move in case it was a capture move

			captureHistory[to_sq(bestMove)][moved_piece][captured] << bonus1;
		// ※　StockfishはcaptureHistory↑は[pc][to][captured]の順なので注意。

		// Extra penalty for a quiet early move that was not a TT move or
		// main killer move in previous ply when it gets refuted.

		// (ss-1)->ttHit : 一つ前のnodeで置換表にhitしたか

		if (((ss - 1)->moveCount == 1 + (ss - 1)->ttHit || ((ss - 1)->currentMove == (ss - 1)->killers[0]))
			&& !pos.captured_piece())
			update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -bonus1);

	    // Decrease stats for all non-best capture moves
		// 捕獲する指し手でベストではなかったものをすべて減点する。

		for (int i = 0; i < captureCount; ++i)
		{
			moved_piece = pos.moved_piece_after(capturesSearched[i]);
			captured = type_of(pos.piece_on(to_sq(capturesSearched[i])));
			captureHistory[to_sq(capturesSearched[i])][moved_piece][captured] << -bonus1;
			// Stockfishは[pc][to][captured]の順なので注意。
		}
	}

	// update_continuation_histories() updates histories of the move pairs formed
	// by moves at ply -1, -2, -4, and -6 with current move.

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
			// Only update first 2 continuation histories if we are in check
			if (ss->inCheck && i > 2)
				break;
			if (is_ok((ss - i)->currentMove))
				(*(ss - i)->continuationHistory)[to][pc] << bonus;
		}
	}

	// update_quiet_stats() updates move sorting heuristics

	// update_quiet_stats()は、新しいbest moveが見つかったときに指し手の並べ替えheuristicsを更新する。
	// 具体的には駒を取らない指し手のstat tables、killer等を更新する。

	// move      = これが良かった指し手
	void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus)
	{
		// Update killers
		// killerの指し手のupdate

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

		// Update countermove history
		if (is_ok((ss - 1)->currentMove))
		{
			// 直前に移動させた升(その升に移動させた駒がある。今回の指し手はcaptureではないはずなので)
			Square prevSq = to_sq((ss - 1)->currentMove);
			thisThread->counterMoves[prevSq][pos.piece_on(prevSq)] = move;
		}
	}

	// When playing with strength handicap, choose best move among a set of RootMoves
	// using a statistical rule dependent on 'level'. Idea by Heinz van Saanen.

	// 手加減が有効であるなら、best moveを'level'に依存する統計ルールに基づくRootMovesの集合から選ぶ。
	// Heinz van Saanenのアイデア。
	Move Skill::pick_best(size_t multiPV) {

		const RootMoves& rootMoves = Threads.main()->rootMoves;
		static PRNG rng(now()); // 乱数ジェネレーターは非決定的であるべき。

		// RootMovesはすでにscoreで降順にソートされている。
		Value topScore = rootMoves[0].score;
		int delta = std::min(topScore - rootMoves[multiPV - 1].score, (Value)PawnValue);
		int maxScore = -VALUE_INFINITE;
		double weakness = 120 - 2 * level;

		// Choose best move. For each move score we add two terms, both dependent on
		// weakness. One is deterministic and bigger for weaker levels, and one is
		// random. Then we choose the move with the resulting highest score.

		// best moveを選ぶ。それぞれの指し手に対して弱さに依存する2つのterm(用語)を追加する。
		// 1つは、決定的で、弱いレベルでは大きくなるもので、1つはランダムである。
		// 次に得点がもっとも高い指し手を選択する。
		for (size_t i = 0; i < multiPV; ++i)
		{
			// This is our magic formula
			// これが魔法の公式
			int push = int((  weakness * int(topScore - rootMoves[i].score)
				+ delta * (rng.rand<unsigned>() % int(weakness))) / 128);

			if (rootMoves[i].score + push >= maxScore)
			{
				maxScore = rootMoves[i].score + push;
				best = rootMoves[i].pv[0];
			}
		}

		return best;
	}

} // namespace

/// MainThread::check_time() is used to print debug info and, more importantly,
/// to detect when we are out of available time and thus stop the search.

// 残り時間をチェックして、時間になっていればThreads.stopをtrueにする。
// main threadからしか呼び出されないのでロジックがシンプルになっている。
void MainThread::check_time()
{
	// When using nodes, ensure checking rate is not lower than 0.1% of nodes
	// 4096回に1回ぐらいのチェックで良い。
	if (--callsCnt > 0)
		return;

	// "stop"待ちなので以降の判定不要。
	if (Threads.main()->time_to_return_bestmove)
		return ;

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

	// We should not stop pondering until told so by the GUI
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
	{
		if (Limits.wait_stop)
		{
			// stopが来るまで待つので、Threads.stopは変化させない。
			// 代わりに"info string time to return bestmove."と出力する。
			output_time_to_return_bestmove();

		} else {

			Threads.stop = true;
		}
	}
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
	// ssにはstack + 7が渡されるものとする。
	void init_for_search(Position& pos, Stack* ss , Move pv[], bool qsearch)
	{

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

			// 探索ノード数のゼロ初期化等。(threads.cppに書いてある初期化コードのコピペ)
			th->nodes = th->bestMoveChanges = /* th->tbHits = */ th->nmpMinPly = 0;
			th->rootDepth = th->completedDepth = 0;

			// history類を全部クリアする。この初期化は少し時間がかかるし、探索の精度はむしろ下がるので善悪はよくわからない。
			// th->clear();

			// 探索スレッド用の初期化(探索部・学習部で共通)
			search_thread_init(th,ss,pv);

			if (!qsearch)
			{
				// 通常探索の時はrootMovesを設定してやる。
				// qsearchの時はrootMovesは用いないので関係がない。

				// rootMovesの設定
				auto& rootMoves = th->rootMoves;

				rootMoves.clear();
				for (auto m : MoveList<LEGAL>(pos))
					rootMoves.push_back(Search::RootMove(m));

				ASSERT_LV3(!rootMoves.empty());
			}

			// 学習用の実行ファイルではスレッドごとに置換表を持っているので
			// 探索前に自分(のスレッド用)の置換表の世代カウンターを回してやる。
			th->tt.new_search();
		}

#if defined(YANEURAOU_ENGINE_NNUE)
		init_fv_scale();
#endif
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

		// 詰まされているのか
		if (pos.is_mated())
		{
			pvs.push_back(MOVE_RESIGN);
			return ValueAndPV(mated_in(/*ss->ply*/ 0 + 1), pvs);
		}

		// 探索の初期化
		init_for_search(pos, ss , pv, /* qsearch = */true);

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
	// 注意)
	// また引数で指定されたdepth == 0の時、qsearch、0未満の時、evaluateを呼び出すが、
	// この時、rootMovesは得られない。
	// 
	// あと、multiPVで値を取り出したい時、
	//   pos.this_thread()->rootMoves[N].value
	// の値は、反復深化での今回のiterationでのupdateされていない場合、previous_scoreを用いないといけない。
	// →　usi.cppの、読み筋の出力部のコードを読むこと。
	// 
	// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。
	// 　また、Threads.stopが来ると探索を中断してしまうので、そのときのPVは正しくない。
	// 　search()から戻ったあと、Threads.stop == trueなら、その探索結果を用いてはならない。
	// 　あと、呼び出し前は、Threads.stop == falseの状態で呼び出さないと、探索を中断して返ってしまうので注意。
	//
	ValueAndPV search(Position& pos, int depth_, size_t multiPV /* = 1 */, u64 nodesLimit /* = 0 */)
	{
		std::vector<Move> pvs;

		// depth == 0の時、qsearch、0未満の時、evaluateを呼び出すが、この時、rootMovesは得られないので注意。
		Depth depth = depth_;
		if (depth < 0)
			return std::pair<Value, std::vector<Move>>(Eval::evaluate(pos), std::vector<Move>());

		if (depth == 0)
			return qsearch(pos);

		Stack stack[MAX_PLY + 10], *ss = stack + 7;
		Move pv[MAX_PLY + 1];

		// 探索の初期化
		init_for_search(pos, ss , pv, /* qsearch = */ false);

		// this_threadに関連する変数のaliasを用意。
		// ※ "th->"と書かずに済むのであれば、Stockfishのsearch()のコードをコピペできるので。
		auto th				 = pos.this_thread();
		auto& rootDepth		 = th->rootDepth;
		auto& pvIdx			 = th->pvIdx;
		auto& rootMoves		 = th->rootMoves;
		auto& completedDepth = th->completedDepth;
		auto& selDepth		 = th->selDepth;

		// bestmoveとしてしこの局面の上位N個を探索する機能
		//size_t multiPV = Options["MultiPV"];

		// この局面での指し手の数を上回ってはいけない
		multiPV = std::min(multiPV, rootMoves.size());

		// ノード制限にMultiPVの値を掛けておかないと、depth固定、MultiPVありにしたときに1つの候補手に同じnodeだけ思考したことにならない。
		nodesLimit *= multiPV;

		Value alpha			 = -VALUE_INFINITE;
		Value beta			 =  VALUE_INFINITE;
		Value delta			 = -VALUE_INFINITE;
		Value bestValue		 = -VALUE_INFINITE;

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
					Value prev = rootMoves[pvIdx].averageScore;
					delta = Value(PARAM_ASPIRATION_SEARCH_DELTA) + int(prev) * prev / 19178;

					alpha = std::max(prev - delta, -VALUE_INFINITE);
					beta  = std::min(prev + delta,  VALUE_INFINITE);
				}

				// aspiration search
				int failedHighCnt = 0;
				while (true)
				{
					Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt);
					bestValue = ::search<Root>(pos, ss, alpha, beta, adjustedDepth, false);

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

					delta += delta / 4 + 2;
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
