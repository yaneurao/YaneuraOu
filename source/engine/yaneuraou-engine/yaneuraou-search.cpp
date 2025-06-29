#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

// -----------------------
//   やねうら王 標準探索部
// -----------------------

// 計測資料置き場 : https://github.com/yaneurao/YaneuraOu/wiki/%E6%8E%A2%E7%B4%A2%E9%83%A8%E3%81%AE%E8%A8%88%E6%B8%AC%E8%B3%87%E6%96%99

// -----------------------
//   includes
// -----------------------

#include "../../search.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>	// std::log(),std::pow(),std::round()
#include <cstring>	// memset()

#include "../../position.h"
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
	//o["SkillLevel"] << Option(20, 0, 20);

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
	o["PvInterval"]     << Option(300, 0, 100000000);

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

	// Stockfishには、Eloレーティングを指定して棋力調整するためのエンジンオプションがあるようだが…。
	// o["UCI_Elo"]               << Option(1320, 1320, 3190);

}

// パラメーターのランダム化のときには、
// USIの"gameover"コマンドに対して、それをログに書き出す。
void gameover_handler([[maybe_unused]] const std::string& cmd)
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

// Futility margin
// depth(残り探索深さ)に応じたfutility margin。
// ※ RazoringはStockfish12で効果がないとされてしまい除去された。
Value futility_margin(Depth d, bool noTtCutNode, bool improving, bool oppWorsening) {
	Value futilityMult       = PARAM_FUTILITY_MARGIN_ALPHA1  - PARAM_FUTILITY_MARGIN_ALPHA2 * noTtCutNode;
	Value improvingDeduction = improving * futilityMult * 2;
	Value worseningDeduction = oppWorsening * futilityMult / 3;
	return futilityMult * d - improvingDeduction - worseningDeduction;
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

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.

// correctionHistoryの値を生のstaticEvalに加算し、
// 評価がテーブルベースの範囲に達しないように保証します。

Value to_corrected_static_eval(Value v /*, const Worker& w, const Position& pos */) {
#if 0
	const Color us    = pos.side_to_move();
	const auto  pcv   = w.pawnCorrectionHistory[us][pawn_structure_index<Correction>(pos)];
	const auto  mcv   = w.materialCorrectionHistory[us][material_index(pos)];
	const auto  macv  = w.majorPieceCorrectionHistory[us][major_piece_index(pos)];
	const auto  micv  = w.minorPieceCorrectionHistory[us][minor_piece_index(pos)];
	const auto  wnpcv = w.nonPawnCorrectionHistory[WHITE][us][non_pawn_index<WHITE>(pos)];
	const auto  bnpcv = w.nonPawnCorrectionHistory[BLACK][us][non_pawn_index<BLACK>(pos)];
	const auto cv =
		(6384 * pcv + 3583 * macv + 6492 * micv + 6725 * (wnpcv + bnpcv) + cntcv * 5880) / 131072;
	v += cv;

	return std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
#endif

	// やねうら王では、evaluate()がVALUE_MAX_EVALの間の値しか返さないことは
	// 保証されているし、現時点では、correction historyを用いていないので、
	// ここでは何の補整もしない。
	return v;
}

// History and stats update bonus, based on depth
// depthに基づく、historyとstatsのupdate bonus

int stat_bonus(Depth d) { return std::min(168 * d - 100, 1718); }
	// →　やねうら王では、Stockfishの統計値、統計ボーナスに関して手を加えないことにしているので
	// この値はStockfishの値そのまま。

// History and stats update malus, based on depth
// depthに基づく、historyとstatsのupdate malus
// ※ malus(「悪い」、「不利益」みたいな意味)は
// 「統計的なペナルティ」または「マイナスの修正値」を計算するために使用される。
// この関数は、ある行動が望ましくない結果をもたらした場合に、その行動の評価を減少させるために使われる
// TODO : あとで
int stat_malus(Depth d) { return std::min(768 * d - 257, 2351); }

// 【計測資料 30.】　Reductionのコード、Stockfish 9と10での比較

// Reductions lookup table initialized at startup
// 探索深さを減らすためのReductionテーブル。起動時に初期化する。
std::array<int, MAX_MOVES> reductions; // [depth or moveNumber]

// 残り探索深さをこの深さだけ減らす。
// 注意 : Stockfish 17(2024.11)で、1024倍して返すことになった。
//
// 引数の意味
//   d  : depth
//   mn : move_count
//   i  : improving , 評価値が2手前から上がっているかのフラグ。
//                    上がっていないなら悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
//   delta, rootDelta : staticEvalとchildのeval(value)の差が一貫して低い時にreduction量を増やしたいので、
//                   そのためのフラグ。(これがtrueだとreduction量が1増える)
Depth reduction(bool i, Depth d, int mn, Value delta, Value rootDelta) {
	int reductionScale = reductions[d] * reductions[mn];
	return (reductionScale + PARAM_REDUCTION_ALPHA - delta * PARAM_REDUCTION_GAMMA / rootDelta)
		+ (!i && reductionScale > PARAM_REDUCTION_BETA) * 1135;
	// PARAM_REDUCTION_BETAの値、将棋ではもう少し小さくして、reductionの適用範囲を広げた方がいいかも？
}

#if 0
// チェスでは、引き分けが0.5勝扱いなので引き分け回避のための工夫がしてあって、
// 以下のようにvalue_drawに揺らぎを加算することによって探索を固定化しない(同じnodeを
// 探索しつづけて千日手にしてしまうのを回避)工夫がある。
// 将棋の場合、普通の千日手と連続王手の千日手と劣等局面による千日手(循環？)とかあるので
// これ導入するのちょっと嫌。
// →　やねうら王では未使用

// Add a small random component to draw evaluations to avoid 3fold-blindness
// 引き分け時の評価値VALUE_DRAW(0)の代わりに±1の乱数みたいなのを与える。
// nodes : 現在の探索node数(乱数のseed代わりに用いる)
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + (nodes & 0x2); }
#endif

#if 0
// Skill structure is used to implement strength limit.
// If we have a UCI_Elo, we convert it to an appropriate skill level, anchored to the Stash engine.
// This method is based on a fit of the Elo results for games played between the master at various
// skill levels and various versions of the Stash engine, all ranked at CCRL.
// Skill 0 .. 19 now covers CCRL Blitz Elo from 1320 to 3190, approximately
// Reference: https://github.com/vondele/Stockfish/commit/a08b8d4e9711c20acedbfe17d618c3c384b339ec
//
// Skill構造体は強さの制限の実装に用いられる。
// (わざと手加減して指すために用いる) →　やねうら王では未使用
struct Skill {
	// skill_level : 手加減のレベル。20未満であれば手加減が有効。0が一番弱い。(R2000以上下がる)
	// uci_elo     : 0以外ならば、そのelo ratingになるように調整される。
	Skill(int skill_level, int uci_elo) {
		if (uci_elo)
		{
			double e = double(uci_elo - 1320) / (3190 - 1320);
			level = std::clamp((((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438), 0.0, 19.0);
		}
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

	Move best = Move::none();
};
#else
// やねうら王ではSkillLevelを実装しない。
struct Skill {
	// dummy constructor
	Skill(int,int) {}

	// 常にfalseを返す。つまり、手加減の無効化。
	bool enabled() { return false;}
	bool time_to_pick(Depth) const { return true; }
	Move pick_best(size_t) { return Move::none();}
	Move best = Move::none();
};

#endif

// quietsSearchedの最大数。
// Stockfish 16ではquietsSearchedの配列サイズが[64]から[32]になったが、
// 将棋ではハズレのquietの指し手が大量にあるので
// それがベストとは限らない。
// →　比較したところ、64より32の方がわずかに良かったので、とりあえず32にしておく。(V7.73mとV7.73m2との比較)
constexpr int MAX_QUIETS_SEARCHED = 32 /*32*/;

template <NodeType nodeType>
Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

template <NodeType nodeType>
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth = 0);

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply /*,int r50c */);
void update_pv(Move* pv, Move move, const Move* childPv);
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void update_quiet_histories(const Position& pos, Stack* ss, /*WorkerThread&*/ Thread& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
	ValueList<Move, MAX_QUIETS_SEARCHED>& quietsSearched,
	ValueList<Move, MAX_QUIETS_SEARCHED>& capturesSearched,
	Depth depth);


// Utility to verify move generation. All the leaf nodes up
// to the given depth are generated and counted, and the sum is returned.
// 
// ※　perftとはperformance testの略。
// 開始局面から深さdepthまで全合法手で進めるときの総node数を数えあげる。
// 指し手生成が正常に行われているかや、生成速度等のテストとして有用。

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


// Search::init() is called at startup to initialize various lookup tables
// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
	// 時間がかかるものは、"isready"応答(Search::clear())でやるべき。
	// また、スレッド数など、起動時には决定しておらず、"isready"のタイミングでしか決定していないものも
	// "isready"応答でやるべき。

	//for (int i = 1; i < MAX_MOVES; ++i)
	//	Reductions[i] = int((20.37 + std::log(Threads.size()) / 2) * std::log(i));
	// →
	// 　このReductionsテーブルの初期化は、Threads.size()に依存するから、
	// 　Search::clear()に移動させた。
}

// Search::clear() resets search state to its initial value
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

	// テーブルの初期化は、↑で探索パラメーターを読み込んだあとに行われなければならない。

	// -----------------------
	//   テーブルの初期化
	// -----------------------

	// LMRで使うreduction tableの初期化
	//
	// pvとnon pvのときのreduction定数
	// 0.05とか変更するだけで勝率えらく変わる

	for (size_t i = 1; i < reductions.size(); ++i)
		reductions[i] = int(PARAM_REDUCTIONS_PARAM1 / 128.0 * std::log(i));

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
	// エンジンオプションのFV_SCALEでEval::NNUE::FV_SCALEを初期化する。
	init_fv_scale();
#endif

}

// MainThread::search() is started when the program receives the UCI 'go'
// command. It searches from the root position and outputs the "bestmove".
// 探索開始時に(goコマンドなどで)呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::search()
{
	// ---------------------
	// perft(performance test)
	// ---------------------

	if (Limits.perft)
	{
		nodes = perft<true>(rootPos, Limits.perft);
		sync_cout << "\nNodes searched: " << nodes << ", time " << Time.elapsed() << "ms.\n" << sync_endl;
		return;
	}

	// root nodeにおける自分の手番
	Color us = rootPos.side_to_move();

	// --- 今回の思考時間の設定。
	// これは、ponderhitした時にponderhitにパラメーターが付随していれば
	// 再計算するする必要性があるので、いずれにせよ呼び出しておく必要がある。

	Time.init(Limits, us, rootPos.game_ply());

	// ---------------------
	// やねうら王固有の初期化
	// ---------------------

	// --- やねうら王独自拡張

	// 今回、通常探索をしたかのフラグ
	// このフラグがtrueなら(定跡にhitしたり1手詰めを発見したりしたので)探索をスキップした。
	bool search_skipped = true;

	// 検討モード用のPVを出力するのか。
	Limits.consideration_mode = Options["ConsiderationMode"];

	// fail low/highのときにPVを出力するかどうか。
	Limits.outout_fail_lh_pv = Options["OutputFailLHPV"];

	// PVが詰まるのを抑制するために、前回出力時刻を記録しておく。
	lastPvInfoTime = 0;

	// ponder用の指し手の初期化
	// やねうら王では、ponderの指し手がないとき、一つ前のiterationのときのPV上の(相手の)指し手を用いるという独自仕様。
	// Stockfish本家もこうするべきだと思う。
	ponder_candidate = Move::none();

	// --- contempt factor(引き分けのスコア)

	// 引き分け時の値として現在の手番に応じた値を設定してやる。

	int draw_value = (int)((us == BLACK ? Options["DrawValueBlack"] : Options["DrawValueWhite"]) * PawnValue / 100);

	// 探索のleaf nodeでは、相手番(root_color != side_to_move)である場合、 +draw_valueではなく、-draw_valueを設定してやらないと非対称な探索となって良くない。
	// 例) 自分は引き分けを勝ち扱いだと思って探索しているなら、相手は、引き分けを負けとみなしてくれないと非対称になる。
	drawValueTable[REPETITION_DRAW][ us] = +draw_value;
	drawValueTable[REPETITION_DRAW][~us] = -draw_value;

	// PVの出力間隔[ms]
	// go infiniteはShogiGUIなどの検討モードで動作させていると考えられるので
	// この場合は、PVを毎回出力しないと読み筋が出力されないことがある。
	Limits.pv_interval = (Limits.infinite || Limits.consideration_mode) ? 0 : (int)Options["PvInterval"];

#if defined(SHOGI24)
	// ---------------------
	//    将棋倶楽部24対策
	// ---------------------

	// 相手玉が取れるなら取る。
	//
	// 相手玉が取れる局面は、(直前で王手放置があったということだから)非合法局面で、
	// 将棋所ではそのような局面からの対局開始はできないが、ShogiGUIでは対局開始できる。
	//
	// また、将棋倶楽部24で王手放置の局面を作ることができるので、
	// 相手玉が取れることがある。
	// 
	// ゆえに、取れるなら取る指し手を指せたほうが良い。
	//
	// 参考動画 : https://www.youtube.com/watch?v=8nwJcKH0x0c

	auto their_king = rootPos.king_square(~us);
	auto our_piece  = rootPos.attackers_to(their_king) & rootPos.pieces(us);
	// 敵玉に利いている自駒があるなら、それを移動させて勝ち。
	if (our_piece)
	{
		Square from = our_piece.pop();
		Square to   = their_king;
		Move16 m16  = make_move16(from, to);
		Move   m    = rootPos.to_move(m16);

		// 玉を取る指し手はcapturesで生成されていない可能性がある。
		// 仕方がないので、rootMoves[0]を書き換えることにする。

		// 玉で玉を取る手はrootMovesに含まれないので、場合によっては、それしか指し手がない場合に、
		// rootMoves.size() == 0だけど、玉で玉を取る指し手だけがあることは起こり得る。
		// (この理由から、玉を取る判定は、合法手がない判定より先にしなければならない)

		if (rootMoves.size() == 0)
			rootMoves.emplace_back(m);
		else
			rootMoves[0].pv[0] = m;

		rootMoves[0].score = rootMoves[0].usiScore = mate_in(1);

		goto SKIP_SEARCH;
	}
#endif

	// ---------------------
	// 合法手がないならここで投了
	// ---------------------

	// 現局面で詰んでいる。
	if (rootMoves.empty())
	{
		// 投了の指し手と評価値をrootMoves[0]に積んでおけばUSI::pv()が良きに計らってくれる。
		// 読み筋にresignと出力されるが、将棋所、ShogiGUIともにバグらないのでこれで良しとする。
		rootMoves.emplace_back(Move::resign());

		// 評価値を用いないなら代入しなくて良いのだが(Stockfishはそうなっている)、
		// このあと、↓USI::pv()を呼び出したいので、scoreをきちんと設定しておいてやる。
		rootMoves[0].score = rootMoves[0].usiScore = mated_in(0);

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
		if (bestMove != Move::none())
		{
			// root movesの集合に突っ込んであるはず。
			// このときMultiPVが利かないが、ここ真面目にMultiPVして指し手を返すのは
			// プログラムがくちゃくちゃになるのでいまはこれは仕様としておく。

			// トライルールのとき、その指し手がgoコマンドで指定された指し手集合に含まれることを
			// 保証しないといけないのでrootMovesのなかにこの指し手が見つからないなら指すわけにはいかない。

			// 入玉宣言の条件を満たしているときは、
			// goコマンドを処理したあとのthreads.cppでMove::win()は追加されているはず。

			auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
			if (it_move != rootMoves.end())
			{
				std::swap(rootMoves[0], *it_move);

				// 1手詰めのときのスコアにしておく。
				rootMoves[0].score = rootMoves[0].usiScore = mate_in(1);;

				goto SKIP_SEARCH;
			}
		}
	}

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

	// 以下のrootMovesが無い時の処理は、やねうら王では別途やっているのでここではしない。

    //if (rootMoves.empty())
    //{
    //    rootMoves.emplace_back(Move::none());
    //    sync_cout << "info depth 0 score "
    //              << UCI::value(rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW) << sync_endl;
    //}
    //else
    //{

	Threads.start_searching(); // main以外のthreadを開始する
	Thread::search();          // main thread(このスレッド)も探索に参加する。

	//}

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

			Skill skill = Skill(/*(int)Options["SkillLevel"]*/ 20, 0);

			// 並列して探索させていたスレッドのうち、ベストのスレッドの結果を選出する。
			if (    int(Options["MultiPV"]) == 1
				&& !Limits.depth
				&& !skill.enabled()
				&&  rootMoves[0].pv[0] != Move::resign()
				// ⇨　やねうら王では、詰んでいるときなどMove::resign()を積むので、
				// 　rootMoves[0].pv[0]がMove::resign()でありうる。
				// 　このとき、main thread以外のth->rootMoves[0]は、指し手がなくアクセス違反になるのでアクセスしてはならない。
				&& !search_skipped
				// ⇨　定跡などの指し手を指させるために、このチェックが必要。(bestThreadを変更されては困るため)
				)

				bestThread = Threads.get_best_thread();


			if (/*bestThread != this && */

				// ⇨　Stockfishでは、ベストな指し手として返すスレッドがmain threadではないのなら、
				// 　その読み筋は出力していなかったので(探索中はmain threadしか読み筋を出力しない)ここで読み筋を出力しておくようになっている。

				// 　やねうら王では、best moveを返す直前に必ず読み筋を送ることにした。
				// →　いずれにせよ、mateを見つけた時に最終的なPVを出力していないと、詰みではないscoreのPVが最終的な読み筋としてGUI上に
				//     残ることになるからよろしくない。PV自体は必ず出力すべき。

				!Limits.silent
				)
				
				sync_cout << Search::pv(bestThread->rootPos, TT, bestThread->completedDepth) << sync_endl;

			/*
				bestThreadがmainThreadではなくなる場合、探索した最大depthが減ることがありうる。
				なので、最後の出力でdepthが減っていることは普通にある。これ、「あれ？」と思う挙動ではあるが…。
			*/

			output_final_pv_done = true;
		}
	};

	// ここで思考は完了したのでwait_stopの処理。
	// まだ思考が完了したことを通知していないならば。

	if (Limits.wait_stop && !Threads.main()->time_to_return_bestmove)
		output_time_to_return_bestmove();

    // When we reach the maximum depth, we can arrive here without a raise of
    // Threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.

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

	// 投了スコアが設定されていて、歩の価値を100として正規化した値がそれを下回るなら投了。(やねうら王独自拡張)
	// ただし定跡の指し手にhitした場合などはrootMoves[0].score == -VALUE_INFINITEになっているのでそれは除外。
	auto resign_value = (int)Options["ResignValue"];
	if (bestThread->rootMoves[0].score != -VALUE_INFINITE
		&& bestThread->rootMoves[0].score * 100 / PawnValue <= -resign_value)
		bestThread->rootMoves[0].pv[0] = Move::resign();

	// サイレントモードでないならbestな指し手を出力
	if (!Limits.silent)
	{
		// sync_cout～sync_endlで全体を挟んでいるのでここを実行中に他スレッドの出力が割り込んでくる余地はない。

		// Send again PV info if we have a new best thread
		// if (bestThread != this)
		//      sync_cout << UCI::pv(bestThread->rootPos, bestThread->completedDepth) << sync_endl;

		// ↑こんなにPV出力するの好きじゃないので省略。

		// ベストなスレッドの指し手を返す。
		sync_cout << "bestmove " << bestThread->rootMoves[0].pv[0];

		// ponderの指し手の出力。
		// pvにはbestmoveのときの読み筋(PV)が格納されているので、ponderとしてpv[1]があればそれを出力してやる。
		// また、pv[1]がない場合(rootでfail highを起こしたなど)、置換表からひねり出してみる。
		if (bestThread->rootMoves[0].pv.size() > 1
			|| bestThread->rootMoves[0].extract_ponder_from_tt(TT, rootPos, Threads.main()->ponder_candidate))
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
	// counterMovesをnullptrに初期化するのではなくNO_PIECEのときの値を番兵として用いる。
	for (int i = 7; i > 0; --i)
	{
		(ss - i)->continuationHistory           = &th->continuationHistory[0][0](NO_PIECE, SQ_ZERO); // Use as a sentinel
//      (ss - i)->continuationCorrectionHistory = &this->continuationCorrectionHistory[NO_PIECE][0];
		(ss - i)->staticEval                    = VALUE_NONE;
	}

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

// Main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has been
// consumed, the user stops the search, or the maximum search depth is reached.

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// Lazy SMPなので、置換表を共有しながらそれぞれのスレッドが勝手に探索しているだけ。

void Thread::search()
{
	// ---------------------
	//      variables
	// ---------------------

	Move  pv[MAX_PLY + 1];

	// Allocate stack with extra size to allow access from (ss-7) to (ss+2)
	// (ss-7) is needed for update_continuation_histories(ss-1, ...) which accesses (ss-6)
	// (ss+2) is needed for initialization of statScore and killers

	// (ss-7)から(ss+2)へのアクセスを許可するために、追加のサイズでスタックを割り当てます
	// (ss-7)はupdate_continuation_histories(ss-1, ...)のために必要であり、これは(ss-6)にアクセスします
	// (ss+2)はstatScoreとkillersの初期化のために必要です

	// continuationHistoryのため、(ss-7)から(ss+2)までにアクセスしたいので余分に確保しておく。

	// ■ 備考
	//
	// stackは、先頭10個を初期化しておけば十分。
	// そのあとはsearch()の先頭でss+1,ss+2を適宜初期化していく。
	// RootNodeはss->ply == 0がその条件。(ss->plyはsearch_thread_initで初期化されている)

	Stack stack[MAX_PLY + 10] = {};
	Stack*ss                  = stack + 7;

	// 探索部、学習部のスレッドの共通初期化コード
	// ※　Stockfishのここにあったコードはこの関数に移動。
	search_thread_init(this, ss, pv);

	// alpha,beta : aspiration searchの窓の範囲(alpha,beta)
	// delta      : apritation searchで窓を動かす大きさdelta
	Value alpha, beta, delta;

	// 探索の安定性を評価するために前回のiteration時のbest moveを記録しておく。
	Move  lastBestMove = Move::none();
	Depth lastBestMoveDepth = 0;

	// もし自分がメインスレッドであるならmainThreadにそのポインタを入れる。
	// 自分がスレーブのときはnullptrになる。
	MainThread* mainThread = (this == Threads.main() ? Threads.main() : nullptr);
	Thread* thisThread = this;

	// timeReduction      : 読み筋が安定しているときに時間を短縮するための係数。
	// Stockfish9までEasyMoveで処理していたものが廃止され、Stockfish10からこれが導入された。
	// totBestMoveChanges : 直近でbestMoveが変化した回数の統計。読み筋の安定度の目安にする。
	double timeReduction = 1.0 , totBestMoveChanges = 0;;

	// この局面の手番側
	Color us = rootPos.side_to_move();

	// 反復深化の時に1回ごとのbest valueを保存するための配列へのindex
	// 0から3までの値をとる。
	int iterIdx = 0;

	// Stockfish 14の頃は、反復深化のiterationが浅いうちはaspiration searchを使わず
	// 探索窓を (-VALUE_INFINITE , +VALUE_INFINITE)としていたが、Stockfish 16では、
	// 浅いうちからaspiration searchを使うようになったので、alpha,betaの初期化はここでやらなくなった。

	bestValue = -VALUE_INFINITE;

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
	//Skill skill((int)Options["SkillLevel"], 0);

	Skill skill(20, 0);


	// When playing with strength handicap enable MultiPV search that we will
	// use behind-the-scenes to retrieve a set of possible moves.

	// 強さの手加減が有効であるとき、MultiPVを有効にして、その指し手のなかから舞台裏で指し手を探す。
	// ※　SkillLevelが有効(設定された値が20未満)のときは、MultiPV = 4で探索。

	if (skill.enabled())
		multiPV = std::max(multiPV, size_t(4));

	// この局面での指し手の数を上回ってはいけない
	multiPV = std::min(multiPV, rootMoves.size());

	// ---------------------
	//   反復深化のループ
	// ---------------------

	// 反復深化の探索深さが深くなって行っているかのチェック用のカウンター
	// これが増えていない時、同じ深さを再度探索していることになる。(fail highし続けている)
	// ※　あまり同じ深さでつっかえている時は、aspiration windowの幅を大きくしてやるなどして回避する必要がある。
	int searchAgainCounter = 0;

	lowPlyHistory.fill(0);

	// Iterative deepening loop until requested to stop or the target depth is reached
	// 要求があるか、または目標深度に達するまで反復深化ループを実行します

	// 1つ目のrootDepthはこのthreadの反復深化での探索中の深さ。
	// 2つ目のrootDepth (Threads.main()->rootDepth)は深さで探索量を制限するためのもの。
	// main threadのrootDepthがLimits.depthを超えた時点で、
	// slave threadはこのループを抜けて良いのでこういう書き方になっている。
	while (   ++rootDepth < MAX_PLY
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

		// Save the last iteration's scores before the first PV line is searched and
		// all the move scores except the (new) PV are set to -VALUE_INFINITE.

		// 最初のPVラインが探索される前に、最後のイテレーションのスコアを保存し、
		// (新しい)PVを除くすべての指し手のスコアを-VALUE_INFINITEに設定します。

		// aspiration window searchのために反復深化の前回のiterationのスコアをコピーしておく
		for (RootMove& rm : rootMoves)
			rm.previousScore = rm.score;

		// 将棋ではこれ使わなくていいような？

		//size_t pvFirst = 0;
		//pvLast         = 0;

		// 探索深さを増やすかのフラグがfalseなら、同じ深さを探索したことになるので、
		// searchAgainCounterカウンターを1増やす
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

			// Reset UCI info selDepth for each depth and each PV line
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

			// aspiration windowの幅
			// 精度の良い評価関数ならばこの幅を小さくすると探索効率が上がるのだが、
			// 精度の悪い評価関数だとこの幅を小さくしすぎると再探索が増えて探索効率が低下する。
			// やねうら王のKPP評価関数では35～40ぐらいがベスト。
			// やねうら王のKPPT(Apery WCSC26)ではStockfishのまま(18付近)がベスト。
			// もっと精度の高い評価関数を用意すべき。
			// この値はStockfish10では20に変更された。
			// Stockfish 12(NNUEを導入した)では17に変更された。
			// Stockfish 12.1では16に変更された。
			// Stockfish 16では10に変更された。
			// Stockfish 17では 5に変更された。

			delta     = PARAM_ASPIRATION_SEARCH1 + std::abs(rootMoves[pvIdx].meanSquaredScore) / PARAM_ASPIRATION_SEARCH2;
			Value avg = rootMoves[pvIdx].averageScore;
			alpha     = std::max(avg - delta,-VALUE_INFINITE);
			beta      = std::min(avg + delta, VALUE_INFINITE);

			//// Adjust optimism based on root move's averageScore
			//optimism[us] = 137 * avg / (std::abs(avg) + 91);
			//optimism[~us] = -optimism[us];
			// → このoptimismは、StockfishのNNUE評価関数で何やら使っているようなのだが…。

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
				// Adjust the effective depth searched, but ensure at least one effective increment for every
				// four searchAgain steps (see issue #2717).

				// fail highするごとにdepthを下げていく処理
				Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
				rootDelta = beta - alpha;
				bestValue = ::search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

				// Bring the best move to the front. It is critical that sorting
				// is done with a stable algorithm because all the values but the
				// first and eventually the new best one is set to -VALUE_INFINITE
				// and we want to keep the same order for all the moves except the
				// new PV that goes to the front. Note that in the case of MultiPV
				// search the already searched PV lines are preserved.

				// それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
				// 一つ目の指し手以外は-VALUE_INFINITEが返る仕様なので並べ替えのために安定ソートを
				// 用いないと前回の反復深化の結果によって得た並び順を変えてしまうことになるのでまずい。

				std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.end());

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

					// 以下、やねうら王独自拡張。
					&& (rootDepth < 3 || mainThread->lastPvInfoTime + Limits.pv_interval <= Time.elapsed())
					// silent modeや検討モードなら出力を抑制する。
					&& !Limits.silent
					// ただし、outout_fail_lh_pvがfalseならfail high/fail lowのときのPVを出力しない。
					&&  Limits.outout_fail_lh_pv
					)
				{
					// 最後に出力した時刻を記録しておく。
					mainThread->lastPvInfoTime = Time.elapsed();
					sync_cout << Search::pv(rootPos, TT, rootDepth) << sync_endl;
				}

				// aspiration窓の範囲外
				if (bestValue <= alpha)
				{
					// fails low

					// betaをalphaにまで寄せてしまうと今度はfail highする可能性があるので
					// betaをalphaのほうに少しだけ寄せる程度に留める。
					beta  = (alpha + beta) / 2;
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
				delta += delta / 3;

				ASSERT_LV3(-VALUE_INFINITE <= alpha && beta <= VALUE_INFINITE);
			}

			// MultiPVの候補手をスコア順に再度並び替えておく。
			// (二番目だと思っていたほうの指し手のほうが評価値が良い可能性があるので…)

			stable_sort(rootMoves.begin() /* + pvFirst */, rootMoves.begin() + pvIdx + 1);

			if (   mainThread
				// メインスレッド以外はPVを出力しない。
				&& !Limits.silent
				// また、silentモードの場合もPVは出力しない。
				// (やねうら王独自拡張。教師生成の時などにPVが出力されたくないので)
				&& (/* Threads.stop || */ pvIdx + 1 == multiPV || Time.elapsed() > 3000)
				// ⇨　Stockfishは、Threads.stopが条件に入っていて、停止するときにもPVを出力している。
				// やねうら王では、Threads::search()の最後で行うから、stopする時にはここでは出力しないことにする。
				// ⇨　MultiPVのときは最後の候補手を求めた直後とする。MultiPVでない時は、毎回。(pvIdx == 0かつmultiPV == 1なので)
				// ただし、MultiPVの時でも時間が3秒以上経過してからは、MultiPVのそれぞれの指し手の更新ごと。
				&&  mainThread->lastPvInfoTime + Limits.pv_interval <= Time.elapsed()
				// ⇨　思考エンジンオプションの"PvInterval"の出力間隔を守る。それより短い間隔では出力しない。
				// デフォルト 300[ms]だが、これは0に設定しても問題はない。(GUI側の表示処理で詰まる可能性はある。)
				)
			{
				mainThread->lastPvInfoTime = Time.elapsed();
				sync_cout << Search::pv(rootPos, TT, rootDepth) << sync_endl;
			}

		} // multi PV

			// ここでこの反復深化の1回分は終了したのでcompletedDepthに反映させておく。
		if (!Threads.stop)
			completedDepth = rootDepth;

		if (rootMoves[0].pv[0] != lastBestMove)
		{
			lastBestMove      = rootMoves[0].pv[0];
			lastBestMoveDepth = rootDepth;
		}

		// Have we found a "mate in x"?
		// x手詰めを発見したのか？

		//if (Limits.mate && bestValue >= VALUE_MATE_IN_MAX_PLY
		//    && VALUE_MATE - bestValue <= 2 * Limits.mate)
		//    Threads.stop = true;

		// multi_pvのときは一つのpvで詰みを見つけただけでは停止するのは良くないので
		// 早期終了はmultiPV == 1のときのみ行なう。(やねうら王独自拡張)

		if (multiPV == 1)
		{
			// 勝ちを読みきっているのに将棋所の表示が追いつかずに、将棋所がフリーズしていて、その間の時間ロスで
			// 時間切れで負けることがある。
			// mateを読みきったとき、そのmateの2.5倍以上、iterationを回しても仕方ない気がするので探索を打ち切るようにする。
			// ⇨ あと、rootで1手詰め呼び出さなくしたので、その影響もあって、VALUE_MATE == bestValueになることはあるから、この時、
			//   rootDepth == 1で探索を終了されては困る。もう少し先まで調べて欲しいので、+2しておく。
			if (!Limits.mate
				&& bestValue >= VALUE_MATE_IN_MAX_PLY
				&& (VALUE_MATE - bestValue + 2) * 5/2 < (Value)(rootDepth))
				break;

			// 詰まされる形についても同様。こちらはmateの2.5倍以上、iterationを回したなら探索を打ち切る。
			if (!Limits.mate
				&& bestValue <= VALUE_MATED_IN_MAX_PLY
				&& (bestValue - (-VALUE_MATE) + 2) * 5/2 < (Value)(rootDepth))
				break;
		}

		if (!mainThread)
			continue;

		//
		// main threadのときは探索の停止判定が必要
		//

		// ponder用の指し手として、2手目の指し手を保存しておく。
		// これがmain threadのものだけでいいかどうかはよくわからないが。
		// とりあえず、無いよりマシだろう。(やねうら王独自拡張)
		if (mainThread->rootMoves[0].pv.size() > 1)
			mainThread->ponder_candidate = mainThread->rootMoves[0].pv[1];

		// -- やねうら王独自の処理ここまで↑↑↑

		// If the skill level is enabled and time is up, pick a sub-optimal best move
		// もしSkillLevelが有効であり、タイムアップになったなら、(手加減用として)最適っぽいbest moveを選ぶ。
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
			// (このへんの仕組み、やねうら王では、Stockfishとは異なる)
			if (!Threads.stop && Time.search_end == 0)
			{
				// 1つしか合法手がない(one reply)であるだとか、利用できる時間を使いきっているだとか、

				double fallingEval = (11 + 2 * (mainThread->bestPreviousAverageScore - bestValue)
										+  (mainThread->iterValue[iterIdx] - bestValue))
									/ 100.0;
				fallingEval = std::clamp(fallingEval, 0.580, 1.667);

				// If the bestMove is stable over several iterations, reduce time accordingly
				// もしbestMoveが何度かのiterationにおいて安定しているならば、思考時間もそれに応じて減らす

				timeReduction = lastBestMoveDepth + 8 < completedDepth ? 1.495 : 0.687;
				double reduction = (1.48 + mainThread->previousTimeReduction) / (2.17 * timeReduction);
				// rootでのbestmoveの不安定性。
				// bestmoveが不安定であるなら思考時間を増やしたほうが良い。
				double bestMoveInstability = 1 + 1.88 * totBestMoveChanges / Threads.size();
				//double recapture = limits.capSq == rootMoves[0].pv[0].to_sq() ? 0.955 : 1.005;
				// ⇨ やねうら王ではcapSqは実装してない。

				double totalTime =
				  /*mainThread->tm*/Time.optimum() * fallingEval * reduction * bestMoveInstability /* *recapture*/;

				// 合法手が1手しかないときはtotalTime = 0として、即指しする。(これはやねうら王独自改良)
				if (rootMoves.size() == 1)
					totalTime = 0;

				// bestMoveが何度も変更になっているならunstablePvFactorが大きくなる。
				// failLowが起きてなかったり、1つ前の反復深化から値がよくなってたりするとimprovingFactorが小さくなる。
				// Stop the search if we have only one legal move, or if available time elapsed

				auto elapsedTime = Time.elapsed();

				if (elapsedTime > totalTime)
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

				// ■ 備考
				// 
				//   ここは固定秒での対局であっても、探索に影響する。

				//threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.506;
				Threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.506;

				// Stockfish 14ではtotalTime * 0.58
				// Stockfish 16では totalTime * 0.50
				// Stockfish 17では、totalTime * 0.506
				// Stockfish 14と16だと14の定数が勝る？(V7.74taya-t3 VS V7.74taya-t4)
				// ※ ここはパラメーター調整すべき。

			}
		}

		// ここで、反復深化の1回前のスコアをiterValueに保存しておく。
		// iterIdxは0..3の値をとるようにローテーションする。
		mainThread->iterValue[iterIdx] = bestValue;
		iterIdx                        = (iterIdx + 1) & 3;

	} // iterative deeping , 反復深化の1回分の終了

	if (!mainThread)
		return;

	mainThread->previousTimeReduction = timeReduction;

	// If the skill level is enabled, swap the best PV line with the sub-optimal one
	// もしSkillLevelが有効なら、最善応手列を(手加減用として)最適っぽい応手列と入れ替える。
	if (skill.enabled())
		std::swap(rootMoves[0], *std::find(rootMoves.begin(), rootMoves.end(),
			skill.best ? skill.best : skill.pick_best(multiPV)));
}

// -----------------------
//      通常探索
// -----------------------

namespace {

// Main search function for both PV and non-PV nodes
// PV , non-PV node共用のメインの探索関数。

// cutNode = LMRで悪そうな指し手に対してreduction量を増やすnode

template <NodeType nodeType>
Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
{
	// -----------------------
	//     nodeの種類
	// -----------------------

	// PV nodeであるか(root nodeはPV nodeに含まれる)
	constexpr bool PvNode   = nodeType != NonPV;

	// root nodeであるか
	constexpr bool rootNode = nodeType == Root;

	// allNodeであるか。
	// ⇨ allNodeとは、PvNodeでもなくcutNodeでもないnodeのこと。
	//   allNodeとは、ゲーム木探索で、全ての子ノードを評価する必要があるノードのこと。
	const bool     allNode = !(PvNode || cutNode);

	// Dive into quiescence search when the depth reaches zero
	// 残り探索深さが1手未満であるなら現在の局面のまま静止探索を呼び出す
	if (depth <= 0)
	{
		constexpr auto nt = PvNode ? PV : NonPV;
		return qsearch<nt>(pos, ss, alpha, beta);
	}

	// Limit the depth if extensions made it too large
	// 拡張によって深さが大きくなりすぎた場合、深さを制限します

	depth = std::min(depth, MAX_PLY - 1);

	// 次の指し手で引き分けに持ち込めてかつ、betaが引き分けのスコアより低いなら
	// 早期枝刈りが実施できる。
	// →　将棋だとあまり千日手が起こらないので効果がなさげ。
#if 0
	// Check if we have an upcoming move that draws by repetition
	// 直近の手が繰り返しによる引き分けになるかを確認します

	if (   !rootNode
		&& alpha < VALUE_DRAW
		&& pos.upcoming_repetition(ss->ply))
	{
		alpha = value_draw(this->nodes);
		if (alpha >= beta)
			return alpha;
	}
#endif

	ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
	ASSERT_LV3(PvNode || (alpha == beta - 1));
	ASSERT_LV3(0 < depth && depth < MAX_PLY);
	// IIDを含め、PvNodeではcutNodeで呼び出さない。
	ASSERT_LV3(!(PvNode && cutNode));

	// -----------------------
	//     変数宣言
	// -----------------------

	// pv               : このnodeからのPV line(読み筋)

	Move pv[MAX_PLY + 1];

	// do_move()するときに必要
	StateInfo st;

	//ASSERT_ALIGNED(&st, Eval::NNUE::CacheLineSize);

	// このnodeのhash key
	HASH_KEY posKey;

	// move					: MovePickerから1手ずつもらうときの一時変数
	// excludedMove			: singular extemsionのときに除外する指し手
	// bestMove				: このnodeのbest move
	Move move, excludedMove, bestMove;

	// extension			: 延長する深さ
	// newDepth				: 新しいdepth(残り探索深さ)
	Depth extension, newDepth;

	// bestValue			: このnodeのbestな探索スコア
	// value				: 探索スコアを受け取る一時変数
	// eval					: このnodeの静的評価値(の見積り)
	// maxValue             : table base probeに用いる。※ 将棋だと用いない。
	// probCutBeta          : prob cutする時のbetaの値。
	Value bestValue, value, eval /*, maxValue */, probCutBeta;

	// givesCheck			: moveによって王手になるのか
	// improving			: 直前のnodeから評価値が上がってきているのか
	//   このフラグを各種枝刈りのmarginの決定に用いる
	//   cf. Tweak probcut margin with 'improving' flag : https://github.com/official-stockfish/Stockfish/commit/c5f6bd517c68e16c3ead7892e1d83a6b1bb89b69
	//   cf. Use evaluation trend to adjust futility margin : https://github.com/official-stockfish/Stockfish/commit/65c3bb8586eba11277f8297ef0f55c121772d82c
	// priorCapture         : 1つ前の局面は駒を取る指し手か？
	// opponentWorsening    : 相手の状況が悪化しているかのフラグ
	bool givesCheck, improving, priorCapture, opponentWorsening;

	// capture              : moveが駒を捕獲する指し手もしくは歩を成る手であるか
	// ttCapture			: 置換表の指し手がcaptureする指し手であるか
	bool capture, ttCapture;

	// 1手前の局面でのreductionの量
	int  priorReduction;

	// moveによって移動させる駒
	Piece movedPiece;

	// capturesSearched : 駒を捕獲する指し手(+歩の成り)
	// quietsSearched   : 駒を捕獲しない指し手(-歩の成り)

	ValueList<Move, MAX_QUIETS_SEARCHED> capturesSearched;
	ValueList<Move, MAX_QUIETS_SEARCHED> quietsSearched;

	// -----------------------
	// Step 1. Initialize node
	// -----------------------

	//     nodeの初期化

	Thread* thisThread = pos.this_thread();
	ss->inCheck		   = pos.checkers();
	priorCapture	   = pos.captured_piece();
	Color us		   = pos.side_to_move();
	ss->moveCount	   = 0;
	bestValue		   = -VALUE_INFINITE;
	//maxValue	       = VALUE_INFINITE;
	// →　将棋ではtable probe使っていないのでmaxValue関係ない。

	//  Timerの監視

	// Check for the available remaining time
	// 残りの利用可能な時間を確認します
	// ⇨ 備考) これはメインスレッドのみが行なう。

	if (thisThread == Threads.main())
		static_cast<MainThread*>(thisThread)->check_time();

	// Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
	// selDepth情報をGUIに送信するために使用します（selDepthは1からカウントし、plyは0からカウントします）

	if (PvNode && thisThread->selDepth < ss->ply + 1)
		thisThread->selDepth = ss->ply + 1;

	// -----------------------
	//  RootNode以外での処理
	// -----------------------

	if (!rootNode)
	{
		// -----------------------
		// Step 2. Check for aborted search and immediate draw
		// Step 2. 中断された探索および即時引き分けを確認します
		// -----------------------

		// 探索の中断と、引き分けについてチェックする
		// 連続王手による千日手、および通常の千日手、優等局面・劣等局面。

		// 連続王手による千日手に対してdraw_value()は、詰みのスコアを返すので、rootからの手数を考慮したスコアに変換する必要がある。
		// そこで、value_from_tt()で変換してから返すのが正解。

		// 教師局面生成時には、これをオフにしたほうが良いかも知れない。
		// ただし、そのときであっても連続王手の千日手の判定は有効にしておく。

		// →　優等局面・劣等局面はrootより遡って判定しない。
		// (USIで出力する評価値がVALUE_SUPERIORになるのはちょっと嫌だし、
		// 　優等局面に突入するからと言って即詰みを逃がすのもちょっと嫌)
		// cf. https://github.com/yaneurao/YaneuraOu/issues/264
		auto draw_type = pos.is_repetition(ss->ply);
		if (draw_type != REPETITION_NONE)
			return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);

		// 最大手数を超えている、もしくは停止命令が来ている。
		if (   Threads.stop.load(std::memory_order_relaxed)
			|| ss->ply >= MAX_PLY
			|| pos.game_ply() > Limits.max_game_ply
			)
			return draw_value(REPETITION_DRAW, pos.side_to_move());

		/*
		■ 備考

			256手ルールで
			1. 256手目の局面で判定を行う場合は、
				「詰まされていない、かつ、連続王手の千日手が成立していない」ならば、
				引き分けとしてreturnして良い。

			2. 257手目の局面で判定を行う場合は、
				この局面に到達したということは、256手目の局面で合法手があったということだから、
				引き分けとしてreturnして良い。

			ということになる。1.の方式でやったほうが、256手目の局面で指し手生成とか
			1手ずつ試していくのとかが丸ごと端折れるので探索効率は良いのだが、コードが複雑になる。

			2.の方式でやっていいならば、257手目の局面なら単にreturnするだけで済む。
			探索効率は少し悪いが、コードは極めてシンプルになる。

			また、256手ルールで256手目である指し手を指して、257手目の局面で連続王手の千日手が
			成立するときはこれは非合法手扱いをすべきだと思う。

			だから、2.の方式で判定するときは、この連続王手の千日手判定を先にやってから、
			257手目の局面であるかの判定を行う必要がある。

			上記のコードは、そうなっている。
		*/

		// -----------------------
		// Step 3. Mate distance pruning.
		// Step 3. 詰みまでの手数による枝刈り
		// -----------------------

		// Step 3. Mate distance pruning. Even if we mate at the next move our score
		// would be at best mate_in(ss->ply+1), but if alpha is already bigger because
		// a shorter mate was found upward in the tree then there is no need to search
		// because we will never beat the current alpha. Same logic but with reversed
		// signs apply also in the opposite condition of being mated instead of giving
		// mate. In this case, return a fail-high score.

		// ステップ3. 詰みまでの手数による枝刈り。たとえ次の手でメイトしても、スコアは最大で
		// mate_in(ss->ply + 1)となります。しかし、もしalphaがすでにそれ以上であれば、
		// ツリー上でより短い詰みが見つかったことを意味し、これ以上探索する必要はありません。
		// 現在のalphaを上回ることはできないからです。
		// 逆に、詰みされる場合も同様のロジックが適用されますが、符号が逆になります。
		// この場合、fail highを返します。

		// ■ 備考
		// 
		//   rootから5手目の局面だとして、このnodeのスコアが5手以内で
		//   詰ますときのスコアを上回ることもないし、
		//   5手以内で詰まさせるときのスコアを下回ることもない。
		//   そこで、alpha , betaの値をまずこの範囲に補正したあと、
		//   alphaがbeta値を超えているならbeta cutする。

		alpha = std::max(mated_in(ss->ply    ), alpha);
		beta  = std::min(mate_in (ss->ply + 1), beta );
		if (alpha >= beta)
			return alpha;
	}

	// -----------------------
	//  探索Stackの初期化
	// -----------------------

	// rootからの手数
	ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

	// 前の指し手で移動させた先の升目
	// null moveのときはis_ok() == falseなのでSQ_NONEとする。
	Square prevSq        = (ss - 1)->currentMove.is_ok() ? (ss - 1)->currentMove.to_sq() : SQ_NONE;
	priorReduction       = (ss - 1)->reduction;
	(ss - 1)->reduction  = 0;
	bestMove             = Move::none();
	ss->statScore        = 0;
	ss->isPvNode         = PvNode;
	(ss + 2)->cutoffCnt  = 0;


	// -----------------------
	// Step 4. Transposition table lookup.
	// Step 4. 置換表の参照
	// -----------------------

	// このnodeで探索から除外する指し手。ss->excludedMoveのコピー。
	excludedMove = ss->excludedMove;

	// excludedMoveがある(singular extension時)は、
	// 前回の全探索の置換表の値を上書きする部分探索のスコアは
	// 欲しくないので、excluded moveがある場合には異なるhash keyを用いて
	// 異なるTTEntryを読み書きすべきだと思うが、
	// Stockfish 16で、同じTTEntryを用いるようになった。
	// (ただしexcluded moveがある時に探索した結果はTTEntryにsaveしない)
	// つまり、probeして情報だけ利用する感じのようだ。情報は使えるということなのだろうか…。

	//posKey = excludedMove == Move::none() ? pos.hash_key() : pos.hash_key() ^ HASH_KEY(make_key(excludedMove));
	// ↑このときpos.key()のbit0を破壊することは許されないので、make_key()でbit0はクリアしておく。
	// excludedMoveがMove::none()の時はkeyを変更してはならない。

	// ↓Stockfish 16で異なるTTEntryを使わないようになって次のように単純化された。
	//    cf. https://github.com/official-stockfish/Stockfish/commit/8d3457a9966f8c744ab7f8536be408196ccd8af9

	/**
	excluded moveについて詳しく。

	singular extensionとは、置換表から拾ってきた指し手だけがすこぶるよろしい指し手である時、
	一本道の変化だから、この指し手はもっと延長してあげようということである。駒のただ捨てなどで
	指し手を引き伸ばすような水平線効果を抑える役割もある。(たぶん)

	だから、置換表の指し手を除外して同じnodeで探索しなおす必要がある。
	この時の探索における置換表に元あった指し手をexcluded moveと呼ぶ。

	つまり、この時の探索結果は、excluded moveを除外して得られた探索結果なので、
	同じTTEntry(置換表のエントリー)に書き出すのはおかしいわけである。

	だからexcluded moveがある時は、局面のhash keyを、このexcluded moveを
	考慮したhash keyに変更して別のTTEntryを用いるようにしていた。

	そのコードが上の pos.hash_key() ^ HASH_KEY(make_key(excludedMove) の部分である。
	(make_keyはexcludedMoveをseedとする疑似乱数を生成する)

	ところが、これをStockfishの上のcommitは、廃止するというのである。

	メリットとしては、make_keyで全然違うTTEntryを見に行くとCPUのcacheにmiss hitするので、
	そこで遅くなるのだが、同じTTEntryを見に行くなら、間違いなくCPU cacheにhitするというものである。
	また、元エントリーの値のうち、staticEval(evaluate()した時の値)ぐらいは使えるんじゃね？ということである。

	デメリットとしては、この時の探索結果をそのTTEntryに保存してしまうとそれはexcluded moveがない時の
	探索結果としては正しくないので、このような保存はできないということである。
	それにより、次回も同じexcluded moveでsingular extensionする時に今回の探索結果が活用できない
	というのはある。

	そのどちらが得なのかということのようである。
	**/

	posKey = pos.hash_key();
	auto [ttHit, ttData, ttWriter] = TT.probe(posKey, pos);

	// Need further processing of the saved data
	// 保存されたデータのさらなる処理が必要です

	ss->ttHit = ttHit;

	// 置換表の指し手
	// 置換表にhitしなければMove::none()
	// RootNodeであるなら、(MultiPVなどでも)現在注目している1手だけがベストの指し手と仮定できるから、
	// それが置換表にあったものとして指し手を進める。
	// 
	// ■ 備考
	//
	//   TTにMove::win()も指し手として書き出す場合は、
	//   tte->move()にはMove::win()も含まれている可能性がある。
	//   この時、pos.to_move(MOVE_WIN) == Move::win()なので、ttMove == Move::win()となる。
	//   ⇨ 現状のやねうら王では、Move::win()はTTに書き出さない。
	//

	ttData.move = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
				: ttHit    ? ttData.move
						   : Move::none();

	// 置換表上のスコア
	// 置換表にhitしなければVALUE_NONE

	// singular searchとIIDとのスレッド競合を考慮して、ttValue , ttMoveの順で取り出さないといけないらしい。
	// cf. More robust interaction of singular search and iid : https://github.com/official-stockfish/Stockfish/commit/16b31bb249ccb9f4f625001f9772799d286e2f04

	ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply /*, pos.rule50_count()*/) : VALUE_NONE;

	ASSERT_LV3(pos.legal_promote(ttData.move));

	ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);

	// 置換表の指し手がcaptureであるか。
	// 置換表の指し手がcaptureなら高い確率でこの指し手がベストなので、他の指し手を
	// そんなに読まなくても大丈夫。なので、このnodeのすべての指し手のreductionを増やす。

	// ここ、capture_or_promotion()とかcapture_or_pawn_promotion()とか色々変えてみたが、
	// 現在では、capture()にするのが良いようだ。[2022/04/13]
	// →　捕獲する指し手で一番小さい価値上昇は歩の捕獲(+ 2*PAWN_VALUE)なのでこれぐらいの差になるもの
	//     歩の成り、香の成り、桂の成り　ぐらいは調べても良さそうな…。
	// → Stockfishでcapture_stage()になっているところはそれに倣うことにした。[2023/11/05]

	ttCapture    = ttData.move && pos.capture_stage(ttData.move);

	// At this point, if excluded, skip straight to step 6, static eval. However,
	// to save indentation, we list the condition in all code between here and there.

	// この時点で、除外された場合は、ステップ6の静的評価に直接進みます。
	// しかし、インデントを減らすために、ここからそこまでのコード内の条件をすべて記載しています。

	// ■ 補足
	// 
	// 置換表にhitしなかった時は、PV nodeのときだけttPvとして扱う。
	// これss->ttPVに保存してるけど、singularの判定等でsearchをss+1ではなくssで呼び出すことがあり、
	// そのときにss->ttPvが破壊される。なので、破壊しそうなときは直前にローカル変数に保存するコードが書いてある。

	// At non-PV nodes we check for an early TT cutoff
	// 置換表の値による枝刈り

	if (  !PvNode                     // PV nodeでは置換表の指し手では枝刈りしない(PV nodeはごくわずかしかないので..)
	    && !excludedMove
	    && ttData.depth > depth - (ttData.value <= beta) // 置換表に登録されている探索深さのほうが深くて
		&& is_valid(ttData.value)     // Can happen when !ttHit or when access race in probe()
								      // !ttHitの場合やprobe()でのアクセス競合時に発生する可能性があります。
		&& (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
		&& (cutNode == (ttData.value >= beta) || depth > 5))
		// ■ 解説
		// 
		// ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
		// ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
		// 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
		// このままこの値でreturnして良い。
	{
		// If ttMove is quiet, update move sorting heuristics on TT hit
		// ttMoveがquietの指し手である場合、置換表ヒット時に指し手のソート用ヒューリスティクスを更新します。

		// ■ 備考
		// 
		// 置換表の指し手でbeta cutが起きたのであれば、この指し手をkiller等に登録する。
		// 捕獲する指し手か成る指し手であればこれは(captureで生成する指し手なので)killerを更新する価値はない。
		//
		// ただし置換表の指し手には、hash衝突によりpseudo-leaglでない指し手である可能性がある。
		// update_quiet_stats()で、この指し手の移動元の駒を取得してCounter Moveとするが、
		// それがこの局面の手番側の駒ではないことがあるのでゆえにここでpseudo_legalのチェックをして、
		// Counter Moveに先手の指し手として後手の指し手が登録されるような事態を回避している。
		// その時に行われる誤ったβcut(枝刈り)は許容できる。(non PVで生じることなのでそこまで探索に対して悪い影響がない)
		// cf. https://yaneuraou.yaneu.com/2021/08/17/about-the-yaneuraou-bug-that-appeared-in-the-long-match/

		//if (/* ttMove && */ is_ok(ttMove))
		// やねうら王ではttMoveがMove::win()であることはありうるので注意が必要。
		// is_ok(m)==falseの時、Position::to_move(m)がmをそのまま帰すことは保証されている。
		// そのためttMoveがMove::win()でありうる。これはstatのupdateをされると困るのでis_ok()で弾く必要がある。
		// is_ok()は、ttMove == Move::none()の時はfalseなのでこの条件を省略できる。
		// ⇨　Move::win()書き出すの、筋が良くないし、入玉自体が超レアケースなので棋力に影響しないし、これやめることにする。

		// If ttMove is quiet, update move sorting heuristics on TT hit (~2 Elo)
		// 置換表にhitした時に、ttMoveがquietの指し手であるなら、指し手並び替えheuristics(quiet_statsのこと)を更新する。

		if (ttData.move && ttData.value >= beta)
		{
			// Bonus for a quiet ttMove that fails high
			// fail highしたquietなquietな(駒を取らない)ttMove(置換表の指し手)に対するボーナス

			if (!ttCapture)
				update_quiet_histories(pos, ss, /* *this */ *thisThread, ttData.move,
					std::min(125 * depth - 77, 1157));

			// Extra penalty for early quiet moves of the previous ply
			// 1手前の早い時点のquietの指し手に対する追加のペナルティ

			// 1手前がMove::null()であることを考慮する必要がある。

			if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 3 && !priorCapture)
				update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2301);

		}

		// Partial workaround for the graph history interaction problem
		// For high rule50 counts don't produce transposition table cutoffs.

		// グラフ履歴の相互作用問題に対する部分的な回避策
		// rule50カウントが高い場合は、トランスポジションテーブルによるカットオフを行いません。

		// ⇨　将棋では関係のないルールなので無視して良いが、rule50_count < 90 が(チェスの)通常の状態なので、
		//    if成立時のreturnはしなければならない。
		/*
		if (pos.rule50_count() < 90)
		{
			// TTによるcut off。
			// TODO : 将棋でもこの処理いるかも？

			if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) && pos.legal(ttData.move)
				&& !is_decisive(ttData.value))
			{
				do_move(pos, ttData.move, st);
				Key nextPosKey = pos.key();
				auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
				undo_move(pos, ttData.move);

				// Check that the ttValue after the tt move would also trigger a cutoff
				if (!is_valid(ttDataNext.value))
					return ttData.value;
				if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
					return ttData.value;
			}
			else
				return ttData.value;
		}
		*/
		return ttData.value;
	}

	// -----------------------
	//     宣言勝ち
	// -----------------------

	// Step 5. Tablebases probe
	// 《StockfishのStep 5.のコードは割愛》

	// chessだと終盤データベースというのがある。
	// これは将棋にはないが、将棋には代わりに宣言勝ちというのがある。
	// 宣言勝ちと1手詰めだと1手詰めの方が圧倒的に多いので、まず1手詰め判定を行う。

	// 以下は、やねうら王独自のコード。

	// -----------------------
	//    1手詰みか？
	// -----------------------

	if (!rootNode && !ttHit
#if !defined(CHECK_MATE1PLY_REGARDLESS_OF_EXCLUDED_MOVE)
		&& !excludedMove
#endif
	) {
		// ttHitしている時は、すでに1手詰め・宣言勝ちの判定は完了しているはずなので行わない。
		// excludedMoveがある時はttHitしてttMoveがあるはずだが、置換表壊されるケースがあるので念のためチェックはしないといけない。
		// !rootnodeではなく!PvNodeの方がいいかも？
		// (PvNodeでは置換表の指し手を信用してはいけないという方針なら)

		// excludedMoveがある時には本当は、それを除外して詰み探索をする必要があるが、
		// 詰みがある場合は、singular extensionの判定の前までにbeta cutするので、結局、
		// 詰みがあるのにexcludedMoveが設定されているということはありえない。
		// よって、「excludedMoveは設定されていない」時だけ詰みがあるかどうかを調べれば良く、
		// この条件を詰み探索の事前条件に追加することができる。
		// 
		// ただし、excludedMoveがある時、singular extensionの事前条件を満たすはずで、
		// singular extensionはttMoveが存在することがその条件に含まれるから、
		// ss->ttHit == trueになっているはず。
		// 
		// RootNodeでは1手詰め判定、ややこしくなるのでやらない。(RootMovesの入れ替え等が発生するので)
		// 置換表にhitしたときも1手詰め判定はすでに行われていると思われるのでこの場合もはしょる。
		// depthの残りがある程度ないと、1手詰めはどうせこのあとすぐに見つけてしまうわけで1手詰めを
		// 見つけたときのリターン(見返り)が少ない。
		// ただ、静止探索で入れている以上、depth == 1でも1手詰めを判定したほうがよさげではある。

		if (PARAM_SEARCH_MATE1 && !ss->inCheck)
		{
			move = Mate::mate_1ply(pos);

			if (move != Move::none())
			{
				// 1手詰めスコアなので確実にvalue > alphaなはず。
				// 1手詰めは次のnodeで詰むという解釈

				// ■ 注意
				//
				//   このとき、bestValueをvalue_to_tt()で変換してはならない。
				//   mate_in()はrootからの計算された詰みのスコアであるから、このまま置換表に格納してよい。
				//
				//	 あるいは、VALUE_MATE - 1をvalue_to_tt()で変換したものを置換表に格納するかだが、
				//   その場合、returnで返す値を用意するのに再度変換が必要になるのでそういう書き方は良くない。

				bestValue = mate_in(ss->ply + 1);

				ASSERT_LV3(pos.legal_promote(move));
				if (!excludedMove)
					ttWriter.write(posKey, bestValue , ss->ttPv, BOUND_EXACT,
						    std::min(MAX_PLY - 1, depth + 6), move, VALUE_NONE , TT.generation());
				// ⇨ excludedMoveがあるときは置換表に書き出さないルールになっているので、
				//   この条件式が必要なので注意。

				// ■　【計測資料 39.】 mate1plyの指し手を見つけた時に置換表の指し手でbeta cutする時と同じ処理をする。

				// 兄弟局面でこのmateの指し手がよい指し手ある可能性があるので
				// ここでttMoveでbeta cutする時と同様の処理を行うと短い時間ではわずかに強くなるっぽいのだが
				// 長い時間で計測できる差ではなかったので削除。

				/*
				   ■ 備考

				    1手詰めを発見した時に、save()でdepthをどのように設定すべきか問題について。

					即詰みは絶対であり、MAX_PLYの深さで探索した時の結果と同じであるから、
					以前はMAX_PLYにしていたのだが、よく考えたら、即詰みがあるなら上位ノードで
					枝刈りが発生してこのノードにはほぼ再訪問しないと考えられるのでこんなものが
					置換表に残っている価値に乏しく、また、MAX_PLYにしてしまうと、
					TTEntryのreplacement strategy上、depthが大きなTTEntryはかなりの優先度になり
					いつまでもreplacementされない。

					こんな情報、lostしたところで1手詰めならmate1ply()で一手も進めずに得られる情報であり、
					最優先にreplaceすべきTTEntryにも関わらずである。

					かと言ってDEPTH_NONEにするとtt->depth()が 0 になってしまい、枝刈りがされなくなる。
					そこで、depth + 6 ぐらいがベストであるようだ。
				*/
				return bestValue;
			}
		}
	}

	// -----------------------
	//    宣言勝ちか？
	// -----------------------
	
	// 置換表にhitしていないときは宣言勝ちの判定をまだやっていないということなので今回やる。
	// PvNodeでは置換表の指し手を信用してはいけないので毎回やる。
	if (!ttData.move || PvNode)
	{
		// 王手がかかってようがかかってまいが、宣言勝ちの判定は正しい。
		// (トライルールのとき王手を回避しながら入玉することはありうるので)
		// トライルールのときここで返ってくるのは16bitのmoveだが、置換表に格納するには問題ない。
		move = pos.DeclarationWin();
		if (move != Move::none())
		{
			bestValue = mate_in(ss->ply + 1); // 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈

			ASSERT_LV3(pos.legal_promote(move));
			/*
			if (!excludedMove)
				ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT,
					std::min(MAX_PLY - 1, depth + 6),
					is_ok(move) ? move : MOVE_NONE, // MOVE_WINはMOVE_NONEとして書き出す。
					VALUE_NONE, TT.generation());
			*/

			// ■ 備考
			// 
			//    is_ok(m)は、MOVE_WINではない通常の指し手(トライルールの時の51玉のような指し手)は
			//    その指手を置換表に書き出すという処理。
			//
			//    probe()でttData.move == MOVE_WINのケースを完全に考慮するのは非常に難しい。
			//  　MOVE_WINの値は、置換表に書き出さないほうがいいと思う。
			//
			//    また、置換表にわざと書き込まないことによって、次にこのnodeに訪問したときに
			//    再度入玉判定を行い、枝刈りされることを期待している。

			return bestValue;
		}
		// 1手詰めと宣言勝ちがなかったのでこの時点でもsave()したほうがいいような気がしなくもない。
	}

	// -----------------------
	//     Lazy Evaluator
	// 評価関数の遅延評価子(やねうら王独自拡張)
	// -----------------------

	// この局面で評価関数を呼び出したのか。(do_move()までには呼ばないと駄目)
	Value evaluatedValue = VALUE_NONE;
	auto evaluate = [&](Position& pos)
		{
			if (evaluatedValue == VALUE_NONE)
				evaluatedValue = ::evaluate(pos);
			return evaluatedValue;
		};
	auto lazy_evaluate = [&](Position& pos) {
#if defined(USE_LAZY_EVALUATE) && defined(USE_DIFF_EVAL)
		// ⇨ 差分計算型の評価関数ではないときは、lazy_evaluateする必要がないので、駒得評価関数のときは何もしない。
		if (evaluatedValue == VALUE_NONE)
		{
			evaluatedValue = VALUE_INFINITE;
			::evaluate_with_no_return(pos);
			//evaluatedValue = ::evaluate(pos);
		}
#endif
		};

	// -----------------------
	// Step 6. Static evaluation of the position
	// Step 6. 局面の静的な評価
	// -----------------------

	Value unadjustedStaticEval = VALUE_NONE;

	if (ss->inCheck)
	{
		// Skip early pruning when in check
		// 王手がかかっているときは、early pruning(早期枝刈り)をスキップする

#if defined(USE_LAZY_EVALUATE)
		// Stockfish 17相当のコード
		ss->staticEval = eval = (ss - 2)->staticEval;
#else
		ss->staticEval = eval = evaluate(pos);
#endif

		improving             = false;
		goto moves_loop;
	}
	else if (excludedMove)
	{
		// Providing the hint that this node's accumulator will be used often
		// brings significant Elo gain (~13 Elo).

		// このノードのアキュムレータが頻繁に使用されることを示唆するヒントを提供することで、
		// かなりのElo向上（約13 Elo）が得られる。

		// Eval::NNUE::hint_common_parent_position(pos, networks[numaAccessToken], refreshTable);
		// TODO : → 今回のNNUEの計算は端折れるのか？

#if defined(USE_LAZY_EVALUATE)
		// Stockfish 17のコード
		unadjustedStaticEval = eval = ss->staticEval;

		// ■ 備考
		//
		//   (excludedMoveがあるということは)singular extensionなので、
		//   ss(search stack)は一つ前の局面から動かしていない。(ss + 1ではなく ssをsearch()に渡している)
		// 
		//   excludedMoveは非合法手扱いのevaluate()があればいいのだが、
		//   そんなものはないので、ss->staticEvalで代用している。
#else

		unadjustedStaticEval = eval = evaluate(pos);
#endif

		// ■ 備考
		// 
		// また、excludedMoveがあるときは、この局面の情報をTTに保存してはならない。
		// (同一局面で異なるexcludedMoveを持つ局面が同じhashkeyを持つので
		// 情報の一貫性がなくなる。)
		//  ⇨ 異なるexcludedMoveに対して異なるhashkeyを持てばいいのだが
		//    例: auto posKey = pos.hash_key() ^ make_key(excludedMove);
		//  (以前のStockfishのバージョンではそのようにしていた)
		//   現在のコードのほうが強いようで、そのコードは取り除かれた。

	}
	else if (ss->ttHit)
	{
		// Never assume anything about values stored in TT
		// TTに格納されている値に関して何も仮定はしない

		unadjustedStaticEval = ttData.eval;

#if defined(USE_LAZY_EVALUATE)
		// Stockfish 17のコード
		if (unadjustedStaticEval == VALUE_NONE)
		{
			// TT raceで破壊されてるっぽい？

			// unadjustedStaticEval = evaluate(networks[numaAccessToken], pos, refreshTable, thisThread->optimism[us]);
			unadjustedStaticEval = evaluate(pos);

			// 置換表にhitしたなら、評価値が記録されているはずだから、それを取り出しておく。
			// あとで置換表に書き込むときにこの値を使えるし、各種枝刈りはこの評価値をベースに行なうから。
		}
		else if (PvNode) {
			//		Eval::NNUE::hint_common_parent_position(pos, networks[numaAccessToken], refreshTable);
			// → TODO : hint_common_parent_position()実装するか検討する。

#if defined(YANEURAOU_ENGINE_NNUE)

			//ASSERT(unadjustedStaticEval != VALUE_NONE);
			unadjustedStaticEval = evaluate(pos);

			// TODO : ここでevaluate()が必須な理由がよくわからない。
			//        Stockfishには無いのに…。なぜなのか…。
			// 
			// lazy_evaluate()に変える                        ⇨ 弱い(V8.38dev-n7)
			// unadjustedStaticEvalに代入せずに単にevaluate() ⇨ 特に弱くはなさそう(V838dev_n8)
			// 結論的には、NNUEのevaluate_with_no_return()とevaluate()の挙動が違うのが原因っぽい。
			// これはあきませんわ…。
#endif
		}
#else
		unadjustedStaticEval = evaluate(pos);
#endif

		ss->staticEval = eval =
			// to_corrected_static_eval(unadjustedStaticEval, *thisThread, pos, ss);
			unadjustedStaticEval;

	    // ttValue can be used as a better position evaluation (~7 Elo)
		// ttValue は、より良い局面評価として使用できる（約7 Eloの向上）。

		// ■ 備考
		// 
		// ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。
		// 1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
		//   evalとしてttValueを採用して良い。
		// 2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
		//   evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。

		if (    ttData.value != VALUE_NONE
			&& (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
			eval = ttData.value;

	}
	else
	{
		unadjustedStaticEval =
			// evaluate(networks[numaAccessToken], pos, refreshTable, thisThread->optimism[us]);
			evaluate(pos);
		ss->staticEval = eval =
			// to_corrected_static_eval(unadjustedStaticEval, *thisThread, pos, ss);
			unadjustedStaticEval;

		// Static evaluation is saved as it was before adjustment by correction history
		// 静的評価は、補正履歴による調整が行われる前の状態で保存される。

		// また、excludedMoveがある時は、これを置換表に保存するのは危ない。
		// cf . Add / remove leaves from search tree ttPv : https://github.com/official-stockfish/Stockfish/commit/c02b3a4c7a339d212d5c6f75b3b89c926d33a800
		// 上の方にある else if (excludedMove) でこの条件は除外されている。

		ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
			unadjustedStaticEval, TT.generation());

		// どうせ毎node評価関数を呼び出すので、evalの値にそんなに価値はないのだが、mate_1ply()を
		// 実行したという証にはなるので意味がある。
	}

	// -----------------------
	//   evalベースの枝刈り
	// -----------------------

	// Use static evaluation difference to improve quiet move ordering (~9 Elo)
	// 静的評価の差を利用して、静かな手の順序付けを改善します。（約9 Elo）

	// ■ 補足
	// 
	// 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
	// 王手のときはここにはこない。(上のinCheckのなかでMOVES_LOOPに突入。)

	// is_ok()はMove::null()かのチェック。
	// 1手前でMove::null()ではなく、王手がかかっておらず、駒を取る指し手ではなかったなら…。

	if ((ss - 1)->currentMove.is_ok() && !(ss - 1)->inCheck && !priorCapture)
	{
		int bonus = std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -1831, 1428) + 623;
		// TODO : この右辺の↑係数、調整すべきか？
		thisThread->mainHistory(~us,((ss - 1)->currentMove).from_to()) << bonus;

#if defined(ENABLE_PAWN_HISTORY)
		if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
			thisThread->pawnHistory(pawn_structure_index(pos),pos.piece_on(prevSq),prevSq)
			<< bonus;
#endif
	}

	// Set up the improving flag, which is true if current static evaluation is
	// bigger than the previous static evaluation at our turn (if we were in
	// check at our previous move we go back until we weren't in check) and is
	// false otherwise. The improving flag is used in various pruning heuristics.

	// improvingフラグを設定します。これは、現在の静的評価が前回の自分の手番での
	// 静的評価より大きい場合にtrueとなります（前回の手で王手を受けていた場合、
	// 王手を受けていない局面まで遡って評価します）。
	// それ以外の場合はfalseとなります。このimprovingフラグは、さまざまな枝刈り手法で使用されます。

	// ■ 備考
	// 
	// 評価値が2手前の局面から上がって行っているのかのフラグ(improving)
	// 上がって行っているなら枝刈りを甘くする。
	// ※ VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
	//   枝刈りを甘くして調べないといけないのでimproving扱いとする。

	improving = ss->staticEval > (ss - 2)->staticEval;
	// ※　VALUE_NONE == 32002なのでこれより大きなstaticEvalの値であることはない。

	// ■ 備考
	//
	//   opponentWorseningは、相手の状況が悪化しているかのフラグ。
	// 
	//   ss->staticEval == - (ss-1)->staticEval であるのが普通だが、
	//   左辺のほうが大きい(相手の評価値が悪化している)ならば、
	//   相手の評価値が悪くなっていっていることを意味している。

	opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

	// 1手前のreductionに応じた残りdepthの調整

	if (priorReduction >= 3 && !opponentWorsening)
		depth++;
	if (priorReduction >= 1 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 175)
		depth--;

	// -----------------------
	// Step 7. Razoring (~1 Elo)
	// -----------------------

	// If eval is really low, check with qsearch if we can exceed alpha. If the
	// search suggests we cannot exceed alpha, return a speculative fail low.

	// 評価値が(alphaより)非常に低い場合、qsearchを使ってalphaを超えられるかを確認します。
	// もし探索結果がalphaを超えられないことを示唆する場合は、仮のfail lowを返します。

	if (eval < alpha - 469 - 307 * depth * depth)
	// ↑ここのパラメーター調整しても ~1 Eloなので調整しないことにする。
	{
		value = qsearch<NonPV>(pos, ss, alpha - 1, alpha);
		if (value < alpha && std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY)
			return value;
	}

	// -----------------------
	// Step 8. Futility pruning: child node (~40 Elo)
	// Step 8. futility枝刈り：子ノード（約40 Elo）
	// -----------------------

	// The depth condition is important for mate finding.
	// depthの条件は詰みを発見するために重要である。

	// ■ 補足
	// 
	// このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
	// evalからこれを引いてbetaより大きいなら、beta cutが出来る。
	// ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
	// ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。

	// Stockfish9までは、futility pruningを、root node以外に適用していたが、
	// Stockfish10でnonPVにのみの適用に変更になった。

	if (   !ss->ttPv
		&&  depth < PARAM_FUTILITY_RETURN_DEPTH
		&&  eval - futility_margin(depth, cutNode && !ss->ttHit, improving, opponentWorsening)
				- (ss - 1)->statScore / 290
			>= beta
		&&  eval >= beta
		&& (!ttData.move || ttCapture)
		&&  beta > VALUE_TB_LOSS_IN_MAX_PLY
		&&  eval < VALUE_TB_WIN_IN_MAX_PLY
		// ⇨ 詰み絡みのスコアはmate distance pruningで枝刈りされるはずで、ここでは枝刈りしない。
		)

		// ※　統計値(mainHistoryとかstatScoreとか)のしきい値に関しては、
		// やねうら王ではStockfishから調整しないことにしているので、
		// 上のif式に出てくる定数については調整しないことにする。

		return beta + (eval - beta) / 3;

	// ここでimproving計算しなおす。
	improving |= ss->staticEval >= beta + 100;

	// -----------------------
	// Step 9. Null move search with verification search (~35 Elo)
	// Step 9. 検証探索を伴うnull move探索（約35 Elo）
	// -----------------------

	if (   cutNode
		&& (ss - 1)->currentMove != Move::null()
		&&  eval >= beta
		//  ⇨ evalがbetaを超えているので1手パスしてもbetaは超えそう。だからnull moveを試す
		&&  ss->staticEval >= beta - PARAM_NULL_MOVE_MARGIN1 * depth + PARAM_NULL_MOVE_MARGIN2
		&& !excludedMove
	//	&&  pos.non_pawn_material(us)  // 盤上にpawn以外の駒がある ≒ pawnだけの終盤ではない。将棋でもこれに相当する条件が必要かも。
		&&  ss->ply >= thisThread->nmpMinPly
        &&  beta > VALUE_TB_LOSS_IN_MAX_PLY
		// 同じ手番側に連続してnull moveを適用しない
		)
	{
		ASSERT_LV3(eval - beta >= 0);

		// Null move dynamic reduction based on depth and eval
		// (残り探索)深さと評価値に基づくnull moveの動的なreduction

		Depth R = std::min(int(eval - beta) / PARAM_NULL_MOVE_DYNAMIC_GAMMA, 7) + depth / 3 + 5;

		ss->currentMove         = Move::null();
		// null moveなので、王手はかかっていなくて駒取りでもない。
		// よって、continuationHistory[0(王手かかってない)][0(駒取りではない)][NO_PIECE][SQ_ZERO]
		ss->continuationHistory = &thisThread->continuationHistory[0][0](NO_PIECE, SQ_ZERO);
		//ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[NO_PIECE][0];

		// 王手がかかっている局面では ⇑の方にある goto moves_loop; によってそっちに行ってるので、
		// ここでは現局面で手番側に王手がかかっていない = 直前の指し手(非手番側)は王手ではない ことがわかっている。
		// do_null_move()は、この条件を満たす必要がある。

		lazy_evaluate(pos);
		pos.do_null_move(st);

		Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

		pos.undo_null_move();

		// Do not return unproven mate or TB scores
		// 証明されていないmate scoreやTB scoreはreturnで返さない。

	    if (nullValue >= beta
			&& nullValue < VALUE_TB_WIN_IN_MAX_PLY
			)
		{
			// 1手パスしてもbetaを上回りそうであることがわかったので
			// これをもう少しちゃんと検証しなおす。

	        if (thisThread->nmpMinPly || depth < PARAM_NULL_MOVE_RETURN_DEPTH)
				return nullValue;

			ASSERT_LV3(!thisThread->nmpMinPly); // Recursive verification is not allowed
											    // 再帰的な検証は認めていない。

			// Do verification search at high depths, with null move pruning disabled
			// until ply exceeds nmpMinPly.

			// null move枝刈りを無効化して、plyがnmpMinPlyを超えるまで
			// 高いdepthで検証のための探索を行う。
			thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4 ;

			// nullMoveせずに(現在のnodeと同じ手番で)同じ深さで探索しなおして本当にbetaを超えるか検証する。cutNodeにしない。
			Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

			thisThread->nmpMinPly = 0;

			if (v >= beta)
				return nullValue;
		}
	}

	// -----------------------
	// Step 10. Internal iterative reductions (~9 Elo)
	// Step 10. 内部反復リダクション（約9 Elo）
	// -----------------------

	// For PV nodes without a ttMove, we decrease depth.
	// ttMoveのないPVノードでは、深さを減少させます。

	if (    PvNode
		&& !ttData.move)
		depth -= 3;

	// Use qsearch if depth <= 0
	// (depthをreductionした結果、)もしdepth <= 0ならqsearchを用いる

	if (depth <= 0)
		return qsearch<PV>(pos, ss, alpha, beta);

	// For cutNodes, if depth is high enough, decrease depth by 2 if there is no ttMove,
	// or by 1 if there is a ttMove with an upper bound.

	// カットノードの場合、深さが十分にある場合は、ttMoveがない場合に深さを2減らし、
	// ttMoveが上限値を持つ場合は深さを1減らします。

	if (cutNode && depth >= 7 && (!ttData.move || ttData.bound == BOUND_UPPER))
		depth -= 1 + !ttData.move;

	// -----------------------
	// Step 11. ProbCut (~10 Elo)
	// -----------------------

	// If we have a good enough capture (or queen promotion) and a reduced search
	// returns a value much above beta, we can (almost) safely prune the previous move.

	// 十分に良い駒取り（またはクイーン昇格）があり、
	// (残り探索深さを)削減された探索でbetaを大幅に上回る値が返される場合、
	// 直前の手を（ほぼ）安全に枝刈りできます。

	// probCutに使うbeta値。
	probCutBeta = beta + PARAM_PROBCUT_MARGIN1
		- PARAM_PROBCUT_MARGIN2A * improving
		- PARAM_PROBCUT_MARGIN2B * opponentWorsening;

	if (   !PvNode
		&&  depth > 3
		&&  std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
			
	    // If value from transposition table is lower than probCutBeta, don't attempt probCut
		// there and in further interactions with transposition table cutoff depth is set to depth - 3
		// because probCut search has depth set to depth - 4 but we also do a move before it
		// So effective depth is equal to depth - 3

		// もし置換表から取り出したvalueがprobCutBetaより小さいなら、そこではprobCutを試みず、
		// 置換表との相互作用では、cutoff depthをdepth - 3に設定されます。
		// なぜなら、probCut searchはdepth - 4に設定されていますが、我々はその前に指すので、
		// 実効的な深さはdepth - 3と同じになるからです。

		&& !(  ttData.depth >= depth - 3
			&& ttData.value != VALUE_NONE
			&& ttData.value < probCutBeta))
	{
		ASSERT_LV3(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

		MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &thisThread->captureHistory);
		Piece      captured;

		// 試行回数は2回(cutNodeなら4回)までとする。(よさげな指し手を3つ試して駄目なら駄目という扱い)
		// cf. Do move-count pruning in probcut : https://github.com/official-stockfish/Stockfish/commit/b87308692a434d6725da72bbbb38a38d3cac1d5f
		while ((move = mp.next_move()) != Move::none())
		{
			// ↑Stockfishでは省略してあるけど、この"{"、省略するとbugの原因になりうるので追加しておく。
			ASSERT_LV3(move.is_ok());
			ASSERT_LV5(pos.pseudo_legal(move) && pos.legal_promote(move));

			if (move == excludedMove)
				continue;

			if (!pos.legal(move))
				continue;

			//ASSERT_LV3(pos.capture_stage(move));
			// moveとして歩の成りも返ってくるが、これがcapture_stage()と一致するとは限らない。
			// MovePickerはprob cutの時に、
			// (GenerateAllLegalMovesオプションがオンであっても)歩の成らずは返してこないことを保証すべき。

			movedPiece      = pos.moved_piece_after(move);
			captured        = pos.piece_on(move.to_sq());

			// Prefetch the TT entry for the resulting position
			//prefetch(TT.first_entry(pos.key_after(move)));
			// → 将棋だとこのprefetch、効果がなさげなのでコメントアウト。

			ss->currentMove         = move;
			ss->continuationHistory = &(thisThread->continuationHistory[ss->inCheck][true])
										(movedPiece, move.to_sq());
			//ss->continuationCorrectionHistory =
			//	&this->continuationCorrectionHistory[movedPiece][move.to_sq()];

			// TODO:あとで
			//thisThread->nodes.fetch_add(1, std::memory_order_relaxed);

			lazy_evaluate(pos);
			pos.do_move(move, st);

			// Perform a preliminary qsearch to verify that the move holds
			// この指し手がよさげであることを確認するための予備的なqsearch

			value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

			// If the qsearch held, perform the regular search
			// よさげであったので、普通に探索する

			if (value >= probCutBeta)
				value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, depth - 4, !cutNode);

			pos.undo_move(move);

			if (value >= probCutBeta)
			{
				thisThread->captureHistory(movedPiece, move.to_sq(), type_of(captured)) << 1300;

				// Save ProbCut data into transposition table
				// ProbCutのdataを置換表に保存する。

				ttWriter.write(posKey, value_to_tt(value, ss->ply),
					ss->ttPv, BOUND_LOWER, depth - 3, move, unadjustedStaticEval, TT.generation());

				return std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY
							? value - (probCutBeta - beta)
							: value;
			}

		} // end of while

		//Eval::NNUE::hint_common_parent_position(pos);
		// TODO : あとで検証する。

	}

moves_loop: // When in check, search starts here
	        // 王手がかかっている局面では、探索はここから始まる。

	// -----------------------
	// Step 12. A small Probcut idea, when we are in check (~4 Elo)
	// Step 12. 王手がかかっている局面のときに用いる小さなProbcutのアイデア（約4 Elo）
	// -----------------------

	probCutBeta = beta + PARAM_PROBCUT_MARGIN3;
	if (  (ttData.bound & BOUND_LOWER)
		&& ttData.depth >= depth - 4
		&& ttData.value >= probCutBeta
		&& std::abs(beta)         < VALUE_TB_WIN_IN_MAX_PLY
		&& std::abs(ttData.value) < VALUE_TB_WIN_IN_MAX_PLY
		)
		return probCutBeta;

	// -----------------------
	// moves loopに入る前の準備
	// -----------------------

	// continuationHistory[0]  = Counter Move History    : ある指し手が指されたときの応手
	// continuationHistory[1]  = Follow up Move History  : 2手前の自分の指し手の継続手
	// continuationHistory[3]  = Follow up Move History2 : 4手前からの継続手
	const PieceToHistory* contHist[] = { (ss - 1)->continuationHistory	, (ss - 2)->continuationHistory,
										 (ss - 3)->continuationHistory	, (ss - 4)->continuationHistory ,
											nullptr						, (ss - 6)->continuationHistory };

	MovePicker mp(pos, ttData.move, depth, &thisThread->mainHistory, &thisThread->lowPlyHistory,
				  &thisThread->captureHistory, contHist,
#if defined(ENABLE_PAWN_HISTORY)
										&thisThread->pawnHistory,
#endif
										ss->ply
	);

	value = bestValue;

	// moveCount			: 調べた指し手の数(合法手に限る)
	// moveCountPruning		: moveCountによって枝刈りをするかのフラグ(quietの指し手を生成しない)
	int  moveCount        = 0;

	// -----------------------
	// Step 13. Loop through all pseudo-legal moves until no moves remain
	//			or a beta cutoff occurs.
	// Step 13. 擬似合法手をすべてループし、手がなくなるか
	//          もしくはbetaカットオフが発生するまで繰り返します。
	// -----------------------

	// ■ 備考
	// 
	//    MovePickerが返す指し手はpseudo-legalであることは保証されている。
	//    ※　do_move()までにはlegalかどうかの判定が必要。

	// moveCountPruningがtrueの時はnext_move()はQUIETの指し手を返さないので注意。
	while ((move = mp.next_move()) != Move::none())
	{
		ASSERT_LV3(pos.pseudo_legal(move) && pos.legal_promote(move));

		if (move == excludedMove)
			continue;

		// Check for legality
		// 指し手の合法性のチェック

		// root nodeなら、rootMovesになければlegalではないのでこのチェックは不要だが、
		// root nodeは全体から見ると極わずかなのでそのチェックを端折るほうが良いようだ。

		// 非合法手はほとんど含まれていないから、以前はこの判定はdo_move()の直前まで遅延させたほうが得だったが、
		// do_move()するまでの枝刈りが増えてきたので、ここでやったほうが良いようだ。

		if (!pos.legal(move))
			continue;

		// At root obey the "searchmoves" option and skip moves not listed in Root
		// Move List. In MultiPV mode we also skip PV moves that have been already
		// searched and those of lower "TB rank" if we are in a TB root position.

		// ルートで "searchmoves" オプションに従い、Root Move Listにリストされていない手をスキップします。
		// MultiPVモードでは、既に検索されたPVの手や、TBルート位置にいる場合のTBランクが低い手も
		// スキップします。
		// ※　root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。

		// Stockfishでは、2行目、
		//  thisThread->rootMoves.end() + thisThread->pvLast となっているが
		// 将棋ではこの処理不要なのでやねうら王ではpvLastは使わない。
		if (rootNode && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
									thisThread->rootMoves.end()                      , move))
			continue;

		// do_move()した指し手の数のインクリメント
		ss->moveCount = ++moveCount;

		// Stockfish本家のこの読み筋の出力、細かすぎるので時間をロスする。しないほうがいいと思う。
#if 0
		// 10000000node以上探索しているなら現在探索している指し手をGUIに出力する。
		if (rootNode && is_mainthread() && nodes > 10000000)
		{
			main_manager()->updates.onIter(
				{ depth, UCIEngine::move(move, pos.is_chess960()), moveCount + thisThread->pvIdx });
		}
#endif

		// 次のnodeのpvをクリアしておく。
		if (PvNode)
			(ss + 1)->pv = nullptr;

		// -----------------------
		//      extension
		//      探索の延長
		// -----------------------

		// 今回延長する残り探索深さ
		extension  = 0;

		// 今回捕獲する駒
		capture    = pos.capture_stage(move);

		// 今回移動させる駒(移動後の駒)
		movedPiece = pos.moved_piece_after(move);

		// 今回の指し手で王手になるかどうか
		givesCheck = pos.gives_check(move);

		// Calculate new depth for this move
		// 今回の指し手に関して新しいdepth(残り探索深さ)を計算する。
		newDepth   = depth - 1;

		Value delta = beta - alpha;

		// reduction()では、depthを1024倍した値が返ってきているので注意。
		Depth r = reduction(improving, depth, moveCount, delta, thisThread->rootDelta);

		// -----------------------
		// Step 14. Pruning at shallow depth (~120 Elo).
		// Step 14. (残り探索深さが)浅い深さでの枝刈り（約120 Elo）
		// -----------------------

		// Depth conditions are important for mate finding.
		// (残り探索)深さの条件は詰みを見つける上で重要である

		if (  !rootNode
			// 【計測資料 7.】 浅い深さでの枝刈りを行なうときに王手がかかっていないことを条件に入れる/入れない
		//	&& pos.non_pawn_material(us)  // これに相当する処理、将棋でも必要だと思う。
			&& bestValue > VALUE_TB_LOSS_IN_MAX_PLY
			)
		{
			// Skip quiet moves if movecount exceeds our FutilityMoveCount threshold (~8 Elo)
			// movecountがFutilityMoveCountの閾値を超えた場合、quietの手をスキップします。（約8 Elo）

			if (moveCount >= futility_move_count(improving, depth))
				mp.skip_quiet_moves();

			// Reduced depth of the next LMR search
			// 次のLMR探索における減らさたあとの深さ

			// rは1024倍されているので注意。
			int lmrDepth = newDepth - r / 1024;

			if (capture || givesCheck)
			{
				Piece capturedPiece = pos.piece_on(move.to_sq());
				int   captHist      = thisThread->captureHistory(movedPiece, move.to_sq(), type_of(capturedPiece));

				// Futility pruning for captures (~2 Elo)
				// 駒を取る指し手に対するfutility枝刈り(約2 Elo)

				if (!givesCheck && lmrDepth < 7 && !ss->inCheck)
				{
					// TODO : ここのパラメーター、調整すべきか？ 2 Eloだから無視していいか…。
					int   futilityValue = ss->staticEval
						+ PARAM_FUTILITY_EVAL1
						+ PARAM_FUTILITY_EVAL2 * lmrDepth
						// StockfishのPieceValueは、やねうら王ではCapturePieceValue[]
						// + CapturePieceValue[capturedPiece]
						// ⇨ CapturePieceValuePlusPromoteのほうがより正確な評価ではないか？
						+ CapturePieceValuePlusPromote(pos, move)
						+ captHist / 7;

					if (futilityValue <= alpha)
						continue;
				}

				// SEE based pruning for captures and checks (~11 Elo)
				// 駒取りや王手に対するSEE（静的交換評価）に基づく枝刈り（約11 Elo）

				int seeHist = std::clamp(captHist / 33, -161 * depth, 156 * depth);

				if (!pos.see_ge(move, - PARAM_LMR_SEE_MARGIN1 * depth - seeHist))
					continue;
			}
			else
			{
				int history = (*contHist[0])(movedPiece, move.to_sq())
							+ (*contHist[1])(movedPiece, move.to_sq())
#if defined(ENABLE_PAWN_HISTORY)
							+ thisThread->pawnHistory(pawn_structure(pos), movedPiece, move.to_sq())
#endif
					;

				// Continuation history based pruning (~2 Elo)
				// Continuation historyに基づいた枝刈り(historyの値が悪いものに関してはskip)

				if (history < -3884 * depth)
					continue;

				history += 2 * thisThread->mainHistory(us, move.from_to());

				lmrDepth += history / 3609;

				// 親nodeの時点で子nodeを展開する前にfutilityの対象となりそうなら枝刈りしてしまう。
				// →　パラメーター調整の係数を調整したほうが良いのかも知れないが、
				// 　ここ、そんなに大きなEloを持っていないので、調整しても…。

				Value futilityValue =
					ss->staticEval + (bestValue < ss->staticEval - 45 ? 140 : 43) + 141 * lmrDepth;

				// Futility pruning: parent node (~13 Elo)
				if (!ss->inCheck && lmrDepth < PARAM_FUTILITY_AT_PARENT_NODE_DEPTH && futilityValue <= alpha)
				{
					if (bestValue <= futilityValue
						&& std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY
						&& futilityValue       < VALUE_TB_WIN_IN_MAX_PLY
						)
						bestValue = futilityValue;
					continue;
				}

				// ※　以下のLMRまわり、棋力に極めて重大な影響があるので枝刈りを入れるかどうかを含めて慎重に調整すべき。

				// Prune moves with negative SEE (~3 Elo)
				// 将棋ではseeが負の指し手もそのあと詰むような場合があるから、あまり無碍にも出来ないようだが…。

				// 【計測資料 20.】SEEが負の指し手を枝刈りする/しない

				lmrDepth = std::max(lmrDepth, 0);

				// Prune moves with negative SEE (~4 Elo)
				// 負のSEEを持つ指し手を枝刈りする(約4 Elo)
				// ⇨ lmrDepthの2乗に比例するのでこのパラメーターの影響はすごく大きい。

				if (!pos.see_ge(move, - PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 * lmrDepth * lmrDepth))
					continue;
			}
		}

		// -----------------------
		// Step 15. Extensions (~100 Elo)
		// Step 15. (探索の)延長(約100 Elo)
		// -----------------------

		// We take care to not overdo to avoid search getting stuck.
		// 探索が停滞しないように、やりすぎないよう注意します。

		if (ss->ply < thisThread->rootDepth * 2)
			// rootDepthの2倍より延長しない
		{

			// singular延長と王手延長。

			// Singular extension search (~76 Elo, ~170 nElo). If all moves but one
			// fail low on a search of (alpha-s, beta-s), and just one fails high on
			// (alpha, beta), then that move is singular and should be extended. To
			// verify this we do a reduced search on the position excluding the ttMove
			// and if the result is lower than ttValue minus a margin, then we will
			// extend the ttMove. Recursive singular search is avoided.

			// Note: the depth margin and singularBeta margin are known for having
			// non-linear scaling. Their values are optimized to time controls of
			// 180+1.8 and longer so changing them requires tests at these types of
			// time controls. Generally, higher singularBeta (i.e closer to ttValue)
			// and lower extension margins scale well.

			// シンギュラー延長探索（約76 Elo、約170 nElo）。もしすべての手が
			// (alpha-s, beta-s)の探索で失敗し、ただ1手だけが(alpha, beta)で成功する場合、
			// その手はシンギュラーと見なされ、延長されるべきです。
			// これを検証するために、ttMoveを除いた位置で減らされた深さでの探索を行い、
			// 結果がttValueからマージンを引いた値よりも小さい場合、
			// ttMoveを延長します。再帰的なシンギュラー探索は回避されます。

			// 注意: 深さのマージンやsingularBetaのマージンは非線形なスケーリングがあることが知られています。
			// これらの値は180+1.8以上の時間設定に最適化されており、これらのタイプの時間設定でテストする必要があります。
			// 一般的に、singularBetaが高いほど（つまり、ttValueに近いほど）、
			// 拡張マージンが低いほどスケールが良くなります。


			// ■ メモ
			// 
			//   Stockfishの実装だとmargin = 2 * depthだが、
			//   将棋だと1手以外はすべてそれぐらい悪いことは多々あり、
			//   ほとんどの指し手がsingularと判定されてしまう。
			//   これでは効果がないので、1割ぐらいの指し手がsingularとなるぐらいの係数に調整する。
			//
			// ■ 備考
			//  
			//   singular延長で強くなるのは、あるnodeで1手だけが特別に良い場合、相手のプレイヤーもそのnodeでは
			//   その指し手を選択する可能性が高く、それゆえ、相手のPVもそこである可能性が高いから、そこを相手よりわずかにでも
			//   読んでいて詰みを回避などできるなら、その相手に対する勝率は上がるという理屈。
			//   いわば、0.5手延長が自己対戦で(のみ)強くなるのの拡張。
			//   そう考えるとベストな指し手のスコアと2番目にベストな指し手のスコアとの差に応じて1手延長するのが正しいのだが、
			//   2番目にベストな指し手のスコアを小さなコストで求めることは出来ないので…。
			//

			// singular延長をするnodeであるか。
			if (!rootNode
				&&  move == ttData.move
				&& !excludedMove // 再帰的なsingular延長を除外する。
		        &&  depth >= PARAM_SINGULAR_EXTENSION_DEPTH - (thisThread->completedDepth > 33) + ss->ttPv
			/*  &&  ttValue != VALUE_NONE Already implicit in the next condition */
				&&  std::abs(ttData.value) < VALUE_TB_WIN_IN_MAX_PLY // 詰み絡みのスコアはsingular extensionはしない。(Stockfish 10～)
				&& (ttData.bound & BOUND_LOWER)
				&&  ttData.depth >= depth - 3)
				// このnodeについてある程度調べたことが置換表によって証明されている。(ttMove == moveなのでttMove != Move::none())
				// (そうでないとsingularの指し手以外に他の有望な指し手がないかどうかを調べるために
				// null window searchするときに大きなコストを伴いかねないから。)
			{
				// このmargin値は評価関数の性質に合わせて調整されるべき。
		        Value singularBeta  = ttData.value
					- (PARAM_SINGULAR_MARGIN1
			         + PARAM_SINGULAR_MARGIN2 * (ss->ttPv && !PvNode)) * depth / 64;
				Depth singularDepth = newDepth / 2;

				// move(ttMove)の指し手を以下のsearch()での探索から除外

				ss->excludedMove = move;
				// 局面はdo_move()で進めずにこのnodeから浅い探索深さで探索しなおす。
				// 浅いdepthでnull windowなので、すぐに探索は終わるはず。
				value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
				ss->excludedMove = Move::none();

				// 置換表の指し手以外がすべてfail lowしているならsingular延長確定。
				// (延長され続けるとまずいので何らかの考慮は必要)
				if (value < singularBeta)
				{
					int doubleMargin = 249 * PvNode - 194 * !ttCapture;
					int tripleMargin = 94 + 287 * PvNode - 249 * !ttCapture + 99 * ss->ttPv;

					// We make sure to limit the extensions in some way to avoid a search explosion
					// 2重延長を制限することで探索の組合せ爆発を回避する。

					// TODO : ここのパラメーター、調整すべきかも？
					extension = 1 + (value < singularBeta - doubleMargin)
						      + (value < singularBeta - tripleMargin);

					depth += ((!PvNode) && (depth < 14));
				}

				// Multi-cut pruning
				// Our ttMove is assumed to fail high based on the bound of the TT entry,
				// and if after excluding the ttMove with a reduced search we fail high
				// over the original beta, we assume this expected cut-node is not
				// singular (multiple moves fail high), and we can prune the whole
				// subtree by returning a softbound.

				// マルチカット枝刈り
				// 私たちのttMoveはfail highすると想定されており、
				// 今、ttMoveなしの(この局面でttMoveの指し手を候補手から除外した)、
				// reduced search(探索深さを減らした探索)でもfail highしました。
				// したがって、この予想されるカットノードはsingular(1つだけ傑出した指し手)ではないと想定し、
				// 複数の手がfail highすると考え、softboundを返すことで全サブツリーを枝刈りすることができます。

				// 訳注)
				// 
				//  cut-node  : αβ探索において早期に枝刈りできるnodeのこと。
				//              つまり、searchの引数で渡されたbetaを上回ることがわかったのでreturnできる(これをbeta cutと呼ぶ)
				//              できるようなnodeのこと。
				// 
				//  softbound : lowerbound(下界)やupperbound(上界)のように真の値がその値より大きい(小さい)
				//              ことがわかっているような値のこと。

				else if (value >= beta && std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY)
					return value;

				// Negative extensions
				// If other moves failed high over (ttValue - margin) without the
				// ttMove on a reduced search, but we cannot do multi-cut because
				// (ttValue - margin) is lower than the original beta, we do not know
				// if the ttMove is singular or can do a multi-cut, so we reduce the
				// ttMove in favor of other moves based on some conditions:

				// 負の延長
				// もしttMoveを使用せずに(ttValue - margin)以上で他の手がreduced search(簡略化した探索)で高いスコアを出したが、
				// (ttValue - margin)が元のbetaよりも低いためにマルチカットを行えない場合、
				// ttMoveがsingularかマルチカットが可能かはわからないので、
				// いくつかの条件に基づいて他の手を優先してttMoveを減らします：

				// If the ttMove is assumed to fail high over current beta (~7 Elo)
				// ttMoveが現在のベータを超えて高いスコアを出すと仮定される場合（約7 Elo)

				else if (ttData.value >= beta)
					extension = -3;

				// If we are on a cutNode but the ttMove is not assumed to fail high
				// over current beta (~1 Elo)

				// 現在のノードがカットノードであるが、ttMoveが現在のbetaを超えて
				// fail highすると思われない場合（約1 Elo）

				else if (cutNode)
					extension = -2;
			}

			// Check extensions (~1 Elo)
			// 王手延長

			//  注意 : 王手延長に関して、Stockfishのコード、ここに持ってくる時には気をつけること！
			// →　将棋では王手はわりと続くのでそのまま持ってくるとやりすぎの可能性が高い。
			// 備考) Stockfishで削除されたが、王手延長自体は何らかあった方が良い可能性はあるので条件を調整してはどうか。
			// Remove check extension : https://github.com/official-stockfish/Stockfish/commit/96837bc4396d205536cdaabfc17e4885a48b0588

			// Extension for capturing the previous moved piece (~1 Elo at LTC)
			// 前に動かされた駒を取る場合の延長（LTCで約1 Elo）

			else if (PvNode && move.to_sq() == prevSq
				&& thisThread->captureHistory(movedPiece,move.to_sq(),type_of(pos.piece_on(move.to_sq()))) > 4321)
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

		// -----------------------
		//      1手進める
		// -----------------------

		// この時点で置換表をprefetchする。将棋においては、指し手に駒打ちなどがあって指し手を適用したkeyを
		// 計算するコストがわりとあるので、これをやってもあまり得にはならない。無効にしておく。

		// Speculative prefetch as early as possible
		// 投機的なprefetch
		//prefetch(TT.first_entry(pos.key_after(move)));

		// ⇨　将棋では効果なさそうなので端折る。

		// Update the current move (this must be done after singular extension search)
		// 現在このスレッドで探索している指し手を更新しておく。(これは、singular extension探索のあとになされるべき)
		ss->currentMove = move;
		ss->continuationHistory = &(thisThread->continuationHistory[ss->inCheck]
																   [capture    ])
																   (movedPiece ,
																    move.to_sq());
		//ss->continuationCorrectionHistory =
		//	&thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];
		//uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

		// -----------------------
		// Step 16. Make the move
		// Step 16. 指し手で進める
		// -----------------------

		// 指し手で1手進める
		lazy_evaluate(pos);
		pos.do_move(move, st, givesCheck);

		// These reduction adjustments have proven non-linear scaling.
		// They are optimized to time controls of 180 + 1.8 and longer,
		// so changing them or adding conditions that are similar requires
		// tests at these types of time controls.

		// これらのリダクション調整は、非線形スケーリングであることが証明されています。
		// 調整は180 + 1.8およびそれ以上の持ち時間に最適化されています。
		// したがって、これらを変更したり、似た条件を追加する場合には、
		// 同じ種類の持ち時間でテストが必要です。

		// Decrease reduction if position is or has been on the PV (~7 Elo)
		// 局面がPVにあるか、あった場合、リダクションを減少させます（約7 Elo）

		// この局面がPV上にあり、fail lowしそうであるならreductionを減らす
		// (fail lowしてしまうとまた探索をやりなおさないといけないので)
        if (ss->ttPv)
            r -= 1024 + (ttData.value > alpha) * 1024 + (ttData.depth >= depth) * 1024;

		// Decrease reduction for PvNodes (~0 Elo on STC, ~2 Elo on LTC)
		// PvNode のreductionを減らす (STC では約0 Elo、LTC では約2 Elo)

		if (PvNode)
			r -= 1024;

		// These reduction adjustments have no proven non-linear scaling
		// これらのreductionの調整には、非線形スケーリングの証明はない

		// Increase reduction for cut nodes (~4 Elo)
		// カットノードのreductionを増やす (約4 Elo)

		// cut nodeにおいてhistoryの値が悪い指し手に対してはreduction量を増やす。
		// ※　PVnodeではIID時でもcutNode == trueでは呼ばないことにしたので、
		// if (cutNode)という条件式は暗黙に && !PvNode を含む。

		// 【計測資料 18.】cut nodeのときにreductionを増やすかどうか。

		if (cutNode)
			r += 2518 - (ttData.depth >= depth && ss->ttPv) * 991;

		// Increase reduction if ttMove is a capture but the current move is not a capture (~3 Elo)
		// ttMove が捕獲する指し手で現在の手が捕獲する指し手でない場合、reductionを増やす (約3 Elo)

		if (ttCapture && !capture)
			r += 1043 + (depth < 8) * 999;

		// Increase reduction if next ply has a lot of fail high (~5 Elo)
		// 次の手でfail highが多い場合、reductionを増やします。（約5 Elo）

		if ((ss + 1)->cutoffCnt > 3)
			r += 938 + allNode * 960;

		// For first picked move (ttMove) reduce reduction (~3 Elo)
		// 最初に選ばれた指し手（ttMove）ではreductionを減らします。（約3 Elo）

		else if (move == ttData.move)
			r -= 1879;

		// 【計測資料 11.】statScoreの計算でcontHist[3]も調べるかどうか。
		// contHist[5]も/2とかで入れたほうが良いのでは…。誤差か…？
		if (capture)
			ss->statScore = 0;
		else
			ss->statScore =   2 * thisThread->mainHistory(us, move.from_to())
						+     (*contHist[0])(movedPiece, move.to_sq())
						+     (*contHist[1])(movedPiece, move.to_sq())
						- 3996;
			
		// Decrease/increase reduction for moves with a good/bad history (~8 Elo)
		// 良い/悪い履歴を持つ手に対して、reductionを減らす/増やす (約8 Elo)

		r -= ss->statScore * 1287 / 16384;

		// -----------------------
		// Step 17. Late moves reduction / extension (LMR)
		// Step 17. 遅い指し手の削減／延長（LMR)
		// -----------------------

		// ■ 備考
		//
		//   reduction(削減)とは、depthを減らした探索のこと。
		//   late move(遅い指し手)とは、このnodeでMovePickerで生成されたのが、あとのほうの指し手のこと。
		//   Late Move Reductionは、遅い指し手の削減を意味していて、LMRと略される。

		if (depth >= 2 && moveCount > 1)
		{
			// ■ 備考
			// 
			//   depthを減らして探索させて、その指し手がfail highしたら元のdepthで再度探索する。
			//   moveCountが大きいものなどは探索深さを減らしてざっくり調べる。
			//   alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。

			// In general we want to cap the LMR depth search at newDepth, but when
			// reduction is negative, we allow this move a limited search extension
			// beyond the first move depth. This may lead to hidden multiple extensions.

			// 一般的に、LMRの探索深さをnewDepthで制限したいですが、
			// 削減が負の場合、この手に対して最初の手の深さを超えた
			// 限定的な探索延長を許可します。

			// To prevent problems when the max value is less than the min value,
			// std::clamp has been replaced by a more robust implementation.

			// 最大値が最小値より小さい場合の問題を防ぐために、
			// std::clampはより堅牢な実装に置き換えられています。
			// ⇨ 備考) C++の仕様上、std::clamp(x, min, max)は、min > maxの時に未定義動作であるから、
			//         clamp()を用いるのではなく、max()とmin()を組み合わせて書かないといけない。

			Depth d = std::max(1, std::min(newDepth - r / 1024,
				newDepth + !allNode + (PvNode && !bestMove)))
				+ (ss - 1)->isPvNode;

			value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);

			// Do a full-depth search when reduced LMR search fails high
			// 深さを減らしたLMR探索がfail highをした時、full depth(元の探索深さ)で探索する。

			if (value > alpha && d < newDepth)
			{
				// Adjust full-depth search based on LMR results - if the result was
				// good enough search deeper, if it was bad enough search shallower.
				// LMRの結果に基づいて完全な探索深さを調整します - 
				// 結果が十分に良ければ深く探索し、十分に悪ければ浅く探索します。

				const bool doDeeperSearch     = value > (bestValue + 42 + 2 * newDepth); // (~1 Elo)
				const bool doShallowerSearch  = value <  bestValue + 10;				 // (~2 Elo)

				newDepth += doDeeperSearch - doShallowerSearch;

				if (newDepth > d)
					value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

				// Post LMR continuation history updates (~1 Elo)
				// LMR後のcontinuation historyの更新（約1 Elo）

				int bonus = 2 * (value >= beta) * stat_bonus(newDepth);
				update_continuation_histories(ss, movedPiece, move.to_sq(), bonus);
			}
		}

		// -----------------------
		// Step 18. Full-depth search when LMR is skipped
		// Step 18. LMRがスキップされた場合の完全な探索
		// -----------------------

		else if (!PvNode || moveCount > 1)
		{

            // Increase reduction if ttMove is not present (~1 Elo)
			// ttMoveが存在しない場合、削減を増やします。（約1 Elo）

            if (!ttData.move)
				r += 2037;

			// Note that if expected reduction is high, we reduce search depth by 1 here (~9 Elo)
			// 期待される削減が大きい場合、ここで探索深さを1減らすことに注意してください。（約9 Elo）

			value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth - (r > 2983), !cutNode);
		}

		// For PV nodes only, do a full PV search on the first move or after a fail high,
		// otherwise let the parent node fail low with value <= alpha and try another move.

		// PVノードの場合のみ、最初の手やfail highの後に完全なPV探索を行います。
		// それ以外の場合、親ノードが value <= alpha でfail lowし、別の手を試みます。

		// ■ 備考
		// 
		// PV nodeにおいては、full depth searchがfail highしたならPV nodeとしてsearchしなおす。
		// ただし、value >= betaなら、正確な値を求めることにはあまり意味がないので、これはせずにbeta cutしてしまう。

		if (PvNode && (moveCount == 1 || value > alpha))
		{
			// 次のnodeのPVポインターはこのnodeのpvバッファを指すようにしておく。
			(ss + 1)->pv    = pv;
			(ss + 1)->pv[0] = Move::none();

			// Extend move from transposition table if we are about to dive into qsearch.
			// qsearchに入ろうとしている場合、置換表からの手を延長します。

			if (move == ttData.move && ss->ply <= thisThread->rootDepth * 2)
				newDepth = std::max(newDepth, 1);

			// full depthで探索するときはcutNodeにしてはいけない。
			value = -search<PV>(pos, ss+1, -beta, -alpha, newDepth, false);
		}

		// -----------------------
		// Step 19. Undo move
		// Step 19. 1手戻す
		// -----------------------

		pos.undo_move(move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

		// -----------------------
		// Step 20. Check for a new best move
		// -----------------------

		// Finished searching the move. If a stop occurred, the return value of
		// the search cannot be trusted, and we return immediately without updating
		// best move, principal variation nor transposition table.

		// 指し手の探索が終了しました。もし停止が発生した場合、探索の返り値は信頼できないため、
		// 最善手、主要変化、トランスポジションテーブルを更新せずに直ちに戻ります。

		if (Threads.stop.load(std::memory_order_relaxed))
			return VALUE_ZERO;

		// -----------------------
		//  root node用の特別な処理
		// -----------------------

		if (rootNode)
		{
			RootMove& rm = *std::find(thisThread->rootMoves.begin(),
									  thisThread->rootMoves.end()  , move);

			// TODO : あとで
			//rm.effort += nodes - nodeCount;

			// rootの平均スコアを求める。aspiration searchで用いる。
			rm.averageScore =
				rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

			// 二乗平均スコア。aspiration searchで用いる。
			rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
				? (value * std::abs(value) + rm.meanSquaredScore) / 2
				:  value * std::abs(value);

			// PV move or new best move?
			// PVの指し手か、新しいbest moveか？
			if (moveCount == 1 || value > alpha)
			{
				// root nodeにおいてPVの指し手または、α値を更新した場合、スコアをセットしておく。
				// (iterationの終わりでsortするのでそのときに指し手が入れ替わる。)

		        rm.score =  rm.usiScore = value;
				rm.selDepth             = thisThread->selDepth;
				rm.scoreLowerbound      = rm.scoreUpperbound = false;

				if (value >= beta)
				{
					rm.scoreLowerbound = true;
					rm.usiScore        = beta;
				}
				else if (value <= alpha)
				{
					rm.scoreUpperbound = true;
					rm.usiScore        = alpha;
				}

				rm.pv.resize(1);
				// PVは変化するはずなのでいったんリセット

				// 1手進めたのだから、何らかPVを持っているはずなのだが。
				ASSERT_LV3((ss + 1)->pv);

				// RootでPVが変わるのは稀なのでここがちょっとぐらい重くても問題ない。
				// 新しく変わった指し手の後続のpvをRootMoves::pvにコピーしてくる。
				for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
					rm.pv.push_back(*m);

				// We record how often the best move has been changed in each iteration.
				// This information is used for time management. In MultiPV mode,
				// we must take care to only do this for the first PV line.

				// 各イテレーションで最善手がどのくらいの頻度で変化したかを記録します。
				// この情報は時間管理に使用されます。MultiPV モードでは、
				// 最初の PV ラインに対してのみこれを行うよう注意する必要があります。

				// ■ 備考
				// 
				//   !thisThread->pvIdx という条件を入れておかないとMultiPVで
				//   time managementがおかしくなる。

				if (    moveCount > 1
					&& !thisThread->pvIdx)
					++thisThread->bestMoveChanges;

			} else {

		        // All other moves but the PV, are set to the lowest value: this
				// is not a problem when sorting because the sort is stable and the
				// move position in the list is preserved - just the PV is pushed up.

				// PV以外のすべての手は最低値に設定されます。
				// これはソート時に問題とならないです。
				// なぜなら、ソートは安定しており、リスト内の手の位置は保持されているからです
				// - PVだけが上に押し上げられます。

				// ■ 備考
				// 
				//   root nodeにおいてα値を更新しなかったのであれば、この指し手のスコアを-VALUE_INFINITEにしておく。
				//   こうしておかなければ、stable_sort()しているにもかかわらず、前回の反復深化のときの値との
				//   大小比較してしまい指し手の順番が入れ替わってしまうことによるオーダリング性能の低下がありうる。

				rm.score = -VALUE_INFINITE;
			}
		}

		// -----------------------
		//  alpha値の更新処理
		// -----------------------

		// In case we have an alternative move equal in eval to the current bestmove,
		// promote it to bestmove by pretending it just exceeds alpha (but not beta).

		// 評価値が現在の最善手と等しい代替手がある場合、
		// その手を最善手に昇格させます。この際、α（アルファ）を少しだけ超えるが、
		// β（ベータ）は超えないように見せかけます。

		int inc =
			(value == bestValue && (int(thisThread->nodes) & 15) == 0 && ss->ply + 2 >= thisThread->rootDepth
				&& std::abs(value) + 1 < VALUE_TB_WIN_IN_MAX_PLY);

		if (value + inc > bestValue)
		{
			bestValue = value;

			if (value + inc > alpha)
			{
				bestMove = move;

				// Update pv even in fail-high case
				// fail-highのときにもPVをupdateする。
				if (PvNode && !rootNode)
					update_pv(ss->pv, move, (ss + 1)->pv);

				if (value >= beta)
				{
					ss->cutoffCnt += !ttData.move + (extension < 2);
					ASSERT_LV3(value >= beta); // Fail high

					// value >= beta なら fail high(beta cut)

					// また、non PVであるなら探索窓の幅が0なのでalphaを更新した時点で、value >= betaが言えて、
					// beta cutである。
					break;
				}
				else
				{
					// Reduce other moves if we have found at least one score improvement (~2 Elo)
					// 少なくとも1つのスコアの改善が見られた場合、他の手(の探索深さ)を削減します。

					if (depth > 2 && depth < 14 && std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY)
						depth -= 2;

					ASSERT_LV3(depth > 0);
					alpha = value; // Update alpha! Always alpha < beta

					// alpha値を更新したので更新しておく
					// このとき相手からの詰みがあるかどうかを調べるなどしたほうが良いなら
					// ここに書くべし。
				}
			}
		}

		// If the move is worse than some previously searched move,
		// remember it, to update its stats later.

		// その手が以前に探索された他の手よりも悪い場合、
		// 後でその統計を更新するために記憶しておきます。

		if (move != bestMove && moveCount <= MAX_QUIETS_SEARCHED)
		{
			// 探索した駒を捕獲する指し手
			if (capture)
				capturesSearched.push_back(move);

			// 探索した駒を捕獲しない指し手
			else
				quietsSearched.push_back(move);
		}

	}
	// end of while

	// The following condition would detect a stop only after move loop has been
	// completed. But in this case, bestValue is valid because we have fully
	// searched our subtree, and we can anyhow save the result in TT.
	/*
		if (Threads.stop)
		return VALUE_DRAW;
	*/

	// -----------------------
	// Step 21. Check for mate and stalemate
	// -----------------------

	// All legal moves have been searched and if there are no legal moves, it
	// must be a mate or a stalemate. If we are in a singular extension search then
	// return a fail low score.

	// 詰みとステイルメイトをチェックする。

	// このStockfishのassert、合法手を生成しているので重すぎる。良くない。
	ASSERT_LV5(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    // Adjust best value for fail high cases at non-pv nodes
	// 非PVノードでのfail highの場合に最良値を調整する

    if (!PvNode && bestValue >= beta
		&& std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY
		&& std::abs(beta)      < VALUE_TB_WIN_IN_MAX_PLY
		&& std::abs(alpha)     < VALUE_TB_WIN_IN_MAX_PLY
		)
        bestValue = (bestValue * depth + beta) / (depth + 1);

	// Stockfishでは、ここのコードは以下のようになっているが、これは、
	// 自玉に王手がかかっておらず指し手がない場合は、stalemateで引き分けだから。
	/*
	if (!moveCount)
		bestValue = excludedMove ? alpha :
					 ss->inCheck ? mated_in(ss->ply)
								 : VALUE_DRAW;
	*/

	// ※　⇓ここ⇓、Stockfishのコードをそのままコピペしてこないように注意！

	// (将棋では)合法手がない == 詰まされている なので、rootの局面からの手数で詰まされたという評価値を返す。
	// ただし、singular extension中のときは、ttMoveの指し手が除外されているので単にalphaを返すべき。
	if (!moveCount)
		bestValue = excludedMove ? alpha : mated_in(ss->ply);


	// If there is a move that produces search value greater than alpha we update stats of searched moves
	// // alphaよりも大きな探索値を生み出す手がある場合、探索された手の統計を更新します

	else if (bestMove)

		// quietな(駒を捕獲しない)best moveなのでkillerとhistoryとcountermovesを更新する。

		update_all_stats(pos, ss, bestMove, bestValue, beta, prevSq, quietsSearched,
			capturesSearched, depth);


	// Bonus for prior countermove that caused the fail low
	// fail lowを引き起こした1手前のcountermoveに対するボーナス

	// bestMoveがない == fail lowしているケース。
	// fail lowを引き起こした前nodeでのcounter moveに対してボーナスを加点する。
	// 【計測資料 15.】search()でfail lowしているときにhistoryのupdateを行なう条件

	else if (!priorCapture && prevSq != SQ_NONE)
	{

		int bonus = (117 * (depth > 5) + 39 * !allNode + 168 * ((ss - 1)->moveCount > 8)
			+ 115 * (!ss->inCheck && bestValue <= ss->staticEval - 108)
			+ 119 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 83));

		// Proportional to "how much damage we have to undo"
		// 「元に戻す必要がある損害の大きさ」に比例する

		bonus += std::min(-(ss - 1)->statScore / 113, 300);

		bonus  = std::max(bonus, 0);

		update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
										stat_bonus(depth) * bonus / 93);

		thisThread->mainHistory(~us, (ss - 1)->currentMove.from_to())
			<< stat_bonus(depth) * bonus / 179;

		//if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
		//	thisThread->pawnHistory[pawn_structure_index(pos)][pos.piece_on(prevSq)][prevSq]
		//	<< stat_bonus(depth) * bonus / 24;
	}

	// Bonus when search fails low and there is a TT move
	// 探索がfail lowし、TT moveがある場合のボーナス

	else if (ttData.move && !allNode)
		thisThread->mainHistory(us, ttData.move.from_to()) << stat_bonus(depth) * 23 / 100;

	// ■ 注意
	//
	//   将棋ではtable probe使っていないのでmaxValue関係ない。
	//   ゆえにStockfishのここのコードは不要。(maxValueでcapする必要がない)
	/*
	if (PvNode)
		bestValue = std::min(bestValue, maxValue);
	*/

	// If no good move is found and the previous position was ttPv, then the previous
	// opponent move is probably good and the new position is added to the search tree. (~7 Elo)

	// もし良い指し手が見つからず(bestValueがalphaを更新せず)、前の局面はttPvを選んでいた場合は、
	// 前の相手の手がおそらく良い手であり、新しい局面が探索木に追加される。
	// (ttPvをtrueに変更してTTEntryに保存する)

	if (bestValue <= alpha)
		ss->ttPv = ss->ttPv || ((ss - 1)->ttPv && depth > 3);

	// -----------------------
	//  置換表に保存する
	// -----------------------

	// Write gathered information in transposition table. Note that the
	// static evaluation is saved as it was before correction history.

	// 収集した情報をトランスポジションテーブルに書き込みます。
	// 静的評価は、修正履歴が適用される前の状態で保存されることに注意してください。

	// ■ 補足
	// 
	// betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
	// すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
	// さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
	// また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
	// ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
	// すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。

	if (!excludedMove && !(rootNode && thisThread->pvIdx))
	{
		ASSERT_LV3(pos.legal_promote(bestMove));
		ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
			  bestValue >= beta  ? BOUND_LOWER
			: PvNode && bestMove ? BOUND_EXACT
			                     : BOUND_UPPER,
			depth, bestMove, unadjustedStaticEval, TT.generation());
	}

	// correction historyは実装しない。
#if 0
	// Adjust correction history
	if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
		&& ((bestValue < ss->staticEval && bestValue < beta)  // negative correction & no fail high
			|| (bestValue > ss->staticEval && bestMove)))     // positive correction & no fail low
	{
		const auto m = (ss - 1)->currentMove;

		auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
			-CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
		thisThread->pawnCorrectionHistory[us][pawn_structure_index<Correction>(pos)]
			<< bonus * 107 / 128;
		thisThread->majorPieceCorrectionHistory[us][major_piece_index(pos)] << bonus * 162 / 128;
		thisThread->minorPieceCorrectionHistory[us][minor_piece_index(pos)] << bonus * 148 / 128;
		thisThread->nonPawnCorrectionHistory[WHITE][us][non_pawn_index<WHITE>(pos)]
			<< bonus * 122 / 128;
		thisThread->nonPawnCorrectionHistory[BLACK][us][non_pawn_index<BLACK>(pos)]
			<< bonus * 185 / 128;

		if (m.is_ok())
			(*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()] << bonus;
	}
#endif

	// qsearch()内の末尾にあるassertの文の説明を読むこと。
	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

	return bestValue;
}

// -----------------------
//      静止探索
// -----------------------

// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves (~155 Elo).

// 静止探索関数です。この関数は、メインの探索関数から深さ0で呼び出されるか、
// 再帰的にさらに深さを減らして呼び出されます。
// 深さが0以下の場合、本来は静的評価のみを使用すべきですが、
// 戦術的な手が静的評価を混乱させることがあります。
// この地平線効果に対抗するため、戦術的な手の静止探索を実装しています。(~155 Elo)

// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search


template <NodeType nodeType>
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
{
	// チェスと異なり将棋では、手駒があるため、王手を無条件で延長するとかなりの長手数、王手が続くことがある。
	// 手駒が複数あると、その組み合わせをすべて延長してしまうことになり、組み合わせ爆発を容易に起こす。
	//
	// この点は、Stockfishを参考にする時に、必ず考慮しなければならない。
	//
	// ここでは、以下の対策をする。
	// 1. qsearch(静止探索)ではcaptures(駒を取る指し手)とchecks(王手の指し手)のみをMovePickerで生成
	// 2. 王手の指し手は、depthがDEPTH_QS_CHECKS(== 0)の時だけ生成。
	// 3. capturesの指し手は、depthがDEPTH_QS_RECAPTURES(== -5)以下なら、直前に駒が移動した升に移動するcaptureの手だけを生成。(取り返す手)
	// 4. captureでも歩損以上の損をする指し手は延長しない。
	// 5. 連続王手の千日手は検出する
	// これらによって、王手ラッシュや連続王手で追い回して千日手(実際は反則負け)に至る手順を排除している。
	//
	// ただし、置換表の指し手に関してはdepth < DEPTH_QS_CHECKS でも王手の指し手が交じるので、
	// 置換表の指し手のみで循環するような場合、探索が終わらなくなる。
	// 
	// そこで、
	// 6. depth < -16 なら、置換表の指し手を無視する
	// のような対策が必要だと思う。
	//  →　引き分け扱いすることにした。
	// 

	// -----------------------
	//     変数宣言
	// -----------------------

	// PV nodeであるか。
	// ※　ここがRoot nodeであることはないので、そのケースは考えなくて良い。
	static_assert(nodeType != Root);
	constexpr bool PvNode = nodeType == PV;

	ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
	ASSERT_LV3(PvNode || (alpha == beta - 1));
	ASSERT_LV3(depth <= 0);

	/*
	// Check if we have an upcoming move that draws by repetition (~1 Elo)
	// 反復による引き分けとなる可能性のある次の手があるかを確認する (約1 Elo)

	if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
	{
		alpha = value_draw(this->nodes);
		if (alpha >= beta)
			return alpha;
	}
	*/

	// ⇨ Stockfishではここで上記のように千日手に突入できるかのチェックがあるようだが
	// 将棋でこれをやっても強くならないので導入しない。
	//
	// ■ 補足
	//
	// Stockfishのコードの原理としては、次の一手で千日手局面に持ち込めるなら、
	// 少なくともこの局面は引き分けであるから、
	// betaが引き分けのスコアより低いならbeta cutできるというもの。

	// PV求める用のbuffer
	// (これnonPVでは不要なので、nonPVでは参照していないの削除される。)
	Move pv[MAX_PLY + 1];

	// make_move()のときに必要
	StateInfo st;

	//ASSERT_ALIGNED(&st, Eval::NNUE::CacheLineSize);

	// この局面のhash key
	HASH_KEY posKey;

	// move				: MovePickerからもらった現在の指し手
	// bestMove			: この局面でのベストな指し手
	Move move, bestMove;

	// bestValue		: best moveに対する探索スコア(alphaとは異なる)
	// value			: 現在のmoveに対する探索スコア
	// futilityBase		: futility pruningの基準となる値
	Value bestValue, value, futilityBase;

	// pvHit			: 置換表から取り出した指し手が、PV nodeでsaveされたものであった。
	// givesCheck		: MovePickerから取り出した指し手で王手になるか
	// capture          : 駒を捕獲する指し手か
	bool pvHit, givesCheck , capture;

	// このnodeで何手目の指し手であるか
	int moveCount;

	// 現局面の手番側のColor
	Color us = pos.side_to_move();

	// -----------------------
	// Step 1. Initialize node
	// Step 1. ノードの初期化
	// -----------------------


	if (PvNode)
	{
		(ss + 1)->pv = pv;
		ss->pv[0]    = Move::none();
	}

	Thread* thisThread = pos.this_thread();

	bestMove           = Move::none();
	ss->inCheck        = pos.checkers();
	moveCount          = 0;

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
	// selDepth情報をGUIに送信するために使用します（selDepthは1からカウントし、plyは0からカウントします）。

	if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

	// -----------------------
	// Step 2. Check for an immediate draw or maximum ply reached
	// Step 2. 即座に引き分けになるか、最大のply(手数)に達していないかを確認します。
	// -----------------------

	// 千日手チェックは、MovePickerでcaptures(駒を取る指し手しか生成しない)なら、
	// 千日手チェックしない方が強いようだ。
	// ただし、MovePickerで、TTの指し手に対してもcapturesであるという制限をかけないと
	// TTの指し手だけで無限ループ(MAX_PLYまで再帰的に探索が進む)になり弱くなるので、注意が必要。

	auto draw_type = pos.is_repetition(ss->ply);
	if (draw_type != REPETITION_NONE)
		return value_from_tt(draw_value(draw_type, us), ss->ply);

	// 16手以内の循環になってないのにqsearchで16手も延長している場合、
	// 置換表の指し手だけで長い循環になっている可能性が高く、
	// これは引き分け扱いにしてしまう。(やねうら王独自改良)
	if (depth <= -16)
		return draw_value(REPETITION_DRAW, pos.side_to_move());

	// 最大手数の到達
	if (ss->ply >= MAX_PLY || pos.game_ply() > Limits.max_game_ply)
		return draw_value(REPETITION_DRAW, pos.side_to_move());

	ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

	// qsearchのTTエントリの置き換えとカットオフの優先順位を決定する

	// Note that unlike regular search, which stores literal depth, in QS we only store the
	// current movegen stage. If in check, we search all evasions and thus store
	// DEPTH_QS_CHECKS. (Evasions may be quiet, and _CHECKS includes quiets.)

	// 置換表に登録するdepthはあまりマイナスの値だとおかしいので、
	// 王手がかかっているときは、DEPTH_QS_CHECKS(=0)、
	// 王手がかかっていないときはDEPTH_QS_NORMAL(-1)とみなす。
	//ttDepth = ss->inCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
	//											      : DEPTH_QS_NORMAL;

	// -----------------------
	// Step 3. Transposition table lookup
	// Step 3. 置換表のlookup
	// -----------------------


	posKey  = pos.hash_key();
	auto [ttHit, ttData, ttWriter] = TT.probe(posKey, pos);

	// Need further processing of the saved data
	// 保存されたデータのさらなる処理が必要です

	ss->ttHit    = ttHit;
	ttData.move  = ttHit ? ttData.move : Move::none();
	ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply /* pos.rule50_count() */) : VALUE_NONE;
	pvHit        = ttHit && ttData.is_pv;

	ASSERT_LV3(pos.legal_promote(ttData.move));

	// At non-PV nodes we check for an early TT cutoff
	// non-PV nodeにおいて、置換表による早期枝刈りをチェックします
	//
	// ■ 備考
	//
	// nonPVでは置換表の指し手で枝刈りする
	// PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)

	if (  !PvNode
		&& ttData.depth >= DEPTH_QS
		&& ttData.value != VALUE_NONE  // Can happen when !ttHit or when access race in probe()
								       // ↑置換表から取り出したときに他スレッドが値を潰している可能性があるのでこのチェックが必要
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))

		//
		// ↑ここは、↓この意味。
		//&& (ttData.value >= beta ? (ttData.bound & BOUND_LOWER)
		//                         : (ttData.bound & BOUND_UPPER)))

		// ■ 備考
		// 
		// ttData.valueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttData.value >= beta超えならbeta cutされる
		// ttData.valueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
		// 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
		// このままこの値でreturnして良い。

		return ttData.value;

	// -----------------------
	//     Lazy Evaluator
	// 評価関数の遅延評価子(やねうら王独自拡張)
	// -----------------------

	// この局面で評価関数を呼び出したのか。(do_move()までには呼ばないと駄目)
	Value evaluatedValue = VALUE_NONE;
	auto evaluate = [&](Position& pos)
		{
			if (evaluatedValue == VALUE_NONE)
				evaluatedValue = ::evaluate(pos);
			return evaluatedValue;
		};
	auto lazy_evaluate = [&](Position& pos) {
#if defined(USE_LAZY_EVALUATE) && defined(USE_DIFF_EVAL)
		if (evaluatedValue == VALUE_NONE)
		{
			evaluatedValue = VALUE_INFINITE;
			::evaluate_with_no_return(pos);
			//evaluatedValue = ::evaluate(pos);
		}
#endif
	};

	// -----------------------
	// Step 4. Static evaluation of the position
	// Step 4. この局面の静止評価
	// -----------------------

	// 置換表に書いてあったevalの値をなるべく壊さずに保存するための変数
	Value unadjustedStaticEval = VALUE_NONE;

	if (ss->inCheck)
	{

		// bestValueはalphaとは違う。
		// 王手がかかっているときは-VALUE_INFINITEを初期値として、すべての指し手を生成してこれを上回るものを探すので
		// alphaとは区別しなければならない。
		bestValue = futilityBase = -VALUE_INFINITE;

#if !defined(USE_LAZY_EVALUATE)
		unadjustedStaticEval = evaluate(pos);
#endif
	} else {

		if (ss->ttHit)
		{
#if defined(USE_LAZY_EVALUATE)
			// Never assume anything about values stored in TT
			// TT（置換表）に保存されている値については、決して何も仮定しないこと。

			// 置換表に評価値が格納されているとは限らないのでその場合は評価関数の呼び出しが必要
			// bestValueの初期値としてこの局面のevaluate()の値を使う。これを上回る指し手があるはずなのだが..

			unadjustedStaticEval = ttData.eval;
			if (unadjustedStaticEval == VALUE_NONE)
				unadjustedStaticEval =
					// evaluate(networks[numaAccessToken], pos, refreshTable, thisThread->optimism[us]);
					evaluate(pos);
#if defined(NNUE_LAZY_EVALUATE) && defined(YANEURAOU_ENGINE_NNUE)
			else if (PvNode) {
				// やねうら王独自
				unadjustedStaticEval = evaluate(pos);
				// ⇨ NNUEだとこれ入れたほうが強い可能性が…。
			}
#endif

			ss->staticEval = bestValue =
				// to_corrected_static_eval(unadjustedStaticEval, *thisThread, pos, ss );
				unadjustedStaticEval;

			// 毎回evaluate()を呼ぶならtte->eval()自体不要なのだが、
			// 置換表の指し手でこのまま枝刈りできるケースがあるから難しい。
			// 評価関数がKPPTより軽ければ、tte->eval()をなくしても良いぐらいなのだが…。
#else
			ss->staticEval = bestValue = unadjustedStaticEval = evaluate(pos);
#endif

			// ttValue can be used as a better position evaluation (~13 Elo)
			// ttValueは、より良い局面評価として使用できます（約13 Elo）

			// ■ 備考
			//
			// 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
			// 精度で探索されたものであるなら、それをbestValueの初期値として使う。
			//
			// ただし、mate valueは変更しない方が良いので、abs(ttValue) < VALUE_TB_WIN_IN_MAX_PLY は、
			// そのための条件。

			// ttValue can be used as a better position evaluation (~13 Elo)
			// ttValue は、より良い局面評価として使用できる（約13 Eloの向上）。

			if (std::abs(ttData.value) < VALUE_TB_WIN_IN_MAX_PLY
				&& (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
				bestValue = ttData.value;

		}
		else
		{
			// -----------------------
			//      一手詰め判定
			// -----------------------

			// 置換表にhitした場合は、すでに詰みを調べたはずなので
			// 置換表にhitしなかったときにのみ調べる。
			// ※ この処理は、やねうら王独自。

			ASSERT_LV3(!ss->inCheck && !ss->ttHit);
			if (PARAM_QSEARCH_MATE1)
			{
				// ■ 備考
				// 
				// ⇨ このqsearch()での1手詰めチェックは、いまのところ、入れたほうが良いようだ。
				//    play_time = b1000 ,  1631 - 55 - 1314(55.38% R37.54) [2016/08/19]
				//    play_time = b6000 ,  538 - 23 - 439(55.07% R35.33) [2016/08/19]
	
				// 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈
				move = Mate::mate_1ply(pos);
				if (move != Move::none())
				{
					bestValue = mate_in(ss->ply + 1);

#if defined(WRITE_QSEARCH_MATE1PLY_TO_TT)
					ttWriter.write(posKey, bestValue , ss->ttPv, BOUND_EXACT,
						DEPTH_QS, move, unadjustedStaticEval, TT.generation());
						// depth - 6 は値の範囲が良くない。DEPTH_QSにすべき。

					// ⇨ 置換表に書き出しても得するかわからなかった。(V7.74taya-t9 vs V7.74taya-t12)
					// ⇨ 再度テスト(V8.36dev-d vs V8.36dev-e)
					// このnodeに再訪問することはまずないだろうから、置換表に保存する価値はない可能性がある。
#endif

					return bestValue;
				}
			}

			// 王手がかかっていないなら置換表の指し手を持ってくる

			// In case of null move search, use previous static eval with a different sign
			// Null Move探索の場合、前回の静的評価を符号を反転させて使用する。

			// ■ 備考
			// 
			//   置換表がhitしなかった場合、bestValueの初期値としてevaluate()を呼び出すしかないが、
			//   NULL_MOVEの場合は前の局面での値を反転させると良い。(手番を考慮しない評価関数であるなら)
			//   NULL_MOVEしているということは王手がかかっていないということであり、前の局面でevaluate()は呼び出しているから
			//   StateInfo.sumは更新されていて、そのあとdo_null_move()ではStateInfoが丸ごとコピーされるから、現在のpos.state().sumは
			//   正しい値のはず。

#if defined(USE_LAZY_EVALUATE)
			unadjustedStaticEval =
				(ss - 1)->currentMove != Move::null()
				//? evaluate(networks[numaAccessToken], pos, refreshTable, thisThread->optimism[us])
				  ? evaluate(pos)
				  : -(ss - 1)->staticEval;
#else
			unadjustedStaticEval = evaluate(pos);
#endif

			// 1手前の局面(相手の手番)において 評価値が 500だとしたら、
			// 自分の手番側から見ると -500 なので、 -(ss - 1)->staticEval はそういう意味。
			// ただ、1手パスしているのに同じ評価値のままなのかという問題はあるが、そこが下限で、
			// そこ以上の指し手があるはずという意味で、bestValueの初期値としてはそうなっている。

			// PARAM_QSEARCH_FORCE_EVALは、null moveでもevaluate()を強制的に呼び出す設定。
			// 評価関数の実行時間・精度によっては、こう書いたほうがいいかもという書き方。
			// 残り探索深さが大きい時は、こっちに切り替えるのはありかも…。
			// どちらが優れているかわからないので、optimizerに任せる。

			ss->staticEval = bestValue =
				// to_corrected_static_eval(unadjustedStaticEval, *thisThread, pos, ss );
				unadjustedStaticEval;

		}

		// Stand pat. Return immediately if static value is at least beta
		// Stand pat。静的評価値が少なくともベータ値に達している場合は直ちに返します

		// ■ 補足
		// 
		//   現在のbestValueは、この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
		//   王手がかかっていないケースにおいては、この時点での静的なevalの値がbetaを上回りそうならこの時点で帰る。

		if (bestValue >= beta)
		{
			// bestValueを少しbetaのほうに寄せる。
			if (std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY)
				bestValue = (bestValue + beta) / 2;

			if (!ss->ttHit)
				ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
					DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval,
					TT.generation());

            return bestValue;
		}

		// 王手がかかっていなくてPvNodeでかつ、bestValueがalphaより大きいならそれをalphaの初期値に使う。
		// 王手がかかっているなら全部の指し手を調べたほうがいい。
		if (bestValue > alpha)
			alpha = bestValue;

		// futilityの基準となる値をbestValueにmargin値を加算したものとして、
		// これを下回るようであれば枝刈りする。
		futilityBase = ss->staticEval + PARAM_FUTILITY_MARGIN_QUIET;

	}

	// -----------------------
	//     1手ずつ調べる
	// -----------------------

	const PieceToHistory* contHist[] = { (ss - 1)->continuationHistory,
		                                 (ss - 2)->continuationHistory };

	// 取り合いの指し手だけ生成する
	// searchから呼び出された場合、直前の指し手がMove::null()であることがありうる。この場合、SQ_NONEを設定する。
	Square prevSq = (ss - 1)->currentMove.is_ok() ? (ss - 1)->currentMove.to_sq() : SQ_NONE;

	// Initialize a MovePicker object for the current position, and prepare to search
	// the moves. We presently use two stages of move generator in quiescence search:
	// captures, or evasions only when in check.

	// 現在の局面に対して MovePicker オブジェクトを初期化し、指し手の探索を準備します。
	// 現在、静止探索では2段階の指し手生成を使用しています：
	// captures(駒を取る指し手)、またはevasions(王手の回避)のみです。

	MovePicker mp(pos, ttData.move, DEPTH_QS, &thisThread->mainHistory, &thisThread->lowPlyHistory,
										&thisThread->captureHistory, contHist
#if defined(ENABLE_PAWN_HISTORY)
										,&thisThread->pawnHistory
#endif
										,ss->ply);

	// -----------------------
	// Step 5. Loop through all pseudo-legal moves until no moves remain or a beta
	// cutoff occurs.
	// Step 5. 疑似合法手をすべてループ処理し、手が残らなくなるか、
	// ベータカットオフが発生するまで続けます。
	// -----------------------

	while ((move = mp.next_move()) != Move::none())
	{
		// MovePickerで生成された指し手はpseudo_legalであるはず。
		ASSERT_LV3(pos.pseudo_legal(move) && pos.legal_promote(move));

		// Check for legality
		// 合法手かどうかのチェック
		// 
		// ■ 備考
		//
		//   指し手の合法性の判定は直前まで遅延させたほうが得だと思われていたのだが
		//   (これが非合法手である可能性はかなり低いので他の判定によりskipされたほうが得)
		//   Stockfish14から、静止探索でも、早い段階でlegal()を呼び出すようになった。

		if (!pos.legal(move))
			continue;

		//  局面を進める前の枝刈り

		givesCheck = pos.gives_check(move);
		capture    = pos.capture_stage(move);

		moveCount++;

		// -----------------------
		// Step 6. Pruning
		// Step 6. 枝刈り
		// -----------------------

		// moveが王手にならない指し手であり、1手前で相手が移動した駒を取り返す指し手でもなく、
		// 今回捕獲されるであろう駒による評価値の上昇分を
		// 加算してもalpha値を超えそうにないならこの指し手は枝刈りしてしまう。

		if (bestValue > VALUE_TB_LOSS_IN_MAX_PLY /* && pos.non_pawn_material(us) */
			)
		{
			// Futility pruning and moveCount pruning (~10 Elo)
			// futility枝刈りとmove countに基づく枝刈り（約10 Eloの向上）

			if (!givesCheck
				&& move.to_sq() != prevSq
				&& futilityBase > VALUE_TB_LOSS_IN_MAX_PLY
				// &&  type_of(move) != PROMOTION
				// この最後の条件、入れたほうがいいのか？
				// →　入れない方が良さげ。(V7.74taya-t50 VS V7.74taya-t51)
				)
			{
				// MoveCountに基づく枝刈り
				if (moveCount > 2)
					continue;

				// moveが成りの指し手なら、その成ることによる価値上昇分もここに乗せたほうが正しい見積りになるはず。
				// 【計測資料 14.】 futility pruningのときにpromoteを考慮するかどうか。
				Value futilityValue = futilityBase + CapturePieceValuePlusPromote(pos, move);

				// ⇨　これ、加算した結果、s16に収まらない可能性があるが、
				//     計算はs32で行って、そのあと、この値を用いないからセーフ。

				// If static eval + value of piece we are going to capture is much lower
				// than alpha we can prune this move. (~2 Elo)

				// 静的評価値とキャプチャする駒の価値を合わせたものがalphaより大幅に低い場合、
				// この手を枝刈りすることができます (約2 Elo)。

				if (futilityValue <= alpha)
				{
					bestValue = std::max(bestValue, futilityValue);
					continue;
				}

				// ⇨ ここ、わりと棋力に影響する。下手なことするとR30ぐらい変わる。

				// If static exchange evaluation is low enough
				// we can prune this move. (~2 Elo)
				// 静的交換評価が十分に低い場合、
				// この手を枝刈りできます。（約2 Elo）

				// ■ 備考
				//
				//   futilityBaseはこの局面のevalにmargin値を加算しているのだが、
				//   それがalphaを超えないのは、悪い手だろうから枝刈りしてしまう。

				if (!pos.see_ge(move, alpha - futilityBase))
				{
					bestValue = std::min(alpha, futilityBase);
					continue;
				}
			}


			// Continuation history based pruning (~3 Elo)
			// 継続履歴に基づく枝刈り (約3 Elo)
			// ※ Stockfish12でqsearch()にも導入された。

			// ■ 備考
			// 
			//   駒を取らない王手回避の指し手はよろしくない可能性が高いのでこれは枝刈りしてしまう。
			//   成りでない && seeが負の指し手はNG。王手回避でなくとも、同様。

			if (!capture
				&&    (*contHist[0])(pos.moved_piece_after(move), move.to_sq())
					+ (*contHist[1])(pos.moved_piece_after(move), move.to_sq())
				/*
					+ thisThread->pawnHistory[pawn_structure_index(pos)][pos.moved_piece(move)]
					[move.to_sq()]
				*/
				<= 5095 /* TODO : ここ、調整すべき */)
				continue;

			// Do not search moves with bad enough SEE values (~5 Elo)
			// SEEが十分悪い指し手は探索しない。

			// →　無駄な王手ラッシュみたいなのを抑制できる？
			// これ-90だとPawnValue == 90なので歩損は許してしまう。
			//   歩損を許さないように +1 して、歩損する指し手は延長しないようにするほうがいいか？
			//  →　 captureの時の歩損は、歩で取る、同角、同角みたいな局面なのでそこにはあまり意味なさげ。

			if (!pos.see_ge(move, -PARAM_BAD_ENOUGH_SEE_VALUE))
				continue;
		}

		// TODO : prefetchは、入れると遅くなりそうだが、many coreだと違うかも。
		// Speculative prefetch as early as possible
		//prefetch(TT.first_entry(pos.key_after(move)));

	    // Update the current move
		// 現在このスレッドで探索している指し手を保存しておく。

		ss->currentMove = move;

		ss->continuationHistory = &(thisThread->continuationHistory [ss->inCheck                ]
																	[capture                    ])
										                            (pos.moved_piece_after(move),
																	 move.to_sq()               );

		//ss->continuationCorrectionHistory =
		//	&thisThread->continuationCorrectionHistory[pos.moved_piece(move)][move.to_sq()];

		// -----------------------
		// Step 7. Make and search the move
		// Step 7. 指し手で進め探索する
		// -----------------------

		// 1手動かして、再帰的にqsearch()を呼ぶ
		lazy_evaluate(pos);
		pos.do_move(move, st, givesCheck);
		value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha, depth - 1);
		pos.undo_move(move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

		// Step 8. Check for a new best move
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

				if (value < beta) // Update alpha here!
					// alpha値の更新はこのタイミングで良い。
					// なぜなら、このタイミング以外だと枝刈りされるから。(else以下を読むこと)
					alpha = value;

				else
					break; // Fail high
			}
		}
	}

	// -----------------------
	// Step 9. Check for mate
	// Step 9. 詰みの確認
	// -----------------------

	// All legal moves have been searched. A special case: if we are
	// in check and no legal moves were found, it is checkmate.

	// すべての合法手を探索しました。特別なケース: もし現在王手を受けていて、
	// かつ合法手が見つからなかった場合、それはチェックメイト（詰み）です。

	// ■ 補足
	//
	//   王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである。
	//   どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
	//   わかるわけだし、そもそもこのnodeが詰みだとわかるとこのnodeに再訪問する確率は極めて低く、
	//   置換表に保存しても置換表を汚すだけでほとんど得をしない。(レアケースなのでほとんど損もしないが)
		 
	//   ※　計測したところ、置換表に保存したほうがわずかに強かったが、有意差ではなさげだし、
	//   Stockfish10のコードが保存しないコードになっているので保存しないことにする。

	//   【計測資料 26.】 qsearchで詰みのときに置換表に保存する/しない。

	//   チェスでは王手されていて、合法手がない時に詰みだが、将棋では、合法手がなければ詰みなので ss->inCheckの条件は不要かと思ったら、
	//   qsearch()で王手されていない時は、captures(駒を捕獲する指し手)とchecks(王手の指し手)の指し手しか生成していないから、
	//   moveCount==0だから詰みとは限らない。
	// 
	//   王手されている局面なら、evasion(王手回避手)を生成するから、moveCount==0なら詰みと確定する。
	//   しかし置換表にhitした時にはbestValueの初期値は -VALUE_INFINITEではないので、そう考えると
	//   ここは(Stockfishのコードのように)bestValue == -VALUE_INFINITEとするのではなくmoveCount == 0としたほうが良いように思うのだが…。
	//   →　置換表にhitしたのに枝刈りがなされていない時点で有効手があるわけで詰みではないことは言えるのか…。
	//   cf. https://yaneuraou.yaneu.com/2022/04/22/yaneuraous-qsearch-is-buggy/

	// ■ 補足
	// 
	//   if (ss->inCheck && bestValue == -VALUE_INFINITE)
	//     ↑Stockfishのコード。↓こう変更したほうが良いように思うが計測してみると大差ない。
	//   Stockfishも12年前は↑ではなく↓この書き方だったようだ。moveCountが除去された時に変更されてしまったようだ。
	//    cf. https://github.com/official-stockfish/Stockfish/commit/452f0d16966e0ec48385442362c94a810feaacd9
	//   moveCountが再度導入されたからには、Stockfishもここは、↓の書き方に戻したほうが良いと思う。

	if (ss->inCheck && moveCount == 0)
	{
		// 合法手は存在しないはずだから指し手生成してもすぐに終わるはず。
		ASSERT_LV5(!MoveList<LEGAL>(pos).size());

		return mated_in(ss->ply); // Plies to mate from the root
								  // rootから詰みまでの手数。
	}

	if (std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY && bestValue >= beta)
		bestValue = (3 * bestValue + beta) / 4;

	// Save gathered info in transposition table. The static evaluation
	// is saved as it was before adjustment by correction history.

	// 収集した情報をトランスポジションテーブルに保存します。
	// 静的評価は、修正履歴による調整が行われる前の状態で保存されます。

	// ■ 補足
	// 
	//   詰みではなかったのでこれを書き出す。
	//   ※　qsearch()の結果は信用ならないのでBOUND_EXACTで書き出すことはない。

	ASSERT_LV3(pos.legal_promote(bestMove));
	ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
				bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
				unadjustedStaticEval, TT.generation());

	// ■ 備考
	// 
	//   置換表には abs(value) < VALUE_INFINITEの値しか書き込まないし、この関数もこの範囲の値しか返さない。
	//   しかし置換表が衝突した場合はそうではない。3手詰めの局面で、置換表衝突により1手詰めのスコアが
	//   返ってきた場合がそれである。
	//
	//   ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));
	//   ⇨ このnodeはrootからss->ply手進めた局面なのでここでss->plyより短い詰みがあるのはおかしいが、
	//   この関数はそんな値を返してしまう。しかしこれは通常探索ならば次のnodeでの
	//   mate distance pruningで補正されるので問題ない。
	//   また、VALUE_INFINITEはint16_tの最大値よりMAX_PLY以上小さいなのでオーバーフローの心配はない。
	//
	//   よってsearch(),qsearch()のassertは次のように書くべきである。

	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

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

// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.

// value_to_tt() の逆操作です。これはトランスポジションテーブルからの詰みや
// TB（テーブルベース）のスコアを調整します（これらは現在の局面から詰みまたは
// 詰まされるまでの手数を指します）。
// しかし、50手ルールや履歴の相互作用による誤った詰みやTBスコアを回避するために、
// 代わりにTB以外で最も高いスコアを返します。

// value_to_tt()の逆関数
// ply : root node からの手数。(ply_from_root)
Value value_from_tt(Value v, int ply) {

	if (v == VALUE_NONE)
		return VALUE_NONE;

	// handle TB win or better
	if (v >= VALUE_TB_WIN_IN_MAX_PLY)
	{
		/*
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
		*/

		return v - ply;
	}

	// handle TB loss or worse
	if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
	{
		/*
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
		*/

		return v + ply;
	}

	return v;
}

// update_pv() adds current move and appends child pv[]

// PV lineをコピーする。
// pv に move(1手) + childPv(複数手,末尾Move::none())をコピーする。
// 番兵として末尾はMove::none()にすることになっている。
void update_pv(Move* pv, Move move, const Move* childPv) {

	for (*pv++ = move; childPv && *childPv != Move::none(); )
		*pv++ = *childPv++;
	*pv = Move::none();
}

// -----------------------
//     Statsのupdate
// -----------------------

// update_all_stats() updates stats at the end of search() when a bestMove is found

// update_all_stats()は、bestmoveが見つかったときにそのnodeの探索の終端で呼び出される。
// 統計情報一式を更新する。
// prevSq : 直前の指し手の駒の移動先。直前の指し手がMove::none()の時はSQ_NONE

void update_all_stats(
		const Position& pos,
		Stack* ss,
		Move bestMove,
		Value bestValue,
		Value beta,
		Square prevSq,
		ValueList<Move, MAX_QUIETS_SEARCHED>& quietsSearched,
		ValueList<Move, MAX_QUIETS_SEARCHED>& capturesSearched,
		Depth depth) {

	Color   us           = pos.side_to_move();
	Thread& workerThread = *pos.this_thread();
	CapturePieceToHistory& captureHistory = workerThread.captureHistory;
	Piece moved_piece  = pos.moved_piece_after(bestMove);
	PieceType captured;

	int bonus = stat_bonus(depth + 1);
	int malus = stat_malus(depth    );

	// Stockfish 14ではcapture_or_promotion()からcapture()に変更された。[2022/3/23]
	// Stockfish 16では、capture()からcapture_stage()に変更された。[2023/10/15]
	if (!pos.capture_stage(bestMove))
	{
		update_quiet_histories(pos, ss, workerThread, bestMove, bonus);

		// Decrease stats for all non-best quiet moves
		// 最善でないquietの指し手すべての統計を減少させる

		for (Move move : quietsSearched)
			update_quiet_histories(pos, ss, workerThread, move, -malus);
	}
	else {
		// Increase stats for the best move in case it was a capture move
		// 最善手が捕獲する指し手だった場合、その統計を増加させる

		captured = type_of(pos.piece_on(bestMove.to_sq()));
		captureHistory(moved_piece, bestMove.to_sq(), captured) << bonus;
	}

	// Extra penalty for a quiet early move that was not a TT move in
	// previous ply when it gets refuted.

	// quietな初期の手が、前の手でトランスポジションテーブル（TT）の手ではなく、
	// かつ反証された場合に追加のペナルティを与えます。

	// ※ (ss-1)->ttHit : 一つ前のnodeで置換表にhitしたか

	// Move::null()の場合、Stockfishでは65(移動後の升がSQ_NONEであることを保証している。やねうら王もそう変更した。)
	if (     prevSq != SQ_NONE
		&& ( (ss - 1)->moveCount == 1 + (ss - 1)->ttHit )
		&&   !pos.captured_piece()
		)
		update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus);

	// Decrease stats for all non-best capture moves
	// 最善の捕獲する指し手以外のすべての手の統計を減少させます

	for (Move move : capturesSearched)
	{
		// ここ、moved_piece_before()で、捕獲前の駒の価値で考えたほうがいいか？
		// → MovePickerでcaptureHistoryを用いる時に、moved_piece_afterの方で表引きしてるので、
		//  それに倣う必要がある。
		moved_piece = pos.moved_piece_after(move);
		captured    = type_of(pos.piece_on(move.to_sq()));
		captureHistory(moved_piece, move.to_sq(), captured) << -malus;
	}
}

// update_continuation_histories() updates histories of the move pairs formed
// by moves at ply -1, -2, -3, -4, and -6 with current move.

// update_continuation_histories() は、形成された手のペアの履歴を更新します。
// 1,2,4,6手前の指し手と現在の指し手との指し手ペアによってcontinuationHistoryを更新する。
// 1手前に対する現在の指し手 ≒ counterMove  (応手)
// 2手前に対する現在の指し手 ≒ followupMove (継続手)
// 4手前に対する現在の指し手 ≒ followupMove (継続手)
// ⇨　Stockfish 10で6手前も見るようになった。
// ⇨　Stockfish 16で3手前も見るようになった。
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus)
{
	// これstat_bonusの計算のときに係数を掛けておくべきでは…。
	bonus = bonus * 50 / 64;

	for (int i : {1, 2, 3, 4, 6})
	//for (int i : {1, 2, 4, 6})
	// ⇨　TODO : 3手前も更新するの、強くなってない気がする。(V7.74taya-t20 vs V7.74taya-t21)
	{
	    // Only update the first 2 continuation histories if we are in check
		// 王手がかかっている場合のみ、最初の2つのcontinuation historiesを更新する

		if (ss->inCheck && i > 2)
			break;
		if ((ss - i)->currentMove.is_ok())
			(*(ss - i)->continuationHistory)(pc, to) << bonus / (1 + (i == 3));
						// ⇨　 i==3は弱くしか影響しないので2で割ると言う意味。
	}
}

// Updates move sorting heuristics
// 手のソートのヒューリスティックを更新します

// ⇨ 新しいbest moveが見つかったときに指し手の並べ替えheuristicsを更新する。
// 具体的には駒を取らない指し手のstat tables等を更新する。

// move      = これが良かった指し手

void update_quiet_histories(
	const Position& pos, Stack* ss, /*Search::Worker& workerThread*/ Thread& workerThread,  Move move, int bonus) {

	Color us = pos.side_to_move();
	workerThread.mainHistory(us, move.from_to()) << bonus;
	if (ss->ply < LOW_PLY_HISTORY_SIZE)
		workerThread.lowPlyHistory(ss->ply, move.from_to()) << bonus;

	update_continuation_histories(ss, pos.moved_piece_after(move), move.to_sq(), bonus);

#if defined(ENABLE_PAWN_HISTORY)
	int pIndex = pawn_structure_index(pos);
	workerThread.pawnHistory[pIndex][pos.moved_piece(move)][move.to_sq()] << bonus / 2;
#endif
}

#if 0
// When playing with strength handicap, choose the best move among a set of RootMoves
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
#endif

} // namespace

// MainThread::check_time() is used to print debug info and, more importantly,
// to detect when we are out of available time and thus stop the search.

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
	// main threadでしか判定しないからチェックに要するコストは微小だと思われる。

	// When using nodes, ensure checking rate is not lower than 0.1% of nodes
	// → NodesLimitを有効にした時、その指定されたノード数の0.1%程度の誤差であって欲しいのでそれくらいの頻度でチェックする。

	callsCnt = Limits.nodes ? std::min(512, int(Limits.nodes / 1024)) : 512;

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
		|| (Limits.nodes && Threads.nodes_searched() >= uint64_t(Limits.nodes))
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

// --------------------
//    読み筋の出力
// --------------------

namespace Search {

	// depth : iteration深さ
	std::string pv(const Position& pos, const TranspositionTable& tt, Depth depth)
	{
		std::stringstream ss;

		TimePoint elapsed = Time.elapsed() + 1;
#if defined(__EMSCRIPTEN__)
		// yaneuraou.wasm
		// Time.elapsed()が-1を返すことがある
		// https://github.com/lichess-org/stockfish.wasm/issues/5
		// https://github.com/lichess-org/stockfish.wasm/commit/4f591186650ab9729705dc01dec1b2d099cd5e29
		elapsed = std::max(elapsed, TimePoint(1));
#endif
		const auto nodes_searched = Threads.nodes_searched();
		const auto& rootMoves     = pos.this_thread()->rootMoves;
		size_t pvIdx              = pos.this_thread()->pvIdx;
		size_t multiPV            = std::min(size_t(Options["MultiPV"]), rootMoves.size());

		// MultiPVでは上位N個の候補手と読み筋を出力する必要がある。
		for (size_t i = 0; i < multiPV; ++i)
		{
			// この指し手のpvの更新が終わっているのか
			bool updated = rootMoves[i].score != -VALUE_INFINITE;

			if (depth == 1 && !updated && i > 0)
				continue;

			// 1より小さな探索depthで出力しない。
			Depth d = updated ? depth : std::max(1, depth - 1);
			Value v = updated ? rootMoves[i].usiScore : rootMoves[i].previousScore;

			// multi pv時、例えば3個目の候補手までしか評価が終わっていなくて(PVIdx==2)、このとき、
			// 3,4,5個目にあるのは前回のiterationまでずっと評価されていなかった指し手であるような場合に、
			// これらのpreviousScoreが-VALUE_INFINITE(未初期化状態)でありうる。
			// (multi pv状態で"go infinite"～"stop"を繰り返すとこの現象が発生する。おそらく置換表にhitしまくる結果ではないかと思う。)
			if (v == -VALUE_INFINITE)
				v = VALUE_ZERO; // この場合でもとりあえず出力は行う。

			//bool tb = worker.tbConfig.rootInTB && std::abs(v) <= VALUE_TB;
			//v = tb ? rootMoves[i].tbScore : v;

			bool isExact = i != pvIdx /* || tb */ || !updated;  // tablebase- and previous-scores are exact

			//// Potentially correct and extend the PV, and in exceptional cases v
			//  if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
			//	    && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
			//	    syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);


			if (ss.rdbuf()->in_avail()) // 1行目でないなら連結のための改行を出力
				ss << std::endl;

			ss << "info"
				<< " depth " << d
				<< " seldepth " << rootMoves[i].selDepth
#if defined(USE_PIECE_VALUE)
				<< " score " << USI::value(v)
#endif
				;

			// これが現在探索中の指し手であるなら、それがlowerboundかupperboundかは表示させる
			if (i == pvIdx && /*!tb &&*/ updated) // tablebase- and previous-scores are exact
				ss << (rootMoves[i].scoreLowerbound ? " lowerbound"
					: (rootMoves[i].scoreUpperbound ? " upperbound" : ""));

			// 将棋所はmultipvに対応していないが、とりあえず出力はしておく。
			if (multiPV > 1)
				ss << " multipv " << (i + 1);

			ss << " nodes " << nodes_searched
				<< " nps " << nodes_searched * 1000 / elapsed
				<< " hashfull " << tt.hashfull()
				<< " time " << elapsed
				<< " pv";


			// PV配列からPVを出力する。
			// ※　USIの"info"で読み筋を出力するときは"pv"サブコマンドはサブコマンドの一番最後にしなければならない。

			auto out_array_pv = [&]()
				{
					for (Move m : rootMoves[i].pv)
						ss << " " << m;
				};

			// 置換表からPVをかき集めてきてPVを出力する。
			auto out_tt_pv = [&]()
				{
					auto pos_ = const_cast<Position*>(&pos);
					Move moves[MAX_PLY + 1];
					StateInfo si[MAX_PLY];
					int ply = 0;

					while (ply < MAX_PLY)
					{
						// 千日手はそこで終了。ただし初手はPVを出力。
						// 千日手がベストのとき、置換表を更新していないので
						// 置換表上はMove::none()がベストの指し手になっている可能性があるので早めに検出する。
						auto rep = pos.is_repetition(ply);
						if (rep != REPETITION_NONE && ply >= 1)
						{
							// 千日手でPVを打ち切るときはその旨を表示
							ss << " " << rep;
							break;
						}

						Move m;

						// まず、rootMoves.pvを辿れるところまで辿る。
						// rootMoves[i].pv[0]は宣言勝ちの指し手(Move::win())の可能性があるので注意。
						if (ply < int(rootMoves[i].pv.size()))
							m = rootMoves[i].pv[ply];
						else
						{
							// 次の手を置換表から拾う。
							// ただし置換表を破壊されるとbenchコマンドの時にシングルスレッドなのに探索内容の同一性が保証されなくて
							// 困るのでread_probe()を用いる。

							auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
							// 置換表になかった
							if (!ttHit)
								break;

							m = ttData.move;

							// leaf nodeはわりと高い確率でMove::none()
							if (m == Move::none())
								break;

							// 置換表にはpsudo_legalではない指し手が含まれるのでそれを弾く。
							// 宣言勝ちでないならこれが合法手であるかのチェックが必要。
							if (m != Move::win())
							{
								// 歩の不成が読み筋に含まれていようともそれは表示できなくてはならないので
								// pseudo_legal_s<true>()を用いて判定。
								if (!(pos.pseudo_legal_s<true>(m) && pos.legal(m)))
									break;
							}
						}

#if defined (USE_ENTERING_KING_WIN)
						// 宣言勝ちである
						if (m == Move::win())
						{
							// これが合法手であるなら宣言勝ちであると出力。
							if (pos.DeclarationWin() != Move::none())
								ss << " " << Move::win();

							break;
						}
#endif
						// leaf node末尾にMove::resign()があることはないが、
						// 詰み局面で呼び出されると1手先がmove resignなので、これでdo_move()するのは
						// 非合法だから、do_move()せずにループを抜ける。
						if (!m.is_ok())
						{
							ss << " " << m;
							break;
						}

						moves[ply] = m;
						ss << " " << m;

						// 注)
						// このdo_moveで Position::nodesが加算されるので探索ノード数に影響が出る。
						// benchコマンドで探索ノード数が一致しない場合、これが原因。
						// → benchコマンドでは、ConsiderationMode = falseにすることで
						// 　PV表示のためにdo_move()を呼び出さないようにした。

						pos_->do_move(m, si[ply]);
						++ply;
					}
					while (ply > 0)
						pos_->undo_move(moves[--ply]);
				};

			// 検討用のPVを出力するモードなら、置換表からPVをかき集める。
			// (そうしないとMultiPV時にPVが欠損することがあるようだ)
			// fail-highのときにもPVを更新しているのが問題ではなさそう。
			// Stockfish側の何らかのバグかも。
			if (Search::Limits.consideration_mode)
				out_tt_pv();
			else
				out_array_pv();
		}

		return ss.str();
	}

	// Called in case we have no ponder move before exiting the search,
	// for instance, in case we stop the search during a fail high at root.
	// We try hard to have a ponder move to return to the GUI,
	// otherwise in case of 'ponder on' we have nothing to think about.

	// 探索を終了する前にponder moveがない場合に呼び出されます。
	// 例えば、rootでfail highが発生して探索を中断した場合などです。
	// GUIに返すponder moveをできる限り準備しようとしますが、
	// そうでない場合、「ponder on」の際に考えるべきものが何もなくなります。

	bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos, Move ponder_candidate)
	{
		StateInfo st;

		ASSERT_LV3(pv.size() == 1);

		// Stockfishでは if (pv[0] == Move::none()) となっているが、
		// 詰みの局面が"ponderhit"で返ってくることがあるので、
		// ここでのpv[0] == Move::resign()であることがありうる。
		// だから、やねうら王では、ここは、is_ok()で判定する。

		if (!pv[0].is_ok())
			return false;

		pos.do_move(pv[0], st);

		auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
		if (ttHit)
		{
			Move m = ttData.move;
			//if (MoveList<LEGAL>(pos).contains(ttData.move))
			// ⇨ Stockfishのこのコード、pseudo_legalとlegalで十分なのではないか？
			if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
				pv.push_back(m);
		}
		// 置換表にもなかったので以前のiteration時のpv[1]をほじくり返す。
		else if (ponder_candidate)
		{
			Move m = ponder_candidate;
			if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
				pv.push_back(m);
		}

		pos.undo_move(pv[0]);
		return pv.size() > 1;
	}

}

// ============================================================
//  学習時に用いる、depth固定探索などの関数を外部に対して公開
// ============================================================


#if defined (EVAL_LEARN)

#include "../../testcmd/unit_test.h"
#include "../../position.h"

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
using ValuePV = std::pair<Value, std::vector<Move>>;

// 対局の初期化。
// 置換表のクリアとhistory table等のクリアを行う。
//
// 注意 : この関数の呼び出しは必須ではない。
// 
// Learner::search()での1回の対局(初期局面から詰み局面に至るまで)ごとに
// この関数は呼び出したほうが良い指し手が指せるが、
// このクリア時間が馬鹿にならない。(例えば、depth = 9での対局において対局時間の30%に相当)
//
// この関数を呼び出すなせば、そのバランスを考慮した上で呼び出すこと。
// 
void init_for_game(Position& pos)
{
	auto thisThread = pos.this_thread();

	TT.clear();          // 置換表のクリア
	thisThread->clear(); // history table等のクリア
}

// 静止探索。
//
// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。
// 　また、Threads.stopが来ると探索を中断してしまうので、そのときのPVは正しくない。
// 　search()から戻ったあと、Threads.stop == trueなら、その探索結果を用いてはならない。
// 　あと、呼び出し前は、Threads.stop == falseの状態で呼び出さないと、探索を中断して返ってしまうので注意。
//
// 詰まされている場合は、PV配列にMove::resign()が返る。
//
// 引数でalpha,betaを指定できるようにしていたが、これがその窓で探索したときの結果を
// 置換表に書き込むので、その窓に対して枝刈りが出来るような値が書き込まれて学習のときに
// 悪い影響があるので、窓の範囲を指定できるようにするのをやめることにした。
ValuePV qsearch(Position& pos)
{

	Move pv[MAX_PLY + 1];
	Stack stack[MAX_PLY + 10] = {};
	Stack*ss                  = stack + 7;
	std::vector<Move> pvs;

	// 詰まされているのか
	if (pos.is_mated())
	{
		pvs.push_back(Move::resign());
		return ValuePV(mated_in(/*ss->ply*/ 0 + 1), pvs);
	}

	// 探索の初期化
	init_for_search(pos, ss , pv, /* qsearch = */true);

	auto bestValue = ::qsearch<PV>(pos, ss, -VALUE_INFINITE, VALUE_INFINITE, 0);

	// 得られたPVを返す。
	for (Move* p = &ss->pv[0]; p->is_ok(); ++p)
		pvs.push_back(*p);

	return ValuePV(bestValue, pvs);
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
ValuePV search(Position& pos, int depth_, size_t multiPV /* = 1 */, u64 nodesLimit /* = 0 */)
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

	th->lowPlyHistory.fill(86);

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

			delta     = PARAM_ASPIRATION_SEARCH1 + std::abs(rootMoves[pvIdx].meanSquaredScore) / PARAM_ASPIRATION_SEARCH2;
			Value avg = rootMoves[pvIdx].averageScore;
			alpha     = std::max(avg - delta, -VALUE_INFINITE);
			beta      = std::min(avg + delta,  VALUE_INFINITE);

			// aspiration search
			int failedHighCnt = 0;
			while (true)
			{
				Depth adjustedDepth =
					std::max(1, rootDepth - failedHighCnt);
				th->rootDelta = beta - alpha;
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

	            delta += delta / 3;
				ASSERT_LV3(-VALUE_INFINITE <= alpha && beta <= VALUE_INFINITE);

				// 暴走チェック
				//ASSERT_LV3(th->nodes.load(std::memory_order_relaxed) <= 1000000 );
			}

			std::stable_sort(rootMoves.begin(), rootMoves.begin() + pvIdx + 1);
			//my_stable_sort(pos.this_thread()->thread_id() , &rootMoves[0] , pvIdx + 1);

		} // multi PV

		completedDepth = rootDepth;
	}

	// このPV、途中でMove::none()の可能性があるかも知れないので排除するためにis_ok()を通す。
	// →　PVなのでMove::none()はしないことになっているはずだし、
	//     Move::win()も突っ込まれていることはない。(いまのところ)
	for (Move move : rootMoves[0].pv)
	{
		if (!move.is_ok())
			break;
		pvs.push_back(move);
	}

	//sync_cout << rootDepth << sync_endl;

	// multiPV時を考慮して、rootMoves[0]のscoreをbestValueとして返す。
	bestValue = rootMoves[0].score;

	return ValuePV(bestValue, pvs);
}

// UnitTest : プレイヤー同士の対局
void UnitTest(Test::UnitTester& tester)
{
	// 対局回数→0ならskip
	s64 auto_player_loop = tester.options["auto_player_loop"];
	if (auto_player_loop)
	{
		Position pos;
		StateInfo si;

		// 平手初期化
		auto hirate_init  = [&] { pos.set_hirate(&si, Threads.main()); };
		// 探索深さ
		auto depth = int(tester.options["auto_player_depth"]);

		auto section2 = tester.section("GamesOfAutoPlayer");

		// seed固定乱数(再現性ある乱数)
		PRNG my_rand;
		StateInfo s[512];

		for (s64 i = 0; i < auto_player_loop; ++i)
		{
			// 平手初期化
			hirate_init();
			bool fail = false;

			// 512手目まで
			for (int ply = 0; ply < 512; ++ply)
			{
				MoveList<LEGAL_ALL> ml(pos);

				// 指し手がない == 負け == 終了
				if (ml.size() == 0)
					break;

				// depth 6で探索
				auto r = search(pos, depth);
				//Move m = ml.at(size_t(my_rand.rand(ml.size()))).move;

				pos.do_move(r.second[0],s[ply]);

				if (!pos.pos_is_ok())
					fail = true;
			}

			// 今回のゲームのなかでおかしいものがなかったか
			tester.test(std::string("game ")+std::to_string(i+1),!fail);
		}
	}
}


} // namespace Learner
#endif

#endif // YANEURAOU_ENGINE
