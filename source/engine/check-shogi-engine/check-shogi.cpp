#include "../../shogi.h"

#ifdef CHECK_SHOGI_ENGINE

// -----------------------
//   王手将棋設定部
// -----------------------

// 開発方針
// やねうら王2016(late)からの改造。
// 特徴)
// 王手すると勝ち


// パラメーターを自動調整するのか
// 自動調整が終われば、ファイルを固定してincludeしたほうが良い。
//#define USE_AUTO_TUNE_PARAMETERS

// 探索パラメーターにstep分のランダム値を加えて対戦させるとき用。
// 試合が終わったときに勝敗と、そのときに用いたパラメーター一覧をファイルに出力する。
//#define USE_RANDOM_PARAMETERS

// 試合が終わったときに勝敗と、そのときに用いたパラメーター一覧をファイルに出力する。
// パラメーターのランダム化は行わない。
// USE_RANDOM_PARAMETERSと同時にdefineしてはならない。
//#define ENABLE_OUTPUT_GAME_RESULT


// -----------------------
//   includes
// -----------------------

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include "../../position.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"
#include "../../extra/book.h"
#include "../../move_picker.h"
#include "../../learn/learn.h"

// ハイパーパラメーターを自動調整するときはstatic変数にしておいて変更できるようにする。
#if defined (USE_AUTO_TUNE_PARAMETERS) || defined(USE_RANDOM_PARAMETERS)
#define PARAM_DEFINE static int
#else
#define PARAM_DEFINE constexpr int
#endif

// 実行時に読み込むパラメーターファイルの名前
#define PARAM_FILE "check-shogi-param.h"
#include "check-shogi-param.h"


using namespace std;
using namespace Search;
using namespace Eval;

// 定跡ファイル名
string book_name;

#if defined (USE_RANDOM_PARAMETERS) || defined(ENABLE_OUTPUT_GAME_RESULT)
// 変更したパラメーター一覧と、リザルト(勝敗)を書き出すためのファイルハンドル
static fstream result_log;
#endif

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	// 
	//   定跡設定
	//

	// 実現確率の低い狭い定跡を選択しない
	o["NarrowBook"] << Option(false);

	// 定跡の指し手を何手目まで用いるか
	o["BookMoves"] << Option(16, 0, 10000);


	//  PVの出力の抑制のために前回出力時間からの間隔を指定できる。

	o["PvInterval"] << Option(300, 0, 100000);

	// 定跡ファイル名

	//  no_book          定跡なし
	//  check_shogi_book1.db 王手将棋定跡
	//  user_book1.db    ユーザー定跡1
	//  user_book2.db    ユーザー定跡2
	//  user_book3.db    ユーザー定跡3

	std::vector<std::string> book_list = { "no_book"
		, "check_shogi_book1.db"
		, "user_book1.db", "user_book2.db", "user_book3.db" };
	o["BookFile"] << Option(book_list, book_list[1], [](auto& o) { book_name = string(o); });
	book_name = book_list[1];

	//  BookEvalDiff: 定跡の指し手で1番目の候補の指し手と、2番目以降の候補の指し手との評価値の差が、
	//    この範囲内であれば採用する。(1番目の候補の指し手しか選ばれて欲しくないときは0を指定する)
	//  BookEvalBlackLimit : 定跡の指し手のうち、先手のときの評価値の下限。これより評価値が低くなる指し手は選択しない。
	//  BookEvalWhiteLimit : 同じく後手の下限。
	//  BookDepthLimit : 定跡に登録されている指し手のdepthがこれを下回るなら採用しない。0を指定するとdepth無視。

	o["BookEvalDiff"] << Option(30, 0, 99999);
	o["BookEvalBlackLimit"] << Option(0, -99999, 99999);
	o["BookEvalWhiteLimit"] << Option(-140, -99999, 99999);
	o["BookDepthLimit"] << Option(16, 0, 99999);

	// 定跡をメモリに丸読みしないオプション。(default = false)
	o["BookOnTheFly"] << Option(false);

	// 投了スコア
	o["ResignValue"] << Option(99999, 0, 99999);

	// nodes as timeモード。
	// ミリ秒あたりのノード数を設定する。goコマンドでbtimeが、ここで設定した値に掛け算されたノード数を探索の上限とする。
	// 0を指定すればnodes as timeモードではない。
	// 600knpsなら600を指定する。
	o["nodestime"] << Option(0, 0, 99999);

	//
	//   パラメーターの外部からの自動調整
	//

	o["Param1"] << Option(0, 0, 100000);
	o["Param2"] << Option(0, 0, 100000);

#ifdef EVAL_LEARN
	// 評価関数の学習を行なうときは、評価関数の保存先のフォルダを変更できる。
	// デフォルトではevalsave。このフォルダは事前に用意されているものとする。
	// このフォルダ配下にフォルダを"0/","1/",…のように自動的に掘り、そこに評価関数ファイルを保存する。
	o["EvalSaveDir"] << Option("evalsave");
#endif

#ifdef DISABLE_TT_PROBE
	// 置換表がオフになっている場合、通常対局は出来ないと考えられるので警告を出す。
	sync_cout << "info string warning!! disable TT.probe()." << sync_endl;
#endif

#if defined (USE_RANDOM_PARAMETERS) || defined(ENABLE_OUTPUT_GAME_RESULT)

#ifdef USE_RANDOM_PARAMETERS
	sync_cout << "info string warning!! USE_RANDOM_PARAMETERS." << sync_endl;
#endif

#ifdef ENABLE_OUTPUT_GAME_RESULT
	sync_cout << "info string warning!! ENABLE_OUTPUT_GAME_RESULT." << sync_endl;
#endif

	// パラメーターのログの保存先のfile path
	o["PARAMETERS_LOG_FILE_PATH"] << Option("param_log.txt");
#endif
}

// -----------------------
//   王手将棋探索部
// -----------------------

namespace CheckShogi
{

	// 外部から調整される探索パラメーター
	int param1 = 0;
	int param2 = 0;

	// 定跡等で用いる乱数
	PRNG prng;

	// Ponder用の指し手
	Move ponder_candidate;

	// -----------------------
	//      探索用の定数
	// -----------------------

	// 探索しているnodeの種類
	// Rootはここでは用意しない。Rootに特化した関数を用意するのが少し無駄なので。
	enum NodeType { PV, NonPV };

	// Razoringのdepthに応じたマージン値
	int razor_margin[4];
	
	// 手番の価値
	const Value Tempo = Value(20);

	// depth(残り探索深さ)に応じたfutility margin。
	Value futility_margin(Depth d) {
		return Value(d * PARAM_FUTILITY_MARGIN_ALPHA / ONE_PLY);
	}

	// 残り探索depthが少なくて、王手がかかっていなくて、王手にもならないような指し手を
	// 枝刈りしてしまうためのmoveCountベースのfutilityで用いるテーブル
	// [improving][残りdepth/ONE_PLY]

#if defined (USE_AUTO_TUNE_PARAMETERS) || defined(USE_RANDOM_PARAMETERS)
	// PARAM_PRUNING_BY_MOVE_COUNT_DEPTHの最大値の分だけ余裕を持って確保する。
	int FutilityMoveCounts[2][32];
#else
	// 16のはずだが。
	int FutilityMoveCounts[2][PARAM_PRUNING_BY_MOVE_COUNT_DEPTH];
#endif

	// 探索深さを減らすためのReductionテーブル
	// [PvNodeであるか][improvingであるか][このnodeで何手目の指し手であるか][残りdepth]
	int reduction_table[2][2][64][64];

	// 残り探索深さをこの深さだけ減らす。depthとmove_countに関して63以上は63とみなす。
	// improvingとは、評価値が2手前から上がっているかのフラグ。上がっていないなら
	// 悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
	template <bool PvNode> Depth reduction(bool improving, Depth depth, int move_count) {
		return reduction_table[PvNode][improving][std::min(depth / ONE_PLY, 63)][std::min(move_count, 63)]*ONE_PLY;
	}

	// -----------------------
	//  EasyMoveの判定用
	// -----------------------

	// EasyMoveManagerは、"easy move"を検出するのに用いられる。
	// PVが、複数の探索iterationにおいて安定しているとき、即座にbest moveを返すことが出来る。
	struct EasyMoveManager {

		void clear() {
			stableCnt = 0;
			expectedPosKey = 0;
			pv[0] = pv[1] = pv[2] = MOVE_NONE;
		}

		// 前回の探索からPVで2手進んだ局面であるかを判定するのに用いる。
		Move get(Key key) const {
			return expectedPosKey == key ? pv[2] : MOVE_NONE;
		}

		void update(Position& pos, const std::vector<Move>& newPv) {

			ASSERT_LV3(newPv.size() >= 3);

			// pvの3手目以降が変化がない回数をカウントしておく。
			stableCnt = (newPv[2] == pv[2]) ? stableCnt + 1 : 0;

			if (!std::equal(newPv.begin(), newPv.begin() + 3, pv))
			{
				std::copy(newPv.begin(), newPv.begin() + 3, pv);

				StateInfo st[2];
				pos.do_move(newPv[0], st[0]);
				pos.do_move(newPv[1], st[1]);
				expectedPosKey = pos.state()->key();
				pos.undo_move(newPv[1]);
				pos.undo_move(newPv[0]);
			}
		}

		int stableCnt;
		Key expectedPosKey;
		Move pv[3];
	};

	EasyMoveManager EasyMove;

	// -----------------------
	//  lazy SMPで用いるテーブル
	// -----------------------

	// 各行のうち、半分のbitを1にして、残り半分を1にする。
	// これは探索スレッドごとの反復深化のときのiteration深さを割り当てるのに用いる。

	// 16個のスレッドがあるとして、このスレッドをそれぞれ、
	// depth   : 8個
	// depth+1 : 4個
	// depth+2 : 2個
	// depth+3 : 1個
	// のように先細るように割り当てたい。

	// ゆえに、反復深化のループで
	//   if (table[thread_id][ rootDepth ]) このdepthをskip;
	// のように書けるテーブルがあると都合が良い。

	typedef std::vector<int> Row;
	const Row HalfDensity[] = {
		// 0番目のスレッドはmain threadだとして。
		{ 0, 1 },        // 1番目のスレッド用
		{ 1, 0 },        // 2番目のスレッド用
		{ 0, 0, 1, 1 },  // 3番目のスレッド用
		{ 0, 1, 1, 0 },  //    (以下略)
		{ 1, 1, 0, 0 },
		{ 1, 0, 0, 1 },
		{ 0, 0, 0, 1, 1, 1 },
		{ 0, 0, 1, 1, 1, 0 },
		{ 0, 1, 1, 1, 0, 0 },
		{ 1, 1, 1, 0, 0, 0 },
		{ 1, 1, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 1, 1 },
		{ 0, 0, 0, 0, 1, 1, 1, 1 },
		{ 0, 0, 0, 1, 1, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0 ,0 },
		{ 0, 1, 1, 1, 1, 0, 0 ,0 },
		{ 1, 1, 1, 1, 0, 0, 0 ,0 },
		{ 1, 1, 1, 0, 0, 0, 0 ,1 },
		{ 1, 1, 0, 0, 0, 0, 1 ,1 },
		{ 1, 0, 0, 0, 0, 1, 1 ,1 },
	};

	const size_t HalfDensitySize = std::extent<decltype(HalfDensity)>::value;

	// -----------------------
	//     Statsのupdate
	// -----------------------

	// update_cm_stats()は、countermoveとfollow-up move historyを更新する。
	void update_cm_stats(Stack* ss, Piece pc, Square s, Value bonus)
	{
		CounterMoveStats* cmh  = (ss - 1)->counterMoves;
		CounterMoveStats* fmh1 = (ss - 2)->counterMoves;
		CounterMoveStats* fmh2 = (ss - 4)->counterMoves;

		if (cmh)
			cmh->update(pc, s, bonus);

		if (fmh1)
			fmh1->update(pc, s, bonus);

		if (fmh2)
			fmh2->update(pc, s, bonus);
	}

	// いい探索結果だったときにkiller等を更新する

	// move      = これが良かった指し手
	// quiets    = 悪かった指し手(このnodeで生成した指し手)
	// quietsCnt = ↑の数
	inline void update_stats(const Position& pos, Stack* ss, Move move,
				Move* quiets, int quietsCnt, Value bonus)
	{
		//   killerのupdate

		// killer 2本しかないので[0]と違うならいまの[0]を[1]に降格させて[0]と差し替え
		if (ss->killers[0] != move)
		{
			ss->killers[1] = ss->killers[0];
			ss->killers[0] = move;
		}

		//   historyのupdate
		Color c = pos.side_to_move();

		auto thisThread = pos.this_thread();

		Square mto = to_sq(move);
		Piece mpc = pos.moved_piece_after(move);

		thisThread->fromTo.update(c, move, bonus);
		thisThread->history.update(mpc, mto, bonus);

		update_cm_stats(ss, mpc, mto, bonus);

		if ((ss - 1)->counterMoves)
		{
			// 直前に移動させた升(その升に移動させた駒がある。今回の指し手はcaptureではないはずなので)
			Square prevSq = to_sq((ss - 1)->currentMove);

			// Moveのなかに移動後の駒が格納されているからそれを取り出すだけ。
			Piece prevPc = pos.moved_piece_after((ss - 1)->currentMove);

			thisThread->counterMoves.update(prevPc, prevSq, move);
		}

		// ToDo :
		//   abs(bonus) >= 324 ならupdate()する必要がないので
		//   ここでreturn出来るはずなのだが…。

		// Decrease all the other played quiet moves
		for (int i = 0; i < quietsCnt; ++i)
		{
			thisThread->fromTo.update(c, quiets[i], -bonus);
			Piece qpc = pos.moved_piece_after(quiets[i]);
			thisThread->history.update(qpc, to_sq(quiets[i]), -bonus);
			update_cm_stats(ss, qpc, to_sq(quiets[i]), -bonus);
		}
	}

	// 残り時間をチェックして、時間になっていればSignals.stopをtrueにする。
	void check_time()
	{
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
		if (Limits.ponder)
			return;

		// "ponderhit"時は、そこからの経過時間で考えないと、elapsed > Time.maximum()になってしまう。
		// elapsed_from_ponderhit()は、"ponderhit"していないときは"go"コマンドからの経過時間を返すのでちょうど良い。
		int elapsed = Time.elapsed_from_ponderhit();

		// 今回のための思考時間を完璧超えているかの判定。

		// 反復深化のループ内でそろそろ終了して良い頃合いになると、Time.search_endに停止させて欲しい時間が代入される。
		// (それまではTime.search_endはゼロであり、これは終了予定時刻が未確定であることを示している。)
		if ((Limits.use_time_management() &&
			(elapsed > Time.maximum() - 10 || (Time.search_end > 0 && elapsed > Time.search_end - 10)))
			|| (Limits.movetime && elapsed >= Limits.movetime)
			|| (Limits.nodes && Threads.nodes_searched() >= (uint64_t)Limits.nodes))
			Signals.stop = true;
	}

	// -----------------------
	//      静止探索
	// -----------------------

	// search()で残り探索深さが0以下になったときに呼び出される。
	// (より正確に言うなら、残り探索深さがONE_PLY未満になったときに呼び出される)

	// InCheck : 王手がかかっているか
	template <NodeType NT, bool InCheck>
	Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
	{
		// -----------------------
		//     変数宣言
		// -----------------------

		// PV nodeであるか。
		const bool PvNode = NT == PV;

		ASSERT_LV3(InCheck == !!pos.checkers());
		ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
		ASSERT_LV3(PvNode || alpha == beta - 1);
		ASSERT_LV3(depth <= DEPTH_ZERO);
		// depthがONE_PLYの倍数である。
		ASSERT_LV3(depth / ONE_PLY * ONE_PLY == depth);

		// PV求める用のbuffer
		// (これnonPVでは不要なので、nonPVでは参照していないの削除される。)
		Move pv[MAX_PLY + 1];

		// 評価値の最大を求めるために必要なもの
		Value bestValue;       // この局面での指し手のベストなスコア(alphaとは違う)
		Move bestMove;         // そのときの指し手
		Value oldAlpha;        // 関数が呼び出されたときのalpha値
		Value futilityBase;    // futility pruningの基準となる値

		// hash key関係
		TTEntry* tte;          // 置換表にhitしたときの置換表のエントリーへのポインタ
		Key posKey;            // この局面のhash key
		bool ttHit;            // 置換表にhitしたかのフラグ
		Move ttMove;           // 置換表に登録されていた指し手
		Value ttValue;         // 置換表に登録されていたスコア
		Depth ttDepth;         // このnodeに関して置換表に登録するときの残り探索深さ

		// 王手関係
		bool givesCheck;       // MovePickerから取り出した指し手で王手になるか

		// -----------------------
		//     nodeの初期化
		// -----------------------

		// rootからの手数
		ss->ply = (ss - 1)->ply + 1;

		// 王手将棋では王手がかかっているなら詰みであるから、呼び出されていれば詰みのスコアを返す。
		// ss->plyを初期化してからしか、ss->plyを返せないので注意。
		if (InCheck)
			return mated_in(ss->ply);

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

		ss->currentMove = bestMove = MOVE_NONE;

		// -----------------------
		//    最大手数へ到達したか？
		// -----------------------

		if (pos.game_ply() > Limits.max_game_ply)
			return draw_value(REPETITION_DRAW, pos.side_to_move());

		// -----------------------
		//     置換表のprobe
		// -----------------------

		// 置換表に登録するdepthは、あまりマイナスの値が登録されてもおかしいので、
		// 王手がかかっているときは、DEPTH_QS_CHECKS(=0)、王手がかかっていないときはDEPTH_QS_NO_CHECKSの深さとみなす。
		ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
			: DEPTH_QS_NO_CHECKS;

		posKey = pos.key();
#ifndef DISABLE_TT_PROBE 
		tte = TT.probe(posKey, ttHit);
		ttMove = ttHit ? pos.move16_to_move(tte->move()) : MOVE_NONE;
		ttValue = ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;
#else
		tte = nullptr; ttHit = false;
		ttMove = MOVE_NONE;
		ttValue = VALUE_NONE;
#endif

		// nonPVでは置換表の指し手で枝刈りする
		// PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)
		if (!PvNode
			&& ttHit
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

		if (InCheck)
		{
			// 王手がかかっているならすべての指し手を調べるべきなのでevaluate()は呼び出さない。
			ss->staticEval = VALUE_NONE;

			// bestValueはalphaとは違う。
			// 王手がかかっているときは-VALUE_INFINITEを初期値として、すべての指し手を生成してこれを上回るものを探すので
			// alphaとは区別しなければならない。
			bestValue = futilityBase = -VALUE_INFINITE;

		} else {

			// 王手がかかっていないなら置換表の指し手を持ってくる

			if (ttHit)
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
				if (ttValue != VALUE_NONE)
					if (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))
						bestValue = ttValue;

			} else {

				// 置換表がhitしなかった場合、bestValueの初期値としてevaluate()を呼び出すしかないが、
				// NULL_MOVEの場合は前の局面での値を反転させると良い。(手番を考慮しない評価関数であるなら)
				// NULL_MOVEしているということは王手がかかっていないということであり、前の局面でevaluate()は呼び出しているから
				// StateInfo.sumは更新されていて、そのあとdo_null_move()ではStateInfoが丸ごとコピーされるから、現在のpos.state().sumは
				// 正しい値のはず。

#if 0
				ss->staticEval = bestValue =
					(ss - 1)->currentMove != MOVE_NULL ? evaluate(pos)
					: -(ss - 1)->staticEval + 2 * Tempo;
#else
				// 長い持ち時間では、ここ、きちんと評価したほうが良いかも。
				ss->staticEval = bestValue = evaluate(pos);
#endif
			}

			// Stand pat.
			// 現在のbestValueは、この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
			// 王手がかかっていないケースにおいては、この時点での静的なevalの値がbetaを上回りそうならこの時点で帰る。
			if (bestValue >= beta)
			{
#ifndef DISABLE_TT_PROBE
				// Stockfishではここ、pos.key()になっているが、posKeyを使うべき。
				if (!ttHit)
					tte->save(posKey, value_to_tt(bestValue, ss->ply), BOUND_LOWER,
						DEPTH_NONE, MOVE_NONE, ss->staticEval, TT.generation());
#endif

				return bestValue;
			}

			// 王手がかかっていなくてPvNodeでかつ、bestValueがalphaより大きいならそれをalphaの初期値に使う。
			// 王手がかかっているなら全部の指し手を調べたほうがいい。
			if (PvNode && bestValue > alpha)
				alpha = bestValue;

			// futilityの基準となる値をbestValueにmargin値を加算したものとして、
			// これを下回るようであれば枝刈りする。
			futilityBase = bestValue + PARAM_FUTILITY_MARGIN_QUIET;
		}

		// -----------------------
		//     1手ずつ調べる
		// -----------------------

		// 取り合いの指し手だけ生成する
		// searchから呼び出された場合、直前の指し手がMOVE_NULLであることがありうるが、
		// 静止探索の1つ目の深さではrecaptureを生成しないならこれは問題とならない。
		MovePicker mp(pos, ttMove, depth, to_sq((ss - 1)->currentMove));
		Move move;
		Value value;

		StateInfo st;

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

			// 王手将棋ではこれにて詰み
			if (givesCheck)
			{
				if (!pos.legal(move))
					continue;

				// これが合法手なら詰み
				// このnodeに再訪問することはまずないだろうから、置換表に保存する価値はない。

				return mate_in(ss->ply + 1);
			}

			//
			//  Futility pruning
			// 

			// 自玉に王手がかかっていなくて、敵玉に王手にならない指し手であるとき、
			// 今回捕獲されるであろう駒による評価値の上昇分を
			// 加算してもalpha値を超えそうにないならこの指し手は枝刈りしてしまう。

			if (!InCheck
				&& !givesCheck
				&&  futilityBase > -VALUE_KNOWN_WIN)

				// 魔女ではここがVALUE_INFINITEになっている。(StockfishではVALUE_KNOWN_WIN)
				// ここをVALUE_INFINITEにしてしまうと、負けを読みきっているときに
				// その値を基準に枝刈りすることになる。

				// ここをVALUE_INFINITEに変更してもほぼ変わらないようなのでこのままで行く。
				// r300, 4463 - 107 - 4480(49.9% R - 0.66) [2016/08/19]

			{
				// moveが成りの指し手なら、その成ることによる価値上昇分もここに乗せたほうが正しい見積りになるはず。

				// is_promote()以下、ないほうがいい？
				// 			T1,b1000,4947 - 256 - 4797(50.77% R5.35)[2016/09/03]
				//			T1,b3000,2416 - 183 - 2401(50.16% R1.08)[2016/09/04]
				// →　有ったほうが良い…かも…。

				Value futilityValue = futilityBase + (Value)CapturePieceValue[pos.piece_on(to_sq(move))]
					+ (is_promote(move) ? (Value)ProDiffPieceValue[pos.piece_on(move_from(move))] : VALUE_ZERO);

				// futilityValueは今回捕獲するであろう駒の価値の分を上乗せしているのに
				// それでもalpha値を超えないというとってもひどい指し手なので枝刈りする。
				if (futilityValue <= alpha)
				{
					bestValue = std::max(bestValue, futilityValue);
					continue;
				}

				// futilityBaseはこの局面のevalにmargin値を加算しているのだが、それがalphaを超えないし、
				// かつseeがプラスではない指し手なので悪い手だろうから枝刈りしてしまう。

				if (futilityBase <= alpha && !pos.see_ge(move , VALUE_ZERO+1))
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

			// ただし、王手されている局面の場合、王手の回避手を1つ以上見つけていないのに
			// これの指し手を枝刈りしてしまうと回避手がないかのように錯覚してしまうので、
			// bestValue > VALUE_MATED_IN_MAX_PLY
			// (実際は-VALUE_INFINITEより大きければ良い)
			// という条件を追加してある。

			if (
				(!InCheck || (!pos.capture(move) && bestValue > VALUE_MATED_IN_MAX_PLY))
#if 0
				// Stockfish8相当のコード
				&& !is_promote(move)
#else
				// ここ、成る手ではなく、歩が成る手のみを除外(したほうが良い)
				// T1,b1000,1439 - 61 - 1220(54.12% R28.68)
				&& !pos.pawn_promotion(move)
#endif
				&& !pos.see_ge(move , VALUE_ZERO))
				continue;

			// -----------------------
			//     局面を1手進める
			// -----------------------

			// 指し手の合法性の判定は直前まで遅延させたほうが得。
			// (これが非合法手である可能性はかなり低いので他の判定によりskipされたほうが得)
			if (!pos.legal(move))
				continue;

			// 現在このスレッドで探索している指し手を保存しておく。
			ss->currentMove = move;

			pos.do_move(move, st, givesCheck);
			value = givesCheck ? -qsearch<NT, true >(pos, ss + 1, -beta, -alpha, depth - ONE_PLY)
				               : -qsearch<NT, false>(pos, ss + 1, -beta, -alpha, depth - ONE_PLY);

			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// bestValue(≒alpha値)を更新するのか
			if (value > bestValue)
			{
				bestValue = value;

				if (value > alpha)
				{
					// fail-highの場合もPVは更新する。
					if (PvNode)
						update_pv(ss->pv, move, (ss + 1)->pv);

					if (PvNode && value < beta)
					{
						// alpha値の更新はこのタイミングで良い。
						// なぜなら、このタイミング以外だと枝刈りされるから。(else以下を読むこと)
						alpha = value;
						bestMove = move;

					} else // fail high
					{
#ifndef DISABLE_TT_PROBE
						// 1. nonPVでのalpha値の更新 →　もうこの時点でreturnしてしまっていい。(ざっくりした枝刈り)
						// 2. PVでのvalue >= beta、すなわちfail high
						tte->save(posKey, value_to_tt(value, ss->ply), BOUND_LOWER,
							ttDepth, move, ss->staticEval, TT.generation());
#endif
						return value;
					}
				}
			}
		}

		// -----------------------
		// 指し手を調べ終わった
		// -----------------------

		// 王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである。
		// どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
		// わかるわけだし、そもそもこのnodeが詰みだとわかるとこのnodeに再訪問する確率は極めて低く、
		// 置換表に保存しても得しない。
		if (InCheck && bestValue == -VALUE_INFINITE)
		{
			bestValue = mated_in(ss->ply); // rootからの手数による詰みである。

		} else {
			// 詰みではなかったのでこれを書き出す。

#ifndef DISABLE_TT_PROBE
			tte->save(posKey, value_to_tt(bestValue, ss->ply),
				(PvNode && bestValue > oldAlpha) ? BOUND_EXACT : BOUND_UPPER,
				ttDepth, bestMove, ss->staticEval, TT.generation());
#endif
		}

		// 置換表には abs(value) < VALUE_INFINITEの値しか書き込まないし、この関数もこの範囲の値しか返さない。
		// このassertを書くべきだし、Stockfishではこのassertが書かれているが、
		// このnodeはrootからss->ply手進めた局面なのでここでss->plyより短い詰みがあるのはおかしい。
		// この事実を利用したほうが、より厳しいassertが書ける。
		ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));

		return bestValue;
	}


	// -----------------------
	//      通常探索
	// -----------------------

	// cutNode = LMRで悪そうな指し手に対してreduction量を増やすnode
	template <NodeType NT>
	Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
	{
		// -----------------------
		//     nodeの種類
		// -----------------------

		// PV nodeであるか(root nodeはPV nodeに含まれる)
		const bool PvNode = NT == PV;

		// root nodeであるか
		const bool RootNode = PvNode && (ss - 1)->ply == 0;

		// -----------------------
		//     変数の宣言
		// -----------------------

		// このnodeからのPV line(読み筋)
		Move pv[MAX_PLY + 1];

		// do_move()するときに必要
		StateInfo st;

		// MovePickerから1手ずつもらうときの一時変数
		Move move;

		// LMRのときにfail highが起きるなどしたので元の残り探索深さで探索することを示すフラグ
		bool doFullDepthSearch;

		// この局面でのベストのスコア
		Value bestValue = -VALUE_INFINITE;

		// search()の戻り値を受ける一時変数
		Value value;

		// この局面に対する評価値の見積り。
		Value eval;

		// -----------------------
		//     nodeの初期化
		// -----------------------

		ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
		ASSERT_LV3(PvNode || (alpha == beta - 1));
		ASSERT_LV3(DEPTH_ZERO < depth && depth < DEPTH_MAX);
		// IIDを含め、PvNodeではcutNodeで呼び出さない。
		ASSERT_LV3(!(PvNode && cutNode));
		// depthがONE_PLYの倍数であるかのチェック
		ASSERT_LV3(depth / ONE_PLY * ONE_PLY == depth);

		Thread* thisThread = pos.this_thread();

		// ss->moveCountはこのあとMovePickerがこのnodeの指し手を生成するより前に
		// 枝刈り等でsearch()を再帰的に呼び出すことがあり、そのときに親局面のmoveCountベースで
		// 枝刈り等を行ないたいのでこのタイミングで初期化しなければならない。
		// ss->moveCountではなく、moveCountのほうはMovePickerで指し手を生成するとき以降で良い。

		ss->moveCount = 0;

		// rootからの手数
		ss->ply = (ss - 1)->ply + 1;

		// seldepthをGUIに出力するために、PVnodeであるならmaxPlyを更新してやる。
		if (PvNode && thisThread->maxPly < ss->ply)
			thisThread->maxPly = ss->ply;

		// -----------------------
		//  Timerの監視
		// -----------------------

		// タイマースレッドを使うとCPU負荷的にちょっと損なので
		// 自分で呼び出し回数をカウントして一定回数ごとにタイマーのチェックを行なう。

		// いずれかのスレッドが4096カウントしたところで全スレッドのカウントをリセットして
		// check_time()を行う。main threadだけが集計してcheck_time()を呼び出せば良さそうなものだが、
		// そうするとmain threadだけがタイマー監視処理が必要になって、負荷分散上の観点から損であるようだ。

		if (thisThread->resetCalls.load(std::memory_order_relaxed))
		{
			thisThread->resetCalls = false;
			thisThread->callsCnt = 0;
		}
		// nps 1コア時でも600kぐらい出るから、10knodeごとに調べれば0.02秒程度の精度は出るはず。
		if (++thisThread->callsCnt > 4096)
		{
			for (Thread* th : Threads)
				th->resetCalls = true;

			check_time();
		}

		// -----------------------
		//  RootNode以外での処理
		// -----------------------

		if (!RootNode)
		{
			// -----------------------
			//     千日手等の検出
			// -----------------------

			// 連続王手による千日手、および通常の千日手、優等局面・劣等局面。

			// 連続王手による千日手に対してdraw_value()は、詰みのスコアを返すので、rootからの手数を考慮したスコアに変換する必要がある。
			// そこで、value_from_tt()で変換してから返すのが正解。

			// 教師局面生成時には、これをオフにしたほうが良いかも知れない。
			// ただし、そのときであっても連続王手の千日手は有効にしておく。
			auto draw_type = pos.is_repetition();
			if (draw_type != REPETITION_NONE)
				return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);

			// 最大手数を超えている、もしくは停止命令が来ている。
			if (Signals.stop.load(std::memory_order_relaxed) || pos.game_ply() > Limits.max_game_ply)
				return draw_value(REPETITION_DRAW, pos.side_to_move());

			// -----------------------
			//  Mate Distance Pruning
			// -----------------------

			// rootから5手目の局面だとして、このnodeのスコアが
			// 5手以内で詰ますときのスコアを上回ることもないし、
			// 5手以内で詰まさせるときのスコアを下回ることもないので
			// これを枝刈りする。

			alpha = std::max(mated_in(ss->ply), alpha);
			beta = std::min(mate_in(ss->ply + 1), beta);
			if (alpha >= beta)
				return alpha;
		}

		// -----------------------
		//  探索Stackの初期化
		// -----------------------

		// この初期化、もう少し早めにしたほうがいい可能性が..
		// このnodeで指し手を進めずにリターンしたときにこの局面でのcurrnetMoveにゴミが入っていると困るような？
		ss->currentMove = MOVE_NONE;
		ss->counterMoves = nullptr;

		// historyの合計値を計算してcacheしておく用。
		ss->history = VALUE_ZERO;

		// 1手先のexcludedMoveの初期化
		(ss + 1)->excludedMove = MOVE_NONE;

		// 1手先のskipEarlyPruningフラグの初期化。
		(ss + 1)->skipEarlyPruning = false;

		// 2手先のkillerの初期化。
		(ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

		// -----------------------
		//   置換表のprobe
		// -----------------------

		// このnodeで探索から除外する指し手。ss->excludedMoveのコピー。
		Move excludedMove = ss->excludedMove;
		// 除外した指し手をxorしてそのままhash keyに使う。
		// 除外した指し手がないときは、0だから、xorしても0。
		// ただし、hash keyのbit0は手番を入れることになっているのでここは0にしておく。
		auto posKey = pos.key() ^ Key(excludedMove << 1);

		bool ttHit;    // 置換表がhitしたか

#ifndef DISABLE_TT_PROBE 
		TTEntry* tte = TT.probe(posKey, ttHit);

		// 置換表の指し手
		// 置換表にhitしなければMOVE_NONE

		// 置換表上のスコア
		// 置換表にhitしなければVALUE_NONE
		Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;

		// RootNodeであるなら、(MultiPVなどでも)現在注目している1手だけがベストの指し手と仮定できるから、
		// それが置換表にあったものとして指し手を進める。
		Move ttMove = RootNode ? thisThread->rootMoves[thisThread->PVIdx].pv[0]
					: ttHit    ? pos.move16_to_move(tte->move()) : MOVE_NONE;

#else
		TTEntry* tte = nullptr; ttHit = false;
		Move ttMove = MOVE_NONE;
		Value ttValue = VALUE_NONE;
#endif

		// 置換表の値による枝刈り

		if (!PvNode        // PV nodeでは置換表の指し手では枝刈りしない(PV nodeはごくわずかしかないので..)
			&& ttHit         // 置換表の指し手がhitして
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
			if (ttValue >= beta && ttMove)
			{
				int d = depth / ONE_PLY;

				if (!pos.capture_or_pawn_promotion(ttMove))
				{
					Value bonus = Value(d * d + 2 * d - 2);
					update_stats(pos, ss, ttMove, nullptr, 0,bonus);
				}

				// 反駁された1手前の置換表のquietな指し手に対する追加ペナルティを課す。
				// 1手前は置換表の指し手であるのでNULL MOVEではありえない。
				// ToDo:ここ、captureだけではなくpawn_promotionも含めるべきかも。
				if ((ss - 1)->moveCount == 1 && !pos.captured_piece())
				{
					Value penalty = Value(d * d + 4 * d + 1);
					Square prevSq = to_sq((ss - 1)->currentMove);
					update_cm_stats(ss - 1, pos.piece_on(prevSq), prevSq, -penalty);
				}
			}

			return ttValue;
		}

		// -----------------------
		//     宣言勝ち
		// -----------------------

		{
			// 王手がかかってようがかかってまいが、宣言勝ちの判定は正しい。
			// (トライルールのとき王手を回避しながら入玉することはありうるので)
			Move m = pos.DeclarationWin();
			if (m != MOVE_NONE)
			{
				bestValue = mate_in(ss->ply + 1); // 1手詰めなのでこの次のnodeで(指し手がなくなって)詰むという解釈
#ifndef DISABLE_TT_PROBE
				tte->save(posKey, value_to_tt(bestValue, ss->ply), BOUND_EXACT,
					DEPTH_MAX, m, ss->staticEval, TT.generation());
#endif
				return bestValue;
			}
		}

		// -----------------------
		//    詰まされているか？
		// -----------------------

		Move bestMove = MOVE_NONE;
		const bool inCheck = pos.checkers();

		// 王手がかかっているなら詰みであるから、王手将棋においてはこれで詰まされている。
		// search()のほうで王手になるならqsearch()を呼ばないのだが、RootNodeからは
		// 呼び出されるのでこの処理が必要になる。
		if (inCheck)
			return mated_in(ss->ply);

		// -----------------------
		//  局面を評価値によって静的に評価
		// -----------------------

		// 差分計算の都合、毎回evaluate()を呼ぶ。
		ss->staticEval = eval = evaluate(pos);

		if (inCheck)
		{
			// 評価値を置換表から取り出したほうが得だと思うが、反復深化でこのnodeに再訪問したときも
			// このnodeでは評価値を用いないであろうから、置換表にこのnodeの評価値があることに意味がない。

			ss->staticEval = eval = VALUE_NONE;
			goto MOVES_LOOP;

		} else if (ttHit) {

			// 置換表にhitしたなら、評価値が記録されているはずだから、それを取り出しておく。
			// あとで置換表に書き込むときにこの値を使えるし、各種枝刈りはこの評価値をベースに行なうから。

			// ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。
			// 1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
			//   evalとしてttValueを採用して良い。
			// 2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
			//   evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。
			if (ttValue != VALUE_NONE)
				if (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER))
					eval = ttValue;

		} else {

			// この処理、入れたほうがいいようだ。一見するとevaluate()は上で手番つきで求めているから
			// これをやると不正確になるだけのようであるが、null moveした局面で手番つきの評価関数を呼ぶと
			// 駒に当たっているものがプラス評価されて、評価値として大きく出すぎて悪作用があるようだ。

			// →　長い持ち時間ではそうでもないかも。
			//  play_time = b5000, 468 - 32 - 500(48.35% R - 11.49) [2016/08/19]
			//  手番込みの評価関数で手番がそれなりに正しく評価されているなら意味があるようだ。

#if 0
			if ((ss - 1)->currentMove == MOVE_NULL)
				eval = ss->staticEval = -(ss - 1)->staticEval + 2 * Tempo;
#endif

#ifndef DISABLE_TT_PROBE
			// 評価関数を呼び出したので置換表のエントリーはなかったことだし、何はともあれそれを保存しておく。
			tte->save(posKey, VALUE_NONE, BOUND_NONE, DEPTH_NONE, MOVE_NONE, ss->staticEval, TT.generation());
			// どうせ毎node評価関数を呼び出すので、evalの値にそんなに価値はないのだが、mate1ply()を
			// 実行したという証にはなるので意味がある。
#endif
		}

		// このnodeで指し手生成前の枝刈りを省略するなら指し手生成ループへ。
		if (ss->skipEarlyPruning)
			goto MOVES_LOOP;

		// -----------------------
		//   evalベースの枝刈り
		// -----------------------

		// 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
		// 王手のときはここにはこない。(上のinCheckのなかでMOVES_LOOPに突入。)

		//
		//   Razoring
		//

		// 残り探索深さが少ないときに、その手数でalphaを上回りそうにないとき用の枝刈り。
		if (!PvNode
			&&  depth < 4 * ONE_PLY
			&&  ttMove == MOVE_NONE
			&&  eval + razor_margin[depth/ONE_PLY] <= alpha
			)
		{
			// 残り探索深さがONE_PLY以下で、alphaを確実に下回りそうなら、ここで静止探索を呼び出してしまう。
			if (depth <= ONE_PLY
			//	&& eval + razor_margin[3] <= alpha
				// →　ここ、razoringとしてはrazor_margin[ZERO_DEPTH]を参照すべき。
				// しかしそれは前提条件として満たしているので結局、ここでは単にqsearch()を
				// 呼び出して良いように思う。
				)
				return  qsearch<NonPV, false>(pos, ss, alpha, beta, DEPTH_ZERO);

			// 残り探索深さが1～3手ぐらいあるときに、alpha - razor_marginを上回るかだけ調べて
			// 上回りそうにないならもうリターンする。
			Value ralpha = alpha - razor_margin[depth/ONE_PLY];
			Value v = qsearch<NonPV, false>(pos, ss, ralpha, ralpha + 1, DEPTH_ZERO);
			if (v <= ralpha)
				return v;
		}

		//
		//   Futility pruning
		//

		// このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
		// evalからこれを引いてbetaより大きいなら、beta cutが出来る。
		// ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
		// ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。

		if (   !RootNode
			&&  depth < PARAM_FUTILITY_RETURN_DEPTH * ONE_PLY
			&&  eval - futility_margin(depth) >= beta
			&&  eval < VALUE_KNOWN_WIN) // 詰み絡み等だとmate distance pruningで枝刈りされるはずで、ここでは枝刈りしない。
			return eval;
		// 以下のようにするより、単にevalを返したほうが良いらしい。
		// cf. https://github.com/official-stockfish/Stockfish/commit/f799610d4bb48bc280ea7f58cd5f78ab21028bf5
		//	return eval - futility_margin(depth);

		//
		//   Null move search with verification search
		//

		//  null move探索。PV nodeではやらない。
		//  evalの見積りがbetaを超えているので1手パスしてもbetaは超えそう。
		if (   !PvNode
			&&  eval >= beta
			&& (ss->staticEval >= beta - PARAM_NULL_MOVE_MARGIN * (depth / ONE_PLY - 6) || depth >= 13 * ONE_PLY)
			)
		{
			ss->currentMove = MOVE_NULL;
			ss->counterMoves = nullptr;

			// 残り探索深さと評価値によるnull moveの深さを動的に減らす
			Depth R = ((PARAM_NULL_MOVE_DYNAMIC_ALPHA + PARAM_NULL_MOVE_DYNAMIC_BETA * depth / ONE_PLY) / 256
				+ std::min((int)((eval - beta) / PawnValue), 3)) * ONE_PLY;

			pos.do_null_move(st);

			(ss + 1)->skipEarlyPruning = true;

			//  王手がかかっているときはここに来ていないのでqsearchはinCheck == falseのほうを呼ぶ。
			Value nullValue = depth - R < ONE_PLY ? -qsearch<NonPV, false>(pos, ss + 1, -beta, -beta + 1, DEPTH_ZERO)
												  : - search<NonPV       >(pos, ss + 1, -beta, -beta + 1, depth - R, !cutNode);
			(ss + 1)->skipEarlyPruning = false;
			pos.undo_null_move();

			if (nullValue >= beta)
			{
				// 1手パスしてもbetaを上回りそうであることがわかったので
				// これをもう少しちゃんと検証しなおす。

				// 証明されていないmate scoreはreturnで返さない。
				if (nullValue >= VALUE_MATE_IN_MAX_PLY)
					nullValue = beta;

				if (depth < PARAM_NULL_MOVE_RETURN_DEPTH * ONE_PLY && abs(beta) < VALUE_KNOWN_WIN)
					return nullValue;

				// nullMoveせずに(現在のnodeと同じ手番で)同じ深さで探索しなおして本当にbetaを超えるか検証する。cutNodeにしない。
				ss->skipEarlyPruning = true;
				Value v = depth - R < ONE_PLY ? qsearch<NonPV, false>(pos, ss, beta - 1, beta, DEPTH_ZERO)
											  :  search<NonPV       >(pos, ss, beta - 1, beta, depth - R, false);
				ss->skipEarlyPruning = false;

				if (v >= beta)
					return nullValue;
			}
		}

		//
		//   ProbCut
		//

		// もし、このnodeで非常に良いcaptureの指し手があり(例えば、SEEの値が動かす駒の価値を上回るようなもの)
		// 探索深さを減らしてざっくり見てもbetaを非常に上回る値を返すようなら、このnodeをほぼ安全に枝刈りすることが出来る。

		if (!PvNode
			&&  depth >= PARAM_PROBCUT_DEPTH * ONE_PLY
			&&  abs(beta) < VALUE_MATE_IN_MAX_PLY)
		{
			Value rbeta = std::min(beta + PARAM_PROBCUT_MARGIN, VALUE_INFINITE);

			// 大胆に探索depthを減らす
			Depth rdepth = depth - (PARAM_PROBCUT_DEPTH - 1) * ONE_PLY;

			ASSERT_LV3(rdepth >= ONE_PLY);
			ASSERT_LV3((ss - 1)->currentMove != MOVE_NONE);
			ASSERT_LV3((ss - 1)->currentMove != MOVE_NULL);

			// rbeta - ss->staticEvalを上回るcaptureの指し手のみを生成。
			MovePicker mp(pos, ttMove, rbeta - ss->staticEval);

			while ((move = mp.next_move()) != MOVE_NONE)
			{
				ASSERT_LV3(pos.pseudo_legal(move));

				if (pos.legal(move))
				{
					// 王手将棋においては王手の時点で詰みなので
					// この指し手が王手になるなら詰みのスコアを返す。
					if (pos.gives_check(move))
						return mate_in(ss->ply + 1);

					ss->currentMove = move;
					ss->counterMoves = &thisThread->counterMoveHistory[to_sq(move)][pos.moved_piece_after(move)];

					pos.do_move(move, st, pos.gives_check(move));
					value = -search<NonPV>(pos, ss + 1, -rbeta, -rbeta + 1, rdepth, !cutNode);
					pos.undo_move(move);
					if (value >= rbeta)
						return value;
				}
			}
		}

		//
		//   Internal iterative deepening
		//

		// いわゆる多重反復深化。残り探索深さがある程度あるのに置換表に指し手が登録されていないとき
		// (たぶん置換表のエントリーを上書きされた)、浅い探索をして、その指し手を置換表の指し手として用いる。
		// 置換表用のメモリが潤沢にあるときはこれによる効果はほとんどないはずではあるのだが…。

		if (depth >= 6 * ONE_PLY
			&& !ttMove
			&& (PvNode || ss->staticEval + PARAM_IID_MARGIN_ALPHA >= beta))
		{
			Depth d = (3 * depth / (4 * ONE_PLY) - 2) * ONE_PLY;
			ss->skipEarlyPruning = true;
			search<NT>(pos, ss, alpha, beta, d , cutNode);
			ss->skipEarlyPruning = false;

#ifndef DISABLE_TT_PROBE
			tte = TT.probe(posKey, ttHit);
			ttMove = ttHit ? pos.move16_to_move(tte->move()) : MOVE_NONE;
#else
			tte = nullptr;
			ttHit = false;
			ttMove = MOVE_NONE;
#endif
		}

		// -----------------------
		// 1手ずつ指し手を試す
		// -----------------------

	MOVES_LOOP:

		// cmh  = Counter Move History    : ある指し手が指されたときの応手
		// fmh  = Follow up Move History  : 2手前の自分の指し手の継続手
		// fmh2 = Follow up Move History2 : 4手前からの継続手
		const CounterMoveStats* cmh  = (ss - 1)->counterMoves;
		const CounterMoveStats* fmh  = (ss - 2)->counterMoves;
		const CounterMoveStats* fmh2 = (ss - 4)->counterMoves;

		// 評価値が2手前の局面から上がって行っているのかのフラグ
		// 上がって行っているなら枝刈りを甘くする。
		// ※ VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
		//   枝刈りを甘くして調べないといけないのでimproving扱いとする。
		bool improving = ss->staticEval >= (ss - 2)->staticEval
		//			  || ss->staticEval == VALUE_NONE
		// この条件は一つ上の式に暗黙的に含んでいる。
		// ※　VALUE_NONE == 32002なのでこれより大きなstaticEvalの値であることはないので。
					  || (ss - 2)->staticEval == VALUE_NONE;

		// singular延長をするnodeであるか。
		bool singularExtensionNode = !RootNode
			&&  depth >= PARAM_SINGULAR_EXTENSION_DEPTH * ONE_PLY // Stockfish , Apreyは、8 * ONE_PLY
			&&  ttMove != MOVE_NONE
			&&  ttValue != VALUE_NONE // 詰み絡みのスコアであってもsingular extensionはしたほうが良いらしい。
			&& !excludedMove // 再帰的なsingular延長はすべきではない
			&& (tte->bound() & BOUND_LOWER)
			&& tte->depth() >= depth - 3 * ONE_PLY;
		// このnodeについてある程度調べたことが置換表によって証明されている。
		// (そうでないとsingularの指し手以外に他の有望な指し手がないかどうかを調べるために
		// null window searchするときに大きなコストを伴いかねないから。)

		// 調べた指し手を残しておいて、statusのupdateを行なうときに使う。
		// ここ、PARAM_QUIET_SEARCH_COUNTにしたいが、これは自動調整時はstatic変数なので指定できない。
		Move quietsSearched[
#if defined (USE_AUTO_TUNE_PARAMETERS) || defined(USE_RANDOM_PARAMETERS)
			128
#else
			PARAM_QUIET_SEARCH_COUNT
#endif
		];
		int quietCount = 0;

		// このnodeでdo_move()された合法手の数
		int moveCount = 0;

		// このあとnodeを展開していくので、evaluate()の差分計算ができないと速度面で損をするから、
		// evaluate()を呼び出していないなら呼び出しておく。
		// ss->staticEvalに代入するとimprovingの判定間違うのでそれはしないほうがよさげ。
		evaluate_with_no_return(pos);

		MovePicker mp(pos, ttMove, depth, ss);

#if defined(__GNUC__)
		// g++でコンパイルするときにvalueが未初期化かも知れないという警告が出るのでその回避策。
		value = bestValue;
#endif

		//  一手ずつ調べていく

		while ((move = mp.next_move()) != MOVE_NONE)
		{
			ASSERT_LV3(is_ok(move));

			if (move == excludedMove)
				continue;

			// root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。
			if (RootNode && !std::count(thisThread->rootMoves.begin() + thisThread->PVIdx,
										thisThread->rootMoves.end(), move))
				continue;

			// do_move()した指し手の数のインクリメント
			// このあとdo_move()の前で枝刈りのためにsearchを呼び出す可能性があるので
			// このタイミングでやっておき、legalでなければ、この値を減らす
			ss->moveCount = ++moveCount;

			// この読み筋の出力、細かすぎるので時間をロスする。しないほうがいいと思う。
#if 0
			// 3秒以上経過しているなら現在探索している指し手をGUIに出力する。
			if (RootNode && !Limits.silent && thisThread == Threads.main() && Time.elapsed() > 3000)
				sync_cout << "info depth " << depth / ONE_PLY
				<< " currmove " << move
				<< " currmovenumber " << moveCount + thisThread->PVIdx << sync_endl;
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

			Depth extension = DEPTH_ZERO;

			// 今回の指し手で王手になるかどうか
			bool givesCheck = pos.gives_check(move);

			// 王手将棋ではこれにて詰み
			// ただしRootNodeでは、th->rootMovesを更新する必要があるので
			// この判定は行わない。
			if (givesCheck && !RootNode)
			{
				if (!pos.legal(move))
				{
					ss->moveCount = --moveCount;
					continue;
				}

				// これが合法手なら詰み
				return mate_in(ss->ply + 1);
			}

			// move countベースの枝刈りを実行するかどうかのフラグ

			bool moveCountPruning = depth < PARAM_PRUNING_BY_MOVE_COUNT_DEPTH * ONE_PLY
				&& moveCount >= FutilityMoveCounts[improving][depth / ONE_PLY];

			// 王手となる指し手でSEE >= 0であれば残り探索深さに1手分だけ足す。
			// また、moveCountPruningでない指し手(置換表の指し手とか)も延長対称。
			// これはYSSの0.5手延長に似たもの。
			// ※　将棋においてはこれはやりすぎの可能性も..

			if (givesCheck
				&& !moveCountPruning
				&&  pos.see_ge(move , VALUE_ZERO)
				)
				extension = ONE_PLY;

			//
			// Singular extension search.
			//

#if 1
			// (alpha-s,beta-s)の探索(sはマージン値)において1手以外がすべてfail lowして、
			// 1手のみが(alpha,beta)においてfail highしたなら、指し手はsingularであり、延長されるべきである。
			// これを調べるために、ttMove以外の探索深さを減らして探索して、
			// その結果がttValue-s 以下ならttMoveの指し手を延長する。

			// Stockfishの実装だとmargin = 2 * depthだが、(ONE_PLY==1として)、
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

			if (singularExtensionNode
				&&  move == ttMove
				&& !extension        // 延長が確定しているところはこれ以上調べても仕方がない。
				&&  pos.legal(move))
			{
				// このmargin値は評価関数の性質に合わせて調整されるべき。
				// PARAM_SINGULAR_MARGIN == 128(無調整)のときはdefault動作。
				Value rBeta;
				if (PARAM_SINGULAR_MARGIN == 128)
					rBeta = std::max(ttValue - 2 * depth / ONE_PLY, -VALUE_MATE);
				else
					rBeta = std::max(ttValue - PARAM_SINGULAR_MARGIN * depth / (64 * ONE_PLY), -VALUE_MATE);

				// PARAM_SINGULAR_SEARCH_DEPTH_ALPHAが16(無調整)のときはデフォルト動作。
				Depth d;
				if (PARAM_SINGULAR_SEARCH_DEPTH_ALPHA == 16)
					d = (depth / (2 * ONE_PLY)) * ONE_PLY;
				else
					d = (depth * PARAM_SINGULAR_SEARCH_DEPTH_ALPHA / (32 * ONE_PLY)) * ONE_PLY;

				// ttMoveの指し手を以下のsearch()での探索から除外
				ss->excludedMove = move;
				ss->skipEarlyPruning = true;
				// 局面はdo_move()で進めずにこのnodeから浅い探索深さで探索しなおす。
				// 浅いdepthでnull windowなので、すぐに探索は終わるはず。
				value = search<NonPV>(pos, ss, rBeta - 1, rBeta, d, cutNode);
				ss->skipEarlyPruning = false;
				ss->excludedMove = MOVE_NONE;

				// 置換表の指し手以外がすべてfail lowしているならsingular延長確定。
				if (value < rBeta)
					extension = ONE_PLY;

				// singular extentionが生じた回数の統計を取ってみる。
				// dbg_hit_on(extension == ONE_PLY);
			}
#endif

			// -----------------------
			//   1手進める前の枝刈り
			// -----------------------

			// 再帰的にsearchを呼び出すとき、search関数に渡す残り探索深さ。
			// これはsingluar extensionの探索が終わってから決めなければならない。(singularなら延長したいので)
			Depth newDepth = depth - ONE_PLY + extension;


			// bool captureOrPawnPromotion = pos.capture_or_promotion(move);

			// これはpromotionをもっと絞ったほうがよさそうだ。
			// (大きく加点されるのは、駒取りと歩の成りだけで、それ以外はそんなに大きな点数上昇ではないから。)

			//  play_time = r300, 2626 - 73 - 2301(53.3% R22.95) [2016/08/20]
			//	play_time = b1000, 1191 - 49 - 1010(54.11% R28.64) [2016/08/20]


			//	bool captureOrPawnPromotion = pos.capture_or_valuable_promotion(move);

			//	捕獲＋歩を捕獲 + 歩、角、飛の成りにした場合弱くなる。
			//	成りを増やすのはオーダリングにいい影響を与えないようだ。
			// 	play_time = r300 ,  2566 - 76 - 2358(52.11% R14.69) [2016/08/20]
			//	play_time = b1000, 311 - 9 - 280(52.62% R18.24)[2016/08/20]

			//	bool captureOrPawnPromotion = pos.capture(move);

			// captureだけにするのは、やりすぎのようだ。


			// 指し手で捕獲する指し手、もしくは歩の成りである。
			bool captureOrPawnPromotion = pos.capture_or_pawn_promotion(move);

			//
			// Pruning at shallow depth
			//

			// 浅い深さでの枝刈り


			// このあと、この指し手のhistoryの値などを調べたいのでいま求めてしまう。
			Square moved_sq = to_sq(move);
			Piece moved_pc = pos.moved_piece_after(move);


			if (!RootNode
			//	&& !inCheck
			// →　王手がかかっていても以下の枝刈りはしたほうが良いらしいが…。
			// cf. 	https://github.com/official-stockfish/Stockfish/commit/ab26c61971c2f73d312b003e6d024373fbacf8e6
			// T1,r300,2501 - 73 - 2426(50.76% R5.29)
			// T1,b1000,2428 - 97 - 2465(49.62% R-2.63)
			// 1秒のほうではやや勝ち越し。計測できない程度の差だが良しとする。

				&& bestValue > VALUE_MATED_IN_MAX_PLY)
			{

				if (!captureOrPawnPromotion
					&& !givesCheck
					// && !pos.advanced_pawn_push(move)
					)
				{

					// Move countに基づいた枝刈り(futilityの亜種)

					if (moveCountPruning)
						continue;

					// 次のLMR探索における軽減された深さ
					int lmrDepth = std::max(newDepth - reduction<PvNode>(improving, depth, moveCount), DEPTH_ZERO) / ONE_PLY;

					// Historyに基づいた枝刈り(history && counter moveの値が悪いものに関してはskip)

					// ToDo : このへん、fmh,fmh2を調べるほうが良いかは微妙
					if (lmrDepth < PARAM_PRUNING_BY_HISTORY_DEPTH
						//					&& move != ss->killers[0]
						// →　このkillerの判定は入れないほうが強いらしい。
						&& (!cmh  || (*cmh)[moved_sq][moved_pc] < VALUE_ZERO)
						&& (!fmh  || (*fmh)[moved_sq][moved_pc] < VALUE_ZERO)
						&& (!fmh2 || (*fmh2)[moved_sq][moved_pc] < VALUE_ZERO || (cmh && fmh)))
						continue;

					// Futility pruning: at parent node
					// 親nodeの時点で子nodeを展開する前にfutilityの対象となりそうなら枝刈りしてしまう。

					if (lmrDepth < PARAM_FUTILITY_AT_PARENT_NODE_DEPTH
						&& !inCheck
						&& ss->staticEval + PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1
						+ PARAM_FUTILITY_MARGIN_BETA * lmrDepth <= alpha)
						continue;

					// ※　このLMRまわり、強さに極めて重大な影響があるので枝刈りを入れるかどうかを含めて慎重に調整すべき。

					// Prune moves with negative SEE
					// SEEが負の指し手を枝刈り

					// 将棋ではseeが負の指し手もそのあと詰むような場合があるから、あまり無碍にも出来ないようだが…。

					if (!pos.see_ge(move , Value(-PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1 * lmrDepth * lmrDepth)))
						continue;
				}

				// 浅い深さでの、危険な指し手を枝刈りする。
				else if (!extension
					&& !pos.see_ge(move , Value(-PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2 * depth / ONE_PLY * depth / ONE_PLY)
#if 0
						// Stockfish 8相当
						+ (ss->staticEval != VALUE_NONE ? ss->staticEval - alpha - PARAM_FUTILITY_AT_PARENT_NODE_MARGIN2 : VALUE_ZERO)))
#else
						// PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2を少し大きめにして調整したほうがよさげ。
					))
#endif
					continue;
			}

			// -----------------------
			//      1手進める
			// -----------------------

			// この時点で置換表をprefetchする。将棋においては、指し手に駒打ちなどがあって指し手を適用したkeyを
			// 計算するコストがわりとあるので、これをやってもあまり得にはならない。無効にしておく。

			// prefetch(TT.first_entry(pos.key_after(move)));

			// legal()のチェック。root nodeだとlegal()だとわかっているのでこのチェックは不要。
			// 非合法手はほとんど含まれていないからこの判定はdo_move()の直前まで遅延させたほうが得。
			if (!RootNode && !pos.legal(move))
			{
				// 足してしまったmoveCountを元に戻す。
				ss->moveCount = --moveCount;
				continue;
			}

			// 現在このスレッドで探索している指し手を保存しておく。
			ss->currentMove = move;
			ss->counterMoves = &thisThread->counterMoveHistory[moved_sq][moved_pc];

			// 指し手で1手進める
			pos.do_move(move, st, givesCheck);

			// -----------------------
			// LMR(Late Move Reduction)
			// -----------------------

			// moveCountが大きいものなどは探索深さを減らしてざっくり調べる。
			// alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。

			if (depth >= 3 * ONE_PLY
				&& moveCount > 1
				&& (!captureOrPawnPromotion || moveCountPruning))
			{
				// Reduction量
				Depth r = reduction<PvNode>(improving, depth, moveCount);

				if (captureOrPawnPromotion)

					r -= r ? ONE_PLY : DEPTH_ZERO;

				else

				{

					// cut nodeにおいてhistoryの値が悪い指し手に対してはreduction量を増やす。
					// ※　PVnodeではIID時でもcutNode == trueでは呼ばないことにしたので、
					// if (cutNode)という条件式は暗黙に && !PvNode を含む。

					// 2 * ONE_PLYは、将棋においてはやりすぎの可能性もある。
					// もう少し細かく調整したほうが好ましいのだが、ONE_PLY == 1のままだと少し難しい。

					if (cutNode)
						r += 2 * ONE_PLY;

#if 1
					// 捕獲から逃れる指し手はreduction量を減らす。

					// do_moveしたあとなのでtoの位置には今回移動させた駒が来ている。
					// fromの位置は空(NO_PIECE)の升となっている。

					// 例えばtoの位置に金があるとして、これをfromに動かす。
					// 仮にこれが歩で取られるならsee() < 0 となる。

					// ただ、KPPT型の評価関数では駒の当たりは評価されているので
					// ここでreduction量を減らすのはあまり良くない。
					// see()のコストが割にあわない。このコードは使わないほうがいいはず。

					// →　入れたほうがわずかに強いようなので残しておく。
					// play_time = b1000, 1446 - 51 - 1503(49.03% R - 6.72) [2016/08/19]

					else if (!is_drop(move) // type_of(move)== NORMAL
						&& type_of(pos.piece_on(to_sq(move))) != PAWN

						// see_sign()だと、toの升の駒でfromの升の駒(NO_PIECE)を取るから
						// 必ず正になってしまうため、see_sign()ではなくsee()を用いる。

						&& !pos.see_ge(make_move(to_sq(move), move_from(move)),VALUE_ZERO))
						r -= 2 * ONE_PLY;
#endif

					// ToDo:ここ、fmh,fmh2を見たほうがいいかは微妙。
					ss->history =   thisThread->history[moved_sq][moved_pc]
									+ (cmh  ? (*cmh )[moved_sq][moved_pc] : VALUE_ZERO)
									+ (fmh  ? (*fmh )[moved_sq][moved_pc] : VALUE_ZERO)
									+ (fmh2 ? (*fmh2)[moved_sq][moved_pc] : VALUE_ZERO)
									+ thisThread->fromTo.get(~pos.side_to_move(), move)
									-PARAM_REDUCTION_BY_HISTORY; // 修正項

					// historyの値に応じて指し手のreduction量を増減する。

#if 0
					// これ、やったほうがいいかどうかは微妙。1秒、3秒においてはやると弱くなるようだが…。
					// T1,b1000,2135 - 84 - 2071(50.76% R5.29)
					// T1,b3000,640 - 34 - 576(52.63% R18.3)

					if (ss->history > VALUE_ZERO && (ss - 1)->history < VALUE_ZERO)
						r -= ONE_PLY;
					else if (ss->history < VALUE_ZERO && (ss - 1)->history > VALUE_ZERO)
						r += ONE_PLY;
#endif

					r = std::max(DEPTH_ZERO, (r / ONE_PLY - ss->history / 20000) * ONE_PLY);
				}

				// depth >= 3なのでqsearchは呼ばれないし、かつ、
				// moveCount > 1 すなわち、このnodeの2手目以降なのでsearch<NonPv>が呼び出されるべき。
				Depth d = std::max(newDepth - r, ONE_PLY);
				value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);

				//
				// ここにその他の枝刈り、何か入れるべき(かも)
				//

				// 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす。
				doFullDepthSearch = (value > alpha) && (d != newDepth);

			} else {

				// non PVか、PVでも2手目以降であればfull depth searchを行なう。
				doFullDepthSearch = !PvNode || moveCount > 1;

			}

			// Full depth search
			// LMRがskipされたか、LMRにおいてfail highを起こしたなら元の探索深さで探索する。

			// ※　静止探索は残り探索深さはDEPTH_ZEROとして開始されるべきである。(端数があるとややこしいため)
			if (doFullDepthSearch)
				value = newDepth < ONE_PLY ?
				givesCheck ? -qsearch<NonPV, true >(pos, ss + 1, -(alpha + 1), -alpha, DEPTH_ZERO)
				           : -qsearch<NonPV, false>(pos, ss + 1, -(alpha + 1), -alpha, DEPTH_ZERO)
				           :  -search<NonPV       >(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

			// PV nodeにおいては、full depth searchがfail highしたならPV nodeとしてsearchしなおす。
			// ただし、value >= betaなら、正確な値を求めることにはあまり意味がないので、これはせずにbeta cutしてしまう。
			if (PvNode && (moveCount == 1 || (value > alpha && (RootNode || value < beta))))
			{
				// 次のnodeのPVポインターはこのnodeのpvバッファを指すようにしておく。
				(ss + 1)->pv = pv;
				(ss + 1)->pv[0] = MOVE_NONE;

				// full depthで探索するときはcutNodeにしてはいけない。
				value = newDepth < ONE_PLY  ?
								givesCheck  ? -qsearch<PV, true >(pos, ss + 1, -beta, -alpha, DEPTH_ZERO)
											: -qsearch<PV, false>(pos, ss + 1, -beta, -alpha, DEPTH_ZERO)
											: - search<PV       >(pos, ss + 1, -beta, -alpha, newDepth, false);

			}

			// -----------------------
			//      1手戻す
			// -----------------------

			pos.undo_move(move);

			ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

			// 停止シグナルが来たら置換表を汚さずに終了。
			if (Signals.stop.load(std::memory_order_relaxed))
				return VALUE_ZERO;

			// -----------------------
			//  root node用の特別な処理
			// -----------------------

			if (RootNode)
			{
				auto& rm = *std::find(thisThread->rootMoves.begin(),
									  thisThread->rootMoves.end(), move);

				if (moveCount == 1 || value > alpha)
				{
					// root nodeにおいてPVの指し手または、α値を更新した場合、スコアをセットしておく。
					// (iterationの終わりでsortするのでそのときに指し手が入れ替わる。)

					rm.score = value;
					rm.pv.resize(1);
					// PVは変化するはずなのでいったんリセット

					// 1手進めたのだから、何らかPVを持っているはずなのだが。
					ASSERT_LV3((ss + 1)->pv);

					// RootでPVが変わるのは稀なのでここがちょっとぐらい重くても問題ない。
					// 新しく変わった指し手の後続のpvをRootMoves::pvにコピーしてくる。
					for (Move* m = (ss + 1)->pv; *m != MOVE_NONE; ++m)
						rm.pv.push_back(*m);

					if (moveCount > 1 && thisThread == Threads.main())
						++static_cast<MainThread*>(thisThread)->bestMoveChanges;

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
					// 不安定なnodeにおいてeasy moveをクリアする。
					// ※　　posKeyは、excludedMoveが指定されていると本来のkeyとは異なることになるが、それは
					// singular extensionのときにしか関係なくて、singular extensionは深いdepthでしかやらないので、
					// EasyMove.get()で返す2手目のkeyには影響を及ぼさない。
					if (PvNode
						&&  thisThread == Threads.main()
						&& EasyMove.get(posKey)
						&& (move != EasyMove.get(posKey) || moveCount > 1))
						EasyMove.clear();
					// ここ、Stockfishのコード、pos.key()になっているが、
					// excludedMoveがあるとき、動作が違うことになるが、
					// excludedMoveがあるときはPvNodeではないのでそのケースは考えなくて良い。

					bestMove = move;

					// fail highのときにもPVをupdateする。
					if (PvNode && !RootNode)
						update_pv(ss->pv, move, (ss + 1)->pv);

					// alpha値を更新したので更新しておく
					if (PvNode && value < beta)
						alpha = value;
					else
					{
						// value >= beta なら fail high(beta cut)

						// また、non PVであるなら探索窓の幅が0なのでalphaを更新した時点で、value >= betaが言えて、
						// beta cutである。

						ASSERT_LV3(value >= beta);
						break;
					}
				}
			}

			// 探索した指し手を64手目までquietsSearchedに登録しておく。
			// あとでhistoryなどのテーブルに加点/減点するときに使う。

			if (!captureOrPawnPromotion && move != bestMove && quietCount < PARAM_QUIET_SEARCH_COUNT)
				quietsSearched[quietCount++] = move;

		}
		// end of while

		// -----------------------
		//  生成された指し手がない？
		// -----------------------

		// このStockfishのassert、合法手を生成しているので重すぎる。良くない。
		ASSERT_LV5(moveCount || !inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

		  // 合法手がない == 詰まされている ので、rootの局面からの手数で詰まされたという評価値を返す。
		  // ただし、singular extension中のときは、ttMoveの指し手が除外されているので単にalphaを返すべき。
		if (!moveCount)
			bestValue = excludedMove ? alpha : mated_in(ss->ply);

		// 詰まされていない場合、bestMoveがあるならこの指し手をkiller等に登録する。
		else if (bestMove)
		{
			int d = depth / ONE_PLY;

			// quietな(駒を捕獲しない)best moveなのでkillerとhistoryとcountermovesを更新する。
			if (!pos.capture_or_pawn_promotion(bestMove))
			{
				Value bonus = Value(d * d + 2 * d - 2);
				update_stats(pos, ss, bestMove, quietsSearched, quietCount, bonus);
			}

			// 反駁された1手前の置換表のquietな指し手に対する追加ペナルティを課す。
			// 1手前は置換表の指し手であるのでNULL MOVEではありえない。
			if ((ss - 1)->moveCount == 1 && !pos.captured_piece())
			{
				Piece prevPc = pos.moved_piece_after((ss - 1)->currentMove);

				Value penalty = Value(d * d + 4 * d + 1);
				Square prevSq = to_sq((ss - 1)->currentMove);
				update_cm_stats(ss - 1, prevPc, prevSq, -penalty);
			}
		}

		// bestMoveがない == fail lowしているケース。
		// fail lowを引き起こした前nodeでのcounter moveに対してボーナスを加点する。
		// ToDo:ここ、captured_piece_type()で見るのではなく、
		//  capture or pawn promotion相当の処理にしたほうが良いのでは…。
		else if (depth >= 3 * ONE_PLY
			&& !pos.captured_piece()
			&& is_ok((ss - 1)->currentMove))
		{
			Piece prevPc = pos.moved_piece_after((ss - 1)->currentMove);

			// 残り探索depthの2乗ぐらいのボーナスを与える。
			int d = depth / ONE_PLY;
			Value bonus = Value(d * d + 2 * d - 2);
			Square prevSq = to_sq((ss - 1)->currentMove);
			update_cm_stats(ss - 1, prevPc, prevSq, bonus);
		}

		// -----------------------
		//  置換表に保存する
		// -----------------------

		// betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
		// すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
		// さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
		// また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
		// ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
		// すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。

#ifndef DISABLE_TT_PROBE
		tte->save(posKey, value_to_tt(bestValue, ss->ply),
			bestValue >= beta ? BOUND_LOWER :
			PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
			depth, bestMove, ss->staticEval, TT.generation());
#endif

		// 置換表には abs(value) < VALUE_INFINITEの値しか書き込まないし、この関数もこの範囲の値しか返さない。
		// ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);
		// →　もっと厳しいassertを書いたほうがいいな…。
		ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));

		return bestValue;
	}
}

using namespace CheckShogi;

// --- 以下に好きなように探索のプログラムを書くべし。

// 定跡ファイル
Book::MemoryBook book;

// パラメーターの初期化
void init_param()
{
	// -----------------------
	//   parameters.hの動的な読み込み
	// -----------------------
#if defined (USE_AUTO_TUNE_PARAMETERS) || defined(USE_RANDOM_PARAMETERS) || defined(ENABLE_OUTPUT_GAME_RESULT)
	{
		vector<string> param_names = {
			"PARAM_FUTILITY_MARGIN_ALPHA" , "PARAM_FUTILITY_MARGIN_BETA" ,
			"PARAM_FUTILITY_MARGIN_QUIET" , "PARAM_FUTILITY_RETURN_DEPTH",
			
			"PARAM_FUTILITY_AT_PARENT_NODE_DEPTH",
			"PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1",
			"PARAM_FUTILITY_AT_PARENT_NODE_MARGIN2",
			"PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1" ,
			"PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2" ,

			"PARAM_NULL_MOVE_DYNAMIC_ALPHA","PARAM_NULL_MOVE_DYNAMIC_BETA",
			"PARAM_NULL_MOVE_MARGIN","PARAM_NULL_MOVE_RETURN_DEPTH",

			"PARAM_PROBCUT_DEPTH","PARAM_PROBCUT_MARGIN",
			
			"PARAM_SINGULAR_EXTENSION_DEPTH","PARAM_SINGULAR_MARGIN","PARAM_SINGULAR_SEARCH_DEPTH_ALPHA",
			
			"PARAM_PRUNING_BY_MOVE_COUNT_DEPTH","PARAM_PRUNING_BY_HISTORY_DEPTH","PARAM_REDUCTION_BY_HISTORY",
			"PARAM_IID_MARGIN_ALPHA",
			"PARAM_RAZORING_MARGIN1","PARAM_RAZORING_MARGIN2","PARAM_RAZORING_MARGIN3","PARAM_RAZORING_MARGIN4",

			"PARAM_REDUCTION_ALPHA",

			"PARAM_FUTILITY_MOVE_COUNT_ALPHA0","PARAM_FUTILITY_MOVE_COUNT_ALPHA1",
			"PARAM_FUTILITY_MOVE_COUNT_BETA0","PARAM_FUTILITY_MOVE_COUNT_BETA1",

			"PARAM_QUIET_SEARCH_COUNT",

			"PARAM_QSEARCH_MATE1","PARAM_SEARCH_MATE1","PARAM_WEAK_MATE_PLY"

		};

#ifdef 		ENABLE_OUTPUT_GAME_RESULT
		vector<const int*> param_vars = {
#else
		vector<int*> param_vars = {
#endif
			&PARAM_FUTILITY_MARGIN_ALPHA , &PARAM_FUTILITY_MARGIN_BETA,
			&PARAM_FUTILITY_MARGIN_QUIET , &PARAM_FUTILITY_RETURN_DEPTH,
			
			&PARAM_FUTILITY_AT_PARENT_NODE_DEPTH,
			&PARAM_FUTILITY_AT_PARENT_NODE_MARGIN1,
			&PARAM_FUTILITY_AT_PARENT_NODE_MARGIN2,
			&PARAM_FUTILITY_AT_PARENT_NODE_GAMMA1,
			&PARAM_FUTILITY_AT_PARENT_NODE_GAMMA2,

			&PARAM_NULL_MOVE_DYNAMIC_ALPHA, &PARAM_NULL_MOVE_DYNAMIC_BETA,
			&PARAM_NULL_MOVE_MARGIN,&PARAM_NULL_MOVE_RETURN_DEPTH,
			
			&PARAM_PROBCUT_DEPTH, &PARAM_PROBCUT_MARGIN,

			&PARAM_SINGULAR_EXTENSION_DEPTH, &PARAM_SINGULAR_MARGIN,&PARAM_SINGULAR_SEARCH_DEPTH_ALPHA,
			
			&PARAM_PRUNING_BY_MOVE_COUNT_DEPTH, &PARAM_PRUNING_BY_HISTORY_DEPTH,&PARAM_REDUCTION_BY_HISTORY,
			&PARAM_IID_MARGIN_ALPHA,
			&PARAM_RAZORING_MARGIN1,&PARAM_RAZORING_MARGIN2,&PARAM_RAZORING_MARGIN3,&PARAM_RAZORING_MARGIN4,

			&PARAM_REDUCTION_ALPHA,

			&PARAM_FUTILITY_MOVE_COUNT_ALPHA0,&PARAM_FUTILITY_MOVE_COUNT_ALPHA1,
			&PARAM_FUTILITY_MOVE_COUNT_BETA0,&PARAM_FUTILITY_MOVE_COUNT_BETA1,

			&PARAM_QUIET_SEARCH_COUNT,

			&PARAM_QSEARCH_MATE1,&PARAM_SEARCH_MATE1,&PARAM_WEAK_MATE_PLY,

		};

		fstream fs;
		fs.open("param\\" PARAM_FILE, ios::in);
		if (fs.fail())
		{
			cout << "info string Error! : can't read " PARAM_FILE << endl;
			return;
		}

		int count = 0;
		string line, last_line;

		// bufのなかにある部分文字列strの右側にある数値を読む。
		auto get_num = [](const string& buf, const string& str)
		{
			auto pos = buf.find(str);
			ASSERT_LV3(pos != -1);
			return stoi(buf.substr(pos + str.size()));
		};

		vector<bool> founds(param_vars.size());

		while (!fs.eof())
		{
			getline(fs, line);
			if (line.find("PARAM_DEFINE") != -1)
			{
				for (int i = 0; i < param_names.size(); ++i)
					if (line.find(param_names[i]) != -1)
					{
						count++;

						// "="の右側にある数値を読む。
#ifndef ENABLE_OUTPUT_GAME_RESULT
						*param_vars[i] = get_num(line, "=");
#endif

						// 見つかった
						founds[i] = true;

#ifdef USE_RANDOM_PARAMETERS
						// PARAM_DEFINEの一つ前の行には次のように書いてあるはずなので、
						// USE_RANDOM_PARAMETERSのときは、このstepをプラスかマイナス方向に加算してやる。
						// ただし、fixedと書いてあるパラメーターに関しては除外する。
						// interval = 2だと、-2*step,-step,+0,+step,2*stepの5つを試す。

						// [PARAM] min:100,max:240,step:3,interval:1,time_rate:1,fixed

						// "fixed"と書かれているパラメーターはないものとして扱う。
						if (last_line.find("fixed") != -1)
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
						vector<int> a;
						
						for (int j = 0; j <= param_interval; ++j)
						{
							// j==0のときは同じ値であり、これはのちに除外される。
							a.push_back(max(v - param_step*j,param_min));
							a.push_back(min(v + param_step*j,param_max));
						}

						// 重複除去。
						// 1) std::unique()は隣接要素しか削除しないので事前にソートしている。
						// 2) std::unique()では末尾にゴミが残るのでそれをerase()で消している。
						std::sort(a.begin(), a.end());
						a.erase(std::unique(a.begin(), a.end()), a.end());

						// 残ったものから1つをランダムに選択
						if (a.size() == 0)
						{
							cout << "Error : param is out of range -> " << line << endl;
						} else {
							*param_vars[i] = a[rand.rand(a.size())];
						}
#endif

						//            cout << param_names[i] << " = " << *param_vars[i] << endl;
						goto NEXT;
					}
				cout << "Error : param not found! in parameters.h -> " << line << endl;

			NEXT:;
			}
			last_line = line; // 1つ前の行を記憶しておく。
		}
		fs.close();

		// 読み込んだパラメーターの数が合致しないといけない。
		// 見つかっていなかったパラメーターを表示させる。
		if (count != param_names.size())
		{
			for (int i = 0; i < founds.size(); ++i)
				if (!founds[i])
					cout << "Error : param not found in " << PARAM_FILE << " -> " << param_names[i] << endl;
		}

#if defined (USE_RANDOM_PARAMETERS) || defined(ENABLE_OUTPUT_GAME_RESULT)
		{
			if (!result_log.is_open())
				result_log.open(Options["PARAMETERS_LOG_FILE_PATH"], ios::app);
			// 今回のパラメーターをログファイルに書き出す。
			for (int i = 0; i < param_names.size(); ++i)
			{
				if (param_names[i] == "FIXED")
					continue;

				result_log << param_names[i] << ":" << *param_vars[i] << ",";
			}
			result_log << endl << flush;
		}
#endif

	}
#endif
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init() {}

// パラメーターのランダム化のときには、
// USIの"gameover"コマンドに対して、それをログに書き出す。
void gameover_handler(const string& cmd)
{
#if defined (USE_RANDOM_PARAMETERS) || defined(ENABLE_OUTPUT_GAME_RESULT)
	result_log << cmd << endl << flush;
#endif
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void Search::clear()
{
	// -----------------------
	//   探索パラメーターの初期化
	// -----------------------

	init_param();

	// -----------------------
	//   テーブルの初期化
	// -----------------------

	// LMRで使うreduction tableの初期化

	// この初期化処理、起動時に1度でも良いのだが、探索パラメーターの調整を行なうときは、
	// init_param()のあとに行なうべきなので、ここで初期化することにする。

	// pvとnon pvのときのreduction定数
	// 0.05とか変更するだけで勝率えらく変わる

	// K[][2] = { nonPV時 }、{ PV時 }

	// パラメーターの自動調整のため、前の値として0以外が入っているかも知れないのでゼロ初期化する。
	memset(&reduction_table, 0, sizeof(reduction_table));

	for (int imp = 0; imp <= 1; ++imp)
		for (int d = 1; d < 64; ++d)
			for (int mc = 1; mc < 64; ++mc)
			{
				// 基本的なアイデアとしては、log(depth) × log(moveCount)に比例した分だけreductionさせるというもの。
				double r = log(d) * log(mc) * PARAM_REDUCTION_ALPHA / 256;

				reduction_table[NonPV][imp][d][mc] = int(round(r)) * ONE_PLY;
				reduction_table[PV][imp][d][mc] = std::max(reduction_table[NonPV][imp][d][mc] - 1, 0);

				// nonPVでimproving(評価値が2手前から上がっている)でないときはreductionの量を増やす。
				// →　これ、ほとんど効果がないようだ…。あとで調整すべき。
				if (!imp && reduction_table[NonPV][imp][d][mc] >= 2)
					reduction_table[NonPV][imp][d][mc] ++;
			}

	// Futilityで用いるテーブルの初期化

	// 残り探索depthが少なくて、王手がかかっていなくて、王手にもならないような指し手を
	// 枝刈りしてしまうためのmoveCountベースのfutilityで用いるテーブル。
	// FutilityMoveCounts[improving][残りdepth/ONE_PLY]
	for (int d = 0; d < PARAM_PRUNING_BY_MOVE_COUNT_DEPTH; ++d)
	{
		//FutilityMoveCounts[0][d] = int(2.4 + 0.773 * pow((float)d + 0.00, 1.8));
		//FutilityMoveCounts[1][d] = int(2.9 + 1.045 * pow((float)d + 0.49, 1.8));
		FutilityMoveCounts[0][d] = int(PARAM_FUTILITY_MOVE_COUNT_ALPHA0/100.0 + PARAM_FUTILITY_MOVE_COUNT_BETA0 / 1000.0 * pow((float)d + 0.00, 1.8));
		FutilityMoveCounts[1][d] = int(PARAM_FUTILITY_MOVE_COUNT_ALPHA1/100.0 + PARAM_FUTILITY_MOVE_COUNT_BETA1 / 1000.0 * pow((float)d + 0.49, 1.8));
	}

	// razor marginの初期化

	razor_margin[0] = PARAM_RAZORING_MARGIN1;
	razor_margin[1] = PARAM_RAZORING_MARGIN2;
	razor_margin[2] = PARAM_RAZORING_MARGIN3;
	razor_margin[3] = PARAM_RAZORING_MARGIN4;

	// -----------------------
	//   定跡の読み込み
	// -----------------------

	if (book_name != "no_book")
		Book::read_book("book/" + book_name, book, (bool)Options["BookOnTheFly"]);

	// -----------------------
	//   置換表のクリアなど
	// -----------------------
	TT.clear();

	// Threadsが変更になってからisreadyが送られてこないとisreadyでthread数だけ初期化しているものはこれではまずい。
	for (Thread* th : Threads)
	{
		th->history.clear();
		th->counterMoves.clear();
		th->fromTo.clear();
		th->counterMoveHistory.clear();
		th->resetCalls = true;
	}

	Threads.main()->previousScore = VALUE_INFINITE;
}


// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// lazy SMPなので、それぞれのスレッドが勝手に探索しているだけ。
void Thread::search()
{
	// ---------------------
	//      variables
	// ---------------------

	// (ss-5)と(ss+2)にアクセスしたいので余分に確保しておく。
	Stack stack[MAX_PLY + 7], *ss = stack + 4;

	// 先頭8つを初期化しておけば十分。そのあとはsearch()の先頭でss+1,ss+2を適宜初期化していく。
	memset(ss - 4, 0, 7 * sizeof(Stack));

	// aspiration searchの窓の範囲(alpha,beta)
	// apritation searchで窓を動かす大きさdelta
	Value bestValue, alpha, beta, delta;

	// 安定したnodeのときに返す指し手
	Move easyMove = MOVE_NONE;

	// 反復深化のiterationが浅いうちはaspiration searchを使わない。
	// 探索窓を (-VALUE_INFINITE , +VALUE_INFINITE)とする。
	bestValue = delta = alpha = -VALUE_INFINITE;
	beta = VALUE_INFINITE;

	// この初期化は、Thread::MainThread()のほうで行なっている。
	// (この関数を直接呼び出すときには注意が必要)
	completedDepth = DEPTH_ZERO;

	// 将棋所のコンソールが詰まるので出力を抑制するために、前回の出力時刻を
	// 記録しておき、そこから一定時間経過するごとに出力するという方式を採る。
	int lastInfoTime = 0;
	// PVの出力間隔[ms]
	int pv_interval = Options["PvInterval"];

	// もし自分がメインスレッドであるならmainThreadにそのポインタを入れる。
	// 自分がスレーブのときはnullptrになる。
	MainThread* mainThread = (this == Threads.main() ? Threads.main() : nullptr);

	// メインスレッド用の初期化処理
	if (mainThread)
	{
		// 前回の局面からPVの指し手で2手進んだ局面であるかを判定する。
		easyMove = EasyMove.get(rootPos.state()->key());
		EasyMove.clear();
		mainThread->easyMovePlayed = mainThread->failedLow = false;
		mainThread->bestMoveChanges = 0;

		// ponder用の指し手の初期化
		ponder_candidate = MOVE_NONE;

		// --- 置換表のTTEntryの世代を進める。
		TT.new_search();
	}

	// --- MultiPV

	// bestmoveとしてしこの局面の上位N個を探索する機能
	size_t multiPV = Options["MultiPV"];
	// この局面での指し手の数を上回ってはいけない
	multiPV = std::min(multiPV, rootMoves.size());

	// ---------------------
	//   反復深化のループ
	// ---------------------

	// 二度目のrootDepthは深さで探索量を制限するときの条件。main threadのrootDepthがLimits.depthを超えた時点で、
	// salve threadはこのループを抜けて良いので。
	while ((rootDepth+=ONE_PLY) < MAX_PLY
		&& !Signals.stop
		&& (!Limits.depth || Threads.main()->rootDepth/ONE_PLY <= Limits.depth))
	{
		// ------------------------
		// lazy SMPのための初期化
		// ------------------------

		// slaveであれば、これはヘルパースレッドという扱いなので条件を満たしたとき
		// 探索深さを次の深さにする。
		if (!mainThread)
		{
			// これにはhalf density matrixを用いる。
			// 詳しくは、このmatrixの定義部の説明を読むこと。
			// game_ply()は加算すべきではない気がする。あとで実験する。
			const Row& row = HalfDensity[(idx - 1) % HalfDensitySize];
			if (row[(rootDepth/ONE_PLY + rootPos.game_ply()) % row.size()])
				continue;
		}

		// bestMoveが変化した回数を記録しているが、反復深化の世代が一つ進むので、
		// 古い世代の情報として重みを低くしていく。
		if (mainThread)
			mainThread->bestMoveChanges *= 0.505, mainThread->failedLow = false;

		// aspiration window searchのために反復深化の前回のiterationのスコアをコピーしておく
		for (RootMove& rm : rootMoves)
			rm.previousScore = rm.score;

		// MultiPVのためにこの局面の候補手をN個選出する。
		for (PVIdx = 0; PVIdx < multiPV && !Signals.stop; ++PVIdx)
		{
			// ------------------------
			// Aspiration window search
			// ------------------------

			// 探索窓を狭めてこの範囲で探索して、この窓の範囲のscoreが返ってきたらラッキー、みたいな探索。

			// 探索が浅いときは (-VALUE_INFINITE,+VALUE_INFINITE)の範囲で探索する。
			// 探索深さが一定以上あるなら前回の反復深化のiteration時の最小値と最大値
			// より少し幅を広げたぐらいの探索窓をデフォルトとする。

			// この値は 5～10ぐらいがベスト？ Stockfish7では、5 * ONE_PLY。
			if (rootDepth >= 5)
			{
				// aspiration windowの幅
				// 精度の良い評価関数ならばこの幅を小さくすると探索効率が上がるのだが、
				// 精度の悪い評価関数だとこの幅を小さくしすぎると再探索が増えて探索効率が低下する。
				// やねうら王のKPP評価関数では35～40ぐらいがベスト。
				// やねうら王のKPPT(Apery WCSC26)ではStockfishのまま(18付近)がベスト。
				// もっと精度の高い評価関数を用意すべき。
				delta = Value(18);

				alpha = std::max(rootMoves[PVIdx].previousScore - delta, -VALUE_INFINITE);
				beta = std::min(rootMoves[PVIdx].previousScore + delta, VALUE_INFINITE);
			}

			while (true)
			{
				// Stockfish、ここrootDepthにONE_PLY掛けてない。Stockfishのbug。
				bestValue = CheckShogi::search<PV>(rootPos, ss, alpha, beta, rootDepth * ONE_PLY, false);

				// それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
				// 一つ目の指し手以外は-VALUE_INFINITEが返る仕様なので並べ替えのために安定ソートを
				// 用いないと前回の反復深化の結果によって得た並び順を変えてしまうことになるのでまずい。
				std::stable_sort(rootMoves.begin() + PVIdx, rootMoves.end());

				if (Signals.stop)
					break;

				// main threadでfail high/lowが起きたなら読み筋をGUIに出力する。
				// ただし出力を散らかさないように思考開始から3秒経ってないときは抑制する。
				if (mainThread && !Limits.silent
					&& multiPV == 1
					&& (bestValue <= alpha || beta <= bestValue)
					&& Time.elapsed() > 3000
					// 将棋所のコンソールが詰まるのを予防するために出力を少し抑制する。
					&& (rootDepth < 3 || lastInfoTime + pv_interval < Time.elapsed())
					)
				{
					lastInfoTime = Time.elapsed();
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

					// fail lowを起こしていて、いま探索を中断するのは危険。
					if (mainThread)
						mainThread->failedLow = true;
				} else if (bestValue >= beta)
				{
					// fails high

					// fails lowのときと同じ意味の処理。
					alpha = (alpha + beta) / 2;
					beta = std::min(bestValue + delta, VALUE_INFINITE);

				} else {
					// 正常な探索結果なのでこれにてaspiration window searchは終了
					break;
				}

				// delta を等比級数的に大きくしていく
				delta += delta / 4 + 5;

				ASSERT_LV3(-VALUE_INFINITE <= alpha && beta <= VALUE_INFINITE);
			}

			// MultiPVの候補手をスコア順に再度並び替えておく。
			// (二番目だと思っていたほうの指し手のほうが評価値が良い可能性があるので…)
			std::stable_sort(rootMoves.begin(), rootMoves.begin() + PVIdx + 1);

			if (!mainThread)
				continue;

			// メインスレッド以外はPVを出力しない。
			// また、silentモードの場合もPVは出力しない。
			if (!Limits.silent)
			{
				// 停止するときにもPVを出力すべき。(少なくともnode数などは出力されるべき)
				// (そうしないと正確な探索node数がわからなくなってしまう)
				if (Signals.stop ||
					// MultiPVのときは最後の候補手を求めた直後とする。
					// ただし、時間が3秒以上経過してからは、MultiPVのそれぞれの指し手ごと。
					((PVIdx + 1 == multiPV || Time.elapsed() > 3000)
					 && (rootDepth < 3 || lastInfoTime + pv_interval < Time.elapsed())))
				{
					lastInfoTime = Time.elapsed();
					sync_cout << USI::pv(rootPos, rootDepth, alpha, beta) << sync_endl;
				}
			}

		} // multi PV

		  // ここでこの反復深化の1回分は終了したのでcompletedDepthに反映させておく。
		if (!Signals.stop)
			completedDepth = rootDepth;

		if (!mainThread)
			continue;

		// ponder用の指し手として、2手目の指し手を保存しておく。
		// これがmain threadのものだけでいいかどうかはよくわからないが。
		// とりあえず、無いよりマシだろう。
		if (mainThread->rootMoves[0].pv.size() > 1)
			ponder_candidate = mainThread->rootMoves[0].pv[1];

		//
		// main threadのときは探索の停止判定が必要
		//

		// go mateで詰み探索として呼び出されていた場合、その手数以内の詰みが見つかっていれば終了。
		if (Limits.mate
			&& bestValue >= VALUE_MATE_IN_MAX_PLY
			&& VALUE_MATE - bestValue <= Limits.mate)
			Signals.stop = true;

		// 勝ちを読みきっているのに将棋所の表示が追いつかずに、将棋所がフリーズしていて、その間の時間ロスで
		// 時間切れで負けることがある。
		// mateを読みきったとき、そのmateの倍以上、iterationを回しても仕方ない気がするので探索を打ち切るようにする。
		if (!Limits.mate
			&& bestValue >= VALUE_MATE_IN_MAX_PLY
			&& (VALUE_MATE - bestValue) * 2 < rootDepth)
			break;

		// 詰まされる形についても同様。こちらはmateの2倍以上、iterationを回したなら探索を打ち切る。
		if (!Limits.mate
			&& bestValue <= VALUE_MATED_IN_MAX_PLY
			&& (bestValue - (-VALUE_MATE)) * 2 < rootDepth)
			break;

		// 残り時間的に、次のiterationに行って良いのか、あるいは、探索をいますぐここでやめるべきか？
		if (Limits.use_time_management())
		{
			// まだ停止が確定していない
			if (!Signals.stop && !Time.search_end)
			{

				// 1つしか合法手がない(one reply)であるだとか、利用できる時間を使いきっているだとか、
				// easyMoveに合致しただとか…。
				const int F[] = { mainThread->failedLow,
					bestValue - mainThread->previousScore };

				int improvingFactor = std::max(229, std::min(715, 357 + 119 * F[0] - 6 * F[1]));
				double unstablePvFactor = 1 + mainThread->bestMoveChanges;

				auto elapsed = Time.elapsed();

				bool doEasyMove = rootMoves[0].pv[0] == easyMove
					&& mainThread->bestMoveChanges < 0.03
					&& elapsed > Time.optimum() * 5 / 42;

				// bestMoveが何度も変更になっているならunstablePvFactorが大きくなる。
				// failLowが起きてなかったり、1つ前の反復深化から値がよくなってたりするとimprovingFactorが小さくなる。
				if (rootMoves.size() == 1
					|| elapsed > Time.optimum() * unstablePvFactor * improvingFactor / 628
					|| (mainThread->easyMovePlayed = doEasyMove , doEasyMove ))
				{
					// 停止条件を満たした

					// 将棋の場合、フィッシャールールではないのでこの時点でも最小思考時間分だけは
					// 思考を継続したほうが得なので、思考自体は継続して、キリの良い時間になったらcheck_time()にて停止する。

					// ponder中なら、終了時刻はponderhit後から計算して、Time.minimum()。
					if (Limits.ponder)
						Time.search_end = Time.minimum();
					else
					{
						// "ponderhit"しているときは、そこからの経過時間を丸める。
						// "ponderhit"していないときは開始からの経過時間を丸める。
						// そのいずれもTime.elapsed_from_ponderhit()で良い。
						Time.search_end = std::max(Time.round_up(Time.elapsed_from_ponderhit()), Time.minimum());
					}
				}
			}

			// pvが3手以上あるならEasyMoveに記録しておく。
			if (rootMoves[0].pv.size() >= 3)
				EasyMove.update(rootPos, rootMoves[0].pv);
			else
				EasyMove.clear();
		}


	} // iterative deeping

	if (!mainThread)
		return;

	// 最後の反復深化のiterationにおいて、easy moveの候補が安定していないならクリアしてしまう。
	// ifの2つ目の条件は連続したeasy moveを行わないようにするためのもの。
	// (どんどんeasy moveで局面が進むといずれわずかにしか読んでいない指し手を指してしまうため。)
	if (EasyMove.stableCnt < 6 || mainThread->easyMovePlayed)
		EasyMove.clear();

}


// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::think()
{
	// ---------------------
	// 探索パラメーターの自動調整用
	// ---------------------

	param1 = Options["Param1"];
	param2 = Options["Param2"];

	// ---------------------
	// 合法手がないならここで投了
	// ---------------------

	// root nodeにおける自分の手番
	auto us = rootPos.side_to_move();

	// lazy SMPではcompletedDepthを最後に比較するのでこれをゼロ初期化しておかないと
	// 探索しないときにThreads.main()の指し手が選ばれない。
	for (Thread* th : Threads)
		th->completedDepth = 0;

	if (rootMoves.size() == 0)
	{
		// 詰みなのでbestmoveを返す前に読み筋として詰みのスコアを出力すべき。
		if (!Limits.silent)
			sync_cout << "info depth 0 score "
			<< USI::score_to_usi(-VALUE_MATE)
			<< sync_endl;

		// rootMoves.at(0)がbestmoveとしてGUIに対して出力される。
		rootMoves.push_back(RootMove(MOVE_RESIGN));
		goto ID_END;
	}

	// ---------------------
	//     定跡の選択部
	// ---------------------

	{
		// 定跡を用いる手数
		int book_ply = Options["BookMoves"];
		if (rootPos.game_ply() <= book_ply)
		{
			auto it = book.find(rootPos);
			if (it != book.end() && it->second.size() != 0) {
				// 定跡にhitした。逆順で出力しないと将棋所だと逆順にならないという問題があるので逆順で出力する。
				// また、it->second->size()!=0をチェックしておかないと指し手のない定跡が登録されていたときに困る。

				const auto& move_list = it->second;

				// 1) やねうら標準定跡のように評価値なしの定跡DBにおいては
				// 出現頻度の高い順で並んでいることが保証されている。
				// 2) やねうら大定跡のように評価値つきの定跡DBにおいては
				// 手番側から見て評価値の良い順に並んでいることは保証されている。
				// 1),2)から、move_list[0]の指し手がベストの指し手と言える。

				if (!Limits.silent)
				{
					// 将棋所では対応していないが、ShogiGUIの検討モードで使うときに
					// 定跡の指し手に対してmultipvを出力しておかないとうまく表示されないので
					// これを出力しておく。
					auto i = move_list.size();
					for (auto it = move_list.rbegin(); it != move_list.rend(); ++it, --i)
						sync_cout << "info pv " << it->bestMove << " " << it->nextMove
						<< " (" << fixed << setprecision(2) << (100 * it->prob) << "%)" // 採択確率
						<< " score cp " << it->value << " depth " << it->depth
						<< " multipv " << i << sync_endl;
				}

				// このなかの一つをランダムに選択

				// 評価値ベースで選ぶのでないなら、
				// 無難な指し手が選びたければ、採択回数が一番多い、最初の指し手(move_list[0])を選ぶべし。
				// 評価値ベースで選ぶときは、NarrowBookはオンにすべきではない。

				// 狭い定跡を用いるのか？
				bool narrowBook = Options["NarrowBook"];
				size_t book_move_max = move_list.size();
				if (narrowBook)
				{
					// 出現確率10%未満のものを取り除く。
					for (size_t i = 0; i < move_list.size(); ++i)
					{
						if (move_list[i].prob < 0.1)
						{
							book_move_max = max(i, size_t(1));
							// 定跡から取り除いたことをGUIに出力
							if (!Limits.silent)
								sync_cout << "info string narrow book moves to " << book_move_max << " moves." << sync_endl;
							break;
						}
					}
				}

				// 評価値の差などを反映。
				if (book_move_max)
				{
					// 定跡として採用するdepthの下限。0 = 無視。
					auto depth_limit = (int)Options["BookDepthLimit"];
					if (depth_limit != 0 && move_list[0].depth < depth_limit)
					{
						sync_cout << "info string BookDepthLimit is lower than the depth of this node." << sync_endl;
						book_move_max = 0;
					} else {
						// ベストな評価値の候補手から、この差に収まって欲しい。
						auto eval_diff = (int)Options["BookEvalDiff"];
						auto value_limit1 = move_list[0].value - eval_diff;
						// 先手・後手の評価値下限の指し手を採用するわけにはいかない。
						auto stm_string = (rootPos.side_to_move() == BLACK) ? "BookEvalBlackLimit" : "BookEvalWhiteLimit";
						auto value_limit2 = (int)Options[stm_string];
						auto value_limit = max(value_limit1, value_limit2);

						for (size_t i = 0; i < book_move_max; ++i)
						{
							if (move_list[i].value < value_limit)
							{
								// 候補手が減った理由を出力
								if (value_limit1 == value_limit)
									sync_cout << "info string BookEvalDiff = " << eval_diff << " , moves to " << i << " moves." << sync_endl;
								else
									sync_cout << "info string " << stm_string << " = " << value_limit2 << " , moves to " << i << " moves." << sync_endl;

								book_move_max = i;
								break;
							}
						}
					}
				}

				if (book_move_max)
				{
					// 不成の指し手がRootMovesに含まれていると正しく指せない。
					const auto& move = move_list[prng.rand(book_move_max)];
					auto bestMove = move.bestMove;
					auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
					if (it_move != rootMoves.end())
					{
						std::swap(rootMoves[0], *it_move);

						// 2手目の指し手も与えないとponder出来ない。
						if (move.nextMove != MOVE_NONE)
						{
							if (rootMoves[0].pv.size() <= 1)
								rootMoves[0].pv.push_back(MOVE_NONE);
							rootMoves[0].pv[1] = move.nextMove; // これが合法手でなかったら将棋所が弾くと思う。
						}
						goto ID_END;
					}
				}
				// 合法手のなかに含まれていなかった、もしくは定跡として選ばれる条件を満たさなかったので
				// 定跡の指し手は指さない。
			}
		}
	}

	// ---------------------
	//    宣言勝ち判定
	// ---------------------

	{
		// 宣言勝ちもあるのでこのは局面で1手勝ちならその指し手を選択
		// 王手がかかっていても、回避しながらトライすることもあるので王手がかかっていようが
		// Position::DeclarationWin()で判定して良い。
		auto bestMove = rootPos.DeclarationWin();
		if (bestMove != MOVE_NONE)
		{
			// 宣言勝ちなのでroot movesの集合にはないかも知れない。強制的に書き換える。
			rootMoves[0] = RootMove(bestMove);
			goto ID_END;
		}
	}

	// ---------------------
	//    通常の思考処理
	// ---------------------

	{
		StateInfo si;
		auto& pos = rootPos;

		// -- MAX_PLYに到達したかの判定が面倒なのでLimits.max_game_plyに一本化する。

		// あとで戻す用
		auto max_game_ply = Limits.max_game_ply;
		Limits.max_game_ply = std::min(Limits.max_game_ply, rootPos.game_ply() + MAX_PLY - 1);

		// --- contempt factor(引き分けのスコア)

		// Contempt: 引き分けを受け入れるスコア。歩を100とする。例えば、この値を100にすると引き分けの局面は
		// 評価値が - 100とみなされる。(互角と思っている局面であるなら引き分けを選ばずに他の指し手を選ぶ)

		int contempt = Options["Contempt"] * PawnValue / 100;
		drawValueTable[REPETITION_DRAW][us] = VALUE_ZERO - Value(contempt);
		drawValueTable[REPETITION_DRAW][~us] = VALUE_ZERO + Value(contempt);

		// --- 今回の思考時間の設定。

		Time.init(Limits, us, rootPos.game_ply());

		// ---------------------
		// 各スレッドがsearch()を実行する
		// ---------------------

		for (Thread* th : Threads)
		{
			th->maxPly = 0;
			th->rootDepth = 0;
			if (th != this)
				th->start_searching();
		}

		Thread::search();

		// 復元する。
		Limits.max_game_ply = max_game_ply;
	}

	// 反復深化の終了。
ID_END:;

	// nodes as time(時間としてnodesを用いるモード)のときは、利用可能なノード数から探索したノード数を引き算する。
	if (Limits.npmsec)
		Time.availableNodes = std::max(Time.availableNodes + Limits.inc[us] - (s64)Threads.nodes_searched(), (s64)0);

	// 最大depth深さに到達したときに、ここまで実行が到達するが、
	// まだSignals.stopが生じていない。しかし、ponder中や、go infiniteによる探索の場合、
	// USI(UCI)プロトコルでは、"stop"や"ponderhit"コマンドをGUIから送られてくるまで
	// best moveを出力すべきではない。
	// それゆえ、単にここでGUIからそれらのいずれかのコマンドが送られてくるまで待つ。
	if (!Signals.stop && (Limits.ponder || Limits.infinite))
	{
		// "stop"が送られてきたらSignals.stop == trueになる。
		// "ponderhit"が送られてきたらLimits.ponder == 0になるので、それを待つ。(stopOnPonderhitは用いない)
		//    また、このときSignals.stop == trueにはならない。(この点、Stockfishとは異なる。)
		// "go infinite"に対してはstopが送られてくるまで待つ。
		while (!Signals.stop && (Limits.ponder || Limits.infinite))
			sleep(1);
		//	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
		// (思考のためには計算資源を使っていないので。)
	}

	Signals.stop = true;

	// 各スレッドが終了するのを待機する(開始していなければいないで構わない)
	for (Thread* th : Threads.slaves)
		th->wait_for_search_finished();

	// ---------------------
	// lazy SMPの結果を取り出す
	// ---------------------

	Thread* bestThread = this;

	// 並列して探索させていたスレッドのうち、ベストのスレッドの結果を選出する。
	if (Options["MultiPV"] == 1
		&& !Limits.depth
		&& rootMoves[0].pv[0] != MOVE_NONE)
	{
		// 深くまで探索できていて、かつそっちの評価値のほうが優れているならそのスレッドの指し手を採用する
		// 単にcompleteDepthが深いほうのスレッドを採用しても良さそうだが、スコアが良いほうの探索深さのほうが
		// いい指し手を発見している可能性があって楽観合議のような効果があるようだ。
		for (Thread* th : Threads)
			if (th->completedDepth > bestThread->completedDepth
				&& th->rootMoves[0].score > bestThread->rootMoves[0].score)
				bestThread = th;
	}

	// 次回の探索のときに何らか使えるのでベストな指し手の評価値を保存しておく。
	previousScore = bestThread->rootMoves[0].score;

	// ベストな指し手として返すスレッドがmain threadではないのなら、その読み筋は出力していなかったはずなので
	// ここで読み筋を出力しておく。
	if (bestThread != this && !Limits.silent)
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

#ifdef EVAL_LEARN

namespace Learner
{
	// 学習用に、1つのスレッドからsearch,qsearch()を呼び出せるようなスタブを用意する。

	// 学習のための初期化。
	// Learner::search(),Learner::qsearch()から呼び出される。
	void init_for_search(Position pos)
	{
		// Search::Limitsに関して
		{
			auto& limits = Search::Limits;

			// 探索を"go infinite"コマンド相当にする。(time managementされると困るため)
			limits.infinite = true;

			// PVを表示されると邪魔なので消しておく。
			limits.silent = true;

			// ここでLimits.max_game_ply破壊するけど、gensfenのとき用だから復元せずともまあいいか…。
			limits.max_game_ply = pos.game_ply() + MAX_PLY - 1;
		}

		// this_threadに関して。
		{
			auto th = pos.this_thread();

			th->completedDepth = 0;
			th->maxPly = 0;
			th->rootDepth = 0;

#if 0
			// 余裕があるならhistory等もクリアしておく。
			// したほうがいいかは微妙だが…。
			th->history.clear();
			th->counterMoves.clear();
			th->fromTo.clear();
			//	th->counterMoveHistory.clear();
			// →　このクリア、時間がかかりすぎるのでまあいいや。
#endif
			th->resetCalls = true;

			// rootMovesの設定
			auto& rootMoves = th->rootMoves;

			rootMoves.clear();
			for (auto m : MoveList<LEGAL>(pos))
				rootMoves.push_back(Search::RootMove(m));

			ASSERT_LV3(rootMoves.size() != 0);
		}

		// DrawValueの設定
		{
			Color us = pos.side_to_move();
			int contempt = Options["Contempt"] * PawnValue / 100;
			drawValueTable[REPETITION_DRAW][us] = VALUE_ZERO - Value(contempt);
			drawValueTable[REPETITION_DRAW][~us] = VALUE_ZERO + Value(contempt);
		}
	}

	// 静止探索。
	// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。
	// 引数でalpha,betaを指定できるようにしていたが、これがその窓で探索したときの結果を
	// 置換表に書き込むので、その窓に対して枝刈りが出来るような値が書き込まれて学習のときに
	// 悪い影響があるので、窓の範囲を指定できるようにするのをやめることにした。
	pair<Value, vector<Move> > qsearch(Position& pos)
	{
		Stack stack[MAX_PLY + 7], *ss = stack + 4;
		memset(ss - 4, 0, 7 * sizeof(Stack));
		Move pv[MAX_PLY + 1];
		ss->pv = pv; // とりあえずダミーでどこかバッファがないといけない。

		init_for_search(pos);
		auto th = pos.this_thread();

		// 現局面で王手がかかっているかで場合分け。
		const bool inCheck = pos.in_check();
		auto bestValue = inCheck ?
			CheckShogi::qsearch<PV, true >(pos, ss, -VALUE_INFINITE, VALUE_INFINITE, DEPTH_ZERO) :
			CheckShogi::qsearch<PV, false>(pos, ss, -VALUE_INFINITE, VALUE_INFINITE, DEPTH_ZERO);

		// 得られたPVを返す。
		vector<Move> pvs;
		for (Move* p = &ss->pv[0]; is_ok(*p); ++p)
			pvs.push_back(*p);

		return pair<Value, vector<Move> >(bestValue, pvs);
	}

	// 通常探索。深さdepth(整数で指定)。
	// 3手読み時のスコアが欲しいなら、
	//   auto v = search(pos,3);
	// のようにすべし。
	// v.firstに評価値、v.secondにPVが得られる。
	// MultiPVが有効のときは、pos.this_thread()->rootMoves[N].pvにそのPV(読み筋)の配列が得られる。
	// 前提条件) pos.set_this_thread(Threads[thread_id])で探索スレッドが設定されていること。

	pair<Value, vector<Move> > search(Position& pos, int depth)
	{
		if (depth < DEPTH_ZERO)
			return pair<Value, vector<Move>>(Eval::evaluate(pos), vector<Move>());

		if (depth == DEPTH_ZERO)
			return qsearch(pos);

		Stack stack[MAX_PLY + 7], *ss = stack + 4;
		memset(ss - 4, 0, 7 * sizeof(Stack));
		Move pv[MAX_PLY + 1];
		ss->pv = pv; // とりあえずダミーでどこかバッファがないといけない。

		init_for_search(pos);

		// this_threadに関連する変数の初期化
		auto th = pos.this_thread();
		auto& rootDepth = th->rootDepth;
		auto& PVIdx = th->PVIdx;
		auto& rootMoves = th->rootMoves;

		// bestmoveとしてしこの局面の上位N個を探索する機能
		size_t multiPV = Options["MultiPV"];
		// この局面での指し手の数を上回ってはいけない
		multiPV = std::min(multiPV, rootMoves.size());

		Value alpha = -VALUE_INFINITE;
		Value beta = VALUE_INFINITE;
		Value delta = -VALUE_INFINITE;
		Value bestValue = -VALUE_INFINITE;

		while (++rootDepth <= depth)
		{
			for (RootMove& rm : rootMoves)
				rm.previousScore = rm.score;

			// MultiPV
			for (PVIdx = 0; PVIdx < multiPV && !Signals.stop; ++PVIdx)
			{
				// depth 5以上においてはaspiration searchに切り替える。
				if (rootDepth >= 5)
				{
					delta = Value(18);

					Value p = rootMoves[PVIdx].previousScore;

					alpha = std::max(p - delta, -VALUE_INFINITE);
					beta = std::min(p + delta, VALUE_INFINITE);
				}

				// aspiration search
				while (true)
				{
					bestValue = CheckShogi::search<PV>(pos, ss, alpha, beta, rootDepth * ONE_PLY, false);
					std::stable_sort(rootMoves.begin() + PVIdx, rootMoves.end());

					// fail low/highに対してaspiration windowを広げる。
					// ただし、引数で指定されていた値になっていたら、もうfail low/high扱いとしてbreakする。
					if (bestValue <= alpha)
					{
						beta = (alpha + beta) / 2;
						alpha = std::max(bestValue - delta, -VALUE_INFINITE);
					} else if (bestValue >= beta)
					{
						alpha = (alpha + beta) / 2;
						beta = std::min(bestValue + delta, VALUE_INFINITE);
					} else {
						break;
					}
					delta += delta / 4 + 5;
					ASSERT_LV3(-VALUE_INFINITE <= alpha && beta <= VALUE_INFINITE);
				}

				std::stable_sort(rootMoves.begin(), rootMoves.begin() + PVIdx + 1);

			} // multi PV
		}

		// このPV、途中でNULL_MOVEの可能性があるかも知れないので排除するためにis_ok()を通す。
		vector<Move> pvs;
		for (Move move : rootMoves[0].pv)
		{
			if (!is_ok(move))
				break;
			pvs.push_back(move);
		}

		// 引数で指定されたalpha,betaの範囲外の値を返すときは、fail low/highなので、
		// PVが正しい保証はない。

		// multiPV時を考慮して、rootMoves[0]のscoreをbestValueとして返す。
		bestValue = rootMoves[0].score;

		return pair<Value, vector<Move> >(bestValue, pvs);
	}

}
#endif

#endif // CHECK_SHOGI_ENGINE

