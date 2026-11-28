#include "config.h"

#if defined(USE_TIME_MANAGEMENT)

#include "timeman.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "search.h"
#include "types.h"
#include "usioption.h"

#if !STOCKFISH
#include "usi.h"
#endif

namespace YaneuraOu {

TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }

#if STOCKFISH
void TimeManagement::clear() {
    availableNodes    = -1;  // When in 'nodes as time' mode
    // nodes-as-timeモードの未初期化状態。
    previousMovesToGo = 0;
}

void TimeManagement::advance_nodes_time(i64 nodes) {
    assert(useNodesTime);
    availableNodes = std::max(i64(0), availableNodes - nodes);
}

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
// 探索開始時に、その手番で使用する思考時間の範囲を計算する。
// 対応する時間制御は、基本持ち時間＋加算、または一定手数ごとの持ち時間＋加算。
void TimeManagement::init(Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options,
                          double&             originalTimeAdjust) {
    TimePoint npmsec = TimePoint(options["nodestime"]);

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    // 持ち時間がなければ時間管理の完全な初期化は不要。
    // startTimeはmovetimeで、useNodesTimeはelapsedの呼び出しで使用する。
    startTime    = limits.startTime;
    useNodesTime = npmsec != 0;

    if (useNodesTime)
        limits.movetime *= npmsec;

    if (limits.time[us] == 0)
    {
        optimumTime = maximumTime = NoBound;
        return;
    }

    TimePoint moveOverhead = TimePoint(options["Move Overhead"]);

    // optScale is a percentage of available time to use for the current move.
    // maxScale is a multiplier applied to optimumTime.
    // optScaleは残り時間のうち今回の指し手に使う割合。
    // maxScaleは最適思考時間に掛ける倍率。
    double optScale, maxScale;

    // If we have to play in 'nodes as time' mode, then convert from time
    // to nodes, and use resulting values in time management formulas.
    // WARNING: to avoid time losses, the given npmsec (nodes per millisecond)
    // must be much lower than the real engine speed.
    // nodes-as-timeモードでは時間をノード数に換算し、時間管理の式に使用する。
    // 時間切れを避けるには、npmsecを実際の探索速度より十分低く設定する必要がある。
    if (useNodesTime)
    {
        if (availableNodes == -1)  // Only once at game start
        // 対局開始時の一度だけ初期化する。
        {
            // First time limit includes increment (both are in milliseconds)
            // 最初の持ち時間には加算時間も含まれる。いずれもミリ秒単位。
            availableNodes = npmsec * limits.time[us];
            cyclicBudget   = npmsec * (limits.time[us] - limits.inc[us]);
        }
        else if (limits.movestogo > 0 && limits.movestogo > previousMovesToGo && cyclicBudget > 0)
            availableNodes += cyclicBudget;

        previousMovesToGo = limits.movestogo;

        // Convert from milliseconds to nodes
        // ミリ秒をノード数に換算する。
        limits.time[us] = TimePoint(availableNodes);
        limits.inc[us] *= npmsec;
        limits.npmsec = npmsec;
        moveOverhead *= npmsec;
    }

    // These numbers are used where multiplications, divisions,
    // or comparisons with constants are involved.
    // 定数との乗算・除算・比較に用いるスケールと、ミリ秒に戻した持ち時間。
    const i64       scaleFactor = useNodesTime ? npmsec : 1;
    const TimePoint scaledTime  = std::max(TimePoint(1), limits.time[us] / scaleFactor);

    // Maximum move horizon
    // 時間配分で考慮する残り手数の上限。
    int mtg = limits.movestogo ? std::min(limits.movestogo, 50) : 50;

    // If less than one second, gradually reduce mtg.
    // In cyclic time controls we keep the actual movestogo as horizon.
    // 残り1秒未満ではmtgを徐々に減らす。
    // 一定手数ごとの時間制御では、実際のmovestogoを用いる。
    if (scaledTime < 1000 && limits.movestogo == 0)
        mtg = int(scaledTime * 0.05);

    // Make sure timeLeft is > 0 since we may use it as a divisor
    // 除数として使用するため、timeLeftを正に保つ。
    TimePoint timeLeft = std::max(TimePoint(1), limits.time[us] + limits.inc[us] * (mtg - 1)
                                                  - moveOverhead * (2 + mtg));

    // x basetime (+ z increment)
    // If there is a healthy increment, timeLeft can exceed the actual available
    // game time for the current move, so also cap to a percentage of available game time.
    // 基本持ち時間x（＋1手ごとの加算時間z）。
    // 加算が十分多いとtimeLeftが実際の残り時間を超えるため、使用割合にも上限を設ける。
    if (limits.movestogo == 0)
    {
        // Extra time according to timeLeft
        // timeLeftに応じて思考時間を調整する。
        if (originalTimeAdjust < 0)
            originalTimeAdjust = 0.3272 * std::log10(timeLeft) - 0.4141;

        // Calculate time constants based on current time left.
        // 現在の残り時間に基づいて時間配分の定数を計算する。
        double logTimeInSec = std::log10(scaledTime / 1000.0);
        double optConstant  = std::min(0.0029869 + 0.00033554 * logTimeInSec, 0.004905);
        double maxConstant  = std::max(3.3744 + 3.0608 * logTimeInSec, 3.1441);

        optScale = std::min(0.012112 + std::pow(ply + 3.22713, 0.46866) * optConstant,
                            0.19404 * limits.time[us] / timeLeft)
                 * originalTimeAdjust;

        maxScale = std::min(6.873, maxConstant + ply / 12.352);
    }

    // x moves in y seconds (+ z increment)
    // y秒以内にx手指す（＋1手ごとの加算時間z）。
    else
    {
        optScale = std::min((0.88 + ply / 116.4) / mtg, 0.88 * limits.time[us] / timeLeft);
        maxScale = 1.3 + 0.11 * mtg;
    }

    // Decrease time usage if behind in time.
    // This is skipped in two cases:
    // - if the nodestime option is used we can't calculate the opponent nodes budget in a deterministic way.
    // - if we use a cyclic time management (like 40/10) calculating time advantage for the last move (movestogo = 1)
    //   can be vastly off, because if the opponent had done his last move before us his time budget includes already
    //   the next cycle time increment but our not. This leads to a unnecessary big decrease in time usage which favors blunders.
    // Warning: don't remove this conditions.
    // 相手より残り時間が少なければ使用時間を減らす。
    // ただし、相手のノード予算を決定論的に計算できないnodes-as-timeでは行わない。
    // また、一定手数ごとの時間制御でmovestogoが1のときは、相手だけが次の周期の
    // 持ち時間を加算済みの場合があり、不必要な時間短縮で悪手を誘発するので行わない。
    // これらの条件を取り除かないこと。
    if (!useNodesTime && limits.movestogo != 1)
    {
        double timeAdvantage =
          (limits.time[us] - limits.time[~us]) / (1.0 + limits.time[us] + limits.time[~us]);
        optScale *= 1 + 0.9 * std::min(timeAdvantage, 0.0);
    }

    // Limit the maximum possible time for this move
    // 今回の指し手に使用できる時間の上限を設定する。
    optimumTime = TimePoint(std::max(1.0, optScale * timeLeft));
    maximumTime =
      TimePoint(std::max(double(optimumTime), std::min(0.8097 * limits.time[us] - moveOverhead,
                                                       maxScale * optimumTime)));

    if (options["Ponder"])
        optimumTime += optimumTime / 4;
}

#else
TimePoint TimeManagement::minimum() const { return minimumTime; }
void  TimeManagement::clear() {}

namespace {

// 時間配分に使う残り手数の見積もりの基準値。手数は先後両方の指し手を数える。
// 局面の手数や時間制御に応じて補正し、さらに2で割って自分の残り手番を見積もる。
const int MoveHorizon = 160;

// 思考時間のrtimeが指定されたときに用いる乱数
PRNG prng;

} // namespace


// 起動時に呼び出す。
// このclassが使用するengine optionを追加する。
void TimeManagement::add_options(OptionsMap& options) {

		// 指し手がGUIに届くまでの時間。
#if defined(YANEURAOU_ENGINE_DEEP)
    // GPUからの結果を待っている時間も込みなのでdefault値を少し上げておく。
    int time_margin = 400;
#else
    int time_margin = 120;
#endif

    // ネットワークの平均遅延時間[ms]
    // この時間だけ早めに指せばだいたい間に合う。
    // 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
    options.add("NetworkDelay", Option(time_margin, 0, 10000));

    // ネットワークの最大遅延時間[ms]
    // 切れ負けの瞬間だけはこの時間だけ早めに指す。
    // 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
    options.add("NetworkDelay2", Option(time_margin + 1000, 0, 10000));

    // 最小思考時間[ms]
    options.add("MinimumThinkingTime", Option(2000, 1, 100000));

    // 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
    // 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
    // 対人のときに短めに設定して強制的に早指しにすることが出来る。
    options.add("SlowMover", Option(100, 1, 1000));

	// 持ち時間、各秒のギリギリまで使うか。
    options.add("RoundUpToFullSecond", true);

}


// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)

// 探索の開始時に呼び出され、現在の手番で使用できる思考時間の
// 上限と下限を計算する。現在、以下の時間制御に対応している。
//      1) 基本持ち時間 x（＋ 1手ごとの加算時間 z）
//      2) y秒以内にx手指す（＋ 1手ごとの加算時間 z）

void TimeManagement::init(const Search::LimitsType& limits,
                          Color                     us,
                          int                       ply,
                          const OptionsMap&         options,
                          int                       max_moves_to_draw) {

    // 今回の思考時間を計算してoptimum()・maximum()へ反映する。
    // plyはここまでの手数で、平手の初期局面なら1（0ではない）。

	// ネットワークのDelayを考慮して少し減らすべき。
	// かつ、minimumとmaximumは端数をなくすべき
    network_delay = (TimePoint) options["NetworkDelay"];

	// 秒未満を切り上げるのか
	round_up_to_fullsecond = options["RoundUpToFullSecond"];

	// 探索開始時刻をコピーする。elapsed_time()はここからの経過時間を返す。
	// 探索開始時にはponderhit時刻も同じ値で初期化する。
	// 終了時刻は0ならば未確定という意味である。
    startTime = ponderhitTime = limits.startTime;
    search_end                = 0;
    // rtimeやmovetimeなどの早期returnでも、前回の秒読み状態を持ち越さない。
    isFinalPush               = false;

	// 今回の最大残り時間(これを超えてはならない)
	// byoyomiの指定は残り時間にこの時点で加算して考える。
	// inctimeは、この指し手のあとに加算される時間であるから、ここでは足さない。
    remain_time = limits.time[us] + limits.byoyomi[us] - (TimePoint) options["NetworkDelay2"];

	// remain_timeを0にすると時間切れのあと自爆するのでとりあえず100はあることにしておく。
	// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
	remain_time = std::max(remain_time, (TimePoint) (round_up_to_fullsecond ? 100 : 1) );

	// 最小思考時間
    minimum_thinking_time = (TimePoint) options["MinimumThinkingTime"];

	// 序盤重視率
	// 　これはこんなパラメーターとして手で調整するべきではなく、探索パラメーターの一種として
	//   別の方法で調整すべき。ただ、対人でソフトに早指ししたいときには意味があるような…。
	int slowMover = (int)options["SlowMover"];

	if (limits.rtime)
	{
		// これが指定されているときは思考時間をランダム化する。
		// 連続自己対戦時に同じ進行になるのを回避するためのもの。
		// 終盤で大きく持ち時間を変えると、勝率が5割に寄ってしまうのでそれには注意。

		auto r = limits.rtime;
		// 指し手が進むごとに減衰していく曲線にする。
        if (ply > 0)
        {
            const auto width = (int)std::min(r * 0.5f, r * 10.0f / ply);
            // 短時間や終盤では幅が0になる。その場合は乱数を加算しない。
            if (width > 0)
                r += (int)prng.rand(width);
        }
		remain_time = minimumTime = optimumTime = maximumTime = r;
		return;
	}

	// 時間固定モード
	// "go movetime 100"のようにして思考をさせた場合。
	if (limits.movetime)
	{
		remain_time = minimumTime = optimumTime = maximumTime = limits.movetime;
		return;
	}

	// 切れ負けであるか？
	bool time_forfeit = limits.inc[us] == 0 && limits.byoyomi[us] == 0;

	// 1. 切れ負けルールの時は、MoveHorizonを + 40して考える。
	// 2. ゲーム開始直後～40手目ぐらいまでは定跡で進むし、そこまで進まなかったとしても勝負どころはそこではないので
	// 　ゲーム開始直後～40手目付近のMoveHorizonは少し大きめに考える必要がある。逆に40手目以降、MoveHorizonは40ぐらい減らして考えていいと思う。
	// 3. 切れ負けでないなら、100手時点で残り60手ぐらいのつもりで指していいと思う。(これくらいしないと勝負どころすぎてからの持ち時間が余ってしまう..)
	// (現在の大会のフィッシャールールは15分+inctime5秒とか5分+inctime10秒そんな感じなので、160手目ぐらいで持ち時間使い切って問題ない)
	int move_horizon;
	if (time_forfeit)
		move_horizon = MoveHorizon + 40 - std::min(ply , 40);
	else
		// + 20は調整項
		move_horizon = MoveHorizon + 20 - std::min(ply , 80);

	// 残りの自分の手番の回数
	// ⇨　plyは平手の初期局面が1。256手ルールとして、max_moves_to_draw == 256だから、256手目の局面においてply == 256
	// 　その1手前の局面においてply == 255。ply == 255 or 256のときにMTGが1にならないといけない。だから2足しておくのが正解。
    const int MTG = std::min(max_moves_to_draw - ply + 2, move_horizon) / 2;

	if (MTG <= 0)
	{
		// 本来、終局までの最大手数が指定されているわけだから、この条件で呼び出されるはずはないのだが…。
		sync_cout << "info string Error! : MaxMovesToDraw is too small." << sync_endl;
		// 事故防止のために何か設定はしておく。
        minimumTime = optimumTime = maximumTime = 500;
        return;
	}
	if (MTG == 1)
	{
		// 時間配分上、残りの自分の手番を1回と見積もるので使い切る。
		minimumTime = optimumTime = maximumTime = remain_time;
		return;
	}

	// minimumとoptimumな時間を適当に計算する。

	{
		// 最小思考時間(これが1000より短く設定されることはないはず..)
		// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
		minimumTime = std::max(minimum_thinking_time - network_delay,
							   (TimePoint)(round_up_to_fullsecond ? 1000 : 1));

		// 最適思考時間と、最大思考時間には、まずは上限値を設定しておく。
		optimumTime = maximumTime = remain_time;

		// optimumTime = min ( minimumTime + α     , remain_time)
		// maximumTime = min ( minimumTime + α * 5 , remain_time)
		// みたいな感じで考える

		// 残り手数において残り時間はあとどれくらいあるのか。
		TimePoint remain_estimate = limits.time[us]
			+ limits.inc[us] * MTG
			// 秒読み時間も残り手数に付随しているものとみなす。
			+ limits.byoyomi[us] * MTG;

		// 秒単位切り捨てルールでは、各手で1秒未満を使う前提で予約しておく。
		if (round_up_to_fullsecond)
			remain_estimate -= (MTG + 1) * 1000;
		remain_estimate = std::max(remain_estimate, TimePoint(0));

		// -- optimumTime
		TimePoint t1 = minimumTime + remain_estimate / MTG;

		// -- maximumTime

		// 5.0fでもうまくマネージメントできる。
		// 📝 上手くタイムマネージメントできない探索部なら3.0fぐらいにするのが無難。
		float max_ratio = 5.0f;

		// 切れ負けルールにおいては、5分を切っていたら、このratioを抑制する。
		if (time_forfeit)
		{
			// 3分     : ratio = 3.0
			// 2分     : ratio = 2.0
			// 1分以下 : ratio = 1.0固定
			max_ratio = std::min(max_ratio, std::max(float(limits.time[us]) / (60 * 1000), 1.0f));
		}
		TimePoint t2 = minimumTime + (int)(remain_estimate * max_ratio / MTG);

		// ただしmaximumは、加算時間・秒読みを含むremain_estimateの30%以下に抑える。
		// optimumが超える分にはいい。それは残り手数が少ないときとかなので構わない。
		t2 = std::min(t2, (TimePoint)(remain_estimate * 0.3));

		// slowMoverは100分率で与えられていて、optimumTimeの係数として作用するものとする。
		optimumTime = std::min(t1, optimumTime) * slowMover / 100;
		maximumTime = std::min(t2, maximumTime);

		// Ponderが有効でStochastic_Ponderが無効の場合、
		// ponderhitすると時間が本来の予測より余っていくので思考時間を心持ち多めにとっておく。
        if (options["USI_Ponder"] && !options["Stochastic_Ponder"])
			optimumTime += optimumTime / 4;

	}

	// 秒読みモードでかつ、持ち時間がないなら、最小思考時間も最大思考時間もその時間にしたほうが得
	if (limits.byoyomi[us])
	{
		// 持ち時間が少ないなら(秒読み時間の1.2倍未満なら)、思考時間を使いきったほうが得
		// これには持ち時間がゼロのケースも含まれる。
        if (limits.time[us] < (int) (limits.byoyomi[us] * 1.2))
        {
            minimumTime = optimumTime = maximumTime = limits.byoyomi[us] + limits.time[us];
			// "ponderhit"の時刻から数えてminimumTime分は使って欲しい。
            isFinalPush                             = true;
        }
	}

	// 残り時間 - network_delay2よりは短くしないと切れ負けになる可能性が出てくる。
	minimumTime = std::min(round_up(minimumTime), remain_time);
	optimumTime = std::min(         optimumTime , remain_time);
	maximumTime = std::min(round_up(maximumTime), remain_time);
}


// 1秒単位で繰り上げてdelayを引く。
// ただし、remain_timeよりは小さくなるように制限する。
TimePoint TimeManagement::round_up(TimePoint t0) {
	if (round_up_to_fullsecond)
	{
		// 📓 秒未満を切り上げる時

		// 1000で繰り上げる。Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(((t0 + 999) / 1000) * 1000, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// これが元の値より小さいなら、もう1秒使わないともったいない。
		if (t < t0)
			t += 1000;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;

	} else {
		// 📓 秒未満を切り上げない時

		// Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(t0, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;
	}
};

// 探索を終了させることが確定しているが、秒単位で切り上げて、search_endにそれを設定したい時に呼び出す。
void TimeManagement::set_search_end(TimePoint e) {
    /*
		🤔 現在時刻から、ponderhitした時刻から計算して、秒単位で切り上げた時刻まで思考させたい。

		📓
		    1. 使用した時間の計測はponderhitした時刻からの経過時間としてなされるために、
			   ponderhitからの経過時間を秒単位で切り上げしたい。
		       そのためには、round_up(elapsed_from_ponderhit())のようにround upする必要がある。

			2. 一方、"go"した時刻から計算して、tm.minimum()の分は思考することを守らせたい。
			   しかし、ponderhitからの経過時間で切り上げはしたい。

			   しかし、秒読みで秒まで使い切りたいとき(isFinalPush)は、
			   "ponderhit"から数えてtm.minimum()の分は思考しないともったいない。
	*/

    // 1. ponderhitからの経過時間。(go ponder～ponderhitしていない場合は、単にgoからの経過時間)
    TimePoint t1 = e + startTime - ponderhitTime;

	// 2. "go"した時刻からminimum()を足して、ponderhitからの経過時間に変換したもの。
	//    ただし、finalpushであるなら、"ponderhit"した時刻から数える。
    TimePoint t2 = isFinalPush ? minimum() : minimum() + startTime - ponderhitTime;

	// t1,t2の大きいほうを秒単位で切り上げて、それをstartTimeからの経過時間に換算したもの。
    // 💡 search_endの値は、startTimeからの経過時間なので。
    search_end = round_up(std::max(t1, t2)) + ponderhitTime - startTime;
}

#endif

} // namespace YaneuraOu

#endif // defined(USE_TIME_MANAGEMENT)
