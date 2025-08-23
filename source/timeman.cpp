#include "config.h"

#if defined(USE_TIME_MANAGEMENT)

#include "misc.h"
#include "search.h"
#include "thread.h"
#include "usi.h"
#include "timeman.h"

namespace YaneuraOu {

namespace {

// これぐらい自分が指すと終局すると考えて計画を練る。
// 近年、将棋ソフトは終局までの平均手数が伸びているので160に設定しておく。
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
    options.add("MinimumThinkingTime", Option(2000, 1000, 100000));

    // 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
    // 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
    // 対人のときに短めに設定して強制的に早指しにすることが出来る。
    options.add("SlowMover", Option(100, 1, 1000));
}

void TimeManagement::init(const Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options
#if STOCKFISH
                          ,
                          double& originalTimeAdjust
// 💡 やねうら王では使わないことにする。
#else
                          ,
                          int max_moves_to_draw
#endif

) {

    // 📝 探索開始時刻をコピーしておく。
    //     以降、elapsed_time()は、ここからの経過時間を返す。
    startTime = limits.startTime;

	// TODO あとで
#if 0
    // reinit()が呼び出された時のために呼び出し条件を保存しておく。
    lastcall_Limits = &limits;
    lastcall_Us     = us;
    lastcall_Ply    = ply;
    lastcall_Opt    = const_cast<OptionsMap*>(&options);
#endif

    init_(limits, us, ply, options, max_moves_to_draw);
}

// 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
// これは探索の開始時に呼び出されて、今回の指し手のための思考時間を計算する。
// limitsで指定された条件に基いてうまく計算する。
// ply : ここまでの手数。平手の初期局面なら1。(0ではない)
void TimeManagement::init_(const Search::LimitsType& limits,
                           Color               us,
                           int                 ply,
                           const OptionsMap&   options, int max_moves_to_draw) {

#if STOCKFISH
	TimePoint npmsec = TimePoint(options["nodestime"]);
	// nodes as timeモード。やねうら王では用いない。

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    startTime    = limits.startTime;

	useNodesTime = npmsec != 0;

    if (limits.time[us] == 0)
        return;

	TimePoint npmsec = Options["nodestime"];

	// npmsecがUSI optionで指定されていれば、時間の代わりに、ここで指定されたnode数をベースに思考を行なう。
	// nodes per millisecondの意味。
	// nodes as timeモードで対局しなければならないなら、時間をノード数に変換して、
	// 持ち時間制御の計算式では、この結果の値を用いる。
	if (npmsec)
	{
		// ゲーム開始時に1回だけ
		if (!availableNodes)
			availableNodes = npmsec * limits.time[us];
		
		// ミリ秒をnode数に変換する
		limits.time[us] = TimePoint(availableNodes);
		for (auto c : COLOR)
		{
			limits.inc[c] *= npmsec;
			limits.byoyomi[c] *= npmsec;
		}
		limits.rtime *= npmsec;
		limits.npmsec = npmsec;

		// NetworkDelay , MinimumThinkingTimeなどもすべてnpmsecを掛け算しないといけないな…。
		// 1000で繰り上げる必要もあるしなー。これtime managementと極めて相性が悪いのでは。
	}
#endif

	// ネットワークのDelayを考慮して少し減らすべき。
	// かつ、minimumとmaximumは端数をなくすべき
    network_delay = (TimePoint) options["NetworkDelay"];

	// 探索開始時刻と終了予定時刻。このタイミングで初期化しておく。
	// 終了時刻は0ならば未確定という意味である。
    startTime = ponderhitTime = limits.startTime;
    search_end                = 0;

	// 今回の最大残り時間(これを超えてはならない)
	// byoyomiとincの指定は残り時間にこの時点で加算して考える。
    remain_time =
      limits.time[us] + limits.byoyomi[us] + limits.inc[us] - (TimePoint) options["NetworkDelay2"];
	// ここを0にすると時間切れのあと自爆するのでとりあえず100はあることにしておく。
    remain_time = std::max(remain_time, (TimePoint) 100);

	// 最小思考時間
    minimum_thinking_time = (TimePoint) options["MinimumThinkingTime"];

	// 序盤重視率
	// 　これはこんなパラメーターとして手で調整するべきではなく、探索パラメーターの一種として
	//   別の方法で調整すべき。ただ、対人でソフトに早指ししたいときには意味があるような…。
	int slowMover = (int)options["SlowMover"];

	if (limits.rtime)
	{
		// これが指定されているときは最小思考時間をランダム化する。
		// 連続自己対戦時に同じ進行になるのを回避するためのもの。
		// 終盤で大きく持ち時間を変えると、勝率が5割に寄ってしまうのでそれには注意。

		auto r = limits.rtime;
#if 1
		// 指し手が進むごとに減衰していく曲線にする。
		if (ply)
			r += (int)prng.rand((int)std::min(r * 0.5f, r * 10.0f / (ply)));
#endif

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
	// ⇨　plyは平手の初期局面が1。256手ルールとして、max_game_ply == 256だから、256手目の局面においてply == 256
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
		// この手番で終了なので使いきれば良い。
		minimumTime = optimumTime = maximumTime = remain_time;
		return;
	}

	// minimumとoptimumな時間を適当に計算する。

	{
		// 最小思考時間(これが1000より短く設定されることはないはず..)
		minimumTime = std::max(minimum_thinking_time - network_delay, (TimePoint)1000);

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

		// 1秒ずつは絶対消費していくねんで！
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

		// ただしmaximumは残り時間の30%以上は使わないものとする。
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
    isFinalPush = false;
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

TimePoint TimeManagement::minimum() const { return minimumTime; }
TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }


} // namespace YaneuraOu

#endif
