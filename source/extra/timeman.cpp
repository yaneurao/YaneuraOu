#include "../shogi.h"

#ifdef  USE_TIME_MANAGEMENT

#include "../misc.h"
#include "../search.h"
#include "../thread.h"

namespace {

  // これぐらい自分が指すと終局すると考えて計画を練る。
  const int MoveHorizon = 80;

  // 思考時間のrtimeが指定されたときに用いる乱数
  PRNG prng;

} // namespace


// 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
// これは探索の開始時に呼び出されて、今回の指し手のための思考時間を計算する。
// limitsで指定された条件に基いてうまく計算する。

void Timer::init(Search::LimitsType& limits, Color us, int ply)
{
	// nodes as timeモード
	int npmsec = (int)Options["nodestime"];

	// npmsecがUSI optionで指定されていれば、時間の代わりに、ここで指定されたnode数をベースに思考を行なう。
	// nodes per milli secondの意味。
	// nodes as timeモードで対局しなければならないなら、時間をノード数に変換して、
	// 持ち時間制御の計算式では、この結果の値を用いる。
	if (npmsec)
	{
		// ゲーム開始時に1回だけ
		if (!availableNodes)
			availableNodes = npmsec * limits.time[us];
		
		// ミリ秒をnode数に変換する
		limits.time[us] = (int)availableNodes;
		for (auto c : COLOR)
		{
			limits.inc[c] *= npmsec;
			limits.byoyomi[c] *= npmsec;
		}
		limits.rtime *= npmsec;
		limits.npmsec = npmsec;
	}

	// ネットワークのDelayを考慮して少し減らすべき。
	// かつ、minimumとmaximumは端数をなくすべき
	network_delay = (int)Options["NetworkDelay"];

	// 探索終了予定時刻。このタイミングで初期化しておく。
	search_end = 0;

	// 今回の最大残り時間(これを超えてはならない)
	remain_time = limits.time[us] + limits.byoyomi[us] - (int)Options["NetworkDelay2"];
	// ここを0にすると時間切れのあと自爆するのでとりあえず100にしておく。
	remain_time = std::max(remain_time, 100);

	// 最小思考時間
	minimum_thinking_time = (int)Options["MinimumThinkingTime"];

	// 序盤重視率
	// 　これはこんなパラメーターとして手で調整するべきではなく、探索パラメーターの一種として
	//   別の方法で調整すべき。ただ、対人でソフトに早指ししたいときには意味があるような…。
	int slowMover = (int)Options["SlowMover"];

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

	// 残り手数
	// plyは開始局面が1。
	// なので、256手ルールとして、max_game_ply == 256
	// 256手目の局面においてply == 256
	// その1手前の局面においてply == 255
	// これらのときにMTGが1にならないといけない。
	// だから2足しておくのが正解。
	const int MTG = std::min((limits.max_game_ply - ply + 2) / 2, MoveHorizon);

	if (MTG <= 0)
	{
		// 本来、終局までの最大手数が指定されているわけだから、この条件で呼び出されるはずはないのだが…。
		sync_cout << "info string max_game_ply is too small." << sync_endl;
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
		minimumTime = std::max(minimum_thinking_time - network_delay, 1000);

		// 最適思考時間と、最大思考時間には、まずは上限値を設定しておく。
		optimumTime = maximumTime = remain_time;

		// optimumTime = min ( minimumTime + α     , remain_time)
		// maximumTime = min ( minimumTime + α * 5 , remain_time)
		// みたいな感じで考える

		// 残り手数において残り時間はあとどれくらいあるのか。
		int remain_estimate = limits.time[us]
			+ limits.inc[us] * MTG
			// 秒読み時間も残り手数に付随しているものとみなす。
			+ limits.byoyomi[us] * MTG;

		// 1秒ずつは絶対消費していくねんで！
		remain_estimate -= (MTG + 1) * 1000;
		remain_estimate = std::max(remain_estimate, 0);

		// -- optimumTime
		int t1 = minimumTime + remain_estimate / MTG;

		// -- maximumTime
		float max_ratio = 5.0f;
		// 切れ負けルールにおいては、5分を切っていたら、このratioを抑制する。
		if (limits.inc[us] == 0 && limits.byoyomi[us] == 0)
		{
			// 3分     : ratio = 3.0
			// 2分     : ratio = 2.0
			// 1分以下 : ratio = 1.0固定
			max_ratio = std::min(max_ratio, std::max(float(limits.time[us]) / (60 * 1000), 1.0f));
		}
		int t2 = minimumTime + (int)(remain_estimate * max_ratio / MTG);

		// ただしmaximumは残り時間の30%以上は使わないものとする。
		// optimumが超える分にはいい。それは残り手数が少ないときとかなので構わない。
		t2 = std::min(t2, (int)(remain_estimate * 0.3));

		// slowMoverは100分率で与えられていて、optimumTimeの係数として作用するものとする。
		optimumTime = std::min(t1, optimumTime) * slowMover / 100;
		maximumTime = std::min(t2, maximumTime);

		// Ponderが有効になっている場合、ponderhitすると時間が本来の予測より余っていくので思考時間を心持ち多めにとっておく。
		// これ本当はゲーム開始時にUSIコマンドで送られてくるべきだと思う。USI 2.0を策定するか…。
		if (Threads.received_go_ponder)
			optimumTime += optimumTime / 4;
	}

	// 秒読みモードでかつ、持ち時間がないなら、最小思考時間も最大思考時間もその時間にしたほうが得
	if (limits.byoyomi[us])
	{
		// 持ち時間が少ないなら(秒読み時間の1.2倍未満なら)、思考時間を使いきったほうが得
		// これには持ち時間がゼロのケースも含まれる。
		if (limits.time[us] < (int)(limits.byoyomi[us] * 1.2))
			minimumTime = optimumTime = maximumTime = limits.byoyomi[us] + limits.time[us];
	}

	// 残り時間 - network_delay2よりは短くしないと切れ負けになる可能性が出てくる。
	minimumTime = std::min(round_up(minimumTime), remain_time);
	optimumTime = std::min(optimumTime, remain_time);
	maximumTime = std::min(round_up(maximumTime), remain_time);

}

#endif
