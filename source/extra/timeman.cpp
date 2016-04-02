#include "../shogi.h"

#ifdef  USE_TIME_MANAGEMENT

#include "../misc.h"
#include "../search.h"

namespace {

  // これぐらい自分が指すと終局すると考えて計画を練る。
  const int MoveHorizon = 70;

  // 思考時間のrtimeが指定されたときに用いる乱数
  PRNG prng;

} // namespace


// 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
// これは探索の開始時に呼び出されて、今回の指し手のための思考時間を計算する。
// limitsで指定された条件に基いてうまく計算する。

void Timer::init(Search::LimitsType& limits, Color us, int ply)
{
  int mtt = Options["MinimumThinkingTime"];
  minimumTime = mtt;
  int moveOverhead = 30; // Options["Move Overhead"];
  int slowMover = Options["SlowMover"];

  if (limits.rtime)
  {
    // これが指定されているときは1～3倍の範囲で最小思考時間をランダム化する。
    // 連続自己対戦時に最小思考時間をばらつかせるためのもの。
    minimumTime = limits.rtime + (int)prng.rand(limits.rtime * 2);
  }

#if 0
  // npmsecがUSI optionで指定されていれば、時間の代わりに、ここで指定されたnode数をベースに思考を行なう。
  // nodes per milli secondの意味。
  int npmsec = Options["nodestime"];

  if (npmsec)
  {
    if (!availableNodes) // 試合開始時に設定しておくべき。
      availableNodes = npmsec * limits.time[us]; // Time is in msec

    // ミリ秒からノード数に単位を変換する。
    limits.time[us] = (int)availableNodes;
    limits.inc[us] *= npmsec;
    limits.npmsec = npmsec;
  }
#endif

  // このTimeクラスの開始時刻をlimitsで与えられた時刻にしてやる。
  startTime = limits.startTime;

  optimumTime = maximumTime = std::max(limits.time[us], minimumTime);

  // 残り手数
  // plyは開始局面が1。
  // なので、256手ルールとして、max_game_ply == 256
  // 256手目の局面においてply == 256
  // その1手前の局面においてply == 255
  // これらのときにMTGが1にならないといけない。
  // だから2足しておくのが正解。
  const int MTG = std::min((limits.max_game_ply - ply + 2)/2, MoveHorizon);
  
  if (MTG <= 0)
  {
    // 本来、終局までの最大手数が指定されているわけだから、この条件で呼び出されるはずはないのだが…。
    sync_cout << "info string max_game_ply is too small." << sync_endl;
    return;
  }
  if (MTG == 1)
  {
    // この手番で終了なので使いきれば良い。
    minimumTime = optimumTime = maximumTime = limits.time[us] + limits.byoyomi[us];
    goto CalcDelay;
  }

  // minimumとoptimumな時間を適当に計算する。
  
  {
    // 残り手数において残り時間はあとどれくらいあるのか。
    int remain_estimate = limits.time[us]
      + limits.inc[us] * MTG
      // 秒読み時間も残り手数に付随しているものとみなす。
      + limits.byoyomi[us] * MTG;

    // 1秒ずつは絶対消費していくねんで！
    remain_estimate -= (MTG + 1) * 1000;
    remain_estimate = std::max(0, remain_estimate);

    int t1 = minimumTime + remain_estimate / MTG;
    int t2 = minimumTime + remain_estimate * 4 / MTG;

    // ただしmaximumは残り時間の30%以上は使わないものとする。
    // optimumが超える分にはいい。それは残り手数が少ないときとかなので構わない。
    t2 = std::min(t2, (int)(remain_estimate * 0.3));

    optimumTime = std::min(t1, optimumTime);
    maximumTime = std::min(t2, maximumTime);


    // Ponderが有効になっている場合、思考時間を心持ち多めにとっておく。
    if (Options["Ponder"])
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

CalcDelay:;

  // ネットワークのDelayを考慮して少し減らすべき。
  // かつ、minimumとmaximumは端数をなくすべき
  int delay = Options["NetworkDelay"];
  auto round = [&](int t)
  {
    // 1秒単位で繰り上げてdelayを引く。
    t = ((t + 999) / 1000) * 1000 - delay;
    return t;
  };

  minimumTime = std::max(mtt - delay,round(minimumTime));
  optimumTime = std::max(mtt - delay,optimumTime - delay);
  maximumTime = std::max(mtt - delay,round(maximumTime));

}

#endif
