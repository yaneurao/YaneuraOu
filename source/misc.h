#ifndef _MISC_H_
#define _MISC_H_

#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <string>

#include "shogi.h"

// --------------------
//  engine info
// --------------------

// "USI"コマンドに応答するために表示する。
const std::string engine_info();


// --------------------
//  sync_out/sync_endl
// --------------------

// スレッド排他しながらcoutに出力するために使う。
// 例)
// sync_out << "bestmove " << m << sync_endl;
// のように用いる。

enum SyncCout { IO_LOCK, IO_UNLOCK };
std::ostream& operator<<(std::ostream&, SyncCout);

#define sync_cout std::cout << IO_LOCK
#define sync_endl std::endl << IO_UNLOCK


// --------------------
//  logger
// --------------------

// cin/coutへの入出力をファイルにリダイレクトを開始/終了する。
extern void start_logger(bool b);


// --------------------
//  ファイルの丸読み
// --------------------

// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。
extern int read_all_lines(std::string filename, std::vector<std::string>& lines);



// --------------------
//  統計情報
// --------------------

// 1秒おきにdbg_print()が呼び出される(やねうら王classic-tceなど)とする。
// このとき、以下の関数を呼び出すと、その統計情報をcerrに出力する。

extern void dbg_print();

// bがtrueであった回数 / dbg_hit_on()が呼び出された回数 を調べるためのもの。
// (どれくらいの割合でXが成り立つか、みたいなのを調べるときに用いる)
extern void dbg_hit_on(bool b);

// vの合計 / 呼びだされた回数 ( = vの平均) みたいなのを求めるときに調べるためのもの。
extern void dbg_mean_of(int v);

// --------------------
//  Time[ms] wrapper
// --------------------

// ms単位での時間計測しか必要ないのでこれをTimePoint型のように扱う。
typedef std::chrono::milliseconds::rep TimePoint;

// ms単位で現在時刻を返す
inline TimePoint now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 指定されたミリ秒だけsleepする。
inline void sleep(int ms)
{
  std::this_thread::sleep_for(std::chrono::microseconds(ms));
}

// -----------------------
//  探索のときに使う時間管理用
// -----------------------

namespace Search { struct LimitsType; }

struct Timer
{
  // タイマーを初期化する。以降、elapsed()でinit()してからの経過時間が得られる。
  void reset() { startTime = startTimeFromPonderhit = now(); }

  // "ponderhit"からの時刻を計測する用
  void reset_for_ponderhit() { startTimeFromPonderhit = now(); }

  // 探索開始からの経過時間。単位は[ms]
  // 探索node数に縛りがある場合、elapsed()で探索node数が返ってくる仕様にすることにより、一元管理できる。
  int elapsed() const;

  // reset_for_ponderhit()からの経過時間。その関数は"ponderhit"したときに呼び出される。
  // reset_for_ponderhit()が呼び出されていないときは、reset()からの経過時間。その関数は"go"コマンドでの探索開始時に呼び出される。
  int elapsed_from_ponderhit() const;

  // reset()されてからreset_for_ponderhit()までの時間
  int elapsed_from_start_to_ponderhit() const { return (int)(startTimeFromPonderhit - startTime); }

  // 探索node数を経過時間の代わりに使う。(こうするとタイマーに左右されない思考が出来るので、思考に再現性を持たせることが出来る)
  // node数を指定して探索するとき、探索できる残りnode数。
  int64_t availableNodes;

  // このシンボルが定義されていると、今回の思考時間を計算する機能が有効になる。
#ifdef  USE_TIME_MANAGEMENT

  // 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
  void init(Search::LimitsType& limits, Color us, int ply);

  int minimum() const { return minimumTime; }
  int optimum() const { return optimumTime; }
  int maximum() const { return maximumTime; }

  // 1秒単位で繰り上げてdelayを引く。
  // ただし、remain_timeよりは小さくなるように制限する。
  int round_up(int t) const {
    // 1000で繰り上げる。Options["MinimalThinkingTime"]が最低値。
    t = std::max(((t + 999) / 1000) * 1000 , minimum_thinking_time );
    // そこから、Options["NetworkDelay"]の値を引くが、remain_timeを上回ってはならない。
    t = std::min(t - network_delay , remain_time);
    return t;
  }

  // 探索終了の時間(startTime + search_end >= now()になったら停止)
  // この値がマイナスのときは、startTimeFromPonderhit - (search_end) >= now() になったら停止。
  int search_end;

private:
  int minimumTime;
  int optimumTime;
  int maximumTime;

  // Options["NetworkDelay"]の値
  int network_delay;
  // Options["MinimalThinkingTime"]の値
  int minimum_thinking_time;

  // 今回の残り時間 - Options["NetworkDelay2"]
  int remain_time;

#endif

private:
  // 探索開始時間
  TimePoint startTime;

  TimePoint startTimeFromPonderhit;
};

extern Timer Time;

// --------------------
//       乱数
// --------------------

// 乱数のseedなどとしてthread idを使いたいが、
// C++のthread idは文字列しか取り出せないので無理やりcastしてしまう。
inline uint64_t get_thread_id()
{
  auto id = std::this_thread::get_id();
  if (sizeof(id) >= 8)
    return *(uint64_t*)(&id);
  else if (sizeof(id) >= 4)
    return *(uint32_t*)(&id);
  else 
    return 0; // give up
}

// 擬似乱数生成器
// Stockfishで用いられているもの + random_deviceによるseedの初期化機能。
// UniformRandomNumberGenerator互換にして、std::shuffle()等でも使えるようにするべきか？
struct PRNG {
  PRNG(uint64_t seed) : s(seed) { ASSERT_LV1(seed); }

  // C++11のrandom_device()によるseedの初期化
  PRNG() { std::random_device rd; s = (u64)rd() + ((u64)rd() << 32); }

  // 乱数を一つ取り出す。
  template<typename T> T rand() { return T(rand64()); }

  // 0からn-1までの乱数を返す。(一様分布ではないが現実的にはこれで十分)
  uint64_t rand(size_t n) { return rand<uint64_t>() % n; }

private:
  uint64_t s;
  uint64_t rand64() {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return s * 2685821657736338717LL;
  }
};

// --------------------
//       Path
// --------------------

// path名とファイル名を結合して、それを返す。
// folder名のほうは空文字列でないときに、末尾に'/'か'\\'がなければそれを付与する。
inline std::string path_combine(const std::string& folder, const std::string& filename)
{
	if (folder.length() >= 1 && *folder.rbegin() != '/' && *folder.rbegin() != '\\')
		return folder + "/" + filename;

	return folder + filename;
}

// --------------------
//  prefetch命令
// --------------------

// prefetch()は、与えられたアドレスの内容をL1/L2 cacheに事前に読み込む。
// これはnon-blocking関数で、CPUがメモリに読み込むのを待たない。

extern void prefetch(void* addr);


#endif // _MISC_H_
