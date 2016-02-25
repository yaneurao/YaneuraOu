#ifndef _MISC_H_
#define _MISC_H_

#include <chrono>
#include <thread>
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

struct Timer
{
  // タイマーを初期化する。以降、elapsed()でinit()してからの経過時間が得られる。
  void init() { startTime = now(); }

  // 探索開始からの経過時間。単位は[ms]
  // 探索node数に縛りがある場合、elapsed()で探索node数が返ってくる仕様にすることにより、一元管理できる。
  int elapsed() const { return int(now() - startTime); }

  // node数を指定して探索するとき、探索できる残りnode数。
  int64_t availableNodes;

private:
  // 探索開始時間
  TimePoint startTime;
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
// Stockfishで用いられているもの
// UniformRandomNumberGenerator互換にして、std::shuffle()等でも使えるようにするべきか？
struct PRNG {
  PRNG(uint64_t seed) : s(seed) { ASSERT_LV1(seed); }

  // 乱数seedを指定しなければ現在時刻をseedとする。ただし、自己対戦のときに同じ乱数seedになる可能性が濃厚になるので
  // このときにthisのアドレスなどを加味してそれを乱数seedとする。(by yaneurao)
  PRNG() : s(now() ^ uint64_t(this) + get_thread_id() * 7) {
    int n = (int)get_thread_id() & 1024; // 最大1024回、乱数を回してみる
    for (int i = 0; i < n; ++i) rand<uint64_t>();
  }

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


#endif // _MISC_H_
