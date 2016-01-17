#ifndef _MISC_H_
#define _MISC_H_

#include "shogi.h"
#include <chrono>
#include <thread>

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

// --------------------
//       乱数
// --------------------

// 擬似乱数生成器
// Stockfishで用いられているもの
struct PRNG {
  PRNG(uint64_t seed) : s(seed) { ASSERT_LV1(seed); }

  // 乱数seedを指定しなければ現在時刻をseedとする。ただし、自己対戦のときに同じ乱数seedになる可能性が濃厚になるので
  // このときにthisのアドレスなどを加味してそれを乱数seedとする。(by yaneurao)
  PRNG() : s(now() ^ uint64_t(this)) {}

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
