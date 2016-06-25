#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "misc.h"
#include "thread.h"

using namespace std;

// --------------------
//  統計情報
// --------------------

static int64_t hits[2], means[2];

void dbg_hit_on(bool b) { ++hits[0]; if (b) ++hits[1]; }
void dbg_mean_of(int v) { ++means[0]; means[1] += v; }

void dbg_print() {

  if (hits[0])
    cerr << "Total " << hits[0] << " Hits " << hits[1]
    << " hit rate (%) " << fixed << setprecision(3) << (100.0f * hits[1] / hits[0]) << endl;

  if (means[0])
    cerr << "Total " << means[0] << " Mean "
    << (double)means[1] / means[0] << endl;
}

// --------------------
//  Timer
// --------------------

Timer Time;

int Timer::elapsed() const { return int(Search::Limits.npmsec ? Threads.nodes_searched() : now() - startTime); }
int Timer::elapsed_from_ponderhit() const { return int(Search::Limits.npmsec ? Threads.nodes_searched()/*これ正しくないがこのモードでponder使わないからいいや*/ : now() - startTimeFromPonderhit); }

// --------------------
//  engine info
// --------------------

const string engine_info() {

  stringstream ss;
  
  ss << ENGINE_NAME << ' '
     << EVAL_TYPE_NAME << ' '
     << ENGINE_VERSION << setfill('0')
     << (Is64Bit ? " 64" : " 32")
     << TARGET_CPU << endl
     << "id author by yaneurao" << endl;

  return ss.str();
}

// --------------------
//  sync_out/sync_endl
// --------------------

std::ostream& operator<<(std::ostream& os, SyncCout sc) {
  static Mutex m;
  if (sc == IO_LOCK)    m.lock();
  if (sc == IO_UNLOCK)  m.unlock();
  return os;
}

// --------------------
//  logger
// --------------------

// logging用のhack。streambufをこれでhookしてしまえば追加コードなしで普通に
// cinからの入力とcoutへの出力をファイルにリダイレクトできる。
// cf. http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81
struct Tie : public streambuf
{
  Tie(streambuf* buf_ , streambuf* log_) : buf(buf_) , log(log_) {}

  int sync() { return log->pubsync(), buf->pubsync(); }
  int overflow(int c) { return write(buf->sputc((char)c), "<< "); }
  int underflow() { return buf->sgetc(); }
  int uflow() { return write(buf->sbumpc(), ">> "); }

  int write(int c, const char* prefix) {
    static int last = '\n';
    if (last == '\n')
      log->sputn(prefix, 3);
    return last = log->sputc((char)c);
  }

  streambuf *buf, *log; // 標準入出力 , ログファイル
};

struct Logger {
  static void start(bool b)
  {
    static Logger log;

    if (b && !log.file.is_open())
    {
      log.file.open("io_log.txt", ifstream::out);
      cin.rdbuf(&log.in);
      cout.rdbuf(&log.out);
      cout << "start logger" << endl;
    } else if (!b && log.file.is_open())
    {
      cout << "end logger" << endl;
      cout.rdbuf(log.out.buf);
      cin.rdbuf(log.in.buf);
      log.file.close();
    }
  }

private:
  Tie in, out;   // 標準入力とファイル、標準出力とファイルのひも付け
  ofstream file; // ログを書き出すファイル

  Logger() : in(cin.rdbuf(),file.rdbuf()) , out(cout.rdbuf(),file.rdbuf()) {}
  ~Logger() { start(false); }

};

void start_logger(bool b) { Logger::start(b); }

// --------------------
//  ファイルの丸読み
// --------------------

// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。
int read_all_lines(std::string filename, std::vector<std::string>& lines)
{
  fstream fs(filename,ios::in);
  if (fs.fail())
    return 1; // 読み込み失敗

  while (!fs.fail() && !fs.eof())
  {
    std::string line;
    getline(fs,line);
    if (line.length())
      lines.push_back(line);
  }
  fs.close();
  return 0;
}

// --------------------
//  prefetch命令
// --------------------

// prefetch命令を使わない。
#ifdef NO_PREFETCH

void prefetch(void*) {}

#else

void prefetch(void* addr) {

// SSEの命令なのでSSE2が使える状況でのみ使用する。
#ifdef USE_SSE2

#  if defined(__INTEL_COMPILER)
  // 最適化でprefetch命令を削除するのを回避するhack。MSVCとgccは問題ない。
  __asm__("");
#  endif

#  if defined(__INTEL_COMPILER) || defined(_MSC_VER)
  _mm_prefetch((char*)addr, _MM_HINT_T0);
#  else
  __builtin_prefetch(addr);
#  endif

#endif
}

#endif
