
// Windows環境下でのプロセッサグループの割当関係
#ifdef _WIN32
#if _WIN32_WINNT < 0x0601
#undef  _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Force to include needed API prototypes
#endif
#include <windows.h>
// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.
extern "C" {
	typedef bool(*fun1_t)(LOGICAL_PROCESSOR_RELATIONSHIP,
		PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
	typedef bool(*fun2_t)(USHORT, PGROUP_AFFINITY);
	typedef bool(*fun3_t)(HANDLE, CONST GROUP_AFFINITY*, PGROUP_AFFINITY);
}

// このheaderのなかでmin,maxを定義してあって、C++のstd::min,maxと衝突して困る。
#undef max
#undef min

#endif

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

void prefetch2(void* addr) {
	prefetch(addr);
	prefetch((uint8_t*)addr + 64);
}


namespace WinProcGroup {

#ifndef _WIN32

	void bindThisThread(size_t) {}

#else

  // スレッドID idxに対し、当該スレッドを実行すべきプロセッサーグループの番号を返す。
  // Windowsではプロセッサーは以下のように扱われる。
  // - システムは1つ以上のプロセッサーグループからなる
  // - 1つのプロセッサーグループは1つ以上のNUMAノードからなる
  // - 1つのNUMAノードは1つ以上の論理プロセッサーからなる
  // - 1つのプロセッサーグループには最大で64個までの論理プロセッサーを含めることができる。
  // https://technet.microsoft.com/ja-jp/windowsserver/ee661585.aspx
  // 
  // Intel Xeon Phi Knights Landings上でWindows Server 2016を動かした場合、
  // 64論理プロセッサー毎にプロセッサーグループに分割される。
  // 例えばIntel Xeon Phi Processor 7250の場合、
  // 論理272コアは64+64+64+64+16の5つのプロセッサーグループに分割される。
  // Stockfishのget_group()は全てのプロセッサーグループに同じ数の論理プロセッサが含まれることを仮定している。
  // このため上記の構成ではCPUを使い切ることが出来ない。
  // 以下の実装では先頭のプロセッサーグループから貪欲にスレッドを割り当てている。
  // これによりIntel Xeon Phi Processor 7250においても100%CPUを使い切ることができる。
	int get_group(size_t idx) {
    WORD activeProcessorGroupCount = ::GetActiveProcessorGroupCount();
    for (WORD processorGroupNumber = 0; processorGroupNumber < activeProcessorGroupCount; ++processorGroupNumber) {
      DWORD activeProcessorCount = ::GetActiveProcessorCount(processorGroupNumber);
      if (idx < activeProcessorCount) {
        return processorGroupNumber;
      }
      idx -= activeProcessorCount;
    }

    return -1;
	}


	/// bindThisThread() set the group affinity of the current thread

	void bindThisThread(size_t idx) {

		// If OS already scheduled us on a different group than 0 then don't overwrite
		// the choice, eventually we are one of many one-threaded processes running on
		// some Windows NUMA hardware, for instance in fishtest. To make it simple,
		// just check if running threads are below a threshold, in this case all this
		// NUMA machinery is not needed.
		if (Threads.size() < 8)
			return;

		// Use only local variables to be thread-safe
		int group = get_group(idx);

		if (group == -1)
			return;

		// Early exit if the needed API are not available at runtime
		HMODULE k32 = GetModuleHandle(L"Kernel32.dll");
		auto fun2 = (fun2_t)GetProcAddress(k32, "GetNumaNodeProcessorMaskEx");
		auto fun3 = (fun3_t)GetProcAddress(k32, "SetThreadGroupAffinity");

		if (!fun2 || !fun3)
			return;

		GROUP_AFFINITY affinity;
		if (fun2(group, &affinity))
			fun3(GetCurrentThread(), &affinity, nullptr);
	}

#endif

} // namespace WinProcGroup
