
// Windows環境下でのプロセッサグループの割当関係
#ifdef _WIN32
#if _WIN32_WINNT < 0x0601
#undef  _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Force to include needed API prototypes
#endif

// windows.hのなかでmin,maxを定義してあって、C++のstd::min,maxと衝突して困る。
// #undef max
// #undef min
// としても良いが、以下のようにdefineすることでこれを回避できるらしい。

#ifndef NOMINMAX
#define NOMINMAX
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

#endif

#include <fstream>
#include <iomanip>
//#include <iostream>
#include <sstream>
//#include <vector>

#include <ctime>	// std::ctime()
#include <cstring>	// std::memset()
#include <cmath>	// std::exp()
#include <cstdio>	// fopen(),fread()

#if defined(__linux__) && !defined(__ANDROID__)
#include <stdlib.h>
#include <sys/mman.h> // madvise()
#endif

#include "misc.h"
#include "thread.h"
#include "usi.h"

using namespace std;

namespace {

	// --------------------
	//  logger
	// --------------------

	// logging用のhack。streambufをこれでhookしてしまえば追加コードなしで普通に
	// cinからの入力とcoutへの出力をファイルにリダイレクトできる。
	// cf. http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81
	struct Tie : public streambuf
	{
		Tie(streambuf* buf_, streambuf* log_) : buf(buf_), log(log_) {}

		int sync() override { return log->pubsync(), buf->pubsync(); }
		int overflow(int c) override { return write(buf->sputc((char)c), "<< "); }
		int underflow() override { return buf->sgetc(); }
		int uflow() override { return write(buf->sbumpc(), ">> "); }

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
			}
			else if (!b && log.file.is_open())
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

		// clangだとここ警告が出るので一時的に警告を抑制する。
#pragma warning (disable : 4068) // MSVC用の不明なpragmaの抑制
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
		Logger() : in(cin.rdbuf(), file.rdbuf()), out(cout.rdbuf(), file.rdbuf()) {}
#pragma clang diagnostic pop

		~Logger() { start(false); }
	};

} // 無名namespace

// Trampoline helper to avoid moving Logger to misc.h
void start_logger(bool b) { Logger::start(b); }

// --------------------
//  engine info
// --------------------

const string engine_info() {

	stringstream ss;

	// カレントフォルダに"engine_name.txt"があればその1行目をエンジン名とする機能
	ifstream ifs("engine_name.txt");
	if (!ifs.fail())
	{
		// 1行目が読み込めなかったときのためにデフォルト値を設定しておく。
		string str = "default engine";
		Tools::getline(ifs, str);
		ss << "id name " << str << endl;

		// 2行目が読み込めなかったときのためにデフォルト値を設定しておく。
		str = "default author";
		Tools::getline(ifs, str);
		ss << "id author " << str << endl;
	}
	else
	{
		ss << "id name " <<
			// Makefileのほうでエンジン表示名が指定されているならそれに従う。
#if defined(ENGINE_NAME_FROM_MAKEFILE)
			// マクロの内容の文字列化
			// cf. https://www.hiroom2.com/2015/09/07/c%E8%A8%80%E8%AA%9E%E3%81%AE-line-%E3%83%9E%E3%82%AF%E3%83%AD%E3%82%92%E3%83%97%E3%83%AA%E3%83%97%E3%83%AD%E3%82%BB%E3%83%83%E3%82%B5%E3%81%AE%E6%AE%B5%E9%9A%8E%E3%81%A7%E6%96%87%E5%AD%97%E5%88%97%E3%81%AB%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B/
#define STRINGIFY(n) #n
#define TOSTRING(n) STRINGIFY(n)
			TOSTRING(ENGINE_NAME_FROM_MAKEFILE)
#undef STRINGIFY
#undef TOSTRING
#else
			ENGINE_NAME
#endif			
			<< ' '
			<< EVAL_TYPE_NAME << ' '
			<< ENGINE_VERSION << std::setfill('0')
			<< (Is64Bit ? " 64" : " 32")
			<< TARGET_CPU
#if defined(FOR_TOURNAMENT)
			<< " TOURNAMENT"
#endif

#if defined(EVAL_LEARN)
			<< " EVAL_LEARN"
#endif

			<< endl
#if !defined(YANEURAOU_ENGINE_DEEP)
			<< "id author by yaneurao" << std::endl;
#else
			<< "id author by Tadao Yamaoka , yaneurao" << std::endl;
#endif
	}

	return ss.str();
}

// 使用したコンパイラについての文字列を返す。
const std::string compiler_info() {

#define stringify2(x) #x
#define stringify(x) stringify2(x)
#define make_version_string(major, minor, patch) stringify(major) "." stringify(minor) "." stringify(patch)

	/// Predefined macros hell:
	///
	/// __GNUC__           Compiler is gcc, Clang or Intel on Linux
	/// __INTEL_COMPILER   Compiler is Intel
	/// _MSC_VER           Compiler is MSVC or Intel on Windows
	/// _WIN32             Building on Windows (any)
	/// _WIN64             Building on Windows 64 bit

	std::string compiler = "\nCompiled by ";

#ifdef __clang__
	compiler += "clang++ ";
	compiler += make_version_string(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif __INTEL_COMPILER
	compiler += "Intel compiler ";
	compiler += "(version ";
	compiler += stringify(__INTEL_COMPILER) " update " stringify(__INTEL_COMPILER_UPDATE);
	compiler += ")";
#elif _MSC_VER
	compiler += "MSVC ";
	compiler += "(version ";
	compiler += stringify(_MSC_FULL_VER) "." stringify(_MSC_BUILD);
	compiler += ")";
#elif __GNUC__
	compiler += "g++ (GNUC) ";
	compiler += make_version_string(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
	compiler += "Unknown compiler ";
	compiler += "(unknown version)";
#endif

#if defined(__APPLE__)
	compiler += " on Apple";
#elif defined(__CYGWIN__)
	compiler += " on Cygwin";
#elif defined(__MINGW64__)
	compiler += " on MinGW64";
#elif defined(__MINGW32__)
	compiler += " on MinGW32";
#elif defined(__ANDROID__)
	compiler += " on Android";
#elif defined(__linux__)
	compiler += " on Linux";
#elif defined(_WIN64)
	compiler += " on Microsoft Windows 64-bit";
#elif defined(_WIN32)
	compiler += " on Microsoft Windows 32-bit";
#else
	compiler += " on unknown system";
#endif

#ifdef __VERSION__
	// __VERSION__が定義されているときだけ、その文字列を出力する。(MSVCだと定義されていないようだ..)
	compiler += "\n __VERSION__ macro expands to: ";
	compiler += __VERSION__;
#else
	compiler += "(undefined macro)";
#endif

	compiler += "\n";

	return compiler;
}

// --------------------
//  統計情報
// --------------------

static std::atomic<int64_t> hits[2], means[2];

void dbg_hit_on(bool b) { ++hits[0]; if (b) ++hits[1]; }
void dbg_hit_on(bool c, bool b) { if (c) dbg_hit_on(b); }
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
//  sync_out/sync_endl
// --------------------

std::ostream& operator<<(std::ostream& os, SyncCout sc) {

	static std::mutex m;

	if (sc == IO_LOCK)
		m.lock();

	if (sc == IO_UNLOCK)
		m.unlock();

	return os;
}

// --------------------
//  prefetch命令
// --------------------

// prefetch命令を使わない。
#if defined (NO_PREFETCH)

void prefetch(void*) {}

#else

void prefetch(void* addr) {

	// SSEの命令なのでSSE2が使える状況でのみ使用する。
#if defined (USE_SSE2)

	// 下位5bitが0でないような中途半端なアドレスのprefetchは、
	// そもそも構造体がalignされていない可能性があり、バグに違いない。
	ASSERT_LV3(((u64)addr & 0x1f) == 0);

#  if defined(__INTEL_COMPILER)
	// 最適化でprefetch命令を削除するのを回避するhack。MSVCとgccは問題ない。
	__asm__("");
#  endif

	// 1 cache lineのprefetch
	// 64bytesの系もあるかも知れないが、Stockfishではcache line = 32bytesだと仮定してある。
	// ちなみにRyzenでは32bytesらしい。

#  if defined(__INTEL_COMPILER) || defined(_MSC_VER)
	_mm_prefetch((char*)addr, _MM_HINT_T0);
	//	cout << hex << (u64)addr << endl;
#  else
	__builtin_prefetch(addr);
#  endif

#endif
}

#endif

// --------------------
//  Large Page確保
// --------------------

namespace {
	// LargeMemoryを使っているかどうかがわかるように初回だけその旨を出力する。
	bool largeMemoryAllocFirstCall = true;
}

/// aligned_ttmem_alloc will return suitably aligned memory, and if possible use large pages.
/// The returned pointer is the aligned one, while the mem argument is the one that needs to be passed to free.
/// With c++17 some of this functionality can be simplified.
#if defined(__linux__) && !defined(__ANDROID__)

void* aligned_ttmem_alloc(size_t allocSize, void*& mem , size_t align /* ignore */ ) {

	constexpr size_t alignment = 2 * 1024 * 1024; // assumed 2MB page sizes
	size_t size = ((allocSize + alignment - 1) / alignment) * alignment; // multiple of alignment
	if (posix_memalign(&mem, alignment, size))
		mem = nullptr;
	madvise(mem, allocSize, MADV_HUGEPAGE);

	// Linux環境で、Hash TableのためにLarge Pageを確保したことを出力する。
	if (largeMemoryAllocFirstCall)
	{
		sync_cout << "info string Hash table allocation: Linux Large Pages used." << sync_endl;
		largeMemoryAllocFirstCall = false;
	}

	return mem;
}

#elif defined(_WIN64)

static void* aligned_ttmem_alloc_large_pages(size_t allocSize) {

	// LargePageはエンジンオプションにより無効化されているなら何もせずに返る。
	if (!Options["LargePageEnable"])
		return nullptr;

	HANDLE hProcessToken{ };
	LUID luid{ };
	void* mem = nullptr;

	const size_t largePageSize = GetLargePageMinimum();

	// 普通、最小のLarge Pageサイズは、2MBである。
	// Large Pageが使えるなら、ここでは 2097152 が返ってきているはず。

	if (!largePageSize)
		return nullptr;

	// Large Pageを使うには、SeLockMemory権限が必要。
	// cf. http://awesomeprojectsxyz.blogspot.com/2017/11/windows-10-home-how-to-enable-lock.html

	// We need SeLockMemoryPrivilege, so try to enable it for the process
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hProcessToken))
		return nullptr;

	if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid))
	{
		TOKEN_PRIVILEGES tp{ };
		TOKEN_PRIVILEGES prevTp{ };
		DWORD prevTpLen = 0;

		tp.PrivilegeCount = 1;
		tp.Privileges[0].Luid = luid;
		tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

		// Try to enable SeLockMemoryPrivilege. Note that even if AdjustTokenPrivileges() succeeds,
		// we still need to query GetLastError() to ensure that the privileges were actually obtained...
		if (AdjustTokenPrivileges(
			hProcessToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), &prevTp, &prevTpLen) &&
			GetLastError() == ERROR_SUCCESS)
		{
			// round up size to full pages and allocate
			allocSize = (allocSize + largePageSize - 1) & ~size_t(largePageSize - 1);
			mem = VirtualAlloc(
				NULL, allocSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);

			// privilege no longer needed, restore previous state
			AdjustTokenPrivileges(hProcessToken, FALSE, &prevTp, 0, NULL, NULL);
		}
	}

	CloseHandle(hProcessToken);

	return mem;
}

void* aligned_ttmem_alloc(size_t allocSize , void*& mem , size_t align /* ignore */) {

	//static bool firstCall = true;

	// try to allocate large pages
	mem = aligned_ttmem_alloc_large_pages(allocSize);

	// Suppress info strings on the first call. The first call occurs before 'uci'
	// is received and in that case this output confuses some GUIs.

	// uciが送られてくる前に"info string"で余計な文字を出力するとGUI側が誤動作する可能性があるので
	// 初回は出力を抑制するコードが入っているが、やねうら王ではisreadyでメモリ初期化を行うので
	// これは気にしなくて良い。

	// 逆に、評価関数用のメモリもこれで確保するので、何度もこのメッセージが表示されると
	// 煩わしいので、このメッセージは初回のみの出力と変更する。

//	if (!firstCall)
	if (largeMemoryAllocFirstCall)
	{
		if (mem)
			sync_cout << "info string Hash table allocation: Windows Large Pages used." << sync_endl;
		else
			sync_cout << "info string Hash table allocation: Windows Large Pages not used." << sync_endl;

		largeMemoryAllocFirstCall = false;
	}

	// fall back to regular, page aligned, allocation if necessary
	// 4KB単位であることは保証されているはず..
	if (!mem)
		mem = VirtualAlloc(NULL, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

	// VirtualAlloc()はpage size(4KB)でalignされていること自体は保証されているはず。

	//cout << (u64)mem << "," << allocSize << endl;

	return mem;
}

#else

void* aligned_ttmem_alloc(size_t allocSize, void*& mem , size_t align) {

	//constexpr size_t alignment = 64; // assumed cache line size
	
	// 引数で指定された値でalignmentされていて欲しい。
	const size_t alignment = align;

	size_t size = allocSize + alignment - 1; // allocate some extra space
	mem = malloc(size);

	if (largeMemoryAllocFirstCall)
	{
		sync_cout << "info string Hash table allocation: Large Pages not used." << sync_endl;
		largeMemoryAllocFirstCall = false;
	}

	void* ret = reinterpret_cast<void*>((uintptr_t(mem) + alignment - 1) & ~uintptr_t(alignment - 1));
	return ret;
}

#endif

/// aligned_ttmem_free will free the previously allocated ttmem
#if defined(_WIN64)

void aligned_ttmem_free(void* mem) {

	if (mem && !VirtualFree(mem, 0, MEM_RELEASE))
	{
		DWORD err = GetLastError();
		std::cerr << "Failed to free transposition table. Error code: 0x" <<
			std::hex << err << std::dec << std::endl;
		Tools::exit();
	}
}

#else

void aligned_ttmem_free(void* mem) {
	free(mem);
}

#endif

// メモリを確保する。Large Pageに確保できるなら、そこにする。
// aligned_ttmem_alloc()を内部的に呼び出すので、アドレスは少なくとも2MBでalignされていることは保証されるが、
// 気になる人のためにalignmentを明示的に指定できるようになっている。
// メモリ確保に失敗するか、引数のalignで指定したalignmentになっていなければ、
// エラーメッセージを出力してプログラムを終了させる。
void* LargeMemory::alloc(size_t size, size_t align , bool zero_clear)
{
	free();
	return static_alloc(size, this->mem, align, zero_clear);
}

// alloc()で確保したメモリを開放する。
// このクラスのデストラクタからも自動でこの関数が呼び出されるので明示的に呼び出す必要はない(かも)
void LargeMemory::free()
{
	static_free(mem);
	mem = nullptr;
}

// alloc()のstatic関数版。memには、static_free()に渡すべきポインタが得られる。
void* LargeMemory::static_alloc(size_t size, void*& mem, size_t align, bool zero_clear)
{
	void* ptr = aligned_ttmem_alloc(size, mem, align);

	auto error_exit = [&](std::string mes) {
		sync_cout << "info string Error! : " << mes << " in LargeMemory::alloc(" << size << "," << align << ")" << sync_endl;
		Tools::exit();
	};

	// メモリが正常に確保されていることを保証する
	if (ptr == nullptr)
		error_exit("can't alloc enough memory.");
		
	// ptrがalignmentされていることを保証する
	if ((reinterpret_cast<size_t>(ptr) % align) != 0)
		error_exit("can't alloc algined memory.");

	// ゼロクリアが必要なのか？
	if (zero_clear)
	{
		// 確保したのが256MB以上なら並列化してゼロクリアする。
		if (size < 256 * 1024 * 1024)
			// そんなに大きな領域ではないから、普通にmemset()でやっとく。
		memset(ptr, 0, size);
		else
			// 並列版ゼロクリア
			Tools::memclear(nullptr, ptr, size);
	}

	return ptr;
}

// static_alloc()で確保したメモリを開放する。
void LargeMemory::static_free(void* mem)
{
	aligned_ttmem_free(mem);
}



// --------------------
//  全プロセッサを使う
// --------------------

namespace WinProcGroup {

#if !defined ( _WIN32 )

	void bindThisThread(size_t) {}

#else


	/// best_group() retrieves logical processor information using Windows specific
	/// API and returns the best group id for the thread with index idx. Original
	/// code from Texel by Peter Österlund.

	int best_group(size_t idx) {

		// スレッド番号idx(0 ～ 論理コア数-1)に対して
		// 適切なNUMA NODEとCPU番号を設定する。
		// 非対称プロセッサのことは考慮していない

		// 論理コアの数
		int threads = 0;

		// NUMA NODEの数
		int nodes = 0;

		// 物理コア数
		int cores = 0;

		DWORD returnLength = 0;
		DWORD byteOffset = 0;

		// Early exit if the needed API is not available at runtime
		HMODULE k32 = GetModuleHandle(L"Kernel32.dll");
		auto fun1 = (fun1_t)(void(*)())GetProcAddress(k32, "GetLogicalProcessorInformationEx");
		if (!fun1)
			return -1;

		// First call to get returnLength. We expect it to fail due to null buffer
		if (fun1(RelationAll, nullptr, &returnLength))
			return -1;

		// Once we know returnLength, allocate the buffer
		SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer, *ptr;
		ptr = buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)malloc(returnLength);

		// Second call, now we expect to succeed
		if (!fun1(RelationAll, buffer, &returnLength))
		{
			free(buffer);
			return -1;
		}

		while (byteOffset < returnLength)
		{
			// NUMA NODEの数
			if (ptr->Relationship == RelationNumaNode)
				nodes++;

			else if (ptr->Relationship == RelationProcessorCore)
			{
				// 物理コアの数
				cores++;

				// 論理コア数の加算。HT対応なら2を足す。HT非対応なら1を足す。
				threads += (ptr->Processor.Flags == LTP_PC_SMT) ? 2 : 1;
			}

			ASSERT_LV3(ptr->Size);
			byteOffset += ptr->Size;
			ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(((char*)ptr) + ptr->Size);
		}

		free(buffer);

		std::vector<int> groups;

		// Run as many threads as possible on the same node until core limit is
		// reached, then move on filling the next node.
		for (int n = 0; n < nodes; n++)
			for (int i = 0; i < cores / nodes; i++)
				groups.push_back(n);

		// In case a core has more than one logical processor (we assume 2) and we
		// have still threads to allocate, then spread them evenly across available
		// nodes.

		// 論理プロセッサー数を上回ってスレッドを割り当てたいならば、あとは均等に
		// 各NUMA NODEに割り当てていくしかない。

		for (int t = 0; t < threads - cores; t++)
			groups.push_back(t % nodes);

		// If we still have more threads than the total number of logical processors
		// then return -1 and let the OS to decide what to do.
		return idx < groups.size() ? groups[idx] : -1;

		// NUMA NODEごとにプロセッサグループは分かれているだろうという想定なので
		// NUMAが2(Dual CPU)であり、片側のCPUが40論理プロセッサであるなら、この関数は、
		// idx = 0..39なら 0 , idx = 40..79なら1を返す。
	}

	/// bindThisThread() set the group affinity of the current thread

	void bindThisThread(size_t idx) {

#if defined(_WIN32)
		idx += Options["ThreadIdOffset"];
#endif

		// Use only local variables to be thread-safe

		// 使うべきプロセッサグループ番号が返ってくる。
		int group = best_group(idx);

		if (group == -1)
			return;

		// Early exit if the needed API are not available at runtime
		HMODULE k32 = GetModuleHandle(L"Kernel32.dll");
		auto fun2 = (fun2_t)(void(*)())GetProcAddress(k32, "GetNumaNodeProcessorMaskEx");
		auto fun3 = (fun3_t)(void(*)())GetProcAddress(k32, "SetThreadGroupAffinity");

		if (!fun2 || !fun3)
			return;

		GROUP_AFFINITY affinity;
		if (fun2(group, &affinity))
			fun3(GetCurrentThread(), &affinity, nullptr);
	}

#endif

} // namespace WinProcGroup


// --------------------
//  Timer
// --------------------

TimePoint Timer::elapsed() const { return TimePoint(Search::Limits.npmsec ? Threads.nodes_searched() : now() - startTime); }
TimePoint Timer::elapsed_from_ponderhit() const { return TimePoint(Search::Limits.npmsec ? Threads.nodes_searched()/*これ正しくないがこのモードでponder使わないからいいや*/ : now() - startTimeFromPonderhit); }

#if defined(USE_TIME_MANAGEMENT)

// 1秒単位で繰り上げてdelayを引く。
// ただし、remain_timeよりは小さくなるように制限する。
TimePoint Timer::round_up(TimePoint t0) const
{
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
}

#endif

Timer Time;


// =====   以下は、やねうら王の独自追加   =====


// --------------------
//  ツール類
// --------------------
namespace Tools
{
	// memclear

	// 進捗を表示しながら並列化してゼロクリア
	// ※ Stockfishのtt.cppのTranspositionTable::clear()にあるコードと同等のコード。
	void memclear(const char* name_, void* table, size_t size)
	{
	// Windows10では、このゼロクリアには非常に時間がかかる。
	// malloc()時点ではメモリを実メモリに割り当てられておらず、
	// 初回にアクセスするときにその割当てがなされるため。
	// ゆえに、分割してゼロクリアして、一定時間ごとに進捗を出力する。

	// memset(table, 0, size);

		if (name_ != nullptr)
			sync_cout << "info string " + std::string(name_) + " Clear begin , Hash size =  " << size / (1024 * 1024) << "[MB]" << sync_endl;

	// マルチスレッドで並列化してクリアする。

	std::vector<std::thread> threads;

	auto thread_num = (size_t)Options["Threads"];

	for (size_t idx = 0; idx < thread_num; idx++)
	{
		threads.push_back(std::thread([table, size, thread_num, idx]() {

			// NUMA環境では、bindThisThread()を呼び出しておいたほうが速くなるらしい。

			// Thread binding gives faster search on systems with a first-touch policy
			if (Options["Threads"] > 8)
				WinProcGroup::bindThisThread(idx);

			// それぞれのスレッドがhash tableの各パートをゼロ初期化する。
			const size_t stride = size / thread_num,
				start = stride * idx,
				len = idx != thread_num - 1 ?
				stride : size - start;

			std::memset((uint8_t*)table + start, 0, len);
		}));
	}

	for (std::thread& th : threads)
		th.join();

		if (name_ != nullptr)
			sync_cout << "info string " + std::string(name_) + " Clear done." << sync_endl;
	}

	// 途中での終了処理のためのwrapper
	// コンソールの出力が完了するのを待ちたいので3秒待ってから::exit(EXIT_FAILURE)する。
	void exit()
	{
		sleep(3000); // エラーメッセージが出力される前に終了するのはまずいのでwaitを入れておく。
		::exit(EXIT_FAILURE);
	}

	// 指定されたミリ秒だけsleepする。
	void sleep(int ms)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	}

	// 現在時刻を文字列化したもを返す。(評価関数の学習時などに用いる)
	std::string now_string()
	{
		// std::ctime(), localtime()を使うと、MSVCでセキュアでないという警告が出る。
		// C++標準的にはそんなことないはずなのだが…。

#if defined(_MSC_VER)
		// C4996 : 'ctime' : This function or variable may be unsafe.Consider using ctime_s instead.
#pragma warning(disable : 4996)
#endif

		auto now = std::chrono::system_clock::now();
		auto tp = std::chrono::system_clock::to_time_t(now);
		auto result = string(std::ctime(&tp));
	
		// 末尾に改行コードが含まれているならこれを除去する
		while (*result.rbegin() == '\n' || (*result.rbegin() == '\r'))
			result.pop_back();
		return result;
	}

	// Linux環境ではgetline()したときにテキストファイルが'\r\n'だと
	// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
	// そのため、ifstreamに対してgetline()を呼び出すときは、
	// std::getline()ではなくこのこの関数を使うべき。
	bool getline(std::ifstream& fs, std::string& s)
	{
		bool b = (bool)std::getline(fs, s);
		StringExtension::trim_inplace(s);
		return b;
	}

	// マルチバイト文字列をワイド文字列に変換する。
	// WindowsAPIを呼び出しているのでWindows環境専用。
	std::wstring MultiByteToWideChar(const std::string& s)
	{
#if !defined(_WIN32)
		return std::wstring(s.begin(), s.end()); // NotImplemented
		// 漢字とか使われているとうまく変換できないけど、とりあえずASCII文字列なら
		// 変換できるのでこれで凌いでおく。
#else

		// WindowsAPIのMultiByteToWideChar()を用いて変換するので
		// Windows環境限定。

		// 変換後の文字列を格納するのに必要なバッファサイズが不明なのでいったんその長さを問い合わせる。

		int length = ::MultiByteToWideChar(
			CP_THREAD_ACP,			// コードページ = 現在のスレッドのコードページ
			MB_PRECOMPOSED,			// 文字の種類を指定するフラグ
			s.c_str(),				// マップ元文字列のアドレス
			(int)s.length() + 1,	// マップ元文字列のバイト数
			nullptr,				// マップ先ワイド文字列を入れるバッファのアドレス
			0						// バッファのサイズ
		);

		// マップ元文字列のバイト数だから、そこに0ではなくサイズを指定するなら
		// s.length() + 1を指定しないといけない。

		// ここをs.length()としてしまうと'\0'が変換されずに末尾にゴミが出る(´ω｀)
		// ググって出てくるMultiByteToWideCharのサンプルプログラム、ここが間違ってるものが多すぎ。

		// また::MultiByteToWideCharの返し値が0であることはない。
		// ('\0'を格納するためにwchar_t 1文字分のバッファは少なくとも必要なので)

		wchar_t* buffer = new wchar_t[length];
		SCOPE_EXIT( delete[] buffer; );

		int result = ::MultiByteToWideChar(
			CP_THREAD_ACP,			// コードページ = 現在のスレッドのコードページ
			MB_PRECOMPOSED,			// 文字の種類を指定するフラグ
			s.c_str(),				// マップ元文字列のアドレス
			(int)s.length() + 1,	// マップ元文字列のバイト数
			buffer,					// マップ先ワイド文字列を入れるバッファのアドレス
			length					// バッファのサイズ
		);
 
		if (result == 0)
			return std::wstring(); // 何故かエラーなのだ…。

		return std::wstring(buffer);
#endif
	}

	// ResultCodeを文字列化する。
	std::string to_string(ResultCode code)
	{
		// enumに対してto_string()したいだけなのだが…。

		switch (code)
		{
		case ResultCode::Ok                   : return "Ok";
		case ResultCode::MemoryAllocationError: return "MemoryAllocationError";
		case ResultCode::SomeError            : return "SomeError";
		case ResultCode::FileOpenError        : return "FileOpenError";
		case ResultCode::FileReadError        : return "FileReadError";
		case ResultCode::FileWriteError       : return "FileWriteError";
		case ResultCode::CreateFolderError    : return "CreateFolderError";
		case ResultCode::NotImplementedError  : return "NotImplementedError";
		default                               : return "OtherError";
		}
	}
}

// --------------------
//  ファイルの丸読み
// --------------------

// -- FileOperator

// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。末尾の改行は除去される。
// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
Tools::Result FileOperator::ReadAllLines(const std::string& filename, std::vector<std::string>& lines,bool trim)
{
#if 0
	ifstream fs(filename);
	if (fs.fail())
		return 1; // 読み込み失敗

	while (!fs.fail() && !fs.eof())
	{
		std::string line;
		Dependency::getline(fs, line);
		if (trim)
			line = StringExtension::trim(line);
		if (line.length())
			lines.push_back(line);
	}
	fs.close();

	return 0;
#endif

	// →　100MB程度のテキストファイルの読み込みにSurface Pro 6(Core i7モデル)で4秒程度かかる。
	// ifstreamを使わない形で書き直す。これで4倍ぐらい速くなる。

	TextFileReader reader;

	// ReadLine()時のトリムの設定を反映させる。
	reader.SetTrim(trim);
	// 空行をスキップするモードにする。
	reader.SkipEmptyLine(true);

	auto result = reader.Open(filename);
	if (!result.is_ok())
		return result;

	string line;
	while (reader.ReadLine(line).is_ok())
		lines.emplace_back(line);

	return Tools::Result::Ok();
}

Tools::Result FileOperator::ReadFileToMemory(const std::string& filename, std::function<void*(u64)> callback_func)
{
	fstream fs(filename, ios::in | ios::binary);
	if (fs.fail())
		return Tools::Result(Tools::ResultCode::FileOpenError);

	fs.seekg(0, fstream::end);
	u64 eofPos = (u64)fs.tellg();
	fs.clear(); // これをしないと次のseekに失敗することがある。
	fs.seekg(0, fstream::beg);
	u64 begPos = (u64)fs.tellg();
	u64 file_size = eofPos - begPos;
	//std::cout << "filename = " << filename << " , file_size = " << file_size << endl;

	// ファイルサイズがわかったのでcallback_funcを呼び出してこの分のバッファを確保してもらい、
	// そのポインターをもらう。
	void* ptr = callback_func(file_size);

	// バッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
	// nullptrを返すことになっている。このとき、読み込みを中断し、エラーリターンする。
	// 原因は不明だが、メモリ割り当ての失敗なのでMemoryAllocationErrorを返しておく。
	if (ptr == nullptr)
		return Tools::Result(Tools::ResultCode::MemoryAllocationError);

	// 細切れに読み込む

	const u64 block_size = 1024 * 1024 * 1024; // 1回のreadで読み込む要素の数(1GB)
	for (u64 pos = 0; pos < file_size; pos += block_size)
	{
		// 今回読み込むサイズ
		u64 read_size = (pos + block_size < file_size) ? block_size : (file_size - pos);
		fs.read((char*)ptr + pos, read_size);

		// ファイルの途中で読み込みエラーに至った。
		if (fs.fail())
			return Tools::Result(Tools::ResultCode::FileReadError); // ファイル読み込み時のエラー

		//cout << ".";
	}
	fs.close();

	return Tools::Result::Ok();
}


Tools::Result FileOperator::WriteMemoryToFile(const std::string& filename, void* ptr, u64 size)
{
	fstream fs(filename, ios::out | ios::binary);
	if (fs.fail())
		return Tools::Result(Tools::ResultCode::FileOpenError);

	const u64 block_size = 1024 * 1024 * 1024; // 1回のwriteで書き出す要素の数(1GB)
	for (u64 pos = 0; pos < size; pos += block_size)
	{
		// 今回書き出すメモリサイズ
		u64 write_size = (pos + block_size < size) ? block_size : (size - pos);
		fs.write((char*)ptr + pos, write_size);
		//cout << ".";

		if (fs.fail())
			return Tools::Result(Tools::ResultCode::FileWriteError); // ファイル書き込み時のエラー
	}

	// fstreamなので、これ不要だが..念の為にcloseする。
	fs.close();

	return Tools::Result::Ok();
}

// --- TextFileReader

// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
// fopen()～fread()で実装されている。
TextFileReader::TextFileReader()
{
	buffer.resize(1024 * 1024);
	line_buffer.reserve(2048);
	clear();

	// この２つのフラグはOpen()したときに設定がクリアされるべきではないので、
	// コンストラクタで一度だけ初期化する。
	trim = false;
	skipEmptyLine = false;
}

TextFileReader::~TextFileReader()
{
	Close();
}

// 各種状態変数の初期化
void TextFileReader::clear()
{
	fp = nullptr;
	is_eof = false;
	cursor = 0;
	read_size = 0;
	is_prev_cr = false;
}

// ファイルをopenする。
Tools::Result TextFileReader::Open(const std::string& filename)
{
	Close();

	// 高速化のためにbinary open
	fp = fopen(filename.c_str(), "rb");
	return (fp == nullptr) ? Tools::Result(Tools::ResultCode::FileOpenError) : Tools::Result::Ok();
}

// Open()を呼び出してオープンしたファイルをクローズする。
void TextFileReader::Close()
{
	if (fp != nullptr)
		fclose(fp);

	clear();
}

// バッファから1文字読み込む。eofに達したら、-1を返す。
int TextFileReader::read_char()
{
	// ファイルからバッファの充填はこれ以上できなくて、バッファの末尾までcursorが進んでいるならeofという扱い。
	while (!(is_eof && cursor >= read_size))
	{
		if (cursor < read_size)
			return (int)buffer[cursor++];

		// カーソル(解析位置)が読み込みバッファを超えていたら次のブロックを読み込む。
		read_next_block();
	}
	return -1;
}

// ReadLineの下請け。何も考えずに1行読み込む。行のtrim、空行のskipなどなし。
// line_bufferに読み込まれた行が代入される。
Tools::Result TextFileReader::read_line_simple()
{
	// buffer[cursor]から読み込んでいく。
	// 改行コードに遭遇するとそこまでの文字列を返す。
	line_buffer.clear();

	/*
		改行コード一覧
			Unix        LF      \n
			Mac（OSX）  LF      \n
			Mac（OS9）  CR      \r
			Windows     CR+LF   \r\n

		ゆえに"\r","\n","\r\n"をすべて1つの改行コードとみなさないといけない。
		よって"\r"(CR)がきたときに次の"\n"(LF)は無視するという処理になる。
	*/

	while (true)
	{
		int c = read_char();
		if (c == -1 /* EOF */)
		{
			// line_bufferが空のままeofに遭遇したなら、eofとして扱う。
			// さもなくば、line_bufferを一度返す。(次回呼び出し時にeofとして扱う)
			if (line_buffer.size() == 0)
					return Tools::ResultCode::Eof;

			break;
		}

		if (c == '\r')
		{
			// 直前は"\r"だった。
			is_prev_cr = true;
			break;
		}

		// 直前は"\r"ではないことは確定したのでこの段階でis_prev_crフラグをクリアしておく。
		// ただし、このあと"\n"の判定の時に使うので古いほうの値をコピーして保持しておく。
		auto prev_cr = is_prev_cr;
		is_prev_cr = false;

		if (c == '\n')
		{
			if (!prev_cr)
				break;
			//else
			//   "\r\n"の(前回"\r"を処理した残りの)"\n"なので無視する。
		}
		else
		{
			// 行バッファに積んでいく。
			line_buffer.push_back(c);
		}
	}

	// 行バッファは完成した。
	// line_bufferに入っているのでこのまま使って問題なし。

	return Tools::ResultCode::Ok;
}


// 1行読み込む(改行まで)
Tools::Result TextFileReader::ReadLine(std::string& line)
{
	while (true)
	{
		if (read_line_simple().is_eof())
			return Tools::ResultCode::Eof;

	// trimフラグが立っているなら末尾スペース、タブを除去する。
	if (trim)
		while (line_buffer.size() > 0)
		{
			char c = *line_buffer.rbegin();
			if (!(c == ' ' || c == '\t'))
				break;

			line_buffer.resize(line_buffer.size() - 1);
		}

		// 空行をスキップするモートであるなら、line_bufferが結果的に空になった場合は繰り返すようにする。
		if (skipEmptyLine && line_buffer.size() == 0)
			continue;

		line = std::string((const char*)line_buffer.data(), line_buffer.size());
		return Tools::ResultCode::Ok;
	}
}

// 次のblockのbufferへの読み込み。
void TextFileReader::read_next_block()
{
	if (::feof(fp))
		read_size = 0;
	else
		read_size = ::fread(&buffer[0], sizeof(u8) , buffer.size(), fp);

	// カーソル(解析位置)のリセット
	cursor = 0;

	// 読み込まれたサイズが0なら、終端に達したと判定する。
	is_eof = read_size == 0;
}


// --------------------
//       Parser
// --------------------

/*
	LineScanner parser("AAA BBB CCC DDD");
	auto token = parser.peek_text();
	cout << token << endl;
	token = parser.get_text();
	cout << token << endl;
	token = parser.get_text();
	cout << token << endl;
	token = parser.get_text();
	cout << token << endl;
	token = parser.get_text();
	cout << token << endl;
*/

// 次のtokenを先読みして返す。get_token()するまで解析位置は進まない。
std::string LineScanner::peek_text()
{
	// 二重にpeek_text()を呼び出すのは合法であるものとする。
	if (!token.empty())
		return token;

	// assert(token.empty());

	while (!raw_eol())
	{
		char c = line[pos++];
		if (c == ' ')
			break;
		token += c;
	}
	return token;
}

// 次のtokenを返す。
std::string LineScanner::get_text()
{
	auto result = (!token.empty() ? token : peek_text());
	token.clear();
	return result;
}

// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
s64 LineScanner::get_number(s64 defaultValue)
{
	std::string token = get_text();
	return token.empty() ? defaultValue : atoll(token.c_str());
}

// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
double LineScanner::get_double(double defaultValue)
{
	std::string token = get_text();
	return token.empty() ? defaultValue : atof(token.c_str());
}



// --------------------
//       Math
// --------------------

double Math::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double Math::dsigmoid(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
}


// --------------------
//       Path
// --------------------

// C#にあるPathクラス的なもの。ファイル名の操作。
// C#のメソッド名に合わせておく。
namespace Path
{
	// path名とファイル名を結合して、それを返す。
	// folder名のほうは空文字列でないときに、末尾に'/'か'\\'がなければ'/'を付与する。
	// ('/'自体は、Pathの区切り文字列として、WindowsでもLinuxでも使えるはずなので。
	std::string Combine(const std::string& folder, const std::string& filename)
	{
		if (folder.length() >= 1 && *folder.rbegin() != '/' && *folder.rbegin() != '\\')
			return folder + "/" + filename;

		return folder + filename;
	}

	// full path表現(ファイル名を含む)から、(フォルダ名を除いた)ファイル名の部分を取得する。
	std::string GetFileName(const std::string& path)
	{
		// "\"か"/"か、どちらを使ってあるかはわからないがなるべく後ろにある、いずれかの文字を探す。
		auto path_index1 = path.find_last_of("\\");
		auto path_index2 = path.find_last_of("/");

		// どちらの文字も見つからなかったのであれば、ディレクトリ名が含まれておらず、
		// 与えられたpath丸ごとがファイル名だと考えられる。
		if (path_index1 == std::string::npos && path_index2 == std::string::npos)
			return path;

		// なるべく後ろのを見つけたいが、string::nposは大きな定数なので単純にstd::max()するとこれを持ってきてしまう。
		// string::nposを0とみなしてmaxをとる。
		path_index1 = path_index1 == string::npos ? 0 : path_index1;
		path_index2 = path_index2 == string::npos ? 0 : path_index2;
		auto path_index = std::max(path_index1, path_index2);

		// そこ以降を返す。
		return path.substr(path_index + 1);
	}

	// full path表現から、ディレクトリ名の部分を取得する。
	std::string GetDirectoryName(const std::string& path)
	{
		// ファイル名部分を引き算してやる。

		auto length = path.length() - GetFileName(path).length() - 1;
		return (length == 0) ? "" : path.substr(0,length);
	}

};

// --------------------
//    文字列 拡張
// --------------------

namespace {
	// 文字列を大文字化する
	string to_upper(const string source)
	{
		std::string destination;
		destination.resize(source.size());
		std::transform(source.cbegin(), source.cend(), destination.begin(), /*toupper*/[](char c) { return (char)toupper(c); });
		return destination;
	}
}

namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// string-case insensitive-compareの略？
	// s1==s2のとき0(false)を返す。
	bool stricmp(const string& s1, const string& s2)
	{
		// Windowsだと_stricmp() , Linuxだとstrcasecmp()を使うのだが、
		// 後者がどうも動作が怪しい。自前実装しておいたほうが無難。

		return to_upper(s1) != to_upper(s2);
	}
	
	// スペースに相当する文字か
	bool is_space(char c) { return c == '\r' || c == '\n' || c == ' ' || c == '\t'; }

	// 数字に相当する文字か
	bool is_number(char c) { return '0' <= c && c <= '9'; }
	
	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	std::string trim(const std::string& input)
	{
		// copyしておく。
		string s = input;

		// curを現在位置( s[cur]のような )カーソルだとして扱うと、最後、-1になるまで
		// ループするコードになり、符号型が必要になる。
		// size_tのまま扱いたいので、curを現在の(注目位置+1)を示すカーソルだという扱いに
		// すればこの問題は起きない。

		auto cur = s.length();

			// 改行文字、スペース、タブではないならループを抜ける。
			// これらの文字が出現しなくなるまで末尾を切り詰める。
		while (cur > 0 && is_space(s[cur-1]))
			cur--;

		s.resize(cur);
		return s;
	}

	// trim()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_inplace(std::string& s)
	{
		auto cur = s.length();

		while (cur > 0 && is_space(s[cur-1]))
			cur--;

		s.resize(cur);
	}

	// 行の末尾の数字を除去した文字列を返す。
	// (行の末尾の"\r","\n",スペース、"\t"を除去したあと)
	std::string trim_number(const std::string& input)
	{
		string s = input;
		auto cur = s.length();

		// 末尾のスペースを詰めたあと数字を詰めてそのあと再度スペースを詰める。
		// 例 : "abc 123 "→"abc"となって欲しいので。

		while (cur > 0 && is_space(s[cur-1]))
			cur--;

		while (cur > 0 && is_number(s[cur-1]))
			cur--;

		while (cur > 0 && is_space(s[cur-1]))
			cur--;

		s.resize(cur);
		return s;
	}

	// trim_number()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_number_inplace(std::string& s)
	{
		auto cur = s.length();

		while (cur > 0 && is_space(s[cur - 1]))
			cur--;

		while (cur > 0 && is_number(s[cur - 1]))
			cur--;

		while (cur > 0 && is_space(s[cur - 1]))
			cur--;

		s.resize(cur);
	}

	// 文字列をint化する。int化に失敗した場合はdefault_の値を返す。
	int to_int(const std::string input, int default_)
	{
		// stoi()は例外を出すので例外を使わないようにしてビルドしたいのでNG。
		// atoi()は、セキュリティ的な脆弱性がある。
		// 仕方ないのでistringstreamを使う。

		std::istringstream ss(input);
		int result = default_; // 失敗したときはこの値のままになる
		ss >> result;
		return result;
	}

	// スペース、タブなど空白に相当する文字で分割して返す。
	std::vector<std::string> split(const std::string& input)
	{
		auto result = std::vector<string>();
		LineScanner scanner(input);
		while (!scanner.eol())
			result.push_back(scanner.get_text());

		return result;
	}

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	bool StartsWith(std::string const& value, std::string const& starting)
	{
		if (starting.size() > value.size()) return false;
		return std::equal(starting.begin(), starting.end(), value.begin());
	};

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	bool EndsWith(std::string const& value, std::string const& ending)
	{
		if (ending.size() > value.size()) return false;
		return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
	};

};

// --------------------
//  FileSystem
// --------------------

#if defined(_MSC_VER)

// C++17から使えるようになり、VC++2019でも2018年末のupdateから使えるようになったらしいのでこれを使う。
#include <filesystem>

#elif defined(__GNUC__)

// GCC/clangのほうはfilesystem使う方法がよくわからないので保留しとく。
/*
 備考)
   GCC 8.1では、リンクオプションとして -lstdc++fsが必要
   Clang 7.0では、リンクオプションとして -lc++fsが必要

 2020/1/17現時点で最新版はClang 9.0.0のようだが、OpenBlas等が使えるかわからないので、使えるとわかってから
 filesystemを使うように修正する。

Mizarさんより。
https://gcc.gnu.org/bugzilla/show_bug.cgi?id=91786#c2
Fixed for GCC 9.3
とあるのでまだMSYS2でfilesystemは無理じゃないでしょうか
*/

#include <dirent.h>
#endif

// ディレクトリに存在するファイルの列挙用
// C#のDirectoryクラスっぽい何か
namespace Directory
{
	// 指定されたフォルダに存在するファイルをすべて列挙する。
	// 列挙するときに拡張子を指定できる。(例 : ".bin")
	// 拡張子として""を指定すればすべて列挙される。
	std::vector<std::string> EnumerateFiles(const std::string& sourceDirectory, const string& extension)
	{
		std::vector<std::string> filenames;

#if defined(_MSC_VER)
		// ※　std::tr2は、std:c++14 の下では既定で非推奨の警告を出し、/std:c++17 では既定で削除された。
		// Visual C++2019がupdateでC++17に対応したので、std::filesystemを素直に使ったほうが良い。

		namespace fs = std::filesystem;

		// filesystemのファイル列挙、ディレクトリとして空の文字列を渡すと例外で落ちる。
		// current directoryにしたい時は明示的に指定してやらなければならない。
		auto src = sourceDirectory.empty() ? fs::current_path() : fs::path(sourceDirectory);

		for (auto ent : fs::directory_iterator(src))
			if (fs::is_regular_file(ent)
				&& StringExtension::EndsWith(ent.path().filename().string(), extension))

				filenames.push_back(Path::Combine(ent.path().parent_path().string(), ent.path().filename().string()));

#elif defined(__GNUC__)

		// 仕方ないのでdirent.hを用いて読み込む。
		DIR* dp;       // ディレクトリへのポインタ
		dirent* entry; // readdir() で返されるエントリーポイント

		dp = opendir(sourceDirectory.c_str());
		if (dp != NULL)
		{
			do {
				entry = readdir(dp);
				// ".bin"で終わるファイルのみを列挙
				// →　連番でファイル生成するときにこの制約ちょっと嫌だな…。
				if (entry != NULL && StringExtension::EndsWith(entry->d_name, extension))
				{
					//cout << entry->d_name << endl;
					filenames.push_back(Path::Combine(sourceDirectory, entry->d_name));
				}
			} while (entry != NULL);
			closedir(dp);
		}
#endif

		return filenames;
	}

	// カレントフォルダを返す(起動時のフォルダ)
	// main関数に渡された引数から設定してある。
	// "GetCurrentDirectory"という名前はWindowsAPI(で定義されているマクロ)と競合する。
	std::string GetCurrentFolder() { return CommandLine::workingDirectory; }
}

// ----------------------------
//     mkdir wrapper
// ----------------------------

// working directory相対で指定する。
// フォルダを作成する。日本語は使っていないものとする。
// どうもMSYS2環境下のgccだと_wmkdir()だとフォルダの作成に失敗する。原因不明。
// 仕方ないので_mkdir()を用いる。
// ※　C++17のfilesystemがどの環境でも問題なく動くようになれば、
//     std::filesystem::create_directories()を用いて書き直すべき。

#if defined(_WIN32)
// Windows用

#if defined(_MSC_VER)

namespace Directory {
	Tools::Result CreateFolder(const std::string& dir_name)
	{
		// working folder相対で指定する。
		// working folderは本ソフトで変更していないので、普通に
		// mkdirすれば、working folderに作られるはずである。

		int result =  _wmkdir(Tools::MultiByteToWideChar(dir_name).c_str());
		//	::CreateDirectory(Tools::MultiByteToWideChar(dir_name).c_str(),NULL);

		return result == 0 ? Tools::Result::Ok() : Tools::Result(Tools::ResultCode::CreateFolderError);
	}
}

#elif defined(__GNUC__) 

#include <direct.h>
namespace Directory {
	Tools::Result CreateFolder(const std::string& dir_name)
	{
		int result = _mkdir(dir_name.c_str());
		return result == 0 ? Tools::Result::Ok() : Tools::Result(Tools::ResultCode::CreateFolderError);
	}
}

#endif
#elif defined(_LINUX)

// linux環境において、この_LINUXというシンボルはmakefileにて定義されるものとする。

// Linux用のmkdir実装。
#include "sys/stat.h"

namespace Directory {
	Tools::Result CreateFolder(const std::string& dir_name)
	{
		int result = ::mkdir(dir_name.c_str(), 0777);
		return result == 0 ? Tools::Result::Ok() : Tools::Result(Tools::ResultCode::CreateFolderError);
	}
}
#else

// Linux環境かどうかを判定するためにはmakefileを分けないといけなくなってくるな..
// Linuxでフォルダ掘る機能は、とりあえずナシでいいや..。評価関数ファイルの保存にしか使ってないし…。

namespace Directory {
	Tools::Result CreateFolder(const std::string& dir_name)
	{
		return Tools::Result(Tools::ResultCode::NotImplementedError);
	}
}

#endif

// ----------------------------
//     working directory
// ----------------------------

#ifdef _WIN32
#include <direct.h>
#define GETCWD _getcwd
#else
#include <unistd.h>
#define GETCWD getcwd
#endif

namespace CommandLine {

	string argv0;            // path+name of the executable binary, as given by argv[0]
	string binaryDirectory;  // path of the executable directory
	string workingDirectory; // path of the working directory

	void init(int argc, char* argv[]) {
		(void)argc;
		string pathSeparator;

		// extract the path+name of the executable binary
		argv0 = argv[0];

#ifdef _WIN32
		pathSeparator = "\\";
#ifdef _MSC_VER
		// Under windows argv[0] may not have the extension. Also _get_pgmptr() had
		// issues in some windows 10 versions, so check returned values carefully.
		char* pgmptr = nullptr;
		if (!_get_pgmptr(&pgmptr) && pgmptr != nullptr && *pgmptr)
			argv0 = pgmptr;
#endif
#else
		pathSeparator = "/";
#endif

		// extract the working directory
		workingDirectory = "";
		char buff[40000];
		char* cwd = GETCWD(buff, 40000);
		if (cwd)
			workingDirectory = cwd;

		// extract the binary directory path from argv0
		binaryDirectory = argv0;
		size_t pos = binaryDirectory.find_last_of("\\/");
		if (pos == std::string::npos)
			binaryDirectory = "." + pathSeparator;
		else
			binaryDirectory.resize(pos + 1);

		// pattern replacement: "./" at the start of path is replaced by the working directory
		if (binaryDirectory.find("." + pathSeparator) == 0)
			binaryDirectory.replace(0, 1, workingDirectory);
	}

}
