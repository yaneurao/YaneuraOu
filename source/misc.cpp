
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
	typedef bool(*fun4_t)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
	typedef WORD(*fun5_t)();
}

#endif

#include <fstream>
#include <iomanip>
//#include <iostream>
#include <sstream>
//#include <vector>
//#include <cstdlib>

#if defined(__linux__) && !defined(__ANDROID__)
#include <stdlib.h>
#include <sys/mman.h> // madvise()
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32)) || defined(__e2k__)
#define POSIXALIGNEDALLOC
#include <stdlib.h>
#endif

#include "misc.h"
#include "thread.h"

// === やねうら王独自追加

#include <ctime>				// std::ctime()
#include <cstring>				// std::memset()
#include <cstdio>				// fopen(),fread()
#include <cmath>				// std::exp()
#include "usi.h"				// Options
#include "testcmd/unit_test.h"	// UnitTester

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

	class Logger {

		// clangだとここ警告が出るので一時的に警告を抑制する。
#pragma warning (disable : 4068) // MSVC用の不明なpragmaの抑制
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
		Logger() : in(cin.rdbuf(), file.rdbuf()), out(cout.rdbuf(), file.rdbuf()) {}
#pragma clang diagnostic pop

		~Logger() { start(""); }

	public:
		// ログ記録の開始。
		// fname : ログを書き出すファイル名
		static void start(const std::string& name) {

			string fname = name;
			string upper_fname = StringExtension::ToUpper(fname);
			// 以前、"WriteDebugLog"オプションはチェックボックスになっていたので
			// GUIがTrue/Falseを渡してくることがある。
			if (upper_fname == "FALSE")
				fname = ""; // なかったことにする。
			else if (upper_fname == "TRUE")
				fname = "io_log.txt";

			static Logger l;

			if (l.file.is_open())
			{
				cout.rdbuf(l.out.buf);
				cin.rdbuf(l.in.buf);
				l.file.close();
			}

			if (!fname.empty())
			{
				l.file.open(fname, ifstream::out);

				if (!l.file.is_open())
				{
					cerr << "Unable to open debug log file " << fname << endl;
					exit(EXIT_FAILURE);
				}

				cin.rdbuf(&l.in);
				cout.rdbuf(&l.out);
			}
		}

	private:
		Tie in, out;   // 標準入力とファイル、標準出力とファイルのひも付け
		ofstream file; // ログを書き出すファイル
	};

} // 無名namespace

/// Trampoline helper to avoid moving Logger to misc.h
void start_logger(const std::string& fname) { Logger::start(fname); }

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

// config.hで設定した値などについて出力する。
const std::string config_info()
{
	std::string config = "\nconfigured by config.h";

	auto o  = [](std::string(p) , std::string(q)) { return "\n" + (p + std::string(20,' ')).substr(0,20) + " : " + q; };
	auto o1 = [&o](const char* p , u64  u ) { return o(std::string(p) , std::to_string(u) ); };
	auto o2 = [&o](const char* p , bool b ) { return o(std::string(p) , b ? "true":"false"); };

	config += o1("ASSERT_LV"           , ASSERT_LV      );
	config += o1("HASH_KEY_BITS"       , HASH_KEY_BITS  );
	config += o1("TT_CLUSTER_SIZE"     , TT_CLUSTER_SIZE);

	bool for_tournament = 
#if defined(FOR_TOURNAMENT)
		true;
#else
		false;
#endif

	bool test_cmd =
#if defined(ENABLE_TEST_CMD)
		true;
#else
		false;
#endif

	bool make_book_cmd = 
#if defined(ENABLE_MAKEBOOK_CMD)
		true;
#else
		false;
#endif

	bool use_super_sort =
#if defined(USE_SUPER_SORT)
		true;
#else
		false;
#endif

	bool tuning_parameters =
#if defined(TUNING_SEARCH_PARAMETERS)
		true;
#else
		false;
#endif

	bool global_options = 
#if defined(USE_GLOBAL_OPTIONS)
		true;
#else
		false;
#endif

	bool eval_learn =
#if defined(EVAL_LEARN)
		true;
#else
		false;
#endif

	bool use_mate_dfpn =
#if defined(USE_MATE_DFPN)
		true;
#else
		false;
#endif


	config += o2("PRETTY_JP"                , pretty_jp          );
	config += o2("FOR_TOURNAMENT"           , for_tournament     );
	config += o2("ENABLE_TEST_CMD"          , test_cmd           );
	config += o2("ENABLE_MAKEBOOK_CMD"      , make_book_cmd      );
	config += o2("USE_SUPER_SORT"           , use_super_sort     );
	config += o2("TUNING_SEARCH_PARAMETERS" , tuning_parameters  );
	config += o2("USE_GLOBAL_OPTIONS"       , global_options     );
	config += o2("EVAL_LEARN"               , eval_learn         );
	config += o2("USE_MATE_DFPN"            , use_mate_dfpn      );
	
	// コンパイラ情報もついでに出力する。
	//config += "\n\n" + compiler_info();

	// その他、欲しいものがあれば追加するかも。

	return config;
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

/// std_aligned_alloc() is our wrapper for systems where the c++17 implementation
/// does not guarantee the availability of aligned_alloc(). Memory allocated with
/// std_aligned_alloc() must be freed with std_aligned_free().

void* std_aligned_alloc(size_t alignment, size_t size) {

#if defined(POSIXALIGNEDALLOC)
	void* mem;
	return posix_memalign(&mem, alignment, size) ? nullptr : mem;
#elif defined(_WIN32)
	return _mm_malloc(size, alignment);
#else
	return std::aligned_alloc(alignment, size);
#endif
}

void std_aligned_free(void* ptr) {

#if defined(POSIXALIGNEDALLOC)
	free(ptr);
#elif defined(_WIN32)
	_mm_free(ptr);
#else
	free(ptr);
#endif
}

// Windows
#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows(size_t allocSize) {

	// Windows 64bit用専用。
	// Windows 32bit用ならこの機能は利用できない。
	#if !defined(_WIN64)
		(void)allocSize; // suppress unused-parameter compiler warning
		return nullptr;
	#else

	// ※ やねうら王独自拡張
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

	#endif
}

void* aligned_large_pages_alloc(size_t allocSize) {

	// ※　ここでは4KB単位でalignされたメモリが返ることは保証されているので
	//     引数でalignを指定できる必要はない。(それを超えた大きなalignを行いたいケースがない)

	//static bool firstCall = true;

	// try to allocate large pages
	void* ptr = aligned_large_pages_alloc_windows(allocSize);

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
		if (ptr)
			sync_cout << "info string Hash table allocation: Windows Large Pages used." << sync_endl;
		else
			sync_cout << "info string Hash table allocation: Windows Large Pages not used." << sync_endl;

		largeMemoryAllocFirstCall = false;
	}

	// fall back to regular, page aligned, allocation if necessary
	// 4KB単位であることは保証されているはず..
	if (!ptr)
		ptr = VirtualAlloc(NULL, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

	// VirtualAlloc()はpage size(4KB)でalignされていること自体は保証されているはず。

	//cout << (u64)mem << "," << allocSize << endl;

	return ptr;
}

#else
// LargePage非対応の環境であれば、std::aligned_alloc()を用いて確保しておく。
// 最低でも4KBでalignされたメモリが返るので、引数でalignを指定できるようにする必要はない。

void* aligned_large_pages_alloc(size_t allocSize) {

#if defined(__linux__)
	constexpr size_t alignment = 2 * 1024 * 1024; // assumed 2MB page size
#else
	constexpr size_t alignment = 4096; // assumed small page size
#endif

	// round up to multiples of alignment
	size_t size = ((allocSize + alignment - 1) / alignment) * alignment;
	void* mem = std_aligned_alloc(alignment, size);
#if defined(MADV_HUGEPAGE)
	madvise(mem, size, MADV_HUGEPAGE);
#endif

	return mem;
}

#endif

/// aligned_large_pages_free() will free the previously allocated ttmem

#if defined(_WIN32)

void aligned_large_pages_free(void* mem) {

	if (mem && !VirtualFree(mem, 0, MEM_RELEASE))
	{
		DWORD err = GetLastError();
		std::cerr << "Failed to free large page memory. Error code: 0x"
			<< std::hex << err
			<< std::dec << std::endl;
		exit(EXIT_FAILURE);
	}
}

#else

void aligned_large_pages_free(void* mem) {
	std_aligned_free(mem);
}

#endif

// --------------------
//  LargeMemory class
// --------------------

// メモリを確保する。Large Pageに確保できるなら、そこにする。
// aligned_ttmem_alloc()を内部的に呼び出すので、アドレスは少なくとも2MBでalignされていることは保証されるが、
// 気になる人のためにalignmentを明示的に指定できるようになっている。
// メモリ確保に失敗するか、引数のalignで指定したalignmentになっていなければ、
// エラーメッセージを出力してプログラムを終了させる。
void* LargeMemory::alloc(size_t size, size_t align , bool zero_clear)
{
	free();
	return static_alloc(size, align, zero_clear);
}

// alloc()で確保したメモリを開放する。
// このクラスのデストラクタからも自動でこの関数が呼び出されるので明示的に呼び出す必要はない(かも)
void LargeMemory::free()
{
	static_free(ptr);
	ptr = nullptr;
}

// alloc()のstatic関数版。memには、static_free()に渡すべきポインタが得られる。
void* LargeMemory::static_alloc(size_t size, size_t align, bool zero_clear)
{
	void* mem = aligned_large_pages_alloc(size);

	auto error_exit = [&](std::string mes) {
		sync_cout << "info string Error! : " << mes << " in LargeMemory::alloc(" << size << "," << align << ")" << sync_endl;
		Tools::exit();
	};

	// メモリが正常に確保されていることを保証する
	if (mem == nullptr)
		error_exit("can't alloc enough memory.");

	// ptrがalignmentされていることを保証する
	if ((reinterpret_cast<size_t>(mem) % align) != 0)
		error_exit("can't alloc algined memory.");

	// ゼロクリアが必要なのか？
	if (zero_clear)
	{
		// 確保したのが256MB以上なら並列化してゼロクリアする。
		if (size < 256 * 1024 * 1024)
			// そんなに大きな領域ではないから、普通にmemset()でやっとく。
			std::memset(mem, 0, size);
		else
			// 並列版ゼロクリア
			Tools::memclear(nullptr, mem, size);
	}

	return mem;
}

// static_alloc()で確保したメモリを開放する。
void LargeMemory::static_free(void* mem)
{
	aligned_large_pages_free(mem);
}



// --------------------
//  全プロセッサを使う
// --------------------

namespace WinProcGroup {

#if !defined ( _WIN32 )

	void bindThisThread(size_t) {}

#else


	/// best_node() retrieves logical processor information using Windows specific
	/// API and returns the best node id for the thread with index idx. Original
	/// code from Texel by Peter Österlund.

	int best_node(size_t idx) {

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

		// First call to GetLogicalProcessorInformationEx() to get returnLength.
		// We expect the call to fail due to null buffer.
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
		int node = best_node(idx);

		if (node == -1)
			return;

		// Early exit if the needed API are not available at runtime
		HMODULE k32 = GetModuleHandle(L"Kernel32.dll");
		auto fun2 = (fun2_t)(void(*)())GetProcAddress(k32, "GetNumaNodeProcessorMaskEx");
		auto fun3 = (fun3_t)(void(*)())GetProcAddress(k32, "SetThreadGroupAffinity");
		auto fun4 = (fun4_t)(void(*)())GetProcAddress(k32, "GetNumaNodeProcessorMask2");

		if (!fun2 || !fun3)
			return;

		if (!fun4) {
			GROUP_AFFINITY affinity;
			if (fun2(node, &affinity))
				fun3(GetCurrentThread(), &affinity, nullptr);
		} else {
			// If a numa node has more than one processor group, we assume they are
			// sized equal and we spread threads evenly across the groups.
			USHORT elements, returnedElements;
			elements = GetMaximumProcessorGroupCount();
			GROUP_AFFINITY *affinity = (GROUP_AFFINITY*)malloc(
				elements * sizeof(GROUP_AFFINITY));
			if (fun4(node, affinity, elements, &returnedElements))
				fun3(GetCurrentThread(), &affinity[idx % returnedElements], nullptr);
			free(affinity);
		}
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

		// Options["Threads"]が使用できるスレッド数とは限らない(ふかうら王など)
		auto thread_num = (size_t)Threads.size(); // Options["Threads"];

		if (name_ != nullptr)
			sync_cout << "info string " + std::string(name_) + " : Start clearing with " <<  thread_num << " threads , Hash size =  " << size / (1024 * 1024) << "[MB]" << sync_endl;

		// マルチスレッドで並列化してクリアする。

		std::vector<std::thread> threads;

		for (size_t idx = 0; idx < thread_num; idx++)
		{
			threads.push_back(std::thread([table, size, thread_num, idx]() {

				// NUMA環境では、bindThisThread()を呼び出しておいたほうが速くなるらしい。

				// Thread binding gives faster search on systems with a first-touch policy
				if (thread_num > 8)
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
			sync_cout << "info string " + std::string(name_) + " : Finish clearing." << sync_endl;
	}

	// 途中での終了処理のためのwrapper
	// コンソールの出力が完了するのを待ちたいので3秒待ってから::exit(EXIT_FAILURE)する。
	void exit()
	{
		sleep(3000); // エラーメッセージが出力される前に終了するのはまずいのでwaitを入れておく。
		::exit(EXIT_FAILURE);
	}

	// 指定されたミリ秒だけsleepする。
	void sleep(u64 ms)
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

	// size_ : 全件でいくらあるかを設定する。
	ProgressBar::ProgressBar(u64 size_) : size(size_)
	{
		if (enable_)
			cout << "0% [";
		dots = 0;
	}

	// 進捗を出力する。
	// current : 現在までに完了している件数
	void ProgressBar::check(u64 current)
	{
		if (!enable_)
			return;

		// 何個dotを打つべきか。
		const size_t all_dots = 70; // 100%になった時に70個打つ。

		// 何dot塗りつぶすのか。
		size_t d = (size == 0) ? all_dots : std::min((size_t)(all_dots * current / size), all_dots);

		for (; dots < d ; ++dots)
			cout << ".";

		if (dots == all_dots)
		{
			cout << "] 100%" << endl;
			dots++; // 1加算しておけば完了したことがわかる。
		}
	}
	bool ProgressBar::enable_ = false;


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
//  ファイル操作
// --------------------

namespace SystemIO
{

	// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。末尾の改行は除去される。
	// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
	Tools::Result ReadAllLines(const std::string& filename, std::vector<std::string>& lines, bool trim)
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

		TextReader reader;

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

	Tools::Result ReadFileToMemory(const std::string& filename, std::function<void* (size_t)> callback_func)
	{
		// fstream、遅いので、FILEを用いて書き換える。

		FILE* fp = fopen(filename.c_str(), "rb");
		if (fp == nullptr)
			return Tools::Result(Tools::ResultCode::FileOpenError);

		fseek(fp, 0, SEEK_END);
		size_t endPos = (size_t)ftell(fp);
		fseek(fp, 0, SEEK_SET);
		size_t beginPos = (size_t)ftell(fp);
		size_t file_size = endPos - beginPos;

		// ファイルサイズがわかったのでcallback_funcを呼び出してこの分のバッファを確保してもらい、
		// そのポインターをもらう。
		void* ptr = callback_func(file_size);

		// バッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
		// nullptrを返すことになっている。このとき、読み込みを中断し、エラーリターンする。
		// 原因は不明だが、メモリ割り当ての失敗なのでMemoryAllocationErrorを返しておく。
		if (ptr == nullptr)
			return Tools::Result(Tools::ResultCode::MemoryAllocationError);

		// 細切れに読み込む

		const size_t block_size = 1024 * 1024 * 1024; // 1回のreadで読み込む要素の数(1GB)
		for (size_t pos = 0; pos < file_size; pos += block_size)
		{
			// 今回読み込むサイズ
			size_t read_size = (pos + block_size < file_size) ? block_size : (file_size - pos);

			if (fread((u8*)ptr + pos, 1, read_size, fp) != read_size)
				// 指定サイズだけ読み込めていないということは、読み込み上のエラーである。
				return Tools::Result(Tools::ResultCode::FileReadError);

			//cout << ".";
		}
		fclose(fp);

		return Tools::Result::Ok();
	}


	Tools::Result WriteMemoryToFile(const std::string& filename, void* ptr, size_t size)
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

	// 通常のftell/fseekは2GBまでしか対応していないので特別なバージョンが必要である。
	// 64bit環境でないと対応していない。まあいいや…。

	size_t ftell64(FILE* f)
	{
#if defined(_MSC_VER)
		return _ftelli64(f);
#elif defined(__GNUC__) && defined(IS_64BIT) && !(defined(__ANDROID__) && defined(__ANDROID_API__) && __ANDROID_API__ < 24) && !defined(__MACH__)
		return ftello64(f);
#else
		return ftell(f);
#endif
	}

	int fseek64(FILE* f, size_t offset, int origin)
	{
#if defined(_MSC_VER)
		return _fseeki64(f, offset, origin);
#elif defined(__GNUC__) && defined(IS_64BIT) && !(defined(__ANDROID__) && defined(__ANDROID_API__) && __ANDROID_API__ < 24) && !defined(__MACH__)
		return fseeko64(f, offset, origin);
#else
		return fseek(f, offset, origin);
#endif
	}

	// --- TextFileReader

	// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
	// fopen()～fread()で実装されている。
	TextReader::TextReader()
	{
		buffer.resize(1024 * 1024);
		line_buffer.reserve(2048);
		clear();

		// この２つのフラグはOpen()したときに設定がクリアされるべきではないので、
		// コンストラクタで一度だけ初期化する。
		trim = false;
		skipEmptyLine = false;
	}

	TextReader::~TextReader()
	{
		Close();
	}

	// 各種状態変数の初期化
	void TextReader::clear()
	{
		fp = nullptr;
		is_eof = false;
		cursor = 0;
		read_size = 0;
		is_prev_cr = false;
		line_number = 0;
	}

	// ファイルをopenする。
	Tools::Result TextReader::Open(const std::string& filename)
	{
		Close();

		// 高速化のためにbinary open
		fp = fopen(filename.c_str(), "rb");
		return (fp == nullptr) ? Tools::Result(Tools::ResultCode::FileOpenError) : Tools::Result::Ok();
	}

	// Open()を呼び出してオープンしたファイルをクローズする。
	void TextReader::Close()
	{
		if (fp != nullptr)
			fclose(fp);

		clear();
	}

	// バッファから1文字読み込む。eofに達したら、-1を返す。
	int TextReader::read_char()
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
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	Tools::Result TextReader::read_line_simple()
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
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	Tools::Result TextReader::ReadLine(std::string& line)
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

			// ファイル先頭のBOMは読み飛ばす
			size_t skip_byte = 0;
			if (line_number == 0)
				// UTF-8 BOM (EF BB BF)
				if (line_buffer.size() >= 3 && line_buffer[0] == 0xef && line_buffer[1] == 0xbb && line_buffer[2] == 0xbf)
					skip_byte = 3;
			// 他のBOMも読み飛ばしても良いが、まあいいや…。

			// この1行のbyte数(BOMは含まず)
			size_t line_size = line_buffer.size() - skip_byte;

			// この時点で1行読み込んだことになるので(行をskipしても1行とカウントするので)行番号をインクリメントしておく。
			line_number++;

			// 空行をスキップするモートであるなら、line_sizeが結果的に空になった場合は次の行を調べる。
			if (skipEmptyLine && line_size == 0)
				continue;

			line = std::string((const char*)line_buffer.data() + skip_byte, line_size );
			return Tools::ResultCode::Ok;
		}
	}

	// 次のblockのbufferへの読み込み。
	void TextReader::read_next_block()
	{
		if (::feof(fp))
			read_size = 0;
		else
			read_size = ::fread(&buffer[0], sizeof(u8), buffer.size(), fp);

		// カーソル(解析位置)のリセット
		cursor = 0;

		// 読み込まれたサイズが0なら、終端に達したと判定する。
		is_eof = read_size == 0;
	}

	// ファイルサイズの取得
	// ファイルポジションは先頭に移動する。
	size_t TextReader::GetSize()
	{
		ASSERT_LV3(fp != nullptr);

		fseek64(fp, 0, SEEK_END);
		// ftell()は失敗した時に-1を返すらしいのだが…。ここでは失敗を想定していない。
		size_t endPos = ftell64(fp);
		fseek64(fp, 0, SEEK_SET);
		size_t beginPos = ftell64(fp);
		size_t file_size = endPos - beginPos;

		return file_size;
	}

	// === TextWriter ===

	Tools::Result TextWriter::Open(const std::string& filename)
	{
		Close();
		fp = fopen(filename.c_str(), "wb");
		return fp == nullptr ? Tools::ResultCode::FileOpenError
                             : Tools::ResultCode::Ok;
	}

	// 文字列を書き出す(改行コードは書き出さない)
	Tools::Result TextWriter::Write(const std::string& str)
	{
		return Write(str.c_str(), str.size());
	}

	// 1行を書き出す(改行コードも書き出す) 改行コードは"\r\n"とする。
	Tools::Result TextWriter::WriteLine(const std::string& line)
	{
		auto result = Write(line.c_str(), line.size());
		if (result.is_not_ok())
			return result;

		// 改行コードも書き出す。
		return Write("\r\n", (size_t)2);
	}

	// ptrの指すところからsize [byte]だけ書き出す。
	Tools::Result TextWriter::Write(const char* ptr, size_t size)
	{
		// Openしていなければ書き出せない。
		if (fp == nullptr)
			return Tools::ResultCode::FileWriteError;

		// 書き込みカーソルの終端がどこに来るのか。
		size_t write_cursor_end = write_cursor + size;
		char* ptr2 = const_cast<char*>(ptr);

		size_t write_size;
		while (write_cursor_end >= buf_size)
		{
			// とりあえず、書けるだけ書いてfwriteする。

			// 今回のループで書き込むbyte数
			write_size = buf_size - write_cursor;
			std::memcpy(&buf[write_cursor], ptr2, write_size);
			if (fwrite(&buf[0], buf_size, 1, fp) == 0)
				return Tools::ResultCode::FileWriteError;

			// buf[0..write_cursor-1]が窓で、ループごとにその窓がbuf_sizeずつずれていくと考える。
			// 例えば、ループ2回目ならbuf[write_cursor..write_cursor*2-1]が窓だと考える。

			ptr2             += write_size;
			size             -= write_size;
			write_cursor_end -= buf_size;
			write_cursor      = 0;
		}
		std::memcpy(&buf[write_cursor], ptr2, size);
		write_cursor += size;

		return Tools::ResultCode::Ok;
	}

	// 内部バッファにあってまだファイルに書き出していないデータをファイルに書き出す。
	// ※　Close()する時に呼び出されるので通常この関数を呼び出す必要はない。
	Tools::Result TextWriter::Flush()
	{
		// Openしていなければ書き出せない。
		if (fp == nullptr)
			return Tools::ResultCode::FileWriteError;

		// bufのwrite_cursorの指している手前までを書き出す。
		if (write_cursor > 0 && fwrite(&buf[0], write_cursor, 1, fp) == 0)
			return Tools::ResultCode::FileWriteError;

		write_cursor = 0;
		return Tools::ResultCode::Ok;
	}

	Tools::Result TextWriter::Close()
	{
		if (fp)
		{
			// バッファ、まだflushが終わっていないデータがあるならそれをflushする。
			if (Flush().is_not_ok())
				return Tools::ResultCode::FileWriteError;

			fclose(fp); // GetLastErrorでエラーを取得することはできるが…。
			fp = nullptr;
		}
		return Tools::ResultCode::Ok;
	}

	// === BinaryBase ===

	// ファイルを閉じる。デストラクタからclose()は呼び出されるので明示的に閉じなくても良い。
	Tools::Result BinaryBase::Close()
	{
		Tools::ResultCode result = Tools::ResultCode::Ok;
		if (fp != nullptr)
		{
			if (fclose(fp) != 0)
				result = Tools::ResultCode::FileCloseError;
			fp = nullptr;
		}
		return Tools::Result(result);
	}

	// === BinaryReader ===

	// ファイルのopen
	Tools::Result BinaryReader::Open(const std::string& filename)
	{
		auto close_result = Close();
		if (!close_result.is_ok()) {
			return close_result;
		}

		fp = fopen(filename.c_str(), "rb");
		if (fp == nullptr)
			return Tools::Result(Tools::ResultCode::FileOpenError);

		return Tools::Result::Ok();
	}

	// ファイルサイズの取得
	// ファイルポジションは先頭に移動する。
	size_t BinaryReader::GetSize()
	{
		ASSERT_LV3(fp != nullptr);

		fseek64(fp, 0, SEEK_END);
		size_t endPos = ftell64(fp);
		fseek64(fp, 0, SEEK_SET);
		size_t beginPos = ftell64(fp);
		size_t file_size = endPos - beginPos;

		return file_size;
	}

	// ptrの指すメモリにsize[byte]だけファイルから読み込む
	Tools::Result BinaryReader::Read(void* ptr, size_t size, size_t* size_of_read_bytes)
	{
		size_t actual_size_of_read_bytes = fread(ptr, 1, size, fp);

		if (size_of_read_bytes) {
			*size_of_read_bytes = actual_size_of_read_bytes;
		}

		if (feof(fp)) {
			// ファイルの末尾を超えて読もうとした場合。
			return Tools::Result(Tools::ResultCode::Eof);
		}

		if (actual_size_of_read_bytes != size)
			return Tools::Result(Tools::ResultCode::FileReadError);

		// ファイルの末尾を超えて読もうとしなかった場合。
		// ファイルの末尾までちょうどを読んだ場合はこちら。
		return Tools::Result::Ok();
	}

	// === BinaryWriter ===

	// ファイルのopen
	Tools::Result BinaryWriter::Open(const std::string& filename)
	{
		fp = fopen(filename.c_str(), "wb");
		if (fp == nullptr)
			return Tools::Result(Tools::ResultCode::FileOpenError);

		return Tools::Result::Ok();
	}

	// ptrの指すメモリからsize[byte]だけファイルに書き込む
	Tools::Result BinaryWriter::Write(void* ptr, size_t size)
	{
		if (fwrite((u8*)ptr, 1, size, fp) != size)
			return Tools::Result(Tools::ResultCode::FileWriteError);

		return Tools::Result::Ok();
	}
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
		// 与えられたfileが絶対Pathであるかの判定
		if (IsAbsolute(filename))
			return filename;

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

	// 絶対Pathであるかの判定。
	// "\\"(WindowsのUNC)で始まるか、"/"で始まるか(Windows / Linuxのroot)、"~"で始まるか、"C:"(ドライブレター + ":")で始まるか。
	bool IsAbsolute(const std::string& path)
	{
		// path separator
		const auto path_char1 = '\\';
		const auto path_char2 = '/';

		// home directory
		const auto home_char  = '~';

		// dirve letter separator
		const auto drive_char = ':';

		if (path.length() >= 1)
		{
			const char c = path[0];
			if (c == path_char1 || c == path_char2 || c == home_char)
				return true;

			// 2文字目が":"なら1文字目をチェックしなくとも良いかな？
			if (path.length() >= 2 && path[1] == drive_char)
				return true;
		}

		return false;
	}
};

// --------------------
//    Directory
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


// --------------------
//       Parser
// --------------------

namespace Parser
{

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
		Parser::LineScanner scanner(input);
		while (!scanner.eol())
			result.push_back(scanner.get_text());

		return result;
	}

	// 先頭にゼロサプライした文字列を返す。
	// 例) n = 123 , digit = 6 なら "000123"という文字列が返る。
	std::string to_string_with_zero(u64 n, int digit)
	{
		// 現在の状態
		std::ios::fmtflags curret_flag = std::cout.flags();

		std::ostringstream ss;
		ss << std::setw(digit) << std::setfill('0') << n;
		string s(ss.str());

		// 状態を戻す
		std::cout.flags(curret_flag);
		return s;
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

	// 文字列valueに対して文字xを文字yに置換した新しい文字列を返す。
	std::string Replace(std::string const& value, char x, char y)
	{
		std::string r(value);
		for (size_t i = 0; i < r.size(); ++i)
			if (r[i] == x)
				r[i] = y;
		return r;
	}

	// 文字列を大文字にして返す。
	std::string ToUpper(std::string const& value)
	{
		std::string s(value);
		transform(s.begin(), s.end(), s.begin(),
			[](unsigned char c){ return toupper(c); });
		return s;
	}

	// sを文字列sepで分割した文字列集合を返す。
	std::vector<std::string> Split(const std::string& s, const std::string& sep)
	{
		std::vector<std::string> v;
		string ss = s;
		size_t p = 0; // 前回の分割場所
		while (true)
		{
			size_t pos = ss.find(sep , p);
			if (pos == string::npos)
			{
				// sepが見つからなかったのでこれでおしまい。
				v.emplace_back(ss.substr(p));
				break;
			}
			v.emplace_back(ss.substr(p, pos - p));
			p = pos + sep.length();
		}
		return v;
	}

};

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

// --------------------
//     UnitTest
// --------------------

namespace Misc {
	// このheaderに書いてある関数のUnitTest。
	void UnitTest(Test::UnitTester& tester)
	{
		auto section1 = tester.section("Misc");

		{
			auto section2 = tester.section("Path");

			{
				auto section3 = tester.section("Combine");

				tester.test("Absolute Path Root1",        Path::Combine("xxxx"  , "/dir"   ) == "/dir"     );
				tester.test("Absolute Path Root2",        Path::Combine("xxxx"  , "\\dir"  ) == "\\dir"    );
				tester.test("Absolute Path Home",         Path::Combine("xxxx"  , "~dir"   ) == "~dir"     );
				tester.test("Absolute Path Drive Letter", Path::Combine("xxxx"  , "c:\\dir") == "c:\\dir"  );
				tester.test("Absolute Path UNC",          Path::Combine("xxxx"  , "\\\\dir") == "\\\\dir"  );
				tester.test("Relative Path1",             Path::Combine("xxxx"  , "yyy"    ) == "xxxx/yyy" );
				tester.test("Relative Path2",             Path::Combine("xxxx/" , "yyy"    ) == "xxxx/yyy" );
				tester.test("Relative Path3",             Path::Combine("xxxx\\", "yyy"    ) == "xxxx\\yyy");
			}

		}
		{
			auto section2 = tester.section("StringExtension");
			{
				tester.test("to_string_with_zero", StringExtension::to_string_with_zero(123 , 6) == "000123");
				tester.test("to_string_with_zero", StringExtension::to_string_with_zero(1234, 6) == "001234");
				tester.test("ToUpper"            , StringExtension::ToUpper("False&True") == "FALSE&TRUE");

				auto v = StringExtension::Split("ABC ; DEF ; GHI", " ; ");
				tester.test("Split"              , v[0]=="ABC" && v[1]=="DEF" && v[2] =="GHI");
			}
		}
	}
}
