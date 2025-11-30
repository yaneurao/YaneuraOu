
// Windows環境下でのプロセッサグループの割当関係
#ifdef _WIN32
#if _WIN32_WINNT < 0x0601
#undef  _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Force to include needed API prototypes
#endif

// windows.hのなかでmin,maxを定義してあって、C++のmin,maxと衝突して困る。
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
using fun1_t = bool(*)(LOGICAL_PROCESSOR_RELATIONSHIP,
                       PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
using fun2_t = bool(*)(USHORT, PGROUP_AFFINITY);
using fun3_t = bool(*)(HANDLE, CONST GROUP_AFFINITY*, PGROUP_AFFINITY);
using fun4_t = bool(*)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
using fun5_t = WORD(*)();
using fun6_t = bool(*)(HANDLE, DWORD, PHANDLE);
using fun7_t = bool(*)(LPCSTR, LPCSTR, PLUID);
using fun8_t = bool(*)(HANDLE, BOOL, PTOKEN_PRIVILEGES, DWORD, PTOKEN_PRIVILEGES, PDWORD);
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
#include <filesystem>           // create_directory()
#include "usi.h"				// Options
#include "thread.h"             // ThreadPool
#include "testcmd/unit_test.h"	// UnitTester

using namespace std;

namespace YaneuraOu {
namespace {

// --------------------
//  logger
// --------------------

// Our fancy logging facility. The trick here is to replace cin.rdbuf() and
// cout.rdbuf() with two Tie objects that tie cin and cout to a file stream. We
// can toggle the logging of cout and std:cin at runtime whilst preserving
// usual I/O functionality, all without changing a single line of code!
// Idea from http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

// logging用のhack。streambufをこれでhookしてしまえば追加コードなしで普通に
// cinからの入力とcoutへの出力をファイルにリダイレクトできる。
// cf. http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

struct Tie : public streambuf {  // MSVC requires split streambuf for cin and cout

	Tie(streambuf* b, streambuf* l) :
		buf(b),
		logBuf(l) {
	}

	int sync() override { return logBuf->pubsync(), buf->pubsync(); }
	int overflow(int c) override { return log(buf->sputc(char(c)), "<< "); }
	int underflow() override { return buf->sgetc(); }
	int uflow() override { return log(buf->sbumpc(), ">> "); }

	streambuf *buf, *logBuf;

	int log(int c, const char* prefix) {

		static int last = '\n';  // Single log file

		if (last == '\n')
			logBuf->sputn(prefix, 3);

		return last = logBuf->sputc(char(c));
	}
};

class Logger {

	Logger() :
		in(cin.rdbuf(), file.rdbuf()),
		out(cout.rdbuf(), file.rdbuf()) {
	}
	~Logger() { start(""); }

	ofstream file;    // ログを書き出すファイル
	Tie           in, out; // 標準入力とファイル、標準出力とファイルのひも付け

public:
	// ログ記録の開始。
	// fname : ログを書き出すファイル名
	static void start(const string& fname) {

		string fname2 = fname;
		string upper_fname = StringExtension::ToUpper(fname2);
		// 以前、"WriteDebugLog"オプションはチェックボックスになっていたので
		// GUIがTrue/Falseを渡してくることがある。
		if (upper_fname == "FALSE")
			fname2 = ""; // なかったことにする。
		else if (upper_fname == "TRUE")
			fname2 = "io_log.txt";

		static Logger l;

		if (l.file.is_open())
		{
			cout.rdbuf(l.out.buf);
			cin.rdbuf(l.in.buf);
			l.file.close();
		}

		if (!fname2.empty())
		{
			l.file.open(fname2, ifstream::out);

			if (!l.file.is_open())
			{
				cerr << "Unable to open debug log file " << fname2 << endl;
				exit(EXIT_FAILURE);
			}

			cin.rdbuf(&l.in);
			cout.rdbuf(&l.out);
		}
	}
};

} // namespace

/// Trampoline helper to avoid moving Logger to misc.h
void start_logger(const string& fname) { Logger::start(fname); }

// --------------------
//  engine info
// --------------------

// Returns the full name of the current Stockfish version.
//
// For local dev compiles we try to append the commit SHA and
// commit date from git. If that fails only the local compilation
// date is set and "nogit" is specified:
//      Stockfish dev-YYYYMMDD-SHA
//      or
//      Stockfish dev-YYYYMMDD-nogit
//
// For releases (non-dev builds) we only include the version number:
//      Stockfish version

// 現在のStockfishのバージョンのフルネームを返します。
//
// ローカルの開発用ビルドでは、gitからコミットSHAと
// コミット日を付加しようとします。これに失敗した場合は、
// ローカルのコンパイル日だけが設定され、「nogit」が指定されます：
//      Stockfish dev-YYYYMMDD-SHA
//      または
//      Stockfish dev-YYYYMMDD-nogit
//
// リリース（開発版でない）ビルドでは、バージョン番号のみを含めます：
//      Stockfish version

std::string engine_version_info() {
	std::stringstream ss;
#if STOCKFISH
	ss << "Stockfish " << version << std::setfill('0');
#else
    ss << "YaneuraOu" << ENGINE_VERSION;
#endif

	// "dev"版であれば日付を出力する機能。
#if STOCKFISH
	if constexpr (version == "dev")
#else
    if (StringExtension::Contains(ENGINE_VERSION, "dev"))
#endif
	{
		ss << "-";
#ifdef GIT_DATE
		ss << stringify(GIT_DATE);
#else
		constexpr std::string_view months("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec");

		std::string       month, day, year;
		std::stringstream date(__DATE__);  // From compiler, format is "Sep 21 2008"

		date >> month >> day >> year;
		ss << year << std::setw(2) << std::setfill('0') << (1 + months.find(month) / 4)
			<< std::setw(2) << std::setfill('0') << day;
#endif

		ss << "-";

#ifdef GIT_SHA
		ss << stringify(GIT_SHA);
#else
		ss << "nogit";
#endif
	}

	return ss.str();
}

std::string engine_info(const std::string& engine_name,
                        const std::string& engine_author,
						const std::string& engine_version,
                        const std::string& eval_name)
{
#if STOCKFISH
    return engine_version_info() + (to_uci ? "\nid author " : " by ")
         + "the Stockfish developers (see AUTHORS file)";
#endif
    // → これ好きじゃない。

	stringstream ss;
	string engine_name_, engine_author_;

	// カレントフォルダに"engine_name.txt"があればその1行目をエンジン名とする機能
	ifstream ifs("engine_name.txt");
	if (!ifs.fail())
	{
		// 1行目が読み込めなかったときのためにデフォルト値を設定しておく。
        engine_name_ = "default engine";
        Tools::getline(ifs, engine_name_);

		// 2行目が読み込めなかったときのためにデフォルト値を設定しておく。
        engine_author_ = "default author";
        Tools::getline(ifs, engine_author_);
	}
	else
	{
		engine_name_ =
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
			engine_name
#endif
            + (eval_name.empty() ? "" : std::string(" ") + eval_name)
			+ ' ' + engine_version
			+ ' ' + (Is64Bit ? "64" : "32")
			+ TARGET_CPU
#if defined(FOR_TOURNAMENT)
			+" TOURNAMENT"
#endif

#if defined(EVAL_LEARN)
			+" EVAL_LEARN"
#endif
			;
			engine_author_ = engine_author;
                // やねうら王 "yaneurao";
                // ふかうら王 "Tadao Yamaoka , yaneurao";
	}

	return engine_name_ + "\n" + "id author " + engine_author_; 
}

// 使用したコンパイラについての文字列を返す。
string compiler_info() {

#define stringify2(x) #x
#define stringify(x) stringify2(x)
#define make_version_string(major, minor, patch) stringify(major) "." stringify(minor) "." stringify(patch)

	/// Predefined macros hell:
	///
	/// __GNUC__				Compiler is gcc, Clang or ICX
	/// __clang__               Compiler is Clang or ICX
	/// __INTEL_LLVM_COMPILER   Compiler is ICX
	/// _MSC_VER				Compiler is MSVC
	/// _WIN32					Building on Windows (any)
	/// _WIN64					Building on Windows 64 bit

	string compiler = "\nCompiled by ";

#if defined(__INTEL_LLVM_COMPILER)
	compiler += "ICX ";
	compiler += stringify(__INTEL_COMPILER) " update " stringify(__INTEL_COMPILER_UPDATE);
#elif defined(__clang__)
	compiler += "clang++ ";
	compiler += make_version_string(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif _MSC_VER
	compiler += "MSVC ";
	compiler += "(version ";
	compiler += stringify(_MSC_FULL_VER) "." stringify(_MSC_BUILD);
	compiler += ")";
#elif defined(__e2k__) && defined(__LCC__)
	#define dot_ver2(n) \
        compiler += char('.'); \
        compiler += char('0' + (n) / 10); \
        compiler += char('0' + (n) % 10);

	compiler += "MCST LCC ";
	compiler += "(version ";
	compiler += to_string(__LCC__ / 100);
	dot_ver2(__LCC__ % 100) dot_ver2(__LCC_MINOR__) compiler += ")";

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

	compiler += "\nCompilation architecture   : ";
#if defined(ARCH)
	compiler += stringify(ARCH);
#else
	compiler += "(undefined architecture)";
#endif

	compiler += "\nCompilation settings       : ";
	compiler += (Is64Bit ? "64bit" : "32bit");
#if defined(USE_VNNI)
	compiler += " VNNI";
#endif
#if defined(USE_AVX512)
	compiler += " AVX512";
#endif

//	compiler += (HasPext ? " BMI2" : "");
// ⇨ このフラグ、やねうら王では持っていない。

#if defined(USE_AVX2)
	compiler += " AVX2";
#endif
#if defined(USE_SSE41)
	compiler += " SSE41";
#endif
#if defined(USE_SSSE3)
	compiler += " SSSE3";
#endif
#if defined(USE_SSE2)
	compiler += " SSE2";
#endif
#if defined(USE_NEON_DOTPROD)
	compiler += " NEON_DOTPROD";
#elif defined(USE_NEON)
	compiler += " NEON";
#endif

#if STOCKFISH
	compiler += (HasPopCnt ? " POPCNT" : "");
	// ⇨ このフラグ、やねうら王では持っていない。
#endif

#if !defined(NDEBUG)
	compiler += " DEBUG";
#endif

	compiler += "\nCompiler __VERSION__ macro : ";
#ifdef __VERSION__
	// __VERSION__が定義されているときだけ、その文字列を出力する。(MSVCだと定義されていないようだ..)
	compiler += __VERSION__;
#else
	compiler += "(undefined macro)";
#endif

	compiler += "\n";

	return compiler;
}

// config.hで設定した値などについて出力する。
string config_info()
{
	string config = "\nconfigured by config.h";

	auto o  = [](string(p) , string(q)) { return "\n" + (p + string(20,' ')).substr(0,20) + " : " + q; };
	auto o1 = [&o](const char* p , u64  u ) { return o(string(p) , std::to_string(u) ); };
	auto o2 = [&o](const char* p , bool b ) { return o(string(p) , b ? "true":"false"); };

	// 評価関数タイプ
	string eval_type =
#if defined(YANEURAOU_ENGINE_DEEP)
	"DEEP";
#elif defined(YANEURAOU_ENGINE_NNUE)

	// NNUE
	#if defined(NNUE_ARCHITECTURE_HEADER)
		NNUE_ARCHITECTURE_HEADER;
	#elif defined(EVAL_NNUE_HALFKP256)
		"halfkp_256x2_32_32";
	#elif defined(EVAL_NNUE_KP256)
		"kp_256x2_32_32";
	#elif defined(EVAL_NNUE_HALFKPE9)
		"halfkpe9_256x2_32_32";
	#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32)
		"halfkp_512x2_16_32";
	#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_32)
		"halfkp_1024x2_8_32";
	#elif defined(YANEURAOU_ENGINE_NNUE_HALFKP_1024X2_8_64)
		"halfkp_1024x2_8_64";
	#elif defined(YANEURAOU_ENGINE_NNUE_SFNNwoP1536)
		"sfnnwop-1536";
	#elif defined(EVAL_NNUE_HALFKP_VM_256X2_32_32)
		"halfkpvm_256x2_32_32";
	#else
		"halfkp_256x2_32_32";
	#endif

#elif defined(YANEURAOU_ENGINE_KPPT)
	"KPPT";
#elif defined(YANEURAOU_ENGINE_KPP_KKPT)
	"KPP_KKPT";
#elif defined(YANEURAOU_ENGINE_MATERIAL)
	"MATERIAL_LV" + to_string(MATERIAL_LEVEL);
#else
	"Unknown";
#endif

	config += o ("EVAL"                , eval_type);

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

static atomic<int64_t> hits[2], means[2];

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

// Used to serialize access to cout
// to avoid multiple threads writing at the same time.
ostream& operator<<(ostream& os, SyncCout sc) {

	static mutex m;

	if (sc == IO_LOCK)
		m.lock();

	if (sc == IO_UNLOCK)
		m.unlock();

	return os;
}

void sync_cout_start() { std::cout << IO_LOCK; }
void sync_cout_end() { std::cout << IO_UNLOCK; }

// --------------------
//  prefetch命令
// --------------------

// prefetch命令を使わない。
#if defined (NO_PREFETCH)

void prefetch(const void*) {}

#else

void prefetch([[maybe_unused]] const void* addr) {

	// SSEの命令なのでSSE2が使える状況でのみ使用する。
#if defined (USE_SSE2)

	// 下位5bitが0でないような中途半端なアドレスのprefetchは、
	// そもそも構造体がalignされていない可能性があり、バグに違いない。
	ASSERT_LV3(((u64)addr & 0x1f) == 0);

	// 1 cache lineのprefetch
	// 64bytesの系もあるかも知れないが、Stockfishではcache line = 32bytesだと仮定してある。
	// ちなみにRyzenでは32bytesらしい。

	#if defined(_MSC_VER)
	_mm_prefetch((char const*)addr, _MM_HINT_T0);
	//	cout << hex << (u64)addr << endl;
	#else
	__builtin_prefetch(addr);
	#endif

#endif
}

#endif

// 📌 ここ以下は、やねうら王の独自追加 📌

// --------------------
//   ElapsedTimer
// --------------------

ElapsedTimer::ElapsedTimer() :
    startTime(now()) {}
ElapsedTimer::ElapsedTimer(TimePoint s) :
    startTime(s) {}

void ElapsedTimer::reset() { reset(now()); }
void ElapsedTimer::reset(TimePoint s) { startTime = s; }

TimePoint ElapsedTimer::elapsed() const { return TimePoint(now() - startTime); }

// --------------------
//  ツール類
// --------------------
namespace Tools {

	// memclear

	// 進捗を表示しながら並列化してゼロクリア
	// ※ Stockfishのtt.cppのTranspositionTable::clear()にあるコードと同等のコード。
	void memclear(ThreadPool& threads, const char* name_, void* table, size_t size)
	{
#if !defined(EVAL_LEARN) && !defined(__EMSCRIPTEN__)

		// Windows10では、このゼロクリアには非常に時間がかかる。
		// malloc()時点ではメモリを実メモリに割り当てられておらず、
		// 初回にアクセスするときにその割当てがなされるため。
		// ゆえに、分割してゼロクリアして、一定時間ごとに進捗を出力する。

		// Options["Threads"]が使用できるスレッド数とは限らない(ふかうら王など)
		const size_t threadCount = threads.num_threads();

		if (name_ != nullptr)
			sync_cout << "info string " + string(name_) + " : Start clearing with " << threadCount << " threads , size =  " << size / (1024 * 1024) << "[MB]" << sync_endl;

		// マルチスレッドで並列化してクリアする。

		for (size_t i = 0; i < threadCount; ++i)
		{
			threads.run_on_thread(i, [table, size, threadCount, i]() {
				// それぞれのスレッドがhash tableの各パートをゼロ初期化する。
				// start  : このスレッドによるゼロクリア開始位置
				// stride : 各スレッドのゼロクリアするサイズ
				// len    : このスレッドによるゼロクリアするサイズ。
				//          strideと等しいが、最後のスレッドだけは端数を考慮し、
				//			size - start のサイズだけクリアする必要がある。
				const size_t stride = size / threadCount,
				start = stride * i,
				len = (i != threadCount - 1) ? stride : size - start;

				memset((uint8_t*)table + start, 0, len);
			});
		}

		for (size_t i = 0; i < threadCount; ++i)
			threads.wait_on_thread(i);

		if (name_ != nullptr)
			sync_cout << "info string " + string(name_) + " : Finish clearing." << sync_endl;

#else
		// yaneuraou.wasm
		// pthread_joinによってブラウザのメインスレッドがブロックされるため、単一スレッドでメモリをクリアする処理に変更

		// LEARN版のときは、
		// 単一スレッドでメモリをクリアする。(他のスレッドは仕事をしているので..)
		// 教師生成を行う時は、対局の最初にスレッドごとのTTに対して、
		// このclear()が呼び出されるものとする。
		// 例) th->tt.clear();
		memset(table, 0, size);
#endif

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
		this_thread::sleep_for(chrono::milliseconds(ms));
	}

	// 現在時刻を文字列化したもを返す。(評価関数の学習時などに用いる)
	string now_string()
	{
		// ctime(), localtime()を使うと、MSVCでセキュアでないという警告が出る。
		// C++標準的にはそんなことないはずなのだが…。

#if defined(_MSC_VER)
		// C4996 : 'ctime' : This function or variable may be unsafe.Consider using ctime_s instead.
#pragma warning(disable : 4996)
#endif

		auto now = chrono::system_clock::now();
		auto tp = chrono::system_clock::to_time_t(now);
		auto result = string(ctime(&tp));

		// 末尾に改行コードが含まれているならこれを除去する
		while (*result.rbegin() == '\n' || (*result.rbegin() == '\r'))
			result.pop_back();
		return result;
	}

	// Linux環境ではgetline()したときにテキストファイルが'\r\n'だと
	// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
	// そのため、ifstreamに対してgetline()を呼び出すときは、
	// getline()ではなくこのこの関数を使うべき。
	bool getline(ifstream& fs, string& s)
	{
		bool b = (bool)std::getline(fs, s);
		StringExtension::trim_inplace(s);
		return b;
	}

	// マルチバイト文字列をワイド文字列に変換する。
	// WindowsAPIを呼び出しているのでWindows環境専用。
	wstring MultiByteToWideChar(const string& s)
	{
#if !defined(_WIN32)
		return wstring(s.begin(), s.end()); // NotImplemented
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
			return wstring(); // 何故かエラーなのだ…。

		return wstring(buffer);
#endif
	}

	// size_ : 全件でいくらあるかを設定する。
	ProgressBar::ProgressBar(u64 size_)
	{
		reset(size_);
	}

	void ProgressBar::reset(u64 size_)
	{
		size = size_;
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
		size_t d = (size == 0) ? all_dots : min((size_t)(all_dots * current / size), all_dots);

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
		case ResultCode::FileNotFound         : return "FileNotFound";
		case ResultCode::FileOpenError        : return "FileOpenError";
		case ResultCode::FileReadError        : return "FileReadError";
		case ResultCode::FileWriteError       : return "FileWriteError";
		case ResultCode::FileCloseError       : return "FileCloseError";
		case ResultCode::FileMismatch         : return "FileMissMatch";
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
	Tools::Result ReadAllLines(const string& filename, vector<string>& lines, bool trim)
	{
#if 0
		ifstream fs(filename);
		if (fs.fail())
			return 1; // 読み込み失敗

		while (!fs.fail() && !fs.eof())
		{
			string line;
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

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		auto result = reader.Open(path);
		if (!result.is_ok())
			return result;

		string line;
		while (reader.ReadLine(line).is_ok())
			lines.emplace_back(line);

		return Tools::Result::Ok();
	}

	// ファイルにすべての行を書き出す。
	Tools::Result WriteAllLines(const string& filename, vector<string>& lines)
	{
		TextWriter writer;

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		if (writer.Open(path).is_not_ok())
			return Tools::ResultCode::FileOpenError;

		for(auto& line : lines)
		{
			if (writer.WriteLine(line).is_not_ok())
			return Tools::ResultCode::FileWriteError;
		}

		return Tools::ResultCode::Ok;
	}

	Tools::Result ReadFileToMemory(const string& filename, function<void* (size_t)> callback_func)
	{
		// fstream、遅いので、FILEを用いて書き換える。

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		FILE* fp = fopen(path.c_str(), "rb");
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


	Tools::Result WriteMemoryToFile(const string& filename, void* ptr, size_t size)
	{
		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		fstream fs(path, ios::out | ios::binary);
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
	Tools::Result TextReader::Open(const string& filename)
	{
		Close();

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		// 高速化のためにbinary open
		fp = fopen(path.c_str(), "rb");
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
	Tools::Result TextReader::ReadLine(string& line)
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

			line = string((const char*)line_buffer.data() + skip_byte, line_size );
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

	Tools::Result TextWriter::Open(const string& filename)
	{
		Close();

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		fp = fopen(path.c_str(), "wb");
		return fp == nullptr ? Tools::ResultCode::FileOpenError
                             : Tools::ResultCode::Ok;
	}

	// 文字列を書き出す(改行コードは書き出さない)
	Tools::Result TextWriter::Write(const string& str)
	{
		return Write(str.c_str(), str.size());
	}

	// 1行を書き出す(改行コードも書き出す) 改行コードは"\r\n"とする。
	Tools::Result TextWriter::WriteLine(const string& line)
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
			memcpy(&buf[write_cursor], ptr2, write_size);
			if (fwrite(buf.data(), buf_size, 1, fp) == 0)
				return Tools::ResultCode::FileWriteError;

			// buf[0..write_cursor-1]が窓で、ループごとにその窓がbuf_sizeずつずれていくと考える。
			// 例えば、ループ2回目ならbuf[write_cursor..write_cursor*2-1]が窓だと考える。

			ptr2             += write_size;
			size             -= write_size;
			write_cursor_end -= buf_size;
			write_cursor      = 0;
		}
		memcpy(&buf[write_cursor], ptr2, size);
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
	Tools::Result BinaryReader::Open(const string& filename)
	{
		auto close_result = Close();
		if (!close_result.is_ok()) {
			return close_result;
		}

		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		fp = fopen(path.c_str(), "rb");
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
	Tools::Result BinaryWriter::Open(const string& filename, bool append)
	{
		// 起動フォルダ相対でのpath
		std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

		fp = fopen(path.c_str(), append ? "ab" : "wb");
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

// Reads the file as bytes.
// Returns nullopt if the file does not exist.

// ファイルをバイトとして読み込みます。
// ファイルが存在しない場合は nullopt を返します。

optional<string> read_file_to_string(const string& filename) {

	// 起動フォルダ相対でのpath
	std::string path = Path::Combine(Directory::GetBinaryFolder(), filename);

	ifstream f(path, ios_base::binary);
	if (!f)
		return nullopt;
	return string(istreambuf_iterator<char>(f), istreambuf_iterator<char>());
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
	string Combine(const string& folder, const string& filename)
	{
		// 与えられたfileが絶対Pathであるかの判定
		if (IsAbsolute(filename))
			return filename;

		if (folder.length() >= 1 && *folder.rbegin() != '/' && *folder.rbegin() != '\\')
			return folder + "/" + filename;

		return folder + filename;
	}

	// full path表現(ファイル名を含む)から、(フォルダ名を除いた)ファイル名の部分を取得する。
	string GetFileName(const string& path)
	{
		// "\"か"/"か、どちらを使ってあるかはわからないがなるべく後ろにある、いずれかの文字を探す。
		auto path_index1 = path.find_last_of("\\");
		auto path_index2 = path.find_last_of("/");

		// どちらの文字も見つからなかったのであれば、ディレクトリ名が含まれておらず、
		// 与えられたpath丸ごとがファイル名だと考えられる。
		if (path_index1 == string::npos && path_index2 == string::npos)
			return path;

		// なるべく後ろのを見つけたいが、string::nposは大きな定数なので単純にmax()するとこれを持ってきてしまう。
		// string::nposを0とみなしてmaxをとる。
		path_index1 = path_index1 == string::npos ? 0 : path_index1;
		path_index2 = path_index2 == string::npos ? 0 : path_index2;
		auto path_index = max(path_index1, path_index2);

		// そこ以降を返す。
		return path.substr(path_index + 1);
	}

	// full path表現から、ディレクトリ名の部分を取得する。
	string GetDirectoryName(const string& path)
	{
		// ファイル名部分を引き算してやる。

		auto length = path.length() - GetFileName(path).length() - 1;
		return (length == 0) ? "" : path.substr(0,length);
	}

	// 絶対Pathであるかの判定。
	// "\\"(WindowsのUNC)で始まるか、"/"で始まるか(Windows / Linuxのroot)、"~"で始まるか、"C:"(ドライブレター + ":")で始まるか。
	bool IsAbsolute(const string& path)
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

	// ファイルが存在するかの確認
	bool Exists(const std::string& path)
	{
		std::ifstream ifs(path);

		// openに成功したら存在する。
        return ifs.is_open();
	}

};

// --------------------
//    Directory
// --------------------

// ディレクトリに存在するファイルの列挙用
// C#のDirectoryクラスっぽい何か
namespace Directory
{
	namespace fs = std::filesystem;

	// 指定されたフォルダに存在するファイルをすべて列挙する。
	// 列挙するときに拡張子を指定できる。(例 : ".bin")
	// 拡張子として""を指定すればすべて列挙される。
	vector<string> EnumerateFiles(const string& sourceDirectory, const string& extension)
	{
		vector<string> filenames;

		// filesystemのファイル列挙、ディレクトリとして空の文字列を渡すと例外で落ちる。
		// current directoryにしたい時は明示的に指定してやらなければならない。
		auto src = sourceDirectory.empty() ? fs::current_path() : fs::path(sourceDirectory);

		for (auto ent : fs::directory_iterator(src))
			if (fs::is_regular_file(ent)
				&& StringExtension::EndsWith(ent.path().filename().string(), extension))

				filenames.push_back(Path::Combine(ent.path().parent_path().string(), ent.path().filename().string()));

		return filenames;
	}

	// フォルダを作成する。日本語は使っていないものとする。
	// 💡 working directory相対で指定する。
	Tools::Result CreateFolder(const std::string& dir_name) {
		std::error_code ec;
		bool created = fs::create_directory(dir_name, ec);
		return created
			? Tools::Result::Ok()
			: Tools::Result(Tools::ResultCode::CreateFolderError);
	}

	// 起動時のフォルダを返す。
	string GetBinaryFolder() { return CommandLine::get_binary_directory(); }
}


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
	string LineScanner::peek_text()
	{
		// 二重にpeek_text()を呼び出すのは合法であるものとする。
		if (!token.empty())
			return token;

		// assert(token.empty());

		// 解析開始位置から連続するスペースは読み飛ばす。
		while (!raw_eol())
		{
			char c = line[pos];
			if (c != ' ')
				break;
			pos++;
		}

		while (!raw_eol())
		{
			// スペースに遭遇するまで。
			char c = line[pos++];
			if (c == ' ')
				break;
			token += c;
		}

		// 次の文字先頭まで解析位置を進めておく。
		while (!raw_eol())
		{
			char c = line[pos];
			if (c != ' ')
				break;
			pos++;
		}

		return token;
	}

	// 次のtokenを返す。
	string LineScanner::get_text()
	{
		auto result = (!token.empty() ? token : peek_text());
		token.clear();
		return result;
	}

	// 現在のcursor位置から残りの文字列を取得する。
	// peek_text()した分があるなら、それも先頭にくっつけて返す。
	string LineScanner::get_rest()
	{
		return token.empty()
			? line.substr(pos)
			: token + " " + line.substr(pos);
	}

	// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
	s64 LineScanner::get_number(s64 defaultValue)
	{
		string token = get_text();
		return token.empty() ? defaultValue : atoll(token.c_str());
	}

	// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
	double LineScanner::get_double(double defaultValue)
	{
		string token = get_text();
		return token.empty() ? defaultValue : atof(token.c_str());
	}

}

// --------------------
//       Math
// --------------------

double Math::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
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
		string destination;
		destination.resize(source.size());
		transform(source.cbegin(), source.cend(), destination.begin(), /*toupper*/[](char c) { return (char)toupper(c); });
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
	string trim(const string& input)
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
	void trim_inplace(string& s)
	{
		auto cur = s.length();

		while (cur > 0 && is_space(s[cur-1]))
			cur--;

		s.resize(cur);
	}

	// 行の末尾の数字を除去した文字列を返す。
	// (行の末尾の"\r","\n",スペース、"\t"を除去したあと)
	string trim_number(const string& input)
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
	void trim_number_inplace(string& s)
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
	int to_int(const string input, int default_)
	{
		// stoi()は例外を出すので例外を使わないようにしてビルドしたいのでNG。
		// atoi()は、セキュリティ的な脆弱性がある。
		// 仕方ないのでistringstreamを使う。

		istringstream ss(input);
		int result = default_; // 失敗したときはこの値のままになる
		ss >> result;
		return result;
	}

	// 文字列をfloat化する。float化に失敗した場合はdefault_の値を返す。
	float to_float(const string input, float default_)
	{
		istringstream ss(input);
		float result = default_; // 失敗したときはこの値のままになる
		ss >> result;
		return result;
	}

	// スペース、タブなど空白に相当する文字で分割して返す。
	vector<string> split(const string& input)
	{
		auto result = vector<string>();
		Parser::LineScanner scanner(input);
		while (!scanner.eol())
			result.push_back(scanner.get_text());

		return result;
	}

	// 先頭にゼロサプライした文字列を返す。
	// 例) n = 123 , digit = 6 なら "000123"という文字列が返る。
	string to_string_with_zero(u64 n, int digit)
	{
		// 現在の状態
		ios::fmtflags curret_flag = cout.flags();

		ostringstream ss;
		ss << setw(digit) << setfill('0') << n;
		string s(ss.str());

		// 状態を戻す
		cout.flags(curret_flag);
		return s;
	}

	// 文字列valueが、文字列startingで始まっていればtrueを返す。
	bool StartsWith(string const& value, string const& starting)
	{
		if (starting.size() > value.size()) return false;
		return equal(starting.begin(), starting.end(), value.begin());
	};

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	bool EndsWith(string const& value, string const& ending)
	{
		if (ending.size() > value.size()) return false;
		return equal(ending.rbegin(), ending.rend(), value.rbegin());
	};

	// 文字列sのなかに文字列tが含まれるかを判定する。含まれていればtrueを返す。
	bool Contains(const string& s, const string& t) {
	   return s.find(t) != string::npos;
	   // C++20ならstring::contains()が使えるのだが…。
	}

	// 文字列valueに対して文字xを文字yに置換した新しい文字列を返す。
	string Replace(string const& value, char x, char y)
	{
		string r(value);
		for (size_t i = 0; i < r.size(); ++i)
			if (r[i] == x)
				r[i] = y;
		return r;
	}

	// 文字列を大文字にして返す。
	string ToUpper(string const& value)
	{
		string s(value);
		transform(s.begin(), s.end(), s.begin(),
			[](unsigned char c){ return toupper(c); });
		return s;
	}

	// sを文字列spで分割した文字列集合を返す。
	// ※　返し値はstring_view(参照を持っている)の配列なので、引数として一時オブジェクトを渡さないように注意してください。
	//    一時オブジェクトへの参照を含むstring_viewをこの関数が返してしまうことになる。
	vector<string_view> Split(string_view s, string_view delimiter) {
		vector<string_view> res;

		if (s.empty())
			return res;

		size_t begin = 0;
		for (;;)
		{
			const size_t end = s.find(delimiter, begin);
			if (end == string::npos)
				break;

			res.emplace_back(s.substr(begin, end - begin));
			begin = end + delimiter.size();
		}

		res.emplace_back(s.substr(begin));

		return res;
	}

	// Pythonの delemiter.join(v) みたいなの。
	// 例: v = [1,2,3] に対して ' '.join(v) == "1 2 3"
	string Join(const vector<string>& v , const string& delimiter)
	{
		string result;
		for (size_t i = 0; i < v.size(); ++i) {
			result += v[i];
			if (i < v.size() - 1) {
				result += delimiter;
			}
		}
		return result;
	}

};

// sを文字列spで分割した文字列集合を返す。
// ※ Stockfishとの互換性のために用意。
vector<string_view> split(string_view s, string_view delimiter)
{
	return StringExtension::Split(s, delimiter);
}

// スペース相当文字列を削除する。⇨ NUMAの処理に必要
void remove_whitespace(string& s) {
	s.erase(remove_if(s.begin(), s.end(), [](char c) { return isspace(c); }), s.end());
}

// スペース相当文字列かどうかを判定する。⇨ NUMAの処理に必要
bool is_whitespace(string_view s) {
	return all_of(s.begin(), s.end(), [](char c) { return isspace(c); });
}

// "123"みたいな文字列を123のように数値型(size_t)に変換する。
size_t str_to_size_t(const string& s) {
	unsigned long long value = stoull(s);
	if (value > numeric_limits<size_t>::max())
		exit(EXIT_FAILURE);
	return static_cast<size_t>(value);
}

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

string CommandLine::get_binary_directory(std::string argv0) {
	std::string pathSeparator;

#ifdef _WIN32
	pathSeparator = "\\";
#ifdef _MSC_VER
	// Under windows argv[0] may not have the extension. Also _get_pgmptr() had
	// issues in some Windows 10 versions, so check returned values carefully.
	char* pgmptr = nullptr;
	if (!_get_pgmptr(&pgmptr) && pgmptr != nullptr && *pgmptr)
		argv0 = pgmptr;
#endif
#else
	pathSeparator = "/";
#endif

	// Extract the working directory
	auto workingDirectory = CommandLine::get_working_directory();

	// Extract the binary directory path from argv0
	auto   binaryDirectory = argv0;
	size_t pos = binaryDirectory.find_last_of("\\/");
	if (pos == string::npos)
		binaryDirectory = "." + pathSeparator;
	else
		binaryDirectory.resize(pos + 1);

	// Pattern replacement: "./" at the start of path is replaced by the working directory
	if (binaryDirectory.find("." + pathSeparator) == 0)
		binaryDirectory.replace(0, 1, workingDirectory);

	return binaryDirectory;
}

// 起動時のworking directory
string CommandLine::get_working_directory() {
	string workingDirectory = "";
	char        buff[40000];
	char* cwd = GETCWD(buff, 40000);
	if (cwd)
		workingDirectory = cwd;

	return workingDirectory;
}

// CommandLine global object
CommandLine CommandLine::g;

// --------------------
// StandardInputWrapper
// --------------------

// 標準入力から1行もらう。Ctrl+Zが来れば"quit"が来たものとする。
// また先行入力でqueueに積んでおくことができる。(次のinput()で取り出される)
string StandardInput::input()
{
	string cmd;
	if (cmds.size() == 0)
	{
		if (!std::getline(cin, cmd)) // 入力が来るかEOFがくるまでここで待機する。
			cmd = "quit";
	} else {
		// 積んであるコマンドがあるならそれを実行する。
		// 尽きれば"quit"だと解釈してdoループを抜ける仕様にすることはできるが、
		// そうしてしまうとgoコマンド(これはノンブロッキングなので)の最中にquitが送られてしまう。
		// ただ、
		// YaneuraOu-mid.exe bench,quit
		// のようなことは出来るのでPGOの役には立ちそうである。
		cmd = cmds.front();
		cmds.pop();
	}
	return cmd;
}

// 先行入力としてqueueに積む。(次のinput()で取り出される)
void StandardInput::push(const string& s)
{
	cmds.push(s);
}

void StandardInput::parse_args(const CommandLine& cli)
{
	// ファイルからコマンドの指定
	if (cli.argc >= 3 && string(cli.argv[1]) == "file")
	{
		vector<string> cmds0;
		SystemIO::ReadAllLines(cli.argv[2], cmds0);

		// queueに変換する。
		for (auto c : cmds0)
			push(c);

	} else {

		string cmd;

		// 引数として指定されたものを一つのコマンドとして実行する機能
		// ただし、','が使われていれば、そこでコマンドが区切れているものとして解釈する。

		for (int i = 1; i < cli.argc; ++i)
		{
			string s = cli.argv[i];

			// sから前後のスペースを除去しないといけない。
			while (*s.rbegin() == ' ') s.pop_back();
			while (*s.begin() == ' ') s = s.substr(1, s.size() - 1);

			if (s != ",")
				cmd += s + " ";
			else
			{
				push(cmd);
				cmd = "";
			}
		}
		if (cmd.size() != 0)
			cmds.push(cmd);
	}
}

// --------------------
//     UnitTest
// --------------------

namespace Misc {
	// このheaderに書いてある関数のUnitTest。
	void UnitTest(Test::UnitTester& tester, IEngine& engine)
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
} // namespace Misc

} // namespace YaneuraOu
