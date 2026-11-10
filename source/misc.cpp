#include "misc.h"
#include "extra/tools.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>

#if defined(_WIN32)
    // windows.hのmin/maxマクロとC++の関数が衝突するのを防ぐ。
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#if STOCKFISH
    #include <direct.h>
    // Stockfish、以前は_getcwd()を使っていた。
    // 現在は使っていないので、このincludeは取り除くべき。
#endif
    #include <windows.h>
    #include <shellapi.h>
#endif

#include "types.h"


namespace YaneuraOu {

namespace fs = std::filesystem;

namespace {

// --------------------
//  logger
// --------------------

#if STOCKFISH
// Version number or dev.
// バージョン番号、または開発版を表すdev。
constexpr std::string_view version = "dev";
#endif

// Our fancy logging facility. The trick here is to replace cin.rdbuf() and
// cout.rdbuf() with two Tie objects that tie cin and cout to a file stream. We
// can toggle the logging of std::cout and std::cin at runtime whilst preserving
// usual I/O functionality, all without changing a single line of code!
// Idea from http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

// ログ記録の仕組み。cin.rdbuf()とcout.rdbuf()を2つのTieオブジェクトに置き換え、
// 標準入力と標準出力をファイルストリームにも記録する。
// 通常の入出力機能を維持したまま、std::coutとstd::cinのログ記録を
// 実行中に切り替えられる。呼び出し側のコードを変更する必要はない。
// 着想元: http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

struct Tie: public std::streambuf {  // MSVC requires split streambuf for cin and cout
    // MSVCではcin用とcout用に別々のstreambufを用意する。

    Tie(std::streambuf* b, std::streambuf* l) :
        buf(b),
        logBuf(l) {}

    int sync() override { return logBuf->pubsync(), buf->pubsync(); }
    int overflow(int c) override { return log(buf->sputc(char(c)), "<< "); }
    int underflow() override { return buf->sgetc(); }
    int uflow() override { return log(buf->sbumpc(), ">> "); }

    std::streambuf *buf, *logBuf;

    int log(int c, const char* prefix) {

        static int last = '\n';  // Single log file
                                  // 入出力で1つのログファイルを共有する。

        if (last == '\n')
            logBuf->sputn(prefix, 3);

        return last = logBuf->sputc(char(c));
    }
};

class Logger {

    Logger() :
        in(std::cin.rdbuf(), file.rdbuf()),
        out(std::cout.rdbuf(), file.rdbuf()) {}
    ~Logger() { start(""); }

    std::ofstream file;    // ログを書き出すファイル
    Tie           in, out; // 標準入力とファイル、標準出力とファイルのひも付け

   public:
    // ログ記録の開始。
    // fname : ログを書き出すファイル名
    static void start(const fs::path& fname) {

        static Logger l;

        if (l.file.is_open())
        {
            std::cout.rdbuf(l.out.buf);
            std::cin.rdbuf(l.in.buf);
            l.file.close();
        }

        if (!fname.empty())
        {
            l.file.open(fname, std::ifstream::out);

            if (!l.file.is_open())
            {
                std::cerr << "Unable to open debug log file " << fname << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cin.rdbuf(&l.in);
            std::cout.rdbuf(&l.out);
        }
    }
};

} // namespace


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

// 現在のStockfishの完全なバージョン名を返す。
//
// ローカルでの開発用ビルドでは、gitのコミットSHAとコミット日を付加する。
// 取得できない場合は、代わりにコンパイル日と「nogit」を使う。
//      Stockfish dev-YYYYMMDD-SHA
//      または
//      Stockfish dev-YYYYMMDD-nogit
//
// リリース用（開発版ではない）ビルドでは、日付やSHAを付けずバージョン番号を使う。
//      Stockfish version
//
// やねうら王では、ソフト名とENGINE_VERSIONを使う。
// ENGINE_VERSIONに小文字の「dev」が含まれる場合に日付とSHAを付加する。
// GIT_DATEやGIT_SHAがビルド時に指定されていなければ、コンパイル日や「nogit」を使う。

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
    if (std::string_view(ENGINE_VERSION).find("dev") != std::string_view::npos)
#endif
	{
		ss << "-";
#ifdef GIT_DATE
		ss << stringify(GIT_DATE);
#else
		constexpr std::string_view months("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec");

		std::string       month, day, year;
		std::stringstream date(__DATE__);  // From compiler, format is "Sep 21 2008"
                                         // コンパイラーが提供する日付。書式は「Sep 21 2008」。

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

#if STOCKFISH
std::string engine_info(bool to_uci) {
    return engine_version_info() + (to_uci ? "\nid author " : " by ")
         + "the Stockfish developers (see AUTHORS file)";
}
#else
std::string engine_info(const std::string& engine_name,
                        const std::string& engine_author,
						const std::string& engine_version,
                        const std::string& eval_name)
{
    // → これ好きじゃない。

	std::stringstream ss;
	std::string engine_name_, engine_author_;

	// カレントフォルダに"engine_name.txt"があればその1行目をエンジン名とする機能
	std::ifstream ifs("engine_name.txt");
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
			stringify(ENGINE_NAME_FROM_MAKEFILE)
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
			;
			engine_author_ = engine_author;
                // やねうら王 "yaneurao";
                // ふかうら王 "Tadao Yamaoka , yaneurao";
	}

	return engine_name_ + "\n" + "id author " + engine_author_; 
}

#endif

// Returns a string trying to describe the compiler we use
// 使用したコンパイラについての文字列を返す。
std::string compiler_info() {

#define make_version_string(major, minor, patch) \
	stringify(major) "." stringify(minor) "." stringify(patch)

	// Predefined macros hell:
	//
    // __GNUC__                Compiler is GCC, Clang or ICX
	// __clang__               Compiler is Clang or ICX
	// __INTEL_LLVM_COMPILER   Compiler is ICX
    // _MSC_VER                Compiler is MSVC
    // _WIN32                  Building on Windows (any)
    // _WIN64                  Building on Windows 64 bit
	// 定義済みマクロによる複雑な判定:
	//
	// __GNUC__                コンパイラーがGCC、Clang、またはICX。
	// __clang__               コンパイラーがClang、またはICX。
	// __INTEL_LLVM_COMPILER   コンパイラーがICX。
	// _MSC_VER                コンパイラーがMSVC。
	// _WIN32                  Windows向けのビルド（32bit、64bitの両方）。
	// _WIN64                  64bit Windows向けのビルド。

	std::string compiler = "\nCompiled by                : ";

#if defined(__INTEL_LLVM_COMPILER)
	compiler += "ICX ";
	compiler += stringify(__INTEL_LLVM_COMPILER);
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
	compiler += std::to_string(__LCC__ / 100);
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
#if defined(USE_AVX512ICL)
    compiler += " AVX512ICL";
#endif
#if defined(USE_VNNI)
	compiler += " VNNI";
#endif
#if defined(USE_AVX512)
	compiler += " AVX512";
#endif

#if defined(USE_BMI2)
    compiler += " BMI2";
#endif

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

#if defined(USE_LASX)
    compiler += " LASX";
#endif
#if defined(USE_LSX)
    compiler += " LSX";
#endif

#if STOCKFISH
	compiler += (HasPopCnt ? " POPCNT" : "");
	// やねうら王にはHasPopCnt定数がないため、Stockfishとの比較用コードとしてのみ残す。
#endif

#if !defined(NDEBUG)
	compiler += " DEBUG";
#endif

	compiler += "\nCompiler __VERSION__ macro : ";
#ifdef __VERSION__
	// 定義されている場合だけ、コンパイラーが提供するバージョン文字列を出力する。
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

// Debug functions used mainly to collect run-time statistics
// 主に実行時の統計を集めるためのデバッグ関数。
constexpr int MaxDebugSlots = 32;

namespace {

template<usize N>
struct DebugInfo {
    std::array<std::atomic<i64>, N> data = {0};

    [[nodiscard]] constexpr std::atomic<i64>& operator[](usize index) {
        assert(index < N);
        return data[index];
    }

    constexpr DebugInfo& operator=(const DebugInfo& other) {
        for (usize i = 0; i < N; i++)
            data[i].store(other.data[i].load());
        return *this;
    }
};

struct DebugExtremes: public DebugInfo<3> {
    DebugExtremes() {
        data[1] = std::numeric_limits<i64>::min();
        data[2] = std::numeric_limits<i64>::max();
    }
};

std::array<DebugInfo<2>, MaxDebugSlots>  hit;
std::array<DebugInfo<2>, MaxDebugSlots>  mean;
std::array<DebugInfo<3>, MaxDebugSlots>  stdev;
std::array<DebugInfo<6>, MaxDebugSlots>  correl;
std::array<DebugExtremes, MaxDebugSlots> extremes;

}  // namespace

bool dbg_hit_on(bool cond, int slot) {

    ++hit.at(slot)[0];
    if (cond)
        ++hit.at(slot)[1];

    return cond;
}

i64 dbg_mean_of(i64 value, int slot) {

    ++mean.at(slot)[0];
    mean.at(slot)[1] += value;

    return value;
}

i64 dbg_stdev_of(i64 value, int slot) {

    ++stdev.at(slot)[0];
    stdev.at(slot)[1] += value;
    stdev.at(slot)[2] += value * value;

    return value;
}

i64 dbg_extremes_of(i64 value, int slot) {
    ++extremes.at(slot)[0];

    i64 current_max = extremes.at(slot)[1].load();
    while (current_max < value && !extremes.at(slot)[1].compare_exchange_weak(current_max, value))
    {}

    i64 current_min = extremes.at(slot)[2].load();
    while (current_min > value && !extremes.at(slot)[2].compare_exchange_weak(current_min, value))
    {}

    return value;
}

void dbg_correl_of(i64 value1, i64 value2, int slot) {

    ++correl.at(slot)[0];
    correl.at(slot)[1] += value1;
    correl.at(slot)[2] += value1 * value1;
    correl.at(slot)[3] += value2;
    correl.at(slot)[4] += value2 * value2;
    correl.at(slot)[5] += value1 * value2;
}

void dbg_print() {

    i64  n;
    auto E   = [&n](i64 x) { return double(x) / n; };
    auto sqr = [](double x) { return x * x; };

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = hit[i][0]))
            std::cerr << "Hit #" << i << ": Total " << n << " Hits " << hit[i][1]
                      << " Hit Rate (%) " << 100.0 * E(hit[i][1]) << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = mean[i][0]))
            std::cerr << "Mean #" << i << ": Total " << n << " Mean " << E(mean[i][1]) << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = stdev[i][0]))
        {
            double r = sqrt(E(stdev[i][2]) - sqr(E(stdev[i][1])));
            std::cerr << "Stdev #" << i << ": Total " << n << " Stdev " << r << std::endl;
        }

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = extremes[i][0]))
            std::cerr << "Extremity #" << i << ": Total " << n << " Min " << extremes[i][2]
                      << " Max " << extremes[i][1] << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = correl[i][0]))
        {
            double r = (E(correl[i][5]) - E(correl[i][1]) * E(correl[i][3]))
                     / (sqrt(E(correl[i][2]) - sqr(E(correl[i][1])))
                        * sqrt(E(correl[i][4]) - sqr(E(correl[i][3]))));
            std::cerr << "Correl. #" << i << ": Total " << n << " Coefficient " << r << std::endl;
        }
}

void dbg_clear() {
    hit.fill({});
    mean.fill({});
    stdev.fill({});
    correl.fill({});
    extremes.fill({});
}

// --------------------
//  sync_cout/sync_endl
// --------------------

// Used to serialize access to std::cout
// to avoid multiple threads writing at the same time.
// 複数のスレッドが同時に書き込まないよう、std::coutへのアクセスを直列化する。
std::ostream& operator<<(std::ostream& os, SyncCout sc) {

	static std::mutex m;

	if (sc == IO_LOCK)
		m.lock();

	if (sc == IO_UNLOCK)
		m.unlock();

	return os;
}

void sync_cout_start() { std::cout << IO_LOCK; }
void sync_cout_end() { std::cout << IO_UNLOCK; }

// Hash function based on public domain MurmurHash64A, by Austin Appleby.
// Austin Applebyがパブリックドメインとして公開したMurmurHash64Aに基づくハッシュ関数。
u64 hash_bytes(const char* data, usize size) {
    const u64 m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    u64 h = size * m;

    const char* end = data + (size & ~(usize) 7);

    for (const char* p = data; p != end; p += 8)
    {
        u64 k;
        std::memcpy(&k, p, sizeof(k));

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    if (size & 7)
    {
        u64 k = 0;
        for (int i = (size & 7) - 1; i >= 0; i--)
            k = (k << 8) | u8(end[i]);

        h ^= k;
        h *= m;
    }

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

// Trampoline helper to avoid moving Logger to misc.h
// Loggerをmisc.hへ移さずに利用できるようにする転送関数。
void start_logger(const fs::path& fname) { Logger::start(fname); }

// Convert a wide string to an UTF8 string
// ワイド文字列をUTF-8文字列へ変換する。
// Windows以外では、各要素をcharへコピーするだけで文字コード変換は行わない。
std::string utf8_from_wstring(std::wstring_view s) {
#ifdef _WIN32
    if (s.empty())
        return {};

    int size =
      WideCharToMultiByte(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0)
        return {};

    std::string out(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.data(), int(s.size()), out.data(), size, nullptr, nullptr);
    return out;
#else
    return std::string(s.begin(), s.end());
#endif
}


// This utility allows us to handle filenames and paths with UTF8 encoding
// UTF-8で指定されたファイル名やパスを扱うための変換関数。
fs::path path_from_utf8(const std::string& path) {
#ifdef _WIN32
    int u8len = static_cast<int>(path.size());

    // First attempt UTF-8, then fall back to ANSI for old GUIs like Arena
    // まずUTF-8として変換し、失敗したらArenaなどの古いGUI向けにANSIで再試行する。
    constexpr int CodePages[2] = {CP_UTF8, CP_ACP};
    for (int cp : CodePages)
    {
        int flags = cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0;
        int wlen  = MultiByteToWideChar(cp, flags, path.c_str(), u8len, NULL, 0);
        if (wlen > 0)
        {
            std::wstring wstr(static_cast<usize>(wlen), L'\0');
            MultiByteToWideChar(cp, 0, path.c_str(), u8len, wstr.data(), wlen);
            return {wstr};
        }
    }

    return {path};
#else
    return {path};
#endif
}


// Constructor for the CommandLine class
// CommandLineクラスのコンストラクタ。
// 起動引数を設定する。グローバルインスタンスのset_arg()と同じ経路を使う。
CommandLine::CommandLine(int _argc, char** _argv) { set_arg(_argc, _argv); }

void CommandLine::set_arg(int _argc, char** _argv) {
    argc = _argc;
    argv = _argv;
#ifdef _WIN32
    // Convert any non-ANSI characters passed on the command line to UTF-8
    // コマンドライン引数に含まれるANSIでは表現できない文字をUTF-8へ変換する。
    int wargc = 0;
    if (LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc))
    {
        // set_arg()が再度呼ばれた場合も、引数を前回分へ追加せず置き換える。
        argv_utf8.clear();
        argv_storage.clear();

        for (int i = 0; i < wargc; ++i)
            argv_storage.push_back(utf8_from_wstring(wargv[i]));
        LocalFree(wargv);

        for (std::string& s : argv_storage)
            argv_utf8.push_back(s.data());
        argv_utf8.push_back(nullptr);

        argc = wargc;
        argv = argv_utf8.data();
    }
#endif
}


// Read a numerical value from the command line
// コマンドライン引数の数値を読み取る。
// 「123」のような数値文字列をusizeへ変換する。変換できなければstd::nulloptを返す。
std::optional<usize> str_to_size_t(const std::string& s) {
    if (s.empty() || s[0] == '-')
        return std::nullopt;
    errno                           = 0;
    char*                    endptr = nullptr;
    const unsigned long long value  = std::strtoull(s.c_str(), &endptr, 10);
    if (errno == ERANGE || (*endptr != '\0' && !std::isspace(static_cast<unsigned char>(*endptr)))
        || value > std::numeric_limits<usize>::max())
        return std::nullopt;
    return static_cast<usize>(value);
}

// Read the given file as bytes (returns std::nullopt if the file does not exist)
// 指定されたファイルをバイト列として読み込む。
// ファイルが存在しない場合など、開けなければstd::nulloptを返す。

std::optional<std::string> read_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios_base::binary);
    if (!f)
        return std::nullopt;
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Remove all spaces from a string
// 文字列中のすべての空白文字を削除する。NUMAの処理でも使用する。
void remove_whitespace(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }),
            s.end());
}

// Test if a string has only spaces
// 文字列が空白文字だけで構成されているかを判定する。NUMAの処理でも使用する。
// 空文字列の場合もtrueを返す。
bool is_whitespace(std::string_view s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
}


// Return the directory where our Stockfish binary sits. This is useful,
// because when the NNUE network is not embedded in the binary, this directory
// is one of the locations where we look for the NNUE file.
// 実行ファイルが置かれているディレクトリを返す。
// NNUEネットワークが実行ファイルに埋め込まれていない場合に、
// NNUEファイルを探すディレクトリの一つとして使える。
fs::path CommandLine::get_binary_directory(fs::path argv0) {

#ifdef _WIN32
    // Prefer the executable path reported by Windows. Unlike _get_wpgmptr,
    // this does not depend on whether the CRT used a narrow or wide entry
    // point. Windows paths cannot exceed 32767 characters, so a fixed
    // buffer is always sufficient. Falls back to argv0 if the API fails.
    // Windowsが報告する実行ファイルのパスを優先する。_get_wpgmptrと異なり、
    // CRTのエントリーポイントが狭文字版かワイド文字版かに依存しない。
    // Windowsのパス長は最大32767文字なので、固定長のバッファで十分。
    // APIが失敗した場合はargv0を使う。
    constexpr DWORD MaxPath = 32768;
    wchar_t         path[MaxPath];

    if (const DWORD length = GetModuleFileNameW(nullptr, path, MaxPath))
        argv0 = fs::path(path, path + length);
#endif

    auto binaryDirectory = argv0.parent_path();
    if (binaryDirectory.empty())
        binaryDirectory = fs::path(".");
    return binaryDirectory;
}


#if !STOCKFISH
// プロセス共通の起動引数から、Unicodeを保持した絶対パスを返す。
fs::path CommandLine::get_binary_directory() {
    return fs::absolute(get_binary_directory(
      g.argc > 0 && g.argv && g.argv[0] ? path_from_utf8(g.argv[0]) : fs::path{}));
}
#endif

// CommandLine global object
// 起動引数をプロセス全体で共有するCommandLineのグローバルオブジェクト。
CommandLine CommandLine::g;

// On Windows, tell the console to use UTF8 encoding
// Windowsではコンソールの入出力をUTF-8に設定する。
void set_console_utf8() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
}

} // namespace YaneuraOu
