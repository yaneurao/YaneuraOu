
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
		Dependency::getline(ifs, str);
		ss << "id name " << str << endl;

		// 2行目が読み込めなかったときのためにデフォルト値を設定しておく。
		str = "default author";
		Dependency::getline(ifs, str);
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
			<< ENGINE_VERSION << setfill('0')
			<< (Is64Bit ? " 64" : " 32")
			<< TARGET_CPU
#if defined(FOR_TOURNAMENT)
			<< " TOURNAMENT"
#endif
			<< endl
			<< "id author by yaneurao" << endl;
	}

	return ss.str();
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

	static Mutex m;

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
#ifdef NO_PREFETCH

void prefetch(void*) {}

#else

void prefetch(void* addr) {

	// SSEの命令なのでSSE2が使える状況でのみ使用する。
#ifdef USE_SSE2

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

void prefetch2(void* addr)
{
	// Stockfishのコードはこうなっている。
	// cache lineが32byteなら、あと2回やる必要があるように思うのだが…。

	prefetch(addr);
	prefetch((uint8_t*)addr + 64);
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

		int threads = 0;
		int nodes = 0;
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
			if (ptr->Relationship == RelationNumaNode)
				nodes++;

			else if (ptr->Relationship == RelationProcessorCore)
			{
				cores++;
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
		for (int t = 0; t < threads - cores; t++)
			groups.push_back(t % nodes);

		// If we still have more threads than the total number of logical processors
		// then return -1 and let the OS to decide what to do.
		return idx < groups.size() ? groups[idx] : -1;
	}

	// たぬきさんのXeon Phi用のコード。
	// Dual Xeonでもう一つのコアが100%使えないようなのであとで修正する。
#if 0
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
#endif

	/// bindThisThread() set the group affinity of the current thread

	void bindThisThread(size_t idx) {

		// Use only local variables to be thread-safe
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
//    memclear
// --------------------

// 進捗を表示しながら並列化してゼロクリア
// ※ Stockfishのtt.cppのTranspositionTable::clear()にあるコードと同等のコード。
void memclear(const char* name_ , void* table, size_t size)
{
	// Windows10では、このゼロクリアには非常に時間がかかる。
	// malloc()時点ではメモリを実メモリに割り当てられておらず、
	// 初回にアクセスするときにその割当てがなされるため。
	// ゆえに、分割してゼロクリアして、一定時間ごとに進捗を出力する。

	// memset(table, 0, size);

	std::string name(name_);
	sync_cout << "info string " + name + " Clear begin , Hash size =  " << size / (1024 * 1024) << "[MB]" << sync_endl;

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

	sync_cout << "info string " + name + " Clear done." << sync_endl;

}

// --- 以下、やねうら王で独自追加したコード

// --------------------
//  ファイルの丸読み
// --------------------

// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。
int read_all_lines(std::string filename, std::vector<std::string>& lines)
{
	ifstream fs(filename);
	if (fs.fail())
		return 1; // 読み込み失敗

	while (!fs.fail() && !fs.eof())
	{
		std::string line;
		Dependency::getline(fs, line);
		if (line.length())
			lines.push_back(line);
	}
	fs.close();
	return 0;
}

int read_file_to_memory(std::string filename, std::function<void*(u64)> callback_func)
{
	fstream fs(filename, ios::in | ios::binary);
	if (fs.fail())
		return 1;

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
	if (ptr == nullptr)
		return 2;

	// 細切れに読み込む

	const u64 block_size = 1024 * 1024 * 1024; // 1回のreadで読み込む要素の数(1GB)
	for (u64 pos = 0; pos < file_size; pos += block_size)
	{
		// 今回読み込むサイズ
		u64 read_size = (pos + block_size < file_size) ? block_size : (file_size - pos);
		fs.read((char*)ptr + pos, read_size);

		// ファイルの途中で読み込みエラーに至った。
		if (fs.fail())
			return 2;

		//cout << ".";
	}
	fs.close();

	return 0;
}


int write_memory_to_file(std::string filename, void *ptr, u64 size)
{
	fstream fs(filename, ios::out | ios::binary);
	if (fs.fail())
		return 1;

	const u64 block_size = 1024 * 1024 * 1024; // 1回のwriteで書き出す要素の数(1GB)
	for (u64 pos = 0; pos < size; pos += block_size)
	{
		// 今回書き出すメモリサイズ
		u64 write_size = (pos + block_size < size) ? block_size : (size - pos);
		fs.write((char*)ptr + pos, write_size);
		//cout << ".";
	}
	fs.close();
	return 0;
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

	while (!raw_eof())
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

// --------------------
//  Timer
// --------------------

Timer Time;

TimePoint Timer::elapsed() const { return TimePoint(Search::Limits.npmsec ? Threads.nodes_searched() : now() - startTime); }
TimePoint Timer::elapsed_from_ponderhit() const { return TimePoint(Search::Limits.npmsec ? Threads.nodes_searched()/*これ正しくないがこのモードでponder使わないからいいや*/ : now() - startTimeFromPonderhit); }

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

void sleep(int ms)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

uint64_t get_thread_id()
{
	auto id = std::this_thread::get_id();
	if (sizeof(id) >= 8)
		return *(uint64_t*)(&id);
	else if (sizeof(id) >= 4)
		return *(uint32_t*)(&id);
	else
		return 0; // give up
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
	
	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	std::string trim(const std::string& input)
	{
		string s = input; // copy
		s64 cur = (s64)s.length() - 1; // 符号つきの型でないとマイナスになったかの判定ができない
		while (cur >= 0)
		{
			char c = s[cur];
			// 改行文字、スペース、タブではないならループを抜ける。
			// これらの文字が出現しなくなるまで末尾を切り詰める。
			if (!(c == '\r' || c == '\n' || c == ' ' || c == '\t'))
				break;
			cur--;
		}
		cur++;
		s.resize((size_t)cur);
		return s;
	}

	// 行の末尾の"\r","\n",スペース、"\t"、数字を除去した文字列を返す。
	std::string trim_number(const std::string& input)
	{
		string s = input; // copy
		s64 cur = (s64)s.length() - 1;
		while (cur >= 0)
		{
			char c = s[cur];
			// 改行文字、スペース、タブではないならループを抜ける。
			// これらの文字が出現しなくなるまで末尾を切り詰める。
			if (!(c == '\r' || c == '\n' || c == ' ' || c == '\t' || ('0' <= c && c <= '9')))
				break;
			cur--;
		}
		cur++;
		s.resize((size_t)cur);
		return s;
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
		while (!scanner.eof())
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
				&& StringExtension::EndsWith(ent.path().filename().string() , extension))

				filenames.push_back(Path::Combine(ent.path().parent_path().string(),ent.path().filename().string()));

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

}

// ----------------------------
//     mkdir wrapper
// ----------------------------

// カレントフォルダ相対で指定する。成功すれば0、失敗すれば非0が返る。
// フォルダを作成する。日本語は使っていないものとする。
// どうもmsys2環境下のgccだと_wmkdir()だとフォルダの作成に失敗する。原因不明。
// 仕方ないので_mkdir()を用いる。
// ※　C++17のfilesystemがどの環境でも問題なく動くようになれば、
//     std::filesystem::create_directories()を用いて書き直すべき。

#if defined(_WIN32)
// Windows用

#if defined(_MSC_VER)
#include <codecvt>	// mkdirするのにwstringが欲しいのでこれが必要
#include <locale>   // wstring_convertにこれが必要。

namespace Directory {
	int CreateFolder(const std::string& dir_name)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
		return _wmkdir(cv.from_bytes(dir_name).c_str());
		//	::CreateDirectory(cv.from_bytes(dir_name).c_str(),NULL);
	}
}

#elif defined(__GNUC__) 

#include <direct.h>
namespace Directory {
	int CreateFolder(const std::string& dir_name)
	{
	return _mkdir(dir_name.c_str());
	}
}

#endif
#elif defined(_LINUX)

// linux環境において、この_LINUXというシンボルはmakefileにて定義されるものとする。

// Linux用のmkdir実装。
#include "sys/stat.h"

namespace Directory {
	int CreateFolder(const std::string& dir_name)
	{
	return ::mkdir(dir_name.c_str(), 0777);
	}
}
#else

// Linux環境かどうかを判定するためにはmakefileを分けないといけなくなってくるな..
// linuxでフォルダ掘る機能は、とりあえずナシでいいや..。評価関数ファイルの保存にしか使ってないし…。

namespace Directory {
	int CreateFolder(const std::string& dir_name)
	{
		return 0;
	}
}

#endif

// --------------------
//  Dependency Wrapper
// --------------------

namespace Dependency
{
	bool getline(std::ifstream& fs, std::string& s)
	{
		bool b = (bool)std::getline(fs, s);
		s = StringExtension::trim(s);
		return b;
	}
}



