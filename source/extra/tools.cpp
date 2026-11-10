#include "tools.h"
#include "parser.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "../misc.h"
#include "../thread.h"

namespace YaneuraOu {
#if !STOCKFISH
// config.hで設定した値などについて出力する。
std::string config_info()
{
	std::string config = "\nconfigured by config.h";

	auto o  = [](std::string(p) , std::string(q)) { return "\n" + (p + std::string(20,' ')).substr(0,20) + " : " + q; };
	auto o1 = [&o](const char* p , u64  u ) { return o(std::string(p) , std::to_string(u) ); };
	auto o2 = [&o](const char* p , bool b ) { return o(std::string(p) , b ? "true":"false"); };

	// 評価関数タイプ
	std::string eval_type =
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
	#elif defined(YANEURAOU_ENGINE_SFNN1536)
		"sfnn-1536";
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
	"MATERIAL_LV" + std::to_string(MATERIAL_LEVEL);
#else
	"Unknown";
#endif

	config += o ("EVAL"                , eval_type);

	config += o1("ASSERT_LV"           , ASSERT_LV      );
	config += o1("HASH_KEY_BITS"       , HASH_KEY_BITS  );

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
	config += o2("USE_MATE_DFPN"            , use_mate_dfpn      );

	// コンパイラ情報もついでに出力する。
	//config += "\n\n" + compiler_info();

	// その他、欲しいものがあれば追加するかも。

	return config;
}



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

#endif

namespace Tools {

	// memclear

	// 進捗を表示しながら並列化してゼロクリア
	// ※ Stockfishのtt.cppのTranspositionTable::clear()にあるコードと同等のコード。
	void memclear(ThreadPool& threads, const char* name_, void* table, size_t size)
	{
#if !defined(__EMSCRIPTEN__)

		// Windows10では、このゼロクリアには非常に時間がかかる。
		// malloc()時点ではメモリを実メモリに割り当てられておらず、
		// 初回にアクセスするときにその割当てがなされるため。
		// ゆえに、分割してゼロクリアして、一定時間ごとに進捗を出力する。

		// Options["Threads"]が使用できるスレッド数とは限らない(ふかうら王など)
		const size_t threadCount = threads.num_threads();

		if (name_ != nullptr)
			sync_cout << "info string " + std::string(name_) + " : Start clearing with " << threadCount << " threads , size =  " << size / (1024 * 1024) << "[MB]" << sync_endl;

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

				std::memset((u8*)table + start, 0, len);
			});
		}

		for (size_t i = 0; i < threadCount; ++i)
			threads.wait_on_thread(i);

		if (name_ != nullptr)
			sync_cout << "info string " + std::string(name_) + " : Finish clearing." << sync_endl;

#else
		// yaneuraou.wasm
		// pthread_joinによってブラウザのメインスレッドがブロックされるため、単一スレッドでメモリをクリアする処理に変更

		std::memset(table, 0, size);
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
	void sleep(std::uint64_t ms)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	}

	// std::getline()で1行読み、末尾のCR・LF・スペース・タブを除去する。
	// 行頭の空白は保持する。戻り値は読み取りが成功したかどうか。
	bool getline(std::ifstream& fs, std::string& s)
	{
		bool b = (bool)std::getline(fs, s);
		StringExtension::trim_inplace(s);
		return b;
	}


	// size_ : 全件でいくらあるかを設定する。
	ProgressBar::ProgressBar(std::uint64_t size_)
	{
		reset(size_);
	}

	void ProgressBar::reset(std::uint64_t size_)
	{
		size = size_;
		if (enable_)
			std::cout << "0% [";
		dots = 0;
	}

	// 進捗を出力する。
	// current : 現在までに完了している件数
	void ProgressBar::check(std::uint64_t current)
	{
		if (!enable_)
			return;

		// 何個dotを打つべきか。
		const std::size_t all_dots = 70; // 100%になった時に70個打つ。

		// 何dot塗りつぶすのか。
		std::size_t d = (size == 0) ? all_dots : std::min((std::size_t)(all_dots * current / size), all_dots);

		for (; dots < d ; ++dots)
			std::cout << ".";

		if (dots == all_dots)
		{
			std::cout << "] 100%" << std::endl;
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
//    文字列 拡張
// --------------------

namespace {
	// 文字列を大文字化する
	std::string to_upper(const std::string source)
	{
		std::string destination;
		destination.resize(source.size());
		std::transform(source.cbegin(), source.cend(), destination.begin(), /*toupper*/[](char c) { return (char)std::toupper(c); });
		return destination;
	}
}

namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// string-case insensitive-compareの略？
	// s1==s2のとき0(false)を返す。
	bool stricmp(const std::string& s1, const std::string& s2)
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
		std::string s = input;

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
		std::string s = input;
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

	// 文字列をfloat化する。float化に失敗した場合はdefault_の値を返す。
	float to_float(const std::string input, float default_)
	{
		std::istringstream ss(input);
		float result = default_; // 失敗したときはこの値のままになる
		ss >> result;
		return result;
	}

	// スペース、タブなど空白に相当する文字で分割して返す。
	std::vector<std::string> split(const std::string& input)
	{
		auto result = std::vector<std::string>();
		Parser::LineScanner scanner(input);
		while (!scanner.eol())
			result.push_back(scanner.get_text());

		return result;
	}

}

}
