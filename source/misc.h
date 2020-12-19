#ifndef MISC_H_INCLUDED
#define MISC_H_INCLUDED

#include <chrono>
#include <vector>
#include <functional>
#include <fstream>
#include <mutex>
#include <atomic>

#include "types.h"

// --------------------
//  engine info
// --------------------

// "USI"コマンドに応答するために表示する。
const std::string engine_info();

// 使用したコンパイラについての文字列を返す。
const std::string compiler_info();

// --------------------
//    prefetch命令
// --------------------

// prefetch()は、与えられたアドレスの内容をL1/L2 cacheに事前に読み込む。
// これはnon-blocking関数で、CPUがメモリに読み込むのを待たない。

void prefetch(void* addr);

// --------------------
//  logger
// --------------------

// cin/coutへの入出力をファイルにリダイレクトを開始/終了する。
void start_logger(bool b);

// --------------------
//  Large Page確保
// --------------------

// Large Pageを確保するwrapper class。
// WindowsのLarge Pageを確保する。
// Large Pageを用いるとメモリアクセスが速くなるらしい。
// 置換表用のメモリなどはこれで確保する。
// cf. やねうら王、Large Page対応で10数%速くなった件 : http://yaneuraou.yaneu.com/2020/05/31/%e3%82%84%e3%81%ad%e3%81%86%e3%82%89%e7%8e%8b%e3%80%81large-page%e5%af%be%e5%bf%9c%e3%81%a710%e6%95%b0%e9%80%9f%e3%81%8f%e3%81%aa%e3%81%a3%e3%81%9f%e4%bb%b6/
//
// Stockfishでは、Large Pageの確保～開放のためにaligned_ttmem_alloc(),aligned_ttmem_free()という関数が実装されている。
// コードの簡単化のために、やねうら王では独自に本classからそれらを用いる。
struct LargeMemory
{
	// LargePage上のメモリを確保する。Large Pageに確保できるなら、そこにする。
	// aligned_ttmem_alloc()を内部的に呼び出すので、アドレスは少なくとも2MBでalignされていることは保証されるが、
	// 気になる人のためにalignmentを明示的に指定できるようになっている。
	// メモリ確保に失敗するか、引数のalignで指定したalignmentになっていなければ、
	// エラーメッセージを出力してプログラムを終了させる。
	// size       : 確保するサイズ [byte]
	// align      : 返されるメモリが守るべきalignment
	// zero_clear : trueならゼロクリアされたメモリ領域を返す。
	void* alloc(size_t size, size_t align = 256 , bool zero_clear = false);

	// alloc()で確保したメモリを開放する。
	// このクラスのデストラクタからも自動でこの関数が呼び出されるので明示的に呼び出す必要はない(かも)
	void free();

	// alloc()が呼び出されてメモリが確保されている状態か？
	bool alloced() const { return mem != nullptr; }

	// alloc()のstatic関数版。この関数で確保したメモリはstatic_free()で開放する。
	// 引数のmemには、static_free()に渡すべきポインタが得られる。
	static void* static_alloc(size_t size, void*& mem, size_t align = 256, bool zero_clear = false);

	// static_alloc()で確保したメモリを開放する。
	static void static_free(void* mem);

	~LargeMemory() { free(); }

private:
	// allocで確保されたメモリの先頭アドレス
	// (free()で開放するときにこのアドレスを用いる)
	void* mem = nullptr;
};

// --------------------
//  統計情報
// --------------------

// bがtrueであった回数 / dbg_hit_on()が呼び出された回数 を調べるためのもの。
// (どれくらいの割合でXが成り立つか、みたいなのを調べるときに用いる)
void dbg_hit_on(bool b);

// if (c) dbg_hit_on(b)と等価。
void dbg_hit_on(bool c , bool b);

// vの合計 / 呼びだされた回数 ( = vの平均) みたいなのを求めるときに調べるためのもの。
void dbg_mean_of(int v);

// 探索部から1秒おきにdbg_print()が呼び出されるものとする。
// このとき、以下の関数を呼び出すと、その統計情報をcerrに出力する。
void dbg_print();


// --------------------
//  Time[ms] wrapper
// --------------------

// ms単位での時間計測しか必要ないのでこれをTimePoint型のように扱う。
typedef std::chrono::milliseconds::rep TimePoint;
static_assert(sizeof(TimePoint) == sizeof(int64_t), "TimePoint should be 64 bits");

// ms単位で現在時刻を返す
static TimePoint now() {
	return std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --------------------
//    HashTable
// --------------------

// このclass、Stockfishにあるんだけど、
// EvalHashとしてLargePageを用いる同等のclassをすでに用意しているので、使わない。

//template<class Entry, int Size>
//struct HashTable {
//	Entry* operator[](Key key) { return &table[(uint32_t)key & (Size - 1)]; }
//
//private:
//	std::vector<Entry> table = std::vector<Entry>(Size);
//};

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
//       乱数
// --------------------

// 擬似乱数生成器
// Stockfishで用いられているもの + 現在時刻によるseedの初期化機能。
// UniformRandomNumberGenerator互換にして、std::shuffle()等でも使えるようにするべきか？
struct PRNG
{
	PRNG(u64 seed) : s(seed) { ASSERT_LV1(seed); }

	// 時刻などでseedを初期化する。
	PRNG() {
		// C++11のrandom_device()によるseedの初期化
		// std::random_device rd; s = (u64)rd() + ((u64)rd() << 32);
		// →　msys2のgccでbuildすると同じ値を返すっぽい。なんぞこれ…。

		// time値とか、thisとか色々加算しておく。
		s = (u64)(time(NULL)) + ((u64)(this) << 32)
			+ (u64)(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	}

	// 乱数を一つ取り出す。
	template<typename T> T rand() { return T(rand64()); }

	// 0からn-1までの乱数を返す。(一様分布ではないが現実的にはこれで十分)
	u64 rand(u64 n) { return rand<u64>() % n; }

	// 内部で使用している乱数seedを返す。
	u64 get_seed() const { return s;  }

private:
	u64 s;
	u64 rand64() {
		s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
		return s * 2685821657736338717LL;
	}
};

// 乱数のseedを表示する。(デバッグ用)
static std::ostream& operator<<(std::ostream& os, PRNG& prng)
{
	os << "PRNG::seed = " << std::hex << prng.get_seed() << std::dec;
	return os;
}

// --------------------
//  64bit×64bitの掛け算の上位64bitを取り出す関数
// --------------------

inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
#if defined(__GNUC__) && defined(IS_64BIT)
	__extension__ typedef unsigned __int128 uint128;
	return ((uint128)a * (uint128)b) >> 64;
#else
	// 64bit同士の掛け算を64bitを32bit 2つに分割して、筆算のようなことをするコード
	uint64_t aL = (uint32_t)a, aH = a >> 32;
	uint64_t bL = (uint32_t)b, bH = b >> 32;
	uint64_t c1 = (aL * bL) >> 32;
	uint64_t c2 = aH * bL + c1;
	uint64_t c3 = aL * bH + (uint32_t)c2;
	return aH * bH + (c2 >> 32) + (c3 >> 32);
#endif
}

// --------------------
//  全プロセッサを使う
// --------------------

// Windows環境において、プロセスが1個の論理プロセッサグループを超えてスレッドを
// 実行するのは不可能である。これは、最大64コアまでの使用に制限されていることを普通、意味する。
// これを克服するためには、いくつかの特殊なプラットフォーム固有のAPIを呼び出して、
// それぞのスレッドがgroup affinityを設定しなければならない。
// 元のコードはPeter ÖsterlundによるTexelから。

namespace WinProcGroup {
	// 各スレッドがidle_loop()などで自分のスレッド番号(0～)を渡す。
	// 1つ目のプロセッサをまず使い切るようにgroup affinityを割り当てる。
	// 1つ目のプロセッサの論理コアを使い切ったら次は2つ目のプロセッサを使っていくような動作。
	void bindThisThread(size_t idx);
}

// -----------------------
//  探索のときに使う時間管理用
// -----------------------

namespace Search { struct LimitsType; }

struct Timer
{
	// タイマーを初期化する。以降、elapsed()でinit()してからの経過時間が得られる。
	void reset() { startTime = startTimeFromPonderhit = now(); }

	// "ponderhit"からの時刻を計測する用
	void reset_for_ponderhit() { startTimeFromPonderhit = now(); }

	// 探索開始からの経過時間。単位は[ms]
	// 探索node数に縛りがある場合、elapsed()で探索node数が返ってくる仕様にすることにより、一元管理できる。
	TimePoint elapsed() const;

	// reset_for_ponderhit()からの経過時間。その関数は"ponderhit"したときに呼び出される。
	// reset_for_ponderhit()が呼び出されていないときは、reset()からの経過時間。その関数は"go"コマンドでの探索開始時に呼び出される。
	TimePoint elapsed_from_ponderhit() const;

	// reset()されてからreset_for_ponderhit()までの時間
	TimePoint elapsed_from_start_to_ponderhit() const { return (TimePoint)(startTimeFromPonderhit - startTime); }

#if 0
	// 探索node数を経過時間の代わりに使う。(こうするとタイマーに左右されない思考が出来るので、思考に再現性を持たせることが出来る)
	// node数を指定して探索するとき、探索できる残りnode数。
	// ※　StockfishでここintになっているのはTimePointにするのが正しいと思う。[2020/01/20]
	TimePoint availableNodes;
	// →　NetworkDelayやMinimumThinkingTimeなどの影響を考慮するのが難しく、将棋の場合、
	// 　相性があまりよろしくないのでこの機能はやねうら王ではサポートしないことにする。
#endif

	// このシンボルが定義されていると、今回の思考時間を計算する機能が有効になる。
#if defined(USE_TIME_MANAGEMENT)

  // 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
	void init(const Search::LimitsType& limits, Color us, int ply);

	TimePoint minimum() const { return minimumTime; }
	TimePoint optimum() const { return optimumTime; }
	TimePoint maximum() const { return maximumTime; }

	// 1秒単位で繰り上げてdelayを引く。
	// ただし、remain_timeよりは小さくなるように制限する。
	TimePoint round_up(TimePoint t) const;

	// 探索終了の時間(startTime + search_end >= now()になったら停止)
	std::atomic<TimePoint> search_end;

private:
	TimePoint minimumTime;
	TimePoint optimumTime;
	TimePoint maximumTime;

	// Options["NetworkDelay"]の値
	TimePoint network_delay;
	// Options["MinimalThinkingTime"]の値
	TimePoint minimum_thinking_time;

	// 今回の残り時間 - Options["NetworkDelay2"]
	TimePoint remain_time;

#endif

private:
	// 探索開始時間
	TimePoint startTime;

	TimePoint startTimeFromPonderhit;
};

extern Timer Time;


// =====   以下は、やねうら王の独自追加   =====

// --------------------
//  ツール類
// --------------------

namespace Tools
{
	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishではtt.cppにこのコードがあるのだが、独自の置換表を確保したいときに
	// これが独立していないと困るので、ここに用意する。
	// nameは"Hash" , "eHash"などクリアしたいものの名前を書く。
	// メモリクリアの途中経過が出力されるときにその名前(引数nameで渡している)が出力される。
	// name == nullptrのとき、途中経過は表示しない。
	extern void memclear(const char* name, void* table, size_t size);

	// insertion sort
	// 昇順に並び替える。学習時のコードで使いたい時があるので用意してある。
	template <typename T >
	void insertion_sort(T* arr, int left, int right)
	{
		for (int i = left + 1; i < right; i++)
		{
			auto key = arr[i];
			int j = i - 1;

			// keyより大きな arr[0..i-1]の要素を現在処理中の先頭へ。
			while (j >= left && (arr[j] > key))
			{
				arr[j + 1] = arr[j];
				j = j - 1;
			}
			arr[j + 1] = key;
		}
	}

	// 途中での終了処理のためのwrapper
	// コンソールの出力が完了するのを待ちたいので3秒待ってから::exit(EXIT_FAILURE)する。
	extern void exit();

	// 指定されたミリ秒だけsleepする。
	extern void sleep(int ms);

	// 現在時刻を文字列化したもを返す。(評価関数の学習時などにログ出力のために用いる)
	extern std::string now_string();

	// Linux環境ではgetline()したときにテキストファイルが'\r\n'だと
	// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
	// そのため、ifstreamに対してgetline()を呼び出すときは、
	// std::getline()ではなくこのこの関数を使うべき。
	extern bool getline(std::ifstream& fs, std::string& s);

	// マルチバイト文字列をワイド文字列に変換する。
	// WindowsAPIを呼び出しているのでWindows環境専用。
	extern std::wstring MultiByteToWideChar(const std::string& s);

	// 他言語にあるtry～finally構文みたいなの。
	// SCOPE_EXIT()マクロの実装で使う。このクラスを直接使わないで。
	struct __FINALLY__ {
		__FINALLY__(std::function<void()> fn_) : fn(fn_) {}
		~__FINALLY__() { fn(); }
	private:
		std::function<void()> fn;
	};

	// --------------------
	//  Result
	// --------------------

	// 一般的な関数の返し値のコード。(エラー理由などのenum)
	enum struct ResultCode
	{
		// 正常終了
		Ok,

		// ファイルの終端に達した
		Eof,

		// 原因の詳細不明。何らかのエラー。
		SomeError,

		// メモリ割り当てのエラー
		MemoryAllocationError,

		// ファイルのオープンに失敗。ファイルが存在しないなど。
		FileOpenError,

		// ファイル読み込み時のエラー。
		FileReadError,

		// ファイル書き込み時のエラー。
		FileWriteError,

		// フォルダ作成時のエラー。
		CreateFolderError,

		// 実装されていないエラー。
		NotImplementedError,
	};

	// ResultCodeを文字列化する。
	extern std::string to_string(ResultCode);

	// エラーを含む関数の返し値を表現する型
	// RustにあるOption型のような何か
	struct Result
	{
		Result(ResultCode code_) : code(code_) {}

		// エラーの種類
		ResultCode code;

		// 返し値が正常終了かを判定する
		bool is_ok() const { return code == ResultCode::Ok; }

		// 返し値が正常終了でなければtrueになる。
		bool is_not_ok() const { return code != ResultCode::Ok; }

		// 返し値がEOFかどうかを判定する。
		bool is_eof() const { return code == ResultCode::Eof; }

		// ResultCodeを文字列化して返す。
		std::string to_string() const { return Tools::to_string(code); }

		//  正常終了の時の型を返すbuilder
		static Result Ok() { return Result(ResultCode::Ok); }
	};
}

// スコープを抜ける時に実行してくれる。BOOST::BOOST_SCOPE_EXITマクロみたいな何か。
// 使用例) SCOPE_EXIT( x = 10 );
#define SCOPE_EXIT(STATEMENT) Tools::__FINALLY__ __clean_up_object__([&]{ STATEMENT });


// --------------------
//  ファイルの丸読み
// --------------------

struct FileOperator
{
	// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。末尾の改行は除去される。
	// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
	// 引数で渡されるtrimはtrueを渡すと末尾のスペース、タブがトリムされる。
	static Tools::Result ReadAllLines(const std::string& filename, std::vector<std::string>& lines, bool trim = false);


	// msys2、Windows Subsystem for Linuxなどのgcc/clangでコンパイルした場合、
	// C++のstd::ifstreamで::read()は、一発で2GB以上のファイルの読み書きが出来ないのでそのためのwrapperである。
	//
	// read_file_to_memory()の引数のcallback_funcは、ファイルがオープン出来た時点でそのファイルサイズを引数として
	// callbackされるので、バッファを確保して、その先頭ポインタを返す関数を渡すと、そこに読み込んでくれる。
	//
	// また、callbackされた関数のなかでバッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
	// nullptrを返せば良い。このとき、read_file_to_memory()は、読み込みを中断し、エラーリターンする。

	static Tools::Result ReadFileToMemory(const std::string& filename, std::function<void* (u64)> callback_func);
	static Tools::Result WriteMemoryToFile(const std::string& filename, void* ptr, u64 size);
};

// C#のTextReaderみたいなもの。
// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
// fopen()～fread()で実装されている。
struct TextFileReader
{
	TextFileReader();
	~TextFileReader();

	// ファイルをopenする。
	Tools::Result Open(const std::string& filename);

	// Open()を呼び出してオープンしたファイルをクローズする。
	void Close();

	// ファイルの終了判定。
	// ファイルを最後まで読み込んだのなら、trueを返す。

	// 1行読み込む(改行まで) 引数のlineに代入される。
	// 改行コードは返さない。
	// SkipEmptyLine(),SetTrim()の設定を反映する。
	// Eofに達した場合は、返し値としてTools::ResultCode::Eofを返す。
	Tools::Result ReadLine(std::string& line);

	// ReadLine()で空行を読み飛ばすかどうかの設定。
	// (ここで行った設定はOpen()/Close()ではクリアされない。)
	// デフォルトでfalse
	void SkipEmptyLine(bool skip = true) { skipEmptyLine = skip;  }

	// ReadLine()でtrimするかの設定。
	// 引数のtrimがtrueの時は、ReadLine()のときに末尾のスペース、タブはトリムする
	// (ここで行った設定はOpen()/Close()ではクリアされない。)
	// デフォルトでfalse
	void SetTrim(bool trim = true) { this->trim = trim; }

private:
	// 各種状態変数の初期化
	void clear();

	// 次のblockのbufferへの読み込み。
	void read_next_block();

	// オープンしているファイル。
	// オープンしていなければnullptrが入っている。
	FILE* fp;

	// バッファから1文字読み込む。eofに達したら、-1を返す。
	int read_char();

	// ReadLineの下請け。何も考えずに1行読み込む。行のtrim、空行のskipなどなし。
	// line_bufferに読み込まれた行が代入される。
	Tools::Result read_line_simple();

	// ファイルの読み込みバッファ 1MB
	std::vector<u8> buffer;

	// 行バッファ
	std::vector<u8> line_buffer;

	// バッファに今回読み込まれたサイズ
	size_t read_size;

	// bufferの解析位置
	// 次のReadLine()でここから解析して1行返す
	// 次の文字 c = buffer[cursor]
	size_t cursor;

	// eofフラグ。
	// fp.eof()は、bufferにまだ未処理のデータが残っているかも知れないのでそちらを信じるわけにはいかない。
	bool is_eof;

	// 直前が\r(CR)だったのか？のフラグ
	bool is_prev_cr;

	// ReadLine()で行の末尾をtrimするかのフラグ。
	bool trim;

	// ReadLine()で空行をskipするかのフラグ
	bool skipEmptyLine;
};


// --------------------
//    PRNGのasync版
// --------------------

// PRNGのasync版
struct AsyncPRNG
{
	// [ASYNC] 乱数を一つ取り出す。
	template<typename T> T rand() {
		std::unique_lock<std::mutex> lk(mutex);
		return prng.rand<T>();
	}

	// [ASYNC] 0からn-1までの乱数を返す。(一様分布ではないが現実的にはこれで十分)
	u64 rand(u64 n) {
		std::unique_lock<std::mutex> lk(mutex);
		return prng.rand(n);
	}

	// 内部で使用している乱数seedを返す。
	u64 get_seed() const { return prng.get_seed(); }

protected:
	std::mutex mutex;
	PRNG prng;
};

// 乱数のseedを表示する。(デバッグ用)
static std::ostream& operator<<(std::ostream& os, AsyncPRNG& prng)
{
	os << "AsyncPRNG::seed = " << std::hex << prng.get_seed() << std::dec;
	return os;
}

// --------------------
//       Parser
// --------------------

// スペースで区切られた文字列を解析するためのparser
struct LineScanner
{
	// 解析したい文字列を渡す(スペースをセパレータとする)
	LineScanner(std::string line_) : line(line_), pos(0) {}

	// 次のtokenを先読みして返す。get_token()するまで解析位置は進まない。
	std::string peek_text();

	// 次のtokenを返す。
	std::string get_text();

	// 次の文字列を数値化して返す。
	// 空の文字列である場合は引数の値がそのまま返る。
	// "ABC"のような文字列で数値化できない場合は0が返る。(あまり良くない仕様だがatoll()を使うので仕方ない)
	s64 get_number(s64 defaultValue);
	double get_double(double defaultValue);

	// 解析位置(カーソル)が行の末尾まで進んだのか？
	// eolとはEnd Of Lineの意味。
	// get_text()をしてpeek_text()したときに保持していたものがなくなるまではこの関数はfalseを返し続ける。
	// このクラスの内部からeol()を呼ばないほうが無難。(token.empty() == trueが保証されていないといけないので)
	// 内部から呼び出すならraw_eol()のほうではないかと。
	bool eol() const { return token.empty() && raw_eol(); }

private:
	// 解析位置(カーソル)が行の末尾まで進んだのか？(内部実装用)
	bool raw_eol() const { return !(pos < line.length()); }

	// 解析対象の行
	std::string line;

	// 解析カーソル(現在の解析位置)
	unsigned int pos;

	// peek_text()した文字列。get_text()のときにこれを返す。
	std::string token;
};

// --------------------
//       Math
// --------------------

// 進行度の計算や学習で用いる数学的な関数
namespace Math {
	// シグモイド関数
	//  = 1.0 / (1.0 + std::exp(-x))
	double sigmoid(double x);

	// シグモイド関数の微分
	//  = sigmoid(x) * (1.0 - sigmoid(x))
	double dsigmoid(double x);

	// vを[lo,hi]の間に収まるようにクリップする。
	// ※　Stockfishではこの関数、bitboard.hに書いてある。
	template<class T> constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
		return v < lo ? lo : v > hi ? hi : v;
	}

	// cの倍数になるようにvを繰り上げる
	template<class T> constexpr const T align(const T v, const int c)
	{
		// cは2の倍数である。(1である一番上のbit以外は0
		ASSERT_LV3((c & (c - 1)) == 0);
		return (v + (T)(c - 1)) & ~(T)(c - 1);
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
	// folder名のほうは空文字列でないときに、末尾に'/'か'\\'がなければそれを付与する。
	extern std::string Combine(const std::string& folder, const std::string& filename);

	// full path表現から、(フォルダ名をすべて除いた)ファイル名の部分を取得する。
	extern std::string GetFileName(const std::string& path);

	// full path表現から、(ファイル名だけを除いた)ディレクトリ名の部分を取得する。
	extern std::string GetDirectoryName(const std::string& path);
};

// --------------------
//    文字列 拡張
// --------------------

namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// Windowsだと_stricmp() , Linuxだとstrcasecmp()を使うのだが、
	// 後者がどうも動作が怪しい。自前実装しておいたほうが無難。
	// stricmpは、string case insensitive compareの略？
	// s1==s2のとき0(false)を返す。
	extern bool stricmp(const std::string& s1, const std::string& s2);

	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	// ios::binaryでopenした場合などには'\r'なども入っていることがあるので…。
	extern std::string trim(const std::string& input);

	// trim()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	extern void trim_inplace(std::string& input);

	// 行の末尾の数字を除去した文字列を返す。
	// sfenの末尾の手数を削除する用
	// 末尾のスペースを詰めたあと数字を詰めてそのあと再度スペースを詰める処理になっている。
	// 例 : "abc 123 "→"abc"となって欲しいので。
	extern std::string trim_number(const std::string& input);

	// trim_number()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	extern void trim_number_inplace(std::string& s);

	// 文字列をint化する。int化に失敗した場合はdefault_の値を返す。
	extern int to_int(const std::string input, int default_);

	// スペース、タブなど空白に相当する文字で分割して返す。
	extern std::vector<std::string> split(const std::string& input);

	// --- 以下、C#のstringクラスにあるやつ。

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	extern bool StartsWith(std::string const& value, std::string const& starting);

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	extern bool EndsWith(std::string const& value, std::string const& ending);

};

// --------------------
//  FileSystem
// --------------------

// ディレクトリに存在するファイルの列挙用
// C#のDirectoryクラスっぽい何か
namespace Directory
{
	// 指定されたフォルダに存在するファイルをすべて列挙する。
	// 列挙するときに引数extensionで列挙したいファイル名の拡張子を指定できる。(例 : ".bin")
	// 拡張子として""を指定すればすべて列挙される。
	extern std::vector<std::string> EnumerateFiles(const std::string& sourceDirectory, const std::string& extension);

	// フォルダを作成する。
	// working directory相対で指定する。dir_nameに日本語は使っていないものとする。
	// ※　Windows環境だと、この関数名、WinAPIのCreateDirectoryというマクロがあって…。
	// 　ゆえに、CreateDirectory()をやめて、CreateFolder()に変更する。
	extern Tools::Result CreateFolder(const std::string& dir_name);

	// working directoryを返す。
	// "GetCurrentDirectory"という名前はWindowsAPI(で定義されているマクロ)と競合する。
	extern std::string GetCurrentFolder();
}

namespace CommandLine {
	// 起動時にmain.cppから呼び出される。
	// CommandLine::binaryDirectory , CommandLine::workingDirectoryを設定する。
	extern void init(int argc, char* argv[]);

	extern std::string binaryDirectory;  // path of the executable directory
	extern std::string workingDirectory; // path of the working directory
}


#endif // #ifndef MISC_H_INCLUDED
