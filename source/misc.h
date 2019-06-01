#ifndef MISC_H_INCLUDED
#define MISC_H_INCLUDED

#include <chrono>
#include <vector>
#include <functional>
#include <fstream>

#include "types.h"
#include "thread_win32.h" // AsyncPRNGで使う

// --------------------
//  engine info
// --------------------

// "USI"コマンドに応答するために表示する。
const std::string engine_info();

// --------------------
//    prefetch命令
// --------------------

// prefetch()は、与えられたアドレスの内容をL1/L2 cacheに事前に読み込む。
// これはnon-blocking関数で、CPUがメモリに読み込むのを待たない。

void prefetch(void* addr);

// 連続する128バイトをprefetchするときに用いる。
void prefetch2(void* addr);

// --------------------
//  logger
// --------------------

// cin/coutへの入出力をファイルにリダイレクトを開始/終了する。
void start_logger(bool b);

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
inline TimePoint now() {
	return std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --------------------
//    HashTable
// --------------------

// 将棋では使わないので要らないや..

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
inline std::ostream& operator<<(std::ostream& os, PRNG& prng)
{
	os << "PRNG::seed = " << std::hex << prng.get_seed() << std::dec;
	return os;
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

// --------------------
//   以下は、やねうら王の独自追加
// --------------------

// 指定されたミリ秒だけsleepする。
extern void sleep(int ms);

// 現在時刻を文字列化したもを返す。(評価関数の学習時などにログ出力のために用いる)
std::string now_string();


// 途中での終了処理のためのwrapper
static void my_exit()
{
	sleep(3000); // エラーメッセージが出力される前に終了するのはまずいのでwaitを入れておく。
	exit(EXIT_FAILURE);
}

// 進捗を表示しながら並列化してゼロクリア
// Stockfishではtt.cppにこのコードがあるのだが、独自の置換表を確保したいときに
// これが独立していないと困るので、ここに用意する。
void memclear(void* table, size_t size);

// insertion sort
// 昇順に並び替える。学習時のコードを使いたい時があるので用意。
template <typename T >
void my_insertion_sort(T* arr, int left, int right)
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

// 乱数のseedなどとしてthread idを使いたいが、
// C++のthread idは文字列しか取り出せないので無理やりcastしてしまう。
extern uint64_t get_thread_id();

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

	// 探索node数を経過時間の代わりに使う。(こうするとタイマーに左右されない思考が出来るので、思考に再現性を持たせることが出来る)
	// node数を指定して探索するとき、探索できる残りnode数。
	int64_t availableNodes;

	// このシンボルが定義されていると、今回の思考時間を計算する機能が有効になる。
#ifdef  USE_TIME_MANAGEMENT

  // 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
	void init(Search::LimitsType& limits, Color us, int ply);

	TimePoint minimum() const { return minimumTime; }
	TimePoint optimum() const { return optimumTime; }
	TimePoint maximum() const { return maximumTime; }

	// 1秒単位で繰り上げてdelayを引く。
	// ただし、remain_timeよりは小さくなるように制限する。
	TimePoint round_up(TimePoint t) const {
		// 1000で繰り上げる。Options["MinimalThinkingTime"]が最低値。
		t = std::max(((t + 999) / 1000) * 1000, minimum_thinking_time);
		// そこから、Options["NetworkDelay"]の値を引くが、remain_timeを上回ってはならない。
		t = std::min(t - network_delay, remain_time);
		return t;
	}

	// 探索終了の時間(startTime + search_end >= now()になったら停止)
	TimePoint search_end;

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

// --------------------
//  ファイルの丸読み
// --------------------

// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。
int read_all_lines(std::string filename, std::vector<std::string>& lines);

// msys2、Windows Subsystem for Linuxなどのgcc/clangでコンパイルした場合、
// C++のstd::ifstreamで::read()は、一発で2GB以上のファイルの読み書きが出来ないのでそのためのwrapperである。
//
// read_file_to_memory()の引数のcallback_funcは、ファイルがオープン出来た時点でそのファイルサイズを引数として
// callbackされるので、バッファを確保して、その先頭ポインタを返す関数を渡すと、そこに読み込んでくれる。
// これらの関数は、ファイルが見つからないときなどエラーの際には非0を返す。
//
// また、callbackされた関数のなかでバッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
// nullptrを返せば良い。このとき、read_file_to_memory()は、読み込みを中断し、エラーリターンする。

int read_file_to_memory(std::string filename, std::function<void*(u64)> callback_func);
int write_memory_to_file(std::string filename, void *ptr, u64 size);


// --------------------
//    PRNGのasync版
// --------------------

// PRNGのasync版
struct AsyncPRNG
{
	// [ASYNC] 乱数を一つ取り出す。
	template<typename T> T rand() {
		std::unique_lock<Mutex> lk(mutex);
		return prng.rand<T>();
	}

	// [ASYNC] 0からn-1までの乱数を返す。(一様分布ではないが現実的にはこれで十分)
	u64 rand(u64 n) {
		std::unique_lock<Mutex> lk(mutex);
		return prng.rand(n);
	}

	// 内部で使用している乱数seedを返す。
	u64 get_seed() const { return prng.get_seed(); }

protected:
	Mutex mutex;
	PRNG prng;
};

// 乱数のseedを表示する。(デバッグ用)
inline std::ostream& operator<<(std::ostream& os, AsyncPRNG& prng)
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

	// 解析位置(カーソル)が行の末尾まで進んだのか？
	// get_text()をしてpeek_text()したときに保持していたものがなくなるまではこの関数はfalseを返し続ける。
	// このクラスの内部からeof()を呼ばないほうが無難。(token.empty() == trueが保証されていないといけないので)
	// 内部から呼び出すならraw_eof()のほうではないかと。
	bool eof() const { return token.empty() && raw_eof(); }

private:
	// 解析位置(カーソル)が行の末尾まで進んだのか？(内部実装用)
	bool raw_eof() const { return !(pos < line.length()); }

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

}

// --------------------
//       Path
// --------------------

// C#にあるPathクラス的なもの。ファイル名の操作。
// C#のメソッド名に合わせておく。
struct Path
{
	// path名とファイル名を結合して、それを返す。
	// folder名のほうは空文字列でないときに、末尾に'/'か'\\'がなければそれを付与する。
	static std::string Combine(const std::string& folder, const std::string& filename)
	{
		if (folder.length() >= 1 && *folder.rbegin() != '/' && *folder.rbegin() != '\\')
			return folder + "/" + filename;

		return folder + filename;
	}

	// full path表現から、(フォルダ名を除いた)ファイル名の部分を取得する。
	static std::string GetFileName(const std::string& path)
	{
		// "\"か"/"か、どちらを使ってあるかはわからない。
		auto path_index1 = path.find_last_of("\\") + 1;
		auto path_index2 = path.find_last_of("/") + 1;
		auto path_index = std::max(path_index1, path_index2);

		return path.substr(path_index);
	}
};

// --------------------
//    文字列 拡張
// --------------------

namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// string case insensitive compareの略？
	// s1==s2のとき0(false)を返す。
	extern bool stricmp(const std::string& s1, const std::string& s2);

	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	// ios::binaryでopenした場合などには'\r'なども入っていることがあるので…。
	extern std::string trim(const std::string& input);

	// 行の末尾の"\r","\n",スペース、"\t"、数字を除去した文字列を返す。
	// sfenの末尾の手数を削除する用
	extern std::string trim_number(const std::string& input);

	// 文字列のstart番目以降を返す
	static std::string mid(const std::string& input, size_t start) {
		return input.substr(start, input.length() - start);
	}

	// 文字列をint化する。int化に失敗した場合はdefault_の値を返す。
	extern int to_int(const std::string input, int default_);
};

// --------------------
//  Tools
// --------------------

namespace Tools
{
	// 他言語にあるtry～finally構文みたいなの。
	struct Finally {
		Finally(std::function<void()> fn_) : fn(fn_){}
		~Finally() { fn(); }
	private:
		std::function<void()> fn;
	};

}

// --------------------
//  Dependency Wrapper
// --------------------

namespace Dependency
{
	// Linux環境ではgetline()したときにテキストファイルが'\r\n'だと
	// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
	// そのため、fstreamに対してgetline()を呼び出すときは、
	// std::getline()ではなく単にgetline()と書いて、この関数を使うべき。
	extern bool getline(std::ifstream& fs, std::string& s);

	// フォルダを作成する。
	// カレントフォルダ相対で指定する。dir_nameに日本語は使っていないものとする。
	// 成功すれば0、失敗すれば非0が返る。
	extern int mkdir(std::string dir_name);
}



#endif // #ifndef MISC_H_INCLUDED
