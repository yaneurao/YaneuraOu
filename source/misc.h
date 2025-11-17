#ifndef MISC_H_INCLUDED
#define MISC_H_INCLUDED

//#include <chrono>
#include <exception>  // IWYU pragma: keep
// IWYU pragma: no_include <__exception/terminate.h>
#include <functional>
#include <optional>
#include <cstring>
#include <memory>
//#include <string>
#include <string_view>
//#include <vector>

#include <fstream>
#include <mutex>
#include <atomic>
#include <sstream>
#include <queue>
#include <unordered_set>
#include <condition_variable>

#include "types.h"

namespace YaneuraOu {

class OptionsMap;
class Engine;

// --------------------
//     engine info
// --------------------

// エンジンのバージョン情報を返す。
std::string engine_version_info();

// "USI"コマンドに応答するために表示する。
//
//  to_usi : これがtrueのときは、"usi"コマンドに対する応答として呼び出されたという意味。
//           これがfalseのときは、起動直後の出力用。
//        	 ⚠ やねうら王ではMultiEngineを採用しており、
//			 起動直後ではエンジン名が確定しないから出力できない。
// 
// 🤔 やねうら王では、以下のように変更する。
// engine_name    : エンジン名
// engine_author  : エンジンの作者名
// engine_version : エンジンのバージョン
// eval_name      : 評価関数名
std::string engine_info(const std::string& engine_name,
						const std::string& engine_author,
                        const std::string& engine_version,
                        const std::string& eval_name);

// 使用したコンパイラについての文字列を返す。
std::string compiler_info();

// config.hで設定した値などについて出力する。
std::string config_info();


// --------------------
//    prefetch命令
// --------------------

// prefetch()は、与えられたアドレスの内容をL1/L2 cacheに事前に読み込む。
// これはnon-blocking関数で、CPUがメモリに読み込むのを待たない。

void prefetch(const void* addr);

// --------------------
//  logger
// --------------------

// cin/coutへの入出力をファイルにリダイレクトを開始/終了する。
void start_logger(const std::string& fname);

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

#if STOCKFISH
// ms単位での時間計測しか必要ないのでこれをTimePoint型のように扱う。
// TimePointの定義。💡 やねうら王では、types.hに移動。
//using TimePoint = std::chrono::milliseconds::rep;  // A value in milliseconds
//static_assert(sizeof(TimePoint) == sizeof(int64_t), "TimePoint should be 64 bits");
#endif


// ms単位で現在時刻を返す
static TimePoint now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    //(std::chrono::steady_clock::now().time_since_epoch()).count() * 10;
    // 💡 10倍早く時間が経過するようにして、持ち時間制御のテストなどを行う時は↑このように10をかけ算する。
}

#if STOCKFISH

#else
// 🌈 時間計測用。経過時間を計測する。
struct ElapsedTimer {
    ElapsedTimer();

    // startTimeを引数sで初期化する。
    ElapsedTimer(TimePoint s);

    // タイマーを初期化する。以降、elapsed()でinit()してからの経過時間が得られる。
    void reset();
    // TimePointを指定して初期化する。この時刻からの経過時間が求められるようになる。
    void reset(TimePoint s);

    // resetしてからの経過時間。
    TimePoint elapsed() const;

   private:
    // reset()された時刻。
    TimePoint startTime;
};

#endif

// --------------------
//  sync_out/sync_endl
// --------------------

// Used to serialize access to std::cout
// to avoid multiple threads writing at the same time.

// スレッド排他しながらcoutに出力するために使う。
// 例)
// sync_out << "bestmove " << m << sync_endl;
// のように用いる。

enum SyncCout {
	IO_LOCK,
	IO_UNLOCK
};
std::ostream& operator<<(std::ostream&, SyncCout);

#define sync_cout std::cout << IO_LOCK
#define sync_endl std::endl << IO_UNLOCK

// sync_cout / sync_endlと同等のlock～unlock。
void sync_cout_start();
void sync_cout_end();

// --------------------
//      ValueList
// --------------------

//  最大サイズが固定長のvectorみたいなやつ。
template<typename T, std::size_t MaxSize>
class ValueList {

public:
	std::size_t size() const { return size_; }
    void        push_back(const T& value) {
        assert(size_ < MaxSize);
        values_[size_++] = value;
    }
	const T* begin() const { return values_; }
	const T* end() const { return values_ + size_; }

	const T& operator[](int index) const { return values_[index]; }
	// ⇨ ここの引数、どうせ大きな配列は確保しないのでsize_tではなくintで良い。

	// 非const版の begin/end(やねうら王独自追加)
	T* begin() { return values_; }
	T* end()   { return values_ + size_; }

private:
	T           values_[MaxSize];
	std::size_t size_ = 0;
};

// --------------------
//      MultiArray
// --------------------

template<typename T, std::size_t Size, std::size_t... Sizes>
class MultiArray;

namespace Detail {

	template<typename T, std::size_t Size, std::size_t... Sizes>
	struct MultiArrayHelper {
		using ChildType = MultiArray<T, Sizes...>;
	};

	template<typename T, std::size_t Size>
	struct MultiArrayHelper<T, Size> {
		using ChildType = T;
	};

	template<typename To, typename From>
	constexpr bool is_strictly_assignable_v =
		std::is_assignable_v<To&, From> && (std::is_same_v<To, From> || !std::is_convertible_v<From, To>);

}

// MultiArray is a generic N-dimensional array.
// The template parameters (Size and Sizes) encode the dimensions of the array.

// MultiArray は汎用的な N 次元配列です。
// テンプレートパラメータ (Size と Sizes) が配列の次元を表します。

template<typename T, std::size_t Size, std::size_t... Sizes>
class MultiArray {
	using ChildType = typename Detail::MultiArrayHelper<T, Size, Sizes...>::ChildType;
	using ArrayType = std::array<ChildType, Size>;
	ArrayType data_;

public:
	using value_type = typename ArrayType::value_type;
	using size_type = typename ArrayType::size_type;
	using difference_type = typename ArrayType::difference_type;
	using reference = typename ArrayType::reference;
	using const_reference = typename ArrayType::const_reference;
	using pointer = typename ArrayType::pointer;
	using const_pointer = typename ArrayType::const_pointer;
	using iterator = typename ArrayType::iterator;
	using const_iterator = typename ArrayType::const_iterator;
	using reverse_iterator = typename ArrayType::reverse_iterator;
	using const_reverse_iterator = typename ArrayType::const_reverse_iterator;

	constexpr auto& at(size_type index) noexcept { return data_.at(index); }
	constexpr const auto& at(size_type index) const noexcept { return data_.at(index); }

	constexpr auto& operator[](size_type index) noexcept { return data_[index]; }
	constexpr const auto& operator[](size_type index) const noexcept { return data_[index]; }

	constexpr auto& front() noexcept { return data_.front(); }
	constexpr const auto& front() const noexcept { return data_.front(); }
	constexpr auto& back() noexcept { return data_.back(); }
	constexpr const auto& back() const noexcept { return data_.back(); }

	auto* data() { return data_.data(); }
	const auto* data() const { return data_.data(); }

	constexpr auto begin() noexcept { return data_.begin(); }
	constexpr auto end() noexcept { return data_.end(); }
	constexpr auto begin() const noexcept { return data_.begin(); }
	constexpr auto end() const noexcept { return data_.end(); }
	constexpr auto cbegin() const noexcept { return data_.cbegin(); }
	constexpr auto cend() const noexcept { return data_.cend(); }

	constexpr auto rbegin() noexcept { return data_.rbegin(); }
	constexpr auto rend() noexcept { return data_.rend(); }
	constexpr auto rbegin() const noexcept { return data_.rbegin(); }
	constexpr auto rend() const noexcept { return data_.rend(); }
	constexpr auto crbegin() const noexcept { return data_.crbegin(); }
	constexpr auto crend() const noexcept { return data_.crend(); }

	constexpr bool      empty() const noexcept { return data_.empty(); }
	constexpr size_type size() const noexcept { return data_.size(); }
	constexpr size_type max_size() const noexcept { return data_.max_size(); }

	template<typename U>
	void fill(const U& v) {
		static_assert(Detail::is_strictly_assignable_v<T, U>,
			"Cannot assign fill value to entry type");
		for (auto& ele : data_)
		{
			if constexpr (sizeof...(Sizes) == 0)
				ele = v;
			else
				ele.fill(v);
		}
	}

	constexpr void swap(MultiArray<T, Size, Sizes...>& other) noexcept { data_.swap(other.data_); }
};

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
		//	+ (u64)(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		// ⇨ MSYS2 + clang18でhigh_resolution_clock::now()を使うとセグフォで落ちるようになった。
		//   代わりにsteady_clockを用いる。
			+ (u64)std::chrono::steady_clock::now().time_since_epoch().count();
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
    __extension__ using uint128 = unsigned __int128;
    return (uint128(a) * uint128(b)) >> 64;
#else
	// 64bit同士の掛け算を64bitを32bit 2つに分割して、筆算のようなことをするコード
    uint64_t aL = uint32_t(a), aH = a >> 32;
    uint64_t bL = uint32_t(b), bH = b >> 32;
    uint64_t c1 = (aL * bL) >> 32;
    uint64_t c2 = aH * bL + c1;
    uint64_t c3 = aL * bH + uint32_t(c2);
    return aH * bH + (c2 >> 32) + (c3 >> 32);
#endif
}

// --------------------
//   hash値の計算
// --------------------

// 📓 SFNNのバイナリに対してhash値を計算するためのヘルパー関数群。

// 2つのハッシュ値を安全に合成するためのユーティリティ。
// seed と v を合成した値を seed に返す。
// 📝 vのほうはT型としてstd::hash<T>を利用する。
template<typename T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// hash_combine の std::size_t 特化版
// 📝 std::hash<std::size_t> は単なる値返しであることが多く、
//     特化させることで不要な hasher オブジェクト生成を削減し、
//     わずかだがパフォーマンス改善になる。
template<>
inline void hash_combine(std::size_t& seed, const std::size_t& v) {
    seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// 任意の POD ライクなデータ構造を、バイト列そのままのハッシュとして利用し、size_tで返す。
// 📝 `reinterpret_cast` でメモリ内容を生のまま string_view にして hash を取る。
//     その実装はコンパイラ依存かつendian依存。
template<typename T>
inline std::size_t get_raw_data_hash(const T& value) {
    return std::hash<std::string_view>{}(
      std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
}

/*
	FixedString

	固定長バッファ上で動作する軽量文字列。
	std::string のような動的メモリアロケーションを一切行わず、
	組み込み用途やパフォーマンス重視のコードで有用。
	💡 StringBuilder やログバッファ用途に近い。

    特徴:
     - Capacity をコンパイル時に決める
     - オーバーフロー時は std::terminate() で即死（安全優先）
     - '\0' 終端を保持しており C 文字列互換
     - std::string / std::string_view への暗黙変換あり
*/

// Capacity : 最大文字列長(byte)
template<std::size_t Capacity>
class FixedString {
   public:

	// 空のFixedStringを構築する。
    FixedString() :
        length_(0) {
        data_[0] = '\0';
    }

	// char* から FixedStringを構築する。
	// ⚠ Capacityを超えた場合は、即座にstd::terminate()を呼び出す。
    FixedString(const char* str) {
        size_t len = std::strlen(str);
        if (len > Capacity)
            std::terminate();
        std::memcpy(data_, str, len);
        length_        = len;
        data_[length_] = '\0';
    }

	// std::string から FixedStringを構築する。
	// ⚠ Capacityを超えた場合は、即座にstd::terminate()を呼び出す。
    FixedString(const std::string& str) {
        if (str.size() > Capacity)
            std::terminate();
        std::memcpy(data_, str.data(), str.size());
        length_        = str.size();
        data_[length_] = '\0';
    }

	// 格納している文字列長さ
    std::size_t size() const { return length_; }

	// template引数で渡されたCapacity
    std::size_t capacity() const { return Capacity; }

	// string::c_str()みたいなの。
    const char* c_str() const { return data_; }
	const char* data() const { return data_; }

	// 文字列の i 番目。
    char& operator[](std::size_t i) { return data_[i]; }
    const char& operator[](std::size_t i) const { return data_[i]; }

	// 文字列のappend
	// ⚠ Capacityを超えた場合は、即座にstd::terminate()を呼び出す。
    FixedString& operator+=(const char* str) {
        size_t len = std::strlen(str);
        if (length_ + len > Capacity)
            std::terminate();
        std::memcpy(data_ + length_, str, len);
        length_ += len;
        data_[length_] = '\0';
        return *this;
    }

	// 文字列のappend
    FixedString& operator+=(const FixedString& other) { return (*this += other.c_str()); }

	// string型への暗黙の変換子
    operator std::string() const { return std::string(data_, length_); }

	// string_view型への暗黙の変換子
    operator std::string_view() const { return std::string_view(data_, length_); }

	// 同一であるかの比較
    template<typename T>
    bool operator==(const T& other) const noexcept {
        return (std::string_view) (*this) == other;
    }

	// 異なる内容であるかの比較
    template<typename T>
    bool operator!=(const T& other) const noexcept {
        return (std::string_view) (*this) != other;
    }

	// 格納している文字列をclearする。
    void clear() {
        length_  = 0;
        data_[0] = '\0';
    }

   private:
	// 文字バッファ(終端の`\0`を考慮して1byte多めに確保)
    char        data_[Capacity + 1];  // +1 for null terminator

	// 格納している文字列の長さ。
    std::size_t length_;
};

// --------------------
//   コマンドライン
// --------------------

struct CommandLine {
public:
	CommandLine() {}
	CommandLine(int _argc, char** _argv) :
		argc(_argc),
		argv(_argv) {}

	// コンストラクタでargc,argvを渡さなかった時に、あとから設定する。
	void set_arg(int _argc, char** _argv) { argc = _argc, argv = _argv; }

	// 起動フォルダを返す
	// 💡 文字列の末尾には`\`がついている。
	// ⚠ set_arg()を事前に呼び出して、コマンドラインから渡されたargc, argvをセットしてあること。
	// 🤔 やねうら王では、Directory::GetBinaryDirectory()を用いる。この関数は内部的に呼び出される。
    static std::string get_binary_directory() { return g.get_binary_directory(g.argv[0]); }

	// argv0 : コマンドラインから渡されたargv[0]を渡して、そこから起動フォルダを返す。
	// Stockfishとの互換性のために用意。やねうら王では呼び出さない。
	static std::string get_binary_directory(std::string argv0);

	// cwd(current working directory)
	static std::string get_working_directory();

	int    argc;
	char** argv;

	// global object
	static CommandLine g;
};

// --------------------
//     Utility
// --------------------

namespace Utility {

// vectorのなかから、条件に合致するものを探して、見つかればそれを先頭に移動させる。
// 元の先頭から、その見つけた要素の1つ前までは後方に1つずらす。
template<typename T, typename Predicate>
void move_to_front(std::vector<T>& vec, Predicate pred) {
    auto it = std::find_if(vec.begin(), vec.end(), pred);

    if (it != vec.end())
    {
        std::rotate(vec.begin(), it, it + 1);
    }
}
}

// 到達しないことを明示して最適化を促す。
// 💡 sf_assume(false)ならば、そこには到達しないことを明示する。sf_assume(true)ならば到達する。
//     clangを除外してあるのは、警告が消えないからっぽい。

#if defined(__GNUC__)
    #define sf_always_inline __attribute__((always_inline))
#elif defined(__MSVC)
    #define sf_always_inline __forceinline
#else
    // do nothign for other compilers
    #define sf_always_inline
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #if __GNUC__ >= 13
        #define sf_assume(cond) __attribute__((assume(cond)))
    #else
        #define sf_assume(cond) \
            do \
            { \
                if (!(cond)) \
                    __builtin_unreachable(); \
            } while (0)
    #endif
#else
    // do nothing for other compilers
    #define sf_assume(cond)
#endif

// --------------------
//    ツール類
// --------------------

class ThreadPool;

namespace Tools
{
	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishではtt.cppにこのコードがあるのだが、独自の置換表を確保したいときに
	// これが独立していないと困るので、ここに用意する。
	// nameは"Hash" , "eHash"などクリアしたいものの名前を書く。
	// メモリクリアの途中経過が出力されるときにその名前(引数nameで渡している)が出力される。
	// name == nullptrのとき、途中経過は表示しない。
	void memclear(YaneuraOu::ThreadPool& threads, const char* name, void* table, size_t size);

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
	void exit();

	// 指定されたミリ秒だけsleepする。
	void sleep(u64 ms);

	// 現在時刻を文字列化したもを返す。(評価関数の学習時などにログ出力のために用いる)
	std::string now_string();

	// Linux環境ではgetline()したときにテキストファイルが'\r\n'だと
	// '\r'が末尾に残るのでこの'\r'を除去するためにwrapperを書く。
	// そのため、ifstreamに対してgetline()を呼び出すときは、
	// std::getline()ではなくこのこの関数を使うべき。
	bool getline(std::ifstream& fs, std::string& s);

	// マルチバイト文字列をワイド文字列に変換する。
	// WindowsAPIを呼び出しているのでWindows環境専用。
	std::wstring MultiByteToWideChar(const std::string& s);

	// 他言語にあるtry～finally構文みたいなの。
	// SCOPE_EXIT()マクロの実装で使う。このクラスを直接使わないで。
	struct __FINALLY__ {
		__FINALLY__(std::function<void()> fn_) : fn(fn_) {}
		~__FINALLY__() { fn(); }
	private:
		std::function<void()> fn;
	};

	// --------------------
	//    ProgressBar
	// --------------------

	// 処理の進捗を0%から100%の間で出力する。
	class ProgressBar
	{
	public:
		ProgressBar(){}

		// size_ : 全件でいくらあるかを設定する。
		ProgressBar(u64 size_);

		// また0%に戻す。このインスタンスを再利用する時に用いる。
		void reset(u64 size_);

		// 進捗を出力する。
		// current : 現在までに完了している件数
		void check(u64 current);

		// Progress Barの有効/無効を切り替える。
		// "readyok"までにProgressBarが被るとよろしくないので
		// learnコマンドとmakebookコマンドの時以外はオフでいいと思う。
		static void enable(bool b) { enable_ = b; }

	private:
		// 全件の数。
		u64 size;

		// 前回までに何個dotを出力したか。
		size_t dots;

		static bool enable_;
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

		// ファイルが存在しないエラー。
		FileNotFound,

		// ファイルのオープンに失敗。
		FileOpenError,

		// ファイル読み込み時のエラー。
		FileReadError,

		// ファイル書き込み時のエラー。
		FileWriteError,

		// ファイルClose時のエラー。
		FileCloseError,

		// ファイルを間違えているエラー。
		FileMismatch,

		// フォルダ作成時のエラー。
		CreateFolderError,

		// 実装されていないエラー。
		NotImplementedError,
	};

	// ResultCodeを文字列化する。
	std::string to_string(ResultCode);

	// エラーを含む関数の返し値を表現する型
	// RustにあるOption型のような何か
	struct Result
	{
		constexpr Result(ResultCode code_) : code(code_) {}

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
		static constexpr Result Ok() { return Result(ResultCode::Ok); }
	};
}

// スコープを抜ける時に実行してくれる。BOOST::BOOST_SCOPE_EXITマクロみたいな何か。
// 使用例) SCOPE_EXIT( x = 10 );
#define SCOPE_EXIT(STATEMENT) Tools::__FINALLY__ __clean_up_object__([&]{ STATEMENT });


// --------------------
//  ファイル操作
// --------------------

namespace SystemIO
{
	// ファイルを丸読みする。ファイルが存在しなくともエラーにはならない。空行はスキップする。末尾の改行は除去される。
	// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
	// 引数で渡されるtrimはtrueを渡すと末尾のスペース、タブがトリムされる。
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	// 💡 filenameは、起動フォルダ相対で指定する。
	Tools::Result ReadAllLines(const std::string& filename, std::vector<std::string>& lines, bool trim = false);

	// ファイルにすべての行を書き出す。
	// 💡 filenameは、起動フォルダ相対で指定する。
	Tools::Result WriteAllLines(const std::string& filename, std::vector<std::string>& lines);


	// msys2、Windows Subsystem for Linuxなどのgcc/clangでコンパイルした場合、
	// C++のstd::ifstreamで::read()は、一発で2GB以上のファイルの読み書きが出来ないのでそのためのwrapperである。
	// 	※　注意　どのみち32bit環境ではsize_tが4バイトなので2(4?)GB以上のファイルは書き出せない。
	//
	// read_file_to_memory()の引数のcallback_funcは、ファイルがオープン出来た時点でそのファイルサイズを引数として
	// callbackされるので、バッファを確保して、その先頭ポインタを返す関数を渡すと、そこに読み込んでくれる。
	//
	// また、callbackされた関数のなかでバッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
	// nullptrを返せば良い。このとき、read_file_to_memory()は、読み込みを中断し、エラーリターンする。
	// 💡 filenameは、起動フォルダ相対で指定する。

	Tools::Result ReadFileToMemory(const std::string& filename, std::function<void* (size_t)> callback_func);
	Tools::Result WriteMemoryToFile(const std::string& filename, void* ptr, size_t size);

	// 通常のftell/fseekは2GBまでしか対応していないので特別なバージョンが必要である。

	size_t ftell64(FILE* f);
	int fseek64(FILE* f, size_t offset, int origin);

	// C#のTextReaderみたいなもの。
	// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
	// fopen()～fread()で実装されている。
	struct TextReader
	{
		TextReader();
		virtual ~TextReader();

		// ファイルをopenする。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::string& filename);

		// Open()を呼び出してオープンしたファイルをクローズする。
		void Close();

		// ファイルの終了判定。
		// ファイルを最後まで読み込んだのなら、trueを返す。

		// 1行読み込む(改行まで) 引数のlineに代入される。
		// 改行コードは返さない。
		// SkipEmptyLine(),SetTrim()の設定を反映する。
		// Eofに達した場合は、返し値としてTools::ResultCode::Eofを返す。
		// 先頭のUTF-8のBOM(EF BB BF)は無視する。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result ReadLine(std::string& line);

		// ReadLine()で空行を読み飛ばすかどうかの設定。
		// (ここで行った設定はOpen()/Close()ではクリアされない。)
		// デフォルトでfalse
		void SkipEmptyLine(bool skip = true) { skipEmptyLine = skip; }

		// ReadLine()でtrimするかの設定。
		// 引数のtrimがtrueの時は、ReadLine()のときに末尾のスペース、タブはトリムする
		// (ここで行った設定はOpen()/Close()ではクリアされない。)
		// デフォルトでfalse
		void SetTrim(bool trim = true) { this->trim = trim; }

		// ファイルサイズの取得
		// ファイルポジションは先頭に移動する。
		size_t GetSize();

		// 現在のファイルポジションを取得する。
		// 先読みしているのでReadLineしている場所よりは先まで進んでいる。
		size_t GetFilePos() { return ftell64(fp); }

		// 現在の行数を返す。(次のReadLine()で返すのがテキストファイルの何行目であるかを返す) 0 origin。
		// またここで返す数値は空行で読み飛ばした時も、その空行を1行としてカウントしている。
		size_t GetLineNumber() const { return line_number; }

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

		// 何行目であるか
		// エラー表示の時などに用いる
		// 現在の行。(0 origin)
		size_t line_number;

		// ReadLine()で行の末尾をtrimするかのフラグ。
		bool trim;

		// ReadLine()で空行をskipするかのフラグ
		bool skipEmptyLine;
	};

	// Text書き出すの速いやつ。
	class TextWriter
	{
	public:
		// 書き出し用のバッファサイズ([byte])
		static constexpr size_t buf_size = 4096;

		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::string& filename);

		// 文字列を書き出す(改行コードは書き出さない)
		Tools::Result Write(const std::string& str);

		// 1行を書き出す(改行コードも書き出す) 改行コードは"\r\n"とする。
		Tools::Result WriteLine(const std::string& line);

		// ptrの指すところからsize [byte]だけ書き出す。
		Tools::Result Write(const char* ptr, size_t size);

		// 内部バッファにあってまだファイルに書き出していないデータをファイルに書き出す。
		// ※　Close()する時に呼び出されるので通常この関数を呼び出す必要はない。
		Tools::Result Flush();

		Tools::Result Close();
		TextWriter() : buf(buf_size) { clear(); }
		virtual ~TextWriter() { Close(); }

	private:
		// 変数を初期化する。
		void clear() { fp = nullptr; write_cursor = 0; }

		FILE* fp = nullptr;

		// 書き出し用のbuffer。これがいっぱいになるごとにfwriteする。
		std::vector<char> buf;

		// 書き出し用のcursor。次に書き出す場所は、buf[write_cursor]。
		size_t write_cursor;
	};

	// BinaryReader,BinaryWriterの基底クラス
	class BinaryBase
	{
	public:
		// ファイルを閉じる。デストラクタからClose()は呼び出されるので明示的に閉じなくても良い。
		Tools::Result Close();

		virtual ~BinaryBase() { Close(); }

	protected:
		FILE* fp = nullptr;
	};

	// binary fileの読み込みお手伝いclass
	class BinaryReader : public BinaryBase
	{
	public:
		// ファイルのopen
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::string& filename);

		// ファイルサイズの取得
		// ファイルポジションは先頭に移動する。
		size_t GetSize();

		// ptrの指すメモリにsize[byte]だけファイルから読み込む
		// ファイルの末尾を超えて読み込もうとした場合、Eofが返る。
		// ファイルの末尾に超えて読み込もうとしなかった場合、Okが返る。
		// 引数で渡されたバイト数読み込むことができなかった場合、FileReadErrorが返る。
		// size_of_read_bytesがnullptrでない場合、実際に読み込まれたバイト数が代入される。
		// ※　sizeは2GB制限があるので気をつけて。
		Tools::Result Read(void* ptr , size_t size, size_t* size_of_read_bytes = nullptr);
	};

	// binary fileの書き出しお手伝いclass
	class BinaryWriter : public BinaryBase
	{
	public:
		// ファイルのopen
		// append == trueで呼び出すと、このあとWriteしたものはファイル末尾に追記される。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::string& filename, bool append = false);

		// ptrの指すメモリからsize[byte]だけファイルに書き込む。
		// ※　sizeは2GB制限があるので気をつけて。
		Tools::Result Write(void* ptr, size_t size);
	};
};

// Reads the file as bytes.
// Returns std::nullopt if the file does not exist.

// ファイルをバイトとして読み込みます。
// ファイルが存在しない場合は std::nullopt を返します。

// 💡 filenameは、起動フォルダ相対で指定する。

std::optional<std::string> read_file_to_string(const std::string& filename);

// --------------------
//       Path
// --------------------

// C#にあるPathクラス的なもの。ファイル名の操作。
// C#のメソッド名に合わせておく。
namespace Path
{
	// path名とファイル名を結合して、それを返す。
	// folder名のほうは空文字列でないときに、末尾に'/'か'\\'がなければそれを付与する。
	// 与えられたfilenameが絶対Pathである場合、folderを連結せずに単にfilenameをそのまま返す。
	// 与えられたfilenameが絶対Pathであるかの判定は、内部的にはPath::IsAbsolute()を用いて行う。
	//
	// 実際の連結のされ方については、UnitTestに例があるので、それも参考にすること。
	std::string Combine(const std::string& folder, const std::string& filename);

	// full path表現から、(フォルダ名をすべて除いた)ファイル名の部分を取得する。
	std::string GetFileName(const std::string& path);

	// full path表現から、(ファイル名だけを除いた)ディレクトリ名の部分を取得する。
	std::string GetDirectoryName(const std::string& path);

	// 絶対Pathであるかの判定。
	// ※　std::filesystem::absolute() は MSYS2 で Windows の絶対パスの判定に失敗するらしいので自作。
	//
	// 絶対Pathの条件 :
	//   "\\"(WindowsのUNC)で始まるか、"/"で始まるか(Windows / Linuxのroot)、"~"で始まるか、"C:"(ドライブレター + ":")で始まるか。
	//
	// 絶対Pathの例)
	//   C:/YaneuraOu/Eval  ← Windowsのドライブレター付きの絶対Path
	//   \\MyNet\MyPC\Eval  ← WindowsのUNC
	//   ~myeval            ← Linuxのhome
	//   /YaneuraOu/Eval    ← Windows、Linuxのroot
	bool IsAbsolute(const std::string& path);

	// ファイルが存在するかの確認
	bool Exists(const std::string& path);

};

// --------------------
//    Directory
// --------------------

// ディレクトリに存在するファイルの列挙用
// C#のDirectoryクラスっぽい何か
namespace Directory
{
	// 指定されたフォルダに存在するファイルをすべて列挙する。
	// 列挙するときに引数extensionで列挙したいファイル名の拡張子を指定できる。(例 : ".bin")
	// 拡張子として""を指定すればすべて列挙される。
	std::vector<std::string> EnumerateFiles(const std::string& sourceDirectory, const std::string& extension);

	// フォルダを作成する。
	// working directory相対で指定する。dir_nameに日本語は使っていないものとする。
	// ※　Windows環境だと、この関数名、WinAPIのCreateDirectoryというマクロがあって…。
	// 　ゆえに、CreateDirectory()をやめて、CreateFolder()に変更する。
	Tools::Result CreateFolder(const std::string& dir_name);

	// 起動時のフォルダを返す。
	std::string GetBinaryFolder();
}

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

namespace Parser
{
	// スペースで区切られた文字列を解析するためのparser
	struct LineScanner
	{
		// 解析したい文字列を渡す(スペースをセパレータとする)
		LineScanner(std::string line_) : line(line_), pos(0) {}

		// 次のtokenを先読みして返す。get_token()するまで解析位置は進まない。
		std::string peek_text();

		// 次のtokenを返す。
		std::string get_text();

		// 現在のcursor位置から残りの文字列を取得する。
		// peek_text()した分があるなら、それも先頭にくっつけて返す。
		std::string get_rest();

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

	// PythonのArgumenetParserみたいなやつ
	// istringstream isを食わせて、そのうしろを解析させて、所定の変数にその値を格納する。
	// 使い方)
	//  isに"min 10 max 80"のような文字列が入っているとする。
	//
	//   ArgumentParser parser;
	//   int min=0,max=100;
	//   parser.add_argument("min",min);
	//   parser.add_argument("max",max);
	//   parser.parse_args(is);
	//
	// とすると min = 10 , max = 80となる。

	class ArgumentParser
	{
	public:
		typedef std::pair<std::string /*arg_name*/, std::function<void(std::istringstream&)>> ArgPair;

		// 引数を登録する。
		template<typename T>
		void add_argument(const std::string& arg_name, T& v)
		{
			auto f = [&](std::istringstream& is) { is >> v; };
			a.emplace_back(ArgPair(arg_name,f));
		}

		// 事前にadd_argument()で登録しておいた内容に基づき、isを解釈する。
		void parse_args(std::istringstream& is)
		{
			std::string token;
			while (is >> token)
				for (auto p : a)
					// 合致すれば次に
					if (p.first == token)
					{
						p.second(is);
						break;
					}
		}

		std::vector <ArgPair> a;
	};
}

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
//    文字列 拡張
// --------------------

// 文字列拡張(やねうら王独自)
namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// Windowsだと_stricmp() , Linuxだとstrcasecmp()を使うのだが、
	// 後者がどうも動作が怪しい。自前実装しておいたほうが無難。
	// stricmpは、string case insensitive compareの略？
	// s1==s2のとき0(false)を返す。
	bool stricmp(const std::string& s1, const std::string& s2);

	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	// ios::binaryでopenした場合などには'\r'なども入っていることがあるので…。
	std::string trim(const std::string& input);

	// trim()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_inplace(std::string& input);

	// 行の末尾の数字を除去した文字列を返す。
	// sfenの末尾の手数を削除する用
	// 末尾のスペースを詰めたあと数字を詰めてそのあと再度スペースを詰める処理になっている。
	// 例 : "abc 123 "→"abc"となって欲しいので。
	std::string trim_number(const std::string& input);

	// trim_number()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_number_inplace(std::string& s);

	// 文字列をint化する。int化に失敗した場合はdefault_の値を返す。
	int to_int(const std::string input, int default_);

	// 文字列をfloat化する。float化に失敗した場合はdefault_の値を返す。
	float to_float(const std::string input, float default_);

	// スペース、タブなど空白に相当する文字で分割して返す。
	std::vector<std::string> split(const std::string& input);

	// 先頭にゼロサプライした文字列を返す。
	// 例) n = 123 , digit = 6 なら "000123"という文字列が返る。
	std::string to_string_with_zero(u64 n, int digit);

	// --- 以下、C#のstringクラスにあるやつ。

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	bool StartsWith(std::string const& value, std::string const& starting);

	// 文字列valueが、文字列endingで終了していればtrueを返す。
	bool EndsWith(std::string const& value, std::string const& ending);

	// 文字列sのなかに文字列tが含まれるかを判定する。含まれていればtrueを返す。
	bool Contains(const std::string& s, const std::string& t);

	// 文字列valueに対して文字xを文字yに置換した新しい文字列を返す。
	std::string Replace(std::string const& value, char x, char y);

	// 文字列を大文字にして返す。
	std::string ToUpper(std::string const& value);

	// sを文字列spで分割した文字列集合を返す。
	std::vector<std::string_view> Split(std::string_view s, std::string_view delimiter);

	// Pythonの delemiter.join(v) みたいなの。
	// 例: v = [1,2,3] に対して ' '.join(v) == "1 2 3"
	std::string Join(const std::vector<std::string>& v , const std::string& delimiter);
};

// sを文字列spで分割した文字列集合を返す。
// ※ Stockfishとの互換性のために用意。
std::vector<std::string_view> split(std::string_view s, std::string_view delimiter);

// スペース相当文字列を削除する。⇨ NUMAの処理に必要
void remove_whitespace(std::string& s);

// スペース相当文字列かどうかを判定する。⇨ NUMAの処理に必要
bool is_whitespace(std::string_view s);

// "123"みたいな文字列を123のように数値型(size_t)に変換する。
size_t str_to_size_t(const std::string& s);

// --------------------
//    Concurrent
// --------------------

// 並列プログラミングでよく使うコンテナ類
namespace Concurrent
{
	// マルチスレッドプログラミングでよく出てくるProducer Consumer Queue
	template <typename T>
	class ConcurrentQueue
	{
	public:
		// [ASYNC] Queueのpop(一番最後にpushされた要素を取り出す)
		T pop()
		{
			std::unique_lock<std::mutex> lk(mutex_);

			// 要素がないなら待つしかない
			while (queue_.empty())
				cond_.wait(lk);

			auto val = queue_.front();
			queue_.pop();

			lk.unlock();
			cond_.notify_one();
			return val;
		}

		// [ASYNC] 先頭要素を返す。
		T& front() {
			// dequeは再配置しないことが保証されている。
			// そのためread-onlyで取得するだけならlock不要。
			return queue_.front();
		}

		// [ASYNC] Queueのpush(queueに要素を一つ追加する)
		void push(const T& item)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			queue_.push(item);
			lk.unlock();
			cond_.notify_one();
		}

		// [ASYNC] Queueの保持している要素数を返す。
		size_t size()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			return queue_.size();
		}

		// [ASYNC] Queueをclearする。
		void clear()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			// clear by assignment
			queue_ = std::queue<T>();
		}

		// copyの禁止
		ConcurrentQueue() = default;
		ConcurrentQueue(const ConcurrentQueue&) = delete;

		// 代入の禁止
		ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

	private:
		std::queue<T> queue_;
		std::mutex mutex_;
		std::condition_variable cond_;
	};

	// std::unordered_setの並列版
	template <typename T>
	class ConcurrentSet
	{
	public:
		// [ASYNC] Setのremove。
		void remove(const T& item)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			set_.remove(item);
		}

		// [ASYNC] Setに要素を一つ追加する。
		void emplace(const T& item)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			set_.insert(item);
		}

		// [ASYNC] Setに要素があるか確認する。
		bool contains(const T& item)
		{
			std::unique_lock<std::mutex> lk(mutex_);
			return set_.find(item) != set_.end();
		}

		// [ASYNC] Setの保持している要素数を返す。
		size_t size()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			return set_.size();
		}

		// [ASYNC] Setをclearする。
		void clear()
		{
			std::unique_lock<std::mutex> lk(mutex_);
			// clear by assignment
			set_ = std::unordered_set<T>();
		}

		// copyの禁止
		ConcurrentSet() = default;
		ConcurrentSet(const ConcurrentSet&) = delete;

		// 代入の禁止
		ConcurrentSet& operator=(const ConcurrentSet&) = delete;

	private:
		std::unordered_set<T> set_;
		std::mutex mutex_;
	};
}

// --------------------
// StandardInputWrapper
// --------------------

// 標準入力のwrapper
// 事前にコマンドを積んだりできる。
class StandardInput
{
public:
	// 標準入力から1行もらう。Ctrl+Zが来れば"quit"が来たものとする。
	// また先行入力でqueueに積んでおくことができる。(次のinput()で取り出される)
	std::string input();

	// 先行入力としてqueueに積む。(次のinput()で取り出される)
	void push(const std::string& s);

	// main()に引数として渡されたパラメーターを解釈してqueueに積む。
	void parse_args(const CommandLine& cli);

private:
	// 先行入力されたものを積んでおくqueue。
	// これが尽きれば標準入力から入力する。
	std::queue<std::string> cmds;
};

// --------------------
//     UnitTest
// --------------------

namespace Misc {
	// このheaderに書いてある関数のUnitTest。
	void UnitTest(Test::UnitTester& tester, IEngine& engine);
}

} // namespace YaneuraOu

// FixedString型のstd::hashの特殊化
// 📝 FixedString<N> を string_view に変換して
//     string_view のハッシュ関数をそのまま使うので高速
template<std::size_t N>
struct std::hash<YaneuraOu::FixedString<N>> {
    std::size_t operator()(const YaneuraOu::FixedString<N>& fstr) const noexcept {
        return std::hash<std::string_view>{}((std::string_view) fstr);
    }
};

#endif // #ifndef MISC_H_INCLUDED
