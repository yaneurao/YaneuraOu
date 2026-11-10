#ifndef MISC_H_INCLUDED
#define MISC_H_INCLUDED

#if !STOCKFISH
#include "config.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>  // IWYU pragma: keep
// IWYU pragma: no_include <__exception/terminate.h>
#include <functional>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cstring>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>

#if !defined(NO_PREFETCH) && (defined(_MSC_VER) || defined(__INTEL_COMPILER))
    #include <immintrin.h>
#endif

#define stringify2(x) #x
#define stringify(x) stringify2(x)

namespace YaneuraOu {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8  = std::uint8_t;

using i64 = std::int64_t;
using i32 = std::int32_t;
using i16 = std::int16_t;
using i8  = std::int8_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

#if defined(__GNUC__) && defined(IS_64BIT)
__extension__ using u128 = unsigned __int128;
__extension__ using i128 = signed __int128;
#endif

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
#if STOCKFISH
std::string engine_info(bool to_uci = false);
#else
std::string engine_info(const std::string& engine_name,
						const std::string& engine_author,
                        const std::string& engine_version,
                        const std::string& eval_name);

// config.hで設定した値などについて出力する。
std::string config_info();

#endif

// 使用したコンパイラについての文字列を返す。
std::string compiler_info();

// --------------------
//    prefetch命令
// --------------------

// prefetch()は、与えられたアドレスの内容をL1/L2 cacheに事前に読み込む。
// これはnon-blocking関数で、CPUがメモリに読み込むのを待たない。

// Prefetch hint enums for explicit call-site control.
// 呼び出し側から読み書きと局所性のヒントを指定する。
enum class PrefetchRw {
    READ,
    WRITE
};

// NOTE: PrefetchLoc controls locality / cache level, not whether a prefetch
//       is issued. In particular, PrefetchLoc::NONE maps to a non-temporal /
//       lowest-locality prefetch (Intel: _MM_HINT_NTA, GCC/Clang: locality = 0)
//       and therefore still performs a prefetch. To completely disable
//       prefetching, define NO_PREFETCH so that prefetch() becomes a no-op.
// PrefetchLocは局所性・キャッシュ階層を指定するもので、発行の有無は指定しない。
// NONEも非一時的／最低局所性のヒントとしてプリフェッチを発行する。
// Intelでは_MM_HINT_NTA、GCC/Clangではlocality=0に対応する。
// 完全な無効化にはNO_PREFETCHを定義し、prefetch()を何もしない関数にする。
enum class PrefetchLoc {
    NONE,      // Non-temporal / no cache locality (still issues a prefetch)
               // 非一時的なアクセス。プリフェッチ自体は発行する。
    LOW,       // Low locality (e.g. T2 / L2)
               // 低い局所性。
    MODERATE,  // Moderate locality (e.g. T1 / L1)
               // 中程度の局所性。
    HIGH       // High locality (e.g. T0 / closest cache)
               // 高い局所性。
};

// Preloads the given address into cache. This is a non-blocking
// function that doesn't stall the CPU waiting for data to be loaded from memory,
// which can be quite slow.
// 指定アドレスのデータを事前にキャッシュへ読み込む。読み込み完了は待たない。
#ifdef NO_PREFETCH
template<PrefetchRw RW = PrefetchRw::READ, PrefetchLoc LOC = PrefetchLoc::HIGH>
void prefetch(const void*) {}
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)

constexpr int get_intel_hint(PrefetchRw rw, PrefetchLoc loc) {
    if (rw == PrefetchRw::WRITE)
    #ifdef _MM_HINT_ET0
        return _MM_HINT_ET0;
    #else
        // Fallback when write-prefetch hint is not available: use T0
        // 書き込み用のヒントが使えない場合はT0を使う。
        return _MM_HINT_T0;
    #endif
    switch (loc)
    {
    case PrefetchLoc::NONE :
        return _MM_HINT_NTA;
    case PrefetchLoc::LOW :
        return _MM_HINT_T2;
    case PrefetchLoc::MODERATE :
        return _MM_HINT_T1;
    case PrefetchLoc::HIGH :
        return _MM_HINT_T0;
    default :
        return _MM_HINT_T0;
    }
}

template<PrefetchRw RW = PrefetchRw::READ, PrefetchLoc LOC = PrefetchLoc::HIGH>
void prefetch(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), get_intel_hint(RW, LOC));
}
#else
template<PrefetchRw RW = PrefetchRw::READ, PrefetchLoc LOC = PrefetchLoc::HIGH>
void prefetch(const void* addr) {
    __builtin_prefetch(addr, static_cast<int>(RW), static_cast<int>(LOC));
}
#endif

// --------------------
//  logger
// --------------------

// cin/coutへの入出力をファイルにリダイレクトを開始/終了する。
void start_logger(const std::filesystem::path& fname);

// "123"みたいな文字列を123のように数値型(usize)に変換する。
std::optional<usize> str_to_size_t(const std::string& s);

// ワイド文字列をUTF-8へ変換する。WindowsのUnicode起動引数で使用する。
std::string utf8_from_wstring(std::wstring_view s);

// UTF-8のパスをOSのネイティブ形式へ変換する。
// WindowsではUTF-8として不正な場合にANSIコードページで再試行する。
std::filesystem::path path_from_utf8(const std::string& path);

// Reads the file as bytes.
// Returns std::nullopt if the file does not exist.

// ファイルをバイトとして読み込みます。
// ファイルが存在しない場合は std::nullopt を返します。

// 💡 filenameは、起動フォルダ相対で指定する。

std::optional<std::string> read_file_to_string(const std::string& path);

// --------------------
//  統計情報
// --------------------

// bがtrueであった回数 / dbg_hit_on()が呼び出された回数を調べる。
// slotを指定すると、複数の統計を独立して記録できる。戻り値は観測した値。
bool dbg_hit_on(bool cond, int slot = 0);
i64  dbg_mean_of(i64 value, int slot = 0);
i64  dbg_stdev_of(i64 value, int slot = 0);
i64  dbg_extremes_of(i64 value, int slot = 0);
void dbg_correl_of(i64 value1, i64 value2, int slot = 0);

// 探索部から1秒おきにdbg_print()が呼び出されるものとする。
// このとき、記録した統計情報をcerrに出力する。
void dbg_print();
void dbg_clear();

// --------------------
//  Time[ms] wrapper
// --------------------

// ms単位での時間計測しか必要ないのでこれをTimePoint型のように扱う。
using TimePoint = std::chrono::milliseconds::rep;  // A value in milliseconds
                                                 // ミリ秒単位の値。
static_assert(sizeof(TimePoint) == sizeof(i64), "TimePoint should be 64 bits");

// ms単位で現在時刻を返す
inline TimePoint now() {
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

// sを文字列spで分割した文字列集合を返す。
// ※ Stockfishとの互換性のために用意。
inline std::vector<std::string_view> split(std::string_view s, std::string_view delimiter) {
    std::vector<std::string_view> res;

    if (s.empty())
        return res;

    usize begin = 0;
    for (;;)
    {
        const usize end = s.find(delimiter, begin);
        if (end == std::string_view::npos)
            break;

        res.emplace_back(s.substr(begin, end - begin));
        begin = end + delimiter.size();
    }

    res.emplace_back(s.substr(begin));

    return res;
}

// スペース相当文字列を削除する。⇨ NUMAの処理に必要
void remove_whitespace(std::string& s);

// スペース相当文字列かどうかを判定する。⇨ NUMAの処理に必要
bool is_whitespace(std::string_view s);

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

// True if and only if the binary is compiled on a little-endian machine
// バイナリがlittle-endian環境向けにコンパイルされている場合に限りtrue。
static inline const u16  Le             = 1;
static inline const bool IsLittleEndian = *reinterpret_cast<const char*>(&Le) == 1;


// --------------------
//      ValueList
// --------------------

//  最大サイズが固定長のvectorみたいなやつ。
template<typename T, usize MaxSize>
class ValueList {

public:
	// 配列のsizeを返す。
    usize size() const { return size_; }

	// 配列のsizeをintで返す。
    int         ssize() const { return int(size_); }

	void        push_back(const T& value) {
        assert(size_ < MaxSize);
        values_[size_++] = value;
    }
    // pushes back value if value < max
    // valueがmax未満のときだけ要素数を増やす。
    void push_back_if_lt(const T& value, const T& max) {
        assert(size_ < MaxSize);
        values_[size_] = value;
        size_ += (value < max);
    }
	const T* begin() const { return values_; }
	const T* end() const { return values_ + size_; }

	const T& operator[](int index) const { return values_[index]; }
	// ⇨ ここの引数、どうせ大きな配列は確保しないのでsize_tではなくintで良い。

	// count個の要素を増やす。
	// 要素は未初期化のまま増やした要素の先頭のポインタを返す。
    T* make_space(usize count) {
        T* result = &values_[size_];
        size_ += count;
        assert(size_ <= MaxSize);
        return result;
    }

private:
	T           values_[MaxSize];
	usize size_ = 0;
};

// --------------------
//      MultiArray
// --------------------

template<typename T, usize Size, usize... Sizes>
class MultiArray;

namespace Detail {

	template<typename T, usize Size, usize... Sizes>
	struct MultiArrayHelper {
		using ChildType = MultiArray<T, Sizes...>;
	};

	template<typename T, usize Size>
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

template<typename T, usize Size, usize... Sizes>
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

	constexpr auto& at(size_type index) { return data_.at(index); }
	constexpr const auto& at(size_type index) const { return data_.at(index); }

	constexpr auto& operator[](size_type index) noexcept { assert(index < Size); return data_[index]; }
	constexpr const auto& operator[](size_type index) const noexcept { assert(index < Size); return data_[index]; }

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
			if constexpr (sizeof...(Sizes) == 0)
				ele = v;
			else
				ele.fill(v);
	}

	constexpr void swap(MultiArray<T, Size, Sizes...>& other) noexcept { data_.swap(other.data_); }
};

// Wrapper around std::atomic<T> which uses relaxed accesses or plain
// accesses, depending on the config. Intended use is e.g. wasm where
// the overhead of atomic instructions can be significant, and we only
// require non-tearing for the updates, while ensuring we use relaxed
// accesses otherwise.
// 更新時に値が途中で分断されないことを保証するためのstd::atomic<T>のラッパー。
// 設定に応じてrelaxedアクセスまたは通常アクセスを使用する。
// atomic命令のコストが大きくなりうるwasmなどでの使用を想定し、
// 通常はrelaxedアクセスを使いつつ、必要に応じて通常アクセスへ切り替えられる。
template<typename T>
class RelaxedAtomic {
    static constexpr bool UseAtomic =
#ifdef USE_SLOPPY_ATOMICS
      !std::atomic<T>::is_always_lock_free || sizeof(T) > sizeof(usize);
#else
      true;
#endif

   public:
    RelaxedAtomic() = default;
    RelaxedAtomic(T val) :
        inner(val) {}
    RelaxedAtomic(const RelaxedAtomic& a) :
        inner(static_cast<T>(a)) {}

    T operator=(T val) {
        this->store(val, std::memory_order_relaxed);
        return val;
    }

    RelaxedAtomic& operator=(const RelaxedAtomic& a) {
        this->store(static_cast<T>(a), std::memory_order_relaxed);
        return *this;
    }

    operator T() const { return this->load(std::memory_order_relaxed); }

    RelaxedAtomic& operator+=(T val) {
        this->store(this->load(std::memory_order_relaxed) + val, std::memory_order_relaxed);
        return *this;
    }

    RelaxedAtomic& operator++() { return *this += 1; }

    RelaxedAtomic& operator--() { return *this -= 1; }

    T operator++(int) {
        T val = this->load(std::memory_order_relaxed);
        this->store(val + 1, std::memory_order_relaxed);
        return val;
    }

    T operator--(int) {
        T val = this->load(std::memory_order_relaxed);
        this->store(val - 1, std::memory_order_relaxed);
        return val;
    }

    RelaxedAtomic& operator-=(T val) {
        this->store(this->load(std::memory_order_relaxed) - val, std::memory_order_relaxed);
        return *this;
    }

    T load(std::memory_order order) const {
        assert(order == std::memory_order_relaxed);
        if constexpr (UseAtomic)
            return inner.load(order);
        else
            return inner;
    }

    void store(T val, std::memory_order order) {
        assert(order == std::memory_order_relaxed);
        if constexpr (UseAtomic)
            inner.store(val, order);
        else
            inner = val;
    }

   private:
    std::conditional_t<UseAtomic, std::atomic<T>, T> inner;
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

#if STOCKFISH
inline usize mul_hi64(u64 a, usize b) {
// Stockfishのこの部分、u64に書き換え忘れているのだと思う。
#else
inline u64 mul_hi64(u64 a, u64 b) {
#endif
#if defined(__GNUC__) && defined(IS_64BIT) && !defined(__wasm__)
    return (u128(a) * u128(b)) >> 64;
#else
    // 64bit同士の掛け算を32bitずつに分割して上位64bitを求める。
    u64 aL = u32(a), aH = a >> 32;
    u64 bL = u32(b), bH = u64(b) >> 32;
    u64 c1 = (aL * bL) >> 32;
    u64 c2 = aH * bL + c1;
    u64 c3 = aL * bH + u32(c2);
    return aH * bH + (c2 >> 32) + (c3 >> 32);
#endif
}

template<typename T1, typename T2>
inline constexpr T2 interpolate(T1 x, T1 x0, T1 x1, T2 y0, T2 y1) {
    assert(x0 != x1);
    return T2(y0 + (y1 - y0) * (x - x0) / (x1 - x0));
}


// --------------------
//   hash値の計算
// --------------------

// 📓 SFNNのバイナリに対してhash値を計算するためのヘルパー関数群。
//     プロセス間でNNUEのパラメーター共有を行うときのチェック用。

// 任意の byte streamをバイト列そのままのハッシュとして利用し、uint64_tで返す。
u64 hash_bytes(const char*, usize);

// 任意の POD ライクなデータ構造を、バイト列そのままのハッシュとして利用し、usizeで返す。
// 📝 `reinterpret_cast` でメモリ内容を生のまま string_view にして hash を取る。
//     その実装はコンパイラ依存かつendian依存。

template<typename T>
inline usize get_raw_data_hash(const T& value) {
    // We must have no padding bytes because we're reinterpreting as char
    // charとして再解釈するため、パディングバイトがあってはならない。
    static_assert(std::has_unique_object_representations_v<T>);

    return static_cast<usize>(
      hash_bytes(reinterpret_cast<const char*>(&value), sizeof(value)));
}

template<typename T>
inline void hash_combine(usize& seed, const T& v) {
    usize x;
    // For primitive types we avoid using the default hasher, which may be
    // nondeterministic across program invocations
    // 整数型には、起動ごとに値が変わりうる標準ハッシャーを使わない。
    if constexpr (std::is_integral_v<T>)
        x = v;
    else
        x = std::hash<T>{}(v);
    seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// 文字列に対してそのhash値を返す。
// 💡 アルゴリズムが固定されていないと値が変わるのでC++のruntimeに依存するわけにはいかない。
inline u64 hash_string(const std::string& sv) { return hash_bytes(sv.data(), sv.size()); }


// --------------------
//   コマンドライン
// --------------------

struct CommandLine {
public:
    CommandLine() = default;
    CommandLine(int _argc, char** _argv);

    CommandLine(const CommandLine&)            = delete;
    CommandLine& operator=(const CommandLine&) = delete;
    CommandLine(CommandLine&&)                 = default;
    CommandLine& operator=(CommandLine&&)      = default;

	// コンストラクタでargc,argvを渡さなかった時に、あとから設定する。
    // WindowsではOSからUnicodeの起動引数を取得し、UTF-8へ変換して保持する。
    void set_arg(int _argc, char** _argv);

    // argv0で指定された実行ファイルの配置ディレクトリを返す。
    // WindowsではOSから取得した実行ファイルのパスを優先する。
    static std::filesystem::path get_binary_directory(std::filesystem::path argv0);

#if !STOCKFISH
    // 起動フォルダを文字列で返す、やねうら王の互換API。
    // 末尾にパス区切り文字を付ける。Directory::GetBinaryFolder()から使用する。
    // Windowsでは既存のファイル入出力に合わせ、ANSIコードページの文字列を返す。
    static std::string get_binary_directory();
#endif

	int    argc;
	char** argv;

	// global object
	static CommandLine g;

   private:
#ifdef _WIN32
    // argvが指すUTF-8文字列とポインター配列。CommandLineと同じ寿命を持つ。
    std::vector<std::string> argv_storage;
    std::vector<char*>       argv_utf8;
#endif
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
        std::rotate(vec.begin(), it, it + 1);
}
}

// コンパイラーごとのインライン化・仮定・到達不能箇所などの指定。
// sf_assumeの条件は副作用を含まず、必ず成立する式を指定する。
#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

#if defined(__GNUC__)
    #define sf_always_inline inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define sf_always_inline __forceinline
#else
    // plain inline for other compilers
    // その他のコンパイラーでは通常のinlineを使う。
    #define sf_always_inline inline
#endif

#if defined(__clang__)
    #define sf_assume(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
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
#elif defined(_MSC_VER)
    #define sf_assume(cond) __assume(cond)
#else
    // do nothing for other compilers
    // その他のコンパイラーでは何もしない。
    #define sf_assume(cond)
#endif

#ifdef __GNUC__
    #define sf_unreachable() __builtin_unreachable()
#elif defined(_MSC_VER)
    #define sf_unreachable() __assume(0)
#else
    #define sf_unreachable()
#endif

#ifdef __GNUC__
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT
#endif

#ifdef __clang__
    #define UNROLLED_FOR(...) _Pragma("clang loop unroll(full)") for (__VA_ARGS__)
#elif defined(__GNUC__)
    #define UNROLLED_FOR(...) _Pragma("GCC unroll 32") for (__VA_ARGS__)
#else
    #define UNROLLED_FOR(...) for (__VA_ARGS__)
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
		i64 get_number(i64 defaultValue);
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

#if !STOCKFISH

class IEngine;
namespace Test { class UnitTester; }

namespace Misc {
	// このheaderに書いてある関数のUnitTest。
	void UnitTest(Test::UnitTester& tester, IEngine& engine);
}
#endif

} // namespace YaneuraOu

#endif // #ifndef MISC_H_INCLUDED
