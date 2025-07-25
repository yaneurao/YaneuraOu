#ifndef TUNE_H_INCLUDED
#define TUNE_H_INCLUDED

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>  // IWYU pragma: keep
#include <utility>
#include <vector>

namespace YaneuraOu {

class OptionsMap;

// オプション項目の最小値と最大値
using Range    = std::pair<int, int>;  // Option's min-max values

// Rangeを設定する関数型。
// 💡 intを引数にとり、Rangeを返す関数。
using RangeFun = Range(int);

// Default Range function, to calculate Option's min-max values
// デフォルトの範囲関数。Option の最小値と最大値を計算するための関数
// 💡 この関数は RangeFun 型として使える。
// 📝 v > 0なら [0, 2v]、v < 0 なら[2v, 0]の範囲となる。
inline Range default_range(int v) { return v > 0 ? Range(0, 2 * v) : Range(2 * v, 0); }

// パラメーターの範囲を持っているfunctor。
struct SetRange {

    // 📓 コンストラクタにRangeFunを渡して設定する方法と、[min, max]を渡して設定する方法がある。

    explicit SetRange(RangeFun f) :
        // 💡 この引数はRangeFun* fの意味だが、
        //     C++の仕様として関数型は自動的に関数型ポインタに自動変換されるので問題ない。
        fun(f) {}

    SetRange(int min, int max) :
        fun(nullptr),
        range(min, max) {}

    // コンストラクタで渡した2種類に応じてRangeが返る。
    Range operator()(int v) const { return fun ? fun(v) : range; }

    RangeFun* fun;
    Range     range;
};

#define SetDefaultRange SetRange(default_range)


// Tune class implements the 'magic' code that makes the setup of a fishtest tuning
// session as easy as it can be. Mainly you have just to remove const qualifiers
// from the variables you want to tune and flag them for tuning, so if you have:
//
//   const Value myValue[][2] = { { V(100), V(20) }, { V(7), V(78) } };
//
// If you have a my_post_update() function to run after values have been updated,
// and a my_range() function to set custom Option's min-max values, then you just
// remove the 'const' qualifiers and write somewhere below in the file:
//
//   TUNE(SetRange(my_range), myValue, my_post_update);
//
// You can also set the range directly, and restore the default at the end
//
//   TUNE(SetRange(-100, 100), myValue, SetDefaultRange);
//
// In case update function is slow and you have many parameters, you can add:
//
//   UPDATE_ON_LAST();
//
// And the values update, including post update function call, will be done only
// once, after the engine receives the last UCI option, that is the one defined
// and created as the last one, so the GUI should send the options in the same
// order in which have been defined.
/*
	 Tuneクラスは、fishtestチューニングセッションのセットアップをできるだけ簡単にする
	 「魔法の」コードを実装しています。基本的には、チューニングしたい変数から
	 const 修飾子を取り除き、それをチューニング対象としてマークするだけです。例えば:

	   const Value myValue[][2] = { { V(100), V(20) }, { V(7), V(78) } };

	 値の更新後に実行する my_post_update() 関数や、カスタムの
	 Option の最小・最大値を設定する my_range() 関数がある場合は、
	 const 修飾子を外し、ファイル内のどこかで以下のように記述します:

	   TUNE(SetRange(my_range), myValue, my_post_update);

	 範囲を直接設定し、最後にデフォルト値に戻すこともできます:

	   TUNE(SetRange(-100, 100), myValue, SetDefaultRange);

	 もし更新関数が遅く、パラメータが多い場合には、

	   UPDATE_ON_LAST();

	 を追加することができます。これにより、値の更新および更新後の関数呼び出しは、
	 エンジンが最後の UCI オプション（つまり定義・生成された最後のオプション）を
	 受け取った後に一度だけ行われます。したがって、GUI はオプションを定義された順に
	 送信する必要があります。
*/
/*
	📓
		上の例で myValueという二次元配列がTUNE()によってどう展開されるか疑問に思うかもしれない。

		実際は、Tune::add()が多次元配列を受け取れるtemplate methodになっているので、
		以下のように展開される。
			myValue[0][0]   = 100
			myValue[0][1]   = 20
			myValue[1][0]   = 7
			myValue[1][1]   = 78
*/

class Tune {

    using PostUpdate = void();  // Post-update function

    Tune() { read_results(); }
    Tune(const Tune&)           = delete;
    void operator=(const Tune&) = delete;
    void read_results();

	// Tuneのinstanceを得る。(これはsingletonである)
    static Tune& instance() {
        static Tune t;
        return t;
    }  // Singleton

    // Use polymorphism to accommodate Entry of different types in the same vector
	// 多態性を利用して、異なる型の Entry を同一のベクタ内で扱えるようにする
	// 📝 1つのオプション項目を表現する基底型。
    struct EntryBase {
        virtual ~EntryBase()       = default;
        virtual void init_option() = 0;
        virtual void read_option() = 0;
    };

	// 📝 1つのオプション項目を表現する型。
    template<typename T>
    struct Entry: public EntryBase {

		// Tがconstだと書き換えられないから駄目。
        static_assert(!std::is_const_v<T>, "Parameter cannot be const!");

		// TはintかPostUpdate型でないと駄目。
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, PostUpdate>,
                      "Parameter type not supported!");

		// オプション名、値、範囲
        Entry(const std::string& n, T& v, const SetRange& r) :
            name(n),
            value(v),
            range(r) {}

		void operator=(const Entry&) = delete;  // Because 'value' is a reference

		// OptionMapsにname,value,rangeのオプション項目を追加する。
        void init_option() override;

        // OptionMapsがこのオプション名と同名のオプションを持っているなら、
        // そちらからオプションの設定値を読み込む。
        void read_option() override;

		// パラメーター名
        std::string name;

		// 保持しているパラメーター(参照で持っている)
        T&          value;

		// パラメーターの範囲
        SetRange    range;
    };

    // Our facility to fill the container, each Entry corresponds to a parameter
    // to tune. We use variadic templates to deal with an unspecified number of
    // entries, each one of a possible different type.
	// コンテナを埋めるための仕組みであり、各 Entry は調整すべきパラメーターに対応する。
    // 可変長テンプレートを使うことで、型が異なる可能性のある任意の数のエントリを扱う。

	/*
		📓
			namesというカンマ区切りの文字列から一つ要素を取り出す。
			"1, (2,3) , 4"のようになっている場合、呼び出すごとに"1", "2 3", "4"が返ってくる。
			pop : 取り出した部分の要素を消すフラグ。
	*/
    static std::string next(std::string& names, bool pop = true);

    int add(const SetRange&, std::string&&) { return 0; }

    template<typename T, typename... Args>
    int add(const SetRange& range, std::string&& names, T& value, Args&&... args) {
        list.push_back(std::unique_ptr<EntryBase>(new Entry<T>(next(names), value, range)));
        return add(range, std::move(names), args...);
    }

    // Template specialization for arrays: recursively handle multi-dimensional arrays
    // 配列用のテンプレート特殊化：多次元配列を再帰的に処理する

	// 💡 TUNE( .. )で配列を指定したときに、それを展開するためのテンプレート。

	template<typename T, size_t N, typename... Args>
    int add(const SetRange& range, std::string&& names, T (&value)[N], Args&&... args) {
        for (size_t i = 0; i < N; i++)
            add(range, next(names, i == N - 1) + "[" + std::to_string(i) + "]", value[i]);
        return add(range, std::move(names), args...);
    }

    // Template specialization for SetRange
    // SetRange 用のテンプレート特殊化

    template<typename... Args>
    int add(const SetRange&, std::string&& names, SetRange& value, Args&&... args) {
        return add(value, (next(names), std::move(names)), args...);
    }

	// OptionMapsにname,value,rangeのオプション項目を追加する。
    // 💡 init_option()から呼び出される。
    static void make_option(OptionsMap* options, const std::string& n, int v, const SetRange& r);

	// このclassが持っているEntry(オプション項目)。
    std::vector<std::unique_ptr<EntryBase>> list;

   public:
    template<typename... Args>
    static int add(const std::string& names, Args&&... args) {
        return instance().add(SetDefaultRange, names.substr(1, names.size() - 2),
                              args...);  // Remove trailing parenthesis
    }

	// 引数で渡されたOptionsMap oに、
	// このclassが持っているオプション項目すべてを生やす。
    static void init(OptionsMap& o) {
        options = &o;
        for (auto& e : instance().list)
            e->init_option();
        read_options();
    }  // Deferred, due to UCIEngine::Options access
       // UCIEngine::Options へのアクセスのため、処理を遅延

	// init()でオプション項目を生やしたOptionsMapから
	// このclassが持っている(OptionsMapに生やしたオプションと)同名の
	// オプション項目の設定値を読み込む。
    static void read_options() {
        for (auto& e : instance().list)
            e->read_option();
    }

    static bool        update_on_last;
    static OptionsMap* options;
};

template<typename... Args>
constexpr void tune_check_args(Args&&...) {
    static_assert((!std::is_fundamental_v<Args> && ...), "TUNE macro arguments wrong");
}

// Some macro magic :-) we define a dummy int variable that the compiler initializes calling Tune::add()
// ちょっとしたマクロのマジック :-)
// コンパイラが Tune::add() を呼び出して初期化するダミーの int 変数を定義している

// 引数 x を文字列リテラルに変換する。
//  例： STRINGIFY(abc) → "abc"
#define STRINGIFY(x) #x

// トークン連結演算子で、 x と y を結合する。
//  例：UNIQUE2(foo, 42) → foo42
#define UNIQUE2(x, y) x##y

// 2段階マクロ展開を行うためのマクロ。
//  __LINE__ のようなマクロは、直接##でつなぐと展開されないことがある。
//  そのため、UNIQUE(x, y) を通して一度展開し、UNIQUE2 で最終的に連結させる。
//  例： UNIQUE(p, __LINE__) → p123（123はその行番号）
#define UNIQUE(x, y) UNIQUE2(x, y)  // Two indirection levels to expand __LINE__

// TUNE(param1, param2, ...)
// のように呼ぶと、 無名ラムダ関数[]()->int{ ... } 
// を即時実行し、 その戻り値を、p行番号 という名前のダミー int 変数に代入する。
/*
	📓
		TUNE(10, 20);

		は、以下のように展開される。

		int p123 = []() -> int {
			tune_check_args(10, 20);
			return Tune::add("(10, 20)", 10, 20);
		}();
*/

#define TUNE(...) \
    int UNIQUE(p, __LINE__) = []() -> int { \
        tune_check_args(__VA_ARGS__); \
        return Tune::add(STRINGIFY((__VA_ARGS__)), __VA_ARGS__); \
    }();


// 呼び出すと Tune::update_on_last = true が実行され、
// その結果を bool p行番号 というダミー変数に代入する。
// 💡 こちらもUNIQUE() で変数名が行ごとにユニーク化される。

#define UPDATE_ON_LAST() bool UNIQUE(p, __LINE__) = Tune::update_on_last = true

}  // namespace Stockfish

#endif  // #ifndef TUNE_H_INCLUDED
