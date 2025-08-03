#ifndef USI_OPTION_H_INCLUDED
#define USI_OPTION_H_INCLUDED

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <cstdint>

namespace YaneuraOu {

// Define a custom comparator, because the UCI options should be case-insensitive
// カスタムコンパレータを定義します。UCIオプションは大文字と小文字を区別しないためです。
// ⇨ USIではここがプロトコル上どうなっているのかはわからないが、同様の処理にしておく。
struct CaseInsensitiveLess {
	bool operator() (const std::string&, const std::string&) const;
};

class OptionsMap;

// The Option class implements each option as specified by the UCI protocol
// Optionクラスは、UCIプロトコルで指定された各オプションを実装します

class Option {
   public:
    // 値が変更された時に呼び出されるevent handlerの型。
    using OnChange = std::function<std::optional<std::string>(const Option&)>;

    Option(const OptionsMap*);
    Option(OnChange = nullptr);

    // bool
    Option(bool v, OnChange = nullptr);

    // string
    Option(const char* v, OnChange = nullptr);

    // integer
#if STOCKFISH
	Option(double v, int minv, int maxv, OnChange = nullptr);
#else
	// ⇨ 💡 やねうら王では、引数をint64_tに変更
    Option(int64_t v, int64_t minv, int64_t maxv, OnChange = nullptr);
#endif

    // combo
#if STOCKFISH
    // 📌 Option("B var A var B var C","B")のような形式。
    //     これは使いづらい。やねうら王では使わない。
    Option(const char* v, const char* cur, OnChange = nullptr);
#else
	// 📌 Option(vector{"A","B","C"},"B")のような形式。
    Option(const std::vector<std::string>& list, const std::string& cur, OnChange = nullptr);
#endif

    Option& operator=(const std::string&);

    //operator int() const;
    // 📌 やねうら王では、int64_tに変更する。
    operator int64_t() const;

    operator std::string() const;
    bool operator==(const char*) const;
    bool operator!=(const char*) const;

    friend std::ostream& operator<<(std::ostream&, const OptionsMap&);

    int operator<<(const Option&) = delete;

    // -- やねうら王独自

    // 固定化フラグ。
    // これを true にすると、operator = で変更できなくなる。
    bool fixed = false;

   private:
    friend class OptionsMap;
    friend class Engine;
    friend class Tune;

    // このオプション設定のdefaultの値、現在の値、type。
    // 💡 typeは USIプロトコルのsetoptionの時に指定できるオプションの型名。
    /*
		 📓 defaultValueは、USIプロトコルの
			"option name XXX type YYY default [defaultValue]"の形で表示するときの
			 文字列をそのまま格納している。
	
			このため、type == comboのときは、
			 USIでは"standard_book.db var no_book var standard_book.db"のようにvarが複数回出てくるし、
			 同じ値が2回でてくる。
			 UCIは、"var"は1度しか出てこないので、このため、Stockfishとはコードが異なるので注意。
	*/
    std::string defaultValue, currentValue, type;

    // このオプション設定がint型であるときに、最小値と最大値。
    // 📒 Stockfishではintだが、やねうら王ではint64_tに変更。
    int64_t min, max;

    // 追加した順に0,1,2,…
    // 💡 これは、OptionsMap.add()で追加する時に設定される。
    // 📝 "usi"コマンド応答で、OptionsMapへの登録順に出力されてほしいので、
    //     カウンターを0から増やしていき、Option::idxが一致したものを表示していくようになっている。
    //     この変数は、そのためのもの。
    size_t idx;

    // このOptionの設定値が変更された時に呼び出されるevent handler。
    OnChange on_change;

    // 親objectへのpointer
    const OptionsMap* parent = nullptr;
};

// 思考エンジンオプションを保持しておくためのclass。
// USIの1つのオプション設定が Option class。
// これを std::map<option名, Option> で保持しているのがこのclass。
class OptionsMap {
   public:
    using InfoListener = std::function<void(std::optional<std::string>)>;

    OptionsMap()                             = default;
    OptionsMap(const OptionsMap&)            = delete;
    OptionsMap(OptionsMap&&)                 = delete;
    OptionsMap& operator=(const OptionsMap&) = delete;
    OptionsMap& operator=(OptionsMap&&)      = delete;

	// option項目が変更されてon_change() handlerが呼び出された時に
	// on_change()の返し値を引数にして呼び出されるhandlerを設定する。
    void add_info_listener(InfoListener&&);

    // USIのsetoptionコマンドのhandler
    void setoption(std::istringstream&);

    // あるoption名に対応するOptionオブジェクトを取得する。
    // これはread onlyで、設定はここからしてはならない。
    const Option& operator[](const std::string&) const;

    // Optionを一つ追加する。options_mapに追加される。
    void add(const std::string& option_name, const Option& option);

    // 保持しているOptionのなかで、このoption_nameを持つものの数。
    // 💡 ある名前のoption項目を持っているかどうかを調べるのに使う。
    std::size_t count(const std::string& option_name) const;

#if !STOCKFISH

    // 📌 やねうら王独自拡張 📌

    // カレントフォルダにfilename(例えば"engine_options.txt")が
    // あればそれをオプションとしてOptions[]の値をオーバーライドする機能。
    // ここで設定した値は、そのあとfixedフラグが立ち、その後、
    // 通常の"setoption"では変更できない。
    void read_engine_options(const std::string& filename);

    // option名を指定して、その値を出力した文字列を構成する。
    // option名が省略された時は、すべてのオプションの値を出力した文字列を構成する。
    std::string get_option(const std::string& option_name);

    // option名とvalueを指定して、そのoption名があるなら、そのoptionの値を変更する。
    // 返し値) 値を変更したとき、変更できなかったときいずれも、出力するメッセージを返す。
    std::string set_option_if_exists(const std::string& option_name,
                                     const std::string& option_value);

    // idxを指定して、それに対応するOptionを取得する。
    // ⚠ 値が存在しないidxを指定すると落ちる。
    std::pair<const std::string, const Option&> get_option_by_idx(size_t idx) const;

#endif

   private:
    friend class Engine;
    friend class Option;

    // OptionsMapの中身一覧を出力する。
    // 💡 "usi"コマンドの応答に用いる。
    friend std::ostream& operator<<(std::ostream&, const OptionsMap&);

    // The options container is defined as a std::map
    // オプションのコンテナは std::map として定義されています
    // 💡 これは思考エンジンのオプション名からOption(オプション設定 object)へのmap。

    using OptionsStore = std::map<std::string, Option, CaseInsensitiveLess>;

    OptionsStore options_map;
    InfoListener info;

#if !STOCKFISH
    // 思考エンジンがGUIからの"usi"に対して返す"option ..."文字列から
    // Optionオブジェクトを構築して、それを *this に突っ込む。
    // "engine_options.txt"というファイルの各行からOptionオブジェクト構築して
    // Optionの値を上書きするためにこの関数が必要。
    // "option name USI_Hash type spin default 256"
    // のような文字列が引数として渡される。
    // このとき、Optionのhandlerとidxは書き換えない。
    void build_option(const std::string& line);
#endif
};

// OptionsMapを参照で使いたい時に使うproxy。(やねうら王独自拡張)
// 📝 C++ではclass memberの参照型はコンストラクタで初期化(代入)しなければならない。コンストラクタ以外で代入するには、
//     std::reference_wrapperを用いればいいのだが、これを用いる場合、operator[]がうまく処理できない。
//     そこで、operator[]を処理できるOptionsMapを用意する。これにより、コンストラクタ以外で代入できる。
class OptionsMapRef
{
public:
	// 参照のsetter/getter
	void set_ref(OptionsMap& o) { options = &o; }
	//OptionsMap& get_ref() const { return *options; }

	// -- 以下のmethodはOptionsMapの同名methodに委譲する。

	const Option& operator[](const std::string& option_name) const { return (*options)[option_name]; };
	void add(const std::string& option_name, const Option& option) { return (*options).add(option_name, option); }

private:
	OptionsMap* options = nullptr;
};

} // namespace YaneuraOu

#endif  // #ifndef USI_OPTION_H_INCLUDED
