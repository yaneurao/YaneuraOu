#ifndef USI_OPTION_H_INCLUDED
#define USI_OPTION_H_INCLUDED

// TODO : このファイル作業中

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>

//namespace Stockfish {

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
	using OnChange = std::function<std::optional<std::string>(const Option&)>;

	Option(const OptionsMap*);
	Option(OnChange = nullptr);
	Option(bool v, OnChange = nullptr);
	Option(const char* v, OnChange = nullptr);
	Option(double v, int minv, int maxv, OnChange = nullptr);
	Option(const char* v, const char* cur, OnChange = nullptr);

	Option& operator=(const std::string&);
	operator int() const;
	operator std::string() const;
	bool operator==(const char*) const;
	bool operator!=(const char*) const;

	friend std::ostream& operator<<(std::ostream&, const OptionsMap&);

private:
	friend class OptionsMap;
	friend class Engine;
	friend class Tune;

	void operator<<(const Option&);

	std::string       defaultValue, currentValue, type;
	int               min, max;
	size_t            idx;
	OnChange          on_change;
	const OptionsMap* parent = nullptr;
};

// USIのoption名と、それに対応する設定内容を保持している。
class OptionsMap {
public:
	using InfoListener = std::function<void(std::optional<std::string>)>;

	OptionsMap() = default;
	OptionsMap(const OptionsMap&) = delete;
	OptionsMap(OptionsMap&&) = delete;
	OptionsMap& operator=(const OptionsMap&) = delete;
	OptionsMap& operator=(OptionsMap&&) = delete;

	void add_info_listener(InfoListener&&);

	void setoption(std::istringstream&);

	Option  operator[](const std::string&) const;
	Option& operator[](const std::string&);

	std::size_t count(const std::string&) const;

private:
	friend class Engine;
	friend class Option;

	friend std::ostream& operator<<(std::ostream&, const OptionsMap&);

	// The options container is defined as a std::map
	// オプションのコンテナは std::map として定義されています

	using OptionsStore = std::map<std::string, Option, CaseInsensitiveLess>;

	OptionsStore options_map;
	InfoListener info;
};

//}

#endif  // #ifndef USI_OPTION_H_INCLUDED
