
//#include "evaluate.h"
#include "misc.h"
//#include "search.h
//#include "syzygy/tbprobe.h"
#include "thread.h"
//#include "tt.h"
#include "usi.h"

namespace YaneuraOu {

// 文字列が(大文字小文字を無視して)一致すればtrueを返す比較operator。
// UCIではオプションはcase insensitive(大文字・小文字の区別をしない)なのでcustom comparatorを用意する。
// USIではここがプロトコル上どうなっているのかはわからないが、同様の処理にしておく。
bool CaseInsensitiveLess::operator()(const std::string& s1, const std::string& s2) const {

	return std::lexicographical_compare(
		s1.begin(), s1.end(), s2.begin(), s2.end(),
		[](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); });
}

// option項目が変更されてon_change() handlerが呼び出された時に
// on_change()の返し値を引数にして呼び出されるhandlerを設定する。
void OptionsMap::add_info_listener(InfoListener&& message_func) { info = std::move(message_func); }

// USIのsetoptionコマンドのhandler
void OptionsMap::setoption(std::istringstream& is) {
	std::string token, name, value;

	is >> token;  // Consume the "name" token

	// Read the option name (can contain spaces)
	while (is >> token && token != "value")
		name += (name.empty() ? "" : " ") + token;

	// Read the option value (can contain spaces)
	while (is >> token)
		value += (value.empty() ? "" : " ") + token;

	if (options_map.count(name))
		options_map[name] = value;
	else
		sync_cout << "No such option: " << name << sync_endl;
}

// あるoption名に対応するOptionオブジェクトを取得する。
// これはread onlyで、設定はここからしてはならない。
const Option& OptionsMap::operator[](const std::string& name) const {
	auto it = options_map.find(name);
#if STOCKFISH
	assert(it != options_map.end());
#else
	if (it == options_map.end())
	{
		// Optionを生やす前に参照したのだと思うので、エラーメッセージを出力して終了する。
		sync_cout << "Error : Options[" << name << "] , not found." << sync_endl;
		Tools::exit();
	}
#endif

	return it->second;
}

// Inits options and assigns idx in the correct printing order
// オプションを初期化し、正しい表示順になるように idx を割り当てます

void OptionsMap::add(const std::string& name, const Option& option) {
	if (!options_map.count(name))
	{
		// 未追加なので追加してやる。

		static size_t insert_order = 0;

		options_map[name] = option;

		options_map[name].parent = this;
		options_map[name].idx = insert_order++;
	}
	else
	{
		// すでに追加されていたのでabort。

		std::cerr << "Option \"" << name << "\" was already added!" << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// 保持しているOptionのなかで、このoption_nameを持つものの数。
std::size_t OptionsMap::count(const std::string& name) const { return options_map.count(name); }

Option::Option(const OptionsMap* map) :
	parent(map) {
}

Option::Option(const char* v, OnChange f) :
	type("string"),
	min(0),
	max(0),
	on_change(std::move(f)) {
	defaultValue = currentValue = v;
}

Option::Option(bool v, OnChange f) :
	type("check"),
	min(0),
	max(0),
	on_change(std::move(f)) {
	defaultValue = currentValue = (v ? "true" : "false");
}

Option::Option(OnChange f) :
	type("button"),
	min(0),
	max(0),
	on_change(std::move(f)) {
}

#if STOCKFISH
Option::Option(double v, int minv, int maxv, OnChange f) :
#else
Option::Option(int64_t v, int64_t minv, int64_t maxv, OnChange f) :
#endif
	type("spin"),
	min(minv),
	max(maxv),
	on_change(std::move(f)) {
	defaultValue = currentValue = std::to_string(v);
}

#if STOCKFISH

Option::Option(const char* v, const char* cur, OnChange f) :
	type("combo"),
	min(0),
	max(0),
	on_change(std::move(f)) {
	defaultValue = v;
	currentValue = cur;
}
#else

// 🤔 Stockfishのcombo optionの設定、すごく使いにくいので定義しなおす。
Option::Option(const std::vector<std::string>& list, const std::string& cur, OnChange f) :
	type("combo"),
	min(0),
	max(0),
	on_change(std::move(f)) {

	// listの文字列をスペース区切りで連結してdefaultValueに突っ込む。
	//defaultValue = StringExtension::Join(list, " ");
	// 📝 間違い。
	//    {"A","B","C"}なるstd::vector<string>とcur = "B"から
	//    "B var A var B var C"のような文字列を構築してdefaultValueに入れる必要がある。
    std::string s = cur;
    for (auto l : list)
        s += " var " + l;
    defaultValue = s;
	currentValue = cur;
}
#endif

Option::operator int64_t() const {
	ASSERT_LV1(type == "check" || type == "spin");
#if STOCKFISH
	return (type == "spin" ? std::stoi(currentValue) : currentValue == "true");
#else
	return (type == "spin" ? std::stoll(currentValue) : currentValue == "true");
#endif
}

Option::operator std::string() const {
#if STOCKFISH
    assert(type == "string");
#else
	ASSERT_LV1(type == "string" || type == "combo");
#endif
	return currentValue;
}

bool Option::operator==(const char* s) const {
	ASSERT_LV1(type == "combo");
    return !CaseInsensitiveLess()(currentValue, s) && !CaseInsensitiveLess()(s, currentValue);
}

bool Option::operator!=(const char* s) const { return !(*this == s); }

// Updates currentValue and triggers on_change() action. It's up to
// the GUI to check for option's limits, but we could receive the new value
// from the user by console window, so let's check the bounds anyway.

// currentValue を更新し、on_change() アクションを発動します。
// オプションの制限チェックは GUI 側で行うべきですが、
// コンソールウィンドウ経由でユーザーから新しい値が入力される可能性もあるため、
// 念のため範囲チェックを行いましょう。

Option& Option::operator=(const std::string& v) {

#if STOCKFISH
    assert(!type.empty());
#else
	ASSERT_LV1(!type.empty());

	// fixedになっていれば代入をキャンセル
	if (fixed)
		return *this;
#endif

	// 範囲外なら設定せずに返る。
	// "EvalDir"などでstringの場合は空の文字列を設定したいことがあるので"string"に対して空の文字チェックは行わない。
	if ((type != "button" && type != "string" && v.empty())
		|| (type == "check" && v != "true" && v != "false")
#if STOCKFISH
		|| (type == "spin" && (std::stof(v) < min || std::stof(v) > max)))
#else
		|| (type == "spin" && (std::stoll(v) < min || std::stoll(v) > max)))
#endif
		return *this;

	if (type == "combo")
	{
		OptionsMap         comboMap;  // To have case insensitive compare
									  // 📝 comboのvalueを小文字化して比較したいのでOptionsMapを流用する。
		std::string        token;
		std::istringstream ss(defaultValue);
		// defaultValueにスペース区切りで書かれているのでこれをparseする。
#if STOCKFISH
		while (ss >> token)
			comboMap.add(token, Option());
		if (!comboMap.count(v) || v == "var")
			// defaultValueのなかに見つからなかったのでリタイア
			// "var"が見つかってもこれは見つかったことにはしない。
			return *this;
#else
        while (ss >> token)
			// 📝 USIではvarは複数回出てくることがある。そこでこの時点で除外する。
			//     同じ値がdefulat値と選択肢との両方に出てくることもある。これも除外する。
            if (token != "var" && !comboMap.count(token))
                comboMap.add(token, Option());

		if (!comboMap.count(v))
        {
            // defaultValueのなかに見つからなかったのでリタイア
            sync_cout << "info string Error! : combo value not found, value = `" << v
                      << "`, values = " << defaultValue << sync_endl;
            return *this;
        }
#endif
	}

	if (type == "string")
		currentValue = v == "<empty>" ? "" : v;

	// ボタン型は値を設定するものではなく、単なるトリガーボタン。
	// ボタン型以外なら入力値をcurrentValueに反映させてやる。
	else if (type != "button")
		currentValue = v;

	// 適切な値の範囲であったので、
	// ハンドラが設定されているならハンドラを呼びだす。
	// 💡 値が変化したとは限らない。
	if (on_change)
	{
		const auto ret = on_change(*this);

		if (ret && parent != nullptr && parent->info != nullptr)
			parent->info(ret);
	}

	return *this;
}

std::ostream& operator<<(std::ostream& os, const OptionsMap& om) {
	// OptionsMapへの登録順に出力されてほしいので、idxを0から増やしていき、Option::idxが一致したものを表示していく。
	for (size_t idx = 0; idx < om.options_map.size(); ++idx)
		for (const auto& it : om.options_map)
			if (it.second.idx == idx)
			{
				const Option& o = it.second;
				// 📝 先頭で改行しているので、必ず1行目が空行になる。
				os << "\noption name " << it.first << " type " << o.type;

				if (o.type == "check" || o.type == "combo")
					os << " default " << o.defaultValue;

				else if (o.type == "string")
				{
					std::string defaultValue = o.defaultValue.empty() ? "<empty>" : o.defaultValue;
					os << " default " << defaultValue;
				}

				else if (o.type == "spin")
					os << " default " << int(stof(o.defaultValue)) << " min " << o.min << " max "
					<< o.max;

				break;
			}

	return os;
}


#if 0
	/// 'On change' actions, triggered by an option's value change
	// オプションの値が変更された時に呼び出されるOn changeアクション。

	//static void on_clear_hash(const Option&) { Search::clear(); }
	//static void on_hash_size(const Option& o) { TT.resize(size_t(o)); }
	static void on_logger(const Option& o) { start_logger(o); }
	//static void on_threads(const Option& o) { Threads.set(size_t(o)); }
	//static void on_tb_path(const Option& o) { Tablebases::init(o); }
	//static void on_eval_file(const Option&) { Eval::NNUE::init(); }

	// --- やねうら王独自拡張分の前方宣言

	// 前回のOptions["EvalDir"]
	std::string last_eval_dir;

#if defined(__EMSCRIPTEN__) && defined(EVAL_NNUE)
	// WASM NNUE
	// 前回のOptions["EvalFile"]
	std::string last_eval_file;
#endif

	std::ostream& operator<<(std::ostream& os, const OptionsMap& om)
	{
		// idxの順番を守って出力する
		for (size_t idx = 0; idx < om.size(); ++idx)
			for (const auto& it : om)
				if (it.second.idx == idx)
				{
					const Option& o = it.second;
					os << "option name " << it.first << " type " << o.type;

					if (o.type == "string" || o.type == "check" || o.type == "combo")
						os << " default " << o.defaultValue;

					if (o.type == "spin")
						// この範囲はStockfishではfloatになっているが、
						// やねうら王では、int64_tに変更する。
						os << " default " << int64_t(stoll(o.defaultValue))
						<< " min " << o.min
						<< " max " << o.max;

					// コンボボックス(やねうら王、独自追加)
					// USIで規定されている。
					if (o.list.size())
						for (auto v : o.list)
							os << " var " << v;

					// Stockfishはこの関数、最初に改行を放り込むように書いてあるけども、
					// 正直、使いにくいと思う。普通に末尾ごとに改行する。
					os << std::endl;

					break;
				}

		return os;
	}

	// --- 以下、やねうら王、独自拡張。

	// 評価関数を読み込んだかのフラグ。これはevaldirの変更にともなってfalseにする。
	bool load_eval_finished = false;

	// エンジンオプションをコンパイル時に設定する機能
	// "ENGINE_OPTIONS"で指定した内容を設定する。
	// 例) #define ENGINE_OPTIONS "FV_SCALE=24;BookFile=no_book"
	void set_engine_options(const std::string& options)
	{
		// ";"で区切って複数指定できるものとする。
		auto v = StringExtension::Split(options, ";");
		for (auto line : v)
			build_option(std::string(line));
	}

#endif

// --------------------
//  やねうら王独自拡張
// --------------------

// カレントフォルダに"engine_options.txt"(これは引数で指定されている)が
// あればそれをオプションとしてOptions[]の値をオーバーライドする機能。
// ここで設定した値は、そのあとfixedフラグが立ち、その後、
// 通常の"setoption"では変更できない。
void OptionsMap::read_engine_options(const std::string& filename)
{
	SystemIO::TextReader reader;
	if (reader.Open(filename).is_not_ok())
		return;

	sync_cout << "info string read engine options, path = " << filename << sync_endl;

	std::string line;
	while (reader.ReadLine(line).is_ok())
		build_option(line);
}

// 思考エンジンがGUIからの"usi"に対して返す"option ..."文字列から
// Optionオブジェクトを構築して、それを *this に突っ込む。
// "engine_options.txt"というファイルの各行からOptionオブジェクト構築して
// Optionの値を上書きするためにこの関数が必要。
// "option name USI_Hash type spin default 256"
// のような文字列が引数として渡される。
// このとき、Optionのhandlerとidxは書き換えない。
void OptionsMap::build_option(const std::string& line)
{
	// 1. "option ..."の形式

	// 2. エンジンオプション名と値だけを指定する形式で書かれているのかも知れない。
	// (既存のオプションの値のoverrideがしたい場合)

	// 3. オプション名=値 のような形式かも知れない(dlshogiの.iniはそうなっている)
	// その形式にも対応する必要がある。
	// よって、最初に"="を" "に置換してしまう。そうすれば、3.は、2.の形式と同じになる。

	auto& Options = options_map;

	const auto& line2 = StringExtension::Replace(line, '=', ' ');

	Parser::LineScanner scanner(line2);
	std::string token = scanner.get_text();

	std::string option_name, option_value;

	if (token != "option")
	{
		// 空行は無視
		if (token == "")
			return;

		auto it = Options.find(token);
		if (it == Options.end())
		{
			// 違うのか。何の形式で書こうとしているのだろうか…。
			std::cout << "Error : option name not found : " << token << std::endl;
			return;
		}

		option_name  = token;
		option_value = scanner.get_text();
	}
	else {

		std::string name, value, option_type;
		int64_t min_value = 0, max_value = 1;
		std::vector<std::string> combo_list;
		while (!scanner.eol())
		{
			std::string token = scanner.get_text();
			if (token == "name") name = scanner.get_text();
			else if (token == "type") option_type = scanner.get_text();
			else if (token == "default") value = scanner.get_text();
			else if (token == "min") min_value = stoll(scanner.get_text());
			else if (token == "max") max_value = stoll(scanner.get_text());
			else if (token == "var") {
				auto varText = scanner.get_text();
				combo_list.push_back(varText);
			}
			else {
				std::cout << "Error : invalid command: " << token << std::endl;
			}
		}

		if (Options.count(name) == 0)
		{
			std::cout << "Error : option name not found : " << name << std::endl;
			return;
		}

		option_name  = name;
		option_value = value;
	}

	Options[option_name] = option_value;
	Options[option_name].fixed = true; // 次にsetoptionで再代入できないようにfixに変更しておく。

	sync_cout << "info string engine option override. name = " << option_name << " , value = " << option_value << sync_endl;

}


// option名とvalueを指定して、そのoption名があるなら、そのoptionの値を変更する。
// 返し値) 値を変更したとき、変更できなかったときいずれも、出力するメッセージを返す。
std::string OptionsMap::set_option_if_exists(const std::string& option_name, const std::string& option_value)
{
	for (auto& o : options_map)
	{
		// 大文字、小文字を無視して比較。
		if (!StringExtension::stricmp(option_name, o.first))
		{
			options_map[o.first] = option_value;
			return std::string("Options[") + o.first + "] = " + option_value;
		}
	}
	return std::string("No such option: ") + option_name;
}

// idxを指定して、それに対応するOptionを取得する。
std::pair<const std::string, const Option&> OptionsMap::get_option_by_idx(size_t idx) const
{
	for (const auto& o : options_map)
		if (o.second.idx == idx)
			return o;

	assert(false);
	return *options_map.begin(); // 警告が出るので..
}

// option名を指定して、その値を出力した文字列を構成する。
// option名が省略された時は、すべてのオプションの値を出力した文字列を構成する。
std::string OptionsMap::get_option(const std::string& option_name)
{
	// すべてを出力するモード
	bool all = option_name == "";

	// キーの最大長を取得
	// 💡 "=="のindentを揃えたいため

	size_t max_key_length = 0;
	for (const auto& o : options_map) {
		max_key_length = std::max(max_key_length, o.first.length());
	}

	std::string result;
	for (size_t idx = 0; idx < options_map.size(); ++idx)
	{
		auto it = get_option_by_idx(idx);

		// 大文字、小文字を無視して比較。また、nameが指定されていなければすべてのオプション設定の現在の値を表示。
		if ((!StringExtension::stricmp(option_name, it.first)) || all)
		{
			result += "Options[" + it.first + "]"
				// "=="のindentを揃えるための処理
				+ std::string(max_key_length - it.first.length() + 1, ' ')
				+ "= "
				+ std::string(it.second) + "\n";

			if (!all)
				return result;
		}
	}
	if (!all)
		result += "No such option: " + option_name + "\n";

	return result;
}

} // namespace YaneuraOu
