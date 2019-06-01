#include "thread.h"
#include "tt.h"
#include "usi.h"
#include "misc.h"

using std::string;

// Option設定が格納されたglobal object。
USI::OptionsMap Options;

namespace USI {

	// --- やねうら王独自拡張分の前方宣言

	extern std::vector<std::string> ekr_rules;
	void set_entering_king_rule(const std::string& rule);
	void read_engine_options();
	

	// USIプロトコルで必要とされるcase insensitiveな less()関数
	bool CaseInsensitiveLess::operator() (const string& s1, const string& s2) const {

		return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
			[](char c1, char c2) { return tolower(c1) < tolower(c2); });
	}

	// optionのdefault値を設定する。
	void init(OptionsMap& o)
	{
		// Hash上限。32bitモードなら2GB、64bitモードなら1024GB
		const int MaxHashMB = Is64Bit ? 1024 * 1024 : 2048;

		// 並列探索するときのスレッド数
		// CPUの搭載コア数をデフォルトとすべきかも知れないが余計なお世話のような気もするのでしていない。

		o["Threads"] << Option(4, 1, 512, [](const Option& o) { Threads.set(o); });

		// USIプロトコルでは、"USI_Hash"なのだが、
		// 置換表サイズを変更しての自己対戦などをさせたいので、
		// 片方だけ変更できなければならない。
		// ゆえにGUIでの対局設定は無視して、思考エンジンの設定ダイアログのところで
		// 個別設定が出来るようにする。

#if !defined(MATE_ENGINE)
		o["Hash"] << Option(16, 1, MaxHashMB, [](const Option&o) { TT.resize(o); });

		// その局面での上位N個の候補手を調べる機能
		o["MultiPV"] << Option(1, 1, 800);

		// 弱くするために調整する。20なら手加減なし。0が最弱。
		o["SkillLevel"] << Option(20, 0, 20);
#else
		o["Hash"] << Option(4096, 1, MaxHashMB);
#endif

		// cin/coutの入出力をファイルにリダイレクトする
		o["WriteDebugLog"] << Option(false, [](const Option& o) { start_logger(o); });

		// ネットワークの平均遅延時間[ms]
		// この時間だけ早めに指せばだいたい間に合う。
		// 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
		o["NetworkDelay"] << Option(120, 0, 10000);

		// ネットワークの最大遅延時間[ms]
		// 切れ負けの瞬間だけはこの時間だけ早めに指す。
		// 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
		o["NetworkDelay2"] << Option(1120, 0, 10000);

		// 最小思考時間[ms]
		o["MinimumThinkingTime"] << Option(2000, 1000, 100000);

		// 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
		// 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
		// 対人のときに短めに設定して強制的に早指しにすることが出来る。
		o["SlowMover"] << Option(100, 1, 1000);

		// 引き分けまでの最大手数。256手ルールのときに256を設定すると良い。0なら無制限。
		o["MaxMovesToDraw"] << Option(0, 0, 100000);

		// 探索深さ制限。0なら無制限。
		o["DepthLimit"] << Option(0, 0, INT_MAX);

		// 探索ノード制限。0なら無制限。
		o["NodesLimit"] << Option(0, 0, INT64_MAX);

		// 引き分けを受け入れるスコア
		// 歩を100とする。例えば、この値を100にすると引き分けの局面は評価値が -100とみなされる。

		// 千日手での引き分けを回避しやすくなるように、デフォルト値を2に変更した。[2017/06/03]
		// ちなみに、2にしてあるのは、
		//  int contempt = Options["Contempt"] * PawnValue / 100; でPawnValueが100より小さいので
		// 1だと切り捨てられてしまうからである。

		o["Contempt"] << Option(2, -30000, 30000);

		// Contemptの設定値を先手番から見た値とするオプション。Stockfishからの独自拡張。
		// 先手のときは千日手を狙いたくなくて、後手のときは千日手を狙いたいような場合、
		// このオプションをオンにすれば、Contemptをそういう解釈にしてくれる。
		// この値がtrueのときは、Contemptを常に先手から見たスコアだとみなす。

		o["ContemptFromBlack"] << Option(false);


#if defined (USE_ENTERING_KING_WIN)
		// 入玉ルール
		o["EnteringKingRule"] << Option(USI::ekr_rules, USI::ekr_rules[EKR_27_POINT], [](const Option& o) { set_entering_king_rule(o); });
#endif
		// 評価関数フォルダ。これを変更したとき、評価関数を次のisreadyタイミングで読み直す必要がある。
		o["EvalDir"] << Option("eval", [](const USI::Option&o) { load_eval_finished = false; });

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32) && \
	 (defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_KPPPT) || defined(EVAL_KPPP_KKPT) || defined(EVAL_KKPP_KKPT) || \
	defined(EVAL_KPP_KKPT_FV_VAR) || defined(EVAL_KKPPT) ||defined(EVAL_EXPERIMENTAL) || defined(EVAL_HELICES) || defined(EVAL_NABLA) )
		// 評価関数パラメーターを共有するか
		// 異種評価関数との自己対局のときにこの設定で引っかかる人が後を絶たないのでデフォルトでオフにする。
		o["EvalShare"] << Option(false);
#endif

#if defined(EVAL_LEARN)
		// isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
		// test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
		// このコマンドの実行前に異常終了してしまう。
		// そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
		// test evalconvertコマンドを叩く。
		o["SkipLoadingEval"] << Option(false);
#endif

#if !defined(MATE_ENGINE) && !defined(FOR_TOURNAMENT) 
		// 読みの各局面ですべての合法手を生成する
		// (普通、歩の2段目での不成などは指し手自体を生成しないのですが、これのせいで不成が必要な詰みが絡む問題が解けないことが
		// あるので、このオプションを用意しました。トーナメントモードではこのオプションは無効化されます。)
		o["GenerateAllLegalMoves"] << Option(false);
#endif

		// 各エンジンがOptionを追加したいだろうから、コールバックする。
		USI::extra_option(o);

		// カレントフォルダに"engine_option.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
		read_engine_options();
	}

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
						// やねうら王では、s64に変更する。
						os << " default " << s64(stoll(o.defaultValue))
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

	// --- Optionクラスのコンストラクタと変換子

	Option::Option(const char* v, OnChange f) : type("string"), min(0), max(0), on_change(f)
	{
		defaultValue = currentValue = v;
	}

	Option::Option(bool v, OnChange f) : type("check"), min(0), max(0), on_change(f)
	{
		defaultValue = currentValue = (v ? "true" : "false");
	}

	Option::Option(OnChange f) : type("button"), min(0), max(0), on_change(f)
	{}

	// Stockfishでは第一引数がdouble型だが、これは使わないと思うのでs64に変更する。
	Option::Option(s64 v, s64 minv, s64 maxv, OnChange f) : type("spin"), min(minv), max(maxv), on_change(f)
	{
		defaultValue = currentValue = std::to_string(v);
	}

	Option::Option(const std::vector<std::string>&list, const std::string& v, OnChange f)
		: type("combo"), on_change(f), list(list)
	{
		defaultValue = currentValue = v;
	}

	// Stockfishでは、これdoubleになっているが、あまりいいと思えないので数値型はs64のみのサポートにする。
	Option::operator s64() const {
		ASSERT_LV1(type == "check" || type == "spin");
		return (type == "spin" ? stoll(currentValue) : currentValue == "true");
	}

	Option::operator std::string() const {
		ASSERT_LV1(type == "string" || type == "combo" /* 将棋用拡張*/);
		return currentValue;
	}

	bool Option::operator==(const char* s) const {
		ASSERT_LV1(type == "combo");
		return    !CaseInsensitiveLess()(currentValue, s)
			   && !CaseInsensitiveLess()(s, currentValue);
	}

	// この関数はUSI::init()から起動時に呼び出されるだけ。
	void Option::operator<<(const Option& o)
	{
		static size_t insert_order = 0;
		*this = o;
		idx = insert_order++; // idxは生成順に0から連番で番号を振る
	}

	// USIプロトコル経由で値を設定されたときにそれをcurrentValueに反映させる。
	Option& Option::operator=(const string& v) {

		ASSERT_LV1(!type.empty());

		// 範囲外なら設定せずに返る。
		// "EvalDir"などでstringの場合は空の文字列を設定したいことがあるので"string"に対して空の文字チェックは行わない。
		if (  ((type != "button" && type != "string") && v.empty())
			|| (type == "check" && v != "true" && v != "false")
			|| (type == "spin" && (stoll(v) < min || stoll(v) > max)))
			return *this;

		// ボタン型は値を設定するものではなく、単なるトリガーボタン。
		// ボタン型以外なら入力値をcurrentValueに反映させてやる。
		if (type != "button")
			currentValue = v;

		// 値が変化したのでハンドラを呼びだす。
		if (on_change)
			on_change(*this);

		return *this;
	}

	// --- 以下、やねうら王、独自拡張。

	// 評価関数を読み込んだかのフラグ。これはevaldirの変更にともなってfalseにする。
	bool load_eval_finished = false;

	// 入玉ルール
#if defined(USE_ENTERING_KING_WIN)
// デフォルトでは27点法
	EnteringKingRule ekr = EKR_27_POINT;
	// 入玉ルールのUSI文字列
	std::vector<std::string> ekr_rules = { "NoEnteringKing", "CSARule24" , "CSARule27" , "TryRule" };

	// 入玉ルールがGUIから変更されたときのハンドラ
	void set_entering_king_rule(const std::string& rule)
	{
		for (size_t i = 0; i < ekr_rules.size(); ++i)
			if (ekr_rules[i] == rule)
			{
				ekr = (EnteringKingRule)i;
				break;
			}
	}
#else
	EnteringKingRule ekr = EKR_NONE;
#endif

	// 思考エンジンがGUIからの"usi"に対して返す"option ..."文字列から
	// Optionオブジェクトを構築して、それをOptions[]に突っ込む。
	// "engine_options.txt"というファイルの各行からOptionオブジェクト構築して
	// Options[]の値を上書きするためにこの関数が必要。
	// "option name USI_Hash type spin default 256"
	// のような文字列が引数として渡される。
	void build_option(string line)
	{
		LineScanner scanner(line);
		if (scanner.get_text() != "option") return;

		string name, value, option_type;
		int64_t min_value = 0, max_value = 1;
		std::vector<string> combo_list;
		while (!scanner.eof())
		{
			auto token = scanner.get_text();
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

		if (Options.count(name) != 0)
		{
			// typeに応じたOptionの型を生成して代入する。このときに "<<"を用いるとidxが変わってしまうので overwriteで代入する。
			if (option_type == "check") Options[name].overwrite(Option(value == "true"));
			else if (option_type == "spin") Options[name].overwrite(Option(stoll(value), min_value, max_value));
			else if (option_type == "string") Options[name].overwrite(Option(value.c_str()));
			else if (option_type == "combo") Options[name].overwrite(Option(combo_list, value));
		}
		else
			std::cout << "Error : option name not found : " << name << std::endl;

	}

	// カレントフォルダに"engine_option.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
	void read_engine_options()
	{
		std::ifstream ifs("engine_options.txt");
		if (!ifs.fail())
		{
			std::string str;
			while (Dependency::getline(ifs, str))
				build_option(str);
		}
	}

	// idxの値を書き換えないoperator "<<"
	void Option::overwrite(const Option& o)
	{
		auto idx_ = idx; // backup
		*this = o;
		idx = idx_; // restore
	}

} // namespace USI


