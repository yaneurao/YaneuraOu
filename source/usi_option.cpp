#include "thread.h"
#include "tt.h"
#include "usi.h"
#include "misc.h"

using std::string;

// Option設定が格納されたglobal object。
USI::OptionsMap Options;

namespace USI {

	// --- やねうら王独自拡張分の前方宣言

	// 入玉ルールのUSI文字列
	std::vector<std::string> ekr_rules = { "NoEnteringKing", "CSARule24" , "CSARule27" , "TryRule" };

	void read_engine_options();

	// USIプロトコルで必要とされるcase insensitiveな less()関数
	bool CaseInsensitiveLess::operator() (const string& s1, const string& s2) const {

		return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
			[](char c1, char c2) { return tolower(c1) < tolower(c2); });
	}

	// 前回のOptions["EvalDir"]
	std::string last_eval_dir;

	// optionのdefault値を設定する。
	void init(OptionsMap& o)
	{
		// Hash上限。32bitモードなら2GB、64bitモードなら33TB
		constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;

		// 並列探索するときのスレッド数
		// CPUの搭載コア数をデフォルトとすべきかも知れないが余計なお世話のような気もするのでしていない。

#if !defined(YANEURAOU_ENGINE_DEEP)

		// ※　やねうら王独自改良
		// スレッド数の変更やUSI_Hashのメモリ確保をそのハンドラでやってしまうと、
		// そのあとThreadIdOffsetや、LargePageEnableを送られても困ることになる。
		// ゆえにこれらは、"isready"に対する応答で行うことにする。
		// そもそもで言うとsetoptionに対してそんなに時間のかかることをするとGUI側がtimeoutになる懸念もある。
		// Stockfishもこうすべきだと思う。

		o["Threads"] << Option(4, 1, 512, [](const Option& o) { /* Threads.set(o); */ });
#endif

#if !defined(TANUKI_MATE_ENGINE) && !defined(YANEURAOU_MATE_ENGINE)
		// 置換表のサイズ。[MB]で指定。
		o["USI_Hash"] << Option(16, 1, MaxHashMB, [](const Option&o) { /* TT.resize(o); */ });

	#if defined(USE_EVAL_HASH)
		// 評価値用のcacheサイズ。[MB]で指定。

		#if defined(FOR_TOURNAMENT)
		// トーナメント用は少し大きなサイズ
		o["EvalHash"] << Option(1024, 1, MaxHashMB, [](const Option& o) { Eval::EvalHash_Resize(o); });
		#else
		o["EvalHash"] << Option(128, 1, MaxHashMB, [](const Option& o) { Eval::EvalHash_Resize(o); });
		#endif // defined(FOR_TOURNAMENT)
	#endif // defined(USE_EVAL_HASH)

		o["USI_Ponder"] << Option(false);

		// その局面での上位N個の候補手を調べる機能
		o["MultiPV"] << Option(1, 1, 800);

		// 指し手がGUIに届くまでの時間。
	#if defined(YANEURAOU_ENGINE_DEEP)
			// GPUからの結果を待っている時間も込みなので少し上げておく。
			int time_margin = 400;
	#else
			int time_margin = 120;
	#endif

		// ネットワークの平均遅延時間[ms]
		// この時間だけ早めに指せばだいたい間に合う。
		// 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
		o["NetworkDelay"] << Option(time_margin, 0, 10000);

		// ネットワークの最大遅延時間[ms]
		// 切れ負けの瞬間だけはこの時間だけ早めに指す。
		// 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
		o["NetworkDelay2"] << Option(time_margin + 1000, 0, 10000);

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

		// 評価関数フォルダ。これを変更したとき、評価関数を次のisreadyタイミングで読み直す必要がある。
		last_eval_dir = "eval";
		o["EvalDir"] << Option("eval", [](const USI::Option&o) {
			if (last_eval_dir != string(o))
			{
				// 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
				last_eval_dir = string(o);
				load_eval_finished = false;
			}
		});

#else
		
		// TANUKI_MATE_ENGINEのとき
		o["USI_Hash"] << Option(4096, 1, MaxHashMB);

#endif // !defined(TANUKI_MATE_ENGINE) && !defined(YANEURAOU_MATE_ENGINE)

		// cin/coutの入出力をファイルにリダイレクトする
		o["WriteDebugLog"] << Option(false, [](const Option& o) { start_logger(o); });


#if defined (USE_ENTERING_KING_WIN)
		// 入玉ルール
		o["EnteringKingRule"] << Option(USI::ekr_rules, USI::ekr_rules[EKR_27_POINT]);
#endif

#if defined(USE_GENERATE_ALL_LEGAL_MOVES)
		// 読みの各局面ですべての合法手を生成する
		// (普通、歩の2段目での不成などは指し手自体を生成しないのですが、これのせいで不成が必要な詰みが絡む問題が解けないことが
		// あるので、このオプションを用意しました。トーナメントモードではこのオプションは無効化されます。)
		o["GenerateAllLegalMoves"] << Option(false);
#endif


#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32) && \
	 (defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) )
		// 評価関数パラメーターを共有するか。
		// デフォルトで有効に変更。(V4.90～)
		o["EvalShare"] << Option(true);
#endif

#if defined(EVAL_LEARN)
		// isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
		// test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
		// このコマンドの実行前に異常終了してしまう。
		// そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
		// test evalconvertコマンドを叩く。
		o["SkipLoadingEval"] << Option(false);
#endif

#if defined(_WIN32)
		// 3990XのようなWindows上で複数のプロセッサグループを持つCPUで、思考エンジンを同時起動したときに
		// 同じプロセッサグループに割り当てられてしまうのを避けるために、スレッドオフセットを
		// 指定できるようにしておく。
		// 例) 128スレッドあって、4つ思考エンジンを起動してそれぞれにThreads = 32を指定する場合、
		// それぞれの思考エンジンにはThreadIdOffset = 0,32,64,96をそれぞれ指定する。
		// (プロセッサグループは64論理コアごとに1つ作られる。上のケースでは、ThreadIdOffset = 0,0,64,64でも同じ意味。)
		//	※　1つのPCで複数の思考エンジンを同時に起動して対局させる場合はこれを適切に設定すべき。

		o["ThreadIdOffset"] << Option(0, 0, std::thread::hardware_concurrency() - 1);
#endif

#if defined(_WIN64)
		// LargePageを有効化するか。
		// これを無効化できないと自己対局の時に片側のエンジンだけがLargePageを使うことがあり、
		// 不公平になるため、無効化する方法が必要であった。
		o["LargePageEnable"] << Option(true);
#endif

		// 各エンジンがOptionを追加したいだろうから、コールバックする。
		USI::extra_option(o);

		// カレントフォルダに"engine_options.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
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
		//ASSERT_LV1(type == "string" || type == "combo" /* 将棋用拡張*/ );
		// →　string化して保存しておいた内容をあとで復元したいことがあるのでこのassertないほうがいい。
		// 代入しないとハンドラが起動しないので、そういう復元の仕方をしたいことがある。(ベンチマークなどで)
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

	// 文字列に対応するEnteringKingRuleを取得する。
	EnteringKingRule to_entering_king_rule(const std::string& rule)
	{
		for (size_t i = 0; i < ekr_rules.size(); ++i)
			if (ekr_rules[i] == rule)
				return (EnteringKingRule)i;

		ASSERT(false);
		return EnteringKingRule::EKR_NONE;
	}
#endif

	// 思考エンジンがGUIからの"usi"に対して返す"option ..."文字列から
	// Optionオブジェクトを構築して、それをOptions[]に突っ込む。
	// "engine_options.txt"というファイルの各行からOptionオブジェクト構築して
	// Options[]の値を上書きするためにこの関数が必要。
	// "option name USI_Hash type spin default 256"
	// のような文字列が引数として渡される。
	// このとき、Optionのhandlerとidxは書き換えない。
	void build_option(string line)
	{
		LineScanner scanner(line);
		if (scanner.get_text() != "option") return;

		string name, value, option_type;
		int64_t min_value = 0, max_value = 1;
		std::vector<string> combo_list;
		while (!scanner.eol())
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

	// カレントフォルダに"engine_options.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
	void read_engine_options()
	{
		std::ifstream ifs("engine_options.txt");
		if (!ifs.fail())
		{
			std::string str;
			while (Tools::getline(ifs, str))
				build_option(str);
		}
	}

	// idxの値を書き換えないoperator "<<"
	void Option::overwrite(const Option& o)
	{
		// 値が書き換わるのか？
		bool modified = this->currentValue != o.currentValue;

		// backup
		auto fn = this->on_change;
		auto idx_ = idx;

		*this = o;

		// restore
		idx = idx_;
		this->on_change = fn;

		// 値が書き換わったならハンドラを呼び出してやる。
		if (modified && fn)
			fn(*this);
	}

} // namespace USI


