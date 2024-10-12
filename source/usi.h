#ifndef USI_H_INCLUDED
#define USI_H_INCLUDED

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>

#include "types.h"
#include "position.h"

// --------------------
//     USI関連
// --------------------

namespace USI
{
	// Normalizes the internal value as reported by evaluate or search
	// to the UCI centipawn result used in output. This value is derived from
	// the win_rate_model() such that Stockfish outputs an advantage of
	// "100 centipawns" for a position if the engine has a 50% probability to win
	// from this position in self-play at fishtest LTC time control.

	// evaluateまたはsearchによって報告される内部値をUSIの出力で使用されるUSIのcenti-pawnの値に正規化します
	// この値はwin_rate_model()から派生しており、
	// Stockfishがこのポジションから自己対局で50%の確率で勝利する場合、
	// "100セントポーン"の利点を出力します。
	// これは、fishtest LTCタイムコントロールでの自己対局においてです。

#if defined(USE_PIECE_VALUE)
	// → やねうら王の場合、PawnValue = 90なので Value = 90なら 100として出力する必要がある。
	// Stockfish 16ではこの値は328になっている。
	const int NormalizeToPawnValue = Eval::PawnValue;
#endif

	class Option;

	/// Define a custom comparator, because the UCI options should be case-insensitive
	// UCIではオプションはcase insensitive(大文字・小文字の区別をしない)なのでcustom comparatorを用意する。
	// USIではここがプロトコル上どうなっているのかはわからないが、同様の処理にしておく。
	struct CaseInsensitiveLess {
		bool operator() (const std::string&, const std::string&) const;
	};

	/// The options container is defined as a std::map
	// USIのoption名と、それに対応する設定内容を保持しているclass。実体はstd::map
	using OptionsMap = std::map<std::string, Option, CaseInsensitiveLess>;

	/// The Option class implements each option as specified by the UCI protocol
	// USIプロトコルで指定されるoptionの内容を保持するclass
	class Option {

		// USIプロトコルで"setoption"コマンドが送られてきたときに呼び出されるハンドラの型。
		//		typedef void(*OnChange)(const Option&);
		// Stockfishでは↑のように関数ポインタになっているが、
		// これだと[&](o){...}みたいなlambda式を受けられないのでここはstd::functionを使うべきだと思う。
		using OnChange = void (*)(const Option&);

	public:
		// (GUI側のエンジン設定画面に出てくる)ボタン
		Option(OnChange f = nullptr);
		
		// 文字列
		Option(const char* v, OnChange f = nullptr);
		
		// (GUI側のエンジン設定画面に出てくる)CheckBox。bool型のoption デフォルト値が v
		Option(bool v, OnChange f = nullptr);
		
		// (GUI側のエンジン設定画面に出てくる)SpinBox。s64型。
		// Stockfishではdouble型になっているけども、GUI側がdoubleを受け付けるようになっていない可能性があるし、
		// doubleだと仮数部が52bitしかないので64bitの値を指定できなくて嫌だというのもある。
		// ゆえに、doubleはサポートせずにs64のみを扱う。
		Option(s64 v, s64 minv, s64 maxv, OnChange = nullptr);
		
		// (GUI側のエンジン設定画面に出てくる)ComboBox。内容的には、string型と同等。
		// list = コンボボックスに表示する値。v = デフォルト値かつ現在の値
		// StockfishにはComboBoxの取扱いがないようなのだが、これは必要だと思うのでやねうら王では独自に追加する。
		Option(const std::vector<std::string>&list, const std::string& v, OnChange f = nullptr);

		// USIプロトコル経由で値を設定されたときにそれをcurrentValueに反映させる。
		Option& operator=(const std::string&);

		// 起動時に設定を代入する。
		void operator<<(const Option&);

		// s64型への暗黙の変換子。
		// Stockfishでは、intになっているが、やねうら王ではs64に拡張している。
		operator s64() const;

		// string型への暗黙の変換子
		// typeが"string"型のとき以外であっても何であれ変換できるようになっているほうが便利なので
		// 変換できるようにしておく。
		operator std::string() const;

		// case insensitiveにしないといけないので比較演算子は独自に用意する。
		bool operator==(const char*) const;

		// idxの値を変えずに上書きする。
		// ※　やねうら王、独自拡張。
		// コマンド文字列からOptionのインスタンスを構築する時にこの機能が必要となる。
		void overwrite(const Option&);

		// 既存のOptionの上書き。
		// min = max = default = param になる。
		void overwrite(const std::string& param);


	private:
		friend std::ostream& operator<<(std::ostream& os, const OptionsMap& om);

		std::string defaultValue, currentValue, type;

		// s64型のときの最小と最大
		// Stockfishではintになっているが、node limitなどs64の範囲の値を扱いたいのでやねうら王では拡張してある。
		s64 min, max;

		// 出力するときの順番。この順番に従ってGUIの設定ダイアログに反映されるので順番重要！
		size_t idx;

		// combo boxのときの表示する文字列リスト
		std::vector<std::string> list;

		// 値が変わったときに呼び出されるハンドラ
		OnChange on_change;
	};

	// optionのdefault値を設定する。
	void init(OptionsMap&);

	// USIメッセージ応答部(起動時に、各種初期化のあとに呼び出される)
	void loop(int argc, char* argv[]);

#if defined(USE_PIECE_VALUE)

	// Valueをcp(centi-pawn)に変換する。
	int to_cp(Value v);

	// cpからValueへ。⇑の逆変換。
	Value cp_to_value(int v);

	// USIプロトコルの形式でValue型を出力する。
	// 歩が100になるように正規化するので、operator <<(Value)をこういう仕様にすると
	// 実際の値と異なる表示になりデバッグがしにくくなるから、そうはしていない。
	// USE_PIECE_VALUEが定義されていない時は正規化しようがないのでこの関数は呼び出せない。
	std::string value(Value v);

#endif

	// Square型をUSI文字列に変換する
	std::string square(Square s);

	// 指し手をUSI文字列に変換する。
	std::string move(Move   m /*, bool chess960*/);
	std::string move(Move16 m /*, bool chess960*/);

	// 読み筋をUSI文字列化して返す。
	// " 7g7f 8c8d" のように返る。
	std::string move(const std::vector<Move>& moves);

	// 局面posとUSIプロトコルによる指し手を与えて
	// もし可能なら等価で合法な指し手を返す。
	// 合法でないときはMOVE_NONEを返す。(この時、エラーである旨を出力する。)
	// "resign"に対してはMOVE_RESIGNを返す。
	// Stockfishでは第二引数にconstがついていないが、これはつけておく。
	// 32bit Moveが返る。(Move16ではないことに注意)
	Move to_move(const Position& pos, const std::string& str);

	// -- 以下、やねうら王、独自拡張。

	// 合法かのテストはせずにともかく変換する版。
	// 返ってくるのは16bitのMoveなので、これを32bitのMoveに変換するには
	// Position::move16_to_move()を呼び出す必要がある。
	// Stockfishにはない関数だが、高速化を要求されるところで欲しいので追加する。
	Move16 to_move16(const std::string& str);

	// USIプロトコルで、idxの順番でoptionを出力する。(デバッグ用)
	std::ostream& operator<<(std::ostream& os, const OptionsMap& om);

	// USIに追加オプションを設定したいときは、この関数を定義すること。
	// USI::init()のなかからコールバックされる。
	void extra_option(USI::OptionsMap& o);

	// 評価関数を読み込んだかのフラグ。これはevaldirの変更にともなってfalseにする。
	extern bool load_eval_finished; // = false;

#if defined (USE_ENTERING_KING_WIN)
	// 入玉ルール文字列をEnteringKingRule型に変換する。
	EnteringKingRule to_entering_king_rule(const std::string& rule);
#endif

	// エンジンオプションをコンパイル時に設定する機能
	// "ENGINE_OPTIONS"で指定した内容を設定する。
	// 例) #define ENGINE_OPTIONS "FV_SCALE=24;BookFile=no_book"
	void set_engine_options(const std::string& options);

	// エンジンオプションのoverrideのためにファイルから設定を読み込む。
	// 1) これは起動時に"engine_options.txt"という設定ファイルを読み込むのに用いる。
	// 2) "isready"応答に対して、EvalDirのなかにある"eval_options.txt"という設定ファイルを読み込むのにも用いる。
	void read_engine_options(const std::string& filename);

	// namespace USI内のUnitTest。
	void UnitTest(Test::UnitTester& tester);
}

// USIのoption設定はここに保持されている。
extern USI::OptionsMap Options;

// === やねうら王独自実装 ===

// USIの"isready"コマンドが呼び出されたときの処理。このときに評価関数の読み込みなどを行なう。
// benchmarkコマンドのハンドラなどで"isready"が来ていないときに評価関数を読み込ませたいときに用いる。
// skipCorruptCheck == trueのときは評価関数の2度目の読み込みのときのcheck sumによるメモリ破損チェックを省略する。
// ※　この関数は、Stockfishにはないがないと不便なので追加しておく。
void is_ready(bool skipCorruptCheck = false);

// positionコマンドのparserを呼び出したいことがあるので外部から呼び出せるようにしておく。
// 使い方はbenchコマンド(benchmark.cpp)のコードを見てほしい。
void position_cmd(Position& pos, std::istringstream& is, StateListPtr& states);

// エンジン本体
// TODO : あとでengine.hに移動させる。
class USIEngine
{
public:
	USIEngine(int argc, char** argv) :
		cli(argc, argv) {
	}

	CommandLine cli; // TODO : あとでUSIEngineに移動させる。
};


#endif // #ifndef USI_H_INCLUDED
