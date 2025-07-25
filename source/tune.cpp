#include "tune.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "usioption.h"

using std::string;

namespace YaneuraOu {

bool          Tune::update_on_last;
const Option* LastOption = nullptr;
OptionsMap*   Tune::options;
namespace {
std::map<std::string, int> TuneResults;

std::optional<std::string> on_tune(const Option& o) {

    if (!Tune::update_on_last || LastOption == &o)
        Tune::read_options();

    return std::nullopt;
}
}

// OptionMapsにname,value,rangeのオプション項目を追加する。
// 💡 init_option()から呼び出される。
void Tune::make_option(OptionsMap* opts, const string& n, int v, const SetRange& r) {

    // Do not generate option when there is nothing to tune (ie. min = max)
	// チューニング対象がない場合（例：min = max）にはオプションを生成しない
    if (r(v).first == r(v).second)
        return;

    if (TuneResults.count(n))
        v = TuneResults[n];

    opts->add(n, Option(v, r(v).first, r(v).second, on_tune));
    LastOption = &((*opts)[n]);

    // Print formatted parameters, ready to be copy-pasted in Fishtest
    // Fishtest にコピペ可能な形式でパラメータを整形して出力する

	/*
		📓
			オプション名 , パラメーターの値 , min , max , C_end , R_end

			C_end = (max-min)/20 のところは、チューニング終了時のstep size。
					現在のパラメーターの値からのstep size(これだけ±したパラメーターの値でテストする)
					📝 最初はもう少し大きな値から徐々に小さくしていく。

			R_end = 0.0020はチューニング終了時の学習率。勾配方向に値の動かす量の係数。500回で1移動する。
	*/

	std::cout << n << ","                                  //
              << v << ","                                  //
              << r(v).first << ","                         //
              << r(v).second << ","                        //
              << (r(v).second - r(v).first) / 20.0 << ","  //
              << "0.0020" << std::endl;
}

// namesというカンマ区切りの文字列から一つ要素を取り出す。
// "1, (2,3) , 4"のようになっている場合、呼び出すごとに"1", "2 3", "4"が返ってくる。
// pop : 取り出した部分の要素を消すフラグ。
string Tune::next(string& names, bool pop) {

    string name;

    do
    {
		// カンマを探して、そこまでをtokenに入れる。
        string token = names.substr(0, names.find(','));

		// pop == trueなら取り出した部分(先頭からカンマまで)を削除する。
        if (pop)
            names.erase(0, token.size() + 1);

        std::stringstream ws(token);
        name += (ws >> token, token);  // Remove trailing whitespace

    } while (std::count(name.begin(), name.end(), '(') - std::count(name.begin(), name.end(), ')'));
	// 残り文字列の ( と )が対応していないと駄目。(数が同じになるまで繰り返す。)

    return name;
}


template<>
void Tune::Entry<int>::init_option() {
	// OptionMapsにname,value,rangeのオプション項目を追加する。
    make_option(options, name, value, range);
}

template<>
void Tune::Entry<int>::read_option() {
	// OptionMapsの同名のオプション項目から値をvalueに設定する。
    if (options->count(name))
        value = int((*options)[name]);
}

// Instead of a variable here we have a PostUpdate function: just call it
// ここでは変数の代わりに PostUpdate 関数がある：単にそれを呼び出す
template<>
void Tune::Entry<Tune::PostUpdate>::init_option() {}

template<>
void Tune::Entry<Tune::PostUpdate>::read_option() {
    value();
}

}  // namespace YaneuraOu


// Init options with tuning session results instead of default values. Useful to
// get correct bench signature after a tuning session or to test tuned values.
// Just copy fishtest tuning results in a result.txt file and extract the
// values with:
//
// cat results.txt | sed 's/^param: \([^,]*\), best: \([^,]*\).*/  TuneResults["\1"] = int(round(\2));/'
//
// Then paste the output below, as the function body

// デフォルト値ではなく、チューニングセッションの結果でオプションを初期化する。
// チューニングセッション後に正しいベンチマーク署名を取得したり、
// 調整済みの値をテストする際に便利です。
// fishtest のチューニング結果を result.txt ファイルにコピーし、
// 以下のコマンドで値を抽出します:
//
// cat results.txt | sed 's/^param: \([^,]*\), best: \([^,]*\).*/  TuneResults["\1"] = int(round(\2));/'
//
// その出力を下記の関数本体に貼り付けてください。


namespace YaneuraOu {

void Tune::read_results() { /* ...insert your values here... */ }

}  // namespace Stockfish
