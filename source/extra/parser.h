#ifndef EXTRA_PARSER_H_INCLUDED
#define EXTRA_PARSER_H_INCLUDED

#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../misc.h"

namespace YaneuraOu {

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

}

#endif
