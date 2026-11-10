#include "parser.h"

#include <cstdlib>

namespace YaneuraOu {

// --------------------
//       Parser
// --------------------

namespace Parser
{

	/*
		LineScanner parser("AAA BBB CCC DDD");
		auto token = parser.peek_text();
		cout << token << endl;
		token = parser.get_text();
		cout << token << endl;
		token = parser.get_text();
		cout << token << endl;
		token = parser.get_text();
		cout << token << endl;
		token = parser.get_text();
		cout << token << endl;
	*/

	// 次のtokenを先読みして返す。get_token()するまで解析位置は進まない。
	std::string LineScanner::peek_text()
	{
		// 二重にpeek_text()を呼び出すのは合法であるものとする。
		if (!token.empty())
			return token;

		// assert(token.empty());

		// 解析開始位置から連続するスペースは読み飛ばす。
		while (!raw_eol())
		{
			char c = line[pos];
			if (c != ' ')
				break;
			pos++;
		}

		while (!raw_eol())
		{
			// スペースに遭遇するまで。
			char c = line[pos++];
			if (c == ' ')
				break;
			token += c;
		}

		// 次の文字先頭まで解析位置を進めておく。
		while (!raw_eol())
		{
			char c = line[pos];
			if (c != ' ')
				break;
			pos++;
		}

		return token;
	}

	// 次のtokenを返す。
	std::string LineScanner::get_text()
	{
		auto result = (!token.empty() ? token : peek_text());
		token.clear();
		return result;
	}

	// 現在のcursor位置から残りの文字列を取得する。
	// peek_text()した分があるなら、それも先頭にくっつけて返す。
	std::string LineScanner::get_rest()
	{
		return token.empty()
			? line.substr(pos)
			: token + " " + line.substr(pos);
	}

	// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
	i64 LineScanner::get_number(i64 defaultValue)
	{
		std::string token = get_text();
		return token.empty() ? defaultValue : atoll(token.c_str());
	}

	// 次の文字列を数値化して返す。数値化できない時は引数の値がそのまま返る。
	double LineScanner::get_double(double defaultValue)
	{
		std::string token = get_text();
		return token.empty() ? defaultValue : std::atof(token.c_str());
	}

}

}
