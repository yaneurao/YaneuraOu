#include "unit_test.h"

#include <iostream>
#include <sstream>
#include "../position.h"
#include "../usi.h"
#include "../search.h"
#include "../misc.h"
#include "../book/book.h"

using namespace std;

namespace Test
{
	// --------------------
	//      UnitTest
	// --------------------

	// UnitTest用。
	// "test unittest"コマンドで用いる。
	// 実際の使い方は、normal_test_cmd.cppのunit_test()とPosition::UnitTest()を見ること。

	UnitTester::UnitTester()
	{
		cout << "=== Start UnitTest ===" << endl;
	}
	UnitTester::~UnitTester()
	{
		cout << "=== Summary UnitTest ===" << endl;

		auto passed = test_count - errors.size();
		cout << "Summary : " << passed << " / " << test_count << " passed." << endl;

		if (errors.empty())
			cout << "-> Passed all UnitTests." << endl;
		else
		{
			cout << "Error List : " << endl;
			for (auto s : errors)
				cout << s << endl;
		}

		cout << "=== Finish UnitTest === " << endl;
	}

	// Test用のコマンド
	// 例) tester.test("局面の整合性テストその1",pos.is_ok());
	void UnitTester::test(const std::string& message, bool b)
	{
		stringstream ss;
		ss << section_name() << " " << message;
		auto left = ss.str();

		string right = b ? "passed" : "failed";

		// 横、80文字ぐらいで収まるように。
		auto padding = std::max((size_t)0, 79 - left.size() - right.size());

		string mes = left + string(padding, '.') + right;

		cout << mes << endl;

		// errorがあったなら、summeryで表示するために保存しておく。
		if (!b)
			errors.emplace_back(mes);

		++test_count;
	}

	void UnitTester::run(std::function<void(UnitTester&)> f)
	{
		// 対象の関数を実行する前に呼び出されるcallback
		if (before_run)
			before_run();

		f(*this);

		if (after_run)
			after_run();
	}

	// Sectionを作る。
	// 使い方は、Position::UnitTest()を見ること。
	const std::string UnitTester::section_name()
	{
		stringstream ss;

		ss << "[";
		bool first = true;
		for (auto section : sections)
		{
			if (!first)
				ss << "::";
			first = false;

			ss << section;
		}
		ss << "]";

		return ss.str();
	}

	// --------------------
	// UnitTest本体。"unittest"コマンドで呼び出される。
	// --------------------

	void UnitTest(Position& pos, istringstream& is)
	{
		// UnitTest開始時に"isready"コマンドを実行したのに相当する初期化はなされているものとする。
		is_ready();

		// UnitTestを呼び出してくれるclass。
		UnitTester tester;

		// 入力文字列を解釈
		string token;
		s64 random_player_loop = 1000; // ランダムプレイヤーの対局回数
		while (is >> token)
		{
			if (token == "random_player_loop")
			{
				is >> random_player_loop;
			}
		}
		cout << "random_player_loop : " << random_player_loop << endl;

		// testerのoptionsに代入しておく。
		tester.options["random_player_loop"] << USI::Option(random_player_loop, (s64)0, INT64_MAX );

		// --- run()の実行ごとに退避させていたものを元に戻す。

		// 退避させるもの
		auto limits_org = Search::Limits;
		tester.after_run = [&]() { Search::Limits = limits_org; };

		// --- 各classに対するUnitTest

		// Book namespace
		tester.run(Book::UnitTest);

		// Bitboard class
		tester.run(Bitboard::UnitTest);
		tester.run(Bitboard256::UnitTest);

		// Position class
		tester.run(Position::UnitTest);

		// USI namespace
		tester.run(USI::UnitTest);

		// Misc tools
		tester.run(Misc::UnitTest);

		// 指し手生成のテスト
		//tester.run(MoveGen::UnitTest)

	}

}

