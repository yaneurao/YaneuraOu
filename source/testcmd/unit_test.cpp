#include "unit_test.h"

#include <iostream>
#include <sstream>
#include "../position.h"
#include "../usi.h"
#include "../search.h"
#include "../misc.h"
#include "../book/book.h"

using namespace std;

#if defined(YANEURAOU_ENGINE) && defined (EVAL_LEARN)
namespace Learner {
	void UnitTest(Test::UnitTester& unittest);
}
#endif

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

	// コマンド例
	//	unittest
	//	→　通常のUnitTest
	//  unittest random_player_loop 1000
	//  →　ランダムプレイヤーでの自己対局1000回を行うUnitTest
	//  unittest auto_player_loop 1000 auto_player_depth 6
	//  →　探索深さ6での自己対局を1000回行うUnitTest。(やねうら王探索部 + EVAL_LEARN版が必要)

	void UnitTest([[maybe_unused]] Position& pos, istringstream& is)
	{
		// UnitTest開始時に"isready"コマンドを実行したのに相当する初期化はなされているものとする。
		is_ready();

		// UnitTestを呼び出してくれるclass。
		UnitTester tester;

		// 入力文字列を解釈
		string token;
		s64 random_player_loop = 0; // ランダムプレイヤーの対局回数(0を指定するとskip)
		s64 auto_player_loop   = 0; // 自己対局の対局回数(0を指定するとskip)
		s64 auto_player_depth  = 6; // 自己対局の時のdepth
		while (is >> token)
		{
			if (token == "random_player_loop")
				is >> random_player_loop;
			else if (token == "auto_player_loop")
				is >> auto_player_loop;
			else if (token == "auto_player_depth")
				is >> auto_player_depth;
		}
		cout << "random_player_loop : " << random_player_loop << endl;
		cout << "auto_player_loop   : " << auto_player_loop   << endl;
		cout << "auto_player_depth  : " << auto_player_depth  << endl;

		// testerのoptionsに代入しておく。
		tester.options["random_player_loop"] << USI::Option(random_player_loop, (s64)0, INT64_MAX );
		tester.options["auto_player_loop"  ] << USI::Option(auto_player_loop  , (s64)0, INT64_MAX );
		tester.options["auto_player_depth" ] << USI::Option(auto_player_depth , (s64)0, INT64_MAX );

		// --- run()の実行ごとに退避させていたものを元に戻す。

		// 退避させるもの
		auto limits_org = Search::Limits;
		tester.after_run = [&]() { Search::Limits = limits_org; };

		// ConsiderationModeをオフにしておかないとPV出力の時に置換表を漁るのでその時にdo_move()をして
		// 探索ノード数が加算されてしまい、depth固定のbenchなのに探索ノード数や読み筋が変化することがある。
		// (これが変化されてしまうと再現性がなくなってしまい、デバッグする時に都合が悪い。)
		Search::Limits.consideration_mode = false;

		// --- 各classに対するUnitTest

#if defined(YANEURAOU_ENGINE) && defined(EVAL_LEARN)
		// 自己対局のテスト(これはデバッガで追いかけたいことがあるので、他のをすっ飛ばして最初にやって欲しい)
		tester.run(Learner::UnitTest);
#endif

		// Book namespace
		tester.run(Book::UnitTest);

		// Bitboard class
		tester.run(Bitboard::UnitTest);
		tester.run(Bitboard256::UnitTest);

		// Position class
		tester.run(Position::UnitTest);

		// Transposition Table
		tester.run(TranspositionTable::UnitTest);

		// USI namespace
		tester.run(USI::UnitTest);

		// Misc tools
		tester.run(Misc::UnitTest);

		// 指し手生成のテスト
		//tester.run(MoveGen::UnitTest)

	}

}

