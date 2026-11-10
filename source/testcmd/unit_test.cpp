#include "unit_test.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <variant>
#include "../benchmark.h"
#include "../position.h"
#include "../usi.h"
#include "../search.h"
#include "../misc.h"
#include "../book/book.h"
#include "../tt.h"

using namespace std;
namespace YaneuraOu {

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

	void UnitTester::run(std::function<void(UnitTester&, IEngine& )> f)
	{
		// 対象の関数を実行する前に呼び出されるcallback
		if (before_run)
			before_run();

		f(*this, *engine);

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

	void UnitTest(istringstream& is, IEngine& engine)
	{
		// UnitTestを呼び出してくれるclass。
		UnitTester tester;

		// 入力文字列を解釈
		string token;
		s64 random_player_loop = 0; // ランダムプレイヤーの対局回数(0を指定するとskip)
		while (is >> token)
		{
			if (token == "random_player_loop")
				is >> random_player_loop;
		}
		cout << "random_player_loop : " << random_player_loop << endl;

		// testerのoptionsに代入しておく。
		tester.options.add("random_player_loop", Option(random_player_loop));

		// --- 各classに対するUnitTest

		tester.engine = &engine;

		// perftの局面設定エラーが、エンジンのインターフェースを通って呼び出し側へ返ることを確認。
        tester.run([](UnitTester& tester, IEngine& engine) {
            auto section = tester.section("Perft");
            for (const auto& sfen : {std::string(), std::string("not-a-sfen"),
                                    std::string("9/9/9/9/9/9/9/9/9 x - 1"),
                                    std::string("9/9/9/9/9/9/9/9/9 b Z 1")})
            {
                auto result = engine.perft(sfen, 1);
                const auto* error = std::get_if<PositionSetError>(&result);
                tester.test("Invalid SFEN returns a diagnostic: " + sfen,
                            error && std::string(error->what()).find("Invalid SFEN") != std::string::npos);
            }

            auto depth1 = engine.perft(StartSFEN, 1);
            const auto* nodes1 = std::get_if<u64>(&depth1);
            tester.test("Start position depth 1 = 30", nodes1 && *nodes1 == 30);

            auto depth3 = engine.perft(StartSFEN, 3);
            const auto* nodes3 = std::get_if<u64>(&depth3);
            tester.test("Start position depth 3 = 25470", nodes3 && *nodes3 == 25470);
        });

        // ベンチマークの既定値と、不正なspeedtest引数の補正を確認する。
        tester.run([](UnitTester& tester, IEngine&) {
            auto section = tester.section("Benchmark");
            std::istringstream benchDefaults;
            auto commands = Benchmark::setup_bench(StartSFEN, benchDefaults);
            tester.test("bench defaults retain four 15-second positions",
                        commands.size() == 11
                          && commands[0] == "setoption name Threads value 1"
                          && commands[1] == "setoption name USI_Hash value 1024"
                          && commands[2] == "ucinewgame"
                          && std::count(commands.begin(), commands.end(), "go movetime 15000") == 4);

            std::istringstream pgoBench("1024 1 14 default depth");
            commands = Benchmark::setup_bench(StartSFEN, pgoBench);
            tester.test("PGO bench retains depth-limited searches",
                        std::count(commands.begin(), commands.end(), "go depth 14") == 4);

            for (const std::string input : {"32 1024 150", "0 0 0", "-1 -1 -1", "1 64 2147484",
                                           "9223372036854775807 9223372036854775807 9223372036854775807"})
            {
                std::istringstream args(input);
                const auto setup = Benchmark::setup_benchmark(args);
                const int maxSeconds = std::numeric_limits<int>::max() / 1000;
                const std::string expected = input == "32 1024 150" ? "32 1024 150"
                  : input == "0 0 0" || input == "-1 -1 -1" ? "1 1 1"
                  : input == "1 64 2147484" ? "1 64 " + std::to_string(maxSeconds)
                  : std::to_string(MaxThreads) + " " + std::to_string(MaxHashMB) + " " + std::to_string(maxSeconds);
                bool positiveTimes = true;
                for (const auto& command : setup.commands)
                    if (command.find("go movetime ") == 0)
                    {
                        std::istringstream time(command.substr(12));
                        i64 milliseconds = 0;
                        positiveTimes = positiveTimes && bool(time >> milliseconds) && milliseconds > 0;
                    }
                tester.test("speedtest accepts and bounds " + input.substr(0, 24),
                            setup.filledInvocation == expected && positiveTimes);
            }

            std::istringstream args("9223372036854775807");
            const auto setup = Benchmark::setup_benchmark(args);
            tester.test("speedtest default hash uses bounded threads",
                        setup.threads == MaxThreads && setup.ttSize == std::min(i64(128) * MaxThreads, i64(MaxHashMB)));
        });

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
		tester.run(USIEngine::UnitTest);

		// 指し手生成のテスト
		//tester.run(MoveGen::UnitTest)

	}

} // namespace Test;
} // namespace YaneuraOu;
