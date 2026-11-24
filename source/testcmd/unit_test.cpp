#include "unit_test.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <variant>
#include "../attacks.h"
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
		// 長いテスト名では、符号なしの減算がアンダーフローしないよう先に幅を比較する。
		const auto width = left.size() + right.size();
		const auto padding = width < 79 ? 79 - width : 0;

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

using namespace BB_Table;
using namespace Attacks;

// UnitTest
void Bitboard::UnitTest(Test::UnitTester& tester, IEngine& engine)
{
	//Bitboard b(SQ_75);
	//std::cout << lanceEffect<WHITE>(SQ_71, b) << std::endl;
	//std::cout << lanceEffect<WHITE>(SQ_72, b) << std::endl;

	auto section1 = tester.section("Bitboard");
#if defined(ENABLE_TEST_CMD)
    Attacks::test_sliding_attacks(tester);
#endif

	{
		// SQの升のbitを立てて、それがちゃんと読み取れるかのテスト
		bool all_ok = true;
		for (Rank r = RANK_1; r <= RANK_9; ++r)
			for (File f = FILE_1; f <= FILE_9; ++f)
			{
				Square sq = f | r;
				Bitboard b = square_bb(sq);
				all_ok &= b.test(sq);    // そのbitが立っているか
				all_ok &= b.pop_lsb() == sq; // 1の立っているbitをもらえば、その升であるか。
			}
		tester.test("sq occupied", all_ok);
	}
	{
		// ByteReverseがちゃんと機能しているかのテスト

		Bitboard b(0x0123456789abcdef, 0xfedcba9876543210);
		Bitboard r = b.byte_reverse();

		tester.test("byte_reverse", r.extract64<0>() == 0x1032547698badcfe && r.extract64<1>() == 0xefcdab8967452301);
	}
	{
		// 9段目が0、そこ以外が1のmask(香の利きを求めるコードのなかで使っている)
		Bitboard mask(0x3fdfeff7fbfdfeffULL , 0x000000000001feffULL);

		tester.test("Rank9BB", Rank9BB == ~mask);
	}

	{
		// 駒の利き

		bool all_ok = true;

		// 何も駒のない盤面上に駒ptを55に置いた時の利きの数。
		int p0_table[] = {
			0,1,4,2,5,16,16,6,	// Empty、歩、香、桂、銀、角、飛、金
			8,6,6,6,6,20,20,6,	// 玉、と、…
		};
		// 駒が敷き詰められた盤面上で駒ptを55に置いた時の利きの数。
		int p1_table[] = {
			0,1,1,2,5, 4, 4,6,	// Empty、歩、香、桂、銀、角、飛
			8,6,6,6,6, 8, 8,6,	// 玉、と、…
		};

		for (Color c : COLOR)
			for (PieceType pt = PAWN ; pt < PIECE_TYPE_NB ; ++pt )
			{
				Piece pc = make_piece(c, pt);
				Bitboard bb0 = attacks_bb(pc, SQ_55, Bitboard(ZERO));
				int p0 = bb0.popcount();
				Bitboard bb1 = attacks_bb(pc, SQ_55, Bitboard(1));
				int p1 = bb1.popcount();
				bool ok0 = (p0 == p0_table[(int)pt]);
				if (!ok0)
				{
					std::cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(ZERO) , popcount = " << p0 << std::endl;
					std::cout << bb0 << std::endl;
				}
				bool ok1 = (p1 == p1_table[(int)pt]);
				if (!ok1)
				{
					std::cout << "Effect " << pc  << "(" << pretty(pc) << ")" << " on SQ_55 in Bitboard(1) , popcount = " << p1 << std::endl;
					std::cout << bb1 << std::endl;
				}
				all_ok &= ok0 & ok1;
			}
		tester.test("attacks_bb", all_ok);
	}

	{
		// 方向利き

		bool all_ok = true;
		Bitboard occ = square_bb(SQ_77);
		Bitboard zero(ZERO);
		all_ok &= rayEffect<Effect8::DIRECT_LD>(SQ_55, occ) == between_bb(SQ_55, SQ_88);
		all_ok &= rayEffect<Effect8::DIRECT_LD>(SQ_55, zero) == QUGIY_STEP_EFFECT[Effect8::DIRECT_LD - 2][SQ_55];

		Bitboard occ2 = square_bb(SQ_33);
		all_ok &= rayEffect<Effect8::DIRECT_RU>(SQ_55, occ2) == between_bb(SQ_55, SQ_22);

		tester.test("rayEffect", all_ok);
	}

	{
		// SSE unpack

		const Bitboard bb_0000(0, 0);
		const Bitboard bb_ff00((u64)0, (u64)-1LL); // Bitboard(low64,high64)と書くようになっている。
		const Bitboard bb_00ff((u64)-1LL, (u64)0);
		const Bitboard bb_ffff((u64)-1LL, (u64)-1LL);

		bool all_ok = true;
		Bitboard hi_out, lo_out;

		Bitboard::unpack(bb_ffff, bb_0000, hi_out, lo_out);

		all_ok &= hi_out == bb_ff00;
		all_ok &= lo_out == bb_ff00;

		Bitboard::unpack(bb_ff00, bb_ff00, hi_out, lo_out);

		all_ok &= hi_out == bb_ffff;
		all_ok &= lo_out == bb_0000;

		Bitboard::unpack(bb_ff00, bb_00ff, hi_out, lo_out);

		all_ok &= hi_out == bb_ff00;
		all_ok &= lo_out == bb_00ff;

		tester.test("unpack", all_ok );
	}

	{
		// Bitboard::decrement() method

		const Bitboard all_bb((u64)-1LL, (u64)-1LL);
		const Bitboard all_bb_minus_one((u64)-2LL, (u64)-2LL);
		const Bitboard zero_bb(0, 0);

		bool all_ok = true;
		Bitboard hi_out, lo_out;

		// これなら桁借りは生じない。
		Bitboard::decrement(zero_bb, all_bb, hi_out, lo_out);

		all_ok &= hi_out == zero_bb;
		all_ok &= lo_out == all_bb_minus_one;

		// これは桁借りが生じる。
		Bitboard::decrement(all_bb, zero_bb, hi_out, lo_out);

		all_ok &= hi_out == all_bb_minus_one;
		all_ok &= lo_out == all_bb;

		tester.test("decrement(static function)", all_ok );
	}
	{
		const Bitboard zero_bb(0, 0);
		const Bitboard minus_one((u64)-1LL, (u64)-1LL);
		const Bitboard one_bb(1, 0);
		bool all_ok = true;

		all_ok &= zero_bb.decrement() == minus_one;
		all_ok &= one_bb.decrement()  == zero_bb;

		tester.test("decrement(method)", all_ok);
	}
	{
		// pawn_attacks_bbのテスト
		tester.test("pawn_attacks_bb<BLACK>",pawn_attacks_bb<BLACK>(Rank7BB) == Rank6BB);
		tester.test("pawn_attacks_bb<WHITE>",pawn_attacks_bb<WHITE>(Rank7BB) == Rank8BB);

		tester.test("attacks_bb(B_KNIGHT)",attacks_bb(B_KNIGHT, SQ_77) == (square_bb(SQ_85) | square_bb(SQ_65)));
		tester.test("attacks_bb(W_KNIGHT)",attacks_bb(W_KNIGHT, SQ_77) == (square_bb(SQ_89) | square_bb(SQ_69)));
	}
	{
		bool all_ok = true;
		Bitboard bb;

		bb = square_bb(SQ_22) | square_bb(SQ_77);
		all_ok &= least_significant_square_bb(bb) == square_bb(SQ_22);
		bb = square_bb(SQ_99) | square_bb(SQ_87);
		all_ok &= least_significant_square_bb(bb) == square_bb(SQ_87);

		tester.test("least_significant_square_bb",all_ok);
	}
}

namespace Attacks {

#if defined(ENABLE_TEST_CMD)
// Verify attacks by walking board coordinates, independently of sliding-attack masks.
// 長い利きのマスクに依存せず、盤上を1升ずつ進む実装と利きを照合する。
void test_sliding_attacks(Test::UnitTester& tester) {
    auto section = tester.section("Sliding attacks");
    enum Axis { FileAxis, RankAxis, Diagonal, AntiDiagonal };
    const auto ray = [](Square sq, int df, int dr, const Bitboard& occupied) {
        Bitboard result(ZERO);
        int f = int(file_of(sq)) + df, r = int(rank_of(sq)) + dr;
        for (; 0 <= f && f < 9 && 0 <= r && r < 9; f += df, r += dr)
        {
            const Square to = YaneuraOu::File(f) | YaneuraOu::Rank(r);
            result |= square_bb(to);
            if (occupied.test(to)) break;
        }
        return result;
    };
    constexpr int df[4] = {0, 1, 1, 1}, dr[4] = {1, 0, 1, -1};
    constexpr int rdf[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    constexpr int rdr[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const auto reference = [&](Square sq, int axis, const Bitboard& occupied) {
        return ray(sq, df[axis], dr[axis], occupied)
             | ray(sq, -df[axis], -dr[axis], occupied);
    };
    PRNG random(20261123);
    u64 cases = 0;
    for (Square sq : SQ)
        for (int axis = 0; axis < 4; ++axis)
        {
            const Bitboard relevant = reference(sq, axis, Bitboard(ZERO));
            std::vector<Square> squares;
            relevant.foreach([&](Square to) { squares.push_back(to); });
            for (unsigned pattern = 0; pattern < (1u << squares.size()); ++pattern)
            {
                // Include noise off the line and in unused bits; vary the source independently.
                // 線の外や未使用bitにもノイズを置き、移動元の占有も独立に切り替える。
                Bitboard occupied = relevant.andnot(Bitboard(random.rand<u64>(), random.rand<u64>()));
                occupied = square_bb(sq).andnot(occupied);
                for (usize i = 0; i < squares.size(); ++i)
                    if (pattern & (1u << i)) occupied |= square_bb(squares[i]);
                for (int sourceOccupied = 0; sourceOccupied < 2; ++sourceOccupied)
                {
                    if (sourceOccupied) occupied |= square_bb(sq);
                    const auto fileAttack = reference(sq, FileAxis, occupied);
                    const auto rankAttack = reference(sq, RankAxis, occupied);
                    const auto bishopAttack = reference(sq, Diagonal, occupied)
                                            | reference(sq, AntiDiagonal, occupied);
                    bool ok = bishopEffect(sq, occupied) == bishopAttack
                           && rookEffect(sq, occupied) == (fileAttack | rankAttack)
                           && rookFileEffect(sq, occupied) == fileAttack
                           && rookRankEffect(sq, occupied) == rankAttack
                           && lanceEffect<BLACK>(sq, occupied) == ray(sq, 0, -1, occupied)
                           && lanceEffect<WHITE>(sq, occupied) == ray(sq, 0, 1, occupied);
                    for (int d = 0; d < 8; ++d)
                        ok &= directEffect(sq, Effect8::Direct(d), occupied)
                           == ray(sq, rdf[d], rdr[d], occupied);
                    if (!ok)
                    {
                        std::cout << "Sliding attack mismatch: square=" << int(sq) << " axis=" << axis
                                  << " pattern=" << pattern << " source=" << sourceOccupied << '\n';
                        tester.test("exhaustive line occupancy", false);
                        return;
                    }
                    ++cases;
                }
            }
        }
    std::cout << "Sliding attacks checked " << cases << " occupancy cases" << std::endl;
    tester.test("exhaustive line occupancy, SIMD and directional rays", true);
    bool sentinel = true;
    for (Bitboard occupied : {Bitboard(ZERO), Bitboard(1), Bitboard(~u64(0), ~u64(0))})
    {
        sentinel &= !rookEffect(SQ_NB, occupied) && !bishopEffect(SQ_NB, occupied)
                 && !lanceEffect<BLACK>(SQ_NB, occupied) && !lanceEffect<WHITE>(SQ_NB, occupied);
        for (int d = 0; d < 8; ++d)
            sentinel &= !directEffect(SQ_NB, Effect8::Direct(d), occupied);
    }
    tester.test("SQ_NB sentinel", sentinel);
}
#endif

} // namespace Attacks

} // namespace YaneuraOu;
