#include "../config.h"

#if defined(ENABLE_TEST_CMD)

// ----------------------------------
//      通常のtestコマンド
// ----------------------------------

#include <sstream>
#include "../position.h"
#include "../usi.h"
#include "../thread.h"
#include "../search.h"

#if defined(EVAL_LEARN)
#include "../eval/evaluate_common.h"
#endif

namespace {

	// "test genmoves" : 指し手生成テストコマンド
	// positionコマンドで設定されている現在の局面から。
	void gen_moves(Position& pos, std::istringstream& is)
	{
		// 試行回数
		int64_t loop = 30000000;

		std::string token;
		while (is >> token)
		{
			if (token == "loop")
				is >> loop;
		}

		//	pos.set("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1");

		auto start = now();
		std::cout << "Generate Moves Test : " << std::endl
				  << "  loop = " << loop << std::endl;

		for (int i = 0; i < loop; ++i)
		{
			if (pos.checkers())
				MoveList<EVASIONS> ml(pos);
			else
				MoveList<NON_EVASIONS> ml(pos);
		}

		auto end = now();

		std::cout << "..done." << std::endl;

		// 局面と生成された指し手を出力しておく。
		std::cout << pos;

		if (pos.checkers())
			for (auto m : MoveList<EVASIONS>(pos))
				std::cout << m.move << ' ';
		else
			for (auto m : MoveList<NON_EVASIONS>(pos))
				std::cout << m.move << ' ';

		std::cout << std::endl << (1000 * loop / (end - start)) << " times per second." << std::endl;
	}

	// "test autoplay" : 自己対局用テストコマンド
	//   ASSERT_LV 5
	//   とかにしてビルドして、このコマンドで連続自己対局をすると探索や指し手生成にバグがあれば
	//	 何らか引っかかることが多いので開発の時に便利。
	//   goコマンド相当の処理で対局させているので通常の探索部であればどの探索部に対しても機能する。

	void auto_play(Position& pos, std::istringstream& is)
	{
		// 対局回数
		uint64_t loop = 50000; // default 5万回

		// 詳細の出力を行うのか(対局棋譜など)
		bool verbose = false;

		// 1手の思考長さ[ms]
		int movetime = 500;    // default 500ms

		// 1手のnodes limit
		s64 nodes_limit = 0;
		
		std::string token;
		while (is >> token)
		{
			if (token == "loop")
				is >> loop;
			else if (token == "verbose")
				verbose = true;
			else if (token == "movetime")
				is >> movetime;
			else if (token == "nodes")
				is >> nodes_limit;
		}

		std::cout   << "Auto Play test : " << std::endl
					<< "  loop     = " << loop << std::endl
					<< "  movetime = " << movetime << std::endl
					<< "  verbose  = " << verbose << std::endl
					<< "  nodes    = " << nodes_limit << std::endl
			;

		const int MAX_PLY = 256; // 256手までテスト

		auto start = now();

		Search::LimitsType lm;
		// 読み筋の出力の抑制
		lm.silent = true;

		//lm.movetime = movetime; // これで時間固定の思考となる。
		// →　毎回同じ指し手になると嫌だから、自分で乱数を加えてばらつかせる

		// ふかうら王の場合、root mate searchが回っていると探索を打ち切らないので、ここで
		// 同じ思考時間になってしまう可能性がある。
		if (Options.count("RootMateSearchNodesLimit"))
			Options["RootMateSearchNodesLimit"] = std::to_string(100); // 100ノードに減らしておく。

		//if (Options.count("DNN_Batch_Size1"))
		//	Options["DNN_Batch_Size1"] = std::to_string(32); // これも減らしておかないとbatchsizeまでで時間がきてしまう。
		// →　このタイミングでやるとmodelのrebuildが起きるのか…。

		// isreadyが呼び出されたものとする。
		Search::clear();

		// 思考時間のランダム化のための乱数
		PRNG prng;

		for (uint64_t i = 0; i < loop ; ++i)
		{

			// 局面を遡るためのStateInfoのlist。
			StateListPtr states(new StateList(1));
			pos.set_hirate(&states->back(),Threads.main());

			if (verbose)
				std::cout << "position startpos moves";

			char result = ' ';
			Color mateScoreColor = (Color)-1;

			for (int ply = 0;; ++ply)
			{
				if (!(ply < MAX_PLY))
				{
					result = 'M'; // MaxMoves
					break;
				}

				// 詰まされているかのチェック
				if (pos.is_mated())
				{
					result = (pos.side_to_move() == BLACK) ? 'W' : 'B'; // 勝ったほうのプレイヤーを出力
					break;
				}

				// 千日手のチェック
				auto rep = pos.is_repetition(16);
				if (rep == REPETITION_DRAW)
				{
					result = '.'; // 引き分け
					break;
				}

#if defined(USE_ENTERING_KING_WIN)
				// MateEngineなど宣言勝ちをサポートしていないエンジンもある…。
				
				// 宣言勝ちのチェック
				if (pos.DeclarationWin())
				{
					result = (pos.side_to_move() == BLACK) ? 'b' : 'w'; // 勝ったほうのプレイヤーを小文字で出力
					break;
				}
#endif

				Time.reset();

				// Stockfish、start_thinking()のなかでstatesの所有権をstd::moveしてしまうので、コピーしておく。
				// ※　start_thinking()でownership transferするの、こういうコードが書きにくくなって良くないと思う。
				StateListPtr states0(new StateList(0));
				*states0.get() = *states.get(); // dequeのcopyを行う。

				// 思考時間、nodes_limit、1～4倍の間でランダム化
				lm.movetime = movetime * (1000 + prng.rand(3000)) / 1000;
				lm.nodes    = nodes_limit * (1000 + prng.rand(3000)) / 1000;

				Threads.start_thinking(pos, states0 , lm);
				Threads.main()->wait_for_search_finished();
				auto rootMoves = Threads.main()->rootMoves;

				ASSERT_LV3(rootMoves.size());

				Move m = rootMoves.at(0).pv[0]; // 1番目に並び変わっているはず。
				Value v = rootMoves.at(0).score;

				if (mateScoreColor == pos.side_to_move() && v < VALUE_MATE)
				{
					// mate scoreをいったん出したのにそうでなくなった。
					std::cout << std::endl << "[Error!:MateColor]" << std::endl; // error!!
				}

				// MateScoreを出したなら、記録しておく。
				if (v >= VALUE_MATE)
				{
					mateScoreColor = pos.side_to_move();
				}

				// verboseモードならば対局棋譜(のsfen)を画面に出力する。
				if (verbose)
					std::cout << " " << m;

				states->emplace_back();
				pos.do_move(m, states->back());
			}

			if (verbose)
				std::cout << std::endl << "result = " << result << std::endl;
			else
				// 1局ごとに結果を出力(進んでいることがわかるように)
				std::cout << result;
		}
	}
}

// ----------------------------------
//      "test" command Decorator
// ----------------------------------

namespace Test
{
	// 通常のテストコマンド。コマンドを処理した時 trueが返る。
	bool normal_test_cmd(Position& pos , std::istringstream& is, const std::string& token)
	{
		if (token == "genmoves")         gen_moves(pos, is);       // 現在の局面に対して指し手生成のテストを行う。
		else if (token == "autoplay")    auto_play(pos, is);       // 連続自己対局を行う。
#if defined (EVAL_LEARN)
		else if (token == "evalsave")    Eval::save_eval("");      // 現在の評価関数のパラメーターをファイルに保存
#endif
		else return false;									       // どのコマンドも処理することがなかった
			
		// いずれかのコマンドを処理した。
		return true;
	}

}


#endif // defined(ENABLE_TEST_CMD)
