#include "../config.h"

#if defined(ENABLE_TEST_CMD)

// ----------------------------------
//      通常のtestコマンド
// ----------------------------------

#include <fstream>
#include <iomanip>
#include <sstream>
#include "../position.h"
#include "../usi.h"
#include "../thread.h"
#include "../search.h"
#include "../movegen.h"
#include "../evaluate.h"

namespace YaneuraOu {
namespace {

	// "test genmoves" : 指し手生成テストコマンド
	// positionコマンドで設定されている現在の局面から。
	void gen_moves(IEngine& engine, std::istringstream& is)
	{
		auto& pos = engine.get_position();

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
				std::cout << Move(m) << ' ';
		else
			for (auto m : MoveList<NON_EVASIONS>(pos))
				std::cout << Move(m) << ' ';

		std::cout << std::endl << (1000 * loop / (end - start)) << " times per second." << std::endl;
	}

	// "test autoplay" : 自己対局用テストコマンド
	//   ASSERT_LV 5
	//   とかにしてビルドして、このコマンドで連続自己対局をすると探索や指し手生成にバグがあれば
	//	 何らか引っかかることが多いので開発の時に便利。
	//   goコマンド相当の処理で対局させているので通常の探索部であればどの探索部に対しても機能する。

	void auto_play(IEngine& engine, std::istringstream& is)
	{
		auto& options = engine.get_options();
		auto& threads = engine.get_threads();

		Position pos;

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

		// 読み筋の出力の抑制
		//global_options.silent = true;
		// TODO : あとで調整する。

		//lm.movetime = movetime; // これで時間固定の思考となる。
		// →　毎回同じ指し手になると嫌だから、自分で乱数を加えてばらつかせる

		// ふかうら王の場合、root mate searchが回っていると探索を打ち切らないので、ここで
		// 同じ思考時間になってしまう可能性がある。
		if (options.count("RootMateSearchNodesLimit"))
			options.set_option_if_exists("RootMateSearchNodesLimit",std::to_string(100)); // 100ノードに減らしておく。

		//if (Options.count("DNN_Batch_Size1"))
		//	Options["DNN_Batch_Size1"] = std::to_string(32); // これも減らしておかないとbatchsizeまでで時間がきてしまう。
		// →　このタイミングでやるとmodelのrebuildが起きるのか…。

		// isreadyが呼び出されたものとする。
		engine.isready();

		// 思考時間のランダム化のための乱数
		PRNG prng;

		for (uint64_t i = 0; i < loop ; ++i)
		{
			// 局面を遡るためのStateInfoのlist。
			StateListPtr states(new StateList(1));
			pos.set_hirate(&states->back());

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

				// Stockfish、start_thinking()のなかでstatesの所有権をstd::moveしてしまうので、コピーしておく。
				// ※　start_thinking()でownership transferするの、こういうコードが書きにくくなって良くないと思う。
				StateListPtr states0(new StateList(0));
				*states0.get() = *states.get(); // dequeのcopyを行う。

				// 思考時間、nodes_limit、1～4倍の間でランダム化
				Search::LimitsType lm;
                lm.startTime = now();
				lm.movetime = movetime * (1000 + prng.rand(3000)) / 1000;
				lm.nodes    = nodes_limit * (1000 + prng.rand(3000)) / 1000;

				threads.start_thinking(options, pos, states0 , lm);
				threads.wait_for_search_finished();
				auto& rootMoves = threads.main_thread()->worker->rootMoves;
				// ⚠ best threadから選出したほうがいいかも知れないが、どれがbest threadか、わからない…。

				ASSERT_LV3(rootMoves.size());

				Move m  = rootMoves.at(0).pv[0]; // 1番目に並び変わっているはず。
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

#if defined(YANEURAOU_ENGINE)
	// "test eval_accuracy <psv_path>" : 検証用 PSV ファイルに対し evaluate() を
	// 呼び、決着のついた局面 (= W/L) のみを対象に sign 一致率を計算する。
	//
	// 慣例:
	//   pred  = (evaluate(pos) >= 0)        側面から見て勝ち予測
	//   truth = (game_result   >  0)        STM が勝った
	//   match = (pred == truth)            game_result == 0 (引き分け) は両側から除外
	//
	// 引き分け除外の理由: dlshogi の旧 binary_accuracy は draw を win 側にバケット
	// していたが、dlshogi 作者が実際に検証に使っている局面集には引き分けが含まれて
	// いない。draw を含めると「model_output ≈ 0 を出すモデル」が draw + win に対し
	// 機械的に正解扱いされる構造的バイアス (= --scale を小さくして 0 に寄せると
	// 単調に accuracy が上がる現象の一因) を持つため、これを排し W vs L の純粋な
	// 符号一致率を測る。BulletOu 側の test_value_accuracy も同じ定義に揃えてある。
	//
	// YANEURAOU_ENGINE ガード: このコマンドは「engine.evaluate() が局面ごとに
	// 意味のある Value を返す」ことに依存する。これを満たすのは YaneuraOuEngine
	// を継承する NNUE / KPPT / KPP_KKPT / Material 系のみ (= YANEURAOU_ENGINE が
	// 定義される構成)。ふかうら王 (Deep)、tanuki/yaneuraou-mate、user-engine は
	// Engine::evaluate() の基底実装 (VALUE_NONE を返す) のままで accuracy 算出が
	// 無意味なため除外する。
	void eval_accuracy(IEngine& engine, std::istringstream& is)
	{
		std::string psv_path;
		is >> psv_path;

		if (psv_path.empty())
		{
			std::cout << "Error: usage: test eval_accuracy <psv_path>" << std::endl;
			return;
		}

		std::ifstream f(psv_path, std::ios::binary);
		if (!f)
		{
			std::cout << "Error: cannot open " << psv_path << std::endl;
			return;
		}

		// PsvRecord 固定長 (40 byte)。size が割り切れない = corrupted/truncated。
		f.seekg(0, std::ios::end);
		auto file_size = (uint64_t)f.tellg();
		f.seekg(0, std::ios::beg);
		if (file_size % sizeof(PsvRecord) != 0)
		{
			std::cout << "Error: file size " << file_size
					  << " is not a multiple of " << sizeof(PsvRecord)
					  << " byte (PsvRecord size) — possibly corrupted/truncated"
					  << std::endl;
			return;
		}
		const uint64_t total_records = file_size / sizeof(PsvRecord);

		std::cout << "test eval_accuracy: " << psv_path << std::endl
				  << "  records = " << total_records << std::endl;

		// nn.bin の load を保証 (= "isready" 相当処理を先に走らせる)
		engine.isready();

		auto& pos = engine.get_position();
		StateInfo si;

		uint64_t compared = 0;
		uint64_t sign_match = 0;
		uint64_t drawn = 0;
		uint64_t skipped = 0;

		auto start_time = now();
		auto last_report = start_time;

		PsvRecord rec;
		while (f.read(reinterpret_cast<char*>(&rec), sizeof(PsvRecord)))
		{
			if (pos.set_from_packed_sfen(rec.sfen, &si, false, rec.gamePly).is_not_ok())
			{
				skipped++;
				continue;
			}

			// 引き分けは accuracy 算出から除外。位置数のカウントだけ別途行う。
			if (rec.game_result == 0)
			{
				drawn++;
				continue;
			}

			// IEngine::evaluate() は内部で verify_networks() + Eval::evaluate(pos) を
			// 呼ぶ。NNUE accumulator は set_from_packed_sfen 後に invalid 状態に
			// なっているはずなので、Eval::evaluate 側で full refresh される (= 通常の
			// 探索開始時と同じ経路)。
			Value v = engine.evaluate();

			bool pred  = v >= VALUE_ZERO;
			bool truth = rec.game_result > 0;
			if (pred == truth) sign_match++;
			compared++;

			// 5 秒ごとに進捗
			auto cur = now();
			if (cur - last_report >= 5000)
			{
				last_report = cur;
				double pct = (total_records > 0)
					? 100.0 * (double)compared / (double)total_records
					: 0.0;
				std::cout << "  processed " << compared << " / " << total_records
						  << " (" << std::fixed << std::setprecision(1) << pct
						  << "%)" << std::endl;
			}
		}

		auto elapsed_ms = now() - start_time;
		double accuracy = (compared > 0)
			? (double)sign_match / (double)compared
			: 0.0;

		std::cout << "----" << std::endl;
		std::cout << "test eval_accuracy result:" << std::endl;
		std::cout << "  decisive   = " << compared
				  << " (skipped " << skipped << " malformed records)" << std::endl;
		std::cout << "  accuracy   = " << std::fixed << std::setprecision(4)
				  << (accuracy * 100.0) << "% (" << sign_match << "/" << compared
				  << " matches; W vs L sign agreement)" << std::endl;
		std::cout << "  drawn      = " << drawn
				  << " (excluded from accuracy)" << std::endl;
		std::cout << "  elapsed    = " << elapsed_ms << " ms ("
				  << (elapsed_ms > 0
					  ? ((compared + drawn) * 1000) / (uint64_t)elapsed_ms
					  : 0)
				  << " positions/sec)" << std::endl;
	}
#endif // defined(YANEURAOU_ENGINE)
} // namespace

// ----------------------------------
//      "test" command Decorator
// ----------------------------------

namespace Test
{
	// 通常のテストコマンド。コマンドを処理した時 trueが返る。
	bool normal_test_cmd(IEngine& engine , std::istringstream& is, const std::string& token)
	{
		if (token == "genmoves")              gen_moves(engine, is);       // 現在の局面に対して指し手生成のテストを行う。
		else if (token == "autoplay")         auto_play(engine, is);       // 連続自己対局を行う。
#if defined(YANEURAOU_ENGINE)
		else if (token == "eval_accuracy")    eval_accuracy(engine, is);   // PSV に対し evaluate() の sign 一致率を測る。
#endif
		else return false;									               // どのコマンドも処理することがなかった
			
		// いずれかのコマンドを処理した。
		return true;
	}

} // namespace Test

} // namespace YaneuraOu

#endif // defined(ENABLE_TEST_CMD)
