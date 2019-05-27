// 古いperftのコード
// Stockfish 9で探索部に追加されたのでそちらに倣うことにした。

// ----------------------------------
//  USI拡張コマンド "perft"(パフォーマンステスト)
// ----------------------------------

// perft()で用いるsolver
// cf. http://qiita.com/ak11/items/8bd5f2bb0f5b014143c8

// 通常のPERFTと、置換表を用いる高速なPERFTと選択できる。
// 後者を用いる場合は、hash keyの衝突を避けるためにHASH_KEY_BITSを128にしておくこと。
// ※　あと、後者は以下のところで置換表を15GBほど固定で確保しているので、動作環境に応じて修正すること。
// >  entryCount = 256 * 1024 * 1024; // * sizeof(PerftSolverResult) == 15GBほど

#define NORMAL_PERFT

// perftのときにeval値も加算していくモード。評価関数のテスト用。
//#define EVAL_PERFT


struct PerftSolverResult {
	uint64_t nodes, captures, promotions, checks, mates;
#ifdef EVAL_PERFT
	int64_t eval;
#endif

	void operator+=(const PerftSolverResult& other) {
		nodes += other.nodes;
		captures += other.captures;
		promotions += other.promotions;
		checks += other.checks;
		mates += other.mates;
#ifdef EVAL_PERFT
		eval += other.eval;
#endif
	}
};

#ifndef NORMAL_PERFT
// perftで用いる置換表
namespace Perft {
	struct TTEntry {
		void save(HASH_KEY key_, PerftSolverResult& result_)
		{
			key = key_;
			result = result_;
		}
		HASH_KEY key;
		PerftSolverResult result;
	};

	struct TranspositionTable {
		TranspositionTable() {
			entryCount = 256*1024*1024; // * sizeof(PerftSolverResult) == 15GBほど
			table = (TTEntry*)calloc( entryCount * sizeof(TTEntry) , 1);
		}
		~TranspositionTable() { free(table); }

		TTEntry* probe(const HASH_KEY key_,int depth,bool& found)
		{
			auto key = key_ ^ DepthHash(depth); // depthの分だけhash keyを変更しておく。
			auto& tte = table[key /*.p(0)*/ & (entryCount - 1)];
			found = (tte.key == key_); // 変更前のhash keyが書かれているはず
			return &tte;
		}
	private:
		TTEntry* table;
		size_t entryCount; // TTEntryの数
	};
	TranspositionTable TT;
}

#endif

#ifdef NORMAL_PERFT
struct PerftSolver {
	template <bool Root>
	PerftSolverResult Perft(Position& pos, int depth) {
		StateInfo st;
		PerftSolverResult result = {};
		if (depth == 0) {
			result.nodes++;
			if (pos.captured_piece() != NO_PIECE) result.captures++;
#if defined (KEEP_LAST_MOVE)
			if (is_promote(pos.state()->lastMove)) result.promotions++;
#endif
#if defined (EVAL_PERFT)
//			cout << pos.sfen() << " , eval = " << Eval::evaluate(pos) << endl;
			/*
			if (pos.sfen() == "1nsgkgsnl/lr5b1/pppppp+Bpp/9/9/2P6/PP1PPPPPP/7R1/LNSGKGSNL w P 4")
			{
//				cout << Eval::evaluate(pos);
				Eval::print_eval_stat(pos);
			}
			*/
			result.eval += Eval::evaluate(pos);
#endif
			if (pos.checkers()) {
				result.checks++;
				if (pos.is_mated()) result.mates++;
			}
		} else {
			for (auto m : MoveList<LEGAL_ALL>(pos)) {
				if (Root)
					cout << ".";
				pos.do_move(m, st);
				result += Perft<false>(pos, depth - 1);
				pos.undo_move(m);
			}
		}
		return result;
	}
};
#else // 置換表を用いる高速なperft
struct PerftSolver {
	template <bool Root>
	PerftSolverResult Perft(Position& pos, int depth) {
		HASH_KEY key = pos.state()->long_key();
		bool found;

		PerftSolverResult result = {};

		auto tt = Perft::TT.probe(key, depth, found); // 置換表に登録されているか。
		if (found)
			return tt->result;

		StateInfo st;
		for (auto m : MoveList<LEGAL_ALL>(pos)) {
			if (Root)
				cout << ".";

			pos.do_move(m.move, st);
			if (depth > 1)
				result += Perft<false>(pos, depth - 1);
			else {
				result.nodes++;
				if (pos.state()->capturedType != NO_PIECE) result.captures++;
				#if defined (KEEP_LAST_MOVE)
				if (is_promote(pos.state()->lastMove)) result.promotions++;
				#endif
				if (pos.checkers()) {
					result.checks++;
					if (pos.is_mated()) result.mates++;
				}
			}
			pos.undo_move(m.move);
		}
		tt->save(key, result); // 登録されていなかったので置換表に保存する

		return result;
	}
};
#endif

// N手で到達できる局面数を計算する。成る手、取る手、詰んだ局面がどれくらい含まれているかも計算する。
void perft(Position& pos, istringstream& is)
{
	int depth = 5 ;
	is >> depth;
	cout << "perft depth = " << depth << endl;
	PerftSolver solver;
	// 局面コピーして並列的に呼び出してやるだけであとはなんとかなる。

	auto result = solver.Perft<true>(pos, depth);

	cout << endl << "nodes = " << result.nodes << " , captures = " << result.captures <<
#if defined (KEEP_LAST_MOVE)
		" , promotion = " << result.promotions <<
#endif
#if defined (EVAL_PERFT)
		" , eval(sum) = " << result.eval <<
#endif
		" , checks = " << result.checks << " , checkmates = " << result.mates << endl;
}



	// perft
	{
		// 最多合法手局面
		const string POS593 = "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1";
		cout << "> genmove sfen = " << POS593;
		pos.set(POS593,&si,th);
		auto mg = MoveList<LEGAL_ALL>(pos);
		cout << " , moves = " << mg.size();
		check( mg.size() == 593);

		cout << "> perft depth 6 ";
		pos.set_hirate(&si,th);
		auto result = PerftSolver().Perft<true>(pos,6);
		check(  result.nodes == 547581517 && result.captures == 3387051
#ifdef      KEEP_LAST_MOVE
			&& result.promotions == 1588324
#endif
			&& result.checks == 1730177 && result.mates == 0);
	}

