#include "progress.h"

#ifdef USE_PROGRESS

#include <fstream>
#include <omp.h>
#include <sstream>
#include <vector>
#include <ctime>

#include "../misc.h"
#include "../position.h"
#include "../search.h"
#include "../shogi.h"

#ifndef _MAX_PATH
#   define _MAX_PATH   260
#endif

using USI::Option;

namespace {
	const constexpr char* kProgressBookFile = "ProgressBookFile";
	const constexpr char* kProgressFilePath = "ProgressFilePath";
	const constexpr char* kProgressLearningRate = "ProgressLearningRate";
	const constexpr char* kProgressNumGamesForTesting = "ProgressNumGamesForTesting";
	const constexpr char* kProgressNumGamesForTraining = "ProgressNumGamesForTraining";
	const constexpr char* kProgressNumIterations = "ProgressNumIterations";
	const constexpr char* kThreads = "Threads";

	constexpr double kAdamBeta1 = 0.9;
	constexpr double kAdamBeta2 = 0.999;
	constexpr double kEps = 1e-8;
}

bool Progress::Initialize(USI::OptionsMap& o) {
	o[kProgressBookFile] << Option("wdoor.sfen");
	o[kProgressFilePath] << Option("progress.bin");
	o[kProgressNumIterations] << Option(1000, 0, INT_MAX);
	o[kProgressLearningRate] << Option("0.0002");
	o[kProgressNumGamesForTraining] << Option(34000, 0, INT_MAX);
	o[kProgressNumGamesForTesting] << Option(1000, 0, INT_MAX);
	return true;
}

bool Progress::Load() {
	std::string file_path = (std::string)Options[kProgressFilePath];
	std::ifstream ifs(file_path, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		sync_cout << "info string Failed to open the progress file. file_path=" << file_path <<
			sync_endl;
		return false;
	}

	if (!ifs.read(reinterpret_cast<char*>(weights_), sizeof(weights_))) {
		sync_cout << "info string Failed to read the progress file. file_path=" << file_path <<
			sync_endl;
		return false;
	}

	return true;
}

bool Progress::Save() {
	std::string file_path = (std::string)Options[kProgressFilePath];
	std::ofstream ofs(file_path, std::ios_base::out | std::ios_base::binary);
	if (!ofs) {
		sync_cout << "info string Failed to open the progress file. file_path=" << file_path <<
			sync_endl;
		return false;
	}

	if (!ofs.write(reinterpret_cast<char*>(weights_), sizeof(weights_))) {
		sync_cout << "info string Failed to write the progress file. file_path=" << file_path <<
			sync_endl;
		return false;
	}

	return true;
}

bool Progress::Learn() {
	int num_threads = (int)Options[kThreads];
	omp_set_num_threads(num_threads);

	// 棋譜を読み込む
	sync_cout << "Reading records..." << sync_endl;
	std::vector<std::vector<Move> > games;
	std::string book_file = (std::string)Options[kProgressBookFile];
	std::ifstream ifs(book_file);
	if (!ifs) {
		sync_cout << "info string Failed to read the progress book file." << sync_endl;
		return false;
	}

	std::string line;
	int index = 0;
	while (std::getline(ifs, line)) {
		std::istringstream iss(line);
		std::string move_str;
		Position pos;
		pos.set_hirate();
		std::vector<Move> game;
		auto state_stack = Search::StateStackPtr(new aligned_stack<StateInfo>);
		while (iss >> move_str) {
			if (move_str == "startpos" || move_str == "moves") {
				continue;
			}

			Move move = move_from_usi(pos, move_str);
			if (!is_ok(move) || !pos.legal(move)) {
				break;
			}

			state_stack->push(StateInfo());
			pos.do_move(move, state_stack->top());
			game.push_back(move);
		}

		if (pos.is_mated()) {
			games.push_back(game);
		}

		if (++index % 10000 == 0) {
			sync_cout << index << sync_endl;
		}
	}

	sync_cout << "num_records: " << games.size() << sync_endl;
	int num_games_for_training = (int)Options[kProgressNumGamesForTraining];
	int num_games_for_testing = (int)Options[kProgressNumGamesForTesting];
	if (games.size() < num_games_for_training + num_games_for_testing) {
		sync_cout << "games.size() < num_games_for_training + num_games_for_testing" << sync_endl;
		std::exit(1);
	}

	// 学習準備
	std::shuffle(games.begin(), games.end(), std::mt19937_64());
	std::vector<std::vector<Move> > games_for_training(games.begin(),
		games.begin() + num_games_for_training);
	std::vector<std::vector<Move> > games_for_testing(games.begin() + num_games_for_training,
		games.begin() + num_games_for_training + num_games_for_testing);
	int num_iterations = (int)Options[kProgressNumIterations];

	// Adam用変数
	int num_dimensions = SQ_NB * static_cast<int>(Eval::fe_end);
	std::vector<std::vector<double> > sum_gradients(num_threads, std::vector<double>(num_dimensions));
	std::vector<double> ws(num_dimensions);
	std::vector<double> ms(num_dimensions);
	std::vector<double> vs(num_dimensions);
	double learning_rate = 0.0;
	std::istringstream((std::string)Options[kProgressLearningRate]) >> learning_rate;

	char loss_file_name[_MAX_PATH];
	std::sprintf(loss_file_name, "learn_progress_loss_%I64d.learning_rate=%f.csv", std::time(nullptr),
		learning_rate);
	std::ofstream ofs_loss(loss_file_name);
	ofs_loss << "offset,rmse_train,rmse_test" << std::endl;

	// 学習開始
	for (int iteration = 0; iteration < num_iterations; ++iteration) {
		// 訓練データの処理
		double offset = 0.0;
		double sum_diff2_train = 0.0;
		int num_moves_in_train = 0;
#pragma omp parallel for reduction(+:offset, sum_diff2_train, num_moves_in_train) schedule(dynamic)
		for (int game_index = 0; game_index < num_games_for_training; ++game_index) {
			int thread_index = ::omp_get_thread_num();
			const auto& game = games_for_training[game_index];
			Position pos;
			pos.set_hirate();
			int num_moves = static_cast<int>(game.size());
			//sync_cout << "num_moves: " << num_moves << sync_endl;
			StateInfo state_infos[300] = { { 0 } };
			for (int move_index = 0; move_index < num_moves; ++move_index) {
				pos.do_move(game[move_index], state_infos[move_index]);

				double expected = move_index / static_cast<double>(num_moves - 1);
				double actual = Estimate(pos);
				double diff = actual - expected;
				offset += diff;
				sum_diff2_train += diff * diff;
				if (std::isnan(sum_diff2_train)) {
					sync_cout <<
						"game_index: " << game_index <<
						"move_index: " << move_index <<
						sync_endl;
					std::exit(1);
				}

				double g = diff * Math::dsigmoid(actual);
				Square sq_bk = pos.king_square(BLACK);
				Square sq_wk = Inv(pos.king_square(WHITE));
				const auto& list0 = pos.eval_list()->piece_list_fb();
				const auto& list1 = pos.eval_list()->piece_list_fw();
				for (int i = 0; i < PIECE_NO_KING; ++i) {
					int black_index = sq_bk * static_cast<int>(Eval::fe_end) + list0[i];
					sum_gradients[thread_index][black_index] += g;
					int white_index = sq_wk * static_cast<int>(Eval::fe_end) + list1[i];
					sum_gradients[thread_index][white_index] += g;
				}
				++num_moves_in_train;
			}
		}

		// テストデータの処理
		double sum_diff2_test = 0.0;
		int num_moves_in_test = 0;
#pragma omp parallel for reduction(+:sum_diff2_test, num_moves_in_test) schedule(dynamic)
		for (int game_index = 0; game_index < num_games_for_testing; ++game_index) {
			int thread_index = ::omp_get_thread_num();
			const auto& game = games_for_testing[game_index];
			Position pos;
			pos.set_hirate();
			int num_moves = static_cast<int>(game.size());
			StateInfo state_infos[300] = { { 0 } };
			for (int move_index = 0; move_index < num_moves; ++move_index) {
				pos.do_move(game[move_index], state_infos[move_index]);

				double expected = move_index / static_cast<double>(num_moves - 1);
				double actual = Estimate(pos);
				double diff = actual - expected;
				sum_diff2_test += diff * diff;
				++num_moves_in_test;
			}
		}

		// ロスの出力
		ofs_loss <<
			offset / num_moves_in_train << "," <<
			std::sqrt(sum_diff2_train / num_moves_in_train) << "," <<
			std::sqrt(sum_diff2_test / num_moves_in_test)
			<< std::endl << std::flush;

		// 重みの更新
		double adam_beta1_t = std::pow(kAdamBeta1, num_iterations + 1);
		double adam_beta2_t = std::pow(kAdamBeta2, num_iterations + 1);
#pragma omp parallel for schedule(dynamic)
		for (int dimension = 0; dimension < num_dimensions; ++dimension) {
			double g = 0.0;
			for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
				g += sum_gradients[thread_index][dimension];
				sum_gradients[thread_index][dimension] = 0;
			}

			double& w = ws[dimension];
			double& m = ms[dimension];
			double& v = vs[dimension];

			// Adam
			m = kAdamBeta1 * m + (1.0 - kAdamBeta1) * g;
			v = kAdamBeta2 * v + (1.0 - kAdamBeta2) * g * g;
			// 高速化のためpow(ADAM_BETA1, t)の値を保持しておく
			double mm = m / (1.0 - adam_beta1_t);
			double vv = v / (1.0 - adam_beta2_t);
			double delta = learning_rate * mm / (std::sqrt(vv) + kEps);
			w -= delta;

			// 重みテーブルに書き戻す
			Square square = static_cast<Square>(dimension / Eval::fe_end);
			Eval::BonaPiece piece = static_cast<Eval::BonaPiece>(dimension % Eval::fe_end);
			weights_[square][piece] = w;
		}

		sync_cout << iteration << sync_endl;
	}

	return true;
}

double Progress::Estimate(const Position& pos) {
	Square sq_bk = pos.king_square(BLACK);
	Square sq_wk = Inv(pos.king_square(WHITE));
	const auto& list0 = pos.eval_list()->piece_list_fb();
	const auto& list1 = pos.eval_list()->piece_list_fw();
	double sum = 0.0;
	for (int i = 0; i < PIECE_NO_KING; ++i) {
		sum += weights_[sq_bk][list0[i]];
		sum += weights_[sq_wk][list1[i]];
	}
	return Math::sigmoid(sum);
}

#endif
