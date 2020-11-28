#include "../../config.h"

#if defined(EVAL_MATERIAL)

#include "../../types.h"
#include "../../position.h"
#include "../../evaluate.h"

// パラメーターの自動調整フレームワーク
#include "../../engine//yaneuraou-engine/yaneuraou-param-common.h"
#include <array>
#include <cmath>
#include <algorithm> // for std::min()
#include <numeric>	 // for std::accumulate()
#define SIZE_OF_ARRAY(array) (sizeof(array)/sizeof(array[0]))

namespace Eval
{
	// 駒得のみの評価関数のとき。
	void load_eval() {}
	void print_eval_stat(Position& pos) {}
	void evaluate_with_no_return(const Position& pos) {}
	Value evaluate(const Position& pos) { return compute_eval(pos); }

#if MATERIAL_LEVEL == 1 || !defined(MATERIAL_LEVEL)
	// 純粋な駒得の評価関数
	// これでも3990XだとR2300ぐらいある。
	// (序盤を定跡などでうまく乗り切れれば)

	void init() {}
	Value compute_eval(const Position& pos) {
		auto score = pos.state()->materialValue;
		ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));

		return pos.side_to_move() == BLACK ? score : -score;
	}

#elif MATERIAL_LEVEL == 2

	// 【連載】評価関数を作ってみよう！その1 : http://yaneuraou.yaneu.com/2020/11/17/make-evaluate-function/
	// 盤上の駒は、その価値の1/10ほど減点してやる評価関数。
	// これだけで+R50ぐらいになる。

	void init() {}
	Value compute_eval(const Position& pos) {
		auto score = pos.state()->materialValue;

		for (auto sq : SQ)
		{
			auto pc = pos.piece_on(sq);
			// この升に駒がなければ次の升へ
			if (pc == NO_PIECE)
				continue;

			// 駒の価値。
			// 後手の駒ならマイナスになるが、
			// いま計算しようとしているのは先手から見た評価値なので
			// これで辻褄が合う。
			auto piece_value = PieceValue[pc];
			score -= piece_value * 104 / 1024;
		}

		return pos.side_to_move() == BLACK ? score : -score;
	}

#elif MATERIAL_LEVEL == 3

	// 【連載】評価関数を作ってみよう！その2 : http://yaneuraou.yaneu.com/2020/11/19/make-evaluate-function-2/
	// 【連載】評価関数を作ってみよう！その3 : http://yaneuraou.yaneu.com/2020/11/20/make-evaluate-function-3/
	// 【連載】評価関数を作ってみよう！その4 : http://yaneuraou.yaneu.com/2020/11/23/make-evaluate-function-4/
	// 盤上の利きを評価する評価関数
	// これで、Lv.2からは+R200程度。

	//std::array<int, 9> our_effect_value  = { 1024, 496, 297, 272, 184, 166, 146, 116, 117};
	//std::array<int, 9> their_effect_value = { 1024, 504, 357, 320, 220, 194, 160, 136, 130};
	// ↑optimizerの回答

	// 王様からの距離に応じたある升の利きの価値。
	int our_effect_value[9];
	int their_effect_value[9];

	void init() {
		for (int i = 0; i < 9; ++i)
		{
			// 利きには、王様からの距離に反比例する価値がある。(と現段階では考えられる)
			our_effect_value[i]  = 68 * 1024 / (i + 1);
			their_effect_value[i] = 96 * 1024 / (i + 1);
		}
	}

	Value compute_eval(const Position& pos) {
		auto score = pos.state()->materialValue;

		for (auto sq : SQ)
		{
			// この升の先手の利きの数、後手の利きの数
			int effects[2] = { pos.board_effect[BLACK].effect(sq) , pos.board_effect[WHITE].effect(sq) };

			for (auto color : COLOR)
			{
				// color側の玉に対して
				auto king_sq = pos.king_square(color);

				// 筋と段でたくさん離れているほうの数をその距離とする。
				int d = dist(sq, king_sq);

				int s1 = effects[ color] * our_effect_value [d] / 1024;
				int s2 = effects[~color] * their_effect_value[d] / 1024;

				// scoreは先手から見たスコアなので、colorが先手の時は、(s1-s2) をscoreに加算。colorが後手の時は、(s2-s1) を加算。
				score += color == BLACK ? (s1 - s2) : (s2 - s1);
			}

			auto pc = pos.piece_on(sq);
			if (pc == NO_PIECE)
				continue;

			// 盤上の駒に対しては、その価値を1/10ほど減ずる。
			auto piece_value = PieceValue[pc];
			score -= piece_value * 104 / 1024;
		}

		return pos.side_to_move() == BLACK ? score : -score;
	}

#elif MATERIAL_LEVEL == 4

	// 【連載】評価関数を作ってみよう！その5 : http://yaneuraou.yaneu.com/2020/11/25/make-evaluate-function-5/
	// ・1つの升に複数の利きがある時にはその価値は指数関数的な減衰をする。(+R30)

	// 王様からの距離に応じたある升の利きの価値。
	int our_effect_value  [9];
	int their_effect_value[9];

	// 利きが一つの升に複数あるときの価値
	// 1024倍した値を格納する。
	// optimizerの答えは、{ 0 , 1024/* == 1.0 */ , 1800, 2300 , 2900,3500,3900,4300,4650,5000,5300 }
	//   6365 - pow(0.8525,m-1)*5341 　みたいな感じ？
	int multi_effect_value[11];

	void init() {

		for (int d = 0; d < 9; ++d)
		{
			// 利きには、王様からの距離に反比例する価値がある。
			our_effect_value[d]   = 85 * 1024 / (d + 1);
			their_effect_value[d] = 98 * 1024 / (d + 1);
		}

		// 利きがm個ある時に、our_effect_value(their_effect_value)の価値は何倍されるのか？
		// 利きは最大で10個のことがある。
		for (int m = 0; m < 11; ++m)
			multi_effect_value[m] = m == 0 ? 0 : int(6365 - std::pow(0.8525 , m - 1)*5341 );
	}

	Value compute_eval(const Position& pos) {
		auto score = pos.state()->materialValue;

		for (auto sq : SQ)
		{
			// この升の先手の利きの数、後手の利きの数
			int effects[2] = { pos.board_effect[BLACK].effect(sq) , pos.board_effect[WHITE].effect(sq) };

			// 盤上の升の利きの価値
			for (auto color : COLOR)
			{
				// color側の玉に対して
				auto king_sq = pos.king_square(color);

				// 筋と段でたくさん離れているほうの数をその距離とする。
				int d = dist(sq, king_sq);

				int s1 = multi_effect_value[effects[ color]] * our_effect_value  [d] / (1024 * 1024);
				int s2 = multi_effect_value[effects[~color]] * their_effect_value[d] / (1024 * 1024);

				// scoreは先手から見たスコアなので、colorが先手の時は、(s1-s2) をscoreに加算。colorが後手の時は、(s2-s1) を加算。
				score += color == BLACK ? (s1 - s2) : (s2 - s1);
			}

			auto pc = pos.piece_on(sq);
			if (pc == NO_PIECE)
				continue;

			// 盤上の駒に対しては、その価値を1/10ほど減ずる。
			auto piece_value = PieceValue[pc];
			score -= piece_value * 104 / 1024;
		}

		return pos.side_to_move() == BLACK ? score : -score;
	}

#elif MATERIAL_LEVEL == 5

	// 【連載】評価関数を作ってみよう！その6 : http://yaneuraou.yaneu.com/2020/11/26/make-evaluate-function-6/
	// 【連載】評価関数を作ってみよう！その7 : http://yaneuraou.yaneu.com/2020/11/27/make-evaluate-function-7/
	//
	//   テーブルを一つに (+R70)

	// 利きの価値を合算した値を求めるテーブル
	// [先手玉の升][後手玉の升][対象升][その升の先手の利きの数][その升の後手の利きの数]
	// 81*81*81*11*11*size_of(int16_t) = 128MB
	int16_t effect_table[SQ_NB][SQ_NB][SQ_NB][11][11];

	void init() {

		// 王様からの距離に応じたある升の利きの価値。

		int our_effect_value[9];
		int their_effect_value[9];

		for (int d = 0; d < 9; ++d)
		{
			// 利きには、王様からの距離に反比例する価値がある。
			our_effect_value[d]   = 83 * 1024  / (d + 1);
			their_effect_value[d] = 92 * 1024  / (d + 1);
		}

		// 利きが1つの升にm個ある時に、our_effect_value(their_effect_value)の価値は何倍されるのか？
		// 利きは最大で10個のことがある。格納されている値は1024を1.0とみなす固定小数。
		// optimizerの答えは、{ 0 , 1024/* == 1.0 */ , 1800, 2300 , 2900,3500,3900,4300,4650,5000,5300 }
		//   6365 - pow(0.8525,m-1)*5341 　みたいな感じ？

		int multi_effect_value[11];

		for (int m = 0; m < 11; ++m)
			multi_effect_value[m] = m == 0 ? 0 : int(6365 - std::pow(0.8525, m - 1) * 5341);

		// 利きを評価するテーブル
		//    [自玉の位置][対象となる升][利きの数(0～10)]
		int our_effect_table  [SQ_NB][SQ_NB][11];
		int their_effect_table[SQ_NB][SQ_NB][11];

		for(auto king_sq : SQ)
			for (auto sq : SQ)
				for(int m = 0 ; m < 11 ; ++m) // 利きの数
				{
					// 筋と段でたくさん離れているほうの数をその距離とする。
					int d = dist(sq, king_sq);

					our_effect_table  [king_sq][sq][m] = multi_effect_value[m] * our_effect_value  [d] / (1024 * 1024);
					their_effect_table[king_sq][sq][m] = multi_effect_value[m] * their_effect_value[d] / (1024 * 1024);
				}

		// ある升の利きの価値のテーブルの初期化
		for (auto king_black : SQ)
			for (auto king_white : SQ)
				for (auto sq : SQ)
					for(int m1 = 0; m1<11;++m1) // 先手の利きの数
						for (int m2 = 0; m2 < 11; ++m2) // 後手の利きの数
						{
							int score = 0;
							score += our_effect_table  [    king_black ][    sq ][m1];
							score -= their_effect_table[    king_black ][    sq ][m2];
							score -= our_effect_table  [Inv(king_white)][Inv(sq)][m2];
							score += their_effect_table[Inv(king_white)][Inv(sq)][m1];
							effect_table[king_black][king_white][sq][m1][m2] = int16_t(score);
						}

	}

	Value compute_eval(const Position& pos) {

		auto score = pos.state()->materialValue;

		for (auto sq : SQ)
		{
			// 盤上の升の利きの価値
			score += effect_table[pos.king_square(BLACK)][pos.king_square(WHITE)][sq][pos.board_effect[BLACK].effect(sq)][pos.board_effect[WHITE].effect(sq)];

			auto pc = pos.piece_on(sq);
			if (pc == NO_PIECE)
				continue;

			// 盤上の駒に対しては、その価値を1/10ほど減ずる。
			auto piece_value = PieceValue[pc];
			score -= piece_value * 104 / 1024;
		}

		return pos.side_to_move() == BLACK ? score : -score;
	}

#elif MATERIAL_LEVEL == 6

	// 【連載】評価関数を作ってみよう！その7 : http://yaneuraou.yaneu.com/2020/11/27/make-evaluate-function-7/
	// テーブルを1本化。KKPEE9テーブル導入。

	// KKPEEテーブル。Eが3通りなのでKKPEE9と呼ぶ。
	// 利きの価値を合算した値を求めるテーブル
	// [先手玉の升][後手玉の升][対象升][その升の先手の利きの数(最大2)][その升の後手の利きの数(最大2)][駒(先後区別あり)]
	// 81*81*81*3*3*size_of(int16_t)*32 = 306MB
	// 1つの升にある利きは、2つ以上の利きは同一視。
	int16_t KKPEE[SQ_NB][SQ_NB][SQ_NB][3][3][PIECE_NB];

	void init() {

		// 王様からの距離に応じたある升の利きの価値。

		int our_effect_value[9];
		int their_effect_value[9];

		for (int d = 0; d < 9; ++d)
		{
			// 利きには、王様からの距離に反比例する価値がある。
			our_effect_value[d]   = 83 * 1024  / (d + 1);
			their_effect_value[d] = 92 * 1024  / (d + 1);
		}

		// 利きが1つの升にm個ある時に、our_effect_value(their_effect_value)の価値は何倍されるのか？
		// 利きは最大で10個のことがある。格納されている値は1024を1.0とみなす固定小数。
		// optimizerの答えは、{ 0 , 1024/* == 1.0 */ , 1800, 2300 , 2900,3500,3900,4300,4650,5000,5300 }
		//   6365 - pow(0.8525,m-1)*5341 　みたいな感じ？

		int multi_effect_value[11];

		for (int m = 0; m < 11; ++m)
			multi_effect_value[m] = m == 0 ? 0 : int(6365 - std::pow(0.8525, m - 1) * 5341);

		// 利きを評価するテーブル
		//    [自玉の位置][対象となる升][利きの数(0～10)]
		double our_effect_table  [SQ_NB][SQ_NB][11];
		double their_effect_table[SQ_NB][SQ_NB][11];

		for(auto king_sq : SQ)
			for (auto sq : SQ)
				for(int m = 0 ; m < 3 ; ++m) // 利きの数
				{
					// 筋と段でたくさん離れているほうの数をその距離とする。
					int d = dist(sq, king_sq);

					our_effect_table  [king_sq][sq][m] = double(multi_effect_value[m] * our_effect_value  [d]) / (1024 * 1024);
					their_effect_table[king_sq][sq][m] = double(multi_effect_value[m] * their_effect_value[d]) / (1024 * 1024);
				}

		// ある升の利きの価値のテーブルの初期化
		for (auto king_black : SQ)
			for (auto king_white : SQ)
				for (auto sq : SQ)
					for(int m1 = 0; m1<3;++m1) // 先手の利きの数
						for (int m2 = 0; m2 <3; ++m2) // 後手の利きの数
							for(Piece pc = NO_PIECE;pc < PIECE_NB ; ++pc) // 駒(先後の区別あり)
							{
								double score = 0;

								score += our_effect_table  [    king_black ][    sq ][m1];
								score -= their_effect_table[    king_black ][    sq ][m2];
								score -= our_effect_table  [Inv(king_white)][Inv(sq)][m2];
								score += their_effect_table[Inv(king_white)][Inv(sq)][m1];

								if (pc != NO_PIECE)
								{
									// 盤上の駒に対しては、その価値を1/10ほど減ずる。
									auto piece_value = PieceValue[pc];
									score -= piece_value * 104 / 1024;
								}

								KKPEE[king_black][king_white][sq][m1][m2][pc] = int16_t(score);
							}

	}

	// KKPEE9 評価関数本体(わずか7行)
	// 変数名短くするなどすれば１ツイート(140文字)に収まる。
	Value compute_eval(const Position& pos) {

		auto score = pos.state()->materialValue;

		for (auto sq : SQ)
			score += KKPEE[pos.king_square(BLACK)][pos.king_square(WHITE)][sq]
				[std::min(int(pos.board_effect[BLACK].effect(sq)),2)][std::min(int(pos.board_effect[WHITE].effect(sq)),2)][pos.piece_on(sq)];

		return pos.side_to_move() == BLACK ? score : -score;
	}

#endif // MATERIAL_LEVEL

}

#endif // defined(EVAL_MATERIAL)

