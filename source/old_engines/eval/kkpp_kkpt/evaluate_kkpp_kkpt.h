#ifndef _EVALUATE_KKPP_KKPT_H_
#define _EVALUATE_KKPP_KKPT_H_

#include "../../shogi.h"

// KKPP_KKPT型評価関数で用いるheader

#if defined(EVAL_KKPP_KKPT)

#include "../../evaluate.h"
#include "../evaluate_common.h"

namespace Eval
{

	// -----------------------------
	//    評価関数パラメーターの型
	// -----------------------------

	// 手番込みの評価値。[0]が手番に無縁な部分。[1]が手番があるときの上乗せ
	//  (これは先手から見たものではなく先後に依存しないボーナス)。
	// 先手から見て、先手の手番があるときの評価値 =  [0] + [1]
	// 先手から見て、先手の手番がないときの評価値 =  [0] - [1]
	// 後手から見て、後手の手番があるときの評価値 = -[0] + [1]

	// KPPTのときとは異なり、KPPが手番なしになっている点が異なる。
	// KK、KKPは16bitでも事足りそうではあるが、ちょっと危ないし、
	// KPPTのKKとKKPのファイルは使いまわしたいので互換性重視でこうしておく。
	
	typedef std::array<int32_t, 2> ValueKk;
	typedef std::array<int32_t, 2> ValueKkp;
	typedef int16_t ValueKpp;
	typedef int16_t ValueKkpp;

	// KKPPの計算対象とする王の段(3段ぐらいを想定している。9段のときはkpp配列自体が要らないが、それは想定していない)
	// この計算対象にない場合は、普通のKPPテーブルを用いて計算する。
	static const Rank KKPP_KING_RANK = (Rank)(RANK_9 - EVAL_KKPP_KKPT / 9 + 1);



	// 学習配列のKKPP対象の先手玉の升の数(ミラーありなので5/9倍)
	static const int KKPP_LEARN_BK_SQ = 5 * (EVAL_KKPP_KKPT / 9);

	// 学習配列のKKPP対象の後手玉の升の数
	static const int KKPP_LEARN_WK_SQ = 9 * (EVAL_KKPP_KKPT / 9);

	// ミラーありにしても、後手玉のほうはミラーできないので、5/9になるだけ。
	// piece_list()の更新処理が難しくなるので、ミラー対応は後回しにする。

	static const int KKPP_EVAL_BK_SQ = 9 * (EVAL_KKPP_KKPT / 9);
	static const int KKPP_EVAL_WK_SQ = 9 * (EVAL_KKPP_KKPT / 9);

	// 先手玉×後手玉の組み合わせの数(学習配列のKKPPに用いる)
	static const int KKPP_LEARN_KING_SQ = KKPP_LEARN_BK_SQ * KKPP_LEARN_WK_SQ;

	// eval用の正方配列に用いる玉の組み合わせの数
	static const int KKPP_EVAL_KING_SQ = KKPP_EVAL_BK_SQ * KKPP_EVAL_WK_SQ;

	// KKPP配列に三角配列を用いる場合の、PPの部分の要素の数。
	//const u64 kkpp_triangle_fe_end = fe_end * (fe_end - 1) / 2;
	static const u64 kkpp_square_fe_end = fe_end * fe_end;
	
	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉には対応出来ない。

	// 評価関数

	extern ValueKk(*kk_)[SQ_NB][SQ_NB];
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	//extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];
	//extern ValueKkpp(*kkpp_)[kkpp_king_sq_king_sq][fe_end][fe_end];

	extern ValueKpp(*kpp_);
	extern ValueKkpp(*kkpp_);
	
	// 2次元配列で添字がでかいやつ(16GBを超えると)Visual C++2017で C2036(サイズが不明です)のコンパイルエラーになる。
	// 詳しい説明は、evaluate_kpppt.hを見よ。

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk   (*kk_  )
#define kkp  (*kkp_ )
#define kpp  ( kpp_ )
#define kkpp ( kkpp_)

	// eval用の配列のサイズ
	const u64 size_of_kk = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_kkpp = (u64)KKPP_EVAL_KING_SQ*(u64)kkpp_square_fe_end*(u64)sizeof(ValueKkpp);
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp + size_of_kkpp;

	// eval用のkpp配列の位置を返すマクロ
	// i,j = piece0,piece1
	static ValueKpp& kpp_ksq_pcpc(int king_, BonaPiece i_, BonaPiece j_) {
		return *(kpp_ + (u64)king_ * kkpp_square_fe_end + u64(i_)*fe_end + u64(j_));
	}

	// eval用のkkpp配列の位置を返すマクロ
	// i,j = piece0,piece1
	// encoded_eval_kk = encode_to_eval_kk()でencodeした先手玉、後手玉の升。
	static ValueKkpp& kkpp_ksq_pcpc(int encoded_eval_kk, BonaPiece i_, BonaPiece j_) {
		return *(kkpp_ + (u64)encoded_eval_kk * kkpp_square_fe_end + u64(i_)*fe_end + u64(j_));
	}

	// 先手玉の位置、後手玉の位置を引数に渡して、
	// それをkkpp配列の第一パラメーターとして使えるように符号化する。
	// kkppの対象升でない場合は、-1が返る。
	// USE_KKPP_KKPT_MIRRORがdefineされてるときはミラーを考慮した処理が必要なのだが、未実装。
	static int encode_to_eval_kk(Square bk, Square wk)
	{
		// KKPPの対象の段でないなら無視
		if (rank_of(bk) < KKPP_KING_RANK)
			return -1;

		// 同様に後手玉側も。
		auto inv_wk = Inv(wk);
		if (rank_of(inv_wk) < KKPP_KING_RANK)
			return -1;

		// encode

		int k0 = ((int)rank_of(bk    ) - (int)KKPP_KING_RANK) * 9 + (int)file_of(    bk);
		int k1 = ((int)rank_of(inv_wk) - (int)KKPP_KING_RANK) * 9 + (int)file_of(inv_wk);

		int k = k0 * KKPP_EVAL_WK_SQ + k1;
		ASSERT_LV3(k < KKPP_EVAL_KING_SQ);

		return k;
	}
	
	// encode_to_eval_kk()の逆変換
	static void decode_from_eval_kk(int encoded_eval_kk, Square& bk, Square& wk)
	{
		ASSERT_LV3(0 <= encoded_eval_kk && encoded_eval_kk < KKPP_EVAL_KING_SQ);

		// KKPP_EVAL_WK_SQ進数表現だとみなしてKKPP_WK_SQ進数変換の逆変換を行なう。
		int k1 = encoded_eval_kk % KKPP_EVAL_WK_SQ;
		int k0 = encoded_eval_kk / KKPP_EVAL_WK_SQ;

		bk            = ((File)(k0 % 9)) | ((Rank)((k0 / 9) + KKPP_KING_RANK));
		Square inv_wk = ((File)(k1 % 9)) | ((Rank)((k1 / 9) + KKPP_KING_RANK));

		wk = Inv(inv_wk);
	}

	// KKPPの学習配列で用いるencoded_kk
	static int encode_to_learn_kk(Square bk, Square wk)
	{
		if (file_of(bk) > FILE_5)
			return -1;
		if (rank_of(bk) < KKPP_KING_RANK)
			return -1;

		auto inv_wk = Inv(wk);

		if (rank_of(inv_wk) < KKPP_KING_RANK)
			return -1;

		// encode

		int k0 = ((int)rank_of(bk    ) - (int)KKPP_KING_RANK) * 5 + (int)file_of(    bk);
		int k1 = ((int)rank_of(inv_wk) - (int)KKPP_KING_RANK) * 9 + (int)file_of(inv_wk);

		int k = k0 * KKPP_LEARN_WK_SQ + k1;
		ASSERT_LV3(k < KKPP_LEARN_KING_SQ);

		return k;
	}

	// encode_to_learn_kk()の逆変換
	static void decode_from_learn_kk(int encoded_learn_kk , Square& bk, Square &wk)
	{
		ASSERT_LV3(0 <= encoded_learn_kk && encoded_learn_kk < KKPP_LEARN_KING_SQ);

		// KKPP_LEARN_WK_SQ進数表記だとみなしてKKPP_WK_SQ進数変換の逆変換を行なう。
		int k1 = encoded_learn_kk % KKPP_LEARN_WK_SQ;
		int k0 = encoded_learn_kk / KKPP_LEARN_WK_SQ;

		bk            = ((File)(k0 % 5)) | ((Rank)((k0 / 5) + KKPP_KING_RANK));
		Square inv_wk = ((File)(k1 % 9)) | ((Rank)((k1 / 9) + KKPP_KING_RANK));
		wk = Inv(inv_wk);
	}

	// 上記の変換と逆変換関数のためのUnit Test的なもの。
	static void encode_and_decode_test()
	{
		std::cout << "encode_kk_test..";

		for (auto sq_bk : SQ)
			for (auto sq_wk : SQ)
			{
				int encoded_eval_kk = encode_to_eval_kk(sq_bk, sq_wk);
				if (encoded_eval_kk != -1)
				{
					Square sq_bk2, sq_wk2;
					decode_from_eval_kk(encoded_eval_kk, sq_bk2, sq_wk2);
					ASSERT_LV1(sq_bk == sq_bk2 && sq_wk == sq_wk2);

					// encode_to_learn_kk()を呼び出すとき、sq_bkは1～5筋にいなければならない。
					if (file_of(sq_bk) > FILE_5)
					{
						sq_bk = Mir(sq_bk);
						sq_wk = Mir(sq_wk);
					}

					int encoded_learn_kk = encode_to_learn_kk(sq_bk, sq_wk);
					ASSERT_LV1(encoded_learn_kk != -1);

					decode_from_learn_kk(encoded_learn_kk, sq_bk2, sq_wk2);
					ASSERT_LV1(sq_bk == sq_bk2 && sq_wk == sq_wk2);
				}
			}

		std::cout << "done." << std::endl;
	}

}      // namespace Eval

#endif // defined(EVAL_KPP_KKPT)


#endif
