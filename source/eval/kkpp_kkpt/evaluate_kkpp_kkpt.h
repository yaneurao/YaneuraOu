#ifndef _EVALUATE_KPP_KKPT_H_
#define _EVALUATE_KPP_KKPT_H_

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

	// ミラーありなので..
	static const int KKPP_KING_SQ = 5 * (EVAL_KKPP_KKPT/9) * 9 * (EVAL_KKPP_KKPT/9);

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

	// 配列のサイズ
	const u64 size_of_kk = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_kkpp = (u64)KKPP_KING_SQ*(u64)kkpp_square_fe_end*(u64)sizeof(ValueKkpp);
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp + size_of_kkpp;

	// kppの配列の位置を返すマクロ
	// i,j = piece0,piece1
	static ValueKpp& kpp_ksq_pcpc(int king_, BonaPiece i_, BonaPiece j_) {
		return *(kpp_ + (u64)king_ * kkpp_square_fe_end + u64(i_)*fe_end + u64(j_));
	}

	// kpppの配列の位置を返すマクロ
	// i,j = piece0,piece1
	static ValueKkpp& kkpp_ksq_pcpc(int king_, BonaPiece i_, BonaPiece j_) {
		return *(kpp_ + (u64)king_ * kkpp_square_fe_end + u64(i_)*fe_end + u64(j_));
	}

}      // namespace Eval

#endif // defined(EVAL_KPP_KKPT)


#endif
