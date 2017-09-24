#ifndef _EVALUATE_KPPT_H_
#define _EVALUATE_KPPT_H_

#include "../../shogi.h"

// KPPT評価関数で用いるheader

#if defined (EVAL_KPPT)

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
	typedef std::array<int32_t, 2> ValueKk;
	typedef std::array<int32_t, 2> ValueKkp;
	typedef std::array<int16_t, 2> ValueKpp;

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉や、駒落ちの盤面には対応出来ない。

	// 評価関数

	extern ValueKk(*kk_)[SQ_NB][SQ_NB];
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk (*kk_)
#define kkp (*kkp_)
#define kpp (*kpp_)

	// 配列のサイズ
	const u64 size_of_kk = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp;

}      // namespace Eval

#endif // defined (EVAL_KPPT)


#endif
