#ifndef _EVALUATE_KPP_PPT_H_
#define _EVALUATE_KPP_PPT_H_

#include "../shogi.h"

// KPP_PPT型評価関数で用いる共用header的なもの。

#if defined(EVAL_KPP_PPT)

#include "../evaluate.h"

// PPファイル名
#define PP_BIN "PP_synthesized.bin"

// KKPファイル名
#define KKP_BIN "KKP_synthesized.bin"

// KPPファイル名
#define KPP_BIN "KPP_synthesized.bin"

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

	// KPPTのときとは異なり、KKPが16bit化されている点とKPPが手番なしになっている点が少し異なる。
	// 16bitにしていてもぎりぎりオーバーフローしないはず。

	typedef std::array<int16_t, 2> ValuePp;
	typedef std::array<int16_t, 2> ValueKkp;
	typedef int16_t ValueKpp;

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉には対応出来ない。

	// PPは、PがKのときはKKPに含まれると考えられるので入れないことにする。

	// 評価関数

	extern ValuePp(*pp_)[fe_end][fe_end];
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define pp (*pp_)
#define kkp (*kkp_)
#define kpp (*kpp_)

	// 配列のサイズ
	const u64 size_of_pp = (u64)fe_end*(u64)fe_end*(u64)sizeof(ValuePp);
	const u64 size_of_kkp = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_eval = size_of_pp + size_of_kkp + size_of_kpp;

}      // namespace Eval

#endif // defined(EVAL_KPP_PPT)


#endif
