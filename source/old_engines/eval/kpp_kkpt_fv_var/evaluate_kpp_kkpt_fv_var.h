#ifndef _EVALUATE_KPP_KKPT_FV_VAR_H_
#define _EVALUATE_KPP_KKPT_FV_VAR_H_

// KPP_KKPTのUSE_FV_VAR(可変長eval_list)を用いるリファレンス実装実装。
// KPP_KKPT型に比べてわずかに遅いが、拡張性が非常に高く、極めて美しい実装なので、
// 今後、評価関数の拡張は、これをベースにやっていくことになると思う。

#include "../../shogi.h"

// KPP_KKPT型

#if defined(EVAL_KPP_KKPT_FV_VAR)

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

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉には対応出来ない。

	// 評価関数

	extern ValueKk(*kk_)[SQ_NB][SQ_NB];
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	//extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];
	extern ValueKpp(*kpp_); /*[SQ_NB][fe_end][fe_end]*/

	// 2次元配列で添字がでかいやつ(16GBを超えると)Visual C++2017で C2036(サイズが不明です)のコンパイルエラーになる。
	// このエラー名がわかりにくいのもさることながら、これがコンパイルエラーになるのも解せない。
	// 1次元配列で確保したら今度はC2148(配列サイズの合計は 0x7fffffffバイトを超えることはできません)になる。
	// これはVC++のx86用のコンパイラでx64のコードをコンパイルするときに生じる、cross platformの古くからあるbugらしい。
	// cf. https://stackoverflow.com/questions/19803162/array-size-error-x64-process
	// 仕方ないのでサイズを指定せずに確保する。なんなんだこれ…。
	// あ、size指定せずに単なるポインタだと(void*)(*kppp)とかでキャストできないから駄目なのか…。サイズ1にしておくか？気持ち悪いのでそれはやめよう…。


	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk  (*kk_ )
#define kkp (*kkp_)
#define kpp ( kpp_)

	const u64 kpp_square_fe_end = (u64)fe_end * (u64)fe_end;

	// 配列のサイズ
	const u64 size_of_kk = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp = (u64)SQ_NB*(u64)kpp_square_fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp;

	// kppの配列の位置を返すマクロ
	// i,j = piece0,piece1
	static ValueKpp& kpp_ksq_pcpc(int king_, BonaPiece i_, BonaPiece j_) {
		return *(kpp_ + (u64)king_ * kpp_square_fe_end + u64(i_)*fe_end + u64(j_));
	}

}      // namespace Eval

#endif // defined(_EVALUATE_KPP_KKPT_FV_VAR_H_)


#endif
