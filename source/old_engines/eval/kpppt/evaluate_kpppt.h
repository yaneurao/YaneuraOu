#ifndef _EVALUATE_KPPPT_H_
#define _EVALUATE_KPPPT_H_

#include "../../shogi.h"

// KPP_KKPT型評価関数で用いるheader

#if defined(EVAL_KPPPT)

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

	// KPPTの自然な拡張。
	// KK、KKPは16bitでも事足りそうではあるが、ちょっと危ないし、
	// KPPTのKKとKKPのファイルは使いまわしたいので互換性重視でこうしておく。
	
	typedef std::array<int32_t, 2> ValueKk;
	typedef std::array<int32_t, 2> ValueKkp;
	typedef std::array<int16_t, 2> ValueKpp;
	typedef std::array<int16_t, 2> ValueKppp;

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉には対応出来ない。

	// -- 評価関数

	// triangle_fe_end = Σn(n-1)/2 , n=0..fe_end-1
	//                 =  fe_end * (fe_end - 1) * (fe_end - 2) / 6
	static const u64 kppp_triangle_fe_end = ((u64)Eval::fe_end)*((u64)Eval::fe_end - 1)*((u64)Eval::fe_end - 2) / 6;

#if defined(USE_KPPPT_MIRROR)
	// ミラーがあるのでKPPPのKはその5/9。
	static const int KPPP_KING_SQ = EVAL_KPPPT * 5 / 9;
#else
	static const int KPPP_KING_SQ = EVAL_KPPPT;
#endif
	// 先手玉がこの段以下にあるなら、KPPPの計算をする。
	static const Rank KPPP_KING_RANK = (Rank)(RANK_9 - EVAL_KPPPT / 9 + 1);

	extern ValueKk  (*kk_  )[SQ_NB][SQ_NB];
	extern ValueKkp (*kkp_ )[SQ_NB][SQ_NB][fe_end];
	extern ValueKpp (*kpp_ )[SQ_NB][fe_end][fe_end];
	extern ValueKppp(*kppp_)/* [KPPP_KING_SQ][kppp_triangle_fe_end] */;

	// 2次元配列で添字がでかいやつ(16GBを超えると)Visual C++2017で C2036(サイズが不明です)のコンパイルエラーになる。
	// このエラー名がわかりにくいのもさることながら、これがコンパイルエラーになるのも解せない。
	// 1次元配列で確保したら今度はC2148(配列サイズの合計は 0x7fffffffバイトを超えることはできません)になる。
	// これはVC++のx86用のコンパイラでx64のコードをコンパイルするときに生じる、cross platformの古くからあるbugらしい。
	// cf. https://stackoverflow.com/questions/19803162/array-size-error-x64-process
	// 仕方ないのでサイズを指定せずに確保する。なんなんだこれ…。
	// あ、size指定せずに単なるポインタだと(void*)(*kppp)とかでキャストできないから駄目なのか…。サイズ1にしておくか。
	// いや、以下のように定義して回避する。

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk   (*kk_)
#define kkp  (*kkp_)
#define kpp  (*kpp_)
#define kppp (kppp_)

	// 配列のサイズ
	const u64 size_of_kk   = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp  = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp  = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);
	const u64 size_of_kppp = (u64)KPPP_KING_SQ*kppp_triangle_fe_end *(u64)sizeof(ValueKppp);
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp + size_of_kppp;

	// kpppの三角配列の位置を返すマクロ
	// king  = 玉の升(ただし、ミラーを考慮している場合、1段が5升分しかないので注意)
	// i,j,k = piece0,piece1,piece2 
	// 前提条件) i > j > k
	static ValueKppp& kppp_ksq_pcpcpc(int king_, BonaPiece i_, BonaPiece j_, BonaPiece k_) {
		ASSERT_LV3(i_ > j_ && j_ > k_);
		return *(kppp_ + (u64)king_ * kppp_triangle_fe_end + u64(i_)*(u64(i_) - 1) * (u64(i_) - 2) / 6 + u64(j_)*(u64(j_) - 1) / 2 + u64(k_));
	}

}      // namespace Eval

#endif // defined(EVAL_KPPPT)


#endif
