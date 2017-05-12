#ifndef _EVALUATE_KPPT_H_
#define _EVALUATE_KPPT_H_

#include "../shogi.h"

// KPPT評価関数で用いる共用header的なもの。

#if defined(EVAL_KKPT) || defined (EVAL_KPPT)

#include "../evaluate.h"


// KKファイル名
#define KK_BIN "KK_synthesized.bin"

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
	typedef std::array<int32_t, 2> ValueKk;
	typedef std::array<int16_t, 2> ValueKpp;
	typedef std::array<int32_t, 2> ValueKkp;

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉や、駒落ちの盤面には対応出来ない。

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32)

	// 評価関数パラメーターを他プロセスと共有するための機能。

	// KK
	extern ValueKk(*kk_)[SQ_NB][SQ_NB];

	// KPP
	extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

	// KKP
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];

	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk (*kk_)
#define kpp (*kpp_)
#define kkp (*kkp_)

	// memory mapped fileを介して共有するデータ
	struct SharedEval
	{
		ValueKk kk_[SQ_NB][SQ_NB];
		ValueKpp kpp_[SQ_NB][fe_end][fe_end];
		ValueKkp kkp_[SQ_NB][SQ_NB][fe_end];
	};

#else

	// 通常の評価関数テーブル

	// KK
	extern ALIGNED(32) ValueKk kk[SQ_NB][SQ_NB];

	// KPP
	extern ALIGNED(32) ValueKpp kpp[SQ_NB][fe_end][fe_end];

	// KKP
	extern ALIGNED(32) ValueKkp kkp[SQ_NB][SQ_NB][fe_end];

#endif


}

#endif // EVAL_KPPT


#endif
