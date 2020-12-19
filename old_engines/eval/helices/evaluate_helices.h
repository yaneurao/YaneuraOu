#ifndef _EVALUATE_HELICES_H_
#define _EVALUATE_HELICES_H_

#include "../../shogi.h"

// 螺旋評価関数(EVAL_HELICES)で用いるheader
// KPPP_KKPTに似ているが、KPPPのPが特定の駒種に限定される点において異なる。

// -----------------------------
//    対象とする駒種
// -----------------------------

// 12駒(桂+銀+金) これはelmo(SDT5)方式。22GB程度。学習に48GB程度。
#define KPPP_PIECE_NUMBER_NB 12

// 16駒(桂+銀+金+角+飛) 三角配列を用いて44GB程度。学習に300GB程度必要…なはず。
//#define KPPP_PIECE_NUMBER_NB 16

#if defined(EVAL_HELICES)

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
	typedef int16_t ValueKppp;

	// -----------------------------
	//     評価関数パラメーター
	// -----------------------------

	// 以下では、SQ_NBではなくSQ_NB_PLUS1まで確保したいが、Apery(WCSC26)の評価関数バイナリを読み込んで変換するのが面倒なので
	// ここではやらない。ゆえに片側の玉には対応出来ない。

	// -- 評価関数

// BonaPiece上のの場合の数。(BonaPieceZeroの場合と0枚目の手駒の場合も含める)
#if KPPP_PIECE_NUMBER_NB == 12
	// 盤上 = 3駒(桂・銀・金)×81升×2(先後)
	// 手駒 = 3駒×2(先後)×5通り(手駒、0から4枚)
	static const u64 kppp_fe_end = 81 * 3 * 2 + 3 * 2 * 5 + 1;
#elif KPPP_PIECE_NUMBER_NB == 16
	// 盤上 = 7種(桂・銀・金・角・飛・馬・龍)が81升×2(先後)
	// 手駒 = 3駒(桂・銀・金)×2(先後)×5通り(手駒、0から4枚) + 2駒(角・飛)×2(先後)×3通り(手駒、0～2枚)
	//static const u64 kppp_fe_end = 7 * 81 * 2 + 3 * 2 * 5 + 2 * 2 * 3 + 1;

	// 馬と龍を入れるのはさすがにやりすぎか。敵陣にいない角と飛車までに限定すべきか？馬・龍を入れたいが、テーブルサイズ的に厳しい。

	// 盤上 = 3種(桂・銀・金)が81升×2(先後) + 2種(角・飛)が63升×2(先後)
	// 手駒 = 3駒(桂・銀・金)×2(先後)×5通り(手駒、0から4枚) + 2駒(角・飛)×2(先後)×3通り(手駒、0～2枚)
//	static const u64 kppp_fe_end =3 *  81 * 2 + 2 * 63 * 2 + 3 * 2 * 5 + 2 * 2 * 3 + 1;

	// →　角・飛の番号を詰めるとBonaPieceからのmappingが簡単な計算で済まなくなるのでとりあえず詰めない実装で実験してみる。
	static const u64 kppp_fe_end = 3 * 81 * 2 + 2 * 81 * 2 + 3 * 2 * 5 + 2 * 2 * 3 + 1;

#else
	static_assert(false, "illegal KPPP_PIECE_NUMBER_NB");
#endif

	// triangle_fe_end = Σn(n-1)/2 , n=0..fe_end-1
	//                 =  fe_end * (fe_end - 1) * (fe_end - 2) / 6
	static const u64 kppp_triangle_fe_end = ((u64)kppp_fe_end)*((u64)kppp_fe_end - 1)*((u64)kppp_fe_end - 2) / 6;
	
	// 正方配列のときのfe_end
	static const u64 kppp_square_fe_end = (u64)kppp_fe_end * (u64)kppp_fe_end * (u64)kppp_fe_end;

#if defined(USE_HELICES_MIRROR)
	// ミラーがあるのでKPPPのKはその5/9。
	static const int KPPP_KING_SQ = EVAL_HELICES * 5 / 9;
#else
	static const int KPPP_KING_SQ = EVAL_HELICES;
#endif
	// 先手玉がこの段以下にあるなら、KPPPの計算をする。
	static const int KPPP_KING_RANK = ((int)RANK_9 - EVAL_HELICES / 9 + 1);

	extern ValueKk(*kk_)[SQ_NB][SQ_NB];
	extern ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	extern ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];
//	extern ValueKppp(*kppp_)[KPPP_KING_SQ][kppp_triangle_fe_end];
	extern ValueKppp(*kppp_);
		
	// 2次元配列で添字がでかいやつ(16GBを超えると)Visual C++2017で C2036(サイズが不明です)のコンパイルエラーになる。
	// このエラー名がわかりにくいのもさることながら、これがコンパイルエラーになるのも解せない。
	// 1次元配列で確保したら今度はC2148(配列サイズの合計は 0x7fffffffバイトを超えることはできません)になる。
	// これはVC++のx86用のコンパイラでx64のコードをコンパイルするときに生じる、cross platformの古くからあるbugらしい。
	// cf. https://stackoverflow.com/questions/19803162/array-size-error-x64-process
	// 仕方ないのでサイズを指定せずに確保する。なんなんだこれ…。
	// あ、size指定せずに単なるポインタだと(void*)(*kppp)とかでキャストできないから駄目なのか…。サイズ1にしておくか？気持ち悪いのでそれはやめよう…。


	// 以下のマクロ定義して、ポインタではない場合と同じ挙動になるようにする。
#define kk (*kk_)
#define kkp (*kkp_)
#define kpp (*kpp_)
#define kppp (kppp_)

	// 配列のサイズ
	const u64 size_of_kk   = (u64)SQ_NB*(u64)SQ_NB*(u64)sizeof(ValueKk);
	const u64 size_of_kkp  = (u64)SQ_NB*(u64)SQ_NB*(u64)fe_end*(u64)sizeof(ValueKkp);
	const u64 size_of_kpp  = (u64)SQ_NB*(u64)fe_end*(u64)fe_end*(u64)sizeof(ValueKpp);

#if KPPP_PIECE_NUMBER_NB == 12
	// 正方配列で持つ実装(22GB)
	const u64 size_of_kppp = (u64)KPPP_KING_SQ * (u64)kppp_square_fe_end * (u64)sizeof(ValueKppp);
#elif KPPP_PIECE_NUMBER_NB == 16
	// 三角配列で持つ実装(角・飛車の番号を詰めて12GB、詰めないなら17GB)
	const u64 size_of_kppp = (u64)KPPP_KING_SQ * (u64)kppp_triangle_fe_end * (u64)sizeof(ValueKppp);
#endif
	const u64 size_of_eval = size_of_kk + size_of_kkp + size_of_kpp + size_of_kppp;

	// kpppの三角配列の位置を返すマクロ
	// king  = 玉の升(ただし、ミラーを考慮している場合、1段が5升分しかないので注意)
	// i,j,k = piece0,piece1,piece2 
	static ValueKppp& kppp_ksq_pcpcpc(int king_, BonaPiece i_, BonaPiece j_, BonaPiece k_) {
		return *(kppp_ + (u64)king_ * kppp_square_fe_end + u64(i_)* (u64)kppp_fe_end * (u64)kppp_fe_end + u64(j_) * (u64)kppp_fe_end + u64(k_));
	}

	// BonaPieceの桂・銀・金を0～kppp_fe_end-1に写像する。
	static BonaPiece to_kppp_bona_piece(BonaPiece p);

	// kppp用に写像されたBonaPieceに対するミラーしたBonaPieceを求める。
	static BonaPiece kppp_mir_piece(BonaPiece p);

}      // namespace Eval

#endif // defined(EVAL_HELICES)


#endif
