#ifndef _EVALUATE_NABLA_H_
#define _EVALUATE_NABLA_H_

#include "../../shogi.h"

// nabla型型評価関数で用いるheader
// これはKPP_KKPTからfe_endの拡張を行なったもの。

#if defined(EVAL_NABLA)

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


	// 拡張されたBonaPiece
	const BonaPiece fe_nabla_pawn = fe_old_end;


	// あるP(>=fe_old_end)に対して、それがどこのPを複合したものであるかを返す関数。

	// 実際はBonaPieceだけ返されても付随情報がないと困るので付随情報を保持する構造体を用意する。
	struct BPInfo
	{
		BonaPiece p;
		File f;
		Rank r;
		Piece pc;
		bool right; // 右側盤面用の歩であるか

		BPInfo(BonaPiece p_, File f_, Rank r_, Piece pc_, bool right_) : p(p_), f(f_), r(r_), pc(pc_), right(right_) {}
	};

	void map_from_nabla_pawn_to_bp(BonaPiece p, std::vector<BPInfo>& a)
	{
		ASSERT_LV1(p >= fe_nabla_pawn);
		int index = p - fe_nabla_pawn;

		a.clear();

		// 0 : 先手1～5筋の歩 , 1 : 先手9～5筋の歩 , 2: 後手の9～5筋の歩 , 3: 後手の1～5筋の歩
		// 例えば、0の集合は1024足すとmir_piece()したもの。2048足すとinv_piece()したものになる。

		int type = index / 1024;

		// その何番目であるか。
		// 4進数としてみなして、それぞれの桁で歩の段を表現してあるものとする。
		// 例)
		// 0 : なし
		// 1 : 17歩
		// 2 : 16歩
		// 3 : 15歩
		// 4 : 27歩
		// 5 : 17歩,27歩
		// 詳しく知りたいなら20行ほど下にある#ifを有効にしてみると良い。
		int m = index % 1024;

		Color c = (type < 2) ? BLACK : WHITE;
		File af[4] = { FILE_1, FILE_9 , FILE_9 , FILE_1 };

		// この升からスキャン
		File f = af[type];
		Rank r = (c == BLACK) ? RANK_7 : RANK_3;
		Piece pc = (c == BLACK) ? B_PAWN : W_PAWN;

		for (int i = 0; i < 5; ++i)
		{
			// 4進数とみなして1桁ずつ取り出す
			int n = m % 4;
			m /= 4;

			// 取り出した桁が0であるなら、この筋の歩は対象外にいるので無視。
			if (n != 0)
			{
				--n;
				Rank r2 = (Rank)((int)r + ((c == BLACK) ? (-n) : (+n)));
				Square sq = f | r2;

				bool right = (type == 0) || (type == 2);
				a.emplace_back(BPInfo(BonaPiece(kpp_board_index[pc].fb + sq), f, r2, pc, right));
			}

			f = (File)((int)f + (((c == BLACK) ^ (type % 2)) ? 1 : -1));
		}
	};


}      // namespace Eval

#endif // defined(EVAL_NABLA)


#endif
