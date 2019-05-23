#include "../config.h"

#if defined (USE_ENTERING_KING_WIN)
// 入玉判定ルーチン

#include "../position.h"
#include "../search.h"

Move Position::DeclarationWin() const
{
	auto rule = Search::Limits.enteringKingRule;

	switch (rule)
	{
		// 入玉ルールなし
	case EKR_NONE: return MOVE_NONE;

		// CSAルールに基づく宣言勝ちの条件を満たしているか
		// 満たしているならば非0が返る。返し値は駒点の合計。
		// cf.http://www.computer-shogi.org/protocol/tcp_ip_1on1_11.html
	case EKR_24_POINT: // 24点法(31点以上で宣言勝ち)
	case EKR_27_POINT: // 27点法 == CSAルール
	{
		/*
		「入玉宣言勝ち」の条件(第13回選手権で使用のもの):

		次の条件が成立する場合、勝ちを宣言できる(以下「入玉宣言勝ち」と云う)。
		条件:
		(a) 宣言側の手番である。
		(b) 宣言側の玉が敵陣三段目以内に入っている。
		(c) 宣言側が(大駒5点小駒1点の計算で)
		・先手の場合28点以上の持点がある。
		・後手の場合27点以上の持点がある。
		・点数の対象となるのは、宣言側の持駒と敵陣三段目
		以内に存在する玉を除く宣言側の駒のみである。
		(d) 宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する。
		(e) 宣言側の玉に王手がかかっていない。
		(詰めろや必死であることは関係ない)
		(f) 宣言側の持ち時間が残っている。(切れ負けの場合)
		以上1つでも条件を満たしていない場合、宣言した方が負けとなる。
		(注) このルールは、日本将棋連盟がアマチュアの公式戦で使用しているものである。

		以上の宣言は、コンピュータが行い、画面上に明示する。
		*/
		// (a)宣言側の手番である。
		// →　手番側でこの関数を呼び出して判定するのでそうだろう。

		Color us = sideToMove;

		// 敵陣
		Bitboard ef = enemy_field(us);

		// (b)宣言側の玉が敵陣三段目以内に入っている。
		if (!(ef & king_square(us)))
			return MOVE_NONE;

		// (e)宣言側の玉に王手がかかっていない。
		if (checkers())
			return MOVE_NONE;


		// (d)宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する。
		int p1 = (pieces(us) & ef).pop_count();
		// p1には玉も含まれているから11枚以上ないといけない
		if (p1 < 11)
			return MOVE_NONE;

		// 敵陣にいる大駒の数
		int p2 = ((pieces(us, BISHOP_HORSE, ROOK_DRAGON)) & ef).pop_count();

		// 小駒1点、大駒5点、玉除く
		// ＝　敵陣の自駒 + 敵陣の自駒の大駒×4 - 玉

		// (c)
		// ・先手の場合28点以上の持点がある。
		// ・後手の場合27点以上の持点がある。
		Hand h = hand[us];
		int score = p1 + p2 * 4 - 1
			+ hand_count(h, PAWN) + hand_count(h, LANCE) + hand_count(h, KNIGHT) + hand_count(h, SILVER)
			+ hand_count(h, GOLD) + (hand_count(h, BISHOP) + hand_count(h, ROOK)) * 5;

		// rule==EKR_27_POINTならCSAルール。rule==EKR_24_POINTなら24点法(30点以下引き分けなので31点以上あるときのみ勝ち扱いとする)
		if (score < (rule == EKR_27_POINT ? (us == BLACK ? 28 : 27) : 31))
			return MOVE_NONE;

		// 評価関数でそのまま使いたいので非0のときは駒点を返しておく。
		return MOVE_WIN;
	}

	// トライルールの条件を満たしているか。
	case EKR_TRY_RULE:
	{
		Color us = sideToMove;
		Square king_try_sq = (us == BLACK ? SQ_51 : SQ_59);
		Square king_sq = king_square(us);

		// 1) 初期陣形で敵玉がいた場所に自玉が移動できるか。
		if (!(kingEffect(king_sq) & king_try_sq))
			return MOVE_NONE;

		// 2) トライする升に自駒がないか。
		if (pieces(us) & king_try_sq)
			return MOVE_NONE;

		// 3) トライする升に移動させたときに相手に取られないか。
		if (effected_to(~us, king_try_sq, king_sq))
			return MOVE_NONE;

		// 王の移動の指し手により勝ちが確定する
		return make_move(king_sq, king_try_sq);
	}

	default:
		UNREACHABLE;
		return MOVE_NONE;
	}
}

#endif // USE_ENTERING_KING_WIN
