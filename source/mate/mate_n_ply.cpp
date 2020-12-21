#include "mate.h"
#if defined(USE_MATE_1PLY)
#include "../position.h"

// ---------------------
// weak_mate_3ply()
// ---------------------

namespace Mate
{
	// 利きのある場所への取れない近接王手からの3手詰め
	Move weak_mate_3ply(const Position& pos, int ply)
	{
		// 1手詰めであるならこれを返す
		Move m = Mate::mate_1ply(pos);
		if (m)
			return m;

		// 詰まない
		if (ply <= 1)
			return MOVE_NONE;

		Color us = pos.side_to_move();
		Color them = ~us;
		Bitboard around8 = kingEffect(pos.king_square(them));

		// const剥がし
		Position* This = ((Position*)&pos);

		StateInfo si;
		StateInfo si2;

		// 近接王手で味方の利きがあり、敵の利きのない場所を探す。
		for (auto m : MoveList<CHECKS>(pos))
		{
			// 近接王手で、この指し手による駒の移動先に敵の駒がない。
			Square to = to_sq(m);
			if ((around8 & to)

#if ! defined(LONG_EFFECT_LIBRARY)
				// toに利きがあるかどうか。mが移動の指し手の場合、mの元の利きを取り除く必要がある。
				&& (is_drop(m) ? pos.effected_to(us, to) : (pos.attackers_to(us, to, pos.pieces() ^ from_sq(m)) ^ from_sq(m)))

				// 敵玉の利きは必ずtoにあるのでそれを除いた利きがあるかどうか。
				&& (pos.attackers_to(them, to, pos.pieces()) ^ pos.king_square(them))
#else
				&& (is_drop(m) ? pos.effected_to(us, to) :
					pos.board_effect[us].effect(to) >= 2 ||
					(pos.long_effect.directions_of(us, from_sq(m)) & Effect8::directions_of(from_sq(m), to)) != 0)

				// 敵玉の利きがあるので2つ以上なければそれで良い。
				&& (pos.board_effect[them].effect(to) <= 1)
#endif
				)
			{
				if (!pos.legal(m))
					continue;

				ASSERT_LV3(pos.gives_check(m));

				This->do_move(m, si, true);

				ASSERT_LV3(pos.in_check());

				// この局面ですべてのevasionを試す
				for (auto m2 : MoveList<EVASIONS>(pos))
				{
					if (!pos.legal(m2))
						continue;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (pos.gives_check(m2))
						goto NEXT_CHECK;

					This->do_move(m2, si2, false);

					ASSERT_LV3(!pos.in_check());

					if (!weak_mate_3ply(pos, ply - 2))
					{
						// 詰んでないので、m2で詰みを逃れている。
						This->undo_move(m2);
						goto NEXT_CHECK;
					}

					This->undo_move(m2);
				}

				// すべて詰んだ
				This->undo_move(m);

				// mによって3手で詰む。
				return m;

			NEXT_CHECK:;
				This->undo_move(m);
			}
		}
		return MOVE_NONE;
	}
} // namespace Mate

// ---------------------
// mate_odd_ply()
// ---------------------

namespace {

	// 王手の可能な組み合わせ。91通り？
	// https://tadaoyamaoka.hatenablog.com/entry/2018/06/03/225012
	// 自信がないので少し多めに確保する。
	const constexpr size_t MaxCheckMoves = 100;

	// n手詰み探索用のMovePicker
	// or_node : どれか一つでも詰みがあれば良いノードであるか。(詰みまで奇数手であればそう)
	// INCHECK : 王手がかかっているか
	// GEN_ALL : 歩の不成なども生成するのか
	template <bool or_node, bool INCHECK , bool GEN_ALL>
	class MovePicker {
	public:
		explicit MovePicker(const Position& pos) {

			// givesCheckを呼び出すのかのフラグ
			bool doGivesCheck = false;
			if (or_node) {
				// ORノードなのですべての王手の指し手を生成。

				// dlshogi、ここ、王手になる指し手を生成して、自玉に王手がかかっている場合、
				// 回避手になっているかをpseudo_legal()でチェックしているが、
				// pseudo_legal()のほうはわりと重い処理なので、自玉に王手がかかっているなら
				// evasionの指し手を生成して、それがgives_check()で王手の指し手になっているか
				// 見たほうが自然では？

				if (INCHECK)
				{
					last = GEN_ALL ? generateMoves<EVASIONS_ALL>(pos, moveList) : generateMoves<EVASIONS>(pos, moveList);
					// これが王手になるかはのちほどチェックする。
					doGivesCheck = true;
				}
				else {
					last = GEN_ALL ? generateMoves<CHECKS_ALL>(pos, moveList) : generateMoves<CHECKS>(pos, moveList);
					}
				}
			else {
				// ANDノードなので回避の指し手のみを生成
				// (王手になる指し手も含まれる)
				last = GEN_ALL ? generateMoves<EVASIONS_ALL>(pos, moveList): generateMoves<EVASIONS>(pos, moveList);
			}

			// legalでない指し手はいまのうちに除外。
			auto* curr = moveList;
			while (curr != last ) {
				// 以下の２つの指し手は除外する。
				// 1. doGivesCheck==trueなのに、王手になる指し手ではない。
				// 2. legalではない。
				if ((doGivesCheck && !pos.gives_check(curr->move)) || !pos.legal(curr->move))
					// 末尾の指し手を現在のcursorに移動させることでこの手を削除する。
					curr->move = (--last)->move;
					else
						++curr;
				}

			ASSERT_LV3(size() <= MaxCheckMoves);
		}
		size_t size() const { return static_cast<size_t>(last - moveList); }
		ExtMove* begin() { return &moveList[0]; }
		ExtMove* end() { return last; }
		bool empty() const { return size() == 0; }

	private:
		ExtMove moveList[MaxCheckMoves];
		ExtMove* last;
	};
}

// dlshogiのN手詰めルーチン

namespace Mate {

	// 3手詰めチェック
	// mated_even_ply()から内部的に呼び出される。
	template <bool INCHECK , bool GEN_ALL>
	FORCE_INLINE Move mate_3ply(Position& pos)
	{
		// OR節点

		StateInfo si;
		StateInfo si2;

		for (const auto& ml : MovePicker<true, INCHECK , GEN_ALL>(pos))
		{
			auto m = ml.move;

			pos.do_move(m, si, true);

			// この局面で詰みの指し手を発見したか？
			bool found_mate = false;

			// 千日手のチェック
			if (pos.is_repetition(16) == REPETITION_WIN)
			{
				// REPETITION_WINの時は、受け側の反則勝ち = この関数の呼び出し時の手番側の負け
				// なので、この指し手に関してはfound_mate == falseのままで良い。
			} else {
				// 以下は、REPETITION_WIN以外の時の処理。

				// この局面ですべてのevasion(王手回避手)を試す
				MovePicker<false, false , GEN_ALL> move_picker2(pos);

				if (move_picker2.size() == 0) {
					// 回避手がない = 1手で詰んだ。
					found_mate = true;
				}
				else {
					for (const auto& ml2 : move_picker2)
					{
						auto m2 = ml2.move;

						// この指し手で逆王手になるなら、不詰めとして扱う
						if (pos.gives_check(m2))
							goto NEXT_CHECK;

						pos.do_move(m2, si2, /* givesCheck */false);

						// mate_1ply()は王手がかかっている時に呼べないが、
						// この局面は王手はかかっていないから問題ない。

						if (! Mate::mate_1ply(pos)) {
							// 詰んでないので、m2で詰みを逃れている。
							pos.undo_move(m2);
							goto NEXT_CHECK;
						}

						pos.undo_move(m2);
					}
					// すべて詰んだので、mで詰む。
					found_mate = true;
		NEXT_CHECK:;
				}
			}

			pos.undo_move(m);

			if (found_mate)
				return m;
		}
		return MOVE_NONE;
	}

	// mated_even_ply() 定義前の宣言。
	// 明示的なテンプレート引数を持つ関数呼び出しで、事前に宣言されていない関数テンプレート名を使用することは、C++20 の拡張機能です。
	template <bool GEN_ALL>
	Move mated_even_ply(Position& pos, const int ply);

	// mate_odd_ply()の王手がかかっているかをtemplateにしたやつ。
	// ※　dlshogiのmateMoveInOddPlyReturnMove()を参考にさせていただいています。
	// InCheck : 王手がかかっているか
	// GEN_ALL : 歩の不成も生成するのか
	template <bool INCHECK , bool GEN_ALL>
	Move mate_odd_ply(Position& pos, const int ply)
	{
		if (ply == 3)
			return mate_3ply<INCHECK,GEN_ALL>(pos);
		else if (ply == 1)
			// 王手がかかっていないなら1手詰めを呼び出せるが、王手がかかっているなら1手詰めを呼べないので
			// evasionのなかから詰む指し手を探す必要がある。レアケースなので、ここでは不詰み扱いをしておく。
			return !pos.in_check() ? mate_1ply(pos) : MOVE_NONE;

		// OR接点なので一つでも詰みを見つけたらそれで良し。

		// すべての合法手について
		for (const auto& ml : MovePicker<true, INCHECK , GEN_ALL>(pos)) {

			// MovePickerの指し手で1手進める。
			// これが合法手であることは、MovePickerのほうで保証されている。
			auto m = ml.move;

			StateInfo state;
			// これが王手であることはわかっているので第3引数はtrueで固定しておく。
			pos.do_move(m, state, true);

			// 詰みを見つけたかのフラグ
			bool found_mate = false;

			// 千日手チェック
			switch (pos.is_repetition(16))
			{
			case REPETITION_DRAW:
			case REPETITION_WIN:      // 相手が勝ち
			case REPETITION_SUPERIOR: // 相手が駒得
				// この指し手は詰まない
				break;

			case REPETITION_LOSE: // 相手が負け
				// 連続王手の千日手は詰み扱い
				found_mate = true;
				break;

			case REPETITION_NONE:
			case REPETITION_INFERIOR: // 相手が駒損
				// この指し手でこのあと詰むかをチェック

				//std::cout << ml.move().toUSI() << std::endl;

				// さらにこの局面から偶数手で相手が詰まされるかのチェック
				found_mate = mated_even_ply<GEN_ALL>(pos, ply - 1) == MOVE_NONE /* 回避手がない == 詰み */;
				break;

			default: UNREACHABLE;
			}

			pos.undo_move(m);

			// 詰みを見つけたならその最初の手を返す。
			if (found_mate)
				return m;
		}

		// 詰みを発見できなかった。
		return MOVE_NONE;
	}

	// 奇数手詰め
	// 詰みがある場合は、その1手目の指し手を返す。詰みがない場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// GEN_ALL : 歩の不成も生成するのか
	Move mate_odd_ply(Position& pos, const int depth , bool GEN_ALL)
	{
		// 2×2の四通りのtemplateを呼び分ける。
		return GEN_ALL ? (pos.in_check() ? mate_odd_ply<true,true >(pos, depth) : mate_odd_ply<false,true >(pos, depth))
		               : (pos.in_check() ? mate_odd_ply<true,false>(pos, depth) : mate_odd_ply<false,false>(pos, depth));
	}

	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// GEN_ALL : 歩の不成も生成するのか。
	template <bool GEN_ALL>
	Move mated_even_ply(Position& pos, const int ply)
	{
		// AND節点なのでこの局面のすべての指し手に対して(手番側が)詰まなければならない。
		// 一つでも詰まない手があるならfalseを返す。


		// すべてのEvasion(王手回避の指し手)について
		for (const auto& ml : MovePicker<false, false , GEN_ALL>(pos))
		{
			//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
			auto m = ml.move;

			// この指し手で王手になるのか
			const bool givesCheck = pos.gives_check(m);

			// 1手動かす
			StateInfo state;
			pos.do_move(m, state, givesCheck);

			// 手番側が逃れる指し手を見つけたか。
			// これがtrueになった時は、その指し手で逃れているので、この関数はfalseを返す。
			bool found_escape = false;

			// 千日手チェック
			switch (pos.is_repetition(16)) {
			case REPETITION_WIN:	  // この関数に与えられた局面から見て相手が勝ち = 詰みを逃れていない
				break;

			case REPETITION_DRAW:
			case REPETITION_LOSE:     // この関数に与えられた局面から見て相手が負け = 詰みを逃れている
			case REPETITION_INFERIOR: // この関数に与えられた局面から見て相手が駒損 = 詰みを逃れている
				// 詰まされる手が見つからなかった(逃れている)時点で終了
				found_escape = true;
				break;

			case REPETITION_NONE:;
			case REPETITION_SUPERIOR: // この関数に与えられた局面から見て相手が駒得 →　逃れているかどうかはさらに調べないと…。
				if (ply == 4)
					// 3手詰めかどうか
					found_escape = !(givesCheck ? mate_3ply<true, GEN_ALL>(pos) : mate_3ply<false, GEN_ALL>(pos));
				//	// 詰みが見つからなかったら、逃れている。
				else
					// 奇数手詰めかどうか
					found_escape = !(givesCheck ? mate_odd_ply<true , GEN_ALL>(pos, ply - 1) : mate_odd_ply<false , GEN_ALL>(pos, ply - 1));
					// 詰みが見つからなかったら、逃れている。
				break;

			default: UNREACHABLE;
			}

			pos.undo_move(m);

			// 1手でも逃れる指し手を見つけた場合は、与えられた局面で詰まされない。
			if (found_escape)
				return m; // この指し手で逃れている。
		}

		// 詰みを逃れる指し手を見つけられなかった = 詰み
		return MOVE_NONE;
	}

	// 偶数手詰め
	// 前提) 手番側が王手されていること。
	// この関数は、その王手が、逃れられずに手番側が詰むのかを判定する。
	// 返し値は、逃れる指し手がある時、その指し手を返す。どうやっても詰む場合は、MOVE_NONEが返る。
	// ply     : 最大で調べる手数
	// gen_all : 歩の不成も生成するのか。
	Move mated_even_ply(Position& pos, const int ply, bool gen_all)
	{
		return gen_all ? mated_even_ply<true>(pos, ply) : mated_even_ply<false>(pos, ply);
	}


} // namespace Mate

#endif // if defined(MATE_1PLY)...
