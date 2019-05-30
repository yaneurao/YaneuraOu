#include "mate1ply.h"

#if defined(USE_MATE_1PLY) && !defined(LONG_EFFECT_LIBRARY)

// 利きを用いない1手詰め判定用。(Bonanza6風)
// やねうら王2014からの移植。

#include "../../position.h"

//#include <iostream>
//using std::cout;

namespace {

	// sgn関数。C++標準になんでこんなもんすらないのか..。
	template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}

	// 1手詰めルーチンで用いる、移動によって王手になるかどうかの判定用テーブルで使う。
	enum PieceTypeCheck
	{
		PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO, // 不成りのまま王手になるところ(成れる場合は含まず)
		PIECE_TYPE_CHECK_PAWN_WITH_PRO, // 成りで王手になるところ
		PIECE_TYPE_CHECK_LANCE,
		PIECE_TYPE_CHECK_KNIGHT,
		PIECE_TYPE_CHECK_SILVER,
		PIECE_TYPE_CHECK_GOLD,
		PIECE_TYPE_CHECK_BISHOP,
		PIECE_TYPE_CHECK_ROOK,
		PIECE_TYPE_CHECK_PRO_BISHOP,
		PIECE_TYPE_CHECK_PRO_ROOK,
		PIECE_TYPE_CHECK_NON_SLIDER, // 王手になる非遠方駒の移動元

		PIECE_TYPE_CHECK_NB,
		PIECE_TYPE_CHECK_ZERO = 0,
	};

	ENABLE_FULL_OPERATORS_ON(PieceTypeCheck);

	// 王手になる候補の駒の位置を示すBitboard
	Bitboard CHECK_CAND_BB[SQ_NB_PLUS1][PIECE_TYPE_CHECK_NB][COLOR_NB];

	// 玉周辺の利きを求めるときに使う、玉周辺に利きをつける候補の駒を表すBB
	// COLORのところは王手する側の駒
	Bitboard CHECK_AROUND_BB[SQ_NB_PLUS1][PIECE_RAW_NB][COLOR_NB];

	// 移動により王手になるbitboardを返す。
	// us側が王手する。sq_king = 敵玉の升。pc = 駒
	inline Bitboard check_cand_bb(Color us, PieceTypeCheck pc, Square sq_king)
	{
		return CHECK_CAND_BB[sq_king][pc][us];
	}

	// 敵玉8近傍の利きに関係する自駒の候補のbitboardを返す。ここになければ玉周辺に利きをつけない。
	// pt = PAWN～HDK
	inline Bitboard check_around_bb(Color us, Piece pt, Square sq_king)
	{
		return CHECK_AROUND_BB[sq_king][pt - 1][us];
	}

	// sq1に対してsq2の升の延長上にある次の升を得る。
	// 隣接していないか、盤外になるときはSQUARE_NB
	// テーブルサイズを小さくしておきたいのでu8にしておく。
	/*Square*/ u8 NextSquare[SQ_NB_PLUS1][SQ_NB_PLUS1];
	inline Square nextSquare(Square sq1, Square sq2) { return (Square)NextSquare[sq1][sq2]; }

	// 上で宣言してある一連のテーブルの初期化。結構たいへん。
	void init_check_bb()
	{
		for (PieceTypeCheck p = PIECE_TYPE_CHECK_ZERO; p < PIECE_TYPE_CHECK_NB; ++p)
			for (auto sq : SQ)
				for (auto c : COLOR)
				{
					Bitboard bb = ZERO_BB, tmp = ZERO_BB;
					Square to;

					// 敵陣
					Bitboard enemyBB = enemy_field(c);

					// 敵陣+1段
					// Bitboard enemy4BB = c == BLACK ? RANK1_4BB : RANK6_9BB;

					switch ((int)p)
					{
					case PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO:
						// 歩が不成りで王手になるところだけ。

						bb = pawnEffect(~c, sq) & ~enemyBB;
						if (!bb)
							break;
						to = bb.pop();
						bb = pawnEffect(~c, to);
						break;

					case PIECE_TYPE_CHECK_PAWN_WITH_PRO:

						bb = goldEffect(~c, sq) & enemy_field(c);
						bb = pawnEffect(~c, bb);
						break;

					case PIECE_TYPE_CHECK_LANCE:

						// 成りによるものもあるからな..候補だけ列挙しておくか。
						bb = lanceStepEffect(~c, sq);
						if (enemy_field(c) ^ sq)
						{
							// 敵陣なので成りで王手できるから、sqより下段の香も足さないと。
							if (file_of(sq) != FILE_1)
								bb |= lanceStepEffect(~c, sq + SQ_R);
							if (file_of(sq) != FILE_9)
								bb |= lanceStepEffect(~c, sq + SQ_L);
						}

						break;

					case PIECE_TYPE_CHECK_KNIGHT:

						// 敵玉から桂の桂にある駒
						tmp = knightEffect(~c, sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= knightEffect(~c, to);
						}
						// 成って王手(金)になる移動元
						tmp = goldEffect(~c, sq) & enemyBB;
						while (tmp)
						{
							to = tmp.pop();
							bb |= knightEffect(~c, to);
						}
						break;

					case PIECE_TYPE_CHECK_SILVER:

						// 敵玉から銀の銀にある駒。
						tmp = silverEffect(~c, sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= silverEffect(~c, to);
						}
						// 成って王手の場合、敵玉から金の銀にある駒
						tmp = goldEffect(~c, sq) & enemyBB;
						while (tmp)
						{
							to = tmp.pop();
							bb |= silverEffect(~c, to);
						}
						// あと4段目の玉に3段目から成っての王手。玉のひとつ下の升とその斜めおよび、
						// 玉のひとつ下の升の2つとなりの升
						{
							Rank r = (c == BLACK ? RANK_4 : RANK_6);
							if (r == rank_of(sq))
							{
								r = (c == BLACK ? RANK_3 : RANK_7);
								to = (file_of(sq) | r);
								bb |= to;
								bb |= cross45StepEffect(to);

								// 2升隣。
								if (file_of(to) >= FILE_3)
									bb |= (to + SQ_R * 2);
								if (file_of(to) <= FILE_7)
									bb |= (to + SQ_L * 2);
							}

							// 5段目の玉に成りでのバックアタック的な..
							if (rank_of(sq) == RANK_5)
								bb |= knightEffect(c, sq);
						}
						break;

					case PIECE_TYPE_CHECK_GOLD:
						// 敵玉から金の金にある駒
						tmp = goldEffect(~c, sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= goldEffect(~c, to);
						}
						break;

						// この4枚、どうせいないときもあるわけで、効果に乏しいので要らないのでは…。
					case PIECE_TYPE_CHECK_BISHOP:
					case PIECE_TYPE_CHECK_PRO_BISHOP:
					case PIECE_TYPE_CHECK_ROOK:
					case PIECE_TYPE_CHECK_PRO_ROOK:
						// 王の8近傍の8近傍(24近傍)か、王の3列、3行か。結構の範囲なのでこれ無駄になるな…。
						break;

						// 非遠方駒の合体bitboard。ちょっとぐらい速くなるんだろう…。
					case PIECE_TYPE_CHECK_NON_SLIDER:
						bb =  CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_GOLD][c]
							| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_KNIGHT][c]
							| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_SILVER][c]
							| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO][c]
							| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_PAWN_WITH_PRO][c];
						break;
					}
					bb &= ~Bitboard(sq); // sqの地点邪魔なので消しておく。
					CHECK_CAND_BB[sq][p][c] = bb;
				}


		for (Piece p = PAWN; p <= KING; ++p)
			for (auto sq : SQ)
				for (auto c : COLOR)
				{
					Bitboard bb = ZERO_BB, tmp = ZERO_BB;
					Square to;
					bb = ZERO_BB;

					switch (p)
					{
					case PAWN:
						// これ用意するほどでもないんだな
						// 一応、用意するコード書いておくか..
						bb = kingEffect(sq);
						bb = pawnEffect(c, bb);
						// →　このシフトでp[0]の63bit目に来るとまずいので..
						bb &= ALL_BB; // ALL_BBでand取っておく。
						break;

					case LANCE:
						// 香で玉8近傍の利きに関与するのは…。玉と同じ段より攻撃側の陣にある香だけか..
						bb = lanceStepEffect(~c, sq);
						if (file_of(sq) != FILE_1)
							bb |= lanceStepEffect(~c, sq + SQ_R) | (sq + SQ_R);
						if (file_of(sq) != FILE_9)
							bb |= lanceStepEffect(~c, sq + SQ_L) | (sq + SQ_L);
						break;

					case KNIGHT:
						// 桂は玉8近傍の逆桂か。
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= knightEffect(~c, to);
						}
						break;

					case SILVER:
						// 同じく
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= silverEffect(~c, to);
						}
						break;

					case GOLD:
						// 同じく
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= goldEffect(~c, to);
						}
						break;

					case BISHOP:
						// 同じく
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= bishopStepEffect(to);
						}
						break;

					case ROOK:
						// 同じく
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= rookStepEffect(to);
						}
						break;

						// HDK相当
					case KING:
						// 同じく
						tmp = kingEffect(sq);
						while (tmp)
						{
							to = tmp.pop();
							bb |= kingEffect(to);
						}
						break;

					default:
						UNREACHABLE;
					}

					bb &= ~Bitboard(sq); // sqの地点邪魔なので消しておく。
										 // CHECK_CAND_BBとは並び順を変えたので注意。
					CHECK_AROUND_BB[sq][p - 1][c] = bb;
				}

		// NextSquareの初期化
		// Square NextSquare[SQUARE_NB][SQUARE_NB];
		// sq1に対してsq2の升の延長上にある次の升を得る。
		// 隣接していないか、盤外になるときはSQUARE_NB

		for (auto s1 : SQ)
			for (auto s2 : SQ)
			{
				Square next_sq = SQ_NB;

				// 隣接していなくてもok。縦横斜かどうかだけ判定すべし。
				if (queenStepEffect(s1) & s2)
				{
					File vf = File(sgn(file_of(s2) - file_of(s1)));
					Rank vr = Rank(sgn(rank_of(s2) - rank_of(s1)));

					File s3f = file_of(s2) + vf;
					Rank s3r = rank_of(s2) + vr;
					// 盤面の範囲外に出ていないかのテスト
					if (is_ok(s3f) && is_ok(s3r))
						next_sq = s3f | s3r;
				}
				NextSquare[s1][s2] = next_sq;
			}

	}

	// 桂馬が次に成れる移動元の表現のために必要となるので用意。
	static Bitboard RANK3_5BB = RANK3_BB | RANK4_BB | RANK5_BB;
	static Bitboard RANK5_7BB = RANK5_BB | RANK6_BB | RANK7_BB;

	//
	//　以下、本当ならPositionに用意すべきヘルパ関数
	//


	// 上の関数群とは異なる。usのSliderの利きを列挙する。
	// avoid升にいる駒の利きは除外される。
	template <Color Us>
	Bitboard AttacksSlider(const Position& pos, const Bitboard& slide)
	{
		Bitboard bb, sum = ZERO_BB;
		Square from;

		bb = pos.pieces(Us, LANCE);
		while (bb)
		{
			from = bb.pop();
			sum |= lanceEffect(Us, from, slide);
		}
		bb = pos.pieces(Us, BISHOP_HORSE);
		while (bb)
		{
			from = bb.pop();
			sum |= bishopEffect(from, slide);
		}
		bb = pos.pieces(Us, ROOK_DRAGON);
		while (bb)
		{
			from = bb.pop();
			sum |= rookEffect(from, slide);
		}
		return sum;
	}

	// usのSliderの利きを列挙する
	// avoid升にいる駒の利きは除外される。
	template <Color Us>
	Bitboard AttacksSlider(const Position& pos, Square avoid_from, const Bitboard& occ)
	{
		Bitboard bb, sum = ZERO_BB;
		Bitboard avoid_bb = ~Bitboard(avoid_from);
		Square from;

		bb = pos.pieces(Us, LANCE) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= lanceEffect(Us, from, occ);
		}
		bb = pos.pieces(Us, BISHOP_HORSE) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= bishopEffect(from, occ);
		}
		bb = pos.pieces(Us, ROOK_DRAGON) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= rookEffect(from, occ);
		}
		return sum;
	}

	// NonSliderの利きのみ列挙
	template<Color ourKing>
	Bitboard AttacksAroundKingNonSlider(const Position& pos)
	{
		Square sq_king = pos.king_square(ourKing);
		Color them = ~ourKing;
		Square from;
		Bitboard bb;

		// 歩は普通でいい
		Bitboard sum = pawnEffect(them, pos.pieces(them, PAWN));

		// ほとんどのケースにおいて候補になる駒はなく、whileで回らずに抜けると期待している。
		bb = pos.pieces(them, KNIGHT) & check_around_bb(them, KNIGHT, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= knightEffect(them, from);
		}
		bb = pos.pieces(them, SILVER) & check_around_bb(them, SILVER, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= silverEffect(them, from);
		}
		bb = pos.pieces(them, GOLDS) & check_around_bb(them, GOLD, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= goldEffect(them, from);
		}
		bb = pos.pieces(them, HDK) & check_around_bb(them, KING, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= kingEffect(from);
		}
		return sum;
	}

	// Sliderの利きのみ列挙
	template <Color ourKing>
	Bitboard AttacksAroundKingSlider(const Position& pos)
	{
		Square sq_king = pos.king_square(ourKing);
		Color them = ~ourKing;
		Square from;
		Bitboard bb;
		Bitboard sum = ZERO_BB;

		bb = pos.pieces(them, LANCE) & check_around_bb(them, LANCE, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= lanceEffect(them, from, pos.pieces());
		}
		bb = pos.pieces(them, BISHOP_HORSE) & check_around_bb(them, BISHOP, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= bishopEffect(from, pos.pieces());
		}
		bb = pos.pieces(them, ROOK_DRAGON) & check_around_bb(them, ROOK, sq_king);
		while (bb)
		{
			from = bb.pop();
			sum |= rookEffect(from, pos.pieces());
		}
		return sum;
	}

	template <Color Us>
	Bitboard AttacksAroundKingNonSliderInAvoiding(const Position& pos, Square avoid_from)
	{
		Square sq_king = pos.king_square(Us);
		Color them = ~Us;
		Bitboard bb;
		Bitboard avoid_bb = ~Bitboard(avoid_from);
		Square from;

		// 歩は普通でいい
		Bitboard sum = pawnEffect(them, pos.pieces(them, PAWN));

		// ほとんどのケースにおいて候補になる駒はなく、whileで回らずに抜けると期待している。
		bb = pos.pieces(them, KNIGHT) & check_around_bb(them, KNIGHT, sq_king) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= knightEffect(them, from);
		}
		bb = pos.pieces(them, SILVER) & check_around_bb(them, SILVER, sq_king) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= silverEffect(them, from);
		}
		bb = pos.pieces(them, GOLDS) & check_around_bb(them, GOLD, sq_king) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= goldEffect(them, from);
		}
		bb = pos.pieces(them, HDK) & check_around_bb(them, KING, sq_king) & avoid_bb;
		while (bb)
		{
			from = bb.pop();
			sum |= kingEffect(from);
		}
		return sum;
	}


	// avoidの駒の利きだけは無視して玉周辺の敵の利きを考えるバージョン。
	// この関数ではわからないため、toの地点から発生する利きはこの関数では感知しない。
	// 王手がかかっている局面において逃げ場所を見るときに裏側からのpinnerによる攻撃を考慮して、玉はいないものとして
	// 考える必要があることに注意せよ。(slide = pos.slide() ^ from ^ king | to) みたいなコードが必要
	// avoidの駒の利きだけは無視して玉周辺の利きを考えるバージョン。
	template <Color Us>
	inline Bitboard AttacksAroundKingInAvoiding(const Position& pos, Square from, const Bitboard& occ)
	{
		return AttacksAroundKingNonSliderInAvoiding<Us>(pos, from) | AttacksSlider<~Us>(pos, from, occ);
	}

	// 歩が打てるかの判定用。
	// 歩を持っているかの判定も含む。
	template<Color Us>
	inline bool can_pawn_drop(const Position& pos, Square sq)
	{
		// 歩を持っていて、二歩ではない。
		return hand_count(pos.hand_of(Us), PAWN) > 0 && !((pos.pieces(Us, PAWN) & FILE_BB[file_of(sq)]));
	}

}

using namespace Effect8;

namespace Mate1Ply
{
	// Mate1Ply関係のテーブル初期化
	void init()
	{
		// CHECK_CAND_BB、CHECK_AROUND_BBの初期化
		init_check_bb();
	}
}

namespace {

	// kingがtoとbb_avoid以外の升に逃げられるか
	// toに駒がない場合、駒が打たれているのでこれによって升は遮断されているものとして考える。
	bool can_king_escape(const Position& pos, Color us, Square to, const Bitboard& bb_avoid, const Bitboard& slide_)
	{
		// toには駒が置かれているのでこれにより利きの遮断は発生している。(attackers_to()で利きを見るときに重要)
		// captureの場合、もともとtoには駒があるわけで、ここをxorで処理するわけにはいかない。
		Bitboard slide = slide_ | to;

		Square sq_king = pos.king_square(us);
		/*
		// kingもいないものとして考える必要がある。
		slide ^= sq_king;
		// これは呼び出し側でbb_avoidを計算するときに保証するものとする。
		*/

		// bbとtoと自駒のないところから移動先を探す
		Bitboard bb = kingEffect(sq_king) & ~(bb_avoid | to | pos.pieces(us));

		while (bb)
		{
			Square escape = bb.pop();

			if (!pos.attackers_to(~us, escape, slide))
				return true;
			// 何も破壊していないので即座に返って良い。

		}
		return false;
	}

	// kingがtoとbb_avoid以外の升に逃げられるか
	// toに駒がない場合、駒が打たれているのでこれによって升は遮断されているものとして考える。
	// またfromからは駒が除去されているものとして考える。
	bool can_king_escape(const Position& pos, Color us, Square from, Square to, const Bitboard& bb_avoid, const Bitboard& slide_)
	{
		// toには駒が置かれているのでこれにより利きの遮断は発生している。(attackers_to()で利きを見るときに重要)
		Bitboard slide = slide_ | to;

		Square sq_king = pos.king_square(us);
		// kingもいないものとして考える必要がある。
		slide ^= sq_king;
		// これは呼び出し側でbb_avoidを計算するときに保証するものとする。
		// →　ああ、だめだ。fromの後ろにあった駒での開き王手が..

		// bb_avoidとtoと自駒のないところから移動先を探す
		Bitboard bb = kingEffect(sq_king) & ~(bb_avoid | to | pos.pieces(us));

		while (bb)
		{
			Square escape = bb.pop();

			if (!(pos.attackers_to(~us, escape, slide) & ~Bitboard(from)))
				// fromにある攻撃駒は移動済なのでこれは対象外。
				return true;
			// 何も破壊していないので即座に返って良い。

		}
		return false;
	}

	// kingがbb_avoid以外の升に逃げられるか
	// toに駒がない場合、駒が打たれているのでこれによって升は遮断されているものとして考える。
	// またfromからは駒が除去されているものとして考える。
	// ただしtoには行けるものとする。
	bool can_king_escape_cangoto(const Position& pos, Color us, Square from, Square to, const Bitboard& bb_avoid, const Bitboard& slide_)
	{
		// toには駒が置かれているのでこれにより利きの遮断は発生している。(attackers_to()で利きを見るときに重要)
		Bitboard slide = slide_ | to;

		Square sq_king = pos.king_square(us);
		// kingもいないものとして考える必要がある。
		slide ^= sq_king;
		// これは呼び出し側でbb_avoidを計算するときに保証するものとする。
		// →　ああ、だめだ。fromの後ろにあった駒での開き王手が..

		// bb_avoid/*とto*/と自駒のないところから移動先を探す
		Bitboard bb = kingEffect(sq_king) & ~((bb_avoid /*| to*/ | pos.pieces(us)) & ~Bitboard(to));

		// toには移動できるのだよ。pos.pieces(us)には玉側の駒がtoにあることがあって、これは取られるものとして
		// 考える必要があるから、toを除外するコードが必要なのだよ。

		while (bb)
		{
			Square escape = bb.pop();

			if (!(pos.attackers_to(~us, escape, slide) & ~Bitboard(from)))
				// fromにある攻撃駒は移動済なのでこれは対象外。
				return true;
			// 何も破壊していないので即座に返って良い。

		}
		return false;
	}

	// 玉以外の駒でtoの駒が取れるのか？(toの地点には敵の利きがある or 届かないので玉では取れないものとする)
	bool can_piece_capture(const Position& pos, Color us, Square to, const Bitboard& pinned, const Bitboard& slide)
	{
		Square sq_king = pos.king_square(us);

		// 玉以外の駒でこれが取れるのか？(toの地点には敵の利きがある or 届かないので玉では取れないものとする)
		Bitboard sum = pos.attackers_to(us, to, slide) & ~pos.pieces(KING);
		while (sum)
		{
			Square from = sum.pop();

			// fromからtoに移動させて素抜きに合わないならばこれをもって良し。
			if (!pinned
				|| !(pinned & from)
				|| aligned(from, to, sq_king)
				)
				return true;
		}

		return false;
	}

	// toにある駒が捕獲できるのか
	// ただしavoid升の駒でのcaptureは除外する。
	bool can_piece_capture(const Position& pos, Color us, Square to, Square avoid, const Bitboard& pinned, const Bitboard& slide)
	{
		ASSERT_LV3(is_ok(to));

		Square sq_king = pos.king_square(us);

		// 玉以外の駒でこれが取れるのか？(toの地点には敵の利きがあるので玉では取れないものとする)
		Bitboard sum = pos.attackers_to(us, to, slide) & ~(pos.pieces(KING) | Bitboard(avoid));
		while (sum)
		{
			Square from = sum.pop();

			// fromからtoに移動させて素抜きに合わないならばこれをもって良し。
			if (!pinned
				|| !(pinned & from)
				|| aligned(from, to, sq_king)
				)
				return true;
		}

		return false;
	}

}

// 1手で詰むならばその指し手を返す。なければMOVE_NONEを返す
template <Color Us>
Move is_mate_in_1ply_imp(const Position& pos)
{
	ASSERT_LV3(!pos.checkers());

	Bitboard dcCandidates = pos.discovered_check_candidates();

	Color them = ~Us;
	Square sq_king = pos.king_square(them);

	// 相手玉側のpinされている駒の列挙(相手玉側は、この駒を動かすと素抜きに遭う)
	Bitboard pinned = pos.pinned_pieces(them);

	Square from, to;

	// -- 駒打ちによる即詰み

	// 駒が打てる場所
	Bitboard bb_drop = ~pos.pieces();

	// テンポラリ用
	Bitboard bb;

	// 攻撃範囲計算用
	Bitboard bb_attacks;

	Hand ourHand = pos.hand_of(Us);
	Hand themHand = pos.hand_of(them);

	// 飛車を短く打つ場合
	if (hand_count(ourHand, ROOK))
	{
		// 敵玉の上下左右の駒の打てる場所
		bb = rookStepEffect(sq_king) & kingEffect(sq_king) & bb_drop;

		while (bb)
		{
			to = bb.pop();
			// toに対して自駒が利いてないと意味ない
			if (!pos.attackers_to(Us, to))
				continue;

			// このtoに飛車を打つものとして…この十字方向には逃げられないわけだから…そこは駄目ですよ、と。
			bb_attacks = rookStepEffect(to);

			if (can_king_escape(pos, them, to, bb_attacks, pos.pieces())) { continue; }
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { continue; }
			return make_move_drop(ROOK, to);
		}
	}

	// 香を短く打つ場合
	if (hand_count(ourHand, LANCE))
	{
		bb = pawnEffect(them, sq_king) & bb_drop;
		if (bb)
		{
			to = bb.pop();
			if (pos.attackers_to(Us, to))
			{
				bb_attacks = lanceStepEffect(Us, to);
				if (can_king_escape(pos, them, to, bb_attacks, pos.pieces())) { goto SKIP_LANCE; }
				if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { goto SKIP_LANCE; }
				return make_move_drop(LANCE, to);

			SKIP_LANCE:;
			}
		}
	}

	// 角を短く打つ
	if (hand_count(ourHand, BISHOP))
	{
		// 敵玉の上下左右の駒の打てる場所
		bb = cross45StepEffect(sq_king) & bb_drop;

		while (bb)
		{
			to = bb.pop();
			// toに対して自駒が利いてないと意味ない
			if (!pos.attackers_to(Us, to))
				continue;

			// このtoに角を打つものとして…この斜め方向には逃げられないわけだから…そこは駄目ですよ、と。
			bb_attacks = bishopStepEffect(to);

			if (can_king_escape(pos, them, to, bb_attacks, pos.pieces())) { continue; }
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { continue; }
			return make_move_drop(BISHOP, to);
		}
	}

	// 金打ち
	if (hand_count(ourHand, GOLD))
	{
		bb = goldEffect(them, sq_king) & bb_drop;

		// 飛車を持っているならすでに調べた上の升は除外して良い。
		// (そこに金をおいて詰むなら飛車をおいて詰んでいるはずだから)
		if (hand_count(ourHand, ROOK))
			bb &= ~pawnEffect(Us, sq_king);

		while (bb)
		{
			to = bb.pop();
			// toに対して自駒が利いてないと意味ない
			if (!pos.attackers_to(Us, to))
				continue;

			bb_attacks = goldEffect(Us, to);

			if (can_king_escape(pos, them, to, bb_attacks, pos.pieces())) { continue; }
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { continue; }
			return make_move_drop(GOLD, to);
		}
	}
	// 銀打ち
	if (hand_count(ourHand, SILVER))
	{
		// 金打ちをすでに調べたのであれば前方向は除外
		if (hand_count(ourHand, GOLD))
		{
			// 角打ちも調べていたのであれば銀で詰むことはない
			if (hand_count(ourHand, BISHOP))
				goto SILVER_DROP_END;

			// 前方向を除外するために金のnotを用いる。
			bb = silverEffect(them, sq_king) & ~goldEffect(them, sq_king) & bb_drop;
		}
		else {
			bb = silverEffect(them, sq_king) & bb_drop;
		}
		while (bb)
		{
			to = bb.pop();
			// toに対して自駒が利いてないと意味ない
			if (!pos.attackers_to(Us, to))
				continue;

			bb_attacks = silverEffect(Us, to);

			if (can_king_escape(pos, them, to, bb_attacks, pos.pieces())) { continue; }
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { continue; }
			return make_move_drop(SILVER, to);
		}
	}
SILVER_DROP_END:;

	// 桂打ち
	if (hand_count(ourHand, KNIGHT))
	{
		bb = knightEffect(them, sq_king) & bb_drop;

		while (bb)
		{
			to = bb.pop();
			//      bb_attacks =knightEffect(Us, to);
			// 桂馬はto以外は王が1手で移動できない場所なので求めるまでもない。

			if (can_king_escape(pos, them, to, ZERO_BB, pos.pieces())) { continue; }
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) { continue; }
			return make_move_drop(KNIGHT, to);
		}
	}

	// -- 移動による1手詰め

	// 駒の移動可能な場所
	Bitboard bb_move = ~pos.pieces(Us);

	// 王手となる移動先
	Bitboard bb_check;

	// 自分のpin駒
	Bitboard our_pinned = pos.pinned_pieces(Us);

	// 自玉
	Square our_king = pos.king_square(Us);

	// 龍
	bb = pos.pieces(Us, DRAGON);
	while (bb)
	{
		from = bb.pop();
		Bitboard slide = pos.pieces() ^ from;
		bb_check = dragonEffect(from, slide) & bb_move & kingEffect(sq_king);
		// 玉の8近傍への移動

		// この龍の移動によってpinが変わるので再計算が必要なのか…。
		// この龍を除外してpinされている駒を計算する必要がある。

		// 龍　^金 　^王
		// 龍突っ込んだときに金のpinが変わる
		// これ、しかし
		// 飛　龍　^金 　^王
		// こういうケースがあるので龍自体を除去しないとおかしくなるな…。

		Bitboard new_pin = pos.pinned_pieces(them, from);
		// ここで調べるのは近接王手だけなのでpinが解除されることはないから移動先toによるpinの解除/増加はなく、
		// 考慮しなくてよい。

		while (bb_check)
		{
			to = bb_check.pop();
			// fromの駒以外のtoへの利きがあるかを調べる。
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }

			// 王手であることは保証されている。

			// この龍の移動がそもそも合法手か？
			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }

			// 玉の斜め四方への移動か
			if (cross45StepEffect(sq_king) & to)
				bb_attacks = dragonEffect(to, slide);
			else
				bb_attacks = rookStepEffect(to) | kingEffect(to);

			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }

			// 龍によるtoが玉8近傍の場合の両王手はない。
			if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }
			return make_move(from, to);
		}
	}

	// 飛
	bb = pos.pieces(Us, ROOK);
	while (bb)
	{
		from = bb.pop();
		Bitboard slide = pos.pieces() ^ from;
		bb_check = rookEffect(from, slide) & bb_move & kingEffect(sq_king);

		// 飛車の移動前の升から王側の駒をpinすることはできない。ゆえにpinはいまのままで良い。
		// →　あ、そうでもないのか

		//        ^玉
		//     ^金
		//   飛
		// 角
		//   この場合、飛車を移動させて金があらたにpinに入るのか..

		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }

			// 敵陣へ/から の移動であれば成りなので
			if (canPromote(Us, from, to))
			{
				if (cross45StepEffect(sq_king) & to)
					bb_attacks = dragonEffect(to, slide);
				else
					bb_attacks = rookStepEffect(to) | kingEffect(to);
			}
			else
				bb_attacks = rookStepEffect(to);

			// これで王手になってないと駄目
			if (!(bb_attacks & sq_king)) { continue; }

			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }

			// 移動元で飛車の、近接王手になっている以上、pin方向と違う方向への移動であるからこれは両王手である。
			if (dcCandidates & from)
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }

			if (!canPromote(Us, from, to))
				return make_move(from, to);
			else
				return make_move_promote(from, to);
		}
	}

	// -- 以下、同様

	// 馬
	bb = pos.pieces(Us, HORSE);
	while (bb)
	{
		from = bb.pop();
		Bitboard slide = pos.pieces() ^ from;
		bb_check = horseEffect(from, slide) & bb_move & kingEffect(sq_king);
		// 玉の8近傍への移動

		//      ^玉
		//   ^金
		// 馬
		// 馬の移動によってpinは解除されるが、王手になることはない。
		// ゆえに、pinは以前のままで良い？

		// →　駄目

		//     ^王
		//     ^金
		//      馬
		//      香
		// こういう配置から斜めに馬が移動しての王手で金が新たにpinされる。

		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }

			// 王手になっていることは保証されている

			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			bb_attacks = bishopStepEffect(to) | kingEffect(to);
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			// 移動元で馬だとpin方向を確認しないといけない。違う方向への移動による攻撃なら、これは両王手である。
			if ((dcCandidates & from) && !aligned(from, to, sq_king))
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }

			return make_move(from, to);
		}
	}

	// 角
	bb = pos.pieces(Us, BISHOP);
	while (bb)
	{
		from = bb.pop();
		Bitboard slide = pos.pieces() ^ from;
		bb_check = bishopEffect(from, slide) & bb_move & kingEffect(sq_king);
		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }

			// 敵陣へ/から の移動であれば成りなので
			if (canPromote(Us, from, to))
				bb_attacks = bishopStepEffect(to) | kingEffect(to);
			else
				bb_attacks = bishopStepEffect(to);

			// これで王手になってないと駄目
			if (!(bb_attacks & sq_king)) { continue; }
			if (pos.discovered(from, to, pos.king_square(Us), our_pinned)) { continue; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			// 移動元で角だとpin方向を変える王手なので、これは両王手である。
			if (dcCandidates & from)
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }

			if (!canPromote(Us, from, to))
				return make_move(from, to);
			else
				return make_move_promote(from, to);
		}
	}

	// 香の移動王手
	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_LANCE, sq_king) & pos.pieces(Us, LANCE);
	while (bb)
	{
		from = bb.pop();
		Bitboard slide = pos.pieces() ^ from;
		bb_attacks = lanceEffect(Us, from, slide);

		// 金で王手になる升を列挙するか…。
		bb_check = bb_attacks & bb_move & goldEffect(them, sq_king);

		// 香の場合も香の移動によってpinが解除されるケースはないのでpinは初期のもので良い。

		while (bb_check)
		{
			to = bb_check.pop();

			if (canPromote(Us, to))
				bb_attacks = goldEffect(Us, to);
			else
				bb_attacks = lanceStepEffect(Us, to);

			if (!(bb_attacks & sq_king)) { goto LANCE_NO_PRO; }
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { goto LANCE_NO_PRO; }
			if (pos.discovered(from, to, our_king, our_pinned)) { goto LANCE_NO_PRO; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { goto LANCE_NO_PRO; }
			// 成って角との両王手
			if (dcCandidates & from)
				;
			else if (can_piece_capture(pos, them, to, pinned, slide)) { goto LANCE_NO_PRO; }

			if (!canPromote(Us, to))
				return make_move(from, to);
			else
				return make_move_promote(from, to);

			// 敵陣で不成りで串刺しにする王手も入れなきゃ..
		LANCE_NO_PRO:;
			if ((Us == BLACK ? RANK3_BB : RANK7_BB) & to)
			{
				bb_attacks = lanceStepEffect(Us, to);
				if (!(bb_attacks & sq_king)) { continue; }
				if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
				if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
				if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
				// 串刺しでの両王手はありえない
				if (can_piece_capture(pos, them, to, pinned, slide)) { continue; }
				return make_move(from, to);
			}
		}
	}

	// 離し角・飛車等で詰むかどうか。
	// これ、レアケースなのでportingしてくるの面倒だし、判定できなくていいや。
#if 1

	// 離し角・離し飛車、移動飛車・龍での合い効かずで詰むかも知れん。
	// Bonanzaにはないが、これを入れておかないと普通の1手詰め判定と判定される集合が違って気持ち悪い。

	// 飛車持ちかつ、相手は歩だけ(歩は二歩で合い効かず)かつ
	// 移動可能箇所が3箇所以内
	// という判定条件で残り2箇所が利きがあり移動不可であることがわかれば…みたいな条件にしとくか。
	// てか、これ利き真面目に考慮してはいかんのか？
	// うーむ..

	// 合い駒なしである可能性が高い

	// 敵は歩以外を持っていないか。
	// これは、 歩の枚数 == hand であることと等価。(いまの手駒のbit layoutにおいて)

	if (hand_count(themHand, PAWN) == (int)themHand)
	{
		// 玉の8近傍の移動可能箇所の列挙
		Bitboard bb_king_movable = ~pos.pieces(them) & kingEffect(sq_king);

		// 玉周辺の利きを列挙。(これ、せっかく求めたならできればあとで使いまわしたいが…)
		// これ王手のかかっていない局面で呼び出すことを想定しているので貫通でなくてもいいか。
		Bitboard aakns = AttacksAroundKingNonSlider<~Us>(pos); // これはあとで使いまわす
		Bitboard aaks = AttacksAroundKingSlider<~Us>(pos);
		Bitboard aak = aakns | aaks;

		Bitboard escape_bb = bb_king_movable & ~aak; // 利きがない場所が退路の候補

													 // 利きが正しく生成できているかのテスト
													 //    sync_cout << aak << sync_endl;

		int esc_count = escape_bb.pop_count();
		if (esc_count >= 4)
			goto NEXT1; // 残念ながら退路がありすぎて話にならんかった。詰む可能性低いので調べなくていいや。
						// 退路3個はまだ許せる…。としよう。

						// 退路がなかろうが、あろうが、玉8近傍の駒のない升に対して順番に探そう。
						// 退路が3以下である以上、そんなに空いてはないはずだ。
		Bitboard bb2 = ~pos.pieces() & kingEffect(sq_king);

		//    bool esc_align = (esc_count == 1);
		// 退路が1個、もしくは
		// 退路が2個でそれらが直線上に並んでいるときはtoの場所を延長して良いと思う。
		if (esc_count == 2)
		{
			// countが2なので2回のpop()は保証される。
			bb = escape_bb;
			from = bb.pop();
			to = bb.pop();
			//      esc_align = aligned(from, to, sq_king);
		}

		while (bb2)
		{
			// 退路
			Square one = bb2.pop();

			// このあと
			// 1. sq_kingのone側のもうひとつ先の升toにsq_kingに利く駒が打てて
			// 2. その升に敵の利きがなくて、
			// 3. oneの升に歩が立たないなら
			// これにて詰み

			// 駒が打つ場所と間の升が空いている
			// →　これは前提条件から自動的に満たす
			// if (!pos.empty(one)) continue; // もう駄目

			// toの地点が盤外
			// このチェックが惜しいのなら最初玉の8近傍ではなく、toが盤外にならない8近傍を列挙すべきだが。
			to = nextSquare(sq_king, one);
			if (to == SQ_NB) continue; // もう駄目

			// toが自駒だとここに移動できないし..
			if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) == Us) continue;

			// oneが二歩で打てないことを確認しよう。
			if (can_pawn_drop<~Us>(pos, one)) continue; // もう駄目

			// toの地点にあるのが歩だと、このtoの地点とoneが同じ筋だと
			// このtoの歩を取ってoneに打てるようになってしまう。
			if (type_of(pos.piece_on(to)) == PAWN && file_of(to) == file_of(one) && hand_count(themHand, PAWN) >= 1) continue;

			auto dr = directions_of(sq_king, one);
			Piece pt;
			bool canLanceAttack = false;
			if (dr & DIRECTIONS_DIAG)
			{
				pt = BISHOP;

				// 斜めなら角を持ってなきゃ
				if (hand_count(ourHand, BISHOP) == 0)
					goto NEXT2;
			}
			else {
				pt = ROOK;

				// 十字なら飛車を持ってなきゃ
				// 上からなら香でもいいのか。
				canLanceAttack = (Us == BLACK ? dr == DIRECTIONS_D : dr == DIRECTIONS_U);
				if (canLanceAttack && hand_count(ourHand, LANCE) >= 1)
				{
					pt = LANCE;
				}
				else if (hand_count(ourHand, ROOK) == 0)
					goto NEXT2;
			}

			if (pos.piece_on(to)) goto NEXT2;
			// このケースはtoの駒を取ればいけるかも知れん。盤上の駒ptを移動できないか調べろ

			// oneに駒の移動合いができない
			if (can_piece_capture(pos, them, one, pinned, pos.pieces())) goto NEXT2;

			// toに打つ駒が取れない
			if (can_piece_capture(pos, them, to, pinned, pos.pieces())) goto NEXT2;

			// 退路が1個以下であればこれで詰みなのだが、もともと利きがあって塞がれていたほうの退路だとそうでもないから
			// 最終的に次のような処理が必要なのだ…。

			// 退路が2個以上ある場合は、これで詰むとは限らない。
			// escape_bbが打った駒の利きによって遮断されているかを調べる。
			// あ、しまった。toを打ったことによってescape_bbのどこかがまた状態が変わるのか…。
			escape_bb = bb_king_movable & ~(aakns | AttacksSlider<Us>(pos, pos.pieces() | to));

			if (dr & DIRECTIONS_DIAG) // pt == BISHOP
			{
				if (!(~bishopStepEffect(to) & escape_bb))
					return make_move_drop(pt, to);
			}
			else // if (pt == ROOK || pt==LANCE)
			{
				// LANCEの場合もtoの地点からの横の利きでは玉の8近傍に到達しないので同列に扱って良い。
				if (!(~rookStepEffect(to) & escape_bb))
					return make_move_drop(pt, to);
			}

			//    STEP2_DROP:;
			// toに打ったからsliderが遮断されたんでねーの？1升だけ延長する。
			if (esc_count <= 2)
			{
				Square nextTo = nextSquare(one, to);
				if (nextTo == SQ_NB) goto NEXT2;
				if (pos.piece_on(nextTo)) goto NEXT2;
				if (can_pawn_drop<~Us>(pos, to)) goto NEXT2;
				if (can_piece_capture(pos, them, nextTo, pinned, pos.pieces())) goto NEXT2;

				escape_bb = bb_king_movable & ~(aakns | AttacksSlider<Us>(pos, pos.pieces() | nextTo));

				if (dr & DIRECTIONS_DIAG) // pt == BISHOP
				{
					if (!(~bishopStepEffect(nextTo) & escape_bb))
						return make_move_drop(pt, nextTo);
				}
				else // if (pt == ROOK || pt==LANCE)
				{
					if (!(~rookStepEffect(nextTo) & escape_bb))
						return make_move_drop(pt, nextTo);
				}
			}

		NEXT2:;
			// この場合、toの地点に遠方駒を移動させてcapれば、高い確率で詰みなのだが。

			if (!(dr & DIRECTIONS_DIAG)) // (pt == ROOK || pt == LANCE)
			{
				// どこかtoの近くに飛車は落ちてないかね..
				// 飛車を移動させた結果、oneに敵の利きが生じるかも知らんけど。
				bool is_rook = rookStepEffect(to) & pos.pieces(Us, ROOK_DRAGON);
				bool is_dragon = kingEffect(to) & pos.pieces(Us, DRAGON);
				bool is_lance = (canLanceAttack) ? (lanceStepEffect(them, to) & pos.pieces(Us, LANCE)) : false;

				if (is_rook || is_dragon || is_lance)
				{
					// 落ちてるっぽい。移動可能かどうか調べる。
					bb = ZERO_BB;
					if (is_rook)
						bb = rookEffect(to, pos.pieces()) & pos.pieces(Us, ROOK_DRAGON);
					if (is_dragon)
						bb |= kingEffect(to) & pos.pieces(Us, DRAGON);
					if (is_lance)
						bb |= lanceEffect(them, to, pos.pieces()) & pos.pieces(Us, LANCE);

					while (bb)
					{
						from = bb.pop();
						// fromからtoに移動させてこれで詰むかどうかテスト

						// この指し手が合法でないといかん。
						if (pos.discovered(from, to, our_king, our_pinned)) continue;

						Bitboard slide = pos.pieces() ^ from;

						// toに移動させた駒が取れない
						if (can_piece_capture(pos, them, to, pinned, slide)) continue;
						// oneも移動合い等ができない。
						// toへの移動でさらにpinされることはあっても、pinが解除されることはない。
						// (将棋にはQUEENがないので)
						// ゆえに古いpinで考えておけばいいや。
						if (can_piece_capture(pos, them, one, to, pinned, slide)) continue;

						if (type_of(pos.piece_on(from)) == LANCE)
						{
							bb_attacks = rookStepEffect(to);
							// 貫通で考えておこう。裏の退路もいけないので。
							// 1升以上離れているので王手にするには不成りでいくしかなく、これは飛車利きに等しい
						}
						else if (canPromote(Us, from, to) || type_of(pos.piece_on(from)) == DRAGON)
						{
							bb_attacks = queenStepEffect(to);
						}
						else
							bb_attacks = rookStepEffect(to);
						// 貫通で考えておこう。裏の退路もいけないので。

						Bitboard new_slide = (pos.pieces() ^ from) | to;

						// aakns、小駒だから関係ないと思いきや、馬を動かすと関係あるんだな
						// aakns使わない実装にしよう..

						if (!(kingEffect(sq_king)
							& ~(pos.pieces(them) | AttacksAroundKingInAvoiding<~Us>(pos, from, new_slide) | bb_attacks)))
						{
							// これで詰みが確定した
							// 香は不成りでの王手
							if (type_of(pos.piece_on(from)) != LANCE && canPromote(Us, from, to) && !(pos.piece_on(from) & PIECE_PROMOTE))
								return make_move_promote(from, to);
							else
								return make_move(from, to);
						}
					}
				}
			}
			else {
				// 同じく角

				bool is_bishop = bishopStepEffect(to) & pos.pieces(Us, BISHOP_HORSE);
				bool is_horse = kingEffect(to) & pos.pieces(Us, HORSE);
				if (is_bishop || is_horse)
				{
					// 落ちてるっぽい。移動可能かどうか調べる。
					bb = ZERO_BB;
					if (is_bishop)
						bb = bishopEffect(to, pos.pieces()) & pos.pieces(Us, BISHOP_HORSE);
					if (is_horse)
						bb |= kingEffect(to) & pos.pieces(Us, HORSE);

					while (bb)
					{
						from = bb.pop();
						// fromからtoに移動させてこれで詰むかどうかテスト

						// この指し手が合法でないといかん。
						if (pos.discovered(from, to, our_king, our_pinned)) continue;

						Bitboard slide = pos.pieces() ^ from;
						// oneに駒の移動合いができない
						// このときtoの駒はcapられている可能性があるのでこの駒による移動合いは除外する。
						if (can_piece_capture(pos, them, one, to, pinned, slide)) continue;

						// toに打つ駒が取れない
						if (can_piece_capture(pos, them, to, pinned, slide)) continue;

						// fromから飛車がいなくなったことにより、利きがどうなるかを調べる必要がある。
						// 王手になることは保証されているから、
						// 玉周辺の退路(自駒のない場所)のうち、攻撃側の利きがないone以外の場所を探すが、それがないということだから..
						// fromの駒を消して玉周辺の利きを調べる必要がある。少し重いがこれを呼ぶか..
						// 馬の場合でも、one以外に玉の8近傍には利いていないので龍のときのような処理は不要。

						//cout << kingEffect(sq_king) << pos.pieces(them) << aakns
						//  << pos.AttacksAroundKingSlider(them, from, to) << pos.StepAttacksQueen(to);

						Bitboard new_slide = (pos.pieces() ^ from) | to;

						if (!(kingEffect(sq_king)
							&  ~(pos.pieces(them) | AttacksAroundKingInAvoiding<~Us>(pos, from, new_slide) | queenStepEffect(to)
								)))
							// 貫通で考えておく
						{
							// これで詰みが確定した
							if (canPromote(Us, from, to) && !(pos.piece_on(from) & PIECE_PROMOTE))
								return make_move_promote(from, to);
							else
								return make_move(from, to);
						}
					}
				}
			}
			// toへ移動させる指し手終わり。

		}
	}
NEXT1:;
#endif

	// 以下、金、銀、桂、歩。ひとまとめにして判定できるが…これらのひとまとめにしたbitboardがないしな…。
	// まあ、一応、やるだけやるか…。

	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_NON_SLIDER, sq_king)
		& (pos.pieces(Us, GOLDS, SILVER, KNIGHT, PAWN));
	if (!bb)
		goto DC_CHECK;

	// 金
	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_GOLD, sq_king)  & pos.pieces(Us, GOLDS);
	while (bb)
	{
		from = bb.pop();
		bb_check = goldEffect(Us, from) & bb_move & goldEffect(them, sq_king);
		// 金は成りがないのでこれで100%王手。

		if (!bb_check) { continue; }

		Bitboard slide = pos.pieces() ^ from;
		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			bb_attacks = goldEffect(Us, to);
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			if ((dcCandidates & from) && !aligned(from, to, sq_king))
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }
			return make_move(from, to);
		}
	}


	// 銀は成りと不成が選択できるので少し嫌らしい
	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_SILVER, sq_king)  & pos.pieces(Us, SILVER);
	while (bb)
	{
		from = bb.pop();
		bb_check = silverEffect(Us, from) & bb_move & kingEffect(sq_king);
		// 敵玉8近傍を移動先の候補とする。金+銀の王手なので…。
		if (!bb_check) { continue; }

		Bitboard slide = pos.pieces() ^ from;
		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			bb_attacks = silverEffect(Us, to);
			// これで王手になってないと駄目
			if (!(bb_attacks & sq_king)) { goto PRO_SILVER; }
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { goto PRO_SILVER; }
			if (pos.discovered(from, to, our_king, our_pinned)) { goto PRO_SILVER; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { goto PRO_SILVER; }
			if ((dcCandidates & from) && !aligned(from, to, sq_king))
				;
			else
				if (can_piece_capture(pos, them, to, new_pin, slide)) { goto PRO_SILVER; }
			// fromから移動したことにより、この背後にあった駒によって新たなpinが発生している可能性がある。
			// fromとtoと玉が直線上にない場合はpinの更新が必要。
			// これは面倒なのですべての場合で事前に新しいpinを求めることにする。

			return make_move(from, to);

		PRO_SILVER:;
			// 銀成りでの王手

			// 敵陣へ/から の移動であれば成りを選択できるので..
			if (!(canPromote(Us, from, to)))
				continue;

			bb_attacks = goldEffect(Us, to);
			if (!(bb_attacks & sq_king)) { continue; }
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			if ((dcCandidates & from) && !aligned(from, to, sq_king))
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }
			return make_move_promote(from, to);
		}
	}

	// 桂も成りと不成が選択できるので少し嫌らしい
	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_KNIGHT, sq_king)  & pos.pieces(Us, KNIGHT);
	while (bb)
	{
		from = bb.pop();
		bb_check = knightEffect(Us, from) & bb_move;
		if (!bb_check) { continue; }

		Bitboard slide = pos.pieces() ^ from;
		Bitboard new_pin = pos.pinned_pieces(them, from);

		while (bb_check)
		{
			to = bb_check.pop();
			bb_attacks = knightEffect(Us, to);
			// 敵陣1,2段目からのStepAttackはZERO_BB相当なのでここへの不成りが生成されることはない
			if (!(bb_attacks & sq_king)) { goto PRO_KNIGHT; }
			// 桂馬の特性上、成りと不成の二通りの王手の両方が同時に可能になることはないので以下ではcontinueで良い。
			//if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
			// →　この駒は取られないならそれで良い。ここへの味方の利きは不要。

			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			// 桂馬はpinされているなら移動で必ず両王手になっているはずである。
			if (dcCandidates & from)
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }
			return make_move(from, to);

		PRO_KNIGHT:;
			// 桂成りでの王手

			if (!(canPromote(Us, from, to))) { continue; }
			bb_attacks = goldEffect(Us, to);
			if (!(bb_attacks & sq_king)) { continue; }
			if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
			if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
			if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
			// 桂馬はpinされているなら移動で必ず両王手になっているはずである。
			if (dcCandidates & from)
				;
			else if (can_piece_capture(pos, them, to, new_pin, slide)) { continue; }
			return make_move_promote(from, to);
		}
	}

	// 歩の移動による詰み
	if (check_cand_bb(Us, PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO, sq_king) & pos.pieces(Us, PAWN))
	{
		// 先手の歩による敵玉の王手だとすると、敵玉の一升下(SQ_D)が歩の移動先。
		to = sq_king + (Us == BLACK ? SQ_D : SQ_U);
		if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != ~Us) { goto SKIP_PAWN; }
		from = to + (Us == BLACK ? SQ_D : SQ_U);

		// 敵陣であれば成りによる詰みチェックで引っかかるだろう。
		if (canPromote(Us, to)) { goto SKIP_PAWN; }

		Bitboard slide = pos.pieces() ^ from;
		if (!(pos.attackers_to(Us, to, slide) ^ from)) { goto SKIP_PAWN; }
		if (pos.discovered(from, to, our_king, our_pinned)) { goto SKIP_PAWN; }
		if (can_king_escape(pos, them, from, to, ZERO_BB, slide)) { goto SKIP_PAWN; }
		// 移動王手となるpinされている歩などはないので両王手は考慮しなくて良い。
		if (can_piece_capture(pos, them, to, pinned, slide)) { goto SKIP_PAWN; }
		return make_move(from, to);
	}
SKIP_PAWN:;

	// 歩の成りによる詰み
	bb = check_cand_bb(Us, PIECE_TYPE_CHECK_PAWN_WITH_PRO, sq_king) & pos.pieces(Us, PAWN);
	while (bb)
	{
		from = bb.pop();
		to = from + (Us == BLACK ? SQ_U : SQ_D);
		if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != ~Us) { continue; }
		bb_attacks = goldEffect(Us, to);

		if (!(bb_attacks & sq_king)) { continue; }

		Bitboard slide = pos.pieces() ^ from;
		if (!(pos.attackers_to(Us, to, slide) ^ from)) { continue; }
		if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
		if (can_king_escape(pos, them, from, to, bb_attacks, slide)) { continue; }
		if (can_piece_capture(pos, them, to, pinned, slide)) { continue; }
		return make_move_promote(from, to);
	}

DC_CHECK:;
	// 両王手による詰み
	if (dcCandidates)
	{
		// せっかくdcCandidatesが使えるのだから両王手も検出しよう。
		// 開き王手になる候補の駒があること自体レアなのでここは全駒列挙でいいだろう。

		// 敵陣
		Bitboard enemyBB = enemy_field(Us);

		bb = dcCandidates;

		while (bb)
		{
			from = bb.pop();
			Piece pt = type_of(pos.piece_on(from));
			switch (pt)
			{
				// 背後にいる駒は角が普通で、pinされているのは歩で成りとか、飛車で両王手とか、そんなのが
				// よくあるパターンではないかいな。

			case PAWN:
			{
				// 同じ筋だとpin方向と違う方向の移動にならない。
				if (file_of(from) == file_of(sq_king)) { continue; }

				// 移動性の保証
				to = from + (Us == BLACK ? SQ_U : SQ_D);
				if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != ~Us) { continue; }

				// toの地点で成れないと駄目
				if (!canPromote(Us, to)) continue;

				// toの地点に敵の利きがあるか、もしくは自分の利きがないなら、
				// この変化で1手詰めにならないことはすでに調べているので除外すべきだが、除外するコストも馬鹿にならないので
				// このまま調べてしまう。

				// これが王手になってないと話にならない。
				bb_attacks = goldEffect(Us, to);
				if (!(bb_attacks & sq_king)) continue;

				// 移動が合法手であること。
				if (pos.discovered(from, to, our_king, our_pinned)) { continue; }

				// 駒がfromからtoに移動したときに退路があるか。ただしbb_attackはいけないものとする。
				Bitboard slide = pos.pieces() ^ from;
				if (can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide)) { continue; }

				// すべての条件が成立したのでこれにて詰み
				return make_move_promote(from, to);
			}
			ASSERT_LV3(false); // こっちくんな
			// FALLTHROUGH
			case LANCE:
				continue; // 香による両王手はない。

			case KNIGHT:
				if (!(check_around_bb(Us, KNIGHT, sq_king) & from)) continue;

				bb = knightEffect(Us, from) &knightEffect(them, sq_king) & bb_move;
				while (bb)
				{
					to = bb.pop();
					if (aligned(from, to, sq_king)) { continue; }
					bb_attacks = knightEffect(Us, to);
					if (bb_attacks & sq_king) { continue; }
					if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
					Bitboard slide = pos.pieces() ^ from;
					if (can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide)) { continue; }
					return make_move(from, to);
				}

				bb = knightEffect(Us, from) &goldEffect(them, sq_king);
				while (bb)
				{
					to = bb.pop();
					if (aligned(from, to, sq_king)) { continue; }
					if (!(canPromote(Us, from, to))) { continue; }
					bb_attacks = goldEffect(Us, to);
					if (bb_attacks & sq_king) { continue; }
					if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
					Bitboard slide = pos.pieces() ^ from;
					if (can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide)) { continue; }
					return make_move_promote(from, to);
				}

				continue; // 気をつけろ！下に落ちたら死ぬぞ！

			case SILVER:
				// 王手になる見込みがない
				if (!(check_around_bb(Us, SILVER, sq_king) & from)) continue;
				// これで王手にはなる。成りも選択したいのでここコピペで書くか..それともlambdaで書くか..コピペでいいか。

				bb = silverEffect(Us, from) & silverEffect(them, sq_king) & bb_move;;
				while (bb)
				{
					to = bb.pop();
					if (aligned(from, to, sq_king)) { continue; }
					bb_attacks = silverEffect(Us, to);
					if (bb_attacks & sq_king) { continue; }
					if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
					Bitboard slide = pos.pieces() ^ from;
					if (can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide)) { continue; }
					return make_move(from, to);
				}

				bb = silverEffect(Us, from) & goldEffect(them, sq_king) & bb_move;;
				while (bb)
				{
					to = bb.pop();
					if (aligned(from, to, sq_king)) { continue; }
					if (!(canPromote(Us, from, to))) { continue; }
					bb_attacks = goldEffect(Us, to);
					if (bb_attacks & sq_king) { continue; }
					if (pos.discovered(from, to, our_king, our_pinned)) { continue; }
					Bitboard slide = pos.pieces() ^ from;
					if (can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide)) { continue; }
					return make_move_promote(from, to);
				}
				continue;

			case PRO_PAWN:
			case PRO_LANCE:
			case PRO_KNIGHT:
			case PRO_SILVER:
				pt = GOLD; // 以下の処理でややこしいのでGOLD扱いにしてしまう。
				// FALLTHROUGH
			case GOLD:
				// 王手になる見込みがない
				if (!(check_around_bb(Us, GOLD, sq_king) & from)) continue;

				// 王手生成ルーチンみたいな処理が必要なんだな..
				bb = goldEffect(Us, from) & goldEffect(them, sq_king);
				// この移動先であれば王手になる。
				break;

			case BISHOP:

				bb = bishopEffect(sq_king, pos.pieces()) |
					(kingEffect(sq_king) & (canPromote(Us, from) ? ALL_BB : enemyBB));
				// 敵陣8近傍、王からの角の利き、fromが敵陣であれば、敵陣にかぎらず玉8近傍も。
				// ここが角が移動してくれば王手になる升
				// これと角の利きとの交差をとって、そこを移動の候補とする。
				bb &= bishopEffect(from, pos.pieces());

				//        bb = pos.AttackBishop(from, pos.pieces()) & around24_bb(sq_king);

				break;

			case HORSE:
				bb = horseEffect(from, pos.pieces()) & horseEffect(sq_king, pos.pieces());

				//        bb = pos.AttackHorse(from, pos.pieces()) & around24_bb(sq_king);

				break;

			case ROOK:
				// 角のときと同様
				bb = rookEffect(sq_king, pos.pieces()) |
					(kingEffect(sq_king) & (canPromote(Us, from) ? ALL_BB : enemyBB));
				bb &= rookEffect(from, pos.pieces());

				// いやー。龍がpinされているということは背後にいるのはたぶん角であって、
				// 玉の24近傍への移動で高い確率で詰むような..

				//        bb = pos.AttackRook(from, pos.pieces()) & around24_bb(sq_king);
				// ここ、両王手専用につき、合駒見てないのよね。だから、この条件をここに入れるわけにはいかんのよ…。

				break;

			case DRAGON:

				bb = dragonEffect(from, pos.pieces()) & dragonEffect(sq_king, pos.pieces());

				//        bb = pos.AttackDragon(from, pos.pieces()) & around24_bb(sq_king);

				break;

			default:
				ASSERT_LV3(pt == KING);
				continue;
			}

			bb &= bb_move;

			bool is_enemy_from = canPromote(Us, from);

			// 候補はほとんどないはずだが。
			while (bb)
			{
				to = bb.pop();
				bool promo = is_enemy_from || canPromote(Us, to);

				// これ、開き王手になってないと駄目
				if (aligned(from, to, sq_king)) { continue; }

				if (pos.discovered(from, to, our_king, our_pinned)) { continue; }

				// この地点でのこの駒の利きは..
				//bb_attacks = pos.attacks_from(make_piece(Us, pt), to, pos.pieces() ^ sq_king); // sq_kingが除去されて貫通である必要がある。
				// この処理気持ち悪いか..王手できることは確定しているのだから駒種別にやってしまうべきか。

				Bitboard slide = pos.pieces() ^ from;
				switch (pt)
				{
				case SILVER:
					if (!promo) goto DC_SILVER_NO_PRO;
					bb_attacks = goldEffect(Us, to); break;
				case GOLD: bb_attacks = goldEffect(Us, to); break;
				case BISHOP:
					bb_attacks = bishopStepEffect(to);
					if (promo)
						bb_attacks |= kingEffect(to);
					break;
				case HORSE: bb_attacks = bishopStepEffect(to) | kingEffect(to); break;
				case ROOK:
					bb_attacks = rookStepEffect(to);
					if (promo)
						bb_attacks |= kingEffect(to);
					break;
				case DRAGON: bb_attacks = rookStepEffect(to) | kingEffect(to); break;
				default:
					ASSERT_LV3(false);
					bb_attacks = ZERO_BB;
				}

				if (!can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide))
				{
					if (promo && !(pt & PIECE_PROMOTE) && pt != GOLD)
						return make_move_promote(from, to);
					return make_move(from, to);
				}

			DC_SILVER_NO_PRO:
				if (pt == SILVER)
				{
					// 銀のときだけ銀成り/不成りの判定が必要だわさ..
					// 上では成りを判定済なので不成りでの王手を判定

					bb_attacks = silverEffect(Us, to);
					if (!can_king_escape_cangoto(pos, them, from, to, bb_attacks, slide))
					{
						return make_move(from, to);
					}
				}
			}
		}
	}
	// 両王手ではないが、玉の24近傍から24-8 = 16近傍への移動で、かつfromに利きがなければ
	// この移動で詰む可能性が濃厚なので、これについては調べることにする。
	// 合い駒なしである可能性が高い場合についてのみ。

	// 歩以外を持っていないか。
	// これは、 歩の枚数 == hand であることと等価。(いまの手駒のbit layoutにおいて)

	if (dcCandidates && hand_count(themHand, PAWN) == (int)themHand)
	{
		// 玉の8近傍にある開き王手可能駒について
		//    bb = dcCandidates & kingEffect(sq_king);
		// 24近傍まで拡張していいだろう。

		bb = dcCandidates & around24_bb(sq_king);

		while (bb)
		{
			from = bb.pop();

			// if (can_piece_capture(pos, them, from, pinned, pos.pieces()))
			//  continue;
			// この駒が取られるというなら、その取られる駒のいる升への移動ぐらい調べれば良さそうだが。
			// 遠方からの利きなら、その利きを遮断できるならその方向に移動させるだとか、いろいろありそうだが…。
			// まあいいか…。判定が難しいしな。

			Bitboard atk = pos.attackers_to(them, from) & ~Bitboard(sq_king);
			if (atk)
			{
				if (atk.pop_count() >= 2)
					continue; // 2つ以上利きがあるなら消せないわ

							  // 1つしかないので、その場所への移動を中心に考えよう。そこは敵駒なのでbb_moveを見るまでもなく
							  // 上の升には移動可能
			}
			else {
				// 24近傍(ただし、馬・龍は16近傍)
				atk = around24_bb(sq_king) & bb_move; // 別にどこでも良いものとする
			}

			Piece pt = type_of(pos.piece_on(from));
			switch ((int)pt)
			{
			case PAWN:
			case LANCE:
				// 歩が玉の24近傍から成って開き王手で詰むパターンはめったに出てこないのでこれはいいや
				continue;

			case KNIGHT:
				// 成って詰みはあるか..それだけ見るか..
				if (!((Us == BLACK ? RANK3_5BB : RANK5_7BB) & from))
					continue;

				bb_attacks = knightEffect(Us, from) & ~goldEffect(them, sq_king);
				break;

			case SILVER:
				bb_attacks = silverEffect(Us, from);
				// 王手にならない升のみ列挙したいが銀は成り/不成りが選択できるので、まあこれはいいや..
				break;

			case PRO_PAWN:
			case PRO_LANCE:
			case PRO_KNIGHT:
			case PRO_SILVER:
			case GOLD:

				bb_attacks = goldEffect(Us, from) & ~goldEffect(them, sq_king);
				pt = GOLD;
				break;

			case BISHOP:
				bb_attacks = bishopEffect(from, pos.pieces());
				break;

			case HORSE:
				bb_attacks = horseEffect(from, pos.pieces()) & ~kingEffect(sq_king);
				// 16近傍に(王手になってしまうので)
				break;

			case ROOK:
				bb_attacks = rookEffect(from, pos.pieces());
				break;

			case DRAGON:
				bb_attacks = dragonEffect(from, pos.pieces()) & ~kingEffect(sq_king);
				break;

				// 玉が敵玉24近傍にいたということを意味している。
				// この移動による詰みはめったに出てこないから無視していいや。
			case KING:
				continue;
			}

			// この駒を玉の16近傍へ移動させる指し手を考える。
			// この処理ループの一回目だけでいいのだが…しかしループはたぶん1回で終了であることがほとんどだから
			// これはこれでいいか..
			// Bitboard target = around24_bb(sq_king) & ~kingEffect(sq_king);

			// 移動先
			Bitboard bb2 = bb_attacks & atk;

			Bitboard slide = pos.pieces() ^ from;
			while (bb2)
			{
				to = bb2.pop();

				// 開き王手になっていない。
				if (aligned(from, to, sq_king))
					continue;

				// 合法手か？
				if (pos.discovered(from, to, our_king, our_pinned))
					continue;

				// fromに歩が打てない
				if (can_pawn_drop<~Us>(pos, from))
					continue;

				// ただし、toが歩のcaptureだとfromに歩が打ててしまう可能性があるのでskip。
				// 盤面最上段だとアレだが、まあ除外していいだろう。
				bool capPawn = type_of(pos.piece_on(to)) == PAWN;
				if (capPawn && file_of(from) == file_of(to))
					continue;

				Bitboard new_slide = slide | to;

				Bitboard new_pinned = pos.pinned_pieces(them, from, to);

				// fromの地点に駒が利いていないか。
				// fromからtoに移動したときに、toの影になってfromの地点に利いていない場合を考慮しこう書く。
				// toの地点に駒がいくのでpinnedが変わるのか…。うわ、なんだこれ..
				if (can_piece_capture(pos, them, from, to, new_pinned, new_slide))
					continue;

				// 玉の8近傍だとcapられる可能性がある。
				if (kingEffect(sq_king) & to)
				{
					// from以外の駒が利いてない == capられる!!
					if (!(pos.attackers_to(Us, to) ^ from))
						continue;
				}

				// ここでの利きを考える。
				switch (pt)
				{

				case SILVER:
					// 成り不成りがある。成りは、ここで調べ、不成りはあとで調べる。

					// 成れないならば不成りの判定へ
					if (!canPromote(Us, from, to))
						goto DISCOVER_ATTACK_CONTINUE_SILVER;
					bb_attacks = goldEffect(Us, to);
					break;

				case KNIGHT:
					// 桂は成れるときのみ列挙しているので、移動先では金という扱いで良い。
				case GOLD:
					bb_attacks = goldEffect(Us, to);
					break;

				case BISHOP:
					if (canPromote(Us, from, to))
						bb_attacks = horseEffect(to, new_slide);
					else
						bb_attacks = bishopEffect(to, new_slide);
					break;

				case HORSE:
					bb_attacks = horseEffect(to, new_slide);
					break;

				case ROOK:
					if (canPromote(Us, from, to))
						bb_attacks = dragonEffect(to, new_slide);
					else
						bb_attacks = rookEffect(to, new_slide);
					break;

				case DRAGON:
					bb_attacks = dragonEffect(to, new_slide);
					break;

				default:
					ASSERT_LV3(false);
					continue;
				}

				// これが王手になっているということは両王手であり、両王手ルーチンで調べたはずだから除外
				if (bb_attacks & sq_king)
					goto DISCOVER_ATTACK_CONTINUE_SILVER;

				// 逃げ場所があるのか？
				// 王手がかかっている局面において逃げ場所を見るときに裏側からのpinnerによる攻撃を考慮して、玉はいないものとして考える必要がある。
				if (kingEffect(sq_king)
					& ~(bb_attacks | AttacksAroundKingInAvoiding<~Us>(pos, from, new_slide ^ sq_king) | pos.pieces(them)))
					goto DISCOVER_ATTACK_CONTINUE_SILVER;

				// ここでは開き王手であることは保証されている。
				// sq_kingの隣の升からpinnerとの間に利きがなければこれにて詰みが確定する。

				// 16近傍への移動なのでs1,s2が近接でないことは保証されているが、sq_king側の升から調べないといけないので..


				// !!!
				// !!!  以下の部分のコードを修正するときは、その下に銀の不成りのコードがあるのでそちらも修正すること！
				// !!!

				{
					Square s1 = sq_king;
					Square s2 = s1 + (nextSquare(s1, from) - from);
					do
					{
						// s2の地点に玉以外の駒が利いていたらアウト

						// 駒はfromからtoに移動しているものとする。ゆえにtoの地点に元いた駒の利きは除外して考える必要がある。
						// それからfromから駒が除去されて遮断が変わったことによる影響はnew_slideによって与えられているものとする。
						if (can_piece_capture(pos, them, s2, to, new_pinned, new_slide))
							goto DISCOVER_ATTACK_CONTINUE_SILVER;

						// s2の地点がfromはなく、かつpinnerであれば、終了
						// ただしpinnerが取られる可能性があるので、上のcaptureの判定が先にある
						if (s2 != from && pos.piece_on(s2)) // 自駒に違いない
							break;

						// s2に合駒ができない。
						if (can_pawn_drop<~Us>(pos, s2) || (capPawn && file_of(s2) == file_of(to)))
							goto DISCOVER_ATTACK_CONTINUE_SILVER;

						Square s3 = nextSquare(s1, s2);
						s1 = s2;
						s2 = s3;
					} while (s2 != SQ_NB);

					// これで詰みが確定した
					// 桂→成りしか調べてないので成れるなら成りで。
					// 銀→不成と成りと選択できる。
					if (canPromote(Us, from, to) && !(pos.piece_on(from) & PIECE_PROMOTE) && pt != GOLD)
						return make_move_promote(from, to);
					else
						return make_move(from, to);
				}

			DISCOVER_ATTACK_CONTINUE_SILVER:;

				if (pt == SILVER)
				{
					// 銀不成も考慮しないと..(成りは上で処理されているものとする)
					// 以下、同様の処理
					bb_attacks = silverEffect(Us, to);
					if (bb_attacks & sq_king)
						goto DISCOVER_ATTACK_CONTINUE;

					if (kingEffect(sq_king)
						& ~(bb_attacks | AttacksAroundKingInAvoiding<~Us>(pos, from, new_slide ^ sq_king) | pos.pieces(them)))
						goto DISCOVER_ATTACK_CONTINUE;

					Square s1 = sq_king;
					Square s2 = s1 + (nextSquare(s1, from) - from);
					do
					{
						if (can_piece_capture(pos, them, s2, to, new_pinned, new_slide))
							goto DISCOVER_ATTACK_CONTINUE;
						if (s2 != from && pos.piece_on(s2))
							break;
						if (can_pawn_drop<~Us>(pos, s2) || (capPawn && file_of(s2) == file_of(to)))
							goto DISCOVER_ATTACK_CONTINUE;
						Square s3 = nextSquare(s1, s2);
						s1 = s2;
						s2 = s3;
					} while (s2 != SQ_NB);
					return make_move(from, to);
				}

			DISCOVER_ATTACK_CONTINUE:;

			}
		}
	}

	// 持将棋の判定入れておくか…。
	// どうせ玉が入玉してないときはほとんど判定コストゼロだしな
	//  return pos.SengenGachi();
	// ↑この関数は、勝ちならばMOVE_NONE以外が返る。勝ちならMOVE_WINもしくは、勝ちになる指し手が返る。

	return MOVE_NONE;
}

Move Position::mate1ply() const
{
	return sideToMove == BLACK ? is_mate_in_1ply_imp<BLACK>(*this) : is_mate_in_1ply_imp<WHITE>(*this);
}


#endif // if defined(MATE_1PLY)...
