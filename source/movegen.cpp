#include <iostream>

#include "shogi.h"
#include "position.h"


//   指し手生成ライブラリ

#include <iostream>
using namespace std;

// ----------------------------------
//      移動による指し手
// ----------------------------------

#ifndef KEEP_PIECE_IN_GENERATE_MOVES

// そのままの駒
#define OurPt(Us,Pt)     Move(0)

// 成るときの駒(+8)
#define OurProPt(Us,Pt)  Move(0)

// 駒打ちするときの駒(+32)
#define OurDropPt(Us,Pt) Move(0)


// 成らない指し手生成を生成する
#define MAKE_MOVE_TARGET(target_) { FOREACH_BB(target_,to,{ mlist++->move = make_move(from,to); }); }

// 成りと不成の指し手を生成する
#define MAKE_MOVE_TARGET_PRO(target_) { FOREACH_BB(target_,to,{ mlist++->move = make_move_promote(from,to); mlist++->move = make_move(from,to); }); }

// 成りの指し手のみを生成する
#define MAKE_MOVE_TARGET_PRO_ONLY(target_) { FOREACH_BB(target_,to,{ mlist++->move = make_move_promote(from,to); });}

// 成れることが確定しているときにAllのときだけ不成の指し手を生成する指し手生成(角、飛車用)
#define MAKE_MOVE_TARGET_PRO_BR(target_) { FOREACH_BB(target_,to,{ mlist++->move = make_move_promote(from,to); if (All) mlist++->move = make_move(from,to); }); }

// 駒打ち
#define MAKE_MOVE_TARGET_DROP(target_,Pt) FOREACH_BB(target_, sq, { mlist++->move = make_move_drop(Pt,sq); });

#define MAKE_MOVE_TARGET_UNKNOWN MAKE_MOVE_TARGET
#define MAKE_MOVE_TARGET_PRO_BR_UNKNOWN MAKE_MOVE_TARGET_PRO_BR

#else
// -----------------------------------------------
// 上位16bitに移動後の駒を格納しながらの指し手生成
// -----------------------------------------------

// 上位16bitに格納する駒を生成する。

// そのままの駒
#define OurPt(Us,Pt)     Move(((Us ? u32(PIECE_WHITE):0)+(Pt)) << 16)

// 成るときの駒(+8)
#define OurProPt(Us,Pt)  Move(((Us ? u32(PIECE_WHITE):0)+(Pt)+u32(PIECE_PROMOTE)) << 16)

// 駒打ちするときの駒(+32)
#define OurDropPt(Us,Pt) Move(((Us ? u32(PIECE_WHITE):0)+(Pt)+u32(PIECE_DROP)) << 16)

// 成らない指し手生成を生成する
#define MAKE_MOVE_TARGET(target_) { FOREACH_BB(target_,to,{                                      \
  mlist++->move = make_move(from,to) + OurPt(Us,Pt);                                             \
});}                                                                                          

// 移動させる駒が不明な場合                                                                    
#define MAKE_MOVE_TARGET_UNKNOWN(target_) { FOREACH_BB(target_,to,{                              \
  mlist++->move = make_move(from,to) + Move(pos.piece_on(from) << 16);                           \
});}                                                                                           

// 成りと不成の指し手を生成する                                                                
#define MAKE_MOVE_TARGET_PRO(target_) { FOREACH_BB(target_,to,{                                  \
  mlist++->move = make_move_promote(from,to) + OurProPt(Us,Pt);                                  \
  mlist++->move = make_move(from,to) + OurPt(Us,Pt);                                             \
});}                                                                                           

// 成りの指し手のみを生成する                                                                  
#define MAKE_MOVE_TARGET_PRO_ONLY(target_) { FOREACH_BB(target_,to,{                             \
  mlist++->move = make_move_promote(from,to) + OurProPt(Us,Pt);                                  \
});}                                                                                           

// 成れることが確定しているときにAllのときだけ不成の指し手を生成する指し手生成(角、飛車用)     
#define MAKE_MOVE_TARGET_PRO_BR(target_) { FOREACH_BB(target_,to,{                               \
  mlist++->move = make_move_promote(from,to) + OurProPt(Us,Pt);                                  \
  if (All) mlist++->move = make_move(from,to) + OurPt(Us,Pt);                                    \
});}

// ↑のと同じだが移動させる駒が確定していないとき。
#define MAKE_MOVE_TARGET_PRO_BR_UNKNOWN(target_) { FOREACH_BB(target_,to,{                       \
  mlist++->move = make_move_promote(from,to) + Move((pos.piece_on(from) + PIECE_PROMOTE) << 16); \
  if (All) mlist++->move = make_move(from,to) + Move(pos.piece_on(from) << 16)  ;                \
});}

// 駒打ち
#define MAKE_MOVE_TARGET_DROP(target_,Pt) { FOREACH_BB(target_, sq, {                           \
  mlist++->move = make_move_drop(Pt,sq) + OurDropPt(Us,Pt);                                     \
});}

#endif



// fromにあるpcをtargetの升に移動させる指し手の生成。
// 遅いので駒王手の指し手生成のときにしか使わない。
template <Piece Pt, Color Us, bool All> struct make_move_target {
	FORCE_INLINE ExtMove* operator()(const Position& pos, Square from, const Bitboard& target_, ExtMove* mlist)
	{
		Square to;
		Bitboard target = target_;
		Bitboard target2;

		switch (Pt)
		{
			// 成れるなら成りも生成するが、2,3段目への不成を生成するのはAllのときだけ。また1段目には不成で移動できない。
		case PAWN:
			if (target)
			{
				to = from + (Us == BLACK ? SQ_U : SQ_D); // to = target.pop(); より少し速い
				if (canPromote(Us, to))
				{
					mlist++->move = make_move_promote(from, to) + OurProPt(Us, Pt);
					if (All && rank_of(to) != (Us == BLACK ? RANK_1 : RANK_9))
						mlist++->move = make_move(from, to) + OurPt(Us, Pt);
				}
				else
					mlist++->move = make_move(from, to) + OurPt(Us, Pt);
			}
			break;

			// 成れるなら成りも生成するが、2段目への不成を生成するのはAllのときだけ。また1段目には不成で移動できない。
		case LANCE:
		{
			target2 = target & enemy_field(Us);
			MAKE_MOVE_TARGET_PRO_ONLY(target2);
			// 不成で移動する升
			target &= All ? (Us == BLACK ? InFrontBB[WHITE][RANK_1] : InFrontBB[BLACK][RANK_9]) :
				(Us == BLACK ? InFrontBB[WHITE][RANK_2] : InFrontBB[BLACK][RANK_8]);
			MAKE_MOVE_TARGET(target);
		}
		break;

		// 成れるなら成りも生成するが、1,2段目には不成で移動できない。
		case KNIGHT:
		{
			while (target)
			{
				to = target.pop();
				if (canPromote(Us, to))
					mlist++->move = make_move_promote(from, to) + OurProPt(Us, Pt);
				if ((Us == BLACK && rank_of(to) >= RANK_3) || (Us == WHITE && rank_of(to) <= RANK_7))
					mlist++->move = make_move(from, to) + OurPt(Us, Pt);
			}
		}
		break;

		// 成れるなら成りも生成する駒 
		case SILVER:
		{
			if (enemy_field(Us) & from) {
				// 敵陣からなら成れることは確定している
				MAKE_MOVE_TARGET_PRO(target);
			}
			else
			{
				// 非敵陣からなので敵陣への移動のみ成り
				target2 = target & enemy_field(Us);
				MAKE_MOVE_TARGET_PRO(target2);
				target &= ~enemy_field(Us);
				MAKE_MOVE_TARGET(target);
			}
		}
		break;

		// 成れない駒
		case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER: case HORSE: case DRAGON: case KING:
			MAKE_MOVE_TARGET(target);
			break;

			// 成れない駒。(対象の駒は不明)
		case GPM_GHDK:
			MAKE_MOVE_TARGET_UNKNOWN(target);
			break;

			// 成れるなら成る駒。ただしAllのときは不成も生成。
		case BISHOP: case ROOK:
			// 移動元が敵陣なら無条件で成れる
			if (canPromote(Us, from)) {
				MAKE_MOVE_TARGET_PRO_BR(target);
			}
			else
			{
				auto target2 = target & enemy_field(Us);
				MAKE_MOVE_TARGET_PRO_BR(target2);
				target &= ~enemy_field(Us);
				MAKE_MOVE_TARGET(target);
			}
			break;

			// 成れるなら成る駒。ただしAllのときは不成も生成。(対象は不明)
		case GPM_BR:
			if (canPromote(Us, from)) {
				MAKE_MOVE_TARGET_PRO_BR_UNKNOWN(target);
			}
			else
			{
				auto target2 = target & enemy_field(Us);
				MAKE_MOVE_TARGET_PRO_BR_UNKNOWN(target2);
				target &= ~enemy_field(Us);
				MAKE_MOVE_TARGET_UNKNOWN(target);
			}
			break;


		default: UNREACHABLE; break;
		}

		return mlist;
	}
};

// 指し手生成のうち、一般化されたもの。香・桂・銀はこの指し手生成を用いる。
template <MOVE_GEN_TYPE GenType, Piece Pt, Color Us, bool All> struct GeneratePieceMoves {
	FORCE_INLINE ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target) {
		// 盤上の駒pc(香・桂・銀)に対して
		auto pieces = pos.pieces(Us, Pt);
		const auto occ = pos.pieces();

		while (pieces)
		{
			auto from = pieces.pop();

			// 移動できる場所 = 利きのある場所
			auto target2 =
				Pt == LANCE ? lanceEffect(Us, from, occ) :
				Pt == KNIGHT ? knightEffect(Us, from) :
				Pt == SILVER ? silverEffect(Us, from) :
				ALL_BB; // error

			target2 &= target;
			mlist = make_move_target<Pt, Us, All>()(pos, from, target2, mlist);
		}

		return mlist;
	}
};

// 歩の移動による指し手生成
template <MOVE_GEN_TYPE GenType, Color Us, bool All> struct GeneratePieceMoves<GenType, PAWN, Us, All> {
	FORCE_INLINE ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target)
	{
		// 盤上の自駒の歩に対して
		auto pieces = pos.pieces(Us, PAWN);

		// 歩の利き
		auto target2 = (Us == BLACK ? shift<SQ_U>(pieces) : shift<SQ_D>(pieces)) & target;

		// 先手に対する1段目(後手ならば9段目)を表す定数
		const Rank T_RANK1 = (Us == BLACK) ? RANK_1 : RANK_9;

		while (target2)
		{
			auto to = target2.pop();
			auto from = to + (Us == BLACK ? SQ_D : SQ_U);
			// 歩が成れるときは成る指し手しか生成しない。
			if (canPromote(Us, to))
			{
				mlist++->move = make_move_promote(from, to) + OurProPt(Us, PAWN);

				// ただしAll(全合法手を生成するとき)だけは不成も生成
				// また、移動先の升が1段目は、成れないのでその指し手生成は除外
				if (All && rank_of(to) != T_RANK1)
					mlist++->move = make_move(from, to) + OurPt(Us, PAWN);
			}
			else
				mlist++->move = make_move(from, to) + OurPt(Us, PAWN);
		}
		return mlist;
	}
};

// 角・飛による移動による指し手生成。これらの駒は成れるなら絶対に成る
template <MOVE_GEN_TYPE GenType, Color Us, bool All> struct GeneratePieceMoves<GenType, GPM_BR, Us, All> {
	FORCE_INLINE ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target)
	{
		// 角と飛に対して(馬と龍は除く)
		auto pieces = pos.pieces(Us,BISHOP,ROOK);
		auto occ = pos.pieces();

		while (pieces)
		{
			auto from = pieces.pop();

			// fromの升にある駒をfromの升においたときの利き
			auto target2 = effects_from(pos.piece_on(from), from, occ) & target;

			mlist = make_move_target<GPM_BR, Us, All>()(pos, from, target2, mlist);
		}
		return mlist;
	}
};

// 成れない駒による移動による指し手。(金相当の駒・馬・龍・王)
template <MOVE_GEN_TYPE GenType, Color Us, bool All> struct GeneratePieceMoves<GenType, GPM_GHDK, Us, All> {
	FORCE_INLINE ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target)
	{
		// 金相当の駒・馬・龍・玉に対して
		auto pieces = pos.pieces(Us,GOLDS,HDK);
		auto occ = pos.pieces();

		while (pieces)
		{
			auto from = pieces.pop();
			// fromの升にある駒をfromの升においたときの利き
			auto target2 = effects_from(pos.piece_on(from), from, occ) & target;

			mlist = make_move_target<GPM_GHDK, Us, All>()(pos, from, target2, mlist);
		}
		return mlist;
	}
};

// 玉を除く成れない駒による移動による指し手。(金相当の駒・馬・龍)
template <MOVE_GEN_TYPE GenType, Color Us, bool All> struct GeneratePieceMoves<GenType, GPM_GHD, Us, All> {
	FORCE_INLINE ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target)
	{
		// 金相当の駒・馬・龍に対して
		auto pieces = pos.pieces(Us,GOLDS,HORSE,DRAGON);
		auto occ = pos.pieces();
		Square to;

		while (pieces)
		{
			auto from = pieces.pop();
			// fromの升にある駒をfromの升においたときの利き
			const auto Pt = pos.piece_on(from);
			auto target2 = effects_from(Pt, from, occ) & target;

			MAKE_MOVE_TARGET_UNKNOWN(target2);
		}
		return mlist;
	}
};


// ----------------------------------
//      駒打ちによる指し手
// ----------------------------------

// 駒打ちの指し手生成
template <Color Us> struct GenerateDropMoves {
	ExtMove* operator()(const Position&pos, ExtMove*mlist, const Bitboard& target) {

		const Hand hand = pos.hand_of(Us);
		// 手駒を持っていないならば終了
		if (hand == 0)
			return mlist;

		// 扱いにくいのでbit情報にしてしまう。
		const HandKind hk = toHandKind(hand);
		Square sq;

		// --- 歩を打つ指し手生成
		if (hand_exists(hk, PAWN))
		{
			// 歩の駒打ちの基本戦略
			// 1) 一段目以外に打てる
			// 2) 二歩のところには打てない
			// 3) 打ち歩詰め回避

			// ここでは2)のためにソフトウェア飽和加算に似たテクニックを用いる
			// cf. http://yaneuraou.yaneu.com/2015/10/15/%E7%B8%A6%E5%9E%8Bbitboard%E3%81%AE%E5%94%AF%E4%B8%80%E3%81%AE%E5%BC%B1%E7%82%B9%E3%82%92%E5%85%8B%E6%9C%8D%E3%81%99%E3%82%8B/
			// このときにテーブルを引くので、用意するテーブルのほうで先に1)の処理をしておく。

			// 縦型Squareにおける二歩のmaskの取得。
			// 各筋に1つしか歩はないので1～8段目が1になっているbitboardを歩の升のbitboardに加算すると9段目に情報が集まる。これをpextで回収する。

			// これにより、RANK9のところに歩の情報がかき集められた。
			Bitboard a = pos.pieces(Us, PAWN) + rank1_n_bb(BLACK, RANK_8); // 1～8段目を意味するbitboard

																		   // このRANK9に集まった情報をpextで回収。後者はPEXT32でもいいがlatencyたぶん変わらないので…。
			uint32_t index = uint32_t(PEXT64(a.p[0], RANK9_BB.p[0]) + (PEXT64(a.p[1], RANK9_BB.p[1]) << 7));

			// 駒の打てる場所
			Bitboard target2 = PAWN_DROP_MASK_BB[index][Us] & target;

			// 打ち歩詰めチェック
			// 敵玉に敵の歩を置いた位置に打つ予定だったのなら、打ち歩詰めチェックして、打ち歩詰めならそこは除外する。
			Bitboard pe = pawnEffect(~Us, pos.king_square(~Us));
			if (pe & target2)
			{
				Square to = pe.pop_c();
				if (!pos.legal_drop(to))
					target2 ^= pe;
			}

			// targetで表現される升に歩を打つ指し手の生成。
			MAKE_MOVE_TARGET_DROP(target2, PAWN);
		}

		// --- 歩以外を打つ指し手生成

		// 歩以外の手駒を持っているか
		if (hand_exceptPawnExists(hk))
		{
			Move drops[6];

			// 打つ先の升を埋めればいいだけの指し手を事前に生成しておく。
			// 基本的な戦略としては、(先手から見て)
			// 1) 1段目に打てない駒( = 香・桂)を除いたループ
			// 2) 2段目に打てない駒( = 桂)を除いたループ
			// 3) 3～9段目に打てる駒( = すべての駒)のループ
			// という3つの場合分けによるループで構成される。
			// そのため、手駒から香・桂を除いた駒と、桂を除いた駒が必要となる。

			int num = 0;
			if (hand_exists(hk, KNIGHT)) drops[num++] = make_move_drop(KNIGHT, SQ_ZERO) + OurDropPt(Us, KNIGHT);

			int nextToKnight = num; // 桂を除いたdropsのindex
			if (hand_exists(hk, LANCE)) drops[num++] = make_move_drop(LANCE, SQ_ZERO) + OurDropPt(Us, LANCE);

			int nextToLance = num; // 香・桂を除いたdropsのindex

			if (hand_exists(hk, SILVER)) drops[num++] = make_move_drop(SILVER, SQ_ZERO) + OurDropPt(Us, SILVER);
			if (hand_exists(hk, GOLD)) drops[num++] = make_move_drop(GOLD, SQ_ZERO) + OurDropPt(Us, GOLD);
			if (hand_exists(hk, BISHOP)) drops[num++] = make_move_drop(BISHOP, SQ_ZERO) + OurDropPt(Us, BISHOP);
			if (hand_exists(hk, ROOK)) drops[num++] = make_move_drop(ROOK, SQ_ZERO) + OurDropPt(Us, ROOK);


			// 以下、コードが膨れ上がるが、dropは比較的、数が多く時間がわりとかかるので展開しておく価値があるかと思う。
			// 動作ターゲットとするプロセッサにおいてbenchを取りながら進めるべき。
			// SSEを用いた高速化など色々考えられるところではあるが、とりあえず速度的に許容できる範囲で、最低限のコードを示す。

			if (nextToLance == 0)
			{
				// 香と桂を持っていないので駒を打てる全域に対して何も考えずに指し手を生成。
				Bitboard target2 = target;

				switch (num)
				{
				case 1: FOREACH_BB(target2, sq, { UNROLLER1({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 2: FOREACH_BB(target2, sq, { UNROLLER2({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 3: FOREACH_BB(target2, sq, { UNROLLER3({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 4: FOREACH_BB(target2, sq, { UNROLLER4({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				default: UNREACHABLE;
				}
			}
			else
			{
				// それ以外のケース

				Bitboard target1 = target & rank1_n_bb(Us, RANK_1); // 1段目
				Bitboard target2 = target & (Us == BLACK ? RANK2_BB : RANK8_BB); // 2段目
				Bitboard target3 = target & rank1_n_bb(~Us, RANK_7); // 3～9段目( == 後手から見たときの1～7段目)

				switch (num - nextToLance) // 1段目に対する香・桂以外の駒打ちの指し手生成(最大で4種の駒)
				{
				case 0: break; // 香・桂以外の持ち駒がないケース
				case 1: FOREACH_BB(target1, sq, { UNROLLER1({ mlist++->move = (Move)(drops[i + nextToLance] + sq); }); }); break;
				case 2: FOREACH_BB(target1, sq, { UNROLLER2({ mlist++->move = (Move)(drops[i + nextToLance] + sq); }); }); break;
				case 3: FOREACH_BB(target1, sq, { UNROLLER3({ mlist++->move = (Move)(drops[i + nextToLance] + sq); }); }); break;
				case 4: FOREACH_BB(target1, sq, { UNROLLER4({ mlist++->move = (Move)(drops[i + nextToLance] + sq); }); }); break;
				default: UNREACHABLE;
				}

				switch (num - nextToKnight) // 2段目に対する桂以外の駒打ちの指し手の生成(最大で5種の駒)
				{
				case 0: break; // 桂以外の持ち駒がないケース
				case 1: FOREACH_BB(target2, sq, { UNROLLER1({ mlist++->move = (Move)(drops[i + nextToKnight] + sq); }); }); break;
				case 2: FOREACH_BB(target2, sq, { UNROLLER2({ mlist++->move = (Move)(drops[i + nextToKnight] + sq); }); }); break;
				case 3: FOREACH_BB(target2, sq, { UNROLLER3({ mlist++->move = (Move)(drops[i + nextToKnight] + sq); }); }); break;
				case 4: FOREACH_BB(target2, sq, { UNROLLER4({ mlist++->move = (Move)(drops[i + nextToKnight] + sq); }); }); break;
				case 5: FOREACH_BB(target2, sq, { UNROLLER5({ mlist++->move = (Move)(drops[i + nextToKnight] + sq); }); }); break;
				default: UNREACHABLE;
				}

				switch (num) // 3～9段目に対する香を含めた指し手生成(最大で6種の駒)
				{
				case 1: FOREACH_BB(target3, sq, { UNROLLER1({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 2: FOREACH_BB(target3, sq, { UNROLLER2({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 3: FOREACH_BB(target3, sq, { UNROLLER3({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 4: FOREACH_BB(target3, sq, { UNROLLER4({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 5: FOREACH_BB(target3, sq, { UNROLLER5({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				case 6: FOREACH_BB(target3, sq, { UNROLLER6({ mlist++->move = (Move)(drops[i] + sq); }); }); break;
				default: UNREACHABLE;
				}
			}

		}

		return mlist;
	}
};


// 手番側が王手がかかっているときに、王手を回避する手を生成する。
template<Color Us, bool All>
ExtMove* generate_evasions(const Position& pos, ExtMove* mlist)
{
	// この実装において引数のtargetは無視する。

	// この関数を呼び出しているということは、王手がかかっているはずであり、
	// 王手をしている駒があるばすだから、checkers(王手をしている駒)が存在しなければおかしいのでassertを入れてある
	ASSERT_LV2(pos.in_check());

	// 自玉に王手をかけている敵の駒の利き(そこには玉は移動できないので)
	Bitboard sliderAttacks = ZERO_BB;

	// 王手している駒
	Bitboard checkers = pos.checkers();

	// 王手をしている駒の数を数えるカウンター
	int checkersCnt = 0;

	// 自玉を移動させるので、この玉はないものとして利きを求める必要がある。
	Square ksq = pos.king_square(Us);
	Bitboard occ = pos.pieces() ^ Bitboard(ksq);

	// 王手している駒のある升
	Square checksq;

	// 王手している駒は必ず1つ以上あるのでdo～whileで回る
	do
	{
		// 王手をしている敵の駒のカウントを加算
		++checkersCnt;

		// 王手をしている敵の駒の位置を取り出す
		checksq = checkers.pop();

		// この駒は敵駒でなくてはならない
		ASSERT_LV3(color_of(pos.piece_on(checksq)) == ~Us);

		// 王手している駒の利きを加えていく。
		sliderAttacks |= effects_from(pos.piece_on(checksq), checksq, occ);

	} while (checkers);

	// 王手回避のための玉の移動先は、玉の利きで、自駒のない場所でかつさきほどの王手していた駒が利いていないところが候補として挙げられる
	// これがまだ自殺手である可能性もあるが、それはis_legal()でチェックすればいいと思う。

	Bitboard bb = kingEffect(ksq) & ~(pos.pieces(Us) | sliderAttacks);
	while (bb) { Square to = bb.pop(); mlist++->move = make_move(ksq, to) + OurPt(Us, KING); }

	// 両王手であるなら、王の移動のみが回避手となる。ゆえにこれで指し手生成は終了。
	if (checkersCnt > 1)
		return mlist;

	// 両王手でないことは確定した

	// このあと生成すべきは
	// 1) 王手している駒を王以外で取る指し手
	// 2) 王手している駒と王の間に駒を移動させる指し手(移動合い)
	// 3) 王手している駒との間に駒を打つ指し手(合駒打ち)

	// target1 == 王手している駒と王との間の升 == 3)の駒打ちの場所
	// target2 == 移動による指し手は1)+2) = 王手している駒と王との間の升 + 王手している駒　の升

	const Bitboard target1 = between_bb(checksq, ksq);
	const Bitboard target2 = target1 | checksq;

	// あとはNON_EVASIONS扱いで普通に指し手生成。
	mlist = GeneratePieceMoves<NON_EVASIONS, PAWN, Us, All>()(pos, mlist, target2);
	mlist = GeneratePieceMoves<NON_EVASIONS, LANCE, Us, All>()(pos, mlist, target2);
	mlist = GeneratePieceMoves<NON_EVASIONS, KNIGHT, Us, All>()(pos, mlist, target2);
	mlist = GeneratePieceMoves<NON_EVASIONS, SILVER, Us, All>()(pos, mlist, target2);
	mlist = GeneratePieceMoves<NON_EVASIONS, GPM_BR, Us, All>()(pos, mlist, target2);
	mlist = GeneratePieceMoves<NON_EVASIONS, GPM_GHD, Us, All>()(pos, mlist, target2); // 玉は除かないといけない
	mlist = GenerateDropMoves<Us>()(pos, mlist, target1);

	return mlist;
}


// ----------------------------------
//      指し手生成器本体
// ----------------------------------

// 指し手の生成器本体
// mlist : 指し手を返して欲しい指し手生成バッファのアドレス
// Us : 生成するほうの手番
// All : 歩・香の2段目での不成や角・飛の不成などをすべて生成するのか。
// 返し値 : 生成した指し手の終端
// generateMovesのほうから内部的に呼び出される。(直接呼び出さないこと。)
template<MOVE_GEN_TYPE GenType, Color Us, bool All>
ExtMove* generate_general(const Position& pos, ExtMove* mlist, Square recapSq = SQ_NB)
{
	// --- 駒の移動による指し手

	// ・移動先の升。
	//  NON_CAPTURESなら駒のない場所
	//  CAPTURESなら敵駒のあるところ
	//  CAPTURE_PRO_PLUsならCAPTURES + 歩の成り。
	//   (価値のある成り以外はオーダリングを阻害するので含めない)

	static_assert(GenType != EVASIONS_ALL && GenType != NON_EVASIONS_ALL && GenType != RECAPTURES_ALL, "*_ALL is not allowed.");

	// 歩以外の駒の移動先
	const Bitboard target =
		(GenType == NON_CAPTURES)      ? pos.empties()      : // 捕獲しない指し手 = 移動先の升は駒のない升
		(GenType == CAPTURES)          ? pos.pieces(~Us)    : // 捕獲する指し手 = 移動先の升は敵駒のある升
		(GenType == NON_CAPTURES_PRO_MINUS) ? pos.empties() : // 捕獲しない指し手 - 歩の成る指し手 = 移動先の升は駒のない升 - 敵陣(歩のときのみ)
		(GenType == CAPTURES_PRO_PLUS) ? pos.pieces(~Us)    : // 捕獲 + 歩の成る指し手 = 移動先の升は敵駒のある升 + 敵陣(歩のときのみ)
		(GenType == NON_EVASIONS)      ? ~pos.pieces(Us)    : // すべて = 移動先の升は自駒のない升
		(GenType == RECAPTURES)        ? Bitboard(recapSq)  : // リキャプチャー用の升(直前で相手の駒が移動したわけだからここには移動できるはず)
		ALL_BB; // error

				// 歩の移動先(↑のtargetと違う部分のみをオーバーライド)
	const Bitboard targetPawn =
		(GenType == NON_CAPTURES_PRO_MINUS) ? (pos.empties() & ~enemy_field(Us)) : // 駒を取らない指し手 かつ、歩の成る指し手を引いたもの
		(GenType == CAPTURES_PRO_PLUS)      ? (pos.pieces(~Us) | (~pos.pieces(Us) & enemy_field(Us))) : // 歩の場合は敵陣での成りもこれに含める
		target;

	// 各駒による移動の指し手の生成
	// 歩の指し手は歩のBitboardをbit shiftすることで移動先が一発で求まるので特別扱い
	mlist = GeneratePieceMoves<GenType, PAWN, Us, All>()(pos, mlist, targetPawn);

	// 香・桂・銀は成れるなら成らない手の他に成る手も生成する駒。これらによる移動の指し手
	mlist = GeneratePieceMoves<GenType, LANCE, Us, All>()(pos, mlist, target);
	mlist = GeneratePieceMoves<GenType, KNIGHT, Us, All>()(pos, mlist, target);
	mlist = GeneratePieceMoves<GenType, SILVER, Us, All>()(pos, mlist, target);

	// 角・飛による移動による指し手生成。これらの駒は成れるなら絶対に成る
	mlist = GeneratePieceMoves<GenType, GPM_BR, Us, All>()(pos, mlist, target);

	// 金相当の駒・馬・龍・王による移動による指し手。(成れない駒による移動による指し手)
	mlist = GeneratePieceMoves<GenType, GPM_GHDK, Us, All>()(pos, mlist, target);

	// --- 駒打ち
	// →　オーダリング性能改善のためにDropをもう少し細分化できるといいのだが、なかなか簡単ではなさげ。
	if (GenType == NON_CAPTURES || GenType == NON_CAPTURES_PRO_MINUS || GenType == NON_EVASIONS)
		mlist = GenerateDropMoves<Us>()(pos, mlist, pos.empties());

	return mlist;
}

// -----------------------------------------------------
//     王手生成関係。(やねうら王nanoでは削除予定)
// -----------------------------------------------------

// make_move_targetを呼び出すための踏み台
// ptの駒をfromに置いたときの移動する指し手を生成する。ただし、targetで指定された升のみ。
template <Color Us, bool All> struct make_move_target_general {
	ExtMove* operator()(const Position& pos, Piece pc, Square from, const Bitboard& target, ExtMove* mlist)
	{
		ASSERT_LV2(pc != NO_PIECE);
		auto effect = effects_from(pc, from, pos.pieces());
		switch (type_of(pc))
		{
		case PAWN      : mlist = make_move_target<PAWN, Us, All>()(pos, from, effect & target, mlist); break;
		case LANCE     : mlist = make_move_target<LANCE, Us, All>()(pos, from, effect & target, mlist); break;
		case KNIGHT    : mlist = make_move_target<KNIGHT, Us, All>()(pos, from, effect & target, mlist); break;
		case SILVER    : mlist = make_move_target<SILVER, Us, All>()(pos, from, effect & target, mlist); break;
		case GOLD      : mlist = make_move_target<GOLD, Us, All>()(pos, from, effect & target, mlist); break;
		case BISHOP    : mlist = make_move_target<BISHOP, Us, All>()(pos, from, effect & target, mlist); break;
		case ROOK      : mlist = make_move_target<ROOK, Us, All>()(pos, from, effect & target, mlist); break;
		case KING      : mlist = make_move_target<KING, Us, All>()(pos, from, effect & target, mlist); break;
		case PRO_PAWN  : mlist = make_move_target<PRO_PAWN, Us, All>()(pos, from, effect & target, mlist); break;
		case PRO_LANCE : mlist = make_move_target<PRO_LANCE, Us, All>()(pos, from, effect & target, mlist); break;
		case PRO_KNIGHT: mlist = make_move_target<PRO_KNIGHT, Us, All>()(pos, from, effect & target, mlist); break;
		case PRO_SILVER: mlist = make_move_target<PRO_SILVER, Us, All>()(pos, from, effect & target, mlist); break;
		case HORSE     : mlist = make_move_target<HORSE, Us, All>()(pos, from, effect & target, mlist); break;
		case DRAGON    : mlist = make_move_target<DRAGON, Us, All>()(pos, from, effect & target, mlist); break;
		default: UNREACHABLE;
		}
		return mlist;
	}
};

// promoteかどうかを呼び出し元で選択できるmake_move_target
template <Piece Pt, Color Us, bool All, bool Promote>
ExtMove* make_move_target_pro(Square from, const Bitboard& target, ExtMove* mlist)
{
	auto bb = target;
	while (bb)
	{
		auto to = bb.pop();
		if (Promote)
			mlist++->move = make_move_promote(from, to) + OurProPt(Us, Pt);
		else
		{
			if (((Pt == PAWN) &&
				((!All && !canPromote(Us, to)) ||
				(All && rank_of(to) != (Us == BLACK ? RANK_1 : RANK_9))))
				|| ((Pt == LANCE) &&
				((!All && ((Us == BLACK && rank_of(to) >= RANK_3) || (Us == WHITE && rank_of(to) <= RANK_7))) ||
					(All && rank_of(to) != (Us == BLACK ? RANK_1 : RANK_9))))
				|| (Pt == KNIGHT && ((Us == BLACK && rank_of(to) >= RANK_3) || (Us == WHITE && rank_of(to) <= RANK_7)))
				|| (Pt == SILVER)
				|| ((Pt == BISHOP || Pt == ROOK) && (!(canPromote(Us, from) || canPromote(Us, to)) || All))
				)

				mlist++->move = make_move(from, to) + OurPt(Us, Pt);
		}
	}
	return mlist;
}

// pcをfromにおいたときにksqにいる敵玉に対して王手になる移動による指し手の生成
template <Color Us, bool All>
ExtMove* make_move_check(const Position& pos, Piece pc, Square from, Square ksq, const Bitboard& target, ExtMove* mlist)
{
	// Xで成るとYになる駒に関する王手になる指し手生成
	// 移動元が敵陣でないなら、移動先が敵陣でないと成れない。
#define GEN_MOVE_NONPRO_CHECK(X,X_Effect,Y_Effect) {                   \
    dst = X_Effect(Us, from) & Y_Effect(~Us, ksq) & target;            \
    if (!(enemy_field(Us) & from))                                     \
      dst &= enemy_field(Us);                                          \
    mlist = make_move_target_pro<X, Us, All, true>(from, dst, mlist);  \
    dst = X_Effect(Us, from) & X_Effect(~Us, ksq) & target;            \
    mlist = make_move_target_pro<X, Us, All, false>(from, dst, mlist); }

	// ↑のX==LANCEのとき
	// 同じ筋にある敵玉と香との間には一つ以上の駒があるはず(ないとしたら、玉が取れるので非合法局面)
	// この間にある駒が2個以上なら香の移動により王手にならない。1個でかつ、それが敵駒でなければ..
#define GEN_MOVE_LANCE_CHECK(X,X_Effect,Y_Effect) {                    \
    occ = pos.pieces();                                                \
    dst = X_Effect(Us, from,occ) & Y_Effect(~Us, ksq) & target;        \
    if (!(enemy_field(Us) & from))                                     \
      dst &= enemy_field(Us);                                          \
    mlist = make_move_target_pro<X, Us, All, true>(from, dst, mlist);  \
    if (file_of(from) == file_of(ksq) && !more_than_one(between_bb(from, ksq) & occ)){ \
      dst = pos.pieces(~Us) & between_bb(from, ksq) & target;            \
      mlist = make_move_target_pro<X, Us, All, false>(from, dst, mlist); \
    }}

	// ↑のBISHOP,ROOK用
#define GEN_MOVE_NONPRO_PRO_CHECK_BR(X,X_Effect,Y_Effect) {            \
  occ = pos.pieces();                                                  \
  dst = X_Effect(from,occ) & Y_Effect(ksq,occ) & target;               \
    if (!(enemy_field(Us) & from))                                     \
      dst &= enemy_field(Us);                                          \
    mlist = make_move_target_pro<X, Us, All, true>(from, dst, mlist);  \
    dst = X_Effect(from,occ) & X_Effect(ksq,occ) & target;             \
    mlist = make_move_target_pro<X, Us, All, false>(from, dst, mlist); }

	// ↑の成れない駒用
#define GEN_MOVE_GOLD_CHECK(X,X_Effect) {                              \
  dst = X_Effect(Us, from) & X_Effect(~Us, ksq) & target;              \
  mlist = make_move_target<X, Us, All>()(pos,from, dst, mlist); }

	// ↑のHORSE,DRAGON駒用
#define GEN_MOVE_HD_CHECK(X,X_Effect) {                                \
  occ = pos.pieces();                                                  \
  dst = X_Effect(from,occ) & X_Effect(ksq,occ) & target;               \
  mlist = make_move_target<X, Us, All>()(pos,from, dst, mlist); }

	Bitboard dst, occ;
	switch (type_of(pc))
	{
		// -- 成れる駒
	case PAWN      : GEN_MOVE_NONPRO_CHECK(PAWN, pawnEffect, goldEffect); break;
	case LANCE     : GEN_MOVE_LANCE_CHECK(LANCE, lanceEffect, goldEffect); break;
	case KNIGHT    : GEN_MOVE_NONPRO_CHECK(KNIGHT, knightEffect, goldEffect); break;
	case SILVER    : GEN_MOVE_NONPRO_CHECK(SILVER, silverEffect, goldEffect); break;
	case BISHOP    : GEN_MOVE_NONPRO_PRO_CHECK_BR(BISHOP, bishopEffect, horseEffect); break;
	case ROOK      : GEN_MOVE_NONPRO_PRO_CHECK_BR(ROOK, rookEffect, dragonEffect); break;

		// -- 成れない駒
	case PRO_PAWN  : GEN_MOVE_GOLD_CHECK(PRO_PAWN, goldEffect); break;
	case PRO_LANCE : GEN_MOVE_GOLD_CHECK(PRO_LANCE, goldEffect); break;
	case PRO_KNIGHT: GEN_MOVE_GOLD_CHECK(PRO_KNIGHT, goldEffect); break;
	case PRO_SILVER: GEN_MOVE_GOLD_CHECK(PRO_SILVER, goldEffect); break;
	case GOLD      : GEN_MOVE_GOLD_CHECK(GOLD, goldEffect); break;
	case HORSE     : GEN_MOVE_HD_CHECK(HORSE, horseEffect); break;
	case DRAGON    : GEN_MOVE_HD_CHECK(DRAGON, dragonEffect); break;

	default:UNREACHABLE;
	}
	return mlist;
}

// 王手になる駒打ち

template <Color Us, Piece Pt> struct GenerateCheckDropMoves {
	ExtMove* operator()(const Position& pos, const Bitboard& target, ExtMove* mlist)
	{
		auto bb = target;
		while (bb)
		{
			auto to = bb.pop();
			mlist++->move = make_move_drop(Pt, to) + OurDropPt(Us, Pt);
		}
		return mlist;
	}
};

// 歩だけ特殊化(2歩の判定/打ち歩詰め判定が必要なため)
template <Color Us> struct GenerateCheckDropMoves<Us, PAWN> {
	ExtMove* operator()(const Position& pos, const Bitboard& target, ExtMove* mlist)
	{
		auto bb = target;
		if (bb) // 歩を打って王手になる箇所は1箇所しかないのでwhileである必要はない。
		{
			auto to = bb.pop_c();

			// 二歩と打ち歩詰めでないならこの指し手を生成。
			if (pos.legal_pawn_drop(Us, to))
				mlist++->move = make_move_drop(PAWN, to) + OurDropPt(Us, PAWN);
		}
		return mlist;
	}
};


// 指し手の生成器本体(王手専用)
template<MOVE_GEN_TYPE GenType, Color Us, bool All>
ExtMove* generate_checks(const Position& pos, ExtMove* mlist)
{
	// --- 駒の移動による王手

	// 王手になる指し手
	//  1) 成らない移動による直接王手
	//  2) 成る移動による直接王手
	//  3) pinされている駒の移動による間接王手
	// 集合としては1),2) <--> 3)は被覆している可能性があるのでこれを除外できるような指し手生成をしなくてはならない。
	// これを綺麗に実装するのは結構難しい。

	// x = 直接王手となる候補
	// y = 間接王手となる候補

	// ほとんどのケースにおいて y == emptyなのでそれを前提に最適化をする。
	// yと、yを含まないxとに分けて処理する。
	// すなわち、y と (x | y)^y

	const Square themKing = pos.king_square(~Us);

	// 以下の方法だとxとして飛(龍)は100%含まれる。角・馬は60%ぐらいの確率で含まれる。事前条件でもう少し省ければ良いのだが…。
	const Bitboard x =
		(
			(pos.pieces(PAWN)   & check_candidate_bb(Us, PAWN  , themKing)) |
			(pos.pieces(LANCE)  & check_candidate_bb(Us, LANCE , themKing)) |
			(pos.pieces(KNIGHT) & check_candidate_bb(Us, KNIGHT, themKing)) |
			(pos.pieces(SILVER) & check_candidate_bb(Us, SILVER, themKing)) |
			(pos.pieces(GOLDS)  & check_candidate_bb(Us, GOLD  , themKing)) |
			(pos.pieces(BISHOP) & check_candidate_bb(Us, BISHOP, themKing)) |
			(pos.pieces(ROOK_DRAGON)) | // ROOK,DRAGONは無条件全域
			(pos.pieces(HORSE)  & check_candidate_bb(Us, ROOK, themKing)) // check_candidate_bbにはROOKと書いてるけど、HORSE
		) & pos.pieces(Us);

	// ここには王を敵玉の8近傍に移動させる指し手も含まれるが、王が近接する形はレアケースなので
	// 指し手生成の段階では除外しなくても良いと思う。

	const Bitboard y = pos.discovered_check_candidates();
	const Bitboard target =
		(GenType == CHECKS || GenType == CHECKS_ALL) ? ~pos.pieces(Us) :                     // 自駒がない場所が移動対象升
		(GenType == QUIET_CHECKS || GenType == QUIET_CHECKS_ALL) ? pos.empties() :           // 捕獲の指し手を除外するため駒がない場所が移動対象升
		ALL_BB; // Error!

				// yのみ。ただしxかつyである可能性もある。
	auto src = y;
	while (src)
	{
		auto from = src.pop();

		// 両王手候補なので指し手を生成してしまう。
		auto pin_line = line_bb(themKing, from); // いまの敵玉とfromを通る線上と違うところに移動させれば開き王手確定
		mlist = make_move_target_general<Us, All>()(pos, pos.piece_on(from), from, target & ~pin_line, mlist);

		if (x & from)
			// 直接王手にもなるので↑で生成した~line_bb以外の升への指し手を生成。
			mlist = make_move_check<Us, All>(pos, pos.piece_on(from), from, themKing, target & pin_line, mlist);
	}

	// yに被覆しないx
	src = (x | y) ^ y;
	while (src)
	{
		auto from = src.pop();

		// 直接王手のみ。
		mlist = make_move_check<Us, All>(pos, pos.piece_on(from), from, themKing, target, mlist);
	}

	// --- 駒打ちによる王手

	const Bitboard& empties = pos.empties();

	Hand h = pos.hand_of(Us);
	if (hand_exists(h, PAWN))
		mlist = GenerateCheckDropMoves<Us, PAWN>()(pos, pos.check_squares(PAWN) & empties, mlist);
	if (hand_exists(h, LANCE))
		mlist = GenerateCheckDropMoves<Us, LANCE>()(pos, pos.check_squares(LANCE) & empties, mlist);
	if (hand_exists(h, KNIGHT))
		mlist = GenerateCheckDropMoves<Us, KNIGHT>()(pos, pos.check_squares(KNIGHT) & empties, mlist);
	if (hand_exists(h, SILVER))
		mlist = GenerateCheckDropMoves<Us, SILVER>()(pos, pos.check_squares(SILVER) & empties, mlist);
	if (hand_exists(h, GOLD))
		mlist = GenerateCheckDropMoves<Us, GOLD>()(pos, pos.check_squares(GOLD) & empties, mlist);
	if (hand_exists(h, BISHOP))
		mlist = GenerateCheckDropMoves<Us, BISHOP>()(pos, pos.check_squares(BISHOP) & empties, mlist);
	if (hand_exists(h, ROOK))
		mlist = GenerateCheckDropMoves<Us, ROOK>()(pos, pos.check_squares(ROOK) & empties, mlist);

	return mlist;
}


// ----------------------------------
//      指し手生成踏み台
// ----------------------------------

// generate_general()を先後分けて実体化するための踏み台
template<MOVE_GEN_TYPE GenType, bool All>
ExtMove* generateMoves(const Position& pos, ExtMove* mlist, Square sq = SQ_NB)
{
	return pos.side_to_move() == BLACK ? generate_general<GenType, BLACK, All>(pos, mlist, sq) : generate_general<GenType, WHITE, All>(pos, mlist, sq);
}

// 同じく、Evasionsの指し手生成を呼ぶための踏み台
template<bool All>
ExtMove* generateEvasionMoves(const Position& pos, ExtMove* mlist)
{
	return pos.side_to_move() == BLACK ? generate_evasions<BLACK, All>(pos, mlist) : generate_evasions<WHITE, All>(pos, mlist);
}

// 同じく、Checksの指し手生成を呼ぶための踏み台
template<MOVE_GEN_TYPE GenType, bool All>
ExtMove* generateChecksMoves(const Position& pos, ExtMove* mlist)
{
	return pos.side_to_move() == BLACK ? generate_checks<GenType, BLACK, All>(pos, mlist) : generate_checks<GenType, WHITE, All>(pos, mlist);
}



// 一般的な指し手生成
template<MOVE_GEN_TYPE GenType>
ExtMove* generateMoves(const Position& pos, ExtMove* mlist, Square recapSq)
{
#ifdef CHECK_SHOGI_ENGINE
	// 王手将棋エンジンにおいては王手がかかっている局面は詰みであるから指し手は存在しない。
	if (pos.in_check())
		return mlist; // no moves
#endif

					  // すべての指し手を生成するのか。
	const bool All = (GenType == EVASIONS_ALL) || (GenType == CHECKS_ALL) || (GenType == LEGAL_ALL)
		|| (GenType == NON_EVASIONS_ALL) || (GenType == RECAPTURES_ALL) || (GenType == QUIET_CHECKS_ALL);

	if (GenType == LEGAL || GenType == LEGAL_ALL)
	{
#ifdef MUST_CAPTURE_SHOGI_ENGINE

		// captureの指し手を生成して、合法なcaptureが1つでもあるなら捕獲する指し手しか
		// 生成してはならない。
		bool mustCapture = false;
		bool inCheck = pos.in_check();
		auto endMoves = inCheck ? generateMoves<EVASIONS>(pos, mlist) : generateMoves<CAPTURES>(pos, mlist);
		for (auto it = mlist; it != endMoves; ++it)
		{
			if (pos.capture(it->move) && pos.legal(it->move))
			{
				mustCapture = true;
				break;
			}
		}
#endif

		// 合法性な指し手のみを生成する。
		// 自殺手や打ち歩詰めが含まれているのでそれを取り除く。かなり重い。ゆえにLEGALは特殊な状況でしか使うべきではない。
		auto last = pos.in_check() ? generateEvasionMoves<All>(pos, mlist) : generateMoves<NON_EVASIONS, All>(pos, mlist);

		// 合法ではない指し手を末尾の指し手と入れ替え
		while (mlist != last)
		{
			if (!pos.legal(*mlist)
#ifdef MUST_CAPTURE_SHOGI_ENGINE
				// 捕獲する指し手しか生成してはいけないモードなのに捕獲する指し手ではない
				|| (mustCapture && !pos.capture(*mlist))
#endif
				)
				mlist->move = (--last)->move;
			else
				++mlist;
		}
		return last;
	}

	// 王手生成
	if (GenType == CHECKS || GenType == CHECKS_ALL || GenType == QUIET_CHECKS || GenType == QUIET_CHECKS_ALL)
	{
		auto last = generateChecksMoves<GenType, All>(pos, mlist);

		// 王手がかかっている局面においては王手生成において、回避手になっていない指し手も含まれるので(王手放置での駒打ち等)
		// pseudo_legal()でない指し手はここで除外する。これはレアケースなので少々の無駄は許容する。
		if (pos.in_check())
			while (mlist != last)
			{
				if (!pos.pseudo_legal(*mlist))
					mlist->move = (--last)->move;
				else
					++mlist;
			}
		return last;
	}

	// 回避手
	if (GenType == EVASIONS || GenType == EVASIONS_ALL)
		return generateEvasionMoves<All>(pos, mlist);

	// 上記のもの以外
	// ただし、NON_EVASIONS_ALL , RECAPTURES_ALLは、ALLではないほうを呼び出す必要がある。
	// EVASIONS_ALLは上で呼び出されているが、実際はここでも実体化されたあと、最適化によって削除されるので、ここでも書く必要がある。
	const auto GenType2 =
		GenType == NON_EVASIONS_ALL ? NON_EVASIONS :
		GenType == RECAPTURES_ALL ? RECAPTURES :
		GenType == EVASIONS_ALL ? EVASIONS :
		GenType; // さもなくば元のまま。
	return generateMoves<GenType2, All>(pos, mlist, recapSq);
}

template<MOVE_GEN_TYPE GenType>
ExtMove* generateMoves(const Position& pos, ExtMove* mlist)
{
	static_assert(GenType != RECAPTURES && GenType != RECAPTURES_ALL, "RECAPTURES , not allowed.");
	return generateMoves<GenType>(pos, mlist, SQ_NB);
}


// テンプレートの実体化。これを書いておかないとリンクエラーになる。
// .h(ヘッダー)ではなく.cppのほうに書くことでコンパイル時間を節約できる。

template ExtMove* generateMoves<NON_CAPTURES          >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<CAPTURES              >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<NON_CAPTURES_PRO_MINUS>(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<CAPTURES_PRO_PLUS     >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<EVASIONS              >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<EVASIONS_ALL          >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<NON_EVASIONS          >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<NON_EVASIONS_ALL      >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<LEGAL                 >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<LEGAL_ALL             >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<CHECKS                >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<CHECKS_ALL            >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<QUIET_CHECKS          >(const Position& pos, ExtMove* mlist);
template ExtMove* generateMoves<QUIET_CHECKS_ALL      >(const Position& pos, ExtMove* mlist);

template ExtMove* generateMoves<RECAPTURES            >(const Position& pos, ExtMove* mlist, Square recapSq);
template ExtMove* generateMoves<RECAPTURES_ALL        >(const Position& pos, ExtMove* mlist, Square recapSq);
