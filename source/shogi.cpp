#include <sstream>
#include <iostream>

#include "shogi.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"

// ----------------------------------------
//    const
// ----------------------------------------

const char* USI_PIECE = ". P L N S B R G K +P+L+N+S+B+R+G+.p l n s b r g k +p+l+n+s+b+r+g+k";

// ----------------------------------------
//    tables
// ----------------------------------------

// これはshogi.hで定義しているのでLONG_EFFECT_LIBRARYがdefineされていないときにも必要。
namespace Effect8 { Directions direc_table[SQ_NB_PLUS1][SQ_NB_PLUS1]; }

File SquareToFile[SQ_NB_PLUS1] =
{
	FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1,
	FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2,
	FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3,
	FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4,
	FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5,
	FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6,
	FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7,
	FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8,
	FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9,
	FILE_NB, // 玉が盤上にないときにこの位置に移動させることがあるので
};

Rank SquareToRank[SQ_NB_PLUS1] = {
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
	RANK_NB, // 玉が盤上にないときにこの位置に移動させることがあるので
};

std::string PieceToCharBW(" PLNSBRGK        plnsbrgk");


// ----------------------------------------
// operator<<(std::ostream& os,...)とpretty() 
// ----------------------------------------

std::string pretty(File f) { return pretty_jp ? std::string("１２３４５６７８９").substr((int32_t)f * 2, 2) : std::to_string((int32_t)f + 1); }
std::string pretty(Rank r) { return pretty_jp ? std::string("一二三四五六七八九").substr((int32_t)r * 2, 2) : std::to_string((int32_t)r + 1); }

char32_t kif_char32(File f, SquareFormat fmt)
{
	switch (fmt)
	{
	case SqFmt_FullWidthArabic:
	case SqFmt_FullWidthMix:
		return U"１２３４５６７８９"[f];
	default:;
		return U"123456789"[f];
	}
}

char32_t kif_char32(Rank r, SquareFormat fmt)
{
	switch (fmt)
	{
	case SqFmt_FullWidthArabic:
		return U"１２３４５６７８９"[r];
	case SqFmt_FullWidthMix:
		return U"一二三四五六七八九"[r];
	default:;
		return U"123456789"[r];
	}
}

std::string pretty(Move m)
{
	if (is_drop(m))
		return (pretty(move_to(m)) + pretty2(Piece(move_from(m))) + (pretty_jp ? "打" : "*"));
	else
		return pretty(move_from(m)) + pretty(move_to(m)) + (is_promote(m) ? (pretty_jp ? "成" : "+") : "");
}

std::string pretty(Move m, Piece movedPieceType)
{
	if (is_drop(m))
		return (pretty(move_to(m)) + pretty2(movedPieceType) + (pretty_jp ? "打" : "*"));
	else
		return pretty(move_to(m)) + pretty2(movedPieceType) + (is_promote(m) ? (pretty_jp ? "成" : "+") : "") + "[" + pretty(move_from(m)) + "]";
}

std::ostream& operator<<(std::ostream& os, Color c) { os << ((c == BLACK) ? (pretty_jp ? "先手" : "BLACK") : (pretty_jp ? "後手" : "WHITE")); return os; }

std::ostream& operator<<(std::ostream& os, Piece pc)
{
	auto s = usi_piece(pc);
	if (s[1] == ' ') s.resize(1); // 手動trim
	os << s;
	return os;
}

std::ostream& operator<<(std::ostream& os, Hand hand)
{
	for (Piece pr = PAWN; pr < PIECE_HAND_NB; ++pr)
	{
		int c = hand_count(hand, pr);
		// 0枚ではないなら出力。
		if (c != 0)
		{
			// 1枚なら枚数は出力しない。2枚以上なら枚数を最初に出力
			// PRETTY_JPが指定されているときは、枚数は後ろに表示。
			const std::string cs = (c != 1) ? std::to_string(c) : "";
			std::cout << (pretty_jp ? "" : cs) << pretty(pr) << (pretty_jp ? cs : "");
		}
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, HandKind hk)
{
	for (Piece pc = PAWN; pc < PIECE_HAND_NB; ++pc)
		if (hand_exists(hk, pc))
			std::cout << pretty(pc);
	return os;
}

std::string to_usi_string(Move m)
{
	std::stringstream ss;
	if (!is_ok(m))
	{
		ss << ((m == MOVE_RESIGN) ? "resign" :
			   (m == MOVE_WIN)    ? "win"    :
			   (m == MOVE_NULL)   ? "null"   :
			   (m == MOVE_NONE)   ? "none"   :
			"");
	}
	else if (is_drop(m))
	{
		ss << move_dropped_piece(m);
		ss << '*';
		ss << move_to(m);
	}
	else {
		ss << move_from(m);
		ss << move_to(m);
		if (is_promote(m))
			ss << '+';
	}
	return ss.str();
}
void kiftoc32(char32_t ** s, Piece p)
{
	char32_t * c = *s;
	if (p & 8)
		switch (p & 7)
		{
		case 0:
			*c++ = U'玉'; break;
		case 1:
			*c++ = U'と'; break;
		case 5:
			*c++ = U'馬'; break;
		case 6:
			*c++ = U'龍'; break;
		default:;
			*c++ = U'成';
			*c++ = U"玉と香桂銀馬龍金"[p & 7];
		}
	else
		*c++ = U"　歩香桂銀角飛金玉"[p & 7];
	*s = c;
}
void kiftoc32(char32_t ** s, Square sq, SquareFormat fmt)
{
	char32_t * c = *s;
	*c++ = kif_char32(file_of(sq), fmt);
	*c++ = kif_char32(rank_of(sq), fmt);
	*s = c;
}
void to_kif1_c32(char32_t ** s, Move m, Piece movedPieceType, Color c, Move prev_m, SquareFormat fmt)
{
	char32_t * p = *s;
	*p++ = ((~c) ? U'▲' : U'△');
	if (!is_ok(m))
	{
		const std::u32string _none(U"エラー");
		const std::u32string _null(U"パス");
		const std::u32string _resign(U"投了");
		const std::u32string _win(U"勝ち宣言");
		switch (m) {
		case MOVE_NONE:
			std::char_traits<char32_t>::copy(p, _none.c_str(), _none.size());
			p += _none.size();
			break;
		case MOVE_NULL:
			std::char_traits<char32_t>::copy(p, _null.c_str(), _null.size());
			p += _null.size();
			break;
		case MOVE_RESIGN:
			std::char_traits<char32_t>::copy(p, _resign.c_str(), _resign.size());
			p += _resign.size();
			break;
		case MOVE_WIN:
			std::char_traits<char32_t>::copy(p, _win.c_str(), _win.size());
			p += _win.size();
			break;
		default:;
		}
	}
	else
	{
		if (is_ok(prev_m) && move_to(prev_m) == move_to(m))
		{
			*p++ = U'同';
			kiftoc32(&p, movedPieceType);
		}
		else
		{
			kiftoc32(&p, move_to(m), fmt);
			kiftoc32(&p, movedPieceType);
		}
		if (is_drop(m))
			*p++ = U'打';
		else if (is_promote(m))
			*p++ = U'成';
		else
		{
			Square from_sq = move_from(m);
			*p++ = U'(';
			*p++ = U"123456789"[file_of(from_sq)];
			*p++ = U"123456789"[rank_of(from_sq)];
			*p++ = U'(';
		}
	}
	*p = U'\0';
	*s = p;
}
void to_kif1_c32(char32_t ** s, Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	return to_kif1_c32(s, m, pos.moved_piece_before(m), pos.side_to_move(), prev_m, fmt);
}
std::u32string to_kif1_u32string(Move m, Piece movedPieceType, Color c, Move prev_m, SquareFormat fmt)
{
	char32_t r[32] = {};
	char32_t * p = r;
	to_kif1_c32(&p, m, movedPieceType, c, prev_m, fmt);
	return std::u32string(r);
}
std::u32string to_kif1_u32string(Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	char32_t r[32] = {};
	char32_t * p = r;
	to_kif1_c32(&p, m, pos, prev_m, fmt);
	return std::u32string(r);
}
std::string to_kif1_string(Move m, Piece movedPieceType, Color c, Move prev_m, SquareFormat fmt)
{
	char32_t r[32] = {};
	char32_t * p = r;
	to_kif1_c32(&p, m, movedPieceType, c, prev_m, fmt);
	return UniConv::char32_to_utf8string(r);
}
std::string to_kif1_string(Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	char32_t r[32] = {};
	char32_t * p = r;
	to_kif1_c32(&p, m, pos, prev_m, fmt);
	return UniConv::char32_to_utf8string(r);
}
void to_kif2_c32(char32_t ** r, Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	char32_t * s = *r;
	Color c = pos.side_to_move();
	*s++ = ((~c) ? U'▲' : U'△');
	if (!is_ok(m))
	{
		const std::u32string _none(U"エラー");
		const std::u32string _null(U"パス");
		const std::u32string _resign(U"投了");
		const std::u32string _win(U"勝ち宣言");
		switch (m) {
		case MOVE_NONE:
			std::char_traits<char32_t>::copy(s, _none.c_str(), _none.size());
			s += _none.size();
			break;
		case MOVE_NULL:
			std::char_traits<char32_t>::copy(s, _null.c_str(), _null.size());
			s += _null.size();
			break;
		case MOVE_RESIGN:
			std::char_traits<char32_t>::copy(s, _resign.c_str(), _resign.size());
			s += _resign.size();
			break;
		case MOVE_WIN:
			std::char_traits<char32_t>::copy(s, _win.c_str(), _win.size());
			s += _win.size();
			break;
		default:;
		}
	}
	else
	{
		// 先後・成の属性を含めた駒種別
		Piece p = pos.moved_piece_before(m);
		// 先後の属性を排除した駒種別
		Piece p_type = type_of(p);
		// 金・銀・成金は直の表記が有り得る
		bool is_like_goldsilver = (
			p_type == SILVER || 
			p_type == GOLD || 
			p_type == PRO_PAWN || 
			p_type == PRO_LANCE || 
			p_type == PRO_KNIGHT || 
			p_type == PRO_SILVER);
		// 移動元・移動先の座標
		Square fromSq = move_from(m), toSq = move_to(m);
		File fromSqF = file_of(fromSq), toSqF = file_of(toSq);
		Rank fromSqR = rank_of(fromSq), toSqR = rank_of(toSq);
		Bitboard fromB = SquareBB[fromSq];
		// 移動先地点への同駒種の利き（駒打ちで無ければ指し手の駒も含む）
		// Position.piecesの第2引数には先後の属性を含めてはいけない。
		Bitboard atkB = (pos.attackers_to(c, toSq) & pos.pieces(c, p_type));

		// 移動先座標
		if (is_ok(prev_m) && move_to(prev_m) == toSq)
			*s++ = U'同';
		else
			kiftoc32(&s, toSq, fmt);
		// 駒種
		kiftoc32(&s, p);
		// 打ち駒ではないかどうか
		if (!is_drop(m)) {
			// toSqに動ける同種駒が他にもあるなら
			if (atkB != fromB) {
				// 真横に寄るMoveの場合
				if (fromSqR == toSqR) {
					// 真横に寄る同種駒がそれだけなら「寄」
					if ((atkB & RANK_BB[toSqR]) == fromB)
						*s++ = U'寄';
					// 右から寄るなら「右」
					else if ((fromB & InRightBB[c][toSqF]) == fromB) {
						*s++ = U'右';
						// 右から動く駒がそれだけでは無いならさらに「寄」
						if ((atkB & InRightBB[c][toSqF]) != fromB)
							*s++ = U'寄';
					}
					// 左から寄るなら「左」
					else if ((fromB & InLeftBB[c][toSqF]) == fromB){
						*s++ = U'左';
						// 左から動く駒がそれだけでは無いならさらに「寄」
						if ((atkB & InLeftBB[c][toSqF]) != fromB)
							*s++ = U'寄';
					} 
				}
				// 前に上がるMoveの場合
				else if ((fromB & InBackBB[c][toSqR]) == fromB) {
					// 前に上がる同種駒がそれだけなら「上」
					if ((atkB & InBackBB[c][toSqR]) == fromB)
						*s++ = U'上';
					// 真っ直ぐ上がるMoveの場合
					else if (fromSqF == toSqF) {
						// 金・銀・成金なら「直」
						if (is_like_goldsilver)
							*s++ = U'直';
						// 同じ筋・より右の筋に他に動ける駒がないなら「右」
						else if ((atkB & OrRightBB[c][toSqF]) == fromB)
							*s++ = U'右';
						// 同じ筋・より左の筋に他に動ける駒がないなら「左」
						else if ((atkB & OrLeftBB[c][toSqF]) == fromB)
							*s++ = U'左';
						// 「右上」の判定
						else if ((atkB & OrRightBB[c][toSqF] & InBackBB[c][toSqR]) == fromB) {
							*s++ = U'右';
							*s++ = U'上';
						}
						// 「左上」の判定
						else if ((atkB & OrLeftBB[c][toSqF] & InBackBB[c][toSqR]) == fromB) {
							*s++ = U'左';
							*s++ = U'上';
						}
					}
					// 右から上がるMoveの場合
					else if ((fromB & InRightBB[c][toSqF]) == fromB) {
						*s++ = U'右';
						// 右から動ける駒が他にもあればさらに「上」
						if ((atkB & InRightBB[c][toSqF]) != fromB)
							*s++ = U'上';
					}
					// 左から上がるMoveの場合
					else if ((fromB & InLeftBB[c][toSqF]) == fromB) {
						*s++ = U'左';
						// 左から動ける駒が他にもあればさらに「上」
						if ((atkB & InLeftBB[c][toSqF]) != fromB)
							*s++ = U'上';
					}
				}
				// 後ろに引くMoveの場合
				else if ((fromB & InFrontBB[c][toSqR]) == fromB) {
					// 後ろに引く同種駒がそれだけなら
					if ((atkB & InFrontBB[c][toSqR]) == fromB)
						*s++ = U'引';
					// 真っ直ぐ引くMoveの場合
					else if (fromSqF == toSqF) {
						// 同じ筋・より右の筋に他に動ける駒がないなら「右」
						if ((atkB & OrRightBB[c][toSqF]) == fromB)
							*s++ = U'右';
						// 同じ筋・より左の筋に他に動ける駒がないなら「左」
						else if ((atkB & OrLeftBB[c][toSqF]) == fromB)
							*s++ = U'左';
						// 「右引」の判定
						else if ((atkB & OrRightBB[c][toSqF] & InFrontBB[c][toSqR]) == fromB) {
							*s++ = U'右';
							*s++ = U'引';
						}
						// 「左引」の判定
						else if ((atkB & OrLeftBB[c][toSqF] & InFrontBB[c][toSqR]) == fromB) {
							*s++ = U'左';
							*s++ = U'引';
						}
					}
					// 右から引くMoveの場合
					else if ((fromB & InRightBB[c][toSqF]) == fromB) {
						*s++ = U'右';
						// 右から動ける駒が他にもあればさらに「引」
						if ((atkB & InRightBB[c][toSqF]) != fromB)
							*s++ = U'引';
					}
					// 左から引くMoveの場合
					else if ((fromB & InLeftBB[c][toSqF]) == fromB) {
						*s++ = U'左';
						// 左から動ける駒が他にもあればさらに「引」
						if ((atkB & InLeftBB[c][toSqF]) != fromB)
							*s++ = U'引';
					}
				}
			}
			// 成ったなら必ず「成」
			if (is_promote(m))
				*s++ = U'成';
			// 成れるのに成らなかったなら「不成」
			else if (p_type < GOLD && canPromote(c, fromSq, toSq))
			{
				*s++ = U'不';
				*s++ = U'成';
			}
		}
		// そこへ移動できる同種駒があるなら「打」
		else if (atkB != ZERO_BB)
			*s++ = U'打';
	}
	*s = U'\0';
	*r = s;
}
std::string to_kif2_string(Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	char32_t r[10] = {};
	char32_t * p = r;
	to_kif2_c32(&p, m, pos, prev_m, fmt);
	return UniConv::char32_to_utf8string(r);
}
std::u32string to_kif2_u32string(Move m, Position& pos, Move prev_m, SquareFormat fmt)
{
	char32_t r[10] = {};
	char32_t * p = r;
	to_kif2_c32(&p, m, pos, prev_m, fmt);
	return std::u32string(r);
}
void to_csa1_string(char ** s, Move m, Piece movedPieceAfterType)
{
	if (!is_ok(m))
	{
		**s = '\0';
		return;
	}
	char * p = *s;
	if (is_drop(m))
	{
		*p++ = '0';
		*p++ = '0';
	}
	else
	{
		Square from_sq = move_from(m);
		*p++ = "123456789"[file_of(from_sq)];
		*p++ = "123456789"[rank_of(from_sq)];
	}
	Square to_sq = move_to(m);
	*p++ = "123456789"[file_of(to_sq)];
	*p++ = "123456789"[rank_of(to_sq)];
	*p++ = "*FKKGKHKOTNNNURKOFKKGKHKOTNNNURKO"[movedPieceAfterType];
	*p++ = "*UYEIAIIUOYKGMYIUUYEIAIIUOYKGMYIU"[movedPieceAfterType];
	*p = '\0';
	*s = p;
}
std::string to_csa1_string(Move m, Piece movedPieceAfterType)
{
	char s[8] = {};
	char * p = s;
	to_csa1_string(&p, m, movedPieceAfterType);
	return std::string(s);
}
std::string to_csa1_string(Move m, Position& pos)
{
	char s[8] = {};
	char * p = s;
	to_csa1_string(&p, m, pos.moved_piece_after(m));
	return std::string(s);
}
void to_csa_string(char ** s, Move m, Piece movedPieceAfterType, Color c)
{
	char * p = *s;
	switch (m)
	{
	case MOVE_NONE:
	case MOVE_NULL:
		*p++ = '%';
		*p++ = 'E';
		*p++ = 'R';
		*p++ = 'R';
		*p++ = 'O';
		*p++ = 'R';
		*p = '\0';
		*s = p;
		return;
	case MOVE_RESIGN:
		*p++ = '%';
		*p++ = 'T';
		*p++ = 'O';
		*p++ = 'R';
		*p++ = 'Y';
		*p++ = 'O';
		*p = '\0';
		*s = p;
		return;
	case MOVE_WIN:
		*p++ = '%';
		*p++ = 'W';
		*p++ = 'I';
		*p++ = 'N';
		*p = '\0';
		*s = p;
		return;
	default:;
	}
	*p++ = ~c ? '+' : '-';
	if (is_drop(m))
	{
		*p++ = '0';
		*p++ = '0';
	}
	else {
		Square from_sq = move_from(m);
		*p++ = "123456789"[file_of(from_sq)];
		*p++ = "123456789"[rank_of(from_sq)];
	}
	Square to_sq = move_to(m);
	*p++ = "123456789"[file_of(to_sq)];
	*p++ = "123456789"[rank_of(to_sq)];
	*p++ = "*FKKGKHKOTNNNURKOFKKGKHKOTNNNURKO"[movedPieceAfterType];
	*p++ = "*UYEIAIIUOYKGMYIUUYEIAIIUOYKGMYIU"[movedPieceAfterType];
	*p = '\0';
	*s = p;
}
std::string to_csa_string(Move m, Piece movedPieceAfterType, Color c)
{
	char s[8] = {};
	char * p = s;
	to_csa_string(&p, m, movedPieceAfterType, c);
	return std::string(s);
}
std::string to_csa_string(Move m, Position& pos)
{
	char s[8] = {};
	char * p = s;
	to_csa_string(&p, m, pos.moved_piece_after(m), pos.side_to_move());
	return std::string(s);
}

// ----------------------------------------
// 探索用のglobalな変数
// ----------------------------------------

namespace Search {
	SignalsType Signals;
	LimitsType Limits;
	StateStackPtr SetupStates;

	// 探索を抜ける前にponderの指し手がないとき(rootでfail highしているだとか)にこの関数を呼び出す。
	// ponderの指し手として何かを指定したほうが、その分、相手の手番において考えられて得なので。

	bool RootMove::extract_ponder_from_tt(Position& pos, Move ponder_candidate)
	{
		StateInfo st;
		bool ttHit;

		//    ASSERT_LV3(pv.size() == 1);

		// 詰みの局面が"ponderhit"で返ってくることがあるので、ここでのpv[0] == MOVE_RESIGNであることがありうる。
		if (!is_ok(pv[0]))
			return false;

		pos.do_move(pv[0], st, pos.gives_check(pv[0]));
		TTEntry* tte = TT.probe(pos.state()->key(), ttHit);
		Move m;
		if (ttHit)
		{
			m = tte->move(); // SMP safeにするためlocal copy
			if (MoveList<LEGAL_ALL>(pos).contains(m))
				goto FOUND;
		}
		// 置換表にもなかったので以前のiteration時のpv[1]をほじくり返す。
		m = ponder_candidate;
		if (MoveList<LEGAL_ALL>(pos).contains(m))
			goto FOUND;

		pos.undo_move(pv[0]);
		return false;
	FOUND:;
		pos.undo_move(pv[0]);
		pv.push_back(m);
		//    std::cout << m << std::endl;
		return true;
	}
}

// 引き分け時のスコア(とそのdefault値)
Value drawValueTable[REPETITION_NB][COLOR_NB] =
{
	{  VALUE_ZERO        ,  VALUE_ZERO        }, // REPETITION_NONE
	{  VALUE_MATE        ,  VALUE_MATE        }, // REPETITION_WIN
	{ -VALUE_MATE        , -VALUE_MATE        }, // REPETITION_LOSE
	{  VALUE_ZERO        ,  VALUE_ZERO        }, // REPETITION_DRAW  : このスコアはUSIのoptionコマンドで変更可能
	{  VALUE_SUPERIOR    ,  VALUE_SUPERIOR    }, // REPETITION_SUPERIOR
	{ -VALUE_SUPERIOR    , -VALUE_SUPERIOR    }, // REPETITION_INFERIOR
};

#if defined(USE_GLOBAL_OPTIONS)
GlobalOptions_ GlobalOptions;
#endif

// ----------------------------------------
//  main()
// ----------------------------------------

int main(int argc, char* argv[])
{
	// --- 全体的な初期化
	USI::init(Options);
	Bitboards::init();
	Position::init();
	Search::init();
	Threads.init();
	// 簡単な初期化のみで評価関数の読み込みはisreadyに応じて行なう。
	Eval::init();

	// USIコマンドの応答部
	USI::loop(argc, argv);

	// 生成して、待機させていたスレッドの停止
	Threads.exit();

	return 0;
}
