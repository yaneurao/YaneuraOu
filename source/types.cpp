﻿#include "types.h"
#include "usi.h"
#include "search.h"
#include "tt.h"

// ----------------------------------------
//    const
// ----------------------------------------

const char* USI_PIECE = ". P L N S B R G K +P+L+N+S+B+R+G+.p l n s b r g k +p+l+n+s+b+r+g+k";

// ----------------------------------------
//    tables
// ----------------------------------------

// これはtypes.hで定義しているのでLONG_EFFECT_LIBRARYがdefineされていないときにも必要。
namespace Effect8 {
	Directions direc_table[SQ_NB_PLUS1][SQ_NB_PLUS1];
}

std::string PieceToCharBW(" PLNSBRGK        plnsbrgk");

// ----------------------------------------
// operator<<(std::ostream& os,...)とpretty() 
// ----------------------------------------

std::string pretty(File f) { return pretty_jp ? std::string("１２３４５６７８９").substr((size_t)f * 2, 2) : std::to_string((size_t)f + 1); }
std::string pretty(Rank r) { return pretty_jp ? std::string("一二三四五六七八九").substr((size_t)r * 2, 2) : std::to_string((size_t)r + 1); }

std::string pretty(Move m)
{
	if (is_drop(m))
		return pretty(to_sq(m)  ) + pretty2(Piece(from_sq(m))) + (pretty_jp ? "打" : "*");
	else
		return pretty(from_sq(m)) + pretty(to_sq(m))           + (is_promote(m) ? (pretty_jp ? "成" : "+") : "");
}

std::string pretty(Move m, Piece movedPieceType)
{
	if (is_drop(m))
		return pretty(to_sq(m)) + pretty2(movedPieceType) + (pretty_jp ? "打" : "*");
	else
		return pretty(to_sq(m)) + pretty2(movedPieceType) + (is_promote(m) ? (pretty_jp ? "成" : "+") : "") + "[" + pretty(from_sq(m)) + "]";
}

std::string to_usi_string(Move   m){ return USI::move(m); }
std::string to_usi_string(Move16 m){ return USI::move(m); }

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
	for (PieceType pr = PAWN; pr < PIECE_HAND_NB; ++pr)
	{
		int c = hand_count(hand, pr);
		// 0枚ではないなら出力。
		if (c != 0)
		{
			// 1枚なら枚数は出力しない。2枚以上なら枚数を最初に出力
			// PRETTY_JPが指定されているときは、枚数は後ろに表示。
			const std::string cs = (c != 1) ? std::to_string(c) : "";
			os << (pretty_jp ? "" : cs) << pretty(pr) << (pretty_jp ? cs : "");
		}
	}
	return os;
}

// RepetitionStateを文字列化する。PVの出力のときにUSI拡張として出力するのに用いる。
std::string to_usi_string(RepetitionState rs)
{
#if !defined(PV_OUTPUT_DRAW_ONLY)
	return ((rs == REPETITION_NONE) ? "rep_none" : // これはデバッグ用であり、実際には出力はしない。
		(rs == REPETITION_WIN)      ? "rep_win" :
		(rs == REPETITION_LOSE)     ? "rep_lose" :
		(rs == REPETITION_DRAW)     ? "rep_draw" :
		(rs == REPETITION_SUPERIOR) ? "rep_sup" :
		(rs == REPETITION_INFERIOR) ? "rep_inf" :
		"")
		;
#else
	return "rep_draw";
#endif
}

// 拡張USIプロトコルにおいてPVの出力に用いる。
std::ostream& operator<<(std::ostream& os, RepetitionState rs)
{
	os << to_usi_string(rs);
	return os;
}

// ----------------------------------------
// 探索用のglobalな変数
// ----------------------------------------

namespace Search {
	LimitsType Limits;

	// Called in case we have no ponder move before exiting the search,
	// for instance, in case we stop the search during a fail high at root.
	// We try hard to have a ponder move to return to the GUI,
	// otherwise in case of 'ponder on' we have nothing to think about.

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
		TTEntry* tte = TT.read_probe(pos.state()->hash_key(), ttHit);
		Move m;
		if (ttHit)
		{
			m = pos.to_move(tte->move()); // SMP safeにするためlocal copy
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
