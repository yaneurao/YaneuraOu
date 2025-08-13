#include "types.h"
#include "usi.h"
#include "search.h"
#include "tt.h"

namespace YaneuraOu {

// ----------------------------------------
//    const
// ----------------------------------------

const char* USI_PIECE = ". P L N S B R G K +P+L+N+S+B+R+G+.p l n s b r g k +p+l+n+s+b+r+g+k";
const std::string StartSFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

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
	if (m.is_drop())
		return pretty(m.to_sq()  ) + pretty2(Piece(m.from_sq())) + (pretty_jp ? "打" : "*");
	else
		return pretty(m.from_sq()) + pretty(m.to_sq())           + (m.is_promote() ? (pretty_jp ? "成" : "+") : "");
}

std::string pretty(Move m, Piece movedPieceType)
{
	if (m.is_drop())
		return pretty(m.to_sq()) + pretty2(movedPieceType) + (pretty_jp ? "打" : "*");
	else
		return pretty(m.to_sq()) + pretty2(movedPieceType) + (m.is_promote() ? (pretty_jp ? "成" : "+") : "") + "[" + pretty(m.from_sq()) + "]";
}

std::string to_usi_string(Move   m){ return USIEngine::move(m); }
std::string to_usi_string(Move16 m){ return USIEngine::move(m); }

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

// エンジンオプションの入玉ルールに関する文字列
std::vector<std::string> EKR_STRINGS = {"NoEnteringKing", "CSARule24", "CSARule24H", "CSARule27",
                                        "CSARule27H",     "TryRule" /* , "EKR_NULL"*/};

// 文字列に対応するEnteringKingRuleを取得する。
EnteringKingRule to_entering_king_rule(const std::string& rule) {
    for (size_t i = 0; i < EKR_STRINGS.size(); ++i)
        if (EKR_STRINGS[i] == rule)
            return (EnteringKingRule) i;

    ASSERT(false);
    return EnteringKingRule::EKR_NONE;
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

Move Move::from_string(const Position& pos, const std::string usi_move) { return USIEngine::to_move(pos, usi_move); }

Move16 Move::to_move16() const { return Move16(data); }
Move16 Move16::from_string(const std::string usi_move) { return USIEngine::to_move16(usi_move); }


} // namespace YaneuraOu
