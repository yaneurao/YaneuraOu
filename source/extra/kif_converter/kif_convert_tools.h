#ifndef _KIF_CONVERT_TOOLS_H_
#define _KIF_CONVERT_TOOLS_H_

#include "../../shogi.h"

// 棋譜の変換などを行なうツールセット
// CSA,KIF,KI2形式などの入出力を担う。

#if defined(USE_KIF_CONVERT_TOOLS)

struct Position;
namespace KifConvertTools
{

	// --- CSA形式

	// CSA形式の指し手表現文字列を取得する。(手番有り)
	extern std::string to_csa_string(Position& pos, Move m);


	// --- UnitTest

	extern void UnitTest();

#if 0

	// KIF形式の文字列にする。
	extern std::string to_kif1_string(Move m, Piece movedPieceType, Color c, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);
	extern std::u32string to_kif1_u32string(Move m, Piece movedPieceType, Color c, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);


	// Squareを棋譜形式で出力する
	inline std::u32string kif_u32str(Square sq, SquareFormat fmt = SqFmt_ASCII)
	{
		char32_t r[3];
		r[0] = kif_char32(file_of(sq), fmt);
		r[1] = kif_char32(rank_of(sq), fmt);
		r[2] = (char32_t)NULL;
		return std::u32string(r);
	}


	// Pieceを日本語文字で出力する
	std::u32string kif_u32str(Piece pc);

	// PieceをCSA形式で出力する
	std::string csa(Piece pc);


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


#endif


}
#endif // defined (USE_KIF_CONVERT_TOOLS)


#endif // defined (_KIF_CONVERT_TOOLS_H_)
