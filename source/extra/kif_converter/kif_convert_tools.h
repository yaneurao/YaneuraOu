#ifndef _KIF_CONVERT_TOOLS_H_
#define _KIF_CONVERT_TOOLS_H_

#include "../../shogi.h"

// 棋譜の変換などを行なうツールセット
// CSA,KIF,KIF2(KI2)形式などの入出力を担う。

#if defined(USE_KIF_CONVERT_TOOLS)

struct Position;
namespace KifConvertTools
{

	// --- CSA形式

	// CSA形式の指し手表現文字列を取得する。(手番有り)
	extern std::string to_csa_string(Position& pos, Move m);

	// CSA形式の指し手表現文字列を取得する。(手番無し)
	// 「CSA1行形式」と呼ばれるものらしい。
	extern std::string to_csa1_string(Position& pos, Move m);

	// --- KIF/KIF2(KI2)形式

	// KIF形式のマス目の棋譜表記時の書式
	enum SquareFormat : int {
		SqFmt_ASCII           , // 普通のASCII形式
		SqFmt_FullWidthArabic , // 全角アラビア数字
		SqFmt_FullWidthMix    , // 全角漢数字・アラビア数字混在
	};

	// KIF形式の指し手表現文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif_string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::string to_kif_u8string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::u16string to_kif_u16string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::u32string to_kif_u32string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);

	// KIF2形式の指し手表現文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif2_string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::string to_kif2_u8string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::u16string to_kif2_u16string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);
	extern std::u32string to_kif2_u32string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);

	// --- UnitTest

	// 現状、テスト用のコードが書き散らしてあるだけ。
	extern void UnitTest();

}
#endif // defined (USE_KIF_CONVERT_TOOLS)


#endif // defined (_KIF_CONVERT_TOOLS_H_)
