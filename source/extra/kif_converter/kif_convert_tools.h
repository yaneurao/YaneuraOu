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

	// --- KIF形式

	// KIF形式のマス目の棋譜表記時の書式
	enum SquareFormat : int {
		SqFmt_ASCII           , // 普通のASCII形式
		SqFmt_FullWidthArabic , // 全角アラビア数字
		SqFmt_FullWidthMix    , // 全角漢数字・アラビア数字混在
	};

	// KIF形式の指し手表現文字列を取得する。
	extern std::string to_kif_string(Position& pos, Move m, SquareFormat fmt = SqFmt_ASCII);


	// --- UnitTest

	extern void UnitTest();


	// 以下、作業中。
#if 0


	Bitboard InBackBB[COLOR_NB][RANK_NB] = {
		{ ~RANK1_BB, ~(RANK1_BB | RANK2_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
		RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB, RANK9_BB | RANK8_BB | RANK7_BB, RANK9_BB | RANK8_BB, RANK9_BB, ZERO_BB },
		{ ZERO_BB, RANK1_BB, RANK1_BB | RANK2_BB, RANK1_BB | RANK2_BB | RANK3_BB, RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
		~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB), ~(RANK9_BB | RANK8_BB | RANK7_BB), ~(RANK9_BB | RANK8_BB), ~RANK9_BB }
	};
	Bitboard InLeftBB[COLOR_NB][FILE_NB] = {
		{ ~FILE1_BB, ~(FILE1_BB | FILE2_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB),
		FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB, FILE9_BB | FILE8_BB | FILE7_BB, FILE9_BB | FILE8_BB, FILE9_BB, ZERO_BB },
		{ ZERO_BB, FILE1_BB, FILE1_BB | FILE2_BB, FILE1_BB | FILE2_BB | FILE3_BB, FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB,
		~(FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB), ~(FILE9_BB | FILE8_BB | FILE7_BB), ~(FILE9_BB | FILE8_BB), ~FILE9_BB }
	};
	Bitboard InRightBB[COLOR_NB][FILE_NB] = {
		{ ZERO_BB, FILE1_BB, FILE1_BB | FILE2_BB, FILE1_BB | FILE2_BB | FILE3_BB, FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB,
		~(FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB), ~(FILE9_BB | FILE8_BB | FILE7_BB), ~(FILE9_BB | FILE8_BB), ~FILE9_BB },
		{ ~FILE1_BB, ~(FILE1_BB | FILE2_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB),
		FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB, FILE9_BB | FILE8_BB | FILE7_BB, FILE9_BB | FILE8_BB, FILE9_BB, ZERO_BB }
	};
	Bitboard OrFrontBB[COLOR_NB][RANK_NB] = {
		{ RANK1_BB, RANK1_BB | RANK2_BB, RANK1_BB | RANK2_BB | RANK3_BB, RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
		~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB), ~(RANK9_BB | RANK8_BB | RANK7_BB), ~(RANK9_BB | RANK8_BB), ~RANK9_BB, ALL_BB },
		{ ALL_BB, ~RANK1_BB, ~(RANK1_BB | RANK2_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
		RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB, RANK9_BB | RANK8_BB | RANK7_BB, RANK9_BB | RANK8_BB, RANK9_BB }
	};
	Bitboard OrBackBB[COLOR_NB][RANK_NB] = {
		{ ALL_BB, ~RANK1_BB, ~(RANK1_BB | RANK2_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB), ~(RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB),
		RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB, RANK9_BB | RANK8_BB | RANK7_BB, RANK9_BB | RANK8_BB, RANK9_BB },
		{ RANK1_BB, RANK1_BB | RANK2_BB, RANK1_BB | RANK2_BB | RANK3_BB, RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
		~(RANK9_BB | RANK8_BB | RANK7_BB | RANK6_BB), ~(RANK9_BB | RANK8_BB | RANK7_BB), ~(RANK9_BB | RANK8_BB), ~RANK9_BB, ALL_BB }
	};
	Bitboard OrLeftBB[COLOR_NB][FILE_NB] = {
		{ ALL_BB, ~FILE1_BB, ~(FILE1_BB | FILE2_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB),
		FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB, FILE9_BB | FILE8_BB | FILE7_BB, FILE9_BB | FILE8_BB, FILE9_BB },
		{ FILE1_BB, FILE1_BB | FILE2_BB, FILE1_BB | FILE2_BB | FILE3_BB, FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB,
		~(FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB), ~(FILE9_BB | FILE8_BB | FILE7_BB), ~(FILE9_BB | FILE8_BB), ~FILE9_BB, ALL_BB }
	};
	Bitboard OrRightBB[COLOR_NB][FILE_NB] = {
		{ FILE1_BB, FILE1_BB | FILE2_BB, FILE1_BB | FILE2_BB | FILE3_BB, FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB,
		~(FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB), ~(FILE9_BB | FILE8_BB | FILE7_BB), ~(FILE9_BB | FILE8_BB), ~FILE9_BB, ALL_BB },
		{ ALL_BB, ~FILE1_BB, ~(FILE1_BB | FILE2_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB), ~(FILE1_BB | FILE2_BB | FILE3_BB | FILE4_BB),
		FILE9_BB | FILE8_BB | FILE7_BB | FILE6_BB, FILE9_BB | FILE8_BB | FILE7_BB, FILE9_BB | FILE8_BB, FILE9_BB }
	};

	// --------------------------------
	//   char32_t -> utf-8 string 変換
	// --------------------------------

	namespace UniConv {

		// std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> だとLNK2001をVS2015,VS2017が吐く不具合の回避。
		// http://qiita.com/benikabocha/items/1fc76b8cea404e9591cf
		// https://social.msdn.microsoft.com/Forums/en-US/8f40dcd8-c67f-4eba-9134-a19b9178e481/vs-2015-rc-linker-stdcodecvt-error

#ifdef _MSC_VER // MSVCの場合
		std::wstring_convert<std::codecvt_utf8<uint32_t>, uint32_t> char32_utf8_converter;
#else // MSVC以外の場合
		std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> char32_utf8_converter;
#endif

		std::string char32_to_utf8string(const char32_t * r)
		{
#ifdef _MSC_VER // MSVCの場合
			return char32_utf8_converter.to_bytes((const uint32_t *)r);
#else // MSVC以外の場合
			return char32_utf8_converter.to_bytes(r);
#endif
		}

	}


	// --------------------------------
	//   char32_t -> utf-8 string 変換
	// --------------------------------

	namespace UniConv
	{

		// std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> だとLNK2001をVS2015,VS2017が吐く不具合の回避。
		// http://qiita.com/benikabocha/items/1fc76b8cea404e9591cf
		// https://social.msdn.microsoft.com/Forums/en-US/8f40dcd8-c67f-4eba-9134-a19b9178e481/vs-2015-rc-linker-stdcodecvt-error

		std::string char32_to_utf8string(const char32_t * r);

	}


	// 後側を表現するBitboard。
	extern Bitboard InBackBB[COLOR_NB][RANK_NB];
	// 左側を表現するBitboard。
	extern Bitboard InLeftBB[COLOR_NB][FILE_NB];
	// 右側を表現するBitboard。
	extern Bitboard InRightBB[COLOR_NB][FILE_NB];
	// 以前を表現するBitboard。
	extern Bitboard OrFrontBB[COLOR_NB][RANK_NB];
	// 以後を表現するBitboard。
	extern Bitboard OrBackBB[COLOR_NB][RANK_NB];
	// 以左を表現するBitboard。
	extern Bitboard OrLeftBB[COLOR_NB][FILE_NB];
	// 以右を表現するBitboard。
	extern Bitboard OrRightBB[COLOR_NB][FILE_NB];

	// --------------------
	//       指し手出力
	// --------------------

	// KIF形式の文字列にする。
	std::string to_kif1_string(Move m, Position& pos, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);
	std::u32string to_kif1_u32string(Move m, Position& pos, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);
	// 伝統形式の文字列にする。
	std::string to_kif2_string(Move m, Position& pos, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);
	std::u32string to_kif2_u32string(Move m, Position& pos, Move prev_m = MOVE_NULL, SquareFormat fmt = SqFmt_ASCII);
	// 手番無しのCSA形式の文字列にする。
	std::string to_csa1_string(Move m, Position& pos);
	// 手番有りのCSA形式の文字列にする。
	std::string to_csa_string(Move m, Position& pos);


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


	// Rankを棋譜形式で出力する
	char32_t kif_char32(Rank r, SquareFormat fmt = SqFmt_ASCII);


	// Fileを棋譜形式で出力する
	char32_t kif_char32(File f, SquareFormat fmt = SqFmt_ASCII);

	// Square書式設定の読み込み
	inline std::istream& operator>>(std::istream& is, SquareFormat& sqfmt) { int i; is >> i; sqfmt = (SquareFormat)i; return is; }


#endif


}
#endif // defined (USE_KIF_CONVERT_TOOLS)


#endif // defined (_KIF_CONVERT_TOOLS_H_)
