#ifndef _KIF_CONVERT_TOOLS_H_
#define _KIF_CONVERT_TOOLS_H_

// 棋譜の変換などを行なうツールセット
// CSA,KIF,KIF2(KI2)形式などの入出力を担う。

#if defined(USE_KIF_CONVERT_TOOLS)

struct Position;
namespace KifConvertTools
{

	// 手番出力の書式
	enum ColorFormat : int {
		ColorFmt_None , // black: "", white: ""
		ColorFmt_CSA  , // black: "+", white: "-"
		ColorFmt_KIF  , // black: "▲", white: "△"
		ColorFmt_Piece, // black: "☗", white: "☖"
		ColorFmt_NB // 番兵
	};

	// KIF形式のマス目の棋譜表記時の書式
	enum SquareFormat : int {
		SqFmt_ASCII           , // 普通のASCII形式
		SqFmt_FullWidthArabic , // 全角アラビア数字
		SqFmt_FullWidthMix    , // 全角漢数字・アラビア数字混在
		SqFmt_NB // 番兵
	};

	// 同～と出力する際の書式
	enum SamePosFormat : int {
		SamePosFmt_Short  , // 例：「同金」「同成銀」「同桂成」
		SamePosFmt_KIFsp  , // 例：「同　金」「同　成銀」「同　桂成」
		SamePosFmt_KI2sp  , // 例：「同　金」「同成銀」「同桂成」
		SamePosFmt_Verbose, // 例：「３二同金」「３二同成銀」「３二同桂成」
		SamePosFmt_NB // 番兵
	};

	struct KifFormat
	{
		ColorFormat color_type;
		SquareFormat square_type;
		SamePosFormat samepos_type;
		KifFormat(ColorFormat cfmt = ColorFmt_KIF, SquareFormat sqfmt = SqFmt_ASCII, SamePosFormat spfmt = SamePosFmt_Short) : color_type(cfmt), square_type(sqfmt), samepos_type(spfmt) {}
	};

	static const struct KifFormat CsaFmt(ColorFmt_CSA);
	static const struct KifFormat Csa1Fmt(ColorFmt_None);
	static const struct KifFormat KifFmt(ColorFmt_None, SqFmt_FullWidthMix, SamePosFmt_KIFsp);
	static const struct KifFormat Kif2Fmt(ColorFmt_KIF, SqFmt_FullWidthMix, SamePosFmt_KI2sp);
	static const struct KifFormat KifNml(ColorFmt_KIF, SqFmt_FullWidthArabic);
	static const struct KifFormat KifNmln(ColorFmt_None, SqFmt_FullWidthArabic);
	static const struct KifFormat KifNmlp(ColorFmt_Piece, SqFmt_FullWidthArabic);

	static const struct KifFormat KifFmtA(ColorFmt_KIF, SqFmt_ASCII);
	static const struct KifFormat KifFmtK(ColorFmt_KIF, SqFmt_FullWidthMix);
	static const struct KifFormat KifFmtF(ColorFmt_KIF, SqFmt_FullWidthArabic);
	static const struct KifFormat KifFmtAn(ColorFmt_None, SqFmt_ASCII);
	static const struct KifFormat KifFmtKn(ColorFmt_None, SqFmt_FullWidthMix);
	static const struct KifFormat KifFmtFn(ColorFmt_None, SqFmt_FullWidthArabic);
	static const struct KifFormat KifFmtAp(ColorFmt_Piece, SqFmt_ASCII);
	static const struct KifFormat KifFmtKp(ColorFmt_Piece, SqFmt_FullWidthMix);
	static const struct KifFormat KifFmtFp(ColorFmt_Piece, SqFmt_FullWidthArabic);

	static const struct KifFormat KifFmtA1(ColorFmt_KIF, SqFmt_ASCII, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtK1(ColorFmt_KIF, SqFmt_FullWidthMix, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtF1(ColorFmt_KIF, SqFmt_FullWidthArabic, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtAn1(ColorFmt_None, SqFmt_ASCII, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtKn1(ColorFmt_None, SqFmt_FullWidthMix, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtFn1(ColorFmt_None, SqFmt_FullWidthArabic, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtAp1(ColorFmt_Piece, SqFmt_ASCII, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtKp1(ColorFmt_Piece, SqFmt_FullWidthMix, SamePosFmt_KIFsp);
	static const struct KifFormat KifFmtFp1(ColorFmt_Piece, SqFmt_FullWidthArabic, SamePosFmt_KIFsp);

	static const struct KifFormat KifFmtA2(ColorFmt_KIF, SqFmt_ASCII, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtK2(ColorFmt_KIF, SqFmt_FullWidthMix, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtF2(ColorFmt_KIF, SqFmt_FullWidthArabic, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtAn2(ColorFmt_None, SqFmt_ASCII, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtKn2(ColorFmt_None, SqFmt_FullWidthMix, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtFn2(ColorFmt_None, SqFmt_FullWidthArabic, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtAp2(ColorFmt_Piece, SqFmt_ASCII, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtKp2(ColorFmt_Piece, SqFmt_FullWidthMix, SamePosFmt_KI2sp);
	static const struct KifFormat KifFmtFp2(ColorFmt_Piece, SqFmt_FullWidthArabic, SamePosFmt_KI2sp);

	static const struct KifFormat KifFmtAv(ColorFmt_KIF, SqFmt_ASCII, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtKv(ColorFmt_KIF, SqFmt_FullWidthMix, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtFv(ColorFmt_KIF, SqFmt_FullWidthArabic, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtAnv(ColorFmt_None, SqFmt_ASCII, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtKnv(ColorFmt_None, SqFmt_FullWidthMix, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtFnv(ColorFmt_None, SqFmt_FullWidthArabic, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtApv(ColorFmt_Piece, SqFmt_ASCII, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtKpv(ColorFmt_Piece, SqFmt_FullWidthMix, SamePosFmt_Verbose);
	static const struct KifFormat KifFmtFpv(ColorFmt_Piece, SqFmt_FullWidthArabic, SamePosFmt_Verbose);

	// --- SFEN形式

	// SFEN形式の棋譜文字列を取得する。
	extern std::string to_sfen_string(Position& pos);

	// --- CSA形式

	// CSA形式の指し手表現文字列を取得する。
	extern std::string to_csa_string(Position& pos, Move m, const KifFormat& fmt = CsaFmt);
	extern std::string to_csa_u8string(Position& pos, Move m, const KifFormat& fmt = CsaFmt);
	extern std::u16string to_csa_u16string(Position& pos, Move m, const KifFormat& fmt = CsaFmt);
	extern std::u32string to_csa_u32string(Position& pos, Move m, const KifFormat& fmt = CsaFmt);
	extern std::wstring to_csa_wstring(Position& pos, Move m, const KifFormat& fmt = CsaFmt);

	// CSA形式の棋譜文字列を取得する。
	extern std::string to_csa_string(Position& pos, const KifFormat& fmt = CsaFmt);
	extern std::string to_csa_u8string(Position& pos, const KifFormat& fmt = CsaFmt);
	extern std::u16string to_csa_u16string(Position& pos, const KifFormat& fmt = CsaFmt);
	extern std::u32string to_csa_u32string(Position& pos, const KifFormat& fmt = CsaFmt);
	extern std::wstring to_csa_wstring(Position& pos, const KifFormat& fmt = CsaFmt);

	// --- KIF/KIF2(KI2)形式

	// KIF形式の指し手表現文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif_string(Position& pos, Move m, const KifFormat& fmt = KifFmtA);
	extern std::string to_kif_u8string(Position& pos, Move m, const KifFormat& fmt = KifFmtA);
	extern std::u16string to_kif_u16string(Position& pos, Move m, const KifFormat& fmt = KifFmtA);
	extern std::u32string to_kif_u32string(Position& pos, Move m, const KifFormat& fmt = KifFmtA);
	extern std::wstring to_kif_wstring(Position& pos, Move m, const KifFormat& fmt = KifFmtA);

	// KIF形式の棋譜文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif_string(Position& pos, const KifFormat& fmt = KifFmtA);
	extern std::string to_kif_u8string(Position& pos, const KifFormat& fmt = KifFmtA);
	extern std::u16string to_kif_u16string(Position& pos, const KifFormat& fmt = KifFmtA);
	extern std::u32string to_kif_u32string(Position& pos, const KifFormat& fmt = KifFmtA);
	extern std::wstring to_kif_wstring(Position& pos, const KifFormat& fmt = KifFmtA);

	// KIF2形式の指し手表現文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif2_string(Position& pos, Move m, const KifFormat& fmt = KifFmtA2);
	extern std::string to_kif2_u8string(Position& pos, Move m, const KifFormat& fmt = KifFmtA2);
	extern std::u16string to_kif2_u16string(Position& pos, Move m, const KifFormat& fmt = KifFmtA2);
	extern std::u32string to_kif2_u32string(Position& pos, Move m, const KifFormat& fmt = KifFmtA2);
	extern std::wstring to_kif2_wstring(Position& pos, Move m, const KifFormat& fmt = KifFmtA2);

	// KIF2形式の棋譜文字列を取得する。
	// 出力文字列のエンコードは、関数名にu8とついているのはutf-8。u16はutf-16、u32はutf-32。
	// 何もついていないものはSJIS。

	extern std::string to_kif2_string(Position& pos, const KifFormat& fmt = KifFmtA2);
	extern std::string to_kif2_u8string(Position& pos, const KifFormat& fmt = KifFmtA2);
	extern std::u16string to_kif2_u16string(Position& pos, const KifFormat& fmt = KifFmtA2);
	extern std::u32string to_kif2_u32string(Position& pos, const KifFormat& fmt = KifFmtA2);
	extern std::wstring to_kif2_wstring(Position& pos, const KifFormat& fmt = KifFmtA2);

	// --- UnitTest

	// 現状、テスト用のコードが書き散らしてあるだけ。
	extern void UnitTest();

}
#endif // defined (USE_KIF_CONVERT_TOOLS)


#endif // defined (_KIF_CONVERT_TOOLS_H_)
