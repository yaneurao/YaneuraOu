#if defined(USE_KIF_CONVERT_TOOLS)

#include <sstream>
#include <tuple>

// kif_convert_tools.cppで用いる文字列定数。
// SJIS,UTF-8/16/32それぞれの文字列を用意してある。
// GCCの場合、SJISを出力するにはコンパイル時に -fexec-charset=cp932 と指定する。

namespace KifConvertTools
{

	// コンソール上での文字幅計算。厳密な判定ではないが、ここで定義する文字リテラルの範囲ではこれでだいたい間に合う。
	constexpr size_t eawidth(const char32_t * s)
	{
		return *s == U'\0' ? 0 : (eawidth(s + 1) + ((uint32_t)(*s) < 128 ? 1 : 2));
	}

	// 定数文字列と文字幅のセット
	template <typename charT> struct KifConstEntry
	{
		KifConstEntry(const charT * _str, size_t _ealen) : str(_str), ealen(_ealen) {}
		const charT * str;
		const size_t ealen;
	};

	// 文字幅カウント付きの stringstream
	template <typename charT> struct EalenSstream : public std::basic_stringstream<charT>
	{
		size_t ealen() { return _ealen; }
		size_t _ealen;
	};

	// 文字幅カウント付きの文字列処理
	template <typename charT> EalenSstream<charT>& operator<< (EalenSstream<charT>& ess, KifConstEntry<charT> s)
	{
		ess._ealen += s.ealen;
		ess << s.str;
		return ess;
	}

#define SC(I, T, S) (KifConstEntry<T>(std::get<I>(std::make_tuple(S, u8"" S, u"" S, U"" S, L"" S)), eawidth(U"" S)))
#if defined(__clang__) || defined(_LINUX)
#define SCU(I, T, S, R) (KifConstEntry<T>(std::get<I>(std::make_tuple(S, u8"" S, u"" S, U"" S, L"" S)), eawidth(U"" S)))
#else
#define SCU(I, T, S, R) (KifConstEntry<T>(std::get<I>(std::make_tuple(R, u8"" S, u"" S, U"" S, L"" S)), std::get<I>(std::make_tuple(eawidth(U"" R), eawidth(U"" S), eawidth(U"" S), eawidth(U"" S), eawidth(U"" S)))))
#endif

	// 文字定数群
	template <typename T> struct KifCharBase {};
	template <size_t I, typename T> struct KifConst : KifCharBase<T>
	{
		typedef T char_type;
		typedef std::basic_string<T> string_type;
		typedef KifConstEntry<T> value_type;
		const value_type csa_color_black = SC(I,T,"+");
		const value_type csa_color_white = SC(I,T,"-");
		const value_type csa_move_none = SC(I,T,"%""ERROR");
		const value_type csa_move_null = SC(I,T,"%""PASS");
		const value_type csa_move_resign = SC(I,T,"%""TORYO");
		const value_type csa_move_win = SC(I,T,"%""WIN");
		const value_type csa_cap_sq = SC(I,T,"00");
		const value_type csa_piece[16] = {
			SC(I,T,"**"), SC(I,T,"FU"), SC(I,T,"KY"), SC(I,T,"KE"), SC(I,T,"GI"), SC(I,T,"KA"), SC(I,T,"HI"), SC(I,T,"KI"),
			SC(I,T,"OU"), SC(I,T,"TO"), SC(I,T,"NY"), SC(I,T,"NK"), SC(I,T,"NG"), SC(I,T,"UM"), SC(I,T,"RY"), SC(I,T,"QU"),
		};
		const value_type csa_ver = SC(I,T,"V2.2");
		const value_type csa_comment = SC(I,T,"'");
		const value_type csa_pos_hirate = SC(I,T,"PI");
		const value_type csa_pos_rank[9] = {
			SC(I,T,"P1"), SC(I,T,"P2"), SC(I,T,"P3"), SC(I,T,"P4"), SC(I,T,"P5"), SC(I,T,"P6"), SC(I,T,"P7"), SC(I,T,"P8"), SC(I,T,"P9"),
		};
		const value_type csa_pos_piece[32] = {
			SC(I,T," * "), SC(I,T,"+FU"), SC(I,T,"+KY"), SC(I,T,"+KE"), SC(I,T,"+GI"), SC(I,T,"+KA"), SC(I,T,"+HI"), SC(I,T,"+KI"),
			SC(I,T,"+OU"), SC(I,T,"+TO"), SC(I,T,"+NY"), SC(I,T,"+NK"), SC(I,T,"+NG"), SC(I,T,"+UM"), SC(I,T,"+RY"), SC(I,T,"+QU"),
			SC(I,T," * "), SC(I,T,"-FU"), SC(I,T,"-KY"), SC(I,T,"-KE"), SC(I,T,"-GI"), SC(I,T,"-KA"), SC(I,T,"-HI"), SC(I,T,"-KI"),
			SC(I,T,"-OU"), SC(I,T,"-TO"), SC(I,T,"-NY"), SC(I,T,"-NK"), SC(I,T,"-NG"), SC(I,T,"-UM"), SC(I,T,"-RY"), SC(I,T,"-QU"),
		};
		const value_type csa_hand_black = SC(I,T,"P+");
		const value_type csa_hand_white = SC(I,T,"P-");
		const value_type kif_color_black = SC(I,T,"▲");
		const value_type kif_color_white = SC(I,T,"△");
		const value_type kif_color_blackinv = SC(I,T,"▼");
		const value_type kif_color_whiteinv = SC(I,T,"▽");
		const value_type piece_color_black = SCU(I,T,"☗","▲");
		const value_type piece_color_white = SCU(I,T,"☖","△");
		const value_type piece_color_blackinv = SCU(I,T,"⛊","▼");
		const value_type piece_color_whiteinv = SCU(I,T,"⛉","▽");
		const value_type kif_move_none = SC(I,T,"エラー");
		const value_type kif_move_null = SC(I,T,"パス");
		const value_type kif_move_resign = SC(I,T,"投了");
		const value_type kif_move_win = SC(I,T,"勝ち宣言");
		const value_type kif_move_samepos = SC(I,T,"同");
		const value_type kif_fwsp = SC(I,T,"　");
		const value_type kif_move_drop = SC(I,T,"打");
		const value_type kif_move_not = SC(I,T,"不");
		const value_type kif_move_promote = SC(I,T,"成");
		const value_type kif_move_straight = SC(I,T,"直");
		const value_type kif_move_upper = SC(I,T,"上");
		const value_type kif_move_lower = SC(I,T,"引");
		const value_type kif_move_slide = SC(I,T,"寄");
		const value_type kif_move_left = SC(I,T,"左");
		const value_type kif_move_right = SC(I,T,"右");
		const value_type kif_lbrack = SC(I,T,"(");
		const value_type kif_rbrack = SC(I,T,")");
		const value_type char1to9_ascii[9] = {
			SC(I,T,"1"), SC(I,T,"2"), SC(I,T,"3"), SC(I,T,"4"), SC(I,T,"5"), SC(I,T,"6"), SC(I,T,"7"), SC(I,T,"8"), SC(I,T,"9"),
		};
		const value_type char1to9_kanji[9] = {
			SC(I,T,"一"), SC(I,T,"二"), SC(I,T,"三"), SC(I,T,"四"), SC(I,T,"五"), SC(I,T,"六"), SC(I,T,"七"), SC(I,T,"八"), SC(I,T,"九"),
		};
		const value_type char1to9_full_width_arabic[9] = {
			SC(I,T,"１"), SC(I,T,"２"), SC(I,T,"３"), SC(I,T,"４"), SC(I,T,"５"), SC(I,T,"６"), SC(I,T,"７"), SC(I,T,"８"), SC(I,T,"９"),
		};
		const value_type kif_piece[16] = {
			SC(I,T,"・"), SC(I,T,"歩"), SC(I,T,"香"), SC(I,T,"桂"), SC(I,T,"銀"), SC(I,T,"角"), SC(I,T,"飛"), SC(I,T,"金"),
			SC(I,T,"玉"), SC(I,T,"と"), SC(I,T,"成香"), SC(I,T,"成桂"), SC(I,T,"成銀"), SC(I,T,"馬"), SC(I,T,"龍"), SC(I,T,"女"),
		};
		const value_type bod_fline = SC(I,T,"  ９ ８ ７ ６ ５ ４ ３ ２ １");
		const value_type bod_hline = SC(I,T,"+---------------------------+");
		const value_type bod_vline = SC(I,T,"|");
		const value_type bod_rank[9] = {
			SC(I,T,"一"), SC(I,T,"二"), SC(I,T,"三"), SC(I,T,"四"), SC(I,T,"五"), SC(I,T,"六"), SC(I,T,"七"), SC(I,T,"八"), SC(I,T,"九"),
		};
		const value_type bod_piece[32] = {
			SC(I,T," ・"), SC(I,T," 歩"), SC(I,T," 香"), SC(I,T," 桂"), SC(I,T," 銀"), SC(I,T," 角"), SC(I,T," 飛"), SC(I,T," 金"),
			SC(I,T," 玉"), SC(I,T," と"), SC(I,T," 杏"), SC(I,T," 圭"), SC(I,T," 全"), SC(I,T," 馬"), SC(I,T," 龍"), SC(I,T," 女"),
			SC(I,T," ・"), SC(I,T,"v歩"), SC(I,T,"v香"), SC(I,T,"v桂"), SC(I,T,"v銀"), SC(I,T,"v角"), SC(I,T,"v飛"), SC(I,T,"v金"),
			SC(I,T,"v玉"), SC(I,T,"vと"), SC(I,T,"v杏"), SC(I,T,"v圭"), SC(I,T,"v全"), SC(I,T,"v馬"), SC(I,T,"v龍"), SC(I,T,"v女"),
		};
		const value_type bod_hand_color_black = SC(I,T,"先手の持駒：");
		const value_type bod_hand_color_white = SC(I,T,"後手の持駒：");
		const value_type bod_hand_piece[16] = {
			SC(I,T,"・"), SC(I,T,"歩"), SC(I,T,"香"), SC(I,T,"桂"), SC(I,T,"銀"), SC(I,T,"角"), SC(I,T,"飛"), SC(I,T,"金"),
			SC(I,T,"玉"), SC(I,T,"と"), SC(I,T,"杏"), SC(I,T,"圭"), SC(I,T,"全"), SC(I,T,"馬"), SC(I,T,"龍"), SC(I,T,"女"),
		};
		const value_type bod_hand_pad = SC(I,T," ");
		const value_type bod_hand_none = SC(I,T,"なし");
		const value_type bod_hand_num[19] = {
			SC(I,T,""), SC(I,T,""), SC(I,T,"二"), SC(I,T,"三"), SC(I,T,"四"), SC(I,T,"五"), SC(I,T,"六"), SC(I,T,"七"), SC(I,T,"八"), SC(I,T,"九"),
			SC(I,T,"十"), SC(I,T,"十一"), SC(I,T,"十二"), SC(I,T,"十三"), SC(I,T,"十四"), SC(I,T,"十五"), SC(I,T,"十六"), SC(I,T,"十七"), SC(I,T,"十八"),
		};
		const value_type bod_turn_black = SC(I,T,"先手番");
		const value_type bod_turn_white = SC(I,T,"後手番");
		const value_type kiflist_hirate = SC(I,T, "手合割：平手");
		const value_type kiflist_head = SC(I,T,"手数----指手---------消費時間--");
		const value_type kiflist_pad = SC(I,T," ");
		const value_type kiflist_spendnotime = SC(I,T,"( 0:00/00:00:00)");
		const value_type char0to9_ascii[10] = {
			SC(I,T,"0"),SC(I,T,"1"), SC(I,T,"2"), SC(I,T,"3"), SC(I,T,"4"), SC(I,T,"5"), SC(I,T,"6"), SC(I,T,"7"), SC(I,T,"8"), SC(I,T,"9"),
		};
	};

	struct KifConstLocale : KifConst<0, char>     {};
	struct KifConstUtf8   : KifConst<1, char>     {};
	struct KifConstUtf16  : KifConst<2, char16_t> {};
	struct KifConstUtf32  : KifConst<3, char32_t> {};
	struct KifConstWchar  : KifConst<4, wchar_t>  {};

#undef SC
#undef SCU

}

#endif // ifdef USE_KIF_CONVERT_TOOLS
