#if defined(USE_KIF_CONVERT_TOOLS)

// kif_convert_tools.cppで用いる文字列定数。
// SJIS、utf-8/16/32それぞれの文字列を用意してある。
// ここの文字列を修正するときは、それぞれの文字列を修正すること。

namespace KifConvertTools
{

	template <typename T> struct KifConstBase {};

	struct KifConstLocale : KifConstBase<char>
	{
		const char * const color_black = "▲";
		const char * const color_white = "△";
		const char * const move_none = "エラー";
		const char * const move_null = "パス";
		const char * const move_resign = "投了";
		const char * const move_win = "勝ち宣言";
		const char * const move_samepos = "同";
		const char * const move_drop = "打";
		const char * const move_not = "不";
		const char * const move_promote = "成";
		const char * const move_straight = "直";
		const char * const move_upper = "上";
		const char * const move_lower = "引";
		const char * const move_slide = "寄";
		const char * const move_left = "左";
		const char * const move_right = "右";
		const char * const lbrack = "(";
		const char * const rbrack = ")";
		const char * const char1to9_ascii[9] = { "1", "2", "3", "4", "5", "6", "7", "8", "9" };
		const char * const char1to9_kanji[9] = { "一", "二", "三", "四", "五", "六", "七", "八", "九" };
		const char * const char1to9_full_width_arabic[9] = { "１", "２", "３", "４", "５", "６", "７", "８", "９" };
		const char * const piece_strings[16] = {
			"空", "歩", "香"  , "桂"   , "銀"   , "角" , "飛" , "金" ,
			"玉", "と", "成香", "成桂" , "成銀" , "馬" , "龍" , "女" ,
		};
	};

	struct KifConstUtf8 : KifConstBase<char>
	{
		const char * const color_black = u8"▲";
		const char * const color_white = u8"△";
		const char * const move_none = u8"エラー";
		const char * const move_null = u8"パス";
		const char * const move_resign = u8"投了";
		const char * const move_win = u8"勝ち宣言";
		const char * const move_samepos = u8"同";
		const char * const move_drop = u8"打";
		const char * const move_not = u8"不";
		const char * const move_promote = u8"成";
		const char * const move_straight = u8"直";
		const char * const move_upper = u8"上";
		const char * const move_lower = u8"引";
		const char * const move_slide = u8"寄";
		const char * const move_left = u8"左";
		const char * const move_right = u8"右";
		const char * const lbrack = u8"(";
		const char * const rbrack = u8")";
		const char * const char1to9_ascii[9] = { u8"1", u8"2", u8"3", u8"4", u8"5", u8"6", u8"7", u8"8", u8"9" };
		const char * const char1to9_kanji[9] = { u8"一", u8"二", u8"三", u8"四", u8"五", u8"六", u8"七", u8"八", u8"九" };
		const char * const char1to9_full_width_arabic[9] = { u8"１", u8"２", u8"３", u8"４", u8"５", u8"６", u8"７", u8"８", u8"９" };
		const char * const piece_strings[16] = {
			u8"空", u8"歩", u8"香"  , u8"桂"   , u8"銀"   , u8"角" , u8"飛" , u8"金" ,
			u8"玉", u8"と", u8"成香", u8"成桂" , u8"成銀" , u8"馬" , u8"龍" , u8"女" ,
		};
	};

	struct KifConstUtf16 : KifConstBase<char16_t>
	{
		const char16_t * const color_black = u"▲";
		const char16_t * const color_white = u"△";
		const char16_t * const move_none = u"エラー";
		const char16_t * const move_null = u"パス";
		const char16_t * const move_resign = u"投了";
		const char16_t * const move_win = u"勝ち宣言";
		const char16_t * const move_samepos = u"同";
		const char16_t * const move_drop = u"打";
		const char16_t * const move_not = u"不";
		const char16_t * const move_promote = u"成";
		const char16_t * const move_straight = u"直";
		const char16_t * const move_upper = u"上";
		const char16_t * const move_lower = u"引";
		const char16_t * const move_slide = u"寄";
		const char16_t * const move_left = u"左";
		const char16_t * const move_right = u"右";
		const char16_t * const lbrack = u"(";
		const char16_t * const rbrack = u")";
		const char16_t * const char1to9_ascii[9] = { u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9" };
		const char16_t * const char1to9_kanji[9] = { u"一", u"二", u"三", u"四", u"五", u"六", u"七", u"八", u"九" };
		const char16_t * const char1to9_full_width_arabic[9] = { u"１", u"２", u"３", u"４", u"５", u"６", u"７", u"８", u"９" };
		const char16_t * const piece_strings[16] = {
			u"空", u"歩", u"香"  , u"桂"   , u"銀"   , u"角" , u"飛" , u"金" ,
			u"玉", u"と", u"成香", u"成桂" , u"成銀" , u"馬" , u"龍" , u"女" ,
		};
	};

	struct KifConstUtf32 : KifConstBase<char32_t>
	{
		const char32_t * const color_black = U"▲";
		const char32_t * const color_white = U"△";
		const char32_t * const move_none = U"エラー";
		const char32_t * const move_null = U"パス";
		const char32_t * const move_resign = U"投了";
		const char32_t * const move_win = U"勝ち宣言";
		const char32_t * const move_samepos = U"同";
		const char32_t * const move_drop = U"打";
		const char32_t * const move_not = U"不";
		const char32_t * const move_promote = U"成";
		const char32_t * const move_straight = U"直";
		const char32_t * const move_upper = U"上";
		const char32_t * const move_lower = U"引";
		const char32_t * const move_slide = U"寄";
		const char32_t * const move_left = U"左";
		const char32_t * const move_right = U"右";
		const char32_t * const lbrack = U"(";
		const char32_t * const rbrack = U")";
		const char32_t * const char1to9_ascii[9] = { U"1", U"2", U"3", U"4", U"5", U"6", U"7", U"8", U"9" };
		const char32_t * const char1to9_kanji[9] = { U"一", U"二", U"三", U"四", U"五", U"六", U"七", U"八", U"九" };
		const char32_t * const char1to9_full_width_arabic[9] = { U"１", U"２", U"３", U"４", U"５", U"６", U"７", U"８", U"９" };
		const char32_t * const piece_strings[16] = {
			U"空", U"歩", U"香"  , U"桂"   , U"銀"   , U"角" , U"飛" , U"金" ,
			U"玉", U"と", U"成香", U"成桂" , U"成銀" , U"馬" , U"龍" , U"女" ,
		};
	};

}

#endif