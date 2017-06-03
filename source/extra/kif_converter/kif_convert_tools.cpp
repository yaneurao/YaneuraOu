#include "kif_convert_tools.h"

#if defined(USE_KIF_CONVERT_TOOLS)

#include <string>
#include <sstream>

#include "../../position.h"

namespace KifConvertTools
{
	// -----------------
	//  for CSA format
	// -----------------

	// CSA文字列の構築器
	struct CsaStringBuilder
	{
		// 暗黙の型変換子
		operator std::string() { return ss.str(); }

		// 升目をCSA形式の文字列に変換して、ssに追加する。
		void append(Square sq)
		{
			ss << str_1to9[file_of(sq)];
			ss << str_1to9[rank_of(sq)];
		}

		// 手番をCSA形式の文字列に変換して、ssに追加する。
		void append(Color c)
		{
			ss << ((c == BLACK) ? "+" : "-");
		}

		// 駒名をCSA形式の文字列に変換して、ssに追加する。
		void append(Piece pt)
		{
			ASSERT_LV3(type_of(pt) == pt);
			ss << piece_strings[pt];
		}

		// csa形式の指し手文字列に変換して返す。
		static std::string to_csa_string(Move m, Piece movedPieceAfterType, Color c)
		{
			CsaStringBuilder builder;

			switch (m)
			{
			case MOVE_NONE:
			case MOVE_NULL:
				builder.ss << "%%ERROR";
				break;
			case MOVE_RESIGN:
				builder.ss << "%%TORYO";
				break;
			case MOVE_WIN:
				builder.ss << "%%WIN";
				break;
			default:
				// --- 普通の指し手

				// 手番
				builder.append(c);

				// 打つ指し手のときは移動元の升は"00"と表現する。
				// さもなくば、"77"のように移動元の升目をアラビア数字で表現。
				if (is_drop(m))
					builder.ss << "00";
				else
					builder.append(move_from(m));

				builder.append(move_to(m));
				builder.append(type_of(movedPieceAfterType));
			}
			return (std::string)builder;
		}

	private:
		// CSAの指し手文字列などで用いる1から9までの数。
		const char* str_1to9 = "123456789";

		// 駒名。独自拡張として、金の成りのところはQUEENの意味で"QU"にしてある。
		const std::string piece_strings[16] = {
			"**", "FU", "KY" , "KE" , "GI" , "KA" , "HI" , "KI" ,
			"OU", "TO", "NY" , "NK" , "NG" , "UM" , "RY" , "QU" ,
		};

		// C#のStringBuilderのように用いる
		std::stringstream ss;
	};


	std::string to_csa_string(Move m, Piece movedPieceAfterType, Color c)
	{
		return CsaStringBuilder::to_csa_string(m, movedPieceAfterType, c);
	}

	std::string to_csa_string(Position& pos, Move m)
	{
		return  to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move());
	}

	// -----------------
	//  for KIF format
	// -----------------

	// KIF文字列の構築器
	// (CsaStringBuilderと同じ作り)
	struct KifStringBuilder
	{
		// 暗黙の型変換子
		operator std::string() { return ss.str(); }

		void append(Rank r, SquareFormat fmt)
		{
			switch (fmt)
			{
			case SqFmt_FullWidthArabic:
				// 2バイト文字をstd::stringで扱っているので2文字切り出す。
				ss << str_1to9_full_width_arabic.substr(2 * r, 2);
				break;
			case SqFmt_FullWidthMix:
				ss << str_1to9_kanji.substr(2 * r, 2);
				break;
			case SqFmt_ASCII:
				ss << str_1to9[r];
				break;
			default:
				UNREACHABLE;
			}
		}

		void append(File f , SquareFormat fmt)
		{
			switch (fmt)
			{
			case SqFmt_FullWidthArabic:
			case SqFmt_FullWidthMix:
				ss << str_1to9_full_width_arabic.substr(f*2,2);
				break;
			case SqFmt_ASCII:
				ss << str_1to9[f];
				break;
			default:
				UNREACHABLE;
			}
		}

		// 升目をKIF形式の文字列に変換して、ssに追加する。
		void append(Square sq , SquareFormat fmt)
		{
			append(file_of(sq), fmt);
			append(rank_of(sq), fmt);
		}

		// 手番をKIF形式の文字列に変換して、ssに追加する。
		void append(Color c)
		{
			ss << ((c == BLACK) ? "▲" : "△");
		}

		// 駒名をKIF形式の文字列に変換して、ssに追加する。
		void append(Piece pt)
		{
			ASSERT_LV3(type_of(pt) == pt);
			ss << piece_strings[pt];
		}

		// KIF形式の指し手文字列に変換して返す。
		// m			  : 今回の指し手
		// prev_m		  : 直前の指し手
		// movedPieceType : 移動させる駒(今回の指し手で成る場合は、成る前の駒)
		static std::string to_kif_string(Move m, Piece movedPieceType, Move prev_m , Color c, SquareFormat fmt)
		{
			KifStringBuilder builder;

			// 手番
			builder.append(c);

			switch (m)
			{
			case MOVE_NONE:
				builder.ss << "エラー";
			case MOVE_NULL:
				builder.ss << "パス";
				break;
			case MOVE_RESIGN:
				builder.ss << "投了";
				break;
			case MOVE_WIN:
				builder.ss << "勝ち宣言";
				break;
			default:
				// --- 普通の指し手

				// 一つ前の指し手の移動先と、今回の移動先が同じ場合、"同"金のように表示する。
				if (is_ok(prev_m) && move_to(prev_m) == move_to(m))
				{
					builder.ss << "同";
					builder.append(movedPieceType);
				}
				else
				{
					builder.append(move_to(m), fmt);
					builder.append(movedPieceType);
				}
				if (is_drop(m))
					builder.ss << "打";
				else if (is_promote(m))
					builder.ss << "成";
				else
				{
					// 参考用に駒の移動元を括弧表記で出力することになっている。
					builder.ss << '(';
					builder.append(move_from(m), SqFmt_ASCII);
					builder.ss << ')';
				}
			}
			return (std::string)builder;
		}

	private:
		// KIFの指し手文字列などで用いる1から9までの数。
		const char* str_1to9 = "123456789";
		const std::string str_1to9_kanji = "一二三四五六七八九";
		const std::string str_1to9_full_width_arabic = "１２３４５６７８９";

		// 駒名。独自拡張として、金の成りのところは、QUEEN(女王)の意味で"女"としてある。
		const std::string piece_strings[16] = {
			"空", "歩", "香"  , "桂"   , "銀"   , "角" , "飛" , "金" ,
			"玉", "と", "成香", "成桂" , "成銀" , "馬" , "龍" , "女" ,
		};

		// C#のStringBuilderのように用いる
		std::stringstream ss;
	};

	std::string to_kif_string(Move m, Piece movedPieceBeforeType, Move prev_m , Color c , SquareFormat fmt)
	{
		return KifStringBuilder::to_kif_string(m, movedPieceBeforeType, prev_m, c , fmt);
	}

	// KIF形式の指し手表現文字列を取得する。
	std::string to_kif_string(Position& pos, Move m , SquareFormat fmt)
	{
		return  to_kif_string(m, pos.moved_piece_before(m), pos.state()->lastMove , pos.side_to_move() , fmt);
	}

	// -----------------
	//	   UnitTest
	// -----------------

	void UnitTest()
	{
		// is_ready()は事前に呼び出されているものとする。

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをCSA文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa_string(pos, m.move) << " ";
#endif

#if 1
		// 初期局面ですべての合法な指し手を生成し、それをKIF文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_kif_string(pos, m.move , SqFmt_FullWidthMix) << " ";

#endif
	}

#if 0

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
						else if ((fromB & InLeftBB[c][toSqF]) == fromB) {
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

#endif

}

#endif
