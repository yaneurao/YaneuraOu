#include "kif_convert_tools.h"

#if defined(USE_KIF_CONVERT_TOOLS)

#include <string>
#include <sstream>

#include "../../position.h"

namespace KifConvertTools
{
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

				return (std::string)builder;
			}
		}

	private:
		// CSAの指し手文字列などで用いる1から9までの数。
		const char* str_1to9 = "123456789";

		const std::string piece_strings[18] = {
			"**", "FU", "KY" , "KE" , "GI" , "KA" , "HI" , "KI" ,
			"OU", "TO", "NY", "NK" , "NG" ,  "UM" , "RY" , "KI" ,
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


	// --- UnitTest

	void UnitTest()
	{
		// is_ready()は事前に呼び出されているものとする。

		// 初期局面ですべての合法な指し手を生成し、それをCSA文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa_string(pos, m.move) << " ";

		//std::cout << "TEST";
	}

#if 0

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

#endif

}

#endif
