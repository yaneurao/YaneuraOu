#include "kif_convert_tools.h"

#if defined(USE_KIF_CONVERT_TOOLS)

#include <string>
#include <sstream>

#include "../../position.h"
extern void is_ready();

#include "kif_convert_consts.h"

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

		// csa形式の指し手文字列に変換して返す。（手番あり）
		std::string to_csa_string(Move m, Piece movedPieceAfterType, Color c)
		{
			switch (m)
			{
			case MOVE_NONE:
			case MOVE_NULL:
				ss << "%%ERROR";
				break;
			case MOVE_RESIGN:
				ss << "%%TORYO";
				break;
			case MOVE_WIN:
				ss << "%%WIN";
				break;
			default:
				// --- 普通の指し手

				// 手番
				append(c);

				// 打つ指し手のときは移動元の升は"00"と表現する。
				// さもなくば、"77"のように移動元の升目をアラビア数字で表現。
				if (is_drop(m))
					ss << "00";
				else
					append(move_from(m));

				append(move_to(m));
				append(type_of(movedPieceAfterType));
			}
			return ss.str();
		}

		// csa形式の指し手文字列に変換して返す。（手番なし）
		std::string to_csa1_string(Move m, Piece movedPieceAfterType, Color c)
		{
			switch (m)
			{
			case MOVE_NONE:
			case MOVE_NULL:
				ss << "%%ERROR";
				break;
			case MOVE_RESIGN:
				ss << "%%TORYO";
				break;
			case MOVE_WIN:
				ss << "%%WIN";
				break;
			default:
				// --- 普通の指し手

				// 打つ指し手のときは移動元の升は"00"と表現する。
				// さもなくば、"77"のように移動元の升目をアラビア数字で表現。
				if (is_drop(m))
					ss << "00";
				else
					append(move_from(m));

				append(move_to(m));
				append(type_of(movedPieceAfterType));
			}
			return ss.str();
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
		CsaStringBuilder builder;
		return builder.to_csa_string(m, movedPieceAfterType, c);
	}
	std::string to_csa1_string(Move m, Piece movedPieceAfterType, Color c)
	{
		CsaStringBuilder builder;
		return builder.to_csa1_string(m, movedPieceAfterType, c);
	}

	std::string to_csa_string(Position& pos, Move m)
	{
		CsaStringBuilder builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move());
	}
	std::string to_csa1_string(Position& pos, Move m)
	{
		CsaStringBuilder builder;
		return builder.to_csa1_string(m, pos.moved_piece_after(m), pos.side_to_move());
	}

	// -----------------
	//  for KIF format
	// -----------------

	// KIF文字列の構築器
	// (CsaStringBuilderと同じ作り)
	template <class strT, class constT>
	struct KifStringBuilder
	{
		// 暗黙の型変換子
		operator strT() { return ss.str(); }

		void append(Rank r, SquareFormat fmt)
		{
			switch (fmt)
			{
			case SqFmt_FullWidthArabic:
				ss << constStr.char1to9_full_width_arabic[r];
				break;
			case SqFmt_FullWidthMix:
				ss << constStr.char1to9_kanji[r];
				break;
			case SqFmt_ASCII:
				ss << constStr.char1to9_ascii[r];
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
				ss << constStr.char1to9_full_width_arabic[f];
				break;
			case SqFmt_ASCII:
				ss << constStr.char1to9_ascii[f];
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
			ss << ((c == BLACK) ? constStr.color_black : constStr.color_white);
		}

		// 駒名をKIF形式の文字列に変換して、ssに追加する。
		void append(Piece pt)
		{
			ASSERT_LV3(type_of(pt) == pt);
			ss << constStr.piece_strings[pt];
		}

		// KIF形式の指し手文字列に変換して返す。
		// m			  : 今回の指し手
		// prev_m		  : 直前の指し手
		// movedPieceType : 移動させる駒(今回の指し手で成る場合は、成る前の駒)
		strT to_kif_string(Move m, Piece movedPieceType, Move prev_m , Color c, SquareFormat fmt)
		{
			// 手番
			append(c);

			switch (m)
			{
			case MOVE_NONE:
				ss << constStr.move_none;
			case MOVE_NULL:
				ss << constStr.move_null;
				break;
			case MOVE_RESIGN:
				ss << constStr.move_resign;
				break;
			case MOVE_WIN:
				ss << constStr.move_win;
				break;
			default:
				// --- 普通の指し手

				// 一つ前の指し手の移動先と、今回の移動先が同じ場合、"同"金のように表示する。
				if (is_ok(prev_m) && move_to(prev_m) == move_to(m))
				{
					ss << constStr.move_samepos;
					append(movedPieceType);
				}
				else
				{
					append(move_to(m), fmt);
					append(movedPieceType);
				}
				if (is_drop(m))
					ss << constStr.move_drop;
				else if (is_promote(m))
					ss << constStr.move_promote;
				else
				{
					// 参考用に駒の移動元を括弧表記で出力することになっている。
					ss << constStr.lbrack;
					append(move_from(m), SqFmt_ASCII);
					ss << constStr.rbrack;
				}
			}
			return ss.str();
		}

		// KIF2形式の指し手文字列に変換して返す。
		// m			  : 今回の指し手
		// prev_m		  : 直前の指し手
		// movedPieceType : 移動させる駒(今回の指し手で成る場合は、成る前の駒)
		strT to_kif2_string(Position& pos, Move m, SquareFormat fmt)
		{
			// 手番
			Color c = pos.side_to_move();
			append(c);

			switch (m)
			{
			case MOVE_NONE:
				ss << constStr.move_none;
			case MOVE_NULL:
				ss << constStr.move_null;
				break;
			case MOVE_RESIGN:
				ss << constStr.move_resign;
				break;
			case MOVE_WIN:
				ss << constStr.move_win;
				break;
			default:
				// --- 普通の指し手

				// KIF形式との違いは、移動元の升を明示しないことである。
				// そのため、"同金左"のような、表記を行なう必要がある。

				// 先後の属性を排除した駒種別
				Piece pt = type_of(pos.moved_piece_before(m));

				// 成りの属性も排除した駒
				Piece pr = raw_type_of(pt);

				// 金・銀・成金は直の表記が有り得る
				bool is_like_goldsilver = (pt == SILVER || pt == GOLD || pt == PRO_PAWN || pt == PRO_LANCE
					|| pt == PRO_KNIGHT || pt == PRO_SILVER);

				// 移動元・移動先の座標
				Square from = move_from(m), to = move_to(m);

				File from_file = file_of(from), to_file = file_of(to);
				Rank from_rank = rank_of(from), to_rank = rank_of(to);

				Bitboard fromBB = Bitboard(from);

				// 移動先地点への同駒種の利き（駒打ちで無ければ指し手の駒も含む）
				// Position.piecesの第2引数には先後の属性を含めてはいけない。
				Bitboard attackerBB = (pos.attackers_to(c, to) & pos.pieces(c, pt));

				// 一つ前の指し手
				Move prev_m = pos.state()->lastMove;

				// 移動先座標
				if (is_ok(prev_m) && move_to(prev_m) == to)
					ss << constStr.move_samepos;
				else
					append(to, fmt);

				// 駒種の出力
				append(pt);

				// toのある升からdirの方角が1になっているbitboardを取得する。
				// (例えば、dir == DIRECT_Rなら、sqより右にある升がすべて1)
				// また、cがWHITEならdirを180度回転させて考える。
				// dir_or_bbのほうは、toの升のある筋or段も含むbitboardが返る。
				// (例えば、dir == DIRECT_Rなら、sqのある筋も1)
				using namespace Effect8;
				auto dir_bb = [c, to](Direct dir) { return dir_bb_(c, to, dir,false); };
				auto dir_or_bb = [c, to](Direct dir) { return dir_bb_(c, to, dir,true); };

				// 打ち駒ではないかどうか
				if (!is_drop(m))
				{
					// toに動ける同種駒が他にもあるなら
					if (attackerBB != fromBB) {
						// 真横に寄るMoveの場合
						if (from_rank == to_rank)
						{
							// 真横に寄る同種駒がそれだけなら「寄」
							if ((attackerBB & RANK_BB[to_rank]) == fromBB)
								ss << constStr.move_slide;
							// 右から寄るなら「右」
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.move_right;
								// 右から動く駒がそれだけでは無いならさらに「寄」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.move_slide;
							}
							// 左から寄るなら「左」
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.move_left;
								// 左から動く駒がそれだけでは無いならさらに「寄」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.move_slide;
							}
						}
						// 前に上がるMoveの場合
						else if ((fromBB & dir_bb(DIRECT_D)) == fromBB) {
							// 前に上がる同種駒がそれだけなら「上」
							if ((attackerBB & dir_bb(DIRECT_D)) == fromBB)
								ss << constStr.move_upper;
							// 真っ直ぐ上がるMoveの場合
							else if (from_file == to_file) {
								// 金・銀・成金なら「直」
								if (is_like_goldsilver)
									ss << constStr.move_straight;
								// 同じ筋・より右の筋に他に動ける駒がないなら「右」
								else if ((attackerBB & dir_or_bb(DIRECT_R)) == fromBB)
									ss << constStr.move_right;
								// 同じ筋・より左の筋に他に動ける駒がないなら「左」
								else if ((attackerBB & dir_or_bb(DIRECT_L)) == fromBB)
									ss << constStr.move_left;
								// 「右上」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_R) & dir_bb(DIRECT_D)) == fromBB)
									ss << constStr.move_right << constStr.move_upper;
								// 「左上」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_L) & dir_bb(DIRECT_D)) == fromBB)
									ss << constStr.move_left << constStr.move_upper;
							}
							// 右から上がるMoveの場合
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.move_right;
								// 右から動ける駒が他にもあればさらに「上」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.move_upper;
							}
							// 左から上がるMoveの場合
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.move_left;
								// 左から動ける駒が他にもあればさらに「上」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.move_upper;
							}
						}
						// 後ろに引くMoveの場合
						else if ((fromBB & InFrontBB[c][to_rank]) == fromBB) {
							// 後ろに引く同種駒がそれだけなら
							if ((attackerBB & InFrontBB[c][to_rank]) == fromBB)
								ss << constStr.move_lower;
							// 真っ直ぐ引くMoveの場合
							else if (from_file == to_file) {
								// 同じ筋・より右の筋に他に動ける駒がないなら「右」
								if ((attackerBB & dir_or_bb(DIRECT_R)) == fromBB)
									ss << constStr.move_right;
								// 同じ筋・より左の筋に他に動ける駒がないなら「左」
								else if ((attackerBB & dir_or_bb(DIRECT_L)) == fromBB)
									ss << constStr.move_left;
								// 「右引」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_R) & InFrontBB[c][to_rank]) == fromBB)
									ss << constStr.move_right << constStr.move_lower;
								// 「左引」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_L) & InFrontBB[c][to_rank]) == fromBB)
									ss << constStr.move_left << constStr.move_lower;
							}
							// 右から引くMoveの場合
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.move_right;
								// 右から動ける駒が他にもあればさらに「引」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.move_lower;
							}
							// 左から引くMoveの場合
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.move_left;
								// 左から動ける駒が他にもあればさらに「引」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.move_lower;
							}
						}
					}

					// 成ったなら必ず「成」
					if (is_promote(m))
						ss << constStr.move_promote;
					// 成れるのに成らなかったなら「不成」
					else if (pt < GOLD && canPromote(c, from, to))
						ss << constStr.move_not << constStr.move_promote;

				}
				// そこへ移動できる同種駒があるなら「打」
				else if (attackerBB != ZERO_BB)
					ss << constStr.move_drop;

				break;
			}

			return ss.str();
		}

	private:
		// KIF2形式で、"同金右"のような出力をしないといけないので、ある升の右側に同種の駒があるかを
		// 判定しなければならず、そのためにある升のX側を示すBitboardが必要となる。
		// sqの升を中心として、dir側の方角がすべて1になっているBitboardが返る。
		// ただし、c == WHITEのときは盤面を180度回転させてdirの方角を考える。
		// またor_flag == trueのときは、「sqの升を含む 筋 or 段」も1であるBitboardが返る。
		static Bitboard dir_bb_(Color color , Square sq, Effect8::Direct dir , bool or_flag)
		{
			// このへんの定数の定義を変更されてしまうと以下のhackが成り立たなくなる。
			static_assert(FILE_1 == FILE_ZERO, "");
			static_assert(RANK_1 == RANK_ZERO, "");

			// color == WHITEならば逆の方角にする。
			dir = (color == BLACK) ? dir : ~dir;
			
			// スタートするときのoffset
			int offset = or_flag ? 0 : 1;

			Bitboard bb = ZERO_BB;
			switch (dir)
			{
			case Effect8::DIRECT_R: for (File f = File(file_of(sq) - offset); f >= FILE_1; f--) bb |= FILE_BB[f]; break;
			case Effect8::DIRECT_L: for (File f = File(file_of(sq) + offset); f <= FILE_9; f++) bb |= FILE_BB[f]; break;
			case Effect8::DIRECT_U: for (Rank r = Rank(rank_of(sq) - offset); r >= RANK_1; r--) bb |= RANK_BB[r]; break;
			case Effect8::DIRECT_D: for (Rank r = Rank(rank_of(sq) + offset); r <= RANK_9; r++) bb |= RANK_BB[r]; break;
			default: ASSERT_LV1(false);
			}
			return bb;
		}
		constT constStr;
		// C#のStringBuilderのように用いる
		std::basic_stringstream<typename strT::value_type> ss;
	};

	// KIF形式の指し手表現文字列を取得する。
	std::string to_kif_string(Move m, Piece movedPieceBeforeType, Move prev_m , Color c , SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstLocale> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c , fmt);
	}
	std::string to_kif_u8string(Move m, Piece movedPieceBeforeType, Move prev_m , Color c , SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstUtf8> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c , fmt);
	}
	std::u16string to_kif_u16string(Move m, Piece movedPieceBeforeType, Move prev_m , Color c , SquareFormat fmt)
	{
		KifStringBuilder<std::u16string, KifConstUtf16> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c , fmt);
	}
	std::u32string to_kif_u32string(Move m, Piece movedPieceBeforeType, Move prev_m , Color c , SquareFormat fmt)
	{
		KifStringBuilder<std::u32string, KifConstUtf32> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c , fmt);
	}

	// KIF形式の指し手表現文字列を取得する。

	std::string to_kif_string(Position& pos, Move m , SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstLocale> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove , pos.side_to_move() , fmt);
	}
	std::string to_kif_u8string(Position& pos, Move m , SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstUtf8> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove , pos.side_to_move() , fmt);
	}
	std::u16string to_kif_u16string(Position& pos, Move m , SquareFormat fmt)
	{
		KifStringBuilder<std::u16string, KifConstUtf16> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove , pos.side_to_move() , fmt);
	}
	std::u32string to_kif_u32string(Position& pos, Move m , SquareFormat fmt)
	{
		KifStringBuilder<std::u32string, KifConstUtf32> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove , pos.side_to_move() , fmt);
	}

	// KIF2形式の指し手表現文字列を取得する。
	//   KIF2形式では、"同金左"のように表現しないといけないから、
	//   同種の他の駒の位置関係がわかる必要があり、
	//   盤面情報が必須であるから、Positionクラスが必要になる。

	std::string to_kif2_string(Position& pos, Move m, SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstLocale> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::string to_kif2_u8string(Position& pos, Move m, SquareFormat fmt)
	{
		KifStringBuilder<std::string, KifConstUtf8> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::u16string to_kif2_u16string(Position& pos, Move m, SquareFormat fmt)
	{
		KifStringBuilder<std::u16string, KifConstUtf16> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::u32string to_kif2_u32string(Position& pos, Move m, SquareFormat fmt)
	{
		KifStringBuilder<std::u32string, KifConstUtf32> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}

	// -----------------
	//	   UnitTest
	// -----------------

	void UnitTest()
	{
		is_ready();

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをCSA文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa_string(pos, m.move) << " ";
#endif

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをCSA文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa1_string(pos, m.move) << " ";
#endif

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをKIF文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_kif_string(pos, m.move , SqFmt_FullWidthMix) << " ";
#endif

#if 1
		// 初期局面ですべての合法な指し手を生成し、それをKIF2文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate();
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_kif2_string(pos, m.move, SqFmt_FullWidthMix) << " ";
#endif

	}

} // namespace KifConvertTools

#endif
