#if defined(USE_KIF_CONVERT_TOOLS)

#include "kif_convert_tools.h"
#include "kif_convert_consts.h"

#include "../../position.h"

#include <iomanip>

extern void is_ready();

namespace KifConvertTools
{
	// -----------------
	//  for SFEN format
	// -----------------
	struct SfenStringBuilder
	{
		// SFEN形式の棋譜文字列に変換して返す。
		std::string to_sfen_string(Position& pos)
		{
			StateInfo* p = pos.state();
			std::stack<StateInfo*> states;
			while (p->previous != nullptr)
			{
				states.push(p);
				pos.undo_move(p->lastMove);
				p = p->previous;
			}
			auto inisfen = pos.sfen();
			if (inisfen == SFEN_HIRATE)
				ss << "position startpos moves";
			else
				ss << "position sfen " << inisfen << " moves";
			while (states.size())
			{
				auto& top = states.top();
				Move m = top->lastMove;
				ss << " " << m;
				pos.do_move(m, *top);
				states.pop();
			}
			ss << std::endl;
			return ss.str();
		}

	private:
		// C#のStringBuilderのように用いる
		std::stringstream ss;
	};

	std::string to_sfen_string(Position& pos)
	{
		SfenStringBuilder builder;
		return builder.to_sfen_string(pos);
	}

	// -----------------
	//  for CSA format
	// -----------------

	// CSA文字列の構築器
	template <class constT>
	struct CsaStringBuilder
	{
		typedef typename constT::string_type string_type;
		// 升目をCSA形式の文字列に変換して、ssに追加する。
		void append(Square sq)
		{
			ss << constStr.char1to9_ascii[file_of(sq)];
			ss << constStr.char1to9_ascii[rank_of(sq)];
		}

		// 手番をCSA形式の文字列に変換して、ssに追加する。
		void append(Color c, const KifFormat& fmt)
		{
			if (fmt.color_type == ColorFmt_CSA)
				ss << ((c == BLACK) ? constStr.csa_color_black : constStr.csa_color_white);
		}

		// 駒名をCSA形式の文字列に変換して、ssに追加する。
		void append(Piece pt)
		{
			ASSERT_LV3(type_of(pt) == pt);
			ss << constStr.csa_piece[pt];
		}

		void append(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
		{
			switch (m)
			{
			case MOVE_NONE:
			case MOVE_NULL:
				ss << constStr.csa_move_none;
				break;
			case MOVE_RESIGN:
				ss << constStr.csa_move_resign;
				break;
			case MOVE_WIN:
				ss << constStr.csa_move_win;
				break;
			default:
				// --- 普通の指し手

				// 手番
				append(c, fmt);

				// 打つ指し手のときは移動元の升は"00"と表現する。
				// さもなくば、"77"のように移動元の升目をアラビア数字で表現。
				if (is_drop(m))
					ss << constStr.csa_cap_sq;
				else
					append(move_from(m));

				append(move_to(m));
				append(type_of(movedPieceAfterType));
			}
		}

		// CSA形式の局面文字列に変換
		void append(const Position& pos, const KifFormat& fmt)
		{
			// 盤面
			for (Rank r = RANK_1; r <= RANK_9; ++r)
			{
				ss << constStr.csa_pos_rank[r];
				for (File f = FILE_9; f >= FILE_1; --f)
					ss << constStr.csa_pos_piece[pos.piece_on(f | r)];
				ss << std::endl;
			}
			// 持駒
			for (Color c : { BLACK, WHITE })
			{
				if (pos.hand_of(c) == HAND_ZERO)
					continue;
				ss << ((c == BLACK) ? constStr.csa_hand_black : constStr.csa_hand_white);
				for (Piece pc : { ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN })
				{
					int cnt = hand_count(pos.hand_of(c), pc);
					for (int i = 0; i < cnt; ++i)
						ss << constStr.csa_cap_sq << constStr.csa_piece[pc];
				}
				ss << std::endl;
			}
			// 手番
			ss << ((pos.side_to_move() == BLACK) ? constStr.csa_color_black : constStr.csa_color_white) << std::endl;
		}

		// csa形式の指し手文字列に変換して返す。
		string_type to_csa_string(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
		{
			append(m, movedPieceAfterType, c, fmt);
			return ss.str();
		}

		// csa形式の棋譜文字列に変換して返す。
		string_type to_csa_string(Position& pos, const KifFormat& fmt)
		{
			StateInfo* p = pos.state();
			std::stack<StateInfo*> states;
			while (p->previous != nullptr)
			{
				states.push(p);
				pos.undo_move(p->lastMove);
				p = p->previous;
			}
			ss << constStr.csa_ver << std::endl;
			if (pos.game_ply() == 1 && pos.side_to_move() == BLACK && pos.sfen() == SFEN_HIRATE)
			{
				ss << constStr.csa_pos_hirate << std::endl;
				ss << ((pos.side_to_move() == BLACK) ? constStr.csa_color_black : constStr.csa_color_white) << std::endl;
			}
			else
				append(pos, fmt);
			while (states.size())
			{
				auto& top = states.top();
				Move m = top->lastMove;
				append(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
				ss << std::endl;
				pos.do_move(m, *top);
				states.pop();
			}
			return ss.str();
		}

	private:
		constT constStr;
		// C#のStringBuilderのように用いる
		EalenSstream<typename constT::char_type> ss;
	};

	// CSA形式の指し手文字列を取得する。

	std::string to_csa_string(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstLocale> builder;
		return builder.to_csa_string(m, movedPieceAfterType, c, fmt);
	}
	std::string to_csa_u8string(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf8> builder;
		return builder.to_csa_string(m, movedPieceAfterType, c, fmt);
	}
	std::u16string to_csa_u16string(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf16> builder;
		return builder.to_csa_string(m, movedPieceAfterType, c, fmt);
	}
	std::u32string to_csa_u32string(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf32> builder;
		return builder.to_csa_string(m, movedPieceAfterType, c, fmt);
	}
	std::wstring to_csa_wstring(Move m, Piece movedPieceAfterType, Color c, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstWchar> builder;
		return builder.to_csa_string(m, movedPieceAfterType, c, fmt);
	}

	std::string to_csa_string(Position& pos, Move m, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstLocale> builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
	}
	std::string to_csa_u8string(Position& pos, Move m, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf8> builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
	}
	std::u16string to_csa_u16string(Position& pos, Move m, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf16> builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
	}
	std::u32string to_csa_u32string(Position& pos, Move m, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf32> builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
	}
	std::wstring to_csa_wstring(Position& pos, Move m, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstWchar> builder;
		return builder.to_csa_string(m, pos.moved_piece_after(m), pos.side_to_move(), fmt);
	}

	// CSA形式の棋譜文字列を取得する。

	std::string to_csa_string(Position& pos, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstLocale> builder;
		return builder.to_csa_string(pos, fmt);
	}
	std::string to_csa_u8string(Position& pos, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf8> builder;
		return builder.to_csa_string(pos, fmt);
	}
	std::u16string to_csa_u16string(Position& pos, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf16> builder;
		return builder.to_csa_string(pos, fmt);
	}
	std::u32string to_csa_u32string(Position& pos, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstUtf32> builder;
		return builder.to_csa_string(pos, fmt);
	}
	std::wstring to_csa_wstring(Position& pos, const KifFormat& fmt)
	{
		CsaStringBuilder<KifConstWchar> builder;
		return builder.to_csa_string(pos, fmt);
	}

	// -----------------
	//  for KIF format
	// -----------------

	// KIF文字列の構築器
	// (CsaStringBuilderと同じ作り)
	template <class constT>
	struct KifStringBuilder
	{
		typedef typename constT::string_type string_type;

		void append(Rank r, const KifFormat& fmt)
		{
			switch (fmt.square_type)
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

		void append(File f, const KifFormat& fmt)
		{
			switch (fmt.square_type)
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
		void append(Square sq, const KifFormat& fmt)
		{
			append(file_of(sq), fmt);
			append(rank_of(sq), fmt);
		}

		// 手番をKIF形式の文字列に変換して、ssに追加する。
		void append(Color c, const KifFormat& fmt)
		{
			switch (fmt.color_type)
			{
			case ColorFmt_CSA:
				ss << ((c == BLACK) ? constStr.csa_color_black : constStr.csa_color_white);
				break;
			case ColorFmt_KIF:
				ss << ((c == BLACK) ? constStr.kif_color_black : constStr.kif_color_white);
				break;
			case ColorFmt_Piece:
				ss << ((c == BLACK) ? constStr.piece_color_black : constStr.piece_color_white);
				break;
			default:;
			}
		}

		// 駒名をKIF形式の文字列に変換して、ssに追加する。
		void append(Piece pt)
		{
			ASSERT_LV3(type_of(pt) == pt);
			ss << constStr.kif_piece[pt];
		}

		// KIF形式の指し手文字列に変換
		// m			  : 今回の指し手
		// prev_m		  : 直前の指し手
		// movedPieceType : 移動させる駒(今回の指し手で成る場合は、成る前の駒)
		void append(Move m, Piece movedPieceType, Move prev_m, Color c, const KifFormat& fmt)
		{
			// 手番
			append(c, fmt);

			switch (m)
			{
			case MOVE_NONE:
				ss << constStr.kif_move_none;
				break;
			case MOVE_NULL:
				ss << constStr.kif_move_null;
				break;
			case MOVE_RESIGN:
				ss << constStr.kif_move_resign;
				break;
			case MOVE_WIN:
				ss << constStr.kif_move_win;
				break;
			default:
				// --- 普通の指し手

				// 一つ前の指し手の移動先と、今回の移動先が同じ場合、"同"金のように表示する。
				if (is_ok(prev_m) && move_to(prev_m) == move_to(m))
				{
					// SamePosFmt_Verbose の場合、座標＋「同」～の表記にする
					if (fmt.samepos_type == SamePosFmt_Verbose)
						append(move_to(m), fmt);
					ss << constStr.kif_move_samepos;
					// SamePosFmt_KIFsp の場合、「同」の後ろに全角空白を入れる
					// SamePosFmt_KI2sp の場合、成り指し手、成香・成桂・成銀の指し手では全角空白を入れない
					if (fmt.samepos_type == SamePosFmt_KIFsp || (fmt.samepos_type == SamePosFmt_KI2sp && !(is_promote(m)) && type_of(movedPieceType) != PRO_LANCE && type_of(movedPieceType) != PRO_KNIGHT && type_of(movedPieceType) != PRO_KNIGHT))
						ss << constStr.kif_fwsp;
				}
				else
					append(move_to(m), fmt);
				append(type_of(movedPieceType));
				if (is_drop(m))
					ss << constStr.kif_move_drop;
				else
				{
					if (is_promote(m))
						ss << constStr.kif_move_promote;
					// 参考用に駒の移動元を括弧表記で出力することになっている。
					ss << constStr.kif_lbrack;
					append(move_from(m), KifFmtA);
					ss << constStr.kif_rbrack;
				}
			}
		}

		// BOD形式の局面文字列に変換
		void append(Position& pos, const KifFormat& fmt)
		{
			// 後手の持駒
			ss << constStr.bod_hand_color_white;
			if (pos.hand_of(WHITE) == HAND_ZERO)
				ss << constStr.bod_hand_none;
			else
				for (Piece pc : { ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN })
				{
					int cnt = hand_count(pos.hand_of(WHITE), pc);
					if (cnt > 0)
						ss << constStr.bod_hand_piece[pc] << constStr.bod_hand_num[cnt] << constStr.bod_hand_pad;
				}
			ss << std::endl;
			// 盤面
			ss << constStr.bod_fline << std::endl;
			ss << constStr.bod_hline << std::endl;
			for (Rank r = RANK_1; r <= RANK_9; ++r)
			{
				ss << constStr.bod_vline;
				for (File f = FILE_9; f >= FILE_1; --f)
					ss << constStr.bod_piece[pos.piece_on(f | r)];
				ss << constStr.bod_vline << constStr.bod_rank[r] << std::endl;
			}
			ss << constStr.bod_hline << std::endl;
			// 先手の持駒
			ss << constStr.bod_hand_color_black;
			if (pos.hand_of(BLACK) == HAND_ZERO)
				ss << constStr.bod_hand_none;
			else
				for (Piece pc : { ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN })
				{
					int cnt = hand_count(pos.hand_of(BLACK), pc);
					if (cnt > 0)
						ss << constStr.bod_hand_piece[pc] << constStr.bod_hand_num[cnt] << constStr.bod_hand_pad;
				}
			ss << std::endl;
			// TODO: 現在手数、直前の指し手出力
			// 後手番のみ追加行
			if (pos.side_to_move() == WHITE)
				ss << constStr.bod_turn_white << std::endl;
		}

		// KIF形式の指し手文字列に変換して返す。
		// m			  : 今回の指し手
		// prev_m		  : 直前の指し手
		// movedPieceType : 移動させる駒(今回の指し手で成る場合は、成る前の駒)
		string_type to_kif_string(Move m, Piece movedPieceType, Move prev_m, Color c, const KifFormat& fmt)
		{
			append(m, movedPieceType, prev_m, c, fmt);
			return ss.str();
		}

		// KIF形式の棋譜文字列に変換して返す。
		string_type to_kif_string(Position& pos, const KifFormat& fmt)
		{
			StateInfo* p = pos.state();
			std::stack<StateInfo*> states;
			while (p->previous != nullptr)
			{
				states.push(p);
				pos.undo_move(p->lastMove);
				p = p->previous;
			}
			if (pos.game_ply() == 1 && pos.side_to_move() == BLACK && pos.sfen() == SFEN_HIRATE)
				ss << constStr.kiflist_hirate << std::endl;
			else
				append(pos, fmt);
			ss << constStr.kiflist_head << std::endl;
			while (states.size())
			{
				auto& top = states.top();
				Move m = top->lastMove;
				ss << std::setw(4) << pos.game_ply();
				ss << constStr.kiflist_pad;
				size_t mpos0 = ss.ealen();
				append(m, pos.moved_piece_before(m), pos.state()->lastMove, pos.side_to_move(), fmt);
				while (ss.ealen() < mpos0 + 18) { ss << constStr.kiflist_pad; }
				ss << constStr.kiflist_spendnotime << std::endl;
				pos.do_move(m, *top);
				states.pop();
			}
			return ss.str();
		}

		// KIF2形式の指し手文字列に変換して返す。
		// pos : 現局面
		// m   : 今回の指し手
		// fmt : 書式設定
		void append(Position& pos, Move m, const KifFormat& fmt)
		{
			// 手番
			Color c = pos.side_to_move();
			append(c, fmt);

			switch (m)
			{
			case MOVE_NONE:
				ss << constStr.kif_move_none;
				break;
			case MOVE_NULL:
				ss << constStr.kif_move_null;
				break;
			case MOVE_RESIGN:
				ss << constStr.kif_move_resign;
				break;
			case MOVE_WIN:
				ss << constStr.kif_move_win;
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
				{
					// SamePosFmt_Verbose の場合、座標＋「同」～の表記にする
					if (fmt.samepos_type == SamePosFmt_Verbose)
						append(to, fmt);
					ss << constStr.kif_move_samepos;
					// SamePosFmt_KIFsp の場合、「同」の後ろに全角空白を入れる
					// SamePosFmt_KI2sp の場合、成り指し手、成香・成桂・成銀の指し手では全角空白を入れない
					if (fmt.samepos_type == SamePosFmt_KIFsp || (fmt.samepos_type == SamePosFmt_KI2sp && !(is_promote(m)) && pt != PRO_LANCE && pt != PRO_KNIGHT && pt != PRO_KNIGHT))
						ss << constStr.kif_fwsp;
				}
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
				auto dir_bb = [c, to](Direct dir) { return dir_bb_(c, to, dir, false); };
				auto dir_or_bb = [c, to](Direct dir) { return dir_bb_(c, to, dir, true); };

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
								ss << constStr.kif_move_slide;
							// 右から寄るなら「右」
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.kif_move_right;
								// 右から動く駒がそれだけでは無いならさらに「寄」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.kif_move_slide;
							}
							// 左から寄るなら「左」
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.kif_move_left;
								// 左から動く駒がそれだけでは無いならさらに「寄」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.kif_move_slide;
							}
						}
						// 前に上がるMoveの場合
						else if ((fromBB & dir_bb(DIRECT_D)) == fromBB) {
							// 前に上がる同種駒がそれだけなら「上」
							if ((attackerBB & dir_bb(DIRECT_D)) == fromBB)
								ss << constStr.kif_move_upper;
							// 真っ直ぐ上がるMoveの場合
							else if (from_file == to_file) {
								// 金・銀・成金なら「直」
								if (is_like_goldsilver)
									ss << constStr.kif_move_straight;
								// 同じ筋・より右の筋に他に動ける駒がないなら「右」
								else if ((attackerBB & dir_or_bb(DIRECT_R)) == fromBB)
									ss << constStr.kif_move_right;
								// 同じ筋・より左の筋に他に動ける駒がないなら「左」
								else if ((attackerBB & dir_or_bb(DIRECT_L)) == fromBB)
									ss << constStr.kif_move_left;
								// 「右上」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_R) & dir_bb(DIRECT_D)) == fromBB)
									ss << constStr.kif_move_right << constStr.kif_move_upper;
								// 「左上」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_L) & dir_bb(DIRECT_D)) == fromBB)
									ss << constStr.kif_move_left << constStr.kif_move_upper;
							}
							// 右から上がるMoveの場合
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.kif_move_right;
								// 右から動ける駒が他にもあればさらに「上」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.kif_move_upper;
							}
							// 左から上がるMoveの場合
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.kif_move_left;
								// 左から動ける駒が他にもあればさらに「上」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.kif_move_upper;
							}
						}
						// 後ろに引くMoveの場合
						else if ((fromBB & ForwardRanksBB[c][to_rank]) == fromBB) {
							// 後ろに引く同種駒がそれだけなら
							if ((attackerBB & ForwardRanksBB[c][to_rank]) == fromBB)
								ss << constStr.kif_move_lower;
							// 真っ直ぐ引くMoveの場合
							else if (from_file == to_file) {
								// 同じ筋・より右の筋に他に動ける駒がないなら「右」
								if ((attackerBB & dir_or_bb(DIRECT_R)) == fromBB)
									ss << constStr.kif_move_right;
								// 同じ筋・より左の筋に他に動ける駒がないなら「左」
								else if ((attackerBB & dir_or_bb(DIRECT_L)) == fromBB)
									ss << constStr.kif_move_left;
								// 「右引」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_R) & ForwardRanksBB[c][to_rank]) == fromBB)
									ss << constStr.kif_move_right << constStr.kif_move_lower;
								// 「左引」の判定
								else if ((attackerBB & dir_or_bb(DIRECT_L) & ForwardRanksBB[c][to_rank]) == fromBB)
									ss << constStr.kif_move_left << constStr.kif_move_lower;
							}
							// 右から引くMoveの場合
							else if ((fromBB & dir_bb(DIRECT_R)) == fromBB) {
								ss << constStr.kif_move_right;
								// 右から動ける駒が他にもあればさらに「引」
								if ((attackerBB & dir_bb(DIRECT_R)) != fromBB)
									ss << constStr.kif_move_lower;
							}
							// 左から引くMoveの場合
							else if ((fromBB & dir_bb(DIRECT_L)) == fromBB) {
								ss << constStr.kif_move_left;
								// 左から動ける駒が他にもあればさらに「引」
								if ((attackerBB & dir_bb(DIRECT_L)) != fromBB)
									ss << constStr.kif_move_lower;
							}
						}
					}

					// 成ったなら必ず「成」
					if (is_promote(m))
						ss << constStr.kif_move_promote;
					// 成れるのに成らなかったなら「不成」
					else if (pt < GOLD && canPromote(c, from, to))
						ss << constStr.kif_move_not << constStr.kif_move_promote;

				}
				// そこへ移動できる同種駒があるなら「打」
				else if (attackerBB != ZERO_BB)
					ss << constStr.kif_move_drop;

				break;
			}
		}

		// KIF2形式の指し手文字列に変換して返す。
		// pos : 現局面
		// m   : 今回の指し手
		// fmt : 書式設定
		string_type to_kif2_string(Position& pos, Move m, const KifFormat& fmt)
		{
			append(pos, m, fmt);
			return ss.str();
		}

		// KIF2形式の棋譜文字列に変換して返す。
		string_type to_kif2_string(Position& pos, const KifFormat& fmt)
		{
			StateInfo* p = pos.state();
			std::stack<StateInfo*> states;
			while (p->previous != nullptr)
			{
				states.push(p);
				pos.undo_move(p->lastMove);
				p = p->previous;
			}
			if (pos.game_ply() == 1 && pos.side_to_move() == BLACK && pos.sfen() == SFEN_HIRATE)
				ss << constStr.kiflist_hirate << std::endl;
			else
				append(pos, fmt);
			size_t mpos0 = ss.ealen();
			size_t mnum = 0;
			while (states.size())
			{
				auto& top = states.top();
				Move m = top->lastMove;
				while (ss.ealen() < mpos0 + mnum * 14) { ss << constStr.kiflist_pad; }
				append(pos, m, fmt);
				if (++mnum > 5)
				{
					ss << std::endl;
					mpos0 = ss.ealen();
					mnum = 0;
				}
				pos.do_move(m, *top);
				states.pop();
			}
			if (mnum > 0) { ss << std::endl; }
			return ss.str();
		}

	private:
		// KIF2形式で、"同金右"のような出力をしないといけないので、ある升の右側に同種の駒があるかを
		// 判定しなければならず、そのためにある升のX側を示すBitboardが必要となる。
		// sqの升を中心として、dir側の方角がすべて1になっているBitboardが返る。
		// ただし、c == WHITEのときは盤面を180度回転させてdirの方角を考える。
		// またor_flag == trueのときは、「sqの升を含む 筋 or 段」も1であるBitboardが返る。
		static Bitboard dir_bb_(Color color, Square sq, Effect8::Direct dir, bool or_flag)
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
		EalenSstream<typename constT::char_type> ss;
	};

	// KIF形式の指し手表現文字列を取得する。
	std::string to_kif_string(Move m, Piece movedPieceBeforeType, Move prev_m, Color c, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstLocale> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c, fmt);
	}
	std::string to_kif_u8string(Move m, Piece movedPieceBeforeType, Move prev_m, Color c, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf8> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c, fmt);
	}
	std::u16string to_kif_u16string(Move m, Piece movedPieceBeforeType, Move prev_m, Color c, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf16> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c, fmt);
	}
	std::u32string to_kif_u32string(Move m, Piece movedPieceBeforeType, Move prev_m, Color c, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf32> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c, fmt);
	}
	std::wstring to_kif_wstring(Move m, Piece movedPieceBeforeType, Move prev_m, Color c, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstWchar> builder;
		return builder.to_kif_string(m, movedPieceBeforeType, prev_m, c, fmt);
	}

	// KIF形式の指し手表現文字列を取得する。

	std::string to_kif_string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstLocale> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove, pos.side_to_move(), fmt);
	}
	std::string to_kif_u8string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf8> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove, pos.side_to_move(), fmt);
	}
	std::u16string to_kif_u16string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf16> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove, pos.side_to_move(), fmt);
	}
	std::u32string to_kif_u32string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf32> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove, pos.side_to_move(), fmt);
	}
	std::wstring to_kif_wstring(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstWchar> builder;
		return builder.to_kif_string(m, type_of(pos.moved_piece_before(m)), pos.state()->lastMove, pos.side_to_move(), fmt);
	}

	// KIF形式の棋譜文字列を取得する。

	std::string to_kif_string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstLocale> builder;
		return builder.to_kif_string(pos, fmt);
	}
	std::string to_kif_u8string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf8> builder;
		return builder.to_kif_string(pos, fmt);
	}
	std::u16string to_kif_u16string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf16> builder;
		return builder.to_kif_string(pos, fmt);
	}
	std::u32string to_kif_u32string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf32> builder;
		return builder.to_kif_string(pos, fmt);
	}
	std::wstring to_kif_wstring(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstWchar> builder;
		return builder.to_kif_string(pos, fmt);
	}

	// KIF2形式の指し手表現文字列を取得する。
	//   KIF2形式では、"同金左"のように表現しないといけないから、
	//   同種の他の駒の位置関係がわかる必要があり、
	//   盤面情報が必須であるから、Positionクラスが必要になる。

	std::string to_kif2_string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstLocale> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::string to_kif2_u8string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf8> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::u16string to_kif2_u16string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf16> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::u32string to_kif2_u32string(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf32> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}
	std::wstring to_kif2_wstring(Position& pos, Move m, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstWchar> builder;
		return builder.to_kif2_string(pos, m, fmt);
	}

	// KIF2形式の棋譜文字列を取得する。

	std::string to_kif2_string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstLocale> builder;
		return builder.to_kif2_string(pos, fmt);
	}
	std::string to_kif2_u8string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf8> builder;
		return builder.to_kif2_string(pos, fmt);
	}
	std::u16string to_kif2_u16string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf16> builder;
		return builder.to_kif2_string(pos, fmt);
	}
	std::u32string to_kif2_u32string(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstUtf32> builder;
		return builder.to_kif2_string(pos, fmt);
	}
	std::wstring to_kif2_wstring(Position& pos, const KifFormat& fmt)
	{
		KifStringBuilder<KifConstWchar> builder;
		return builder.to_kif2_string(pos, fmt);
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
		pos.set_hirate(Threads.main());
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa_string(pos, m.move) << " ";
#endif

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをCSA文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate(Threads.main());
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_csa1_string(pos, m.move) << " ";
#endif

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをKIF文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate(Threads.main());
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_kif_string(pos, m.move, KifFmtK) << " ";
#endif

#if 0
		// 初期局面ですべての合法な指し手を生成し、それをKIF2文字列として出力してみるテスト。
		Position pos;
		pos.set_hirate(Threads.main());
		for (auto m : MoveList<LEGAL>(pos))
			std::cout << to_kif2_string(pos, m.move, KifFmtK) << " ";
#endif

	}

} // namespace KifConvertTools

#endif
