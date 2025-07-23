#include "../config.h"

#if defined (USE_SFEN_PACKER)

#include "../misc.h"
#include "../position.h"

#include <sstream>
#include <fstream>
#include <cstring>	// std::memset()

using namespace std;
namespace YaneuraOu {

// -----------------------------------
//        局面の圧縮・解凍
// -----------------------------------

// ビットストリームを扱うクラス
// 局面の符号化を行なうときに、これがあると便利
struct BitStream
{
	// データを格納するメモリを事前にセットする。
	// そのメモリは0クリアされているものとする。
	void  set_data(u8* data_) { data = data_; reset(); }

	// set_data()で渡されたポインタの取得。
	u8* get_data() const { return data; }

	// カーソルの取得。
	int get_cursor() const { return bit_cursor; }

	// カーソルのリセット
	void reset() { bit_cursor = 0; }

	// ストリームに1bit書き出す。
	// bは非0なら1を書き出す。0なら0を書き出す。
	FORCE_INLINE void write_one_bit(int b)
	{
	if (b)
		data[bit_cursor / 8] |= 1 << (bit_cursor & 7);

	++bit_cursor;
	}

	// ストリームから1ビット取り出す。
	FORCE_INLINE int read_one_bit()
	{
	int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
	++bit_cursor;

	return b;
	}

	// nビットのデータを書き出す
	// データはdの下位から順に書き出されるものとする。
	void write_n_bit(int d, int n)
	{
	for (int i = 0; i < n; ++i)
		write_one_bit(d & (1 << i));
	}

	// nビットのデータを読み込む
	// write_n_bit()の逆変換。
	int read_n_bit(int n)
	{
	int result = 0;
	for (int i = 0; i < n; ++i)
		result |= read_one_bit() ? (1 << i) : 0;

	return result;
	}

private:
	// 次に読み書きすべきbit位置。
	int bit_cursor;

	// データの実体
	u8* data;
};


//  ハフマン符号化
//   ※　 なのはminiの符号化から、変換が楽になるように単純化。
//
//   盤上の1升(NO_PIECE以外) = 2～6bit ( + 成りフラグ1bit+ 先後1bit )
//   手駒の1枚               = 1～5bit ( + 成りフラグ1bit+ 先後1bit )
//
//          　盤上　　　　　手駒       駒箱
//    空     xxxxxxx0     (none)      (none)
//    歩     xxxxcp01     xxxx000     xxxx010
//    香     xxcp0011     xx00001     xx01001
//    桂     xxcp1011     xx00101     xx01101
//    銀     xxcp0111     xx00011     xx01011
//    金     xxc01111     xx00111     xx11011
//    角     cp011111     0001111     0101111
//    飛     cp111111     0011111     0111111
//
//		c = Color(先後フラグ) , p = promote(成りフラグ)。
//    ただし、金には成りフラグはない。空の升も成りも先後フラグもない。
// 　　・手駒では、盤上の表現から
// 　　　　bit0を削ったあと、p = 0、
// 　　・駒箱では
//         bit0を削ったあと、p = 1 , c = 0 だが、金がこれだと表せないので別の表現が必要。
// 　　　⇨　cshogiに倣う。
// 　　　⇨　金は後手の銀の成った奴扱い
//
//
// すべての駒が盤上にあるとして、
//     空 81 - 40駒 = 41升 = 41bit
//     歩      4bit*18駒   = 72bit
//     香      6bit* 4駒   = 24bit
//     桂      6bit* 4駒   = 24bit
//     銀      6bit* 4駒   = 24bit            
//     金      6bit* 4駒   = 24bit
//     角      8bit* 2駒   = 16bit
//     飛      8bit* 2駒   = 16bit
//                          -------
//                          241bit + 1bit(手番) + 7bit×2(王の位置先後) = 256bit
//
// 盤上の駒が手駒に移動すると盤上の駒が空になるので盤上のその升は1bitで表現でき、
// 手駒は、盤上の駒より1bit少なく表現できる。(手駒には、空はないので、盤上のハフマン符号のbit0を省略できる)
// 
// よって、結局、全体のbit数に変化はない。
//
// ゆえに、この表現において、どんな局面でもこのbit数で表現できる。
// 
// 手駒に成りフラグは不要だが、これも含めておくと盤上の駒のbit数-1になるので
// 全体のbit数が固定化できるのでこれも含めておくことにする。
// 
// PackedSfenの内部データの並び順
//	 ・手番(1bit)
//   ・玉の位置先後(7bit×2)
//   ・盤上の駒(最大で241bit)
//   ・手駒(最小で0bit)
//
// 駒落ちに関しては、cshogiの実装に合わせる
//  https://github.com/TadaoYamaoka/cshogi/commit/ddcaff984fe5734cc97d3528f22ab5d9a69a2f15
// つまり、成りフラグ = 1となっている場合、それは手駒ではなく駒箱を意味するものとする。
//
// 詰将棋盤面で、片方の玉の場合、玉の位置をSQ_NB(81)を設定するものとする。
// ⇨　ただ、そのような盤面には評価関数が対応してないかも知れないが…。
//

struct HuffmanedPiece
{
	int code; // どうコード化されるか
	int bits; // 何bit専有するのか
};

HuffmanedPiece huffman_table[] =
{
	{0x00,1}, // NO_PIECE
	{0x01,2}, // PAWN
	{0x03,4}, // LANCE
	{0x0b,4}, // KNIGHT
	{0x07,4}, // SILVER
	{0x1f,6}, // BISHOP
	{0x3f,6}, // ROOK
	{0x0f,5}, // GOLD
};

// 駒箱用の符号化。cshogiに倣う。
HuffmanedPiece huffman_table_piecebox[] =
{
	{0x00,1}, // not use
	{0x02,2}, // PAWN
	{0x09,4}, // LANCE
	{0x0d,4}, // KNIGHT
	{0x0b,4}, // SILVER
	{0x2f,6}, // BISHOP
	{0x3f,6}, // ROOK
	{0x1b,5}, // GOLD
};


// sfenを圧縮/解凍するためのクラス
// sfenはハフマン符号化をすることで256bit(32bytes)にpackできる。
// このことはなのはminiにより証明された。上のハフマン符号化である。
//
// 内部フォーマット = 手番1bit+王の位置7bit*2 + 盤上の駒(ハフマン符号化) + 手駒(ハフマン符号化)
//
struct SfenPacker
{
	// sfenをpackしてdata[32]に格納する。
	void pack(const Position& pos)
	{
	//    cout << pos;

		// 以下の書き出し順が、GOLD,BISHOP,ROOKの順になるように調整しておく。
		// (cshogiの変換例とバイナリレベルで一致させたいため)
		constexpr PieceType to_apery_pieces[]     = { NO_PIECE_TYPE , PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP , ROOK };

		// 駒箱枚数
		// 最終、余った駒を駒箱として出力する必要がある。
		int32_t hp_count[8] =
		{
			0,
			18/*PAWN*/, 4/*LANCE*/, 4/*KNIGHT*/, 4/*SILVER*/,
			2/*BISHOP*/, 2/*ROOK*/, 4/*GOLD*/
		};

		memset(data, 0, 32 /* 256bit */);
		stream.set_data(data);

		// 手番
		stream.write_one_bit((int)(pos.side_to_move()));

		// 先手玉、後手玉の位置、それぞれ7bit
		for(auto c : COLOR)
			stream.write_n_bit(pos.square<KING>(c), 7);

		// 盤面の駒は王以外はそのまま書き出して良し！
		for (auto sq : SQ)
		{
			// 盤上の玉以外の駒をハフマン符号化して書き出し
			Piece pc = pos.piece_on(sq);
			if (type_of(pc) == KING)
				continue;
			write_board_piece_to_stream(pc);

			// 駒箱から減らす
			hp_count[type_of(raw_of(pc))]--;
		}

		// 手駒をハフマン符号化して書き出し
		for (auto c : COLOR)
			for (PieceType pr = PAWN; pr < KING; ++pr)
			{
				// Aperyの手駒の並び順で列挙するように変更する。
				PieceType pr2 = to_apery_pieces[pr];

				int n = hand_count(pos.hand_of(c), pr2);

				// この駒、n枚持ってるよ
				for (int i = 0; i < n; ++i)
					write_hand_piece_to_stream(make_piece(c, pr2));

				// 駒箱から減らす
				hp_count[pr2] -= n;
			}

		// 最後に駒箱の分を出力
		for (PieceType pr = PAWN ; pr < KING ; ++pr)
		{
			PieceType pr2 = to_apery_pieces[pr];

			int n = hp_count[pr2];

			// この駒、n枚駒箱にあるよ
			for (int i = 0; i < n ; ++i)
				write_piecebox_piece_to_stream(pr2);
		}

		// 綺麗に書けた..気がする。

		// 全部で256bitのはず。(普通の盤面であれば)
		ASSERT_LV3(stream.get_cursor() == 256);
	}

	// data[32]をsfen化して返す。
	std::string unpack()
	{
		stream.set_data(data);

		// 盤上の81升
		Piece board[81];
		memset(board, 0, sizeof(Piece)*81);

		// 手番
		Color turn = (Color)stream.read_one_bit();
    
		// まず玉の位置
		for (auto c : COLOR)
			board[stream.read_n_bit(7)] = make_piece(c, KING);

		// 盤上の駒
		for (auto sq : SQ)
		{
			// すでに玉がいるようだ
			if (type_of(board[sq]) == KING)
				continue;

			board[sq] = read_board_piece_from_stream();

			//cout << sq << ' ' << board[sq] << ' ' << stream.get_cursor() << endl;

			ASSERT_LV3(stream.get_cursor() <= 256);
		}

		// 手駒
		Hand hand[2] = { HAND_ZERO,HAND_ZERO };
		while (stream.get_cursor() != 256)
		{
			// 256になるまで手駒か駒箱の駒が格納されているはず
			auto pc = read_hand_piece_from_stream();

			// 成り駒が返ってきたら、これは駒箱の駒。
			// 例) 駒箱の金 = 後手の成銀
			if (is_promoted(pc))
				continue;

			add_hand(hand[(int)color_of(pc)], type_of(pc));
		}

		// boardとhandが確定した。これで局面を構築できる…かも。
		// Position::sfen()は、board,hand,side_to_move,game_plyしか参照しないので
		// 無理やり代入してしまえば、sfen()で文字列化できるはず。

		return Position::sfen_from_rawdata(board, hand, turn, 0);
	}

	// pack()でpackされたsfen(256bit = 32bytes)
	// もしくはunpack()でdecodeするsfen
	u8 *data; // u8[32];

	//private:
	// Position::set_from_packed_sfen(u8 data[32])でこれらの関数を使いたいので筋は悪いがpublicにしておく。

	BitStream stream;

	// 盤面の駒をstreamに出力する。
	void write_board_piece_to_stream(Piece pc)
	{
		// 駒種
		PieceType pr = raw_type_of(pc);
		auto c = huffman_table[pr];
		stream.write_n_bit(c.code, c.bits);
 
		if (pc == NO_PIECE)
			return;

		// 成りフラグ
		// (金はこのフラグはない)
		if (pr!=GOLD)
			stream.write_one_bit((PIECE_PROMOTE & pc) ? 1 : 0);

		// 先後フラグ
		stream.write_one_bit(color_of(pc));
	}

	// 手駒をstreamに出力する
	void write_hand_piece_to_stream(Piece pc)
	{
		ASSERT_LV3(pc != NO_PIECE);

		// 駒種
		PieceType pr = raw_type_of(pc);
		auto c = huffman_table[pr];
		stream.write_n_bit(c.code >> 1, c.bits - 1);

		// 金以外は手駒であっても成りフラグを(不成で)出力して、盤上の駒のbit数-1を保つ
		if (pr != GOLD)
			stream.write_one_bit(false);

		// 先後フラグ
		stream.write_one_bit(color_of(pc));
	}

	// 駒箱をstreamに出力する。
	void write_piecebox_piece_to_stream(PieceType pr)
	{
		ASSERT_LV3(pr != NO_PIECE_TYPE);

		// 駒種
		auto c = huffman_table_piecebox[pr];
		stream.write_n_bit(c.code , c.bits);

		// 成りフラグを用いて出力しているので、これで成りの分も終わり。

		// 先後フラグ = 0 で書き出す。金は、この先後フラグ消費してる(後手の駒扱い)のでこれで終わり。
		if (pr != GOLD)
			stream.write_one_bit(false);
	}

	// 盤面の駒を1枚streamから読み込む
	Piece read_board_piece_from_stream()
	{
		PieceType pr = NO_PIECE_TYPE;
		int code = 0, bits = 0;
		while (true)
		{
			code |= stream.read_one_bit() << bits;
			++bits;

			ASSERT_LV3(bits <= 6);

			for (pr = NO_PIECE_TYPE; pr < KING; ++pr)
			if (huffman_table[pr].code == code
				&& huffman_table[pr].bits == bits)
				goto Found;
		}
		Found:;
		if (pr == NO_PIECE_TYPE)
			return NO_PIECE;

		// 成りフラグ
		// (金はこのフラグはない)
		bool promote = (pr == GOLD) ? false : stream.read_one_bit();

		// 先後フラグ
		Color c = (Color)stream.read_one_bit();
    
		return make_piece(c, pr + (promote ? PIECE_TYPE_PROMOTE : NO_PIECE_TYPE));
	}

	// 手駒を1枚streamから読み込む
	// 駒箱の駒である場合、成り駒が返ってくることが保証されている。
	Piece read_hand_piece_from_stream()
	{
		PieceType pr = NO_PIECE_TYPE;
		int code = 0, bits = 0;
		while (true)
		{
			code |= stream.read_one_bit() << bits;
			++bits;

			ASSERT_LV3(bits <= 6);

			for (pr = PAWN; pr < KING; ++pr)
				if ((huffman_table[pr].code >> 1) == code
					&& (huffman_table[pr].bits -1) == bits)
				goto Found;
		}
	Found:;
		ASSERT_LV3(pr != NO_PIECE_TYPE);

		// 金以外であれば成りフラグを1bit捨てる(これが1なら駒箱の駒なので成り駒を返す)
		if (pr != GOLD)
		{
			bool promote = stream.read_one_bit();
			if (promote)
				pr = PieceType(pr | PIECE_PROMOTE);
		}

		// 先後フラグ
		Color c = (Color)stream.read_one_bit();

		return make_piece(c, pr);
	}
};

// -----------------------------------
//        Positionクラスに追加
// -----------------------------------

// 高速化のために直接unpackする関数を追加。かなりしんどい。
// packer::unpack()とPosition::set()とを合体させて書く。
Tools::Result Position::set_from_packed_sfen(const PackedSfen& sfen , StateInfo * si, bool mirror , int gamePly_ /* = 0 */)
{
	SfenPacker packer;
	auto& stream = packer.stream;
	stream.set_data((u8*)&sfen);

	std::memset(static_cast<void*>(this), 0, sizeof(Position));
	std::memset(static_cast<void*>(si), 0, sizeof(StateInfo));
	st = si;

	// 手番
	sideToMove = (Color)stream.read_one_bit();

	#if defined(USE_EVAL_LIST)

	// evalListのclear。上でmemsetでゼロクリアしたときにクリアされているが…。
	evalList.clear();

	// PieceListを更新する上で、どの駒がどこにあるかを設定しなければならないが、
	// それぞれの駒をどこまで使ったかのカウンター
	PieceNumber piece_no_count[KING] = { PIECE_NUMBER_ZERO,PIECE_NUMBER_PAWN,PIECE_NUMBER_LANCE,PIECE_NUMBER_KNIGHT,
		PIECE_NUMBER_SILVER, PIECE_NUMBER_BISHOP, PIECE_NUMBER_ROOK,PIECE_NUMBER_GOLD };
	#endif

	kingSquare[BLACK] = kingSquare[WHITE] = SQ_NB;

	// まず玉の位置
	if (mirror)
	{
		for (auto c : COLOR)
			board[Mir((Square)stream.read_n_bit(7))] = make_piece(c, KING);
	}
	else
	{
		for (auto c : COLOR)
			board[stream.read_n_bit(7)] = make_piece(c, KING);
	}

	// 盤上の駒
	for (auto sq : SQ)
	{
		if (mirror)
			sq = Mir(sq);

		// すでに玉がいるようだ
		Piece pc;
		if (type_of(board[sq]) != KING)
		{
			ASSERT_LV3(board[sq] == NO_PIECE);
			pc = packer.read_board_piece_from_stream();
		}
		else
		{
			pc = board[sq];
			board[sq] = NO_PIECE; // いっかい取り除いておかないとput_piece()でASSERTに引っかかる。
		}

		// 駒がない場合もあるのでその場合はスキップする。
		if (pc == NO_PIECE)
			continue;

		put_piece(Piece(pc), sq);

	#if defined(USE_EVAL_LIST)
		// evalListの更新
		PieceNumber piece_no =
			(pc == B_KING) ? PIECE_NUMBER_BKING : // 先手玉
			(pc == W_KING) ? PIECE_NUMBER_WKING : // 後手玉
			piece_no_count[raw_type_of(pc)]++; // それ以外

		evalList.put_piece(piece_no, sq, pc); // sqの升にpcの駒を配置する
	#endif

		//cout << sq << ' ' << board[sq] << ' ' << stream.get_cursor() << endl;

		if (stream.get_cursor() > 256)
			return Tools::Result(Tools::ResultCode::SomeError);
		//ASSERT_LV3(stream.get_cursor() <= 256);
	}

	// 手駒
	hand[BLACK] = hand[WHITE] = (Hand)0;

	int i = 0;
	Piece lastPc = NO_PIECE;

	while (stream.get_cursor() < 256)
	{
		// 256になるまで手駒が格納されているはず
		auto pc = packer.read_hand_piece_from_stream();

		// 成り駒は、無視する。(これは駒箱の駒)
		if (is_promoted(pc))
			continue;

		add_hand(hand[(int)color_of(pc)], type_of(pc));

	#if defined(USE_EVAL_LIST)
		// 何枚目のその駒であるかをカウントしておく。
		if (lastPc != pc)
			i = 0;
		lastPc = pc;

		// FV38などではこの個数分だけpieceListに突っ込まないといけない。
		PieceType rpc = raw_type_of(pc);

		PieceNumber piece_no = piece_no_count[rpc]++;
		ASSERT_LV1(is_ok(piece_no));
		evalList.put_piece(piece_no, color_of(pc), rpc, i++);
	#endif
	}

	if (stream.get_cursor() != 256)
	{
		// こんな局面はおかしい。デバッグ用。
		//cout << "Error : set_from_packed_sfen() , position = " << endl << *this << endl;
		//ASSERT_LV1(false);
		return Tools::Result(Tools::ResultCode::SomeError);
	}

	gamePly = gamePly_;

	// put_piece()したのでこのタイミングでupdate
	// set_state()で駒種別のbitboardを参照するのでそれまでにこの関数を呼び出す必要がある。
	update_bitboards();
	update_kingSquare();

	set_state();

	// --- effect

	#if defined (LONG_EFFECT_LIBRARY)
	// 利きの全計算による更新
	LongEffect::calc_effect(*this);
	#endif

	// --- evaluate

#if defined(USE_CLASSIC_EVAL)
    st->materialValue = Eval::material(*this);
	Eval::compute_eval(*this);
#endif

	// --- 入玉の駒点の設定

	update_entering_point();


	//	sync_cout << sfen() << *this << pieces(BLACK) << pieces(WHITE) << pieces() << sync_endl;

	//if (!is_ok(*this))
	//	std::cout << "info string Illigal Position?" << endl;

	return Tools::Result::Ok();
}

// 盤面と手駒、手番を与えて、そのsfenを返す。
std::string Position::sfen_from_rawdata(Piece board[81], Hand hands[2], Color turn, int gamePly_)
{
	// 内部的な構造体にコピーして、sfen()を呼べば、変換過程がそこにしか依存していないならば
	// これで正常に変換されるのでは…。
	Position pos;

	memcpy(pos.board, board, sizeof(Piece) * 81);
	memcpy(pos.hand, hands, sizeof(Hand) * 2);
	pos.sideToMove = turn;
	pos.gamePly = gamePly_;

	return pos.sfen();

	// ↑の実装、美しいが、いかんせん遅い。
	// 棋譜を大量に読み込ませて学習させるときにここがボトルネックになるので直接unpackする関数を書く。
}

// packされたsfenを得る。引数に指定したバッファに返す。
void Position::sfen_pack(PackedSfen& sfen)
{
	SfenPacker sp;
	sp.data = (u8*)&sfen;
	sp.pack(*this);
}

// packされたsfenを解凍する。sfen文字列が返る。
std::string Position::sfen_unpack(const PackedSfen& sfen)
{
	SfenPacker sp;
	sp.data = (u8*)&sfen;
	return sp.unpack();
}

} // namespace YaneuraOu

#endif // USE_SFEN_PACKER

