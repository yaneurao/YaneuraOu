#include "../config.h"

#if defined (USE_SFEN_PACKER)

#include "../misc.h"
#include "../position.h"

#include <sstream>
#include <fstream>
#include <cstring>	// std::memset()

using namespace std;

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
//    空     xxxxx0 + 0    (none)
//    歩     xxxx01 + 2    xxxx0 + 2
//    香     xx0011 + 2    xx001 + 2
//    桂     xx1011 + 2    xx101 + 2
//    銀     xx0111 + 2    xx011 + 2
//    金     x01111 + 1    x0111 + 1 // 金は成りフラグはない。
//    角     011111 + 2    01111 + 2
//    飛     111111 + 2    11111 + 2
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
// 手駒は、盤上の駒より1bit少なく表現できるので結局、全体のbit数に変化はない。
// ゆえに、この表現において、どんな局面でもこのbit数で表現できる。
// 手駒に成りフラグは不要だが、これも含めておくと盤上の駒のbit数-1になるので
// 全体のbit数が固定化できるのでこれも含めておくことにする。

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

    memset(data, 0, 32 /* 256bit */);
    stream.set_data(data);

    // 手番
    stream.write_one_bit((int)(pos.side_to_move()));

    // 先手玉、後手玉の位置、それぞれ7bit
    for(auto c : COLOR)
      stream.write_n_bit(pos.king_square(c), 7);

    // 盤面の駒は王以外はそのまま書き出して良し！
    for (auto sq : SQ)
    {
      // 盤上の玉以外の駒をハフマン符号化して書き出し
      Piece pc = pos.piece_on(sq);
      if (type_of(pc) == KING)
        continue;
      write_board_piece_to_stream(pc);
    }

    // 手駒をハフマン符号化して書き出し
    for (auto c : COLOR)
      for (PieceType pr = PAWN; pr < KING; ++pr)
      {
        int n = hand_count(pos.hand_of(c), pr);

        // この駒、n枚持ってるよ
        for (int i = 0; i < n; ++i)
          write_hand_piece_to_stream(make_piece(c, pr));
      }

    // 綺麗に書けた..気がする。

    // 全部で256bitのはず。(普通の盤面であれば)
    ASSERT_LV3(stream.get_cursor() == 256);
  }

  // data[32]をsfen化して返す。
  string unpack()
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
      // 256になるまで手駒が格納されているはず
      auto pc = read_hand_piece_from_stream();
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

    // 金以外は手駒であっても不成を出力して、盤上の駒のbit数-1を保つ
    if (pr != GOLD)
      stream.write_one_bit(false);

    // 先後フラグ
    stream.write_one_bit(color_of(pc));
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

    // 金以外であれば成りフラグを1bit捨てる
    if (pr != GOLD)
      stream.read_one_bit();

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
Tools::Result Position::set_from_packed_sfen(const PackedSfen& sfen , StateInfo * si, Thread* th, bool mirror , int gamePly_ /* = 0 */)
{
	SfenPacker packer;
	auto& stream = packer.stream;
	stream.set_data((u8*)&sfen);

	std::memset(this, 0, sizeof(Position));
	std::memset(si, 0, sizeof(StateInfo));
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

		put_piece(sq, Piece(pc));

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

	set_state(st);

	// --- effect

#if defined (LONG_EFFECT_LIBRARY)
	// 利きの全計算による更新
	LongEffect::calc_effect(*this);
#endif

	// --- evaluate

	st->materialValue = Eval::material(*this);
	Eval::compute_eval(*this);

//	sync_cout << sfen() << *this << pieces(BLACK) << pieces(WHITE) << pieces() << sync_endl;

	//if (!is_ok(*this))
	//	std::cout << "info string Illigal Position?" << endl;

	thisThread = th;

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


#endif // USE_SFEN_PACKER

