#ifndef _LONG_EFFECT_H_
#define _LONG_EFFECT_H_

// 遠方駒による利きのライブラリ
// Bitboard/Byteboardに対して8近傍,24近傍を高速に求める等

#include "../types.h"

#if defined (LONG_EFFECT_LIBRARY)

struct Bitboard;

// ----------------------
// 8近傍関係のライブラリ
// ----------------------
namespace Effect8
{
  // --- Directions

  // *** shogi.hに持って行ったものはここではコメントアウトしてある。 ***

  // 方角を表す。遠方駒の利きや、玉から見た方角を表すのに用いる。
  // bit0..右上、bit1..右、bit2..右下、bit3..上、bit4..下、bit5..左上、bit6..左、bit7..左下
  // 同時に複数のbitが1であることがありうる。

  //  enum Directions : uint8_t { DIRECTIONS_ZERO = 0 };

  const uint32_t DIRECTIONS_NB = 256; // Directionsの範囲(uint8_t)で表現できない値なので外部に出しておく。

  // Bitboardのsq周辺の8近傍の状態を8bitに直列化する。
  // ただし盤外に相当するbitの値は不定。盤外を0にしたいのであれば、Effect8::board_mask(sq)と & すること。
  Directions around8(const Bitboard& b, Square sq);

  // around8()などである升の8近傍の情報を回収したときに壁の位置のマスクが欲しいときがあるから、そのマスク情報。
  // 壁のところが0、盤面上が1になっている。
  extern Directions board_mask_table[SQ_NB_PLUS1];
  inline Directions board_mask(Square sq) { return board_mask_table[sq]; }


  // sq1にとってsq2がどのdirectionにあるか。
  //extern Directions direc_table[SQ_NB][SQ_NB];
  //inline Directions directions_of(Square sq1, Square sq2) { return direc_table[sq1][sq2]; }

  // sq1とsq2が桂の移動元、移動先の関係にあることがわかっているときにsq1から見たsq2の方角を返す。
  template <Color c> Direct direct_knight_of(Square sq1, Square sq2) {
    return Direct(DIRECT_RUU + (( sq1 < sq2 ) ? 1 : 0) + ((c == WHITE) ? 0 : 2));
  }

  // ...
  // .+.  3×3のうち、中央の情報はDirectionsは持っていないので'+'を出力して、
  // ...  8近傍は、1であれば'*'、さもなくば'.'を出力する。
  //
  std::ostream& operator<<(std::ostream& os, Directions d);

  // --- Direct

  // Directionsをpopしたもの。複数の方角を同時に表すことはない。
  // おまけで桂馬の移動も追加しておく。
  //enum Direct {
  //  DIRECT_RU, DIRECT_R, DIRECT_RD, DIRECT_U, DIRECT_D, DIRECT_LU, DIRECT_L, DIRECT_LD,
  //  DIRECT_NB, DIRECT_ZERO = 0, DIRECT_RUU = 8, DIRECT_LUU, DIRECT_RDD, DIRECT_LDD, DIRECT_NB_PLUS4
  //};
  // inline bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB; }

  // Directionsに相当するものを引数に渡して1つ方角を取り出す。
  // inline Direct pop_directions(Directions& d) { return (Direct)pop_lsb(*(uint8_t*)&d); }

  // dの方角が右上、右下、左上、左下であるか。
  static_assert(DIRECT_RU == 0 && DIRECT_RD == 2 && DIRECT_LU == 5 && DIRECT_LD == 7, "ERROR , DIRECT_DIAG");
  inline bool is_diag(Direct d) { return ((d & 1) - (d >> 2))==0; /* (d==0 || d==2 || d==5 || d==7) */}

  // DirectをSquare型の差分値で表現したもの。
  const Square DirectToDelta_[DIRECT_NB_PLUS4] = { SQ_RU,SQ_R,SQ_RD,SQ_U,SQ_D,SQ_LU,SQ_L,SQ_LD,SQ_RUU,SQ_LUU,SQ_RDD,SQ_LDD, };
  inline Square DirectToDelta(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDelta_[d]; }

  // DirectからDirectionsへの逆変換
  // inline Directions to_directions(Direct d) { return Directions(1 << d); }

  // DirectをSquareWithWall型の差分値で表現したもの。
  //const SquareWithWall DirectToDeltaWW_[DIRECT_NB] = { SQWW_RU,SQWW_R,SQWW_RD,SQWW_U,SQWW_D,SQWW_LU,SQWW_L,SQWW_LD, };
  //inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }

  // → shogi.hに持って行った。

  ENABLE_BIT_OPERATORS_ON(Directions)
  ENABLE_RANGE_OPERATORS_ON(Direct, DIRECT_ZERO, DIRECT_NB)

  // --- 8近傍に関する利き

  // 駒pcを敵玉から見てdirの方向に置いたときの敵玉周辺に対するの利き。利きがある場所が0、ない場所が1。
  extern Directions piece_effect_not_table[PIECE_NB][DIRECT_NB];
  inline Directions piece_effect_not(Piece pc, Direct drop_sq) { return piece_effect_not_table[pc][drop_sq]; }

  // 駒pcを敵玉の8近傍において王手になる場所が1。
  extern Directions piece_check_around8_table[PIECE_NB];
  inline Directions piece_check_around8(Piece pc) { return piece_check_around8_table[pc]; }

  // 玉の8近傍、玉から見てd1の方角の升にd2方向の長い利きが発生したとして、その利きを玉から見て。
  extern Directions cutoff_directions_table[DIRECT_NB_PLUS4][DIRECTIONS_NB];
  inline Directions cutoff_directions(Direct d1, Directions d2) { return cutoff_directions_table[d1][d2]; }

  // --- init

  // このnamespaceで用いるテーブルの初期化
  void init();
}

// ----------------------
// 24近傍関係のライブラリ
// ----------------------

namespace Effect24
{
  // --- Directions

  // 方角を表す。24近傍を表現するのに用いる
  // bit0..右右上上、bit1..右右上、bit2..右右、…
  // 同時に複数のbitが1であることがありうる。
  enum Directions : uint32_t { DIRECTIONS_NB = 1 << 24 };
  
  inline Directions operator |(const Directions d1, const Directions d2) { return Directions(int(d1) + int(d2)); }

  // Bitboardのsq周辺の24近傍の状態を24bitに直列化する。
  // ただし盤外に相当するbitの値は不定。盤外を0にしたいのであれば、Effect8::board_mask(sq)と & すること。
  Directions around24(const Bitboard& b, Square sq);

  // around24()などである升の24近傍の情報を回収したときに壁の位置のマスクが欲しいときがあるから、そのマスク情報。
  // 壁のところが0、盤面上が1になっている。
  extern Directions board_mask_table[SQ_NB];
  inline Directions board_mask(Square sq) { return board_mask_table[sq]; }

  // .....
  // .....
  // ..+..  5×5のうち、中央の情報はDirectionsは持っていないので'+'を出力して、
  // .....  5近傍は、1であれば'*'、さもなくば'.'を出力する。
  // .....
  //
  std::ostream& operator<<(std::ostream& os, Directions d);

  // --- Direct

  // Directionsをpopしたもの。複数の方角を同時に表すことはない。
  enum Direct { DIRECT_ZERO = 0, DIRECT_NB = 24 };

  inline bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB; }

  // Directionsに相当するものを引数に渡して1つ方角を取り出す。
  inline Direct pop_directions(uint32_t& d) { return (Direct)pop_lsb(d); }

  // DirectからDirectionsへの逆変換
  inline Directions to_directions(Direct d) { return Directions(1 << d); }

  // DirectをSquare型の差分値で表現したもの。
  const SquareWithWall DirectToDeltaWW_[DIRECT_NB] = {
    SQWW_RU + SQWW_RU   , SQWW_R  + SQWW_RU   , SQWW_R + SQWW_R , SQWW_R + SQWW_RD  , SQWW_RD   + SQWW_RD    ,
    SQWW_RU + SQWW_U    , SQWW_RU             , SQWW_R          , SQWW_RD           , SQWW_RD   + SQWW_D     ,
    SQWW_U  + SQWW_U    , SQWW_U                                , SQWW_D            , SQWW_D    + SQWW_D     ,
    SQWW_LU + SQWW_U    , SQWW_LU             , SQWW_L          , SQWW_LD           , SQWW_LD   + SQWW_D     ,
    SQWW_LU + SQWW_LU   , SQWW_L  + SQWW_LU   , SQWW_L + SQWW_L , SQWW_L  + SQWW_LD , SQWW_LD   + SQWW_LD    ,
  };
  inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }

  // DirectをSquare型の差分値で表現したもの。
  const Square DirectToDelta_[DIRECT_NB] = {
    SQ_RU + SQ_RU   , SQ_R + SQ_RU   , SQ_R + SQ_R , SQ_R  + SQ_RD  , SQ_RD + SQ_RD  ,
    SQ_RU + SQ_U    , SQ_RU          , SQ_R        , SQ_RD          , SQ_RD + SQ_D   ,
    SQ_U  + SQ_U    , SQ_U                         , SQ_D           , SQ_D  + SQ_D   ,
    SQ_LU + SQ_U    , SQ_LU          , SQ_L        , SQ_LD          , SQ_LD + SQ_D   ,
    SQ_LU + SQ_LU   , SQ_L + SQ_LU   , SQ_L + SQ_L , SQ_L  + SQ_LD  , SQ_LD + SQ_LD  ,
  };
  inline Square DirectToDelta(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDelta_[d]; }

  // 24近傍で8近傍に利く長い利きの方向。24近傍 = 4筋*9段+5升=41升 ≦ 48升*WordBoard = 96byte = ymm(32) * 3
  extern ymm ymm_direct_to_around8[3];

  // --- init

  // このnamespaceで用いるテーブルの初期化
  void init();
} // namespace Effect24

// ----------------------
//  長い利きに関するライブラリ
// ----------------------

namespace LongEffect
{
  using namespace Effect8;

  // ----------------------
  //  ByteBoard(利きの数を表現)
  // ----------------------

  // ある升における利きの数を表現するByteBoard
  // 玉の8近傍を回収するなど、アライメントの合っていないアクセスをするのでこの構造体にはalignasをつけないことにする。
  struct ByteBoard
  {
    // ある升の利きの数
    uint8_t effect(Square sq) const { return e[sq]; }

    // ある升の周辺8近傍の利きを取得。1以上の値のところが1になる。さもなくば0。ただし壁(盤外)は不定。必要ならEffect8::board_maskでmaskすること。
    Directions around8(Square sq) const {
      // This algorithm is developed by tanuki- and yaneurao in 2016.

      // メモリアクセス違反ではあるが、Positionクラスのなかで使うので隣のメモリが
      // ±10bytesぐらい確保されているだろうから問題ない。
      // →　念のためpaddingしておくことにする。

      // sqの升の右上の升から32升分は取得できたので、これをPEXTで回収する。
      return (Directions)PEXT32(ymm(&e[sq - SQ_22]).cmp(ymm_zero).to_uint32(), 0b111000000101000000111);
    }

    // ある升の周辺8近傍の利きを取得。2以上の値のところが1になる。さもなくば0。ただし壁(盤外)は不定。必要ならEffect8::board_maskでmaskすること。
    Directions around8_greater_than_one(Square sq) const
    {
      return (Directions)PEXT32(ymm(&e[sq - SQ_22]).cmp(ymm_one).to_uint32(), 0b111000000101000000111);
    }

    // ゼロクリア
	void clear();

    // around8で回収するときのpadding
    uint8_t padding[SQ_22];

    // 各升の利きの数
    uint8_t e[SQ_NB_PLUS1];

    // around8で回収するときのpadding
    uint8_t padding2[32-SQ_22-1];
  };

  // 各升の利きの数を出力する。
  std::ostream& operator<<(std::ostream& os, const ByteBoard& board);

  // ----------------------
  //  WordBoard(利きの方向を先後同時に表現)
  // ----------------------

  // Directions先後用
  union LongEffect16 {
    Directions dirs[COLOR_NB]; // 先後個別に扱いたいとき用
    uint16_t u16;              // 直接整数として扱いたいとき用。long_effect_of()で取得可能
  };

  // 先手の香と角と飛車の長い利きの方向
  const Directions BISHOP_DIR = DIRECTIONS_LU | DIRECTIONS_LD | DIRECTIONS_RU | DIRECTIONS_RD;
  const Directions ROOK_DIR = DIRECTIONS_R | DIRECTIONS_U | DIRECTIONS_D | DIRECTIONS_L;

  // ある駒に対する長い利きの方向
  // 1) 長い利きに関して、馬と龍は角と飛車と同じ。
  // 2) 後手の駒は (dir << 8)して格納する。(DirectionsBWの定義より。)
  const uint16_t long_effect16_table[PIECE_NB] = {
    0,0,DIRECTIONS_U/*香*/,0,0,BISHOP_DIR/*角*/,ROOK_DIR/*飛*/,0,0,0,0,0,0,BISHOP_DIR/*馬*/,ROOK_DIR/*龍*/,0,                                          // 先手
    0,0,uint16_t(DIRECTIONS_D<<8)/*香*/,0,0,uint16_t(BISHOP_DIR<<8),uint16_t(ROOK_DIR<<8),0,0,0,0,0,0,uint16_t(BISHOP_DIR<<8),uint16_t(ROOK_DIR<<8),0, // 後手
  };
  inline uint16_t long_effect16_of(Piece pc) { return long_effect16_table[pc]; }

  // ↑の先後、どちらの駒に対してもDirections(8-bit)が返る仕様の関数。
  const uint8_t long_effect8_table[PIECE_NB] = {
    0,0,DIRECTIONS_U/*香*/,0,0,BISHOP_DIR/*角*/,ROOK_DIR/*飛*/,0,0,0,0,0,0,BISHOP_DIR/*馬*/,ROOK_DIR/*龍*/,0,                                          // 先手
    0,0,DIRECTIONS_D/*香*/,0,0,BISHOP_DIR,ROOK_DIR,0,0,0,0,0,0,BISHOP_DIR,ROOK_DIR,0, // 後手
  };
  inline Directions long_effect_of(Piece pc) { return (Directions)long_effect8_table[pc]; }

  // ある升における利きの数を表現するWordBoard
  // 玉の8近傍を回収するなど、アライメントの合っていないアクセスをするのでこの構造体にはalignasをつけないことにする。
  struct WordBoard
  {
    // ゼロクリア
	void clear();

    // ある升にある長い利きの方向
    // この方向に利いている(遠方駒は、この逆方向にいる。sqの駒を取り除いたときにさらにこの方角に利きが伸びる)
    Directions directions_of(Color us , Square sq) const { return le16[sq].dirs[us]; }

    // ある升の長い利きの方向を得る(DirectionsBW型とみなして良い)
    uint16_t long_effect16(Square sq) const { return le16[sq].u16; }

    // sqの升の周辺24近傍に対して、sqの9近傍(sqの地点を含む)への長い利きを持っている升を列挙する
    template <Color Us> Effect24::Directions long_effect24_to_around9(Square sq) const
    {
      // This algorithm is developed by tanuki- and yaneurao in 2016.

      // SQ_33の地点で正規化して(並行移動させて)、24近傍(24bit)分を取り出す。

      const int offset = Us == BLACK ? 0 : 1;
      u32 bits0 = PEXT32((ymm((u8*)&le16[sq - SQ_33 +  0] + offset) & Effect24::ymm_direct_to_around8[0]).eq(ymm_zero).to_uint32(), 0b00000101010101000000000101010101); // 16升中10升
      u32 bits1 = PEXT32((ymm((u8*)&le16[sq - SQ_33 + 16] + offset) & Effect24::ymm_direct_to_around8[1]).eq(ymm_zero).to_uint32(), 0b01010101010000000001010001010000); // 16升中 9升
      u32 bits2 = PEXT32((ymm((u8*)&le16[sq - SQ_33 + 32] + offset) & Effect24::ymm_direct_to_around8[2]).eq(ymm_zero).to_uint32(), 0b00000000000000010101010100000000); // 16升中 5升
      return Effect24::Directions(~(bits0 + (bits1 << 10) + (bits2 << 19)));
    }

    // long_effect24_to_around9で回収するときのpadding
    LongEffect16 padding[SQ_33];

    // 各升のDirections(先後)
    // 先手のほうは下位8bit。後手のほうは上位8bit
    LongEffect16 le16[SQ_NB_PLUS1];

    // long_effect24_to_around9で回収するときのpadding
    LongEffect16 padding2[48 - SQ_33 - 1];

  };

  // 各升の利きの方向を出力する。
  // 先手の右上方向の長い利き = 0 , 右方向 = 1 , 右下方向 = 2,…,左下方向 = 7,
  // 後手の右上方向の長い利き = a , 右方向 = b , 右下方向 = c,…,左下方向 = h
  // で各升最大4つまで。(これ以上表示すると見づらくなるため)
  std::ostream& operator<<(std::ostream& os, const WordBoard& board);

  // ----------------------
  //  Positionクラスの初期化時の利きの全計算
  // ----------------------

  // 利きの全計算(Positionクラスの初期化時に呼び出される)
  void calc_effect(Position& pos);

  // ----------------------
  //  do_move()での利きの更新用
  // ----------------------

  // Usの手番で駒pcをtoに配置したときの盤面の利きの更新
  template <Color Us> void update_by_dropping_piece(Position& pos, Square to, Piece moved_pc);

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっており、捕獲された駒captured_pcがあるときの盤面の利きの更新
  template <Color Us> void update_by_capturing_piece(Position& pos, Square from , Square to, Piece moved_pc, Piece moved_after_pc,Piece captured_pc);

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっている(捕獲された駒はない)ときの盤面の利きの更新
  template <Color Us> void update_by_no_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);

  // ----------------------
  //  undo_move()での利きの更新用
  // ----------------------

  // 上の3つの関数の逆変換を行なう関数

  template <Color Us> void rewind_by_dropping_piece(Position& pos, Square to, Piece moved_pc);
  template <Color Us> void rewind_by_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc);
  template <Color Us> void rewind_by_no_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);

  // --- initialize for LONG_EFFECT_LIBRARY
  void init();
}

#endif // LONG_EFFECT_LIBRARY

#endif // _LONG_EFFECT_H_

