#ifndef _LONG_EFFECT_H_
#define _LONG_EFFECT_H_

// 遠方駒による利きのライブラリ
// Bitboard/Byteboardに対して8近傍,24近傍を高速に求める等

#include "../shogi.h"

#ifdef LONG_EFFECT_LIBRARY

struct Bitboard;

// ----------------------
// 8近傍関係のライブラリ
// ----------------------
namespace Effect8
{
  // --- Directions

  // 方角を表す。遠方駒の利きや、玉から見た方角を表すのに用いる。
  // bit0..右上、bit1..右、bit2..右下、bit3..上、bit4..下、bit5..左上、bit6..左、bit7..左下
  // 同時に複数のbitが1であることがありうる。
  //  enum Directions : uint8_t { DIRECTIONS_ZERO = 0 };
  // →この定義はshogi.hに移動させた。

  const uint32_t DIRECTIONS_NB = 256; // Directionsの範囲(uint8_t)で表現できない値なので外部に出しておく。

  // Bitboardのsq周辺の8近傍の状態を8bitに直列化する。
  // ただし盤外に相当するbitの値は不定。盤外を0にしたいのであれば、Effect8::board_mask(sq)と & すること。
  static Directions to_directions(const Bitboard& b, Square sq);

  // around8()などである升の8近傍の情報を回収したときに壁の位置のマスクが欲しいときがあるから、そのマスク情報。
  // 壁のところが0、盤面上が1になっている。
  extern Directions board_mask_table[SQ_NB];
  inline Directions board_mask(Square sq) { return board_mask_table[sq]; }


  // sq1にとってsq2がどのdirectionにあるか。
  //extern Directions direc_table[SQ_NB][SQ_NB];
  //inline Directions directions_of(Square sq1, Square sq2) { return direc_table[sq1][sq2]; }

  // →　この定義もshogi.hに移動させた。

  // ...
  // .+.  3×3のうち、中央の情報はDirectionsは持っていないので'+'を出力して、
  // ...  8近傍は、1であれば'*'、さもなくば'.'を出力する。
  //
  std::ostream& operator<<(std::ostream& os, Directions d);

  // --- Direct

  // Directionsをpopしたもの。複数の方角を同時に表すことはない。
  // enum Direct { DIRECT_RU, DIRECT_R, DIRECT_RD, DIRECT_U, DIRECT_D, DIRECT_LU, DIRECT_L, DIRECT_LD, DIRECT_NB, DIRECT_ZERO = 0, };
  // inline bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB; }

  // →　以上の定義もshogi.hに移動させた。

  // Directionsに相当するものを引数に渡して1つ方角を取り出す。
  // inline Direct pop_directions(Directions& d) { return (Direct)pop_lsb(*(uint8_t*)&d); }

  // DirectをSquare型の差分値で表現したもの。
  const Square DirectToDelta_[DIRECT_NB] = { SQ_RU,SQ_R,SQ_RD,SQ_U,SQ_D,SQ_LU,SQ_L,SQ_LD, };
  inline Square DirectToDelta(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDelta_[d]; }

  // DirectからDirectionsへの逆変換
  // inline Directions to_directions(Direct d) { return Directions(1 << d); }

  // DirectをSquareWithWall型の差分値で表現したもの。
  //const SquareWithWall DirectToDeltaWW_[DIRECT_NB] = { SQWW_RU,SQWW_R,SQWW_RD,SQWW_U,SQWW_D,SQWW_LU,SQWW_L,SQWW_LD, };
  //inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }

  // → shogi.hに持って行った。

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

  // Bitboardのsq周辺の8近傍の状態を8bitに直列化する。
  // ただし盤外に相当するbitの値は不定。盤外を0にしたいのであれば、Effect8::board_mask(sq)と & すること。
  static Directions to_directions(const Bitboard& b, Square sq);

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

  // -- ByteBoard

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
      // sqの升の右上の升から32升分は取得できたので、これをPEXTで回収する。
      return (Directions)PEXT32(ymm(&e[sq - 10]).cmp(ymm_zero).to_uint32(), 0b111000000101000000111);
    }

    // ある升の周辺8近傍の利きを取得。2以上の値のところが1になる。さもなくば0。ただし壁(盤外)は不定。必要ならEffect8::board_maskでmaskすること。
    Directions around8_greater_than_one(Square sq) const
    {
      return (Directions)PEXT32(ymm(&e[sq - 10]).cmp(ymm_one).to_uint32(), 0b111000000101000000111);
    }

    // ゼロクリア
    void clear() { memset(e, 0, sizeof(e)); }

    // 各升の利きの数
    uint8_t e[SQ_NB_PLUS1];
  };

  // 各升の利きの数を出力する。
  std::ostream& operator<<(std::ostream& os, const ByteBoard& board);

  // --- LongEffectBoard

  // Directions先後用
  union DirectionsBW {
    Directions dirs[COLOR_NB]; // 先後個別に扱いたいとき用
    uint16_t u16;              // 直接整数として扱いたいとき用
  };

  // 先手の香と角と飛車の長い利きの方向
  const Directions BISHOP_DIR = DIRECTIONS_LU | DIRECTIONS_LD | DIRECTIONS_RU | DIRECTIONS_RD;
  const Directions ROOK_DIR = DIRECTIONS_R | DIRECTIONS_U | DIRECTIONS_D | DIRECTIONS_L;

  // ある駒に対する長い利きの方向
  // 1) 長い利きに関して、馬と龍は角と飛車と同じ。
  // 2) 後手の駒は (dir << 8)して格納する。(DirectionsBWの定義より。)
  const uint16_t dir_bw_table[PIECE_NB] = {
    0,0,DIRECTIONS_U/*香*/,0,0,BISHOP_DIR/*角*/,ROOK_DIR/*飛*/,0,0,0,0,0,0,BISHOP_DIR/*馬*/,ROOK_DIR/*龍*/,0,                                          // 先手
    0,0,uint16_t(DIRECTIONS_D<<8)/*香*/,0,0,uint16_t(BISHOP_DIR<<8),uint16_t(ROOK_DIR<<8),0,0,0,0,0,0,uint16_t(BISHOP_DIR<<8),uint16_t(ROOK_DIR<<8),0, // 後手
  };
  inline uint16_t dir_bw_of(Piece pc) { return dir_bw_table[pc]; }

  // ある升における利きの数を表現するWordBoard
  // 玉の8近傍を回収するなど、アライメントの合っていないアクセスをするのでこの構造体にはalignasをつけないことにする。
  struct WordBoard
  {
    // ゼロクリア
    void clear() { memset(dir_bw, 0, sizeof(dir_bw)); }

    // ある升にある長い利きの方向
    // この方向に利いている(遠方駒は、この逆方向にいる。sqの駒を取り除いたときにさらにこの方角に利きが伸びる)
    Directions directions_of(Color us , Square sq) const { return dir_bw[sq].dirs[us]; }

    // ある升の長い利きの方向を得る(DirectionsBW型とみなして良い)
    uint16_t dir_bw_on(Square sq) const { return dir_bw[sq].u16; }

    // 各升のDirections(先後)
    // 先手のほうは下位8bit。後手のほうは上位8bit
    DirectionsBW dir_bw[SQ_NB_PLUS1];
  };

  // 各升の利きの方向を出力する。
  // 先手の右上方向の長い利き = 0 , 右方向 = 1 , 右下方向 = 2,…,左下方向 = 7,
  // 後手の右上方向の長い利き = a , 右方向 = b , 右下方向 = c,…,左下方向 = h
  // で各升最大4つまで。(これ以上表示すると見づらくなるため)
  std::ostream& operator<<(std::ostream& os, const WordBoard& board);

  // --- do_move()のときに利きを更新するためのヘルパー関数

  // Usの手番で駒pcをtoに配置したときの盤面の利きの更新
  template <Color Us> void update_by_dropping_piece(Position& pos, Square to, Piece moved_pc);

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっており、捕獲された駒captured_pcがあるときの盤面の利きの更新
  template <Color Us> void update_by_capturing_piece(Position& pos, Square from , Square to, Piece moved_pc, Piece moved_after_pc,Piece captured_pc);

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっている(捕獲された駒はない)ときの盤面の利きの更新
  template <Color Us> void update_by_no_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);


  // --- initialize for LONG_EFFECT_LIBRARY
  void init();
}

#endif LONG_EFFECT_LIBRARY

#endif // _LONG_EFFECT_H_

