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
  enum Directions : uint8_t {};
  const uint32_t DIRECTIONS_NB = 256; // Directionsの範囲(uint8_t)で表現できない値なので外部に出しておく。

  inline Directions operator |(const Directions d1, const Directions d2) { return Directions(int(d1) + int(d2)); }

  // Bitboardのsq周辺の8近傍の状態を8bitに直列化する。
  // ただし盤外に相当するbitの値は不定。盤外を0にしたいのであれば、Effect8::board_mask(sq)と & すること。
  static Directions to_directions(const Bitboard& b, Square sq);

  // around8()などである升の8近傍の情報を回収したときに壁の位置のマスクが欲しいときがあるから、そのマスク情報。
  // 壁のところが0、盤面上が1になっている。
  extern Directions board_mask_table[SQ_NB];
  inline Directions board_mask(Square sq) { return board_mask_table[sq]; }

  // sq1にとってsq2がどのdirectionにあるか。
  extern Directions direc_table[SQ_NB][SQ_NB];
  inline Directions directions_of(Square sq1, Square sq2) { return direc_table[sq1][sq2]; }

  // ...
  // .+.  3×3のうち、中央の情報はDirectionsは持っていないので'+'を出力して、
  // ...  8近傍は、1であれば'*'、さもなくば'.'を出力する。
  //
  std::ostream& operator<<(std::ostream& os, Directions d);

  // --- Direct

  // Directionsをpopしたもの。複数の方角を同時に表すことはない。
  enum Direct{ DIRECT_RU, DIRECT_R, DIRECT_RD, DIRECT_U, DIRECT_D, DIRECT_LU, DIRECT_L, DIRECT_LD, DIRECT_NB, DIRECT_ZERO = 0, };
  ENABLE_OPERATORS_ON(Direct);

  inline bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB; }

  // Directionsに相当するものを引数に渡して1つ方角を取り出す。
  inline Direct pop_directions(uint32_t& d) { return (Direct)pop_lsb(d); }

  // DirectからDirectionsへの逆変換
  inline Directions to_directions(Direct d) { return Directions(1 << d); }

  // DirectをSquare型の差分値で表現したもの。
  const Square DirectToDelta_[DIRECT_NB] = { SQ_RU,SQ_RIGHT,SQ_RD,SQ_UP,SQ_DOWN,SQ_LU,SQ_LEFT,SQ_LD, };
  inline Square DirectToDelta(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDelta_[d]; }

  // DirectをSquareWithWall型の差分値で表現したもの。
  const SquareWithWall DirectToDeltaWW_[DIRECT_NB] = { SQWW_RU,SQWW_RIGHT,SQWW_RD,SQWW_UP,SQWW_DOWN,SQWW_LU,SQWW_LEFT,SQWW_LD, };
  inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }

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
    SQWW_RU + SQWW_RU    , SQWW_RIGHT + SQWW_RU   , SQWW_RIGHT + SQWW_RIGHT , SQWW_RIGHT + SQWW_RD , SQWW_RD   + SQWW_RD    ,
    SQWW_RU + SQWW_UP    , SQWW_RU                , SQWW_RIGHT              , SQWW_RD              , SQWW_RD   + SQWW_DOWN  ,
    SQWW_UP + SQWW_UP    , SQWW_UP                                          , SQWW_DOWN            , SQWW_DOWN + SQWW_DOWN  ,
    SQWW_LU + SQWW_UP    , SQWW_LU                , SQWW_LEFT               , SQWW_LD              , SQWW_LD   + SQWW_DOWN  ,
    SQWW_LU + SQWW_LU    , SQWW_LEFT  + SQWW_LU   , SQWW_LEFT  + SQWW_LEFT  , SQWW_LEFT  + SQWW_LD , SQWW_LD   + SQWW_LD    ,
  };
  inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }

  // DirectをSquare型の差分値で表現したもの。
  const Square DirectToDelta_[DIRECT_NB] = {
    SQ_RU + SQ_RU    , SQ_RIGHT + SQ_RU   , SQ_RIGHT + SQ_RIGHT , SQ_RIGHT + SQ_RD , SQ_RD + SQ_RD    ,
    SQ_RU + SQ_UP    , SQ_RU              , SQ_RIGHT            , SQ_RD            , SQ_RD + SQ_DOWN  ,
    SQ_UP + SQ_UP    , SQ_UP                                    , SQ_DOWN          , SQ_DOWN + SQ_DOWN,
    SQ_LU + SQ_UP    , SQ_LU              , SQ_LEFT             , SQ_LD            , SQ_LD + SQ_DOWN  ,
    SQ_LU + SQ_LU    , SQ_LEFT + SQ_LU    , SQ_LEFT + SQ_LEFT   , SQ_LEFT + SQ_LD  , SQ_LD + SQ_LD    ,
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

  // 利きの数や遠方駒の利きを表現するByteBoard
  // 玉の8近傍を回収するなど、アライメントの合っていないアクセスをするのでこの構造体にはalignasをつけないことにする。
  struct ByteBoard
  {
    // ゼロクリア
    void clear() { memset(e, 0, sizeof(e)); }

    // 各升の利きの数 or Directions
    uint8_t e[SQ_NB];
  };

  // -- EffectNumBoard

  // ある升における利きの数を表現するByteBoard
  struct EffectNumBoard : public ByteBoard
  {
    // この構造体が利きの数が格納されている構造体だとしてある升の利きの数
    uint8_t count(Square sq) const { return e[sq]; }

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
  };

  std::ostream& operator<<(std::ostream& os, const EffectNumBoard& board);

  // --- LongEffectBoard

  // ある升における長い利きを表現するByteBoard
  struct LongEffectBoard : public ByteBoard
  {
    // ある升にある長い利きの方向
    // この方向に利いている(遠方駒は、この逆方向にいる。sqの駒を取り除いたときにさらにこの方角に利きが伸びる)
    Directions directions(Square sq) const { return (Directions)e[sq]; }
  };

  std::ostream& operator<<(std::ostream& os, const LongEffectBoard board);
}

#endif LONG_EFFECT_LIBRARY

#endif // _LONG_EFFECT_H_

