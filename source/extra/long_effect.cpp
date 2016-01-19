#include "long_effect.h"

#ifdef LONG_EFFECT_LIBRARY

#include "../bitboard.h"
#include "../position.h"

// Effect8::operator<<()で用いるヘルパー関数
// 3*3ならN=3 , 5*5ならN=5..
std::ostream& output_around_n(std::ostream& os, uint32_t d,int n)
{
  int c = n / 2;
  for (int j = -c; j <= +c; ++j)
  {
    for (int i = -c; i <= +c; ++i)
    {
      int bit = (j + c) + (-i + c) * n - (((i < 0) || (i == 0 && j >= 0)) ? 1 : 0);
      os << ((i == 0 && j == 0) ? '+' : (((1 << bit) & d) ? '*' : '.'));
    }
    os << std::endl;
  }
  return os;
}

namespace Effect8
{
  Directions board_mask_table[SQ_NB];
  Directions direc_table[SQ_NB][SQ_NB];

  void init()
  {
    // -- board_mask_tableの初期化
    for (auto sq : SQ)
      board_mask_table[sq] = to_directions(kingEffect(sq), sq);

    // -- direct_tableの初期化

    for (auto sq1 : SQ)
      for (auto dir = DIRECT_ZERO; dir < DIRECT_NB; ++dir)
      {
        // dirの方角に壁にぶつかる(盤外)まで延長していく。このとき、sq1から見てsq2のDirectionsは (1 << dir)である。
        auto delta = DirectToDeltaWW(dir);
        for (auto sq2 = to_sqww(sq1) + delta ; is_ok(sq2); sq2 += delta)
          direc_table[sq1][to_sq(sq2)] = to_directions(dir);
      }
  }

  Directions to_directions(const Bitboard& b, Square sq)
  {
    // This algorithm is developed by tanuki- and yaneurao in 2016.

    // sqがSQ_32(p[1]から見るとSQ_92の左の升)に来るように正規化する。(SQ_22だと後半が64回以上のシフトが必要になる)
    auto t = uint32_t((sq < SQ_32) ? (b.p[0] << int(SQ_32 - sq)) :
      ((b.p[0] >> int(sq - SQ_32)) | (b.p[1] << int(SQ_92 + SQ_LEFT - sq)))); // p[1]のSQ_92の左の升は、p[0]のSQ_32相当。

                                                                              // PEXTで8近傍の状態を回収。
    return (Directions)PEXT32(t, 0b111000000101000000111000000000);
  }

  std::ostream& operator<<(std::ostream& os, Directions d) { return output_around_n(os, d, 3); }

}

namespace Effect24
{
  Directions board_mask_table[SQ_NB];

  void init()
  {
    // -- board_mask_tableの初期化
    for (auto sq : SQ)
      board_mask_table[sq] = to_directions(around24_bb(sq), sq);
  }

  Directions to_directions(const Bitboard& b, Square sq)
  {
    // sqがSQ_33に来るように正規化する。
    auto t = (sq < SQ_33) ? (b.p[0] << int(SQ_33 - sq)) :
      ((b.p[0] >> int(sq - SQ_33)) | (b.p[1] << int(SQ_93 + SQ_LEFT - sq))); // p[1]のSQ_93の左は、p[0]のSQ_33

    // PEXTで24近傍の状態を回収。
    return (Directions)PEXT64(t, 0b11111000011111000011011000011111000011111);
  }

  std::ostream& operator<<(std::ostream& os, Directions d) { return output_around_n(os, d, 5); }
}

// Positionクラスの長い利きの初期化処理
void Position::set_effect()
{
  // effect,long_effect配列はゼロ初期化されているものとする。

  // すべての駒に対して利きを列挙して、その先の升の利きを更新
  for (auto sq : pieces())
  {
    Piece pc = piece_on(sq);
    auto effect = effects_from(pc, sq, pieces());
    Color c = color_of(pc);
    // ある升の利きの総和を更新。
    for (auto to : effect)
      INC_BOARD_EFFECT(c, to);

    // ある升の長い利きの更新。
    Piece pt = type_of(pc);
    if (pt == LANCE || pt == BISHOP || pt == ROOK)
      for (auto to : effect)
      {
        auto dir = Effect8::directions_of(sq,to);
        INC_LONG_EFFECT(c, to, dir);
      }
  }

  // デバッグ用に表示させて確認。
  //std::cout << "BLACK board effect\n" << board_effect[BLACK] << "WHITE board effect\n" << board_effect[WHITE];
  //std::cout << "BLACK long effect\n" << long_effect[BLACK] << "WHITE long effect\n" << long_effect[WHITE];

}

namespace LongEffect
{
  std::ostream& operator<<(std::ostream& os, const EffectNumBoard& board)
  {
    // 利きの数をそのまま表示。10以上あるところの利きの表示がおかしくなるが、まあそれはレアケースなので良しとする。
    for (auto r : Rank())
    {
      for (File f = FILE_9; f >= FILE_1; --f)
        os << (int)board.e[f | r];
      os << std::endl;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const LongEffectBoard board)
  {
    for (auto r : Rank())
    {
      for (File f = FILE_9; f >= FILE_1; --f)
      {
        auto e = (uint32_t)board.e[f | r];
        // 方角を表示。複数あるなら3個まで表示
        os << '[';
        int i;
        for (i = 0; i < 3; ++i)
          if (e)
            os << (int)pop_directions(e);
          else
            os << ' ';
        os << ']';
      }
      os << std::endl;
    }
    return os;
  }

}

#endif // LONG_EFFECT_LIBRARY
