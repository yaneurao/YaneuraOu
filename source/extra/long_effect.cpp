#include "long_effect.h"
#include "../bitboard.h"

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

  Directions board_mask_table[SQ_NB];

  void init()
  {
    // -- board_mask_tableの初期化
    for (auto sq : SQ)
      board_mask_table[sq] = to_directions(kingEffect(sq), sq);
  }
}

namespace Effect24
{
  Directions to_directions(const Bitboard& b, Square sq)
  {
    // sqがSQ_33に来るように正規化する。
    auto t = (sq < SQ_33) ? (b.p[0] << int(SQ_33 - sq)) :
      ((b.p[0] >> int(sq - SQ_33)) | (b.p[1] << int(SQ_93 + SQ_LEFT - sq))); // p[1]のSQ_93の左は、p[0]のSQ_33

    // PEXTで24近傍の状態を回収。
    return (Directions)PEXT64(t, 0b11111000011111000011011000011111000011111);
  }

  std::ostream& operator<<(std::ostream& os, Directions d) { return output_around_n(os, d, 5); }

  Directions board_mask_table[SQ_NB];

  void init()
  {
    // -- board_mask_tableの初期化
    for (auto sq : SQ)
      board_mask_table[sq] = to_directions(around24_bb(sq), sq);
  }
}
