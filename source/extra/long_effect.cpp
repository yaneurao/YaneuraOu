#include "long_effect.h"

// これはshogi.hで定義しているのでLONG_EFFECT_LIBRARYがdefineされていないときにも必要。
namespace Effect8 { Directions direc_table[SQ_NB_PLUS1][SQ_NB_PLUS1]; }


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

  void init()
  {
    // -- board_mask_tableの初期化
    for (auto sq : SQ)
      board_mask_table[sq] = to_directions(kingEffect(sq), sq);

    //Directions d = Directions(1 + 2 + 8);
    //for (auto dir : d)
    //  std::cout << dir;
  }

  Directions to_directions(const Bitboard& b, Square sq)
  {
    // This algorithm is developed by tanuki- and yaneurao in 2016.

    // sqがSQ_32(p[1]から見るとSQ_92の左の升)に来るように正規化する。(SQ_22だと後半が64回以上のシフトが必要になる)
    auto t = uint32_t((sq < SQ_32) ? (b.p[0] << int(SQ_32 - sq)) :
      ((b.p[0] >> int(sq - SQ_32)) | (b.p[1] << int(SQ_92 + SQ_L - sq)))); // p[1]のSQ_92の左の升は、p[0]のSQ_32相当。

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
      ((b.p[0] >> int(sq - SQ_33)) | (b.p[1] << int(SQ_93 + SQ_L - sq))); // p[1]のSQ_93の左は、p[0]のSQ_33

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
    UPDATE_EFFECT_BY_PUTTING_PIECE(pc, sq); // pcをsqに置いて利きを更新する。
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
        auto e = Directions(board.e[f | r]);
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

  // LONG_EFFECT_LIBRARYの初期化
  void init()
  {
    Effect8::init();
    Effect24::init();
  }

}

#endif // LONG_EFFECT_LIBRARY
