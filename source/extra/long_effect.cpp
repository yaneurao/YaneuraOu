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
  using namespace Effect8;

  // effect,long_effect配列はゼロ初期化されているものとする。

  // すべての駒に対して利きを列挙して、その先の升の利きを更新
  for (auto sq : pieces())
  {
    Piece pc = piece_on(sq);
    // pcをsqに置くことによる利きのupdate
    auto effect = effects_from(pc, sq, pieces());
    Color c = color_of(pc);
    for (auto to : effect)
      INC_BOARD_EFFECT(c, to);
    if (has_long_effect(pc))
      for (auto to : effect)
      {
        auto dir = directions_of(sq, to);
        long_effect.dir_bw[to].dirs[c] ^= dir;
      }
  }

  // デバッグ用に表示させて確認。
  std::cout << "BLACK board effect\n" << board_effect[BLACK] << "WHITE board effect\n" << board_effect[WHITE];
  std::cout << "long effect\n" << long_effect;

}

namespace LongEffect
{
  std::ostream& operator<<(std::ostream& os, const ByteBoard& board)
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

  std::ostream& operator<<(std::ostream& os, const WordBoard& board)
  {
    for (auto r : Rank())
    {
      for (File f = FILE_9; f >= FILE_1; --f)
      {
        auto e = board.dir_bw[f | r];
        // 方角を表示。複数あるなら4個まで表示
        os << '[';
        int i;
        for (i = 0; i < 4; ++i)
          if (e.u16)
            os << "01234567abcdefgh"[pop_lsb(e.u16)]; // 先手は01234567 , 後手はabcdefghで表示する。
          else
            os << ' ';
        os << ']';
      }
      os << std::endl;
    }
    return os;
  }

  // --- LONG_EFFECT_LIBRARYの初期化
  void init()
  {
    Effect8::init();
    Effect24::init();
  }

  // --- do_move()時の利きの更新処理

  using namespace Effect8;

  // Usの手番で駒pcをtoに配置したときの盤面の利きの更新
  template <Color Us> void update_by_dropping_piece(Position& pos, Square to, Piece pc)
  {
    auto& board_effect = pos.board_effect;

    // 駒打ちなので
    // 1) 打った駒による利きの数の加算処理
    auto inc_target = effects_from(pc, to, pos.pieces());
    while (inc_target)
    {
      auto sq = inc_target.pop();
      INC_BOARD_EFFECT(Us, sq);
    }
    // 2) この駒が遠方駒なら長い利きの加算処理 + この駒によって遮断された利きの減算処理

    // これらは実は一度に行なうことが出来る。

    // trick a) (右側から)左方向への長い利きがあり、toの地点でそれを遮ったとして、しかしtoの地点に持って行った駒が飛車であれば、
    // この左方向の長い利きは遮ったことにはならならず、左方向への長い利きの更新処理は不要である。
    // このようにtoの地点の升での現在の長い利きと、toの地点に持って行った駒から発生する長い利きとをxorすることで
    // この利きの相殺処理がうまく行える。

    // trick b) (右側から)左方向への後手の長い利きをtoの升で遮断し、toの升に持って行った駒が飛車で左方向の長い利きが発生した場合、
    // WordBoardを採用しているとこの２つの利きを同時に更新して行ける。更新のときにxorを用いれば、利きの消失と発生を同時に行なうことが出来る。

    // This tricks are developed by yaneurao in 2016.

    auto& long_effect = pos.long_effect;

    // trick a)
    auto dir_bw = pos.long_effect.dir_bw_on(to) ^ LongEffect::dir_bw_of(pc);
    auto toww = to_sqww(to);
    while (dir_bw)
    {
      // 更新していく方角
      int dir = LSB32(dir_bw) & 7; // Effect8::Direct型
      
      // trick b)

      // 更新していく値。
      // これは先後の分、同時に扱いたいので、先手の分と後手の分。
      uint16_t value = uint16_t((1 << dir) | (1 << (dir+8)));
      dir_bw &= ~value; // dir_bwのうち、↑の2つのbitをクリア

      auto delta = DirectToDeltaWW((Direct)dir);

      do {
        toww += delta;
        if (!is_ok(toww)) // 壁に当たったのでこのrayは更新終了
          break;
        long_effect.dir_bw[to_sq(toww)].u16 ^= value; // xorで先後同時にこの方向の利きを更新
      } while (pos.piece_on(to_sq(toww)) == NO_PIECE);
    }
  }

  // ↑の関数の明示的な実体化
  template void update_by_dropping_piece<BLACK>(Position& pos, Square to, Piece pc);
  template void update_by_dropping_piece<WHITE>(Position& pos, Square to, Piece pc);

}

#endif // LONG_EFFECT_LIBRARY
