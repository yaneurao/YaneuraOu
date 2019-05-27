#include "long_effect.h"

// namespace Effect8 { Directions direc_table[SQ_NB_PLUS1][SQ_NB_PLUS1]; }
// → shogi.cppで定義している。LONG_EFFECT_LIBRARYが必要なときは、この定義も用意すること。

// ----------------------
//  長い利きに関するライブラリ
// ----------------------

#if defined(LONG_EFFECT_LIBRARY)

#include "../bitboard.h"
#include "../position.h"

#include <cstring>	// std::memset()

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

// ----------------------
//    8近傍ライブラリ
// ----------------------

namespace Effect8
{
  Directions board_mask_table[SQ_NB_PLUS1];
  Directions piece_effect_not_table[PIECE_NB][DIRECT_NB];
  Directions piece_check_around8_table[PIECE_NB];
  Directions cutoff_directions_table[DIRECT_NB_PLUS4][DIRECTIONS_NB];

  void init()
  {
    // -- board_mask_tableの初期化
    
    for (auto sq : SQ)
      board_mask_table[sq] = around8(kingEffect(sq), sq);

    // -- piece_effect_not_tableの初期化

    for (auto pc : Piece())
      for (auto d : Direct())
      {
        // SQ_22の地点でその周辺8近傍の状態を使ってテーブルを初期化する
        auto effect = effects_from(pc, SQ_22 + DirectToDelta(d), ZERO_BB);
        // 利きがある場所が0、ない場所が1なのでnotしておく。
        auto effect8_not = ~around8(effect, SQ_22);

        piece_effect_not_table[pc][d] = effect8_not;
      }

    // -- piece_check_around8_tableの初期化

    for (auto pc : Piece())
    {
      // SQ_22に相手の駒を置いたときの利きの場所が、そこに置いてSQ_22に王手になる場所
      auto effect = effects_from(Piece(pc ^ PIECE_WHITE), SQ_22,ZERO_BB);
      auto effect8 = around8(effect, SQ_22);
      piece_check_around8_table[pc] = effect8;
    }

    // -- cutoff_directions_tableの初期化

    for (int d1 = 0; d1 < DIRECT_NB_PLUS4; ++d1)
      for (uint16_t d2 = 0; d2 < 0x100; ++d2)
      {
        Bitboard bb = ZERO_BB;

        // SQ_55の地点でやってみる。

        auto d2t = Directions(d2);
        while (d2t)
        {
          auto dir = pop_directions(d2t);

          auto sq = to_sqww(SQ_55 + DirectToDelta(Direct(d1)));
          auto delta = DirectToDeltaWW(dir);
          
          for (int i = 0; i < 3; ++i)
          {
            sq += delta;
            if (!is_ok(sq)) break;
            bb ^= sqww_to_sq(sq);
          }
        }
        
        auto effect8 = around8(bb, SQ_55);

        cutoff_directions_table[d1][d2] = effect8;
      }

  }

  Directions around8(const Bitboard& b, Square sq)
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

// ----------------------
//    24近傍ライブラリ
// ----------------------

namespace Effect24
{
  Directions board_mask_table[SQ_NB];
  ymm ymm_direct_to_around8[3];

  void init()
  {
    // -- board_mask_tableの初期化

    for (auto sq : SQ)
      board_mask_table[sq] = around24(around24_bb(sq), sq);

    // -- ymm_direct_to_around8[]の初期化
    {
      u16 d81[81];
      memset(d81, 0, sizeof(d81));
      auto sq33_around9 = kingEffect(SQ_33) ^ SQ_33;

      for (auto sq : SQ)
      {
        Effect8::Directions d = Effect8::DIRECTIONS_ZERO;
        for (int i = 0; i < 8; ++i)
        {
          // SQ_33に玉がいるとして、この9近傍に移動できる長い利きの方向。
          auto dir = Effect8::Direct(i);
          auto delta = Effect8::DirectToDeltaWW(dir);

          auto sqww = to_sqww(sq) + delta;
          if (!is_ok(sqww))
            continue;
          if (sq33_around9 & sqww_to_sq(sqww))
            d |= Effect8::to_directions(dir);
        }
        d81[sq] = d;
      }
      for (int i = 0; i < 3; ++i)
        ymm_direct_to_around8[i] = ymm(&d81[i * 16]); // 32byteずつ
    }

  }

  Directions around24(const Bitboard& b, Square sq)
  {
    // sqがSQ_33に来るように正規化する。
    auto t = (sq < SQ_33) ? (b.p[0] << int(SQ_33 - sq)) :
      ((b.p[0] >> int(sq - SQ_33)) | (b.p[1] << int(SQ_93 + SQ_L - sq))); // p[1]のSQ_93の左は、p[0]のSQ_33

    // PEXTで24近傍の状態を回収。
    return (Directions)PEXT64(t, 0b11111000011111000011011000011111000011111);
  }

  std::ostream& operator<<(std::ostream& os, Directions d) { return output_around_n(os, d, 5); }
}

// ----------------------
//  長い利きに関するライブラリ
// ----------------------

namespace LongEffect
{
  // ----------------------
  //  ByteBoard(利きの数を表現)
  // ----------------------

  std::ostream& operator<<(std::ostream& os, const ByteBoard& board)
  {
    // 利きの数をそのまま表示。10以上あるところの利きの表示がおかしくなるので16進数表示にしておく。
    for (auto r : Rank())
    {
      for (File f = FILE_9; f >= FILE_1; --f)
      {
        int e = uint8_t(board.e[f | r]);
        if (e < 16)
          os << "0123456789ABCDEF"[e];
        else
          os << "[" << e << "]"; // プログラムのバグで-1(255)などになっているのだろうが、その値が表示されていて欲しい。
      }
      os << std::endl;
    }
    return os;
  }

  // ゼロクリア
  void ByteBoard::clear() { memset(e, 0, sizeof(e)); }

  // ----------------------
  //  WordBoard(利きの方向を先後同時に表現)
  // ----------------------

  std::ostream& operator<<(std::ostream& os, const WordBoard& board)
  {
    for (auto r : Rank())
    {
      for (File f = FILE_9; f >= FILE_1; --f)
      {
        auto e = board.le16[f | r];
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

  // ゼロクリア
  void WordBoard::clear() { memset(le16, 0, sizeof(le16)); }

  // ----------------------
  //  Positionクラスの初期化時の利きの全計算
  // ----------------------

  void calc_effect(Position& pos)
  {
    using namespace Effect8;

    auto& board_effect = pos.board_effect;
    auto& long_effect = pos.long_effect;

    // ゼロクリア
    for (auto c : COLOR) board_effect[c].clear();
    long_effect.clear();

    // すべての駒に対して利きを列挙して、その先の升の利きを更新
    for (auto sq : pos.pieces())
    {
      Piece pc = pos.piece_on(sq);
      // pcをsqに置くことによる利きのupdate
      auto effect = effects_from(pc, sq, pos.pieces());
      Color c = color_of(pc);
      for (auto to : effect)
        ADD_BOARD_EFFECT(c, to, 1);
      if (has_long_effect(pc))
      {
        // ただし、馬・龍に対しては、長い利きは、角・飛車と同じ方向だけなので成り属性を消して、利きを求め直す。
        if (type_of(pc) != LANCE)
          effect = effects_from(Piece(pc & ~PIECE_PROMOTE), sq, pos.pieces());

        for (auto to : effect)
        {
          auto dir = directions_of(sq, to);
          long_effect.le16[to].dirs[c] ^= dir;
        }
      }
    }

    // デバッグ用に表示させて確認。
    //std::cout << "BLACK board effect\n" << board_effect[BLACK] << "WHITE board effect\n" << board_effect[WHITE];
    //std::cout << "long effect\n" << long_effect;

  }

  // ----------------------
  //  do_move()での利きの更新用
  // ----------------------

  using namespace Effect8;
 
  // 駒pcをsqの地点においたときの短い利きを取得する(長い利きは含まれない)
  inline Bitboard short_effects_from(Piece pc,Square sq)
  {
    switch (pc)
    {
    case B_PAWN: return pawnEffect(BLACK, sq);
    case W_PAWN: return pawnEffect(WHITE, sq);

    case B_KNIGHT: return knightEffect(BLACK, sq);
    case W_KNIGHT: return knightEffect(WHITE, sq);

    case B_SILVER: return silverEffect(BLACK, sq);
    case W_SILVER: return silverEffect(WHITE, sq);

      // 金相当の駒
    case B_GOLD: case B_PRO_PAWN: case B_PRO_LANCE: case B_PRO_KNIGHT: case B_PRO_SILVER: return goldEffect(BLACK, sq);
    case W_GOLD: case W_PRO_PAWN: case W_PRO_LANCE: case W_PRO_KNIGHT: case W_PRO_SILVER: return goldEffect(WHITE, sq);
      
    // 馬の短い利きは上下左右
    case B_HORSE : case W_HORSE:
      return cross00StepEffect(sq);

    // 龍の短い利きは斜め長さ1
    case B_DRAGON : case W_DRAGON:
      return cross45StepEffect(sq);

    case B_KING : case W_KING:
      return kingEffect(sq);

      // 短いを持っていないもの
    case B_LANCE: case B_BISHOP: case B_ROOK:
    case W_LANCE: case W_BISHOP: case W_ROOK:
      return ZERO_BB;

    default:
      UNREACHABLE; return ZERO_BB;
    }
  }

  // ある升から8方向のrayに対する長い利きの更新処理。先後同時に更新が行えて、かつ、
  // 発生と消滅が一つのコードで出来る。

  // dir_bw_usの方角のrayを更新するときはUs側の利きが+pされる。
  // dir_bw_othersの方角のrayを更新するときはそのrayの手番側の利きが-pされる。
  // これは
  // 1) toの地点にUsの駒を移動させるなら、toの地点で発生する利き(dir_bw_us)以外は、遮断された利きであるから、このray上の利きは減るべき。
  // 2) toの地点からUsの駒を移動させるなら、toの地点から取り除かれる利き(dir_bw_us)以外は、遮断されていたものが回復する利きであるから、このray上の利きは増えるべき。
  // 1)の状態か2の状態かをpで選択する。1)ならp=+1 ,  2)なら p=-1。

#define UPDATE_LONG_EFFECT_FROM_(EFFECT_FUNC,to,dir_bw_us,dir_bw_others,p) {  \
    Square sq;                                                                                       \
    uint16_t dir_bw = dir_bw_us ^ dir_bw_others;  /* trick a) */                                     \
    auto toww = to_sqww(to);                                                                         \
    while (dir_bw)                                                                                   \
    {                                                                                                \
      /* 更新していく方角*/                                                                          \
      int dir = LSB32(dir_bw) & 7; /* Effect8::Direct型*/                                            \
      /* 更新していく値。これは先後の分、同時に扱いたい。*/                                          \
      uint16_t value = uint16_t((1 << dir) | (1 << (dir + 8)));                                      \
      /* valueに関する上記の2つのbitをdir_bwから取り出す */                                          \
      value &= dir_bw;                                                                               \
      dir_bw &= ~value; /* dir_bwのうち、上記の2つのbitをクリア*/                                    \
      auto delta = DirectToDeltaWW((Direct)dir);                                                     \
      /* valueにUs側のrayを含むか */                                                                 \
      bool the_same_color = (Us == BLACK && (value & 0xff)) || ((Us == WHITE) && (value & 0xff00));  \
      int8_t e1 = (dir_bw_us & value) ? (+(p)) : (the_same_color ? (-(p)) : 0);                      \
      bool not_the_same = (Us == BLACK && (value & 0xff00)) || ((Us == WHITE) && (value & 0xff));    \
      int8_t e2 = not_the_same ? (-(p)) : 0;                                                         \
      auto toww2 = toww;                                                                             \
      do {                                                                                           \
        toww2 += delta;                                                                              \
        if (!is_ok(toww2)) break; /* 壁に当たったのでこのrayは更新終了*/                             \
        sq = sqww_to_sq(toww2);                                                                      \
        /* trick b) xorで先後同時にこの方向の利きを更新*/                                            \
        long_effect.le16[sq].u16 ^= value;                                                           \
        EFFECT_FUNC(Us,sq,e1,e2);                                                                    \
      } while (pos.piece_on(sq) == NO_PIECE);                                                        \
    }}

// do_move()のときに使う用。
#define UPDATE_LONG_EFFECT_FROM(to,dir_bw_us,dir_bw_others,p) { UPDATE_LONG_EFFECT_FROM_(ADD_BOARD_EFFECT_BOTH,to,dir_bw_us,dir_bw_others,p); }

// undo_move()で巻き戻すときに使う用。(利きの更新関数が違う)
#define UPDATE_LONG_EFFECT_FROM_REWIND(to,dir_bw_us,dir_bw_others,p) { UPDATE_LONG_EFFECT_FROM_(ADD_BOARD_EFFECT_BOTH_REWIND,to,dir_bw_us,dir_bw_others,p); }


  // Usの手番で駒pcをtoに配置したときの盤面の利きの更新
  template <Color Us> void update_by_dropping_piece(Position& pos, Square to, Piece dropped_pc)
  {
    auto& board_effect = pos.board_effect;

    // 駒打ちなので
    // 1) 打った駒による利きの数の加算処理
    auto inc_target = short_effects_from(dropped_pc, to);
    while (inc_target)
    {
      auto sq = inc_target.pop();
      ADD_BOARD_EFFECT(Us, sq, +1);
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

    auto dir_bw_us = LongEffect::long_effect16_of(dropped_pc); // 自分の打った駒による利きは増えて
    auto dir_bw_others = pos.long_effect.long_effect16(to); // その駒によって遮断された利きは減る
    UPDATE_LONG_EFFECT_FROM(to , dir_bw_us, dir_bw_others, +1);
  }

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっており、捕獲された駒captured_pcがあるときの盤面の利きの更新
  template <Color Us> void update_by_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc)
  {
    auto& board_effect = pos.board_effect;
    auto& long_effect = pos.long_effect;

    // -- 移動させた駒と捕獲された駒による利きの更新

    // 利きを減らさなければならない場所 = fromの地点における動かした駒の利き
    auto dec_target = short_effects_from(moved_pc, from);

    // 利きを増やさなければならない場所 = toの地点における移動後の駒の利き
    auto inc_target = short_effects_from(moved_after_pc, to);

    // 利きのプラス・マイナスが相殺する部分を消しておく。
    auto and_target = inc_target & dec_target;
    inc_target ^= and_target;
    dec_target ^= and_target;

    while (inc_target) { auto sq = inc_target.pop(); ADD_BOARD_EFFECT( Us, sq , +1); }
    while (dec_target) { auto sq = dec_target.pop(); ADD_BOARD_EFFECT( Us, sq , -1); }

    // 捕獲された駒の利きの消失
    dec_target = short_effects_from(captured_pc, to);
    while (dec_target) { auto sq = dec_target.pop(); ADD_BOARD_EFFECT(~Us, sq , -1); }

    // -- fromの地点での長い利きの更新。
    // この駒が移動することにより、ここに利いていた長い利きが延長されるのと、この駒による長い利きに関する更新。

    // このタイミングでは(captureではない場合)toにまだ駒はない。
    // fromには駒はあるが、toに駒をおいてもfromより向こう側(toと直線上)の長い利きの状態は変わらない。
    // (toに移動した駒による長い利きが、移動前もfromから同じように発生していたと考えられるから。)
    // だから、移動させる方向と反対方向のrayは更新してはならない。(敵も味方も共通で)

    // fromからtoへの反対方向への利きをマスクする(ちょっとこれ求めるの嫌かも..)
    auto dir = directions_of(from, to);
    uint16_t dir_mask;
    if (dir != 0)
    {
      // 桂以外による移動
      auto dir_cont = uint16_t(1 << (7 - LSB32(dir)));
      dir_mask = (uint16_t)~(dir_cont | (dir_cont << 8));
    } else {
      // 桂による移動(non mask)
      dir_mask = 0xffff;
    }

    auto dir_bw_us = LongEffect::long_effect16_of(moved_pc) & dir_mask;  // 移動させた駒による長い利きは無くなって
    auto dir_bw_others = pos.long_effect.long_effect16(from) & dir_mask; // そこで遮断されていた利きの分だけ増える
    UPDATE_LONG_EFFECT_FROM(from, dir_bw_us, dir_bw_others, -1);

    // -- toの地点での長い利きの更新。
    // ここはもともと今回捕獲された駒があって利きが遮断されていたので、
    // ここに移動させた駒からの長い利きと、今回捕獲した駒からの長い利きに関する更新だけで十分

    dir_bw_us = LongEffect::long_effect16_of(moved_after_pc);
    dir_bw_others = LongEffect::long_effect16_of(captured_pc);
    UPDATE_LONG_EFFECT_FROM(to, dir_bw_us , dir_bw_others , +1);
  }

  // Usの手番で駒pcをtoに移動させ、成りがある場合、moved_after_pcになっている(捕獲された駒はない)ときの盤面の利きの更新
  template <Color Us>void update_by_no_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc)
  {
    auto& board_effect = pos.board_effect;
    auto& long_effect = pos.long_effect;

    // -- 移動させた駒と捕獲された駒による利きの更新

    auto dec_target = short_effects_from(moved_pc, from);
    auto inc_target = short_effects_from(moved_after_pc, to);

    auto and_target = inc_target & dec_target;
    inc_target ^= and_target;
    dec_target ^= and_target;

    while (inc_target) { auto sq = inc_target.pop(); ADD_BOARD_EFFECT(Us, sq , +1); }
    while (dec_target) { auto sq = dec_target.pop(); ADD_BOARD_EFFECT(Us, sq , -1); }

    // -- fromの地点での長い利きの更新。(capturesのときと同様)

    auto dir = directions_of(from, to);
    uint16_t dir_mask;
    if (dir != 0)
    {
      // 桂以外による移動
      auto dir_cont = uint16_t(1 << (7 - LSB32(dir)));
      dir_mask = (uint16_t)~(dir_cont | (dir_cont << 8));
    } else {
      // 桂による移動(non mask)
      dir_mask = 0xffff;
    }

    auto dir_bw_us = LongEffect::long_effect16_of(moved_pc) & dir_mask;
    auto dir_bw_others = pos.long_effect.long_effect16(from) & dir_mask;
    UPDATE_LONG_EFFECT_FROM(from, dir_bw_us, dir_bw_others, -1);

    // -- toの地点での長い利きの更新。
    // ここに移動させた駒からの長い利きと、これにより遮断された長い利きに関する更新

    dir_bw_us = LongEffect::long_effect16_of(moved_after_pc);
    dir_bw_others = pos.long_effect.long_effect16(to);
    
    UPDATE_LONG_EFFECT_FROM(to, dir_bw_us, dir_bw_others, +1);
  }

  // ----------------------
  //  undo_move()での利きの更新用
  // ----------------------

  // 上の3つの関数の逆変換を行なう関数。

  template <Color Us> void rewind_by_dropping_piece(Position& pos, Square to, Piece dropped_pc)
  {
    auto& board_effect = pos.board_effect;

    auto inc_target = short_effects_from(dropped_pc, to);
    while (inc_target)
    {
      auto sq = inc_target.pop();
      ADD_BOARD_EFFECT_REWIND(Us, sq, -1); // rewind時には-1
    }

    auto& long_effect = pos.long_effect;

    auto dir_bw_us = LongEffect::long_effect16_of(dropped_pc);
    auto dir_bw_others = pos.long_effect.long_effect16(to);
    UPDATE_LONG_EFFECT_FROM_REWIND(to, dir_bw_us, dir_bw_others, -1); // rewind時には-1
  }

  template <Color Us> void rewind_by_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc)
  {
    auto& board_effect = pos.board_effect;
    auto& long_effect = pos.long_effect;

    auto inc_target = short_effects_from(moved_pc, from);
    auto dec_target = short_effects_from(moved_after_pc, to);

    auto and_target = inc_target & dec_target;
    inc_target ^= and_target;
    dec_target ^= and_target;

    while (inc_target) { auto sq = inc_target.pop(); ADD_BOARD_EFFECT_REWIND(Us, sq, +1); }
    while (dec_target) { auto sq = dec_target.pop(); ADD_BOARD_EFFECT_REWIND(Us, sq, -1); }

    // 捕獲された駒の利きの復活
    inc_target = short_effects_from(captured_pc, to);
    while (inc_target) { auto sq = inc_target.pop(); ADD_BOARD_EFFECT_REWIND(~Us, sq, +1); }

    // -- toの地点での長い利きの更新。

    auto dir_bw_us = LongEffect::long_effect16_of(moved_after_pc);
    auto dir_bw_others = LongEffect::long_effect16_of(captured_pc);
    UPDATE_LONG_EFFECT_FROM_REWIND(to, dir_bw_us, dir_bw_others, -1); // rewind時はこの符号が-1
                                                                    
    // -- fromの地点での長い利きの更新。

    auto dir = directions_of(from, to);
    uint16_t dir_mask;
    if (dir != 0)
    {
      // 桂以外による移動
      auto dir_cont = uint16_t(1 << (7 - LSB32(dir)));
      dir_mask = (uint16_t)~(dir_cont | (dir_cont << 8));
    } else {
      // 桂による移動(non mask)
      dir_mask = 0xffff;
    }

    dir_bw_us = LongEffect::long_effect16_of(moved_pc) & dir_mask;
    dir_bw_others = pos.long_effect.long_effect16(from) & dir_mask;
    UPDATE_LONG_EFFECT_FROM_REWIND(from, dir_bw_us, dir_bw_others, +1); // rewind時はこの符号が+1
  }

  template <Color Us> void rewind_by_no_capturing_piece(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc)
  {
    auto& board_effect = pos.board_effect;
    auto& long_effect = pos.long_effect;

    auto inc_target = short_effects_from(moved_pc, from);
    auto dec_target = short_effects_from(moved_after_pc, to);

    auto and_target = inc_target & dec_target;
    inc_target ^= and_target;
    dec_target ^= and_target;

    while (inc_target) { auto sq = inc_target.pop(); ADD_BOARD_EFFECT_REWIND(Us, sq, +1); }
    while (dec_target) { auto sq = dec_target.pop(); ADD_BOARD_EFFECT_REWIND(Us, sq, -1); }

    // -- toの地点での長い利きの更新。

    auto dir_bw_us = LongEffect::long_effect16_of(moved_after_pc);
    auto dir_bw_others = pos.long_effect.long_effect16(to);

    UPDATE_LONG_EFFECT_FROM_REWIND(to, dir_bw_us, dir_bw_others, -1); // rewind時はこの符号が-1
                                                                    
    // -- fromの地点での長い利きの更新。(capturesのときと同様)

    auto dir = directions_of(from, to);
    uint16_t dir_mask;
    if (dir != 0)
    {
      // 桂以外による移動
      auto dir_cont = uint16_t(1 << (7 - LSB32(dir)));
      dir_mask = (uint16_t)~(dir_cont | (dir_cont << 8));
    } else {
      // 桂による移動(non mask)
      dir_mask = 0xffff;
    }

    dir_bw_us = LongEffect::long_effect16_of(moved_pc) & dir_mask;
    dir_bw_others = pos.long_effect.long_effect16(from) & dir_mask;
    UPDATE_LONG_EFFECT_FROM_REWIND(from, dir_bw_us, dir_bw_others, +1); // rewind時はこの符号が+1
  }

  // --- 関数の明示的な実体化
  
  template void update_by_dropping_piece<BLACK>(Position& pos, Square to, Piece pc);
  template void update_by_dropping_piece<WHITE>(Position& pos, Square to, Piece pc);
  template void update_by_capturing_piece<BLACK>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc);
  template void update_by_capturing_piece<WHITE>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc);
  template void update_by_no_capturing_piece<BLACK>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);
  template void update_by_no_capturing_piece<WHITE>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);
  template void rewind_by_dropping_piece<BLACK>(Position& pos, Square to, Piece pc);
  template void rewind_by_dropping_piece<WHITE>(Position& pos, Square to, Piece pc);
  template void rewind_by_capturing_piece<BLACK>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc);
  template void rewind_by_capturing_piece<WHITE>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc, Piece captured_pc);
  template void rewind_by_no_capturing_piece<BLACK>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);
  template void rewind_by_no_capturing_piece<WHITE>(Position& pos, Square from, Square to, Piece moved_pc, Piece moved_after_pc);

  // --- LONG_EFFECT_LIBRARYの初期化

  void init()
  {
    Effect8::init();
    Effect24::init();
  }

}

#endif // LONG_EFFECT_LIBRARY
