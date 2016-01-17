#include "mate1ply.h"

// 超高速1手詰め判定ライブラリ

#ifdef MATE_1PLY

#include "../position.h"

// 1手詰め判定に用いるテーブル。
// 添字の意味。
//  bit  0.. 8 : (1) 駒を打つ候補の升(受け方の利きがなく、攻め方の利きがある升) 壁の扱いは0でも1でも問題ない。
//  bit  9..15 : (2) 王が移動可能な升(攻め方の利きがなく、受け方の駒もない) 壁は0(玉はそこに移動できないので)
// を与えて、そのときに詰ませられる候補の駒種(HandKind)と打つ場所(玉から見た方角)を返すテーブル。
// 駒打ちの場合 =>  駒を打つことで大駒の利きが遮断されなければ、これで詰む。
// 駒移動の場合 =>  移動させることによって移動元から利きが損なわれなく、かつ、移動先の升で大駒の利きが遮断されなければこれで詰む
// (その他にも色々例外はあるが、詳しくは以下の実装を見るべし)
// ただし、歩は打ち歩詰めになるので、移動させることが前提。桂は打つ升に敵の利きがないことが前提。
MateInfo  mate1ply_drop_tbl[0x10000][COLOR_NB];

// 玉から見て、dの方向(Directionsの値をpop_lsb()したもの)にpcを置いたときに
// 大駒の利きを遮断してしまう利きの方向。(遮断しても元の駒の利きがそこに到達しているならそれは遮断しているとはみなさない)
Effect8::Directions cutoff_directions[PIECE_NB][8];

// 各升の利きの数
LongEffect::EffectNumBoard effect[COLOR_NB];

// 長い利き。
LongEffect::LongEffectBoard long_effect[COLOR_NB];

// 駒pcを敵玉から見てdirの方向に置いたときの敵玉周辺に対するの利きが届かない場所が1、届く場所が0(普通の利きと逆なので注意)
Effect8::Directions piece_effect_mask_around8[PIECE_NB][Effect8::DIRECTIONS_NB];

// 駒pcを敵玉周辺におくとして、そのときに王手になる升を敵玉から見た方角で表現したbit列
Effect8::Directions piece_check_around8[PIECE_NB];

// --------------------------------------
//        超高速1手詰め判定
// --------------------------------------

// 解説)
// directions : PIECE 打ちで詰む可能性のある、玉から見た方角
//  ※　大駒の利きを遮断していなければこれで詰む。
//  これは、以下の  1) & 2) & 3)
//     1) 龍か馬をそこにおけば詰むことがわかっている升 : mi.directions
//     2) 王手になる : piece_check_around8[pc]
//     3) 攻め方の利きがある : info1
// 
// to         : PIECEを実際に打つ升
//     4) 攻め方がtoに駒を置いたときにその駒による利きがない場所 : piece_effect_mask_around8[pc][to_direct]
//     5) 敵玉の移動できる升(受け方の駒がなく、かつ、攻め方の利きがない) : info2
//   toにおいて詰む条件 とは、敵玉の行き場所がなくなることだから、 4) & 5) == 0
//
// cut_off    : toの升で大駒の利きを遮断しているとしたら、その方角
//  ※  それを補償する利きがあるかを調べる)
// to2        : 大駒の利きが遮断されたであろう、詰みに関係する升
//  ※　to2の升は、mate1ply_drop_tblがうまく作られていて、(2)の条件により盤外は除外されている。
// toにPIECEを打つことでto2の升の大駒の利きが1つ減るが、いまのto2の地点の利きが1より大きければ、
// 他の駒が利いているということでto2の升に関しては問題ない。

#define CHECK_PIECE(DROP_PIECE) \
  if (hk & HAND_KIND_ ## DROP_PIECE) {                                                              \
    Piece pc = make_piece(DROP_PIECE, Us);                                                          \
    uint32_t directions = mi.directions & piece_check_around8[pc] & info1;                          \
    while (directions) {                                                                            \
      Effect8::Direct to_direct = Effect8::pop_directions(directions);                              \
      if (~piece_effect_mask_around8[pc][to_direct] & info2) continue;                              \
      to = themKing + Effect8::DirectToDelta(to_direct);                                            \
      uint32_t cut_off = cutoff_directions[pc][to_direct] & long_effect[Us].directions(to);         \
      while (cut_off) {                                                                             \
        Effect8::Direct cut_direction = Effect8::pop_directions(cut_off);                           \
        Square to2 = to + Effect8::DirectToDelta(cut_direction);                                    \
          if (effect[Us].count(to2) <= 1)                                                           \
          goto Next ## DROP_PIECE;                                                                  \
      }                                                                                             \
      return make_move_drop(DROP_PIECE, to);                                                        \
    Next ## DROP_PIECE:;                                                                            \
    }                                                                                               \
  }

// TARGETのbitboardで指定される升の一つをfromとして、fromにある駒が
// pinされていなければfromからtoに移動させる指し手を生成して、returnする。
#define MAKE_MOVE_UNLESS_PINNED(TARGET) \
  while (TARGET) {                \
  from = TARGET.pop();            \
  if (!(pinned & from))           \
    return make_move(from, to);   \
  }

// 超高速1手詰め判定、本体。
template <Color Us>
Move Position::mate1ply_impl() const
{
  Square from, to;
  auto them = ~Us;
  auto themKing = kingSquare[them];

  // --- 1手詰め判定テーブルのlook up

  // 敵玉周辺の受け方の利きのある升
  uint8_t a8_effect_them = effect[them].around8_greater_than_one(themKing);
  
  // 敵玉周辺の攻め方の利きのある升
  uint8_t a8_effect_us = effect[Us].around8(themKing);

  // 受け方の駒がない升
  uint8_t a8_them_movable = 0; //  ~pieces(them).arunrd8(themKing); あとで

  // 敵玉の8近傍において、盤上が1、壁(盤外)が0であるbitboard
  uint8_t board_mask = Effect8::board_mask(themKing);

  //  bit  0.. 8 : (1) 駒を打つ候補の升(受け方の玉以外の利きがなく、攻め方の利きがある升) 壁の扱いは0でも1でも問題ない。
  //  bit  9..15 : (2) 王が移動可能な升(攻め方の利きがなく、受け方の駒もない) 壁は0(玉はそこに移動できないので)
  uint8_t info1 = ~a8_effect_them & a8_effect_us;  // (1)
  uint8_t info2 = a8_them_movable & ~a8_effect_us & board_mask; // (2)

  uint16_t info = ((uint16_t)info2 << 8) | info1;

  // 打つことで詰ませられる候補の駒(歩は打ち歩詰めになるので除外されている)
  auto& mi = mate1ply_drop_tbl[info][Us];

  // 玉(HAND_KIND_KING)は、何も持ってきても詰まないことを表現するbit。
  // 詰ませられる前提条件を満たしていないなら、これにて不詰めが証明できる。
  if (mi.hand_kind & HAND_KIND_KING)
    return MOVE_NONE;

  // -----------------------
  //     駒打ちによる詰み
  // -----------------------

  // 持っている手駒の種類
  auto ourHand = toHandKind(hand[Us]);

  // 歩と桂はあとで移動の指し手のところでチェックするのでいまは問題としない。
  auto hk = (HandKind)(ourHand & ~(HAND_KIND_PAWN | HAND_KIND_KNIGHT | HAND_KIND_KING)) & mi.hand_kind;

  // 駒打ちで詰む条件を満たしていない。
  if (!hk) goto MOVE_MATE;

  // 一番詰みそうな金から調べていく
  CHECK_PIECE(GOLD);
  CHECK_PIECE(SILVER);
  CHECK_PIECE(ROOK) else CHECK_PIECE(LANCE); // 飛車打ちで詰まないときに香打ちで詰むことはないのでチェックを除外
  CHECK_PIECE(BISHOP);

MOVE_MATE:

  // -----------------------
  //     移動による詰み
  // -----------------------

  auto& pinned = state()->checkInfo.pinned;

  // 玉の逃げ道がないことはわかっているのであとは桂が打ててかつ、その場所に敵の利きがなければ詰む。
  // あるいは、桂馬を持っていないとしても、その地点に桂馬を跳ねればその桂がpinされていなければ詰む。
  if (mi.hand_kind & HAND_KIND_KNIGHT)
  {
    auto drop_target = knightEffect(them, themKing) & ~pieces(Us);
    while (drop_target) {
      to = drop_target.pop();
      if (!effect[them].count(to))
      {
        // 桂馬を持っていて、ここに駒がなければ(自駒は上で除外済みだが)、ここに打って詰み
        if ((ourHand & HAND_KIND_KNIGHT) && !(pieces() & to) ) return make_move_drop(KNIGHT, to);

        // toに利く桂があるならそれを跳ねて詰み。ただしpinされていると駄目。
        // toに自駒がないことはdrop_targetを求めるときに保証している。敵駒はあって移動によって捕獲できるので問題ない。
        auto froms = knightEffect(them, to);
        MAKE_MOVE_UNLESS_PINNED(froms);
      }
    }
  }

  // ここで判定するのは近接王手におる詰みのみ。両王手による詰みは稀なので除外。
  // まず王手の候補となる升を列挙、そのあと、この移動によって本当に王手になるかだとか、
  // 遠方駒の利きを遮断するかどうか等のチェックが必要。
  // 影の利きがあって詰むパターンは最後に追加。

  // 受け方の駒がない升
  uint8_t a8_us_movable = 0; //  ~pieces(Us).arunrd8(themKing); あとで
  Bitboard froms;

  // 1) 敵玉の8近傍で、味方の利きが2つ以上ある升 : effect[Us].around8_larger_than_two(themKing)
  // 2) 盤内である       : board_mask
  // 3) 攻め方の駒がない : a8_us_movable
  // 4) そこに馬か龍を置いて詰む場所(これで詰まないなら、詰みようがないので) : mi.directions
  // 移動先の候補の升 = 1) & 2) & 3) & 4)

  uint8_t a8_effect_us_gt1 = effect[Us].around8_greater_than_one(themKing); // 1)
  uint32_t to_candicate = a8_effect_us_gt1 & board_mask & a8_us_movable & mi.directions; // 1) & 2) & 3) & 4)

  while (to_candicate)
  {
    auto to_direct = Effect8::pop_directions(to_candicate);
    to = themKing + Effect8::DirectToDelta(to_direct);

    // toに利かせている駒を列挙して、それぞれについて考える。
    froms = attackers_to(Us, to);
    while (froms)
    {
      from = froms.pop();
      Piece pt = type_of(piece_on(from));

      // ptをfromからtoに持って行って詰むための条件
      // 4) ptをtoに持って行ったときに王手になること   : { piece_check_around8[pt] & (1 << to_direct) } != 0
      // 5) ptをtoに配置したときに王に行き場所がなくなること
      //     5a) toに置いたときの利き : piece_effect_mask_around8[pc][to_direct]
      //     5b) 敵玉の移動できる升(受け方の駒がなく、かつ、攻め方の利きがない) : info2
      //   toにおいて詰む条件 とは、敵玉の行き場所がなくなることだから、 5a) & 5b) == 0

      // 6) fromの升で、fromからtoの方向の敵の長い利きがないこと。(back attackによりtoに移動させたときに取られてしまう)
      // 7) toの地点において大駒の利きが遮断されず
      // 8) 移動元から利きが消失しても敵玉8近傍の攻め方の利きが存在すること。

      // 7)は駒打ちのときと同じだが、8)も考慮しないといけないから難しい。

      if (!(piece_check_around8[pt] & (1 << to_direct))) // (4)
        goto PromoteCheck;

      if ((piece_effect_mask_around8[pt][to_direct] & info2) != 0) // 5)
        goto PromoteCheck;

      // 6)
      uint32_t cut_off = cutoff_directions[pt][to_direct] & long_effect[Us].directions(to);
      while (cut_off) {
        Effect8::Direct cut_direction = Effect8::pop_directions(cut_off);
        Square to2 = to + Effect8::DirectToDelta(cut_direction);
        if (effect[Us].count(to2) <= 1)
        {

        }
      }

    PromoteCheck:;
      // ただし、成っていない駒でかつ、fromかtoが敵陣なら成りによる王手も調べないといけない。
      if (pt < PIECE_PROMOTE && ((Bitboard(from) ^ Bitboard(to)) & enemy_field(them)))
      {
        pt = (Piece)(pt | PIECE_PROMOTE);

        if (!(piece_check_around8[pt] & (1 << to_direct))) // (4)
          continue;

        if ((piece_effect_mask_around8[pt][to_direct] & info2) != 0) // 5)
          continue;

      }

      // かきかけ
    }
  }

  return MOVE_NONE;
}

Move Position::mate1ply() const
{
  return sideToMove == BLACK ? mate1ply_impl<BLACK>() : mate1ply_impl<WHITE>();
}

#endif

