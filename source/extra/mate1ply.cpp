#include "mate1ply.h"

// 超高速1手詰め判定ライブラリ

#if defined(MATE_1PLY) && defined(LONG_EFFECT_LIBRARY)

#include "../position.h"
#include "long_effect.h"

using namespace Effect8; // Effect24のほうは必要に応じて書く。

namespace {

  // 超高速1手詰め判定ライブラリ
  // cf. 新規節点で固定深さの探索を併用するdf-pnアルゴリズム gpw05.pdf
  // →　この論文に書かれている手法をBitboard型の将棋プログラム(やねうら王mini)に適用し、さらに発展させ、改良した。

  // 1手詰め判定高速化テーブルに使う1要素
  struct alignas(2) MateInfo
  {
    // この形において詰ませるのに必要な駒種
    // bit0..歩(の移動)で詰む。打ち歩は打ち歩詰め。
    // bit1..香打ちで詰む。
    // bit2..桂(の移動または打ちで)詰む
    // bit3,4,5,6 ..銀・角・飛・金打ち(もしくは移動)で詰む
    // bit7..何を持ってきても詰まないことを表現するbit(directions==0かつhand_kindの他のbit==0)
    uint8_t/*HandKind*/ hand_kind;

    // 敵玉から見てこの方向に馬か龍を配置すれば詰む。(これが論文からの独自拡張)
    // これが0ならどの駒をどこに打っても、または移動させても詰まないと言える。(桂は除く)
    Directions directions;
  };

  // 1手詰め判定に用いるテーブル。
  // 添字の意味。
  //  bit  0.. 7 : (1) 駒を打つ候補の升(受け方の利きがなく、攻め方の利きがある升) 壁の扱いは0でも1でも問題ない。
  //  bit  8..15 : (2) 王が移動可能な升(攻め方の利きがなく、受け方の駒もない) 壁は0(玉はそこに移動できないので)
  // を与えて、そのときに詰ませられる候補の駒種(HandKind)と打つ場所(玉から見た方角)を返すテーブル。
  // 駒打ちの場合 =>  駒を打つことで大駒の利きが遮断されなければ、これで詰む。
  // 駒移動の場合 =>  移動させることによって移動元から利きが損なわれなく、かつ、移動先の升で大駒の利きが遮断されなければこれで詰む
  // (その他にも色々例外はあるが、詳しくは以下の実装を見るべし)
  // ただし、歩は打ち歩詰めになるので、移動させることが前提。桂は打つ升に敵の利きがないことが前提。
  MateInfo  mate1ply_drop_tbl[0x10000][COLOR_NB];
}

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
// cut_dirs   : toの升で大駒の利きを遮断しているとしたら、その方角
//  ※  それを補償する利きがあるかを調べる)
// to2        : 大駒の利きが遮断されたであろう、詰みに関係する升
//  ※　to2の升は、mate1ply_drop_tblがうまく作られていて、(2)の条件により盤外は除外されている。
// toにPIECEを打つことでto2の升の大駒の利きが1つ減るが、いまのto2の地点の利きが1より大きければ、
// 他の駒が利いているということでto2の升に関しては問題ない。

#define CHECK_PIECE(DROP_PIECE) \
  if (hk & HAND_KIND_ ## DROP_PIECE) {                                                                        \
    Piece pc = make_piece(Us,DROP_PIECE);                                                                     \
    Directions directions = mi.directions & piece_check_around8(pc) & info1;                                  \
    while (directions) {                                                                                      \
      Direct to_direct = pop_directions(directions);                                                          \
      Directions effect_not = piece_effect_not(pc, to_direct);                                                \
      if (effect_not & info2) continue; /*玉に逃げ道がある*/                                                  \
      to = themKing + DirectToDelta(to_direct);                                                               \
      Directions cut_dirs_from_king = cutoff_directions(to_direct, long_effect.directions_of(Us, to))         \
                & effect_not & a8_board_mask & a8_them_movable;                                                                 \
      while (cut_dirs_from_king)                                                                              \
      {                                                                                                       \
        Direct cut_dir_from_king = pop_directions(cut_dirs_from_king);                                        \
        Square to2 = themKing + DirectToDelta(cut_dir_from_king);                                             \
        if (board_effect[Us].e[to2] <= 1)                                                                     \
        goto Next ## DROP_PIECE;                                                                              \
      }                                                                                                       \
      return make_move_drop(DROP_PIECE, to);                                                                  \
    Next ## DROP_PIECE:;                                                                                      \
    }                                                                                                         \
  }

// 超高速1手詰め判定、本体。
template <Color Us>
Move Position::mate1ply_impl() const
{
  Square from, to;
  auto them = ~Us;
  auto themKing = king_square(them);

  // --- 1手詰め判定テーブルのlook up

  // 敵玉周辺の受け方の利きのある升
  Directions a8_effect_them = board_effect[them].around8_greater_than_one(themKing);

  // 敵玉周辺の攻め方の利きのある升
  Directions a8_effect_us = board_effect[Us].around8(themKing);

  // 敵玉周辺の受け方の駒がない升
  Directions a8_them_movable = ~around8(pieces(them), themKing);

  // 敵玉周辺の駒が打てる升 == 駒がない場所
  Directions a8_droppable = around8(~pieces(), themKing);

  // 敵玉の周辺において、盤上が1、壁(盤外)が0であるbitboard
  Directions a8_board_mask = board_mask(themKing);

  //  bit  0.. 7 : (1) 駒を打つ候補の升(受け方の玉以外の利きがなく、攻め方の利きがある升)かつ、駒がない升。
  //  bit  8..15 : (2) 王が移動可能な升(攻め方の利きがなく、受け方の駒もない) 壁は0(玉はそこに移動できないので)
  Directions info1 = ~a8_effect_them & a8_effect_us & a8_droppable & a8_board_mask;  // (1)
  Directions info2 = a8_them_movable & ~a8_effect_us & a8_board_mask; // (2)

  uint16_t info = ((uint16_t)info2 << 8) | info1;

  // 打つことで詰ませられる候補の駒(歩は打ち歩詰めになるので除外されている)
  auto& mi = mate1ply_drop_tbl[info][Us];

  // Queenを持ってきて詰む場所がmi.directionsなので、これがなければ
  // 詰ませられる前提条件を満たしていないなら、これにて不詰めが証明できる。
  // ただし、桂で詰む場合は、mi.hand_kind == HAND_KIND_KNIGHTになっているので…。
  if (!(mi.directions | mi.hand_kind))
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

  auto themKingWW = to_sqww(themKing);

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

  // 利きが2つ以上ある場所
  Directions a8_effect_us_gt1 = board_effect[Us].around8_greater_than_one(themKing); // 1)

  // 桂の打ちと桂の不成による移動による直接王手による詰み

  // 玉の逃げ道がないことはわかっているのであとは桂が打ててかつ、その場所に敵の利きがなければ詰む。
  // あるいは、桂馬を持っていないとしても、その地点に桂馬を跳ねればその桂がpinされていなければ詰む。
  if (mi.hand_kind & HAND_KIND_KNIGHT)
  {
    auto drop_target = knightEffect(them, themKing) & ~pieces(Us);
    while (drop_target) {
      to = drop_target.pop();
      if (!board_effect[them].effect(to))
      {
        // 桂馬を持っていて、8近傍にかかる長い利きを遮断していなくて
        // ここに駒がなければ(自駒は上で除外済みだが)、ここに打って詰み
        if ((ourHand & HAND_KIND_KNIGHT) && !(pieces() & to)) {
          auto cut_directions = long_effect.le16[to].dirs[Us];
          // 遮断していないならこれにて詰み
          if (!cut_directions)
            return make_move_drop(KNIGHT, to);

          // 敵玉から見てここの利きを損失するのでここの利きが2以上でないといけない
          auto direct = direct_knight_of<Us>(themKing, to);
          auto dec_effect = cutoff_directions(direct, cut_directions);
          // 敵玉の8近傍に影響を及ぼす遮断された利きがないのでこれにて詰み
          if (!dec_effect)
            return make_move_drop(KNIGHT, to);

          // dec_effectの各bitに対して2個以上の利きがあるか、敵玉が移動できないか、盤外であればこれにて詰み。
          if (!(dec_effect & ~(a8_effect_us_gt1 | ~a8_board_mask | ~a8_them_movable)))
            return make_move_drop(KNIGHT, to);

          // 長い利きを遮断してしまうのでこのtoの地点はよくない。移動によっては詰まない。
          continue;
        }

        // toに利く桂があるならそれを跳ねて詰み。ただしpinされていると駄目。
        // toに自駒がないことはdrop_targetを求めるときに保証している。敵駒はあって移動によって捕獲できるので問題ない。
        auto froms = knightEffect(them, to) & pieces(Us, KNIGHT);
        while (froms) {
          from = froms.pop();
          if (!(pinned & from))
          {
            auto cut_directions = long_effect.le16[to].dirs[Us];
            // 遮断していないならこれにて詰み
            if (!cut_directions)
              return make_move(from, to);

            // 敵玉から見てここの利きを損失するのでここの利きが2以上でないといけない
            auto direct = direct_knight_of<Us>(themKing, to);
            auto dec_effect = cutoff_directions(direct, cut_directions);
            // 敵玉の8近傍に影響を及ぼす遮断された利きがないのでこれにて詰み
            if (!dec_effect)
              return make_move(from, to);

            // dec_effectの各bitに対して2個以上の利きがあるか、敵玉が移動できないか、盤外であればこれにて詰み。
            if (!(dec_effect & ~(a8_effect_us_gt1 | ~a8_board_mask | ~a8_them_movable)))
              return make_move(from, to);

            // 長い利きを遮断していて詰まない。
          }
        }
      }
    }
  }

  auto ourKing = king_square(Us);

  // ---------------------------------------------------------------
  //     fromをtoに移動させて詰むかどうかを判定する関数
  // ---------------------------------------------------------------

  // ここで判定するのは近接王手におる詰みのみ。両王手による詰みは稀なので除外。
  // まず王手の候補となる升を列挙、そのあと、この移動によって本当に王手になるかだとか、
  // 遠方駒の利きを遮断するかどうか等のチェックが必要。
  // 影の利きがあって詰むパターンは最後に追加。

  // fromからtoに移動させて詰むかどうかを判定する関数
  // to_direct = 敵玉から見てtoの升はどの方向か
  auto is_mated_by_from_to = [&]( Square from, Square to, Direct to_direct)
  {
    Piece pc = piece_on(from);

    // ptをfromからtoに持って行って詰むための条件
    // 6) fromからtoへの移動がpinされていない。
    // 7) ptをtoに持って行ったときに王手になること   : { piece_check_around8[pt] & (1 << to_direct) } != 0
    // 8) ptをtoに配置したときに王に行き場所がなくなること
    //     8a) toに置いたときの利き : piece_effect_mask_around8[pc][to_direct]
    //     8b) 敵玉の移動できる升(受け方の駒がなく、かつ、攻め方の利きがない) : info2
    //   toにおいて詰む条件 とは、敵玉の行き場所がなくなることだから、 8a) & 8b) == 0

    // 9) fromの升で、fromからtoの方向の敵の長い利きがないこと。(back attackによりtoに移動させたときに取られてしまう)
    // 10) toの地点において大駒の利きが遮断されず
    // 11) 移動元から利きが消失しても敵玉8近傍の攻め方の利きが存在すること。
    // ※　9),10)は駒打ちのときと同じだが、11)も考慮しないといけないから難しい。

    // 12) 8近傍において、馬の利きは遮断されないが、龍で4隅からの王手は遮断されうる。
    //  このケースにおいて利きを再計算する必要がある。

    // 13) 桂の成らない(直接)王手で詰むことはない。(桂打ちのときに調査済み)
    // 14) 玉の移動による(直接)王手で詰むことはない。(自殺手)
    // 15) 金の成る手は存在しないし、成れない駒、敵陣にない駒による成る王手も存在しない。

    // ---------------------------------------------------------------
    // 先に成りでの詰みのチェック(こちらのほうが詰む可能性が高いので)
    // ---------------------------------------------------------------

    {
      if (type_of(pc) == KING)
        goto NextCandidate; // 14)

      // 開き王手なら、成らなくとも駄目だから..
      if (discovered(from, to, ourKing, pinned))          // 6)
        goto NextCandidate;

      // 成っていない駒でかつ、fromかtoが敵陣なら成りによる王手も調べないといけない。
      static_assert(GOLD == 7, "GOLD must be 7.");
      if ((type_of(pc) >= GOLD) || !((Bitboard(from) ^ Bitboard(to)) & enemy_field(Us)))
        goto NonProCheck2; // 15)

      auto pro_pc = (Piece)(pc + PIECE_PROMOTE);

      if (!(piece_check_around8(pro_pc) & (1 << to_direct))) // 7)
        goto NonProCheck;

      auto effect_us_not = (type_of(pro_pc) == DRAGON && is_diag(to_direct)) ?
        ~around8(dragonEffect(to, pieces()), themKing) : piece_effect_not(pro_pc, to_direct); // 12)

      if (effect_us_not & info2)     // 8)
        goto NonProCheck;

      if (directions_of(from, to) & long_effect.le16[from].dirs[~Us])  // 9)
        goto NextCandidate; // このとき成ったところでback attackで取られてしまうので次の候補を調べよう。

      // themKingから見て移動元から消失する利き
      auto dec_effect = effects_from(pc, from, pieces());
      // ここの利きがマイナス1
      auto dec_around8 = around8(dec_effect, themKing);

      // これで遮断される方角の利きもマイナス1
      // ただし、ここに駒を置いて発生した長い利きの方角は免除される。
      auto cut_dirs = cutoff_directions(to_direct, long_effect.directions_of(Us, to) & ~LongEffect::long_effect_of(pc));

      // 上の2つの条件の升で、toにその駒を移動させたときに利きがない升に対して調べる。
      // toの地点はすでに条件を満たしているので調べない。(影の利きで王手する場合などがあるので調べてはならない)
      auto dec = (dec_around8 | cut_dirs) & effect_us_not & a8_board_mask & a8_them_movable & ~to_directions(to_direct);

      while (dec)
      {
        Direct cut_direct = pop_directions(dec);

        // ここにおいていくつ利きが減衰しているのか調べる
        uint8_t dec_num = ((dec_around8 >> cut_direct) & 1) + ((cut_dirs >> cut_direct) & 1);

        Square to2 = themKing + DirectToDelta(cut_direct);
        if (board_effect[Us].effect(to2) <= dec_num) // 10),11)
          goto NonProCheck;
      }

      // 利きが足りていたのでこれにて詰む
      return make_move_promote(from, to);
    }

  NonProCheck:;

    {
      Piece pt = type_of(pc);

      // 歩・角・飛で成って詰まないなら不成で詰むことはない。(1手詰めにおいては)
      if (pt == PAWN || pt == BISHOP || pt == ROOK)
        goto NextCandidate;

      // 香で2段目での成で詰まないなら不成で詰むことはない。(1手詰めにおいては)
      if (pt == LANCE && (rank_of(to) == ((Us == BLACK) ? RANK_2 : RANK_8)))
        goto NextCandidate;
    }

  NonProCheck2:;
  // ---------------------------------------------------------------
  //          成らない王手による詰みのチェック
  // ---------------------------------------------------------------

  // 以下、同様。

    {
      if (type_of(pc) == KNIGHT) // 13)
        goto NextCandidate;

      if (discovered(from, to, ourKing, pinned))
        goto NextCandidate;

      if (!(piece_check_around8(pc) & (1 << to_direct)))
        goto NextCandidate;

      auto effect_us_not = (type_of(pc) == DRAGON && is_diag(to_direct)) ?
        ~around8(dragonEffect(to, pieces()), themKing) : piece_effect_not(pc, to_direct);

      if (effect_us_not & info2)
        goto NextCandidate;

      if (directions_of(from, to) & long_effect.le16[from].dirs[~Us])
        goto NextCandidate;

      auto dec_effect = effects_from(pc, from, pieces());
      auto dec_around8 = around8(dec_effect, themKing);
      auto cut_dirs = cutoff_directions(to_direct, long_effect.directions_of(Us, to) & ~LongEffect::long_effect_of(pc));
      auto dec = (dec_around8 | cut_dirs) & effect_us_not & a8_board_mask & a8_them_movable & ~to_directions(to_direct);

      while (dec)
      {
        Direct cut_direct = pop_directions(dec);
        uint8_t dec_num = ((dec_around8 >> cut_direct) & 1) + ((cut_dirs >> cut_direct) & 1);
        Square to2 = themKing + DirectToDelta(cut_direct);
        if (board_effect[Us].effect(to2) <= dec_num)
          goto NextCandidate;
      }
      return make_move(from, to);
    }
    NextCandidate:;

    return MOVE_NONE;
  };


  // ---------------------------------------------------------------
  //       利きが２つある箇所へ移動させることによる詰み
  // ---------------------------------------------------------------

  // 1) 敵玉の8近傍で、味方の利きが2つ以上ある升 : effect[Us].around8_larger_than_two(themKing)
  // 2) 盤内である       : a8_board_mask
  // 3) 攻め方の駒がない : a8_us_movable
  // 4) そこに馬か龍を置いて詰む場所(これで詰まないなら、詰みようがないので) : mi.directions
  // 5) 敵の利きがない   : ~a8_effect_them
  // 移動先の候補の升 = 1) & 2) & 3) & 4) & 5)

  // 攻め方の駒がない升
  Directions a8_us_movable = ~around8(pieces(Us), themKing);
  Bitboard froms;
  Directions to_candicate = a8_effect_us_gt1 & a8_board_mask & a8_us_movable & mi.directions & ~a8_effect_them; // 1) & 2) & 3) & 4) & 5)

  while (to_candicate)
  {
    auto to_direct = pop_directions(to_candicate);
    to = themKing + DirectToDelta(to_direct);

    // toに利かせている駒を列挙して、それぞれについて考える。
    froms = attackers_to(Us, to);
    while (froms)
    {
      from = froms.pop();
      Move m = is_mated_by_from_to(from, to, to_direct);
      if (m != MOVE_NONE)
        return m;
    }
  }

  // ---------------------------------------------------------------
  //     影の利きと自分の利きが1つある箇所へ移動させることによる詰み
  // ---------------------------------------------------------------
  
  // 調べるのは利きが1つのところのみで良い。2つ以上あるところは上ですでにチェック済み。

  // 24近傍で長い利きが敵玉の9近傍に掛かっている升
  auto a24_candidate = long_effect.long_effect24_to_around9<Us>(themKing);

  // 24近傍の自駒のある場所
  auto a24_us_occ = Effect24::around24(pieces(Us),themKing);

  // 盤上の24升のmask
  auto a24_board_mask = Effect24::board_mask(themKing);

  // 移動元の候補となる駒
  auto froms24 = a24_candidate & a24_us_occ & a24_board_mask;

  while (froms24)
  {
    // a) これを玉の8近傍に移動させられるか

    auto from_direct = Effect24::pop_directions(froms24);
    auto from = themKing + Effect24::DirectToDelta(from_direct);

    auto pc = piece_on(from);

    // 利きがあってかつ敵玉の8近傍で自駒がない場所
    auto target = effects_from(pc, from, pieces()) & kingEffect(themKing) & ~pieces(Us);

    while (target)
    {
      auto to = target.pop();

      // b) この移動の方向にこの升での長い利きがなければ移動後、toの升に利きが2つにならないのでNG
      if (!(directions_of(from, to) & long_effect.directions_of(Us,from)))
        continue;

      // c) toの升が利きが1つでなければNG。(2つ以上あるところはチェック済み)
      // d) toの升に(玉を除く)敵の利きがあってはならない。
      if (board_effect[Us].effect(to) != 1 || board_effect[them].effect(to) !=1)
        continue;

      auto to_direct = (Direct)LSB32(directions_of(themKing, to));
      Move m = is_mated_by_from_to(from, to, to_direct);
      if (m != MOVE_NONE)
        return m;
    }
  }

  return MOVE_NONE;
}

Move Position::mate1ply() const
{
  return sideToMove == BLACK ? mate1ply_impl<BLACK>() : mate1ply_impl<WHITE>();
}

namespace Mate1Ply
{
  // Mate1Ply関係のテーブル初期化
  void init()
  {
    for (auto c : COLOR)
      for (uint32_t info = 0; info < 0x10000; ++info)
      {
        //  bit  0.. 7 : (1) 駒を打つ候補の升(受け方の玉以外の利きがなく、攻め方の利きがある升) 壁は0とする。
        Directions info1 = Directions(info & 0xff);

        //  bit  8..15 : (2) 王が移動可能な升(攻め方の利きがなく、受け方の駒もない) 壁は0(玉はそこに移動できないので)
        Directions info2 = Directions(info >> 8);

        Directions directions = DIRECTIONS_ZERO;
        HandKind hk = HAND_KIND_ZERO;

        // info1の場所に駒を打ってみて詰むかどうかを確認
        for (int i = 0; i < 8;++i)
        {
//          Direct dir = pop_directions(info1);
          Direct dir = Direct(i);
          // 駒が打てる場所か
          bool droppable = (info1 & (1 << i));

          // 打つ駒 .. 香、銀、金、角、飛、or Queen
          const Piece drop_pieces[6] = { LANCE,SILVER,GOLD,BISHOP,ROOK,QUEEN };
          for (auto pt : drop_pieces)
          {
            // 駒が打てない場所ならそこにQUEENを持ってきて詰むかだけチェックする
            if (!droppable && pt != QUEEN)
              continue;

            // piece_effectは利きのある場所が0。
            Directions effect_not = piece_effect_not(make_piece(c, pt),dir);

            // 玉が移動できる升
            Directions king_movable = effect_not & info2;

            // 玉が移動できる升がない
            if (!king_movable)
            {
              if (pt == QUEEN)
                directions |= to_directions(dir);
              else
                hk |= HandKind(1 << (pt-1));
            }
          }
          // 王に行き場がないなら桂で詰む
          if (!info2)
            hk |= HandKind(1 << (KNIGHT - 1));
        }

        mate1ply_drop_tbl[info][c].hand_kind = hk;
        mate1ply_drop_tbl[info][c].directions = directions;
      }
  }
}


#endif
