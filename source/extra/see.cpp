#include "../shogi.h"

#if defined (USE_SEE) || defined (USE_SIMPLE_SEE)

#include "../position.h"

using namespace Eval;
using namespace Effect8;

namespace {

  // min_attacker()はsee()で使われるヘルパー関数であり、(目的升toに利く)
  // 手番側の最も価値の低い攻撃駒の場所を特定し、その見つけた駒をビットボードから取り除き
  // その背後にあった遠方駒をスキャンする。(あればstmAttackersに追加する)

  // またこの関数はmin_attacker<PAWN>()として最初呼び出され、PAWNの攻撃駒がなければ次に
  // KNIGHTの..というように徐々に攻撃駒をアップグレードしていく。

  // occupied = 駒のある場所のbitboard。今回発見された駒は取り除かれる。
  // stmAttackers = 手番側の攻撃駒
  // attackers = toに利く駒(先後両方)。min_attacker(toに利く最小の攻撃駒)を見つけたら、その駒を除去して
  //  その影にいたtoに利く攻撃駒をattackersに追加する。
  // stm = 攻撃駒を探すほうの手番
  // uncapValue = 最後にこの駒が取れなかったときにこの駒が「成り」の指し手だった場合、その価値分の損失が
  // 出るのでそれが返る。

  // 返し値は今回発見されたtoに利く最小の攻撃駒。これがtoの地点において成れるなら成ったあとの駒を返すべき。

  template <Color stm>
  Piece min_attacker(const Position& pos,const Square& to
    , const Bitboard& stmAttackers, Bitboard& occupied, Bitboard& attackers
#ifndef USE_SIMPLE_SEE
    ,int& uncapValue
#endif
  ) {

    // 駒種ごとのbitboardのうち、攻撃駒の候補を調べる
//:      Bitboard b = stmAttackers & bb[Pt];

    // 歩、香、桂、銀、金、角、飛…の順で取るのに使う駒を調べる。

    Bitboard b;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_PAWN  ][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_LANCE ][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_KNIGHT][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_SILVER][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_GOLD  ][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_BISHOP][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_ROOK  ][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_HDK][stm] & pos.piece_bb[PIECE_TYPE_BITBOARD_BISHOP][stm]; if (b) goto found;
    b = stmAttackers & pos.piece_bb[PIECE_TYPE_BITBOARD_HDK][stm] & pos.piece_bb[PIECE_TYPE_BITBOARD_ROOK  ][stm]; if (b) goto found;

    // 攻撃駒があるというのが前提条件だから、以上の駒で取れなければ、最後は玉でtoの升に移動出来て
    // 駒を取れるはず。

#ifndef USE_SIMPLE_SEE
    uncapValue = VALUE_ZERO;
#endif

    // ここでサイクルは停止するのだ。
    return KING;

  found:;

    // bにあった駒を取り除く

    Square sq = b.pop();
    occupied ^= sq;

    // このときpinされているかの判定を入れられるなら入れたほうが良いのだが…。
    // この攻撃駒の種類によって場合分け

    const auto& bb = pos.piece_bb;

    auto dirs = directions_of(to, sq);
    if (dirs) switch (pop_directions(dirs))
    {
    case DIRECT_RU: case DIRECT_RD: case DIRECT_LU: case DIRECT_LD:
      // 斜め方向なら斜め方向の升をスキャンしてその上にある角・馬を足す
      attackers |= bishopEffect(to, occupied) & (bb[PIECE_TYPE_BITBOARD_BISHOP][BLACK] | bb[PIECE_TYPE_BITBOARD_BISHOP][WHITE]);

      ASSERT_LV3((bishopStepEffect(to) & sq));
      break;

    case DIRECT_U:
      // 後手の香 + 先後の飛車
      attackers |= rookEffect(to, occupied) & lanceStepEffect(BLACK, to)
        & (bb[PIECE_TYPE_BITBOARD_ROOK][BLACK] | bb[PIECE_TYPE_BITBOARD_ROOK][WHITE] | bb[PIECE_TYPE_BITBOARD_LANCE][WHITE]);

      ASSERT_LV3((lanceStepEffect(BLACK, to) & sq));
      break;

    case DIRECT_D:
      // 先手の香 + 先後の飛車
      attackers |= rookEffect(to, occupied) & lanceStepEffect(WHITE, to)
        & (bb[PIECE_TYPE_BITBOARD_ROOK][BLACK] | bb[PIECE_TYPE_BITBOARD_ROOK][WHITE] | bb[PIECE_TYPE_BITBOARD_LANCE][BLACK]);

      ASSERT_LV3((lanceStepEffect(WHITE, to) & sq));
      break;

    case DIRECT_L: case DIRECT_R:
      // 左右なので先後の飛車
      attackers |= rookEffect(to, occupied)
        & (bb[PIECE_TYPE_BITBOARD_ROOK][BLACK] | bb[PIECE_TYPE_BITBOARD_ROOK][WHITE]);

      ASSERT_LV3(((rookStepEffect(to) & sq)));
      break;

    default:
      UNREACHABLE;
    } else {
      // DIRECT_MISC
      ASSERT_LV3(!(bishopStepEffect(to) & sq));
      ASSERT_LV3(!((rookStepEffect(to) & sq)));
    }

    attackers &= occupied;

    // この駒が成れるなら、成りの値を返すべき。
    // ※　最後にこの地点に残る駒を返すべきなのか。相手が取る/取らないを選択するので。
    Piece pt = type_of(pos.piece_on(sq));
    if (!(pt & PIECE_PROMOTE) && (pt != GOLD)
      && (canPromote(stm, to) || canPromote(stm,sq)))
      // 成りは敵陣へと、敵陣からの二種類あるので…。
    {

#ifndef USE_SIMPLE_SEE
      uncapValue = ProDiffPieceValue[pt]; // この駒が取り返せなかったときこの分、最後に損をする。
#endif
      return pt;
    }
    else
    {
      // GOLDの場合、この駒の実体は成香とかかも知れんし。
      // KINGはHDKを意味するから、馬か龍だし…。馬・龍に関しては成り駒かも知れんし。
#ifndef USE_SIMPLE_SEE
      uncapValue = VALUE_ZERO;
#endif
      return pt;
    }
}


} // namespace


/// Position::see() is a static exchange evaluator: It tries to estimate the
/// material gain or loss resulting from a move.

// Position::see()は静的交換評価器(SEE)である。これは、指し手による駒による得失の結果
// を見積ろうと試みる。

// 最初に動かす駒側の手番から見た値が返る。

// ※　SEEの解説についてはググれ。
//
// ある升での駒の取り合いの結果、どれくらい駒得/駒損するかを評価する。
// 最初に引数として、指し手mが与えられる。この指し手に対して、同金のように取り返され、さらに同歩成のように
// 取り返していき、最終的な結果(評価値のうちの駒割りの部分の増減)を返す。
// ただし、途中の手順では、同金とした場合と同金としない場合とで、(そのプレイヤーは自分が)得なほうを選択できるものとする。

// ※　KINGを敵の利きに移動させる手は非合法手なので、ここで与えられる指し手にはそのような指し手は含まないものとする。
// また、SEEの地点(to)の駒をKINGで取る手は含まれるが、そのKINGを取られることは考慮しなければならない。

#ifndef USE_SIMPLE_SEE

// seeの符号だけわかればいいときに使う。
// 正か、0か負かが返る。
Value Position::see_sign(Move m) const {

  // 捕獲される駒の価値が捕獲する駒の価値より低いのであれば、
  // SEEは負であるはずがないので、このときは速攻帰る。
  // ※　例) 歩で飛車をとって、その升でどう取り合おうと、駒損になることはない。
  // KINGの指し手はつねにここでリターンすることに注意せよ。なぜなら、KINGの
  // 中盤での価値はゼロにセットされているからである。
  // ※　KINGを取り返すような手を読まれても困るのでこれでいいのか…。

  // ※　PieceValueには先後どちらの駒に対してもプラスの値が格納されている。
  // 将棋だと成りがあるのでこの判定怪しいな。成られては元も子もない。歩で飛車を丸取りしたら
  // そのあと馬を作られてもプラスには違いないだろうから、そういうケースでしか明確なプラスではないな。

  // この関数、あとで高速化を考える。

  return see(m);
}

// ↑の関数の下請け。
Value Position::see(Move m) const {

  Square from, to;

  // occupied : 盤上の全駒を表すbitboard
  // attackers : toに利いている駒
  // stmAttackers : 手番側のattackers

  // 初期化されていない警告が出るかも知れんが、コードは正しいので抑制してちょうだい。
  Bitboard occupied, attackers, stmAttackers;
  Color stm;

  // to_sq(m)の地点で捕獲された駒の価値のリスト
  // slIndexは、swapListのindexの意味。swapList[slIndex++] = ... のようにして使う。
  int/*Value*/ swapList[32];

  // 次の手でこの駒を取り返さないときに出る損失。(例 : 相手が最後、歩成で終わった場合、成り分の損失が出るとみなす)
  int /*Value*/ uncapValue[32];

  Piece captured;

  ASSERT_LV3(is_ok(m));

  // 与えられた指し手の移動元、移動先

  to = move_to(m);

  // toの地点にある駒を格納(最初に捕獲できるであろう駒)
  // 捕獲する駒がない場合はNO_PIECEとなり、PieceValue[NO_PIECE] == 0である。

  // 移動させた駒が取り除かれたものとして
  // 行き先の升(to)に対して利いているすべての駒を見つけ出すが、
  // その背後にいるかも知れない遠方駒を追加する。

  // ※　最後、"& occupied"としてあるのは、最初に取り合いを開始した駒、このPositionクラス上からは除去されていないので
  // そいつも列挙されてしまうので除去するためのmask

  // pinされている駒の移動は除外されていることが望ましいのだが…。
  // さもなくば、このなかで素抜きチェックか、1手詰め判定があるべき。

  // 手番stm側のtoへ利いている駒を任意のSlideTableに対して求める関数があるものとする。

  // 移動先の升が防御されていると、計算するのがやや困難になる。
  // 我々は移動先の升に対する一連の捕獲のそれぞれの指し手の捕獲に使った駒の
  // 得失を含む"swap list"を構築することによって、つねに最も価値の低い駒で捕獲する。
  // このそれぞれの捕獲のあと、我々は捕獲に使った駒の背後にある新しい遠方駒を探し出す。

  // 最初に捕獲される駒はfromにあった駒である

  // fromに駒のあった側から見たseeの値を返す。

  stm = is_drop(m) ? sideToMove : color_of(piece_on(move_from(m)));

  // 相手番(の攻撃駒を列挙していく)
  stm = ~stm;

  if (is_drop(m))
  {
    // 駒打ちの場合、敵駒が利いていなければ0は確定する。

#ifdef LONG_EFFECT_LIBRARY
    // 移動させる升に相手からの利きがないなら、seeが負であることはない。
    if (!effected_to(stm,to))
      return VALUE_ZERO;

    // 手番側(相手番)の攻撃駒
    occupied = pieces();

    // ↑の条件から、敵の駒がtoに利いている。
    // このとき、味方の駒もtoに利いているなら、その両方を列挙。
    // さもなくば、敵の駒だけ列挙。
    stmAttackers = attackers_to(stm, to, occupied);
    if (effected_to(~stm,to))
      attackers = stmAttackers | attackers_to(~stm, to, occupied);
    else
      attackers = stmAttackers;

#else
    occupied = pieces();
    stmAttackers = attackers_to(stm, to, occupied);
    if (!stmAttackers)
      return VALUE_ZERO;

    // 自駒のうちtoに利く駒も足したものをattackersとする。
    attackers = stmAttackers | attackers_to(~stm, to, occupied);
#endif

    swapList[0] = VALUE_ZERO;
    uncapValue[0] = VALUE_ZERO;

    captured = move_dropped_piece(m);

   } else if (is_promote(m)) {

    // 成りの処理も上の処理とほぼ同様だが、駒を取り返されないときの値が成りのスコア分だけ大きい

    from = move_from(m);
     
    // 最初に捕獲する駒
    swapList[0] = CapturePieceValue[piece_on(to)];

    // この駒を取り返されなかったときに成りの分だけ儲かる。
    uncapValue[0] = ProDiffPieceValue[piece_on(from)];

#ifdef LONG_EFFECT_LIBRARY

    // 移動させる升に相手からの利きがないなら、seeが負であることはない。
    // fromから移動させているのでfromに相手のlong effectがあるとこのfromの駒をtoに移動させたときに
    // 取られてしまうのでそこは注意が必要。
    if (!effected_to(stm,to,from))
      return Value(swapList[0] + uncapValue[0]);

    // fromの駒がないものとして考える。
    occupied = pieces() ^ from;

    // 手番側(相手番)の攻撃駒
    // ※　"stm"はSideToMoveの略。
    stmAttackers = attackers_to(stm, to, occupied);

#else

    occupied = pieces() ^ from;
    stmAttackers = attackers_to(stm, to, occupied);

    // なくなったので最初に手番側が捕獲した駒の価値をSEE値として返す
    if (!stmAttackers)
      return Value(swapList[0] + uncapValue[0]);
    //  ↑ここが不成りのときと違う。その2。

#endif

    // fromの駒を取り除いて自駒のうちtoに利く駒も足したものをattackersとする。
    attackers = (stmAttackers | attackers_to(~stm, to, occupied)) & occupied;

    captured = type_of(piece_on(from));
    // ↑このSEEの処理ルーチンではPROMOTEのときも生駒にしてしまっていい。

   } else {

    // 成る手と成らない指し手とで処理をわける。
    // 成る手である場合、それが玉である可能性はないのでそのへんの処理が違うし…。
     
    from = move_from(m);
    swapList[0] = CapturePieceValue[piece_on(to)]; // 最初に捕獲する駒

    //      attackers = attackers_to(stm, to, slide) & occupied;
    //　→　fromは自駒であるのでこの時点で取り除く必要はない。

#ifdef LONG_EFFECT_LIBRARY
    if (!effected_to(stm,to,from))
      return Value(swapList[0]);

    // fromの駒がないものとして考える。
    occupied = pieces() ^ from;
    stmAttackers = attackers_to(stm, to, occupied);

#else
    occupied = pieces() ^ from;
    stmAttackers = attackers_to(stm, to, occupied);

    // なくなったので最初に手番側が捕獲した駒の価値をSEE値として返す
    if (!stmAttackers)
      return Value(swapList[0]);

#endif

    // fromの駒を取り除いて自駒のうちtoに利く駒も足したものをattackersとする。
    attackers = (stmAttackers | attackers_to(~stm, to, occupied)) & occupied;

    captured = type_of(piece_on(from));

    // 自殺手だったので大きな負の値を返しておく。
    if (captured == KING)
      return Value(-CapturePieceValue[KING]);

    uncapValue[0] = VALUE_ZERO;
  }

  int slIndex = 1;

  do {
    // 盤上の駒は有限であり、32回以上の取り合いが一箇所で起こりえない。
    // ※　将棋でも香4,桂4,角2,飛2と8方向で..一箇所での取り合いの最大は20か？

    ASSERT_LV3(slIndex < 32);


    // Add the new entry to the swap list
    // swap listに対する新しいentryとして追加する

    // 今回のSEE値をいまの手番側から見たもの = - (ひとつ前のSEE値) + 今回捕獲した駒の価値
    //:    swapList[slIndex] = -swapList[slIndex - 1] + PieceValue[MG][captured];
    // swapList[slIndex] = -swapList[slIndex - 1] + PieceValue[captured];
    // →　これ累積されると最後の処理わかりにくいな..
    swapList[slIndex] = CapturePieceValue[captured];

    // Locate and remove the next least valuable attacker
    // 次のもっとも価値の低い攻撃駒の位置を示し、取り除く

    // 最も価値の低い駒
    // ※　この関数が呼び出されたあと、occupied,attackersは更新されている。影にあった遠方駒はこのとき追加されている。
    //:      captured = min_attacker<PAWN>(byTypeBB, to, stmAttackers, occupied, attackers);
    captured = (stm == BLACK)
      ? min_attacker<BLACK>(*this, to, stmAttackers, occupied, attackers, uncapValue[slIndex])
      : min_attacker<WHITE>(*this, to, stmAttackers, occupied, attackers, uncapValue[slIndex]);

    ++slIndex;

    // 相手番に
    stm = ~stm;

    // 次の手番側の攻撃駒
    stmAttackers = attackers & pieces(stm);

    // KINGの捕獲が処理される前に停止する。

    if (captured == KING && stmAttackers)
    {
      // 王を取ったみたいなので相手は直前の指し手は指さないはず。
      // ゆえにループをもう抜けてもいいはず。
      swapList[slIndex] = CapturePieceValue[KING];
      uncapValue[slIndex] = VALUE_ZERO;
      ++slIndex;
      break;
    }

    // 手番側の攻撃駒がある限り回る。
  } while (stmAttackers);

  // swap listの構築が完了したので、手番側から見て到達可能な最大の点数を
  // 見つけるためにそれ(swap list)を通じてnegamaxを行う。

  // この状況でのnegamaxは、手番側にとって
  // 取りあいを続ける or そこで取り合いをやめるの2択であり、
  // その2択のうち高いほうを採用する。ゆえに、ひとつ前の手番側から見ると
  // この2択の低いほうがそのノードのスコアとなる。

  // ※　ToDo : このSEEの処理、途中で切り上げるような処理にしたほうがいいのではなかろうか…。

  // →　uncapValueも考慮しながら伝播するように以下のように修正する。

  while (--slIndex)
  {

    //  次のうちの小さいほうを選択する。
    //  slIndex深さのnodeにおいて : 
    //  取り合いを続けた場合の損失 , 取り合いをやめた場合の損失
    //   - swapList[slIndex] - uncapValue[slIndex] ,  +uncapValue[slIndex - 1]

    if (-swapList[slIndex] - uncapValue[slIndex] < uncapValue[slIndex - 1])
    {
      // 取り合いを続けたほうが得なので続ける値を上位ノードに伝播
      uncapValue[slIndex - 1] = -swapList[slIndex] - uncapValue[slIndex];
    }
    else {
      // 取り合いやめたほうが得だということなので上位nodeには伝播しない
    }
  }

  return Value(swapList[0] + uncapValue[0]);
}

#else // USE_SIMPLE_SEE

// もっと単純化されたsee()
// 最後になった駒による成りの上昇値は考えない。

Value Position::see_sign(const Move move) const
{
  if (capture(move))
  {
    // 捕獲する指し手で、移動元の駒の価値のほうが移動先の駒の価値より低い場合、これでプラスになるはず。
    // (取り返されたあとの成りを考慮しなければ)
    // 少しいい加減な判定だが、see_sign()を呼び出す状況においてはこれぐらいの精度で良い。
    // KINGで取る手は合法手であるなら取り返されないということだから、ここではプラスを返して良い。
    // ゆえに、中盤用のCapturePieceValue[KING]はゼロを返す。
    
    const Piece ptFrom = type_of(piece_on(move_from(move)));
    const Piece ptTo = type_of(piece_on(move_to(move)));
    if (CapturePieceValue[ptFrom] <= CapturePieceValue[ptTo])
      return static_cast<Value>(1);
  }

  return see(move);
}

Value Position::see(const Move move /*, const int asymmThreshold*/) const
{
  Square to = move_to(move);
  Square from;

  // 次にtoの升で捕獲される駒
  Piece captured;

  Bitboard occupied = pieces();
  
  // 先後の攻撃駒(toに利く駒)
  Bitboard attackers;

  // 手番側の攻撃駒(toに利く駒)
  Bitboard stmAttackers;

  // 移動させる駒側のturnから始まるものとする。
  // 次に列挙すべきは、この駒を取れる敵の駒なので、相手番に。
  Color stm;
  Value swapList[32];

  if (is_drop(move))
  {
    stm = ~sideToMove;
    occupied = pieces();
    stmAttackers = attackers_to(stm, to, occupied);
    if (!stmAttackers)
      return VALUE_ZERO;
    attackers = stmAttackers | attackers_to(~stm, to, occupied);
    swapList[0] = VALUE_ZERO;
    captured = move_dropped_piece(move);
  } else {
    from = move_from(move);
    stm = ~color_of(piece_on(from));
    occupied ^= from;
    stmAttackers = attackers_to(stm, to, occupied);
    if (!stmAttackers) {
      if (is_promote(move)) {
        const Piece ptFrom = type_of(piece_on(from));
        return (Value)(CapturePieceValue[piece_on(to)] + ProDiffPieceValue[ptFrom]);
      }
      return (Value)CapturePieceValue[piece_on(to)];
    }
    captured = type_of(piece_on(from));

    swapList[0] = (Value)CapturePieceValue[piece_on(to)];
    if (is_promote(move)) {
      const Piece ptFrom = type_of(piece_on(from));
      swapList[0] += ProDiffPieceValue[ptFrom];
      captured += PIECE_PROMOTE;
    }

    // fromの駒を除外する必要があるが、次の攻撃駒はstmAttacker側なので、
    // min_attacker()のなかで、そのあとにattackers &= occupiedされるからそのときに除外されるが、
    // KINGのときにおかしくなる。
    attackers = (stmAttackers | attackers_to(~stm, to, occupied)) & occupied;
  }

  int slIndex = 1;
  do {
    ASSERT_LV3(slIndex < 32);

    swapList[slIndex] = -swapList[slIndex - 1] + CapturePieceValue[captured];
    captured = (stm == BLACK)
      ? min_attacker<BLACK>(*this, to, stmAttackers, occupied, attackers)
      : min_attacker<WHITE>(*this, to, stmAttackers, occupied, attackers);

    stm = ~stm;
    stmAttackers = attackers & pieces(stm);
    ++slIndex;

  } while (stmAttackers && (captured != KING || (--slIndex, false)));
  // kingを捕獲する前にslIndexをデクリメントして停止する。

  while (--slIndex)
    swapList[slIndex - 1] = std::min(-swapList[slIndex], swapList[slIndex - 1]);

  return swapList[0];
}

#endif // USE_SIMPLE_SEE


#endif // USE_SEE
