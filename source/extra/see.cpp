#include "../shogi.h"

#ifdef USE_SEE

#include "../position.h"

using namespace Eval;
using namespace Effect8;

namespace {

  // min_attacker()はsee()で使われるヘルパー関数であり、(目的升toに利く)
  // 手番側の最も価値の低い攻撃駒の場所を特定し、その見つけた駒をビットボードから取り除き
  // その背後にあった遠方駒をスキャンする。(あればstmAttackersに追加する)

  // またこの関数はmin_attacker<PAWN>()として最初呼び出され、PAWNの攻撃駒がなければ次に
  // KNIGHTの..というように徐々に攻撃駒をアップグレードしていく。

  // bb = byTypeBB
  // byTypeBBは駒の価値の低い順に並んでいるものとする。
  // occupied = 駒のある場所のbitboard。今回発見された駒は取り除かれる。
  // stmAttackers = 手番側の攻撃駒
  // attackers = toに利く駒(先後両方)。min_attacker(toに利く最小の攻撃駒)を見つけたら、その駒を除去して
  //  その影にいたtoに利く攻撃駒をattackersに追加する。
  // stm = 攻撃駒を探すほうの手番
  // uncapValue = 最後にこの駒が取れなかったときにこの駒が「成り」の指し手だった場合、その価値分の損失が
  // 出るのでそれが返る。

  // 返し値は今回発見されたtoに利く最小の攻撃駒。これがtoの地点において成れるなら成ったあとの駒を返すべき。

  template<int Pt> inline
    Piece min_attacker(const Position& pos,const Bitboard(&bb)[PIECE_TYPE_BITBOARD_NB][COLOR_NB],const Square& to
      , const Bitboard& stmAttackers, Bitboard& occupied, Bitboard& attackers, Color stm,int& uncapValue) {

      // 駒種ごとのbitboardのうち、攻撃駒の候補を調べる
//:      Bitboard b = stmAttackers & bb[Pt];

      Bitboard b = stmAttackers & bb[Pt-1][stm];

      // HDK用の処理なら、KINGを除かないといけないのか…。なんだこりゃ…。
      if (Pt == HDK)
      {
        // bからKINGの場所を取り除いてHorse/Dragonを得る
        b &= ~(Bitboard(pos.king_square(BLACK)) | Bitboard(pos.king_square(WHITE)));
      }

      // なければ、もうひとつ価値の高い攻撃駒について再帰的に調べる
      if (!b)
        return min_attacker<Pt + 1>(pos,bb, to, stmAttackers, occupied, attackers,stm,uncapValue);

      // bにあった駒を取り除く

      Square sq = b.pop();
      occupied ^= sq;

      // このときpinされているかの判定を入れられるなら入れたほうが良いのだが…。
      // この攻撃駒の種類によって場合分け

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
      if (Pt != GOLD && Pt != HDK // 馬・龍
        && !(pos.piece_on(sq) & PIECE_PROMOTE)
        && (canPromote(stm, to) || canPromote(stm,sq)))
        // 成りは敵陣へと、敵陣からの二種類あるので…。
      {
        uncapValue = ProDiffPieceValue[Pt]; // この駒が取り返せなかったときこの分、最後に損をする。

        return (Piece)Pt;
      }
      else
      {
        // GOLDの場合、この駒の実体は成香とかかも知れんし。
        // KINGはHDKを意味するから、馬か龍だし…。馬・龍に関しては成り駒かも知れんし。
        uncapValue = VALUE_ZERO;
        return type_of(pos.piece_on(sq));
      }
  }

  template<> inline
    // 馬、龍もHDKに含めて、KINGの処理もしたいのでこの処理はKING+1としておく。
    Piece min_attacker<KING + 1>(const Position& pos, const Bitboard(&)[PIECE_TYPE_BITBOARD_NB][COLOR_NB], const Square&
      , const Bitboard& , Bitboard&, Bitboard& occ, Color, int& uncapValue) {
      uncapValue = VALUE_ZERO;
      return Piece(KING + 1);

      // bitboardを更新する必要はない。これが最後のサイクルである。

      // KINGの背後から利いていた駒を足す必要はないわけで…
      // (次に敵駒があればKINGを取られるし、自駒がこれ以上toに利いていても意味がない)

      // min_attacker<>()は、stmAttackers(手番側の攻撃駒)がnon zeroであるときに呼び出されることが
      // 呼び出し側で保証されているから、ここに来たということはtoにKINGが利いていたということを意味するので
      // KINGでtoの駒が取れる。
  }

} // namespace


/// Position::see() is a static exchange evaluator: It tries to estimate the
/// material gain or loss resulting from a move.

// Position::see()は静的交換評価器(SEE)である。これは、指し手による駒による得失の結果
// を見積ろうと試みる。

// ※　SEEの解説についてはググれ。いわゆる静止評価。

// ※　KINGを敵の利きに移動させる手は非合法手なので、ここで与えられる指し手にはそのような指し手は含まないものとする。
// また、SEEの地点(to)の駒をKINGで取る手は含まれるが、そのKINGを取られることは考慮しなければならない。

// seeの符号だけわかればいいときに使う。
// 正か、0か負かが返る。
Value Position::see_sign(Move m) const {

  ASSERT_LV3(is_ok(m));

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

   if (is_drop(m))
   {
     // 駒打ちの場合、敵駒が利いていなければ0は確定する。

     // 相手番の攻撃駒
     stm = ~sideToMove;

     // 手番側(相手番)の攻撃駒
     stmAttackers = attackers_to(stm, to, pieces());
     // pieces()をいじる必要はないので、とりあえずコピーせずに元のまま渡しておく。

     if (!stmAttackers)
       return VALUE_ZERO;

     swapList[0] = VALUE_ZERO;
     uncapValue[0] = VALUE_ZERO;
     occupied = pieces();

     // 自駒のうちtoに利く駒も足したものをattackersとする。
     attackers = stmAttackers | attackers_to(~stm, to, occupied);

     captured = move_dropped_piece(m);
   } else if (is_promote(m)) {

     // 成りの処理も上の処理とほぼ同様だが、駒を取り返されないときの値が成りのスコア分だけ大きい

     from = move_from(m);
     swapList[0] = PieceValueCapture[piece_on(to)]; // 最初に捕獲する駒
     uncapValue[0] = ProDiffPieceValue[piece_on(from)]; // この駒を取り返されなかったときに成りの分だけ儲かる。
                                                        //  see()は変動する評価値の半分を返すように設計する

                                                        //  ↑ここが不成りのときと違う。その1。

                                                        // fromの駒がないものとして考える。
     occupied = pieces() ^ from;

     // 相手番に
     stm = ~color_of(piece_on(from));
     //      attackers = attackers_to(stm, to, slide) & occupied;
     //　→　fromは自駒であるのでこの時点で取り除く必要はない。

     // 手番側(相手番)の攻撃駒
     // ※　"stm"はSideToMoveの略だと思われる。
     stmAttackers = attackers_to(stm, to, occupied);

     // なくなったので最初に手番側が捕獲した駒の価値をSEE値として返す
     if (!stmAttackers)
       return Value(swapList[0] + uncapValue[0]);
     //  ↑ここが不成りのときと違う。その2。

     // fromの駒を取り除いて自駒のうちtoに利く駒も足したものをattackersとする。
     attackers = (stmAttackers | attackers_to(~stm, to, occupied)) & occupied;

     captured = type_of(piece_on(from));
     // ↑このSEEの処理ルーチンではPROMOTEのときも生駒にしてしまっていい。

   } else {
     // 成る手と普通の手でわける必要あるのか？よくわからん。
     from = move_from(m);
     swapList[0] = PieceValueCapture[piece_on(to)]; // 最初に捕獲する駒

     // 相手番に
     stm = ~color_of(piece_on(from));
     //      attackers = attackers_to(stm, to, slide) & occupied;
     //　→　fromは自駒であるのでこの時点で取り除く必要はない。

     // fromの駒がないものとして考える。
     occupied = pieces() ^ from;

     // 手番側(相手番)の攻撃駒
     // ※　"stm"はSideToMoveの略だと思われる。
     stmAttackers = attackers_to(stm, to, occupied);

     // なくなったので最初に手番側が捕獲した駒の価値をSEE値として返す
     if (!stmAttackers)
       return Value(swapList[0]);

     // fromの駒を取り除いて自駒のうちtoに利く駒も足したものをattackersとする。
     attackers = (stmAttackers | attackers_to(~stm, to, occupied)) & occupied;

     captured = type_of(piece_on(from));

     // 自殺手だったので大きな負の値を返しておく。
     if (captured == KING)
       return Value(-PieceValueCapture[KING]);

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
    swapList[slIndex] = PieceValueCapture[captured];

    // Locate and remove the next least valuable attacker
    // 次のもっとも価値の低い攻撃駒の位置を示し、取り除く

    // 最も価値の低い駒
    // ※　この関数が呼び出されたあと、occupied,attackersは更新されている。影にあった遠方駒はこのとき追加されている。
    //:      captured = min_attacker<PAWN>(byTypeBB, to, stmAttackers, occupied, attackers);
    captured = min_attacker<PAWN>(*this, piece_bb, to, stmAttackers, occupied, attackers, stm, uncapValue[slIndex]);

    ++slIndex;

    // 相手番に
    stm = ~stm;

    // 次の手番側の攻撃駒
    stmAttackers = attackers & pieces(stm);

    // KINGの捕獲が処理される前に停止する。

    if (captured == (KING + 1) && stmAttackers)
    {
      // 王を取ったみたいなので相手は直前の指し手は指さないはず。
      // ゆえにループをもう抜けてもいいはず。
      swapList[slIndex] = PieceValueCapture[KING];
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


#endif // USE_SEE
