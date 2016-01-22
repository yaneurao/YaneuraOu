#include <iostream>
#include <sstream>

#include "position.h"
#include "misc.h"

using namespace std;
using namespace Effect8;

// 局面のhash keyを求めるときに用いるZobrist key
namespace Zobrist {
  HASH_KEY zero; // ゼロ(==0)
  HASH_KEY side; // 手番(==1)
  HASH_KEY psq[SQ_NB][PIECE_NB]; // 駒pcが盤上sqに配置されているときのZobrist Key
  HASH_KEY hand[COLOR_NB][PIECE_HAND_NB]; // c側の手駒prが一枚増えるごとにこれを加算するZobristKey
  HASH_KEY depth[MAX_PLY]; // 深さも考慮に入れたHASH KEYを作りたいときに用いる(実験用)
}

// ----------------------------------
//           CheckInfo
// ----------------------------------

void CheckInfo::update(const Position& pos) {

  // このクラスのメンバー変数は、このコンストラクタで適切な値をセットしてやる必要がある。

  // 相手の手番
  Color them = ~pos.side_to_move();

  // 敵玉の位置
  ksq = pos.king_square(them);

  // 手番側のpinされている駒
  pinned = pos.pinned_pieces(pos.side_to_move());

  // 動かすと開き王手になる自駒の候補
  dcCandidates = pos.discovered_check_candidates();

  // 駒種Xによって敵玉に王手となる升のbitboard

  // 歩であれば、自玉に敵の歩を置いたときの利きにある場所に自分の歩があればそれは敵玉に対して王手になるので、
  // そういう意味で(ksq,them)となっている。

  Bitboard occ = pos.pieces();

  // この指し手が二歩でないかは、この時点でテストしない。指し手生成で除外する。なるべくこの手のチェックは遅延させる。
  checkSq[PAWN  ] = pawnEffect(them, ksq);
  checkSq[LANCE ] = lanceEffect(them, ksq,occ);
  checkSq[KNIGHT] = knightEffect(them, ksq);
  checkSq[SILVER] = silverEffect(them, ksq);
  checkSq[BISHOP] = bishopEffect(ksq,occ);
  checkSq[ROOK  ] = rookEffect(ksq,occ);
  checkSq[GOLD  ] = goldEffect(them, ksq);

  // 王を移動させて直接王手になることはない。それは自殺手である。
  checkSq[KING  ] = ZERO_BB;

  // 成り駒。この初期化は馬鹿らしいようだが、gives_check()は指し手ごとに呼び出されるので、その処理を軽くしたいので
  // ここでの初期化は許容できる。(このコードはノードの最初に1回呼び出されるだけなので)
  checkSq[PRO_PAWN  ] = checkSq[GOLD];
  checkSq[PRO_LANCE ] = checkSq[GOLD];
  checkSq[PRO_KNIGHT] = checkSq[GOLD];
  checkSq[PRO_SILVER] = checkSq[GOLD];
  checkSq[HORSE     ] = checkSq[BISHOP] | kingEffect(ksq);
  checkSq[DRAGON    ] = checkSq[ROOK  ] | kingEffect(ksq);
}

// ----------------------------------
//       Zorbrist keyの初期化
// ----------------------------------

void Position::init() {
  PRNG rng(20151225); // 開発開始日 == 電王トーナメント2015,最終日

  // 手番としてbit0を用いる。それ以外はbit0を使わない。これをxorではなく加算して行ってもbit0は汚されない。
  SET_HASH(Zobrist::side,1, 0, 0, 0);
  SET_HASH(Zobrist::zero,0, 0, 0, 0);

  // 64bit hash keyは256bit hash keyの下位64bitという解釈をすることで、256bitと64bitのときとでhash keyの下位64bitは合致するようにしておく。
  // これは定跡DBなどで使うときにこの性質が欲しいからである。
  for (auto pc : Piece() )
    for (auto sq : SQ)
      SET_HASH(Zobrist::psq[sq][pc],rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

  for (auto c : COLOR)
    for (Piece pr = PIECE_ZERO ; pr < PIECE_HAND_NB;++pr)
      SET_HASH(Zobrist::hand[c][pr],rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

  for (int i = 0; i < MAX_PLY;++i)
    SET_HASH(Zobrist::depth[i], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>())
}

// depthに応じたZobrist Hashを得る。depthを含めてhash keyを求めたいときに用いる。
HASH_KEY DepthHash(int depth) { return Zobrist::depth[depth]; }

// ----------------------------------
//  Position::set()とその逆変換sfen()
// ----------------------------------

Position& Position::operator=(const Position& pos){

  std::memcpy(this, &pos, sizeof(Position));

  // コピー元にぶら下がっているStateInfoに依存してはならないのでコピーしてdetachする。
  std::memcpy(&startState, st, sizeof(StateInfo));
  st = &startState;

  nodes = 0;

  return *this;
}

void Position::clear()
{
  std::memset(this, 0, sizeof(Position));
  st = &startState;
}


// Pieceを綺麗に出力する(USI形式ではない) 先手の駒は大文字、後手の駒は小文字、成り駒は先頭に+がつく。盤面表示に使う。
#ifndef PRETTY_JP
std::string pretty(Piece pc) { return std::string(USI_PIECE).substr(pc * 2, 2); }
#else
std::string pretty(Piece pc) { return std::string(" □ 歩 香 桂 銀 角 飛 金 玉 と 杏 圭 全 馬 龍 菌 王^歩^香^桂^銀^角^飛^金^玉^と^杏^圭^全^馬^龍^菌^王").substr(pc * 3, 3); }
#endif

// sfen文字列で盤面を設定する
void Position::set(std::string sfen)
{
  clear();

  // 変な入力をされることはあまり想定していない。
  // sfen文字列は、普通GUI側から渡ってくるのでおかしい入力であることはありえないからである。

  // --- 盤面

  // 盤面左上から。Square型のレイアウトに依らずに処理を進めたいため、Square型は使わない。
  File f = FILE_9;
  Rank r = RANK_1;

  std::istringstream ss(sfen);
  // 盤面を読むときにスペースが盤面と手番とのセパレーターなのでそこを読み飛ばさないようにnoskipwsを指定しておく。
  ss >> std::noskipws;

  uint8_t token;
  bool promote = false;
  size_t idx;

  evalList.clear();

  // PieceListを更新する上で、どの駒がどこにあるかを設定しなければならないが、
  // それぞれの駒をどこまで使ったかのカウンター
  PieceNo piece_no_count[KING] = { PIECE_NO_ZERO,PIECE_NO_PAWN,PIECE_NO_LANCE,PIECE_NO_KNIGHT,
    PIECE_NO_SILVER, PIECE_NO_BISHOP, PIECE_NO_ROOK,PIECE_NO_GOLD};

  // 先手玉のいない詰将棋とか、駒落ちに対応させるために、存在しない駒はすべてBONA_PIECE_ZEROにいることにする。
  for (PieceNo pn = PIECE_NO_ZERO; pn < PIECE_NO_NB; ++pn)
    evalList.put_piece(pn, SQ_ZERO , PRO_GOLD); // 金成りはないのでこれでBONA_PIECE_ZEROとなる。
  kingSquare[BLACK] = kingSquare[WHITE] = SQ_NB;

  while ((ss >> token) && !isspace(token))
  {
    // 数字は、空の升の数なのでその分だけ筋(File)を進める
    if (isdigit(token))
      f -= File(token - '0');
    // '/'は次の段を意味する                              
    else if (token == '/')
    {
      f = FILE_9;
      ++r;
    }
    // '+'は次の駒が成駒であることを意味する
    else if (token == '+')
      promote = true;
    // 駒文字列か？
    else if ((idx = PieceToCharBW.find(token)) != string::npos)
    {
      PieceNo piece_no = 
        (idx == B_KING) ? PIECE_NO_BKING : // 先手玉
        (idx == W_KING) ? PIECE_NO_WKING : // 後手玉
        piece_no_count[raw_type_of(Piece(idx))]++; // それ以外

      // 盤面の(f,r)の駒を設定する
      put_piece(f | r, Piece(idx + (promote ? PIECE_PROMOTE : 0)), piece_no);

      // 1升進める
      --f;

      // 成りフラグ、戻しておく。
      promote = false;
    }

  }

  // --- 手番

  ss >> token;
  sideToMove = (token == 'w' ? WHITE : BLACK);
  ss >> token; // 手番と手駒とを分かつスペース

  // --- 手駒

  hand[BLACK] = hand[WHITE] = (Hand)0;
  int ct = 0;
  while ((ss >> token) && !isspace(token))
  {
    // 手駒なし
    if (token == '-')
      break;

    if (isdigit(token))
      // 駒の枚数。歩だと18枚とかあるので前の値を10倍して足していく。
      ct = (token - '0') + ct * 10;
    else if ((idx = PieceToCharBW.find(token)) != string::npos)
    {
      // 個数が省略されていれば1という扱いをする。
      ct = max(ct, 1);
      add_hand(hand[color_of(Piece(idx))],type_of(Piece(idx)), ct);

      // FV38などではこの個数分だけpieceListに突っ込まないといけない。
      for (int i = 0; i < ct; ++i)
      {
        Piece rpc = raw_type_of(Piece(idx));
        PieceNo piece_no = piece_no_count[rpc]++;
        ASSERT_LV1(is_ok(piece_no));
        evalList.put_piece(piece_no, color_of(Piece(idx)), rpc, i);
      }

      ct = 0;
    }
  }

  // --- 手数(平手の初期局面からの手数)

  // gamePlyとして将棋所では(検討モードなどにおいて)ここで常に1が渡されている。
  // 検討モードにおいても棋譜上の手数を渡して欲しい気がするし、棋譜上の手数がないなら0を渡して欲しい気はする。
  // ここで渡されてきた局面をもとに探索してその指し手を定跡DBに登録しようとするときに、ここの手数が不正確であるのは困る。
  gamePly = 0;
  ss >> std::skipws >> gamePly;

  // --- StateInfoの更新

  set_state(st);

  // --- evaluate

  st->materialValue = Eval::material(*this);

  // --- effect

#ifdef LONG_EFFECT_LIBRARY
  // 利きの全計算による更新
  LongEffect::calc_effect(*this);
#endif

  // --- validation

  // これassertにしてしまうと、先手玉のいない局面や駒落ちの局面で落ちて困る。
  if (!is_ok(*this))
      std::cout << "info string Illigal Position?" << endl;
}

// 局面のsfen文字列を取得する。
// Position::set()の逆変換。
const std::string Position::sfen() const
{
  std::ostringstream ss;

  // --- 盤面
  int emptyCnt;
  for (Rank r = RANK_1; r <= RANK_9; ++r)
  {
    for (File f = FILE_9; f >= FILE_1; --f)
    {
      // それぞれの升に対して駒がないなら
      // その段の、そのあとの駒のない升をカウントする
      for (emptyCnt = 0; f >= FILE_1 && piece_on(f | r) == NO_PIECE; --f)
        ++emptyCnt;

      // 駒のなかった升の数を出力
      if (emptyCnt)
        ss << emptyCnt;

      // 駒があったなら、それに対応する駒文字列を出力
      if (f >= FILE_1)
        ss << (piece_on(f | r));
    }

    // 最下段以外では次の行があるのでセパレーターである'/'を出力する。
    if (r < RANK_9)
      ss << '/';
  }

  // --- 手番
  ss << (sideToMove == WHITE ? " w " : " b ");

  // --- 手駒(UCIプロトコルにはないがUSIプロトコルにはある)
  int n;
  bool found = false;
  for (Color c = BLACK; c <= WHITE; ++c)
    for (Piece p = PAWN; p < PIECE_HAND_NB; ++p)
    {
      // その種類の手駒の枚数
      n = hand_count( hand[c] , p);
      // その種類の手駒を持っているか
      if (n != 0)
      {
        // 手駒が1枚でも見つかった
        found = true;

        // その種類の駒の枚数。1ならば出力を省略
        if (n != 1)
          ss << n;

        ss << PieceToCharBW[make_piece(c, p)];
      }
    }

  // 手駒がない場合はハイフンを出力
  ss << (found ? " " : "- " );
  
  // --- 初期局面からの手数
  ss << gamePly;

  return ss.str();
}

void Position::set_state(StateInfo* si) const {

  // --- bitboard

  // この局面で自玉に王手している敵駒
  st->checkersBB = attackers_to(~sideToMove, king_square(sideToMove));

  // --- hash keyの計算
  si->key_board_ = sideToMove == BLACK ? Zobrist::zero : Zobrist::side;
  si->key_hand_ = Zobrist::zero;
  for (auto sq : pieces())
  {
    auto pc = piece_on(sq);
    si->key_board_ += Zobrist::psq[sq][pc];
  }
  for (auto c : COLOR)
    for (Piece pr = PAWN; pr < PIECE_HAND_NB; ++pr)
      si->key_hand_ += Zobrist::hand[c][pr] * (int64_t)hand_count(hand[c], pr) ; // 手駒はaddにする(差分計算が楽になるため)

  // --- hand
  si->hand = hand[sideToMove];

}

// ----------------------------------
//           Positionの表示
// ----------------------------------

// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os,const Position& pos)
{
  // 盤面
  for (Rank r = RANK_1; r <= RANK_9; ++r)
  {
    for (File f = FILE_9; f >= FILE_1; --f)
      os << pretty(pos.board[f | r]);
    os << endl;
  }

#ifndef PRETTY_JP
  // 手駒
  os << "BLACK HAND : " << pos.hand[BLACK] << " , WHITE HAND : " << pos.hand[WHITE] << endl;

  // 手番
  os << "Turn = " << pos.sideToMove << endl;
#else
  os << "先手 手駒 : " << pos.hand[BLACK] << " , 後手 手駒 : " << pos.hand[WHITE] << endl;
  os << "手番 = " << pos.sideToMove << endl;
#endif

  // sfen文字列もおまけで表示させておく。(デバッグのときに便利)
  os << "sfen " << pos.sfen() << endl;

  return os;
}

#ifdef KEEP_LAST_MOVE
#include <stack>

// 開始局面からこの局面にいたるまでの指し手を表示する。
std::string Position::moves_from_start(bool is_pretty) const
{
  StateInfo* p = st;
  std::stack<StateInfo*> states;
  while (p->previous != nullptr)
  {
    states.push(p);
    p = p->previous;
  }

  stringstream ss;
  while (states.size())
  {
    auto& top = states.top();
    if (is_pretty)
      ss << pretty(top->lastMove , top->lastMovedPieceType) << ' ';
    else
      ss << top->lastMove << ' ';
    states.pop();
  }
  return ss.str();
}
#endif


// ----------------------------------
//      ある升へ利いている駒など
// ----------------------------------

// sに利きのあるc側の駒を列挙する。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
Bitboard Position::attackers_to(Color c, Square sq, const Bitboard& occ) const
{
  Color them = ~c;

  // sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
  return
    (pawnEffect(them, sq) & pieces(c, PAWN))
    | (lanceEffect(them, sq, occ) & pieces(c, LANCE))
    | (knightEffect(them, sq) & pieces(c, KNIGHT))
    | (silverEffect(them, sq) & (pieces(c, SILVER) | pieces(c, HDK)))
    | (goldEffect(them, sq) & (pieces(c, GOLD) | pieces(c, HDK)))
    | (bishopEffect(sq, occ) & pieces(c, BISHOP))
    | (rookEffect(sq, occ) & pieces(c, ROOK));
//    | (kingEffect(sq) & pieces(c, HDK));
  // →　HDKは、銀と金のところに含めることによって、参照するテーブルを一個減らして高速化しようというAperyのアイデア。

}

// 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
inline Bitboard Position::attackers_to_pawn(Color c, Square pawn_sq) const
{
  Color them = ~c;
  const Bitboard& occ = pieces();

  // 馬と龍
  const Bitboard bb_hd = (kingEffect(pawn_sq) & pieces(c, HDK) & ~Bitboard(king_square(c)));
  // 馬、龍の利きは考慮しないといけない。しかしここに玉が含まれるので玉は取り除く必要がある。
  // bb_hdは銀と金のところに加えてしまうことでテーブル参照を一回減らす。

    // sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
    // 打ち歩詰め判定なので、その打たれた歩を歩、香、王で取れることはない。(王で取れないことは事前にチェック済)
    return
    (knightEffect(them, pawn_sq) & pieces(c, KNIGHT))
    | (silverEffect(them, pawn_sq) & (pieces(c, SILVER) | bb_hd))
    | (goldEffect(them, pawn_sq) & (pieces(c, GOLD) | bb_hd))
    | (bishopEffect(pawn_sq, occ) & pieces(c, BISHOP))
    | (rookEffect(pawn_sq, occ) & pieces(c, ROOK));
}

// 指し手が、(敵玉に)王手になるかをテストする。

bool Position::gives_check(Move m) const {

  const CheckInfo& ci = st->checkInfo;

  // 指し手がおかしくないか
  ASSERT_LV2(is_ok(m));

  // 開き王手になる候補手を表現するbitboardは正しく設定されているか
  ASSERT_LV3(ci.dcCandidates == discovered_check_candidates());

  // 移動先
  const Square to = move_to(m);
  if (is_drop(m))
  {
    // 打つ駒
    const Piece pt = move_dropped_piece(m);
    // その駒をtoの地点において王手になるかを判定してそのままreturnする。
    // 王手にならないとしても、駒打ちによって開き王手になることはないから、
    // そういう追加の判定は不要。
    return ci.checkSq[pt] & to;

  } else {
    // 移動元
    const Square from = move_from(m);
    // 移動先に来る駒
    const Piece pt = type_of(piece_on(from)) + (is_promote(m) ? PIECE_PROMOTE : NO_PIECE);

    // 直接王手
    if (ci.checkSq[pt] & to)
      return true;

    // 開き王手になる駒の候補があるとして、fromにあるのがその駒で、fromからtoは玉と直線上にないなら
    if (ci.dcCandidates
      && (ci.dcCandidates & from)
      && !is_aligned(ci.ksq , from, to))
      return true;
  }

  return false;
}

// 敵の大駒の影の利きにより、pinされている駒(自駒)および移動させるとkingColor側の玉に対して
// 開き王手になる駒(敵駒)を列挙する。
// c = 列挙する駒の手番
// →　 kingColorと同じであればpinされている駒 , kingColorと異なれば開き王手になる駒
Bitboard Position::check_blockers(Color c, Color kingColor) const
{
  Bitboard b, pinners, result = ZERO_BB;
  Square ksq = king_square(kingColor);

  // pinnersとは、ピンしている駒が取り除かれたときに王手となる大駒(複数あり)である。

  pinners = (pieces(~kingColor,ROOK) & rookStepEffect(ksq))
    | (pieces(~kingColor,BISHOP) & bishopStepEffect(ksq))
    // 香に関しては攻撃駒が先手なら、玉より下側をサーチして、そこにある先手の香を探す。
    | (pieces(~kingColor,LANCE) & lanceStepEffect(kingColor, ksq));

  while (pinners)
  {
    // ksqと大駒とで挟まれている駒を列挙。
    b = between_bb(ksq, pinners.pop()) & pieces();

    // 挟まれている駒が2駒以上あれば、これはpinされているとはみなさない。
    if (!more_than_one(b))
      result |= b & pieces(c);
  }
  return result;
}

// 現局面で指し手がないかをテストする。指し手生成ルーチンを用いるので速くない。探索中には使わないこと。
bool Position::is_mated() const
{
  // a. 王手している駒がない
  // b. stalemateではない
  // ならば、この時点で不詰めが証明される。
  // b.のstatemateは、c. 玉以外いなくて、d. 手駒がない、かつ、e. 玉の行き場所すべてに相手の利きがある
  // ことがその条件であるが、c.かつd.であることだけここでは調べて、e.は実際の指し手生成で調べることにする。
  // (stalemate自体レアケースなのでそこを高速化してもあまり意味がない)
  // if (a && b) return false;
  // → if (a && !(c && d && e)) return false; // b = !(c && d && e) より
  // → if (a && !(c && d)) return false;      // eはこのあと判定する
  // → if (a && (!c || !d)) return false;     // ドモルガンの法則より
  auto Us = sideToMove;
  if (!st->checkersBB && (hand[Us] != HAND_ZERO || pieces(Us) != Bitboard(king_square(Us))))
    return false;

  // 不成で詰めろを回避できるパターンはないのでLEGAL_ALLである必要はない。
  return MoveList<LEGAL>(*this).size() == 0;
}

// ----------------------------------
//      指し手の合法性のテスト
// ----------------------------------

bool Position::legal_drop(const Square to) const
{
  // この歩に利いている自駒がなければ詰みには程遠いのでtrue
  if (!effected_to(sideToMove, to))
    return true;

  // ここに利いている敵の駒があり、その駒で取れるなら打ち歩詰めではない
  // ここでは玉は除外されるし、香が利いていることもないし、そういう意味では、特化した関数が必要。
  Bitboard b = attackers_to_pawn(~sideToMove, to);
  Bitboard pinned = check_blockers(~sideToMove, ~sideToMove);

  if (b & ~pinned)
    return true; // pinされていない駒が1つでもあるなら、その駒で取って何事もない。

    // 攻撃駒はすべてpinされていたということであり、
    // 王の頭に打たれた打ち歩をpinされている駒で取れるケースは、
    // いろいろあるが、いずれも玉の頭方向以外のところから頭方向への移動であるから、pinされている方向への移動ということはありえない。
    // ゆえに、この歩は取れないことが確定した。

    // 玉の退路を探す
    // 自駒がなくて、かつ、to(はすでに調べたので)以外の地点

    // 相手玉の場所
  Square sq_king = king_square(~sideToMove);
  Bitboard escape_bb = kingEffect(sq_king) & ~pieces(~sideToMove);
  escape_bb ^= to;
  auto occ = pieces() ^ to; // toには歩をおく前提なので、ここには駒があるものとして、これでの利きの遮断は考えないといけない。
  while (escape_bb)
  {
    Square king_to = escape_bb.pop();
    if (!attackers_to(sideToMove, king_to, occ))
      return true; // 退路が見つかったので打ち歩詰めではない。
  }

  // すべての検査を抜けてきたのでこれは打ち歩詰めの条件を満たしている。
  return false;
}

// ※　mがこの局面においてpseudo_legalかどうかを判定するための関数。
bool Position::pseudo_legal(const Move m) const {

  const Color us = sideToMove;
  const Color them = ~us;
  const Square to = move_to(m); // 移動先

  if (is_drop(m))
  {
    const Piece pr = move_dropped_piece(m);
    // 置換表から取り出してきている以上、一度は指し手生成ルーチンで生成した指し手のはずであり、
    // KING打ちのような値であることはないものとする。

    // 打つ先の升が埋まっていたり、その手駒を持っていなかったりしたら駄目。
    if (piece_on(to) != NO_PIECE || hand_count(hand[us], pr) == 0)
      return 0;

    if (in_check())
    {
      // 王手されている局面なので合駒でなければならない
      Bitboard target = checkers();
      Square checksq = target.pop();

      // 王手している駒を1個取り除いて、もうひとつあるということは王手している駒が
      // 2つあったということであり、両王手なので合い利かず。
      if (target)
        return false;

      // 王と王手している駒との間の升に駒を打っていない場合、それは王手を回避していることに
      // ならないので、これは非合法手。
      if (!(between_bb(checksq, king_square(us)) & to))
        return false;
    }

    // 二歩の判定
    if (pr == PAWN && (pieces(us, PAWN) & FILE_BB[file_of(to)]))
      return false;

    // 打ち歩詰めの判定はlegal()のほうで行なうのでここではしない。

    // --- 移動できない升への歩・香・桂打ちについて

    // 打てない段に打つ歩・香・桂の指し手はそもそも生成されていない。
    // 置換表のhash衝突で、後手の指し手が先手の指し手にならないことは保証されている。
    // (先手の手番の局面と後手の手番の局面とのhash keyはbit0で区別しているので)

    // ゆえに、ここではこれ以上のチェックは不要なのである。

  } else {

    const Square from = move_from(m);
    const Piece pc = piece_on(from);

    // 動かす駒が自駒でなければならない
    if (pc == NO_PIECE || color_of(pc) != us)
      return false;

    // toに移動できないといけない。
    if (!(effects_from(pc, from, pieces()) & to))
      return false;

    // toの地点に自駒があるといけない
    if (pieces(us) & to)
      return false;

    Piece pt = type_of(pc);
    if (is_promote(m))
    {
      // --- 成る指し手

      // 成れない駒の成りではないことを確かめないといけない。
      if (pt >= KING)
        return false;

      // 移動先が敵陣でないと成れない。先手が置換表衝突で後手の指し手を引いてきたら、こういうことになりかねない。
      if (!(enemy_field(us) & (Bitboard(from) | Bitboard(to))))
        return false;

      // すでに成っているならこれ以上成れないわけで…
      if (raw_type_of(pt) != pt)
        return false;

    } else {

      // --- 成らない指し手

      // 駒打ちのところに書いた理由により、不成で進めない升への指し手のチェックも不要。

    }

    // 王手している駒があるのか
    if (checkers())
    {
      // このとき、指し手生成のEVASIONで生成される指し手と同等以上の条件でなければならない。

      // 動かす駒は王以外か？
      if (type_of(pc) != KING)
      {
        // 両王手なら王の移動をさせなければならない。
        if (more_than_one(checkers()))
          return false;

        // 指し手は、王手を遮断しているか、王手している駒の捕獲でなければならない。
        // ※　王手している駒と王の間に王手している駒の升を足した升が駒の移動先であるか。
        // 例) 王■■■^飛
        // となっているときに■の升か、^飛 のところが移動先であれば王手は回避できている。
        // (素抜きになる可能性はあるが、そのチェックはここでは不要)
        if (!((between_bb(checkers().pop(), king_square(us)) | checkers()) & to))
          return false;
      }

      // 玉の自殺手のチェックはlegal()のほうで調べているのでここではやらない。

    }
  }

  return true;
}


// ----------------------------------
//      局面を進める/戻す
// ----------------------------------

// 指し手で盤面を1手進める。
template <Color Us>
void Position::do_move_impl(Move m, StateInfo& new_st, bool givesCheck)
{
  // 現在の局面のhash keyはこれで、これを更新していき、次の局面のhash keyを求めてStateInfo::key_に格納。
  auto k = st->key_board_ ^ Zobrist::side;
  auto h = st->key_hand_;

  // --- StateInfoの更新

  // StateInfoの構造体のメンバーの上からkeyのところまでは前のを丸ごとコピーしておく。
  // undo_moveで戻すときにこの部分はundo処理が要らないので細かい更新処理が必要なものはここに載せておけばundoが速くなる。
  std::memcpy(&new_st, st, offsetof(StateInfo, checkersBB));

  // StateInfoを遡れるようにpreviousを設定しておいてやる。
  new_st.previous = st;
  st = &new_st;

  // 駒割りの差分計算用
  int materialDiff;

#ifdef KEEP_LAST_MOVE
  st->lastMove = m;
  st->lastMovedPieceType = is_drop(m) ? (Piece)move_from(m) : type_of(piece_on(move_from(m)));
#endif

  // --- 盤面の更新処理

  // 移動先の升
  Square to = move_to(m);
  ASSERT_LV2(is_ok(to));

  if (is_drop(m))
  {
    // --- 駒打ち

    // 移動先の升は空のはず
    ASSERT_LV2(piece_on(to) == NO_PIECE);

    Piece pr = Piece(move_from(m));
    ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

    Piece pc = make_piece(Us, pr);
    PieceNo piece_no = piece_no_of(Us, pr);
    put_piece(to, pc , piece_no);

    // 駒打ちなので手駒が減る
    sub_hand(hand[Us], pr);

    // 王手している駒のbitboardを更新する。
    // 駒打ちなのでこの駒で王手になったに違いない。駒打ちで両王手はありえないので王手している駒はいまtoに置いた駒のみ。
    st->checkersBB = givesCheck ? Bitboard(to) : ZERO_BB;

    // 駒打ちは捕獲した駒がない。
    st->capturedType = NO_PIECE;

    // Zobrist keyの更新
    h -= Zobrist::hand[Us][pr];
    k += Zobrist::psq[to][pc];

    materialDiff = 0;

#ifdef LONG_EFFECT_LIBRARY
    // 駒打ちによる利きの更新処理
    LongEffect::update_by_dropping_piece<Us>(*this,to,pc);
#endif

  } else {

    // -- 駒の移動
    Square from = move_from(m);
    ASSERT_LV2(is_ok(from));

    // 移動させる駒
    Piece moved_pc = piece_on(from);
    ASSERT_LV2(moved_pc != NO_PIECE);

    // 移動先に駒の配置
    // もし成る指し手であるなら、成った後の駒を配置する。
    PieceNo piece_no = piece_no_of(moved_pc, from); // 移動元にあった駒のpiece_noを得る
    Piece moved_after_pc;
    if (is_promote(m))
    {
      materialDiff = Eval::ProDiffPieceValue[moved_pc];
      moved_after_pc = moved_pc + PIECE_PROMOTE;
    } else {
      materialDiff = 0;
      moved_after_pc = moved_pc;
    }

    // 移動先の升にある駒
    Piece to_pc = piece_on(to);
    if (to_pc != NO_PIECE)
    {
      // --- capture(駒の捕獲)

#ifdef LONG_EFFECT_LIBRARY
      // 移動先で駒を捕獲するときの利きの更新
      LongEffect::update_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc, to_pc);
#endif

      // 玉を取る指し手が実現することはない。この直前の局面で玉を逃げる指し手しか合法手ではないし、
      // 玉を逃げる指し手がないのだとしたら、それは詰みの局面であるから。

      ASSERT_LV1(type_of(to_pc) != KING);

      Piece pr = raw_type_of(to_pc);

      // このPieceNoの駒が手駒に移動したのでEvalListのほうを更新しておく。
      PieceNo piece_no = piece_no_of(to_pc, to);
      evalList.put_piece(piece_no, Us, pr, hand_count(hand[Us], pr));

      // 駒取りなら現在の手番側の駒が増える。
      add_hand(hand[Us], pr);

      // 捕獲される駒の除去
      remove_piece(to);

      // 捕獲された駒が盤上から消えるので局面のhash keyを更新する
      k -= Zobrist::psq[to][to_pc];
      h += Zobrist::hand[Us][pr];

      // 捕獲した駒をStateInfoに保存しておく。(undo_moveのため)
      st->capturedType = type_of(to_pc);
      // 評価関数で使う駒割りの値も更新
      materialDiff += Eval::PieceValueCapture[to_pc];

    } else {
      st->capturedType = NO_PIECE;

#ifdef LONG_EFFECT_LIBRARY
      // 移動先で駒を捕獲しないときの利きの更新
      LongEffect::update_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif
    }

    // 移動元の升からの駒の除去
    remove_piece(from);

    put_piece(to, moved_after_pc,piece_no);

    // fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
    k -= Zobrist::psq[from][moved_pc];
    k += Zobrist::psq[to][moved_after_pc];

    // 王手している駒のbitboardを更新する。
    if (givesCheck)
    {
      ASSERT_LV1(is_ok(st->previous->checkInfo.ksq)); // CheckInfoが初期化されていないときはこれが変な値になっているかも。
      const CheckInfo& ci = st->previous->checkInfo;

      // 1) 直接王手であるかどうかは、移動によって王手になる駒別のBitboardを調べればわかる。
      st->checkersBB = ci.checkSq[type_of(moved_after_pc)] & to;

      // 2) 開き王手になるのか
      const Square ksq = king_square(~Us);
      if (discovered(from, to, ksq, ci.dcCandidates))
      {
        auto directions = directions_of(from, ksq);
        switch (pop_directions(directions)) {

          // 斜めに利く遠方駒は角(+馬)しかないので、玉の位置から角の利きを求めてその利きのなかにいる角を足す。

        case DIRECT_RU: case DIRECT_RD: case DIRECT_LU: case DIRECT_LD:
          st->checkersBB |= bishopEffect(ksq, pieces()) & pieces(Us, BISHOP); break;

          // 横に利く遠方駒は飛車(+龍)しかないので、玉の位置から飛車の利きを求めてその利きのなかにいる飛車を足す。

        case DIRECT_R: case DIRECT_L:
          st->checkersBB |= rookEffect(ksq,pieces()) & pieces(Us, ROOK); break;

          // fromと敵玉とは同じ筋にあり、かつfromから駒を移動させて空き王手になる。
          // つまりfromから上下を見ると、敵玉と、自分の開き王手をしている遠方駒(飛車 or 香)があるはずなのでこれを追加する。
          // 的玉はpieces(Us)なので含まれないはずであり、結果として自分の開き王手している駒だけが足される。

        case DIRECT_U: case DIRECT_D:
          st->checkersBB |= rookEffectFile(from, pieces()) & pieces(Us); break;

        default: UNREACHABLE;
        }

        // 差分更新したcheckersBBが正しく更新されているかをテストするためのassert
        ASSERT_LV3(st->checkersBB == attackers_to(Us, king_square(~Us)));
      }

    } else
      st->checkersBB = ZERO_BB;
  }
    
  st->materialValue = (Value)(st->previous->materialValue + (Us == BLACK ? materialDiff : -materialDiff));

  // 相手番に変更する。
  sideToMove = ~Us;

  // --- 探索ノード数、rootからの手数などを更新。
 
  // 更新されたhash keyをStateInfoに書き戻す。
  st->key_board_ = k;
  st->key_hand_ = h;

  st->hand = hand[sideToMove];

  ++nodes;
  ++gamePly; // 厳密には、これはrootからの手数ではなく、初期盤面からの手数ではあるが。

}


// 指し手で盤面を1手戻す。do_move()の逆変換。
template <Color Us>
void Position::undo_move_impl(Move m)
{
  // Usは1手前の局面での手番(に呼び出し元でしてある)

  auto to = move_to(m);
  ASSERT_LV2(is_ok(to));

  // 移動後の駒
  auto moved_after_pc = board[to];

  PieceNo piece_no = piece_no_of(moved_after_pc, to); // 移動元のpiece_no == いまtoの場所にある駒のpiece_no

  // 移動前の駒
  Piece moved_pc = is_promote(m) ? (moved_after_pc - PIECE_PROMOTE) : moved_after_pc;

  if (is_drop(m))
  {
    // --- 駒打ち

    // toの場所にある駒を手駒に戻す
    Piece pt = raw_type_of(moved_after_pc);
   
    evalList.put_piece(piece_no, Us, pt, hand_count(hand[Us], pt));
    add_hand(hand[Us], pt);

    // toの場所から駒を消す
    remove_piece(to);

#ifdef LONG_EFFECT_LIBRARY
    // 駒打ちのundoによる利きの復元
    LongEffect::rewind_by_dropping_piece<Us>(*this, to, moved_after_pc);
#endif

  } else {

    // --- 通常の指し手

    auto from = move_from(m);
    ASSERT_LV2(is_ok(from));

    // toの場所から駒を消す
    remove_piece(to);

    // toの地点には捕獲された駒があるならその駒が盤面に戻り、手駒から減る。
    // 駒打ちの場合は捕獲された駒があるということはありえない。
    // (なので駒打ちの場合は、st->capturedTypeを設定していないから参照してはならない)
    if (st->capturedType != NO_PIECE)
    {
      Piece to_pc = st->capturedType;

      // 盤面のtoの地点に捕獲されていた駒を復元する
      PieceNo piece_no = piece_no_of(~Us, raw_type_of(to_pc)); // 捕っていた駒(手駒にある)のpiece_no
      put_piece(to, make_piece(~Us, st->capturedType), piece_no);

      // 手駒から減らす
      sub_hand(hand[Us], raw_type_of(st->capturedType));

      // 成りの指し手だったなら非成りの駒がfromの場所に戻る。さもなくばそのまま戻る。
      put_piece(from, moved_pc, piece_no);

#ifdef LONG_EFFECT_LIBRARY
      // 移動先で駒を捕獲するときの利きの更新
      LongEffect::rewind_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc, make_piece(~Us, to_pc));
#endif

    } else {

      // 成りの指し手だったなら非成りの駒がfromの場所に戻る。さもなくばそのまま戻る。
      put_piece(from, moved_pc, piece_no);

#ifdef LONG_EFFECT_LIBRARY
      // 移動先で駒を捕獲しないときの利きの更新
      LongEffect::rewind_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif

    }
  }

  // --- 相手番に変更
  sideToMove = Us; // Usは先後入れ替えて呼び出されているはず。

  // --- StateInfoを巻き戻す
  st = st->previous;

  --gamePly;
}

// do_move()を先後分けたdo_move_impl<>()を呼び出す。
void Position::do_move(Move m, StateInfo& st, bool givesCheck)
{
  if (sideToMove == BLACK)
    do_move_impl<BLACK>(m, st, givesCheck);
  else
    do_move_impl<WHITE>(m, st, givesCheck);
}

// undo_move()を先後分けたdo_move_impl<>()を呼び出す。
void Position::undo_move(Move m)
{
  if (sideToMove == BLACK)
    undo_move_impl<WHITE>(m); // 1手前の手番が返らないとややこしいので入れ替えておく。
  else
    undo_move_impl<BLACK>(m);
}


// ----------------------------------
//      内部情報の正当性のテスト
// ----------------------------------

bool Position::pos_is_ok() const
{
  // Bitboardの完全なテストには時間がかかるので、あまりややこしいテストは現実的ではない。

#if 0
  // 1) 盤上の駒と手駒を合わせて40駒あるか。

  // それぞれの駒のあるべき枚数
  const int ptc0[KING] = { 2/*玉*/,18/*歩*/,4/*香*/,4/*桂*/,4/*銀*/,2/*角*/,2/*飛*/,4/*金*/};
  // カウント用の変数
  int ptc[PIECE_WHITE] = { 0 };

  int count = 0;
  for (auto sq : SQ)
  {
    Piece pc = piece_on(sq);
    if (pc != NO_PIECE)
    {
      ++count;
      ++ptc[raw_type_of(pc)];
    }
  }
  for (auto c : COLOR)
    for (Piece pr = PIECE_HAND_ZERO; pr < PIECE_HAND_NB; ++pr)
    {
      int ct = hand_count(hand[c], pr);
      count += ct;
      ptc[pr] += ct;
    }

  if (count != 40)
    return false;

  // 2) それぞれの駒の枚数は合っているか
  for (Piece pt = PIECE_ZERO ; pt < KING; ++pt)
    if (ptc[pt] != ptc0[pt])
      return false;
#endif

  // 3) 王手している駒
  if (st->checkersBB != attackers_to(~sideToMove, king_square(sideToMove)))
    return false;

  // 4) 相手玉が取れるということはないか
  if (attackers_to(sideToMove, king_square(~sideToMove)))
    return false;

  // 5) occupied bitboardは合っているか
  if ((pieces() != (pieces(BLACK) | pieces(WHITE))) || (pieces(BLACK) & pieces(WHITE)))
    return false;

  return true;
}
