#ifndef _POSITION_H_
#define _POSITION_H_

#include "shogi.h"
#include "bitboard.h"
#include "evaluate.h"
#include "extra/key128.h"
#include "extra/long_effect.h"

// --------------------
//     局面の定数
// --------------------

// 平手の開始局面
const std::string SFEN_HIRATE = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

// --------------------
//     局面の情報
// --------------------

// 駒を動かしたときに王手になるかどうかに関係する情報構造体。
// 指し手が王手になるかどうかを調べるときに使う。やや遅い。
// コンストラクタで必ず局面を渡して、この構造体の情報を初期化させて使う。参照透明。
// 目的 )
//   探索中にある指し手が王手になるかどうかを局面を進めずに知りたい。
//   そうすることで王手にならない指し手なら枝刈り対象にしたりしたいからである。
//   高速に調べるためには盤面上でpinされている駒であるかどうかなどの情報が事前に用意されていなければならない。
// 　それがこの構造体であり、指し手が王手になるかどうかを判定するための関数がPosition::gives_check()である。
struct CheckInfo {

  // 盤面を与えると、この構造体のメンバー変数を適切な値にセットする。
  // 局面探索前などにこのupdateを呼び出して初期化して使う。
  void update(const Position&);

  // 動かすと敵玉に対して空き王手になるかも知れない自駒の候補
  // チェスの場合、駒がほとんどが大駒なのでこれらを動かすと必ず開き王手となる。
  // 将棋の場合、そうとも限らないので移動方向について考えなければならない。
  // ※　dcはdouble check(両王手)の意味。
  Bitboard dcCandidates;

  // 自分側(手番側)の(敵駒によって)pinされている駒
  Bitboard pinned;

  // 自駒の駒種Xによって敵玉が王手となる升のbitboard
  Bitboard checkSq[PIECE_WHITE];

  // 敵王の位置
  Square ksq;
};


// StateInfoは、undo_move()で局面を戻すときに情報を元の状態に戻すのが面倒なものを詰め込んでおくための構造体。
// do_move()のときは、ブロックコピーで済むのでそこそこ高速。
struct StateInfo {

  // ---- ここから下のやつは do_move()のときにコピーされる

  // ---- ここから下のやつは do_move()のときにコピーされない

  // -- Bitboards alignasの関係で最初のほうに持ってきておく。

  // 現局面で手番側に対して王手をしている駒のbitboard。Position::do_move()で更新される。
  Bitboard checkersBB;

  // 王手になる駒等の情報
  CheckInfo checkInfo;

  // この局面のハッシュキー
  // ※　次の局面にdo_move()で進むときに最終的な値が設定される
  // key_board()は盤面のhash。key_hand()は手駒のhash。それぞれ加算したのがkey() 盤面のhash。
  // key_board()のほうは、手番も込み。
  Key key() const { return key_board_ + key_hand_; }
  Key key_board() const { return key_board_; }
  Key key_hand() const { return key_hand_; }

  // HASH_KEY_BITSが128のときはKey128が返るhash key,256のときはKey256
  HASH_KEY long_key() const { return key_board_ + key_hand_; }
  HASH_KEY long_key_board() const { return key_board_; }
  HASH_KEY long_key_hand() const { return key_hand_; }

  // この局面における手番側の持ち駒。優等局面の判定のために必要。
  Hand hand;

  // この局面で捕獲された駒
  // ※　次の局面にdo_move()で進むときにこの値が設定される
  // 先後の区別はなし。馬とか龍など成り駒である可能性はある。
  Piece capturedType;

  friend struct Position;

  // --- evaluate

  // この局面での評価関数の駒割
  Value materialValue;
  // 評価値。(次の局面で評価値を差分計算するときに用いる)
  // まだ計算されていなければsumBKPPの値は、VALUE_NONE
  Value sumBKPP;
  Value sumWKPP;
  Value sumKKP;

#ifdef  KEEP_LAST_MOVE
  // 直前の指し手。デバッグ時などにおいてその局面までの手順を表示出来ると便利なことがあるのでそのための機能
  Move lastMove;

  // lastMoveで移動させた駒(先後の区別なし)
  Piece lastMovedPieceType;
#endif

  // HASH_KEY_BITSで128を指定した場合はBitboardにHashKeyが入っている。
  HASH_KEY key_board_;
  HASH_KEY key_hand_;

  // 一つ前の局面に遡るためのポインタ。
  // NULL MOVEなどでそこより遡って欲しくないときはnullptrを設定しておく。
  StateInfo* previous;

};

// --------------------
//       盤面
// --------------------

// 盤面
struct Position
{
  // --- ctor

  // コンストラクタではおまけとして平手の開始局面にする。
  Position() { clear(); set_hirate(); }

  // コピー。startStateもコピーして、外部のデータに依存しないように(detach)する。
  Position& operator=(const Position& pos);

  // 初期化
  void clear();

  // Positionで用いるZobristテーブルの初期化
  static void init();
  
  // sfen文字列で盤面を設定する
  // ※　内部的にinit()は呼び出される。
  void set(std::string sfen);

  // 局面のsfen文字列を取得する
  // ※ USIプロトコルにおいては不要な機能ではあるが、デバッグのために局面を標準出力に出力して
  // 　その局面から開始させたりしたいときに、sfenで現在の局面を出力出来ないと困るので用意してある。
  const std::string sfen() const;

  // 平手の初期盤面を設定する。
  void set_hirate() { set(SFEN_HIRATE); }

  // --- properties

  // 現局面の手番を返す。
  Color side_to_move() const { return sideToMove; }

  // (将棋の)開始局面からの手数を返す。
  int game_ply() const { return gamePly; }

  // 盤面上の駒を返す
  Piece piece_on(Square sq) const { return board[sq]; }

  // c側の手駒を返す
  Hand hand_of(Color c) const { return hand[c]; }

  // c側の玉の位置を返す
  Square king_square(Color c) const { return kingSquare[c]; }

  // 保持しているデータに矛盾がないかテストする。
  bool pos_is_ok() const;

  // 探索したノード数(≒do_move()が呼び出された回数)を設定/取得する
  void set_nodes_searched(uint64_t n) { nodes = n; }
  int64_t nodes_searched() const { return nodes; }

  // --- Bitboard

  // 先手か後手か、いずれかの駒がある場所が1であるBitboardが返る。
  Bitboard pieces() const { return occupied[COLOR_ALL]; }

  // c == BLACK : 先手の駒があるBitboardが返る
  // c == WHITE : 後手の駒があるBitboardが返る
  Bitboard pieces(Color c) const { ASSERT_LV3(is_ok(c)); return occupied[c]; }

  // 駒がない升が1になっているBitboardが返る
  Bitboard empties() const { return pieces() ^ ALL_BB; }

  // 駒に対するBitboardを得る
  Bitboard pieces(Color c,PieceTypeBitboard pr) const { return piece_bb[pr][c]; }

  // 駒に対応するBitboardを得る。
  Bitboard pieces(Color c,Piece pr) const { ASSERT_LV3(PAWN <= pr && pr <= KING);  return piece_bb[(PieceTypeBitboard)(pr - 1)][c]; }

  // --- 利き

  // sに利きのあるc側の駒を列挙する。
  // (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
  Bitboard attackers_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }
  Bitboard attackers_to(Color c, Square sq, const Bitboard& occ) const;

  // 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
  Bitboard attackers_to_pawn(Color c, Square pawn_sq) const;

  // attackers_to()で駒があればtrueを返す版。(利きの情報を持っているなら、軽い実装に変更できる)
  bool effected_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }

  // --- 局面を進める/戻す

  // 指し手で盤面を1手進める
  // m = 指し手。mとして非合法手を渡してはならない。
  // info = StateInfo。局面を進めるときに捕獲した駒などを保存しておくためのバッファ。
  // このバッファはこのdo_move()の呼び出し元の責任において確保されている必要がある。
  // givesCheck = mの指し手によって王手になるかどうか。
  // この呼出までにst.checkInfo.update(pos)が呼び出されている必要がある。
  void do_move(Move m,StateInfo& st, bool givesCheck);

  // do_move()の4パラメーター版のほうを呼び出すにはgivesCheckも渡さないといけないが、
  // mで王手になるかどうかがわからないときはこちらの関数を用いる。都度CheckInfoのコンストラクタが呼び出されるので遅い。探索中には使わないこと。
  void do_move(Move m, StateInfo& st) { check_info_update(); do_move(m, st, gives_check(m)); }

  // 指し手で盤面を1手戻す
  void undo_move(Move m);

  // --- legality(指し手の合法性)のチェック

  // 生成した指し手(CAPTUREとかNON_CAPTUREとか)が、合法であるかどうかをテストする。
  // 注意 : 事前にcheck_info_update()が呼び出されていること。
  //
  // 指し手生成で合法手であるか判定が漏れている項目についてチェックする。
  // 王手のかかっている局面についてはEVASION(回避手)で指し手が生成されているはずなので
  // ここでは王手のかかっていない局面における合法性のチェック。
  // 具体的には、
  //  1) 移動させたときに素抜きに合わないか
  //  2) 敵の利きのある場所への王の移動でないか
  // ※　連続王手の千日手などについては探索の問題なのでこの関数のなかでは行わない。
  // ※　それ以上のテストは行わないので、置換表から取ってきた指し手などについては、
  // pseudo_legal()を用いて、そのあとこの関数で判定すること。
  bool legal(Move m) const
  {
    ASSERT_LV1(is_ok(st->checkInfo.ksq)); // CheckInfoが初期化されていないときはこれが変な値になっているかも。

    if (is_drop(m))
      // 打ち歩詰めは指し手生成で除外されている。
      return true; //  move_dropped_piece(m) != PAWN || legal_drop(m);
    else
    {
      Color us = sideToMove;
      Square from = move_from(m);

      // もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
      return (type_of(piece_on(from)) == KING) ? !effected_to(~us, move_to(m)) :
        // 玉以外の駒であれば、その駒を動かして自玉が素抜きに合わなければ合法。
        !discovered(from, move_to(m), king_square(us), st->checkInfo.pinned);
    }
  }

  // toの地点に歩を打ったときに打ち歩詰めにならないならtrue。toの前に敵玉がいることまでは確定しているものとする。
  bool legal_drop(const Square to) const;

  // mがpseudo_legalな指し手であるかを判定する。
  // ※　pseudo_legalとは、擬似合法手(自殺手が含まれていて良い)
  // 置換表の指し手でdo_move()して良いのかの事前判定のために使われる。
  // 指し手生成ルーチンのテストなどにも使える。(指し手生成ルーチンはpseudo_legalな指し手を返すはずなので)
  // killerのような兄弟局面の指し手がこの局面において合法かどうかにも使う。
  // ※　置換表の検査だが、pseudo_legal()で擬似合法手かどうかを判定したあとlegal()で自殺手でないことを
  // 確認しなくてはならない。このためpseudo_legal()とlegal()とで重複する自殺手チェックはしていない。
  bool pseudo_legal(const Move m) const;

  // --- StateInfo

  // 現在の局面に対応するStateInfoを返す。
  // たとえば、state()->capturedTypeであれば、前局面で捕獲された駒が格納されている。
  StateInfo* state() const { return st; }

  // gives_check()やlegal()、do_move()を呼び出すときは事前にこれを呼び出しておくこと。
  void check_info_update() { st->checkInfo.update(*this); }

  // --- Evaluation

  // 評価関数で使うための、どの駒番号の駒がどこにあるかなどの情報。
  Eval::EvalList eval_list() { return evalList; }

  // --- misc

  // 現局面で王手がかかっているか
  bool in_check() const { return checkers(); }

  // 敵の大駒の影の利きにより、pinされている駒(自駒)および移動させるとkingColor側の玉に対して
  // 開き王手になる駒(敵駒)を列挙する。
  // c = 列挙する駒の手番
  // →　 kingColorと同じであればpinされている駒 , kingColorと異なれば開き王手になる駒
  Bitboard check_blockers(Color c, Color kingColor) const;

  // 現局面で手番側に王手している駒
  Bitboard checkers() const { return st->checkersBB; }

  // 移動させると敵玉に対して空き王手となる手番側の駒のbitboardを得る
  Bitboard discovered_check_candidates() const { return check_blockers(sideToMove, ~sideToMove); }

  // c側のpinされている駒(その駒を動かすとc側の玉がとられる)
  Bitboard pinned_pieces(Color c) const { return check_blockers(c, c);  }

  // 駒を配置して、内部的に保持しているBitboardなどを更新する。
  void put_piece( Square sq , Piece pc, PieceNo piece_no);

  // 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
  void remove_piece(Square sq);
  
  // 指し手mで王手になるかを判定する。
  // 指し手mはpseudo-legal(擬似合法)の指し手であるものとする。
  // 事前にcheck_info_update()が呼び出されていること。
  bool gives_check(Move m) const;

  // 手番側の駒をfromからtoに移動させると素抜きに遭うのか？
  bool discovered(Square from, Square to, Square ourKing, const Bitboard& pinned) const
  {
    // 1) pinされている駒がないなら素抜きにならない。
    // 2) pinされている駒でなければ素抜き対象ではない
    // 3) pinされている駒でも王と(縦横斜において)直線上への移動であれば合法
    return pinned                        // 1)
      && (pinned & from)                 // 2)
      && !is_aligned(from, to, ourKing); // 3)
  }

  // 現局面で指し手がないかをテストする。指し手生成ルーチンを用いるので速くない。探索中には使わないこと。
  bool is_mated() const;

  // --- 超高速1手詰め判定
#ifdef  MATE_1PLY
  // 現局面で1手詰めであるかを判定する。1手詰めであればその指し手を返す。
  // ただし1手詰めであれば確実に詰ませられるわけではなく、簡単に判定できそうな近接王手による
  // 1手詰めのみを判定する。(要するに判定に漏れがある。)
  Move mate1ply() const;

  // ↑の先後別のバージョン。(内部的に用いる)
  template <Color Us> Move mate1ply_impl() const;
#endif

  // --- デバッグ用の出力

#ifdef KEEP_LAST_MOVE
  // 開始局面からこの局面にいたるまでの指し手を表示する。
  std::string moves_from_start() const { return moves_from_start(false); }
  std::string moves_from_start_pretty() const { return moves_from_start(true); }
  std::string moves_from_start(bool is_pretty) const;
#endif

  // 盤面を出力する。(USI形式ではない) デバッグ用。
  friend std::ostream& operator<<(std::ostream& os, const Position& pos);

  // MoveGenerator(指し手生成器)からは盤面・手駒にアクセス出来る必要があるのでfriend
  template <MOVE_GEN_TYPE gen_type,bool gen_all>
  friend struct MoveGenerator;

protected:

  // --- Bitboards
  // alignas(16)を要求するものを先に宣言。
  
  // 盤上の先手/後手/両方の駒があるところが1であるBitboard
  Bitboard occupied[COLOR_NB + 1];

  // 駒が存在する升を表すBitboard
  Bitboard piece_bb[PIECE_TYPE_BITBOARD_NB][COLOR_NB];

  // stが初期状態で指している、空のStateInfo
  StateInfo startState;

  // sqの地点にpcを置く/取り除く、したとして内部で保持しているBitboardを更新する。
  void xor_piece(Piece pc, Square sq);

  // --- 盤面を更新するときにEvalListの更新のために必要なヘルパー関数

  // c側の手駒ptの最後の1枚のBonaPiece番号を返す
  Eval::BonaPiece bona_piece_of(Color c,Piece pt) const {
    // c側の手駒ptの枚数
    int ct = hand_count(hand[c], pt);
    return (Eval::BonaPiece)(Eval::kpp_hand_index[c][pt].fb + ct - 1);
  }

  // c側の手駒ptの(最後の1枚の)PieceNoを返す
  PieceNo piece_no_of(Color c, Piece pt) const { return evalList.piece_no_of(bona_piece_of(c,pt));}

  // 盤上のpcの駒のPieceNoを返す
  PieceNo piece_no_of(Piece pc, Square sq) const { return evalList.piece_no_of((Eval::BonaPiece)(Eval::kpp_board_index[pc].fb + sq)); }

  // ---

  // 盤面、81升分の駒。
  Piece board[SQ_NB];

  // 手駒
  Hand hand[COLOR_NB];

  // 手番
  Color sideToMove;

  // 玉の位置
  Square kingSquare[COLOR_NB];

  // 初期局面からの手数(初期局面 == 1)
  int gamePly;

  // 探索ノード数 ≒do_move()の呼び出し回数。
  int64_t nodes;

  // 現局面に対応するStateInfoのポインタ。
  // do_move()で次の局面に進むときは次の局面のStateInfoへの参照をdo_move()の引数として渡される。
  //   このとき、undo_move()で戻れるようにStateInfo::previousに前のstの値を設定しておく。
  // undo_move()で前の局面に戻るときはStateInfo::previousから辿って戻る。
  StateInfo* st;

  // 評価関数で用いる駒のリスト
  Eval::EvalList evalList;

  // -- 利き
#ifdef LONG_EFFECT_LIBRARY
  // 利きの初期化
  void set_effect();

  // 各升の利きの数
  LongEffect::EffectNumBoard board_effect[COLOR_NB];

  // 長い利き。
  LongEffect::LongEffectBoard long_effect[COLOR_NB];
#endif

  // --- 

  // StateInfoの初期化(初期化するときに内部的に用いる)
  void set_state(StateInfo* si) const;

};

// PieceからPieceTypeBitboardへの変換テーブル
const PieceTypeBitboard piece2ptb[PIECE_WHITE] = {
  PIECE_TYPE_BITBOARD_NB /*NO_PIECE*/,PIECE_TYPE_BITBOARD_PAWN /*歩*/,PIECE_TYPE_BITBOARD_LANCE /*香*/,PIECE_TYPE_BITBOARD_KNIGHT /*桂*/,
  PIECE_TYPE_BITBOARD_SILVER /*銀*/,PIECE_TYPE_BITBOARD_BISHOP /*角*/,PIECE_TYPE_BITBOARD_ROOK /*飛*/,PIECE_TYPE_BITBOARD_GOLD /*金*/,
  PIECE_TYPE_BITBOARD_HDK /*玉*/, PIECE_TYPE_BITBOARD_GOLD /*歩成*/ , PIECE_TYPE_BITBOARD_GOLD /*香成*/,PIECE_TYPE_BITBOARD_GOLD/*桂成*/,
  PIECE_TYPE_BITBOARD_GOLD /*銀成*/,PIECE_TYPE_BITBOARD_BISHOP/*馬*/,PIECE_TYPE_BITBOARD_ROOK/*龍*/ ,PIECE_TYPE_BITBOARD_NB/*金成*/ };

inline void Position::xor_piece(Piece pc, Square sq)
{
  Color c = color_of(pc);
  const Bitboard q = Bitboard(sq);
  // 先手・後手の駒のある場所を示すoccupied bitboardの更新
  occupied[c] ^= q;
  // 先手 or 後手の駒のある場所を示すoccupied bitboardの更新
  occupied[COLOR_ALL] ^= sq;

  // 駒別のBitboardの更新
  Piece pt = type_of(pc);
  piece_bb[piece2ptb[pt]][c] ^= sq;

  // 馬、龍は、piece_bbのPIECE_TYPE_BITBOARD_BISHOP(ROOK)とPIECE_TYPE_BITBOARD_HDKの両方のbitboardにまたがって存在するので
  // PIECE_TYPE_BITBOARD_HDKのほうも更新する必要がある。
  if (pt >= HORSE)
    piece_bb[PIECE_TYPE_BITBOARD_HDK][c] ^= sq;
}

// 駒を配置して、内部的に保持しているBitboardも更新する。
inline void Position::put_piece(Square sq, Piece pc,PieceNo piece_no)
{
  ASSERT_LV2(board[sq] == NO_PIECE);
  board[sq] = pc;
  xor_piece(pc, sq);

  // 駒番号をセットしておく必要がある。
  ASSERT_LV3(is_ok(piece_no));
  
  // evalListのほうを更新しないといけない
  evalList.put_piece(piece_no,sq,pc); // sqの升にpcの駒を配置する

  // 王なら、その升を記憶しておく。
  // (王の升はBitboardなどをみればわかるが、頻繁にアクセスするのでcacheしている。)
  if (type_of(pc) == KING)
    kingSquare[color_of(pc)] = sq;
}

// 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
inline void Position::remove_piece(Square sq)
{
  Piece pc = board[sq];
  ASSERT_LV3(pc != NO_PIECE);
  board[sq] = NO_PIECE;
  xor_piece(pc, sq);
}

inline bool is_ok(Position& pos) { return pos.pos_is_ok(); }

// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os, const Position& pos);

// depthに応じたZobrist Hashを得る。depthを含めてhash keyを求めたいときに用いる。
HASH_KEY DepthHash(int depth);

#endif // of #ifndef _SHOGI_H_
