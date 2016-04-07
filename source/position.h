#ifndef _POSITION_H_
#define _POSITION_H_

#include "shogi.h"
#include "bitboard.h"
#include "evaluate.h"
#include "extra/key128.h"
#include "extra/long_effect.h"
struct Thread;

// --------------------
//     局面の定数
// --------------------

// 平手の開始局面
extern std::string SFEN_HIRATE;

// --------------------
//     局面の情報
// --------------------

enum CheckInfoUpdate
{
  CHECK_INFO_UPDATE_NONE,           // 何もupdateされていない状態
  CHECK_INFO_UPDATE_ALL,            // 以下の2つともupdate
  CHECK_INFO_UPDATE_PINNED,         // pinnedの情報だけupdate
  CHECK_INFO_UPDATE_WITHOUT_PINNED, // pinned以外の情報だけupdate
};

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
  template <CheckInfoUpdate>
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

#ifdef USE_EVAL_DIFF
// 評価値の差分計算の管理用
// 前の局面から移動した駒番号を管理するための構造体
struct DirtyPiece
{
  // dirtyになった個数。null moveだと0ということもありうる。
  int dirty_num;

  // dirtyになった駒番号
  PieceNo pieceNo[2];

  // その駒番号の駒が何から何に変わったのか
  Eval::ExtBonaPiece piecePrevious[2];
  Eval::ExtBonaPiece pieceNow[2];
};
#endif

// StateInfoは、undo_move()で局面を戻すときに情報を元の状態に戻すのが面倒なものを詰め込んでおくための構造体。
// do_move()のときは、ブロックコピーで済むのでそこそこ高速。
struct StateInfo {

  // ---- ここから下のやつは do_move()のときにコピーされる

  // 遡り可能な手数(previousポインタを用いて局面を遡るときに用いる)
  int pliesFromNull;

  // この手番側の連続王手は何手前からやっているのか(連続王手の千日手の検出のときに必要)
  int continuousCheck[COLOR_NB];

  // ---- ここから下のやつは do_move()のときにコピーされない
  // ※　ただし、do_null_move()のときは丸ごとコピーされる。

  // 現局面で手番側に対して王手をしている駒のbitboard。Position::do_move()で更新される。
  Bitboard checkersBB;

  // 王手になる駒等の情報
  CheckInfo checkInfo;

  // この局面のハッシュキー
  // ※　次の局面にdo_move()で進むときに最終的な値が設定される
  // board_key()は盤面のhash。hand_key()は手駒のhash。それぞれ加算したのがkey() 盤面のhash。
  // board_key()のほうは、手番も込み。
  // exclusion_key()は、singular extensionのために現在のkey()に一定の値を足したものを返す。
  Key key()                     const { return long_key(); }
  Key board_key()               const { return board_long_key(); }
  Key hand_key()                const { return hand_long_key(); }
  Key exclusion_key()           const { return exclusion_long_key(); }

  // HASH_KEY_BITSが128のときはKey128が返るhash key,256のときはKey256
  HASH_KEY long_key()           const { return board_key_ + hand_key_; }
  HASH_KEY board_long_key()     const { return board_key_; }
  HASH_KEY hand_long_key()      const { return hand_key_; }
  HASH_KEY exclusion_long_key() const;
  
  // この局面における手番側の持ち駒。優等局面の判定のために必要。
  Hand hand;

  // この局面で捕獲された駒
  // ※　次の局面にdo_move()で進むときにこの値が設定される
  // 先後の区別はなし。馬とか龍など成り駒である可能性はある。
  Piece capturedType;

  friend struct Position;

  // --- evaluate

#ifndef EVAL_NO_USE
  // この局面での評価関数の駒割
  Value materialValue;
#endif

#ifdef EVAL_KPP
  // 評価値。(次の局面で評価値を差分計算するときに用いる)
  // まだ計算されていなければsumKPPの値は、VALUE_NONE
  int sumKKP;
  int sumBKPP;
  int sumWKPP;
#endif

#ifdef USE_EVAL_DIFF
  // 評価値の差分計算の管理用
  DirtyPiece dirtyPiece;
#endif

#ifdef  KEEP_LAST_MOVE
  // 直前の指し手。デバッグ時などにおいてその局面までの手順を表示出来ると便利なことがあるのでそのための機能
  Move lastMove;

  // lastMoveで移動させた駒(先後の区別なし)
  Piece lastMovedPieceType;
#endif

  // HASH_KEY_BITSで128を指定した場合はBitboardにHashKeyが入っている。
  HASH_KEY board_key_;
  HASH_KEY hand_key_;

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

  // この局面クラスを用いて探索しているスレッドを返す。 
  Thread* this_thread() const { return thisThread; }
  // この局面クラスを用いて探索しているスレッドを設定する。(threads.cppのなかで設定してある。)
  void set_this_thread(Thread*th) { thisThread = th; }

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

  // この指し手によって移動させる駒を返す
  // 後手の駒打ちは後手の駒が返る。
  Piece moved_piece(Move m) const { return is_drop(m) ? (move_dropped_piece(m) + (sideToMove==WHITE ? PIECE_WHITE : NO_PIECE)) : piece_on(move_from(m)); }

  // moved_pieceの拡張版。駒打ちのときは、打ち駒(+32)を加算した駒種を返す。
  // historyなどでUSE_DROPBIT_IN_STATSを有効にするときに用いる。
  Piece moved_piece_ex(Move m) const {
    return is_drop(m)
      ? Piece((move_dropped_piece(m) + (sideToMove == WHITE ? PIECE_WHITE : NO_PIECE)) + 32)
      : piece_on(move_from(m));
  }

  // 連続王手の千日手等で引き分けかどうかを返す
  RepetitionState is_repetition(const int repPly = 16) const;

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
  // kingSqの地点からは玉を取り除いての利きの判定を行なう。
#ifndef LONG_EFFECT_LIBRARY
  bool effected_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }
  bool effected_to(Color c, Square sq , Square kingSq) const { return attackers_to(c, sq, pieces()^kingSq); }
#else 
  bool effected_to(Color c, Square sq) const { return board_effect[c].effect(sq) != 0; }
  bool effected_to(Color c, Square sq, Square kingSq) const {
    return board_effect[c].effect(sq) != 0 ||
      ((long_effect.directions_of(c,kingSq) & Effect8::directions_of(kingSq, sq)) != 0); // 影の利きがある
  }
#endif

  // --- 局面を進める/戻す

  // 指し手で盤面を1手進める
  // m = 指し手。mとして非合法手を渡してはならない。
  // info = StateInfo。局面を進めるときに捕獲した駒などを保存しておくためのバッファ。
  // このバッファはこのdo_move()の呼び出し元の責任において確保されている必要がある。
  // givesCheck = mの指し手によって王手になるかどうか。
  // この呼出までにst.checkInfo.update(pos)が呼び出されている必要がある。
  void do_move(Move m, StateInfo& st, bool givesCheck);

  // do_move()の4パラメーター版のほうを呼び出すにはgivesCheckも渡さないといけないが、
  // mで王手になるかどうかがわからないときはこちらの関数を用いる。都度CheckInfoのコンストラクタが呼び出されるので遅い。探索中には使わないこと。
  void do_move(Move m, StateInfo& st) { check_info_update(); do_move(m, st, gives_check(m)); }

  // 指し手で盤面を1手戻す
  void undo_move(Move m);

  // null move用のdo_move()
  void do_null_move(StateInfo& st);
  // null move用のundo_move()
  void undo_null_move();

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
      return true;
    else
    {
      Color us = sideToMove;
      Square from = move_from(m);

      // もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
      return (type_of(piece_on(from)) == KING) ? !effected_to(~us, move_to(m), from) :
        // 玉以外の駒であれば、その駒を動かして自玉が素抜きに合わなければ合法。
        !discovered(from, move_to(m), king_square(us), st->checkInfo.pinned);
    }
  }

  // mがpseudo_legalな指し手であるかを判定する。
  // ※　pseudo_legalとは、擬似合法手(自殺手が含まれていて良い)
  // 置換表の指し手でdo_move()して良いのかの事前判定のために使われる。
  // 指し手生成ルーチンのテストなどにも使える。(指し手生成ルーチンはpseudo_legalな指し手を返すはずなので)
  // killerのような兄弟局面の指し手がこの局面において合法かどうかにも使う。
  // ※　置換表の検査だが、pseudo_legal()で擬似合法手かどうかを判定したあとlegal()で自殺手でないことを
  // 確認しなくてはならない。このためpseudo_legal()とlegal()とで重複する自殺手チェックはしていない。
  // 注意 : 事前にcheck_info_update()が呼び出されていること。
  bool pseudo_legal(const Move m) const { return pseudo_legal_s<true>(m); }

  // All == false        : 歩や大駒の不成に対してはfalseを返すpseudo_legal()
  template <bool All> bool pseudo_legal_s(const Move m) const;

  // toの地点に歩を打ったときに打ち歩詰めにならないならtrue。
  // 歩をtoに打つことと、二歩でないこと、toの前に敵玉がいることまでは確定しているものとする。
  // 二歩の判定もしたいなら、legal_pawn_drop()のほうを使ったほうがいい。
  bool legal_drop(const Square to) const;

  // 二歩でなく、かつ打ち歩詰めでないならtrueを返す。
  bool legal_pawn_drop(const Color us, const Square to) const
  {
    return !((pieces(us, PAWN) & FILE_BB[file_of(to)])                             // 二歩
      || ((pawnEffect(us, to) == Bitboard(king_square(~us)) && !legal_drop(to)))); // 打ち歩詰め
  }

  // --- StateInfo

  // 現在の局面に対応するStateInfoを返す。
  // たとえば、state()->capturedTypeであれば、前局面で捕獲された駒が格納されている。
  StateInfo* state() const { return st; }

  // gives_check()やlegal()、do_move()を呼び出すときは事前にこれを呼び出しておくこと。
  void check_info_update() { st->checkInfo.update<CHECK_INFO_UPDATE_ALL>(*this); }

  // check_infoのupdateを段階的に行ないたいときに用いる。
  // 引数としてはすでに初期化が終わっているupdateを指定する。
  void check_info_update(CheckInfoUpdate c)
  {
    switch (c)
    {
      // 何も終わっていないので丸ごと初期化
    case CHECK_INFO_UPDATE_NONE: check_info_update(); break;

      // 全部終わっている
    case CHECK_INFO_UPDATE_ALL: break; 

      // pinnedだけ終わっているのでそれ以外を初期化
    case CHECK_INFO_UPDATE_PINNED: check_info_update_without_pinned(); break;

    default: UNREACHABLE;
    }
  }

  // CheckInfoのpinnedだけ更新したいとき(mate1ply()で必要なので)
  void check_info_update_pinned() { st->checkInfo.update<CHECK_INFO_UPDATE_PINNED>(*this); }
  // CheckInfoのpinned以外を更新したいとき(mate1ply()のあとに初期化するときに必要なので)
  void check_info_update_without_pinned() { st->checkInfo.update<CHECK_INFO_UPDATE_WITHOUT_PINNED>(*this); }

  // --- Evaluation

#ifndef EVAL_NO_USE
  // 評価関数で使うための、どの駒番号の駒がどこにあるかなどの情報。
  const Eval::EvalList* eval_list() const { return &evalList; }
#endif

#ifdef  USE_SEE
  // 指し手mの(Static Exchange Evaluation : 静的取り合い評価)の値を返す。
  Value see(Move m) const;

  // SEEの符号だけが欲しい場合はこちらのほうがsee()より速い。
  Value see_sign(Move m) const;
#endif

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
      && !is_aligned(ourKing, from, to); // 3)
  }

  // 現局面で指し手がないかをテストする。指し手生成ルーチンを用いるので速くない。探索中には使わないこと。
  bool is_mated() const;

  // 直前の指し手によって捕獲した駒。
  Piece captured_piece_type() const { return st->capturedType; }

  // 捕獲する指し手か、成りの指し手であるかを返す。
  bool capture_or_promotion(Move m) const { return (m & MOVE_PROMOTE) || capture(m); }

  // 捕獲する指し手であるか。
  bool capture(Move m) const { return !is_drop(m) && piece_on(move_to(m)) != NO_PIECE; }

  // 捕獲する指し手もしくは歩を成る指し手であるか。
  bool capture_or_pawn_promotion(Move m) const {
    return capture(m) ||
      (type_of(piece_on(move_from(m))) == PAWN && (m & MOVE_PROMOTE));
  }

  // --- 超高速1手詰め判定
#if defined(MATE_1PLY) && defined(LONG_EFFECT_LIBRARY)
  // 現局面で1手詰めであるかを判定する。1手詰めであればその指し手を返す。
  // ただし1手詰めであれば確実に詰ませられるわけではなく、簡単に判定できそうな近接王手による
  // 1手詰めのみを判定する。(要するに判定に漏れがある。)
  // 先行して、CheckInfo.pinnedを更新しておく必要がある。
  // →　check_info_update_pinned()を利用するのが吉。
  Move mate1ply() const;

  // ↑の先後別のバージョン。(内部的に用いる)
  template <Color Us> Move mate1ply_impl() const;
#endif

  // 入玉時の宣言勝ち
#ifdef USE_ENTERING_KING_WIN
  // Search::Limits.enteringKingRuleに基いて、宣言勝ちを行なう。
  // 条件を満たしているとき、MOVE_WINや、玉を移動する指し手(トライルール時)が返る。さもなくば、MOVE_NONEが返る。
  // mate1ply()から内部的に呼び出す。(そうするとついでに処理出来て良い)
  Move DeclarationWin() const;
#endif

  // -- 利き
#ifdef LONG_EFFECT_LIBRARY

  // 各升の利きの数
  LongEffect::ByteBoard board_effect[COLOR_NB];

  // 長い利き(これは先後共用)
  LongEffect::WordBoard long_effect;
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

  // do_move()の先後分けたもの。内部的に呼び出される。
  template <Color Us> void do_move_impl(Move m, StateInfo& st, bool givesCheck);

  // undo_move()の先後分けたもの。内部的に呼び出される。
  template <Color Us> void undo_move_impl(Move m);

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

#ifndef EVAL_NO_USE
  // --- 盤面を更新するときにEvalListの更新のために必要なヘルパー関数

  // c側の手駒ptの最後の1枚のBonaPiece番号を返す
  Eval::BonaPiece bona_piece_of(Color c,Piece pt) const {
    // c側の手駒ptの枚数
    int ct = hand_count(hand[c], pt);
    ASSERT_LV3(ct > 0);
    return (Eval::BonaPiece)(Eval::kpp_hand_index[c][pt].fb + ct - 1);
  }

  // c側の手駒ptの(最後の1枚の)PieceNoを返す
  PieceNo piece_no_of(Color c, Piece pt) const { return evalList.piece_no_of(bona_piece_of(c,pt));}

  // 盤上のpcの駒のPieceNoを返す
  PieceNo piece_no_of(Piece pc, Square sq) const { return evalList.piece_no_of((Eval::BonaPiece)(Eval::kpp_board_index[pc].fb + sq)); }
#else
  // 駒番号を使わないとき用のダミー
  PieceNo piece_no_of(Color c, Piece pt) const { return PIECE_NO_ZERO; }
  PieceNo piece_no_of(Piece pc, Square sq) const { return PIECE_NO_ZERO; }
#endif
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

  // この局面クラスを用いて探索しているスレッド
  Thread* thisThread;

  // 探索ノード数 ≒do_move()の呼び出し回数。
  int64_t nodes;

  // 現局面に対応するStateInfoのポインタ。
  // do_move()で次の局面に進むときは次の局面のStateInfoへの参照をdo_move()の引数として渡される。
  //   このとき、undo_move()で戻れるようにStateInfo::previousに前のstの値を設定しておく。
  // undo_move()で前の局面に戻るときはStateInfo::previousから辿って戻る。
  StateInfo* st;

#ifndef EVAL_NO_USE
  // 評価関数で用いる駒のリスト
  Eval::EvalList evalList;
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
  
#ifndef EVAL_NO_USE
  // evalListのほうを更新しないといけない
  evalList.put_piece(piece_no,sq,pc); // sqの升にpcの駒を配置する
#endif

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
