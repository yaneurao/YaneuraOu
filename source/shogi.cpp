#include <sstream>
#include <iostream>

#include "shogi.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"

// ----------------------------------------
//    tables
// ----------------------------------------

File SquareToFile[SQ_NB] = {
  FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1, FILE_1,
  FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2, FILE_2,
  FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3, FILE_3,
  FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4, FILE_4,
  FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5, FILE_5,
  FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6, FILE_6,
  FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7, FILE_7,
  FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8, FILE_8,
  FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9, FILE_9
};

Rank SquareToRank[SQ_NB] = {
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9,
};

std::string PieceToCharBW(" PLNSBRGK        plnsbrgk");


// ----------------------------------------
// operator<<(std::ostream& os,...)とpretty() 
// ----------------------------------------

std::string pretty(File f) { return pretty_jp ? std::string("１２３４５６７８９").substr((int32_t)f * 2, 2) : std::to_string((int32_t)f + 1); }
std::string pretty(Rank r) { return pretty_jp ? std::string("一二三四五六七八九").substr((int32_t)r * 2, 2) : std::to_string((int32_t)r + 1); }

std::string pretty(Move m)
{
  if (is_drop(m))
    return (pretty(move_to(m)) + pretty2(Piece(move_from(m))) + (pretty_jp ? "打" : "*"));
  else
    return pretty(move_from(m)) + pretty(move_to(m)) + (is_promote(m) ? (pretty_jp ? "成" : "+") : "");
}

std::string pretty(Move m, Piece movedPieceType)
{
  if (is_drop(m))
    return (pretty(move_to(m)) + pretty2(movedPieceType) + (pretty_jp ? "打" : "*"));
  else
    return pretty(move_to(m)) + pretty2(movedPieceType) + (is_promote(m) ? (pretty_jp ? "成" : "+") : "") + "["+ pretty(move_from(m))+"]";
}

std::ostream& operator<<(std::ostream& os, Color c) { os << ((c == BLACK) ? (pretty_jp ? "先手" : "BLACK") : (pretty_jp ? "後手" : "WHITE")); return os; }

std::ostream& operator<<(std::ostream& os, Piece pc)
{
  auto s = usi_piece(pc);
  if (s[1] == ' ') s.resize(1); // 手動trim
  os << s;
  return os;
}

std::ostream& operator<<(std::ostream& os, Hand hand)
{
  for (Piece pr = PAWN; pr < PIECE_HAND_NB; ++pr)
  {
    int c = hand_count(hand, pr);
    // 0枚ではないなら出力。
    if (c != 0)
    {
      // 1枚なら枚数は出力しない。2枚以上なら枚数を最初に出力
      // PRETTY_JPが指定されているときは、枚数は後ろに表示。
      const std::string cs = (c != 1) ? std::to_string(c) : "";
      std::cout << (pretty_jp ? "" : cs) << pretty(pr) << (pretty_jp ? cs : "");
    }
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, HandKind hk)
{
  for (Piece pc = PAWN; pc < PIECE_HAND_NB; ++pc)
    if (hand_exists(hk, pc))
      std::cout << pretty(pc);
  return os;
}

std::string to_usi_string(Move m)
{
  std::stringstream ss;
  if (!is_ok(m))
  {
    ss <<((m == MOVE_RESIGN) ? "resign" :
          (m == MOVE_NULL  ) ? "null" :
          (m == MOVE_NONE  ) ? "none" :
          "");
  }
  else if (is_drop(m))
  {
    ss << Piece(move_from(m));
    ss << '*';
    ss << move_to(m);
  }
  else {
    ss << move_from(m);
    ss << move_to(m);
    if (is_promote(m))
      ss << '+';
  }
  return ss.str();
}

// ----------------------------------------
// 探索用のglobalな変数
// ----------------------------------------

namespace Search {
  SignalsType Signals;
  LimitsType Limits;
  StateStackPtr SetupStates;

  void RootMove::insert_pv_in_tt(Position& pos) {

    StateInfo state[MAX_PLY], *st = state;
    bool ttHit;

    // 細かいことだがpvのtailから前方に向かって置換表に書き込んでいくほうが、
    // pvの前のほうがエントリーの価値が高いので上書きされてしまう場合にわずかに得ではある。
    // ただ、現実的にはほとんど起こりえないので気にしないことにする。
    
    for (Move m : pv)
    {
      // 銀の不成の指し手をcounter moveとして登録して、この位置に角が来ると
      // 角の不成の指し手を生成することになるからLEGALではなくLEGAL_ALLで判定しないといけない。
      ASSERT_LV3(MoveList<LEGAL_ALL>(pos).contains(m));
      TTEntry* tte = TT.probe(pos.state()->key(), ttHit);

      // 正しいエントリーは書き換えない。
      if (!ttHit || tte->move() != m)
        tte->save(pos.state()->key(), VALUE_NONE, BOUND_NONE, DEPTH_NONE,
          m, VALUE_NONE, TT.generation());

      pos.do_move(m, *st++);
    }

    for (size_t i = pv.size(); i > 0; )
      pos.undo_move(pv[--i]);
  }
}

// 引き分け時のスコア(とそのdefault値)
Value drawValueTable[REPETITION_NB][COLOR_NB] =
{
  {  VALUE_ZERO      ,  VALUE_ZERO      }, // REPETITION_NONE
  {  VALUE_MATE      ,  VALUE_MATE      }, // REPETITION_WIN
  { -VALUE_MATE      , -VALUE_MATE      }, // REPETITION_LOSE
  {  VALUE_ZERO      ,  VALUE_ZERO      }, // REPETITION_DRAW  : このスコアはUSIのoptionコマンドで変更可能
  {  VALUE_KNOWN_WIN ,  VALUE_KNOWN_WIN }, // REPETITION_SUPERIOR
  { -VALUE_KNOWN_WIN , -VALUE_KNOWN_WIN }, // REPETITION_INFERIOR
};

// ----------------------------------------
// inlineで書くとVC++2015の内部コンパイルエラーになる
// ----------------------------------------

// VC++のupdateで内部コンパイルエラーにならないように修正されたら、これをまたshogi.hに移動させる。

int hand_count(Hand hand, Piece pr) { ASSERT_LV2(PIECE_HAND_ZERO <= pr && pr < PIECE_HAND_NB); return (hand >> PIECE_BITS[pr]) & PIECE_BIT_MASK[pr]; }
int hand_exists(Hand hand, Piece pr) { ASSERT_LV2(PIECE_HAND_ZERO <= pr && pr < PIECE_HAND_NB); return hand & PIECE_BIT_MASK2[pr]; }
void add_hand(Hand &hand, Piece pr, int c) { hand = (Hand)(hand + PIECE_TO_HAND[pr] * c); }
void sub_hand(Hand &hand, Piece pr, int c) { hand = (Hand)(hand - PIECE_TO_HAND[pr] * c); }

// ----------------------------------------
//  main()
// ----------------------------------------

int main(int argc, char* argv[])
{
  // --- 全体的な初期化
  USI::init(Options);
  Bitboards::init();
  Position::init();
  Search::init();
  Threads.init();
  Eval::init(); // 簡単な初期化のみで評価関数の読み込みはisreadyに応じて行なう。

  // USIコマンドの応答部
  USI::loop(argc,argv);

  // 生成して、待機させていたスレッドの停止
  Threads.exit();

  return 0;
}
