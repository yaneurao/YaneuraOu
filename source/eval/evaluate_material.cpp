#include "../shogi.h"
#include "../position.h"
#include "../evaluate.h"

namespace Eval
{
#ifndef EVAL_NO_USE
  // 何らかの評価関数を用いる以上、駒割りの計算は必須。
  // すなわち、EVAL_NO_USE以外のときはこの関数が必要。

  // 駒割りの計算
  // 手番側から見た評価値
  Value material(const Position& pos)
  {
    int v = VALUE_ZERO;

    for (auto i : SQ)
      v = v + PieceValue[pos.piece_on(i)];

    // 手駒も足しておく
    for (auto c : COLOR)
      for (auto pc = PAWN; pc < PIECE_HAND_NB; ++pc)
        v += (c == BLACK ? 1 : -1) * Value(hand_count(pos.hand_of(c), pc) * PieceValue[pc]);

    return (Value)v;
  }
#endif

#ifdef EVAL_MATERIAL
  // 駒得のみの評価関数のとき。

  void load_eval() {}
  void print_eval_stat(Position& pos) {}
  Value evaluate(const Position& pos) {
    auto score = pos.state()->materialValue;
    ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
    return pos.side_to_move() == BLACK ? score : -score;
  }
  Value compute_eval(const Position& pos) { return material(pos); }

#endif
}
