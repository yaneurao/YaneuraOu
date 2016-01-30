#include "../../shogi.h"

#ifdef YANEURAOU_NANO_ENGINE

#include <sstream>
#include <iostream>

#include "../../position.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"

using namespace std;
using namespace Search;


// --- やねうら王nano探索部

namespace YaneuraOuNano
{
  // nano用のMovePicker
  struct MovePicker
  {
    // 通常探索から呼び出されるとき用。
    MovePicker(const Position& pos_) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、すべての指し手(NON_EVASIONS)で指し手を生成する。
      if (pos_.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
        endMoves = generateMoves<NON_EVASIONS>(pos, currentMoves);
    }

    // 静止探索から呼び出される時用。
    MovePicker(const Position& pos_, Depth depth) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、取り合いの指し手(CAPTURES_PRO_PLUS)で指し手を生成して
      // そのなかでSSE > 0の指し手のみ返す。
      if (pos_.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
        endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, currentMoves);
    }

    // 次の指し手をひとつ返す
    // 指し手が尽きればMOVE_NONEが返る。
    Move nextMove() {
      if (currentMoves == endMoves)
        return MOVE_NONE;
      return *currentMoves++;
    }

  private:
    const Position& pos;

    ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;
  };

  // 静止探索
  Value qsearch(Position& pos, Value alpha, Value beta, Depth depth)
  {
    // 取り合いの指し手だけ生成する
    MovePicker mp(pos,depth);
    Value score;
    Move m;

    StateInfo si;
    pos.check_info_update();

    // この局面でdo_move()された合法手の数
    int moveCount = 0;

    while (m = mp.nextMove())
    {
      if (!pos.legal(m))
        continue;

      pos.do_move(m, si, pos.gives_check(m));
      score = -YaneuraOuNano::qsearch(pos, -beta, -alpha, depth - ONE_PLY);
      pos.undo_move(m);

      ++moveCount;

      if (score > alpha)
      {
        alpha = score;
        if (alpha >= beta)
          return alpha; // beta cut
      }
    }

    if (moveCount == 0)
    {
      // 王手がかかっているなら回避手をすべて生成しているはずで、つまりここで詰んでいたということだから
      // 詰みの評価値を返す。
      if (pos.in_check())
        return mated_in(pos.game_ply());

      // captureの指し手が尽きたということだから、評価関数を呼び出して評価値を返す。
      return Eval::eval(pos);
    }

    return alpha;
  }

  template <bool RootNode>
  Value search(Position& pos, Value alpha, Value beta, Depth depth)
  {
    ASSERT_LV3(alpha < beta);

    // 残り探索深さがなければ静止探索を呼び出して評価値を返す。
    if (depth < ONE_PLY)
      return qsearch(pos, alpha, beta, depth);

    // 指し手して一手ずつ返す
    MovePicker mp(pos);

    Value score;
    Move m;

    StateInfo si;
    pos.check_info_update();

    // この局面でdo_move()された合法手の数
    int moveCount = 0;

    while (m = mp.nextMove())
    {
      if (!pos.legal(m))
        continue;

      pos.do_move(m, si, pos.gives_check(m));
      score = -YaneuraOuNano::search<false>(pos, -beta, -alpha, depth - ONE_PLY);
      pos.undo_move(m);

      ++moveCount;

      // αを超えたならαを更新
      if (score > alpha)
      {
        alpha = score;
        if (RootNode)
        {
          rootBestMove = m;

          // root nodeでalpha値を更新するごとに
          // GUIに対して歩1枚の価値を100とする評価値と、現在のベストの指し手を読み筋(pv)として出力する。
          sync_cout << "info score cp " << int(score) * 100 / Eval::PawnValue << " pv " << m << sync_endl;
        }
        // αがβを上回ったらbeta cut
        if (alpha >= beta)
          return alpha;
      }
    }

    // 合法手がない == 詰まされている ので、rootの局面からの手数で詰まされたという評価値を返す。
    if (moveCount == 0)
      return mated_in(pos.game_ply());

    return alpha;
  }

}

using namespace YaneuraOuNano;


// --- 以下に好きなように探索のプログラムを書くべし。
template <bool RootNode>
Value search(Position& pos, Value alpha, Value beta, Depth depth);


// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init(){}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear(){}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

Move rootBestMove;
void MainThread::think() {
  YaneuraOuNano::search<true>(rootPos,-VALUE_INFINITE,VALUE_INFINITE,3*ONE_PLY);
  sync_cout << "bestmove " << rootBestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search(){}

#endif // YANEURAOU_NANO_ENGINE
