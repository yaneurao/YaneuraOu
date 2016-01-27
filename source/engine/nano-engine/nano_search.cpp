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
    // ttMove = 置換表の指し手
    MovePicker(const Position& pos_) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、すべての指し手(NON_EVASIONS)で指し手を生成する。
      if (pos_.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
        endMoves = generateMoves<NON_EVASIONS>(pos, currentMoves);
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

  template <bool RootNode>
  Value search(Position& pos, Value alpha, Value beta, Depth depth)
  {
    if (depth < ONE_PLY)
      return Eval::eval(pos);

    // 指し手して一手ずつ返す
    MovePicker mp(pos);

    // 指し手のなかのベストなスコア
    Value score = -VALUE_INFINITE;
    Move m;

    StateInfo si;
    pos.check_info_update();

    while (m = mp.nextMove())
    {
      if (!pos.legal(m))
        continue;

      pos.do_move(m, si, pos.gives_check(m));
      Value s = -YaneuraOuNano::search<false>(pos, -beta, -alpha, depth - ONE_PLY);
      if (s > score)
      {
        score = s;
        if (RootNode)
        {
          rootBestMove = m;

          // GUIに対して歩1枚の価値を100とする評価値と、現在のベストの指し手を読み筋(pv)として出力する。
          sync_cout << "info score cp " << int(score) * 100 / Eval::PAWN_VALUE << " pv " << m << sync_endl;
        }
      }
      pos.undo_move(m);
    }

    return score;
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
  YaneuraOuNano::search<true>(rootPos,-VALUE_INFINITE,VALUE_INFINITE,4*ONE_PLY);
  sync_cout << "bestmove " << rootBestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search(){}

#endif // YANEURAOU_NANO_ENGINE
