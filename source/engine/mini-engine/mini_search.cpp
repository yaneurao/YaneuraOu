#include "../../shogi.h"

#ifdef YANEURAOU_MINI_ENGINE

#include <sstream>
#include <iostream>

#include "../../position.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"

using namespace std;
using namespace Search;

// --- 以下に好きなように探索のプログラムを書くべし。

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
//  TT.clear();
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
  for (auto th : Threads.slaves) th->search_start();

  search();

  Signals.stop = true;

  for (auto th : Threads.slaves) th->join();

  /*
  auto& pos = rootPos;

  Move bestMove = MOVE_RESIGN;
  Value maxValue = -VALUE_INFINITE;
  StateInfo si;
  for (auto m : MoveList<LEGAL_ALL>(pos))
  {
    // 合法手mで1手進めて、そのときの評価関数を呼び出して、その値が一番良い指し手を選ぶ。
    // (1手進めた局面は後手番なので一番小さなものが先手から見たベスト)
    pos.do_move(m,si);
    auto value = -Eval::eval(pos);

    // toの地点に敵の駒が利いてたら、この駒を損してしまう(ことにする)
    if (pos.effected_to(pos.side_to_move(), move_to(m)))
    {
      // 移動させた駒
      auto pc = pos.piece_on(move_to(m));
      value -= (Value)Eval::PieceValue[type_of(pc)]*2;
    }

    pos.undo_move(m);
    if (value > maxValue)
    {
      maxValue = value;
      bestMove = m;
    }
  }
  sync_cout << "bestmove " << bestMove << sync_endl;
  */
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search()
{
//  sync_cout << "thread id = " << thread_id() << " is_main() = " << is_main() << sync_endl;
}

Value search(Position& pos, Value alpha, Value beta, Depth depth)
{
  return VALUE_ZERO;
}
#endif

#if 0
// 残り深さdepthで探索して返す
Value search(Position& pos, Value alpha, Value beta,Depth depth)
{
  if (depth < ONE_PLY)
    return evaluate(pos);

  StateInfo st;

  // 指し手して一手ずつ返す
  MovePicker mp(pos);

  // 指し手のなかのベストなスコア
  Value score = -VALUE_INFINITE;
  Move m;
  while (m = mp.nextMove())
  {
    st.checkInfo.update(pos);
    pos.do_move(m,st,pos.gives_check(m));
    Value s = -search(pos, -beta/*todo*/ , -alpha , depth - ONE_PLY);
    if (s > score)
      score = s;
    pos.undo_move(m);
  }

  return score;
}
#endif // YANEURAOU_MINI_ENGINE
