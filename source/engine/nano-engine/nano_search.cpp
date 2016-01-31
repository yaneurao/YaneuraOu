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
    MovePicker(const Position& pos_,Move ttMove) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、すべての指し手(NON_EVASIONS)で指し手を生成する。
      if (pos_.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
        endMoves = generateMoves<NON_EVASIONS>(pos, currentMoves);

      // 置換表の指し手が、この生成された集合のなかにあるなら、その先頭の指し手に置換表の指し手が来るようにしておく。
      if (ttMove != MOVE_NONE)
      {
        auto p = currentMoves;
        while (p != endMoves)
        {
          if (*p == ttMove)
          {
            swap(*p, *currentMoves);
            break;
          }
          ++p;
        }
      }
    }

    // 静止探索から呼び出される時用。
    MovePicker(const Position& pos_, Depth depth) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、取り合いの指し手(CAPTURES_PRO_PLUS)で指し手を生成して
      // そのなかで直前でcaptureされた駒以上の駒を捕獲する指し手のみを生成する。
      if (pos_.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
      {
        endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, currentMoves);
        Piece lastCaptured = pos.state()->capturedType;
        int value = Eval::PieceValue[lastCaptured];
        while (currentMoves != endMoves)
        {
          Piece captured = pos.piece_on(move_to(*currentMoves)); // この指し手で捕獲される駒
          int v = Eval::PieceValue[captured];
          if (value > v) // 直前での捕獲された駒の価値を下回るのでこの指し手は削除
          {
            *currentMoves = *(--endMoves);
          } else {
            currentMoves++;
          }
        }
      }
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
    // 静止探索では4手以上は延長しない。
    if (depth < -4 * ONE_PLY)
      return Eval::eval(pos);

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

  Value search(Position& pos, Value alpha, Value beta, Depth depth)
  {
    // -----------------------
    // 残り深さがないなら静止探索へ
    // -----------------------

    ASSERT_LV3(alpha < beta);

    // 残り探索深さがなければ静止探索を呼び出して評価値を返す。
    if (depth < ONE_PLY)
      return qsearch(pos, alpha, beta, depth);

    // -----------------------
    //   置換表のprobe
    // -----------------------

    auto key = pos.state()->key();

    bool ttHit;    // 置換表がhitしたか
    TTEntry* tte = TT.probe(key, ttHit);

    // 置換表上のスコア
    // 置換表にhitしなければVALUE_NONE
    Value ttValue = ttHit ? value_from_tt(tte->value(), pos.game_ply()) : VALUE_NONE;

    // 置換表の指し手
    // 置換表にhitしなければMOVE_NONE
    Move ttMove = ttHit ? tte->move() : MOVE_NONE;

    // 置換表の値によるbeta cut
    
    if (ttHit                   // 置換表の指し手がhitして
      && tte->depth() >= depth   // 置換表に登録されている探索深さのほうが深くて
      && ttValue != VALUE_NONE   // (他スレッドからTTEntryがこの瞬間に破壊された可能性が..)
      && (ttValue >= beta && tte->bound() & BOUND_LOWER) // ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値。
      )
    {
      return ttValue;
    }

    // -----------------------
    // 1手ずつ指し手を試す
    // -----------------------

    // 指し手して一手ずつ返す
    MovePicker mp(pos,ttMove);

    Value score;
    Move m;

    StateInfo si;
    pos.check_info_update();

    // この局面でdo_move()された合法手の数
    int moveCount = 0;
    Move bestMove = MOVE_NONE;

    while (m = mp.nextMove())
    {
      if (!pos.legal(m))
        continue;

      pos.do_move(m, si, pos.gives_check(m));
      score = -YaneuraOuNano::search(pos, -beta, -alpha, depth - ONE_PLY);
      pos.undo_move(m);

      // 停止シグナルが来たら置換表を汚さずに終了。
      if (Signals.stop)
        return VALUE_NONE;

      ++moveCount;

      // αを超えたならαを更新
      if (score > alpha)
      {
        alpha = score;
        bestMove = m;

        // αがβを上回ったらbeta cut
        if (alpha >= beta)
          break;
      }
    }

    // 合法手がない == 詰まされている ので、rootの局面からの手数で詰まされたという評価値を返す。
    if (moveCount == 0)
      alpha = mated_in(pos.game_ply());

    // 置換表に保存する

    tte->save(key, value_to_tt(alpha, pos.game_ply()),
      alpha >= beta ? BOUND_LOWER : BOUND_EXACT,
      // betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
      // すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
      // さもなくば、枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
      depth, bestMove, VALUE_NONE,TT.generation());

    return alpha;
  }

}

using namespace YaneuraOuNano;

/*
// root nodeでalpha値を更新するごとに
// GUIに対して歩1枚の価値を100とする評価値と、現在のベストの指し手を読み筋(pv)として出力する。
sync_cout << "info score cp " << int(score) * 100 / Eval::PawnValue << " pv " << m << sync_endl;
*/

// --- 以下に好きなように探索のプログラムを書くべし。
template <bool RootNode>
Value search(Position& pos, Value alpha, Value beta, Depth depth);


// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init(){}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear() { TT.clear(); }

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::think() {

  // 合法手がないならここで投了。
  if (rootMoves.size() == 0)
  {
    sync_cout << "bestmove " << MOVE_RESIGN << sync_endl;
    return;
  }

  // TTEntryの世代を進める。
  TT.new_search();

  int rootDepth = 0;
  Value alpha,beta;
  StateInfo si;
  auto& pos = rootPos;

  // 今回に用いる思考時間 = 残り時間の1/60 + 秒読み時間
  auto us = pos.side_to_move();
  auto availableTime = Limits.time[us] / 60 + Limits.byoyomi[us];
  auto endTime = Limits.startTime + availableTime;

  // タイマースレッドを起こして、終了時間を監視させる。
  auto timerThread = new std::thread([&] {
    while (now() < endTime && !Signals.stop)
      sleep(10);
    Signals.stop = true;
  });

  // --- 反復深化のループ

  while (++rootDepth < MAX_PLY && !Signals.stop && (!Limits.depth || rootDepth <= Limits.depth))
  {
    // 本当はもっと探索窓を絞ったほうが効率がいいのだが…。
    alpha = -VALUE_INFINITE;
    beta = VALUE_INFINITE;

    // それぞれの指し手について調べていく。これは合法手であることは保証されているので合法手チェックは不要。
    for (auto& m : rootMoves)
    {
      auto move = m.pv[0];
      pos.do_move(move,si,pos.gives_check(move));
      m.score = -YaneuraOuNano::search(rootPos, -beta, -alpha , rootDepth * ONE_PLY);
      pos.undo_move(move);

      // 停止シグナルが来たら終了。
      if (Signals.stop)
        break;

      // alpha値を更新
      if (alpha < m.score)
        alpha = m.score;
    }

    // それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
    std::stable_sort(rootMoves.begin(), rootMoves.end());

    // 読み筋を出力しておく。
    sync_cout << "info string depth " << rootDepth << " pv ";
    for (auto m : rootMoves.at(0).pv)
      cout << m;
    cout << " score cp " << rootMoves.at(0).score*100/Eval::PawnValue << sync_endl;
  }
  
  Move bestMove = rootMoves.at(0).pv[0];
  sync_cout << "bestmove " << bestMove << sync_endl;

  Signals.stop = true;
  timerThread->join();
  delete timerThread;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search(){}

#endif // YANEURAOU_NANO_ENGINE
