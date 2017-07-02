#include "../../shogi.h"
#ifdef HELP_MATE_ENGINE

#include "../../extra/all.h"
#include "help-mate-search.h" 

using namespace std;
using namespace Search;

// --- 協力詰め探索

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
  // Hash上限。32bitモードなら2GB、64bitモードなら1024GB
  const int MaxHashMB = Is64Bit ? 1024 * 1024 : 2048;

  // 協力詰めで確保する置換表
  o["CM_Hash"] << Option(16, 1, MaxHashMB, [](auto&o) { HelpMate::TT.resize(o); });
}

namespace HelpMate
{
  // 協力詰め用のMovePicker
  struct MovePicker
  {
    // ttMove = 置換表の指し手
    MovePicker(const Position& pos_, Move ttMove) : pos(pos_)
    {
      if (ttMove == MOVE_NONE)
      {
        // 協力詰めであれば段階的に指し手を生成する必要はない。
        // 先手ならば王手の指し手(CHECKS)、後手ならば回避手(EVASIONS)を生成。
        endMoves = (pos.side_to_move() == BLACK) ? generateMoves<CHECKS_ALL>(pos, currentMoves)
          : generateMoves<EVASIONS_ALL>(pos, currentMoves);
      } else {
        // 置換表に載っていた指し手が一つしかないのはone replyなのでこれで指し手生成をはしょれる。
        *currentMoves = ttMove;
        endMoves++;
      }
    }

    // 次の指し手をひとつ返す
    // 指し手が尽きればMOVE_NONEが返る。
    Move next_move() {
      if (currentMoves == endMoves)
        return MOVE_NONE;
      return *currentMoves++;
    }

  private:
    const Position& pos;

    ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;

  };

  TranspositionTable TT;

  // 現在、詰まないとわかっている探索深さ
  std::atomic<uint32_t> search_depth;

  // 詰みが見つかったか
  std::atomic_bool mate_found;

  // このスレッドの反復深化の深さ
  thread_local uint32_t id_depth_thread = 0;

  // 協力詰め
  // depth = 残り探索深さ
  // no_mate_depth = この局面は、この深さの残り探索深さがあっても詰まない(あるいは詰みを発見して出力済み)
  void search(Position& pos, uint32_t depth, int& no_mate_depth)
  {
    // 強制停止
    if (Signals.stop || mate_found)
    {
      no_mate_depth = MAX_PLY;
      return;
    }

    Key128 key = pos.state()->long_key();
    Move tt_move;
    // 置換表がヒットするか
    if (TT.probe(key, depth, tt_move))
    {
      no_mate_depth = depth; // foundのときにdepthはTTEntry.depth()で書き換わっている。
      return;
      // このnodeに関しては現在の残り探索深さ以上の深さにおいて
      //不詰めが証明されているのでもう帰ってよい。(枝刈り)
    }

    StateInfo si;

    MovePicker mp(pos, tt_move);
    Move m;

    int replyCount = 0; // 確定局面以外の応手の数
    Move oneReply = MOVE_NONE;

    no_mate_depth = MAX_PLY; // 有効な指し手が一つもなければこのnodeはいくらdepthがあろうと詰まない。

    while ((m = mp.next_move()) && !Signals.stop && !mate_found)
    {
      if (!pos.legal(m))
        continue;

      pos.do_move(m, si, pos.gives_check(m));

      if (pos.is_mated())
      {
        // 後手の詰みなら手順を表示する。先手の詰みは必要ない。
        if (pos.side_to_move() == WHITE)
        {
          // 現在詰まないことが判明している探索深さ(search_depth)+2の長さの詰みを発見したときのみ。
          while (!Signals.stop && !mate_found) // 他のスレッドが見つけるかも知れないのでそれを待ちながら…。
          {
            if (search_depth + 2 >= id_depth_thread /*- depth + 1*/)
            {
              mate_found = true;
              sync_cout << "checkmate " << pos.moves_from_start() << sync_endl; // 開始局面からそこまでの手順
              break;
            }
            sleep(100);
          }
        }
      } else if (depth > 1) {
        // 残り探索深さがあるなら再帰的に探索する。
        int child_no_mate_depth;
        search(pos, depth - 1, child_no_mate_depth);
        no_mate_depth = min(child_no_mate_depth + 1, no_mate_depth);

        if (child_no_mate_depth != MAX_PLY)
        {
          replyCount++;
          oneReply = m;
        }

      } else {
        no_mate_depth = 1; // frontier node。この先、まだ探索すれば詰むかも知れないので..

        replyCount++;
        oneReply = m;
      }
      pos.undo_move(m);
    }

    // このnodeに関して残り探索深さdepthについては詰みを調べきったので不詰めとして扱い、置換表に記録しておく。
    // また、確定局面以外の子が1つしかなればそれを置換表に書き出しておく。(次回の指し手生成をはしょるため)
    if (replyCount != 1)
      oneReply = MOVE_NONE;

    TT.save(key, no_mate_depth, oneReply);
  }

  // 協力詰め探索の反復深化のループ
  void id_loop(Position& pos, int thread_id, int thread_num)
  {
    pos.this_thread()->nodes = 0;
    auto start_time = now();

    // 協力詰めの反復深化は2手ずつ深くして良い。
    // lazy SMPっぽい並列化をする。
    for (uint32_t depth = 1 + thread_id * 2; depth < MAX_PLY; depth += 2 * thread_num)
    {
      // 置換表のgenerationをインクリメントするのはmain threadだけ。
      if (thread_id == 0)
        TT.new_search();

      int no_mate_depth;
      id_depth_thread = depth;
      search(pos, depth, no_mate_depth);

      if (Signals.stop || mate_found)
        break;

      // 定期的にdepth、nodes、npsを出力する。
      auto end_time = now();
      auto node_searched = Threads.nodes_searched(); // 全スレッドでの探索合計
      sync_cout << "info  depth " << depth
        << " nodes " << node_searched
        << " nps " << (node_searched * 1000 / ((int64_t)(end_time - start_time + 1)))
        << " hashfull " << TT.hashfull()
        << sync_endl;

      // 最大探索深さに到達する前に王手が続かなくなっていたなら終了
      if (no_mate_depth == MAX_PLY)
      {
        sync_cout << "checkmate nomate" << sync_endl;
        break;
      }

      // depth手では詰まないことが証明できたのでsearch_depthを書き換える。
      // 他のスレッドが書き換える可能性もあるので値が大きいときのみ。
      while (true)
      {
        auto sd = search_depth.load();
        if (depth > sd)
        {
          if (search_depth.compare_exchange_weak(sd, depth))
            break;
        } else break; // 下回っているので書き込む価値はない。
      }
    }
  }

  void init()
  {
    search_depth = 0;
    mate_found = false;
  }

  void finalize()
  {
    if (!Signals.stop && !mate_found)
    {
      sync_cout << "info string give up." << sync_endl;
      sync_cout << "checkmate nomate" << sync_endl; // checkmateコマンドを返さないと将棋所が待機したままになる
    }
  }

} // end of namespace

// --- Search

void Search::init() {}
void Search::clear() { HelpMate::TT.clear(); }
void MainThread::think() {
  HelpMate::init();
  for (auto th : Threads.slaves) th->start_searching();
  Thread::search();
  for (auto th : Threads.slaves) th->wait_for_search_finished();
  HelpMate::finalize();
}
void Thread::search() { HelpMate::id_loop(rootPos, (int)thread_id(), (int)Options["Threads"]); }

#endif
