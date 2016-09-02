#include "../../shogi.h"

#ifdef YANEURAOU_NANO_ENGINE

// -----------------------
//   やねうら王nano探索部
// -----------------------

// 開発方針
// ・並列探索をしない(1スレッド動作)
// ・αβ探索以外の枝刈り手法を極力使わない。
// ・CAPTURESを優先する以外の指し手オーダリングをしない。
// ・1手詰め判定を用いない。
// ・静止探索において置換表に書き出さない。
// ・静止探索ではRECAPTURESの指し手のみを生成。
// このあと改造していくためのベースとなる教育的なコードを目指す。

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include "../../position.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"
#include "../../extra/book.h"

using namespace std;
using namespace Search;
using namespace Eval;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
  //
  //   パラメーターの外部からの自動調整
  //

  o["Param1"] << Option(0, 0, 100000);
  o["Param2"] << Option(0, 0, 100000);
}

namespace YaneuraOuNano
{
  // -----------------------
  //   指し手オーダリング
  // -----------------------

  struct MovePicker
  {
    // 通常探索から呼び出されるとき用。
    MovePicker(const Position& pos_,Move ttMove) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、すべての指し手(NON_EVASIONS)で指し手を生成する。
      if (pos.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
      {
        // CAPTURESを先に生成しておくことで枝刈り性能をupさせる。
        // 本当は、CAPTURESの指し手が尽きてから段階的に生成すべきだが、MovePickerが少し複雑になるので簡略しておく。
        endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, currentMoves);
        endMoves = generateMoves<NON_CAPTURES_PRO_MINUS>(pos, endMoves);
      }

      // 置換表の指し手が、この生成された集合のなかにあるなら、その先頭の指し手に置換表の指し手が来るようにしておく。
      if (ttMove != MOVE_NONE && pos.pseudo_legal(ttMove))
      {
        auto p = currentMoves;
        for (; p != endMoves;++p)
          if (*p == ttMove)
          {
            swap(*p, *currentMoves);
            break;
          }
      }
    }

    // 静止探索から呼び出される時用。
    MovePicker(const Position& pos_, Square recapSq) : pos(pos_)
    {
      // 王手がかかっているなら回避手(EVASIONS)、さもなくば、recaptureのみ生成。
      if (pos.in_check())
        endMoves = generateMoves<EVASIONS>(pos, currentMoves);
      else
        // recapture以外生成しない。
        endMoves = generateMoves<RECAPTURES>(pos, currentMoves,recapSq);
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

  // -----------------------
  //      探索用の定数
  // -----------------------

  // 探索しているnodeの種類
  enum NodeType { Root, PV, NonPV };

  // -----------------------
  //      静止探索
  // -----------------------

  // search()で残り探索深さが0以下になったときに呼び出される。
  // (より正確に言うなら、残り探索深さがONE_PLY未満になったときに呼び出される)

  template <NodeType NT>
  Value qsearch(Position& pos, Value alpha, Value beta, Depth depth)
  {
    // 現在のnodeのrootからの手数。これカウンターが必要。
    // nanoだとこのカウンター持ってないので適当にごまかす。
    const int ply_from_root = (pos.this_thread()->rootDepth - depth / ONE_PLY) + 1;

    // この局面で王手がかかっているのか
    bool InCheck = pos.checkers();

    Value value;
    if (InCheck)
    {
      // 王手がかかっているならすべての指し手を調べる。
      alpha = -VALUE_INFINITE;

    } else {
      // 王手がかかっていないなら置換表の指し手を持ってくる

      // この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
      value = evaluate(pos);

      if (alpha < value)
      {
        alpha = value;
        if (alpha >= beta)
          return alpha; // beta cut
      }

      // 探索深さが-3以下ならこれ以上延長しない。
      if (depth < -3 * ONE_PLY)
        return alpha;
    }

    // 取り合いの指し手だけ生成する
    MovePicker mp(pos,move_to(pos.state()->lastMove));
    Move move;

    StateInfo si;

    while (move = mp.next_move())
    {
      if (!pos.legal(move))
        continue;

      pos.do_move(move, si, pos.gives_check(move));
      value = -YaneuraOuNano::qsearch<NT>(pos, -beta, -alpha, depth - ONE_PLY);
      pos.undo_move(move);

      if (Signals.stop)
        return VALUE_ZERO;

      if (value > alpha) // update alpha?
      {
        alpha = value;
        if (alpha >= beta)
          return alpha; // beta cut
      }
    }

    // 王手がかかっている状況ですべての指し手を調べたということだから、これは詰みである
    if (InCheck && alpha == -VALUE_INFINITE)
      return mated_in(ply_from_root);

    return alpha;
  }

  // -----------------------
  //      通常探索
  // -----------------------

  template <NodeType NT>
  Value search(Position& pos, Value alpha, Value beta, Depth depth)
  {
    ASSERT_LV3(alpha < beta);

    // -----------------------
    //     nodeの種類
    // -----------------------

    // root nodeであるか
    const bool RootNode = NT == Root;

    // PV nodeであるか(root nodeはPV nodeに含まれる)
    const bool PvNode = NT == PV || NT == Root;

    // -----------------------
    //     変数宣言
    // -----------------------

    // 現在のnodeのrootからの手数。これカウンターが必要。
    // nanoだとこのカウンター持ってないので適当にごまかす。
    const int ply_from_root = (pos.this_thread()->rootDepth - depth / ONE_PLY) + 1;

    // -----------------------
    //   置換表のprobe
    // -----------------------

    auto key = pos.state()->key();

    bool ttHit;    // 置換表がhitしたか
    TTEntry* tte = TT.probe(key, ttHit);

    // 置換表上のスコア
    // 置換表にhitしなければVALUE_NONE
    Value ttValue = ttHit ? value_from_tt(tte->value(), ply_from_root) : VALUE_NONE;

    auto thisThread = pos.this_thread();

    // 置換表の指し手
    // 置換表にhitしなければMOVE_NONE

    // RootNodeであるなら、指し手は現在注目している1手だけであるから、それが置換表にあったものとして指し手を進める。
    Move ttMove = RootNode ? thisThread->rootMoves[thisThread->PVIdx].pv[0]
                  :  ttHit ? pos.move16_to_move(tte->move()) : MOVE_NONE;

    // 置換表の値による枝刈り
    
    if (!PvNode        // PV nodeでは置換表の指し手では枝刈りしない(PV nodeはごくわずかしかないので..)
      && ttHit         // 置換表の指し手がhitして
      && tte->depth() >= depth   // 置換表に登録されている探索深さのほうが深くて
      && ttValue != VALUE_NONE   // (VALUE_NONEだとすると他スレッドからTTEntryが読みだす直前に破壊された可能性がある)
      && (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
                          : (tte->bound() & BOUND_UPPER))
      // ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
      // ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
      // 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
      // このままこの値でreturnして良い。
      )
    {
      return ttValue;
    }

    // -----------------------
    // 1手ずつ指し手を試す
    // -----------------------

    MovePicker mp(pos,ttMove);

    Value value;
    Move move;

    StateInfo si;

    // この局面でdo_move()された合法手の数
    int moveCount = 0;
    Move bestMove = MOVE_NONE;

    while (move = mp.next_move())
    {
      // root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。
      if (RootNode && !std::count(thisThread->rootMoves.begin() + thisThread->PVIdx,
        thisThread->rootMoves.end(), move))
        continue;

      // legal()のチェック。root nodeだとlegal()だとわかっているのでこのチェックは不要。
      if (!RootNode && !pos.legal(move))
        continue;

      // -----------------------
      //      1手進める
      // -----------------------

      pos.do_move(move, si, pos.gives_check(move));

      // do_moveした指し手の数のインクリメント
      ++moveCount;

      // -----------------------
      // 再帰的にsearchを呼び出す
      // -----------------------

      // PV nodeの1つ目の指し手で進めたnodeは、PV node。さもなくば、non PV nodeとして扱い、
      // alphaの値を1でも超えるかどうかだけが問題なので簡単なチェックで済ませる。

      // また、残り探索深さがなければ静止探索を呼び出して評価値を返す。
      // (searchを再帰的に呼び出して、その先頭でチェックする呼び出しのオーバーヘッドが嫌なのでここで行なう)

      bool fullDepthSearch = (PV && moveCount == 1);

      if (!fullDepthSearch)
      {
        // nonPVならざっくり2手ぐらい深さを削っていいのでは..(本当はもっとちゃんとやるべき)
        Depth R = ONE_PLY * 2;

        value = depth - R < ONE_PLY ?
          -qsearch<NonPV>(pos, -beta, -alpha, depth - R) :
          -YaneuraOuNano::search<NonPV>(pos, -(alpha + 1), -alpha, depth - R);

        // 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす
        fullDepthSearch = value > alpha;
      }

      if ( fullDepthSearch)
        value = depth - ONE_PLY < ONE_PLY ?
            -qsearch<PV>(pos, -beta, -alpha, depth - ONE_PLY) :
            -YaneuraOuNano::search<PV>(pos, -beta, -alpha, depth - ONE_PLY);

      // -----------------------
      //      1手戻す
      // -----------------------

      pos.undo_move(move);

      // 停止シグナルが来たら置換表を汚さずに終了。
      if (Signals.stop)
        return VALUE_ZERO;

      // -----------------------
      //  root node用の特別な処理
      // -----------------------

      if (RootNode)
      {
        auto& rm = *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);

        if (moveCount == 1 || value > alpha)
        {
          // root nodeにおいてPVの指し手または、α値を更新した場合、スコアをセットしておく。
          // (iterationの終わりでsortするのでそのときに指し手が入れ替わる。)

          rm.score = value;
          rm.pv.resize(1); // PVは変化するはずなのでいったんリセット

          // ここにPVを代入するコードを書く。(か、置換表からPVをかき集めてくるか)

        } else {

          // root nodeにおいてα値を更新しなかったのであれば、この指し手のスコアを-VALUE_INFINITEにしておく。
          // こうしておかなければ、stable_sort()しているにもかかわらず、前回の反復深化のときの値との
          // 大小比較してしまい指し手の順番が入れ替わってしまうことによるオーダリング性能の低下がありうる。
          rm.score = -VALUE_INFINITE;
        }
      }

      // -----------------------
      //  alpha値の更新処理
      // -----------------------

      if (value > alpha)
      {
        alpha = value;
        bestMove = move;

        // αがβを上回ったらbeta cut
        if (alpha >= beta)
          break;
      }

    } // end of while

    // -----------------------
    //  生成された指し手がない？
    // -----------------------
      
    // 合法手がない == 詰まされている ので、rootの局面からの手数で詰まされたという評価値を返す。
    if (moveCount == 0)
      alpha = mated_in(ply_from_root);

    // -----------------------
    //  置換表に保存する
    // -----------------------

    tte->save(key, value_to_tt(alpha, ply_from_root),
      alpha >= beta ? BOUND_LOWER : 
      PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
      // betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
      // すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
      // さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
      // また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
      // ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
      // すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。
      depth, bestMove, VALUE_NONE,TT.generation());

    return alpha;
  }

}

using namespace YaneuraOuNano;

// --- 以下に好きなように探索のプログラムを書くべし。

// 定跡ファイル
Book::MemoryBook book;

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init() {}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear() {
  static bool first = true;
  if (first)
  {
    Book::read_book("book/standard_book.db", book);
    first = false;
  }
  TT.clear();
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::think() {

  static PRNG prng;
  Move bestMove;

  // ---------------------
  // 合法手がないならここで投了
  // ---------------------

  if (rootMoves.size() == 0)
  {
    bestMove = MOVE_RESIGN;
    Signals.stop = true;
    goto ID_END;
  }

  // ---------------------
  //     定跡の選択部
  // ---------------------

  {
    auto it = book.find(rootPos);
    if (it != book.end()) {
      // 定跡にhitした。逆順で出力しないと将棋所だと逆順にならないという問題があるので逆順で出力する。
      const auto& move_list = it->second;
      for (auto it = move_list.rbegin(); it != move_list.rend();it++ )
        sync_cout << "info pv " << it->bestMove << " " << it->nextMove
        << " (" << fixed << setprecision(2) << (100* it->prob) << "%)" // 採択確率
        << " score cp " << it->value << " depth " << it->depth << sync_endl;

      // このなかの一つをランダムに選択
      // 無難な指し手が選びたければ、採択回数が一番多い、最初の指し手(move_list[0])を選ぶべし。
      bestMove = move_list[prng.rand(move_list.size())].bestMove;

      Signals.stop = true;
      goto ID_END;
    }
  }

  // ---------------------
  //    通常の思考処理
  // ---------------------
  {
    
    rootDepth = 0;
    Value alpha, beta;
    StateInfo si;
    auto& pos = rootPos;

    // --- 置換表のTTEntryの世代を進める。

    TT.new_search();

    // ---------------------
    //   思考の終了条件
    // ---------------------

    std::thread* timerThread = nullptr;

    // 探索深さ、ノード数、詰み手数が指定されていない == 探索時間による制限
    if (!(Limits.depth || Limits.nodes || Limits.mate))
    {
      // 時間制限があるのでそれに従うために今回の思考時間を計算する。
      // 今回に用いる思考時間 = 残り時間の1/60 + 秒読み時間

      auto us = pos.side_to_move();
      // 2秒未満は2秒として問題ない。(CSAルールにおいて)

      int availableTime;
      if (!Limits.rtime)
      {
        availableTime = std::max(2000, Limits.time[us] / 60 + Limits.byoyomi[us]);
        // 思考時間は秒単位で繰り上げ
        availableTime = (availableTime / 1000) * 1000;
        // 50msより小さいと思考自体不可能なので下限を50msに。
        availableTime = std::max(50, availableTime - Options["NetworkDelay"]);
      } else {
        // 1～3倍の範囲でランダム化する。
        availableTime = Limits.rtime + (int)prng.rand(Limits.rtime * 2);
      }

      auto endTime = availableTime;

      // タイマースレッドを起こして、終了時間を監視させる。

      timerThread = new std::thread([&] {
        while (Time.elapsed() < endTime && !Signals.stop)
          sleep(10);
        Signals.stop = true;
      });
    }

    // ---------------------
    //   反復深化のループ
    // ---------------------

    while (++rootDepth < MAX_PLY && !Signals.stop && (!Limits.depth || rootDepth <= Limits.depth))
    {
      // 本当はもっと探索窓を絞ったほうが効率がいいのだが…。
      alpha = -VALUE_INFINITE;
      beta = VALUE_INFINITE;

      PVIdx = 0; // MultiPVではないのでPVは1つで良い。

      YaneuraOuNano::search<Root>(rootPos, alpha, beta, rootDepth*ONE_PLY);

      // それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
      std::stable_sort(rootMoves.begin(), rootMoves.end());

      // 読み筋を出力しておく。
      sync_cout << USI::pv(pos, rootDepth, alpha, beta) << sync_endl;
    }

    bestMove = rootMoves.at(0).pv[0];

    // ---------------------
    // タイマースレッド終了
    // ---------------------

    Signals.stop = true;
    if (timerThread != nullptr)
    {
      timerThread->join();
      delete timerThread;
    }
  }

ID_END:; // 反復深化の終了。

  // ---------------------
  // 指し手をGUIに返す
  // ---------------------

  // ponder中であるならgoコマンドか何かが送られてきてからのほうがいいのだが、とりあえずponderの処理は後回しで…。

  sync_cout << "bestmove " << bestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search(){}

#endif // YANEURAOU_NANO_ENGINE
