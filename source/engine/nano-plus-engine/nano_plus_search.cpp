#include "../../shogi.h"

#ifdef YANEURAOU_NANO_PLUS_ENGINE

// -----------------------
//   やねうら王nano plus探索部
// -----------------------

// 開発方針
// ・nanoに似た読みやすいソースコード
// ・nanoからオーダリングを改善。
// ・超高速1手詰めを使用。
// ・250行程度のシンプルな探索部でR2500を目指す。
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

namespace YaneuraOuNanoPlus
{

  // 外部から調整される探索パラメーター
  int param1 = 0;
  int param2 = 0;

  // -----------------------
  //  探索のときに使うStack
  // -----------------------

  struct Stack {
    Move killers[2];    // killer move
    Move currentMove;   // そのスレッドの探索においてこの局面で現在選択されている指し手
    Value staticEval;   // 評価関数を呼び出して得た値。NULL MOVEのときに親nodeでの評価値が欲しいので保存しておく。
    int ply;            // rootからの手数
  };

  // -----------------------
  //     Stackのupdate
  // -----------------------

  // いい探索結果だったときにkiller等を更新する

  void update_stats(const Position& pos, Stack* ss, Move move,
    Depth depth, Move* quiets, int quietsCnt) {

    // killer 2本しかないので[0]と違うならいまの[0]を[1]に降格させて[0]と差し替え
    if (ss->killers[0] != move)
    {
      ss->killers[1] = ss->killers[0];
      ss->killers[0] = move;
    }
  }


  // -----------------------
  //   指し手オーダリング
  // -----------------------

  // 指し手を段階的に生成するために現在どの段階にあるかの状態を表す定数
  enum Stages {
    // -----------------------------------------------------
    //   王手がかっていない通常探索時用の指し手生成
    // -----------------------------------------------------
    MAIN_SEARCH_START,            // 置換表の指し手を返すフェーズ
    GOOD_CAPTURES,                // 捕獲する指し手(CAPTURES_PRO_PLUS)を生成して指し手を一つずつ返す
    KILLERS,                      // KILLERの指し手
    BAD_CAPTURES,                 // 捕獲する悪い指し手
    GOOD_QUIETS,                  // CAPTURES_PRO_PLUSで生成しなかった指し手を生成して、一つずつ返す
    BAD_QUIETS,                   // ↑で点数悪そうなものを後回しにしていたのでそれを一つずつ返す

    // -----------------------------------------------------
    //   王手がかっている/静止探索時用の指し手生成
    // -----------------------------------------------------
    EVASION_START,                // 置換表の指し手を返すフェーズ
    ALL_EVASIONS,                 // 回避する指し手(EVASIONS)を生成した指し手を一つずつ返す
    
    // -----------------------------------------------------
    //   王手がかっていない静止探索時用の指し手生成
    // -----------------------------------------------------

    QSEARCH_WITH_CHECKS_START,    // 王手がかかっているときはここから開始
    QCAPTURES_1,                  // 捕獲する指し手
    QCHECKS,                      // 王手となる指し手(上で生成している捕獲の指し手を除外した王手)

    QSEARCH_WITHOUT_CHECKS_START, // 王手がかかっていないときはここから開始
    QCAPTURES_2,                  // 捕獲する指し手

    // 静止探索で深さ-2以降は組み合わせ爆発を防ぐためにrecaptureのみを生成
    RECAPTURE_START,              // ↓のstageに行くためのラベル
    GOOD_RECAPTURES,              // 最後の移動した駒を捕獲する指し手(RECAPTURES)を生成した指し手を一つずつ返す

    STOP,                         // 終端
  };
  ENABLE_OPERATORS_ON(Stages); // 次の状態にするためにインクリメントを使いたい。

  // 指し手オーダリング器
  struct MovePicker
  {
    // 通常探索から呼び出されるとき用。
    MovePicker(const Position& pos_,Move ttMove_,Stack*ss_) : pos(pos_),ss(ss_)
    {
      // 次の指し手生成の段階
      // 王手がかかっているなら回避手、かかっていないなら通常探索用の指し手生成
      stage = pos.in_check() ? EVASION_START : MAIN_SEARCH_START;

      // 置換表の指し手があるならそれを最初に返す。ただしpseudo_legalでなければならない。
      ttMove = ttMove_ && pos.pseudo_legal(ttMove_) ? ttMove_ : MOVE_NONE;

      // 置換表の指し手が引数で渡されていたなら1手生成したことにする。
      // (currentMoves != endMovesであることを、指し手を生成するかどうかの判定に用いている)
      endMoves += (ttMove_!= MOVE_NONE);
    }

    // 静止探索から呼び出される時用。
    MovePicker(const Position& pos_, Move ttMove_, Depth depth, Square recapSq, Stack*ss_) : pos(pos_),ss(ss_)
    {
      if (pos.in_check())
        stage = EVASION_START;
      else if (depth > DEPTH_QS_NO_CHECKS)
        stage = QSEARCH_WITH_CHECKS_START;

      else if (depth > DEPTH_QS_RECAPTURES)
        stage = QSEARCH_WITHOUT_CHECKS_START;

      else
      {
        stage = RECAPTURE_START;
        recaptureSquare = recapSq;
        ttMove = MOVE_NONE; // 置換表の指し手はrecaptureの升に移動させる指し手ではないので忘れる
        return;
      }

      ttMove = ttMove_ && pos.pseudo_legal(ttMove_) ? ttMove_ : MOVE_NONE;
      endMoves += (ttMove_ != MOVE_NONE);
    }

    // 次のstageにするため、必要なら指し手生成器で指し手を生成する。
    void generate_next_stage()
    {
      ASSERT_LV3(stage != STOP);

      // 指し手生成バッファの先頭を指すように
      currentMoves = moves;

      // 次のステージに移行して、そのときに指し手生成が必要なステージに達したなら指し手を生成する。
      switch (++stage)
      {
      case GOOD_CAPTURES: case QCAPTURES_1 : case QCAPTURES_2:
        endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, moves);
        break;

      case GOOD_RECAPTURES:
        endMoves = generateMoves<RECAPTURES>(pos, moves, recaptureSquare);
        break;

        // あとで実装する(↑で生成して返さなかった指し手を返すフェーズ)
      case BAD_CAPTURES:
        endMoves = moves;
        break;

      case GOOD_QUIETS:
        endMoves = generateMoves<NON_CAPTURES_PRO_MINUS>(pos, moves);
        break;

        // あとで実装する(↑で生成して返さなかった指し手を返すフェーズ)
      case BAD_QUIETS:
        endMoves = moves;
        break;

      case KILLERS:
        killers[0] = ss->killers[0];
        killers[1] = ss->killers[1];
        currentMoves = killers;
        endMoves = currentMoves + 2;
        break;

      case ALL_EVASIONS:
        endMoves = generateMoves<EVASIONS>(pos, moves);
        break;

      case QCHECKS:
        endMoves = generateMoves<QUIET_CHECKS>(pos, moves);
        break;

        // そのステージの末尾に達したのでMovePickerを終了する。
      case EVASION_START: case QSEARCH_WITH_CHECKS_START: case QSEARCH_WITHOUT_CHECKS_START:
      case RECAPTURE_START: case STOP:
        stage = STOP;
        break;

      default:
        UNREACHABLE;
        break;
      }

    }

    // 次の指し手をひとつ返す
    // 指し手が尽きればMOVE_NONEが返る。
    Move nextMove() {

      Move move;

      while (true)
      {
        while (currentMoves == endMoves && stage != STOP)
          generate_next_stage();

        switch (stage)
        {
          // 置換表の指し手を返すフェーズ
        case MAIN_SEARCH_START: case EVASION_START:
        case QSEARCH_WITH_CHECKS_START: case QSEARCH_WITHOUT_CHECKS_START:
          ++currentMoves;
          return ttMove;

          // killer moveを1手ずつ返すフェーズ
          // (直前に置換表の指し手を返しているし、CAPTURES_PRO_PLUSでの指し手も返しているのでそれらの指し手は除外されるべき)
        case KILLERS:
          move = *currentMoves++;
          if (  move != MOVE_NONE         // ss->killer[0],[1]からコピーしただけなのでMOVE_NONEの可能性がある
            &&  move != ttMove            // 置換表の指し手を重複除去しないといけない
            &&  pos.pseudo_legal(move)
            && !pos.capture_or_pawn_promotion(move))  // 直前にCAPTURES_PRO_PLUSで生成している指し手を除外
            return move;
          break;

          // 置換表の指し手を返したあとのフェーズ
          // (killer moveの前のフェーズなのでkiller除去は不要)
        case GOOD_CAPTURES:
        case ALL_EVASIONS: case QCAPTURES_1: case QCAPTURES_2:
          move = *currentMoves++;
          if (move != ttMove)
            return move;
          break;

          // 指し手を一手ずつ返すフェーズ
          // (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
        case BAD_CAPTURES: case GOOD_QUIETS: case BAD_QUIETS:
          move = *currentMoves++;
          // 置換表の指し手、killerと同じものは返してはならない。
          if ( move != ttMove
            && move != killers[0]
            && move != killers[1])
            return move;
          break;

          // 王手になる指し手を一手ずつ返すフェーズ
          // (置換表の指し手とCAPTURES_PRO_PLUSの指し手は返したあとなのでこれらの指し手は除外する必要がある)
        case QCHECKS:
          move = *currentMoves++;
          if (  move != ttMove
            && !pos.capture_or_pawn_promotion(move)) // 直前にCAPTURES_PRO_PLUSで生成している指し手を除外
            return move;
          break;

          // 取り返す指し手。これはすでに生成されているのでそのまま返すだけで良い。
        case GOOD_RECAPTURES:
          move = *currentMoves++;
          return move;

        case STOP:
          return MOVE_NONE;

        default:
          UNREACHABLE;
          break;
        }
      }
    }

  private:
    const Position& pos;

    // 指し手生成の段階
    Stages stage;

    // RECAPUTREの指し手で移動させる先の升
    Square recaptureSquare;

    // 置換表の指し手
    Move ttMove;

    // killer move
    ExtMove killers[2];

    // node stack
    Stack* ss;
    
    // 指し手生成バッファと、次に返す指し手、生成された指し手の末尾
    ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;
  };

  // -----------------------
  //      探索用の定数
  // -----------------------

  // 探索しているnodeの種類
  enum NodeType { Root, PV, NonPV };

  // 探索深さを減らすためのReductionテーブル
  // [PvNodeであるか][improvingであるか][このnodeで何手目の指し手であるか][残りdepth]
  Depth reduction_table[2][2][64][64];

  // 残り探索深さをこの深さだけ減らす。depthとmove_countに関して63以上は63とみなす。
  // improvingとは、評価値が2手前から上がっているかのフラグ。上がっていないなら
  // 悪化していく局面なので深く読んでも仕方ないからreduction量を心もち増やす。
  template <bool PvNode> Depth reduction(bool improving, Depth depth, int move_count) {
    return reduction_table[PvNode][improving][std::min((int)depth/ONE_PLY, 63 )][std::min(move_count, 63)];
  }

  // -----------------------
  //      静止探索
  // -----------------------

  // InCheck : 王手がかかっているか
  template <NodeType NT,bool InCheck>
  Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
  {
    // -----------------------
    //     変数宣言
    // -----------------------

    // PV nodeであるか。
    const bool PvNode = NT == PV;

    // 評価値の最大を求めるために必要なもの
    Value bestValue;       // この局面での指し手のベストなスコア(alphaとは違う)
    Move bestMove;         // そのときの指し手
    Value oldAlpha;        // 関数が呼び出されたときのalpha値

    // hash key関係
    TTEntry* tte;          // 置換表にhitしたときの置換表のエントリーへのポインタ
    Key posKey;            // この局面のhash key
    bool ttHit;            // 置換表にhitしたかのフラグ
    Move ttMove;           // 置換表に登録されていた指し手
    Value ttValue;         // 置換表に登録されていたスコア
    Depth ttDepth;         // このnodeに関して置換表に登録するときの残り探索深さ

    // 王手関係
    bool givesCheck;       // MovePickerから取り出した指し手で王手になるか

    // -----------------------
    //     nodeの初期化
    // -----------------------

    if (PvNode)
    {
      // PV nodeではalpha値を上回る指し手が存在した場合は(そこそこ指し手を調べたので)置換表にはBOUND_EXACTで保存したいから、
      // そのことを示すフラグとして元の値が必要(non PVではこの変数は参照しない)
      // PV nodeでalpha値を上回る指し手が存在しなかった場合は、調べ足りないのかも知れないからBOUND_UPPERとしてbestValueを保存しておく。
      oldAlpha = alpha;
    }

    ss->currentMove = bestMove = MOVE_NONE;

    // rootからの手数
    ss->ply = (ss - 1)->ply + 1;

    // -----------------------
    //  引き分け、および、最大手数到達
    // -----------------------

    // これ以上探索できない。give up。
    if (ss->ply >= MAX_PLY)
      return Eval::eval(pos);

    // -----------------------
    //     置換表のprobe
    // -----------------------

    posKey = pos.state()->key();
    tte = TT.probe(posKey, ttHit);
    ttMove = ttHit ? tte->move() : MOVE_NONE;
    ttValue = ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;

    // 置換表に登録するdepthは、あまりマイナスの値が登録されてもおかしいので、
    // 王手がかかっているときは、DEPTH_QS_CHECKS(=0)、王手がかかっていないときはDEPTH_QS_NO_CHECKSの深さとみなす。
    ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
                                                  : DEPTH_QS_NO_CHECKS;

    // nonPVでは置換表の指し手で枝刈りする
    // PVでは置換表の指し手では枝刈りしない(前回evaluateした値は使える)
    if (!PvNode
      && ttHit
      && tte->depth() >= ttDepth
      && ttValue != VALUE_NONE // 置換表から取り出したときに他スレッドが値を潰している可能性があるのでこのチェックが必要
      && (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
                          : (tte->bound() & BOUND_UPPER)))
      // ttValueが下界(真の評価値はこれより大きい)もしくはジャストな値で、かつttValue >= beta超えならbeta cutされる
      // ttValueが上界(真の評価値はこれより小さい)だが、tte->depth()のほうがdepthより深いということは、
      // 今回の探索よりたくさん探索した結果のはずなので、今回よりは枝刈りが甘いはずだから、その値を信頼して
      // このままこの値でreturnして良い。
    {
      ss->currentMove = ttMove; // MOVE_NONEでありうるが
      return ttValue;
    }

    // -----------------------
    //     eval呼び出し
    // -----------------------

    Value value;
    if (InCheck)
    {
      // 王手がかかっているならすべての指し手を調べるべきなのでevaluate()は呼び出さない。
      ss->staticEval = VALUE_NONE;

      // bestValueはalphaとは違う。
      // 王手がかかっているときは-VALUE_INFINITEを初期値として、すべての指し手を生成してこれを上回るものを探すので
      // alphaとは区別しなければならない。
      bestValue = -VALUE_INFINITE;

    } else {

      // 王手がかかっていないなら置換表の指し手を持ってくる

      if (ttHit)
      {

        // 置換表に評価値が格納されているとは限らないのでその場合は評価関数の呼び出しが必要
        // bestValueの初期値としてこの局面のevaluate()の値を使う。これを上回る指し手があるはずなのだが..
        if ((ss->staticEval = bestValue = tte->eval()) == VALUE_NONE)
          ss->staticEval = bestValue = Eval::eval(pos);

        // 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
        // 精度で探索されたものであるなら、それをbestValueの初期値として使う。
        if (ttValue != VALUE_NONE)
          if (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))
            bestValue = ttValue;

      } else {

        // 置換表がhitしなかった場合、bestValueの初期値としてevaluate()を呼び出すしかないが、
        // NULL_MOVEの場合は前の局面での値を反転させると良い。(手番を考慮しない評価関数であるなら)
        ss->staticEval = bestValue =
          (ss - 1)->currentMove != MOVE_NULL ? Eval::eval(pos)
                                             : -(ss - 1)->staticEval;
      }

      // Stand pat.
      // 現在のbestValueは、この局面で何も指さないときのスコア。recaptureすると損をする変化もあるのでこのスコアを基準に考える。
      // 王手がかかっていないケースにおいては、この時点での静的なevalの値がbetaを上回りそうならこの時点で帰る。
      if (bestValue >= beta)
      {
        if (!ttHit)
          tte->save(posKey, value_to_tt(bestValue, ss->ply), BOUND_LOWER,
            DEPTH_NONE, MOVE_NONE, ss->staticEval, TT.generation());

        return bestValue;
      }

      // 一手詰め判定
      Move m;
      m = pos.mate1ply();
      if (m != MOVE_NONE)
      {
        bestValue = mate_in(ss->ply);
        tte->save(posKey, value_to_tt(bestValue, ss->ply), BOUND_EXACT,
          DEPTH_MAX, m , ss->staticEval, TT.generation());

        return bestValue;
      }

      // 王手がかかっていなくてPvNodeでかつ、bestValueがalphaより大きいならそれをalphaの初期値に使う。
      // 王手がかかっているなら全部の指し手を調べたほうがいい。
      if (PvNode && bestValue > alpha)
        alpha = bestValue;
    }

    // -----------------------
    //     1手ずつ調べる
    // -----------------------

    // 取り合いの指し手だけ生成する
    // searchから呼び出された場合、直前の指し手がMOVE_NULLであることがありうるが、
    // 静止探索の1つ目の深さではrecaptureを生成しないならこれは問題とならない。
    // ToDo: あとでNULL MOVEを実装したときにrecapture以外も生成するように修正する。
    pos.check_info_update();
    MovePicker mp(pos,ttMove,depth,move_to((ss-1)->currentMove),ss);
    Move move;

    StateInfo si;

    while (move = mp.nextMove())
    {
      if (!pos.legal(move))
        continue;

      // 現在このスレッドで探索している指し手を保存しておく。
      ss->currentMove = move;

      givesCheck = pos.gives_check(move);

      pos.do_move(move, si, pos.gives_check(move));
      value = givesCheck ? -YaneuraOuNanoPlus::qsearch<NT, true>(pos, ss + 1, -beta, -alpha, depth - ONE_PLY)
                         : -YaneuraOuNanoPlus::qsearch<NT,false>(pos, ss + 1, -beta, -alpha, depth - ONE_PLY);
      pos.undo_move(move);

      if (Signals.stop)
        return VALUE_ZERO;

      // bestValue(≒alpha値)を更新するのか
      if (value > bestValue)
      {
        bestValue = value;

        if (value > alpha)
        {
          if (PvNode && value < beta)
          {
            // alpha値の更新はこのタイミングで良い。
            // なぜなら、このタイミング以外だと枝刈りされるから。(else以下を読むこと)
            alpha = value;
            bestMove = move;
          } else
          {
            // 1. nonPVでのalpha値の更新 →　もうこの時点でreturnしてしまっていい。(ざっくりした枝刈り)
            // 2. PVでのvalue >= beta、すなわちfail high
            tte->save(posKey, value_to_tt(value, ss->ply), BOUND_LOWER,
              ttDepth, move, ss->staticEval, TT.generation());
            return value;
          }
        }
      }
    }

    // -----------------------
    // 指し手を調べ終わった
    // -----------------------

    // 王手がかかっている状況ではすべての指し手を調べたということだから、これは詰みである
    // どうせ指し手がないということだから、次にこのnodeに訪問しても、指し手生成後に詰みであることは
    // わかるわけだから、この種の詰みを置換表に登録する価値があるかは微妙であるが、とりあえず保存しておくことにする。
    if (InCheck && bestValue == -VALUE_INFINITE)
    {
      bestValue = mated_in(ss->ply); // rootからの手数による詰みである。
      tte->save(posKey, value_to_tt(bestValue, ss->ply),BOUND_EXACT,
        DEPTH_MAX, MOVE_NONE, ss->staticEval, TT.generation());
      return bestValue;
    }

    tte->save(posKey, value_to_tt(bestValue, ss->ply),
      PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
      ttDepth, bestMove, ss->staticEval, TT.generation());

    ASSERT_LV3(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
  }

  // -----------------------
  //      通常探索
  // -----------------------

  template <NodeType NT>
  Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth)
  {
    // -----------------------
    //     nodeの種類
    // -----------------------

    // root nodeであるか
    const bool RootNode = NT == Root;

    // PV nodeであるか(root nodeはPV nodeに含まれる)
    const bool PvNode = NT == PV || NT == Root;

    // -----------------------
    //     変数の宣言
    // -----------------------

    // 調べた指し手を残しておいて、statusのupdateを行なうときに使う。
    Move quietsSearched[64];
    int quietCount;

    // この局面でdo_move()された合法手の数
    int moveCount;

    // LMRのときにfail highが起きるなどしたので元の残り探索深さで探索することを示すフラグ
    bool fullDepthSearch;

    // 指し手で捕獲する指し手、もしくは成りである。
    bool captureOrPromotion;

    // -----------------------
    //     nodeの初期化
    // -----------------------

    ASSERT_LV3(alpha < beta);

    // rootからの手数
    ss->ply = (ss - 1)->ply + 1;

    // -----------------------
    //  Mate Distance Pruning
    // -----------------------

    if (!RootNode)
    {
      // rootから5手目の局面だとして、このnodeのスコアが
      // 5手以内で詰ますときのスコアを上回ることもないし、
      // 5手以内で詰まさせるときのスコアを下回ることもないので
      // これを枝刈りする。

      alpha = std::max(mated_in(ss->ply), alpha);
      beta = std::min(mate_in(ss->ply + 1), beta);
      if (alpha >= beta)
        return alpha;
    }

    // -----------------------
    //  探索Stackの初期化
    // -----------------------

    // 2手先のkillerの初期化。
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

    // -----------------------
    //   置換表のprobe
    // -----------------------

    auto key = pos.state()->key();

    bool ttHit;    // 置換表がhitしたか
    TTEntry* tte = TT.probe(key, ttHit);

    // 置換表上のスコア
    // 置換表にhitしなければVALUE_NONE
    Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;

    auto thisThread = pos.this_thread();

    // 置換表の指し手
    // 置換表にhitしなければMOVE_NONE

    // RootNodeであるなら、指し手は現在注目している1手だけであるから、それが置換表にあったものとして指し手を進める。
    Move ttMove = RootNode ? thisThread->rootMoves[thisThread->PVIdx].pv[0]
      : ttHit ? tte->move() : MOVE_NONE;

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
      ss->currentMove = ttMove; // この指し手で枝刈りをした。ただしMOVE_NONEでありうる。

      // 置換表の指し手でbeta cutが起きたのであれば、この指し手をkiller等に登録する。
      // ただし、捕獲する指し手か成る指し手であればこれはkillerを更新する価値はない。
      if (ttValue >= beta && ttMove && !pos.capture_or_promotion(ttMove))
        update_stats(pos, ss, ttMove, depth, nullptr, 0);

      return ttValue;
    }

    // -----------------------
    //    1手詰みか？
    // -----------------------

    Move bestMove = MOVE_NONE;

#if 0
    // 通常探索において1手詰めを発見することはあまりないので
    // ここで詰将棋探索を呼び出すのはあまり得をしない。

    // RootNodeでは1手詰め判定、ややこしくなるのでやらない。
    // 置換表にhitしたときも1手詰め判定は行われていると思われるのでこの場合もはしょる
    if (!RootNode && !ttHit)
    {
      bestMove = pos.mate1ply();
      if (bestMove != MOVE_NONE)
      {
        // 1手詰めスコアなので確実にvalue > alphaなはず。
        alpha = mate_in(ss->ply);
        goto TT_SAVE;
      }
    }
#endif

    // -----------------------
    // 1手ずつ指し手を試す
    // -----------------------

    {
      pos.check_info_update();
      MovePicker mp(pos, ttMove,ss);

      Value value;
      Move move;
      StateInfo si;
      bool givesCheck; // 今回の指し手で王手になるかどうか

      moveCount = quietCount = 0;

      while (move = mp.nextMove())
      {
        // root nodeでは、rootMoves()の集合に含まれていない指し手は探索をスキップする。
        if (RootNode && !std::count(thisThread->rootMoves.begin() + thisThread->PVIdx,
          thisThread->rootMoves.end(), move))
          continue;

        // legal()のチェック。root nodeだとlegal()だとわかっているのでこのチェックは不要。
        if (!RootNode && !pos.legal(move))
          continue;

        captureOrPromotion = pos.capture_or_promotion(move);
        givesCheck = pos.gives_check(move);

        // -----------------------
        //      1手進める
        // -----------------------

        // 現在このスレッドで探索している指し手を保存しておく。
        ss->currentMove = move;

        // 指し手で1手進める
        pos.do_move(move, si, givesCheck);

        // do_moveした指し手の数のインクリメント
        ++moveCount;

        // -----------------------
        // 再帰的にsearchを呼び出す
        // -----------------------

        // Reduced depth search(LMR)
        // 探索深さを減らしてざっくり調べる。alpha値を更新しそうなら(fail highが起きたら)、
        // full depthで探索しなおす。
        if (depth >= 3 * ONE_PLY && moveCount > 1 && !captureOrPromotion)
        {
          // Reduction量
          Depth r = reduction<PvNode>(false,depth,moveCount);

          // depth >= 3なのでqsearchは呼ばれないし、かつ、
          // moveCount > 1 すなわち、2手目移行なのでsearch<NonPv>が呼び出されるべき。
          Depth d = max(depth - r, ONE_PLY);
          value = -YaneuraOuNanoPlus::search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d);

          // 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす
          fullDepthSearch = value > alpha;
        } else {
          fullDepthSearch = true;
        }

        if (fullDepthSearch)
          value = depth - ONE_PLY < ONE_PLY ?
                                 givesCheck ? -qsearch<PV , true>(pos, ss+1 , -beta, -alpha, depth - ONE_PLY)
                                            : -qsearch<PV, false>(pos, ss + 1, -beta, -alpha, depth - ONE_PLY)
                                            : -YaneuraOuNanoPlus::search<PV>(pos, ss+1 , -beta, -alpha, depth - ONE_PLY);

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
        alpha = mated_in(ss->ply);

      // 詰まされていない場合、bestMoveがあるならこの指し手をkiller等に登録する。
      else if (bestMove && !pos.capture_or_promotion(bestMove))
        update_stats(pos, ss, bestMove, depth, quietsSearched, quietCount);
    }

    // -----------------------
    //  置換表に保存する
    // -----------------------

//TT_SAVE:;

    tte->save(key, value_to_tt(alpha, ss->ply),
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

using namespace YaneuraOuNanoPlus;

// --- 以下に好きなように探索のプログラムを書くべし。

// 定跡ファイル
Book::MemoryBook book;

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init() {

  // -----------------------
  //   定跡の読み込み
  // -----------------------
  Book::read_book("book/standard_book.db", book);

  // -----------------------
  // reduction tableの初期化
  // -----------------------

  // pvとnon pvのときのreduction定数
  // とりあえずStockfishに合わせておく。あとで調整する。
  const double K[][2] = { { 0.799, 2.281 },{ 0.484, 3.023 } };

  for (int pv = 0; pv <= 1; ++pv)
    for (int imp = 0; imp <= 1; ++imp)
      for (int d = 1; d < 64; ++d)
        for (int mc = 1; mc < 64; ++mc)
        {
          double r = K[pv][0] + log(d) * log(mc) / K[pv][1];

          if (r >= 1.5)
            reduction_table[pv][imp][d][mc] = int(r) * ONE_PLY;

          // improving(評価値が2手前から上がっている)でないときはreductionの量を増やす。
          if (!pv && !imp && reduction_table[pv][imp][d][mc] >= 2 * ONE_PLY)
            reduction_table[pv][imp][d][mc] += ONE_PLY;
        }


} 

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear() { TT.clear(); }

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::think() {

  // ---------------------
  // 探索パラメーターの自動調整用
  // ---------------------
  param1 = Options["Param1"];
  param2 = Options["Param2"];

  // ---------------------
  //      variables
  // ---------------------

  Stack stack[MAX_PLY + 4], *ss = stack + 2; // (ss-2)と(ss+2)にアクセスしたいので4つ余分に確保しておく。
  Move bestMove;

  memset(stack, 0, 5 * sizeof(Stack)); // 先頭5つを初期化しておけば十分。そのあとはsearchの先頭でss+2を初期化する。

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
    static PRNG prng;
    auto it = book.find(rootPos.sfen());
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
      auto availableTime = std::max(2000, Limits.time[us] / 60 + Limits.byoyomi[us]);
      // 思考時間は秒単位で繰り上げ
      availableTime = (availableTime / 1000) * 1000;
      // 50msより小さいと思考自体不可能なので下限を50msに。
      availableTime = std::max(50, availableTime - Options["NetworkDelay"]);
      auto endTime = Limits.startTime + availableTime;

      // タイマースレッドを起こして、終了時間を監視させる。

      timerThread = new std::thread([&] {
        while (now() < endTime && !Signals.stop)
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

      YaneuraOuNanoPlus::search<Root>(rootPos, ss+1 , alpha, beta, rootDepth * ONE_PLY);

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

#endif // YANEURAOU_NANO_PLUS_ENGINE
