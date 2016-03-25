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
#include "../../move_picker.h"

using namespace std;
using namespace Search;
using namespace Eval;

// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o) {}

namespace YaneuraOuNanoPlus
{

  // 外部から調整される探索パラメーター
  int param1 = 0;
  int param2 = 0;

  // -----------------------
  //      探索用の定数
  // -----------------------

  // 探索しているnodeの種類
  // Rootはここでは用意しない。Rootに特化した関数を用意するのが少し無駄なので。
  enum NodeType { PV, NonPV };

  // game ply(≒進行度)とdepth(残り探索深さ)に応じたfutility margin。
  Value futility_margin(Depth d,int game_ply) {
    // 80手目を終盤と定義して、終盤に近づくほどmarginの幅を上げるように調整する。
//    game_ply = min(80, game_ply);
//    return Value(d * ((param1+1) * 30 + game_ply * param2 / 2));
    return Value(d * 90);
  }

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
  //     Statsのupdate
  // -----------------------

  // MovePickerで用いる直前の指し手に対するそれぞれの指し手のスコア
  CounterMoveHistoryStats CounterMoveHistory;

  // いい探索結果だったときにkiller等を更新する

  // move      = これが良かった指し手
  // quiets    = 悪かった指し手(このnodeで生成した指し手)
  // quietsCnt = ↑の数
  inline void update_stats(const Position& pos, Stack* ss, Move move,
    Depth depth, Move* quiets, int quietsCnt)
  {

    //   killerのupdate

    // killer 2本しかないので[0]と違うならいまの[0]を[1]に降格させて[0]と差し替え
    if (ss->killers[0] != move)
    {
      ss->killers[1] = ss->killers[0];
      ss->killers[0] = move;
    }

    //   historyのupdate

    // depthの二乗に比例したbonusをhistory tableに加算する。
    Value bonus = Value(depth*(int)depth + (int)depth + 1);

    // 直前に移動させた升(その升に移動させた駒がある)
    Square prevSq = move_to((ss - 1)->currentMove);
    auto& cmh = CounterMoveHistory[prevSq][pos.piece_on(prevSq)];
    auto thisThread = pos.this_thread();

    thisThread->history.update(pos.moved_piece(move), move_to(move), bonus);

    // 前の局面の指し手がMOVE_NULLでないならcounter moveもupdateしておく。
    if (is_ok((ss - 1)->currentMove))
    {
      thisThread->counterMoves.update(pos.piece_on(prevSq), prevSq, move);
      cmh.update(pos.moved_piece(move), move_to(move), bonus);
    }

    // このnodeのベストの指し手以外の指し手はボーナス分を減らす
    for (int i = 0; i < quietsCnt; ++i)
    {
      thisThread->history.update(pos.moved_piece(quiets[i]), move_to(quiets[i]), -bonus);

      if (is_ok((ss - 1)->currentMove))
        cmh.update(pos.moved_piece(quiets[i]), move_to(quiets[i]), -bonus);
    }

  }


  // -----------------------
  //      静止探索
  // -----------------------

  // search()で残り探索深さが0以下になったときに呼び出される。
  // (より正確に言うなら、残り探索深さがONE_PLY未満になったときに呼び出される)

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
    //    千日手等の検出
    // -----------------------

    // 連続王手による千日手、および通常の千日手、優等局面・劣等局面。
    auto draw_type = pos.is_repetition();
    if (draw_type != REPETITION_NONE)
      return draw_value(draw_type,pos.side_to_move());

    if (ss->ply >= MAX_PLY)
      return draw_value(REPETITION_DRAW,pos.side_to_move());

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

    // mate1ply()でCheckInfo.pinnedを使うのでここで初期化しておく。
    pos.check_info_update();

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
          ss->staticEval = bestValue = evaluate(pos);

        // 置換表に格納されていたスコアは、この局面で今回探索するものと同等か少しだけ劣るぐらいの
        // 精度で探索されたものであるなら、それをbestValueの初期値として使う。
        if (ttValue != VALUE_NONE)
          if (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))
            bestValue = ttValue;

      } else {

        // 置換表がhitしなかった場合、bestValueの初期値としてevaluate()を呼び出すしかないが、
        // NULL_MOVEの場合は前の局面での値を反転させると良い。(手番を考慮しない評価関数であるなら)
        ss->staticEval = bestValue =
          (ss - 1)->currentMove != MOVE_NULL ? evaluate(pos)
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
        bestValue = mate_in(ss->ply + 1); // 1手詰めなのでこの次のnodeで詰むという解釈
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
    MovePicker mp(pos,ttMove,depth,pos.this_thread()->history,move_to((ss-1)->currentMove));
    Move move;

    StateInfo si;

    while (move = mp.next_move())
    {
      if (!pos.legal(move))
        continue;

      givesCheck = pos.gives_check(move);

      // 現在このスレッドで探索している指し手を保存しておく。
      ss->currentMove = move;

      pos.do_move(move, si, givesCheck);
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

    } else {

      tte->save(posKey, value_to_tt(bestValue, ss->ply),
        PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
        ttDepth, bestMove, ss->staticEval, TT.generation());
    }

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

    // PV nodeであるか(root nodeはPV nodeに含まれる)
    const bool PvNode = NT == PV;

    // root nodeであるか
    const bool RootNode = PvNode && (ss-1)->ply == 0;

    // -----------------------
    //     変数の宣言
    // -----------------------

    // この局面に対する評価値の見積り。
    Value eval;

    // 調べた指し手を残しておいて、statusのupdateを行なうときに使う。
    Move quietsSearched[64];
    int quietCount;

    // この局面でdo_move()された合法手の数
    int moveCount;

    // LMRのときにfail highが起きるなどしたので元の残り探索深さで探索することを示すフラグ
    bool fullDepthSearch;

    // 指し手で捕獲する指し手、もしくは成りである。
    bool captureOrPromotion;

    // この局面でのベストのスコア
    Value bestValue;

    // -----------------------
    //     nodeの初期化
    // -----------------------

    ASSERT_LV3(alpha < beta);

    bestValue = -VALUE_INFINITE;

    // rootからの手数
    ss->ply = (ss - 1)->ply + 1;

    // -----------------------
    //     千日手等の検出
    // -----------------------

    if (!RootNode)
    {
      auto draw_type = pos.is_repetition();
      if (draw_type != REPETITION_NONE)
        return draw_value(draw_type,pos.side_to_move());

      // 最大手数を超えている
      if (ss->ply >= MAX_PLY)
        return draw_value(REPETITION_DRAW,pos.side_to_move());
    }

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

    // 1手先のskipEarlyPruningフラグの初期化。
    (ss + 1)->skipEarlyPruning = false;

    // 2手先のkillerの初期化。
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

    // -----------------------
    //   置換表のprobe
    // -----------------------

    auto posKey = pos.state()->key();

    bool ttHit;    // 置換表がhitしたか
    TTEntry* tte = TT.probe(posKey, ttHit);

    // 置換表上のスコア
    // 置換表にhitしなければVALUE_NONE
    Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply) : VALUE_NONE;

    auto thisThread = pos.this_thread();

    // 置換表の指し手
    // 置換表にhitしなければMOVE_NONE
                   
    // RootNodeであるなら、(MultiPVなどでも)現在注目している1手だけがベストの指し手と仮定できるから、
    // それが置換表にあったものとして指し手を進める。
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

    // RootNodeでは1手詰め判定、ややこしくなるのでやらない。(RootMovesの入れ替え等が発生するので)
    // 置換表にhitしたときも1手詰め判定は行われていると思われるのでこの場合もはしょる。
    // depthの残りがある程度ないと、1手詰めはどうせこのあとすぐに見つけてしまうわけで1手詰めを
    // 見つけたときのリターン(見返り)が少ない。
    if (!RootNode && !ttHit && depth > ONE_PLY)
    {
      bestMove = pos.mate1ply();
      if (bestMove != MOVE_NONE)
      {
        // 1手詰めスコアなので確実にvalue > alphaなはず。
        bestValue = mate_in(ss->ply);
        tte->save(posKey, value_to_tt(bestValue, ss->ply), BOUND_EXACT,
          DEPTH_MAX , bestMove, ss->staticEval, TT.generation());

        return bestValue;
      }
    }

    // -----------------------
    //  局面を評価値によって静的に評価
    // -----------------------

    const bool InCheck = pos.checkers();

    if (InCheck)
    {
      // 評価値を置換表から取り出したほうが得だと思うが、反復深化でこのnodeに再訪問したときも
      // このnodeでは評価値を用いないであろうから、置換表にこのnodeの評価値があることに意味がない。

      ss->staticEval = eval = VALUE_NONE;
      goto MOVES_LOOP;
    }
    else if (ttHit)
    {
      // 置換表にhitしたなら、評価値が記録されているはずだから、それを取り出しておく。
      // あとで置換表に書き込むときにこの値を使えるし、各種枝刈りはこの評価値をベースに行なうから。

      // tte->eval()へのアクセスは1回にしないと他のスレッドが壊してしまう可能性がある。
      if ((ss->staticEval = eval = tte->eval()) == VALUE_NONE)
        ss->staticEval = evaluate(pos);

      eval = ss->staticEval;

      // ttValueのほうがこの局面の評価値の見積もりとして適切であるならそれを採用する。
      // 1. ttValue > evaluate()でかつ、ttValueがBOUND_LOWERなら、真の値はこれより大きいはずだから、
      //   evalとしてttValueを採用して良い。
      // 2. ttValue < evaluate()でかつ、ttValueがBOUND_UPPERなら、真の値はこれより小さいはずだから、
      //   evalとしてttValueを採用したほうがこの局面に対する評価値の見積りとして適切である。
      if (  ttValue != VALUE_NONE && (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
        eval = ttValue;

    } else {

      ss->staticEval = eval =
        (ss - 1)->currentMove != MOVE_NULL ? evaluate(pos)
                                           : -(ss - 1)->staticEval;

      // 評価関数を呼び出したので置換表のエントリーはなかったことだし、何はともあれそれを保存しておく。
      tte->save(posKey, VALUE_NONE, BOUND_NONE, DEPTH_NONE, MOVE_NONE, ss->staticEval, TT.generation());
    }

    // このnodeで指し手生成前の枝刈りを省略するなら指し手生成ループへ。
    if (ss->skipEarlyPruning)
      goto MOVES_LOOP;

    // -----------------------
    //   evalベースの枝刈り
    // -----------------------

    // 局面の静的評価値(eval)が得られたので、以下ではこの評価値を用いて各種枝刈りを行なう。
    // 王手のときはここにはこない。(上のInCheckのなかでMOVES_LOOPに突入。)

    //
    //   Futility pruning
    //

    // このあとの残り探索深さによって、評価値が変動する幅はfutility_margin(depth)だと見積れるので
    // evalからこれを引いてbetaより大きいなら、beta cutが出来る。
    // ただし、将棋の終盤では評価値の変動の幅は大きくなっていくので、進行度に応じたfutility_marginが必要となる。
    // ここでは進行度としてgamePly()を用いる。このへんはあとで調整すべき。
    
    if (!RootNode
      &&  depth < 7 * ONE_PLY
      &&  eval - futility_margin(depth, pos.game_ply()) >= beta
      &&  eval < VALUE_KNOWN_WIN ) // 詰み絡み等だとmate distance pruningで枝刈りされるはずで、ここでは枝刈りしない。
      return eval - futility_margin(depth , pos.game_ply());


    // -----------------------
    // 1手ずつ指し手を試す
    // -----------------------

  MOVES_LOOP:

    {
      Value value = bestValue; // gccの初期化されていないwarningの抑制
      Move move;
      StateInfo si;

      // 今回の指し手で王手になるかどうか
      bool givesCheck;

      // 評価値が2手前の局面から上がって行っているのかのフラグ
      // 上がって行っているなら枝刈りを甘くする。
      // ※ VALUE_NONEの場合は、王手がかかっていてevaluate()していないわけだから、
      //   枝刈りを甘くして調べないといけないのでimproving扱いとする。
      bool improving = ss->staticEval >= (ss - 2)->staticEval
        || ss->staticEval == VALUE_NONE
        || (ss - 2)->staticEval == VALUE_NONE;

      moveCount = quietCount = 0;


      //  MovePickerでのオーダリングのためにhistory tableなどを渡す

      // 親nodeでの指し手でのtoの升
      auto prevSq = move_to((ss - 1)->currentMove);
      // その升へ移動させた駒
      auto prevPc = pos.piece_on(prevSq);
      // toの升に駒pcを動かしたことに対する応手
      auto cm = thisThread->counterMoves[prevSq][prevPc];
      // counter history
      const auto& cmh = CounterMoveHistory[prevSq][prevPc];

      pos.check_info_update();
      MovePicker mp(pos, ttMove, depth, thisThread->history, cmh, cm, ss);


      //  一手ずつ調べていく

      while (move = mp.next_move())
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

        // 再帰的にsearchを呼び出すとき、search関数に渡す残り探索深さ。
        Depth newDepth = depth - ONE_PLY;

        // Reduced depth search(LMR)
        // 探索深さを減らしてざっくり調べる。alpha値を更新しそうなら(fail highが起きたら)、full depthで探索しなおす。

        if (depth >= 3 * ONE_PLY && moveCount > 1 && !captureOrPromotion)
        {
          // Reduction量
          Depth r = reduction<PvNode>(improving,depth,moveCount);

          // captureとpromotionに関してはreduction量を減らしてもう少し突っ込んで調べる。
          // 歩以外の捕獲の指し手は、捕獲から逃れる手があって有利になるかも知れないので
          // このときの探索を浅くしてしまうと局面の評価の精度が下がる。
          if (r
              && pos.capture(move)
              && type_of(pos.piece_on(move_to(move))) != PAWN // 歩以外の捕獲
            )
            r = std::max(DEPTH_ZERO, r - ONE_PLY);

          // depth >= 3なのでqsearchは呼ばれないし、かつ、
          // moveCount > 1 すなわち、このnodeの2手目以降なのでsearch<NonPv>が呼び出されるべき。
          Depth d = max(newDepth - r, ONE_PLY);
          value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d);

          // 上の探索によりalphaを更新しそうだが、いい加減な探索なので信頼できない。まともな探索で検証しなおす
          fullDepthSearch = (value > alpha) && (r != DEPTH_ZERO);

        } else {

          // non PVか、PVでも2手目以降であればfull depth searchを行なう。
          fullDepthSearch = !PvNode || moveCount > 1;

        }

        // Full depth search
        // LMRがskipされたか、LMRにおいてfail highを起こしたなら元の探索深さで探索する。

        // ※　静止探索は残り探索深さはDEPTH_ZEROとして開始されるべきである。(端数があるとややこしいため)
        if (fullDepthSearch)
          value = newDepth < ONE_PLY ?
                          givesCheck ? -qsearch<NonPV , true>(pos, ss + 1 , -(alpha + 1), -alpha, DEPTH_ZERO)
                                     : -qsearch<NonPV, false>(pos, ss + 1 , -(alpha + 1), -alpha, DEPTH_ZERO)
                                     : - search<NonPV>       (pos, ss + 1 , -(alpha + 1), -alpha, newDepth);

        // PV nodeにおいては、full depth searchがfail highしたならPV nodeとしてsearchしなおす。
        // ただし、value >= betaなら、正確な値を求めることにはあまり意味がないので、これはせずにbeta cutしてしまう。
        if (PvNode && (moveCount == 1 || (value > alpha && (RootNode || value < beta))))
        {
          value = newDepth < ONE_PLY ?
                          givesCheck ? -qsearch<PV,  true>(pos, ss + 1, -beta, -alpha, DEPTH_ZERO)
                                     : -qsearch<PV, false>(pos, ss + 1, -beta, -alpha, DEPTH_ZERO)
                                     : - search<PV>       (pos, ss + 1, -beta, -alpha, newDepth);
        }


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

        if (value > bestValue)
        {
          bestValue = value;

          if (value > alpha)
          {
            bestMove = move;

            if (PvNode && value < beta) // alpha値を更新したので更新しておく
              alpha = value;
            else
            {
              // value >= beta なら fail high(beta cut)
              // また、non PVであるなら探索窓の幅が0なのでalphaを更新した時点で、value >= betaが言えて、
              // beta cutである。
              break;
            }
          }
        }

        // 探索した指し手を64手目までquietsSearchedに登録しておく。
        // あとでhistoryなどのテーブルに加点/減点するときに使う。

        if (!captureOrPromotion && move != bestMove && quietCount < 64)
          quietsSearched[quietCount++] = move;

      } // end of while

      // -----------------------
      //  生成された指し手がない？
      // -----------------------
      
      // 合法手がない == 詰まされている ので、rootの局面からの手数で詰まされたという評価値を返す。
      if (!moveCount)
        bestValue = mated_in(ss->ply);

      // 詰まされていない場合、bestMoveがあるならこの指し手をkiller等に登録する。
      else if (bestMove && !pos.capture_or_promotion(bestMove))
        update_stats(pos, ss, bestMove, depth, quietsSearched, quietCount);
    }

    // -----------------------
    //  置換表に保存する
    // -----------------------

    tte->save(posKey, value_to_tt(bestValue, ss->ply),
      bestValue >= beta ? BOUND_LOWER : 
      PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
      // betaを超えているということはbeta cutされるわけで残りの指し手を調べていないから真の値はまだ大きいと考えられる。
      // すなわち、このとき値は下界と考えられるから、BOUND_LOWER。
      // さもなくば、(PvNodeなら)枝刈りはしていないので、これが正確な値であるはずだから、BOUND_EXACTを返す。
      // また、PvNodeでないなら、枝刈りをしているので、これは正確な値ではないから、BOUND_UPPERという扱いにする。
      // ただし、指し手がない場合は、詰まされているスコアなので、これより短い/長い手順の詰みがあるかも知れないから、
      // すなわち、スコアは変動するかも知れないので、BOUND_UPPERという扱いをする。
      depth, bestMove, ss->staticEval ,TT.generation());

    return bestValue;
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
  // LMRで使うreduction tableの初期化
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
          // →　これ、ほとんど効果がないようだ…。あとで調整すべき。
          if (!pv && !imp && reduction_table[pv][imp][d][mc] >= 2 * ONE_PLY)
            reduction_table[pv][imp][d][mc] += ONE_PLY;
        }


} 

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void Search::clear()
{
  TT.clear();
  CounterMoveHistory.clear();
  for (Thread* th : Threads)
  {
    th->history.clear();
    th->counterMoves.clear();
  }
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。

void MainThread::think()
{
  // ---------------------
  // 探索パラメーターの自動調整用
  // ---------------------
  param1 = Options["Param1"];
  param2 = Options["Param2"];

  // ---------------------
  //      variables
  // ---------------------

  static PRNG prng;

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

    // 探索時間が指定されているなら監視タイマーを起動しておく。
    if (Limits.use_time_management())
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

      YaneuraOuNanoPlus::search<PV>(rootPos, ss , alpha, beta, rootDepth * ONE_PLY);

      // それぞれの指し手に対するスコアリングが終わったので並べ替えおく。
      std::stable_sort(rootMoves.begin(), rootMoves.end());

      // 読み筋を出力しておく。
      if (!Limits.silent)
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

  if (!Limits.silent)
    sync_cout << "bestmove " << bestMove << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
void Thread::search(){}

#endif // YANEURAOU_NANO_PLUS_ENGINE
