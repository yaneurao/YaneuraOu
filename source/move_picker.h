#ifndef _MOVE_PICKER_H_
#define _MOVE_PICKER_H_

#include "shogi.h"

// -----------------------
//   MovePicker
// -----------------------

#ifdef USE_MOVE_PICKER

#include "search.h"

// -----------------------
//   insertion sort
// -----------------------

// stableであることが保証されたinsertion sort。指し手オーダリングのために使う。
inline void insertion_sort(ExtMove* begin, ExtMove* end)
{
  ExtMove tmp, *p, *q;

  for (p = begin + 1; p < end; ++p)
  {
    tmp = *p;
    for (q = p; q != begin && *(q - 1) < tmp; --q)
      *q = *(q - 1);
    *q = tmp;
  }
}

// -----------------------
//  history , counter move
// -----------------------

// Pieceを升sqに移動させるときの値(T型)
// CM : CounterMove用フラグ
template<typename T, bool CM = false>
struct Stats {

  // このtableの要素の最大値
  static const Value Max = Value(1 << 28);

  // tableの要素の値を取り出す
  const T* operator[](Square to) const {
    ASSERT_LV4(is_ok(to));
    return table[to];
  }
  T* operator[](Square to) {
    ASSERT_LV4(is_ok(to));
    return table[to];
  }

  // tableのclear
  void clear() { std::memset(table, 0, sizeof(table)); }

  // tableに指し手を格納する。(Tの型がMoveのとき)
  void update(Piece pc, Square to, Move m)
  {
    ASSERT_LV4(is_ok(pc));
    ASSERT_LV4(is_ok(to));
    table[to][pc] = m;
  }

  // tableに値を格納する(Tの型がValueのとき)
  void update(Piece pc, Square to, Value v) {

    ASSERT_LV4(is_ok(pc));
    ASSERT_LV4(is_ok(to));

    // abs(v) <= 324に制限する。
    v = max((Value)-324, v);
    v = min((Value)+324, v);

    // if (abs(int(v) >= 324) return ; のほうが良いのでは..

    table[to][pc] -= table[to][pc] * abs(int(v)) / (CM ? 512 : 324);
    table[to][pc] += int(v) * (CM ? 64 : 32);
  }

private:
  // Pieceを升sqに移動させるときの値
  // ※　Stockfishとは添字が逆順だが、将棋ではPIECE_NBのほうだけが2^Nなので仕方がない。
  T table[SQ_NB_PLUS1][PIECE_NB];
};

// Statsは、pcをsqの升に移動させる指し手に対してT型の値を保存する。
// TがMoveのときは、指し手に対する指し手、すなわち、"応手"となる。
// TがValueのときは指し手に対するスコアとなる。これがhistory table(HistoryStatsとCounterMoveStats)
// このStats<CounterMoveStats>は、直前の指し手に対する、あらゆる指し手に対するスコアである。

typedef Stats<Move            > MoveStats;
typedef Stats<Value, false    > HistoryStats;
typedef Stats<Value, true     > CounterMoveStats;
typedef Stats<CounterMoveStats> CounterMoveHistoryStats;
  
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

  // -----------------------------------------------------
  //   通常探索のProbCutの処理のなかから呼び出される用
  // -----------------------------------------------------

  PROBCUT_START,                // 置換表の指し手を返すフェーズ
  PROBCUT_CAPTURES,             // 直前の指し手での駒の価値を上回る駒取りの指し手のみを生成するフェーズ

  STOP,                         // 終端
};
ENABLE_OPERATORS_ON(Stages); // 次の状態にするためにインクリメントを使いたい。

// 指し手オーダリング器
struct MovePicker
{
  // 通常探索から呼び出されるとき用。
  MovePicker(const Position& pos_,Move ttMove_,Depth depth_, const HistoryStats& history_,
    const CounterMoveStats& cms,Move counterMove_, Search::Stack*ss_)
    : pos(pos_),history(history_),counterMoveHistory(&cms), ss(ss_),counterMove(counterMove_),depth(depth_)
  {
    // 次の指し手生成の段階
    // 王手がかかっているなら回避手、かかっていないなら通常探索用の指し手生成
    stage = pos.in_check() ? EVASION_START : MAIN_SEARCH_START;

    // 置換表の指し手があるならそれを最初に返す。ただしpseudo_legalでなければならない。
    ttMove = ttMove_ && pos.pseudo_legal_s<false,false>(ttMove_) ? ttMove_ : MOVE_NONE;

    // 置換表の指し手が引数で渡されていたなら1手生成したことにする。
    // (currentMoves != endMovesであることを、指し手を生成するかどうかの判定に用いている)
    endMoves += (ttMove!= MOVE_NONE);
  }

  // 静止探索から呼び出される時用。
  MovePicker(const Position& pos_, Move ttMove_, Depth depth, const HistoryStats& history_, Square recapSq)
    : pos(pos_),history(history_),counterMoveHistory(nullptr)
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

    // 歩の不成、香の2段目への不成、大駒の不成を除外
    ttMove = ttMove_ && pos.pseudo_legal_s<false,false>(ttMove_) ? ttMove_ : MOVE_NONE;
    endMoves += (ttMove != MOVE_NONE);
  }
  
  // 通常探索時にProbCutの処理から呼び出されるの専用
  MovePicker(const Position& p, Move ttMove_, const HistoryStats& h, Value threshold_)
    : pos(p), history(h), counterMoveHistory(nullptr), threshold(threshold_) {

    ASSERT_LV3(!pos.checkers());

    stage = PROBCUT_START;

    // In ProbCut we generate captures with SEE higher than the given threshold
    ttMove = ttMove_
      && pos.pseudo_legal_s<false,false>(ttMove_)
      && pos.capture(ttMove_)
      && pos.see(ttMove_) > threshold ? ttMove_ : MOVE_NONE;

    endMoves += (ttMove != MOVE_NONE);
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
    case GOOD_CAPTURES: case QCAPTURES_1: case QCAPTURES_2: case PROBCUT_CAPTURES:
      endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, moves);
      score_captures(); // CAPTUREの指し手の並べ替え。
      break;

    case KILLERS:
      killers[0] = ss->killers[0];
      killers[1] = ss->killers[1];
      killers[2] = counterMove;
      currentMoves = killers;
      endMoves = currentMoves + 2 + (counterMove != killers[0] && counterMove != killers[1]);
      break;

    case GOOD_RECAPTURES:
      endMoves = generateMoves<RECAPTURES>(pos, moves, recaptureSquare);
      break;

      // あとで実装する(↑で生成して返さなかった指し手を返すフェーズ)
    case BAD_CAPTURES:
      // SEE<0の指し手を指し手生成バッファの末尾に回していたのでそれを順番に返す。
      currentMoves = moves + MAX_MOVES - 1; // 指し手生成バッファの末尾
      endMoves = endBadCaptures;
      break;

    case GOOD_QUIETS:
      endQuiets = endMoves = generateMoves<NON_CAPTURES_PRO_MINUS>(pos, moves);
      score_quiets();
      // プラスの符号のものだけ前方に移動させて、今回のフェーズではそれを返す。
      endMoves = std::partition(currentMoves, endMoves, [](const ExtMove& m) { return m.value > VALUE_ZERO; });
      // その移動させたものは少数のはずなので、sortしても遅くない。
      insertion_sort(currentMoves, endMoves);
      break;

      // ↑で生成して返さなかった指し手を返すフェーズ
    case BAD_QUIETS:
      currentMoves = endMoves;
      endMoves = endQuiets; // 生成自体は前のフェーズで完了している。
      // 残り探索深さがある程度あるなら、ソートする時間は相対的に無視できる。
      if (depth >= 3 * ONE_PLY)
        insertion_sort(currentMoves, endMoves);
      break;

    case ALL_EVASIONS:
      endMoves = generateMoves<EVASIONS>(pos, moves);
      // 生成された指し手が2手以上あるならオーダリングする。
      if (endMoves - moves > 1)
        score_evasions();
      break;

    case QCHECKS:
      endMoves = generateMoves<QUIET_CHECKS>(pos, moves);
      break;

      // そのステージの末尾に達したのでMovePickerを終了する。
    case EVASION_START: case QSEARCH_WITH_CHECKS_START: case QSEARCH_WITHOUT_CHECKS_START:
    case PROBCUT_START: case RECAPTURE_START: case STOP:
      stage = STOP;
      break;

    default:
      UNREACHABLE;
      break;
    }

  }

  // 次の指し手をひとつ返す
  // 指し手が尽きればMOVE_NONEが返る。
  Move next_move() {

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
      case PROBCUT_START:
        ++currentMoves;
        return ttMove;

        // killer moveを1手ずつ返すフェーズ
        // (直前に置換表の指し手を返しているし、CAPTURES_PRO_PLUSでの指し手も返しているのでそれらの指し手は除外されるべき)
        // また、killerの3つ目はcounter moveでこれは先後の区別がないのでpseudo_legal_s<X,true>()を呼び出す必要がある。
      case KILLERS:
        move = *currentMoves++;
        if (move != MOVE_NONE                       // ss->killer[0],[1]からコピーしただけなのでMOVE_NONEの可能性がある
          &&  move != ttMove                        // 置換表の指し手を重複除去しないといけない
          &&  pos.pseudo_legal_s<false,true>(move)  // pseudo_legalでない指し手以外に歩や大駒の不成なども除外
          && !pos.capture_or_pawn_promotion(move))  // 直前にCAPTURES_PRO_PLUSで生成している指し手を除外
          return move;
        break;

        // 置換表の指し手を返したあとのフェーズ
        // (killer moveの前のフェーズなのでkiller除去は不要)
        // SSEの符号がマイナスのものはbad captureのほうに回す。
      case GOOD_CAPTURES:
        move = pick_best(currentMoves++, endMoves);
        if (move != ttMove)
        {
#ifdef USE_SEE
          // ここでSSEの符号がマイナスならbad captureのほうに回す。
          if (pos.see_sign(move) >= VALUE_ZERO)
            return move;

          // 損をするCAPTUREの指し手は、後回しにする。
          // これは指し手生成バッファの末尾から使っていく。
          *endBadCaptures-- = move;
#else
          return move;
#endif
        }
        break;

        // 置換表の指し手を返したあとのフェーズ
        // (killer moveの前のフェーズなのでkiller除去は不要)
        // また、SSEの符号がマイナスのものもbad captureのほうに回す処理は不要なのでこのまま
        // 置換表の指し手と異なるなら指し手を返していけば良い。
      case ALL_EVASIONS: case QCAPTURES_1: case QCAPTURES_2:
        move = pick_best(currentMoves++, endMoves);
        if (move != ttMove)
          return move;
        break;

        // 指し手を一手ずつ返すフェーズ
        // (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
      case GOOD_QUIETS: case BAD_QUIETS:
        move = *currentMoves++;
        // 置換表の指し手、killerと同じものは返してはならない。
        // ※　これ、指し手の数が多い場合、AVXを使って一気に削除しておいたほうが良いのでは..
        if ( move != ttMove
          && move != killers[0]
          && move != killers[1]
          && move != killers[2])
          return move;
        break;

        // BAD CAPTURESは、指し手生成バッファの終端から先頭方向に向かって使う。
      case BAD_CAPTURES:
        return *currentMoves--;

        // 王手になる指し手を一手ずつ返すフェーズ
        // (置換表の指し手とCAPTURES_PRO_PLUSの指し手は返したあとなのでこれらの指し手は除外する必要がある)
      case QCHECKS:
        move = *currentMoves++;
        if (  move != ttMove
          && !pos.capture_or_pawn_promotion(move)) // 直前にCAPTURES_PRO_PLUSで生成している指し手を除外
          return move;
        break;

        // 通常探索のProbCutの処理から呼び出されるとき用。
        // 直前に捕獲された駒の価値を上回るようなcaptureの指し手のみを生成する。
      case PROBCUT_CAPTURES:
        move = pick_best(currentMoves++, endMoves);
        if (move != ttMove && pos.see(move) > threshold)
          return move;
        break;

        // 取り返す指し手。これはすでにrecaptureの指し手だけが生成されているのでそのまま返す。
      case GOOD_RECAPTURES:
        move = *currentMoves++;
        //        move = pick_best(currentMoves++, endMoves);
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

  // range-based forを使いたいので。
  ExtMove* begin() { return moves; }
  ExtMove* end() { return endMoves; }

  // beginからendのなかでベストのスコアのものを先頭(begin)に移動させる。
  Move pick_best(ExtMove* begin, ExtMove* end)
  {
    std::swap(*begin, *std::max_element(begin, end));
    return *begin;
  }

  // CAPTUREの指し手をオーダリング
  void score_captures() 
  {
    // Position::SSE()を用いると遅い。単に取る駒の価値順に調べたほうがパフォーマンス的にもいい。
    // 歩が成る指し手もあるのでこれはある程度優先されないといけない。
    // CAPTURE系である以上、打つ指し手は除外されている。
    for (auto& m : *this)
    {
      // CAPTURES_PRO_PLUSで生成しているので歩の成りに対しては金と歩の価値の差の点数とする。
      bool pawn_promo = (m & MOVE_PROMOTE) && type_of(pos.piece_on(move_from(m))) == PAWN;
      m.value = pawn_promo ?
        (Value)(Eval::GoldValue - Eval::PawnValue) :
        (Value)Eval::PieceValueCapture[pos.piece_on(move_to(m))];

      // 盤の上のほうの段にあるほど価値があるので下の方の段に対して小さなペナルティを課す。
      // (基本的には取る駒の価値が大きいほど優先であるから..)
      m.value -= Value(1 * relative_rank(pos.side_to_move(), rank_of(move_to(m))));
    }
  }

  // QUIETの指し手をスコアリングする。
  void score_quiets()
  {
    for (auto& m : *this)
      m.value = history[move_to(m)][pos.moved_piece(m)]
      + (*counterMoveHistory)[move_to(m)][pos.moved_piece(m)];
  }

  void score_evasions()
  {
    Value see;

    for (auto& m : *this)

      // 駒を打つ指し手であるならゼロ扱い
      if (is_drop(m))
        m.value = VALUE_ZERO;

#ifdef USE_SEE

      // seeが負の指し手ならマイナスの値を突っ込んで後回しにする
      else if ((see = pos.see(m)) < VALUE_ZERO)
        m.value = see - HistoryStats::Max; // At the bottom
#endif

      // 駒を取る指し手ならseeがプラスだったということなのでプラスの符号になるようにStats::Maxを足す。
      // あとは取る駒の価値を足して、動かす駒の番号を引いておく(小さな価値の駒で王手を回避したほうが
      // 価値が高いので(例えば合駒に安い駒を使う的な…)
      else if (pos.capture(m))
        m.value = (Value)Eval::PieceValueCapture[pos.piece_on(move_to(m))]
        - Value(type_of(pos.moved_piece(m))) + HistoryStats::Max;
      else
        m.value = history[move_to(m)][pos.moved_piece(m)];
  }

  const Position& pos;

  const HistoryStats& history;
  const CounterMoveStats* counterMoveHistory;

  // node stack
  Search::Stack* ss;

  // コンストラクタで渡された、前の局面の指し手に対する応手
  Move counterMove;

  // コンストラクタで渡された探索深さ
  Depth depth;

  // RECAPUTREの指し手で移動させる先の升
  Square recaptureSquare;

  // 置換表の指し手
  Move ttMove;

  // killer move 2個 + counter move 1個 = 3個
  ExtMove killers[3];

  // ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
  Value threshold;

  // 指し手生成の段階
  Stages stage;

  // QUIETの終端、BadCaptureの終端(これはメモリの前方に向かって使っていく)。
  ExtMove *endQuiets, *endBadCaptures = moves + MAX_MOVES - 1;
  // 指し手生成バッファと、次に返す指し手、生成された指し手の末尾
  ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;
};
#endif

#endif // _MOVE_PICKER_H_
