#ifndef _SEARCH_H_
#define _SEARCH_H_

#include <atomic>

#include "position.h"
#include "misc.h"

// 探索関係
namespace Search {

  // root(探索開始局面)での指し手として使われる。それぞれのroot moveに対して、
  // その指し手で進めたときのscore(評価値)とPVを持っている。(PVはfail lowしたときには信用できない)
  // scoreはnon-pvの指し手では-VALUE_INFINITEで初期化される。
  struct RootMove
  {
    explicit RootMove(Move m) : pv(1,m) {}

    // 今回の(反復深化の)iterationでの探索結果のスコア
    Value score = -VALUE_INFINITE;

    // 前回の(反復深化の)iterationでの探索結果のスコア
    // 次のiteration時の探索窓の範囲を決めるときに使う。
    Value previousScore = -VALUE_INFINITE;

    // この指し手で進めたときのpv
    std::vector<Move> pv;
  };

  // goコマンドでの探索時に用いる、持ち時間設定などが入った構造体
  struct LimitsType {

    // コンストラクタで明示的にゼロクリア(MSVCがPODでないと破壊することがあるため)
    LimitsType() { memset(this, 0, offsetof(LimitsType, startTime)); }

    // 時間制御を行うのか。
    // 詰み専用探索、思考時間0、探索深さが指定されている、探索ノードが指定されている、思考時間無制限
    // であるときは、時間制御に意味がないのでやらない。
    bool use_time_management() const {
      return !(mate | movetime | depth | nodes | infinite);
    }

    // root(探索開始局面)で、探索する指し手集合。特定の指し手を除外したいときにここから省く
    std::vector<Move> searchmoves;

    // 残り時間(ms換算で)
    int time[COLOR_NB];

    // 秒読み(ms換算で)
    int byoyomi[COLOR_NB];

    // depth    : 探索深さ固定(0以外を指定してあるなら)
    // movetime : 思考時間固定(0以外が指定してあるなら) : 単位は[ms]
    // mate     : 詰み専用探索(USIの'go mate'コマンドを使ったとき)
    //  詰み探索モードのときは、ここに思考時間が指定されている。
    //  この思考時間いっぱいまで考えて良い。
    // infinite : 思考時間無制限かどうかのフラグ。非0なら無制限。
    // ponder   : ponder中であるかのフラグ
    //  これがtrueのときはbestmoveがあっても探索を停止せずに"ponderhit"か"stop"が送られてきてから停止する。
    //  ※ ただし今回用の探索時間を超えていれば、stopOnPonderhitフラグをtrueにしてあるのでponderhitに対して即座に停止する。
    int depth, movetime, mate, infinite, ponder;

    // 今回のgoコマンドでの探索ノード数
    int64_t nodes;

    // 入玉ルール設定
    EnteringKingRule enteringKingRule;

    // ---- ↑ここまでコンストラクタでゼロ初期化↑ ----

    // goコマンドで探索を開始した時刻。
    TimePoint startTime;
  };

  struct SignalsType {
    std::atomic_bool stop; // これがtrueになったら探索を強制終了すること。
  };

  typedef std::unique_ptr<std::stack<StateInfo>> StateStackPtr;

  extern SignalsType Signals;
  extern LimitsType Limits;
  extern StateStackPtr SetupStates;

  // 探索部の初期化。
  void init();

  // 探索部のclear。
  // 置換表のクリアなど時間のかかる探索の初期化処理をここでやる。isreadyに対して呼び出される。
  void clear();

} // end of namespace Search

#endif // _SEARCH_H_