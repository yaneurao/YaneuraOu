#include "all.h"

#if 1
// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos, istringstream& is)
{
  cout << Effect8::Directions(1);
  return ;

  pos.set("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1");

  auto t1 = now();

  pos.check_info_update();

  const int64_t n = 20000000;
  for (int i = 0; i < n; ++i)
  {
    MoveList<NON_EVASIONS> ml(pos);
  }

  auto t2 = now();
  cout << (1000*n/(t2 - t1)) << endl;

  cout << pos;

  MoveList<NON_EVASIONS> ml(pos);
  cout << ml.size() << endl;


  /*
  Key key = DepthHash(0);//.p(0);
  cout << std::hex << "HashKey = " << key << endl;
  */

/*
  CheckInfo ci(pos);
  for (auto m : MoveList<CHECKS>(pos,ci))
    cout << m.move << ' ';
    */

  return;



  /*
  Square sq = SQ_28;
  Rank r = rank_of(sq);
  cout << r;

  return;
  */

  //cout << pos;

  //cout << pos.king_square(BLACK);
  //cout << pos.king_square(WHITE);

    auto SetupStates = Search::StateStackPtr(new std::stack<StateInfo>);
    for (int i = 0; i < 100; ++i)
    {
      cout << pos;

      MoveList<LEGAL_ALL> mg(pos);
      for (auto m : mg)
        cout << m.move << ' ';
      cout << endl;

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move move = mg.begin()[rand() % mg.size()].move;
      cout << "moved by " << move << endl;

      SetupStates->push(StateInfo());
      pos.do_move(move, SetupStates->top());
    }

}
#endif

void f(){

  Position pos;

  auto SetupStates = Search::StateStackPtr(new std::stack<StateInfo>);

  //  Position pos;
  for (int i = 0; i < 100; ++i)
  {
    cout << pos;
    //    cout << toHandKind(pos.hand_of(BLACK)) << " / " << toHandKind(pos.hand_of(WHITE)) << endl;

    MoveList<LEGAL_ALL> mg(pos);

    for (auto m : mg)
      cout << m.move << ' ';
    cout << endl;

    // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
    Move move = mg.begin()[rand() % mg.size()].move;
    cout << "moved by " << move << endl;

    SetupStates->push(StateInfo());
    pos.do_move(move, SetupStates->top());
  }

  /*
    Position pos;
    const int move_times = MAX_PLY;
    Move moves[move_times];
    Move move;
    for (int i = 0; i < move_times; ++i)
    {
      cout << pos; // 現在の盤面を表示
      MoveList<LEGAL_ALL> mg(pos);
      for (auto m : mg)
        cout << m.move << ' ';
      cout << endl;
      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      move = mg.begin()[rand() % mg.size()].move;
      pos.do_move(move);
      cout << "moved by " << move << endl;
      moves[i] = move;
    }

    // 局面を1手ずつ戻しながら表示する。
    for (int i = move_times - 1; i >= 0; --i)
    {
      move = moves[i];
      cout << "unmoved by " << move << endl;
      pos.undo_move(move);
      cout << pos;
    }
  */


  // 3手深さで探索する
//  search(pos,SCORE_MIN , SCORE_MAX,ONE_PLY * 3);

  string s;
  cin >> s;
}
