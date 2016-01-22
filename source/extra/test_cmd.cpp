#include "../shogi.h"

// USI拡張コマンドのうち、開発上のテスト関係のコマンド。
// 思考エンジンの実行には関係しない。GitHubにはcommitしないかも。

#ifdef ENABLE_TEST_CMD

#include "all.h"

// ----------------------------------
//      USI拡張コマンド "perft"
// ----------------------------------

// perft()で用いるsolver
// cf. http://qiita.com/ak11/items/8bd5f2bb0f5b014143c8

// 通常のPERFTと、置換表を用いる高速なPERFTと選択できる。
// 後者を用いる場合は、hash keyの衝突を避けるためにHASH_KEY_BITSを128にしておくこと。
#define NORMAL_PERFT

struct PerftSolverResult {
  uint64_t nodes, captures, promotions, checks, mates;
  void operator+=(const PerftSolverResult& other) {
    nodes += other.nodes;
    captures += other.captures;
    promotions += other.promotions;
    checks += other.checks;
    mates += other.mates;
  }
};

#ifndef NORMAL_PERFT
// perftで用いる置換表
namespace Perft {
  struct TTEntry {
    void save(HASH_KEY key_, PerftSolverResult& result_)
    {
      key = key_;
      result = result_;
    }
    HASH_KEY key;
    PerftSolverResult result;
  };

  struct TranspositionTable {
    TranspositionTable() {
      entryCount = 256*1024*1024; // * sizeof(PerftSolverResult) == 15GBほど
      table = (TTEntry*)calloc( entryCount * sizeof(TTEntry) , 1);
    }
    ~TranspositionTable() { free(table); }

    TTEntry* probe(const HASH_KEY key_,int depth,bool& found)
    {
      auto key = key_ ^ DepthHash(depth); // depthの分だけhash keyを変更しておく。
      auto& tte = table[key /*.p(0)*/ & (entryCount - 1)];
      found = (tte.key == key_); // 変更前のhash keyが書かれているはず
      return &tte;
    }
  private:
    TTEntry* table;
    size_t entryCount; // TTEntryの数
  };
  TranspositionTable TT;
}

#endif

#ifdef NORMAL_PERFT
struct PerftSolver {
  template <bool Root>
  PerftSolverResult Perft(Position& pos, int depth) {
    StateInfo st;
    PerftSolverResult result = {};
    if (depth == 0) {
      result.nodes++;
      if (pos.state()->capturedType != NO_PIECE) result.captures++;
#ifdef KEEP_LAST_MOVE
      if (is_promote(pos.state()->lastMove)) result.promotions++;
#endif
      if (pos.checkers()) {
        result.checks++;
        if (pos.is_mated()) result.mates++;
      }
    } else {
      for (auto m : MoveList<LEGAL_ALL>(pos)) {
        if (Root)
          cout << ".";
        pos.do_move(m.move, st);
        result += Perft<false>(pos, depth - 1);
        pos.undo_move(m.move);
      }
    }
    return result;
  }
};
#else // 置換表を用いる高速なperft
struct PerftSolver {
  template <bool Root>
  PerftSolverResult Perft(Position& pos, int depth) {
    HASH_KEY key = pos.state()->long_key();
    bool found;

    PerftSolverResult result = {};

    auto tt = Perft::TT.probe(key, depth, found); // 置換表に登録されているか。
    if (found)
      return tt->result;

    StateInfo st;
    for (auto m : MoveList<LEGAL_ALL>(pos)) {
      if (Root)
        cout << ".";

      pos.do_move(m.move, st);
      if (depth > 1)
        result += Perft<false>(pos, depth - 1);
      else {
        result.nodes++;
        if (pos.state()->capturedType != NO_PIECE) result.captures++;
        #ifdef        KEEP_LAST_MOVE
        if (is_promote(pos.state()->lastMove)) result.promotions++;
        #endif
        if (pos.checkers()) {
          result.checks++;
          if (pos.is_mated()) result.mates++;
        }
      }
      pos.undo_move(m.move);
    }
    tt->save(key, result); // 登録されていなかったので置換表に保存する

    return result;
  }
};
#endif

// N手で到達できる局面数を計算する。成る手、取る手、詰んだ局面がどれくらい含まれているかも計算する。
void perft(Position& pos, istringstream& is)
{
  int depth = 5 ;
  is >> depth;
  cout << "perft depth = " << depth << endl;
  PerftSolver solver;
  // 局面コピーして並列的に呼び出してやるだけであとはなんとかなる。

  auto result = solver.Perft<true>(pos, depth);

  cout << endl << "nodes = " << result.nodes << " , captures = " << result.captures <<
#ifdef        KEEP_LAST_MOVE
    " , promotion = " << result.promotions <<
#endif
    " , checks = " << result.checks << " checkmates = " << result.mates << endl;
}

// ----------------------------------
//      USI拡張コマンド "test"
// ----------------------------------

// 利きの整合性のチェック
void effect_check(Position& pos)
{
#ifdef LONG_EFFECT_LIBRARY
  // 利きは、Position::set_effect()で全計算され、do_move()のときに差分更新されるが、
  // 差分更新された値がset_effect()の値と一致するかをテストする。
  using namespace LongEffect;
  ByteBoard bb[2] = { pos.board_effect[0] , pos.board_effect[1] };
  WordBoard wb = pos.long_effect;

  pos.set_effect();

  for(auto c : COLOR)
    for(auto sq : SQ)
      if (bb[c].effect(sq) != pos.board_effect[c].effect(sq))
      {
        cout << "Error effect count of " << c << " at " << sq << endl << pos << "wrong\n" << bb[c] << endl << "correct\n" << pos.board_effect[c];
        ASSERT(false);
      }

  for(auto sq : SQ)
    if (wb.dir_bw_on(sq) != pos.long_effect.dir_bw_on(sq))
    {
      cout << "Error long effect at " << sq << endl << pos << "wrong\n" << wb << endl << "correct\n" << pos.long_effect;
      ASSERT(false);
    }

#endif
}


// --- "test rp"コマンド

void random_player(Position& pos,uint64_t loop_max)
{
  pos.set_hirate();
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  Move moves[MAX_PLY]; // 局面の巻き戻し用に指し手を記憶
  int ply; // 初期局面からの手数

  for (int i = 0; i < loop_max; ++i)
  {
    for (ply = 0; ply < MAX_PLY; ++ply)
    {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

      // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 局面がおかしくなっていないかをテストする
      ASSERT_LV3(is_ok(pos));

      // 利きの整合性のテスト(重いのでテストが終わったらコメントアウトする)
      effect_check(pos);

      pos.check_info_update();

      // ここで生成された指し手がすべて合法手であるかテストをする
      for (auto m : mg)
      {
        ASSERT_LV3(pos.pseudo_legal(m.move));
        ASSERT_LV2(pos.legal(m.move));
      }

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[rand() % mg.size()].move;

      pos.do_move(m, state[ply]);
      moves[ply] = m;
    }
    // 局面を巻き戻してみる(undo_moveの動作テストを兼ねて)
    while (ply > 0)
      pos.undo_move(moves[--ply]);

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 1000) == 0)
      cout << ".";
  }
}

// ランダムプレイヤー(指し手をランダムに選ぶプレイヤー)による自己対戦テスト
// これを1000万回ほどまわせば、指し手生成などにバグがあればどこかで引っかかるはず。

void random_player_cmd(Position& pos, istringstream& is)
{
  uint64_t loop_max = 100000000; // 1000万回
  is >> loop_max;
  cout << "Random Player test , loop_max = " << loop_max << endl;
  random_player(pos, loop_max);
  cout << "finished." << endl;
}

// --- "test genchecks"コマンド

// ランダムプレイヤーに合法手を生成させて、そのなかの王手になる指し手が
// 王手生成ルーチンで生成した指し手と合致するかを判定して、王手生成ルーチンの正しさを証明する。
void test_genchecks(Position& pos, uint64_t loop_max)
{
  pos.init();
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  Move moves[MAX_PLY]; // 局面の巻き戻し用に指し手を記憶
  int ply; // 初期局面からの手数

  for (int i = 0; i < loop_max; ++i)
  {
    for (ply = 0; ply < MAX_PLY; ++ply)
    {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

                               // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 局面がおかしくなっていないかをテストする
      ASSERT_LV3(is_ok(pos));

      pos.check_info_update();
      MoveList<CHECKS_ALL> mc(pos);

      // ここで生成された指し手と王手生成ルーチンで生成した指し手とが王手する指し手について一致するかをテストする。
      for (auto m : mg)
      {
        if (pos.gives_check(m))
        {
          for (auto m2 : mc)
            if (m2.move == m)
              goto Exit;

          cout << endl << pos << "not found : move = " << m.move << endl;
          MoveList<CHECKS_ALL> mc2(pos); // ここにブレークポイントを仕掛けてデバッグする。
          ASSERT_LV1(false);
        }
      Exit:;
      }

      // 逆もチェックする。
      for (auto m : mc)
      {
        if (!pos.gives_check(m))
        {
          cout << endl << pos << "not checks : move = " << m.move << endl;
          MoveList<CHECKS_ALL> mc2(pos); // ここにブレークポイントを仕掛けてデバッグする。
          ASSERT_LV1(false);
        }
      }


      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[rand() % mg.size()].move;

      pos.do_move(m, state[ply]);
      moves[ply] = m;
    }
    // 局面を巻き戻してみる(undo_moveの動作テストを兼ねて)
    while (ply > 0)
      pos.undo_move(moves[--ply]);

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 1000) == 0)
      cout << ".";
  }
}

void test_genchecks(Position& pos, istringstream& is)
{
  uint64_t loop_max = 100000000; // 1000万回
  is >> loop_max;
  cout << "Generate Checks test , loop_max = " << loop_max << endl;
  test_genchecks(pos, loop_max);
  cout << "finished." << endl;
}


// --- "test cm"コマンド

// 協力詰め。n手で協力詰めで詰むかを調べる。

// 再帰的に呼び出される
// depth = 残り探索深さ
void cooperation_mate(Position& pos, int depth)
{
  // moves_from_start_pretty()がKEEP_LAST_MOVEを要求する。
#ifdef  KEEP_LAST_MOVE

  StateInfo st;
  for (auto m : MoveList<LEGAL_ALL>(pos))
  {
    pos.do_move(m.move, st);

    if (pos.is_mated())
    {
      // 後手の詰みなら手順を表示する。先手の詰みは必要ない。
      if (pos.side_to_move() == WHITE)
        cout << pos << pos.moves_from_start_pretty() << endl; // 開始局面からそこまでの手順
    } else {
      // 残り探索深さがあるなら再帰的に探索する。
      if (depth > 1)
        cooperation_mate(pos, depth - 1);
    }
    pos.undo_move(m.move);
  }
#endif
}

// 協力詰めコマンド
// 例)
//  position sfen 9/9/9/3bkb3/9/3+R1+R3/9/9/9 b - 1
//  position sfen 9/6b2/7k1/5b3/7sL/9/9/9/9 b R - 1
//  などで
//  test cm 5  (5手詰め)
void cooperation_mate_cmd(Position& pos, istringstream& is)
{
  int depth = 5;
  is >> depth;
  cout << "Cooperation Mate test , depth = " << depth << endl;
  cooperation_mate(pos, depth); // 与えられた局面からdepth深さの協力詰みを探す
  cout << "finished." << endl;
}

// --- "s" 指し手生成テストコマンド
void generate_moves_cmd(Position& pos)
{
  cout << "Generate Moves Test.." << endl;
//  pos.set("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1");
  auto start = now();
  pos.check_info_update();

  // 試行回数
  const int64_t n = 30000000;
  for (int i = 0; i < n; ++i)
  {
    if (pos.checkers())
      MoveList<EVASIONS> ml(pos);
    else
      MoveList<NON_EVASIONS> ml(pos);
  }

  auto end = now();

  // 局面と生成された指し手を出力しておく。
  cout << pos;

  if (pos.checkers())
    for (auto m : MoveList<EVASIONS>(pos))
      cout << m.move << ' ';
  else
    for (auto m : MoveList<NON_EVASIONS>(pos))
      cout << m.move << ' ';

  cout << endl << (1000 * n / (end - start)) << " times per second." << endl;
}

// --- 駒の優劣関係などのテスト

void test_hand()
{
  cout << "Test Hand start : is_superior()" << endl;
  
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
    {
      Hand h1 = (Hand)0, h2 = (Hand)0;
      for (int k = 0; k < 7; ++k)
      {
        int bit = (1 << k);
        if (i & bit) add_hand(h1, (Piece)(k + 1), 1);
        if (j & bit) add_hand(h2, (Piece)(k + 1), 1);
      }

      // h1のほうがh2より優れているか。
      bool is_equal_or_superior =
        hand_count(h1, PAWN) >= hand_count(h2, PAWN) &&
        hand_count(h1, LANCE) >= hand_count(h2, LANCE) &&
        hand_count(h1, KNIGHT) >= hand_count(h2, KNIGHT) &&
        hand_count(h1, SILVER) >= hand_count(h2, SILVER) &&
        hand_count(h1, BISHOP) >= hand_count(h2, BISHOP) &&
        hand_count(h1, ROOK) >= hand_count(h2, ROOK) &&
        hand_count(h1, GOLD) >= hand_count(h2, GOLD);

      if (is_equal_or_superior != hand_is_equal_or_superior(h1, h2))
      {
        cout << "error " << h1 << " & " << h2 << endl;
      }
    }

  cout << "Test Hand end." << endl;
}

// 棋譜を丸読みして、棋譜の指し手が生成した合法手のなかに含まれるかをテストする
void test_read_record(Position& pos, istringstream& is)
{
  string filename = "records.sfen";
  is >> filename;
  cout << "read " << filename << endl;

  fstream fs;
  fs.open(filename, ios::in | ios::binary);
  if (fs.fail())
  {
    cout << "read error.." << endl;
    return;
  }

  // ファイル内の行番号
  uint64_t line_no = 0;

  string line;
  while (getline(fs, line))
  {
    ++line_no;
    if ((line_no % 100) == 0) cout << '.'; // 100行おきに'.'を一つ出力。

    stringstream ss(line);
    string token;
    string sfen;

    ss >> token;

    if (token == "startpos")
    {
      // 初期局面として初期局面のFEN形式の入力が与えられたとみなして処理する。
      sfen = SFEN_HIRATE;
      ss >> token; // もしあるなら"moves"トークンを消費する。
    } else if(token == "sfen")
      while (ss >> token && token != "moves")
        sfen += token;

    auto& SetupStates = Search::SetupStates;
    SetupStates = Search::StateStackPtr(new aligned_stack<StateInfo>());

    pos.set(sfen);

    while (ss >> token)
    {
      for (auto m : MoveList<LEGAL_ALL>(pos))
      {
        if (token == to_usi_string(m))
        {
          SetupStates->push(StateInfo());
          pos.do_move(m, SetupStates->top());
          goto Ok;
        }
      }
      // 生成した合法手にない文字列であった。

      cout << "\nError at line number = " << line_no << " : illigal moves = " << token << endl;
      cout << "> " << line << endl; // この行、まるごと表示させておく。
      break;

    Ok:;
    }

  }
  cout << "done." << endl;
}

// --- "test unit"コマンド

// 単体テスト

// これを実行するときはASSERT_LV5にしておくといいと思う。
void unit_test(Position& pos, istringstream& is)
{
  cout << "UnitTest start" << endl;
  bool success_all = true;
  auto check = [&](bool success) {
    cout << (success ? "..success" : "..failed") << endl;
    success_all &= success;
  };

  // -- 色んな関数
  {
    cout << "> verious function ";

    // -- long effect
    bool success = true;
    for (auto pc : Piece())
    {
      auto pt = type_of(pc);
      success &= (pt == LANCE || pt == BISHOP || pt == ROOK || pt == HORSE || pt == DRAGON) == (has_long_effect(pc));
    }
    check(success);
  }

  // hash key
  // この値が変わると定跡DBがhitしなくなってしまうので変えてはならない。
  {
    cout << "> hash key check ";
    pos.set_hirate();
    check( pos.state()->key() == UINT64_C(0x75a12070b8bd438a));
  }

  // perft
  {
    // 最多合法手局面
    const string POS593 = "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1";
    cout << "> genmove sfen = " << POS593;
    pos.set(POS593);
    auto mg = MoveList<LEGAL_ALL>(pos);
    cout << " , moves = " << mg.size();
    check( mg.size() == 593);

    cout << "> perft depth 6 ";
    pos.set_hirate();
    auto result = PerftSolver().Perft<true>(pos,6);
    check(  result.nodes == 547581517 && result.captures == 3387051
#ifdef      KEEP_LAST_MOVE
      && result.promotions == 1588324
#endif
      && result.checks == 1730177 && result.mates == 0);
  }

  // random pleyer
  {
    cout << "> random player 100000 ";
    random_player(pos, 100000);

    // ASSERTに引っかからずに抜けたということは成功だという扱い。
    check(true);
  }

  cout << "UnitTest end : " << (success_all ? "success" : "failed") << endl;
}

void test_cmd(Position& pos, istringstream& is)
{
  std::string param;
  is >> param;
  if (param == "unit") unit_test(pos, is); // 単体テスト
  else if (param == "rp") random_player_cmd(pos,is); // ランダムプレイヤー
  else if (param == "cm") cooperation_mate_cmd(pos, is); // 協力詰めルーチン
  else if (param == "checks") test_genchecks(pos, is); // 王手生成ルーチンのテスト
  else if (param == "hand") test_hand(); // 手駒の優劣関係などのテスト
  else if (param == "records") test_read_record(pos,is); // 棋譜の読み込みテスト 
  else {
    cout << "test unit               // UnitTest" << endl;
    cout << "test rp                 // Random Player" << endl;
    cout << "test cm [depth]         // Cooperation Mate" << endl;
    cout << "test checks             // Generate Checks Test" << endl;
    cout << "test records [filename] // read records.sfen Test" << endl;
  }
}

#endif // ENABLE_TEST_CMD
