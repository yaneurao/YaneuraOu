#include "../shogi.h"

// 学習関係のルーチン
// 1) 棋譜の自動生成
// 2) 生成した棋譜からの評価関数パラメーターの学習
// 3) 定跡の自動生成
// 4) 局後自動検討モード
// etc..


#if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

// packされたsfenを書き出す
#define WRITE_PACKED_SFEN

// search()のleaf nodeまでの手順が合法手であるかを検証する。
#define TEST_LEGAL_LEAF

// packしたsfenをunpackして元の局面と一致するかをテストする。
// →　十分テストしたのでもう大丈夫やろ…。
//#define TEST_UNPACK_SFEN

// 棋譜を生成するときに一定手数の局面まで定跡を用いる機能
// これはOptions["BookMoves"]の値が反映される。この値が0なら、定跡を用いない。
// 用いる定跡は、Options["BookFile"]が反映される。

// 2駒の入れ替えを5手に1回ぐらいの確率で行なう。
#define USE_SWAPPING_PIECES


#include <sstream>
#include <fstream>

#include "../misc.h"
#include "../thread.h"
#include "../position.h"
#include "../extra/book.h"

using namespace std;

extern void is_ready(Position& pos);
extern Book::MemoryBook book;

namespace Learner
{

// いまのところ、やねうら王2016Midしか、このスタブを持っていない。
extern pair<Value, vector<Move> >  search(Position& pos, Value alpha, Value beta, int depth);
extern pair<Value, vector<Move> > qsearch(Position& pos, Value alpha, Value beta);


// -----------------------------------
//    局面のファイルへの書き出し
// -----------------------------------

// Sfenを書き出して行くためのヘルパクラス
struct SfenWriter
{
  SfenWriter(int thread_num,u64 sfen_count_limit_)
  {
    sfen_count_limit = sfen_count_limit_;
    sfen_buffers.resize(thread_num);
    fs.open("generated_kifu.sfen", ios::out | ios::binary);
  }

  // この行数ごとにファイルにflushする。
  // 探索深さ3なら1スレッドあたり1秒で1000局面ほど作れる。
  // 40コア80HTで秒間10万局面。1日で80億ぐらい作れる。
  // 80億=8G , 1局面100バイトとしたら 800GB。
  
  const size_t FILE_WRITE_INTERVAL = 1000;

  // 停止信号。これがtrueならslave threadは終了。
  bool is_stop() const
  {
    return sfen_count > sfen_count_limit;
  }

#ifdef  WRITE_PACKED_SFEN
  struct PackedSfenValue
  {
    u8 data[34];
  };

  // 局面と評価値をペアにして1つ書き出す(packされたsfen形式で)
  void write(size_t thread_id, u8 data[32], int16_t value)
  {
    // スレッドごとにbufferを持っていて、そこに追加する。
    // bufferが溢れたら、ファイルに書き出す。

    auto& buf = sfen_buffers[thread_id];
    
    PackedSfenValue ps;
    memcpy(ps.data, data, 32);
    memcpy(ps.data + 32, &value, 2);
    
    buf.push_back(ps);
    if (buf.size() >= FILE_WRITE_INTERVAL)
    {
      std::unique_lock<Mutex> lk(mutex);

      for (auto line : buf)
        fs.write((const char*)line.data,34);

      fs.flush();

      sfen_count += buf.size();
      buf.clear();

      // 棋譜を書き出すごとに'.'を出力。
      cout << ".";
    }
  }
#else
  // 局面を1行書き出す
  void write(size_t thread_id, string line)
  {
    // スレッドごとにbufferを持っていて、そこに追加する。
    // bufferが溢れたら、ファイルに書き出す。

    auto& buf = sfen_buffers[thread_id];
    buf.push_back(line);
    if (buf.size() >= FILE_WRITE_INTERVAL)
    {
      std::unique_lock<Mutex> lk(mutex);

      for (auto line : buf)
        fs << line;

      fs.flush();
      sfen_count += buf.size();
      buf.clear();

      // 棋譜を書き出すごとに'.'を出力。
      cout << ".";
    }
  }
#endif

private:
  // ファイルに書き出す前に排他するためのmutex
  Mutex mutex;

  fstream fs;

  // ファイルに書き出す前のバッファ
#ifdef  WRITE_PACKED_SFEN
  vector<vector<PackedSfenValue>> sfen_buffers;
#else
  vector<vector<string>> sfen_buffers;
#endif

  // 生成したい局面の数
  u64 sfen_count_limit;

  // 生成した局面の数
  u64 sfen_count;

};

// -----------------------------------
//  棋譜を生成するworker(スレッドごと)
// -----------------------------------

// ループ回数の上限とカウンター
// 排他してないけど、まあいいや。回数、そこまで厳密でなくて問題ない。
volatile u64 g_loop_max;
volatile u64 g_loop_count;

ASYNC_PRNG prng(20160101);

//  thread_id    = 0..Threads.size()-1
//  search_depth = 通常探索の探索深さ
void gen_sfen_worker(size_t thread_id, int search_depth, SfenWriter& sw)
{
  const int MAX_PLY = 256; // 256手までテスト
  StateInfo state[MAX_PLY + 64]; // StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
  int ply; // 初期局面からの手数
  Move m = MOVE_NONE;

  // 定跡の指し手を用いるモード
  int book_ply = Options["BookMoves"];

  // g_loop_maxになるまで繰り返し
  while (++g_loop_count <= g_loop_max)
  {
    auto& pos = Threads[thread_id]->rootPos;
    pos.set_hirate();
    
    // Positionに対して従属スレッドの設定が必要。
    // 並列化するときは、Threads (これが実体が vector<Thread*>なので、
    // Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
    auto th = Threads[thread_id];
    pos.set_this_thread(th);

    //    cout << endl;
    for (ply = 0; ply < MAX_PLY; ++ply)
    {
      // mate1ply()の呼び出しのために必要。
      pos.check_info_update();

      if (pos.is_mated())
        break;

      // 定跡を使用するのか？
      if (pos.game_ply() <= book_ply)
      {
        auto it = book.find(pos);
        if (it != book.end() && it->second.size() != 0)
        {
          // 定跡にhitした。it->second->size()!=0をチェックしておかないと
          // 指し手のない定跡が登録されていたときに困る。

          const auto& move_list = it->second;

          const auto& move = move_list[prng.rand(move_list.size())];
          auto bestMove = move.bestMove;
          // この指し手に不成があってもLEGALであるならこの指し手で進めるべき。
          if (pos.pseudo_legal(bestMove) && pos.legal(bestMove))
          {
            // この指し手で1手進める。
            m = bestMove;
//            cout << m << ' ';
            goto DO_MOVE;
          }
        }
      }

      {
        // 3手読みの評価値とPV(最善応手列)
        auto pv_value1 = Learner::search(pos, -VALUE_INFINITE, VALUE_INFINITE, search_depth);
        auto value1 = pv_value1.first;
        auto pv1 = pv_value1.second;

#if 0
        // 0手読み(静止探索のみ)の評価値とPV(最善応手列)
        auto pv_value2 = Learner::qsearch(pos, -VALUE_INFINITE, VALUE_INFINITE);
        auto value2 = pv_value2.first;
        auto pv2 = pv_value2.second;
#endif

        // 上のように、search()の直後にqsearch()をすると、search()で置換表に格納されてしまって、
        // qsearch()が置換表にhitして、search()と同じ評価値が返るので注意。

        // 局面のsfen,3手読みでの最善手,0手読みでの評価値
        // これをファイルか何かに書き出すと良い。
        //      cout << pos.sfen() << "," << value1 << "," << value2 << "," << endl;


#ifdef WRITE_PACKED_SFEN
        u8 data[32];
        // packを要求されているならpackされたsfenとそのときの評価値を書き出す。
        pos.sfen_pack(data);
        sw.write(thread_id, data, value1);

#ifdef TEST_UNPACK_SFEN

        // sfenのpack test
        // pack()してunpack()したものが元のsfenと一致するのかのテスト。

  //      pos.sfen_pack(data);
        auto sfen = pos.sfen_unpack(data);
        auto pos_sfen = pos.sfen();

        // 手数の部分の出力がないので異なる。末尾の数字を消すと一致するはず。
        auto trim = [](std::string& s)
        {
          while (true)
          {
            auto c = *s.rbegin();
            if (c < '0' || '9' < c)
              break;
            s.pop_back();
          }
        };
        trim(sfen);
        trim(pos_sfen);

        if (sfen != pos_sfen)
        {
          cout << "Error: sfen packer error\n" << sfen << endl << pos_sfen << endl;
        }
#endif

#else // WRITE_PACKED_SFEN

        // sfenとそのときの評価値を書き出す。
        string line = pos.sfen() + "," + to_string(value1) + "\n";
        sw.write(thread_id, line);
#endif


#if 0
        // デバッグ用に局面と読み筋を表示させてみる。
        cout << pos;
        cout << "search() PV = ";
        for (auto pv_move : pv1)
          cout << pv_move << " ";
        cout << endl;

        // 静止探索のpvは存在しないことがある。(駒の取り合いがない場合など)　その場合は、現局面がPVのleafである。
        cout << "qsearch() PV = ";
        for (auto pv_move : pv2)
          cout << pv_move << " ";
        cout << endl;

#endif

#ifdef TEST_LEGAL_LEAF
        // デバッグ用の検証として、
        // PVの指し手でleaf nodeまで進めて、非合法手が混じっていないかをテストする。
        auto go_leaf_test = [&](auto pv) {
          int ply2 = ply;
          for (auto m : pv)
          {
            // 非合法手はやってこないはずなのだが。
            if (!pos.pseudo_legal(m) || !pos.legal(m))
            {
              cout << pos << m << endl;
              ASSERT_LV3(false);
            }
            pos.do_move(m, state[ply2++]);
            pos.check_info_update();
          }
          // leafに到達
          //      cout << pos;

          // 巻き戻す
          auto pv_r = pv;
          std::reverse(pv_r.begin(), pv_r.end());
          for (auto m : pv_r)
            pos.undo_move(m);
        };

        go_leaf_test(pv1); // 通常探索のleafまで行くテスト
  //      go_leaf_test(pv2); // 静止探索のleafまで行くテスト
#endif

        // 3手読みの指し手で局面を進める。
        m = pv1[0];
      }

#ifdef      USE_SWAPPING_PIECES
      // 2駒をときどき入れ替える機能
      
      // このイベントは、王手がかかっていない局面において一定の確率でこの指し手が発生する。
      // 王手がかかっていると王手回避しないといけないので良くない。
      // 二枚とも歩が選ばれる可能性がそこそこあるため、1/5に設定しておく。
      if (prng.rand(5) == 0 && !pos.in_check())
      {
        for (int retry = 0; retry < 10; ++retry)
        {
          // 手番側の駒を2駒入れ替える。

          // 与えられたBitboardからランダムに1駒を選び、そのSquareを返す。
          auto get_one = [](Bitboard pieces)
          {
            // 駒の数
            int num = pieces.pop_count();
            // 何番目かの駒
            int n = (int)prng.rand(num) + 1;
            Square sq = SQ_NB;
            for (int i = 0; i < n; ++i)
              sq = pieces.pop();
            return sq;
          };

          // この升の2駒を入れ替える。
          auto pieces = pos.pieces(pos.side_to_move());
          auto sq1 = get_one(pieces);
          // sq1を除くbitboard
          auto sq2 = get_one(pieces ^ sq1);

        // sq2は王しかいない場合、SQ_NBになるから、これを調べておく
        // この指し手に成功したら、それはdo_moveの代わりであるから今回、do_move()は行わない。

          if (sq2 != SQ_NB
            && pos.do_move_by_swapping_pieces(sq1, sq2))
          {
#if 1
            // 検証用のassert
            if (!is_ok(pos))
              cout << pos << sq1 << sq2;
#endif
            goto DO_MOVE_FINISH;
          }
        }
      }
#endif
    DO_MOVE:;
      pos.do_move(m, state[ply]);

    DO_MOVE_FINISH:;

    }
  }
}

// -----------------------------------
//    棋譜を生成するコマンド(master)
// -----------------------------------

// 棋譜を生成するコマンド
void gen_sfen(Position& pos, istringstream& is)
{
  // 生成棋譜の個数 default = 80億局面(Ponanza仕様)
  u64 loop_max = 8000000000UL;

  // スレッド数(これは、USIのsetoptionで与えられる)
  u32 thread_num = Options["Threads"];

  // 探索深さ
  int search_depth = 3;

  is >> search_depth >> loop_max;

  std::cout << "gen_sfen : "
    << "search_depth = " << search_depth
    << " , loop_max = " << loop_max
    << " , thread_num (set by USI setoption) = " << thread_num
    << endl;

  // 評価関数の読み込み等
  is_ready(pos);

  // 途中でPVの出力されると邪魔。
  Search::Limits.silent = true;

  SfenWriter sw(thread_num,loop_max);

  // スレッド数だけ作って実行。
  g_loop_max = loop_max;
  g_loop_count = 0;

  vector<std::thread> threads;
  for (size_t i = 0; i < thread_num; ++i)
  {
    threads.push_back( std::thread([i, search_depth,&sw] { gen_sfen_worker(i, search_depth, sw);  }));
  }

  // すべてのthreadの終了待ち

  for (auto& th:threads)
  {
    th.join();
  }

  std::cout << "gen_sfen finished." << endl;
}

} // namespace Learner

#endif // EVAL_LEARN
