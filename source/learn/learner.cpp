#include "../shogi.h"

// 学習関係のルーチン
// 1) 棋譜の自動生成
// 2) 生成した棋譜からの評価関数パラメーターの学習
// 3) 定跡の自動生成
// 4) 局後自動検討モード
// etc..


#if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

#include <sstream>
#include <fstream>

#include "../misc.h"
#include "../thread.h"
#include "../position.h"

using namespace std;

extern void is_ready(Position& pos);

namespace Learner
{

// いまのところ、やねうら王2016Midしか、このスタブを持っていない。
extern pair<Value, vector<Move> >  search(Position& pos, Value alpha, Value beta, int depth);
extern pair<Value, vector<Move> > qsearch(Position& pos, Value alpha, Value beta);

// -----------------------------------
//        局面の圧縮・解凍
// -----------------------------------

const struct HuffmanedPiece
{
  int code; // どうコード化されるか
  int bits; // 何bit専有するのか
};

// ビットストリームを扱うクラス
// 局面の符号化を行なうときに、これがあると便利
struct BitStream
{
  // データを格納するメモリを事前にセットする。
  // そのメモリは0クリアされているものとする。
  void  set_data(u8* data_) { data = data_; reset(); }

  // set_data()で渡されたポインタの取得。
  u8* get_data() const { return data; }

  // カーソルの取得。
  int get_cursor() const { return bit_cursor; }

  // カーソルのリセット
  void reset() { bit_cursor = 0; }

  // ストリームに1bit書き出す。
  // bは非0なら1を書き出す。0なら0を書き出す。
  FORCE_INLINE void write_one_bit(int b)
  {
    if (b)
      data[bit_cursor / 8] |= 1 << (bit_cursor & 7);

    ++bit_cursor;
  }

  // ストリームから1ビット取り出す。
  FORCE_INLINE int read_one_bit()
  {
    int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
    ++bit_cursor;

    return b;
  }

  // nビットのデータを書き出す
  // データはdの下位から順に書き出されるものとする。
  void write_n_bit(int d, int n)
  {
    for (int i = 0; i < n; ++i)
      write_one_bit(d & (1 << i));
  }

  // nビットのデータを読み込む
  // write_n_bit()の逆変換。
  int read_n_bit(int n)
  {
    int result = 0;
    for (int i = 0; i < n; ++i)
      result |= read_one_bit() ? (1 << i) : 0;

    return result;
  }

private:
  // 次に読み書きすべきbit位置。
  int bit_cursor;

  // データの実体
  u8* data;
};


//  ハフマン符号化
//   ※　 なのはminiの符号化から、変換が楽になるように単純化。
//
//   盤上の1升(NO_PIECE以外) = 2～6bit ( + 成りフラグ1bit+ 先後1bit )
//   手駒の1枚               = 1～5bit ( + 成りフラグ1bit+ 先後1bit )
//
//    空     xxxxx0 + 0    (none)
//    歩     xxxx01 + 2    xxxx0 + 2
//    香     xx0011 + 2    xx001 + 2
//    桂     xx1011 + 2    xx101 + 2
//    銀     xx0111 + 2    xx011 + 2
//    金     x01111 + 2    x0111 + 2
//    角     011111 + 2    01111 + 2
//    飛     111111 + 2    11111 + 2
//
// すべての駒が盤上にあるとして、
//     空 81 - 40駒 = 41升 = 41bit
//     歩      4bit*18駒   = 72bit
//     香      6bit* 4駒   = 24bit
//     桂      6bit* 4駒   = 24bit
//     銀      6bit* 4駒   = 24bit            
//     金      7bit* 4駒   = 28bit
//     角      7bit* 2駒   = 14bit
//     飛      8bit* 2駒   = 16bit
//                          -------
//                          241bit + 1bit(手番) + 7bit×2(王の位置先後) - 2(王の升2つ) = 256bit
//
// 盤上の駒が手駒に移動すると盤上の駒が空になるので盤上のその升は1bitで表現でき、
// 手駒は、盤上の駒より1bit少なく表現できるので結局、全体のbit数に変化はない。
// ゆえに、この表現において、どんな局面でもこのbit数で表現できる。
// 金は成りはないのでここで4bit減らすことは出来るがコードがややこしくなるのでこれはこれでいいや。

HuffmanedPiece huffman_table[] =
{
  {0x00,1}, // NO_PIECE
  {0x01,2}, // PAWN
  {0x03,4}, // LANCE
  {0x0b,4}, // KNIGHT
  {0x07,4}, // SILVER
  {0x1f,6}, // BISHOP
  {0x3f,6}, // ROOK
  {0x0f,5}, // GOLD
};

// sfenを圧縮/解凍するためのクラス
// sfenはハフマン符号化をすることで256bit(32bytes)にpackできる。
// このことはなのはminiにより証明された。上のハフマン符号化である。
struct SfenPacker
{
  // sfenをpackしてdata[32]に格納する。
  void pack(const Position& pos)
  {
    memset(data, 0, sizeof(data));
    stream.set_data(data);

    // 手番
    stream.write_one_bit((pos.side_to_move() == BLACK) ? 0 : 1);

    // 先手玉、後手玉の位置、それぞれ7bit
    for(auto c : COLOR)
      stream.write_n_bit(pos.king_square(c), 7);

    // 盤面の駒は王以外はそのまま書き出して良し！
    for (auto sq : SQ)
    {
      // 盤上の玉以外の駒をハフマン符号化して書き出し
      Piece pc = pos.piece_on(sq);
      if (type_of(pc) == KING)
        continue;
      write_board_piece_to_stream(pc);

      // 手駒をハフマン符号化して書き出し
      for(auto c: COLOR)
        for (Piece pr = PAWN; pr < KING; ++pr)
        {
          int n = hand_count(pos.hand_of(c), pr);

          // この駒、n枚持ってるよ
          for(int i=0;i<n;++n)
            write_hand_piece_to_stream(make_piece(c,pr));
        }

      // 綺麗に書けた..気がする。

      // 全部で256bitのはず。(普通の盤面であれば)
      ASSERT_LV3(stream.get_cursor() == 256);
    }
  }

  // data[32]をsfen化して返す。
  string unpack()
  {
    stream.set_data(data);

    // 盤上の81升
    Piece board[81];
    memset(board, 0, sizeof(board));

    // 手番
    Color turn = (stream.read_one_bit()==0) ? BLACK : WHITE;

    // まず玉の位置
    for (auto c : COLOR)
      board[stream.read_n_bit(7)] = make_piece(c, KING);

    // 盤上の駒
    for (auto sq : SQ)
    {
      // すでに玉がいるようだ
      if (type_of(board[sq]) == KING)
        continue;

      board[sq] = read_board_piece_from_stream();
    }

    // 手駒
    Hand hand[2] = { HAND_ZERO,HAND_ZERO };
    while (stream.get_cursor() != 256)
    {
      // 256になるまで手駒が格納されているはず
      auto pc = read_hand_piece_from_stream();
      add_hand(hand[(color_of(pc)==BLACK)?0:1], type_of(pc));
    }

    // boardとhandが確定した。これで局面を構築できる…かも。
    // Position::sfen()は、board,hand,side_to_move,game_plyしか参照しないので
    // 無理やり代入してしまえば、sfen()で文字列化できるはず。

    // 疲れた。あとで書く。
//    return Position::sfen_from_rawdata(board, hand, turn, 0);
    return "";
  }

  // pack()でpackされたsfen(256bit = 32bytes)
  // もしくはunpack()でdecodeするsfen
  u8 data[32];

private:
  BitStream stream;

  void write_board_piece_to_stream(Piece pc)
  {

  }

  void write_hand_piece_to_stream(Piece pc)
  {

  }

  Piece read_board_piece_from_stream()
  {

  }

  Piece read_hand_piece_from_stream()
  {

  }
};

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
    fs.open("generated_kifu.sfen", ios::out);
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

  // 局面を1行書き出す
  void write(size_t thread_id,string line)
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

private:
  // ファイルに書き出す前に排他するためのmutex
  Mutex mutex;

  fstream fs;

  // ファイルに書き出す前のバッファ
  vector<vector<string>> sfen_buffers;

  // 生成したい局面の数
  u64 sfen_count_limit;

  // 生成した局面の数
  u64 sfen_count;

};

// -----------------------------------
//  棋譜を生成するworker(スレッドごと)
// -----------------------------------

//  thread_id    = 0..Threads.size()-1
//  search_depth = 通常探索の探索深さ
void gen_sfen_worker(size_t thread_id , int search_depth , SfenWriter& sw)
{
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY + 64]; // StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
  Move moves[MAX_PLY]; // 局面の巻き戻し用に指し手を記憶
  int ply; // 初期局面からの手数

  PRNG prng(20160101);

  auto& pos = Threads[thread_id]->rootPos;

  pos.set_hirate();

  // Positionに対して従属スレッドの設定が必要。
  // 並列化するときは、Threads (これが実体が vector<Thread*>なので、
  // Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
  auto th = Threads[thread_id];
  pos.set_this_thread(th);

  // loop_maxになるまで繰り返し
  while(true)
  {
    for (ply = 0; ply < MAX_PLY; ++ply)
    {
      // mate1ply()の呼び出しのために必要。
      pos.check_info_update();

      if (pos.is_mated())
        break;

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

#if 1
      // sfenのpack test
      // pack()してunpack()したものが元のsfenと一致するのかのテスト。

      SfenPacker sp;
      sp.pack(pos);
      auto sfen = sp.unpack();
      auto pos_sfen = pos.sfen();

      // 手数の部分の出力がないので異なる。末尾の数字を消すと一致するはず。
      while (true)
      {
        auto c = *pos_sfen.rbegin();
        if (c < '0' || '9' < c)
          break;
        pos_sfen.pop_back();
      }

      if (sfen != pos_sfen)
      {
        cout << "Error: sfen packer error\n" << sfen << endl << pos_sfen << endl;
      }
#endif


      string line = pos.sfen() + "," + to_string(value1) + "\n";
      sw.write(thread_id,line);
      
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

#if 1
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
      auto m = pv1[0];

      pos.do_move(m, state[ply]);
      moves[ply] = m;
    }

    // 局面を最初まで巻き戻してみる(undo_moveの動作テストを兼ねて)
    while (ply > 0)
    {
      pos.undo_move(moves[--ply]);
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
