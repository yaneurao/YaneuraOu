#include "../shogi.h"

// 学習関係のルーチン
// 1) 棋譜の自動生成
//   → gensfenコマンド
// 2) 生成した棋譜からの評価関数パラメーターの学習
//   → learnコマンド
// 3) 定跡の自動生成
//   → makebook thinkコマンド
// 4) 局後自動検討モード
//   →　考え中
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
#include "multi_think.h"

using namespace std;

extern Book::MemoryBook book;
extern void is_ready();

namespace Learner
{

// packされたsfen
struct PackedSfenValue
{
	u8 data[34];
};

// -----------------------------------
//    局面のファイルへの書き出し
// -----------------------------------

// Sfenを書き出して行くためのヘルパクラス
struct SfenWriter
{
  SfenWriter(int thread_num)
  {
    sfen_buffers.resize(thread_num);
    fs.open(
#ifdef WRITE_PACKED_SFEN
      "generated_kifu.bin"
#else
      "generated_kifu.sfen"
#endif
      , ios::out | ios::binary | ios::app);
  }

  // この行数ごとにファイルにflushする。
  // 探索深さ3なら1スレッドあたり1秒で1000局面ほど作れる。
  // 40コア80HTで秒間10万局面。1日で80億ぐらい作れる。
  // 80億=8G , 1局面100バイトとしたら 800GB。
  
  const size_t FILE_WRITE_INTERVAL = 5000;

#ifdef  WRITE_PACKED_SFEN

  // 局面と評価値をペアにして1つ書き出す(packされたsfen形式で)
  void write(size_t thread_id, u8 data[32], int16_t value)
  {
    // スレッドごとにbufferを持っていて、そこに追加する。
    // bufferが溢れたら、ファイルに書き出す。

    // このバッファはスレッドごとに用意されている。
    auto& buf = sfen_buffers[thread_id];
    
    PackedSfenValue ps;
    memcpy(ps.data, data, 32);
    memcpy(ps.data + 32, &value, 2);
    
    // スレッドごとに用意されており、一つのスレッドが同時にこのwrite()関数を呼び出さないので
    // この時点では排他する必要はない。
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

      // 40回出力するごとに出力した局面数を出力。
      total_write += FILE_WRITE_INTERVAL;

      if ((total_write % (FILE_WRITE_INTERVAL * 40)) == 0)
      {
        // 現在時刻も出力
        auto now = std::chrono::system_clock::now();
        auto tp = std::chrono::system_clock::to_time_t(now);
        
        cout << endl << total_write << " sfens , at " << std::ctime(&tp);
      }
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

  u64 total_write = 0;

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

// 複数スレッドでsfenを生成するためのクラス
struct MultiThinkGenSfen: public MultiThink
{
  MultiThinkGenSfen(int search_depth_, SfenWriter& sw_) : search_depth(search_depth_), sw(sw_)
  {
    // 乱数を時刻で初期化しないとまずい。
    // (同じ乱数列だと同じ棋譜が生成されかねないため)
    set_prng(PRNG());
  }
  virtual void thread_worker(size_t thread_id);

  //  search_depth = 通常探索の探索深さ
  int search_depth;

  // sfenの書き出し器
  SfenWriter& sw;
};

//  thread_id    = 0..Threads.size()-1
void MultiThinkGenSfen::thread_worker(size_t thread_id)
{
  const int MAX_PLY = 256; // 256手までテスト
  StateInfo state[MAX_PLY + 64]; // StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
  int ply; // 初期局面からの手数
  Move m = MOVE_NONE;

  // 定跡の指し手を用いるモード
  int book_ply = Options["BookMoves"];

  // 規定回数回になるまで繰り返し
  while (get_next_loop_count() != UINT64_MAX)
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

          const auto& move = move_list[rand(move_list.size())];
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

        // 定跡の局面は書き出さない＆思考しない。
        continue;
      }

      {
        // search_depth手読みの評価値とPV(最善応手列)
        auto pv_value1 = search(pos, -VALUE_INFINITE, VALUE_INFINITE, search_depth);
        auto value1 = pv_value1.first;
        auto pv1 = pv_value1.second;

#if 1
		// 評価値の絶対値が2000以上の局面については
		// その局面を学習に使うのはあまり意味がないのでこの試合を終了する。
		// 2000という数字自体は調整したほうが良いが…。
		if (abs(value1) >= 2000)
			break;
#endif

#if 0
        // 0手読み(静止探索のみ)の評価値とPV(最善応手列)
        auto pv_value2 = qsearch(pos, -VALUE_INFINITE, VALUE_INFINITE);
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
        // このwriteがスレッド排他を行うので、ここでの排他は不要。
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

        {
          // C++のiostreamに対するスレッド排他は自前で行なう必要がある。
          std::unique_lock<Mutex> lk(io_mutex);

          // sfenとそのときの評価値を書き出す。
          string line = pos.sfen() + "," + to_string(value1) + "\n";
          sw.write(thread_id, line);
      }
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
      if (rand(5) == 0 && !pos.in_check())
      {
        for (int retry = 0; retry < 10; ++retry)
        {
          // 手番側の駒を2駒入れ替える。

          // 与えられたBitboardからランダムに1駒を選び、そのSquareを返す。
          auto get_one = [this](Bitboard pieces)
          {
            // 駒の数
            int num = pieces.pop_count();
            // 何番目かの駒
            int n = (int)rand(num) + 1;
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
//    棋譜を生成するコマンド(master thread)
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

  // Options["Threads"]の数だけスレッドを作って実行。
  SfenWriter sw(thread_num);
  MultiThinkGenSfen multi_think(search_depth,sw);
  multi_think.set_loop_max(loop_max);
  multi_think.go_think();

  std::cout << "gen_sfen finished." << endl;
}

// -----------------------------------
// 生成した棋譜から学習させるコマンド(learn)
// -----------------------------------

// Sfenの読み込み機
struct SfenReader
{
	SfenReader(int thread_num)
	{
		packed_sfens.resize(thread_num);
		total_read = 0;
		files_end = false;
	}

	// [ASYNC] スレッドが局面を一つ返す。なければfalseが返る。
	bool read_to_thread_buffer(size_t thread_id, PackedSfenValue& ps)
	{
		// スレッドバッファに局面が残っているなら、それを1つ取り出して返す。
		auto& thread_ps = packed_sfens[thread_id];

		// バッファに残りがなかったらread bufferから充填するが、それすらなかったらもう終了。
		if (thread_ps.size() == 0 && !read_to_thread_buffer_impl(thread_ps))
			return false;

		ps = *thread_ps.rbegin();
		thread_ps.pop_back();
		return true;
	}

	// [ASYNC] スレッドバッファに局面を10000局面ほど読み込む。
	bool read_to_thread_buffer_impl(vector<PackedSfenValue>& ps_vec)
	{
		// スレッドバッファが尽きたのでグローバルバッファから充填する。
		std::unique_lock<Mutex> lk(mutex);

		// もうすでに読み込むファイルは無くなっている。
		if (files_end)
			return false;

		auto read_from_buffer = [&]()
		{
			// read bufferから充填可能か？
			const int THREAD_BUFFER_SIZE = 10000;
			int size = (int)read_buffer.size();
			if (size >= THREAD_BUFFER_SIZE)
			{
				// 高速化のためにブロックコピーしてしまう。
				ps_vec.resize(THREAD_BUFFER_SIZE);
				int new_size = size - THREAD_BUFFER_SIZE;
				memcpy(&ps_vec[0], &read_buffer[new_size], sizeof(PackedSfenValue)*THREAD_BUFFER_SIZE);
				read_buffer.resize(new_size);

				// 1万局面読むごとに'.'をひとつ出力。
				cout << '.';

				return true;
			}
			return false;
		};

		// ファイルバッファから充填できたなら、それで良し。
		if (read_from_buffer())
			return true;

		// 次の10M局面(34*10MB = 340MB)を読み込む
		// (この単位でshuffleする)

		const int SFEN_READ_SIZE = 1000 * 1000 * 10; // 10M局面
//		const int SFEN_READ_SIZE = 1000 * 1000 * 1; // 1M局面
		auto open_next_file = [&]()
		{
			if (fs.is_open())
				fs.close();

			// もう無い
			if (filenames.size() == 0)
				return false;

			string filename = *filenames.rbegin();
			filenames.pop_back();

			// 生成した棋譜をテスト的に読み込むためのコード
			fs.open(filename, ios::in | ios::binary);
			cout << endl << "open filename = " << filename << " ";

			return true;
		};

		// SFEN_READ_SIZE分だけ読むごとに時刻を表示。

		// 現在時刻も出力
		auto now = std::chrono::system_clock::now();
		auto tp = std::chrono::system_clock::to_time_t(now);
		cout << endl << total_read << " sfens , at " << std::ctime(&tp);

		// ファイルバッファにファイルから読み込む。
		while (read_buffer.size() < SFEN_READ_SIZE)
		{
			PackedSfenValue p;
			if (fs.read((char*)&p, sizeof(PackedSfenValue)))
			{
				read_buffer.push_back(p);
			}
			else
			{
				// 読み込み失敗
				if (!open_next_file())
				{
					// 次のファイルもなかった。あぼーん。
					cout << "..end of files.\n";
					files_end = true;
					return false;
				}
			}
		}
		
#if 1
		// この読み込んだ局面データをシャッフルする。
		{
			auto size = read_buffer.size();
			for (size_t i = 0; i < size; ++i)
				swap(read_buffer[i], read_buffer[prng.rand(size-i) + i]);
		}
#endif

		total_read += SFEN_READ_SIZE;

		if (read_from_buffer())
			return true;

		// バッファから充填しようとしたがデーターが足りなかった模様。んな馬鹿な…。
		// THREAD_BUFFER_SIZE < SFEN_READ_SIZE 
		// だから、それはないはず。
		ASSERT_LV3(false);

		return false;
	}


	virtual void thread_worker(size_t thread_id)
	{
		while (true)
		{
			PackedSfenValue ps;
			auto hit = read_to_thread_buffer(thread_id, ps);
			if (!hit)
				break;
		}
	}

	// sfenファイル群
	vector<string> filenames;

protected:

	PRNG prng;

	// 読み込んだ局面数
	u64 total_read;

	// ファイル群を読み込んでいき、最後まで到達したか。
	bool files_end;

	// ファイルを読み込むためのmutex
	Mutex mutex;

	// sfenファイルのハンドル
	fstream fs;

	// 各スレッド用のsfen
	vector<vector<PackedSfenValue>> packed_sfens;

	// 10M局面分の読み込みバッファ
	vector<PackedSfenValue> read_buffer;
};


// 複数スレッドでsfenを生成するためのクラス
struct LearnerThink: public MultiThink
{
	LearnerThink(SfenReader& sr_):sr(sr_){}
	virtual void thread_worker(size_t thread_id);

	//  search_depth = 通常探索の探索深さ
	int search_depth;

	// sfenの読み出し器
	SfenReader& sr;
};

void LearnerThink::thread_worker(size_t thread_id)
{
	auto& pos = Threads[thread_id]->rootPos;

	while (true)
	{
		PackedSfenValue ps;
		if (!sr.read_to_thread_buffer(thread_id, ps))
			break;

		auto sfen = pos.sfen_unpack(ps.data);

		pos.set(sfen);
		auto th = Threads[thread_id];
		pos.set_this_thread(th);

		// 評価値は棋譜生成のときに、この34バイトの末尾2バイトに埋めてある。
		Value value = (Value)*(int16_t*)&ps.data[32];

		// 読み込めたので試しに表示してみる。
		//		cout << pos << value << endl;

		// ここに静止探索してなんやかやするコードを書く。

	}
}

// 生成した棋譜からの学習
void learn(Position& pos, istringstream& is)
{

	auto thread_num = (int)Options["Threads"];
	SfenReader sr(thread_num);

	LearnerThink learn_think(sr);
	vector<string> filenames;

	// ファイル名が後ろにずらずらと書かれていると仮定している。
	while (true)
	{
		string filename;
		is >> filename;

		if (filename == "")
			break;

		filenames.push_back(filename);
	}

	cout << "learn from ";
	for (auto s : filenames)
		cout << s << " , ";

	reverse(filenames.begin(), filenames.end());
	sr.filenames = filenames;

	learn_think.go_think();
}


} // namespace Learner

#endif // EVAL_LEARN
