#include "../shogi.h"

#include <fstream>
#include <sstream>
#include <unordered_set>

#include "book.h"
#include "../position.h"
#include "../misc.h"
#include "../search.h"
#include "../thread.h"
#include "../learn/multi_think.h"

using namespace std;
using std::cout;

namespace Book
{
#ifdef ENABLE_MAKEBOOK_CMD
  // ----------------------------------
  // USI拡張コマンド "makebook"(定跡作成)
  // ----------------------------------

  // 局面を与えて、その局面で思考させるために、やねうら王2016Midが必要。
  #if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)
  struct MultiThinkBook : public MultiThink
  {
    MultiThinkBook(int search_depth_, MemoryBook & book_) : search_depth(search_depth_), book(book_) {}
    virtual void MultiThinkBook::thread_worker(size_t thread_id);

    // 定跡を作るために思考する局面
    vector<string> sfens;

    // 定跡を作るときの通常探索の探索深さ
    int search_depth;

    // メモリ上の定跡ファイル(ここに追加していく)
    MemoryBook& book;
  };


  //  thread_id    = 0..Threads.size()-1
  //  search_depth = 通常探索の探索深さ
  void MultiThinkBook::thread_worker(size_t thread_id )
  {
    // g_loop_maxになるまで繰り返し
    u64 id;
    while ((id = get_next_loop_count()) != UINT64_MAX)
    {
      auto sfen = sfens[id];

      auto& pos = Threads[thread_id]->rootPos;
      pos.set(sfen);
      auto th = Threads[thread_id];
      pos.set_this_thread(th);

      if (pos.is_mated())
        continue;

      // depth手読みの評価値とPV(最善応手列)
      search(pos, -VALUE_INFINITE, VALUE_INFINITE, search_depth);

      // MultiPVで局面を足す、的な

      vector<BookPos> move_list;

      int multi_pv = std::min((int)Options["MultiPV"], (int)th->rootMoves.size());
      for (int i = 0; i<multi_pv; ++i)
      {
        // 出現頻度は、ベストの指し手が一番大きな数値になるようにしておく。
        // (narrow bookを指定したときにベストの指し手が選ばれるように)
        Move nextMove = (th->rootMoves[i].pv.size() >= 1) ? th->rootMoves[i].pv[1] : MOVE_NONE;
        BookPos bp(th->rootMoves[i].pv[0], nextMove, th->rootMoves[i].score
          , search_depth, multi_pv - i);
        move_list.push_back(bp);
      }

      {
        std::unique_lock<Mutex> lk(io_mutex);
        // 前のエントリーは上書きされる。
        book.book_body[sfen] = move_list;
      }

      // 1局面思考するごとに'.'をひとつ出力する。
      cout << '.';
    }
  }
  #endif

  // フォーマット等についてはdoc/解説.txt を見ること。
  void makebook_cmd(Position& pos, istringstream& is)
  {
    string token;
    is >> token;

    // sfenから生成する
    bool from_sfen = token == "from_sfen";
    // 自ら思考して生成する
    bool from_thinking = token == "think";

    #if !defined(EVAL_LEARN) || !defined(YANEURAOU_2016_MID_ENGINE)
    if (from_thinking)
    {
      cout << "Error!:define EVAL_LEARN and YANEURAOU_2016_MID_ENGINE" << endl;
      return;
    }
    #endif

    if (from_sfen || from_thinking)
    {
      // sfenファイル名
      is >> token;
      string sfen_name = token;

      // 定跡ファイル名
      string book_name;
      is >> book_name;

      // 手数、探索深さ
      int moves = 16;
      int depth = 24;
      while (true)
      {
        token = "";
        is >> token;
        if (token == "")
          break;
        if (token == "moves")
          is >> moves;
        else if (token == "depth")
          is >> depth;
        else
        {
          cout << "Error! : Illigal token = " << token << endl;
          return;
        }
      }

      cout << "read sfen moves " << moves;
      if (from_thinking)
        cout << " depth = " << depth;

      vector<string> sfens;
      read_all_lines(sfen_name, sfens);

      cout << "..done" << endl;

      MemoryBook book;

      if (from_thinking)
      {
        cout << "read book..";
        // 初回はファイルがないので読み込みに失敗するが無視して続行。
        if (read_book(book_name, book) !=0 )
        {
          cout << "..but , create new file." << endl;
        } else
          cout << "..done" << endl;
      }

      cout << "parse..";

      // 思考すべき局面のsfen
      unordered_set<string> thinking_sfens;

      // 各行の局面をparseして読み込む(このときに重複除去も行なう)
      for (size_t k = 0; k < sfens.size(); ++k)
      {
        auto sfen = sfens[k];

        if (sfen.length() == 0)
          continue;

        istringstream iss(sfen);
        token = "";
        do {
          iss >> token;
        } while (token == "startpos" || token == "moves");

        vector<Move> m;    // 初手から(moves+1)手までの指し手格納用
        vector<string> sf; // 初手から(moves+0)手までのsfen文字列格納用

        StateInfo si[MAX_PLY];

        pos.set_hirate();

        // sfenから直接生成するときはponderのためにmoves + 1の局面まで調べる必要がある。
        for (int i = 0; i < moves + (from_sfen ? 1 : 0); ++i)
        {
          // 初回は、↑でfeedしたtokenが入っているはず。
          if (i != 0)
          {
            token = "";
            iss >> token;
          }
          if (token == "")
            break;

          Move move = move_from_usi(pos, token);
          // illigal moveであるとMOVE_NONEが返る。
          if (move == MOVE_NONE)
          {
            cout << "illegal move : line = " << (k + 1) << " , " << sfen << " , move = " << token << endl;
            break;
          }

          // MOVE_WIN,MOVE_RESIGNでは局面を進められないのでここで終了。
          if (!is_ok(move))
            break;

          sf.push_back(pos.sfen());
          m.push_back(move);

          pos.do_move(move, si[i]);
        }
        
        for (int i = 0; i < m.size() - (from_sfen ? 1 : 0) ; ++i)
        {
          if (from_sfen)
          {
            // この場合、m[i + 1]が必要になるので、m.size()-1までしかループできない。
            BookPos bp(m[i], m[i + 1], VALUE_ZERO, 32, 1);
            insert_book_pos(book, sf[i], bp);
          }
          else if (from_thinking)
          {
            // posの局面で思考させてみる。(あとでまとめて)
            if (thinking_sfens.count(sf[i]) == 0)
              thinking_sfens.insert(sf[i]);
          }
        }

        // sfenから生成するモードの場合、1000棋譜処理するごとにドットを出力。
        if ((k % 1000) == 0)
          cout << '.';  
      }
      cout << "done." << endl;

#if defined(EVAL_LEARN) && defined(YANEURAOU_2016_MID_ENGINE)

      if (from_thinking)
      {
        // thinking_sfensを並列的に探索して思考する。
        // スレッド数(これは、USIのsetoptionで与えられる)
        u32 multi_pv = Options["MultiPV"];

        // 思考する局面をsfensに突っ込んで、この局面数をg_loop_maxに代入しておき、この回数だけ思考する。
        MultiThinkBook multi_think(depth, book);

        auto& sfens_ = multi_think.sfens;
        for (auto& s : thinking_sfens)
        {
          
          // この局面のいま格納されているデータを比較して、この局面を再考すべきか判断する。
          auto it = book.book_body.find(s);

          // MemoryBookにエントリーが存在しないなら無条件で、この局面について思考して良い。
          if (it == book.book_body.end())
            sfens_.push_back(s);
          else
          {
            auto& bp = it->second;
            if (bp[0].depth < depth // 今回の探索depthのほうが深い
              || (bp[0].depth == depth && bp.size() < multi_pv) // 探索深さは同じだが今回のMultiPVのほうが大きい
              )
              sfens_.push_back(s);
          }
        }
        
#if 1
        // 思考対象局面が求まったので、sfenを表示させてみる。
        cout << "thinking sfen = " << endl;
        for (auto& s : sfens_)
          cout << "sfen " << s << endl;
#endif

        multi_think.set_loop_max(sfens_.size());
        multi_think.go_think();

      }

#endif

      cout << "write..";
      
      write_book(book_name, book);

      cout << "finished." << endl;

    } else {
      cout << "usage" << endl;
      cout << "> makebook book.sfen book.db moves 24" << endl;
    }
  }
#endif

  // 定跡ファイルの読み込み(book.db)など。
  int read_book(const std::string& filename, MemoryBook& book)
  {
    // 読み込み済であるかの判定
    if (book.book_name == filename)
      return 0;

    vector<string> lines;
    if (read_all_lines(filename, lines))
    {
      cout << "info string Error! : can't read " + filename << endl;
//      exit(EXIT_FAILURE);
      return 1; // 読み込み失敗
    }

    uint64_t num_sum = 0;
    string sfen;

    auto calc_prob = [&] {
      auto& move_list = book.book_body[sfen];
      std::stable_sort(move_list.begin(), move_list.end());
      num_sum = std::max(num_sum, UINT64_C(1)); // ゼロ除算対策
      for (auto& bp : move_list)
        bp.prob = float(bp.num) / num_sum;
      num_sum = 0;
    };

    for (auto line : lines)
    {
      // バージョン識別文字列(とりあえず読み飛ばす)
      if (line.length() >= 1 && line[0] == '#')
        continue;

      // コメント行(とりあえず読み飛ばす)
      if (line.length() >= 2 && line.substr(0, 2) == "//")
        continue;

      // "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
      if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
      {
        // ひとつ前のsfen文字列に対応するものが終わったということなので採択確率を計算して、かつ、採択回数でsortしておく
        // (sortはされてるはずだが他のソフトで生成した定跡DBではそうとも限らないので)。
        calc_prob();

        sfen = line.substr(5,line.length()-5); // 新しいsfen文字列を"sfen "を除去して格納
        continue;
      }

      Move best, next;
      int value;
      int depth;

      istringstream is(line);
      string bestMove, nextMove;
      uint64_t num;
      is >> bestMove >> nextMove >> value >> depth >> num;

      // 起動時なので変換に要するオーバーヘッドは最小化したいので合法かのチェックはしない。
      if (bestMove == "none" || bestMove == "resign")
        best = MOVE_NONE;
      else
        best = move_from_usi(bestMove);

      if (nextMove == "none" || nextMove == "resign")
        next = MOVE_NONE;
      else
        next = move_from_usi(nextMove);

      BookPos bp(best,next,value,depth,num);
      insert_book_pos(book,sfen,bp);
      num_sum += num;
    }
    // ファイルが終わるときにも最後の局面に対するcalc_probが必要。
    calc_prob();

    // 読み込んだファイル名を保存しておく。二度目のread_book()はskipする。
    book.book_name = filename;

    return 0;
  }

  // 定跡ファイルの書き出し
  int write_book(const std::string& filename, const MemoryBook& book)
  {
    fstream fs;
    fs.open(filename, ios::out);

    // バージョン識別用文字列
    fs << "#YANEURAOU-DB2016 1.00" << endl;

    for (auto& it : book.book_body )
    {
      fs << "sfen " << it.first /* is sfen string */ << endl; // sfen

      // const性を消すためにcopyする
      auto move_list = it.second;

      // 採択回数でソートしておく。
      std::stable_sort(move_list.begin(), move_list.end());

      for (auto& bp : move_list)
        fs << bp.bestMove << ' ' << bp.nextMove << ' ' << bp.value << " " << bp.depth << " " << bp.num << endl;
      // 指し手、相手の応手、そのときの評価値、探索深さ、採択回数
    }

    fs.close();
    
    return 0;
  }

  void insert_book_pos(MemoryBook& book, const std::string sfen,const BookPos& bp)
  {
    auto it = book.book_body.find(sfen);
    if (it == book.end())
    {
      // 存在しないので要素を作って追加。
      vector<BookPos> move_list;
      move_list.push_back(bp);
      book.book_body[sfen] = move_list;
    } else {
      // この局面での指し手のリスト
      auto& move_list = it->second;
      // すでに格納されているかも知れないので同じ指し手がないかをチェックして、なければ追加
      for (auto& b : move_list)
        if (b == bp)
        {
          // すでに存在していたのでエントリーを置換。ただし採択回数はインクリメント
          auto num = b.num;
          b = bp;
          b.num += num;
          goto FOUND_THE_SAME_MOVE;
        }

      move_list.push_back(bp);

    FOUND_THE_SAME_MOVE:;
    }

  }
}
