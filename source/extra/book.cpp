#include "book.h"
#include "../position.h"
#include "../misc.h"
#include <fstream>
#include <sstream>

using namespace std;
using std::cout;

namespace Book
{


  // ----------------------------------
  // USI拡張コマンド "makebook"(定跡作成)
  // ----------------------------------

  // フォーマット等についてはdoc/解説.txt を見ること。
  void makebook_cmd(Position& pos, istringstream& is)
  {
    string token;
    is >> token;
    if (token != "")
    {
      // sfen→txtの変換

      string sfen_name = token;
      string book_name;
      is >> book_name;
      is >> token;
      int moves = 16;
      if (token == "moves")
        is >> moves;

      cout << "read";

      vector<string> sfens;
      read_all_lines(sfen_name, sfens);

      cout << "..done" << endl;
      cout << "parse";

      MemoryBook book;

      // 各行の局面をparseして読み込む(このときに重複除去も行なう)
      for (int k = 0; k < sfens.size(); ++k)
      {
        auto sfen = sfens[k];

        if (sfen.length() == 0)
          continue;

        istringstream iss(sfen);
        token = "";
        do {
          iss >> token;
        } while (token == "startpos" || token == "moves");

        vector<Move> m;    // 初手から(moves+1)手までのsfen文字列格納用
        m.resize(moves + 1);
        vector<string> sf; // 初手から(moves+0)手までのsfen文字列格納用
        sf.resize(moves + 1);

        StateInfo si[MAX_PLY];

        pos.set_hirate();
        for (int i = 0; i < moves + 1; ++i)
        {
          m[i] = move_from_usi(pos, token);
          if (m[i] == MOVE_NONE || !pos.pseudo_legal(m[i]) || !pos.legal(m[i]))
          {
            cout << "ilegal move : line = " << (k + 1) << " , " << sfen << " , move = " << token << endl;
            break;
          }
          sf[i] = pos.sfen();
          pos.do_move(m[i], si[i]);
          iss >> token;
        }
        for (int i = 0; i < moves; ++i)
        {
          BookPos bp(m[i], m[i + 1], VALUE_ZERO, 32);
          insert_book_pos(book, sf[i], bp);
        }

        if ((k % 1000) == 0)
          cout << '.';
      }
      cout << "done." << endl;
      cout << "write..";
      
      write_book(book_name, book);

      cout << "finished." << endl;

    } else {
      cout << "usage" << endl;
      cout << "> makebook book.sfen book.db moves 24" << endl;
    }

  }

  // 定跡ファイルの読み込み(book.db)など。
  int read_book(const std::string& filename, MemoryBook& book)
  {
    vector<string> lines;
    if (read_all_lines(filename, lines))
      return 1; // 読み込み失敗

    string sfen;
    for (auto line : lines)
    {
      // "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
      if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
      {
        sfen = line.substr(5,line.length()-5); // "sfen "を除去して格納
        continue;
      }

      Move best, next;
      int value;
      int depth;

      istringstream is(line);
      string bestMove, nextMove;
      is >> bestMove >> nextMove >> value >> depth;

      // 起動時なので変換に要するオーバーヘッドは最小化したいので合法かのチェックはしない。
      if (bestMove == "none" || bestMove == "resign")
        best = MOVE_NONE;
      else
        best = move_from_usi(bestMove);

      if (nextMove == "none" || nextMove == "resign")
        next = MOVE_NONE;
      else
        next = move_from_usi(nextMove);

      BookPos bp(best,next,value,depth);
      insert_book_pos(book,sfen,bp);
    }

    return 0;
  }

  // 定跡ファイルの書き出し
  int write_book(const std::string& filename, const MemoryBook& book)
  {
    fstream fs;
    fs.open(filename, ios::out);
    for (auto it = book.begin(); it != book.end(); ++it)
    {
      fs << "sfen " << it->first /* is sfen string */ << endl; // sfen
      for (auto& bp : it->second)
        fs << bp.bestMove << ' ' << bp.nextMove << ' ' << bp.value << " " << bp.depth << endl; // 指し手、相手の応手、そのときの評価値、探索深さ
    }

    fs.close();
    
    return 0;
  }

  void insert_book_pos(MemoryBook& book, const std::string sfen,const BookPos& bp)
  {
    auto it = book.find(sfen);
    if (it == book.end())
    {
      // 存在しないので要素を作って追加。
      vector<BookPos> v;
      v.push_back(bp);
      book[sfen] = v;
    } else {
      // すでに格納されているかも知れないので同じ指し手がないかをチェックして、なければ追加
      for (auto& b : it->second)
        if (b == *(const_cast<BookPos*>(&bp)))
        {
          // すでに存在していたのでエントリーを置換
          b = bp;
          goto FOUND_THE_SAME_MOVE;
        }

      it->second.push_back(bp);

    FOUND_THE_SAME_MOVE:;
    }

  }


}
