#ifndef _BOOK_H_
#define _BOOK_H_

#include "../shogi.h"
#include "../position.h"
#include <unordered_map>

// 定跡処理関係
namespace Book
{
  // 局面における指し手(定跡を格納するのに用いる)
  struct BookPos
  {
    Move bestMove; // この局面での指し手
    Move nextMove; // その指し手を指したときの予想される相手の指し手
    int value;     // bestMoveを指したときの局面の評価値
    int depth;     // bestMoveの探索深さ
    uint64_t num;  // 何らかの棋譜集において、この指し手が採択された回数。
    float prob;    // ↑のnumをパーセンテージで表現したもの。(read_bookしたときには反映される。ファイルには書き出していない。)

    BookPos(Move best, Move next, int v, int d,uint64_t n) : bestMove(best), nextMove(next), value(v), depth(d),num(n) {}
    bool operator == (const BookPos& rhs) const { return bestMove == rhs.bestMove; }
    bool operator < (const BookPos& rhs) const { return num > rhs.num; } // std::sortで降順ソートされて欲しいのでこう定義する。
  };

  // メモリ上にある定跡ファイル
  // sfen文字列をkeyとして、局面の指し手へ変換。(重複した指し手は除外するものとする)
  struct MemoryBook
  {
    typedef std::unordered_map<std::string, std::vector<BookPos> > BookType;
    BookType::iterator find(const Position& pos)
    {
      auto it = book_body.find(pos.sfen());
      if (it != book_body.end())
      {
        // 定跡のMoveは16bitであり、rootMovesは32bitのMoveであるからこのタイミングで補正する。
        for (auto& m : it->second)
          m.bestMove = pos.move16_to_move(m.bestMove);
      }
      return it;
    }
    const BookType::iterator end() { return book_body.end(); }

    BookType book_body;
  };

  // USI拡張コマンド。"makebook"。定跡ファイルを作成する。
  // フォーマット等についてはdoc/解説.txt を見ること。
  extern void makebook_cmd(Position& pos, std::istringstream& is);

  // 定跡ファイルの読み込み(book.db)など。
  extern int read_book(const std::string& filename, MemoryBook& book);

  // 定跡ファイルの書き出し
  extern int write_book(const std::string& filename, const MemoryBook& book);

  // bookにBookPosを一つ追加。(その局面ですでに同じbestMoveの指し手が登録されている場合は上書き動作)
  extern void insert_book_pos(MemoryBook& book, const std::string sfen,const BookPos& bp);

}

#endif // #ifndef _BOOK_H_
