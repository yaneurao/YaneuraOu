#ifndef _BOOK_H_
#define _BOOK_H_

#include "../shogi.h"

// 定跡処理関係
namespace Book
{
  // 局面における指し手(定跡を格納するのに用いる)
  struct BookPos
  {
    Move bestMove; // この局面での指し手
    Move nextMove; // その指し手を指したときの予想される相手の指し手
    int value;   // bestMoveを指したときの局面の評価値
    int depth;     // bestMoveの探索深さ

    BookPos(Move best, Move next, int v, int d) : bestMove(best), nextMove(next), value(v), depth(d) {}
    bool operator == (BookPos& rhs) { return bestMove == rhs.bestMove; }
  };

  // メモリ上にある定跡ファイル
  // sfen文字列をkeyとして、局面の指し手へ変換。(重複した指し手は除外するものとする)
  typedef std::map<std::string, std::vector<BookPos> > MemoryBook;

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
