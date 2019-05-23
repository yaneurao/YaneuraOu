/*
  Apery, a USI shogi playing engine derived from Stockfish, a UCI chess playing engine.
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Copyright (C) 2011-2016 Hiraoka Takuya

  Apery is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Apery is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "apery_book.h"

#include "../../misc.h"
#include <fstream>

namespace Book {

Key AperyBook::ZobPiece[PIECE_NB - 1][SQ_NB];
Key AperyBook::ZobHand[PIECE_HAND_NB - 1][19]; // 持ち駒の同一種類の駒の数ごと
Key AperyBook::ZobTurn;

void AperyBook::init() {
	// 定跡のhash生成用なので、seedは固定でデフォルト値を使う。
	// 未初期化乱数でhashが毎回変更されるのを防ぐため、init()が呼ばれる度に乱数は初期化する。
	// （そもそもinit()を呼ぶのは1回きりでも良いのだけど、今の実装ではコンストラクタから毎回呼ばれるので。）
	MT64bit mt64bit_;
    for (Piece p = PIECE_ZERO; p < PIECE_NB - 1; ++p) {
        for (Square sq = SQ_ZERO; sq < SQ_NB; ++sq)
            ZobPiece[p][sq] = mt64bit_.random() * (p != NO_PIECE);
    }
    for (Piece hp = PIECE_ZERO; hp < PIECE_HAND_NB - 1; ++hp) {
        for (int num = 0; num < 19; ++num)
            ZobHand[hp][num] = mt64bit_.random();
    }
    ZobTurn = mt64bit_.random();
}

AperyBook::AperyBook(const std::string& filename) {
    init();

    std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);

    if (!file.is_open())
        sync_cout << "info string could not open an apery book file. " << filename << sync_endl;

    AperyBookEntry entry;
    while (file.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
        book_[entry.key].push_back(entry);
    }
}

Key AperyBook::bookKey(const Position& pos) {
    static constexpr int hand_piece_map[] = {
        -1, 0, 1, 2, 3, 5, 6, 4
    };

    Key key = 0;
    for (Square sq = SQ_ZERO; sq < SQ_NB; ++sq) {
        key ^= ZobPiece[pos.piece_on(sq)][sq];
    }
    const Hand hand = pos.hand_of(pos.side_to_move());
    for (Piece hp = PAWN; hp < PIECE_HAND_NB; ++hp)
        key ^= ZobHand[hand_piece_map[hp]][hand_count(hand, hp)];
    if (pos.side_to_move() == WHITE)
        key ^= ZobTurn;
    return key;
}

const std::vector<AperyBookEntry>& AperyBook::get_entries(const Position& pos) const {
    const auto it = book_.find(bookKey(pos));
    if (it == book_.end()) return empty_entries_;
    return it->second;
}

}
