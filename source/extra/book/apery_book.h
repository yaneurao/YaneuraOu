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

#ifndef APERY_BOOK_HPP
#define APERY_BOOK_HPP

#include <unordered_map>
#include "mt64bit.h"
#include "../../shogi.h"
#include "../../position.h"

namespace Book {

using Key = uint64_t;
using Score = int;

struct AperyBookEntry {
    Key key;
    uint16_t fromToPro;
    uint16_t count;
    Score score;
};

class AperyBook {
public:
    explicit AperyBook(const char* fName);
    const std::vector<AperyBookEntry>& get_entries(const Position& pos) const;
    static Key bookKey(const Position& pos);
    size_t size() const { return book_.size(); }

private:
    static void init();

    std::vector<AperyBookEntry> empty_entries_;
    std::unordered_map<Key, std::vector<AperyBookEntry>> book_;

    static Key ZobPiece[PIECE_NB - 1][SQ_NB];
    static Key ZobHand[PIECE_HAND_NB - 1][19];
    static Key ZobTurn;
};

}

#endif // #ifndef APERY_BOOK_HPP
