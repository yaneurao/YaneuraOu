#include "score.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "usi.h"

namespace YaneuraOu {

Score::Score(Value v /*, const Position& pos*/) {
    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

    if (!is_decisive(v))
    {
        score = InternalUnits{USIEngine::to_cp(v /*, pos*/)};
    }
    //else if (std::abs(v) <= VALUE_TB)
    //{
    //    auto distance = VALUE_TB - std::abs(v);
    //    score         = (v > 0) ? Tablebase{distance, true} : Tablebase{-distance, false};
    //}
    else
    {
		// 詰みまでの手数
        auto distance = VALUE_MATE - std::abs(v);
        score         = (v > 0) ? Mate{distance} : Mate{-distance};
    }
}

#if !STOCKFISH
// 🌈 Valueの値を(cpへの変換をせずに)そのままScoreに変換する。
Score Score::from_internal_value(Value v) {
    Score score;
    score.score = InternalUnits{ v };
    return score;
}

// 🌈 いま保持している値をValueに変換する。
Value Score::to_value() const {
    if (is<InternalUnits>())
		// cpで保持している。
        return USIEngine::cp_to_value(get<InternalUnits>().value);

	// Mateで保持している。
    int plies = get<Mate>().plies;
    return (plies > 0) ? mate_in(plies) : mated_in(plies);
}
#endif

}
