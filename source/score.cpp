#include "score.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "usi.h"

namespace YaneuraOu {

#if STOCKFISH
Score::Score(Value v, const Position& pos) {
#else
Score::Score(Value v) {
#endif

    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

    if (!is_decisive(v))
#if STOCKFISH
        score = InternalUnits{UCIEngine::to_cp(v, pos)};

    else if (std::abs(v) <= VALUE_TB)
    {
        auto distance = VALUE_TB - std::abs(v);
        score         = (v > 0) ? Tablebase{distance, true} : Tablebase{-distance, false};
    }

#else
        score = InternalUnits{USIEngine::to_cp(v)};
#endif

    else
    {
        // 詰みまでの手数
        auto distance = VALUE_MATE - std::abs(v);
        score         = (v > 0) ? Mate{distance} : Mate{-distance};
    }
}

#if !STOCKFISH
// 🌈 変換済みのcp値をそのままScoreに格納する。
Score Score::from_cp(int cp) {
    Score score;
    score.score = InternalUnits{cp};
    return score;
}

// 🌈 保持しているcp値または詰み手数を内部評価値に変換する。
Value Score::to_value() const {
    if (is<InternalUnits>())
        // cpで保持している。
        return USIEngine::cp_to_value(get<InternalUnits>().value);

    // Mateで保持している。
    int plies = get<Mate>().plies;
    return (plies > 0) ? mate_in(plies) : mated_in(-plies);
}
#endif

}
