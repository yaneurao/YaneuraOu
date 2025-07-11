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

}
