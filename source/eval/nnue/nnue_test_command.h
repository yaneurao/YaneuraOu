// NNUE評価関数に関するUSI拡張コマンドのインターフェイス

#ifndef CLASSIC_NNUE_TEST_COMMAND_H
#define CLASSIC_NNUE_TEST_COMMAND_H

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

namespace YaneuraOu {
namespace Eval::NNUE {

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(Position& pos, std::istream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#endif
