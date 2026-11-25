// NNUE評価関数に関するUSI拡張コマンドのインターフェイス

#ifndef NNUE_TEST_COMMAND_H_INCLUDED
#define NNUE_TEST_COMMAND_H_INCLUDED

#include <iosfwd>

#include "../config.h"

#if defined(ENABLE_TEST_CMD) && (defined(EVAL_NNUE) || defined(EVAL_SFNN))

namespace YaneuraOu {
class IEngine;
} // namespace YaneuraOu

namespace YaneuraOu::Eval::NNUE {

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(IEngine& engine, std::istream& stream);

} // namespace YaneuraOu::Eval::NNUE

#endif  // defined(ENABLE_TEST_CMD) && (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
