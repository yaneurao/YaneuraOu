// NNUE評価関数に関するUSI拡張コマンドのインターフェイス

#ifndef NNUE_TEST_COMMAND_H_INCLUDED
#define NNUE_TEST_COMMAND_H_INCLUDED

#include "../config.h"

#if defined(ENABLE_TEST_CMD) && (defined(EVAL_NNUE) || defined(EVAL_SFNN))

namespace YaneuraOu {
class IEngine;
namespace Eval::NNUE {

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(IEngine& engine, std::istream& stream);

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
