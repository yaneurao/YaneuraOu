#ifndef TANUKI_PROGRESS_H_INCLUDED
#define TANUKI_PROGRESS_H_INCLUDED

#include "evaluate.h"
#include "usioption.h"

namespace YaneuraOu {
class Position;
}

namespace Tanuki {
namespace Progress {

bool add_options(YaneuraOu::OptionsMap& options);
bool Load();
int LayerStackIndex(const YaneuraOu::Position& pos);

}  // namespace Progress
}  // namespace Tanuki

#endif  // TANUKI_PROGRESS_H_INCLUDED
