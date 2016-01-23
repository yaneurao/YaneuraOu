#ifndef _MATE1PLY_H_
#define _MATE1PLY_H_

#include "../shogi.h"
#if defined(MATE_1PLY) && defined(LONG_EFFECT_LIBRARY)

#include "long_effect.h"

namespace Mate1Ply
{
  // Mate1Ply関係のテーブル初期化
  void init();
}

#endif // MATE_1PLY
#endif // _MATE1PLY_H_
