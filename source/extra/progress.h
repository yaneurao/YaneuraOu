#ifndef _PROGRESS_H
#define _PROGRESS_H

#include <map>
#include <string>

#include "../evaluate.h"
#include "../shogi.h"

#ifdef USE_PROGRESS

struct Position;

class Progress {
public:
  static bool Initialize(USI::OptionsMap& o);
  bool Load();
  bool Save();
  bool Learn();
  double Estimate(const Position& pos);

private:
  double weights_[SQ_NB][Eval::fe_end] = {{ 0 }};
};

#endif

#endif
