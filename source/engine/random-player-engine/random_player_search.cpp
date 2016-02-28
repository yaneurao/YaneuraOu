#include "../../shogi.h"
#ifdef RANDOM_PLAYER_ENGINE

#include "../../extra/all.h"

// --- Search

PRNG my_rand; // 乱数生成器。引数なしなら自動的にrandomizeされる。
void USI::extra_option(USI::OptionsMap & o) {}
void Search::init() {}
void Search::clear() {}
void MainThread::think() {
  MoveList<LEGAL_ALL> ml(rootPos);
  Move m = (ml.size() == 0) ? MOVE_RESIGN : ml.at(size_t(my_rand.rand(ml.size()))).move;
  sync_cout << "bestmove " << m << sync_endl;
}
void Thread::search() {}

#endif
