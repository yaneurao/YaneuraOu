#include "../../shogi.h"
#ifdef MATE_ENGINE

#include "../../extra/all.h"
#include "mate_engine.h" 

using namespace std;
using namespace Search;

// --- 詰み将棋探索

namespace MateEngine
{
}

void USI::extra_option(USI::OptionsMap & o) {}

// --- Search

void Search::init() {}
void Search::clear() { }
void MainThread::think() {
}
void Thread::search() {}

#endif
