#ifndef __ALL_H_INCLUDED__
#define __ALL_H_INCLUDED__

// すべてのheaderを読み込むheader
// 実験時などで手っ取り早くすべてのheaderを読み込んでしまいたいときに使う。
// (ビルドが遅くなるので最終的には使わないように)

#include <sstream>
#include <iostream>
#include <fstream>

#include "../types.h"
#include "../bitboard.h"
#include "../position.h"
#include "../search.h"
#include "../thread.h"
#include "../misc.h"
#include "../tt.h"
#include "../usi.h"
#include "long_effect.h"
#include "../book/book.h"
#include "../learn/learn.h"
#include "../mate/mate.h"

// これもおまけしておく。
using namespace std;

#endif // ndef __ALL_H_INCLUDED__
