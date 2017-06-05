#ifndef _ALL_H_
#define _ALL_H_

// すべてのheaderを読み込むheader
// 実験時などですべてのheaderを読み込んでしまいたいときに使う。

#include <sstream>
#include <iostream>
#include <fstream>

#include "../shogi.h"
#include "../bitboard.h"
#include "../position.h"
#include "../search.h"
#include "../thread.h"
#include "../misc.h"
#include "../tt.h"
#include "long_effect.h"
#include "book/book.h"
#include "../learn/learn.h"
#include "kif_converter/kif_convert_tools.h"

// これもおまけしておく。
using namespace std;

#endif // #ifndef _ALL_H_
