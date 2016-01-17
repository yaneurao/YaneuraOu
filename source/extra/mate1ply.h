#ifndef _MATE1PLY_H_
#define _MATE1PLY_H_

#include "../shogi.h"
#ifdef MATE_1PLY

#include "long_effect.h"

// 超高速1手詰め判定ライブラリ
// cf. 新規節点で固定深さの探索を併用するdf-pnアルゴリズム gpw05.pdf
// →　この論文に書かれている手法をBitboard型の将棋プログラム(やねうら王mini)に適用し、さらに発展させ、改良した。

// 1手詰め判定高速化テーブルに使う1要素
struct alignas(2) MateInfo
{
  union {
    // この形において詰ませるのに必要な駒種
    // bit0..歩(の移動)で詰む。打ち歩は打ち歩詰め。
    // bit1..香打ちで詰む。
    // bit2..桂(の移動または打ちで)詰む
    // bit3,4,5,6 ..銀・角・飛・金打ち(もしくは移動)で詰む
    // bit7..何を持ってきても詰まないことを表現するbit(directions==0かつhand_kindの他のbit==0)
    uint8_t/*HandKind*/ hand_kind;

    // 敵玉から見てこの方向に馬か龍を配置すれば詰む。(これが論文からの独自拡張)
    // これが0ならどの駒をどこに打っても、または移動させても詰まないと言える。(桂は除く)
    Effect8::Directions directions;
  };
};

#endif // MATE_1PLY
#endif // _MATE1PLY_H_
