#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include "config.h"
#if defined(USE_MOVE_PICKER)

#include <array>
//#include <cassert>
#include <cmath>
//#include <cstdint>
//#include <cstdlib>
#include <limits>
//#include <type_traits>

#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "types.h"
#include "position.h"

namespace YaneuraOu {

enum Stages : i8;

// -----------------------
//   MovePicker
// -----------------------

// 指し手オーダリング器


// The MovePicker class is used to pick one legal move at a time from the
// current position. The most important method is next_move(), which emits one
// new legal move on every call, until there are no moves left, when
// Move::none() is returned. In order to improve the efficiency of the alpha-beta
// algorithm, MovePicker attempts to return the moves which are most likely to get
// a cut-off first.
// MovePickerクラスは、現在の局面から一回につき合法手を一手取り出すために用いる。
// 最も重要なメソッドはnext_move()であり、呼び出すたびに新しい合法手を一手返し、
// 指し手が尽きるとMove::none()を返す。alpha-beta探索を効率化するため、
// MovePickerはカットオフを起こす可能性の高い手を先に返そうとする。
//
// やねうら王では擬似合法手を返し、呼び出し元の探索でpos.legal()を確認する。
class MovePicker {
   public:
    // このクラスは指し手生成バッファが大きいので、コピーして使うような使い方は禁止。
    MovePicker(const MovePicker&)            = delete;
    MovePicker& operator=(const MovePicker&) = delete;

    // 通常探索(main search)と静止探索から呼び出されるとき用のコンストラクタ。
    MovePicker(const Position&              /* pos_    */,
               Move                         /* ttMove_ */,
               Depth                        /* depth_  */,
               const ButterflyHistory*      /* mh      */,
               const LowPlyHistory*         /* lph     */,
               const CapturePieceToHistory* /* cph     */,
               const PieceToHistory**       /* ch      */,
               const SharedHistories*       /* sh      */,
               int                          /* ply_    */
#if !STOCKFISH
              ,bool generate_all_legal_moves
#endif
    );

    // 通常探索のProbCut用。SEEがthreshold以上の捕獲手を返す。
    // thresholdは呼び出し元指定の枝刈りの閾値で、直前に取られた駒の価値ではない。
    // 先行するTT手は捕獲手かつpseudo_legalなら返し、SEEの条件は課さない。
    MovePicker(const Position&, Move /* ttMove_ */, int /* threshold_ */, const CapturePieceToHistory*
#if STOCKFISH
#else
    , bool generate_all_legal_moves
#endif
    );

    // 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
    // 指し手が尽きればMove::none()が返る。
    // 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
    Move next_move();

    // 以降のnext_move()でquietを省略し、生成済みの悪いquietも返却対象から外す。
    void skip_quiet_moves();

   private:
    template<typename Pred>
    Move select(Pred);

    // 指し手のオーダリング用
    // GenType == CAPTURES : 捕獲する指し手のオーダリング
    // GenType == QUIETS   : 捕獲しない指し手のオーダリング
    // GenType == EVASIONS : 王手回避の指し手のオーダリング
    template<GenType T>
    ExtMove* generate_and_score();

    const Position& pos;

    // コンストラクタで渡されたhistroyのポインタを保存しておく変数。
    const ButterflyHistory*      mainHistory;
    const LowPlyHistory*         lowPlyHistory;
    const CapturePieceToHistory* captureHistory;
    const PieceToHistory**       continuationHistory;
	const SharedHistories*       sharedHistory;

    // 置換表の指し手(コンストラクタで渡される)
    Move ttMove;

    // cur            : 次に返す候補
    // endCur         : 現在走査する領域の終端
    // endBadCaptures : movesから始まる悪い捕獲手の領域の終端
    // endBadMoves    : 悪い捕獲手と、その直後へ詰めた悪いquietの領域の終端
    ExtMove *cur, *endCur, *endBadCaptures, *endBadMoves;

    // 現在の段階と、TT手の処理後に開始する段階
    Stages stage;
    Stages first_stage;

    // ProbCutでSEE判定に用いる、呼び出し元指定の閾値
    int threshold;

    // コンストラクタで渡された探索深さ
    Depth depth;

    // コンストラクタで渡されたrootからの手数
    int ply;

    // next_move()で、quietな指し手をskipするかのフラグ
    bool skipQuiets = false;

    // 指し手生成バッファ
    // 最大合法手の数 = 593
    // cf. https://www.nara-wu.ac.jp/math/personal/shinoda/bunki.html
    ExtMove moves[MAX_MOVES];

#if STOCKFISH
#else
    // Position::pseudo_legal()に渡す第2パラメーター。
    // 📝 やねうら王では、MovePickerのコンストラクタでもらう。
    bool generate_all_legal_moves;
#endif
};

}  // namespace YaneuraOu

#endif // defined(USE_MOVE_PICKER)
#endif // #ifndef MOVEPICK_H_INCLUDED

