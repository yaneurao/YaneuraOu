#ifndef __PRINT_INFO_H_INCLUDED__
#define __PRINT_INFO_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)
#include "../../position.h"
#include "Node.h"
#include "UctSearch.h"

namespace dlshogi
{
	struct SearchLimits;
	struct SearchOptions;
}

namespace dlshogi::UctPrint {
// --- 探索情報の表示 ---

// 探索の情報の表示
void PrintPlayoutInformation(const Node*         root,
                             const SearchLimits* po_info,
                             const TimePoint     finish_time,
                             const NodeCountType pre_simulated);

// 探索時間の出力
void PrintPlayoutLimits(const TimeManagement& time_manager, const int playout_limit);

// 再利用した探索回数の出力
void PrintReuseCount(const int count);

// --- bestなnodeの選択 ---

// あるNodeで選択すべき指し手とその時のponderの指し手(そのあとの相手の指し手)を表現する。
struct BestMovePonder {
    // 指し手
    Move move;

    // Ponderの指し手
    Move ponder;

    // moveを選んだ時の期待勝率
    WinType wp;

    BestMovePonder();
    BestMovePonder(Move move_, WinType wp_, Move ponder_) :
        move(move_),
        wp(wp_),
        ponder(ponder_) {}
};

// ベストの指し手とponderの指し手の取得
BestMovePonder get_best_move_multipv(const Node*                       rootNode,
                                     const SearchLimits&               po_info,
                                     const SearchOptions&              options,
                                     YaneuraOu::Search::UpdateContext& context);

}  // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __PRINT_INFO_H_INCLUDED__
