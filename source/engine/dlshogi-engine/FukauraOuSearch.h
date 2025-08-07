#ifndef __FUKAURAOU_SEARCH_H_INCLUDED__
#define __FUKAURAOU_SEARCH_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../types.h"
#include "dlshogi_types.h"

namespace dlshogi {
// エンジンオプションで設定された定数。
// ※　dlshogiでは構造体化されていない。
struct SearchOptions {
    // PUCTの定数

    // KataGoの論文に説明がある。Leela Chess Zeroでも実装されている。
    // https://blog.janestreet.com/accelerating-self-play-learning-in-go/

    float c_fpu_reduction      = 0.27f;
    float c_fpu_reduction_root = 0.00f;

    // AlphaZeroの論文の疑似コードで使われているPuctの式に出てくる。
    // https://science.sciencemag.org/content/362/6419/1140/tab-figures-data

    float         c_init      = 1.44f;
    NodeCountType c_base      = 28288;
    float         c_init_root = 1.16f;
    NodeCountType c_base_root = 25617;

    // --- 千日手の価値
    // →これらの値を直接使わずに、draw_value()を用いること。

    // エンジンオプションの"Draw_Value_Black","Draw_Value_White"の値。
    // 💡 1.0fにすると勝ちと同じ扱い。0.0fにすると負けと同じ扱い。

    float draw_value_black = 0.5f;  // rootcolorが先手で、千日手にした時の価値(≒評価値)。
    float draw_value_white = 0.5f;  // rootcolorが後手で、千日手にした時の価値(≒評価値)。

    // 投了する勝率。これを下回った時に投了する。
    // エンジンオプションの"Resign_Threshold"を1000で割った値
    float RESIGN_THRESHOLD = 0.0f;

    // ノードを再利用するかの設定。
    // エンジンオプションの"ReuseSubtree"の値。
    bool reuse_subtree = true;

    // PVの出力間隔
    // エンジンオプションの"PV_Interval"の値。
    TimePoint pv_interval = 0;

    // 勝率を評価値に変換する時の定数値。
    // dlsearcher::SetEvalCoef()で設定する。
    float eval_coef;

    // 決着つかずで引き分けとなる手数
    // エンジンオプションの"MaxMovesToDraw"の値。
    // ※　"MaxMovesToDraw"が0(手数による引き分けなし)のときは、この変数には
    //   100000が代入されることになっているので、この変数が0であることは考慮しなくて良い。
    int max_moves_to_draw = 512;

    // 予測読みの設定
    // エンジンオプションの"USI_Ponder"の値。
    // これがtrueのときだけ "bestmove XXX ponder YYY" のようにponderの指し手を返す。
    // ※　dlshogiは変数名pondering_mode。
    bool usi_ponder = false;

    // エンジンオプションの "UCT_NodeLimit" の値。
    // これは、Nodeを作る数の制限。これはメモリ使用量に影響する。
    // 探索したノード数とは異なる。
    // ※　探索rootでの訪問回数(move_count)がこれを超えたらhashfullとして探索を中止する。
    NodeCountType uct_node_limit;

    // エンジンオプションの"MultiPV"の値。
    ChildNumType multi_pv;

    // デバッグ用のメッセージの出力を行うかのフラグ。
    // エンジンオプションの"DebugMessage"の値。
    bool debug_message = false;

    // (歩の不成、敵陣2段目の香の不成など)全合法手を生成するのか。
    bool generate_all_legal_moves = false;

    // leaf node(探索の末端の局面)でのdf-pn詰みルーチンを呼び出す時のノード数上限
    // 0 = 呼び出さない。
    // エンジンオプションの"LeafDfpnNodesLimit"の値。
    int leaf_dfpn_nodes_limit;

    // root node(探索開始局面)でのdf-pnによる詰み探索を行う時の調べるノード(局面)数
    // これが0だと詰み探索を行わない。最大で指定したノード数まで詰み探索を行う。
    // ここで指定した数×16バイト、詰み探索用に消費する。
    // 例) 100万を指定すると16MB消費する。
    // 1秒間に100万～200万局面ぐらい調べられると思うので、最大で5秒調べさせたいのであれば500万～1000万ぐらいを設定するのが吉。
    // デフォルトは100万
    // 不詰が証明できた場合はそこで詰み探索は終了する。
    uint32_t root_mate_search_nodes_limit;
};

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __FUKAURAOU_SEARCH_H_INCLUDED__
