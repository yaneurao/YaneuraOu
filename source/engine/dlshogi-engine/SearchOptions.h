#ifndef __SEARCH_OPTIONS_H_INCLUDED__
#define __SEARCH_OPTIONS_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "dlshogi_types.h"

namespace dlshogi {

using namespace YaneuraOu;

// エンジンオプションで設定された定数。
// ※　dlshogiでは構造体化されていない。
struct SearchOptions {

    // エンジンオプションを生やす。
    void add_options(OptionsMap& options);

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

    // エンジンオプションの"DrawValueBlack","DrawValueWhite"の値。
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
    TimePoint pv_interval = 500;

    // 勝率を評価値に変換する時の定数値。
	// 勝率から評価値に変換する際の係数を設定する。
	// 変換部の内部的には、ここで設定した値が1/1000倍されて計算時に使用される。
    float eval_coef = 285;

    // 決着つかずで引き分けとなる手数
    // エンジンオプションの"MaxMovesToDraw"の値。
    // ※　"MaxMovesToDraw"が0(手数による引き分けなし)のときは、この変数には
    //   100000が代入されることになっているので、この変数が0であることは考慮しなくて良い。
    int max_moves_to_draw = 100000;

    // 予測読みの設定
    // エンジンオプションの"USI_Ponder"の値。
    // これがtrueのときだけ "bestmove XXX ponder YYY" のようにponderの指し手を返す。
    // ※　dlshogiは変数名pondering_mode。
    //bool usi_ponder = false;
	// 🌈 ふかうら王では、Engine::usi_ponderがあるのでそちらを見に行けばいいから不要。

    // エンジンオプションの "UCT_NodeLimit" の値。
    // これは、Nodeを作る数の制限。これはメモリ使用量に影響する。
    // 探索したノード数とは異なる。
    // ※　探索rootでの訪問回数(move_count)がこれを超えたらhashfullとして探索を中止する。
    NodeCountType uct_node_limit = 10000000;

    // エンジンオプションの"MultiPV"の値。
    ChildNumType multi_pv = 1;

    // デバッグ用のメッセージの出力を行うかのフラグ。
    // エンジンオプションの"DebugMessage"の値。
    bool debug_message = false;

    // (歩の不成、敵陣2段目の香の不成など)全合法手を生成するのか。
    bool generate_all_legal_moves = false;

	// 入玉ルール
	EnteringKingRule enteringKingRule = EKR_27_POINT;

    // leaf node(探索の末端の局面)でのdf-pn詰みルーチンを呼び出す時のノード数上限
    // 0 = 呼び出さない。
    // エンジンオプションの"LeafDfpnNodesLimit"の値。
    int leaf_dfpn_nodes_limit = 40;
};

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __SEARCH_OPTIONS_H_INCLUDED__
