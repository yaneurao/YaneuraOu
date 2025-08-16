#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "SearchOptions.h"

#include "../../position.h"
#include "../../usi.h"
#include "../../thread.h"
#include "../../movegen.h" // MAX_MOVES

#include "../../eval/deep/nn.h"
#include "../../eval/deep/nn_types.h"

namespace dlshogi {

using namespace YaneuraOu;

void SearchOptions::add_options(OptionsMap& options) {

    // その局面での上位N個の候補手を調べる機能
    // ⇨　これMAX_MOVESで十分。
    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES, [&](const Option& o) {
          multi_pv = ChildNumType(o);
          return std::nullopt;
      }));

    // PV出力間隔
    options.add(  //
      "PV_Interval", Option(500, 0, int_max, [&](const Option& o) {
          pv_interval = TimePoint(o);
          return std::nullopt;
      }));

    // 最大手数
    options.add(  //
      "MaxMovesToDraw", Option(0, 0, 100000, [&](const Option& o) {
          // これ0の時、何らか設定しておかないと探索部でこの手数を超えた時に
          // 引き分け扱いにしてしまうので、無限大みたいな定数の設定が必要。
          max_moves_to_draw = int(o);
          if (max_moves_to_draw == 0)
              max_moves_to_draw = 100000;
          return std::nullopt;
      }));

    // ノードを再利用するか。
    options.add(  //
      "ReuseSubtree", Option(true, [&](const Option& o) {
          reuse_subtree = o;
          return std::nullopt;
      }));


    // 勝率を評価値に変換する時の定数。
    options.add(  //
      "Eval_Coef", Option(285, 1, 10000, [&](const Option& o) {
          eval_coef = float(o);
          return std::nullopt;
      }));

    // 投了値 : 1000分率で
    options.add(  //
      "Resign_Threshold", Option(0, 0, 1000, [&](const Option& o) {
          RESIGN_THRESHOLD = int(options["Resign_Threshold"]) / 1000.0f;
          return std::nullopt;
      }));

    // デバッグ用のメッセージ出力の有無。
    options.add(  //
      "DebugMessage", Option(false, [&](const Option& o) {
          debug_message = o;
          return std::nullopt;
      }));

    // 💡 UCTノードの上限(この値を10億以上にするならWIN_TYPE_DOUBLEをdefineしてコンパイルしないと
    //     MCTSする時の勝率の計算精度足りないし、あとメモリも2TBは載ってないと足りないと思う…)

    //     これはノード制限ではなく、ノード上限を示す。この値を超えたら思考を中断するが、
    // 　  この値を超えていなくとも、持ち時間制御によって思考は中断する。
    // ※　探索ノード数を固定したい場合は、NodesLimitオプションを使うべし。
    options.add(  //
      "UCT_NodeLimit", Option(10000000, 10, 1000000000, [&](const Option& o) {
          uct_node_limit = NodeCountType(o);
          return std::nullopt;
      }));

    // 引き分けの時の値 : 1000分率で
    // 引き分けの局面では、この値とみなす。
    // root color(探索開始局面の手番)に応じて、2通り。

    options.add(  //
      "DrawValueBlack", Option(500, 0, 1000, [&](const Option& o) {
          draw_value_black = int(o) / 1000.0f;
          return std::nullopt;
      }));
    options.add(  //
      "DrawValueWhite", Option(500, 0, 1000, [&](const Option& o) {
          draw_value_white = int(o) / 1000.0f;
          return std::nullopt;
      }));

    // --- PUCTの時の定数

    // これ、探索パラメーターの一種と考えられるから、最適な値を事前にチューニングして設定するように
    // しておき、ユーザーからは触れない(触らなくても良い)ようにしておく。
    // →　dlshogiはoptimizerで最適化するために外だししているようだ。

    // fpu_reductionの値を100分率で設定。
    // c_fpu_reduction_rootは、rootでのfpu_reductionの値。
    options.add(  //
      "C_fpu_reduction", Option(27, 0, 100, [&](const Option& o) {
          c_fpu_reduction = o / 100.0f;
          return std::nullopt;
      }));
    options.add(  //
      "C_fpu_reduction_root", Option(0, 0, 100, [&](const Option& o) {
          c_fpu_reduction_root = o / 100.0f;
          return std::nullopt;
      }));

    options.add(  //
      "C_init", Option(144, 0, 500, [&](const Option& o) {
          c_init = o / 100.0f;
          return std::nullopt;
      }));
    options.add(  //
      "C_base", Option(28288, 10000, 100000, [&](const Option& o) {
          c_base = NodeCountType(o);
          return std::nullopt;
      }));
    options.add(  //
      "C_init_root", Option(116, 0, 500, [&](const Option& o) {
          c_init_root = o / 100.0f;
          return std::nullopt;
      }));
    options.add(  //
      "C_base_root", Option(25617, 10000, 100000, [&](const Option& o) {
          c_base_root = NodeCountType(o);
          return std::nullopt;
      }));

    // softmaxの時のボルツマン温度設定
    // これは、dlshogiの"Softmax_Temperature"の値。(174) = 1.74
    // ※ 100分率で指定する。
    // hcpe3から学習させたmodelの場合、1.40～1.50ぐらいにしないといけない。
    // cf. https://tadaoyamaoka.hatenablog.com/entry/2021/04/05/215431

    options.add(  //
      "Softmax_Temperature", Option(174, 1, 10000, [&](const Option& o) {
          Eval::dlshogi::set_softmax_temperature(o / 100.0f);
          return std::nullopt;
      }));

#if DLSHOGI
    //(*this)["Const_Playout"]               = USIOption(0, 0, int_max);
    // 🤔 Playout数固定。これはNodesLimitでできるので不要。

    dfpn_min_search_millisecs = options["DfPn_Min_Search_Millisecs"];
    // →　ふかうら王では、rootのdf-pnは、node数を指定することにした。
#endif

    // → leaf nodeではdf-pnに変更。
    // 探索ノード数の上限値を設定する。0 : 呼び出さない。
    options.add(  //
      "LeafDfpnNodesLimit", Option(40, 0, 10000, [&](const Option& o) {
          leaf_dfpn_nodes_limit = NodeCountType(o);
          return std::nullopt;
      }));

    // PV lineの即詰みを調べるスレッドの数と1局面当たりの最大探索ノード数。
    options.add("PV_Mate_Search_Threads", Option(1, 0, 256));
    options.add("PV_Mate_Search_Nodes", Option(500000, 0, UINT32_MAX));

    // すべての合法手を生成するのか
    options.add(  //
      "GenerateAllLegalMoves", Option(false, [&](const Option& o) {
          generate_all_legal_moves = o;
          return std::nullopt;
      }));

    // 入玉ルール
    options.add(  //
      "EnteringKingRule", Option(EKR_STRINGS, EKR_STRINGS[EKR_27_POINT], [&](const Option& o) {
          enteringKingRule = to_entering_king_rule(o);
          return std::nullopt;
      }));
}

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
