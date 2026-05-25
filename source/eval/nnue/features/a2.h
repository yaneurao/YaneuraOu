// NNUE評価関数の入力特徴量A2の定義
// 玉を含む全駒のBonaPiece。後手玉を自玉と同じplaneにマージするv2エンコーディング。

#ifndef CLASSIC_NNUE_FEATURES_A2_H
#define CLASSIC_NNUE_FEATURES_A2_H

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "../../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量A2：玉を含む全駒のBonaPiece
// 後手玉(e_king..fe_end2)は自玉plane(f_king..e_king)にマージするので次元数は e_king
class A2 {
 public:
  static constexpr const char* kName = "A2";
  static constexpr std::uint32_t kHashValue = 0xA20DCB9Bu;
  static constexpr IndexType kDimensions = e_king;
  static constexpr IndexType kMaxActiveDimensions = PIECE_NUMBER_NB;
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);
};

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
