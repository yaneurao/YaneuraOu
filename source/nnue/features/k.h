// NNUE評価関数の入力特徴量Kの定義

#ifndef NNUE_FEATURES_K_H_INCLUDED
#define NNUE_FEATURES_K_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../../evaluate.h"
#include "features_common.h"

namespace YaneuraOu::Eval::NNUE::Features {

// 特徴量K：玉の位置
class K {
 public:
  // K+PではKのオフセット適用後の番号がBonaPieceと一致する。
  struct IndexContext {
    IndexContext(Square, Color c) : perspective(c) {}
    IndexType board(Piece pc, Square sq) const {
      return kpp_board_index[pc].from[perspective] + IndexType(perspective == BLACK ? sq : Inv(sq));
    }
    IndexType hand(Color c, PieceType pt, int slot) const {
      return kpp_hand_index[c][pt].from[perspective] + slot;
    }
    IndexType zero() const { return 0; }
    Color perspective;
  };

  // K+A2のA2部分では両玉のplaneを一つにまとめる。
  struct IndexContextA2 {
    IndexContextA2(Square, Color c) : perspective(c) {}
    IndexType board(Piece pc, Square sq) const {
      const IndexType base = kpp_board_index[pc].from[perspective];
      return (base >= e_king ? base - SQ_NB : base) + IndexType(perspective == BLACK ? sq : Inv(sq));
    }
    IndexType hand(Color c, PieceType pt, int slot) const {
      return kpp_hand_index[c][pt].from[perspective] + slot;
    }
    IndexType zero() const { return 0; }
    Color perspective;
  };
  // 特徴量名
  static constexpr const char* kName = "K";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr u32 HashValue = 0xD3CEE169u;

  // 特徴量の次元数
  static constexpr IndexType Dimensions = u32(SQ_NB) * 2;

  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType MaxActiveDimensions = 2;
  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void append_active_indices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void append_changed_indices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added, const Eval::DirtyPiece& dp);
};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif
