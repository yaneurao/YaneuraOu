// NNUE評価関数の学習クラステンプレートのClippedReLU用特殊化

#ifndef _NNUE_TRAINER_CLIPPED_RELU_H_
#define _NNUE_TRAINER_CLIPPED_RELU_H_

#include "../../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/clipped_relu.h"
#include "trainer.h"

namespace Eval {

namespace NNUE {

// 学習：アフィン変換層
template <typename PreviousLayer>
class Trainer<Layers::ClippedReLU<PreviousLayer>> {
private:
  // 学習対象の層の型
  using LayerType = Layers::ClippedReLU<PreviousLayer>;

public:
  // ファクトリ関数
  static std::shared_ptr<Trainer> Create(
      LayerType* target_layer, FeatureTransformer* feature_transformer) {
    return std::shared_ptr<Trainer>(
        new Trainer(target_layer, feature_transformer));
  }

  // ハイパーパラメータなどのオプションを設定する
  void SendMessage(Message* message) {
    previous_layer_trainer_->SendMessage(message);
    if (ReceiveMessage("check_health", message)) {
      CheckHealth();
    }
  }

  // パラメータを乱数で初期化する
  template <typename RNG>
  void Initialize(RNG& rng) {
    previous_layer_trainer_->Initialize(rng);
  }

  // 順伝播
	// 返し値は出力配列の先頭アドレス。
	// ※　配列の要素の個数は出力次元数×バッチサイズ
	// 　　すなわち、kOutputDimensions×batch.size()
  const LearnFloatType* Propagate(const std::vector<Example>& batch) {
    if (output_.size() < kOutputDimensions * batch.size()) {
      output_.resize(kOutputDimensions * batch.size());
      gradients_.resize(kInputDimensions * batch.size());
    }
    const auto input = previous_layer_trainer_->Propagate(batch);

		// Backpropagate()のために、Propagate()のときのbatch_sizeを記録しておく。
    batch_size_ = static_cast<IndexType>(batch.size());
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        const IndexType index = batch_offset + i;
				// 入力を0.0～1.0でclipする。
        output_[index] = std::max(+kZero, std::min(+kOne, input[index]));

				// 前回CheckHealth()を呼び出してからoutput_[i]で最小の値を出力したときの値、最大の値を出力したときの値を記録しておきたい。
        min_activations_[i] = std::min(min_activations_[i], output_[index]);
        max_activations_[i] = std::max(max_activations_[i], output_[index]);
      }
    }
    return output_.data();
  }

  // 逆伝播
  void Backpropagate(const LearnFloatType* gradients,
                     LearnFloatType learning_rate) {
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        const IndexType index = batch_offset + i;
        gradients_[index] = gradients[index] *
            (output_[index] > kZero) * (output_[index] < kOne);
				
				// gradientsとしては、output_[index]が 0.0から1.0の間であったところのみそのまま出力して、さもなくば0
				// cf. 【学習メモ】ゼロから作るDeep Learning【5章】 : https://qiita.com/yakof11/items/5d37042f689760515072
      }
    }
    previous_layer_trainer_->Backpropagate(gradients_.data(), learning_rate);
  }

private:
  // コンストラクタ
  Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer) :
      batch_size_(0),
      previous_layer_trainer_(Trainer<PreviousLayer>::Create(
          &target_layer->previous_layer_, feature_transformer)),
      target_layer_(target_layer) {


		reset_minmax_activations();
  }

  // 学習に問題が生じていないかチェックする
  void CheckHealth() {
    const auto largest_min_activation = *std::max_element(
        std::begin(min_activations_), std::end(min_activations_));
    const auto smallest_max_activation = *std::min_element(
        std::begin(max_activations_), std::end(max_activations_));

		// largest_min_activationが1.0だと常に1.0しか出力していないわけで、いても仕方ないニューロン
		// largest_max_activationが0.0だと常に0しか出力していないわけで、これもいても仕方ないニューロン
		// こいつらは、何らかの方法でresetすべきだと思う。

    std::cout << "INFO: largest min activation = " << largest_min_activation
              << ", smallest max activation = " << smallest_max_activation
              << std::endl;

		// このタイミングでresetする。Propagate()で記録していく。
		reset_minmax_activations();
  }

  // 入出力の次元数
  static constexpr IndexType kInputDimensions = LayerType::kOutputDimensions;
  static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

  // LearnFloatTypeの定数
  static constexpr LearnFloatType kZero = static_cast<LearnFloatType>(0.0);
  static constexpr LearnFloatType kOne = static_cast<LearnFloatType>(1.0);

  // ミニバッチのサンプル数
	// Backpropagate()のために、Propagate()のときのbatch_sizeを記録しておくための変数。
  IndexType batch_size_;

  // 直前の層のTrainer
  const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

  // 学習対象の層
  LayerType* const target_layer_;

  // 順伝播用バッファ
  std::vector<LearnFloatType> output_;

  // 逆伝播用バッファ
  std::vector<LearnFloatType> gradients_;

	// activationsの統計用の配列のクリア
	void reset_minmax_activations()
	{
		std::fill(std::begin(min_activations_), std::end(min_activations_),
			std::numeric_limits<LearnFloatType>::max());
		std::fill(std::begin(max_activations_), std::end(max_activations_),
			std::numeric_limits<LearnFloatType>::lowest());
	}

  // ヘルスチェック用統計値
  LearnFloatType min_activations_[kOutputDimensions];
  LearnFloatType max_activations_[kOutputDimensions];
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif
