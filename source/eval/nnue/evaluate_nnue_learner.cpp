﻿// NNUE評価関数の学習時用のコード

#include "../../config.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include <random>
#include <fstream>

#include "../../learn/learn.h"
#include "../../learn/learning_tools.h"

#include "../../position.h"
#include "../../usi.h"
#include "../../misc.h"

#include "../evaluate_common.h"

#include "evaluate_nnue.h"
#include "evaluate_nnue_learner.h"
#include "trainer/features/factorizer_feature_set.h"
#if defined(EVAL_NNUE_HALFKPE9)
// halfKPE9の時用のfactorizer
#include "trainer/features/factorizer_half_kpe9.h"
#else
// 通常のNNUEのときのfactorizer
#include "trainer/features/factorizer_half_kp.h"
#endif
#include "trainer/trainer_feature_transformer.h"
#include "trainer/trainer_input_slice.h"
#include "trainer/trainer_affine_transform.h"
#include "trainer/trainer_affine_transform_sparse_input.h"
#include "trainer/trainer_clipped_relu.h"
#include "trainer/trainer_sum.h"

namespace Eval {

namespace NNUE {

namespace {

// 学習データ
std::vector<Example> examples;

// examplesの排他制御をするMutex
std::mutex examples_mutex;

// ミニバッチのサンプル数
u64 batch_size;

// 乱数生成器
std::mt19937 rng;

// 学習器
std::shared_ptr<Trainer<Network>> trainer;

// 学習率のスケール
double global_learning_rate_scale;

// 学習率のスケールを取得する
double GetGlobalLearningRateScale() {
  return global_learning_rate_scale;
}

// ハイパーパラメータなどのオプションを学習器に伝える
void SendMessages(std::vector<Message> messages) {
  for (auto& message : messages) {
    trainer->SendMessage(&message);
    ASSERT_LV3(message.num_receivers > 0);
  }
}

}  // namespace

// 学習の初期化を行う
void InitializeTraining(double eta1, u64 eta1_epoch,
                        double eta2, u64 eta2_epoch, double eta3) {
  std::cout << "Initializing NN training for "
            << GetArchitectureString() << std::endl;

  ASSERT(feature_transformer);
  ASSERT(network);
  trainer = Trainer<Network>::Create(network.get(), feature_transformer.get());

  if (Options["SkipLoadingEval"]) {
    trainer->Initialize(rng);
  }

  global_learning_rate_scale = 1.0;
  EvalLearningTools::Weight::init_eta(eta1, eta2, eta3, eta1_epoch, eta2_epoch);
}

// ミニバッチのサンプル数を設定する
void SetBatchSize(u64 size) {
  ASSERT_LV3(size > 0);
  batch_size = size;
}

// 学習率のスケールを設定する
void SetGlobalLearningRateScale(double scale) {
  global_learning_rate_scale = scale;
}

// ハイパーパラメータなどのオプションを設定する
void SetOptions(const std::string& options) {
  std::vector<Message> messages;
  for (const auto& option : Split(options, ',')) {
    const auto fields = Split(option, '=');
    ASSERT_LV3(fields.size() == 1 || fields.size() == 2);
    if (fields.size() == 1) {
      messages.emplace_back(fields[0]);
    } else {
      messages.emplace_back(fields[0], fields[1]);
    }
  }
  SendMessages(std::move(messages));
}

// 学習用評価関数パラメータをファイルから読み直す
void RestoreParameters(const std::string& dir_name) {
  const std::string file_name = Path::Combine(dir_name, NNUE::kFileName);
  std::ifstream stream(file_name, std::ios::binary);
  Tools::Result result = ReadParameters(stream);
  ASSERT(result.is_ok());

  SendMessages({{"reset"}});
}

// 学習データを1サンプル追加する
void AddExample(Position& pos, Color rootColor,
                const Learner::PackedSfenValue& psv, double weight) {
  Example example;
  if (rootColor == pos.side_to_move()) {
    example.sign = 1;
  } else {
    example.sign = -1;
  }
  example.psv = psv;
  example.weight = weight;

  Features::IndexList active_indices[2];
  for (const auto trigger : kRefreshTriggers) {
    RawFeatures::AppendActiveIndices(pos, trigger, active_indices);
  }
  if (pos.side_to_move() != BLACK) {
    active_indices[0].swap(active_indices[1]);
  }
  for (const auto color : COLOR) {
    std::vector<TrainingFeature> training_features;
    for (const auto base_index : active_indices[color]) {
      static_assert(Features::Factorizer<RawFeatures>::GetDimensions() <
                    (1 << TrainingFeature::kIndexBits), "");
      Features::Factorizer<RawFeatures>::AppendTrainingFeatures(
          base_index, &training_features);
    }
    std::sort(training_features.begin(), training_features.end());

    auto& unique_features = example.training_features[color];
    for (const auto& feature : training_features) {
      if (!unique_features.empty() &&
          feature.GetIndex() == unique_features.back().GetIndex()) {
        unique_features.back() += feature;
      } else {
        unique_features.push_back(feature);
      }
    }
  }

  std::lock_guard<std::mutex> lock(examples_mutex);
  examples.push_back(std::move(example));
}

// 評価関数パラメーターを更新する
void UpdateParameters(u64 epoch) {
  ASSERT_LV3(batch_size > 0);

  EvalLearningTools::Weight::calc_eta(epoch);
  const auto learning_rate = static_cast<LearnFloatType>(
      get_eta() / batch_size);

  std::lock_guard<std::mutex> lock(examples_mutex);
  std::shuffle(examples.begin(), examples.end(), rng);
  while (examples.size() >= batch_size) {
    std::vector<Example> batch(examples.end() - batch_size, examples.end());
    examples.resize(examples.size() - batch_size);

    const auto network_output = trainer->Propagate(batch);

    std::vector<LearnFloatType> gradients(batch.size());
    for (std::size_t b = 0; b < batch.size(); ++b) {
      const auto shallow = static_cast<Value>(Round<std::int32_t>(
          batch[b].sign * network_output[b] * kPonanzaConstant));
      const auto& psv = batch[b].psv;
      const double gradient = batch[b].sign * Learner::calc_grad(shallow, psv);
      gradients[b] = static_cast<LearnFloatType>(gradient * batch[b].weight);
    }

    trainer->Backpropagate(gradients.data(), learning_rate);
  }
  SendMessages({{"quantize_parameters"}});
}

// 学習に問題が生じていないかチェックする
void CheckHealth() {
  SendMessages({{"check_health"}});
}

}  // namespace NNUE

// 評価関数パラメーターをファイルに保存する
void save_eval(std::string dir_name) {
  auto eval_dir = Path::Combine(Options["EvalSaveDir"], dir_name);
  std::cout << "save_eval() start. folder = " << eval_dir << std::endl;

  // すでにこのフォルダがあるならCreateFolder()に失敗するが、
  // 別にそれは構わない。なければ作って欲しいだけ。
  // また、EvalSaveDirまでのフォルダは掘ってあるものとする。
  Directory::CreateFolder(eval_dir);

  if (Options["SkipLoadingEval"] && NNUE::trainer) {
    NNUE::SendMessages({{"clear_unobserved_feature_weights"}});
  }

  const std::string file_name = Path::Combine(eval_dir, NNUE::kFileName);
  std::ofstream stream(file_name, std::ios::binary);
  const bool result = NNUE::WriteParameters(stream);

  if (!result)
  {
      std::cout << "Error!! : save_eval() failed." << std::endl;
      Tools::exit();
  }

  std::cout << "save_eval() finished." << std::endl;
}

// 現在のetaを取得する
double get_eta() {
  return NNUE::GetGlobalLearningRateScale() * EvalLearningTools::Weight::eta;
}

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)
