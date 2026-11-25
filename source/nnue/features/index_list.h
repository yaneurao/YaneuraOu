// Definition of index list of input features
// 入力特徴量のインデックスリストの定義

#ifndef NNUE_FEATURES_INDEX_LIST_H_INCLUDED
#define NNUE_FEATURES_INDEX_LIST_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"
#include "../nnue_architecture.h"

namespace YaneuraOu::Eval::NNUE::Features {

// Storage dedicated to feature indices; the search's ValueList has a different API.
// 特徴量インデックス専用の格納領域。探索用のmisc.hのValueListとはAPIが異なる。
// 既存のコード生成を維持するため、IndexListとの継承構造は残す。
class IndexListStorage {

public:
	usize size() const { return size_; }
	void resize(usize size) { size_ = size; }
	void push_back(const IndexType& value) { values_[size_++] = value; }
	IndexType& operator[](usize index) { return values_[index]; }
	IndexType* begin() { return values_; }
	IndexType* end() { return values_ + size_; }
	const IndexType& operator[](usize index) const { return values_[index]; }
	const IndexType* begin() const { return values_; }
	const IndexType* end() const { return values_ + size_; }

private:
	IndexType values_[RawFeatures::MaxActiveDimensions];
	usize size_ = 0;
};

//Type of feature index list
// 特徴量のインデックスリストの型
class IndexList
	: public IndexListStorage {
};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // NNUE_FEATURES_INDEX_LIST_H_INCLUDED
