// Definition of index list of input features
// 入力特徴量のインデックスリストの定義

#ifndef NNUE_FEATURES_INDEX_LIST_H_INCLUDED
#define NNUE_FEATURES_INDEX_LIST_H_INCLUDED

#include "../../config.h"

#if (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#include "../nnue_common.h"
#include "../nnue_architecture.h"

namespace YaneuraOu::Eval::NNUE::Features {

// Class template used for feature index list
// 特徴量のインデックスリストに使うクラステンプレート
template <typename T, usize MaxSize>
class ValueList {

public:
	usize size() const { return size_; }
	void resize(usize size) { size_ = size; }
	void push_back(const T& value) { values_[size_++] = value; }
	T& operator[](usize index) { return values_[index]; }
	T* begin() { return values_; }
	T* end() { return values_ + size_; }
	const T& operator[](usize index) const { return values_[index]; }
	const T* begin() const { return values_; }
	const T* end() const { return values_ + size_; }

private:
	T values_[MaxSize];
	usize size_ = 0;
};

//Type of feature index list
// 特徴量のインデックスリストの型
class IndexList
	: public ValueList<IndexType, RawFeatures::MaxActiveDimensions> {
};

} // namespace YaneuraOu::Eval::NNUE::Features

#endif  // (defined(EVAL_NNUE) || defined(EVAL_SFNN))

#endif // NNUE_FEATURES_INDEX_LIST_H_INCLUDED
