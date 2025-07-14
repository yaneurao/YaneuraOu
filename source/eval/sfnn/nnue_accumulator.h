/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include "../../config.h"
#if defined(EVAL_SFNN)

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace YaneuraOu {
class Position;
}

namespace YaneuraOu::Eval::SFNN {

template<IndexType Size>
struct alignas(CacheLineSize) Accumulator;

template<IndexType TransformedFeatureDimensions>
class FeatureTransformer;

// Class that holds the result of affine transformation of input features
template<IndexType Size>
struct alignas(CacheLineSize) Accumulator {
    std::int16_t               accumulation[COLOR_NB][Size];
    std::int32_t               psqtAccumulation[COLOR_NB][PSQTBuckets];

	// 📓 comuptedとは
	//     上の2つの配列が初期化済みであるかのフラグ。
	//     (Position::set()で探索が開始される時には)最初falseで初期化。
    std::array<bool, COLOR_NB> computed;
};

// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".

// AccumulatorCaches構造体は、スレッドごとのアキュムレータキャッシュを提供します。
// 各キャッシュは、可能なすべてのキングの位置ごとに複数のエントリを含みます。
// アキュムレータをリフレッシュする必要があるとき、キャッシュされたエントリを使うことで、
// 最初から再構築する代わりに効率的にアキュムレータを更新します。
// このアイデアはLuecx（Koivistoの作者）によって最初に提案され、
// 一般に「Finny Tables」と呼ばれています。

struct AccumulatorCaches {

    template<typename Networks>
    AccumulatorCaches(const Networks& networks) {
        clear(networks);
    }

    template<IndexType Size>
    struct alignas(CacheLineSize) Cache {

        struct alignas(CacheLineSize) Entry {
            BiasType       accumulation[Size];
            PSQTWeightType psqtAccumulation[PSQTBuckets];
            Bitboard       byColorBB[COLOR_NB];
            Bitboard       byTypeBB[PIECE_TYPE_NB];

            // To initialize a refresh entry, we set all its bitboards empty,
            // so we put the biases in the accumulation, without any weights on top
            void clear(const BiasType* biases) {

                std::memcpy(accumulation, biases, sizeof(accumulation));
                std::memset((uint8_t*) this + offsetof(Entry, psqtAccumulation), 0,
                            sizeof(Entry) - offsetof(Entry, psqtAccumulation));
            }
        };

        template<typename Network>
        void clear(const Network& network) {
            for (auto& entries1D : entries)
                for (auto& entry : entries1D)
                    entry.clear(network.featureTransformer->biases);
        }

        std::array<Entry, COLOR_NB>& operator[](Square sq) { return entries[sq]; }

        std::array<std::array<Entry, COLOR_NB>, SQUARE_NB> entries;
    };

    template<typename Networks>
    void clear(const Networks& networks) {
        big.clear(networks.big);
        small.clear(networks.small);
    }

    Cache<TransformedFeatureDimensionsBig>   big;
    Cache<TransformedFeatureDimensionsSmall> small;
};

/*
	📓 AccumulatorStateとは

		各Worker(各スレッド)がPosition::do_move()で一手進める時に
		移動させた駒の情報(DirtyPiece)などを保持したい。

		この1回分のdo_move()の情報を保持しておく構造体が、
		AccumulatorStateである。
*/
struct AccumulatorState {
    Accumulator<TransformedFeatureDimensionsBig>   accumulatorBig;
    Accumulator<TransformedFeatureDimensionsSmall> accumulatorSmall;

	// Position::do_move()で移動させた駒の情報を格納している構造体。
    DirtyPiece                                     dirtyPiece;

	// accumulatorを取得する。
	// Size : TransformedFeatureDimensionsBigかTransformedFeatureDimensionsSmallを指定して、
	//        どちらのAccumulatorを取得するかを選ぶ。
    template<IndexType Size>
    auto& acc() noexcept {
        static_assert(Size == TransformedFeatureDimensionsBig
                        || Size == TransformedFeatureDimensionsSmall,
                      "Invalid size for accumulator");

        if constexpr (Size == TransformedFeatureDimensionsBig)
            return accumulatorBig;
        else if constexpr (Size == TransformedFeatureDimensionsSmall)
            return accumulatorSmall;
    }

	// acc()のconst版
    template<IndexType Size>
    const auto& acc() const noexcept {
        static_assert(Size == TransformedFeatureDimensionsBig
                        || Size == TransformedFeatureDimensionsSmall,
                      "Invalid size for accumulator");

        if constexpr (Size == TransformedFeatureDimensionsBig)
            return accumulatorBig;
        else if constexpr (Size == TransformedFeatureDimensionsSmall)
            return accumulatorSmall;
    }

	// DirtyPieceをセットして、かつ、計算済みフラグ(computed)をfalseにする。
    void reset(const DirtyPiece& dp) noexcept;
};

/*
	📓 AccumulatorStackとは

		各Worker(各スレッド)がPosition::do_move()で一手進める時に
		移動させた駒(DirtyPiece)を保持したい。

		従来、StateInfo構造体に格納していたが、それだとエンジンごとに
		StateInfo構造体に手を入れることになり、いい設計とは言いがたい。

		また、StateInfo構造体は、StartSFEN(平手の開始局面)からの情報を持っているが、
		評価関数をroot(探索開始局面)を遡って呼び出すことはないので、そういう意味でも
		StateInfo構造体に持たせるのは無駄でもある。

		そこで、この情報を保持するstackが必要となるが、最大でもMAX_PLYまでしか
		do_move()しないことは保証されているので、その分だけ事前にstd::vectorで
		確保してしまい、std::stackのような操作ができるようにしたclassが、この
		AccumulatorStackである。

		この1回のdo_move()で格納する分が
		AccumulatorState(StackでなくState)構造体である。
*/
class AccumulatorStack {
   public:
    AccumulatorStack() :
		// 💡 stackは事前にMAX_PLY + 1個用意しておけば十分。
        accumulators(MAX_PLY + 1),
		// 事前に1個だけ確保して、計算済みフラグをfalseにしておく。
		// 📝 AccumulatorState.computedが計算済みフラグ。
        size{1} {}

	// 最後にpush()で積んだAccumulatorStateを取得する。std::stack.top()に相当する。
	// 📝 popと違い、要素は取り除かない。
	//     要素を使わないが要素を取り除く目的でpop()を呼び出すことはあるが、
	//     std::stack.top()をして要素を使わないことはありえないので(取り除く機能がないので)
	//     nodiscard属性をつけてある。
    [[nodiscard]] const AccumulatorState& latest() const noexcept;

	// このclassをstackとみなして初期状態に戻す
	// size = 1に戻り、かつ、その1つのAccumulatorState.computed = falseとなる。
    void reset() noexcept;

	// このclassをstackとみなしてDirtyPieceを積む。
    void push(const DirtyPiece& dirtyPiece) noexcept;

	// このclassをstackとみなして最後に積んだDirtyPieceを一つ取り除く。
	void pop() noexcept;

	// 評価関数 本体
    template<IndexType Dimensions>
    void evaluate(const Position&                       pos,
                  const FeatureTransformer<Dimensions>& featureTransformer,
                  AccumulatorCaches::Cache<Dimensions>& cache) noexcept;

   private:
	// latest()と同じだが、mut(mutable : 変更可能)の意味。
	// 最後にpush()した要素の内容を変更したい時に、内部的に用いる。
    [[nodiscard]] AccumulatorState& mut_latest() noexcept;

	// 評価関数の下請け。片側の玉から見た評価値を求める。
	// 評価関数の値 = evaluate_side<BLACK>() + evlauate_side<WHITE>
    template<Color Perspective, IndexType Dimensions>
    void evaluate_side(const Position&                       pos,
                       const FeatureTransformer<Dimensions>& featureTransformer,
                       AccumulatorCaches::Cache<Dimensions>& cache) noexcept;

    template<Color Perspective, IndexType Dimensions>
    [[nodiscard]] std::size_t find_last_usable_accumulator() const noexcept;

    template<Color Perspective, IndexType Dimensions>
    void forward_update_incremental(const Position&                       pos,
                                    const FeatureTransformer<Dimensions>& featureTransformer,
                                    const std::size_t                     begin) noexcept;

    template<Color Perspective, IndexType Dimensions>
    void backward_update_incremental(const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     const std::size_t                     end) noexcept;

	// size : AccumulatorStateを何個accumulatorsに格納しているかの個数。
	//        💡格納するにはこのclassのpush()を用いる。

	std::vector<AccumulatorState> accumulators;
    std::size_t                   size;
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #if defined(EVAL_SFNN)

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
