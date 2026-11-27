#ifndef SCORE_H_INCLUDED
#define SCORE_H_INCLUDED

#include <variant>
#include <utility>

#include "types.h"

namespace YaneuraOu {

#if STOCKFISH
class Position;
#endif

// USI出力用の通常評価値(cp)、または詰みまでの手数を保持する。
class Score {
   public:
	// 詰み
    struct Mate {
		// 詰みまでの手数
        int plies;
    };
#if STOCKFISH

	struct Tablebase {
        int  plies;
        bool win;
    };
	// 💡 将棋ではTablebaseは使わない
#endif

	// 出力用に変換された評価値
    struct InternalUnits {
        int value;
    };

    Score() = default;

	// ValueからこのScore構造体を構築する。
	// Stockfishは局面の駒の構成に応じた勝率モデルでcpへ換算するため、Positionが必要。
	// 将棋版はPawnValueによる固定換算なので、Positionを受け取らない。
#if STOCKFISH
    Score(Value v, const Position& pos);
#else
    Score(Value v);

	// 🌈 Valueの値を(cpへの変換をせずに)そのままScoreに変換する。
    static Score from_internal_value(Value v);
	// 🌈 いま保持している値をValueに変換する。
    Value        to_value() const;
#endif

	// MateかInternalUnitsか、どちらで保持しているかを判定する。
    template<typename T>
    bool is() const {
        return std::holds_alternative<T>(score);
    }

	// MateかInternalUnitsかを選んで変換された値を取得する。
	template<typename T>
    T get() const {
        return std::get<T>(score);
    }

    template<typename F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), score);
    }

   private:
#if STOCKFISH
    std::variant<Mate, Tablebase, InternalUnits> score;
#else
    // 将棋版はMateかInternalUnitsで保持する。
    std::variant<Mate, InternalUnits> score;
#endif
};

}

#endif  // #ifndef SCORE_H_INCLUDED
