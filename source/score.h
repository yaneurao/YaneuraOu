#ifndef SCORE_H_INCLUDED
#define SCORE_H_INCLUDED

#include <variant>
#include <utility>

#include "types.h"

namespace YaneuraOu {

//class Position;

// GUIで要求されている評価値の単位でValueを保持しているclass
class Score {
   public:
	// 詰み
    struct Mate {
		// 詰みまでの手数
        int plies;
    };

    //struct Tablebase {
    //    int  plies;
    //    bool win;
    //};
	// 💡 将棋ではTablebaseは使わない

	// 出力用に変換された評価値
    struct InternalUnits {
        int value;
    };

    Score() = default;

	// ValueからこのScore構造体を構築する。
	// 💡 Stockfishでは、Position classが勝率への変換パラメーターを保持しているので
	//     変換に際してPosition classが必要なのだが、やねうら王では使わないのでコメントアウト。
    Score(Value v /*, const Position& pos*/);

#if !STOCKFISH
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
	// MateかInternalUnitsかのいずれか。
    std::variant<Mate, /* Tablebase ,*/ InternalUnits> score;
};

}

#endif  // #ifndef SCORE_H_INCLUDED
