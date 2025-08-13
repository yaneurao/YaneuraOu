#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include <algorithm>  // IWYU pragma: keep
#include <cstddef>

#include "types.h"

// --------------------
//    指し手生成器
// --------------------

namespace YaneuraOu {

#if STOCKFISH

enum GenType {
    CAPTURES,
    QUIETS,
    EVASIONS,
    NON_EVASIONS,
    LEGAL
};

#else

// 将棋のある局面の合法手の最大数。593らしいが、保険をかけて少し大きめにしておく。
constexpr int MAX_MOVES = 600;

// 生成する指し手の種類
enum GenType {
    //
    // 注意)
    // 指し手生成器で生成される指し手はすべてpseudo-legalであるが、
    // LEGAL/LEGAL_ALL以外は自殺手が含まれることがある。
    // (pseudo-legalは自殺手も含むので)
    //
    // そのため、do_moveの前にPosition::legal()でのチェックが必要である。
    //

    QUIETS,                 // 駒を取らない指し手
    CAPTURES,               // 駒を取る指し手

    QUIETS_ALL,             // QUIETS + 歩の不成、大駒の不成で駒を取る手
    CAPTURES_ALL,           // CAPTURES     + 歩の不成、大駒の不成で駒を取る手

    CAPTURES_PRO_PLUS,      // CAPTURES     + 価値のかなりあると思われる成り(歩だけ)
    QUIETS_PRO_MINUS,       // QUIETS - 価値のかなりあると思われる成り(歩だけ)

    CAPTURES_PRO_PLUS_ALL,  // CAPTURES_PRO_PLUS      + 歩の不成、大駒の不成で駒を取る手
    QUIETS_PRO_MINUS_ALL,   // QUIETS_PRO_MINUS + 歩の不成、大駒の不成で駒を取らない手

    // note : 歩の不成で駒を取らない指し手は後者に含まれるべきだが、指し手生成の実装が難しくなるので前者に含めることにした。
    //        オーダリング(movepicker)でなんとかするだろうからそこまで悪くはならないだろうし、普段は
    //		  GenerateAllLegalMovesがオンにして動かさないから良しとする。

    // BonanzaではCAPTURESに銀以外の成りを含めていたが、Aperyでは歩の成り以外は含めない。
    // あまり変な成りまで入れるとオーダリングを阻害する。
    // 本ソースコードでは、NON_CAPTURESとCAPTURESは使わず、CAPTURES_PRO_PLUSとNON_CAPTURES_PRO_MINUSを使う。

    // note : NON_CAPTURESとCAPTURESとの生成される指し手の集合は被覆していない。
    // note : CAPTURES_PRO_PLUSとNON_CAPTURES_PRO_MINUSとの生成される指し手の集合も被覆していない。
    // note : CAPTURES_PRO_PLUS_ALLとNON_CAPTURES_PRO_MINUS_ALLとの生成される指し手の集合も被覆していない。
    // →　被覆させないことで、二段階に指し手生成を分解することが出来る。

    EVASIONS,  // 王手の回避(指し手生成元で王手されている局面であることがわかっているときはこちらを呼び出す)
    EVASIONS_ALL,  // EVASIONS + 歩の不成なども含む。

    NON_EVASIONS,  // 王手の回避ではない手(指し手生成元で王手されていない局面であることがわかっているときのすべての指し手)
    NON_EVASIONS_ALL,  // NON_EVASIONS + 歩の不成などを含む。

    // 以下の2つは、pos.legalを内部的に呼び出すので生成するのに時間が少しかかる。棋譜の読み込み時などにしか使わない。
    LEGAL,      // 合法手すべて。ただし、2段目の歩・香の不成や角・飛の不成は生成しない。
    LEGAL_ALL,  // 合法手すべて

    CHECKS,      // 王手となる指し手(歩の不成などは含まない)
    CHECKS_ALL,  // 王手となる指し手(歩の不成なども含む)

    QUIET_CHECKS,  // 王手となる指し手(歩の不成などは含まない)で、CAPTURESの指し手は含まない指し手
    QUIET_CHECKS_ALL,  // 王手となる指し手(歩の不成なども含む)でCAPTURESの指し手は含まない指し手

    // QUIET_CHECKS_PRO_MINUS,	  // 王手となる指し手(歩の不成などは含まない)で、CAPTURES_PRO_PLUSの指し手は含まない指し手
    // QUIET_CHECKS_PRO_MINUS_ALL, // 王手となる指し手(歩の不成なども含む)で、CAPTURES_PRO_PLUSの指し手は含まない指し手
    // →　これらは実装が難しいので、QUIET_CHECKSで生成してから、歩の成る指し手を除外したほうが良いと思う。

    RECAPTURES,      // 指定升への移動の指し手のみを生成する。(歩の不成などは含まない)
    RECAPTURES_ALL,  // 指定升への移動の指し手のみを生成する。(歩の不成なども含む)
};
#endif

class Position;

// --------------------
//   拡張された指し手
// --------------------

// 指し手とオーダリングのためのスコアがペアになっている構造体。
// オーダリングのときにスコアで並べ替えしたいが、一つになっているほうが並び替えがしやすいのでこうしてある。
// ⇨ Moveがclassになったので、このclass memberを呼び出したいから、Moveから派生させるように変更になった。

struct ExtMove: public Move {

	int value;  // 指し手オーダリング(並び替え)のときのスコア(符号つき32bit)

#if STOCKFISH
    void operator=(Move m) { data = m.raw(); }
#else
    // Move型から暗黙で代入できる。
    // 💡 こうしておけば、MoveList* curに対して *cur++ = move; のように書ける。
    void operator=(const Move m) { data = m.to_u32(); }
#endif

	// 補足 : このクラスの変数をMove型にしたいときは、このクラスの変数を Move(m) のようにすれば良い。

    // Inhibit unwanted implicit conversions to Move
    // with an ambiguity that yields to a compile error.
    // 意図しない暗黙のMoveへの変換を防ぎ、あいまいさによってコンパイルエラーを引き起こします。

    // cf. Fix involuntary conversions of ExtMove to Move : https://github.com/official-stockfish/Stockfish/commit/d482e3a8905ee194bda3f67a21dda5132c21f30b
    operator float() const = delete;


};

// partial_insertion_sort()でExtMoveの並べ替えを行なうので比較オペレーターを定義しておく。
inline bool operator<(const ExtMove& f, const ExtMove& s) { return f.value < s.value; }

// 指し手を生成器本体
// gen_typeとして生成する指し手の種類を指定する。
// mlist : 指し手を返して欲しい指し手生成バッファのアドレス
// 返し値 : 生成した指し手の終端
template<GenType>
Move* generate(const Position& pos, Move* moveList);

#if !STOCKFISH
std::ostream& operator<<(std::ostream& os, ExtMove m);
#endif

// The MoveList struct wraps the generate() function and returns a convenient
// list of moves. Using MoveList is sometimes preferable to directly calling
// the lower level generate() function.
// MoveList構造体はgenerate()関数をラップし、便利な指し手リストを返します。
// MoveListを使用することは、低レベルのgenerate()関数を直接呼び出すよりも
// 好ましい場合があります。

// 💡 指し手生成器のwrapper。範囲forで回すときに便利。

template<GenType T>
struct MoveList {
	/*
		📓
			 局面をコンストラクタの引数に渡して使う。

			 すると指し手が生成され、lastが初期化されるので、
			 このclassのbegin(),end()が正常な値を返すようになる。
			 lastは内部のバッファを指しているので、このクラスのコピーは不可。
    
			 for(auto extmove : MoveList<LEGAL_ALL>(pos)) ...
			 のような書き方ができる。
	*/

    explicit MoveList(const Position& pos) :
        last(generate<T>(pos, moveList)) {}

	// 内部的に持っている指し手生成バッファの先頭
    const Move* begin() const { return moveList; }

	// 生成された指し手の末尾のひとつ先
    const Move* end() const { return last; }

	// 生成された指し手の数
    size_t size() const { return last - moveList; }

	// 生成された指し手のなかに引数で指定された指し手が含まれているかの判定。
    // ⚠ ASSERTなどで用いる。遅いので通常探索等では用いないこと。
    bool        contains(Move move) const { return std::find(begin(), end(), move) != end(); }

#if !STOCKFISH
	// 🌈 i番目の要素を返す
	// 📝 生成された合法手からランダムに選びたい時などに用いる。
    const Move at(size_t i) const {
        ASSERT_LV3(i < size());
        return begin()[i];
    }
#endif

   private:
    // moveList : 指し手生成バッファ。自前で持っている。
	// last     : 生成された指し手の末尾の要素 + 1 を指すポインター。
    Move moveList[MAX_MOVES], *last;
};

// 以前のやねうら王との互換性のために用意。
template<GenType Type>
ExtMove* generateMoves(const Position& pos, ExtMove* moveList)
{
    auto moveList0 = static_cast<Move*>(moveList);
    auto moveList1 = generate<Type>(pos, moveList0);

	// これをexpandして返す。

	// ⚠ 符号型にしておかないと i >= 0 が意味をなさない。
	int size = int(moveList1 - moveList0);
	for (int i = size - 1; i >= 0; --i)
		// MoveからMoveExtへのコピー
		moveList[i] = moveList0[i];

	return moveList + size;
}

}  // namespace YaneuraOu

#endif  // #ifndef MOVEGEN_H_INCLUDED
