#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <cstdint>
#include <variant>

#include "movegen.h"
#include "position.h"
#include "types.h"
#include "usi.h"

namespace YaneuraOu::Benchmark {

// Utility to verify move generation. All the leaf nodes up
// to the given depth are generated and counted, and the sum is returned.

// 指し手生成を検証するためのユーティリティ。
// 指定された深さまでの全ての葉ノードを生成してカウントし、その合計を返す。

// 💡 perftとはperformance testの略。
//     開始局面から深さdepthまで全合法手で進めるときの総node数を数えあげる。
//     指し手生成が正常に行われているかや、生成速度等のテストとして有用。

template<bool Root>
u64 perft(Position& pos, Depth depth) {

    StateInfo st;

    u64   cnt, nodes = 0;
    const bool leaf = (depth == 2);

    for (const auto& m : MoveList<LEGAL_ALL>(pos))
    {
        if (Root && depth <= 1)
            cnt = 1, nodes++;
        else
        {
            pos.do_move(m, st);
            cnt = leaf ? MoveList<LEGAL_ALL>(pos).size() : perft<false>(pos, depth - 1);
            nodes += cnt;
            pos.undo_move(m);
        }
        if (Root)
#if STOCKFISH
            sync_cout << USIEngine::move(m, pos.is_chess960() ) << ": " << cnt << sync_endl;
#else
            sync_cout << USIEngine::move(m /*, pos.is_chess960() */) << ": " << cnt << sync_endl;
#endif
    }
    return nodes;
}

#if STOCKFISH
inline std::variant<u64, PositionSetError>
perft(const std::string& fen, Depth depth , bool isChess960) {
#else
inline std::variant<u64, PositionSetError>
perft(const std::string& fen, Depth depth /*, bool isChess960 */) {

	ElapsedTimer time;
    time.reset();
#endif

    StateInfo st;
	Position  p;

#if STOCKFISH
    if (auto err = p.set(fen, isChess960, &st))
        return {*err};

    return perft<true>(p, depth);
#else
    if (auto err = p.set(fen, &st))
        return {*err};

	// 🌈 やねうら王では、NPS(leaf nodeの数/elapsed)と計測に要した時間も出力する。
	auto nodes = perft<true>(p, depth);
    auto elapsed = time.elapsed() + 1; // ゼロ割防止のために +1

	sync_cout << "Elapsed Time = " << elapsed << " [ms]" << sync_endl;
    sync_cout << 1000 * nodes / elapsed << " NPS" << sync_endl;

	return nodes;
#endif

}
}

#endif  // PERFT_H_INCLUDED
