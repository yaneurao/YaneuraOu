#include "movepick.h"

#include <cassert>
#include <limits>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace YaneuraOu {
#if !STOCKFISH
using Eval::PieceValue;
#endif

// 指し手生成・返却の段階。headerの前方宣言と同じYaneuraOu名前空間で定義する。
enum Stages : i8 {
    TT,
    INIT_FIRST_STAGE,

    GOOD_CAPTURE,
    GOOD_QUIET,
    BAD_MOVES,

    QSEARCH,

    EVASION,

    PROBCUT,
};

/*
状態遷移（有効なTT手がなければINIT_FIRST_STAGEから開始する）。

    通常探索: TT → INIT_FIRST_STAGE → GOOD_CAPTURE → GOOD_QUIET → BAD_MOVES
    静止探索: TT → INIT_FIRST_STAGE → QSEARCH
    王手中  : TT → INIT_FIRST_STAGE → EVASION
    ProbCut : TT → INIT_FIRST_STAGE → PROBCUT

    first_stageはGOOD_CAPTURE / QSEARCH / EVASION / PROBCUTのいずれか。
    INIT_FIRST_STAGEでは王手回避手または捕獲手を生成・採点・全数ソートする。
    GOOD_CAPTUREの終了後、quietを生成して部分ソートし、GOOD_QUIETで良い手を返す。
    BAD_MOVESでは悪い捕獲手、悪いquietの順に返す。
    skip_quiet_moves()はquietの生成・返却を省略し、生成済みの悪いquietも除外する。
    静止探索の生成対象は捕獲手のみ。ただし、先行するTT手には捕獲の制約がない。
    王手中は捕獲手以外の王手回避手も生成する。

    将棋ではBAD_CAPTUREをquietの前にした方がよいという従来説を覆した実験:
     r300, 2585 - 62 - 2993(46.34% R - 25.46)[2016/08/19]
    b1000, 1051 - 43 - 1256(45.56% R - 30.95)[2016/08/19]
*/

namespace {

#ifdef USE_AVX512
// Load the Move, and the ExtMove value, into all lanes of 512-bit registers
// 指し手とExtMoveのスコアを、それぞれ512bitレジスターの全レーンへ複製する。
static void splat_extmove(const ExtMove& m, __m512i& move, __m512i& value) {
#if STOCKFISH
    move  = _mm512_set1_epi32(m.raw());
#else
    // やねうら王のraw()は下位16bitのみなので、駒情報を含む32bit全体を保持する。
    move  = _mm512_set1_epi32(m.to_u32());
#endif
    value = _mm512_set1_epi32(m.value);
}

// Sorts up to 16 moves.
// 最大16手をソートする。
struct MoveSorter {
    static constexpr int MAX_ELEMENTS = 16;
    __m512i              sortedValues, sortedMoves;

    explicit MoveSorter(const ExtMove& first) {
        splat_extmove(first, sortedMoves, sortedValues);

        // Set the uninitialized move values to INT_MIN, so that they sort less than any other move
        // 未使用レーンのスコアをINT_MINにして、実際の指し手より後ろに並ぶようにする。
        sortedValues = _mm512_mask_set1_epi32(sortedValues, ~1, std::numeric_limits<int>::min());
    }

    void insert(const ExtMove& m) {
        __m512i move, value;
        splat_extmove(m, move, value);

        // Mask of all elements except the insertion point
        // 挿入位置以外の全要素を示すマスク。
        assert(m.value != std::numeric_limits<int>::min());
        const u16 expand = _kadd_mask16(_mm512_cmplt_epi32_mask(sortedValues, value), -1);

        sortedValues = _mm512_mask_expand_epi32(value, expand, sortedValues);
        sortedMoves  = _mm512_mask_expand_epi32(move, expand, sortedMoves);
    }

    void write_sorted(ExtMove* moves, isize count) const {
        static_assert(sizeof(ExtMove) == 8);
        assert(count <= MAX_ELEMENTS);

        // Because values and moves are stored separately, we need to reassemble the ExtMoves
        // スコアと指し手を別々に保持しているため、書き戻す際にExtMoveへ組み直す。
        auto write = [&](int offset, const __m512i indices) {
            const isize storeCount = count - offset;

            if (storeCount > 0)
            {
                const __m512i extMoves =
                  _mm512_permutex2var_epi32(sortedMoves, indices, sortedValues);
                _mm512_mask_storeu_epi64(moves + offset, (1 << storeCount) - 1, extMoves);
            }
        };

        write(0, _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
        write(8, _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
    }
};
#endif


// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
// 指し手のスコアがlimit以上のものを降順に並べる。limit未満の順序は規定しない。
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    if (end - begin <= 1)
        return;

#if !STOCKFISH
	// 🌈 これもやったほうが良いようだ。

    // With two moves, avoid setting up the vector sorter.
    // 2手なら必要な場合だけ交換し、ベクトルソーターの準備を省く。
    if (end - begin == 2)
    {
        if (begin[1].value >= limit && begin[0].value < begin[1].value)
        {
            const ExtMove first = begin[0];
            begin[0] = begin[1];
            begin[1] = first;
        }
        return;
    }
#endif

    ExtMove *sortedEnd = begin, *p = begin + 1;

#ifdef USE_AVX512
    MoveSorter sorter(*begin);
    for (; p < end; ++p)
    {
        if (p->value >= limit)
        {
            if (sortedEnd - begin + 1 >= MoveSorter::MAX_ELEMENTS)  // sorter full
                // ソーターが満杯なら、残りはスカラー処理へ回す。
                break;

            sorter.insert(*p);
            *p = *++sortedEnd;
        }
    }
    sorter.write_sorted(begin, sortedEnd - begin + 1);
    // Use scalar implementation for any remaining elements
    // 残りの要素にはスカラー版の処理を使う。
#endif

    for (; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            // ExtMoveのoperator<()はvalueを比較する。
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

} // namespace

// 指し手オーダリング器

// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.
// MovePickerのコンストラクタ。返す指し手の種類、良さそうな手を先に並べるための情報、
// 現在のノードで指し手の順序がどれほど重要かを判断する情報を引数として渡す。

// MovePicker constructor for the main search and for the quiescence search
// 通常探索と静止探索で用いるMovePickerのコンストラクタ。
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl
#if !STOCKFISH
                       ,bool generate_all_legal_moves
#endif
                       ) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    ttMove(ttm),
    depth(d),
    ply(pl)
#if !STOCKFISH
    ,
    generate_all_legal_moves(generate_all_legal_moves)
#endif
{
    // 王手中は王手回避、王手なしではdepth > 0なら通常探索、それ以外は静止探索。
    first_stage = pos.checkers() ? EVASION : depth > 0 ? GOOD_CAPTURE : QSEARCH;

#if STOCKFISH
    stage = ttm && pos.legal(ttm) ? TT : INIT_FIRST_STAGE;
#else
    // やねうら王では、不成を省略するかどうかの指定もpseudo_legal()に渡す。
    stage = ttm && pos.pseudo_legal(ttm, generate_all_legal_moves) && pos.legal(ttm) ? TT : INIT_FIRST_STAGE;
#endif

    // TT手はpseudo_legalとlegalの両方を満たすなら最初に返す。静止探索でも捕獲手には限定しない。
    // TT手が無効なら、対応する指し手生成の段階から開始する。
    // 生成手はselect()で返却直前に合法性を確認する。千日手への対処は探索側で行う。
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
// ProbCut用のコンストラクタ。SEEが指定された閾値以上の捕獲手を返す。
// thresholdは呼び出し元が指定する枝刈りの閾値（probCutBeta - ss->staticEval）。
// ただし、先に返すTT手は捕獲手かつ合法手であることを確認し、SEEでは制限しない。

MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph
#if !STOCKFISH
    , bool generate_all_legal_moves
#endif

                       ) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th)
#if !STOCKFISH
    ,generate_all_legal_moves(generate_all_legal_moves)
#endif

    {

    // ProbCutから呼び出されているので王手はかかっていないはず。
    ASSERT_LV3(!pos.in_check());

    // PROBCUT段階ではSEEがthreshold以上の手を返す。
    // 先行するTT手については、捕獲と合法性の条件を確認する。
    first_stage = PROBCUT;
#if STOCKFISH
    stage = ttm && pos.capture_stage(ttm) && pos.legal(ttm) ? TT : INIT_FIRST_STAGE;
#else
    stage = ttm && pos.capture(ttm) && pos.pseudo_legal(ttm, generate_all_legal_moves) && pos.legal(ttm)
              ? TT : INIT_FIRST_STAGE;
#endif
    // ⇨ qsearch()のTTと同様、置換表の指し手に関してはsee_geの条件、
    // つけないほうがいい可能性があるが、やってみたら良くなかった。(V774v2 vs V774v3)
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiet moves are ordered using the history tables.

// リスト内の各手にソート用の数値を割り当てる。
// 捕獲手は取る駒の価値（MVV）と履歴を用いて採点する。
// quietな手は履歴テーブルなどを用いて採点する。

// QUIETS、EVASIONS、CAPTURESの指し手のオーダリングのためのスコアリング。似た処理なので一本化。
template<GenType Type>
ExtMove* MovePicker::generate_and_score() {

#if STOCKFISH
	static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");
#else
    static_assert(Type == CAPTURES || Type == CAPTURES_ALL || Type == QUIETS || Type == QUIETS_ALL
                    || Type == EVASIONS || Type == EVASIONS_ALL,
                  "Wrong type");
#endif

	const Color us = pos.side_to_move();

	// 自分より価値の安い駒で当たりになっているか

#if STOCKFISH
    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
	if constexpr (Type == QUIETS)
#else
    if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
	{
#if STOCKFISH
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.threats_by(PAWN);
        threatByLesser[ROOK] =
          pos.threats_by(KNIGHT) | pos.threats_by(BISHOP) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.threats_by(ROOK) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;

#else

#if 0
		// 🌈　Stockfishのコードを忠実に実装すると将棋ではたくさんの利きを計算しなくてはならないので
		//     非常に計算コストが高くなる。ここでは歩による当たりになっている駒だけ考える。

		// 歩による脅威だけ。
		// squares threatened by pawns
        // 相手の歩の利きがある升。
		threatenedByPawn = (~us == BLACK) ? pos.attacks_by<BLACK, PAWN>() : pos.attacks_by<WHITE, PAWN>();

		// 歩以外の自駒で、相手の歩の利きにある駒
		threatened =  (pos.pieces(us,PAWN).andnot(pos.pieces(us))                 & threatenedByPawn );
#endif
		// →　やってみたが強くならないのでコメントアウトする。[2022/04/26]

#endif
	}

    MoveList<Type> ml(pos);  // move generation
                            // 指し手を生成する。

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

		const Square    from          = m.from_sq();
		const Square    to            = m.to_sq();
		const Piece     pc            = pos.moved_piece(m);
		const PieceType pt            = type_of(pc);
		const Piece     capturedPiece = pos.piece_on(to);

		if constexpr (Type == CAPTURES)
            // MVVとcaptureHistoryで採点する。ここではSEEを呼ばない。
            // CAPTURESは捕獲手だけを生成し、駒を取らない歩の成りは含めない。

			m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
						+ 7 * PieceValue[capturedPiece];
            // GOOD_CAPTUREでは、このスコアからSEEの閾値（-cur->value / 18）を求める。
            // 駒の価値だけでなくcaptureHistoryも、良い捕獲手と後回しにする捕獲手の分類に影響する。

#if STOCKFISH
		else if constexpr (Type == QUIETS)
#else
        else if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
		{
            // 駒を取らない指し手を履歴で採点する。駒打ちや駒を取らない成りもここに含む。
            // この段階ではevaluate()を呼ばず、以下のスコアで探索順を決める。
            // Use a local accumulator to eliminate redundant stores to m.value
            // ローカル変数へ集計し、m.valueへの冗長な書き込みをなくす。

            // histories
            // 各種の履歴テーブル。

            int value = 2 * (*mainHistory)[us][m.raw()];
            value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            value += (*continuationHistory[0])[pc][to];
            value += (*continuationHistory[1])[pc][to];
            value += (*continuationHistory[2])[pc][to];
            value += (*continuationHistory[3])[pc][to];
            value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            // SEEの条件を満たす王手にボーナスを加える。
            value += ((pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;
            // これ、効果があるのか検証したほうが良さげ。

#if STOCKFISH
            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.

            // より価値の低い敵駒の利きへ移動する手にはペナルティ、
            // そのような利きから逃れる手にはボーナスを与える。

            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            value += PieceValue[pt] * v;

            // → Stockfishのコードそのままは書けない。
#endif

            // lowPlyHistoryも加算
            if (ply < LOW_PLY_HISTORY_SIZE)
                value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);

            m.value = value;

        }
		else // Type == EVASIONS
			 // || EVASIONS_ALL
		{
			// 王手回避の指し手をスコアリングする。
			if (pos.capture_stage(m))
				m.value = PieceValue[capturedPiece] + (1 << 28);
            // 捕獲手は(1 << 28)を加えて非捕獲手より優先し、取る駒の価値で並べる。
            // ここではSEEもLVAも用いず、動かす駒の価値や成りの価値は採点に含めない。

			else
				// それ以外の指し手に関してはhistoryの値の順番
				m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
		}
	}
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.

// 指定された条件を満たす次の手を返す。
// TT手は先に返しているため、ここでは重複して返さない。
// filter()は引数を取らず、必要に応じてcurの指す候補を参照する。

template<typename Pred>
Move MovePicker::select(Pred filter) {

	for (; cur < endCur; ++cur)
		// filter()のなかで*curにアクセスして判定するのでfilter()は引数を取らない。
        // Check legality only after the candidate passes the stage's filter.
        // 段階ごとの選別を通過し、実際に返す候補だけ合法性を確認する。
#if STOCKFISH
        if (*cur != ttMove && filter())
		// Stockfishでは、指し手生成で疑似合法手を生成するのではなく、
		// いきなり合法手を生成するようになった。だから、ここでlegal判定は不要である。
#else
		// 将棋では、利き盤面を持っていないなら、いきなり合法手を生成するのはコストが大きい。
		// ここで指し手を返すときにlegalityのチェックをしたほうが低コストとなる。
		if (*cur != ttMove && filter() && pos.legal(*cur))
#endif
			return *cur++;

	return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
// MovePickerの中心となるメソッド。呼び出しごとに合法手を一手返し、
// 指し手が尽きるとMove::none()を返す。各段階の採点と並び順に従って選ぶ。
// 通常探索ではquietを部分ソートするため、全体の厳密なスコア降順とは限らない。
// TT手は最初に返し、それ以降は重複を除外する。skipQuietsならquietを返さない。
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
dispatch:
    switch (stage)
    {

    case TT:
        stage = INIT_FIRST_STAGE;
        return ttMove;

    case INIT_FIRST_STAGE:
        cur = endBadMoves = moves;

        if (first_stage == EVASION)
#if STOCKFISH
            endCur = generate_and_score<EVASIONS>();
#else
            endCur = generate_all_legal_moves ? generate_and_score<EVASIONS_ALL>()
                                             : generate_and_score<EVASIONS>();
#endif
        else
#if STOCKFISH
            endCur = generate_and_score<CAPTURES>();
#else
            endCur = generate_all_legal_moves ? generate_and_score<CAPTURES_ALL>()
                                             : generate_and_score<CAPTURES>();
#endif

        // 捕獲手と王手回避手は、生成後に全数ソートする。
        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        stage = first_stage;
        goto dispatch;

    case GOOD_CAPTURE:
        if (Move move = select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                *endBadMoves++ = *cur;
                return false;
            }))
            return move;

        endBadCaptures = endBadMoves;

        if (!skipQuiets)
        {
            // initialize quiet moves
            // quietな指し手を生成するための初期化。
            /*
            指し手バッファ（区間はすべて半開区間）。

                [ 悪い捕獲手 | quietを生成する領域                       ]
                moves        endBadCaptures = endBadMoves = cur

            返却済み等の捕獲手の領域を再利用してquietを生成する。
            GOOD_QUIETで条件を満たさなかった手をendBadMovesへ順に詰める。

                [ 悪い捕獲手 | 悪いquiet | 処理済みquiet | 未処理quiet ]
                moves        endBadCaptures             cur          endCur
                                        endBadMoves

            BAD_MOVESでは[moves, endBadMoves)を走査する。
            quietをskipする場合はendBadCapturesまでに縮める。
            */
            cur = endBadMoves;
#if STOCKFISH
            endCur = generate_and_score<QUIETS>();
#else
            // 駒を取らない成りと駒打ちもQUIETSに含む。
            endCur = generate_all_legal_moves ? generate_and_score<QUIETS_ALL>()
                                             : generate_and_score<QUIETS>();
#endif

            // 深さに応じた閾値で部分ソートする。浅い探索では全数ソートを避ける。
            // goodQuietThresholdとは別の閾値であり、良いquietの全数ソートを保証しない。
            partial_insertion_sort(cur, endCur, -3560 * depth);

            stage = GOOD_QUIET;
            [[fallthrough]];

        case GOOD_QUIET:
            if (Move move = select([&]() {
                    if (cur->value > goodQuietThreshold)
                        return true;
                    *endBadMoves++ = *cur;
                    return false;
                }))
                return move;
        }

        // Prepare the pointers to loop over the bad captures
        // 悪い捕獲手から走査を始め、その後に悪いquietを返すようポインタを設定する。
        cur    = moves;
        endCur = endBadMoves;

        stage = BAD_MOVES;
        [[fallthrough]];

    case BAD_MOVES:
    case QSEARCH:
    case EVASION:
        // TT手はselect()で除外する。王手回避手と静止探索の手は全数ソート済み。
        return select([]() { return true; });

    case PROBCUT:
        return select([&]() { return pos.see_ge(*cur, threshold); });

    default:
        ASSERT(false);
        sf_unreachable();
    }
}

void MovePicker::skip_quiet_moves() {
    skipQuiets = true;
    if (stage == BAD_MOVES || stage == GOOD_QUIET)
        endCur = endBadMoves = endBadCaptures;  // discards current-stage and bad quiet moves
                                              // 現段階の残りのquietと、退避した悪いquietを除外する。
}

} // namespace YaneuraOu
