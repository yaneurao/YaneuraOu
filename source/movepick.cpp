#include "movepick.h"
#if defined(USE_MOVE_PICKER)

#include <algorithm>
//#include <cassert>
#include <iterator>
#include <utility>

#include "bitboard.h"
#include "position.h"

namespace YaneuraOu {
using namespace Eval; // Eval::PieceValue

namespace {
  
// -----------------------
//   指し手オーダリング
// -----------------------

// 指し手を段階的に生成するために現在どの段階にあるかの状態を表す定数
enum Stages: int {

	// -----------------------------------------------------
	//   王手がかっていない通常探索時用の指し手生成
	// -----------------------------------------------------

	MAIN_TT,					// 置換表の指し手を返すフェーズ
	CAPTURE_INIT,				// (CAPTURESの指し手生成)
	GOOD_CAPTURE,				// 捕獲する指し手(CAPTURES_PRO_PLUS)を生成して指し手を一つずつ返す。ただし、SEE値の悪い手(=BAD_CAPTURE)は後回し。
	QUIET_INIT,					// (QUIETの指し手生成)
	GOOD_QUIET,					// CAPTURES_PRO_PLUSで生成しなかった指し手を生成して、一つずつ返す。
	BAD_CAPTURE,				// 捕獲する  悪い指し手(将棋においてそこまで悪い手とは限らない)
	BAD_QUIET,                  // 捕獲しない悪い指し手(これはBAD_CAPTUREのあとに処理する)

	// 将棋ではBAD_CAPTUREをQUIET_の前にやったほうが良いという従来説は以下の実験データにより覆った。
	//  r300, 2585 - 62 - 2993(46.34% R - 25.46)[2016/08/19]
	// b1000, 1051 - 43 - 1256(45.56% R - 30.95)[2016/08/19]

	// -----------------------------------------------------
	//   王手がかっているときの静止探索/通常探索の指し手生成
	// -----------------------------------------------------

	EVASION_TT,					// 置換表の指し手を返すフェーズ
	EVASION_INIT,				// (EVASIONSの指し手を生成)
	EVASION,					// 回避する指し手(EVASIONS)を生成した指し手を一つずつ返す

	// -----------------------------------------------------
	//   通常探索のProbCutの処理のなかから呼び出される用
	// -----------------------------------------------------

	PROBCUT_TT,					// 置換表の指し手を返すフェーズ
	PROBCUT_INIT,				// (PROBCUTの指し手を生成)
	PROBCUT,					// 直前の指し手での駒の価値を上回る駒取りの指し手のみを生成するフェーズ

	// -----------------------------------------------------
	//   静止探索時用の指し手生成
	// -----------------------------------------------------

	QSEARCH_TT,					// 置換表の指し手を返すフェーズ
	QCAPTURE_INIT,				// (QCAPTUREの指し手生成)
	QCAPTURE,					// 捕獲する指し手 + 歩を成る指し手を一手ずつ返す
};

/*
状態遷移の順番は、
	王手がかかっていない時。
		通常探索時 : MAIN_TT    → CAPTURE_INIT  → GOOD_CAPTURE → GOOD_QUIET → BAD_CAPTURE → BAD_QUIET
		静止探索時 : QSEARCH_TT → QCAPTURE_INIT → QCAPTURE
		ProbCut時  : PROBCUT_TT → PROBCUT_INIT  → PROBCUT

		静止探索では、captureする指し手しか生成しないので単にスコア順に並び替えて順番に返せば良い。

	王手がかかっている時。
		通常探索、静止探索共通 : EVASION_TT → EVASION_INIT → EVASION

*/

// -----------------------
//   partial insertion sort
// -----------------------

// Sort moves in descending order up to and including
// a given limit. The order of moves smaller than the limit is left unspecified.
//
// partial_insertion_sort()は指し手を与えられたlimitより、ExtMove::valueが大きいものだけを降順でソートする。
// limitよりも小さい値の指し手の順序については、不定。(sortしたときに末尾のほうに移動する)
// 将棋だと指し手の数が多い(ことがある)ので、数が多いときは途中で打ち切ったほうがいいかも。
// 現状、全体時間の6.5～7.5%程度をこの関数で消費している。
// (長い時間思考させるとこの割合が増えてくる)
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

	for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
		if (p->value >= limit)
		{
			ExtMove tmp = *p, *q;
			*p          = *++sortedEnd;
			for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
				// ■ 備考
				// ここ、ExtMove同士の operator<()を呼び出している。
				// これはExtMove::valueを比較するように定義されている。
				*q = *(q - 1);
			*q = tmp;
		}
}

} // end of namespace

// 指し手オーダリング器

// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search

// MovePickerクラスのコンストラクタ。引数として、どの種類の手を生成するかを決定するための情報、
// どの手を優先的に（おそらく良い手を）ソートするか、そして現在のノードで手順の順序がどれほど重要かを渡します。

MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const PawnHistory* ph,
                       int pl
#if !STOCKFISH
                       ,bool generate_all_legal_moves
#endif
                       ) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    pawnHistory(ph),
    ttMove(ttm),
    depth(d),
    ply(pl)
#if STOCKFISH
#else
    ,
    generate_all_legal_moves(generate_all_legal_moves)
#endif
{
    // 次の指し手生成の段階
    // 王手がかかっているなら王手回避のフェーズへ。さもなくばQSEARCHのフェーズへ。

#if STOCKFISH
	if (pos.in_check())
        // 王手がかかっているなら回避手
		stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        // 王手がかかっていないなら通常探索用/静止探索の指し手生成
        // ⇨ 通常探索から呼び出されたのか、静止探索から呼び出されたのかについてはdepth > 0 によって判定できる。
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT)
              + !(ttm && pos.pseudo_legal(ttm));
#else
	// 🌈 やねうら王では、pos.pseudo_legal()にgenerate_all_legal_movesを渡してやる必要がある。

    if (pos.in_check())
        // 王手がかかっているなら回避手
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm, generate_all_legal_moves));

    else
        // 王手がかかっていないなら通常探索用/静止探索の指し手生成
        // ⇨ 通常探索から呼び出されたのか、静止探索から呼び出されたのかについてはdepth > 0 によって判定できる。
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT)
              + !(ttm && pos.pseudo_legal(ttm, generate_all_legal_moves));
#endif

	// ⇨ Stockfish 16のコード、ttm(置換表の指し手)は無条件でこのMovePickerが返す1番目の指し手としているが、これだと
    //    TTの指し手だけで千日手になってしまうことがある。これは、将棋ではわりと起こりうる。
    //    対策としては、qsearchで千日手チェックをしたり、SEEが悪いならskipするなど。
    //  ※　ここでStockfish 14のころのように置換表の指し手に条件をつけるのは良くなさげ。(V7.74l3 と V7.74mとの比較)
    //  →　ただし、その場合、qsearch()で千日手チェックが必要になる。
    //    qsearchでの千日手チェックのコストが馬鹿にならないので、
    //    ⇓このコードを有効にして、qsearch()での千日手チェックをやめた方が得。

    // recaptureの制約なくす。(ただし、やねうら王ではttmは何らかの制約を課す)
    // →　この制約入れないと、TTの指し手だけで16手超えで循環されてしまうとqsearch()で
    //    is_repetition()入れたところで永久ループになる。

    // 置換表の指し手を優遇するコード。
    // depth > -5なら、TT優先。depth <= -5でもcaptureである制約。
    // 単にcapture()にするより、この制約にしたほうが良さげ。(V775a7 vs V775a8)

    // ↓これ参考に変更したほうがよさげ？

    /*
	stage = (pos.in_check() ? EVASION_TT : QSEARCH_TT) +
		!(ttm
			&& (pos.in_check() || depth > -5 || pos.capture(ttm))
			&& pos.pseudo_legal(ttm));
*/

    // 置換表の指し手があるならそれを最初に試す。ただしpseudo_legalでなければならない。
    // 置換表の指し手がないなら、次のstageから開始する。
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.

// 通常探索のProbCutのためのMovePickerコンストラクタ
// : 与えられた閾値以上の静的交換評価（SEE）を持つキャプチャを生成します。
// th = 枝刈りのしきい値
// ⇨ SEEの値がth以上となるcaptureの指し手(歩の成りは含む)だけを生成する。

MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph
#if STOCKFISH
#else
    , bool generate_all_legal_moves
#endif

                       ) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(Value(th))
#if STOCKFISH
#else
    ,generate_all_legal_moves(generate_all_legal_moves)
#endif

    {

    // ProbCutから呼び出されているので王手はかかっていないはず。
    ASSERT_LV3(!pos.in_check());

    // ProbCutにおいて、SEEが与えられたthresholdの値以上の指し手のみ生成する。
    // (置換表の指し手も、この条件を満たさなければならない)
    // 置換表の指し手がないなら、次のstageから開始する。

    stage = PROBCUT_TT
      + !(
        ttm
#if STOCKFISH        
        && pos.capture_stage(ttm)
        && pos.pseudo_legal(ttm) 
#else
        && pos.capture(ttm)
        // 注意 : ⇑ ProbCutの指し手生成(PROBCUT_INIT)で、
        // 歩の成りも生成するなら、ここはcapture_or_pawn_promotion()、しないならcapture()にすること。
        // ただし、TTの指し手は優遇した方が良い可能性もある。
        && pos.pseudo_legal(ttm, generate_all_legal_moves) 
#endif
        
	    && pos.see_ge(ttm, threshold)
	    );
    // ⇨ qsearch()のTTと同様、置換表の指し手に関してはsee_geの条件、
    // つけないほうがいい可能性があるが、やってみたら良くなかった。(V774v2 vs V774v3)
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.

// 各手に数値を割り当ててリストをソートします。
// キャプチャは最も価値のある駒（MVV）に基づいて順序付けられ、
// 良好な履歴を持つキャプチャが優先されます。
// 静かな手は履歴テーブルを使用して順序付けられます。

// QUIETS、EVASIONS、CAPTURESの指し手のオーダリングのためのスコアリング。似た処理なので一本化。
template<GenType Type>
ExtMove* MovePicker::score(MoveList<Type>& ml) {

#if STOCKFISH
	static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");
#else
    static_assert(Type == CAPTURES || Type == CAPTURES_ALL || Type == QUIETS || Type == QUIETS_ALL
                    || Type == EVASIONS || Type == EVASIONS_ALL,
                  "Wrong type");
#endif

	Color us = pos.side_to_move();

	// 自分より価値の安い駒で当たりになっているか
	//[[maybe_unused]] Bitboard threatByLesser[QUEEN + 1];

#if STOCKFISH
	if constexpr (Type == QUIETS)
#else
    if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
	{
#if STOCKFISH
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];

#else

#if 0
		// 🌈　Stockfishのコードを忠実に実装すると将棋ではたくさんの利きを計算しなくてはならないので
		//     非常に計算コストが高くなる。ここでは歩による当たりになっている駒だけ考える。

		// 歩による脅威だけ。
		// squares threatened by pawns
		threatenedByPawn = (~us == BLACK) ? pos.attacks_by<BLACK, PAWN>() : pos.attacks_by<WHITE, PAWN>();

		// 歩以外の自駒で、相手の歩の利きにある駒
		threatened =  (pos.pieces(us,PAWN).andnot(pos.pieces(us))                 & threatenedByPawn );
#endif
		// →　やってみたが強くならないのでコメントアウトする。[2022/04/26]

#endif
	}

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
		{
			// Position::see()を用いると遅い。単に取る駒の価値順に調べたほうがパフォーマンス的にもいい。
			// 歩が成る指し手もあるのでこれはある程度優先されないといけない。
			// CAPTURE系である以上、打つ指し手は除外されている。
			// CAPTURES_PRO_PLUSで生成しているので歩の成る指し手が混じる。
			
			// MVV-LVAだが、将棋ではLVAあんまり関係なさげだが(複数の駒である1つの駒が取れるケースがチェスより少ない)、
			// Stockfish 9に倣いMVV + captureHistoryで処理する。

			// ここに来るCAPTURESに歩の成りを含めているので、捕獲する駒(pos.piece_on(to_sq(m)))がNO_PIECEで
			// ある可能性については考慮しておく必要がある。
			// → Eval::CapturePieceValuePlusPromote()を用いて計算。
			// → しかしこのあとsee_ge()の引数に使うのだが、see_ge()ではpromotionの価値を考慮してないので、
			//    ここでpromotionの価値まで足し込んでしまうとそこと整合性がとれなくなるのか…。

			m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
						+ 7 * int(Eval::PieceValue[capturedPiece]) + 1024 * bool(pos.check_squares(pt) & to);
			// →　係数を掛けてるのは、
			// このあと、GOOD_CAPTURE で、
			//	return pos.see_ge(*cur, Value(-cur->value))
			// のようにしてあって、要するにsee_ge()の時のスケール(PieceValue)に変換するため。
			// 
			// Stockfishとは駒点が異なるので、この部分の係数を調整する必要がある。
			//
		}
#if STOCKFISH
		else if constexpr (Type == QUIETS)
#else
        else if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
		{
			// 駒を取らない指し手をオーダリングする。
			// ここ、歩以外の成りも含まれているのだが…。
			// →　指し手オーダリングは、quietな指し手の間での優劣を付けたいわけで、
			//    駒を成るような指し手はどうせevaluate()で大きな値がつくからそっちを先に探索することになる。

			m.value  =  2 * (*mainHistory)[us][m.from_to()];
            m.value +=  2 * (*pawnHistory)[pawn_history_index(pos)][pc][to];
			m.value +=      (*continuationHistory[0])[pc][to];
			m.value +=      (*continuationHistory[1])[pc][to];
			m.value +=      (*continuationHistory[2])[pc][to];
			m.value +=      (*continuationHistory[3])[pc][to];
			m.value +=      (*continuationHistory[5])[pc][to];

			// bonus for checks
			m.value += (bool(pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;
			// これ、効果があるのか検証したほうが良さげ。

#if STOCKFISH
			// penalty for moving to a square threatened by a lesser piece
			// or bonus for escaping an attack by a lesser piece.

			// 格下の駒に脅かされているマスに移動する際のペナルティ  
			// または格下の駒による攻撃から逃れる際のボーナス

			//  📓 移動元の駒が安い駒で当たりになっている場合、
			//      移動させることでそれを回避できるなら価値を上げておく。

			if (KNIGHT <= pt && pt <= QUEEN)
			{
				static constexpr int bonus[QUEEN + 1] = { 0, 0, 144, 144, 256, 517 };
				int v = threatByLesser[pt] & to ? -95 : 100 * bool(threatByLesser[pt] & from);
				m.value += bonus[pt] * v;
			}

			// → Stockfishのコードそのままは書けない。
#endif

			// lowPlyHistoryも加算
			if (ply < LOW_PLY_HISTORY_SIZE)
				m.value += 8 * (*lowPlyHistory)[ply][m.from_to()] / (1 + ply);
			
		}
		else // Type == EVASIONS || EVASIONS_ALL
		{
			// 王手回避の指し手をスコアリングする。
			if (pos.capture_stage(m))
				m.value = PieceValue[capturedPiece] + (1 << 28);
			/* 📓 捕獲する指し手に関しては簡易SEE + MVV/LVA
				  
				  被害が小さいように、LVA(価値の低い駒)を動かして取ることを
				  優先されたほうが良いので駒に価値の低い順に番号をつける。
				  そのためのテーブル。

				  💡 LVA = Least Valuable Aggressor。cf.MVV-LVA

				ここ、moved_piece_before()を用いるのが正しい。
				そうしておかないと、同じto,fromである角成りと角成らずの2つの指し手がある時、
				moved_piece_after()だと、角成りの方は、取られた時の損失が多いとみなされてしまい、
				オーダリング上、後回しになってしまう。

				⇑これは、captureの指し手のスコアがそうでない指し手のスコアより
				常に大きくなるようにするための下駄履き。
				
				　captureの指し手の方がそうでない指し手より稀なので、この下駄履きは、captureの時にしておく。
			*/

			else
			{
				// それ以外の指し手に関してはhistoryの値の順番
				m.value = (*mainHistory)[us][m.from_to()] + (*continuationHistory[0])[pc][to];
				if (ply < LOW_PLY_HISTORY_SIZE)
					m.value += 2 * (*lowPlyHistory)[ply][m.from_to()] / (1 + ply);
			}
		}
	}
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.

// 条件を満たす次の手を返します。
// この関数は、トランスポジションテーブル（TT）の手は既に出力されているため、決して返しません。

// ※　この関数の返し値は同時にthis->moveにも格納されるので活用すると良い。filterのなかでも
//   この変数にアクセスできるので、指し手によってfilterするかどうかを選べる。

template<typename Pred>
Move MovePicker::select(Pred filter) {

	for (; cur < endCur; ++cur)
		// filter()のなかで*curにアクセスして判定するのでfilter()は引数を取らない。
		if (*cur != ttMove && filter())
			return *cur++;

	return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.

// これはMovePickerクラスで最も重要なメソッドです。呼び出すたびに、新しい擬似合法手を1つ生成し、
// すべての手が尽きるまで続けます。生成された手のリストから、最も高いスコアを持つ手を選びます。

// ※ 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
// skipQuiets : これがtrueだとQUIETな指し手は返さない。
Move MovePicker::next_move() {

	// 💡 good Quietの閾値
	constexpr int goodQuietThreshold = -14000;

top:
    switch (stage)
    {

    // 置換表の指し手を返すフェーズ
    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    // 置換表の指し手を返したあとのフェーズ
    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {

#if STOCKFISH
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;

        // 駒を捕獲する指し手に対してオーダリングのためのスコアをつける
        endCur = endCaptures = score<CAPTURES>(ml);
#else
        if (generate_all_legal_moves)
        {
            MoveList<CAPTURES_ALL> ml(pos);
            cur = endBadCaptures = moves;
            endCur = endCaptures = score<CAPTURES_ALL>(ml);
        }
        else
        {
            MoveList<CAPTURES> ml(pos);
            cur = endBadCaptures = moves;
            endCur = endCaptures = score<CAPTURES>(ml);
        }
#endif

        // captureの指し手はそんなに数多くないので全数ソートで問題ないし、全数ソートした方が良い。
        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        ++stage;
        goto top;
    }

	// 置換表の指し手を返したあとのフェーズ
	// (killer moveの前のフェーズなのでkiller除去は不要)
	case GOOD_CAPTURE:
		if (select([&]() {
				// moveは駒打ちではないからsee()の内部での駒打ちは判定不要だが…。
				if (pos.see_ge(*cur, -cur->value / 18))
					return true;
				std::swap(*endBadCaptures++, *cur);
				// 損をする捕獲する指し手はあとのほうで試行されるようにendBadCapturesに移動させる
				return false;
			}))
			return *(cur - 1);

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を生成してオーダリング
	case QUIET_INIT:

		if (!skipQuiets)
		{
			/*
			moves          : バッファの先頭
			endBadCaptures : movesから(endBadCaptures - 1) までに bad capturesの指し手が格納されている。

				■ quiet(captures)の指し手を生成した直後。

				|--- 指し手生成用のバッファ ---------------------------------------|
				|    quiet(captures)                        |  未使用のバッファ    |
				|------------------------------------------------------------------|
				↑                                          ↑
				moves = endBadCaptures                    endMoves


				■ quiet(captures)の指し手を生成して、curポインタをインクリメントしながら読み進めているとき。

				|--- 指し手生成用のバッファ ---------------------------------------|
				|   badCaptures |  quiet(captures)           |  未使用のバッファ   |
				|------------------------------------------------------------------|
				↑             ↑              ↑            ↑
				moves      endBadCaptures      cur         endMoves


				■ quiet(captures)の指し手をpartial_sortで並び替え後にある程度スコアがいい指し手を返したとき

				残りの悪いquietの指し手の先頭をbeginBadQuietsにして、curはmovesに移動。これで、badCaptureを処理する。
				そのあと、beginBadQuietsを処理する。


				|--- 指し手生成用のバッファ ---------------------------------------|
				|   badCaptures |  quiet(captures)           |  未使用のバッファ   |
				|------------------------------------------------------------------|
				↑             ↑              ↑            ↑
				moves      endBadCaptures    beginBadQuiets endMoves
				 ↑cur

			*/

#if STOCKFISH
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);
#else
			if (generate_all_legal_moves)
			{
                MoveList<QUIETS_ALL> ml(pos);
                endCur = endGenerated = score<QUIETS_ALL>(ml);
			}
			else
			{
                MoveList<QUIETS> ml(pos);
                // 駒を捕獲しない指し手に対してオーダリングのためのスコアをつける
                endCur = endGenerated = score<QUIETS>(ml);
			}
            // ⚠ ここ⇑、CAPTURE_INITで生成した指し手に歩の成りの指し手が
			//     含まれているなら、それを除外しなければならない。
#endif						

			/*
				📓

					指し手を部分的にソートする。depthに線形に依存する閾値で。
					(depthが低いときに真面目に全要素ソートするのは無駄だから)
			
					将棋では平均合法手は100手程度。(以前は80手程度だったが、
					AI同士の対局では終局までの平均手数が伸びたので相対的に
					終盤が多くなり、終盤は手駒を持っていることが多いから、
					そのため平均合法手が増えた。)

					また、合法手の最大は、593手。
			 
					それに対して、チェスの平均合法手は40手、合法手の最大は、218手と言われている。
			
					insertion sortの計算量は、O(n^2) で、将棋ではわりと悩ましいところ。
					sortする個数が64以上などはquick sortに切り替えるなどした方がいい可能性もある。
			*/


			partial_insertion_sort(cur, endCur, -3560 * depth);
		}

		++stage;
		[[fallthrough]];

		// 駒を捕獲しない指し手を返す。
		// (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
		// ※　これ、指し手の数が多い場合、AVXを使って一気に削除しておいたほうが良いのでは..
	case GOOD_QUIET:
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

		// Prepare the pointers to loop over the bad captures
		// bad capturesの指し手を返すためにポインタを準備する。
		// 📝　bad capturesの先頭を指すようにする。これは指し手生成バッファの先頭からの領域を再利用している。

		cur    = moves;
        endCur = endBadCaptures;

		++stage;
		[[fallthrough]];

		// see()が負の指し手を返す。
	case BAD_CAPTURE:
		if (select([]() { return true; }))
			return *(cur - 1);

		// Prepare the pointers to loop over the bad quiets
		// 悪いquietの手をループするためのポインタを準備します

        cur    = endCaptures;
		endCur = endGenerated;

		++stage;
		[[fallthrough]];

	case BAD_QUIET:
		if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

		return Move::none();

	// 王手回避手の生成
    case EVASION_INIT : {
#if STOCKFISH
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);
#else
		if (generate_all_legal_moves)
		{
            MoveList<EVASIONS_ALL> ml(pos);
            cur    = moves;
            endCur = endGenerated = score<EVASIONS_ALL>(ml);
		}
		else
		{
            MoveList<EVASIONS> ml(pos);
            cur    = moves;
			// 王手を回避する指し手に対してオーダリングのためのスコアをつける
            endCur = endGenerated = score<EVASIONS>(ml);
		}
#endif

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        ++stage;
        [[fallthrough]];
    }
		// 王手回避の指し手を返す
	case EVASION:
		// 静止探索用の指し手を返す処理
	case QCAPTURE:
		// そんなに数は多くないはずだから、オーダリングがベストのスコアのものを選択する
		return select([]() { return true; });

		// PROBCUTの指し手を返す
	case PROBCUT:
		return select([&]() { return pos.see_ge(*cur, threshold); });
		// threadshold以上のSEE値で、ベストのものを一つずつ返す

	default:
		UNREACHABLE;
		return Move::none();
	}

	ASSERT(false);
	return Move::none(); // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }


// this function must be called after all quiet moves and captures have been generated
// この関数は、すべての静かな手と捕獲手が生成された後に呼び出されなければならない
// 📝 チェス固有の問題っぽいので、この関数は将棋では使わない。
bool MovePicker::can_move_king_or_pawn() const {
	// SEE negative captures shouldn't be returned in GOOD_CAPTURE stage
	// SEEが負になる捕獲手はGOOD_CAPTURE段階では返されるべきではない

	assert(stage > GOOD_CAPTURE && stage != EVASION_INIT);

    for (const ExtMove* m = moves; m < endGenerated; ++m)
	{
		PieceType movedPieceType = type_of(pos.moved_piece(*m));
		if ((movedPieceType == PAWN || movedPieceType == KING) && pos.legal(*m))
			return true;
	}
	return false;
}

} // namespace YaneuraOu

#endif // defined(USE_MOVE_PICKER)
