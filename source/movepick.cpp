#include "movepick.h"
#if defined(USE_MOVE_PICKER)

#include <algorithm>
//#include <cassert>
#include <iterator>
#include <utility>

#include "bitboard.h"
#include "position.h"

// 以下、やねうら王独自拡張

// search(),qsearch()の時にcaptureの指し手として歩の成りも含める。(V7.74w1 vs V7.74w2)
#define GENERATE_PRO_PLUS

#include "search.h" // Search::Limits.generate_all_legal_movesによって生成される指し手を変えたいので…。

// パラメーターの自動調整フレームワークからパラメーターの値を読み込む
#include "engine/yaneuraou-engine/yaneuraou-param-common.h"

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
// partial_insertion_sort()のSuperSortを用いた実装
extern void partial_super_sort(ExtMove* start, ExtMove* end, int limit);
extern void super_sort(ExtMove* start, ExtMove* end);

/*
  - 少し高速化されるらしい。
  - 安定ソートではないので並び順が以前のとは異なるから、benchコマンドの探索ノード数は変わる。
  - CPU targetによって実装が変わるのでCPUによってbenchコマンドの探索ノード数は変わる。
*/
#endif

namespace {

// -----------------------
//   LVA
// -----------------------

// 被害が小さいように、LVA(価値の低い駒)を動かして取るほうが優先されたほうが良いので駒に価値の低い順に番号をつける。そのためのテーブル。
// ※ LVA = Least Valuable Aggressor。cf.MVV-LVA

constexpr Value LVATable[PIECE_WHITE] = {
  Value(0), Value(1) /*歩*/, Value(2)/*香*/, Value(3)/*桂*/, Value(4)/*銀*/, Value(7)/*角*/, Value(8)/*飛*/, Value(6)/*金*/,
  Value(10000)/*王*/, Value(5)/*と*/, Value(5)/*杏*/, Value(5)/*圭*/, Value(5)/*全*/, Value(9)/*馬*/, Value(10)/*龍*/,Value(11)/*成金*/
};
constexpr Value LVA(const PieceType pt) { return LVATable[pt]; }
  
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
	GOOD_CAPTURE,				// 捕獲する指し手(CAPTURES_PRO_PLUS)を生成して指し手を一つずつ返す
	REFUTATION,					// killer move,counter move
	QUIET_INIT,					// (QUIETの指し手生成)
	QUIET,						// CAPTURES_PRO_PLUSで生成しなかった指し手を生成して、一つずつ返す。SEE値の悪い手は後回し。
	BAD_CAPTURE,				// 捕獲する悪い指し手(SEE < 0 の指し手だが、将棋においてそこまで悪い手とは限らないが…)

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
	QCHECK_INIT,				// 王手となる指し手を生成
	QCHECK						// 王手となる指し手(- 歩を成る指し手)を返すフェーズ
};

/*
状態遷移の順番は、
	王手がかかっていない時。
		通常探索時 : MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE → REFUTATION → QUIET_INIT → QUIET → BAD_CAPTURE
		ProbCut時  : PROBCUT_TT → PROBCUT_INIT → PROBCUT
		静止探索時 : QSEARCH_TT → QCAPTURE_INIT → QCAPTURE → (王手を生成するなら) QCHECK_INIT → QCHECK

		※ 通常探索時にしか、REFUTATIONを呼び出していないので、すなわちProbCut時と静止探索時には
			killerとかcountermoveの生成はしない。
		※ 通常探索時に GOOD_CAPTUREとBAD_CAPTUREがあるのは、前者でスコアが悪かった指し手をBAD_CAPTUREに回すためである。

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
			*p = *++sortedEnd;
			for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
				*q = *(q - 1);
			*q = tmp;
		}
}

} // end of namespace

// 指し手オーダリング器

// Constructors of the MovePicker class. As arguments, we pass information
// to help it return the (presumably) good moves first, to decide which
// moves to return (in the quiescence search, for instance, we only want to
// search captures, promotions, and some checks) and how important a good
// move ordering is at the current node.

// 引数として、我々は情報を渡します。
// それが最初に（おそらく）良い手を返す手助けとなるため、どの手を返すかを決定するために
// （例えば、静止探索（quiescence search）では、我々は駒を取る手、
// 成る手、およびいくつかの王手だけを探索したい）そして、
// 現在のノードにおいて良い手の順序付けがどれほど重要であるかについてです。

// MovePicker constructor for the main search
// 通常探索(main search)から呼び出されるとき用のコンストラクタ。
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
	const CapturePieceToHistory* cph ,
	const PieceToHistory** ch,
#if defined(ENABLE_PAWN_HISTORY)
	const PawnHistory* ph,
#endif
	Move cm,
	const Move* killers)
	: pos(p), mainHistory(mh), captureHistory(cph) , continuationHistory(ch),
#if defined(ENABLE_PAWN_HISTORY)
	pawnHistory(ph),
#endif
	ttMove(ttm), refutations{ { killers[0], 0 },{ killers[1], 0 },{ cm, 0 } }, depth(d)
{
	// 通常探索から呼び出されているので残り深さはゼロより大きい。
	ASSERT_LV3(d > 0);

	// 次の指し手生成の段階
	// 王手がかかっているなら回避手、かかっていないなら通常探索用の指し手生成
	stage = (pos.in_check() ? EVASION_TT : MAIN_TT) + !(ttm && pos.pseudo_legal(ttm));

	// 置換表の指し手があるならそれを最初に試す。ただしpseudo_legalでなければならない。
	// 置換表の指し手がないなら、次のstageから開始する。
}

// Constructor for quiescence search
//
// qsearch(静止探索)から呼び出される時用。
// rs : recapture square(直前の駒の移動先。この駒を取り返す指し手をいまから生成する)
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
	const CapturePieceToHistory* cph,
	const PieceToHistory** ch
#if defined(ENABLE_PAWN_HISTORY)
	, const PawnHistory* ph
#endif
	, Square rs)
	: pos(p), mainHistory(mh), captureHistory(cph) , continuationHistory(ch)
#if defined(ENABLE_PAWN_HISTORY)
	, pawnHistory(ph)
#endif
	, ttMove(ttm), recaptureSquare(rs), depth(d)
{

	// 静止探索から呼び出されているので残り深さはゼロ以下。
	ASSERT_LV3(d <= 0);

	// 王手がかかっているなら王手回避のフェーズへ。さもなくばQSEARCHのフェーズへ。
//	stage = (pos.in_check() ? EVASION_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));

	// ⇨ Stockfish 16のコード、ttm(置換表の指し手)は無条件でこのMovePickerが返す1番目の指し手としているが、これだと
	//    TTの指し手だけで千日手になってしまうことがある。これは、将棋ではわりと起こりうる。
	//    対策としては、qsearchで千日手チェックをしたり、SEEが悪いならskipするなど。
	//  ※　ここでStockfish 14のころのように置換表の指し手に条件をつけるのは良くなさげ。(V7.74l3 と V7.74mとの比較)
	//  →　ただし、その場合、qsearch()で千日手チェックが必要になる。
	//    qsearchでの千日手チェックのコストが馬鹿にならないので、
	//    ⇓このコードを有効にして、qsearch()での千日手チェックをやめた方が得。  

	// Stockfish 14のコード

	stage = (pos.in_check() ? EVASION_TT : QSEARCH_TT) +
		!(ttm
			&& (pos.in_check() || depth > DEPTH_QS_RECAPTURES || to_sq(ttm) == recaptureSquare)
			&& pos.pseudo_legal(ttm));

}

// Constructor for ProbCut: we generate captures with SEE greater
// than or equal to the given threshold.

// 通常探索時にProbCutの処理から呼び出されるのコンストラクタ。
// th = 枝刈りのしきい値
// SEEの値がth以上となるcaptureの指し手(歩の成りは含む)だけを生成する。
MovePicker::MovePicker(const Position& p, Move ttm, Value th, const CapturePieceToHistory* cph)
	: pos(p), captureHistory(cph) , ttMove(ttm), threshold(th)
{
	ASSERT_LV3(!pos.in_check());

	// ProbCutにおいて、SEEが与えられたthresholdの値以上の指し手のみ生成する。
	// (置換表の指し手も、この条件を満たさなければならない)
	// 置換表の指し手がないなら、次のstageから開始する。
	stage = PROBCUT_TT + !(ttm
#if defined(GENERATE_PRO_PLUS)
								&& pos.capture_or_pawn_promotion(ttm)
#else
								&& pos.capture(ttm)
#endif
								// 注意 : ⇑ ProbCutの指し手生成(PROBCUT_INIT)で、
								// 歩の成りも生成するなら、ここはcapture_or_pawn_promotion()、しないならcapture()にすること。
								// ただし、TTの指し手は優遇した方が良い可能性もある。
								&& pos.pseudo_legal(ttm)
								&& pos.see_ge(ttm, threshold));
	// ⇨ qsearch()のTTと同様、置換表の指し手に関してはsee_geの条件、
	// つけないほうがいい可能性があるが、やってみたら良くなかった。(V774v2 vs V774v3)

}


// MovePicker::score() assigns a numerical value to each move in a list, used
// for sorting. Captures are ordered by Most Valuable Victim (MVV), preferring
// captures with a good history. Quiets moves are ordered using the history tables.

// QUIETS、EVASIONS、CAPTURESの指し手のオーダリングのためのスコアリング。似た処理なので一本化。
template<MOVE_GEN_TYPE Type>
void MovePicker::score()
{
	static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

	// threatened        : 自分より価値の安い駒で当たりになっているか
	// threatenedByPawn  : 敵の歩の利き。
	// threatenedByMinor : 敵の歩・小駒による利き
	// threatenedByRook  : 敵の大駒による利き(やねうら王では使わず)

	// [[maybe_unused]] Bitboard threatenedByPawn, threatenedByMinor, threatenedByRook, threatenedPieces;

	if constexpr (Type == QUIETS)
	{
#if 0
		Color us = pos.side_to_move();
		// squares threatened by pawns
		threatenedByPawn  = pos.attacks_by<PAWN>(~us);
		// squares threatened by minors or pawns
		threatenedByMinor = pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatenedByPawn;
		// squares threatened by rooks, minors or pawns
		threatenedByRook  = pos.attacks_by<ROOK>(~us) | threatenedByMinor;

		// pieces threatened by pieces of lesser material value
		threatened =  (pos.pieces(us, QUEEN) & threatenedByRook)
					| (pos.pieces(us, ROOK)  & threatenedByMinor)
					| (pos.pieces(us, KNIGHT, BISHOP) & threatenedByPawn);
#endif

#if 0
		// →　Stockfishのコードを忠実に実装すると将棋ではたくさんの利きを計算しなくてはならないので
		//     非常に計算コストが高くなる。ここでは歩による当たりになっている駒だけ考える。

		const Color us = pos.side_to_move();

		// 歩による脅威だけ。
		// squares threatened by pawns
		threatenedByPawn = (~us == BLACK) ? pos.attacks_by<BLACK, PAWN>() : pos.attacks_by<WHITE, PAWN>();

		// 歩以外の自駒で、相手の歩の利きにある駒
		threatened =  (pos.pieces(us,PAWN).andnot(pos.pieces(us))                 & threatenedByPawn );
#endif
		// →　やってみたが強くならないのでコメントアウトする。[2022/04/26]
	}

	for (auto& m : *this)
	{
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

			m.value = (7 * int(Eval::CapturePieceValuePlusPromote(pos, m))
					   + (*captureHistory)(pos.moved_piece_after(m), to_sq(m), type_of(pos.piece_on(to_sq(m)))))
					  / 16;
			// →　係数を掛けたり全体を16で割ったりしているのは、
			// このあと、GOOD_CAPTURE で、
			//	return pos.see_ge(*cur, Value(-cur->value))
			// のようにしてあって、要するにsee_ge()の時のスケール(PieceValue)に変換するため。
			// 
			// Stockfishとは駒点が異なるので、この部分の係数を調整する必要がある。
			//
		}
		else if constexpr (Type == QUIETS)
		{
			// 駒を取らない指し手をオーダリングする。
			// ここ、歩以外の成りも含まれているのだが…。
			// →　指し手オーダリングは、quietな指し手の間での優劣を付けたいわけで、
			//    駒を成るような指し手はどうせevaluate()で大きな値がつくからそっちを先に探索することになる。

			Piece     pc = pos.moved_piece_after(m);
			//PieceType pt = type_of(pos.moved_piece_before(m));
			//Square    from = from_sq(m);
			Square    to = to_sq(m);

			m.value  =  2 * (*mainHistory)(pos.side_to_move(), from_to(m));
#if defined(ENABLE_PAWN_HISTORY)
			m.value +=  2 * (*pawnHistory)(pawn_structure(pos), pc, to);
#endif
			m.value +=  2 * (*continuationHistory[0])(pc,to);
			m.value +=      (*continuationHistory[1])(pc,to);
			m.value +=      (*continuationHistory[2])(pc,to) / 4;
			m.value +=      (*continuationHistory[3])(pc,to);
			m.value +=      (*continuationHistory[5])(pc,to);

			// bonus for checks
            //m.value += bool(pos.check_squares(pt) & to) * 16384;
			// TODO : あとで効果を検証する[2023/10/29]


			//	移動元の駒が安い駒で当たりになっている場合、移動させることでそれを回避できるなら価値を上げておく。
#if 0
					+     (threatened & from_sq(m) ?
							 (type_of(pos.moved_piece_before(m)) == QUEEN && !(to_sq(m) & threatenedByRook ) ? 50000
							: type_of(pos.moved_piece_before(m)) == ROOK  && !(to_sq(m) & threatenedByMinor) ? 25000
							:                                                !(to_sq(m) & threatenedByPawn ) ? 15000
							:																					0)
																											    0);
				// → Stockfishのコードそのままは書けない。
#endif

#if 0
					+     (threatened & from_sq(m) ?
							 ((moved_piece == ROOK || moved_piece == BISHOP) && !threatenedByPawn.test(to_sq(m)) ? 50000
						:                                                       !threatenedByPawn.test(to_sq(m)) ? 15000
						:                                                                                          0)
						:                                                                                          0);
#endif
				// →　強くならなかったのでコメントアウト。
					;
		}
		else // Type == EVASIONS
		{
			// 王手回避の指し手をスコアリングする。

			if (pos.capture_or_promotion(m))
				// 捕獲する指し手に関しては簡易SEE + MVV/LVA
				// 被害が小さいように、LVA(価値の低い駒)を動かして取ることを優先されたほうが良いので駒に価値の低い順に番号をつける。そのためのテーブル。
				// ※ LVA = Least Valuable Aggressor。cf.MVV-LVA

				// ここ、moved_piece_before()を用いるのが正しい。
				// そうしておかないと、同じto,fromである角成りと角成らずの2つの指し手がある時、
				// moved_piece_after()だと、角成りの方は、取られた時の損失が多いとみなされてしまい、
				// オーダリング上、後回しになってしまう。

				//m.value = PieceValue[pos.piece_on(to_sq(m))] - Value(type_of(pos.moved_piece(m)))
				//		+ (1 << 28);

				// 上記のStockfishのコードのValue()は関数ではなく単にValue型へcastしているだけ。
				// 駒番号順に価値が低いと考えて(普通は成り駒ではないから)、LVAとしてそれを優先して欲しいという意味。

				m.value = Eval::CapturePieceValuePlusPromote(pos, m)
				        - Value(LVA(type_of(pos.moved_piece_before(m))))
						// ↑ここ、LVAテーブル使わずにPieceValueを64で割るとかできなくもないが、
						// 　下手にやると、香と桂にような価値が近い駒に対して優先順位がつけられない。
						//   愚直にLVAテーブル使うべき。
                        + (1 << 28);
                        // ⇑これは、captureの指し手のスコアがそうでない指し手のスコアより
                        //   常に大きくなるようにするための下駄履き。
						// ※　captureの指し手の方がそうでない指し手より稀なので、この下駄履きは、
						//     captureの時にしておく。
						
			else
				// それ以外の指し手に関してはhistoryの値の順番
				m.value =     (*mainHistory)(pos.side_to_move(), from_to(m))
						  +   (*continuationHistory[0])(pos.moved_piece_after(m), to_sq(m))
#if defined(ENABLE_PAWN_HISTORY)
						  +   (*pawnHistory)(pawn_structure(pos), pos.moved_piece_after(m), to_sq(m))
#endif
				;

		}
	}
}

// Returns the next move satisfying a predicate function.
// It never returns the TT move.
// 
// MovePicker::select()は、Pred(predicate function:述語関数)を満たす次の指し手を返す。
// 置換表の指し手は決して返さない。
// ※　この関数の返し値は同時にthis->moveにも格納されるので活用すると良い。filterのなかでも
//   この変数にアクセスできるので、指し手によってfilterするかどうかを選べる。
template<MovePicker::PickType T, typename Pred>
Move MovePicker::select(Pred filter) {

	while (cur < endMoves)
	{
		// TがBestならBestを探してcurが指す要素と入れ替える。
		// それがttMoveであるなら、もう一周する。
		if constexpr (T == Best)
			std::swap(*cur, *std::max_element(cur, endMoves));

		// filter()のなかで*curにアクセスして判定するのでfilter()は引数を取らない。
		if (*cur != ttMove && filter())
			return *cur++;

		cur++;
	}
	return MOVE_NONE;
}

// Most important method of the MovePicker class. It
// returns a new pseudo-legal move every time it is called until there are no more
// moves left, picking the move with the highest score from a list of generated moves.

// 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
// 指し手が尽きればMOVE_NONEが返る。
// 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
// skipQuiets : これがtrueだとQUIETな指し手は返さない。
Move MovePicker::next_move(bool skipQuiets) {

top:
	switch (stage) {

	// 置換表の指し手を返すフェーズ
	case MAIN_TT:
	case EVASION_TT:
	case QSEARCH_TT:
	case PROBCUT_TT:
		++stage;
		return ttMove;

	// 置換表の指し手を返したあとのフェーズ
	case CAPTURE_INIT:
	case PROBCUT_INIT:
	case QCAPTURE_INIT:
		cur = endBadCaptures = moves;

#if defined(GENERATE_PRO_PLUS)
		// CAPTURE_INITのあとはこのあと残りの指し手を生成する必要があるので、generate_all_legal_movesがtrueなら、CAPTURE_PRO_PLUSで歩の成らずの指し手も生成する。
		// PROBCUT_INIT、QCAPTURE_INITの時は、このあと残りの指し手を生成しないので歩の成らずを生成しても仕方がない。
		if (stage == CAPTURE_INIT)
			endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<CAPTURES_PRO_PLUS_ALL>(pos, cur) : generateMoves<CAPTURES_PRO_PLUS>(pos, cur);
		else if (stage == PROBCUT_INIT)
			// ProbCutでは、歩の成りも生成する。
			endMoves = generateMoves<CAPTURES_PRO_PLUS>(pos, cur);
		else if (stage == QCAPTURE_INIT)
			// qsearchでは歩の成りは不要。駒を取る手だけ生成すれば十分。
			endMoves = generateMoves<CAPTURES>(pos, cur);
#else
		endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<CAPTURES_ALL>(pos, cur) : generateMoves<CAPTURES>(pos, cur);
#endif

		// 駒を捕獲する指し手に対してオーダリングのためのスコアをつける
		score<CAPTURES>();

		// captureの指し手はそんなに数多くないので全数ソートで問題ないし、全数ソートした方が良い。
		partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());

		++stage;
		goto top;

	// 置換表の指し手を返したあとのフェーズ
	// (killer moveの前のフェーズなのでkiller除去は不要)
	case GOOD_CAPTURE:
		if (select<Next>([&]() {
				// Move losing capture to endBadCaptures to be tried later
				// 損をする(SEE値が悪い)captureの指し手はあとで試すためにendBadCapturesに移動させる

				// moveは駒打ちではないからsee()の内部での駒打ちは判定不要だが…。
                return pos.see_ge(*cur, Value(-cur->value)) ?
						// 損をする捕獲する指し手はあとのほうで試行されるようにendBadCapturesに移動させる
						true : (*endBadCaptures++ = *cur, false);
			}))
			return *(cur -1);

		// Prepare the pointers to loop over the refutations array
		// refutations配列に対して繰り返すためにポインターを準備する。
		cur      = std::begin(refutations);
		endMoves = std::end(refutations);

		// If the countermove is the same as a killer, skip it
		// countermoveがkillerと同じならばそれをskipする。

		// ※  refutations[0]と[1]はkiller、refutations[2]はcounter move
		// ※　refutations[]はすべてMove16ではなくMove(上位に移動後の駒が格納されている)と仮定している。
		//     (ゆえにこれが一致すれば同じ駒を移動させる同一の指し手)
		if (   refutations[0].move == refutations[2].move
			|| refutations[1].move == refutations[2].move)
			--endMoves;

		++stage;
		[[fallthrough]];

	// killer move , counter moveを返すフェーズ
	case REFUTATION:

		// 直前に置換表の指し手を返しているし、CAPTURES_PRO_PLUSでの指し手も返しているので
		// それらの指し手は除外する。
		// 直前にCAPTURES_PRO_PLUSで生成している指し手を除外
		// pseudo_legalでない指し手以外に歩や大駒の不成なども除外
		if (select<Next>([&]() { return    *cur != MOVE_NONE
#if defined(GENERATE_PRO_PLUS)
			                            && !pos.capture_or_pawn_promotion(*cur)
#else
										&& !pos.capture(*cur)
#endif
										// 注意 : ここ⇑、CAPTURE_INITで生成した指し手に歩の成りが混じっているなら、
										//		capture_or_pawn_promotion()の方を用いなければならないので注意。
										&&  pos.pseudo_legal(*cur); }))
			return *(cur - 1);

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を生成してオーダリング
	case QUIET_INIT:

		if (!skipQuiets)
		{
			cur = endBadCaptures;

			/*
			moves          : バッファの先頭
			endBadCaptures : movesから(endBadCaptures - 1) までに bad capturesの指し手が格納されている。
				そこ以降はバッファの末尾まで自由に使って良い。

				|--- 指し手生成用のバッファ -----------------------------------|
				| ttMove | killer | captures |  未使用のバッファ               |  captures生成時 (CAPTURES_PRO_PLUS)
				|--------------------------------------------------------------|
				|   badCaptures      |     quiet         |  未使用のバッファ   |  quiet生成時    (NON_CAPTURES_PRO_MINUS)
				|--------------------------------------------------------------|
				↑                  ↑ 
				moves          endBadCaptures
			*/


#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
			// curを32の倍数アドレスになるように少し進めてしまう。
			// これにより、curがalignas(32)されているような効果がある。
			// このあとSuperSortを使うときにこれが前提条件として必要。
			cur = (ExtMove*)Math::align((size_t)cur, 32);
#endif

#if defined(GENERATE_PRO_PLUS)
			endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<NON_CAPTURES_PRO_MINUS_ALL>(pos, cur) : generateMoves<NON_CAPTURES_PRO_MINUS>(pos, cur);
#else
			endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<NON_CAPTURES_ALL          >(pos, cur) : generateMoves<NON_CAPTURES          >(pos, cur);
#endif
			// 注意 : ここ⇑、CAPTURE_INITで生成した指し手に歩の成りの指し手が含まれているなら、それを除外しなければならない。

			// 駒を捕獲しない指し手に対してオーダリングのためのスコアをつける
			score<QUIETS>();

			// 指し手を部分的にソートする。depthに線形に依存する閾値で。
			// (depthが低いときに真面目に全要素ソートするのは無駄だから)

			// メモ書き)
			//
			// 将棋では平均合法手は100手程度。(以前は80手程度だったが、AI同士の対局では
			// 終局までの平均手数が伸びたので相対的に終盤が多くなり、終盤は手駒を持っていることが多いから、
			// そのため平均合法手が増えた。)
			// また、合法手の最大は、593手。
			// 
			// それに対して、チェスの平均合法手は40手、合法手の最大は、218手と言われている。
			//
			// insertion sortの計算量は、O(n^2) で、将棋ではわりと悩ましいところ。
			// sortする個数が64以上などはquick sortに切り替えるなどした方がいい可能性もある。

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
			// SuperSortを有効にするとinsertion_sortと結果が異なるのでbenchコマンドの探索node数が変わって困ることがあるので注意。
			partial_super_sort    (cur, endMoves, - PARAM_MOVEPICKER_SORT_TH1 /*1960*/ - PARAM_MOVEPICKER_SORT_ALPHA1 /*3130*/ * depth);
#else
			partial_insertion_sort(cur, endMoves, - PARAM_MOVEPICKER_SORT_TH2 /*1960*/ - PARAM_MOVEPICKER_SORT_ALPHA2 /*3130*/ * depth);
#endif

			// →　sort時間がもったいないのでdepthが浅いときはscoreの悪い指し手を無視するようにしているだけで
			//   sort時間がゼロでできるなら全部した方が良いがどうせ早い段階で枝刈りされるのでほとんど効果がない。
			//
			//   ここでsortする数、将棋ではチェスと同じ程度の個数になるように、減らすようにチューニングした方が良い。
			//   つまり、PARAM_MOVEPICKER_SORT_THとPARAM_MOVEPICKER_SORT_ALPHAの絶対値を小さめにする。
			//   super sortを用いる時は、PARAM_MOVEPICKER_SORT_ALPHAを少し大きめにした方がいいかも知れない。
			//   (ただし、それでもStockfishの値は大きすぎるっぽい)
		}

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を返す。
	// (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
	// ※　これ、指し手の数が多い場合、AVXを使って一気に削除しておいたほうが良いのでは..
	case QUIET:
		if (   !skipQuiets
			&& select<Next>([&]() { return  *cur != refutations[0].move
										 && *cur != refutations[1].move
										 && *cur != refutations[2].move;
			}))
			return *(cur - 1);

		// bad capturesの指し手を返すためにポインタを準備する。
		// bad capturesの先頭を指すようにする。これは指し手生成バッファの先頭からの領域を再利用している。
		cur = moves;
		endMoves = endBadCaptures;

		++stage;
		[[fallthrough]];

	// see()が負の指し手を返す。
	case BAD_CAPTURE:
		return select<Next>([]() { return true; });

	// 王手回避手の生成
	case EVASION_INIT:
		cur = moves;

		endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<EVASIONS_ALL>(pos, cur) : generateMoves<EVASIONS>(pos, cur);

		// 王手を回避する指し手に対してオーダリングのためのスコアをつける
		score<EVASIONS>();

		++stage;
		[[fallthrough]];

	// 王手回避の指し手を返す
	case EVASION:
		// そんなに数は多くないはずだから、オーダリングがベストのスコアのものを選択する
		return select<Best>([](){ return true; });

	// PROBCUTの指し手を返す
	case PROBCUT:
		return select<Next>([&]() { return pos.see_ge(*cur, threshold); });
		// threadshold以上のSEE値で、ベストのものを一つずつ返す

	// 静止探索用の指し手を返す処理
	case QCAPTURE:
		// depthがDEPTH_QS_RECAPTURES(-5)以下(深い)なら、recaptureの升に移動する指し手(直前で取られた駒を取り返す指し手)のみを生成。
		if (select<Next>([&]() { return    depth > DEPTH_QS_RECAPTURES
										|| to_sq(*cur) == recaptureSquare; }))
			return *(cur - 1);

		// 指し手がなくて、depthが0(DEPTH_QS_CHECKS)より深いなら、これで終了
		// depthが0のときは特別に、王手になる指し手も生成する。
		if (depth != DEPTH_QS_CHECKS)
			return MOVE_NONE;

		++stage;
		[[fallthrough]];

	// 王手となる指し手の生成
	case QCHECK_INIT:
		// この前のフェーズでCAPTURES_PRO_PLUSで生成していたので、駒を取らない王手の指し手生成(QUIET_CHECKS) - 歩の成る指し手の除外 が必要。
		// (歩の成る指し手は王手であろうとすでに生成して試したあとである)
		// QUIET_CHECKS_PRO_MINUSがあれば良いのだが、実装が難しいので、QUIET_CHECKSで生成して、このあとQCHECK_で歩の成る指し手を除外する。
		cur = moves;

		//endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<QUIET_CHECKS_ALL>(pos, cur) : generateMoves<QUIET_CHECKS>(pos, cur);
		// → qsearch()なので歩の成らずは生成しなくてもいいや..
		endMoves = generateMoves<QUIET_CHECKS>(pos, cur);

		++stage;
		[[fallthrough]];

	// 王手になる指し手を一手ずつ返すフェーズ
	case QCHECK:
		return select<Next>([](){ return true; });
		//return select<Next>([&]() { return !pos.pawn_promotion(*cur); });

		// 王手する指し手は、即詰みを狙うものであり、駒捨てとかがあるからオーダリングが難しく、効果に乏しいのでオーダリングしない。

	default:
		UNREACHABLE;
		return MOVE_NONE;
	}

	ASSERT(false);
	return MOVE_NONE; // Silence warning
}

#endif // defined(USE_MOVE_PICKER)
