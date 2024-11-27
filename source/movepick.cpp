#include "movepick.h"
#if defined(USE_MOVE_PICKER)

#include <algorithm>
//#include <cassert>
#include <iterator>
#include <utility>

#include "bitboard.h"
#include "position.h"
#include "search.h" // Search::Limits.generate_all_legal_movesによって生成される指し手を変えたいので…。

// パラメーターの自動調整フレームワークからパラメーターの値を読み込む
#include "engine/yaneuraou-engine/yaneuraou-param-common.h"

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
// partial_insertion_sort()のSuperSortを用いた実装
void partial_super_sort(ExtMove* start, ExtMove* end, int limit);
void super_sort(ExtMove* start, ExtMove* end);

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
			*p = *++sortedEnd;
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

MovePicker::MovePicker(
	const Position&              p,
	Move                         ttm,
	Depth                        d,
	const ButterflyHistory*      mh,
	const LowPlyHistory*         lph,
	const CapturePieceToHistory* cph,
	const PieceToHistory**       ch,
#if defined(ENABLE_PAWN_HISTORY)
	const PawnHistory*           ph,
#endif
	int pl) :
	pos(p),
	mainHistory(mh),
	lowPlyHistory(lph),
	captureHistory(cph),
	continuationHistory(ch),
#if defined(ENABLE_PAWN_HISTORY)
	pawnHistory(ph),
#endif
	ttMove(ttm),
	depth(d),
	ply(pl)
{
	// 次の指し手生成の段階
	// 王手がかかっているなら王手回避のフェーズへ。さもなくばQSEARCHのフェーズへ。
#if 1
	if (pos.in_check())
		// 王手がかかっているなら回避手
		stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

	else
		// 王手がかかっていないなら通常探索用/静止探索の指し手生成
		// ⇨ 通常探索から呼び出されたのか、静止探索から呼び出されたのかについてはdepth > 0 によって判定できる。
		stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
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

MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
	pos(p),
	captureHistory(cph),
	ttMove(ttm),
	threshold(Value(th))
{

	// ProbCutから呼び出されているので王手はかかっていないはず。
	ASSERT_LV3(!pos.in_check());

	// ProbCutにおいて、SEEが与えられたthresholdの値以上の指し手のみ生成する。
	// (置換表の指し手も、この条件を満たさなければならない)
	// 置換表の指し手がないなら、次のstageから開始する。

	stage = PROBCUT_TT
		+ !(ttm 
			// && pos.capture_stage(ttm)
#if !defined(MOVE_PICKER_GENERATE_CAPTURE)
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

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.

// 各手に数値を割り当ててリストをソートします。
// キャプチャは最も価値のある駒（MVV）に基づいて順序付けられ、
// 良好な履歴を持つキャプチャが優先されます。
// 静かな手は履歴テーブルを使用して順序付けられます。

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

			m.value = 7 * int(Eval::CapturePieceValuePlusPromote(pos, m))
					   + (*captureHistory)(pos.moved_piece_after(m), m.to_sq(), type_of(pos.piece_on(m.to_sq())));
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
			Square    to = m.to_sq();

			m.value  =      (*mainHistory)(pos.side_to_move(), m.from_to());
#if defined(ENABLE_PAWN_HISTORY)
			m.value +=  2 * (*pawnHistory)(pawn_structure(pos), pc, to);
#endif
			m.value +=  2 * (*continuationHistory[0])(pc,to);
			m.value +=      (*continuationHistory[1])(pc,to);
			m.value +=      (*continuationHistory[2])(pc,to) / 3;
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

			// lowPlyHistoryも加算
			if (ply < LOW_PLY_HISTORY_SIZE)
				m.value += 8 * (*lowPlyHistory)(ply , m.from_to()) / (1 + 2 * ply);

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
				m.value =     (*mainHistory)(pos.side_to_move(), m.from_to())
						  +   (*continuationHistory[0])(pos.moved_piece_after(m), m.to_sq())
#if defined(ENABLE_PAWN_HISTORY)
						  +   (*pawnHistory)(pawn_structure(pos), pos.moved_piece_after(m), m.to_sq())
#endif
				;

		}
	}
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.

// 条件を満たす次の手を返します。
// この関数は、トランスポジションテーブル（TT）の手は既に出力されているため、決して返しません。

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

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
	auto quiet_threshold = [](Depth d) { return -PARAM_MOVEPICKER_SORT_ALPHA1 * d; };
#else
	auto quiet_threshold = [](Depth d) { return -PARAM_MOVEPICKER_SORT_ALPHA2 * d; };
#endif

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

#if !defined(MOVE_PICKER_GENERATE_CAPTURE)
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
                return pos.see_ge(*cur, Value(-cur->value / 18)) ? true
																 : (*endBadCaptures++ = *cur, false);
				// 損をする捕獲する指し手はあとのほうで試行されるようにendBadCapturesに移動させる
			}))
			return *(cur -1);

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


#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
			// curを32の倍数アドレスになるように少し進めてしまう。
			// これにより、curがalignas(32)されているような効果がある。
			// このあとSuperSortを使うときにこれが前提条件として必要。
			cur = (ExtMove*)Math::align((size_t)cur, 32);
#endif

#if !defined(MOVE_PICKER_GENERATE_CAPTURE)
			endMoves = beginBadQuiets = endBadQuiets = Search::Limits.generate_all_legal_moves ? generateMoves<NON_CAPTURES_PRO_MINUS_ALL>(pos, cur) : generateMoves<NON_CAPTURES_PRO_MINUS>(pos, cur);
#else
			endMoves = beginBadQuiets = endBadQuiets = Search::Limits.generate_all_legal_moves ? generateMoves<NON_CAPTURES_ALL          >(pos, cur) : generateMoves<NON_CAPTURES          >(pos, cur);
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
			partial_super_sort(cur, endMoves, quiet_threshold(depth));
#else
			partial_insertion_sort(cur, endMoves, quiet_threshold(depth));
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
	case GOOD_QUIET:
		if (!skipQuiets
			&& select<Next>([&]() { return true; }))
		{
			if ((cur - 1)->value > -7998 || (cur - 1)->value <= quiet_threshold(depth))
				return *(cur - 1);

			// Remaining quiets are bad
			// 残っているquietの手は悪手です

			beginBadQuiets = cur - 1;
		}

		// bad capturesの指し手を返すためにポインタを準備する。
		// bad capturesの先頭を指すようにする。これは指し手生成バッファの先頭からの領域を再利用している。
		cur = moves;
		endMoves = endBadCaptures;

		++stage;
		[[fallthrough]];

		// see()が負の指し手を返す。
	case BAD_CAPTURE:
		if (select<Next>([]() { return true; }))
			return *(cur - 1);

		// Prepare the pointers to loop over the bad quiets
		// 悪いquietの手をループするためのポインタを準備します

		cur = beginBadQuiets;
		endMoves = endBadQuiets;

		++stage;
		[[fallthrough]];

	case BAD_QUIET:
		if (!skipQuiets)
			return select<Next>([]() { return true; });

		return Move::none();

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
		return select<Best>([]() { return true; });

		// PROBCUTの指し手を返す
	case PROBCUT:
		return select<Next>([&]() { return pos.see_ge(*cur, threshold); });
		// threadshold以上のSEE値で、ベストのものを一つずつ返す

	// 静止探索用の指し手を返す処理
	case QCAPTURE:
		return select<Next>([]() { return true; });

	default:
		UNREACHABLE;
		return Move::none();
	}

	ASSERT(false);
	return Move::none(); // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

#endif // defined(USE_MOVE_PICKER)
