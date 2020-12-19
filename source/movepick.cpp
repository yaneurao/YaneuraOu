#include "movepick.h"
#if defined(USE_MOVE_PICKER)

#include "thread.h"

// パラメーターの自動調整フレームワークからパラメーターの値を読み込む
#include "engine/yaneuraou-engine/yaneuraou-param-common.h"

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)
// partial_insertion_sort()のSuperSortを用いた実装
extern void partial_super_sort(ExtMove* start, ExtMove* end, int limit);

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
	QUIET_,						// CAPTURES_PRO_PLUSで生成しなかった指し手を生成して、一つずつ返す。SEE値の悪い手は後回し。
	BAD_CAPTURE,				// 捕獲する悪い指し手(SEE < 0 の指し手だが、将棋においてそこまで悪い手とは限らないが…)

	// 将棋ではBAD_CAPTUREをQUIET_の前にやったほうが良いという従来説は以下の実験データにより覆った。
	//  r300, 2585 - 62 - 2993(46.34% R - 25.46)[2016/08/19]
	// b1000, 1051 - 43 - 1256(45.56% R - 30.95)[2016/08/19]

	// -----------------------------------------------------
	//   王手がかっているときの静止探索/通常探索の指し手生成
	// -----------------------------------------------------

	EVASION_TT,					// 置換表の指し手を返すフェーズ
	EVASION_INIT,				// (EVASIONSの指し手を生成)
	EVASION_,					// 回避する指し手(EVASIONS)を生成した指し手を一つずつ返す

	// -----------------------------------------------------
	//   通常探索のProbCutの処理のなかから呼び出される用
	// -----------------------------------------------------

	PROBCUT_TT,					// 置換表の指し手を返すフェーズ
	PROBCUT_INIT,				// (PROBCUTの指し手を生成)
	PROBCUT_,					// 直前の指し手での駒の価値を上回る駒取りの指し手のみを生成するフェーズ

	// -----------------------------------------------------
	//   静止探索時用の指し手生成
	// -----------------------------------------------------

	QSEARCH_TT,					// 置換表の指し手を返すフェーズ
	QCAPTURE_INIT,				// (QCAPTUREの指し手生成)
	QCAPTURE_,					// 捕獲する指し手 + 歩を成る指し手を一手ずつ返す
	QCHECK_INIT,				// 王手となる指し手を生成
	QCHECK_						// 王手となる指し手(- 歩を成る指し手)を返すフェーズ
};

// -----------------------
//   partial insertion sort
// -----------------------

// partial_insertion_sort()は指し手を与えられたlimitまで降順でソートする。
// limitよりも小さい値の指し手の順序については、不定。
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


// 合法手か判定する
// ※　やねうら王独自追加。
// 歩の不成を生成するモードが必要なのでこの部分をwrapしておく必要があった。
bool pseudo_legal(const Position& pos, Move ttm)
{
	// 歩の不成を生成するかは、Options["GenerateAllLegalMoves"]に依存する。
	return Search::Limits.generate_all_legal_moves ? pos.pseudo_legal_s<true>(ttm) : pos.pseudo_legal_s<false>(ttm);
}


} // end of namespace

// 指し手オーダリング器

// 通常探索から呼び出されるとき用。
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh, const LowPlyHistory* lp,
	const CapturePieceToHistory* cph , const PieceToHistory** ch, Move cm, const Move* killers,int pl)
	: pos(p), mainHistory(mh), lowPlyHistory(lp),captureHistory(cph) , continuationHistory(ch),
	ttMove(ttm), refutations{ { killers[0], 0 },{ killers[1], 0 },{ cm, 0 } }, depth(d), ply(pl)
{
	// 通常探索から呼び出されているので残り深さはゼロより大きい。
	ASSERT_LV3(d > 0);

	// 次の指し手生成の段階
	// 王手がかかっているなら回避手、かかっていないなら通常探索用の指し手生成
	stage = (pos.in_check() ? EVASION_TT : MAIN_TT) +
		!(ttm && pseudo_legal(pos,ttm));

	// 置換表の指し手があるならそれを最初に試す。ただしpseudo_legalでなければならない。
	// 置換表の指し手がないなら、次のstageから開始する。
}

// 静止探索から呼び出される時用。
// rs : recapture square
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
						const CapturePieceToHistory* cph, const PieceToHistory** ch, Square rs)
	: pos(p), mainHistory(mh), captureHistory(cph) , continuationHistory(ch) , ttMove(ttm), recaptureSquare(rs), depth(d) {

	// 静止探索から呼び出されているので残り深さはゼロ以下。
	ASSERT_LV3(d <= 0);

	// 王手がかかっているなら王手回避のフェーズへ。さもなくばQSEARCHのフェーズへ。
	// 歩の不成、香の2段目への不成、大駒の不成を除外

	stage = (pos.in_check() ? EVASION_TT : QSEARCH_TT) +
		!(ttm
			&& (pos.in_check() || depth > DEPTH_QS_RECAPTURES || to_sq(ttm) == recaptureSquare)
			&& pseudo_legal(pos,ttm));

}

// 通常探索時にProbCutの処理から呼び出されるの専用
// th = 枝刈りのしきい値
MovePicker::MovePicker(const Position& p, Move ttm, Value th , const CapturePieceToHistory* cph)
			: pos(p), captureHistory(cph) , ttMove(ttm),threshold(th) {

	ASSERT_LV3(!pos.in_check());

	// ProbCutにおいて、SEEが与えられたthresholdの値以上の指し手のみ生成する。
	// (置換表の指しても、この条件を満たさなければならない)
	// 置換表の指し手がないなら、次のstageから開始する。
	stage = PROBCUT_TT + !(ttm && pos.capture(ttm)
		&& pseudo_legal(pos,ttm)
		&& pos.see_ge(ttm, threshold));

}

// QUIETS、EVASIONS、CAPTURESの指し手のオーダリングのためのスコアリング。似た処理なので一本化。
template<MOVE_GEN_TYPE Type>
void MovePicker::score()
{
	static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

	for (auto& m : *this)
	{
		if (Type == CAPTURES)
		{
			// Position::see()を用いると遅い。単に取る駒の価値順に調べたほうがパフォーマンス的にもいい。
			// 歩が成る指し手もあるのでこれはある程度優先されないといけない。
			// CAPTURE系である以上、打つ指し手は除外されている。
			// CAPTURES_PRO_PLUSで生成しているので歩の成る指し手が混じる。
			
			// MVV-LVAだが、将棋ではLVAあんまり関係なさげだが(複数の駒である1つの駒が取れるケースがチェスより少ない)、
			// Stockfish 9に倣いMVV + captureHistoryで処理する。

			// 歩の成りは別途考慮してもいいような気はするのだが…。

			m.value = int(Eval::CapturePieceValue[pos.piece_on(to_sq(m))])*6
					+ (*captureHistory)[to_sq(m)][pos.moved_piece_after(m)][type_of(pos.piece_on(to_sq(m)))];
		}
		else if (Type == QUIETS)
		{
			// 駒を取らない指し手をオーダリングする。
			// ここ、歩以外の成りも含まれているのだが…。
			// →　指し手オーダリングは、quietな指し手の間での優劣を付けたいわけで、
			//    駒を成るような指し手はどうせevaluate()で大きな値がつくからそっちを先に探索することになる。

			Piece movedPiece = pos.moved_piece_after(m);
			Square movedSq = to_sq(m);

#if 0
			m.value = (*mainHistory)[from_to(m)][pos.side_to_move()]
					+ 2 * (*continuationHistory[0])[movedSq][movedPiece]
					+ 2 * (*continuationHistory[1])[movedSq][movedPiece]
					+ 2 * (*continuationHistory[3])[movedSq][movedPiece]
					+     (*continuationHistory[5])[movedSq][movedPiece]
					+ (ply < MAX_LPH ? std::min(4, depth / 3) * (*lowPlyHistory)[ply][from_to(m)] : 0);
#endif

			// パラメーターの調整フレームワークを利用するために、値を16倍して最適値を探す。
			m.value = 16 * (*mainHistory)[from_to(m)][pos.side_to_move()]
				+ MOVE_PICKER_Q_PARAM1 * (*continuationHistory[0])[movedSq][movedPiece]
				+ MOVE_PICKER_Q_PARAM2 * (*continuationHistory[1])[movedSq][movedPiece]
				+ MOVE_PICKER_Q_PARAM3 * (*continuationHistory[3])[movedSq][movedPiece]
				+ MOVE_PICKER_Q_PARAM4 * (*continuationHistory[5])[movedSq][movedPiece]
				+ MOVE_PICKER_Q_PARAM5 * (ply < MAX_LPH ? std::min(4, depth / 3) * (*lowPlyHistory)[ply][from_to(m)] : 0);

		}
		else // Type == EVASIONS
		{
			// 王手回避の指し手をスコアリングする。

			// 駒を取る指し手ならseeがプラスだったということなのでプラスの符号になるようにStats::Maxを足す。
			// あとは取る駒の価値を足して、動かす駒の番号を引いておく(小さな価値の駒で王手を回避したほうが
			// 価値が高いので(例えば合駒に安い駒を使う的な…)

			//  ・成るなら、その成りの価値を加算したほうが見積もりとしては正しい？
			// 　それは取り返されないことが前提にあるから、そうでもない。
			//		T1,r300,2491 - 78 - 2421(50.71% R4.95)
			//		T1,b1000,2483 - 103 - 2404(50.81% R5.62)
			//      T1,b3000,2459 - 148 - 2383(50.78% R5.45)
			//   →　やはり、改造前のほうが良い。[2016/10/06]

			// ・moved_piece_before()とmoved_piece_after()との比較
			// 　厳密なLVAではなくなるが、afterのほうが良さげ。
			// 　例えば、歩を成って取るのと、桂で取るのとでは、安い駒は歩だが、桂で行ったほうが、
			// 　歩はあとで成れるとすれば潜在的な価値はそちらのほうが高いから、そちらを残しておくという理屈はあるのか。
			//		T1, b1000, 2402 - 138 - 2460(49.4% R - 4.14) win black : white = 51.04% : 48.96%
			//		T1,b3000,1241 - 108 - 1231(50.2% R1.41) win black : white = 50.53% : 49.47%
			//		T1,b5000,1095 - 118 - 1047(51.12% R7.79) win black : white = 52.33% : 47.67%
			//  →　moved_piece_before()のほうで問題なさげ。[2017/5/20]

			if (pos.capture(m))
				// 捕獲する指し手に関しては簡易SEE + MVV/LVA
				m.value = (Value)Eval::CapturePieceValue[pos.piece_on(to_sq(m))]
				        - (Value)(LVA(type_of(pos.moved_piece_before(m))));
			else
				// 捕獲しない指し手に関してはhistoryの値の順番
				m.value = (*mainHistory)[from_to(m)][pos.side_to_move()]
						+ (*continuationHistory[0])[to_sq(m)][pos.moved_piece_after(m)]
						- (1 << 28);

		}
	}
}

/// MovePicker::select()は、Pred(predicate function:述語関数)を満たす次の指し手を返す。
/// 置換表の指し手は決して返さない。
/// ※　この関数の返し値は同時にthis->moveにも格納されるので活用すると良い。filterのなかでも
///   この変数にアクセスできるので、指し手によってfilterするかどうかを選べる。
template<MovePicker::PickType T, typename Pred>
Move MovePicker::select(Pred filter) {

	while (cur < endMoves)
	{
		// TがBestならBestを探してcurが指す要素と入れ替える。
		// それがttMoveであるなら、もう一周する。
		if (T == Best)
			std::swap(*cur, *std::max_element(cur, endMoves));

		// filter()のなかで*curにアクセスして判定するのでfilter()は引数を取らない。
		if (*cur != ttMove && filter())
			return *cur++;
		cur++;
	}
	return MOVE_NONE;
}


// 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
// 指し手が尽きればMOVE_NONEが返る。
// 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
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

		endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<CAPTURES_PRO_PLUS_ALL>(pos, cur) : generateMoves<CAPTURES_PRO_PLUS>(pos, cur);

		// 駒を捕獲する指し手に対してオーダリングのためのスコアをつける
		score<CAPTURES>();

		++stage;
		goto top;

	// 置換表の指し手を返したあとのフェーズ
	// (killer moveの前のフェーズなのでkiller除去は不要)
	// SSEの値が悪いものはbad captureのほうに回す。
	case GOOD_CAPTURE:
		if (select<Best>([&]() {
				// moveは駒打ちではないからsee()の内部での駒打ちは判定不要だが…。
				return pos.see_ge(*cur, Value(-69 * cur->value / 1024)) ?
						// 損をする捕獲する指し手はあとのほうで試行されるようにendBadCapturesに移動させる
						true : (*endBadCaptures++ = *cur, false); }))
			return *(cur -1);

			// refutations配列に対して繰り返すためにポインターを準備する。
			cur = std::begin(refutations);
			endMoves = std::end(refutations);

			// countermoveがkillerと同じならばそれをskipする。
			// ※　killer[]は32bit化されている(上位に移動後の駒が格納されている)と仮定している。
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
										&& !pos.capture_or_pawn_promotion(*cur)
										&&  pseudo_legal(pos,*cur); }))
			return *(cur - 1);

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を生成してオーダリング
	case QUIET_INIT:

		if (!skipQuiets)
		{
			cur = endBadCaptures;

#if defined(USE_AVX2)
			// curを32の倍数アドレスになるように少し進めてしまう。
			// これにより、curがalignas(32)されているような効果がある。
			// このあとSuperSortを使うときにこれが前提条件として必要。
			cur = (ExtMove*)Math::align((size_t)cur, 32);
#endif

			endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<NON_CAPTURES_PRO_MINUS_ALL>(pos, cur) : generateMoves<NON_CAPTURES_PRO_MINUS>(pos, cur);

			// 駒を捕獲しない指し手に対してオーダリングのためのスコアをつける
			score<QUIETS>();

			// 指し手を部分的にソートする。depthに線形に依存する閾値で。
			// (depthが低いときに真面目に全要素ソートするのは無駄だから)

#if defined(USE_SUPER_SORT) && defined(USE_AVX2)

			// AVX2なので自動的にlittle endianと仮定できるのでint64_tとみなしてソートして良い。
			// このとき、sortの高速化として、SuperSortが使える。
			// cur == movesの先頭だし、MAX_MOVESの分だけbufferは確保されているので32の倍数になるように
			// 後方のpaddingをしてAVX2を使ったsortをして良い。このとき、他にbuffer不要。
			
			partial_super_sort(cur, endMoves , -3000 * depth);

			// ↑を有効にするとinsertion_sortと結果が異なるのでbenchコマンドの探索node数が変わって困ることがあるので注意。

#else

			// TODO : このへん係数調整したほうが良いのでは…。
			// →　sort時間がもったいないのでdepthが浅いときはscoreの悪い指し手を無視するようにしているだけで
			//   sortできるなら全部したほうが良い。上のSuperSortを使う実装の場合、全部sortしている。
			partial_insertion_sort(cur, endMoves, -3000 * depth);
#endif
		}

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を返す。
	// (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
	// ※　これ、指し手の数が多い場合、AVXを使って一気に削除しておいたほうが良いのでは..
	case QUIET_:
		if (   !skipQuiets
			&& select<Next>([&]() {return  *cur != refutations[0].move
										&& *cur != refutations[1].move
										&& *cur != refutations[2].move; }))

			return *(cur - 1);

		// bad capturesの指し手に対して繰り返すためにポインタを準備する。
		// bad capturesの先頭を指すようにする。これは指し手生成バッファの先頭付近を再利用している。
		cur = moves;
		endMoves = endBadCaptures;

		++stage;
		/* fallthrough */

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
	case EVASION_:
		// そんなに数は多くないはずだから、オーダリングがベストのスコアのものを選択する
		return select<Best>([](){ return true; });

	// PROBCUTの指し手を返す
	case PROBCUT_:
		return select<Best>([&]() { return pos.see_ge(*cur, threshold); });
		// threadshold以上のSEE値で、ベストのものを一つ

	// 静止探索用の指し手を返す処理
	case QCAPTURE_:
		// depthがDEPTH_QS_RECAPTURES(-5)より深いなら、recaptureの升に移動する指し手のみを生成。
		if (select<Best>([&]() { return    depth > DEPTH_QS_RECAPTURES
										|| to_sq(*cur) == recaptureSquare; }))
			return *(cur - 1);

		// 指し手がなくて、depthが0(DEPTH_QS_CHECKS)より深いなら、これで終了
		// depthが0のときは特別に、王手になる指し手も試す。ただし、他にcaptureの指し手がないなら、王手も試さない。
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

		endMoves = Search::Limits.generate_all_legal_moves ? generateMoves<QUIET_CHECKS_ALL>(pos, cur) : generateMoves<QUIET_CHECKS>(pos, cur);

		++stage;
		[[fallthrough]];

	// 王手になる指し手を一手ずつ返すフェーズ
	case QCHECK_:
		// return select<Next>([](){ return true; });
		return select<Next>([&]() { return !pos.pawn_promotion(*cur); });

	default:
		UNREACHABLE;
		return MOVE_NONE;
	}

	ASSERT(false);
	return MOVE_NONE;
}

#endif // defined(USE_MOVE_PICKER)
