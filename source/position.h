#ifndef _POSITION_H_
#define _POSITION_H_

#include "shogi.h"
#include "bitboard.h"
#include "evaluate.h"
#include "extra/key128.h"
#include "extra/long_effect.h"
struct Thread;

// --------------------
//     局面の定数
// --------------------

// 平手の開始局面
extern std::string SFEN_HIRATE;

// --------------------
//     局面の情報
// --------------------

#ifdef USE_EVAL_DIFF
// 評価値の差分計算の管理用
// 前の局面から移動した駒番号を管理するための構造体
struct DirtyPiece
{
  // dirtyになった個数。null moveだと0ということもありうる。
  int dirty_num;

  // dirtyになった駒番号
  PieceNo pieceNo[2];

  // その駒番号の駒が何から何に変わったのか
  Eval::ExtBonaPiece piecePrevious[2];
  Eval::ExtBonaPiece pieceNow[2];
};
#endif

// StateInfoは、undo_move()で局面を戻すときに情報を元の状態に戻すのが面倒なものを詰め込んでおくための構造体。
// do_move()のときは、ブロックコピーで済むのでそこそこ高速。
struct StateInfo {

  // ---- ここから下のやつは do_move()のときにコピーされる

  // 遡り可能な手数(previousポインタを用いて局面を遡るときに用いる)
  int pliesFromNull;

  // この手番側の連続王手は何手前からやっているのか(連続王手の千日手の検出のときに必要)
  int continuousCheck[COLOR_NB];

  // ---- ここから下のやつは do_move()のときにコピーされない
  // ※　ただし、do_null_move()のときは丸ごとコピーされる。

  // --- 以下のメンバーは、Position::do_move()で更新される。

  // 現局面で手番側に対して王手をしている駒のbitboard
  Bitboard checkersBB;

  // 動かすと手番側の王に対して空き王手になるかも知れない駒の候補
  // チェスの場合、駒がほとんどが大駒なのでこれらを動かすと必ず開き王手となる。
  // 将棋の場合、そうとも限らないので移動方向について考えなければならない。
  // color = 手番側 なら pinされている駒(動かすと開き王手になる)
  // color = 相手側 なら 両王手の候補となる駒。

  // 自玉に対して(敵駒によって)pinされている駒
  Bitboard blockersForKing[COLOR_NB];

  // 自玉に対してpinしている(可能性のある)敵の大駒。
  // 自玉に対して上下左右方向にある敵の飛車、斜め十字方向にある敵の角、玉の前方向にある敵の香、…
  Bitboard pinnersForKing[COLOR_NB];

  // 自駒の駒種Xによって敵玉が王手となる升のbitboard
  Bitboard checkSquares[PIECE_WHITE];


  // この局面のハッシュキー
  // ※　次の局面にdo_move()で進むときに最終的な値が設定される
  // board_key()は盤面のhash。hand_key()は手駒のhash。それぞれ加算したのがkey() 盤面のhash。
  // board_key()のほうは、手番も込み。
  // exclusion_key()は、singular extensionのために現在のkey()に一定の値を足したものを返す。
  Key key()                     const { return long_key(); }
  Key board_key()               const { return board_long_key(); }
  Key hand_key()                const { return hand_long_key(); }

  // HASH_KEY_BITSが128のときはKey128が返るhash key,256のときはKey256
  HASH_KEY long_key()           const { return board_key_ + hand_key_; }
  HASH_KEY board_long_key()     const { return board_key_; }
  HASH_KEY hand_long_key()      const { return hand_key_; }
  
  // この局面における手番側の持ち駒。優等局面の判定のために必要。
  Hand hand;

  // この局面で捕獲された駒
  // ※　次の局面にdo_move()で進むときにこの値が設定される
  // 先後の区別はなし。馬とか龍など成り駒である可能性はある。
  Piece capturedPiece;

  friend struct Position;

  // --- evaluate

#ifndef EVAL_NO_USE
  // この局面での評価関数の駒割
  Value materialValue;
#endif

#ifdef EVAL_KPP
  // 評価値。(次の局面で評価値を差分計算するときに用いる)
  // まだ計算されていなければsumKPPの値は、INT_MAX
  int sumKKP;
  int sumBKPP;
  int sumWKPP;
#endif

#if defined(EVAL_KPPT) || defined(EVAL_KPPT_FAST)
  // 評価値。(次の局面で評価値を差分計算するときに用いる)
  // まだ計算されていなければsum.p[2][0]の値はINT_MAX
  Eval::EvalSum sum;
#endif

#ifdef USE_EVAL_DIFF
  // 評価値の差分計算の管理用
  DirtyPiece dirtyPiece;
#endif

#ifdef  KEEP_LAST_MOVE
  // 直前の指し手。デバッグ時などにおいてその局面までの手順を表示出来ると便利なことがあるのでそのための機能
  Move lastMove;

  // lastMoveで移動させた駒(先後の区別なし)
  Piece lastMovedPieceType;
#endif

  // HASH_KEY_BITSで128を指定した場合はBitboardにHashKeyが入っている。
  HASH_KEY board_key_;
  HASH_KEY hand_key_;

  // 一つ前の局面に遡るためのポインタ。
  // この値としてnullptrが設定されているケースは、
  // 1) root node
  // 2) 直前がnull move
  // のみである。
  // 評価関数を差分計算するときに、
  // 1)は、compute_eval()を呼び出して差分計算しないからprevious==nullで問題ない。
  // 2)は、このnodeのEvalSum sum(これはdo_move_null()でコピーされている)から
  //   計算出来るから問題ない。
  StateInfo* previous;

};

// --------------------
//       盤面
// --------------------

// packされたsfen
struct PackedSfen { u8 data[32]; };

// 盤面
struct Position
{
	// --- ctor

	// コンストラクタではおまけとして平手の開始局面にする。
	Position() {
		clear();
#ifndef USE_SHARED_MEMORY_IN_EVAL
		// Positionのコンストラクタで平手に初期化すると、compute_eval()が呼び出され、このときに
		// 評価関数テーブルを参照するが、isready()が呼び出されていないのでこの初期化が出来ない。
		// ゆえに、この処理は本来ならやめたほうが良い。
		// (特にisready()が呼び出されるまでcompute_eval()が呼び出せない状況においては。)
		set_hirate();
#endif
	}

	// コピー。startStateもコピーして、外部のデータに依存しないように(detach)する。
	// 積極的に使うべきではない。探索開始時にslaveに局面をコピーするときに仕方なく使っているだけ。
	Position& operator=(const Position& pos);

	// 初期化
	void clear();

	// Positionで用いるZobristテーブルの初期化
	static void init();

	// sfen文字列で盤面を設定する
	// ※　内部的にinit()は呼び出される。
	void set(std::string sfen);

	// 局面のsfen文字列を取得する
	// ※ USIプロトコルにおいては不要な機能ではあるが、デバッグのために局面を標準出力に出力して
	// 　その局面から開始させたりしたいときに、sfenで現在の局面を出力出来ないと困るので用意してある。
	const std::string sfen() const;

	// 平手の初期盤面を設定する。
	void set_hirate() { set(SFEN_HIRATE); }

	// --- properties

	// 現局面の手番を返す。
	Color side_to_move() const { return sideToMove; }

	// (将棋の)開始局面からの手数を返す。
	// 平手の開始局面なら1が返る。(0ではない)
	int game_ply() const { return gamePly; }

	// この局面クラスを用いて探索しているスレッドを返す。 
	Thread* this_thread() const { return thisThread; }
	// この局面クラスを用いて探索しているスレッドを設定する。(threads.cppのなかで設定してある。)
	void set_this_thread(Thread*th) { thisThread = th; }

	// 盤面上の駒を返す
	Piece piece_on(Square sq) const { return board[sq]; }

	// c側の手駒を返す
	Hand hand_of(Color c) const { return hand[c]; }

	// c側の玉の位置を返す
	FORCE_INLINE Square king_square(Color c) const { return kingSquare[c]; }

	// 保持しているデータに矛盾がないかテストする。
	bool pos_is_ok() const;

	// 探索したノード数(≒do_move()が呼び出された回数)を取得する
	uint64_t nodes_searched() const { return nodes; }

	// 現局面に対して
	// この指し手によって移動させる駒を返す。(移動前の駒)
	// 後手の駒打ちは後手の駒が返る。
	Piece moved_piece_before(Move m) const
	{
		return is_drop(m)
			? (move_dropped_piece(m) + (sideToMove == WHITE ? PIECE_WHITE : NO_PIECE))
			: piece_on(move_from(m));
	}

	// moved_pieceの拡張版。駒打ちのときは、打ち駒(+32 == PIECE_DROP)を加算した駒種を返す。
	// historyなどでUSE_DROPBIT_IN_STATSを有効にするときに用いる。
	// 成りの指し手のときは成りの指し手を返す。(移動後の駒)
	// KEEP_PIECE_IN_GENERATE_MOVESのときは単にmoveの上位16bitを返す。
	Piece moved_piece_after(Move m) const {
#ifdef    KEEP_PIECE_IN_GENERATE_MOVES
		// 上位16bitにそのまま格納されているはず。
		return Piece(m >> 16);
#else
		return is_drop(m)
			? Piece(move_dropped_piece(m) + (sideToMove == WHITE ? PIECE_WHITE : NO_PIECE) + PIECE_DROP)
			: is_promote(m) ? Piece(piece_on(move_from(m)) + PIECE_PROMOTE) : piece_on(move_from(m));
#endif
	}

	// 置換表から取り出したMoveを32bit化する。
	Move move16_to_move(Move m) const {
		return Move(u16(m) +
			((is_drop(m) ? Piece(move_dropped_piece(m) + (sideToMove == WHITE ? PIECE_WHITE : NO_PIECE) + PIECE_DROP)
				: is_promote(m) ? Piece(piece_on(move_from(m)) + PIECE_PROMOTE) : piece_on(move_from(m))) << 16)
		);
	}

	// 連続王手の千日手等で引き分けかどうかを返す
	RepetitionState is_repetition(const int repPly = 16) const;

	// --- Bitboard

	// 先手か後手か、いずれかの駒がある場所が1であるBitboardが返る。
	Bitboard pieces() const { return occupied[COLOR_ALL]; }

	// c == BLACK : 先手の駒があるBitboardが返る
	// c == WHITE : 後手の駒があるBitboardが返る
	Bitboard pieces(Color c) const { ASSERT_LV3(is_ok(c)); return occupied[c]; }

	// 駒がない升が1になっているBitboardが返る
	Bitboard empties() const { return pieces() ^ ALL_BB; }

	// 駒に対するBitboardを得る
	Bitboard pieces(Color c, PieceTypeBitboard pr) const { return piece_bb[pr][c]; }

	// 駒に対応するBitboardを得る。
	Bitboard pieces(Color c, Piece pr) const { ASSERT_LV3(PAWN <= pr && pr <= KING);  return piece_bb[(PieceTypeBitboard)(pr - 1)][c]; }

	// 駒が存在する升を表すBitboard
	// 高速化したいときだけ用いる。普段は使うべきではない。
	// ※　see()の高速化のために用いている。
	Bitboard piece_bb[PIECE_TYPE_BITBOARD_NB][COLOR_NB];

	// --- 升

	// ある駒の存在する升を返す
	// 現状、Pt==KINGしか渡せない。Stockfishとの互換用。
	template<Piece Pt> Square square(Color c) const
	{
		static_assert(Pt == KING,"Pt must be a KING in Position::square().");
		return king_square(c);
	}

	// --- 王手

	// 現局面で王手している駒
	Bitboard checkers() const { return st->checkersBB; }

	// 移動させると(相手側＝非手番側)の玉に対して空き王手となる候補の(手番側)駒のbitboard。
	Bitboard discovered_check_candidates() const { return st->blockersForKing[~sideToMove] & pieces(sideToMove); }

	// ピンされているc側の駒。下手な方向に移動させるとc側の玉が素抜かれる。
	// 手番側のpinされている駒はpos.pinned_pieces(pos.side_to_move())のようにして取得できる。
	Bitboard pinned_pieces(Color c) const { return st->blockersForKing[c] & pieces(c); }

	// 現局面で駒Ptを動かしたときに王手となる升を表現するBitboard
	Bitboard check_squares(Piece pt) const { return st->checkSquares[pt]; }

	// --- 利き

	// sに利きのあるc側の駒を列挙する。
	// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
	Bitboard attackers_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }
	Bitboard attackers_to(Color c, Square sq, const Bitboard& occ) const;
	Bitboard attackers_to(Square sq, const Bitboard& occ) const;

	// 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
	Bitboard attackers_to_pawn(Color c, Square pawn_sq) const;

	// attackers_to()で駒があればtrueを返す版。(利きの情報を持っているなら、軽い実装に変更できる)
	// kingSqの地点からは玉を取り除いての利きの判定を行なう。
#ifndef LONG_EFFECT_LIBRARY
	bool effected_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }
	bool effected_to(Color c, Square sq, Square kingSq) const { return attackers_to(c, sq, pieces() ^ kingSq); }
#else 
	bool effected_to(Color c, Square sq) const { return board_effect[c].effect(sq) != 0; }
	bool effected_to(Color c, Square sq, Square kingSq) const {
		return board_effect[c].effect(sq) != 0 ||
			((long_effect.directions_of(c, kingSq) & Effect8::directions_of(kingSq, sq)) != 0); // 影の利きがある
	}
#endif

	// 升sに対して、c側の大駒に含まれる長い利きを持つ駒の利きを遮っている駒のBitboardを返す(先後の区別なし)
	// ※　Stockfishでは、sildersを渡すようになっているが、大駒のcolorを渡す実装のほうが優れているので変更。
	// [Out] pinnersとは、pinされている駒が取り除かれたときに升sに利きが発生する大駒である。これは返し値。
	// また、升sにある玉は~c側のKINGであるとする。
	Bitboard slider_blockers(Color c, Square s, Bitboard& pinners) const;

	// --- 局面を進める/戻す

	// 指し手で盤面を1手進める
	// m = 指し手。mとして非合法手を渡してはならない。
	// info = StateInfo。局面を進めるときに捕獲した駒などを保存しておくためのバッファ。
	// このバッファはこのdo_move()の呼び出し元の責任において確保されている必要がある。
	// givesCheck = mの指し手によって王手になるかどうか。
	// この呼出までにst.checkInfo.update(pos)が呼び出されている必要がある。
	void do_move(Move m, StateInfo& newSt, bool givesCheck);

	// do_move()の4パラメーター版のほうを呼び出すにはgivesCheckも渡さないといけないが、
	// mで王手になるかどうかがわからないときはこちらの関数を用いる。
	void do_move(Move m, StateInfo& newSt) { do_move(m, newSt, gives_check(m)); }

	// 指し手で盤面を1手戻す
	void undo_move(Move m);

	// null move用のdo_move()
	void do_null_move(StateInfo& st);
	// null move用のundo_move()
	void undo_null_move();

	// --- legality(指し手の合法性)のチェック

	// 生成した指し手(CAPTUREとかNON_CAPTUREとか)が、合法であるかどうかをテストする。
	//
	// 指し手生成で合法手であるか判定が漏れている項目についてチェックする。
	// 王手のかかっている局面についてはEVASION(回避手)で指し手が生成されているはずなので
	// ここでは王手のかかっていない局面における合法性のチェック。
	// 具体的には、
	//  1) 移動させたときに素抜きに合わないか
	//  2) 敵の利きのある場所への王の移動でないか
	// ※　連続王手の千日手などについては探索の問題なのでこの関数のなかでは行わない。
	// ※　それ以上のテストは行わないので、置換表から取ってきた指し手などについては、
	// pseudo_legal()を用いて、そのあとこの関数で判定すること。
	bool legal(Move m) const
	{
		if (is_drop(m))
			// 打ち歩詰めは指し手生成で除外されている。
			return true;
		else
		{
			Color us = sideToMove;
			Square from = move_from(m);

			// もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
			if (type_of(piece_on(from)) == KING)
				return !effected_to(~us, move_to(m), from);

			return   !(pinned_pieces(us) & from)
				|| aligned(from, to_sq(m), square<KING>(us));
		}
	}

	// mがpseudo_legalな指し手であるかを判定する。
	// ※　pseudo_legalとは、擬似合法手(自殺手が含まれていて良い)
	// 置換表の指し手でdo_move()して良いのかの事前判定のために使われる。
	// 指し手生成ルーチンのテストなどにも使える。(指し手生成ルーチンはpseudo_legalな指し手を返すはずなので)
	// killerのような兄弟局面の指し手がこの局面において合法かどうかにも使う。
	// ※　置換表の検査だが、pseudo_legal()で擬似合法手かどうかを判定したあとlegal()で自殺手でないことを
	// 確認しなくてはならない。このためpseudo_legal()とlegal()とで重複する自殺手チェックはしていない。
	bool pseudo_legal(const Move m) const { return pseudo_legal_s<true>(m); }

	// All == false        : 歩や大駒の不成に対してはfalseを返すpseudo_legal()
	template <bool All> bool pseudo_legal_s(const Move m) const;

	// toの地点に歩を打ったときに打ち歩詰めにならないならtrue。
	// 歩をtoに打つことと、二歩でないこと、toの前に敵玉がいることまでは確定しているものとする。
	// 二歩の判定もしたいなら、legal_pawn_drop()のほうを使ったほうがいい。
	bool legal_drop(const Square to) const;

	// 二歩でなく、かつ打ち歩詰めでないならtrueを返す。
	bool legal_pawn_drop(const Color us, const Square to) const
	{
		return !((pieces(us, PAWN) & FILE_BB[file_of(to)])                             // 二歩
			|| ((pawnEffect(us, to) == Bitboard(king_square(~us)) && !legal_drop(to)))); // 打ち歩詰め
	}

	// --- StateInfo

	// 現在の局面に対応するStateInfoを返す。
	// たとえば、state()->capturedPieceであれば、前局面で捕獲された駒が格納されている。
	StateInfo* state() const { return st; }

	// --- Evaluation

#ifndef EVAL_NO_USE
  // 評価関数で使うための、どの駒番号の駒がどこにあるかなどの情報。
	const Eval::EvalList* eval_list() const { return &evalList; }
#endif

#if defined (USE_SEE)
	// 指し手mのsee(Static Exchange Evaluation : 静的取り合い評価)において
	// v(しきい値)以上になるかどうかを返す。
	// see_geのgeはgreater or equal(「以上」の意味)の略。
	bool see_ge(Move m, Value v) const;

#endif

	// --- Accessing hash keys

	// StateInfo::key()への簡易アクセス。
	Key key() const { return st->key(); }


	// --- misc

	// 現局面で王手がかかっているか
	bool in_check() const { return checkers(); }

	// avoidで指定されている遠方駒は除外して、pinされている駒のbitboardを得る。
	// ※利きのない1手詰め判定のときに必要。
	Bitboard pinned_pieces(Color c, Square avoid) const;

	// fromからtoに駒が移動したものと仮定して、pinを得る
	// ※利きのない1手詰め判定のときに必要。
	Bitboard pinned_pieces(Color c, Square from, Square to) const;

	// 駒を配置して、内部的に保持しているBitboardなどを更新する。
	void put_piece(Square sq, Piece pc, PieceNo piece_no);

	// 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
	void remove_piece(Square sq);

	// 指し手mで王手になるかを判定する。
	// 指し手mはpseudo-legal(擬似合法)の指し手であるものとする。
	bool gives_check(Move m) const;

	// 手番側の駒をfromからtoに移動させると素抜きに遭うのか？
	bool discovered(Square from, Square to, Square ourKing, const Bitboard& pinned) const
	{
		// 1) pinされている駒がないなら素抜きにならない。
		// 2) pinされている駒でなければ素抜き対象ではない
		// 3) pinされている駒でも王と(縦横斜において)直線上への移動であれば合法
		return pinned                        // 1)
			&& (pinned & from)               // 2)
			&& !aligned(from, to , ourKing); // 3)
	}

	// 現局面で指し手がないかをテストする。指し手生成ルーチンを用いるので速くない。探索中には使わないこと。
	bool is_mated() const;

	// 直前の指し手によって捕獲した駒。先後の区別あり。
	Piece captured_piece() const { return st->capturedPiece; }

	// 捕獲する指し手か、成りの指し手であるかを返す。
	bool capture_or_promotion(Move m) const { return (m & MOVE_PROMOTE) || capture(m); }

	// 歩の成る指し手であるか？
	bool pawn_promotion(Move m) const
	{
#ifdef KEEP_PIECE_IN_GENERATE_MOVES
		// 移動させる駒が歩かどうかは、Moveの上位16bitを見れば良い
		return (is_promote(m) && raw_type_of(moved_piece_after(m)) == PAWN);
#else
		return (is_promote(m) && type_of(piece_on(move_from(m))) == PAWN);
#endif
	}

	// 捕獲する指し手か、歩の成りの指し手であるかを返す。
	bool capture_or_pawn_promotion(Move m) const
	{
		return pawn_promotion(m) || capture(m);
	}

#if 1
	// 捕獲か価値のある駒の成り。(歩、角、飛車)
	bool capture_or_valuable_promotion(Move m) const
	{
#ifdef KEEP_PIECE_IN_GENERATE_MOVES
		// 歩の成りを角・飛車の成りにまで拡大する。
		auto pr = raw_type_of(moved_piece_after(m));
		return (is_promote(m) && (pr == PAWN || pr == BISHOP || pr == ROOK)) || capture(m);
#else
		auto pr = type_of(piece_on(move_from(m)));
		return (is_promote(m) && (pr == PAWN || pr == BISHOP || pr == ROOK)) || capture(m);
#endif
	}
#endif

	// 捕獲する指し手であるか。
	bool capture(Move m) const { return !is_drop(m) && piece_on(move_to(m)) != NO_PIECE; }

	// --- 1手詰め判定
#ifdef USE_MATE_1PLY
  // 現局面で1手詰めであるかを判定する。1手詰めであればその指し手を返す。
  // ただし1手詰めであれば確実に詰ませられるわけではなく、簡単に判定できそうな近接王手による
  // 1手詰めのみを判定する。(要するに判定に漏れがある。)
  // 
  // 返し値は、16bitのMove。このあとpseudo_legal()等を使いたいなら、
  // pos.move16_to_move()を使って32bitのMoveに変換すること。

	Move mate1ply() const;

	// ↑の先後別のバージョン。(内部的に用いる)
	template <Color Us> Move mate1ply_impl() const;

	// 利きのある場所への取れない近接王手からのply手詰め
	// ply = 1,3,5,…,
	Move weak_mate_n_ply(int ply) const;

#endif

	// 入玉時の宣言勝ち
#ifdef USE_ENTERING_KING_WIN
  // Search::Limits.enteringKingRuleに基いて、宣言勝ちを行なう。
  // 条件を満たしているとき、MOVE_WINや、玉を移動する指し手(トライルール時)が返る。さもなくば、MOVE_NONEが返る。
  // mate1ply()から内部的に呼び出す。(そうするとついでに処理出来て良い)
	Move DeclarationWin() const;
#endif

	// -- sfen化ヘルパ
#ifdef USE_SFEN_PACKER
  // packされたsfenを得る。引数に指定したバッファに返す。
  // gamePlyはpackに含めない。
	void sfen_pack(PackedSfen& sfen);

	// packされたsfenを解凍する。sfen文字列が返る。
	// gamePly = 0となる。
	static std::string sfen_unpack(const PackedSfen& sfen);

	// ↑sfenを経由すると遅いので直接packされたsfenをセットする関数を作った。
	// pos.set(sfen_unpack(data)); と等価。
	void set_from_packed_sfen(const PackedSfen& sfen);

	// 盤面と手駒、手番を与えて、そのsfenを返す。
	static std::string sfen_from_rawdata(Piece board[81], Hand hands[2], Color turn, int gamePly);

	// sq1,sq2の駒を入れ替える。(という指し手だと思うと良い)
	// 棋譜生成のときなど特殊な用途に用いる。王手されている局面で呼び出してはならない。
	// もし歩が1段目にあるなど、非合法局面に突入するなら2駒を入れ替えずにfalseを返す。
	// ※　内部的に一端sfen()化するのだが、そのときにsfen_from_rawdata()を用いるのでsfen_packer.cppに依存。
	bool do_move_by_swapping_pieces(Square sq1, Square sq2);
#endif

	// -- 利き
#ifdef LONG_EFFECT_LIBRARY

  // 各升の利きの数
	LongEffect::ByteBoard board_effect[COLOR_NB];

	// 長い利き(これは先後共用)
	LongEffect::WordBoard long_effect;
#endif

	// --- デバッグ用の出力

#ifdef KEEP_LAST_MOVE
  // 開始局面からこの局面にいたるまでの指し手を表示する。
	std::string moves_from_start() const { return moves_from_start(false); }
	std::string moves_from_start_pretty() const { return moves_from_start(true); }
	std::string moves_from_start(bool is_pretty) const;
#endif

	// 盤面を出力する。(USI形式ではない) デバッグ用。
	friend std::ostream& operator<<(std::ostream& os, const Position& pos);

	// MoveGenerator(指し手生成器)からは盤面・手駒にアクセス出来る必要があるのでfriend
	template <MOVE_GEN_TYPE gen_type, bool gen_all>
	friend struct MoveGenerator;

private:
	// StateInfoの初期化(初期化するときに内部的に用いる)
	void set_state(StateInfo* si) const;

	// 王手になるbitboard等を更新する。set_state()とdo_move()のときに自動的に行われる。
	// null moveのときは利きの更新を少し端折れるのでフラグを渡すことに。
	template <bool doNullMove>
	void set_check_info(StateInfo* si) const;

	// do_move()の先後分けたもの。内部的に呼び出される。
	template <Color Us> void do_move_impl(Move m, StateInfo& st, bool givesCheck);

	// undo_move()の先後分けたもの。内部的に呼び出される。
	template <Color Us> void undo_move_impl(Move m);

	// --- Bitboards
	// alignas(16)を要求するものを先に宣言。

	// 盤上の先手/後手/両方の駒があるところが1であるBitboard
	Bitboard occupied[COLOR_NB + 1];

	// stが初期状態で指している、空のStateInfo
	StateInfo startState;

	// sqの地点にpcを置く/取り除く、したとして内部で保持しているBitboardを更新する。
	void xor_piece(Piece pc, Square sq);

#ifndef EVAL_NO_USE
	// --- 盤面を更新するときにEvalListの更新のために必要なヘルパー関数

	// c側の手駒ptの最後の1枚のBonaPiece番号を返す
	Eval::BonaPiece bona_piece_of(Color c, Piece pt) const {
		// c側の手駒ptの枚数
		int ct = hand_count(hand[c], pt);
		ASSERT_LV3(ct > 0);
		return (Eval::BonaPiece)(Eval::kpp_hand_index[c][pt].fb + ct - 1);
	}

	// c側の手駒ptの(最後の1枚の)PieceNoを返す
	PieceNo piece_no_of(Color c, Piece pt) const { return evalList.piece_no_of(bona_piece_of(c, pt)); }

	// 盤上のpcの駒のPieceNoを返す
	PieceNo piece_no_of(Piece pc, Square sq) const { return evalList.piece_no_of((Eval::BonaPiece)(Eval::kpp_board_index[pc].fb + sq)); }
#else
	// 駒番号を使わないとき用のダミー
	PieceNo piece_no_of(Color c, Piece pt) const { return PIECE_NO_ZERO; }
	PieceNo piece_no_of(Piece pc, Square sq) const { return PIECE_NO_ZERO; }
#endif
	// ---

	// 盤面、81升分の駒。
	Piece board[SQ_NB];

	// 手駒
	Hand hand[COLOR_NB];

	// 手番
	Color sideToMove;

	// 玉の位置
	Square kingSquare[COLOR_NB];

	// 初期局面からの手数(初期局面 == 1)
	int gamePly;

	// この局面クラスを用いて探索しているスレッド
	Thread* thisThread;

	// 探索ノード数 ≒do_move()の呼び出し回数。
	uint64_t nodes;

	// 現局面に対応するStateInfoのポインタ。
	// do_move()で次の局面に進むときは次の局面のStateInfoへの参照をdo_move()の引数として渡される。
	//   このとき、undo_move()で戻れるようにStateInfo::previousに前のstの値を設定しておく。
	// undo_move()で前の局面に戻るときはStateInfo::previousから辿って戻る。
	StateInfo* st;

#ifndef EVAL_NO_USE
	// 評価関数で用いる駒のリスト
	Eval::EvalList evalList;
#endif

};

// PieceからPieceTypeBitboardへの変換テーブル
const PieceTypeBitboard piece2ptb[PIECE_WHITE] = {
  PIECE_TYPE_BITBOARD_NB /*NO_PIECE*/,PIECE_TYPE_BITBOARD_PAWN /*歩*/,PIECE_TYPE_BITBOARD_LANCE /*香*/,PIECE_TYPE_BITBOARD_KNIGHT /*桂*/,
  PIECE_TYPE_BITBOARD_SILVER /*銀*/,PIECE_TYPE_BITBOARD_BISHOP /*角*/,PIECE_TYPE_BITBOARD_ROOK /*飛*/,PIECE_TYPE_BITBOARD_GOLD /*金*/,
  PIECE_TYPE_BITBOARD_HDK /*玉*/, PIECE_TYPE_BITBOARD_GOLD /*歩成*/ , PIECE_TYPE_BITBOARD_GOLD /*香成*/,PIECE_TYPE_BITBOARD_GOLD/*桂成*/,
  PIECE_TYPE_BITBOARD_GOLD /*銀成*/,PIECE_TYPE_BITBOARD_BISHOP/*馬*/,PIECE_TYPE_BITBOARD_ROOK/*龍*/ ,PIECE_TYPE_BITBOARD_NB/*金成*/ };

inline void Position::xor_piece(Piece pc, Square sq)
{
  Color c = color_of(pc);
  const Bitboard q = Bitboard(sq);
  // 先手・後手の駒のある場所を示すoccupied bitboardの更新
  occupied[c] ^= q;
  // 先手 or 後手の駒のある場所を示すoccupied bitboardの更新
  occupied[COLOR_ALL] ^= sq;

  // 駒別のBitboardの更新
  Piece pt = type_of(pc);
  piece_bb[piece2ptb[pt]][c] ^= sq;

  // 馬、龍は、piece_bbのPIECE_TYPE_BITBOARD_BISHOP(ROOK)とPIECE_TYPE_BITBOARD_HDKの両方のbitboardにまたがって存在するので
  // PIECE_TYPE_BITBOARD_HDKのほうも更新する必要がある。
  if (pt >= HORSE)
    piece_bb[PIECE_TYPE_BITBOARD_HDK][c] ^= sq;
}

// 駒を配置して、内部的に保持しているBitboardも更新する。
inline void Position::put_piece(Square sq, Piece pc,PieceNo piece_no)
{
  ASSERT_LV2(board[sq] == NO_PIECE);
  board[sq] = pc;
  xor_piece(pc, sq);

  // 駒番号をセットしておく必要がある。
  ASSERT_LV3(is_ok(piece_no));
  
#ifndef EVAL_NO_USE
  // evalListのほうを更新しないといけない
  evalList.put_piece(piece_no,sq,pc); // sqの升にpcの駒を配置する
#endif

  // 王なら、その升を記憶しておく。
  // (王の升はBitboardなどをみればわかるが、頻繁にアクセスするのでcacheしている。)
  if (type_of(pc) == KING)
    kingSquare[color_of(pc)] = sq;
}

// 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
inline void Position::remove_piece(Square sq)
{
  Piece pc = board[sq];
  ASSERT_LV3(pc != NO_PIECE);
  board[sq] = NO_PIECE;
  xor_piece(pc, sq);
}

inline bool is_ok(Position& pos) { return pos.pos_is_ok(); }

// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os, const Position& pos);

// depthに応じたZobrist Hashを得る。depthを含めてhash keyを求めたいときに用いる。
HASH_KEY DepthHash(int depth);

#endif // of #ifndef _SHOGI_H_
