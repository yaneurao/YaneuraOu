#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED
#include <deque>
#include <memory> // For std::unique_ptr

#include "bitboard.h"
#include "evaluate.h"
#include "types.h"

#if defined(EVAL_NNUE)
#include "eval/nnue/nnue_accumulator.h"
#endif

#include "extra/key128.h"
#include "extra/long_effect.h"
#include "misc.h"

namespace YaneuraOu {

// --------------------
//     局面の情報
// --------------------

// StateInfo struct stores information needed to restore a Position object to
// its previous state when we retract a move. Whenever a move is made on the
// board (by calling Position::do_move), a StateInfo object must be passed.

// StateInfo構造体は、指し手を戻す際にPositionオブジェクトを
// 以前の状態に復元するために必要な情報を格納する。
// 盤上で指し手が行われるたびに（Position::do_moveを呼び出す際に）、
// StateInfoオブジェクトを渡さなければならない。

// 💡 StateInfoは、undo_move()で局面を戻すときに情報を元の状態に戻すのが
//     面倒なものを詰め込んでおくための構造体。
//     do_move()のときは、ブロックコピーで済むのでそこそこ高速。

struct StateInfo {

	// Copied when making a move
    // 指し手で局面を進めるときにコピーされる。

#if defined(USE_PARTIAL_KEY)
	// 位置を無視したPiece(手番考慮ありの駒)によるhash key
    /*  📓
			StockfishはZobrist::psq[pc][8 + 枚数] でxorしていく。
			枚数にすると実装が面倒になるので、やねうら王は、
			Zobrist::psq[pc][8]をaddしていく。
	*/
    Key materialKey;

	// 歩のhash key(盤上のみ)
    Key pawnKey;

	// 小駒(香、桂、銀、金 とその成り駒)によるhash key
	// 📝 チェスだとKnight, Bishop。
    Key minorPieceKey;

	// 歩以外の駒によるhash key(盤上のみ)
    Key nonPawnKey[COLOR_NB];
#endif

#if STOCKFISH
	// 歩以外の駒割。
	// 🤔 やねうら王では使っていない。使ったほうがいいか？
    Value nonPawnMaterial[COLOR_NB];

    int castlingRights;
    int rule50;
#endif

	// 遡り可能な手数(previousポインタを用いて局面を遡るときに用いる)
	int pliesFromNull;

#if STOCKFISH
    Square epSquare;
#else
    // 🌈 この手番側の連続王手は何手前からやっているのか(連続王手の千日手の検出のときに必要)
    int continuousCheck[COLOR_NB];
#endif


	// 📌 ここまではdo_move()のなかでmemcpy()でコピーされる 📌


	// Not copied when making a move (will be recomputed anyhow)
	// 指し手で局面を進めるときにコピーされない(なんにせよ再計算される)

#if STOCKFISH
	Key        key;
#else
	// 盤面(盤上の駒)と手駒に関するhash key
	// 直接アクセスせずに、hand_key()、board_key(),key()を用いること。

	// 💡 board_keyはZobrist::psqをxorしていく。hand_keyはZobrist::handを加算していく。key = board_key ^ hand_key。

	Key board_key;
    Key hand_key;

	// この局面のハッシュキー
	// ※　次の局面にdo_move()で進むときに最終的な値が設定される
	// board_keyは盤面のhash。hand_keyは手駒のhash。それぞれxorしたのがkey 盤面のhash。
	// board_keyのほうは、手番も込み。
    /*
		📓 board_keyがなぜ必要なのか？

		盤面が同じで手駒だけ損している局面(劣等局面)を検出するためには、
		同一の盤面であるかを高速に調べる必要があり、それには盤面のhash keyが必要となる。
		それがboard_keyである。

		⚠ KeyからKey64(64bit key)が欲しい場合、
		    暗黙の変換子が定義されているので単にKey64へcastすると良い。
	*/

	Key key() const { return board_key ^ hand_key; }

#endif

	// 現局面で手番側に対して王手をしている駒のbitboard
	Bitboard checkersBB;

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

	// 動かすと手番側の王に対して空き王手になるかも知れない駒の候補
	// チェスの場合、駒がほとんどが大駒なのでこれらを動かすと必ず開き王手となる。
	// 将棋の場合、そうとも限らないので移動方向について考えなければならない。
	// color = 手番側 なら pinされている駒(動かすと開き王手になる)
	// color = 相手側 なら 両王手の候補となる駒。

	// 自玉に対して(敵駒によって)pinされている駒
	// blockersForKing[c]は、c側の玉に対するpin駒。すなわちc側,~c側、どちらの駒をも含む。
	Bitboard blockersForKing[COLOR_NB];

	// 自玉に対してpinしている(可能性のある)敵の大駒。
	// 自玉に対して上下左右方向にある敵の飛車、斜め十字方向にある敵の角、玉の前方向にある敵の香、…
	// ※ pinners[BLACK]は、BLACKの王に対して(pin駒が移動した時に)王手になる駒だから、WHITE側の駒。
	Bitboard pinners[COLOR_NB];

	// 自駒の駒種Xによって敵玉が王手となる升のbitboard
	Bitboard checkSquares[PIECE_TYPE_NB];

	// この局面で捕獲された駒。先後の区別あり。
    // ※　次の局面にdo_move()で進むときにこの値が設定される
    Piece capturedPiece;

#if !defined(ENABLE_QUICK_DRAW)
    //  循環局面であることを示す。
    //   0    = 循環なし
    //   ply  = ply前の局面と同じ局面であることを表す。(ply > 0) 3回目までの繰り返し。
    //  -ply  = ply前の局面と同じ局面であることを示す。4回目の繰り返しに到達していることを示す。
    int repetition;

#if !STOCKFISH
    //  繰り返された回数 - 1。
    //  📝 repetition != 0の時に意味をなす。
	//      将棋では同じ局面は連続4回目で千日手が成立するのでこのためのカウンター。
    int repetition_times;

    //  その時の繰り返しの種類
    RepetitionState repetition_type;
#endif

#endif


#if !STOCKFISH
	// この局面における手番側の持ち駒。優等局面の判定のために必要。
	Hand hand;
#endif

	// --- evaluate

#if defined(USE_PIECE_VALUE)
        // この局面での評価関数の駒割
        Value materialValue;
#endif

#if defined(USE_CLASSIC_EVAL)

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)

	// 評価値。(次の局面で評価値を差分計算するときに用いる)
	// まだ計算されていなければsum.p[2][0]の値はint_max
	Eval::EvalSum sum;

#endif

#if defined(EVAL_NNUE)
	Eval::NNUE::Accumulator accumulator;
#endif

#if defined (USE_EVAL_LIST)
	// 評価値の差分計算の管理用
	Eval::DirtyPiece dirtyPiece;
#endif

#if defined(KEEP_LAST_MOVE)
	// 直前の指し手。デバッグ時などにおいてその局面までの手順を表示出来ると便利なことがあるのでそのための機能
	Move lastMove;

	// lastMoveで移動させた駒(先後の区別なし)
	PieceType lastMovedPieceType;
#endif

#endif
};

// --------------------
//     局面の定数
// --------------------

/// A list to keep track of the position states along the setup moves (from the
/// start position to the position just before the search starts). Needed by
/// 'draw by repetition' detection. Use a std::deque because pointers to
/// elements are not invalidated upon list resizing.

// setup moves("position"コマンドで設定される、現局面までの指し手)に沿った局面の状態を追跡するためのStateInfoのlist。
// 千日手の判定のためにこれが必要。std::dequeを使っているのは、StateInfoがポインターを内包しているので、resizeに対して
// 無効化されないように。
using StateList    = std::deque<StateInfo>;
using StateListPtr = std::unique_ptr<StateList>;

// --------------------
//       盤面
// --------------------

#if defined(USE_SFEN_PACKER)

// packされたsfen
struct PackedSfen {
	u8 data[32];

	// 手番を返す。
	Color color() const {
		// これは、data[0]のbit0に格納されていることは保証されている。
		return Color(data[0] & 1);
	}

	// std::unordered_mapで使用できるように==と!=を定義しておく。

	bool operator==(const PackedSfen& rhs) const {
		static_assert(sizeof(PackedSfen) % sizeof(u64) == 0);

		for (size_t i = 0; i < sizeof(PackedSfen); i += sizeof(size_t))
		{
			// 8バイト単位で比較していく。一度でも内容が異なれば不一致。
			if (*(u64*)&data[i] != *(u64*)&rhs.data[i])
				return false;
		}
		return true;
	}

	bool operator!=(const PackedSfen& rhs) const {
		return !(this->operator==(rhs));
	}
};

// std::unordered_mapで使用できるようにhash関数を定義しておく。
// std::unordered_map<PackedSfen,int,PackedSfenHash> packed_sfen_to_int;のようにtemplateの第3引数に指定する。
struct PackedSfenHash {
	size_t operator()(const PackedSfen& ps) const
	{
		static_assert(sizeof(PackedSfen) % sizeof(size_t) == 0);

		size_t s = 0;
		for (size_t i = 0; i < sizeof(PackedSfen) ; i+= sizeof(size_t))
		{
			// size_tのsize分ずつをxorしていき、それをhash keyとする。
			s ^= *(size_t*)&ps.data[i];
		}
		return s;
	}
};
#endif

// 盤面
class Position
{
public:
    // Positionで用いるZobristテーブルの初期化
    // 起動時に呼び出す。
    static void init();

	/*
		⚠ Positionのコンストラクタで平手に初期化すると、compute_eval() が呼び出され、このときに
		    評価関数テーブルを参照するが、isready()が呼び出されていないので評価関数パラメーターが
		    読み込まれておらず、この初期化が出来ない。ゆえにコンストラクタでは平手に初期化しない。
	*/

	Position()                           = default;
	Position(const Position&)            = delete;
	Position& operator=(const Position&) = delete;

    // -----------------------
    // FEN string input/output
	//   SFEN文字列の入出力
    // -----------------------

    // sfen文字列で盤面を設定する
    /* 📓 StateInfoとは？

		局面を遡るために、rootまでの局面の情報が必要であるから、それを引数のsiで渡してやる。
	*/
#if STOCKFISH
	Position& set(const std::string& fenStr,bool isChess960, StateInfo* si);
    Position& set(const std::string& code, Color c, StateInfo* si);
#else
    Position& set(const std::string& sfenStr, StateInfo* si);

	// 平手の初期盤面を設定する。
    // siについては、上記のset()にある説明を読むこと。
    void set_hirate(StateInfo* si) { set(StartSFEN, si); }
#endif

	// 局面のsfen文字列を取得する
	// 📝 USIプロトコルにおいては不要な機能ではあるが、デバッグのために局面を標準出力に出力して
	// 　 その局面から開始させたりしたいときに、sfenで現在の局面を出力出来ないと困るので用意してある。
	//    引数としてintを取るほうのsfen()は、出力するsfen文字列の末尾の手数を指定できるバージョン。
	// 💡 裏技 : gamePlyが負なら、sfen文字列末尾の手数を出力しない。

#if STOCKFISH
    std::string fen() const;
#else
	const std::string sfen() const { return sfen(game_ply()); }
	const std::string sfen(int gamePly) const;

	// sfen()のflip(先後反転 = 盤面を180度回転)させた時のsfenを返す。
	const std::string flipped_sfen() const { return flipped_sfen(game_ply()); }
	const std::string flipped_sfen(int gamePly) const;

	// sfen文字列をflip(先後反転)したsfen文字列に変換する。
	static const std::string sfen_to_flipped_sfen(std::string sfen);

#endif


	// c側の手駒を返す。
	Hand hand_of(Color c) const { ASSERT_LV3(is_ok(c));  return hand[c]; }

	// ↑のtemplate版
	template <Color C>
	Hand hand_of() const { ASSERT_LV3(is_ok(C));  return hand[C]; }


	// 現局面に対して
	// この指し手によって移動させる駒を返す。(移動前の駒)
	// 駒打ちに対しては、その打つ駒が返る。
	// また、後手の駒打ちは後手の(その打つ)駒が返る。
	Piece moved_piece_before(Move m) const
	{
		ASSERT_LV3(m.is_ok());
#if defined( KEEP_PIECE_IN_GENERATE_MOVES)
		// 上位16bitに格納されている値を利用する。
		// return is_promote(m) ? (piece & ~PIECE_PROMOTE) : piece;
		// みたいなコードにしたいので、
		// mのMOVE_PROMOTEのbitで、PIECE_PROMOTEのbitを反転させてやる。
		static_assert(MOVE_PROMOTE == (1 << 15) && PIECE_PROMOTE == 8, "");
		return (Piece)((m ^ ((m & MOVE_PROMOTE) << 4)) >> 16);

#else
		return m.is_drop() ? make_piece(sideToMove , m.move_dropped_piece()) : piece_on(m.from_sq());
#endif
	}

	// moved_pieceの拡張版。
	// 成りの指し手のときは成りの指し手を返す。(移動後の駒)
	// Moveの上位16bitにそれが格納されているので、単にそれを返しているだけ。
	Piece moved_piece_after(Move m) const
	{
		// ASSERT_LV3(is_ok(m));
		// ⇨ MovePicker から Move::none()に対してこの関数が呼び出されることがあるのでこのASSERTは書けない。

		return m.moved_after_piece();
	}

	// 指し手mで移動させた駒(成りの指し手である場合は、成った後の駒)
    // 💡 Stockfishの探索部で用いているので、それと互換性を保つために用意。
    // 📝 Stockfishでは移動させた駒(moved_piece_before())を期待しているが、
	//     moved_piece_after()にしたほうが強いっぽいので、やねうら王では
	//     moved_piece()は、moved_piece_after()のaliasとする。
    Piece moved_piece(Move m) const;


	// 定跡DBや置換表から取り出したMove16(16bit型の指し手)を32bit化する。
	// is_ok(m) == false ならば、mをそのまま返す。
	// 例 : MOVE_WINやMOVE_NULLに対してはそれがそのまま返ってくる。つまり、この時、上位16bitは0(NO_PIECE)である。
	//
	// ※　このPositionクラスが保持している現在の手番(side_to_move)が移動させる駒に反映される。
	// ※　mの移動元の駒が現在の手番の駒でなければ、MOVE_NONEが返ることは保証される。
	// ※  mの移動元に駒がない場合も、MOVE_NONEが返ることは保証される。
	Move to_move(Move16 m) const;

	// 1. ENABLE_QUICK_DRAWがdefineされている時
	//		この関数は無視される。
	//
	// 2. ENABLE_QUICK_DRAWがdefineされていない時
	// 　　is_repetition() , has_repeted()で最大で何手前からの千日手をチェックするか。デフォルト16手。
	// 
	// ※　これを MAX_PLY に設定すると初手からのチェックになるが、将棋はチェスと異なり
	// 　　終局までの平均手数がわりと長いので、そこまでするとスピードダウンしてR40ほど弱くなる。
	void set_max_repetition_ply(int ply){ max_repetition_ply = ply;}

	// -----------------------
    // Position representation
    //       局面の表現
    // -----------------------

	// 駒に対応するBitboardを得る。
    /*
		📓
			・引数でcの指定がないものは先後両方の駒が返る。
			・引数がPieceTypeのものは、PieceTypeのPAWN～DRAGON 以外に
				PieceTypeの GOLDS(金相当の駒) , HDK(馬・龍・玉) , BISHOP_HORSE(角・馬) , ROOK_DRAGON(飛車・龍)などが指定できる。
				💡 詳しくは、PieceTypeの定義を見ること。
			・引数でPieceTypeを複数取るものはそれらの駒のBitboardを合成したものが返る。
			・pr として ALL_PIECESを指定した場合、先手か後手か、いずれかの駒がある場所が1であるBitboardが返る。
	*/

	// すべての駒のBitboardが返る。
	Bitboard pieces() const;  // All pieces

	// pieces(PAWN, LANCE)のように使える。
	template<typename... PieceTypes>
	Bitboard pieces(PieceTypes... pts) const;

	// c == BLACK : 先手の駒があるBitboardが返る
    // c == WHITE : 後手の駒があるBitboardが返る
    Bitboard pieces(Color c) const;

	// c側のptsで指定した駒種のBitboardが返る。
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;

	// sのマスにある駒を返す。
    // 💡 sq == SQ_NBの時、NO_PIECEが返ることは保証されている。
    Piece piece_on(Square sq) const;

#if STOCKFISH
    Square ep_square() const;
#else
	// 駒がない升が1になっているBitboardが返る
    Bitboard empties() const { return pieces() ^ Bitboard(1); }
#endif

	// マスsに駒がなければtrueが返る。
    bool empty(Square s) const;

#if STOCKFISH
	// 駒の枚数を返す。
    template<PieceType Pt>
    int count(Color c) const;

	// 駒の枚数を返す。(先後のPtの合計)
    template<PieceType Pt>
    int count() const;
#endif

	// c側のPtの場所(1枚しかない場合)を取得する。
    // ⚠ やねうら王では、Pt == KINGに対してしか使えないが、Stockfishも
	//     そういう使い方しかしていないので問題ない。
    template<PieceType Pt>
    Square square(Color c) const;

	// -----------------------
    //      Castling
    //     キャスリング
    // -----------------------

	// 💡 将棋では使わない。
#if STOCKFISH
    CastlingRights castling_rights(Color c) const;
    bool           can_castle(CastlingRights cr) const;
    bool           castling_impeded(CastlingRights cr) const;
    Square         castling_rook_square(CastlingRights cr) const;
#endif

	// -----------------------
    //      Checking
	//        王手
    // -----------------------

	// 現局面で王手している駒
	Bitboard checkers() const { return st->checkersBB; }

	// c側の玉に対してpinしている駒(その駒をc側の玉との直線上から動かしたときにc側の玉に王手となる)
	Bitboard blockers_for_king(Color c) const { return st->blockersForKing[c]; }

	// ↑のtemplate版
	template <Color C>
	Bitboard blockers_for_king() const { return st->blockersForKing[C]; }

	// 現局面で駒Ptを動かしたときに王手となる升を表現するBitboard
	Bitboard check_squares(PieceType pt) const { ASSERT_LV3(pt!= NO_PIECE_TYPE && pt < PIECE_TYPE_NB); return st->checkSquares[pt]; }

	// c側の玉に対してpinしている駒
	// ※ pinされているではなく、pinしているということに注意。
	// 　すなわち、pinされている駒が移動した時に、この大駒によって王が素抜きにあう。
	Bitboard pinners(Color c) const { return st->pinners[c]; }

#if !STOCKFISH
	// c側の玉に対して、指し手mが空き王手となるのか。
	bool is_discovery_check_on_king(Color c, Move m) const { return st->blockersForKing[c] & m.from_sq(); }

	// 現局面で王手がかかっているか。
	// 📝 StockfishのBitboardはuint64_tだが、将棋では128bit構造体なので
	//     boolへの変換を暗黙でやりたくない。
    bool in_check() const { return checkers(); }
#endif

    // -----------------------
    // Attacks to/from a given square
	// 与えられたマスへ/から の 利き
    // -----------------------

	// sqに利きのあるc側の駒を列挙する。cの指定がないものは先後両方の駒が返る。
	// occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして。
	// sq == SQ_NBでの呼び出しは合法。Bitboard(ZERO)が返る。

	Bitboard attackers_to(Square sq) const { return attackers_to(sq, pieces()); }
    Bitboard attackers_to(Square sq, const Bitboard& occ) const;

#if STOCKFISH
    // これはcastlingの判定のためのもの。将棋では使わない。
    bool attackers_to_exist(Square s, Bitboard occupied, Color c) const;

#else

	Bitboard attackers_to(Color c, Square sq) const {
        return c == BLACK ? attackers_to<BLACK>(sq, pieces()) : attackers_to<WHITE>(sq, pieces());
    }
    Bitboard attackers_to(Color c, Square sq, const Bitboard& occ) const {
        return c == BLACK ? attackers_to<BLACK>(sq, occ) : attackers_to<WHITE>(sq, occ);
    }

    template<Color C>
    Bitboard attackers_to(Square sq) const {
        return attackers_to<C>(sq, pieces());
    }

    template<Color C>
    Bitboard attackers_to(Square sq, const Bitboard& occ) const;

    // 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
    Bitboard attackers_to_pawn(Color c, Square pawn_sq) const;

    // attackers_to()で駒があればtrueを返す版。(利きの情報を持っているなら、軽い実装に変更できる)
    // kingSqの地点からは玉を取り除いての利きの判定を行なう。
#if !defined(LONG_EFFECT_LIBRARY)
    bool effected_to(Color c, Square sq) const { return attackers_to(c, sq, pieces()); }
    bool effected_to(Color c, Square sq, Square kingSq) const {
        return attackers_to(c, sq, pieces() ^ kingSq);
    }
#else
    bool effected_to(Color c, Square sq) const { return board_effect[c].effect(sq) != 0; }
    bool effected_to(Color c, Square sq, Square kingSq) const {
        return board_effect[c].effect(sq) != 0
            || ((long_effect.directions_of(c, kingSq) & Effect8::directions_of(kingSq, sq))
                != 0);  // 影の利きがある
    }
#endif

#endif

	// update_slider_blockers() calculates st->blockersForKing[c] and st->pinners[~c],
    // which store respectively the pieces preventing king of color c from being in check
    // and the slider pieces of color ~c pinning pieces of color c to the king.

    // update_slider_blockers()はst->blockersForKing[c]およびst->pinners[~c]を計算します。
    // これらはそれぞれ、色cの王が王手状態になるのを防ぐ駒と、色cの駒を王にピン留めする手番~cの
    // スライダー駒を格納しています。
    // ※　「ピン留め」とは、移動させた時に開き王手となること。

	// 注意)
    // 	 王 歩 ^飛 ^飛
    //  のようなケースにおいては、この両方の飛車がpinnersとして列挙される。(SEEの処理でこういう列挙がなされて欲しいので)

    void update_slider_blockers(Color c) const;

	// c側の駒Ptの利きのある升を表現するBitboardを返す。(MovePickerで用いている。)
	template<Color C , PieceType Pt> Bitboard attacks_by() const;

    // -----------------------
	// Doing and undoing moves
	// 局面を進める/戻す
    // -----------------------

	// 指し手で盤面を1手進める
	// m = 指し手。mとして非合法手を渡してはならない。
	// info = StateInfo。局面を進めるときに捕獲した駒などを保存しておくためのバッファ。
	// このバッファはこのdo_move()の呼び出し元の責任において確保されている必要がある。
	// givesCheck = mの指し手によって王手になるかどうか。
	// この呼出までにst.checkInfo.update(pos)が呼び出されている必要がある。
    /*
		📓 Stockfish 17.1からTTのprefetchのために引数でTranspositionTable*を
			渡すようになったが、やねうら王では、このclassをPosition classと癒着したくない。
			そこで、第四引数をtemplate T*にするが、ここに何も指定しないということもできてほしい。
	*/
#if STOCKFISH
    void do_move(Move m, StateInfo& newSt, bool givesCheck, const TranspositionTable* tt);
#else
	template <typename T = void>
    void do_move(Move m, StateInfo& newSt, bool givesCheck, const T* tt = nullptr);
#endif

	// 💡 上記のdo_move()にはgivesCheckも渡さないといけないが、
	//     mで王手になるかどうかがわからないときはこちらの関数を用いる。
    template<typename T = void>
    void do_move(Move m, StateInfo& newSt, const T* tt = nullptr) {
        do_move(m, newSt, gives_check(m), tt);
    }

	// do_move()で進めた局面を1手戻す。
	void undo_move(Move m);

	// null move用のdo_move()
	// 📝 Tのところには、TranspositionTable ttを渡すことができる。

	void do_null_move(StateInfo& st);

	template<typename T>
    void do_null_move(StateInfo& st, const T& tt);

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
	// 歩の不成に関しては、この関数は常にtrueを返す。(合法扱い)
	bool legal(Move m) const;

	// mがpseudo_legalな指し手であるかを判定する。
    /*
	    📓　pseudo_legalとは

			pseudo_legalとは擬似合法手のこと。ここには、自殺手が含まれている。

			置換表の指し手でdo_move()して良いのかの事前判定のために使われる。
			指し手生成ルーチンのテストなどにも使える。(指し手生成ルーチンはpseudo_legalな指し手を返すはずなので)

			killerのような兄弟局面の指し手がこの局面において合法かどうかにも使う。
			※　置換表の検査だが、pseudo_legal()で擬似合法手かどうかを判定したあとlegal()で自殺手でないことを
			確認しなくてはならない。このためpseudo_legal()とlegal()とで重複する自殺手チェックはしていない。

			is_ok(m)==falseの時、すなわち、m == MOVE_WINやMOVE_NONEのような時に
			Position::to_move(m) == mは保証されており、この時、本関数pseudo_legal(m)がfalseを返すことは保証する。
			generate_all_legal_moves : これがtrueならば、歩の不成も合法手扱い。

		⚠ 常に歩の不成の指し手も合法手として扱いたいならば、
			この関数ではなく、pseudo_legal_s<true>()を用いること。
	*/
    bool pseudo_legal(const Move m, bool generate_all_legal_moves) const;

	// All == false        : 歩や大駒の不成に対してはfalseを返すpseudo_legal()
	template <bool All> bool pseudo_legal_s(const Move m) const;

	// toの地点に歩を打ったときに打ち歩詰めにならないならtrue。
	// 歩をtoに打つことと、二歩でないこと、toの前に敵玉がいることまでは確定しているものとする。
	// 二歩の判定もしたいなら、legal_pawn_drop()のほうを使ったほうがいい。
	bool legal_drop(const Square to) const;

	// 二歩でなく、かつ打ち歩詰めでないならtrueを返す。
	bool legal_pawn_drop(const Color us, const Square to) const;

	// leagl()では、成れるかどうかのチェックをしていない。
	// (先手の指し手を後手の指し手と混同しない限り、指し手生成された段階で
	// 成れるという条件は満たしているはずだから)
	// しかし、先手の指し手を後手の指し手と取り違えた場合、この前提が崩れるので
	// これをチェックするための関数。成れる条件を満たしていない場合、falseが返る。
	bool legal_promote(Move m) const;

	// --- Evaluation

#if defined(USE_EVAL_LIST)
	// 評価関数で使うための、どの駒番号の駒がどこにあるかなどの情報。
	const Eval::EvalList* eval_list() const { return &evalList; }
#endif

#if defined (USE_SEE)
	// 指し手mのsee(Static Exchange Evaluation : 静的取り合い評価)において
	// v(しきい値)以上になるかどうかを返す。
	// see_geのgeはgreater or equal(「以上」の意味)の略。
	bool see_ge(Move m, Value threshold = VALUE_ZERO) const;

#endif

    // -----------------------
    // Accessing hash keys
	// hash keysへのアクセス
    // -----------------------

	// 📝 StateInfoの同名のメンバーへの簡易アクセス。
	// 💡 USE_PARTIAL_KEYがdefineされていない時は、key()以外は使えない。

    Key key() const;
    Key material_key() const;
    Key pawn_key() const;
    Key minor_piece_key() const;
    Key non_pawn_key(Color c) const;

#if !STOCKFISH
	// ある指し手を指した後のhash keyを返す。
	// 将棋だとこの計算にそこそこ時間がかかるので、通常の探索部でprefetch用に
	// これを計算するのはあまり得策ではないが、詰将棋ルーチンでは置換表を投機的に
	// prefetchできるとずいぶん速くなるのでこの関数を用意しておく。
	Key key_after(Move m) const;
#endif

    // -----------------------
    // Other properties of the position
    // この局面についての他のプロパティ
    // -----------------------

    // 現局面の手番を返す。
    Color side_to_move() const;

    // (将棋の)開始局面からの手数を返す。
    // 平手の開始局面なら1が返る。(0ではない)
    int game_ply() const;

#if STOCKFISH
    bool is_chess960() const;
    bool is_draw(int ply) const;
    bool is_repetition(int ply) const;
    bool upcoming_repetition(int ply) const;
    bool has_repeated() const;
    int  rule50_count() const;

#else

    // 普通の千日手、連続王手の千日手等を判定する。
    // そこまでの局面と同一局面であるかを、局面を遡って調べる。
    //
    // 1. ENABLE_QUICK_DRAWがdefineされている時(大会用に少しでも強くしたい時)
    // plyは無視される。遡る手数は16手固定。
    //
    // 2. ENABLE_QUICK_DRAWがdefineされていない時(正確に千日手の判定を行いたい時)
    // 遡る手数は、set_max_repetition_ply()で設定された手数だけ遡る。
    // ply         : rootからの手数。3回目の同一局面の出現まではrootよりは遡って千日手と判定しない。4回目は判定する。
    RepetitionState is_repetition(int ply = 16) const;

    // is_repetition()の、千日手が見つかった時に、現局面から何手遡ったかを返すバージョン。
    // REPETITION_NONEではない時は、found_plyにその値が返ってくる。	// ※　定跡生成の時にしか使わない。
    RepetitionState is_repetition(int ply, int& found_ply) const;

#if !defined(ENABLE_QUICK_DRAW)
    // Tests whether there has been at least one repetition
    // of positions since the last capture or pawn move.
    bool has_repeated() const;
#endif
#endif

#if STOCKFISH
	// 歩以外の駒の駒割。やねうら王では使わない。
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;
#endif

    // Position consistency check, for debugging
	// デバッグのために局面の一貫性のチェック
	// 📝 保持しているデータに矛盾がないかテストする。

	bool pos_is_ok() const;

	// 盤面を180°回転させる。
	void flip();

    // 現在の局面に対応するStateInfoを返す。
    // たとえば、state()->capturedPieceであれば、前局面で捕獲された駒が格納されている。
    StateInfo* state() const { return st; }

	// put_piece()やremove_piece()を用いたときは、最後にupdate_bitboards()を呼び出して
    // bitboardの整合性を保つこと。
    // また、put_piece_simple()は、put_piece()の王の升(kingSquare)を更新しない版。do_move()で用いる。

    // 駒を配置して、内部的に保持しているBitboard、pieceCountも更新する。
    /*
		⚠ : kingを配置したときには、このクラスのkingSqaure[]を更新しないといけないが、
			  この関数のなかでは行っていないので呼び出し側で更新すること。
			  (StockfishはkingSquare[]を持っていないのでStockfishにはこれに該当する処理はない。)

		⚠ :  evalListのほうの更新もこの関数のなかでは行っていないので必要ならば
			   呼び出し側で更新すること。
		例) 
			if (type_of(pc) == KING)
				kingSquare[color_of(pc)] = sq;
			もしくはupdate_kingSquare()を呼び出すこと。
	*/

    void put_piece(Piece pc, Square sq);

    // 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
    void remove_piece(Square sq);

#if !STOCKFISH
	// put_pieceの手駒版
    /*
		📓 pieceCountについて。

		盤面への駒の配置を put_piece(),remove_piece(),put_hand_piece(),remove_hand_piece()の
		4つだけを用いれば、pieceCount[]は正しく更新される。
	*/
    void put_hand_piece(Color c, PieceType pt);

	// remove_pieceの手駒版
    // 💡 pieceCountもきちんと更新する。
    void remove_hand_piece(Color c, PieceType pt);

    // put_piece(),remove_piece()を用いたあとに呼び出す必要がある。
    // 📝 やねうら王ではHDKのような駒が合成されたBitboardを用いるため。
    void update_bitboards();

    // このクラスが保持しているkingSquare[]の更新。
    // put_piece(),remove_piece()では玉の位置(kingSquare[])を
    // 更新してくれないので、自前で更新するか、一連の処理のあとにこの関数を呼び出す必要がある。
    void update_kingSquare();
#endif

	// --- misc

	// ピンされているc側の駒。下手な方向に移動させるとc側の玉が素抜かれる。
	// 手番側のpinされている駒はpos.pinned_pieces(pos.side_to_move())のようにして取得できる。
	// LONG_EFFECT_LIBRARYを使うときのmateルーチンで使用しているので消さないで！
	Bitboard pinned_pieces(Color c) const { ASSERT_LV3(is_ok(c)); return st->blockersForKing[c] & pieces(c); }

	// ↑のtemplate版
	template<Color C>
	Bitboard pinned_pieces() const { ASSERT_LV3(is_ok(C)); return st->blockersForKing[C] & pieces<C>(); }

	// avoidで指定されている遠方駒は除外して、pinされている駒のbitboardを得る。
	// ※利きのない1手詰め判定のときに必要。
	Bitboard pinned_pieces(Color c, Square avoid) const { return c == BLACK ? pinned_pieces<BLACK>(avoid) : pinned_pieces<WHITE>(avoid); }

	template<Color C>
	Bitboard pinned_pieces(Square avoid) const;

	// fromからtoに駒が移動したものと仮定して、pinを得る
	// ※利きのない1手詰め判定のときに必要。
	Bitboard pinned_pieces(Color c, Square from, Square to) const
	{
		return c == BLACK ? pinned_pieces<BLACK>(from,to) : pinned_pieces<WHITE>(from,to);
	}

	// ↑のtemplate版
	template<Color C>
	Bitboard pinned_pieces(Square from, Square to) const;


	// 指し手mで王手になるかを判定する。
	// 前提条件 : 指し手mはpseudo-legal(擬似合法)の指し手であるものとする。
	// (つまり、mのfromにある駒は自駒であることは確定しているものとする。)
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

	// 捕獲する指し手か、成りの指し手であるかを判定する。
	bool capture_or_promotion(Move m) const { return m.is_promote() || capture(m); }

	// 歩の成る指し手であるか？
	bool pawn_promotion(Move m) const
	{
		// 移動させる駒が歩かどうかは、Moveの上位16bitを見れば良い
		return (m.is_promote() && raw_type_of(moved_piece_after(m)) == PAWN);
	}

	// 捕獲する指し手か、歩の成りの指し手であるかを返す。
	bool capture_or_pawn_promotion(Move m) const
	{
		return pawn_promotion(m) || capture(m);
	}

	// 捕獲か価値のある駒の成り。(歩、角、飛車)
	bool capture_or_valuable_promotion(Move m) const
	{
		// 歩の成りを角・飛車の成りにまで拡大する。
		auto pr = raw_type_of(moved_piece_after(m));
		return (m.is_promote() && (pr == PAWN || pr == BISHOP || pr == ROOK)) || capture(m);
	}

	// 捕獲する指し手であるか。
    bool capture(Move m) const;

	// capture_or_pawn_promotion()みたいなもの。
	/*
	    📓 Stockfishでは、この関数は、「捕獲する指し手かQUEENにpromoteする
			指し手かのどちらかであるか」を判定する。

			Stockfishとの互換性のために用意。
			やねうら王では、capture()と同義。
	*/
    bool capture_stage(Move m) const;

	// 入玉時の宣言勝ち
    /*
		📓 宣言勝ちの条件を満たしているとき、MOVE_WINや、玉を移動する指し手(トライルール時)が返る。
			さもなくば、MOVE_NONEが返る。

		⚠ 事前にset_ekr()で入玉ルールが設定されていること。
	*/
	Move DeclarationWin() const;

	// 入玉の宣言勝ちのルールを設定する。このルールに基づいて入玉の計算が行われる。
	// 現在の盤面を見て、平手から足りない駒を計算するので、現在の盤面は適切に設定されている必要がある。
	// 📝 start_searching()のなかで設定すると良いと思う。
    void set_ekr(EnteringKingRule ekr) {
        this->ekr = ekr;
        update_entering_point();
    }

	// -- sfen化ヘルパ
#if defined(USE_SFEN_PACKER)
  // packされたsfenを得る。引数に指定したバッファに返す。
  // gamePlyはpackに含めない。
	void sfen_pack(PackedSfen& sfen);

	// packされたsfenを解凍する。sfen文字列が返る。
	// gamePly = 0となる。
	static std::string sfen_unpack(const PackedSfen& sfen);

	// ↑sfenを経由すると遅いので直接packされたsfenをセットする関数を作った。
	// pos.set(sfen_unpack(data),si); と等価。
	// 渡された局面に問題があって、エラーのときはTools::Result::SomeErrorを返す。
	// PackedSfenにgamePlyは含まないので復元できない。そこを設定したいのであれば引数で指定すること。
	Tools::Result set_from_packed_sfen(const PackedSfen& sfen , StateInfo * si , bool mirror=false , int gamePly_ = 0);

	// 盤面と手駒、手番を与えて、そのsfenを返す。
	static std::string sfen_from_rawdata(Piece board[81], Hand hands[2], Color turn, int gamePly);
#endif

	// -- 利き
#if defined(LONG_EFFECT_LIBRARY)

	// 各升の利きの数
	LongEffect::ByteBoard board_effect[COLOR_NB];

	// NNUE halfKPE9で局面の差分計算をするときに用いる
#if defined(USE_BOARD_EFFECT_PREV)
	// 前局面のboard_effect（評価値の差分計算用）

	// 構造的には、StateInfoが持つべきなのだが、探索のほうで
	// do_move()して次のnodeのsearch()が呼び出された直後にしかevaluate()は
	// 呼び出さないので、do_move()でこの利きをboard_effectからコピーすれば
	// KPE9の差分計算で困ることはない。
	LongEffect::ByteBoard board_effect_prev[COLOR_NB];
#endif

	// 長い利き(これは先後共用)
	LongEffect::WordBoard long_effect;

#endif

	// --- デバッグ用の出力

#if defined(KEEP_LAST_MOVE)
  // 開始局面からこの局面にいたるまでの指し手を表示する。
	std::string moves_from_start() const { return moves_from_start(false); }
	std::string moves_from_start_pretty() const { return moves_from_start(true); }
	std::string moves_from_start(bool is_pretty) const;
#endif

	// 盤面を出力する。(USI形式ではない) デバッグ用。
	friend std::ostream& operator<<(std::ostream& os, const Position& pos);

	// UnitTest
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);

private:
    // Initialization helpers (used while setting up a position)
	 // 初期化用のヘルパー（局面を設定する際に使用）

#if STOCKFISH
	void set_castling_right(Color c, Square rfrom);
#endif

	// StateInfoの初期化。Position::set()のタイミングで行われる。
	void set_state() const;

#if STOCKFISH
    void set_check_info() const;
#else
	// 王手になるbitboard等を更新する。set_state()とdo_move()から呼び出される。
	// 🌈 null moveのときは利きの更新を少し端折れるのでフラグを渡すことにした。
	template <bool doNullMove,Color Us>
	void set_check_info() const;

	template <bool doNullMove>
	void set_check_info() const
	{
		sideToMove == BLACK ? set_check_info<doNullMove, BLACK>() : set_check_info<doNullMove, WHITE>();
	}
#endif

    // Other helpers


	// do_move()の先後分けたもの。内部的に呼び出される。
    template<Color Us, typename T>
    void do_move_impl(Move m, StateInfo& st, bool givesCheck, const T* tt);

	// undo_move()の先後分けたもの。内部的に呼び出される。
    template<Color Us>
    void undo_move_impl(Move m);

	// 📝 update_entering_point()は、現在の盤面から、事前にset_ekr_ruleで設定されたルールに基づき、
	//     入玉に必要な駒点を計算し、enteringKingPoint[]に設定する。
	//     ここで設定された値が、DeclarationWin()の時に用いられる。

	void update_entering_point();
    EnteringKingRule ekr = EKR_NULL;
	int enteringKingPoint[COLOR_NB];


#if defined (USE_EVAL_LIST)
	// --- 盤面を更新するときにEvalListの更新のために必要なヘルパー関数

	// c側の手駒ptの最後の1枚のBonaPiece番号を返す
	Eval::BonaPiece bona_piece_of(Color c, PieceType pt) const {
		// c側の手駒ptの枚数
		int ct = hand_count(hand[c], pt);
		ASSERT_LV3(ct > 0);
		return (Eval::BonaPiece)(Eval::kpp_hand_index[c][pt].fb + ct - 1);
	}

	// c側の手駒ptの(最後の1枚の)PieceNumberを返す。
	PieceNumber piece_no_of(Color c, PieceType pt) const { return evalList.piece_no_of_hand(bona_piece_of(c, pt)); }

	// 盤上のsqの升にある駒のPieceNumberを返す。
	PieceNumber piece_no_of(Square sq) const
	{
		ASSERT_LV3(piece_on(sq) != NO_PIECE);
		PieceNumber n = evalList.piece_no_of_board(sq);
		ASSERT_LV3(is_ok(n));
		return n;
	}
#else
	// 駒番号を使わないとき用のダミー
	PieceNumber piece_no_of(Color c, Piece pt) const { return PIECE_NUMBER_ZERO; }
	PieceNumber piece_no_of(Piece pc, Square sq) const { return PIECE_NUMBER_ZERO; }
	PieceNumber piece_no_of(Square sq) const { return PIECE_NUMBER_ZERO; }
#endif

	// --------------------
    //    Data members
    // --------------------

	// 盤面、81升分の駒 + 1
    Piece board[SQ_NB_PLUS1];

	// 駒が存在する升を表すBitboard。先後混在。
    // pieces()の引数と同じく、ALL_PIECES,HDKなどのPieceで定義されている特殊な定数が使える。
    Bitboard byTypeBB[PIECE_BB_NB];

    // 盤上の先手/後手/両方の駒があるところが1であるBitboard
    Bitboard byColorBB[COLOR_NB];

#if STOCKFISH
    // 各駒の数
	// 💡 手駒も含む。後手の手駒は、後手のPieceとみなしてカウントする。
    int pieceCount[PIECE_NB];
	// ⇨  やねうら王では使わないことにした。

    int      castlingRightsMask[SQUARE_NB];
    Square   castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
#endif

	// 現局面に対応するStateInfoのポインタ。
    // do_move()で次の局面に進むときは次の局面のStateInfoへの参照をdo_move()の引数として渡される。
    //   このとき、undo_move()で戻れるようにStateInfo::previousに前のstの値を設定しておく。
    // undo_move()で前の局面に戻るときはStateInfo::previousから辿って戻る。
    StateInfo* st;

	// 初期局面からの手数(初期局面 == 1)
    int gamePly;

    // 手番
    Color sideToMove;

#if STOCKFISH
    bool chess960;
#else
    // 手駒
    Hand hand[COLOR_NB];

    // 玉の位置
    Square kingSquare[COLOR_NB];

	// set_max_repetition_ply()で設定される、千日手の最大遡り手数
    static int max_repetition_ply /* = 16 */;

#if defined(USE_EVAL_LIST)
    // 評価関数で用いる駒のリスト
    Eval::EvalList evalList;
#endif

#endif
};


// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os, const Position& pos);

inline Color Position::side_to_move() const { return sideToMove; }

inline Piece Position::piece_on(Square s) const {
    ASSERT_LV3(is_ok(s));
    return board[s];
}

inline bool Position::empty(Square s) const { return piece_on(s) == NO_PIECE; }

inline Piece Position::moved_piece(Move m) const {
#if STOCKFISH
    return piece_on(m.from_sq());
#else
    // 📝 Stockfishでは移動させた駒(moved_piece_before())を期待しているが、
    //     moved_piece_after()にしたほうが強いっぽいので、やねうら王では
    //     moved_piece()は、moved_piece_after()のaliasとする。

	return moved_piece_after(m);
#endif
}

inline Bitboard Position::pieces() const { return byTypeBB[ALL_PIECES]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(PieceTypes... pts) const {
	// 🤔 fold expressionで実装する。
    return (byTypeBB[pts] | ...);
}

inline Bitboard Position::pieces(Color c) const { return byColorBB[c]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(Color c, PieceTypes... pts) const {
    return pieces(c) & pieces(pts...);
}

#if STOCKFISH
template<PieceType Pt>
inline int Position::count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
}

template<PieceType Pt>
inline int Position::count() const {
    return count<Pt>(WHITE) + count<Pt>(BLACK);
}

template<PieceType Pt>
inline Square Position::square(Color c) const {
    assert(count<Pt>(c) == 1);
    return lsb(pieces(c, Pt));
}
#else
template<PieceType Pt>
inline Square Position::square(Color c) const {
    // やねうら王ではPt == KINGしか許容しない。
    static_assert(Pt == KING);

    // 📝 やねうら王では、lsb()が重いのでこれを使わない実装を考える。
    return kingSquare[c];
}
#endif

#if STOCKFISH
inline Square Position::ep_square() const { return st->epSquare; }

inline bool Position::can_castle(CastlingRights cr) const { return st->castlingRights & cr; }

inline CastlingRights Position::castling_rights(Color c) const {
    return c & CastlingRights(st->castlingRights);
}

inline bool Position::castling_impeded(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return pieces() & castlingPath[cr];
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return castlingRookSquare[cr];
}
#endif

// 🚧


#if STOCKFISH
inline Key Position::key() const { return adjust_key50<false>(st->key); }

template<bool AfterMove>
inline Key Position::adjust_key50(Key k) const {
    return st->rule50 < 14 - AfterMove ? k : k ^ make_key((st->rule50 - (14 - AfterMove)) / 8);
}
#else
inline Key Position::key() const { return st->key(); }
#endif

#if defined(USE_PARTIAL_KEY)
inline Key Position::pawn_key() const { return st->pawnKey; }

inline Key Position::material_key() const { return st->materialKey; }

inline Key Position::minor_piece_key() const { return st->minorPieceKey; }

inline Key Position::non_pawn_key(Color c) const { return st->nonPawnKey[c]; }
#else

// 使わないのでダミーの値を返すようにしておく。

inline Key Position::pawn_key() const { return Key(); }
inline Key Position::material_key() const { return Key(); }
inline Key Position::minor_piece_key() const { return Key(); }
inline Key Position::non_pawn_key(Color c) const { return Key(); }

#endif

#if STOCKFISH
inline Value Position::non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }

inline Value Position::non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
}
#endif

inline int Position::game_ply() const { return gamePly; }

#if STOCKFISH
inline int Position::rule50_count() const { return st->rule50; }

inline bool Position::is_chess960() const { return chess960; }

inline bool Position::capture(Move m) const {
    assert(m.is_ok());
    return (!empty(m.to_sq()) && m.type_of() != CASTLING) || m.type_of() == EN_PASSANT;
}
#else
inline bool Position::capture(Move m) const {
    ASSERT_LV3(m.is_ok());
    return !m.is_drop() && piece_on(m.to_sq()) != NO_PIECE;
}
#endif


// returns true if a move is generated from the capture stage
// having also queen promotions covered, i.e. consistency with the capture stage move generation
// is needed to avoid the generation of duplicate moves.

// キャプチャ段階で生成された指し手であれば true を返す
// クイーン昇格も含めてカバーする。つまり、キャプチャ段階での指し手生成との整合性を
// 保つ必要があり、重複した指し手の生成を避けるためである。
// 💡 「キャプチャ段階」とは、MovePickerでの指し手生成のうち捕獲する指し手の生成フェーズのこと。

inline bool Position::capture_stage(Move m) const
{
#if STOCKFISH
    assert(is_ok(m));
    return  capture(m) || promotion_type(m) == QUEEN;
#else

    // 📊　V7.73y3とy4,y5の比較。
    //      return capture_or_valuable_promotion(m);
    //      return capture_or_pawn_promotion(m);
    //      よりは、単にcapture()にするのが良かった。

    return capture(m);
#endif
}

// 駒を配置して、内部的に保持しているBitboard、pieceCountも更新する。
inline void Position::put_piece(Piece pc, Square s) {

	board[s] = pc;

    // byTypeBB[type_of(pc)]は、駒別のBitboard
	// byTypeBB[ALL_PIECES ]は、任意の駒がある場所を示すBitboard
	// これらを同時に更新する。
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;

	// 先手・後手の駒のある場所を示すoccupied bitboardの更新
    byColorBB[color_of(pc)] |= s;

#if STOCKFISH
	// 駒のカウント
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;
#endif
}

// 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
inline void Position::remove_piece(Square s) {

	Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
#if STOCKFISH
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
#endif
}

#if !STOCKFISH

// put_pieceの手駒版
inline void Position::put_hand_piece(Color c, PieceType pr)
{
	add_hand(hand[c], pr);
    //pieceCount[make_piece(c, pr)]++;
    //pieceCount[make_piece(c, ALL_PIECES)]++;
}

// remove_pieceの手駒版
inline void Position::remove_hand_piece(Color c, PieceType pr)
{
	sub_hand(hand[c], pr);
    //pieceCount[make_piece(c, pr)]--;
    //pieceCount[make_piece(c, ALL_PIECES)]--;
}

inline bool is_ok(Position& pos) { return pos.pos_is_ok(); }
#endif

// 🚧



// sに利きのあるc側の駒を列挙する。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
template<Color C>
inline Bitboard Position::attackers_to(Square sq, const Bitboard& occ) const {
    ASSERT_LV3(is_ok(C) && sq <= SQ_NB);

    constexpr Color Them = ~C;

    // clang-format off

	// sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
	// 香の利きを求めるコストが惜しいのでrookEffect()を利用する。
	return
		(     (pawnEffect  <Them>(sq)	&  pieces(PAWN)        )
			| (knightEffect<Them>(sq)	&  pieces(KNIGHT)      )
			| (silverEffect<Them>(sq)	&  pieces(SILVER_HDK)  )
			| (goldEffect  <Them>(sq)	&  pieces(GOLDS_HDK)   )
			| (bishopEffect(sq, occ)	&  pieces(BISHOP_HORSE))
			| (rookEffect(sq, occ)		& (pieces(ROOK_DRAGON) | (lanceStepEffect<Them>(sq) & pieces(LANCE))))
			//  | (kingEffect(sq) & pieces(c, HDK));
			// →　HDKは、銀と金のところに含めることによって、参照するテーブルを一個減らして高速化しようというAperyのアイデア。
			) & pieces(C); // 先後混在しているのでc側の駒だけ最後にマスクする。
	;
    // clang-format on
}

// c側の駒Ptの利きのある升を表現するBitboardを返す。(MovePickerで用いている。)
// 遠方駒に関しては盤上の駒を考慮した利き。
template<Color C, PieceType Pt>
Bitboard Position::attacks_by() const {
    if constexpr (Pt == PAWN)
        return C == WHITE ? pawn_attacks_bb<WHITE>(pieces(WHITE, PAWN))
                          : pawn_attacks_bb<BLACK>(pieces(BLACK, PAWN));
    else
    {
        Bitboard threats   = Bitboard(ZERO);
        Bitboard attackers = pieces(C, Pt);
        while (attackers)
            threats |= attacks_bb<make_piece(C, Pt)>(attackers.pop(), pieces());
        return threats;
    }
}

// ピンされているc側の駒。下手な方向に移動させるとc側の玉が素抜かれる。
// avoidで指定されている遠方駒は除外して、pinされている駒のbitboardを得る。
template<Color C>
Bitboard Position::pinned_pieces(Square avoid) const {
    // clang-format off

	Bitboard b, pinners, result = Bitboard(ZERO);
    Square   ksq = square<KING>(C);

	// avoidを除外して考える。
	Bitboard avoid_bb = ~Bitboard(avoid);

	pinners = (
		(  pieces(ROOK_DRAGON)   & rookStepEffect    (ksq))
		| (pieces(BISHOP_HORSE)  & bishopStepEffect  (ksq))
		| (pieces(LANCE)         & lanceStepEffect<C>(ksq)))
                & avoid_bb & pieces(~C);

	while (pinners)
	{
		b = between_bb(ksq, pinners.pop()) & pieces() & avoid_bb;
		if (!b.more_than_one())
            result |= b & pieces(C);
	}
	return result;

    // clang-format on
}

// ピンされているc側の駒。下手な方向に移動させるとc側の玉が素抜かれる。
// fromからtoに駒が移動したものと仮定して、pinを得る
template<Color C>
Bitboard Position::pinned_pieces(Square from, Square to) const {
    // clang-format off

	Bitboard b, pinners, result = Bitboard(ZERO);
    Square   ksq = square<KING>(C);

	// avoidを除外して考える。
	Bitboard avoid_bb = ~Bitboard(from);

	pinners = (
		(  pieces(ROOK_DRAGON)  & rookStepEffect    (ksq))
		| (pieces(BISHOP_HORSE) & bishopStepEffect  (ksq))
		| (pieces(LANCE)        & lanceStepEffect<C>(ksq)))
            & avoid_bb & pieces(~C);

	// fromからは消えて、toの地点に駒が現れているものとして
	Bitboard new_pieces = (pieces() & avoid_bb) | to;
	while (pinners)
	{
		b = between_bb(ksq, pinners.pop()) & new_pieces;
		if (!b.more_than_one())
			result |= b & pieces(C);
	}
	return result;

    // clang-format on
}



} // namespace YaneuraOu

#endif // of #ifndef POSITION_H_INCLUDED
