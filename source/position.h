#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <array>
#include <cassert>
#include <iosfwd>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

#include "attacks.h"
#include "bitboard.h"
#include "misc.h"
#include "types.h"

#if !STOCKFISH
#include "evaluate.h"

#include "extra/key128.h"
#include "extra/long_effect.h"
#endif

namespace YaneuraOu {

struct SharedHistories;

// --------------------
//     局面の情報
// --------------------

// StateInfo struct stores information needed to restore a Position object to
// its previous state when we retract a move. Whenever a move is made on the
// board (by calling Position::do_move), a StateInfo object must be passed.
// StateInfo構造体は、指し手を戻す際にPositionオブジェクトを以前の状態へ復元する情報を保持する。
// 上の原文は、着手時に呼び出し側からStateInfoを渡していた旧APIの説明。
// 現在はStockfish・やねうら王ともにPosition内部で保持し、do_move()で次の状態を保存する。

// 💡 StateInfoは、undo_move()で局面を戻すときに情報を元の状態に戻すのが
//     面倒なものを詰め込んでおくための構造体。
//     do_move()のときは、ブロックコピーで済むのでそこそこ高速。

struct StateInfo {
#if STOCKFISH
    // Copied when making a move
    // 着手時に前の状態からコピーする。
    Key    materialKey;
    Key    pawnKey;
    Key    minorPieceKey;
    Key    nonPawnKey[COLOR_NB];
    Value  nonPawnMaterial[COLOR_NB];
    int    castlingRights;
    int    rule50;
    u16    repWindow;
    Square epSquare;

    // Not copied when making a move (will be recomputed anyhow)
    // 着手時にはコピーせず、再計算する。
    Key                key;
    Bitboard           checkersBB;
    Bitboard           blockersForKing[COLOR_NB];
    Bitboard           pinners[COLOR_NB];
    Bitboard           checkSquares[PIECE_TYPE_NB];
    Bitboard           threats[PIECE_TYPE_NB];
    Piece              capturedPiece;
    int                repetition;
    std::array<u8, 32> piecePlacement;
#else
	// Copied when making a move
    // 指し手で局面を進めるときにコピーされる。

#if defined(USE_PARTIAL_KEY)
	// 位置を無視したPiece(手番考慮ありの駒)によるhash key
    /*  📓
			StockfishはZobrist::psq[pc][8 + 枚数] でxorしていく。
			枚数にすると実装が面倒になるので、やねうら王は、
			Zobrist::psq[pc][8]をaddしていく。
	*/
    HASH_KEY materialKey;

	// 歩のhash key(盤上のみ)
    HASH_KEY pawnKey;

	// 小駒(香、桂、銀、金 とその成り駒)によるhash key
	// 📝 チェスだとKnight, Bishop。
    HASH_KEY minorPieceKey;

	// 歩以外の駒によるhash key(盤上のみ)
    HASH_KEY nonPawnKey[COLOR_NB];
#endif


    // 最後のnull move以降で、保持している履歴を実際に遡れる手数。
    // 捕獲・歩の移動でも以前の局面へ戻り得るため、Stockfishと異なり0へ戻さない。
    // Stockfishと異なり型はintのまま。設定上限は検査時に適用する。
    int repWindow;

    // 🌈 この手番側の連続王手は何手前からやっているのか(連続王手の千日手の検出のときに必要)
    int continuousCheck[COLOR_NB];

	// 📌 ここまではdo_move()のなかでmemcpy()でコピーされる 📌

	// Not copied when making a move (will be recomputed anyhow)
	// 指し手で局面を進めるときにコピーされない(なんにせよ再計算される)

	// 盤面(盤上の駒)と手駒に関するhash key
	// Positionの外部からはboard_key()、key()で取得する。

	// 💡 board_keyはZobrist::psqをxorしていく。hand_keyはZobrist::handを加算していく。key = board_key ^ hand_key。

	HASH_KEY board_key;
    HASH_KEY hand_key;

	// この局面全体のキーはboard_keyとhand_keyのXORで求める。
	// set()で初期化し、do_move()で次局面のキーへ更新する。
	// board_keyのほうは、手番も込み。
    /*
		📓 board_keyがなぜ必要なのか？

		盤面が同じで手駒だけ損している局面(劣等局面)を検出するためには、
		同一の盤面であるかを高速に調べる必要があり、それには盤面のhash keyが必要となる。
		それがboard_keyである。
	*/

	HASH_KEY key() const { return board_key ^ hand_key; }


	// 現局面で手番側に対して王手をしている駒のbitboard
	Bitboard checkersBB;


	// blockersForKing[c]は、c側の玉と敵の長い利きの間を遮る唯一の駒。
	// c側・~c側のどちらの駒も含む。c側の駒なら自玉に対するピン、
	// ~c側の駒ならc側の玉への開き王手の候補となる。
	// チェス・将棋とも、移動後も利きを遮る場合があるので移動方向も考慮する。
	Bitboard blockersForKing[COLOR_NB];

	// pinners[c]は、~c側の駒を~c側の玉に対してピンしているc側の長い利きの駒。
	// 飛車・龍、角・馬、向きの合う香が対象。添字はピンする駒自身の色を表す。
	// 例: pinners[BLACK]には、WHITEの玉に対してWHITEの駒をピンするBLACKの駒が入る。
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

    //  繰り返された回数 - 1。
    //  📝 repetition != 0の時に意味をなす。
	//      将棋では同じ局面は連続4回目で千日手が成立するのでこのためのカウンター。
    int repetition_times;

    //  その時の繰り返しの種類
    RepetitionState repetition_type;

#endif


	// この局面における手番側の持ち駒。優等局面の判定のために必要。
	Hand hand;

	// --- evaluate

#if defined(USE_MATERIAL_VALUE)
        // この局面での評価関数の駒割
        Value materialValue;
#endif

#if defined(USE_CLASSIC_EVAL)

#if defined(KEEP_LAST_MOVE)
	// 直前の指し手。デバッグ時などにおいてその局面までの手順を表示出来ると便利なことがあるのでそのための機能
	Move lastMove;

	// lastMoveで移動させた駒(先後の区別なし)
	PieceType lastMovedPieceType;
#endif

#endif
#endif
};

// --------------------
//     局面の定数
// --------------------

// This error should be used whenever a position is suspected to be unsupported
// by the engine. In particular positions that may cause hard errors like segmentation fault.
// エンジンが扱えない疑いのある局面、特にセグメンテーション違反などの
// 致命的なエラーを引き起こすおそれがある局面に対して用いる。
struct PositionSetError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

// --------------------
//       盤面
// --------------------

#if !STOCKFISH
// packされたsfen
struct PackedSfen {
	u8 data[32];

	// 盤面を180度回転し、先後と手番を入れ替える。
	// SFEN文字列やPosition::set()を経由せず、packed sfen上で変換する。
	void flip();
	PackedSfen flipped() const;

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

// PSV形式(.psv)の1レコード。
// PackedSfenに、手番側視点のscoreなどを付与した40 bytes固定長形式。
struct PsvRecord {
	PackedSfen sfen;
	s16        score;
	u16        move;
	u16        gamePly;
	s8         game_result;
	u8         padding;
};

static_assert(sizeof(PsvRecord) == 40, "PsvRecord must be 40 bytes");

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

// Position class stores information regarding the board representation as
// pieces, side to move, hash keys, castling info, etc. Important methods are
// do_move() and undo_move(), used by the search to update node info when
// traversing the search tree.
// Positionクラスは、駒の配置、手番、ハッシュキーなどの局面情報を保持する。
// do_move()とundo_move()は、探索木をたどる際に局面情報を更新する重要なメソッドである。
// 将棋では手駒や入玉ルールも保持する。
class Position {
   public:
#if !STOCKFISH
    // Positionで用いるZobristテーブルの初期化
    // 起動時に呼び出す。
    static void init();
#endif

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
    // StateInfoはPosition内部に保持する。スレッド間のコピーはcopy_state_infos()を使う。
#if STOCKFISH
	std::optional<PositionSetError> set(const std::string& fenStr, bool isChess960);
    std::optional<PositionSetError> set(const std::string& code, Color c);
#else
    std::optional<PositionSetError> set(const std::string& sfenStr);

	// 平手の初期盤面を設定する。
    std::optional<PositionSetError> set_hirate();
#endif

	// 局面のsfen文字列を取得する
	// 📝 USIプロトコルにおいては不要な機能ではあるが、デバッグのために局面を標準出力に出力して
	// 　 その局面から開始させたりしたいときに、sfenで現在の局面を出力出来ないと困るので用意してある。
	//    引数としてintを取るほうのsfen()は、出力するsfen文字列の末尾の手数を指定できるバージョン。
	// 💡 裏技 : gamePlyが負なら、sfen文字列末尾の手数を出力しない。

#if STOCKFISH
    std::string fen() const;
#else
	const std::string sfen() const;
	const std::string sfen(int gamePly) const;

	// sfen()のflip(先後反転 = 盤面を180度回転)させた時のsfenを返す。
	const std::string flipped_sfen() const;
	const std::string flipped_sfen(int gamePly) const;

	// sfen文字列をflip(先後反転)したsfen文字列に変換する。
	static const std::string sfen_to_flipped_sfen(std::string sfen);

#endif


#if !STOCKFISH
    // 現在の手番側の持ち駒を返す。
    Hand hand_of() const;

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

	// 定跡DBや置換表から取り出したMove16(16bit型の指し手)を32bit化する。
	// is_ok(m) == false ならば、mをそのまま返す。
	// 例 : MOVE_WINやMOVE_NULLに対してはそれがそのまま返ってくる。つまり、この時、上位16bitは0(NO_PIECE)である。
	//
	// ※　このPositionクラスが保持している現在の手番(side_to_move)が移動させる駒に反映される。
	// ※　mの移動元の駒が現在の手番の駒でなければ、MOVE_NONEが返ることは保証される。
	// ※  mの移動元に駒がない場合も、MOVE_NONEが返ることは保証される。
	Move to_move(Move16 m) const;

    // 千日手検査の遡り上限を設定する。既定値は16手。
    // QUICK無効時はdo_move()での繰り返し情報の計算とhas_repeated()に用いる。
    // QUICK有効時のis_repetition(ply)でも、この設定を遡り上限に用いる。
    // found_plyを返す版は、QUICK有効時にもplyと設定上限の両方で検査範囲を制限する。
    // 両モードとも、局面コピーではmax(16, 設定値)を基準に過去のStateInfoを保持する。
    // 上限を増やすと検査コストやコピーする履歴量が増える。許容範囲は下の保持領域の検査による。
	// Reject a window that could overwrite the root history during a full-depth search.
    // 最大深さの探索でルート以前の履歴を上書きする長さは拒否する。失敗時は設定を維持する。
    bool set_max_repetition_ply(int ply);

#endif

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
    // 💡 s == SQ_NBの時、NO_PIECEが返ることは保証されている。
    Piece piece_on(Square s) const;

	// (盤上の)駒の配列をもらう。
#if STOCKFISH
    const std::array<Piece, SQUARE_NB>& piece_array() const;
#else
    const std::array<Piece, SQ_NB_PLUS1>& piece_array() const;
#endif

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
    // Castling
    //     キャスリング
    // -----------------------

	// 💡 将棋では使わない。
#if STOCKFISH
    bool   can_castle(CastlingRights cr) const;
    bool   castling_impeded(CastlingRights cr) const;
    Square castling_rook_square(CastlingRights cr) const;
#endif

	// -----------------------
    // Checking
	//        王手
    // -----------------------

	// 現局面で王手している駒
	Bitboard checkers() const;

	// c側の玉への遠方駒の利きを遮る駒。自駒・敵駒の両方を含む。
	Bitboard blockers_for_king(Color c) const;

	// 現局面で駒Ptを動かしたときに王手となる升を表現するBitboard
	Bitboard check_squares(PieceType pt) const;

	// c側の遠方駒のうち、相手の駒を相手玉に対してピンしている駒。
	// blockers_for_king(c)とは異なり、c側の玉を遮る駒の集合ではない。
	// ピンされた相手駒が直線を外れると、この遠方駒が相手玉へ利く。
	Bitboard pinners(Color c) const;

    // -----------------------
    // Attacks to/from a given square
	// 与えられたマスへ/から の 利き
    // -----------------------

	// sqに利きのあるc側の駒を列挙する。cの指定がないものは先後両方の駒が返る。
	// occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして。
	// sq == SQ_NBでの呼び出しは合法。Bitboard(ZERO)が返る。

	Bitboard attackers_to(Square s) const;
#if STOCKFISH
    Bitboard attackers_to(Square s, Bitboard occupied) const;
#else
    Bitboard attackers_to(Square s, const Bitboard& occupied) const;
#endif

#if STOCKFISH
    void     update_slider_blockers(Color c) const;
    Bitboard xattacks_by(PieceType pt, Color c, Bitboard occupied) const;
    Bitboard threats_by(PieceType pt) const;

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
    Bitboard attackers_to(Square s, const Bitboard& occupied) const;

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

#if !STOCKFISH
    void update_slider_blockers(Color c) const;

	// c側の駒Ptの利きのある升を表現するBitboardを返す。(MovePickerで用いている。)
	template<Color C , PieceType Pt> Bitboard attacks_by() const;
#endif

    // Properties of moves
    // 指し手の性質。
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

#if !STOCKFISH
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

#endif

	// 捕獲する指し手であるか。
    bool capture(Move m) const;

	/*
	    📓 Stockfishでは、この関数は、「捕獲する指し手かQUEENにpromoteする
			指し手かのどちらかであるか」を判定する。

			Stockfishとの互換性のために用意。
			やねうら王では、capture()と同義。
	*/
    bool capture_stage(Move m) const;

	// 指し手mで王手になるかを判定する。
	// 前提条件 : 指し手mはpseudo-legal(擬似合法)の指し手であるものとする。
	// (つまり、mのfromにある駒は自駒であることは確定しているものとする。)
	bool gives_check(Move m) const;

    // 指し手mで移動する駒。Stockfishは移動前、将棋版は成り後の駒を返す。
    // 将棋版はこの値を継続履歴などの添字に用いるため、従来どおり成り後の駒を使う。
    Piece moved_piece(Move m) const;

	// 直前の指し手によって捕獲した駒。先後の区別あり。
	Piece captured_piece() const;

    // -----------------------
	// Doing and undoing moves
	// 局面を進める/戻す
    // -----------------------

	// 指し手で盤面を1手進める
	// m = 指し手。mとして非合法手を渡してはならない。
	// StateInfoはPosition内部のリングバッファに保存する。
	// givesCheck = mの指し手によって王手になるかどうか。
	// 王手情報は局面設定や直前のdo_move()/do_null_move()で更新されている。
#if STOCKFISH
    void do_move(Move m);
    void
    do_move(Move m, bool givesCheck, Dirties& dirties, const SharedHistories* worker = nullptr);
#else
    // ttは次局面のTTエントリーのprefetch専用。TTの読み書きは行わない。
    // PositionをTranspositionTableの具体型に依存させず、省略もできるようにtemplate T*で渡す。
    // historyは省略可能。探索で次局面の履歴を先読みするときだけ渡す。
	template <typename T = void>
    void do_move(Move m, bool givesCheck, const T* tt = nullptr,
                 const SharedHistories* history = nullptr);
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
    template <typename T = void>
    void do_move(Move m, bool givesCheck, Eval::NNUE::Dirties& dirties,
                 const T* tt = nullptr, const SharedHistories* history = nullptr);
#endif
#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
    template <typename T = void>
    void do_move(Move m, bool givesCheck, Eval::DirtyPiece& dirty,
                 const T* tt = nullptr, const SharedHistories* history = nullptr);
#endif
#if !defined(EVAL_SFNN) && !defined(EVAL_NNUE) && !defined(EVAL_KPPT) && !defined(EVAL_KPP_KKPT)
    // 空の差分は無視し、既存の差分記録なしの経路へ渡す。
    template <typename T = void>
    void do_move(Move m, bool givesCheck, Eval::EmptyDirties&,
                 const T* tt = nullptr, const SharedHistories* history = nullptr) {
        do_move(m, givesCheck, tt, history);
    }
#endif
#endif

	// 💡 上記のdo_move()にはgivesCheckも渡さないといけないが、
	//     mで王手になるかどうかがわからないときはこちらの関数を用いる。
#if !STOCKFISH
    void do_move(Move m) { do_move(m, gives_check(m)); }
#endif

	// do_move()で進めた局面を1手戻す。
	void undo_move(Move m);

	// null move用のdo_move()
	// 📝 Tのところには、TranspositionTable ttを渡すことができる。

	void do_null_move();

#if !STOCKFISH
	template<typename T>
    void do_null_move(const T& tt);
#endif

	// null move用のundo_move()
	void undo_null_move();

	// --- Evaluation

#if !STOCKFISH


#endif

    // Static Exchange Evaluation
    // 静的交換評価。
	// 指し手mのsee(Static Exchange Evaluation : 静的取り合い評価)において
	// v(しきい値)以上になるかどうかを返す。
	// see_geのgeはgreater or equal(「以上」の意味)の略。
    SYSV_ABI bool see_ge(Move m, int threshold = 0) const;


    // -----------------------
    // Accessing hash keys
	// hash keysへのアクセス
    // -----------------------

	// 📝 StateInfoの同名のメンバーへの簡易アクセス。
	// 💡 USE_PARTIAL_KEYがdefineされていない時は、key()以外は使えない。
    // Stockfishの同名アクセサーは64bitのKeyを返すが、将棋版はHASH_KEY_BITSに応じた
    // HASH_KEYを返す。TT等で使うKey自体は将棋版も64bit固定なので混同しないこと。

#if STOCKFISH
    Key key() const;
    Key prefetch_key(Move m) const;
    Key material_key() const;
    Key pawn_key() const;
    Key minor_piece_key() const;
    Key non_pawn_key(Color c) const;
#else
    HASH_KEY key() const;
    HASH_KEY material_key() const;
    HASH_KEY pawn_key() const;
    HASH_KEY minor_piece_key() const;
    HASH_KEY non_pawn_key(Color c) const;
#endif

#if !STOCKFISH
    // 盤上の駒と手番からなるキー。持ち駒は含まない。
    HASH_KEY board_key() const;

	// ある指し手を指した後のhash keyを返す。
	// 将棋だとこの計算にそこそこ時間がかかるので、通常の探索部でprefetch用に
	// これを計算するのはあまり得策ではないが、詰将棋ルーチンでは置換表を投機的に
	// prefetchできるとずいぶん速くなるのでこの関数を用意しておく。
	HASH_KEY key_after(Move m) const;
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
    // plyは無視される。遡る手数はmax_repetition_plyで制限する(既定値16手)。
    //
    // 2. ENABLE_QUICK_DRAWがdefineされていない時
    // do_move()で設定上限まで遡って計算した繰り返し情報を使う。
    // ply         : rootからの手数。3回目の同一局面の出現まではrootよりは遡って千日手と判定しない。4回目は判定する。
    RepetitionState is_repetition(int ply = 16) const;

    // 千日手が見つかったとき、現局面から何手遡ったかをfound_plyにも返す版。
    // QUICK有効時は、設定上限に加えて引数plyでも遡り範囲を制限する。
    // QUICK無効時は計算済みの繰り返し情報を使う。定跡生成で使用する。
    RepetitionState is_repetition(int ply, int& found_ply) const;

#if !defined(ENABLE_QUICK_DRAW)
    // Tests whether there has been at least one repetition within the retained history.
    // 保持履歴の範囲で、計算済みの繰り返し情報を調べる。
    // 遡る範囲はmax_repetition_ply・最後のnull moveからの手数・保持履歴長で制限する。
    bool has_repeated() const;
#endif
#endif

#if STOCKFISH
	// 歩以外の駒の駒割。やねうら王では使わない。
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;
#endif
    // 反復判定で保持履歴を遡れる手数。将棋版は捕獲・歩の移動ではリセットしない。
    int   rep_window() const;
#if STOCKFISH
    bool  dtz_is_dtm() const;  // Pawnless && (3-men || 4-men-minors-only)
    // 歩がなく、3駒または小駒のみの4駒の局面。
#endif

    // Position consistency check, for debugging
	// デバッグのために局面の一貫性のチェック
	// 📝 保持しているデータに矛盾がないかテストする。

	bool pos_is_ok() const;

#if STOCKFISH
    bool                            material_key_is_ok() const;
#endif
    // 将棋では盤面を180°回転させ、先後を交換する。局面の再設定に失敗したらエラーを返す。
    std::optional<PositionSetError> flip();

#if !STOCKFISH
#if defined(USE_MATERIAL_VALUE)
    // 差分更新された駒割り。先手側から見た値を返す。
    Value material_value() const;
#endif

    static constexpr usize MAX_HIST = POSITION_HISTORY_SIZE;
#if defined(YANEURAOU_MATE_ENGINE) || defined(TANUKI_MATE_ENGINE)
    usize remaining_state_capacity() const { return MAX_HIST - 1 - stateIdx; }
#endif
#endif
    // Copy only the history needed by shogi repetition detection.
    // 千日手判定に必要な過去の履歴を、自分の領域へコピーする。
    void copy_state_infos(const Position& other);
#if !STOCKFISH
    void copy_from(const Position& other);
    // Used for repeated UCT rollouts after copy_from(), without copying history again.
    // copy_from()後のUCT反復では、履歴を再コピーせず盤面と履歴の位置だけ戻す。
    void reset_to_root(const Position& root);
#endif

	// put_piece()やremove_piece()を用いたときは、最後にupdate_bitboards()を呼び出して
    // bitboardの整合性を保つこと。

    // 駒を配置し、駒種別・先後別の基本Bitboardを更新する。合成Bitboardはupdate_bitboards()で更新する。
    /*
		⚠ : kingを配置したときには、このクラスのkingSqaure[]を更新しないといけないが、
			  この関数のなかでは行っていないので呼び出し側で更新すること。
			  (StockfishはkingSquare[]を持っていないのでStockfishにはこれに該当する処理はない。)

		例) 
			if (type_of(pc) == KING)
				kingSquare[color_of(pc)] = sq;
			もしくはupdate_kingSquare()を呼び出すこと。
	*/

#if STOCKFISH
    void put_piece(Piece pc, Square s, DirtyThreats* const dts = nullptr);
    void remove_piece(Square s, DirtyThreats* const dts = nullptr);
    void swap_piece(Square s, Piece pc, DirtyThreats* const dts = nullptr);
#else
    void put_piece(Piece pc, Square sq);

    // 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
    void remove_piece(Square sq);
#endif

#if !STOCKFISH
    // put_piece(),remove_piece()を用いたあとに呼び出す必要がある。
    // 📝 やねうら王ではHDKのような駒が合成されたBitboardを用いるため。
    void update_bitboards();

    // このクラスが保持しているkingSquare[]の更新。
    // put_piece(),remove_piece()では玉の位置(kingSquare[])を
    // 更新してくれないので、自前で更新するか、一連の処理のあとにこの関数を呼び出す必要がある。
    void update_kingSquare();
#endif

#if !STOCKFISH
	// --- misc

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
    void set_ekr(EnteringKingRule ekr);

	// -- sfen化ヘルパ
  // packされたsfenを得る。引数に指定したバッファに返す。
  // gamePlyはpackに含めない。
	void sfen_pack(PackedSfen& sfen);

	// packされたsfenを解凍する。sfen文字列が返る。
	// gamePly = 0となる。
	static std::string sfen_unpack(const PackedSfen& sfen);

	// ↑sfenを経由すると遅いので直接packされたsfenをセットする関数を作った。
	// pos.set(sfen_unpack(data)); と等価。
	// 成功時は空、復元に失敗したときはPositionSetErrorを返す。
	// PackedSfenにgamePlyは含まないので復元できない。そこを設定したいのであれば引数で指定すること。
	std::optional<PositionSetError> set_from_packed_sfen(const PackedSfen& sfen , bool mirror=false , int gamePly_ = 0);

	// 盤面と手駒、手番を与えて、そのsfenを返す。
	static std::string sfen_from_rawdata(Piece board[81], Hand hands[2], Color turn, int gamePly);

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
	std::string moves_from_start() const;
	std::string moves_from_start_pretty() const;
	std::string moves_from_start(bool is_pretty) const;
#endif

	// 盤面を出力する。(USI形式ではない) デバッグ用。
	friend std::ostream& operator<<(std::ostream& os, const Position& pos);

	// UnitTest
	static void UnitTest(Test::UnitTester& tester, IEngine& engine);

#endif

private:
    // Initialization helpers (used while setting up a position)
	 // 初期化用のヘルパー（局面を設定する際に使用）

#if STOCKFISH
	void set_castling_right(Color c, Square rfrom);
    Key  compute_material_key() const;
#endif

	// StateInfoの初期化。Position::set()のタイミングで行われる。
	void set_state() const;

#if STOCKFISH
    void set_extra_bitboards() const;
    void compute_threats() const;
#else
	// 王手になるbitboard等を更新する。set_state()とdo_move()から呼び出される。
	// 🌈 null moveのときは利きの更新を少し端折れるのでフラグを渡すことにした。
	template <bool doNullMove,Color Us>
	void set_extra_bitboards() const;

	template <bool doNullMove>
	void set_extra_bitboards() const
	{
		sideToMove == BLACK ? set_extra_bitboards<doNullMove, BLACK>() : set_extra_bitboards<doNullMove, WHITE>();
	}
#endif

#if STOCKFISH
    // Other helpers
    // その他のヘルパー。
    template<bool ComputeRay = true>
    void update_piece_threats(Piece               pc,
                              bool                putPiece,
                              Square              s,
                              DirtyThreats* const dts,
                              Bitboard            noRaysContaining = -1ULL) const;
    void move_piece(Square from, Square to, DirtyThreats* const dts = nullptr);
    template<bool Do>
    void do_castling(Color               us,
                     Square              from,
                     Square&             to,
                     Square&             rfrom,
                     Square&             rto,
                     DirtyThreats* const dts = nullptr,
                     DirtyPiece* const   dp  = nullptr);
    template<bool AfterMove = false>
    Key adjust_key50(Key k) const;
#endif

#if !STOCKFISH
	// do_move()の先後分けたもの。内部的に呼び出される。
    template<Color Us, typename T>
    void do_move_impl(Move m, bool givesCheck, const T* tt,
                       const SharedHistories* history
#if defined(EVAL_SFNN) || defined(EVAL_NNUE) || defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
                       , Eval::DirtyPiece* dirty
#endif
                       );

	// undo_move()の先後分けたもの。内部的に呼び出される。
    template<Color Us>
    void undo_move_impl(Move m);

	// 📝 update_entering_point()は、現在の盤面から、事前にset_ekr_ruleで設定されたルールに基づき、
	//     入玉に必要な駒点を計算し、enteringKingPoint[]に設定する。
	//     ここで設定された値が、DeclarationWin()の時に用いられる。

	void update_entering_point();
    EnteringKingRule ekr = EKR_NULL;
	int enteringKingPoint[COLOR_NB];




#endif

    // Data members
    // データメンバー。

	// 盤面、81升分の駒 + 1
#if STOCKFISH
    std::array<Piece, SQUARE_NB>        board;
    std::array<Bitboard, PIECE_TYPE_NB> byTypeBB;
#else
    std::array<Piece, SQ_NB_PLUS1>    board;

	// 駒が存在する升を表すBitboard。先後混在。
    // pieces()の引数と同じく、ALL_PIECES,HDKなどのPieceで定義されている特殊な定数が使える。
    std::array<Bitboard, PIECE_BB_NB> byTypeBB;
#endif

    // 盤上の先手・後手の駒を色別に保持する2要素のBitboard配列。
    // 両方の色を合わせた盤面はbyTypeBB[ALL_PIECES]で保持する。
    std::array<Bitboard, COLOR_NB>    byColorBB;

#if STOCKFISH
    // Stockfish側の駒数カウンター。将棋版では保持しない。
    int pieceCount[PIECE_NB];
	// ⇨  やねうら王では使わないことにした。

    int      castlingRightsMask[SQUARE_NB];
    Square   castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
#endif


	// 初期局面からの手数(初期局面 == 1)
    int gamePly;

    // 手番
    Color sideToMove;

#if STOCKFISH
    bool chess960;
    Dirties scratchDirties;
#else
    // 手駒
    Hand hand[COLOR_NB];

    // 玉の位置
    Square kingSquare[COLOR_NB];

	// 千日手検査の遡り上限と、局面コピー時の履歴保持量の基準。set_max_repetition_ply()で設定する。
    static int max_repetition_ply /* = 16 */;



#endif

#if STOCKFISH
    u16        stateIdx;
    StateInfo* st;
    StateInfo  stateInfos[MAX_HIST];
#else
    // State storage is last: copying the board must not copy the entire ring.
    // 状態の領域は末尾に置き、盤面だけのコピーにリング全体を巻き込まない。
    // Keep a logical index; mask only when accessing the ring. This also bounds valid history.
    // 添字は論理的に数え、配列アクセス時だけマスクする。有効な履歴長もこの添字から得る。
    usize stateIdx = 0;
    StateInfo* st = nullptr;
#if !defined(YANEURAOU_MATE_ENGINE) && !defined(TANUKI_MATE_ENGINE)
    static_assert(MAX_HIST >= MAX_PLY + 17);
#endif
    StateInfo stateInfos[MAX_HIST];
    // 手駒の枚数を更新する。盤上の駒にはput_piece()/remove_piece()を使う。
    void put_hand_piece(Color c, PieceType pt);
    void remove_hand_piece(Color c, PieceType pt);

    void reset_state();
    StateInfo* push_state();
    void pop_state();
    StateInfo* previous_state(usize distance = 1) const;
#endif

};


// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os, const Position& pos);

inline Color Position::side_to_move() const { return sideToMove; }

inline Piece Position::piece_on(Square s) const {
#if STOCKFISH
    assert(is_ok(s));
#else
    ASSERT_LV3(is_ok(s));
#endif
    return board[s];
}

#if STOCKFISH
inline const std::array<Piece, SQUARE_NB>& Position::piece_array() const { return board; }
#else
inline const std::array<Piece, SQ_NB_PLUS1>& Position::piece_array() const { return board; }
#endif

inline bool Position::empty(Square s) const { return piece_on(s) == NO_PIECE; }

inline Piece Position::moved_piece(Move m) const {
#if STOCKFISH
    return piece_on(m.from_sq());
#else
    // Stockfishは盤上の移動前の駒を返すが、将棋版はMoveに保持した成り後の駒を返す。
    // Move::none()に対しても呼ばれるため、通常手であるというassertは置かない。
    return m.moved_after_piece();
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

inline bool Position::castling_impeded(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return pieces() & castlingPath[cr];
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return castlingRookSquare[cr];
}
#endif

inline Bitboard Position::attackers_to(Square s) const { return attackers_to(s, pieces()); }

#if STOCKFISH
// includes X-ray attacks through the king of color ~c
// ~c側の玉を透過するX線の利きも含む。
inline Bitboard Position::xattacks_by(PieceType pt, Color c, Bitboard occupied) const {

    if (pt == PAWN)
        return c == WHITE ? pawn_attacks_bb<WHITE>(pieces(WHITE, PAWN))
                          : pawn_attacks_bb<BLACK>(pieces(BLACK, PAWN));
    else
    {
        Bitboard threats   = 0;
        Bitboard attackers = pieces(c, pt);
        while (attackers)
            threats |= Attacks::attacks_bb(pt, pop_lsb(attackers), occupied);
        return threats;
    }
}

inline Bitboard Position::threats_by(PieceType pt) const {
    if (!st->threats[ALL_PIECES])
        compute_threats();
    return st->threats[pt];
}
#endif

inline Bitboard Position::checkers() const { return st->checkersBB; }

inline Bitboard Position::blockers_for_king(Color c) const { return st->blockersForKing[c]; }

inline Bitboard Position::pinners(Color c) const { return st->pinners[c]; }

inline Bitboard Position::check_squares(PieceType pt) const {
#if !STOCKFISH
    ASSERT_LV3(pt != NO_PIECE_TYPE && pt < PIECE_TYPE_NB);
#endif
    return st->checkSquares[pt];
}

#if STOCKFISH
inline Key Position::key() const { return adjust_key50(st->key); }

template<bool AfterMove>
inline Key Position::adjust_key50(Key k) const {
    return st->rule50 < 14 - AfterMove ? k : k ^ make_key((st->rule50 - (14 - AfterMove)) / 8) << 1;
}

#else
inline HASH_KEY Position::key() const { return st->key(); }

inline HASH_KEY Position::board_key() const { return st->board_key; }
#endif

#if STOCKFISH
inline Key Position::pawn_key() const { return st->pawnKey ^ sideToMove; }

inline Key Position::material_key() const { return st->materialKey; }

inline Key Position::minor_piece_key() const { return st->minorPieceKey ^ sideToMove; }

inline Key Position::non_pawn_key(Color c) const { return st->nonPawnKey[c] ^ sideToMove; }

#else
#if defined(USE_PARTIAL_KEY)
inline HASH_KEY Position::pawn_key() const { return st->pawnKey ^ HASH_KEY(sideToMove); }

inline HASH_KEY Position::material_key() const { return st->materialKey; }

inline HASH_KEY Position::minor_piece_key() const { return st->minorPieceKey ^ HASH_KEY(sideToMove); }

inline HASH_KEY Position::non_pawn_key(Color c) const { return st->nonPawnKey[c] ^ HASH_KEY(sideToMove); }
#else

// 使わないのでダミーの値を返すようにしておく。

inline HASH_KEY Position::pawn_key() const { return HASH_KEY(); }
inline HASH_KEY Position::material_key() const { return HASH_KEY(); }
inline HASH_KEY Position::minor_piece_key() const { return HASH_KEY(); }
inline HASH_KEY Position::non_pawn_key(Color c) const { return HASH_KEY(); }

#endif

#endif

#if STOCKFISH
inline Value Position::non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }
#endif

inline int Position::rep_window() const { return st->repWindow; }

#if STOCKFISH
inline Value Position::non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
}
#endif

inline int Position::game_ply() const { return gamePly; }

#if STOCKFISH
inline int Position::rule50_count() const { return st->rule50; }

inline bool Position::is_chess960() const { return chess960; }

inline bool Position::dtz_is_dtm() const {
    return !count<PAWN>()
        && (count<ALL_PIECES>() == 3 || (count<ALL_PIECES>() == 4 && !pieces(QUEEN, ROOK)));
}

inline bool Position::capture(Move m) const {
    assert(m.is_ok());
    const MoveType mt = m.type_of();

    if (mt == NORMAL || mt == PROMOTION)
        return !empty(m.to_sq());

    return mt == EN_PASSANT;
}
#else
inline bool Position::capture(Move m) const {
    ASSERT_LV3(m.is_ok());
    return !m.is_drop() && !empty(m.to_sq());
}
#endif


// Returns true if a move is generated from the capture stage, having also
// queen promotions covered, i.e. consistency with the capture stage move
// generation is needed to avoid the generation of duplicate moves.
// 捕獲段階で生成される指し手ならtrueを返す。Stockfishではクイーンへの成りも含み、
// 重複生成を避けるため、捕獲段階の指し手生成と判定条件を一致させる必要がある。
// 将棋側は捕獲段階で実際の捕獲だけを生成するため、capture()と同じ判定にする。
inline bool Position::capture_stage(Move m) const
{
#if STOCKFISH
    assert(m.is_ok());

    const MoveType mt = m.type_of();

    if (mt == NORMAL)
        return !empty(m.to_sq());

    if (mt == PROMOTION)
        return !empty(m.to_sq()) || m.promotion_type() == QUEEN;

    return mt == EN_PASSANT;
#else

    // 📊　V7.73y3とy4,y5の比較。
    //      捕獲に歩・角・飛車の成りを含める案よりも、単にcapture()にするのが良かった。

    return capture(m);
#endif
}

inline Piece Position::captured_piece() const { return st->capturedPiece; }

#if STOCKFISH
inline void Position::put_piece(Piece pc, Square s, DirtyThreats* const dts) {
    board[s] = pc;
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;

    if (dts)
        update_piece_threats(pc, true, s, dts);
}

inline void Position::remove_piece(Square s, DirtyThreats* const dts) {
    Piece pc = board[s];

    if (dts)
        update_piece_threats(pc, false, s, dts);

    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
}

inline void Position::move_piece(Square from, Square to, DirtyThreats* const dts) {
    Piece    pc     = board[from];
    Bitboard fromTo = from | to;

    if (dts)
        update_piece_threats(pc, false, from, dts, fromTo);

    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to]   = pc;

    if (dts)
        update_piece_threats(pc, true, to, dts, fromTo);
}

inline void Position::swap_piece(Square s, Piece pc, DirtyThreats* const dts) {
    Piece old = board[s];

    remove_piece(s);

    if (dts)
        update_piece_threats<false>(old, false, s, dts);

    put_piece(pc, s);

    if (dts)
        update_piece_threats<false>(pc, true, s, dts);
}

#else
// 駒を配置し、駒種別・先後別の基本Bitboardを更新する。合成Bitboardはupdate_bitboards()で更新する。
inline void Position::put_piece(Piece pc, Square s) {

	board[s] = pc;

    // byTypeBB[type_of(pc)]は、駒別のBitboard
	// byTypeBB[ALL_PIECES ]は、任意の駒がある場所を示すBitboard
	// これらを同時に更新する。
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;

	// 先手・後手の駒のある場所を示すoccupied bitboardの更新
    byColorBB[color_of(pc)] |= s;

}

// 駒を盤面から取り除き、内部的に保持しているBitboardも更新する。
inline void Position::remove_piece(Square s) {

	Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
}

// StateInfo内の現在手番側のスナップショットを読む。DFPNでの従来の参照先を維持する。
inline Hand Position::hand_of() const { return st->hand; }

#if defined(USE_MATERIAL_VALUE)
inline Value Position::material_value() const { return st->materialValue; }
#endif
#endif

#if !STOCKFISH

inline bool is_ok(Position& pos) { return pos.pos_is_ok(); }
#endif

#if !STOCKFISH
// 🚧


// sに利きのあるc側の駒を列挙する。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
template<Color C>
inline Bitboard Position::attackers_to(Square sq, const Bitboard& occ) const {
    ASSERT_LV3(is_ok(C) && sq <= SQ_NB);

    constexpr Color Them = ~C;
    const auto [bishopAttacks, rookAttacks] = Attacks::both_attacks_bb(sq, occ);

    // clang-format off

	// sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
	// 香の利きを求めるコストが惜しいのでrookEffect()を利用する。
	return
		(     (Attacks::pawnEffect  <Them>(sq)	&  pieces(PAWN)        )
			| (Attacks::knightEffect<Them>(sq)	&  pieces(KNIGHT)      )
			| (Attacks::silverEffect<Them>(sq)	&  pieces(SILVER_HDK)  )
			| (Attacks::goldEffect  <Them>(sq)	&  pieces(GOLDS_HDK)   )
			| (bishopAttacks	&  pieces(BISHOP_HORSE))
			| (rookAttacks		& (pieces(ROOK_DRAGON) | (Attacks::lanceStepEffect<Them>(sq) & pieces(LANCE))))
			//  | (Attacks::kingEffect(sq) & pieces(c, HDK));
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
            threats |= Attacks::attacks_bb(make_piece(C, Pt), attackers.pop_lsb(), pieces());
        return threats;
    }
}
#endif


#if STOCKFISH
inline void Position::do_move(Move m) {
    new (&scratchDirties.dirtyThreats) DirtyThreats;
    do_move(m, gives_check(m), scratchDirties);
}
#endif

} // namespace YaneuraOu

#endif  // #ifndef POSITION_H_INCLUDED
