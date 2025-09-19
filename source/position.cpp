﻿#include <iostream>
#include <sstream>
#include <cstring> // std::memset()
#include <stack>

#include "position.h"
#include "misc.h"
#include "tt.h"
#include "mate/mate.h"
#include "book/book.h"
#include "movegen.h"
#include "testcmd/unit_test.h"

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_NNUE)
#include "eval/evaluate_common.h"
#endif

using namespace std;
namespace YaneuraOu {

#if !STOCKFISH
using namespace Effect8;
using namespace Eval;

// set_max_repetition_ply()で設定される、千日手の最大遡り手数
int Position::max_repetition_ply = 16;

// minor pieceは、香・桂・銀・金とその成駒に限ることにする。
constexpr bool minor_piece_table[PIECE_NB] = {
  false,         false /*歩*/, true /*香*/,  true /*桂*/,      true /*銀*/,   false /*角*/,
  false /*飛*/,  true /*金*/,  false /*玉*/, true /*と*/,      true /*成香*/, true /*成桂*/,
  true /*成銀*/, false /*馬*/, false /*龍*/, false /*未使用*/,

  false,         false /*歩*/, true /*香*/,  true /*桂*/,      true /*銀*/,   false /*角*/,
  false /*飛*/,  true /*金*/,  false /*玉*/, true /*と*/,      true /*成香*/, true /*成桂*/,
  true /*成銀*/, false /*馬*/, false /*龍*/, false /*未使用*/,
};

// minor pieceかを判定する。
constexpr bool is_minor_piece(Piece pc) { return minor_piece_table[pc]; }

#endif

// 局面のhash keyを求めるときに用いるZobrist key
// 💡 board_keyはZobrist::psqをxorしていく。hand_keyはZobrist::handを加算していく。key = board_key ^ hand_key。
namespace Zobrist {

#if STOCKFISH

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side, noPawns;

#else

// 💡 Stockfishでは、Keyはuint64_tなので zeroは単に 0と書けるが、
//     やねうら王は、HASH_KEYは64,128,256bitに拡張しているので…。
Key zero;

// 手番
Key side;

// 駒pcが盤上sqに配置されているときのZobrist Key
// 💡 玉などは盤上にない場合、SQ_NBになるのでSQ_NB_PLUS1で確保する。
Key psq[PIECE_NB][SQ_NB_PLUS1];

// c側の手駒prが一枚増えるごとにこれを加算するZobristKey
// 枚数ごとにhash keyのtableを用意するのは嫌なので、加算型にしてある。
Key hand[COLOR_NB][PIECE_HAND_NB];

#if defined(USE_PARTIAL_KEY)
// 歩の陣形に関して盤上に歩が一枚もない時のhash key
Key noPawns;
#endif

#endif
};

// ----------------------------------
//       Zorbrist keyの初期化
// ----------------------------------

void Position::init() {
	PRNG rng(20151225); // 開発開始日 == 電王トーナメント2015,最終日

	// 乱数で初期化するコード
	auto set_rand = [&](Key& h) {
        auto r1 = rng.rand<Key64>();
        auto r2 = rng.rand<Key64>();
        auto r3 = rng.rand<Key64>();
        auto r4 = rng.rand<Key64>();
        SET_HASH(h, r1, r2, r3, r4);
    };

	SET_HASH(Zobrist::zero, 0, 0, 0, 0);

	set_rand(Zobrist::side);

#if defined(USE_PARTIAL_KEY)
    set_rand(Zobrist::noPawns);
#endif

	// 64bit hash keyは256bit hash keyの下位64bitという解釈をすることで、256bitと64bitのときとでhash keyの下位64bitは合致するようにしておく。
	// これは定跡DBなどで使うときにこの性質が欲しいからである。
	// またpc==NO_PIECEのときは0であることを保証したいのでSET_HASHしない。
	// psqは、C++の規約上、事前にゼロであることは保証される。
	for (auto pc : Piece())
		for (auto sq : SQ)
			if (pc)
                set_rand(Zobrist::psq[pc][sq]);
	
	// またpr==NO_PIECEのときは0であることを保証したいのでSET_HASHしない。
	for (auto c : COLOR)
		for (PieceType pr = NO_PIECE_TYPE; pr < PIECE_HAND_NB; ++pr)
			if (pr)
                set_rand(Zobrist::hand[c][pr]);

}

// ----------------------------------
//  Partial Keyの更新のためのヘルパー
// ----------------------------------

namespace {
inline void xor_piece_for_partial_key(StateInfo* st, Piece pc, Square s) {
#if defined(USE_PARTIAL_KEY)
    /*
		🤔 手駒も含めたPARTIAL KEYにしたほうがいいかも知れないが、
			手駒は足し算にしているので、それに対応するのは容易ではない。
			盤上が同じで手駒違いの兄弟局面が現れることはレアケースなので
			気にしないことにする。

		📓 通常のhash keyのほうも、この関数内で更新したほうが一元化できて良いのだが、
		    通常のhash keyはdo_move()のなかでなるべく早いタイミングで確定させ、
			それを用いてprefetchしたいという意味があるので、このなかでやるわけにはいかない。
	*/
    if (type_of(pc) == PAWN)
    {
        st->pawnKey ^= Zobrist::psq[pc][s];
    }
    else
    {
        if (is_minor_piece(pc))
            st->minorPieceKey ^= Zobrist::psq[pc][s];

        st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];
    }
#endif
}

inline void put_piece_for_partial_key(StateInfo* st, Piece pc, Square s) {
#if defined(USE_PARTIAL_KEY)
    xor_piece_for_partial_key(st, pc, s);
    // 🌈 materialKeyは足し算にする。これで、pieceCount()が不要になる。
    st->materialKey += Zobrist::psq[pc][8];
#endif
}

inline void remove_piece_for_partial_key(StateInfo* st, Piece pc, Square s) {
#if defined(USE_PARTIAL_KEY)
    xor_piece_for_partial_key(st, pc, s);
    st->materialKey -= Zobrist::psq[pc][8];
#endif
}
} // namespace

// ----------------------------------
//  Position::set()とその逆変換sfen()
// ----------------------------------

// Pieceを綺麗に出力する(USI形式ではない) 先手の駒は大文字、後手の駒は小文字、成り駒は先頭に+がつく。盤面表示に使う。
#if !defined (PRETTY_JP)
std::string pretty(Piece pc) { return std::string(USI_PIECE).substr(pc * 2, 2); }
#else
// "□"(四角)は文字フォントによっては半分の幅しかない。"口"(くち)にする。
std::string USI_PIECE_KANJI[] = {
	" 口"," 歩"," 香"," 桂"," 銀"," 角"," 飛"," 金"," 玉"," と"," 杏"," 圭"," 全"," 馬"," 龍"," 菌"," 王",
		  "^歩","^香","^桂","^銀","^角","^飛","^金","^玉","^と","^杏","^圭","^全","^馬","^龍","^菌","^王"
};
std::string pretty(Piece pc) {
#if 1
	return USI_PIECE_KANJI[pc];
#else
	// 色を変えたほうがわかりやすい。Linuxは簡単だが、MS-DOSは設定が面倒。
	// Linux : https://qiita.com/dojineko/items/49aa30018bb721b0b4a9
	// MS-DOS : https://one-person.hatenablog.jp/entry/2017/02/23/125809

	std::string result;
	if (pc != NO_PIECE)
		result = (color_of(pc) == BLACK) ? "\\e[32;40;1m" : "\\e[33;40;1m";
	result += USI_PIECE_KANJI[pc];
	if (pc != NO_PIECE)
		result += "\\e[m";

	return result;
#endif
}
#endif



#if STOCKFISH
// Helper function used to set castling
// rights given the corresponding color and the rook starting square.
void Position::set_castling_right(Color c, Square rfrom) {

    Square         kfrom = square<KING>(c);
    CastlingRights cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}
#endif

// Sets king attacks to detect if a move gives check
// 移動によってチェック（王手）になるかを検出するために、kingへの利きを設定する
#if STOCKFISH
void Position::set_check_info() const {

    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq = square<KING>(~sideToMove);

    st->checkSquares[PAWN]   = attacks_bb<PAWN>(ksq, ~sideToMove);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq);
    st->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;
}
#else
template<bool doNullMove, Color Us>
void Position::set_check_info() const {

    // 🌈　やねうら王独自の改良
    //      null moveのときは前の局面でこの情報は設定されているので更新する必要がない。
    if (!doNullMove)
    {
        update_slider_blockers(WHITE);
        update_slider_blockers(BLACK);
    }

    constexpr Color Them = ~Us;

    Square ksq = square<KING>(Them);

    // 駒種Xによって敵玉に王手となる升のbitboard

    // 歩であれば、自玉に敵の歩を置いたときの利きにある場所に自分の歩があればそれは敵玉に対して王手になるので、
    // そういう意味で(ksq,them)となっている。

    Bitboard occ = pieces();

    // この指し手が二歩でないかは、この時点でテストしない。指し手生成で除外する。なるべくこの手のチェックは遅延させる。
    st->checkSquares[PAWN]   = pawnEffect<Them>(ksq);
    st->checkSquares[KNIGHT] = knightEffect<Them>(ksq);
    st->checkSquares[SILVER] = silverEffect<Them>(ksq);
    st->checkSquares[BISHOP] = bishopEffect(ksq, occ);
    st->checkSquares[ROOK]   = rookEffect(ksq, occ);
    st->checkSquares[GOLD]   = goldEffect<Them>(ksq);

    // 香で王手になる升は利きを求め直さずに飛車で王手になる升を香のstep effectでマスクしたものを使う。
    st->checkSquares[LANCE] = st->checkSquares[ROOK] & lanceStepEffect<Them>(ksq);

    // 王を移動させて直接王手になることはない。それは自殺手である。
    st->checkSquares[KING] = Bitboard(ZERO);

    // 成り駒。この初期化は馬鹿らしいようだが、gives_check()は指し手ごとに呼び出されるので、その処理を軽くしたいので
    // ここでの初期化は許容できる。(このコードはdo_move()に対して1回呼び出されるだけなので)
    st->checkSquares[PRO_PAWN]   = st->checkSquares[GOLD];
    st->checkSquares[PRO_LANCE]  = st->checkSquares[GOLD];
    st->checkSquares[PRO_KNIGHT] = st->checkSquares[GOLD];
    st->checkSquares[PRO_SILVER] = st->checkSquares[GOLD];
    st->checkSquares[HORSE]      = st->checkSquares[BISHOP] | kingEffect(ksq);
    st->checkSquares[DRAGON]     = st->checkSquares[ROOK] | kingEffect(ksq);
}
#endif


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up

// 局面のハッシュキーおよび、
// 一度計算すればその後は指し手に応じてインクリメンタルに更新される
// その他のデータを計算する。
// この関数は新しい局面をセットアップするときだけ使われる。

void Position::set_state() const {

#if STOCKFISH

    st->key = st->materialKey = 0;
    st->minorPieceKey         = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    set_check_info();

#else

    // 🌈 やねうら王では、st->keyはboard_keyとhand_keyに分かれる。
    st->board_key = Zobrist::zero;
    st->hand_key  = Zobrist::zero;

#if defined(USE_PARTIAL_KEY)

    st->materialKey       = Zobrist::zero;
    st->minorPieceKey     = Zobrist::zero;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = Zobrist::zero;
    st->pawnKey                                   = Zobrist::noPawns;

#endif
    // 歩以外の駒の価値。やねうら王では使っていない。
    // st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;

    // この局面で自玉に王手している敵駒
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    // 王手情報の初期化
    set_check_info<false>();
#endif


#if STOCKFISH
    for (Bitboard b = pieces(); b;)
    {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];

        if (type_of(pc) == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];

        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

            if (type_of(pc) != KING)
            {
                st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];

                // 📝 StockfishはKNIGHTとBISHOPをminor pieceとして扱っているっぽい。

                if (type_of(pc) <= BISHOP)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }
#else
    for (auto s : pieces())
    {
        auto pc = piece_on(s);

        st->board_key ^= Zobrist::psq[pc][s];

		// やねうら王では、partial keyの更新はこの関数に一元化されている。
        put_piece_for_partial_key(st, pc, s);
    }

    for (auto c : COLOR)
        for (PieceType pr = PAWN; pr < PIECE_HAND_NB; ++pr)
            st->hand_key +=
              Zobrist::hand[c][pr]
              * (int64_t) hand_count(hand[c], pr);  // 手駒はaddにする(差分計算が楽になるため)

    // --- hand

	st->hand = hand[sideToMove];

#endif

#if STOCKFISH
    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];

    for (Piece pc : Pieces)
        for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
            st->materialKey ^= Zobrist::psq[pc][8 + cnt];

#else

    // 🌈 将棋では、WHITEが後手番なので、WHITEのほうをZobrist::sideにしておく。
    if (sideToMove == WHITE)
        st->board_key ^= Zobrist::side;

	// st->materialKeyは、put_piece_for_partial_key()で更新済み。

#endif
}


// Initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.

// 指定されたFEN文字列でPositionオブジェクトを初期化します。
// この関数はあまり堅牢ではありません。入力されるFEN文字列が正しいことを確認してください。
// FENの正当性はGUI側の責任であると想定されています。

// sfen文字列で盤面を設定する
Position& Position::set(const std::string& sfen, StateInfo* si) {
#if STOCKFISH

	std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
#else

    // 🌈 やねうら王ではPositionがPODでない(BitboardやHASH_KEYがPODでない)ので
    //    コンパイル時にwarningが出るからvoid*にcastする。
    std::memset(static_cast<void*>(this), 0, sizeof(Position));

    // 局面をrootより遡るためには、ここまでの局面情報が必要で、それは引数のsiとして渡されているという解釈。
    // ThreadPool::start_thinking()では、
    // ここをいったんゼロクリアしたのちに、呼び出し側で、そのsiを復元することにより、局面を遡る。
    std::memset(static_cast<void*>(si), 0, sizeof(StateInfo));
#endif

    st = si;

    // 変な入力をされることはあまり想定していない。
    // sfen文字列は、普通GUI側から渡ってくるのでおかしい入力であることはありえないからである。

    // --- 盤面

    // 盤面左上から。Square型のレイアウトに依らずに処理を進めたいため、Square型は使わない。
    File f = FILE_9;
    Rank r = RANK_1;

    std::istringstream ss(sfen);
    // 盤面を読むときにスペースが盤面と手番とのセパレーターなのでそこを読み飛ばさないようにnoskipwsを指定しておく。
    ss >> std::noskipws;

    uint8_t token;
    bool    promote = false;
    size_t  idx;

#if defined(USE_EVAL_LIST)
    // evalListのclear。上でmemsetでゼロクリアしたときにクリアされているが…。
    evalList.clear();

    // PieceListを更新する上で、どの駒がどこにあるかを設定しなければならないが、
    // それぞれの駒をどこまで使ったかのカウンター
    PieceNumber piece_no_count[KING] = {
      PIECE_NUMBER_ZERO,   PIECE_NUMBER_PAWN,   PIECE_NUMBER_LANCE, PIECE_NUMBER_KNIGHT,
      PIECE_NUMBER_SILVER, PIECE_NUMBER_BISHOP, PIECE_NUMBER_ROOK,  PIECE_NUMBER_GOLD};

    // 先手玉のいない詰将棋とか、駒落ちに対応させるために、存在しない駒はすべてBONA_PIECE_ZEROにいることにする。
    // 上のevalList.clear()で、ゼロクリアしているので、それは達成しているはず。
#endif

    kingSquare[BLACK] = kingSquare[WHITE] = SQ_NB;

    // 1. Piece placement
    // 1. 駒の配置

    while ((ss >> token) && !isspace(token))
    {
        // 数字は、空の升の数なのでその分だけ筋(File)を進める
        if (isdigit(token))
            f -= File(token - '0');
        // '/'は次の段を意味する
        else if (token == '/')
        {
            f = FILE_9;
            ++r;
        }
        // '+'は次の駒が成駒であることを意味する
        else if (token == '+')
            promote = true;
        // 駒文字列か？
        else if ((idx = PieceToCharBW.find(token)) != string::npos)
        {
            // 盤面の(f,r)の駒を設定する
            auto sq = f | r;
            auto pc = promote ? make_promoted_piece(Piece(idx)) : Piece(idx);
            put_piece(pc, sq);

#if defined(USE_EVAL_LIST)
            PieceNumber piece_no = (idx == B_KING) ? PIECE_NUMBER_BKING :  // 先手玉
                                     (idx == W_KING) ? PIECE_NUMBER_WKING
                                                     :                           // 後手玉
                                     piece_no_count[raw_type_of(Piece(idx))]++;  // それ以外
            evalList.put_piece(piece_no, sq, pc);  // sqの升にpcの駒を配置する
#endif

            // 1升進める
            --f;

            // 成りフラグ、戻しておく。
            promote = false;
        }
    }

    // put_piece()を使ったので更新しておく。
    // set_state()で駒種別のbitboardを参照するのでそれまでにこの関数を呼び出す必要がある。
    update_bitboards();

    // kingSquare[]の更新
    update_kingSquare();

    // 2. Active color
    // 2. 手番

    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;  // 手番と手駒とを分かつスペース

    //    手駒

    hand[BLACK] = hand[WHITE] = (Hand) 0;
    int ct                    = 0;
    while ((ss >> token) && !isspace(token))
    {
        // 手駒なし
        if (token == '-')
            break;

        if (isdigit(token))
            // 駒の枚数。歩だと18枚とかあるので前の値を10倍して足していく。
            ct = (token - '0') + ct * 10;
        else if ((idx = PieceToCharBW.find(token)) != string::npos)
        {
            // 個数が省略されていれば1という扱いをする。
            ct = max(ct, 1);
            Piece pc = Piece(idx);
            PieceType rpc = raw_type_of(Piece(idx));

            // FV38などではこの個数分だけpieceListに突っ込まないといけない。
            for (int i = 0; i < ct; ++i)
            {
				// 手駒を1枚増やす。
                put_hand_piece(color_of(pc), rpc);

#if defined(USE_EVAL_LIST)
                PieceNumber piece_no = piece_no_count[rpc]++;
                ASSERT_LV1(is_ok(piece_no));
                evalList.put_piece(piece_no, color_of(pc), rpc, i);
#endif
            }
            ct = 0;
        }
    }

#if STOCKFISH
    // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
    // Shredder-FEN that uses the letters of the columns on which the rooks began
    // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
    // if an inner rook is associated with the castling right, the castling tag is
    // replaced by the file letter of the involved rook, as for the Shredder-FEN.

    while ((ss >> token) && !isspace(token))
    {
        Square rsq;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);

        token = char(toupper(token));

        if (token == 'K')
            for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq)
            {}

        else if (token == 'Q')
            for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq)
            {}

        else if (token >= 'A' && token <= 'H')
            rsq = make_square(File(token - 'A'), relative_rank(c, RANK_1));

        else
            continue;

        set_castling_right(c, rsq);
    }

    // 4. En passant square.
    // Ignore if square is invalid or not on side to move relative rank 6.
    bool enpassant = false;

    if (((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3'))))
    {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

        // En passant square will be considered only if
        // a) side to move have a pawn threatening epSquare
        // b) there is an enemy pawn in front of epSquare
        // c) there is no piece on epSquare or behind epSquare
        enpassant = attacks_bb<PAWN>(st->epSquare, ~sideToMove) & pieces(sideToMove, PAWN)
                 && (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)))
                 && !(pieces() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))));
    }

    if (!enpassant)
        st->epSquare = SQ_NONE;

#endif

    // 5-6. Halfmove clock and fullmove number
    // 5-6. 手数(平手の初期局面からの手数)

#if STOCKFISH
    // 📝 Stockfishの場合、rule50とgamePlyが渡される。
    //     このgamePlyは自分が指した回数なのでこれを2倍して、現在の手番(先手なら+0、後手なら+1)を
    //     足してやる必要があ。

    ss >> std::skipws >> st->rule50 >> gamePly;

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    chess960 = isChess960;

#else

    // gamePlyとして将棋所では(検討モードなどにおいて)ここで常に1が渡されている。

    // 検討モードにおいても棋譜上の手数を渡して欲しい気がするし、棋譜上の手数がないなら0を渡して欲しい気はする。
    // ここで渡されてきた局面をもとに探索してその指し手を定跡DBに登録しようとするときに、ここの手数が不正確であるのは困る。
    gamePly = 0;
    ss >> std::skipws >> gamePly;

#endif

    // --- StateInfoの更新

    set_state();

    // 現局面で王手がかかっているならst->continuous_check[them] = 1にしないと
    // 連続王手の千日手の判定が不正確な気がするが、どのみち2回目の出現を負け扱いしているのでまあいいか..

    // --- long effect

#if defined(LONG_EFFECT_LIBRARY)
    // 利きの全計算による更新
    LongEffect::calc_effect(*this);
#endif

    // --- evaluate

#if defined(USE_PIECE_VALUE)
    st->materialValue = Eval::material(*this);
#endif

#if defined(USE_CLASSIC_EVAL)
    Eval::compute_eval(*this);
#endif

    // --- validation

#if STOCKFISH
    assert(pos_is_ok());
#else

#if ASSERT_LV >= 3
    // これassertにしてしまうと、先手玉のいない局面や駒落ちの局面で落ちて困る。
    if (!is_ok(*this))
        std::cout << "info string Illigal Position?" << endl;
		// ⚠ UnitTestで駒落ちのテストをするので、そのときに引っかかるが…。
#endif
#endif

    return *this;
}

// Returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
// 局面のFEN表現を返します。Chess960の場合はShredder-FEN表記が使われます。
// これは主にデバッグ用の関数です。

// 📝 局面のsfen文字列を取得する。Position::set()の逆変換。

#if STOCKFISH

string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');

    if (can_castle(WHITE_OOO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');

    if (can_castle(BLACK_OO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');

    if (can_castle(BLACK_OOO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (ep_square() == SQ_NONE ? " - " : " " + UCIEngine::square(ep_square()) + " ")
       << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

#else

const std::string Position::sfen(int gamePly_) const
{
	std::ostringstream ss;

	// --- 盤面
	int emptyCnt;
	for (Rank r = RANK_1; r <= RANK_9; ++r)
	{
		for (File f = FILE_9; f >= FILE_1; --f)
		{
			// それぞれの升に対して駒がないなら
			// その段の、そのあとの駒のない升をカウントする
			for (emptyCnt = 0; f >= FILE_1 && piece_on(f | r) == NO_PIECE; --f)
				++emptyCnt;

			// 駒のなかった升の数を出力
			if (emptyCnt)
				ss << emptyCnt;

			// 駒があったなら、それに対応する駒文字列を出力
			if (f >= FILE_1)
				ss << (piece_on(f | r));
		}

		// 最下段以外では次の行があるのでセパレーターである'/'を出力する。
		if (r < RANK_9)
			ss << '/';
	}

	// --- 手番
	ss << (sideToMove == WHITE ? " w " : " b ");

	// --- 手駒(UCIプロトコルにはないがUSIプロトコルにはある)
	int n;
	bool found = false;
	for (Color c = BLACK; c <= WHITE; ++c)
		for (int pn = 0 ; pn < 7; ++ pn)
		{
			// 手駒の出力順はUSIプロトコルでは規定されていないが、
			// USI原案によると、飛、角、金、銀、桂、香、歩の順である。
			// sfen文字列を一意にしておかないと定跡データーをsfen文字列で書き出したときに
			// 他のソフトで文字列が一致しなくて困るので、この順に倣うことにする。

			const PieceType USI_Hand[7] = { ROOK,BISHOP,GOLD,SILVER,KNIGHT,LANCE,PAWN };
			auto p = USI_Hand[pn];

			// その種類の手駒の枚数
			n = hand_count(hand[c], p);
			// その種類の手駒を持っているか
			if (n != 0)
			{
				// 手駒が1枚でも見つかった
				found = true;

				// その種類の駒の枚数。1ならば出力を省略
				if (n != 1)
					ss << n;

				ss << PieceToCharBW[make_piece(c, p)];
			}
		}

	// 手駒がない場合はハイフンを出力
	if (!found)
		ss << '-';

	// --- 初期局面からの手数

	// ※　裏技 : gamePlyが負なら、sfen文字列末尾の手数を出力しない。
	if (gamePly_ >= 0)
		ss << ' ' << gamePly_;

	return ss.str();
}
#endif


// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.

// st->blockersForKing[c] と st->pinners[~c] を計算します。
// これらはそれぞれ、色 c のキングがチェックされるのを防いでいる駒、
// および、色 ~c のスライダー駒で、色 c の駒をキングに対してピンしている駒を格納します。

void Position::update_slider_blockers(Color c) const {

	Square ksq = square<KING>(c);

    st->blockersForKing[c] = ZERO;
    st->pinners[~c]        = ZERO;

    // Snipers are sliders that attack 's' when a piece and other snipers are removed
    // snipersとは、pinされている駒が取り除かれたときに王の升に利きが発生する大駒である。

#if STOCKFISH
    Bitboard snipers = (  (attacks_bb<  ROOK>(ksq) & pieces(QUEEN, ROOK))
                        | (attacks_bb<BISHOP>(ksq) & pieces(QUEEN, BISHOP))) & pieces(~c);
#else

	// cが与えられていないと香の利きの方向を確定させることが出来ない。
    // ゆえに将棋では、この関数は手番を引数に取るべき。(チェスとはこの点において異なる。)

    Bitboard snipers =
      ((pieces(ROOK_DRAGON) & rookStepEffect(ksq))
       | (pieces(BISHOP_HORSE) & bishopStepEffect(ksq))
       // 香に関しては先手玉へのsniperなら、玉より上側をサーチして、そこにある後手の香を探す必要がある。
       | (pieces(LANCE) & lanceStepEffect(c, ksq)))
      & pieces(~c);
#endif

    // snipersを取り除いた障害物(駒)
    Bitboard occupancy = pieces() ^ snipers;

    /*
		1.
			王 歩 ^角 ^飛
			のようなケースはない(王から見て斜め方向にいる角しか列挙していないのでsnipersのbitboardは王の横方向に角がいることはない。)

		2.
		    王 歩 ^飛 ^飛
			のようなケースにおいては、この両方の飛車がpinnersとして列挙されて欲しい。(SEEの処理でこういう列挙がなされて欲しいので)
	*/

    while (snipers)
    {
        Square   sniperSq = snipers.pop();
        Bitboard b        = between_bb(ksq, sniperSq) & occupancy;

        // snipperと玉との間にある駒が1個であるなら。
        if (b && !b.more_than_one())
        {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.

// sに利きのあるc側の駒を列挙する。先後両方。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)

// 🌈 sq == SQ_NBでの呼び出しは合法。この時、Bitboard(ZERO)が返る。

Bitboard Position::attackers_to(Square sq, const Bitboard& occ) const {
    // clang-format off

	ASSERT_LV3(sq <= SQ_NB);

    // sqの地点に敵駒ptをおいて、その利きに自駒のptがあればsqに利いているということだ。
    return
      // 先手の歩・桂・銀・金・HDK
      ((  (pawnEffect<WHITE>(sq)   & pieces(PAWN))
		| (knightEffect<WHITE>(sq) & pieces(KNIGHT))
        | (silverEffect<WHITE>(sq) & pieces(SILVER_HDK))
        | (goldEffect<WHITE>(sq)   & pieces(GOLDS_HDK)))
       & pieces(BLACK))
      |

      // 後手の歩・桂・銀・金・HDK
      (((pawnEffect<BLACK>(sq)     & pieces(PAWN))
		| (knightEffect<BLACK>(sq) & pieces(KNIGHT))
        | (silverEffect<BLACK>(sq) & pieces(SILVER_HDK))
        | (goldEffect<BLACK>(sq)   & pieces(GOLDS_HDK)))
       & pieces(WHITE))

      // 先後の角・飛・香
      | (bishopEffect(sq, occ)     & pieces(BISHOP_HORSE))
      | (rookEffect(sq, occ)       & (pieces(ROOK_DRAGON) | (pieces(BLACK, LANCE) & lanceStepEffect<WHITE>(sq))
      | (pieces(WHITE, LANCE)      & lanceStepEffect<BLACK>(sq))
        // 香も、StepEffectでマスクしたあと飛車の利きを使ったほうが香の利きを求めなくて済んで速い。
    ));

	// clang-format on

}

#if STOCKFISH

// castlingの判定のために使う。

bool Position::attackers_to_exist(Square s, Bitboard occupied, Color c) const {

    return ((attacks_bb<ROOK>(s) & pieces(c, ROOK, QUEEN))
            && (attacks_bb<ROOK>(s, occupied) & pieces(c, ROOK, QUEEN)))
        || ((attacks_bb<BISHOP>(s) & pieces(c, BISHOP, QUEEN))
            && (attacks_bb<BISHOP>(s, occupied) & pieces(c, BISHOP, QUEEN)))
        || (((attacks_bb<PAWN>(s, ~c) & pieces(PAWN)) | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT))
             | (attacks_bb<KING>(s) & pieces(KING)))
            & pieces(c));
}
#endif


// 🚧



// 盤面を先後反転させた時のsfen文字列を取得する。
const std::string Position::flipped_sfen(int gamePly_) const
{
	std::ostringstream ss;

	// --- 盤面
	int emptyCnt;
	for (Rank r = RANK_9; r >= RANK_1; --r)
	{
		for (File f = FILE_1; f <= FILE_9; ++f)
		{
			// それぞれの升に対して駒がないなら
			// その段の、そのあとの駒のない升をカウントする
			for (emptyCnt = 0; f <= FILE_9 && piece_on(f | r) == NO_PIECE; ++f)
				++emptyCnt;

			// 駒のなかった升の数を出力
			if (emptyCnt)
				ss << emptyCnt;

			// 駒があったなら、それに対応する駒文字列を出力
			if (f <= FILE_9)
				// ※　flippedなのでこの駒、先後逆にしないといけないので PIECE_WHITEのbitを反転させる。
				ss << Piece(piece_on(f | r) ^ PIECE_WHITE);
		}

		// 最下段以外では次の行があるのでセパレーターである'/'を出力する。
		if (r > RANK_1)
			ss << '/';
	}

	// --- 手番
	// ※　flippedなのでsideToMoveの逆を出力
	ss << (~sideToMove == WHITE ? " w " : " b ");

	// --- 手駒(UCIプロトコルにはないがUSIプロトコルにはある)
	int n;
	bool found = false;
	for (Color c = BLACK; c <= WHITE; ++c)
		for (int pn = 0 ; pn < 7; ++ pn)
		{
			// 手駒の出力順はUSIプロトコルでは規定されていないが、
			// USI原案によると、飛、角、金、銀、桂、香、歩の順である。
			// sfen文字列を一意にしておかないと定跡データーをsfen文字列で書き出したときに
			// 他のソフトで文字列が一致しなくて困るので、この順に倣うことにする。

			const PieceType USI_Hand[7] = { ROOK,BISHOP,GOLD,SILVER,KNIGHT,LANCE,PAWN };
			auto p = USI_Hand[pn];

			// その種類の手駒の枚数
			// ※ flippedなので、ここをcではなく~c側を見ればflipしたことになる。
			n = hand_count(hand[~c], p);
			// その種類の手駒を持っているか
			if (n != 0)
			{
				// 手駒が1枚でも見つかった
				found = true;

				// その種類の駒の枚数。1ならば出力を省略
				if (n != 1)
					ss << n;

				ss << PieceToCharBW[make_piece(c, p)];
			}
		}

	// 手駒がない場合はハイフンを出力
	if (!found)
		ss << '-';

	// --- 初期局面からの手数

	// ※　裏技 : gamePlyが負なら、sfen文字列末尾の手数を出力しない。
	if (gamePly_ >= 0)
		ss << ' ' << gamePly_;

	return ss.str();
}

// sfen文字列をflip(先後反転)したsfen文字列に変換する。
const std::string Position::sfen_to_flipped_sfen(std::string sfen)
{
#if 1
	Position pos;
	StateInfo si;
	pos.set(sfen, &si);
	return pos.flipped_sfen();
#else
	// この局面クラスを利用せず文字列操作だけで求めて返す。
	// https://yaneuraou.yaneu.com/2023/12/15/chatgpt-wrote-a-program-to-flip-a-shogi-board/

	// 文字列操作だけで書く。あとで書くかも。
#endif
}


// put_piece(),remove_piece(),xor_piece()を用いたあとに呼び出す必要がある。
// これらは指し手生成や1手詰め判定の時に用いる。
void Position::update_bitboards()
{
	// 王・馬・龍を合成したbitboard
	byTypeBB[HDK]			= pieces(KING , HORSE , DRAGON);

	// 金と同じ移動特性を持つ駒
	byTypeBB[GOLDS]			= pieces(GOLD , PRO_PAWN , PRO_LANCE , PRO_KNIGHT , PRO_SILVER);

	// 以下、attackers_to()で頻繁に用いるのでここで1回計算しておいても、トータルでは高速化する。

	// 角と馬
	byTypeBB[BISHOP_HORSE]	= pieces(BISHOP , HORSE);

	// 飛車と龍
	byTypeBB[ROOK_DRAGON]	= pieces(ROOK   , DRAGON);

	// 銀とHDK
	byTypeBB[SILVER_HDK]	= pieces(SILVER , HDK);

	// 金相当の駒とHDK
	byTypeBB[GOLDS_HDK]		= pieces(GOLDS  , HDK);
}

// このクラスが保持しているkingSquare[]の更新。
void Position::update_kingSquare()
{
	for (auto c : COLOR)
	{
		// 玉がいなければSQ_NBを設定しておいてやる。(片玉対応)
		auto b = pieces(c, KING);
		kingSquare[c] = b ? b.pop() : SQ_NB;
	}
}

// ----------------------------------
//           Positionの表示
// ----------------------------------

// 盤面を出力する。(USI形式ではない) デバッグ用。
std::ostream& operator<<(std::ostream& os, const Position& pos)
{
	// 盤面
	for (Rank r = RANK_1; r <= RANK_9; ++r)
	{
		for (File f = FILE_9; f >= FILE_1; --f)
			os << pretty(pos.board[f | r]);
		os << endl;
	}

#if !defined (PRETTY_JP)
	// 手駒
	os << "BLACK HAND : " << pos.hand[BLACK] << " , WHITE HAND : " << pos.hand[WHITE] << endl;

	// 手番
	os << "Turn = " << pos.sideToMove << endl;
#else
	os << "先手 手駒 : " << pos.hand[BLACK] << " , 後手 手駒 : " << pos.hand[WHITE] << endl;
	os << "手番 = " << pos.sideToMove << endl;
#endif

	// sfen文字列もおまけで表示させておく。(デバッグのときに便利)
	os << "sfen " << pos.sfen() << endl;

	return os;
}

#if defined (KEEP_LAST_MOVE)

// 開始局面からこの局面にいたるまでの指し手を表示する。
std::string Position::moves_from_start(bool is_pretty) const
{
	StateInfo* p = st;
	std::stack<StateInfo*> states;
	while (p->previous != nullptr)
	{
		states.push(p);
		p = p->previous;
	}

	stringstream ss;
	while (states.size())
	{
		auto& top = states.top();
		if (is_pretty)
			ss << pretty(top->lastMove, top->lastMovedPieceType) << ' ';
		else
			ss << top->lastMove << ' ';
		states.pop();
	}
	return ss.str();
}
#endif




// 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
inline Bitboard Position::attackers_to_pawn(Color c, Square pawn_sq) const
{
	ASSERT_LV3(is_ok(c) && pawn_sq <= SQ_NB);

	Color them = ~c;
	const Bitboard& occ = pieces();

	// 馬と龍
	const Bitboard bb_hd = /* kingEffect(pawn_sq) & */ pieces(HORSE,DRAGON);
	// 馬、龍の利きは考慮しないといけない。しかしここに玉が含まれるので玉は取り除く必要がある。
	// bb_hdは銀と金のところに加えてしまうことでテーブル参照を一回減らす。

	// sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
	// 打ち歩詰め判定なので、その打たれた歩を歩、香、王で取れることはない。(王で取れないことは事前にチェック済)
	return
		(     (knightEffect(them, pawn_sq) &  pieces(KNIGHT)          )
			| (silverEffect(them, pawn_sq) & (pieces(SILVER) | bb_hd) )
			| (goldEffect(them, pawn_sq)   & (pieces(GOLDS)  | bb_hd) )
			| (bishopEffect(pawn_sq, occ)  &  pieces(BISHOP_HORSE)    )
			| (rookEffect(pawn_sq, occ)    &  pieces(ROOK_DRAGON)     )
			) & pieces(c);
}

// 指し手が、(敵玉に)王手になるかをテストする。

bool Position::gives_check(Move m) const
{
	// 指し手がおかしくないか
	ASSERT_LV3(m.is_ok());

	// 移動先
	const Square to = m.to_sq();

	// 駒打ち・移動する指し手どちらであってもmove_piece_after(m)で移動後の駒が取得できるので
	// 直接王手の処理は共通化できる。
	if (st->checkSquares[type_of(moved_piece_after(m))] & to)
		return true;

	// -- 移動する指し手ならば、これで開き王手になるかどうかの判定が必要。

	// 移動元
	const Square from = m.from_sq();

	// 開き王手になる駒の候補があるとして、fromにあるのがその駒で、fromからtoは玉と直線上にないなら
	// 前提条件より、fromにあるのが自駒であることは確定しているので、pieces(sideToMove)は不要。
	return  !m.is_drop()
		&& (((blockers_for_king(~sideToMove) /*& pieces(sideToMove)*/) & from)
		&&  !aligned(from, to, square<KING>(~sideToMove)));
}

// 現局面で指し手がないかをテストする。指し手生成ルーチンを用いるので速くない。探索中には使わないこと。
bool Position::is_mated() const
{
	// 不成で詰めろを回避できるパターンはないのでLEGAL_ALLである必要はない。
	return MoveList<LEGAL>(*this).size() == 0;
}

// ----------------------------------
//      指し手の合法性のテスト
// ----------------------------------

bool Position::legal_drop(const Square to) const
{
	const auto us = sideToMove;

	// 打とうとする歩の利きに相手玉がいることは前提条件としてクリアしているはず。
	ASSERT_LV3(pawnEffect(us, to) == Bitboard(square<KING>(~us)));

	// この歩に利いている自駒(歩を打つほうの駒)がなければ詰みには程遠いのでtrue
	if (!effected_to(us, to))
		return true;

	// ここに利いている敵の駒があり、その駒で取れるなら打ち歩詰めではない
	// ここでは玉は除外されるし、香が利いていることもないし、そういう意味では、特化した関数が必要。
	Bitboard b = attackers_to_pawn(~us, to);

	// 敵玉に対するpinしている駒(自駒も含むが、bが敵駒なので問題ない。)
	Bitboard pinned = blockers_for_king(~us);

	// pinされていない駒が1つでもあるなら、相手はその駒で取って何事もない。
	if (b & (~pinned | file_bb(file_of(to))))
		return true;

	// 攻撃駒はすべてpinされていたということであり、
	// 王の頭に打たれた打ち歩をpinされている駒で取れるケースは、
	// いろいろあるが、例1),例2)のような場合であるから、例3)のケースを除き、
	// いずれも玉の頭方向以外のところからの玉頭方向への移動であるから、
	// pinされている方向への移動ということはありえない。
	// 例3)のケースを除くと、この歩は取れないことが確定する。
	// 例3)のケースを除外するために同じ筋のものはpinされていないものとして扱う。
	//    上のコードの　 " | FILE_BB[file_of(to)] " の部分がそれ。

	// 例1)
	// ^玉 ^角  飛
	//  歩

	// 例2)
	// ^玉
	//  歩 ^飛
	//          角

	// 例3)
	// ^玉
	//  歩
	// ^飛
	//  香

	// 玉の退路を探す
	// 自駒がなくて、かつ、to(はすでに調べたので)以外の地点

	// 相手玉の場所
	Square sq_king = square<KING>(~us);

#if !defined(LONG_EFFECT_LIBRARY)
	// LONG EFFECT LIBRARYがない場合、愚直に8方向のうち逃げられそうな場所を探すしかない。

	Bitboard escape_bb = kingEffect(sq_king) & ~pieces(~us);
	escape_bb ^= to;
	auto occ = pieces() ^ to; // toには歩をおく前提なので、ここには駒があるものとして、これでの利きの遮断は考えないといけない。
	while (escape_bb)
	{
		Square king_to = escape_bb.pop();
		if (!attackers_to(us, king_to, occ))
			return true; // 退路が見つかったので打ち歩詰めではない。
	}

	// すべての検査を抜けてきたのでこれは打ち歩詰めの条件を満たしている。
	return false;
#else
	// LONG EFFECT LIBRARYがあれば、玉の8近傍の利きなどを直列化して逃げ場所があるか調べるだけで良いはず。

	auto a8_effet_us = board_effect[us].around8(sq_king);
	// 玉の逃げ場所は、本来はEffect8::around8(escape_bb,sq_king)なのだが、どうせaround8が8近傍だけを直列化するので、
	// これが玉の利きと一致しているからこのkingEffect(sq_king)でマスクする必要がない。
	auto a8_them_movable = Effect8::around8(~pieces(~us), sq_king) & Effect8::board_mask(sq_king);

	// 打った歩での遮断を考える前の段階ですでにすでに歩を打つ側の利きがない升があり、
	// そこに移動できるのであれば、これは打ち歩ではない。
	if (~a8_effet_us & a8_them_movable)
		return true;

	// 困ったことに行けそうな升がなかったので打った歩による利きの遮断を考える。
	// いまから打つ歩による遮断される升の利きが2以上でなければそこに逃げられるはず。
	auto a8_long_effect_to = long_effect.directions_of(us, to);
	auto to_dir = (us == BLACK) ? DIRECT_D : DIRECT_U;  // 王から見た歩の方角
	auto a8_cutoff_dir = Effect8::cutoff_directions(to_dir, a8_long_effect_to);
	auto a8_target = a8_cutoff_dir & a8_them_movable & ~board_effect[us].around8_greater_than_one(sq_king);

	return a8_target != 0;
#endif
}

// 二歩でなく、かつ打ち歩詰めでないならtrueを返す。
bool Position::legal_pawn_drop(const Color us, const Square to) const
{
	return !((pieces(us, PAWN) & file_bb(file_of(to)))                               // 二歩
		|| ((pawnEffect(us, to) == Bitboard(square<KING>(~us)) && !legal_drop(to)))); // 打ち歩詰め
}

// mがpseudo_legalな指し手であるかを判定する。
// ※　pseudo_legalとは、擬似合法手(自殺手が含まれていて良い)
// 置換表の指し手でdo_move()して良いのかの事前判定のために使われる。
// 指し手生成ルーチンのテストなどにも使える。(指し手生成ルーチンはpseudo_legalな指し手を返すはずなので)
// killerのような兄弟局面の指し手がこの局面において合法かどうかにも使う。
// ※　置換表の検査だが、pseudo_legal()で擬似合法手かどうかを判定したあとlegal()で自殺手でないことを
// 確認しなくてはならない。このためpseudo_legal()とlegal()とで重複する自殺手チェックはしていない。
//
//
// is_ok(m)==falseの時、すなわち、m == Move::win()やMove::none()のような時に
// Position::to_move(m) == mは保証されており、この時、本関数pseudo_legal(m)がfalseを返すことは保証する。
//
// generate_all_legal_moves : これがtrueならば、歩の不成も合法手扱い。
bool Position::pseudo_legal(const Move m, bool generate_all_legal_moves) const {
	return generate_all_legal_moves ? pseudo_legal_s<true>(m) : pseudo_legal_s<false>(m);
}

// ※　mがこの局面においてpseudo_legalかどうかを判定するための関数。
template <bool All>
bool Position::pseudo_legal_s(const Move m) const {

	const Color  us = sideToMove;
	const Square to = m.to_sq(); // 移動先

	if (m.is_drop())
	{
		const PieceType pr = m.move_dropped_piece();
		// 置換表から取り出してきている以上、一度は指し手生成ルーチンで生成した指し手のはずであり、
		// KING打ちのような値であることはないと仮定できる。

		// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
		if (moved_piece_after(m) != make_piece(us, pr))
			return false;

		ASSERT_LV3(PAWN <= pr && pr < KING);

		// 打つ先の升が埋まっていたり、その手駒を持っていなかったりしたら駄目。
		if (piece_on(to) != NO_PIECE || hand_count(hand[us], pr) == 0)
			return false;

		if (in_check())
		{
			// 王手されている局面なので合駒でなければならない
			Bitboard target = checkers();
			Square checksq = target.pop();

			// 王手している駒を1個取り除いて、もうひとつあるということは王手している駒が
			// 2つあったということであり、両王手なので合い利かず。
			if (target)
				return false;

			// 王と王手している駒との間の升に駒を打っていない場合、それは王手を回避していることに
			// ならないので、これは非合法手。
			if (!(between_bb(checksq, square<KING>(us)) & to))
				return false;
		}

		// 歩のとき、二歩および打ち歩詰めであるなら非合法手
		if (pr == PAWN && !legal_pawn_drop(us, to))
			return false;

		// --- 移動できない升への歩・香・桂打ちについて

		// 打てない段に打つ歩・香・桂の指し手はそもそも生成されていない。
		// 置換表のhash衝突で、後手の指し手が先手の指し手にならないことは保証されている。
		// (先手の手番の局面と後手の手番の局面とのhash keyはbit0で区別しているので)

		// しかし、Counter Moveの手は手番に関係ないので(駒種を保持していないなら)取り違える可能性があるため
		// (しかも、その可能性はそこそこ高い)、ここで合法性をチェックする必要がある。
		// →　指し手生成の段階で駒種を保存するようにしたのでこのテスト不要。

	}
	else {

		const Square from = m.from_sq();
		const Piece pc    = piece_on(from);

		// 動かす駒が自駒でなければならない
		if (pc == NO_PIECE || color_of(pc) != us)
			return false;

		// toに移動できないといけない。
		if (!(effects_from(pc, from, pieces()) & to))
			return false;

		// toの地点に自駒があるといけない
		if (pieces(us) & to)
			return false;

		PieceType pt = type_of(pc);
		if (m.is_promote())
		{
			// --- 成る指し手

			// 成れない駒の成りではないことを確かめないといけない。
			if (is_non_promotable_piece(pc))
				return false;

			// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
			if (moved_piece_after(m) != make_promoted_piece(pc))
				return false;
		}
		else {

			// --- 成らない指し手
			// ※　MOVE_WINやMOVE_NULLに対してfalseが返ることを保証しなければならないので以下の実装時に注意すること。

			// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
			if (moved_piece_after(m) != pc)
				return false;

			// is_ok(m) == falseな指し手、具体的にはMOVE_WINやMOVE_NULLに対して本関数がfalseを返すことを保証する。
			// 
			// is_ok(m) == falseの時、Position::to_move(m)はそのままmを返すので、この時、
			// moved_piece_after(m) == NO_PIECEであるが、from_sq(m)のPieceがたまたまNO_PIECEであった場合、
			// ここまで抜けてくるのでその場合、falseを返すことを保証しなければならない。
			if (pt == NO_PIECE_TYPE)
				return false;

			// 駒打ちのところに書いた理由により、不成で進めない升への指し手のチェックも不要。
			// 間違い　→　駒種をmoveに含めていないならこのチェック必要だわ。
			// 52から51銀のような指し手がkillerやcountermoveに登録されていたとして、52に歩があると
			// 51歩不成という指し手を生成してしまう…。
			// あと、歩や大駒が敵陣において成らない指し手も不要なのでは..。

			if (All)
			{
				// 歩と香に関しては1段目への不成は不可。
				// 桂は、桂飛びが出来る駒は桂しかないので
				// 移動元と移動先がこれであるかぎり、それは桂の指し手生成によって生成されたものだから
				// これが非合法手であることはない。

				if (pt == PAWN || pt == LANCE)
					if ((us == BLACK && rank_of(to) == RANK_1) || (us == WHITE && rank_of(to) == RANK_9))
						return false;
			}
			else {

				// 歩の不成と香の2段目への不成を禁止。
				// 大駒の不成を禁止
				switch (pt)
				{
				case PAWN:
					if (enemy_field(us) & to)
						return false;
					break;

				case LANCE:
					if ((us == BLACK && rank_of(to) <= RANK_2) || (us == WHITE && rank_of(to) >= RANK_8))
						return false;
					break;

				case BISHOP:
				case ROOK:
					if (enemy_field(us) & (Bitboard(from) | Bitboard(to)))
						return false;
					break;

				default:
					break;
				}
			}

		}

		// 王手している駒があるのか
		if (checkers())
		{
			// このとき、指し手生成のEVASIONで生成される指し手と同等以上の条件でなければならない。

			// 動かす駒は王以外か？
			if (type_of(pc) != KING)
			{
				// 両王手なら王の移動をさせなければならない。
				if (checkers().more_than_one())
					return false;

				// 指し手は、王手を遮断しているか、王手している駒の捕獲でなければならない。
				// ※　王手している駒と王の間に王手している駒の升を足した升が駒の移動先であるか。
				// 例) 王■■■^飛
				// となっているときに■の升か、^飛 のところが移動先であれば王手は回避できている。
				// (素抜きになる可能性はあるが、そのチェックはここでは不要)
				if (!((between_bb(checkers().pop(), square<KING>(us)) | checkers()) & to))
					return false;
			}

			// 玉の自殺手のチェックはlegal()のほうで調べているのでここではやらない。

		}
	}

	return true;
}

// 生成した指し手(CAPTUREとかNON_CAPTUREとか)が、合法であるかどうかをテストする。
bool Position::legal(Move m) const
{
	if (m.is_drop())
		// 打ち歩詰めは指し手生成で除外されている。
		return true;
	else
	{
		Color us    = sideToMove;
		Square from = m.from_sq();

		ASSERT_LV5(color_of(piece_on(m.from_sq())) == us);
		ASSERT_LV5(piece_on(square<KING>(us)) == make_piece(us, KING));

		// もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
		if (type_of(piece_on(from)) == KING)
			return !effected_to(~us, m.to_sq(), from);

		// blockers_for_king()は、pinされている駒(自駒・敵駒)を表現するが、fromにある駒は自駒であることは
		// わかっているのでこれで良い。
		return !(blockers_for_king(us) & from)
			 || aligned(from, m.to_sq(), square<KING>(us));
	}
}

// leagl()では、成れるかどうかのチェックをしていない。
// (先手の指し手を後手の指し手と混同しない限り、指し手生成された段階で
// 成れるという条件は満たしているはずだから)
// しかし、先手の指し手を後手の指し手と取り違えた場合、この前提が崩れるので
// これをチェックするための関数。成れる条件を満たしていない場合、falseが返る。
bool Position::legal_promote(Move m) const
{
	// 成りの指し手にしか関与しない
	if (!m.is_promote())
		return true;

	Color us = sideToMove;
	Square from = m.from_sq();
	Square to   = m.to_sq();

	// 移動元か移動先が敵陣でなければ成れる条件を満たしていない。
	return enemy_field(us) & (Bitboard(from) | Bitboard(to));
}

// 置換表から取り出したMoveを32bit化する。
Move Position::to_move(Move16 m16) const
{
	//		ASSERT_LV3(is_ok(m));
	// 置換表から取り出した値なので m==Move::none()である可能性があり、ASSERTは書けない。

	// 上位16bitは0でなければならない
	//      ASSERT_LV3((m >> 16) == 0);

	Move m = (Move)m16.to_u16();

	// MOVE_NULLの可能性はないはずだが、MOVE_WINである可能性はある。
	// それはそのまま返す。(MOVE_WINの機会はごくわずかなのでこれのために
	// このチェックが探索時に起きるのは少し馬鹿らしい気もする。
	// どうせ探索時はlegalityのチェックに引っかかり無視されるわけで…)
	if (!m.is_ok())
		return m;

	if (m.is_drop())
		return Move(m.to_u16() + (u32(make_piece(side_to_move(), m.move_dropped_piece())) << 16));
		// また、move_dropped_piece()はおかしい値になっていないことは保証されている(置換表に自分で書き出した値のため)
		// これにより、配列境界の外側に書き出してしまう心配はない。

	// 移動元にある駒が、現在の手番の駒であることを保証する。
	// 現在の手番の駒でないか、駒がなければMOVE_NONEを返す。
	Piece moved_piece = piece_on(m.from_sq());
	if (color_of(moved_piece) != side_to_move() || moved_piece == NO_PIECE)
		return Move::none();

	// promoteで成ろうとしている駒は成れる駒であることを保証する。
	if (m.is_promote())
	{
		// 成駒や金・玉であるなら、これ以上成れない。これは非合法手である。
		if (is_non_promotable_piece(moved_piece))
			return Move::none();

		return Move(m.to_u16() + (u32(make_promoted_piece(moved_piece)) << 16));
	}

	// 通常の移動
	return Move(m.to_u16() + (u32(moved_piece) << 16));
}


// ----------------------------------
//      局面を進める/戻す
// ----------------------------------

// 指し手で盤面を1手進める。
// ⚠ m として Move::none()はもちろん、Move::null() , Move::resign()などお断り。
template<Color Us, typename T>
void Position::do_move_impl(Move m, StateInfo& newSt, bool givesCheck, const T* tt) {

    ASSERT_LV3(m.is_ok());
    ASSERT_LV3(&newSt != st);

    // ----------------------
    //  StateInfoの更新
    // ----------------------

    // 現在の局面のhash keyはこれで、これを更新していき、
    // 次の局面のhash keyを求めてStateInfo::key_に格納。
#if STOCKFISH
    Key k = st->key ^ Zobrist::side;
#else
    Key k = st->board_key ^ Zobrist::side;

    // 🌈 将棋だと手駒がある。手駒用のhash keyを別途用意
    Key h = st->hand_key;
#endif

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.

    // 古い状態の一部のフィールドを新しいStateInfoオブジェクトにコピーする。
    // ただし、どうせ最初から再計算されるフィールドは除外する。
    // そして、stateポインタを新しい（これから更新される）状態を指すように切り替える。

    /* 📓 StateInfoのmemcpy()について

		StateInfoの構造体のメンバーの上からkeyのところまでは前のを丸ごとコピーしておく。
		こうしたほうが、in-placeで書き換えができるのでプログラムがすっきりする。

		undo_moveで戻すときにStateInfoはundo処理が要らないので(stack上にあるので自動的に破棄される)
		細かい更新処理が必要なものはここに載せておけばundoが速くなる。
	*/

#if STOCKFISH
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
#else
    std::memcpy(static_cast<void*>(&newSt), st, offsetof(StateInfo, board_key));
#endif
    newSt.previous = st;
    st             = &newSt;

    // --- 手数がらみのカウンターのインクリメント

    // Increment ply counters. In particular, rule50 will be reset to zero later on
    // in case of a capture or a pawn move.

    // 手数カウンタをインクリメントする。
    // 特に、キャプチャやポーンの手の場合は、後でrule50が0にリセットされる。

    ++gamePly;

#if STOCKFISH
    ++st->rule50;
#endif
    ++st->pliesFromNull;

#if STOCKFISH
    Color us   = sideToMove;
    Color them = ~us;
#else
    // 🌈 やねうら王では手番がtemplate引数になっている。
    constexpr Color us   = Us;
    constexpr Color them = ~Us;
#endif


    // 評価値の差分計算用の初期化
#if defined(USE_CLASSIC_EVAL)

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
    st->sum.p[0][0] = VALUE_NOT_EVALUATED;
#endif
#if defined(EVAL_NNUE)
    st->accumulator.computed_accumulation = false;
    st->accumulator.computed_score        = false;
#endif

#if defined(USE_BOARD_EFFECT_PREV)
    // NNUE-HalfKPE9
    // 現局面のboard_effectをコピー
    std::memcpy(board_effect_prev, board_effect, sizeof(board_effect));
#endif

    // 直前の指し手を保存するならばここで行なう。

#if defined(KEEP_LAST_MOVE)
    st->lastMove           = m;
    st->lastMovedPieceType = m.is_drop() ? PieceType(m.from_sq()) : type_of(piece_on(m.from_sq()));
#endif

#endif

    // ----------------------
    //    盤面の更新処理
    // ----------------------

	/*
		📓
			1. drop(駒打ち)
			2. 通常移動
				a. capture(移動先で駒を取る場合)
				b. promote(移動させる駒を成る場合)

				a. b. は独立。
				promoteしなければ、通常移動。

			以下のような処理フローになる。

			if (drop) {
				remove_hand_piece(us, pr);
				put_piece(to, pc);
			} else {
				if (captured) {
					remove_piece(to);
				}
				if (promote) {
					remove_piece(from);
					put_piece(to, promoted_pc);
				} else {
					move_piece(from,to);
				}
			}

			🤔 promoteのところを分けるの面倒くさいな…。
	*/


#if STOCKFISH
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);
#else
    // 将棋だと手駒から打てるのでこの時点ではfromとpcは確定できない。

    // 移動先の升
    Square to = m.to_sq();
    ASSERT_LV2(is_ok(to));

    // 捕獲される駒
    Piece captured = piece_on(to);

    // 玉を取る指し手が実現することはない。この直前の局面で玉を逃げる指し手しか合法手ではないし、
    // 玉を逃げる指し手がないのだとしたら、それは詰みの局面であるから。

    ASSERT_LV3(type_of(captured) != KING);

#endif

#if defined(USE_PIECE_VALUE)
    // 駒割りの差分計算用
    int materialDiff;
#endif

#if defined(USE_CLASSIC_EVAL)

#if defined(USE_EVAL_LIST)
    auto& dp = st->dirtyPiece;
#endif
#endif

    if (m.is_drop())
    {
        // --- 駒打ち

        // 移動先の升は空のはず
        ASSERT_LV3(empty(to));

        Piece     pc = moved_piece_after(m);
        PieceType pr = raw_type_of(pc);
        ASSERT_LV3(PAWN <= pr && pr < PIECE_HAND_NB);

        // Zobrist keyの更新
        h -= Zobrist::hand[Us][pr];
        k ^= Zobrist::psq[pc][to];

        // なるべく早い段階でのTTに対するprefetch
        // 駒打ちのときはこの時点でTT entryのアドレスが確定できる
        if constexpr (std::is_same_v<T, TranspositionTable>)
        {
            const auto key = k ^ h;
            prefetch(tt->first_entry(key, them));
        }

#if defined(USE_PARTIAL_KEY) & 0
        // 打ち歩なら、pawnKeyの更新が必要
        if (pr == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][to];
#endif

        put_piece(pc, to);
        put_piece_for_partial_key(st, pc, to);

        // 打駒した駒に関するevalListの更新。
#if defined(USE_EVAL_LIST)
        PieceNumber piece_no = piece_no_of(Us, pr);
        ASSERT_LV3(is_ok(piece_no));

        // KPPの差分計算のために移動した駒をStateInfoに記録しておく。
        dp.dirty_num                  = 1;  // 動いた駒は1個
        dp.pieceNo[0]                 = piece_no;
        dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no);
        evalList.put_piece(piece_no, to, pc);
        dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no);
#endif

        // ⚠ piece_no_of()のときに、いまの手駒の枚数を参照するので↑のあとで更新する必要がある。
        remove_hand_piece(Us, pr);

        // 王手している駒のbitboardを更新する。
        // 駒打ちなのでこの駒で王手になったに違いない。
		// 駒打ちで両王手はありえないので王手している駒はいまtoに置いた駒のみ。
        if (givesCheck)
        {
            st->checkersBB = Bitboard(to);
            st->continuousCheck[Us] += 2;
        }
        else
        {
            st->checkersBB          = Bitboard(ZERO);
            st->continuousCheck[Us] = 0;
        }

        // 駒打ちは捕獲した駒がない。
        st->capturedPiece = NO_PIECE;

        // put_piece()などを用いたのでupdateする
        update_bitboards();

#if defined(USE_PIECE_VALUE)
        // 駒打ちなので駒割りの変動なし。
        materialDiff = 0;
#endif

#if defined(LONG_EFFECT_LIBRARY)
        // 駒打ちによる利きの更新処理
        LongEffect::update_by_dropping_piece<Us>(*this, to, pc);
#endif
    }
    else
    {
        // -- 駒の移動

		Square from = m.from_sq();
        ASSERT_LV2(is_ok(from));

        // 移動させる駒
        Piece moved_pc = piece_on(from);
        ASSERT_LV2(moved_pc != NO_PIECE);

        // 移動先に駒の配置
        // もし成る指し手であるなら、成った後の駒を配置する。
        Piece moved_after_pc = moved_piece_after(m);

#if defined(USE_PIECE_VALUE)
        materialDiff = m.is_promote() ? Eval::ProDiffPieceValue[moved_pc] : 0;
#endif
        // 📌 ここから下はStockfishのdo_move()のコードの一部 📌

		// 駒を取るのか？
        if (captured)
        {
            // --- capture(駒の捕獲)

			// 取られた駒のあったマス        	
            Square capsq = to;

#if defined(LONG_EFFECT_LIBRARY)
            // 移動先で駒を捕獲するときの利きの更新
            LongEffect::update_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc,
                                                      captured);
#endif

			PieceType pr = raw_type_of(captured);

            // 捕獲した駒に関するevalListの更新
#if defined(USE_EVAL_LIST)
			// このPieceNumberの駒が手駒に移動したのでEvalListのほうを更新しておく。
			PieceNumber piece_no = piece_no_of(to);
			ASSERT_LV3(is_ok(piece_no));
			dp.dirty_num                  = 2;  // 動いた駒は2個
			dp.pieceNo[1]                 = piece_no;
			dp.changed_piece[1].old_piece = evalList.bona_piece(piece_no);
			evalList.put_piece(piece_no, Us, pr, hand_count(hand[Us], pr));
			dp.changed_piece[1].new_piece = evalList.bona_piece(piece_no);
#endif

            // 駒取りなら現在の手番側の駒が増える。
            // ⚠ piece_no_of()で手駒の枚数を参照するので
            //    ↑のあとに行う必要がある。
            put_hand_piece(Us, pr);

#if defined(USE_SFNN)
            dp.remove_pc = captured;
            dp.remove_sq = capsq;
#endif

	        // Update board and piece lists
            // 捕獲される駒の除去
            remove_piece(to);
            remove_piece_for_partial_key(st, captured, to);

            // 捕獲された駒が盤上から消えるので局面のhash keyを更新する
            k ^= Zobrist::psq[captured][capsq];
#if !STOCKFISH
            h += Zobrist::hand[Us][pr];
#endif

#if defined(USE_PIECE_VALUE)
            // 評価関数で使う駒割りの値も更新
            materialDiff += Eval::CapturePieceValue[captured];
#endif

			// 捕獲した駒をStateInfoに保存しておく。(undo_moveのため)
            st->capturedPiece = captured;
        }
        else
        {
            // 駒を取らない指し手

            st->capturedPiece = NO_PIECE;

#if defined(LONG_EFFECT_LIBRARY)
            // 移動先で駒を捕獲しないときの利きの更新
            LongEffect::update_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif
#if defined(USE_EVAL_LIST)
            dp.dirty_num = 1;  // 動いた駒は1個
#endif
        }

        // fromにあったmoved_pcがtoにmoved_after_pcとして移動した。

        k ^= Zobrist::psq[moved_pc][from] ^ Zobrist::psq[moved_after_pc][to];

        // 💪 これでdo_move()後のhash keyが確定したのでprefetchしておく。

        if constexpr (std::is_same_v<T, TranspositionTable>)
        {
            const auto key = k ^ h;
            prefetch(tt->first_entry(key, them));
        }

#if defined(USE_EVAL_LIST)
        // 移動元にあった駒のpiece_noを得る
        PieceNumber piece_no2         = piece_no_of(from);
        dp.pieceNo[0]                 = piece_no2;
        dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no2);
#endif

        // 移動元の升からの駒の除去
        remove_piece(from);
        remove_piece_for_partial_key(st, moved_pc, from);

        // 移動先の升に駒を配置
        put_piece(moved_after_pc, to);
        put_piece_for_partial_key(st, moved_after_pc, to);

#if defined(USE_EVAL_LIST)
        evalList.put_piece(piece_no2, to, moved_after_pc);
        dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no2);
#endif

        // 王を移動させる手であるなら、kingSquareを更新しておく。
        // ⚠ これを更新しておかないとsquare<KING>()が使えなくなってしまう。
        //     王は駒打できないのでdropの指し手に含まれていることはないから
        //     dropのときにはkingSquareを更新する必要はない。

        if (type_of(moved_pc) == KING)
            kingSquare[Us] = to;

        // put_piece()などを用いたのでupdateする。
        // ROOK_DRAGONなどをこの直後で用いるのでここより後ろにやるわけにはいかない。
        update_bitboards();

        // 王手している駒のbitboardを更新する。
        if (givesCheck)
        {
#if 1
            // 高速化のために差分更新する時用

            const StateInfo* prevSt = st->previous;

            // 1) 直接王手であるかどうかは、移動によって王手になる駒別のBitboardを調べればわかる。
            st->checkersBB = prevSt->checkSquares[type_of(moved_after_pc)] & to;

            // 2) 開き王手になるのか
            const Square ksq = square<KING>(them);
            // pos->discovered_check_candidates()で取得したいが、もうstを更新してしまっているので出来ないので
            // prevSt->blockersForKing[~Us] & pieces(Us)と愚直に書く。
            // また、pieces(Us)のうち今回移動させる駒は、実はすでに移動させてしまっているので、fromと書く。

            if (discovered(from, to, ksq, prevSt->blockersForKing[them] & from))
            {
                // fromと敵玉とは同じ筋にあり、かつfromから駒を移動させて空き王手になる。
                // つまりfromから上下を見ると、敵玉と、自分の開き王手をしている遠方駒(飛車 or 香)があるはずなのでこれを追加する。
                // 敵玉はpieces(Us)なので含まれないはずであり、結果として自分の開き王手している駒だけが足される。

                // rookEffect()を用いると、香での王手に対応するのが難しくなるので、
                // 利きの方向ごとに場合分けするほうが簡単

                //   玉
                //   □
                //   駒 ← 今回動かした駒のfrom
                //   □
                //   香
                // のようになっているとして、玉から見て駒のfromが(DIRECT_D)にあるということは、
                // 駒のfromの下に王手している駒があって、それによって開き王手になったということ。

                st->checkersBB |= directEffect(from, direct_of(ksq, from), pieces()) & pieces(Us);
            }

            // 差分更新したcheckersBBが正しく更新されているかをテストするためのassert
            ASSERT_LV3(st->checkersBB == attackers_to<Us>(square<KING>(them)));
#else
            // 差分更新しないとき用。(デバッグ等の目的で用いる)
            st->checkersBB = attackers_to<Us>(square<KING>(Them));
#endif
            // 手番側は2手前のものからの継続。
            st->continuousCheck[Us] += 2;
        }
        else
        {

            st->checkersBB          = Bitboard(ZERO);
            st->continuousCheck[Us] = 0;
        }
    }
    // 非手番側のほうは関係ないので前ノードの値をそのまま受け継ぐ。
    //st->continuousCheck[them] = prev->continuousCheck[them];
    // 💡 memcpy()するので自動的にそうなっている。

#if defined(USE_PIECE_VALUE)
    st->materialValue =
      (Value) (st->previous->materialValue + (Us == BLACK ? materialDiff : -materialDiff));
    //ASSERT_LV5(st->materialValue == Eval::material(*this));
#endif

    // 相手番に変更する。
    sideToMove = them;

    // 更新されたhash keyをStateInfoに書き戻す。
    st->board_key = k;
    st->hand_key  = h;

    st->hand = hand[them];

    // このタイミングで王手関係の情報を更新しておいてやる。
    set_check_info<false>();

    // Calculate the repetition info. It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero
    // if the position was not repeated.

    // 繰り返し情報を計算します。これは、同じ局面の前回の発生からの手数で(1,2回目)、
    // 3回繰り返しの場合は負の値、または局面が繰り返されていない場合はゼロです。
    // ⇨　要するに千日手成立時にだけ負。つまり、やねうら王では、1,2,3回目は正、4回目を負。

#if !defined(ENABLE_QUICK_DRAW)
    st->repetition       = 0;
    st->repetition_times = 0;
    st->repetition_type  = REPETITION_NONE;

    //int end        = std::min(st->rule50, st->pliesFromNull);
    int end = std::min(max_repetition_ply /*16*/, st->pliesFromNull);  // 遡り最大16手

    // ※　チェスは終局までの平均手数が100手程度らしいが、将棋AIの対局では平均手数は160手以上で
    // 　長い手数の対局では300手以上になることも珍しくはないので、初手まで千日手判定のために遡ると
    //   ここで非常に時間がかかり、R40程度弱くなってしまう。

    // 最低でも4手はないと同一局面に戻ってこない。
    if (end >= 4)
    {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            if (stp->board_key == st->board_key)
            {
                // 手駒が一致するなら同一局面である。(2手ずつ遡っているので手番は同じである)
                if (stp->hand == st->hand)
                {
                    // 同一局面が見つかった。

                    // 以下、Stockfishのコードは利用せず、将棋風に書き換えてある。

                    // 繰り返し回数のカウント
                    st->repetition_times = stp->repetition_times + 1;

                    // (同一局面の)3回目までは正(4回目以降は負)の手数にする。
                    // ※　st->repetition_timesは、4回目の時点において、3になっている。
                    // これにより、
                    //  if (st->repetition && st->repetition < ply)
                    // のようなif式は必ず成立するようになる。(plyはrootからの手数とする)
                    //
                    st->repetition = st->repetition_times >= 3 ? -i : i;

                    // 自分が王手をしている連続王手の千日手なのか？
                    // 相手が王手をしている連続王手の千日手なのか？
                    st->repetition_type = (i <= st->continuousCheck[sideToMove])  ? REPETITION_LOSE
                                        : (i <= st->continuousCheck[~sideToMove]) ? REPETITION_WIN
                                                                                  : REPETITION_DRAW;

                    // 途中が連続王手でない場合、4回目の同一局面で連続王手の千日手は成立せず、普通の千日手となる。
                    //
                    // よって、例えば、3..4回目までの間が連続王手であっても、前回(2..3回目までの間)がREPETITION_DRAW
                    // であれば、今回をREPETITION_DRAWとして扱わなければならない。
                    //
                    // これは、『将棋ガイドブック』P.14に以下のように書かれている。
                    //
                    // > 一局中同一局面の最初と4回目出現の局面の間の一方の指し手が王手の連続であった時、
                    // > 連続王手をしていた側にとって4回目の同一局面が出現した時

                    // 同様の理屈により、1..2回目が先手の連続王手で、2..3回目が後手の連続王手のような場合も、
                    // このまま4回目に達した場合、これは普通の千日手局面である。
                    // ゆえに、3回目以降の同一局面の出現において、
                    // 前回のrepetition_typeと今回のrepetition_typeが異なるならば、今回のrepetition_typeを
                    // 普通の千日手(REPETITION_DRAW)として扱わなければならない。

                    if (stp->repetition_times && st->repetition_type != stp->repetition_type)
                        st->repetition_type = REPETITION_DRAW;

                    break;
                }
                else
                {

                    // 盤上の駒は一致したが、手駒が一致しないケース。

                    // 優等局面か劣等局面であるか。(手番が相手番になっている場合はいま考えない)

                    if (hand_is_equal_or_superior(st->hand, stp->hand))
                    {
                        st->repetition_type = REPETITION_SUPERIOR;
                        st->repetition      = i;
                        // 劣等局面かつ千日手局面とかもありうるのだが、超レアケースなので考えないことにする。
                        break;
                    }

                    if (hand_is_equal_or_superior(stp->hand, st->hand))
                    {
                        st->repetition_type = REPETITION_INFERIOR;
                        st->repetition      = i;
                        break;
                    }

                    // 上記のどちらにも該当しない場合は、盤上の駒がたまたま一致しただけの局面。
                }
            }
        }
    }
#endif

	ASSERT_LV5(pos_is_ok());
}

// ある指し手を指した後のhash keyを返す。
Key Position::key_after(Move m) const {

    Color Us = side_to_move();
    auto  k  = st->board_key ^ Zobrist::side;
    auto  h  = st->hand_key;

    // 移動先の升
    Square to = m.to_sq();
    ASSERT_LV2(is_ok(to));

    if (m.is_drop())
    {
        // --- 駒打ち
        PieceType pr = m.move_dropped_piece();
        ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

        Piece pc = make_piece(Us, pr);

        // Zobrist keyの更新
        k ^= Zobrist::psq[pc][to];
        h -= Zobrist::hand[pr][Us];
    }
    else
    {
        // -- 駒の移動
        Square from = m.from_sq();
        ASSERT_LV2(is_ok(from));

        // 移動させる駒
        Piece moved_pc = piece_on(from);
        ASSERT_LV2(moved_pc != NO_PIECE);

        // 移動先に駒の配置
        // もし成る指し手であるなら、成った後の駒を配置する。
        Piece moved_after_pc = m.is_promote() ? make_promoted_piece(moved_pc) : moved_pc;

        // 移動先の升にある駒
        Piece captured = piece_on(to);
        if (captured != NO_PIECE)
        {
            PieceType pr = raw_type_of(captured);

            // 捕獲された駒が盤上から消えるので局面のhash keyを更新する
            k ^= Zobrist::psq[captured][to];
            h += Zobrist::hand[Us][pr];
        }

        // fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
        k ^= Zobrist::psq[moved_pc][from];
        k ^= Zobrist::psq[moved_after_pc][to];
    }

    return k ^ h;
}

// 指し手で盤面を1手戻す。do_move()の逆変換。
template <Color Us>
void Position::undo_move_impl(Move m) {
    // Usは1手前の局面での手番(に呼び出し元でしてある)

    auto to = m.to_sq();
    ASSERT_LV2(is_ok(to));

    // --- 移動後の駒

    Piece moved_after_pc = moved_piece_after(m);

#if defined(USE_EVAL_LIST)
    PieceNumber piece_no = piece_no_of(to);  // 移動元のpiece_no == いまtoの場所にある駒のpiece_no
    ASSERT_LV3(is_ok(piece_no));
#endif

    // 移動前の駒
    // Piece moved_pc = is_promote(m) ? (moved_after_pc - PIECE_PROMOTE) : moved_after_pc;

    // ↑の処理、mの成りを表現するbitを直接、Pieceの成りを表現するbitに持ってきたほうが速い。
    static_assert((u32) MOVE_PROMOTE / (u32) PIECE_PROMOTE == 4096, "");
    // log(2)4096 == 12
    Piece moved_pc = Piece(moved_after_pc ^ ((m.to_u16() & MOVE_PROMOTE) >> 12));

    if (m.is_drop())
    {
        // --- 駒打ち

        // toの場所にある駒を手駒に戻す
        PieceType pt = raw_type_of(moved_after_pc);

#if defined(USE_EVAL_LIST)
        evalList.put_piece(piece_no, Us, pt, hand_count(hand[Us], pt));
#endif
		// 手駒が増える
        put_hand_piece(Us, pt);

        // toの場所から駒を消す
        remove_piece(to);

#if defined(LONG_EFFECT_LIBRARY)
        // 駒打ちのundoによる利きの復元
        LongEffect::rewind_by_dropping_piece<Us>(*this, to, moved_after_pc);
#endif
    }
    else
    {

        // --- 通常の指し手

        auto from = m.from_sq();
        ASSERT_LV2(is_ok(from));

        // toの場所から駒を消す
        remove_piece(to);

        // toの地点には捕獲された駒があるならその駒が盤面に戻り、手駒から減る。
        // 駒打ちの場合は捕獲された駒があるということはありえない。
        // (なので駒打ちの場合は、st->capturedTypeを設定していないから参照してはならない)
        if (st->capturedPiece != NO_PIECE)
        {
            Piece to_pc = st->capturedPiece;

            // 盤面のtoの地点に捕獲されていた駒を復元する
            put_piece(to_pc, to);
            put_piece(moved_pc, from);

#if defined(USE_EVAL_LIST)
            PieceNumber piece_no2 =
              piece_no_of(Us, raw_type_of(to_pc));  // 捕っていた駒(手駒にある)のpiece_no
            ASSERT_LV3(is_ok(piece_no2));

            evalList.put_piece(piece_no2, to, to_pc);

            // 手駒から減らす
            remove_hand_piece(Us, raw_type_of(to_pc));

            // 成りの指し手だったなら非成りの駒がfromの場所に戻る。さもなくばそのまま戻る。
            // moved_pcが玉であることはあるが、いまkingSquareを更新してしまうと
            // rewind_by_capturing_piece()でその位置を用いているのでまずい。(かも)
            evalList.put_piece(piece_no, from, moved_pc);
#else
            // 手駒から減らす
            remove_hand_piece(Us, raw_type_of(to_pc));
#endif

#if defined(LONG_EFFECT_LIBRARY)
            // 移動先で駒を捕獲するときの利きの更新
            LongEffect::rewind_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc,
                                                      to_pc);
#endif
        }
        else
        {

            put_piece(moved_pc, from);

#if defined(USE_EVAL_LIST)
            // 成りの指し手だったなら非成りの駒がfromの場所に戻る。さもなくばそのまま戻る。
            evalList.put_piece(piece_no, from, moved_pc);
#endif

#if defined(LONG_EFFECT_LIBRARY)
            // 移動先で駒を捕獲しないときの利きの更新
            // このときに元あった玉の位置を用いるのでkingSquareはまだ更新してはならない。
            LongEffect::rewind_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif
        }

        if (type_of(moved_pc) == KING)
            kingSquare[Us] = from;
    }

    // put_piece()などを使ったのでbitboardを更新する。
    // kingSquareは自前で更新したのでupdate_kingSquare()を呼び出す必要はない。
    update_bitboards();

    // --- 相手番に変更
    sideToMove = Us;  // Usは先後入れ替えて呼び出されているはず。

    // --- StateInfoを巻き戻す
    st = st->previous;

    --gamePly;

    // ASSERT_LV5(evalList.is_valid(*this));
    //evalList.is_valid(*this);

    ASSERT_LV5(pos_is_ok());
}

// do_move()を先後分けたdo_move_impl<>()を呼び出す。
template <typename T>
void Position::do_move(Move m, StateInfo& newSt, bool givesCheck, const T* tt)
{
    if (sideToMove == BLACK)
        do_move_impl<BLACK, T>(m, newSt, givesCheck, tt);
    else
        do_move_impl<WHITE, T>(m, newSt, givesCheck, tt);
}

// undo_move()を先後分けたdo_move_impl<>()を呼び出す。
void Position::undo_move(Move m)
{
	if (sideToMove == BLACK)
		undo_move_impl<WHITE>(m); // 1手前の手番が返らないとややこしいので入れ替えておく。
	else
		undo_move_impl<BLACK>(m);
}

// null move searchに使われる。手番だけ変更する。
template <typename T>
void Position::do_null_move(StateInfo& newSt, const T& tt) {

	ASSERT_LV3(!checkers());
	ASSERT_LV3(&newSt != st);

#if STOCKFISH
    std::memcpy(&newSt, st, sizeof(StateInfo));
#else
	std::memcpy(static_cast<void*>(& newSt), st, sizeof(StateInfo));
#endif

	newSt.previous = st;
    st             = &newSt;

#if STOCKFISH
	if (st->epSquare != SQ_NONE)
    {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    st->key ^= Zobrist::side;
    prefetch(tt.first_entry(key()));

	st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;

    assert(pos_is_ok());

#else

#if defined(USE_CLASSIC_EVAL) && defined(EVAL_NNUE)
    // NNUEの場合、KPPT型と違って、手番が違う場合、計算なしに済ますわけにはいかない。
    st->accumulator.computed_score = false;
#endif

	// このタイミングでアドレスが確定するのでprefetchしたほうが良い。(かも)
    // →　将棋では評価関数の計算時のメモリ帯域がボトルネックになって、ここでprefetchしても
    // 　prefetchのスケジューラーが処理しきれない可能性が…。
    // CPUによっては有効なので一応やっておく。

	// 📝 以下のprefetchのfirst_entry()でやねうら王は、この時の手番が必要なので
    //     先に手番を変えておく。
    sideToMove = ~sideToMove;

	st->board_key ^= Zobrist::side;

	// 🌈 やねうら王では、TTを引数に取らないこともできるように
	//     template引数で実装している。
	if constexpr (std::is_same_v<T, TranspositionTable>)
    {
        const auto key = st->key();
        prefetch(tt.first_entry(key, sideToMove));
    }

	st->pliesFromNull = 0;

	// 手番が変わるので手番側の手駒情報であるst->handの更新が必要。
    st->hand = hand[sideToMove];

	// 現局面には王手はかかっていないので、直前には王手はされていない、
	// すなわちこの関数が呼び出された時の非手番側(いまのsideToMove)である
    //   st->continuousCheck[sideToMove] == 0
    // が言える。連続王手の千日手の誤判定を防ぐためにこの関数が呼び出された時の手番側(~sideToMove)も
    // 0にリセットする必要がある。

	ASSERT_LV3(st->continuousCheck[sideToMove] == 0);
    st->continuousCheck[~sideToMove] = 0;

	set_check_info<true>();

#if !defined(ENABLE_QUICK_DRAW)
	st->repetition       = 0;

	// 🌈 やねうら王では繰り返し回数のカウントがある。
	st->repetition_times = 0;
#endif

	ASSERT_LV5(pos_is_ok());
#endif
}

void Position::do_null_move(StateInfo& newSt) {
    // Tの型としてTranspositionTable以外を渡すと
    // 最適化によって消えるはず。
    do_null_move<int>(newSt, 0);
}

void Position::undo_null_move()
{
	ASSERT_LV3(!checkers());

	st = st->previous;
	sideToMove = ~sideToMove;
}


#if defined (USE_SEE)

// Tests if the SEE (Static Exchange Evaluation)
// value of move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.


// Position::see()は指し手のSEE(静的交換評価)の値が、与えられたthreshold(しきい値)以上であるかをテストする。
// null windowの時のalpha-beta法に似たアルゴリズムを用いる。
//
// ※　SEEの解説についてはググれ。
//
// ある升での駒の取り合いの結果、どれくらい駒得/駒損するかを評価する。
// 最初に引数として、指し手mが与えられる。この指し手に対して、同金のように取り返され、さらに同歩成のように
// (価値の低い駒を優先して用いて)取り返していき、最終的な結果(評価値のうちの駒割りの部分の増減)を返すのが本来のSEE。
//
// ただし、途中の手順では、同金とした場合と同金としない場合とで、(そのプレイヤーは自分が)得なほうを選択できるものとする。
//
// ※　KINGを敵の利きに移動させる手は非合法手なので、ここで与えられる指し手にはそのような指し手は含まないものとする。
// また、SEEの地点(to)の駒をKINGで取る手は含まれるが、そのKINGを取られることは考慮しなければならない。
// 最後になった駒による成りの上昇値は考えない。
//
// このseeの最終的な値が、しきい値threshold以上になるかどうかを判定するのがsee_ge()である。
// こういう設計にすることで早期にthresholdを超えないことが確定した時点でreturn出来る。

bool Position::see_ge(Move m, Value threshold) const
{
	ASSERT_LV3(m.is_ok());

#if STOCKFISH
    // Only deal with normal moves, assume others pass a simple SEE
    if (type_of(m) != NORMAL)
        return VALUE_ZERO >= threshold;
#endif

	bool drop = m.is_drop();

	// 以下、Stockfishの挙動をなるべく忠実に再現する。

#if STOCKFISH
    Square from = m.from_sq(), to = m.to_sq();
#else
	// 駒の移動元(駒打ちの場合は)と移動先。
	// dropのときにはSQ_NBにしておくことで、pieces() ^ fromを無効化するhack
	// ※　piece_on(SQ_NB)で NO_PIECE が返ることは保証されている。
	Square from = drop ? SQ_NB : m.from_sq();
	Square to   = m.to_sq();
#endif

	/*
		📓
			将棋だと、駒打ちで、SEE > 0になることはないので(打った駒を取られてマイナスになることはあっても)
			threshold > 0なら、即座に falseが返せる。

			if (drop && threshold > 0)
				return false;

			しかし、この判定、以下の条件式が含むから、入れる必要がない。
	*/

	// toの地点にある駒の価値がthreshold以上ではない。
	// この場合、取り返されなかったとしても、条件を満たすことはないので即座にfalseを返せる。

    int swap = PieceValue[piece_on(to)] - threshold;
	if (swap < 0)
        return false;

	// この時点で、
	//   PieceValue[piece_on(to)] - 最初に動かす駒の価値 >= threshold
	// なら、取り返されたところですでにしきい値以上になることは確定しているのでtrueが返せる。

#if STOCKFISH
	swap = PieceValue[piece_on(from)] - swap;
#else
	// →　駒打ちの時は、移動元にその駒がないので、これを復元してやる必要がある。
	PieceType from_pt = drop ? m.move_dropped_piece() : type_of(piece_on(from));
    swap = PieceValue[from_pt] - swap;
#endif

	if (swap <= 0)
        return true;

#if STOCKFISH
	assert(color_of(piece_on(from)) == sideToMove);
#else
	ASSERT_LV3(drop || color_of(piece_on(from)) == sideToMove);
#endif

    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        // 手番側がtoに利く駒が尽きたなら、お手上げ。(see_geの判定は)手番側の負け。
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        // 元のマスにピンしている駒がある限り、ピンされている駒が攻撃することを許可しない。
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to
        // the bitboard 'attackers' any X-ray attackers behind it.
        // 次に価値の低い攻撃駒を特定して取り除き、その背後にいるX線攻撃駒を
        // ビットボード 'attackers' に追加する。

        // 歩で取れるなら、まず歩で取る。
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            // この時点で、歩で取れることは確定した。

            // この時点でPawnValue以上に得しているなら、この歩を取り返されたところで、手抜いてthresholdを下回らないので、returnできる。
            if ((swap = PawnValue - swap) < res)
                break;

#if STOCKFISH
            occupied ^= least_significant_square_bb(bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
            // →　チェスではPAWNで取る時、PAWNが斜めに移動するので、toの斜め(X-ray)にある駒を
            //    attackersとして追加する必要があるが、将棋の場合は、歩の背後にいる香・飛車を追加する必要がある。
#endif
        }

#if !STOCKFISH
        // 香を試す(将棋only)
        else if ((bb = stmAttackers & pieces(LANCE)))
        {
            if ((swap = LanceValue - swap) < res)
                break;
        }
#endif
        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            // 桂で取ったところでその背後にある駒がattckersに追加されることはないので、何も追加する必要はなく、
            // ループ先頭のwhileに戻る。
            continue;
        }

#if !STOCKFISH
        // 銀を試す(将棋only)
        else if ((bb = stmAttackers & pieces(SILVER)))
        {
            if ((swap = SilverValue - swap) < res)
                break;
        }
        // 金を試す(将棋only)
        else if ((bb = stmAttackers & pieces(GOLDS)))
        {
            // ここ、今回捕獲する金相当の駒の価値にすべきかも知れないが、
            // この時点ではまだ今回動かす駒の移動元が得られていないので、その処理書きにくい。
            if ((swap = GoldValue - swap) < res)
                break;
        }
#endif

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
#if STOCKFISH
            occupied ^= least_significant_square_bb(bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
#endif
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
#if STOCKFISH
            occupied ^= least_significant_square_bb(bb);
            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
#endif
        }

#if !STOCKFISH
        // 馬を試す(将棋only)
        else if ((bb = stmAttackers & pieces(HORSE)))
        {
            if ((swap = Eval::HorseValue - swap) < res)
                break;
            //occupied ^= least_significant_square_bb(bb);
            //attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        // 竜を試す(将棋only)
        else if ((bb = stmAttackers & pieces(DRAGON)))
        {
            if ((swap = Eval::DragonValue - swap) < res)
                break;
            //occupied ^= least_significant_square_bb(bb);
            //attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        }
#endif

#if STOCKFISH
        else if ((bb = stmAttackers & pieces(QUEEN)))
        {
            if ((swap = QueenValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
            attackers |= (attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN))
                       | (attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN));
        }
#endif
        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;

#if !STOCKFISH
        /*
			occupied ^= least_significant_square_bb(bb);
			attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);

			みたいなのに相当する処理。将棋では、この部分、もう少し最適化できる。
		*/

        // 今回移動させてtoの駒を取るための駒の移動元の升
        Square sq = bb.pop();
        // bbにあった駒を取り除く
        occupied ^= sq;

        // sqにあった駒が消えるので、toから見てsqの延長線上にある駒を追加する。

        auto dirs = directions_of(to, sq);

        // 桂以外の移動なので8方向であるはず。
        ASSERT_LV3(dirs);

        // clang-format off

		switch(pop_directions(dirs))
		{
		// 斜め方向なら斜め方向の升をスキャンしてその上にある角・馬を足す
		case DIRECT_RU: attackers |= rayEffect<DIRECT_RU>(to, occupied) & pieces(BISHOP_HORSE); break;
		case DIRECT_LD: attackers |= rayEffect<DIRECT_LD>(to, occupied) & pieces(BISHOP_HORSE); break;
		case DIRECT_RD: attackers |= rayEffect<DIRECT_RD>(to, occupied) & pieces(BISHOP_HORSE); break;
		case DIRECT_LU: attackers |= rayEffect<DIRECT_LU>(to, occupied) & pieces(BISHOP_HORSE); break;

		// (toに対してsqが)上方向。背後の駒によってtoの地点に利くのは、後手の香 + 先後の飛車
		case DIRECT_U : attackers |= rayEffect<DIRECT_U >(to, occupied) & (pieces(ROOK_DRAGON) | pieces(WHITE, LANCE)); break;

		// (toに対してsqが)下方向。背後の駒によってtoの地点に利くのは、先手の香 + 先後の飛車
		case DIRECT_D : attackers |= rayEffect<DIRECT_D >(to, occupied) & (pieces(ROOK_DRAGON) | pieces(BLACK, LANCE)); break;

		// 左右方向に移動した時の背後の駒によってtoの地点に利くのは、飛車・龍。
		case DIRECT_L : attackers |= rayEffect<DIRECT_L >(to, occupied) & pieces(ROOK_DRAGON); break;
		case DIRECT_R : attackers |= rayEffect<DIRECT_R >(to, occupied) & pieces(ROOK_DRAGON); break;

		default: UNREACHABLE; break;
		}

        // clang-format on

        // SEEって、最後、toの地点で成れるなら、その成ることによる価値上昇分も考慮すべきだと思うのだが、
        // そうすると早期枝刈りができないことになるので、とりあえず、このままでいいや。
#endif
	}

    return bool(res);
}

#endif // defined (USE_SEE)

// ----------------------------------
//      千日手判定
// ----------------------------------

#if 0
// Tests whether the position is drawn by 50-move rule
// or by repetition. It does not detect stalemates.
// この処理は、局面が50手ルールまたは繰り返しによって
// 引き分けになっているかどうかをテストします。ステイルメイトは検出されません。
bool Position::is_draw(int ply) const {

    if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
        return true;

    // Return a draw score if a position repeats once earlier but strictly
    // after the root, or repeats twice before or at the root.
    return st->repetition && st->repetition < ply;
}
#endif

// 連続王手の千日手等で引き分けかどうかを返す
RepetitionState Position::is_repetition(int ply) const
{
#if !defined(ENABLE_QUICK_DRAW)

	// Return a draw score if a position repeats once earlier but strictly
    // after the root, or repeats twice before or at the root.
	// ルートより厳密に後である場合に局面が一度繰り返された場合、
	// またはルートの前またはルートで局面が2回繰り返された場合に、引き分けのスコアを返します。
	// ⇨　将棋では、「2回」ではなく「3回」。(現局面を含めると4回目の同一局面の出現)

	// cf.
	//   Don't score as an immediate draw 2-fold repetitions of the root position
	//   https://github.com/official-stockfish/Stockfish/commit/6d89d0b64a99003576d3e0ed616b43333c9eca01

	// 📝　基本的にrootより遡って判定しないのだが、しかし、4回目の同一局面の場合は、強制的に千日手となるため、
	// 　   ここで探索は打ち切られなければならない。よって、4回目の同一局面の場合のみ、plyに関わらず
	//      REPETITION_NONE以外が返る。

    if (st->repetition && st->repetition < ply)
		return st->repetition_type;

	return REPETITION_NONE;
#else
	// pliesFromNullが未初期化になっていないかのチェックのためのassert
	ASSERT_LV3(st->pliesFromNull >= 0);

	// 遡り可能な手数。
	// 最大でも(root以降であっても)16手までしか遡らないことにする。
	// (これ以上遡っても千日手が見つかることが稀)
	// ここss->ply(rootからの手数)にするとR5ぐらい弱くなる。
	// また、root以前にも遡る。こうした方が+R5ぐらい強くなる。

	int end = std::min(16, st->pliesFromNull);

	// 少なくとも4手かけないと千日手にはならないから、4手前から調べていく。
	if (end < 4)
		return REPETITION_NONE;

	StateInfo* stp = st->previous->previous;

	for (int i = 4; i <= end ; i += 2)
	{
		stp = stp->previous->previous;

		// board_key : 盤上の駒のみのhash(手駒を除く)
		// 盤上の駒が同じ状態であるかを判定する。
		if (stp->board_key == st->board_key)
		{
			// 手駒が一致するなら同一局面である。(2手ずつ遡っているので手番は同じである)
			if (stp->hand == st->hand)
			{
				// 自分が王手をしている連続王手の千日手なのか？
				if (i <= st->continuousCheck[ sideToMove])
					return REPETITION_LOSE;

				// 相手が王手をしている連続王手の千日手なのか？
				if (i <= st->continuousCheck[~sideToMove])
					return REPETITION_WIN;

				return REPETITION_DRAW;
			}
			else {
				// 優等局面か劣等局面であるか。(手番が相手番になっている場合はいま考えない)
				if (hand_is_equal_or_superior(st ->hand , stp->hand))
					return REPETITION_SUPERIOR;
				if (hand_is_equal_or_superior(stp->hand , st ->hand))
					return REPETITION_INFERIOR;
			}
		}
	}

	// 同じhash keyの局面が見つからなかったので…。
	return REPETITION_NONE;

#endif
}

#if !defined(ENABLE_QUICK_DRAW)
// Tests whether there has been at least one repetition
// of positions since the last capture or pawn move.
bool Position::has_repeated() const {

    StateInfo* stc = st;
    //int        end = std::min(st->rule50, st->pliesFromNull);
    int        end = std::min(max_repetition_ply, st->pliesFromNull);
    while (end-- >= 4)
    {
        if (stc->repetition)
            return true;

        stc = stc->previous;
    }
    return false;
}
#endif

// is_repetition()の、千日手が見つかった時に、現局面から何手遡ったかを返すバージョン。
// found_plyにその値が返ってくる。
RepetitionState Position::is_repetition(int ply, int& found_ply) const
{
#if !defined(ENABLE_QUICK_DRAW)
	// ただ、ここでply >= 16を指定しても、do_move()の時にmax_repetition_ply(=16)手までしか
	// 遡っていない。無限に遡りたいなら、set_max_repetition_ply()を用いてこの値を変更しておくこと。
    if (st->repetition && st->repetition < ply)
	{
		// st->repetitionは負もありうるのでabs()が必要。
		found_ply = abs(st->repetition);
		return st->repetition_type;
	}

	return REPETITION_NONE;
#else
	// pliesFromNullが未初期化になっていないかのチェックのためのassert
	ASSERT_LV3(st->pliesFromNull >= 0);

	// 遡り可能な手数。
	// 最大でもply手までしか遡らないことにする。
	int end = std::min(ply, std::min(max_repetition_ply, st->pliesFromNull));

	found_ply = 0;

	// 少なくとも4手かけないと千日手にはならないから、4手前から調べていく。
	if (end < 4)
		return REPETITION_NONE;

	StateInfo* stp = st->previous->previous;

	for (found_ply = 4; found_ply <= end ; found_ply += 2)
	{
		stp = stp->previous->previous;

		// board_key : 盤上の駒のみのhash(手駒を除く)
		// 盤上の駒が同じ状態であるかを判定する。
		if (stp->board_key == st->board_key)
		{
			// 手駒が一致するなら同一局面である。(2手ずつ遡っているので手番は同じである)
			if (stp->hand == st->hand)
			{
				// 自分が王手をしている連続王手の千日手なのか？
				if (found_ply <= st->continuousCheck[ sideToMove])
					return REPETITION_LOSE;

				// 相手が王手をしている連続王手の千日手なのか？
				if (found_ply <= st->continuousCheck[~sideToMove])
					return REPETITION_WIN;

				return REPETITION_DRAW;
			}
			else {
				// 優等局面か劣等局面であるか。(手番が相手番になっている場合はいま考えない)
				if (hand_is_equal_or_superior(st ->hand, stp->hand))
					return REPETITION_SUPERIOR;
				if (hand_is_equal_or_superior(stp->hand, st ->hand))
					return REPETITION_INFERIOR;
			}
		}
	}

	// 同じhash keyの局面が見つからなかったので…。
	return REPETITION_NONE;
#endif
}

// ----------------------------------
//      入玉判定
// ----------------------------------

// 現在の盤面から、入玉に必要な駒点を計算し、enteringKingPointに設定する。
void Position::update_entering_point() {

	int points[COLOR_NB];

	switch (ekr)
	{
	case EKR_24_POINT:   // 24点法(31点以上で宣言勝ち)
	case EKR_24_POINT_H: // 24点法 , 駒落ち対応
		points[BLACK] = points[WHITE] = 31;
		break;

	case EKR_27_POINT:   // 27点法 == CSAルール
	case EKR_27_POINT_H: // 27点法 , 駒落ち対応
		points[BLACK] = 28;
		points[WHITE] = 27;
		break;

    case EKR_NULL :
		// set_erk()を呼び出すのを忘れている。
        sync_cout << "ekr == EKR_NULL in update_entering_point()" << sync_endl;
        Tools::exit();
        return;

	default:
		// それ以外では入玉の駒点を用いないので無視できる。
		return;
	}

	// --- 盤上の駒を数える。大駒5点、小駒1点とする。これは、Bitboardを用いると速い。

	// 盤上のすべての駒の枚数
	auto p1 = pieces().pop_count();

	// 盤上の大駒の枚数
	auto p2 = pieces(BISHOP_HORSE,ROOK_DRAGON).pop_count();

	// 盤上の駒点
	// 大駒の枚数の4倍を加算すれば、小駒を1点、大駒を5点とみなして計算したことになる。
	auto p = p1 + p2 * 4;

	// 手駒の駒点
	for (auto c : COLOR )
	{
		auto h = hand[c];
		p += hand_count(h, PAWN) + hand_count(h, LANCE) + hand_count(h, KNIGHT) + hand_count(h, SILVER) + hand_count(h, GOLD)
			+ hand_count(h, BISHOP) * 5 + hand_count(h, ROOK) * 5;
	}
	// すべての駒があるなら、p == 56になるはず。
	// (先手、小駒9*2段(=18枚*1点=18点) + 大駒2枚(=2枚*5点=10点) = 28点。後手も同様で、全体ではこの倍 = 56点)
        if (p != 56 && (ekr == EKR_24_POINT_H || ekr == EKR_27_POINT_H))
	{
		// 56から足りない分だけ後手が駒落ちにしていると考えられる。
		// 駒落ち対応入玉ルールであるなら、この分を引き算して考える。

		// 駒落ちにおいては上手(うわて)が先手だと考えるなら、
		// BLACKとWHITEの駒点を入れ替える必要がある。
		// std::swap(points[BLACK], points[WHITE]);
		// →　AobaZeroでは駒落ちの場合、上手は後手とみなすらしい。やねうら王もこれに倣う。
		// cf. やねうら王がAobaZeroに駒落ちで負けまくっている件について : https://yaneuraou.yaneu.com/2021/09/14/yaneuraou-is-losing-too-much-to-aobazero/

		// 56 - p だけ駒落ち。マイナスになることはない(上の計算法だと裸玉でも1点あるので..)
		points[WHITE] -= 56 - p;
	}

	enteringKingPoint[BLACK] = points[BLACK];
	enteringKingPoint[WHITE] = points[WHITE];
}

Move Position::DeclarationWin() const
{
    switch (ekr)
	{
		// 入玉ルールなし
	case EKR_NONE: return Move::none();

		// CSAルールに基づく宣言勝ちの条件を満たしているか
		// 満たしているならば非0が返る。返し値は駒点の合計。
		// cf.http://www.computer-shogi.org/protocol/tcp_ip_1on1_11.html
	case EKR_24_POINT: // 24点法(31点以上で宣言勝ち)
	case EKR_27_POINT: // 27点法 == CSAルール
	case EKR_24_POINT_H: // 24点法 , 駒落ち対応
	case EKR_27_POINT_H: // 27点法 , 駒落ち対応
	{
		/*
		「入玉宣言勝ち」の条件(第13回選手権で使用のもの):

		次の条件が成立する場合、勝ちを宣言できる(以下「入玉宣言勝ち」と云う)。
		条件:
		(a) 宣言側の手番である。
		(b) 宣言側の玉が敵陣三段目以内に入っている。
		(c) 宣言側が(大駒5点小駒1点の計算で)
		・先手の場合28点以上の持点がある。
		・後手の場合27点以上の持点がある。
		・点数の対象となるのは、宣言側の持駒と敵陣三段目
		以内に存在する玉を除く宣言側の駒のみである。
		(d) 宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する。
		(e) 宣言側の玉に王手がかかっていない。
		(詰めろや必死であることは関係ない)
		(f) 宣言側の持ち時間が残っている。(切れ負けの場合)
		以上1つでも条件を満たしていない場合、宣言した方が負けとなる。
		(注) このルールは、日本将棋連盟がアマチュアの公式戦で使用しているものである。

		以上の宣言は、コンピュータが行い、画面上に明示する。
		*/
		// (a)宣言側の手番である。
		// →　手番側でこの関数を呼び出して判定するのでそうだろう。

		Color us = sideToMove;

		// 敵陣
		Bitboard ef = enemy_field(us);

		// (b)宣言側の玉が敵陣三段目以内に入っている。
		if (!(ef & square<KING>(us)))
			return Move::none();

		// (e)宣言側の玉に王手がかかっていない。
		if (checkers())
			return Move::none();


		// (d)宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する。
		int p1 = (pieces(us) & ef).pop_count();
		// p1には玉も含まれているから11枚以上ないといけない
		if (p1 < 11)
			return Move::none();

		// 敵陣にいる大駒の数
		int p2 = ((pieces(us, BISHOP_HORSE, ROOK_DRAGON)) & ef).pop_count();

		// 小駒1点、大駒5点、玉除く
		// ＝　敵陣の自駒 + 敵陣の自駒の大駒×4 - 玉

		// (c)
		// ・先手の場合28点以上の持点がある。
		// ・後手の場合27点以上の持点がある。
		Hand h = hand[us];
		int score = p1 + p2 * 4 - 1
			+ hand_count(h, PAWN) + hand_count(h, LANCE) + hand_count(h, KNIGHT) + hand_count(h, SILVER)
			+ hand_count(h, GOLD) + (hand_count(h, BISHOP) + hand_count(h, ROOK)) * 5;

		// rule==EKR_27_POINTならCSAルール。rule==EKR_24_POINTなら24点法(30点以下引き分けなので31点以上あるときのみ勝ち扱いとする)
		//if (score < (rule == EKR_27_POINT ? (us == BLACK ? 28 : 27) : 31))
			//return Move::none();

		// ↓ 駒落ち対応などを考慮して、enteringKingPoint[]を参照することにした。

		if (score < enteringKingPoint[us])
			return Move::none();

		// 評価関数でそのまま使いたいので駒点を返しておくのもアリか…。
		return Move::win();
	}

	// トライルールの条件を満たしているか。
	case EKR_TRY_RULE:
	{
		Color us = sideToMove;
		Square king_try_sq = (us == BLACK ? SQ_51 : SQ_59);
		Square king_sq = square<KING>(us);

		// 1) 初期陣形で敵玉がいた場所に自玉が移動できるか。
		if (!(kingEffect(king_sq) & king_try_sq))
			return Move::none();

		// 2) トライする升に自駒がないか。
		if (pieces(us) & king_try_sq)
			return Move::none();

		// 3) トライする升に移動させたときに相手に取られないか。
		if (effected_to(~us, king_try_sq, king_sq))
			return Move::none();

		// 王の移動の指し手により勝ちが確定する
		return make_move(king_sq, king_try_sq, us,KING);
	}

	default:
		UNREACHABLE;
		return Move::none();
	}
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
// 白と黒の立場を反転させて局面を反転させる。
// 評価関数の対称性バグを見つけるなど、デバッグの目的でのみ有用。

// 盤面を180°回転させる。

void Position::flip() {

#if STOCKFISH
    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_8; r >= RANK_1; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_1 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;  // Castling availability
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;  // En passant square
    f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

    std::getline(ss, token);  // Half and full moves
    f += token;

    set(f, is_chess960(), st);

    assert(pos_is_ok());

#else

	auto f_sfen = flipped_sfen();
    set(f_sfen, st);

	ASSERT_LV5(pos_is_ok());

#endif
}



// ----------------------------------
//      内部情報の正当性のテスト
// ----------------------------------

// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.

// この処理は、局面オブジェクトに対していくつかの整合性チェックを行い、
// 何かおかしい箇所が検出された場合にassertを発生させます。
// これはデバッグ時に役立つことを意図しています。

bool Position::pos_is_ok() const
{
	// Bitboardの完全なテストには時間がかかるので、あまりややこしいテストは現実的ではない。

#if ASSERT_LV >= 5
	// 1) 盤上の駒と手駒を合わせて40駒あるか。
	// →　駒落ちに対応させたいのでコメントアウト

	// それぞれの駒のあるべき枚数
	const int ptc0[KING] = { 2/*玉*/,18/*歩*/,4/*香*/,4/*桂*/,4/*銀*/,2/*角*/,2/*飛*/,4/*金*/ };
	// カウント用の変数
	int ptc[PIECE_WHITE] = { 0 };

	int count = 0;
	for (auto sq : SQ)
	{
		Piece pc = piece_on(sq);
		if (pc != NO_PIECE)
		{
			++count;
			++ptc[raw_type_of(pc)];
		}
	}
	for (auto c : COLOR)
		for (PieceType pr = PIECE_HAND_ZERO; pr < PIECE_HAND_NB; ++pr)
		{
			int ct = hand_count(hand[c], pr);
			count += ct;
			ptc[pr] += ct;
		}

	if (count != 40)
		return false;

	// 2) それぞれの駒の枚数は合っているか
	for (Piece pt = PIECE_ZERO; pt < KING; ++pt)
		if (ptc[pt] != ptc0[pt])
			return false;


#endif
	// 3) st->handは手番側の駒でなければならない。
	if (st->hand != hand[sideToMove])
		return false;

	// 4) 王手している駒
	if (st->checkersBB != attackers_to(~sideToMove, square<KING>(sideToMove)))
		return false;

	// 5) 相手玉が取れるということはないか
	if (effected_to(sideToMove, square<KING>(~sideToMove)))
		return false;

	// 6) occupied bitboardは合っているか
	if ((pieces() != (pieces(BLACK) | pieces(WHITE))) || (pieces(BLACK) & pieces(WHITE)))
		return false;

	// 7) 王手している駒は敵駒か
	if (checkers() & pieces(side_to_move()))
		return false;

	// 二歩のチェックなど云々かんぬん..面倒くさいので省略。

	return true;
}

// ----------------------------------
//			UnitTest
// ----------------------------------

namespace {
	// performance test
	// ある局面から、全合法手を生成して depth深さまで辿り、局面数がいくらあったかを返す。
	u64 perft(Position& pos, Depth depth)
	{
		StateInfo st;
		u64 cnt, nodes = 0;

		auto ml = MoveList<LEGAL_ALL>(pos);
		if (depth == 1)
			return ml.size();

		const bool leaf = (depth == 2);
		for (const auto& m : ml)
		{
			pos.do_move(m, st);
			cnt = leaf ? MoveList<LEGAL_ALL>(pos).size() : perft(pos, depth - 1);
			nodes += cnt;
			pos.undo_move(m);
		}
		return nodes;
	};
}

void Position::UnitTest(Test::UnitTester& tester, IEngine& engine) {
    auto section1 = tester.section("Position");

    Position  pos;
    StateInfo si;

    // 任意局面での初期化。
    auto pos_init = [&](const std::string& sfen_) { pos.set(sfen_, &si); };

    // 平手初期化
    auto hirate_init = [&] { pos.set_hirate(&si); };
    // 2枚落ち初期化
    auto handi2_sfen = "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    auto handi2_init = [&] { pos.set(handi2_sfen, &si); };

    // 4枚落ち初期化
    auto handi4_sfen = "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    auto handi4_init = [&] { pos.set(handi4_sfen, &si); };

    // 指し手生成祭りの局面
    auto matsuri_sfen = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1";
    auto matsuri_init = [&] { pos.set(matsuri_sfen, &si); };

    Move16 m16;
    Move   m;

    // to_move() のテスト
    {
        auto section2 = tester.section("to_move()");

        // 平手初期化
        hirate_init();

        // is_ok(m) == falseな指し手に対して、to_move()がその指し手をそのまま返すことを保証する。
        tester.test("MOVE_NONE", pos.to_move(Move16::none()) == Move::none());
        tester.test("MOVE_WIN", pos.to_move(Move16::win()) == Move::win());
        tester.test("MOVE_NULL", pos.to_move(Move16::null()) == Move::null());

        // 88の角を22に不成で移動。(非合法手) 移動後の駒は先手の角。
        m16 = make_move16(SQ_88, SQ_22);
        tester.test("make_move16(SQ_88, SQ_22)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (B_BISHOP << 16)));

        // 88の角を22に成る移動。(非合法手) 移動後の駒は先手の馬。
        m16 = make_move_promote16(SQ_88, SQ_22);
        tester.test("make_move_promote16(SQ_88, SQ_22)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (B_HORSE << 16)));

        // 22の角を88に不成で移動。(非合法手) 移動後の駒は後手の角。
        m16 = make_move16(SQ_22, SQ_88);
        tester.test("make_move16(SQ_22, SQ_88)", pos.to_move(m16) == Move::none());

        // 22の角を88に成る移動。(非合法手) 移動後の駒は後手の馬。
        m16 = make_move_promote16(SQ_22, SQ_88);
        tester.test("make_move_promote16(SQ_22, SQ_88)", pos.to_move(m16) == Move::none());

        matsuri_init();
        m16 = make_move_drop16(GOLD, SQ_55);
        tester.test("make_move_drop(SQ_55,GOLD)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (W_GOLD << 16)));
    }

    // pseudo_legal() , legal() のテスト
    {
        auto section2 = tester.section("legality");

        // 平手初期化
        hirate_init();

        // 77の歩を76に移動。(合法手)
        // これはpseudo_legalではある。
        m16 = make_move16(SQ_77, SQ_76);
        m   = pos.to_move(m16);
        tester.test("make_move(SQ_77, SQ_76) is pseudo_legal == true",
                    pos.pseudo_legal(m, true) == true);

#if 0
		// 後手の駒の場合、現在の手番の駒ではないので、pseudo_legalではない。(pseudo_legalは手番側の駒であることを保証する)
		m16 = make_move16(SQ_83, SQ_84);
		m = pos.to_move(m16);
		// →　pos.to_move()で現在の手番側の駒ではないからMove::none()が返るか…。このテスト、意味ないな。
		tester.test("make_move(SQ_83, SQ_84) is pseudo_legal == false", pos.pseudo_legal(m) == false);
#endif

        // 88の先手の角を22に移動。これは途中に駒があって移動できないのでpseudo_legalではない。
        // (pseudo_legalは、その駒が移動できる(移動先の升にその駒の利きがある)ことを保証する)
        m16 = make_move16(SQ_88, SQ_22);
        m   = pos.to_move(m16);
        tester.test("make_move(SQ_88, SQ_22) is pseudo_legal == false",
                    pos.pseudo_legal(m, true) == false);
    }

    // attacks_bb() のテスト
    {
        auto section2 = tester.section("attacks_bb");
        hirate_init();
        tester.test("attacks_by<BLACK,PAWN>", pos.attacks_by<BLACK, PAWN>() == BB_Table::RANK6_BB);
        tester.test("attacks_by<WHITE,PAWN>", pos.attacks_by<WHITE, PAWN>() == BB_Table::RANK4_BB);
        tester.test("attacks_by<BLACK,KNIGHT>",
                    pos.attacks_by<BLACK, KNIGHT>()
                      == (Bitboard(SQ_97) | Bitboard(SQ_77) | Bitboard(SQ_37) | Bitboard(SQ_17)));
        tester.test("attacks_by<BLACK,GOLDS>",
                    pos.attacks_by<BLACK, GOLDS>()
                      == (goldEffect<BLACK>(SQ_69) | goldEffect<BLACK>(SQ_49)));
        tester.test("attacks_by<WHITE,GOLDS>",
                    pos.attacks_by<WHITE, GOLDS>()
                      == (goldEffect<WHITE>(SQ_61) | goldEffect<WHITE>(SQ_41)));
    }

#if defined(ENABLE_QUICK_DRAW)
	// ENABLE_QUICK_DRAWを定義していない時は、4回出現しないと千日手として扱わない。

    // 千日手検出のテスト
    {
        auto section2 = tester.section("is_repetition");

        std::deque<StateInfo> sis;

        // 4手前の局面に戻っているパターン
        BookTools::feed_position_string(pos, "startpos moves 5i5h 5a5b 5h5i 5b5a", sis);

        int  found_ply;
        auto rep = pos.is_repetition(16, found_ply);

        tester.test("REPETITION_DRAW", rep == REPETITION_DRAW && found_ply == 4);

        StateInfo s[512];
        // 初期局面から先手の飛車が46,後手玉が54に移動している局面。
        // ここから56飛(46)→44玉(54)→46飛(56)→54玉(44)で先手の反則負け
        pos_init("lnsg1gsnl/1r5b1/ppppppppp/4k4/9/5R3/PPPPPPPPP/1B7/LNSGKGSNL b - 1");

        m = pos.to_move(make_move16(SQ_46, SQ_56));
        pos.do_move(m, s[0]);
        m = pos.to_move(make_move16(SQ_54, SQ_44));
        pos.do_move(m, s[1]);
        m = pos.to_move(make_move16(SQ_56, SQ_46));
        pos.do_move(m, s[2]);
        m = pos.to_move(make_move16(SQ_44, SQ_54));
        pos.do_move(m, s[3]);

        // いま先手番であり、先手の反則負けが確定しているはず。
        auto draw_value = pos.is_repetition();
        tester.test("REPETITION_LOSE", draw_value == REPETITION_LOSE);

        // 初期局面から先手の飛車が56,後手玉が54に移動している局面。(王手がかかっていて後手番)
        // ここから44玉(54)→46飛(56)→54玉(44)→56飛(46)で(後手番において)先手の反則負け
        pos_init("lnsg1gsnl/1r5b1/ppppppppp/4k4/9/4R4/PPPPPPPPP/1B7/LNSGKGSNL w - 1");

        m = pos.to_move(make_move16(SQ_54, SQ_44));
        pos.do_move(m, s[0]);
        m = pos.to_move(make_move16(SQ_56, SQ_46));
        pos.do_move(m, s[1]);
        m = pos.to_move(make_move16(SQ_44, SQ_54));
        pos.do_move(m, s[2]);
        m = pos.to_move(make_move16(SQ_46, SQ_56));
        pos.do_move(m, s[3]);

        draw_value = pos.is_repetition();
        tester.test("REPETITION_WIN", draw_value == REPETITION_WIN);
    }
#endif

    // 入玉のテスト
    {
        auto section2 = tester.section("EnteringKing");

        {
            // 27点法の入玉可能点数 平手 : 先手=28,後手=27
            auto section3 = tester.section("EKR_27_POINT");

            hirate_init();
            pos.set_ekr(EKR_27_POINT);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);

            // 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=28,後手=27
            handi2_init();
            pos.set_ekr(EKR_27_POINT);

            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);
        }

        {
            // 24点法の入玉可能点数 平手 : 先手=31,後手=31
            auto section3 = tester.section("EKR_24_POINT");

            hirate_init();
            pos.set_ekr(EKR_24_POINT);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);

            // 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=31,後手=31
            handi2_init();
            pos.set_ekr(EKR_24_POINT);

			tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);
        }

        {
            // 27点法の入玉可能点数 平手 : 先手=28,後手=27
            auto section3 = tester.section("EKR_27_POINT_H");

            hirate_init();
            pos.set_ekr(EKR_27_POINT_H);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);

            // 2枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=17,下手(BLACK)=28
            handi2_init();
            pos.set_ekr(EKR_27_POINT_H);
            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 17);

            // 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=15,下手(BLACK)=28
            handi4_init();
            pos.set_ekr(EKR_27_POINT_H);
            tester.test("handi4",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 15);
        }

        {
            // 24点法の入玉可能点数 平手 : 先手=31,後手=31
            auto section3 = tester.section("EKR_24_POINT_H");

            hirate_init();
            pos.set_ekr(EKR_24_POINT_H);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);

            // 2枚落ち初期化 , 駒落ち対応なのでこの時 上手(WHITE)=21,下手(BLACK)=31
            handi2_init();
            pos.set_ekr(EKR_24_POINT_H);
            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 21);

            // 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=19,下手(BLACK)=31
            handi4_init();
            pos.set_ekr(EKR_24_POINT_H);
            tester.test("handi4",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 19);
        }
    }

    {
        // 指し手生成のテスト
        auto section2 = tester.section("GenMove");

        {
            // 23歩不成ができ、かつ、23歩不成では駒の捕獲にはならない局面。
            pos_init("lnsgk1snl/1r4g2/p1ppppb1p/6pP1/7R1/2P6/P2PPPP1P/1SG6/LN2KGSNL b BP2p 21");
            Move move1 = make_move(SQ_24, SQ_23, B_PAWN);
            Move move2 = make_move_promote(SQ_24, SQ_23, B_PAWN);

            Move move_buf[MAX_MOVES], *move_last;
            // move_bufからmove_lastのなかにmoveがあるかを探す。あればtrueを返す。
            auto find_move = [&](Move m) {
                for (Move* em = &move_buf[0]; em != move_last; ++em)
                    if (*em == m)
                        return true;
                return false;
            };

            bool all = true;

            move_last = generate<QUIETS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<NON_EVASIONS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<NON_EVASIONS_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS_ALL>(pos, move_buf);
            all &= find_move(
              move1);  // 歩の不成はこちらに含めることになった。(movegenの実装の修正が難しいので)
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS_ALL>(pos, move_buf);
            all &= !find_move(move1);  // 歩の不成はこちらには含まれていないので注意。
            all &= !find_move(move2);

            tester.test("pawn's unpromoted move", all);

            // 23角不成で5手詰め
            // https://github.com/yaneurao/YaneuraOu/issues/257
            pos_init("5B1n1/8k/6Rpp/9/9/9/1+p7/9/K8 b rb4g4s3n4l15p 1");
            // 23角不成(41)と23角成(41)
            move1 = make_move(SQ_41, SQ_23, B_BISHOP);
            move2 = make_move_promote(SQ_41, SQ_23, B_BISHOP);
            all   = true;

            move_last = generate<LEGAL_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS_ALL>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            tester.test("bishop's unpromoted move", all);
        }
    }
#if defined(USE_SEE)
    {
        // see_ge()のテスト
        auto      section = tester.section("see_ge");
        StateInfo s[512];

        // 平手初期化
        hirate_init();

        // see_geのしきい値がv以下の時だけtrueが返ってくるかをテストする。
        // つまりはsee値がvであるかをテストする関数。
        auto see_ge_th = [&](int v) {
            Value th     = Value(v);
            bool  all_ok = true;
            all_ok &= pos.see_ge(m, th);       // see_ge(m, th) == true
            all_ok &= !pos.see_ge(m, th + 1);  // see_ge(m,  1) == false
            all_ok &= pos.see_ge(m, th - 1);   // see_ge(m, -1) == true
            return all_ok;
        };

        // 76歩、34歩の局面を作る。
        m = pos.to_move(make_move16(SQ_77, SQ_76));
        pos.do_move(m, s[0]);
        m = pos.to_move(make_move16(SQ_33, SQ_34));
        pos.do_move(m, s[1]);
        // 22角成りの指し手について
        m = pos.to_move(make_move_promote16(SQ_88, SQ_22));
        // 角を取るが、see値は、同銀と取り返されて、駒の損得なし。

        tester.test("pos1move", see_ge_th(0));

        pos.do_move(m, s[2]);
        // 馬を取り返さずにあえて84歩
        m = pos.to_move(make_move16(SQ_83, SQ_24));
        pos.do_move(m, s[3]);

        // この局面で31馬は、同金とされて、(see値は)馬、銀の交換 = 馬を損して銀を得する
        m = pos.to_move(make_move16(SQ_22, SQ_31));
        tester.test("pos2move", see_ge_th(-Eval::HorseValue + Eval::SilverValue));

        // この局面で33角打ちは、同桂で同馬。(see値は)角損 + 桂得。
        m = pos.to_move(make_move_drop16(BISHOP, SQ_33));
        tester.test("pos2drop", see_ge_th(-Eval::BishopValue + Eval::KnightValue));

        // この局面で33馬は、同桂でタダ。(see値は)馬損。
        m = pos.to_move(make_move16(SQ_22, SQ_33));
        tester.test("pos2move", see_ge_th(-Eval::HorseValue));
    }
#endif

    {
        // null moveのテスト
        auto section = tester.section("nullmove");
        matsuri_init();
        StateInfo s[512];

        // null moveして、局面情報がおかしくならないかのテスト。
        pos.do_null_move(s[0]);
        tester.test("pos_is_ok()", pos.pos_is_ok());
    }

#if defined(USE_SFEN_PACKER)
    {
        // packed sfenのテスト
        auto section = tester.section("PackedSfen");

        vector<string> test_sfens = {
          "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
          "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
          "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
          "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
        };

        // packed by cshogi
        /*
			board = cshogi.Board()
			psfen = np.zeros(32, dtype=np.uint8)
			board.set_sfen(sfen)
			board.to_psfen(psfen)
			print(np.array2string(psfen, separator=', '))
		*/
        vector<PackedSfen> packed_sfens = {
          {89,  164, 81, 34, 12,  171, 68, 252, 44,  167, 68, 56, 94,  137, 240, 72,
           132, 87,  34, 60, 167, 68,  56, 86,  137, 248, 88, 70, 137, 48,  188, 126},
          {89, 164, 81,  34, 12, 171, 68,  252, 44,  167, 68, 56, 94,  137, 240, 72,
           4,  18,  225, 57, 37, 194, 177, 74,  196, 199, 50, 74, 132, 97,  191, 126},
          {89,  164, 81, 34,  88, 37,  226, 199, 41,  17,  188, 18, 129, 68,  120, 37,
           194, 115, 74, 132, 99, 149, 136, 143, 101, 148, 8,   67, 106, 107, 191, 126},
          {89,  36,  18, 1,  137, 128, 68,  64,  34, 144, 8,   175, 68,  120, 78,  137,
           112, 172, 18, 97, 25,  37,  194, 112, 30, 159, 251, 252, 166, 212, 218, 90}};

        bool success = true;
        for (size_t i = 0; i < test_sfens.size(); ++i)
        {
            auto  sfen        = test_sfens[i];
            auto& packed_sfen = packed_sfens[i];

            StateInfo si;
            pos.set(sfen, &si);

            PackedSfen ps;
            pos.sfen_pack(ps);

            // バイナリ列として一致するか。
            success &= ps == packed_sfen;

            // decodeで元のsfenになることは、このあとのランダムプレイヤーのテストで散々やっているから
            // ここでやる必要なし。
        }
        tester.test("handicapped sfen", success);
    }
#endif

#if defined(USE_PARTIAL_KEY)
    {
        // 部分hashkeyのテスト
        auto section = tester.section("PartialKey");
        {
            PRNG     my_rand(114514);
            Position pos2;

            bool ok = true;

            for (int i = 0; i < 1000; ++i)
            {
                StateInfo si[MAX_PLY];
                pos.set_hirate(&si[0]);
                for (int j = 1; j < MAX_PLY; ++j)
                {
                    MoveList<LEGAL_ALL> ml(pos);

                    // 指し手がない == 負け == 終了
                    if (ml.size() == 0)
                        break;

                    Move m = Move(ml.at(size_t(my_rand.rand(ml.size()))));
                    pos.do_move(m, si[j + 1]);

					// 📓 sfen経由でset()を呼び出す。この時、set()によってpartial keyが初期化される。
					//     差分更新したpartial keyと一致するかをテストする。

					auto      sfen = pos.sfen();
                    StateInfo si2;
                    pos2.set(sfen, &si2);

                    // clang-format off

                    // 部分keyが一致するか。
                    ok &= pos.state()->board_key     == pos2.state()->board_key;
                    ok &= pos.state()->hand_key      == pos2.state()->hand_key;
                    ok &= pos.state()->pawnKey       == pos2.state()->pawnKey;
                    ok &= pos.state()->nonPawnKey[0] == pos2.state()->nonPawnKey[0];
                    ok &= pos.state()->nonPawnKey[1] == pos2.state()->nonPawnKey[1];
                    ok &= pos.state()->materialKey   == pos2.state()->materialKey;
                    ok &= pos.state()->minorPieceKey == pos2.state()->minorPieceKey;

                    // clang-format on

                    //if (!ok)
                    //    tester.test("game " + to_string(i) + " ply = " + to_string(j), ok);
                }
            }
            tester.test("partial hash key", ok);
        }
    }
#endif

    {
        // それ以外のテスト
        auto section = tester.section("misc");
        {
            // 盤面の反転

            // 23歩不成ができ、かつ、23歩不成では駒の捕獲にはならない局面。
            pos_init("lnsgk1snl/1r4g2/p1ppppb1p/6pP1/7R1/2P6/P2PPPP1P/1SG6/LN2KGSNL b BP2p 21");
            auto flipped = pos.flipped_sfen();
            tester.test(
              "flip sfen",
              flipped == "lnsgk2nl/6gs1/p1pppp2p/6p2/1r7/1pP6/P1BPPPP1P/2G4R1/LNS1KGSNL w 2Pbp 21");
        }
    }

    {
        // 深いdepthのperftのテストが通っていれば、利きの計算、指し手生成はおおよそ間違っていないと言える。

        auto section2 = tester.section("Perft");

        {
            auto section3 = tester.section("hirate");
            hirate_init();
            const s64 p_nodes[] = {0, 30, 900, 25470, 719731, 19861490, 547581517};

            for (Depth d = 1; d <= 6; ++d)
            {
                u64 nodes = perft(pos, d);
                u64 pn    = p_nodes[d];
                tester.test("depth " + to_string(d) + " = " + to_string(pn),
                            nodes == pn && pos.pos_is_ok());
            }
        }

        {
            auto section3 = tester.section("matsuri");
            matsuri_init();

            const s64 p_nodes[] = {0, 207, 28684, 4809015, 516925165};

            for (Depth d = 1; d <= 4; ++d)
            {
                u64 nodes = perft(pos, d);
                u64 pn    = p_nodes[d];
                tester.test("depth " + to_string(d) + " = " + to_string(nodes),
                            nodes == pn && pos.pos_is_ok());
            }
        }
    }

    // ランダムプレイヤーでの対局によるテスト

    // packed sfenのtest
    auto extra_test1 = [&](Position& pos) {
#if defined(USE_SFEN_PACKER)
        PackedSfen ps;
        StateInfo  si;
        string     sfen     = pos.sfen();
        int        game_ply = pos.game_ply();
        pos.sfen_pack(ps);

        Position pos2;
        pos2.set_from_packed_sfen(ps, &si);
        string sfen2 = pos2.sfen(game_ply);

        return sfen == sfen2;
#else
        return true;
#endif
    };

    // 駒落ちのpacked sfenのテスト
    auto extra_test2 = [&](Position& pos) {
#if defined(USE_SFEN_PACKER)
        PackedSfen ps;
        StateInfo  si;
        string     sfen     = pos.sfen();
        int        game_ply = pos.game_ply();
        pos.sfen_pack(ps);

        Position pos2;
        pos2.set_from_packed_sfen(ps, &si);
        // ここから駒を5枚ほど落とす。
        int count = 0;
        for (auto sq : SQ)
        {
            auto pc = pos2.piece_on(sq);
            if (pc != NO_PIECE && type_of(pc) != KING)
            {
                pos2.board[sq] = NO_PIECE;  // 自分のclass内なので直接書き換えてしまう。
                if (++count >= 5)
                    break;
            }
        }
        string sfen2 = pos2.sfen(game_ply);
        pos2.sfen_pack(ps);  // 駒落ちのpacked sfenができた。

        Position pos3;
        pos3.set_from_packed_sfen(ps, &si);

        string sfen3 = pos3.sfen(game_ply);

        return sfen2 == sfen3;
#else
        return true;
#endif
    };

    {
        // 対局回数→0ならskip
        s64 random_player_loop = tester.options["random_player_loop"];
        if (random_player_loop)
        {
            auto section2 = tester.section("GamesOfRandomPlayer");

            // seed固定乱数(再現性ある乱数)
            PRNG      my_rand(114514);
            StateInfo s[512];

            for (s64 i = 0; i < random_player_loop; ++i)
            {
                // 平手初期化
                hirate_init();
                bool fail = false;

                // 512手目まで
                for (int ply = 0; ply < 512; ++ply)
                {
                    MoveList<LEGAL_ALL> ml(pos);

                    // 指し手がない == 負け == 終了
                    if (ml.size() == 0)
                        break;

                    Move m = Move(ml.at(size_t(my_rand.rand(ml.size()))));

                    pos.do_move(m, s[ply]);

                    if (!pos.pos_is_ok() || !extra_test1(pos) || !extra_test2(pos))
                        fail = true;
                }

                // 今回のゲームのなかでおかしいものがなかったか
                tester.test(string("game ") + to_string(i + 1), !fail);
            }
        }
    }
}


// ----------------------------------
//         明示的な実体化
// ----------------------------------
template bool Position::pseudo_legal_s<false>(const Move m) const;
template bool Position::pseudo_legal_s< true>(const Move m) const;

template void Position::do_move(Move m, StateInfo& newSt, bool givesCheck, const TranspositionTable* tt);
template void Position::do_move(Move m, StateInfo& newSt, bool givesCheck, const void* tt);

template void Position::do_null_move(StateInfo& st, const TranspositionTable& tt);
template void Position::do_null_move(StateInfo& st, const int& tt);

} // namespace YaneuraOu
