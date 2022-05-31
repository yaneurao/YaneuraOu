#include "position.h"
#include "misc.h"
#include "tt.h"
#include "thread.h"
#include "mate/mate.h"
#include "book/book.h"
#include "testcmd/unit_test.h"

#include <iostream>
#include <sstream>
#include <cstring> // std::memset()

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_NNUE)
#include "eval/evaluate_common.h"
#endif

using namespace std;
using namespace Effect8;

std::string SFEN_HIRATE = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

// 局面のhash keyを求めるときに用いるZobrist key
namespace Zobrist {
	HASH_KEY zero;							// ゼロ(==0)
	HASH_KEY side;							// 手番(==1)
	HASH_KEY psq[SQ_NB_PLUS1][PIECE_NB];	// 駒pcが盤上sqに配置されているときのZobrist Key
	HASH_KEY hand[COLOR_NB][PIECE_HAND_NB];	// c側の手駒prが一枚増えるごとにこれを加算するZobristKey
	HASH_KEY depth[MAX_PLY];				// 深さも考慮に入れたHASH KEYを作りたいときに用いる(実験用)
}

// ----------------------------------
//           CheckInfo
// ----------------------------------

// 王手情報の初期化
template <bool doNullMove , Color Us>
void Position::set_check_info(StateInfo* si) const {

	//: si->blockersForKing[WHITE] = slider_blockers(pieces(BLACK), square<KING>(WHITE),si->pinners[WHITE]);
	//: si->blockersForKing[BLACK] = slider_blockers(pieces(WHITE), square<KING>(BLACK),si->pinners[BLACK]);

	// ↓Stockfishのこの部分の実装、将棋においては良くないので、以下のように変える。
	// ※　将棋においては駒の動きが上下対称ではないので手番を引数で渡す必要がある。

	if (!doNullMove)
	{
		// null moveのときは前の局面でこの情報は設定されているので更新する必要がない。
		si->blockersForKing[WHITE] = slider_blockers(BLACK, king_square(WHITE), si->pinners[WHITE]);
		si->blockersForKing[BLACK] = slider_blockers(WHITE, king_square(BLACK), si->pinners[BLACK]);
	}

	constexpr Color Them = ~Us;

	Square ksq = king_square(Them);

	// 駒種Xによって敵玉に王手となる升のbitboard

	// 歩であれば、自玉に敵の歩を置いたときの利きにある場所に自分の歩があればそれは敵玉に対して王手になるので、
	// そういう意味で(ksq,them)となっている。

	Bitboard occ = pieces();

	// この指し手が二歩でないかは、この時点でテストしない。指し手生成で除外する。なるべくこの手のチェックは遅延させる。
	si->checkSquares[PAWN]   = pawnEffect<Them>  (ksq);
	si->checkSquares[KNIGHT] = knightEffect<Them>(ksq);
	si->checkSquares[SILVER] = silverEffect<Them>(ksq);
	si->checkSquares[BISHOP] = bishopEffect      (ksq, occ);
	si->checkSquares[ROOK]   = rookEffect        (ksq, occ);
	si->checkSquares[GOLD]   = goldEffect<Them>  (ksq);

	// 香で王手になる升は利きを求め直さずに飛車で王手になる升を香のstep effectでマスクしたものを使う。
	si->checkSquares[LANCE]  = si->checkSquares[ROOK] & lanceStepEffect<Them>(ksq);

	// 王を移動させて直接王手になることはない。それは自殺手である。
	si->checkSquares[KING]   = Bitboard(ZERO);

	// 成り駒。この初期化は馬鹿らしいようだが、gives_check()は指し手ごとに呼び出されるので、その処理を軽くしたいので
	// ここでの初期化は許容できる。(このコードはdo_move()に対して1回呼び出されるだけなので)
	si->checkSquares[PRO_PAWN]   = si->checkSquares[GOLD];
	si->checkSquares[PRO_LANCE]  = si->checkSquares[GOLD];
	si->checkSquares[PRO_KNIGHT] = si->checkSquares[GOLD];
	si->checkSquares[PRO_SILVER] = si->checkSquares[GOLD];
	si->checkSquares[HORSE]      = si->checkSquares[BISHOP] | kingEffect(ksq);
	si->checkSquares[DRAGON]     = si->checkSquares[ROOK]   | kingEffect(ksq);
}

// ----------------------------------
//       Zorbrist keyの初期化
// ----------------------------------

void Position::init() {
	PRNG rng(20151225); // 開発開始日 == 電王トーナメント2015,最終日

	// 手番としてbit0を用いる。それ以外はbit0を使わない。これをxorではなく加算して行ってもbit0は汚されない。
	SET_HASH(Zobrist::side, 1, 0, 0, 0);
	SET_HASH(Zobrist::zero, 0, 0, 0, 0);

	// 64bit hash keyは256bit hash keyの下位64bitという解釈をすることで、256bitと64bitのときとでhash keyの下位64bitは合致するようにしておく。
	// これは定跡DBなどで使うときにこの性質が欲しいからである。
	// またpc==NO_PIECEのときは0であることを保証したいのでSET_HASHしない。
	// psqは、C++の規約上、事前にゼロであることは保証される。
	for (auto pc : Piece())
		for (auto sq : SQ)
			if (pc)
				SET_HASH(Zobrist::psq[sq][pc], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

	// またpr==NO_PIECEのときは0であることを保証したいのでSET_HASHしない。
	for (auto c : COLOR)
		for (PieceType pr = NO_PIECE_TYPE; pr < PIECE_HAND_NB; ++pr)
			if (pr)
				SET_HASH(Zobrist::hand[c][pr], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

	for (int i = 0; i < MAX_PLY; ++i)
		SET_HASH(Zobrist::depth[i], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());
}

// depthに応じたZobrist Hashを得る。depthを含めてhash keyを求めたいときに用いる。
HASH_KEY DepthHash(int depth) { return Zobrist::depth[depth]; }

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

// sfen文字列で盤面を設定する
void Position::set(std::string sfen , StateInfo* si , Thread* th)
{
	std::memset(this, 0, sizeof(Position));

	// 局面をrootより遡るためには、ここまでの局面情報が必要で、それは引数のsiとして渡されているという解釈。
	// ThreadPool::start_thinking()では、
	// ここをいったんゼロクリアしたのちに、呼び出し側で、そのsiを復元することにより、局面を遡る。
	std::memset(si, 0, sizeof(StateInfo));
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
	bool promote = false;
	size_t idx;

#if defined (USE_EVAL_LIST)
	// evalListのclear。上でmemsetでゼロクリアしたときにクリアされているが…。
	evalList.clear();

	// PieceListを更新する上で、どの駒がどこにあるかを設定しなければならないが、
	// それぞれの駒をどこまで使ったかのカウンター
	PieceNumber piece_no_count[KING] = { PIECE_NUMBER_ZERO,PIECE_NUMBER_PAWN,PIECE_NUMBER_LANCE,PIECE_NUMBER_KNIGHT,
	  PIECE_NUMBER_SILVER, PIECE_NUMBER_BISHOP, PIECE_NUMBER_ROOK,PIECE_NUMBER_GOLD };

	// 先手玉のいない詰将棋とか、駒落ちに対応させるために、存在しない駒はすべてBONA_PIECE_ZEROにいることにする。
	// 上のevalList.clear()で、ゼロクリアしているので、それは達成しているはず。
#endif

	kingSquare[BLACK] = kingSquare[WHITE] = SQ_NB;

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
			put_piece(sq, pc);

#if defined (USE_EVAL_LIST)
			PieceNumber piece_no =
				(idx == B_KING) ? PIECE_NUMBER_BKING : // 先手玉
				(idx == W_KING) ? PIECE_NUMBER_WKING : // 後手玉
				piece_no_count[raw_type_of(Piece(idx))]++; // それ以外
			evalList.put_piece(piece_no, sq, pc); // sqの升にpcの駒を配置する
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

	// --- 手番

	ss >> token;
	sideToMove = (token == 'w' ? WHITE : BLACK);
	ss >> token; // 手番と手駒とを分かつスペース

	// --- 手駒

	hand[BLACK] = hand[WHITE] = (Hand)0;
	int ct = 0;
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
			add_hand(hand[color_of(Piece(idx))], type_of(Piece(idx)), ct);

			// FV38などではこの個数分だけpieceListに突っ込まないといけない。
			for (int i = 0; i < ct; ++i)
			{
				PieceType rpc = raw_type_of(Piece(idx));

#if defined (USE_EVAL_LIST)
				PieceNumber piece_no = piece_no_count[rpc]++;
				ASSERT_LV1(is_ok(piece_no));
				evalList.put_piece(piece_no, color_of(Piece(idx)), rpc, i);
#endif
			}
			ct = 0;
		}
	}

	// --- 手数(平手の初期局面からの手数)

	// gamePlyとして将棋所では(検討モードなどにおいて)ここで常に1が渡されている。
	// 検討モードにおいても棋譜上の手数を渡して欲しい気がするし、棋譜上の手数がないなら0を渡して欲しい気はする。
	// ここで渡されてきた局面をもとに探索してその指し手を定跡DBに登録しようとするときに、ここの手数が不正確であるのは困る。
	gamePly = 0;
	ss >> std::skipws >> gamePly;

	// --- StateInfoの更新

	set_state(st);

	// 現局面で王手がかかっているならst->continuous_check[them] = 1にしないと
	// 連続王手の千日手の判定が不正確な気がするが、どのみち2回目の出現を負け扱いしているのでまあいいか..

	// --- effect

#if defined (LONG_EFFECT_LIBRARY)
  // 利きの全計算による更新
	LongEffect::calc_effect(*this);
#endif

	// --- evaluate

#if defined (USE_PIECE_VALUE)
	st->materialValue = Eval::material(*this);
#endif

	Eval::compute_eval(*this);

	// --- 入玉の駒点の設定

	update_entering_point();

	// --- validation

#if ASSERT_LV >= 3
  // これassertにしてしまうと、先手玉のいない局面や駒落ちの局面で落ちて困る。
	if (!is_ok(*this))
		std::cout << "info string Illigal Position?" << endl;
#endif

	thisThread = th;

}

// 局面のsfen文字列を取得する。
// Position::set()の逆変換。
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
	ss << (found ? " " : "- ");

	// --- 初期局面からの手数
	ss << gamePly_;

	return ss.str();
}

void Position::set_state(StateInfo* si) const {

	// --- bitboard

	// この局面で自玉に王手している敵駒
	st->checkersBB = attackers_to(~sideToMove, king_square(sideToMove));

	// 王手情報の初期化
	set_check_info<false>(si);

	// --- hash keyの計算
	si->board_key_ = sideToMove == BLACK ? Zobrist::zero : Zobrist::side;
	si->hand_key_ = Zobrist::zero;
	for (auto sq : pieces())
	{
		auto pc = piece_on(sq);
		si->board_key_ += Zobrist::psq[sq][pc];
	}
	for (auto c : COLOR)
		for (PieceType pr = PAWN; pr < PIECE_HAND_NB; ++pr)
			si->hand_key_ += Zobrist::hand[c][pr] * (int64_t)hand_count(hand[c], pr); // 手駒はaddにする(差分計算が楽になるため)

	// --- hand
	si->hand = hand[sideToMove];

}

// put_piece(),remove_piece(),xor_piece()を用いたあとに呼び出す必要がある。
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
#include <stack>

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


// ----------------------------------
//      ある升へ利いている駒など
// ----------------------------------

// Position::slider_blockers() は、c側の長い利きを持つ駒(sliders)から、升sへの利きを
// 遮っている先後の駒の位置をBitboardで返す。ただし、２重に遮っている場合はそれらの駒は返さない。
// もし、この関数のこの返す駒を取り除いた場合、升sに対してsliderによって利きがある状態になる。
// 升sにある玉に対してこの関数を呼び出した場合、それはpinされている駒と両王手の候補となる駒である。
// また、升sにある玉は~c側のKINGであるとする。

Bitboard Position::slider_blockers(Color c, Square s , Bitboard& pinners) const {

	Bitboard blockers(ZERO);

	// pinnersは返し値。
	pinners = Bitboard(ZERO);
	
	// cが与えられていないと香の利きの方向を確定させることが出来ない。
	// ゆえに将棋では、この関数は手番を引数に取るべき。(チェスとはこの点において異なる。)

	// snipersとは、pinされている駒が取り除かれたときに升sに利きが発生する大駒である。
	Bitboard snipers =
		( (pieces(ROOK_DRAGON)  & rookStepEffect(s))
		| (pieces(BISHOP_HORSE) & bishopStepEffect(s))
		// 香に関しては攻撃駒が先手なら、玉より下側をサーチして、そこにある先手の香を探す。
		| (pieces(LANCE) & lanceStepEffect(~c, s))
		) & pieces(c);

	//Bitboard occupancy = pieces() ^ snipers;

	// ↑このStockfishの元のコード、snipersを除いた盤上の駒で考えているが、
	// ^王 歩 角 飛
	// このような状況で飛車に対して角を取り除いてから敵玉への射線を考えるので、
	// 歩がslider_blocker扱いになってしまう。つまり、このコードは間違っているのでは？
	
	while (snipers)
	{
		Square sniperSq = snipers.pop();
		Bitboard b = between_bb(s, sniperSq) & pieces() /* occupancy */;

		// snipperと玉との間にある駒が1個であるなら。
		if (b && !b.more_than_one())
		{
			blockers |= b;
			if (b & pieces(~c))
				// sniperと玉に挟まれた駒が玉と同じ色の駒であるなら、pinnerに追加。
				pinners |= sniperSq;
		}
	}
	return blockers;
}


// sに利きのあるc側の駒を列挙する。先後両方。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
// sq == SQ_NBでの呼び出しは合法。Bitboard(ZERO)が返る。
Bitboard Position::attackers_to(Square sq, const Bitboard& occ) const
{
	ASSERT_LV3(sq <= SQ_NB);

	// sqの地点に敵駒ptをおいて、その利きに自駒のptがあればsqに利いているということだ。
	return
		// 先手の歩・桂・銀・金・HDK
		((    (pawnEffect  <WHITE>(sq) & pieces(PAWN)        )
			| (knightEffect<WHITE>(sq) & pieces(KNIGHT)      )
			| (silverEffect<WHITE>(sq) & pieces(SILVER_HDK)  )
			| (goldEffect  <WHITE>(sq) & pieces(GOLDS_HDK)   )
			) & pieces<BLACK>())
		|

		// 後手の歩・桂・銀・金・HDK
		((    (pawnEffect  <BLACK>(sq) & pieces(PAWN)        )
			| (knightEffect<BLACK>(sq) & pieces(KNIGHT)      )
			| (silverEffect<BLACK>(sq) & pieces(SILVER_HDK)  )
			| (goldEffect  <BLACK>(sq) & pieces(GOLDS_HDK)   )
			) & pieces<WHITE>())

		// 先後の角・飛・香
		| (bishopEffect(sq, occ) & pieces(BISHOP_HORSE) )
		| (rookEffect(sq, occ) & (
			   pieces(ROOK_DRAGON)
			| (pieces(BLACK , LANCE) & lanceStepEffect<WHITE>(sq))
			| (pieces(WHITE , LANCE) & lanceStepEffect<BLACK>(sq))
			// 香も、StepEffectでマスクしたあと飛車の利きを使ったほうが香の利きを求めなくて済んで速い。
			));
}

// 打ち歩詰め判定に使う。王に打ち歩された歩の升をpawn_sqとして、c側(王側)のpawn_sqへ利いている駒を列挙する。香が利いていないことは自明。
inline Bitboard Position::attackers_to_pawn(Color c, Square pawn_sq) const
{
	ASSERT_LV3(is_ok(c) && pawn_sq <= SQ_NB);

	Color them = ~c;
	const Bitboard& occ = pieces();

	// 馬と龍
	const Bitboard bb_hd = kingEffect(pawn_sq) & pieces(HORSE,DRAGON);
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
	ASSERT_LV3(is_ok(m));

	// 移動先
	const Square to = to_sq(m);

	// 駒打ち・移動する指し手どちらであってもmove_piece_after(m)で移動後の駒が取得できるので
	// 直接王手の処理は共通化できる。
	if (st->checkSquares[type_of(moved_piece_after(m))] & to)
		return true;

	// -- 移動する指し手ならば、これで開き王手になるかどうかの判定が必要。

	// 移動元
	const Square from = from_sq(m);

	// 開き王手になる駒の候補があるとして、fromにあるのがその駒で、fromからtoは玉と直線上にないなら
	// 前提条件より、fromにあるのが自駒であることは確定しているので、pieces(sideToMove)は不要。
	return  !is_drop(m)
		&& (((blockers_for_king(~sideToMove) /*& pieces(sideToMove)*/) & from)
		&&  !aligned(from, to, king_square(~sideToMove)));
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
	ASSERT_LV3(pawnEffect(us, to) == Bitboard(king_square(~us)));

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
	Square sq_king = king_square(~us);

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
		|| ((pawnEffect(us, to) == Bitboard(king_square(~us)) && !legal_drop(to)))); // 打ち歩詰め
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
// is_ok(m)==falseの時、すなわち、m == MOVE_WINやMOVE_NONEのような時に
// Position::to_move(m) == mは保証されており、この時、本関数pseudo_legal(m)がfalseを返すことは保証する。
// 
// Options["GenerateAllLegalMoves"]を反映させる。
// ↑これがtrueならば、歩の不成も合法手扱い。
bool Position::pseudo_legal(const Move m) const
{
	return Search::Limits.generate_all_legal_moves ? pseudo_legal_s<true>(m) : pseudo_legal_s<false>(m);
}

// ※　mがこの局面においてpseudo_legalかどうかを判定するための関数。
template <bool All>
bool Position::pseudo_legal_s(const Move m) const {

	const Color us = sideToMove;
	const Square to = to_sq(m); // 移動先

	if (is_drop(m))
	{
		const PieceType pr = move_dropped_piece(m);
		// 置換表から取り出してきている以上、一度は指し手生成ルーチンで生成した指し手のはずであり、
		// KING打ちのような値であることはないものとする。

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
			if (!(between_bb(checksq, king_square(us)) & to))
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

		const Square from = from_sq(m);
		const Piece pc = piece_on(from);

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
		if (is_promote(m))
		{
			// --- 成る指し手

			// 成れない駒の成りではないことを確かめないといけない。
			if (is_promoted_piece(pc))
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
				if (!((between_bb(checkers().pop(), king_square(us)) | checkers()) & to))
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
	if (is_drop(m))
		// 打ち歩詰めは指し手生成で除外されている。
		return true;
	else
	{
		Color us = sideToMove;
		Square from = from_sq(m);

		ASSERT_LV5(color_of(piece_on(from_sq(m))) == us);
		ASSERT_LV5(piece_on(king_square(us)) == make_piece(us, KING));

		// もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
		if (type_of(piece_on(from)) == KING)
			return !effected_to(~us, to_sq(m), from);

		// blockers_for_king()は、pinされている駒(自駒・敵駒)を表現するが、fromにある駒は自駒であることは
		// わかっているのでこれで良い。
		return !(blockers_for_king(us) & from)
			 || aligned(from, to_sq(m), king_square(us));
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
	if (!is_promote(m))
		return true;

	Color us = sideToMove;
	Square from = from_sq(m);
	Square to   =   to_sq(m);

	// 移動元か移動先が敵陣でなければ成れる条件を満たしていない。
	return enemy_field(us) & (Bitboard(from) | Bitboard(to));
}

// 置換表から取り出したMoveを32bit化する。
Move Position::to_move(Move16 m16) const
{
	//		ASSERT_LV3(is_ok(m));
	// 置換表から取り出した値なので m==MOVE_NONE(0)である可能性があり、ASSERTは書けない。

	// 上位16bitは0でなければならない
	//      ASSERT_LV3((m >> 16) == 0);

	Move m = (Move)m16.to_u16();

	// MOVE_NULLの可能性はないはずだが、MOVE_WINである可能性はある。
	// それはそのまま返す。(MOVE_WINの機会はごくわずかなのでこれのために
	// このチェックが探索時に起きるのは少し馬鹿らしい気もする。
	// どうせ探索時はlegalityのチェックに引っかかり無視されるわけで…)
	if (!is_ok(m))
		return m;

	if (is_drop(m))
		return Move(u16(m) + ((u32)make_piece(side_to_move(), move_dropped_piece(m)) << 16));
		// また、move_dropped_piece()はおかしい値になっていないことは保証されている(置換表に自分で書き出した値のため)
		// これにより、配列境界の外側に書き出してしまう心配はない。

	// 移動元にある駒が、現在の手番の駒であることを保証する。
	// 現在の手番の駒でないか、駒がなければMOVE_NONEを返す。
	Piece moved_piece = piece_on(from_sq(m));
	if (color_of(moved_piece) != side_to_move() || moved_piece == NO_PIECE)
		return MOVE_NONE;

	// promoteで成ろうとしている駒は成れる駒であることを保証する。
	if (is_promote(m))
	{
		// 成駒や金・玉であるなら、これ以上成れない。これは非合法手である。
		if (is_promoted_piece(moved_piece))
			return MOVE_NONE;

		return Move(u16(m) + ((u32)(make_promoted_piece(moved_piece) << 16)));
	}

	// 通常の移動
	return Move(u16(m) + ((u32)moved_piece << 16));
}


// ----------------------------------
//      局面を進める/戻す
// ----------------------------------

// 指し手で盤面を1手進める。
template <Color Us>
void Position::do_move_impl(Move m, StateInfo& new_st, bool givesCheck)
{
	// MOVE_NONEはもちろん、MOVE_NULL , MOVE_RESIGNなどお断り。
	ASSERT_LV3(is_ok(m));

	ASSERT_LV3(&new_st != st);

	constexpr Color Them = ~Us;

	// 探索ノード数 ≒do_move()の呼び出し回数のインクリメント。
	thisThread->nodes.fetch_add(1, std::memory_order_relaxed);

	//std::cout << *this << m << std::endl;

	// ----------------------
	//  StateInfoの更新
	// ----------------------

	// hash key

	// 現在の局面のhash keyはこれで、これを更新していき、次の局面のhash keyを求めてStateInfo::key_に格納。
	HASH_KEY k = st->board_key_ ^ Zobrist::side;
	HASH_KEY h = st->hand_key_;

	// StateInfoの構造体のメンバーの上からkeyのところまでは前のを丸ごとコピーしておく。
	// undo_moveで戻すときにこの部分はundo処理が要らないので細かい更新処理が必要なものはここに載せておけばundoが速くなる。

	// std::memcpy(&new_st, st, offsetof(StateInfo, checkersBB));
	// 将棋ではこの処理、要らないのでは…。

	// StateInfoを遡れるようにpreviousを設定しておいてやる。
	StateInfo* prev;
	new_st.previous = prev = st;
	st = &new_st;

	// --- 手数がらみのカウンターのインクリメント

	// 厳密には、これはrootからの手数ではなく、初期盤面からの手数ではあるが。
	++gamePly;
	
	// st->previousで遡り可能な手数カウンタ
	st->pliesFromNull = prev->pliesFromNull + 1;

	// 評価値の差分計算用の初期化

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
	st->sum.p[0][0] = VALUE_NOT_EVALUATED;
#endif
#if defined(EVAL_NNUE)
	st->accumulator.computed_accumulation = false;
	st->accumulator.computed_score = false;
#endif

#if defined(USE_BOARD_EFFECT_PREV)
	// NNUE-HalfKPE9
	// 現局面のboard_effectをコピー
	std::memcpy(board_effect_prev, board_effect, sizeof(board_effect));
#endif

	// 直前の指し手を保存するならばここで行なう。

#if defined(KEEP_LAST_MOVE)
	st->lastMove = m;
	st->lastMovedPieceType = is_drop(m) ? (PieceType)from_sq(m) : type_of(piece_on(from_sq(m)));
#endif

	// ----------------------
	//    盤面の更新処理
	// ----------------------

	// 移動先の升
	Square to = to_sq(m);
	ASSERT_LV2(is_ok(to));

#if defined (USE_PIECE_VALUE)
	// 駒割りの差分計算用
	int materialDiff;
#endif

#if defined (USE_EVAL_LIST)
	auto& dp = st->dirtyPiece;
#endif

	if (is_drop(m))
	{
		// --- 駒打ち

		// 移動先の升は空のはず
		ASSERT_LV2(piece_on(to) == NO_PIECE);

		Piece pc = moved_piece_after(m);
		PieceType pr = raw_type_of(pc);
		ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

		// Zobrist keyの更新
		h -= Zobrist::hand[Us][pr];
		k += Zobrist::psq[to][pc];

		// なるべく早い段階でのTTに対するprefetch
		// 駒打ちのときはこの時点でTT entryのアドレスが確定できる
		const HASH_KEY key = k + h;
		prefetch(TT.first_entry(key));
#if defined(USE_EVAL_HASH)
		Eval::prefetch_evalhash(hash_key_to_key(key));
#endif

		put_piece(to, pc);

		// 打駒した駒に関するevalListの更新。
#if defined (USE_EVAL_LIST)
		PieceNumber piece_no = piece_no_of(Us, pr);
		ASSERT_LV3(is_ok(piece_no));

		// KPPの差分計算のために移動した駒をStateInfoに記録しておく。
		dp.dirty_num = 1; // 動いた駒は1個
		dp.pieceNo[0] = piece_no;
		dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no);
		evalList.put_piece(piece_no , to, pc);
		dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no);
#endif

		// piece_no_of()のときにこの手駒の枚数を参照するのであとで更新。
		sub_hand(hand[Us], pr);
		
		// 王手している駒のbitboardを更新する。
		// 駒打ちなのでこの駒で王手になったに違いない。駒打ちで両王手はありえないので王手している駒はいまtoに置いた駒のみ。
		if (givesCheck)
		{
			st->checkersBB = Bitboard(to);
			st->continuousCheck[Us] = prev->continuousCheck[Us] + 2;

			// Stockfishのコードは、ここのコード、" += 2 "になっているが、
			// やねうら王ではStateInfoのmemcpy()をしないことにしたので
			// 前ノードの値に対して、" + 2 "しないといけない。

		} else {
			st->checkersBB = Bitboard(ZERO);
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

	} else {

		// -- 駒の移動
		Square from = from_sq(m);
		ASSERT_LV2(is_ok(from));

		// 移動させる駒
		Piece moved_pc = piece_on(from);
		ASSERT_LV2(moved_pc != NO_PIECE);

		// 移動先に駒の配置
		// もし成る指し手であるなら、成った後の駒を配置する。
		Piece moved_after_pc = moved_piece_after(m);

#if defined (USE_PIECE_VALUE)
		materialDiff = is_promote(m) ? Eval::ProDiffPieceValue[moved_pc] : 0;
#endif

		// 移動先の升にある駒
		Piece to_pc = piece_on(to);
		if (to_pc != NO_PIECE)
		{
			// --- capture(駒の捕獲)

#if defined(LONG_EFFECT_LIBRARY)
	  // 移動先で駒を捕獲するときの利きの更新
			LongEffect::update_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc, to_pc);
#endif

			// 玉を取る指し手が実現することはない。この直前の局面で玉を逃げる指し手しか合法手ではないし、
			// 玉を逃げる指し手がないのだとしたら、それは詰みの局面であるから。

			ASSERT_LV1(type_of(to_pc) != KING);

			PieceType pr = raw_type_of(to_pc);

			// 捕獲した駒に関するevalListの更新
#if defined (USE_EVAL_LIST)
			// このPieceNumberの駒が手駒に移動したのでEvalListのほうを更新しておく。
			PieceNumber piece_no = piece_no_of(to);
			ASSERT_LV3(is_ok(piece_no));
			dp.dirty_num = 2; // 動いた駒は2個
			dp.pieceNo[1] = piece_no;
			dp.changed_piece[1].old_piece = evalList.bona_piece(piece_no);
			evalList.put_piece(piece_no, Us, pr, hand_count(hand[Us], pr));
			dp.changed_piece[1].new_piece = evalList.bona_piece(piece_no);
#endif

			// 駒取りなら現在の手番側の駒が増える。
			add_hand(hand[Us], pr);

			// 捕獲される駒の除去
			remove_piece(to);

			// 捕獲された駒が盤上から消えるので局面のhash keyを更新する
			k -= Zobrist::psq[to][to_pc];
			h += Zobrist::hand[Us][pr];

			// 捕獲した駒をStateInfoに保存しておく。(undo_moveのため)
			st->capturedPiece = to_pc;

#if defined (USE_PIECE_VALUE)
			// 評価関数で使う駒割りの値も更新
			materialDiff += Eval::CapturePieceValue[to_pc];
#endif

		} else {
			// 駒を取らない指し手

			st->capturedPiece = NO_PIECE;

#if defined (LONG_EFFECT_LIBRARY)
			// 移動先で駒を捕獲しないときの利きの更新
			LongEffect::update_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif
#if defined (USE_EVAL_LIST)
			dp.dirty_num = 1; // 動いた駒は1個
#endif
		}

#if defined (USE_EVAL_LIST)
		// 移動元にあった駒のpiece_noを得る
		PieceNumber piece_no2 = piece_no_of(from);
		dp.pieceNo[0] = piece_no2;
		dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no2);
#endif

		// 移動元の升からの駒の除去
		remove_piece(from);
		// 移動先の升に駒を配置
		put_piece(to, moved_after_pc);

#if defined (USE_EVAL_LIST)
		evalList.put_piece(piece_no2, to, moved_after_pc);
		dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no2);
#endif

		// 王を移動させる手であるなら、kingSquareを更新しておく。
		// これを更新しておかないとking_square()が使えなくなってしまう。
		// 王は駒打できないのでdropの指し手に含まれていることはないから
		// dropのときにはkingSquareを更新する必要はない。
		if (type_of(moved_pc) == KING)
			kingSquare[Us] = to;

		// fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
		k -= Zobrist::psq[from][moved_pc];
		k += Zobrist::psq[to][moved_after_pc];

		// 駒打ちでないときはprefetchはこの時点まで延期される。
		const HASH_KEY key = k + h;
		prefetch(TT.first_entry(key));
#if defined(USE_EVAL_HASH)
		Eval::prefetch_evalhash(hash_key_to_key(key));
#endif

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
			const Square ksq = king_square(Them);
			// pos->discovered_check_candidates()で取得したいが、もうstを更新してしまっているので出来ないので
			// prevSt->blockersForKing[~Us] & pieces(Us)と愚直に書く。
			// また、pieces(Us)のうち今回移動させる駒は、実はすでに移動させてしまっているので、fromと書く。

			if (discovered(from, to, ksq, prevSt->blockersForKing[Them] & from))
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

				st->checkersBB |= directEffect(from, direct_of(ksq, from), pieces()) & pieces<Us>();
			}

			// 差分更新したcheckersBBが正しく更新されているかをテストするためのassert
			ASSERT_LV3(st->checkersBB == attackers_to<Us>(king_square(Them)));
#else
			// 差分更新しないとき用。(デバッグ等の目的で用いる)
			st->checkersBB = attackers_to<Us>(king_square(Them));
#endif
			st->continuousCheck[Us] = prev->continuousCheck[Us] + 2;

		} else {

			st->checkersBB = Bitboard(ZERO);
			st->continuousCheck[Us] = 0;
		}
	}
	// 相手番のほうは関係ないので前ノードの値をそのまま受け継ぐ。
	st->continuousCheck[Them] = prev->continuousCheck[Them];

#if defined (USE_PIECE_VALUE)
	st->materialValue = (Value)(st->previous->materialValue + (Us == BLACK ? materialDiff : -materialDiff));
	//ASSERT_LV5(st->materialValue == Eval::material(*this));
#endif

	// 相手番に変更する。
	sideToMove = Them;

	// 更新されたhash keyをStateInfoに書き戻す。
	st->board_key_ = k;
	st->hand_key_ = h;

	st->hand = hand[Them];

	// このタイミングで王手関係の情報を更新しておいてやる。
	set_check_info<false>(st);

	//ASSERT_LV5(evalList.is_valid(*this));

	//state()->dirtyPiece.do_update(evalList);
	//evalList.is_valid(*this);
}

// ある指し手を指した後のhash keyを返す。
Key Position::key_after(Move m) const {
	return hash_key_to_key(hash_key_after(m));
}

// ある指し手を指した後のhash keyを返す。
HASH_KEY Position::hash_key_after(Move m) const {

	Color Us = side_to_move();
	auto k = st->board_key_ ^ Zobrist::side;
	auto h = st->hand_key_;

	// 移動先の升
	Square to = to_sq(m);
	ASSERT_LV2(is_ok(to));

	if (is_drop(m))
	{
		// --- 駒打ち
		PieceType pr = move_dropped_piece(m);
		ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

		Piece pc = make_piece(Us, pr);

		// Zobrist keyの更新
		h -= Zobrist::hand[Us][pr];
		k += Zobrist::psq[to][pc];
	}
	else
	{
		// -- 駒の移動
		Square from = from_sq(m);
		ASSERT_LV2(is_ok(from));

		// 移動させる駒
		Piece moved_pc = piece_on(from);
		ASSERT_LV2(moved_pc != NO_PIECE);

		// 移動先に駒の配置
		// もし成る指し手であるなら、成った後の駒を配置する。
		Piece moved_after_pc = is_promote(m) ? make_promoted_piece(moved_pc) : moved_pc;

		// 移動先の升にある駒
		Piece to_pc = piece_on(to);
		if (to_pc != NO_PIECE)
		{
			PieceType pr = raw_type_of(to_pc);

			// 捕獲された駒が盤上から消えるので局面のhash keyを更新する
			k -= Zobrist::psq [to][to_pc];
			h += Zobrist::hand[Us][pr   ];
		}

		// fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
		k -= Zobrist::psq[from][moved_pc      ];
		k += Zobrist::psq[to  ][moved_after_pc];
	}

	return k + h;
}

// 指し手で盤面を1手戻す。do_move()の逆変換。
template <Color Us>
void Position::undo_move_impl(Move m)
{
	// Usは1手前の局面での手番(に呼び出し元でしてある)

	auto to = to_sq(m);
	ASSERT_LV2(is_ok(to));

	// --- 移動後の駒

	Piece moved_after_pc = moved_piece_after(m);

#if defined (USE_EVAL_LIST)
	PieceNumber piece_no = piece_no_of(to); // 移動元のpiece_no == いまtoの場所にある駒のpiece_no
	ASSERT_LV3(is_ok(piece_no));
#endif

	// 移動前の駒
	// Piece moved_pc = is_promote(m) ? (moved_after_pc - PIECE_PROMOTE) : moved_after_pc;

	// ↑の処理、mの成りを表現するbitを直接、Pieceの成りを表現するbitに持ってきたほうが速い。
	static_assert((u32)MOVE_PROMOTE / (u32)PIECE_PROMOTE == 4096,"");
	// log(2)4096 == 12
	Piece moved_pc = Piece(moved_after_pc ^ ((m & MOVE_PROMOTE) >> 12));

	if (is_drop(m))
	{
		// --- 駒打ち

		// toの場所にある駒を手駒に戻す
		PieceType pt = raw_type_of(moved_after_pc);

#if defined (USE_EVAL_LIST)
		evalList.put_piece(piece_no, Us, pt, hand_count(hand[Us], pt));
#endif

		add_hand(hand[Us], pt);

		// toの場所から駒を消す
		remove_piece(to);

#if defined(LONG_EFFECT_LIBRARY)
		// 駒打ちのundoによる利きの復元
		LongEffect::rewind_by_dropping_piece<Us>(*this, to, moved_after_pc);
#endif

	} else {

		// --- 通常の指し手

		auto from = from_sq(m);
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
			put_piece(to, to_pc);
			put_piece(from, moved_pc);

#if defined (USE_EVAL_LIST)
			PieceNumber piece_no2 = piece_no_of(Us, raw_type_of(to_pc)); // 捕っていた駒(手駒にある)のpiece_no
			ASSERT_LV3(is_ok(piece_no2));

			evalList.put_piece(piece_no2, to, to_pc);

			// 手駒から減らす
			sub_hand(hand[Us], raw_type_of(to_pc));

			// 成りの指し手だったなら非成りの駒がfromの場所に戻る。さもなくばそのまま戻る。
			// moved_pcが玉であることはあるが、いまkingSquareを更新してしまうと
			// rewind_by_capturing_piece()でその位置を用いているのでまずい。(かも)
			evalList.put_piece(piece_no, from , moved_pc);
#else
			// 手駒から減らす
			sub_hand(hand[Us], raw_type_of(to_pc));
#endif

#if defined(LONG_EFFECT_LIBRARY)
			// 移動先で駒を捕獲するときの利きの更新
			LongEffect::rewind_by_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc, to_pc);
#endif

		}
		else {

			put_piece(from, moved_pc);

#if defined (USE_EVAL_LIST)
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
	sideToMove = Us; // Usは先後入れ替えて呼び出されているはず。

	// --- StateInfoを巻き戻す
	st = st->previous;

	--gamePly;

	// ASSERT_LV5(evalList.is_valid(*this));
	//evalList.is_valid(*this);
}

// do_move()を先後分けたdo_move_impl<>()を呼び出す。
void Position::do_move(Move m, StateInfo& newSt, bool givesCheck)
{
	if (sideToMove == BLACK)
		do_move_impl<BLACK>(m, newSt, givesCheck);
	else
		do_move_impl<WHITE>(m, newSt, givesCheck);
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
void Position::do_null_move(StateInfo& newSt) {

	ASSERT_LV3(!checkers());
	ASSERT_LV3(&newSt != st);

	// この場合、StateInfo自体は丸ごとコピーしておかないといけない。(他の初期化をしないので)
	// よく考えると、StateInfo、新しく作る必要もないのだが…。まあ、CheckInfoがあるので仕方ないか…。
	std::memcpy(&newSt, st, sizeof(StateInfo));

	// TODO : NNUEの場合、accumulatorのコピー不要なのでは…？
	//std::memcpy(&newSt, st, offsetof(StateInfo, accumulator));

	newSt.previous = st;
	st = &newSt;

#if defined(EVAL_NNUE)
	// NNUEの場合、KPPT型と違って、手番が違う場合、計算なしに済ますわけにはいかない。
	st->accumulator.computed_score = false;
#endif

	st->board_key_ ^= Zobrist::side;

	// このタイミングでアドレスが確定するのでprefetchしたほうが良い。(かも)
	// →　将棋では評価関数の計算時のメモリ帯域がボトルネックになって、ここでprefetchしても
	// 　prefetchのスケジューラーが処理しきれない可能性が…。
	// CPUによっては有効なので一応やっておく。

	const HASH_KEY key = st->hash_key();
	prefetch(TT.first_entry(key));

	// これは、さっきアクセスしたところのはずなので意味がない。
	//  Eval::prefetch_evalhash(key);

#if defined(EVAL_NNUE) && defined(USE_EVAL_HASH)
	// NNUEのEvalHashの場合、手番が違うと異なるentry(のはず)
	Eval::prefetch_evalhash(key);
#endif

	//++st->rule50;

	st->pliesFromNull = 0;

	sideToMove = ~sideToMove;

	set_check_info<true>(st);

	//st->repetition = 0;

	//assert(pos_is_ok());

}

void Position::undo_null_move()
{
	ASSERT_LV3(!checkers());

	st = st->previous;
	sideToMove = ~sideToMove;
}


#if defined (USE_SEE)

namespace {

	using namespace Eval;
	using namespace Effect8;

	// min_attacker()はsee_ge()で使われるヘルパー関数であり、(目的升toに利く)
	// 手番側の最も価値の低い攻撃駒の場所を特定し、その見つけた駒をビットボードから取り除き
	// その背後にあった遠方駒をスキャンする。(あればstmAttackersに追加する)

	// またこの関数はmin_attacker<PAWN>()として最初呼び出され、PAWNの攻撃駒がなければ次に
	// KNIGHTの..というように徐々に攻撃駒をアップグレードしていく。

	// occupied = 駒のある場所のbitboard。今回発見された駒は取り除かれる。
	// stmAttackers = 手番側の攻撃駒
	// attackers = toに利く駒(先後両方)。min_attacker(toに利く最小の攻撃駒)を見つけたら、その駒を除去して
	//  その影にいたtoに利く攻撃駒をattackersに追加する。
	// uncapValue = 最後にこの駒が取れなかったときにこの駒が「成り」の指し手だった場合、その価値分の損失が
	// 出るのでそれが返る。

	// 返し値は今回発見されたtoに利く最小の攻撃駒。これがtoの地点において成れるなら成ったあとの駒を返すべき。

	PieceType min_attacker(const Position& pos, const Square& to
		, const Bitboard& stmAttackers, Bitboard& occupied, Bitboard& attackers
	) {

		// 駒種ごとのbitboardのうち、攻撃駒の候補を調べる
	//:      Bitboard b = stmAttackers & bb[Pt];

		// 歩、香、桂、銀、金(金相当の駒)、角、飛、馬、龍…の順で取るのに使う駒を調べる。
		// 金相当の駒については、細かくしたほうが良いかどうかは微妙。

		Bitboard b;
		b = stmAttackers & pos.pieces(PAWN);   if (b) goto found;
		b = stmAttackers & pos.pieces(LANCE);  if (b) goto found;
		b = stmAttackers & pos.pieces(KNIGHT); if (b) goto found;
		b = stmAttackers & pos.pieces(SILVER); if (b) goto found;
		b = stmAttackers & pos.pieces(GOLDS);  if (b) goto found;
		b = stmAttackers & pos.pieces(BISHOP); if (b) goto found;
		b = stmAttackers & pos.pieces(ROOK);   if (b) goto found;
		b = stmAttackers & pos.pieces(HORSE);  if (b) goto found;
		b = stmAttackers & pos.pieces(DRAGON); if (b) goto found;

		// 攻撃駒があるというのが前提条件だから、以上の駒で取れなければ、最後は玉でtoの升に移動出来て駒を取れるはず。
		// 玉を移動させた結果、影になっていた遠方駒によってこの王が取られることはないから、
		// sqに利く遠方駒が追加されることはなく、このままreturnすれば良い。

		return KING;

	found:;

		// bにあった駒を取り除く

		Square sq = b.pop();
		occupied ^= sq;

		// このときpinされているかの判定を入れられるなら入れたほうが良いのだが…。
		// この攻撃駒の種類によって場合分け

		// sqにあった駒が消えるので、toから見てsqの延長線上にある駒を追加する。

		auto dirs = directions_of(to, sq);
		if (dirs) switch(pop_directions(dirs))
		{
		// 斜め方向なら斜め方向の升をスキャンしてその上にある角・馬を足す
		case DIRECT_RU: attackers |= rayEffect<DIRECT_RU>(to, occupied) & pos.pieces<BISHOP_HORSE>(); break;
		case DIRECT_LD: attackers |= rayEffect<DIRECT_LD>(to, occupied) & pos.pieces<BISHOP_HORSE>(); break;
		case DIRECT_RD: attackers |= rayEffect<DIRECT_RD>(to, occupied) & pos.pieces<BISHOP_HORSE>(); break;
		case DIRECT_LU: attackers |= rayEffect<DIRECT_LU>(to, occupied) & pos.pieces<BISHOP_HORSE>(); break;

		// 上方向に移動した時の背後の駒によってtoの地点に利くのは、後手の香 + 先後の飛車
		case DIRECT_U : attackers |= rayEffect<DIRECT_U >(to, occupied) & (pos.pieces<ROOK_DRAGON>() | pos.pieces<WHITE, LANCE>()); break;

		// 下方向に移動した時の背後の駒によってtoの地点に利くのは、先手の香 + 先後の飛車
		case DIRECT_D : attackers |= rayEffect<DIRECT_D >(to, occupied) & (pos.pieces<ROOK_DRAGON>() | pos.pieces<BLACK, LANCE>()); break;

		// 左右方向に移動した時の背後の駒によってtoの地点に利くのは、飛車・龍。
		case DIRECT_L : attackers |= rayEffect<DIRECT_L> (to, occupied) & pos.pieces<ROOK_DRAGON>(); break;
		case DIRECT_R : attackers |= rayEffect<DIRECT_R> (to, occupied) & pos.pieces<ROOK_DRAGON>(); break;

		default: UNREACHABLE; break;
		}
		else {
			// DIRECT_MISC
			ASSERT_LV3(!(bishopStepEffect(to) & sq));
			ASSERT_LV3(!((rookStepEffect(to) & sq)));
		}

		// toに利く攻撃駒は、occupiedのその升が1になっている駒に限定する。
		// 処理した駒はoccupiedのその升が0になるので自動的に除外される。
		attackers &= occupied;

		// この駒が成れるなら、成りの値を返すほうが良いかも。
		// ※　最後にこの地点に残る駒を返すべきなのか。相手が取る/取らないを選択するので。
		return type_of(pos.piece_on(sq));
	}

} // namespace


/// Position::see() is a static exchange evaluator: It tries to estimate the
/// material gain or loss resulting from a move.

// Position::see()は静的交換評価器(SEE)である。これは、指し手による駒による得失の結果
// を見積ろうと試みる。

// 最初に動かす駒側の手番から見た値が返る。

// ※　SEEの解説についてはググれ。
//
// ある升での駒の取り合いの結果、どれくらい駒得/駒損するかを評価する。
// 最初に引数として、指し手mが与えられる。この指し手に対して、同金のように取り返され、さらに同歩成のように
// 取り返していき、最終的な結果(評価値のうちの駒割りの部分の増減)を返すのが本来のSEE。

// ただし、途中の手順では、同金とした場合と同金としない場合とで、(そのプレイヤーは自分が)得なほうを選択できるものとする。

// ※　KINGを敵の利きに移動させる手は非合法手なので、ここで与えられる指し手にはそのような指し手は含まないものとする。
// また、SEEの地点(to)の駒をKINGで取る手は含まれるが、そのKINGを取られることは考慮しなければならない。
// 最後になった駒による成りの上昇値は考えない。

// このseeの最終的な値が、vを以上になるかどうかを判定する。
// こういう設計にすることで早期にvを超えないことが確定した時点でreturn出来る。

bool Position::see_ge(Move m, Value threshold) const
{
	// null windowのときのαβ探索に似たアルゴリズムを用いる。

	// 少し無駄ではあるが、Stockfishの挙動をなるべく忠実に再現する。

	bool drop = is_drop(m);
	// 駒の移動元(駒打ちの場合は)、移動先
	// dropのときにはSQ_NBにしておくことで、pieces() ^ fromを無効化するhack
	Square from = drop ? SQ_NB : from_sq(m);
	Square to = to_sq(m);

	// 次にtoの升で捕獲される駒
	// 成りなら成りを評価したほうが良い可能性があるが、このあとの取り合いで指し手の成りを評価していないので…。
	PieceType nextVictim = drop ? move_dropped_piece(m) : type_of(piece_on(from));

	// 以下のwhileで想定している手番。
	// 移動させる駒側の手番から始まるものとする。
	// 次に列挙すべきは、この駒を取れる敵の駒なので、相手番に。
	// ※「stm」とは"side to move"(手番側)を意味する用語。
	Color us = color_of(moved_piece_after(m));
	Color stm = ~us;

	// 取り合いにおける収支。取った駒の価値と取られた駒の価値の合計。
	// いまthresholdを超えるかどうかが問題なので、この分だけbiasを加えておく。
	Value balance = (Value)Eval::CapturePieceValue[piece_on(to)] - threshold;

	// この時点でマイナスになっているので早期にリターン。
	if (balance < VALUE_ZERO)
		return false;

	// nextVictim == Kingの場合もある。玉が取られる指し手は考えなくて良いので
	// この場合プラス収支と考えてよく、CapturePieceValue[KING] == 0が格納されているので
	// 以下の式によりtrueが返る。

	balance -= (Value)Eval::CapturePieceValue[nextVictim];

	if (balance >= VALUE_ZERO)
		return true;

	// 相手側の手番ならtrue、自分側の手番であるならfalse
	bool relativeStm = true;

	// いま、以下のwhileのなかで想定している手番側の、sqの地点に利く駒
	Bitboard stmAttackers;

	// 盤上の駒(取り合いしていくうちにここから駒が無くなっていく)
	// すでにfromとtoの駒は取られたはずなので消しておく。
	Bitboard occupied = pieces() ^ from ^ to;

	// すべてのattackerを列挙する。
	Bitboard attackers = attackers_to(to, occupied) & occupied;

	while (true)
	{
		stmAttackers = attackers & pieces(stm);

		// pinnersが元の升にいる限りにおいては、pinされた駒から王以外への移動は許さない。

		if (!(st->pinners[~stm] & occupied))
			stmAttackers &= ~st->blockersForKing[stm];

		// 手番側のtoに利いている駒がもうないなら、手番側の負けである。
		if (!stmAttackers)
			break;

		// 次に価値の低い攻撃駒を調べて取り除く。

		nextVictim = min_attacker(*this, to, stmAttackers, occupied, attackers);

		stm = ~stm; // 相手番に

		// Negamax the balance with alpha = balance, beta = balance+1 and
		// add nextVictim's value.
		//
		//      (balance, balance+1) -> (-balance-1, -balance)
		//
		ASSERT_LV3(balance < VALUE_ZERO);

		balance = -balance - 1 - CapturePieceValue[nextVictim];

		// もしbalanceがnextVictimを取り去っても依然として非負(0か正)であるなら、これをもって勝利である。
		// ただし最後に玉が残って、相手側がまだattackerを持っているときはstmを反転しないといけないので注意。
		if (balance >= VALUE_ZERO)
		{
			if (nextVictim == KING && (attackers & pieces(stm)))
				stm = ~stm;
			break;
		}
		ASSERT_LV3(nextVictim != KING);
	}
	return us != stm; // 上のループは、手番側のtoへの利きがある駒が尽きたときに抜ける
}

#endif // defined (USE_SEE)

// ----------------------------------
//      千日手判定
// ----------------------------------

// 連続王手の千日手等で引き分けかどうかを返す
RepetitionState Position::is_repetition(int repPly /* = 16 */) const
{
	// repPlyまで遡る
	// 現在の局面と同じhash keyを持つ局面があれば、それは千日手局面であると判定する。

	// 　rootより遡るなら、2度出現する(3度目の同一局面である)必要がある。
	//   rootより遡らないなら、1度目(2度目の同一局面である)で千日手と判定する。
	// cf.
	//   Don't score as an immediate draw 2-fold repetitions of the root position
	//   https://github.com/official-stockfish/Stockfish/commit/6d89d0b64a99003576d3e0ed616b43333c9eca01

	// チェスだと千日手は同一局面3回(将棋だと4回)である。
	// root以降で同一局面が2度出現した場合は、それを千日手として扱うのは妥当である。
	// root以前の局面と現在の局面が一致している場合は、即座に千日手成立として扱うのは無理があるという判断のもと、
	// 千日手確定のときのみ千日手とする処理がStockfishにはある。
	// しかし、将棋では千日手成立には同一局面が4回出現する必要があるので、この場合、root以前に3回同じ局面が出現して
	// いるかチェックする必要があるが、そこまでする必要があるとは思えない。ゆえに、このチェックを省略する。

	// 【計測資料 35.】is_repetition() 同一局面をrootより遡って見つけたときに即座に千日手として扱うか。
	
	// pliesFromNullが未初期化になっていないかのチェックのためのassert
	ASSERT_LV3(st->pliesFromNull >= 0);

	// 遡り可能な手数。
	// 最大でもrepPly手までしか遡らないことにする。
	int end = std::min(repPly, st->pliesFromNull);

	// 少なくとも4手かけないと千日手にはならないから、4手前から調べていく。
	if (end < 4)
		return REPETITION_NONE;

	StateInfo* stp = st->previous->previous;

	for (int i = 4; i <= end ; i += 2)
	{
		stp = stp->previous->previous;

		// board_key : 盤上の駒のみのhash(手駒を除く)
		// 盤上の駒が同じ状態であるかを判定する。
		if (stp->board_key() == st->board_key())
		{
			// 手駒が一致するなら同一局面である。(2手ずつ遡っているので手番は同じである)
			if (stp->hand == st->hand)
			{
				// 自分が王手をしている連続王手の千日手なのか？
				if (i <= st->continuousCheck[sideToMove])
					return REPETITION_LOSE;

				// 相手が王手をしている連続王手の千日手なのか？
				if (i <= st->continuousCheck[~sideToMove])
					return REPETITION_WIN;

				return REPETITION_DRAW;
			}
			else {
				// 優等局面か劣等局面であるか。(手番が相手番になっている場合はいま考えない)
				if (hand_is_equal_or_superior(st->hand, stp->hand))
					return REPETITION_SUPERIOR;
				if (hand_is_equal_or_superior(stp->hand, st->hand))
					return REPETITION_INFERIOR;
			}
		}
	}

	// 同じhash keyの局面が見つからなかったので…。
	return REPETITION_NONE;
}

// is_repetition()の、千日手が見つかった時に、原局面から何手遡ったかを返すバージョン。
// found_plyにその値が返ってくる。
RepetitionState Position::is_repetition(int repPly, int& found_ply) const
{
	// pliesFromNullが未初期化になっていないかのチェックのためのassert
	ASSERT_LV3(st->pliesFromNull >= 0);

	// 遡り可能な手数。
	// 最大でもrepPly手までしか遡らないことにする。
	int end = std::min(repPly, st->pliesFromNull);

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
		if (stp->board_key() == st->board_key())
		{
			// 手駒が一致するなら同一局面である。(2手ずつ遡っているので手番は同じである)
			if (stp->hand == st->hand)
			{
				// 自分が王手をしている連続王手の千日手なのか？
				if (found_ply <= st->continuousCheck[sideToMove])
					return REPETITION_LOSE;

				// 相手が王手をしている連続王手の千日手なのか？
				if (found_ply <= st->continuousCheck[~sideToMove])
					return REPETITION_WIN;

				return REPETITION_DRAW;
			}
			else {
				// 優等局面か劣等局面であるか。(手番が相手番になっている場合はいま考えない)
				if (hand_is_equal_or_superior(st->hand, stp->hand))
					return REPETITION_SUPERIOR;
				if (hand_is_equal_or_superior(stp->hand, st->hand))
					return REPETITION_INFERIOR;
			}
		}
	}

	// 同じhash keyの局面が見つからなかったので…。
	return REPETITION_NONE;
}


// ----------------------------------
//      入玉判定
// ----------------------------------

// 現在の盤面から、入玉に必要な駒点を計算し、Search::Limits::enteringKingPointに設定する。
void Position::update_entering_point()
{
	auto& limits = Search::Limits;
	auto rule = limits.enteringKingRule;
	int points[COLOR_NB];

	switch (rule)
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
	if (p != 56 && (rule == EKR_24_POINT_H || rule == EKR_27_POINT_H))
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

	limits.enteringKingPoint[BLACK] = points[BLACK];
	limits.enteringKingPoint[WHITE] = points[WHITE];
}

Move Position::DeclarationWin() const
{
	auto rule = Search::Limits.enteringKingRule;

	switch (rule)
	{
		// 入玉ルールなし
	case EKR_NONE: return MOVE_NONE;

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
		if (!(ef & king_square(us)))
			return MOVE_NONE;

		// (e)宣言側の玉に王手がかかっていない。
		if (checkers())
			return MOVE_NONE;


		// (d)宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する。
		int p1 = (pieces(us) & ef).pop_count();
		// p1には玉も含まれているから11枚以上ないといけない
		if (p1 < 11)
			return MOVE_NONE;

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
			//return MOVE_NONE;

		// ↓ 駒落ち対応などを考慮して、enteringKingPoint[]を参照することにした。

		if (score < Search::Limits.enteringKingPoint[us])
			return MOVE_NONE;

		// 評価関数でそのまま使いたいので駒点を返しておくのもアリか…。
		return MOVE_WIN;
	}

	// トライルールの条件を満たしているか。
	case EKR_TRY_RULE:
	{
		Color us = sideToMove;
		Square king_try_sq = (us == BLACK ? SQ_51 : SQ_59);
		Square king_sq = king_square(us);

		// 1) 初期陣形で敵玉がいた場所に自玉が移動できるか。
		if (!(kingEffect(king_sq) & king_try_sq))
			return MOVE_NONE;

		// 2) トライする升に自駒がないか。
		if (pieces(us) & king_try_sq)
			return MOVE_NONE;

		// 3) トライする升に移動させたときに相手に取られないか。
		if (effected_to(~us, king_try_sq, king_sq))
			return MOVE_NONE;

		// 王の移動の指し手により勝ちが確定する
		return make_move(king_sq, king_try_sq, us,KING);
	}

	default:
		UNREACHABLE;
		return MOVE_NONE;
	}
}

// ----------------------------------
//      内部情報の正当性のテスト
// ----------------------------------

bool Position::pos_is_ok() const
{
	// Bitboardの完全なテストには時間がかかるので、あまりややこしいテストは現実的ではない。

#if 0
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
		for (Piece pr = PIECE_HAND_ZERO; pr < PIECE_HAND_NB; ++pr)
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

	// 3) 王手している駒
	if (st->checkersBB != attackers_to(~sideToMove, king_square(sideToMove)))
		return false;

	// 4) 相手玉が取れるということはないか
	if (effected_to(sideToMove, king_square(~sideToMove)))
		return false;

	// 5) occupied bitboardは合っているか
	if ((pieces() != (pieces(BLACK) | pieces(WHITE))) || (pieces(BLACK) & pieces(WHITE)))
		return false;

	// 6) 王手している駒は敵駒か
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

void Position::UnitTest(Test::UnitTester& tester)
{
	auto section1 = tester.section("Position");

	// Search::Limitsのalias
	auto& limits = Search::Limits;

	Position pos;
	StateInfo si;

	// 任意局面での初期化。
	auto pos_init = [&](const std::string& sfen_) { pos.set(sfen_, &si, Threads.main()); };

	// 平手初期化
	auto hirate_init  = [&] { pos.set_hirate(&si, Threads.main()); };
	// 2枚落ち初期化
	auto handi2_sfen = "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
	auto handi2_init = [&] { pos.set(handi2_sfen , &si, Threads.main()); };

	// 4枚落ち初期化
	auto handi4_sfen = "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
	auto handi4_init = [&] { pos.set(handi4_sfen, &si, Threads.main()); };

	// 指し手生成祭りの局面
	auto matsuri_sfen = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1";
	auto matsuri_init = [&] { pos.set(matsuri_sfen, &si, Threads.main()); };

	Move16 m16;
	Move m;

	// 歩の不成の指し手を生成しない状態でテストする。
	limits.generate_all_legal_moves = false;

	// to_move() のテスト
	{
		auto section2 = tester.section("to_move()");

		// 平手初期化
		hirate_init();

		// is_ok(m) == falseな指し手に対して、to_move()がその指し手をそのまま返すことを保証する。
		tester.test("MOVE_NONE", pos.to_move(MOVE_NONE) == MOVE_NONE);
		tester.test("MOVE_WIN" , pos.to_move(MOVE_WIN) == MOVE_WIN);
		tester.test("MOVE_NULL", pos.to_move(MOVE_NULL) == MOVE_NULL);

		// 88の角を22に不成で移動。(非合法手) 移動後の駒は先手の角。
		m16 = make_move16(SQ_88, SQ_22);
		tester.test("make_move16(SQ_88, SQ_22)", pos.to_move(m16) == (Move)((u32)m16.to_u16() + (u32)(B_BISHOP << 16)));

		// 88の角を22に成る移動。(非合法手) 移動後の駒は先手の馬。
		m16 = make_move_promote16(SQ_88, SQ_22);
		tester.test("make_move_promote16(SQ_88, SQ_22)", pos.to_move(m16) == (Move)((u32)m16.to_u16() + (u32)(B_HORSE << 16)));

		// 22の角を88に不成で移動。(非合法手) 移動後の駒は後手の角。
		m16 = make_move16(SQ_22, SQ_88);
		tester.test("make_move16(SQ_22, SQ_88)", pos.to_move(m16) == MOVE_NONE);

		// 22の角を88に成る移動。(非合法手) 移動後の駒は後手の馬。
		m16 = make_move_promote16(SQ_22, SQ_88);
		tester.test("make_move_promote16(SQ_22, SQ_88)", pos.to_move(m16) == MOVE_NONE);

		matsuri_init();
		m16 = make_move_drop16(GOLD,SQ_55);
		tester.test("make_move_drop(SQ_55,GOLD)", pos.to_move(m16) == (Move)((u32)m16.to_u16() + (u32)(W_GOLD << 16)));
	}

	// pseudo_legal() , legal() のテスト
	{
		auto section2 = tester.section("legality");

		// 平手初期化
		hirate_init();

		// 77の歩を76に移動。(合法手)
		// これはpseudo_legalではある。
		m16 = make_move16(SQ_77, SQ_76);
		m = pos.to_move(m16);
		tester.test("make_move(SQ_77, SQ_76) is pseudo_legal == true", pos.pseudo_legal(m) == true);

		// 後手の駒の場合、現在の手番の駒ではないので、pseudo_legalではない。(pseudo_legalは手番側の駒であることを保証する)
		m16 = make_move16(SQ_83, SQ_84);
		m = pos.to_move(m16);
		tester.test("make_move(SQ_83, SQ_84) is pseudo_legal == false", pos.pseudo_legal(m) == false);

		// 88の先手の角を22に移動。これは途中に駒があって移動できないのでpseudo_legalではない。
		// (pseudo_legalは、その駒が移動できる(移動先の升にその駒の利きがある)ことを保証する)
		m16 = make_move16(SQ_88, SQ_22);
		m = pos.to_move(m16);
		tester.test("make_move(SQ_88, SQ_22) is pseudo_legal == false", pos.pseudo_legal(m) == false);
	}

	// attacks_bb() のテスト
	{
		auto section2 = tester.section("attacks_bb");
		hirate_init();
		tester.test("attacks_by<BLACK,PAWN>"  , pos.attacks_by<BLACK,PAWN>() == BB_Table::RANK6_BB);
		tester.test("attacks_by<WHITE,PAWN>"  , pos.attacks_by<WHITE,PAWN>() == BB_Table::RANK4_BB);
		tester.test("attacks_by<BLACK,KNIGHT>", pos.attacks_by<BLACK,KNIGHT>() ==
			(Bitboard(SQ_97) | Bitboard(SQ_77) | Bitboard(SQ_37) | Bitboard(SQ_17))
		);
		tester.test("attacks_by<BLACK,GOLDS>", pos.attacks_by<BLACK,GOLDS>() ==
			(goldEffect<BLACK>(SQ_69) | goldEffect<BLACK>(SQ_49))
		);
		tester.test("attacks_by<WHITE,GOLDS>", pos.attacks_by<WHITE,GOLDS>() ==
			(goldEffect<WHITE>(SQ_61) | goldEffect<WHITE>(SQ_41))
		);

	}

	// 千日手検出のテスト
	{
		auto section2 = tester.section("is_repetition");

		std::deque<StateInfo> sis;

		// 4手前の局面に戻っているパターン
		BookTools::feed_position_string(pos, "startpos moves 5i5h 5a5b 5h5i 5b5a", sis);

		int found_ply;
		auto rep = pos.is_repetition(16, found_ply);

		tester.test("REPETITION_DRAW", rep == REPETITION_DRAW && found_ply == 4);
	}

	// 入玉のテスト
	{
		auto section2 = tester.section("EnteringKing");
		
		{
			// 27点法の入玉可能点数 平手 : 先手=28,後手=27
			auto section3 = tester.section("EKR_27_POINT");

			limits.enteringKingRule = EKR_27_POINT;
			hirate_init();
			
			tester.test("hirate", limits.enteringKingPoint[BLACK] == 28 && limits.enteringKingPoint[WHITE] == 27);

			// 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=28,後手=27
			handi2_init();
			tester.test("handi2", limits.enteringKingPoint[BLACK] == 28 && limits.enteringKingPoint[WHITE] == 27);
		}

		{
			// 24点法の入玉可能点数 平手 : 先手=31,後手=31
			auto section3 = tester.section("EKR_24_POINT");

			limits.enteringKingRule = EKR_24_POINT;
			hirate_init();

			tester.test("hirate", limits.enteringKingPoint[BLACK] == 31 && limits.enteringKingPoint[WHITE] == 31);

			// 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=31,後手=31
			handi2_init();
			tester.test("handi2", limits.enteringKingPoint[BLACK] == 31 && limits.enteringKingPoint[WHITE] == 31);
		}

		{
			// 27点法の入玉可能点数 平手 : 先手=28,後手=27
			auto section3 = tester.section("EKR_27_POINT_H");

			limits.enteringKingRule = EKR_27_POINT_H;
			hirate_init();

			tester.test("hirate", limits.enteringKingPoint[BLACK] == 28 && limits.enteringKingPoint[WHITE] == 27);

			// 2枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=17,下手(BLACK)=28
			handi2_init();
			tester.test("handi2", limits.enteringKingPoint[BLACK] == 28 && limits.enteringKingPoint[WHITE] == 17);

			// 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=15,下手(BLACK)=28
			handi4_init();
			tester.test("handi4", limits.enteringKingPoint[BLACK] == 28 && limits.enteringKingPoint[WHITE] == 15);
		}

		{
			// 24点法の入玉可能点数 平手 : 先手=31,後手=31
			auto section3 = tester.section("EKR_24_POINT_H");

			limits.enteringKingRule = EKR_24_POINT_H;
			hirate_init();

			tester.test("hirate", limits.enteringKingPoint[BLACK] == 31 && limits.enteringKingPoint[WHITE] == 31);

			// 2枚落ち初期化 , 駒落ち対応なのでこの時 上手(WHITE)=21,下手(BLACK)=31
			handi2_init();
			tester.test("handi2", limits.enteringKingPoint[BLACK] == 31 && limits.enteringKingPoint[WHITE] == 21);

			// 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=19,下手(BLACK)=31
			handi4_init();
			tester.test("handi4", limits.enteringKingPoint[BLACK] == 31 && limits.enteringKingPoint[WHITE] == 19);
		}
	}

	{
		// 深いdepthのperftのテストが通っていれば、利きの計算、指し手生成はおおよそ間違っていないと言える。

		auto section2 = tester.section("Perft");

		{
			auto section3 = tester.section("hirate");
			hirate_init();
			const s64 p_nodes[] = { 0 , 30 , 900, 25470, 719731, 19861490, 547581517 };

			for (Depth d = 1; d <= 6; ++d)
			{
				u64 nodes = perft(pos, d);
				u64 pn = p_nodes[d];
				tester.test("depth " + to_string(d) + " = " + to_string(pn), nodes == pn);
			}
		}

		{
			auto section3 = tester.section("matsuri");
			matsuri_init();

			const s64 p_nodes[] = { 0 , 207 , 28684, 4809015, 516925165};

			for (Depth d = 1; d <= 4; ++d)
			{
				u64 nodes = perft(pos, d);
				u64 pn = p_nodes[d];
				tester.test("depth " + to_string(d) + " = " + to_string(nodes), nodes == pn);
			}
		}
	}

	{
		// 指し手生成のテスト
		auto section2 = tester.section("GenMove");

		{
			// 23歩不成ができ、かつ、23歩不成では駒の捕獲にはならない局面。
			pos_init("lnsgk1snl/1r4g2/p1ppppb1p/6pP1/7R1/2P6/P2PPPP1P/1SG6/LN2KGSNL b BP2p 21");
			Move move1 = make_move(SQ_24, SQ_23,B_PAWN);
			Move move2 = make_move_promote(SQ_24, SQ_23,B_PAWN);

			ExtMove move_buf[MAX_MOVES] , *move_last;
			// move_bufからmove_lastのなかにmoveがあるかを探す。あればtrueを返す。
			auto find_move = [&](Move m) {
				for (ExtMove* em = &move_buf[0]; em != move_last; ++em)
					if (em->move == m)
						return true;
				return false;
			};

			bool all = true;

			move_last = generateMoves<NON_CAPTURES>(pos, move_buf);
			all &= !find_move(move1);
			all &=  find_move(move2);

			move_last = generateMoves<CAPTURES>(pos, move_buf);
			all &= !find_move(move1);
			all &= !find_move(move2);

			move_last = generateMoves<NON_EVASIONS>(pos, move_buf);
			all &= !find_move(move1);
			all &=  find_move(move2);

			move_last = generateMoves<NON_EVASIONS_ALL>(pos, move_buf);
			all &=  find_move(move1);
			all &=  find_move(move2);

			move_last = generateMoves<CAPTURES>(pos, move_buf);
			all &= !find_move(move1);
			all &= !find_move(move2);

			move_last = generateMoves<CAPTURES_PRO_PLUS>(pos, move_buf);
			all &= !find_move(move1);
			all &=  find_move(move2);

			move_last = generateMoves<NON_CAPTURES_PRO_MINUS>(pos, move_buf);
			all &= !find_move(move1);
			all &= !find_move(move2);

			tester.test("pawn's unpromoted move", all);
		}
	}


#if 0
	// ランダムプレイヤーでの対局
	{
		auto section2 = tester.section("GamesOfRandomPlayer");

		// 対局回数
		s64 random_player_loop = tester.options["random_player_loop"];

		// seed固定乱数(再現性ある乱数)
		PRNG my_rand;
		StateInfo si[512];

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

				Move m = ml.at(size_t(my_rand.rand(ml.size()))).move;

				pos.do_move(m,si[ply]);

				if (!pos.pos_is_ok())
					fail = true;
			}

			// 今回のゲームのなかでおかしいものがなかったか
			tester.test(string("game ")+to_string(i+1),!fail);
		}
	}
#endif

}


// ----------------------------------
//         明示的な実体化
// ----------------------------------
template bool Position::pseudo_legal_s<false>(const Move m) const;
template bool Position::pseudo_legal_s< true>(const Move m) const;
