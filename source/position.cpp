﻿#include "position.h"
#include "misc.h"
#include "tt.h"
#include "thread.h"

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

// Bitboardクラスにはalignasが指定されているが、StateListPtrは、このStateInfoクラスを内部的にnewするときに
// alignasを無視するのでcustom allocatorを定義しておいてやる。
void* StateInfo::operator new(std::size_t s) {	return aligned_malloc(s, alignof(StateInfo)); }
void StateInfo::operator delete(void*p) noexcept { aligned_free(p); }

// 王手情報の初期化
template <bool doNullMove>
void Position::set_check_info(StateInfo* si) const {

	//: si->blockersForKing[WHITE] = slider_blockers(pieces(BLACK), square<KING>(WHITE),si->pinners[WHITE]);
	//: si->blockersForKing[BLACK] = slider_blockers(pieces(WHITE), square<KING>(BLACK),si->pinners[BLACK]);

	// ↓Stockfishのこの部分の実装、将棋においては良くないので、以下のように変える。

	if (!doNullMove)
	{
		// null moveのときは前の局面でこの情報は設定されているので更新する必要がない。
		si->blockersForKing[WHITE] = slider_blockers(BLACK, square<KING>(WHITE), si->pinners[WHITE]);
		si->blockersForKing[BLACK] = slider_blockers(WHITE, square<KING>(BLACK), si->pinners[BLACK]);
	}

	Square ksq = square<KING>(~sideToMove);

	// 駒種Xによって敵玉に王手となる升のbitboard

	// 歩であれば、自玉に敵の歩を置いたときの利きにある場所に自分の歩があればそれは敵玉に対して王手になるので、
	// そういう意味で(ksq,them)となっている。

	Bitboard occ = pieces();
	Color them = ~sideToMove;

	// この指し手が二歩でないかは、この時点でテストしない。指し手生成で除外する。なるべくこの手のチェックは遅延させる。
	si->checkSquares[PAWN]   = pawnEffect(them, ksq);
	si->checkSquares[KNIGHT] = knightEffect(them, ksq);
	si->checkSquares[SILVER] = silverEffect(them, ksq);
	si->checkSquares[BISHOP] = bishopEffect(ksq, occ);
	si->checkSquares[ROOK]   = rookEffect(ksq, occ);
	si->checkSquares[GOLD]   = goldEffect(them, ksq);

	// 香で王手になる升は利きを求め直さずに飛車で王手になる升を香のstep effectでマスクしたものを使う。
	si->checkSquares[LANCE]  = si->checkSquares[ROOK] & lanceStepEffect(them,ksq);

	// 王を移動させて直接王手になることはない。それは自殺手である。
	si->checkSquares[KING]   = ZERO_BB;

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

#if defined(CUCKOO)
// Marcel van Kervinck's cuckoo algorithm for fast detection of "upcoming repetition"
// situations. Description of the algorithm in the following paper:
// https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// Stockfishの2倍の配列を確保

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return h & 0x3fff; }
inline int H2(Key h) { return (h >> 16) & 0x3fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
Key cuckoo[8192*2];
Move cuckooMove[8192*2];

//  →　cuckooアルゴリズムとやらで、千日手局面に到達する指し手の検出が高速化できるらしい。
// (数手前の局面と現在の局面の差が、ある駒の移動(+捕獲)だけであることが高速に判定できれば、
// 　早期枝刈りとしてdraw_valueを返すことができる。)
#endif

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
		for (Piece pr = PIECE_ZERO; pr < PIECE_HAND_NB; ++pr)
			if (pr)
				SET_HASH(Zobrist::hand[c][pr], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

	for (int i = 0; i < MAX_PLY; ++i)
		SET_HASH(Zobrist::depth[i], rng.rand<Key>() & ~1ULL, rng.rand<Key>(), rng.rand<Key>(), rng.rand<Key>());

#if defined(CUCKOO)
	// Prepare the cuckoo tables
	std::memset(cuckoo, 0, sizeof(cuckoo));
	std::memset(cuckooMove, 0, sizeof(cuckooMove));
	int count = 0;
	// 重複カウント用
	int count2 = 0;
	for (auto pc : Piece())
	{
		auto pt = type_of(pc);
		if (!(pt == PAWN || pt == LANCE || pt == KNIGHT || pt == SILVER || pt == GOLD || pt == BISHOP || pt == ROOK || pt == KING
			|| pt == PRO_PAWN || pt == PRO_LANCE || pt == PRO_KNIGHT || pt == PRO_SILVER || pt == HORSE || pt == DRAGON))
			continue;

		// 将棋だとチェスと異なり、from →　toに動かせるからと言ってto→fromに動かせるとは限らないので
		// ここのコード、ずいぶん違ってくる。
		for (auto s1 : SQ)
			for (Square s2 : SQ)
				if (effects_from(pc, s1, ZERO_BB) & s2)
				{
					Move move = (Move)(make_move(s1, s2) + (pc << 16));
					// 手番のところ使わない。無視するために潰す。
					Key key = (Zobrist::psq[s2][pc] - Zobrist::psq[s1][pc]) >> 1/* Zobrist::side*/;
					int i = H1(key);
					while (true)
					{
						std::swap(cuckoo[i], key);
						std::swap(cuckooMove[i], move);
						if (move == MOVE_NONE) // Arrived at empty slot?
							break;

						//i = (i == H1(key)) ? H2(key) : H1(key); // Push victim to alternative slot
						// →　これ、テーブル小さいので衝突しつづける…(´ω｀)　H1になかったらH2でええで..

						i = H2(key);
						std::swap(cuckoo[i], key);
						std::swap(cuckooMove[i], move);

						if (move != MOVE_NONE)
							count2++;

						break;
					}
					count++;
				}
	}
	//assert(count == 3668); // chessの場合

	// cout << "count = " << count << " , count2 = " << count2 << endl;
	// chessの2倍の配列時 : count = 16456 , count2 = 4499 
	// chessの4倍の配列時 : count = 16456 , count2 = 1623
	ASSERT_LV3(count == 16456);
#endif
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
		  "^歩","^香","^桂","^銀","^角","^飛","^金","^玉","^と","^杏","^圭","^全","^馬","^龍","^菌","^王" };
std::string pretty(Piece pc) { return USI_PIECE_KANJI[pc]; }
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

	// evalListのclear。上でmemsetでゼロクリアしたときにクリアされているが…。
	evalList.clear();

#if defined (USE_FV38)
	// PieceListを更新する上で、どの駒がどこにあるかを設定しなければならないが、
	// それぞれの駒をどこまで使ったかのカウンター
	PieceNumber piece_no_count[KING] = { PIECE_NUMBER_ZERO,PIECE_NUMBER_PAWN,PIECE_NUMBER_LANCE,PIECE_NUMBER_KNIGHT,
	  PIECE_NUMBER_SILVER, PIECE_NUMBER_BISHOP, PIECE_NUMBER_ROOK,PIECE_NUMBER_GOLD };

	// 先手玉のいない詰将棋とか、駒落ちに対応させるために、存在しない駒はすべてBONA_PIECE_ZEROにいることにする。
	// 上のevalList.clear()で、ゼロクリアしているので、それは達成しているはず。
#elif defined(USE_FV_VAR)
	auto& dp = st->dirtyPiece;
	// FV_VARのときは直接evalListに追加せず、DirtyPieceにいったん追加して、
	// そのあと、DirtyPiece::update()でevalListに追加する。このupdate()の時に組み換えなどの操作をしたいため。
	dp.set_state_info(st);
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
			auto pc = Piece(idx + (promote ? u32(PIECE_PROMOTE) : 0));
			put_piece(sq, pc);

#if defined(USE_FV38)
			PieceNumber piece_no =
				(idx == B_KING) ? PIECE_NUMBER_BKING : // 先手玉
				(idx == W_KING) ? PIECE_NUMBER_WKING : // 後手玉
				piece_no_count[raw_type_of(Piece(idx))]++; // それ以外
			evalList.put_piece(piece_no, sq, pc); // sqの升にpcの駒を配置する
#elif defined(USE_FV_VAR)
			if (type_of(pc) != KING)
			{
				dp.add_piece(sq, pc);
				dp.do_update(evalList);
				dp.clear();
				// DirtyPieceのBonaPieceを格納するバッファ、極めて小さいのでevalListに反映させるごとにクリアしておく。

				//Eval::print_eval_list(*this);
			}
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
				Piece rpc = raw_type_of(Piece(idx));
#if defined (USE_FV38)
				PieceNumber piece_no = piece_no_count[rpc]++;
				ASSERT_LV1(is_ok(piece_no));
				evalList.put_piece(piece_no, color_of(Piece(idx)), rpc, i);
#elif defined(USE_FV_VAR)
				dp.add_piece(color_of(Piece(idx)), rpc, i);
				dp.do_update(evalList);
				dp.clear();
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

	st->materialValue = Eval::material(*this);
	Eval::compute_eval(*this);

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
const std::string Position::sfen() const
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

			const Piece USI_Hand[7] = { ROOK,BISHOP,GOLD,SILVER,KNIGHT,LANCE,PAWN };
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
	ss << gamePly;

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
		for (Piece pr = PAWN; pr < PIECE_HAND_NB; ++pr)
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
// 遮っている両方の手番を返す。ただし、２重に遮っている場合はそれらの駒は返さない。
// もし、この関数のこの返す駒を取り除いた場合、升sに対してsliderによって利きがある状態になる。
// 升sにある玉に対してこの関数を呼び出した場合、それはpinされている駒と両王手の候補となる駒である。
// また、升sにある玉は~c側のKINGであるとする。

Bitboard Position::slider_blockers(Color c, Square s , Bitboard& pinners) const {

	Bitboard result = ZERO_BB;

	// pinnersは返し値。
	pinners = ZERO_BB;

	// cが与えられていないと香の利きの方向を確定させることが出来ない。
	// ゆえに将棋では、この関数は手番を引数に取るべき。(チェスとはこの点において異なる。)

	// snipersとは、pinされている駒が取り除かれたときに升sに利きが発生する大駒である。
	Bitboard snipers =
		( (pieces(ROOK_DRAGON)  & rookStepEffect(s))
		| (pieces(BISHOP_HORSE) & bishopStepEffect(s))
		// 香に関しては攻撃駒が先手なら、玉より下側をサーチして、そこにある先手の香を探す。
		| (pieces(LANCE) & lanceStepEffect(~c, s))
		) & pieces(c);

	while (snipers)
	{
		Square sniperSq = snipers.pop();
		Bitboard b = between_bb(s, sniperSq) & pieces();

		// snipperと玉との間にある駒が1個であるなら。
		// (間にある駒が0個の場合、b == ZERO_BBとなり、何も変化しない。)
		if (!more_than_one(b))
		{
			result |= b;
			if (b & pieces(~c))
				// sniperと玉に挟まれた駒が玉と同じ色の駒であるなら、pinnerに追加。
				pinners |= sniperSq;
		}
	}
	return result;
}


// sに利きのあるc側の駒を列挙する。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
Bitboard Position::attackers_to(Color c, Square sq, const Bitboard& occ) const
{
	ASSERT_LV3(is_ok(c) && sq <= SQ_NB);

	Color them = ~c;

	// sの地点に敵駒ptをおいて、その利きに自駒のptがあればsに利いているということだ。
	// 香の利きを求めるコストが惜しいのでrookEffect()を利用する。
	return
		(     (pawnEffect(them, sq)		&  pieces(PAWN)        )
			| (knightEffect(them, sq)	&  pieces(KNIGHT)      )
			| (silverEffect(them, sq)	&  pieces(SILVER_HDK)  )
			| (goldEffect(them, sq)		&  pieces(GOLDS_HDK)   )
			| (bishopEffect(sq, occ)	&  pieces(BISHOP_HORSE))
			| (rookEffect(sq, occ)		& (
					pieces(ROOK_DRAGON)
				|  (lanceStepEffect(them,sq) & pieces(LANCE))
			  ))
		//  | (kingEffect(sq) & pieces(c, HDK));
		// →　HDKは、銀と金のところに含めることによって、参照するテーブルを一個減らして高速化しようというAperyのアイデア。
			) & pieces(c); // 先後混在しているのでc側の駒だけ最後にマスクする。
		;

}

// sに利きのあるc側の駒を列挙する。先後両方。
// (occが指定されていなければ現在の盤面において。occが指定されていればそれをoccupied bitboardとして)
Bitboard Position::attackers_to(Square sq, const Bitboard& occ) const
{
	ASSERT_LV3(sq <= SQ_NB);

	// sqの地点に敵駒ptをおいて、その利きに自駒のptがあればsqに利いているということだ。
	return
		// 先手の歩・桂・銀・金・HDK
		((    (pawnEffect(WHITE, sq)   & pieces(PAWN)        )
			| (knightEffect(WHITE, sq) & pieces(KNIGHT)      )
			| (silverEffect(WHITE, sq) & pieces(SILVER_HDK)  )
			| (goldEffect(WHITE, sq)   & pieces(GOLDS_HDK)   )
			) & pieces(BLACK))
		|

		// 後手の歩・桂・銀・金・HDK
		((    (pawnEffect(BLACK, sq)   & pieces(PAWN)        )
			| (knightEffect(BLACK, sq) & pieces(KNIGHT)      )
			| (silverEffect(BLACK, sq) & pieces(SILVER_HDK)  )
			| (goldEffect(BLACK, sq)   & pieces(GOLDS_HDK)   )
			) & pieces(WHITE))

		// 先後の角・飛・香
		| (bishopEffect(sq, occ) & pieces(BISHOP_HORSE) )
		| (rookEffect(sq, occ) & (
			   pieces(ROOK_DRAGON)
			| (pieces(BLACK , LANCE) & lanceStepEffect(WHITE , sq))
			| (pieces(WHITE , LANCE) & lanceStepEffect(BLACK , sq))
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
	const Square to = move_to(m);

	// 駒打ち・移動する指し手どちらであってもmove_piece_after(m)で移動後の駒が取得できるので
	// 直接王手の処理は共通化できる。
	if (st->checkSquares[type_of(moved_piece_after(m))] & to)
		return true;

	// -- 移動する指し手ならば、これで開き王手になるかどうかの判定が必要。

	// 移動元
	const Square from = move_from(m);

	// 開き王手になる駒の候補があるとして、fromにあるのがその駒で、fromからtoは玉と直線上にないなら
	// 前提条件より、fromにあるのが自駒であることは確定しているので、pieces(sideToMove)は不要。
	return !is_drop(m)
		&& (((blockers_for_king(~sideToMove) /*& pieces(sideToMove)*/) & from)
		&& !aligned(from, to, square<KING>(~sideToMove)));
}

Bitboard Position::pinned_pieces(Color c, Square avoid) const {
	Bitboard b, pinners, result = ZERO_BB;
	Square ksq = king_square(c);

	// avoidを除外して考える。
	Bitboard avoid_bb = ~Bitboard(avoid);

	pinners = (
		  (pieces(ROOK_DRAGON)   & rookStepEffect(ksq))
		| (pieces(BISHOP_HORSE)  & bishopStepEffect(ksq))
		| (pieces(LANCE)         & lanceStepEffect(c, ksq))
		) & avoid_bb & pieces(~c);

	while (pinners)
	{
		b = between_bb(ksq, pinners.pop()) & pieces() & avoid_bb;
		if (!more_than_one(b))
			result |= b & pieces(c);
	}
	return result;
}

Bitboard Position::pinned_pieces(Color c, Square from, Square to) const {
	Bitboard b, pinners, result = ZERO_BB;
	Square ksq = king_square(c);

	// avoidを除外して考える。
	Bitboard avoid_bb = ~Bitboard(from);

	pinners = (
		(pieces(ROOK_DRAGON)    & rookStepEffect(ksq))
		| (pieces(BISHOP_HORSE) & bishopStepEffect(ksq))
		| (pieces(LANCE)        & lanceStepEffect(c, ksq))
		) & avoid_bb & pieces(~c);

	// fromからは消えて、toの地点に駒が現れているものとして
	Bitboard new_pieces = (pieces() & avoid_bb) | to;
	while (pinners)
	{
		b = between_bb(ksq, pinners.pop()) & new_pieces;
		if (!more_than_one(b))
			result |= b & pieces(c);
	}
	return result;
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
  if (b & (~pinned | FILE_BB[file_of(to)]))
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

#ifndef LONG_EFFECT_LIBRARY
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

// ※　mがこの局面においてpseudo_legalかどうかを判定するための関数。
template <bool All>
bool Position::pseudo_legal_s(const Move m) const {

	const Color us = sideToMove;
	const Square to = move_to(m); // 移動先

	if (is_drop(m))
	{
		const Piece pr = move_dropped_piece(m);
		// 置換表から取り出してきている以上、一度は指し手生成ルーチンで生成した指し手のはずであり、
		// KING打ちのような値であることはないものとする。

#if defined(KEEP_PIECE_IN_GENERATE_MOVES)
		// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
		if (moved_piece_after(m) != Piece(pr + ((us == WHITE) ? u32(PIECE_WHITE) : 0) + PIECE_DROP))
			return false;
#endif

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

		const Square from = move_from(m);
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

		Piece pt = type_of(pc);
		if (is_promote(m))
		{
			// --- 成る指し手

			// 成れない駒の成りではないことを確かめないといけない。
			static_assert(GOLD == 7, "GOLD must be 7.");
			if (pt >= GOLD)
				return false;

#if defined(KEEP_PIECE_IN_GENERATE_MOVES)
			// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
			// pcが成っていない駒であることは上で確認してあるので、"+ PIECE_PROMOTE"でも十分。
			if (moved_piece_after(m) != Piece(pc + PIECE_PROMOTE))
				return false;
#endif

#if !defined(KEEP_PIECE_IN_GENERATE_MOVES)
			// Zobrist::side == 1なので先手と後手は常にハッシュ値が異なる。
			// よって先手と後手の手が置換表衝突する事はない。
			// killer move等に関しては32bit形式であればPieceと成と移動元(先)が矛盾する事はない。
			// そのため、32bit形式ではこのチェックは不要。
			if (!(enemy_field(us) & (Bitboard(from) | Bitboard(to))))
				return false;
#endif

		}
		else {

			// --- 成らない指し手

#if defined(KEEP_PIECE_IN_GENERATE_MOVES)
			// 上位32bitに移動後の駒が格納されている。それと一致するかのテスト
			if (moved_piece_after(m) != pc)
				return false;
#endif


			// 駒打ちのところに書いた理由により、不成で進めない升への指し手のチェックも不要。
			// 間違い　→　駒種をmoveに含めていないならこのチェック必要だわ。
			// 52から51銀のような指し手がkillerやcountermoveに登録されていたとして、52に歩があると
			// 51歩不成という指し手を生成してしまう…。
			// あと、歩や大駒が敵陣において成らない指し手も不要なのでは..。

			if (All)
			{
				// 歩と香に関しては1段目への不成は不可。桂は、桂飛びが出来る駒は桂しかないので
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
				if (more_than_one(checkers()))
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
		Square from = move_from(m);

		ASSERT_LV5(color_of(piece_on(from_sq(m))) == us);
		ASSERT_LV5(piece_on(square<KING>(us)) == make_piece(us, KING));

		// もし移動させる駒が玉であるなら、行き先の升に相手側の利きがないかをチェックする。
		if (type_of(piece_on(from)) == KING)
			return !effected_to(~us, move_to(m), from);

		// blockers_for_king()は、pinされている駒(自駒・敵駒)を表現するが、fromにある駒は自駒であることは
		// わかっているのでこれで良い。
		return   !(blockers_for_king(us) & from)
			|| aligned(from, to_sq(m), square<KING>(us));
	}
}


// 置換表から取り出したMoveを32bit化する。
Move Position::move16_to_move(Move m) const
{
	//		ASSERT_LV3(is_ok(m));
	// 置換表から取り出した値なので m==MOVE_NONE(0)である可能性があり、ASSERTは書けない。

	// 上位16bitは0でなければならない
	//      ASSERT_LV3((m >> 16) == 0);

	return Move(u16(m) +
			((is_drop(m) ? (Piece)(make_piece(sideToMove, move_dropped_piece(m)) + PIECE_DROP)
			: is_promote(m) ? (Piece)(piece_on(move_from(m)) | PIECE_PROMOTE) : piece_on(move_from(m))) << 16)
		// "+ PIECE_PROMOTE" だと、玉や成り駒に対して 8足しておかしくなってしまう。(置換表の指し手をpseudo-legalか
		// 確認せずに置換表の値で枝刈りして、そのあとupdate_statsを行う時に配列境界を超えかねない)
		// " | PIECE_PROMOTE"が正しいコード。 WCSC29で平岡さんから教えてもらった。[2019/05/04]
		// また、move_dropped_piece()はおかしい値になっていないことは保証されている(置換表に自分で書き出した値のため)
		// これにより、配列境界の外側に書き出してしまうことはない。

		// m==MOVE_NONE(0)の場合、piece_on(SQ_ZERO)の駒が上位16bitに格納されると少し気持ち悪いが、
		// 移動元と移動先が SQ_11なので実質的に無害。
	);

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

	// 探索ノード数 ≒do_move()の呼び出し回数のインクリメント。
	thisThread->nodes.fetch_add(1, std::memory_order_relaxed);

	//std::cout << *this << m << std::endl;

#if defined(USE_FV_VAR)
	// 前nodeでのdirtyPieceをevalListに反映させていない可能性がある。
	// 毎node、evaluate()を呼び出すならevaluate()側の責任においてそれは行われるのだが、
	// positionコマンドなどで特定局面までevaluate()を呼び出さずにdo_move()することがあるので、
	// ここでそのチェックをしておかなければならない。
	if (!st->dirtyPiece.updated())
		st->dirtyPiece.do_update(evalList);
#endif

	// ----------------------
	//  StateInfoの更新
	// ----------------------

	// hash key

	// 現在の局面のhash keyはこれで、これを更新していき、次の局面のhash keyを求めてStateInfo::key_に格納。
	auto k = st->board_key_ ^ Zobrist::side;
	auto h = st->hand_key_;

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

	// 直前の指し手を保存するならばここで行なう。

#if defined (KEEP_LAST_MOVE)
	st->lastMove = m;
	st->lastMovedPieceType = is_drop(m) ? (Piece)move_from(m) : type_of(piece_on(move_from(m)));
#endif

	// ----------------------
	//    盤面の更新処理
	// ----------------------

	// 移動先の升
	Square to = move_to(m);
	ASSERT_LV2(is_ok(to));

	// 駒割りの差分計算用
	int materialDiff;

	auto& dp = st->dirtyPiece;

#if defined(USE_FV_VAR)
	// add()していくので、length = 0にしないといけない。
	dp.clear();
	dp.set_state_info(st);
#endif

	if (is_drop(m))
	{
		// --- 駒打ち

		// 移動先の升は空のはず
		ASSERT_LV2(piece_on(to) == NO_PIECE);

		Piece pc = moved_piece_after(m);
		Piece pr = raw_type_of(pc);
		ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

		// Zobrist keyの更新
		h -= Zobrist::hand[Us][pr];
		k += Zobrist::psq[to][pc];

		// なるべく早い段階でのTTに対するprefetch
		// 駒打ちのときはこの時点でTT entryのアドレスが確定できる
		const Key key = k + h;
		prefetch(TT.first_entry(key));
#if defined(USE_EVAL_HASH)
		Eval::prefetch_evalhash(key);
#endif

		put_piece(to, pc);

		// 打駒した駒に関するevalListの更新。
#if defined (USE_FV38)
		PieceNumber piece_no = piece_no_of(Us, pr);
		ASSERT_LV3(is_ok(piece_no));

		// KPPの差分計算のために移動した駒をStateInfoに記録しておく。
		dp.dirty_num = 1; // 動いた駒は1個
		dp.pieceNo[0] = piece_no;
		dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no);
		evalList.put_piece(piece_no , to, pc);
		dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no);

		// piece_no_of()のときにこの手駒の枚数を参照するのであとで更新。
		sub_hand(hand[Us], pr);

#elif defined(USE_FV_VAR)

		// 駒打ちなので手駒が減る。この場合は次のhand_countで必要なので先に更新。
		sub_hand(hand[Us], pr);

		// 駒打ちなのでpcが玉である可能性はない。
		dp.remove_piece(Us, pr, hand_count(hand[Us], pr));
		dp.add_piece(to, pc);

		// 玉の移動ではないことを示しておく。
		dp.moved_king = COLOR_NB;
#endif
		
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
			st->checkersBB = ZERO_BB;
			st->continuousCheck[Us] = 0;
		}

		// 駒打ちは捕獲した駒がない。
		st->capturedPiece = NO_PIECE;

		// put_piece()などを用いたのでupdateする
		update_bitboards();

		// 駒打ちなので駒割りの変動なし。
		materialDiff = 0;

#if defined(LONG_EFFECT_LIBRARY)
		// 駒打ちによる利きの更新処理
		LongEffect::update_by_dropping_piece<Us>(*this, to, pc);
#endif

	} else {

		// -- 駒の移動
		Square from = move_from(m);
		ASSERT_LV2(is_ok(from));

		// 移動させる駒
		Piece moved_pc = piece_on(from);
		ASSERT_LV2(moved_pc != NO_PIECE);

		// 移動先に駒の配置
		// もし成る指し手であるなら、成った後の駒を配置する。
		Piece moved_after_pc = moved_piece_after(m);

		materialDiff = is_promote(m) ? Eval::ProDiffPieceValue[moved_pc] : 0;

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

			Piece pr = raw_type_of(to_pc);

			// 捕獲した駒に関するevalListの更新
#if defined (USE_FV38)
			// このPieceNumberの駒が手駒に移動したのでEvalListのほうを更新しておく。
			PieceNumber piece_no = piece_no_of(to);
			ASSERT_LV3(is_ok(piece_no));
			dp.dirty_num = 2; // 動いた駒は2個
			dp.pieceNo[1] = piece_no;
			dp.changed_piece[1].old_piece = evalList.bona_piece(piece_no);
			evalList.put_piece(piece_no, Us, pr, hand_count(hand[Us], pr));
			dp.changed_piece[1].new_piece = evalList.bona_piece(piece_no);
#elif defined(USE_FV_VAR)

			// 捕獲された駒の処理なので、これが玉である可能性はない。
			dp.remove_piece(to,to_pc);
			dp.add_piece(Us, pr, hand_count(hand[Us], pr));

			//std::cout << hand_count(hand[Us], pr) << Us << pr << std::endl;

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

			// 評価関数で使う駒割りの値も更新
			materialDiff += Eval::CapturePieceValue[to_pc];

		} else {
			// 駒を取らない指し手

			st->capturedPiece = NO_PIECE;

#if defined (LONG_EFFECT_LIBRARY)
			// 移動先で駒を捕獲しないときの利きの更新
			LongEffect::update_by_no_capturing_piece<Us>(*this, from, to, moved_pc, moved_after_pc);
#endif
#if defined (USE_FV38)
			dp.dirty_num = 1; // 動いた駒は1個
#endif
		}

#if defined (USE_FV38)
		// 移動元にあった駒のpiece_noを得る
		PieceNumber piece_no2 = piece_no_of(from);
		dp.pieceNo[0] = piece_no2;
		dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no2);
#endif

		// 移動元の升からの駒の除去
		remove_piece(from);
		// 移動先の升に駒を配置
		put_piece(to, moved_after_pc);

#if defined (USE_FV38)
		evalList.put_piece(piece_no2, to, moved_after_pc);
		dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no2);

		// 王を移動させる手であるなら、kingSquareを更新しておく。
		// 王は駒打できないのでdropの指し手に含まれていることはないから
		// dropのときにはkingSquareを更新する必要はない。
		if (type_of(moved_pc) == KING)
			kingSquare[Us] = to;

#elif defined(USE_FV_VAR)
		// 移動させる駒が玉であるときはevalListを更新する必要がない。
		// ただし、玉が移動したことを示す必要はある。
		if (type_of(moved_pc) == KING)
		{
			kingSquare[Us] = to;
			dp.moved_king = Us;
		}
		else
		{
			//dp.remove_piece(from, moved_pc);
			//dp.add_piece(to, moved_after_pc);
			dp.remove_and_add_piece(from, moved_pc, to, moved_after_pc);
			dp.moved_king = COLOR_NB; // 玉の移動ではないことを示しておく。
		}
#endif

		// fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
		k -= Zobrist::psq[from][moved_pc];
		k += Zobrist::psq[to][moved_after_pc];

		// 駒打ちでないときはprefetchはこの時点まで延期される。
		const Key key = k + h;
		prefetch(TT.first_entry(key));
#if defined(USE_EVAL_HASH)
		Eval::prefetch_evalhash(key);
#endif

		// put_piece()などを用いたのでupdateする。
		// ROOK_DRAGONなどをこの直後で用いるのでここより後ろにやるわけにはいかない。
		update_bitboards();

		// 王手している駒のbitboardを更新する。
		if (givesCheck)
		{
			// 高速化のためにごにょごにょ。
#if 1
			const StateInfo* prevSt = st->previous;

			// 1) 直接王手であるかどうかは、移動によって王手になる駒別のBitboardを調べればわかる。
			st->checkersBB = prevSt->checkSquares[type_of(moved_after_pc)] & to;

			// 2) 開き王手になるのか
			const Square ksq = king_square(~Us);
			// pos->discovered_check_candidates()で取得したいが、もうstを更新してしまっているので出来ないので
			// prevSt->blockersForKing[~Us] & pieces(Us)と愚直に書く。
			// また、pieces(Us)のうち今回移動させる駒は、実はすでに移動させてしまっているので、fromと書く。

			if (discovered(from, to, ksq, prevSt->blockersForKing[~Us] & from))
			{
				auto directions = directions_of(from, ksq);
				switch (pop_directions(directions)) {

					// fromと敵玉とは同じ筋にあり、かつfromから駒を移動させて空き王手になる。
					// つまりfromから上下を見ると、敵玉と、自分の開き王手をしている遠方駒(飛車 or 香)があるはずなのでこれを追加する。
					// 敵玉はpieces(Us)なので含まれないはずであり、結果として自分の開き王手している駒だけが足される。

				case DIRECT_U: case DIRECT_D:
					st->checkersBB |= rookFileEffect(from, pieces()) & pieces(Us); break;

					// 横に利く遠方駒は飛車(+龍)しかないので、玉の位置から飛車の利きを求めてその利きのなかにいる飛車を足す。
					// →　飛車の横だけの利きを求める関数を用意したので、それを用いると上と同様の手法で求まる。

				case DIRECT_R: case DIRECT_L:
					st->checkersBB |= rookRankEffect(from, pieces()) & pieces(Us); break;

					// 斜めに利く遠方駒は角(+馬)しかないので、玉の位置から角の利きを求めてその利きのなかにいる角を足す。
					// →　上と同様の方法が使える。以下同様。

				case DIRECT_RU: case DIRECT_LD:
					st->checkersBB |= bishopEffect0(from, pieces()) & pieces(Us); break;

				case DIRECT_RD: case DIRECT_LU:
					st->checkersBB |= bishopEffect1(from, pieces()) & pieces(Us); break;


				default: UNREACHABLE;
				}
			}

			// 差分更新したcheckersBBが正しく更新されているかをテストするためのassert
			ASSERT_LV3(st->checkersBB == attackers_to(Us, king_square(~Us)));
#else
			// 差分更新しないとき用。
			st->checkersBB = attackers_to(Us, king_square(~Us));
#endif
			st->continuousCheck[Us] = prev->continuousCheck[Us] + 2;

		} else {

			st->checkersBB = ZERO_BB;
			st->continuousCheck[Us] = 0;
		}
	}
	// 相手番のほうは関係ないので前ノードの値をそのまま受け継ぐ。
	st->continuousCheck[~Us] = prev->continuousCheck[~Us];

	st->materialValue = (Value)(st->previous->materialValue + (Us == BLACK ? materialDiff : -materialDiff));
	//ASSERT_LV5(st->materialValue == Eval::material(*this));

	// 相手番に変更する。
	sideToMove = ~Us;

	// 更新されたhash keyをStateInfoに書き戻す。
	st->board_key_ = k;
	st->hand_key_ = h;

	st->hand = hand[sideToMove];

	// このタイミングで王手関係の情報を更新しておいてやる。
	set_check_info<false>(st);

	//ASSERT_LV5(evalList.is_valid(*this));

	//state()->dirtyPiece.do_update(evalList);
	//evalList.is_valid(*this);
}

#if defined(USE_KEY_AFTER)

// ある指し手を指した後のhash keyを返す。
Key Position::key_after(Move m) const {

	Color Us = side_to_move();
	auto k = st->board_key_ ^ Zobrist::side;
	auto h = st->hand_key_;

	// 移動先の升
	Square to = move_to(m);
	ASSERT_LV2(is_ok(to));

	if (is_drop(m))
	{
		// --- 駒打ち
		Piece pr = move_dropped_piece(m);
		ASSERT_LV2(PAWN <= pr && pr < PIECE_HAND_NB);

		Piece pc = make_piece(Us, pr);

		// Zobrist keyの更新
		h -= Zobrist::hand[Us][pr];
		k += Zobrist::psq[to][pc];
	}
	else
	{
		// -- 駒の移動
		Square from = move_from(m);
		ASSERT_LV2(is_ok(from));

		// 移動させる駒
		Piece moved_pc = piece_on(from);
		ASSERT_LV2(moved_pc != NO_PIECE);

		// 移動先に駒の配置
		// もし成る指し手であるなら、成った後の駒を配置する。
		Piece moved_after_pc;

		if (is_promote(m))
		{
			moved_after_pc = moved_pc + PIECE_PROMOTE;
		}
		else {
			moved_after_pc = moved_pc;
		}

		// 移動先の升にある駒
		Piece to_pc = piece_on(to);
		if (to_pc != NO_PIECE)
		{
			Piece pr = raw_type_of(to_pc);

			// 捕獲された駒が盤上から消えるので局面のhash keyを更新する
			k -= Zobrist::psq[to][to_pc];
			h += Zobrist::hand[Us][pr];
		}

		// fromにあったmoved_pcがtoにmoved_after_pcとして移動した。
		k -= Zobrist::psq[from][moved_pc];
		k += Zobrist::psq[to][moved_after_pc];
	}

	return k + h;
}
#endif

// 指し手で盤面を1手戻す。do_move()の逆変換。
template <Color Us>
void Position::undo_move_impl(Move m)
{
	// Usは1手前の局面での手番(に呼び出し元でしてある)

	auto to = move_to(m);
	ASSERT_LV2(is_ok(to));

	// --- 移動後の駒

	// 手番が変わるのでKEEP_PIECE_IN_GENERATE_MOVESを定義していないときに
	// moved_piece_after()を呼び出すのは正しく動作しない。
	Piece moved_after_pc =
#if defined(KEEP_PIECE_IN_GENERATE_MOVES)
		moved_piece_after(m);
#else
		piece_on(to);
#endif

#if defined(USE_FV38)
	PieceNumber piece_no = piece_no_of(to); // 移動元のpiece_no == いまtoの場所にある駒のpiece_no
	ASSERT_LV3(is_ok(piece_no));
#elif defined(USE_FV_VAR)
	// do_move()のあとevaluate()を呼び出していないなら、eval_listの更新がなされていないのでundo不要。
	auto&dp = st->dirtyPiece;
	if (dp.updated())
		st->dirtyPiece.undo_update(evalList);
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
		Piece pt = raw_type_of(moved_after_pc);

#if defined(USE_FV38)
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

		auto from = move_from(m);
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

#if defined(USE_FV38)
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

#if defined(USE_FV38)
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

#if defined(USE_FV_VAR)
	// evalListがupdateされずにdo_null_move()を呼び出している可能性がある。do_move()のほうの説明を読むこと。
	if (!st->dirtyPiece.updated())
		st->dirtyPiece.do_update(evalList);
#endif

	// この場合、StateInfo自体は丸ごとコピーしておかないといけない。(他の初期化をしないので)
	// よく考えると、StateInfo、新しく作る必要もないのだが…。まあ、CheckInfoがあるので仕方ないか…。
	std::memcpy(&newSt, st, sizeof(StateInfo));
	newSt.previous = st;
	st = &newSt;

	st->board_key_ ^= Zobrist::side;

	// このタイミングでアドレスが確定するのでprefetchしたほうが良い。(かも)
	// →　将棋では評価関数の計算時のメモリ帯域がボトルネックになって、ここでprefetchしても
	// 　prefetchのスケジューラーが処理しきれない可能性が…。
	// CPUによっては有効なので一応やっておく。

	const Key key = st->key();
	prefetch(TT.first_entry(key));

	// これは、さっきアクセスしたところのはずなので意味がない。
	//  Eval::prefetch_evalhash(key);

#if defined(EVAL_NNUE)
#if defined(USE_EVAL_HASH)
	Eval::prefetch_evalhash(key);
#endif
	st->accumulator.computed_score = false;
#endif

	st->pliesFromNull = 0;

	sideToMove = ~sideToMove;

	set_check_info<true>(st);
}

void Position::undo_null_move()
{
	ASSERT_LV3(!checkers());

	st = st->previous;
	sideToMove = ~sideToMove;
}

// ----------------------------------
//      千日手判定
// ----------------------------------

// 連続王手の千日手等で引き分けかどうかを返す
RepetitionState Position::is_repetition(int ply , int repPly_ /* = 32 */) const
{
	// repPlyまで遡る
	// rootより遡るのであれば2度同一局面が出現する必要があるので16の倍にしておく。
	// この値が増えるの、多少、気分が悪いところではあるが…。
	//
	// これ16から32に変更したことで1%ぐらい速度低下するようだ。
	// 16手目までに1度も同一局面が出現しなければリタイアしたいが、この処理を綺麗に書くのは難しい…。
	const int repPly = repPly_;

	// 現在の局面と同じhash keyを持つ局面があれば、それは千日手局面であると判定する。

	// 　rootより遡るなら、2度出現する(3度目の同一局面である)必要がある。
	//   rootより遡らないなら、1度目(2度目の同一局面である)で千日手と判定する。
	// cf.
	//   Don't score as an immediate draw 2-fold repetitions of the root position
	//   https://github.com/official-stockfish/Stockfish/commit/6d89d0b64a99003576d3e0ed616b43333c9eca01

	// pliesFromNullが未初期化になっていないかのチェックのためのassert
	ASSERT_LV3(st->pliesFromNull >= 0);

	// 遡り可能な手数。
	// 最大でもrepPly手までしか遡らないことにする。
	int end = std::min(repPly, st->pliesFromNull);

	// 少なくとも4手かけないと千日手にはならないから、4手前から調べていく。
	if (end < 4)
		return REPETITION_NONE;

	StateInfo* stp = st->previous->previous;

	// 盤上の駒が同一である局面が出現した回数
	int cnt = 0;

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
				// root(==ply)より遡る(ply <= i)なら2度出現(cnt == 2)する必要がある。
				// rootより遡らない(ply > i)なら1度目の出現(cnt == 1)で千日手と判定する。
				if (++cnt + (ply > i) == 2)
				{
					// 自分が王手をしている連続王手の千日手なのか？
					if (i <= st->continuousCheck[sideToMove])
						return REPETITION_LOSE;

					// 相手が王手をしている連続王手の千日手なのか？
					if (i <= st->continuousCheck[~sideToMove])
						return REPETITION_WIN;

					return REPETITION_DRAW;
				}
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

#if defined(CUCKOO)
// この局面から以前の局面に到達する指し手があるか。
// ply : 遡る手数
bool Position::has_game_cycle(int plies_from_root, int rep_ply /*= 16*/) const
{
	int j;

	int end = std::min(/* st->rule50*/ rep_ply , st->pliesFromNull);

	if (end < 3)
		return false;

	Key originalKey = st->key();
	StateInfo* stp = st->previous;

	for (int i = 3; i <= end; i += 2)
	{
		stp = stp->previous->previous;

		// やねうら王ではZobrist Hashに足し算を使っているので、差を取る必要がある。
		// bit 0はside(手番)なので、ここは削る。
		Key moveKey = (stp->key() - originalKey) >> 1;
		if ((j = H1(moveKey), cuckoo[j] == moveKey)
			|| (j = H2(moveKey), cuckoo[j] == moveKey))
		{
			Move move = cuckooMove[j];
			Square s1 = from_sq(move);
			Square s2 = to_sq(move);
			Piece pc = (Piece)(move >> 16);
			if (piece_on(s1) != pc)
				continue;

			// 間に駒がないのでたぶんいける。開き王手とか知らん。
			// ざっくりした枝刈りにしか使わないのでこのへんの判定甘くても問題ない。
			if (!(between_bb(s1, s2) & pieces()))
			{
				if (plies_from_root > i)
					return true;

				// For repetitions before or at the root, require one more
				// rootまでに指し手が見つかった場合、もう一度同じ局面に遭遇する必要がある。

				// Stockfish10のコード
				//if (stp->repetition)
				//	return true;

				// Stockfish9の時のコード。(StateInfo->repetitionを持っていないのでこんなコードになる)
				StateInfo* next_stp = stp;
				for (int k = i + 2; k <= end; k += 2)
				{
					next_stp = next_stp->previous->previous;
					if (next_stp->key() == stp->key())
						return true;
				}

			}
		}
	}
	return false;
}
#endif

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

// 明示的な実体化
template bool Position::pseudo_legal_s<false>(const Move m) const;
template bool Position::pseudo_legal_s< true>(const Move m) const;
