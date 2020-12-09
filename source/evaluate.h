#ifndef _EVALUATE_H_
#define _EVALUATE_H_

#include "config.h"
#include "types.h"

// -------------------------------------
//   評価関数に対応するheaderの読み込み
// -------------------------------------

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
#include "eval/evalsum.h"
#endif

// -------------------------------------
//             評価関数
// -------------------------------------

struct StateInfo;

namespace Eval {

	// init()は評価関数の初期化。
	// これは起動直後に1度だけ呼び出される。
	// ただし、(探索部に対して) TUNING_SEARCH_PARAMETERS が defineされている時は、
	// "isready"タイミングで毎回(探索部から)呼び出される。
	void init();

	// 駒割り以外の全計算して、その合計を返す。Position::set()で一度だけ呼び出される。
	// あるいは差分計算が不可能なときに呼び出される。
	Value compute_eval(const Position& pos);

	// 評価関数本体
	// このあとのdo_move()のあとのevaluate()で差分計算ができるように、
	// 現在の前局面から差分計算ができるときだけ計算しておく。
	// 評価値自体は返さない。
	void evaluate_with_no_return(const Position& pos);

	// 評価値の内訳表示(デバッグ用)
	void print_eval_stat(Position& pos);

	// 評価関数ファイルを読み込む。
	// 時間のかかる評価関数の初期化処理はここに書くこと。
	// これは、"is_ready"コマンドの応答時に1度だけ呼び出される。2度呼び出すことは想定していない。
	// (ただし、EvalDir(評価関数フォルダ)が変更になったあと、isreadyが再度送られてきたら読みなおす。)
	void load_eval();

	// 評価関数本体
	Value evaluate(const Position& pos);

	// 駒割りを計算する。Position::set()から呼び出されて、以降do_move()では差分計算されるのでこの関数は呼び出されない。
	Value material(const Position& pos);


#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
	// 評価関数パラメーターのチェックサムを返す。
	u64 calc_check_sum();

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum);
#else
	static u64 calc_check_sum() { return 0; }
	static void print_softname(u64 check_sum) {}
#endif

#if defined (USE_PIECE_VALUE)

	// Apery(WCSC26)の駒割り
	enum {
		PawnValue = 90,
		LanceValue = 315,
		KnightValue = 405,
		SilverValue = 495,
		GoldValue = 540,
		BishopValue = 855,
		RookValue = 990,
		ProPawnValue = 540,
		ProLanceValue = 540,
		ProKnightValue = 540,
		ProSilverValue = 540,
		HorseValue = 945,
		DragonValue = 1395,
		KingValue = 15000,
	};

	// 駒の価値のテーブル(後手の駒は負の値)
	extern int PieceValue[PIECE_NB];

	// 駒の交換値(＝捕獲したときの価値の上昇値)
	// 例)「と」を取ったとき、評価値の変動量は手駒歩+盤面の「と」。
	// MovePickerとSEEの計算で用いる。
	extern int CapturePieceValue[PIECE_NB];

	// 駒を成ったときの成る前との価値の差。SEEで用いる。
	// 駒の成ったものと成っていないものとの価値の差
	// ※　PAWNでもPRO_PAWNでも　と金 - 歩 の価値が返る。
	extern int ProDiffPieceValue[PIECE_NB];
#endif


#if defined(USE_EVAL_LIST)

	// --- 評価関数で使う定数 KPP(玉と任意2駒)のPに相当するenum

	// BonanzaでKKP/KPPと言うときのP(Piece)を表現する型。
	// Σ KPPを求めるときに、39の地点の歩のように、升×駒種に対して一意な番号が必要となる。
	enum BonaPiece : int32_t
	{
		// f = friend(≒先手)の意味。e = enemy(≒後手)の意味

		// 未初期化の時の値
		BONA_PIECE_NOT_INIT = -1,

		// 無効な駒。駒落ちのときなどは、不要な駒をここに移動させる。
		BONA_PIECE_ZERO = 0,

		// --- 手駒

#if defined (EVAL_MATERIAL) || defined (EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_NNUE)
		// Apery(WCSC26)方式。0枚目の駒があるので少し隙間がある。
		// 定数自体は1枚目の駒のindexなので、EVAL_KPPの時と同様の処理で問題ない。
		// 例)
		//  f_hand_pawn  = 先手の1枚目の手駒歩
		//  e_hand_lance = 後手の1枚目の手駒の香
		// Aperyとは手駒に関してはこの部分の定数の意味が1だけ異なるので注意。

		f_hand_pawn = BONA_PIECE_ZERO + 1,//0//0+1
		e_hand_pawn = 20,//f_hand_pawn + 19,//19+1
		f_hand_lance = 39,//e_hand_pawn + 19,//38+1
		e_hand_lance = 44,//f_hand_lance + 5,//43+1
		f_hand_knight = 49,//e_hand_lance + 5,//48+1
		e_hand_knight = 54,//f_hand_knight + 5,//53+1
		f_hand_silver = 59,//e_hand_knight + 5,//58+1
		e_hand_silver = 64,//f_hand_silver + 5,//63+1
		f_hand_gold = 69,//e_hand_silver + 5,//68+1
		e_hand_gold = 74,//f_hand_gold + 5,//73+1
		f_hand_bishop = 79,//e_hand_gold + 5,//78+1
		e_hand_bishop = 82,//f_hand_bishop + 3,//81+1
		f_hand_rook = 85,//e_hand_bishop + 3,//84+1
		e_hand_rook = 88,//f_hand_rook + 3,//87+1
		fe_hand_end = 90,//e_hand_rook + 3,//90

#else 
		fe_hand_end = 0,
#endif                     

		// Bonanzaのように盤上のありえない升の歩や香の番号を詰めない。
		// 理由1) 学習のときに相対PPで1段目に香がいるときがあって、それを逆変換において正しく表示するのが難しい。
		// 理由2) 縦型BitboardだとSquareからの変換に困る。

		// --- 盤上の駒
		f_pawn = fe_hand_end,
		e_pawn = f_pawn + 81,
		f_lance = e_pawn + 81,
		e_lance = f_lance + 81,
		f_knight = e_lance + 81,
		e_knight = f_knight + 81,
		f_silver = e_knight + 81,
		e_silver = f_silver + 81,
		f_gold = e_silver + 81,
		e_gold = f_gold + 81,
		f_bishop = e_gold + 81,
		e_bishop = f_bishop + 81,
		f_horse = e_bishop + 81,
		e_horse = f_horse + 81,
		f_rook = e_horse + 81,
		e_rook = f_rook + 81,
		f_dragon = e_rook + 81,
		e_dragon = f_dragon + 81,
		fe_old_end = e_dragon + 81,

		// === 以下、拡張領域 ===

		// 金と小駒の成り駒を区別する
#if defined(DISTINGUISH_GOLDS)
		f_pro_pawn = fe_old_end,
		e_pro_pawn = f_pro_pawn + 81,
		f_pro_lance = e_pro_pawn + 81,
		e_pro_lance = f_pro_lance + 81,
		f_pro_knight = e_pro_lance + 81,
		e_pro_knight = f_pro_knight + 81,
		f_pro_silver = e_pro_knight + 81,
		e_pro_silver = f_pro_silver + 81,
		fe_new_end = e_pro_silver + 81,
#else
		fe_new_end = fe_old_end,
#endif

		fe_end = fe_new_end,

		// fe_end がKPP配列などのPの値の終端と考えられる。
		// 例) kpp[SQ_NB][fe_end][fe_end];

		// 王も一意な駒番号を付与。これは2駒関係をするときに王に一意な番号が必要なための拡張
		f_king = fe_end,
		e_king = f_king + SQ_NB,
		fe_end2 = e_king + SQ_NB, // 玉も含めた末尾の番号。

		// 末尾は評価関数の性質によって異なるので、BONA_PIECE_NBを定義するわけにはいかない。
	};

	// BonaPieceの内容を表示する。手駒ならH,盤上の駒なら升目。例) HP3 (3枚目の手駒の歩)
	std::ostream& operator<<(std::ostream& os, BonaPiece bp);

	// BonaPieceを後手から見たとき(先手の39の歩を後手から見ると後手の71の歩)の番号とを
	// ペアにしたものをExtBonaPiece型と呼ぶことにする。
	union ExtBonaPiece
	{
		struct {
			BonaPiece fb; // from black
			BonaPiece fw; // from white
		};
		BonaPiece from[2];

		ExtBonaPiece() {}
		ExtBonaPiece(BonaPiece fb_, BonaPiece fw_) : fb(fb_) , fw(fw_){}
	};

	// BonaPiece、f側だけを表示する。
	inline std::ostream& operator<<(std::ostream& os, ExtBonaPiece bp) { os << bp.fb; return os; }

	// 駒が今回の指し手によってどこからどこに移動したのかの情報。
	// 駒はExtBonaPiece表現であるとする。
	struct ChangedBonaPiece
	{
		ExtBonaPiece old_piece;
		ExtBonaPiece new_piece;
	};

	// KPPテーブルの盤上の駒pcに対応するBonaPieceを求めるための配列。
	// 例)
	// BonaPiece fb = kpp_board_index[pc].fb + sq; // 先手から見たsqにあるpcに対応するBonaPiece
	// BonaPiece fw = kpp_board_index[pc].fw + sq; // 後手から見たsqにあるpcに対応するBonaPiece
	extern ExtBonaPiece kpp_board_index[PIECE_NB];

	// KPPの手駒テーブル
	extern ExtBonaPiece kpp_hand_index[COLOR_NB][KING];

	// 評価関数で用いる駒リスト。どの駒(PieceNumber)がどこにあるのか(BonaPiece)を保持している構造体
	struct EvalList
	{
		// 評価関数(FV38型)で用いる駒番号のリスト
		BonaPiece* piece_list_fb() const { return const_cast<BonaPiece*>(pieceListFb); }
		BonaPiece* piece_list_fw() const { return const_cast<BonaPiece*>(pieceListFw); }

		// 指定されたpiece_noの駒をExtBonaPiece型に変換して返す。
		ExtBonaPiece bona_piece(PieceNumber piece_no) const
		{
			ExtBonaPiece bp;
			bp.fb = pieceListFb[piece_no];
			bp.fw = pieceListFw[piece_no];
			return bp;
		}

		// 盤上のsqの升にpiece_noのpcの駒を配置する
		void put_piece(PieceNumber piece_no, Square sq, Piece pc) {
			set_piece_on_board(piece_no, BonaPiece(kpp_board_index[pc].fb + sq), BonaPiece(kpp_board_index[pc].fw + Inv(sq)),sq);
		}

		// c側の手駒ptのi+1枚目の駒のPieceNumberを設定する。(1枚目の駒のPieceNumberを設定したいならi==0にして呼び出すの意味)
		void put_piece(PieceNumber piece_no, Color c, PieceType pt, int i) {
			set_piece_on_hand(piece_no, BonaPiece(kpp_hand_index[c][pt].fb + i), BonaPiece(kpp_hand_index[c][pt].fw + i));
		}

		// あるBonaPieceに対応するPieceNumberを返す。
		PieceNumber piece_no_of_hand(BonaPiece bp) const { return piece_no_list_hand[bp]; }
		// 盤上のある升sqに対応するPieceNumberを返す。
		PieceNumber piece_no_of_board(Square sq) const { return piece_no_list_board[sq]; }

		// pieceListを初期化する。
		// 駒落ちに対応させる時のために、未使用の駒の値はBONA_PIECE_ZEROにしておく。
		// 通常の評価関数を駒落ちの評価関数として流用できる。
		// piece_no_listのほうはデバッグが捗るようにPIECE_NUMBER_NBで初期化。
		void clear()
		{

			for (auto& p : pieceListFb)
				p = BONA_PIECE_ZERO;

			for (auto& p : pieceListFw)
				p = BONA_PIECE_ZERO;

			for (auto& v : piece_no_list_hand)
				v = PIECE_NUMBER_NB;

			for (auto& v : piece_no_list_board)
				v = PIECE_NUMBER_NB;
		}

		// 内部で保持しているpieceListFb[]が正しいBonaPieceであるかを検査する。
		// 注 : デバッグ用。遅い。
		bool is_valid(const Position& pos);


	protected:

		// 盤上sqにあるpiece_noの駒のBonaPieceがfb,fwであることを設定する。
		inline void set_piece_on_board(PieceNumber piece_no, BonaPiece fb , BonaPiece fw, Square sq)
		{
			ASSERT_LV3(is_ok(piece_no));
			pieceListFb[piece_no] = fb;
			pieceListFw[piece_no] = fw;
			piece_no_list_board[sq] = piece_no;
		}

		// 手駒であるpiece_noの駒のBonaPieceがfb,fwであることを設定する。
		inline void set_piece_on_hand(PieceNumber piece_no, BonaPiece fb, BonaPiece fw)
		{
			ASSERT_LV3(is_ok(piece_no));
			pieceListFb[piece_no] = fb;
			pieceListFw[piece_no] = fw;
			piece_no_list_hand[fb] = piece_no;
		}

		// 駒リスト。駒番号(PieceNumber)いくつの駒がどこにあるのか(BonaPiece)を示す。FV38などで用いる。

		// 駒リストの長さ
		// 38固定
	public:
		int length() const { return PIECE_NUMBER_KING; }

		// VPGATHERDDを使う都合、4の倍数でなければならない。
		// また、KPPT型評価関数などは、39,40番目の要素がゼロであることを前提とした
		// アクセスをしている箇所があるので注意すること。
		static const int MAX_LENGTH = 40;

	private:

	#if defined(USE_AVX2)
		// AVX2を用いたKPPT評価関数は高速化できるので特別扱い。
		// Skylake以降でないとほぼ効果がないが…。

		// AVX2の命令でアクセスするのでalignas(32)が必要。
		alignas(32) BonaPiece pieceListFb[MAX_LENGTH];
		alignas(32) BonaPiece pieceListFw[MAX_LENGTH];

	#else

		BonaPiece pieceListFb[MAX_LENGTH];
		BonaPiece pieceListFw[MAX_LENGTH];

	#endif

		// 手駒である、任意のBonaPieceに対して、その駒番号(PieceNumber)を保持している配列
		PieceNumber piece_no_list_hand[fe_hand_end];

		// 盤上の駒に対して、その駒番号(PieceNumber)を保持している配列
		// 玉がSQ_NBに移動しているとき用に+1まで保持しておくが、
		// SQ_NBの玉を移動させないので、この値を使うことはないはず。
		PieceNumber piece_no_list_board[SQ_NB_PLUS1];

	};

	// --- 局面の評価値の差分更新用
	// 局面の評価値を差分更新するために、移動した駒を管理する必要がある。
	// この移動した駒のことをDirtyPieceと呼ぶ。
	// 1) FV38方式だと、DirtyPieceはたかだか2個。
	// 2) FV_VAR方式だと、DirtyPieceは可変。
	//       →　こっちは廃止した。

	// 評価値の差分計算の管理用
	// 前の局面から移動した駒番号を管理するための構造体
	// 動く駒は、最大で2個。
	struct DirtyPiece
	{
		// その駒番号の駒が何から何に変わったのか
		Eval::ChangedBonaPiece changed_piece[2];

		// dirtyになった駒番号
		PieceNumber pieceNo[2];

		// dirtyになった個数。
		// null moveだと0ということもありうる。
		// 動く駒と取られる駒とで最大で2つ。
		int dirty_num;
	};
#endif // defined(USE_EVAL_LIST)

#if defined(USE_EVAL_HASH)
	// EvalHashのリサイズ
	extern void EvalHash_Resize(size_t mbSize);

	// EvalHashのクリア
	extern void EvalHash_Clear();
#endif

}

#endif // #ifndef _EVALUATE_H_
