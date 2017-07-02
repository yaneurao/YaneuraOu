#ifndef EVALUATE_H
#define EVALUATE_H

#include "shogi.h"

// 手番込みの評価関数であれば手番を込みで値を計算するhelper classを使う。
#if defined(EVAL_KKPT) || defined(EVAL_KPPT)
#include "eval/kppt_evalsum.h"
#endif

// 実験中の評価関数。現状、非公開。
#if defined(EVAL_EXPERIMENTAL)
#define EVAL_EXPERIMENTAL_HEADER
#include "eval/experimental/evaluate_experimental.h"
#undef EVAL_EXPERIMENTAL_HEADER
#else
#define BonaPieceExpansion 0
#endif


// --------------------
//    評価関数
// --------------------

namespace Eval {

	// evaluateの起動時に行なう軽量な初期化はここで行なう。
	extern void init();

	// 評価関数ファイルを読み込む。
	void load_eval();

	// 駒割りを計算する。Position::set()から呼び出されて、以降do_move()では差分計算されるのでこの関数は呼び出されない。
	Value material(const Position& pos);

	// 評価関数本体
	Value evaluate(const Position& pos);

	// 評価関数本体
	// このあとのdo_move()のあとのevaluate()で差分計算ができるように、
	// 現在の前局面から差分計算ができるときだけ計算しておく。
	// 評価値自体は返さない。
	void evaluate_with_no_return(const Position& pos);

	// 駒割り以外の全計算して、その合計を返す。Position::set()で一度だけ呼び出される。
	// あるいは差分計算が不可能なときに呼び出される。
	Value compute_eval(const Position& pos);

#if defined(USE_EVAL_HASH) && (defined(EVAL_KKPT) || defined(EVAL_KPPT) )
	// prefetchする関数
	void prefetch_evalhash(const Key key);
#endif

#if defined(EVAL_KKPT) || defined(EVAL_KPPT)
	// 評価関数パラメーターのチェックサムを返す。
	u64 calc_check_sum();

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum);
#else
	static u64 calc_check_sum() { return 0; }
	static void print_softname(u64 check_sum) {}
#endif

	// 評価値の内訳表示(デバッグ用)
	void print_eval_stat(Position& pos);

#if defined(EVAL_LEARN) && (defined(EVAL_KKPT) || defined(EVAL_KPPT))
	// 学習のときの勾配配列の初期化
	// 学習率を引数に渡しておく。0.0なら、defaultの値を採用する。
	void init_grad(double eta);

	// 現在の局面で出現している特徴すべてに対して、勾配の差分値を勾配配列に加算する。
	void add_grad(Position& pos, Color rootColor, double delt_grad);

	// 現在の勾配をもとにSGDかAdaGradか何かする。
	void update_weights(/*u64 epoch*/);

	// 評価関数パラメーターをファイルに保存する。
	// ファイルの末尾につける拡張子を指定できる。
	void save_eval(std::string suffix);

#endif

#ifdef EVAL_NO_USE

	// 評価関数を用いないときもValueを正規化するときに歩の価値は必要。
	enum { PawnValue = 86 };

#else

#if defined (EVAL_MATERIAL) || defined (EVAL_PP) || defined(EVAL_KPP)
	// Bona6の駒割りを初期値に。それぞれの駒の価値。
	enum {
		PawnValue = 86,
		LanceValue = 227,
		KnightValue = 256,
		SilverValue = 365,
		GoldValue = 439,
		BishopValue = 563,
		RookValue = 629,
		ProPawnValue = 540,
		ProLanceValue = 508,
		ProKnightValue = 517,
		ProSilverValue = 502,
		HorseValue = 826,
		DragonValue = 942,
		KingValue = 15000,
	};
#elif defined(EVAL_KKPT) || defined (EVAL_KPPT)

	// Aperyの駒割り
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
#endif

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

	// --- 評価関数で使う定数 KPP(玉と任意2駒)のPに相当するenum

	// (評価関数の実験のときには、BonaPieceは自由に定義したいのでここでは定義しない。)


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

#if defined (EVAL_MATERIAL) || defined (EVAL_PP) || defined(EVAL_KPP)

		f_hand_pawn = BONA_PIECE_ZERO + 1,
		e_hand_pawn = f_hand_pawn + 18,
		f_hand_lance = e_hand_pawn + 18,
		e_hand_lance = f_hand_lance + 4,
		f_hand_knight = e_hand_lance + 4,
		e_hand_knight = f_hand_knight + 4,
		f_hand_silver = e_hand_knight + 4,
		e_hand_silver = f_hand_silver + 4,
		f_hand_gold = e_hand_silver + 4,
		e_hand_gold = f_hand_gold + 4,
		f_hand_bishop = e_hand_gold + 4,
		e_hand_bishop = f_hand_bishop + 2,
		f_hand_rook = e_hand_bishop + 2,
		e_hand_rook = f_hand_rook + 2,
		fe_hand_end = e_hand_rook + 2,

#elif defined(EVAL_KKPT) || defined (EVAL_KPPT)
		// Apery(WCSC26)方式。0枚目の駒があるので少し隙間がある。
		// 定数自体は1枚目の駒のindexなので、KPPの時と同様の処理で問題ない。

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
		// 理由1) 学習のときに相対PPで1段目に香がいるときがあって、それが逆変換において正しく表示するのが難しい。
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

		// fe_endの値をBonaPieceExpansionを定義することで変更できる。
		// このときfe_old_end～fe_endの間の番号をBonaPiece拡張領域として自由に用いることが出来る。
		fe_end = fe_old_end + BonaPieceExpansion,

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
	struct ExtBonaPiece
	{
		BonaPiece fb; // from black
		BonaPiece fw; // from white
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

	// 評価関数で用いる駒リスト。どの駒(PieceNo)がどこにあるのか(BonaPiece)を保持している構造体
	struct EvalList
	{
		// 評価関数(FV38型)で用いる駒番号のリスト
		BonaPiece* piece_list_fb() const { return const_cast<BonaPiece*>(pieceListFb); }
		BonaPiece* piece_list_fw() const { return const_cast<BonaPiece*>(pieceListFw); }

		// 指定されたpiece_noの駒をExtBonaPiece型に変換して返す。
		ExtBonaPiece bona_piece(PieceNo piece_no) const
		{
			ExtBonaPiece bp;
			bp.fb = pieceListFb[piece_no];
			bp.fw = pieceListFw[piece_no];
			return bp;
		}

		// 盤上のsqの升にpiece_noのpcの駒を配置する
		void put_piece(PieceNo piece_no, Square sq, Piece pc) {
			set_piece_on_board(piece_no, BonaPiece(kpp_board_index[pc].fb + sq), BonaPiece(kpp_board_index[pc].fw + Inv(sq)),sq);
		}

		// c側の手駒ptのi+1枚目の駒のPieceNoを設定する。(1枚目の駒のPieceNoを設定したいならi==0にして呼び出すの意味)
		void put_piece(PieceNo piece_no, Color c, Piece pt, int i) {
			set_piece_on_hand(piece_no, BonaPiece(kpp_hand_index[c][pt].fb + i), BonaPiece(kpp_hand_index[c][pt].fw + i));
		}

		// あるBonaPieceに対応するPieceNoを返す。
		PieceNo piece_no_of_hand(BonaPiece bp) const { return piece_no_list_hand[bp]; }
		// 盤上のある升sqに対応するPieceNoを返す。
		PieceNo piece_no_of_board(Square sq) const { return piece_no_list_board[sq]; }

		// pieceListを初期化する。
		// 駒落ちに対応させる時のために、未使用の駒の値はBONA_PIECE_ZEROにしておく。
		// 通常の評価関数を駒落ちの評価関数として流用できる。
		// piece_no_listのほうはデバッグが捗るようにPIECE_NO_NBで初期化。
		void clear()
		{
			for (auto& p : pieceListFb)
				p = BONA_PIECE_ZERO;
			for (auto& p : pieceListFw)
				p = BONA_PIECE_ZERO;

			for (auto& v : piece_no_list_hand)
				v = PIECE_NO_NB;
			for (auto& v : piece_no_list_board)
				v = PIECE_NO_NB;
		}

	protected:

		// 盤上sqにあるpiece_noの駒のBonaPieceがfb,fwであることを設定する。
		inline void set_piece_on_board(PieceNo piece_no, BonaPiece fb, BonaPiece fw , Square sq)
		{
			ASSERT_LV3(is_ok(piece_no));
			pieceListFb[piece_no] = fb;
			pieceListFw[piece_no] = fw;
			piece_no_list_board[sq] = piece_no;
		}

		// 手駒であるpiece_noの駒のBonaPieceがfb,fwであることを設定する。
		inline void set_piece_on_hand(PieceNo piece_no, BonaPiece fb, BonaPiece fw)
		{
			ASSERT_LV3(is_ok(piece_no));
			pieceListFb[piece_no] = fb;
			pieceListFw[piece_no] = fw;
			piece_no_list_hand[fb] = piece_no;
		}

		// 駒リスト。駒番号(PieceNo)いくつの駒がどこにあるのか(BonaPiece)を示す。FV38などで用いる。
#if defined(EVAL_KPPT) && defined(USE_AVX2)
		// AVX2を用いたKPPT評価関数は高速化できるので特別扱い。
		// Skylake以降でないとほぼ効果がないが…。

		// AVX2の命令でアクセスするのでalignas(32)が必要。
		alignas(32) BonaPiece pieceListFb[PIECE_NO_NB];
		alignas(32) BonaPiece pieceListFw[PIECE_NO_NB];
#else
		BonaPiece pieceListFb[PIECE_NO_NB];
		BonaPiece pieceListFw[PIECE_NO_NB];
#endif

		// 手駒である、任意のBonaPieceに対して、その駒番号(PieceNo)を保持している配列
		PieceNo piece_no_list_hand[fe_hand_end];

		// 盤上の駒に対して、その駒番号(PieceNo)を保持している配列
		// 玉がSQ_NBに移動しているとき用に+1まで保持しておくが、
		// SQ_NBの玉を移動させないので、この値を使うことはないはず。
		PieceNo piece_no_list_board[SQ_NB_PLUS1];
	};
#endif

#if defined(EVAL_KPPT)
	// 評価関数のそれぞれのパラメーターに対して関数fを適用してくれるoperator。
	// パラメーターの分析などに用いる。
	void foreach_eval_param(std::function<void(s32,s32)>f);
#endif

#if defined (USE_EVAL_MAKE_LIST_FUNCTION)

	// 評価関数の実験のためには、EvalListの組み換えが必要になるのでその機能を提供する。
	// Eval::compute_eval()とLearner::add_grad()に作用する。またこのとき差分計算は無効化される。
	// 現状、KPPT型評価関数に対してしか提供していない。

	// compute_eval()やLearner::add_grad()からBonaPiece番号の組み換えのために呼び出される関数
	// make_listと言う名前は、Bonanza6のソースコードに由来する。
	extern std::function<void(const Position&, BonaPiece[40], BonaPiece[40])> make_list_function;

	// 旧評価関数から新評価関数に変換するときにKPPのP(BonaPiece)がどう写像されるのかを定義したmapper。
	// EvalIO::eval_convert()の引数として渡される。
	extern std::vector<u16 /*BonaPiece*/> eval_mapper;
#endif
}

#endif // #ifndef EVALUATE_H
