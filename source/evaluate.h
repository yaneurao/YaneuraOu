#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include "config.h"
#include <string>

#include "types.h"

// -------------------------------------
//             評価関数
// -------------------------------------

namespace YaneuraOu {

class Position;
template<typename T> class LazyNumaReplicatedSystemWide;

namespace Eval {

#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
// Shogi models use fixed filenames instead of Stockfish's hash-based default name.
// Stockfishのハッシュ付きファイル名とは異なり、将棋用評価関数は固定名を使う。
// The embedding macro requires a string literal; share the name with loading and options.
// 埋め込みには文字列リテラルが必要。同じ既定名を外部読み込み・オプション・ヘルプでも使う。
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
#if !defined(EVAL_SFNN)
#error ENABLE_SFNN_16BIT_WEIGHT requires an SFNN architecture
#endif
#define EvalFileDefaultName "nn16.bin"
#else
#define EvalFileDefaultName "nn.bin"
#endif

namespace NNUE {
struct Network;
class AccumulatorCaches;
class AccumulatorStack;
struct Dirties;
}
#endif

#if !defined(EVAL_SFNN) && !defined(EVAL_NNUE) && !defined(EVAL_KPPT) && !defined(EVAL_KPP_KKPT)
    // 差分評価を使わない構成でも、探索から同じ形でpush()/do_move()を呼べるようにする。
    // 駒の差分は保持せず、Positionもこの型に対して差分を書き込まない。
    struct EmptyDirties {};
#endif

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
    namespace Classic { class AccumulatorStack; }
    Value evaluate(const Position& pos, Classic::AccumulatorStack& stack);
    void evaluate_with_no_return(const Position& pos, Classic::AccumulatorStack& stack);
#endif

#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
    using NetworkStorage = LazyNumaReplicatedSystemWide<NNUE::Network>;
#else
    // 実体は探索・Engine側の共通入口nnue/network.hで定義する。
    class NetworkStorage;
#endif

#if defined(USE_CLASSIC_EVAL) || defined(EVAL_SFNN)
	// 評価関数の初期化。これは起動直後に1度だけ呼び出される。
	// 💡 引数で受け取ったoptionsを用いて、
	//     評価関数に必要なエンジンオプションを追加することができる。
#if defined(USE_CLASSIC_EVAL) && !defined(EVAL_NNUE)
	void add_options(OptionsMap& options, ThreadPool& threads);

	// 駒割りを含む評価値を全計算し、手番側から見た値を返す。
	// 探索外の評価や検証用の再計算に用いる。差分計算不能時は内部の全計算処理も使う。
	Value compute_eval(const Position& pos);

	// 後続の差分計算に備え、必要なら現在局面の評価を準備する。評価値自体は返さない。
	// KPPT/KPP_KKPTでは未計算ならevaluate()を呼び、差分計算できなければ全計算する。
	// USE_EVAL_MAKE_LIST_FUNCTIONによる実験用構成では、この準備を省略する。
	// 備考) 差分計算型の評価関数ではないときは、この関数は何もしなくて良い。
	void evaluate_with_no_return(const Position& pos);
#endif

	// 評価値の内訳表示(デバッグ用)
	void print_eval_stat(Position& pos);

	// 評価関数パラメーターを読み込む。
	// 時間のかかる評価関数の初期化処理はここに書くこと。
	// これは、"isready"コマンドの応答時に(毎回)呼び出される。
	// ⚠ EvalDir(評価関数フォルダ)が変更になったあと、isreadyが再度送られてきたら読みなおす処理は自分で書くこと。
#if defined(EVAL_SFNN) || defined(EVAL_NNUE)
    void load_eval(LazyNumaReplicatedSystemWide<NNUE::Network>& network,
                   const OptionsMap& options);
    Value evaluate(const NNUE::Network& network, const Position& pos);
    Value evaluate(const NNUE::Network& network, const Position& pos,
                   NNUE::AccumulatorStack& stack, NNUE::AccumulatorCaches& refreshTable);
#if defined(EVAL_NNUE)
    void evaluate_with_no_return(const NNUE::Network& network, const Position& pos,
                                 NNUE::AccumulatorStack& stack, NNUE::AccumulatorCaches& refreshTable);
#endif
#else
    void load_eval();
    // 評価関数本体
    Value evaluate(const Position& pos);
#endif

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)
	// 評価関数パラメーターのチェックサムを返す。
	u64 calc_check_sum();

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum);
#else
	static u64 calc_check_sum() { return 0; }
	static void print_softname([[maybe_unused]] u64 check_sum) {}
#endif

#endif // defined(USE_CLASSIC_EVAL)

    // 駒割りを計算する。
	// 💡 この関数は、classic evalでなくとも用いることがある。(例えば、dfpnのMovePickerでmaterialに従ってorderingする場合など)
	// Position::set()での駒割り初期化のほか、詰み探索のMovePickerや検証処理でも用いる。
	// 通常のdo_move()では、保持している駒割りを差分更新する。
    Value material(const Position& pos);

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

	// 駒の価値のテーブル
	// ⚠ 後手の駒に対してもプラスの値が返る。
	extern int PieceValue[PIECE_NB];

	// 駒の価値、駒割計算用
	// ⚠ 後手の駒が負の値
    extern int PieceValueM[PIECE_NB];

	// 駒の交換値(＝捕獲したときの価値の上昇値)
	// 例)「と」を取ったとき、評価値の変動量は手駒歩+盤面の「と」。
	// MovePickerとSEEの計算で用いる。
	// ⚠  後手の駒に対してもプラスの値が返る。
	extern int CapturePieceValue[PIECE_NB];

	// 駒を成ったときの成る前との価値の差。SEEで用いる。
	// 駒の成ったものと成っていないものとの価値の差
	// 💡　PAWNでもPRO_PAWNでも　と金 - 歩 の価値が返る。
	// ⚠  後手の駒に対してもプラスの値が返る。
	extern int ProDiffPieceValue[PIECE_NB];

#if defined(USE_BONA_PIECE)

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

#if defined (EVAL_MATERIAL) || defined (EVAL_KPPT) || defined(EVAL_KPP_KKPT) || (defined(EVAL_NNUE) || defined(EVAL_SFNN))
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

		// 玉についても、駒種・升に対応する入力番号を定義する。
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

    // 入力番号は従来のnn.bin / KPPTと互換。配列は評価時だけ生成し、Positionには保持しない。
    struct BonaPieceList {
        alignas(32) BonaPiece pieces[2][40];
        static constexpr int length() { return 38; }
    };

    void collect_bona_pieces(const Position& pos, Color perspective, BonaPiece* pieces);
    BonaPieceList collect_bona_pieces(const Position& pos);

    inline ExtBonaPiece board_bona_piece(Piece pc, Square s) {
        return {BonaPiece(kpp_board_index[pc].fb + s),
                BonaPiece(kpp_board_index[pc].fw + Inv(s))};
    }

    inline ExtBonaPiece hand_bona_piece(Color c, PieceType pt, int slot) {
        return {BonaPiece(kpp_hand_index[c][pt].fb + slot),
                BonaPiece(kpp_hand_index[c][pt].fw + slot)};
    }

    // 着手が変える盤面・手駒だけを記録する。特徴量番号への変換は評価側で行う。
    // handSlotは、打つ手駒または捕獲で追加する手駒の0始まりの位置。
    struct DirtyPiece {
        Piece pc = NO_PIECE;
        Piece after = NO_PIECE;
        Piece captured = NO_PIECE;
        Square from = SQ_NONE;
        Square to = SQ_NONE;
        int handSlot = 0;

        int count() const { return pc == NO_PIECE ? 0 : captured == NO_PIECE ? 1 : 2; }
        bool king_moved() const { return type_of(pc) == KING; }
        bool is_king(int i) const { return i == 0 && king_moved(); }
        ChangedBonaPiece change(int i) const {
            if (i == 0)
                return {from == SQ_NONE ? hand_bona_piece(color_of(pc), raw_type_of(pc), handSlot)
                                        : board_bona_piece(pc, from),
                        board_bona_piece(after, to)};
            return {board_bona_piece(captured, to),
                    hand_bona_piece(color_of(pc), raw_type_of(captured), handSlot)};
        }
    };
#endif // defined(USE_BONA_PIECE)



} // namespace Eval
} // namespace YaneuraOu

#endif // #ifndef EVALUATE_H_INCLUDED
