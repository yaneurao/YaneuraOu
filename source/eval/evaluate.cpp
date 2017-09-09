#include "../shogi.h"
#include "../position.h"
#include "../evaluate.h"
#include "../misc.h"

// 全評価関数に共通の処理などもここに記述する。

// 実験中の(非公開の)評価関数の.cppの読み込みはここで行なう。
#if defined (EVAL_KPPP_KKPT)
#include "kppp_kkpt/evaluate_kppp_kkpt.cpp"
#include "kppp_kkpt/evaluate_kppp_kkpt_learner.cpp"
#endif
#if defined (EVAL_NABLA)
#include "nabla/evaluate_nabla.cpp"
#include "nabla/evaluate_nabla_learner.cpp"
#endif
#if defined (EVAL_AKASHIC)
#include "akashic/evaluate_akashic.cpp"
#include "akashic/evaluate_akashic_learner.cpp"
#endif

namespace Eval
{
#if !defined (EVAL_NO_USE)
  // 何らかの評価関数を用いる以上、駒割りの計算は必須。
  // すなわち、EVAL_NO_USE以外のときはこの関数が必要。

  // 駒割りの計算
  // 手番側から見た評価値
	Value material(const Position& pos)
	{
		int v = VALUE_ZERO;

		for (auto i : SQ)
			v = v + PieceValue[pos.piece_on(i)];

		// 手駒も足しておく
		for (auto c : COLOR)
			for (auto pc = PAWN; pc < PIECE_HAND_NB; ++pc)
				v += (c == BLACK ? 1 : -1) * Value(hand_count(pos.hand_of(c), pc) * PieceValue[pc]);

		return (Value)v;
	}
#endif

#if defined (EVAL_MATERIAL)
	// 駒得のみの評価関数のとき。
	void init() {}
	void load_eval() {}
	void print_eval_stat(Position& pos) {}
	Value evaluate(const Position& pos) {
		auto score = pos.state()->materialValue;
		ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
		return pos.side_to_move() == BLACK ? score : -score;
	}
	Value compute_eval(const Position& pos) { return material(pos); }
#endif

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_KPPP_KKPT)

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum)
	{
		// 評価関数ファイルの正体
		std::string softname = "unknown";

		// ソフト名自動判別
		std::map<u64, std::string> list = {
			{ 0x0000da0f36d1b4b , "Apery(WCSC26)"         } , // Apery(WCSC26)
			{ 0x0000d7213c45779 , "Ukamuse(sdt4)"         } , // 浮かむ瀬     = Apery(sdt4)

			{ 0x0000c9e81cef72b , "elmo(WCSC27)"          } , // elmo(WCSC27)
			{ 0x0000900f1fbe7a6 , "Yomita(WCSC27)"        } , // 読み太(WCSC27)   
			{ 0x0000d5214c4e6d8 , "Qhapaq(WCSC27)"        } , // Qhapaq(WCSC27)
			{ 0x0000d291a9942bb , "tanuki(WCSC27)"        } , // tanuki(WCSC27)

			{ 0x0000a516345897e , "rezero epoch8"         } , // リゼロ評価関数epoch8
			{ 0x0000a6fbc5087ce , "yaseiyomita(20170703)" } , // 野生の読みの太(20170703)

			// ここに加えて欲しい評価関数があれば、
			// "isready"コマンドに対する応答時に表示されるcheck sumの値を連絡ください(｀･ω･´)ｂ
		};

		if (list.count(check_sum))
			softname = list[check_sum];

		sync_cout << "info string Eval Check Sum = " << std::hex << check_sum << std::dec
			<< " , Eval File = " << softname << sync_endl;
	}
#endif

#if defined (USE_EVAL_MAKE_LIST_FUNCTION)

	// compute_eval()やLearner::add_grad()からBonaPiece番号の組み換えのために呼び出される関数
	std::function<void(const Position&, BonaPiece[40], BonaPiece[40])> make_list_function;

	// 旧評価関数から新評価関数に変換するときにKPPのP(BonaPiece)がどう写像されるのかを定義したmapper。
	// EvalIO::eval_convert()の引数として渡される。
	std::vector<u16 /*BonaPiece*/> eval_mapper;
#endif

}
