#include "../shogi.h"
#include "../position.h"
#include "../evaluate.h"
#include "../misc.h"

// 全評価関数に共通の処理などもここに記述する。

// 実験中の(非公開の)評価関数の.cppの読み込みはここで行なう。
#if defined (EVAL_HELICES)
#include "helices/evaluate_helices.cpp"
#include "helices/evaluate_helices_learner.cpp"
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

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_KPPPT) || defined(EVAL_KPPP_KKPT) || defined(EVAL_HELICES) || defined(EVAL_NABLA)

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum)
	{
		// 評価関数ファイルの正体
		std::string softname = "unknown";

		// ソフト名自動判別
		std::map<u64, std::string> list = {
			// KPPT型
			{ 0x0000d6799926233 , "ShinYane(20161010)"    } , // 真やねうら王 = やねうら王(2016)
			{ 0x0000da0f36d1b4b , "Apery(WCSC26)"         } , // Apery(WCSC26)
			{ 0x0000d7213c45779 , "Ukamuse(sdt4)"         } , // 浮かむ瀬     = Apery(sdt4)

			{ 0x0000c9e81cef72b , "elmo(WCSC27)"          } , // elmo(WCSC27)
			{ 0x0000900f1fbe7a6 , "Yomita(WCSC27)"        } , // 読み太(WCSC27)   
			{ 0x0000d5214c4e6d8 , "Qhapaq(WCSC27)"        } , // Qhapaq(WCSC27)
			{ 0x0000d291a9942bb , "tanuki(WCSC27)"        } , // tanuki(WCSC27)

			{ 0x000000000000000 , "rezero epoch0"         } , // リゼロ評価関数epoch0
			{ 0x00008b66a4ecea9 , "rezero epoch0.1"       } , // リゼロ評価関数epoch0.1
			{ 0x000093a87cc91ad , "rezero epoch1"         } , // リゼロ評価関数epoch1
			{ 0x000096a5706b9ec , "rezero epoch2"         } , // リゼロ評価関数epoch2
			{ 0x000096b3df75de8 , "rezero epoch3"         } , // リゼロ評価関数epoch3
			{ 0x0000964a98dc84b , "rezero epoch4"         } , // リゼロ評価関数epoch4
			{ 0x000095f8443a079 , "rezero epoch5"         } , // リゼロ評価関数epoch5
			{ 0x0000953fbdc48d6 , "rezero epoch6"         } , // リゼロ評価関数epoch6
			{ 0x0000a4896775602 , "rezero epoch7"         } , // リゼロ評価関数epoch7
			{ 0x0000a516345897e , "rezero epoch8"         } , // リゼロ評価関数epoch8
			{ 0x0000a6eea7db958 , "yaseiyomita(20170626)" } , // 野生の読み太(20170626)
			{ 0x0000a6fbc5087ce , "yaseiyomita(20170703)" } , // 野生の読み太(20170703)
			{ 0x0000a63c328ca42 , "yaseiyomita(20170802)" } , // 野生の読み太(20170802)
			{ 0x0000c8e89414b88 , "elmoqhapaq ver1.0"     } , // elmo-qhapaq version 1.0
			{ 0x0000c8ac0b10c78 , "elmoqhapaq ver1.1"     } , // elmo-qhapaq version 1.1

			{ 0x0000c62cf4f6b91 , "relmo8"                } , // リゼロ評価関数epoch8とelmo(WCSC27)を1:1でブレンド
			{ 0x0000c85005252ee , "yaselmo(20170705)"     } , // 野生の読み太(20170703)とelmo(WCSC27)を1:1でブレンド

			// KPP_KKPT型
			{ 0x0000574a685513c , "kppkkpt rezero epoch4" } , // KPP_KKPT型リゼロ評価関数epoch4

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
