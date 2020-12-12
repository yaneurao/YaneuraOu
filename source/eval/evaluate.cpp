#include "../types.h"
#include "../position.h"
#include "../evaluate.h"
#include "../misc.h"
#include <map>

namespace Eval
{
#if !defined(USE_EVAL)
	// 評価関数を用いない時用に、ここでダミーの評価関数を定義しておく。

	void init() {}
	Value compute_eval(const Position& pos) { return VALUE_ZERO; }
	void evaluate_with_no_return(const Position& pos) {}
	void print_eval_stat(Position& pos) {}
	void load_eval() {}
	Value evaluate(const Position& pos) { return VALUE_ZERO; }

#endif

#if defined(USE_PIECE_VALUE)
	// 何らかの評価関数を用いる以上、駒割りの計算は必須。
	// 評価関数を一切呼び出さないならこの計算は要らないが、
	// 実行時のオーバーヘッドは小さいので、そこまで考慮することもなさげ。

	// 駒割りの計算
	// 先手側から見た評価値
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

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)

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

#if defined(USE_EVAL_LIST)
	// 内部で保持しているpieceListFb[]が正しいBonaPieceであるかを検査する。
	// 注 : デバッグ用。遅い。
	bool EvalList::is_valid(const Position& pos)
	{
		// 各駒種の手駒の最大枚数
		int hand_max[KING] = { 0,18/*歩*/,4/*香*/,4/*桂*/,4/*銀*/,2/*角*/,2/*飛*/,4/*金*/ };

		for (int i = 0; i < length() ; ++i)
		{
			BonaPiece fb = pieceListFb[i];
			// このfbが本当に存在するかをPositionクラスのほうに調べに行く。

			// 範囲外
			if (!( 0 <= fb && fb < fe_end))
				return false;

			if (fb < fe_hand_end)
			{
				// 手駒なので手駒のほう調べに行く。
				for(auto c : COLOR)
					for (PieceType pr = PAWN; pr < KING; ++pr)
					{
						// 駒ptの手駒のBonaPieceの開始番号
						auto s = kpp_hand_index[c][pr].fb;
						if (s <= fb && fb < s + hand_max[pr])
						{
							// 見つかったのでこの駒の手駒の枚数がnに一致するか調べる。
							// ただしFV38だと手駒の枚数だけBonaPieceが存在する。
							// 例えば、先手が歩を5枚持っているならBonaPieceとして
							// f_hand_pawn,f_hand_pawn+1,..,f_hand_pawn+4がpieceListFbに存在する。
							int n = (int)(fb - s) + 1;
							if (hand_count(pos.hand_of(c),pr) < n)
								return false;

							goto Found;
						}
					}
			}
			else {
				// 盤上の駒なのでこの駒が本当に存在するか調べにいく。
				for (Piece pc = NO_PIECE; pc < PIECE_NB; ++pc)
				{
					PieceType pt = type_of(pc);
					if (pt == NO_PIECE_TYPE || pt == QUEEN) // 存在しない駒
						continue;

					// 駒pcのBonaPieceの開始番号
					auto s = BonaPiece(kpp_board_index[pc].fb);
					if (s <= fb && fb < s + SQ_NB)
					{
						// 見つかったのでこの駒がsqの地点にあるかを調べる。
						Square sq = (Square)(fb - s);
						Piece pc2 = pos.piece_on(sq);

						// BonaPieceでは、歩成,成香,成桂,成銀も金扱いなので、
						// 盤上の駒がこれらであるなら金に変更しておく。
						PieceType pt2 = type_of(pc2);
						if (pt2 == PRO_PAWN || pt2 == PRO_LANCE || pt2 == PRO_KNIGHT || pt2 == PRO_SILVER)
							pc2 = make_piece(color_of(pc2), GOLD);

						if (pc2 != pc)
							return false;

						goto Found;
					}
				}
			}
			// 何故か存在しない駒であった..
			return false;
		Found:;
		}

		return true;
	}
#endif

} // namespace Eval
