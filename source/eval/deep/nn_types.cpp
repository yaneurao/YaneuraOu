#include "nn_types.h"

#if defined(EVAL_DEEP)

#include <cstring> // memset,wchar_t
#include <cmath>   // expf,logf

#include "../../usi.h"
#include "../../bitboard.h"

using namespace std;
using namespace Tools;

namespace
{
	// 指し手に対して、Policy Networkの返してくる配列のindexを返すためのテーブル
	// Eval::init()で初期化する。
	u16 MoveLabel[0x10000][COLOR_NB];
}


namespace Eval::dlshogi
{
	// モデルファイル名へのpath
	std::vector<std::string> ModelPaths;

	// Aperyの手駒は、GOLDが末尾になっていないので変換テーブルを用意する。
	PieceType HandPiece2PieceType[HandPieceNum ] = { PAWN , LANCE , KNIGHT , SILVER , GOLD , BISHOP , ROOK };
	//int       PieceType2HandPiece[PIECE_TYPE_NB] = { 0 , 1 , 2 , 3 , 4 , 6 , 7 , 5 };

#if defined(TRT_NN_FP16)
	const DType dtype_zero = __float2half(0.0f);
	const DType dtype_one  = __float2half(1.0f);
#endif

#if 0 // dlshogiに忠実に書かれたコード

	// 入力特徴量を生成する。
	//   position  : このあとEvalNode()を呼び出したい局面
	//   packed_features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   packed_features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2)
	{
		int f1idx_b = batch_index * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB);
		int f2idx_b = batch_index * ((int)MAX_FEATURES2_NUM);
		// set all zero
		// 特徴量の配列をゼロ初期化
		{
			// std::fill_n((DType*)features1, sizeof(NN_Input1)/sizeof(DType) , dtype_zero );
			// std::fill_n((DType*)features2, sizeof(NN_Input2)/sizeof(DType) , dtype_zero );
			const PType bmask[8] = { 0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f };
			const PType emask[8] = { 0xfe, 0xfc, 0xf8, 0xf0, 0xe0, 0xc0, 0x80, 0x00 };
			int f1idx_e = f1idx_b + ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB - 1);
			int f2idx_e = f2idx_b + ((int)MAX_FEATURES2_NUM - 1);
			int f1idx_bu = f1idx_b >> 3;
			int f1idx_bl = f1idx_b & 7;
			int f1idx_eu = f1idx_e >> 3;
			int f1idx_el = f1idx_e & 7;
			int f2idx_bu = f2idx_b >> 3;
			int f2idx_bl = f2idx_b & 7;
			int f2idx_eu = f2idx_e >> 3;
			int f2idx_el = f2idx_e & 7;
			packed_features1[f1idx_bu] &= bmask[f1idx_bl];
			std::fill_n(&packed_features1[f1idx_bu + 1], f1idx_eu - f1idx_bu - 1, (PType)0);
			packed_features1[f1idx_eu] &= emask[f1idx_el];
			packed_features2[f2idx_bu] &= bmask[f2idx_bl];
			std::fill_n(&packed_features2[f2idx_bu + 1], f2idx_eu - f2idx_bu - 1, (PType)0);
			packed_features2[f2idx_eu] &= emask[f2idx_el];
		}

		const Bitboard occupied_bb = position.pieces();

		// 駒種ごとの利き・先後の区別はある。
		Bitboard attacks[COLOR_NB][PieceTypeNum];
		memset(&attacks, 0, sizeof(attacks));
		// これ本当にゼロクリア必要なのか？あとで

		for (Square sq = SQ_11; sq < SQ_NB; sq++)
		{
			Piece pc = position.piece_on(sq);
			if (pc != NO_PIECE)
				attacks[color_of(pc)][type_of(pc)] |= effects_from(pc, sq, occupied_bb);
		}

		// 入力特徴量1を生成する。
		for (Color c = BLACK; c < COLOR_NB ; ++c)
		{
			// 手番が後手の場合、色を反転して考える。
			Color c2 = (position.side_to_move() == BLACK) ? c : ~c;

			// 駒種ごとの配置
			Bitboard bb[PieceTypeNum];
			memset(&bb, 0, sizeof(bb));

			for (PieceType pt = PAWN; pt < (u32)PieceTypeNum; ++pt)
				bb[pt] = position.pieces(c , pt);

			for (Square sq = SQ_11; sq < SQ_NB; ++sq)
			{
				// 後手の場合、盤面を180度回転させて考える。
				Square sq2 = (position.side_to_move() == BLACK) ? sq : Flip(sq);

				// 以下、(盤面の)参照先は、cとsqによって表現されるが、(NNに渡すための特徴量の)格納先は、c2とsq2である。

				for (PieceType pt = PAWN; pt < (u32)PieceTypeNum; ++pt)
				{
					// 駒の配置
					if (bb[pt].test(sq))
					{
						// (*features1)[c2][pt - 1][sq2] = dtype_one;
						int f1idx = f1idx_b + (int)c2 * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)pt - 1) * ((int)SQ_NB) + (int)sq2;
						packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
					}
					// 駒種ごとの利き(有るか無いか)
					if (attacks[c][pt].test(sq))
					{
						// (*features1)[c2][PIECETYPE_NUM + pt - 1][sq2] = dtype_one;
						int f1idx = f1idx_b + (int)c2 * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + (PIECETYPE_NUM + pt - 1) * ((int)SQ_NB) + (int)sq2;
						packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
					}
				}

				// ある升sqに対するc側の利き数。MAX_ATTACK_NUM以上の利きは、MAX_ATTACK_NUM個であるとみなす。
				const int num = std::min(MAX_ATTACK_NUM, position.attackers_to(c, sq, occupied_bb).pop_count());
				for (int k = 0; k < num; k++)
				{
					// 利きの数のlayer数だけ、各layerに対してその升を1にしておく。
					// (*features1)[c2][PIECETYPE_NUM + PIECETYPE_NUM + k][sq2] = dtype_one;
					int f1idx = f1idx_b + (int)c2 * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + (PIECETYPE_NUM + PIECETYPE_NUM + k) * ((int)SQ_NB) + (int)sq2;
					packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
				}
			}

			// 手駒
			/*
				dlshogiのNNへの入力特徴量の手駒、順番がAperyのHandPiece enumの順になってて、これが普通に駒順(PieceType)と異なり、
				末尾がRookなのでわりと嫌らしい。(これに気づくのに1時間ぐらい要した)
				※ PieceTypeは末尾がGold。

				そもそもで言うと、Aperyが手駒をいつまでもPieceTypeと異なる順にしているのが悪いのだけども(Aperyのpiece.hppに
				次のようなコメントがある)、dlshogiでは、これはどちらかに統一したほうが良かったのではなかろうか。
				(やねうら王的には前者に統一してあるほうが嬉しい)

				> // 持ち駒を表すときに使用する。
				> // todo: HGold を HRook の後ろに持っていき、PieceType との変換を簡単に出来るようにする。

				これ、あとでdlshogiのNNを自分の探索部で使おうとする人は、同じ問題でハマると思う。😥
			*/

			// NN_Input2は、[COLOR_NB * MAX_PIECES_IN_HAND_SUM + 王手か(1) ][SQ_NB]
			// なので、この一つ目のindexを[COLOR_NB][MAX_PIECES_IN_HAND_SUM]と解釈しなおす。
			// ※　こうしたほうがコードが簡単になるので。
			// auto features2_hand = reinterpret_cast<DType(*)[COLOR_NB][MAX_PIECES_IN_HAND_SUM][SQ_NB]>(features2);
			Hand hand = position.hand_of(c);
			int p = 0;
			for (int hp = 0; hp < HandPieceNum; ++hp)
			{
				PieceType pt = HandPiece2PieceType[hp];
				int num = std::min(hand_count(hand, pt), MAX_PIECES_IN_HAND[hp]);
				{
					// std::fill_n((*features2_hand)[c2][p], (int)SQ_NB * num, dtype_one);
					int f2idx = f2idx_b + (int)c2 * ((int)MAX_PIECES_IN_HAND_SUM) + p;
					for (int i = 0; i < num; ++i) {
						packed_features2[(f2idx + i) >> 3] |= (1 << ((f2idx + i) & 7));
					}
				}
				p += MAX_PIECES_IN_HAND[hp]; // 駒種ごとに割り当てられているlayer数が決まっているので、次の駒種用のlayerにいく。
			}
		}

		// 王手がかかっているか(のlayerが1枚)
		if (position.in_check()) {
			// std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SQ_NB, dtype_one);
			int f2idx = f2idx_b + (int)MAX_FEATURES2_HAND_NUM;
			packed_features2[f2idx >> 3] |= (1 << (f2idx & 7));
		}
	}

#endif

#if 1 // 頑張って独自の最適化を行ったコード

	// make_input_features()の下請け。
	// SideToMove : 現局面の手番。
	// 引数の意味は、make_input_features()と同じ。
	template <Color SideToMove>
	void make_input_features_sub(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2)
	{
		int f1idx_b = batch_index * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB);
		int f2idx_b = batch_index * ((int)MAX_FEATURES2_NUM);
		// set all zero
		// 特徴量の配列をゼロ初期化
		{
			// std::fill_n((DType*)features1, sizeof(NN_Input1)/sizeof(DType) , dtype_zero );
			// std::fill_n((DType*)features2, sizeof(NN_Input2)/sizeof(DType) , dtype_zero );
			const PType bmask[8] = { 0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f };
			const PType emask[8] = { 0xfe, 0xfc, 0xf8, 0xf0, 0xe0, 0xc0, 0x80, 0x00 };
			int f1idx_e = f1idx_b + ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB - 1);
			int f2idx_e = f2idx_b + ((int)MAX_FEATURES2_NUM - 1);
			int f1idx_bu = f1idx_b >> 3;
			int f1idx_bl = f1idx_b & 7;
			int f1idx_eu = f1idx_e >> 3;
			int f1idx_el = f1idx_e & 7;
			int f2idx_bu = f2idx_b >> 3;
			int f2idx_bl = f2idx_b & 7;
			int f2idx_eu = f2idx_e >> 3;
			int f2idx_el = f2idx_e & 7;
			packed_features1[f1idx_bu] &= bmask[f1idx_bl];
			std::fill_n(&packed_features1[f1idx_bu + 1], f1idx_eu - f1idx_bu - 1, (PType)0);
			packed_features1[f1idx_eu] &= emask[f1idx_el];
			packed_features2[f2idx_bu] &= bmask[f2idx_bl];
			std::fill_n(&packed_features2[f2idx_bu + 1], f2idx_eu - f2idx_bu - 1, (PType)0);
			packed_features2[f2idx_eu] &= emask[f2idx_el];
		}

		// 各升の利きの数の集計用の配列
		u8 effect_num[SQ_NB][COLOR_NB] = {};

		// 盤上の駒のある場所
		const Bitboard pieces = position.pieces();

		// 歩
		auto pawn_bb = position.pieces(PAWN);

		// 歩以外の盤上の駒
		auto pieces_without_pawns = position.pieces() & ~pawn_bb;

		// 先手の歩 , 後手の歩
		auto pawn_black = pawn_bb & position.pieces(BLACK);
		auto pawn_white = pawn_bb & position.pieces(WHITE);

		// 歩以外の駒それぞれに対して
		pieces_without_pawns.foreach([&](Square sq) {
			Piece pc = position.piece_on(sq);
			auto attacks = effects_from(pc, sq, pieces);
			Color c = color_of(pc);

			/*後手なら符号を反転させる*/;
			if (SideToMove == WHITE)
			{
				c = ~c;
				sq = Flip(sq);
			}

			{
				// この駒のある場所を1にする
				// (*features1)[c][type_of(pc) - 1][sq] = dtype_one;
				int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)type_of(pc) - 1) * ((int)SQ_NB) + (int)sq;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}

			// この駒の利きを1にする
			attacks.foreach([&](Square to) {
				// 後手なら、180度盤面を回転させた場所に
				if (SideToMove == WHITE)
					to = Flip(to);

				{
					// この駒の利きのある場所を1にする
					// (*features1)[c][PIECETYPE_NUM + type_of(pc) - 1][to] = dtype_one;
					int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PIECETYPE_NUM + (int)type_of(pc) - 1) * ((int)SQ_NB) + (int)to;
					packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
				}

				// 各升の利きの数の集計用
				effect_num[to][c]++;
			});
		});

		// 先手の歩

		pawn_black.foreach([&](Square sq) {
			if (SideToMove == WHITE)
				sq = Flip(sq);
			Color c = SideToMove;

			// 歩の升を1にする
			{
				// (*features1)[c][PAWN - 1][sq] = dtype_one;
				int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PAWN - 1) * ((int)SQ_NB) + (int)sq;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}

			// 歩の利きのある升を1にする。
			Square to = (Square)(sq + (SideToMove == BLACK ? -1 : +1)) /*1升上*/;
			{
				// (*features1)[c][PIECETYPE_NUM + PAWN - 1][to] = dtype_one;
				int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PIECETYPE_NUM + (int)PAWN - 1) * ((int)SQ_NB) + (int)to;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}

			// 各升の利きの数の集計用
			effect_num[to][c]++;
		});

		// 後手の歩

		pawn_white.foreach([&](Square sq) {
			if (SideToMove == WHITE)
				sq = Flip(sq);
			Color c = ~SideToMove;

			{
				// (*features1)[c][PAWN - 1][sq] = dtype_one;
				int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PAWN - 1) * ((int)SQ_NB) + (int)sq;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}
			Square to = (Square)(sq + (SideToMove == BLACK ? +1 : -1)) /*1升下*/;
			{
				// (*features1)[c][PIECETYPE_NUM + PAWN - 1][to] = dtype_one;
				int f1idx = f1idx_b + (int)c * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PIECETYPE_NUM + (int)PAWN - 1) * ((int)SQ_NB) + (int)to;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}

			effect_num[to][c]++;
		});

		// 利きの数を書いたlayerを生成する。
		// effect_numの集計結果が入っている。後手番なら盤面を180度回転させた時の値(先後が逆、升も逆)が入っている。
		for (Square sq = SQ_11; sq < SQ_NB; ++sq)
		{
			// ある升sqに対するc側の利き数。MAX_ATTACK_NUM以上の利きは、MAX_ATTACK_NUM個であるとみなす。
			int num = std::min(MAX_ATTACK_NUM, (int)effect_num[sq][BLACK]);
			for (int k = 0; k < num; k++)
			{
				// 利きの数のlayer数だけ、各layerに対してその升を1にしておく。
				// (*features1)[BLACK][PIECETYPE_NUM + PIECETYPE_NUM + k][sq] = dtype_one;
				int f1idx = f1idx_b + (int)BLACK * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PIECETYPE_NUM + (int)PIECETYPE_NUM + k) * ((int)SQ_NB) + (int)sq;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}

			// 後手も同様
			num = std::min(MAX_ATTACK_NUM, (int)effect_num[sq][WHITE]);
			for (int k = 0; k < num; k++)
			{
				// (*features1)[WHITE][PIECETYPE_NUM + PIECETYPE_NUM + k][sq] = dtype_one;
				int f1idx = f1idx_b + (int)WHITE * ((int)MAX_FEATURES1_NUM * (int)SQ_NB) + ((int)PIECETYPE_NUM + (int)PIECETYPE_NUM + k) * ((int)SQ_NB) + (int)sq;
				packed_features1[f1idx >> 3] |= (1 << (f1idx & 7));
			}
		}

		// 手駒
		for (Color c = BLACK; c < COLOR_NB ; ++c)
		{
			// 手番が後手の場合、色を反転して考える。
			Color c2 = (position.side_to_move() == BLACK) ? c : ~c;
			/*
				dlshogiのNNへの入力特徴量の手駒、順番がAperyのHandPiece enumの順になってて、これが普通に駒順(PieceType)と異なり、
				末尾がRookなのでわりと嫌らしい。(これに気づくのに1時間ぐらい要した)
				※ PieceTypeは末尾がGold。

				そもそもで言うと、Aperyが手駒をいつまでもPieceTypeと異なる順にしているのが悪いのだけども(Aperyのpiece.hppに
				次のようなコメントがある)、dlshogiでは、これはどちらかに統一したほうが良かったのではなかろうか。
				(やねうら王的には前者に統一してあるほうが嬉しい)

				> // 持ち駒を表すときに使用する。
				> // todo: HGold を HRook の後ろに持っていき、PieceType との変換を簡単に出来るようにする。

				これ、あとでdlshogiのNNを自分の探索部で使おうとする人は、同じ問題でハマると思う。😥
			*/

			// NN_Input2は、[COLOR_NB * MAX_PIECES_IN_HAND_SUM + 王手か(1) ][SQ_NB]
			// なので、この一つ目のindexを[COLOR_NB][MAX_PIECES_IN_HAND_SUM]と解釈しなおす。
			// ※　こうしたほうがコードが簡単になるので。
			// auto features2_hand = reinterpret_cast<DType(*)[COLOR_NB][MAX_PIECES_IN_HAND_SUM][SQ_NB]>(features2);
			Hand hand = position.hand_of(c);
			int p = 0;
			for (int hp = 0; hp < HandPieceNum; ++hp)
			{
				PieceType pt = HandPiece2PieceType[hp];
				const int mp = MAX_PIECES_IN_HAND[hp];
				int num = std::min(hand_count(hand, pt), mp);
				// std::fill_n((*features2_hand)[c2][p] , (int)SQ_NB * num, dtype_one);
				int f2idx = f2idx_b + (int)c2 * ((int)MAX_PIECES_IN_HAND_SUM) + p;
				for (int i = 0; i < num; ++i) {
					packed_features2[(f2idx + i) >> 3] |= (1 << ((f2idx + i) & 7));
				}

				// そこから後ろをゼロクリア
				// int rest = mp - num;
				// if (rest)
				//	std::fill_n((*features2_hand)[c2][p+num] ,(int)SQ_NB * rest, dtype_zero);

				p += MAX_PIECES_IN_HAND[hp]; // 駒種ごとに割り当てられているlayer数が決まっているので、次の駒種用のlayerにいく。
			}
		}

		if (position.in_check()) {
			// 王手がかかっているか(のlayerが1枚)
			// std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SQ_NB, position.in_check() ? dtype_one : dtype_zero);
			int f2idx = f2idx_b + (int)MAX_FEATURES2_HAND_NUM;
			packed_features2[f2idx >> 3] |= (1 << (f2idx & 7));
		}
	}


	// 入力特徴量を生成する。
	//   position  : このあとEvalNode()を呼び出したい局面
	//   packed_features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   packed_features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2)
	{
		position.side_to_move() == BLACK
			? make_input_features_sub<BLACK>(position, batch_index, packed_features1, packed_features2)
			: make_input_features_sub<WHITE>(position, batch_index, packed_features1, packed_features2);
	}
#endif

	// 入力特徴量を展開する。GPU側で展開する場合は不要。
	void extract_input_features(int batch_size, PType* packed_features1, PType* packed_features2, NN_Input1* features1, NN_Input2* features2)
	{
		int p1len = batch_size * ((int)COLOR_NB * (int)MAX_FEATURES1_NUM * (int)SQ_NB);
		int p2len = batch_size * ((int)MAX_FEATURES2_NUM);
		DType* f1 = (DType*)features1;
		DType* f2 = (DType*)features2;
		for (int i = 0; i < p1len; ++i)
		{
			f1[i] = ((packed_features1[i >> 3] >> (i & 7)) & 1) != 0 ? dtype_one : dtype_zero;
		}
		for (int i = 0; i < p2len; ++i)
		{
			std::fill_n(&f2[i * (int)SQ_NB], (int)SQ_NB, ((packed_features2[i >> 3] >> (i & 7)) & 1) != 0 ? dtype_one : dtype_zero);
		}
	}

	// MoveLabel配列を事前に初期化する。
	// "isready"に対して呼び出される。
	void init_move_label()
	{
		// 指し手としてありうる組み合わせ全部のMove16を生成して、
		// それの指し手に足してどんなlabelをつけるのかdlshogiのルーチンを
		// そのまま再現して、その返し値でMoveLabel配列を初期化する。

		// dlshogiのmake_move_label()
		auto dl_make_move_label = [](Move16 move16, Color color)
		{
			Square   to =   to_sq(move16);
			Square from = from_sq(move16);
			bool   drop = is_drop(move16);

			// move direction
			int move_direction_label;
			if (!drop) {
				// 白の場合、盤面を180度回転
				if (color == WHITE) {
					to   =  Flip(to  );
					from =  Flip(from);
				}

				const div_t to_d   = div(to   , 9);
				const int to_x     = to_d.quot;
				const int to_y     = to_d.rem;
				const div_t from_d = div(from , 9);
				const int from_x   = from_d.quot;
				const int from_y   = from_d.rem;
				const int dir_x    = from_x - to_x;
				const int dir_y    = to_y - from_y;

				MOVE_DIRECTION move_direction = MOVE_DIRECTION_NONE; // どれにも該当しないときは、このラベルにする。
				if (dir_y < 0 && dir_x == 0) {
					move_direction = UP;
				}
				else if (dir_y == -2 && dir_x == -1) {
					move_direction = UP2_LEFT;
				}
				else if (dir_y == -2 && dir_x == 1) {
					move_direction = UP2_RIGHT;
				}
				else if (dir_y < 0 && dir_x < 0) {
					move_direction = UP_LEFT;
				}
				else if (dir_y < 0 && dir_x > 0) {
					move_direction = UP_RIGHT;
				}
				else if (dir_y == 0 && dir_x < 0) {
					move_direction = LEFT;
				}
				else if (dir_y == 0 && dir_x > 0) {
					move_direction = RIGHT;
				}
				else if (dir_y > 0 && dir_x == 0) {
					move_direction = DOWN;
				}
				else if (dir_y > 0 && dir_x < 0) {
					move_direction = DOWN_LEFT;
				}
				else if (dir_y > 0 && dir_x > 0) {
					move_direction = DOWN_RIGHT;
				}

				// promote
				if (is_promote(move16)) {
					move_direction = MOVE_DIRECTION_PROMOTED[move_direction];
				}
				move_direction_label = move_direction;
			}
			// 駒打ちの場合
			else {
				// 後手の駒打ちなのでtoの場所だけを反転。
				if (color == WHITE)
					to = Flip(to);
				PieceType pt = move_dropped_piece(move16);
				const int hand_piece = int(pt - PAWN); // Aperyの駒打ち、駒の順番はPieceType順なのでPieceType2HandPiece[pt];は不要。
				move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
			}

			return 9 * 9 * move_direction_label + to;
		};

		for (auto color : COLOR)
			// 手駒はfromの位置がSQ_NB～SQ_NB+6
			for(Square from_sq = SQ_ZERO ;from_sq < SQ_NB + 7 ; ++from_sq)
				for (auto to_sq : SQ)
					// 成りと成らずと
					for (int promote = 0; promote < 2; ++promote)
					{
						// 駒打ちであるか
						bool drop = from_sq >= SQ_NB;

						// 駒打ちの成りはない
						if (drop && promote)
							continue;

						Move16 move;
						if (!drop)
						{
							move = !promote
								? make_move16(from_sq, to_sq)
								: make_move_promote16(from_sq, to_sq);
						}
						else {
							PieceType pt = (PieceType)(from_sq - (int)SQ_NB + PAWN);
							move = make_move_drop16(pt, to_sq);
						}

						// dlshogiのmake_move_label()を呼び出して初期化する。
						MoveLabel[move.to_u16()][color] = dl_make_move_label(move, color);
					}

#if 0
		std::set<u16> s;
		for (auto c : COLOR)
			for (u32 i = 0; i < 0x10000; ++i)
				s.insert(MoveLabel[i][c]);
		std::cout << s.size() << std::endl;
		// 出現する組み合わせは、1969 - 1 = 1968通り。
		// いまは、27*81 = 2187通りPolicy Networkが返してくるので、1割ぐらい削減できそう。
#endif

	}

	// 指し手に対して、Policy Networkの返してくる配列のindexを返す。
	int make_move_label(Move move, Color color)
	{
		return MoveLabel[move & 0xffff][color];
	}

	// Boltzmann distribution
	// see: Reinforcement Learning : An Introduction 2.3.SOFTMAX ACTION SELECTION
	// →　第二版が無料で公開されているので、そちらを参照するようにしたほうが良いのでは。
	//		Reinforcement Learning: An Introduction 2nd Ed.
	//		http://incompleteideas.net/book/the-book.html

	// Softmaxの時の温度パラメーター。
	// エンジンオプションの"Softmax_Temperature"で設定できる。
	constexpr float default_softmax_temperature = 1.0f;
	float beta = 1.0f / default_softmax_temperature;
	void set_softmax_temperature(const float temperature) {
		beta = 1.0f / temperature;
	}

	void softmax_temperature_with_normalize(std::vector<float> &log_probabilities) {
		// apply beta exponent to probabilities(in log space)
		float max = 0.0f;
		for (float& x : log_probabilities) {
			x *= beta;
			if (x > max) {
				max = x;
			}
		}
		// オーバーフローを防止するため最大値で引く
		float sum = 0.0f;
		for (float& x : log_probabilities) {
			x = expf(x - max);
			sum += x;
		}
		// normalize
		for (float& x : log_probabilities) {
			x /= sum;
		}
	}

	Result init_model_paths()
	{
		const std::string model_paths[max_gpu] = {
			Options["DNN_Model1"], Options["DNN_Model2"], Options["DNN_Model3"], Options["DNN_Model4"],
			Options["DNN_Model5"], Options["DNN_Model6"], Options["DNN_Model7"], Options["DNN_Model8"],
			Options["DNN_Model9"], Options["DNN_Model10"], Options["DNN_Model11"], Options["DNN_Model12"],
			Options["DNN_Model13"], Options["DNN_Model14"], Options["DNN_Model15"], Options["DNN_Model16"]
		};

		string eval_dir = Options["EvalDir"];

		ModelPaths.clear();

		// ファイルが存在することが既知であるpath。
		// (一度調べたやつは記憶しておく)
		// モデルファイルはたかだか1,2個だと思うのでvectorで十分。
		static std::vector<std::string> checked_paths;

		// モデルファイル存在チェック
		bool is_err = false;
		for (int i = 0; i < max_gpu; ++i) {
			if (model_paths[i] != "")
			{
				string path = Path::Combine(eval_dir, model_paths[i].c_str());
				if (std::find(checked_paths.begin(), checked_paths.end(), path) == checked_paths.end())
				{
					// 未チェックのやつなので調べる。
					std::ifstream ifs(path);
					if (!ifs.is_open()) {
						sync_cout << "Error! : " << path << " file not found" << sync_endl;
						is_err = true;
						break;
					}
					// 記憶しておく。
					checked_paths.push_back(path);
				}
				ModelPaths.push_back(path);
			}
			else {
				ModelPaths.push_back("");
			}
		}
		if (is_err)
			return ResultCode::FileOpenError;

		return ResultCode::Ok;
	}

	// エンジンオプションで設定されたモデルファイル名。
	// この返し値のvectorのsize() == max_gpuのはず。
	std::vector<std::string>* get_model_paths()
	{
		return &ModelPaths;
	}

	// 評価値から価値(勝率)に変換
	// スケールパラメータは、elmo_for_learnの勝率から調査した値
	// 何かの変換の時に必要になる。
	float cp_to_value(const Value score , const float eval_coef)
	{
		return 1.0f / (1.0f + expf(-(float)score / eval_coef));
	}

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	// 	 eval_coef : 勝率を評価値に変換する時の定数。default = 756
	//
	// 返し値 :
	//   +29900は、評価値の最大値
	//   -29900は、評価値の最小値
	//   +30000,-30000は、(おそらく)詰みのスコア
	Value value_to_cp(const float score , const float eval_coef)
	{
		int cp;
		if (score == 1.0f)
			cp =  30000;
		else if (score == 0.0f)
			cp = -30000;
		else
		{
			cp = (int)(-logf(1.0f / score - 1.0f) * eval_coef);

			// 勝率がオーバーフローしてたらclampしておく。
			cp = std::clamp(cp, -29900, +29900);
		}

		return (Value)cp;
	}


} // namespace Eval::dlshogi

using namespace Eval::dlshogi;

namespace Eval
{
	void init(){}
	Value compute_eval(const Position& pos) { return VALUE_ZERO; }
	void evaluate_with_no_return(const Position& pos) {}
	void print_eval_stat(Position& pos) {}

	// 時間のかかる初期化処理はここに書く。
	// 毎回呼び出されるようになっている。
	void load_eval()
	{
		// 初回初期化。
		static bool init = false;
		if (!init)
		{
			// 指し手に対して、Policy Networkの返してくる配列のindexを返すテーブルの初期化
			dlshogi::init_move_label();
			init = true;
		}

		// モデルファイル(NNで使う評価関数ファイル)の確認。
		// これは"isready"に対して毎回初期化する。(ファイル名が変わるかも知れないので)
		dlshogi::init_model_paths();
	}

	// 考え中。
	// NN::forward()を呼ぶ実装にするかも。
	Value evaluate(const Position& pos) { return VALUE_ZERO; }
}

#endif // defined(EVAL_DEEP
