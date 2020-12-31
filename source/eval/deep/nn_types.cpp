#include "nn_types.h"

#if defined(EVAL_DEEP)

#include <cstring> // memset,wchar_t
#include <cmath>   // expf,logf

#include "../../usi.h"

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


	// 入力特徴量を生成する。
	//   position  : このあとEvalNode()を呼び出したい局面
	//   features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, NN_Input1* features1, NN_Input2* features2)
	{
		// set all zero
		// 特徴量の配列をゼロ初期化
		std::fill_n((DType*)features1, sizeof(NN_Input1)/sizeof(DType) , dtype_zero );
		std::fill_n((DType*)features2, sizeof(NN_Input2)/sizeof(DType) , dtype_zero );

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
						(*features1)[c2][pt - 1][sq2] = dtype_one;

					// 駒種ごとの利き(有るか無いか)
					if (attacks[c][pt].test(sq))
						(*features1)[c2][PIECETYPE_NUM + pt - 1][sq2] = dtype_one;
				}

				// ある升に対する利き数。MAX_ATTACK_NUM以上の利きは、MAX_ATTACK_NUM個であるとみなす。
				const int num = std::min(MAX_ATTACK_NUM, position.attackers_to(c, sq, occupied_bb).pop_count());
				for (int k = 0; k < num; k++)
					// 利きの数のlayer数だけ、各layerに対してその升を1にしておく。
					(*features1)[c2][PIECETYPE_NUM + PIECETYPE_NUM + k][sq2] = dtype_one;
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
			auto features2_hand = reinterpret_cast<DType(*)[COLOR_NB][MAX_PIECES_IN_HAND_SUM][SQ_NB]>(features2);
			Hand hand = position.hand_of(c);
			int p = 0;
			for (int hp = 0; hp < HandPieceNum; ++hp)
			{
				PieceType pt = HandPiece2PieceType[hp];
				int num = std::min(hand_count(hand, pt), MAX_PIECES_IN_HAND[hp]);
				std::fill_n((*features2_hand)[c2][p], (int)SQ_NB * num, dtype_one);
				p += MAX_PIECES_IN_HAND[hp]; // 駒種ごとに割り当てられているlayer数が決まっているので、次の駒種用のlayerにいく。
			}
		}

		// 王手がかかっているか(のlayerが1枚)
		if (position.in_check()) {
 			std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SQ_NB, dtype_one);
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
			Options["DNN_Model5"], Options["DNN_Model6"], Options["DNN_Model7"], Options["DNN_Model8"]
		};

		string eval_dir = Options["EvalDir"];

		ModelPaths.clear();

		// モデルファイル存在チェック
		bool is_err = false;
		for (int i = 0; i < max_gpu; ++i) {
			if (model_paths[i] != "")
			{
				string path = Path::Combine(eval_dir, model_paths[i].c_str());
				std::ifstream ifs(path);
				if (!ifs.is_open()) {
					sync_cout << "Error! : " << path << " file not found" << sync_endl;
					is_err = true;
					break;
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
	float cp_to_value(const Value score)
	{
		return 1.0f / (1.0f + expf(-(float)score * 0.0013226f));
	}

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	Value value_to_cp(const float score)
	{
		int cp;
		if (score == 1.0f)
			cp =  30000;
		else if (score == 0.0f)
			cp = -30000;
		else
		{
			cp = (int)(-logf(1.0f / score - 1.0f) * 756.0864962951762f);

			// 勝率がオーバーフローしてたらclampしとく。
			cp = std::min(cp, +29900);
			cp = std::max(cp, -29900);
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
