#ifndef __DLSHOGI_EVALUATE_H_INCLUDED__
#define __DLSHOGI_EVALUATE_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"

namespace Eval::dlshogi
{
	// === GPU関連の設定 ===

	// GPUの最大数(これ以上のGPUは扱えない)
	constexpr int max_gpu = 8;

	// === 入出力の特徴量の定義 ===

	// 各手駒の上限枚数。

	// 歩は多くなるので最大でも8枚とみなす。
	// 歩が先手の手駒に9枚の状況だとして、残り7枚は相手の手駒 or 盤上にあるはずだし、盤上の歩は入力特徴量として持っているので
	// 駒割自体は正しく計算できるはず。
	// MAX_HPAWN_NUMが7だと、手駒を先手が9枚、後手が7枚持っているような状況だと、どちらが数多く持っているのかが判定できないのでまずい。

	constexpr int MAX_HPAWN_NUM   = 8; // 歩の持ち駒の上限
	constexpr int MAX_HLANCE_NUM  = 4;
	constexpr int MAX_HKNIGHT_NUM = 4;
	constexpr int MAX_HSILVER_NUM = 4;
	constexpr int MAX_HGOLD_NUM   = 4;
	constexpr int MAX_HBISHOP_NUM = 2;
	constexpr int MAX_HROOK_NUM   = 2;

	// AperyのHandPiece enumの順が、やねうら王のPieceType順と異なるので、
	// このテーブルを参照するときには順番に注意。
	// ※　歩、香、桂、銀、金、角、飛の順。
	const int MAX_PIECES_IN_HAND[] =
	{
		MAX_HPAWN_NUM   , // PAWN
		MAX_HLANCE_NUM  , // LANCE
		MAX_HKNIGHT_NUM , // KNIGHT
		MAX_HSILVER_NUM , // SILVER
		MAX_HGOLD_NUM   , // GOLD
		MAX_HBISHOP_NUM , // BISHOP
		MAX_HROOK_NUM   , // ROOK
	};

	// 手駒の枚数の合計
	constexpr u32 MAX_PIECES_IN_HAND_SUM = MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;

	// 先後含めた手駒の枚数の合計
	constexpr u32 MAX_FEATURES2_HAND_NUM = (int)COLOR_NB * MAX_PIECES_IN_HAND_SUM;

	// 駒の種類(成り駒含む。先後の区別はない) : 空の駒は含まないので14種類。
	const int PIECETYPE_NUM = 14;

	// 駒の種類 : 空の駒を含むので15種類。Aperyで定義されている定数
	const int PieceTypeNum = PieceType::DRAGON + 1;

	// 手駒になりうる駒種の数。: Aperyで定義されている定数
	const int HandPieceNum = 7;

	// 入力特徴量の利きの数の上限。これ以上は、同じ数の利きとみなす。
	constexpr int MAX_ATTACK_NUM = 3; // 利き数の最大値

	// 盤上の駒に関する入力特徴量のチャンネルの数。(先手分に関して)
	// 駒の順はAperyのPieceTypeの順。これは、やねうら王と同じ。
	// ※　歩、香、桂、銀、角、飛、金。
	constexpr u32 MAX_FEATURES1_NUM = PIECETYPE_NUM/*駒の配置*/ + PIECETYPE_NUM/*駒種ごとの利き*/ + MAX_ATTACK_NUM/*利き数*/;

	// 手駒に関する入力特徴量のチャンネルの数。
	// 手駒は、AperyのHandPiece enumの順なので注意が必要。
	// ※　歩、香、桂、銀、金、角、飛の順。
	constexpr u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/;

	// 移動の定数
	// 成らない移動。10方向 + 成る移動 10方向。= 20方向
	enum MOVE_DIRECTION {
		UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
		UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
		MOVE_DIRECTION_NUM,
		MOVE_DIRECTION_NONE = -1 // 移動できないはずの組み合わせ
	};

	// 成る移動。10方向。
	const MOVE_DIRECTION MOVE_DIRECTION_PROMOTED[] = {
		UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
	};

	// 指し手を表すラベルの数
	// この数(27)×升の数(SQ_NB=81)だけPolicy Networkが値を出力する。
	// 駒の順はAperyのPieceTypeの順。これは、やねうら王と同じ。
	// ※　歩、香、桂、銀、角、飛、金。
	constexpr int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HandPieceNum;

	// 特徴量などに使う型。
	// 16bitが使えるのは、cuDNNのときだけだが、cuDNNの利用はdlshogiでは廃止する予定らしいので、
	// ここでは32bit floatとして扱う。
	typedef float DType;
	constexpr const DType dtype_zero = 0.0f; // DTypeで 0 を表現する型
	constexpr const DType dtype_one  = 1.0f; // DTypeで 1 を表現する型

	// NNの入力特徴量その1
	// ※　dlshogiでは、features1_tという型名。
	typedef DType NN_Input1[COLOR_NB][MAX_FEATURES1_NUM][SQ_NB];

	// NNの入力特徴量その2
	// ※　dlshogiでは、features2_tという型名。
	typedef DType NN_Input2[MAX_FEATURES2_NUM][SQ_NB];

	// NNの出力特徴量その1 (ValueNetwork) : 期待勝率
	typedef DType NN_Output_Value;

	// NNの出力特徴量その2 (PolicyNetwork) : それぞれの指し手の実現確率
	typedef DType NN_Output_Policy[MAX_MOVE_LABEL_NUM*SQ_NB];

	// 入力特徴量を生成する。
	//   position  : このあとEvalNode()を呼び出したい局面
	//   features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, NN_Input1* features1, NN_Input2* features2);

	// 指し手に対して、Policy Networkの返してくる配列のindexを返す。
	int make_move_label(Move move, Color color);

	// Softmax関数
	void softmax_temperature_with_normalize(std::vector<float>& log_probabilities);

	// Softmaxの時のボルツマン温度の設定。
	void set_softmax_temperature(const float temperature);

	// "isready"に対して設定されたNNのModel Pathを取得する。
	//std::vector<std::string> GetModelPath();

	// 評価値から価値(勝率)に変換
	// スケールパラメータは、elmo_for_learnの勝率から調査した値
	// 何かの変換の時に必要になる。
	float cp_to_value(const Value score);

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	Value value_to_cp(const float score);

	// エンジンオプションで設定されたモデルファイル名。(フォルダ名含む)
	// このsize() == max_gpuのはず。
	// "isready"で初期化されている。
	extern std::vector<std::string> ModelPaths;

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __DLSHOGI_EVALUATE_H_INCLUDED__

