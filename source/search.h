#ifndef _SEARCH_H_
#define _SEARCH_H_

#include <atomic>

#include "position.h"
#include "move_picker.h"
#include "misc.h"

// 探索関係
namespace Search {

	// root(探索開始局面)での指し手として使われる。それぞれのroot moveに対して、
	// その指し手で進めたときのscore(評価値)とPVを持っている。(PVはfail lowしたときには信用できない)
	// scoreはnon-pvの指し手では-VALUE_INFINITEで初期化される。
	struct RootMove
	{
		// sortするときに必要。std::stable_sort()で降順になって欲しいので比較の不等号を逆にしておく。
		// 同じ値のときは、previousScoreも調べる。
		bool operator<(const RootMove& m) const {
			return m.score != score ? m.score < score : m.previousScore < previousScore;
		}

		// std::count(),std::find()などで指し手と比較するときに必要。
		bool operator==(const Move& m) const { return pv[0] == m; }

		// pv[0]には、このコンストラクタの引数で渡されたmを設定する。
		explicit RootMove(Move m) : pv(1, m) {}

		// ponderの指し手がないときにponderの指し手を置換表からひねり出す。pv[1]に格納する。
		// ponder_candidateが2手目の局面で合法手なら、それをpv[1]に格納する。
		// それすらなかった場合はfalseを返す。
		bool extract_ponder_from_tt(Position& pos, Move ponder_candidate);

		// ConsiderationModeのときにTTからPVをかき集めるのでTTのPVが破壊されていると困るから
		// PV配列をTTに書き戻すことでそれをなるべく防ぐ。
		// 旧Stockfishにあった機能。Stockfish7あたりでなくなった。
		// tt_gen : TT.generation()に相当するもの。
		void insert_pv_to_tt(Position& pos , u8 tt_gen);

		// 今回の(反復深化の)iterationでの探索結果のスコア
		Value score = -VALUE_INFINITE;

		// 前回の(反復深化の)iterationでの探索結果のスコア
		// 次のiteration時の探索窓の範囲を決めるときに使う。
		Value previousScore = -VALUE_INFINITE;

		// このスレッドがrootから最大、何手目まで探索したか(選択深さの最大)
		int selDepth = 0;

		// この指し手で進めたときのpv
		std::vector<Move> pv;
	};

	typedef std::vector<RootMove> RootMoves;

	// goコマンドでの探索時に用いる、持ち時間設定などが入った構造体
	struct LimitsType {

		// PODでない型をmemsetでゼロクリアするとMSVCは破壊してしまうので明示的に初期化する。
		LimitsType() {
			nodes = time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = byoyomi[WHITE] = byoyomi[BLACK] = npmsec
				= depth = movetime = mate = infinite = ponder = rtime = 0;
			silent = bench = ponder_mode = consideration_mode = outout_fail_lh_pv = false;
			max_game_ply = 100000;
			enteringKingRule = EKR_NONE;
		}

		// 時間制御を行うのか。
		// 詰み専用探索、思考時間0、探索深さが指定されている、探索ノードが指定されている、思考時間無制限
		// であるときは、時間制御に意味がないのでやらない。
		bool use_time_management() const {
			return !(mate | movetime | depth | nodes | infinite);
		}

		// root(探索開始局面)で、探索する指し手集合。特定の指し手を除外したいときにここから省く
		std::vector<Move> searchmoves;

		// 残り時間(ms換算で)
		int time[COLOR_NB];

		// 秒読み(ms換算で)
		int byoyomi[COLOR_NB];

		// 1手ごとに増加する時間(フィッシャールール)
		int inc[COLOR_NB];

		// 探索node数を思考経過時間の代わりに用いるモードであるかのフラグ(from UCI)
		int npmsec;

		// この手数で引き分けとなる。256なら256手目を指したあとに引き分け。
		// USIのoption["MaxMovesToDraw"]の値。引き分けなしならINT_MAX。
		int max_game_ply;

		// depth    : 探索深さ固定(0以外を指定してあるなら)
		int depth;

		// movetime : 思考時間固定(0以外が指定してあるなら) : 単位は[ms]
		int movetime;

		// mate     : 詰み専用探索(USIの'go mate'コマンドを使ったとき)
		//  詰み探索モードのときは、ここに思考時間が指定されている。
		//  この思考時間いっぱいまで考えて良い。
		int mate;

		// infinite : 思考時間無制限かどうかのフラグ。非0なら無制限。
		int infinite;

		// ponder   : ponder中であるかのフラグ
		//  これがtrueのときはbestmoveがあっても探索を停止せずに"ponderhit"か"stop"が送られてきてから停止する。
		// main threadからしか参照しないのでatomicはつけてない。
		bool ponder;

		// "go rtime 100"とすると100～300msぐらい考える。
		int rtime;

		// 今回のgoコマンドでの探索ノード数
		uint64_t nodes;

		// 入玉ルール設定
		EnteringKingRule enteringKingRule;

		// 画面に出力しないサイレントモード(プロセス内での連続自己対戦のとき用)
		// このときPVを出力しない。
		bool silent;

		// 試合開始後、ponderが一度でも送られてきたか
		bool ponder_mode;

		// 検討モード用のPVを出力するのか
		bool consideration_mode;

		// fail low/highのときのPVを出力するのか
		bool outout_fail_lh_pv;

		// ベンチマークモード(このときPVの出力時に置換表にアクセスしない)
		bool bench;
	};

	typedef std::unique_ptr<aligned_stack<StateInfo>> StateStackPtr;

	extern LimitsType Limits;
	extern StateStackPtr SetupStates;

	// 探索部の初期化。
	void init();

	// 探索部のclear。
	// 置換表のクリアなど時間のかかる探索の初期化処理をここでやる。isreadyに対して呼び出される。
	void clear();

	// -----------------------
	//  探索のときに使うStack
	// -----------------------

	struct Stack {
		Move* pv;				// PVへのポインター。RootMovesのvector<Move> pvを指している。
		int ply;				// rootからの手数
		Move currentMove;		// そのスレッドの探索においてこの局面で現在選択されている指し手
		Move excludedMove;		// singular extension判定のときに置換表の指し手をそのnodeで除外して探索したいのでその除外する指し手
		Move killers[2];		// killer move
		Value staticEval;		// 評価関数を呼び出して得た値。NULL MOVEのときに親nodeでの評価値が欲しいので保存しておく。

#if defined (PER_STACK_HISTORY)
		int statScore;			// 一度計算したhistoryの合計値をcacheしておくのに用いる。
#endif

		int moveCount;          // このnodeでdo_move()した生成した何手目の指し手か。(1ならおそらく置換表の指し手だろう)

#if defined(USE_MOVE_PICKER_2017Q2)
		PieceToHistory* history;		// history絡み、refactoringにより名前が変わった。
#endif
	};

} // end of namespace Search

#endif // _SEARCH_H_