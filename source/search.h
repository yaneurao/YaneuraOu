#ifndef _SEARCH_H_
#define _SEARCH_H_

#include "misc.h"
#include "movepick.h"
#include "position.h"

// 探索関係
namespace Search {

	// countermoves based pruningで使う閾値
	constexpr int CounterMovePruneThreshold = 0;

	// -----------------------
	//  探索のときに使うStack
	// -----------------------

	struct Stack {
		Move* pv;					// PVへのポインター。RootMovesのvector<Move> pvを指している。
		PieceToHistory* continuationHistory;// historyのうち、counter moveに関するhistoryへのポインタ。実体はThreadが持っている。
		int ply;					// rootからの手数。rootならば0。
		Move currentMove;			// そのスレッドの探索においてこの局面で現在選択されている指し手
		Move excludedMove;			// singular extension判定のときに置換表の指し手をそのnodeで除外して探索したいのでその除外する指し手
		Move killers[2];			// killer move
		Value staticEval;			// 評価関数を呼び出して得た値。NULL MOVEのときに親nodeでの評価値が欲しいので保存しておく。
		int statScore;				// 一度計算したhistoryの合計値をcacheしておくのに用いる。
		int moveCount;				// このnodeでdo_move()した生成した何手目の指し手か。(1ならおそらく置換表の指し手だろう)
	};

	// root(探索開始局面)での指し手として使われる。それぞれのroot moveに対して、
	// その指し手で進めたときのscore(評価値)とPVを持っている。(PVはfail lowしたときには信用できない)
	// scoreはnon-pvの指し手では-VALUE_INFINITEで初期化される。
	struct RootMove
	{
		// pv[0]には、このコンストラクタの引数で渡されたmを設定する。
		explicit RootMove(Move m) : pv(1, m) {}

		// ponderの指し手がないときにponderの指し手を置換表からひねり出す。pv[1]に格納する。
		// ponder_candidateが2手目の局面で合法手なら、それをpv[1]に格納する。
		// それすらなかった場合はfalseを返す。
		// ※　Stockfishにはこの関数に第二引数はない。やねうら王が独自に追加した。
		bool extract_ponder_from_tt(Position& pos, Move ponder_candidate);

		// std::count(),std::find()などで指し手と比較するときに必要。
		bool operator==(const Move& m) const { return pv[0] == m; }

		// sortするときに必要。std::stable_sort()で降順になって欲しいので比較の不等号を逆にしておく。
		// 同じ値のときは、previousScoreも調べる。
		bool operator<(const RootMove& m) const {
			return m.score != score ? m.score < score
								    : m.previousScore < previousScore;
		}

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
	// "ponder"のフラグはここに含まれず、Threads.ponderにあるので注意。
	struct LimitsType {

		// PODでない型をmemsetでゼロクリアすると破壊してしまうので明示的に初期化する。
		LimitsType() {
			time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
			depth = mate = perft = infinite = 0;
			nodes = 0;

			// やねうら王で、将棋用に追加したメンバーの初期化。

			byoyomi[WHITE] = byoyomi[BLACK] = TimePoint(0);
			max_game_ply = 100000;
			rtime = 0;
			enteringKingRule = EKR_NONE;
			silent = bench = consideration_mode = outout_fail_lh_pv = false;
			pv_interval = 0;

#if !defined(FOR_TOURNAMENT)
			generate_all_legal_moves = false;
#endif
		}

		// 時間制御を行うのか。
		// 詰み専用探索、思考時間0、探索深さが指定されている、探索ノードが指定されている、思考時間無制限
		// であるときは、時間制御に意味がないのでやらない。
		bool use_time_management() const {
			return !(mate | movetime | depth | nodes | perft | infinite);
		}

		// root(探索開始局面)で、探索する指し手集合。特定の指し手を除外したいときにここから省く
		std::vector<Move> searchmoves;

		// time[]   : 残り時間(ms換算で)
		// inc[]    : 1手ごとに増加する時間(フィッシャールール)
		// npmsec   : 探索node数を思考経過時間の代わりに用いるモードであるかのフラグ(from UCI)
		// movetime : 思考時間固定(0以外が指定してあるなら) : 単位は[ms]
		TimePoint time[COLOR_NB] , inc[COLOR_NB] , npmsec , movetime;

		// depth    : 探索深さ固定(0以外を指定してあるなら)
		// mate     : 詰み専用探索(USIの'go mate'コマンドを使ったとき)
		//		詰み探索モードのときは、ここに詰みの手数が指定されている。
		//		その手数以内の詰みが見つかったら探索を終了する。
		//		※　Stockfishの場合、この変数は先後分として将棋の場合の半分の手数が格納されているので注意。
		//		USIプロトコルでは、この値に詰将棋探索に使う時間[ms]を指定することになっている。
		//		時間制限なしであれば、INT32_MAXが入っている。
		// perft    : perft(performance test)中であるかのフラグ。非0なら、perft時の深さが入る。
		// infinite : 思考時間無制限かどうかのフラグ。非0なら無制限。
		int depth , mate, perft, infinite;

		// 今回のgoコマンドでの探索ノード数
		int64_t nodes;

		// -- やねうら王が将棋用に追加したメンバー

		// 秒読み(ms換算で)
		TimePoint byoyomi[COLOR_NB];

		// この手数で引き分けとなる。256なら256手目を指したあとに引き分け。
		// USIのoption["MaxMovesToDraw"]の値。引き分けなしなら100000。
		// (残り手数を計算する時に桁あふれすると良くないのでINT_MAXにはしていない)
		int max_game_ply;

		// "go rtime 100"とすると100～300msぐらい考える。
		TimePoint rtime;

		// 入玉ルール設定
		EnteringKingRule enteringKingRule;

		// 画面に出力しないサイレントモード(プロセス内での連続自己対戦のとき用)
		// このときPVを出力しない。
		bool silent;

		// 検討モード用のPVを出力するのか
		bool consideration_mode;

		// fail low/highのときのPVを出力するのか
		bool outout_fail_lh_pv;

		// ベンチマークモード(このときPVの出力時に置換表にアクセスしない)
		bool bench;

		// PVの出力間隔(探索のときにMainThread::search()内で初期化する)
		TimePoint pv_interval;

#if !defined(FOR_TOURNAMENT)
		// 全合法手を生成するのか
		bool generate_all_legal_moves;
#endif
	};

	extern LimitsType Limits;

	// 探索部の初期化。
	void init();

	// 探索部のclear。
	// 置換表のクリアなど時間のかかる探索の初期化処理をここでやる。isreadyに対して呼び出される。
	void clear();

} // end of namespace Search

#endif // _SEARCH_H_
