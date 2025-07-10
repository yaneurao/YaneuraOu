#ifndef CLUSTER_STRATEGY_H_INCLUDED
#define CLUSTER_STRATEGY_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ClusterObserver.h"

namespace YaneuraouTheCluster
{
	// GUIから"go"コマンドが来たらエンジンに"go"コマンドを送って、
	// エンジン側から"bestmove"が返ってきたら、そのままGUIにそれを返す1エンジンからなる単純な戦略。
	class SingleEngineStrategy : public IClusterStrategy
	{
	public:
		// エンジンにconnect()した直後に呼び出される。
		virtual void on_connected(StrategyParam& param);

		// GUI側から"go"コマンドが来た時のhandler。
		// command : GUI側から来たコマンド詳細が格納されている。
		virtual void on_go_command(StrategyParam& param, const Message& command);

		// idleな時に呼び出される。(通常、1秒間に100回以上呼び出される)
		// エンジン側から"bestmove"が返ってきていたらGUIにそれを投げる、などの処理はここで行う。
		virtual void on_idle(StrategyParam& param);
	};

	// エンジン側が"bestmove XX ponder YY"のYYで指定してきた局面をエンジン側に事前に"go ponder"で送っておき
	// GUIから"go"コマンドが来たとき、その局面であったならエンジン側に"ponderhit"を送信、
	// さもなくばその局面を"go"するコマンドをエンジン側に送信。
	// エンジン側から"bestmove"が返ってきたらGUIにそれを投げ、さらに…(最初の説明に戻る)
	class SinglePonderEngineStrategy : public IClusterStrategy
	{
	public:
		virtual void on_connected(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);
	};

	// 楽観合議
	// 
	// SinglePonderStrategyを複数エンジンに対応させて、
	// goした時に一番良い評価値を返してきたエンジンのbestmoveを採用するように変えたもの。
	// →　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
	class OptimisticConsultationStrategy : public IClusterStrategy
	{
	public:
		virtual void on_connected(StrategyParam& param);
		virtual void on_isready(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);

	protected:
		// "stop"をエンジンに対して送信したか。
		bool stop_sent = false;
	};

	// MultiPonder
	//
	// Ponderする時に相手の予想手を複数用意する。
	class MultiPonderStrategy : public IClusterStrategy
	{
	public:
		virtual void on_connected(StrategyParam& param);
		virtual void on_isready(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);
	protected:
		// 余っているエンジンに対して、思考対象局面を複数用意してponderする。
		// root_sfen         : 現在の基準局面
		// available_engines : ponderに使うエンジンの番号。(engines[available_engines[i]]を用いる)
		// same_color        : ponder対象とする局面がroot_sfenの局面と同じ手番なのか。(trueなら2手先、falseなら1手先)
		// except_move       : ponder対象から除外する指し手
		void search_and_ponder(std::vector<EngineNegotiator>& engines,
			const std::string& root_sfen, std::vector<size_t> available_engines, bool same_color, std::string except_move);

		// "stop"をエンジンに対して送信したか。
		bool stop_sent = false;

		// 現在思考中のsfen
		std::string searching_sfen;

	};

	// RootSplit
	// 
	// Rootの指し手をエンジンごとに分割して思考する。
	//	→　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
	//		さらに、workerは goコマンドの"wait_stop"機能に対応している必要がある。(やねうら王NNUEは対応している)
	class RootSplitStrategy : public IClusterStrategy
	{
	public:
		virtual void on_connected(StrategyParam& param);
		virtual void on_isready(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);

	protected:
		// sfenを与えて、その局面の合法手を生成して、それをエンジンの数で分割したものを返す。
		std::vector<std::string> make_search_moves(const std::string& sfen , size_t engine_num);

		// "stop"をエンジンに対して送信したか。
		bool stop_sent = false;

		// 現在、GOなのかGO_PONDERなのか、そうでないのか。
		EngineState state = EngineState::DISCONNECTED;
	};

	// GPS将棋のクラスター手法
	//
	// これは、次の論文にある手法である。rootでMultiPV 2で探索して、上位2手に1台ずつ割り当て、その他の指し手に1台割り当てる。
	// この時、計3台である。もっと台数が多い場合は、この上位2手を展開して、その子において同じように3台ずつ割り当てる。
	// この時、3 + 3 + 1 = 7台である。同様に台数が増えた時は、上位2手を展開していく。
	// 
	// 最善手の予測に基づくゲーム木探索の分散並列実行
	// https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=71329&item_no=1&page_id=13&block_id=8
	// 
	class GpsClusterStrategy : public IClusterStrategy
	{
	public:
		GpsClusterStrategy(size_t n) : split_const(n) {}
		virtual void on_connected(StrategyParam& param);
		virtual void on_isready(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);

	protected:
		// sfenを与えて、その局面の合法手を生成して、それをエンジンの数で分割したものを返す。
		std::vector<std::string> make_search_moves(const std::string& sfen , size_t engine_num);

		// 1局面を何分割するのか
		size_t split_const;

		// splitする時の
		u64 nodes_limit = 10000;

		// "stop"をエンジンに対して送信したか。
		bool stop_sent = false;
	};

	// GPS将棋のクラスター手法　その2
	//  1局面をsplit_const(==N)に分割する。
	//  すなわち、上位(N - 1)の指し手を展開し、それ以外の指し手を1台workerを割り当てる。
	//  これを繰り返す。N = 3がdefaultだが、ここを可変にできる。
	//  このNはコンストラクタで渡す。
	class GpsClusterStrategy2 : public IClusterStrategy
	{
	public:
		GpsClusterStrategy2(size_t n) : split_const(n) {}
		virtual void on_connected(StrategyParam& param);
		virtual void on_isready(StrategyParam& param);
		virtual void on_go_command(StrategyParam& param, const Message& command);
		virtual void on_idle(StrategyParam& param);

	protected:
		// sfenを与えて、その局面の合法手を生成して、それをエンジンの数で分割したものを返す。
		std::vector<std::string> make_search_moves(const std::string& sfen , size_t engine_num);

		// 1局面を何分割するのか
		size_t split_const;

		// splitする時の
		u64 nodes_limit = 10000;

		// "stop"をエンジンに対して送信したか。
		bool stop_sent = false;
	};
}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif // ifndef CLUSTER_STRATEGY_H_INCLUDED
