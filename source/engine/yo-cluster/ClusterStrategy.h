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
		virtual void on_connected(StrategyParam& param) {}
		virtual void on_go_command(StrategyParam& param, const Message& command){}
		virtual void on_idle(StrategyParam& param){}
	};

	// MultiPonder
	class MultiPonderStrategy : public IClusterStrategy
	{
	public:
		virtual void on_go_command(StrategyParam& param, const Message& command){}
		virtual void on_idle(StrategyParam& param){}
	};

	// 楽観合議
	class OptimisticConsultationStrategy : public IClusterStrategy
	{
	public:
		virtual void on_go_command(StrategyParam& param, const Message& command){}
		virtual void on_idle(StrategyParam& param){}
	};

}


#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif // ifndef CLUSTER_STRATEGY_H_INCLUDED
