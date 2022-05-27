#ifndef CLUSTER_OBSERVER_H_INCLUDED
#define CLUSTER_OBSERVER_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include <vector>

#include "ClusterCommon.h"
#include "EngineNegotiator.h"

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//        ClusterStrategy interface
	// ---------------------------------------

	// IClusterStrategyで用いるパラメーター。
	class StrategyParam
	{
	public:
		StrategyParam(std::vector<EngineNegotiator>& engines_ , ClusterOptions& options_)
			:engines(engines_) , options(options_)
		{}

		// 子プロセスで起動している思考エンジンの集合。
		std::vector<EngineNegotiator>& engines;

		ClusterOptions& options;
	};

	// Cluster化する時の戦略を記述するためのclass用のinterface。
	// MultiPonderとか楽観合議とかは、このinterfaceを持つclassを作ってそこに実装する。
	// ユーザーが自分の新しい戦略を作って、ClusterObserverにセットすることができる。
	// ※　実際の使用例についてはClusterStrategy.hを見ること。
	class IClusterStrategy
	{
	public:
		// エンジンにconnect()した直後に呼び出される。
		virtual void on_connected(StrategyParam& param) {}

		// GUI側から"go"コマンドが来た時のhandler。
		// command : GUI側から来たコマンド詳細が格納されている。
		virtual void on_go_command(StrategyParam& param, const Message& command) {}

		// idleな時に呼び出される。(通常、1秒間に100回以上呼び出される)
		// エンジン側から"bestmove"が返ってきていたらGUIにそれを投げる、などの処理はここで行う。
		virtual void on_idle(StrategyParam& param) {}

		virtual ~IClusterStrategy(){}
	};
}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif // ndef CLUSTER_OBSERVER_H_INCLUDED
