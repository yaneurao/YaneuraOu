#include "ClusterStrategy.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//          SingleEngineStrategy 
	// ---------------------------------------

	// GUIから"go"コマンドが来たらエンジンに"go"コマンドを送って、
	// エンジン側から"bestmove"が返ってきたら、そのままGUIにそれを返す1エンジンからなる単純な戦略。

	// エンジンにconnect()した直後に呼び出される。
	void SingleEngineStrategy::on_connected(StrategyParam& param)
	{
		// engines[0]が存在することは保証されている。
		param.engines[0].set_engine_mode(EngineMode(
			// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
			  EngineMode::SEND_INFO_BEFORE_GAME_0
			// エンジンに"go"を送信したあと、エンジン側から流れてくる"info ..."を、そのままGUIに流す。
			| EngineMode::SEND_INFO_ON_GO
		));
	}

	// GUI側から"go"コマンドが来た時のhandler。
	void SingleEngineStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		// "go"コマンドが送られてきた時は、局面のsfen文字列は command.position_sfen に含まれているから、
		// これをこのままengine側に投げれば思考してくれるはず。
		param.engines[0].send(command);
	}

	// idleな時に呼び出される。(通常、1秒間に100回以上呼び出される)
	// エンジン側から"bestmove"が返ってきていたらGUIにそれを投げる、などの処理はここで行う。
	void SingleEngineStrategy::on_idle(StrategyParam& param)
	{
		// bestmoveが来ているかを確認する。
		// bestmoveが来ていれば、GUI側にそれをそのまま投げる。

		auto bestmove = param.engines[0].pull_bestmove();
		if (bestmove.empty())
			return ; // 来てない。

		// これこのままGUI側に送信すればOK。
		send_to_gui(bestmove);
	}

	// 通常のエンジンを模倣するClusterは、上記のように3つの関数をoverrideするだけで書ける。
	// しかもengineへ思考を委譲するのは、on_go_commandにあるようにそのままエンジンに対してsend()するだけで良い。
	// これだけ簡潔にClusterのアルゴリズムを定義できるということが、この設計の正しさを物語っている。

}

#endif
