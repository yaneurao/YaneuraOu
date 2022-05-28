#include "ClusterStrategy.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

using namespace std;

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
		// (終了したエンジンはengines配列から自動的に取り除かれる＆engines.size()==0だとこのプログラム自体が終了する)

		param.engines[0].set_engine_mode(EngineMode(
			// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
			  EngineMode::SEND_INFO_BEFORE_GAME
			// エンジンに"go"を送信したあと、エンジン側から流れてくる"info ..."を、そのままGUIに流す。
			| EngineMode::SEND_INFO_ON_GO
		));
	}

	// GUI側から"go"コマンドが来た時のhandler。
	void SingleEngineStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		// "go"コマンドが送られてきた時は、直前の"position"コマンドは command.position_cmd に格納されているから、
		// これをこのままengine側に投げれば思考してくれる。
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

		// これこのままGUI側に送信すればGUIに対して"bestmove XX"を返したことになる。
		send_to_gui(bestmove);
	}

	// 通常のエンジンを模倣するClusterは、上記のように3つの関数をoverrideするだけで書ける。
	// しかもengineへ思考を委譲するのは、on_go_commandにあるようにそのままエンジンに対してsend()するだけで良い。
	// これだけ簡潔にClusterのアルゴリズムを定義できるということが、この設計の正しさを物語っている。

	// ---------------------------------------
	//       SinglePonderEngineStrategy 
	// ---------------------------------------

	// エンジン側が"bestmove XX ponder YY"のYYで指定してきた局面をエンジン側に事前に"go ponder"で送っておき
	// GUIから"go"コマンドが来たとき、その局面であったならエンジン側に"ponderhit"を送信、
	// さもなくばその局面を"go"するコマンドをエンジン側に送信。
	// エンジン側から"bestmove"が返ってきたらGUIにそれを投げ、さらに…(最初の説明に戻る)

	// エンジンにconnect()した直後に呼び出される。
	void SinglePonderEngineStrategy::on_connected(StrategyParam& param)
	{
		// engines[0]が存在することは保証されている。
		// (終了したエンジンはengines配列から自動的に取り除かれる＆engines.size()==0だとこのプログラム自体が終了する)

		param.engines[0].set_engine_mode(EngineMode(
			// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
			  EngineMode::SEND_INFO_BEFORE_GAME
			// エンジンに"go"を送信したあと、エンジン側から流れてくる"info ..."を、そのままGUIに流す。
			| EngineMode::SEND_INFO_ON_GO
		));
	}

	// GUI側から"go"コマンドが来た時のhandler。
	void SinglePonderEngineStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		auto& engine = param.engines[0];

		// goコマンドの対象局面
		auto sfen = command.position_sfen;

		// いま go ponderで思考している局面と、command.position_sfenが一致するなら、エンジンに"ponderhit"を送ってやれば良い。
		if (engine.is_state_go_ponder() && engine.get_searching_sfen() == sfen)
			// 前回の"go"の時に渡された残り時間等のパラメーターをそのままに、"ponderhit XXX .."の形式でエンジン側に送信。
			engine.send(Message(USI_Message::PONDERHIT, last_go_param));
		else
			// この局面についてponderしていなかったので愚直に"go"コマンドで思考させる。
			engine.send(command);
	}

	// idleな時に呼び出される。(通常、1秒間に100回以上呼び出される)
	// エンジン側から"bestmove"が返ってきていたらGUIにそれを投げる、などの処理はここで行う。
	void SinglePonderEngineStrategy::on_idle(StrategyParam& param)
	{
		// bestmoveが来ているかを確認する。
		// bestmoveが来ていれば、GUI側にそれをそのまま投げる。

		auto& engine = param.engines[0];

		auto bestmove = engine.pull_bestmove();
		if (bestmove.empty())
			return ; // 来てない。

		// これこのままGUI側に送信すればGUIに対して"bestmove XX"を返したことになる。
		send_to_gui(bestmove);

		// "bestmove XX ponder YY"の形だと思うので、YYを抽出して、その局面について思考エンジンにGO_PONDERさせる。

		// 探索していた局面は、bestmoveを返したあとも次の"GO","GO_PONDER"が来るまではget_searching_sfen()で取得できることが保証されている。
		// そこに、XXとYYを追加したsfen文字列を用意して、GO_PONDERする。
		auto sfen = concat_bestmove( engine.get_searching_sfen() , bestmove);

		// XXかYYが"resign"のような、それによって局面を進められない指し手である場合、
		// concat_bestmove()の戻り値は 空の文字列になることが保証されている。
		// この場合、GO_PONDERしてはならない。
		if (sfen.empty())
			return ;

		engine.send(Message(USI_Message::GO_PONDER, string() , sfen));
	}

}

#endif
