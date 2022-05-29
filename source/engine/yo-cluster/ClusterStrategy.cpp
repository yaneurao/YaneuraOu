#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "../../types.h"
#include "../../misc.h"

#include "ClusterStrategy.h"
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
			// 今回の"go"の時に渡された残り時間等のパラメーターをそのままに、"ponderhit XXX .."の形式でエンジン側に送信。
			engine.send(Message(USI_Message::PONDERHIT, strip_command(command.command) ));
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

		auto bestmove_str = engine.pull_bestmove();
		if (bestmove_str.empty())
			return ; // 来てない。

		// これこのままGUI側に送信すればGUIに対して"bestmove XX"を返したことになる。
		send_to_gui(bestmove_str);

		// "bestmove XX ponder YY"の形だと思うので、YYを抽出して、その局面について思考エンジンにGO_PONDERさせる。

		// 探索していた局面は、bestmoveを返したあとも次の"GO","GO_PONDER"が来るまではget_searching_sfen()で取得できることが保証されている。
		// そこに、XXとYYを追加したsfen文字列を用意して、GO_PONDERする。

		string bestmove, ponder;
		parse_bestmove(bestmove_str, bestmove , ponder);

		// XXかYYが"resign"のような、それによって局面を進められない指し手である場合(このとき、空の文字列となる)、
		// GO_PONDERしてはならない。
		if (bestmove.empty() || ponder.empty())
			return ;

		auto sfen = concat_sfen(engine.get_searching_sfen(), bestmove + " " + ponder);
		engine.send(Message(USI_Message::GO_PONDER, string() , sfen));
	}

	// ---------------------------------------
	//     OptimisticConsultationStrategy 
	// ---------------------------------------

	// SinglePonderStrategyを複数エンジンに対応させて、
	// goした時に一番良い評価値を返してきたエンジンのbestmoveを採用するように変えたもの。

	void OptimisticConsultationStrategy::on_connected(StrategyParam& param)
	{
		for(auto& engine : param.engines)
			engine.set_engine_mode(EngineMode(
				// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
				EngineMode::SEND_INFO_BEFORE_GAME
			));
	}

	void OptimisticConsultationStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		auto& engines = param.engines;

		// goコマンドの対象局面
		auto  sfen    = command.position_sfen;

		for(auto& engine : engines)
		{
			// いま go ponderで思考している局面と、command.position_sfenが一致するなら、エンジンに"ponderhit"を送ってやれば良い。
			if (engine.is_state_go_ponder() && engine.get_searching_sfen() == sfen)
				// 今回の"go"の時に渡された残り時間等のパラメーターをそのままに、"ponderhit XXX .."の形式でエンジン側に送信。
				engine.send(Message(USI_Message::PONDERHIT, strip_command(command.command) ));
			else
				// この局面についてponderしていなかったので愚直に"go"コマンドで思考させる。
				engine.send(command);
		}
	}

	void OptimisticConsultationStrategy::on_idle(StrategyParam& param)
	{
		// すべてのbestmoveが来てから。
		auto& engines = param.engines;

		for(auto& engine : engines)
			if (engine.peek_bestmove().empty())
				return ; // 来てない。

		// 一番良い評価値を返してきているエンジンを探す。

		size_t best_engine = size_max;
		int    best_value  = int_min;
		vector<string> best_log;
		for(size_t i = 0 ; i < engines.size(); ++i)
		{
			auto& engine = engines[i];

			auto log = engine.pull_thinklog();
			// 末尾からvalueの書いてあるlogを探す。
			for(size_t j = log.size() ; j != 0 ; j --)
			{
				UsiInfo info;
				parse_usi_info(log[j-1], info);

				if (info.value != VALUE_NONE)
				{
					if (info.value > best_value)
					{
						best_value  = info.value;
						best_engine = i;
						best_log    = log;
					}
					// valueが書いてあったのでこのエンジンに関して
					// ログを調べるのはこれで終わり。
					break;
				}
			}
		}

		// 思考ログが存在しない。そんな馬鹿な…。
		if (best_engine == size_max)
		{
			// こんなことをしてくるエンジンがいると楽観合議できない。
			// 必ずbestmoveの手前で読み筋と評価値を送ってくれないと駄目。
			error_to_gui("OptimisticConsultationStrategy::on_idle , No think_log");
			Tools::exit();
		}

		// ここまでの思考logをまとめてGUIに送信する。
		for(auto& line : best_log)
			send_to_gui(line);

		auto bestmove_str = engines[best_engine].peek_bestmove();
		// engineすべてからbestmoveを取り除いておく。
		for(auto& engine : engines)
			engine.pull_bestmove();

		 // これこのままGUI側に送信すればGUIに対して"bestmove XX"を返したことになる。
		send_to_gui(bestmove_str);

		// "bestmove XX ponder YY"の形だと思うので、YYを抽出して、その局面について思考エンジンにGO_PONDERさせる。

		// 探索していた局面は、bestmoveを返したあとも次の"GO","GO_PONDER"が来るまではget_searching_sfen()で取得できることが保証されている。
		// そこに、XXとYYを追加したsfen文字列を用意して、GO_PONDERする。

		string bestmove, ponder;
		parse_bestmove(bestmove_str, bestmove , ponder);

		// XXかYYが"resign"のような、それによって局面を進められない指し手である場合(このとき、空の文字列となる)、
		// GO_PONDERしてはならない。
		if (bestmove.empty() || ponder.empty())
			return ;

		auto sfen = concat_sfen(engines[0].get_searching_sfen(), bestmove + " " + ponder);

		// すべてのengineのponderを送って、ベストを拾う。
		for(auto& engine : param.engines)
			engine.send(Message(USI_Message::GO_PONDER, string() , sfen));
	}

}

#endif
