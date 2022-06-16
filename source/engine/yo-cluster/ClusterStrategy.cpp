#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "../../types.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../book/book.h"

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

	void OptimisticConsultationStrategy::on_isready(StrategyParam& param)
	{
		stop_sent = false;
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

		stop_sent = false;
	}

	void OptimisticConsultationStrategy::on_idle(StrategyParam& param)
	{
		// すべてのbestmoveが来てから。
		// ただし、一番最初にbestmoveを返してきたengineを基準とする。
		auto& engines = param.engines;

		// bestmoveを返したエンジンの数
		int bestmove_received = 0;
		for(auto& engine : engines)
			if (!engine.peek_bestmove().empty())
				++bestmove_received;

		// まだすべてのエンジンがbestmoveを返していない。
		if (bestmove_received < engines.size())
		{
			if (bestmove_received > 0)
			{
				// 少なくとも1つのエンジンはbestmoveを返したが、
				// まだ全部のエンジンからbestmoveきてない。

				// stopを送信していないならすべてのengineに"stop"を送信してやる。
				if (!stop_sent)
				{
					for(auto& engine : engines)
						engine.send(USI_Message::STOP);

					stop_sent = true;
				}
			}
			return ;
		}

		// 一番良い評価値を返してきているエンジンを探す。

		size_t best_engine = size_max;
		int    best_value  = int_min;
		vector<string>* best_log = nullptr;
		for(size_t i = 0 ; i < engines.size(); ++i)
		{
			auto& engine = engines[i];

			auto& log = *engine.peek_thinklog();
			// 末尾からvalueの書いてあるlogを探す。
			for(size_t j = log.size() ; j != 0 ; j --)
			{
				UsiInfo info;
				parse_usi_info(log[j-1], info);

				if (info.value != VALUE_NONE )
				{
#if 0
					// 詰みに関するスコアだけ合議する場合。
					if (i == 0 ||

						(info.value > best_value
						&& ( abs(info.value) >= VALUE_MATE_IN_MAX_PLY || abs(best_value) >= VALUE_MATE_IN_MAX_PLY)
							)
						)
#endif
					if (info.value > best_value)
					{
						best_value  = info.value;
						best_engine = i;
						best_log    = &log;
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

		// ここまでの思考ログをまとめてGUIに送信する。
		if (best_log != nullptr)
			for(auto& line : *best_log)
				send_to_gui(line);

		// 思考ログのクリア
		for(auto& engine : engines)
			engine.clear_thinklog();

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

	// ---------------------------------------
	//     MultiPonderStrategy 
	// ---------------------------------------

	// Ponderする時に相手の予想手を複数用意する。

	void MultiPonderStrategy::on_connected(StrategyParam& param)
	{
		for(auto& engine : param.engines)
			engine.set_engine_mode(EngineMode(
				// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
				EngineMode::SEND_INFO_BEFORE_GAME
			));
	}

	void MultiPonderStrategy::on_isready(StrategyParam& param)
	{
		stop_sent = false;
		searching_sfen.clear();
	}

	void MultiPonderStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		auto& engines = param.engines;

		// goコマンドの対象局面
		auto  sfen    = command.position_sfen;
		// 現在思考中のsfen
		this->searching_sfen = sfen;

		// ponderhitしたエンジンの数をカウントする。
		// いま go ponderで思考している局面と、command.position_sfenが一致するなら、エンジンに"ponderhit"を送ってやれば良い。
		// MultiPonderでも予想手の合法手が少ない時は、複数のエンジンに同じponderの指し手を割り当てるのでこれは複数ある可能性が。

		vector<size_t> ponderhit_engines;
		vector<size_t> not_ponderhit_engines;

		for(size_t i = 0 ; i < engines.size() ; ++i)
		{
			auto& engine = engines[i];
			if (engine.is_state_go_ponder() && engine.get_searching_sfen() == sfen)
				ponderhit_engines.push_back(i);
			else
				not_ponderhit_engines.push_back(i);
		}

		// ponderhitしたエンジンに対して、PONDERHITを送信する。
		if (ponderhit_engines.size())
		{
			for(size_t i : ponderhit_engines)
				engines[i].send(Message(USI_Message::PONDERHIT, strip_command(command.command) ));
		}
		else
			// ponderhitしたエンジンが一つもないのですべてのエンジンで思考する。(合議)
		{
			for(auto& engine : engines)
				engine    .send(Message(USI_Message::GO       , command.command  , sfen ));
			not_ponderhit_engines.clear();
		}

		// 空いているエンジンがあるのか？
		if (not_ponderhit_engines.size())
		{
			// エンジンが余っているので、これについて2手先の局面のなかから適当に思考対象局面を用意してponderする。
			search_and_ponder(engines, sfen, not_ponderhit_engines, true /* same_color */ , string());
		}

		stop_sent = false;
	}

	// 余っているエンジンに対して、思考対象局面を複数用意してponderする。
	// root_sfen         : 現在の基準局面
	// available_engines : ponderに使うエンジンの番号。(engines[available_engines[i]]を用いる)
	// same_color        : ponder対象とする局面がroot_sfenの局面と同じ手番なのか。(trueなら2手先、falseなら1手先)
	// except_move       : ponder対象から除外する指し手
	void MultiPonderStrategy::search_and_ponder(std::vector<EngineNegotiator>& engines,
		const std::string& root_sfen, std::vector<size_t> available_engines, bool same_color , std::string except_move )
	{
		string sfen = root_sfen;

		// same_colorならまず1手進める。
		if (same_color)
		{
			ExtMoves snlist;
			nnue_search(sfen , 1 , 10000 , snlist );
			// 指し手がない。この局面でresign。
			if (!snlist.size())
				return ;

			// この指し手で一手進めた局面のsfen文字列を作る。
			sfen = concat_sfen(sfen , to_usi_string(snlist[0].move));
		}

		// ここでponderする局面作る。
		{
			// 欲しい候補手の数。
			size_t multi_pv = available_engines.size() + (except_move.empty() ? 0 : 1);

			ExtMoves snlist;
			nnue_search(sfen , multi_pv , 10000 , snlist );
			// 指し手がない。この局面でresign。
			if (!snlist.size())
				return ;

			if (snlist.size() == 1 && to_usi_string(snlist[0].move) == except_move)
			{
				// 割り当て不可能。下手に割り当てると損する可能性があるからやめとく。
				return ;
			}

			ASSERT_LV3(snlist.size() <= available_engines.size());

			size_t next_snlist = 0;
			for(size_t i = 0 ; i < available_engines.size() ; ++i)
			{
				auto& engine = engines[available_engines[i]];

				ExtMove move;
				string move_str;

				do {
					move = snlist[next_snlist].move;
					next_snlist = (next_snlist + 1) % snlist.size(); // 足りなければ同じ指し手を2回(以上)割り当てる。
					move_str = to_usi_string(move);
				} while (move_str == except_move); // except_moveの指し手は除外する。

				engine.send(Message(USI_Message::GO_PONDER, string() , sfen + " " + move_str));
			}
		}
	}

	void MultiPonderStrategy::on_idle(StrategyParam& param)
	{
		// GUI側からのGOコマンドで探索していないなら何もしない。
		if (searching_sfen.empty())
			return ;

		// すべてのbestmoveが来てから。
		// ただし、一番最初にbestmoveを返してきたengineを基準とする。
		auto& engines = param.engines;

		// bestmoveを返したエンジンの数
		int bestmove_received = 0;
		// GUI側から"go"で指定された局面を探索しているエンジンの数。
		vector<size_t> go_engines;
		for(size_t i = 0 ; i < engines.size() ; ++i)
		{
			auto& engine = engines[i];
			if (!engine.peek_bestmove().empty())
				++bestmove_received;
			if (engine.get_searching_sfen() == searching_sfen)
				go_engines.push_back(i);
		}

		// まだすべてのエンジンがbestmoveを返していない。
		if (bestmove_received < go_engines.size())
		{
			if (bestmove_received > 0)
			{
				// 少なくとも1つのエンジンはbestmoveを返したが、
				// まだ(この局面について思考している)全部のエンジンからbestmoveきてない。

				// stopを送信していないならすべてのengineに"stop"を送信してやる。
				if (!stop_sent)
				{
					for(size_t i : go_engines)
						engines[i].send(USI_Message::STOP);

					stop_sent = true;
				}
			}
			return ;
		}

		// 一番良い評価値を返してきているエンジンを探す。

		size_t best_engine = size_max;
		int    best_value  = int_min;
		vector<string>* best_log = nullptr;
		for(size_t i : go_engines)
		{
			auto& engine = engines[i];

			auto& log = *engine.peek_thinklog();
			// 末尾からvalueの書いてあるlogを探す。
			for(size_t j = log.size() ; j != 0 ; j --)
			{
				UsiInfo info;
				parse_usi_info(log[j-1], info);

				if (info.value != VALUE_NONE )
				{
					if (info.value > best_value)
					{
						best_value  = info.value;
						best_engine = i;
						best_log    = &log;
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
			error_to_gui("MultiPonderStrategy::on_idle , No think_log");
			Tools::exit();
		}

		// ここまでの思考ログをまとめてGUIに送信する。
		if (best_log != nullptr)
			for(auto& line : *best_log)
				send_to_gui(line);


		auto bestmove_str = engines[best_engine].peek_bestmove();

		// 思考ログのクリア
		// engineすべてからbestmoveを取り除いておく。
		for(size_t i : go_engines)
		{
			engines[i].clear_thinklog();
			engines[i].pull_bestmove();
		}

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

		auto sfen = concat_sfen(engines[best_engine].get_searching_sfen(), bestmove + " " + ponder);

		// エンジンの返した局面についてすでにponderをしているのか？
		bool already_ponder = false;
		vector<size_t> available_engines;
		for(size_t i = 0 ; i < engines.size() ; ++i)
		{
			auto& engine = engines[i];
			if (engine.get_searching_sfen() == sfen)
			{
				// 見つかったので、余っているエンジンに対して適当な局面を見繕ってponderする。
				already_ponder = true;
			}
			else {
				available_engines.push_back(i);
			}
		}

		auto sfen2 = concat_sfen(engines[best_engine].get_searching_sfen() , bestmove);

		if (already_ponder)
		{
			// 残りのエンジンで、それ以外の指し手を予想して思考する。
			search_and_ponder(engines, sfen2 , available_engines, false , ponder);

		} else {

			// すでにこの局面についてponderしているエンジンがなかった。
			// 仕方ないので、まず、bestmoveを返したエンジンでponderして、それ以外のエンジンで
			// それ以外の指し手を予想して思考する。

			engines[best_engine].send(Message(USI_Message::GO_PONDER, string() , sfen));

			available_engines.clear();
			for(size_t i = 0 ; i < engines.size() ; ++i)
				if (i != best_engine)
					available_engines.push_back(i);

			// 残りのエンジンで、それ以外の指し手を予想して思考する。
			search_and_ponder(engines, sfen2 , available_engines, false , ponder);
		}

		// 探索中の局面、複数あるので不定。
		searching_sfen.clear();

	}

	// ---------------------------------------
	//     RootSplitStrategy 
	// ---------------------------------------

	// root局面で指し手を分割するようにしたもの。
	// →　強くなかったので使っていない。コードの参考用。

	void RootSplitStrategy::on_connected(StrategyParam& param)
	{
		for(auto& engine : param.engines)
			engine.set_engine_mode(EngineMode(
				// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
				EngineMode::SEND_INFO_BEFORE_GAME
			));
	}

	void RootSplitStrategy::on_isready(StrategyParam& param)
	{
		stop_sent = false;
		state = EngineState::CONNECTED;
	}

	void RootSplitStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		auto& engines = param.engines;

		// goコマンドの対象局面
		auto  sfen    = command.position_sfen;

		// いま go ponderで思考している局面と、command.position_sfenが一致するなら、エンジンに"ponderhit"を送ってやれば良い。
		// いま、すべてのエンジンが同じ局面について"go ponder"しているはずなので、engines[0]だけ見て判定する。
		if (engines[0].is_state_go_ponder() && engines[0].get_searching_sfen() == sfen)
		{
			// 今回の"go"の時に渡された残り時間等のパラメーターをそのままに、"ponderhit XXX .."の形式でエンジン側に送信。
			for(auto& engine : engines)
				engine.send(Message(USI_Message::PONDERHIT, strip_command(command.command) ));
		}
		else
		{
			// この局面についてponderしていなかったので愚直に"go"コマンドで思考させる。
			//engine.send(command);

			// → root_splitのためにrootの指し手を分割して、"go"の"searchmoves"として指定する。
			const auto& sfen = command.position_sfen;
			auto  moves_list = make_search_moves(sfen, engines.size());
			for(size_t i = 0; i < engines.size() ; ++i)
			{
				auto& engine       = engines   [i];
				auto& search_moves = moves_list[i];
				engine.send(Message(USI_Message::GO, command.command + " wait_stop" + search_moves , sfen));
				// return_to_bestmoveが来るまでは待たないと。
			}
		}

		stop_sent = false;
		state = EngineState::GO;
	}

	void RootSplitStrategy::on_idle(StrategyParam& param)
	{
		// GOしてなければ何もする必要がない。
		if (state != EngineState::GO)
			return ;

		// すべてのbestmoveが来てから。
		// ただし、一番最初にbestmoveを返してきたengineを基準とする。
		auto& engines = param.engines;

		// time_to_return_bestmoveを返したエンジンの数
		int time_to_return_bestmove = 0;
		for(auto& engine : engines)
			if (engine.received_time_to_return_bestmove())
				++time_to_return_bestmove;
		// 1つもtime_to_return_bestmoveを返してないなら何もしない。
		//if (!time_to_return_bestmove)
		//	return ; 

		// →　全部のエンジンがtime_to_return_bestmoveを返していないと
		// 候補手が3手しかなくて..2手がmatedでそれが速攻返ってきたときにmatedの指し手を選んで困る。
		// ただし、この方式だと、bestmoveと2nd bestmoveのvalueの差が大きくて、本来bestmoveだけなら
		// もっと早くに返せる時に思考時間が長くなって困る。
		if (time_to_return_bestmove < engines.size())
			return ;

		// bestmoveを返したエンジンの数
		int bestmove_received = 0;
		for(auto& engine : engines)
			if (!engine.peek_bestmove().empty())
				++bestmove_received;

		// 一番良い評価値を返してきているエンジンを探す。

		size_t best_engine = size_max;
		int    best_value  = int_min;
		for(size_t i = 0 ; i < engines.size(); ++i)
		{
			auto& engine = engines[i];

			auto& log = *engine.peek_thinklog();
			// 末尾からvalueの書いてあるlogを探す。
			for(size_t j = log.size() ; j != 0 ; j --)
			{
				UsiInfo info;
				parse_usi_info(log[j-1], info);

				if (info.value != VALUE_NONE )
				{
					if (info.value > best_value)
					{
						best_value  = info.value;
						best_engine = i;
					}
					// valueが書いてあったのでこのエンジンに関して
					// ログを調べるのはこれで終わり。
					goto NEXT;
				}
			}
			// log上にvalueが一つも書いてなかった。こんなエンジンがあったのでは話にならない。
			return ;

		NEXT:;
		}
		// まだベストエンジンが求まっていない。
		//if (best_engine == size_max)
		//	return ;

		// ベストを返しているエンジンがまだbestmoveを返せる状態ではない。
		if (!engines[best_engine].received_time_to_return_bestmove())
			return ;

		// bestmoveを返せそう。
		// stopを送信していないならすべてのengineに"stop"を送信してやる。
		if (!stop_sent)
		{
			for(auto& engine : engines)
				engine.send(USI_Message::STOP);

			stop_sent = true;
			return ;
		}

		// まだすべてのエンジンがbestmoveを返していない。
		if (bestmove_received < engines.size())
			return ;

		// 一番良い評価値を返してきているエンジンを探す。

		vector<string>* best_log = engines[best_engine].peek_thinklog();

		// ここまでの思考ログをまとめてGUIに送信する。
		if (best_log != nullptr)
			for(auto& line : *best_log)
				send_to_gui(line);

		// 思考ログのクリア
		for(auto& engine : engines)
			engine.clear_thinklog();

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
		{
			stop_sent = false;
			state = EngineState::IDLE_IN_GAME;
			return ;
		}

		auto sfen = concat_sfen(engines[0].get_searching_sfen(), bestmove + " " + ponder);

		// すべてのengineのponderを送って、ベストを拾う。
		auto moves_list = make_search_moves(sfen, engines.size());
		for(size_t i = 0; i < engines.size() ; ++i)
		{
			auto& engine       = engines   [i];
			auto& search_moves = moves_list[i];
			engine.send(Message(USI_Message::GO_PONDER, "go ponder wait_stop" + search_moves , sfen));
		}
		stop_sent = false;
		state = EngineState::GO_PONDER;
	}

	// sfenを与えて、その局面の合法手を生成して、それをエンジンの数で分割したものを返す。
	std::vector<std::string> RootSplitStrategy::make_search_moves(const std::string& sfen , size_t engine_num)
	{
		vector<string> moves_list;
		for(size_t i = 0 ; i < engine_num ; ++i)
			moves_list.emplace_back(string());

		Position pos;
		std::deque<StateInfo> si;
		BookTools::feed_position_string(pos, sfen, si, [](Position&){});

		auto ml = MoveList<LEGAL>(pos);
		if (ml.size() < engine_num * 2)
		{
			// エンジン数より少ないと1手すら割り当てられない。
			// →　この時は普通の楽観合議で良い。
			// あと、one replyだと(1手しかない場合)、探索によってその指し手が正常に求まらない。
			// ゆえに、少なくとも2手が割り当たる状況でないならroot splitしない。
			// →　GPS clusterのように1手先で展開すべきだが、そこでもone replyだと困るわけで…。
			// きちんとした探索木の分割が必要である。

		} else {

			for(size_t i = 0 ; i < engine_num ; ++i)
				moves_list[i] = " searchmoves";

			// 生成された順に割り当てていく。
			for(size_t i = 0; i < ml.size(); ++i)
			{
				auto move = ml.at(i);
				moves_list[i % engine_num] += " " + to_usi_string(move.move);
			}
		}

		return moves_list;
	}

	// ---------------------------------------
	//     GpsClusterStrategy 
	// ---------------------------------------

	void GpsClusterStrategy::on_connected(StrategyParam& param)
	{
		for(auto& engine : param.engines)
			engine.set_engine_mode(EngineMode(
				// 接続後、対局前までにエンジン側から送られてきた"info ..."を、そのままGUIに流す。
				EngineMode::SEND_INFO_BEFORE_GAME
			));

		stop_sent = false;
	}

	void GpsClusterStrategy::on_go_command(StrategyParam& param, const Message& command)
	{
		auto& engines = param.engines;

		// goコマンドの対象局面
		auto  sfen    = command.position_sfen;

		// いま go ponderで思考している局面と、command.position_sfenが一致するなら、エンジンに"ponderhit"を送ってやれば良い。
		// いま、すべてのエンジンが同じ局面について"go ponder"しているはずなので、engines[0]だけ見て判定する。
		if (engines[0].is_state_go_ponder() && engines[0].get_searching_sfen() == sfen)
		{
			// 今回の"go"の時に渡された残り時間等のパラメーターをそのままに、"ponderhit XXX .."の形式でエンジン側に送信。
			for(auto& engine : engines)
				engine.send(Message(USI_Message::PONDERHIT, strip_command(command.command) ));
		}
		else
		{
			// この局面についてponderしていなかったので愚直に"go"コマンドで思考させる。
			//engine.send(command);

			// → root_splitのためにrootの指し手を分割して、"go"の"searchmoves"として指定する。
			const auto& sfen = command.position_sfen;
			auto  moves_list = make_search_moves(sfen, engines.size());
			for(size_t i = 0; i < engines.size() ; ++i)
			{
				auto& engine = engines   [i];
				auto& moves  = moves_list[i];
				engine.send(Message(USI_Message::GO, command.command + moves , sfen));
			}
		}
		stop_sent = false;
	}

	void GpsClusterStrategy::on_idle(StrategyParam& param)
	{
		// すべてのbestmoveが来てから。
		// ただし、一番最初にbestmoveを返してきたengineを基準とする。
		auto& engines = param.engines;

		// bestmoveを返したエンジンの数
		int bestmove_received = 0;
		for(auto& engine : engines)
			if (!engine.peek_bestmove().empty())
				++bestmove_received;

		// まだすべてのエンジンがbestmoveを返していない。
		if (bestmove_received < engines.size())
		{
			if (bestmove_received > 0)
			{
				// 少なくとも1つのエンジンはbestmoveを返したが、
				// まだ全部のエンジンからbestmoveきてない。

				// stopを送信していないならすべてのengineに"stop"を送信してやる。
				if (!stop_sent)
				{
					for(auto& engine : engines)
						engine.send(USI_Message::STOP);

					stop_sent = true;
				}
			}
			return ;
		}

		// 一番良い評価値を返してきているエンジンを探す。

		size_t best_engine = size_max;
		int    best_value  = int_min;
		vector<string>* best_log = nullptr;
		for(size_t i = 0 ; i < engines.size(); ++i)
		{
			auto& engine = engines[i];

			auto& log = *engine.peek_thinklog();
			// 末尾からvalueの書いてあるlogを探す。
			for(size_t j = log.size() ; j != 0 ; j --)
			{
				UsiInfo info;
				parse_usi_info(log[j-1], info);

				if (info.value != VALUE_NONE )
				{
					// 子節点なのでvalueの符号を反転。
					if (j == 0 || j == 1)
						info.value = -info.value;

					if (info.value > best_value)
					{
						best_value  = info.value;
						best_engine = i;
						best_log    = &log;
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
			error_to_gui("GpsClusterStrategy::on_idle , No think_log");
			Tools::exit();
		}

		// ここまでの思考ログをまとめてGUIに送信する。
		if (best_log != nullptr)
			for(auto& line : *best_log)
				send_to_gui(line);

		// 思考ログのクリア
		for(auto& engine : engines)
			engine.clear_thinklog();

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
		auto moves_list = make_search_moves(sfen, engines.size());
		for(size_t i = 0; i < engines.size() ; ++i)
		{
			auto& engine = engines   [i];
			auto& moves  = moves_list[i];
			engine.send(Message(USI_Message::GO_PONDER, "go ponder" + moves , sfen));
		}
	}

	// sfenを与えて、その局面の合法手を生成して、それをエンジンの数で分割したものを返す。
	std::vector<std::string> GpsClusterStrategy::make_search_moves(const std::string& sfen , size_t engine_num)
	{
		vector<string> moves_list;
		for(size_t i = 0 ; i < engine_num ; ++i)
			moves_list.emplace_back(string());

		Position pos;
		std::deque<StateInfo> si;
		BookTools::feed_position_string(pos, sfen, si, [](Position&){});

		auto ml = MoveList<LEGAL>(pos);
		if (ml.size() >= engine_num)
		{
			// MultiPV 2で探索して上位2手を1台ずつに割り当てる。
			// エンジン数より少ないと1手すら割り当てられない。

			ExtMoves snlist;
			nnue_search(sfen , 2 , 10000 , snlist );


			for(size_t i = 0 ; i < 2 ; ++i)
				//moves_list[i] = " searchmoves " + to_usi_string(snlist[i].move);
			// →　これ、一手しかないので、
			// 　　最小思考時間でbestmoveを返してくるエンジンがある。
			// (やねうら王もそうなっている)　これleafを展開してからでないとまずい。
				moves_list[i] = " " + to_usi_string(snlist[i].move);
				// 局面を1手進めて全探索。

			// 残りは3つ目のエンジンに割り当てる。
			moves_list[2] = " searchmoves";
			for(size_t i = 0; i < ml.size(); ++i)
			{
				auto move = ml.at(i).move;

				// MultiPVの上位2手は除外する。
				if (   move == snlist[0].move
					|| move == snlist[1].move)
					continue;

				moves_list[2] += " " + to_usi_string(move);
			}
		}

		return moves_list;
	}

}

#endif
