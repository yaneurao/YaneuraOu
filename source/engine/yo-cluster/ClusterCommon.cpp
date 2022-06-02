#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ClusterCommon.h"
#include "../../misc.h"
#include "../../usi.h"
#include "../../search.h"
#include "../../thread.h"

using namespace std;

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//          メッセージ出力
	// ---------------------------------------

	// DebugMessageCommon()を標準出力に出力するかのフラグ。
	bool debug_mode = false;
	// 余計な"info string"をdebug_modeにおいても出力しないようにするフラグ。
	bool skip_info  = false;
	// メッセージ出力をファイルに書き出すかのフラグ。
	bool file_log   = false;

	static FILE* file_log_ptr = nullptr;
	#pragma warning(disable:4996) // fopen()

	// DebugMessageCommon()でfileに書き出す時のmutex
	static std::mutex file_mutex;

	void DebugMessageCommon(const string& message)
	{
		if (file_log)
		{
			std::unique_lock<std::mutex> lk(file_mutex);

			if (file_log_ptr == nullptr)
				file_log_ptr = fopen("cluster-log.txt","a");
			fprintf(file_log_ptr , "%s", (const char*)message.c_str());
			fprintf(file_log_ptr , "\n");
			fflush(file_log_ptr);
			return ;
		}

		// デバッグモードの時は標準出力にも出力する。
		if (debug_mode)
		{
			// skip_infoがtrueなら、"info"文字列は出力しない。(これされると画面が流れて行って読めない。)
			if (skip_info && StringExtension::Contains(message, "info "))
				return;

			sync_cout << message << sync_endl;
		}

		// あとはファイルに出力したければ出力すれば良い。
	}

	// GUIに対してメッセージを送信する。
	void send_to_gui(const string& message)
	{
		DebugMessageCommon("[H]> " + message);

		// 標準出力に対して出力する。(これはGUI側に届く)
		sync_cout << message << sync_endl;
	}

	// "info string Error! : "をmessageの前方に付与してGUIにメッセージを出力する。
	void error_to_gui(const string& message)
	{
		send_to_gui("info string Error! : " + message);
	}

	// USI_Messageの文字列化
	std::string to_string(USI_Message usi)
	{
		const string s[] = {
			"NONE",

			"USI","ISREADY","SETOPTION",
			"USINEWGAME","GAMEOVER",
			"GO","GO_PONDER","PONDERHIT",

			"STOP",
			"QUIT"
		};

		return s[(int)usi];
	}

	// Messageクラスのメンバーを文字列化する
	std::string Message::to_string() const
	{
		if (command.empty())
			return "Message[" + YaneuraouTheCluster::to_string(message) + "]";

		if (position_sfen.empty())
			return "Message[" + YaneuraouTheCluster::to_string(message) + " : " + command + "]";

		return "Message[" + YaneuraouTheCluster::to_string(message) + " : " + command + "] : " + position_sfen;
	}

	// EngineNegotiatorStateを文字列化する。
	std::string to_string(EngineState state)
	{
		const string s[] = {
			"DISCONNECTED", "CONNECTED",
			"WAIT_USI", "WAIT_ISREADY", "WAIT_READYOK",
			"IDLE_IN_GAME", "GO", "PONDERING",
			"QUIT"
		};
		return s[(int)state];
	}

	// ---------------------------------------
	//          文字列操作
	// ---------------------------------------

	// "go XX YY"に対して1つ目のcommand("go")を取り除き、"XX YY"を返す。
	// コピペミスで"  go XX YY"のように先頭にスペースが入るパターンも正常に"XX YY"にする。
	std::string strip_command(const std::string& m)
	{
		// 現在の注目位置(cursor)
		size_t i = 0;

		// スペース以外になるまでcursorを進める。(先頭にあるスペースの除去)
		while (i < m.size() && m[i]==' ')
			++i;

		// スペースを発見するまで cursorを進める。(トークンの除去)
		while (i < m.size() && m[i]!=' ')
			++i;

		// スペース以外になるまでcursorを進める。(次のトークンの発見)
		while (i < m.size() && m[i]==' ')
			++i;

		// 現在のcursor位置以降の文字列を返す。
		return m.substr(i);
	}

	// sfen文字列("position"で渡されてくる文字列)を連結する。
	// sfen1 == "startpos" , moves = "7g7f"の時に、
	// "startpos moves 7g7f"のように連結する。
	std::string concat_sfen(const std::string&sfen, const std::string& moves)
	{
		bool is_startpos = sfen == "startpos";
		return sfen + (is_startpos ? " moves " : " ") + moves;
	}

	// エンジン側が返してくる"bestmove XX ponder YY"の文字列からXXとYYを取り出す。
	// XX,YYが普通の指し手でない場合("win"とか"resign"とかの場合)、空の文字列を返す。
	// bestmove_str : [in ] "bestmove XX ponder YY" のような文字列
	// bestmove     : [out] XXの部分。
	// ponder       : [out] YYの部分。
	void parse_bestmove(const std::string& bestmove_str, std::string& bestmove, std::string& ponder)
	{
		Parser::LineScanner scanner(bestmove_str);

		string token;
		while (!scanner.eol())
		{
			token = scanner.get_text();
			if (token == "bestmove")
				bestmove = scanner.get_text();
			else if (token == "ponder")
				ponder = scanner.get_text();
		}

		// "win"とか"resign"なら空の文字列を返す。

		if (!is_ok(USI::to_move16(bestmove)))
			bestmove.clear();

		if (!is_ok(USI::to_move16(ponder)))
			ponder.clear();
	}

	// エンジン側から送られてきた"info .."をparseする。
	void parse_usi_info(const std::string& usi_info_string, UsiInfo& info)
	{
		Parser::LineScanner scanner(usi_info_string);

		string token;
		token = scanner.get_text();

		// "info"以外が来るのはおかしい。
		if (token != "info")
		{
			error_to_gui("parse_usi_info failed.");
			return ;
		}

		// "info string .."には評価値が付随していないので何もせずに帰る。
		if (scanner.peek_text() == "string")
			return ;

		while (!scanner.eol())
		{
			token = scanner.get_text();
			if (token == "score")
			{
				token = scanner.get_text();

				// score cp   <x>
				// score mate <y>

				// のいずれかだと思うので、VALUE型に変換する。

				if (token == "cp")
				{
					// cpからValueへのの変換が必要。
					info.value = Value(scanner.get_number(VALUE_NONE)) * /*PawnValue*/ 90 / 100;

				} else if (token == "mate")
				{
					int m = (int)scanner.get_number(0);
					info.value = (m >= 0)
						? mate_in ( m)
						: mated_in(-m);

				} else {
					// 知らんやつきた。
					error_to_gui("parse_usi_info , unknown token = " + token);
					return;
				}

				token = scanner.peek_text();
				if (token == "lowerbound")
				{
					info.lowerbound = true;
					scanner.get_text();
				}
				else if (token == "upperbound")
				{
					info.upperbound = true;
					scanner.get_text();
				}
			}
			else if (token == "nodes" || token == "time" || token == "depth" || token == "seldetph" || token == "multipv"
				|| token == "hashfull" || token == "nps")
				// パラメーターが一つ付随しているはずなので読み飛ばす。
				token = scanner.get_text();
			else if (token == "pv")
			{
				// PVは残りすべてが読み筋の文字列であることが保証されている。
				info.pv = scanner.get_rest();
				break;
			}
		}
	}

	// ---------------------------------------
	//          Search with multi pv
	// ---------------------------------------

#if defined(YANEURAOU_ENGINE_NNUE)

	// 与えられた局面をMultiPVで探索して上位の候補手を返す。
	// search_sfen : [in ] 探索したい局面のsfen("startpos moves .."みたいな文字列)
	// multi_pv    : [in ] multi pvの数
	// nodes_limit : [in ] 探索ノード数
	// snlist      : [out] 上位の候補手
	void nnue_search(const std::string& search_sfen , size_t multi_pv , int64_t nodes_limit , ExtMoves& snlist )
	{
		// ================================
		//        Limitsの設定
		// ================================

		Search::LimitsType limits = Search::Limits;

		// ノード数制限
		limits.nodes = nodes_limit;

		// 探索中にPVの出力を行わない。
		limits.silent = true;

		// 入玉ルールも考慮しておかないと。
		limits.enteringKingRule = EnteringKingRule::EKR_27_POINT;

		// MultiPVの値、無理やり変更してしまう。(本来、このあと元に戻すべきではある)
		Options["MultiPV"] = std::to_string(multi_pv);

		// ここで"go"に相当することをやろうとしているのでTimerはresetされていないと気持ち悪い。
		Time.reset();

		// ================================
		//           思考開始
		// ================================

		// SetupStatesは破壊したくないのでローカルに確保
		StateListPtr states(new StateList(1));

		// sfen文字列、Positionコマンドのparserで解釈させる。
		istringstream is(search_sfen);

		Position pos;
		position_cmd(pos, is, states);

		// 思考部にUSIのgoコマンドが来たと錯覚させて思考させる。
		Threads.start_thinking(pos, states , limits);
		Threads.main()->wait_for_search_finished();

		// 探索が完了したので結果を取得する。
		// 定跡にhitした場合、MultiPVの数だけ整列されて並んでないのか…。そうか…。

		snlist.clear();
		auto& rm = Threads.main()->rootMoves;
		for(size_t i = 0 ; i < multi_pv && i < rm.size(); ++i)
		{
			auto& r = rm[i];

			// "MOVE_WIN"の可能性はあるかも？
			if (!is_ok(r.pv[0]))
				continue;

			// この指し手のpvの更新が終わっているのか
			bool updated = r.score != -VALUE_INFINITE;
			Value v = updated ? r.score : r.previousScore;

			// 評価値、u64で表現できないので100で割って1000足しておく。
			ExtMove e;
			e.value = v;
			e.move  = r.pv[0];
			snlist.emplace_back(e);
		}
	}
#endif
}

#endif
