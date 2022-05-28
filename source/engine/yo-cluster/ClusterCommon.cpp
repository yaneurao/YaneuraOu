#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ClusterCommon.h"
#include "../../misc.h"
#include "../../usi.h"

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

	// sfen文字列("position"で渡されてくる文字列)に、
	// "bestmove XX ponder YY"の XX と YYの指し手を結合したsfen文字列を作る。
	// ただし、YYが普通の指し手でない場合("win"とか"resign"とかの場合)、この連結を諦め、空の文字列が返る。
	std::string concat_bestmove(const std::string&sfen, const std::string& bestmove)
	{
		Parser::LineScanner parser(bestmove);

		string token;
		string best_move;
		string ponder_move;

		while (!parser.eol())
		{
			token = parser.get_text();
			if (token == "bestmove")
				best_move = parser.get_text();
			else if (token == "ponder")
				ponder_move = parser.get_text();
		}

		// bestmoveで進めた局面を対局局面とする。
		if (!is_ok(USI::to_move16(best_move)))
			return string();

		// さらにponderの指し手が有効手なのであるなら、ここを第一ponderとすべき。
		if (!is_ok(USI::to_move16(ponder_move)))
			return string();

		string sfen2;
		sfen2 = concat_sfen(sfen , best_move);
		sfen2 = concat_sfen(sfen2, ponder_move);

		return sfen2;
	}

}

#endif
