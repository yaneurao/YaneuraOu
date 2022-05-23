#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ClusterCommon.h"
#include "../../misc.h"

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
			"POSITION","GO","GO_PONDER","PONDERHIT",

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

}

#endif
