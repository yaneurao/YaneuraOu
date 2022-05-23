#ifndef CLUSTER_COMMOM_H_INCLUDED
#define CLUSTER_COMMOM_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include <string>

// YO Cluster、共通ヘッダー

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//          メッセージ出力
	// ---------------------------------------

	// デバッグ用のメッセージ(エンジンの状態や進捗など)の出力
	void DebugMessageCommon(const std::string& message);

	// GUIに対してメッセージを送信する。
	// yo_clusterからGUIに対してメッセージを送信したい時は、必ずこの関数を用いる。
	void send_to_gui(const std::string& message);

	// "info string Error! : "をmessageの前方に付与してGUIにメッセージを出力する。
	// yo_clusterからGUIに対してエラーメッセージを送信したい時は、必ずこの関数を用いる。
	void error_to_gui(const std::string& message);

	// -- 設定

	// DebugMessageCommon()を標準出力に出力するかのフラグ。
	extern bool debug_mode;
	// 余計な"info string"をdebug_modeにおいても出力しないようにするフラグ。
	extern bool skip_info ;
	// メッセージ出力をファイルに書き出すかのフラグ。
	extern bool file_log  ;

	// ---------------------------------------
	//          Message System
	// ---------------------------------------

	// Message定数
	// ※　ここに追加したら、to_string(USI_Message usi)のほうを修正すること。
	enum class USI_Message
	{
		// 何もない(無効な)メッセージ
		NONE,

		USI,
		ISREADY,
		SETOPTION,
		USINEWGAME,
		GAMEOVER,
		POSITION,
		GO,
		GO_PONDER,
		PONDERHIT,

		STOP,
		QUIT,
	};

	// USI_Messageの文字列化
	extern std::string to_string(USI_Message usi);

	// Supervisorに対して通信スレッドから送信するメッセージ。
	// SupervisorからObserverに対して送信するメッセージもこのメッセージを用いる。
	// 
	// エンジン側からresultを返したいことは無いと思うので完了を待つ futureパターンを実装する必要はない。
	// 単に完了が待てれば良い。また完了は逐次実行なので何番目のMessageまで実行したかをカウントしておけば良いので
	// この構造体に終了フラグを持たせる必要がない。(そういう設計にしてしまうと書くのがとても難しくなる)
	//
	struct Message
	{
		Message(USI_Message message_)
			: message(message_) , command()                                           {}
		Message(USI_Message message_, const std::string& command_)
			: message(message_) , command(command_)                                   {}
		Message(USI_Message message_, const std::string& command_, const std::string& position_sfen_)
			: message(message_) , command(command_) , position_sfen(position_sfen_)   {}

		// メッセージ種別。
		const USI_Message message;

		// パラメーター。
		// GUI側から送られてきた1行がそのまま入る。
		const std::string command;

		// 追加のパラメーター
		// GO , GO_PONDER に対しては、思考すべき局面のsfen文字列が入る。
		// (positionコマンドに付随している局面文字列。例 : "startpos moves 7g7f")
		const std::string position_sfen;

		// このクラスのメンバーを文字列化する
		std::string to_string() const;
	};

	// ---------------------------------------
	//          EngineNegotiator
	// ---------------------------------------

	// Engineに対して現在何をやっている状態なのかを表現するenum
	// ただしこれはEngineNegotiatorの内部状態だから、この状態をEngineNegotiatorの外部から参照してはならない。
	// (勝手にこれを見て状態遷移をされると困るため)
	enum class EngineState
	{
		DISCONNECTED,      // 切断状態
		CONNECTED,         // 接続直後の状態
		WAIT_USIOK,        // エンジンからの"usiok"待ち。エンジンから"usiok"が返ってきたら、WAIT_ISREADYになる。
		WAIT_ISREADY,      // "usiok"コマンドがエンジンから返ってきた直後の状態。あるいは、GUIからの"isready"待ち。"gameover"直後もこれ。
		WAIT_READYOK,      // エンジンからの"readyok"待ち。エンジンから"readyok"が返ってきたらIN_GAMEになる。

		IDLE_IN_GAME,      // エンジンが対局中の状態。"position"コマンドなど受信できる状態

		GO,                // エンジンが"go"で思考中。 GUI側から"ponderhit"か"stop"が来ると状態はWAIT_BESTMOVEに。
		GO_PONDER,         // エンジンが"go ponder"中。GUI側から"ponderhit"か"stop"が来ると状態はWAIT_BESTMOVEに。
		QUIT,              // "quit"コマンド送信後。
	};

	// EngineNegotiatorStateを文字列化する。
	extern std::string to_string(EngineState state);

}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif
