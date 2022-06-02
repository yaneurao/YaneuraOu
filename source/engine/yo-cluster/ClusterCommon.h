#ifndef CLUSTER_COMMOM_H_INCLUDED
#define CLUSTER_COMMOM_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include <string>
#include <vector>
#include "../../types.h"

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
	// また、POSITIONは存在しない。局面は、GO, GO_PONDERコマンドに付随しているという考え。
	enum class USI_Message
	{
		// 何もない(無効な)メッセージ
		NONE,

		USI,
		ISREADY,
		SETOPTION,
		USINEWGAME,
		GAMEOVER,
		GO,
		GO_PONDER,
		PONDERHIT,

		STOP,
		QUIT,
	};

	// USI_Messageの文字列化
	extern std::string to_string(USI_Message usi);

	// EngineNegotiatorに対してClusterObserverから送信するメッセージ。
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

		/*
			USI_Messageに対する、commandとposition_sfenの意味について。

				GO        : command = "go btime 0 wtime 0 byoyomi 1000"のような文字列。
							position_sfen = "startpos moves XX .."のような文字列。

				GO_PONDER : command = "go ponder searchmoves XXX"のような文字列。command が 空の文字列なら、"go ponder"を指定したことになる。 
							position_sfen = "startpos moves XX .."のような文字列。

				SET_OPTION: command = "setoption name XX value YY"のような文字列。
							position_sfen 未使用。

				USI       :
				ISREADY   :
				USINEWGAME:
				STOP      :
				GAMEOVER  :
				QUIT      :
							command , position_sfen 未使用 
		*/


		// メッセージ種別。
		const USI_Message message;

		// パラメーター。
		// GUI側から送られてきた1行がそのまま入る。
		const std::string command;

		// 追加のパラメーター。
		// message が GO , GO_PONDER の時は、思考すべき局面。
		// (直前に送られてきたpositionコマンドの"position"の文字列を剥がしたもの。例 : "startpos moves 7g7f")
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

		GO,                // エンジンが"go"("go ponder"は含まない)で思考中。
						   // GUI側から"ponderhit"か"stop"が来ると状態はWAIT_BESTMOVEに。
		GO_PONDER,         // エンジンが"go ponder"中。
						   // GUI側から"ponderhit"か"stop"が来ると状態はWAIT_BESTMOVEに。

		QUIT,              // "quit"コマンド送信後。
	};

	// EngineNegotiatorStateを文字列化する。
	extern std::string to_string(EngineState state);

	// ---------------------------------------
	//          文字列操作
	// ---------------------------------------

	// "go XX YY"に対して1つ目のcommand("go")を取り除き、"XX YY"を返す。
	// コピペミスで"  go XX YY"のように先頭にスペースが入るパターンも正常に"XX YY"にする。
	std::string strip_command(const std::string& m);

	// sfen文字列("position"で渡されてくる文字列)を連結する。
	// sfen1 == "startpos" , moves = "7g7f"の時に、
	// "startpos moves 7g7f"のように連結する。
	std::string concat_sfen(const std::string&sfen, const std::string& moves);

	// エンジン側が返してくる"bestmove XX ponder YY"の文字列からXXとYYを取り出す。
	// XX,YYが普通の指し手でない場合("win"とか"resign"とかの場合)、空の文字列を返す。
	// bestmove_str : [in ] "bestmove XX ponder YY" のような文字列
	// bestmove     : [out] XXの部分。
	// ponder       : [out] YYの部分。
	void parse_bestmove(const std::string& bestmove_str, std::string& bestmove, std::string& ponder);

	// エンジン側から送られてきた"info .."の文字列をparseした結果を格納する構造体。
	struct UsiInfo
	{
		// 評価値
		// 文字列中に見つからなければVALUE_NONEのまま。
		Value value = VALUE_NONE;

		// upperboundがついていたのか
		bool upperbound = false;

		// lowerboundがついていたのか
		bool lowerbound = false;

		// pv文字列。("7g7f 2c2d"のような指し手文字列)
		std::string pv;
	};

	// エンジン側から送られてきた"info .."をparseする。
	void parse_usi_info(const std::string& usi_info_string, UsiInfo& info);

	// ---------------------------------------
	//          ClusterOptions
	// ---------------------------------------

	// クラスタリング時のオプション設定
	struct ClusterOptions
	{
		// すべてのエンジンが起動するのを待つかどうかのフラグ。(1つでも起動しなければ、終了する)
		bool wait_all_engines_wakeup = true;

		// GUI側から送られてきた"setoption"を無視する。
		// "cluster ignore_setoption"と指定すると、これがtrueになる。
		// エンジン設定を個別に"engine_options.txt"を用いて設定する場合、
		// GUI側から全部のエンジンにbroadcastされると迷惑なのでこのオプションを用意した。
		bool ignore_setoption = false;

		// go ponderする局面を決める時にふかうら王で探索するノード数
		// 3万npsだとしたら、1000で1/30秒。GPUによって調整すべし。
#if defined(YANEURAOU_ENGINE_DEEP)
		uint64_t  nodes_limit = 1000;
#elif defined(YANEURAOU_ENGINE_NNUE)
		uint64_t  nodes_limit = 10000; // 1スレでも0.01秒未満だと思う。
#endif

	};

	// ---------------------------------------
	//          Search with multi pv
	// ---------------------------------------

#if defined(YANEURAOU_ENGINE_NNUE)

	typedef std::vector<ExtMove> ExtMoves;

	// 与えられた局面をMultiPVで探索して上位の候補手を返す。
	// search_sfen : [in ] 探索したい局面のsfen("startpos moves .."みたいな文字列)
	// multi_pv    : [in ] multi pvの数
	// nodes_limit : [in ] 探索ノード数
	// snlist      : [out] 上位の候補手
	void nnue_search(const std::string& search_sfen , size_t multi_pv , int64_t nodes_limit , ExtMoves& snlist );
#endif

}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif
