#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)
#if !defined(_WIN32)

// Windows以外の環境は未サポート

#include <sstream>
#include "../../position.h"

namespace YaneuraouTheCluster
{
	// cluster時のUSIメッセージの処理ループ
	void cluster_usi_loop(Position& pos, std::istringstream& is)
	{
		std::cout << "YaneuraouTheCluster does not work on non-Windows systems now." << std::endl;
	}
}

#else

// ------------------------------------------------------------------------------------------
// YaneuraouTheCluster
// 
// ※　ここで言うClusterとは、ネットワークを介して複数のUSI対応思考エンジンが協調動作すること。
// ------------------------------------------------------------------------------------------
//
// 現状、子プロセスを起動する部分、Windows用の実装しか用意していない。
//
// 
// ■　用語の説明
//
// parent(host)  : このプログラム。ふかうら王を用いて動作する。USIエンジンのふりをする。思考する局面をworkerに割り振る。
// worker(guest) : 実際に思考するプログラム。USI対応の思考エンジンであれば何でも良い。実際はsshで接続する。
//
// 思考エンジンのリスト)
//	  "engines/engine_list.txt"に思考エンジンの実行pathを書く。(何個でも書ける)
//    1行目は、リカバリー用のエンジンなので、途中で切断されないようにすること。(切断されると、本エンジン自体が終了してしまう)
//	  上のファイルに記述するエンジンの実行pathは、"engines/"相対path。例えば、"engine1/YaneuraOuNNUE.exe"と書いた場合、
//	  "engines/engine1/YaneuraOuNNUE.exe"を見に行く。
//	  また、エンジンとして、 .bat も書ける。
//    あと、sshコマンド自体も書ける。
//    例) ssh -i "yaneen-wcsc32.pem" ubuntu@xx.xxx.xxx.xx ./YaneuraOu-by-gcc
// 
// 思考エンジンの用意)
//    ローカルPCに配置するなら普通に思考エンジンの実行pathを書けば良い。
//    リモートPCに配置するならsshを経由して接続すること。例えばWindowsの .bat ファイルとして ssh 接続先 ./yaneuraou-clang
//        のように書いておけば、この.batファイルを思考エンジンの実行ファイルの代わりに指定した時、これが実行され、
//        sshで接続し、リモートにあるエンジンが起動できる。

// 接続の安定性
//     接続は途中で切断されないことが前提ではある。
//     少なくとも、1つ目に指定したエンジンは切断されないことを想定している。
//     2つ目以降は切断された場合は1つ目に指定したエンジンでの思考結果を返し、その思考エンジンへの再接続は行わない。

// 起動後 "cluster"というコマンドが入力されることを想定している。
// 起動時の引数で指定すればいいと思う。

// yane-cluster.bat
// に
// YaneuraOuCluster.exe
// と書いて(↑これが実行ファイル名)、
//   yane-cluster.bat cluster
// とすればいいと思う。

#include <sstream>
#include <thread>
#include <variant>
#include "../../position.h"
#include "../../thread.h"
#include "../../usi.h"
#include "../dlshogi-engine/dlshogi_min.h"

#include <Windows.h>

// ↓これを↑これより先に書くと、byteがC++17で追加されているから、Windows.hのbyteの定義のところでエラーが出る。
using namespace std;

namespace YaneuraouTheCluster
{
	// 構成)
	//  GUI側 -- host(本プログラム) -- Worker(実際の探索に用いる思考エンジン) ×複数
	// となっている。
	//  hostである本プログラムは、guiとやりとりしつつ、それをうまくWorker×複数とやりとりする。

	// ---------------------------------------
	//          メッセージ出力
	// ---------------------------------------

	// デバッグ用に標準出力にデバッグメッセージ(進捗など)を出力するのか？
	static bool debug_mode = false;

	void DebugMessageCommon(const string& message)
	{
		// デバッグモードの時は標準出力にも出力する。
		if (debug_mode)
			sync_cout << message << sync_endl;

		// あとはファイルに出力したければ出力すれば良い。
	}

	// GUIに対してメッセージを送信する。
	void send_to_gui(const string& message)
	{
		DebugMessageCommon("[H]> " + message);

		// 標準出力に対して出力する。(これはGUI側に届く)
		sync_cout << message << sync_endl;
	}

	// ---------------------------------------
	//          ProcessNegotiator
	// ---------------------------------------

	// 子プロセスを実行して、子プロセスの標準入出力をリダイレクトするのをお手伝いするクラス。
	// 1つの子プロセスのつき、1つのProcessNegotiatorの instance が必要。
	// 
	// 親プロセス(このプログラム)の終了時に、子プロセスを自動的に終了させたいが、それは簡単ではない。
	// アプリケーションが終了するときに、子プロセスを自動的に終了させる方法 : https://qiita.com/kenichiuda/items/3079ab93dae564dd5d17
	// 親プロセスは必ず quit コマンドか何かで正常に終了させるものとする。
	struct ProcessNegotiator
	{
		// 子プロセスの実行
		// workingDirectory : エンジンを実行する時の作業ディレクトリ 
		// app_path         : エンジンの実行ファイルのpath (.batファイルでも可) 絶対pathで。
		void connect(const string& workingDirectory , const string& app_path)
		{
			disconnect();
			terminated = false;

			ZeroMemory(&pi, sizeof(pi));
			ZeroMemory(&si, sizeof(si));

			si.cb = sizeof(si);
			si.hStdInput = child_std_in_read;
			si.hStdOutput = child_std_out_write;
			si.dwFlags |= STARTF_USESTDHANDLES;

			// Create the child process

			DebugMessageCommon("workingDirectory = " + workingDirectory + " , " + app_path);

			bool success = ::CreateProcess(
				NULL,                                         // ApplicationName
				(LPWSTR)to_wstring(app_path).c_str(),         // CmdLine
				NULL,                                         // security attributes
				NULL,                                         // primary thread security attributes
				TRUE,                                         // handles are inherited
				CREATE_NO_WINDOW,                             // creation flags
				NULL,                                         // use parent's environment
				(LPWSTR)to_wstring(workingDirectory).c_str(), // ここに作業ディレクトリを指定する。(NULLなら親プロセスと同じ)
				&si,                                          // STARTUPINFO pointer
				&pi                                           // receives PROCESS_INFOMATION
			);

			if (success)
			{
				engine_path = app_path;

			} else {
				terminated = true;
				engine_path = string();
			}
		}

		// 子プロセスへの接続を切断する。
		void disconnect()
		{
			if (pi.hProcess) {
				if (::WaitForSingleObject(pi.hProcess, 1000) != WAIT_OBJECT_0) {
					::TerminateProcess(pi.hProcess, 0);
				}
				::CloseHandle(pi.hProcess);
				pi.hProcess = nullptr;
			}
			if (pi.hThread)
			{
				::CloseHandle(pi.hThread);
				pi.hThread = nullptr;
			}
		}

		// 接続されている子プロセスから1行読み込む。
		string receive()
		{
			// 子プロセスが終了しているなら何もできない。
			if (terminated)
				return string();

			auto result = receive_next();
			if (!result.empty())
				return result;

			DWORD dwExitCode;
			::GetExitCodeProcess(pi.hProcess, &dwExitCode);
			if (dwExitCode != STILL_ACTIVE)
			{
				// 切断されているので 空の文字列を返す。
				terminated = true;
				return string();
			}

			// ReadFileは同期的に使いたいが、しかしデータがないときにブロックされるのは困るので
			// pipeにデータがあるのかどうかを調べてからReadFile()する。

			DWORD dwRead, dwReadTotal, dwLeft;
			CHAR chBuf[BUF_SIZE];

			// bufferサイズは1文字少なく申告して終端に'\0'を付与してstring化する。

			BOOL success = ::PeekNamedPipe(
				child_std_out_read, // [in]  handle of named pipe
				chBuf,              // [out] buffer     
				BUF_SIZE - 1,       // [in]  buffer size
				&dwRead,            // [out] bytes read
				&dwReadTotal,       // [out] total bytes avail
				&dwLeft             // [out] bytes left this message
			);

			if (success && dwReadTotal > 0)
			{
				success = ::ReadFile(child_std_out_read, chBuf, BUF_SIZE - 1, &dwRead, NULL);

				if (success && dwRead != 0)
				{
					chBuf[dwRead] = '\0'; // 終端マークを書いて文字列化する。
					read_buffer += string(chBuf);
				}
			}
			return receive_next();
		}

		// 接続されている子プロセス(の標準入力)に1行送る。改行は自動的に付与される。
		bool send(const string& message)
		{
			// すでに切断されているので送信できない。
			if (terminated)
				return false;

			string s = message + "\r\n"; // 改行コードの付与
			DWORD dwWritten;
			BOOL success = ::WriteFile(child_std_in_write, s.c_str(), DWORD(s.length()), &dwWritten, NULL);
			
			return success;
		}

		// プロセスの終了判定
		bool is_terminated() const { return terminated; }

		// エンジンの実行path
		// これはconnectの直後に設定され、そのあとは変更されない。connect以降でしか
		// このプロパティにはアクセスしないので同期は問題とならない。
		string get_engine_path() const { return engine_path; }

		ProcessNegotiator() { init(); }
		virtual ~ProcessNegotiator() { disconnect(); }

		// atomicメンバーを含むので、copy constructorとmove constructorが必要。

		// copy constructor
		ProcessNegotiator(ProcessNegotiator& other)
		{
			pi = other.pi;
			si = other.si;

			child_std_out_read  = other.child_std_out_read;
			child_std_out_write = other.child_std_out_write;
			child_std_in_read   = other.child_std_in_read;
			child_std_in_write  = other.child_std_in_write;

			terminated  = other.terminated.load();
			read_buffer = other.read_buffer;
			engine_path = other.engine_path;

			// move元のpiを潰すことで、move元のdestructorの呼び出しに対してdisconnectされないようにする。
			// cf. C++でemplace_backを使う際の注意点 : https://tadaoyamaoka.hatenablog.com/entry/2018/02/24/230731

			other.pi.hProcess = nullptr;
			other.pi.hThread = nullptr;
		}

		// move constuctor
		ProcessNegotiator(ProcessNegotiator&& other)
		{
			pi = other.pi;
			si = other.si;

			child_std_out_read  = other.child_std_out_read;
			child_std_out_write = other.child_std_out_write;
			child_std_in_read   = other.child_std_in_read;
			child_std_in_write  = other.child_std_in_write;

			terminated  = other.terminated.load();
			read_buffer = other.read_buffer;
			engine_path = other.engine_path;

			// move元のpiを潰すことで、move元のdestructorの呼び出しに対してdisconnectされないようにする。
			// cf. C++でemplace_backを使う際の注意点 : https://tadaoyamaoka.hatenablog.com/entry/2018/02/24/230731

			other.pi.hProcess = nullptr;
			other.pi.hThread = nullptr;
		}

	protected:

		// 確保している読み書きの行buffer size
		// 長手数になるかも知れないので…。512手×5byte(指し手)として4096あれば512手まではいけるだろう。
		static const size_t BUF_SIZE = 4096;

		void init()
		{
			terminated = false;

			// disconnectでpi.hProcessにアクセスするので0クリア必須。
			ZeroMemory(&pi, sizeof(pi));

			// pipeの作成

			SECURITY_ATTRIBUTES saAttr;

			saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
			saAttr.bInheritHandle = TRUE;
			saAttr.lpSecurityDescriptor = NULL;

			// エラーメッセージ出力用
			// 失敗しても継続はしないといけない。
			// これしかし、通常エラーにはならないので、失敗したら終了していいと思う。
			// (たぶんメモリ枯渇など)
			auto ERROR_MES = [&](const string& mes) {
				sync_cout << mes << sync_endl;
				Tools::exit();
			};

			if (!::CreatePipe(&child_std_out_read, &child_std_out_write, &saAttr, 0))
				ERROR_MES("Error! : CreatePipe : std out");

			if (!::SetHandleInformation(child_std_out_read, HANDLE_FLAG_INHERIT, 0))
				ERROR_MES("Error! : SetHandleInformation : std out");

			if (!::CreatePipe(&child_std_in_read, &child_std_in_write, &saAttr, 0))
				ERROR_MES("Error! : CreatePipe : std in");

			if (!::SetHandleInformation(child_std_in_write, HANDLE_FLAG_INHERIT, 0))
				ERROR_MES("Error! : SetHandleInformation : std in");
		}

		string receive_next()
		{
			// read_bufferから改行までを切り出す
			auto it = read_buffer.find("\n");
			if (it == string::npos)
				return string();
			// 切り出したいのは"\n"の手前まで(改行コード不要)、このあと"\n"は捨てたいので
			// it+1から最後までが次回まわし。
			auto result = read_buffer.substr(0, it);
			read_buffer = read_buffer.substr(it + 1, read_buffer.size() - it);
			// "\r\n"かも知れないので"\r"も除去。
			if (result.size() && result[result.size() - 1] == '\r')
				result = result.substr(0, result.size() - 1);

			return result;
		}

		// wstring変換
		wstring to_wstring(const string& src)
		{
			size_t ret;
			wchar_t *wcs = new wchar_t[src.length() + 1];
			::mbstowcs_s(&ret, wcs, src.length() + 1, src.c_str(), _TRUNCATE);
			wstring result = wcs;
			delete[] wcs;
			return result;
		}

		PROCESS_INFORMATION pi;
		STARTUPINFO si;

		HANDLE child_std_out_read;
		HANDLE child_std_out_write;
		HANDLE child_std_in_read;
		HANDLE child_std_in_write;

		// プロセスが終了したかのフラグ
		atomic<bool> terminated;

		// 受信バッファ
		string read_buffer;

		// プロセスのpath
		string engine_path;
	};

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

		QUIT,
	};

	string to_string(USI_Message usi)
	{
		const string s[] = {
			"NONE",
			"USI","ISREADY","SETOPTION",
			"USINEWGAME","GAMEOVER",
			"POSITION","GO","GO_PONDER","PONDERHIT",
			"QUIT"
		};

		return s[(int)usi];
	}

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
			: message(message_) , param()         {}
		Message(USI_Message message_, const string& param_)
			: message(message_) , param(param_)   {}

		// メッセージ本体。
		const USI_Message message;

		// パラメーター。
		const string param;

		// このクラスのメンバーを文字列化する
		string to_string() const
		{
			if (param.empty())
				return "Message[" + YaneuraouTheCluster::to_string(message) + "]";

			return "Message[" + YaneuraouTheCluster::to_string(message) + " : " + param + "]";
		}
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
		WAIT_BESTMOVE,	   // エンジンが思考が終了するのを待っている。("go"コマンドであり、"go ponder"ではない。
						   // 自動的にbestmoveが返ってくるはず。この間にくる"stop"は思考エンジンにそのまま送れば良い)
		QUIT,              // "quit"コマンド送信後。
	};

	// EngineNegotiatorStateを文字列化する。
	string to_string(EngineState state)
	{
		const string s[] = {
			"DISCONNECTED", "CONNECTED",
			"WAIT_USI", "WAIT_ISREADY", "WAIT_READYOK",
			"IDLE_IN_GAME", "GO", "PONDERING", "WAIT_BESTMOVE",
			"QUIT"
		};
		return s[(int)state];
	}

	// エンジンとやりとりするためのクラス
	// 状態遷移を指示するメソッドだけをpublicにしてあるので、
	// 外部からはこれを呼び出すことで状態管理を簡単化する考え。
	// 
	// このクラスにはsend()メソッドは用意しない。
	// 勝手にメッセージをエンジンにsendされては困る。現在の状態を見ながらしか送信してはならない。
	class EngineNegotiator
	{
	public:

		// -------------------------------------------------------
		//    constructor/destructor
		// -------------------------------------------------------

		EngineNegotiator()
		{
			state = EngineState::DISCONNECTED;
		}

		// copy constuctor
		EngineNegotiator(EngineNegotiator& other)
			: neg(std::move(other.neg)), state(other.state), engine_id(other.engine_id)
		{}

		// move constuctor
		EngineNegotiator(EngineNegotiator&& other)
			: neg(std::move(other.neg)), state(other.state), engine_id(other.engine_id)
		{}

		// -------------------------------------------------------
		//    Methods
		// -------------------------------------------------------

		// [main thread]
		// エンジンを起動する。
		// このメソッドは起動直後に、最初に一度だけmain threadから呼び出す。
		// (これとdisconnect以外のメソッドは observer が生成したスレッドから呼び出される)
		// path : エンジンの実行ファイルpath ("engines/" 相対)
		void connect(const string& path, size_t engine_id_)
		{
			engine_id = engine_id_;

			// エンジンの作業ディレクトリ。これはエンジンを実行するフォルダにしておいてやる。
			string working_directory = Path::GetDirectoryName(Path::Combine(CommandLine::workingDirectory , "engines/" + path));

			// エンジンのファイル名。
			string engine_path = Path::Combine("engines/", path);

			// 特殊なコマンドを実行したいなら"engines/"とかつけたら駄目。
			if (StringExtension::StartsWith(path,"ssh"))
			{
				working_directory = Path::GetDirectoryName(Path::Combine(CommandLine::workingDirectory , "engines"));
				engine_path = path;
			}

			neg.connect(working_directory, engine_path);

			if (is_terminated())
				// 起動に失敗したくさい。
				DebugMessage(": Error! : fail to connect = " + path);
			else
			{
				// エンジンが起動したので出力しておく。
				DebugMessage(": Invoke Engine , engine_path = " + path + " , engine_id = " + std::to_string(engine_id));

				// 起動直後でまだメッセージの受信スレッドが起動していないので例外的にmain threadからchange_state()を
				// 呼び出しても大丈夫。
				change_state(EngineState::CONNECTED);
			}
		}

		// [SYNC] Messageを解釈してエンジンに送信する。
		// 結果はすぐに返る。
		void send(Message message)
		{
			// エンジンがすでに終了していたらコマンド送信も何もあったものではない。
			if (is_terminated())
				return ;

			switch(message.message)
			{
			case USI_Message::USI:
				state = EngineState::WAIT_USIOK;
				send_to_engine("usi");
				break;

			case USI_Message::SETOPTION:
				// 一応、警告だしとく。
				// "usiok"が返ってきて、ゲーム対局前("usinewgame"が来る前)の状態。
				if (state != EngineState::WAIT_ISREADY)
					EngineError("'setoption' should be sent before 'isready'.");

				// そのまま転送すれば良い。
				send_to_engine(message.param);
				break;

			case USI_Message::ISREADY:
				state = EngineState::WAIT_READYOK;
				send_to_engine("isready");
				break;

			case USI_Message::USINEWGAME:
				// 一応警告出しておく。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'usinewgame' should be sent after 'isready'.");
				send_to_engine("usinewgame");
				break;

			case USI_Message::GO:
				// TODO : エンジン側からbestmove来るまで次のgo送れないのでは…。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'go' should be sent when state is 'IDLE_IN_GAME'.");
				searching_sfen = message.param;
				send_to_engine("go ponder " + searching_sfen);
				state = EngineState::GO;
				break;

			case USI_Message::GO_PONDER:
				// TODO : エンジン側からbestmove来るまで次のgo ponder送れないのでは…。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'go ponder' should be sent when state is 'IDLE_IN_GAME'.");
				searching_sfen = message.param;
				send_to_engine("go ponder " + searching_sfen);
				state = EngineState::GO_PONDER;
				break;

			case USI_Message::GAMEOVER:
				// 一応警告出しておく。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'gameover' should be sent after 'isready'.");
				state = EngineState::WAIT_ISREADY;
				send_to_engine("gameover");
				break;

			case USI_Message::QUIT:
				state = EngineState::QUIT;
				send_to_engine("quit");
				break;
			}
		}

		// [SYNC]
		// エンジンからメッセージを受信する(これは受信用スレッドから定期的に呼び出される
		// メッセージを一つでも受信したならtrueを返す。
		bool receive()
		{
			if (is_terminated() && state != EngineState::DISCONNECTED)
			{
				// 初回切断時にメッセージを出力。
				DebugMessage(": Error : process terminated , path = " + neg.get_engine_path());

				state = EngineState::DISCONNECTED;
			}

			if (state == EngineState::DISCONNECTED)
				return false;

			// stateはchange_state()でしか変更されないが、
			// それを呼び出すのはreceive threadだけであり、かつ、
			// この関数はreceive threadで実行されているので、
			// ここ以降、stateがDISCONNECTEDではないことが保証されている。

			bool received = false;

			while (true)
			{
				string message = neg.receive();

				if (message.empty())
					break;

				received = true;

				// このメッセージを配る。
				dispatch_message(message);
			}

#if 0
			// コマンドがあるか
			while (commands.size() > 0)
			{
				// このコマンドを処理する。
				auto& command = commands.front();
				if (dispatch_command(command))
				{
					// コマンドが処理できたなら、いまのコマンドをPC-Queueから取り除いて
					// コマンド受信処理を継続する。
					commands.pop();
					received = true;
				}
				else {
					break;
				}
			}
#endif

			return received;
		}


		// -------------------------------------------------------
		//    Property
		// -------------------------------------------------------

		// [main thread][receive thread]
		// プロセスの終了判定
		bool is_terminated() const { return neg.is_terminated(); }

		// [main thread][receive thread]
		// エンジンIDを取得。
		size_t get_engine_id() const { return engine_id; }

		// [main thread][receive thread]
		// エンジンが対局中のモードに入っているのか？
		bool is_idle_in_game()       const { return state == EngineState::IDLE_IN_GAME; }

		// [main thread][receive thread]
		// 現在のstateが"isready"の送信待ちの状態か？
		bool does_wait_isready() const { return state == EngineState::WAIT_ISREADY; }

	private:
		// メッセージをエンジン側に送信する。
		void send_to_engine(const string& message)
		{
			DebugMessage("< " + message);

			neg.send(message);
		}

		// [main thread][receive thread]
		// ProcessID(engine_id)を先頭に付与してDebugMessage()を呼び出す。
		// "[0] >usi" のようになる。
		void DebugMessage(const string& message)
		{
			DebugMessageCommon("[" + std::to_string(engine_id) + "]" + message);
		}

		// エンジン番号を付与して、GUIに送信する。
		void EngineError(const string& message)
		{
			send_to_gui("info string [" +  std::to_string(engine_id) + "] Error! : " + message);
		}

		// [receive thread]
		// エンジンに対する状態を変更する。
		// ただしこれは内部状態なので外部からstateを直接変更したり参照したりしないこと。
		// 変更は、receive threadにおいてのみなされるので、mutexは必要ない。
		void change_state(EngineState new_state)
		{
			DebugMessage(": change_state " + to_string(state) + " -> " + to_string(new_state));
			state = new_state;
		}

		// [receive thread]
		// エンジン側から送られてきたメッセージを配る(解読して適切な配達先に渡す)
		void dispatch_message(const string& message)
		{
			ASSERT_LV3(state != EngineState::DISCONNECTED);

			// 受信したメッセージをログ出力しておく。
			DebugMessage("> " + message);

			istringstream is(message);
			string token;
			is >> token;

			if (token == "info")
			{
				// "Error"という文字列が含まれていたなら(おそらく"info string Error : "みたいな形)、
				// 何も考えずにGUIにそれを投げる。
				// "info string [engine id] : xxx"の形にしたほうがいいかな？
				if (StringExtension::Contains(message, "Error"))
				{
					send_to_gui("info string [" + std::to_string(engine_id) + "]> " + message);
					return ;
				}
			}

			switch (state)
			{
			case EngineState::WAIT_USIOK:
				// この間に送られてくるエンジン0のメッセージはguiに出力してやる。
				// ただしusiokは送ってはダメ
				if (token == "usiok")
					change_state(EngineState::WAIT_ISREADY);
				else if (get_engine_id() == 0)
					send_to_gui(message);
				return;

			case EngineState::WAIT_READYOK:
				// この間に送られてくるエンジン0のメッセージはguiに出力してやる。
				// ただしreadyokは送ってはダメ
				// → readyokは全部のスレッドが IDLE_IN_GAMEになった時に親クラス(Observer)が送る。
				if (token == "readyok")
					change_state(EngineState::IDLE_IN_GAME);
				else if (get_engine_id() == 0)
					send_to_gui(message);
				return;
			}

			// これは"usi"応答として送られてくる。
			// WAIT_USIOK/WAIT_READYOKの時しか送られてこないはずなのだが…。
			if (token == "id" || token == "option" || token == "usiok" || token == "isready")
			{
				// Warning
				DebugMessage(": Warning! : Illegal Message , state = " + to_string(state)
					+ " , message = " + message);
			}


		}

		// -------------------------------------------------------
		//    private members
		// -------------------------------------------------------

		// 子プロセスとやりとりするためのhelper
		ProcessNegotiator neg;

		// エンジンID
		size_t engine_id;

		// エンジンに対して何をやっている状態であるのか。
		EngineState state;

		// Supervisorから送られてくるMessageのqueue
		Concurrent::ConcurrentQueue<Message> queue;

		// Messageを処理した個数
		atomic<u64> done_counter = 0;

		// Messageをsendした回数
		atomic<u64> send_counter = 0;

		// 探索中の局面
		// state == GO or GO_PONDER において探索中の局面。
		string searching_sfen;
	};

	// ---------------------------------------
	//          cluster observer
	// ---------------------------------------

	// クラスタリング時のオプション設定
	struct ClusterOptions
	{
		// すべてのエンジンが起動するのを待つかどうかのフラグ。(1つでも起動しなければ、終了する)
		//bool wait_all_engines_wakeup = true;
		// →　これ今回はデフォルトでtrueでないとclusterの処理が煩雑になるので
		//    前提としてすべて起動していて、すべて生きている、切断されないことをその条件とする。

		// go ponderする局面を決める時にふかうら王で探索するノード数
		// 3万npsだとしたら、1000で1/30秒。GPUによって調整すべし。
		u64  nodes_limit = 1000;
	};

	class ClusterObserver
	{
	public:
		ClusterObserver(const ClusterOptions& options_)
		{
			// スレッドを開始する。
			worker_thread = std::thread([&](){ worker(); });
			options       = options_; 
		}

		~ClusterObserver()
		{
			send_wait(USI_Message::QUIT);
			worker_thread.join();
		}

		// [main thread]
		// 起動後に一度だけ呼び出すべし。
		void connect() {

			engines.clear();

			vector<string> lines;

			// エンジンリストが書かれているファイル
			string engine_list_path = "engines/engine_list.txt";

			if (SystemIO::ReadAllLines(engine_list_path, lines, true).is_not_ok())
			{
				cout << "Error! engine list file not found. path = " << engine_list_path << endl;
				Tools::exit();
			}

			// それぞれのengineを起動する。
			for (const auto& line : lines)
			{
				if (line.empty())
					continue;

				// engineの実行path。engines/配下にあるものとする。
				string engine_path = line;

				// エンジンを起動する。
				size_t engine_id = engines.size();
				engines.emplace_back(EngineNegotiator());
				auto& engine = engines.back();
				engine.connect(engine_path , engine_id);
			}

			// すべてのエンジンの起動完了を待つ設定なら起動を待機する。
			// →　現状、強制的にこのモードで良いと思う。
			if (/* options.wait_all_engines_wakeup */ true)
				wait_all_engines_wakeup();
		}

		// [ASYNC] 通信スレッドで受け取ったメッセージをこのSupervisorに伝える。
		//    waitとついているほうのメソッドは送信し、処理の完了を待機する。
		void send(USI_Message usi                     )       { send(Message(usi            )); }
		void send(USI_Message usi, const string& param)       { send(Message(usi, param     )); }
		void send_wait(USI_Message& usi)                      { send_wait(Message(usi       )); }
		void send_wait(USI_Message& usi, const string& param) { send_wait(Message(usi, param)); }

		// [ASYNC] Messageを解釈してエンジンに送信する。
		void send(Message message)
		{
			DebugMessageCommon("Observer send : " + message.to_string());

			queue.push(message);
			send_counter++;
		}

		// [ASNYC] 通信スレッドで受け取ったメッセージをこのSupervisorに伝える。
		//   また、そのあとメッセージの処理の完了を待つ。
		void send_wait(Message message)
		{
			send(message);

			// この積んだメッセージの処理がなされるまでwait。
			while (done_counter < send_counter)
				Tools::sleep(0);
		}

	private:
		// worker thread
		void worker()
		{
			bool quit = false;
			while (!quit)
			{
				bool received = false;

				// --------------------------------------------
				// 親クラスからの受信
				// --------------------------------------------

				if (queue.size() && usi == USI_Message::NONE)
				{
					received = true;
					auto message = queue.pop();

					// messageのdispatch
					switch(message.message)
					{
					case USI_Message::SETOPTION: // ←　これは状態関係なしに送れるコマンドのはずなので送ってしまう。
						broadcast(message);
						break;

					case USI_Message::USI:
					case USI_Message::ISREADY:
						usi = message.message; // ← この変数の状態変化まではエンジンの次のメッセージを処理しない。
						broadcast(message);
						break;

					case USI_Message::USINEWGAME:
						// まず各エンジンに通知は必要。(各エンジンがこのタイミングで何かをする可能性はあるので)
						broadcast(message);

						// ゲームが開始した。いま以降、エンジンに対して"go ponder"とかしてOk. むしろ積極的にすべき。
						usi = USI_Message::USINEWGAME;

						// 現在、相手が初期局面("startpos")について思考しているものとする。
						searching_sfen = "startpos";
						our_searching = false;
						
						// 各エンジンのponderの開始
						start_pondering();

						break;

					case USI_Message::GAMEOVER:
						usi = USI_Message::GAMEOVER;
						// 各エンジンへの通知は思考を停止させてからの話なので、いますぐは何も送らない。
						// エンジンが思考中なら停止させるような命令がいくので、停止してから"gameover"を送信すれば良いという考え。

						break;

					case USI_Message::QUIT:
						broadcast(message);

						// エンジン停止させて、それを待機する必要はある。
						quit = true;
						break;
					}

					done_counter++;
				}

				// --------------------------------------------
				// 子クラス(EngineNegotiator)のメッセージの受信
				// --------------------------------------------

				for (auto& engine : engines)
					received |= engine.receive();

				// 一つもメッセージを受信していないならsleepを入れて休ませておく。
				if (!received)
					Tools::sleep(1);

				// --------------------------------------------
				// 何かの状態変化を待っていたなら..
				// --------------------------------------------

				if (usi != USI_Message::NONE)
				{
					bool allOk = true;
					switch (usi)
					{
					case USI_Message::USI:
						// "usiok"をそれぞれのエンジンから受信するのを待機していた。
						for(auto& engine : engines)
							// 終了しているエンジンは無視してカウントしないと
							// いつまでもusiokが出せない状態でhangする。
							allOk &= engine.does_wait_isready() || engine.is_terminated();
						if (allOk)
						{
							send_to_gui("usiok");
							usi = USI_Message::NONE;
							output_number_of_live_engines();
						}
						break;

					case USI_Message::ISREADY:
						// "readyok"をそれぞれのエンジンから受信するのを待機していた。
						for(auto& engine : engines)
							allOk &= engine.is_idle_in_game() || engine.is_terminated();
						if (allOk)
						{
							send_to_gui("readyok");
							usi = USI_Message::NONE;
							output_number_of_live_engines();
						}
						break;

					case USI_Message::USINEWGAME:
						// 対局は開始しているので各エンジンに思考させたりする必要がある。
						break;

					case USI_Message::GAMEOVER:
						// 対局は終了しているので、探索中のエンジンは停止させる必要がある。
						break;
					}
				}

				// エンジンの死活監視
				engine_check();
			}

			// engine止める必要がある。
		}

		// 生きているエンジンの数を返す。
		size_t get_number_of_live_engines()
		{
			size_t num = 0;
			for(auto& engine: engines)
				if (!engine.is_terminated())
					num ++;
			return num;
		}

		// 生きているエンジンが 0 なら終了する。
		// 実際は、最初に起動させたエンジンの数と一致しないなら、終了すべきだと思うが…。
		// ※　エンジンが1つでも生存していればなんとか頑張って凌ぐようなプログラムを書きたいところである。
		void engine_check()
		{
			size_t num = get_number_of_live_engines();
			if (num == 0)
			{
				send_to_gui("info string All engines are terminated.");
				Tools::exit();
			}
		}

		// 生きているエンジンの数を出力する。
		void output_number_of_live_engines()
		{
			size_t num = get_number_of_live_engines();
			send_to_gui("info string The number of live engines = " + std::to_string(num));
		}

		// すべてのエンジンが起動するのを待つ。(1つでも起動しなければ、exitを呼び出して終了する)
		void wait_all_engines_wakeup()
		{
			Tools::sleep(3000); // 3秒待つ(この間にtimeoutになるやつとかおるかも)
			for(auto& engine: engines)
				if (engine.is_terminated())
				{
					size_t num = get_number_of_live_engines();
					send_to_gui("info string The number of live engines = " + std::to_string(num));
					send_to_gui("info string Some engines are failing to start.");
					
					Tools::exit();
				}
		}

		// 全エンジンに同じメッセージを送信する。
		void broadcast(Message message)
		{
			for(auto& engine: engines)
				engine.send(message);
		}


		// 各エンジンのponderを開始する。
		void start_pondering()
		{
			// 探索中の局面は定まっているか？
			if (searching_sfen.empty())
				return ; // ない

			// 我々が探索中の局面があるなら、その2手、4手、のように偶数手先の局面について局面を選出し、ponderする。
			// さもなくば現在相手が思考中の局面に対して、1手、3手のように奇数手先の局面について選出し、ponderする。

			search_for_ponder(searching_sfen, our_searching);

			// デバッグ用に逆側も出力してみる。
			DebugMessageCommon("---");
			search_for_ponder(searching_sfen, !our_searching);
		}

		// ponderする局面の選出。
		// search_sfen : この局面から探索してくれる。
		// same_color  : search_sfenと同じ手番の局面をponderで思考するのか？
		void search_for_ponder(string search_sfen,  bool same_color)
		{
			dlshogi::SfenNodeList snlist;

			// エンジンの数だけ選出する。
			size_t num = engines.size();

			// ただし、自分の手番であるなら、エンジンのうち一つはsearch_sfenを探索しているので、
			// 1つ数を減らす。
			if (our_searching)
				--num;

			dl_search(num, snlist, search_sfen, same_color);

			// debug用に出力してみる。
			for(auto& sn : snlist)
				DebugMessageCommon("sfen for pondering :" + sn.sfen + "(" + std::to_string(sn.nodes) + ")");

			// 局面が求まったので各エンジンに対して"go ponder"で思考させる。

			size_t i = 0;
			bool is_startpos = search_sfen == "startpos";
			for(auto& engine : engines)
			{
				// "startpos"に連結するなら"moves"を付与。
				string sfen = search_sfen + (is_startpos ? " moves" : "") + snlist[i].sfen;
				engine.send(Message(USI_Message::GO_PONDER, sfen));
				++i;
			}
		}

		// ノード数固定でふかうら王で探索させる。
		// 訪問回数の上位 n個のノードのsfenが返る。
		// n           : 上位n個
		// snlist      : 訪問回数上位のsfen配列
		// search_sfen : この局面から探索してくれる。
		// same_color  : search_sfenと同じ手番の局面をponderで思考するのか？
		void dl_search(size_t n, dlshogi::SfenNodeList& snlist, string search_sfen, bool same_color)
		{
			// ================================
			//        Limitsの設定
			// ================================

			Search::LimitsType limits = Search::Limits;

			// ノード数制限
			limits.nodes = options.nodes_limit;

			// 探索中にPVの出力を行わない。
			limits.silent = true;

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
			
			// ================================
			//        探索結果の取得
			// ================================

			dlshogi::GetTopVisitedNodes(n, snlist, same_color);
		}

		// --- private members ---

		ClusterOptions options;

		// すべての思考エンジンを表現する。
		vector<EngineNegotiator> engines;

		// Supervisorから送られてくるMessageのqueue
		Concurrent::ConcurrentQueue<Message> queue;

		// 現在エンジンに対して行っているコマンド
		// これがNONEになるまで次のメッセージは送信できない。
		USI_Message usi = USI_Message::NONE;

		// workerスレッド
		std::thread worker_thread;

		// Messageを処理した個数
		atomic<u64> done_counter = 0;

		// Messageをsendした回数
		atomic<u64> send_counter = 0;

		// 現在思考している局面のsfen。(startpos moves XX XX..の形式)
		// ponderする時は、この局面を中心として行う。
		string searching_sfen;  // 自分か相手がこの局面について思考しているものとする。(ponderする時の中心となる局面)
		bool our_searching;     // search_sfenを探索しているのは自分ならばtrue。相手ならばfalse。
	};

	// ---------------------------------------
	//        main loop for cluster
	// ---------------------------------------

	// クラスター本体
	class Cluster
	{
	public:
		// cluster時のUSIメッセージの処理ループ
		// これがUSIの通信スレッドであり、main thread。
		void message_loop(Position& pos, std::istringstream& is)
		{
			// ふかうら王のエンジン初期化(評価関数の読み込みなど)
			is_ready();

			// clusterのオプション設定
			ClusterOptions options;

			// "cluster"コマンドのパラメーター解析
			parse_cluster_param(is, options);

			// GUIとの通信を行うmain threadのmessage loop
			message_loop_main(pos, is, options);

			// quitコマンドは受け取っているはず。
			// ここで終了させないと、cluster engineが単体のengineのように見えない。

			Tools::exit();
		}

	private:
		// "cluster"コマンドのパラメーターを解析して処理する。
		// 
		// 指定できるオプション一覧)
		// 
		//   debug   : debug用に通信のやりとりをすべて標準出力に出力する。
		//   nodes   : go ponderする局面を選ぶために探索するノード数(ふかうら王で探索する)
		//
		void parse_cluster_param(std::istringstream& is, ClusterOptions& options)
		{
			// USIメッセージの処理を開始している。いま何か出力してはまずい。

			// USI拡張コマンドの"cluster"コマンドに付随できるオプション
			// 例)
			// cluster debug waitall
			{
				string token;
				while (is >> token)
				{
					// debug mode
					if (token == "debug")
						debug_mode = true;

					else if (token == "nodes")
						is >> options.nodes_limit;
				}
			}
		}

		// "cluster"のメインループ
		// wait_all : "waitall"(エンジンすべての起動を待つ)が指定されていたか。
		void message_loop_main(Position& pos, std::istringstream& is, const ClusterOptions& options)
		{
			// Clusterの監視者
			ClusterObserver observer(options);

			// 全エンジンの起動。
			observer.connect();

			while (true)
			{
				string cmd = std_input.input();

				// GUI側から受け取ったメッセージをログに記録しておく。
				// (ロギングしている時は、標準入力からの入力なのでファイルに書き出されているはず)
				DebugMessageCommon("[H]< " + cmd);

				istringstream iss(cmd);
				string token;
				iss >> token;

				if (token.empty())
					continue;

				if (token == "usi")
					observer.send_wait(USI_Message::USI);
				else if (token == "isready")
					observer.send_wait(USI_Message::ISREADY);
				else if (token == "setoption")
					// setoption は普通 isreadyの直前にしか送られてこないので
					// 何も考えずに エンジンにそのまま投げて問題ない。
					observer.send(USI_Message::SETOPTION, cmd);
				else if (token == "usinewgame")
					observer.send(USI_Message::USINEWGAME);
				else if (token == "gameover")
					observer.send(USI_Message::GAMEOVER);
				else if (token == "quit")
					break;
				// 拡張コマンド。途中でdebug出力がしたい時に用いる。
				else if (token == "debug")
					debug_mode = true;
				// 拡張コマンド。途中でdebug出力をやめたい時に用いる。
				else if (token == "nodebug")
					debug_mode = false;
				else {
					// 知らないコマンドなのでデバッグのためにエラー出力しておく。
					// 利便性からすると何も考えずにエンジンに送ったほうがいいかも？
					send_to_gui("Error! : Unknown Command : " + token);
				}
			}
			
			// ClusterObserverがスコープアウトする時に自動的にQUITコマンドは送信されるはず。
		}
	};

	// cluster時のUSIメッセージの処理ループ
	// これがUSIの通信スレッドであり、main thread。
	void cluster_usi_loop(Position& pos, std::istringstream& is)
	{
		Cluster theCluster;
		theCluster.message_loop(pos, is);
	}

} // namespace YaneuraouTheCluster

#endif // !defined(_WIN32)
#endif //  defined(YANEURAOU_ENGINE_DEEP)
