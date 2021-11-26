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
	struct ProcessNegotiator
	{
		// 子プロセスの実行
		// app_path_  : エンジンの実行ファイルのpath (.batファイルでも可)
		void connect(const string& app_path_)
		{
			disconnect();
			terminated = false;
			
			wstring app_path = to_wstring(app_path_);

			ZeroMemory(&pi, sizeof(pi));
			ZeroMemory(&si, sizeof(si));

			si.cb = sizeof(si);
			si.hStdInput = child_std_in_read;
			si.hStdOutput = child_std_out_write;
			si.dwFlags |= STARTF_USESTDHANDLES;

			// Create the child process

			// カレントフォルダを実行ファイルの存在しているフォルダにして起動する。
			// current directoryは、ドライブレターから始まるpath文字列である必要があるのでそれを生成する。
			auto folder_path = Path::GetDirectoryName(Path::Combine(CommandLine::workingDirectory ,app_path_));
			// これ、wstringに変換する時に、workingDirectoryに日本語混じってると死ぬような気がしなくもないが…。

			bool success = ::CreateProcess(
				NULL, // ApplicationName
				(LPWSTR)app_path.c_str(),  // CmdLine
				NULL, // security attributes
				NULL, // primary thread security attributes
				TRUE, // handles are inherited
				0,    // creation flags
				NULL, // use parent's environment

				(LPWSTR)to_wstring(folder_path).c_str(),
				//NULL, // use parent's current directory
					  // ここにカレントディレクトリを指定する。
				
				&si,  // STARTUPINFO pointer
				&pi   // receives PROCESS_INFOMATION
			);

			if (success)
			{
				engine_path = app_path_;

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
	//          EngineNegotiator
	// ---------------------------------------

	// Engineに対して現在何をやっている状態なのかを表現するenum
	// ただしこれはEngineNegotiatorの内部状態だから、この状態をEngineNegotiatorの外部から参照してはならない。
	// (勝手にこれを見て状態遷移をされると困るため)
	enum EngineNegotiatorState
	{
		DISCONNECTED,      // 切断状態
		CONNECTED,         // 接続直後の状態
		WAIT_USIOK,        // "usiok"がをエンジンに送信待ち(connect直後)
		RECEIVED_USIOK,    // "usiok"コマンドがエンジンから返ってきた直後の状態。
		WAIT_READYOK,      // "readyok"待ち。"readyok"が返ってきたらIDLE_IN_GAMEになる。

		IDLE_IN_GAME,      // エンジンが対局中の状態。"position"コマンドなど受信できる状態

		PONDERING,         // "go ponder"中。ponderhitかstopが来ると状態はWAIT_BESTMOVEに。
		WAIT_BESTMOVE,	   // 思考が終了するのを待っている。("go"コマンドであり、"go ponder"ではない。
						   // 自動的にbestmoveが返ってくるはず。この間にくる"stop"は思考エンジンにそのまま送れば良い)
	};

	// EngineNegotiatorStateを文字列化する。
	string to_string(EngineNegotiatorState state)
	{
		const string s[] = { "DISCONNECTED", "CONNECTED", "WAIT_USI", "RECEIVED_USIOK", "WAIT_READYOK", "IDLE_IN_GAME",};
		return s[state];
	}

	// Engineに対して、Observerから送られてくるメッセージ
	enum EngineCommand
	{
		SEND_MESSAGE,   // "setoption"などいつでも実行できるコマンドだから何も考えずに送れ

		SEND_USI,       // "usi"を送れ

		SEND_ISREADY,   // "isready"を送れ
		SEND_POSITION,  // "position"コマンドを送れ。

		SEND_GO_PONDER, // "go ponder"コマンドを送れ。(ここは思考には流さないがponderhitした時はいままでのlogもGUIに流す) , param : message = 局面
		SEND_PONDERHIT, // "ponderhit"コマンドを送れ。ここ以降は、GUIに流す。また、ここまでのlogもGUIに流す。

		SEND_GO,        // "go"コマンドを送れ。(これは思考をGUIに流さないといけない) , param : message = 局面
	};

	// 汎用型
	struct Variant
	{
		Variant() {}
		Variant(string s) : content(s) {}
		Variant(s64 n) : content(n) {}

		// 文字列が格納されているとわかっている時にそれを取り出す
		string get_string() const { return std::get<0>(content); }

		// s64が格納されているとわかっている時にそれを取り出す
		s64 get_int() const { return std::get<1>(content); }

	private:
		std::variant<string, s64> content;
	};

	// Engineに対して、Observerから送られてくるメッセージ
	// これをConcurrentQueueで送信する。
	struct EngineCommandInfo
	{
		EngineCommandInfo(EngineCommand command_)
		{
			command = command_;
		}

		EngineCommandInfo(EngineCommand command_, string message_)
		{
			command = command_;
			message = Variant(message_);
		}

		// 文字列が格納されているとわかっている時にそれを取り出す
		string get_string() const { return message.get_string(); }

		// s64が格納されているとわかっている時にそれを取り出す
		s64 get_int() const { return message.get_int(); }

		// メッセージ種別
		EngineCommand command;

		// メッセージ内容
		Variant message;
	};

	// エンジンとやりとりするためのクラス
	// 状態遷移を指示するメソッドだけをpublicにしてあるので、
	// 外部からはこれを呼び出すことで状態管理を簡単化する考え。
	// 
	// このクラスにはsend()メソッドは用意しない。
	// 勝手にメッセージをエンジンにsendされては困る。現在の状態を見ながらしか送信してはならない。
	class EngineNegotiator
	{
	public:

		// [main thread]
		// エンジンを起動する。
		// このメソッドは起動直後に、最初に一度だけmain threadから呼び出す。
		// (これとdisconnect以外のメソッドは observer が生成したスレッドから呼び出される)
		// path : エンジンの実行ファイルpath
		void connect(const string& path,size_t engine_id_)
		{
			engine_id = engine_id_;
			neg.connect(path);

			if (is_terminated())
				// 起動に失敗したくさい。
				DebugMessage(": Error! : fail to connect = " + path);
			else
			{
				// エンジンが起動したので出力しておく。
				DebugMessage(": Invoke Engine , engine_path = " + path + " , engine_id = " + std::to_string(engine_id));

				// 起動直後でまだメッセージの受信スレッドが起動していないので例外的にmain threadからchange_state()を
				// 呼び出しても大丈夫。
				change_state(EngineNegotiatorState::CONNECTED);
			}
		}

		// [main thread]
		// エンジンと切断する。(これはmain threadから呼び出すことを想定)
		void disconnect()
		{
			neg.disconnect();
		}

		// [receive thread]
		// エンジンからメッセージを受信する(これは受信用スレッドから定期的に呼び出される
		// メッセージを一つでも受信したならtrueを返す。
		bool negotiate_engine()
		{
			if (is_terminated() && state != DISCONNECTED)
			{
				// 初回切断時にメッセージを出力。
				DebugMessage(": Error : process terminated , path = " + neg.get_engine_path());

				state = DISCONNECTED;
			}

			if (state == DISCONNECTED)
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
		bool is_gamemode() const { return state == IDLE_IN_GAME; }

		// [main thread][receive thread]
		// 現在のstateがRECEIVED_USIOK("usiok"を受信したの)か？
		bool is_received_usiok() const { return state == RECEIVED_USIOK; }

		// [main thread][receive thread]
		// エンジンの実行path
		string get_engine_path() const { return neg.get_engine_path(); }

		// -------------------------------------------------------
		//    USI message handler
		// -------------------------------------------------------

		// [main thread]
		// receive threadにメッセージを送信する。
		// 必ずこのメソッドを経由して行う。
		void send_command(const EngineCommandInfo& message)
		{
			commands.push(message);
		}

		// -------------------------------------------------------
		//    constructor/destructor
		// -------------------------------------------------------

		EngineNegotiator()
		{
			state = EngineNegotiatorState::DISCONNECTED;
		}

		// move constuctor
		EngineNegotiator(EngineNegotiator&& other)
			: neg(std::move(other.neg)), state(other.state), engine_id(other.engine_id)
		{}

	private:

		// [main thread][receive thread]
		// ProcessID(engine_id)を先頭に付与してDebugMessage()を呼び出す。
		// "[0] >usi" のようになる。
		void DebugMessage(const string& message)
		{
			DebugMessageCommon("[" + std::to_string(engine_id) + "]" + message);
		}

		// [receive thread]
		// エンジンに対する状態を変更する。
		// ただしこれは内部状態なので外部からstateを直接変更したり参照したりしないこと。
		// 変更は、receive threadにおいてのみなされるので、mutexは必要ない。
		void change_state(EngineNegotiatorState new_state)
		{
			DebugMessage(": change_state " + to_string(state) + " -> " + to_string(new_state));
			state = new_state;
		}

		// [receive thread]
		// エンジン側から送られてきたメッセージを配る(解読して適切な配達先に渡す)
		void dispatch_message(const string& message)
		{
			ASSERT_LV3(state != DISCONNECTED);

			// 受信したメッセージをログ出力しておく。
			DebugMessage("> " + message);

			istringstream is(message);
			string token;
			is >> token;

			switch (state)
			{
			case WAIT_USIOK:
				// この間に送られてくるエンジン0のメッセージはguiに出力してやる。
				// ただしusiokは送ってはダメ
				if (token == "usiok")
					change_state(RECEIVED_USIOK);
				else if (get_engine_id() == 0)
					send_to_gui(message);
				return;

			case WAIT_READYOK:
				// この間に送られてくるエンジン0のメッセージはguiに出力してやる。
				// ただしreadyokは送ってはダメ
				if (token == "readyok")
					change_state(IDLE_IN_GAME);
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

		// [receive thread]
		// コマンドを配信する。
		// 処理できたらtrue。できなかったらfalseを返す。
		bool dispatch_command(const EngineCommandInfo& info)
		{
			ASSERT_LV3(state != DISCONNECTED);

			switch (info.command)
			{
			case SEND_USI:
				if (state == CONNECTED || state == RECEIVED_USIOK || state == IDLE_IN_GAME)
				{
					change_state(WAIT_USIOK);
					send("usi");
					return true;
				}
				break;

			case SEND_ISREADY:
				if (state == CONNECTED || state == RECEIVED_USIOK || state == IDLE_IN_GAME)
				{
					change_state(WAIT_READYOK);
					send("isready");
					return true;
				}
				break;

			case SEND_MESSAGE:
				// 思考中以外ならいつでも実行できる系のコマンド。
				if (state == CONNECTED || state == RECEIVED_USIOK || state == IDLE_IN_GAME)
				{
					send(info.get_string());
					return true;
				}
			}

			// メッセージを処理できなかった。
			return false;
		}

		// メッセージをエンジン側に送信する。
		void send(const string& message)
		{
			DebugMessage("< " + message);

			neg.send(message);
		}

		// -------------------------------------------------------
		//    private members
		// -------------------------------------------------------

		// 子プロセスとやりとりするためのhelper
		ProcessNegotiator neg;

		// エンジンID
		size_t engine_id;

		// エンジンに対して何をやっている状態であるのか。
		EngineNegotiatorState state;

		// Observerから送られてきたメッセージ
		Concurrent::ConcurrentQueue<EngineCommandInfo> commands;
	};

	// ---------------------------------------
	//          cluster thinker
	// ---------------------------------------

	// エンジンの思考している状態
	struct ThinkingInfo
	{
		// 思考中の局面(思考してなければ empty)
		string thinking_pos;

		// ponderでの思考か？
		bool is_ponder;

		// これがいま読み筋を返しているメインのエンジンであり、
		// こいつの返すbestmoveはGUIにそのまま返さなければならない。
		bool is_main;

		// エンジン
		EngineNegotiator* engine;

		ThinkingInfo()
		{
			is_ponder = false;
			is_main = false;
		}
	};


	// 局面をどのエンジンに割り振るかなどを管理してくれるやつ。
	class ClusterThinker
	{
	public:

		// エンジンを設定する。
		// これはconnectの時に行われる。
		void set_engines(vector<EngineNegotiator>& engines)
		{
			think_engines.clear();
			think_engines.resize(engines.size());
			for (size_t i = 0; i < engines.size(); ++i)
				think_engines[i].engine = &engines[i];
		}

		void position_handler(istringstream& is)
		{
			// "position"までは解析が終わっているはず。これはis.tellg()で取れるから…。
			pos_string = is.str().substr((size_t)is.tellg() + 1);
		}

		void go_handler(istringstream& is)
		{
			// "go"までは解析が終わっているはず。これはis.tellg()で取れるから…。
			string pos_string = is.str().substr((size_t)is.tellg() + 1);

			// この局面についてすでに思考しているか
			const auto it = std::find_if(think_engines.begin(), think_engines.end(), [&](const ThinkingInfo& info) { return info.thinking_pos == pos_string; });
			if (it != think_engines.end())
			{
				if (it->is_ponder)
				{
					// このエンジンにponderhitを送信する。

				}
				else {
					// ponderしてない。ありえないはず？
					send_to_gui("already thinking sfen = " + pos_string);
					return;
				}
			}
			else {

			}


		}

	private:

		// 最後に送られてきた"position"コマンドの"position"以降の文字列
		string pos_string;

		// 最後に送られてきた"go"コマンドの"go"以降の文字列
		string go_string;

		vector<ThinkingInfo> think_engines;
	};

	// ---------------------------------------
	//          cluster observer
	// ---------------------------------------

	class ClusterObserver
	{
	public:
		// [main thread]
		// 起動後に一度だけ呼び出すべし。
		// すべてのエンジンが起動され、"usi","isready"が自動的に送信される。
		void connect() {

			engines.clear();
			stop = false;

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
				string engine_path = Path::Combine("engines/", line);

				// エンジンを起動する。
				size_t engine_id = engines.size();
				engines.emplace_back(EngineNegotiator());
				auto& engine = engines.back();
				engine.connect(engine_path , engine_id);
			}

			// 思考すべきエンジンを選ぶやつの初期化。
			thinker.set_engines(engines);

			neg_thread = std::thread([&](){ neg_thread_func(); });
		}

		// [main thread]
		// 全エンジンを停止させ、監視スレッドを終了する
		void disconnect()
		{
			// quitはすでにbroadcastしているのでいずれ自動的に切断される。

			for (auto& engine : engines)
				engine.disconnect();

			stop = true;
			neg_thread.join();
		}

		// [main thread][receive thread]
		// 生きているengineの数を返す。
		size_t live_workers_num() const {
			size_t c = 0;
			for (auto& engine : engines)
				c += engine.is_terminated() ? 0 : 1;
			return c;
		}

		// [main thread][receive thread]
		// すべてのWorkerがterminateしている。
		bool all_terminated() const {
			return live_workers_num() == 0;
		}

		// [main thread][receive thread]
		// engine 0 が生きているのか。
		bool is_engine0_alive() const
		{
			return engines.size() > 0 && !engines[0].is_terminated();
		}

		// -------------------------------------------------------
		//    USI message handler
		// -------------------------------------------------------

		// [main thread]
		// "setoption","getoption"などの、思考中以外ならいつでも実行できるコマンドをエンジンに送信する。
		void broadcast(const string& message)
		{
			EngineCommandInfo command(SEND_MESSAGE, message);

			for (auto& engine : engines)
				engine.send_command(command);
		}

		// [main thread]
		// "usi"コマンドを処理する。
		void usi_handler()
		{
			EngineCommandInfo command(SEND_USI);

			for (auto& engine : engines)
				engine.send_command(command);

			// 全エンジンがUSI_OKを返したか(終了したエンジンは除く)
			while (true)
			{
				bool all = true;
				for (auto& engine : engines)
					all &= engine.is_terminated() || engine.is_received_usiok();

				if (all)
					break;

				Tools::sleep(1);
			}

			// 全エンジン、"usiok"を返した。
			send_to_gui("usiok");
		}

		// [main thread]
		// "isready"コマンドを処理する。
		void isready_handler()
		{
			EngineCommandInfo command(SEND_ISREADY);

			// 全エンジンにまず"isready"を送信する。
			for (auto& engine : engines)
				engine.send_command(command);
			
			// 全エンジンがUSI_OKを返したか(終了したエンジンは除く)
			while (true)
			{
				bool all = true;
				for (auto& engine : engines)
					all &= engine.is_terminated() || engine.is_gamemode();

				if (all)
					break;

				Tools::sleep(1);

				// これ待ち時間にタイムアウトになると嫌だな…。定期的に改行とか送信すべきかも。
			}

			// 生きているエンジンの数と、生きているエンジンの番号の内訳を出力。
			// エンジンの番号は 0から連番。

			string lived_enigne;
			for (auto& engine : engines)
				if (!engine.is_terminated())
					lived_enigne += " " + std::to_string(engine.get_engine_id());

			send_to_gui("info string Number of live engines = " + std::to_string(live_workers_num()) + ",{" + lived_enigne +" }");

			send_to_gui("readyok");
		}

		// [main thread]
		// "position"コマンドを処理する。
		void position_handler(istringstream& position_string)
		{
			// とりあえず預かっておく。
			// どのエンジンがどう思考するかはこのあと考える。
			thinker.position_handler(position_string);
		}

		// [main thread]
		// "go"コマンドを処理する。
		void go_handler(istringstream& iss)
		{
			thinker.go_handler(iss);
		}

		// =============================================
		//             GUIに対する応答
		// =============================================

		// 先頭に"[H]"と付与してDebugMessageを出力する。
		// "[H]"はhost側の意味。
		void DebugMessage(const string& message)
		{
			DebugMessageCommon("[H]" + message);
		}

	private:

		// 各エンジンに対してメッセージを受信する。
		// これは受信専用スレッド。全エンジン共通。
		void neg_thread_func()
		{
			while (!stop)
			{
				bool received = false;
				for (auto& engine : engines)
					received |= engine.negotiate_engine();

				// 一つもメッセージを受信していないならsleepを入れて休ませておく。
				if (!received)
					Tools::sleep(1);
			}
		}

		// すべての思考エンジンを表現する。
		vector<EngineNegotiator> engines;

		// Engineからのメッセージをpumpするスレッドの停止信号
		atomic<bool> stop;

		// Engineに対してメッセージを送受信するスレッド
		std::thread neg_thread;

		// どの局面を思考すべきかを管理しているclass。
		ClusterThinker thinker;
	};

	// ---------------------------------------
	//        main loop for cluster
	// ---------------------------------------


	// cluster時のUSIメッセージの処理ループ
	void cluster_usi_loop(Position& pos, std::istringstream& is)
	{
		// USIメッセージの処理を開始している。いま何か出力してはまずい。

		// USI拡張コマンドの"cluster"コマンドに付随できるオプション
		// 例)
		// cluster debug
		//
		string token;
		while (is >> token)
		{
			// debug mode
			if (token == "debug")
				debug_mode = true;
		}

		// クラスターの子を生成してくれるやつ。
		ClusterObserver observer;
		observer.connect();

		// エンジン0 が 起動していなかったなら終了。
		if (!observer.is_engine0_alive())
		{
			send_to_gui("info string Error! Engine[0] is not alive.");
			Tools::exit();
		}

		string cmd;
		while (std::getline(cin, cmd))
		{
			// GUI側から受け取ったメッセージをログに記録しておく。
			observer.DebugMessage("< " + cmd);

			istringstream iss(cmd);
			iss >> token;

			if (token.empty())
				continue;

			if (token == "usi")
			{
				observer.usi_handler();
			}
			else if (token == "isready")
			{
				observer.isready_handler();
			}
			// "setoption"などはそのまま全エンジンに流してやる。
			// setoptionは全workerに同じメッセージが流される。この仕様まずいか？
			// 対称性のあるWorker(同じスペックのPC、同じエンジンを想定しているからいいか..)
			else if (token == "setoption" || token == "getoption" || token == "gameover" || token == "usinewgame" || token=="quit")
			{
				observer.broadcast(cmd);
			}
			else if (token == "position")
			{
				// 局面はObserverにとりあえず預けておいて、goコマンドが来たときに考えればいいか。
				observer.position_handler(iss);
			}
			else if (token == "go")
			{
				observer.go_handler(iss);
			}
			else {
				// 知らないコマンドなのでデバッグのためにエラー出力しておく。
				// 利便性からすると何も考えずにエンジンに送ったほうがいいかも？
				send_to_gui("Error! : Unknown Command : " + token);
			}

			if (token == "quit")
				break;

		}

		// 全エンジンの終了
		observer.disconnect();

		// 本プログラムの終了
		Tools::exit();
	}

} // namespace YaneuraouTheCluster

#endif // !defined(_WIN32)
#endif //  defined(YANEURAOU_ENGINE_DEEP)
