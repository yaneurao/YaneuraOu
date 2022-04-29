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
//     エンジンが不正終了したり、
//     ssh経由でエンジンに接続している時にエンジンとの接続が切断されてしまうと、本プログラムは思考を継続できなくなってしまう。
//

// 起動後 "cluster"というコマンドが入力されることを想定している。
// 起動時の引数で指定すればいいと思う。

// yane-cluster.bat
// に
// YaneuraOuCluster.exe
// と書いて(↑これが実行ファイル名)、
//   yane-cluster.bat cluster
// とすればいいと思う。

// 注意)
// 本プログラムが不正終了したりquitされる前に終了してしまうと、実行していたworkerのエンジンは実行したままになることがある。
// その場合は、実行していたエンジンをタスクマネージャーから終了させるなり何なりしなければならない。

#include <sstream>
#include <thread>
#include <variant>
#include "../../position.h"
#include "../../thread.h"
#include "../../usi.h"
#include "../dlshogi-engine/dlshogi_min.h"

// std::numeric_limits::max()みたいなのを壊さないように。
#define NOMINMAX
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

	// "info string Error! : "をmessageの前方に付与してGUIにメッセージを出力する。
	void error_to_gui(const string& message)
	{
		send_to_gui("info string Error! : " + message);
	}

	// ---------------------------------------
	//          文字列操作
	// ---------------------------------------

	// "go XX YY"に対して1つ目のcommand("go")を取り除き、"XX YY"を返す。
	// コピペミスで"  go XX YY"のように先頭にスペースが入るパターンも正常に"XX YY"にする。
	string strip_command(const string& m)
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

	// 何文字目まで一致したかを返す。
	size_t get_match_length(const string& s1, const string& s2)
	{
		size_t i = 0;
		while (i < s1.size()
			&& i < s2.size()
			&& s1[i] == s2[i])
			++i;

		return i;
	}

	// sfen文字列("position"で渡されてくる文字列)を連結する。
	// sfen1 == "startpos" , moves = " 7g7f"の時に、
	// "startpos moves 7g7f"のように連結する。
	// 引数のmovesの文字列の先頭にはスペースが入っていること。
	string concat_sfen(const string&sfen1, const string& moves)
	{
		bool is_startpos = sfen1 == "startpos";
		return sfen1 + (is_startpos ? " moves" : "") + moves;
	}

	// ---------------------------------------
	//          コンテナ
	// ---------------------------------------

	// 固定長のvector
	// 
	// 通常のvectorはコンテナにcopy,moveが可能なことを要求するのだが、それをされたくない。
	// かと言って、shared_ptrで配列確保するのも嫌。vectorのように使えて欲しい。
	// 次善策としては、vector<shared_ptr<T>>なんだけど、ことあるごとに .get() とか -> とか書きたくない。
	template <typename T>
	class fixed_size_vector
	{
	public:

		size_t size() const       { return s; }
		void   resize(size_t n)   { release(); ptr = new T[n]; s = n; }
		void   release()          { if (ptr) { delete[] ptr; ptr=nullptr; } }

		T& operator[](size_t n)   { return ptr[n]; }

		typedef T* iterator;
		typedef const T* const_iterator;

		iterator       begin()       { return ptr    ; }
		iterator       end  ()       { return ptr + s; }
		const_iterator begin() const { return ptr    ; }
		const_iterator end  () const { return ptr + s; }

		~fixed_size_vector()    { release(); }

	private:
		T*     ptr = nullptr;
		size_t s   = 0;
	};

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

		// これcopyされてはかなわんので、copyとmoveを禁止しておく。
		ProcessNegotiator(const ProcessNegotiator& other)         = delete;
		ProcessNegotiator&& operator = (const ProcessNegotiator&) = delete;

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

		STOP,
		QUIT,
	};

	string to_string(USI_Message usi)
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
		Message(USI_Message message_, const string& command_)
			: message(message_) , command(command_)                                   {}
		Message(USI_Message message_, const string& command_, const string& position_sfen_)
			: message(message_) , command(command_) , position_sfen(position_sfen_)   {}

		// メッセージ種別。
		const USI_Message message;

		// パラメーター。
		// GUI側から送られてきた1行がそのまま入る。
		const string command;

		// 追加のパラメーター
		// GO , GO_PONDER に対しては、思考すべき局面のsfen文字列が入る。
		// (positionコマンドに付随している局面文字列。例 : "startpos moves 7g7f")
		const string position_sfen;

		// このクラスのメンバーを文字列化する
		string to_string() const
		{
			if (command.empty())
				return "Message[" + YaneuraouTheCluster::to_string(message) + "]";

			if (position_sfen.empty())
				return "Message[" + YaneuraouTheCluster::to_string(message) + " : " + command + "]";

			return "Message[" + YaneuraouTheCluster::to_string(message) + " : " + command + "] : " + position_sfen;
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
		QUIT,              // "quit"コマンド送信後。
	};

	// EngineNegotiatorStateを文字列化する。
	string to_string(EngineState state)
	{
		const string s[] = {
			"DISCONNECTED", "CONNECTED",
			"WAIT_USI", "WAIT_ISREADY", "WAIT_READYOK",
			"IDLE_IN_GAME", "GO", "PONDERING",
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

		// これcopyされてはかなわんので、copyとmoveを禁止しておく。
		EngineNegotiator(const EngineNegotiator& other)         = delete;
		EngineNegotiator&& operator = (const EngineNegotiator&) = delete;

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
				error_to_gui("fail to connect = " + path);
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
		// 結果はすぐに返る。親クラス(ClusterObserver)の送受信用スレッドから呼び出す。
		// send()とreceive()とは同時に呼び出されない。(親クラスの送受信用のスレッドは、送受信のために1つしかスレッドが走っていないため。)
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
				send_to_engine(message.command);
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
				// "go ponder"中に次のgoが来ることはありうる。
				stop_thinking();

				// エンジン側からbestmove来るまで次のgo送れないのでは…。いや、go ponderなら送れるのか…。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'go' should be sent when state is 'IDLE_IN_GAME'.");

				searching_sfen = message.position_sfen;
				send_to_engine("position " + searching_sfen);
				send_to_engine(message.command);

				state = EngineState::GO;
				break;

			case USI_Message::GO_PONDER:
				// "go ponder"中に次の"go ponder"が来ることはありうる。
				stop_thinking();

				// 本来、エンジン側からbestmove来るまで次のgo ponder送れないが、
				// ここでは、ignore_bestmoveをインクリメントしておき、この回数だけエンジン側からのbestmoveを
				// 無視することによってこれを実現する。
				if (   state != EngineState::IDLE_IN_GAME
					&& state != EngineState::GO_PONDER
					)
					EngineError("'go ponder' should be sent when state is 'IDLE_IN_GAME'.");

				searching_sfen = message.position_sfen;
				send_to_engine("position " + searching_sfen);
				send_to_engine("go ponder");

				state = EngineState::GO_PONDER;

				++ignore_bestmove;
				break;

			case USI_Message::PONDERHIT:
				// go ponder中以外にponderhitが送られてきた。
				if (state != EngineState::GO_PONDER)
					EngineError("'ponderhit' should be sent when state is 'GO_PONDER'.");
				else
					// 次のエンジンからの"bestmove"が送られてくるまでを無視する予定であったが、事情が変わった。
					--ignore_bestmove;

				// ここまでの思考ログを出力してやる必要がある。
				output_thinklog();

				send_to_engine("ponderhit " + message.command); // "ponderhit XXX"
				state = EngineState::GO;

				// 以降は、EngineStateがGOになっているのでエンジン側から送られてきた"info .."は、
				// 直接GUIに出力されるはず。

				break;

			case USI_Message::GAMEOVER:
				// 思考の停止
				stop_thinking();

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

			default:
				EngineError("illegal message from ClusterObserver : " + message.to_string());
				break;
			}
		}

		// [SYNC]
		// エンジンからメッセージを受信して、dispatchする。
		// このメソッドは親クラス(ClusterObserver)の送受信用スレッドから定期的に呼び出される。(呼び出さなければならない)
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

			return received;
		}

		// 思考中であったなら、思考を停止させる。
		void stop_thinking()
		{
			if (state == EngineState::GO_PONDER)
			{
				// 前の思考("go ponder"によるもの)を停止させる必要がある。
				send_to_engine("stop");
				state = EngineState::IDLE_IN_GAME;
			}
			else if (state == EngineState::GO)
			{
				// 警告を出しておく。
				error_to_gui("illegal state in stop_thinking() , state = " + to_string(state));

				send_to_engine("stop");
				// この場合、bestmoveを待ってから状態を変更してやる必要があるのだが…。
				// そもそもで言うと "go"して stopが来る前に gameoverが来ているのがおかしいわけだが。
				state = EngineState::IDLE_IN_GAME;
			}
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

		// 現在、"go","go ponder"によって探索中の局面。
		// ただし、"go"に対してエンジンが"bestmove"を返したあとも
		// その探索していた局面のsfenを、このメソッドで取得できる。
		string get_searching_sfen() const { return searching_sfen; }

		// GO状態なのか？
		bool is_state_go() const { return state == EngineState::GO; }

		// GO_PONDER状態なのか？
		bool is_state_go_ponder() const { return state == EngineState::GO_PONDER; }

		// [main thread][receive thread]
		// エンジンが対局中のモードに入っているのか？
		bool is_idle_in_game()       const { return state == EngineState::IDLE_IN_GAME; }

		// [main thread][receive thread]
		// 現在のstateが"isready"の送信待ちの状態か？
		bool does_wait_isready() const { return state == EngineState::WAIT_ISREADY; }

		// エンジン側から受け取った"bestmove XX ponder YY"を返す。
		// 一度このメソッドを呼び出すと、次以降は(エンジン側からさらに"bestmove XX ponder YY"を受信するまで)空の文字列が返る。
		// つまりこれは、size = 1 の PC-queueとみなしている。
		string get_bestmove() {
			auto result = bestmove_string;
			bestmove_string.clear();
			return result;
		}

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
			error_to_gui("[" +  std::to_string(engine_id) + "] " + message);
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

			// GUIに転送するのか？のフラグ
			bool send_gui = false;

			if (token == "info")
			{
				// ponder中であればその間に送られてきたメッセージは全部ログに積んでおく。
				// (ponderhitが送られてきた時に、そこまでのlogをGUIに出力しなければならないため)
				if (state == EngineState::GO_PONDER)
				{
					if (ignore_bestmove == 0)
						DebugMessage(": Warning! : Illegal state , state = " + to_string(state) + " , ignore_bestmove == 0");
					else if (ignore_bestmove == 1)
						think_log.push_back(message);
					else
						;
						// ignore_bestmove >= 2なら、どうせいま受信したメッセージは捨てることになるのでthink_logに積まない。
				}
				// "go"("go ponder"ではない)で思考させているなら、そのままGUIに転送。
				else if (state == EngineState::GO)
					send_gui = true;
				// "usiok", "readyok" 待ちの時は、engine id == 0のメッセージだけをGUIに転送。
				else if (state == EngineState::WAIT_USIOK
					  || state == EngineState::WAIT_READYOK
					)
					send_gui = get_engine_id() == 0;
				else
					// usiok/readyok待ちと go ponder , go 以外のタイミングでエンジン側からinfo stringでメッセージが来るのおかしいのでは…。
					DebugMessage(": Warning! : Illegal info , state = " + to_string(state) + " , message = " + message);

				// "Error"という文字列が含まれていたなら(おそらく"info string Error : "みたいな形)、
				// 即座に何も考えずにGUIにそれを投げる。
				// "info string [engine id] : xxx"の形にしたほうがいいかな？
				if (!send_gui && StringExtension::Contains(message, "Error"))
					send_to_gui("info string [" + std::to_string(engine_id) + "]> " + message);
			}
			// "bestmove XX"を受信した。
			else if (token == "bestmove")
			{
				if (ignore_bestmove > 0)
				{
					--ignore_bestmove;

					// bestmoveまでは無視して良かったことが確定したのでこの時点でクリアしてしまう。
					think_log.clear();

				} else {

					// 無視したらあかんやつなのでこのままGUIに投げる。
					send_gui = true;

					// 思考は停止している。
					change_state(EngineState::IDLE_IN_GAME);

					// 探索中の局面のsfenを示す変数をクリア。
					//searching_sfen = string();
					// →　これは空にしては駄目。この情報使う。

					// これを設定しておけば親クラスが検知してくれる。
					bestmove_string = message;
				}
			}
			else if (token == "usiok")
				// "usiok"は全部のエンジンが WAIT_ISREADYになった時に親クラス(Observer)がGUIに送るので状態の変更だけ。
				change_state(EngineState::WAIT_ISREADY);
			else if (token == "readyok")
				// → "readyok"は全部のエンジンが IDLE_IN_GAMEになった時に親クラス(Observer)がGUIに送るので状態の変更だけ。
				change_state(EngineState::IDLE_IN_GAME);
			// "id"はエンジン起動時にしか来ないはずだが？
			else if (token == "id")
			{
				StateAssert(EngineState::WAIT_USIOK);

				send_gui = engine_id == 0;

				// Warning
				DebugMessage(": Warning! : Illegal Message , state = " + to_string(state)
					+ " , message = " + message);
			}

			// GUIに転送しないといけないメッセージであった。
			if (send_gui)
				send_to_gui(message);

		}

		// そこまでの思考ログを出力する。
		void output_thinklog()
		{
			// "GO_PONDER"が何重にも送られてきている。
			// まだ直前のGO_PONDERのログがエンジン側から送られてきていない。
			if (ignore_bestmove >= 1)
				return ;

			// ここまでの思考ログを吐き出してやる。
			for(auto& log : think_log)
				send_to_gui(log);
			think_log.clear();
		}

		// EngineStateの状態がsではない時に警告をGUIに出力する。
		void StateAssert(EngineState s)
		{
			if (state != s)
				error_to_gui("illegal state : state = " + to_string(s));
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

		// 探索中の局面
		// state == GO or GO_PONDER において探索中の局面。
		string searching_sfen;

		// この回数だけエンジン側から送られてきたbestmoveを無視する。
		// 連続して"go ponder"を送った時など、bestmoveを受信する前に次の"go ponder"を送信することがあるので
		// その時に、前の"go ponder"に対応するbestmoveは無視しないといけないため。
		atomic<int> ignore_bestmove;

		// "go ponder"時にエンジン側から送られてきた思考ログ。
		// そのあと、"ponderhit"が送られてきたら、その時点までの思考ログをGUIにそこまでのログを出力するために必要。
		vector<string> think_log;

		// エンジン側から返ってきた、"bestmove XXX ponder YYY" みたいな文字列。
		string bestmove_string;
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
			// エンジン生成してからスレッドを開始しないと、エンジンが空で困る。
			connect();

			// スレッドを開始する。
			options       = options_; 
			worker_thread = std::thread([&](){ worker(); });
		}

		~ClusterObserver()
		{
			send_wait(USI_Message::QUIT);
			worker_thread.join();
		}

		// [main thread]
		// 起動後に一度だけ呼び出すべし。
		void connect()
		{
			vector<string> lines;

			// エンジンリストが書かれているファイル
			string engine_list_path = "engines/engine_list.txt";

			if (SystemIO::ReadAllLines(engine_list_path, lines, true).is_not_ok())
			{
				error_to_gui("engine list file not found. path = " + engine_list_path);
				Tools::exit();
			}

			size_t engine_num = 0;
			for (const auto& line : lines)
				if (!line.empty())
					engine_num++;

			engines.resize(engine_num);
			size_t engine_id = 0;

			// それぞれのengineを起動する。
			for (const auto& line : lines)
			{
				if (line.empty())
					continue;

				// engineの実行path。engines/配下にあるものとする。
				string engine_path = line;

				// エンジンを起動する。
				engines[engine_id].connect(engine_path , engine_id);
				engine_id++;
			}

			// すべてのエンジンの起動完了を待つ設定なら起動を待機する。
			// →　現状、強制的にこのモードで良いと思う。
			if (/* options.wait_all_engines_wakeup */ true)
				wait_all_engines_wakeup();
		}

		// [ASYNC] 通信スレッドで受け取ったメッセージをこのClusterObserverに伝える。
		//    waitとついているほうのメソッドは送信し、処理の完了を待機する。
		void send(USI_Message usi                     )       { send(Message(usi            )); }
		void send(USI_Message usi, const string& param)       { send(Message(usi, param     )); }
		void send_wait(USI_Message& usi)                      { send_wait(Message(usi       )); }
		void send_wait(USI_Message& usi, const string& param) { send_wait(Message(usi, param)); }

		// [ASYNC] Messageを解釈してエンジンに送信する。
		void send(Message message)
		{
			// Observerからengineに対するメッセージ
			DebugMessageCommon("Cluster to ClusterObserver : " + message.to_string());

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

						// 現在、相手が初期局面("startpos")について思考しているものとする。
						searching_sfen2 = "startpos";
						our_searching2 = false;

						// 対局中である。
						is_in_game = true;

						// 対局中であれば自動的にponderingが始まるはず。

						break;

					case USI_Message::GAMEOVER:

						// GAMEOVERが来れば、各エンジンは自動的に停止するようになっている。
						broadcast(message);

						// 対局中ではない。
						is_in_game = false;

						// 対局中でなければ自動的にエンジンは停止するはず。

						break;

					case USI_Message::POSITION:
						// 最後に受け取った"position"コマンドの内容。
						// 次の"go"コマンドの時に、この局面について投げてやる。
						position_string = message.command;
						break;

					case USI_Message::GO:
						// "go"コマンド。
						handle_go_cmd(message);
						break;

					case USI_Message::QUIT:
						broadcast(message);

						// エンジン停止させて、それを待機する必要はある。
						quit = true;
						break;

					default:
						// ハンドラが書かれていない、送られてくること自体が想定されていないメッセージ。
						error_to_gui("illegal message : " + message.to_string());
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
		// 実際は、最初に起動させたエンジンの数と一致しないなら、終了すべきだと思うが、
		// 1つでも生きてたら頑張って凌ぐコードにする。
		void engine_check()
		{
			size_t num = get_number_of_live_engines();
			if (num == 0)
			{
				send_to_gui("info string All engines are terminated.");
				Tools::exit();
			}

			if (is_in_game)
			{
				// 対局中ならば、遊んでいるエンジンがないかのチェック

				for(auto& engine : engines)
				{
					auto bestmove = engine.get_bestmove();
					if (bestmove.empty())
						continue;

					// 何か積まれていたので、これをparseする。
					auto searching_sfen = engine.get_searching_sfen();
					istringstream is(bestmove);
					string token;
					string best_move;
					string ponder_move;
					while (is >> token)
					{
						if (token == "bestmove")
							is >> best_move;
						else if (token == "ponder")
							is >> ponder_move;
					}
					// bestmoveで進めた局面を対局局面とする。
					if (is_ok(USI::to_move16(best_move)))
					{
						searching_sfen2 = concat_sfen(searching_sfen , " " + best_move);
						our_searching2  = false;

						// さらにponderの指し手が有効手なのであるなら、ここを第一ponderとすべき。
						if (is_ok(USI::to_move16(ponder_move)))
						{
							// 先頭にわざとスペース入れておく。
							// ※　そうしてあったほうがdlエンジンが返してくる候補手と比較する時に便利。
							engine_ponder = " " + ponder_move;
							DebugMessageCommon("engine's ponder : " + searching_sfen2 + engine_ponder);
						}
					}
				}

				// 想定している局面と実際の対局局面が異なる。
				if (searching_sfen1 != searching_sfen2)
				{
					if (our_searching2)
						start_go();
					else
						// go ponderで局面を割当て。
						start_pondering();
				} else {

					// 探索局面は合致しているが、自分手番なのに"go"しているエンジンが見当たらないパターン。
					// (たぶん途中でエンジンが落ちた。)
					// この時、再度 go してやる必要がある。
					
					if (our_searching2)
					{
						bool found = false;
						for(auto& engine : engines)
						{
							if (engine.is_state_go())
							{
								if (engine.get_searching_sfen() != searching_sfen2)
									error_to_gui("engine.get_searching_sfen() != searching_sfen2");

								found = true;
								break;
							}
						}
						if (!found)
							start_go();
					}

					// 局面が合致しているが遊んでいるエンジンがある。
					// →　dl部が指し手列挙できなかった可能性があるのでこのパターンは気にしないことにする。
				}
			} else {
				// 対局中でないなら、動いているエンジンがないかのチェックして動いているエンジンを停止させなければならない。
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

		// 親から送られてきた"position"～"go"コマンドに対して処理する。
		void handle_go_cmd(const Message& message)
		{
			searching_sfen2 = strip_command(position_string);
			our_searching2  = true;
			go_string       = message.command;

			if (searching_sfen2.empty())
			{
				error_to_gui("Illegal position command : " + position_string);
				return ;
			}

			start_go();
		}

		// searching_sfen2の"go"での探索を開始する。
		void start_go()
		{
			// ここ、局面に関して何らかのassert追加するかも。

			// 現在、与えられた局面についてGO_PONDERで思考しているエンジンがあるか？
			// あるなら、そのエンジンに対して"ponderhit"を送信して、残りのエンジンに対しては
			// 次に思考すべき局面の選出をした上で、それを残りのエンジンにGO_PONDERで思考させる。

			bool found = false;
			for(auto& engine: engines)
				if (engine.is_state_go_ponder() && engine.get_searching_sfen() == searching_sfen2)
				{
					// 見つかった。
					found = true;

					// PONDERHITの時は、commandとして"go XXX"のXXXの部分を送ることになっている。
					DebugMessageCommon("ponderhit [" + std::to_string(engine.get_engine_id()) + "] : " + searching_sfen2);
					engine.send(Message(USI_Message::PONDERHIT, strip_command(go_string)));
					break;
				}

			// 見つからなかった。
			if (!found)
			{
				// どうしようもない。この探索局面に近い局面を探索しているエンジンもないので、エンジン 0 に探索させておく。
				DebugMessageCommon("go [" + std::to_string(engines[0].get_engine_id()) + "] : " + searching_sfen2);
				engines[0].send(Message(USI_Message::GO, go_string , searching_sfen2));
			}

			// 他のエンジンは、それ以外の局面を"go ponder"しておく。
			start_pondering();
		}

		// 各エンジンのponderを開始する。
		void start_pondering()
		{
			// 探索中の局面は定まっているか？
			if (searching_sfen2.empty())
				return ; // ない

			// 我々が探索中の局面があるなら、その2手、4手、のように偶数手先の局面について局面を選出し、ponderする。
			// さもなくば現在相手が思考中の局面に対して、1手、3手のように奇数手先の局面について選出し、ponderする。

			search_for_ponder(searching_sfen2, our_searching2);

			// デバッグ用に逆側も出力してみる。
			//DebugMessageCommon("---");
			//search_for_ponder(searching_sfen, !our_searching);

			// ponderの中心局面の更新
			searching_sfen1 = searching_sfen2;
			our_searching1  = our_searching1;
		}

		// ponderする局面の選出。
		// search_sfen : この局面から探索してくれる。
		// same_color  : search_sfenと同じ手番の局面をponderで思考するのか？
		void search_for_ponder(string search_sfen,  bool same_color)
		{
			// --- 空いてるエンジンの数だけ局面を選出する。

			// 空いていたエンジンの数
			size_t num = 0;

			vector<bool> engine_empty;

			for(size_t i = 0 ; i < engines.size() ; ++i)
			{
				// 現在ponderしているか、何もしていないエンジンは空きエンジンとみなす。
				bool is_empty = engines[i].is_state_go_ponder() || engines[i].is_idle_in_game();
				engine_empty.push_back(is_empty);

				if (is_empty)
					++num;
			}

			// なぜか空いているエンジンがない…。なんで？
			if (num == 0)
			{
				error_to_gui("search_for_ponder() : No empty engine.");
				return ;
			}

			dlshogi::SfenNodeList snlist;

			dl_search(num, snlist, search_sfen, same_color);

			// debug用に出力してみる。

			bool found = false;
			for(size_t i = 0; i < snlist.size() ; ++i)
			{
				string sfen = snlist[i].sfen;
				if (engine_ponder == sfen)
				{
					DebugMessageCommon("sfen for pondering (" + std::to_string(i) + ") (engine's ponder) : " + sfen + "(" + std::to_string(snlist[i].nodes) + ")");
					found = true;
				}
				else
					DebugMessageCommon("sfen for pondering (" + std::to_string(i) + ") : " + sfen + "(" + std::to_string(snlist[i].nodes) + ")");
			}

			// エンジン側がponderで指定してきた局面が見つからからなかった。
			if (!found && !engine_ponder.empty())
			{
				// 先頭に追加。
				snlist.insert(snlist.begin(), dlshogi::SfenNode(engine_ponder,99999));

				// 末尾要素を一つremove
				snlist.resize(snlist.size() - 1);
			}

			// 局面が求まったので各エンジンに対して"go ponder"で思考させる。

			for(size_t i = 0 ; i < snlist.size() ; ++i)
			{
				string sfen = concat_sfen(search_sfen , snlist[i].sfen);

				// 一番近くを探索していたエンジンに割り当てる
				// すなわち探索中のsfen文字列が一番近いものに割り当てると良い。
				size_t t = numeric_limits<size_t>::max();
				size_t max_match_length = 0;
				for(size_t j = 0; j < engines.size() ; ++j)
				{
					if (!engine_empty[j])
						continue;
					 
					auto& engine = engines[j];
					auto& sfen2  = engine.get_searching_sfen();

					// ドンピシャでこれいま探索しとるで…。go ponderしなおす必要すらない。
					if (sfen == sfen2)
					{
						engine_empty[j] = true;
						goto Next;
					}

					// なるべく長い文字列が一致したほど、近い局面を探索していると言えると思うので、そのエンジンを使い回す。
					// また、全く一致しなかった場合、0が返るが、それだとmax_match_lengthの初期値と同じなので + 1足してから比較する。
					// (max_match_lengthは unsingedなので -1 のような負の値が取れないため)
					size_t match_length = get_match_length(sfen, sfen2) + 1;
					if (match_length > max_match_length)
					{
						max_match_length = match_length;
						t = j;
					}
				}

				// 空きがなかった。おかしいなぁ…。
				if (t == numeric_limits<size_t>::max() )
				{
					error_to_gui("no empty engine.");
					break;
				}

				{
					auto& engine = engines[t];
					DebugMessageCommon("go ponder [" + std::to_string(engine.get_engine_id()) + "] : " + sfen);
					engines[t].send(Message(USI_Message::GO_PONDER, string() , sfen));
					engine_empty[t] = false;
				}

			Next:;
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
		fixed_size_vector<EngineNegotiator> engines;

		// Supervisorから送られてくるMessageのqueue
		Concurrent::ConcurrentQueue<Message> queue;

		// 現在エンジンに対して行っているコマンド
		// これがNONEになるまで次のメッセージは送信できない。
		USI_Message usi = USI_Message::NONE;

		// 対局中であるかのフラグ
		// "usinewgame"を受信してから"gameover"を受信するまで。
		bool is_in_game = false;

		// workerスレッド
		std::thread worker_thread;

		// Messageを処理した個数
		atomic<u64> done_counter = 0;

		// Messageをsendした回数
		atomic<u64> send_counter = 0;

		// 最後に受け取った"position"コマンド。次に"go"がやってきた時にこの局面に対して思考させる。
		// "position"を含む。
		string position_string;

		// 最後に受け取った"go"コマンド。"go"を含む。
		string go_string;

		// 現在go ponderで思考している中心局面のsfen。(startpos moves XX XX..の形式)
		string searching_sfen1;  // 自分か相手がこの局面について思考しているものとする。(ponderする時の中心となる局面)
		bool   our_searching1;     // search_sfenを探索しているのは自分ならばtrue。相手ならばfalse。

		// 現在の本当の局面
		// これは searching_sfen1 == searching_sfen2
		// であるのが普通なのだが、goしていたエンジンからbestmoveが返ってきた時に、
		// ↓だけが更新されて、それを監視スレッド検知して、各エンジンに思考させなおすことで↑に反映される。
		string searching_sfen2;
		bool   our_searching2;

		// エンジン側が"bestmove XX ponder YY"と返してきた時のYY。先頭にスペースわざと入れてある。
		// ※　そうしてあったほうがdlエンジンが返してくる候補手と比較する時に便利。
		string engine_ponder;
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
			// コンストラクタで全エンジンが起動する。
			ClusterObserver observer(options);

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
				else if (token == "position")
					observer.send(USI_Message::POSITION, cmd);
				else if (token == "go")
					observer.send(USI_Message::GO      , cmd);
				else if (token == "stop")
					observer.send(USI_Message::STOP    , cmd);
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
					error_to_gui("Unknown Command : " + token);
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
