#include "../../config.h"
#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ProcessNegotiator.h"
#include "../../misc.h"

// Windows環境である。
#if defined(_WIN32)

// std::numeric_limits::max()みたいなのを壊さないように。
//#define NOMINMAX
// → redefineになる環境があるな。std::max()を(std::max)()にように書いて回避することにする。
#include <Windows.h>

using namespace std;

struct ProcessNegotiatorImpl : public IProcessNegotiator
{
	// あとで何とかするかも。(しないかも)
	void DebugMessageCommon(const std::string& message){}

	// 子プロセスの実行
	// workingDirectory : エンジンを実行する時の作業ディレクトリ("engines/"相対で指定)
	// app_path         : エンジンの実行ファイルのpath (.batファイルでも可) workingDirectory相対で指定。
	virtual void connect(const std::string& workingDirectory , const std::string& app_path)
	{
		disconnect();
		terminated = false;
		// →　子プロセス接続前に必ず terminated == falseにしておく。

		ZeroMemory(&pi, sizeof(pi));
		ZeroMemory(&si, sizeof(si));

		si.cb = sizeof(si);
		si.hStdInput = child_std_in_read;
		si.hStdOutput = child_std_out_write;
		si.dwFlags |= STARTF_USESTDHANDLES;

		// Create the child process

		DebugMessageCommon("workingDirectory = " + workingDirectory + " , " + app_path);

		string app_path2 = Path::Combine(workingDirectory, app_path);

		// 注意 : to_wstring()、encodingの問題があって日本語混じってると駄目。

		bool success = ::CreateProcess(
			NULL,                                         // ApplicationName
			(LPWSTR)to_wstring(app_path2).c_str(),        // CmdLine
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
			// ファイルが存在しなかったのか何か。
			terminated = true;
			engine_path = std::string();
		}
	}

	// 子プロセスへの接続を切断する。
	virtual void disconnect()
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
	// 子プロセスと切断されていることがわかったら、以降is_terminated()==trueを返すようになる。
	// (この関数のなかで切断されているかをチェックしている。)
	virtual std::string receive()
	{
		// 子プロセスが終了しているなら何もできない。
		if (terminated)
			return std::string();

		auto result = receive_next();
		if (!result.empty())
			return result;

		DWORD dwExitCode;
		::GetExitCodeProcess(pi.hProcess, &dwExitCode);
		if (dwExitCode != STILL_ACTIVE)
		{
			// 切断されているので 空の文字列を返す。
			terminated = true;
			return std::string();
		}

		DWORD dwRead, dwReadTotal, dwLeft;

		// 確保している読み書きの1回のread用の buffer size
		static const size_t BUF_SIZE = 1024;
		CHAR chBuf[BUF_SIZE];

		// ReadFileは同期的に使いたいが、しかしデータがないときにブロックされるのは困るので
		// pipeにデータがあるのかどうかを調べてからReadFile()する。

		BOOL success = ::PeekNamedPipe(
			child_std_out_read, // [in]  handle of named pipe
			NULL,               // [out] buffer                → NULLにすれば読み込まれない
			0,                  // [in]  buffer size           → 何バイトあるか知りたいだけだから0で十分。
			&dwRead,            // [out] bytes read
			&dwReadTotal,       // [out] total bytes avail
			&dwLeft             // [out] bytes left this message
		);

		// ReadFileしている間に増える可能性があるので、totalを上回る量のreadが出来てしまうかも知れない。
		// そこで、符号型に代入しておき、これが負になるまで読み込む。
		s64 total = (s64)dwReadTotal;

		if (success)
		{
			while (total > 0)
			{
				// bufferサイズは1文字少なく申告して読み込める限り読み込んで、終端に'\0'を付与してstring化する。
				success = ::ReadFile(child_std_out_read, chBuf, BUF_SIZE - 1, &dwRead, NULL);

				if (success && dwRead != 0)
				{
					chBuf[dwRead] = '\0'; // 終端マークを書いてstringに連結する。
					read_buffer  += chBuf;
					total        -= dwRead;
				}
			}
		}
		return receive_next();
	}

	// 接続されている子プロセス(の標準入力)に1行送る。改行は自動的に付与される。
	// sendでは、terminatedの判定はしていない。
	virtual bool send(const std::string& message)
	{
		// すでに切断されているので送信できない。
		if (terminated)
			return false;

		std::string s = message + "\r\n"; // 改行コードの付与
		DWORD dwWritten;
		BOOL success = ::WriteFile(child_std_in_write, s.c_str(), DWORD(s.length()), &dwWritten, NULL);

		return success;
	}

	// プロセスの終了判定
	virtual bool is_terminated() const { return terminated; }

	// エンジンの実行path
	// これはconnectの直後に設定され、そのあとは変更されない。connect以降でしか
	// このプロパティにはアクセスしないので同期は問題とならない。
	virtual std::string get_engine_path() const { return engine_path; }

	ProcessNegotiatorImpl() { init(); }
	virtual ~ProcessNegotiatorImpl() { disconnect(); }

	// これcopyされてはかなわんので、copyとmoveを禁止しておく。
	ProcessNegotiatorImpl(const ProcessNegotiatorImpl& other)         = delete;
	ProcessNegotiatorImpl&& operator = (const ProcessNegotiatorImpl&) = delete;

protected:

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
		auto ERROR_MES = [&](const std::string& mes) {
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

	std::string receive_next()
	{
		// read_bufferから改行までを切り出す
		// stringで処理しているのでたくさん溜まっているとsubstr()が遅いことになるが、
		// これをうまく書くのは難しいのでこれで我慢する。

		auto it = read_buffer.find("\n");
		if (it == std::string::npos)
			return std::string();
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
	std::wstring to_wstring(const std::string& src)
	{
		size_t ret;
		wchar_t *wcs = new wchar_t[src.length() + 1];
		::mbstowcs_s(&ret, wcs, src.length() + 1, src.c_str(), _TRUNCATE);
		std::wstring result = wcs;
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
	std::atomic<bool> terminated;

	// 受信バッファ
	std::string read_buffer;

	// プロセスのpath
	std::string engine_path;
};

#else // defined(_WIN32)

// Linux環境である。Linux用の実装を頑張って書いた。

#include <unistd.h> // pid_t
#include <fcntl.h>  // open
#include <sys/wait.h> // waitpid

using namespace std;

// Linuxのpipeのwrapper

enum PIPE_TYPE : int
{
    READ  = 0,
    WRITE = 1,
};

class Pipe
{
public:
    // pipeをopenする。
    // close_pipe()を呼び出す時までopenされたままになる。
    // pipeはPIPE_TYPE::READとPIPE_TYPE::WRITEの２つの方向がある。
    void create_pipe()
    {
        if (pipe(handles) < 0)
        {
            // pipe生成に失敗。(そんなんある？)
            opened[PIPE_TYPE::READ] = opened[PIPE_TYPE::WRITE] = false;
        } else {
            opened[PIPE_TYPE::READ] = opened[PIPE_TYPE::WRITE] =  true;
        }
    }

    // pipeを閉じる。
    void close_pipe(PIPE_TYPE t)
    {
        // openしている時だけこのpipeを閉じる。
        if (opened[t])
        {
            close(handles[t]);
            opened[t] = false;
        }
    }

    // すべてのpipeを閉じる。
    void close_all_pipes()
    {
        close(PIPE_TYPE:: READ);
        close(PIPE_TYPE::WRITE);
    }

    // Pipeがopenされているか。
    bool is_opened(PIPE_TYPE t) const
    {
        return opened[t];
    }

    // non blockingフラグを設定する。
    void set_non_blocking(PIPE_TYPE t)
    {
        if (is_opened(t))
        {
            // handlesの設定を読み込み、non blockingフラグを付与して書き戻す。
            int retval = fcntl( handles[t], F_SETFL, fcntl(handles[t], F_GETFL) | O_NONBLOCK);
        }
    }

    // PIPEを標準入力/出力に割り当てる。
    // f = STDIN_FILENO or STDOUT_FILENO
    void duplicate(PIPE_TYPE t , int f)
    {
        dup2(handles[t] , f);
    }

    // PIPEに1行を書き込む。
    bool write(std::string mes)
    {
        ssize_t result = ::write(handles[PIPE_TYPE::WRITE], mes.c_str() , mes.size());
		return result != (ssize_t)-1;
    }

    // PIPEから1行を読み出す。何も読みだせなければ空のstringが返る。
    std::string read()
    {
        char buf[1024];
        std::string s;
        while (true)
        {
            ssize_t read_bytes = ::read(handles[PIPE_TYPE::READ], buf, sizeof(buf)-1 );
            if (read_bytes == (ssize_t)-1 /* is Empty */|| read_bytes == 0 /* is EOF */)
                return s;
            // 末尾に'\0'を付与して文字連結してしまう。
            buf[read_bytes] = '\0';
            s += buf;
        }
    }

    Pipe() { opened[PIPE_TYPE::READ] = opened[PIPE_TYPE::WRITE] = false; }
    ~Pipe() { close_all_pipes(); }

protected:
    // handle of pipe
    int  handles[2];
    
    // pipeをopenしているか？
    // create_pipeに成功した場合、
    // opened[READ] = opened[WRITE] = trueとなる。
    bool opened[2];
};

class ProcessNegotiatorImpl : public IProcessNegotiator
{
public:
	// あとで何とかするかも。(しないかも)
	void DebugMessageCommon(const std::string& message){}

	// workingDirectory : エンジンの作業フォルダ
    // app_path         : 起動するエンジンのpath。同じフォルダにあるならLinuxの場合、"./YO_engine.out"のように"./"をつけてやる必要があるが、
    //                    同じフォルダにエンジンを配置しないと思うので、そういう仕様だということにしておく。
	virtual void connect(const std::string& workingDirectory , const std::string& app_path)
    {
        p2c.create_pipe();
        c2p.create_pipe();
        terminated  = false;

		DebugMessageCommon("workingDirectory = " + workingDirectory + " , " + app_path);
		string app_path2 = Path::Combine(workingDirectory, app_path);

		engine_path = app_path2;

        pid = fork();
        if (pid == 0) {
            // 子プロセスで実行される

#if 0
            // vector<string> args で起動させたい場合。

            // 起動する時の引数をこねこねする。
            char** arg = NULL;
            arg = new char*[args.size() + 1];
            for( size_t i = 0 ; i < args.size(); i++ ) {
                arg[i] = (char*) args[i].c_str();
            }
            arg[ args.size() ] = NULL;
#endif

            // エンジンの実行ファイルに起動パラメーターが存在しないはずなので、何も考えずに渡してしまう。
            char** arg = NULL;
            arg = new char*[1 + 1];
            arg[0] = (char*) engine_path.c_str();
            arg[1] = NULL;

            // 子プロセスの場合は、親→子への書き込みはありえないのでcloseする
            p2c.close_pipe(PIPE_TYPE::WRITE);
            
            // 子プロセスの場合は、子→親の読み込みはありえないのでcloseする
            c2p.close_pipe(PIPE_TYPE::READ);
            
            // 親→子への出力を標準入力として割り当て
            p2c.duplicate(PIPE_TYPE::READ  , STDIN_FILENO);

            // 子→親への入力を標準出力に割り当て
            c2p.duplicate(PIPE_TYPE::WRITE , STDOUT_FILENO);

            // 子へのwriteをnon blockingにする。
            p2c.set_non_blocking(PIPE_TYPE::WRITE);

            // 割り当てたファイルディスクリプタは閉じる
            p2c.close_pipe(PIPE_TYPE:: READ);
            c2p.close_pipe(PIPE_TYPE::WRITE);

            // フォルダをworking directoryに変更しておく。
            chdir(workingDirectory.c_str()); 

            // exec
            int rc = execvp( arg[0], (char*const*) arg );    
            // std::cerr << "failed to execute the command. rc : " << rc << std::endl;

            exit(-1);
        }

        // 親プロセスなので、このまま返る。
        if (pid < 0)
        {
            // forkに失敗している。
            terminated = true;
            return;
        }

        // if (pid > 0)

        // std::cout << "pid: " << pid << std::endl;
        int status = 0;

        p2c.close_pipe(PIPE_TYPE::READ );
        c2p.close_pipe(PIPE_TYPE::WRITE);

        // 読み出しをnon blockingに
        c2p.set_non_blocking(PIPE_TYPE::READ);
    }

    // 子プロセスを強制的に終了させる。
    // 本来は"quit"コマンドで終了させるべき。
    virtual void disconnect()
    {
        if (!terminated && pid)
        {
            kill(pid, SIGTERM);

            // 本来、プロセスの終了を待機すべきではあるが、
            // 次の子プロセスをちゃっちゃとdisconnectしていきたいのでこの待機いらないや…。
            // int ret = waitpid(pid, &status, 0);

            // もう終了した扱いで良いや。
            terminated = true;
        }
    }

    // 子プロセスに送信する。
    // ※　これはnon blocking method。
    virtual bool send(const std::string& mes)
    {
        if (!terminated)
            p2c.write(mes + "\r\n");
		return true;
    }

    // 子プロセスから受信する。
    // メッセージがないときは空の文字列が返る。
    // 改行は含まない。
    // ※　これはnon blocking method。
    virtual std::string receive()
    {
        if (terminated)
            return std::string();

        // バッファのなかにすでに1行分あるのか？
		auto result = receive_next();
		if (!result.empty())
			return result;

        // child processが生きているかを調べるには、waitpidにWNOHANG(non blockingにするoption)をつければわかる。
        int status;
        pid_t pid2 = waitpid(pid, &status, WNOHANG);
        // result is  0   : still alive
        // result is -1   : waitpid error
        // result is else : exited
        if (pid2 != 0) {
            terminated = true;
            return string();
        }

        read_buffer += c2p.read();
        return receive_next();
    }

    // 子プロセスが終了したかの判定
    // フラグは、receive()のタイミングで更新される。
    virtual bool is_terminated() const
    {
        return terminated;
    }

	// エンジンの実行path
	// これはconnectの直後に設定され、そのあとは変更されない。connect以降でしか
	// このプロパティにはアクセスしないので同期は問題とならない。
	virtual std::string get_engine_path() const { return engine_path; }

    ProcessNegotiatorImpl() : pid(0) , terminated(false) {}
    ~ProcessNegotiatorImpl() { disconnect(); }

	// これcopyされてはかなわんので、copyとmoveを禁止しておく。
	ProcessNegotiatorImpl(const ProcessNegotiatorImpl& other)         = delete;
	ProcessNegotiatorImpl&& operator = (const ProcessNegotiatorImpl&) = delete;

protected:
	std::string receive_next()
	{
		// read_bufferから改行までを切り出す
		// stringで処理しているのでたくさん溜まっているとsubstr()が遅いことになるが、
		// これをうまく書くのは難しいのでこれで我慢する。

		auto it = read_buffer.find("\n");
		if (it == std::string::npos)
			return std::string();
		// 切り出したいのは"\n"の手前まで(改行コード不要)、このあと"\n"は捨てたいので
		// it+1から最後までが次回まわし。
		auto result = read_buffer.substr(0, it);
		read_buffer = read_buffer.substr(it + 1, read_buffer.size() - it);
		// "\r\n"かも知れないので"\r"も除去。
		if (result.size() && result[result.size() - 1] == '\r')
			result = result.substr(0, result.size() - 1);

		return result;
	}
    // 受信バッファ
    std::string read_buffer;

    // エンジン起動フォルダ
    std::string engine_path;

    // 子プロセスのprocess id
    pid_t pid;

    // 起動したprocessが終了したのか？
    bool terminated;

    Pipe p2c; // pipe parent to child
    Pipe c2p; // pipe child  to parent
};


#endif // defined(_WIN32)

ProcessNegotiator::ProcessNegotiator()
{
	ptr = std::make_unique<ProcessNegotiatorImpl>();
}

#endif //defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE)) && defined(_WIN32)
