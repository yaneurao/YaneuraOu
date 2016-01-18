#include "../../shogi.h"
#ifdef LOCAL_GAME_SERVER

#include "../../extra/all.h"

#include <windows.h>

// 子プロセスを実行して、子プロセスの標準入出力をリダイレクトするのをお手伝いするクラス。
struct ProcessNegotiator
{
  ProcessNegotiator() { init(); }

  // 子プロセスの実行
  void run(wstring app_path)
  {
    ZeroMemory(&pi, sizeof(pi));
    ZeroMemory(&si, sizeof(si));

    si.cb = sizeof(si);
    //si.hStdInput = in;
    //si.hStdOutput = out;
    si.dwFlags |= STARTF_USESTDHANDLES;

    // Create the child process

    BOOL success = CreateProcess(app_path.c_str(), // ApplicationName
      NULL, // CmdLine
      NULL, // security attributes
      NULL, // primary thread security attributes
      TRUE, // handles are inherited
      0,    // creation flags
      NULL, // use parent's environment
      NULL, // use parent's current directory
      &si,  // STARTUPINFO pointer
      &pi   // receives PROCESS_INFOMATION
      );

    if (!success)
      sync_cout << "CreateProcessに失敗" << endl;
  }

  // 長手数になるかも知れないので…。
  static const int BUF_SIZE = 4096;

  string read()
  {
    DWORD dwRead;
    CHAR chBuf[BUF_SIZE];

    BOOL success = ReadFile(child_std_out_read, chBuf, BUF_SIZE-1, &dwRead, NULL);

    if (success && dwRead != 0)
    {
      chBuf[dwRead] = '\0'; // 終端マークを書いて文字列化する。
      return string(chBuf);
    }
    return string();
  }

  bool write(string str)
  {
    DWORD dwWritten;
    BOOL success = WriteFile(child_std_in_write, str.c_str(), DWORD(str.length() + 1) , &dwWritten, NULL);
    return success;
  }

protected:

  void init()
  {
    // pipeの作成

    SECURITY_ATTRIBUTES saAttr;

    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&child_std_out_read, &child_std_out_write, &saAttr, 0))
    {
      sync_cout << "error CreatePipe : std out" << sync_endl;
      return;
    }

    if (!CreatePipe(&child_std_in_read, &child_std_in_write, &saAttr, 0))
    {
      sync_cout << "error CreatePipe : std in" << sync_endl;
      return;
    }
  }

  PROCESS_INFORMATION pi;
  STARTUPINFO si;

  HANDLE child_std_out_read;
  HANDLE child_std_out_write;
  HANDLE child_std_in_read;
  HANDLE child_std_in_write;
};


// --- Search

void Search::init() {}
void Search::clear() {}
void MainThread::think() {
  
  ProcessNegotiator pn;
  pn.run(L"test.exe");
  pn.write("usi");
  string ret;
  while (true)
  {
    ret = pn.read();
    sync_cout << ret << sync_endl;
  }
}
void Thread::search() {}

#endif
