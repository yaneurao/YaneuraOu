#include "../../shogi.h"
#ifdef LOCAL_GAME_SERVER

#include "../../extra/all.h"

#include <windows.h>

// 子プロセスとの通信ログをデバッグのために表示するオプション
//#define OUTPUT_PROCESS_LOG

// 1行ずつ結果を出力するモード
#define ONE_LINE_OUTPUT_MODE

// 勝敗の出力のときについでに対局棋譜を出力する機能
//#define OUTPUT_KIF_LOG


// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
	// "engine_configX.txt"が配置してあるフォルダ
	o["EngineConfigDir"] << Option("");

	// 子プロセスでEngineを実行するプロセッサグループ(Numa node)
	// -1なら、指定なし。
	o["EngineNuma"] << Option(-1, 0, 99999);

	// 連続自己対戦のときに定跡の局面まで進めるためのsfenファイル。
	// このファイルの棋譜のまま32手目まで進める。
	o["BookSfenFile"] << Option("book.sfen");
}

// 子プロセスを実行して、子プロセスの標準入出力をリダイレクトするのをお手伝いするクラス。
struct ProcessNegotiator
{
  ProcessNegotiator() { init(); }

  virtual ~ProcessNegotiator() {
    if (pi.hProcess) {
      if (::WaitForSingleObject(pi.hProcess, 1000) != WAIT_OBJECT_0) {
        ::TerminateProcess(pi.hProcess, 0);
      }
      ::CloseHandle(pi.hProcess);
      pi.hProcess = nullptr;
    }
  }

#ifdef OUTPUT_PROCESS_LOG
  // 子プロセスとの通信ログを出力するときにプロセス番号を設定する
  void set_process_id(int pn_) { pn = pn_; }
#endif

  // 子プロセスの実行
  void run(string app_path_)
  {
	int numa = (int)Options["EngineNuma"];
	if (numa != -1)
	{
		// numaが指定されているので、それを反映させる。
		// cmd.exe経由でstartコマンドを叩く。(やねうらお考案)

		app_path_ = "cmd.exe /c start /B /WAIT /NODE " + to_string(numa) + " " + app_path_;

		/*
			/NODE
			Numa node(実行するプロセッサグループ)の指定

			/B
			実行したコマンドを別窓で開かないためのオプション。

			/WAIT
			/Bを指定したときにはこれを指定して、本窓の終了を待機しないと
			プロセス実行中に制御を抜けてしまい、file descriptorがleakする。
		*/

	}

	  wstring app_path = to_wstring(app_path_);

	  ZeroMemory(&pi, sizeof(pi));
	  ZeroMemory(&si, sizeof(si));

	  si.cb = sizeof(si);
	  si.hStdInput = child_std_in_read;
	  si.hStdOutput = child_std_out_write;
	  si.dwFlags |= STARTF_USESTDHANDLES;

	  // Create the child process

	  success = ::CreateProcess(
		  NULL, // ApplicationName
		  (LPWSTR)app_path.c_str(),  // CmdLine
		  NULL, // security attributes
		  NULL, // primary thread security attributes
		  TRUE, // handles are inherited
		  0,    // creation flags
		  NULL, // use parent's environment
		  NULL, // use parent's current directory
		  // ここにカレントディレクトリを指定する。
		  // engine/xxx.exe を起動するなら engine/ を指定するほうがいいような気は少しする。

		  &si,  // STARTUPINFO pointer
		  &pi   // receives PROCESS_INFOMATION
	  );

	  if (!success)
		  sync_cout << "CreateProcessに失敗" << sync_endl;

	  if (pi.hThread) {
		  ::CloseHandle(pi.hThread);
		  pi.hThread = nullptr;
	  }
  }
  bool success;

  // 長手数になるかも知れないので…。
  static const int BUF_SIZE = 4096;

  string read()
  {
    auto result = read_next();
    if (!result.empty())
      return result;

    // ReadFileは同期的に使いたいが、しかしデータがないときにブロックされるのは困るので
    // pipeにデータがあるのかどうかを調べてからReadFile()する。

    DWORD dwRead , dwReadTotal , dwLeft;
    CHAR chBuf[BUF_SIZE];

    // bufferサイズは1文字少なく申告して終端に'\0'を付与してstring化する。

    BOOL success = ::PeekNamedPipe(
      child_std_out_read, // [in]  handle of named pipe
      chBuf,              // [out] buffer     
      BUF_SIZE-1,         // [in]  buffer size
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
    return read_next();
  }

  bool write(string str)
  {
    str += "\r\n"; // 改行コードの付与
    DWORD dwWritten;
    BOOL success = ::WriteFile(child_std_in_write, str.c_str(), DWORD(str.length()) , &dwWritten, NULL);

#ifdef OUTPUT_PROCESS_LOG
    sync_cout << "[" << pn << "] >" << str << sync_endl;
#endif

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

#define ERROR_MES(mes) { sync_cout << mes << sync_endl; return ; }

    if (!::CreatePipe(&child_std_out_read, &child_std_out_write, &saAttr, 0))
      ERROR_MES("error CreatePipe : std out");

    if (!::SetHandleInformation(child_std_out_read, HANDLE_FLAG_INHERIT, 0))
      ERROR_MES("error SetHandleInformation : std out");
    
    if (!::CreatePipe(&child_std_in_read, &child_std_in_write, &saAttr, 0))
      ERROR_MES("error CreatePipe : std in");

    if (!::SetHandleInformation(child_std_in_write, HANDLE_FLAG_INHERIT, 0))
      ERROR_MES("error SetHandleInformation : std in");

#undef ERROR_MES
  }

  string read_next()
  {
    // read_bufferから改行までを切り出す
    auto it = read_buffer.find("\n");
    if (it == string::npos)
      return string();
    // 切り出したいのは"\n"の手前まで(改行コード不要)、このあと"\n"は捨てたいので
    // it+1から最後までが次回まわし。
    auto result = read_buffer.substr(0, it);
    read_buffer = read_buffer.substr(it+1, read_buffer.size() - it);
    // "\r\n"かも知れないので"\r"も除去。
    if (result.size() && result[result.size() - 1] == '\r')
      result = result.substr(0, result.size() - 1);

#ifdef OUTPUT_PROCESS_LOG
    sync_cout << "[" << pn << "] >" <<  result << sync_endl;
#endif

    if (result.find("Error") != string::npos)
    {
      // 何らかエラーが起きたので表示させておく。
      sync_cout << "Error : " << result << sync_endl;
    }

    return result;
  }

  // wstring変換
  wstring to_wstring(const string& src)
  {
    size_t ret;
    wchar_t *wcs = new wchar_t[src.length() + 1];
    ::mbstowcs_s(&ret,wcs, src.length()+1, src.c_str(), _TRUNCATE);
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

  // 受信バッファ
  string read_buffer;

#ifdef  OUTPUT_PROCESS_LOG
  // プロセス番号(ログ出力のときに使う)
  int pn;
#endif
};

struct EngineState
{
	void run(string path, int process_id)
	{
#ifdef  OUTPUT_PROCESS_LOG
		pn.set_process_id(process_id);
#endif
		pn.run(path);
		state = START_UP;
		engine_exe_name_ = path;
	}

	// エンジンに対する終了処理
	~EngineState()
	{
		// 思考エンジンにquitコマンドを送り終了する
		// プロセスの終了は~ProcessNegotiator()で待機し、
		// 終了しなかった場合はTerminateProcess()で強制終了する。
		pn.write("quit");
	}

	void on_idle()
	{
		switch (state)
		{
		case START_UP:
			pn.write("usi");
			state = WAIT_USI_OK;
			break;

		case WAIT_USI_OK:
		{
			string line = pn.read();
			if (line == "usiok")
				state = IS_READY;
			else if (line.substr(0, min(line.size(), 8)) == "id name ")
				engine_name_ = line.substr(8, line.size() - 8);
			break;
		}

		case IS_READY:
			// エンジンの初期化コマンドを送ってやる
			for (auto line : engine_config)
				pn.write(line);

			pn.write("isready");
			state = WAIT_READY_OK;
			break;

		case WAIT_READY_OK:
			if (pn.read() == "readyok")
			{
				pn.write("usinewgame");
				state = GAME_START;
			}
			break;

		case GAME_START:
			break;

		case GAME_OVER:
			pn.write("gameover");
			state = START_UP;
			break;
		}

	}

	Move think(const Position& pos, const string& think_cmd, const string& engine_name)
	{
		string sfen;
		sfen = "position startpos moves " + pos.moves_from_start();
		pn.write(sfen);
		pn.write(think_cmd);
		string bestmove;

		auto start = now();
		while (true)
		{
			bestmove = pn.read();
			if (bestmove.find("bestmove") != string::npos)
				break;

			// タイムアウトチェック(連続自己対戦で1手に1分以上考えさせない
			if (now() >= start + 60 * 1000)
			{
				sync_cout << "Error : engine timeout , engine name = " << engine_name << endl << pos << sync_endl;
				// これ、プロセスが落ちてると思われる。
				// プロセスを再起動したほうが良いのでは…。

				return MOVE_NULL; // これを返して、終了してもらう。
			}

			sleep(5);
		}
		istringstream is(bestmove);
		string token;
		is >> skipws >> token; // "bestmove"
		is >> token; // "7g7f" etc..

		Move m = move_from_usi(pos, token);
		if (m == MOVE_NONE)
		{
			sync_cout << "Error : bestmove = " << token << endl << pos << sync_endl;
			m = MOVE_RESIGN;
		}
		return m;
	}

	enum State {
		START_UP, WAIT_USI_OK, IS_READY, WAIT_READY_OK, GAME_START, GAME_OVER,
	};

	// 対局の準備が出来たのか？
	bool is_game_started() const { return state == GAME_START; }

	// エンジンの初期化時に渡したいメッセージ
	void set_engine_config(vector<string>& lines) { engine_config = lines; }

	// ゲーム終了時に呼び出すべし。
	void game_over() { state = GAME_OVER; }

	// usiコマンドに対して思考エンジンが"is name ..."で返してきたengine名
	string engine_name() const { return engine_name_; }

	// 実行したエンジンのバイナリ名
	string engine_exe_name() const { return engine_exe_name_; }

	ProcessNegotiator pn;

protected:

	// 内部状態
	State state;

	// エンジン起動時に送信すべきコマンド
	vector<string> engine_config;

	// usiコマンドに対して思考エンジンが"is name ..."で返してきたengine名
	string engine_name_;

	// 実行したエンジンのバイナリ名
	string engine_exe_name_;

};

// --- Search

/*
  実行ファイルと同じフォルダに次の2つのファイルを配置。

  engine-config1.txt : 1つ目の思考エンジン
  engine-config2.txt : 2つ目の思考エンジン
  
    1行目にengineの実行ファイル名
    2行目に思考時のコマンド
    3行目以降にsetoption等、初期化時に思考エンジンに送りたいコマンドを書く。

  例)
    test.exe
    go btime 100 wtime 100 byoyomi 0
    setoption name Threads value 1
    setoption name Hash value 1024

  次に
  goコマンドを打つ。(実行中、stopは利かない)
  
  O : engine1勝ち
  X : engine1負け
  . : 引き分け

  あるいは、ONE_LINE_OUTPUT_MODEをdefineすることによって
  win,sfenで終局の局面図
  lose,sfenで終局の局面図
  draw,sfenで終局の局面図
  が返る。

  対局回数の指定など)
    go btime [対局回数] wtime [定跡の手数] byoyomi [自動終了オプション]

    定跡はbook.sfenとしてsfen形式のファイルを与える。1行に1局が書かれているものとする。
    このなかからランダムに1行が選ばれてその手数は上のwtimeのところで指定した手数まで進められる。
    デフォルトでは対局回数は100回。定跡の手数は32手目から。

	対局回数は並列化されている分も考慮した、トータルでの対局回数。
	この回数に達すると対局中のものは中断して打ち切る。

	また、byoyomiのところは、自動終了オプションを指定するようになっていて、
	ここが1だと、btimeで指定された回数の対局数をこなすと自動的にquitする。
*/

void Search::init(){}
void Search::clear() {}

namespace
{
  // 思考エンジンの実行ファイル名
  string engine_name[2];

  // usiコマンドに応答として返ってきたエンジン名
  string usi_engine_name[2];

  // 思考エンジンの設定
  vector<string> engine_config_lines[2];

  // 思考コマンド
  string think_cmd[2];

  // 勝数のトータル
  uint64_t win,draw,lose;

  vector<string> book;
  PRNG book_rand; // 定跡用の乱数生成器
  Mutex local_mutex;

  int64_t get_rand(size_t n)
  {
    std::unique_lock<Mutex> lk(local_mutex);
    return book_rand.rand(n);
  }

  // 対局数
  volatile int games = 0;

  // gamesをインクリメントするときに必要なmutex
  Mutex games_mutex;
}

void MainThread::think() {

  // 設定の読み込み
  fstream f[2];
  string config_dir = Options["EngineConfigDir"];
  f[0].open(path_combine(config_dir,"engine-config1.txt"));
  f[1].open(path_combine(config_dir,"engine-config2.txt"));

  getline(f[0], engine_name[0]);
  getline(f[1], engine_name[1]);

  getline(f[0], think_cmd[0]);
  getline(f[1], think_cmd[1]);

  for (int i = 0; i < 2; ++i) {
    auto& lines = engine_config_lines[i];
    lines.clear();
    string line;
    while (!f[i].eof())
    {
      getline(f[i], line);
      if (!line.empty())
        lines.push_back(line);
    }
    f[i].close();
  }


  win = draw = lose = 0;
  games = 0;

  // -- 定跡
  book.clear();

  // 定跡ファイル(というか単なる棋譜ファイル)の読み込み
  fstream fs_book;
  string book_file_name = Options["BookSfenFile"];
  fs_book.open(book_file_name);
  if (!fs_book.fail())
  {
    sync_cout << "read " + book_file_name << sync_endl;
    string line;
    while (!fs_book.eof())
    {
      getline(fs_book, line);
      if (!line.empty())
        book.push_back(line);
      if ((book.size() % 100) == 0)
        cout << ".";
    }
    cout << endl;
  } else {
    sync_cout << "Error! : can't read book.sfen" << sync_endl;
  }

  sync_cout << "local game server start : " << engine_name[0] << " vs " << engine_name[1] << sync_endl;

  // マルチスレッド対応
  for (auto th : Threads.slaves) th->start_searching(); 
  Thread::search();
  for (auto th : Threads.slaves) th->wait_for_search_finished();

  sync_cout << endl << "local game server end : [" << engine_name[0] << "] vs [" << engine_name[1] << "]" << sync_endl;
  sync_cout << "GameResult " << win << " - " << draw << " - " << lose << sync_endl;

#ifdef ONE_LINE_OUTPUT_MODE
  sync_cout << "finish" << sync_endl;
#endif

}

void Thread::search()
{
	EngineState es[2];
	es[0].run(engine_name[0], 0);
	es[1].run(engine_name[1], 1);

	// プロセスの生成に失敗しているなら終了。
	if (!es[0].pn.success || !es[1].pn.success)
		return;

	for (int i = 0; i < 2; ++i)
		es[i].set_engine_config(engine_config_lines[i]);

	Color player1_color = BLACK;

	bool game_started = false;

	// 対局回数。btimeの値がmax_games
	int max_games = Search::Limits.time[BLACK];
	if (max_games == 0)
		max_games = 100; // デフォルトでは100回
	//  int games = 0;

	  // 定跡の手数
	int max_book_move = Search::Limits.time[WHITE];
	if (max_book_move == 0)
		max_book_move = 32; // デフォルトでは32手目から


	auto SetupStates = Search::StateStackPtr(new aligned_stack<StateInfo>);

	// 対局開始時のハンドラ
	auto game_start = [&] {
		rootPos.set_hirate();
		game_started = true;

		// 定跡が設定されているならその局面まで進める
		if (book.size())
		{
			int book_number = (int)get_rand(book.size());
			istringstream is(book[book_number]);
			string token;
			while (rootPos.game_ply() < max_book_move)
			{
				is >> token;
				if (token == "startpos" || token == "moves")
					continue;

				Move m = move_from_usi(rootPos, token);
				if (!is_ok(m))
				{
					//  sync_cout << "Error book.sfen , line = " << book_number << " , moves = " << token << endl << rootPos << sync_endl;
					// →　エラー扱いはしない。
					break;
				} else {
					SetupStates->push(StateInfo());
					rootPos.do_move(m, SetupStates->top());
				}
			}
			//cout << rootPos;
		}
	};

	// 対局終了時のハンドラ
	// 投了(resign)である場合、手番側の負け。
	// 宣言勝ち(!resign)である場合、手番側の勝ち。
	auto game_over = [&](bool resign) {
		std::unique_lock<Mutex> lk(local_mutex);

		auto kif =
#ifdef OUTPUT_KIF_LOG
			// sfen形式の棋譜を出力する。
			"startpos moves " + rootPos.moves_from_start();
#else
			rootPos.sfen();
#endif

		{
			std::unique_lock<Mutex> lk(games_mutex);
			if (games < max_games)
			{
				games++;

				if (rootPos.game_ply() >= 256) // 長手数につき引き分け
				{
					draw++;
#ifdef ONE_LINE_OUTPUT_MODE
					sync_cout << "draw," << kif << sync_endl;
#else
					cout << '.'; // 引き分けマーク
#endif
				} else if ((rootPos.side_to_move() == player1_color) ^ !resign)
				{
					lose++;
#ifdef ONE_LINE_OUTPUT_MODE
					sync_cout << "lose," << kif << sync_endl;
#else
					cout << 'X'; // 負けマーク
#endif
				} else
				{
					win++;
#ifdef ONE_LINE_OUTPUT_MODE
					sync_cout << "win," << kif << sync_endl;
#else
					cout << 'O'; // 勝ちマーク
#endif
				}
			} else {
				// 終了条件は満たしているはずなのでこれにて終了。
			}
		}
		player1_color = ~player1_color; // 先後入れ替える。
										//    sync_cout << rootPos << sync_endl; // デバッグ用に投了の局面を表示させてみる
		game_started = false;

		es[0].game_over();
		es[1].game_over();

		// これクリアしておかないとメモリを消費し続けてもったいない。
		SetupStates->clear();
	};

	string line;
	while (/*!Search::Signals.stop &&*/ games < max_games)
	{
		// stopは受け付けないようにする。
		// そうしないとコマンドラインから実行するときにquitコマンドをqueueに積んでおくことが出来ない。

		es[0].on_idle();
		es[1].on_idle();

		if (!game_started && es[0].is_game_started() && es[1].is_game_started())
		{
			game_start();
			//sync_cout << "game start" << sync_endl;
		}

		// ゲーム中であれば局面を送って思考させる
		if (game_started)
		{
			int player = (rootPos.side_to_move() == player1_color) ? 0 : 1;
			auto engine_name = es[player].engine_exe_name(); // engine_name()だとエラーが起きたときにどれだかわからない可能性がある。
			Move m = es[player].think(rootPos, think_cmd[player], engine_name);

			// timeoutしたので終了させてしまう。
			if (m == MOVE_NULL)
				break;

			// 宣言勝ち
			if (m == rootPos.DeclarationWin())
			{
				game_over(false);
				continue;
			}

			rootPos.check_info_update();

			// 非合法手を弾く
			if (m != MOVE_RESIGN && (!rootPos.pseudo_legal(m) || !rootPos.legal(m)))
			{
				sync_cout << "Error : illigal move , move = " << m << " , engine name = " << engine_name << endl << rootPos << sync_endl;
				m = MOVE_RESIGN;
			} else {

				SetupStates->push(StateInfo());
				rootPos.do_move(m, SetupStates->top());
			}

			if (m == MOVE_RESIGN || rootPos.is_mated() || rootPos.game_ply() >= 256)
			{
				game_over(true);
				//sync_cout << "game over" << sync_endl;
			}
		}
		sleep(5);
	}

	// 試合が規定数に達したので強制終了させた(かも)なので、終了処理代わりにこれをやっておく。
	if (games == max_games)
		game_over(false);

	if (is_main())
	{
		usi_engine_name[0] = es[0].engine_name();
		usi_engine_name[1] = es[1].engine_name();
	}
}

#endif
