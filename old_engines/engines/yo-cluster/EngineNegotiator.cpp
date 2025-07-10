#include "../../config.h"
#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "EngineNegotiator.h"
#include "ClusterCommon.h"
#include "../../misc.h"

using namespace std;
using namespace YaneuraouTheCluster;

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//          EngineNegotiator
	// ---------------------------------------

	// EngineNegotiatorの実装。
	// 注意 : このクラスは、スレッドを持たない。
	//    親クラス側から定期的に receive()、必要に応じて send() を呼び出すことを想定している。
	//    receive()とsend()は同時には呼び出されない。(ことを想定している)
	//    それゆえ、このクラスでは排他は必要ない。
	class EngineNegotiatorImpl : public IEngineNegotiator
	{
	public:
		// -------------------------------------------------------
		//    constructor/destructor
		// -------------------------------------------------------

		EngineNegotiatorImpl()
		{
			state       = EngineState::DISCONNECTED;
			engine_mode = (EngineMode)
				( EngineMode::SEND_INFO_BEFORE_GAME
				| EngineMode::SEND_INFO_ON_GO
				);
			in_game     = false;
			engine_id   = size_max; // 未初期化
			go_count    = 0;
			ponderhit   = false;
		}

		// これcopyされてはかなわんので、copyとmoveを禁止しておく。
		EngineNegotiatorImpl(const EngineNegotiatorImpl& other)         = delete;
		EngineNegotiatorImpl&& operator = (const EngineNegotiatorImpl&) = delete;

		// -------------------------------------------------------
		//    Methods
		// -------------------------------------------------------

		// エンジンを起動する。
		virtual void connect(const string& path, size_t engine_id_)
		{
			engine_id = engine_id_;

			// エンジンの作業ディレクトリ。これはエンジンを実行するフォルダにしておいてやる。
			string working_directory = Path::GetDirectoryName(Path::Combine(CommandLine::workingDirectory , "engines/" + path));

			// エンジンのファイル名。(エンジンのworking_directory相対)
			string engine_name = Path::GetFileName(path);

			// 特殊なコマンドを実行したいなら起動したいプロセス名に"engines/"とかつけたら駄目。
			if (StringExtension::StartsWith(path,"ssh"))
			{
				// ただし、working directoryは、enginesではある。
				working_directory = Path::GetDirectoryName(Path::Combine(CommandLine::workingDirectory , "engines"));

				// "ssh"コマンドをそのまま実行できるように、ProcessNegotiatorにはコマンドそのまま渡す。
				engine_name = path;
			}

			// ProcessNegotiatorを使ってこのエンジンを起動。
			neg.connect(working_directory, engine_name);

			if (is_terminated())
				// 起動に失敗したくさい。
				error_to_gui("fail to connect = " + engine_name);
			else
			{
				// エンジンが起動したので出力しておく。
				DebugMessage(": Invoke Engine , engine_path = " + engine_name + " , engine_id = " + std::to_string(engine_id));

				// 起動直後でまだメッセージの受信スレッドが起動していないので例外的にmain threadからchange_state()を
				// 呼び出しても大丈夫。
				change_state(EngineState::CONNECTED);
			}
		}

		// Messageを解釈してエンジンに送信する。
		// 結果はすぐに返る。親クラス(ClusterObserver)の送受信用スレッドから呼び出す。
		// send()とreceive()とは同時に呼び出されない。
		// (親クラスの送受信用のスレッドは、送受信のために1つしかスレッドが走っていないため。)
		virtual void send(Message message)
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
#if 0
				if (state != EngineState::WAIT_ISREADY)
					EngineError("'setoption' should be sent before 'isready'.");
#endif
				// →　これ書いてあると自己対局フレームワークがisreadyのあとにsetoption送っていてこのエラーが出る。

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

				// 対局中である。
				in_game = true;

				break;

			case USI_Message::GO:
				// "go ponder"中に次のgoが来ることはありうる。
				stop_thinking();

				// 対局中ではないので受理しない。
				if (!is_in_game())
				{
					error_to_gui("'go' before usinewgame");
					return ;
				}

				// エンジン側からbestmove来るまで次のgo送れないのでは…。いや、go ponderなら送れるのか…。
				if (state != EngineState::IDLE_IN_GAME)
					EngineError("'go' should be sent when state is 'IDLE_IN_GAME'.");

				// positionコマンドとgoコマンド
				searching_sfen = message.position_sfen;
				send_to_engine("position " + searching_sfen);
				send_to_engine(message.command);

				state = EngineState::GO;
				ponderhit = false;
				time_to_return_bestmove = false;
				++go_count;

				break;

			case USI_Message::GO_PONDER:
				// "go ponder"中に次の"go ponder"が来ることはありうる。
				stop_thinking();

				// 対局中ではないので受理しない。
				if (!is_in_game())
				{
					error_to_gui("'go ponder' before usinewgame");
					return ;
				}

				// 本来、エンジン側からbestmove来るまで次のgo ponder送れないが、
				// ここでは、ignore_bestmoveをインクリメントしておき、この回数だけエンジン側からのbestmoveを
				// 無視することによってこれを実現する。
				if (   state != EngineState::IDLE_IN_GAME
					&& state != EngineState::GO_PONDER
					)
					EngineError("'go ponder' should be sent when state is 'IDLE_IN_GAME'.");

				// positionコマンドとgo ponderコマンド
				{
					searching_sfen = message.position_sfen;
					send_to_engine("position " + searching_sfen);
					const string& cmd = message.command.empty() ? "go ponder" : message.command;
					send_to_engine(cmd);
				}

				state = EngineState::GO_PONDER;
				time_to_return_bestmove = false;
				++go_count;

				break;

			case USI_Message::STOP:
				// これ、処理したくはないのだが、通常対局では送られてこないので
				// 深く考えずにエンジンに丸投げしておく。
				// ただし、go_count == 0だとそれは明らかに無効なstopなので無視する。
				if (go_count > 0)
					send_to_engine("stop");
				break;

			case USI_Message::PONDERHIT:
				// go ponder中以外にponderhitが送られてきた。
				if (state != EngineState::GO_PONDER)
					EngineError("'ponderhit' should be sent when state is 'GO_PONDER'.");

				send_to_engine("ponderhit " + message.command); // "ponderhit XXX"

				state = EngineState::GO;
				ponderhit = true;

				// ここまでの思考ログをGUIに出力してやる必要がある。
				// ただしこの出力には時間がかかる可能性があるので(GUI側で詰まる可能性がある)、
				// ponderhitのメッセージは先行してエンジンに送ったほうが良い。
				output_thinklog();

				break;

			case USI_Message::GAMEOVER:
				// 思考の停止
				stop_thinking();
				in_game = false;

				send_to_engine("gameover");

				// bestmove受け取っていないのに状態変更するの、ちょっと危ない気がしなくはない。
				// まあ、GUI側は一定時間は無視するのだろうからいいや。
				state = EngineState::WAIT_ISREADY;
				bestmove_string.clear();

				break;

			case USI_Message::QUIT:
				state = EngineState::QUIT;
				send_to_engine("quit");
				break;

			default:
				EngineError("Illegal message from ClusterObserver : " + message.to_string());
				break;
			}
		}

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

		// -------------------------------------------------------
		//    Property
		// -------------------------------------------------------

		// プロセスの終了判定
		virtual bool is_terminated() const { return neg.is_terminated(); }

		// エンジンが対局中であるか。
		// "usinewgame"～"gameover"の間であるか。
		virtual bool is_in_game() const { return in_game; }

		// エンジンIDを取得する。
		virtual size_t get_engine_id() const { return engine_id; }

		// 現在、"go","go ponder"によって探索中の局面。
		// ただし、"go"に対してエンジンが"bestmove"を返したあとも
		// その探索していた局面のsfenを、このメソッドで取得できる。
		virtual string get_searching_sfen() const { return searching_sfen; }

		// 直前のコマンドはponderhitであるか？
		// (現在is_state_go()==true (探索中)であるとして、
		// 送られてきたコマンドが"go"なのか"ponderhit"なのかを区別するのに使う)
		virtual bool   is_ponderhit() const { return ponderhit; }

		// "info string time to return bestmove"をエンジンから受け取った。
		virtual bool   received_time_to_return_bestmove() const { return time_to_return_bestmove; }

		// エンジン側から受け取った"bestmove XX ponder YY"を返す。
		// 一度このメソッドを呼び出すと、次以降は(エンジン側からさらに"bestmove XX ponder YY"を受信するまで)空の文字列が返る。
		// つまりこれは、size = 1 の PC-queueとみなしている。
		virtual string pull_bestmove() {
			auto result = bestmove_string;
			bestmove_string.clear();
			return result;
		}

		// エンジン側から受け取った"bestmove XX ponder YY"を返す。
		// pull_bestmove()と違って、このクラスの保持しているbestmove_stringは空にならない。
		virtual string peek_bestmove() {
			return bestmove_string;
		}

		// 思考ログを取得する。
		// (エンジン側から送られてきた"info ..."の文字列)
		// 前回"go","go ponder"されて以降のログ。
		virtual vector<string>* peek_thinklog()
		{
			return &think_log;
		}

		// 思考ログをクリアする。
		// この関数を呼び出すと、保持していた思考ログはクリアされる。
		virtual void clear_thinklog()
		{
			think_log.clear();
		}

		// エンジンの状態を取得する。
		// エンジンの状態は、send() , receive()でしか変化しないから、これで取得中に変化することはない。
		virtual EngineState get_state() const { return state; }

		// エンジンの動作モードを設定する。
		virtual void        set_engine_mode(EngineMode m) {        engine_mode = m; }

		// エンジンの動作モードを取得する。
		virtual EngineMode  get_engine_mode() const       { return engine_mode;     }
		
	private:
		// -------------------------------------------------------
		//    private methods
		// -------------------------------------------------------

		// [main thread][receive thread]
		// ProcessID(engine_id)を先頭に付与してDebugMessageCommon()を呼び出す。
		// "[0] >usi" のようになる。
		// ここで送信したメッセージは、DebugMessageCommon()も呼び出されるので、
		// file logを有効にしているとファイルにも書き出される。
		void DebugMessage(const string& message)
		{
			DebugMessageCommon("[" + std::to_string(engine_id) + "]" + message);
		}

		// メッセージをエンジン側に送信する。
		// ここで送信したメッセージは、DebugMessageCommon()も呼び出されるので、
		// file logを有効にしているとファイルにも書き出される。
		void send_to_engine(const string& message)
		{
			DebugMessage("< " + message);

			neg.send(message);
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
		// (receive thread起動前に、process起動を行うが、その時に呼び出されることはある)
		void change_state(EngineState new_state)
		{
			DebugMessage(": change_state " + to_string(state) + " -> " + to_string(new_state));
			state = new_state;
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
				//error_to_gui("Illegal state in stop_thinking() , state = " + to_string(state));
				// →　gameoverのときにstopさせることはあるからおかしくはない。

				send_to_engine("stop");
				// この場合、bestmoveを待ってから状態を変更してやる必要があるのだが…。
				// そもそもで言うと "go"して stopが来る前に gameoverが来ているのがおかしいわけだが。
				state = EngineState::IDLE_IN_GAME;
			}
		}

		// そこまでの思考ログを出力する。
		void output_thinklog()
		{
			// ログを送信するモードではないなら何もせずに帰る。
			if (!(engine_mode & EngineMode::SEND_INFO_ON_GO))
				return ;

			// "GO_PONDER"が何重にも送られてきている。
			// まだ直前のGO_PONDERのログがエンジン側から送られてきていない。
			if (go_count >= 2)
				return ;

			// ここまでの思考ログを吐き出してやる。
			for(auto& log : think_log)
				send_to_gui(log);
			think_log.clear();
		}

		// 状態変数のクリア
		// ゲーム開始時/終局時の初期化。
		void init_for_think()
		{
			searching_sfen.clear();
			think_log.clear();
			bestmove_string.clear();
			in_game    = false;
			go_count   = 0; // 前のgameover以降にgoしてたり、stopが来ずにgameoverが来てる可能性とかがある。
			ponderhit  = false;
		}

		// EngineStateの状態がsではない時に警告をGUIに出力する。
		void StateAssert(EngineState s)
		{
			if (state != s)
				error_to_gui("StateAssert failed, Illegal state : state = " + to_string(s));
		}

		// エンジン側から送られてきたメッセージを配る(解読して適切な配達先に渡す)
		void dispatch_message(const string& message_)
		{
			ASSERT_LV3(state != EngineState::DISCONNECTED);

			// 書き換えたいのでcopyしておく。
			string message = message_;

			// 受信したメッセージをログ出力しておく。
			DebugMessage("> " + message);

			Parser::LineScanner scanner(message);
			string token = scanner.get_text();

			// GUIに転送するのか？のフラグ
			bool send_gui = false;

			if (token == "info")
			{
				// ponder中であればその間に送られてきたメッセージは全部ログに積んでおく。
				// (ponderhitが送られてきた時に、そこまでのlogをGUIに出力しなければならないため)
				if (state == EngineState::GO_PONDER || state == EngineState::GO)
				{
					if (go_count == 0)
						DebugMessage(": Warning! : Illegal state , state = " + to_string(state) + " , go_count == 0");
					else if (go_count == 1)
					{
						if (StringExtension::Contains(message,"time to return bestmove"))
						{
							// これは、フラグを変化させるだけで、このメッセージ自体はなかったことにする。

							send_gui = false;
							time_to_return_bestmove = true;
						}
						else if (state == EngineState::GO)
						{
							send_gui = engine_mode & EngineMode::SEND_INFO_ON_GO;
							if (!send_gui)
								think_log.emplace_back(message);
						}
						else // state == EngineState::GO_PONDER
							think_log.emplace_back(message);
					}
					else
						;
						// ignore_bestmove >= 2なら、どうせいま受信したメッセージは捨てることになるのでthink_logに積まない。
						// 当然ながら出力もしない。
				}
				// "usiok", "readyok" 待ちの時は、engine_modeでそのまま垂れ流し設定になっていれば垂れ流す。
				else if (state == EngineState::WAIT_USIOK
					  || state == EngineState::WAIT_READYOK
					)
				{
					send_gui = engine_mode & EngineMode::SEND_INFO_BEFORE_GAME;
					// ただし、どのエンジンから送られてきたかを区別するために、"info string [0] xxx"のように
					// engine_idを付与したいところではある。
					if (scanner.get_text() == "string")
						message = "info string [" + std::to_string(engine_id) + "] " + scanner.get_rest();
				}
				else
					// usiok/readyok待ちと go ponder , go 以外のタイミングでエンジン側からinfo stringでメッセージが来るのおかしいのでは…。
					// ただしignore_bestmove > 0なら、bestmove来るまでは無視していいのか…。
					if (go_count == 0)
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
				if (go_count > 0)
					--go_count;
				else
					error_to_gui("bestmove received when go_count == 0");

				if (go_count > 0)
				{
					// bestmoveまでは無視して良かったことが確定したのでこの時点でクリアしてしまう。
					think_log.clear();

				} else {

					// GO以外でbestmove返ってくるのおかしい。
					// ただし、gameoverのあとかも知れんが…。
					// てか、bestmove返してないのに"gameover"送ってくる実装がおかしい気もするが…。
					if (state != EngineState::GO && state != EngineState::WAIT_ISREADY)
						error_to_gui("Illegal state , bestmove received when state = " + to_string(state));

					// GOで思考していたなら、bestmoveを親クラスに返す必要がある。
					// これを設定しておけば親クラスが検知してくれる。
					if (state == EngineState::GO)
					{
						// bestmoveは、常に親クラスの責任においてGUIに送信する。
						// →　そうしておかないとエンジンが途中で切断された場合、bestmoveをGUIに送信したのかしていないのかが
						// 　　親クラスがわからない。
						// ただし、ゲーム中でないなら無視して良いし(遅れてやってきたbestmove)、go_count > 0 なら
						// なかったことにして良い。(もう次のgoが来て次の局面について考えている)
						if (go_count == 0 && in_game)
							bestmove_string = message;

						send_gui = false;

						// 思考は停止している。
						change_state(EngineState::IDLE_IN_GAME);

						// 探索中の局面のsfenを示す変数をクリア。
						//searching_sfen = string();
						// →　これは空にしては駄目。この情報使う。
					}
				}
			}
			else if (token == "usiok")
				// "usiok"は全部のエンジンが WAIT_ISREADYになった時に親クラス(Observer)がGUIに送るので状態の変更だけ。
				change_state(EngineState::WAIT_ISREADY);
			else if (token == "readyok")
			{
				// このタイミングで内部状態の初期化を行う。
				init_for_think();

				// → "readyok"は全部のエンジンが IDLE_IN_GAMEになった時に親クラス(Observer)がGUIに送るので状態の変更だけ。
				change_state(EngineState::IDLE_IN_GAME);
			}
			// "id"はエンジン起動時にしか来ないはずだが？
			else if (token == "id")
			{
				string token2 = scanner.get_text();
				// "id author"のタイミングでこのエンジンのinfoを出しておく。
				// それ以外の"id XXX"は無視する。
				if (engine_id == 0 && token2 == "author")
					send_to_gui(engine_info());
				send_gui = false;
			}
			else if (token == "option")
			{
				StateAssert(EngineState::WAIT_USIOK);
				send_gui = engine_id == 0;
			}

			// GUIに転送しないといけないメッセージであった。
			if (send_gui)
				send_to_gui(message);

		}

		// -------------------------------------------------------
		//    private members
		// -------------------------------------------------------

		// 子プロセスとやりとりするためのhelper
		ProcessNegotiator neg;

		// エンジンID。これはconnect()の時に親classから渡される。
		size_t engine_id;

		// エンジンに対して何をやっている状態であるのか。
		EngineState state;

		// 対局中であるか。("usinewgame"～"gameover"の間であるか)
		bool in_game;

		// エンジンに送信した"go","go ponder"の回数。
		// エンジン側から"bestmove"を受信すると 1 減る。
		// このカウンターが2以上ならエンジン側から送られてきたbestmoveを無視する。
		// (エンジン側からの"bestmove"を待たずして、次の"go"を送っているゆえに生じる問題なので)
		int go_count;

		// 探索中の局面
		// state == GO or GO_PONDER において探索中の局面。
		// 探索終了後は、前回探索(GO or GO_PONDER)で探索していたsfen。
		string searching_sfen;

		// 直前のコマンドはponderhitであるか？
		// (現在is_state_go()==true (探索中)であるとして、
		// 送られてきたコマンドが"go"なのか"ponderhit"なのかを区別するのに使う)
		bool ponderhit;

		// "go ponder"時にエンジン側から送られてきた思考ログ。
		// そのあと、"ponderhit"が送られてきたら、
		// その時点までの思考ログをGUIにそこまでのログを出力するために必要。
		vector<string> think_log;

		// エンジン側から返ってきた、"bestmove XXX ponder YYY" みたいな文字列。
		// getter(get_bestmove())で取得したら、なくなる。(clearされる)
		string bestmove_string;

		// エンジンの動作モード。
		EngineMode engine_mode;

		// "info string time to return bestmove"を受信したのか。
		bool time_to_return_bestmove;
	};

	EngineNegotiator::EngineNegotiator()
	{
		ptr = make_unique<EngineNegotiatorImpl>();
	}
}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
