#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

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
// parent(host)  : このプログラム。ふかうら王/やねうら王NNUEを用いて動作する。USIエンジンのふりをする。思考する局面をworkerに割り振る。
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
//	  ※　実行path、full pathにした時に日本語が混じっていると起動に失敗する。日本語の混じったフォルダを使わないように。
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

// このエンジンの実行ファイルと同じフォルダに
//   startup.txt
// を配置して、そこに、例えば次のように書きます。
// 
//   setoption name BookFile value no_book
//   setoption name Threads value 4
//   cluster log nodes 100000
//   setoption name BookFile value no_book
//   setoption name Threads value 32
//
//  →　cluster engine側、定跡なしで4スレッド。workerは定跡なしで32スレッド。
//      clusterの思考ログをファイルに書き出す。次のponder対象局面を探す時の探索ノード数100000。
//
//
// 注意)
// 本プログラムが不正終了したりquitされる前に終了してしまうと、実行していたworkerのエンジンは実行したままになることがある。
// その場合は、実行していたエンジンをタスクマネージャーから終了させるなり何なりしなければならない。

#include <sstream>
#include <thread>
#include <variant>

#include "../../position.h"
#include "../../thread.h"
#include "../../usi.h"
#include "ClusterCommon.h"
#include "ProcessNegotiator.h"
#include "EngineNegotiator.h"
#include "ClusterObserver.h"
#include "ClusterStrategy.h"

#if defined(YANEURAOU_ENGINE_DEEP)
#include "../dlshogi-engine/dlshogi_min.h"
using namespace dlshogi;
#endif

using namespace std;

namespace YaneuraouTheCluster
{
	// ---------------------------------------
	//          cluster observer
	// ---------------------------------------

	class ClusterObserver
	{
	public:

		ClusterObserver(const ClusterOptions& options_ , unique_ptr<IClusterStrategy>& strategy_)
		{
			// エンジン生成してからスレッドを開始しないと、エンジンが空で困る。
			connect();

			// スレッドを開始する。
			options       = options_;
			strategy      = std::move(strategy_);

			// エンジン接続後のイベントの呼び出し。
			garbage_engines();
			strategy->on_connected(StrategyParam(engines,options));

			worker_thread = std::thread([&](){ worker(); });
		}

		~ClusterObserver()
		{
			send_wait(USI_Message::QUIT);
			worker_thread.join();
		}

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

		// 通信スレッドで受け取ったメッセージをこのClusterObserverに伝える。
		//    waitとついているほうのメソッドは送信し、処理の完了を待機する。
		void send_wait(USI_Message& usi)                      { send_wait(Message(usi       )); }
		void send_wait(USI_Message& usi, const string& param) { send_wait(Message(usi, param)); }

		// Messageを解釈してエンジンに送信する。
		void send(Message message)
		{
			// Observerからengineに対するメッセージ
			DebugMessageCommon("Cluster to ClusterObserver : " + message.to_string());

			queue.push(message);
			send_counter++;
		}

		// 通信スレッドで受け取ったメッセージをこのSupervisorに伝える。
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
						if (!options.ignore_setoption)
							broadcast(message);
						break;

					case USI_Message::USI:
					case USI_Message::ISREADY:
						usi = message.message; // ← この変数の状態変化まではエンジンの次のメッセージを処理しない。
						broadcast(message);
						strategy->on_isready(StrategyParam(engines,options));
						break;

					case USI_Message::USINEWGAME:
						// まず各エンジンに通知は必要。(各エンジンがこのタイミングで何かをする可能性はあるので)
						broadcast(message);

						// 現在、相手が初期局面("startpos")について思考しているものとする。
						searching_sfen2 = "startpos";
						our_searching2  = false;
						engine_ponder.clear();
						engine_ponder_engine_id = 0;

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

					case USI_Message::GO:

						// GOコマンドの処理は、Strategyに丸投げ
						garbage_engines();
						strategy->on_go_command(StrategyParam(engines,options), message);

						break;

					case USI_Message::STOP:
						// "stop"コマンド。これ処理したくはないが、通常対局中にではどうせ
						// "stop"は送られてこないので深く考えずにエンジンに丸投げしとく。
						broadcast(message);
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
							allOk &= engine.is_waiting_isready() || engine.is_terminated();
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

					default:
						break; // avoid warning
					}
				}

				// idle時の処理。
				on_idle();
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

		// enginesからterminateしているengineを除外する。
		void garbage_engines()
		{
			// terminateしたengineを除外して、teminateしたengineがない状態を保つ。
			auto itrNewEnd = std::remove_if(engines.begin(), engines.end(), [](EngineNegotiator& e)->bool { return e.is_terminated(); });
			engines.erase(itrNewEnd, engines.end());

			if (engines.size() == 0)
			{
				// 生きているengineが1つもない。
				sync_cout << "Error! : No engines." << sync_endl;
				Tools::exit();
			}
		}

		// idle時の処理。
		void on_idle()
		{
			// terminateしているengineがないことを保証する。
			garbage_engines();

			// idleなので、Strategy::on_idle()を呼び出してやる。
			strategy->on_idle(StrategyParam(engines,options));

			// エンジンの死活監視
			//engine_check();
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



		// --- private members ---

		// クラスターのoptions(設定値)
		ClusterOptions options;

		// クラスターの戦略
		unique_ptr<IClusterStrategy> strategy;

		// すべての思考エンジンを表現する。
		std::vector<EngineNegotiator> engines;

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

		// TODO : あとで整理する。

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
		// engine_ponderを返したエンジンのid
		size_t engine_ponder_engine_id;
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
			// clusterのオプション設定
			ClusterOptions options;
			unique_ptr<IClusterStrategy> strategy;

			// "cluster"コマンドのパラメーター解析
			parse_cluster_param(is, options, strategy);

			// GUIとの通信を行うmain threadのmessage loop
			message_loop_main(pos, options, strategy );

			// quitコマンドは受け取っているはず。
			// ここで終了させないと、cluster engineが単体のengineのように見えない。

			Tools::exit();
		}

	private:
		// "cluster"コマンドのパラメーターを解析して処理する。
		// 
		// 指定できるオプション一覧)
		// 
		//   debug            : debug用に通信のやりとりをすべて標準出力に出力する。
		//   nodes            : go ponderする局面を選ぶために探索するノード数(ふかうら王で探索する)
		//   skipinfo         : "info"文字列はdebugがオンでも出力しない。("info"で画面が流れていくの防止)
		//   log              : このcluster engineのログをfileに書き出す。
		//   ignore_setoption : GUI側からのsetoptionコマンドを無視する。(エンジンを個別にそのエンジンオプションを設定したい場合)
		//   mode
		//		single       : 単一エンジン、ponderなし(defaultでこれ)
		//		ponder       : 単一エンジン、ponderあり
		//		optimistic   : 楽観合議モード
		//				→　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
		//		multi        : MultiPonder 
		//				→　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
		//      split        : rootで指し手分割を行うモード
		//				→　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
		//					さらに、workerは goコマンドの"wait_stop"機能に対応している必要がある。(やねうら王NNUEは対応している)
		//      gps  [分割数] : GPS将棋のクラスター手法。rootのみ分割。worker数N。
		//				→　workerは ConsiderationMode = falseにしてbestmoveを返す直前には必ず評価値を出力するように設定する必要がある。
		//                  分割数は、1局面を何個の指し手に分割するか。default == 3 , 分割数 == worker数でなければならない。
		//		gps2 [分割数] : GPS将棋のクラスター手法。worker数可変。
		//					さらに、workerは goコマンドの"wait_stop"機能に対応している必要がある。(やねうら王NNUEは対応している)
		void parse_cluster_param(istringstream& is_, ClusterOptions& options , unique_ptr<IClusterStrategy>& strategy)
		{
			// USIメッセージの処理を開始している。いま何か出力してはまずい。

			// USI拡張コマンドの"cluster"コマンドに付随できるオプション
			// 例)
			// cluster debug waitall
			{
				Parser::LineScanner is(is_.str());
				is.get_text(); // 先頭に書いてあった"cluster"

				strategy = make_unique<SingleEngineStrategy>();

				while (!is.eol())
				{
					string token = is.get_text();

					// debug mode
					if (token == "debug")
						debug_mode = true;

					else if (token == "nodes")
						options.nodes_limit = is.get_number(1000);

					else if (token == "skipinfo")
						skip_info = true;

					else if (token == "log")
						file_log  = true;

					else if (token == "ignore_setoption")
						options.ignore_setoption = true;

					else if (token == "mode")
					{
						token = is.get_text();
						if (token == "single")
							strategy = std::make_unique<SingleEngineStrategy>();
						else if (token == "ponder")
							strategy = std::make_unique<SinglePonderEngineStrategy>();
						else if (token == "optimistic")
							strategy = std::make_unique<OptimisticConsultationStrategy>();
						else if (token == "multi")
							strategy = std::make_unique<MultiPonderStrategy>();
						else if (token == "split")
							strategy = std::make_unique<RootSplitStrategy>();
						else if (token == "gps")
						{
							const size_t split_const = (size_t)is.get_number(3);
							strategy = std::make_unique<GpsClusterStrategy>(split_const);
						}
						else if (token == "gps2")
						{
							const size_t split_const = (size_t)is.get_number(3);
							strategy = std::make_unique<GpsClusterStrategy2>(split_const);
						}
						// ..
					}
				}
			}
		}

		// "cluster"のメインループ
		// USIプロトコルでGUI側から送られてくるコマンドとほぼ同じコマンドを理解できる。
		void message_loop_main(Position& pos, const ClusterOptions& options, unique_ptr<IClusterStrategy>& strategy)
		{
			// Clusterの監視者
			// コンストラクタで全エンジンが起動する。
			ClusterObserver observer(options , strategy);

			// 最後に"position"コマンドで送られてきた文字列("position")も含む。
			string lastPosition;

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

				// ==============================================
				// だいたいは送られてきたコマンドを
				// そのままClusterObserverに送れば良いと思う。
				// ==============================================

				else if (token == "usi")
					observer.send_wait(USI_Message::USI);
				else if (token == "setoption")
					// setoption は普通 isreadyの直前にしか送られてこないので
					// 何も考えずに エンジンにそのまま投げて問題ない。
					observer.send(Message(USI_Message::SETOPTION, cmd));
				else if (token == "position")
					// USI_Message::POSITIONは存在しない。局面は、GOコマンドに付随する。
					// "position XXXX ..."の"position"の文字わ剥がして保存する。
					lastPosition = strip_command(cmd);
				else if (token == "go")
					observer.send(Message(USI_Message::GO       , cmd, lastPosition /* 局面も付随して送ることになっている */));
				else if (token == "stop")
					observer.send(Message(USI_Message::STOP     , cmd));
				else if (token == "usinewgame")
					observer.send(USI_Message::USINEWGAME);
				else if (token == "gameover")
					observer.send(USI_Message::GAMEOVER);

				// ==============================================
				else if (token == "isready")
				{
					// ふかうら王のエンジン初期化(評価関数の読み込みなど)
					is_ready();

					// 先行してエンジンに"isready"コマンド送るべきかと思ったが、
					// すべてのエンジンから"readyok"が返ってくると自動的にGUIに対して
					// "readyok"を返答してしまうので、ふかうら王のエンジンの初期化が
					// 終わっていないことがある。

					observer.send_wait(USI_Message::ISREADY);
				}
				else if (token == "quit")
					break;
				// 拡張コマンド。途中でdebug出力がしたい時に用いる。
				else if (token == "debug")
					debug_mode = true;
				// 拡張コマンド。途中でdebug出力をやめたい時に用いる。
				else if (token == "nodebug")
					debug_mode = false;
				else {
					// "ponderhit"はサポートしていない。
					// "go ponderも送られてこないものと仮定している。
					
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

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
