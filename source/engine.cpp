#include "engine.h"
#include "thread.h"
#include "perft.h"
#include "usi_option.h"
#include "book/book.h"
#include "search.h"

namespace YaneuraOu {

Engine::Engine() :
	numaContext(NumaConfig::from_system()),
	states(new std::deque<StateInfo>(1)),
	threads()
{

#if !defined(USE_CLASSIC_EVAL)
	// 局面は平手の開始局面にしておく。
	pos.set(StartSFEN, &states->back());
	// ⚠ CLASSIC EVALは、Position::set()の途中でcompute_eval()を呼び出すので
	//     その時に評価関数の初期化がなされるが、その時点では評価関数の読み込みが
	//     完了していないので、アクセス違反で落ちる。
#endif

	//resize_threads();
	// ⚠ スレッドは置換表のクリアなどで必要になるので、
	//     このタイミングでresize_threads()を呼び出すことで、
	//      options["Threads"]の設定を仮に反映させたいのだが、
	//     派生classのコンストラクタの初期化が終わっていないので、ここから呼び出しても
	//     派生class側のresize_threads()が呼び出されない。
	//     そこで仕方がないのでadd_options()のタイミングでresize_threads()を呼び出すことにする。

}

void Engine::usi()
{
#if STOCKFISH
    sync_cout << "id name " << engine_info(true) << "\n" << engine.get_options() << sync_endl;
    sync_cout << "uciok" << sync_endl;
#else
    sync_cout << "id name "
              << engine_info(get_engine_name(), get_engine_author(), get_engine_version(),
                             get_eval_name())
              << get_options() << sync_endl;

    sync_cout << "usiok" << sync_endl;
#endif
}

void Engine::add_options()
{
	// 📌 最低限のoptionを生やす。
	//     これが要らなければ、このEngine classを派生させて、add_optionsをoverrideして、
	//     このadd_options()を呼び出さないようにしてください。
	// ⚠ だとして、その時にもresize_threads()は呼び出して、スレッド自体は生成するようにしてください。

	options.add(
		// 📝 やねうら王では default threadを4に変更する。
		//     過去にdefault設定のまま対局させて「やねうら王弱い」という人がいたため。
		"Threads", Option(4, 1, MaxThreads, [this](const Option&) {
			resize_threads();
			//return thread_allocation_information_as_string();
			return std::nullopt;
		}));

	// NumaPolicy
	//   Numaの割り当て方針
	// 
	// auto     : 自動
	// system   : OS任せ
	// hardware : hardwareに従う
	// none     : なし

	options.add(
		"NumaPolicy", Option("auto", [this](const Option& o) {
			set_numa_config_from_option(o);
			//return numa_config_information_as_string() + "\n"
			//	+ thread_allocation_information_as_string();
			return std::nullopt;
		}));

	// ponderの有無
	// 📝 TimeManagementがこのoptionを持っていることを仮定している。
	// 🤔 思考Engineである以上はUSI_Ponderをサポートすべきだと思う。
    options.add("USI_Ponder", Option(false));

	// 🤔 思考エンジンである以上、limits.depth, nodesには従うはずで、
	//     これを固定で制限する思考エンジンオプションはdefaultで生えてていいと思うんだよなー。

	// 探索深さ制限。0なら無制限。
	// 📝 "go"コマンドで、このオプションが指定されていたら、limits.depthのdefault値をこれに変更する。
    options.add("DepthLimit", Option(0, 0, int_max));

    // 探索ノード制限。0なら無制限。
    // 📝 "go"コマンドで、このオプションが指定されていたら、limits.nodesのdefault値をこれに変更する。
    options.add("NodesLimit", Option(0, 0, int64_max));

	// このタイミングで"Threads"の設定を仮に反映させる。
	// 📝 Threadsを1以上にしておかないと、このあと置換表のクリアなど、
	//     複数スレッドを用いて行うことができなくなるため。
	// ⚠ ここで、派生class側のresize_threads()ではなく、
	//	   このclassのresize_threads()を呼び出すことに注意。
	//     派生class側のresize_threads()は、"USI_Hash"を参照して
	//     置換表を初期化するコードが書かれているかもしれないが、
	//     いま時点では、"USI_Hash"のoptionをaddしていないのでエラーとなる。
	Engine::resize_threads();
}

// NumaConfig(numaContextのこと)を Options["NumaPolicy"]の値 から設定する。
void Engine::set_numa_config_from_option(const std::string& o) {
	if (o == "auto" || o == "system")
	{
		numaContext.set_numa_config(NumaConfig::from_system());
	}
	else if (o == "hardware")
	{
		// Don't respect affinity set in the system.
		numaContext.set_numa_config(NumaConfig::from_system(false));
	}
	else if (o == "none")
	{
		numaContext.set_numa_config(NumaConfig{});
	}
	else
	{
		numaContext.set_numa_config(NumaConfig::from_string(o));
	}

	// Force reallocation of threads in case affinities need to change.
	resize_threads();
	threads.ensure_network_replicated();
}

// 局面を視覚化した文字列を取得する。
std::string Engine::visualize() const {
	std::stringstream ss;
	ss << pos;
	return ss.str();
}

// blocking call to wait for search to finish
// 探索が完了のを待機する。(完了したらリターンする)
void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

// "position"コマンドの下請け。
// sfen文字列 + movesのあとに書かれていた(USIの)指し手文字列から、現在の局面を設定する。
void Engine::set_position(const std::string& sfen, const std::vector<std::string>& moves) {

	// Drop the old state and create a new one
	// 古い状態を破棄して新しい状態を作成する

	states = StateListPtr(new std::deque<StateInfo>(1));
	pos.set(sfen /*, options["UCI_Chess960"]*/ , &states->back());

	for (const auto& move : moves)
	{
		auto m = USIEngine::to_move(pos, move);

		if (m == Move::none())
			break;

		states->emplace_back();
		pos.do_move(m, states->back());
	}
}


#if 0
void Engine::usinewgame()
{
	wait_for_search_finished();

	//tt.clear(threads);
	threads.clear();

	// @TODO wont work with multiple instances
	//Tablebases::init(options["SyzygyPath"]);  // Free mapped files
	// 📌 将棋ではTablebasesは用いない。
}
#endif

void Engine::isready()
{
	// エンジン設定のスレッド数を反映させる。
	resize_threads();

	sync_cout << "readyok" << sync_endl;
}

std::uint64_t Engine::perft(const std::string& fen, Depth depth /*, bool isChess960 */) {
	verify_networks();

	return Benchmark::perft(fen, depth /*, isChess960 */);
}


void Engine::go(Search::LimitsType& limits) {
	ASSERT_LV3(limits.perft == 0);
	//verify_networks();

	threads.start_thinking(options, pos, states, limits);
}

void Engine::stop() { threads.stop = true; }

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_update_string(std::function<void(std::string_view)>&& f) {
    updateContext.onUpdateString = std::move(f);
}

void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
    //onVerifyNetworks = std::move(f);
	// TODO : あとで
}


void Engine::resize_threads() {

	// 📌 探索の終了を待つ
	threads.wait_for_search_finished();

	// 📌 スレッド数のリサイズ

	//threads.set(numaContext.get_numa_config(), { options, threads, tt, networks }, updateContext);

	// ⇨  やねうら王ではここでWorkerFactoryを渡すように変更。
	//    これにより、生成Worker(Worker派生class)をEngine派生classで選択できる。

	auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken)
		{ return std::make_unique<Search::Worker>(options, threads, threadIdx, numaAccessToken); };
        threads.set(numaContext.get_numa_config(), options, options["Threads"], worker_factory);

	// 📌 置換表の再割り当て。

	// Reallocate the hash with the new threadpool size
	// 新しいスレッドプールのサイズに合わせてハッシュを再割り当てする
	//set_tt_size(options["USI_Hash"]);
	//  ⇨  EngineがTTを持っているとは限らないので、この部分を分離したい。

	// 📌 NUMAの設定

	// スレッドの用いる評価関数パラメーターが正しいNUMAに属するようにする
	threads.ensure_network_replicated();
}

// 💡 USIで"isready"に対して時間のかかる処理を実行したい時に用いる。
void Engine::run_heavy_job(std::function<void()> job)
{
    // --- Keep Alive的な処理 ---

    // "isready"を受け取ったあと、"readyok"を返すまで5秒ごとに改行を送るように修正する。(keep alive的な処理)
    // →　これ、よくない仕様であった。
    // cf. USIプロトコルでisready後の初期化に時間がかかる時にどうすれば良いのか？
    //     http://yaneuraou.yaneu.com/2020/01/05/usi%e3%83%97%e3%83%ad%e3%83%88%e3%82%b3%e3%83%ab%e3%81%a7isready%e5%be%8c%e3%81%ae%e5%88%9d%e6%9c%9f%e5%8c%96%e3%81%ab%e6%99%82%e9%96%93%e3%81%8c%e3%81%8b%e3%81%8b%e3%82%8b%e6%99%82%e3%81%ab%e3%81%a9/
    // cf. isready後のkeep alive用改行コードの送信について
    //		http://yaneuraou.yaneu.com/2020/03/08/isready%e5%be%8c%e3%81%aekeep-alive%e7%94%a8%e6%94%b9%e8%a1%8c%e3%82%b3%e3%83%bc%e3%83%89%e3%81%ae%e9%80%81%e4%bf%a1%e3%81%ab%e3%81%a4%e3%81%84%e3%81%a6/

    // これを送らないと、将棋所、ShogiGUIでタイムアウトになりかねない。
    // ワーカースレッドを一つ生成して、そいつが5秒おきに改行を送信するようにする。
    // このあと重い処理を行うのでスレッドの起動が遅延する可能性があるから、先にスレッドを生成して、そのスレッドが起動したことを
    // 確認してから処理を行う。

    // スレッドが起動したことを通知するためのフラグ
    auto thread_started = false;

    // この関数を抜ける時に立つフラグ(スレッドを停止させる用)
    auto thread_end = false;

    // 定期的な改行送信用のスレッド
    auto th = std::thread([&] {
        // スレッドが起動した
        thread_started = true;

        int count = 0;
        while (!thread_end)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (++count >= 50 /* 5秒 */)
            {
                count = 0;
                sync_cout << sync_endl;  // 改行を送信する。

                // 定跡の読み込み部などで"info string.."で途中経過を出力する場合、
                // sync_cout ～ sync_endlを用いて送信しないと、この改行を送るタイミングとかち合うと
                // 変なところで改行されてしまうので注意。
            }
        }
    });
    SCOPE_EXIT({
        thread_end = true;
        th.join();
    });

    // スレッド起動待ち
    while (!thread_started)
        Tools::sleep(100);

    // --- Keep Alive的な処理ここまで ---

	// 評価関数の読み込みなど時間のかかるであろう処理はこのタイミングで行なう。
    // 起動時に時間のかかる処理をしてしまうと将棋所がタイムアウト判定をして、思考エンジンとしての認識をリタイアしてしまう。
    job();
}


#if 0

// 開始局面
//constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// 📌 やねうら王では、StartSFENを type.h で宣言している。



YaneuraOuEngine::YaneuraOuEngine(/* std::optional<std::string> path */) :
	//binaryDirectory(path ? CommandLine::get_binary_directory(*path) : ""),
	numaContext(NumaConfig::from_system()),
	states(new std::deque<StateInfo>(1)),
	threads(),
	networks(numaContext)
	// TODO : あとで

	// networks(
	//	numaContext,
	//	NN::Networks(
	//		NN::NetworkBig({ EvalFileDefaultNameBig, "None", "" }, NN::EmbeddedNNUEType::BIG),
	//		NN::NetworkSmall({ EvalFileDefaultNameSmall, "None", "" }, NN::EmbeddedNNUEType::SMALL)))
{
	// 定跡DB classの初期化
	book = std::make_shared<Book::BookMoveSelector>();
	book->init(options);

	// 局面を初期局面に設定する。
	pos.set(StartSFEN, &states->back());



#if defined(YANEURAOU_ENGINE)
	constexpr int HashMB = 1024;
#elif defined(TANUKI_MATE_ENGINE)
	constexpr int HashMB = 4096;
#elif defined(YANEURAOU_MATE_ENGINE)
	constexpr int HashMB = 1024;
#else
	// other engine
	constexpr int HashMB = 16; // maybe not used
#endif

	// 置換表のサイズ。[MB]で指定。
	options.add(  //
		"USI_Hash", Option(HashMB, 1, MaxHashMB, [this](const Option& o) {
			set_tt_size(o);
			return std::nullopt;
			}));

#if defined(USE_EVAL_HASH)
	// 評価値用のcacheサイズ。[MB]で指定。

#if defined(FOR_TOURNAMENT)
		// トーナメント用は少し大きなサイズ
	o["EvalHash"] << Option(1024, 1, MaxHashMB, [](const Option& o) { Eval::EvalHash_Resize(o); });
#else
	o["EvalHash"] << Option(128, 1, MaxHashMB, [](const Option& o) { Eval::EvalHash_Resize(o); });
#endif // defined(FOR_TOURNAMENT)
#endif // defined(USE_EVAL_HASH)


#if 0

	// Stockfishには、探索部を初期化するエンジンオプションがあるが使わないので未サポートとする。
	//o["Clear Hash"]            << Option(on_clear_hash);

	// 確率的ponder , defaultでfalseにしとかないと、読み筋の表示がおかしくなって、初心者混乱する。
	o["Stochastic_Ponder"] << USI::Option(false);



#if defined(__EMSCRIPTEN__) && defined(EVAL_NNUE)
	// WASM NNUE
	const char* default_eval_file = "nn.bin";
	last_eval_file = default_eval_file;
	o["EvalFile"] << Option(default_eval_file, [](const USI::Option& o) {
		if (last_eval_file != std::string(o))
		{
			// 評価関数ファイル名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
			last_eval_file = std::string(o);
			load_eval_finished = false;
		}
		});
#endif

	// cin/coutの入出力をファイルにリダイレクトする
	o["WriteDebugLog"] << Option("", [](const Option& o) { on_logger(o); });

	// 読みの各局面ですべての合法手を生成する
	// (普通、歩の2段目での不成などは指し手自体を生成しないが、
	// これのせいで不成が必要な詰みが絡む問題が解けないことがあるので、このオプションを用意した。)
#if defined(TANUKI_MATE_ENGINE) || defined(YANEURAOU_MATE_ENGINE)
		// 詰将棋エンジンではデフォルトでオン。
	o["GenerateAllLegalMoves"] << Option(true);
#else
		// 通常探索エンジンではデフォルトでオフ。
	o["GenerateAllLegalMoves"] << Option(false);
#endif

#if defined (USE_ENTERING_KING_WIN)
	// 入玉ルール
	o["EnteringKingRule"] << Option(USI::ekr_rules, USI::ekr_rules[EKR_27_POINT]);
#endif

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32) && \
	 (defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) )
	// 評価関数パラメーターを共有するか。
	// デフォルトで有効に変更。(V4.90～)
	o["EvalShare"] << Option(true);
#endif

#if defined(EVAL_LEARN)
	// isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
	// test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
	// このコマンドの実行前に異常終了してしまう。
	// そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
	// test evalconvertコマンドを叩く。
	o["SkipLoadingEval"] << Option(false);
#endif

#if defined(_WIN64)
	// LargePageを有効化するか。
	// これを無効化できないと自己対局の時に片側のエンジンだけがLargePageを使うことがあり、
	// 不公平になるため、無効化する方法が必要であった。
	o["LargePageEnable"] << Option(true);
#endif

	// 各エンジンがOptionを追加したいだろうから、コールバックする。
	USI::extra_option(o);

	// コンパイル時にエンジンオプションが指定されている。
#if defined(ENGINE_OPTIONS)
	const std::string opt = ENGINE_OPTIONS;
	set_engine_options(opt);
#endif

	// カレントフォルダに"engine_options.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
	read_engine_options("engine_options.txt");


#endif

	// 並列探索するときのスレッド数
	// CPUの搭載コア数をデフォルトとすべきかも知れないが余計なお世話のような気もするのでしていない。

#if !defined(YANEURAOU_ENGINE_DEEP)

		// ※　やねうら王独自改良
		// スレッド数の変更やUSI_Hashのメモリ確保をそのハンドラでやってしまうと、
		// そのあとLargePageEnableを送られても困ることになる。
		// ゆえにこれらは、"isready"に対する応答で行うことにする。
		// そもそもで言うとsetoptionに対してそんなに時間のかかることをするとGUI側がtimeoutになる懸念もある。
		// Stockfishもこうすべきだと思う。

#if !defined(__EMSCRIPTEN__)
	options.add(
		// 📝 やねうら王では default threadを4に変更する。
		//     過去にdefault設定のまま対局させて「やねうら王弱い」という人がいたため。
		"Threads", Option(4, 1, MaxThreads, [this](const Option&) {
			resize_threads();
			return thread_allocation_information_as_string();
			}));

#else
		// yaneuraou.wasm
		// スレッド数などの調整
		// stockfish.wasmの数値を基本的に使用している
	options.add(  //
		"Threads", Option(1, 1, 32, [this]([[maybe_unused]] const Option&) {
			resize_threads();
			return thread_allocation_information_as_string();
			}));

#endif
#endif

	// 評価関数フォルダと評価関数ファイル名。
	// これらを変更したとき、評価関数を次のisreadyタイミングで読み直す必要がある。
	// 📝 これらのhandlerは存在しない。
	//     verify_networks()のなかで前回のpathと違うなら読み直す。

#if defined(EVAL_EMBEDDING)
	const char* default_eval_dir = "<internal>";
#elif !defined(__EMSCRIPTEN__)
	const char* default_eval_dir = "eval";
#else
		// WASM
	const char* default_eval_dir = ".";
#endif
	options.add("EvalDir", Option(default_eval_dir));

#if defined(YANEURAOU_ENGINE_NNUE)
	const char* default_eval_file = "nn.bin";
#elif defined(USER_ENGINE)
	const char* default_eval_file = "eval.bin";
#else
	const char* default_eval_file = "eval.bin";
#endif
	options.add("EvalFile", Option(default_eval_file));


	// 📌 やねうら王独自
	//     各エンジン向けに、追加でオプションを生やす。
	extra_option();

	// 📝 Optionのhandlerは options.add()の時点では呼び出されない。
	//     そこで、反映が必要なhandlerはここで呼び出してやる。
	//     ここでOption名を指定してhandlerだけ呼び出せたほうが良いのではなかろうか。

	//load_networks();
	resize_threads();
}

// かきかけ


// utility functions

void Engine::trace_eval() const {
	StateListPtr trace_states(new std::deque<StateInfo>(1));
	Position     p;
	p.set(pos.sfen() /*, options["UCI_Chess960"]*/, &trace_states->back());

	verify_networks();
	//sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap& Engine::get_options()             { return options; }

// 現在の局面のsfen形式の表現を取得する。
std::string Engine::sfen() const { return pos.sfen(); }

// 盤面を180°回転させる。
void Engine::flip() { /* pos.flip(); */ }

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
	updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
	updateContext.onBestmove = std::move(f);
}

// verify_network()を呼び出した時に、NN::network.verify()からcallbackされるfunctionを設定する。
void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
	onVerifyNetworks = std::move(f);
}



int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }


// modifiers

void Engine::set_tt_size(size_t mb) {
	wait_for_search_finished();

#if defined(TANUKI_MATE_ENGINE) || defined(YANEURAOU_MATE_ENGINE) || defined(YANEURAOU_ENGINE_DEEP)
	//　これらのengineでは、この標準TTを用いない。
	return;
#endif

	tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() const {
	//networks->big.verify(options["EvalFile"], onVerifyNetworks);
	//networks->small.verify(options["EvalFileSmall"], onVerifyNetworks);

	auto& path = Path::Combine(options["EvalDir"], options["EvalFile"]);
	networks->verify(path, onVerifyNetworks);
}

void Engine::load_networks() {

	networks.modify_and_replicate([this](Eval::Evaluator& networks_) {
		//networks_.big.load(binaryDirectory, options["EvalFile"]);
		//networks_.small.load(binaryDirectory, options["EvalFileSmall"]);

		auto path = Path::Combine(this->options["EvalDir"], this->options["EvalFile"]);
		networks_.load(path);
	});

	threads.clear();
	threads.ensure_network_replicated();
}

//void Engine::load_big_network(const std::string& file) {
//	networks.modify_and_replicate(
//		[this, &file](NN::Networks& networks_) { networks_.big.load(binaryDirectory, file); });
//	threads.clear();
//	threads.ensure_network_replicated();
//}
//
//void Engine::load_small_network(const std::string& file) {
//	networks.modify_and_replicate(
//		[this, &file](NN::Networks& networks_) { networks_.small.load(binaryDirectory, file); });
//	threads.clear();
//	threads.ensure_network_replicated();
//}

void Engine::save_network(/*const std::pair<std::optional<std::string>, std::string> files[2]*/ const std::string& filename) {
	//networks.modify_and_replicate([&files](Eval::Evaluator& networks_) {
	//	networks_.big.save(files[0].first);
	//	networks_.small.save(files[1].first);
	//	});
	networks.modify_and_replicate([&filename](Eval::Evaluator& networks_) {
		networks_.save(filename);
		});
}


std::vector<std::pair<size_t, size_t>> Engine::get_bound_thread_count_by_numa_node() const {
	auto                                   counts = threads.get_bound_thread_count_by_numa_node();
	const NumaConfig& cfg = numaContext.get_numa_config();
	std::vector<std::pair<size_t, size_t>> ratios;
	NumaIndex                              n = 0;
	for (; n < counts.size(); ++n)
		ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
	if (!counts.empty())
		for (; n < cfg.num_numa_nodes(); ++n)
			ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
	return ratios;
}

std::string Engine::get_numa_config_as_string() const {
	return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
	auto cfgStr = get_numa_config_as_string();
	return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
	auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
	std::stringstream ss;
	if (boundThreadsByNode.empty())
		return ss.str();

	bool isFirst = true;

	for (auto&& [current, total] : boundThreadsByNode)
	{
		if (!isFirst)
			ss << ":";
		ss << current << "/" << total;
		isFirst = false;
	}

	return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
	std::stringstream ss;

	size_t threadsSize = threads.size();
	ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

	auto boundThreadsByNodeStr = thread_binding_information_as_string();
	if (boundThreadsByNodeStr.empty())
		return ss.str();

	ss << " with NUMA node thread binding: ";
	ss << boundThreadsByNodeStr;

	return ss.str();
}

#endif

// --------------------
//  やねうら王独自拡張
// --------------------


// 📌 Engineのentry pointを登録しておく仕組み 📌

using EngineEntry = std::tuple<std::function<void()>, std::string, int>;

// エンジンの共通の登録先
// 📝 static EngineFuncRegister reg_a(engine_main_a, 1); のようにしてengine_main_a()を登録する。
//     USER_ENGINEであるuser-engine.cpp を参考にすること。
static std::vector<EngineEntry>& engineFuncs() {
	// 💡 関数のなかのstatic変数は最初に呼び出された時に初期化されることが保証されている。
	//     なので、初期化順の問題は発生しない。
	static std::vector<EngineEntry> funcs;
	return funcs;
}

// エンジンの登録用のヘルパー
EngineFuncRegister::EngineFuncRegister(std::function<void()> f, const std::string& engine_name, int priority)
{
	engineFuncs().push_back({ f , engine_name, priority });
}

// EngineFuncRegisterで登録されたEngineのうち、priorityの一番高いエンジンを起動する。
void run_engine_entry()
{
	auto& v = engineFuncs();
	// priorityの最大
	EngineEntry* m = nullptr;
	for (auto& entry : v)
	{
		//sync_cout << "info string engine name = " << std::get<1>(entry) << ", priority = " << std::get<2>(entry) << sync_endl;
		if (!m || std::get<2>(*m) < std::get<2>(entry))
		{
			m = &entry;
		}
	}

	// priority最大のentry pointを開始する。
	if (m == nullptr) {
		sync_cout << "Error: no engine entry point." << sync_endl;
		Tools::exit();
	}
	else {
		//sync_cout << "info string startup engine = " << std::get<1>(*m) << sync_endl;
		std::get<0>(*m)(); // このエンジンを実行
	}
}

} // namespace YaneuraOu
