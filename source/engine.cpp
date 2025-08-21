#include "engine.h"
#include "thread.h"
#include "perft.h"
#include "usioption.h"
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

// どのエンジンでも共通で必要なエンジンオプションを生やす。
// "NumaPolicy","DebugLogFile","DepthLimit", "NodesLimit", "DebugLogFile"
void Engine::add_base_options() {

    // NumaPolicy
    //   Numaの割り当て方針
    //
    // none       : 単一のNUMAノード、スレッドバインディングなしを想定。
    // system     : システムから利用可能なNUMA情報を使用し、それに応じてスレッドをバインドします。
    // auto       : デフォルト;システムに基づいてsystemとnoneを自動的に選択。
    // hardware   : 基盤ハードウェアからのNUMA情報を使用し、それに応じてスレッドをバインドし、
    //				以前のアフィニティをオーバーライドします。
    //				すべてのスレッドを使用しない場合（Windows 10やChessBaseなどの特定のGUIなど）に使用してください。
    // [[custom]] : NUMAドメインごとに利用可能なCPUを正確に指定します。
    //				':'はNUMAノードを区切り、','はCPUインデックスを区切ります。
    //				CPUインデックスには「最初-最後」の範囲構文をサポートします。
    //				例:0-15,32-47:16-31,48-63
    //
    // 🔍  https://github.com/official-stockfish/Stockfish/wiki/UCI-&-Commands#numapolicy

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          set_numa_config_from_option(o);
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    // ponderの有無
    // 📝 TimeManagementがこのoptionを持っていることを仮定している。
    // 🤔 思考Engineである以上はUSI_Ponderをサポートすべきだと思う。
    options.add(  //
      "USI_Ponder", Option(false, [this](const Option& o) {
          usi_ponder = o;
          return std::nullopt;
      }));

	// 確率的Ponder
	options.add(  //
      "Stochastic_Ponder", Option(false, [this](const Option& o) {
          stochastic_ponder = o;
		return std::nullopt;
	}));

    // 🤔 思考エンジンである以上、limits.depth, nodesには従うはずで、
    //     これを固定で制限する思考エンジンオプションはdefaultで生えてていいと思うんだよなー。

    // 探索深さ制限。0なら無制限。
    // 📝 "go"コマンドで、このオプションが指定されていたら、limits.depthのdefault値をこれに変更する。
    options.add(  //
      "DepthLimit", Option(0, 0, int_max));

    // 探索ノード制限。0なら無制限。
    // 📝 "go"コマンドで、このオプションが指定されていたら、limits.nodesのdefault値をこれに変更する。
    options.add(  //
      "NodesLimit", Option(0, 0, int64_max));

    // デバッグ用にログファイルへ書き出す。
    options.add(  //
      "DebugLogFile", Option("", [](const Option& o) {
          start_logger(o);
          return std::nullopt;
      }));
}

void Engine::add_options() {

    // 📌 最低限のoptionを生やす。
    //     これが要らなければ、このEngine classを派生させて、add_optionsをoverrideして、
    //     このadd_options()を呼び出さないようにしてください。
    // ⚠ だとして、その時にもresize_threads()は呼び出して、スレッド自体は生成するようにしてください。

    options.add(  //
      // 📝 やねうら王では default threadを4に変更する。
      //     過去にdefault設定のまま対局させて「やねうら王弱い」という人がいたため。
      "Threads", Option(4, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    // 基本オプションを生やす。
    add_base_options();

#if STOCKFISH
    // Stockfishには、探索部を初期化するエンジンオプションがあるが使わないので未サポートとする。
    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));
#endif

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


// blocking call to wait for search to finish
// 探索が完了のを待機する。(完了したらリターンする)
void Engine::wait_for_search_finished() {
#if !STOCKFISH
	// やねうら王では、まだスレッド初期化が終わっていない可能性がある。
	// スレッドが生成されていないとmain_thread()がないので、この場合、無視する。
    if (!threads.size())
        return;
#endif

	threads.main_thread()->wait_for_search_finished();
}

// "position"コマンドの下請け。
// sfen文字列 + movesのあとに書かれていた(USIの)指し手文字列から、現在の局面を設定する。
void Engine::set_position(const std::string& sfen, const std::vector<std::string>& moves) {

	// Drop the old state and create a new one
	// 古い状態を破棄して新しい状態を作成する

	states = StateListPtr(new std::deque<StateInfo>(1));
	pos.set(sfen /*, options["UCI_Chess960"]*/ , &states->back());

#if !STOCKFISH
    std::vector<Move> moves0;
#endif

	for (const auto& move : moves)
	{
		auto m = USIEngine::to_move(pos, move);

		if (m == Move::none())
			break;

		states->emplace_back();
		pos.do_move(m, states->back());

#if !STOCKFISH
		moves0.emplace_back(m);
#endif
	}

#if !STOCKFISH
	// 🌈 やねうら王では、ここに保存しておくことになっている。
    game_root_sfen = sfen;
	moves_from_game_root = std::move(moves0);
#endif

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

void Engine::search_clear() {
#if STOCKFISH
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();

    // @TODO wont work with multiple instances
    Tablebases::init(options["SyzygyPath"]);  // Free mapped files
#else
	// benchコマンドから内部的に呼び出す。
    wait_for_search_finished();
    isready();
#endif
}

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

void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
    //onVerifyNetworks = std::move(f);
	// TODO : あとで
}

#if !STOCKFISH
void Engine::set_on_update_string(std::function<void(std::string_view)>&& f) {
    updateContext.onUpdateString = std::move(f);
}

std::function<void(std::string_view, std::string_view)> Engine::get_on_bestmove() {
    return updateContext.onBestmove;
}
#endif

void Engine::resize_threads() {

	// 📌 探索の終了を待つ
	threads.wait_for_search_finished();

	// 📌 スレッド数のリサイズ

#if STOCKFISH
	threads.set(numaContext.get_numa_config(), { options, threads, tt, networks }, updateContext);
#else

	// 🌈  やねうら王ではここでWorkerFactoryを渡すように変更。
	//    これにより、生成Worker(Worker派生class)をEngine派生classで選択できる。

	// Engine派生classが"Threads"オプションを用意していない。
	// Engine派生class側のresize_threads()かThreadPool::set()が直接が呼び出されるべき。
	if (!options.count("Threads"))
        return;

	auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken)
		{ return std::make_unique<Search::Worker>(options, threads, threadIdx, numaAccessToken); };
    threads.set(numaContext.get_numa_config(), options, options["Threads"], worker_factory);
#endif

	// 📌 置換表の再割り当て。

#if STOCKFISH
	// Reallocate the hash with the new threadpool size
	// 新しいスレッドプールのサイズに合わせてハッシュを再割り当てする
	set_tt_size(options["Hash"]);
	//  ⇨  EngineがTTを持っているとは限らないので、やねうら王ではこの部分を分離したい。
#endif

	// 📌 NUMAの設定

	// スレッドの用いる評価関数パラメーターが正しいNUMAに属するようにする
	threads.ensure_network_replicated();
}

void Engine::set_tt_size(size_t mb) {
#if STOCKFISH
	wait_for_search_finished();
    tt.resize(mb, threads);
#endif
    // 🌈 やねうら王ではEngine classはTTを持たない。派生class側で処理する。
}

void Engine::set_ponderhit(bool b) {
#if STOCKFISH
	threads.main_manager()->ponder = b;
#endif
    // 🌈 やねうら王ではThreadPool classはmain_managerを持たない。Engine派生class側で処理する。
}

// network related


// 🚧 工事中 🚧


// utility functions

void Engine::trace_eval() const {
	// 🌈 やねうら王では、Engine派生classで定義する。
#if STOCKFISH
	StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;

	p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());

    verify_networks();
	sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
#endif
}

#if !STOCKFISH
Value Engine::evaluate() const { return VALUE_NONE; }
#endif

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

// 現在の局面のsfen形式の表現を取得する。
#if STOCKFISH
std::string Engine::fen() const { return pos.fen(); }
#else
std::string Engine::sfen() const { return pos.sfen(); }
#endif

// 盤面を180°回転させる。
void Engine::flip() { pos.flip(); }

// 局面を視覚化した文字列を取得する。
std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

#if STOCKFISH
int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }
#else
int Engine::get_hashfull(int maxAge) const { return 0; }
#endif

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

// --------------------
//  やねうら王独自拡張
// --------------------

// 💡 USIで"isready"に対して時間のかかる処理を実行したい時に用いる。
void Engine::run_heavy_job(std::function<void()> job) {
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

// ----------------------------------------------
// 📌 Engineのentry pointを登録しておく仕組み 📌
// ----------------------------------------------

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
