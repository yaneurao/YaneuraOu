#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
//#include "nnue/network.h"
#include "numa.h"
#include "position.h"
#include "search.h"
//#include "syzygy/tbprobe.h"  // for Stockfish::Depth
#include "thread.h"
#include "tt.h"
#include "usioption.h"

namespace YaneuraOu {

// --------------------
//       定数
// --------------------

// 最大スレッド数
static int    MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

// 最大USI_Hash
#if !defined(__EMSCRIPTEN__)
// Hash上限。32bitモードなら2GB、64bitモードなら32TB
constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;
#else
// yaneuraou.wasm
// メモリの調整
// stockfish.wasmの数値を基本的に使用している
constexpr int MaxHashMB = 2048;
#endif

// 前方宣言
namespace Book { struct BookMoveSelector; }


// 思考エンジンのinterface
class IEngine
{
public:
	/*
		📌 自作エンジンでoverrideすると良いmethod。📌

		💡 Engineは、思考エンジン1インスタンスにつき、1インスタンス。
		    Worker, Threadは、探索スレッド1つにつき1インスタンス。

		add_options()
			エンジンに追加オプションを設定したいときは、この関数をoverrideする。
			📝 GetOptions()->add()を用いて、Optionを追加する。
			💡 Engine::add_options()で"Threads","NumaPolicy"などのエンジンオプションを生やしているので
			    これらが必要なのであれば、add_options()をoverrideしてEngine::add_options()を呼び出すこと。

		resize_threads()
			options["Threads"]やoptions["NumaPolicy"]が変更になった時に呼び出され、
			スレッドを必要な数だけ再生成するhandler。
			ここでスレッド生成のためにThreadPool::set()を呼び出しており、その時に
			Worker派生classのfactoryを渡す必要がある。
			この部分を変更することによって、生成するWorker派生classを変更することができる。

		set_tt_size()
			options["USI_Hash"]などの置換表サイズに対して、それが変更された時に呼び出されるhandler。
			置換表的なものを使用するときは、これをoverrideすると便利。

		isready()
			"isready"コマンドが送られてきた時の応答。
	        評価関数パラメーターの読み込みや、置換表の初期化など時間のかかる処理はここに書く。

		usinewgame()
			"usinewgame"コマンドが送られてきた時の応答。
			毎局、開始時にGUI側から送られてくるので1局ごとに行う探索部の初期化などはここに書く。

		verify_networks()
			評価関数部の初期化が行えているかのチェックを行う。

		save_network()
			評価関数パラメーターを保存する。

	*/

	// 📌 読み筋の表現 📌

    using InfoShort  = Search::InfoShort;
	using InfoFull   = Search::InfoFull;
	using InfoIter   = Search::InfoIteration;

#if STOCKFISH
        // 💡 StockfishではEngineのコンストラクタでは、起動path(main関数で渡されたargv[0])を渡す。
        //     やねうら王では、起動フォルダはCommandLine::gから取得できるので不要。
        IEngine(std::optional<std::string> path = std::nullopt);
#else
        IEngine() {}
#endif

	// Cannot be movable due to components holding backreferences to fields
    // フィールドへの逆参照を持つコンポーネントがあるため、ムーブ可能にはできない

    IEngine(const IEngine&)            = delete;
    IEngine(IEngine&&)                 = delete;
    IEngine& operator=(const IEngine&) = delete;
    IEngine& operator=(IEngine&&)      = delete;

	// 💡 interfaceなので仮想デストラクタが必要
	//     また、終了するときにwait_for_search_finished()を呼び出して探索が終了するのを待つ。
    virtual ~IEngine() { wait_for_search_finished(); }

	// "perft"コマンド。perftとは、performance testの略。
#if STOCKFISH
    std::uint64_t perft(const std::string& fen, Depth depth, bool isChess960);
#else
    virtual std::uint64_t perft(const std::string& fen, Depth depth) = 0;
#endif

	// "go"コマンド。ThreadPoolのmain threadに対して探索を開始(start_searching)する。
    // non blocking call to start searching
    // 探索を開始する。(non blocking呼び出し)
    virtual void go(Search::LimitsType& limits) = 0;

    // "stop"コマンド。ThreadPoolに対してstop信号を送信する。
    // non blocking call to stop searching
    // 探索を停止させる。(non blocking呼び出し)
    virtual void stop() = 0;

	// blocking call to wait for search to finish
    // 探索が完了のを待機する。(blocking呼び出し)
    // 📝 ThreadPoolのmain_threadの完了を待機している。
	// 💡 「blocking呼び出し」とは、完了するまでreturnしないような関数のこと。
    virtual void wait_for_search_finished() {};

	// set a new position, moves are in UCI format
    // 新しい局面を設定する。手はUCI形式で指定される
    // 💡 "position"コマンドの下請け。
    //      sfen文字列 + movesのあとに書かれていた(USIの)指し手文字列から、現在の局面を設定する。
    virtual void set_position(const std::string& sfen, const std::vector<std::string>& moves) = 0;

    // modifiers

	// NumaConfigをエンジンオプションの"NumaPolicy"から設定する。
    virtual void set_numa_config_from_option(const std::string& o) = 0;

	// options["Threads"]やoptions["NumaPolicy"]が変更になった時に呼び出され、
    // スレッドを必要な数だけ再生成するhandler。
    //
    // ここでスレッド生成のためにThreadPool::set()を呼び出しており、その時に
    // Worker派生classのfactoryを渡す必要がある。この部分を変更することによって、
    // 生成するWorker派生classを変更することができる。
    //
    // 💡 Worker::resize_threads()がその処理なので、Worker::resize_threads()の実装を参考にすること。
    //     また、USER_ENGINEの実装(user-engine.cpp)も参考にすること。
    virtual void resize_threads() = 0;

	// options["USI_Hash"]などの置換表サイズに対して、それが変更された時に呼び出されるhandler。
    // 置換表的なものを使用するときは、これをoverrideすると便利。
	virtual void set_tt_size(size_t mb) = 0;

	// USIの"ponderhit"に対するhandler。
	// 💡 "ponderhit"に対して set_ponderhit(false)として呼び出される。
    // 📓 Engine派生classを"ponderhit"に対応させるには、これをEngine派生class側でoverrideして、
    //    "ponderhit"に対してmain_manager()->ponder = false;にするなどの処理が必要である。
    // 🤔 このbool、どう見ても不要なのだが…。
    virtual void set_ponderhit(bool b) = 0;

	// "ucinewgame"に対してWorkerを初期化する。(tt.clear(), threads.clear()を呼び出す)
	//	🤔 将棋だと"isready"に対するhandlerで処理したほうがいいと思う。
	//      "bench"コマンドから内部的に呼び出すので用意しておく。
	virtual void search_clear() = 0;

	// 📌 読み筋の出力など、event handlerのsetter
    //     例えば、verify_network()を呼び出した時に、NN::network.verify()からcallbackされるfunctionを設定する。
    // 📝 InfoShort, InfoFull, InfoIterなどの意味については、search.hにある定義に書いてある。

    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&&) {}
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&&) {}
    virtual void set_on_iter(std::function<void(const InfoIter&)>&&) {}
    virtual void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&) {}
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&&) {}

#if !STOCKFISH
    virtual void set_on_update_string(std::function<void(std::string_view)>&&) {}

	// 🌈 待避させたいことがあるので、getterも用意しておく。
    virtual std::function<void(std::string_view, std::string_view)> get_on_bestmove() {
        return [](auto, auto) {};
    }
#endif

    // network related
	// ネットワーク(評価関数)関連

	// ネットワークの読み込みができているか確認する。
	// 読み込まれていなければload_networks()で読み込む。
    virtual void verify_networks() const {}

	// TODO あとで考える。
#if STOCKFISH
    // 評価関数の読み込み
    void load_networks();

	// Stockfishは大きいnetworkと小さいnetworkとがある。それぞれの読み込み。

    void load_big_network(const std::string& file);
    void load_small_network(const std::string& file);
#endif
    // 評価関数パラメーターを保存する。
    // "export_net"コマンドに対して呼び出される。
#if STOCKFISH
    void         save_network(const std::pair<std::optional<std::string>, std::string> files[2]);
#else
    // 評価関数をファイルに保存する。
    virtual void save_network(const std::string& path) = 0;
#endif

    // utility functions
	// ユーティリティ関数

    // "trace"コマンド。現在の局面に対して評価関数を呼び出して評価値の詳細を出力する。
    virtual void trace_eval() const = 0;

    // 🌈 "e"コマンド。現在の局面に対して評価関数を呼び出して評価値を返す。
    virtual Value evaluate() const = 0;

	// OptionsMap(エンジンオプション)を取得

    virtual const OptionsMap& get_options() const = 0;
    virtual OptionsMap&       get_options()       = 0;

	// 置換表サイズの取得。
	// 🤔 置換表はやねうら王ではEngine基底classでは関与しないが、
	//	   "bench"コマンドから呼び出されるので用意しておく。
    virtual int get_hashfull(int maxAge = 0) const = 0;

	// (探索中の)現在の局面のsfen文字列を返す。
    virtual std::string sfen() const = 0;

	// 盤面を180°回転させる。デバッグ用
    virtual void flip() = 0;

    // 局面を視覚化した文字列を取得する。(デバッグ用)
    virtual std::string visualize() const = 0;

	// スレッド割り当てをした時などのメッセージ出力用handler。

	virtual std::vector<std::pair<size_t, size_t>> get_bound_thread_count_by_numa_node() const {
        return std::vector<std::pair<size_t, size_t>>();
    }
	virtual std::string get_numa_config_as_string() const { return ""; }
    virtual std::string numa_config_information_as_string() const { return ""; }
    virtual std::string thread_allocation_information_as_string() const { return ""; }
    virtual std::string thread_binding_information_as_string() const { return ""; }

#if STOCKFISH
   private:
    const std::string binaryDirectory;
#endif

	// 🌈 エンジンに追加オプションを設定したいときは、この関数をoverrideする。
	//     この関数は、USIEngine::set_engine()のタイミングで呼び出される。
	// 📝 この関数のなかでGetOptions()->add()を用いて、Optionを追加する。
	// 💡 Engine::add_options()で"Threads", "NumaPolicy"などのエンジンオプションを生やしているので
	//     これらが必要なのであれば、add_options()をoverrideしてEngine::add_options()を呼び出すこと。
	virtual void add_options() = 0;

	// 📌 Properties

#if STOCKFISH
	// 📝 やねうら王では、これはEngineに持つ
    OptionsMap options;

	// 📝 やねうら王では、これはEngineに持つ
	ThreadPool threads;

	// 📝 やねうら王では、これはYaneuraOuEngineに持つ
    TranspositionTable                       tt;

	// 📝 やねうら王では、これはEngineに持つ
    LazyNumaReplicated<Eval::NNUE::Networks> networks;

	// 📝 やねうら王では、これはYaneuraOuEngineに持つ
    Search::SearchManager::UpdateContext  updateContext;

	// TODO : あとで
    std::function<void(std::string_view)> onVerifyNetworks;
#else
	// スレッドプール(探索用スレッド)の取得
	virtual ThreadPool& get_threads() = 0;
	virtual const ThreadPool& get_threads() const = 0;
#endif

	// 局面の取得
	// 探索開始局面(root)を格納するPositionクラス
	// "position"コマンドで設定された局面が格納されている。
	virtual Position& get_position() = 0;

	// 📌 USIコマンドのhandler(同名のUSIコマンドが送られてきた時のhandler)

	// "usi"コマンド。`id`(エンジン名)と`author`(エンジン作者)を出力する。
	// 💡 出力のされ方を丸ごとカスタマイズしたければ、これをoverrideする。
    virtual void usi() = 0;

	// "isready"コマンド。時間のかかる初期化処理はここで行うこと。
    /*
		📓 将棋(USIプロトコル)では、"isready"と"usinewgame"との両方が
		    対局ごとに送られてくることが保証されている。
			このうち、"isready"は、"readyok"を返すまでGUIに待ってもらえる。
			そこで、時間がかかる初期化も含め、すべての初期化は"isready"で行い、
			"usinewgame"は単に無視をするのが簡単である。

			一方、チェス(UCIプロトコル)では、"isready"は初回のみしか送られてこず(?)、
			"ucinewgame"は対局ごとに毎回送られてくるので、"isready"は何もせずに
			"ucinewgame"で初期化(search_clear呼び出し)を行っている。

			やねうら王では、"usinewgame"を無視して、search_clear()を実装しない。
	*/
	virtual void isready() = 0;

	// "usinewgame"コマンド。
	// GUIからはこのコマンドが1局の最初に送られてくることは保証されているので、
	// 1局ごとに行いたい探索部の初期化は、ここで行うこと。
	virtual void usinewgame() = 0;

	// "user"コマンド。ユーザー(エンジン実装者)の実験用。
	virtual void user(std::istringstream& is) = 0;

	// 📌 "usi"コマンドに対して表示するengineのprofile

	virtual std::string get_engine_name() const    = 0;
    virtual std::string get_engine_author() const  = 0;
    virtual std::string get_engine_version() const = 0;
    virtual std::string get_eval_name() const      = 0;
};

// エンジンの基底クラス
// 📝 これを派生させて、自作のエンジンを作成する。
//     USIEngine::set_engine()でその派生クラスを渡して使う。
class Engine: public IEngine {
   public:
    Engine();

    virtual std::uint64_t perft(const std::string& fen,
                                Depth              depth /*, bool isChess960 */) override;

    virtual void go(Search::LimitsType& limits) override;
    virtual void stop() override;

    virtual void wait_for_search_finished() override;
    virtual void set_position(const std::string&              sfen,
                              const std::vector<std::string>& moves) override;

    virtual void set_numa_config_from_option(const std::string& o) override;
    virtual void resize_threads() override;
    virtual void set_tt_size(size_t mb) override;
    virtual void set_ponderhit(bool b) override;
    virtual void search_clear() override;

    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&&) override final;
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&&) override final;
    virtual void set_on_iter(std::function<void(const InfoIter&)>&&) override final;
    virtual void
    set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&) override final;
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&&) override;
    virtual void set_on_update_string(std::function<void(std::string_view)>&&) override final;
    virtual std::function<void(std::string_view, std::string_view)>
    get_on_bestmove() override final;

    virtual void verify_networks() const override {}
    virtual void save_network(const std::string& path) override {}

    virtual void              trace_eval() const override;
    virtual Value             evaluate() const override;
    virtual const OptionsMap& get_options() const override;
    virtual OptionsMap&       get_options() override;

	virtual int get_hashfull(int maxAge = 0) const override;

    virtual std::string sfen() const override;
    virtual void        flip() override;
    virtual std::string visualize() const override;
	virtual std::vector<std::pair<size_t, size_t>> get_bound_thread_count_by_numa_node() const override;
    virtual std::string get_numa_config_as_string() const override;
    virtual std::string numa_config_information_as_string() const override;
    virtual std::string thread_allocation_information_as_string() const override;
    virtual std::string thread_binding_information_as_string() const override;

    virtual void              add_options() override;
    virtual ThreadPool&       get_threads() override { return threads; }
    virtual const ThreadPool& get_threads() const override { return threads; }
    virtual Position&         get_position() override { return pos; }

    virtual void usi() override;
    virtual void isready() override;
    virtual void usinewgame() override {};
    virtual void user(std::istringstream& is) override {};

    virtual std::string get_engine_name() const override { return "YaneuraOu"; }
    virtual std::string get_engine_author() const override { return "yaneurao"; }
    virtual std::string get_engine_version() const override { return ENGINE_VERSION; }
    virtual std::string get_eval_name() const override { return EVAL_TYPE_NAME; }

	// 🌈 どのエンジンでも共通で必要なエンジンオプションを生やす。
	// "USI_Ponder", "StochasticPonder", "NumaPolicy","DebugLogFile","DepthLimit", "NodesLimit", "DebugLogFile"
	void add_base_options();

	// 🌈 エンジンオプション"USI_Ponder"の値
    bool usi_ponder        = false;

	// 🌈 エンジンオプション"StochasticPonder"の値
	bool stochastic_ponder = false;

#if STOCKFISH    
   protected:
#endif   
    // 📌 エンジンを実装するために必要な最低限のコンポーネント

    //const std::string binaryDirectory;
    // 📝 やねうら王では、CommandLine::gから取得できるので不要。

    // Numaの管理用(どのNumaを使うかというIDみたいなもの)
    NumaReplicationContext numaContext;

    // 探索開始局面(root)を格納するPositionクラス
    // "position"コマンドで設定された局面が格納されている。
    Position pos;

    // ここまでの局面に対するStateInfoのlist
    StateListPtr states;

    // 思考エンジンオプション
    OptionsMap options;

    // スレッドプール(探索用スレッド)
    ThreadPool threads;

    //TranspositionTable tt;
    // 📝 やねうら王ではEngine基底classはTTを持たない。
    //     (Engineが必ずStockfishのTTを必要とするわけではないので)

    //LazyNumaReplicated<Eval::NNUE::Networks> networks;
    // TODO : あとで検討する

    // 読み筋を出力するためのlistener
    Search::UpdateContext updateContext;

    // TODO : あとで検討する
    std::function<void(std::string_view)> onVerifyNetworks;

    // 📌 エンジンで用いるヘルパー関数

    // 🌈 実行に時間がかかるjobを実行する。
    // job : 実行するjob
    //
    // 📝 実行中にkeep aliveのために定期的に改行を標準出力に出力する。
    //     USIで"isready"に対して時間のかかる処理を実行したい時に用いる。
    void run_heavy_job(std::function<void()> job);

	/*
		📓 dlshogi(ふかうら王)では、

		1. "Position"コマンドで1つ目に送られてきた文字列("startpos" or sfen文字列)
		2. "Position"コマンドで"moves"以降にあった、rootの局面からこの局面に至るまでの手順

		が必要なので、これらを用意する。
	*/

	// "Position"コマンドで1つ目に送られてきた文字列("startpos" or sfen文字列)
	std::string game_root_sfen;

	// "Position"コマンドで"moves"以降にあった、rootの局面からこの局面に至るまでの手順
	std::vector<Move> moves_from_game_root;
};

// IEngine派生classを入れておいて、使うためのwrapper
// 📝 これを用意せずにIEngine*を直接用いてもいいのだが
//    そうすると engine-> のように参照型を使う必要があって、
class EngineWrapper: public IEngine {
   public:
    // 🌈 Engine派生classをセットする。
    void set_engine(IEngine& _engine) { engine = &_engine; }

    // Engineのoverride
    // 📌 すべてset_engine()で渡されたengineに委譲する。

    virtual std::uint64_t perft(const std::string& fen,
                                Depth              depth /*, bool isChess960 */) override {
        return engine->perft(fen, depth);
    }

    virtual void go(Search::LimitsType& limits) override { engine->go(limits); }
    virtual void stop() override { engine->stop(); }

    virtual void wait_for_search_finished() override { engine->wait_for_search_finished(); }
    virtual void set_position(const std::string&              sfen,
                              const std::vector<std::string>& moves) override {
        engine->set_position(sfen, moves);
    }

    virtual void set_numa_config_from_option(const std::string& o) override { engine->set_numa_config_from_option(o); }
    virtual void resize_threads() override { engine->resize_threads(); }
    virtual void set_tt_size(size_t mb) override { engine->set_tt_size(mb); }
    virtual void set_ponderhit(bool b) override { engine->set_ponderhit(b); }
    virtual void search_clear() override { engine->search_clear(); }

    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&& f) override final {
        engine->set_on_update_no_moves(std::move(f));
    }
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&& f) override final {
        engine->set_on_update_full(std::move(f));
    }
    virtual void set_on_iter(std::function<void(const InfoIter&)>&& f) override final {
        engine->set_on_iter(std::move(f));
    }
    virtual void
    set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) override final {
        engine->set_on_bestmove(std::move(f));
    }
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&& f) override {
        engine->set_on_verify_networks(std::move(f));
    }
    virtual void set_on_update_string(std::function<void(std::string_view)>&& f) override final {
        engine->set_on_update_string(std::move(f));
    }
    virtual std::function<void(std::string_view, std::string_view)> get_on_bestmove() override {
        return engine->get_on_bestmove();
    }

    virtual void verify_networks() const override { engine->verify_networks(); }
    virtual void save_network(const std::string& path) override { engine->save_network(path); }

    virtual void trace_eval() const override { engine->trace_eval(); }
    virtual Value evaluate() const override { return engine->evaluate(); }

    virtual const OptionsMap& get_options() const override { return engine->get_options(); }
    virtual OptionsMap&       get_options() override { return engine->get_options(); }

    virtual int get_hashfull(int maxAge = 0) const override { return engine->get_hashfull(maxAge); }

    virtual std::string sfen() const override { return engine->sfen(); }
    virtual void        flip() override { return engine->flip(); }
    virtual std::string visualize() const override { return engine->visualize(); }

    virtual std::vector<std::pair<size_t, size_t>>
    get_bound_thread_count_by_numa_node() const override {
        return engine->get_bound_thread_count_by_numa_node();
    }
    virtual std::string get_numa_config_as_string() const override {
        return engine->get_numa_config_as_string();
    }
    virtual std::string numa_config_information_as_string() const override {
        return engine->numa_config_information_as_string();
    }
    virtual std::string thread_allocation_information_as_string() const override {
        return engine->thread_allocation_information_as_string();
    }
    virtual std::string thread_binding_information_as_string() const override {
        return engine->thread_binding_information_as_string();
    }

    virtual void              add_options() override { return engine->add_options(); }
    virtual ThreadPool&       get_threads() override { return engine->get_threads(); }
    virtual const ThreadPool& get_threads() const override { return engine->get_threads(); }
    virtual Position&         get_position() override { return engine->get_position(); }

    virtual void usi() override { engine->usi(); }
    virtual void isready() override { engine->isready(); }
    virtual void usinewgame() override { engine->usinewgame(); }
    virtual void user(std::istringstream& is) override { engine->user(is); }

    virtual std::string get_engine_name() const override { return engine->get_engine_name(); }
    virtual std::string get_engine_author() const override { return engine->get_engine_author(); }
    virtual std::string get_engine_version() const override { return engine->get_engine_version(); }
    virtual std::string get_eval_name() const override { return engine->get_eval_name(); }

   private:
    IEngine* engine;
};


// 🌈 エンジンを登録するヘルパー
// 📝 static EngineFuncRegister reg_a(engine_main_a, 0); のようにしてengineのentry pointを登録する。
//     USER_ENGINEであるuser-engine.cpp を参考にすること。
//     priorityが一番高いものが実行される。
struct EngineFuncRegister {
	EngineFuncRegister(std::function<void()> f, const std::string& engine_name, int priority);
};

} // namespace YaneuraOu

#endif // #ifndef ENGINE_H_INCLUDED
