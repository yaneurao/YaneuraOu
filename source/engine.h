// TODO : このファイル、作業中

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
#include "usi_option.h"

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

	// エンジンに追加オプションを設定したいときは、この関数をoverrideする。
	// この関数は、Engineのコンストラクタから呼び出される。
	// 📝 GetOptions()->add()を用いて、Optionを追加する。
	// 💡 Engine::add_options()で"Threads", "NumaPolicy"などのエンジンオプションを生やしているので
	//     これらが必要なのであれば、add_options()をoverrideしてEngine::add_options()を呼び出すこと。
	virtual void add_options() = 0;

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

	// 📌 読み筋の出力など、event handlerのsetter

    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&&) {}
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&&) {}
    virtual void set_on_iter(std::function<void(const InfoIter&)>&&) {}
    virtual void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&) {}
    virtual void set_on_update_string(std::function<void(std::string_view)>&&) {}
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&&) {}

	// blocking call to wait for search to finish
	// 探索が完了するのを待機する。(完了したらリターンする)
	// 📝 ThreadPoolのmain_threadの完了を待機している。
	virtual void wait_for_search_finished() = 0;


	// 評価関数部の初期化が行えているかのチェックを行う。
	virtual void verify_networks() = 0;

	// 評価関数パラメーターを保存する。
	// "export_net"コマンドに対して呼び出される。
	virtual void save_network(const std::string& path) = 0;

	// 📌 Properties

	// スレッドプール(探索用スレッド)の取得
	virtual ThreadPool& get_threads() = 0;
	virtual const ThreadPool& get_threads() const = 0;

	// 局面の取得
	// 探索開始局面(root)を格納するPositionクラス
	// "position"コマンドで設定された局面が格納されている。
	virtual Position& get_position() = 0;

	// OptionsMapを取得
	virtual OptionsMap& get_options() = 0;

	// (探索中の)現在の局面のsfen文字列を返す。
	virtual std::string sfen() const = 0;

	// 局面を視覚化した文字列を取得する。(デバッグ用)
	virtual std::string visualize() const = 0;

	// 📌 USIコマンドのhandler(同名のUSIコマンドが送られてきた時のhandler)

	// "usi"コマンド。`id`(エンジン名)と`author`(エンジン作者)を出力する。
	// 💡 出力のされ方を丸ごとカスタマイズしたければ、これをoverrideする。
    virtual void usi() = 0;

	// "isready"コマンド。時間のかかる初期化処理はここで行うこと。
	virtual void isready() = 0;

	// "usinewgame"コマンド。
	// GUIからはこのコマンドが1局の最初に送られてくることは保証されているので、
	// 1局ごとに行いたい探索部の初期化は、ここで行うこと。
	virtual void usinewgame() = 0;

	// "position"コマンドの下請け。
	// sfen文字列 + movesのあとに書かれていた(USIの)指し手文字列から、現在の局面を設定する。
	virtual void set_position(const std::string& sfen, const std::vector<std::string>& moves) = 0;

	// "go"コマンド。ThreadPoolのmain threadに対して探索を開始(start_searching)する。
	// non blocking call to start searching
	// 探索を開始する。(non blocking呼び出し)
	virtual void go(Search::LimitsType& limits) = 0;

	// "stop"コマンド。ThreadPoolに対してstop信号を送信する。
	// non blocking call to stop searching
	// 探索を停止させる。(non blocking呼び出し)
	virtual void stop() = 0;

	// "perft"コマンド。perftとは、performance testの略。
	virtual std::uint64_t perft(const std::string& fen, Depth depth /*, bool isChess960 */) = 0;

	// "trace"コマンド。現在の局面に対して評価関数を呼び出して結果を出力する。
	virtual void trace_eval() const = 0;

	// "user"コマンド。ユーザー(エンジン実装者)の実験用。
	virtual void user(std::istringstream& is) = 0;

	// "usi"コマンドに対して表示するengineのprofile

	virtual std::string get_engine_name() const    = 0;
    virtual std::string get_engine_author() const  = 0;
    virtual std::string get_engine_version() const = 0;
    virtual std::string get_eval_name() const      = 0;

	// 💡 interfaceなので仮想デストラクタが必要
	virtual ~IEngine() {}
};

// エンジンの基底クラス
// 📝 これを派生させて、自作のエンジンを作成する。
//     USIEngine::set_engine()でその派生クラスを渡して使う。
class Engine : public IEngine
{
public:
	Engine();

	virtual void add_options() override;
	virtual void resize_threads() override;
	virtual void set_tt_size(size_t mb) override {}
    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&&) override final;
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&&) override final;
    virtual void set_on_iter(std::function<void(const InfoIter&)>&&) override final;
    virtual void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&) override final;
    virtual void set_on_update_string(std::function<void(std::string_view)>&&) override final;
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&&) override;
    virtual void wait_for_search_finished() override;
	virtual void verify_networks() override {}
	virtual void save_network(const std::string& path) override {}
	virtual ThreadPool& get_threads() override { return threads; }
	virtual const ThreadPool& get_threads() const override { return threads; }
	virtual Position& get_position() override { return pos; }
	virtual OptionsMap& get_options() override { return options; }
	virtual std::string sfen() const override { return pos.sfen(); }
	virtual std::string visualize() const override;
    virtual void usi() override;
    virtual void isready() override;
	virtual void usinewgame() override {};
	virtual void set_position(const std::string& sfen, const std::vector<std::string>& moves) override;
	virtual void go(Search::LimitsType& limits) override;
	virtual void stop() override;
	virtual std::uint64_t perft(const std::string& fen, Depth depth /*, bool isChess960 */) override;
	virtual void trace_eval() const override {}
	virtual void user(std::istringstream& is) override {};
    virtual std::string get_engine_name() const override { return "YaneuraOu"; }
    virtual std::string get_engine_author() const override { return "yaneurao"; }
    virtual std::string get_engine_version() const override { return ENGINE_VERSION; }
    virtual std::string get_eval_name() const override { return EVAL_TYPE_NAME; }

protected:

	// 📌 エンジンを実装するために必要な最低限のコンポーネント

    //const std::string binaryDirectory;
	// 📝 やねうら王では、CommandLine::gから取得できるので不要。

	// Numaの管理用(どのNumaを使うかというIDみたいなもの)
	NumaReplicationContext numaContext;

	// 探索開始局面(root)を格納するPositionクラス
	// "position"コマンドで設定された局面が格納されている。
	Position           pos;

	// ここまでの局面に対するStateInfoのlist
	StateListPtr       states;

	// 思考エンジンオプション
	OptionsMap         options;

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

	// NumaConfig(numaContextのこと)を Options["NumaPolicy"]の値 から設定する。
	void set_numa_config_from_option(const std::string& o);

};

// IEngine派生classを入れておいて、使うためのwrapper
// 📝 これを用意せずにIEngine*を直接用いてもいいのだが
//    そうすると engine-> のように参照型を使う必要があって、
class EngineWrapper : public IEngine
{
public:
	// Engine派生classをセットする。
	void set_engine(IEngine& _engine) { engine = &_engine; }

	// Engineのoverride
	// 📌 すべてset_engine()で渡されたengineに委譲する。

	virtual void add_options() override { engine->add_options(); }
	virtual void resize_threads() override { engine->resize_threads(); }
    virtual void set_tt_size(size_t mb) override { engine->set_tt_size(mb); }
    virtual void set_on_update_no_moves(std::function<void(const InfoShort&)>&& f) override final { engine->set_on_update_no_moves(std::move(f));}
    virtual void set_on_update_full(std::function<void(const InfoFull&)>&& f) override final { engine->set_on_update_full(std::move(f)); }
    virtual void set_on_iter(std::function<void(const InfoIter&)>&& f) override final { engine->set_on_iter(std::move(f));}
    virtual void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) override final { engine->set_on_bestmove(std::move(f)); }
    virtual void set_on_update_string(std::function<void(std::string_view)>&& f) override final { engine->set_on_update_string(std::move(f)); }
    virtual void set_on_verify_networks(std::function<void(std::string_view)>&& f) override { engine->set_on_verify_networks(std::move(f)); }
    virtual void wait_for_search_finished() override { engine->wait_for_search_finished(); }
	virtual void verify_networks() override { engine->verify_networks(); }
	virtual void save_network(const std::string& path) override { engine->save_network(path); }
	virtual ThreadPool& get_threads() override { return engine->get_threads(); }
	virtual const ThreadPool& get_threads() const override { return engine->get_threads(); }
	virtual Position& get_position() override { return engine->get_position(); }
	virtual OptionsMap& get_options() override { return engine->get_options(); }
	virtual std::string sfen() const override { return engine->sfen(); }
	virtual std::string visualize() const override { return engine->visualize(); }
    virtual void usi() override { engine->usi(); }
	virtual void isready() override { engine->isready(); }
	virtual void usinewgame() override { engine->usinewgame(); }
	virtual void set_position(const std::string& sfen, const std::vector<std::string>& moves) override { engine->set_position(sfen, moves); }
	virtual void go(Search::LimitsType& limits) override { engine->go(limits); }
	virtual void stop() override { engine->stop(); }
	virtual std::uint64_t perft(const std::string& fen, Depth depth /*, bool isChess960 */) override { return engine->perft(fen, depth); }
	virtual void trace_eval() const override { engine->trace_eval(); }
	virtual void user(std::istringstream& is) override { engine->user(is); }
    virtual std::string get_engine_name() const override { return engine->get_engine_name(); }
    virtual std::string get_engine_author() const override { return engine->get_engine_author(); }
	virtual std::string get_engine_version() const override { return engine->get_engine_version(); }
    virtual std::string get_eval_name() const override { return engine->get_eval_name(); }

private:
	IEngine* engine;
};

#if 0
// やねうら王の通常探索部
// 📌 これがStockfishのEngine classに相当する。
//     エンジン共通で必要なものは、IEngine/Engine(これが、それぞれエンジンのinterfaceとエンジン基底class)に移動させた。
class YaneuraOuEngine : public Engine
{
   public:
	// 読み筋
	using InfoShort = Search::InfoShort;
	using InfoFull  = Search::InfoFull;
	using InfoIter  = Search::InfoIteration;

	// pathとして起動path(main関数で渡されたargv[0])を渡す。
	// ⇨  やねうら王ではこれをやめることにした。
	//    CommandLine::gから取得する。
	YaneuraOuEngine(/* std::optional<std::string> path = std::nullopt*/ );


	// blocking call to wait for search to finish
	// 探索が完了のを待機する。(blocking呼び出し)
	void wait_for_search_finished();

	// modifiers

	// NumaConfigをエンジンオプションの"NumaPolicy"から設定する。
	void set_numa_config_from_option(const std::string& o);

	void resize_threads();
	void set_tt_size(size_t mb);
	void set_ponderhit(bool);

	// "usinewgame"に対してWorkerを初期化する。
	void search_clear();

	// 読み筋(InfoShort)のセット
	void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);

	// 読み筋(InfoFull)のセット
	void set_on_update_full(std::function<void(const InfoFull&)>&&);

	void set_on_iter(std::function<void(const InfoIter&)>&&);
	void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&);

	// verify_network()を呼び出した時に、NN::network.verify()からcallbackされるfunctionを設定する。
	void set_on_verify_networks(std::function<void(std::string_view)>&&);

	// network related
	// 評価関数関連

	// 評価関数が読み込まれているかを確認する。読み込まれていなければload_networks()で読み込む。
	void verify_networks() const;

	// 評価関数の読み込み
	void load_networks();

	// Stockfishは大きいnetworkと小さいnetworkとがある。それぞれの読み込み。
	//void load_big_network(const std::string& file);
	//void load_small_network(const std::string& file);

	// 評価関数をファイルに保存する。
	//void save_network(const std::pair<std::optional<std::string>, std::string> files[2]);
	void save_network(const std::string& file);

	// utility functions

	// 現在の局面の評価値を出力する(デバッグ用)
	void trace_eval() const;

	// エンジンオプション
	const OptionsMap& get_options() const;
	OptionsMap&       get_options();

	int get_hashfull(int maxAge = 0) const;

	// 現在の局面のsfen形式の表現を取得する。
	std::string                            sfen() const;

	// 盤面を180°回転させる。
	void                                   flip();

	// 局面を視覚化した文字列を取得する。(デバッグ用)
	std::string                            visualize() const;

	std::vector<std::pair<size_t, size_t>> get_bound_thread_count_by_numa_node() const;
	std::string                            get_numa_config_as_string() const;
	std::string                            numa_config_information_as_string() const;
	std::string                            thread_allocation_information_as_string() const;
	std::string                            thread_binding_information_as_string() const;

	// 📌 やねうら王独自 📌

	// "isready"のタイミングのcallback。時間のかかる初期化処理はここで行うこと。
	void isready();

	// 自作のエンジンに追加のエンジンオプションを用意したいときは、この関数のなかで定義する。
	// Engineのコンストラクタからコールバックされる。
	void extra_option();

#if defined(USER_ENGINE)
	void user_cmd(std::istringstream& is);
#endif

	// スレッドプールの取得
	ThreadPool* getThreads() { return &threads; }

	// 局面の取得
	Position* getPosition() { return &pos; }


   private:

	// このEngineで保有している置換表
	TranspositionTable                       tt;

	// 評価関数本体
	LazyNumaReplicated<Eval::Evaluator>      networks;

	std::shared_ptr<Book::BookMoveSelector>  book;

	Search::SearchManager::UpdateContext  updateContext;

	// verify_network()を呼び出した時に、NN::network.verify()からcallbackされる。
	std::function<void(std::string_view)> onVerifyNetworks;

};
#endif

// エンジンを登録するヘルパー
// 📝 static EngineFuncRegister reg_a(engine_main_a, 0); のようにしてengineのentry pointを登録する。
//     USER_ENGINEであるuser-engine.cpp を参考にすること。
//     priorityが一番高いものが実行される。
struct EngineFuncRegister {
	EngineFuncRegister(std::function<void()> f, const std::string& engine_name, int priority);
};

} // namespace YaneuraOu

#endif // #ifndef ENGINE_H_INCLUDED
