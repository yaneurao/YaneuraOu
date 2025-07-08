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

// 前方宣言
namespace Book { struct BookMoveSelector; }

// エンジン本体
class Engine
{
   public:
	// 読み筋
	using InfoShort = Search::InfoShort;
	using InfoFull  = Search::InfoFull;
	using InfoIter  = Search::InfoIteration;

	// pathとして起動path(main関数で渡されたargv[0])を渡す。
	Engine(std::optional<std::string> path = std::nullopt);

	// performance test ("perft"コマンドの処理 )
	std::uint64_t perft(const std::string& fen, Depth depth /*, bool isChess960 */ );

	// non blocking call to start searching
	// 探索を開始する。(non blocking呼び出し)

	void go(Search::LimitsType&);

	// non blocking call to stop searching
	// 探索を停止させる。(non blocking呼び出し)

	void stop();

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

	// Numaの管理用(どのNumaを使うかというIDみたいなもの)
	NumaReplicationContext numaContext;

	// 探索開始局面(root)を格納するPositionクラス
	// "position"コマンドで設定された局面が格納されている。
	Position     pos;

	// ここまでの局面に対するStateInfoのlist
	StateListPtr states;

	// 思考エンジン設定
	OptionsMap                               options;

	// スレッドプール(探索用スレッド)
	ThreadPool                               threads;

	// このEngineで保有している置換表
	TranspositionTable                       tt;

	// 評価関数本体
	LazyNumaReplicated<Eval::Evaluator>      networks;

	std::shared_ptr<Book::BookMoveSelector>  book;

	Search::SearchManager::UpdateContext  updateContext;

	// verify_network()を呼び出した時に、NN::network.verify()からcallbackされる。
	std::function<void(std::string_view)> onVerifyNetworks;

};

} // namespace YaneuraOu

#endif // #ifndef ENGINE_H_INCLUDED

