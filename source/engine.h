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

//#include "nnue/network.h"
#include "numa.h"
#include "position.h"
#include "search.h"
//#include "syzygy/tbprobe.h"  // for Stockfish::Depth
#include "thread.h"
#include "tt.h"
#include "usi_option.h"

//namespace Stockfish {

enum Square : int;

class Engine {
public:
	using InfoShort = Search::InfoShort;
	using InfoFull  = Search::InfoFull;
	using InfoIter  = Search::InfoIteration;

	Engine(std::optional<std::string> path = std::nullopt);

	// Cannot be movable due to components holding backreferences to fields
	// フィールドへの後方参照を保持しているコンポーネントのため、移動可能ではありません

	Engine(const Engine&) = delete;
	Engine(Engine&&) = delete;
	Engine& operator=(const Engine&) = delete;
	Engine& operator=(Engine&&) = delete;

	~Engine() { wait_for_search_finished(); }

	std::uint64_t perft(const std::string& fen, Depth depth, bool isChess960);

	// non blocking call to start searching
	// 非ブロッキング呼び出しで探索を開始します

	void go(Search::LimitsType&);

	// non blocking call to stop searching
	// 非ブロッキング呼び出しで探索を停止します

	void stop();

	// blocking call to wait for search to finish
	// 探索の完了を待つためのブロッキング呼び出し

	void wait_for_search_finished();

	// set a new position, moves are in UCI format
	// 新しい局面を設定し、指し手はUCI形式です
	// ⇨ 将棋ではUCIではなくUSI。

	void set_position(const std::string& fen, const std::vector<std::string>& moves);

	// modifiers
	// 修飾子

	void set_numa_config_from_option(const std::string& o);
	void resize_threads();
	void set_tt_size(size_t mb);
	void set_ponderhit(bool);
	void search_clear();

	void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);
	void set_on_update_full(std::function<void(const InfoFull&)>&&);
	void set_on_iter(std::function<void(const InfoIter&)>&&);
	void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&);
	void set_on_verify_networks(std::function<void(std::string_view)>&&);

	// network related
	// ネットワーク関連

	void verify_networks() const;
	void load_networks();
	void load_big_network(const std::string& file);
	void load_small_network(const std::string& file);
	void save_network(const std::pair<std::optional<std::string>, std::string> files[2]);

	// utility functions
	// ユーティリティ関数

	void trace_eval() const;

	const OptionsMap& get_options() const;
	OptionsMap& get_options();

	int get_hashfull(int maxAge = 0) const;

	std::string                            fen() const;
	void                                   flip();
	std::string                            visualize() const;
	std::vector<std::pair<size_t, size_t>> get_bound_thread_count_by_numa_node() const;
	std::string                            get_numa_config_as_string() const;
	std::string                            numa_config_information_as_string() const;
	std::string                            thread_allocation_information_as_string() const;
	std::string                            thread_binding_information_as_string() const;

private:
	const std::string binaryDirectory;

	NumaReplicationContext numaContext;

	Position     pos;
	StateListPtr states;
	Square       capSq;

	OptionsMap                               options;
	ThreadPool                               threads;
	TranspositionTable                       tt;
	//LazyNumaReplicated<Eval::NNUE::Networks> networks;

	//Search::SearchManager::UpdateContext  updateContext;
	std::function<void(std::string_view)> onVerifyNetworks;
};

//}  // namespace Stockfish


#endif  // #ifndef ENGINE_H_INCLUDED
