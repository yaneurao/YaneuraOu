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

namespace YaneuraOu {

// エンジン本体
class Engine
{
   public:
	// 読み筋
	using InfoShort = Search::InfoShort;
	using InfoFull  = Search::InfoFull;
	using InfoIter  = Search::InfoIteration;

	// TODO : あとで
	Engine(int argc, char** argv) :
		cli(argc, argv) {
	}

	// modifiers

	// 読み筋(InfoShort)のセット
	void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);

	// 読み筋(InfoFull)のセット
	void set_on_update_full(std::function<void(const InfoFull&)>&&);


   private:

	Search::SearchManager::UpdateContext  updateContext;

	CommandLine cli;
};

} // namespace YaneuraOu

#endif // #ifndef ENGINE_H_INCLUDED

