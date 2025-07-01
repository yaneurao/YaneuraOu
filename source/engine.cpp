#include "engine.h"

namespace YaneuraOu {

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
	updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
	updateContext.onUpdateFull = std::move(f);
}


} // namespace YaneuraOu
