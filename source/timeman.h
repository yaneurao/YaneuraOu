#ifndef TIMEMAN_H_INCLUDED
#define TIMEMAN_H_INCLUDED

#include <cstdint>

#include "misc.h"

namespace YaneuraOu {

	class OptionsMap;
	enum Color : int8_t;

	namespace Search {
		struct LimitsType;
	}

	// The TimeManagement class computes the optimal time to think depending on
	// the maximum available time, the game move number, and other parameters.

	// TimeManagementクラスは、使用可能な最大時間、対局の手数、
	// その他のパラメータに応じて、思考に費やす最適な時間を計算します。

	class TimeManagement {
	public:
		void init(Search::LimitsType& limits,
			Color               us,
			int                 ply,
			const OptionsMap& options,
			double& originalTimeAdjust);

		TimePoint optimum() const;
		TimePoint maximum() const;
		template<typename FUNC>
		TimePoint elapsed(FUNC nodes) const {
			return useNodesTime ? TimePoint(nodes()) : elapsed_time();
		}
		TimePoint elapsed_time() const { return now() - startTime; };

		// 初期化。
		// ※　やねうら王では使わない。
		void clear() {}

		void advance_nodes_time(std::int64_t nodes);

	private:
		TimePoint startTime;
		TimePoint optimumTime;
		TimePoint maximumTime;

		std::int64_t availableNodes = -1;     // When in 'nodes as time' mode
		bool         useNodesTime = false;  // True if we are in 'nodes as time' mode
	};

}  // namespace YaneuraOu

#endif  // #ifndef TIMEMAN_H_INCLUDED
