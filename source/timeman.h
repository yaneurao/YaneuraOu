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

	// 起動時に呼び出す。
	// このclassが使用するengine optionを追加する。
	void add_options(OptionsMap& options);

	// 今回の思考時間を決定する。
	// optimum(),maximum(),minimum()に反映する。
    void init(Search::LimitsType& limits, Color us, int ply, const OptionsMap& options
                /* , double& originalTimeAdjust */
                // 💡 やねうら王では使わないことにする。
    );

    TimePoint optimum() const;
    TimePoint maximum() const;
    TimePoint minimum() const;  // 📌 やねうら王独自追加。

	// "go"からの経過時間を返す。
    template<typename FUNC>
    TimePoint elapsed(FUNC nodes) const {
#if 0
		return useNodesTime ? TimePoint(nodes()) : elapsed_time();
#endif
        // 📝 やねうら王ではNodesTimeを使わないのでelapsed()はそのままelapsed_Time()に委譲しておく。
        return elapsed_time();
    }

    // startTimeからの経過時間。
    // 💡 startTimeは、init()した時にLimitsType::startTimeがコピーされる。そこからの経過時間。
    TimePoint elapsed_time() const { return now() - startTime; };

    // 初期化。
    // ※　やねうら王では使わない。
    void clear() {}

    //void advance_nodes_time(std::int64_t nodes);
    // 💡 やねうら王ではNodesTimeを使わない。

    // 📌 以下、やねうら王独自追加。

    // 探索終了の時間(startTime + search_end >= now()になったら停止)
    TimePoint search_end;

	// 秒単位で切り上げる。ただし、NetworkDelayの値などを考慮する。
	TimePoint round_up(TimePoint t);

private:
    TimePoint startTime;    // 💡 探索開始時刻。LimitsType startTimeの値。
    TimePoint minimumTime;  // 📌 やねうら王独自追加。
    TimePoint optimumTime;
    TimePoint maximumTime;

    //std::int64_t availableNodes = -1;     // When in 'nodes as time' mode
    //bool         useNodesTime   = false;  // True if we are in 'nodes as time' mode
    // 💡 やねうら王ではNodesTimeを使わない。

    // 📌 以下、やねうら王独自追加。

    // init()の内部実装。
    void init_(Search::LimitsType& limits, Color us, int ply, const OptionsMap& options);

	// optionsのそれぞれの値
	TimePoint minimum_thinking_time;
    TimePoint network_delay;
    TimePoint remain_time;

    // 前回のinit()の値
    Search::LimitsType* lastcall_Limits;
    Color               lastcall_Us;
    int                 lastcall_Ply;
    OptionsMap*         lastcall_Opt;
};

}  // namespace YaneuraOu

#endif  // #ifndef TIMEMAN_H_INCLUDED
