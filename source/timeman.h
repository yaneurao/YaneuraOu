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
	// ⚠ optionsに"USI_Ponder"と"Stochastic_Ponder"オプションが含まれていること。
    void init(const Search::LimitsType& limits, Color us, int ply, const OptionsMap& options
#if STOCKFISH
			  , double& originalTimeAdjust
    // 💡 やねうら王では使わないことにする。
#else
			  , int max_moves_to_draw
#endif
    );

    TimePoint optimum() const;
    TimePoint maximum() const;
    TimePoint minimum() const;  // 📌 やねうら王独自追加。

    // "go"からの経過時間を返す。
    template<typename FUNC>
    TimePoint elapsed(FUNC nodes) const {
#if STOCKFISH
        return useNodesTime ? TimePoint(nodes()) : elapsed_time();
#else
        // 📝 やねうら王ではNodesTimeを使わないのでelapsed()はそのままelapsed_Time()に委譲しておく。
        return elapsed_time();
#endif
    }

    // startTimeからの経過時間。
    // 💡 startTimeは、init()した時にLimitsType::startTimeがコピーされる。そこからの経過時間。
    TimePoint elapsed_time() const { return now() - startTime; };


	// 初期化。
    // 🌈　やねうら王では使わない。Stockfishとの互換性のために用意。
    void clear() {}

#if STOCKFISH

    void advance_nodes_time(std::int64_t nodes);
    // 💡 やねうら王ではNodesTimeを使わない。

#else

    // 📌 以下、やねうら王独自追加。

    // 探索終了の時刻
    /*
	  📓 startTimeからの経過時間がこの値以上になれば探索の終了時間。
		  0ならば終了時刻は未確定。init()で初期化される。

		  つまり、startTime + search_end <= now()になったら停止。
	      この条件式の左辺のstartTimeを右辺に移項させると、

		    search_end <= now() - startTime = elpased()

		  であり、つまり search_end <= elapsed()がその条件。

		  また、search_endはinit()のなかで 0 に初期化されているので、
		  search_endが非0であれば、終了が確定していてその時刻が設定されているという意味になる。

		  よって、
			  search_end && search_end <= elapsed()
		  という条件式が必要で、これはMainManager::check_time()に書いてある。
	*/
    TimePoint search_end;

    // 秒単位で切り上げる。ただし、NetworkDelayの値などを考慮する。
    TimePoint round_up(TimePoint t);

    // 探索を終了させることが確定しているが、秒単位で切り上げて、search_endにそれを設定したい時に呼び出す。
	// 💡 引数のeはelapsedTime。これはなくてもelapsedTime()を呼び出せばいいのだが、
	//     呼び出し元ですでに持っているので、二度elpasedTime()を呼び出すのは嫌だから引数で渡している。
    void set_search_end(TimePoint e);

    // ponderhitTimeからの経過時間。
    // 💡 ponderhitTimeは、init()した時にLimitsType::startTimeがコピーされる。
	//     "ponderhit"したときには、MainManager::set_ponderhit()でその時刻がコピーされる。
	//     よって、"go ponder"での思考でなくともこの変数は使える。
    TimePoint elapsed_time_from_ponderhit() const { return now() - ponderhitTime; };

#endif


#if STOCKFISH
   private:
    TimePoint startTime;      // 💡 探索開始時刻。LimitsType startTimeの値。
#else
    bool      isFinalPush;    // 🌈 秒読みに突入しているので持ち時間を使い切るべきであるフラグ。
							  //     つまりはponderhitTimeから数えてminimumTime分は使って欲しい。
    TimePoint startTime;      // 💡 探索開始時刻。LimitsType startTimeの値。
    TimePoint ponderhitTime;  // 🌈 "ponderhit"した時刻。startTimeからの経過時間ではなく、Timer::now()の生の値。
                              //     "ponderhit"するまではstartTimeと同じ値。
   private:
    TimePoint minimumTime;  // 🌈 やねうら王独自追加。
#endif

    TimePoint optimumTime;
    TimePoint maximumTime;

#if STOCKFISH

	std::int64_t availableNodes = -1;     // When in 'nodes as time' mode
    bool         useNodesTime   = false;  // True if we are in 'nodes as time' mode
    // 💡 やねうら王ではNodesTimeを使わない。

#else
    // 📌 以下、やねうら王独自追加。

    // init()の内部実装。
    void init_(const Search::LimitsType& limits,
               Color                     us,
               int                       ply,
               const OptionsMap&         options,
               int                       max_moves_to_draw);

    // optionsのそれぞれの値
    TimePoint minimum_thinking_time;
    TimePoint network_delay;
    TimePoint remain_time;

	// TODO : あとで
#if 0
    // 前回のinit()の値。
	// このあとround_up()で用いるので保存しておく。
    Search::LimitsType* lastcall_Limits;
    Color               lastcall_Us;
    int                 lastcall_Ply;
    OptionsMap*         lastcall_Opt;
#endif

#endif
};

}  // namespace YaneuraOu

#endif  // #ifndef TIMEMAN_H_INCLUDED
