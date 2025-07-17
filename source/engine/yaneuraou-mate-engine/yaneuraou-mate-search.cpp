#include "../../config.h"

#if defined(YANEURAOU_MATE_ENGINE)

#include <sstream>

#include "../../usi.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../mate/mate.h"

using namespace std;
namespace YaneuraOu {

using namespace Mate::Dfpn;

namespace {
// Solver本体
MateDfpnSolver solver(DfpnSolverType::None);

// Solverの種類(エンジンオプションに使う文字列)
vector<string> solver_list = {"32bitNodeSolver", "64bitNodeSolver"};
}

namespace Search {

class YaneuraOuMateWorker: public Worker {
   public:
    YaneuraOuMateWorker(OptionsMap&               options,
                        ThreadPool&               threads,
                        size_t                    threadIdx,
                        NumaReplicatedAccessToken numaAccessToken) :
        // 基底classのconstructorの呼び出し
        Worker(options, threads, threadIdx, numaAccessToken) {}

    // このworker(探索用の1つのスレッド)の初期化
    // 📝 これは、"usinewgame"のタイミングで、すべての探索スレッド(エンジンオプションの"Threads"で決まる)に対して呼び出される。
    virtual void clear() override {}
	
    // Workerによる探索の開始
    // 📝　メインスレッドに対して呼び出される。
    //     そのあと非メインスレッドに対してstart_searching()を呼び出すのは、threads.start_searching()を呼び出すと良い。
    virtual void start_searching() override {
        // 思考エンジンからの返し値
        // 詰将棋ルーチンからMove::resign()が返ってくることはないので、この値が変化していたら返し値があったことを意味する。
        atomic<Move> move = Move::resign();

        // 探索ノード数制限
        u64 nodes_limit = options["NodesLimit"];

		// 探索深さ制限
        int depth_limit = int(options["DepthLimit"]);
        if (depth_limit == 0)
			// 探索深さの制限なし。
            solver.set_max_game_ply(0);
        else
			// 探索深さは現在のgame_ply + DepthLimit - 1の値
			solver.set_max_game_ply(rootPos.game_ply() + depth_limit - 1);

        // 詰将棋の探索用スレッド
        auto thread = std::thread([&]() { move = solver.mate_dfpn(rootPos, nodes_limit); });

        ElapsedTimer time;
        time.reset();                                    // 探索開始からの経過時間を記録しておく。
        TimePoint lastPvOutput = 0;                      // 前回のPV出力時刻
        TimePoint pvInterval   = options["PvInterval"];  // PV出力間隔

        // 読み筋の出力するヘルパ
        auto print_pv = [&]() {
            auto elapsed        = time.elapsed();
            u64  nodes_searched = solver.get_nodes_searched();

            // nps算出
            u64 nps = nodes_searched * 1000 / elapsed;

            sync_cout << "info time " << elapsed << " nodes " << nodes_searched << " nps " << nps
                      << " hashfull " << solver.hashfull() << " pv"
                      << USIEngine::move(solver.get_current_pv()) << sync_endl;
        };


        // 時間切れ判定するヘルパ
        // 将棋ではgo mateのあとmateに使う秒数が入ってきている。
        auto time_up = [&]() { return limits.mate && time.elapsed() >= limits.mate; };

        // 探索の終了を待つ
        while (!threads.stop && !time_up() && move.load() == Move::resign())
        {
            Tools::sleep(100);

            auto elapsed = time.elapsed();
            if (pvInterval && elapsed > lastPvOutput + pvInterval)
            {
                print_pv();
                lastPvOutput = time.elapsed();
            }
        }

        thread.join();

        // 最後に必ず1回PVを出力する。
        print_pv();

        if (time_up())
        {
            sync_cout << "checkmate timeout" << sync_endl;
        }
        else if (move.load() == Move::none())
        {
            if (solver.is_out_of_memory())
                sync_cout << "info string Out Of Memory." << sync_endl;
            else if (solver.get_nodes_searched() >= nodes_limit)
                sync_cout << "info string Exceeded NodesLimit." << sync_endl;

            sync_cout << "checkmate none" << sync_endl;  // 不明
        }
        else if (move.load() == Move::null())
        {
            // 不詰が証明された
            sync_cout << "checkmate nomate" << sync_endl;
        }
        else
        {
            auto pv = solver.get_pv();
            sync_cout << "checkmate" << USIEngine::move(pv) << sync_endl;
        }
    }
};

class YaneuraOuMateEngine: public Engine {
    // エンジン名
    virtual std::string get_engine_name() const override { return "YaneuraOuMateEngine"; }

    // "isready"のタイミングのcallback。時間のかかる初期化処理はここで行う。
    virtual void isready() override {
        // Sovler種別
        auto solver_type = (string) options["SolverType"];
        if (solver_type == solver_list[0])
            solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node32bit);
        else if (solver_type == solver_list[1])
            solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::Node64bit);
        else
            solver.ChangeSolverType(Mate::Dfpn::DfpnSolverType::None);

        u64 mem = options["USI_Hash"];
        sync_cout << "info string DfPn memory allocation , USI_Hash = " << mem << " [MB]"
                  << sync_endl;
        solver.alloc(mem);

        // Engine classのisready()でスレッド数の反映処理などがあるので、そちらに委譲してやる。
        Engine::isready();
    }

    // エンジンに追加オプションを設定したいときは、この関数を定義する。
    virtual void add_options() override {
        // 基底classのadd_options()を呼び出して"Threads", "NumaPolicy"など基本的なオプションを生やす。
        Engine::add_options();

        // 置換表のサイズ。[MB]で指定。
        options.add(  //
          "USI_Hash", Option(1024, 1, MaxHashMB, [this](const Option& o) {
              // set_tt_size();
              // ⇨  どうせisready()で確保するので、handlerを呼び出して反映させる必要はない。
              return std::nullopt;
          }));

        //  PVの出力の抑制のために前回出力時間からの間隔を指定できる。
        //  0なら出力なし。
        options.add("PvInterval", Option(1000, 0, 100000));

        options.add("SolverType", Option(solver_list, solver_list[0]));

		// 詰みの手数制限。0を指定すると手数制限なし。
		// 💡 UCIだと"go mate [手数]"だが、USIでは"go mate [秒数]"なので
		//    手数は別途オプションを設定しないといけない。
        //options.add("DepthLimit", Option(0, 0, 100000));
		// 📝 Engine classですでに追加されている。

        // 探索ノード制限。0なら無制限。
        //options.add("NodesLimit", Option(0, 0, INT64_MAX));
		// 📝 Engine.add_options()ですでに追加されている。
    }

    // USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使う。
    virtual void user(std::istringstream& is) override {
        sync_cout << "UserEngine::user_cmd" << sync_endl;
    }

    // スレッド数を反映させる関数
    virtual void resize_threads() override {
        // 💡 Engine::resize_threads()を参考に書くと良いでしょう。

        // 📌 探索の終了を待つ
        threads.wait_for_search_finished();

        // 📌 スレッド数のリサイズ

        // 💡　難しいことは考えずにコピペして使ってください。"Search::UserWorker"と書いてあるところに、
        //      あなたの作成したWorker派生classの名前を書きます。
        auto worker_factory = [&](size_t threadIdx, NumaReplicatedAccessToken numaAccessToken) {
            return std::make_unique<Search::YaneuraOuMateWorker>(options, threads, threadIdx,
                                                                 numaAccessToken);
        };
        threads.set(numaContext.get_numa_config(), options, options["Threads"], worker_factory);

        // 📌 NUMAの設定

        // スレッドの用いる評価関数パラメーターが正しいNUMAに属するようにする
        threads.ensure_network_replicated();
    }
};
} // namespace Search

namespace Eval {
/*
	📓 StateInfo classの旧メンバーを用いているので、
        USE_CLASSIC_EVALをdefineせざるを得ない。

		これをdefineすると、初期化のためのEval::add_options()を
		用意しなければならない。

		しかしYaneuraOuWorker::add_options()でエンジンオプションの
		追加をしているので、空のEval::add_options()を用意しておく。
*/
void add_options(OptionsMap& options, ThreadPool& threads) {}

} // namespace Eval
} // namespace YaneuraOu

using namespace YaneuraOu;

namespace {

	// 自作のエンジンのentry point
	void engine_main()
	{
		// ここで作ったエンジン
		Search::YaneuraOuMateEngine engine;

		// USIコマンドの応答部
		USIEngine usi;
		usi.set_engine(engine); // エンジン実装を差し替える。

		// USIコマンドの応答のためのループ
		usi.loop();
	}

	// このentry pointを登録しておく。
	static EngineFuncRegister r(engine_main, "YaneuraOuMateEngine", 0);

} // namespace

#endif // YANEURAOU_MATE_ENGINE
