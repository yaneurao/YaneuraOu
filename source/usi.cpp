#include <sstream>
#include <queue>

#include "shogi.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "misc.h"

using namespace std;

// ユーザーの実験用に開放している関数。
// USI拡張コマンドで"user"と入力するとこの関数が呼び出される。
// "user"コマンドの後続に指定されている文字列はisのほうに渡される。
extern void user_test(Position& pos, std::istringstream& is);

// USI拡張コマンドの"test"コマンドなど。
// サンプル用のコードを含めてtest.cppのほうに色々書いてあるのでそれを呼び出すために使う。
#ifdef ENABLE_TEST_CMD
extern void test_cmd(Position& pos, istringstream& is);
extern void perft(Position& pos, istringstream& is);
extern void generate_moves_cmd(Position& pos);
#ifdef MATE_ENGINE
extern void test_mate_engine_cmd(Position& pos, istringstream& is);
#endif
#endif

// "bench"コマンドは、"test"コマンド群とは別。常に呼び出せるようにしてある。
extern void bench_cmd(Position& pos, istringstream& is);

namespace
{
	// 評価関数を読み込んだかのフラグ。これはevaldirの変更にともなってfalseにする。
	bool load_eval_finished = false;
}

// 定跡を作るコマンド
#ifdef ENABLE_MAKEBOOK_CMD
namespace Book { extern void makebook_cmd(Position& pos, istringstream& is); }
#endif

// 協力詰めsolverモード
#ifdef    COOPERATIVE_MATE_SOLVER
#include "cooperate_mate/cooperative_mate_solver.h"
#endif

// 棋譜を自動生成するコマンド
#if defined (EVAL_LEARN)
namespace Learner
{
  // 教師局面の自動生成
  void gen_sfen(Position& pos, istringstream& is);

  // 生成した棋譜からの学習
  void learn(Position& pos, istringstream& is);

#if defined(USE_GENSFEN2018)
  // 開発中の教師局面の自動生成コマンド
  void gen_sfen2018(Position& pos, istringstream& is);
#endif

  // 読み筋と評価値のペア。Learner::search(),Learner::qsearch()が返す。
  typedef std::pair<Value, std::vector<Move> > ValueAndPV;

  ValueAndPV qsearch(Position& pos);
  ValueAndPV search(Position& pos, int depth_, size_t multiPV /* = 1*/);

}
#endif

// "gameover"コマンドに対するハンドラ
#ifdef USE_GAMEOVER_HANDLER
extern void gameover_handler(const string& cmd);
#endif

// Option設定が格納されたglobal object。
USI::OptionsMap Options;

// 引き分けになるまでの手数。(MaxMovesToDrawとして定義される)
// これは、"go"コマンドのときにLimits.max_game_plyに反映される。
// INT_MAXにすると残り手数を計算するときにあふれかねない。
int max_game_ply = 100000;

namespace USI
{
	// 入玉ルール
#ifdef USE_ENTERING_KING_WIN
	EnteringKingRule ekr = EKR_27_POINT;
	// 入玉ルールのUSI文字列
	std::vector<std::string> ekr_rules = { "NoEnteringKing", "CSARule24" , "CSARule27" , "TryRule" };

	// 入玉ルールがGUIから変更されたときのハンドラ
	void set_entering_king_rule(const std::string& rule)
	{
		for (size_t i = 0; i < ekr_rules.size(); ++i)
			if (ekr_rules[i] == rule)
			{
				ekr = (EnteringKingRule)i;
				break;
			}
	}
#else
	EnteringKingRule ekr = EKR_NONE;
#endif

	// --------------------
	//    読み筋の出力
	// --------------------

	  // スコアを歩の価値を100として正規化して出力する。
	std::string score_to_usi(Value v)
	{
		ASSERT_LV3(-VALUE_INFINITE < v && v < VALUE_INFINITE);

		std::stringstream s;

		// 置換表上、値が確定していないことがある。
		if (v == VALUE_NONE)
			s << "none";
		else if (abs(v) < VALUE_MATE_IN_MAX_PLY)
			s << "cp " << v * 100 / int(Eval::PawnValue);
		else if (v == -VALUE_MATE)
			// USIプロトコルでは、手数がわからないときには "mate -"と出力するらしい。
			// 手数がわからないというか詰んでいるのだが…。これを出力する方法がUSIプロトコルで定められていない。
			// ここでは"-0"を出力しておく。
			// 将棋所では検討モードは、go infiniteで呼び出されて、このときbestmoveを返さないから
			// 結局、このときのスコアは画面に表示されない。
			// ShogiGUIだと、これできちんと"+詰"と出力されるようである。
			s << "mate -0";
		else
			s << "mate " << (v > 0 ? VALUE_MATE - v : -VALUE_MATE - v);

		return s.str();
	}

	// depth : iteration深さ
	std::string pv(const Position& pos, Depth depth, Value alpha, Value beta)
	{
		std::stringstream ss;
		int elapsed = Time.elapsed() + 1;

		const auto& rootMoves = pos.this_thread()->rootMoves;
		size_t PVIdx = pos.this_thread()->PVIdx;
		size_t multiPV = std::min((size_t)Options["MultiPV"], rootMoves.size());

		uint64_t nodes_searched = Threads.nodes_searched();

		// MultiPVでは上位N個の候補手と読み筋を出力する必要がある。
		for (size_t i = 0; i < multiPV; ++i)
		{
			// この指し手のpvの更新が終わっているのか
			bool updated = (i <= PVIdx && rootMoves[i].score != -VALUE_INFINITE);

			if (depth == ONE_PLY && !updated)
				continue;

			Depth d = updated ? depth : depth - ONE_PLY;
			Value v = updated ? rootMoves[i].score : rootMoves[i].previousScore;

			// multi pv時、例えば3個目の候補手までしか評価が終わっていなくて(PVIdx==2)、このとき、
			// 3,4,5個目にあるのは前回のiterationまでずっと評価されていなかった指し手であるような場合に、
			// これらのpreviousScoreが-VALUE_INFINITE(未初期化状態)でありうる。
			// (multi pv状態で"go infinite"～"stop"を繰り返すとこの現象が発生する。おそらく置換表にhitしまくる結果ではないかと思う。)
			// なので、このとき、その評価値を出力するわけにはいかないので、この場合、その出力処理を省略するのが正しいと思う。
			// おそらく2017/09/09時点で最新のStockfishにも同様の問題があり、何らかの対策コードが必要ではないかと思う。
			// (Stockfishのテスト環境がないため、試してはいない。)
			if (v == -VALUE_INFINITE)
				continue;

			if (ss.rdbuf()->in_avail()) // 1行目でないなら連結のための改行を出力
				ss << endl;

			ss  << "info"
				<< " depth "    << d / ONE_PLY
				<< " seldepth " << rootMoves[i].selDepth
				<< " score "    << USI::score_to_usi(v);

			// これが現在探索中の指し手であるなら、それがlowerboundかupperboundかは表示させる
			if (i == PVIdx)
				ss << (v >= beta ? " lowerbound" : v <= alpha ? " upperbound" : "");

			// 将棋所はmultipvに対応していないが、とりあえず出力はしておく。
			if (multiPV > 1)
				ss << " multipv " << (i + 1);

			ss << " nodes " << nodes_searched
			   << " nps " << nodes_searched * 1000 / elapsed;

			// 置換表使用率。経過時間が短いときは意味をなさないので出力しない。
			if (elapsed > 1000)
				ss << " hashfull " << TT.hashfull();

			ss << " time " << elapsed
			   << " pv";


			// PV配列からPVを出力する。
			auto out_array_pv = [&]()
			{
				for (Move m : rootMoves[i].pv)
					ss << " " << m;
			};

			// 置換表からPVをかき集めてきてPVを出力する。
			auto out_tt_pv = [&]()
			{
				auto pos_ = const_cast<Position*>(&pos);
				Move moves[MAX_PLY + 1];
				StateInfo si[MAX_PLY];
				int ply = 0;

				while ( ply < MAX_PLY )
				{
					// 千日手はそこで終了。ただし初手はPVを出力。
					// 千日手がベストのとき、置換表を更新していないので
					// 置換表上はMOVE_NONEがベストの指し手になっている可能性があるので早めに検出する。
					auto rep = pos.is_repetition(ply);
					if (rep != REPETITION_NONE && ply >= 1)
					{
						// 千日手でPVを打ち切るときはその旨を表示
						ss << " " << rep;
						break;
					}

					Move m;

					// MultiPVを考慮して初手は置換表からではなくrootMovesから取得
					// rootMovesには宣言勝ちも含まれるので注意。
					if (ply == 0)
						m = rootMoves[i].pv[0];
					else
					{
						// 次の手を置換表から拾う。
						bool found;
						auto* tte = TT.probe(pos.state()->key(), found);

						// 置換表になかった
						if (!found)
							break;

						m = tte->move();

						// 置換表にはpsudo_legalではない指し手が含まれるのでそれを弾く。
						// 宣言勝ちでないならこれが合法手であるかのチェックが必要。
						if (m != MOVE_WIN)
						{
							m = pos.move16_to_move(m);
							if (!(pos.pseudo_legal(m) && pos.legal(m)))
								break;
						}
					}

#if defined (USE_ENTERING_KING_WIN)
					// 宣言勝ちである
					if (m == MOVE_WIN)
					{
						// これが合法手であるなら宣言勝ちであると出力。
						if (pos.DeclarationWin() != MOVE_NONE)
							ss << " " << MOVE_WIN;

						break;
					}
#endif

					moves[ply] = m;
					ss << " " << m;

					pos_->do_move(m, si[ply]);
					++ply;
				}
				while (ply > 0)
					pos_->undo_move(moves[--ply]);
			};

#if !defined (USE_TT_PV)
			// 検討用のPVを出力するモードなら、置換表からPVをかき集める。
			// (そうしないとMultiPV時にPVが欠損することがあるようだ)
			// fail-highのときにもPVを更新しているのが問題ではなさそう。
			// Stockfish側の何らかのバグかも。
			if (Search::Limits.consideration_mode)
				out_tt_pv();
			else
				out_array_pv();

#else
			// 置換表からPVを出力するモード。
			// ただし、probe()するとTTEntryのgenerationが変わるので探索に影響する。
			// benchコマンド時、これはまずいのでbenchコマンド時にはこのモードをオフにする。
			if (Search::Limits.bench)
				out_array_pv();
			else
				out_tt_pv();

#endif
		}

		return ss.str();
	}

	// --------------------
	//     USI::Option
	// --------------------

	// この関数はUSI::init()から起動時に呼び出されるだけ。
	void Option::operator<<(const Option& o)
	{
		static size_t insert_order = 0;
		*this = o;
		idx = insert_order++; // idxは0から連番で番号を振る
	}

	// idxの値を書き換えないoperator "<<"
	void Option::overwrite(const Option& o)
	{
		auto idx_ = idx; // backup
		*this = o;
		idx = idx_; // restore
	}

	// 思考エンジンがGUIからの"usi"に対して返す"option ..."文字列から
	// Optionオブジェクトを構築して、それをOptions[]に突っ込む。
	// "engine_options.txt"というファイルの各行からOptionオブジェクト構築して
	// Options[]の値を上書きするためにこの関数が必要。
	// "option name USI_Hash type spin default 256"
	// のような文字列が引数として渡される。
	void build_option(string line)
	{
		LineScanner scanner(line);
		if (scanner.get_text() != "option") return;

		string name, value, option_type;
		int64_t min_value = 0, max_value = 1;
		vector<string> combo_list;
		while (!scanner.eof())
		{
			auto token = scanner.get_text();
			if (token == "name") name = scanner.get_text();
			else if (token == "type") option_type = scanner.get_text();
			else if (token == "default") value = scanner.get_text();
			else if (token == "min") min_value = stoll(scanner.get_text());
			else if (token == "max") max_value = stoll(scanner.get_text());
			else if (token == "var") {
				auto varText = scanner.get_text();
				combo_list.push_back(varText);
			}
			else {
				cout << "Error : invalid command: " << token << endl;
			}
		}

		if (Options.count(name) != 0)
		{
			// typeに応じたOptionの型を生成して代入する。このときに "<<"を用いるとidxが変わってしまうので overwriteで代入する。
			if (option_type == "check") Options[name].overwrite(Option(value == "true"));
			else if (option_type == "spin") Options[name].overwrite(Option(stoll(value), min_value, max_value));
			else if (option_type == "string") Options[name].overwrite(Option(value.c_str()));
			else if (option_type == "combo") Options[name].overwrite(Option(combo_list, value));
		}
		else
			cout << "Error : option name not found : " << name << endl;

	}

	// カレントフォルダに"engine_option.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
	void read_engine_options()
	{
		ifstream ifs("engine_options.txt");
		if (!ifs.fail())
		{
			string str;
			while (getline(ifs, str))
				build_option(str);
		}
	}


	// optionのdefault値を設定する。
	void init(OptionsMap& o)
	{
		// Hash上限。32bitモードなら2GB、64bitモードなら1024GB
		const int MaxHashMB = Is64Bit ? 1024 * 1024 : 2048;

		// 並列探索するときのスレッド数
		// CPUの搭載コア数をデフォルトとすべきかも知れないが余計なお世話のような気もするのでしていない。

		o["Threads"] << Option(4, 1, 512, [](const Option& o) { Threads.set(o); });

		// USIプロトコルでは、"USI_Hash"なのだが、
		// 置換表サイズを変更しての自己対戦などをさせたいので、
		// 片方だけ変更できなければならない。
		// ゆえにGUIでの対局設定は無視して、思考エンジンの設定ダイアログのところで
		// 個別設定が出来るようにする。

#if !defined(MATE_ENGINE)
		o["Hash"] << Option(16, 1, MaxHashMB, [](const Option&o) { TT.resize(o); });

		// その局面での上位N個の候補手を調べる機能
		o["MultiPV"] << Option(1, 1, 800);

		// 弱くするために調整する。20なら手加減なし。0が最弱。
		o["SkillLevel"] << Option(20, 0, 20);
#else
		o["Hash"] << Option(4096, 1, MaxHashMB);
#endif

		// cin/coutの入出力をファイルにリダイレクトする
		o["WriteDebugLog"] << Option(false, [](const Option& o) { start_logger(o); });

		// ネットワークの平均遅延時間[ms]
		// この時間だけ早めに指せばだいたい間に合う。
		// 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
		o["NetworkDelay"] << Option(120, 0, 10000);

		// ネットワークの最大遅延時間[ms]
		// 切れ負けの瞬間だけはこの時間だけ早めに指す。
		// 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
		o["NetworkDelay2"] << Option(1120, 0, 10000);

		// 最小思考時間[ms]
		o["MinimumThinkingTime"] << Option(2000, 1000, 100000);

		// 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
		// 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
		// 対人のときに短めに設定して強制的に早指しにすることが出来る。
		o["SlowMover"] << Option(100, 1, 1000);

		// 引き分けまでの最大手数。256手ルールのときに256を設定すると良い。
		// 0なら無制限。(桁あふれすると良くないので内部的には100000として扱う)
		o["MaxMovesToDraw"] << Option(0, 0, 100000, [](const Option& o) { max_game_ply = (o == 0) ? 100000 : (int)o; });

		// 探索深さ制限。0なら無制限。
		o["DepthLimit"] << Option(0, 0, INT_MAX);

		// 探索ノード制限。0なら無制限。
		o["NodesLimit"] << Option(0, 0, INT64_MAX);

		// 引き分けを受け入れるスコア
		// 歩を100とする。例えば、この値を100にすると引き分けの局面は評価値が -100とみなされる。

		// 千日手での引き分けを回避しやすくなるように、デフォルト値を2に変更した。[2017/06/03]
		// ちなみに、2にしてあるのは、
		//  int contempt = Options["Contempt"] * PawnValue / 100; でPawnValueが100より小さいので
		// 1だと切り捨てられてしまうからである。

		o["Contempt"] << Option(2, -30000, 30000);

		// Contemptの設定値を先手番から見た値とするオプション。Stockfishからの独自拡張。
		// 先手のときは千日手を狙いたくなくて、後手のときは千日手を狙いたいような場合、
		// このオプションをオンにすれば、Contemptをそういう解釈にしてくれる。
		// この値がtrueのときは、Contemptを常に先手から見たスコアだとみなす。

		o["ContemptFromBlack"] << Option(false);


#if defined (USE_ENTERING_KING_WIN)
		// 入玉ルール
		o["EnteringKingRule"] << Option(ekr_rules, ekr_rules[EKR_27_POINT], [](const Option& o) { set_entering_king_rule(o); });
#endif
		// 評価関数フォルダ。これを変更したとき、評価関数を次のisreadyタイミングで読み直す必要がある。
		o["EvalDir"] << Option("eval", [](const USI::Option&o) { load_eval_finished = false; });

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32) && \
	 (defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_KPPPT) || defined(EVAL_KPPP_KKPT) || defined(EVAL_KKPP_KKPT) || \
	defined(EVAL_KPP_KKPT_FV_VAR) || defined(EVAL_KKPPT) ||defined(EVAL_EXPERIMENTAL) || defined(EVAL_HELICES) || defined(EVAL_NABLA) )
		// 評価関数パラメーターを共有するか
		// 異種評価関数との自己対局のときにこの設定で引っかかる人が後を絶たないのでデフォルトでオフにする。
		o["EvalShare"] << Option(false);
#endif

#if defined(EVAL_LEARN)
		// isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
		// test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
		// このコマンドの実行前に異常終了してしまう。
		// そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
		// test evalconvertコマンドを叩く。
		o["SkipLoadingEval"] << Option(false);
#endif

		// 各エンジンがOptionを追加したいだろうから、コールバックする。
		USI::extra_option(o);

		// カレントフォルダに"engine_option.txt"があればそれをオプションとしてOptions[]の値をオーバーライドする機能。
		read_engine_options();
	}


	// USIプロトコル経由で値を設定されたときにそれをcurrentValueに反映させる。
	Option& Option::operator=(const string& v) {

		ASSERT_LV1(!type.empty());

		// 範囲外なら設定せずに返る。
		// "EvalDir"などでstringの場合は空の文字列を設定したいことがあるので"string"に対して空の文字チェックは行わない。
		if (((type != "button" && type != "string") && v.empty())
			|| (type == "check" && v != "true" && v != "false")
			|| (type == "spin" && (stoll(v) < min || stoll(v) > max)))
			return *this;

		// ボタン型は値を設定するものではなく、単なるトリガーボタン。
		// ボタン型以外なら入力値をcurrentValueに反映させてやる。
		if (type != "button")
			currentValue = v;

		// 値が変化したのでハンドラを呼びだす。
		if (on_change)
			on_change(*this);

		return *this;
	}

	std::ostream& operator<<(std::ostream& os, const OptionsMap& om)
	{
		// idxの順番を守って出力する
		for (size_t idx = 0; idx < om.size(); ++idx)
			for (const auto& it : om)
				if (it.second.idx == idx)
				{
					const Option& o = it.second;
					os << "option name " << it.first << " type " << o.type;

					if (o.type != "button")
						os << " default " << o.defaultValue;

					// コンボボックス
					if (o.list.size())
						for (auto v : o.list)
							os << " var " << v;

					if (o.type == "spin")
						os << " min " << o.min << " max " << o.max;
					os << endl;
					break;
				}

		return os;
	}
}

// --------------------
// USI関係のコマンド処理
// --------------------

// check sumを計算したとき、それを保存しておいてあとで次回以降、整合性のチェックを行なう。
u64 eval_sum;

// is_ready_cmd()を外部から呼び出せるようにしておく。(benchコマンドなどから呼び出したいため)
// 局面は初期化されないので注意。
void is_ready(bool skipCorruptCheck)
{
	// 評価関数の読み込みなど時間のかかるであろう処理はこのタイミングで行なう。
	// 起動時に時間のかかる処理をしてしまうと将棋所がタイムアウト判定をして、思考エンジンとしての認識をリタイアしてしまう。
	if (!load_eval_finished)
	{
		// 評価関数の読み込み
		Eval::load_eval();

		// チェックサムの計算と保存(その後のメモリ破損のチェックのため)
		eval_sum = Eval::calc_check_sum();

		// ソフト名の表示
		Eval::print_softname(eval_sum);

		load_eval_finished = true;

	}
	else
	{
		// メモリが破壊されていないかを調べるためにチェックサムを毎回調べる。
		// 時間が少しもったいない気もするが.. 0.1秒ぐらいのことなので良しとする。
		if (!skipCorruptCheck && eval_sum != Eval::calc_check_sum())
			sync_cout << "Error! : EVAL memory is corrupted" << sync_endl;
	}

	// isreadyに対してはreadyokを返すまで次のコマンドが来ないことは約束されているので
	// このタイミングで各種変数の初期化もしておく。

	TT.resize(Options["Hash"]);
	Search::clear();
	Time.availableNodes = 0;

	Threads.received_go_ponder = false;
	Threads.stop = false;
}

// isreadyコマンド処理部
void is_ready_cmd(Position& pos, StateListPtr& states)
{
	// 対局ごとに"isready","usinewgame"の両方が来るはずだが、
	// "isready"は起動後に1度だけしか来ないGUI実装がありうるかも知れない。
	// 将棋では、"isready"が毎回来るようなので、"usinewgame"のほうは無視して、
	// "isready"に応じて評価関数、定跡、探索部を初期化する。

	is_ready();

	// Positionコマンドが送られてくるまで評価値の全計算をしていないの気持ち悪いのでisreadyコマンドに対して
	// evalの値を返せるようにこのタイミングで平手局面で初期化してしまう。

	// 新しく渡す局面なので古いものは捨てて新しいものを作る。
	states = StateListPtr(new StateList(1));
	pos.set_hirate(&states->back(),Threads.main());

	sync_cout << "readyok" << sync_endl;
}

// "position"コマンド処理部
void position_cmd(Position& pos, istringstream& is , StateListPtr& states)
{
	Move m;
	string token, sfen;

	is >> token;

	if (token == "startpos")
	{
		// 初期局面として初期局面のFEN形式の入力が与えられたとみなして処理する。
		sfen = SFEN_HIRATE;
		is >> token; // もしあるなら"moves"トークンを消費する。
	}
	// 局面がfen形式で指定されているなら、その局面を読み込む。
	// UCI(チェスプロトコル)ではなくUSI(将棋用プロトコル)だとここの文字列は"fen"ではなく"sfen"
	// この"sfen"という文字列は省略可能にしたいので..
	else {
		if (token != "sfen")
			sfen += token + " ";
		while (is >> token && token != "moves")
			sfen += token + " ";
	}

	// 新しく渡す局面なので古いものは捨てて新しいものを作る。
	states = StateListPtr(new StateList(1));
	pos.set(sfen , &states->back() , Threads.main());

	// 指し手のリストをパースする(あるなら)
	while (is >> token && (m = move_from_usi(pos, token)) != MOVE_NONE)
	{
		// 1手進めるごとにStateInfoが積まれていく。これは千日手の検出のために必要。
		states->emplace_back();
		if (m == MOVE_NULL) // do_move に MOVE_NULL を与えると死ぬので
			pos.do_null_move(states->back());
		else
			pos.do_move(m, states->back());
	}
}

// "setoption"コマンド応答。
void setoption_cmd(istringstream& is)
{
	string token, name, value;

	while (is >> token && token != "value")
		// "name"トークンはあってもなくても良いものとする。(手打ちでコマンドを打つときには省略したい)
		if (token != "name")
			// スペース区切りで長い名前のoptionを使うことがあるので2つ目以降はスペースを入れてやる
			name += (name.empty() ? "" : " ") + token;

	// valueの後ろ。スペース区切りで複数文字列が来ることがある。
	while (is >> token)
		value += (value.empty() ? "" : " ") + token;

	if (Options.count(name))
		Options[name] = value;
	else {
		// USI_HashとUSI_Ponderは無視してやる。
		if (name != "USI_Hash" && name != "USI_Ponder")
			// この名前のoptionは存在しなかった
			sync_cout << "Error! : No such option: " << name << sync_endl;
	}
}

// getoptionコマンド応答(USI独自拡張)
// オプションの値を取得する。
void getoption_cmd(istringstream& is)
{
	// getoption オプション名
	string name = "";
	is >> name;

	// すべてを出力するモード
	bool all = name == "";

	for (auto& o : Options)
	{
		// 大文字、小文字を無視して比較。また、nameが指定されていなければすべてのオプション設定の現在の値を表示。
		if ((!_stricmp(name.c_str(), o.first.c_str())) || all)
		{
			sync_cout << "Options[" << o.first << "] == " << (string)Options[o.first] << sync_endl;
			if (!all)
				return;
		}
	}
	if (!all)
		sync_cout << "No such option: " << name << sync_endl;
}


// go()は、思考エンジンがUSIコマンドの"go"を受け取ったときに呼び出される。
// この関数は、入力文字列から思考時間とその他のパラメーターをセットし、探索を開始する。
void go_cmd(const Position& pos, istringstream& is , StateListPtr& states) {

	Search::LimitsType limits;
	string token;
	bool ponderMode = false;

	// 思考開始時刻の初期化。なるべく早い段階でこれをしておかないとサーバー時間との誤差が大きくなる。
	Time.reset();

	// 入玉ルール
	limits.enteringKingRule = USI::ekr;

	// 終局(引き分け)になるまでの手数
	limits.max_game_ply = max_game_ply;

	// エンジンオプションによる探索制限(0なら無制限)
	if (Options["DepthLimit"] >= 0)    limits.depth = (int)Options["DepthLimit"];
	if (Options["NodesLimit"] >= 0)    limits.nodes = (u64)Options["NodesLimit"];

	while (is >> token)
	{
		// 探索すべき指し手。(探索開始局面から特定の初手だけ探索させるとき)
		if (token == "searchmoves")
			// 残りの指し手すべてをsearchMovesに突っ込む。
			while (is >> token)
				limits.searchmoves.push_back(move_from_usi(pos, token));

		// 先手、後手の残り時間。[ms]
		else if (token == "wtime")     is >> limits.time[WHITE];
		else if (token == "btime")     is >> limits.time[BLACK];

		// フィッシャールール時における時間
		else if (token == "winc")      is >> limits.inc[WHITE];
		else if (token == "binc")      is >> limits.inc[BLACK];

		// "go rtime 100"だと100～300[ms]思考する。
		else if (token == "rtime")     is >> limits.rtime;

		// 秒読み設定。
		else if (token == "byoyomi") {
			int t = 0;
			is >> t;

			// USIプロトコルで送られてきた秒読み時間より少なめに思考する設定
			// ※　通信ラグがあるときに、ここで少なめに思考しないとタイムアップになる可能性があるので。

			// t = std::max(t - Options["ByoyomiMinus"], Time::point(0));

			// USIプロトコルでは、これが先手後手同じ値だと解釈する。
			limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = t;
		}
		// この探索深さで探索を打ち切る
		else if (token == "depth")     is >> limits.depth;

		// この探索ノード数で探索を打ち切る
		else if (token == "nodes")     is >> limits.nodes;

		// 詰み探索。"UCI"プロトコルではこのあとには手数が入っており、その手数以内に詰むかどうかを判定するが、
		// "USI"プロトコルでは、ここは探索のための時間制限に変更となっている。
		else if (token == "mate") {
			is >> token;
			if (token == "infinite")
				limits.mate = INT32_MAX;
			else
				limits.mate = stoi(token);
		}

		// 時間無制限。
		else if (token == "infinite")  limits.infinite = 1;

		// ponderモードでの思考。
		else if (token == "ponder")
		{
			ponderMode = true;

			// 試合開始後、一度でも"go ponder"が送られてきたら、それを記録しておく。
			Threads.received_go_ponder = true;
		}
	}

	// goコマンド、デバッグ時に使うが、そのときに"go btime XXX wtime XXX byoyomi XXX"と毎回入力するのが面倒なので
	// デフォルトで1秒読み状態で呼び出されて欲しい。
	if (limits.byoyomi[BLACK] == 0 && limits.inc[BLACK] == 0 && limits.time[BLACK] == 0 && limits.rtime == 0)
		limits.byoyomi[BLACK] = limits.byoyomi[WHITE] = 1000;

	Threads.start_thinking(pos, states , limits , ponderMode);
}

// --------------------
// テスト用にqsearch(),search()を直接呼ぶ
// --------------------

#if defined(EVAL_LEARN)
void qsearch_cmd(Position& pos)
{
	cout << "qsearch : ";
	auto pv = Learner::qsearch(pos);
	cout << "Value = " << pv.first << " , PV = ";
	for (auto m : pv.second)
		cout << m << " ";
	cout << endl;
}

void search_cmd(Position& pos, istringstream& is)
{
	string token;
	int depth = 1;
	int multi_pv = (int)Options["MultiPV"];
	while (is >> token)
	{
		if (token == "depth")
			is >> depth;
		if (token == "multipv")
			is >> multi_pv;
	}

	cout << "search depth = " << depth << " , multi_pv = " << multi_pv << " : ";
	auto pv = Learner::search(pos , depth , multi_pv);
	cout << "Value = " << pv.first << " , PV = ";
	for (auto m : pv.second)
		cout << m << " ";
	cout << endl;
}

#endif

// --------------------
// 　　USI応答部
// --------------------

// USI応答部本体
void USI::loop(int argc, char* argv[])
{
	// 探索開始局面(root)を格納するPositionクラス
	Position pos;

	string cmd, token;

	// 局面を遡るためのStateInfoのlist。
	StateListPtr states(new StateList(1));

	// 先行入力されているコマンド
	// コマンドは前から取り出すのでqueueを用いる。
	queue<string> cmds;

	// ファイルからコマンドの指定
	if (argc >= 3 && string(argv[1]) == "file")
	{
		vector<string> cmds0;
		read_all_lines(argv[2], cmds0);

		// queueに変換する。
		for (auto c : cmds0)
			cmds.push(c);

	} else {

		// 引数として指定されたものを一つのコマンドとして実行する機能
		// ただし、','が使われていれば、そこでコマンドが区切れているものとして解釈する。

		for (int i = 1; i < argc; ++i)
		{
			string s = argv[i];

			// sから前後のスペースを除去しないといけない。
			while (*s.rbegin() == ' ') s.pop_back();
			while (*s.begin() == ' ') s = s.substr(1, s.size() - 1);

			if (s != ",")
				cmd += s + " ";
			else
			{
				cmds.push(cmd);
				cmd = "";
			}
		}
		if (cmd.size() != 0)
			cmds.push(cmd);
	}

	do
	{
		if (cmds.size() == 0)
		{
			if (!getline(cin, cmd)) // 入力が来るかEOFがくるまでここで待機する。
				cmd = "quit";
		} else {
			// 積んであるコマンドがあるならそれを実行する。
			// 尽きれば"quit"だと解釈してdoループを抜ける仕様にすることはできるが、
			// そうしてしまうとgoコマンド(これはノンブロッキングなので)の最中にquitが送られてしまう。
			// ただ、
			// YaneuraOu-mid.exe bench,quit
			// のようなことは出来るのでPGOの役には立ちそうである。
			cmd = cmds.front();
			cmds.pop();
		}

		istringstream is(cmd);

		token = "";
		is >> skipws >> token;

		if (token == "quit" || token == "stop" || token == "gameover")
		{
			// USIプロトコルにはUCIプロトコルから、
			// gameover win | lose | draw
			// が追加されているが、stopと同じ扱いをして良いと思う。
			// これハンドルしておかないとponderが停止しなくて困る。
			// gameoverに対してbestmoveは返すべきではないのかも知れないが、
			// それを言えばstopにだって…。

#ifdef USE_GAMEOVER_HANDLER
			// "gameover"コマンドに対するハンドラを呼び出したいのか？
			if (token == "gameover")
				gameover_handler(cmd);
#endif

			// "go infinite" , "go ponder"などで思考を終えて寝てるかも知れないが、
			// そいつらはThreads.stopを待っているので問題ない。
			Threads.stop = true;

		} else if (token == "ponderhit")
		{
			Time.reset_for_ponderhit(); // ponderhitから計測しなおすべきである。
			Threads.ponder = false; // 通常探索に切り替える。
		}

		// 与えられた局面について思考するコマンド
		else if (token == "go") go_cmd(pos, is , states);

		// (思考などに使うための)開始局面(root)を設定する
		else if (token == "position") position_cmd(pos, is , states);

		// 起動時いきなりこれが飛んでくるので速攻応答しないとタイムアウトになる。
		else if (token == "usi")
			sync_cout << engine_info() << Options << "usiok" << sync_endl;

		// オプションを設定する
		else if (token == "setoption") setoption_cmd(is);

		// オプションを取得する(USI独自拡張)
		else if (token == "getoption") getoption_cmd(is);

		// 思考エンジンの準備が出来たかの確認
		else if (token == "isready") is_ready_cmd(pos,states);

		// ユーザーによる実験用コマンド。user.cppのuser()が呼び出される。
		else if (token == "user") user_test(pos, is);

		// 現在の局面を表示する。(デバッグ用)
		else if (token == "d") cout << pos << endl;

		// 指し手生成祭りの局面をセットする。
		else if (token == "matsuri") pos.set("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1",&states->back(),Threads.main());

		// "position sfen"の略。
		else if (token == "sfen") position_cmd(pos, is , states);

		// ログファイルの書き出しのon
		else if (token == "log") start_logger(true);

		// 現在の局面について評価関数を呼び出して、その値を返す。
		else if (token == "eval") cout << "eval = " << Eval::compute_eval(pos) << endl;
		else if (token == "evalstat") Eval::print_eval_stat(pos);

#if defined(EVAL_LEARN)
		// テスト用にqsearch(),search()を直接呼ぶコマンド
		else if (token == "qsearch") qsearch_cmd(pos);
		else if (token == "search") search_cmd(pos,is);
#endif

		// この局面での指し手をすべて出力
		else if (token == "moves") {
			for (auto m : MoveList<LEGAL_ALL>(pos))
				cout << m.move << ' ';
			cout << endl;
		}

		// この局面が詰んでいるかの判定
		else if (token == "mated") cout << pos.is_mated() << endl;

		// この局面のhash keyの値を出力
		else if (token == "key") cout << hex << pos.state()->key() << dec << endl;

#if defined(MATE_1PLY) && defined(LONG_EFFECT_LIBRARY)
		// この局面での1手詰め判定
		else if (token == "mate1") cout << pos.mate1ply() << endl;
#endif

		// ベンチコマンド(これは常に使える)
		else if (token == "bench") bench_cmd(pos, is);

#ifdef ENABLE_TEST_CMD
		// 指し手生成のテスト
		else if (token == "s") generate_moves_cmd(pos);

		// パフォーマンステスト(Stockfishにある、合法手N手で到達できる局面を求めるやつ)
		else if (token == "perft") perft(pos, is);

		// テストコマンド
		else if (token == "test") test_cmd(pos, is);

#ifdef MATE_ENGINE
		else if (token == "test_mate_engine") test_mate_engine_cmd(pos, is);
#endif
#endif

#ifdef ENABLE_MAKEBOOK_CMD
		// 定跡を作るコマンド
		else if (token == "makebook") Book::makebook_cmd(pos, is);
#endif

#if defined (EVAL_LEARN)
		else if (token == "gensfen") Learner::gen_sfen(pos, is);
		else if (token == "learn") Learner::learn(pos, is);
#if defined (USE_GENSFEN2018)
		// 開発中の教師局面生成コマンド
		else if (token == "gensfen2018") Learner::gen_sfen2018(pos, is);
#endif

#endif
		// "usinewgame"はゲーム中にsetoptionなどを送らないことを宣言するためのものだが、
		// 我々はこれに関知しないので単に無視すれば良い。
		else if (token == "usinewgame") continue;

		else
		{
			//    簡略表現として、
			//> threads 1
			//      のように指定したとき、
			//> setoption name Threads value 1
			//      と等価なようにしておく。

			if (!token.empty())
			{
				string value;
				is >> value;

				for (auto& o : Options)
				{
					// 大文字、小文字を無視して比較。
					if (!_stricmp(token.c_str(), o.first.c_str()))
					{
						Options[o.first] = value;
						sync_cout << "Options[" << o.first << "] = " << value << sync_endl;

						goto OPTION_FOUND;
					}
				}
				sync_cout << "No such option: " << token << sync_endl;
			OPTION_FOUND:;
			}
		}

	} while (token != "quit");

	// quitが来た時点ではまだ探索中かも知れないのでmain threadの停止を待つ。
	Threads.main()->wait_for_search_finished();
}

// --------------------
// USI関係の記法変換部
// --------------------

// USIの指し手文字列などに使われている盤上の升を表す文字列をSquare型に変換する
// 変換できなかった場合はSQ_NBが返る。
Square usi_to_sq(char f, char r)
{
	File file = toFile(f);
	Rank rank = toRank(r);

	if (is_ok(file) && is_ok(rank))
		return file | rank;

	return SQ_NB;
}

// usi形式から指し手への変換。本来この関数は要らないのだが、
// 棋譜を大量に読み込む都合、この部分をそこそこ高速化しておきたい。
Move move_from_usi(const string& str)
{
	// さすがに3文字以下の指し手はおかしいだろ。
	if (str.length() <= 3)
		return MOVE_NONE;

	Square to = usi_to_sq(str[2], str[3]);
	if (!is_ok(to))
		return MOVE_NONE;

	bool promote = str.length() == 5 && str[4] == '+';
	bool drop = str[1] == '*';

	Move move = MOVE_NONE;
	if (!drop)
	{
		Square from = usi_to_sq(str[0], str[1]);
		if (is_ok(from))
			move = promote ? make_move_promote(from, to) : make_move(from, to);
	} else
	{
		for (int i = 1; i <= 7; ++i)
			if (PieceToCharBW[i] == str[0])
			{
				move = make_move_drop((Piece)i, to);
				break;
			}
	}

	return move;
}


// 局面posとUSIプロトコルによる指し手を与えて
// もし可能なら等価で合法な指し手を返す。(合法でないときはMOVE_NONEを返す)
Move move_from_usi(const Position& pos, const std::string& str)
{
	// 全合法手のなかからusi文字列に変換したときにstrと一致する指し手を探してそれを返す
	//for (const ExtMove& ms : MoveList<LEGAL_ALL>(pos))
	//  if (str == move_to_usi(ms.move))
	//    return ms.move;

	// ↑のコードは大変美しいコードではあるが、棋譜を大量に読み込むときに時間がかかるうるのでもっと高速な実装をする。

	if (str == "resign")
		return MOVE_RESIGN;

	if (str == "win")
		return MOVE_WIN;

	// パス(null move)入力への対応 {UCI: "0000", GPSfish: "pass"}
	if (str == "0000" || str == "null" || str == "pass")
		return MOVE_NULL;

	// usi文字列をmoveに変換するやつがいるがな..
	Move move = move_from_usi(str);

	// 上位bitに駒種を入れておかないとpseudo_legal()で引っかかる。
	move = pos.move16_to_move(move);

#if defined(MUST_CAPTURE_SHOGI_ENGINE)
	// 取る一手将棋は合法手かどうかをGUI側でチェックしてくれないから、
	// 合法手かどうかのチェックを入れる。
	if (!MoveList<LEGAL>(pos).contains(move))
		sync_cout << "info string Error!! Illegal Move = " << move << sync_endl;
#endif

	if (pos.pseudo_legal(move) && pos.legal(move))
		return move;

	// いかなる状況であろうとこのような指し手はエラー表示をして弾いていいと思うが…。
	// cout << "\nIlligal Move : " << str << "\n";

	return MOVE_NONE;
}
