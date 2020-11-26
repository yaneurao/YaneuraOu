//
// 教師局面の生成ルーチン2019年度版
//

#include "../config.h"

#if defined(EVAL_LEARN) && defined(GENSFEN2019)

#include <unordered_set>

using namespace std;

namespace
{
	// C#のstring.Split()みたいなの
	vector<string> split(const string &s, char delim) {
		vector<string> elems;
		stringstream ss(s);
		string item;
		while (getline(ss, item, delim)) {
		if (!item.empty()) {
				elems.push_back(item);
			}
		}
		return elems;
	}
}

namespace Learner {

	// -----------------------------------
	//  棋譜を生成するworker(スレッドごと)
	// -----------------------------------

	// 複数スレッドでsfenを生成するためのクラス
	struct MultiThinkGenSfen2019 : public MultiThink
	{
		// hash_size : NodeInfoを格納するためのhash sizeを指定する。単位は[MB]。
		// メモリに余裕があるなら大きめの値を指定するのが好ましい。
		MultiThinkGenSfen2019(SfenWriter& sw_ , int search_depth_ , u64 nodes_limit_ , const string& book_file_name_)
		: sw(sw_) , search_depth(search_depth_) , nodes_limit(nodes_limit_) , book_file_name(book_file_name_){}

		// コンストラクタとは別に初期化用のコード。(write_maxplyなどを設定後に呼び出す)
		// このタイミングで定跡ファイルから読み込む
		void init()
		{
			// PCを並列化してgensfenするときに同じ乱数seedを引いていないか確認用の出力。
			std::cout << endl << prng << std::endl;

			cout << "read book" << endl;
			if (FileOperator::ReadAllLines(book_file_name, my_book).is_not_ok())
			{
				cout << endl << "info string Error! read book error!";
				// 定跡ファイルがないと、開始局面に困るのでこの時点でexitする。				
				exit(0);
			}
			else
			{
				// 丸読みして、局面に落とし込む＆重複除去する
				cout << "..done" << endl;

				parse_book_file();
			}
		}

		virtual void thread_worker(size_t thread_id);
		void start_file_write_worker() { sw.start_file_write_worker(); }

		// 読み込んだ定跡ファイルをparseして各局面を取得する。
		void parse_book_file();

		// 開始局面をランダムに一つ選択する。
		void set_start_pos(Position&pos, Thread& th , StateInfo* si);

		// 1手進める関数
		void do_move(Position& pos , Move move, StateInfo* states)
		{
			ASSERT_LV3(is_ok(move) && pos.pseudo_legal(move) && pos.legal(move));

			pos.do_move(move, states[pos.game_ply()]);

			ASSERT_LV3(pos.pos_is_ok());

			//			Eval::evaluate_with_no_return(pos);
			Eval::evaluate(pos);

		};

		// 生成する局面の評価値の上限
		int eval_limit;

		// 書き出す局面のply(初期局面からの手数)の最大。
		int write_minply;
		int write_maxply;

		// 探索ノード数
		u64 nodes_limit;

		// 探索depth
		int search_depth;

		// 定跡ファイル名
		string book_file_name;

		// sfenの書き出し器
		SfenWriter& sw;

		// 定跡
		vector<string> my_book;

		// 定跡の各局面
		vector<PackedSfenValue> my_book_sfens;
	};

	void MultiThinkGenSfen2019::parse_book_file()
	{
		// -- 定跡の各局面

		// unordered_setで用いるhashとequal関数

		struct PackedSfenValueHash {
			size_t operator()(const PackedSfenValue & s) const {
				// packされたバイナリの全部の値をxorして返す程度でいいや…。
				size_t tmp = 0;
				for(int i=0;i<(int)(sizeof(PackedSfen) / sizeof(size_t)) ;++i)
					tmp ^= ((size_t*)&s.sfen.data)[i];
				return tmp;
			}
		};
		struct PackedSfenValueEqual {
			bool operator()(const PackedSfenValue &left, const PackedSfenValue&right) const
			{
				// 局面が一致すればあとは無視する。
				return memcmp(&left.sfen, &right.sfen, sizeof(PackedSfen)) == 0;
			}
		};

		// unordered_setを用いて局面の重複除去を行う。
		unordered_set<PackedSfenValue, PackedSfenValueHash , PackedSfenValueEqual> book_sfens;

		// -- 1手進める関数

		Position pos;
		auto th = Threads.main();

		const int MAX_PLY2 = write_maxply;
		std::vector<StateInfo> states_(MAX_PLY2 + MAX_PLY /* == search_depth + α */);
		StateInfo* const states = &states_[0];

		Move move;
		u64 count = 0; // 局面数
		u64 line_number = 0; // 定跡ファイル行番号

		auto my_do_move = [&move, &pos, &states ,&count , &line_number , &book_sfens ]()
		{
			ASSERT_LV3(is_ok(move) && pos.pseudo_legal(move) && pos.legal(move));

			ASSERT_LV3(pos.game_ply() != 0);
			pos.do_move(move, states[pos.game_ply()]);

			ASSERT_LV3(pos.pos_is_ok());

			// 評価値使わないので、評価関数の計算しなくていいや。
//			Eval::evaluate(pos);

			// 局面の保存(手数も保存しておかないといけない)
			PackedSfenValue ps;
			pos.sfen_pack(ps.sfen);
			ps.gamePly = pos.game_ply();
			ASSERT_LV3(ps.gamePly != 0);

			// すでに挿入済であればこの局面は無視する。
			if (book_sfens.find(ps) != book_sfens.end())
				return;

			book_sfens.insert(ps);
			++count;
		};

		auto out_status = [&count,&line_number]
		{
			cout << count << " positions , line_number = " << line_number << endl;
		};

		ASSERT_LV3(Search::Limits.enteringKingRule = EKR_27_POINT);

		for (auto book_line : my_book)
		{
			if ((++line_number % 1000) == 0)
				out_status();

			auto book_moves = split(book_line, ' ');

			pos.set_hirate(&states[0], th);

			// "startpos moves"を読み飛ばしてそこ以降の指し手文字列で指し手を進める
			for (int book_move_index = 2; book_move_index < (int)book_moves.size()
					&& pos.game_ply() <= MAX_PLY2 - 32 /* あまり直前の局面だと即シミュレーションが終了してしまうので… */
					; ++book_move_index)
			{
				// /* 詰みの局面もゴミでしかない。1手詰め、宣言勝ちの局面も除外。*/
				if (pos.is_mated()
					|| (!pos.checkers() && pos.mate1ply() != MOVE_NONE)
					|| pos.DeclarationWin() != MOVE_NONE
					)
					break;

				// 定跡の指し手で一手進める
				auto book_move = book_moves[book_move_index];
				move = USI::to_move(pos, book_move);
				// 信用できない定跡の場合、このチェックが必要。
				if (!is_ok(move) || !pos.pseudo_legal(move) || !pos.legal(move))
					break;

				my_do_move();

#if 1
				// 32手目までとする。
				// ・Apery(SDT5)は手数制限をしていないらしい。
				// ・tanuki-(2018)は、手数制限をしているらしい。
				// 手数制限をしないと終盤の局面に偏ってしまうように思うのだが…。
				if (pos.game_ply() > 32)
					break;
#endif
			}
		}

		// vectorに局面をcopy
		my_book_sfens.clear();
		for(auto& it : book_sfens)
			my_book_sfens.push_back(it);

		out_status();
	}

	void MultiThinkGenSfen2019::set_start_pos(Position&pos, Thread& th , StateInfo* states)
	{
	Retry:;

		// 定跡の局面を一つ取り出す
		auto& ps = my_book_sfens[prng.rand(my_book_sfens.size())];
		ASSERT_LV3(ps.gamePly != 0);
		pos.set_from_packed_sfen(ps.sfen , &states[0 /* ここは確実に空いてる */] , &th , /*mirror = */ false , ps.gamePly);

		// ランダムムーブで1手進める
		// 実現確率が高い局面の周辺局面ということならランダムムーブ1手がベスト

		// ランダムムーブの手数
		const int random_move_ply = 2;

		for(int i=0;i< random_move_ply;++i)
		{
			Move move = MOVE_NONE;
			MoveList<LEGAL> legal_moves(pos);
			if (legal_moves.size() == 0)
				goto Retry;
				// なぜか合法手がないので局面の選択に戻る。

#if 0
			// 1/2の確率で玉を移動させる指し手を選択する。(Apery(SDT5)のアイデア)
			// 玉が移動している局面を開始局面にしたほうがhalfKPなどでは0になる要素が減って良いと考えられる。
			if (prng.rand(2) == 0)
			{
				vector<Move> moves;
				for (auto m : legal_moves)
				{
					if (!is_drop(m.move) && type_of(pos.piece_on(from_sq(m.move))) == KING)
						moves.push_back(m);
				}

				if (moves.size())
				{
					// 玉を移動させる指し手があったので、このなかから指し手を採用する。
					move = moves.at(prng.rand(moves.size()));
				}
			}
#endif

			// 玉を移動する指し手ではなかったので全合法手のなかから指し手を選択する。
			if (move == MOVE_NONE)
				move = legal_moves.at(prng.rand(legal_moves.size())).move;

			do_move(pos, move, states);

			// 詰みの局面、1手詰めの局面を除外
			if (pos.is_mated()
				|| (!pos.checkers() && pos.mate1ply() != MOVE_NONE)
				|| pos.DeclarationWin() != MOVE_NONE
				)
				goto Retry;
		}

		// 局面の生成に成功したのでこれにて終了。
	}

	//  thread_id    = 0..Threads.size()-1
	void MultiThinkGenSfen2019::thread_worker(size_t thread_id)
	{
		// とりあえず、書き出す手数の最大のところで引き分け扱いになるものとする。
		const int MAX_PLY2 = write_maxply;

		// StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
		// leaf nodeに行くのであれば、search_depth分ぐらいは必要。
		std::vector<StateInfo> states_(MAX_PLY2 + MAX_PLY /* == search_depth + α */);
		StateInfo* const states = &states_[0];

		// Positionに対して従属スレッドの設定が必要。
		// 並列化するときは、Threads (これが実体が vector<Thread*>なので、
		// Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
		auto& th = *Threads[thread_id];

		auto& pos = th.rootPos;

		// 終了フラグ
		bool quit = false;

		Move move;

		// 1局分の局面を保存しておき、終局のときに勝敗を含めて書き出す。
		PSVector a_psv;
		a_psv.reserve(MAX_PLY2 + MAX_PLY);

		// 対局シミュレーションのループ
		// 規定回数の局面を書き出すまで繰り返し
		while (!quit)
		{
			// -- 1局分スタート

			// 自分スレッド用の置換表があるはずなので自分の置換表だけをクリアする。
			th->tt.clear();

			// 局面の初期化
			set_start_pos(pos, th , states);

			// 局面バッファのクリア
			a_psv.clear();

			Value lastValue = VALUE_NONE;

			/* 本局の探索ノード数。平均5%のゆらぎ。これで指し手をある程度ばらつかせる。
				本局を通じたNodes数なので、シミュレーションの精度への影響はない。
				あまり大きくすると勝敗項に対するノイズになりかねないので自重して10%に留める。
			*/
			// u64 nodes = nodes_limit + (nodes_limit * prng.rand(100) / 1000);

			// →　ノイズになるのでノードは固定しておき、置換表をスレッド間で共有することにより揺らぎをもたせる。
			u64 nodes = nodes_limit;

			// 対局シミュレーションのループ
			while (pos.game_ply() < MAX_PLY2
				&& !pos.is_mated() && pos.DeclarationWin() == MOVE_NONE
				&& pos.is_repetition() != REPETITION_DRAW /* 千日手 */)
			{
				// -- 普通に探索してその指し手で局面を進める。

				// NodesLimitで制限しているのでdepthは24ぐらいで問題ない。
				// しかし、ここをあまり大きくすると詰み周りの局面で延長がかかって、探索が終わらなくなる。(´ω｀)
				auto pv_value = search(pos, search_depth , /*multi_pv*/1 , nodes );

				lastValue = pv_value.first;
				auto& pv = pv_value.second;

				// eval_limitの値を超えていれば勝ち(or 負け)として扱うのでここで対局シミュレーションを終了。
				if (abs(lastValue) > eval_limit)
					break;

				// --- 局面の一時保存
					
				// 初期局面周辺は類似局面ばかりなので学習に用いると過学習になりかねない。

				if (write_minply <= pos.game_ply())
				{
					a_psv.emplace_back(PackedSfenValue());
					auto &psv = a_psv.back();

					// packを要求されているならpackされたsfenとそのときの評価値を書き出す。
					// 最終的な書き出しは、勝敗がついてから。
					pos.sfen_pack(psv.sfen);

					// PV leafのevaluate()の値とどちらが良いかはよくわからない。
					// PV leafの値だと詰みかけの局面で駒を捨ててて自分不利に見えるのが少し嫌。
					psv.score = (s16)lastValue;
					psv.gamePly = (u16)pos.game_ply();

					// この局面の手番を仮で入れる。この値はファイルに書き出すまでに書き換える。
					psv.game_result = (s8)pos.side_to_move();
					
					// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
					psv.move = (u16)pv[0];
				}

				// search_depth手読みの指し手で局面を進める。
				// is_mated()ではないので、pv[0]として合法手が存在するはずなのだが..
				move = pv[0];
				do_move(pos,move,states);

			} // 対局シミュレーション終わり
			
			// lastValue == VALUE_NONEの場合は一度も探索していないということであり、
			// 書き出す局面がないはずであるから、以下の処理で問題ない。
			// ただ、その状態でこのwhileループに突入しているのがおかしくて…。
			ASSERT_LV3(lastValue != VALUE_NONE);

			// 勝利した側
			Color win;
			//RepetitionState repetition_state = pos.is_repetition(20);

			if (pos.is_mated()) {
				// 負け
				// 詰まされた
				win = ~pos.side_to_move();
			}
			else if (pos.DeclarationWin() != MOVE_NONE) {
				// 勝ち
				// 入玉勝利
				win = pos.side_to_move();
			}
			else if (lastValue > eval_limit) {
				// 勝ち
				win = pos.side_to_move();
			}
			else if (lastValue < -eval_limit) {
				// 負け
				win = ~pos.side_to_move();
			}
			else {
				// それ以外は引き分け等なので書き出さない
				// 千日手も同様。
				continue;
			}

			// 各局面に関して、対局の勝敗の情報を付与しておく。
			// a_psvに保存されている局面は(手番的に)連続しているものとする。
			// 終局の局面(現在の局面)は書き出されていないことに注意すべき。
			for (auto& psv : a_psv)
			{
				// 局面を書き出そうと思ったら規定回数に達していた。
				// get_next_loop_count()内でカウンターを加算するので
				// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
				auto loop_count = get_next_loop_count();
				if (loop_count == UINT64_MAX)
				{
					// 終了フラグを立てておく。
					quit = true;
					break;
				}

				// この局面の手番側が仮でgame_resultに入っている。
				// 最後の局面の手番側の勝利であれば1 , 負けであれば -1 を入れる。
				auto stm = (Color)psv.game_result;
				psv.game_result = (stm == win) ? 1 : -1;

				//cout << (int)psv.game_result << endl;

				// 局面を一つ書き出す。
				sw.write(thread_id, psv);
			}

		} // while(!quit)

		sw.finalize(thread_id);
	}

	// gensfen2019コマンド本体
	void gen_sfen2019(Position& pos, istringstream& is)
	{
		// スレッド数(これは、USIのsetoptionで与えられる)
		u32 thread_num = (u32)Options["Threads"];
		
		// 生成棋譜の個数 default = 80億局面(Ponanza仕様)
		u64 loop_max = 8000000000UL;

		// 評価値がこの値を超えたら生成を打ち切る。
		// デフォルトのこの値だと超えることはないので、評価値での打ち切りは無し。
		int eval_limit = 32000;

		// 探索深さ
		// NodesLimitで制限するが王手延長で延長されると探索終わらないので何らかの上限が必要。
		int search_depth = 24;
		
		// 探索ノード数
		u64 nodes_limit = 10000;

		// 書き出す局面のply(初期局面からの手数)の最小、最大。
		// 重複局面を除去するので初手から書き出して良いと思う。
		// ここの手数、あまり大きくすると入玉局面ばかりになり、引き分けになる確率が高いので無駄なシミュレーションになる。
		// ※　tanuki-(WCSC28)ではwrite_maxply == 400
		int write_minply = 1;
		int write_maxply = 300;

		// 使用する定跡ファイル。
		// この定跡ファイルの各局面から1局面を選んでランダムムーブで1手進めてから対局シミュレーションを開始する。
		string book_file_name = "book/flood2018.sfen";

		// 教師局面を書き出すファイル名
		string output_file_name = "generated_kifu.bin";

		string token;

		// eval hashにhitすると初期局面付近の評価値として、hash衝突して大きな値を書き込まれてしまうと
		// eval_limitが小さく設定されているときに初期局面で毎回eval_limitを超えてしまい局面の生成が進まなくなる。
		// そのため、eval hashは無効化する必要がある。
		// あとeval hashのhash衝突したときに、変な値の評価値が使われ、それを教師に使うのが気分が悪いというのもある。
		bool use_eval_hash = false;

		// この単位でファイルに保存する。
		// ファイル名は file_1.bin , file_2.binのように連番がつく。
		u64 save_every = UINT64_MAX;

		// ファイル名の末尾にランダムな数値を付与する。
		bool random_file_name = false;

		while (true)
		{
			token = "";
			is >> token;
			if (token == "")
				break;

			if (token == "loop")
				is >> loop_max;
			else if (token == "output_file_name")
				is >> output_file_name;
			else if (token == "eval_limit")
				is >> eval_limit;
			else if (token == "search_depth")
				is >> search_depth;
			else if (token == "write_minply")
				is >> write_minply;
			else if (token == "write_maxply")
				is >> write_maxply;
			else if (token == "nodes_limit")
				is >> nodes_limit;
			else if (token == "use_eval_hash")
				is >> use_eval_hash;
			else if (token == "save_every")
				is >> save_every;
			else if (token == "random_file_name")
				is >> random_file_name;
			else if (token == "book_file_name")
				is >> book_file_name;
			else
				cout << "Error! : Illegal token " << token << endl;
		}

#if defined(USE_GLOBAL_OPTIONS)
		// あとで復元するために保存しておく。
		auto oldGlobalOptions = GlobalOptions;
		GlobalOptions.use_eval_hash = use_eval_hash;
#endif

		if (random_file_name)
		{
			// output_file_nameにこの時点でランダムな数値を付与してしまう。
			PRNG r;
			// 念のため乱数振り直しておく。
			for (int i = 0; i<10; ++i)
				r.rand(1);
			auto to_hex = [](u64 u) {
				std::stringstream ss;
				ss << std::hex << u;
				return ss.str();
			};
			// 64bitの数値で偶然かぶると嫌なので念のため64bitの数値２つくっつけておく。
			output_file_name = output_file_name + "_" + to_hex(r.rand<u64>()) + to_hex(r.rand<u64>());
		}

		std::cout << "gensfen2019 : " << endl
			<< "  search_depth = " << search_depth << endl
			<< "  nodes_limit = " << nodes_limit << endl
			<< "  loop_max = " << loop_max << endl
			<< "  eval_limit = " << eval_limit << endl
			<< "  thread_num (set by USI setoption) = " << thread_num << endl
			<< "  write_minply            = " << write_minply << endl
			<< "  write_maxply            = " << write_maxply << endl
			<< "  output_file_name        = " << output_file_name << endl
			<< "  use_eval_hash           = " << use_eval_hash << endl
			<< "  save_every              = " << save_every << endl
			<< "  random_file_name        = " << random_file_name << endl
			<< "  book_file_name          = " << book_file_name << endl
			;

		// Options["Threads"]の数だけスレッドを作って実行。
		{
			SfenWriter sw(output_file_name, thread_num);
			sw.save_every = save_every;

			MultiThinkGenSfen2019 multi_think( sw , search_depth , nodes_limit , book_file_name);
			multi_think.set_loop_max(loop_max);
			multi_think.eval_limit = eval_limit;
			multi_think.write_minply = write_minply;
			multi_think.write_maxply = write_maxply;
			multi_think.start_file_write_worker();
			multi_think.go_think();

			// SfenWriterのデストラクタでjoinするので、joinが終わってから終了したというメッセージを
			// 表示させるべきなのでここをブロックで囲む。
		}

		std::cout << "gensfen2019 finished." << endl;

#if defined(USE_GLOBAL_OPTIONS)
		// GlobalOptionsの復元。
		GlobalOptions = oldGlobalOptions;
#endif

	}
}


#endif // defined(EVAL_LEARN) && defined(GENSFEN2019)

