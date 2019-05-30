//
// 開発中の教師局面の自動生成
//

// 現在の評価関数で、なるべく実現確率の高い現実的な局面でかつ、形勢が互角に近い局面を教師生成の開始局面として用いる。


#include "../shogi.h"

#if defined(EVAL_LEARN) && defined(USE_GENSFEN2018)

#include <sstream>
using namespace std;

namespace Learner {

	// -----------------------------------
	//  棋譜を生成するworker(スレッドごと)
	// -----------------------------------
//	const static int GENSFEN_MULTI_PV = 24; // 1)
	const static int GENSFEN_MULTI_PV = 8;  // 2)

	// 訪問済みnodeに関する情報を記録しておく構造体。
	// このnodeで、MultiPVで探索したときのよさげな指し手上位8手を記録しておく。
	struct NodeInfo
	{
		u16 moves[GENSFEN_MULTI_PV];	// 指し手(最大8手)
		u32 length;						// 指し手が何手あるか。
		s32 game_ply;					// game plyが異なるなら循環してるのかも知れないので適用しない。
		u64 key;						// 局面のhash key
		// 1) = 48+4+4+8 = 64bytes
		// 2) = 16+4+4+8 = 32bytes
	};

	// 複数スレッドでsfenを生成するためのクラス
	struct MultiThinkGenSfen2018 : public MultiThink
	{
		// hash_size : NodeInfoを格納するためのhash sizeを指定する。単位は[MB]。
		// メモリに余裕があるなら大きめの値を指定するのが好ましい。
		MultiThinkGenSfen2018(int search_depth_, int search_depth2_, SfenWriter& sw_ , u64 hash_size)
			: search_depth(search_depth_), search_depth2(search_depth2_), sw(sw_)
		{
			hash.resize(GENSFEN_HASH_SIZE);

			node_info_size = (hash_size * (u64)1024 * (u64)1024) / (u64)sizeof(NodeInfo);
			node_info_hash.resize(node_info_size);

			// PCを並列化してgensfenするときに同じ乱数seedを引いていないか確認用の出力。
			std::cout << endl << prng << std::endl;
		}

		virtual void thread_worker(size_t thread_id);
		void start_file_write_worker() { sw.start_file_write_worker(); }

		//  search_depth = 通常探索の探索深さ
		int search_depth;
		int search_depth2;

		// 生成する局面の評価値の上限
		int eval_limit;

		// 書き出す局面のply(初期局面からの手数)の最小、最大。
		int write_minply;
		int write_maxply;

		// sfenの書き出し器
		SfenWriter& sw;

		// 同一局面の書き出しを制限するためのhash
		// hash_indexを求めるためのmaskに使うので、2**Nでなければならない。
		vector<NodeInfo> node_info_hash;
		u64 node_info_size;

		// node_info_hashにnode情報を書き出した回数(ざっくり)
		atomic<u64> node_info_count;

		// 同一局面の書き出しを制限するためのhash
		// hash_indexを求めるためのmaskに使うので、2**Nでなければならない。
		static const u64 GENSFEN_HASH_SIZE = 64 * 1024 * 1024;

		vector<Key> hash; // 64MB*sizeof(HASH_KEY) = 512MB
	};

	//  thread_id    = 0..Threads.size()-1
	void MultiThinkGenSfen2018::thread_worker(size_t thread_id)
	{
		// とりあえず、書き出す手数の最大のところで引き分け扱いになるものとする。
		const int MAX_PLY2 = write_maxply;

		// StateInfoを最大手数分 + SearchのPVでleafにまで進めるbuffer
		std::vector<StateInfo, AlignedAllocator<StateInfo>> states(MAX_PLY2 + 50 /* == search_depth + α */);
		StateInfo si;

		// 今回の指し手。この指し手で局面を進める。
		Move m = MOVE_NONE;

		// 終了フラグ
		bool quit = false;

		// 規定回数回になるまで繰り返し
		while (!quit)
		{
			// Positionに対して従属スレッドの設定が必要。
			// 並列化するときは、Threads (これが実体が vector<Thread*>なので、
			// Threads[0]...Threads[thread_num-1]までに対して同じようにすれば良い。
			auto th = Threads[thread_id];

			auto& pos = th->rootPos;
			pos.set_hirate(&si, th);

			// 1局分の局面を保存しておき、終局のときに勝敗を含めて書き出す。
			// 書き出す関数は、この下にあるflush_psv()である。
			PSVector a_psv;
			a_psv.reserve(MAX_PLY2 + 50);

			// a_psvに積まれている局面をファイルに書き出す。
			// lastTurnIsWin : a_psvに積まれている最終局面の次の局面での勝敗
			// 勝ちのときは1。負けのときは-1。引き分けのときは0を渡す。
			// 返し値 : もう規定局面数に達したので終了する場合にtrue。
			auto flush_psv = [&](s8 lastTurnIsWin)
			{
				s8 isWin = lastTurnIsWin;

				// 終局の局面(の一つ前)から初手に向けて、各局面に関して、対局の勝敗の情報を付与しておく。
				// a_psvに保存されている局面は(手番的に)連続しているものとする。
				for (auto it = a_psv.rbegin(); it != a_psv.rend(); ++it)
				{
					// isWin == 0(引き分け)なら -1を掛けても 0(引き分け)のまま
					isWin = -isWin;
					it->game_result = isWin;

					// 局面を書き出そうと思ったら規定回数に達していた。
					// get_next_loop_count()内でカウンターを加算するので
					// 局面を出力したときにこれを呼び出さないとカウンターが狂う。
					auto loop_count = get_next_loop_count();
					if (loop_count == UINT64_MAX)
					{
						// 終了フラグを立てておく。
						quit = true;
						return;
					}

					// 2の累乗であるならnode_info_countの数を出力。
					if (POPCNT64(loop_count) == 1)
					{
						sync_cout << endl << "loop_count = " << loop_count << " , node_info_count = " << node_info_count << sync_endl;

#if 0
						// 1～32手目までのnodeがどれくらい格納されているか出力する。
						for (int i = 1; i <= 32; ++i)
						{
							u64 c = 0;
							for (u64 j = 0; j < node_info_size; ++j)
								if (node_info_hash[j].game_ply == i)
									c++;
							if (c)
								sync_cout << "PLY = " << i << " , nodes = " << c << sync_endl;
						}
#endif
					}

					// 局面を一つ書き出す。
					sw.write(thread_id, *it);

#if 0
					pos.set_from_packed_sfen(it->sfen);
					cout << pos << "Win : " << it->isWin << " , " << it->score << endl;
#endif
				}
			};

			// node_info_hashを調べるのかのフラグ
			// 初期局面から、node_info_hashにhitし続ける限りは調べる。
			bool opening = true;

			// ply : 初期局面からの手数
			for (int ply = 0; ply < MAX_PLY2; ++ply)
			{
				//cout << pos << endl;

				// 今回の探索depth
				// gotoで飛ぶので先に宣言しておく。
				int depth = search_depth + (int)prng.rand(search_depth2 - search_depth + 1);

				// 全駒されて詰んでいたりしないか？
				if (pos.is_mated())
				{
					// (この局面の一つ前の局面までは書き出す)
					flush_psv(-1);
					break;
				}

				// 宣言勝ち
				if (pos.DeclarationWin() != MOVE_NONE)
				{
					// (この局面の一つ前の局面までは書き出す)
					flush_psv(1);
					break;
				}

				// 訪問済みnodeであるなら、そのなかからランダムに一つ指し手を選び、進める。
				if (opening)
				{
					auto key = pos.key();
					auto hash_index = (size_t)(key & (node_info_size - 1));
					auto& entry = node_info_hash[hash_index];

					// このnodeと局面のhash keyがぴったり一致したなら..
					if (key == entry.key)
					{
						// 局面、おそらく循環しちゃってるので終了。
						if (pos.game_ply() > entry.game_ply)
							break;

						const int length = entry.length;
						if (length)
						{
							// 前回MultiPVで探索したときの指し手のうち、よさげなものをランダムに指し手を一つ選択
							u16 move16 = entry.moves[prng.rand(length)];
							if (move16)
							{
								m = pos.move16_to_move((Move)move16);
								if (pos.pseudo_legal(m) && pos.legal(m))
								{
									// 指し手はmに入っているので、これでdo_move()してしまう。
									goto DO_MOVE;
								}
							}
						}
					}

					// どうもこの局面用の情報が書き込まれていないようなので、MultiPVで探索してこのNodeの情報を書き込んでしまう。

					// 32手目以降であるなら、次のnodeからは普通に対局シミュレーションを行なう。
					// node_info_hashを調べる必要がない。
					if (ply >= 32)
						opening = false;

					Learner::search(pos, depth, GENSFEN_MULTI_PV);
					// rootMovesの上位N手のなかから一つ選択

					auto& rm = pos.this_thread()->rootMoves;

					u64 length = min((u64)rm.size(), (u64)GENSFEN_MULTI_PV);
					int count = 0;
					u16 moves[GENSFEN_MULTI_PV];
					for (u64 i = 0; i < length /* && count < GENSFEN_MULTI_PV */ ; ++i)
					{
						auto value = rm[i].score;

						// 互角から程遠くなる指し手は除外。(bestmoveがこれだとしたら、そもそもこのnodeの直前のnodeがおかしいのだが…)
						if (value < -400 || 400 < value)
							continue;

						moves[count++] = (u16)rm[i].pv[0];
					}
					if (count)
					{
						for (int i = 0; i < count; ++i)
							entry.moves[i] = moves[i];
						entry.game_ply = pos.game_ply();
						entry.length = count;
						entry.key = key;
						node_info_count.fetch_add(1, std::memory_order_relaxed);

						// ランダムに1手選んでそれで進める
						m = pos.move16_to_move((Move)moves[prng.rand(count)]);
						ASSERT_LV3(pos.pseudo_legal(m) && pos.legal(m));

						// 指し手はmに入っているので、これでdo_move()してしまう。
						goto DO_MOVE;
					}

					// 互角の局面ではないので序盤は抜けたという扱い。
					opening = false;
				}

				// -- 普通に探索してその指し手で局面を進める。

				{
					// 置換表の世代カウンターを進めておかないと
					// 初期局面周辺でhash衝突したTTEntryに当たり、変な評価値を拾ってきて、
					// eval_limitが低いとそれをもって終了してしまうので、いつまでも教師局面が生成されなくなる。
					// 置換表自体は、スレッドごとに保持しているので、ここでTT.new_search()を呼び出して問題ない。

					// 評価値の絶対値がこの値以上の局面については
					// その局面を学習に使うのはあまり意味がないのでこの試合を終了する。
					// これをもって勝敗がついたという扱いをする。
					TT.new_search();

					auto pv_value1 = search(pos, depth);

					auto value1 = pv_value1.first;
					auto& pv1 = pv_value1.second;

					// 1手詰め、宣言勝ちならば、ここでmate_in(2)が返るのでeval_limitの上限値と同じ値になり、
					// このif式は必ず真になる。resignについても同様。

					if (abs(value1) >= eval_limit)
					{
						//					sync_cout << pos << "eval limit = " << eval_limit << " over , move = " << pv1[0] << sync_endl;

						// この局面でvalue1 >= eval_limitならば、(この局面の手番側の)勝ちである。
						flush_psv((value1 >= eval_limit) ? 1 : -1);
						break;
					}

					// おかしな指し手の検証
					if (pv1.size() > 0
						&& (pv1[0] == MOVE_RESIGN || pv1[0] == MOVE_WIN || pv1[0] == MOVE_NONE)
						)
					{
						// MOVE_WINは、この手前で宣言勝ちの局面であるかチェックしているので
						// ここで宣言勝ちの指し手が返ってくることはないはず。
						// また、MOVE_RESIGNのときvalue1は1手詰めのスコアであり、eval_limitの最小値(-31998)のはずなのだが…。
						cout << "Error! : " << pos.sfen() << m << value1 << endl;
						break;
					}

					// 各千日手に応じた処理。

					s8 is_win = 0;
					bool game_end = false;
					auto draw_type = pos.is_repetition(0);
					switch (draw_type)
					{
					case REPETITION_WIN: is_win = 1; game_end = true; break;
					case REPETITION_DRAW: is_win = 0; game_end = true; break;
					case REPETITION_LOSE: is_win = -1; game_end = true; break;

						// case REPETITION_SUPERIOR: break;
						// case REPETITION_INFERIOR: break;
						// これらは意味があるので無視して良い。
					default: break;
					}

					if (game_end)
					{
						break;
					}

					// PVの指し手でleaf nodeまで進めて、そのleaf nodeでevaluate()を呼び出した値を用いる。
					auto evaluate_leaf = [&](Position& pos, vector<Move>& pv)
					{
						auto rootColor = pos.side_to_move();

						int ply2 = ply;
						for (auto m : pv)
						{
							// デバッグ用の検証として、途中に非合法手が存在しないことを確認する。
							// NULL_MOVEはこないものとする。

							// 十分にテストしたのでコメントアウトで良い。
#if 1
						// 非合法手はやってこないはずなのだが。
						// 宣言勝ちとmated()でないことは上でテストしているので
						// 読み筋としてMOVE_WINとMOVE_RESIGNが来ないことは保証されている。(はずだが…)
							if (!pos.pseudo_legal(m) || !pos.legal(m))
							{
								cout << "Error! : " << pos.sfen() << m << endl;
							}
#endif
							pos.do_move(m, states[ply2++]);

							// 毎ノードevaluate()を呼び出さないと、evaluate()の差分計算が出来ないので注意！
							// depthが8以上だとこの差分計算はしないほうが速いと思われる。
							if (depth < 8)
								Eval::evaluate_with_no_return(pos);
						}

						// leafに到達
						//      cout << pos;

						auto v = Eval::evaluate(pos);
						// evaluate()は手番側の評価値を返すので、
						// root_colorと違う手番なら、vを反転させて返さないといけない。
						if (rootColor != pos.side_to_move())
							v = -v;

						// 巻き戻す。
						// C++x14にもなって、いまだreverseで回すforeachすらないのか…。
						//  for (auto it : boost::adaptors::reverse(pv))

						for (auto it = pv.rbegin(); it != pv.rend(); ++it)
							pos.undo_move(*it);

						return v;
					};

					// depth 0の場合、pvが得られていないのでdepth 2で探索しなおす。
					if (search_depth <= 0)
					{
						pv_value1 = search(pos, 2);
						pv1 = pv_value1.second;
					}

					// 初期局面周辺はは類似局面ばかりなので
					// 学習に用いると過学習になりかねないから書き出さない。
					// →　比較実験すべき
					if (ply < write_minply - 1)
					{
						a_psv.clear();
						goto SKIP_SAVE;
					}

					// 同一局面を書き出したところか？
					// これ、複数のPCで並列して生成していると同じ局面が含まれることがあるので
					// 読み込みのときにも同様の処理をしたほうが良い。
					{
						auto key = pos.key();
						auto hash_index = (size_t)(key & (GENSFEN_HASH_SIZE - 1));
						auto key2 = hash[hash_index];
						if (key == key2)
						{
							// スキップするときはこれ以前に関する
							// 勝敗の情報がおかしくなるので保存している局面をクリアする。
							// どのみち、hashが合致した時点でそこ以前の局面も合致している可能性が高いから
							// 書き出す価値がない。
							a_psv.clear();
							goto SKIP_SAVE;
						}
						hash[hash_index] = key; // 今回のkeyに入れ替えておく。
					}

					// 局面の一時保存。
					{
						a_psv.emplace_back(PackedSfenValue());
						auto &psv = a_psv.back();

						// packを要求されているならpackされたsfenとそのときの評価値を書き出す。
						// 最終的な書き出しは、勝敗がついてから。
						pos.sfen_pack(psv.sfen);

						// PV lineのleaf nodeでのroot colorから見たevaluate()の値を取得。
						// search()の返し値をそのまま使うのとこうするのとの善悪は良くわからない。
						psv.score = evaluate_leaf(pos, pv1);
						psv.gamePly = ply;

						// PVの初手を取り出す。これはdepth 0でない限りは存在するはず。
						ASSERT_LV3(pv_value1.second.size() >= 1);
						Move pv_move1 = pv_value1.second[0];
						psv.move = pv_move1;
					}

				SKIP_SAVE:;

					// 何故かPVが得られなかった(置換表などにhitして詰んでいた？)ので次の対局に行く。
					// かなりのレアケースなので無視して良いと思う。
					if (pv1.size() == 0)
						break;

					// search_depth手読みの指し手で局面を進める。
					m = pv1[0];
				}

			DO_MOVE:;
				pos.do_move(m, states[ply]);

				// 差分計算を行なうために毎node evaluate()を呼び出しておく。
				Eval::evaluate_with_no_return(pos);

			} // for (int ply = 0; ply < MAX_PLY2 ; ++ply)

		} // while(!quit)

		sw.finalize(thread_id);
	}


	// gensfen2018コマンド本体
	void gen_sfen2018(Position& pos, istringstream& is)
	{
		// スレッド数(これは、USIのsetoptionで与えられる)
		u32 thread_num = (u32)Options["Threads"];

		// 生成棋譜の個数 default = 80億局面(Ponanza仕様)
		u64 loop_max = 8000000000UL;

		// 評価値がこの値になったら生成を打ち切る。
		int eval_limit = 3000;

		// 探索深さ
		int search_depth = 3;
		int search_depth2 = INT_MIN;

		// 書き出す局面のply(初期局面からの手数)の最小、最大。
		int write_minply = 16;
		int write_maxply = 400;

		// 書き出すファイル名
		string output_file_name = "generated_kifu.bin";

		// NodeInfoのためのhash size
		// 単位は[MB] 2の累乗であること。
		u64 node_info_hash_size = 2 * 1024; // 2GB

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

			if (token == "depth")
				is >> search_depth;
			else if (token == "depth2")
				is >> search_depth2;
			else if (token == "loop")
				is >> loop_max;
			else if (token == "output_file_name")
				is >> output_file_name;
			else if (token == "eval_limit")
			{
				is >> eval_limit;
				// 最大値を1手詰みのスコアに制限する。(そうしないとループを終了しない可能性があるので)
				eval_limit = std::min(eval_limit, (int)mate_in(2));
			}
			else if (token == "write_minply")
				is >> write_minply;
			else if (token == "write_maxply")
				is >> write_maxply;
			else if (token == "use_eval_hash")
				is >> use_eval_hash;
			else if (token == "save_every")
				is >> save_every;
			else if (token == "random_file_name")
				is >> random_file_name;
			else if (token == "node_info_hash_size")
				is >> node_info_hash_size;
			else
				cout << "Error! : Illegal token " << token << endl;
		}

#if defined(USE_GLOBAL_OPTIONS)
		// あとで復元するために保存しておく。
		auto oldGlobalOptions = GlobalOptions;
		GlobalOptions.use_eval_hash = use_eval_hash;
#endif

		// search depth2が設定されていないなら、search depthと同じにしておく。
		if (search_depth2 == INT_MIN)
			search_depth2 = search_depth;

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

		std::cout << "gensfen2018 : " << endl
			<< "  search_depth = " << search_depth << " to " << search_depth2 << endl
			<< "  loop_max = " << loop_max << endl
			<< "  eval_limit = " << eval_limit << endl
			<< "  thread_num (set by USI setoption) = " << thread_num << endl
			<< "  book_moves (set by USI setoption) = " << Options["BookMoves"] << endl
			<< "  write_minply            = " << write_minply << endl
			<< "  write_maxply            = " << write_maxply << endl
			<< "  output_file_name        = " << output_file_name << endl
			<< "  use_eval_hash           = " << use_eval_hash << endl
			<< "  save_every              = " << save_every << endl
			<< "  random_file_name        = " << random_file_name << endl
			<< "  node_info_hash_size[MB] = " << node_info_hash_size;

		// Options["Threads"]の数だけスレッドを作って実行。
		{
			SfenWriter sw(output_file_name, thread_num);
			sw.save_every = save_every;

			MultiThinkGenSfen2018 multi_think(search_depth, search_depth2, sw , node_info_hash_size);
			multi_think.set_loop_max(loop_max);
			multi_think.eval_limit = eval_limit;
			multi_think.write_minply = write_minply;
			multi_think.write_maxply = write_maxply;
			multi_think.start_file_write_worker();
			multi_think.go_think();

			// SfenWriterのデストラクタでjoinするので、joinが終わってから終了したというメッセージを
			// 表示させるべきなのでここをブロックで囲む。
		}

		std::cout << "gensfen2018 finished." << endl;

#if defined(USE_GLOBAL_OPTIONS)
		// GlobalOptionsの復元。
		GlobalOptions = oldGlobalOptions;
#endif

	}
}


#endif // defined(EVAL_LEARN) && defined(USE_GENSFEN2018)

