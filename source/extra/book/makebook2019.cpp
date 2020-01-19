#include "../../types.h"

#if defined (ENABLE_MAKEBOOK_CMD)

#include "book.h"
#include "../../position.h"
#include "../../thread.h"
#include <fstream>
#include <sstream>
#include <unordered_set>

using namespace std;
using namespace Book;

namespace Book { extern void makebook_cmd(Position& pos, istringstream& is); }

namespace {

	// ----------------------------
	// テラショック定跡の生成
	// ----------------------------

	// cf.
	// テラショック定跡の生成手法
	// http://yaneuraou.yaneu.com/2019/04/19/%E3%83%86%E3%83%A9%E3%82%B7%E3%83%A7%E3%83%83%E3%82%AF%E5%AE%9A%E8%B7%A1%E3%81%AE%E7%94%9F%E6%88%90%E6%89%8B%E6%B3%95/

	// root局面の集合。駒落ちの各局面も含めてroot局面の集合。この定義はMyShogiプロジェクトのほうから持ってきた。
	// 定跡を掘り進むときに枝刈りする評価値が駒落ちの度合いで異なるので同じ枠組みでうまく扱うのは結構難しい気も。

	std::vector<std::string> start_sfens = {
	/*public static readonly string HIRATE = */       "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" ,
	/*public static readonly string HANDICAP_KYO = */ "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1" ,
	/*public static readonly string HANDICAP_RIGHT_KYO = */ "1nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_KAKU = */ "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_HISYA = */ "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_HISYA_KYO = */ "lnsgkgsn1/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_2 =      */ "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_3 =      */ "lnsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_4 =      */ "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_5 =      */ "2sgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_LEFT_5 = */ "1nsgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_6 =      */ "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_8 =      */ "3gkg3/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_10 =     */ "4k4/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
	/*public static readonly string HANDICAP_PAWN3 =  */ "4k4/9/9/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w 3p 1",
	};

	// build_tree_nega_max()で用いる返し値に用いる。
	// 候補手の評価値、指し手、leaf nodeまでの手数
	struct VMD
	{
		VMD() : value(-VALUE_INFINITE), move(MOVE_NONE), depth(DEPTH_ZERO) {}
		VMD(Value value_, Move move_, Depth depth_) : value(value_), move(move_), depth(depth_) {}

		Value value; // 評価値
		Move move;   // 候補手
		Depth depth; // これはleaf nodeまでの手数
	};

	// build_tree_nega_max()で用いる返し値に用いる。
	// root_colorがBLACK用とwhite用とで個別にVMDを格納している。
	// root_colorというのはNegaMaxするときの現在のnodeのcolorだと考えて問題ない。
	struct VMD_Pair
	{
		// 何も初期化しないが、VMDクラス側の規定のコンストラクタで初期化はされている。
		VMD_Pair() {}

		// blackとwhiteとを同じ値で初期化する。
		VMD_Pair(Value v, Move m, Depth d) : black(v, m, d), white(v, m, d) {}

		// black,whiteをそれぞれの値で初期化する。
		VMD_Pair(Value black_v, Move black_m, Depth black_d, Value white_v, Move white_m, Depth white_d) :
			black(black_v, black_m, black_d), white(white_v, white_m, white_d) {}
		VMD_Pair(VMD black_, VMD white_) : black(black_), white(white_) {}
		VMD_Pair(VMD best[COLOR_NB]) : black(best[BLACK]), white(best[WHITE]) {}

		VMD black; // root_color == BLACK用の評価値
		VMD white; // root_color == WHITE用の評価値
	};

	// 一度調べたnodeをcacheしておくためのもの。
	struct VmdPairGamePly {
		VmdPairGamePly(){}
		VmdPairGamePly(const VMD_Pair& vmd_pair_, int gamePly_) : vmd_pair(vmd_pair_), gamePly(gamePly_) {}
		VMD_Pair vmd_pair;
		int gamePly;
	};

	// 定跡のbuilder
	struct BookTreeBuilder
	{
		// 定跡game treeを生成する機能
		void build_tree(Position& pos, istringstream& is);

		// 定跡ファイルを読み込んで、指定局面から深掘りするために必要な棋譜を生成する。
		void extend_tree(Position& pos, istringstream& is);

		// 定跡の無限自動掘り
		void endless_extend_tree(Position& pos, istringstream& is);

	private:
		// 再帰的に最善手を調べる。
		VMD_Pair build_tree_nega_max(Position& pos, MemoryBook& read_book, MemoryBook& write_book);

		// "position ..."の"..."の部分を解釈する。
		int feed_position_string(Position& pos, const string& line, StateInfo* states, Thread* th);

		//  定跡ファイルの特定局面から定跡を掘る
		void extend_tree_sub(Position& pos, MemoryBook& read_book, fstream& fs, const string& sfen , bool bookhit);

		// 進捗の表示
		void output_progress();

		std::unordered_map<std::string /*sfen*/, VmdPairGamePly> vmd_write_cache;

		// 書き出したnodeをcacheしておく。
		std::unordered_set<std::string /*sfen*/> done_sfen;

		// do_moveした指し手を記録しておく。
		std::vector<Move> lastMoves;

		// Position::do_move(),undo_move()のwrapper
		void do_move(Position& pos, Move m, StateInfo& si) { lastMoves.push_back(m);  pos.do_move(m, si); }
		void undo_move(Position& pos,Move m) { lastMoves.pop_back(); pos.undo_move(m); }

		// 処理したnode数/書き出したnode数
		u64 total_node = 0;
		u64 total_write_node = 0;

		// build_tree_nega_max()で用いる先手/後手のcontempt。
		int black_contempt, white_contempt;

		// extend_tree_sub()で用いる先手/後手のevalの下限
		int black_eval_limit, white_eval_limit;

		// 延長するleafの値の範囲(enable_extend_range == trueのときだけこの機能が有効化される)
		//   extend_range1 <= lastEval <= extend_range2
		// のleaf node(の候補手)だけが延長される。
		bool enable_extend_range;
		int extend_range1, extend_range2;

		// extend_tree_sub()の一つ前の時のevalの値
		int lastEval;

		// 引き分けになる手数
		int max_game_ply;
	};

	void BookTreeBuilder::output_progress()
	{
		if ((total_node % 1000) == 0)
		{
			cout << endl << total_node;
			if (total_write_node)
				cout << "|" << total_write_node;
		}
		if ((total_node % 10) == 0)
			cout << ".";
		++total_node;
	}

	// 再帰的に最善手を調べる。
	VMD_Pair BookTreeBuilder::build_tree_nega_max(Position& pos, MemoryBook& read_book, MemoryBook& write_book)
	{
		// -- 定跡にhitしないにせよ、詰みと宣言勝ち、千日手に関しては処理できるのでそれ相応の値を返す必要がある。

		// 現局面で詰んでいる
		if (pos.is_mated())
			return VMD_Pair(mated_in(0), MOVE_NONE, DEPTH_ZERO);

			// 現局面で宣言勝ちできる。
			// 定跡ファイルにMOVE_WINが紛れたときの解釈を規定していないのでここでは入れないことにする。
			if (pos.DeclarationWin() != MOVE_NONE)
			return VMD_Pair(mate_in(1), MOVE_NONE, DEPTH_ZERO);

		// この局面の手番
		auto stm = pos.side_to_move();

			// 千日手の検出などが必要でごじゃる。
		auto draw_type = pos.is_repetition(pos.game_ply());
			if (draw_type != REPETITION_NONE)
			{
				// この次の一手が欲しい気はする。is_repetition()が返して欲しい気はするのだが、
				// StateInfoが指し手を保存していなくて返せないのか…。(´ω｀)
				// これのためだけに"KEEP_LAST_MOVE"をdefineするのちょっと嫌だな…。自前で持つか…。

				// 千日手
				switch (draw_type)
				{
					// 千日手は-1にしてしまいたいが、先後で同じ定跡を用いるのでそれはできない(´ω｀)
					// ここ、きちんとやらないと後手だと必ず千日手狙いになってしまう…。
					// 解決策)
					// rootColorがBLACK,WHITEの時それぞれ用のVMDを返すべき。
					// value = rootColor == pos.side_to_move() ? -comtempt : +comtempt;
					// みたいな感じ。
					// ただ、後手番で千日手のcomtemptが30(千日手をevalの-30扱いにする)だとして、
					// 後手だからと言って積極的に千日手を狙われても…みたいな問題はある。
					// 定跡上は、先手のcontempt = 0 , 後手のcontempt = 70ぐらいがいいように思う。

					// PawnValue/100を掛けて正規化する処理はここではしないことにする。どんな探索部で生成された定跡かわからないので
					// book上のevalの値は正規化されているものと仮定する。
					// (makebook thinkコマンドだと正規化されないが…まあいいだろう..)

				case REPETITION_DRAW:
					// 千日手局面を実際に見つけて、その直後の指し手を取得する。
				{
					auto key = pos.key();
					int i = 0;
					auto* statePtr = pos.state();
					do {
						// 2手ずつ遡る
						statePtr = statePtr->previous->previous;
						i += 2;
					} while (key != statePtr->key());
					// i手前が同一局面であることがわかったので、その次の指し手を得る。

					// 例) 4手前の局面とkey()が同じなら4手前から循環して千日手が成立。すなわち、lastMovesの後ろから5つ目の指し手で千日手局面に突入しているので
					// その次の指し手(4手前の指し手)が、ここの次の一手のはず…。
					auto draw_move = lastMoves[lastMoves.size() - i];
					// この普通の千日手以外のケースでこれをやると非合法手になる可能性があって…。

					//  contempt * Eval::PawnValue / 100 という処理はしない。
					// 定跡DBはmakebook thinkコマンドで作成されていて、この正規化はすでになされている。

					// 現局面の手番を見て符号を決めないといけない。
				return VMD_Pair(
						(Value)(stm == BLACK ? -black_contempt : +black_contempt) /*先手のcomtempt */, draw_move, DEPTH_ZERO,
						(Value)(stm == WHITE ? -white_contempt : +white_contempt) /*後手のcomtempt */, draw_move, DEPTH_ZERO
					);
				}

			case REPETITION_INFERIOR: return VMD_Pair(-VALUE_SUPERIOR, MOVE_NONE, DEPTH_ZERO);
			case REPETITION_SUPERIOR: return VMD_Pair(VALUE_SUPERIOR, MOVE_NONE, DEPTH_ZERO);
			case REPETITION_WIN     : return VMD_Pair(mate_in(MAX_PLY), MOVE_NONE, DEPTH_ZERO);
			case REPETITION_LOSE    : return VMD_Pair(mated_in(MAX_PLY), MOVE_NONE, DEPTH_ZERO);

					// これ入れておかないとclangで警告が出る。
				case REPETITION_NONE:
				case REPETITION_NB:
					break;

				}
			}

		// -- すでに探索済みであるなら、そのときの値を返す。

		auto sfen = pos.sfen();

		// 手数違いの局面もread_bookの定跡を調べるので、千日手スコアはvmd_write_cacheにcacheしてはならない。
		// ※　次の局面で千日手になるパターンは仕方がない。
		// そこで千日手判定(上のコード)は、以下のコードより先に行う必要がある。
		// 経路違いである経路から来た時だけ千日手になるケースもある。このへん、DAGのあるゲーム木ではとても難しい問題に直面する。
		// 現実的に滅多に起こらないことならあまり深く考えても仕方ないだが、角換わりでわりと色んな問題が生じるようだ。

		// 手数を無視してvmd_write_cacheに保持させると、角換わりなどで先後協力すれば同一局面で2手ずつ手数違いの局面を容易に作れるようで、
		// max_game_plyの1手前の局面がすべての指し手が引き分けのスコアとなり、それがvmd_write_cacheに書き出されて、それを同一局面の若い手数
		// のときに、このvmd_write_cacheにヒットするため、おかしなことになる。(´ω｀)
		// ゆえに、vmd_write_cacheには、手数も含めて保持するか、max_game_plyを無視するかしないといけない。
		// 前者にしても角換わりで先後協力して手数をかなり伸ばせるのでいい実装とは言えないようだ。

		// df-pnの知見を利用すれば、もっと賢い方法がありそうだが、お手軽に実装するにはこのへんで妥協するしかなさそう。

		// あと、手数違いの局面は書き出しのときに手数が一番若いもの以外は間引くので、手数違いの局面を検出する処理を書いたほうがいいような気はするが、
		// vmd_write_cacheにはヒットするので、まあ、大したオーバーヘッドではないので良しとする。

		// 同一局面の場合、gamePlyが最小のものを採用する。(それ以外は千日手絡みのスコアが混じっている可能性があるので)
		// ※　it_write->second.gamePly > gamePly のとき、cacheにhitしなかったものとして再度調べる。

		auto sfen_left = StringExtension::trim_number(sfen);
		int gamePly = StringExtension::to_int(StringExtension::mid(sfen, sfen_left.length()), 0);
		
		auto it_write = vmd_write_cache.find(sfen_left);
		if (it_write != vmd_write_cache.end() && it_write->second.gamePly <= gamePly)
			return it_write->second.vmd_pair;

		// -- 定跡にhitするのか？(手数無視で)

			auto it_read = read_book.find(pos);
			if (it_read == nullptr || it_read->size() == 0)
				// このnodeについて、これ以上、何も処理できないでござる。
			{
			// 保存する価値がないと思うでvmd_write_cacheには保存しない
			return VMD_Pair(VALUE_NONE, MOVE_NONE, DEPTH_ZERO);
			}

			// -- このnodeを展開する。

			// 新しいほうの定跡ファイルに登録すべきこのnodeの候補手
			auto list = PosMoveListPtr(new PosMoveList());

			StateInfo si;

			// このnodeの最善手。rootColorがBLACK,WHITE用、それぞれ。
			VMD best[COLOR_NB];

			// ↑のbest.valueを上回る指し手であればその指し手でbest.move,best.depthを更新する。
			auto add_list = [&](Book::BookPos& bp, Color c /* このnodeのColor */, bool update_list)
			{
				ASSERT_LV3(bp.value != VALUE_NONE);

				// 定跡に登録する。
				bp.num = 1; // 出現頻度を1に固定しておかないとsortのときに評価値で降順に並ばなくて困る。

				if (update_list)
					list->push_back(bp);

				// このnodeのbestValueを更新したら、それをreturnのときに返す必要があるので保存しておく。
				VMD vmd((Value)bp.value, bp.bestMove, (Depth)bp.depth);

				// 値を上回ったのでこのnodeのbestを更新。
				if (best[c].value < vmd.value)
					best[c] = vmd;
			};

			// すべての合法手で1手進める。
			// 1) 子ノードがない　→　思考したスコアがあるならそれで代用　なければ　その子ノードについては考えない
			// 2) 子ノードがある　→　そのスコアを定跡として登録

			for (const auto& m : MoveList<LEGAL_ALL>(pos))
			{
				// この指し手をたどる
				this->do_move(pos, m, si);
				auto vmd_pair = build_tree_nega_max(pos, read_book, write_book);
				this->undo_move(pos, m);

				for (auto color : COLOR)
				{
					// root_colorが先手用のbestの更新と後手用のbestの更新とが、個別に必要である。(DRAW_VALUEの処理のため)
					auto& vmd = color == BLACK ? vmd_pair.black : vmd_pair.white;

					// colorがこの局面の手番(≒root_color)であるときだけこのnodeの候補手リストを更新する。
					// そうでないときもbestの更新は行う。
					auto update_list = color == pos.side_to_move();

					// 子nodeの探索結果を取り出す。
					// depthは、この先にbestMoveを辿っていくときleaf nodeまで何手あるかという値なのでここで定跡が途切れるならDEPTH_ZERO。
					auto value = vmd.value;
					auto nextMove = vmd.move;
					auto depth = vmd.depth + 1;

					if (value == VALUE_NONE)
					{
						// 子がなかった

						// 定跡にこの指し手があったのであれば、それをコピーしてくる。なければこの指し手については何も処理しない。
						auto it = std::find_if(it_read->begin(), it_read->end(), [m](const auto& x) { return x.bestMove == m; });
						if (it != it_read->end())
						{
							it->depth = DEPTH_ZERO; // depthはここがleafなので0扱い
							add_list(*it, color, update_list);
						}
					}
					else
					{
						// 子があったのでその値で定跡を登録したい。この場合、このnodeの思考の指し手にhitしてようと関係ない。

						// nega maxなので符号を反転させる
						value = -value;

						// 詰みのスコアはrootから詰みまでの距離に応じてスコアを修正しないといけない。
						if (value >= VALUE_MATE)
							--value;
						else if (value <= -VALUE_MATE)
							++value;

						//ASSERT_LV3(nextMove != MOVE_NONE);

						Book::BookPos bp(m, nextMove, value, depth, 1);
						add_list(bp, color, update_list);
					}
				}
			}

			// このnodeについて調べ終わったので格納
			std::stable_sort(list->begin(), list->end());
		write_book.book_body[sfen] = list;

			// 10 / 1000 node 処理したので進捗を出力
			output_progress();

#if 0
			// デバッグのためにこのnodeに関して、書き出す予定の定跡情報を表示させてみる。

			cout << pos.sfen() << endl;
			for (const auto& it : *list)
			{
				cout << it << endl;
			}
#endif

		// このnodeの情報をwrite_cacheに保存
		vmd_write_cache[sfen_left] = VmdPairGamePly(best,gamePly);

		return best;
	}

	// 定跡game treeを生成する機能
	void BookTreeBuilder::build_tree(Position & pos, istringstream & is)
	{
		// 定跡ファイル名
		// Option["book_file"]ではなく、ここで指定したものが処理対象である。
		string read_book_name, write_book_name;
		is >> read_book_name >> write_book_name;

		// 処理対象ファイル名の出力
		cout << "makebook build_tree .." << endl;
		cout << "read_book_name   = " << read_book_name << endl;
		cout << "write_book_name  = " << write_book_name << endl;

		auto oldValue = (std::string)Options["IgnoreBookPly"];
		Options["IgnoreBookPly"] = true;
		Tools::Finally f([&](){ Options["IgnoreBookPly"] = oldValue; });

		MemoryBook read_book, write_book;
		if (read_book.read_book(read_book_name) != 0)
		{
			cout << "Error! : failed to read " << read_book_name << endl;
			return;
		}

		// 定跡では絶対千日手回避するマンの設定
		int black_contempt =  50;  // 先手側の千日手を -50とみなす
		int white_contempt = 150;  // 後手側の千日手を-150とみなす
		int max_game_ply = 320; /* 320手で引き分けルール */

		string token;
		while ((is >> token))
		{
			if (token == "black_contempt")
				is >> black_contempt;
			else if (token == "white_contempt")
				is >> white_contempt;
			else if (token == "max_game_ply")
				is >> max_game_ply;
		}

		cout << "black_contempt = " << black_contempt << endl;
		cout << "white_contempt = " << white_contempt << endl;
		cout << "max_game_ply = " << max_game_ply << endl;

		// 初期局面から(depth 10000ではないものを)辿ってgame treeを構築する。

		total_node = 0;
		total_write_node = 0;

		this->black_contempt = black_contempt;
		this->white_contempt = white_contempt;
		this->max_game_ply = max_game_ply;

		// 平手、駒落ちなどの各初期局面をroot局面として、そこから辿っていく。
		for (auto sfen : start_sfens)
		{
			sync_cout << endl << "root sfen = " << sfen << sync_endl;

			StateInfo si;
			pos.set(sfen , &si, Threads.main());
			this->lastMoves.clear();
			vmd_write_cache.clear();

		// 定跡ファイルには手数無視でヒットしてくれないと、先後協力してplyが2手だけ増えた
		// 局面の定跡がいつまでも掘り進められなくなる。
		build_tree_nega_max(pos, read_book, write_book);
		}

		cout << endl;

		// 書き出し
		write_book.write_book(write_book_name);

		cout << "done." << endl;
	}

	// ----------------------------
	//  定跡ファイルの特定局面から定跡を掘る
	// ----------------------------

	void BookTreeBuilder::extend_tree_sub(Position & pos, MemoryBook & read_book, fstream & fs, const string & sfen , bool book_hit)
	{
		// 千日手に到達した局面は思考対象としてはならない。
		auto draw_type = pos.is_repetition(MAX_PLY);
		if (draw_type == REPETITION_DRAW || pos.game_ply() > this->max_game_ply)
			return;
		// →　それ以外の反復は大きなスコアがつくはずで、除外されるはず。

		// 詰み、宣言勝ちの局面も思考対象としてはならない。
		if (pos.is_mated() || pos.DeclarationWin() != MOVE_NONE)
			return;
		// →　直前のnodeで大きなスコアがつくはずだから除外されるはずだが。

		// このnodeを処理済みであるか
		// →　この処理遅いかなぁ..pos.key()で十分なような気もするが…。
		string this_sfen = pos.sfen();
		if (done_sfen.find(this_sfen) != done_sfen.end())
			return;
		done_sfen.insert(this_sfen);

		// 定跡にヒットしなかったのであれば、ここがleaf nodeの次の局面なので、
		// この局面までの"startpos moves..."を書き出す。
		auto it_read = read_book.find(pos);
		if (it_read == nullptr || it_read->size() == 0)
		{
			// 直前のnodeで定跡の指し手を指したのではないなら帰る。
			if (!book_hit)
				return;

			// 直前のnodeで定跡の指し手を指して、かつ、ここで定跡にhitしなかったので
			// ここが定跡treeのleaf nodeの一つ先のnodeであることが確定した。

			// 直前のnodeのscoreが +1,-1 であるなら、これは千日手スコアだと考えられるので、このときにのみ
			// この局面を延長する。(Learner::search()の仕様がダサくて、"makebook think"コマンドは千日手のとき+1もありうる…)
			// →　Contemptの値は無視して常に0になるように修正した。[2019/05/19]

			// この条件式は単純化できるが、条件を追加するかも知れないので過剰な単純化はしない。
			bool extend = 
				(this->enable_extend_range && extend_range1 <= this->lastEval && this->lastEval <= extend_range2) ||
				(!this->enable_extend_range);

			// この局面に到達するまでの"startpos moves ..."をとりま出力。
			if (extend)
			{
				fs << sfen << endl;
				//fs << pos.sfen() << endl;

				total_write_node++;
			}

			return;
		}

		// -- 定跡に登録されているこのnodeの指し手を展開する。

		StateInfo si;
		auto turn = pos.side_to_move();

		for (const auto& m : MoveList<LEGAL_ALL>(pos))
		{
			// 定跡の指し手以外の指し手でも、次の局面で定跡にhitする指し手を探す必要がある。

			auto it = std::find_if(it_read->begin(), it_read->end(), [m](const auto & x) { return x.bestMove == m; });
			// 定跡にhitしたのか
			bool book_hit = it != it_read->end();

			if (book_hit)
			{
				// eval_limitを超える枝だけを展開する。
				if ((turn == BLACK && it->value >= black_eval_limit) ||
					(turn == WHITE && it->value >= white_eval_limit))
					this->lastEval = it->value;
				else
					continue; // この枝はこの時点で先を辿らない
			}
			else {
				// 定跡の指し手ではないが、次のnodeで定跡にhitするなら辿って欲しい。
				this->lastEval = 0;
			}
			this->do_move(pos,m, si);
			extend_tree_sub(pos, read_book, fs, sfen + " " + to_usi_string(m) , book_hit);
			this->undo_move(pos,m);
		}

		output_progress();
	}

	// "position ..."の"..."の部分を解釈する。
	int BookTreeBuilder::feed_position_string(Position & pos, const string & line, StateInfo * states, Thread * th)
	{
		this->lastMoves.clear();
		stringstream is(line);

		// 以下、usi.cppのposition_cmd()に似たコード

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

		pos.set(sfen, &states[0], th);

		// 指し手のリストをパースする(あるなら)
		Move move;
		while (is >> token && (move = USI::to_move(pos, token)) != MOVE_NONE)
		{
			if (!pos.pseudo_legal(move) || !pos.legal(move))
			{
				cout << "Error ! : " << line << " unknown token = " << token << endl;
				return 1;
			}
			pos.do_move(move , states[pos.game_ply()]);
		}
		return 0; // 読み込み終了
	}

	// 定跡ファイルを読み込んで、指定局面から深掘りするために必要な棋譜を生成する。
	void BookTreeBuilder::extend_tree(Position & pos, istringstream & is)
	{
		string read_book_name, read_sfen_name, write_sfen_name;
		is >> read_book_name >> read_sfen_name >> write_sfen_name;

		// 延長する評価値の下限
		int black_eval_limit = -50, white_eval_limit = -150;

		int max_game_ply = 320; /* 320手で引き分けルール */

		// 延長するleafの評価値の範囲(千日手スコアの局面のみを延長する場合に用いる)
		bool enable_extend_range = false;
		int extend_range1, extend_range2;

		string token;
		while ((is >> token))
		{
			if (token == "black_eval_limit")
				is >> black_eval_limit;
			else if (token == "white_eval_limit")
				is >> white_eval_limit;
			else if (token == "extend_range")
			{
				enable_extend_range = true;
				is >> extend_range1 >> extend_range2;
			}
			else if (token == "max_game_ply")
				is >> max_game_ply;
		}

		// 処理対象ファイル名の出力
		cout << "makebook extend tree .." << endl;
		
		cout << "read_book_name   = " << read_book_name << endl;
		cout << "read_sfen_name  = " << read_sfen_name << endl;
		cout << "write_sfen_name  = " << write_sfen_name << endl;

		cout << "black_eval_limit = " << black_eval_limit << endl;
		cout << "white_eval_limit = " << white_eval_limit << endl;

		if (enable_extend_range)
			cout << "extend_range = [" << extend_range1 << "," << extend_range2 << "]" << endl;

		cout << "max_game_ply = " << max_game_ply << endl;

		// read_book_name  : やねうら王の定跡形式(拡張子.db)
		// read_sfen_name  : USIのposition形式。例:"startpos moves ..."
		// write_sfen_name : 同上。

		MemoryBook read_book;
		if (read_book.read_book(read_book_name) != 0)
		{
			cout << "Error! : failed to read " << read_book_name << endl;
			return;
		}
		vector<string> lines;
		read_all_lines(read_sfen_name, lines);

		// 初期局面から(depth 10000ではないものを)辿ってgame treeを構築する。

		fstream fs;
		fs.open(write_sfen_name , ios::out);

		total_node = 0;
		total_write_node = 0;
		this->done_sfen.clear();

		this->black_eval_limit = black_eval_limit;
		this->white_eval_limit = white_eval_limit;
		this->enable_extend_range = enable_extend_range;
		this->extend_range1 = extend_range1;
		this->extend_range2 = extend_range2;
		this->max_game_ply = max_game_ply;

		// これより長い棋譜、食わせないやろ…。
		std::vector<StateInfo> states(1024);

		for (int i = 0; i < (int)lines.size(); ++i)
		{
			auto& line = lines[i];

			cout << "extend[" << i << "] : " << line << endl;
			feed_position_string(pos, line, &states[0], Threads.main());

			// "startpos"や、"sfen ..."の形でmovesの文字が含まれていなければ
			// "moves"をこの時点で追加しておく。

			auto sp = StringExtension::split(line);
			if (std::find(sp.begin(), sp.end(), "moves") == sp.end())
				line = line + " moves";

			this->lastEval = 0;
			extend_tree_sub(pos, read_book, fs, line , true);
		}
		fs.close();
		cout << endl;
		cout << "done." << endl;
	}

	// 定跡の無限自動掘り
	/*
		課題局面から自動的に定跡を掘っていく。

		// 一度目だけ課題局面までを思考させる。(課題局面までの定跡を掘らなくていいなら、この動作は不要)
		MultiPV 4
		makebook think book/kadai_sfen.txt book/book_test.db depth 8 startmoves 1 moves 32
		// →　これやめよう。不要だわ。

		// そのあと、以下を無限に繰り返す。
		MultiPV 4
		makebook extend_tree book/book_test.db book/kadai_sfen.txt book/think_sfen.txt
		makebook think book/think_sfen.txt book/book_test.db depth 8 startmoves 1 moves 32

		↓この２つを統合したコマンドを作成する

		例)
		makebook endless_extend_tree book/book_test.db book/kadai_sfen book/think_sfen.txt depth 8 startmoves 1 moves 32 loop 10 nodes 100000
	*/
	void BookTreeBuilder::endless_extend_tree(Position& pos, istringstream& is)
	{
		string read_book_name, read_sfen_name, think_sfen_name;
		is >> read_book_name >> read_sfen_name >> think_sfen_name;

		cout << "endless_extend_tree" << endl;

		cout << "read_book_name   = " << read_book_name << endl;
		cout << "read_sfen_name  = " << read_sfen_name << endl;
		cout << "write_sfen_name  = " << think_sfen_name << endl;

		string token;
		int depth = 8, start_moves = 1, end_moves = 32 , iteration = 256;
		int black_eval_limit = -50, white_eval_limit = -150;
		uint64_t nodes = 0; // 指定がなければ0にしとかないと..
		bool enable_extend_range = false;
		int extend_range1, extend_range2;
		int max_game_ply = 320;

		while ((is >> token))
		{
			if (token == "depth")
				is >> depth;
			else if (token == "startmoves")
				is >> start_moves;
			else if (token == "moves")
				is >> end_moves;
			else if (token == "loop")
				is >> iteration;
			else if (token == "black_eval_limit")
				is >> black_eval_limit;
			else if (token == "white_eval_limit")
				is >> white_eval_limit;
			else if (token == "nodes")
				is >> nodes;
			else if (token == "extend_range")
			{
				enable_extend_range = true;
				is >> extend_range1 >> extend_range2;
			}
			else if (token == "max_game_ply")
				is >> max_game_ply;
		}

		cout << "startmoves " << start_moves << " moves " << end_moves << " nodes " << nodes << endl;
		cout << "loop = " << iteration << endl;
		cout << "black_eval_limit = " << black_eval_limit << endl;
		cout << "white_eval_limit = " << white_eval_limit << endl;
		if (enable_extend_range)
			cout << "extend_range = [" << extend_range1 << "," << extend_range2 << "]" << endl;
		cout << "max_game_ply = " << max_game_ply << endl;

		// コマンドの実行
		auto do_command = [&](string command)
		{
			cout << "> makebook " + command << endl;

			// "makebook"コマンド処理部にコマンドテキスト経由で移譲してしまう。
			istringstream iss(command);
			makebook_cmd(pos, iss);
		};

		for (int i = 0; i < iteration; ++i)
		{
			cout << "makebook endless_extend_tree : iteration " << i << endl;

			string command;
#if 0
			if (i == 0)
			{
				// 初回だけ
				// "makebook think book/kadai_sfen.txt book/book_test.db depth 8 startmoves 1 moves 32"

				command = "think " + read_sfen_name + " " + read_book_name + " depth " + to_string(depth)
					+ " startmoves " + to_string(start_moves) + " moves " + to_string(end_moves) + " nodes " + to_string(nodes);
				do_command(command);
			}
			else
#endif
			{
				// 2回目以降

				// "makebook extend_tree book/book_test.db book/kadai_sfen.txt book/think_sfen.txt"
				// "makebook think book/think_sfen.txt book/book_test.db depth 8 startmoves 1 moves 32"

				command = "extend_tree " + read_book_name + " " + read_sfen_name + " " + think_sfen_name +
					" black_eval_limit " + to_string(black_eval_limit) + " white_eval_limit " + to_string(white_eval_limit) +
					(enable_extend_range ? " extend_range " + to_string(extend_range1) + " " + to_string(extend_range2):"") +
					" max_game_ply " + to_string(max_game_ply);
				do_command(command);


				// ここで思考対象局面がなくなっていれば終了する必要がある。
				// movesを指定してあるのでthink_sfen_nameのファイルが空になるとは限らなくて、この終了判定難しい…。
				// なので、やらない(´ω｀)

				command = "think " + think_sfen_name + " " + read_book_name + " depth " + to_string(depth)
					+ " startmoves " + to_string(start_moves) + " moves " + to_string(end_moves) + " nodes " + to_string(nodes);
				do_command(command);

			}
		}

	}

}

namespace Book
{
	// 2019年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2019(Position& pos, istringstream& is, const string& token)
	{
		if (token == "build_tree")
		{
			BookTreeBuilder builder;
			builder.build_tree(pos, is);
			return 1;
		}

		if (token == "extend_tree")
		{
			BookTreeBuilder builder;
			builder.extend_tree(pos, is);
			return 1;
		}

		if (token == "endless_extend_tree")
		{
			BookTreeBuilder builder;
			builder.endless_extend_tree(pos, is);
			return 1;
		}

		return 0;
	}
}

#endif
