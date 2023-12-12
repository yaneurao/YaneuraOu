#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD)

// ------------------------------
//    ペタショック化コマンド
// ------------------------------

// 定跡自体は、別の何らかのスクリプトによって、やねうら王形式の定跡ファイルが書き出されているものとする。
// その定跡ファイルに対してそれぞれの局面に対してmin-max探索を行った結果の定跡ファイルを書き出すのが、
// このペタショック化である。

// コマンド例)
//    makebook petashock book1.db user_book1.db
// 
// book1.dbをmin-max探索してuser_book1.dbを書き出す。
// 
// エンジンオプションのFlippedBookがtrueなら、先手番の局面しか書き出さない。(後手番の局面はそれをflipした局面が書き出されているはずだから)
//

#include <sstream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <limits>
#include <random>
#include "book.h"
#include "../thread.h"
#include "../position.h"
#include "../misc.h"

using namespace std;

namespace MakeBook2023
{
	// BookMoveのポインターみたいなやつ。これで局面の行き来を行う。
	// しばらくは42億を超えることはないと思うので32bitでいいや。
	typedef u32 BookNodeIndex;
	// BookNodeIndexのnullptrの時の値。
	const BookNodeIndex BookNodeIndexNull = numeric_limits<BookNodeIndex>::max();

	// BoonNodeの評価値で∞を表現する定数。
	const int BOOK_VALUE_INF  = numeric_limits<int>::max();

	// ペタショック前の定跡DBに指し手の評価値を99999にして書き出しておくと、
	// これは指し手は存在するけど評価値は不明の指し手である。(という約束にする)
	// これは棋譜の指し手などを定跡DBに登録する時に評価値が確定しないのでそういう時に用いる。
	const int BOOK_VALUE_NONE = -99999;

	// 定跡で千日手手順の時のdepth。∞であることがわかる特徴的な定数にしておくといいと思う。
	const int BOOK_DEPTH_INF = 9999;

	// 千日手の状態
	// 最終的には書き出した定跡DBのdepthに反映させる。
	// depth
	//  +10000 : 先手は千日手を打開できない。
	//  +20000 : 後手は千日手を打開できない。
	//  +30000 : 先手・後手ともに千日手を打開できない。
	struct DrawState
	{
		DrawState(u8 state):state(state){}

		// bit0 : 先手は千日手を打開する権利を持っていない。
		// bit1 : 後手は千日手を打開する権利を持っていない。

		// つまり、
		//   00b : 先後千日手を打開する権利を持っている。
		//   01b : 先手は(後手が千日手を選んだ時に)千日手を打開できない。(後手は打開できる)
		//   10b : 後手は(先手が千日手を選んだ時に)千日手を打開できない。(先手は打開できる)
		//   11b : 先手・後手ともに千日手を打開できない。

		u8 state;

		/*
			後手番で、11bと00bとの指し手があった時に 後手は00bの指し手を選択するが、親には01b を伝播する。
			この時、01bの指し手を選択すると、これがループである可能性がある。

			A→B→C→D→A
				  D→E
			となっている場合、後手が誤ったほうを選択するとループになる。

			だから、bestの選出は、後手は
			  00b > 01b > 10b > 11b
			先手は、
			  00b > 10b > 01b > 11b
			の順番でなければならない。
		*/

		// 比較オペレーター
		bool operator==(const DrawState& v) const { return state==v.state;}
		bool operator!=(const DrawState& v) const { return !(*this==v); }

		// 手番cにおいて評価値が同じ時に this のほうが yより勝るか。
		bool is_superior(DrawState y, Color c) const
		{
			// 先手にとってのbestを選出する順番
			constexpr int black_[] = {1,3,2,4};
			// 後手にとってのbestを選出する順番
			constexpr int white_[] = {1,2,3,4};

			if (c==BLACK)
				return black_[this->state] < black_[y.state];
			else
				return white_[this->state] < white_[y.state];
		}

		// 手番cの時、評価値が同じでDrawStateだけ異なる指し手がある時に
		// このnodeのDrawStateを求める。
		// 
		// 例)
		//   後手番で評価値が同じである 00bと11bの指し手があるとして
		//   後手はどちらかを選べるので、後手には千日手の権利があることになる。(先手はそれを決める権利がない)
		//   ゆえに、このとき、このnodeの draw_stateは、01b となる。
		//
		//   つまり、
		//     手番側のbitは、0と1があるなら0(bit and)
		//     非手番側のbitは、0と1があるなら1(bit or)
		//   をすればいいということである。
		void select(DrawState y, Color c)
		{
			*this = select_static(*this, y , c);
		}

		// ↑のstatic版
		static DrawState select_static(DrawState x , DrawState y , Color c)
		{
			u8 our_bit  =  c == BLACK ? 1 : 2;
			u8 them_bit = ~c == BLACK ? 1 : 2;
			
			return ((x.state & our_bit ) & (y.state & our_bit ))
				 | ((x.state & them_bit) | (y.state & them_bit));
		}

		// depthに変換する。
		// stateを1万倍しておく。
		int to_depth() const
		{
			return state * 10000;
		}
	};

	// 定跡の評価値とその時のdepth、千日手の状態をひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0), draw_state(0){}

		ValueDepth(int value, int depth)
			: value(value) , depth(depth) , draw_state(0){}

		ValueDepth(int value, int depth, DrawState state)
			: value(value) , depth(depth) , draw_state(state){}

		int value;
		int depth;
		DrawState draw_state;

		// 比較オペレーター
		bool operator==(const ValueDepth& v) const { return value==v.value && depth==v.depth && draw_state==v.draw_state;}
		bool operator!=(const ValueDepth& v) const { return !(*this==v); }

		// 優れているかの比較
		// 1. 評価値が優れている。
		// 2. 評価値が同じ場合、DrawStateが優れている
		// 3. 評価値とDrawStateも同じ場合、評価値が正ならdepthが小さいほうが優れている。
		// 4. 評価値とDrawStateも同じ場合、評価値が負ならdepthが大きいほうが優れている。
		//
		// 3., 4.の理屈としては、同じ評価値を持つ２つのノード(two nodes with the same value)問題があることを発見したからだ。
		// いま 
		//     A→B→C→D→A
		//     A→E→F→G
		//     B→X
		// のような経路があるとする。
		//
		// X,Gは同じ評価値でAの手番側から見て100であり、Aの手番側はここを目指すとする。
		// A→B→XはAからXに2手でvalue=100にいける。
		// A→E→F→GはAからGに3手でvalue=100にいける。
		// Aの手番を持たされた側は、depthが小さいからと言ってA→Bを選ぶとBの手番側がXに行く手を選択せずにCに行く手を選択してしまう。
		// そうするとC→D→AとAに戻ってきて千日手になる。
		// Bの手番側は、B→C→D→A→E→F→Gのコースを選んだほうが得なので、depthの大きい側を選ぶべきなのである。
		// この理屈から、評価値が正(千日手スコアより大きい)のほうは、千日手を回避するためにdepthが小さいほうを目指すが(親にそれを伝播する)、
		// 評価値が負(千日手スコアより小さい)のほうは、千日手にするためにdepthが大きいほうを目指す(親にそれを伝播する)べきなのである。
		// ゆえに、評価値の正負によって、どちらのdepthの指し手を選ぶかが変わるのである。
		// 
		// ※ PVが同じleafに到達する２つの指し手があるとして、depthが大きいほうは循環を含んでいる可能性が高いので、
		//    千日手にしたい側はdepthが大きいほうを、したくない側はdepthが小さいほうを選ぶという理屈。
		//
		bool is_superior(ValueDepth& v, Color color) const
		{
			// 評価値ベースの比較
			if (this->value != v.value)
				return this->value > v.value;

			// DrawStateベースの比較

			// 値が同じならdepthの低いほうを採用する。
			// なぜなら、循環してきて、別の枝に進むことがあり、それはdepthが高いはずであるから。
			if (this->draw_state != v.draw_state)
				return this->draw_state.is_superior(v.draw_state, color);

			// depthベースの比較。評価値の符号で場合分けが生ずる。
			// 一貫性をもたせるためにvalueが千日手スコアの場合は、先手ならdepthの小さいほうを目指すことにしておく。
			auto dv = draw_value(REPETITION_DRAW, color);
			if ((this->value > dv) || (this->value == dv && (color == BLACK)))
				return this->depth < v.depth;
			else
				return this->depth > v.depth;
		}

		// depthにdraw_stateの状態を反映させる。
		void draw_state_to_depth() { depth += draw_state.to_depth(); }
	};

	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	struct BookMove
	{
		// moveの指し手がleafである場合。
		BookMove(Move move,int value,int depth):
			move(move),vd(ValueDepth(value,depth)),next(BookNodeIndexNull){}

		// moveの指し手がleafではない場合。
		BookMove(Move move,ValueDepth vd, BookNodeIndex next):
			move(move),vd(vd),next(next){}

		// move(4) + value(4) + depth(4) + next(4) = 16 bytes

		// 指し手
		Move move;

		// ↑の指し手を選んだ時の定跡ツリー上の評価値
		// (定跡ツリー上でmin-max探索をした時の評価値)
		ValueDepth vd;

		// ↑のmoveで局面を進めた時の次の局面のBookMoveポインター。
		// これが存在しない時は、BookMoveIndexNull。
		BookNodeIndex next;
	};


	// あるnodeに対してその親nodeとその何番目の指し手がこのnodeに接続されているのかを保持する構造体。
	struct ParentMove
	{
		ParentMove(BookNodeIndex parent,size_t move_index):parent(parent),move_index((u32)move_index){}

		BookNodeIndex parent;
		u32 move_index;
	};

	// 定跡の1つの局面を表現する構造体。
	// 高速化のために、hashkeyで行き来をする。
	struct BookNode
	{
		// この局面の手番(これがないと探索するときに不便)
		Color color;

		// このnodeからの出次数
		u64 out_count = 0;

		// このnodeへの入次数
		//u64 in_count = 0;
		// これは、↓このlistを見ればいいから、あえて持つ必要はなさそう。

		// このnodeの親のlist
		// (つまり、その局面から指し手で1手進めてこの局面に到達できる)
		vector<ParentMove> parents;

		// 後退解析IIの時の前回の親に伝播したValueDepthの値。
		// 全ノード、これが前回と変わっていないなら後退解析を終了できる。
		ValueDepth lastParentVd = ValueDepth(BOOK_VALUE_INF, 0);

		// 指し手
		vector<BookMove> moves;

		// key(この局面からhash keyを逆引きしたい時に必要になるので仕方なく追加してある)
		HASH_KEY key;

		// 初期局面からの手数
		int ply = 0;
	};

	// 後退解析IVで用いる構造体。
	struct ParentMoveEx
	{
		ParentMoveEx(ParentMove parent_move , bool is_deleted , ValueDepth best)
			: parent_move(parent_move), is_deleted(is_deleted) , best(best){}

		// ある子局面にいたる指し手
		ParentMove parent_move;

		// その子の指し手がすべてMOVE_NONEで、parent_moveは削除されるべきであるか。
		bool is_deleted;

		// is_deleted == falseの時、子のbest_value。これを反転させたものが、parent_moveの評価値となる。
		ValueDepth best;
	};

	// plyを求めるためにBFSする時に必要な構造体
	struct BookNodeIndexPly
	{
		BookNodeIndexPly(BookNodeIndex index, int ply):
			index(index), ply(ply){}

		BookNodeIndex index;
		int ply;
	};

	// hashkeyのbit数をチェックする。
	void hashbit_check()
	{
		// 高速化のために、HASH_KEYだけで局面を処理したいので、hashの衝突はあってはならない。
		// そのため、HASH_KEY_BITSは128か256が望ましい。
		if (HASH_KEY_BITS < 128)
		{
			cout << "WARNING! : HASH_KEY_BITS = " << HASH_KEY_BITS << " is too short." << endl;
			cout << "    Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
		}
	}

	// ペタショック化
	class PetaShock
	{
	public:

		// 定跡をペタショック化する。
		// next : これが非0の時はペタショック化ではなく次に思考対象とすべきsfenをファイルに出力する。
		void make_book(Position& pos , istringstream& is, bool next)
		{
			hashbit_check();

			// 千日手の遡り手数を初手まで遡ることにする。
			pos.set_max_repetition_ply(MAX_PLY);

			string readbook_path;
			string writebook_path;
			string root_sfens_path ;

			// 次の思考対象とすべきsfenを書き出す時のその局面の数。
			u64 next_nodes = 0;

			// leaf nodeの指し手に加える乱数の大きさ
			int eval_noise = 0;
			// 1番目の指し手のdepthが0なら、その指し手で進めた局面を列挙するモード。
			bool low_depth = false;

			is >> readbook_path >> writebook_path;

			readbook_path  = Path::Combine("book",readbook_path );
			writebook_path = Path::Combine("book",writebook_path);

			if (next)
			{
				is >> next_nodes;
				string token;
				while (is >> token)
				{
					if (token == "eval_noise")
						is >> eval_noise;
					else if (token == "low_depth")
						low_depth = true;
				}

			}

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			if (next)
			{
				// 書き出すsfenの数
				cout << "write next_sfens : " << next_nodes << endl;
				cout << "eval_noise       : " << eval_noise << endl;
				cout << "low_depth        : " << low_depth << endl;

				// これは現状ファイル名固定でいいや。
				root_sfens_path = Path::Combine("book","root_sfens.txt");
				cout << "root_sfens_path    : " << root_sfens_path << endl;
			}

			cout << "readbook_path      : " << readbook_path  << endl;
			cout << "writebook_path     : " << writebook_path << endl;

			// DrawValueBlackの反映(DrawValueWhiteは無視)
			drawValueTable[REPETITION_DRAW][BLACK] =   Value((int)Options["DrawValueBlack"]);
			drawValueTable[REPETITION_DRAW][WHITE] = - Value((int)Options["DrawValueBlack"]);

			// 引き分けのスコアを変更したいなら先に変更しておいて欲しい。
			cout << "draw_value_black   : " << draw_value(REPETITION_DRAW, BLACK) << endl;
			cout << "draw_value_white   : " << draw_value(REPETITION_DRAW, WHITE) << endl;

			// 反転された局面を書き出すのか。(FlippedBookがtrueなら書き出さない)
			// すなわち、後手番の局面はすべて書き出さない。(反転された先手番の局面を書き出しているはずだから)
			bool flipped_book = Options["FlippedBook"];
			cout << "FlippedBook        : " << flipped_book << endl;

			cout << endl;

			// 手数無視のオプションを有効にすると、MemoryBook::read_book()は、
			// sfen文字列末尾の手数を無視してくれる。
			Options["IgnoreBookPly"] = true;
			// 定跡生成書き出したあとに普通に探索させることはありえないだろうから
			// 元のオプションへの復元は行わない。(行いたいならば、benchmark.cppを参考にコードを修正すべし。)

			Book::MemoryBook book;
			if (book.read_book(readbook_path).is_not_ok())
			{
				cout << "read book error" << endl;
				return ;
			}

			// memo : 指し手の存在しない局面はそんな定跡ファイル読み込ませていないか、
			//        あるいはMemoryBookが排除してくれていると仮定している。

			// === helper function ===

			// あるnodeのbestと親に伝播すべきparent_vdとを得るヘルパー関数。
			// best_index : 何番目の指し手がbestであったのかを返す。
			auto get_bestvalue = [&](BookNode& node , ValueDepth& parent_vd , size_t& best_index)
			{
				// まずこのnodeのbestを得る。
				ValueDepth best(-BOOK_VALUE_INF, BOOK_DEPTH_INF);
				// 親に伝播するbest
				parent_vd = ValueDepth(-BOOK_VALUE_INF,BOOK_DEPTH_INF,DrawState(3));

				best_index = 0;
				for(size_t i = 0 ; i< node.moves.size() ; ++i)
				{
					const auto& book_move = node.moves[i];

					// MOVE_NONEならこの枝はないものとして扱う。
					if (book_move.move == MOVE_NONE)
						continue;

					if (book_move.vd.is_superior(best, node.color))
					{
						best = book_move.vd;
						best_index = i;
					}

					if (parent_vd.value < book_move.vd.value)
						parent_vd = book_move.vd;
					else if (parent_vd.value == book_move.vd.value)
					{
						// valueが同じなのでdraw_stateはORしていく必要がある。
						parent_vd.draw_state.select(best.draw_state, node.color);

						// depthに関しては、bestのdepthを採用すればbestの指し手を追いかけていった時の手数になる。
						parent_vd.depth = best.depth;
					}
				}

				// 親に伝播するほうはvalueを反転させておく。
				parent_vd.value =   - parent_vd.value;
				parent_vd.depth = min(parent_vd.depth + 1 , BOOK_DEPTH_INF);

				return best;
			};

			// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
			// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスする処理を行うヘルパー関数。
			auto adjust_second_bestvalue = [&](BookNode& node)
			{
				ValueDepth vd;
				size_t _;
				auto best = get_bestvalue(node, vd , _);

				for(auto& book_move : node.moves)
				{

					if (   best.value == book_move.vd.value
						&& (
								// draw_state違い。これはbestが優先されるべき。
							   ( best.draw_state != book_move.vd.draw_state)
								// あるいは、draw_stateは同じだが、経路の長さが違う場合。これはbestのほうのみが選ばれるべき。
							|| ( best.draw_state == book_move.vd.draw_state && best.depth != book_move.vd.depth)
							)
						)
					{
						// depthが最小でない指し手の評価値を1だけ減らしておく。

						// bestの値しか親には伝播しないので、ここで引いたところで
						// このnodeにしか影響はない。
						book_move.vd.value--;
					}

					// depthにdraw stateを反映させる。
					book_move.vd.depth += book_move.vd.draw_state.to_depth();
				}
			};

			// 定跡の開始局面。
			// "book/root_sfens.txt"からrootのsfen集合を読み込むことにする。
			// これはUSIのposition文字列の"position"を省略した文字列であること。
			// つまり、
			//   startpos
			//   startpos moves ...
			//   sfen ...
			// のような文字列である。
			vector<string> root_sfens;
			if (next)
			{
				if (SystemIO::ReadAllLines(root_sfens_path,root_sfens).is_not_ok())
					root_sfens.emplace_back(BookTools::get_start_sfens()[0]);
			}

			// 局面数などをカウントするのに用いるカウンター。
			u64 counter = 0;

			// progress表示用
			Tools::ProgressBar progress;

			// 盤面を反転させた局面が元の定跡DBにどれだけ含まれていたかを示すカウンター。
			u64 flipped_counter = 0;

			// 盤面を反転させた局面も定跡に登録するかのフラグ。
			// makebookコマンドのオプションでON/OFF切り替えられるようにすべきか？
			const bool register_flipped_position = true;

			// 反転局面の登録
			if (register_flipped_position)
			{
				cout << "Register flipped pos:" << endl;

				progress.reset(book.size());

				Book::MemoryBook book2;
				book.foreach([&](const string& sfen,const Book::BookMovesPtr book_moves){
					StateInfo si;
					pos.set(sfen,&si,Threads.main());
					string flip_sfen = pos.flipped_sfen(-1); // 手数なしのsfen文字列
					progress.check(++counter);

					if (book.find(flip_sfen) != nullptr)
					{
						// すでに登録されていた
						++flipped_counter;
						return;
					}

					Book::BookMovesPtr flip_book_moves(new Book::BookMoves());
					for(const auto& bm : *book_moves)
					{
						// 盤面を反転させた指し手として設定する。
						// ponderがMOVE_NONEでもflip_move()がうまく動作することは保証されている。
						Book::BookMove flip_book_move(flip_move(bm.move), flip_move(bm.ponder), bm.value , bm.depth , bm.move_count);
						flip_book_moves->push_back(flip_book_move);
					}
					book2.append(flip_sfen, flip_book_moves);
				});

				// 生成されたflipped bookをmergeする。
				book.merge(book2);
			}

			cout << "Register SFENs      : " << endl;

			counter = 0;
			progress.reset(book.size());

			// まず、出現する局面すべてのsfenに対して、それをsfen_to_hashkeyに登録する。
			// sfen文字列の末尾に手数が付与されているなら、それを除外する。→ IgnoreBookPly = trueなので除外されている。
			book.foreach([&](string sfen, const Book::BookMovesPtr book_moves){

				// 同じ値のキーがすでに登録されていないかをチェックしておく。
				if (this->sfen_to_index.count(sfen) > 0)
				{
					cout << "Error! : Hash Conflict! Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
					Tools::exit();
				}

				StateInfo si,si2;
				pos.set(sfen,&si,Threads.main());

				// 局面ひとつ登録する。
				BookNodeIndex index = (BookNodeIndex)this->book_nodes.size();
				this->book_nodes.emplace_back(BookNode());

				auto key = pos.state()->hash_key();
				this->sfen_to_index[sfen]   = index;
				this->hashkey_to_index[key] = index;
				// 逆引きするのに必要なのでhash keyも格納しておく。
				book_nodes.back().key = key;

				progress.check(++counter);
			});

			// 局面の合流チェック

			cout << "Convergence Check   :" << endl;

			// sfen nextの時はこの処理端折りたいのだが、parent局面の登録などが必要で
			// この工程を端折るのはそう簡単ではないからやめておく。

			// 合流した指し手の数
			u64 converged_moves = 0;

			counter = 0;
			progress.reset(book.size());

			book.foreach([&](string sfen, const Book::BookMovesPtr book_moves){

				StateInfo si,si2;
				pos.set(sfen,&si,Threads.main());
				HASH_KEY key = pos.state()->hash_key();
				// 先に定跡局面は登録したので、このindexが存在することは保証されている。
				BookNodeIndex index = this->hashkey_to_index[key];

				// いまからこのBookNodeを設定していく。
				BookNode& book_node = this->book_nodes[index];

				// 手番をBookNodeに保存しておく。
				book_node.color = pos.side_to_move();

				// ここから全合法手で一手進めて既知の局面に行き着くかを調べる。
				for(auto move:MoveList<LEGAL_ALL>(pos))
				{
					pos.do_move(move,si2);

					// moveで進めた局面が存在する時のhash値。
					HASH_KEY next_hash = pos.state()->hash_key();

					if (this->hashkey_to_index.count(next_hash) > 0)
					{
						// 定跡局面が存在した。

						// 元のnodeの出次数と、next_nodeへの入次数をインクリメントしてやる。
						// (後退解析みたいなことをしたいので)
						book_node.out_count++;
						BookNodeIndex next_book_node_index = this->hashkey_to_index[next_hash];
						BookNode&     next_book_node       = this->book_nodes[next_book_node_index];

						// parentのlistに、元のnodeを追加しておく。
						next_book_node.parents.emplace_back(ParentMove(index,book_node.moves.size()));

						// どうせmin-maxして、ここの評価値とdepthは上書きされるが、後退解析するので千日手の時のスコアで初期化する。

						// 千日手の時のvalueとdepth。
						// これは、
						//	value = draw_value
						//	depth = ∞
						//  draw_state = 先後ともに回避できない
						BookMove book_move(move,
							ValueDepth(
								draw_value(REPETITION_DRAW, book_node.color),
								BOOK_DEPTH_INF,
								DrawState(3)
							),
							next_book_node_index);


						book_node.moves.emplace_back(book_move);
						converged_moves++;
					}

					pos.undo_move(move);
				}

				// 定跡DB上のこの局面の指し手も登録しておく。
				book_moves->foreach([&](const Book::BookMove& bm){
						Move move = pos.to_move(bm.move);
						int depth = bm.depth;
						int value = bm.value;

						// これがbook_nodeにすでに登録されているか？
						if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
						{
							// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので登録しなくて良い。)
							// 登録されていなかったということは、ここから接続されているnodeはないので、出次数には影響を与えない。
							BookMove book_move(move, value, depth);
							book_node.moves.emplace_back(book_move);
						} else {
							// 登録されていたのでconvergeしたやつではなかったから、convergeカウンターはデクリメントしておく。
							converged_moves--;
						}
					}
				);

				progress.check(++counter);
			});

			//cout << "converged_moves : " << converged_moves << endl;

			// やねうら王の定跡DBの構築
			//cout << "build yaneuraou book : " << endl;

			// 下準備がすべて終わったのでbuildしてみる。

			// sfen_to_hashkeyとhashkey_to_book_nodeを頼りに、
			// 定跡上のmin-max探索する。

			// まず後退解析みたいなことをして、ループ以外はある程度処理しておく。
			// アルゴリズム的には、
			//   queue = 出次数0のnodeの集合
			//   while queue:
			//     node = queue.pop()
			//     eval  = max(node.各定跡の指し手の評価値)
			//     depth = evalがmaxだった時の定跡のdepth
			//     foreach parent in node.parents:
			//       parentのnodeへの指し手の評価値 = - eval     # min-maxなので評価値は反転する
			//       parentのnodeへの指し手のdepth  =   depth+1  # 1手深くなる
			//       if --parent.出次数 == 0:                    # 出次数が0になったのでこの局面もqueueに追加する。
			//         queue.push(parent)


			// leaf nodeの指し手の評価値に乱数を加える。
			if (eval_noise != 0)
			{
				cout << "add random bonus for every leaf move." << endl;
				progress.check(book_nodes.size());

				// 乱数生成器
				std::random_device rd;
				std::mt19937 gen(rd());

				// 正規分布を定義（平均 = 0 , 標準偏差 = eval_noise）
				std::normal_distribution<> d(0, eval_noise);

				u64 counter = 0;

				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					BookNode& book_node = book_nodes[i];

					// leaf nodeでbest move以外の指し手が展開されるのは嫌だ。
					// 
					// そこで、このnodeの指し手すべてに同一のノイズを加算する。
					// こうすることでbest valueを持つmoveが展開される。
					// (その指し手がleaf nodeでないなら、それが伝播してきて置き換わるから問題なし)

					int noise = int(d(gen));
					for(auto& move : book_node.moves)
						move.vd.value += noise;

					progress.check(++counter);
				}
			}

			// 後退解析その1 : 出次数0の局面を削除
			cout << "Retrograde Analysis : Step I   -> delete nodes with zero out-degree." << endl;
			u64 retro_counter1 = 0;
			if (next)
			{
				// 次に掘るsfenを探す時は、出次数0の局面を削除してしまうとleafまで到達できなくなるのでまずい。
				// この工程をskipする。
				cout << "..skip" << endl;

			} else {

				// 作業対象nodeが入っているqueue
				// このqueueは処理順は問題ではないので両端queueでなくて良いからvectorで実装しておく。
				vector<BookNodeIndex> queue;

				progress.reset((u64)book_nodes.size());

				// 出次数0のnodeをqueueに追加。
				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					BookNode& book_node = book_nodes[i];
					if (book_node.out_count == 0)
						queue.emplace_back((BookNodeIndex)i);
				}

				// 出次数0のnodeがなくなるまで繰り返す。
				while (queue.size())
				{
					progress.check(++retro_counter1);

					auto& index = queue[queue.size()-1];
					queue.pop_back();
					auto& book_node = book_nodes[index];

					ValueDepth parent_vd;
					size_t _;
					auto best = get_bestvalue(book_node , parent_vd , _);

					for(auto& parent_ki : book_node.parents)
					{
						auto& parent     = book_nodes[parent_ki.parent];
						auto& parent_move_index = parent_ki.move_index;

						auto& m = parent.moves[parent_move_index];
						m.vd = parent_vd;
						// →　出自数0なのでdepthがBOOK_DEPTH_INFであることは無いからそのチェックは不要。

						m.next  = BookNodeIndexNull; // この指し手はもう辿る必要がないのでここがnodeがleaf nodeに変わったことにしておく。

						// parentの出次数が1減る。parentの出次数が0になったなら、処理対象としてqueueに追加。
						if (--parent.out_count == 0)
							queue.emplace_back(index);
					}

					// 元のnodeの入次数 = 0にするためparentsをクリア。(もう親を辿ることはない)
					book_node.parents.clear();
				}
				// progress barを100%にする。
				progress.check(book_nodes.size());

				// 処理したノード数
				//cout << "processed nodes  : " << counter << endl;

			}

			// 出次数0以外の残りのnodeの数をカウント。
			// これらは何らかloopしているということなのできちんとしたmin-max探索が必要となる。
			u64 retro_counter2 = 0;
			if (next)
				retro_counter2 = book_nodes.size();
			else
			{
				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					if (book_nodes[i].out_count !=0)
						retro_counter2++;
				}
			}

			// 後退解析その2 : すべてのノードの親に評価値を伝播。MAX_PLY回行われる。

			cout << "Retrograde Analysis : Step II  -> Propagate the eval to the parents of all nodes." << endl;

			// ループがあって処理できていないnodeの数。
			//cout << "loop nodes       : " << counter << endl;

			// これらのnodeに対してはmin-max探索を行わなければならない。
			// ただし、min-maxだと組み合わせ爆発するので、non leafな指し手の値を
			// draw_valueで初期化して、全ノードに対してMAX_PLY回だけparentに評価値を伝播する。
			// これでループから抜け出せないところはdraw_valueになり、そうでないところは、正しい値が伝播されるということである。

			progress.reset(counter * MAX_PLY);
			counter = 0;


			// MAX_PLY回だけ評価値を伝播させる。
			for(size_t loop = 0 ; loop < MAX_PLY ; ++loop)
			{
				// 1つのノードでもupdateされたのか？
				bool updated = false;

				for(auto& book_node : book_nodes)
				{
					// 入次数1以上のnodeなのでparentがいるから、このnodeの情報をparentに伝播。
					// これがmin-maxの代わりとなる。
					if (book_node.parents.size() != 0)
					{
						ValueDepth parent_vd;
						size_t _;
						auto best = get_bestvalue(book_node , parent_vd, _);

						// 親nodeをupdateするのか。
						bool update_parent = false;

						// valueかdepthが違う限り伝播し続けて良い。
						if (   book_node.lastParentVd != parent_vd )
						{
							book_node.lastParentVd = parent_vd;
							updated = true;
							update_parent = true;
						}

						if (update_parent)
						{
							// このnodeの評価が決まり、更新が確定したので、これをparentに伝達する。
							// 子:親は1:Nだが、親のある指し手によって進める子は高々1つしかいないので
							// 子の評価値は、それぞれの親のこの局面に進む指し手の評価値としてそのまま伝播される。
							for(auto& parent : book_node.parents)
							{
								BookNodeIndex parent_index = parent.parent;
								BookNode&     parent_node  = book_nodes[parent_index];
								BookMove&     parent_move  = parent_node.moves[parent.move_index];

								// 親に伝播させている以上、前回のvalueの値と異なることは確定している。

								parent_move.vd = parent_vd;
							}
						}
					}

					progress.check(++counter);
				}

				// すべてのnodeがupdateされていないならこれ以上更新を繰り返しても仕方がない。
				if (!updated)
					break;
			}
			progress.check(counter * MAX_PLY);

			// 後退解析その3 : 

			// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
			// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスしておく。

			cout << "Retrograde Analysis : step III -> Adjust the bestvalue at all nodes." << endl;

			if (next)
			{
				cout << "..skip" << endl;
			} else {
				progress.reset(book_nodes.size());
				for(size_t i = 0 ; i < book_nodes.size() ; ++i)
				{
					auto& book_node = book_nodes[i];
					adjust_second_bestvalue(book_node);

					progress.check(i);
				}
				progress.check(book_nodes.size());
			}

			// 書き出したsfenの個数
			u64 write_counter = 0;

			if (next)
			{
				// 書き出すsfen
				unordered_set<string> write_sfens;

				if (low_depth)
				{
					// bestmoveでdepth == 0の指し手(で進めた局面)だけ書き出す。

					// 次に探索すべき定跡局面についてsfenを書き出していく。
					// これはmin-max探索した時のPVのleaf node。
					cout << "Low Depth           : step IV  -> pick up next sfens to search." << endl;

					progress.reset(sfen_to_index.size());

					size_t i = 0;
					for(auto&sfen_index : sfen_to_index)
					{
						auto& sfen  = sfen_index.first;
						auto& index = sfen_index.second;
						auto& book_node = book_nodes[index];

						auto& moves = book_node.moves;
						if (moves.size() == 0)
							continue;

						// 一つでもdepthが0ではない
						bool depth_not_zero = false;

						// best valueのmoveを探す。
						// ※ valueでsortされてないことに注意。
						size_t best_index = 0;
						int best_value = int_min;
						for(size_t j = 0 ; j < moves.size() ; j++)
						{
							auto move = moves[j];
							if (best_value < move.vd.value)
							{
								best_value = move.vd.value;
								best_index = j;
							}
							auto depth = move.vd.depth;
							depth_not_zero |= depth > 0; 
						}
						// best moveのdepth == 0
						auto best_move = moves[best_index];
						if (depth_not_zero && best_move.vd.depth == 0)
						{
							// bestmoveのnextはないことはわかっている。(depth == 0なので)
							Move move = best_move.move;
							if (is_ok(move))
							{
								StateInfo si, si2;
								pos.set(sfen, &si ,Threads.main());
								pos.do_move(move, si2);
								auto write_sfen = pos.sfen();
								write_sfens.emplace(write_sfen);
							}
						}

						progress.check(i++);
					}

				} else {

					// 次に探索すべき定跡局面についてsfenを書き出していく。
					// これはmin-max探索した時のPVのleaf node。
					cout << "Retrograde Analysis : step IV  -> pick up next sfens to search." << endl;

					// rootから辿っていきPV leafに到達したらそのsfenを書き出す。
					// そのPV leaf nodeを削除して後退解析により、各局面の評価値を更新する。
					// これを繰り返す。

					progress.reset(next_nodes);

					// それぞれのroot_sfenに対して。
					for(auto root_sfen : root_sfens)
					{
						u64 timeup_counter = 0;

						// 所定の行数のsfenを書き出すまで回る。
						// ただし、局面が尽きることがあるのでrootが存在しなければループは抜ける。
						while (true)
						{
							deque<StateInfo> si;
							BookTools::feed_position_string(pos, root_sfen, si);

							progress.check(write_sfens.size());

							// 規定の行数を書き出した or rootの局面が定跡DB上に存在しない
							if (write_sfens.size() >= next_nodes
								|| hashkey_to_index.count(pos.state()->hash_key()) == 0)
								break;

							// 書き出す指し手がすべてのrootでなくなっていることを証明するロジックを書くのわりと面倒なので
							// timeup_counterをカウントすることにする。
							// next_nodesの10倍も回ったらもうあかんやろ…。
							if (++timeup_counter > next_nodes * 10)
								break;

							// PVを辿った時の最後のParentMove
							ParentMove last_parent_move(BookNodeIndexNull,0);

							// leafの局面までの手順
							//string sfen_path = "startpos moves ";


							// まずPV leafまで辿る。
							while (true)
							{
								// 千日手がPVになってる。
								if (pos.is_repetition(MAX_PLY) == REPETITION_DRAW || pos.game_ply() >= MAX_PLY )
								{
									// たまに循環がひたすら回避されながらMAX_PLYを超えて手数が増えることがあるのでgame_ply()の判定必須。
									// ここ、切断してはいけないところが切断される可能性があるか…。まあ仕方ないな…。

									//if (last_parent_move.parent == BookNodeIndexNull)
									//	goto NEXT_ROOT;

									// この局面に至る指し手を定跡DBから除外することにより千日手にならないようにする。
									goto AVOID_REPETITION;
								}

								auto hash_key = pos.state()->hash_key();
								if (hashkey_to_index.count(hash_key) == 0)
								{
									// 局面が定跡DBの範囲から外れた。この局面は定跡未探索。このsfenを書き出しておく。
									write_sfens.emplace(pos.sfen());
									break;

									// このnodeへのleaf node move(一つ前のnodeがleaf nodeであるはずだから、そのleaf nodeからこの局面へ至る指し手)を削除する。
									// そのあと、そこから後退解析のようなことをしてrootに評価値を伝播する。
								} else {
									// bestmoveを辿っていく。
									BookNodeIndex index = hashkey_to_index[hash_key];
									auto& book_node = book_nodes[index];
									auto& moves     = book_node.moves;
									// movesが0の局面は定跡から除外されているはずなのだが…。
									ASSERT_LV3(moves.size());

									// 指し手のなかでbestを選ぶ。同じvalueならdepthが最小であること。
									last_parent_move = ParentMove(index,0);

									/*
									// →　BOOK_VALUE_NONE、書き出すのやめることにする。これあまりいいアイデアではなかった。
									for(auto& move : moves)
									{
										if (move.vd.value == BOOK_VALUE_NONE)
										{
											// 評価値未確定のやつ。これは値によってはこれが即PVになるのでここも探索すべき。
											// またvalueがBOOK_VALUE_NONEであるということは子からの評価値の伝播がなかったということだから、
											// この指し手で進めてもbook_nodesに格納されている局面には進行しないことは保証されている。
											si.emplace_back(StateInfo());
											pos.do_move(move.move,si.back());
											write_sfens.emplace(pos.sfen());

											pos.undo_move(move.move);

											// 書き出したのでこの枝は死んだことにする。
											move.move = MOVE_NONE;
										}
									}
									*/

									ValueDepth parent_vd;
									size_t i;
									auto best = get_bestvalue(book_node, parent_vd , i);
									// このbestのやつのindexが必要なので何番目にあるか調べる
									last_parent_move.move_index = (u32)i;

									//sfen_path += to_usi_string(best.move) + ' ';

									si.emplace_back(StateInfo());
									pos.do_move(moves[i].move, si.back());
								}
							}

							// 手順もデバッグ用に書き出す。
							//write_sfens.emplace_back(sfen_path);

							// PV leaf nodeまで到達したので、ここからrootまで遡ってbest move,best valueの更新を行う。
							// rootまで遡る時にparentsが複数あったりloopがあったりするので単純に遡ると組み合わせ爆発を起こす。
							// そこで、次のアルゴリズムを用いる。

							/*
								queue = [処理すべき局面]
								while queue:
									p = queue.pop_left()
									b = pの指し手がすべてMOVE_NONE(無効)になったのか
									v = 局面pのbestvalue
									for parent in 局面p.parents():
										if b:
											parentの局面pに行く指し手 = MOVE_NONE
										else:
											parentの局面pにいく指し手の評価値 = v

										if ↑この代入によりこのparent nodeのbestvalueが変化したなら
											queue.push_right(p)
							*/
							// 上記のアルゴリズムで停止すると思うのだが、この停止性の証明ができていない。
							// 永久ループになることがある。同じ局面は2回更新しないようにする。

						AVOID_REPETITION:;

							deque<ParentMoveEx> queue;
							queue.emplace_back(ParentMoveEx(last_parent_move,true,ValueDepth()));

							// update済みノード
							unordered_set<BookNodeIndex> already_searched_node;

							while (queue.size())
							{
								auto pm = queue[0];
								queue.pop_front();

								auto index      = pm.parent_move.parent;
								auto move_index = pm.parent_move.move_index;
								bool is_deleted = pm.is_deleted;
								auto& book_node = book_nodes[index];

								// 親がいなくなっている = bestmoveを求めても伝播する先かない = このnodeは死んでいる。
								// →　ただし、rootの可能性があるので、このノードの処理は行う必要がある。
								//if (book_node.parents.size()==0)
								//	continue;

								// 子の局面の指し手がすべてMOVE_NONEなので、子に至るこの指し手を無効化する。
								// これでbest_valueに変化が生じるのか？
								if (is_deleted)
								{
									book_node.moves[move_index].move = MOVE_NONE;
									// これによりすべてがMOVE_NONEになったか？
									for(auto const& move : book_node.moves)
										is_deleted &= (move.move == MOVE_NONE);
								}
								else
								{
									// 子からの評価値を伝播させる。
									// これによって、このnodeの指し手がすべてMOVE_NONEになることはないから
									// そのチェックは端折ることができる。
									book_node.moves[move_index].vd = pm.best;
								}

								if (book_node.parents.size()==0)
									continue;

								// 親に伝播させる。

								if (is_deleted)
								{
									// このnodeのすべて枝が死んだのでこのnodeは消滅させる。
									hashkey_to_index.erase(book_node.key);

									// 親に伝播させる。
									for(auto& parent_move : book_node.parents)
										queue.emplace_back(parent_move, is_deleted, ValueDepth());

									// deleteしているので、deleteが二重になされることはないから、
									// これによって永久ループになることはない。ゆえにこれは2回 parent nodeをupdateしても問題ない。

									// ここからparentへの経路はここで絶たれたことにしておかないと
									// 死んでいるノードだけが循環しているケースでwhileが終了しない。
									book_node.parents.clear();

								} else {

									ValueDepth parent_vd;
									size_t _;
									auto best = get_bestvalue(book_node , parent_vd, _);

									// 異なる値へのupdateの時だけ親に伝播させる。
									if (   parent_vd != book_node.lastParentVd )
									{
										// 次回用にlastParentVdを更新しておく。
										book_node.lastParentVd = parent_vd;

										// 親に伝播させる。
										for(auto& parent_move : book_node.parents)
										{
											// すでに一度updateしてあるならスキップする
											BookNodeIndex parent = parent_move.parent;
											if (already_searched_node.count(parent) > 0)
												continue;
											// これは一度updateしたのでこれ以上追加しないようにしておく。
											already_searched_node.emplace(parent);

											queue.emplace_back(parent_move, is_deleted, parent_vd);
										}
									}
								}
							}

						}

					//NEXT_ROOT:;

					}
				}

				progress.check(next_nodes);

				// write_sfensのなかにある局面とそれをflipした局面の組が含まれないかを
				// チェックする。
				// flipした局面に対しても辿っているので、これはわりとありうる。

				SystemIO::TextWriter writer;
				writer.Open(writebook_path);
				for(auto& write_sfen : write_sfens)
				{
					StateInfo si;
					pos.set(write_sfen, &si, Threads.main());
					auto key = pos.hash_key();
					// 既出ならskip
					if (hashkey_to_index.count(key) > 0)
						continue;
					// 登録しておく。
					hashkey_to_index[key] = -1;

					// flipした局面に対してもhitするか調べる。
					auto flipped_sfen = pos.flipped_sfen();
					pos.set(flipped_sfen, &si, Threads.main());
					key = pos.hash_key();
					if (hashkey_to_index.count(key) > 0)
						continue;
					//hashkey_to_index[key] = -1;
					// ⇨　flipした局面は登録しなくとも、元の局面に対してhitするから問題ない。
					
					writer.WriteLine(write_sfen);
					++write_counter;
				}

			} else {

				// メモリ上の定跡DBを再構成。
				cout << "Rebuild MemoryBook  : " << endl;
				progress.reset(sfen_to_index.size());
				counter = 0;

				// これはメモリ上にまずBook classを用いて定跡DBを構築して、それを書き出すのが間違いがないと思う。
				Book::MemoryBook new_book;
				for(auto&sfen_index : sfen_to_index)
				{
					auto& sfen  = sfen_index.first;
					auto& index = sfen_index.second;
					auto& book_node = book_nodes[index];

					// FlippedBookがtrueなら、後手番の局面は書き出さない。
					if (flipped_book && book_node.color == WHITE)
						continue;

					Book::BookMoves bookMoves;
					for(auto& move : book_node.moves)
					{
						Book::BookMove bookMove(move.move,0/*ponder move*/,move.vd.value,move.vd.depth ,0/* move_count */);
						bookMoves.push_back(bookMove);
					}
					shared_ptr<Book::BookMoves> bookMovesPtr(new Book::BookMoves(bookMoves));
					new_book.append(sfen, bookMovesPtr);

					progress.check(++counter);
				}
				progress.check(sfen_to_index.size());

				// 定跡ファイルの書き出し
				new_book.write_book(writebook_path);
				write_counter = new_book.size();
			}

			cout << "[ PetaShock Result ]" << endl;

			cout << "flipped counter  : " << flipped_counter << endl;

			// 合流チェックによって合流させた指し手の数。
			if (!next)
				cout << "converged_moves  : " << converged_moves << endl;

			// 後退解析において判明した、leafから見てループではなかったノード数
			cout << "retro_counter1   : " << retro_counter1 << endl;
			// 後退解析において判明した、ループだったノード数
			cout << "retro_counter2   : " << retro_counter2 << endl;

			// 書き出したrecord数
			if (next)
			{
				cout << "write sfen nodes : " << write_counter << endl << endl;
				cout << "Making peta-shock next sfens has been completed." << endl;
			}
			else
			{
				cout << "write book nodes : " << write_counter << endl << endl;
				cout << "Making a peta-shock book has been completed." << endl;
			}

		}

	private:

		// 定跡本体
		vector<BookNode> book_nodes;

		// sfen文字列からBookMoveIndexへのmapper
		// this->book_nodesの何番目の要素であるかが返る。
		// sfen文字列は先頭の"sfen "と、末尾の手数は省略されているものとする。
		unordered_map<string,BookNodeIndex> sfen_to_index;

		// 同様に、HASH_KEYからBookMoveIndexへのmapper
		unordered_map<HASH_KEY,BookNodeIndex> hashkey_to_index;
	};
}

namespace Book
{
	// 2023年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2023(Position& pos, istringstream& is, const string& token)
	{
		if (token == "peta_shock")
		{
			// ペタショックコマンド
			// やねうら王の定跡ファイルに対して定跡ツリー上でmin-max探索を行い、その結果を別の定跡ファイルに書き出す。
			//   makebook peta_shock book.db user_book1.db
			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is, false);
			return 1;

		} else if (token == "peta_shock_next"){

			// ペタショックnext
			// ペタショック手法と組み合わせてmin-maxして、有望な局面をsfen形式でテキストファイルに書き出す。
			//   makebook peta_shock_next book.db sfens.txt 1000
			//   makebook peta_shock_next book.db sfens.txt 1000 eval_noise 20
			// ⇨　1000局面を書き出す。20はleaf nodeの指し手の評価値に加える乱数の大きさ。
			// 　 この場合、評価値に、平均 = 0 , 標準偏差 = 20 ガウスノイズを加算する。
			// 　　(これを加えることで序盤の指し手を開拓しやすくなる)
			//   makebook peta_shock_next book.db sfens.txt 1000 low_depth
			// ⇨  leaf nodeで2番目の指し手はdepth > 0なのに1番目の指し手のdepth == 0だと、
			// これは掘らないとおかしいので、こういう局面を探して列挙するモード。
			// (これ以外の局面は列挙しない。列挙する局面数は制限なし。)

			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is , true);
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)
