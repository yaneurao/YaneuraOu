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

#include <sstream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <limits>
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


	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	struct BookMove
	{
		// moveの指し手がleafである場合。
		BookMove::BookMove(Move move,int value,int depth):
			move(move),value(value),depth(depth),next(BookNodeIndexNull){}

		// moveの指し手がleafではない場合。
		BookMove::BookMove(Move move,int value,int depth, BookNodeIndex next):
			move(move),value(value),depth(depth),next(next){}

		// move(4) + value(4) + depth(4) + next(4) = 16 bytes

		// 指し手
		Move move;

		// ↑の指し手を選んだ時の定跡ツリー上の評価値
		// (定跡ツリー上でmin-max探索をした時の評価値)
		int value;

		// 探索深さ。
		int depth;

		// ↑のmoveで局面を進めた時の次の局面のBookMoveポインター。
		// これが存在しない時は、BookMoveIndexNull。
		BookNodeIndex next;
	};

	// BoonNodeの評価値で∞を表現する定数。
	const int BOOK_VALUE_INF = numeric_limits<int>::max();

	// ペタショック前の定跡DBに指し手の評価値を99999にして書き出しておくと、
	// これは指し手は存在するけど評価値は不明の指し手である。(という約束にする)
	// これは棋譜の指し手などを定跡DBに登録する時に評価値が確定しないのでそういう時に用いる。
	const int BOOK_VALUE_NONE = -99999;

	// あるnodeに対してその親nodeとその何番目の指し手がこのnodeに接続されているのかを保持する構造体。
	struct ParentMove
	{
		ParentMove::ParentMove(BookNodeIndex parent,size_t move_index):parent(parent),move_index((u32)move_index){}

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

		// 後退解析IIの時の前回のbestValue。全ノード、これが前回と変わっていないなら後退解析を終了できる。
		int lastBestValue = BOOK_VALUE_INF;

		// 指し手
		vector<BookMove> moves;

		// key(この局面からhash keyを逆引きしたい時に必要になるので仕方なく追加してある)
		HASH_KEY key;
	};

	// 定跡の評価値とその時のdepthをひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0) {}

		ValueDepth(int value, int depth)
			: value(value) , depth(depth){}

		int value;
		int depth;
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

			string readbook_path;
			string writebook_path;
			string root_sfens_path ;

			// 次の思考対象とすべきsfenを書き出す時のその局面の数。
			u64 next_nodes = 0;
			if (next)
				is >> next_nodes;

			is >> readbook_path >> writebook_path;

			readbook_path  = Path::Combine("book",readbook_path );
			writebook_path = Path::Combine("book",writebook_path);

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			if (next)
			{
				// 書き出すsfenの数
				cout << "write next_sfens : " << next_nodes << endl;

				// これは現状ファイル名固定でいいや。
				root_sfens_path = Path::Combine("book","root_sfens.txt");
				cout << "root_sfens_path  : " << root_sfens_path << endl;
			}

			cout << "readbook_path    : " << readbook_path  << endl;
			cout << "writebook_path   : " << writebook_path << endl;

			// 引き分けのスコアを変更したいなら先に変更しておいて欲しい。
			cout << "draw_value black : " << draw_value(REPETITION_DRAW,BLACK) << endl;
			cout << "draw_value white : " << draw_value(REPETITION_DRAW,WHITE) << endl;

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

			cout << "Register SFENs      : " << endl;

			// memo : 指し手の存在しない局面はそんな定跡ファイル読み込ませていないか、
			//        あるいはMemoryBookが排除してくれていると仮定している。

			// 局面数などをカウントするのに用いるカウンター。
			u64 counter = 0;

			Tools::ProgressBar progress(book.size());

			// まず、出現する局面すべてのsfenに対して、それをsfen_to_hashkeyに登録する。
			// sfen文字列の末尾に手数が付与されているなら、それを除外する。→ IgnoreBookPly = trueなので除外されている。
			book.foreach([&](string sfen,Book::BookMovesPtr& book_moves){

				// unordered_mapは同じキーに対して複数の値を登録できる便利構造なので、
				// 同じ値のキーが2つ以上登録されていないかをチェックしておく。
				if (this->sfen_to_index.count(sfen) > 0)
				{
					cout << "Error! : Hash Conflict! Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
					Tools::exit();
				}

				// 局面ひとつ登録する。
				BookNodeIndex index = (BookNodeIndex)this->book_nodes.size();
				this->book_nodes.emplace_back(BookNode());

				StateInfo si,si2;
				pos.set(sfen,&si,Threads.main());
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

			book.foreach([&](string sfen,Book::BookMovesPtr& book_moves){

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

						// 元のnodeの出次数をと、next_nodeへの入次数をインクリメントしてやる。
						// (後退解析みたいなことをしたいので)
						book_node.out_count++;
						BookNodeIndex next_book_node_index = this->hashkey_to_index[next_hash];
						BookNode&     next_book_node       = this->book_nodes[next_book_node_index];

						// parentのlistに、元のnodeを追加しておく。
						next_book_node.parents.emplace_back(ParentMove(index,book_node.moves.size()));

						// どうせmin-maxして、ここの評価値とdepthは上書きされるから何でも良い。
						BookMove book_move(move,0,0,next_book_node_index);
						book_node.moves.emplace_back(book_move);
						converged_moves++;
					}

					pos.undo_move(move);
				}

				// 定跡DB上のこの局面の指し手も登録しておく。
				book_moves->foreach([&](Book::BookMove& bm){
						Move move = pos.to_move(bm.move);
						int depth = bm.depth;
						int value = bm.value;

						// これがbook_nodeにすでに登録されているか？
						if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
						{
							// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので登録しなくて良い。)
							// 登録されていなかったということは、ここから接続されているnodeはないので、出次数には影響を与えない。
							BookMove book_move(move,value,depth);
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

					// この局面の評価値のうち、一番いいやつを親に伝播する
					int best_value = numeric_limits<int>::min();
					int best_depth = 0;
					for(auto& book_move : book_node.moves)
					{
						// 値が同じならdepthの低いほうを採用する。
						// 同じvalueの局面ならば最短で行けたほうが得であるから。
						if (best_value == book_move.value)
							best_depth = std::min(best_depth , book_move.depth);
						else if (best_value < book_move.value)
						{
							best_value = book_move.value;
							best_depth = book_move.depth;
						}
					}

					for(auto& parent_ki : book_node.parents)
					{
						auto& parent     = book_nodes[parent_ki.parent];
						auto& parent_move_index = parent_ki.move_index;

						auto& m = parent.moves[parent_move_index];
						m.value = -best_value;            // min-maxなので符号を反転
						m.depth =  best_depth + 1;        // depthは一つ深くなった
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
			for(size_t i = 0 ; i < book_nodes.size() ; i++)
			{
				if (book_nodes[i].out_count !=0)
					retro_counter2++;
			}

			// 後退解析その2 : すべてのノードの親に評価値を伝播。MAX_PLY回行われる。

			cout << "Retrograde Analysis : Step II  -> Propagate the eval to the parents of all nodes." << endl;

			// ループがあって処理できていないnodeの数。
			//cout << "loop nodes       : " << counter << endl;

			// これらのnodeに対してはmin-max探索を行わなければならない。
			// ただし、min-maxだと組み合わせ爆発するので、non leafな指し手の値を
			// draw_valueで初期化して、全ノードに対してMAX_PLY回だけparentに評価値を伝播する。
			// これでループから抜け出せないところはdraw_valueになり、そうでないところは、正しい値が伝播されるということである。

			// まずnon leaf moveのvalueをdraw valueで初期化。
			// non leaf moveが存在するならnodeの出次数 > 0であるから、これを先に判定して良い。
			for(auto& book_node : book_nodes)
				if (book_node.out_count != 0)
					for(auto& book_move : book_node.moves)
						if (book_move.next != BookNodeIndexNull)
						{
							book_move.depth = 1;
							book_move.value = draw_value(REPETITION_DRAW, book_node.color);
						}

			progress.reset(counter * MAX_PLY);
			counter = 0;

			// あるnodeのbest_valueとbest_depthを得るヘルパー関数。
			auto get_bestvalue = [&](BookNode& node)
			{
				// まずこのnodeの結論を得る。
				ValueDepth best(numeric_limits<int>::min(),0);
						
				for(auto& book_move:node.moves)
				{
					// MOVE_NONEならこの枝はないものとして扱う。
					if (book_move.move == MOVE_NONE)
						continue;

					// 値が同じならdepthの低いほうを採用する。
					// なぜなら、循環してきて、別の枝に進むことがあり、それはdepthが高いはずであるから。
					if (best.value == book_move.value)
						best.depth = std::min(best.depth , book_move.depth);
					else if (best.value < book_move.value)
					{
						// BookNodeには、必ずINT_MINより大きなvalueを持つ指し手が一つは存在するので、
						// そのdepthで上書きされることは保証されている。
						best = ValueDepth(book_move.value, book_move.depth);
					}
				}
				return best;
			};

			// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
			// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスする処理を行うヘルパー関数。
			auto adjust_second_bestvalue = [&](BookNode& node)
			{
				auto best = get_bestvalue(node);

				for(auto& book_move : node.moves)
				{
					if (   best.value == book_move.value
						&& best.depth  < book_move.depth
						)
						// depthが最小でない指し手の評価値を1だけ減らしておく。

						// bestの値しか親には伝播しないので、ここで引いたところで
						// このnodeにしか影響はない。
						book_move.value--;
				}
			};

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
						auto best = get_bestvalue(book_node);

						if (book_node.lastBestValue != best.value)
						{
							book_node.lastBestValue = best.value;
							updated = true;
						}

						// このnodeの評価が決まったので、これをparentに伝達する。
						// 子:親は1:Nだが、親のある指し手によって進める子は高々1つしかいないので
						// 子の評価値は、それぞれの親のこの局面に進む指し手の評価値としてそのまま伝播される。
						for(auto& parent : book_node.parents)
						{
							BookNodeIndex parent_index = parent.parent;
							BookNode&     parent_node  = book_nodes[parent_index];
							BookMove& my_parent_move   = parent_node.moves[parent.move_index];

							my_parent_move.value = - best.value;
							my_parent_move.depth =   best.depth + 1;
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

			u64 write_counter = 0;

			if (next)
			{
				// 次に探索すべき定跡局面についてsfenを書き出していく。
				// これはmin-max探索した時のPVのleaf node。
				cout << "Retrograde Analysis : step IV  -> pick up next sfens to search." << endl;

				// rootから辿っていきPV leafに到達したらそのsfenを書き出す。
				// そのPV leaf nodeを削除して後退解析により、各局面の評価値を更新する。
				// これを繰り返す。

				// 定跡の開始局面。
				// "book/root_sfens.txt"からrootのsfen集合を読み込むことにする。
				// これはUSIのposition文字列の"position"を省略した文字列であること。
				// つまり、
				//   startpos
				//   startpos moves ...
				//   sfen ...
				// のような文字列である。
				vector<string> root_sfens;
				if (SystemIO::ReadAllLines(root_sfens_path,root_sfens).is_not_ok())
					root_sfens.emplace_back(BookTools::get_start_sfens()[0]);

				progress.reset(next_nodes);

				// 書き出すsfen
				vector<string> write_sfens;

				// それぞれのroot_sfenに対して。
				for(auto root_sfen : root_sfens)
				{
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
								// たまに循環がひたすら回避されながら無限に手数が増えることがあるのでgame_ply()の判定必須。
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
								write_sfens.emplace_back(pos.sfen());
								break;

								// このnodeへのleaf node move(一つ前のnodeがleaf nodeであるはずだから、そのleaf nodeからこの局面へ至る指し手)を削除する。
								// そのあと、そこから後退解析のようなことをしてrootに評価値を伝播する。
							} else {
								// bestmoveを辿っていく。
								BookNodeIndex index = hashkey_to_index[hash_key];
								auto& moves = book_nodes[index].moves;
								// movesが0の局面は定跡から除外されているはずなのだが…。
								ASSERT_LV3(moves.size());

								// 指し手のなかでbestを選ぶ。同じvalueならdepthが最小であること。
								BookMove best = BookMove(MOVE_NONE,-BOOK_VALUE_INF,0);
								last_parent_move = ParentMove(index,0);

								for(size_t i = 0 ; i < moves.size() ; ++i)
								{
									auto& move = moves[i];

									// MOVE_NONEは無視する。これは死んでる枝。
									if ( move.move == MOVE_NONE)
										continue;

									if (move.value == BOOK_VALUE_NONE)
									{
										// 評価値未確定のやつ。これは値によってはこれが即PVになるのでここも探索すべき。
										si.emplace_back(StateInfo());
										pos.do_move(move.move,si.back());
										write_sfens.emplace_back(pos.sfen());
										pos.undo_move(move.move);

										// 書き出したのでこの枝は死んだことにする。
										move.move = MOVE_NONE;
										continue;
									}

									// bestを更新するか。
									if (    move.value > best.value
										|| (move.value == best.value && move.depth < best.depth)
										)
									{
										best = move;
										// この指し手でdo_moveすることになりそうなのでmove_indexを記録しておく。
										last_parent_move.move_index = (u32)i;
									}
								}

								//sfen_path += to_usi_string(best.move) + ' ';

								si.emplace_back(StateInfo());
								pos.do_move(best.move, si.back());
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
						// もしかして評価値が振動して永遠に終わらないかも？
						// →　であるなら、同じ局面についてはqueueにX回以上pushしないみたいな制限をしてやる必要がある。

					AVOID_REPETITION:;

						deque<ParentMoveEx> queue;
						queue.emplace_back(ParentMoveEx(last_parent_move,true,ValueDepth()));

						while (queue.size())
						{
							auto pm = queue[0];
							queue.pop_front();

							auto index      = pm.parent_move.parent;
							auto move_index = pm.parent_move.move_index;
							bool is_deleted = pm.is_deleted;
							auto& book_node = book_nodes[index];

							// 子の局面の指し手がすべてMOVE_NONEなので、子に至るこの指し手を無効化する。
							// これでbest_valueに変化が生じるのか？
							if (is_deleted)
							{
								book_node.moves[move_index].move  =   MOVE_NONE;
								// これによりすべてがMOVE_NONEになったか？
								for(auto move : book_node.moves)
									is_deleted &= (move.move == MOVE_NONE);
							}
							else
							{
								// 子からの評価値を伝播させる。
								// これによって、このnodeの指し手がすべてMOVE_NONEになることはないから
								// そのチェックは端折ることができる。
								book_node.moves[move_index].value = pm.best.value;
								book_node.moves[move_index].depth = pm.best.depth;
							}

							// 親に伝播させる。

							if (is_deleted)
							{
								// このnodeのすべて枝が死んだのでこのnodeは消滅させる。
								hashkey_to_index.erase(book_node.key);

								// 親に伝播させる。
								for(auto& parent_move : book_node.parents)
									queue.emplace_back(parent_move, is_deleted, ValueDepth());

								// ここからparentへの経路はここで絶たれたことにしておかないと
								// 死んでいるノードだけが循環しているケースでwhileが終了しない。
								book_node.parents.clear();

							} else {

								auto& best = get_bestvalue(book_node);

								if (best.value != book_node.lastBestValue)
								{
									// 次回用にlastBestValueを更新しておく。
									book_node.lastBestValue = best.value;

									// 親に伝播するのでvalueの符号はこの時点で反転させておく。
									best.value = - best.value;
									best.depth =   best.depth + 1;

									// 親に伝播させる。
									for(auto& parent_move : book_node.parents)
										queue.emplace_back(parent_move, is_deleted, best);
								}
							}

						}
					}

				//NEXT_ROOT:;

				}

				progress.check(next_nodes);

				SystemIO::WriteAllLines(writebook_path, write_sfens);
				write_counter = write_sfens.size();

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

					Book::BookMoves bookMoves;
					for(auto& move : book_node.moves)
					{
						Book::BookMove bookMove(move.move,0/*ponder move*/,move.value,move.depth,0/* move_count */);
						bookMoves.push_back(bookMove);
					}
					shared_ptr<Book::BookMoves> bookMovesPtr(new Book::BookMoves(bookMoves));
					new_book.append(sfen, bookMovesPtr);

					progress.check(++counter);
				}
				// 定跡ファイルの書き出し
				new_book.write_book(writebook_path);
				write_counter = new_book.size();
			}

			cout << "[ PetaShock Result ]" << endl;
			// 合流チェックによって合流させた指し手の数。
			if (!next)
			{
				cout << "converged_moves  : " << converged_moves << endl;
			}
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
			//   makebook peta_shock_next 1000 book.db sfens.txt
			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is , true);
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)
