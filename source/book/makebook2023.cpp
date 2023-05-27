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
		BookMove::BookMove(Move move,int value,int depth,BookNodeIndex next):
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

		// 指し手
		vector<BookMove> moves;
	};

	// 評価値とdepthをひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth(int value, int depth)
			:value(value),depth(depth){}

		// 評価値
		int value;

		// 探索深さ
		int depth;
	};

	// ペタショック化
	class PetaShock
	{
	public:
		void make_book(Position& pos , istringstream& is)
		{
			cout << "peta-shock command start : " << endl;
			cout << "================================================================================" << endl;

			// 高速化のために、HASH_KEYだけで局面を処理したいので、hashの衝突はあってはならない。
			// そのため、HASH_KEY_BITSは128か256が望ましい。
			if (HASH_KEY_BITS < 128)
			{
				cout << "WARNING! : HASH_KEY_BITS = " << HASH_KEY_BITS << " is too short." << endl;
				cout << "    Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
			}

			string readbook_name;
			string writebook_name;
			is >> readbook_name >> writebook_name;

			readbook_name  = Path::Combine("book",readbook_name );
			writebook_name = Path::Combine("book",writebook_name);

			cout << "readbook  : " << readbook_name  << endl;
			cout << "writebook : " << writebook_name << endl;

			// 引き分けのスコアを変更したいなら先に変更しておいて欲しい。
			cout << "draw_value black : " << draw_value(REPETITION_DRAW,BLACK) << endl;
			cout << "draw_value white : " << draw_value(REPETITION_DRAW,WHITE) << endl;

			// 手数無視のオプションを有効にすると、MemoryBook::read_book()は、
			// sfen文字列末尾の手数を無視してくれる。
			Options["IgnoreBookPly"] = true;
			// 定跡生成書き出したあとに普通に探索させることはありえないだろうから
			// 元のオプションへの復元は行わない。(行いたいならば、benchmark.cppを参考に)

			Book::MemoryBook book;
			if (book.read_book(readbook_name).is_not_ok())
			{
				cout << "read book error" << endl;
				return ;
			}

			cout << "register sfen       : " << endl;

			// memo : 指し手の存在しない局面はそんな定跡ファイル読み込ませていないか、
			//        あるいはMemoryBookが排除してくれていると仮定している。

			// 局面数などをカウントするのに用いるカウンター。
			u64 counter = 0;

			Tools::ProgressBar progress(book.size());

			// まず、出現する局面すべてのsfenに対して、それをsfen_to_hashkeyに登録する。
			book.foreach([&](string sfen,Book::BookMovesPtr& book_moves){

				// 局面ひとつ登録する。
				BookNodeIndex index = (BookNodeIndex)this->book_nodes.size();
				this->book_nodes.emplace_back(BookNode());

				StateInfo si,si2;
				pos.set(sfen,&si,Threads.main());
				auto key = pos.state()->hash_key();
				this->sfen_to_index[sfen] = index;
				this->hashkey_to_index[key] = index;

				// sfen文字列の末尾に手数が付与されているなら、それを除外する。

				// unordered_mapは同じキーに対して複数の値を登録できる便利構造なので、
				// 同じ値のキーが2つ以上登録されていないかをチェックしておく。
				if (this->sfen_to_index.count(sfen) > 1)
				{
					cout << "Error! : Hash Conflict! Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
					Tools::exit();
				}

				progress.check(++counter);
			});

			// 局面の合流チェック
			cout << "converge check      :" << endl;

			counter = 0;

			// 合流した指し手の数
			u64 converged_moves = 0;
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
				book_moves->foreach([&](Book::BookMove& book_move){
						Move move = pos.to_move(book_move.move);
						int depth = book_move.depth;
						int value = book_move.value;

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

			// 後退解析その1
			cout << "retrograde analysis : step I" << endl;

			// 作業対象nodeが入っているqueue
			// このqueueは処理順は問題ではないので両端queueでなくて良いからvectorで実装しておく。
			vector<BookNodeIndex> queue;
			u64 retro_counter1 = 0;

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

			// 出次数0以外の残りのnodeの数をカウント。
			// これらは何らかloopしているということなのできちんとしたmin-max探索が必要となる。
			u64 retro_counter2 = 0;
			for(size_t i = 0 ; i < book_nodes.size() ; i++)
			{
				if (book_nodes[i].out_count !=0)
					retro_counter2++;
			}

			// ループがあって処理できていないnodeの数。これはmin-max探索を行う。
			// ただし、min-maxだと組み合わせ爆発するので、non leafな指し手の値を
			// draw_valueで初期化して、全ノードに対してMAX_PLY回だけparentに評価値を伝播する。
			// これでループから抜け出せないところはdraw_valueになり、そうでないところは、正しい値が伝播されるということである。
			//cout << "loop nodes       : " << counter << endl;

			// 後退解析その2
			cout << "retrograde analysis : step II" << endl;

			// まずnon leaf moveのvalueをdraw valueで初期化。
			// non leaf moveが存在するなら出次数 > 0であるから、これを先に判定して良い。
			for(auto& book_node : book_nodes)
				if (book_node.out_count != 0)
					for(auto& book_move : book_node.moves)
						if (book_move.next != BookNodeIndexNull)
						{
							book_move.depth = 1;
							book_move.value = draw_value(REPETITION_DRAW,book_node.color);
						}

			progress.reset(counter * MAX_PLY);
			counter = 0;

			// MAX_PLY回だけ評価値を伝播させる。
			for(size_t loop = 0 ; loop < MAX_PLY ; ++loop)
			{
				for(auto& book_node : book_nodes)
				{
					// 入次数1以上のnodeなのでparentがいるから、このnodeの情報をparentに伝播。
					// これがmin-maxの代わりとなる。
					if (book_node.parents.size() != 0)
					{
						// まずこのnodeの結論を得る。
						int best_value = numeric_limits<int>::min();
						int best_depth = 0;
						
						for(auto& book_move:book_node.moves)
						{
							// 値が同じならdepthの低いほうを採用する。
							// なぜなら、循環してきて、別の枝に進むことがあり、それはdepthが高いはずであるから。
							if (best_value == book_move.value)
								best_depth = std::min(best_depth , book_move.depth);
							else if (best_value < book_move.value)
							{
								// BookNodeには、必ずINT_MINより大きなvalueを持つ指し手が一つは存在するので、
								// そのdepthで上書きされることは保証されている。
								best_value = book_move.value;
								best_depth = book_move.depth;
							}
						}

						// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
						// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスしておく。
						for(auto& book_move:book_node.moves)
							if (best_value == book_move.value && best_depth < book_move.depth)
								book_move.value --;

						// このnodeの評価が決まったので、これをparentに伝達する。
						// 子:親は1:Nだが、親のある指し手によって進める子は高々1つしかいないので
						// 子の評価値は、親のその指し手の評価値としてそのまま伝播される。
						for(auto& parent : book_node.parents)
						{
							BookNodeIndex parent_index = parent.parent;
							BookMove& parent_move = book_nodes[parent_index].moves[parent.move_index];
							parent_move.value = - best_value;
							parent_move.depth =   best_depth + 1;
						}

						progress.check(++counter);
					}
				}
			}
			progress.check(counter * MAX_PLY);

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
			}

			new_book.write_book(writebook_name);

			cout << "================================================================================" << endl;
			// 合流チェックによって合流させた指し手の数。
			cout << "converged_moves  : " << converged_moves << endl;
			// 後退解析において判明した、leafから見てループではなかったノード数
			cout << "retro_counter1   : " << retro_counter1 << endl;
			// 後退解析において判明した、ループだったノード数
			cout << "retro_counter2   : " << retro_counter2 << endl;
			cout << "write book nodes : " << new_book.size() << endl;
			cout << "peta-shock done." << endl;
		}

	private:
		u64 search_nodes;

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
	// 2019年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2023(Position& pos, istringstream& is, const string& token)
	{
		// makebook steraコマンド
		// スーパーテラショック定跡手法での定跡生成。
		if (token == "peta_shock")
		{
			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is);
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)
