#include "../config.h"

#if defined (ENABLE_MAKEBOOK_CMD)

#include "book.h"
#include "../position.h"
#include "../misc.h"
#include "../search.h"
#include "../thread.h"
#include "../tt.h"
#include "apery_book.h"

#include <sstream>
#include <unordered_set>
#include <iomanip>		// std::setprecision()
#include <numeric>      // std::accumulate()

using namespace std;
namespace YaneuraOu {
namespace Book {

	// 2015年ごろに作ったmakebookコマンド
	int makebook2015(Position& pos,istringstream& is,const std::string& token_)
	{
		// const外す
		string token = token_;

		// sfenから生成する
		bool from_sfen = token == "from_sfen";

		// 自ら思考して生成する
		// bool from_thinking = token == "think";
		// 思考するコマンド、LEARN版の削除に伴い無効化。
        const bool from_thinking = false;

		// 定跡のマージ
		bool book_merge = token == "merge";
		// 定跡のsort
		bool book_sort = token == "sort";
		// 定跡の変換
		bool convert_from_apery = token == "convert_from_apery";
		// 定跡の変換
		bool convert_to_apery = token == "convert_to_apery";
		
		// いずれのコマンドでもないなら、このtokenのコマンドを自分は処理できない。
		if (!(from_sfen || from_thinking || book_merge || book_sort || convert_from_apery || convert_to_apery))
			return 0;

		if (from_sfen || from_thinking)
		{
			// sfenファイル名
			is >> token;

			// 読み込むべきファイル名
			string sfen_file_name[COLOR_NB];

			// ここに "bw"(black and whiteの意味)と指定がある場合、
			// 先手局面用と後手局面用とのsfenファイルが異なるという意味。
			// つまり、このあとsfenファイル名の指定が2つ来ることを想定している。

			// 先後両方のsfenファイルを指定されているときはこのフラグをtrueに設定しておく。
			bool bw_files;
			if (token == "bw")
			{
				is >> sfen_file_name[BLACK];
				is >> sfen_file_name[WHITE];
				bw_files = true;
			}
			else {
				/*BLACKとWHITEと共通*/
				sfen_file_name[0] = token;
				bw_files = false;
			}

			// 定跡ファイル名
			// Option["BookFile"]ではなく、ここで指定したものが処理対象である。
			string book_name;
			is >> book_name;

			// 開始手数、終了手数、探索深さ、node数の指定
			int start_moves = 1;
			int moves = 16;
			int depth = 24;
			u64 nodes = 0;

			// 分散生成用。
			// 2/7なら、7個に分けて分散生成するときの2台目のPCの意味。
			// cluster 2 7
			// のように指定する。
			// 1/1なら、分散させない。
			// cluster 1 1
			// のように指定する。
			int cluster_id = 1;
			int cluster_num = 1;

			// "makebook think"コマンド時の自動保存の間隔。デフォルト15分。
			u64 book_save_interval = 15 * 60;

			while (true)
			{
				token = "";
				is >> token;
				if (token == "")
					break;
				if (token == "moves")
					is >> moves;
				else if (token == "depth")
					is >> depth;
				else if (token == "nodes")
					is >> nodes;
				else if (from_thinking && token == "startmoves")
					is >> start_moves;
				else if (from_thinking && token == "cluster")
					is >> cluster_id >> cluster_num;
				else if (from_thinking && token == "book_save_interval")
					is >> book_save_interval;
				else
				{
					cout << "Error! : Illigal token = " << token << endl;
					return 1;
				}
			}

			// 処理対象ファイル名の出力
			cout << "makebook think.." << endl;

			if (bw_files)
			{
				// 先後、個別の定跡ファイルを用いる場合
				cout << "sfen_file_name[BLACK] = " << sfen_file_name[BLACK] << endl;
				cout << "sfen_file_name[WHITE] = " << sfen_file_name[WHITE] << endl;
			}
			else {
				// 先後、同一の定跡ファイルを用いる場合
				cout << "sfen_file_name        = " << sfen_file_name[BLACK] << endl;
			}

			cout << "book_name             = " << book_name << endl;

			if (from_sfen)
				cout << "read sfen moves " << moves << endl;

			if (from_thinking)
				cout << "read sfen moves from " << start_moves << " to " << moves << endl
					 << " depth = " << depth << endl
					 << " nodes = " << nodes << endl
					 << " cluster = " << cluster_id << "/" << cluster_num << endl
					 << " book_save_interval = " << book_save_interval << endl;

			// 解析対象とするsfen集合。
			// 読み込むべきsfenファイル名が2つ指定されている時は、
			// 先手用と後手用の局面で個別のsfenファイルが指定されているということ。

			// Colorは、例えばBLACKが指定されていれば、ここで与えられるsfenの一連の局面は
			// 先手番のときのみ処理対象とする。
			typedef pair<string, Color> SfenAndColor;
			vector<SfenAndColor> sfens;

			if (! bw_files)
			{
				vector<string> tmp_sfens;
				SystemIO::ReadAllLines(sfen_file_name[0], tmp_sfens);

				// こちらは先後、どちらの手番でも解析対象とするのでCOLOR_NBを指定しておく。
				for (auto& sfen : tmp_sfens)
					sfens.push_back(SfenAndColor(sfen, COLOR_NB));
			}
			else
			{
				// sfenファイルを2つとも読み込み、手番を指定しておく。
				for (auto c : COLOR)
				{
					auto& filename = sfen_file_name[c];

					// ファイル名として"no_file"が指定されていれば、先手用 or 後手用のsfenはファイルは
					// 読み込まないという指定になる。
					if (filename == "no_file")
						continue;

					vector<string> tmp_sfens;
					SystemIO::ReadAllLines(filename, tmp_sfens);
					for (auto& sfen : tmp_sfens)
						sfens.push_back(SfenAndColor(sfen, c));
				}
			}

			cout << "..done" << endl;

			MemoryBook book;

			if (from_thinking)
			{
				cout << "read book.." << endl;
				// 初回はファイルがないので読み込みに失敗するが無視して続行。
				if (book.read_book(book_name).is_not_ok())
				{
					cout << "..failed but , create new file." << endl;
				}
				else
					cout << "..done" << endl;
			}

			cout << "parse..";

			// 思考すべき局面のsfen
			unordered_set<string> thinking_sfens;

			// 各行の局面をparseして読み込む(このときに重複除去も行なう)
			for (size_t k = 0; k < sfens.size(); ++k)
			{
				// sfenを取り出す(普通のsfen文字列とは限らない。"startpos"から書かれているかも)
				auto sfen = sfens[k].first;

				// ここで指定されている手番の局面しか処理対象とはしない。
				// ただしCOLOR_NBが指定されているときは、「希望する手番はない」の意味。
				auto color = sfens[k].second;

				if (sfen.length() == 0)
					continue;

				// issから次のtokenを取得する
				auto feed_next = [](istringstream& iss)
				{
					string token = "";
					iss >> token;
					return token;
				};

				// "sfen"に後続するsfen文字列をissからfeedする
				auto feed_sfen = [&feed_next](istringstream& iss)
				{
					stringstream sfen;

					// ループではないが条件外であるときにbreakでreturnのところに行くためのhack
					while(true)
					{
						string token;

						// 盤面を表すsfen文字列
						sfen << feed_next(iss);

						// 手番
						token = feed_next(iss);
						if (token != "w" && token != "b")
							break;
						sfen << " " << token;

						// 手駒
						sfen << " " << feed_next(iss);

						// 初期局面からの手数
						sfen <<  " " << feed_next(iss);

						break;
					}
					return sfen.str();
				};

				StateInfo state;

				bool hirate = true;
				istringstream iss(sfen);
				do {
					token = feed_next(iss);
					if (token == "sfen")
					{
						// 駒落ちなどではsfen xxx movesとなるのでこれをfeedしなければならない。
						auto sfen = feed_sfen(iss);
						pos.set(sfen,&state);
						hirate = false;
					}
				} while (token == "startpos" || token == "moves" || token == "sfen");

				if (hirate)
					pos.set_hirate(&state);

				vector<Move> m;				// 初手から(moves+1)手までの指し手格納用

				// is_validは、この局面を処理対象とするかどうかのフラグ
				// 処理対象としない局面でもとりあえずsfにpush_back()はしていく。(indexの番号が狂うため)
				typedef pair<string, bool /*is_valid*/> SfenAndBool;
				vector<SfenAndBool> sf;		// 初手から(moves+0)手までのsfen文字列格納用

				// これより長い棋譜、食わせない＆思考対象としないやろ
				std::vector<StateInfo> states(1024);

				// 変数sfに解析対象局面としてpush_backする。
				// ただし、
				// 1) color == COLOR_NB (希望する手番なし)のとき
				// 2) この局面の手番が、希望する手番の局面のとき
				// に限る。
				auto append_to_sf = [&sf,&pos,&color]()
				{
					sf.push_back(SfenAndBool(pos.sfen(),
						/* is_valid = */ color == COLOR_NB || color == pos.side_to_move()));
				};

				// sfenから直接生成するときはponderのためにmoves + 1の局面まで調べる必要がある。
				for (int i = 0; i < moves + (from_sfen ? 1 : 0); ++i)
				{
					// 初回は、↑でfeedしたtokenが入っているはず。
					if (i != 0)
					{
						token = "";
						iss >> token;
					}
					if (token == "")
					{
						// この局面、未知の局面なのでpushしないといけないのでは..
						if (!from_sfen)
							append_to_sf();
						break;
					}

					Move move = USIEngine::to_move(pos, token);
					// illigal moveであるとMOVE_NONEが返る。
					if (move == Move::none())
					{
						cout << "illegal move : line = " << (k + 1) << " , " << sfen << " , move = " << token << endl;
						break;
					}

					// MOVE_WIN,MOVE_RESIGNでは局面を進められないのでここで終了。
					if (!move.is_ok())
						break;

					append_to_sf();
					m.push_back(move);

					pos.do_move(move, states[i]);
				}

				for (int i = 0; i < (int)sf.size() - (from_sfen ? 1 : 0); ++i)
				{
					if (i < start_moves - 1)
						continue;

					// 現局面の手番が望むべきものではないので解析をskipする。
					if (!sf[i].second /* sf[i].is_valid */)
						continue;

					const auto& sfen = sf[i].first;
					if (from_sfen)
					{
						// この場合、m[i + 1]が必要になるので、m.size()-1までしかループできない。
						BookMove bp(m[i].to_move16(), m[i + 1].to_move16(), VALUE_ZERO, 32, 1);
						book.insert(sfen, bp);
					}
					else if (from_thinking)
					{
						// posの局面で思考させてみる。(あとでまとめて)
						if (thinking_sfens.count(sfen) == 0)
							thinking_sfens.insert(sfen);
					}
				}

				// sfenから生成するモードの場合、1000棋譜処理するごとにドットを出力。
				if ((k % 1000) == 0)
					cout << '.';
			}
			cout << "done." << endl;

			book.write_book(book_name);
		}
		else if (book_merge) {

			// 定跡のマージ
			MemoryBook book[3];
			string book_name[3];
			is >> book_name[0] >> book_name[1] >> book_name[2];
			if (book_name[2] == "")
			{
				cout << "Error! book name is empty." << endl;
				return 1;
			}
			cout << "book merge from " << book_name[0] << " and " << book_name[1] << " to " << book_name[2] << endl;
			for (int i = 0; i < 2; ++i)
			{
				if (book[i].read_book(book_name[i]).is_not_ok())
					return 1;
			}

			// 読み込めたので合体させる。
			cout << "merge..";

			// 同一nodeと非同一nodeの統計用
			// diffrent_nodes1 = book0側にのみあったnodeの数
			// diffrent_nodes2 = book1側にのみあったnodeの数
			u64 same_nodes = 0;
			u64 diffrent_nodes1 = 0, diffrent_nodes2 = 0;

			// 1) 探索が深いほうを採用。
			// 2) 同じ探索深さであれば、MultiPVの大きいほうを採用。
			book[0].foreach([&](string sfen, BookMovesPtr it0)
				{
					// このエントリーがbook1のほうにないかを調べる。
					auto it1 = book[1].find(sfen);
					if (it1 != nullptr)
					{
						same_nodes++;

						// あったので、良いほうをbook2に突っ込む。
						// 1) 登録されている候補手の数がゼロならこれは無効なのでもう片方を登録
						// 2) depthが深いほう
						// 3) depthが同じならmulti pvが大きいほう(登録されている候補手が多いほう)
						if (it0->size() == 0)
							book[2].append(sfen, it1);
						else if (it1->size() == 0)
							book[2].append(sfen, it0);
						else if ((*it0)[0].depth > (*it1)[0].depth)
							book[2].append(sfen, it0);
						else if ((*it0)[0].depth < (*it1)[0].depth)
							book[2].append(sfen, it1);
						else if (it0->size() >= it1->size())
							book[2].append(sfen, it0);
						else
							book[2].append(sfen, it1);
					}
					else {
						// なかったので無条件でbook2に突っ込む。
						book[2].append(sfen, it0);
						diffrent_nodes1++;
					}
				});

			// book0の精査が終わったので、book1側で、まだ突っ込んでいないnodeを探して、それをbook2に突っ込む
			book[1].foreach([&](string sfen, BookMovesPtr it1)
				{
					if (book[2].find(sfen) == nullptr)
					{
						book[2].append(sfen, it1);
						diffrent_nodes2++;
					}
				});

			cout << "..done" << endl;

			cout << "same nodes = " << same_nodes
				<< " , different nodes =  " << diffrent_nodes1 << " + " << diffrent_nodes2 << endl;

			book[2].write_book(book_name[2]);

		}
		else if (book_sort) {
			// 定跡のsort
			MemoryBook book;
			string book_src, book_dst;
			is >> book_src >> book_dst;
			cout << "book sort from " << book_src << " , write to " << book_dst << endl;
			book.read_book(book_src);

			book.write_book(book_dst);

		}
		else if (convert_from_apery) {
			MemoryBook book;
			string book_src, book_dst;
			is >> book_src >> book_dst;
			cout << "convert apery book from " << book_src << " , write to " << book_dst << endl;
			book.read_apery_book(book_src);

			book.write_book(book_dst);
		}
		else if (convert_to_apery) {
			MemoryBook book;
			string book_src, book_dst;
			is >> book_src >> book_dst;
			cout << "convert book from " << book_src << " , write apery book to " << book_dst << endl;
			book.read_book(book_src);

			book.write_apery_book(book_dst);
		}

		return 1;
	}

} // namespace Book
} // namespace YaneuraOu

#endif // defined(YANEURAOU_ENGINE)
