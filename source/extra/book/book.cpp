#include "../../shogi.h"
#include "book.h"
#include "../../position.h"
#include "../../misc.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../learn/multi_think.h"
#include "../../tt.h"
#include "apery_book.h"

#include <fstream>
#include <sstream>
#include <unordered_set>
#include <iomanip> 

using namespace std;
using std::cout;

void is_ready();

namespace Book
{
#ifdef ENABLE_MAKEBOOK_CMD
	// ----------------------------------
	// USI拡張コマンド "makebook"(定跡作成)
	// ----------------------------------

	// 局面を与えて、その局面で思考させるために、やねうら王2017Earlyが必要。
#if defined(EVAL_LEARN) && defined(YANEURAOU_2017_EARLY_ENGINE)

	struct MultiThinkBook : public MultiThink
	{
		MultiThinkBook(int search_depth_, MemoryBook & book_)
			: search_depth(search_depth_), book(book_), appended(false) {}

		virtual void thread_worker(size_t thread_id);

		// 定跡を作るために思考する局面
		vector<string> sfens;

		// 定跡を作るときの通常探索の探索深さ
		int search_depth;

		// メモリ上の定跡ファイル(ここに追加していく)
		MemoryBook& book;

		// 前回から新たな指し手が追加されたかどうかのフラグ。
		bool appended;
	};


	//  thread_id    = 0..Threads.size()-1
	//  search_depth = 通常探索の探索深さ
	void MultiThinkBook::thread_worker(size_t thread_id)
	{
		// g_loop_maxになるまで繰り返し
		u64 id;
		while ((id = get_next_loop_count()) != UINT64_MAX)
		{
			auto sfen = sfens[id];

			auto& pos = Threads[thread_id]->rootPos;
			pos.set(sfen);
			auto th = Threads[thread_id];
			pos.set_this_thread(th);

			if (pos.is_mated())
				continue;

			// depth手読みの評価値とPV(最善応手列)を取得。
			// 内部的にはLearner::search()を呼び出す。
			// Learner::search()は、現在のOptions["MultiPV"]の値に従い、MultiPVで思考することが保証されている。
			search(pos, search_depth);

			// MultiPVで局面を足す、的な

			vector<BookPos> move_list;

			int multi_pv = std::min((int)Options["MultiPV"], (int)th->rootMoves.size());
			for (int i = 0; i < multi_pv; ++i)
			{
				// 出現頻度は、バージョンナンバーを100倍したものにしておく)
				Move nextMove = (th->rootMoves[i].pv.size() >= 1) ? th->rootMoves[i].pv[1] : MOVE_NONE;
				BookPos bp(th->rootMoves[i].pv[0], nextMove, th->rootMoves[i].score
					, search_depth, int(atof(ENGINE_VERSION) * 100));

				// MultiPVで思考しているので、手番側から見て評価値の良い順に並んでいることは保証される。
				// (書き出しのときに並び替えなければ)
				move_list.push_back(bp);
			}

			{
				std::unique_lock<Mutex> lk(io_mutex);
				// 前のエントリーは上書きされる。
				book.book_body[sfen] = move_list;

				// 新たなエントリーを追加したのでフラグを立てておく。
				appended = true;
			}

			// 1局面思考するごとに'.'をひとつ出力する。
			cout << '.' << flush;
		}
	}
#endif

	// フォーマット等についてはdoc/解説.txt を見ること。
	void makebook_cmd(Position& pos, istringstream& is)
	{
		string token;
		is >> token;

		// sfenから生成する
		bool from_sfen = token == "from_sfen";
		// 自ら思考して生成する
		bool from_thinking = token == "think";
		// 定跡のマージ
		bool book_merge = token == "merge";
		// 定跡のsort
		bool book_sort = token == "sort";
		// 定跡の変換
		bool convert_from_apery = token == "convert_from_apery";

#if !defined(EVAL_LEARN) || !defined(YANEURAOU_2017_EARLY_ENGINE)
		if (from_thinking)
		{
			cout << "Error!:define EVAL_LEARN and YANEURAOU_2017_EARLY_ENGINE " << endl;
			return;
		}
#endif

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
			string book_name;
			is >> book_name;

			// 開始手数、終了手数、探索深さ
			int start_moves = 1;
			int moves = 16;
			int depth = 24;

			// 分散生成用。
			// 2/7なら、7個に分けて分散生成するときの2台目のPCの意味。
			// cluster 2 7
			// のように指定する。
			// 1/1なら、分散させない。
			// cluster 1 1
			// のように指定する。
			int cluster_id = 1;
			int cluster_num = 1;

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
				else if (from_thinking && token == "startmoves")
					is >> start_moves;
				else if (from_thinking && token == "cluster")
					is >> cluster_id >> cluster_num;
				else
				{
					cout << "Error! : Illigal token = " << token << endl;
					return;
				}
			}

			if (from_sfen)
				cout << "read sfen moves " << moves << endl;
			if (from_thinking)
				cout << "read sfen moves from " << start_moves << " to " << moves
				<< " , depth = " << depth
				<< " , cluster = " << cluster_id << "/" << cluster_num << endl;

			// 解析対象とするsfen集合。
			// 読み込むべきsfenファイル名が2つ指定されている時は、
			// 先手用と後手用の局面で個別のsfenファイルが指定されているということ。

			// Colorは、例えばBLACKが指定されていれば、ここで与えられるsfenの一連の局面は
			// 先手番のときのみ処理対象とする。
			typedef pair<string, Color> SfenAndColor;
			vector<SfenAndColor> sfens;
			
			if (bw_files)
			{
				vector<string> tmp_sfens;
				read_all_lines(sfen_file_name[0], tmp_sfens);

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
					read_all_lines(filename, tmp_sfens);
					for (auto& sfen : tmp_sfens)
						sfens.push_back(SfenAndColor(sfen, c));
				}
			}

			cout << "..done" << endl;

			MemoryBook book;

			if (from_thinking)
			{
				cout << "read book..";
				// 初回はファイルがないので読み込みに失敗するが無視して続行。
				if (read_book(book_name, book) != 0)
				{
					cout << "..but , create new file." << endl;
				}
				else
					cout << "..done" << endl;
			}

			// この時点で評価関数を読み込まないとKPPTはPositionのset()が出来ないので…。
			is_ready();

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

				istringstream iss(sfen);
				token = "";
				do {
					iss >> token;
				} while (token == "startpos" || token == "moves");

				vector<Move> m;				// 初手から(moves+1)手までの指し手格納用

				// is_validは、この局面を処理対象とするかどうかのフラグ
				// 処理対象としない局面でもとりあえずsfにpush_back()はしていく。(indexの番号が狂うため)
				typedef pair<string, bool /*is_valid*/> SfenAndBool;
				vector<SfenAndBool> sf;		// 初手から(moves+0)手までのsfen文字列格納用

				StateInfo si[MAX_PLY];

				pos.set_hirate();

				// 変数sfに解析対象局面としてpush_backする。
				// ただし、
				// 1) color == COLOR_NB (希望する手番なし)のとき
				// 2) この局面の手番が、希望する手番の局面のとき
				// に限る。
				auto append_to_sf = [&sf,pos,&color]()
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

					Move move = move_from_usi(pos, token);
					// illigal moveであるとMOVE_NONEが返る。
					if (move == MOVE_NONE)
					{
						cout << "illegal move : line = " << (k + 1) << " , " << sfen << " , move = " << token << endl;
						break;
					}

					// MOVE_WIN,MOVE_RESIGNでは局面を進められないのでここで終了。
					if (!is_ok(move))
						break;

					append_to_sf();
					m.push_back(move);

					pos.do_move(move, si[i]);
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
						BookPos bp(m[i], m[i + 1], VALUE_ZERO, 32, 1);
						insert_book_pos(book, sfen, bp);
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

#if defined(EVAL_LEARN) && defined(YANEURAOU_2017_EARLY_ENGINE)

			if (from_thinking)
			{
				// thinking_sfensを並列的に探索して思考する。
				// スレッド数(これは、USIのsetoptionで与えられる)
				u32 multi_pv = Options["MultiPV"];

				// 思考する局面をsfensに突っ込んで、この局面数をg_loop_maxに代入しておき、この回数だけ思考する。
				MultiThinkBook multi_think(depth, book);

				auto& sfens_ = multi_think.sfens;
				for (auto& s : thinking_sfens)
				{

					// この局面のいま格納されているデータを比較して、この局面を再考すべきか判断する。
					auto it = book.book_body.find(s);

					// MemoryBookにエントリーが存在しないなら無条件で、この局面について思考して良い。
					if (it == book.book_body.end())
						sfens_.push_back(s);
					else
					{
						auto& bp = it->second;
						if (bp[0].depth < depth // 今回の探索depthのほうが深い
							|| (bp[0].depth == depth && bp.size() < multi_pv) // 探索深さは同じだが今回のMultiPVのほうが大きい
							)
							sfens_.push_back(s);
					}
				}

#if 0
				// 思考対象局面が求まったので、sfenを表示させてみる。
				cout << "thinking sfen = " << endl;
				for (auto& s : sfens_)
					cout << "sfen " << s << endl;
#endif

				// 思考対象node数の出力。
				cout << "total " << sfens_.size() << " nodes " << endl;

				// クラスターの指定に従い、間引く。
				if (cluster_id != 1 || cluster_num != 1)
				{
					vector<string> a;
					for (int i = 0; i < (int)sfens_.size(); ++i)
					{
						if ((i % cluster_num) == cluster_id - 1)
							a.push_back(sfens_[i]);
					}
					sfens_ = a;

					// このPCに割り振られたnode数を表示する。
					cout << "for my PC : " << sfens_.size() << " nodes " << endl;
				}

				multi_think.set_loop_max(sfens_.size());

				// 30分ごとに保存
				// (ファイルが大きくなってくると保存の時間も馬鹿にならないのでこれくらいの間隔で妥協)
				multi_think.callback_seconds = 30 * 60;
				multi_think.callback_func = [&]()
				{
					std::unique_lock<Mutex> lk(multi_think.io_mutex);
					// 前回書き出し時からレコードが追加された？
					if (multi_think.appended)
					{
						write_book(book_name, book);
						cout << 'S' << endl;
						multi_think.appended = false;
					}
					else {
						// 追加されていないときは小文字のsマークを表示して
						// ファイルへの書き出しは行わないように変更。
						cout << 's' << endl;
					}

					// 置換表が同じ世代で埋め尽くされるとまずいのでこのタイミングで世代カウンターを足しておく。
					TT.new_search();
				};

				multi_think.go_think();

			}

#endif

			cout << "write..";
			write_book(book_name, book);
			cout << "finished." << endl;

		}
		else if (book_merge) {

			// 定跡のマージ
			MemoryBook book[3];
			string book_name[3];
			is >> book_name[0] >> book_name[1] >> book_name[2];
			if (book_name[2] == "")
			{
				cout << "Error! book name is empty." << endl;
				return;
			}
			cout << "book merge from " << book_name[0] << " and " << book_name[1] << " to " << book_name[2] << endl;
			for (int i = 0; i < 2; ++i)
			{
				if (read_book(book_name[i], book[i]) != 0)
					return;
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
			for (auto& it0 : book[0].book_body)
			{
				auto sfen = it0.first;
				// このエントリーがbook1のほうにないかを調べる。
				auto it1_ = book[1].book_body.find(sfen);
				auto& it1 = *it1_;
				if (it1_ != book[1].book_body.end())
				{
					same_nodes++;

					// あったので、良いほうをbook2に突っ込む。
					// 1) 登録されている候補手の数がゼロならこれは無効なのでもう片方を登録
					// 2) depthが深いほう
					// 3) depthが同じならmulti pvが大きいほう(登録されている候補手が多いほう)
					if (it0.second.size() == 0)
						book[2].book_body.insert(it1);
					else if (it1.second.size() == 0)
						book[2].book_body.insert(it0);
					else if (it0.second[0].depth > it1.second[0].depth)
						book[2].book_body.insert(it0);
					else if (it0.second[0].depth < it1.second[0].depth)
						book[2].book_body.insert(it1);
					else if (it0.second.size() >= it1.second.size())
						book[2].book_body.insert(it0);
					else
						book[2].book_body.insert(it1);
				}
				else {
					// なかったので無条件でbook2に突っ込む。
					book[2].book_body.insert(it0);
					diffrent_nodes1++;
				}
			}
			// book0の精査が終わったので、book1側で、まだ突っ込んでいないnodeを探して、それをbook2に突っ込む
			for (auto& it1 : book[1].book_body)
			{
				if (book[2].book_body.find(it1.first) == book[2].book_body.end())
				{
					book[2].book_body.insert(it1);
					diffrent_nodes2++;
				}
			}
			cout << "..done" << endl;

			cout << "same nodes = " << same_nodes
				<< " , different nodes =  " << diffrent_nodes1 << " + " << diffrent_nodes2 << endl;

			cout << "write..";
			write_book(book_name[2], book[2]);
			cout << "..done!" << endl;

		}
		else if (book_sort) {
			// 定跡のsort
			MemoryBook book;
			string book_src, book_dst;
			is >> book_src >> book_dst;
			cout << "book sort from " << book_src << " , write to " << book_dst << endl;
			Book::read_book(book_src, book);

			cout << "write..";
			write_book(book_dst, book, true);
			cout << "..done!" << endl;

		}
		else if (convert_from_apery) {
			MemoryBook book;
			string book_src, book_dst;
			is >> book_src >> book_dst;
			cout << "convert apery book from " << book_src << " , write to " << book_dst << endl;
			Book::read_apery_book(book_src, book);

			cout << "write..";
			write_book(book_dst, book, true);
			cout << "..done!" << endl;

		}
		else {
			cout << "usage" << endl;
			cout << "> makebook from_sfen book.sfen book.db moves 24" << endl;
			cout << "> makebook think book.sfen book.db moves 16 depth 18" << endl;
			cout << "> makebook merge book_src1.db book_src2.db book_merged.db" << endl;
			cout << "> makebook sort book_src.db book_sorted.db" << endl;
			cout << "> makebook convert_from_apery book_src.bin book_converted.db" << endl;
		}
	}
#endif

	string trim_sfen(string sfen);

	// Apery用定跡ファイルの読み込み
	int read_apery_book(const std::string& filename, MemoryBook& book)
	{
		// 読み込み済であるかの判定
		if (book.book_name == filename)
			return 0;

		auto convert_move_from_apery = [](uint16_t apery_move) {
			const uint16_t to = apery_move & 0x7f;
			const uint16_t from = (apery_move >> 7) & 0x7f;
			const bool is_promotion = (apery_move & (1 << 14)) != 0;
			if (is_promotion) {
				return make_move_promote(static_cast<Square>(from), static_cast<Square>(to));
			}
			const bool is_drop = ((apery_move >> 7) & 0x7f) >= SQ_NB;
			if (is_drop) {
				const uint16_t piece = from - SQ_NB + 1;
				return make_move_drop(static_cast<Piece>(piece), static_cast<Square>(to));
			}
			return make_move(static_cast<Square>(from), static_cast<Square>(to));
		};

		AperyBook apery_book(filename.c_str());
		cout << "size of apery book = " << apery_book.size() << endl;
		unordered_set<string> seen;
		uint64_t collisions = 0;

		auto report = [&]() {
			cout << "# seen positions = " << seen.size()
				<< ", size of converted book = " << book.book_body.size()
				<< ", # hash collisions detected = " << collisions
				<< endl;
		};

		function<void(Position&)> search = [&](Position& pos) {
			const string sfen = pos.sfen();
			const string sfen_for_key = trim_sfen(sfen);
			if (seen.count(sfen_for_key)) return;
			seen.insert(sfen_for_key);

			if (seen.size() % 100000 == 0) report();

			const auto& entries = apery_book.get_entries(pos);
			if (entries.empty()) return;
			bool has_illegal_move = false;
			for (const auto& entry : entries) {
				const Move move = convert_move_from_apery(entry.fromToPro);
				has_illegal_move |= !pos.legal(move);
			}
			if (has_illegal_move) {
				++collisions;
				return;
			}

			StateInfo st;
			for (const auto move : MoveList<LEGAL_ALL>(pos)) {
				pos.do_move(move, st);
				search(pos);
				pos.undo_move(move);
			}

			for (const auto& entry : entries) {
				const Move move = convert_move_from_apery(entry.fromToPro);
				BookPos bp(move, MOVE_NONE, entry.score, 1, entry.count);
				insert_book_pos(book, sfen, bp);
			}

			auto& move_list = book.book_body[sfen];
			std::stable_sort(move_list.begin(), move_list.end());
			uint64_t num_sum = 0;
			for (const auto& bp : move_list) {
				num_sum += bp.num;
			}
			num_sum = std::max(num_sum, UINT64_C(1)); // ゼロ除算対策
			for (auto& bp : move_list) {
				bp.prob = float(bp.num) / num_sum;
				pos.do_move(bp.bestMove, st);
				auto it = book.find(pos);
				if (it != book.end() && it->second.size() != 0) {
					bp.nextMove = it->second.front().bestMove;
				}
				pos.undo_move(bp.bestMove);
			}
		};

		Position pos;
		pos.set_hirate();
		search(pos);
		report();

		// 読み込んだファイル名を保存しておく。二度目のread_book()はskipする。
		book.book_name = filename;

		return 0;
	}

	static std::unique_ptr<AperyBook> apery_book;
	static const constexpr char* kAperyBookName = "book/book.bin";

	// 定跡ファイルの読み込み(book.db)など。
	int read_book(const std::string& filename, MemoryBook& book, bool on_the_fly)
	{
		// 読み込み済であるかの判定
		if (book.book_name == filename)
			return 0;

		// 別のファイルを開こうとしているので前回メモリに丸読みした定跡をクリアしておかないといけない。
		book.clear();

		// 読み込み済み、もしくは定跡を用いない(no_book)であるなら正常終了。
		if (filename == "book/no_book")
		{
			book.book_name = filename;
			return 0;
		}

		if (filename == kAperyBookName) {
			// Apery定跡データベースを読み込む
		//	apery_book = std::make_unique<AperyBook>(kAperyBookName);
			// これ、C++14の機能。C++11用に以下のように書き直す。
			apery_book = std::unique_ptr<AperyBook>(new AperyBook(kAperyBookName));
		}
		else {
			// やねうら王定跡データベースを読み込む

			// ファイルだけオープンして読み込んだことにする。
			if (on_the_fly)
			{
				if (book.fs.is_open())
					book.fs.close();

				book.fs.open(filename, ios::in);
				if (book.fs.fail())
				{
					cout << "info string Error! : can't read " + filename << endl;
					return 1;
				}

				book.on_the_fly = true;
				book.book_name = filename;
				return 0;
			}

			vector<string> lines;
			if (read_all_lines(filename, lines))
			{
				cout << "info string Error! : can't read " + filename << endl;
				//      exit(EXIT_FAILURE);
				return 1; // 読み込み失敗
			}

			uint64_t num_sum = 0;
			string sfen;

			auto calc_prob = [&] {
				auto& move_list = book.book_body[sfen];
				std::stable_sort(move_list.begin(), move_list.end());
				num_sum = std::max(num_sum, UINT64_C(1)); // ゼロ除算対策
				for (auto& bp : move_list)
					bp.prob = float(bp.num) / num_sum;
				num_sum = 0;
			};

			for (auto line : lines)
			{
				// バージョン識別文字列(とりあえず読み飛ばす)
				if (line.length() >= 1 && line[0] == '#')
					continue;

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					// ひとつ前のsfen文字列に対応するものが終わったということなので採択確率を計算して、かつ、採択回数でsortしておく
					// (sortはされてるはずだが他のソフトで生成した定跡DBではそうとも限らないので)。
					calc_prob();

					sfen = line.substr(5, line.length() - 5); // 新しいsfen文字列を"sfen "を除去して格納
					continue;
				}

				Move best, next;
				int value;
				int depth;

				istringstream is(line);
				string bestMove, nextMove;
				uint64_t num;
				is >> bestMove >> nextMove >> value >> depth >> num;

#if 0
				// 思考した指し手に対しては指し手の出現頻度のところを強制的にエンジンバージョンを100倍したものに変更する。
				// この#ifを有効にして、makebook mergeコマンドを叩いて、別のファイルに書き出すなどするときに便利。
				num = int(atof(ENGINE_VERSION) * 100);
#endif

				// 起動時なので変換に要するオーバーヘッドは最小化したいので合法かのチェックはしない。
				if (bestMove == "none" || bestMove == "resign")
					best = MOVE_NONE;
				else
					best = move_from_usi(bestMove);

				if (nextMove == "none" || nextMove == "resign")
					next = MOVE_NONE;
				else
					next = move_from_usi(nextMove);

				BookPos bp(best, next, value, depth, num);
				insert_book_pos(book, sfen, bp);
				num_sum += num;
			}
			// ファイルが終わるときにも最後の局面に対するcalc_probが必要。
			calc_prob();
		}

		// 読み込んだファイル名を保存しておく。二度目のread_book()はskipする。
		book.book_name = filename;

		return 0;
	}

	// 定跡ファイルの書き出し
	int write_book(const std::string& filename, const MemoryBook& book, bool sort)
	{
		fstream fs;
		fs.open(filename, ios::out);

		// バージョン識別用文字列
		fs << "#YANEURAOU-DB2016 1.00" << endl;

		vector<pair<string, vector<BookPos>> > vectored_book;
		for (auto& it : book.book_body)
		{
			// 指し手のない空っぽのentryは書き出さないように。
			if (it.second.size() == 0)
				continue;
			vectored_book.push_back(it);
		}

		if (sort)
		{
			// sfen文字列は手駒の表記に揺れがある。
			// (USI原案のほうでは規定されているのだが、将棋所が採用しているUSIプロトコルではこの規定がない。)
			// sortするタイミングで、一度すべての局面を読み込み、sfen()化しなおすことで
			// やねうら王が用いているsfenの手駒表記(USI原案)に統一されるようにする。

			{
				// Position::set()で評価関数の読み込みが必要。
				is_ready();
				Position pos;

				// std::vectorにしてあるのでit.firstを書き換えてもitは無効にならないはず。
				for (auto& it : vectored_book)
				{
					pos.set(it.first);
					it.first = pos.sfen();
				}
			}


			// ここvectored_bookが、sfen文字列でsortされていて欲しいのでsortする。
			// アルファベットの範囲ではlocaleの影響は受けない…はず…。
			std::sort(vectored_book.begin(), vectored_book.end(),
				[](const pair<string, vector<BookPos>>&lhs, const pair<string, vector<BookPos>>&rhs) {
				return lhs.first < rhs.first;
			});
		}

		for (auto& it : vectored_book)
		{
			fs << "sfen " << it.first /* is sfen string */ << endl; // sfen

			auto& move_list = it.second;

			// 採択回数でソートしておく。
			std::stable_sort(move_list.begin(), move_list.end());

			for (auto& bp : move_list)
				fs << bp.bestMove << ' ' << bp.nextMove << ' ' << bp.value << " " << bp.depth << " " << bp.num << endl;
			// 指し手、相手の応手、そのときの評価値、探索深さ、採択回数
		}

		fs.close();

		return 0;
	}

	void insert_book_pos(MemoryBook& book, const std::string sfen, const BookPos& bp)
	{
		auto it = book.book_body.find(sfen);
		if (it == book.end())
		{
			// 存在しないので要素を作って追加。
			vector<BookPos> move_list;
			move_list.push_back(bp);
			book.book_body[sfen] = move_list;
		}
		else {
			// この局面での指し手のリスト
			auto& move_list = it->second;
			// すでに格納されているかも知れないので同じ指し手がないかをチェックして、なければ追加
			for (auto& b : move_list)
				if (b == bp)
				{
					// すでに存在していたのでエントリーを置換。ただし採択回数はインクリメント
					auto num = b.num;
					b = bp;
					b.num += num;
					goto FOUND_THE_SAME_MOVE;
				}

			move_list.push_back(bp);

		FOUND_THE_SAME_MOVE:;
		}

	}

	// sfen文字列から末尾のゴミを取り除いて返す。
	// ios::binaryでopenした場合などには'\r'なども入っていると思われる。
	string trim_sfen(string sfen)
	{
		string s = sfen;
		int cur = (int)s.length() - 1;
		while (cur >= 0)
		{
			char c = s[cur];
			// 改行文字、スペース、数字(これはgame ply)ではないならループを抜ける。
			// これらの文字が出現しなくなるまで末尾を切り詰める。
			if (c != '\r' && c != '\n' && c != ' ' && !('0' <= c && c <= '9'))
				break;
			cur--;
		}
		s.resize((int)(cur + 1));
		return s;
	}

	Move convert_move_from_apery(uint16_t apery_move) {
		const uint16_t to = apery_move & 0x7f;
		const uint16_t from = (apery_move >> 7) & 0x7f;
		const bool is_promotion = (apery_move & (1 << 14)) != 0;
		if (is_promotion) {
			return make_move_promote(static_cast<Square>(from), static_cast<Square>(to));
		}
		const bool is_drop = ((apery_move >> 7) & 0x7f) >= SQ_NB;
		if (is_drop) {
			const uint16_t piece = from - SQ_NB + 1;
			return make_move_drop(static_cast<Piece>(piece), static_cast<Square>(to));
		}
		return make_move(static_cast<Square>(from), static_cast<Square>(to));
	};

	MemoryBook::BookType::iterator MemoryBook::find(const Position& pos)
	{
		// "no_book"は定跡なしという意味なので定跡の指し手が見つからなかったことにする。
		if (book_name == "no_book")
			return end();

		if (book_name == kAperyBookName) {
			// Apery定跡データベースを用いて指し手を選択する
			book_body.clear();
			const auto& entries = apery_book->get_entries(pos);
			int64_t sum_count = 0;
			for (const auto& entry : entries) {
				sum_count += entry.count;
			}

			for (const auto& entry : apery_book->get_entries(pos)) {
				BookPos book_pos(pos.move16_to_move(convert_move_from_apery(entry.fromToPro)), MOVE_NONE, entry.score, 256, entry.count);
				book_pos.prob = entry.count / static_cast<float>(sum_count);
				insert_book_pos(*this, pos.sfen(), book_pos);
			}

			return book_body.begin();
		}
		else {
			// やねうら王定跡データベースを用いて指し手を選択する

			// 定跡がないならこのまま返る。(sfen()を呼び出すコストの節約)
			if (!on_the_fly && book_body.size() == 0)
				return book_body.end();

			auto sfen = pos.sfen();
			BookType::iterator it;

			if (on_the_fly)
			{
				// ディスクから読み込むなら、いずれにせよ、book_bodyをクリアして、
				// ディスクから読み込んだエントリーをinsertしてそのiteratorを返すべき。
				book_body.clear();

				// 末尾の手数は取り除いておく。
				// read_book()で取り除くと、そのあと書き出すときに手数が消失するのでまずい。(気がする)
				sfen = trim_sfen(sfen);

				// ファイル自体はオープンされてして、ファイルハンドルはfsだと仮定して良い。

				// ファイルサイズ取得
				// C++的には未定義動作だが、これのためにsys/stat.hをincludeしたくない。
				// ここでfs.clear()を呼ばないとeof()のあと、tellg()が失敗する。
				fs.clear();
				fs.seekg(0, std::ios::end);
				auto file_end = fs.tellg();

				fs.clear();
				fs.seekg(0, std::ios::beg);
				auto file_start = fs.tellg();

				auto file_size = u64(file_end - file_start);

				// 与えられたseek位置から"sfen"文字列を探し、それを返す。どこまでもなければ""が返る。
				// hackとして、seek位置は-2しておく。(1行読み捨てるので、seek_fromぴったりのところに
				// "sfen"から始まる文字列があるとそこを読み捨ててしまうため。-2してあれば、そこに
				// CR+LFがあるはずだから、ここを読み捨てても大丈夫。)
				auto next_sfen = [&](u64 seek_from)
				{
					string line;

					fs.seekg(max(s64(0), (s64)seek_from - 2), fstream::beg);

					// --- 1行読み捨てる

					// seek_from == 0の場合も、ここで1行読み捨てられるが、1行目は
					// ヘッダ行であり、問題ない。
					getline(fs, line);

					// getlineはeof()を正しく反映させないのでgetline()の返し値を用いる必要がある。
					while (getline(fs, line))
					{
						if (!line.compare(0, 4, "sfen"))
							return trim_sfen(line.substr(5));
						// "sfen"という文字列は取り除いたものを返す。
						// 手数の表記も取り除いて比較したほうがいい。
						// ios::binaryつけているので末尾に'\R'が付与されている。禿げそう。
					}
					return string();
				};

				// バイナリサーチ
				// [s,e) の範囲で求める。

				u64 s = 0, e = file_size, m;

				while (true)
				{
					m = (s + e) / 2;

					auto sfen2 = next_sfen(m);
					if (sfen2 == "" || sfen < sfen2)
					{ // 左(それより小さいところ)を探す
						e = m;
					}
					else if (sfen > sfen2)
					{ // 右(それより大きいところ)を探す
						s = u64(fs.tellg() - file_start);
					}
					else {
						// 見つかった！
						break;
					}

					// 40バイトより小さなsfenはありえないので探索範囲がこれより小さいなら終了。
					// s,eは無符号型であることに注意。if (s-40 < e) と書くとs-40がマイナスになりかねない。
					if (s + 40 > e)
					{
						// ただしs = 0のままだと先頭要素が探索されていないということなので
						// このケースに限り先頭要素を再探索
						if (s == 0 && next_sfen(s) == sfen)
							break;

						// 見つからなかった
						return book_body.end();
					}

				}
				// 見つけた処理

				// read_bookとほとんど同じ読み込み処理がここに必要。辛い。

				uint64_t num_sum = 0;

				auto calc_prob = [&] {
					auto& move_list = book_body[sfen];
					std::stable_sort(move_list.begin(), move_list.end());
					num_sum = std::max(num_sum, UINT64_C(1)); // ゼロ除算対策
					for (auto& bp : move_list)
						bp.prob = float(bp.num) / num_sum;
					num_sum = 0;
				};

				// sfen文字列が合致したところまでは確定しており、そこまでfileのseekは完了している。
				// その直後に指し手が書かれているのでそれをgetline()で読み込めば良い。

				while (!fs.eof())
				{
					string line;
					getline(fs, line);

					// バージョン識別文字列(とりあえず読み飛ばす)
					if (line.length() >= 1 && line[0] == '#')
						continue;

					// コメント行(とりあえず読み飛ばす)
					if (line.length() >= 2 && line.substr(0, 2) == "//")
						continue;

					// 次のsfenに遭遇したらこれにて終了。
					if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
					{
						break;
					}

					Move best, next;
					int value;
					int depth;

					istringstream is(line);
					string bestMove, nextMove;
					uint64_t num;
					is >> bestMove >> nextMove >> value >> depth >> num;

					// 起動時なので変換に要するオーバーヘッドは最小化したいので合法かのチェックはしない。
					if (bestMove == "none" || bestMove == "resign")
						best = MOVE_NONE;
					else
						best = move_from_usi(bestMove);

					if (nextMove == "none" || nextMove == "resign")
						next = MOVE_NONE;
					else
						next = move_from_usi(nextMove);

					BookPos bp(best, next, value, depth, num);
					insert_book_pos(*this, sfen, bp);
					num_sum += num;
				}
				// ファイルが終わるときにも最後の局面に対するcalc_probが必要。
				calc_prob();

				it = book_body.begin();

			}
			else {

				// on the flyではない場合
				it = book_body.find(sfen);
			}


			if (it != book_body.end())
			{
				// 定跡のMoveは16bitであり、rootMovesは32bitのMoveであるからこのタイミングで補正する。
				for (auto& m : it->second)
					m.bestMove = pos.move16_to_move(m.bestMove);
			}
			return it;
		}
	}

	//
	// BookMoveSelector
	//

	using namespace USI;

	void BookMoveSelector::init(USI::OptionsMap & o)
	{
		// 実現確率の低い狭い定跡を選択しない
		o["NarrowBook"] << Option(false);

		// 定跡の指し手を何手目まで用いるか
		o["BookMoves"] << Option(16, 0, 10000);

		// 定跡ファイル名

		//  no_book          定跡なし
		//  standard_book.db 標準定跡
		//	yaneura_book1.db やねうら大定跡(公開用 concept proof)
		//	yaneura_book2.db 超やねうら定跡(大会用2015)
		//	yaneura_book3.db 真やねうら定跡(大会用2016)
		//	yaneura_book4.db 極やねうら定跡(大会用2017)
		//  user_book1.db    ユーザー定跡1
		//  user_book2.db    ユーザー定跡2
		//  user_book3.db    ユーザー定跡3
		//  book.bin         Apery型の定跡DB

		std::vector<std::string> book_list = { "no_book" , "standard_book.db"
			, "yaneura_book1.db" , "yaneura_book2.db" , "yaneura_book3.db", "yaneura_book4.db"
			, "user_book1.db", "user_book2.db", "user_book3.db", "book.bin" };

		o["BookFile"] << Option(book_list, book_list[1], [&](const Option& o){ this->book_name = string(o); });
		book_name = book_list[1];

		//  BookEvalDiff: 定跡の指し手で1番目の候補の指し手と、2番目以降の候補の指し手との評価値の差が、
		//    この範囲内であれば採用する。(1番目の候補の指し手しか選ばれて欲しくないときは0を指定する)
		//  BookEvalBlackLimit : 定跡の指し手のうち、先手のときの評価値の下限。これより評価値が低くなる指し手は選択しない。
		//  BookEvalWhiteLimit : 同じく後手の下限。
		//  BookDepthLimit : 定跡に登録されている指し手のdepthがこれを下回るなら採用しない。0を指定するとdepth無視。

		o["BookEvalDiff"] << Option(30, 0, 99999);
		o["BookEvalBlackLimit"] << Option(0, -99999, 99999);
		o["BookEvalWhiteLimit"] << Option(-140, -99999, 99999);
		o["BookDepthLimit"] << Option(16, 0, 99999);

		// 定跡をメモリに丸読みしないオプション。(default = false)
		o["BookOnTheFly"] << Option(false);

		// 定跡データベースの採択率に比例して指し手を選択するオプション
		o["ConsiderBookMoveCount"] << Option(false);

	}

	// 定跡の指し手の選択
	bool BookMoveSelector::probe(Thread& th, Search::LimitsType& Limits, PRNG& prng)
	{
		auto& rootPos = th.rootPos;
		auto& rootMoves = th.rootMoves;

		// 定跡を用いる手数
		int book_ply = Options["BookMoves"];
		if (rootPos.game_ply() > book_ply)
			return false;

		auto it = memory_book.find(rootPos);
		if (it == memory_book.end() || it->second.size() == 0)
			return false;

		// 定跡にhitした。逆順で出力しないと将棋所だと逆順にならないという問題があるので逆順で出力する。
		// また、it->second->size()!=0をチェックしておかないと指し手のない定跡が登録されていたときに困る。

		// 1) やねうら標準定跡のように評価値なしの定跡DBにおいては
		// 出現頻度の高い順で並んでいることが保証されている。
		// 2) やねうら大定跡のように評価値つきの定跡DBにおいては
		// 手番側から見て評価値の良い順に並んでいることは保証されている。
		// 1),2)から、move_list[0]の指し手がベストの指し手と言える。

		// ただ、この並び替えを仮定するのはよろしくないので仮定しない。
		// コピーして、不要なものを削除していく。

		auto move_list = it->second;

		if (!Limits.silent)
		{
			// 将棋所では対応していないが、ShogiGUIの検討モードで使うときに
			// 定跡の指し手に対してmultipvを出力しておかないとうまく表示されないので
			// これを出力しておく。
			auto i = move_list.size();
			for (auto it = move_list.rbegin(); it != move_list.rend(); ++it, --i)
				sync_cout << "info pv " << it->bestMove << " " << it->nextMove
				<< " (" << fixed << std::setprecision(2) << (100 * it->prob) << "%)" // 採択確率
				<< " score cp " << it->value << " depth " << it->depth
				<< " multipv " << i << sync_endl;
		}

		// このなかの一つをランダムに選択

		// 評価値ベースで選ぶのでないなら、
		// 無難な指し手が選びたければ、採択回数が一番多い、最初の指し手(move_list[0])を選ぶべし。
		// 評価値ベースで選ぶときは、NarrowBookはオンにすべきではない。

		// 狭い定跡を用いるのか？
		bool narrowBook = Options["NarrowBook"];

		// この局面における定跡の指し手のうち、条件に合わないものを取り除いたあとの指し手の数
		if (narrowBook)
		{
			auto n = move_list.size();

			// 出現確率10%未満のものを取り除く。
			auto it_end = std::remove_if(move_list.begin(), move_list.end(), [](Book::BookPos& m) { return m.prob < 0.1; });
			move_list.erase(it_end, move_list.end());

			// 1手でも取り除いたなら、定跡から取り除いたことをGUIに出力
			if (!Limits.silent && (n!=move_list.size()))
				sync_cout << "info string NarrowBook : " << n << " moves to " << move_list.size() << " moves." << sync_endl;
		}

		if (move_list.size() == 0)
			return false;

		// 評価値の差などを反映。

		// 定跡として採用するdepthの下限。0 = 無視。
		auto depth_limit = (int)Options["BookDepthLimit"];
		if (depth_limit != 0 && move_list[0].depth < depth_limit)
		{
			sync_cout << "info string BookDepthLimit is lower than the depth of this node." << sync_endl;
			move_list.clear();
		}
		else {
			// ベストな評価値の候補手から、この差に収まって欲しい。
			auto eval_diff = (int)Options["BookEvalDiff"];
			auto value_limit1 = move_list[0].value - eval_diff;
			// 先手・後手の評価値下限の指し手を採用するわけにはいかない。
			auto stm_string = (rootPos.side_to_move() == BLACK) ? "BookEvalBlackLimit" : "BookEvalWhiteLimit";
			auto value_limit2 = (int)Options[stm_string];
			auto value_limit = max(value_limit1, value_limit2);

			auto n = move_list.size();
		
			// 評価値がvalue_limitを下回るものを削除
			auto it_end = std::remove_if(move_list.begin(), move_list.end(), [&](Book::BookPos& m) { return m.value < value_limit; });
			move_list.erase(it_end, move_list.end());

			// 候補手が1手でも減ったなら減った理由を出力
			if (n!=move_list.size())
				sync_cout << "info string BookEvalDiff = " << eval_diff << " ,  " << stm_string << " = " << value_limit2 
					<< " , " << n << " moves to " << move_list.size() << " moves." << sync_endl;
		}

		if (move_list.size() == 0)
			return false;

		{
			// move_list[0]～move_list[book_move_max-1]までのなかからまずはランダムに選ぶ。

			auto bestPos = move_list[prng.rand(move_list.size())];

			// 定跡ファイルの採択率に応じて指し手を選択するか
			if (Options["ConsiderBookMoveCount"])
			{
				// 1-passで採択率に従って指し手を決めるオンラインアルゴリズム
				// http://yaneuraou.yaneu.com/2015/01/03/stockfish-dd-book-%E5%AE%9A%E8%B7%A1%E9%83%A8/

				u64 sum_move_counts = 0;
				for(auto &move : move_list)
				{
					u64 move_count = std::max<u64>(1, move.num);
					sum_move_counts += move_count;
					if (prng.rand(sum_move_counts) < move_count)
						bestPos = move;
				}
			}
			auto bestMove = bestPos.bestMove;

			// RootMovesに含まれているかどうかをチェックしておく。
			auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
			if (it_move != rootMoves.end())
			{
				std::swap(rootMoves[0], *it_move);

				// 2手目の指し手も与えないとponder出来ない。
				// 定跡ファイルに2手目が書いてあったなら、それをponder用に出力する。
				if (bestPos.nextMove != MOVE_NONE)
				{
					if (rootMoves[0].pv.size() <= 1)
						rootMoves[0].pv.push_back(MOVE_NONE);
					rootMoves[0].pv[1] = bestPos.nextMove; // これが合法手でなかったら将棋所が弾くと思う。
				}

				// この指し手を指す
				return true;
			}
		}

		// 合法手のなかに含まれていなかった、もしくは定跡として選ばれる条件を満たさなかったので
		// 定跡の指し手は指さない。
		return false;
	}

}
