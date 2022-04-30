﻿#include "../config.h"
#include "book.h"
#include "../position.h"
#include "../misc.h"
#include "../search.h"
#include "../thread.h"
#include "../learn/multi_think.h"
#include "../tt.h"
#include "apery_book.h"

#include <unordered_set>
#include <iomanip>		// std::setprecision()
#include <numeric>      // std::accumulate()

using namespace std;
using std::cout;

namespace Book
{
	std::ostream& operator<<(std::ostream& os, BookMove c)
	{
		os << "move " << c.move << " , ponder " << c.ponder << " , value " << c.value << " , depth " << c.depth;
		return os;
	}

	// Aperyの指し手の変換。
	Move16 convert_move_from_apery(uint16_t apery_move) {
		Move16 m;

		const uint16_t to = apery_move & 0x7f;
		const uint16_t from = (apery_move >> 7) & 0x7f;
		const bool is_promotion = (apery_move & (1 << 14)) != 0;
		if (is_promotion)
			m =  make_move_promote16(static_cast<Square>(from), static_cast<Square>(to));
		else
		{
			const bool is_drop = ((apery_move >> 7) & 0x7f) >= SQ_NB;
			if (is_drop) {
				const uint16_t piece = from - SQ_NB + 1;
				m = make_move_drop16(static_cast<PieceType>(piece), static_cast<Square>(to));
			}
			else
				m = make_move16(static_cast<Square>(from), static_cast<Square>(to));
		}

		return Move16(m);
	};

	// Aperyの指し手の変換。
	uint16_t convert_move_to_apery(Move16 m) {
		const uint16_t ispromote = is_promote(m) ? (1 << 14) : 0;
		const uint16_t from = ((is_drop(m)?
			(static_cast<uint16_t>(move_dropped_piece(m)) + SQ_NB - 1):
			static_cast<uint16_t>(from_sq(m))
		) & 0x7f) << 7;
		const uint16_t to = static_cast<uint16_t>(to_sq(m)) & 0x7f;
		return (ispromote | from | to);
	}

	// BookMoveを一つ追加する。
	// 動作はinsert()とほぼ同じだが、この局面に同じ指し手は存在しないことがわかっている時に用いる。
	// こちらのほうが、同一の指し手が含まれるかのチェックをしない分だけ高速。
	void BookMoves::push_back(const BookMove& book_move)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		moves.push_back(book_move);
		sorted = false; // sort関係が崩れたのでフラグをfalseに戻しておく。
	}

	// ある指し手と同じBookMoveのエントリーを探して、あればそのiteratorを返す。
	// thread safeにするためにコピーして返す。
	std::shared_ptr<BookMove> BookMoves::find_move(const Move16 m16) const
	{
		std::lock_guard<std::recursive_mutex> lock(const_cast<BookMoves*>(this)->mutex_);

		auto it = std::find_if(moves.begin(), moves.end(), [m16](const BookMove& book_move) { return book_move.move == m16; });
		return shared_ptr<BookMove>(it == moves.end() ? nullptr : new BookMove(*it));
	}

	// 指し手を出現回数、評価値順に並び替える。
	// ※　より正確に言うなら、BookMoveのoperator <()で定義されている順。
	// すでに並び替わっているなら、何もしない。
	void BookMoves::sort_moves()
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		// すでに並び替わっているなら何もしない。
		if (sorted)
			return ;

		std::stable_sort(moves.begin(), moves.end());

		// 並び替えたのでtrueにしておく。
		sorted = true;
	}

	// 定跡フォーマットの、ある局面の指し手を書いてある一行を引数lineに渡して、
	// BookMoveのオブジェクトを構築するbuilder。
	BookMove BookMove::from_string(std::string line)
	{
		Move16 move, ponder;

		string move_str, ponder_str;
		int value = 0;
		int depth = 0;
		u64 move_count = 1;

		//istringstream is(line);
		// value以降は、元データに欠落してるかもですよ。
		//is >> bestMove >> nextMove >> value >> depth >> num;

		// → istringstream、げろげろ遅いので、自前でparseする。

		Parser::LineScanner scanner(line);
		move_str   = scanner.get_text();
		ponder_str = scanner.get_text();

		// ここ以降、optional。

		value = (int)scanner.get_number(value);
		depth = (int)scanner.get_number(depth);
		move_count = (u64)scanner.get_number(move_count);

		// 起動時なので変換に要するオーバーヘッドは最小化したいので合法かのチェックはしない。

		move = (move_str == "none" || move_str == "resign") ? MOVE_NONE : USI::to_move16(move_str);
		ponder = (ponder_str == "none" || ponder_str == "resign") ? MOVE_NONE : USI::to_move16(ponder_str);

		return BookMove(move,ponder,value,depth,move_count);
	}

	void BookMoves::insert(const BookMove& bp, bool overwrite)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		// すでに格納されているかも知れないので同じ指し手がないかをチェックして、なければ追加
		for (auto& b : moves)
			if (b == bp)
			{
				// 上書きモードなのか？
				if (overwrite)
				{
					// すでに存在していたのでエントリーを置換。ただし採択回数は合算する。
					auto move_count = b.move_count;
					b = bp;
					b.move_count += move_count;

					sorted = false; // sort関係が崩れたのでフラグをfalseに戻しておく。
				}
				return;
			}

		// この指し手が見つからなかったので追加する。
		moves.push_back(bp);

		sorted = false; // sort関係が崩れたのでフラグをfalseに戻しておく。
	}

	// [ASYNC] このクラスの持つ指し手集合に対して、それぞれの局面を列挙する時に用いる
	void BookMoves::foreach(std::function<void(BookMove&)> f)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		for(auto& it : moves)
			f(it);
	}

	void MemoryBook::insert(const std::string& sfen, const BookMove& bp , bool overwrite)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		auto it = book_body.find(sfen);
		if (it == book_body.end())
		{
			// 存在しないので要素を作って追加。
			BookMovesPtr move_list(new BookMoves());
			move_list->push_back(bp);
			book_body[sfen] = move_list;
		}
		else {
			// この局面での指し手のリスト
			auto& book_moves = *it->second;
			book_moves.insert(bp, overwrite);
		}
	}

	// [ASYNC] このクラスの持つ定跡DBに対して、それぞれの局面を列挙する時に用いる
	void MemoryBook::foreach(std::function<void(std::string /*sfen*/, BookMovesPtr)> f)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		for(auto& it : book_body)
			f(it.first,it.second);
	}

	// ----------------------------------
	//			MemoryBook
	// ----------------------------------

	static std::unique_ptr<AperyBook> apery_book;
	static const constexpr char* kAperyBookName = "book.bin";

	std::string MemoryBook::trim(std::string input) const
	{
		return Options["IgnoreBookPly"] ? StringExtension::trim_number(input) : StringExtension::trim(input);
	}

	// 定跡ファイルの読み込み(book.db)など。
	Tools::Result MemoryBook::read_book(const std::string& filename, bool on_the_fly_)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		// 読み込み済であるかの判定
		// 一度read_book()が呼び出されたなら、そのときに読み込んだ定跡ファイル名が
		// book_nameに設定されているはずなので、これと一致したなら、ここでは今回の読み込み動作を終了して良い。
		// ただしon_the_flyの状態が変更になっているのであればファイルの読み直し等の処理が必要となる。
		// (前回はon_the_fly == trueであったのに今回はfalseであるというような場合、メモリに丸読みしなくては
		// 　ならないので、ここで終了してしまってはまずい。また逆に、前回はon_the_fly == falseだったものが
		// 　今回はtrueになった場合、本来ならメモリにすでに読み込まれているのだから読み直しは必要ないが、
		//　 何らかの目的で変更したのであろうから、この場合もきちんと反映しないとまずい。)
		bool ignore_book_ply_ = Options["IgnoreBookPly"];
		if (this->book_name == filename && this->on_the_fly == on_the_fly_ && this->ignoreBookPly == ignore_book_ply_)
			return Tools::Result::Ok();

		// 一度このクラスのメンバーが保持しているファイル名はクリアする。(何も読み込んでいない状態になるので)
		this->book_name = "";
		this->pure_book_name = "";

		// 別のファイルを開こうとしているので前回メモリに丸読みした定跡をクリアしておかないといけない。
		book_body.clear();
		this->on_the_fly = false;
		this->ignoreBookPly = ignore_book_ply_;

		// フォルダ名を取り去ったものが"no_book"(定跡なし)もしくは"book.bin"(Aperyの定跡ファイル)であるかを判定する。
		auto pure_filename = Path::GetFileName(filename);

		// 読み込み済み、もしくは定跡を用いない(no_book)であるなら正常終了。
		if (pure_filename == "no_book")
		{
			this->book_name = filename;
			this->pure_book_name = pure_filename;
			return Tools::Result::Ok();
		}

		if (pure_filename == kAperyBookName) {
			// Apery定跡データベースを読み込む
			//	apery_book = std::make_unique<AperyBook>(kAperyBookName);
			// これ、C++14の機能。C++11用に以下のように書き直す。
			apery_book = std::unique_ptr<AperyBook>(new AperyBook(filename));
		}
		else {
			// やねうら王定跡データベースを読み込む

			// ファイルだけオープンして読み込んだことにする。
			if (on_the_fly_)
			{
				if (fs.is_open())
					fs.close();

				fs.open(filename, ios::in);
				if (fs.fail())
				{
					sync_cout << "info string Error! : can't read file : " + filename << sync_endl;
					return Tools::Result(Tools::ResultCode::FileOpenError);
				}

				// 定跡ファイルのopenにも成功したし、on the flyできそう。
				// このときに限りこのフラグをtrueにする。
				this->on_the_fly = true;
				this->book_name = filename;
				return Tools::Result::Ok();
			}

			sync_cout << "info string read book file : " << filename << sync_endl;

			SystemIO::TextReader reader;
			// ReadLine()の時に行の末尾のスペース、タブを自動トリム。空行は自動スキップ。
			reader.SetTrim(true);
			reader.SkipEmptyLine(true);

			auto result = reader.Open(filename);
			if (result.is_not_ok())
			{
				sync_cout << "info string Error! : can't read file : " + filename << sync_endl;
				//      exit(EXIT_FAILURE);
				return result; // 読み込み失敗
			}

			string sfen;

			// 手数違いの重複エントリーは、手数の一番若いほうだけをMemoryBook::write_book()で書き出すようにしたので、
			// 以下のコードは不要(のはず)
#if 0
			// 一つ前のsfen文字列と、同一sfenエントリー内の手数の最小値
			string last_sfen;
			int last_sfen_ply = 0;
			bool ignore_book_ply = Options["IgnoreBookPly"];
#endif

			// 定跡に登録されている手数を無視するのか？
			// (これがtrueならばsfenから手数を除去しておく)
			bool ignoreBookPly = Options["IgnoreBookPly"];

			Tools::ProgressBar progress(reader.GetSize());

			std::string line;
			while(reader.ReadLine(line).is_ok())
			{
				progress.check(reader.GetFilePos());

				// バージョン識別文字列(とりあえず読み飛ばす)
				if (line.length() >= 1 && line[0] == '#')
					continue;

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					// 5文字目から末尾までをくり抜く。
					// 末尾のゴミは除去されているはずなので、Options["IgnoreBookPly"] == trueのときは、手数(数字)を除去。

					sfen = line.substr(5); // 新しいsfen文字列を"sfen "を除去して格納
					if (ignoreBookPly)
						StringExtension::trim_number_inplace(sfen); // 末尾の数字除去

#if 0
					if (ignore_book_ply)
					{
						int ply = StringExtension::to_int(StringExtension::mid(line, sfen.length() + 5), 0);

						// Options["IgnoreBookPly"] == trueのときに手数違いの重複エントリーがある場合がある。
						// すでに見つけたentryなら、このentryに対して一切の操作を行わない。
						// 若い手数のほうの局面情報を優先すべき。
						// ※　定跡DBはsfen文字列順にソートされているので、手数違いのエントリーは連続していると仮定できる。

						if (last_sfen != sfen)
							last_sfen_ply = INT_MAX;
						else if (last_sfen_ply < ply)
							sfen = "";

						last_sfen_ply = std::min(last_sfen_ply,ply); // 同一sfenエントリーのなかでの最小値にする
						last_sfen = sfen;
					}
#endif

					continue;
				}

				// Options["IgnoreBookPly"]==true絡みでskipするエントリーであるかの判定
				if (sfen.size() == 0)
					continue;

				insert(sfen, BookMove::from_string(line));

				// このinsert()、この関数の40%ぐらいの時間を消費している。
				// 他の部分をせっかく高速化したのに…。

				// これ、ファイルから読み込んだ文字列のまま保存しておき、
				// アクセスするときに解凍するほうが良かったか…。

				// unorderedmapなのでこれ以上どうしようもないな…。

				// テキストそのまま持っておいて、メモリ上で二分探索する実装のほうが良かったか。
				// (定跡がsfen文字列でソート済みであることが保証されているなら。保証されてないんだけども。)

			}
		}

		// 読み込んだファイル名を保存しておく。二度目のread_book()はskipする。
		this->book_name = filename;
		this->pure_book_name = pure_filename;

		sync_cout << "info string read book done. number of positions = " << size() << sync_endl;

		return Tools::Result::Ok();
	}

	// 定跡ファイルの書き出し
	Tools::Result MemoryBook::write_book(const std::string& filename /*, bool sort*/) const
	{
		std::lock_guard<std::recursive_mutex> lock(const_cast<MemoryBook*>(this)->mutex_);

		// Position::set()で評価関数の読み込みが必要。
		//is_ready();

		// →　この関数はbookコマンドからしか呼び出さず、bookコマンドの処理の先頭付近でis_ready()を
		// 呼び出しているため、この関数のなかでのis_ready()は呼び出さないことにする。

		SystemIO::TextWriter writer;
		if (writer.Open(filename).is_not_ok())
			return Tools::Result(Tools::ResultCode::FileOpenError);

		cout << endl << "write " + filename << endl;

		// バージョン識別用文字列
		writer.WriteLine("#YANEURAOU-DB2016 1.00");

		vector<pair<string, BookMovesPtr> > vectored_book;

		// 重複局面の手数違いを除去するのに用いる。
		// 手数違いの重複局面はOptions["IgnoreBookPly"]==trueのときに有害であるため、plyが最小のもの以外を削除する必要がある。
		// (Options["BookOnTheFly"]==true かつ Options["IgnoreBookPly"] == true のときに、手数違いのものがヒットするだとか、そういう問題と、
		// Options["IgnoreBookPly"]==trueのときにMemoryBook::read_book()で読み込むときに重複エントリーがあって何か地雷を踏んでしまう的な問題を回避。

		// sfenの手数の手前までの文字列とそのときの手数
		std::unordered_map<string, int> book_ply;

		for (auto& it : book_body)
		{
			// 指し手のない空っぽのentryは書き出さないように。
			if (it.second->size() == 0)
				continue;
			vectored_book.push_back(it);
		}

		// sfen文字列は手駒の表記に揺れがある。
		// (USI原案のほうでは規定されているのだが、将棋所が採用しているUSIプロトコルではこの規定がない。)
		// sortするタイミングで、一度すべての局面を読み込み、sfen()化しなおすことで
		// やねうら王が用いているsfenの手駒表記(USI原案)に統一されるようにする。

		// 進捗の出力
		u64 counter = 0;
		Tools::ProgressBar progress(vectored_book.size() * 2);

		{
			Position pos;

			// std::vectorにしてあるのでit.firstを書き換えてもitは無効にならないはず。
			for (auto& it : vectored_book)
			{
				StateInfo si;
				pos.set(it.first,&si,Threads.main());
				auto sfen = pos.sfen();
				it.first = sfen;

				auto sfen_left = StringExtension::trim_number(sfen); // 末尾にplyがあるはずじゃろ
				int ply = StringExtension::to_int(sfen.substr(sfen_left.length()), 0);

				auto it2 = book_ply.find(sfen_left);
				if (it2 == book_ply.end())
					book_ply[sfen_left] = ply; // エントリーが見つからなかったので何も考えずに追加
				else
					it2->second = std::min(it2->second, ply); // 手数の短いほうを代入しておく。

				progress.check(++counter);
			}
		}

		// ここvectored_bookが、sfen文字列でsortされていて欲しいのでsortする。
		// アルファベットの範囲ではlocaleの影響は受けない…はず…。
		std::sort(vectored_book.begin(), vectored_book.end(),
			[](const pair<string, BookMovesPtr>&lhs, const pair<string, BookMovesPtr>&rhs) {
			return lhs.first < rhs.first;
		});

		for (auto& it : vectored_book)
		{
			// -- 重複局面の手数違いの局面はスキップする(ファイルに書き出さない)

			auto sfen = it.first;
			auto sfen_left = StringExtension::trim_number(sfen); // 末尾にplyがあるはずじゃろ
			int ply = StringExtension::to_int(sfen.substr(sfen_left.length()), 0);
			if (book_ply[sfen_left] != ply)
				continue;

			// -- このentryを書き出す

			writer.WriteLine("sfen " + it.first /* is sfen string */); // sfen

			auto& move_list = *it.second;

			// 何らかsortしておく。
			move_list.sort_moves();

			// 指し手、相手の応手、そのときの評価値、探索深さ、採択回数
			for (auto& bp : move_list)
				if (writer.WriteLine(to_usi_string(bp.move) + ' ' + to_usi_string(bp.ponder) + ' '
					+ std::to_string(bp.value) + " " + std::to_string(bp.depth) + " " + std::to_string(bp.move_count)).is_not_ok())
					return Tools::Result(Tools::ResultCode::FileWriteError);

			progress.check(++counter);
		}

		writer.Close();

		return Tools::Result::Ok();
	}

	// book_body.find()のwrapper。book_body.find()ではなく、こちらのfindを呼び出して用いること。
	// sfen : sfen文字列(末尾にplyまで書かれているものとする)
	BookMovesPtr MemoryBook::find(const std::string& sfen) const
	{
		std::lock_guard<std::recursive_mutex> lock(const_cast<MemoryBook*>(this)->mutex_);

		auto it = book_body.find(trim(sfen));
		return it == book_body.end() ? BookMovesPtr() : it->second;
	}

	// [ASYNC] メモリに保持している定跡に局面を一つ追加する。
	//   book_body[sfen] = ptr;
	// と等価。
	void MemoryBook::append(const std::string& sfen, const Book::BookMovesPtr& ptr)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);
		book_body[sfen] = ptr;
	}


	BookMovesPtr MemoryBook::find(const Position& pos)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		// "no_book"は定跡なしという意味なので定跡の指し手が見つからなかったことにする。
		if (pure_book_name == "no_book")
			return BookMovesPtr();

		if (pure_book_name == kAperyBookName) {

			BookMovesPtr pml_entry(new BookMoves());

			// Apery定跡データベースを用いて指し手を選択する
			const auto& entries = apery_book->get_entries(pos);
			int64_t sum_count = 0;
			for (const auto& entry : entries) {
				sum_count += entry.count;
			}

			// 定跡ファイルによっては、採用率がすべて0ということがありうる。
			// cf. https://github.com/yaneurao/YaneuraOu/issues/65
			// この場合、sum_count == 0になるので、採用率を当確率だとみなして、1.0 / entries.size() にしておく。

			for (const auto& entry : entries) {

				Move16 theMove16 = convert_move_from_apery(entry.fromToPro);
				Move16 thePonder = MOVE_NONE;
#if 0
				// Aperyの定跡、ponderの指し手が書かれていない。合法手であるなら、1手進めて、その局面の定跡をprobe()して取得する。
				// →　呼び出し元で定跡PVの構築のためにこれに該当する処理は行われるから、ここでやらなくてよさそう。
				Move   theMove   = pos.to_move(theMove16);
				if (pos.pseudo_legal_s<true>(theMove) && pos.legal(theMove))
				{
					StateInfo si;
					Position* thePos = const_cast<Position*>(&pos);
					thePos->do_move(theMove,si);
					const auto& entries2 = apery_book->get_entries(pos);
					if (entries2.size())
						thePonder = convert_move_from_apery(entries2[0].fromToPro); // 1つ目の指し手にしておく。
					thePos->undo_move(theMove);
				}
#endif
				BookMove book_pos(theMove16 , thePonder , entry.score, 256, entry.count);
				pml_entry->insert(book_pos,true);
			}
			pml_entry->sort_moves();

			return 	pml_entry;
		}
		else {
			// やねうら王定跡データベースを用いて指し手を選択する

			// 定跡がないならこのまま返る。(sfen()を呼び出すコストの節約)
			if (!on_the_fly && book_body.size() == 0)
				return BookMovesPtr();

			auto sfen = pos.sfen();

			BookType::iterator it;

			// "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"のような文字列である。
			// IgnoreBookPlyがtrueのときは、
			// "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"まで一致したなら一致したとみなせば良い。
			// これはStringExtension::trim_number()でできる。

			if (on_the_fly)
			{
				// ディスクから読み込むなら、いずれにせよ、新規エントリーを作成してそれを返す必要がある。
				BookMovesPtr pml_entry(new BookMoves());

				// IgnoreBookPlyのときは末尾の手数は取り除いておく。
				// read_book()で取り除くと、そのあと書き出すときに手数が消失するのでまずい。(気がする)
				sfen = trim(sfen);

				// ファイル自体はオープンされてして、ファイルハンドルはfsだと仮定して良い。

				// ファイルサイズ取得
				// C++的には未定義動作だが、これのためにsys/stat.hをincludeしたくない。
				// ここでfs.clear()を呼ばないとeof()のあと、tellg()が失敗する。
				fs.clear();
				fs.seekg(0, std::ios::beg);
				auto file_start = fs.tellg();

				fs.clear();
				fs.seekg(0, std::ios::end);

				// ファイルサイズ
				auto file_size = s64(fs.tellg() - file_start);

				// 与えられたseek位置から"sfen"文字列を探し、それを返す。どこまでもなければ""が返る。
				// hackとして、seek位置は-2しておく。(1行読み捨てるので、seek_fromぴったりのところに
				// "sfen"から始まる文字列があるとそこを読み捨ててしまうため。-2してあれば、そこに
				// CR+LFがあるはずだから、ここを読み捨てても大丈夫。)

				// last_posには、現在のファイルポジションが返ってくる。
				// ※　実際の位置より改行コードのせいで少し手前である可能性はある。
				// ftell()を用いると、MSYS2 + g++ 環境でtellgが嘘を返す(getlineを呼び出した時に内部的に
				// bufferingしているため(?)、かなり先のファイルポジションを返す)ので自前で計算する。
				auto next_sfen = [&](s64 seek_from , s64& last_pos)
				{
					string line;

					seek_from = std::max( s64(0), seek_from - 2);
					fs.seekg(seek_from , fstream::beg);

					// --- 1行読み捨てる

					// seek_from == 0の場合も、ここで1行読み捨てられるが、1行目は
					// ヘッダ行であり、問題ない。
					getline(fs, line);
					last_pos = seek_from + (s64)line.size() + 1;
					// 改行コードが1文字はあるはずだから、+1しておく。

					// getlineはeof()を正しく反映させないのでgetline()の返し値を用いる必要がある。
					while (getline(fs, line))
					{
						last_pos += s64(line.size()) + 1;

						if (!line.compare(0, 4, "sfen"))
						{
							// ios::binaryつけているので末尾に'\r'が付与されている。禿げそう。
							// →　trim()で吸収する。(trimがStringExtension::trim_number()を呼び出すがそちらで吸収される)
							return trim(line.substr(5));
							// "sfen"という文字列は取り除いたものを返す。
							// IgnoreBookPly == trueのときは手数の表記も取り除いて比較したほうがいい。
						}
					}
					return string();
				};

				// バイナリサーチ
				//
				// 区間 [s,e) で解を求める。現時点での中間地点がm。
				// 解とは、探しているsfen文字列が書いてある行の先頭のファイルポジションのことである。
				//
				// next_sfen()でm以降にある"sfen"で始まる行を読み込んだ時、そのあとのファイルポジションがlast_pos。

				s64 s = 0, e = file_size, m , last_pos;
				// s,eは無符号型だと、s - 2のような式が負にならないことを保証するのが面倒くさい。
				// こういうのを無符号型で扱うのは筋が悪い。

				while (true)
				{
					m = (s + e) / 2;

					auto sfen2 = next_sfen(m, last_pos);
					if (sfen2 == "" || sfen < sfen2) {

						// 左(それより小さいところ)を探す
						e = m;

					} else if (sfen > sfen2) {

						// 右(それより大きいところ)を探す

						// next_sfen()のなかでgetline()し終わった時の位置より後ろに解がある。
						// ここでftell()を使いたいが、上に書いた理由で嘘が返ってくるようだ。
						s = last_pos;

					} else {
						// 見つかった！
						break;
					}

					// 40バイトより小さなsfenはありえないので、この範囲に２つの"sfen"で始まる文字列が
					// 入っていないことは保証されている。
					// ゆえに、探索範囲がこれより小さいなら先頭から調べて("sfen"と書かれている文字列を探して)終了。
					if (s + 40 > e)
					{
						if ( next_sfen(s, last_pos) == sfen)
							// 見つかった！
							break;

						// 見つからなかった
						return BookMovesPtr();
					}

				}
				// 見つけた処理

				// read_bookとほとんど同じ読み込み処理がここに必要。辛い。

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

					pml_entry->push_back(BookMove::from_string(line));
				}
				pml_entry->sort_moves();
				return pml_entry;

			} else {

				// on the flyではない場合
				it = book_body.find(trim(sfen));
				if (it != book_body.end())
				{
					// メモリ上に丸読みしてあるので参照透明だと思って良い。
					it->second->sort_moves();
					return BookMovesPtr(it->second);
				}

				// 空のentryを返す。
				return BookMovesPtr();
			}
		}
	}

	// Apery用定跡ファイルの読み込み（定跡コンバート用）
	// ・Aperyの定跡ファイルはAperyBookで別途読み込んでいるため、read_apery_bookは定跡のコンバート専用。
	// ・unreg_depth は定跡未登録の局面を再探索する深さ。デフォルト値1。
	Tools::Result MemoryBook::read_apery_book(const std::string& filename, const int unreg_depth)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		/*
		// 読み込み済であるかの判定
		if (book_name == filename)
			return Tools::Result::Ok();
		*/

		AperyBook apery_book(filename.c_str());
		cout << "size of apery book = " << apery_book.size() << endl;
		unordered_set<string> seen;
		uint64_t collisions = 0;

		auto report = [&]() {
			cout << "# seen positions = " << seen.size()
				<< ", size of converted book = " << book_body.size()
				<< ", # hash collisions detected = " << collisions
				<< endl;
		};

		function<void(Position&, int)> search = [&](Position& pos, int unreg_depth_current) {
			const string sfen = pos.sfen();
			if (unreg_depth == unreg_depth_current) {
				// 探索済みチェック: 未登録局面の深掘り時は探索済みセットのメモリ消費量が溢れるのを防ぐため、ここではチェックしない
				const string sfen_for_key = StringExtension::trim_number(sfen);
				if (seen.count(sfen_for_key)) return;
				seen.insert(sfen_for_key);

				if (seen.size() % 100000 == 0) report();
			}

			const auto& entries = apery_book.get_entries(pos);
			if (entries.empty()) {
				if (unreg_depth_current < 1) return;
			} else {
				if (unreg_depth != unreg_depth_current) {
					// 探索済みチェック: 未登録局面の深堀り時は、登録局面にヒットした時のみここでチェックする
					const string sfen_for_key = StringExtension::trim_number(sfen);

					if (seen.count(sfen_for_key))
						return;

					seen.insert(sfen_for_key);

					if (seen.size() % 100000 == 0)
						report();
				}
				bool has_illegal_move = false;
				for (const auto& entry : entries) {
					const Move move = pos.to_move(convert_move_from_apery(entry.fromToPro));
					has_illegal_move |= !pos.legal(move);
				}
				if (has_illegal_move) {
					++collisions;
					return;
				}
			}

			StateInfo st;
			for (const auto move : MoveList<LEGAL_ALL>(pos)) {
				pos.do_move(move, st);
				search(pos, entries.empty() ? unreg_depth_current - 1 : unreg_depth);
				pos.undo_move(move);
			}

			if (entries.empty()) return;
			for (const auto& entry : entries) {
				const Move16 move = convert_move_from_apery(entry.fromToPro);
				BookMove bp(move, MOVE_NONE , entry.score, 256, entry.count);
				insert(sfen, bp);
			}

			auto& move_list = *book_body[sfen];
			move_list.sort_moves();

			for (auto& bp : move_list) {
				Move move = pos.to_move(bp.move);
				pos.do_move(move, st);
				auto it = find(pos);
				if (it != nullptr && it->size()) {
					// Aperyの定跡DBではponderの指し手を持っていないので、
					// 次の局面での定跡のbestmoveをponderとしてやる。
					bp.ponder = (*it)[0].move;
				}
				pos.undo_move(move);
			}
		};

		Position pos;
		StateInfo si;
		pos.set_hirate(&si,Threads.main());
		search(pos, unreg_depth);
		report();

		/*
		// 読み込んだファイル名を保存しておく。二度目のread_book()はskipする。
		book_name = filename;
		*/

		return Tools::Result::Ok();
	}

	// Apery用定跡ファイルの書き出し（定跡コンバート用）
	Tools::Result MemoryBook::write_apery_book(const std::string& filename)
	{
		std::lock_guard<std::recursive_mutex> lock(mutex_);

		std::ofstream fs(filename, std::ios::binary);

		if (fs.fail())
			return Tools::Result(Tools::ResultCode::FileOpenError);

		std::cout << std::endl << "write " + filename;

		std::vector<std::pair<Key, BookMovesPtr> > vectored_book;

		{
			// 検索キー生成

			// ZobristHash初期化
			AperyBook::init();

			Position pos;

			for (auto& it : book_body)
			{
				std::string sfen = it.first;
				BookMovesPtr movesptr = it.second;

				StateInfo si;
				pos.set(sfen, &si, Threads.main());
				Key key = AperyBook::bookKey(pos);

				vectored_book.emplace_back(key, movesptr);
			}
		}

		// key順でsort
		std::sort(vectored_book.begin(), vectored_book.end(),
			[](const std::pair<Key, BookMovesPtr>&lhs, const std::pair<Key, BookMovesPtr>&rhs) {
			return lhs.first < rhs.first;
		});

		for (auto& it : vectored_book)
		{
			Key key = it.first;
			BookMoves& move_list = *it.second;

			// 何らかsortしておく。
			move_list.sort_moves();

			for (auto& bp : move_list)
			{
				AperyBookEntry entry = {
					key,
					convert_move_to_apery(bp.move),
					static_cast<uint16_t>(std::min(bp.move_count, static_cast<uint64_t>(UINT16_MAX))),
					bp.value
				};
				fs.write(reinterpret_cast<char*>(&entry), sizeof(entry));
			}

			if (fs.fail())
				return Tools::Result(Tools::ResultCode::FileWriteError);
		}

		fs.close();

		if (fs.fail())
			return Tools::Result(Tools::ResultCode::FileCloseError);

		std::cout << std::endl << "done!" << std::endl;

		return Tools::Result::Ok();
	}

	// ----------------------------------
	//			BookMoveSelector
	// ----------------------------------

	using namespace USI;

	void BookMoveSelector::init(USI::OptionsMap & o)
	{
		// エンジン側の定跡を有効化するか
		// USI原案にこのオプションがあり、ShogiGUI、ShogiDroidで対応しているらしいので
		// このオプションを追加。[2020/3/9]
		o["USI_OwnBook"] << Option(true);

		// 実現確率の低い狭い定跡を選択しない
		o["NarrowBook"] << Option(false);

		// 定跡の指し手を何手目まで用いるか
		o["BookMoves"] << Option(16, 0, 10000);

		// 一定の確率で定跡を無視して自力で思考させる
		o["BookIgnoreRate"] << Option(0, 0, 100);

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

#if !defined(__EMSCRIPTEN__)
		o["BookFile"] << Option(book_list, book_list[1]);
#else
		// WASM では no_book をデフォルトにする
		o["BookFile"] << Option(book_list, book_list[0]);
#endif

#if !defined(__EMSCRIPTEN__)
		o["BookDir"] << Option("book");
#else
		// WASM
		o["BookDir"] << Option(".");
#endif

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

		// 定跡にヒットしたときにPVを何手目まで表示するか。あまり長いと時間がかかりうる。
		o["BookPvMoves"] << Option(8, 1, MAX_PLY);

		// 定跡データベース上のply(開始局面からの手数)を無視するオプション。
		// 例) 局面図が同じなら、DBの36手目の局面に40手目でもヒットする。
		// これ変更したときに定跡ファイルの読み直しが必要になるのだが…(´ω｀)
		o["IgnoreBookPly"] << Option(false);
	}

	// 与えられたmで進めて定跡のpv文字列を生成する。
	string BookMoveSelector::pv_builder(Position& pos, Move16 m16 , int depth)
	{
		// 千日手検出
		auto rep = pos.is_repetition(MAX_PLY);
		if (rep != REPETITION_NONE)
		{
			// 千日手でPVを打ち切るときはその旨を表示(USI拡張)
			return " " + to_usi_string(rep);
		}

		string result = "";

		Move m = pos.to_move(m16);

		if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
		{
			StateInfo si;
			pos.do_move(m, si);

			Move16 bestMove16, ponderMove16;
			if (!probe_impl(pos, true, bestMove16, ponderMove16 , true /* 強制的にhitさせる */))
				goto UNDO;

			if (depth > 0)
				result = pv_builder(pos, bestMove16 , depth - 1); // さらにbestMoveで指し手を進める。

			result = " " + bestMove16.to_usi_string()
				+ ((result == "" /* is leaf node? */) ? (" " + ponderMove16.to_usi_string()) : result);

		UNDO:;
			pos.undo_move(m);
		}
		return result;
	}

	// probe()の下請け
	bool BookMoveSelector::probe_impl(Position& rootPos, bool silent , Move16& bestMove , Move16& ponderMove , bool forceHit)
	{
		if (!forceHit)
		{
			// 一定確率で定跡を無視
			if ((int)Options["BookIgnoreRate"] > (int)prng.rand(100)) {
				return false;
			}

			// 定跡を用いる手数
			int book_ply = (int)Options["BookMoves"];
			if (!forceHit && rootPos.game_ply() > book_ply)
				return false;
		}

		auto it = memory_book.find(rootPos);
		if (it == nullptr || it->size()==0)
			return false;

		// 定跡にhitした。逆順で出力しないと将棋所だと逆順にならないという問題があるので逆順で出力する。
		// →　将棋所、updateでMultiPVに対応して改良された
		// 　ShogiGUIでの表示も問題ないようなので正順に変更する。

		// また、it->size()!=0をチェックしておかないと指し手のない定跡が登録されていたときに困る。

		// 1) やねうら標準定跡のように評価値なしの定跡DBにおいては
		// 出現頻度の高い順で並んでいることが保証されている。
		// 2) やねうら大定跡のように評価値つきの定跡DBにおいては
		// 手番側から見て評価値の良い順に並んでいることは保証されている。
		// 1),2)から、move_list[0]の指し手がベストの指し手と言える。

		// ただ、この並び替えを仮定するのはよろしくないので仮定しない。
		// やや、オーバーヘッドはあるがコピーして、不要なものを削除していく。
		// 定跡にhitしたときに発生するオーバーヘッドなので通常は無視できるはず。

		auto move_list = *it;

		// 非合法手の排除(歩の不成を生成しないモードなら、それも排除)
		{
			auto it_end = std::remove_if(move_list.begin(), move_list.end(), [&](Book::BookMove& m) {
				Move move = rootPos.to_move(m.move);
				bool legal =  rootPos.pseudo_legal_s<true>(move) && rootPos.legal(move);

				// moveが非合法手ならば、エラーメッセージを出力しておいてやる。
				if (!silent && !legal)
				{
					sync_cout << "info string Error! : Illegal Move In Book DB : move = " << move
							  << " , sfen = " << rootPos.sfen() << sync_endl;

					// Position::legal()を用いて合法手判定をする時、これが連続王手の千日手を弾かないが、
					// 定跡で連続王手の千日手の指し手があると指してしまう。
					// これは回避が難しいので、仕様であるものとする。
					//
					// "position"コマンドでも千日手局面は弾かないし、この仕様は仕方ない意味はある。
				}
				else {
					// GenerateAllLegalMovesがfalseの時は歩の不成での移動は非合法手扱いで、この時点で除去してこのあとの抽選を行う。
					// 不成の指し手が選択されて、このあとrootMovesに登録されていないので定跡にhitしなかった扱いになってしまうのはもったいない。
					legal &= rootPos.pseudo_legal(move);
				}

				// 非合法手の排除
				return !legal;
				});
			move_list.erase(it_end, move_list.end());
		}

		// 出現回数のトータル(このあと出現頻度を求めるのに使う)
		u64 move_count_total = std::accumulate(move_list.begin(), move_list.end(), (u64)0, [](u64 acc, BookMove& b) { return acc + b.move_count; });
		move_count_total = std::max(move_count_total, (u64)1); // ゼロ除算対策

		if (!silent)
		{
			int pv_moves = (int)Options["BookPvMoves"];

			for (size_t i = 0; i < move_list.size() ; ++ i)
			{
				// PVを構築する。pv_movesで指定された手数分だけ表示する。
				// bestMoveを指した局面でさらに定跡のprobeを行なって…。
				auto& it = move_list[i];

				string pv_string;
				if (pv_moves <= 1)
					pv_string = it.move.to_usi_string();
				else if (pv_moves == 2)
					pv_string =it.move.to_usi_string() + " " + it.ponder.to_usi_string();
				else {
					// 次の局面で定跡にhitしない場合があって、その場合、この局面のnextMoveを出力してやる必要がある。
					auto rest = pv_builder(rootPos, it.move, pv_moves - 3);
					pv_string = (rest != "") ?
						(it.move.to_usi_string() + rest) :
						(it.move.to_usi_string() + " " + it.ponder.to_usi_string());
				}

				// USIの"info"で読み筋を出力するときは"pv"サブコマンドはサブコマンドの一番最後にしなければならない。
				// 複数出力するときに"multipv"は連番なのでこれが先頭に来ているほうが見やすいと思うので先頭に"multipv"を出力する。

				sync_cout << "info"
#if !defined(NICONICO)
					<< " multipv " << (i + 1)
#endif
					<< " score cp " << it.value << " depth " << it.depth
					<< " pv " << pv_string
					<< " (" << fixed << std::setprecision(2) << (100 * it.move_count / double(move_count_total)) << "%" << ")" // 採択確率
					<< sync_endl;

				// 電王盤はMultiPV非対応なので1番目の読み筋だけを"multipv"をつけずに送信する。
				// ("multipv"を出力してはならない)
#if defined(NICONICO)
				break;
#endif
			}
		}

		// このなかの一つをランダムに選択

		// 評価値ベースで選ぶのでないなら、
		// 無難な指し手が選びたければ、採択回数が一番多い、最初の指し手(move_list[0])を選ぶべし。
		// 評価値ベースで選ぶときは、NarrowBookはオンにすべきではない。

		if (forceHit)
		{
			// ベストな評価値のもののみを残す
			auto value_limit = move_list[0].value;

			// 評価値がvalue_limitを下回るものを削除
			auto it_end = std::remove_if(move_list.begin(), move_list.end(), [&](Book::BookMove & m) { return m.value < value_limit; });
			move_list.erase(it_end, move_list.end());

		} else {

			// 狭い定跡を用いるのか？
			bool narrowBook = Options["NarrowBook"];

			// この局面における定跡の指し手のうち、条件に合わないものを取り除いたあとの指し手の数
			if (narrowBook)
			{
				auto n = move_list.size();

				// 出現確率10%未満のものを取り除く。
				auto it_end = std::remove_if(move_list.begin(), move_list.end(), [&](Book::BookMove & m)
					{ return ((double)m.move_count / move_count_total) < 0.1; });
				move_list.erase(it_end, move_list.end());

				// 1手でも取り除いたなら、定跡から取り除いたことをGUIに出力
				if (!silent && (n != move_list.size()))
					sync_cout << "info string NarrowBook : " << n << " moves to " << move_list.size() << " moves." << sync_endl;
			}

			if (move_list.size() == 0)
				return false;

			// 評価値の差などを反映。

			// 定跡として採用するdepthの下限。0 = 無視。
			auto depth_limit = (int)Options["BookDepthLimit"];
			if ((depth_limit != 0 && move_list[0].depth < depth_limit))
			{
				if (!silent)
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
				auto it_end = std::remove_if(move_list.begin(), move_list.end(), [&](Book::BookMove & m) { return m.value < value_limit; });
				move_list.erase(it_end, move_list.end());

				// これを出力するとShogiGUIの棋譜解析で読み筋として表示されてしまう…。
				// 棋譜解析でinfo stringの文字列を拾う実装になっているのがおかしいのだが。
				// ShogiGUIの作者に要望を出す。[2019/06/20]
				// →　対応してもらえるらしい。[2019/06/22]

				// 候補手が1手でも減ったなら減った理由を出力
				if (!silent && n != move_list.size())
					sync_cout << "info string BookEvalDiff = " << eval_diff << " , " << stm_string << " = " << value_limit2
					<< " , " << n << " moves to " << move_list.size() << " moves." << sync_endl;
			}
		}
		if (move_list.size() == 0)
			return false;

		{
			// move_list[0]～move_list[book_move_max-1]までのなかからまずはランダムに選ぶ。

			auto bestBookMove = move_list[prng.rand(move_list.size())];

			// 定跡ファイルの採択率に応じて指し手を選択するか
			if (forceHit || Options["ConsiderBookMoveCount"])
			{
				// 1-passで採択率に従って指し手を決めるオンラインアルゴリズム
				// http://yaneuraou.yaneu.com/2015/01/03/stockfish-dd-book-%E5%AE%9A%E8%B7%A1%E9%83%A8/

				// 採用回数が0になっている定跡ファイルがあるらしい。
				// cf. https://github.com/yaneurao/YaneuraOu/issues/65
				// 1.すべての指し手の採用回数が0の場合 →　すべての指し手の採用回数を1とみなす
				// 2.特定の指し手の採用回数が0の場合 → 0割にならないように気をつける
				// 上記1.のため2-passとなってしまう…。

				// 採用回数の合計。
				u64 sum = 0;
				for (auto &move : move_list)
					sum += move.move_count;

				u64 sum_move_counts = 0;
				for (auto &move : move_list)
				{
					u64 move_count = (sum == 0) ? 1 : move.move_count; // 上記 1.
					sum_move_counts += move_count;
					if (sum_move_counts != 0 // 上記 2.
						&& prng.rand(sum_move_counts) < move_count)
						bestBookMove = move;
				}
			}

			bestMove = bestBookMove.move;
			ponderMove = bestBookMove.ponder;

			// ponderが登録されていなければ、bestMoveで一手進めてそこの局面のbestを拾ってくる。
			if (!is_ok((Move)ponderMove.to_u16()))
			{
				Move best = rootPos.to_move(bestMove);
				if (rootPos.pseudo_legal_s<true>(best) && rootPos.legal(best))
				{
					StateInfo si;
					rootPos.do_move(best,si);

					auto it = memory_book.find(rootPos);
					if (it != nullptr && it->size())
						// 1つ目に登録されている指し手が一番いい指し手であろう。
						ponderMove = (*it)[0].move;

					rootPos.undo_move(best);
				}
			}

			return true;
		}

		// 合法手のなかに含まれていなかった、もしくは定跡として選ばれる条件を満たさなかったので
		// 定跡の指し手は指さない。
		return false;
	}

	Move BookMoveSelector::probe(Position& pos)
	{
		const bool silent = true;
		Move16 bestMove16, ponderMove16;
		if (!probe_impl(pos, silent, bestMove16, ponderMove16))
			return MOVE_NONE;

		Move bestMove = pos.to_move(bestMove16);

		// bestMoveが合法であることは保証されている。(非合法手は除外してから選択を行うので)
		// なので、ここではそのチェックは行わない。

		return bestMove;
	}

	// 定跡の指し手の選択
	bool BookMoveSelector::probe(Thread& th, Search::LimitsType& Limits)
	{
		// エンジン側の定跡を有効化されていないなら、probe()に失敗する。
		if (!Options["USI_OwnBook"])
			return false;

		Move16 bestMove16, ponderMove16;
		auto& pos = th.rootPos;
		if (probe_impl(pos , Limits.silent, bestMove16, ponderMove16))
		{
			auto & rootMoves = th.rootMoves;

			// bestMoveは16bit Moveなので32bit化する必要がある。
			Move bestMove = pos.to_move(bestMove16);

			// RootMovesに含まれているかどうかをチェックしておく。
			// RootMovesをgoコマンドで指定されることがあるので、必ずこれはチェックしないといけない。
			// 注意)
			// 定跡で歩の不成の指し手がある場合、
			// "GenerateAllLegalMoves"がfalseだとrootMovesにはそれが生成されておらず、find()に失敗する。
			// この時、定跡にhitしなかった扱いとする。
			auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
			if (it_move != rootMoves.end())
			{
				std::swap(rootMoves[0], *it_move);

				// 2手目の指し手も与えないとponder出来ない。
				// 定跡ファイルに2手目が書いてあったなら、それをponder用に出力する。
				// これが合法手でなかったら将棋所が弾くと思う。
				// (ただし、"ponder resign"などと出力してしまうと投了と判定されてしまうらしいので
				//  普通の指し手でなければならない。これは、is_ok(Move)で判定できる。)
				if (is_ok((Move)ponderMove16.to_u16()))
				{
					if (rootMoves[0].pv.size() <= 1)
						rootMoves[0].pv.push_back(MOVE_NONE);

					// これ32bit Moveに変換してあげるほうが親切なのか…。
					StateInfo si;
					pos.do_move(bestMove,si);
					rootMoves[0].pv[1] = pos.to_move(ponderMove16);
					pos.undo_move(bestMove);
				}
				// この指し手を指す
				return true;
			}
		}
		return false;
	}

	// 定跡部のUnitTest
	void UnitTest(Test::UnitTester& tester)
	{
		// 少し書こうとしたが、ファイルから読み込むテストでないと大したテストにならないので考え中。
#if 0
		auto s1 = tester.section("Book");

		// Search::Limitsのalias
		auto& limits = Search::Limits;

		Position pos;
		StateInfo si;

		// 平手初期化
		auto hirate_init = [&] { pos.set_hirate(&si, Threads.main()); };

		{
			// Bookのprobeのテスト
			auto s2 = tester.section("probe");
			MemoryBook book;

			limits.generate_all_legal_moves = true;

			tester.test("pawn's unpromoted move", true);
		}
#endif
	}
}

// ===================================================
//                  BookTools
// ===================================================

// 定跡関係の処理のための補助ツール群
namespace BookTools
{
	// USIの"position"コマンドに設定できる文字列で、局面を初期化する。
	// 例)
	// "startpos"
	// "startpos moves xxx ..."
	// "sfen xxx"
	// "sfen xxx moves yyy ..."
	// また、局面を1つ進めるごとにposition_callback関数が呼び出される。
	// 辿った局面すべてに対して何かを行いたい場合は、これを利用すると良い。
	// 注意) siは、vector<StateInfo> si; si.reserve(MAX_PLY);のようにして事前に十分確保しておくこと。
	//  (そうでないとsi.emplace_back()で配列の要素のメモリ移動が起きると previous等のポインタが無効になってしまう)
	void feed_position_string(Position& pos, const std::string& root_sfen, std::vector<StateInfo>& si, const std::function<void(Position&)>& position_callback)
	{
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

		si.clear();
		si.emplace_back(StateInfo()); // このあとPosition::set()かset_hirate()を呼び出すので一つは必要。

		istringstream iss(root_sfen);
		string token;
		do {
			token = feed_next(iss);
			if (token == "sfen")
			{
				// 駒落ちなどではsfen xxx movesとなるのでこれをfeedしなければならない。
				auto sfen = feed_sfen(iss);
				pos.set(sfen,&si[0],Threads.main());
			}
			else if (token == "startpos")
			{
				// 平手初期化
				pos.set_hirate(&si[0], Threads.main());
			}
		} while (token == "startpos" || token == "sfen" || token == "moves"/* movesは無視してループを回る*/ );

		// callbackを呼び出してやる。
		position_callback(pos);

		// moves以降は1手ずつ進める
		while (token != "")
		{
			// 非合法手ならUSI::to_moveはMOVE_NONEを返すはず…。
			Move move = USI::to_move(pos, token);
			if (move == MOVE_NONE)
				break;

			// MOVE_NULL,MOVE_WINでは局面を進められないのでここで終了。
			if (!is_ok(move))
				break;

			si.emplace_back(StateInfo());
			pos.do_move(move, si.back());

			// callbackを呼び出してやる。
			position_callback(pos);

			token = feed_next(iss);
		}
	}

	// 平手、駒落ちの開始局面集
	// ここで返ってきた配列の、[0]は平手のsfenであることは保証されている。
	std::vector<std::string> get_start_sfens()
	{
		std::vector<std::string> start_sfens = {
			/*public static readonly string HIRATE = */       "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" ,
			/*public static readonly string HANDICAP_KYO = */ "sfen lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1" ,
			/*public static readonly string HANDICAP_RIGHT_KYO = */ "sfen 1nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_KAKU = */ "sfen lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_HISYA = */ "sfen lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_HISYA_KYO = */ "sfen lnsgkgsn1/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_2 =      */ "sfen lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_3 =      */ "sfen lnsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_4 =      */ "sfen 1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_5 =      */ "sfen 2sgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_LEFT_5 = */ "sfen 1nsgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_6 =      */ "sfen 2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_8 =      */ "sfen 3gkg3/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_10 =     */ "sfen 4k4/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
			/*public static readonly string HANDICAP_PAWN3 =  */ "sfen 4k4/9/9/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w 3p 1",
		};

		return start_sfens;
	}

	// "position"コマンドに設定できるsfen文字列を渡して、そこから全合法手で１手進めたsfen文字列を取得する。
	std::vector<std::string> get_next_sfens(std::string root_sfen)
	{
		Position pos;
		std::vector<StateInfo> si;
		si.reserve(MAX_PLY);
		feed_position_string(pos, root_sfen, si);
		StateInfo si2;
		vector<string> sfens;

		for (auto ml : MoveList<LEGAL_ALL>(pos))
		{
			auto m = ml.move;
			pos.do_move(m, si2);
			sfens.emplace_back("sfen " + pos.sfen());
			pos.undo_move(m);
		}

		return sfens;
	}
}
