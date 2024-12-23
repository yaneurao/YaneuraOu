#include "../config.h"

#if defined(USE_POLICY_BOOK)

#include <numeric> // accumurate

#include "policybook.h"
#include "../misc.h"
#include "../position.h"
#include "../thread.h"
#include "../usi.h"
#include "../book/book.h"
#include "../eval/deep/nn.h"

// freqの和がUINT16_MAXに収まるようにする。
u16 MoveFreq32Record::overflow_check()
{
	u32 total = 0;
	for (auto& mf : move_freq32)
		total += mf.freq;

	// totalがUINT16_MAXに収まるようにだけしておく。(u16に格納したいため)
	// ⇨ totalがUINT16_MAX未満なので、freqsのそれぞれの要素がUINT16_MAX未満であることは保証される。
	while (total >= UINT16_MAX)
	{
		total /= 2;
		for (auto& mf : move_freq32 )
			mf.freq /= 2;
	}
	return u16(total);
}

// MoveFreq32Record構造体の値をこの構造体のmove_freqに代入する。
void PolicyBookEntry::from_move_freq32rec(const MoveFreq32Record& mf32r)
{
	for (size_t i = 0; i < POLICY_BOOK_NUM; ++i)
	{
		move_freq[i].move16 =     mf32r.move_freq32[i].move16 ;
		move_freq[i].freq   = u16(mf32r.move_freq32[i].freq  );
	}
	value = mf32r.value;
}


// ----------------------
// Policy Bookの読み込み
// ----------------------
Tools::Result PolicyBook::read_book_db(std::string path)
{
	SystemIO::TextReader   reader;
	auto result = reader.Open(path);

	if (result.is_ok())
	{
		// ないなら、いったん読み込んで、binary化する。

		sync_cout << "info string read " << path << sync_endl;

		u64 counter = 0;
		std::string sfen, moves_str;
		Position pos;
		StateInfo si;
		reader.ReadLine(sfen);
		if (sfen != POLICY_BOOK_HEADER)
		{
			sync_cout << "info string Error! invalid policy book header" << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
		while (true)
		{
			if (reader.ReadLine(sfen).is_not_ok())
				break;
			pos.set(sfen, &si, Threads.main());

			reader.ReadLine(moves_str);
			auto moves = StringExtension::split(moves_str);

			// 末尾判定が面倒なので、delimiterをセットしておく。
			moves.emplace_back("");
			moves.emplace_back("");

			// 7g7f 123 のように指し手と出現頻度が書いてあるので正規化する。
			// 7g7f 123 value 0.50
			// のように、その局面での手番側から見たvalueを付与することができる。
			// 7g7f 123 eval 400
			// のようにevalを付与することもできる。この場合、eval_coef=300として勝率に変換される。
			// いまどきのソフトは550程度だが、ここをあえて小さな値にすることで、valueとして
			// (0.5からの乖離が)大きめの値にする。

			// 出現頻度を保管しておく。
			MoveFreq32Record mf32r;

			for (size_t i = 0 , j = 0 ; ; i += 2)
			{
				auto& movestr = moves[i];
				if (movestr == "")
					break;
				if (movestr == "value")
					mf32r.value = StringExtension::to_float(moves[i + 1], FLT_MAX);
				else if (movestr == "eval")
					mf32r.value = Eval::dlshogi::cp_to_value(StringExtension::to_int(moves[i + 1], 0), 300);
				else {
					// もうお腹いっぱい。ここPOLICY_BOOK_NUM以上
					// 書いてもいいことにはなっているのでこの判定が必要。
					// まだこのあとvalueかevalが来ることはあるのでループは続行する。
					if (j >= POLICY_BOOK_NUM)
						continue;

					Move16 move16 = USI::to_move16(movestr);

					// 不成の指し手があるなら、それを無視する。
					// (これを計算に入れてしまうと、Policyの合計が100%にならなくなる)
					if (!pos.pseudo_legal_s<false>(pos.to_move(move16)))
						continue;

					mf32r.move_freq32[j].move16 = move16;
					mf32r.move_freq32[j].freq   = StringExtension::to_int(moves[i + 1], 1);
					j++;
				}
			}

			// 不成の指し手を除外した結果、0になることはある。これは正規化できない意味がないentry。
			if (mf32r.overflow_check() == 0)
				continue;

			PolicyBookEntry pbe;

			pbe.key = pos.hash_key();
			pbe.from_move_freq32rec(mf32r);

			book_body.push_back(pbe);

			if (++counter % 100000 == 0)
				sync_cout << "info string read " << counter << sync_endl;
		}

		sync_cout << "info string read " << counter << "..done." << sync_endl;

		// sortしないと二分探索できない。
		// sort_book();

		// 重複レコードがあるかも知れないのでgarbageしておく。
		garbage_book();

		/*
		// テストコード
		pos.set_hirate(&si,Threads.main());
		auto key = pos.hash_key();
		auto ptr = probe_policy_book(key);
		if (ptr != nullptr)
			for(int i=0; i < POLICY_BOOK_NUM; ++i)
			{
				auto& mr = ptr->move_ratio[i];
				sync_cout << "move = " << mr.move16 << " , ratio = " << mr.ratio << sync_endl;
			}
		*/
	}

	return result;
}

// PolicyBookを読み込む。(".db.bin"形式)
Tools::Result PolicyBook::read_book_db_bin(std::string path)
{
	SystemIO::BinaryReader bin_reader;
	auto result = bin_reader.Open(path);
	if (result.is_ok())
	{
		// ファイルサイズ
		size_t bin_size = bin_reader.GetSize();

		// PolicyBookEntryのentry数
		size_t num_of_records = bin_size / sizeof(PolicyBookEntry);

		sync_cout << "info string read " << path
			<< " , " << num_of_records << " records." << sync_endl;

		book_body.resize(num_of_records);
		result = bin_reader.Read(book_body.data(), bin_size);
		if (result.is_ok())
			sync_cout << "info string read done." << sync_endl;
		else
			sync_cout << "info string Error! : read error." << sync_endl;
	}
	return result;
}

// PolicyBookを読み込み、."db.bin"ファイルを書き出す。
Tools::Result PolicyBook::read_book()
{

#if !defined(ENABLE_POLICY_BOOK_LEARN)
	// まだ読み込んでいないならば..
	if (!is_loaded())
	{
		// binary化されたPolicyBookがあるなら、それを読み込む。
		if (read_book_db_bin().is_ok())
			return Tools::Result::Ok();

		auto result = read_book_db();
		if (result.is_ok())
		{
			// "db.bin"形式で書き出しておく。(次回の読み込み高速化のため)
			return write_book_db_bin();
		}
		return result;
	}
	return Tools::Result::Ok();

#else
	// ただし、ENABLE_POLICY_BOOK_LEARNが定義されているときは、毎回読み込む。(局後学習データがあるため)

	// binary化されたPolicyBookがあるなら、それを読み込む。
	Tools::Result result = read_book_db_bin();
	if (result.is_not_ok())
	{
		result = read_book_db();
		if (result.is_ok())
		{
			// "db.bin"形式で書き出しておく。(次回の読み込み高速化のため)
			result = write_book_db_bin();
		}
	}

	// そもそも読み込んでいないのでmerge不要。
	if (result.is_not_ok())
		result = read_book_db_bin(POLICY_BOOK_LEARN_DB_BIN_NAME);
	else {
		PolicyBook pb;
		result = pb.read_book_db_bin(POLICY_BOOK_LEARN_DB_BIN_NAME);
		// 読み込みに成功したのでmergeする。
		if (result.is_ok())
			merge_book(pb);
	}
	return result; // 読み込めたことにしておく。

#endif
}


// book_bodyをsortする。
// ※ probeは二分探索を用いるため、sortされていないとprobeできない。
void PolicyBook::sort_book()
{
	sync_cout << "info string sorting the policy book." << sync_endl;
	// keyでsortしておかないと、二分探索ができなくて困る。
	std::sort(book_body.begin(), book_body.end(), [](PolicyBookEntry& x, PolicyBookEntry& y)
		{ return x.key < y.key; });

	// keyが重複しないことを確認する。(これが成り立っていないと二分探索したあと局面が一意に定まらない。)
	for (size_t i = 0; i < book_body.size() - 1; ++i)
		if (book_body[i].key == book_body[i + 1].key)
		{
			sync_cout << "info string warning! PolicyBookEntry.key is duplicated." << sync_endl;
			break;
		}
}

// merge処理
void PolicyBook::merge_book(const PolicyBook& book)
{
	sync_cout << "info string merge the policy book. : "
		<< book_body.size() << " records + " << book.book_body.size() << " records." << sync_endl;

	// 連結
	book_body.insert(book_body.end(), book.book_body.begin(), book.book_body.end());

	// お掃除
	garbage_book();

	// 最終的なレコード数を出力する。
	sync_cout << "..done. " << book_body.size() << " records." << sync_endl;
}

// PolicyBookの重複レコードなどを掃除する。
void PolicyBook::garbage_book()
{
	// sort
	std::sort(book_body.begin(), book_body.end(), [](PolicyBookEntry& x, PolicyBookEntry& y)
		{ return x.key < y.key; });

	// 重複keyを持つEntryの削除
	size_t read_cursor = 0, write_cursor = 0;
	for (; read_cursor < book_body.size(); )
	{
		// 同じkeyが(2つ以上)連続しているなら、それらを集計する。
		if (read_cursor < book_body.size() - 1
			&& book_body[read_cursor].key == book_body[read_cursor + 1].key)
		{
			// 範囲が1以上なので、その区間をすべて集計して、book_body[write_cursor]に反映させる。
			std::unordered_map<Move16, u32> counter;

			HASH_KEY key = book_body[read_cursor].key;
			// ⇨ このkeyと一致するところを集計して一つにまとめる。

			// これも集計しないといけない。
			float value = FLT_MAX;

			for (; read_cursor < book_body.size() && key == book_body[read_cursor].key; ++read_cursor)
			{
				auto& mf = book_body[read_cursor].move_freq;
				for (int k = 0; k < POLICY_BOOK_NUM && mf[k].move16 != Move16::none(); ++k)
					counter[mf[k].move16] += mf[k].freq;
				float v = book_body[read_cursor].value;
				if (v != FLT_MAX)
					value = v;
			}
			// summarize終わったので、book_body[i]に反映させる。

			// counterの内容をvectorに変換(頻度でsortするすため)
			std::vector<MoveFreq32> sorted_counter(counter.begin(), counter.end());

			// 値でソート（降順）
			std::sort(sorted_counter.begin(), sorted_counter.end(),
				[](const MoveFreq32& a, const MoveFreq32& b) {
					return a.freq > b.freq; // 値で降順ソート
				});

			// book_body[write_cursor]に反映。(POLICY_BOOK_NUM個、entryを埋めるのを忘れずに)
			book_body[write_cursor].key = key;
			book_body[write_cursor].value = value;

			MoveFreq32Record mf32r;
			for (size_t k = 0; k < POLICY_BOOK_NUM; ++k)
				mf32r.move_freq32[k] = (k < sorted_counter.size()) ? sorted_counter[k] : MoveFreq32();

			// ここで不成の指し手除外するべきかも知れないが…。db.binでは除外されているものとしよう。
			mf32r.overflow_check();

			book_body[write_cursor].from_move_freq32rec(mf32r);

			// read_cursorは、keyが一致しないところ(今回集計していないところ)まで進んだ。
		}
		else {
			if (write_cursor != read_cursor)
				book_body[write_cursor] = book_body[read_cursor];
			// ⇨ write_cursor == read_cursor の場合はコピーのコストがもったいないから何もしない。

			read_cursor++;
		}
		write_cursor++;
	}
	// write_cursor - 1 までが有効なデータなので切り詰める。
	book_body.resize(write_cursor);
}


// book_bodyを"db.bin"形式で書き出す。
Tools::Result PolicyBook::write_book_db_bin(std::string path)
{
	SystemIO::BinaryWriter bin_writer;
	auto result = bin_writer.Open(path);
	if (result.is_ok())
	{
		size_t write_size = book_body.size() * sizeof(PolicyBookEntry);
		sync_cout << "info string write " << path << " , write size = " << write_size << " bytes." << sync_endl;

		result = bin_writer.Write(book_body.data(), write_size);
		sync_cout << "info string ..done!" << sync_endl;
	}
	return result;
}

// Policy Bookのなかに指定されたkeyの局面があるか。
// 見つからなければnullptrが返る。
PolicyBookEntry* PolicyBook::probe_policy_book(HASH_KEY key)
{
	PolicyBookEntry pbe;
	pbe.key = key;

	// std::lower_boundを使って検索
	auto it = std::lower_bound(book_body.begin(), book_body.end(), pbe,
		[](const PolicyBookEntry& a, const PolicyBookEntry& b) {
			return a.key < b.key;
		});

	// 見つかった場合にキーが一致するか確認

	// 同じキーを持つentryが複数ないことが保証されている。(読み込み時にwarningがでる)
	return (it != book_body.end() && it->key == key) ? &*it : nullptr;
}

// "position "コマンドのposition以降の文字列を渡して、それを
// POLICY_BOOK_LEARN_DB_BIN_NAMEにappendで書き出す。
void PolicyBook::append_sfen_to_db_bin(const std::string& sfen)
{
	Position pos;
	StateList si;
	std::vector<PolicyBookEntry> entries;

	BookTools::feed_position_string(pos, sfen, si, [&](Position& p, Move m) {
		// 最後の局面は、m==Move::none()が入ってくる。
		// また、不成の指し手は無視しないと対局時のPolicyの確率の合計が100%にならなくなる。
		if (m == Move::none() || !pos.pseudo_legal_s<false>(m))
			return;
		PolicyBookEntry entry;
		entry.key = p.hash_key();
		entry.move_freq[0] = MoveFreq(m.to_move16(), 1);
		entries.push_back(entry);
		});

	// ファイルにappendする。
	SystemIO::BinaryWriter writer;
	writer.Open(POLICY_BOOK_LEARN_DB_BIN_NAME, true);
	auto result = writer.Write(entries.data(), sizeof(PolicyBookEntry) * entries.size());
	sync_cout << "info string append " << POLICY_BOOK_LEARN_DB_BIN_NAME << ". status = " << result.to_string() << sync_endl;
}

#if 0
// PolicyBookのmergeが正常にできているかをテストするコード。
void merge_test()
{
	PolicyBook book1;
	book1.read_book_db_bin("C:/Users/yaneen/largefile/YaneuraOuBookWork/policy_book_2020-2024.20241204.db.bin");

	Position pos;
	StateInfo si;
	pos.set_hirate(&si, Threads.main());
	auto key = pos.hash_key();

	{
		auto ptr = book1.probe_policy_book(key);
		if (ptr != nullptr)
			for (int i = 0; i < POLICY_BOOK_NUM; ++i)
			{
				auto& mr = ptr->move_freq[i];
				sync_cout << "move = " << mr.move16 << " , ratio = " << mr.freq << sync_endl;
			}
	}

	PolicyBook book2;
	book2.read_book_db_bin("C:/Users/yaneen/largefile/YaneuraOuBookWork/policy_book_2020-2024.20241204.db.bin");
	book1.merge_book(book2);

	{
		auto ptr = book1.probe_policy_book(key);
		if (ptr != nullptr)
			for (int i = 0; i < POLICY_BOOK_NUM; ++i)
			{
				auto& mr = ptr->move_freq[i];
				sync_cout << "move = " << mr.move16 << " , ratio = " << mr.freq << sync_endl;
			}
	}

}
#endif

#endif // defined(USE_POLICY_BOOK)
