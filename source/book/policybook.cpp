#include "../config.h"

#if defined(USE_POLICY_BOOK)

#include <numeric> // accumurate

#include "policybook.h"
#include "../misc.h"
#include "../position.h"
#include "../thread.h"
#include "../usi.h"

#define POLICY_BOOK_FILE_NAME "eval/policy_book.db"
#define POLICY_BOOK_BIN_NAME  "eval/policy_book.db.bin"

// ----------------------
// Policy Bookの読み込み
// ----------------------
void PolicyBook::read_book()
{
	SystemIO::TextReader reader;
	/* 2度目のisreadyに対しては読み込まない措置 */
	if (book_body.size() == 0)
	{
		// binary化されたPolicyBookがあるなら、それを読み込む。
		SystemIO::BinaryReader bin_reader;
		if (bin_reader.Open(POLICY_BOOK_BIN_NAME).is_ok())
		{
			// ファイルサイズ
			size_t bin_size = bin_reader.GetSize();

			// PolicyBookEntryのentry数
			size_t num_of_records = bin_size / sizeof(PolicyBookEntry);

			sync_cout << "info string read " << POLICY_BOOK_BIN_NAME
					  << " , " << num_of_records << " records." << sync_endl;

			book_body.resize(num_of_records);
			if (bin_reader.Read(book_body.data(), bin_size).is_ok())
			{
				sync_cout << "info string read done." << sync_endl;
			}
			else {
				sync_cout << "info string Error! : read error." << sync_endl;
				book_body.clear();
			}
		}
		else if (reader.Open(POLICY_BOOK_FILE_NAME).is_ok())
		{
			// ないなら、いったん読み込んで、binary化する。

			sync_cout << "info string read " << POLICY_BOOK_FILE_NAME << sync_endl;

			u64 counter = 0;
			std::string sfen, moves_str;
			Position pos;
			StateInfo si;
			std::vector<int> ratios;
			reader.ReadLine(sfen);
			if (sfen != "#YANEURAOU-POLICY-DB2024 1.00")
			{
				sync_cout << "info string Error! policy book header" << sync_endl;
				Tools::exit();
			}
			while (true)
			{
				if (reader.ReadLine(sfen).is_not_ok())
					break;
				pos.set(sfen, &si, Threads.main());

				reader.ReadLine(moves_str);
				auto moves = StringExtension::split(moves_str);
				// 7g7f 123 のように指し手と出現頻度が書いてあるので正規化する。
				ratios.clear();
				PolicyBookEntry pbe;
				for (size_t i = 0, j = 0; i < POLICY_BOOK_NUM; ++i)
				{
					if (i * 2 + 1 < moves.size())
					{
						Move16 move16 = USI::to_move16(moves[i * 2 + 0]);

						// 不成の指し手があるなら、それを無視する。
						// (これを計算に入れてしまうと、Policyの合計が100%にならなくなる)
						if (!pos.pseudo_legal_s<false>(pos.to_move(move16)))
							continue;

						pbe.move_ratio[j].move16 = move16;
						ratios.push_back(StringExtension::to_int(moves[i * 2 + 1], 1));
					}
					else {
						// 残りをmove::none()にしておく。
						pbe.move_ratio[j].move16 = Move16::none();
						pbe.move_ratio[j].ratio = 0;
					}
					++j;
				}

				// vlを足して (1 << 16) -1になるように正規化
				int total = std::accumulate(ratios.begin(), ratios.end(), 0);
				// 不成の指し手を除外した結果、0になることはある。これは正規化できない意味がないentry。
				if (total == 0)
					continue;

				for (size_t i = 0; i < ratios.size(); ++i)
					pbe.move_ratio[i].ratio = u16(u64((1 << 16) - 1) * ratios[i] / total);

				pbe.key = pos.hash_key();
				book_body.push_back(pbe);

				if (++counter % 100000 == 0)
					sync_cout << "info string read " << counter << sync_endl;
			}
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

			sync_cout << "info string read " << counter << "..done." << sync_endl;

			SystemIO::BinaryWriter bin_writer;
			if (bin_writer.Open(POLICY_BOOK_BIN_NAME).is_ok())
			{
				size_t write_size = book_body.size() * sizeof(PolicyBookEntry);
				sync_cout << "info string write " << POLICY_BOOK_BIN_NAME << " , write size = " << write_size << " bytes." << sync_endl;

				bin_writer.Write(book_body.data(), write_size);
				sync_cout << "info string ..done!" << sync_endl;
			}

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
	}
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

#endif
