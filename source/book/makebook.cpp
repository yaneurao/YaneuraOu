#include "../config.h"
#include "book.h"
#include "../position.h"
#include <sstream>

using namespace std;

namespace Book
{

#if defined (ENABLE_MAKEBOOK_CMD)
	// ----------------------------------
	// USI拡張コマンド "makebook"(定跡作成)
	// ----------------------------------

	// 定跡生成用の関数はplug-inのようになっていて、その関数は、自分の知っているコマンドを処理した場合、1を返す。

	// 定跡関連のコマンド。2015年ごろに作ったmakebookコマンド。
	int makebook2015(Position& pos, istringstream& is, const string& token);

	// 定跡生成コマンド2019年度版。makebook2019.cppで定義されている。テラショック定跡手法。
	int makebook2019(Position& pos, istringstream& is, const string& token);

	// 定跡生成コマンド2021年度版。makebook2021.cppで定義されている。MCTSによる生成。
	//int makebook2021(Position& pos, istringstream& is, const string& token);

	// ---------------------------------------------------------------------------------------------

	// makebookコマンドの処理本体
	// フォーマット等については docs/解説.txt, docs/USI拡張コマンド.txt を見ること。
	void makebook_cmd(Position& pos, istringstream& is)
	{
		// makebookコマンドは、ほとんどがEVAL_LEARNが有効でないと使えないので、丸ごと使えないようにしておく。
#if !(defined(EVAL_LEARN) && defined(YANEURAOU_ENGINE))
		cout << "Error!:define EVAL_LEARN and YANEURAOU_ENGINE" << endl;
		return;
#endif

		// 評価関数を読み込まないとPositionのset()が出来ないのでis_ready()の呼び出しが必要。
		// ただし、このときに定跡ファイルを読み込まれると読み込みに時間がかかって嫌なので一時的にno_bookに変更しておく。
		auto original_book_file = Options["BookFile"];
		Options["BookFile"] = string("no_book");

		// IgnoreBookPlyオプションがtrue(デフォルトでtrue)のときは、定跡書き出し時にply(手数)のところを無視(0)にしてしまうので、
		// これで書き出されるとちょっと嫌なので一時的にfalseにしておく。
		auto original_ignore_book_ply = (bool)Options["IgnoreBookPly"];
		Options["IgnoreBookPly"] = false;

		SCOPE_EXIT(Options["BookFile"] = original_book_file; Options["IgnoreBookPly"] = original_ignore_book_ply; );

		// ↑ SCOPE_EXIT()により、この関数を抜けるときには復旧する。

		is_ready();

		string token;
		is >> token;

		// 2015年ごろに作ったmakebookコマンド
		if (makebook2015(pos, is, token))
			return;

		// 2019年以降に作ったmakebook拡張コマンド
		if (makebook2019(pos, is, token))
			return;

		// 2021年に作ったmakebook拡張コマンド
		//if (makebook2021(pos, is, token))
		//	return;

		// いずれのコマンドも処理しなかったので、使用方法を出力しておく。

		cout << "usage" << endl;
		cout << "> makebook from_sfen book.sfen book.db moves 24" << endl;
		cout << "> makebook think book.sfen book.db moves 16 depth 18" << endl;
		cout << "> makebook merge book_src1.db book_src2.db book_merged.db" << endl;
		cout << "> makebook sort book_src.db book_sorted.db" << endl;
		cout << "> makebook convert_from_apery book_src.bin book_converted.db" << endl;
		cout << "> makebook build_tree book2019.db user_book1.db" << endl;

	}
#endif

} // namespace Book
