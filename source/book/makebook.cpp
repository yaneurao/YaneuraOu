#include "../config.h"

#if defined (ENABLE_MAKEBOOK_CMD)

#include "book.h"
#include "../position.h"
#include <sstream>

using namespace std;
namespace YaneuraOu {
namespace Book
{

	// ----------------------------------
	// USI拡張コマンド "makebook"(定跡作成)
	// ----------------------------------

	// 定跡生成用の関数はplug-inのようになっていて、その関数は、自分の知っているコマンドを処理した場合、1を返す。

	// 定跡関連のコマンド。2015年ごろに作ったmakebookコマンド。
	int makebook2015(Position& pos, istringstream& is, const string& token);

	// 定跡生成コマンド2025年度版。ペタショック化コマンド。
    int makebook2025(istringstream& is, const string& token, const OptionsMap& options);

	// ---------------------------------------------------------------------------------------------

	// makebookコマンドの処理本体
	// フォーマット等については docs/解説.txt, docs/USI拡張コマンド.txt を見ること。
    void makebook(IEngine& engine, istringstream& is)
	{
        // このタイミングでisready()で初期化しておく。
        // (そうしないとPosition classやevalが使えなくて困る)
        engine.isready();

		string token;
		is >> token;

		// 2015年ごろに作ったmakebookコマンド
		if (makebook2015(engine.get_position(), is, token))
			return;

		// 2025年に作ったmakebook拡張コマンド
        if (makebook2025(is, token, engine.get_options()))
			return;

		// いずれのコマンドも処理しなかったので、使用方法を出力しておく。

		cout << "usage" << endl;
		cout << "> makebook from_sfen book.sfen book.db moves 24" << endl;
		cout << "> makebook think book.sfen book.db moves 16 depth 18" << endl;
		cout << "> makebook merge book_src1.db book_src2.db book_merged.db" << endl;
		cout << "> makebook sort book_src.db book_sorted.db" << endl;
		cout << "> makebook convert_from_apery book_src.bin book_converted.db" << endl;
		cout << "> makebook build_tree book2019.db user_book1.db" << endl;
		cout << "> makebook peta_shock book.db user_book1.db" << endl;
	}

} // namespace Book
} // namespace YaneuraOu

#endif // defined (ENABLE_MAKEBOOK_CMD)
