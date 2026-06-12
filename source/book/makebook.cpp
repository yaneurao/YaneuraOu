#include "../config.h"

#if defined (ENABLE_MAKEBOOK_CMD)

#include "book.h"
#include "../misc.h"
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
		Tools::ProgressBar::enable(true);

		string token;
		is >> token;

		// 2025年に作ったmakebook拡張コマンド
        if (makebook2025(is, token, engine.get_options()))
		{
			Tools::ProgressBar::enable(false);
			return;
		}

		// いずれのコマンドも処理しなかったので、使用方法を出力しておく。

		cout << "usage" << endl;
		cout << "> makebook peta_shock book.db user_book1.db" << endl;
		Tools::ProgressBar::enable(false);
	}

} // namespace Book
} // namespace YaneuraOu

#endif // defined (ENABLE_MAKEBOOK_CMD)
