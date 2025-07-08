#ifndef UNIT_TEST_H_INCLUDED
#define UNIT_TEST_H_INCLUDED

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include "../usi.h"

namespace YaneuraOu {
namespace Test {

	// --------------------
	//      UnitTest
	// --------------------

	// UnitTest用。
	// "test unit"コマンドで用いる。
	// 実際の使い方は、normal_test_cmd.cppのunit_test()とPosition::UnitTest()を見ること。

	class UnitTester
	{
	public:

		// このclassはコンストラクタでUnitTestのheaderを出力し、
		// デストラクタで、UnitTestのsummeryを出力する。

		UnitTester();
		~UnitTester();

		// Test用のコマンド
		// 例) tester.test("局面の整合性テストその1",pos.is_ok());
		void test(const std::string& message, bool b);

		// Section()を呼び出した時に返す、インスタンス用の型
		struct UnitTesterSection
		{
			UnitTesterSection(UnitTester* t, const std::string& section_name) :tester(t) { tester->sections.push_back(section_name); }
			~UnitTesterSection() { tester->sections.pop_back(); }
			UnitTester* tester;
		};

		// Sectionを作る。
		// 使い方は、Position::UnitTest()を見ること。
		// 例)
		//   // Positionクラスのテストコード
		//   auto s1 = tester.section("Position");
		//   {
		// 	    auto s2 = tester.section("EnteringWin");
		// 	    {
		// 	       tester.test("test1",xxx);
		// 	    }
		//   }
		// のように書くとテストの時に "[Position::EnteringWin] test1 ... passed "のように
		// 左側に現在のセクション名を表示してくれる。
		UnitTesterSection section(const std::string& section_name)
		{
			return UnitTesterSection(this, section_name);
		}

		// 各classに実装されたstatic UnitTest(UnitTest&)を呼び出す時に用いる。
		// この関数fが呼び出される前にbefore_run()、呼び出された後にafter_run()が呼び出されるので
		// そこに共通の初期化コードを書いたりできる。(かも)
		void run(std::function<void(UnitTester&, IEngine& engine)> f);

		// run()の直前、直後に呼び出されるcallback
		std::function<void()> before_run;
		std::function<void()> after_run;

		// それぞれのUnitTestに渡すオプション一式。
		// 繰り返し回数などをこのパラメーターで設定する。
		// string->Variant(何でも取れる型)みたいなものがあれば良いだけなのだが、
		// C++にそういうのはないし、いまやりたいのは、パラメーター名からu64とstringにmapできる程度で良いので
		// OptionsMapを流用して用いることにする。
		OptionsMap options;

		IEngine* engine;

	protected:
		// 現在のsection名。
		std::vector<std::string> sections;

		// summery出力用に、エラーになった行を保存しておく。
		std::vector<std::string> errors;

		// 実施したテストの回数
		int test_count = 0;

		// 現在のsection名(入れ子あり)を返す。
		// 例) "[Position::EnteringWin]"
		const std::string section_name();
	};

	// --------------------
	// UnitTest本体。"unittest"コマンドで呼び出される。
	// --------------------

	void UnitTest(std::istringstream& is, IEngine& engine);

} // namespace Test
} // namespace YaneuraOu

#endif // UNIT_TEST_H_INCLUDED
