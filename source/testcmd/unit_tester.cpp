#include "unit_tester.h"

#include <iostream>
#include <sstream>

using namespace std;

// --------------------
//      UnitTest
// --------------------

// UnitTest用。
// "test unittest"コマンドで用いる。
// 実際の使い方は、normal_test_cmd.cppのunit_test()とPosition::UnitTest()を見ること。

UnitTester::UnitTester()
{
	cout << "=== Start UnitTest ===" << endl;
}
UnitTester::~UnitTester()
{
	cout << "=== Summary UnitTest ===" << endl;

	auto passed = test_count - errors.size();
	cout << "Summary : " << passed << " / " << test_count << " passed." << endl;
	
	if (errors.empty())
		cout << "-> Passed all UnitTest." << endl;
	else
	{
		cout << "Error List : " << endl;
		for (auto s : errors)
			cout << s << endl;
	}

	cout << "=== Finish UnitTest === " << endl;
}

// Test用のコマンド
// 例) tester.test("局面の整合性テストその1",pos.is_ok());
void UnitTester::test(const std::string& message, bool b)
{
	stringstream ss;
	ss << section_name() << " " << message;
	auto left = ss.str();

	string right = b ? "passed" : "failed";

	// 横、80文字ぐらいで収まるように。
	auto padding = std::max((size_t)0, 79 - left.size() - right.size());

	string mes = left + string(padding, '.') + right;

	cout << mes << endl;

	// errorがあったなら、summeryで表示するために保存しておく。
	if (!b)
		errors.emplace_back(mes);

	++test_count;
}

// Sectionを作る。
// 使い方は、Position::UnitTest()を見ること。
const std::string UnitTester::section_name()
{
	stringstream ss;

	ss << "[";
	bool first = true;
	for (auto section : sections)
	{
		if (!first)
			ss << "::";
		first = false;

		ss << section;
	}
	ss << "]";
		
	return ss.str();
}
