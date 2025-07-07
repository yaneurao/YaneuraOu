#ifndef BENCHMARK_H_INCLUDED
#define BENCHMARK_H_INCLUDED

#include <iosfwd>
#include <string>
#include <vector>

namespace YaneuraOu::Benchmark {

	// ベンチマーク用のUSIコマンド列をstringの配列に入れて返してくれる。
	//  currentFen : 現在のSfen文字列
	//  is         : コマンドライン("bench"に続くパラメーターを取得するためのもの)
	std::vector<std::string> setup_bench(const std::string& currentFen, std::istream& is);

	// ベンチマークのセットアップの構造体
	struct BenchmarkSetup {
		// 置換表サイズ
		// 📝 Stockfishではintだが、やねうら王ではsize_tに変更しておく。
		size_t                   ttSize;

		// スレッド数
		// 📝 Stockfishではintだが、やねうら王ではsize_tに変更しておく。
		size_t                   threads;

		// setup_bench()で生成したコマンド列
		std::vector<std::string> commands;

		// 呼び出した時の引数の情報
		std::string              originalInvocation;

		// 省略されていたところを埋めた引数の情報
		std::string              filledInvocation;
	};

	// benchコマンドのコマンドラインからBenchmarkSetupの構造体に情報を詰め込んで返す。
	BenchmarkSetup setup_benchmark(std::istream& is);

}  // namespace YaneuraOu

#endif  // #ifndef BENCHMARK_H_INCLUDED
