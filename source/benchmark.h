#ifndef BENCHMARK_H_INCLUDED
#define BENCHMARK_H_INCLUDED

#include <iosfwd>
#include <string>
#include <vector>

namespace YaneuraOu::Benchmark {

// ベンチマーク用のUSIコマンド列を返す。
// 第1引数は現在のSFEN、第2引数はbenchに続く引数の入力ストリーム。
std::vector<std::string> setup_bench(const std::string&, std::istream&);

// speedtest用の設定とコマンド列。
struct BenchmarkSetup {
    // 置換表サイズ（MiB）。
    int                      ttSize;
    // スレッド数。
    int                      threads;
    // setup_benchmark()が生成するコマンド列。
    std::vector<std::string> commands;
    // 呼び出し時に指定された引数。
    std::string              originalInvocation;
    // 省略された引数を既定値で補った呼び出し内容。
    std::string              filledInvocation;
};

// speedtestに続く引数（スレッド数、置換表サイズMiB、合計時間秒）を読み込み、
// 設定とコマンド列を返す。省略時は順にハードウェアのスレッド数、
// 1スレッドあたり128 MiB、150秒を使用する。
BenchmarkSetup setup_benchmark(std::istream&);

}  // namespace YaneuraOu::Benchmark

#endif  // #ifndef BENCHMARK_H_INCLUDED
