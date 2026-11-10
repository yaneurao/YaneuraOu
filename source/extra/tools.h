#ifndef EXTRA_TOOLS_H_INCLUDED
#define EXTRA_TOOLS_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "../misc.h"

namespace YaneuraOu {

class ThreadPool;

#if !STOCKFISH
// config.hで設定した値などについて出力する。
std::string config_info();

// 🌈 時間計測用。経過時間を計測する。
struct ElapsedTimer {
    ElapsedTimer();

    // startTimeを引数sで初期化する。
    ElapsedTimer(TimePoint s);

    // タイマーを初期化する。以降、elapsed()でinit()してからの経過時間が得られる。
    void reset();
    // TimePointを指定して初期化する。この時刻からの経過時間が求められるようになる。
    void reset(TimePoint s);

    // resetしてからの経過時間。
    TimePoint elapsed() const;

   private:
    // reset()された時刻。
    TimePoint startTime;
};

#endif


namespace Tools {

	// 進捗を表示しながら並列化してゼロクリア
	// Stockfishではtt.cppにこのコードがあるのだが、独自の置換表を確保したいときに
	// これが独立していないと困るので、ここに用意する。
	// nameは"Hash" , "eHash"などクリアしたいものの名前を書く。
	// メモリクリアの途中経過が出力されるときにその名前(引数nameで渡している)が出力される。
	// name == nullptrのとき、途中経過は表示しない。
	void memclear(YaneuraOu::ThreadPool& threads, const char* name, void* table, std::size_t size);

	// 途中での終了処理のためのwrapper
	// コンソールの出力が完了するのを待ちたいので3秒待ってから::exit(EXIT_FAILURE)する。
	void exit();

	// 指定されたミリ秒だけsleepする。
	void sleep(std::uint64_t ms);

	// std::getline()で1行読み、末尾のCR・LF・スペース・タブを除去する。
	// 行頭の空白は保持する。戻り値は読み取りが成功したかどうか。
	bool getline(std::ifstream& fs, std::string& s);


	// --------------------
	//    ProgressBar
	// --------------------

	// 処理の進捗を0%から100%の間で出力する。
	class ProgressBar
	{
	public:
		ProgressBar(){}

		// size_ : 全件でいくらあるかを設定する。
		ProgressBar(std::uint64_t size_);

		// また0%に戻す。このインスタンスを再利用する時に用いる。
		void reset(std::uint64_t size_);

		// 進捗を出力する。
		// current : 現在までに完了している件数
		void check(std::uint64_t current);

		// Progress Barの有効/無効を切り替える。
		// "readyok"までにProgressBarが被るとよろしくないので
		// learnコマンドとmakebookコマンドの時以外はオフでいいと思う。
		static void enable(bool b) { enable_ = b; }

	private:
		// 全件の数。
		std::uint64_t size;

		// 前回までに何個dotを出力したか。
		std::size_t dots;

		static bool enable_;
	};

	// --------------------
	//  Result
	// --------------------

	// 一般的な関数の返し値のコード。(エラー理由などのenum)
	enum struct ResultCode
	{
		// 正常終了
		Ok,

		// ファイルの終端に達した
		Eof,

		// 原因の詳細不明。何らかのエラー。
		SomeError,

		// メモリ割り当てのエラー
		MemoryAllocationError,

		// ファイルが存在しないエラー。
		FileNotFound,

		// ファイルのオープンに失敗。
		FileOpenError,

		// ファイル読み込み時のエラー。
		FileReadError,

		// ファイル書き込み時のエラー。
		FileWriteError,

		// ファイルClose時のエラー。
		FileCloseError,

		// ファイルを間違えているエラー。
		FileMismatch,

		// フォルダ作成時のエラー。
		CreateFolderError,

		// 実装されていないエラー。
		NotImplementedError,
	};

	// ResultCodeを文字列化する。
	std::string to_string(ResultCode);

	// エラーを含む関数の返し値を表現する型
	// RustにあるOption型のような何か
	struct Result
	{
		constexpr Result(ResultCode code_) : code(code_) {}

		// エラーの種類
		ResultCode code;

		// 返し値が正常終了かを判定する
		bool is_ok() const { return code == ResultCode::Ok; }

		// 返し値が正常終了でなければtrueになる。
		bool is_not_ok() const { return code != ResultCode::Ok; }

		// 返し値がEOFかどうかを判定する。
		bool is_eof() const { return code == ResultCode::Eof; }

		// ResultCodeを文字列化して返す。
		std::string to_string() const { return Tools::to_string(code); }

		//  正常終了の時の型を返すbuilder
		static constexpr Result Ok() { return Result(ResultCode::Ok); }
	};

}

// --------------------
//    文字列 拡張
// --------------------

// 文字列拡張(やねうら王独自)
namespace StringExtension
{
	// 大文字・小文字を無視して文字列の比較を行う。
	// Windowsだと_stricmp() , Linuxだとstrcasecmp()を使うのだが、
	// 後者がどうも動作が怪しい。自前実装しておいたほうが無難。
	// stricmpは、string case insensitive compareの略？
	// s1==s2のとき0(false)を返す。
	bool stricmp(const std::string& s1, const std::string& s2);

	// 行の末尾の"\r","\n",スペース、"\t"を除去した文字列を返す。
	// ios::binaryでopenした場合などには'\r'なども入っていることがあるので…。
	std::string trim(const std::string& input);

	// trim()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_inplace(std::string& input);

	// 行の末尾の数字を除去した文字列を返す。
	// sfenの末尾の手数を削除する用
	// 末尾のスペースを詰めたあと数字を詰めてそのあと再度スペースを詰める処理になっている。
	// 例 : "abc 123 "→"abc"となって欲しいので。
	std::string trim_number(const std::string& input);

	// trim_number()の高速版。引数で受け取った文字列を直接trimする。(この関数は返し値を返さない)
	void trim_number_inplace(std::string& s);

	// 文字列をint化する。int化に失敗した場合はdefault_の値を返す。
	int to_int(const std::string input, int default_);

	// 文字列をfloat化する。float化に失敗した場合はdefault_の値を返す。
	float to_float(const std::string input, float default_);

	// スペース、タブなど空白に相当する文字で分割して返す。
	std::vector<std::string> split(const std::string& input);

}

}

#endif
