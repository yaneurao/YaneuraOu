#ifndef EXTRA_SYSTEM_IO_H_INCLUDED
#define EXTRA_SYSTEM_IO_H_INCLUDED

#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "../misc.h"
#include "tools.h"

namespace YaneuraOu {

// --------------------
//  ファイル操作
// --------------------

namespace SystemIO
{
	// ファイルを丸読みする。存在しないなど、開けない場合はFileOpenErrorを返す。空行はスキップする。末尾の改行は除去される。
	// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
	// 引数で渡されるtrimはtrueを渡すと末尾のスペース、タブがトリムされる。
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	// 💡 filenameは、起動フォルダ相対で指定する。
	Tools::Result ReadAllLines(const std::filesystem::path& filename, std::vector<std::string>& lines, bool trim = false);
    inline Tools::Result ReadAllLines(const std::string& filename, std::vector<std::string>& lines, bool trim = false) {
        return ReadAllLines(path_from_utf8(filename), lines, trim);
    }
    inline Tools::Result ReadAllLines(const char* filename, std::vector<std::string>& lines, bool trim = false) {
        return ReadAllLines(path_from_utf8(filename), lines, trim);
    }

	// ファイルにすべての行を書き出す。
	// 💡 filenameは、起動フォルダ相対で指定する。
	Tools::Result WriteAllLines(const std::filesystem::path& filename, std::vector<std::string>& lines);
    inline Tools::Result WriteAllLines(const std::string& filename, std::vector<std::string>& lines) {
        return WriteAllLines(path_from_utf8(filename), lines);
    }
    inline Tools::Result WriteAllLines(const char* filename, std::vector<std::string>& lines) {
        return WriteAllLines(path_from_utf8(filename), lines);
    }


	// ファイルサイズを64bit対応の位置取得APIで調べ、1GB単位で読み込む。
	// ファイルサイズはsize_tで表現できる範囲に限る。
	//
	// read_file_to_memory()の引数のcallback_funcは、ファイルがオープン出来た時点でそのファイルサイズを引数として
	// callbackされるので、バッファを確保して、その先頭ポインタを返す関数を渡すと、そこに読み込んでくれる。
	//
	// また、callbackされた関数のなかでバッファが確保できなかった場合や、想定していたファイルサイズと異なった場合は、
	// nullptrを返せば良い。このとき、read_file_to_memory()は、読み込みを中断し、エラーリターンする。
	// 💡 filenameは、起動フォルダ相対で指定する。

	Tools::Result ReadFileToMemory(const std::filesystem::path& filename, std::function<void* (size_t)> callback_func);
    inline Tools::Result ReadFileToMemory(const std::string& filename, std::function<void* (size_t)> callback_func) {
        return ReadFileToMemory(path_from_utf8(filename), std::move(callback_func));
    }
    inline Tools::Result ReadFileToMemory(const char* filename, std::function<void* (size_t)> callback_func) {
        return ReadFileToMemory(path_from_utf8(filename), std::move(callback_func));
    }
	Tools::Result WriteMemoryToFile(const std::filesystem::path& filename, void* ptr, size_t size);
    inline Tools::Result WriteMemoryToFile(const std::string& filename, void* ptr, size_t size) {
        return WriteMemoryToFile(path_from_utf8(filename), ptr, size);
    }
    inline Tools::Result WriteMemoryToFile(const char* filename, void* ptr, size_t size) {
        return WriteMemoryToFile(path_from_utf8(filename), ptr, size);
    }

	// Windowsの通常のftell/fseekは32bitの位置を扱うため、64bit対応版を使う。

	size_t ftell64(FILE* f);
	int fseek64(FILE* f, size_t offset, int origin);

	// C#のTextReaderみたいなもの。
	// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
	// fopen()～fread()で実装されている。
	struct TextReader
	{
		TextReader();
		virtual ~TextReader();

		// ファイルをopenする。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::filesystem::path& filename);
        Tools::Result Open(const std::string& filename) { return Open(path_from_utf8(filename)); }
        Tools::Result Open(const char* filename) { return Open(path_from_utf8(filename)); }

		// Open()を呼び出してオープンしたファイルをクローズする。
		void Close();

		// ファイルの終了判定。
		// ファイルを最後まで読み込んだのなら、trueを返す。

		// 1行読み込む(改行まで) 引数のlineに代入される。
		// 改行コードは返さない。
		// SkipEmptyLine(),SetTrim()の設定を反映する。
		// Eofに達した場合は、返し値としてTools::ResultCode::Eofを返す。
		// 先頭のUTF-8のBOM(EF BB BF)は無視する。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result ReadLine(std::string& line);

		// ReadLine()で空行を読み飛ばすかどうかの設定。
		// (ここで行った設定はOpen()/Close()ではクリアされない。)
		// デフォルトでfalse
		void SkipEmptyLine(bool skip = true) { skipEmptyLine = skip; }

		// ReadLine()でtrimするかの設定。
		// 引数のtrimがtrueの時は、ReadLine()のときに末尾のスペース、タブはトリムする
		// (ここで行った設定はOpen()/Close()ではクリアされない。)
		// デフォルトでfalse
		void SetTrim(bool trim = true) { this->trim = trim; }

		// ファイルサイズの取得
		// ファイルポジションは先頭に移動する。
		size_t GetSize();

		// 現在のファイルポジションを取得する。
		// 先読みしているのでReadLineしている場所よりは先まで進んでいる。
		size_t GetFilePos() { return ftell64(fp); }

		// 現在の行数を返す。(次のReadLine()で返すのがテキストファイルの何行目であるかを返す) 0 origin。
		// またここで返す数値は空行で読み飛ばした時も、その空行を1行としてカウントしている。
		size_t GetLineNumber() const { return line_number; }

	private:
		// 各種状態変数の初期化
		void clear();

		// 次のblockのbufferへの読み込み。
		void read_next_block();

		// オープンしているファイル。
		// オープンしていなければnullptrが入っている。
		FILE* fp;

		// バッファから1文字読み込む。eofに達したら、-1を返す。
		int read_char();

		// ReadLineの下請け。何も考えずに1行読み込む。行のtrim、空行のskipなどなし。
		// line_bufferに読み込まれた行が代入される。
		Tools::Result read_line_simple();

		// ファイルの読み込みバッファ 1MB
		std::vector<u8> buffer;

		// 行バッファ
		std::vector<u8> line_buffer;

		// バッファに今回読み込まれたサイズ
		size_t read_size;

		// bufferの解析位置
		// 次のReadLine()でここから解析して1行返す
		// 次の文字 c = buffer[cursor]
		size_t cursor;

		// eofフラグ。
		// fp.eof()は、bufferにまだ未処理のデータが残っているかも知れないのでそちらを信じるわけにはいかない。
		bool is_eof;

		// 直前が\r(CR)だったのか？のフラグ
		bool is_prev_cr;

		// 何行目であるか
		// エラー表示の時などに用いる
		// 現在の行。(0 origin)
		size_t line_number;

		// ReadLine()で行の末尾をtrimするかのフラグ。
		bool trim;

		// ReadLine()で空行をskipするかのフラグ
		bool skipEmptyLine;
	};

	// Text書き出すの速いやつ。
	class TextWriter
	{
	public:
		// 書き出し用のバッファサイズ([byte])
		static constexpr size_t buf_size = 4096;

		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::filesystem::path& filename);
        Tools::Result Open(const std::string& filename) { return Open(path_from_utf8(filename)); }
        Tools::Result Open(const char* filename) { return Open(path_from_utf8(filename)); }

		// 文字列を書き出す(改行コードは書き出さない)
		Tools::Result Write(const std::string& str);

		// 1行を書き出す(改行コードも書き出す) 改行コードは"\r\n"とする。
		Tools::Result WriteLine(const std::string& line);

		// ptrの指すところからsize [byte]だけ書き出す。
		Tools::Result Write(const char* ptr, size_t size);

		// 内部バッファにあってまだファイルに書き出していないデータをファイルに書き出す。
		// ※　Close()する時に呼び出されるので通常この関数を呼び出す必要はない。
		Tools::Result Flush();

		Tools::Result Close();
		TextWriter() : buf(buf_size) { clear(); }
		virtual ~TextWriter() { Close(); }

	private:
		// 変数を初期化する。
		void clear() { fp = nullptr; write_cursor = 0; }

		FILE* fp = nullptr;

		// 書き出し用のbuffer。これがいっぱいになるごとにfwriteする。
		std::vector<char> buf;

		// 書き出し用のcursor。次に書き出す場所は、buf[write_cursor]。
		size_t write_cursor;
	};

	// BinaryReader,BinaryWriterの基底クラス
	class BinaryBase
	{
	public:
		// ファイルを閉じる。デストラクタからClose()は呼び出されるので明示的に閉じなくても良い。
		Tools::Result Close();

		virtual ~BinaryBase() { Close(); }

	protected:
		FILE* fp = nullptr;
	};

	// binary fileの読み込みお手伝いclass
	class BinaryReader : public BinaryBase
	{
	public:
		// ファイルのopen
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::filesystem::path& filename);
        Tools::Result Open(const std::string& filename) { return Open(path_from_utf8(filename)); }
        Tools::Result Open(const char* filename) { return Open(path_from_utf8(filename)); }

		// ファイルサイズの取得
		// ファイルポジションは先頭に移動する。
		size_t GetSize();

		// ptrの指すメモリにsize[byte]だけファイルから読み込む
		// ファイルの末尾を超えて読み込もうとした場合、Eofが返る。
		// ファイルの末尾に超えて読み込もうとしなかった場合、Okが返る。
		// 引数で渡されたバイト数読み込むことができなかった場合、FileReadErrorが返る。
		// size_of_read_bytesがnullptrでない場合、実際に読み込まれたバイト数が代入される。
		// ※　sizeは2GB制限があるので気をつけて。
		Tools::Result Read(void* ptr , size_t size, size_t* size_of_read_bytes = nullptr);
	};

	// binary fileの書き出しお手伝いclass
	class BinaryWriter : public BinaryBase
	{
	public:
		// ファイルのopen
		// append == trueで呼び出すと、このあとWriteしたものはファイル末尾に追記される。
		// 💡 filenameは、起動フォルダ相対で指定する。
		Tools::Result Open(const std::filesystem::path& filename, bool append = false);
        Tools::Result Open(const std::string& filename, bool append = false) { return Open(path_from_utf8(filename), append); }
        Tools::Result Open(const char* filename, bool append = false) { return Open(path_from_utf8(filename), append); }

		// ptrの指すメモリからsize[byte]だけファイルに書き込む。
		// ※　sizeは2GB制限があるので気をつけて。
		Tools::Result Write(void* ptr, size_t size);
	};
};

}

#endif
