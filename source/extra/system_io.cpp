#include "system_io.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

namespace YaneuraOu {

namespace fs = std::filesystem;

// --------------------
//  ファイル操作
// --------------------

namespace SystemIO
{
    namespace {
    // ネイティブのパスを保ったままFILE*を開く。modeはASCIIのモード指定。
    FILE* open_file(const fs::path& path, const char* mode) {
#ifdef _WIN32
        const std::wstring wide_mode(mode, mode + std::strlen(mode));
        return _wfopen(path.c_str(), wide_mode.c_str());
#else
        return std::fopen(path.c_str(), mode);
#endif
    }
    }


	// ファイルを丸読みする。存在しないなど、開けない場合はFileOpenErrorを返す。空行はスキップする。末尾の改行は除去される。
	// 引数で渡されるlinesは空であるを期待しているが、空でない場合は、そこに追加されていく。
	Tools::Result ReadAllLines(const fs::path& filename, std::vector<std::string>& lines, bool trim)
	{
#if 0
		std::ifstream fs(filename);
		if (fs.fail())
			return 1; // 読み込み失敗

		while (!fs.fail() && !fs.eof())
		{
			std::string line;
			Dependency::getline(fs, line);
			if (trim)
				line = StringExtension::trim(line);
			if (line.length())
				lines.push_back(line);
		}
		fs.close();

		return 0;
#endif

		// →　100MB程度のテキストファイルの読み込みにSurface Pro 6(Core i7モデル)で4秒程度かかる。
		// ifstreamを使わない形で書き直す。これで4倍ぐらい速くなる。

		TextReader reader;

		// ReadLine()時のトリムの設定を反映させる。
		reader.SetTrim(trim);
		// 空行をスキップするモードにする。
		reader.SkipEmptyLine(true);

		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		auto result = reader.Open(path);
		if (!result.is_ok())
			return result;

		std::string line;
		while (reader.ReadLine(line).is_ok())
			lines.emplace_back(line);

		return Tools::Result::Ok();
	}

	// ファイルにすべての行を書き出す。
	Tools::Result WriteAllLines(const fs::path& filename, std::vector<std::string>& lines)
	{
		TextWriter writer;

		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		if (writer.Open(path).is_not_ok())
			return Tools::ResultCode::FileOpenError;

		for(auto& line : lines)
		{
			if (writer.WriteLine(line).is_not_ok())
			return Tools::ResultCode::FileWriteError;
		}

		return Tools::ResultCode::Ok;
	}

    Tools::Result ReadFileToMemory(const fs::path& filename, std::function<void* (size_t)> callback_func)
    {
        // 起動フォルダ相対でのpath。途中で戻る場合もファイルを閉じる。
        const auto path = CommandLine::get_binary_directory() / filename;
        std::unique_ptr<FILE, decltype(&std::fclose)> fp(open_file(path, "rb"), &std::fclose);
        if (!fp)
            return Tools::ResultCode::FileOpenError;

        // Windowsでも2GBを超えるファイルのサイズを取得できるようにする。
        if (fseek64(fp.get(), 0, SEEK_END) != 0)
            return Tools::ResultCode::FileReadError;
        const size_t file_size = ftell64(fp.get());
        if (file_size == size_t(-1) || fseek64(fp.get(), 0, SEEK_SET) != 0)
            return Tools::ResultCode::FileReadError;

        // callbackでバッファを確保してもらう。nullptrなら読み込みを中断する。
        void* ptr = callback_func(file_size);
        if (ptr == nullptr)
            return Tools::ResultCode::MemoryAllocationError;

        // 1回のreadで読み込む要素の数を1GBに制限する。
        constexpr size_t block_size = 1024 * 1024 * 1024;
        for (size_t pos = 0; pos < file_size;)
        {
            const size_t read_size = std::min(block_size, file_size - pos);
            if (std::fread(static_cast<u8*>(ptr) + pos, 1, read_size, fp.get()) != read_size)
                return Tools::ResultCode::FileReadError;
            pos += read_size;
        }
        return Tools::Result::Ok();
    }


	Tools::Result WriteMemoryToFile(const fs::path& filename, void* ptr, size_t size)
	{
		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		std::fstream fs(path, std::ios::out | std::ios::binary);
		if (fs.fail())
			return Tools::Result(Tools::ResultCode::FileOpenError);

		const u64 block_size = 1024 * 1024 * 1024; // 1回のwriteで書き出す要素の数(1GB)
		for (u64 pos = 0; pos < size; pos += block_size)
		{
			// 今回書き出すメモリサイズ
			u64 write_size = (pos + block_size < size) ? block_size : (size - pos);
			fs.write((char*)ptr + pos, write_size);
			//cout << ".";

			if (fs.fail())
				return Tools::Result(Tools::ResultCode::FileWriteError); // ファイル書き込み時のエラー
		}

		// fstreamなので、これ不要だが..念の為にcloseする。
		fs.close();

		return Tools::Result::Ok();
	}

    // WindowsではMSVC/MinGWともに64bitのファイル位置APIを使う。
    size_t ftell64(FILE* f)
    {
#if defined(_WIN32)
        const auto pos = _ftelli64(f);
#elif defined(__GNUC__) && defined(IS_64BIT) && !(defined(__ANDROID__) && defined(__ANDROID_API__) && __ANDROID_API__ < 24) && !defined(__MACH__)
        const auto pos = ftello64(f);
#else
        const auto pos = std::ftell(f);
#endif
        // APIの失敗、またはsize_tで表現できない位置はエラーとして返す。
        if (pos < 0 || static_cast<std::uint64_t>(pos) > std::numeric_limits<size_t>::max())
            return size_t(-1);
        return static_cast<size_t>(pos);
    }

    int fseek64(FILE* f, size_t offset, int origin)
    {
#if defined(_WIN32)
        return _fseeki64(f, offset, origin);
#elif defined(__GNUC__) && defined(IS_64BIT) && !(defined(__ANDROID__) && defined(__ANDROID_API__) && __ANDROID_API__ < 24) && !defined(__MACH__)
        return fseeko64(f, offset, origin);
#else
        return std::fseek(f, offset, origin);
#endif
    }

	// --- TextFileReader

	// C++のifstreamが遅すぎるので、高速化されたテキストファイル読み込み器
	// fopen()～fread()で実装されている。
	TextReader::TextReader()
	{
		buffer.resize(1024 * 1024);
		line_buffer.reserve(2048);
		clear();

		// この２つのフラグはOpen()したときに設定がクリアされるべきではないので、
		// コンストラクタで一度だけ初期化する。
		trim = false;
		skipEmptyLine = false;
	}

	TextReader::~TextReader()
	{
		Close();
	}

	// 各種状態変数の初期化
	void TextReader::clear()
	{
		fp = nullptr;
		is_eof = false;
		cursor = 0;
		read_size = 0;
		is_prev_cr = false;
		line_number = 0;
	}

	// ファイルをopenする。
	Tools::Result TextReader::Open(const fs::path& filename)
	{
		Close();

		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		// 高速化のためにbinary open
		fp = open_file(path, "rb");
		return (fp == nullptr) ? Tools::Result(Tools::ResultCode::FileOpenError) : Tools::Result::Ok();
	}

	// Open()を呼び出してオープンしたファイルをクローズする。
	void TextReader::Close()
	{
		if (fp != nullptr)
			std::fclose(fp);

		clear();
	}

	// バッファから1文字読み込む。eofに達したら、-1を返す。
	int TextReader::read_char()
	{
		// ファイルからバッファの充填はこれ以上できなくて、バッファの末尾までcursorが進んでいるならeofという扱い。
		while (!(is_eof && cursor >= read_size))
		{
			if (cursor < read_size)
				return (int)buffer[cursor++];

			// カーソル(解析位置)が読み込みバッファを超えていたら次のブロックを読み込む。
			read_next_block();
		}
		return -1;
	}

	// ReadLineの下請け。何も考えずに1行読み込む。行のtrim、空行のskipなどなし。
	// line_bufferに読み込まれた行が代入される。
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	Tools::Result TextReader::read_line_simple()
	{
		// buffer[cursor]から読み込んでいく。
		// 改行コードに遭遇するとそこまでの文字列を返す。
		line_buffer.clear();

		/*
			改行コード一覧
				Unix        LF      \n
				Mac（OSX）  LF      \n
				Mac（OS9）  CR      \r
				Windows     CR+LF   \r\n

			ゆえに"\r","\n","\r\n"をすべて1つの改行コードとみなさないといけない。
			よって"\r"(CR)がきたときに次の"\n"(LF)は無視するという処理になる。
		*/

		while (true)
		{
			int c = read_char();
			if (c == -1 /* EOF */)
			{
				// line_bufferが空のままeofに遭遇したなら、eofとして扱う。
				// さもなくば、line_bufferを一度返す。(次回呼び出し時にeofとして扱う)
				if (line_buffer.size() == 0)
					return Tools::ResultCode::Eof;

				break;
			}

			if (c == '\r')
			{
				// 直前は"\r"だった。
				is_prev_cr = true;
				break;
			}

			// 直前は"\r"ではないことは確定したのでこの段階でis_prev_crフラグをクリアしておく。
			// ただし、このあと"\n"の判定の時に使うので古いほうの値をコピーして保持しておく。
			auto prev_cr = is_prev_cr;
			is_prev_cr = false;

			if (c == '\n')
			{
				if (!prev_cr)
					break;
				//else
				//   "\r\n"の(前回"\r"を処理した残りの)"\n"なので無視する。
			}
			else
			{
				// 行バッファに積んでいく。
				line_buffer.push_back(c);
			}
		}

		// 行バッファは完成した。
		// line_bufferに入っているのでこのまま使って問題なし。

		return Tools::ResultCode::Ok;
	}


	// 1行読み込む(改行まで)
	// 先頭のUTF-8のBOM(EF BB BF)は無視する。
	Tools::Result TextReader::ReadLine(std::string& line)
	{
		while (true)
		{
			if (read_line_simple().is_eof())
				return Tools::ResultCode::Eof;

			// trimフラグが立っているなら末尾スペース、タブを除去する。
			if (trim)
				while (line_buffer.size() > 0)
				{
					char c = *line_buffer.rbegin();
					if (!(c == ' ' || c == '\t'))
						break;

					line_buffer.resize(line_buffer.size() - 1);
				}

			// ファイル先頭のBOMは読み飛ばす
			size_t skip_byte = 0;
			if (line_number == 0)
				// UTF-8 BOM (EF BB BF)
				if (line_buffer.size() >= 3 && line_buffer[0] == 0xef && line_buffer[1] == 0xbb && line_buffer[2] == 0xbf)
					skip_byte = 3;
			// 他のBOMも読み飛ばしても良いが、まあいいや…。

			// この1行のbyte数(BOMは含まず)
			size_t line_size = line_buffer.size() - skip_byte;

			// この時点で1行読み込んだことになるので(行をskipしても1行とカウントするので)行番号をインクリメントしておく。
			line_number++;

			// 空行をスキップするモートであるなら、line_sizeが結果的に空になった場合は次の行を調べる。
			if (skipEmptyLine && line_size == 0)
				continue;

			line = std::string((const char*)line_buffer.data() + skip_byte, line_size );
			return Tools::ResultCode::Ok;
		}
	}

	// 次のblockのbufferへの読み込み。
	void TextReader::read_next_block()
	{
		if (::feof(fp))
			read_size = 0;
		else
			read_size = ::fread(&buffer[0], sizeof(u8), buffer.size(), fp);

		// カーソル(解析位置)のリセット
		cursor = 0;

		// 読み込まれたサイズが0なら、終端に達したと判定する。
		is_eof = read_size == 0;
	}

	// ファイルサイズの取得
	// ファイルポジションは先頭に移動する。
	size_t TextReader::GetSize()
	{
		ASSERT_LV3(fp != nullptr);

		fseek64(fp, 0, SEEK_END);
		// ftell()は失敗した時に-1を返すらしいのだが…。ここでは失敗を想定していない。
		size_t endPos = ftell64(fp);
		fseek64(fp, 0, SEEK_SET);
		size_t beginPos = ftell64(fp);
		size_t file_size = endPos - beginPos;

		return file_size;
	}

	// === TextWriter ===

	Tools::Result TextWriter::Open(const fs::path& filename)
	{
		Close();

		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		fp = open_file(path, "wb");
		return fp == nullptr ? Tools::ResultCode::FileOpenError
                             : Tools::ResultCode::Ok;
	}

	// 文字列を書き出す(改行コードは書き出さない)
	Tools::Result TextWriter::Write(const std::string& str)
	{
		return Write(str.c_str(), str.size());
	}

	// 1行を書き出す(改行コードも書き出す) 改行コードは"\r\n"とする。
	Tools::Result TextWriter::WriteLine(const std::string& line)
	{
		auto result = Write(line.c_str(), line.size());
		if (result.is_not_ok())
			return result;

		// 改行コードも書き出す。
		return Write("\r\n", (size_t)2);
	}

	// ptrの指すところからsize [byte]だけ書き出す。
	Tools::Result TextWriter::Write(const char* ptr, size_t size)
	{
		// Openしていなければ書き出せない。
		if (fp == nullptr)
			return Tools::ResultCode::FileWriteError;

		// 書き込みカーソルの終端がどこに来るのか。
		size_t write_cursor_end = write_cursor + size;
		char* ptr2 = const_cast<char*>(ptr);

		size_t write_size;
		while (write_cursor_end >= buf_size)
		{
			// とりあえず、書けるだけ書いてfwriteする。

			// 今回のループで書き込むbyte数
			write_size = buf_size - write_cursor;
			std::memcpy(&buf[write_cursor], ptr2, write_size);
			if (std::fwrite(buf.data(), buf_size, 1, fp) == 0)
				return Tools::ResultCode::FileWriteError;

			// buf[0..write_cursor-1]が窓で、ループごとにその窓がbuf_sizeずつずれていくと考える。
			// 例えば、ループ2回目ならbuf[write_cursor..write_cursor*2-1]が窓だと考える。

			ptr2             += write_size;
			size             -= write_size;
			write_cursor_end -= buf_size;
			write_cursor      = 0;
		}
		std::memcpy(&buf[write_cursor], ptr2, size);
		write_cursor += size;

		return Tools::ResultCode::Ok;
	}

	// 内部バッファにあってまだファイルに書き出していないデータをファイルに書き出す。
	// ※　Close()する時に呼び出されるので通常この関数を呼び出す必要はない。
	Tools::Result TextWriter::Flush()
	{
		// Openしていなければ書き出せない。
		if (fp == nullptr)
			return Tools::ResultCode::FileWriteError;

		// bufのwrite_cursorの指している手前までを書き出す。
		if (write_cursor > 0 && std::fwrite(&buf[0], write_cursor, 1, fp) == 0)
			return Tools::ResultCode::FileWriteError;

		write_cursor = 0;
		return Tools::ResultCode::Ok;
	}

	Tools::Result TextWriter::Close()
	{
		if (fp)
		{
			// バッファ、まだflushが終わっていないデータがあるならそれをflushする。
			if (Flush().is_not_ok())
				return Tools::ResultCode::FileWriteError;

			std::fclose(fp); // GetLastErrorでエラーを取得することはできるが…。
			fp = nullptr;
		}
		return Tools::ResultCode::Ok;
	}

	// === BinaryBase ===

	// ファイルを閉じる。デストラクタからclose()は呼び出されるので明示的に閉じなくても良い。
	Tools::Result BinaryBase::Close()
	{
		Tools::ResultCode result = Tools::ResultCode::Ok;
		if (fp != nullptr)
		{
			if (std::fclose(fp) != 0)
				result = Tools::ResultCode::FileCloseError;
			fp = nullptr;
		}
		return Tools::Result(result);
	}

	// === BinaryReader ===

	// ファイルのopen
	Tools::Result BinaryReader::Open(const fs::path& filename)
	{
		auto close_result = Close();
		if (!close_result.is_ok()) {
			return close_result;
		}

		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		fp = open_file(path, "rb");
		if (fp == nullptr)
			return Tools::Result(Tools::ResultCode::FileOpenError);

		return Tools::Result::Ok();
	}

	// ファイルサイズの取得
	// ファイルポジションは先頭に移動する。
	size_t BinaryReader::GetSize()
	{
		ASSERT_LV3(fp != nullptr);

		fseek64(fp, 0, SEEK_END);
		size_t endPos = ftell64(fp);
		fseek64(fp, 0, SEEK_SET);
		size_t beginPos = ftell64(fp);
		size_t file_size = endPos - beginPos;

		return file_size;
	}

	// ptrの指すメモリにsize[byte]だけファイルから読み込む
	Tools::Result BinaryReader::Read(void* ptr, size_t size, size_t* size_of_read_bytes)
	{
		size_t actual_size_of_read_bytes = std::fread(ptr, 1, size, fp);

		if (size_of_read_bytes) {
			*size_of_read_bytes = actual_size_of_read_bytes;
		}

		if (std::feof(fp)) {
			// ファイルの末尾を超えて読もうとした場合。
			return Tools::Result(Tools::ResultCode::Eof);
		}

		if (actual_size_of_read_bytes != size)
			return Tools::Result(Tools::ResultCode::FileReadError);

		// ファイルの末尾を超えて読もうとしなかった場合。
		// ファイルの末尾までちょうどを読んだ場合はこちら。
		return Tools::Result::Ok();
	}

	// === BinaryWriter ===

	// ファイルのopen
	Tools::Result BinaryWriter::Open(const fs::path& filename, bool append)
	{
		// 起動フォルダ相対でのpath
		const auto path = CommandLine::get_binary_directory() / filename;

		fp = open_file(path, append ? "ab" : "wb");
		if (fp == nullptr)
			return Tools::Result(Tools::ResultCode::FileOpenError);

		return Tools::Result::Ok();
	}

	// ptrの指すメモリからsize[byte]だけファイルに書き込む
	Tools::Result BinaryWriter::Write(void* ptr, size_t size)
	{
		if (std::fwrite((u8*)ptr, 1, size, fp) != size)
			return Tools::Result(Tools::ResultCode::FileWriteError);

		return Tools::Result::Ok();
	}
}

}
