#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD)

// ------------------------------
//    ペタショック化コマンド
// ------------------------------

// 定跡自体は、別の何らかのスクリプトによって、やねうら王形式の定跡ファイルが書き出されているものとする。
// その定跡ファイルに対してそれぞれの局面に対してmin-max探索を行った結果の定跡ファイルを書き出すのが、
// このペタショック化である。

// コマンド例)
//    makebook peta_shock book1.db user_book1.db
// 
// book1.dbをmin-max探索してuser_book1.dbを書き出す。
// 
// エンジンオプションのFlippedBookがtrueなら、先手番の局面しか書き出さない。(後手番の局面はそれをflipした局面が書き出されているはずだから)
//
// 値は逆辺を使って単調に確定・伝播し、反復処理は先後別の王手循環部分に限定する。
// 選択手が通常千日手や連続王手によって保証値を破られないか検査した後、保存時だけ同点手を補正する。
// 値の計算と保証検査の実装はpeta_shock_solver.hを参照。

#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <limits>
#include <random>
#include <utility> // For std::forward
#include <new>

#include "book.h"
#include "peta_shock_solver.h"
#include "../thread.h"
#include "../position.h"
#include "../movegen.h"
#include "../misc.h"

using namespace std;
namespace YaneuraOu {

// ある局面の指し手の配列、std::vector<Move>だとsize_t(64bit環境で8バイト)でcapacityとかsizeとか格納するので
// 非常にもったいない。定跡のある局面の指し手が255手を超えることはないだろうから1byteでも十分。
// そこで、size_tではなくint16_tでサイズなどを保持しているvectorのsubsetを用意する。
// dataポインタが8バイトなのももったいないが…。これは仕方がないか…。

template <typename T>
class SmallVector {
public:
	SmallVector() noexcept : data(nullptr), count(0), capacity(0) {

		// 定跡の指し手、MultiPVで探索していて、
		// 4の倍数なのでcapacityは4を初期値にしておく。
		reserve(4);
	}

	// コピーコンストラクタ
	SmallVector(const SmallVector& other)
		: data(nullptr), count(0), capacity(0) {
		if (other.count > 0) {
			reserve(other.count);
			for (uint16_t i = 0; i < other.count; ++i) {
				new (&data[i]) T(other.data[i]);
			}
			count = other.count;
		}
	}

	// コピー代入演算子
	SmallVector& operator=(const SmallVector& other) {
		if (this != &other) {
			clear();
			release();
			if (other.count > 0) {
				reserve(other.count);
				for (uint16_t i = 0; i < other.count; ++i) {
					new (&data[i]) T(other.data[i]);
				}
				count = other.count;
			}
		}
		return *this;
	}

	// ムーブコンストラクタ
	SmallVector(SmallVector&& other) noexcept
		: data(nullptr), count(0), capacity(0) {
		swap(other);
	}

	// ムーブ代入演算子
	SmallVector& operator=(SmallVector&& other) noexcept {
		if (this != &other) {
			clear();
			release();
			swap(other);
		}
		return *this;
	}

	~SmallVector() noexcept {
		clear();
		release();
	}

	void push_back(const T& value) {
		emplace_back(value);
	}

	template <typename... Args>
	void emplace_back(Args&&... args) {
		if (count == capacity) {
			increase_capacity();
		}
		new (data + count) T(std::forward<Args>(args)...);
		++count;
	}

	void erase(T* position) {
		position->~T();
		for (T* it = position; it != data + count - 1; ++it) {
			new (it) T(std::move(*(it + 1)));
			(it + 1)->~T();
		}
		--count;
	}

	T& operator[](size_t idx) {
		return data[idx];
	}

	const T& operator[](size_t idx) const {
		return data[idx];
	}

	size_t size() const noexcept {
		return count;
	}

	T* begin() noexcept { return data; }
	T* end() noexcept { return data + count; }
	const T* begin() const noexcept { return data; }
	const T* end() const noexcept { return data + count; }

	void clear() noexcept {
		for (uint16_t i = 0; i < count; ++i) {
			data[i].~T();
		}
		count = 0;
	}

	void swap(SmallVector& other) noexcept {
		using std::swap;
		swap(data, other.data);
		swap(count, other.count);
		swap(capacity, other.capacity);
	}

private:
	void increase_capacity() {
		size_t new_capacity = capacity == 0 ? 1 : capacity * 2;
		if (new_capacity > UINT16_MAX) {
			std::cout << "Error! : SmallVector exceeds maximum capacity" << std::endl;
			Tools::exit();
		}
		reserve(static_cast<uint16_t>(new_capacity));
	}

	void reserve(uint16_t new_capacity) {
		T* new_data = reinterpret_cast<T*>(new char[new_capacity * sizeof(T)]);
		for (uint16_t i = 0; i < count; ++i) {
			new (&new_data[i]) T(std::move(data[i]));
			data[i].~T();
		}
		release();
		data = new_data;
		capacity = new_capacity;
	}

	void release() noexcept {
		delete[] reinterpret_cast<char*>(data);
		data = nullptr;
		capacity = 0;
	}

	T* data;
	uint16_t count;
	uint16_t capacity;
};

// ADLを利用したswapの非メンバ関数バージョン
// SmallVectorの名前空間内で定義する
template <typename T>
void swap(SmallVector<T>& a, SmallVector<T>& b) noexcept {
    a.swap(b);
}

namespace MakeBook2025
{
	static_assert(Book::PetaRetrograde::Mate == VALUE_MATE);
	// peta_shockコマンド実行時にsfen文字列を一時保存するファイル名
	string SFEN_TEMP_FILENAME = "sfen_tmp.txt";

	// BookMoveのポインターみたいなやつ。これで局面の行き来を行う。
	// しばらくは42億を超えることはないと思うので32bitでいいや。
	typedef u32 BookNodeIndex;
	// ペタショック前の定跡DBに指し手の評価値をminにして書き出しておくと、
	// これは指し手は存在するけど評価値は不明の指し手である。(という約束にする)
	// これは棋譜の指し手などを定跡DBに登録する時に評価値が確定しないのでそういう時に用いる。
	const int BOOK_VALUE_NONE = numeric_limits<s16>::min();

	// MATEより大きなスコアを使うと、表示のときにバグる可能性がある。
	const int BOOK_VALUE_MAX  =  VALUE_MATE;
	const int BOOK_VALUE_MIN  = -VALUE_MATE;

	// 定跡で千日手手順の時のdepth。∞であることがわかる特徴的な定数にしておくといいと思う。
	const u16 BOOK_DEPTH_MAX = 9999;
	static constexpr std::array<char, 16> YBB_MAGIC = {
		'Y', 'A', 'N', 'E', '-', 'B', 'I', 'N',
		'B', 'O', 'O', 'K', '-', 'V', '1', '\0',
	};
	static constexpr u64 YBB_HEADER_SIZE = 32;
	static constexpr u64 YBB_INDEX_RECORD_SIZE = 44;
	static constexpr u64 YBB_FLAG_MOVE_DEPTH = 1;
	static constexpr u64 YBB_KNOWN_FLAGS = YBB_FLAG_MOVE_DEPTH;

	struct YbbIndexEntry
	{
		PackedSfen packed_sfen{};
		u64 moves_offset = 0;
		u16 ply = 0;
		u16 move_count = 0;
	};

	static bool ends_with(const string& text, const string& suffix)
	{
		return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	static bool is_ybb_book(const string& filename)
	{
		return ends_with(filename, ".ybb");
	}

	static bool is_db_book(const string& filename)
	{
		return ends_with(filename, ".db");
	}

	static string ybb_book_name_from_db_name(const string& db_filename)
	{
		if (!ends_with(db_filename, ".db"))
			return string();
		string ybb_filename = db_filename;
		ybb_filename.resize(ybb_filename.size() - string(".db").size());
		ybb_filename += ".ybb";
		return ybb_filename;
	}

	static string resolve_book_filename_with_ybb_fallback(const string& filename)
	{
		if (Path::Exists(filename))
			return filename;

		const auto ybb_filename = ybb_book_name_from_db_name(filename);
		if (!ybb_filename.empty() && Path::Exists(ybb_filename))
			return ybb_filename;

		return filename;
	}

	static bool read_u16_le(istream& is, u16& value)
	{
		array<unsigned char, 2> bytes{};
		is.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
		if (!is)
			return false;
		value = u16(bytes[0] | (bytes[1] << 8));
		return true;
	}

	static bool read_u64_le(istream& is, u64& value)
	{
		array<unsigned char, 8> bytes{};
		is.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
		if (!is)
			return false;
		value = 0;
		for (int i = 7; i >= 0; --i)
		{
			value <<= 8;
			value |= bytes[size_t(i)];
		}
		return true;
	}

	static bool write_u16_le(ostream& os, u16 value)
	{
		array<unsigned char, 2> bytes{
			static_cast<unsigned char>(value & 0xff),
			static_cast<unsigned char>((value >> 8) & 0xff),
		};
		os.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		return bool(os);
	}

	static bool write_u64_le(ostream& os, u64 value)
	{
		array<unsigned char, 8> bytes{};
		for (size_t i = 0; i < bytes.size(); ++i)
		{
			bytes[i] = static_cast<unsigned char>(value & 0xff);
			value >>= 8;
		}
		os.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		return bool(os);
	}

	static bool read_ybb_header(istream& is, u64& record_count, u64& flags)
	{
		array<char, 16> magic{};
		is.read(magic.data(), magic.size());
		if (!is || magic != YBB_MAGIC)
			return false;
		if (!read_u64_le(is, record_count))
			return false;
		if (!read_u64_le(is, flags))
			return false;
		return (flags & ~YBB_KNOWN_FLAGS) == 0;
	}

	static bool ybb_index_size(u64 record_count, u64& index_size)
	{
		if (record_count > (std::numeric_limits<u64>::max() - YBB_HEADER_SIZE) / YBB_INDEX_RECORD_SIZE)
			return false;
		index_size = YBB_HEADER_SIZE + record_count * YBB_INDEX_RECORD_SIZE;
		return true;
	}

	static bool read_ybb_index_entry(istream& is, YbbIndexEntry& entry)
	{
		is.read(reinterpret_cast<char*>(entry.packed_sfen.data), 32);
		if (!is)
			return false;
		if (!read_u64_le(is, entry.moves_offset))
			return false;
		if (!read_u16_le(is, entry.ply))
			return false;
		if (!read_u16_le(is, entry.move_count))
			return false;
		return true;
	}

	static bool write_ybb_header(ostream& os, u64 record_count, u64 flags)
	{
		os.write(YBB_MAGIC.data(), YBB_MAGIC.size());
		return bool(os) && write_u64_le(os, record_count) && write_u64_le(os, flags);
	}

	static bool write_ybb_index_entry(ostream& os, const YbbIndexEntry& entry)
	{
		os.write(reinterpret_cast<const char*>(entry.packed_sfen.data), 32);
		return bool(os)
			&& write_u64_le(os, entry.moves_offset)
			&& write_u16_le(os, entry.ply)
			&& write_u16_le(os, entry.move_count);
	}

	// 定跡の評価値とその時のdepthをひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0) {}

		ValueDepth(s16 value, u16 depth)
			: value(value) , depth(depth) {}

		s16 value;
		// depthは最大値 BOOK_DEPTH_MAXまでなので2bitほど余ってはいるか…。
		u16 depth;
	};

	std::ostream& operator<<(std::ostream& os, ValueDepth vd)
	{
		os << "( " << vd.value << ", " << vd.depth << ")";
		return os;
	}

	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	// 📝 : immutableにしたいのだが、メンバ変数にconstをつけるとコピー/代入ができなくて困る。
	struct BookMove
	{
		// moveの指し手がleaf nodeである場合。
		BookMove(Move16 move, s16 value, u16 depth) :
			move(move), vd(ValueDepth(value, depth)), leaf(true) {
		}

		// moveの指し手がleaf nodeである場合。
		BookMove(Move16 move, ValueDepth vd):
			move(move), vd(vd), leaf(true) { }

		// moveの指し手がleaf nodeではない場合。
		BookMove(Move16 move, BookNodeIndex next) :
			move(move), next(next), leaf(false) {
		}

		// 指し手
		Move16 move;

		// ここがleaf nodeであるか？
		bool leaf;

		union
		{
			// 1. leaf_node == trueの時。
			//    moveを選んだ時の評価値
			ValueDepth vd;

			// 2. leaf_node == falseの時。
			//	  moveで局面を進めた時の次の局面のBookMoveポインター。
			BookNodeIndex next;
		};

		// move(2) + leaf_node(1) + { ValueDepth(4) or next(4) } = 7 bytes → 8
	};

	// 定跡の1つの局面を表現する構造体。
	// 高速化のために、BookNodeIndexで行き来をする。
	struct BookNode
	{
		// この局面での指し手
		// 📝 : read_book(), convergence_check()で初期化している。
		// ⚠ この局面は先手化された局面であるから、movesは先手番としての指し手
		SmallVector<BookMove> moves;

		// 親の指し手から参照する値。自身の最善値の符号反転と、選択手のdepth+1を保持する。
		ValueDepth vd;

		// 棋譜に出現したこの局面の手番。(書き出す時にこれを再現する)
		// 📝 : read_book()で初期化している。
		u8 color;

		// この局面で王手されているか
		// 📝 : read_book()で初期化している。
		bool checked;

	};

	// 正規化した1局面を、実際の手番が先手/後手の2状態としてSolverへ見せる。
	// 状態番号は2*局面番号+手番。指し手と逆辺配列は先後で共有し、定跡全体を二重には持たない。
	struct PetaGraph
	{
		using Index = Book::PetaRetrograde::Index;
		const vector<BookNode>& nodes;
		vector<size_t> offsets; // 局面iの逆辺はpredecessors[offsets[i]..offsets[i+1])。
		vector<pair<Index, Index>> predecessors; // (親局面番号, 親の指し手番号)を連続領域に格納するCSR形式。
		explicit PetaGraph(const vector<BookNode>& nodes) : nodes(nodes), offsets(nodes.size() + 1, 0)
		{
			// 入次数の累積和で格納範囲を確保し、各内部手を一度ずつ登録する。leafには逆辺を作らない。
			for (const auto& node : nodes)
				for (const auto& move : node.moves)
					if (!move.leaf) ++offsets[size_t(move.next) + 1];
			for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i-1];
			predecessors.resize(offsets.back());
			auto cursor = offsets;
			for (Index v = 0; v < nodes.size(); ++v)
				for (Index j = 0; j < nodes[v].moves.size(); ++j)
					if (!nodes[v].moves[j].leaf)
						predecessors[cursor[nodes[v].moves[j].next]++] = {v, j};
		}
		Index size() const { return Index(nodes.size() * 2); }
		int owner(Index v) const { return int(v & 1); }
		bool checked(Index v) const { return nodes[v / 2].checked; }
		Index degree(Index v) const { return Index(nodes[v / 2].moves.size()); }
		Index child(Index v, Index j) const
		{
			const auto& move = nodes[v / 2].moves[j];
			// 内部手で1手進むたびに実際の手番も反転する。Noneは定跡外への評価値付きの手。
			return move.leaf ? Book::PetaRetrograde::None : Index(move.next * 2 + (1 - (v & 1)));
		}
		int score(Index v, Index j) const { return nodes[v / 2].moves[j].vd.value; }
		template<class F> void parents(Index v, F f) const
		{
			// 共有している逆辺へ手番情報を付け直す。訪問管理と組み合わせれば再合流で経路数が膨らまない。
			for (auto i = offsets[v / 2]; i < offsets[v / 2 + 1]; ++i)
				f(predecessors[i].first * 2 + (1 - (v & 1)), predecessors[i].second);
		}
	};

	// ペタショック化
	class PetaShock
	{
	public:
		// 定跡をペタショック化する。
		void make_book(istringstream& is, string book_dir)
		{
			// 初期化等
			if (initialize(is, book_dir).is_not_ok())
				return;

			// ペタショック化する定跡ファイルの読み込み
			if (read_book().is_not_ok())
				return;

			// 局面の合流チェック
			if (convergence_check(thread_count).is_not_ok())
				return;

			if (!solve_repetition_game())
				return;

			// ペタショック化した定跡の書き出し
			if (write_peta_shock_book(writebook_path, book_nodes).is_not_ok())
				return;

			// 結果出力
			output_result();

			if (!fast)
				std::remove(sfen_temp_path.c_str());
		}

	protected:

		// === helper function ===

		// 初期化
		Tools::Result initialize(istringstream& is, string book_dir)
		{
			// hashkeyのbit数をチェックして、128bit未満であれば警告を出す。

			// 高速化のために、HASH_KEYだけで局面を処理したいので、hashの衝突はあってはならない。
			// そのため、HASH_KEY_BITSは128か256が望ましい。
			if (HASH_KEY_BITS < 128)
			{
				std::cout << "WARNING! : HASH_KEY_BITS = " << HASH_KEY_BITS << " is too short." << endl;
				std::cout << "    Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
			}

			is >> readbook_path >> writebook_path;
			if (readbook_path.empty() || writebook_path.empty())
			{
				cout << "Error! : makebook peta_shock requires readbook and writebook paths." << endl;
				return Tools::ResultCode::FileMismatch;
			}

			// コマンドラインオプションの読み込み
			string token;
			shrink = fast = false;
			thread_count = 1;
			while (is >> token)
			{
				if (token == "shrink")
					shrink = true;

				else if (token == "fast")
					fast = true;

				else if (token == "threads")
				{
					// USIのThreadsとは独立した合流探索の並列数。重い読込の前に欠落・0・桁あふれなどを拒否する。
					string value;
					is >> value;
					const auto parsed = std::from_chars(value.data(), value.data() + value.size(), thread_count);
					if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || thread_count == 0)
					{
						cout << "Error! : makebook peta_shock threads requires a positive integer." << endl;
						return Tools::ResultCode::FileMismatch;
					}
				}
			}

			string BOOK_DIR = book_dir;

			readbook_path  = Path::Combine(BOOK_DIR, readbook_path);
			writebook_path = Path::Combine(BOOK_DIR, writebook_path);
			sfen_temp_path = writebook_path + "." + SFEN_TEMP_FILENAME;

			const auto actual_readbook_path = resolve_book_filename_with_ybb_fallback(readbook_path);
			if (actual_readbook_path != readbook_path)
			{
				cout << "readbook fallback   : " << readbook_path << " -> " << actual_readbook_path << endl;
				readbook_path = actual_readbook_path;
			}

			const bool read_is_db = is_db_book(readbook_path);
			const bool write_is_db = is_db_book(writebook_path);
			const bool read_is_ybb = is_ybb_book(readbook_path);
			const bool write_is_ybb = is_ybb_book(writebook_path);
			if (!((read_is_db && write_is_db) || (read_is_ybb && write_is_ybb)))
			{
				cout << "Error! : makebook peta_shock supports only .db -> .db or .ybb -> .ybb." << endl;
				cout << "        readbook_path  : " << readbook_path << endl;
				cout << "        writebook_path : " << writebook_path << endl;
				return Tools::ResultCode::FileMismatch;
			}

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			cout << "readbook_path      : " << readbook_path << endl;
			cout << "writebook_path     : " << writebook_path << endl;

			cout << "shrink             : " << shrink << endl;
			cout << "fast               : " << fast << endl;
			cout << "threads            : " << thread_count << endl;

			/*
				note: DrawValueの変更について。

				引き分けのスコアを0から変更するには、各局面について、初期局面が{ 先手, 後手 }の2通りを持つ必要がある。
				ここでは、その処理をしていないので、DrawValueの変更した場合の動作は未定義。
			*/

			drawValueTable[REPETITION_DRAW][BLACK] = 0;
			drawValueTable[REPETITION_DRAW][WHITE] = 0;

			// 統計値の初期化

			converged_moves    = 0;
			finite_depth_nodes = 0;
			in_check_counter   = 0;

			// 各種配列のクリア

			original_sfens.clear();

			return Tools::Result::Ok();
		}

		// ペタショック化前の定跡ファイルの読み込み。
		Tools::Result read_book()
		{
			// progress表示用
			Tools::ProgressBar progress;

			cout << "Read a book DB      : " << endl;

			// MemoryBookに読み込むと時間かかる + メモリ消費量が大きくなるので
			// 直接自前でbook_nodesに読み込む。

			if (is_ybb_book(readbook_path))
			{
				ifstream index_reader(readbook_path, ios::binary);
				if (!index_reader)
				{
					sync_cout << "info string Error! : can't read file : " + readbook_path << sync_endl;
					return Tools::ResultCode::FileNotFound;
				}
				ifstream moves_reader(readbook_path, ios::binary);
				if (!moves_reader)
				{
					sync_cout << "info string Error! : can't read file : " + readbook_path << sync_endl;
					return Tools::ResultCode::FileNotFound;
				}

				u64 noe = 0;
				u64 ybb_flags = 0;
				if (!read_ybb_header(index_reader, noe, ybb_flags))
				{
					sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
					return Tools::ResultCode::FileReadError;
				}
				u64 moves_base = 0;
				if (!ybb_index_size(noe, moves_base))
				{
					sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
					return Tools::ResultCode::FileReadError;
				}

				cout << "Number Of Elements : " << noe << endl;
				book_nodes.reserve(size_t(noe));
				hashkey_to_index.reserve(size_t(noe));
				if (fast)
					original_sfens.reserve(size_t(noe));

				SystemIO::TextWriter sfen_writer;
				if (!fast)
				{
					if (sfen_writer.Open(sfen_temp_path).is_not_ok())
					{
						sync_cout << "info string Error! : can't write file : " + sfen_temp_path << sync_endl;
						return Tools::ResultCode::FileOpenError;
					}
				}

				progress.reset(noe == 0 ? 0 : noe - 1);
				Position pos;

				for (u64 i = 0; i < noe; ++i)
				{
					progress.check(i);

					YbbIndexEntry entry;
					if (!read_ybb_index_entry(index_reader, entry))
					{
						if (!fast)
							sfen_writer.Close();
						sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
						return Tools::ResultCode::FileReadError;
					}
					string sfen = Position::sfen_unpack(entry.packed_sfen);
					StringExtension::trim_number_inplace(sfen);
					sfen += " " + std::to_string(entry.ply);

					if (fast)
						original_sfens.push_back(sfen);
					else
						sfen_writer.WriteLine(sfen);

					StringExtension::trim_number_inplace(sfen);
					Color stm = (sfen.find('w') != std::string::npos) ? WHITE : BLACK;
					string white_sfen = stm == WHITE ? sfen : Position::sfen_to_flipped_sfen(sfen);

					StateInfo si;
					pos.set(white_sfen, &si);
					Key white_hash_key = pos.key();

					auto book_node_index = BookNodeIndex(book_nodes.size());
					hashkey_to_index[white_hash_key] = book_node_index;

					bool checked = pos.checkers();
					book_nodes.emplace_back(BookNode());
					auto& book_node = book_nodes.back();

					book_node.color       = stm;
					book_node.checked     = checked;
					in_check_counter     += checked;

					moves_reader.clear();
					moves_reader.seekg(std::streamoff(moves_base + entry.moves_offset), ios::beg);
					if (!moves_reader)
					{
						if (!fast)
							sfen_writer.Close();
						sync_cout << "info string Error! : invalid ybb moves area : " << readbook_path << sync_endl;
						return Tools::ResultCode::FileReadError;
					}

					for (u16 move_index = 0; move_index < entry.move_count; ++move_index)
					{
						u16 move16_value = 0;
						u16 eval_value = 0;
						if (!read_u16_le(moves_reader, move16_value) || !read_u16_le(moves_reader, eval_value))
						{
							if (!fast)
								sfen_writer.Close();
							sync_cout << "info string Error! : invalid ybb moves area : " << readbook_path << sync_endl;
							return Tools::ResultCode::FileReadError;
						}
						if (ybb_flags & YBB_FLAG_MOVE_DEPTH)
						{
							// depthフィールドは読み飛ばすが、入力値は後退解析に使用しない。
							u16 ignored_depth;
							if (!read_u16_le(moves_reader, ignored_depth))
							{
								if (!fast)
									sfen_writer.Close();
								sync_cout << "info string Error! : invalid ybb moves area : " << readbook_path << sync_endl;
								return Tools::ResultCode::FileReadError;
							}
						}
						Move16 move16 = Move16(move16_value);
						auto value = (s16)std::clamp((int)(s16)eval_value, BOOK_VALUE_MIN, BOOK_VALUE_MAX);

						if (book_node.color == WHITE)
							move16 = flip_move(move16);

						// 入力側の指し手はleaf候補なので、入力depthを引き継がず0として後退解析する。
						book_node.moves.emplace_back(BookMove(move16, value, 0));
					}
				}

				if (!fast)
					sfen_writer.Close();

				return Tools::Result::Ok();
			}

			SystemIO::TextReader reader;
			// ReadLine()の時に行の末尾のスペース、タブを自動トリム。空行は自動スキップ。
			reader.SetTrim(true);
			reader.SkipEmptyLine(true);

			auto result = reader.Open(readbook_path);
			if (result.is_not_ok())
			{
				sync_cout << "info string Error! : can't read file : " + readbook_path << sync_endl;
				return Tools::ResultCode::FileNotFound;
			}

			// 行バッファ
			std::string line;

			// headerとnoeを調べる。
			while (reader.ReadLine(line).is_ok())
			{
				auto line_no = reader.GetLineNumber();
				if (line_no == 1)
				{
					// バージョン識別文字列がなければwarningを出す。
					if (line.length() < 1 || line[0] != '#' || line != YaneuraOu::Book::BookDBHeader2016_100)
						cout << "WARNING : illegal YaneuraOu Book header 2016" << endl;
				}
				else if (line_no == 2)
				{
					// 2行目には
					// # NOE:258
					// よって、この文字列をparseする必要がある。
					if (line.length() >= 1 && line[0] == '#')
					{
						// いったん変数に入れておかないと、temporary objectのstring_viewを作ることになってそのあと不正になる。
						auto line2 = line.substr(2);
						auto splited = StringExtension::Split(line2, ",");
						for (auto command : splited)
						{
							auto splited2 = StringExtension::Split(command, ":");
							if (splited2.size() >= 1)
							{
								auto& token = splited2[0];
								if (token == "NOE" && splited2.size() == 2) // numbers of entires
								{
									size_t noe = StringExtension::to_int(string(splited2[1]), 0);
									cout << "Number Of Elements : " << noe << endl;

									// エントリー数が事前にわかったので、その分だけそれぞれの構造体配列を確保する。
									book_nodes.reserve(noe);
									hashkey_to_index.reserve(noe);
									if (fast)
										original_sfens.reserve(noe);
								}
							}
						}
					}
				}
				else
					break;
			}
			reader.Close();
			reader.Open(readbook_path);

			progress.reset(reader.GetSize() - 1);

			// sfen文字列はファイルに書き出す。
			SystemIO::TextWriter sfen_writer;
			if (!fast)
			{
				if (sfen_writer.Open(sfen_temp_path).is_not_ok())
				{
					sync_cout << "info string Error! : can't write file : " + sfen_temp_path << sync_endl;
					return Tools::ResultCode::FileOpenError;
				}
			}

			Position pos;

			while (reader.ReadLine(line).is_ok())
			{
				progress.check(reader.GetFilePos());

				// #で始まる行は読み飛ばす。
				if (line.length() >= 1 && line[0] == '#')
					continue;

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					string sfen = line.substr(5); // 新しいsfen文字列を"sfen "を除去して格納

					// sfen文字列はテンポラリファイルに書き出しておく。(もし末尾に手数があるなら、それも含めてそのまま書き出す)
					// あとで局面を書き出す時に用いる。
					if (fast)
						original_sfens.push_back(sfen);
					else
						sfen_writer.WriteLine(sfen);

					// 末尾のplyは使わないので除去。
					StringExtension::trim_number_inplace(sfen);

					// この局面の(元の)手番
					// ⇨ "w"の文字は駒には使わないので"w"があれば後手番であることが確定する。
					Color stm = (sfen.find('w') != std::string::npos) ? WHITE : BLACK;

					// 後手番化したsfen。
					// hashkeyは、すべて後手番の局面で考えるから、hashkeyを求めるときに後手番の局面にしておく。
					string white_sfen = stm == WHITE ? sfen : Position::sfen_to_flipped_sfen(sfen);

					// hashkey_to_indexには後手番の局面のhash keyからのindexを登録する。
					StateInfo si;
					pos.set(white_sfen, &si);
					Key white_hash_key = pos.key();
					// 元の定跡ファイルにflipした局面は登録されていないものとする。
					// ⇨  登録されていたら、あとから出現した局面を優先する。

					auto book_node_index = BookNodeIndex(book_nodes.size()); // emplace_back()する前のsize()が今回追加されるindex
					hashkey_to_index[white_hash_key] = book_node_index;

					// この局面は王手されているのか？
					bool checked = pos.checkers();

					book_nodes.emplace_back(BookNode());
					auto& book_node = book_nodes.back();

					book_node.color        = stm; // 元の手番。これを維持してファイルに書き出さないと、sfen文字列でsortされていたのが狂う。
					book_node.checked      = checked;
					in_check_counter      += checked;

					// この直後にやってくる指し手をこの局面の指し手として取り込む。
					continue;
				}

				// 先に"sfen .."が出現しているはずなので、back()は有効なはず。
				// ("sfen"が出現せずに指し手が出現するような定跡ファイルはおかしいわけで…)
				auto& book_node = book_nodes.back();

				// この行に Move PonderMove value depthが書かれている。これをそのままparseする。

				Parser::LineScanner scanner(line);
				auto move_str   = scanner.get_text();
				auto ponder_str = scanner.get_text(); // 使わないがskipはしないといけない。
				auto value      = (s16)std::clamp((int)scanner.get_number(0), BOOK_VALUE_MIN, BOOK_VALUE_MAX);
				Move16 move16 = (move_str == "none" || move_str == "None" || move_str == "resign") ? Move16::none() : USIEngine::to_move16(move_str);
				//Move16 ponder = (ponder_str == "none" || ponder_str == "None" || ponder_str == "resign") ? Move16::none() : USI::to_move16(ponder_str);

				// 先手の局面として登録しないといけないので後手番であるなら指し手もflipする。
				// posは上でblack_sfenが設定されているので先手番になるようにflipされている。
				if (book_node.color == WHITE)
					move16 = flip_move(move16);

				// 入力側の指し手はleaf候補なので、入力depthを引き継がず0として後退解析する。
				book_node.moves.emplace_back(BookMove(move16, value, 0));
				// あとで合流のチェックをしてleaf nodeであるかを確認する。
			}
			if (!fast)
				sfen_writer.Close();

			return Tools::Result::Ok();
		}

		// 局面の合流チェック
		Tools::Result convergence_check(size_t thread_count)
		{
			cout << "Convergence Check   :" << endl;

			// テンポラリとして書き出していたsfenファイル。
			// これは、元の定跡ファイルに出現したsfen文字列がそのまま書き出されている。
			SystemIO::TextReader sfen_reader;
			if (!fast)
			{
				if (sfen_reader.Open(sfen_temp_path).is_not_ok())
				{
					sync_cout << "info string Error! : can't read file : " + sfen_temp_path << sync_endl;
					return Tools::ResultCode::FileOpenError;
				}
			}

			auto read_sfen = [&](size_t i, string& sfen) {
				if (fast) sfen = original_sfens[i];
				else if (sfen_reader.ReadLine(sfen).is_not_ok())
				{
					sync_cout << "info string Error! : can't read sfen temp file : " + sfen_temp_path << sync_endl;
					return false;
				}
				return true;
			};

			// 並列処理中はhash表を読取専用とし、各workerは担当する局面のmovesだけを更新する。
			const auto& lookup = hashkey_to_index;
			auto convert = [&](BookNodeIndex i, string sfen, Position& pos, u64& count) {
				auto& book_node = book_nodes[i];
				StateInfo si;

				// この局面が後手番なら、sfenを先手の局面化する。
				// 💡: BookNodeは先手の局面で考えている。hashkeyは後手の局面で考えている。
				if (book_node.color == WHITE)
					sfen = Position::sfen_to_flipped_sfen(sfen);

				pos.set(sfen, &si);
				ASSERT_LV3(pos.side_to_move() == BLACK);

				// 元ファイルの定跡DBに登録されていた指し手
				SmallVector<BookMove> book_moves;
				std::swap(book_node.moves, book_moves); // swapしていったんbook_node.movesはクリアしてしまう。

				// ここから全合法手で一手進めて既知の(定跡ツリー上の他の)局面に行くかを調べる。
				for (auto move : MoveList<LEGAL_ALL>(pos))
				{
					// moveで進めた局面が存在する時のhash値。
					Key next_hash = pos.key_after(move);

					auto it = lookup.find(next_hash);
					if (it != lookup.end())
					{
						// 定跡局面が存在した。

						BookNodeIndex next_book_node_index = it->second;
						Move16        move16               = move.to_move16();
						BookMove      book_move(move16, next_book_node_index);

						// これが定跡DBのこの局面の指し手に登録されていないなら、
						// これは(定跡DBにはなかった指し手で進めたら既知の局面に)合流したということだから
						// 合流カウンターをインクリメントしておく。
						if (std::find_if(book_moves.begin(), book_moves.end(), [&](const auto& bm) { return bm.move == move16; }) == book_moves.end())
							++count;

						book_node.moves.emplace_back(book_move);
					}
				}

				// 元ファイルの定跡DB上のこの局面の指し手も登録しておく。
				for (auto& book_move : book_moves)
				{
					Move16 move = book_move.move;
					if (move == Move16::none())
						continue;

					// これがbook_nodeにすでに登録されているか？
					// 登録されているということは合流する( = 子局面がある)ということだから、評価値は子局面のものを使うので
					// ここで評価値を反映させる必要はない。
					if (std::find_if(book_node.moves.begin(), book_node.moves.end(), [&](auto& book_move) { return book_move.move == move; }) == book_node.moves.end())
						// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので元の定跡ファイルの評価値は反映させなくて良い。)
						book_node.moves.emplace_back(book_move);
				}

			};

			// SFENだけを一定数ずつ読み、定跡本体は複製しない。通常モードでもメモリの増加を抑える。
			constexpr size_t BatchSize = 65536;
			const size_t worker_limit = std::max(size_t(1),
			    std::min({thread_count, book_nodes.size(), BatchSize}));
			cout << "peta_shock: convergence workers = " << worker_limit << endl;
			Tools::ProgressBar progress;
			progress.reset(book_nodes.empty() ? 0 : book_nodes.size() - 1);

			if (worker_limit == 1)
			{
				// 1スレッドなら従来どおり1局面ずつ読み、バッチ用バッファや追加スレッドを作らない。
				Position pos;
				for (size_t i = 0; i < book_nodes.size(); ++i)
				{
					string sfen;
					if (!read_sfen(i, sfen)) return Tools::ResultCode::FileReadError;
					convert(BookNodeIndex(i), std::move(sfen), pos, converged_moves);
					progress.check(i);
				}
			}
			else for (size_t first = 0; first < book_nodes.size(); first += BatchSize)
			{
				const size_t count = std::min(BatchSize, book_nodes.size() - first);
				// ファイル読込と進捗表示は呼出元だけで行い、worker間でreaderや表示状態を共有しない。
				vector<string> sfens(count);
				for (size_t k = 0; k < count; ++k)
					if (!read_sfen(first+k, sfens[k])) return Tools::ResultCode::FileReadError;
				const size_t workers = std::min(worker_limit, count);
				vector<u64> counts(workers, 0);
				auto work = [&](size_t worker) {
					// Position/StateInfoと集計値はworkerごとに分離し、共有カウンターへの書込み競合を避ける。
					Position pos;
					u64 local_count = 0;
					for (size_t k = count * worker / workers; k < count * (worker+1) / workers; ++k)
						convert(BookNodeIndex(first+k), std::move(sfens[k]), pos, local_count);
					counts[worker] = local_count;
				};
				vector<std::thread> pool;
				pool.reserve(workers - 1);
				for (size_t worker = 1; worker < workers; ++worker) pool.emplace_back(work, worker);
				// 呼出元も1workerとして数える。全workerの完了後にだけ集計し、次のバッチへ進む。
				work(0);
				for (auto& thread : pool) thread.join();
				for (auto count : counts) converged_moves += count;
				progress.check(first + count - 1);
			}
			if (!fast)
				sfen_reader.Close();

			//cout << "converged_moves : " << converged_moves << endl;
			return Tools::Result::Ok();
		}

		vector<BookNodeIndex> selected_moves; // 各局面で保証検査を通過した選択手の番号。
		vector<u8> keep_equal_depth_ties; // 同評価値・同depthの他の手も、安全に選択可能なら1。

		// 値の確定、先後で共通の保証手の構成、depth優先、保存できる同点候補の検証を順に行う。
		bool solve_repetition_game()
		{
			using namespace Book::PetaRetrograde;
			if (book_nodes.size() >= size_t(None) / 2)
			{
				cout << "Error! : too many book positions for repetition analysis." << endl;
				return false;
			}
			cout << "Retrograde Analysis : solve checking cores and propagate guarantees." << endl;
			PetaGraph graph(book_nodes);
			Solver<PetaGraph> solver(graph);
			if (!solver.solve())
			{
				cout << "Error! : peta_shock: " << solver.error << endl;
				return false;
			}
			// 正規化した定跡では先後2状態に別々の手を保存できないので、同じ選択手へ揃えて再検証する。
			for (Index v = 0; v < graph.size(); v += 2)
			{
				if (solver.values[v] != solver.values[v+1])
				{
					cout << "Error! : peta_shock: asymmetric canonical values." << endl;
					return false;
				}
				solver.policy[v+1] = solver.policy[v];
			}
			if (!solver.certify())
			{
				cout << "Error! : peta_shock canonical policy: " << solver.error << endl;
				return false;
			}

			// 同評価値なら、非負では短い出口、負では長い出口を優先する。
			// ただし、自分の連続王手による反則負けの回避はdepthの選好より優先する。
			if (!solver.prefer_signed_depth(true))
			{
				cout << "Error! : peta_shock signed-depth policy: " << solver.error << endl;
				return false;
			}

			// 検証済みの選択手だけを辿って、各局面からleafまでの距離を求める。
			// 循環やそこへ至る経路はdepthを9999へ飽和させる。評価値を詰み値へ置き換えてはいけない。
			selected_moves.resize(book_nodes.size());
			vector<u16> depth(book_nodes.size(), BOOK_DEPTH_MAX);
			vector<u8> state(book_nodes.size(), 0); // 0:未訪問、1:今回の経路上、2:距離確定済み。
			vector<Index> path;
			for (Index v = 0; v < book_nodes.size(); ++v) selected_moves[v] = solver.policy[2*v];
			for (Index root = 0; root < book_nodes.size(); ++root)
			{
				if (state[root]) continue;
				auto v = root;
				path.clear();
				while (!state[v])
				{
					state[v] = 1;
					path.push_back(v);
					const auto& move = book_nodes[v].moves[selected_moves[v]];
					if (move.leaf) break;
					v = move.next;
				}
				while (!path.empty())
				{
					v = path.back(); path.pop_back();
					const auto& move = book_nodes[v].moves[selected_moves[v]];
					depth[v] = move.leaf ? 0 : u16(std::min(int(depth[move.next]) + 1, int(BOOK_DEPTH_MAX)));
					state[v] = 2;
				}
			}
			// 同評価値・同depthでも、手を切り替えることで危険な循環ができることがある。
			// 全候補を任意に切り替えても双方の保証が成立する局面だけ、複数のbest候補を残す。
			keep_equal_depth_ties.assign(book_nodes.size(), 1);
			auto permits_tie = [&](Index v, Index j) {
				if (!keep_equal_depth_ties[v/2]) return j == solver.policy[v];
				auto w = graph.child(v,j);
				int d = w == None ? 0 : std::min(int(depth[w/2]) + 1, int(BOOK_DEPTH_MAX));
				return solver.edge_value(v,j) == solver.values[v] && d == depth[v/2];
			};
			vector<u8> restricted(graph.size(), 0);
			solver.certify_with(permits_tie, &restricted);
			// 一方の手番で危険なら、同じ正規化局面のもう一方も単一の保証手へ揃える。
			for (Index v = 0; v < book_nodes.size(); ++v)
				if (restricted[2*v] || restricted[2*v+1]) keep_equal_depth_ties[v] = 0;
			if (!solver.certify_with(permits_tie))
			{
				cout << "Error! : peta_shock alternatives: " << solver.error << endl;
				return false;
			}
			cout << "peta_shock: equal-depth alternatives certified; restricted positions = "
				 << std::count(keep_equal_depth_ties.begin(), keep_equal_depth_ties.end(), u8(0)) << endl;
			for (Index v = 0; v < book_nodes.size(); ++v)
			{
				// BookNode::vdは親から使う形式へ変換する。Solver内部の値に出力用の-1補正は加えない。
				book_nodes[v].vd = ValueDepth(s16(-solver.values[2*v]), u16(std::min(int(depth[v]) + 1, int(BOOK_DEPTH_MAX))));
				finite_depth_nodes += depth[v] < BOOK_DEPTH_MAX;
			}
			cout << "peta_shock: checking core states = " << solver.core_size(0) << " / " << solver.core_size(1)
				 << ", propagated edges = " << solver.propagation_edges
				 << ", local calls = " << solver.local_calls << ", local edges = " << solver.local_edges << endl;
			cout << "peta_shock: both physical-side policies certified." << endl;
			return true;
		}

		// 保存用のコピーを作り、元のグラフや伝播に使った評価値は変更しない。
		SmallVector<BookMove> make_output_moves(BookNode& node)
		{
			const size_t index = size_t(&node - book_nodes.data());
			const auto selected = selected_moves[index];
			const int best = -node.vd.value;
			const auto& chosen = node.moves[selected];
			const auto chosen_depth = chosen.leaf ? 0 : book_nodes[chosen.next].vd.depth;
			SmallVector<BookMove> result;
			for (size_t j = 0; j < node.moves.size(); ++j)
			{
				const auto& move = node.moves[j];
				auto vd = move.leaf ? ValueDepth(move.vd.value, 0) : book_nodes[move.next].vd;
				if (shrink && vd.value != best) continue;
				// depth優先と保証検査で選んだ手がbestに残るよう、優先しない同点手だけ保存値を1下げる。
				// 同depthでも安全性を確認できない他の手は1下げ、検証済みの選択手を残す。
				if (j != selected && vd.value == best && (!keep_equal_depth_ties[index] || vd.depth != chosen_depth)) --vd.value;
				result.emplace_back(BookMove(move.move, vd));
			}
			std::sort(result.begin(), result.end(), [](const BookMove& a, const BookMove& b) {
				if (a.vd.value != b.vd.value) return a.vd.value > b.vd.value;
				return a.vd.value >= 0 ? a.vd.depth < b.vd.depth : a.vd.depth > b.vd.depth;
			});
			return result;
		}

		// ペタショック化した定跡ファイルを書き出す。
		//	shrink : bestvalueの指し手のみを書き出す。
		Tools::Result write_peta_shock_book(std::string writebook_path, std::vector<BookNode>& book_nodes)
		{
			// 通常のpeta_shockコマンド時の処理。(peta_shock_nextコマンドではなく)

			// メモリ上の定跡DBを再構成。
			// この時点でもうhash_key_to_index不要なので解放する。
			// (clear()では解放されないので、swap trickを用いる。)
			HashKey2Index().swap(this->hashkey_to_index);

			if (is_ybb_book(writebook_path))
				return write_peta_shock_ybb_book(writebook_path, book_nodes);

			return write_peta_shock_db_book(writebook_path, book_nodes);
		}

		Tools::Result write_peta_shock_db_book(std::string writebook_path, std::vector<BookNode>& book_nodes)
		{
			// progress表示用
			Tools::ProgressBar progress;

			// MemoryBookを用いるとオーバーヘッドが大きいので自前で直接ファイルに書き出す。

			cout << "Write to a book DB  : " << endl;

			SystemIO::TextWriter writer;
			if (writer.Open(writebook_path).is_not_ok())
			{
				cout << "Error! : open file error , path = " << writebook_path << endl;
				return Tools::ResultCode::FileOpenError;
			}

			progress.reset(book_nodes.size() == 0 ? 0 : book_nodes.size() - 1);

			// バージョン識別用文字列
			writer.WriteLine(YaneuraOu::Book::BookDBHeader2016_100);
			writer.WriteLine("# NOE:" + to_string(book_nodes.size()));

			SystemIO::TextReader sfen_reader;
			if (!fast)
			{
				if (sfen_reader.Open(sfen_temp_path).is_not_ok())
				{
					sync_cout << "info string Error! : can't read file : " + sfen_temp_path << sync_endl;
					return Tools::ResultCode::FileOpenError;
				}
			}

			for(BookNodeIndex i = 0 ; i < BookNodeIndex(book_nodes.size()) ; ++i)
			{
				auto& book_node = book_nodes[i];
				string sfen;
				if (fast)
					sfen = original_sfens[i];
				else
				{
					if (sfen_reader.ReadLine(sfen).is_not_ok())
					{
						sync_cout << "info string Error! : can't read sfen temp file : " + sfen_temp_path << sync_endl;
						return Tools::ResultCode::FileReadError;
					}
				}

				if (writer.WriteLine("sfen " + sfen).is_not_ok())
					return Tools::ResultCode::FileWriteError;
				// TextWriterは固定4KBバッファが満杯になると自動で書く。局面ごとのFlushは不要。

				const auto moves = make_output_moves(book_node);
				for(const auto& move : moves)
				{
					// 元のDB上で後手の局面なら後手の局面として書き出したいので、
					// 後手の局面であるなら指し手を反転させる。
					Move16 m16 = (book_node.color == WHITE) ? flip_move(move.move) : move.move;
					if (writer.WriteLine(to_usi_string(m16) + " none " + to_string(move.vd.value) + " " + to_string(move.vd.depth)).is_not_ok())
						return Tools::ResultCode::FileWriteError;
				}

				progress.check(i);
			}
			if (!fast)
				sfen_reader.Close();

			cout << "write " + writebook_path << endl;
			return Tools::Result::Ok();
		}

		Tools::Result write_peta_shock_ybb_book(std::string writebook_path, std::vector<BookNode>& book_nodes)
		{
			cout << "Write to a book DB  : " << endl;

			ifstream index_reader(readbook_path, ios::binary);
			if (!index_reader)
			{
				sync_cout << "info string Error! : can't read file : " + readbook_path << sync_endl;
				return Tools::ResultCode::FileNotFound;
			}

			u64 input_record_count = 0;
			u64 input_flags = 0;
			if (!read_ybb_header(index_reader, input_record_count, input_flags))
			{
				sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
				return Tools::ResultCode::FileReadError;
			}

			u64 input_moves_base = 0;
			if (!ybb_index_size(input_record_count, input_moves_base) || input_record_count != u64(book_nodes.size()))
			{
				sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
				return Tools::ResultCode::FileReadError;
			}

			u64 index_size = 0;
			if (!ybb_index_size(u64(book_nodes.size()), index_size))
				return Tools::ResultCode::FileWriteError;

			ofstream writer(writebook_path, ios::binary | ios::out | ios::trunc);
			if (!writer)
			{
				cout << "Error! : open file error , path = " << writebook_path << endl;
				return Tools::ResultCode::FileOpenError;
			}

			constexpr u64 output_flags = YBB_FLAG_MOVE_DEPTH;
			if (!write_ybb_header(writer, u64(book_nodes.size()), output_flags))
				return Tools::ResultCode::FileWriteError;

			u64 index_offset = YBB_HEADER_SIZE;
			u64 move_offset = 0;
			// YBBはヘッダー・全局面の索引・指し手領域の順。各局面の索引と指し手を所定の位置へ書く。

			Tools::ProgressBar progress;
			progress.reset(book_nodes.size() == 0 ? 0 : book_nodes.size() - 1);

			for(BookNodeIndex i = 0 ; i < BookNodeIndex(book_nodes.size()) ; ++i)
			{
				YbbIndexEntry entry;
				if (!read_ybb_index_entry(index_reader, entry))
				{
					sync_cout << "info string Error! : invalid ybb file : " << readbook_path << sync_endl;
					return Tools::ResultCode::FileReadError;
				}

				auto& book_node = book_nodes[i];
				const auto moves = make_output_moves(book_node);
				if (moves.size() > numeric_limits<u16>::max())
				{
					sync_cout << "info string Error! : too many moves in ybb record : " << writebook_path << sync_endl;
					return Tools::ResultCode::FileWriteError;
				}

				entry.moves_offset = move_offset;
				entry.move_count = u16(moves.size());

				writer.seekp(std::streamoff(index_offset), ios::beg);
				if (!writer || !write_ybb_index_entry(writer, entry))
					return Tools::ResultCode::FileWriteError;

				writer.seekp(std::streamoff(index_size + move_offset), ios::beg);
				if (!writer)
					return Tools::ResultCode::FileWriteError;

				for(const auto& move : moves)
				{
					Move16 m16 = (book_node.color == WHITE) ? flip_move(move.move) : move.move;
					if (!write_u16_le(writer, m16.raw())
						|| !write_u16_le(writer, u16(move.vd.value))
						|| !write_u16_le(writer, move.vd.depth))
						return Tools::ResultCode::FileWriteError;
				}

				index_offset += YBB_INDEX_RECORD_SIZE;
				move_offset += u64(moves.size()) * 6;
				progress.check(i);
			}

			cout << "write " + writebook_path << endl;
			return Tools::Result::Ok();
		}

		// 結果出力(統計値など)
		void output_result()
		{
			cout << "[ PetaShock Result ]" << endl;

			// 書き出したrecord数
			cout << "write book nodes   : " << book_nodes.size() << endl;

			cout << "finite-depth nodes : " << finite_depth_nodes << endl;
			cout << "saturated nodes    : " << book_nodes.size() - finite_depth_nodes << endl;

			// 王手されていた局面の数
			cout << "in-check nodes     : " << in_check_counter << endl;

			cout << "converged_moves    : " << converged_moves << endl;

			cout << endl << "Making a peta-shock book has been completed." << endl;
		}

	private:

		// -- 定跡データ

		// 定跡本体
		vector<BookNode> book_nodes;

		// 局面のHASH_KEYからBookMoveIndexへのmapper
		// ただし、flipして後手番にしたhashkeyを登録してある。
		// ⇨　後手の局面はflipして先手の局面として格納している。ゆえに、格納されているのはすべて先手の局面であり、
		// 　そこから1手進めると後手の局面となる。この時に、hash keyから既存の局面かどうかを調べたいので…。
		using HashKey2Index = unordered_map<Key, BookNodeIndex>;
		HashKey2Index hashkey_to_index;

		// fast == trueのときは、テンポラリファイルではなくここに元の定跡ファイル上のSFEN文字列を溜めておく。
		vector<string> original_sfens;


		// -- path

		// 読み込む定跡ファイルのpath(ペタショック化前の定跡ファイル)
		string readbook_path;

		// 書き出す定跡ファイルのpath(ペタショック化後の定跡ファイル)
		string writebook_path;

		// sfenの一時ファイルを書き出すpath
		string sfen_temp_path;

		// -- 統計値

		// 合流した指し手の数
		u64 converged_moves;

		// 選択手を辿ったleafまでの距離が、飽和値9999未満になった局面数。
		u64 finite_depth_nodes;

		// 王手していた局面の数
		u64 in_check_counter;



		// -- option設定

		// その局面の最善手しか書き出さない。
		bool shrink;

		// テンポラリファイルを書き出さない。
		bool fast;

		// 合流チェックに使うスレッド数。エンジンのThreadsとは独立。
		size_t thread_count;

	};

} // namespace MakeBook2025

namespace Book
{
	// 2025年以降に作ったmakebook拡張コマンド。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2025(std::istringstream& is, const std::string& token, const OptionsMap& options)
    {
		if (token == "peta_shock") {

			// ペタショックコマンド
			// 
			// やねうら王の定跡ファイルに対して定跡ツリー上でmin-max探索を行い、その結果を別の定跡ファイルに書き出す。
			//   makebook peta_shock book.db user_book1.db
			// 　⇨　先手か後手か、片側の局面しか書き出さない。エンジンオプションの FlippedBook を必ずオンにして用いること。
			//   オプション指定
			//		shrink : 最善手しか書き出さない
			//      fast   : テンポラリファイルを書き出さない。(メモリ上に格納するのでその分だけメモリを消費する。)
			//      threads N : 通常モード・fastとも合流チェックをNスレッドで行う。省略時は1。
			//                  エンジンオプションのThreadsは参照しない。
			MakeBook2025::PetaShock ps;
			auto book_dir = options["BookDir"];
			ps.make_book(is, book_dir);
			return 1;
		}

		return 0;
	}

} // namespace Book
} // namespace YaneuraOu

#endif // defined (ENABLE_MAKEBOOK_CMD)
